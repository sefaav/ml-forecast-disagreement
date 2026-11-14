"""Fama-MacBeth inference with the frozen Bartlett HAC standard error for one monthly coefficient series.

Pre-specified in docs/pre_analysis_plan.md, "Estimation protocol" ("Fama-MacBeth
inference"). For a series beta_tau on the calendar grid tau = 1..N of OOS
months, with a_tau = 1 for estimable months and T = sum a_tau,

    mean    = T^-1 sum a_tau beta_tau,
    x_tau   = a_tau (beta_tau - mean), zero for non-estimable months,
    Gamma_l = T^-1 sum_{tau = l+1}^{N} x_tau x_{tau-l},
    Omega   = Gamma_0 + 2 sum_{l=1}^{L} (1 - l / (L + 1)) Gamma_l,
    SE      = sqrt(Omega / T),

Bartlett kernel, normalisation 1/T, no finite-sample correction. Non-estimable
months keep their calendar position: the series is never compressed before the
lagged products are formed, and the lag counts calendar months. L = 6 is the
primary lag; L = 0 (the classical Fama-MacBeth standard error with the
population variance) and L = 12 are reported as sensitivity analyses.
Inference uses the standard normal distribution: t = mean / SE,
p = 2 (1 - Phi(|t|)) and the 95% interval mean +/- 1.96 SE.

The component is purely numerical and scalar: one series in, one result out.
Vector Wald statistics, marginal slopes and decision rules are not here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

HAC_LAG = 6  # the primary lag of "Fama-MacBeth inference"
SENSITIVITY_LAGS = (0, 12)
CRITICAL_95 = 1.96  # the frozen normal critical value of the 95% interval
MIN_MONTHS = 2  # below, no dispersion can be measured


@dataclass(frozen=True)
class FamaMacBethResult:
    """The frozen Fama-MacBeth estimate and HAC inference of one monthly coefficient series."""

    mean: float
    se: float
    t: float
    p_value: float
    ci_low: float
    ci_high: float
    n_months: int  # T, the estimable months
    n_grid: int  # N, the calendar grid length
    lag: int  # L
    hac_variance: float  # Omega


def _validated(series: np.ndarray, estimable: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    series, estimable = np.asarray(series), np.asarray(estimable)
    if series.ndim != 1 or estimable.ndim != 1 or series.shape != estimable.shape:
        raise ValueError("the series and the estimable indicator must be one-dimensional arrays of the same length")
    if estimable.dtype != np.bool_:
        raise ValueError("the estimable indicator must be strictly Boolean")
    if isinstance(lag, bool) or not isinstance(lag, (int, np.integer)) or lag < 0:
        raise ValueError("the lag must be a non-negative integer number of calendar months")
    values = series.astype("float64")
    if not np.isfinite(values[estimable]).all():
        raise ValueError("every estimable month must have a finite coefficient")
    if int(estimable.sum()) < MIN_MONTHS:
        raise ValueError(f"at least {MIN_MONTHS} estimable months are needed, {int(estimable.sum())} given")
    return values, estimable


def hac_variance(x: np.ndarray, lag: int, n_months: int) -> float:
    """Omega of the module docstring from the zero-filled centred series x on the calendar grid."""
    omega = float(x @ x) / n_months
    for lead in range(1, lag + 1):
        if lead >= len(x):
            break  # no calendar pair at this lag
        omega += 2.0 * (1.0 - lead / (lag + 1.0)) * float(x[lead:] @ x[:-lead]) / n_months
    return omega


def fama_macbeth(series: np.ndarray, estimable: np.ndarray, lag: int = HAC_LAG) -> FamaMacBethResult:
    """Mean over estimable months, Bartlett HAC standard error with the frozen normalisation, normal inference.

    `series` holds one coefficient per calendar month of the grid; its values at non-estimable months are ignored.
    A zero HAC variance (a series constant over the estimable months) has no defined t statistic and is rejected.
    """
    values, estimable = _validated(series, estimable, lag)
    n_months = int(estimable.sum())
    mean = float(values[estimable].sum()) / n_months
    x = np.where(estimable, values - mean, 0.0)
    omega = hac_variance(x, lag, n_months)
    if not omega > 0.0:
        raise ValueError("the HAC variance is zero: the coefficient series is constant over the estimable months")
    se = math.sqrt(omega / n_months)
    t = mean / se
    return FamaMacBethResult(
        mean=mean,
        se=se,
        t=t,
        p_value=math.erfc(abs(t) / math.sqrt(2.0)),  # 2 (1 - Phi(|t|))
        ci_low=mean - CRITICAL_95 * se,
        ci_high=mean + CRITICAL_95 * se,
        n_months=n_months,
        n_grid=int(values.size),
        lag=int(lag),
        hac_variance=omega,
    )
