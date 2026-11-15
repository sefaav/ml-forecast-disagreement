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
    if np.ptp(values[estimable]) == 0.0:  # exactly constant: rounding of the mean could leave a spurious variance
        raise ValueError("the HAC variance is zero: the coefficient series is constant over the estimable months")
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


# ----------------------------------------------------------------------------------------------------------------
# Vector series: the joint Wald test of the two H3 interactions

WALD_DIMENSION = 2  # beta_3 (S^c x CD^B) and beta_CW (S^c x CW) of the same monthly H3 regression


@dataclass(frozen=True)
class WaldResult:
    """The frozen joint Wald test of a two-coefficient monthly series with the Bartlett HAC covariance."""

    mean: np.ndarray  # the two Fama-MacBeth means
    covariance: np.ndarray  # Omega, 2 x 2
    wald: float  # T mean' Omega^-1 mean
    df: int
    p_value: float  # chi-square survival probability with df degrees of freedom
    n_months: int  # T
    n_grid: int  # N
    lag: int  # L


def _validated_pair(series: np.ndarray, estimable: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    series, estimable = np.asarray(series), np.asarray(estimable)
    if series.ndim != 2 or series.shape[1] != WALD_DIMENSION or estimable.shape != (series.shape[0],):
        raise ValueError(f"the series must be an N x {WALD_DIMENSION} array with an N-vector estimable indicator")
    if estimable.dtype != np.bool_:
        raise ValueError("the estimable indicator must be strictly Boolean")
    if isinstance(lag, bool) or not isinstance(lag, (int, np.integer)) or lag < 0:
        raise ValueError("the lag must be a non-negative integer number of calendar months")
    values = series.astype("float64")
    if not np.isfinite(values[estimable]).all():
        raise ValueError("every estimable month must have finite coefficients")
    if int(estimable.sum()) < MIN_MONTHS:
        raise ValueError(f"at least {MIN_MONTHS} estimable months are needed, {int(estimable.sum())} given")
    return values, estimable


def hac_covariance(x: np.ndarray, lag: int, n_months: int) -> np.ndarray:
    """Omega = Gamma_0 + sum_l w_l (Gamma_l + Gamma_l') from the zero-filled centred N x k series x on the grid.

    Gamma_l = T^-1 sum_{tau = l+1}^{N} x_tau x_{tau-l}' over calendar pairs; w_l = 1 - l / (L + 1).
    """
    omega = x.T @ x / n_months
    for lead in range(1, lag + 1):
        if lead >= len(x):
            break  # no calendar pair at this lag
        gamma = x[lead:].T @ x[:-lead] / n_months
        omega = omega + (1.0 - lead / (lag + 1.0)) * (gamma + gamma.T)
    return omega


def fama_macbeth_wald(series: np.ndarray, estimable: np.ndarray, lag: int = HAC_LAG) -> WaldResult:
    """Joint test that both means of a two-coefficient monthly series are zero, with the frozen HAC covariance.

    `series` holds the two coefficients of the same monthly regression per calendar month; the rows of
    non-estimable months are ignored. The covariance uses the same calendar-gap treatment and normalisation as
    the scalar `fama_macbeth`: its diagonal equals the scalar HAC variances. W = T mean' Omega^-1 mean is
    referred to chi-square with 2 degrees of freedom, whose survival function is exp(-W / 2). A covariance that
    is not finite and positive definite (collinear or constant coefficients) is rejected, never regularised.
    """
    values, estimable = _validated_pair(series, estimable, lag)
    n_months = int(estimable.sum())
    mean = values[estimable].sum(axis=0) / n_months
    x = np.where(estimable[:, None], values - mean, 0.0)
    omega = hac_covariance(x, lag, n_months)
    # positive definite at machine precision: a theoretically singular Omega (collinear coefficients) can show a
    # tiny positive eigenvalue through rounding, so the numerical rank (SVD, NumPy's default machine-precision
    # tolerance relative to the largest singular value) is required to be full as well
    if not (np.isfinite(omega).all() and (np.linalg.eigvalsh(omega) > 0.0).all()
            and np.linalg.matrix_rank(omega) == WALD_DIMENSION):
        raise ValueError("the HAC covariance is singular: the two coefficient series are collinear or constant "
                         "over the estimable months")
    try:
        solved = np.linalg.solve(omega, mean)
    except np.linalg.LinAlgError as exc:  # an exactly singular matrix that passed the eigenvalue check
        raise ValueError("the HAC covariance is singular") from exc
    wald = float(n_months * (mean @ solved))
    return WaldResult(
        mean=mean,
        covariance=omega,
        wald=wald,
        df=WALD_DIMENSION,
        p_value=math.exp(-wald / 2.0),  # chi-square(2) survival function
        n_months=n_months,
        n_grid=int(values.shape[0]),
        lag=int(lag),
    )
