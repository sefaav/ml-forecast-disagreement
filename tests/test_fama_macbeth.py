"""Fama-MacBeth mean and frozen Bartlett HAC inference on deterministic synthetic series. No real data is used."""

import math

import numpy as np
import pytest
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.sandwich_covariance import cov_hac_simple

from disagreement.inference.fama_macbeth import (
    CRITICAL_95,
    HAC_LAG,
    SENSITIVITY_LAGS,
    FamaMacBethResult,
    fama_macbeth,
    hac_variance,
)


def _explicit_pairs(series, estimable, lag):
    """Independent reference: mean over estimable months, lag products summed over observed calendar pairs."""
    idx = [i for i in range(len(series)) if estimable[i]]
    T = len(idx)
    mean = sum(series[i] for i in idx) / T
    x = {i: series[i] - mean for i in idx}
    omega = 0.0
    for lead in range(0, lag + 1):
        gamma = sum(x[i] * x[i - lead] for i in idx if (i - lead) in x) / T
        omega += gamma if lead == 0 else 2.0 * (1.0 - lead / (lag + 1.0)) * gamma
    return mean, omega, math.sqrt(omega / T)


def test_hand_calculated_mean_hac_variance_and_standard_error():
    # grid of 6 calendar months, the fourth not estimable: beta = 1, 3, 2, (ignored), 5, 4
    series = np.array([1.0, 3.0, 2.0, np.nan, 5.0, 4.0])
    estimable = np.array([True, True, True, False, True, True])
    r = fama_macbeth(series, estimable, lag=1)
    # T = 5, mean = 15 / 5 = 3, x = (-2, 0, -1, 0, 2, 1)
    assert r.n_months == 5 and r.n_grid == 6 and r.lag == 1 and r.mean == 3.0
    gamma0 = (4 + 0 + 1 + 0 + 4 + 1) / 5  # = 2
    gamma1 = ((0 * -2) + (-1 * 0) + (0 * -1) + (2 * 0) + (1 * 2)) / 5  # the pair across the gap is zero: = 0.4
    omega = gamma0 + 2 * (1 - 1 / 2) * gamma1  # = 2.4
    assert r.hac_variance == pytest.approx(omega) and omega == pytest.approx(2.4)
    assert r.se == pytest.approx(math.sqrt(2.4 / 5))
    assert r.t == pytest.approx(3.0 / math.sqrt(2.4 / 5))
    assert r.ci_low == pytest.approx(3.0 - CRITICAL_95 * r.se) and r.ci_high == pytest.approx(3.0 + CRITICAL_95 * r.se)
    assert isinstance(r, FamaMacBethResult) and HAC_LAG == 6 and SENSITIVITY_LAGS == (0, 12)


def test_complete_series_matches_statsmodels_cov_hac_simple_without_correction():
    rng = np.random.default_rng(20261114)
    n = 480
    e = rng.standard_normal(n)
    series = 0.002 + 0.01 * (e + 0.5 * np.concatenate([[0.0], e[:-1]]))  # autocorrelated
    estimable = np.ones(n, dtype=bool)
    fit = sm.OLS(series, np.ones((n, 1))).fit()
    for lag in (0, HAC_LAG, 12):
        reference = float(cov_hac_simple(fit, nlags=lag, use_correction=False)[0, 0])
        r = fama_macbeth(series, estimable, lag=lag)
        assert r.mean == pytest.approx(float(fit.params[0]), rel=1e-12)
        assert r.se ** 2 == pytest.approx(reference, rel=1e-10)
        assert r.hac_variance / n == pytest.approx(reference, rel=1e-10)


def test_calendar_gaps_match_the_explicit_lag_pair_summation_and_are_never_compressed():
    rng = np.random.default_rng(3)
    n = 120
    series = 0.01 * rng.standard_normal(n)
    estimable = rng.random(n) > 0.2
    estimable[[0, 1, 2, 50, 51, 52, 53, 54, 55, 56, 57, 58]] = False  # a 9-month hole
    for lag in (0, 1, HAC_LAG, 12):
        mean, omega, se = _explicit_pairs(series, estimable, lag)
        r = fama_macbeth(series, estimable, lag=lag)
        assert r.mean == pytest.approx(mean, rel=1e-12)
        assert r.hac_variance == pytest.approx(omega, rel=1e-12)
        assert r.se == pytest.approx(se, rel=1e-12)
    # compressing the series (dropping the gaps) is a different, non-frozen estimator for L > 0 ...
    compressed = fama_macbeth(series[estimable], np.ones(int(estimable.sum()), dtype=bool), lag=HAC_LAG)
    gapped = fama_macbeth(series, estimable, lag=HAC_LAG)
    assert compressed.mean == gapped.mean and compressed.n_months == gapped.n_months
    assert compressed.hac_variance != pytest.approx(gapped.hac_variance, rel=1e-6)
    # ... but identical for L = 0, where no lagged pair enters
    assert fama_macbeth(series[estimable], np.ones(int(estimable.sum()), dtype=bool), lag=0).se == pytest.approx(
        fama_macbeth(series, estimable, lag=0).se, rel=1e-12
    )
    # values at non-estimable months never matter, even when they are absurd
    other = series.copy()
    other[~estimable] = 1e6
    assert fama_macbeth(other, estimable).se == gapped.se and fama_macbeth(other, estimable).mean == gapped.mean
    other[~estimable] = np.nan
    assert fama_macbeth(other, estimable).hac_variance == gapped.hac_variance


def test_lag_zero_is_the_population_variance_fama_macbeth_standard_error():
    rng = np.random.default_rng(4)
    n = 200
    series = rng.standard_normal(n)
    complete = np.ones(n, dtype=bool)
    r0 = fama_macbeth(series, complete, lag=0)
    assert r0.se == pytest.approx(series.std(ddof=0) / math.sqrt(n), rel=1e-12)  # population SD, not ddof = 1
    assert r0.se != pytest.approx(series.std(ddof=1) / math.sqrt(n), rel=1e-6)
    gaps = rng.random(n) > 0.3
    rg = fama_macbeth(series, gaps, lag=0)
    assert rg.se == pytest.approx(series[gaps].std(ddof=0) / math.sqrt(gaps.sum()), rel=1e-12)
    # L = 6 and L = 12 weigh the same autocovariances with Bartlett weights 1 - l / (L + 1)
    x = np.where(gaps, series - series[gaps].mean(), 0.0)
    T = int(gaps.sum())
    gammas = [float(x[l:] @ x[: n - l]) / T if l else float(x @ x) / T for l in range(13)]
    for lag in (HAC_LAG, 12):
        expected = gammas[0] + 2 * sum((1 - l / (lag + 1)) * gammas[l] for l in range(1, lag + 1))
        assert fama_macbeth(series, gaps, lag=lag).hac_variance == pytest.approx(expected, rel=1e-12)
        assert hac_variance(x, lag, T) == pytest.approx(expected, rel=1e-12)
    # a lag longer than the grid simply has no pairs beyond the grid
    short = fama_macbeth(np.array([1.0, 2.0, 4.0]), np.ones(3, dtype=bool), lag=12)
    assert short.hac_variance == pytest.approx(_explicit_pairs([1.0, 2.0, 4.0], [True] * 3, 12)[1])


def test_two_sided_normal_p_values_and_confidence_intervals():
    rng = np.random.default_rng(5)
    n = 300
    estimable = np.ones(n, dtype=bool)
    for shift in (0.0, 0.001, 0.005, -0.02):
        series = shift + 0.02 * rng.standard_normal(n)
        r = fama_macbeth(series, estimable)
        assert r.t == pytest.approx(r.mean / r.se)
        assert r.p_value == pytest.approx(2.0 * norm.sf(abs(r.t)), rel=1e-10, abs=1e-300)
        assert r.ci_low == pytest.approx(r.mean - 1.96 * r.se) and r.ci_high == pytest.approx(r.mean + 1.96 * r.se)
        assert (r.p_value < 0.05) == (abs(r.t) > norm.ppf(0.975)) == (not r.ci_low <= 0.0 <= r.ci_high) or (
            abs(abs(r.t) - 1.96) < 1e-3  # 1.96 against 1.959964: only an exact boundary case could differ
        )
    # the sign of t follows the mean, p is symmetric and in [0, 1]
    up = fama_macbeth(np.array([1.0, 2.0, 3.0, 4.0]), np.ones(4, dtype=bool), lag=0)
    down = fama_macbeth(-np.array([1.0, 2.0, 3.0, 4.0]), np.ones(4, dtype=bool), lag=0)
    assert up.t > 0 > down.t and up.p_value == down.p_value and 0.0 < up.p_value < 1.0
    assert up.t == pytest.approx(2.5 / (math.sqrt(1.25) / 2.0))  # mean 2.5, population SD sqrt(1.25), T = 4
    exact = fama_macbeth(np.array([0.0, 1.0, -1.0, 0.0]), np.ones(4, dtype=bool), lag=0)
    assert exact.t == 0.0 and exact.p_value == 1.0  # a zero mean


def test_invalid_inputs_insufficient_information_and_degenerate_variance_are_explicit():
    series = np.array([0.1, 0.2, 0.3, 0.4])
    ok = np.ones(4, dtype=bool)

    def rejected(message: str, s: np.ndarray = series, e: np.ndarray = ok, lag: object = HAC_LAG):
        with pytest.raises(ValueError, match=message):
            fama_macbeth(s, e, lag=lag)  # type: ignore[arg-type]  # invalid lag types are rejected at run time

    rejected("same length", e=ok[:-1])
    rejected("one-dimensional", s=series.reshape(2, 2), e=ok.reshape(2, 2))
    rejected("strictly Boolean", e=ok.astype(int))  # 0/1 integers are not accepted
    rejected("strictly Boolean", e=ok.astype(object))
    rejected("finite coefficient", s=np.array([0.1, np.nan, 0.3, 0.4]))  # a NaN at an estimable month
    rejected("finite coefficient", s=np.array([0.1, np.inf, 0.3, 0.4]))
    rejected("non-negative integer", lag=-1)
    rejected("non-negative integer", lag=6.0)
    rejected("non-negative integer", lag=True)
    rejected("at least 2 estimable months", e=np.array([True, False, False, False]))
    rejected("at least 2 estimable months", e=np.zeros(4, dtype=bool))
    rejected("HAC variance is zero", s=np.array([0.3, 0.3, 0.3, 0.3]))  # constant: no defined t statistic
    rejected("HAC variance is zero", s=np.array([0.3, 9.0, 0.3, 0.3]), e=np.array([True, False, True, True]))
    with pytest.raises(ValueError, match="at least 2"):
        fama_macbeth(np.array([]), np.array([], dtype=bool))
    # two estimable months are enough for a (crude) estimate
    r = fama_macbeth(np.array([1.0, np.nan, 3.0]), np.array([True, False, True]), lag=0)
    assert r.n_months == 2 and r.mean == 2.0 and r.se == pytest.approx(1.0 / math.sqrt(2))
