"""Fama-MacBeth mean and frozen Bartlett HAC inference on deterministic synthetic series. No real data is used."""

import math

import numpy as np
import pytest
import statsmodels.api as sm
from scipy.stats import chi2, norm
from statsmodels.stats.sandwich_covariance import cov_hac_simple

from disagreement.inference.fama_macbeth import (
    CRITICAL_95,
    HAC_LAG,
    SENSITIVITY_LAGS,
    WALD_DIMENSION,
    FamaMacBethResult,
    WaldResult,
    fama_macbeth,
    fama_macbeth_wald,
    hac_covariance,
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


# ----------------------------------------------------------------------------------------------------------------
# Joint Wald test of two coefficients


def _explicit_pair_covariance(series, estimable, lag):
    """Independent reference: 2 x 2 Omega from explicit loops over observed calendar pairs, Wald and p-value."""
    idx = [i for i in range(len(series)) if estimable[i]]
    T = len(idx)
    mean = [sum(series[i][j] for i in idx) / T for j in range(2)]
    x = {i: [series[i][0] - mean[0], series[i][1] - mean[1]] for i in idx}
    omega = [[0.0, 0.0], [0.0, 0.0]]
    for lead in range(0, lag + 1):
        gamma = [[0.0, 0.0], [0.0, 0.0]]
        for i in idx:
            if (i - lead) in x:
                for a in range(2):
                    for b in range(2):
                        gamma[a][b] += x[i][a] * x[i - lead][b] / T
        w = 1.0 if lead == 0 else (1.0 - lead / (lag + 1.0))
        for a in range(2):
            for b in range(2):
                omega[a][b] += gamma[a][b] if lead == 0 else w * (gamma[a][b] + gamma[b][a])
    det = omega[0][0] * omega[1][1] - omega[0][1] * omega[1][0]
    inverse = [[omega[1][1] / det, -omega[0][1] / det], [-omega[1][0] / det, omega[0][0] / det]]
    wald = T * sum(mean[a] * inverse[a][b] * mean[b] for a in range(2) for b in range(2))
    return np.array(mean), np.array(omega), wald, math.exp(-wald / 2.0)


def _pair_series(rng, n=240, missing=0.15):
    e = rng.standard_normal((n, 2))
    series = np.array([0.003, -0.001]) + 0.01 * (e + 0.4 * np.vstack([np.zeros((1, 2)), e[:-1]]))
    series[:, 1] += 0.3 * series[:, 0]  # correlated coefficients
    estimable = rng.random(n) > missing
    estimable[100:110] = False
    return series, estimable


def test_wald_hand_calculation_on_a_short_gapped_grid():
    # 5 calendar months, the third not estimable; T = 4; L = 1
    series = np.array([[1.0, 2.0], [3.0, 1.0], [np.nan, np.nan], [2.0, 4.0], [4.0, 1.0]])
    estimable = np.array([True, True, False, True, True])
    r = fama_macbeth_wald(series, estimable, lag=1)
    # means (10/4, 8/4) = (2.5, 2); x = (-1.5, 0), (0.5, -1), 0, (-0.5, 2), (1.5, -1)
    np.testing.assert_allclose(r.mean, [2.5, 2.0])
    g0 = np.array([[1.5 ** 2 + 0.5 ** 2 + 0.5 ** 2 + 1.5 ** 2, -0.5 - 1.0 - 1.5], [-0.5 - 1.0 - 1.5, 1 + 4 + 1]]) / 4
    # lag-1 calendar pairs: (2,1) and (5,4); the pairs (3,2) and (4,3) involve the gap and vanish
    g1 = (np.outer([0.5, -1.0], [-1.5, 0.0]) + np.outer([1.5, -1.0], [-0.5, 2.0])) / 4
    omega = g0 + 0.5 * (g1 + g1.T)  # w_1 = 1 - 1/2
    np.testing.assert_allclose(r.covariance, omega, rtol=1e-12)
    wald = 4 * r.mean @ np.linalg.solve(omega, r.mean)
    assert r.wald == pytest.approx(wald, rel=1e-12) and r.df == 2 == WALD_DIMENSION
    assert r.p_value == pytest.approx(math.exp(-wald / 2)) and r.n_months == 4 and r.n_grid == 5 and r.lag == 1
    assert isinstance(r, WaldResult)
    mean, omega_ref, wald_ref, p_ref = _explicit_pair_covariance(series, estimable, 1)
    np.testing.assert_allclose(r.covariance, omega_ref, rtol=1e-12)
    assert r.wald == pytest.approx(wald_ref, rel=1e-12) and r.p_value == pytest.approx(p_ref, rel=1e-12)


def test_wald_covariance_with_calendar_gaps_matches_explicit_pairs_and_the_scalar_variances():
    rng = np.random.default_rng(31)
    series, estimable = _pair_series(rng)
    for lag in (0, HAC_LAG, 12):
        r = fama_macbeth_wald(series, estimable, lag=lag)
        mean, omega, wald, p = _explicit_pair_covariance(series, estimable, lag)
        np.testing.assert_allclose(r.mean, mean, rtol=1e-12)
        np.testing.assert_allclose(r.covariance, omega, rtol=1e-11)
        assert r.wald == pytest.approx(wald, rel=1e-10) and r.p_value == pytest.approx(p, rel=1e-10)
        # the diagonal is the scalar HAC variance of each coefficient series; the matrix is symmetric
        for j in range(2):
            scalar = fama_macbeth(series[:, j], estimable, lag=lag)
            assert r.covariance[j, j] == pytest.approx(scalar.hac_variance, rel=1e-12)
            assert r.mean[j] == pytest.approx(scalar.mean, rel=1e-12)
        np.testing.assert_allclose(r.covariance, r.covariance.T, rtol=1e-14)
        # the series is never compressed: dropping the gaps changes the covariance for L > 0 only
        compressed = fama_macbeth_wald(series[estimable], np.ones(int(estimable.sum()), dtype=bool), lag=lag)
        if lag == 0:
            np.testing.assert_allclose(compressed.covariance, r.covariance, rtol=1e-12)
        else:
            assert not np.allclose(compressed.covariance, r.covariance, rtol=1e-6)
    # Bartlett weights: Omega(L) is the weighted sum of the lag matrices with 1 - l / (L + 1)
    T = int(estimable.sum())
    x = np.where(estimable[:, None], series - series[estimable].mean(axis=0), 0.0)
    gammas = [x[l:].T @ x[: len(x) - l] / T for l in range(13)]
    for lag in (0, HAC_LAG, 12):
        expected = gammas[0] + sum((1 - l / (lag + 1)) * (gammas[l] + gammas[l].T) for l in range(1, lag + 1))
        np.testing.assert_allclose(hac_covariance(x, lag, T), expected, rtol=1e-12)
    # values stored at non-estimable months never matter
    other = series.copy()
    other[~estimable] = np.array([1e6, -1e6])
    r_other = fama_macbeth_wald(other, estimable)
    np.testing.assert_array_equal(r_other.covariance, fama_macbeth_wald(series, estimable).covariance)
    assert r_other.wald == fama_macbeth_wald(series, estimable).wald


def test_wald_p_value_matches_scipy_chi_square_and_the_test_is_symmetric_in_the_columns():
    rng = np.random.default_rng(32)
    for shift in ([0.0, 0.0], [0.002, 0.0], [0.0, -0.003], [0.004, 0.004]):
        series, estimable = _pair_series(rng, missing=0.1)
        series = series - series[estimable].mean(axis=0) + np.array(shift)
        r = fama_macbeth_wald(series, estimable)
        assert r.p_value == pytest.approx(float(chi2.sf(r.wald, 2)), rel=1e-10, abs=1e-300)
        assert 0.0 < r.p_value <= 1.0 and r.wald >= 0.0
        swapped = fama_macbeth_wald(series[:, ::-1], estimable)
        assert swapped.wald == pytest.approx(r.wald, rel=1e-12) and swapped.p_value == pytest.approx(r.p_value)
        np.testing.assert_allclose(swapped.mean, r.mean[::-1])
        np.testing.assert_allclose(swapped.covariance, r.covariance[::-1, ::-1])
    # a zero mean vector gives W = 0 and p = 1; the Wald statistic of one coefficient alone is its squared t
    series, estimable = _pair_series(rng)
    centred = series - series[estimable].mean(axis=0)
    assert fama_macbeth_wald(centred, estimable).wald == pytest.approx(0.0, abs=1e-20)
    assert fama_macbeth_wald(centred, estimable).p_value == 1.0
    independent = series.copy()
    independent[:, 1] = rng.standard_normal(len(series))  # the second column is pure noise
    r = fama_macbeth_wald(independent, estimable, lag=0)
    scalar = fama_macbeth(independent[:, 0], estimable, lag=0)
    partial = r.n_months * r.mean[0] ** 2 / r.covariance[0, 0]
    assert partial == pytest.approx(scalar.t ** 2, rel=1e-10)
    assert r.wald >= partial - 1e-12  # adding a coefficient never lowers the quadratic form of a PD matrix


def test_wald_rejects_singular_covariance_and_invalid_inputs():
    rng = np.random.default_rng(33)
    series, estimable = _pair_series(rng)

    def rejected(message: str, s: np.ndarray = series, e: np.ndarray = estimable, lag: object = HAC_LAG):
        with pytest.raises(ValueError, match=message):
            fama_macbeth_wald(s, e, lag=lag)  # type: ignore[arg-type]  # invalid lag types are rejected at run time

    collinear = series.copy()
    collinear[:, 1] = 2.0 * collinear[:, 0]  # perfectly collinear coefficients
    rejected("singular", s=collinear)
    constant = series.copy()
    constant[:, 1] = 0.25  # one constant coefficient
    rejected("singular", s=constant)
    rejected("singular", s=np.ones_like(series))
    two = np.zeros(len(series), dtype=bool)
    two[[3, 7]] = True  # two months: centred vectors are opposite, rank one
    rejected("singular", e=two)
    rejected("at least 2 estimable months", e=np.zeros(len(series), dtype=bool))
    rejected("at least 2 estimable months", e=np.eye(1, len(series), 5, dtype=bool)[0])
    rejected(r"N x 2 array", s=series[:, :1])
    rejected(r"N x 2 array", s=np.column_stack([series, series[:, 0]]))
    rejected(r"N x 2 array", s=series[:, 0])
    rejected(r"N x 2 array", e=estimable[:-1])
    rejected("strictly Boolean", e=estimable.astype(int))
    rejected("finite coefficients", s=np.where(np.arange(len(series))[:, None] == np.flatnonzero(estimable)[0],
                                               np.nan, series))
    rejected("non-negative integer", lag=-1)
    rejected("non-negative integer", lag=6.0)
    rejected("non-negative integer", lag=True)
    # three estimable months are the smallest set with a positive definite 2 x 2 covariance
    three = np.zeros(len(series), dtype=bool)
    three[[3, 7, 11]] = True
    assert fama_macbeth_wald(series, three, lag=0).n_months == 3


def test_wald_rejects_affine_collinearity_that_rounding_leaves_slightly_positive_definite():
    rng = np.random.default_rng(0)
    beta = rng.standard_normal(240)
    series = np.column_stack([beta, 1.0 + 2.0 * beta])  # affine, not proportional: Omega has rank one in theory
    estimable = np.ones(240, dtype=bool)
    T = 240
    x = np.where(estimable[:, None], series - series.mean(axis=0), 0.0)
    omega = hac_covariance(x, HAC_LAG, T)
    assert np.linalg.matrix_rank(omega) == 1  # the smallest eigenvalue is only rounding noise
    assert np.linalg.eigvalsh(omega).min() > -1e-12
    with pytest.raises(ValueError, match="singular"):
        fama_macbeth_wald(series, estimable, lag=HAC_LAG)
    for lag in (0, 12):
        with pytest.raises(ValueError, match="singular"):
            fama_macbeth_wald(series, estimable, lag=lag)
    with pytest.raises(ValueError, match="singular"):  # the same with calendar gaps
        fama_macbeth_wald(series, rng.random(240) > 0.2, lag=HAC_LAG)
    # genuinely positive definite covariances with very different scales are still accepted
    scaled = np.column_stack([beta, 1e-4 * rng.standard_normal(240)])  # variances 1 and 1e-8
    r = fama_macbeth_wald(scaled, estimable)
    assert np.isfinite(r.wald) and r.covariance[1, 1] < 1e-6 * r.covariance[0, 0]
    assert np.linalg.matrix_rank(r.covariance) == 2
    # breaking the collinearity by a single month restores a full-rank covariance
    broken = series.copy()
    broken[17, 1] += 0.05
    assert fama_macbeth_wald(broken, estimable).df == 2
