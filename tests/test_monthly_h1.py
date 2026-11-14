"""Monthly H1 regression engine (specification C) on deterministic synthetic data. No real data is used."""

import numpy as np
import pandas as pd
import pytest

from disagreement.data.validation import DataValidationError
from disagreement.inference.monthly import (
    MIN_CELL,
    MIN_DISTINCT,
    MIN_SINGULAR_RATIO,
    BinBasis,
    bin_basis,
    MonthlyEstimate,
    estimate_h1_month,
    estimate_month,
    SpecificationC,
    scaled_condition,
)
from disagreement.signals.conditional import SIGNAL_BINS, signal_bins

N_BINS = SIGNAL_BINS


def _fitted(estimate: MonthlyEstimate) -> SpecificationC:
    assert estimate.coefficients is not None
    return estimate.coefficients


def _universe(rng, n=800):
    """One synthetic month: S uniform, the existing bin rule, S^c = 2 (S - 0.5), CD^B uniform on [0, 1]."""
    s = rng.random(n)
    bins = signal_bins(pd.Series(s), pd.Series(np.zeros(n, dtype=int)), N_BINS).to_numpy(dtype="int64")
    sc = 2.0 * (s - 0.5)
    cd = rng.random(n)
    return sc, bins, cd


def _true_coefficients(rng):
    theta = np.concatenate([
        rng.normal(0.0, 0.05, N_BINS),  # alpha_b
        rng.normal(0.02, 0.01, N_BINS),  # gamma~_b
        rng.normal(0.0, 0.01, N_BINS),  # eta~_b
        [0.004, -0.012],  # beta_2, beta_3
    ])
    return theta


def test_recovers_known_coefficients_from_synthetic_outcomes():
    rng = np.random.default_rng(20261114)
    sc, bins, cd = _universe(rng)
    basis = bin_basis(sc, bins)
    theta = _true_coefficients(rng)
    X = basis.design(sc, bins, cd)
    y = X @ theta
    est = estimate_month(y, sc, bins, cd, basis)
    assert est.estimable and est.reasons == () and est.n_obs == len(y)
    c = _fitted(est)
    np.testing.assert_allclose(c.intercept, theta[:N_BINS], atol=1e-10)
    np.testing.assert_allclose(c.linear, theta[N_BINS:2 * N_BINS], atol=1e-10)
    np.testing.assert_allclose(c.quadratic, theta[2 * N_BINS:3 * N_BINS], atol=1e-10)
    assert c.disagreement == pytest.approx(0.004, abs=1e-10)
    assert c.interaction == pytest.approx(-0.012, abs=1e-10)
    # with noise the estimates stay close to the truth and the fit equals a direct least-squares solve
    noisy = y + 0.01 * rng.standard_normal(len(y))
    est_noisy = estimate_month(noisy, sc, bins, cd, basis)
    direct = np.linalg.lstsq(X, noisy, rcond=None)[0]
    assert _fitted(est_noisy).interaction == pytest.approx(direct[-1], abs=1e-12)
    assert _fitted(est_noisy).disagreement == pytest.approx(direct[-2], abs=1e-12)
    assert abs(_fitted(est_noisy).interaction - (-0.012)) < 0.02
    assert est.condition_number < 20 and est.singular_ratio > 0.05


def test_bin_local_basis_matches_the_plan():
    rng = np.random.default_rng(1)
    sc, bins, cd = _universe(rng)
    basis = bin_basis(sc, bins)
    assert basis.n_bins == N_BINS
    z, q = basis.local(sc, bins)
    for b in range(1, N_BINS + 1):
        v = sc[bins == b]
        assert basis.centre[b - 1] == pytest.approx(v.mean(), rel=1e-12)  # cell mean of S^c
        assert basis.half_range[b - 1] == pytest.approx((v.max() - v.min()) / 2.0, rel=1e-12)  # half-range
        zb = (v - v.mean()) / basis.half_range[b - 1]
        assert basis.mean_z2[b - 1] == pytest.approx((zb * zb).mean(), rel=1e-12)
        np.testing.assert_allclose(z[bins == b], zb, rtol=1e-9, atol=1e-12)
        assert zb.max() - zb.min() == pytest.approx(2.0, abs=1e-12)  # z spans a range of 2 inside the cell
        assert q[bins == b].mean() == pytest.approx(0.0, abs=1e-12)  # q is centred inside the cell
    X = basis.design(sc, bins, cd)
    assert X.shape == (len(sc), 3 * N_BINS + 2)
    indicator = np.eye(N_BINS)[bins - 1]
    np.testing.assert_array_equal(X[:, :N_BINS], indicator)
    np.testing.assert_allclose(X[:, N_BINS:2 * N_BINS], indicator * z[:, None])
    np.testing.assert_allclose(X[:, 2 * N_BINS:3 * N_BINS], indicator * q[:, None])
    np.testing.assert_array_equal(X[:, -2], cd)  # CD^B enters as saved
    np.testing.assert_array_equal(X[:, -1], sc * cd)  # the interaction uses the raw S^c


def test_interaction_equals_the_unscaled_bin_specific_quadratic_specification():
    rng = np.random.default_rng(2)
    sc, bins, cd = _universe(rng)
    s = sc / 2.0 + 0.5
    y = 0.03 * np.sin(5.0 * s) + 0.01 * cd - 0.02 * sc * cd + 0.05 * rng.standard_normal(len(sc))
    basis = bin_basis(sc, bins)
    est = estimate_month(y, sc, bins, cd, basis)
    # the reference design is built here, independently of the production helpers: per bin an intercept,
    # S^c and (S^c)^2 without rescaling, then CD^B and S^c CD^B
    columns = []
    for power in (0, 1, 2):
        for b in range(1, N_BINS + 1):
            columns.append(np.where(bins == b, sc ** power, 0.0))
    X_original = np.column_stack([*columns, cd, sc * cd])
    assert X_original.shape == (len(sc), 3 * N_BINS + 2)
    beta_o = np.linalg.lstsq(X_original, y, rcond=None)[0]
    assert _fitted(est).interaction == pytest.approx(beta_o[-1], rel=1e-8, abs=1e-12)
    assert _fitted(est).disagreement == pytest.approx(beta_o[-2], rel=1e-8, abs=1e-12)
    # the bin-local coefficients describe the same fitted signal relation: slope on S^c inside bin b
    c = _fitted(est)
    z, _ = basis.local(sc, bins)
    idx = bins - 1
    local_slope = (c.linear[idx] + 2.0 * c.quadratic[idx] * z) / basis.half_range[idx]
    original_slope = beta_o[N_BINS:2 * N_BINS][idx] + 2.0 * beta_o[2 * N_BINS:3 * N_BINS][idx] * sc
    np.testing.assert_allclose(local_slope, original_slope, rtol=1e-6, atol=1e-9)
    assert scaled_condition(basis.design(sc, bins, cd))[0] < scaled_condition(X_original)[0] / 10


def test_basis_constants_come_from_the_full_universe_and_ignore_missing_outcomes():
    rng = np.random.default_rng(3)
    sc, bins, cd = _universe(rng, n=1200)
    theta = _true_coefficients(rng)
    full_basis = bin_basis(sc, bins)
    outcome = full_basis.design(sc, bins, cd) @ theta + 0.01 * rng.standard_normal(len(sc))
    complete = estimate_h1_month(outcome, sc, bins, cd)
    missing = outcome.copy()
    gone = rng.random(len(sc)) < 0.25
    missing[gone] = np.nan
    missing[np.flatnonzero(~gone)[:5]] = np.inf  # a non-finite outcome is missing too
    partial = estimate_h1_month(missing, sc, bins, cd)
    for attr in ("centre", "half_range", "mean_z2"):
        np.testing.assert_array_equal(getattr(partial.basis, attr), getattr(complete.basis, attr))
        np.testing.assert_array_equal(getattr(partial.basis, attr), getattr(full_basis, attr))
    sample = np.isfinite(missing)
    assert partial.n_obs == int(sample.sum()) < complete.n_obs == len(sc)
    assert partial.cell_counts.sum() == partial.n_obs
    # the sample regression equals estimate_month on the sample with the full-universe basis ...
    same = estimate_month(missing[sample], sc[sample], bins[sample], cd[sample], full_basis)
    assert _fitted(same).interaction == _fitted(partial).interaction
    np.testing.assert_array_equal(_fitted(same).linear, _fitted(partial).linear)
    # ... and a basis recomputed on the sample would differ: formation-time constants are not recomputed
    recomputed = bin_basis(sc[sample], bins[sample])
    assert not np.allclose(recomputed.centre, full_basis.centre, atol=1e-6)
    assert not np.allclose(recomputed.half_range, full_basis.half_range, atol=1e-6)
    assert partial.estimable and abs(_fitted(partial).interaction - theta[-1]) < 0.02


def _grid_universe(n_per_bin=40):
    """Bins given directly, S^c on a fixed grid inside disjoint ranges, CD^B on a deterministic grid."""
    bins = np.repeat(np.arange(1, N_BINS + 1), n_per_bin)
    offsets = np.tile(np.linspace(0.0, 0.09, n_per_bin), N_BINS)
    sc = -1.0 + (bins - 1) * 0.1 + offsets
    cd = ((7 * np.arange(len(bins))) % n_per_bin) / (n_per_bin - 1)  # a permutation of the grid inside each bin
    return sc, bins, cd


def test_exactly_30_observations_and_3_distinct_signal_values_per_bin_are_permitted():
    sc, bins, cd = _grid_universe(n_per_bin=MIN_CELL)  # exactly 30 observations in every cell
    three = np.repeat(np.array([0.00, 0.045, 0.09]), 10)  # exactly 3 distinct S^c values per cell, 10 each
    sc = -1.0 + (bins - 1) * 0.1 + np.tile(three, N_BINS)
    basis = bin_basis(sc, bins)
    y = 0.01 * sc + 0.002 * cd - 0.005 * sc * cd
    est = estimate_month(y, sc, bins, cd, basis)
    assert est.cell_counts.tolist() == [MIN_CELL] * N_BINS
    assert est.cell_distinct.tolist() == [MIN_DISTINCT] * N_BINS
    assert est.singular_ratio > MIN_SINGULAR_RATIO  # the full design satisfies the singular-value criterion
    assert est.estimable and est.reasons == () and est.coefficients is not None
    assert _fitted(est).interaction == pytest.approx(-0.005, abs=1e-10)
    # one observation or one distinct value fewer in a single cell, and the month is no longer estimable
    keep = np.arange(len(sc)) != 0  # bin 1 keeps 29 observations
    assert not estimate_month(y[keep], sc[keep], bins[keep], cd[keep], basis).estimable
    two = sc.copy()
    two[10:20] = -1.0  # the middle value of bin 1 joins the lowest: bin 1 keeps 2 distinct values
    assert len(np.unique(two[bins == 1])) == 2
    assert not estimate_month(y, two, bins, cd, bin_basis(two, bins)).estimable


def test_rejects_insufficiently_populated_or_singular_designs_explicitly():
    sc, bins, cd = _grid_universe()
    basis = bin_basis(sc, bins)
    y = 0.01 * sc + 0.002 * cd
    assert estimate_month(y, sc, bins, cd, basis).estimable

    # a cell below 30 sample observations
    keep = ~((bins == 5) & (np.arange(len(sc)) % 40 < 11))  # 29 of 40 left in bin 5
    short = estimate_month(y[keep], sc[keep], bins[keep], cd[keep], basis)
    assert not short.estimable and short.coefficients is None
    assert short.cell_counts[4] == 29 and short.cell_counts.min() == 29
    assert any(f"fewer than {MIN_CELL} observations" in r for r in short.reasons)

    # a cell with only two distinct values of S^c in the sample (enough observations)
    sc2 = sc.copy()
    sc2[bins == 7] = np.where(np.arange(len(sc))[bins == 7] % 2 == 0, -0.4, -0.35)
    basis2 = bin_basis(sc2, bins)
    two = estimate_month(y, sc2, bins, cd, basis2)
    assert not two.estimable and two.cell_distinct[6] == 2 and two.cell_distinct.min() == 2
    assert any(f"fewer than {MIN_DISTINCT} distinct" in r for r in two.reasons)
    assert all(f"fewer than {MIN_CELL} observations" not in r for r in two.reasons)

    # a singular design: a constant CD^B is collinear with the bin intercepts
    constant = np.full_like(cd, 0.5)
    singular = estimate_month(y, sc, bins, constant, basis)
    assert not singular.estimable and singular.singular_ratio <= MIN_SINGULAR_RATIO
    assert any("singular-value ratio" in r for r in singular.reasons) and singular.coefficients is None

    # a bin whose outcomes are all missing is an empty cell of the H1 sample, reported, not dropped
    outcome = y.astype("float64")
    outcome[bins == 3] = np.nan
    empty = estimate_h1_month(outcome, sc, bins, cd)
    assert not empty.estimable and empty.cell_counts[2] == 0 and empty.singular_ratio == 0.0
    assert len(empty.reasons) >= 2  # population and singularity are both reported

    # a universe that does not populate every bin has no basis
    with pytest.raises(DataValidationError, match="populate"):
        bin_basis(sc, np.where(bins == 3, 4, bins))


def test_coefficients_are_recovered_when_the_signal_range_is_tiny():
    """A valid design whose raw interaction column has a tiny norm: the solve must follow the scaled criterion."""
    rng = np.random.default_rng(7)
    n = 1000
    u = rng.random(n)
    sc = 1e-14 * (2.0 * u - 1.0)  # S^c spans about [-1e-14, 1e-14]: a legitimate but extreme input
    bins = signal_bins(pd.Series(sc), pd.Series(np.zeros(n, dtype=int)), N_BINS).to_numpy(dtype="int64")
    raw = rng.random(n)
    cd = np.empty(n)
    for b in range(1, N_BINS + 1):  # conditional disagreement ranks inside each bin
        m = bins == b
        cd[m] = (pd.Series(raw[m]).rank().to_numpy() - 1.0) / (m.sum() - 1.0)
    basis = bin_basis(sc, bins)
    X = basis.design(sc, bins, cd)
    # a pure interaction outcome: the raw interaction column has norm ~1e-14 and a least-squares solve on the
    # unscaled design truncates it, returning beta_3 near 0; the column-normalised criterion accepts the design
    y = -0.4 * sc * cd
    est = estimate_month(y, sc, bins, cd, basis)
    assert est.estimable and est.singular_ratio > 0.1
    assert abs(np.linalg.lstsq(X, y, rcond=None)[0][-1]) < 0.01  # the failure the scaled solve avoids
    assert _fitted(est).interaction == pytest.approx(-0.4, rel=1e-8)
    assert _fitted(est).disagreement == pytest.approx(0.0, abs=1e-8)
    np.testing.assert_allclose(_fitted(est).intercept, 0.0, atol=1e-16)
    # with every coefficient active the interaction contributes ~1e-15 to an outcome of size ~0.05, so the
    # synthetic outcome itself resolves beta_3 only to about one percent; the estimate is within that limit
    theta = _true_coefficients(rng)
    theta[-2:] = [0.05, -0.4]
    y = X @ theta
    est = estimate_month(y, sc, bins, cd, basis)
    assert abs(np.linalg.lstsq(X, y, rcond=None)[0][-1]) < 0.01  # the unscaled solve still drops it
    assert _fitted(est).interaction == pytest.approx(-0.4, rel=0.03)
    assert _fitted(est).disagreement == pytest.approx(0.05, abs=1e-6)
    np.testing.assert_allclose(_fitted(est).intercept, theta[:N_BINS], atol=1e-6)
    np.testing.assert_allclose(_fitted(est).linear, theta[N_BINS:2 * N_BINS], atol=1e-6)
    np.testing.assert_allclose(_fitted(est).quadratic, theta[2 * N_BINS:3 * N_BINS], atol=1e-6)
    # the coefficients are reported in the original units: they reproduce the outcome to machine precision
    fitted = X @ np.concatenate([_fitted(est).intercept, _fitted(est).linear, _fitted(est).quadratic,
                                 [_fitted(est).disagreement, _fitted(est).interaction]])
    np.testing.assert_allclose(fitted, y, atol=1e-15)


def test_results_do_not_depend_on_row_order():
    rng = np.random.default_rng(5)
    sc, bins, cd = _universe(rng)
    outcome = 0.02 * sc - 0.01 * sc * cd + 0.05 * rng.standard_normal(len(sc))
    outcome[rng.random(len(sc)) < 0.1] = np.nan
    a = estimate_h1_month(outcome, sc, bins, cd)
    perm = rng.permutation(len(sc))
    b = estimate_h1_month(outcome[perm], sc[perm], bins[perm], cd[perm])
    assert a.estimable and b.estimable and a.n_obs == b.n_obs
    for attr in ("centre", "half_range", "mean_z2"):
        np.testing.assert_allclose(getattr(a.basis, attr), getattr(b.basis, attr), rtol=1e-13)
    np.testing.assert_array_equal(a.cell_counts, b.cell_counts)
    np.testing.assert_array_equal(a.cell_distinct, b.cell_distinct)
    for attr in ("intercept", "linear", "quadratic"):
        np.testing.assert_allclose(getattr(_fitted(a), attr), getattr(_fitted(b), attr), rtol=1e-9, atol=1e-13)
    assert _fitted(a).interaction == pytest.approx(_fitted(b).interaction, rel=1e-10, abs=1e-14)
    assert _fitted(a).disagreement == pytest.approx(_fitted(b).disagreement, rel=1e-10, abs=1e-14)


def test_invalid_inputs_are_rejected():
    rng = np.random.default_rng(6)
    sc, bins, cd = _universe(rng)
    basis = bin_basis(sc, bins)
    y = rng.standard_normal(len(sc))

    def rejected(message: str, y_: np.ndarray = y, sc_: np.ndarray = sc, bins_: np.ndarray = bins,
                 cd_: np.ndarray = cd, basis_: BinBasis = basis):
        with pytest.raises(ValueError, match=message):
            estimate_month(y_, sc_, bins_, cd_, basis_)

    rejected("finite", y_=np.where(np.arange(len(y)) == 0, np.nan, y))  # the sample outcome must be finite
    rejected("one value per sample", y_=y[:-1])
    rejected("same length", sc_=sc[:-1])
    rejected("one-dimensional", sc_=sc.reshape(-1, 1), bins_=bins.reshape(-1, 1), cd_=cd.reshape(-1, 1))
    rejected(r"S\^c must be finite", sc_=np.where(np.arange(len(sc)) == 3, np.nan, sc))
    rejected(r"S\^c must be finite", sc_=np.where(np.arange(len(sc)) == 3, 1.5, sc))  # outside [-1, 1]
    rejected(r"CD\^B must be finite", cd_=np.where(np.arange(len(cd)) == 3, np.inf, cd))
    rejected(r"CD\^B must be finite", cd_=cd + 0.5)  # above 1
    rejected("signal_bin must lie", bins_=np.where(np.arange(len(bins)) == 2, 0, bins))
    rejected("signal_bin must lie", bins_=np.where(np.arange(len(bins)) == 2, N_BINS + 1, bins))
    rejected("signal_bin must be integers", bins_=bins.astype("float64"))
    with pytest.raises(ValueError, match="one value per member"):
        estimate_h1_month(y[:-1], sc, bins, cd)
    none = estimate_h1_month(np.full(len(sc), np.nan), sc, bins, cd)  # an empty H1 sample is reported, not raised
    assert not none.estimable and none.coefficients is None and none.n_obs == 0
    assert none.reasons == ("no member of U_t has a finite outcome: the H1 sample is empty",)
    assert none.cell_counts.tolist() == [0] * N_BINS and none.singular_ratio == 0.0
    np.testing.assert_array_equal(none.basis.centre, basis.centre)  # the full-universe basis is still reported
    with pytest.raises(ValueError, match="no observations"):
        bin_basis(np.array([]), np.array([], dtype="int64"))
    # a basis of another width than the bins is a programming error, not silently broadcast
    narrow = BinBasis(basis.centre[:10], basis.half_range[:10], basis.mean_z2[:10])
    with pytest.raises(ValueError, match="signal_bin must lie"):
        estimate_month(y, sc, bins, cd, narrow)
