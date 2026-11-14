"""Specification B, H2, H3 and the marginal signal slopes on deterministic synthetic data. No real data is used."""

import numpy as np
import pytest

from disagreement.controls.h2 import INTERACTIONS, RANKS
from disagreement.inference.monthly import (
    H2_EXTRAS,
    H3_EXTRAS,
    SLOPE_DISAGREEMENT,
    SPECIFICATION_B,
    SPECIFICATION_C,
    SPECIFICATION_C_H2,
    SPECIFICATION_H2,
    SPECIFICATION_H3,
    Coefficients,
    MonthlyEstimate,
    bin_basis,
    design_matrix,
    estimate_b,
    estimate_h2,
    estimate_h3,
    estimate_month,
    estimate_month_specifications,
    estimate_specification,
    marginal_slopes,
)
from test_monthly_h1 import N_BINS, _true_coefficients, _universe


def _fitted(estimate: MonthlyEstimate) -> Coefficients:
    assert estimate.coefficients is not None
    return estimate.coefficients


def _month(rng, n=900):
    """A synthetic month with CW, three centred control ranks (some missing) and a complete-case flag."""
    sc, bins, cd = _universe(rng, n)
    cw = rng.random(n)
    controls = 2.0 * rng.random((n, 3)) - 1.0
    controls[rng.random(n) < 0.08, 2] = np.nan  # illiquidity missing for some stocks
    controls[rng.random(n) < 0.02, 1] = np.nan
    complete_case = np.isfinite(controls).all(axis=1)
    return sc, bins, cd, cw, controls, complete_case


def _reference_design(sc, bins, cd, degree, extra=None):
    """An independent construction of the design: indicators, bin-specific z and q from the full universe."""
    columns = []
    for b in range(1, N_BINS + 1):
        columns.append((bins == b).astype(float))
    for b in range(1, N_BINS + 1):
        v = sc[bins == b]
        columns.append(np.where(bins == b, (sc - v.mean()) / ((v.max() - v.min()) / 2.0), 0.0))
    if degree == 2:
        for b in range(1, N_BINS + 1):
            v = sc[bins == b]
            z = (v - v.mean()) / ((v.max() - v.min()) / 2.0)
            zz = np.where(bins == b, ((sc - v.mean()) / ((v.max() - v.min()) / 2.0)) ** 2 - (z * z).mean(), 0.0)
            columns.append(zz)
    columns += [cd, sc * cd]
    if extra is not None:
        columns += [extra[:, j] for j in range(extra.shape[1])]
    return np.column_stack(columns)


def _theta(rng, spec):
    theta = _true_coefficients(rng)
    if spec == SPECIFICATION_B:
        theta = np.concatenate([theta[:2 * N_BINS], theta[-2:]])
    if spec == SPECIFICATION_H2:
        theta = np.concatenate([theta, [0.01, -0.02, 0.015, 0.03, -0.01, 0.02]])  # delta levels, delta interactions
    if spec == SPECIFICATION_H3:
        theta = np.concatenate([theta, [0.006, -0.009]])  # CW level, S^c x CW
    return theta


def _estimate(spec, y, sc, bins, cd, basis, cw: np.ndarray, controls: np.ndarray):
    if spec == SPECIFICATION_B:
        return estimate_b(y, sc, bins, cd, basis)
    if spec == SPECIFICATION_H2:
        return estimate_h2(y, sc, bins, cd, controls, basis)
    return estimate_h3(y, sc, bins, cd, cw, basis)


def _extra(spec, sc, cw, controls):
    if spec == SPECIFICATION_H2:
        return np.column_stack([controls, sc[:, None] * controls])
    if spec == SPECIFICATION_H3:
        return np.column_stack([cw, sc * cw])
    return None


@pytest.mark.parametrize("spec", [SPECIFICATION_B, SPECIFICATION_H2, SPECIFICATION_H3])
def test_recovers_known_coefficients_and_matches_an_independent_ols(spec):
    rng = np.random.default_rng(22)
    sc, bins, cd, cw, controls, complete = _month(rng)
    controls = np.where(np.isfinite(controls), controls, 0.3)  # a complete sample for this test
    basis = bin_basis(sc, bins)
    extra = _extra(spec, sc, cw, controls)
    X = _reference_design(sc, bins, cd, 1 if spec == SPECIFICATION_B else 2, extra)
    theta = _theta(rng, spec)
    assert X.shape[1] == len(theta)
    y = X @ theta
    est = _estimate(spec, y, sc, bins, cd, basis, cw, controls)
    assert est.specification == spec and est.estimable and est.n_obs == len(y)
    c = _fitted(est)
    flat = np.concatenate([c.intercept, c.linear, c.quadratic, [c.disagreement, c.interaction], list(c.extra.values())])
    np.testing.assert_allclose(flat, theta, atol=1e-9)
    noisy = y + 0.02 * rng.standard_normal(len(y))
    reference = np.linalg.lstsq(X, noisy, rcond=None)[0]
    c = _fitted(_estimate(spec, noisy, sc, bins, cd, basis, cw, controls))
    flat = np.concatenate([c.intercept, c.linear, c.quadratic, [c.disagreement, c.interaction], list(c.extra.values())])
    np.testing.assert_allclose(flat, reference, rtol=1e-8, atol=1e-12)


def test_design_columns_and_coefficient_layout():
    rng = np.random.default_rng(23)
    sc, bins, cd, cw, controls, complete = _month(rng, n=700)
    controls = np.where(np.isfinite(controls), controls, -0.2)
    basis = bin_basis(sc, bins)
    z, q = basis.local(sc, bins)
    indicator = np.eye(N_BINS)[bins - 1]
    # B: intercepts, z, CD^B, S^c CD^B
    XB = design_matrix(basis, sc, bins, cd, 1)
    assert XB.shape == (len(sc), 2 * N_BINS + 2)
    np.testing.assert_array_equal(XB[:, :N_BINS], indicator)
    np.testing.assert_allclose(XB[:, N_BINS:2 * N_BINS], indicator * z[:, None])
    np.testing.assert_array_equal(XB[:, -2], cd)
    np.testing.assert_array_equal(XB[:, -1], sc * cd)
    # H2: C columns, then the three ranks, then S^c times each rank, in the order of H2_EXTRAS
    extra = np.column_stack([controls, sc[:, None] * controls])
    XH2 = design_matrix(basis, sc, bins, cd, 2, extra)
    assert XH2.shape == (len(sc), 3 * N_BINS + 2 + 6)
    np.testing.assert_array_equal(XH2[:, :3 * N_BINS + 2], basis.design(sc, bins, cd))
    np.testing.assert_array_equal(XH2[:, 3 * N_BINS + 2:3 * N_BINS + 5], controls)
    np.testing.assert_array_equal(XH2[:, 3 * N_BINS + 5:], sc[:, None] * controls)  # the raw S^c, not z
    assert H2_EXTRAS == (*RANKS, *INTERACTIONS) and H3_EXTRAS == ("cw", "signal_x_cw")
    # H3: C columns, CW, S^c CW
    XH3 = design_matrix(basis, sc, bins, cd, 2, np.column_stack([cw, sc * cw]))
    np.testing.assert_array_equal(XH3[:, -2], cw)
    np.testing.assert_array_equal(XH3[:, -1], sc * cw)
    assert XH3.shape[1] == 3 * N_BINS + 4
    # coefficient objects follow the same layout and keep every added term by name
    y = rng.standard_normal(len(sc))
    b, h2, h3 = (estimate_b(y, sc, bins, cd, basis), estimate_h2(y, sc, bins, cd, controls, basis),
                 estimate_h3(y, sc, bins, cd, cw, basis))
    assert _fitted(b).quadratic.size == 0 and _fitted(b).extra == {}
    assert tuple(_fitted(h2).extra) == H2_EXTRAS and tuple(_fitted(h3).extra) == H3_EXTRAS
    assert _fitted(estimate_month(y, sc, bins, cd, basis)).extra == {}
    for est in (b, h2, h3):
        assert _fitted(est).intercept.shape == _fitted(est).linear.shape == (N_BINS,)
        assert isinstance(_fitted(est).interaction, float) and isinstance(_fitted(est).disagreement, float)
    beta_h2 = np.linalg.lstsq(XH2, y, rcond=None)[0]
    assert _fitted(h2).interaction == pytest.approx(beta_h2[3 * N_BINS + 1], rel=1e-8)  # beta_3 stays identifiable
    assert _fitted(h2).extra["signal_x_illiquidity_rank"] == pytest.approx(beta_h2[-1], rel=1e-8)


def test_samples_follow_the_frozen_rules_and_c_is_re_estimated_on_the_exact_h2_sample():
    rng = np.random.default_rng(24)
    sc, bins, cd, cw, controls, complete = _month(rng, n=1500)
    basis = bin_basis(sc, bins)
    outcome = basis.design(sc, bins, cd) @ _true_coefficients(rng) + 0.03 * rng.standard_normal(len(sc))
    outcome[rng.random(len(sc)) < 0.15] = np.nan
    out = estimate_month_specifications(outcome, sc, bins, cd, cw, controls, complete)
    assert set(out) == {SPECIFICATION_C, SPECIFICATION_B, SPECIFICATION_H3, SPECIFICATION_H2, SPECIFICATION_C_H2}
    h1 = np.isfinite(outcome)
    h2 = h1 & complete
    assert 0 < h2.sum() < h1.sum() < len(sc)
    for spec in (SPECIFICATION_C, SPECIFICATION_B, SPECIFICATION_H3):
        assert out[spec].n_obs == h1.sum() and out[spec].specification == spec
    for spec in (SPECIFICATION_H2, SPECIFICATION_C_H2):
        assert out[spec].n_obs == h2.sum() and out[spec].specification == spec
    for est in out.values():  # one basis, from every member of U_t, for every specification and sample
        assert est.estimable
        np.testing.assert_array_equal(est.basis.centre, basis.centre)
        np.testing.assert_array_equal(est.basis.half_range, basis.half_range)
    # C on the H2 sample equals C estimated by hand on exactly that sample with the full-universe basis
    by_hand = estimate_month(outcome[h2], sc[h2], bins[h2], cd[h2], basis)
    assert _fitted(out[SPECIFICATION_C_H2]).interaction == _fitted(by_hand).interaction
    np.testing.assert_array_equal(_fitted(out[SPECIFICATION_C_H2]).linear, _fitted(by_hand).linear)
    assert _fitted(out[SPECIFICATION_C_H2]).interaction != _fitted(out[SPECIFICATION_C]).interaction
    # H2 by hand on the same sample, with the controls of the complete cases only
    h2_hand = estimate_h2(outcome[h2], sc[h2], bins[h2], cd[h2], controls[h2], basis)
    assert _fitted(out[SPECIFICATION_H2]).extra == _fitted(h2_hand).extra
    # H3 uses the H1 sample, keeps CD^B and CW apart, and the H1 coefficient of C is unchanged by the call
    h3_hand = estimate_h3(outcome[h1], sc[h1], bins[h1], cd[h1], cw[h1], basis)
    assert _fitted(out[SPECIFICATION_H3]).extra == _fitted(h3_hand).extra
    assert set(_fitted(out[SPECIFICATION_H3]).extra) == {"cw", "signal_x_cw"}
    assert _fitted(out[SPECIFICATION_H3]).interaction != _fitted(out[SPECIFICATION_H3]).extra["signal_x_cw"]
    # a complete case with a missing rank is a data error; ranks missing outside complete cases are fine
    bad = complete.copy()
    bad[np.flatnonzero(~complete)[0]] = True
    with pytest.raises(ValueError, match="complete case has a missing control rank"):
        estimate_month_specifications(outcome, sc, bins, cd, cw, controls, bad)
    # empty samples are reported, not raised
    none = estimate_month_specifications(np.full(len(sc), np.nan), sc, bins, cd, cw, controls, complete)
    assert all(not e.estimable and e.coefficients is None and e.n_obs == 0 for e in none.values())
    assert "H1 sample is empty" in none[SPECIFICATION_B].reasons[0]
    assert "H2 sample is empty" in none[SPECIFICATION_H2].reasons[0]
    only_h1 = estimate_month_specifications(outcome, sc, bins, cd, cw, controls, np.zeros(len(sc), dtype=bool))
    assert only_h1[SPECIFICATION_C].estimable and not only_h1[SPECIFICATION_C_H2].estimable


def test_added_terms_can_make_a_month_non_estimable_while_c_remains_estimable():
    rng = np.random.default_rng(25)
    sc, bins, cd, cw, controls, complete = _month(rng, n=1000)
    basis = bin_basis(sc, bins)
    y = 0.01 * sc + 0.1 * rng.standard_normal(len(sc))
    assert estimate_month(y, sc, bins, cd, basis).estimable
    # a control rank identical to CD^B (rescaled) is collinear with the CD^B level and interaction columns
    collinear = np.column_stack([2.0 * cd - 1.0, rng.random(len(sc)), rng.random(len(sc))])
    h2 = estimate_h2(y, sc, bins, cd, collinear, basis)
    assert not h2.estimable and h2.coefficients is None
    assert h2.cell_counts.min() >= 30 and h2.cell_distinct.min() >= 3  # only the design criterion fails
    assert h2.reasons == (f"scaled singular-value ratio {h2.singular_ratio:.2e} not above 1e-08",)
    # CW equal to CD^B makes H3 singular while C is estimable on the same sample
    h3 = estimate_h3(y, sc, bins, cd, cd, basis)
    assert not h3.estimable and "singular-value ratio" in h3.reasons[0]
    assert estimate_h3(y, sc, bins, cd, cw, basis).estimable
    # the population rules apply to the added specifications with the same thresholds
    keep = ~((bins == 2) & (np.cumsum(bins == 2) <= (bins == 2).sum() - 29))  # bin 2 keeps 29 observations
    for est in (estimate_b(y[keep], sc[keep], bins[keep], cd[keep], basis),
                estimate_h3(y[keep], sc[keep], bins[keep], cd[keep], cw[keep], basis)):
        assert not est.estimable and est.cell_counts[1] == 29
        assert any("fewer than 30 observations" in r for r in est.reasons)


def test_marginal_slopes_at_the_bin_centre_and_the_three_disagreement_levels():
    rng = np.random.default_rng(26)
    sc, bins, cd, cw, controls, complete = _month(rng)
    controls = np.where(np.isfinite(controls), controls, 0.1)
    basis = bin_basis(sc, bins)
    y = rng.standard_normal(len(sc))
    c = _fitted(estimate_month(y, sc, bins, cd, basis))
    expected_centre = np.mean([c.linear[b] / basis.half_range[b] for b in range(N_BINS)])
    slopes = marginal_slopes(c, basis)
    assert tuple(slopes) == SLOPE_DISAGREEMENT == (0.1, 0.5, 0.9)
    for level in SLOPE_DISAGREEMENT:
        assert slopes[level] == pytest.approx(expected_centre + level * c.interaction, rel=1e-12)
    # the quadratic coefficients do not enter at z = 0: B and C slopes share the same formula
    b = _fitted(estimate_b(y, sc, bins, cd, basis))
    assert marginal_slopes(b, basis)[0.5] == pytest.approx(np.mean(b.linear / basis.half_range) + 0.5 * b.interaction)
    # H2: primary evaluation at X^c = 0, descriptive evaluation at caller-supplied control means
    h2 = _fitted(estimate_h2(y, sc, bins, cd, controls, basis))
    primary = marginal_slopes(h2, basis)
    assert primary[0.9] == pytest.approx(np.mean(h2.linear / basis.half_range) + 0.9 * h2.interaction, rel=1e-12)
    assert primary == marginal_slopes(h2, basis, controls_at=np.zeros(3))
    means = controls.mean(axis=0)
    descriptive = marginal_slopes(h2, basis, controls_at=means)
    delta = np.array([h2.extra[name] for name in INTERACTIONS])
    for level in SLOPE_DISAGREEMENT:
        assert descriptive[level] == pytest.approx(primary[level] + float(delta @ means), rel=1e-12)
    assert set(h2.extra) - set(INTERACTIONS) == set(RANKS)  # the level coefficients delta_k^level do not enter
    # other levels can be requested; H3 coefficients and misuse are rejected
    assert set(marginal_slopes(c, basis, levels=(0.0, 1.0))) == {0.0, 1.0}
    h3 = _fitted(estimate_h3(y, sc, bins, cd, cw, basis))
    with pytest.raises(ValueError, match="B, C and H2 only"):
        marginal_slopes(h3, basis)
    with pytest.raises(ValueError, match="H2 coefficients only"):
        marginal_slopes(c, basis, controls_at=np.zeros(3))
    with pytest.raises(ValueError, match="3 finite numbers"):
        marginal_slopes(h2, basis, controls_at=np.array([0.0, 2.0, 0.0]))
    with pytest.raises(ValueError, match="same number of bins"):
        marginal_slopes(c, bin_basis(sc[bins <= 10], bins[bins <= 10], n_bins=10))


def test_extra_regressors_must_match_the_specification_exactly():
    rng = np.random.default_rng(28)
    sc, bins, cd, cw, controls, complete = _month(rng, n=700)
    controls = np.where(np.isfinite(controls), controls, 0.0)
    basis = bin_basis(sc, bins)
    y = rng.standard_normal(len(sc))
    h2_extra = np.column_stack([controls, sc[:, None] * controls])
    h3_extra = np.column_stack([cw, sc * cw])

    def rejected(message, spec, extra=None, names=()):
        with pytest.raises(ValueError, match=message):
            estimate_specification(y, sc, bins, cd, basis, spec, extra, names)

    # the valid calls, for reference
    assert estimate_specification(y, sc, bins, cd, basis, SPECIFICATION_H2, h2_extra, H2_EXTRAS).estimable
    assert estimate_specification(y, sc, bins, cd, basis, SPECIFICATION_H3, h3_extra, H3_EXTRAS).estimable
    assert estimate_specification(y, sc, bins, cd, basis, SPECIFICATION_C_H2).estimable
    # omitted extras
    rejected("needs the extra regressors", SPECIFICATION_H2, None, H2_EXTRAS)
    rejected("exactly the extra regressors", SPECIFICATION_H2)
    rejected("exactly the extra regressors", SPECIFICATION_H3)
    # mismatched or reordered names, wrong name count
    rejected("exactly the extra regressors", SPECIFICATION_H2, h2_extra, H3_EXTRAS)
    rejected("exactly the extra regressors", SPECIFICATION_H3, h3_extra, H2_EXTRAS)
    rejected("exactly the extra regressors", SPECIFICATION_H2, h2_extra, (*INTERACTIONS, *RANKS))
    rejected("exactly the extra regressors", SPECIFICATION_H2, h2_extra, H2_EXTRAS[:5])
    rejected("exactly the extra regressors", SPECIFICATION_H3, h3_extra, ("signal_x_cw", "cw"))
    rejected("exactly the extra regressors", SPECIFICATION_H2, h2_extra, (*H2_EXTRAS[:5], "other"))
    # a matrix whose columns do not match the names, or with a wrong number of rows, or not finite
    rejected("6 columns", SPECIFICATION_H2, h2_extra[:, :5], H2_EXTRAS)
    rejected("6 columns", SPECIFICATION_H2, np.column_stack([h2_extra, cw]), H2_EXTRAS)
    rejected("2 columns", SPECIFICATION_H3, cw, H3_EXTRAS)  # a single vector is not the two H3 columns
    rejected("one row per sample", SPECIFICATION_H3, h3_extra[:-1], H3_EXTRAS)
    rejected("no imputation", SPECIFICATION_H2, np.where(np.arange(len(y))[:, None] == 3, np.nan, h2_extra), H2_EXTRAS)
    # unexpected extras on specifications that take none
    for spec in (SPECIFICATION_B, SPECIFICATION_C, SPECIFICATION_C_H2):
        rejected("takes no extra regressors", spec, h3_extra)
        rejected("exactly the extra regressors", spec, h3_extra, H3_EXTRAS)
    with pytest.raises(KeyError):
        estimate_specification(y, sc, bins, cd, basis, "H4")


def test_row_order_independence_and_invalid_inputs():
    rng = np.random.default_rng(27)
    sc, bins, cd, cw, controls, complete = _month(rng, n=1500)
    outcome = 0.02 * sc + 0.05 * rng.standard_normal(len(sc))
    outcome[rng.random(len(sc)) < 0.1] = np.nan
    a = estimate_month_specifications(outcome, sc, bins, cd, cw, controls, complete)
    p = rng.permutation(len(sc))
    b = estimate_month_specifications(outcome[p], sc[p], bins[p], cd[p], cw[p], controls[p], complete[p])
    for spec in a:
        assert a[spec].n_obs == b[spec].n_obs and a[spec].estimable and b[spec].estimable
        assert _fitted(a[spec]).interaction == pytest.approx(_fitted(b[spec]).interaction, rel=1e-9, abs=1e-13)
        for name in _fitted(a[spec]).extra:
            assert _fitted(a[spec]).extra[name] == pytest.approx(_fitted(b[spec]).extra[name], rel=1e-9, abs=1e-13)
        np.testing.assert_allclose(_fitted(a[spec]).linear, _fitted(b[spec]).linear, rtol=1e-9, atol=1e-13)
    basis = bin_basis(sc, bins)
    finite = np.isfinite(outcome)
    y, s, bn, d, w, x = outcome[finite], sc[finite], bins[finite], cd[finite], cw[finite], controls[finite]
    x = np.where(np.isfinite(x), x, 0.0)
    with pytest.raises(ValueError, match="3 columns"):
        estimate_h2(y, s, bn, d, x[:, :2], basis)
    with pytest.raises(ValueError, match="no imputation"):
        estimate_h2(y, s, bn, d, np.where(np.arange(len(y))[:, None] == 0, np.nan, x), basis)
    with pytest.raises(ValueError, match=r"lie in \[-1, 1\]"):
        estimate_h2(y, s, bn, d, x + 1.5, basis)
    with pytest.raises(ValueError, match=r"CW must lie in \[0, 1\]"):
        estimate_h3(y, s, bn, d, w + 0.5, basis)
    with pytest.raises(ValueError, match="no imputation"):
        estimate_h3(y, s, bn, d, np.where(np.arange(len(y)) == 0, np.nan, w), basis)
    with pytest.raises(ValueError, match="one row per sample"):
        estimate_h3(y, s, bn, d, w[:-1], basis)
    with pytest.raises(ValueError, match="strictly Boolean"):
        estimate_month_specifications(outcome, sc, bins, cd, cw, controls, complete.astype(int))
    with pytest.raises(ValueError, match=r"CW must be finite"):
        estimate_month_specifications(outcome, sc, bins, cd, np.where(np.arange(len(sc)) == 0, np.nan, cw),
                                      controls, complete)
    with pytest.raises(ValueError, match="3 columns"):
        estimate_month_specifications(outcome, sc, bins, cd, cw, controls[:, :2], complete)
    with pytest.raises(ValueError, match="one value per member"):
        estimate_month_specifications(outcome[:-1], sc, bins, cd, cw, controls, complete)
