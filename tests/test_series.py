"""Calendar-aligned series, paired comparisons, slope series and inference on synthetic monthly estimates."""

import numpy as np
import pytest

from disagreement.controls.h2 import INTERACTIONS
from disagreement.inference.fama_macbeth import fama_macbeth, fama_macbeth_wald
from disagreement.inference.monthly import (
    H2_EXTRAS,
    H3_EXTRAS,
    SPECIFICATION_B,
    SPECIFICATION_C,
    SPECIFICATION_C_H2,
    SPECIFICATION_H2,
    SPECIFICATION_H3,
    BinBasis,
    Coefficients,
    MonthlyEstimate,
    estimate_month_specifications,
)
from disagreement.inference.series import (
    LAGS,
    SPECIFICATIONS,
    MonthlySeries,
    align,
    baseline_grid,
    coefficient,
    coefficient_series,
    infer,
    infer_all,
    infer_wald,
    paired_h2,
    paired_h3,
    slope_series,
    standard_series,
)
from disagreement.walkforward.calendar import month_range
from test_monthly_extensions import _month

N_BINS = 20
C, B, CH2, H2, H3 = SPECIFICATION_C, SPECIFICATION_B, SPECIFICATION_C_H2, SPECIFICATION_H2, SPECIFICATION_H3


def _basis(rng):
    return BinBasis(np.linspace(-0.95, 0.95, N_BINS), rng.uniform(0.04, 0.06, N_BINS), rng.uniform(0.3, 0.4, N_BINS))


def _coefficients(rng, spec, interaction, disagreement):
    extra = {}
    if spec == H2:
        extra = {name: float(v) for name, v in zip(H2_EXTRAS, rng.normal(0.0, 0.01, 6))}
    if spec == H3:
        extra = {name: float(v) for name, v in zip(H3_EXTRAS, rng.normal(0.0, 0.01, 2))}
    return Coefficients(
        intercept=rng.normal(0.0, 0.05, N_BINS),
        linear=rng.normal(0.02, 0.01, N_BINS),
        quadratic=np.array([]) if spec == B else rng.normal(0.0, 0.01, N_BINS),
        disagreement=float(disagreement),
        interaction=float(interaction),
        extra=extra,
    )


def _estimate(spec, estimable, basis, coefficients=None, reason="1 cells with fewer than 30 observations"):
    counts = np.full(N_BINS, 45)
    return MonthlyEstimate(spec, estimable, () if estimable else (reason,), int(counts.sum()) if estimable else 0,
                           counts, np.full(N_BINS, 40), 0.3 if estimable else 0.0, 3.5 if estimable else np.inf,
                           basis, coefficients if estimable else None)


def _fixture(rng, grid, masks):
    """Synthetic monthly results on `grid` with one estimability mask per specification; returns them and the truth."""
    basis = _basis(rng)
    truth = {spec: {"interaction": rng.normal(-0.01, 0.01, len(grid)), "disagreement": rng.normal(0, 0.01, len(grid))}
             for spec in SPECIFICATIONS}
    monthly, coefficients = {}, {spec: [] for spec in SPECIFICATIONS}
    for i, month in enumerate(grid):
        results = {}
        for spec in SPECIFICATIONS:
            c = _coefficients(rng, spec, truth[spec]["interaction"][i], truth[spec]["disagreement"][i])
            coefficients[spec].append(c)
            results[spec] = _estimate(spec, bool(masks[spec][i]), basis, c)
        monthly[month] = results
    return monthly, truth, coefficients, basis


def _masks(rng, n, gaps):
    """Different masks per specification with gaps in the middle of the calendar."""
    masks = {}
    for spec, (lo, hi) in zip(SPECIFICATIONS, gaps):
        m = rng.random(n) > 0.1
        m[lo:hi] = False
        masks[spec] = m
    return masks


GAPS = [(40, 46), (41, 44), (60, 70), (62, 75), (20, 30)]  # C, B, C|H2, H2, H3


def test_calendar_grid_baseline_and_short_grid_across_a_year_end():
    grid = baseline_grid()
    assert len(grid) == 480 and grid[0] == 198501 and grid[-1] == 202412 and grid == month_range(198501, 202412)
    assert grid[11:13] == [198512, 198601]
    rng = np.random.default_rng(1)
    short = month_range(202410, 202503)  # December to January inside the grid
    assert short == [202410, 202411, 202412, 202501, 202502, 202503]
    masks = {spec: np.ones(6, dtype=bool) for spec in SPECIFICATIONS}
    monthly, truth, _, _ = _fixture(rng, short, masks)
    aligned = align(short, monthly)
    assert aligned.grid.tolist() == short
    reversed_order = dict(reversed(list(monthly.items())))  # input order never matters
    np.testing.assert_array_equal(coefficient_series(align(short, reversed_order), C, "interaction").values,
                                  truth[C]["interaction"])
    with pytest.raises(ValueError, match="consecutive"):
        align([202410, 202412], {202410: monthly[202410], 202412: monthly[202412]})
    with pytest.raises(ValueError, match="strictly increasing"):
        align([202411, 202410], monthly)
    with pytest.raises(ValueError, match="non-empty"):
        align([], {})


def test_malformed_calendar_keys_and_masks_are_rejected_before_any_conversion():
    rng = np.random.default_rng(11)
    grid = month_range(202410, 202503)
    masks = {spec: np.ones(6, dtype=bool) for spec in SPECIFICATIONS}
    monthly, truth, _, _ = _fixture(rng, grid, masks)
    # grid values: floats (even integral or truncatable), Booleans, strings and invalid month numbers
    for bad in ([202410.0, *grid[1:]], [202410.7, *grid[1:]], [True, *grid[1:]], ["202410", *grid[1:]]):
        with pytest.raises(ValueError, match="not an integer yyyymm month"):
            align(bad, monthly)
    for bad in ([202400, *grid[1:]], [*grid[:-1], 202513], [0, *grid[1:]]):
        with pytest.raises(ValueError, match="not a valid yyyymm month"):
            align(bad, monthly)
    # monthly-result keys with the same defects, even when they would compare equal to a grid month
    float_key = {**{m: r for m, r in monthly.items() if m != 202410}, 202410.0: monthly[202410]}
    with pytest.raises(ValueError, match="monthly results: 202410.0 is not an integer"):
        align(grid, float_key)  # type: ignore[arg-type]  # the float key is the defect under test
    with pytest.raises(ValueError, match="monthly results: 202413 is not a valid"):
        align(grid, list(monthly.items()) + [(202413, monthly[202412])])
    string_key = [(m if m != 202411 else "202411", r) for m, r in monthly.items()]
    with pytest.raises(ValueError, match="monthly results: '202411' is not an integer"):
        align(grid, string_key)  # type: ignore[arg-type]  # the string key is the defect under test
    # NumPy integer keys are genuine integers and behave exactly as before
    numpy_keys = {np.int64(m): r for m, r in monthly.items()}
    aligned = align(np.asarray(grid, dtype="int64"), numpy_keys)  # type: ignore[arg-type]
    np.testing.assert_array_equal(aligned.grid, grid)
    series = coefficient_series(align(grid, monthly), C, "interaction")
    np.testing.assert_array_equal(series.values, truth[C]["interaction"])
    # restricted(): strictly Boolean, one-dimensional, one entry per grid month
    keep = np.array([True, False, True, True, False, True])
    restricted = series.restricted(keep)
    np.testing.assert_array_equal(restricted.estimable, keep)
    np.testing.assert_array_equal(restricted.values[keep], truth[C]["interaction"][keep])
    assert np.isnan(restricted.values[~keep]).all() and restricted.grid.tolist() == grid
    for bad_mask in (keep.astype(int), keep.astype(float), True, np.array(True), keep[:-1],
                     np.ones(7, dtype=bool), keep.reshape(2, 3), [1, 0, 1, 1, 0, 1]):
        with pytest.raises(ValueError, match="strictly Boolean one-dimensional"):
            series.restricted(bad_mask)  # type: ignore[arg-type]  # deliberately invalid masks
    boolean_list = [True, True, False, True, True, True]
    assert series.restricted(boolean_list).n_months == 5  # type: ignore[arg-type]  # a Boolean list is fine


def test_missing_duplicate_unexpected_or_inconsistent_monthly_results_are_rejected():
    rng = np.random.default_rng(2)
    grid = month_range(200001, 200006)
    monthly, _, coefficients, basis = _fixture(rng, grid, {spec: np.ones(6, dtype=bool) for spec in SPECIFICATIONS})
    with pytest.raises(ValueError, match=r"missing \[200003\]"):
        align(grid, {m: r for m, r in monthly.items() if m != 200003})
    with pytest.raises(ValueError, match=r"unexpected \[200007\]"):
        align(grid, {**monthly, 200007: monthly[200006]})
    with pytest.raises(ValueError, match="duplicate months"):
        align(grid, list(monthly.items()) + [(200002, monthly[200002])])
    with pytest.raises(ValueError, match="expected the specifications"):
        align(grid, {**monthly, 200004: {k: v for k, v in monthly[200004].items() if k != B}})
    with pytest.raises(ValueError, match="not a MonthlyEstimate of H2"):
        align(grid, {**monthly, 200004: {**monthly[200004], H2: monthly[200004][H3]}})
    inconsistent = MonthlyEstimate(C, True, (), 900, np.full(N_BINS, 45), np.full(N_BINS, 40), 0.3, 3.5, basis, None)
    with pytest.raises(ValueError, match="estimability and coefficients disagree"):
        align(grid, {**monthly, 200001: {**monthly[200001], C: inconsistent}})


def test_coefficient_extraction_names_zero_values_and_non_estimable_positions():
    rng = np.random.default_rng(3)
    grid = month_range(199901, 200012)  # 24 months
    masks = {spec: np.ones(24, dtype=bool) for spec in SPECIFICATIONS}
    masks[C][[5, 17]] = False
    masks[H2][[5, 6, 7]] = False
    monthly, truth, coefficients, basis = _fixture(rng, grid, masks)
    # an exactly zero coefficient at an estimable month is a value, not a gap
    zero = Coefficients(coefficients[B][3].intercept, coefficients[B][3].linear, np.array([]), 0.0, 0.0, {})
    monthly[grid[3]][B] = _estimate(B, True, basis, zero)
    aligned = align(grid, monthly)
    series = standard_series(aligned)
    assert len(series) == 2 * 5 + 6 + 2 and set(series) >= {"C.interaction", "H2.size_rank", "H3.signal_x_cw"}
    b3 = series["B.interaction"]
    assert b3.values[3] == 0.0 and b3.estimable[3] and b3.n_months == 24 and b3.missing_months == []
    c3 = series["C.interaction"]
    assert c3.grid.tolist() == grid and c3.missing_months == [grid[5], grid[17]] and c3.n_months == 22
    assert np.isnan(c3.values[[5, 17]]).all() and not c3.estimable[[5, 17]].any()
    keep = np.ones(24, dtype=bool)
    keep[[5, 17]] = False
    np.testing.assert_array_equal(c3.values[keep], truth[C]["interaction"][keep])
    np.testing.assert_array_equal(series["H2.disagreement"].values[masks[H2]], truth[H2]["disagreement"][masks[H2]])
    assert series["H2.signal_x_size_rank"].values[0] == coefficients[H2][0].extra["signal_x_size_rank"]
    assert series["H3.cw"].values[1] == coefficients[H3][1].extra["cw"]
    assert aligned.non_estimable(H2) == {grid[5]: ("1 cells with fewer than 30 observations",),
                                         grid[6]: ("1 cells with fewer than 30 observations",),
                                         grid[7]: ("1 cells with fewer than 30 observations",)}
    assert aligned.non_estimable(B) == {}
    # bin-local coefficients by name, for later full reporting
    assert coefficient(coefficients[C][0], "linear:3") == coefficients[C][0].linear[2]
    assert coefficient(coefficients[C][0], "intercept:20") == coefficients[C][0].intercept[19]
    assert coefficient_series(aligned, H3, "quadratic:1").values[2] == coefficients[H3][2].quadratic[0]
    for bad in ("quadratic:1", "size_rank", "linear:0", "linear:21", "slope"):
        with pytest.raises(KeyError):
            coefficient(coefficients[B][0], bad)


def test_paired_h2_differences_use_the_exact_common_months_and_keep_the_unrestricted_h1_series():
    rng = np.random.default_rng(4)
    grid = month_range(198501, 199212)  # 96 months
    masks = _masks(rng, 96, GAPS)
    monthly, truth, _, _ = _fixture(rng, grid, masks)
    aligned = align(grid, monthly)
    pair = paired_h2(aligned)
    common = masks[C] & masks[CH2] & masks[H2]
    assert 0 < common.sum() < min(masks[C].sum(), masks[CH2].sum(), masks[H2].sum())
    np.testing.assert_array_equal(pair.common, common)
    for s in (pair.delta, pair.delta_sample, pair.beta3_c, pair.beta3_c_h2, pair.beta3_h2):
        np.testing.assert_array_equal(s.estimable, common)
        assert np.isnan(s.values[~common]).all() and s.grid.tolist() == grid
    np.testing.assert_allclose(pair.delta.values[common],
                               (truth[H2]["interaction"] - truth[CH2]["interaction"])[common])
    np.testing.assert_allclose(pair.delta_sample.values[common],
                               (truth[CH2]["interaction"] - truth[C]["interaction"])[common])
    # signs: when H2's beta_3 is closer to zero than C|H2's, Delta is positive for a negative C|H2
    i = np.flatnonzero(common)[0]
    assert np.sign(pair.delta.values[i]) == np.sign(truth[H2]["interaction"][i] - truth[CH2]["interaction"][i])
    np.testing.assert_array_equal(pair.beta3_c.values[common], truth[C]["interaction"][common])
    # the paired series are month-by-month differences, never differences of independent means
    unrestricted = standard_series(aligned)["C.interaction"]
    np.testing.assert_array_equal(unrestricted.estimable, masks[C])
    assert unrestricted.n_months > pair.beta3_c.n_months
    full = fama_macbeth(unrestricted.values, unrestricted.estimable, 6)
    restricted = fama_macbeth(pair.beta3_c.values, pair.beta3_c.estimable, 6)
    assert full.mean != restricted.mean and full.n_months == masks[C].sum() and restricted.n_months == common.sum()
    assert fama_macbeth(pair.delta.values, pair.delta.estimable, 6).mean != pytest.approx(
        fama_macbeth(pair.beta3_h2.values, pair.beta3_h2.estimable, 6).mean
        - fama_macbeth(standard_series(aligned)["C|H2.interaction"].values, masks[CH2], 6).mean
    )


def test_paired_h3_common_months_and_wald_input_order():
    rng = np.random.default_rng(5)
    grid = month_range(199001, 199712)  # 96 months
    masks = _masks(rng, 96, GAPS)
    monthly, truth, coefficients, _ = _fixture(rng, grid, masks)
    aligned = align(grid, monthly)
    pair = paired_h3(aligned)
    common = masks[C] & masks[H3]
    np.testing.assert_array_equal(pair.common, common)
    np.testing.assert_array_equal(pair.beta3_c.estimable, common)
    np.testing.assert_array_equal(pair.beta3_h3.values[common], truth[H3]["interaction"][common])
    cw = np.array([c.extra["signal_x_cw"] for c in coefficients[H3]])
    np.testing.assert_array_equal(pair.beta_cw_h3.values[common], cw[common])
    # the Wald input: column 0 = S^c x CD^B, column 1 = S^c x CW, both from the H3 regression, H3 mask only
    assert pair.wald_values.shape == (96, 2)
    np.testing.assert_array_equal(pair.wald_mask, masks[H3])
    np.testing.assert_array_equal(pair.wald_values[masks[H3], 0], truth[H3]["interaction"][masks[H3]])
    np.testing.assert_array_equal(pair.wald_values[masks[H3], 1], cw[masks[H3]])
    assert np.isnan(pair.wald_values[~masks[H3]]).all()
    wald = infer_wald(pair.wald_values, pair.wald_mask)
    assert set(wald.results) == set(LAGS) == {6, 0, 12} and wald.failures == {}
    direct = fama_macbeth_wald(np.column_stack([truth[H3]["interaction"], cw]), masks[H3], 6)
    assert wald.results[6].wald == direct.wald and wald.results[6].n_months == masks[H3].sum()
    assert wald.results[6].n_months > common.sum()  # the Wald test is not restricted to the H1 common months


def test_slope_series_match_independent_references_and_validate_control_means():
    rng = np.random.default_rng(6)
    grid = month_range(200501, 200612)
    masks = {spec: np.ones(24, dtype=bool) for spec in SPECIFICATIONS}
    masks[C][[3]] = False
    masks[H2][[3, 4]] = False
    monthly, _, coefficients, basis = _fixture(rng, grid, masks)
    aligned = align(grid, monthly)
    for spec in (C, H2):
        slopes = slope_series(aligned, spec)
        assert set(slopes) == {0.1, 0.5, 0.9}
        for c, s in slopes.items():
            np.testing.assert_array_equal(s.estimable, masks[spec])
            assert s.name == f"{spec}.g({c})"
            for i in np.flatnonzero(masks[spec]):
                coef = coefficients[spec][i]
                expected = np.mean(coef.linear / basis.half_range) + c * coef.interaction
                assert s.values[i] == pytest.approx(expected, rel=1e-12)
            assert np.isnan(s.values[~masks[spec]]).all()
    # descriptive H2 evaluation at monthly control means: the delta_k x_k term is added, the name differs
    means = {m: rng.uniform(-0.3, 0.3, 3) for m in grid}
    descriptive = slope_series(aligned, H2, controls_at=means)
    primary = slope_series(aligned, H2)
    for i in np.flatnonzero(masks[H2]):
        delta = np.array([coefficients[H2][i].extra[name] for name in INTERACTIONS])
        assert descriptive[0.5].values[i] == pytest.approx(primary[0.5].values[i] + delta @ means[grid[i]], rel=1e-12)
    assert descriptive[0.1].name == "H2.g|controls_at_means(0.1)" != primary[0.1].name
    # means may be absent for non-estimable months but not for estimable ones, and never for months off the grid
    assert slope_series(aligned, H2, controls_at={m: v for m, v in means.items() if m != grid[3]})[0.9].n_months == 22
    with pytest.raises(ValueError, match="every H2-estimable month"):
        slope_series(aligned, H2, controls_at={m: v for m, v in means.items() if m != grid[0]})
    with pytest.raises(ValueError, match="only for grid months"):
        slope_series(aligned, H2, controls_at={**means, 200701: np.zeros(3)})
    with pytest.raises(ValueError, match="3 values"):
        slope_series(aligned, H2, controls_at={**means, grid[1]: np.zeros(2)})
    with pytest.raises(ValueError, match="H2 slopes only"):
        slope_series(aligned, C, controls_at=means)
    for spec in (B, CH2, H3):
        with pytest.raises(ValueError, match="pre-specified"):
            slope_series(aligned, spec)


def test_inference_matches_direct_fama_macbeth_and_reports_failures_explicitly():
    rng = np.random.default_rng(7)
    grid = month_range(198501, 199412)  # 120 months
    masks = _masks(rng, 120, GAPS)
    monthly, truth, _, _ = _fixture(rng, grid, masks)
    aligned = align(grid, monthly)
    series = standard_series(aligned)
    inferred = infer_all(series)
    for spec in SPECIFICATIONS:
        for name in ("interaction", "disagreement"):
            s = series[f"{spec}.{name}"]
            np.testing.assert_array_equal(s.estimable, masks[spec])
            for lag in (6, 0, 12):
                direct = fama_macbeth(truth[spec][name], masks[spec], lag)  # the independent construction
                result = inferred[f"{spec}.{name}"].results[lag]
                assert result == direct and result.n_grid == 120 and result.n_months == masks[spec].sum()
            assert inferred[f"{spec}.{name}"].failures == {}
    # a constant series keeps its values and reports the reason instead of a statistic
    constant = MonthlySeries("x", aligned.grid, np.where(masks[C], 0.25, np.nan), masks[C])  # exactly constant
    out = infer(constant)
    assert out.results == {} and set(out.failures) == {6, 0, 12}
    assert all("HAC variance is zero" in reason for reason in out.failures.values())
    assert out.series is constant and np.nansum(out.series.values) > 0
    # too few estimable months
    one = MonthlySeries("y", aligned.grid, np.where(np.arange(120) == 7, 0.1, np.nan), np.arange(120) == 7)
    assert "at least 2 estimable months" in infer(one).failures[6]
    # a degenerate Wald covariance is reported, not regularised
    collinear = np.column_stack([truth[H3]["interaction"], 2.0 * truth[H3]["interaction"]])
    wald = infer_wald(collinear, masks[H3])
    assert wald.results == {} and all("singular" in r for r in wald.failures.values())
    np.testing.assert_array_equal(wald.values, collinear)


def test_end_to_end_with_the_monthly_engine_on_a_short_grid():
    """Real monthly regressions on four synthetic months across a year end, then the aggregation layer."""
    rng = np.random.default_rng(8)
    grid = month_range(202311, 202402)
    monthly = {}
    for i, month in enumerate(grid):
        sc, bins, cd, cw, controls, complete = _month(rng, n=1500)
        outcome = 0.02 * sc - 0.01 * sc * cd + 0.004 * cd + 0.05 * rng.standard_normal(len(sc))
        outcome[rng.random(len(sc)) < 0.1] = np.nan
        if i == 1:  # the second month has no complete case: H2 and C|H2 are not estimable there
            complete = np.zeros(len(sc), dtype=bool)
        monthly[month] = estimate_month_specifications(outcome, sc, bins, cd, cw, controls, complete)
    aligned = align(grid, monthly)
    assert aligned.mask(C).tolist() == [True] * 4 and aligned.mask(H2).tolist() == [True, False, True, True]
    assert "H2 sample is empty" in aligned.non_estimable(H2)[202312][0]
    series = standard_series(aligned)
    assert series["C.interaction"].n_months == 4 and series["H2.interaction"].missing_months == [202312]
    pair = paired_h2(aligned)
    assert pair.common.tolist() == [True, False, True, True]
    assert pair.delta.values[2] == pytest.approx(
        monthly[202401][H2].coefficients.interaction - monthly[202401][CH2].coefficients.interaction)
    slopes = slope_series(aligned, C)
    est = monthly[202402][C]
    expected = np.mean(est.coefficients.linear / est.basis.half_range) + 0.9 * est.coefficients.interaction
    assert slopes[0.9].values[3] == pytest.approx(expected, rel=1e-12)
    wald = infer_wald(*(lambda p: (p.wald_values, p.wald_mask))(paired_h3(aligned)), lags=(0,))
    assert wald.results[0].n_months == 4 and wald.results[0].df == 2
    inferred = infer(series["B.interaction"], lags=(0,))
    assert inferred.results[0].n_months == 4 and inferred.results[0].n_grid == 4
