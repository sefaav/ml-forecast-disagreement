"""Frozen decision rules and structured reporting on synthetic inference objects. No real data is used."""

import math

import numpy as np
import pytest

from disagreement.inference.fama_macbeth import FamaMacBethResult, WaldResult, fama_macbeth
from disagreement.inference.monthly import (
    SPECIFICATION_B,
    SPECIFICATION_C,
    SPECIFICATION_H2,
    SPECIFICATION_H3,
    MonthlyEstimate,
)
from disagreement.inference.report import (
    ALPHA,
    DELTA_AMPLIFICATION,
    DELTA_ATTENUATION,
    DELTA_NEUTRAL,
    DELTA_NO_CHANGE,
    ECONOMIC_NOT_CLAIMED,
    ECONOMIC_NOT_ESTABLISHED,
    ECONOMIC_SUPPORTED,
    H1_INCONCLUSIVE,
    H1_OPPOSITE,
    H1_SUPPORTED,
    H2_DESCRIPTIVE,
    H2_DOES_NOT_SURVIVE,
    H2_SURVIVES,
    H3_DISTINCT,
    H3_NOT_DISTINCT,
    NOT_ASSESSABLE,
    PRIMARY_LAG,
    Report,
    Verdict,
    build_report,
    delta_interpretation,
    economic_interpretation,
    h1_verdict,
    h2_verdict,
    h3_verdict,
    primary,
)
from disagreement.inference.series import SPECIFICATIONS, Inference, MonthlySeries, align, infer
from disagreement.walkforward.calendar import month_range
from test_series import GAPS, _fixture, _masks


def _fmr(mean, se, p=None, lag=6, n=400):
    """A FamaMacBethResult with the frozen normal inference, or an imposed p-value for boundary cases."""
    t = mean / se
    p_value = math.erfc(abs(t) / math.sqrt(2.0)) if p is None else p
    return FamaMacBethResult(mean, se, t, p_value, mean - 1.96 * se, mean + 1.96 * se, n, 480, lag, se * se * n)


def _wald(p):
    return WaldResult(np.array([0.0, 0.0]), np.eye(2), -2.0 * math.log(p), 2, p, 400, 480, 6)


SIG_NEG = _fmr(-0.012, 0.003)  # t = -4
SIG_POS = _fmr(0.012, 0.003)
WEAK_NEG = _fmr(-0.002, 0.003)  # t = -0.67
WEAK_POS = _fmr(0.002, 0.003)


def test_h1_verdicts_including_the_exact_boundary_and_unavailable_inference():
    assert h1_verdict(SIG_NEG).verdict == H1_SUPPORTED
    assert h1_verdict(SIG_POS).verdict == H1_OPPOSITE
    assert h1_verdict(WEAK_NEG).verdict == H1_INCONCLUSIVE and h1_verdict(WEAK_POS).verdict == H1_INCONCLUSIVE
    none = h1_verdict(None, "the HAC variance is zero")
    assert none.verdict == NOT_ASSESSABLE and none.conditions == {"negative": None, "significant": None}
    assert "HAC variance is zero" in none.notes[0] and none.basis == {"C.interaction": None}
    # exactly p = 0.05 is not significant; just below is
    assert h1_verdict(_fmr(-0.01, 0.005, p=0.05)).verdict == H1_INCONCLUSIVE
    assert h1_verdict(_fmr(-0.01, 0.005, p=0.05 - 1e-12)).verdict == H1_SUPPORTED
    assert h1_verdict(_fmr(0.01, 0.005, p=0.05)).verdict == H1_INCONCLUSIVE
    # two-sided normal: |t| = 1.95 gives p = 0.0512 (inconclusive), |t| = 1.96 gives p = 0.04996 (significant)
    assert h1_verdict(_fmr(-1.95, 1.0)).verdict == H1_INCONCLUSIVE
    assert h1_verdict(_fmr(-1.96, 1.0)).verdict == H1_SUPPORTED
    assert h1_verdict(_fmr(1.96, 1.0)).verdict == H1_OPPOSITE
    v = h1_verdict(SIG_NEG)
    assert v.conditions == {"negative": True, "significant": True} and v.basis["C.interaction"] is SIG_NEG
    assert ALPHA == 0.05 and PRIMARY_LAG == 6


def test_h2_gate_open_and_closed_with_descriptive_reporting():
    supported, inconclusive, opposite = h1_verdict(SIG_NEG), h1_verdict(WEAK_NEG), h1_verdict(SIG_POS)
    assert h2_verdict(supported, SIG_NEG).verdict == H2_SURVIVES
    assert h2_verdict(supported, WEAK_NEG).verdict == H2_DOES_NOT_SURVIVE
    assert h2_verdict(supported, SIG_POS).verdict == H2_DOES_NOT_SURVIVE  # significant but positive
    assert h2_verdict(supported, _fmr(-0.01, 0.005, p=0.05)).verdict == H2_DOES_NOT_SURVIVE
    unavailable = h2_verdict(supported, None, "at least 2 estimable months are needed, 1 given")
    assert unavailable.verdict == NOT_ASSESSABLE and "estimable months" in unavailable.notes[0]
    # gate closed: a significant negative H2 is still reported, with its conditions, but labelled descriptive
    for h1 in (inconclusive, opposite, h1_verdict(None, "x")):
        closed = h2_verdict(h1, SIG_NEG)
        assert closed.verdict == H2_DESCRIPTIVE and closed.verdict not in (H2_SURVIVES, H2_DOES_NOT_SURVIVE)
        assert closed.conditions == {"gate_open": False, "negative": True, "significant": True}
        assert closed.basis["H2.interaction"] is SIG_NEG and "descriptive" in closed.notes[0]
    closed_none = h2_verdict(inconclusive, None, "singular")
    assert closed_none.verdict == H2_DESCRIPTIVE and closed_none.conditions["negative"] is None
    assert "unavailable: singular" in closed_none.notes[0]


def test_delta_interpretation_conditional_on_a_negative_paired_baseline():
    baseline = _fmr(-0.010, 0.002)
    attenuation = _fmr(0.004, 0.001)  # interval (0.00204, 0.00596) excludes zero, positive
    amplification = _fmr(-0.004, 0.001)
    no_change = _fmr(0.001, 0.001)  # interval (-0.00096, 0.00296) contains zero
    assert delta_interpretation(baseline, attenuation).verdict == DELTA_ATTENUATION
    assert delta_interpretation(baseline, amplification).verdict == DELTA_AMPLIFICATION
    v = delta_interpretation(baseline, no_change)
    assert v.verdict == DELTA_NO_CHANGE and "not absence of attenuation" in v.notes[0]
    assert v.conditions == {"baseline_negative": True, "interval_excludes_zero": False}
    # the decision uses the reported 95% interval, not the p-value
    edge = FamaMacBethResult(0.002, 0.001, 2.0, 0.0455, 0.0, 0.004, 400, 480, 6, 0.4)  # ci_low exactly 0
    assert delta_interpretation(baseline, edge).verdict == DELTA_NO_CHANGE
    # a non-negative baseline is reported neutrally, whatever Delta does
    for base in (_fmr(0.010, 0.002), _fmr(0.0, 0.002)):
        neutral = delta_interpretation(base, attenuation)
        assert neutral.verdict == DELTA_NEUTRAL and neutral.conditions["baseline_negative"] is False
    # unavailable paired inference: no attenuation claim
    missing = delta_interpretation(baseline, None, {"Delta": "the HAC variance is zero"})
    assert missing.verdict == NOT_ASSESSABLE and "Delta: the HAC variance is zero" in missing.notes[0]
    assert missing.conditions == {"baseline_negative": True, "interval_excludes_zero": None}
    assert delta_interpretation(None, attenuation).verdict == NOT_ASSESSABLE


def test_economic_interpretation_requires_h1_support_and_a_positive_low_disagreement_slope():
    supported = h1_verdict(SIG_NEG)
    g_pos, g_neg, g_weak = _fmr(0.03, 0.01), _fmr(-0.03, 0.01), _fmr(0.01, 0.01)
    v = economic_interpretation(supported, g_pos, g_weak)
    assert v.verdict == ECONOMIC_SUPPORTED and v.conditions["sign_reversal"] is False
    assert economic_interpretation(supported, g_weak, g_pos).verdict == ECONOMIC_NOT_ESTABLISHED
    assert economic_interpretation(supported, g_neg, g_neg).verdict == ECONOMIC_NOT_ESTABLISHED  # negative g(0.1)
    weak = economic_interpretation(supported, g_weak, g_neg)
    assert "not evidence that the signal is uninformative" in weak.notes[0] and weak.conditions["sign_reversal"] is None
    # sign reversal only inside the supported reading, and only with a significantly negative g(0.9)
    reversal = economic_interpretation(supported, g_pos, g_neg)
    assert reversal.verdict == ECONOMIC_SUPPORTED and reversal.conditions["sign_reversal"] is True
    assert any("sign reversal" in n for n in reversal.notes)
    assert economic_interpretation(supported, g_pos, _fmr(-0.01, 0.01)).conditions["sign_reversal"] is False
    assert economic_interpretation(supported, g_weak, g_neg).conditions["sign_reversal"] is None
    # H1 not supported: not claimed, even with a strongly positive g(0.1) and a negative g(0.9)
    for h1 in (h1_verdict(WEAK_NEG), h1_verdict(SIG_POS)):
        v = economic_interpretation(h1, g_pos, g_neg)
        assert v.verdict == ECONOMIC_NOT_CLAIMED and v.conditions["h1_supported"] is False
        assert v.conditions["low_disagreement_slope_positive_significant"] is True


def test_economic_reading_distinguishes_unavailable_inference_from_negative_findings():
    supported = h1_verdict(SIG_NEG)
    g_pos, g_neg, g_weak = _fmr(0.03, 0.01), _fmr(-0.03, 0.01), _fmr(0.01, 0.01)
    # H1 supported, g(0.1) unavailable: not assessable, never an observed non-significant slope
    v = economic_interpretation(supported, None, g_neg, {"C.g(0.1)": "zero variance"})
    assert v.verdict == NOT_ASSESSABLE and v.verdict != ECONOMIC_NOT_ESTABLISHED
    assert v.conditions == {"h1_supported": True, "low_disagreement_slope_positive_significant": None,
                            "sign_reversal": None}
    assert v.notes == ("g(0.1) inference unavailable: zero variance",) and v.basis["C.g(0.1)"] is None
    # H1 itself unavailable: distinguished from an assessed but unsupported H1
    missing_h1 = h1_verdict(None, "the HAC variance is zero")
    v = economic_interpretation(missing_h1, g_pos, g_neg)
    assert v.verdict == NOT_ASSESSABLE and v.conditions["h1_supported"] is None
    assert v.notes[0].startswith("H1 is not assessable: primary H1 inference unavailable: the HAC variance is zero")
    assert v.conditions["low_disagreement_slope_positive_significant"] is True  # the slope itself was evaluable
    assessed = economic_interpretation(h1_verdict(WEAK_NEG), g_pos, g_neg)
    assert assessed.verdict == ECONOMIC_NOT_CLAIMED and assessed.conditions["h1_supported"] is False
    # an available, non-significant g(0.1) keeps the frozen wording
    v = economic_interpretation(supported, g_weak, g_neg)
    assert v.verdict == ECONOMIC_NOT_ESTABLISHED
    assert v.conditions["low_disagreement_slope_positive_significant"] is False
    assert "not evidence that the signal is uninformative" in v.notes[0]
    # g(0.9) unavailable: the supported reading stands, the sign reversal is unassessable with its reason
    v = economic_interpretation(supported, g_pos, None, {"C.g(0.9)": "at least 2 estimable months are needed"})
    assert v.verdict == ECONOMIC_SUPPORTED and v.conditions["sign_reversal"] is None
    assert v.notes == ("g(0.9) inference unavailable: at least 2 estimable months are needed",)
    assert economic_interpretation(supported, g_pos, g_weak).conditions["sign_reversal"] is False


def test_sample_sizes_are_preserved_for_non_estimable_regressions():
    rng = np.random.default_rng(101)
    grid = month_range(201001, 201006)
    masks = {spec: np.ones(6, dtype=bool) for spec in SPECIFICATIONS}
    monthly, _, coefficients, basis = _fixture(rng, grid, masks)
    counts = np.full(20, 35)
    counts[4] = 12  # a cell below the minimum: 19 x 35 + 12 = 677 observations, not estimable
    failed = MonthlyEstimate(SPECIFICATION_H2, False, ("1 cells with fewer than 30 observations (minimum 12)",),
                             int(counts.sum()), counts, np.full(20, 10), 0.3, 3.5, basis, None)
    empty = MonthlyEstimate(SPECIFICATION_H2, False, ("no member of U_t has a finite outcome and complete "
                                                      "controls: the H2 sample is empty",),
                            0, np.zeros(20, dtype="int64"), np.zeros(20, dtype="int64"), 0.0, np.inf, basis, None)
    monthly[grid[1]] = {**monthly[grid[1]], SPECIFICATION_H2: failed}
    monthly[grid[4]] = {**monthly[grid[4]], SPECIFICATION_H2: empty}
    report = build_report(align(grid, monthly))
    assert report.sample_sizes[SPECIFICATION_H2].tolist() == [900, 677, 900, 900, 0, 900]
    assert set(report.non_estimable[SPECIFICATION_H2]) == {grid[1], grid[4]}
    assert report.non_estimable[SPECIFICATION_H2][grid[1]] == failed.reasons
    assert report.coefficients["H2.interaction"].series.missing_months == [grid[1], grid[4]]
    assert report.sample_sizes[SPECIFICATION_C].tolist() == [900] * 6


def test_h3_intersection_union_rule_and_the_complementary_wald_test():
    assert h3_verdict(SIG_NEG, SIG_POS).verdict == H3_DISTINCT  # signs are not required
    assert h3_verdict(SIG_NEG, SIG_NEG, _wald(0.001)).verdict == H3_DISTINCT
    assert h3_verdict(SIG_NEG, WEAK_POS).verdict == H3_NOT_DISTINCT
    assert h3_verdict(WEAK_NEG, SIG_POS).verdict == H3_NOT_DISTINCT
    assert h3_verdict(WEAK_NEG, WEAK_POS).verdict == H3_NOT_DISTINCT
    assert h3_verdict(SIG_NEG, _fmr(0.01, 0.005, p=0.05)).verdict == H3_NOT_DISTINCT  # boundary
    # a significant joint Wald test with only one significant interaction does not establish distinct information
    one = h3_verdict(SIG_NEG, WEAK_POS, _wald(1e-6))
    assert one.verdict == H3_NOT_DISTINCT and one.conditions["wald_significant"] is True
    assert one.conditions == {"cd_interaction_significant": True, "cw_interaction_significant": False,
                              "wald_significant": True}
    # a non-significant Wald test does not remove distinct information established individually
    assert h3_verdict(SIG_NEG, SIG_POS, _wald(0.5)).verdict == H3_DISTINCT
    # unavailable individual inference: not assessable; an unavailable Wald test leaves the individual verdict
    v = h3_verdict(None, SIG_POS, _wald(0.01), {"H3.interaction": "singular"})
    assert v.verdict == NOT_ASSESSABLE and any("H3.interaction inference unavailable: singular" in n for n in v.notes)
    assert h3_verdict(SIG_NEG, None).verdict == NOT_ASSESSABLE
    v = h3_verdict(SIG_NEG, SIG_POS, None, {"H3.wald": "the HAC covariance is singular"})
    assert v.verdict == H3_DISTINCT and v.conditions["wald_significant"] is None
    assert any("Wald test unavailable: the HAC covariance is singular" in n for n in v.notes)
    assert v.basis["H3.wald"] is None and v.basis["H3.signal_x_cw"] is SIG_POS


def test_primary_lag_only_decides_even_when_sensitivity_lags_disagree():
    grid = np.arange(200001, 200013)
    series = MonthlySeries("C.interaction", grid, np.zeros(12), np.ones(12, dtype=bool))
    opposite = {6: SIG_NEG, 0: _fmr(0.02, 0.001, lag=0), 12: _fmr(0.02, 0.001, lag=12)}
    result, failure = primary(Inference(series, opposite, {}))
    assert result is SIG_NEG and failure is None and h1_verdict(result).verdict == H1_SUPPORTED
    # L = 6 missing although the sensitivity lags are available: the verdict is not assessable
    result, failure = primary(Inference(series, {0: SIG_NEG, 12: SIG_NEG}, {6: "the HAC variance is zero"}))
    assert result is None and failure == "the HAC variance is zero"
    assert h1_verdict(result, failure).verdict == NOT_ASSESSABLE
    assert primary(None) == (None, "inference not computed")


def test_end_to_end_report_from_aligned_synthetic_estimates():
    rng = np.random.default_rng(99)
    grid = month_range(198501, 199412)  # 120 months
    masks = _masks(rng, 120, GAPS)
    monthly, truth, coefficients, basis = _fixture(rng, grid, masks)
    aligned = align(grid, monthly)
    means = {m: rng.uniform(-0.2, 0.2, 3) for m in grid}
    report = build_report(aligned, h2_control_means=means)
    assert isinstance(report, Report) and report.grid.tolist() == grid
    # verdicts come from the L = 6 unrestricted series, exactly as a direct call would give
    direct = fama_macbeth(truth[SPECIFICATION_C]["interaction"], masks[SPECIFICATION_C], 6)
    assert report.h1.basis["C.interaction"] == direct
    assert report.h1.verdict == h1_verdict(direct).verdict
    assert report.coefficients["C.interaction"].results[6] == direct
    assert set(report.coefficients["C.interaction"].results) == {6, 0, 12}
    assert report.h2.basis["H2.interaction"] == report.coefficients["H2.interaction"].results[6]
    assert report.h2.verdict == h2_verdict(report.h1, report.coefficients["H2.interaction"].results[6]).verdict
    # the gate uses the unrestricted H1 series, not the paired one, and both are reported
    paired_c = report.paired_h2["C.interaction|paired"].results[6]
    assert paired_c.n_months < direct.n_months == masks[SPECIFICATION_C].sum()
    common = masks[SPECIFICATION_C] & masks["C|H2"] & masks[SPECIFICATION_H2]
    for name in ("Delta", "Delta_S", "C.interaction|paired", "C|H2.interaction|paired", "H2.interaction|paired"):
        assert report.paired_h2[name].results[6].n_months == common.sum()
    assert report.delta.basis["Delta"] == report.paired_h2["Delta"].results[6]
    assert report.delta.basis["C|H2.interaction|paired"] == report.paired_h2["C|H2.interaction|paired"].results[6]
    # H3: individual tests and the Wald complement on the H3 mask; paired H3 series on C and H3 common months
    assert report.h3.basis["H3.interaction"] == report.coefficients["H3.interaction"].results[6]
    assert report.h3.basis["H3.wald"] == report.wald.results[6]
    assert report.wald.results[6].n_months == masks[SPECIFICATION_H3].sum()
    assert report.paired_h3["C.interaction|H3-paired"].results[6].n_months == (
        masks[SPECIFICATION_C] & masks[SPECIFICATION_H3]).sum()
    # slopes: primary C and H2, descriptive H2 kept separate; the economic reading uses C.g(0.1) and C.g(0.9)
    assert {"C.g(0.1)", "C.g(0.5)", "C.g(0.9)", "H2.g(0.1)", "H2.g|controls_at_means(0.1)"} <= set(report.slopes)
    assert report.economic.basis["C.g(0.1)"] == report.slopes["C.g(0.1)"].results[6]
    assert report.economic.verdict in (ECONOMIC_SUPPORTED, ECONOMIC_NOT_ESTABLISHED, ECONOMIC_NOT_CLAIMED)
    assert report.slopes["H2.g|controls_at_means(0.5)"].results[6].mean != report.slopes["H2.g(0.5)"].results[6].mean
    # coverage of the statistical tables
    assert len(report.coefficients) == 18
    assert len(report.bin_local) == 20 * 3 * 5 - 20  # B has no quadratic terms
    assert report.bin_local["C.linear:3"].results[6].mean == pytest.approx(
        np.mean([c.linear[2] for c, m in zip(coefficients[SPECIFICATION_C], masks[SPECIFICATION_C]) if m]))
    for inf in list(report.coefficients.values()) + list(report.slopes.values()):
        for lag, r in inf.results.items():
            assert r.lag == lag and r.n_grid == 120 and r.n_months == inf.series.n_months
            assert r.ci_low == pytest.approx(r.mean - 1.96 * r.se) and 0.0 <= r.p_value <= 1.0
    # non-estimable months with their reasons, sample sizes, and no failure on this well-behaved fixture
    for spec in SPECIFICATIONS:
        assert set(report.non_estimable[spec]) == {grid[i] for i in np.flatnonzero(~masks[spec])}
        assert all(r == ("1 cells with fewer than 30 observations",) for r in report.non_estimable[spec].values())
        np.testing.assert_array_equal(report.sample_sizes[spec] > 0, masks[spec])
        assert report.sample_sizes[spec][masks[spec]].min() == 900
    assert report.failures == {}
    with pytest.raises(ValueError, match="primary lag"):
        build_report(aligned, lags=(0, 12))


def test_report_records_inference_failures_instead_of_statistics():
    rng = np.random.default_rng(100)
    grid = month_range(200001, 200112)
    masks = {spec: np.ones(24, dtype=bool) for spec in SPECIFICATIONS}
    masks[SPECIFICATION_H3][:] = False  # H3 never estimable: its series, Wald and paired H3 all fail explicitly
    masks[SPECIFICATION_H3][[3, 9]] = True  # ... except two months: enough for a scalar mean, not for the Wald
    monthly, truth, coefficients, basis = _fixture(rng, grid, masks)
    aligned = align(grid, monthly)
    report = build_report(aligned)
    assert report.h3.verdict in (H3_DISTINCT, H3_NOT_DISTINCT)  # two months give valid scalar tests
    assert report.wald.results == {} and all("singular" in r for r in report.wald.failures.values())
    assert report.h3.conditions["wald_significant"] is None and report.h3.basis["H3.wald"] is None
    assert set(report.failures["H3.wald"]) == {6, 0, 12}
    assert report.non_estimable[SPECIFICATION_H3] == {m: ("1 cells with fewer than 30 observations",)
                                                      for m in grid if m not in (grid[3], grid[9])}
    # a specification with a single estimable month: scalar inference unavailable, reported as such
    masks[SPECIFICATION_B][:] = False
    masks[SPECIFICATION_B][5] = True
    monthly, *_ = _fixture(rng, grid, masks)
    report = build_report(align(grid, monthly))
    assert report.coefficients["B.interaction"].results == {}
    assert "at least 2 estimable months" in report.failures["B.interaction"][6]
    assert report.coefficients["B.interaction"].series.n_months == 1  # the series itself is preserved
    assert report.h1.verdict != NOT_ASSESSABLE  # B's failure does not touch the C-based H1 verdict
    assert isinstance(report.h1, Verdict) and infer(report.coefficients["B.interaction"].series).results == {}
