"""Frozen scientific verdicts and structured reporting from the calendar-aligned inference.

Pre-specified in docs/pre_analysis_plan.md ("Reporting rules for H1 and H2",
"Fama-MacBeth inference", "H3") and docs/hypotheses.md (verdicts of H1, H2
and H3). Every scientific decision uses the primary Bartlett HAC inference at
L = 6, two-sided at the 5% level with a strict p < 0.05; L = 0 and L = 12 are
reported as sensitivity analyses and never enter a verdict.

* H1 (unrestricted specification C, beta_3): "supported" if the mean is
  negative with p < 0.05, "evidence in the opposite direction" if positive
  with p < 0.05, "inconclusive" otherwise; "not assessable" when the primary
  inference could not be computed.
* H2 (fixed sequence): confirmatory only when H1 is supported; then
  "survives" if H2's beta_3 is negative with p < 0.05, "does not survive"
  otherwise, "not assessable" without inference. Otherwise H2 is reported in
  full and labelled descriptive, with no confirmatory verdict.
* Paired differences on the common months of C, C|H2 and H2: when the paired
  mean of beta_3(C|H2) is negative, a Delta interval excluding zero reads
  "attenuation towards zero" (Delta > 0) or "amplification" (Delta < 0), an
  interval containing zero "no detected change"; a non-negative baseline is
  reported neutrally. Delta^S is descriptive.
* Economic reading of H1: claimed only if H1 is supported and the primary
  slope g(0.1) is positive with p < 0.05; otherwise "informativeness not
  established at this precision". A sign reversal is reported inside that
  reading only if g(0.9) is negative with p < 0.05.
* H3: "distinct information" only if both interactions of the same H3
  regression reject individually (intersection-union, no correction); the
  joint Wald test is a complement that never replaces the two individual
  tests.

An unavailable inference is always reported as such, never as a
non-significant result. Nothing is estimated or recomputed here, no file is
written, and the numbers of the inference engines are passed on unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from disagreement.inference.fama_macbeth import HAC_LAG, FamaMacBethResult, WaldResult
from disagreement.inference.monthly import (
    SPECIFICATION_B,
    SPECIFICATION_C,
    SPECIFICATION_H2,
    SPECIFICATION_H3,
)
from disagreement.inference.series import (
    CW_INTERACTION,
    INTERACTION,
    LAGS,
    SPECIFICATIONS,
    AlignedEstimates,
    Inference,
    WaldInference,
    coefficient_series,
    infer,
    infer_all,
    infer_wald,
    paired_h2,
    paired_h3,
    slope_series,
    standard_series,
)

ALPHA = 0.05  # two-sided, strict p < ALPHA
PRIMARY_LAG = HAC_LAG  # verdicts use L = 6 only

H1_SUPPORTED = "supported"
H1_OPPOSITE = "evidence in the opposite direction"
H1_INCONCLUSIVE = "inconclusive"
NOT_ASSESSABLE = "not assessable"
H2_SURVIVES = "survives"
H2_DOES_NOT_SURVIVE = "does not survive"
H2_DESCRIPTIVE = "descriptive (non-confirmatory: the H1 gate is not passed)"
DELTA_ATTENUATION = "attenuation towards zero"
DELTA_AMPLIFICATION = "amplification"
DELTA_NO_CHANGE = "no detected change"
DELTA_NEUTRAL = "difference reported neutrally (the paired mean of beta_3(C|H2) is not negative)"
ECONOMIC_SUPPORTED = "a positively informative signal becomes less reliable where disagreement is higher"
ECONOMIC_NOT_ESTABLISHED = "informativeness not established at this precision"
ECONOMIC_NOT_CLAIMED = "not claimed (H1 interaction not supported)"
H3_DISTINCT = "distinct information"
H3_NOT_DISTINCT = "distinct information not established"
LOW, MID, HIGH = 0.1, 0.5, 0.9  # the CD^B levels of the marginal slopes
BIN_COMPONENTS = ("intercept", "linear", "quadratic")


@dataclass(frozen=True)
class Verdict:
    """A frozen decision, the named conditions it rests on (None: not evaluable) and the statistics it used."""

    hypothesis: str
    verdict: str
    conditions: dict[str, bool | None]
    basis: dict[str, FamaMacBethResult | WaldResult | None]
    notes: tuple[str, ...]


def significant(result: FamaMacBethResult) -> bool:
    return result.p_value < ALPHA


def primary(inference: Inference | None) -> tuple[FamaMacBethResult | None, str | None]:
    """The L = 6 result of an inference, or the reason it is missing."""
    if inference is None:
        return None, "inference not computed"
    if PRIMARY_LAG in inference.results:
        return inference.results[PRIMARY_LAG], None
    return None, inference.failures.get(PRIMARY_LAG, f"no result at L = {PRIMARY_LAG}")


# ----------------------------------------------------------------------------------------------------------------
# Decision rules


def h1_verdict(beta3: FamaMacBethResult | None, failure: str | None = None) -> Verdict:
    """H1 from the unrestricted specification C beta_3 at L = 6."""
    if beta3 is None:
        return Verdict("H1", NOT_ASSESSABLE, {"negative": None, "significant": None}, {"C.interaction": None},
                       (f"primary H1 inference unavailable: {failure}",))
    negative, sig = beta3.mean < 0.0, significant(beta3)
    verdict = H1_SUPPORTED if negative and sig else H1_OPPOSITE if beta3.mean > 0.0 and sig else H1_INCONCLUSIVE
    return Verdict("H1", verdict, {"negative": negative, "significant": sig}, {"C.interaction": beta3},
                   ("statistical verdict on the S^c x CD^B interaction, not on the signal's informativeness",))


def h2_verdict(h1: Verdict, beta3_h2: FamaMacBethResult | None, failure: str | None = None) -> Verdict:
    """H2 behind the fixed-sequence gate: confirmatory only when H1 is supported."""
    basis: dict[str, FamaMacBethResult | WaldResult | None] = {"H2.interaction": beta3_h2}
    gate = h1.verdict == H1_SUPPORTED
    if not gate:
        note = "H2 statistics reported in full but descriptive: H1 is not supported in the negative direction"
        if beta3_h2 is None:
            note += f"; H2 inference unavailable: {failure}"
        conditions: dict[str, bool | None] = {"gate_open": False, "negative": None, "significant": None}
        if beta3_h2 is not None:
            conditions |= {"negative": beta3_h2.mean < 0.0, "significant": significant(beta3_h2)}
        return Verdict("H2", H2_DESCRIPTIVE, conditions, basis, (note,))
    if beta3_h2 is None:
        return Verdict("H2", NOT_ASSESSABLE, {"gate_open": True, "negative": None, "significant": None}, basis,
                       (f"H1 supported but the primary H2 inference is unavailable: {failure}",))
    negative, sig = beta3_h2.mean < 0.0, significant(beta3_h2)
    return Verdict("H2", H2_SURVIVES if negative and sig else H2_DOES_NOT_SURVIVE,
                   {"gate_open": True, "negative": negative, "significant": sig}, basis,
                   ("confirmatory: H1 supported, fixed sequence at 5% without correction",))


def delta_interpretation(
    baseline_c_h2: FamaMacBethResult | None, delta: FamaMacBethResult | None, failures: Mapping[str, str] = {}
) -> Verdict:
    """Delta = beta_3(H2) - beta_3(C|H2) on the paired months, read only when the paired C|H2 mean is negative."""
    basis: dict[str, FamaMacBethResult | WaldResult | None] = {"C|H2.interaction|paired": baseline_c_h2,
                                                                 "Delta": delta}
    if baseline_c_h2 is None or delta is None:
        missing = [f"{k}: {v}" for k, v in failures.items()]
        return Verdict("Delta", NOT_ASSESSABLE,
                       {"baseline_negative": None if baseline_c_h2 is None else baseline_c_h2.mean < 0.0,
                        "interval_excludes_zero": None}, basis,
                       ("paired inference unavailable: " + ("; ".join(missing) or "no result at L = 6"),))
    baseline_negative = baseline_c_h2.mean < 0.0
    excludes = delta.ci_low > 0.0 or delta.ci_high < 0.0
    conditions: dict[str, bool | None] = {"baseline_negative": baseline_negative, "interval_excludes_zero": excludes}
    if not baseline_negative:
        return Verdict("Delta", DELTA_NEUTRAL, conditions, basis,
                       ("the attenuation / amplification reading applies only to a negative paired baseline",))
    reading = DELTA_NO_CHANGE if not excludes else DELTA_ATTENUATION if delta.mean > 0.0 else DELTA_AMPLIFICATION
    note = ("controls absorb part of the covariation; they do not explain it; the interval is reported with "
            "the reading" + (", and 'no detected change' is not absence of attenuation" if not excludes else ""))
    return Verdict("Delta", reading, conditions, basis, (note,))


def economic_interpretation(
    h1: Verdict, g_low: FamaMacBethResult | None, g_high: FamaMacBethResult | None, failures: Mapping[str, str] = {}
) -> Verdict:
    """The two statements of "Reporting rules": the economic reading needs H1 supported and g(0.1) > 0 with p < 0.05."""
    basis: dict[str, FamaMacBethResult | WaldResult | None] = {"C.g(0.1)": g_low, "C.g(0.9)": g_high}
    conditions: dict[str, bool | None] = {"h1_supported": h1.verdict == H1_SUPPORTED,
                                          "low_disagreement_slope_positive_significant": None,
                                          "sign_reversal": None}
    notes: list[str] = []
    if g_low is not None:
        conditions["low_disagreement_slope_positive_significant"] = g_low.mean > 0.0 and significant(g_low)
    else:
        notes.append(f"g(0.1) inference unavailable: {failures.get('C.g(0.1)', 'no result at L = 6')}")
    if h1.verdict == NOT_ASSESSABLE:  # no H1 result at all: nothing to read, as opposed to an unsupported H1
        conditions["h1_supported"] = None
        notes.insert(0, "H1 is not assessable: " + "; ".join(h1.notes))
        return Verdict("H1 economic reading", NOT_ASSESSABLE, conditions, basis, tuple(notes))
    if h1.verdict != H1_SUPPORTED:
        return Verdict("H1 economic reading", ECONOMIC_NOT_CLAIMED, conditions, basis, tuple(notes))
    if g_low is None:  # H1 supported but the slope condition cannot be checked: not an observed failure of it
        return Verdict("H1 economic reading", NOT_ASSESSABLE, conditions, basis, tuple(notes))
    if not conditions["low_disagreement_slope_positive_significant"]:
        notes.append("g is a local within-bin slope estimated with far less precision than across-bin contrasts; "
                     "this is not evidence that the signal is uninformative")
        return Verdict("H1 economic reading", ECONOMIC_NOT_ESTABLISHED, conditions, basis, tuple(notes))
    if g_high is None:
        notes.append(f"g(0.9) inference unavailable: {failures.get('C.g(0.9)', 'no result at L = 6')}")
    else:
        conditions["sign_reversal"] = g_high.mean < 0.0 and significant(g_high)
        if conditions["sign_reversal"]:
            notes.append("sign reversal: g(0.9) is significantly negative")
    return Verdict("H1 economic reading", ECONOMIC_SUPPORTED, conditions, basis, tuple(notes))


def h3_verdict(
    beta3: FamaMacBethResult | None,
    beta_cw: FamaMacBethResult | None,
    wald: WaldResult | None = None,
    failures: Mapping[str, str] = {},
) -> Verdict:
    """Intersection-union test: both H3 interactions must reject individually; the Wald test is a complement."""
    basis: dict[str, FamaMacBethResult | WaldResult | None] = {"H3.interaction": beta3,
                                                                 f"H3.{CW_INTERACTION}": beta_cw, "H3.wald": wald}
    conditions: dict[str, bool | None] = {
        "cd_interaction_significant": None if beta3 is None else significant(beta3),
        "cw_interaction_significant": None if beta_cw is None else significant(beta_cw),
        "wald_significant": None if wald is None else wald.p_value < ALPHA,
    }
    notes = ["the joint Wald test is a complement and does not establish distinct information by itself",
             "no magnitude comparison: neither measure is claimed to dominate"]
    if wald is None:
        notes.append(f"Wald test unavailable: {failures.get('H3.wald', 'no result at L = 6')}")
    if beta3 is None or beta_cw is None:
        for key in ("H3.interaction", f"H3.{CW_INTERACTION}"):
            if basis[key] is None:
                notes.append(f"{key} inference unavailable: {failures.get(key, 'no result at L = 6')}")
        return Verdict("H3", NOT_ASSESSABLE, conditions, basis, tuple(notes))
    both = bool(conditions["cd_interaction_significant"] and conditions["cw_interaction_significant"])
    return Verdict("H3", H3_DISTINCT if both else H3_NOT_DISTINCT, conditions, basis, tuple(notes))


# ----------------------------------------------------------------------------------------------------------------
# The structured report


@dataclass(frozen=True)
class Report:
    """Every verdict, series and inference of the frozen reporting rules, ready for serialisation."""

    grid: np.ndarray
    h1: Verdict
    h2: Verdict
    delta: Verdict
    economic: Verdict
    h3: Verdict
    coefficients: dict[str, Inference]  # the standard series: beta_2, beta_3, H2 control terms, H3 CW terms
    bin_local: dict[str, Inference]  # intercept:b, linear:b, quadratic:b of every specification
    paired_h2: dict[str, Inference]  # Delta, Delta_S and the three beta_3 series on the paired months
    paired_h3: dict[str, Inference]  # C and H3 interactions on their common months
    wald: WaldInference  # the joint test of the two H3 interactions on the H3 mask
    slopes: dict[str, Inference]  # C.g(c), H2.g(c) and, if supplied, H2.g|controls_at_means(c)
    non_estimable: dict[str, dict[int, tuple[str, ...]]]  # specification -> month -> reasons
    sample_sizes: dict[str, np.ndarray]  # specification -> n_obs per grid month, kept where not estimable
    failures: dict[str, dict[int, str]]  # statistic -> lag -> reason, for every statistic that failed


def _bin_local(aligned: AlignedEstimates, lags: tuple[int, ...]) -> dict[str, Inference]:
    out = {}
    n_bins = aligned.estimates[SPECIFICATION_C][0].basis.n_bins
    for spec in SPECIFICATIONS:
        for component in BIN_COMPONENTS:
            if spec == SPECIFICATION_B and component == "quadratic":
                continue
            for b in range(1, n_bins + 1):
                series = coefficient_series(aligned, spec, f"{component}:{b}")
                out[series.name] = infer(series, lags)
    return out


def build_report(
    aligned: AlignedEstimates,
    h2_control_means: Mapping[int, np.ndarray] | None = None,
    lags: tuple[int, ...] = LAGS,
) -> Report:
    """Verdicts and statistics from the aligned monthly estimates; `h2_control_means` adds the descriptive slopes."""
    if PRIMARY_LAG not in lags:
        raise ValueError(f"the primary lag L = {PRIMARY_LAG} must be among the lags")
    coefficients = infer_all(standard_series(aligned), lags)
    pair2, pair3 = paired_h2(aligned), paired_h3(aligned)
    paired2 = infer_all({s.name: s for s in (pair2.delta, pair2.delta_sample, pair2.beta3_c, pair2.beta3_c_h2,
                                             pair2.beta3_h2)}, lags)
    paired3 = infer_all({s.name: s for s in (pair3.beta3_c, pair3.beta3_h3, pair3.beta_cw_h3)}, lags)
    wald = infer_wald(pair3.wald_values, pair3.wald_mask, lags)
    slope_sets = [slope_series(aligned, SPECIFICATION_C), slope_series(aligned, SPECIFICATION_H2)]
    if h2_control_means is not None:
        slope_sets.append(slope_series(aligned, SPECIFICATION_H2, controls_at=h2_control_means))
    slopes = infer_all({s.name: s for group in slope_sets for s in group.values()}, lags)

    all_inferences = {**coefficients, **paired2, **paired3, **slopes}
    failures = {name: dict(inf.failures) for name, inf in all_inferences.items() if inf.failures}
    if wald.failures:
        failures["H3.wald"] = dict(wald.failures)
    at_primary = {name: failures.get(name, {}).get(PRIMARY_LAG, "") for name in failures}

    beta3_c, f_c = primary(coefficients[f"{SPECIFICATION_C}.{INTERACTION}"])
    h1 = h1_verdict(beta3_c, f_c)
    beta3_h2, f_h2 = primary(coefficients[f"{SPECIFICATION_H2}.{INTERACTION}"])
    h2 = h2_verdict(h1, beta3_h2, f_h2)
    baseline, _ = primary(paired2[pair2.beta3_c_h2.name])
    delta, _ = primary(paired2[pair2.delta.name])
    delta_reading = delta_interpretation(baseline, delta, {k: at_primary[k] for k in at_primary
                                                           if k in (pair2.delta.name, pair2.beta3_c_h2.name)})
    g_low, _ = primary(slopes[f"{SPECIFICATION_C}.g({LOW})"])
    g_high, _ = primary(slopes[f"{SPECIFICATION_C}.g({HIGH})"])
    economic = economic_interpretation(h1, g_low, g_high, at_primary)
    b3_h3, _ = primary(coefficients[f"{SPECIFICATION_H3}.{INTERACTION}"])
    bcw, _ = primary(coefficients[f"{SPECIFICATION_H3}.{CW_INTERACTION}"])
    wald_primary = wald.results.get(PRIMARY_LAG)
    h3 = h3_verdict(b3_h3, bcw, wald_primary, at_primary)

    return Report(
        grid=aligned.grid,
        h1=h1,
        h2=h2,
        delta=delta_reading,
        economic=economic,
        h3=h3,
        coefficients=coefficients,
        bin_local=_bin_local(aligned, lags),
        paired_h2=paired2,
        paired_h3=paired3,
        wald=wald,
        slopes=slopes,
        non_estimable={spec: aligned.non_estimable(spec) for spec in SPECIFICATIONS},
        sample_sizes={spec: np.array([e.n_obs for e in aligned.estimates[spec]]) for spec in SPECIFICATIONS},
        failures=failures,
    )
