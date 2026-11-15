"""Calendar-aligned coefficient series, paired comparisons, marginal-slope series and their Fama-MacBeth inference.

Pre-specified in docs/pre_analysis_plan.md, "Estimation protocol" ("Estimable
months", "Fama-MacBeth inference", "Reporting rules for H1 and H2", "H3").
The layer takes, for every month of an explicit calendar grid, the five
MonthlyEstimate results of `estimate_month_specifications` (C, B, C|H2, H2,
H3) and builds:

* monthly coefficient series on the grid: a value where the monthly
  regression was estimable, NaN and a False mask elsewhere, so that a
  coefficient estimated at exactly zero is never read as missing;
* the paired H2 differences Delta_t = beta_3(H2) - beta_3(C|H2) and
  Delta^S_t = beta_3(C|H2) - beta_3(C), taken month by month on the common
  estimable months of C, C|H2 and H2, with the three beta_3 series restricted
  to that same set (the unrestricted H1 series stays available), and the
  H3-versus-H1 common months;
* the marginal signal slope series g(c) for c in {0.1, 0.5, 0.9} of C (H1)
  and of H2 at X^c = 0 (primary) or at caller-supplied monthly means of the
  centred control ranks (descriptive), inheriting the regression's mask;
* Fama-MacBeth inference with the frozen Bartlett HAC standard error at
  L = 6 (primary), 0 and 12, on the original grid, and the joint Wald test of
  the two H3 interactions in the order (S^c x CD^B, S^c x CW).

The calendar is never compressed: non-estimable months keep their position
and their diagnostic reasons. A series that cannot support inference
(too few estimable months, degenerate variance) keeps its values and reports
the failure with its reason instead of a statistic. Nothing is read from disk
and no monthly regression is run here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from disagreement.controls.h2 import RANKS
from disagreement.data.config import DEFAULT_CONFIG
from disagreement.inference.fama_macbeth import (
    HAC_LAG,
    SENSITIVITY_LAGS,
    FamaMacBethResult,
    WaldResult,
    fama_macbeth,
    fama_macbeth_wald,
)
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
    marginal_slopes,
)
from disagreement.walkforward.calendar import baseline_episodes, month_range

SPECIFICATIONS = (SPECIFICATION_C, SPECIFICATION_B, SPECIFICATION_C_H2, SPECIFICATION_H2, SPECIFICATION_H3)
LAGS = (HAC_LAG, *SENSITIVITY_LAGS)  # 6 primary, then 0 and 12 as sensitivity analyses
DISAGREEMENT = "disagreement"  # beta_2
INTERACTION = "interaction"  # beta_3
CW_INTERACTION = H3_EXTRAS[1]  # beta_CW, the S^c x CW coefficient of H3
SLOPE_SPECIFICATIONS = (SPECIFICATION_C, SPECIFICATION_H2)  # marginal slopes are pre-specified for H1 and H2 only


def baseline_grid(config_path: Path = DEFAULT_CONFIG) -> list[int]:
    """The frozen OOS calendar grid: the formation months of the baseline episodes, first to last."""
    episodes = baseline_episodes(config_path)
    return month_range(episodes[0].test_start, episodes[-1].test_end)


# ----------------------------------------------------------------------------------------------------------------
# Series on the calendar grid


@dataclass(frozen=True)
class MonthlySeries:
    """One monthly quantity on the calendar grid: NaN and a False mask where the month is not estimable."""

    name: str
    grid: np.ndarray  # yyyymm, int64, in calendar order
    values: np.ndarray  # float64; meaningful where `estimable`
    estimable: np.ndarray  # bool

    @property
    def n_months(self) -> int:
        return int(self.estimable.sum())

    @property
    def missing_months(self) -> list[int]:
        return [int(m) for m in self.grid[~self.estimable]]

    def restricted(self, mask: np.ndarray, name: str | None = None) -> MonthlySeries:
        """The same series kept only on `mask` (and its own estimable months); the grid is unchanged."""
        mask = np.asarray(mask)
        if mask.dtype != np.bool_ or mask.shape != self.estimable.shape:
            raise ValueError("the mask must be a strictly Boolean one-dimensional array with one entry per grid month")
        keep = self.estimable & mask
        return MonthlySeries(name or self.name, self.grid, np.where(keep, self.values, np.nan), keep)


@dataclass(frozen=True)
class AlignedEstimates:
    """The five monthly specifications of every month of the grid, in calendar order."""

    grid: np.ndarray
    estimates: dict[str, tuple[MonthlyEstimate, ...]]  # specification -> one estimate per grid month

    def mask(self, specification: str) -> np.ndarray:
        return np.array([e.estimable for e in self.estimates[specification]], dtype=bool)

    def non_estimable(self, specification: str) -> dict[int, tuple[str, ...]]:
        """Calendar month -> diagnostic reasons of the months whose regression was not estimable."""
        return {int(m): e.reasons for m, e in zip(self.grid, self.estimates[specification]) if not e.estimable}


def _yyyymm(value: object, where: str) -> int:
    """A genuine integer year-month key (month 1..12); floats, Booleans and other types are rejected."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{where}: {value!r} is not an integer yyyymm month")
    month = int(value)
    if not 1 <= month % 100 <= 12 or month // 100 < 1:
        raise ValueError(f"{where}: {month} is not a valid yyyymm month")
    return month


def align(
    grid: Iterable[int],
    monthly: Mapping[int, Mapping[str, MonthlyEstimate]] | Sequence[tuple[int, Mapping[str, MonthlyEstimate]]],
) -> AlignedEstimates:
    """Place the monthly results on the explicit grid; missing, duplicate or unexpected months are rejected."""
    grid_array = np.asarray([_yyyymm(m, "calendar grid") for m in grid], dtype="int64")
    if grid_array.size == 0:
        raise ValueError("the calendar grid must be a non-empty sequence of yyyymm months")
    if len(set(grid_array.tolist())) != grid_array.size or (np.diff(grid_array) <= 0).any():
        raise ValueError("the calendar grid must be strictly increasing without duplicates")
    if grid_array.tolist() != month_range(int(grid_array[0]), int(grid_array[-1])):
        raise ValueError("the calendar grid must be consecutive calendar months")
    pairs = monthly.items() if isinstance(monthly, Mapping) else monthly
    items = [(_yyyymm(k, "monthly results"), v) for k, v in pairs]
    keys = [k for k, _ in items]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate months: {sorted(k for k in set(keys) if keys.count(k) > 1)}")
    expected = set(grid_array.tolist())
    if set(keys) != expected:
        missing, extra = sorted(expected - set(keys)), sorted(set(keys) - expected)
        raise ValueError(f"monthly results do not cover the grid exactly: missing {missing}, unexpected {extra}")
    by_month = dict(zip(keys, (v for _, v in items)))
    estimates: dict[str, list[MonthlyEstimate]] = {spec: [] for spec in SPECIFICATIONS}
    for month in grid_array.tolist():
        results = by_month[month]
        if set(results) != set(SPECIFICATIONS):
            raise ValueError(f"{month}: expected the specifications {list(SPECIFICATIONS)}, got {sorted(results)}")
        for spec in SPECIFICATIONS:
            estimate = results[spec]
            if not isinstance(estimate, MonthlyEstimate) or estimate.specification != spec:
                raise ValueError(f"{month}: the result filed under {spec} is not a MonthlyEstimate of {spec}")
            if estimate.estimable != (estimate.coefficients is not None):
                raise ValueError(f"{month} {spec}: estimability and coefficients disagree")
            estimates[spec].append(estimate)
    return AlignedEstimates(grid_array, {spec: tuple(v) for spec, v in estimates.items()})


# ----------------------------------------------------------------------------------------------------------------
# Coefficient series


def coefficient(coefficients: Coefficients, name: str) -> float:
    """One coefficient by name: 'disagreement', 'interaction', an extra term, or 'intercept:b' / 'linear:b' /
    'quadratic:b' for the bin-local coefficient of bin b (1-based)."""
    if name in (DISAGREEMENT, INTERACTION):
        return float(getattr(coefficients, name))
    if name in coefficients.extra:
        return coefficients.extra[name]
    component, sep, index = name.partition(":")
    if sep and component in ("intercept", "linear", "quadratic") and index.isdigit():
        values = getattr(coefficients, component)
        if 1 <= int(index) <= len(values):
            return float(values[int(index) - 1])
    raise KeyError(f"no coefficient named {name!r}")


def coefficient_series(aligned: AlignedEstimates, specification: str, name: str) -> MonthlySeries:
    """The monthly series of one coefficient of one specification; non-estimable months contribute nothing."""
    estimates = aligned.estimates[specification]
    mask = aligned.mask(specification)
    values = np.full(len(estimates), np.nan)
    for i, estimate in enumerate(estimates):
        if estimate.coefficients is not None:
            values[i] = coefficient(estimate.coefficients, name)
    return MonthlySeries(f"{specification}.{name}", aligned.grid, values, mask)


def standard_series(aligned: AlignedEstimates) -> dict[str, MonthlySeries]:
    """beta_2 and beta_3 of every specification, the six H2 control terms and the two H3 CW terms."""
    names = [(spec, name) for spec in SPECIFICATIONS for name in (DISAGREEMENT, INTERACTION)]
    names += [(SPECIFICATION_H2, name) for name in H2_EXTRAS] + [(SPECIFICATION_H3, name) for name in H3_EXTRAS]
    series = {f"{spec}.{name}": coefficient_series(aligned, spec, name) for spec, name in names}
    return series


# ----------------------------------------------------------------------------------------------------------------
# Paired comparisons


@dataclass(frozen=True)
class PairedH2:
    """The frozen H2 paired differences on the common estimable months of C, C|H2 and H2."""

    common: np.ndarray  # the common estimable months (mask on the grid)
    delta: MonthlySeries  # beta_3(H2) - beta_3(C|H2): the change due to the controls on a fixed sample
    delta_sample: MonthlySeries  # beta_3(C|H2) - beta_3(C): the change due to sample composition
    beta3_c: MonthlySeries  # the three beta_3 series restricted to the common months
    beta3_c_h2: MonthlySeries
    beta3_h2: MonthlySeries


@dataclass(frozen=True)
class PairedH3:
    """H3 against H1 on their common estimable months, and the Wald input of the two H3 interactions."""

    common: np.ndarray  # estimable for both C and H3
    beta3_c: MonthlySeries  # H1 restricted to the common months
    beta3_h3: MonthlySeries  # H3's S^c x CD^B restricted to the common months
    beta_cw_h3: MonthlySeries  # H3's S^c x CW restricted to the common months
    wald_values: np.ndarray  # N x 2: (S^c x CD^B, S^c x CW) of every month, NaN where H3 is not estimable
    wald_mask: np.ndarray  # the H3 estimability mask, unrestricted


def _difference(a: MonthlySeries, b: MonthlySeries, common: np.ndarray, name: str) -> MonthlySeries:
    values = np.where(common, a.values - b.values, np.nan)
    return MonthlySeries(name, a.grid, values, common.copy())


def paired_h2(aligned: AlignedEstimates) -> PairedH2:
    c, c_h2, h2 = (coefficient_series(aligned, s, INTERACTION) for s in (SPECIFICATION_C, SPECIFICATION_C_H2,
                                                                          SPECIFICATION_H2))
    common = c.estimable & c_h2.estimable & h2.estimable
    return PairedH2(
        common=common,
        delta=_difference(h2, c_h2, common, "Delta"),
        delta_sample=_difference(c_h2, c, common, "Delta_S"),
        beta3_c=c.restricted(common, f"{c.name}|paired"),
        beta3_c_h2=c_h2.restricted(common, f"{c_h2.name}|paired"),
        beta3_h2=h2.restricted(common, f"{h2.name}|paired"),
    )


def paired_h3(aligned: AlignedEstimates) -> PairedH3:
    c = coefficient_series(aligned, SPECIFICATION_C, INTERACTION)
    h3 = coefficient_series(aligned, SPECIFICATION_H3, INTERACTION)
    cw = coefficient_series(aligned, SPECIFICATION_H3, CW_INTERACTION)
    common = c.estimable & h3.estimable
    return PairedH3(
        common=common,
        beta3_c=c.restricted(common, f"{c.name}|H3-paired"),
        beta3_h3=h3.restricted(common, f"{h3.name}|H3-paired"),
        beta_cw_h3=cw.restricted(common, f"{cw.name}|H3-paired"),
        wald_values=np.column_stack([h3.values, cw.values]),
        wald_mask=h3.estimable.copy(),
    )


# ----------------------------------------------------------------------------------------------------------------
# Marginal signal slopes


def slope_series(
    aligned: AlignedEstimates,
    specification: str,
    controls_at: Mapping[int, np.ndarray] | None = None,
    levels: tuple[float, ...] = SLOPE_DISAGREEMENT,
) -> dict[float, MonthlySeries]:
    """g(c) of every grid month for specification C or H2, with the regression's estimability mask.

    For H2, `controls_at` None is the primary evaluation (X^c = 0); a mapping yyyymm -> three centred control
    means is the descriptive evaluation and must cover every H2-estimable month with months of the grid only.
    """
    if specification not in SLOPE_SPECIFICATIONS:
        raise ValueError(f"marginal slopes are pre-specified for {list(SLOPE_SPECIFICATIONS)} only")
    estimates = aligned.estimates[specification]
    mask = aligned.mask(specification)
    label = f"{specification}.g"
    if controls_at is not None:
        if specification != SPECIFICATION_H2:
            raise ValueError("control values apply to the H2 slopes only")
        months = {int(m) for m in controls_at}
        estimable_months = {int(m) for m in aligned.grid[mask]}
        if not months <= set(aligned.grid.tolist()) or not estimable_months <= months:
            raise ValueError("control means must be given for every H2-estimable month and only for grid months")
        label += "|controls_at_means"
    values = {float(c): np.full(len(estimates), np.nan) for c in levels}
    for i, (month, estimate) in enumerate(zip(aligned.grid.tolist(), estimates)):
        if estimate.coefficients is None:
            continue
        at = None if controls_at is None else np.asarray(controls_at[month], dtype="float64")
        if at is not None and at.shape != (len(RANKS),):
            raise ValueError(f"{month}: control means must be {len(RANKS)} values")
        slopes = marginal_slopes(estimate.coefficients, estimate.basis, at, levels)
        for c in levels:
            values[float(c)][i] = slopes[float(c)]
    return {c: MonthlySeries(f"{label}({c})", aligned.grid, v, mask.copy()) for c, v in values.items()}


# ----------------------------------------------------------------------------------------------------------------
# Inference


@dataclass(frozen=True)
class Inference:
    """Fama-MacBeth inference of one series at each lag, or the explicit reason it could not be computed."""

    series: MonthlySeries
    results: dict[int, FamaMacBethResult]
    failures: dict[int, str]


@dataclass(frozen=True)
class WaldInference:
    """The joint Wald test of the two H3 interactions at each lag, or the explicit reason it failed."""

    values: np.ndarray
    estimable: np.ndarray
    results: dict[int, WaldResult]
    failures: dict[int, str]


def infer(series: MonthlySeries, lags: tuple[int, ...] = LAGS) -> Inference:
    results, failures = {}, {}
    for lag in lags:
        try:
            results[lag] = fama_macbeth(series.values, series.estimable, lag)
        except ValueError as exc:
            failures[lag] = str(exc)
    return Inference(series, results, failures)


def infer_wald(values: np.ndarray, estimable: np.ndarray, lags: tuple[int, ...] = LAGS) -> WaldInference:
    results, failures = {}, {}
    for lag in lags:
        try:
            results[lag] = fama_macbeth_wald(values, estimable, lag)
        except ValueError as exc:
            failures[lag] = str(exc)
    return WaldInference(values, estimable, results, failures)


def infer_all(series: Mapping[str, MonthlySeries], lags: tuple[int, ...] = LAGS) -> dict[str, Inference]:
    return {name: infer(s, lags) for name, s in series.items()}
