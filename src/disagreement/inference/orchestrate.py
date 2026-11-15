"""Orchestration of the frozen inference: monthly estimation, calendar alignment, report and outcome coverage.

Pre-specified in docs/pre_analysis_plan.md, "Estimation protocol". The
orchestrator only wires existing components together, year by year:

1. the validated inputs of an OOS year (``load_inference_year``);
2. for every formation month, the five specifications of
   ``estimate_month_specifications`` (C, B, C|H2, H2, H3) on the saved
   ``target_rank`` with all members of U_t, and specification C on the saved
   raw ``ret_next`` (``estimate_h1_month``), the pre-specified H1 outcome
   robustness; the estimator alone forms the samples (H1: finite outcome;
   H2: finite outcome and complete case), the full-U_t basis and the
   estimability checks;
3. the descriptive monthly means of the three centred control ranks on the
   H2 estimation sample (finite ``target_rank`` and complete case), for the
   optional H2 slopes at the observed control means;
4. the descriptive outcome coverage (``outcome_coverage``) of each year, whose
   monthly rows are concatenated and turned into the equal-month-weight
   profile at the end.

After the last year the monthly results are placed on the calendar grid
(``align``), the report is built (``build_report``: verdicts at L = 6,
sensitivity at L = 0 and 12) and the raw-return interaction series receives
the same Fama-MacBeth inference. Raw-return evidence never enters a verdict.
No stock-level table is kept in the result, no file is written, and nothing
is estimated, ranked or tested here beyond those calls.

Pre-specified robustness analyses still outstanding before any analysis of
real outcomes (pre-analysis plan, "Robustness and subperiods"): excluding
2021, the raw, range and MAD disagreement measures, 10 bins,
leave-one-family-out, the two subperiods.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from disagreement.controls.h2 import RANKS
from disagreement.data.config import DEFAULT_CONFIG, DataPaths
from disagreement.inference.coverage import coverage_profile, outcome_coverage
from disagreement.inference.inputs import CD, CW, INFERENCE_COLUMNS, load_inference_year
from disagreement.inference.monthly import (
    SPECIFICATION_C,
    MonthlyEstimate,
    estimate_h1_month,
    estimate_month_specifications,
)
from disagreement.inference.report import Report, build_report
from disagreement.inference.series import (
    INTERACTION,
    LAGS,
    AlignedEstimates,
    Inference,
    MonthlySeries,
    _yyyymm,
    align,
    baseline_grid,
    coefficient,
    infer,
)
from disagreement.walkforward.calendar import baseline_episodes, month_range

PRIMARY_OUTCOME = "target_rank"
RAW_OUTCOME = "ret_next"
RAW_SERIES = f"{SPECIFICATION_C}.{INTERACTION}|{RAW_OUTCOME}"
SUMMARY_COLUMNS = ("yyyymm", "n_stocks", "n_finite_outcome", "n_h2_sample", "finite_outcome_share")
OUTSTANDING_ROBUSTNESS = (
    "excluding the twelve 2021 OOS months",
    "raw, range and MAD disagreement measures",
    "10 signal bins",
    "leave-one-family-out ensembles",
    "two subperiods",
)

MonthEstimator = Callable[..., Mapping[str, MonthlyEstimate]]
RawEstimator = Callable[..., MonthlyEstimate]


@dataclass(frozen=True)
class RawReturnRobustness:
    """Specification C on the raw ret_next: the monthly estimates and the inference of its interaction."""

    estimates: tuple[MonthlyEstimate, ...]  # one per grid month, in calendar order
    interaction: Inference  # beta_3 at L = 6, 0 and 12, or the reasons it could not be computed
    non_estimable: dict[int, tuple[str, ...]]
    sample_sizes: np.ndarray  # n_obs per grid month


@dataclass(frozen=True)
class InferenceRun:
    """Everything the frozen inference produces in memory, ready for serialisation."""

    grid: np.ndarray
    aligned: AlignedEstimates  # the five primary specifications of every month
    report: Report  # verdicts and statistics on target_rank
    raw_return: RawReturnRobustness  # the H1 outcome robustness on ret_next, outside every verdict
    monthly_summary: pd.DataFrame  # SUMMARY_COLUMNS, one row per grid month
    coverage_monthly: pd.DataFrame  # outcome coverage by (month, bin, CD^B tercile)
    coverage_profile: pd.DataFrame  # its equal-month-weight profile over the whole grid
    h2_control_means: dict[int, np.ndarray]  # month -> means of RANKS on the H2 sample (months with one only)


def _validated_grid(grid: Iterable[int]) -> list[int]:
    months = [_yyyymm(m, "calendar grid") for m in grid]  # genuine integers with a month 1..12, never truncated
    if not months or months != month_range(months[0], months[-1]):
        raise ValueError("the calendar grid must be a non-empty sequence of consecutive yyyymm months")
    return months


def _month_inputs(month: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "sc": month["centered_signal"].to_numpy(dtype="float64"),
        "bins": month["signal_bin"].to_numpy(dtype="int64"),
        "cd": month[CD].to_numpy(dtype="float64"),
        "cw": month[CW].to_numpy(dtype="float64"),
        "controls": month.loc[:, list(RANKS)].to_numpy(dtype="float64", na_value=np.nan),
        "complete_case": month["complete_case"].to_numpy(dtype=bool),
    }


def run_inference(
    grid: Iterable[int],
    years: Iterable[tuple[int, pd.DataFrame]],
    estimate: MonthEstimator = estimate_month_specifications,
    estimate_raw: RawEstimator = estimate_h1_month,
    lags: tuple[int, ...] = LAGS,
) -> InferenceRun:
    """The frozen inference on an explicit calendar from validated yearly input frames, processed one at a time.

    `years` yields (oos_year, frame) in calendar order, each frame with INFERENCE_COLUMNS for exactly the grid
    months of that year; a missing, repeated, unexpected or out-of-order year or month is an error, never skipped.
    `estimate` and `estimate_raw` default to the frozen estimators and are replaceable for tests only.
    """
    months = _validated_grid(grid)
    expected_years = sorted({m // 100 for m in months})
    monthly: dict[int, Mapping[str, MonthlyEstimate]] = {}
    raw: dict[int, MonthlyEstimate] = {}
    means: dict[int, np.ndarray] = {}
    summary_rows, coverage_rows = [], []
    seen_years: list[int] = []
    for oos_year, frame in years:
        if isinstance(oos_year, bool) or not isinstance(oos_year, (int, np.integer)):
            raise ValueError(f"{oos_year!r} is not an integer OOS year")
        oos_year = int(oos_year)
        if oos_year not in expected_years or oos_year in seen_years or (seen_years and oos_year < seen_years[-1]):
            raise ValueError(f"{oos_year}: not the next calendar year of the grid (expected {expected_years})")
        seen_years.append(oos_year)
        if list(frame.columns) != list(INFERENCE_COLUMNS):
            raise ValueError(f"{oos_year}: the input frame must have exactly the validated inference columns")
        year_months = sorted(int(m) for m in frame["yyyymm"].unique())
        expected = [m for m in months if m // 100 == oos_year]
        if year_months != expected:
            raise ValueError(f"{oos_year}: input months {year_months} differ from the grid months {expected}")
        finite = np.isfinite(frame[PRIMARY_OUTCOME].to_numpy(dtype="float64", na_value=np.nan))
        year_coverage, _ = outcome_coverage(frame["yyyymm"], frame["signal_bin"], frame[CD],
                                            pd.Series(finite, index=frame.index))
        coverage_rows.append(year_coverage)
        for month, rows in frame.groupby("yyyymm", sort=True):
            month = int(month)  # type: ignore[arg-type]  # yyyymm is an integer column
            inputs = _month_inputs(rows)
            outcome = rows[PRIMARY_OUTCOME].to_numpy(dtype="float64", na_value=np.nan)
            monthly[month] = estimate(outcome=outcome, **inputs)
            raw[month] = estimate_raw(rows[RAW_OUTCOME].to_numpy(dtype="float64", na_value=np.nan),
                                      inputs["sc"], inputs["bins"], inputs["cd"])
            h2_sample = np.isfinite(outcome) & inputs["complete_case"]
            if h2_sample.any():  # an empty H2 sample has no mean: none is invented
                means[month] = inputs["controls"][h2_sample].mean(axis=0)
            summary_rows.append((month, len(rows), int(np.isfinite(outcome).sum()), int(h2_sample.sum()),
                                 float(np.isfinite(outcome).mean())))
    missing_years = sorted(set(expected_years) - set(seen_years))
    if missing_years:
        raise ValueError(f"no input frame for the grid years {missing_years}")

    aligned = align(months, monthly)  # fails closed on any calendar mismatch
    report = build_report(aligned, h2_control_means=means, lags=lags)
    raw_estimates = tuple(raw[m] for m in months)
    raw_mask = np.array([e.estimable for e in raw_estimates], dtype=bool)
    raw_values = np.array([coefficient(e.coefficients, INTERACTION) if e.coefficients is not None else np.nan
                           for e in raw_estimates])
    raw_series = MonthlySeries(RAW_SERIES, aligned.grid, raw_values, raw_mask)
    raw_return = RawReturnRobustness(
        estimates=raw_estimates,
        interaction=infer(raw_series, lags),
        non_estimable={m: e.reasons for m, e in zip(months, raw_estimates) if not e.estimable},
        sample_sizes=np.array([e.n_obs for e in raw_estimates]),
    )
    coverage_monthly = pd.concat(coverage_rows, ignore_index=True)
    return InferenceRun(
        grid=aligned.grid,
        aligned=aligned,
        report=report,
        raw_return=raw_return,
        monthly_summary=pd.DataFrame(summary_rows, columns=list(SUMMARY_COLUMNS)),
        coverage_monthly=coverage_monthly,
        coverage_profile=coverage_profile(coverage_monthly),
        h2_control_means=means,
    )


def baseline_calendar(config_path: Path = DEFAULT_CONFIG) -> tuple[list[int], list[int]]:
    """The frozen OOS grid and its years from the configuration; every year must contribute its twelve months."""
    grid = baseline_grid(config_path)
    years = [e.oos_year for e in baseline_episodes(config_path)]
    if grid != month_range(years[0] * 100 + 1, years[-1] * 100 + 12) or len(grid) != 12 * len(years):
        raise ValueError("the baseline episodes do not form consecutive complete OOS calendar years")
    return grid, years


def run_baseline_inference(
    paths: DataPaths,
    config_path: Path = DEFAULT_CONFIG,
    estimate: MonthEstimator = estimate_month_specifications,
    estimate_raw: RawEstimator = estimate_h1_month,
) -> InferenceRun:
    """The frozen inference on the complete baseline calendar, loading one validated OOS year at a time."""
    grid, years = baseline_calendar(config_path)
    return run_inference(grid, ((year, load_inference_year(paths, year)) for year in years), estimate, estimate_raw)
