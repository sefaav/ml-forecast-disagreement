"""Design evidence: the forecast-only diagnostics and synthetic-outcome reruns behind the H1 specification.

This module regenerates, from the saved OOS signals and model metadata only,
the evidence recorded in docs/pre_analysis_plan.md, "Signal controls in the
H1 specification" and "Reproducibility of the evidence". It keeps two
categories apart:

* deterministic forecast-only diagnostics, expected to reproduce the recorded
  figures within rounding: within-bin correlations and slopes of CD and CW on
  the signal, their linearity, the 30- and 40-bin sensitivity, the
  specification diagnostics of A, B, C, D (partialled-out interaction
  regressors, within-bin variance shares, identifying variance and the
  standard-error cost of C relative to B, scaled condition numbers), the
  Frisch-Waugh-Lovell identities and the conditions under which the two-step
  specification E coincides with C, and the Elastic Net 2021 checks;
* synthetic-outcome experiments (the A/B/C/D rejection rates and the
  noise-free stress scenario), rerun under the recorded definitions with a
  newly documented implementation. Their results are new simulations, dated,
  never a reproduction of the recorded 18.3%, 5.7%, 5.3% and 20.7%.

Only the saved conditional disagreement, ensemble signals, H2 controls
(complete-case flags and control ranks), OOS prediction metadata and, for a
family-year with constant ranks, the raw OOS predictions are read; the
SHA-256 of every file read, and of the configuration, is recorded. No target,
return or H1-H4 result is used, and no model is refitted. Aggregates only are
written, under ``data/diagnostics/design_evidence/``; ``summary.json`` is
written last and marks a complete run. The deterministic checkpoint fails
closed: a fixed set of expected checks (EXPECTED_DETERMINISTIC_CHECKS) must
all be computed; entries outside their tolerance count as resolved only if
they match the reconstructed value recorded in RESOLVED within
RESOLVED_TOLERANCE; anything else, including an undefined extreme-bin
correlation, is an unresolved failure that sets ``deterministic_checkpoint``
to ``blocked``.

Command line: ``python -m disagreement.diagnostics.design_evidence``
(the complete baseline OOS sample, 300 replications).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import SupportsInt, cast

import numpy as np
import pandas as pd

from disagreement.controls.h2 import INTERACTIONS, RANKS, load_h2_controls
from disagreement.data.config import DEFAULT_CONFIG, DEFAULT_DATA_DIR, DataPaths
from disagreement.data.validation import KEY, DataValidationError
from disagreement.diagnostics.mechanical_bias import load_conditional
from disagreement.models.selection import FAMILIES, model_candidates
from disagreement.signals.between import load_ensemble
from disagreement.signals.conditional import SIGNAL_BINS, conditional_disagreement
from disagreement.signals.family import configured_replicas, load_predictions, unit_rank
from disagreement.walkforward.calendar import baseline_episodes

SEED = 20261109
REPLICATIONS = 300
NOISE_SD = 0.10
ALTERNATIVE_BINS = (30, 40)
SPECIFICATIONS = ("A", "B", "C", "D", "D_S")
SPLINE_QUANTILES = (0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95)
HAC_LAG = 6
MIN_CELL = 30  # the estimability rule of "Estimable months"
MIN_DISTINCT = 3
MIN_SINGULAR_RATIO = 1e-8
CD, CW = "conditional_between_family_disagreement", "conditional_within_family_uncertainty"
RAW_CD, RAW_CW = "between_family_disagreement_raw", "within_family_uncertainty_raw"
ACTIVE_FAMILIES = ("random_forest", "lightgbm", "mlp")  # the three families active in 2021

# Figures recorded on 2026-11-09 from temporary scripts (pre_analysis_plan.md). Reference values for the
# comparison report only: they are neither test expectations nor reproduction targets for the simulations.
HISTORICAL = {
    "cd_vs_signal": {1: 0.65, 2: 0.21, 19: -0.21, 20: -0.66},
    "cw_vs_signal": {1: 0.72, 20: -0.69},
    "cd_slope": {1: 9.0, 20: -8.0},
    "alternative_bins_extreme_abs_corr": 0.66,
    "partialled_corr_extreme_A_D": -0.65,
    "variance_share_extreme_A_D": 0.5,
    "identifying_variance_C_over_B": 0.9815,
    "se_cost_C_over_B": 0.009,
    "condition_number_C_original": 4e3,
    "condition_number_C_local": 4.0,
    "basis_invariance": 3e-13,
    "constant_family_year": {"family": "elastic_net", "oos_year": 2021},
    "cd4_2021": {1: -0.947, 20: 0.788},
    "cd3_2021": {1: 0.674, 20: -0.629},
    "cd4_neighbours": {1: 0.67, 20: -0.67},
    "rejection_boundary": {"A": 0.183, "B": 0.057, "C": 0.053, "D": 0.207},
    "stress_fm_se_reference": 0.00078,
}
CORRELATION_TOLERANCE = 0.005
SHARE_TOLERANCE = 0.0005  # 0.05 percentage point
MIN_IDENTIFYING_SHARE = 1e-10  # of the interaction's squared norm left after partialling; below, no weights
# Discrepancies resolved and recorded in docs/methodology_changes.md (2026-11-09): recorded figures that were
# coarse summaries of several cells or years. The recorded value and its tolerance are kept unchanged; the
# resolution applies only if the reproduced value matches the reconstructed value verified on 2026-11-09
# within RESOLVED_TOLERANCE. A materially changed value is an unresolved failure.
RESOLVED_TOLERANCE = 0.0005
RESOLVED = {
    "partialled_corr_A_bin20": (-0.6588, "recorded 'about -0.65' is one rounded value for A/D x bins 1/20"),
    "cd4_vs_signal_2020_bin20": (-0.6944, "recorded 'about +0.67 and -0.67 in 2020 and 2022' is a two-year summary"),
    "cd4_vs_signal_2022_bin1": (0.6821, "recorded 'about +0.67 and -0.67 in 2020 and 2022' is a two-year summary"),
    "cd4_vs_signal_2022_bin20": (-0.6444, "recorded 'about +0.67 and -0.67 in 2020 and 2022' is a two-year summary"),
    "cd4_vs_signal_neighbours_pooled_bin1": (0.6761, "recorded 'about +0.67' is a rounded two-year summary"),
}
# The deterministic checks the checkpoint requires. A missing or not-computed entry blocks it (fail closed).
EXPECTED_DETERMINISTIC_CHECKS = (
    "cd_vs_signal_bin1", "cd_vs_signal_bin2", "cd_vs_signal_bin19", "cd_vs_signal_bin20",
    "cw_vs_signal_bin1", "cw_vs_signal_bin20", "cd_slope_bin1", "cd_slope_bin20",
    "cd_vs_signal_30bins_bin1", "cd_vs_signal_30bins_bin30", "cd_vs_signal_40bins_bin1", "cd_vs_signal_40bins_bin40",
    "partialled_corr_A_bin1", "partialled_corr_A_bin20", "partialled_corr_D_bin1", "partialled_corr_D_bin20",
    "partialled_corr_A_D_extreme_mean",
    "identifying_variance_C_over_B_pooled", "identifying_variance_C_over_B_mean_monthly",
    "se_cost_C_over_B_pooled", "se_cost_C_over_B_fama_macbeth",
    "cd4_vs_signal_2021_bin1", "cd4_vs_signal_2021_bin20", "cd3_vs_signal_2021_bin1", "cd3_vs_signal_2021_bin20",
    "cd4_vs_signal_2020_bin1", "cd4_vs_signal_2020_bin20", "cd4_vs_signal_2022_bin1", "cd4_vs_signal_2022_bin20",
    "cd4_vs_signal_neighbours_pooled_bin1", "cd4_vs_signal_neighbours_pooled_bin20",
)
PASSING_STATUSES = ("reproduced within tolerance", "outside tolerance")  # 'outside tolerance' needs a resolution

SCENARIOS = {  # name: (d, description of g); the formulas are implemented in scenario_means()
    "linear": (0.0, "g = 0.01 S^c"),
    "cubic": (0.0, "g = 0.01 S^c + 0.015 (S^c)^3"),
    "tanh": (0.0, "g = 0.012 tanh(2.5 S^c)"),
    "boundary": (0.0, "g = 0.01 S^c + 0.05 max(S - q95_t, 0) - 0.05 max(q05_t - S, 0), monthly 5th/95th pct of S"),
    "linear_negative": (-0.01, "g = 0.01 S^c"),
    "cubic_negative": (-0.01, "g = 0.01 S^c + 0.015 (S^c)^3"),
    "linear_positive": (0.01, "g = 0.01 S^c"),
}
STRESS = "g = 0.01 S^c + 0.01 sign(S^c) z^2, z = (S^c - c_b) / h_b with the cell mean c_b and half-range h_b"


# ----------------------------------------------------------------------------------------------------------------
# Cells of a sorted array


@dataclass(frozen=True)
class Cells:
    """Contiguous groups of arrays sorted by group: offsets and sizes."""

    starts: np.ndarray
    counts: np.ndarray

    def sum(self, v: np.ndarray) -> np.ndarray:
        return np.add.reduceat(v, self.starts)

    def mean(self, v: np.ndarray) -> np.ndarray:
        return self.sum(v) / self.counts

    def expand(self, g: np.ndarray) -> np.ndarray:
        return np.repeat(g, self.counts)


def cells_of(codes: np.ndarray) -> Cells:
    """The cells of an array of group codes that is already sorted by code."""
    if codes.size == 0:
        raise ValueError("no observations")
    if (np.diff(codes) < 0).any():
        raise ValueError("codes must be sorted")
    starts = np.concatenate([[0], np.flatnonzero(np.diff(codes)) + 1])
    return Cells(starts, np.diff(np.append(starts, codes.size)))


def cell_statistics(x: np.ndarray, y: np.ndarray, cells: Cells) -> dict[str, np.ndarray]:
    """Correlation, OLS slope and R-squared of y on x inside each cell, with the gain of adding x squared.

    Undefined statistics (fewer than two observations, a constant variable, fewer than three distinct x for the
    quadratic fit) are NaN, never 0.
    """
    n = cells.counts
    dx, dy = x - cells.expand(cells.mean(x)), y - cells.expand(cells.mean(y))
    sxx, syy, sxy = cells.sum(dx * dx), cells.sum(dy * dy), cells.sum(dx * dy)
    with np.errstate(invalid="ignore", divide="ignore"):
        valid = (n >= 2) & (sxx > 0) & (syy > 0)
        corr = np.where(valid, sxy / np.sqrt(sxx * syy), np.nan)
        slope = np.where((n >= 2) & (sxx > 0), sxy / sxx, np.nan)
        r2_linear = np.where(valid, sxy * sxy / (sxx * syy), np.nan)
        x2 = dx * dx
        x2c = x2 - cells.expand(cells.mean(x2))
        a12, a22, b2 = cells.sum(dx * x2c), cells.sum(x2c * x2c), cells.sum(x2c * dy)
        det = sxx * a22 - a12 * a12
        quadratic = valid & (n >= 3) & (det > 1e-12 * sxx * a22)
        beta1 = np.where(quadratic, (a22 * sxy - a12 * b2) / det, np.nan)
        beta2 = np.where(quadratic, (sxx * b2 - a12 * sxy) / det, np.nan)
        r2_quadratic = np.where(quadratic, (beta1 * sxy + beta2 * b2) / syy, np.nan)
    return {
        "n_stocks": n,
        "corr": corr,
        "slope": slope,
        "r2_linear": r2_linear,
        "r2_quadratic_gain": r2_quadratic - r2_linear,
    }


# ----------------------------------------------------------------------------------------------------------------
# Inputs


def load_signals(paths: DataPaths, years: list[int]) -> pd.DataFrame:
    """The saved conditional disagreement of the baseline years, sorted by (yyyymm, permno), with the centred signal.

    The saved 20-bin conditional ranks are recomputed from the saved raw measures and must match exactly.
    """
    frame = pd.concat([load_conditional(paths, y) for y in years], ignore_index=True)
    frame = frame.sort_values(["yyyymm", "permno"], ignore_index=True)
    recomputed = conditional_disagreement(frame, SIGNAL_BINS)
    for column in (CD, CW):
        differs = int((recomputed[column].to_numpy() != frame[column].to_numpy()).sum())
        if differs:
            raise DataValidationError(
                f"conditional disagreement: {column} differs from its recomputation in {differs} rows."
            )
    frame["centered_signal"] = 2.0 * (frame["ensemble_signal"] - 0.5)
    return frame


def load_controls(paths: DataPaths, years: list[int], signals: pd.DataFrame) -> pd.DataFrame:
    """The H2 control ranks, interactions and complete-case flags on exactly the signals' stock-months."""
    controls = pd.concat([load_h2_controls(paths, y) for y in years], ignore_index=True)
    controls = controls.sort_values(["yyyymm", "permno"], ignore_index=True)
    keys, other = pd.MultiIndex.from_frame(signals[list(KEY)]), pd.MultiIndex.from_frame(controls[list(KEY)])
    if not other.equals(keys):
        missing, extra = len(keys.difference(other)), len(other.difference(keys))
        raise DataValidationError(
            f"H2 controls cover different stock-months than the signals: {missing} missing, {extra} extra."
        )
    differs = int((controls["centered_signal"].to_numpy() != signals["centered_signal"].to_numpy()).sum())
    if differs:
        raise DataValidationError(f"H2 controls: centered_signal differs from the saved signal in {differs} rows.")
    return controls


def load_family_ranks(paths: DataPaths, years: list[int], signals: pd.DataFrame) -> pd.DataFrame:
    """The four family ranks of the selected years, on the signals' stock-months with the same ensemble signal."""
    ranks = pd.concat([load_ensemble(paths, y) for y in years], ignore_index=True)
    ranks = ranks.sort_values(["yyyymm", "permno"], ignore_index=True)
    subset = signals[signals["yyyymm"].isin(ranks["yyyymm"].unique())]
    keys, other = pd.MultiIndex.from_frame(subset[list(KEY)]), pd.MultiIndex.from_frame(ranks[list(KEY)])
    if not other.equals(keys):
        raise DataValidationError("ensemble signals cover different stock-months than the conditional disagreement.")
    differs = int((ranks["ensemble_signal"].to_numpy() != subset["ensemble_signal"].to_numpy()).sum())
    if differs:
        raise DataValidationError(
            f"ensemble signals: ensemble_signal differs from the conditional disagreement in {differs} rows."
        )
    return ranks


def selection_metadata(paths: DataPaths, years: list[int]) -> dict[str, dict[int, dict]]:
    """The frozen model-selection metadata of every family-year (validation scores and selected candidate)."""
    out: dict[str, dict[int, dict]] = {}
    for family in FAMILIES:
        out[family] = {}
        for year in years:
            marker = paths.oos_predictions / family / f"{year}.json"
            if not marker.exists():
                raise DataValidationError(f"{family} {year}: no OOS prediction metadata.")
            metadata = json.loads(marker.read_text(encoding="utf-8"))
            scores = metadata.get("validation_scores")
            required = ("selected_candidate_index", "selected_validation_score")
            if not isinstance(scores, list) or any(k not in metadata for k in required):
                raise DataValidationError(f"{family} {year}: metadata without validation scores or selected candidate.")
            out[family][year] = metadata
    return out


# ----------------------------------------------------------------------------------------------------------------
# A. Signal and disagreement geometry


def within_bin_table(frame: pd.DataFrame, bins: int) -> pd.DataFrame:
    """Within-(month, bin) statistics of CD and CW against S, one row per observed cell, for the given bin count."""
    if bins == SIGNAL_BINS:
        data = frame.loc[:, ["permno", "yyyymm", "ensemble_signal", "signal_bin", CD, CW]]
    else:
        data = conditional_disagreement(frame.loc[:, ["permno", "yyyymm", "ensemble_signal", RAW_CD, RAW_CW]], bins)
    order = np.lexsort((data["signal_bin"].to_numpy(), data["yyyymm"].to_numpy()))
    month, signal_bin = data["yyyymm"].to_numpy()[order], data["signal_bin"].to_numpy()[order]
    s, cd, cw = (data[c].to_numpy(dtype="float64")[order] for c in ("ensemble_signal", CD, CW))
    cells = cells_of(month.astype("int64") * 100 + signal_bin)
    cd_stats, cw_stats = cell_statistics(s, cd, cells), cell_statistics(s, cw, cells)
    extremity = np.abs(2.0 * (s - 0.5))
    ext_stats = cell_statistics(extremity, cd, cells)
    return pd.DataFrame(
        {
            "bins": bins,
            "yyyymm": month[cells.starts],
            "signal_bin": signal_bin[cells.starts],
            "n_stocks": cells.counts,
            "mean_signal": cells.mean(s),
            "mean_extremity": cells.mean(extremity),
            "cd_vs_signal": cd_stats["corr"],
            "cd_slope": cd_stats["slope"],
            "cd_r2_linear": cd_stats["r2_linear"],
            "cd_r2_quadratic_gain": cd_stats["r2_quadratic_gain"],
            "cd_vs_extremity": ext_stats["corr"],
            "cw_vs_signal": cw_stats["corr"],
            "cw_slope": cw_stats["slope"],
        }
    )


def bin_profile(cells: pd.DataFrame) -> pd.DataFrame:
    """Equal-month-weight profile of the bins: finite values only, one row per (bins, signal_bin)."""
    rows = []
    for (bins, signal_bin), group in cells.groupby(["bins", "signal_bin"], sort=True):
        row = {
            "bins": int(cast(SupportsInt, bins)), "signal_bin": int(cast(SupportsInt, signal_bin)),
            "n_months": len(group),
            "mean_n_stocks": float(group["n_stocks"].mean()),
            "mean_signal": float(group["mean_signal"].mean()),
            "mean_extremity": float(group["mean_extremity"].mean()),
        }
        for name in ("cd_vs_signal", "cd_slope", "cd_r2_linear", "cd_r2_quadratic_gain", "cd_vs_extremity",
                     "cw_vs_signal", "cw_slope"):
            v = group[name].to_numpy(dtype="float64")
            v = v[np.isfinite(v)]
            row[f"{name}_n_valid"] = len(v)
            row[f"{name}_mean"] = float(v.mean()) if len(v) else np.nan
            row[f"{name}_median"] = float(np.median(v)) if len(v) else np.nan
        v = group["cd_vs_signal"].to_numpy(dtype="float64")
        row["cd_vs_signal_n_positive"] = int((v > 0).sum())
        row["cd_vs_signal_n_negative"] = int((v < 0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def extreme_bin_months(cells: pd.DataFrame) -> pd.DataFrame:
    """Monthly correlations of CD and CW with S in bins 1 and 20 of the 20-bin design."""
    base = cells[cells["bins"] == SIGNAL_BINS]
    out = pd.DataFrame({"yyyymm": np.sort(base["yyyymm"].unique())}).set_index("yyyymm")
    for b, label in ((1, "bin1"), (SIGNAL_BINS, "bin20")):
        rows = base[base["signal_bin"] == b].set_index("yyyymm")
        out[f"{label}_cd_vs_signal"] = rows["cd_vs_signal"]
        out[f"{label}_cw_vs_signal"] = rows["cw_vs_signal"]
    return out.reset_index()


def monthly_cw_correlations(frame: pd.DataFrame) -> dict:
    """Equal-weight means over months of the full-cross-section correlations of CW and W^raw with S and |S^c|."""
    month = frame["yyyymm"].to_numpy()
    cells = cells_of(month)  # sorted by month
    s = frame["ensemble_signal"].to_numpy(dtype="float64")
    ext = np.abs(2.0 * (s - 0.5))
    cd, raw_cd = frame[CD].to_numpy(dtype="float64"), frame[RAW_CD].to_numpy(dtype="float64")
    out = {}
    pairs = [
        (f"{name}_{label}", x, frame[column].to_numpy(dtype="float64"))
        for name, column in (("conditional_uncertainty", CW), ("raw_uncertainty", RAW_CW))
        for label, x in (("vs_signal", s), ("vs_extremity", ext))
    ]
    pairs.append(("conditional_disagreement_vs_conditional_uncertainty", cd, frame[CW].to_numpy(dtype="float64")))
    pairs.append(("raw_disagreement_vs_raw_uncertainty", raw_cd, frame[RAW_CW].to_numpy(dtype="float64")))
    for key, x, v in pairs:
        corr = cell_statistics(x, v, cells)["corr"]
        finite = corr[np.isfinite(corr)]
        out[key] = {
            "n_valid": int(len(finite)),
            "mean_correlation": float(finite.mean()) if len(finite) else None,
            "mean_absolute_correlation": float(np.abs(finite).mean()) if len(finite) else None,
        }
    return out


def fwl_weights(x: np.ndarray, partialled: np.ndarray, label: str) -> tuple[np.ndarray, float]:
    """The weights w = x~ / ||x~||^2 whose inner product with y is the OLS coefficient of x, and ||x~||^2.

    Rejects an interaction that is (numerically) in the span of its controls: its coefficient is not identified
    and the weights would be arbitrarily large.
    """
    ss, total = float(partialled @ partialled), float(x @ x)
    if not (np.isfinite(ss) and total > 0.0 and ss > MIN_IDENTIFYING_SHARE * total):
        raise DataValidationError(f"{label}: the interaction regressor is collinear with its controls.")
    return partialled / ss, ss


# ----------------------------------------------------------------------------------------------------------------
# B. Specification designs


def natural_spline_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Natural cubic spline basis with K knots (K columns: 1, x and K - 2 truncated-power differences)."""
    knots = np.asarray(knots, dtype="float64")
    if knots.ndim != 1 or len(knots) < 3 or not (np.diff(knots) > 0).all():
        raise DataValidationError("spline knots must be at least three strictly increasing values.")
    k_last = knots[-1]

    def d(knot: float) -> np.ndarray:
        return (np.maximum(x - knot, 0.0) ** 3 - np.maximum(x - k_last, 0.0) ** 3) / (k_last - knot)

    d_penultimate = d(knots[-2])
    columns = [np.ones_like(x), x, *(d(knot) - d_penultimate for knot in knots[:-2])]
    return np.column_stack(columns)


def monthly_spline_basis(x: np.ndarray) -> np.ndarray:
    """The spline basis of specification D: knots at the fixed quantiles of x within the month."""
    return natural_spline_basis(x, np.quantile(x, SPLINE_QUANTILES))


def bin_constants(sc: np.ndarray, bins: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cell mean, half-range of S^c and mean of z^2 for bins 1..n_bins, from every member of the month's cell."""
    order = np.argsort(bins, kind="stable")
    sorted_bins, sorted_sc = bins[order], sc[order]
    present = np.unique(sorted_bins)
    if len(present) != n_bins or present[0] != 1 or present[-1] != n_bins:
        raise DataValidationError(f"a month does not populate all {n_bins} signal bins.")
    cells = cells_of(sorted_bins)
    c = cells.mean(sorted_sc)
    h = (np.maximum.reduceat(sorted_sc, cells.starts) - np.minimum.reduceat(sorted_sc, cells.starts)) / 2.0
    h = np.where(h > 0, h, 1.0)  # a constant cell gives z = 0; the estimability check reports it
    z = (sorted_sc - cells.expand(c)) / cells.expand(h)
    return c, h, cells.mean(z * z)


def local_basis(sc: np.ndarray, bins: np.ndarray, n_bins: int, degree: int) -> np.ndarray:
    """Bin-local basis of "Monthly regressions": per bin an intercept, z = (S^c - c_b)/h_b and, with degree 2, q."""
    c, h, z2 = bin_constants(sc, bins, n_bins)
    idx = bins - 1
    z = (sc - c[idx]) / h[idx]
    q = z * z - z2[idx]
    indicator = np.eye(n_bins)[idx]
    columns = [indicator, indicator * z[:, None]]
    if degree == 2:
        columns.append(indicator * q[:, None])
    return np.hstack(columns)


def original_basis(sc: np.ndarray, bins: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin intercepts and bin-specific S^c and (S^c)^2 without rescaling: the same span as local_basis(degree 2)."""
    indicator = np.eye(n_bins)[bins - 1]
    return np.hstack([indicator, indicator * sc[:, None], indicator * (sc * sc)[:, None]])


def residualise(y: np.ndarray, W: np.ndarray) -> np.ndarray:
    """y minus its least-squares projection on the columns of W."""
    coef = np.linalg.lstsq(W, y, rcond=None)[0]
    return y - W @ coef


def scaled_condition(X: np.ndarray) -> tuple[float, float]:
    """(condition number, sigma_min / sigma_max) of X after scaling every column to unit Euclidean norm."""
    norms = np.linalg.norm(X, axis=0)
    if not (norms > 0).all():
        return float("inf"), 0.0
    sv = np.linalg.svd(X / norms, compute_uv=False)
    return float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"), float(sv[-1] / sv[0])


@dataclass(frozen=True)
class MonthDesign:
    """The regressors of one month for every specification: controls W, interaction x and its partialled version."""

    W: dict[str, np.ndarray]
    x: dict[str, np.ndarray]
    partialled: dict[str, np.ndarray]
    z: np.ndarray  # the local z of each stock, for the stress scenario


def month_designs(sc: np.ndarray, bins: np.ndarray, cd: np.ndarray, n_bins: int = SIGNAL_BINS) -> MonthDesign:
    s = sc / 2.0 + 0.5
    n = len(s)
    rank = pd.Series(s).rank(method="average").to_numpy(dtype="float64")
    u = (rank - 1.0) / (n - 1.0) if n > 1 else np.full(n, 0.5)  # the within-month unit rank of S
    ext = np.abs(sc)
    ones = np.ones_like(sc)
    basis1, basis2 = local_basis(sc, bins, n_bins, 1), local_basis(sc, bins, n_bins, 2)
    spline_u, spline_s = monthly_spline_basis(u), monthly_spline_basis(sc)
    cd_r_u, cd_r_s = residualise(cd, spline_u), residualise(cd, spline_s)
    W = {
        "A": np.column_stack([ones, sc, ext, sc * ext, cd]),
        "B": np.column_stack([basis1, cd]),
        "C": np.column_stack([basis2, cd]),
        "D": np.column_stack([spline_u, cd]),
        "D_S": np.column_stack([spline_s, cd]),
    }
    x = {"A": sc * cd, "B": sc * cd, "C": sc * cd, "D": cd_r_u * (2.0 * u - 1.0), "D_S": cd_r_s * sc}
    partialled = {k: residualise(x[k], W[k]) for k in SPECIFICATIONS}
    c, h, _ = bin_constants(sc, bins, n_bins)
    z = (sc - c[bins - 1]) / h[bins - 1]
    return MonthDesign(W, x, partialled, z)


def two_step_coefficient(
    y: np.ndarray, sc: np.ndarray, cd: np.ndarray, V: np.ndarray, W: np.ndarray, level: bool
) -> float:
    """Specification E: CD detrended on V, then y on [W, u~ (if level), S^c u~]; the coefficient of S^c u~."""
    u = residualise(cd, V)
    X = np.column_stack([W, u, sc * u] if level else [W, sc * u])
    return float(np.linalg.lstsq(X, y, rcond=None)[0][-1])


def interaction_coefficient(y: np.ndarray, W: np.ndarray, x: np.ndarray) -> float:
    """The coefficient of x in the regression of y on [W, x]."""
    return float(np.linalg.lstsq(np.column_stack([W, x]), y, rcond=None)[0][-1])


def extreme_cell_diagnostics(partialled: np.ndarray, s: np.ndarray, bins: np.ndarray, n_bins: int) -> dict:
    """Within-bin correlation with S and share of variance explained by [1, S, S^2] of a partialled regressor."""
    out = {}
    shares, floors = [], []
    for b in range(1, n_bins + 1):
        mask = bins == b
        x, v = s[mask], partialled[mask]
        n = int(mask.sum())
        corr, share = np.nan, np.nan
        if n >= 4 and np.ptp(x) > 0 and np.ptp(v) > 0 and len(np.unique(x)) >= 3:
            corr = float(np.corrcoef(x, v)[0, 1])
            xc = x - x.mean()
            fitted = v - residualise(v, np.column_stack([np.ones(n), xc, xc * xc - (xc * xc).mean()]))
            share = float(((fitted - fitted.mean()) ** 2).sum() / ((v - v.mean()) ** 2).sum())
            shares.append(share)
            floors.append(2.0 / (n - 1.0))  # expected share of a pure-noise regressor on two regressors
        if b in (1, n_bins):
            out[f"bin{b}_corr_vs_signal"] = corr
            out[f"bin{b}_signal_variance_share"] = share
    out["all_bins_mean_variance_share"] = float(np.mean(shares)) if shares else np.nan
    out["all_bins_mean_noise_floor"] = float(np.mean(floors)) if floors else np.nan
    return out


# ----------------------------------------------------------------------------------------------------------------
# D. Synthetic outcomes


def scenario_means(sc: np.ndarray, cd: np.ndarray, month_cells: Cells) -> dict[str, np.ndarray]:
    """The deterministic part g(S) + d S^c CD of every scenario, on the stacked stock-months."""
    s = sc / 2.0 + 0.5
    slices = [slice(a, a + n) for a, n in zip(month_cells.starts, month_cells.counts)]
    q05 = month_cells.expand(np.array([np.quantile(s[sl], 0.05) for sl in slices]))
    q95 = month_cells.expand(np.array([np.quantile(s[sl], 0.95) for sl in slices]))
    g = {
        "linear": 0.01 * sc,
        "cubic": 0.01 * sc + 0.015 * sc**3,
        "tanh": 0.012 * np.tanh(2.5 * sc),
        "boundary": 0.01 * sc + 0.05 * np.maximum(s - q95, 0.0) - 0.05 * np.maximum(q05 - s, 0.0),
    }
    return {name: g[name.split("_")[0]] + d * sc * cd for name, (d, _) in SCENARIOS.items()}


def fama_macbeth(series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and the simple Fama-MacBeth standard error (sample SD / sqrt T) along the last axis."""
    T = series.shape[-1]
    return series.mean(axis=-1), series.std(axis=-1, ddof=1) / np.sqrt(T)


def bartlett_hac_se(series: np.ndarray, lag: int) -> np.ndarray:
    """The frozen HAC standard error of "Fama-MacBeth inference" for complete series along the last axis."""
    T = series.shape[-1]
    x = series - series.mean(axis=-1, keepdims=True)
    omega = (x * x).sum(axis=-1) / T
    for lead in range(1, lag + 1):
        omega = omega + 2.0 * (1.0 - lead / (lag + 1.0)) * (x[..., lead:] * x[..., :-lead]).sum(axis=-1) / T
    return np.sqrt(omega / T)


def monte_carlo(
    weights: dict[str, np.ndarray], means: dict[str, np.ndarray], month_cells: Cells, replications: int, seed: int
) -> pd.DataFrame:
    """Rejection rates of the interaction coefficient under every scenario and specification, common random numbers.

    The monthly OLS coefficient of the interaction is the linear functional w_t' y_t (Frisch-Waugh-Lovell), so the
    estimate is the deterministic part w_t' mu_t plus w_t' eps_t; eps is drawn once per replication, in the order of
    the stacked stock-months, and shared by every scenario and specification.
    """
    rng = np.random.default_rng(seed)
    specs = list(weights)
    T, N = len(month_cells.starts), int(month_cells.counts.sum())
    noise = np.empty((len(specs), replications, T))
    for r in range(replications):
        eps = NOISE_SD * rng.standard_normal(N)
        for i, spec in enumerate(specs):
            noise[i, r] = month_cells.sum(weights[spec] * eps)
    rows = []
    for scenario, (d, description) in SCENARIOS.items():
        for i, spec in enumerate(specs):
            deterministic = month_cells.sum(weights[spec] * means[scenario])  # (T,)
            beta = deterministic[None, :] + noise[i]  # (R, T)
            mean, se = fama_macbeth(beta)
            t_fm = mean / se
            t_hac = mean / bartlett_hac_se(beta, HAC_LAG)
            reject_fm, reject_hac = np.abs(t_fm) > 1.96, np.abs(t_hac) > 1.96
            p = reject_fm.mean()
            rows.append(
                {
                    "scenario": scenario, "true_interaction": d, "g": description, "specification": spec,
                    "replications": replications, "months": T, "stock_months": N,
                    "mean_estimate": float(mean.mean()), "bias": float(mean.mean() - d),
                    "noise_free_estimate": float(deterministic.mean()),
                    "mean_fm_se": float(se.mean()), "sd_of_estimates": float(mean.std(ddof=1)),
                    "rejection_rate_fm": float(p),
                    "rejection_rate_fm_mc_se": float(np.sqrt(p * (1 - p) / replications)),
                    "rejection_rate_hac6": float(reject_hac.mean()),
                    "share_rejections_negative": float((reject_fm & (mean < 0)).sum() / max(reject_fm.sum(), 1)),
                }
            )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------------
# C. Constant-family checks (Elastic Net in 2021)


def constant_family_months(ranks: pd.DataFrame) -> dict[str, list[int]]:
    """For each family, the months in which its rank is constant across stocks."""
    out = {}
    for family in FAMILIES:
        n_distinct = ranks.groupby("yyyymm")[f"{family}_rank"].nunique()
        out[family] = [int(m) for m in n_distinct.index[n_distinct == 1]]
    return out


def three_family_disagreement(ranks: pd.DataFrame, signals: pd.DataFrame) -> pd.Series:
    """CD built from the three active families only, ranked within the saved (month, bin) cells."""
    r = np.column_stack([ranks[f"{f}_rank"].to_numpy(dtype="float64") for f in ACTIVE_FAMILIES])
    d3 = pd.Series(r.std(axis=1, ddof=0), index=ranks.index)
    bins = signals.set_index(list(KEY)).loc[pd.MultiIndex.from_frame(ranks[list(KEY)]), "signal_bin"].to_numpy()
    return unit_rank(d3, [ranks["yyyymm"], pd.Series(bins, index=ranks.index)])


def constant_family_report(
    paths: DataPaths, config_path: Path, metadata: dict, signals: pd.DataFrame, years: list[int]
) -> dict:
    """Which family-years have constant ranks or a selected validation score of 0, and the 2021 geometry."""
    zero_selected = [
        {"family": f, "oos_year": y, "selected_candidate_index": m["selected_candidate_index"],
         "n_candidates": len(m["validation_scores"]),
         "n_negative_scores": int(sum(s < 0 for s in m["validation_scores"])),
         "n_zero_scores": int(sum(s == 0 for s in m["validation_scores"])),
         "zero_score_candidates": [
             model_candidates(f, config_path)[i] for i, s in enumerate(m["validation_scores"]) if s == 0
         ],
         "first_zero_selected": m["selected_candidate_index"] == min(
             (i for i, s in enumerate(m["validation_scores"]) if s == 0), default=-1)}
        for f in FAMILIES for y, m in metadata[f].items() if m["selected_validation_score"] == 0
    ]
    affected_years = sorted({e["oos_year"] for e in zero_selected})
    years_to_scan = sorted({y for y in years if any(abs(y - a) <= 1 for a in affected_years)}) or []
    report: dict = {
        "zero_selected_score": zero_selected, "constant_rank_months": {}, "constant_prediction_cross_sections": {},
        "geometry": {}, "files_read": {"ensemble_signals": {}, "oos_predictions": {}},
    }
    if not years_to_scan:
        return report
    ranks = load_family_ranks(paths, years_to_scan, signals)
    report["files_read"]["ensemble_signals"] = {y: paths.ensemble_signals / f"{y}.parquet" for y in years_to_scan}
    constant = {f"{f}": {} for f in FAMILIES}
    for year in years_to_scan:
        months = constant_family_months(ranks[ranks["yyyymm"] // 100 == year])
        for family, m in months.items():
            if m:
                constant[family][str(year)] = m
    report["constant_rank_months"] = {f: v for f, v in constant.items() if v}
    replicas = configured_replicas(config_path)
    for entry in zero_selected:
        predictions = load_predictions(paths, entry["family"], entry["oos_year"], replicas)
        n_distinct = predictions.groupby(["yyyymm", "replica"])["prediction"].nunique()
        key = f"{entry['family']} {entry['oos_year']}"
        report["constant_prediction_cross_sections"][key] = {
            "cross_sections": int(len(n_distinct)), "constant": int((n_distinct == 1).sum()),
        }
        report["files_read"]["oos_predictions"][key] = (
            paths.oos_predictions / entry["family"] / f"{entry['oos_year']}.parquet"
        )
    sub = signals[signals["yyyymm"].isin(ranks["yyyymm"].unique())].reset_index(drop=True)
    cd3 = three_family_disagreement(ranks, sub).to_numpy()
    s, cd4 = sub["ensemble_signal"].to_numpy(dtype="float64"), sub[CD].to_numpy(dtype="float64")
    bins, month = sub["signal_bin"].to_numpy(), sub["yyyymm"].to_numpy()
    order = np.lexsort((bins, month))
    cells = cells_of(month[order].astype("int64") * 100 + bins[order])
    stats4, stats3 = cell_statistics(s[order], cd4[order], cells), cell_statistics(s[order], cd3[order], cells)
    table = pd.DataFrame({"yyyymm": month[order][cells.starts], "signal_bin": bins[order][cells.starts],
                          "cd4": stats4["corr"], "cd3": stats3["corr"]})
    table["oos_year"] = table["yyyymm"] // 100
    geometry = {}
    for year, group in table.groupby("oos_year"):
        geometry[str(int(cast(SupportsInt, year)))] = {
            f"cd{k}_vs_signal_bin{b}": float(group.loc[group["signal_bin"] == b, f"cd{k}"].mean())
            for k in (4, 3) for b in (1, SIGNAL_BINS)
        }
    others = [y for y in years_to_scan if y not in affected_years]
    if others:
        pooled = table[table["oos_year"].isin(others)]
        geometry["neighbours_pooled"] = {"years": others, **{
            f"cd{k}_vs_signal_bin{b}": float(pooled.loc[pooled["signal_bin"] == b, f"cd{k}"].mean())
            for k in (4, 3) for b in (1, SIGNAL_BINS)
        }}
    report["geometry"] = geometry
    # The identities of "Elastic Net in 2021" when one family is fixed at 0.5.
    r = np.column_stack([ranks[f"{f}_rank"].to_numpy(dtype="float64") for f in ACTIVE_FAMILIES])
    s3, d3 = r.mean(axis=1), r.std(axis=1, ddof=0)
    raw = sub[RAW_CD].to_numpy(dtype="float64")
    for entry in zero_selected:
        rows = (ranks["yyyymm"] // 100 == entry["oos_year"]).to_numpy()
        if entry["family"] != "elastic_net" or not rows.any():
            continue
        entry["identity_max_abs_error"] = {
            "signal": float(np.abs(s[rows] - (0.5 + 3.0 * s3[rows]) / 4.0).max()),
            "squared_disagreement": float(
                np.abs(raw[rows] ** 2 - (0.75 * d3[rows] ** 2 + 3.0 / 16.0 * (s3[rows] - 0.5) ** 2)).max()
            ),
        }
    return report


# ----------------------------------------------------------------------------------------------------------------
# Run


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cell(frame: pd.DataFrame, index, column: str) -> float:
    """One numeric cell of a DataFrame as a float."""
    return float(np.asarray(frame.loc[index, column], dtype="float64").reshape(()))


def _finite(value) -> float | None:
    return None if value is None or not np.isfinite(value) else float(value)


def _clean(obj):
    """JSON-ready: NaN and infinities become None, numpy scalars become Python scalars."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return _finite(float(obj))
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def classify_failures(comparison: dict, undefined_months: list | None = None) -> tuple[dict, list]:
    """Fail-closed evaluation of the deterministic checkpoint.

    Returns the documented, resolved discrepancies and the unresolved failures. Unresolved: an expected check that
    is missing or not computed, an entry outside its tolerance without a resolution, an entry whose reproduced
    value differs from the reconstructed value of its resolution by more than RESOLVED_TOLERANCE, and any month
    whose extreme-bin correlation is undefined. A resolution is written into the entry (``resolution``) and never
    changes the entry's recorded value, status or tolerance.
    """
    resolved, unresolved = {}, []
    for key in EXPECTED_DETERMINISTIC_CHECKS:
        entry = comparison.get(key)
        if entry is None or entry["status"] not in PASSING_STATUSES:
            unresolved.append(f"{key}: missing or not computed")
    for key, entry in comparison.items():
        if entry["status"] != "outside tolerance":
            continue
        value = entry["reproduced"]
        if key in RESOLVED and value is not None and np.isfinite(value):
            reconstructed, note = RESOLVED[key]
            if abs(value - reconstructed) <= RESOLVED_TOLERANCE:
                entry["resolution"] = (
                    f"{note}; reconstructed value {reconstructed} verified on 2026-11-09 "
                    f"(methodology_changes.md), tolerance {RESOLVED_TOLERANCE}"
                )
                resolved[key] = entry["resolution"]
                continue
            unresolved.append(f"{key}: differs from the reconstructed value {reconstructed}")
            continue
        unresolved.append(key)
    for month in undefined_months or []:
        unresolved.append(f"extreme-bin correlation undefined in {month}")
    return resolved, sorted(unresolved)


def _within(value, recorded, tolerance) -> str:
    if value is None or not np.isfinite(value):
        return "not computed"
    return "reproduced within tolerance" if abs(value - recorded) <= tolerance else "outside tolerance"


def _write(directory: Path, tables: dict[str, pd.DataFrame], summary: dict) -> None:
    """Atomic writes; summary.json, replaced last, marks a complete diagnostic."""
    directory.mkdir(parents=True, exist_ok=True)
    tmp = [directory / f"{name}.tmp" for name in (*tables, "summary.json")]
    try:
        for (name, table), tmp_path in zip(tables.items(), tmp):
            table.to_csv(tmp_path, index=False)
            os.replace(tmp_path, directory / name)
        text = json.dumps(_clean(summary), indent=2, allow_nan=False)
        tmp[-1].write_text(text, encoding="utf-8")
        os.replace(tmp[-1], directory / "summary.json")  # the last operation: marks success
    finally:
        for tmp_path in tmp:
            tmp_path.unlink(missing_ok=True)


def run_design_evidence(
    paths: DataPaths, config_path: Path = DEFAULT_CONFIG, replications: int = REPLICATIONS, seed: int = SEED, log=print
) -> dict:
    """Run every diagnostic on the complete baseline OOS sample and write the aggregates."""
    directory = paths.design_evidence_diagnostics
    (directory / "summary.json").unlink(missing_ok=True)  # not complete until rewritten
    if replications < 2:
        raise ValueError("at least two replications are needed")
    years = [e.oos_year for e in baseline_episodes(config_path)]
    frame = load_signals(paths, years)
    controls = load_controls(paths, years, frame)
    metadata = selection_metadata(paths, years)
    log(f"{frame['yyyymm'].nunique()} OOS months, {len(frame):,} stock-months")

    # A. geometry
    cells_table = pd.concat([within_bin_table(frame, b) for b in (SIGNAL_BINS, *ALTERNATIVE_BINS)], ignore_index=True)
    profile = bin_profile(cells_table)
    extremes = extreme_bin_months(cells_table)
    cw_monthly = monthly_cw_correlations(frame)

    # B. specifications, month by month
    month = frame["yyyymm"].to_numpy()
    month_cells = cells_of(month)
    sc = frame["centered_signal"].to_numpy(dtype="float64")
    s = frame["ensemble_signal"].to_numpy(dtype="float64")
    bins = frame["signal_bin"].to_numpy().astype("int64")
    cd = frame[CD].to_numpy(dtype="float64")
    complete = controls["complete_case"].to_numpy(dtype=bool)
    control_columns = controls.loc[:, [*RANKS, *INTERACTIONS]].to_numpy(dtype="float64")
    check_rng = np.random.default_rng(0)  # a fixed synthetic outcome for the algebraic identities only
    y_check = 0.02 * np.sin(7.0 * s) + 0.01 * sc * cd + 0.05 * check_rng.standard_normal(len(s))
    weights = {spec: np.empty(len(s)) for spec in SPECIFICATIONS}
    z_all = np.empty(len(s))
    monthly_rows, identity_rows = [], []
    for start, n in zip(month_cells.starts, month_cells.counts):
        sl = slice(start, start + n)
        design = month_designs(sc[sl], bins[sl], cd[sl])
        z_all[sl] = design.z
        row: dict = {"yyyymm": int(month[start]), "n_stocks": int(n), "n_complete_case": int(complete[sl].sum())}
        cell_sizes = np.bincount(bins[sl], minlength=SIGNAL_BINS + 1)[1:]
        cc_sizes = np.bincount(bins[sl][complete[sl]], minlength=SIGNAL_BINS + 1)[1:]
        distinct = [len(np.unique(sc[sl][bins[sl] == b])) for b in range(1, SIGNAL_BINS + 1)]
        distinct_cc = [len(np.unique(sc[sl][(bins[sl] == b) & complete[sl]])) for b in range(1, SIGNAL_BINS + 1)]
        row |= {"min_cell_size": int(cell_sizes.min()), "min_cell_size_complete_case": int(cc_sizes.min()),
                "min_distinct_signal": int(min(distinct)), "min_distinct_signal_complete_case": int(min(distinct_cc))}
        for spec in SPECIFICATIONS:
            xt = design.partialled[spec]
            weights[spec][sl], ss = fwl_weights(design.x[spec], xt, f"{int(month[start])} specification {spec}")
            row[f"{spec}_identifying_variance"] = ss
            row[f"{spec}_condition_number"] = scaled_condition(np.column_stack([design.W[spec], design.x[spec]]))[0]
            for k, v in extreme_cell_diagnostics(xt, s[sl], bins[sl], SIGNAL_BINS).items():
                row[f"{spec}_{k}"] = v
        X_c = np.column_stack([design.W["C"], design.x["C"]])
        X_original = np.column_stack([original_basis(sc[sl], bins[sl], SIGNAL_BINS), cd[sl], design.x["C"]])
        row["C_original_basis_condition_number"] = scaled_condition(X_original)[0]
        cc = complete[sl]
        X_h2 = np.column_stack([X_c[cc], control_columns[sl][cc]])
        cond_h2, ratio_h2 = scaled_condition(X_h2)
        row["H2_complete_case_condition_number"] = cond_h2
        # identifying variance of the interaction on the complete-case sample, with and without the H2 controls
        x_cc, w_cc = design.x["C"][cc], design.W["C"][cc]
        xt_cc = residualise(x_cc, w_cc)
        xt_h2 = residualise(x_cc, np.column_stack([w_cc, control_columns[sl][cc]]))
        ss_cc, ss_h2 = float(xt_cc @ xt_cc), float(xt_h2 @ xt_h2)
        row["C_complete_case_identifying_variance"] = ss_cc
        row["H2_identifying_variance"] = ss_h2
        row["H2_identifying_variance_share_of_C"] = ss_h2 / ss_cc if ss_cc > 0 else np.nan
        row["H2_variance_inflation"] = ss_cc / ss_h2 if ss_h2 > 0 else np.nan
        row["estimable_full"] = bool(cell_sizes.min() >= MIN_CELL and min(distinct) >= MIN_DISTINCT
                                     and scaled_condition(X_c)[1] > MIN_SINGULAR_RATIO)
        row["estimable_complete_case"] = bool(cc_sizes.min() >= MIN_CELL and min(distinct_cc) >= MIN_DISTINCT
                                              and ratio_h2 > MIN_SINGULAR_RATIO)
        monthly_rows.append(row)

        # algebraic identities on the saved regressors with the fixed check outcome
        y = y_check[sl]
        beta_c = np.linalg.lstsq(X_c, y, rcond=None)[0]
        beta_o = np.linalg.lstsq(X_original, y, rcond=None)[0]
        b3_c, b2_c = float(beta_c[-1]), float(beta_c[-2])
        b3_b = interaction_coefficient(y, design.W["B"], design.x["B"])
        V_demean, V_linear, V_quadratic = design.W["B"][:, :SIGNAL_BINS], design.W["B"][:, :-1], design.W["C"][:, :-1]
        xt_c = design.partialled["C"]

        def e(V, W, level=True):  # specification E on this month's regressors
            return two_step_coefficient(y, sc[sl], cd[sl], V, W, level)

        identity_rows.append(
            {
                "yyyymm": int(month[start]),
                "scale_beta3_C": abs(b3_c),
                "basis_invariance_beta2": abs(b2_c - float(beta_o[-2])),
                "basis_invariance_beta3": abs(b3_c - float(beta_o[-1])),
                "fwl_C": abs(b3_c - float(xt_c @ y / (xt_c @ xt_c))),
                "E_demean_controlsB_vs_B": abs(e(V_demean, V_linear) - b3_b),
                "E_demean_controlsC_vs_C": abs(e(V_demean, V_quadratic) - b3_c),
                "E_linear_controlsC_vs_C": abs(e(V_linear, V_quadratic) - b3_c),
                "E_linear_controlsB_vs_B": abs(e(V_linear, V_linear) - b3_b),
                "E_linear_controlsB_vs_C": abs(e(V_linear, V_linear) - b3_c),
                "E_quadratic_controlsC_vs_C": abs(e(V_quadratic, V_quadratic) - b3_c),
                "E_linear_controlsC_no_level_vs_C": abs(e(V_linear, V_quadratic, False) - b3_c),
            }
        )
    monthly = pd.DataFrame(monthly_rows)
    identities = pd.DataFrame(identity_rows)

    spec_rows = []
    iv_b, inv_iv_b = monthly["B_identifying_variance"].to_numpy(), 1.0 / monthly["B_identifying_variance"].to_numpy()
    for spec in SPECIFICATIONS:
        iv = monthly[f"{spec}_identifying_variance"].to_numpy()
        spec_rows.append(
            {
                "specification": spec,
                "identifying_variance_pooled_share_of_B": float(iv.sum() / iv_b.sum()),
                "identifying_variance_mean_monthly_share_of_B": float((iv / iv_b).mean()),
                "se_cost_vs_B_pooled": float(np.sqrt(iv_b.sum() / iv.sum()) - 1.0),
                "se_cost_vs_B_fama_macbeth": float(np.sqrt((1.0 / iv).sum() / inv_iv_b.sum()) - 1.0),
                "bin1_partialled_corr_vs_signal_mean": float(np.nanmean(monthly[f"{spec}_bin1_corr_vs_signal"])),
                "bin20_partialled_corr_vs_signal_mean": float(np.nanmean(monthly[f"{spec}_bin20_corr_vs_signal"])),
                "bin1_signal_variance_share_mean": float(np.nanmean(monthly[f"{spec}_bin1_signal_variance_share"])),
                "bin20_signal_variance_share_mean": float(np.nanmean(monthly[f"{spec}_bin20_signal_variance_share"])),
                "all_bins_signal_variance_share_mean": float(
                    np.nanmean(monthly[f"{spec}_all_bins_mean_variance_share"])
                ),
                "all_bins_noise_floor_mean": float(np.nanmean(monthly[f"{spec}_all_bins_mean_noise_floor"])),
                "condition_number_mean": float(monthly[f"{spec}_condition_number"].mean()),
                "condition_number_max": float(monthly[f"{spec}_condition_number"].max()),
            }
        )
    specifications = pd.DataFrame(spec_rows)

    # D. synthetic outcomes: the Monte-Carlo rerun and the noise-free stress scenario
    means = scenario_means(sc, cd, month_cells)
    simulations = monte_carlo(weights, means, month_cells, replications, seed)
    stress_mean = 0.01 * sc + 0.01 * np.sign(sc) * z_all * z_all
    stress = {}
    for spec in SPECIFICATIONS:
        bias = month_cells.sum(weights[spec] * stress_mean)
        iv = monthly[f"{spec}_identifying_variance"].to_numpy()
        T = len(iv)
        stress[spec] = {
            "noise_free_fm_bias": float(bias.mean()),
            "max_abs_monthly_bias": float(np.abs(bias).max()),
            "analytical_fm_se_at_noise_sd": float(np.sqrt((NOISE_SD**2 / iv).mean() / T)),
        }
        se_reference = stress[spec]["analytical_fm_se_at_noise_sd"]
        stress[spec]["bias_in_standard_errors"] = stress[spec]["noise_free_fm_bias"] / se_reference

    # C. constant-family checks
    constant = constant_family_report(paths, config_path, metadata, frame, years)

    # Comparison with the recorded figures
    def entry(recorded, value, tolerance: float | str = CORRELATION_TOLERANCE, status: str | None = None, **extra):
        status = status or _within(value, recorded, tolerance)
        return {"recorded": recorded, "reproduced": value, "tolerance": tolerance, "status": status, **extra}

    p20 = profile[profile["bins"] == SIGNAL_BINS].set_index("signal_bin")
    comparison = {}
    for b, recorded in HISTORICAL["cd_vs_signal"].items():
        comparison[f"cd_vs_signal_bin{b}"] = entry(recorded, _cell(p20, b, "cd_vs_signal_mean"))
    for b, recorded in HISTORICAL["cw_vs_signal"].items():
        comparison[f"cw_vs_signal_bin{b}"] = entry(recorded, _cell(p20, b, "cw_vs_signal_mean"))
    for b, recorded in HISTORICAL["cd_slope"].items():
        v = _cell(p20, b, "cd_slope_mean")
        status = "reproduced within tolerance" if round(v) == recorded else "outside tolerance"
        comparison[f"cd_slope_bin{b}"] = entry(recorded, v, "same rounded integer", status)
    for nb in ALTERNATIVE_BINS:
        p = profile[profile["bins"] == nb].set_index("signal_bin")
        for b in (1, nb):
            recorded = HISTORICAL["alternative_bins_extreme_abs_corr"] * (1 if b == 1 else -1)
            comparison[f"cd_vs_signal_{nb}bins_bin{b}"] = entry(recorded, _cell(p, b, "cd_vs_signal_mean"))
    sp = specifications.set_index("specification")
    extreme = []
    for spec in ("A", "D"):
        for b in (1, 20):
            v = _cell(sp, spec, f"bin{b}_partialled_corr_vs_signal_mean")
            extreme.append(v)
            comparison[f"partialled_corr_{spec}_bin{b}"] = entry(HISTORICAL["partialled_corr_extreme_A_D"], v)
            share = _cell(sp, spec, f"bin{b}_signal_variance_share_mean")
            comparison[f"variance_share_{spec}_bin{b}"] = entry(
                HISTORICAL["variance_share_extreme_A_D"], share, "qualitative (about half)", "qualitative"
            )
    comparison["partialled_corr_A_D_extreme_mean"] = entry(
        HISTORICAL["partialled_corr_extreme_A_D"], float(np.mean(extreme)),
        note="the recorded figure is one value for the four cells (A and D, bins 1 and 20)",
    )
    for key, recorded, column in (
        ("identifying_variance_C_over_B_pooled", HISTORICAL["identifying_variance_C_over_B"],
         "identifying_variance_pooled_share_of_B"),
        ("identifying_variance_C_over_B_mean_monthly", HISTORICAL["identifying_variance_C_over_B"],
         "identifying_variance_mean_monthly_share_of_B"),
        ("se_cost_C_over_B_pooled", HISTORICAL["se_cost_C_over_B"], "se_cost_vs_B_pooled"),
        ("se_cost_C_over_B_fama_macbeth", HISTORICAL["se_cost_C_over_B"], "se_cost_vs_B_fama_macbeth"),
    ):
        comparison[key] = entry(recorded, _cell(sp, "C", column), SHARE_TOLERANCE)
    comparison["condition_number_C_original_mean"] = entry(
        HISTORICAL["condition_number_C_original"], float(monthly["C_original_basis_condition_number"].mean()),
        "order of magnitude", "qualitative",
    )
    comparison["condition_number_C_local_mean"] = entry(
        HISTORICAL["condition_number_C_local"], _cell(sp, "C", "condition_number_mean"),
        "order of magnitude", "qualitative",
    )
    invariance = float(identities[["basis_invariance_beta2", "basis_invariance_beta3"]].to_numpy().max())
    comparison["basis_invariance_max_abs"] = entry(
        HISTORICAL["basis_invariance"], invariance, "order of magnitude", "qualitative"
    )
    geometry = constant["geometry"]
    if "2021" in geometry:
        for k, hist in (("cd4", HISTORICAL["cd4_2021"]), ("cd3", HISTORICAL["cd3_2021"])):
            for b, recorded in hist.items():
                comparison[f"{k}_vs_signal_2021_bin{b}"] = entry(recorded, geometry["2021"][f"{k}_vs_signal_bin{b}"])
    for label in ("2020", "2022", "neighbours_pooled"):
        if label in geometry:
            for b, recorded in HISTORICAL["cd4_neighbours"].items():
                comparison[f"cd4_vs_signal_{label}_bin{b}"] = entry(recorded, geometry[label][f"cd4_vs_signal_bin{b}"])
    boundary = simulations[simulations["scenario"] == "boundary"].set_index("specification")
    for spec, recorded in HISTORICAL["rejection_boundary"].items():
        comparison[f"rejection_boundary_{spec}"] = entry(
            recorded, _cell(boundary, spec, "rejection_rate_fm"), "none: new simulation", "new simulation",
            mc_se=_cell(boundary, spec, "rejection_rate_fm_mc_se"),
        )
    comparison["stress_fm_se_reference"] = entry(
        HISTORICAL["stress_fm_se_reference"], stress["C"]["analytical_fm_se_at_noise_sd"], "order of magnitude",
        "qualitative",
    )
    comparison["stress_bias_B_in_standard_errors"] = entry(
        -1.0, stress["B"]["bias_in_standard_errors"], "qualitative (approaching one standard error)", "qualitative"
    )
    comparison["stress_bias_C_in_standard_errors"] = entry(
        0.0, stress["C"]["bias_in_standard_errors"], "qualitative (absorbed exactly)", "qualitative"
    )

    bin1, bin20 = extremes["bin1_cd_vs_signal"], extremes["bin20_cd_vs_signal"]
    defined = bin1.notna() & bin20.notna()
    reversed_months = [int(m) for m in extremes.loc[defined & ~((bin1 > 0) & (bin20 < 0)), "yyyymm"]]
    undefined_months = [int(m) for m in extremes.loc[~defined, "yyyymm"]]
    resolved, unresolved = classify_failures(comparison, undefined_months)
    deterministic_failures = sorted(k for k, v in comparison.items() if v["status"] == "outside tolerance")
    sources = {  # every file read by this run
        "config": _sha256(config_path),
        "conditional_disagreement": {str(y): _sha256(paths.conditional_disagreement / f"{y}.parquet") for y in years},
        "conditional_disagreement_metadata": {
            str(y): _sha256(paths.conditional_disagreement / f"{y}.json") for y in years
        },
        "h2_controls": {str(y): _sha256(paths.h2_controls / f"{y}.parquet") for y in years},
        "h2_controls_metadata": {str(y): _sha256(paths.h2_controls / f"{y}.json") for y in years},
        "oos_prediction_metadata": {
            f: {str(y): _sha256(paths.oos_predictions / f / f"{y}.json") for y in years} for f in FAMILIES
        },
        "ensemble_signals": {str(y): _sha256(p) for y, p in constant["files_read"]["ensemble_signals"].items()},
        "oos_predictions": {k: _sha256(p) for k, p in constant["files_read"]["oos_predictions"].items()},
        "transitive_inputs": "the universe, panel and conditional-disagreement files behind each H2-control year "
                             "are verified by load_h2_controls against the inputs_sha256 of its validated "
                             "metadata (hashed above); they are not hashed again here",
    }
    constant = {k: v for k, v in constant.items() if k != "files_read"}
    summary = {
        "status": "complete",
        "deterministic_checkpoint": "passed" if not unresolved else "blocked",
        "expected_deterministic_checks": list(EXPECTED_DETERMINISTIC_CHECKS),
        "resolved_discrepancies": resolved,
        "unresolved_deterministic_failures": unresolved,
        "executed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "first_month": int(month.min()), "last_month": int(month.max()), "months": int(len(month_cells.starts)),
        "n_stock_months": int(len(frame)), "n_complete_case": int(complete.sum()), "signal_bins": SIGNAL_BINS,
        "alternative_bins": list(ALTERNATIVE_BINS),
        "deterministic_diagnostics": {
            "category": "forecast-only; expected to reproduce the recorded figures within rounding",
            "cw_full_cross_section": cw_monthly,
            "months_without_the_usual_extreme_bin_signs": reversed_months,
            "months_with_undefined_extreme_bin_correlation": undefined_months,
            "h2_design_complete_case": {
                "mean_identifying_variance_share_of_C": float(monthly["H2_identifying_variance_share_of_C"].mean()),
                "min_identifying_variance_share_of_C": float(monthly["H2_identifying_variance_share_of_C"].min()),
                "mean_variance_inflation": float(monthly["H2_variance_inflation"].mean()),
                "max_variance_inflation": float(monthly["H2_variance_inflation"].max()),
                "complete_case_share_of_C_identifying_variance": float(
                    monthly["C_complete_case_identifying_variance"].sum() / monthly["C_identifying_variance"].sum()
                ),
            },
            "estimability": {
                "months_estimable_full_sample": int(monthly["estimable_full"].sum()),
                "months_estimable_complete_case": int(monthly["estimable_complete_case"].sum()),
                "min_cell_size": int(monthly["min_cell_size"].min()),
                "min_cell_size_complete_case": int(monthly["min_cell_size_complete_case"].min()),
                "min_distinct_signal": int(monthly["min_distinct_signal"].min()),
                "min_distinct_signal_complete_case": int(monthly["min_distinct_signal_complete_case"].min()),
                "condition_number_C_local_max": float(monthly["C_condition_number"].max()),
                "condition_number_C_original_max": float(monthly["C_original_basis_condition_number"].max()),
                "condition_number_H2_complete_case_max": float(monthly["H2_complete_case_condition_number"].max()),
            },
            "algebraic_identities_max_abs_difference": {
                c: float(identities[c].max()) for c in identities.columns if c != "yyyymm"
            },
            "two_step_specification_E": {
                "identical_to_C": ["demeaning within (month, bin) then the controls of C",
                                   "linear detrending within (month, bin) then the controls of C"],
                "not_identical_to_C": ["quadratic detrending within (month, bin) then the controls of C "
                                       "(S^c times the fitted bin quadratic has a cubic part outside the span of C)",
                                       "linear detrending then the controls of B only (distinct from B and C)",
                                       "any variant without the detrended level term"],
                "condition": "u~ = M_V CD and S^c * P_V CD must both lie in the span of the second-stage controls W: "
                             "V within W and S^c V within W; with bin-polynomial bases, degree(V) + 1 <= degree(W).",
            },
            "constant_family": constant,
        },
        "synthetic_outcome_experiments": {
            "category": "new simulations under the recorded definitions; not a reproduction of the recorded rates",
            "seed": seed, "generator": "numpy.random.default_rng(seed)",
            "draw_order": "per replication r = 0..R-1, eps_r = 0.10 * standard_normal(N) in the (yyyymm, permno) "
                          "order of the stacked stock-months; the same eps_r enters every scenario and specification "
                          "(common random numbers)",
            "replications": replications, "noise_sd": NOISE_SD, "stock_months": int(len(frame)),
            "months": int(len(month_cells.starts)),
            "outcome": "y = g(S) + d S^c CD + eps; the monthly OLS interaction coefficient is w_t'y_t with w_t the "
                       "Frisch-Waugh-Lovell weight of the partialled interaction",
            "inference": {
                "primary": "Fama-MacBeth mean, SE = sample SD / sqrt(T), reject if |t| > 1.96 (recorded convention)",
                "also_reported": f"Bartlett HAC with L = {HAC_LAG}, 1/T, the frozen inference for real H1",
            },
            "scenarios": {k: {"true_interaction": d, "g": g} for k, (d, g) in SCENARIOS.items()},
            "stress_scenario": {"definition": STRESS, "noise_free": True, "results": stress},
        },
        "comparison_with_recorded_figures": comparison,
        "deterministic_checks_outside_tolerance": deterministic_failures,
        "sources_sha256": sources,
    }
    tables = {
        "geometry_bins.csv": profile,
        "extreme_bin_months.csv": extremes,
        "specification_monthly.csv": monthly,
        "specification_summary.csv": specifications,
        "algebraic_identities_monthly.csv": identities,
        "monte_carlo.csv": simulations,
    }
    _write(directory, tables, summary)

    for b in (1, 2, 19, 20):
        log(f"  bin {b:>2}: CD vs S {_cell(p20, b, 'cd_vs_signal_mean'):+.4f}, "
            f"slope {_cell(p20, b, 'cd_slope_mean'):+.2f}, CW vs S {_cell(p20, b, 'cw_vs_signal_mean'):+.4f}")
    log(f"  C keeps {100 * _cell(sp, 'C', 'identifying_variance_pooled_share_of_B'):.3f}% of B's identifying "
        f"variance (SE cost {100 * _cell(sp, 'C', 'se_cost_vs_B_pooled'):.3f}%)")
    for _, r in simulations[simulations["scenario"] == "boundary"].iterrows():
        log(f"  boundary scenario, {r['specification']:<3}: rejection {100 * r['rejection_rate_fm']:.1f}% "
            f"(MC SE {100 * r['rejection_rate_fm_mc_se']:.1f} pp), HAC6 {100 * r['rejection_rate_hac6']:.1f}%")
    log(f"  deterministic checks outside tolerance: {len(deterministic_failures)} "
        f"({len(resolved)} resolved and documented, {len(unresolved)} unresolved); "
        f"checkpoint {summary['deterministic_checkpoint']}")
    log("Diagnostic only; the synthetic results are new simulations, and no methodology was changed automatically.")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m disagreement.diagnostics.design_evidence")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--replications", type=int, default=REPLICATIONS,
        help=f"Monte-Carlo replications (the recorded design uses {REPLICATIONS}; fewer only for synthetic tests)",
    )
    args = parser.parse_args(argv)
    try:
        run_design_evidence(DataPaths(args.data_dir), args.config, args.replications)
    except DataValidationError as exc:
        print(f"Data validation failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
