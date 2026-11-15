"""Validated inference inputs of one OOS year: saved regressors, H2 controls and outcomes on the same stock-months.

Pre-specified in docs/pre_analysis_plan.md, "Estimation protocol" (Samples):
the regressions of a formation month use every member of U_t with its saved
ensemble signal S, signal bin, conditional disagreement CD^B and conditional
within-family uncertainty CW, the three centred H2 control ranks with their
``complete_case`` flag, and the evaluation outcomes ``target_rank`` (primary)
and ``ret_next`` (raw-return robustness). Nothing is recomputed here:

* conditional disagreement: the complete year written by
  ``disagreement.signals.conditional`` (JSON marker, exact schema, unique
  keys, the twelve OOS months); S^c = 2 (S - 0.5) uses the saved S;
* H2 controls: ``load_h2_controls``, which also checks the control
  definition, the SHA-256 of its inputs (the conditional-disagreement file
  among them) and of its own output;
* outcomes: ``target_<year>.parquet`` of ``disagreement.data.target``. The
  target directory has no per-year marker: it is built in ``target.partial``
  and published as a whole by renaming, and a rebuilt target removes the
  conditional disagreement and the H2 controls derived after it. Only the
  published directory is read.

The three sources must cover exactly the same stock-months, which are joined
by key with one-to-one validation, never by position and never by an inner
join: a member of U_t without a next-month return keeps its row with a
missing outcome. The H1 and H2 samples (finite outcome; finite outcome and
complete case) are left to the estimation step. Errors name the year, the
source and the failure with counts only, never row values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from disagreement.controls import h2
from disagreement.controls.h2 import RANKS, load_h2_controls
from disagreement.data.config import DataPaths
from disagreement.data.target import TARGET_COLUMNS, rank_target
from disagreement.data.validation import KEY, DataValidationError, as_int64, validate_yyyymm
from disagreement.signals import conditional
from disagreement.signals.conditional import SIGNAL_BINS, signal_bins
from disagreement.walkforward.calendar import month_range

CD = "conditional_between_family_disagreement"
CW = "conditional_within_family_uncertainty"
OUTCOMES = ("target_rank", "ret_next")  # primary, then the raw-return robustness outcome
INFERENCE_COLUMNS = (*KEY, "ensemble_signal", "signal_bin", "centered_signal", CD, CW, *RANKS, "complete_case",
                     *OUTCOMES)
SIGNAL_TOLERANCE = 1e-12  # saved centred signal against 2 (S - 0.5) of the same saved S
# saved target_rank against the frozen normalised rank of the saved ret_next: both come from the same few
# floating-point operations on values in [-1, 1], so rounding differences are of order 1e-16; any departure
# above 1e-12 is a different transformation, not rounding
TARGET_TOLERANCE = 1e-12


def _keyed(frame: pd.DataFrame, columns: tuple[str, ...], oos_year: int, label: str) -> pd.DataFrame:
    """Exact schema, integer keys, unique stock-months, rows of the OOS year only and its twelve months."""
    if set(frame.columns) != set(columns) or len(frame.columns) != len(columns):
        raise DataValidationError(f"{label}: columns {sorted(map(str, frame.columns))}, expected {sorted(columns)}.")
    out = frame.loc[:, list(columns)].copy()
    for column in columns:  # checked before any conversion, so that no raw value can reach an error message
        dtype = out[column].dtype
        numeric = pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)
        if column != "complete_case" and not numeric:
            raise DataValidationError(f"{label}: {column} must be numeric, got dtype {dtype}.")
    out["permno"] = as_int64(frame["permno"], "permno", label)
    out["yyyymm"] = as_int64(frame["yyyymm"], "yyyymm", label)
    if not (out["permno"] > 0).all():
        raise DataValidationError(f"{label}: {int((out['permno'] <= 0).sum())} rows have a non-positive permno.")
    validate_yyyymm(out["yyyymm"].to_numpy(), label)
    n_dup = int(out.duplicated(list(KEY)).sum())
    if n_dup:
        raise DataValidationError(f"{label}: {n_dup} rows repeat a permno/yyyymm key.")
    outside = int((out["yyyymm"] // 100 != oos_year).sum())
    if outside:
        raise DataValidationError(f"{label}: {outside} rows outside the OOS year.")
    expected, observed = month_range(oos_year * 100 + 1, oos_year * 100 + 12), set(out["yyyymm"].tolist())
    if observed != set(expected):
        raise DataValidationError(f"{label}: not the 12 OOS months; missing {sorted(set(expected) - observed)}.")
    return out.sort_values(["yyyymm", "permno"], ignore_index=True)


def _unit_interval(values: pd.Series, name: str, label: str) -> None:
    x = values.to_numpy(dtype="float64", na_value=np.nan)
    bad = int((~np.isfinite(x) | (x < 0.0) | (x > 1.0)).sum())
    if bad:
        raise DataValidationError(f"{label}: {bad} rows have a {name} that is not finite in [0, 1].")


def _check_conditional(frame: pd.DataFrame, label: str) -> None:
    _unit_interval(frame["ensemble_signal"], "ensemble_signal", label)
    _unit_interval(frame[CD], CD, label)
    _unit_interval(frame[CW], CW, label)
    bins = frame["signal_bin"]
    if bins.isna().any():  # pandas reductions skip pd.NA in nullable integer columns
        raise DataValidationError(f"{label}: {int(bins.isna().sum())} rows have a missing signal_bin.")
    if not (pd.api.types.is_integer_dtype(bins) and bins.between(1, SIGNAL_BINS).all()):
        raise DataValidationError(f"{label}: signal_bin must be an integer in [1, {SIGNAL_BINS}].")
    differs = int((signal_bins(frame["ensemble_signal"], frame["yyyymm"], SIGNAL_BINS) != bins).sum())
    if differs:  # the documented rule, checked against the saved bins, which are the ones used
        raise DataValidationError(f"{label}: signal_bin differs from the bins of ensemble_signal in {differs} rows.")


def _check_controls(frame: pd.DataFrame, label: str) -> None:
    if frame["complete_case"].dtype != np.bool_:
        raise DataValidationError(f"{label}: complete_case must be strictly Boolean.")
    ranks = frame.loc[:, list(RANKS)].to_numpy(dtype="float64", na_value=np.nan)
    if np.isinf(ranks).any():
        raise DataValidationError(f"{label}: {int(np.isinf(ranks).sum())} control ranks are infinite.")
    finite = np.isfinite(ranks)
    out_of_range = int(((ranks[finite] < -1.0) | (ranks[finite] > 1.0)).sum())
    if out_of_range:
        raise DataValidationError(f"{label}: {out_of_range} finite control ranks lie outside [-1, 1].")
    inconsistent = int((frame["complete_case"].to_numpy() != finite.all(axis=1)).sum())
    if inconsistent:
        raise DataValidationError(f"{label}: complete_case disagrees with the presence of the three ranks "
                                  f"in {inconsistent} rows.")
    signal = frame["centered_signal"].to_numpy(dtype="float64", na_value=np.nan)
    if not np.isfinite(signal).all():
        raise DataValidationError(f"{label}: centered_signal must be finite.")


def _check_target(frame: pd.DataFrame, label: str) -> None:
    ret = frame["ret_next"].to_numpy(dtype="float64", na_value=np.nan)
    rank = frame["target_rank"].to_numpy(dtype="float64", na_value=np.nan)
    if np.isinf(ret).any() or np.isinf(rank).any():
        raise DataValidationError(f"{label}: outcomes must be finite or missing, never infinite.")
    mismatch = int((np.isnan(ret) != np.isnan(rank)).sum())
    if mismatch:
        raise DataValidationError(f"{label}: target_rank and ret_next are missing in different rows ({mismatch}).")
    observed = ~np.isnan(rank)
    out_of_range = int(((rank[observed] < -1.0) | (rank[observed] > 1.0)).sum())
    if out_of_range:
        raise DataValidationError(f"{label}: {out_of_range} target_rank values lie outside [-1, 1].")
    obs = pd.DataFrame({"yyyymm": frame["yyyymm"].to_numpy()[observed], "ret": ret[observed],
                        "rank": rank[observed]}).sort_values(["yyyymm", "ret"], kind="mergesort", ignore_index=True)
    counts = obs.groupby("yyyymm")["rank"].transform("size").to_numpy()
    single = int(((counts == 1) & (obs["rank"].to_numpy() != 0.0)).sum())
    if single:
        raise DataValidationError(f"{label}: {single} months with a single return do not have target_rank 0.")
    same = obs["yyyymm"].to_numpy()[1:] == obs["yyyymm"].to_numpy()[:-1]
    d_ret, d_rank = np.diff(obs["ret"].to_numpy()), np.diff(obs["rank"].to_numpy())
    disordered = int((same & (((d_ret == 0.0) & (d_rank != 0.0)) | ((d_ret > 0.0) & (d_rank <= 0.0)))).sum())
    if disordered:
        raise DataValidationError(f"{label}: target_rank is not ordered like ret_next within the month "
                                  f"in {disordered} adjacent pairs.")
    # the exact frozen transformation 2 (r - 1) / (n - 1) - 1 of the saved ret_next, built for comparison only:
    # the saved target_rank is the value used and is never replaced
    expected = rank_target(frame.loc[:, ["yyyymm", "ret_next"]]).to_numpy(dtype="float64", na_value=np.nan)
    differs = int((np.abs(rank[observed] - expected[observed]) > TARGET_TOLERANCE).sum())
    if differs:
        raise DataValidationError(f"{label}: target_rank differs from the frozen normalised rank of ret_next "
                                  f"in {differs} rows.")


def _same_keys(reference: pd.DataFrame, other: pd.DataFrame, label: str, reference_name: str) -> None:
    merged = reference.loc[:, list(KEY)].merge(other.loc[:, list(KEY)], on=list(KEY), how="outer", indicator=True)
    missing, extra = int((merged["_merge"] == "left_only").sum()), int((merged["_merge"] == "right_only").sum())
    if missing or extra:
        raise DataValidationError(f"{label} covers different stock-months than {reference_name}: "
                                  f"{missing} missing, {extra} extra.")


def assemble_inference_year(
    conditional_frame: pd.DataFrame, controls: pd.DataFrame, target: pd.DataFrame, oos_year: int
) -> pd.DataFrame:
    """The validated inputs of one OOS year from already loaded sources, one row per member of U_t.

    Returns INFERENCE_COLUMNS sorted by (yyyymm, permno); the saved values are passed on unchanged and missing
    outcomes and control ranks stay missing.
    """
    if isinstance(oos_year, bool) or not isinstance(oos_year, (int, np.integer)):
        raise ValueError("oos_year must be an integer year")
    year = int(oos_year)
    names = {
        "conditional": f"{year} conditional disagreement",
        "controls": f"{year} H2 controls",
        "target": f"{year} target",
    }
    cond = _keyed(conditional_frame, tuple(conditional.OUTPUT_COLUMNS), year, names["conditional"])
    ctrl = _keyed(controls, tuple(h2.OUTPUT_COLUMNS), year, names["controls"])
    tgt = _keyed(target, tuple(TARGET_COLUMNS), year, names["target"])
    _check_conditional(cond, names["conditional"])
    _check_controls(ctrl, names["controls"])
    _check_target(tgt, names["target"])
    _same_keys(cond, ctrl, names["controls"], "the conditional disagreement")
    _same_keys(cond, tgt, names["target"], "the conditional disagreement")

    out = cond.loc[:, [*KEY, "ensemble_signal", "signal_bin", CD, CW]]
    out = out.merge(ctrl.loc[:, [*KEY, "centered_signal", *RANKS, "complete_case"]], on=list(KEY), how="left",
                    validate="one_to_one")
    out = out.merge(tgt.loc[:, [*KEY, *OUTCOMES]], on=list(KEY), how="left", validate="one_to_one")
    if len(out) != len(cond):
        raise DataValidationError(f"{year}: joining the sources changed the number of stock-months.")
    derived = 2.0 * (out["ensemble_signal"].to_numpy(dtype="float64") - 0.5)
    differs = int((np.abs(out["centered_signal"].to_numpy(dtype="float64") - derived) > SIGNAL_TOLERANCE).sum())
    if differs:
        raise DataValidationError(f"{names['controls']}: centered_signal differs from 2 (S - 0.5) of the "
                                  f"conditional disagreement in {differs} rows.")
    out["centered_signal"] = derived  # S^c from the saved S; within SIGNAL_TOLERANCE of the saved centred signal
    if not ((derived >= -1.0) & (derived <= 1.0)).all():
        raise DataValidationError(f"{year}: S^c must lie in [-1, 1].")
    return out.sort_values(["yyyymm", "permno"], ignore_index=True).loc[:, list(INFERENCE_COLUMNS)]


def load_target(paths: DataPaths, oos_year: int) -> pd.DataFrame:
    """The saved outcomes of one OOS year from the published target directory (exact schema; contents unchanged)."""
    path = paths.target / f"target_{oos_year}.parquet"
    if not paths.target.is_dir():
        raise DataValidationError(f"{oos_year}: no published target directory; build the targets first.")
    if not path.exists():
        raise DataValidationError(f"{oos_year}: no target file in the published target directory.")
    label = f"{oos_year} target"
    names = pq.read_schema(path).names
    if list(names) != list(TARGET_COLUMNS):
        raise DataValidationError(f"{label}: columns {list(names)}, expected {list(TARGET_COLUMNS)}.")
    return pd.read_parquet(path)


def load_inference_year(paths: DataPaths, oos_year: int) -> pd.DataFrame:
    """Load, validate and align the three sources of one OOS year (see the module docstring)."""
    conditional_frame = conditional._load(
        paths.conditional_disagreement, oos_year, conditional.OUTPUT_COLUMNS, "conditional disagreement"
    )  # marker, exact schema, unique keys, the twelve OOS months
    controls = load_h2_controls(paths, oos_year)  # definition, input and output SHA-256, schema, keys, months, ranges
    target = load_target(paths, oos_year)
    return assemble_inference_year(conditional_frame, controls, target, oos_year)
