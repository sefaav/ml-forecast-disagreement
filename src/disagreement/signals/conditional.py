"""Conditional disagreement: disagreement and uncertainty ranked among stocks with similar ensemble signals.

Within each OOS formation month, stocks are sorted into 20 signal bins by
their ensemble signal alone: with r the average rank of the signal among the
month's n stocks,

    signal_bin = 1 + floor(20 * (r - 1) / n),

from 1 (lowest signals) to 20 (highest). Equal signals share their average
rank and therefore their bin, so the bins are only approximately
equal-frequency when signals tie. Inside each (month, signal_bin) group, the
raw between-family disagreement and the raw within-family uncertainty are
ranked separately (average ranks for ties) and mapped to [0, 1] as in the
family signals, 0.5 for a single stock or a constant group. Only the saved
ensemble, disagreement and uncertainty files are read: no target or return.

Command line: ``python -m disagreement.signals.conditional [--year Y]``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from disagreement.data.config import DEFAULT_CONFIG, DEFAULT_DATA_DIR, DataPaths
from disagreement.data.validation import KEY, DataValidationError, require_unique_keys
from disagreement.signals import between, within
from disagreement.signals.between import load_ensemble
from disagreement.signals.family import remove_mechanical_bias_diagnostics, unit_rank
from disagreement.walkforward.calendar import baseline_episodes, month_range

SIGNAL_BINS = 20  # the primary specification
OUTPUT_COLUMNS = (
    "permno",
    "yyyymm",
    "ensemble_signal",
    "signal_bin",
    "between_family_disagreement_raw",
    "within_family_uncertainty_raw",
    "conditional_between_family_disagreement",
    "conditional_within_family_uncertainty",
)
CONDITIONED = ("between_family_disagreement_raw", "within_family_uncertainty_raw")


def signal_bins(signal: pd.Series, month: pd.Series, bins: int) -> pd.Series:
    """1 + floor(bins * (r - 1) / n), with r the average rank of the signal within its month of n stocks."""
    grouped = signal.groupby(month)
    twice_rank = (2.0 * grouped.rank(method="average")).round().astype("int64")  # average ranks are half-integers
    n = grouped.transform("count").astype("int64")
    return 1 + (bins * (twice_rank - 2)) // (2 * n)  # exact integer floor


def conditional_disagreement(frame: pd.DataFrame, bins: int) -> pd.DataFrame:
    """Signal bins and the two conditional ranks for aligned stock-months (see the module docstring)."""
    out = frame.loc[:, ["permno", "yyyymm", "ensemble_signal", *CONDITIONED]].copy()
    out["signal_bin"] = signal_bins(out["ensemble_signal"], out["yyyymm"], bins)
    groups = [out["yyyymm"], out["signal_bin"]]
    out["conditional_between_family_disagreement"] = unit_rank(out["between_family_disagreement_raw"], groups)
    out["conditional_within_family_uncertainty"] = unit_rank(out["within_family_uncertainty_raw"], groups)
    return out.sort_values(["yyyymm", "permno"], ignore_index=True).loc[:, list(OUTPUT_COLUMNS)]


def _in_range(values: np.ndarray, high: float) -> bool:
    return bool(np.isfinite(values).all() and (values >= 0.0).all() and (values <= high).all())


def _load(directory: Path, oos_year: int, columns: Sequence[str], name: str) -> pd.DataFrame:
    """A complete year of one source: marker, exact schema, unique keys and the twelve OOS months."""
    path, marker = directory / f"{oos_year}.parquet", directory / f"{oos_year}.json"
    if not (path.exists() and marker.exists()):
        raise DataValidationError(f"{oos_year}: no complete {name} (parquet and JSON marker).")
    frame = pd.read_parquet(path)
    label = f"{oos_year} {name}"
    if set(frame.columns) != set(columns):
        raise DataValidationError(f"{label}: columns {sorted(frame.columns)}, expected {sorted(columns)}.")
    require_unique_keys(frame, label)
    expected, observed = month_range(oos_year * 100 + 1, oos_year * 100 + 12), set(frame["yyyymm"].tolist())
    if observed != set(expected):
        missing, extra = sorted(set(expected) - observed), sorted(observed - set(expected))
        raise DataValidationError(f"{label}: not the 12 OOS months; missing {missing}, extra {extra}.")
    return frame.sort_values(["yyyymm", "permno"], ignore_index=True)


def load_sources(paths: DataPaths, oos_year: int) -> pd.DataFrame:
    """The three sources of one OOS year, validated and aligned on identical stock-months."""
    ensemble = load_ensemble(paths, oos_year).sort_values(["yyyymm", "permno"], ignore_index=True)
    disagreement = _load(
        paths.between_family_disagreement, oos_year, between.OUTPUT_COLUMNS, "between-family disagreement"
    )
    uncertainty = _load(paths.within_family_uncertainty, oos_year, within.OUTPUT_COLUMNS, "within-family uncertainty")

    if not _in_range(disagreement["ensemble_signal"].to_numpy(dtype="float64"), 1.0):
        raise DataValidationError(f"{oos_year} between-family disagreement: ensemble_signal must be in [0, 1].")
    if not _in_range(disagreement["between_family_disagreement_raw"].to_numpy(dtype="float64"), 0.5):
        raise DataValidationError(f"{oos_year} between-family disagreement: values must be finite and in [0, 0.5].")
    sd_columns = [c for c in within.OUTPUT_COLUMNS if c.endswith("_uncertainty_sd")]
    rank_columns = [c for c in within.OUTPUT_COLUMNS[2:] if c not in sd_columns]  # four ranks and their mean
    sds, ranks = (uncertainty[c].to_numpy(dtype="float64") for c in (sd_columns, rank_columns))
    if not (_in_range(sds, np.inf) and _in_range(ranks, 1.0)):
        raise DataValidationError(f"{oos_year} within-family uncertainty: SDs must be >= 0 and ranks in [0, 1].")

    keys = pd.MultiIndex.from_frame(ensemble[list(KEY)])
    for name, frame in (("between-family disagreement", disagreement), ("within-family uncertainty", uncertainty)):
        other = pd.MultiIndex.from_frame(frame[list(KEY)])
        if not other.equals(keys):
            missing, extra = len(keys.difference(other)), len(other.difference(keys))
            raise DataValidationError(
                f"{oos_year} {name} covers different stock-months than the ensemble: {missing} missing, {extra} extra."
            )
    differs = int((disagreement["ensemble_signal"] != ensemble["ensemble_signal"]).sum())
    if differs:
        raise DataValidationError(f"{oos_year} between-family disagreement: ensemble_signal differs in {differs} rows.")
    aligned = ensemble[[*KEY, "ensemble_signal"]].copy()
    aligned["between_family_disagreement_raw"] = disagreement["between_family_disagreement_raw"].to_numpy()
    aligned["within_family_uncertainty_raw"] = uncertainty["within_family_uncertainty_raw"].to_numpy()
    return aligned


def _write(directory: Path, oos_year: int, frame: pd.DataFrame, metadata: dict) -> None:
    """Atomic writes; the JSON sidecar, replaced last, marks a complete year."""
    directory.mkdir(parents=True, exist_ok=True)
    output_path, metadata_path = directory / f"{oos_year}.parquet", directory / f"{oos_year}.json"
    metadata_path.unlink(missing_ok=True)
    tmp_output = output_path.with_suffix(".parquet.tmp")
    tmp_metadata = metadata_path.with_suffix(".json.tmp")
    try:
        frame.loc[:, list(OUTPUT_COLUMNS)].to_parquet(tmp_output, index=False)
        os.replace(tmp_output, output_path)
        text = json.dumps(metadata, indent=2)
        tmp_metadata.write_text(text, encoding="utf-8")
        os.replace(tmp_metadata, metadata_path)  # the last operation: marks success
    finally:
        tmp_output.unlink(missing_ok=True)
        tmp_metadata.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_conditional_disagreement(
    paths: DataPaths, config_path: Path = DEFAULT_CONFIG, years: Iterable[int] | None = None, log=print
) -> list[dict]:
    """Conditional disagreement and uncertainty of the selected OOS years (all by default), with 20 signal bins."""
    baseline_years = [e.oos_year for e in baseline_episodes(config_path)]
    if years is not None:
        missing = sorted(set(years) - set(baseline_years))
        if missing:
            raise ValueError(f"not baseline OOS years: {missing}")
        baseline_years = [y for y in baseline_years if y in set(years)]

    summaries = []
    for oos_year in baseline_years:
        (paths.conditional_disagreement / f"{oos_year}.json").unlink(missing_ok=True)  # not complete until rewritten
        remove_mechanical_bias_diagnostics(paths)
        out = conditional_disagreement(load_sources(paths, oos_year), SIGNAL_BINS)
        metadata = {
            "oos_year": oos_year,
            "signal_bins": SIGNAL_BINS,
            "n_stock_months": len(out),
            "months": sorted(int(m) for m in out["yyyymm"].unique()),
            "ensemble_signals_sha256": _sha256(paths.ensemble_signals / f"{oos_year}.parquet"),
            "between_family_disagreement_sha256": _sha256(paths.between_family_disagreement / f"{oos_year}.parquet"),
            "within_family_uncertainty_sha256": _sha256(paths.within_family_uncertainty / f"{oos_year}.parquet"),
        }
        _write(paths.conditional_disagreement, oos_year, out, metadata)
        log(f"{oos_year}: {len(out):,} stock-months, {SIGNAL_BINS} signal bins")
        summaries.append(metadata)
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m disagreement.signals.conditional")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--year", action="append", type=int, help="OOS year, repeatable; default: all")
    args = parser.parse_args(argv)
    try:
        run_conditional_disagreement(DataPaths(args.data_dir), args.config, args.year)
    except DataValidationError as exc:
        print(f"Data validation failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
