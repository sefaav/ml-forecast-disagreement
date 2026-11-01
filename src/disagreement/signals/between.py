"""Raw between-family disagreement: the dispersion of the four family ranks.

For each stock and OOS formation month, with S the ensemble signal,

    between_family_disagreement_raw = sqrt(sum_f (family_rank_f - S) ** 2 / 4),

the population standard deviation (ddof = 0) of the four family ranks: the
four families are the complete fixed set of the design, not a sample. As the
ranks lie in [0, 1], the disagreement lies in [0, 0.5]. It is not ranked,
conditioned or rescaled. Only the saved ensemble signals are read: no
prediction, target, return or model is involved.

Command line: ``python -m disagreement.signals.between [--year Y]``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from disagreement.data.config import DEFAULT_CONFIG, DEFAULT_DATA_DIR, DataPaths
from disagreement.data.validation import DataValidationError, require_unique_keys
from disagreement.signals.ensemble import ENSEMBLE_COLUMNS
from disagreement.signals.family import remove_conditional_disagreement
from disagreement.walkforward.calendar import baseline_episodes, month_range

OUTPUT_COLUMNS = ("permno", "yyyymm", "ensemble_signal", "between_family_disagreement_raw")


def between_family_disagreement(ensemble: pd.DataFrame) -> pd.DataFrame:
    """The ensemble signal and the population SD of the four family ranks, one row per stock-month."""
    out = ensemble.loc[:, list(ENSEMBLE_COLUMNS)].sort_values(["yyyymm", "permno"], ignore_index=True)
    s = out["ensemble_signal"]
    out["between_family_disagreement_raw"] = np.sqrt(
        (
            (out["elastic_net_rank"] - s) ** 2
            + (out["random_forest_rank"] - s) ** 2
            + (out["lightgbm_rank"] - s) ** 2
            + (out["mlp_rank"] - s) ** 2
        )
        / 4.0
    )
    return out.loc[:, list(OUTPUT_COLUMNS)]


def load_ensemble(paths: DataPaths, oos_year: int) -> pd.DataFrame:
    """A complete year of ensemble signals, checked against its own family ranks."""
    signals_path, marker = paths.ensemble_signals / f"{oos_year}.parquet", paths.ensemble_signals / f"{oos_year}.json"
    if not (signals_path.exists() and marker.exists()):
        raise DataValidationError(f"{oos_year}: no complete ensemble signals (parquet and JSON marker).")
    frame = pd.read_parquet(signals_path)
    label = f"{oos_year} ensemble signals"
    if set(frame.columns) != set(ENSEMBLE_COLUMNS):
        raise DataValidationError(f"{label}: columns {sorted(frame.columns)}, expected {sorted(ENSEMBLE_COLUMNS)}.")
    require_unique_keys(frame, label)
    values = frame.loc[:, list(ENSEMBLE_COLUMNS[2:])].to_numpy(dtype="float64")
    if not (np.isfinite(values).all() and (values >= 0.0).all() and (values <= 1.0).all()):
        raise DataValidationError(f"{label}: ranks and signal must be finite and in [0, 1].")
    expected, observed = month_range(oos_year * 100 + 1, oos_year * 100 + 12), set(frame["yyyymm"].tolist())
    if observed != set(expected):
        missing, extra = sorted(set(expected) - observed), sorted(observed - set(expected))
        raise DataValidationError(f"{label}: not the 12 OOS months; missing {missing}, extra {extra}.")
    mean = (frame["elastic_net_rank"] + frame["random_forest_rank"] + frame["lightgbm_rank"] + frame["mlp_rank"]) / 4.0
    inconsistent = int((frame["ensemble_signal"] != mean).sum())
    if inconsistent:
        raise DataValidationError(
            f"{label}: ensemble_signal is not the mean of the family ranks in {inconsistent} rows."
        )
    return frame


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


def run_between_family_disagreement(
    paths: DataPaths, config_path: Path = DEFAULT_CONFIG, years: Iterable[int] | None = None, log=print
) -> list[dict]:
    """Raw between-family disagreement of the selected OOS years (all by default)."""
    baseline_years = [e.oos_year for e in baseline_episodes(config_path)]
    if years is not None:
        missing = sorted(set(years) - set(baseline_years))
        if missing:
            raise ValueError(f"not baseline OOS years: {missing}")
        baseline_years = [y for y in baseline_years if y in set(years)]

    summaries = []
    for oos_year in baseline_years:
        (paths.between_family_disagreement / f"{oos_year}.json").unlink(missing_ok=True)  # not complete until rewritten
        remove_conditional_disagreement(paths, oos_year)
        disagreement = between_family_disagreement(load_ensemble(paths, oos_year))
        source = paths.ensemble_signals / f"{oos_year}.parquet"
        metadata = {
            "oos_year": oos_year,
            "n_stock_months": len(disagreement),
            "months": sorted(int(m) for m in disagreement["yyyymm"].unique()),
            "ensemble_signals_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
        _write(paths.between_family_disagreement, oos_year, disagreement, metadata)
        log(f"{oos_year}: {len(disagreement):,} stock-months")
        summaries.append(metadata)
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m disagreement.signals.between")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--year", action="append", type=int, help="OOS year, repeatable; default: all")
    args = parser.parse_args(argv)
    try:
        run_between_family_disagreement(DataPaths(args.data_dir), args.config, args.year)
    except DataValidationError as exc:
        print(f"Data validation failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
