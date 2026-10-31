"""Family signals: monthly replica ranks and one consensus rank per model family.

For each family and OOS formation month, the raw predictions of each replica
are ranked across stocks (average ranks for ties) and mapped to [0, 1] by
(rank - 1) / (n - 1), or 0.5 when n = 1: ``replica_rank``. Each stock's
``mean_replica_rank`` is the mean of its replica ranks, and ``family_rank``
ranks these means again across the month's stocks on the same scale. Ranking
never crosses months, replicas or families. Only the saved OOS predictions
are read: no target, return or model is involved.

The mean is computed from the exact sum of the replica ranks (half-integers),
so stocks with the same replica ranks in a different replica order tie
exactly in ``family_rank``.

Command line: ``python -m disagreement.signals.family [--family F] [--year Y]``.
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
import yaml

from disagreement.data.config import DEFAULT_CONFIG, DEFAULT_DATA_DIR, DataPaths
from disagreement.data.validation import DataValidationError, require_unique_keys
from disagreement.models.selection import FAMILIES
from disagreement.walkforward.calendar import baseline_episodes

PREDICTION_COLUMNS = ("permno", "yyyymm", "family", "replica", "prediction")
SIGNAL_COLUMNS = ("permno", "yyyymm", "family", "replica", "replica_rank", "mean_replica_rank", "family_rank")
SIGNAL_KEY = ("permno", "yyyymm", "family", "replica")


def _unit_scale(rank: pd.Series, n: pd.Series) -> pd.Series:
    """(rank - 1) / (n - 1) in [0, 1]; 0.5 for a single-stock cross-section."""
    return ((rank - 1.0) / (n - 1.0).where(n > 1)).fillna(0.5)


def unit_rank(values: pd.Series, groups: Sequence[pd.Series]) -> pd.Series:
    """Average-tie ranks within each group, mapped to [0, 1]."""
    grouped = values.groupby(list(groups))
    return _unit_scale(grouped.rank(method="average"), grouped.transform("count"))


def family_signals(predictions: pd.DataFrame, replicas: int) -> pd.DataFrame:
    """Replica ranks, their mean and the consensus rank, one row per prediction row."""
    out = predictions.loc[:, list(PREDICTION_COLUMNS)].sort_values(["yyyymm", "permno", "replica"], ignore_index=True)
    family, month, permno, replica = out["family"], out["yyyymm"], out["permno"], out["replica"]
    by_cross_section = out["prediction"].groupby([family, month, replica])
    rank = by_cross_section.rank(method="average")
    n = by_cross_section.transform("count")  # the same in every replica of a month
    out["replica_rank"] = _unit_scale(rank, n)

    rank_sum = rank.groupby([family, permno, month]).transform("sum")  # exact: ranks are half-integers
    out["mean_replica_rank"] = _unit_scale(rank_sum / replicas, n)
    stock = out["replica"] == out["replica"].min()  # one row per stock-month
    consensus = unit_rank(rank_sum[stock], [family[stock], month[stock]])
    out["family_rank"] = consensus.reindex(out.index).groupby([family, permno, month]).transform("first")
    return out.loc[:, list(SIGNAL_COLUMNS)]


def configured_replicas(config_path: Path = DEFAULT_CONFIG) -> int:
    with open(config_path, encoding="utf-8") as fh:
        return int(yaml.safe_load(fh)["bootstrap"]["replicas"])


def load_predictions(paths: DataPaths, family: str, oos_year: int, replicas: int) -> pd.DataFrame:
    """A complete family-year of raw OOS predictions, checked before any ranking."""
    directory = paths.oos_predictions / family
    predictions_path, marker = directory / f"{oos_year}.parquet", directory / f"{oos_year}.json"
    if not (predictions_path.exists() and marker.exists()):
        raise DataValidationError(f"{family} {oos_year}: no complete OOS predictions (parquet and JSON marker).")
    frame = pd.read_parquet(predictions_path)
    label = f"{family} {oos_year} predictions"
    if set(frame.columns) != set(PREDICTION_COLUMNS):
        raise DataValidationError(f"{label}: columns {sorted(frame.columns)}, expected {sorted(PREDICTION_COLUMNS)}.")
    if not (frame["family"] == family).all():
        raise DataValidationError(f"{label}: rows of another family.")
    if not (frame["yyyymm"] // 100 == oos_year).all():
        raise DataValidationError(f"{label}: rows outside the OOS year.")
    if not np.isfinite(frame["prediction"].to_numpy(dtype="float64")).all():
        raise DataValidationError(f"{label}: non-finite predictions.")
    require_unique_keys(frame, label, SIGNAL_KEY)
    counts = frame.groupby(["permno", "yyyymm"])["replica"].count()
    if not (frame["replica"].isin(range(replicas)).all() and (counts == replicas).all()):
        raise DataValidationError(f"{label}: every stock-month needs exactly replicas 0..{replicas - 1}.")
    return frame


def _write(directory: Path, oos_year: int, frame: pd.DataFrame, metadata: dict) -> None:
    """Atomic writes; the JSON sidecar, replaced last, marks a complete family-year."""
    directory.mkdir(parents=True, exist_ok=True)
    signals_path, metadata_path = directory / f"{oos_year}.parquet", directory / f"{oos_year}.json"
    metadata_path.unlink(missing_ok=True)
    tmp_signals = signals_path.with_suffix(".parquet.tmp")
    tmp_metadata = metadata_path.with_suffix(".json.tmp")
    try:
        frame.loc[:, list(SIGNAL_COLUMNS)].to_parquet(tmp_signals, index=False)
        os.replace(tmp_signals, signals_path)
        text = json.dumps(metadata, indent=2)
        tmp_metadata.write_text(text, encoding="utf-8")
        os.replace(tmp_metadata, metadata_path)  # the last operation: marks success
    finally:
        tmp_signals.unlink(missing_ok=True)
        tmp_metadata.unlink(missing_ok=True)


def run_family_signals(
    paths: DataPaths,
    config_path: Path = DEFAULT_CONFIG,
    families: Iterable[str] | None = None,
    years: Iterable[int] | None = None,
    log=print,
) -> list[dict]:
    """Rank the saved OOS predictions of the selected families and years (all by default)."""
    replicas = configured_replicas(config_path)
    families = list(FAMILIES if families is None else families)
    unknown = sorted(set(families) - set(FAMILIES))
    if unknown:
        raise ValueError(f"unknown model families: {unknown}")
    baseline_years = [e.oos_year for e in baseline_episodes(config_path)]
    if years is not None:
        missing = sorted(set(years) - set(baseline_years))
        if missing:
            raise ValueError(f"not baseline OOS years: {missing}")
        baseline_years = [y for y in baseline_years if y in set(years)]

    summaries = []
    for oos_year in baseline_years:
        for family in families:
            directory = paths.family_signals / family
            (directory / f"{oos_year}.json").unlink(missing_ok=True)  # not complete until rewritten
            predictions = load_predictions(paths, family, oos_year, replicas)
            signals = family_signals(predictions, replicas)
            source = paths.oos_predictions / family / f"{oos_year}.parquet"
            metadata = {
                "family": family,
                "oos_year": oos_year,
                "replicas": replicas,
                "n_prediction_rows": len(signals),
                "n_stock_months": len(signals) // replicas,
                "months": sorted(int(m) for m in signals["yyyymm"].unique()),
                "predictions_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
            _write(directory, oos_year, signals, metadata)
            log(f"{family} {oos_year}: {metadata['n_stock_months']:,} stock-months, {len(signals):,} rows")
            summaries.append(metadata)
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m disagreement.signals.family")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--family", action="append", choices=FAMILIES, help="repeatable; default: all four")
    parser.add_argument("--year", action="append", type=int, help="OOS year, repeatable; default: all")
    args = parser.parse_args(argv)
    try:
        run_family_signals(DataPaths(args.data_dir), args.config, args.family, args.year)
    except DataValidationError as exc:
        print(f"Data validation failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
