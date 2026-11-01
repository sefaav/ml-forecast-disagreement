"""Ensemble signal: the equal-weight average of the four family consensus ranks.

For each stock and OOS formation month,

    ensemble_signal = (elastic_net_rank + random_forest_rank + lightgbm_rank + mlp_rank) / 4,

where each column is that family's ``family_rank`` on [0, 1]. The average is
not ranked again, centred or rescaled. Only the saved family signals are read:
no prediction, target, return or model is involved. The four families must
cover exactly the same stock-months, spanning all twelve months of the OOS
year; none is dropped or filled.

Command line: ``python -m disagreement.signals.ensemble [--year Y]``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from disagreement.data.config import DEFAULT_CONFIG, DEFAULT_DATA_DIR, DataPaths
from disagreement.data.validation import DataValidationError, require_unique_keys
from disagreement.models.selection import FAMILIES
from disagreement.signals.family import (
    SIGNAL_COLUMNS, SIGNAL_KEY, configured_replicas, remove_between_family_disagreement,
)
from disagreement.walkforward.calendar import baseline_episodes, month_range

ENSEMBLE_COLUMNS = (
    "permno", "yyyymm", "elastic_net_rank", "random_forest_rank", "lightgbm_rank", "mlp_rank", "ensemble_signal",
)
RANK_COLUMNS = ("replica_rank", "mean_replica_rank", "family_rank")


def _in_unit_interval(values: np.ndarray) -> bool:
    return bool(np.isfinite(values).all() and (values >= 0.0).all() and (values <= 1.0).all())


def family_ranks(signals: pd.DataFrame, family: str, replicas: int, label: str) -> pd.Series:
    """One ``family_rank`` per stock-month, after checking that it repeats exactly across replicas."""
    if set(signals.columns) != set(SIGNAL_COLUMNS):
        raise DataValidationError(f"{label}: columns {sorted(signals.columns)}, expected {sorted(SIGNAL_COLUMNS)}.")
    if not (signals["family"] == family).all():
        raise DataValidationError(f"{label}: rows of another family.")
    require_unique_keys(signals, label, SIGNAL_KEY)
    counts = signals.groupby(["permno", "yyyymm"])["replica"].count()
    if not (signals["replica"].isin(range(replicas)).all() and (counts == replicas).all()):
        raise DataValidationError(f"{label}: every stock-month needs exactly replicas 0..{replicas - 1}.")
    if not _in_unit_interval(signals.loc[:, list(RANK_COLUMNS)].to_numpy(dtype="float64")):
        raise DataValidationError(f"{label}: ranks must be finite and in [0, 1].")
    by_stock_month = signals.groupby(["permno", "yyyymm"])["family_rank"]
    inconsistent = int((by_stock_month.nunique() != 1).sum())
    if inconsistent:
        raise DataValidationError(f"{label}: family_rank differs across replicas in {inconsistent} stock-months.")
    return by_stock_month.first()


def ensemble_signals(ranks: Mapping[str, pd.Series]) -> pd.DataFrame:
    """The four family ranks side by side and their equal-weight average, one row per stock-month."""
    ranks = {family: ranks[family].sort_index() for family in FAMILIES}
    keys = ranks[FAMILIES[0]].index
    for family in FAMILIES[1:]:
        other = ranks[family].index
        if not (other.is_unique and other.equals(keys)):
            missing, extra = len(keys.difference(other)), len(other.difference(keys))
            raise DataValidationError(
                f"{family} covers different stock-months than {FAMILIES[0]}: {missing} missing, {extra} extra."
            )
    out = pd.DataFrame({f"{family}_rank": ranks[family].to_numpy() for family in FAMILIES}, index=keys)
    out["ensemble_signal"] = (
        out["elastic_net_rank"] + out["random_forest_rank"] + out["lightgbm_rank"] + out["mlp_rank"]
    ) / 4.0
    out = out.reset_index().sort_values(["yyyymm", "permno"], ignore_index=True).loc[:, list(ENSEMBLE_COLUMNS)]
    require_unique_keys(out, "ensemble signals")
    if not _in_unit_interval(out.loc[:, list(ENSEMBLE_COLUMNS[2:])].to_numpy(dtype="float64")):
        raise DataValidationError("ensemble signals: ranks and signal must be finite and in [0, 1].")
    return out


def load_family_ranks(paths: DataPaths, family: str, oos_year: int, replicas: int) -> pd.Series:
    """The family ranks of a complete family-year of family signals."""
    directory = paths.family_signals / family
    signals_path, marker = directory / f"{oos_year}.parquet", directory / f"{oos_year}.json"
    if not (signals_path.exists() and marker.exists()):
        raise DataValidationError(f"{family} {oos_year}: no complete family signals (parquet and JSON marker).")
    label = f"{family} {oos_year} family signals"
    ranks = family_ranks(pd.read_parquet(signals_path), family, replicas, label)  # schema checked first
    if not (ranks.index.get_level_values("yyyymm") // 100 == oos_year).all():
        raise DataValidationError(f"{label}: rows outside the OOS year.")
    return ranks


def _write(directory: Path, oos_year: int, frame: pd.DataFrame, metadata: dict) -> None:
    """Atomic writes; the JSON sidecar, replaced last, marks a complete year."""
    directory.mkdir(parents=True, exist_ok=True)
    signals_path, metadata_path = directory / f"{oos_year}.parquet", directory / f"{oos_year}.json"
    metadata_path.unlink(missing_ok=True)
    tmp_signals = signals_path.with_suffix(".parquet.tmp")
    tmp_metadata = metadata_path.with_suffix(".json.tmp")
    try:
        frame.loc[:, list(ENSEMBLE_COLUMNS)].to_parquet(tmp_signals, index=False)
        os.replace(tmp_signals, signals_path)
        text = json.dumps(metadata, indent=2)
        tmp_metadata.write_text(text, encoding="utf-8")
        os.replace(tmp_metadata, metadata_path)  # the last operation: marks success
    finally:
        tmp_signals.unlink(missing_ok=True)
        tmp_metadata.unlink(missing_ok=True)


def run_ensemble_signals(
    paths: DataPaths, config_path: Path = DEFAULT_CONFIG, years: Iterable[int] | None = None, log=print
) -> list[dict]:
    """Average the four family ranks of the selected OOS years (all by default)."""
    replicas = configured_replicas(config_path)
    baseline_years = [e.oos_year for e in baseline_episodes(config_path)]
    if years is not None:
        missing = sorted(set(years) - set(baseline_years))
        if missing:
            raise ValueError(f"not baseline OOS years: {missing}")
        baseline_years = [y for y in baseline_years if y in set(years)]

    summaries = []
    for oos_year in baseline_years:
        (paths.ensemble_signals / f"{oos_year}.json").unlink(missing_ok=True)  # not complete until rewritten
        remove_between_family_disagreement(paths, oos_year)
        ranks = {family: load_family_ranks(paths, family, oos_year, replicas) for family in FAMILIES}
        signals = ensemble_signals(ranks)
        expected, observed = month_range(oos_year * 100 + 1, oos_year * 100 + 12), set(signals["yyyymm"].tolist())
        if observed != set(expected):
            missing, extra = sorted(set(expected) - observed), sorted(observed - set(expected))
            raise DataValidationError(f"{oos_year}: not the 12 OOS months; missing {missing}, extra {extra}.")
        sources = {family: paths.family_signals / family / f"{oos_year}.parquet" for family in FAMILIES}
        metadata = {
            "oos_year": oos_year,
            "families": list(FAMILIES),
            "n_stock_months": len(signals),
            "months": sorted(int(m) for m in signals["yyyymm"].unique()),
            "family_signals_sha256": {f: hashlib.sha256(p.read_bytes()).hexdigest() for f, p in sources.items()},
        }
        _write(paths.ensemble_signals, oos_year, signals, metadata)
        log(f"{oos_year}: {len(signals):,} stock-months")
        summaries.append(metadata)
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m disagreement.signals.ensemble")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--year", action="append", type=int, help="OOS year, repeatable; default: all")
    args = parser.parse_args(argv)
    try:
        run_ensemble_signals(DataPaths(args.data_dir), args.config, args.year)
    except DataValidationError as exc:
        print(f"Data validation failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
