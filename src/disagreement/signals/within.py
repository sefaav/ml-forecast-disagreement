"""Within-family estimation uncertainty from the bootstrap replica ranks.

For each family, stock and OOS formation month, the uncertainty is the sample
standard deviation (ddof = 1) of the K replica ranks: the replicas are
bootstrap draws approximating estimation uncertainty. Each family's standard
deviations are ranked across the month's stocks (average ranks for ties) and
mapped to [0, 1] as in the family signals, separately by family and month.
``within_family_uncertainty_raw`` is the equal-weight mean of the four
family uncertainty ranks; it is not ranked again and not yet conditioned on
the ensemble signal. Only the saved family signals are read: no prediction,
target, return or model is involved.

The replica ranks of each stock-month are sorted before the standard
deviation is computed, so stocks with the same replica ranks in a different
replica order tie exactly.

Command line: ``python -m disagreement.signals.within [--year Y]``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd

from disagreement.data.config import DEFAULT_CONFIG, DEFAULT_DATA_DIR, DataPaths
from disagreement.data.validation import DataValidationError, require_unique_keys
from disagreement.models.selection import FAMILIES
from disagreement.signals.ensemble import family_ranks
from disagreement.signals.family import configured_replicas, remove_conditional_disagreement, unit_rank
from disagreement.walkforward.calendar import baseline_episodes, month_range

OUTPUT_COLUMNS = (
    "permno",
    "yyyymm",
    *(f"{family}_uncertainty_sd" for family in FAMILIES),
    *(f"{family}_uncertainty_rank" for family in FAMILIES),
    "within_family_uncertainty_raw",
)


def uncertainty_sd(signals: pd.DataFrame, replicas: int) -> pd.Series:
    """Sample SD (ddof = 1) of the replica ranks of each stock-month, indexed by (permno, yyyymm)."""
    ordered = signals.sort_values(["yyyymm", "permno", "replica_rank"], ignore_index=True)
    values = ordered["replica_rank"].to_numpy(dtype="float64").reshape(-1, replicas)  # one sorted row per stock-month
    index = pd.MultiIndex.from_frame(ordered.iloc[::replicas][["permno", "yyyymm"]])
    return pd.Series(values.std(axis=1, ddof=1), index=index)


def within_family_uncertainty(signals: Mapping[str, pd.DataFrame], replicas: int) -> pd.DataFrame:
    """Family SDs, their monthly ranks and the mean of the four ranks, one row per stock-month."""
    sds = {family: uncertainty_sd(signals[family], replicas) for family in FAMILIES}
    keys = sds[FAMILIES[0]].index
    for family in FAMILIES[1:]:
        other = sds[family].index
        if not other.equals(keys):
            missing, extra = len(keys.difference(other)), len(other.difference(keys))
            raise DataValidationError(
                f"{family} covers different stock-months than {FAMILIES[0]}: {missing} missing, {extra} extra."
            )
    out = keys.to_frame(index=False)
    for family in FAMILIES:
        out[f"{family}_uncertainty_sd"] = sds[family].to_numpy()
    for family in FAMILIES:  # ranked within each family and month only
        out[f"{family}_uncertainty_rank"] = unit_rank(out[f"{family}_uncertainty_sd"], [out["yyyymm"]])
    out["within_family_uncertainty_raw"] = (
        out["elastic_net_uncertainty_rank"]
        + out["random_forest_uncertainty_rank"]
        + out["lightgbm_uncertainty_rank"]
        + out["mlp_uncertainty_rank"]
    ) / 4.0
    out = out.sort_values(["yyyymm", "permno"], ignore_index=True).loc[:, list(OUTPUT_COLUMNS)]
    require_unique_keys(out, "within-family uncertainty")
    return out


def load_family_signals(paths: DataPaths, family: str, oos_year: int, replicas: int) -> pd.DataFrame:
    """A complete family-year of family signals, checked as for the ensemble."""
    directory = paths.family_signals / family
    signals_path, marker = directory / f"{oos_year}.parquet", directory / f"{oos_year}.json"
    if not (signals_path.exists() and marker.exists()):
        raise DataValidationError(f"{family} {oos_year}: no complete family signals (parquet and JSON marker).")
    label = f"{family} {oos_year} family signals"
    signals = pd.read_parquet(signals_path)
    keys = family_ranks(signals, family, replicas, label).index  # schema, family, keys, replicas, rank bounds
    if not (keys.get_level_values("yyyymm") // 100 == oos_year).all():
        raise DataValidationError(f"{label}: rows outside the OOS year.")
    return signals


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


def run_within_family_uncertainty(
    paths: DataPaths, config_path: Path = DEFAULT_CONFIG, years: Iterable[int] | None = None, log=print
) -> list[dict]:
    """Within-family uncertainty of the selected OOS years (all by default)."""
    replicas = configured_replicas(config_path)
    if replicas < 2:
        raise ValueError("a sample standard deviation needs at least two bootstrap replicas")
    baseline_years = [e.oos_year for e in baseline_episodes(config_path)]
    if years is not None:
        missing = sorted(set(years) - set(baseline_years))
        if missing:
            raise ValueError(f"not baseline OOS years: {missing}")
        baseline_years = [y for y in baseline_years if y in set(years)]

    summaries = []
    for oos_year in baseline_years:
        (paths.within_family_uncertainty / f"{oos_year}.json").unlink(missing_ok=True)  # not complete until rewritten
        remove_conditional_disagreement(paths, oos_year)
        signals = {family: load_family_signals(paths, family, oos_year, replicas) for family in FAMILIES}
        uncertainty = within_family_uncertainty(signals, replicas)
        expected, observed = month_range(oos_year * 100 + 1, oos_year * 100 + 12), set(uncertainty["yyyymm"].tolist())
        if observed != set(expected):
            missing, extra = sorted(set(expected) - observed), sorted(observed - set(expected))
            raise DataValidationError(f"{oos_year}: not the 12 OOS months; missing {missing}, extra {extra}.")
        sources = {family: paths.family_signals / family / f"{oos_year}.parquet" for family in FAMILIES}
        metadata = {
            "oos_year": oos_year,
            "replicas": replicas,
            "n_stock_months": len(uncertainty),
            "months": sorted(int(m) for m in uncertainty["yyyymm"].unique()),
            "family_signals_sha256": {f: hashlib.sha256(p.read_bytes()).hexdigest() for f, p in sources.items()},
        }
        _write(paths.within_family_uncertainty, oos_year, uncertainty, metadata)
        log(f"{oos_year}: {len(uncertainty):,} stock-months")
        summaries.append(metadata)
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m disagreement.signals.within")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--year", action="append", type=int, help="OOS year, repeatable; default: all")
    args = parser.parse_args(argv)
    try:
        run_within_family_uncertainty(DataPaths(args.data_dir), args.config, args.year)
    except DataValidationError as exc:
        print(f"Data validation failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
