"""Build the formation-time universe and predictor panel from the local extracts.

Licensed rows are read and written under data/ only. The build stops on any
violated check; data/diagnostics/summary.json exists only after a successful
build.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import crsp, osap
from .config import DataConfig, DataPaths
from .panel import join_universe_osap
from .universe import UNIVERSE_COLUMNS, build_universe
from .validation import KEY, DataValidationError, isin_sorted, make_keys, require_unique_keys

COMPRESSION = "zstd"


def _year(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def check_osap(path: Path, cfg: DataConfig) -> list[str]:
    """The pinned OSAP release: exact file, predictor count and column names."""
    if not path.exists():
        raise DataValidationError(f"OSAP file not found: {path} (see data/README.md).")
    if osap.sha256_file(path) != cfg.osap_sha256:
        raise DataValidationError(f"OSAP file does not match the SHA-256 of release {cfg.osap_release}.")
    signals = osap.signal_columns(osap.read_header(path))
    if len(signals) != cfg.osap_predictors:
        raise DataValidationError(f"OSAP: expected {cfg.osap_predictors} predictors, found {len(signals)}.")
    reserved = {c.lower() for c in (*UNIVERSE_COLUMNS, *crsp.CRSP_PREDICTORS)}
    clash = sorted(s for s in signals if s.lower() in reserved)
    if clash:
        raise DataValidationError(f"OSAP predictors collide with panel columns: {', '.join(clash)}")
    return signals


def _crsp_extract(paths: DataPaths, cfg: DataConfig) -> tuple[list[Path], dict]:
    """The yearly CRSP files, checked against the extraction manifest."""
    directory = paths.crsp(cfg.crsp_table)
    manifest_path = directory / "_manifest.json"
    if not manifest_path.exists():
        raise DataValidationError("no CRSP extract found; run crsp-extract first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest["library"], manifest["table"]) != (cfg.crsp_library, cfg.crsp_table) or set(
        manifest["fields"]
    ) != set(crsp.REQUIRED_FIELDS):
        raise DataValidationError("CRSP manifest: unexpected source table or fields; rerun crsp-extract.")
    files = sorted(directory.glob("crsp_monthly_*.parquet"))
    if sorted(str(_year(f)) for f in files) != sorted(manifest["rows_by_year"]):
        raise DataValidationError("CRSP: the yearly files differ from the manifest; rerun crsp-extract.")
    return files, manifest


def build_universe_files(paths: DataPaths, cfg: DataConfig, files: list[Path], manifest: dict) -> pd.DataFrame:
    """U_t from the CRSP extract, one file per year; returns the monthly counts."""
    paths.universe.mkdir(parents=True, exist_ok=True)
    tables, keys = [], []
    for path in files:
        year = _year(path)
        raw = pd.read_parquet(path)
        if len(raw) != manifest["rows_by_year"][str(year)]:
            raise DataValidationError(f"CRSP {year}: row count differs from the manifest; rerun crsp-extract.")
        formation = crsp.normalize_monthly(raw)
        n_other = int((formation["yyyymm"] // 100 != year).sum())
        if n_other:
            raise DataValidationError(f"CRSP {year}: {n_other} rows fall outside the file's year.")
        keys.append(make_keys(formation["permno"].to_numpy(), formation["yyyymm"].to_numpy()))
        members, by_month = build_universe(
            formation,
            exchanges=cfg.exchanges,
            breakpoint_exchange=cfg.breakpoint_exchange,
            quantile=cfg.breakpoint_quantile,
        )
        members.to_parquet(paths.universe / f"universe_{year}.parquet", index=False, compression=COMPRESSION)
        tables.append(by_month)
    keys = np.concatenate(keys)
    n_dup = int(keys.size - np.unique(keys).size)
    if n_dup:
        raise DataValidationError(f"CRSP: {n_dup} rows repeat a permno/yyyymm key across files.")
    by_month = pd.concat(tables, ignore_index=True).sort_values("yyyymm", ignore_index=True)

    months = by_month["yyyymm"].to_numpy()
    first, last = int(months[0]), int(months[-1])
    expected = tuple(int(manifest[k][:4] + manifest[k][5:7]) for k in ("table_first_date", "table_last_date"))
    if (first, last) != expected:
        raise DataValidationError(f"CRSP: data span {first}-{last} differs from the manifest {expected[0]}-{expected[1]}.")
    span = (last // 100 - first // 100) * 12 + last % 100 - first % 100 + 1
    n_missing = span - len(np.unique(months))
    if n_missing:
        raise DataValidationError(f"CRSP: {n_missing} months between {first} and {last} have no records.")
    return by_month


def _stage_osap(osap_path: Path, universe_keys: np.ndarray, signals: list[str], staging: Path, block_size: int) -> None:
    """Stream OSAP once: reject duplicate keys and stage the rows of U_t members by year."""
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    writers: dict[int, pq.ParquetWriter] = {}
    schema, all_keys = None, []
    try:
        for frame in osap.iter_frames(osap_path, block_size):
            keys = make_keys(frame["permno"].to_numpy(), frame["yyyymm"].to_numpy())
            all_keys.append(keys)
            mask = isin_sorted(keys, universe_keys)
            if not mask.any():
                continue
            matched = frame.loc[mask, [*KEY, *signals]]
            table = pa.Table.from_pandas(matched, preserve_index=False)
            if schema is None:
                schema = table.schema
            years = matched["yyyymm"].to_numpy() // 100
            for year in np.unique(years):
                if year not in writers:
                    writers[year] = pq.ParquetWriter(staging / f"osap_{year}.parquet", schema, compression=COMPRESSION)
                writers[year].write_table(table.cast(schema).filter(pa.array(years == year)))
    finally:
        for writer in writers.values():
            writer.close()
    keys = np.concatenate(all_keys) if all_keys else np.empty(0, dtype="int64")
    n_dup = int(keys.size - np.unique(keys).size)
    if n_dup:
        raise DataValidationError(f"OSAP: {n_dup} rows repeat a permno/yyyymm key.")


def build_panel_files(paths: DataPaths, cfg: DataConfig, osap_path: Path, signals: list[str], block_size: int) -> dict:
    """Left-join U_t with the OSAP signals and the three CRSP predictors, one file per year."""
    predictors = [*signals, *crsp.CRSP_PREDICTORS]
    universe_files = sorted(paths.universe.glob("universe_*.parquet"))
    if not universe_files:
        raise DataValidationError("no universe files were built.")
    universe_keys = np.sort(
        np.concatenate(
            [make_keys(*pd.read_parquet(f, columns=list(KEY)).to_numpy().T) for f in universe_files]
        )
    )
    _stage_osap(osap_path, universe_keys, signals, paths.staging, block_size)

    paths.panel.mkdir(parents=True, exist_ok=True)
    empty = pd.DataFrame({c: pd.Series(dtype="int64" if c in KEY else "float64") for c in [*KEY, *signals]})
    rows = matched = infinite = 0
    usable = np.zeros(len(predictors), dtype="int64")
    for path in universe_files:
        year = _year(path)
        members = pd.read_parquet(path)
        staged = paths.staging / f"osap_{year}.parquet"
        panel = join_universe_osap(members, pd.read_parquet(staged) if staged.exists() else empty, signals)
        matched += int(panel.pop("osap_matched").sum())  # join rate only; not stored
        raw = pd.read_parquet(paths.crsp(cfg.crsp_table) / f"crsp_monthly_{year}.parquet")
        panel = panel.merge(crsp.crsp_predictors(raw), on=list(KEY), how="left", validate="one_to_one")
        if len(panel) != len(members):
            raise DataValidationError("adding the CRSP predictors changed the panel rows.")
        values = panel[predictors].to_numpy(dtype="float64")
        is_inf = np.isinf(values)
        infinite += int(is_inf.sum())
        values[is_inf] = np.nan  # +/- infinity is treated as missing
        panel[predictors] = values
        require_unique_keys(panel, "panel")
        panel.to_parquet(paths.panel / f"panel_{year}.parquet", index=False, compression=COMPRESSION)
        rows += len(panel)
        usable += (~np.isnan(values)).sum(axis=0)
    shutil.rmtree(paths.staging, ignore_errors=True)
    return {
        "panel_rows": rows,
        "osap_join_rate": matched / rows if rows else None,
        "candidate_predictors": len(predictors),
        "predictors_without_usable_values": [p for p, n in zip(predictors, usable) if n == 0],
        "infinite_values_set_missing": infinite,
    }


def remove_derived_outputs(paths: DataPaths) -> None:
    """Universe, panel, target and their diagnostics, all derived from the CRSP extract."""
    for stale in (paths.diagnostics / "summary.json", paths.diagnostics / "universe_by_month.csv"):
        stale.unlink(missing_ok=True)
    for directory in (paths.universe, paths.panel, paths.target):
        if directory.exists():
            shutil.rmtree(directory)


def build(paths: DataPaths, cfg: DataConfig, osap_path: Path | None = None, block_size: int = osap.DEFAULT_BLOCK_SIZE) -> dict:
    """Universe, panel, universe_by_month.csv and summary.json.

    The inputs are checked before any output is touched. Previous outputs are
    then removed, and the two diagnostics files are written only once every
    check has passed, so a failed build leaves neither a mix of old and new
    outputs nor diagnostics that look current.
    """
    osap_path = osap_path or paths.osap / cfg.osap_file
    signals = check_osap(osap_path, cfg)
    files, manifest = _crsp_extract(paths, cfg)

    summary_path = paths.diagnostics / "summary.json"
    by_month_path = paths.diagnostics / "universe_by_month.csv"
    remove_derived_outputs(paths)

    by_month = build_universe_files(paths, cfg, files, manifest)
    n_no_breakpoint = int((by_month["n_nyse_eligible"] == 0).sum())
    if n_no_breakpoint:
        raise DataValidationError(f"{n_no_breakpoint} months have no eligible NYSE stock, hence no breakpoint.")
    panel = build_panel_files(paths, cfg, osap_path, signals, block_size)
    if panel["panel_rows"] != int(by_month["n_universe"].sum()):
        raise DataValidationError("the panel does not have exactly one row per universe member.")

    summary = {
        "validation": "PASS",
        "first_month": int(by_month["yyyymm"].min()),
        "last_month": int(by_month["yyyymm"].max()),
        "months": int(len(by_month)),
        "security_months": panel["panel_rows"],
        "duplicate_keys": 0,  # any duplicate stops the build
        **{k: v for k, v in panel.items() if k != "panel_rows"},
        "crsp_table": f"{cfg.crsp_library}.{cfg.crsp_table}",
        "osap_release": cfg.osap_release,
        "osap_sha256": cfg.osap_sha256,
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    paths.diagnostics.mkdir(parents=True, exist_ok=True)
    by_month.to_csv(by_month_path, index=False)
    tmp = summary_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    os.replace(tmp, summary_path)
    return summary
