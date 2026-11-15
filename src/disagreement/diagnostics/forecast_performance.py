"""Baseline OOS forecasting diagnostics: monthly rank IC of the four family ranks and the ensemble signal.

Descriptive evaluation of the saved forecasts; it tests no hypothesis and is separate from the H1-H3
inference. The metric is the frozen model-selection score (pre_analysis_plan.md, "Model selection";
``disagreement.models.selection.validation_score``), computed month by month: in every OOS formation month, the
stocks of U_t with a finite saved ``target_rank`` are kept, the signal is average-ranked among them (ties share
their rank) and correlated (Pearson) with ``target_rank``; a constant signal scores 0; a month without a finite
target or with a constant target, or a non-finite signal value, is an error. The saved family ranks and the
ensemble signal S (the equal-weight mean of the four family ranks, checked by ``load_ensemble``) are ranked again
on that subset for the correlation only; they are never modified or saved. The same scored stocks are used for
the five signals of a month, and every month weighs the same in the summaries.

Published (``data/diagnostics/forecast_performance/``), written to a staging directory and renamed into place,
with ``summary.json`` written last:

* ``monthly_ic.csv``: one row per formation month and signal: rank IC, whether the scored signal was constant,
  the number of U_t members and of scored stocks;
* ``summary.json``: per signal the mean, median and sample SD (ddof = 1) of the monthly IC, the shares of
  positive, zero and negative months, the constant and evaluated months; the outcome coverage; the SHA-256 of the
  configuration and of every input file, each hashed before and after it is read and all checked again just
  before publication.

No stock-level value is written or printed. An existing result set is never replaced. Without ``--execute`` the
command only reads the configuration and checks that the input files exist.

Command line: ``python -m disagreement.diagnostics.forecast_performance [--execute]``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from disagreement.data.config import DEFAULT_CONFIG, DEFAULT_DATA_DIR, REPO_ROOT, DataPaths
from disagreement.data.target import TARGET_COLUMNS
from disagreement.data.validation import KEY, DataValidationError
from disagreement.inference.inputs import _check_target, _keyed, load_target
from disagreement.signals.between import load_ensemble
from disagreement.walkforward.calendar import baseline_episodes, month_range

SIGNALS = {  # published name -> saved column of the ensemble file
    "elastic_net": "elastic_net_rank",
    "random_forest": "random_forest_rank",
    "lightgbm": "lightgbm_rank",
    "mlp": "mlp_rank",
    "ensemble": "ensemble_signal",
}
MONTHLY = "monthly_ic.csv"
SUMMARY = "summary.json"
METRIC = ("monthly rank IC as in disagreement.models.selection.validation_score: Pearson correlation of the "
          "signal's average ranks among the stocks with a finite target_rank with target_rank; 0 for a constant "
          "signal; equal month weights")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unchanged(path: Path, recorded: str) -> bool:
    """The file still exists, can be read and has the recorded SHA-256."""
    try:
        return path.is_file() and _sha256(path) == recorded
    except OSError:
        return False


def monthly_rank_ic(yyyymm, prediction, target) -> pd.DataFrame:
    """Monthly rank IC with the rules of validation_score: yyyymm, rank_ic, constant_signal, n_evaluated."""
    frame = pd.DataFrame({"yyyymm": np.asarray(yyyymm), "prediction": np.asarray(prediction, dtype="float64"),
                          "target": np.asarray(target, dtype="float64")})
    months = frame["yyyymm"].unique()
    frame = frame[np.isfinite(frame["target"])]
    n_empty = len(months) - frame["yyyymm"].nunique()
    if n_empty or frame.empty:
        raise DataValidationError(f"{n_empty or len(months)} months have no observation with a finite target.")
    n_bad = int((~np.isfinite(frame["prediction"])).sum())
    if n_bad:
        raise DataValidationError(f"{n_bad} scored observations have a non-finite signal value.")
    rows = []
    for month, scored in frame.groupby("yyyymm", sort=True):
        target = scored["target"].to_numpy()
        if np.ptp(target) == 0:
            raise DataValidationError(f"{month}: target_rank has no cross-sectional variation.")
        rank = scored["prediction"].rank(method="average").to_numpy()
        constant = bool(np.ptp(rank) == 0)
        ic = 0.0 if constant else float(np.corrcoef(rank, target)[0, 1])
        rows.append((int(month), ic, constant, len(scored)))  # type: ignore[arg-type]  # yyyymm is an integer key
    return pd.DataFrame(rows, columns=["yyyymm", "rank_ic", "constant_signal", "n_evaluated"])


def summarise(ic: pd.Series, constant: pd.Series) -> dict:
    """Equal-weight descriptive statistics of one signal's monthly IC (no test, no interval)."""
    values = ic.to_numpy(dtype="float64")
    return {
        "mean_rank_ic": float(np.mean(values)), "median_rank_ic": float(np.median(values)),
        "sd_rank_ic": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        "share_positive": float((values > 0).mean()), "share_zero": float((values == 0).mean()),
        "share_negative": float((values < 0).mean()),
        "months_constant_signal": int(constant.sum()), "months_evaluated": int(len(values)),
    }


def baseline_calendar(config_path: Path) -> tuple[list[int], list[int]]:
    """The frozen OOS years and their consecutive formation months."""
    years = [e.oos_year for e in baseline_episodes(config_path)]
    grid = month_range(years[0] * 100 + 1, years[-1] * 100 + 12)
    if len(grid) != 12 * len(years) or years != list(range(years[0], years[-1] + 1)):
        raise DataValidationError("the baseline episodes do not form consecutive complete OOS years.")
    return years, grid


def year_files(paths: DataPaths, year: int) -> dict[str, Path]:
    return {f"{year}/ensemble_signals": paths.ensemble_signals / f"{year}.parquet",
            f"{year}/ensemble_signals_marker": paths.ensemble_signals / f"{year}.json",
            f"{year}/target": paths.target / f"target_{year}.parquet"}


def load_year(paths: DataPaths, year: int) -> pd.DataFrame:
    """The five signals and the saved outcomes of one OOS year, validated and aligned one to one on U_t."""
    ensemble = load_ensemble(paths, year)  # marker, schema, unique keys, ranges, 12 months, S = exact mean
    ensemble = _keyed(ensemble, tuple(ensemble.columns), year, f"{year} ensemble signals")
    target = _keyed(load_target(paths, year), TARGET_COLUMNS, year, f"{year} target")  # unique keys, 12 months
    _check_target(target, f"{year} target")  # missingness, range, the frozen transformation of ret_next
    keys, other = pd.MultiIndex.from_frame(ensemble[list(KEY)]), pd.MultiIndex.from_frame(target[list(KEY)])
    if not keys.equals(other):
        missing, extra = len(keys.difference(other)), len(other.difference(keys))
        raise DataValidationError(f"{year}: the target covers different stock-months than the ensemble signals: "
                                  f"{missing} missing, {extra} extra.")
    out = ensemble.loc[:, [*KEY, *SIGNALS.values()]].copy()
    out["target_rank"] = target["target_rank"].to_numpy(dtype="float64", na_value=np.nan)
    return out


def evaluate(frame: pd.DataFrame, grid: list[int]) -> pd.DataFrame:
    """Monthly IC of the five signals on the same scored stocks, with the U_t and scored counts per month."""
    if sorted(frame["yyyymm"].unique().tolist()) != grid:
        raise DataValidationError("the evaluated months are not the frozen OOS calendar.")
    universe = frame.groupby("yyyymm").size()
    tables = []
    for name, column in SIGNALS.items():
        ic = monthly_rank_ic(frame["yyyymm"], frame[column], frame["target_rank"])
        ic.insert(1, "signal", name)
        ic["n_universe"] = universe.reindex(ic["yyyymm"]).to_numpy()
        tables.append(ic)
    return pd.concat(tables, ignore_index=True).loc[
        :, ["yyyymm", "signal", "rank_ic", "constant_signal", "n_universe", "n_evaluated"]]


def coverage(monthly: pd.DataFrame) -> dict:
    one = monthly[monthly["signal"] == "ensemble"]
    share = one["n_evaluated"] / one["n_universe"]
    return {"stock_months_universe": int(one["n_universe"].sum()),
            "stock_months_evaluated": int(one["n_evaluated"].sum()),
            "share_evaluated": float(one["n_evaluated"].sum() / one["n_universe"].sum()),
            "mean_monthly_share_evaluated": float(share.mean()), "min_monthly_share_evaluated": float(share.min()),
            "months_below_95_percent": int((share < 0.95).sum())}


def default_output_dir(paths: DataPaths) -> Path:
    return paths.diagnostics / "forecast_performance"


def run(paths: DataPaths, config_path: Path = DEFAULT_CONFIG) -> dict:
    """Validate, evaluate and publish the complete baseline; an existing result set is never replaced."""
    destination = default_output_dir(paths)
    if os.path.lexists(destination):
        raise DataValidationError(f"{destination.name}: a result set already exists; remove it to rerun.")
    config_hash = _sha256(config_path)
    years, grid = baseline_calendar(config_path)
    frames, inputs = [], {}
    for year in years:
        files = year_files(paths, year)
        before = {key: _sha256(path) for key, path in files.items()}
        frames.append(load_year(paths, year))
        changed = sorted(key for key, path in files.items() if _sha256(path) != before[key])
        if changed:
            raise DataValidationError(f"{year}: input files changed while they were read: {changed}.")
        inputs |= {key: {"path": path.relative_to(paths.root).as_posix(), "sha256": before[key]}
                   for key, path in files.items()}
    monthly = evaluate(pd.concat(frames, ignore_index=True), grid)
    if not _unchanged(config_path, config_hash):
        raise DataValidationError("the configuration is missing, unreadable or changed during the run.")
    changed = sorted(key for key, entry in inputs.items()
                     if not _unchanged(paths.root / entry["path"], entry["sha256"]))
    if changed:  # every recorded input, again, just before publication
        raise DataValidationError(f"input files missing, unreadable or changed since they were read: {changed}.")
    summary = {
        "status": "completed",
        "meaning": "descriptive OOS forecasting diagnostics; no hypothesis test, not an H1-H3 result",
        "metric": METRIC,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "grid": {"first": grid[0], "last": grid[-1], "n_months": len(grid), "n_years": len(years)},
        "signals": {name: summarise(g["rank_ic"], g["constant_signal"])
                    for name, g in monthly.groupby("signal", sort=False)},
        "coverage": coverage(monthly),
        "inputs": {"config": {"path": _portable(config_path), "sha256": config_hash}, "files": inputs},
    }
    _publish(destination, monthly, summary)
    return summary


def _portable(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _publish(destination: Path, monthly: pd.DataFrame, summary: dict) -> None:
    """Staging directory beside the destination, summary.json last, then one rename; nothing left on failure."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        monthly.to_csv(staging / MONTHLY, index=False, float_format="%.17g", lineterminator="\n")
        (staging / SUMMARY).write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        os.rename(staging, destination)  # fails if the destination appeared meanwhile: never replaced
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m disagreement.diagnostics.forecast_performance",
                                     description="Descriptive OOS rank IC of the four family ranks and the ensemble.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--execute", action="store_true", help="read the saved outcomes, evaluate and publish")
    args = parser.parse_args(argv)
    paths = DataPaths(args.data_dir)
    try:
        years, grid = baseline_calendar(args.config)
        if not args.execute:
            missing = sum(not p.exists() for y in years for p in year_files(paths, y).values())
            print(f"Dry run: {len(grid)} months ({grid[0]}-{grid[-1]}, {len(years)} OOS years); {missing} input files "
                  f"missing; results would go to {default_output_dir(paths)}. Nothing was loaded; pass --execute.")
            return 0
        summary = run(paths, args.config)
    except (DataValidationError, OSError, ValueError, KeyError) as exc:
        print(f"Forecast diagnostics not published: {exc}", file=sys.stderr)
        return 2
    print(f"Published {summary['grid']['n_months']} months x {len(SIGNALS)} signals to {default_output_dir(paths)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
