"""Mechanical-bias diagnostics: does signal-bin conditioning remove the bound on between-family dispersion?

The four family ranks lie in [0, 1], so their dispersion around the ensemble
signal S is mechanically limited when S is near 0 or 1. For diagnosis only,
with the centred signal S^c = 2 (S - 0.5) and the signal extremity |S^c|,
four Pearson correlations are computed in every OOS formation month:

    raw_vs_signal             corr(between_family_disagreement_raw, S)
    raw_vs_extremity          corr(between_family_disagreement_raw, |S^c|)
    conditional_vs_signal     corr(conditional_between_family_disagreement, S)
    conditional_vs_extremity  corr(conditional_between_family_disagreement, |S^c|)

They are summarised across months with equal month weights, and computed
again inside every (month, signal bin), which gives an equal-weight profile
of the 20 signal bins (bins 1 and 20 matter most). A correlation is NaN, never
0, with fewer than two stocks or a constant variable. Only the saved
conditional-disagreement files are read: no target or return. The output is
descriptive: there is no test, threshold or automatic change of the design.

Command line: ``python -m disagreement.diagnostics.mechanical_bias``
(the complete baseline OOS sample).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import SupportsInt, cast

import numpy as np
import pandas as pd

from disagreement.data.config import DEFAULT_CONFIG, DEFAULT_DATA_DIR, DataPaths
from disagreement.data.validation import DataValidationError
from disagreement.signals import conditional
from disagreement.signals.conditional import SIGNAL_BINS, signal_bins
from disagreement.walkforward.calendar import baseline_episodes

RELATIONSHIPS = {  # name: (disagreement, signal variable)
    "raw_vs_signal": ("between_family_disagreement_raw", "ensemble_signal"),
    "raw_vs_extremity": ("between_family_disagreement_raw", "signal_extremity"),
    "conditional_vs_signal": ("conditional_between_family_disagreement", "ensemble_signal"),
    "conditional_vs_extremity": ("conditional_between_family_disagreement", "signal_extremity"),
}
USED_COLUMNS = (
    "permno", "yyyymm", "ensemble_signal", "signal_bin",
    "between_family_disagreement_raw", "conditional_between_family_disagreement",
)
MEANS = {
    "mean_ensemble_signal": "ensemble_signal",
    "mean_signal_extremity": "signal_extremity",
    "mean_raw_disagreement": "between_family_disagreement_raw",
    "mean_conditional_disagreement": "conditional_between_family_disagreement",
}
MONTHLY_COLUMNS = ("yyyymm", "n_stocks", *RELATIONSHIPS)
MONTHLY_BIN_COLUMNS = ("yyyymm", "signal_bin", "n_stocks", *MEANS, *RELATIONSHIPS)
BIN_SUMMARY_COLUMNS = (
    "signal_bin", "n_months", "total_stock_months", "mean_n_stocks", *MEANS,
    *(f"{name}_{stat}" for name in RELATIONSHIPS for stat in ("n_valid", "mean", "median", "mean_abs")),
)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation; NaN with fewer than two observations or a constant variable."""
    x, y = np.asarray(x, dtype="float64"), np.asarray(y, dtype="float64")
    if len(x) < 2 or x.min() == x.max() or y.min() == y.max():
        return float("nan")
    dx, dy = x - x.mean(), y - y.mean()
    return float(dx @ dy / np.sqrt((dx @ dx) * (dy @ dy)))


def with_extremity(frame: pd.DataFrame) -> pd.DataFrame:
    """The used columns plus the centred signal 2 (S - 0.5) and the extremity |S^c|, for diagnosis only."""
    out = frame.loc[:, list(USED_COLUMNS)].copy()
    out["centered_signal"] = 2.0 * (out["ensemble_signal"] - 0.5)
    out["signal_extremity"] = out["centered_signal"].abs()
    return out


def _correlations(group: pd.DataFrame) -> dict:
    return {name: pearson(group[d].to_numpy(), group[s].to_numpy()) for name, (d, s) in RELATIONSHIPS.items()}


def monthly_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    """The four correlations over each month's full cross-section, one row per month."""
    rows = [
        {"yyyymm": int(cast(SupportsInt, month)), "n_stocks": len(group), **_correlations(group)}
        for month, group in frame.groupby("yyyymm", sort=True)
    ]
    return pd.DataFrame(rows, columns=list(MONTHLY_COLUMNS))


def monthly_bin_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    """Means and the four correlations inside each observed (month, signal bin)."""
    rows = [
        {
            "yyyymm": int(cast(SupportsInt, month)),
            "signal_bin": int(cast(SupportsInt, signal_bin)),
            "n_stocks": len(group),
            **{name: float(group[column].mean()) for name, column in MEANS.items()},
            **_correlations(group),
        }
        for (month, signal_bin), group in frame.groupby(["yyyymm", "signal_bin"], sort=True)
    ]
    return pd.DataFrame(rows, columns=list(MONTHLY_BIN_COLUMNS))


def _stats(values: pd.Series) -> dict:
    """Summary of the finite values only; NaN when there is none."""
    v = values.to_numpy(dtype="float64")
    v = v[np.isfinite(v)]
    if not len(v):
        return {"n_valid": 0, "mean": np.nan, "median": np.nan, "mean_abs": np.nan, "q05": np.nan, "q95": np.nan}
    q05, q95 = np.quantile(v, [0.05, 0.95])
    return {
        "n_valid": len(v),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "mean_abs": float(np.abs(v).mean()),
        "q05": float(q05),
        "q95": float(q95),
    }


def signal_bin_summary(monthly_bins: pd.DataFrame) -> pd.DataFrame:
    """One row per signal bin; every month-bin has the same weight, whatever its number of stocks."""
    rows = []
    for signal_bin, group in monthly_bins.groupby("signal_bin", sort=True):
        row = {
            "signal_bin": int(cast(SupportsInt, signal_bin)),
            "n_months": len(group),
            "total_stock_months": int(group["n_stocks"].sum()),
            "mean_n_stocks": float(group["n_stocks"].mean()),
            **{name: float(group[name].mean()) for name in MEANS},
        }
        for name in RELATIONSHIPS:
            stats = _stats(group[name])
            row |= {f"{name}_{stat}": stats[stat] for stat in ("n_valid", "mean", "median", "mean_abs")}
        rows.append(row)
    return pd.DataFrame(rows, columns=list(BIN_SUMMARY_COLUMNS))


def correlation_summary(monthly: pd.DataFrame) -> dict:
    """Equal-weight summary of each monthly correlation series; None where undefined."""
    summary = {}
    for name in RELATIONSHIPS:
        stats = _stats(monthly[name])
        values = {
            "n_valid": stats["n_valid"],
            "mean_correlation": stats["mean"],
            "median_correlation": stats["median"],
            "mean_absolute_correlation": stats["mean_abs"],
            "q05": stats["q05"],
            "q95": stats["q95"],
        }
        summary[name] = {k: (v if k == "n_valid" or np.isfinite(v) else None) for k, v in values.items()}
    return summary


def load_conditional(paths: DataPaths, oos_year: int) -> pd.DataFrame:
    """A complete conditional-disagreement year, with its signal bins checked against the ensemble signal."""
    frame = conditional._load(
        paths.conditional_disagreement, oos_year, conditional.OUTPUT_COLUMNS, "conditional disagreement"
    )  # marker, exact schema, unique keys, the twelve OOS months
    label = f"{oos_year} conditional disagreement"
    bounds = {
        "ensemble_signal": 1.0,
        "between_family_disagreement_raw": 0.5,
        "conditional_between_family_disagreement": 1.0,
    }
    for column, high in bounds.items():
        values = frame[column].to_numpy(dtype="float64")
        if not (np.isfinite(values).all() and (values >= 0.0).all() and (values <= high).all()):
            raise DataValidationError(f"{label}: {column} must be finite and in [0, {high}].")
    if not (pd.api.types.is_integer_dtype(frame["signal_bin"]) and frame["signal_bin"].between(1, SIGNAL_BINS).all()):
        raise DataValidationError(f"{label}: signal_bin must be an integer in [1, {SIGNAL_BINS}].")
    differs = int((signal_bins(frame["ensemble_signal"], frame["yyyymm"], SIGNAL_BINS) != frame["signal_bin"]).sum())
    if differs:
        raise DataValidationError(f"{label}: signal_bin differs from the bins of ensemble_signal in {differs} rows.")
    return frame


def _write(directory: Path, tables: dict[str, pd.DataFrame], summary: dict) -> None:
    """Atomic writes; summary.json, replaced last, marks a complete diagnostic."""
    directory.mkdir(parents=True, exist_ok=True)
    tmp = [directory / f"{name}.tmp" for name in (*tables, "summary.json")]
    try:
        for (name, table), tmp_path in zip(tables.items(), tmp):
            table.to_csv(tmp_path, index=False)
            os.replace(tmp_path, directory / name)
        text = json.dumps(summary, indent=2, allow_nan=False)
        tmp[-1].write_text(text, encoding="utf-8")
        os.replace(tmp[-1], directory / "summary.json")  # the last operation: marks success
    finally:
        for tmp_path in tmp:
            tmp_path.unlink(missing_ok=True)


def _fmt(value) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:+.3f}"


def run_mechanical_bias(paths: DataPaths, config_path: Path = DEFAULT_CONFIG, log=print) -> dict:
    """Diagnose the complete baseline OOS sample."""
    directory = paths.mechanical_bias_diagnostics
    (directory / "summary.json").unlink(missing_ok=True)  # not complete until rewritten
    years = [e.oos_year for e in baseline_episodes(config_path)]
    frame = with_extremity(pd.concat([load_conditional(paths, y) for y in years], ignore_index=True))
    monthly = monthly_correlations(frame)
    monthly_bins = monthly_bin_diagnostics(frame)
    bins = signal_bin_summary(monthly_bins)
    sources = {y: paths.conditional_disagreement / f"{y}.parquet" for y in years}
    summary = {
        "first_month": int(monthly["yyyymm"].min()),
        "last_month": int(monthly["yyyymm"].max()),
        "months": len(monthly),
        "n_stock_months": len(frame),
        "signal_bins": SIGNAL_BINS,
        "monthly_correlations": correlation_summary(monthly),
        "conditional_disagreement_sha256": {
            str(y): hashlib.sha256(p.read_bytes()).hexdigest() for y, p in sources.items()
        },
    }
    tables = {
        "monthly_correlations.csv": monthly,
        "monthly_bin_diagnostics.csv": monthly_bins,
        "signal_bin_summary.csv": bins,
    }
    _write(directory, tables, summary)

    log(f"{summary['months']} OOS months, {summary['n_stock_months']:,} stock-months")
    for name, stats in summary["monthly_correlations"].items():
        mean, mean_abs = _fmt(stats["mean_correlation"]), _fmt(stats["mean_absolute_correlation"])
        log(f"  {name:<25} mean {mean}, mean |corr| {mean_abs}")
    for _, row in bins[bins["signal_bin"].isin([1, SIGNAL_BINS])].iterrows():
        log(
            f"  bin {int(row['signal_bin']):>2}: mean raw {_fmt(row['mean_raw_disagreement'])}, "
            f"mean conditional {_fmt(row['mean_conditional_disagreement'])}, "
            f"conditional vs signal {_fmt(row['conditional_vs_signal_mean'])}, "
            f"conditional vs extremity {_fmt(row['conditional_vs_extremity_mean'])}"
        )
    log("Descriptive diagnostic only; no methodology was changed automatically.")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m disagreement.diagnostics.mechanical_bias")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args(argv)
    try:
        run_mechanical_bias(DataPaths(args.data_dir), args.config)
    except DataValidationError as exc:
        print(f"Data validation failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
