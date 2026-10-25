"""Next-month realized return and the primary ranked training target.

For each member of U_t (formation month t), ``ret_next`` is the same PERMNO's
CIZ MthRet in calendar month t+1, which already includes the delisting return
when CRSP has one; no separate delisting return is compounded. Nothing else
from t+1 is used: a member that delists, changes exchange or leaves the
universe in t+1 keeps its formation observation. Without a finite MthRet at
exactly t+1, ``ret_next`` is missing.

``target_rank`` ranks the finite ``ret_next`` values within each formation
month's U_t (average ranks for ties) and maps them linearly to [-1, 1]; a
month with a single valid return gets 0.
"""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd

from . import crsp
from .config import DataConfig, DataPaths
from .validation import KEY, DataValidationError, require_unique_keys

TARGET_COLUMNS = ("permno", "yyyymm", "ret_next", "target_rank")
RETURN_FIELDS = ["permno", "mthcaldt", "mthret"]


def next_month(yyyymm: np.ndarray) -> np.ndarray:
    return np.where(yyyymm % 100 == 12, (yyyymm // 100 + 1) * 100 + 1, yyyymm + 1)


def attach_next_return(universe: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """U_t keys with the same PERMNO's MthRet from the CIZ records of calendar month t+1."""
    require_unique_keys(universe, "universe")
    permno, yyyymm = crsp.month_keys(raw.reset_index(drop=True))
    ret = raw["mthret"].to_numpy(dtype="float64", na_value=np.nan)
    returns = pd.DataFrame({"permno": permno, "next": yyyymm, "ret_next": np.where(np.isfinite(ret), ret, np.nan)})
    require_unique_keys(returns, "CRSP returns", ("permno", "next"))
    keys = universe.loc[:, list(KEY)].astype("int64").reset_index(drop=True)
    keys["next"] = next_month(keys["yyyymm"].to_numpy())
    out = keys.merge(returns, on=["permno", "next"], how="left", validate="one_to_one").drop(columns="next")
    if len(out) != len(keys):
        raise DataValidationError("attaching next-month returns changed the number of rows.")
    return out


def rank_target(frame: pd.DataFrame) -> pd.Series:
    """Average rank of ret_next within each formation month, mapped to [-1, 1]."""
    by_month = frame.groupby("yyyymm")["ret_next"]
    rank = by_month.rank(method="average")
    n = by_month.transform("count")
    target = (2.0 * (rank - 1.0) / (n - 1.0).where(n > 1) - 1.0).where(n > 1, 0.0)
    return target.where(frame["ret_next"].notna())


def build_targets(paths: DataPaths, cfg: DataConfig) -> dict:
    """ret_next and target_rank for the formation months of the baseline sample, one file per year."""
    if not (paths.diagnostics / "summary.json").exists():
        raise DataValidationError("no successful build found; run build first.")
    years = range(cfg.sample_start // 100, cfg.sample_end // 100 + 1)
    crsp_dir = paths.crsp(cfg.crsp_table)
    needed = [paths.universe / f"universe_{y}.parquet" for y in years]
    needed += [crsp_dir / f"crsp_monthly_{y}.parquet" for y in range(years.start, years.stop + 1)]
    missing = [p.name for p in needed if not p.exists()]
    if missing:
        raise DataValidationError(f"missing input files: {', '.join(missing)}")

    # Written to a staging directory that replaces target/ only once every year has been built.
    partial = paths.target.with_name(paths.target.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    rows = finite = ranked = 0
    months: set[int] = set()
    last_return_month = 0
    try:
        for year in years:
            members = pd.read_parquet(paths.universe / f"universe_{year}.parquet", columns=list(KEY))
            members = members[(members["yyyymm"] >= cfg.sample_start) & (members["yyyymm"] <= cfg.sample_end)]
            raw = pd.concat(
                [pd.read_parquet(crsp_dir / f"crsp_monthly_{y}.parquet", columns=RETURN_FIELDS) for y in (year, year + 1)],
                ignore_index=True,
            )
            target = attach_next_return(members, raw)
            target["target_rank"] = rank_target(target)
            require_unique_keys(target, "target")
            target.loc[:, list(TARGET_COLUMNS)].to_parquet(
                partial / f"target_{year}.parquet", index=False, compression="zstd"
            )

            has_return = target["ret_next"].notna()
            rows += len(target)
            finite += int(has_return.sum())
            ranked += int(target["target_rank"].notna().sum())
            months.update(target["yyyymm"].tolist())
            if has_return.any():
                last_return_month = max(last_return_month, int(next_month(target.loc[has_return, "yyyymm"].to_numpy()).max()))
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise

    previous = paths.target.with_name(paths.target.name + ".previous")
    if previous.exists():
        shutil.rmtree(previous)
    if paths.target.exists():
        paths.target.rename(previous)
    try:
        partial.rename(paths.target)
    except BaseException:
        if previous.exists() and not paths.target.exists():
            previous.rename(paths.target)  # restore the previous valid target
        raise
    if previous.exists():
        shutil.rmtree(previous)
    if paths.oos_predictions.exists():  # forecasts of the replaced targets are stale
        shutil.rmtree(paths.oos_predictions)
    return {
        "first_month": min(months),
        "last_month": max(months),
        "months": len(months),
        "formation_rows": rows,
        "ret_next_finite": finite,
        "ret_next_finite_share": finite / rows,
        "target_rank_nonmissing": ranked,
        "target_rank_nonmissing_share": ranked / rows,
        "last_return_month": last_return_month,
    }
