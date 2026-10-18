"""Formation-time stock universe U_t.

A security-month is eligible if, on its own month-t record, it is a common
stock listed (regular way, active) on NYSE, AMEX or NASDAQ, did not delist
during month t, and has a finite positive market capitalisation. U_t keeps the
eligible stocks whose market capitalisation is at or above the 20th percentile
of eligible NYSE stocks in the same month.

The builder projects its input onto FORMATION_COLUMNS and computes everything
within each month, so returns, later months or later delistings cannot affect
membership. A member that delists during t+1 stays in U_t.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .validation import DataValidationError, require_unique_keys

FORMATION_COLUMNS = ("permno", "yyyymm", "exchange", "is_common", "mktcap", "delisted_in_month")
UNIVERSE_COLUMNS = ("permno", "yyyymm", "exchange", "mktcap")


def build_universe(
    panel: pd.DataFrame,
    *,
    exchanges: Sequence[str] = ("NYSE", "AMEX", "NASDAQ"),
    breakpoint_exchange: str = "NYSE",
    quantile: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (members, by_month) for every month in ``panel``.

    The breakpoint uses linear interpolation (the pandas/numpy default);
    stocks exactly at the breakpoint are kept. A month without eligible
    breakpoint-exchange stocks has a missing breakpoint and no members.
    """
    missing = [c for c in FORMATION_COLUMNS if c not in panel.columns]
    if missing:
        raise DataValidationError(f"universe input lacks columns: {', '.join(missing)}")
    df = panel.loc[:, list(FORMATION_COLUMNS)].reset_index(drop=True)
    require_unique_keys(df, "universe input")

    cap = df["mktcap"].to_numpy(dtype="float64", na_value=np.nan)
    exchange = df["exchange"].astype("string")
    eligible = (
        df["is_common"].astype("boolean").fillna(False).to_numpy(dtype=bool)
        & exchange.isin(list(exchanges)).fillna(False).to_numpy(dtype=bool)
        & ~df["delisted_in_month"].astype("boolean").fillna(False).to_numpy(dtype=bool)
        & np.isfinite(cap)
        & (cap > 0)
    )
    nyse = eligible & exchange.eq(breakpoint_exchange).fillna(False).to_numpy(dtype=bool)
    breakpoints = df.loc[nyse].groupby("yyyymm")["mktcap"].quantile(quantile)
    month_breakpoint = df["yyyymm"].map(breakpoints).to_numpy(dtype="float64", na_value=np.nan)
    with np.errstate(invalid="ignore"):
        member = eligible & (cap >= month_breakpoint)  # missing breakpoint -> False

    members = (
        df.loc[member, list(UNIVERSE_COLUMNS)].sort_values(["yyyymm", "permno"]).reset_index(drop=True)
    )
    by_month = (
        pd.DataFrame(
            {
                "yyyymm": df["yyyymm"].to_numpy(),
                "n_eligible": eligible,
                "n_nyse_eligible": nyse,
                "n_universe": member,
            }
        )
        .groupby("yyyymm")
        .sum()
        .astype("int64")
    )
    by_month.insert(2, "nyse_breakpoint", breakpoints.reindex(by_month.index))
    return members, by_month.reset_index()
