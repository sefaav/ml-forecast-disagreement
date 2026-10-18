"""Synthetic CRSP- and OSAP-like records for tests. No licensed data is used."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

COMMON_STOCK = {
    "sharetype": "NS",
    "securitytype": "EQTY",
    "securitysubtype": "COM",
    "usincflg": "Y",
    "issuertype": "CORP",
}


def crsp_row(permno: int, month: str, cap: float, exch: str | None = "N", **overrides) -> dict:
    """One raw CIZ-style monthly record; ``month`` is 'YYYY-MM'."""
    row = {
        "permno": permno,
        "mthcaldt": pd.Timestamp(month) + pd.offsets.MonthEnd(0),
        "mthret": 0.01,
        "mthprc": 10.0,
        "mthcap": cap,
        "primaryexch": exch,
        "conditionaltype": "RW",
        "mthretflg": "CR",
        "mthdelflg": "N",
        "tradingstatusflg": "A",
        **COMMON_STOCK,
    }
    row.update(overrides)
    return row


def formation_row(
    permno: int,
    yyyymm: int,
    exchange: str | None,
    mktcap: float,
    is_common: bool = True,
    delisted_in_month: bool = False,
    **extra,
) -> dict:
    """One normalized formation-month record, as consumed by build_universe."""
    return {
        "permno": permno,
        "yyyymm": yyyymm,
        "exchange": exchange,
        "is_common": is_common,
        "mktcap": mktcap,
        "delisted_in_month": delisted_in_month,
        **extra,
    }


def write_osap_zip(path: Path, frame: pd.DataFrame, member: str = "signed_predictors_dl_wide.csv") -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, frame.to_csv(index=False))
    return path
