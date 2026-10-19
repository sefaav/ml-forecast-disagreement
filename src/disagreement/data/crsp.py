"""CRSP CIZ (Stock Version 2) monthly security data from WRDS.

Field names and codes follow the CRSP "US Stock & Indexes Database Guide,
File Format 2.0 (CIZ)" and were verified against the live WRDS schema.
Licensed rows are read and written locally only; nothing here prints them.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from .validation import DataValidationError, as_int64, require_unique_keys, validate_yyyymm

# MthRet already includes delisting returns in CIZ; never compound a separate
# delisting return into it.
REQUIRED_FIELDS = (
    "permno",
    "mthcaldt",
    "mthret",
    "mthprc",
    "mthcap",
    "primaryexch",
    "conditionaltype",
    "tradingstatusflg",
    "sharetype",
    "securitytype",
    "securitysubtype",
    "usincflg",
    "issuertype",
    "mthretflg",
    "mthdelflg",
)
FLOAT_FIELDS = ("mthret", "mthprc", "mthcap")

# CIZ equivalent of legacy share codes 10/11 (US common stock)...
COMMON_STOCK_CODES = {
    "sharetype": ("NS",),
    "securitytype": ("EQTY",),
    "securitysubtype": ("COM",),
    "usincflg": ("Y",),
    "issuertype": ("ACOR", "CORP"),
}
# ...and of legacy exchange codes 1/2/3: NYSE, NYSE American (AMEX) or Nasdaq,
# regular way, active trading status.
EXCHANGE_LABELS = {"N": "NYSE", "A": "AMEX", "Q": "NASDAQ"}
# Delisting during the record's own month: MthRetFlg 'DE', or any delisting
# flag MthDelFlg A/P/V/G/M, whether or not the delisting return is in MthRet.
DELISTING_RETFLG = "DE"
DELISTING_DELFLG = ("A", "P", "V", "G", "M")

# Predictors OSAP leaves to be built from CRSP (OSAP CrossSection v2 code,
# SignalDoc sign -1), stored signed like the OSAP wide file.
CRSP_PREDICTORS = ("Price", "Size", "STreversal")

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _codes(df: pd.DataFrame, field: str) -> pd.Series:
    return df[field].astype("string").str.strip()  # CHAR columns can be padded


def _is(df: pd.DataFrame, field: str, codes) -> np.ndarray:
    return _codes(df, field).isin(list(codes)).fillna(False).to_numpy(dtype=bool)


def is_common_stock(df: pd.DataFrame) -> np.ndarray:
    mask = np.ones(len(df), dtype=bool)
    for field, codes in COMMON_STOCK_CODES.items():
        mask &= _is(df, field, codes)
    return mask


def exchange_labels(df: pd.DataFrame) -> pd.Series:
    """NYSE/AMEX/NASDAQ for active regular-way listings, missing otherwise."""
    listed = (
        _is(df, "primaryexch", EXCHANGE_LABELS)
        & _is(df, "conditionaltype", ("RW",))
        & _is(df, "tradingstatusflg", ("A",))
    )
    return _codes(df, "primaryexch").map(EXCHANGE_LABELS).where(listed).astype(object)


def delisted_in_month(df: pd.DataFrame) -> np.ndarray:
    return _is(df, "mthretflg", (DELISTING_RETFLG,)) | _is(df, "mthdelflg", DELISTING_DELFLG)


def month_keys(raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    dates = pd.to_datetime(raw["mthcaldt"])
    n_missing = int(dates.isna().sum())
    if n_missing:
        raise DataValidationError(f"CRSP: {n_missing} rows have no mthcaldt.")
    yyyymm = (dates.dt.year * 100 + dates.dt.month).to_numpy(dtype="int64")
    validate_yyyymm(yyyymm, "CRSP")
    return as_int64(raw["permno"], "permno", "CRSP"), yyyymm


def normalize_monthly(raw: pd.DataFrame) -> pd.DataFrame:
    """Formation-month fields keyed by (permno, yyyymm), from each record's own month."""
    missing = [f for f in REQUIRED_FIELDS if f not in raw.columns]
    if missing:
        raise DataValidationError(f"CRSP data lack required fields: {', '.join(missing)}")
    raw = raw.reset_index(drop=True)
    permno, yyyymm = month_keys(raw)
    out = pd.DataFrame(
        {
            "permno": permno,
            "yyyymm": yyyymm,
            "exchange": exchange_labels(raw),
            "is_common": is_common_stock(raw),
            "mktcap": raw["mthcap"].to_numpy(dtype="float64", na_value=np.nan),
            "delisted_in_month": delisted_in_month(raw),
        }
    )
    require_unique_keys(out, "CRSP monthly")
    return out


def crsp_predictors(raw: pd.DataFrame) -> pd.DataFrame:
    """Signed Price, Size and STreversal from the month-t record.

    Price = -ln|MthPrc|; Size = -ln(MthCap), market equity in $ thousands
    (|prc| x shrout in the legacy files) at the PERMNO level; STreversal =
    -100 x MthRet, in percent, with a missing return set to 0 (OSAP's
    convention). Non-positive prices or caps give missing values.
    """
    raw = raw.reset_index(drop=True)
    permno, yyyymm = month_keys(raw)
    price = np.abs(raw["mthprc"].to_numpy(dtype="float64", na_value=np.nan))
    cap = raw["mthcap"].to_numpy(dtype="float64", na_value=np.nan)
    ret = raw["mthret"].to_numpy(dtype="float64", na_value=np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        price_log = np.where(price > 0, np.log(price), np.nan)
        cap_log = np.where(cap > 0, np.log(cap), np.nan)
    return pd.DataFrame(
        {
            "permno": permno,
            "yyyymm": yyyymm,
            "Price": -price_log,
            "Size": -cap_log,
            "STreversal": -100.0 * np.nan_to_num(ret, nan=0.0) + 0.0,  # + 0.0 avoids -0.0
        }
    )


def coerce_raw_types(df: pd.DataFrame) -> pd.DataFrame:
    """Stable dtypes for the extract, whatever the database driver returned."""
    df = df.reset_index(drop=True)
    columns: dict[str, object] = {
        "permno": as_int64(df["permno"], "permno", "CRSP extract"),
        "mthcaldt": pd.to_datetime(df["mthcaldt"]),
    }
    for field in REQUIRED_FIELDS[2:]:
        if field in FLOAT_FIELDS:
            columns[field] = df[field].to_numpy(dtype="float64", na_value=np.nan)
        else:
            columns[field] = _codes(df, field)
    return pd.DataFrame(columns)


# ---------------------------------------------------------------------------
# WRDS extraction (requires the 'wrds' extra and an interactive WRDS login)
# ---------------------------------------------------------------------------


def _identifier(name: str) -> str:
    if not _IDENTIFIER.match(name):
        raise ValueError(f"invalid SQL identifier: {name!r}")
    return name.lower()


def connect(username: str | None = None):
    import wrds  # optional dependency, imported lazily

    return wrds.Connection(wrds_username=username) if username else wrds.Connection()


def live_columns(conn, library: str, table: str) -> dict[str, str]:
    desc = conn.describe_table(library=_identifier(library), table=_identifier(table))
    return {str(row.name).lower(): str(row.type) for row in desc.itertuples()}


def extract_monthly(conn, library: str, table: str, out_dir: Path, log=print) -> dict:
    """Fresh download of the required fields, one Parquet file per year.

    Every year is written to a temporary directory, which replaces out_dir
    only once the whole table has been extracted and checked; a failed run
    leaves the previous extract untouched.
    """
    library, table = _identifier(library), _identifier(table)
    columns = live_columns(conn, library, table)
    missing = [f for f in REQUIRED_FIELDS if f not in columns]
    if missing:
        raise DataValidationError(
            f"{library}.{table} lacks required fields: {', '.join(missing)}"
        )
    span = conn.raw_sql(
        f"select min(mthcaldt) as first_date, max(mthcaldt) as last_date from {library}.{table}",
        date_cols=["first_date", "last_date"],
    )
    first, last = span["first_date"].iloc[0], span["last_date"].iloc[0]

    partial = out_dir.with_name(out_dir.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    rows_by_year: dict[str, int] = {}
    dates = []
    try:
        for year in range(first.year, last.year + 1):
            frame = coerce_raw_types(
                conn.raw_sql(
                    f"select {', '.join(REQUIRED_FIELDS)} from {library}.{table} "
                    f"where mthcaldt between '{year}-01-01' and '{year}-12-31'",
                    date_cols=["mthcaldt"],
                )
            )
            n_other = int((frame["mthcaldt"].dt.year != year).sum())
            if n_other:
                raise DataValidationError(f"CRSP {year}: {n_other} extracted rows fall outside the year.")
            frame.to_parquet(partial / f"crsp_monthly_{year}.parquet", index=False)
            rows_by_year[str(year)] = len(frame)
            dates += [frame["mthcaldt"].min(), frame["mthcaldt"].max()] if len(frame) else []
            log(f"  {year}: {len(frame):,} rows")
        if (min(dates), max(dates)) != (first, last):
            raise DataValidationError("CRSP: the extract does not span the table's first and last dates.")
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise

    manifest = {
        "library": library,
        "table": table,
        "fields": list(REQUIRED_FIELDS),
        "table_first_date": str(first.date()),
        "table_last_date": str(last.date()),
        "rows_by_year": rows_by_year,
        "extracted_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wrds_package_version": _version("wrds"),
    }
    (partial / "_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    previous = out_dir.with_name(out_dir.name + ".previous")
    if previous.exists():
        shutil.rmtree(previous)
    if out_dir.exists():
        out_dir.rename(previous)
    try:
        partial.rename(out_dir)
    except BaseException:
        if previous.exists() and not out_dir.exists():
            previous.rename(out_dir)  # restore the previous valid extract
        raise
    if previous.exists():
        shutil.rmtree(previous)
    return manifest


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None
