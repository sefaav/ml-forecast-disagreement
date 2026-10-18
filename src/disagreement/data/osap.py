"""Open Source Asset Pricing (Chen and Zimmermann) firm-level signals.

The wide file ``signed_predictors_dl_wide.zip`` is keyed by ``permno`` and
``yyyymm``. OSAP data "are always end-of-month": a signal dated yyyymm is known
at the end of formation month t = yyyymm and is joined to that month without
an additional lag. The CSV is streamed out of the zip archive and never held in
memory at once.
"""

from __future__ import annotations

import csv
import hashlib
import re
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from .validation import DataValidationError, as_int64, validate_yyyymm

ID_COLUMNS = ("permno", "yyyymm")
DEFAULT_BLOCK_SIZE = 32 * 1024 * 1024


@contextmanager
def open_csv_stream(path: Path) -> Iterator[IO[bytes]]:
    """Binary stream of the CSV inside the official OSAP zip archive."""
    path = Path(path)
    if path.suffix.lower() != ".zip":
        raise DataValidationError("OSAP source must be the official .zip archive.")
    with zipfile.ZipFile(path) as archive:
        members = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if len(members) != 1:
            raise DataValidationError(f"OSAP archive must contain exactly one CSV file; found {len(members)}.")
        with archive.open(members[0]) as fh:
            yield fh


def read_header(path: Path) -> list[str]:
    with open_csv_stream(path) as fh:
        line = fh.readline().decode("utf-8-sig")
    return [name.strip() for name in next(csv.reader([line]))]


def signal_columns(header: list[str]) -> list[str]:
    """Characteristic columns, after checking that the identifiers exist."""
    lowered = [name.lower() for name in header]
    missing = [c for c in ID_COLUMNS if c not in lowered]
    if missing:
        raise DataValidationError(f"OSAP header lacks identifier columns: {', '.join(missing)}")
    if len(set(lowered)) != len(lowered) or "" in lowered:
        raise DataValidationError("OSAP header has repeated or empty column names.")
    return [name for name in header if name.lower() not in ID_COLUMNS]


def _sanitized(exc: pa.ArrowInvalid, header: list[str]) -> str:
    # Arrow messages can quote cell values or whole rows: keep the column only.
    match = re.search(r"CSV column #(\d+)", str(exc))
    if match:
        index = int(match.group(1))
        name = header[index] if index < len(header) else "?"
        return f"OSAP: column '{name}' contains a non-numeric value."
    return "OSAP: the CSV could not be parsed (details suppressed: they may contain row values)."


def iter_frames(path: Path, block_size: int = DEFAULT_BLOCK_SIZE) -> Iterator[pd.DataFrame]:
    """Yield validated chunks: int64 permno/yyyymm and float64 characteristics."""
    header = read_header(path)
    signal_columns(header)
    rename = {name: name.lower() for name in header if name.lower() in ID_COLUMNS}
    convert = pacsv.ConvertOptions(column_types={name: pa.float64() for name in header})
    with open_csv_stream(path) as fh:
        try:
            reader = pacsv.open_csv(
                fh, read_options=pacsv.ReadOptions(block_size=block_size), convert_options=convert
            )
            for batch in reader:
                frame = batch.to_pandas().rename(columns=rename)
                frame["permno"] = as_int64(frame["permno"], "permno", "OSAP")
                frame["yyyymm"] = as_int64(frame["yyyymm"], "yyyymm", "OSAP")
                validate_yyyymm(frame["yyyymm"].to_numpy(), "OSAP")
                yield frame
        except pa.ArrowInvalid as exc:
            raise DataValidationError(_sanitized(exc, header)) from None


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
