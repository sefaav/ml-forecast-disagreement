"""Key checks shared by the pipeline. Error messages carry counts only, never row values."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

KEY = ("permno", "yyyymm")
KEY_FACTOR = 1_000_000  # permno * KEY_FACTOR + yyyymm is a unique int64 key


class DataValidationError(RuntimeError):
    """A data check failed. The message contains aggregate information only."""


def require_unique_keys(df: pd.DataFrame, stage: str, keys: Sequence[str] = KEY) -> None:
    """Reject duplicate security-month keys instead of resolving them silently."""
    n_dup = int(df.duplicated(list(keys)).sum())
    if n_dup:
        raise DataValidationError(f"{stage}: {n_dup} rows repeat a {'/'.join(keys)} key.")


def as_int64(values: pd.Series, name: str, stage: str) -> np.ndarray:
    """Identifier column as int64, rejecting missing or fractional values."""
    arr = values.to_numpy(dtype="float64", na_value=np.nan)
    n_missing = int(np.isnan(arr).sum())
    if n_missing:
        raise DataValidationError(f"{stage}: {n_missing} rows have a missing {name}.")
    n_fractional = int((arr != np.floor(arr)).sum())
    if n_fractional:
        raise DataValidationError(f"{stage}: {n_fractional} rows have a non-integer {name}.")
    return arr.astype("int64")


def validate_yyyymm(yyyymm: np.ndarray, stage: str) -> None:
    month, year = yyyymm % 100, yyyymm // 100
    n_bad = int(((month < 1) | (month > 12) | (year < 1900) | (year > 2100)).sum())
    if n_bad:
        raise DataValidationError(f"{stage}: {n_bad} rows have an invalid yyyymm.")


def make_keys(permno: np.ndarray, yyyymm: np.ndarray) -> np.ndarray:
    return permno.astype("int64") * KEY_FACTOR + yyyymm.astype("int64")


def isin_sorted(keys: np.ndarray, sorted_reference: np.ndarray) -> np.ndarray:
    """Boolean mask of keys present in an already sorted reference array."""
    if sorted_reference.size == 0:
        return np.zeros(keys.shape, dtype=bool)
    pos = np.minimum(np.searchsorted(sorted_reference, keys), sorted_reference.size - 1)
    return sorted_reference[pos] == keys
