"""Baseline predictor preprocessing: monthly cross-sectional ranks in [-1, 1].

Each predictor is transformed separately within each formation month t, using
only the month-t values of the stocks supplied (U_t). Finite values are
average-ranked and mapped linearly to [-1, 1]; a predictor-month with a single
finite value maps it to 0. Missing and non-finite values take no part in the
ranking and are set to 0, the neutral value in rank space, afterwards. No
statistic is estimated across months, and no outcome enters the
transformation. Which predictors to transform is decided by the caller.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from disagreement.data.target import TARGET_COLUMNS
from disagreement.data.universe import UNIVERSE_COLUMNS
from disagreement.data.validation import KEY, DataValidationError, require_unique_keys

RESERVED_COLUMNS = frozenset((*UNIVERSE_COLUMNS, *TARGET_COLUMNS))


def rank_predictors(frame: pd.DataFrame, predictors: Sequence[str]) -> pd.DataFrame:
    """permno, yyyymm and the ranked predictors, in the input row order."""
    predictors = list(predictors)
    reserved = [p for p in predictors if p in RESERVED_COLUMNS]
    if reserved:
        raise DataValidationError(f"not predictors: {', '.join(reserved)}")
    missing = [p for p in predictors if p not in frame.columns]
    if missing:
        raise DataValidationError(f"frame lacks predictors: {', '.join(missing)}")
    if len(set(predictors)) != len(predictors):
        raise DataValidationError("predictors are listed more than once.")
    require_unique_keys(frame, "predictor frame")

    values = frame[predictors].to_numpy(dtype="float64", na_value=np.nan)
    values[~np.isfinite(values)] = np.nan  # +/- infinity is missing
    by_month = pd.DataFrame(values, columns=predictors, index=frame.index).groupby(frame["yyyymm"].to_numpy())
    rank = by_month.rank(method="average")
    n = by_month.transform("count")
    ranked = (2.0 * (rank - 1.0) / (n - 1.0).where(n > 1) - 1.0).fillna(0.0)  # missing or n = 1 -> 0
    return pd.concat([frame.loc[:, list(KEY)], ranked], axis=1)
