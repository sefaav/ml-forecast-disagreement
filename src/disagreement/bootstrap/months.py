"""Month-cluster bootstrap of a historical sample of formation months.

The resampling unit is the formation month, never the stock-month row. With
M distinct months in the sample, M month labels are drawn with replacement
from the months in ascending order, using ``np.random.default_rng(seed)``.
Each drawn month contributes all of its rows, in their original order, once
per draw; a month that is not drawn contributes none. The function returns
positional row indices, so the caller resamples its already prepared
predictors, targets and identifiers consistently; nothing is recomputed.
"""

from __future__ import annotations

import numpy as np

from disagreement.data.validation import DataValidationError


def _month_labels(yyyymm) -> np.ndarray:
    labels = np.asarray(yyyymm)
    if labels.ndim != 1 or labels.size == 0:
        raise DataValidationError("month labels must form a non-empty one-dimensional array.")
    if labels.dtype.kind not in "iuf":
        raise DataValidationError("month labels must be numeric.")
    values = labels.astype("float64")
    if not np.isfinite(values).all():
        raise DataValidationError("month labels contain missing or non-finite values.")
    if (values != np.floor(values)).any():
        raise DataValidationError("month labels must be integer-valued.")
    return values.astype("int64")


def month_cluster_bootstrap(yyyymm, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw the sample's M months with replacement; return (sampled_months, row_indices).

    ``sampled_months`` holds the M drawn labels in draw order. ``row_indices``
    concatenates, draw by draw, the positions of the drawn month's rows in
    their original order, so ``X[row_indices]`` is the bootstrap sample.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer; got {seed!r}")
    labels = _month_labels(yyyymm)
    months, month_of_row = np.unique(labels, return_inverse=True)  # ascending months
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(len(months), size=len(months), replace=True)

    rows_by_month = np.argsort(month_of_row, kind="stable")  # original order within each month
    counts = np.bincount(month_of_row, minlength=len(months))
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    row_indices = np.concatenate([rows_by_month[starts[d] : starts[d] + counts[d]] for d in draws])
    return months[draws], row_indices.astype("int64")
