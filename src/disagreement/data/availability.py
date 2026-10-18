"""Predictor-availability rule of the frozen baseline.

A candidate predictor is eligible for a training episode if, in at least
``min_month_share`` of the historical formation months supplied at or before
the training cutoff, it is observed (finite) for at least
``min_stock_coverage`` of the stocks in U_t. Months after the cutoff are
discarded before anything is computed. The rule uses predictor coverage only:
no return, target, prediction or forecasting result.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .universe import UNIVERSE_COLUMNS
from .validation import DataValidationError


def eligible_predictors(
    panel: pd.DataFrame,
    candidates: Sequence[str],
    *,
    cutoff: int,
    min_stock_coverage: float,
    min_month_share: float,
) -> list[str]:
    """Candidates meeting the rule on the panel's months up to ``cutoff`` (yyyymm, inclusive)."""
    reserved = [c for c in candidates if c in UNIVERSE_COLUMNS]
    if reserved:
        raise DataValidationError(f"not predictors: {', '.join(reserved)}")
    missing = [c for c in candidates if c not in panel.columns]
    if missing:
        raise DataValidationError(f"panel lacks candidate predictors: {', '.join(missing)}")
    history = panel.loc[panel["yyyymm"] <= cutoff]
    if history.empty:
        raise DataValidationError(f"no formation month at or before {cutoff}.")

    observed = pd.DataFrame(
        np.isfinite(history[list(candidates)].to_numpy(dtype="float64", na_value=np.nan)),
        columns=list(candidates),
        index=history["yyyymm"].to_numpy(),
    )
    coverage = observed.groupby(level=0).mean()  # share of U_t members observed, per month
    month_share = (coverage >= min_stock_coverage).mean()
    return [c for c in candidates if month_share[c] >= min_month_share]
