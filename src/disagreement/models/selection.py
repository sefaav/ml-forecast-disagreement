"""Frozen model-selection protocol: validation score, deterministic choice, candidates.

For each family and annual episode, every candidate configuration is fitted
on the training block and scored on the validation block. The score is the
equal-weighted mean over validation formation months of the monthly rank
information coefficient: the correlation between the month's
cross-sectionally ranked predictions and ``target_rank``. Higher is better;
an exact tie selects the first candidate in the configured order. This
selection metric is distinct from the training objectives (squared error on
``target_rank``). Candidates are the Cartesian product of the configured
grid, in the order the grid lists its parameters and values, merged with the
family's fixed settings.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from disagreement.data.config import DEFAULT_CONFIG
from disagreement.data.validation import DataValidationError

FAMILIES = ("elastic_net", "random_forest", "lightgbm", "mlp")


def validation_score(
    yyyymm: Sequence[int] | np.ndarray, prediction: Sequence[float] | np.ndarray, target: Sequence[float] | np.ndarray
) -> float:
    """Equal-weighted mean over months of the monthly rank IC.

    In each month, observations with a missing or non-finite target are
    excluded; the remaining predictions are average-ranked and correlated
    (Pearson) with target_rank. Constant predictions carry no ranking
    information and score 0. A month without any finite target, a month whose
    target does not vary, or a non-finite prediction for a scored observation
    is an error.
    """
    frame = pd.DataFrame(
        {
            "yyyymm": np.asarray(yyyymm),
            "prediction": np.asarray(prediction, dtype="float64"),
            "target": np.asarray(target, dtype="float64"),
        }
    )
    months = frame["yyyymm"].unique()
    frame = frame[np.isfinite(frame["target"])]
    n_empty = len(months) - frame["yyyymm"].nunique()
    if n_empty or frame.empty:
        raise ValueError(f"{n_empty or len(months)} validation months have no observation with a finite target.")
    n_bad = int((~np.isfinite(frame["prediction"])).sum())
    if n_bad:
        raise DataValidationError(f"{n_bad} scored observations have a non-finite prediction.")

    monthly_ic = []
    for month, scored in frame.groupby("yyyymm", sort=True):
        target = scored["target"].to_numpy()
        if np.ptp(target) == 0:
            raise ValueError(f"validation month {month}: target_rank has no cross-sectional variation.")
        rank = scored["prediction"].rank(method="average").to_numpy()
        monthly_ic.append(0.0 if np.ptp(rank) == 0 else float(np.corrcoef(rank, target)[0, 1]))
    return float(np.mean(monthly_ic))  # each month weighs the same


def select_candidate(scores: Sequence[float]) -> int:
    """Index of the highest validation score; an exact tie keeps the first candidate."""
    values = np.asarray(scores, dtype="float64")
    if values.size == 0 or not np.isfinite(values).all():
        raise DataValidationError("validation scores must be finite and non-empty.")
    return int(np.argmax(values))  # argmax returns the first maximum


def model_candidates(family: str, config_path: Path = DEFAULT_CONFIG) -> list[dict]:
    """The family's candidate configurations, in the frozen order."""
    if family not in FAMILIES:
        raise ValueError(f"unknown model family: {family}")
    with open(config_path, encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)["model_selection"][family]
    fixed, grid = spec.get("fixed") or {}, spec["grid"]
    names = list(grid)
    return [{**fixed, **dict(zip(names, values))} for values in itertools.product(*(grid[n] for n in names))]


def training_objective(family: str, config_path: Path = DEFAULT_CONFIG) -> str:
    """The family's frozen training loss on target_rank (squared error for all four)."""
    if family not in FAMILIES:
        raise ValueError(f"unknown model family: {family}")
    with open(config_path, encoding="utf-8") as fh:
        return str(yaml.safe_load(fh)["model_selection"][family]["objective"])


def tuning_seed(config_path: Path = DEFAULT_CONFIG) -> int:
    with open(config_path, encoding="utf-8") as fh:
        return int(yaml.safe_load(fh)["model_selection"]["seed"])
