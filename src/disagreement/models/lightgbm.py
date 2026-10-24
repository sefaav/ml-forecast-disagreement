"""LightGBM family: lightgbm.LGBMRegressor on target_rank.

Each candidate sets ``min_child_samples`` (fixed across candidates) and
``n_estimators``, ``learning_rate`` and ``num_leaves`` (tuned) from the frozen
grid. The fixed settings implement the L2 (squared-error) training objective
with standard GBDT. No evaluation set, early-stopping callback or other
early-stopping mechanism is used: the candidate's ``n_estimators`` is passed
directly to LightGBM, which may still finish with fewer fitted iterations if
further tree growth becomes impossible. The tuning seed, CPU execution,
deterministic mode and column-wise histogram construction make repeated fits
reproducible in a fixed software environment; ``n_jobs = -1`` only
parallelizes the fit. Predictions go through ``predict_lightgbm``, which
requests a single prediction thread. The predictors are already monthly
cross-sectional ranks with missing values set to 0, so no preprocessing is
added and LightGBM's native missing-value handling is not used. All other
parameters keep their LightGBM defaults.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from lightgbm import LGBMRegressor

from disagreement.data.validation import DataValidationError

from .elastic_net import _check_inputs
from .selection import tuning_seed

FIXED = ("min_child_samples",)  # same value in every candidate
TUNED = ("n_estimators", "learning_rate", "num_leaves")  # vary across candidates
CANDIDATE_PARAMS = (*FIXED, *TUNED)  # keys of each candidate from the frozen grid


def fixed_settings() -> dict:
    return {
        "objective": "regression",
        "boosting_type": "gbdt",
        "random_state": tuning_seed(),
        "n_jobs": -1,
        "device_type": "cpu",
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }


def fit_lightgbm(X, y, params: Mapping[str, object]) -> LGBMRegressor:
    """Fit one frozen LightGBM candidate to target_rank."""
    if set(params) != set(CANDIDATE_PARAMS):
        raise ValueError(f"LightGBM candidates set exactly {CANDIDATE_PARAMS}; got {sorted(params)}")
    X, y = _check_inputs(X, y)
    model = LGBMRegressor(**{name: params[name] for name in CANDIDATE_PARAMS}, **fixed_settings())
    return model.fit(X, y)


def predict_lightgbm(model: LGBMRegressor, X) -> np.ndarray:
    """Prediction with a single thread; the fitted model is not modified."""
    X = np.asarray(X, dtype="float64")
    if X.ndim != 2 or X.shape[1] != model.n_features_in_:
        raise DataValidationError(f"X must be two-dimensional with {model.n_features_in_} predictors.")
    if not np.isfinite(X).all():
        raise DataValidationError("prediction predictors contain non-finite values.")
    return np.asarray(model.predict(X, num_threads=1))
