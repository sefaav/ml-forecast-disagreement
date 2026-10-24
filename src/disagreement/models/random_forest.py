"""Random Forest family: sklearn RandomForestRegressor on target_rank.

The candidate sets exactly ``n_estimators``, ``max_depth``,
``min_samples_leaf`` and ``max_features`` from the frozen grid. The fixed
settings implement the squared-error training objective with standard
bootstrap sampling; ``random_state`` is the tuning seed, the same for every
candidate, and ``n_jobs = -1`` only parallelizes the fit. Predictions go
through ``predict_random_forest``, which runs serially so that the trees are
always averaged in the same order and predictions are reproducible bit for
bit. All other parameters keep their scikit-learn defaults. The predictors are
already monthly cross-sectional ranks with missing values set to 0, so no
preprocessing is added and the native NaN handling of the trees is not used.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from disagreement.data.validation import DataValidationError

from .elastic_net import _check_inputs
from .selection import tuning_seed

FIXED = ("n_estimators",)  # same value in every candidate
TUNED = ("max_depth", "min_samples_leaf", "max_features")  # vary across candidates
CANDIDATE_PARAMS = (*FIXED, *TUNED)  # keys of each candidate from the frozen grid


def fixed_settings() -> dict:
    return {"criterion": "squared_error", "bootstrap": True, "random_state": tuning_seed(), "n_jobs": -1}


def fit_random_forest(X, y, params: Mapping[str, object]) -> RandomForestRegressor:
    """Fit one frozen Random Forest candidate to target_rank."""
    if set(params) != set(CANDIDATE_PARAMS):
        raise ValueError(f"Random Forest candidates set exactly {CANDIDATE_PARAMS}; got {sorted(params)}")
    X, y = _check_inputs(X, y)
    model = RandomForestRegressor(**{name: params[name] for name in CANDIDATE_PARAMS}, **fixed_settings())
    return model.fit(X, y)


def predict_random_forest(model: RandomForestRegressor, X) -> np.ndarray:
    """Serial prediction (n_jobs = 1), restoring the model's n_jobs afterwards."""
    X = np.asarray(X, dtype="float64")
    if X.ndim != 2 or X.shape[1] != model.n_features_in_:
        raise DataValidationError(f"X must be two-dimensional with {model.n_features_in_} predictors.")
    if not np.isfinite(X).all():
        raise DataValidationError("prediction predictors contain non-finite values.")
    n_jobs = model.get_params()["n_jobs"]
    model.set_params(n_jobs=1)
    try:
        return model.predict(X)
    finally:
        model.set_params(n_jobs=n_jobs)
