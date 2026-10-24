"""LightGBM family on synthetic data (all frozen candidates)."""

import numpy as np
import pytest

from disagreement.data.validation import DataValidationError
from disagreement.models.lightgbm import (
    CANDIDATE_PARAMS,
    FIXED,
    TUNED,
    fit_lightgbm,
    predict_lightgbm,
)
from disagreement.models.selection import model_candidates

CANDIDATES = model_candidates("lightgbm")
FIXED_SETTINGS = {
    "objective": "regression",
    "boosting_type": "gbdt",
    "random_state": 0,
    "n_jobs": -1,
    "device_type": "cpu",
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
}


def _data(n=1500, p=5, seed=3):
    """Rank-like predictors in [-1, 1] with an interaction and a threshold effect."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, p))
    signal = 0.6 * X[:, 0] * X[:, 1] + 0.4 * (X[:, 2] > 0.3) - 0.2
    y = np.clip(signal + 0.2 * rng.standard_normal(n), -1.0, 1.0)
    return X, y


def test_eight_frozen_candidates_and_the_parameter_partition():
    assert len(CANDIDATES) == 8
    assert FIXED == ("min_child_samples",)
    assert TUNED == ("n_estimators", "learning_rate", "num_leaves")
    assert CANDIDATE_PARAMS == (*FIXED, *TUNED)
    assert all(set(c) == set(CANDIDATE_PARAMS) for c in CANDIDATES)
    assert all(c["min_child_samples"] == 100 for c in CANDIDATES)


@pytest.mark.parametrize(
    "params", CANDIDATES, ids=lambda c: f"trees={c['n_estimators']}-lr={c['learning_rate']}-leaves={c['num_leaves']}"
)
def test_every_frozen_candidate_reaches_requested_iterations_on_synthetic_data(params):
    X, y = _data()
    model = fit_lightgbm(X, y, params)
    settings = model.get_params()
    assert {name: settings[name] for name in CANDIDATE_PARAMS} == params
    assert {name: settings[name] for name in FIXED_SETTINGS} == FIXED_SETTINGS
    assert model.n_iter_ == params["n_estimators"]  # all requested iterations are reached on this synthetic dataset
    predictions = predict_lightgbm(model, X[:40])
    assert predictions.shape == (40,) and np.isfinite(predictions).all()
    assert model.get_params()["n_jobs"] == -1  # prediction does not alter the model


def test_independent_fits_predict_identically():
    X, y = _data()
    first, second = (fit_lightgbm(X, y, CANDIDATES[-1]) for _ in range(2))
    np.testing.assert_array_equal(predict_lightgbm(first, X), predict_lightgbm(second, X))


def test_invalid_inputs_and_parameters_are_rejected():
    X, y = _data(n=300, p=3)
    params = CANDIDATES[0]
    bad_X = X.copy()
    bad_X[4, 2] = np.nan
    with pytest.raises(DataValidationError, match="predictors contain non-finite"):
        fit_lightgbm(bad_X, y, params)
    bad_y = y.copy()
    bad_y[7] = np.inf
    with pytest.raises(DataValidationError, match="targets contain non-finite"):
        fit_lightgbm(X, bad_y, params)
    with pytest.raises(DataValidationError, match="rows"):
        fit_lightgbm(X, y[:-1], params)
    with pytest.raises(DataValidationError, match="two-dimensional"):
        fit_lightgbm(X[:, 0], y, params)
    with pytest.raises(DataValidationError, match="at least one"):
        fit_lightgbm(X[:0], y[:0], params)
    with pytest.raises(ValueError, match="exactly"):
        fit_lightgbm(X, y, {**params, "subsample": 0.8})
    with pytest.raises(ValueError, match="exactly"):
        fit_lightgbm(X, y, {k: v for k, v in params.items() if k != "num_leaves"})

    model = fit_lightgbm(X, y, params)
    with pytest.raises(DataValidationError, match="prediction predictors contain non-finite"):
        predict_lightgbm(model, bad_X)
    with pytest.raises(DataValidationError, match="two-dimensional"):
        predict_lightgbm(model, X[0])
    with pytest.raises(DataValidationError, match="3 predictors"):
        predict_lightgbm(model, X[:, :2])


def test_prediction_requests_a_single_thread(monkeypatch):
    X, y = _data(n=300, p=3)
    model = fit_lightgbm(X, y, CANDIDATES[0])
    calls = []

    def recording_predict(X, **kwargs):
        calls.append(kwargs)
        return np.zeros(len(X))

    monkeypatch.setattr(model, "predict", recording_predict)
    predictions = predict_lightgbm(model, X[:25])
    assert len(calls) == 1 and calls[0]["num_threads"] == 1
    assert predictions.shape == (25,)
    assert model.get_params()["n_jobs"] == -1
