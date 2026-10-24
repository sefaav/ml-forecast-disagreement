"""Random Forest family on synthetic data (all frozen candidates)."""

import numpy as np
import pytest

from disagreement.data.validation import DataValidationError
from disagreement.models.random_forest import (
    CANDIDATE_PARAMS,
    FIXED,
    TUNED,
    fit_random_forest,
    predict_random_forest,
)
from disagreement.models.selection import model_candidates

CANDIDATES = model_candidates("random_forest")


def _data(n=300, p=6, seed=2):
    """Rank-like predictors in [-1, 1] with an interaction and a threshold effect."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, p))
    signal = 0.6 * X[:, 0] * X[:, 1] + 0.4 * (X[:, 2] > 0.3) - 0.2
    y = np.clip(signal + 0.2 * rng.standard_normal(n), -1.0, 1.0)
    return X, y


def test_twelve_frozen_candidates():
    assert len(CANDIDATES) == 12
    assert FIXED == ("n_estimators",)
    assert TUNED == ("max_depth", "min_samples_leaf", "max_features")
    assert CANDIDATE_PARAMS == (*FIXED, *TUNED)
    assert all(set(c) == set(CANDIDATE_PARAMS) for c in CANDIDATES)
    assert all(c["n_estimators"] == 300 for c in CANDIDATES)


@pytest.mark.parametrize(
    "params", CANDIDATES, ids=lambda c: f"depth={c['max_depth']}-leaf={c['min_samples_leaf']}-feat={c['max_features']}"
)
def test_every_frozen_candidate_fits_with_the_fixed_settings(params):
    X, y = _data()
    model = fit_random_forest(X, y, params)
    settings = model.get_params()
    assert {name: settings[name] for name in CANDIDATE_PARAMS} == params
    fixed = (settings["criterion"], settings["bootstrap"], settings["random_state"], settings["n_jobs"])
    assert fixed == ("squared_error", True, 0, -1)
    predictions = predict_random_forest(model, X[:40])
    assert predictions.shape == (40,) and np.isfinite(predictions).all()
    assert model.get_params()["n_jobs"] == -1  # restored after the serial prediction


def test_independent_fits_predict_bit_for_bit_identically():
    X, y = _data()
    first, second = (fit_random_forest(X, y, CANDIDATES[-1]) for _ in range(2))
    np.testing.assert_array_equal(predict_random_forest(first, X), predict_random_forest(second, X))
    assert first.get_params()["n_jobs"] == second.get_params()["n_jobs"] == -1


def test_invalid_inputs_and_parameters_are_rejected():
    X, y = _data(n=30, p=3)
    params = CANDIDATES[0]
    bad_X = X.copy()
    bad_X[4, 2] = np.nan
    with pytest.raises(DataValidationError, match="predictors contain non-finite"):
        fit_random_forest(bad_X, y, params)
    bad_y = y.copy()
    bad_y[7] = -np.inf
    with pytest.raises(DataValidationError, match="targets contain non-finite"):
        fit_random_forest(X, bad_y, params)
    with pytest.raises(DataValidationError, match="rows"):
        fit_random_forest(X, y[:-1], params)
    with pytest.raises(DataValidationError, match="two-dimensional"):
        fit_random_forest(X, y.reshape(-1, 1), params)
    with pytest.raises(DataValidationError, match="at least one"):
        fit_random_forest(X[:0], y[:0], params)
    with pytest.raises(ValueError, match="exactly"):
        fit_random_forest(X, y, {**params, "oob_score": True})
    with pytest.raises(ValueError, match="exactly"):
        fit_random_forest(X, y, {k: v for k, v in params.items() if k != "max_features"})

    model = fit_random_forest(X, y, params)
    with pytest.raises(DataValidationError, match="prediction predictors contain non-finite"):
        predict_random_forest(model, bad_X)
    with pytest.raises(DataValidationError, match="3 predictors"):
        predict_random_forest(model, X[:, :2])
    with pytest.raises(DataValidationError, match="two-dimensional"):
        predict_random_forest(model, X[0])
    assert model.get_params()["n_jobs"] == -1


def test_n_jobs_is_restored_when_prediction_fails(monkeypatch):
    X, y = _data(n=30, p=3)
    model = fit_random_forest(X, y, CANDIDATES[0])
    seen = []

    def failing_predict(X):
        seen.append(model.get_params()["n_jobs"])
        raise RuntimeError("synthetic prediction failure")

    monkeypatch.setattr(model, "predict", failing_predict)
    with pytest.raises(RuntimeError, match="synthetic prediction failure"):
        predict_random_forest(model, X)
    assert seen == [1]  # switched to serial prediction before the call
    assert model.get_params()["n_jobs"] == -1  # restored by the finally block
