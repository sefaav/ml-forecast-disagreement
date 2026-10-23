"""Elastic Net family on synthetic data (all frozen candidates)."""

import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from disagreement.data.validation import DataValidationError
from disagreement.models import elastic_net
from disagreement.models.elastic_net import ElasticNetConvergenceError, fit_elastic_net
from disagreement.models.selection import model_candidates

CANDIDATES = model_candidates("elastic_net")


def _data(n=600, p=8, seed=1):
    """Rank-like predictors in [-1, 1] with a genuine linear signal in the target."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, p))
    y = np.clip(0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.2 * rng.standard_normal(n), -1.0, 1.0)
    return X, y


@pytest.mark.parametrize("params", CANDIDATES, ids=lambda c: f"alpha={c['alpha']}-l1={c['l1_ratio']}")
def test_every_frozen_candidate_fits_with_the_fixed_settings(params):
    X, y = _data()
    model = fit_elastic_net(X, y, params)
    settings = model.get_params()
    assert (settings["alpha"], settings["l1_ratio"]) == (params["alpha"], params["l1_ratio"])
    fixed = (settings["fit_intercept"], settings["selection"], settings["max_iter"], settings["tol"])
    assert fixed == (True, "cyclic", 10_000, 1e-6)
    predictions = model.predict(X[:50])
    assert predictions.shape == (50,) and np.isfinite(predictions).all()


def test_all_twelve_candidates_are_exercised_and_fits_are_deterministic():
    assert len(CANDIDATES) == 12
    X, y = _data()
    first = fit_elastic_net(X, y, CANDIDATES[0]).predict(X)
    np.testing.assert_array_equal(first, fit_elastic_net(X, y, CANDIDATES[0]).predict(X))


def test_invalid_inputs_are_rejected():
    X, y = _data(n=20, p=3)
    params = CANDIDATES[0]
    bad_X = X.copy()
    bad_X[3, 1] = np.inf
    with pytest.raises(DataValidationError, match="predictors contain non-finite"):
        fit_elastic_net(bad_X, y, params)
    bad_y = y.copy()
    bad_y[5] = np.nan
    with pytest.raises(DataValidationError, match="targets contain non-finite"):
        fit_elastic_net(X, bad_y, params)
    with pytest.raises(DataValidationError, match="rows"):
        fit_elastic_net(X, y[:-1], params)
    with pytest.raises(DataValidationError, match="two-dimensional"):
        fit_elastic_net(X[:, 0], y, params)
    with pytest.raises(DataValidationError, match="at least one"):
        fit_elastic_net(X[:, :0], y, params)
    with pytest.raises(ValueError, match="exactly"):
        fit_elastic_net(X, y, {**params, "max_iter": 50})  # only the frozen grid parameters


def test_a_convergence_warning_fails_the_fit(monkeypatch):
    def unconverged_fit(self, X, y, *args, **kwargs):
        warnings.warn("did not converge", ConvergenceWarning)
        return self

    monkeypatch.setattr(elastic_net.ElasticNet, "fit", unconverged_fit)
    X, y = _data(n=20, p=3)
    with pytest.raises(ElasticNetConvergenceError, match="did not converge"):
        fit_elastic_net(X, y, CANDIDATES[0])
