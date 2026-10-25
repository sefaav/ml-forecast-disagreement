"""MLP family on synthetic data (all frozen candidates)."""

import numpy as np
import pytest
import torch
from torch import nn

from disagreement.data.validation import DataValidationError
from disagreement.models.mlp import (
    ADAM_SETTINGS,
    CANDIDATE_PARAMS,
    FIXED,
    LOSS_REDUCTION,
    TUNED,
    fit_mlp,
    predict_mlp,
)
from disagreement.models.selection import model_candidates

CANDIDATES = model_candidates("mlp")
N_OBS = 5000  # one full mini-batch of 4096 and a final partial batch of 904


def _data(n=N_OBS, p=4, seed=4):
    """Rank-like predictors in [-1, 1] with an interaction and a threshold effect."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, p))
    signal = 0.6 * X[:, 0] * X[:, 1] + 0.4 * (X[:, 2] > 0.3) - 0.2
    y = np.clip(signal + 0.2 * rng.standard_normal(n), -1.0, 1.0)
    return X, y


def test_eight_frozen_candidates_and_the_parameter_partition():
    assert len(CANDIDATES) == 8
    assert FIXED == ("optimizer", "batch_size", "max_epochs")
    assert TUNED == ("hidden_layers", "learning_rate", "weight_decay")
    assert CANDIDATE_PARAMS == (*FIXED, *TUNED)
    assert all(set(c) == set(CANDIDATE_PARAMS) for c in CANDIDATES)
    assert all((c["optimizer"], c["batch_size"], c["max_epochs"]) == ("adam", 4096, 30) for c in CANDIDATES)


def test_frozen_adam_settings_and_loss_reduction():
    assert ADAM_SETTINGS == {
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "amsgrad": False,
        "decoupled_weight_decay": False,
        "foreach": False,
        "fused": False,
    }
    assert LOSS_REDUCTION == "mean"


@pytest.mark.parametrize(
    "params",
    CANDIDATES,
    ids=lambda c: f"layers={c['hidden_layers']}-lr={c['learning_rate']}-wd={c['weight_decay']}",
)
def test_every_frozen_candidate_trains_with_the_frozen_protocol(params):
    X, y = _data()
    model = fit_mlp(X, y, params)
    layers = list(model.network)
    linear = [m for m in layers if isinstance(m, nn.Linear)]
    assert [m.out_features for m in linear[:-1]] == params["hidden_layers"]
    assert all(isinstance(m, nn.ReLU) for m in layers[1:-1:2])  # every hidden Linear is followed by ReLU
    assert isinstance(layers[-1], nn.Linear) and layers[-1].out_features == 1  # no output activation
    assert len(layers) == 2 * len(params["hidden_layers"]) + 1
    assert all(p.dtype == torch.float32 and p.device.type == "cpu" for p in model.parameters())
    assert model.hidden_layers == tuple(params["hidden_layers"]) and model.n_features_in_ == X.shape[1]
    assert (model.learning_rate, model.weight_decay, model.batch_size) == (
        params["learning_rate"], params["weight_decay"], 4096,
    )
    assert model.epochs_completed_ == 30
    assert len(model.loss_history_) == 30 and np.isfinite(model.loss_history_).all()
    predictions = predict_mlp(model, X[:40])
    assert predictions.shape == (40,) and np.isfinite(predictions).all()


def test_exactly_max_epochs_and_the_final_partial_batch_is_kept():
    X, y = _data(n=4096 + 904)
    model = fit_mlp(X, y, {**CANDIDATES[0], "max_epochs": 3})
    assert model.epochs_completed_ == len(model.loss_history_) == 3  # no early stop
    assert model.batch_sizes_ == (4096, 904)  # every observation once per epoch, last batch kept


def test_independent_fits_predict_identically_and_leave_global_state_unchanged():
    X, y = _data()
    rng_state = torch.random.get_rng_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    first, second = (fit_mlp(X, y, CANDIDATES[-1]) for _ in range(2))
    np.testing.assert_array_equal(predict_mlp(first, X), predict_mlp(second, X))
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    assert torch.are_deterministic_algorithms_enabled() == deterministic


def test_invalid_inputs_and_parameters_are_rejected():
    X, y = _data(n=300, p=3)
    params = CANDIDATES[0]
    bad_X = X.copy()
    bad_X[4, 2] = np.nan
    with pytest.raises(DataValidationError, match="predictors contain non-finite"):
        fit_mlp(bad_X, y, params)
    bad_y = y.copy()
    bad_y[7] = np.inf
    with pytest.raises(DataValidationError, match="targets contain non-finite"):
        fit_mlp(X, bad_y, params)
    with pytest.raises(DataValidationError, match="rows"):
        fit_mlp(X, y[:-1], params)
    with pytest.raises(DataValidationError, match="two-dimensional"):
        fit_mlp(X[:, 0], y, params)
    with pytest.raises(DataValidationError, match="at least one"):
        fit_mlp(X[:0], y[:0], params)
    with pytest.raises(ValueError, match="exactly"):
        fit_mlp(X, y, {**params, "dropout": 0.1})
    with pytest.raises(ValueError, match="exactly"):
        fit_mlp(X, y, {k: v for k, v in params.items() if k != "weight_decay"})
    with pytest.raises(ValueError, match="'adam'"):
        fit_mlp(X, y, {**params, "optimizer": "sgd"})

    model = fit_mlp(X, y, {**params, "max_epochs": 1})
    with pytest.raises(DataValidationError, match="prediction predictors contain non-finite"):
        predict_mlp(model, bad_X)
    with pytest.raises(DataValidationError, match="two-dimensional"):
        predict_mlp(model, X[0])
    with pytest.raises(DataValidationError, match="3 predictors"):
        predict_mlp(model, X[:, :2])
