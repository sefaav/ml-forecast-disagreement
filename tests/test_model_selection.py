"""Frozen model-selection protocol (synthetic predictions, no model)."""

import numpy as np
import pytest
import yaml

from disagreement.data.config import DEFAULT_CONFIG
from disagreement.data.validation import DataValidationError
from disagreement.models.selection import (
    FAMILIES,
    model_candidates,
    select_candidate,
    training_objective,
    tuning_seed,
    validation_score,
)

MONTHS = [200001] * 4 + [200002] * 4
TARGET = [-1.0, -1 / 3, 1 / 3, 1.0] * 2


def test_perfect_reversed_constant_and_weak_orderings():
    perfect = [0.1, 0.2, 0.3, 0.4, 500.0, 600.0, 700.0, 800.0]  # months differ in scale
    assert validation_score(MONTHS, perfect, TARGET) == pytest.approx(1.0)
    assert validation_score(MONTHS, perfect[::-1], TARGET) == pytest.approx(-1.0)
    constant = [7.0] * 8
    assert validation_score(MONTHS, constant, TARGET) == 0.0
    weak = [0.2, 0.1, 0.4, 0.3] * 2  # mostly right, two pairs swapped
    assert 0.0 < validation_score(MONTHS, weak, TARGET) < 1.0


def test_monotonic_rescaling_leaves_the_score_unchanged():
    predictions = np.array([0.3, -1.2, 0.8, 0.1, 5.0, 2.0, 9.0, 1.0])
    score = validation_score(MONTHS, predictions, TARGET)
    assert validation_score(MONTHS, np.exp(3 * predictions) - 4, TARGET) == pytest.approx(score)


def test_tied_predictions_get_average_ranks():
    # ranks 1.5, 1.5, 3, 4 against targets -1, -1/3, 1/3, 1
    expected = np.corrcoef([1.5, 1.5, 3.0, 4.0], TARGET[:4])[0, 1]
    assert validation_score(MONTHS[:4], [5.0, 5.0, 9.0, 11.0], TARGET[:4]) == pytest.approx(expected)


def test_months_are_scored_separately_and_weigh_the_same():
    big = 1000
    months = [1] * 2 + [2] * big
    predictions = [0.2, 0.1] + list(range(big))  # month 1 reversed (IC -1), month 2 perfect (IC 1)
    targets = [-1.0, 1.0] + list(np.linspace(-1.0, 1.0, big))
    assert validation_score(months, predictions, targets) == pytest.approx(0.0)  # not about +1


def test_scoring_errors_and_excluded_targets():
    score = validation_score([1, 1, 1, 1], [0.1, 9.0, 0.2, -9.0], [-1.0, np.nan, 1.0, np.inf])
    assert score == pytest.approx(1.0)  # only the two finite targets are scored
    with pytest.raises(DataValidationError, match="non-finite prediction"):
        validation_score([1, 1], [0.1, np.nan], [-1.0, 1.0])
    with pytest.raises(ValueError, match="1 validation months have no observation"):
        validation_score([1, 1, 2, 2], [0.1, 0.2, 0.3, 0.4], [-1.0, 1.0, np.nan, np.inf])
    with pytest.raises(ValueError, match="no cross-sectional variation"):
        validation_score([1, 1, 2, 2], [0.1, 0.2, 0.3, 0.4], [-1.0, 1.0, 0.0, 0.0])


def test_highest_score_wins_and_an_exact_tie_keeps_the_first_candidate():
    assert select_candidate([0.010, 0.031, 0.025]) == 1
    assert select_candidate([0.010, 0.031, 0.031, -0.5]) == 1
    with pytest.raises(DataValidationError):
        select_candidate([0.4, np.nan])


def test_frozen_candidates_objectives_and_metric():
    counts = {family: len(model_candidates(family)) for family in FAMILIES}
    assert counts == {"elastic_net": 12, "random_forest": 12, "lightgbm": 8, "mlp": 8}
    first_last = {family: (model_candidates(family)[0], model_candidates(family)[-1]) for family in FAMILIES}
    assert first_last == {
        "elastic_net": ({"alpha": 1e-4, "l1_ratio": 0.1}, {"alpha": 1e-1, "l1_ratio": 0.9}),
        "random_forest": (
            {"n_estimators": 300, "max_depth": 4, "min_samples_leaf": 20, "max_features": "sqrt"},
            {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 100, "max_features": 0.5},
        ),
        "lightgbm": (
            {"min_child_samples": 100, "n_estimators": 200, "learning_rate": 0.03, "num_leaves": 15},
            {"min_child_samples": 100, "n_estimators": 500, "learning_rate": 0.10, "num_leaves": 31},
        ),
        "mlp": (
            {"optimizer": "adam", "batch_size": 4096, "max_epochs": 30,
             "hidden_layers": [64], "learning_rate": 3e-4, "weight_decay": 1e-5},
            {"optimizer": "adam", "batch_size": 4096, "max_epochs": 30,
             "hidden_layers": [64, 32], "learning_rate": 1e-3, "weight_decay": 1e-3},
        ),
    }
    assert tuning_seed() == 0
    objectives = {family: training_objective(family) for family in FAMILIES}
    assert objectives == {"elastic_net": "squared_error", "random_forest": "squared_error",
                          "lightgbm": "regression", "mlp": "mse"}
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))["model_selection"]
    assert config["validation_metric"] == "monthly_rank_ic"
    assert {family: config[family]["grid"] for family in FAMILIES} == {
        "elastic_net": {"alpha": [1e-4, 1e-3, 1e-2, 1e-1], "l1_ratio": [0.1, 0.5, 0.9]},
        "random_forest": {"max_depth": [4, 8, 12], "min_samples_leaf": [20, 100], "max_features": ["sqrt", 0.5]},
        "lightgbm": {"n_estimators": [200, 500], "learning_rate": [0.03, 0.10], "num_leaves": [15, 31]},
        "mlp": {"hidden_layers": [[64], [64, 32]], "learning_rate": [3e-4, 1e-3], "weight_decay": [1e-5, 1e-3]},
    }
