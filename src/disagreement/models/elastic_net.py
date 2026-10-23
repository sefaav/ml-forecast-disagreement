"""Elastic Net family: sklearn ElasticNet regression on target_rank.

Training objective (squared error with an elastic-net penalty):

    1 / (2n) * ||y - Xw||^2 + alpha * l1_ratio * ||w||_1 + alpha * (1 - l1_ratio) / 2 * ||w||^2

Only ``alpha`` and ``l1_ratio`` come from the frozen candidate grid. The
predictors are already monthly cross-sectional ranks, so no scaling is added;
an unpenalized intercept is estimated, and cyclic coordinate descent keeps the
fit deterministic. A convergence warning is an error: an unconverged fit is
never accepted, and no setting is adjusted to make it converge.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet

from disagreement.data.validation import DataValidationError

FIXED_SETTINGS = {"fit_intercept": True, "selection": "cyclic", "max_iter": 10_000, "tol": 1e-6}
TUNED = ("alpha", "l1_ratio")


class ElasticNetConvergenceError(RuntimeError):
    """The coordinate descent did not converge within the fixed budget."""


def _check_inputs(X, y) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype="float64")
    y = np.asarray(y, dtype="float64")
    if X.ndim != 2 or y.ndim != 1:
        raise DataValidationError("X must be two-dimensional and y one-dimensional.")
    if X.shape[0] != y.shape[0]:
        raise DataValidationError(f"X has {X.shape[0]} rows but y has {y.shape[0]}.")
    if X.shape[0] < 1 or X.shape[1] < 1:
        raise DataValidationError("at least one observation and one predictor are required.")
    if not np.isfinite(X).all():
        raise DataValidationError("training predictors contain non-finite values.")
    if not np.isfinite(y).all():
        raise DataValidationError("training targets contain non-finite values.")
    return X, y


def fit_elastic_net(X, y, params: Mapping[str, float]) -> ElasticNet:
    """Fit one frozen Elastic Net candidate ({alpha, l1_ratio}) to target_rank."""
    if set(params) != set(TUNED):
        raise ValueError(f"Elastic Net candidates set exactly {TUNED}; got {sorted(params)}")
    X, y = _check_inputs(X, y)
    model = ElasticNet(alpha=float(params["alpha"]), l1_ratio=float(params["l1_ratio"]), **FIXED_SETTINGS)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            model.fit(X, y)
        except ConvergenceWarning as exc:
            raise ElasticNetConvergenceError(
                f"Elastic Net did not converge (alpha={params['alpha']}, l1_ratio={params['l1_ratio']}, "
                f"max_iter={FIXED_SETTINGS['max_iter']}, tol={FIXED_SETTINGS['tol']})."
            ) from exc
    return model
