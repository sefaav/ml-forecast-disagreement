"""MLP family: a small fully connected PyTorch network trained on target_rank.

Each candidate sets ``optimizer``, ``batch_size`` and ``max_epochs`` (fixed
across candidates) and ``hidden_layers``, ``learning_rate`` and
``weight_decay`` (tuned) from the frozen grid. Every hidden layer is Linear
followed by ReLU, and the output is a single Linear unit without activation,
with PyTorch's default initialization. Training minimizes the mean squared
error with Adam (coupled L2 weight decay, not AdamW) for exactly
``max_epochs`` epochs, without early stopping, scheduler or gradient
clipping. Each epoch visits every observation once, in mini-batches of
``batch_size`` taken from a fresh permutation, the last smaller batch
included. The tuning seed fixes the initialization and the sequence of
permutations; training and prediction run on the CPU in float32 with
PyTorch's deterministic algorithms, so repeated runs are reproducible in a
fixed software and hardware environment. The caller's random state and
deterministic setting are restored afterwards. The predictors are already
monthly cross-sectional ranks with missing values set to 0, so no
preprocessing is added.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn

from disagreement.data.validation import DataValidationError

from .elastic_net import _check_inputs
from .selection import tuning_seed

FIXED = ("optimizer", "batch_size", "max_epochs")  # same value in every candidate
TUNED = ("hidden_layers", "learning_rate", "weight_decay")  # vary across candidates
CANDIDATE_PARAMS = (*FIXED, *TUNED)  # keys of each candidate from the frozen grid
ADAM_SETTINGS = {
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "amsgrad": False,
    "decoupled_weight_decay": False,
    "foreach": False,
    "fused": False,
}
LOSS_REDUCTION = "mean"  # nn.MSELoss averages the squared errors of a mini-batch


class MLP(nn.Module):
    """Linear -> ReLU per hidden layer, then a linear scalar output."""

    # Training protocol recorded by fit_mlp.
    batch_size: int
    learning_rate: float
    weight_decay: float
    epochs_completed_: int
    loss_history_: list[float]
    batch_sizes_: tuple[int, ...]

    def __init__(self, n_features: int, hidden_layers: Sequence[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = n_features
        for size in hidden_layers:
            layers += [nn.Linear(width, size, device="cpu", dtype=torch.float32), nn.ReLU()]
            width = size
        layers.append(nn.Linear(width, 1, device="cpu", dtype=torch.float32))
        self.network = nn.Sequential(*layers)
        self.n_features_in_ = n_features
        self.hidden_layers = tuple(hidden_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


@contextmanager
def _deterministic() -> Iterator[None]:
    """PyTorch deterministic algorithms, restoring the caller's setting afterwards."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def fit_mlp(X, y, params: Mapping[str, object]) -> MLP:
    """Train one frozen MLP candidate on target_rank."""
    if set(params) != set(CANDIDATE_PARAMS):
        raise ValueError(f"MLP candidates set exactly {CANDIDATE_PARAMS}; got {sorted(params)}")
    if params["optimizer"] != "adam":
        raise ValueError(f"the MLP optimizer must be 'adam'; got {params['optimizer']!r}")
    X, y = _check_inputs(X, y)
    hidden_layers = [int(size) for size in params["hidden_layers"]]  # type: ignore[union-attr]
    batch_size, max_epochs = int(params["batch_size"]), int(params["max_epochs"])  # type: ignore[arg-type]
    learning_rate, weight_decay = float(params["learning_rate"]), float(params["weight_decay"])  # type: ignore[arg-type]
    predictors = torch.from_numpy(X.astype(np.float32))
    targets = torch.from_numpy(y.astype(np.float32))
    n = len(targets)
    seed = tuning_seed()

    with _deterministic(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)  # initialization
        model = MLP(X.shape[1], hidden_layers)
        shuffle = torch.Generator(device="cpu").manual_seed(seed)  # epoch permutations
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay, **ADAM_SETTINGS
        )
        loss_fn = nn.MSELoss(reduction=LOSS_REDUCTION)
        model.train()
        loss_history, batch_sizes = [], []
        for _ in range(max_epochs):
            order = torch.randperm(n, generator=shuffle)
            total, batch_sizes = 0.0, []
            for start in range(0, n, batch_size):  # the last, smaller batch is kept
                batch = order[start : start + batch_size]
                optimizer.zero_grad()
                loss = loss_fn(model(predictors[batch]), targets[batch])
                loss.backward()
                optimizer.step()
                total += loss.item() * len(batch)
                batch_sizes.append(len(batch))
            loss_history.append(total / n)  # mean training loss over the epoch
        model.eval()

    model.batch_size = batch_size
    model.learning_rate = learning_rate
    model.weight_decay = weight_decay
    model.epochs_completed_ = len(loss_history)
    model.loss_history_ = loss_history
    model.batch_sizes_ = tuple(batch_sizes)  # mini-batch sizes of an epoch
    return model


def predict_mlp(model: MLP, X) -> np.ndarray:
    """CPU float32 forward pass without gradients; the model is not modified."""
    X = np.asarray(X, dtype="float64")
    if X.ndim != 2 or X.shape[1] != model.n_features_in_:
        raise DataValidationError(f"X must be two-dimensional with {model.n_features_in_} predictors.")
    if not np.isfinite(X).all():
        raise DataValidationError("prediction predictors contain non-finite values.")
    with _deterministic(), torch.no_grad():
        predictions = model(torch.from_numpy(X.astype(np.float32))).numpy().astype("float64")
    if not np.isfinite(predictions).all():
        raise DataValidationError("the MLP produced non-finite predictions.")
    return predictions
