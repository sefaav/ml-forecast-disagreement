"""Monthly H1 regression engine: specification C of the pre-analysis plan, estimated month by month.

Pre-specified in docs/pre_analysis_plan.md, "Estimation protocol" ("Monthly
regressions", "Samples", "Estimable months"). One OOS formation month is
estimated by equal-weighted OLS in the bin-local basis

    y_i = sum_b 1[b(i) = b] (alpha_b + gamma~_b z_i + eta~_b q_i) + beta_2 CD_i + beta_3 S^c_i CD_i + e_i,

with z_i = (S^c_i - c_b) / h_b, q_i = z_i^2 - mean_b(z^2), where c_b, h_b and
mean_b(z^2) are the mean, the half-range and the mean squared z of S^c in the
cell (t, b) over every member of U_t: they are computed before the outcome
restriction and never recomputed on a subsample. The regressors are the
original centred ensemble signal S^c = 2 (S - 0.5), the existing signal bins
and the existing conditional disagreement CD^B; nothing is re-ranked here.
The H1 sample is the members of U_t with a finite outcome.

A month is estimable if every cell holds at least 30 sample observations with
at least 3 distinct values of S^c and if the design matrix, columns scaled to
unit Euclidean norm, has sigma_min / sigma_max > 1e-8. A month that fails is
reported as such, with the reasons, and carries no coefficients: it is never
regularised, re-specified or silently dropped.

The module is purely numerical: arrays in, a MonthlyEstimate out. No data
file is read and no inference across months is done here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from disagreement.data.validation import DataValidationError
from disagreement.signals.conditional import SIGNAL_BINS

MIN_CELL = 30  # the estimability rule of "Estimable months"
MIN_DISTINCT = 3
MIN_SINGULAR_RATIO = 1e-8


# ----------------------------------------------------------------------------------------------------------------
# Cells of a sorted array


@dataclass(frozen=True)
class Cells:
    """Contiguous groups of arrays sorted by group: offsets and sizes."""

    starts: np.ndarray
    counts: np.ndarray

    def sum(self, v: np.ndarray) -> np.ndarray:
        return np.add.reduceat(v, self.starts)

    def mean(self, v: np.ndarray) -> np.ndarray:
        return self.sum(v) / self.counts

    def expand(self, g: np.ndarray) -> np.ndarray:
        return np.repeat(g, self.counts)


def cells_of(codes: np.ndarray) -> Cells:
    """The cells of an array of group codes that is already sorted by code."""
    if codes.size == 0:
        raise ValueError("no observations")
    if (np.diff(codes) < 0).any():
        raise ValueError("codes must be sorted")
    starts = np.concatenate([[0], np.flatnonzero(np.diff(codes)) + 1])
    return Cells(starts, np.diff(np.append(starts, codes.size)))


# ----------------------------------------------------------------------------------------------------------------
# Bin-local basis


def bin_constants(sc: np.ndarray, bins: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cell mean, half-range of S^c and mean of z^2 for bins 1..n_bins, from every member of the month's cell."""
    order = np.argsort(bins, kind="stable")
    sorted_bins, sorted_sc = bins[order], sc[order]
    present = np.unique(sorted_bins)
    if len(present) != n_bins or present[0] != 1 or present[-1] != n_bins:
        raise DataValidationError(f"a month does not populate all {n_bins} signal bins.")
    cells = cells_of(sorted_bins)
    c = cells.mean(sorted_sc)
    h = (np.maximum.reduceat(sorted_sc, cells.starts) - np.minimum.reduceat(sorted_sc, cells.starts)) / 2.0
    h = np.where(h > 0, h, 1.0)  # a constant cell gives z = 0; the estimability check reports it
    z = (sorted_sc - cells.expand(c)) / cells.expand(h)
    return c, h, cells.mean(z * z)


def local_basis(sc: np.ndarray, bins: np.ndarray, n_bins: int, degree: int) -> np.ndarray:
    """Bin-local basis of "Monthly regressions": per bin an intercept, z = (S^c - c_b)/h_b and, with degree 2, q."""
    c, h, z2 = bin_constants(sc, bins, n_bins)
    idx = bins - 1
    z = (sc - c[idx]) / h[idx]
    q = z * z - z2[idx]
    indicator = np.eye(n_bins)[idx]
    columns = [indicator, indicator * z[:, None]]
    if degree == 2:
        columns.append(indicator * q[:, None])
    return np.hstack(columns)


def original_basis(sc: np.ndarray, bins: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin intercepts and bin-specific S^c and (S^c)^2 without rescaling: the same span as local_basis(degree 2)."""
    indicator = np.eye(n_bins)[bins - 1]
    return np.hstack([indicator, indicator * sc[:, None], indicator * (sc * sc)[:, None]])


def scaled_condition(X: np.ndarray) -> tuple[float, float]:
    """(condition number, sigma_min / sigma_max) of X after scaling every column to unit Euclidean norm."""
    norms = np.linalg.norm(X, axis=0)
    if not (norms > 0).all():
        return float("inf"), 0.0
    sv = np.linalg.svd(X / norms, compute_uv=False)
    return float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"), float(sv[-1] / sv[0])


@dataclass(frozen=True)
class BinBasis:
    """The formation-time constants c_b, h_b and mean_b(z^2) of one month, from every member of U_t."""

    centre: np.ndarray
    half_range: np.ndarray
    mean_z2: np.ndarray

    @property
    def n_bins(self) -> int:
        return len(self.centre)

    def local(self, sc: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """z = (S^c - c_b) / h_b and q = z^2 - mean_b(z^2) of observations in the given bins."""
        idx = bins - 1
        z = (sc - self.centre[idx]) / self.half_range[idx]
        return z, z * z - self.mean_z2[idx]

    def design(self, sc: np.ndarray, bins: np.ndarray, cd: np.ndarray) -> np.ndarray:
        """Specification C: [bin intercepts, bin z, bin q, CD^B, S^c CD^B], 3 n_bins + 2 columns."""
        z, q = self.local(sc, bins)
        indicator = np.eye(self.n_bins)[bins - 1]
        return np.column_stack([indicator, indicator * z[:, None], indicator * q[:, None], cd, sc * cd])


def _formation_arrays(
    sc: np.ndarray, bins: np.ndarray, cd: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three regressor arrays as float64 / int64 after rejecting anything the engine would misread."""
    sc, bins, cd = np.asarray(sc), np.asarray(bins), np.asarray(cd)
    if not (sc.ndim == bins.ndim == cd.ndim == 1 and sc.shape == bins.shape == cd.shape):
        raise ValueError("S^c, signal_bin and CD^B must be one-dimensional arrays of the same length")
    if sc.size == 0:
        raise ValueError("no observations")
    if not np.issubdtype(bins.dtype, np.integer):
        raise ValueError("signal_bin must be integers")
    if not ((bins >= 1) & (bins <= n_bins)).all():
        raise ValueError(f"signal_bin must lie in [1, {n_bins}]")
    sc = sc.astype("float64")
    if not (np.isfinite(sc).all() and (np.abs(sc) <= 1.0).all()):
        raise ValueError("S^c must be finite and in [-1, 1]")
    cd = cd.astype("float64")
    if not (np.isfinite(cd).all() and (cd >= 0.0).all() and (cd <= 1.0).all()):
        raise ValueError("CD^B must be finite and in [0, 1]")
    return sc, bins.astype("int64"), cd


def bin_basis(sc: np.ndarray, bins: np.ndarray, n_bins: int = SIGNAL_BINS) -> BinBasis:
    """The basis constants of one month from every member of U_t (before any outcome restriction)."""
    sc, bins, _ = _formation_arrays(sc, bins, np.zeros_like(sc, dtype="float64"), n_bins)
    return BinBasis(*bin_constants(sc, bins, n_bins))


# ----------------------------------------------------------------------------------------------------------------
# Monthly estimation


@dataclass(frozen=True)
class SpecificationC:
    """The OLS coefficients of one month in the bin-local basis."""

    intercept: np.ndarray  # alpha_b, one per bin
    linear: np.ndarray  # gamma~_b: the slope on z; the slope on S^c inside bin b is gamma~_b / h_b at z = 0
    quadratic: np.ndarray  # eta~_b: the coefficient of q
    disagreement: float  # beta_2
    interaction: float  # beta_3, the H1 coefficient of interest


@dataclass(frozen=True)
class MonthlyEstimate:
    """One month's estimate of specification C on one sample, or the explicit reasons it is not estimable."""

    estimable: bool
    reasons: tuple[str, ...]
    n_obs: int
    cell_counts: np.ndarray  # sample observations per bin
    cell_distinct: np.ndarray  # distinct S^c values per bin in the sample
    singular_ratio: float  # sigma_min / sigma_max of the scaled design matrix
    condition_number: float
    basis: BinBasis  # the full-universe constants used; h_b is needed for marginal slopes
    coefficients: SpecificationC | None


def estimate_month(
    y: np.ndarray, sc: np.ndarray, bins: np.ndarray, cd: np.ndarray, basis: BinBasis
) -> MonthlyEstimate:
    """Specification C on one sample of a month, with the basis constants computed beforehand on all of U_t."""
    n_bins = basis.n_bins
    sc, bins, cd = _formation_arrays(sc, bins, cd, n_bins)
    y = np.asarray(y, dtype="float64")
    if y.shape != sc.shape:
        raise ValueError("the outcome must have one value per sample observation")
    if not np.isfinite(y).all():
        raise ValueError("the sample outcome must be finite: the H1 sample is restricted before estimation")

    counts = np.bincount(bins, minlength=n_bins + 1)[1:]
    distinct = np.array([len(np.unique(sc[bins == b])) for b in range(1, n_bins + 1)])
    X = basis.design(sc, bins, cd)
    condition_number, ratio = scaled_condition(X)

    reasons = []
    if (counts < MIN_CELL).any():
        reasons.append(f"{int((counts < MIN_CELL).sum())} cells with fewer than {MIN_CELL} observations "
                       f"(minimum {int(counts.min())})")
    if (distinct < MIN_DISTINCT).any():
        reasons.append(f"{int((distinct < MIN_DISTINCT).sum())} cells with fewer than {MIN_DISTINCT} distinct S^c "
                       f"values (minimum {int(distinct.min())})")
    if not ratio > MIN_SINGULAR_RATIO:
        reasons.append(f"scaled singular-value ratio {ratio:.2e} not above {MIN_SINGULAR_RATIO:g}")

    coefficients = None
    if not reasons:
        # solved on the column-normalised design that the estimability criterion judged, then mapped back to the
        # original units: a raw column of tiny norm (a very narrow S^c range) would otherwise be truncated by the
        # least-squares rank cut-off although the scaled design is well conditioned
        norms = np.linalg.norm(X, axis=0)  # all positive: a zero column fails the singular-value criterion
        beta = np.linalg.lstsq(X / norms, y, rcond=None)[0] / norms
        coefficients = SpecificationC(
            intercept=beta[:n_bins],
            linear=beta[n_bins:2 * n_bins],
            quadratic=beta[2 * n_bins:3 * n_bins],
            disagreement=float(beta[3 * n_bins]),
            interaction=float(beta[3 * n_bins + 1]),
        )
    return MonthlyEstimate(
        estimable=not reasons,
        reasons=tuple(reasons),
        n_obs=int(sc.size),
        cell_counts=counts,
        cell_distinct=distinct,
        singular_ratio=ratio,
        condition_number=condition_number,
        basis=basis,
        coefficients=coefficients,
    )


def estimate_h1_month(
    outcome: np.ndarray, sc: np.ndarray, bins: np.ndarray, cd: np.ndarray, n_bins: int = SIGNAL_BINS
) -> MonthlyEstimate:
    """H1 for one month: inputs are all members of U_t; the outcome may be missing (non-finite) for some.

    The basis constants come from every member of U_t; the regression uses the members with a finite outcome.
    """
    sc, bins, cd = _formation_arrays(sc, bins, cd, n_bins)
    outcome = np.asarray(outcome, dtype="float64")
    if outcome.shape != sc.shape:
        raise ValueError("the outcome must have one value per member of U_t")
    basis = BinBasis(*bin_constants(sc, bins, n_bins))
    sample = np.isfinite(outcome)
    if not sample.any():  # an empty H1 sample is a non-estimable month, reported, not an invalid input
        return MonthlyEstimate(
            estimable=False,
            reasons=("no member of U_t has a finite outcome: the H1 sample is empty",),
            n_obs=0,
            cell_counts=np.zeros(n_bins, dtype="int64"),
            cell_distinct=np.zeros(n_bins, dtype="int64"),
            singular_ratio=0.0,
            condition_number=float("inf"),
            basis=basis,
            coefficients=None,
        )
    return estimate_month(outcome[sample], sc[sample], bins[sample], cd[sample], basis)
