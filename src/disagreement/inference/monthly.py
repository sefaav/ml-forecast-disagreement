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

The other monthly specifications share the basis, the sample rules, the
estimability rule and the column-normalised solve: B drops the bin-specific
q terms (the pre-specified H1 robustness); H2 adds the three centred control
ranks X^c_k and S^c X^c_k on the H2 sample (finite outcome and complete case),
with C re-estimated on that exact sample for the paired comparisons; H3 adds
CW and S^c CW on the H1 sample. The marginal signal slope g(c) of "Reporting
rules for H1 and H2" is computed from the monthly coefficients.

The module is purely numerical: arrays in, MonthlyEstimate objects out. No
data file is read and no inference across months is done here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from disagreement.controls.h2 import INTERACTIONS, RANKS
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
# Specifications

SPECIFICATION_B = "B"  # bin intercepts and bin-local linear terms: the pre-specified H1 robustness
SPECIFICATION_C = "C"  # the primary H1 specification: bin-local linear and quadratic terms
SPECIFICATION_H2 = "H2"  # C plus the three control ranks and their interactions with S^c, on the H2 sample
SPECIFICATION_C_H2 = "C|H2"  # C re-estimated on the exact H2 sample, for the paired comparisons
SPECIFICATION_H3 = "H3"  # C plus CW and S^c CW, on the H1 sample
DEGREE = {SPECIFICATION_B: 1, SPECIFICATION_C: 2, SPECIFICATION_H2: 2, SPECIFICATION_C_H2: 2, SPECIFICATION_H3: 2}
H2_EXTRAS = (*RANKS, *INTERACTIONS)  # three centred control ranks, then S^c times each, in this order
H3_EXTRAS = ("cw", "signal_x_cw")
EXTRAS = {  # the exact extra regressors, in order, that each specification takes after S^c CD^B
    SPECIFICATION_B: (), SPECIFICATION_C: (), SPECIFICATION_C_H2: (), SPECIFICATION_H2: H2_EXTRAS,
    SPECIFICATION_H3: H3_EXTRAS,
}
SLOPE_DISAGREEMENT = (0.1, 0.5, 0.9)  # the CD^B levels c of the marginal signal slopes


def design_matrix(
    basis: BinBasis, sc: np.ndarray, bins: np.ndarray, cd: np.ndarray, degree: int, extra: np.ndarray | None = None
) -> np.ndarray:
    """[bin intercepts, bin z, (bin q if degree 2), CD^B, S^c CD^B, extra columns] in this order."""
    z, q = basis.local(sc, bins)
    indicator = np.eye(basis.n_bins)[bins - 1]
    columns = [indicator, indicator * z[:, None]]
    if degree == 2:
        columns.append(indicator * q[:, None])
    columns += [cd, sc * cd]
    if extra is not None:
        columns.append(extra)
    return np.column_stack(columns)


@dataclass(frozen=True)
class Coefficients:
    """The OLS coefficients of one month in the bin-local basis, in the units of the regressors."""

    intercept: np.ndarray  # alpha_b, one per bin
    linear: np.ndarray  # gamma~_b: the slope on z; the slope on S^c inside bin b is gamma~_b / h_b at z = 0
    quadratic: np.ndarray  # eta~_b: the coefficient of q (empty for specification B)
    disagreement: float  # beta_2
    interaction: float  # beta_3, the H1 coefficient of interest
    extra: dict[str, float]  # H2: the six control terms of H2_EXTRAS; H3: CW and S^c CW; otherwise empty


SpecificationC = Coefficients  # the name of Step 22A


@dataclass(frozen=True)
class MonthlyEstimate:
    """One month's estimate of a specification on one sample, or the explicit reasons it is not estimable."""

    specification: str
    estimable: bool
    reasons: tuple[str, ...]
    n_obs: int
    cell_counts: np.ndarray  # sample observations per bin
    cell_distinct: np.ndarray  # distinct S^c values per bin in the sample
    singular_ratio: float  # sigma_min / sigma_max of the scaled design matrix
    condition_number: float
    basis: BinBasis  # the full-universe constants used; h_b is needed for marginal slopes
    coefficients: Coefficients | None


def _not_estimable(specification: str, reason: str, basis: BinBasis) -> MonthlyEstimate:
    n_bins = basis.n_bins
    return MonthlyEstimate(specification, False, (reason,), 0, np.zeros(n_bins, dtype="int64"),
                           np.zeros(n_bins, dtype="int64"), 0.0, float("inf"), basis, None)


def _extra_columns(extra: np.ndarray, names: tuple[str, ...], n: int, label: str) -> np.ndarray:
    extra = np.asarray(extra, dtype="float64")
    if extra.shape != (n, len(names)):
        raise ValueError(f"{label} must have one row per sample observation and {len(names)} columns")
    if not np.isfinite(extra).all():
        raise ValueError(f"{label} must be finite on the estimation sample: no imputation")
    return extra


def estimate_specification(
    y: np.ndarray,
    sc: np.ndarray,
    bins: np.ndarray,
    cd: np.ndarray,
    basis: BinBasis,
    specification: str = SPECIFICATION_C,
    extra: np.ndarray | None = None,
    extra_names: tuple[str, ...] = (),
) -> MonthlyEstimate:
    """One specification on one sample of a month, with the basis constants computed beforehand on all of U_t.

    `extra` holds the already validated additional regressors (H2 controls and interactions, or CW and S^c CW),
    one column per name, appended after S^c CD^B. The frozen estimability rule is applied to this very design.
    """
    n_bins, degree = basis.n_bins, DEGREE[specification]
    sc, bins, cd = _formation_arrays(sc, bins, cd, n_bins)
    y = np.asarray(y, dtype="float64")
    if y.shape != sc.shape:
        raise ValueError("the outcome must have one value per sample observation")
    if not np.isfinite(y).all():
        raise ValueError("the sample outcome must be finite: the sample is restricted before estimation")
    expected = EXTRAS[specification]
    if tuple(extra_names) != expected:
        raise ValueError(f"specification {specification} takes exactly the extra regressors {list(expected)}, "
                         f"in this order; {list(extra_names)} given")
    if expected:
        if extra is None:
            raise ValueError(f"specification {specification} needs the extra regressors {list(expected)}")
        extra = _extra_columns(extra, expected, sc.size, f"the extra regressors of {specification}")
    elif extra is not None:
        raise ValueError(f"specification {specification} takes no extra regressors")

    counts = np.bincount(bins, minlength=n_bins + 1)[1:]
    distinct = np.array([len(np.unique(sc[bins == b])) for b in range(1, n_bins + 1)])
    X = design_matrix(basis, sc, bins, cd, degree, extra)
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
        k = degree + 1  # bin blocks: intercept, z and, with degree 2, q
        coefficients = Coefficients(
            intercept=beta[:n_bins],
            linear=beta[n_bins:2 * n_bins],
            quadratic=beta[2 * n_bins:3 * n_bins] if degree == 2 else beta[:0],
            disagreement=float(beta[k * n_bins]),
            interaction=float(beta[k * n_bins + 1]),
            extra={name: float(b) for name, b in zip(extra_names, beta[k * n_bins + 2:])},
        )
    return MonthlyEstimate(
        specification=specification,
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


def estimate_month(
    y: np.ndarray, sc: np.ndarray, bins: np.ndarray, cd: np.ndarray, basis: BinBasis
) -> MonthlyEstimate:
    """Specification C on one sample of a month (Step 22A interface)."""
    return estimate_specification(y, sc, bins, cd, basis, SPECIFICATION_C)


def estimate_b(y: np.ndarray, sc: np.ndarray, bins: np.ndarray, cd: np.ndarray, basis: BinBasis) -> MonthlyEstimate:
    """Specification B: C without the bin-specific quadratic terms, same basis constants, same sample rule."""
    return estimate_specification(y, sc, bins, cd, basis, SPECIFICATION_B)


def estimate_h2(
    y: np.ndarray, sc: np.ndarray, bins: np.ndarray, cd: np.ndarray, controls: np.ndarray, basis: BinBasis
) -> MonthlyEstimate:
    """H2 on the H2 sample: C plus the three centred control ranks X^c_k and S^c X^c_k, with the raw S^c."""
    sc_valid = np.asarray(sc, dtype="float64")
    controls = _extra_columns(controls, RANKS, sc_valid.size, "the H2 control ranks")
    if (np.abs(controls) > 1.0).any():
        raise ValueError("the H2 control ranks must lie in [-1, 1]")
    extra = np.column_stack([controls, sc_valid[:, None] * controls])
    return estimate_specification(y, sc, bins, cd, basis, SPECIFICATION_H2, extra, H2_EXTRAS)


def estimate_h3(
    y: np.ndarray, sc: np.ndarray, bins: np.ndarray, cd: np.ndarray, cw: np.ndarray, basis: BinBasis
) -> MonthlyEstimate:
    """H3 on the H1 sample: C plus CW and S^c CW; CD^B and S^c CD^B stay in the regression."""
    sc_valid = np.asarray(sc, dtype="float64")
    cw = np.asarray(cw, dtype="float64")
    cw = _extra_columns(cw.reshape(-1, 1) if cw.ndim == 1 else cw, ("cw",), sc_valid.size, "CW")[:, 0]
    if ((cw < 0.0) | (cw > 1.0)).any():
        raise ValueError("CW must lie in [0, 1]")
    return estimate_specification(y, sc, bins, cd, basis, SPECIFICATION_H3, np.column_stack([cw, sc_valid * cw]),
                                  H3_EXTRAS)


# ----------------------------------------------------------------------------------------------------------------
# One month, every specification, with the frozen sample rules


def estimate_h1_month(
    outcome: np.ndarray, sc: np.ndarray, bins: np.ndarray, cd: np.ndarray, n_bins: int = SIGNAL_BINS
) -> MonthlyEstimate:
    """H1 (specification C) for one month: inputs are all members of U_t; the outcome may be missing for some.

    The basis constants come from every member of U_t; the regression uses the members with a finite outcome.
    """
    sc, bins, cd = _formation_arrays(sc, bins, cd, n_bins)
    outcome = np.asarray(outcome, dtype="float64")
    if outcome.shape != sc.shape:
        raise ValueError("the outcome must have one value per member of U_t")
    basis = BinBasis(*bin_constants(sc, bins, n_bins))
    sample = np.isfinite(outcome)
    if not sample.any():  # an empty H1 sample is a non-estimable month, reported, not an invalid input
        return _not_estimable(SPECIFICATION_C, "no member of U_t has a finite outcome: the H1 sample is empty", basis)
    return estimate_month(outcome[sample], sc[sample], bins[sample], cd[sample], basis)


def estimate_month_specifications(
    outcome: np.ndarray,
    sc: np.ndarray,
    bins: np.ndarray,
    cd: np.ndarray,
    cw: np.ndarray,
    controls: np.ndarray,
    complete_case: np.ndarray,
    n_bins: int = SIGNAL_BINS,
) -> dict[str, MonthlyEstimate]:
    """C, B and H3 on the H1 sample, H2 and C on the H2 sample, for one month given every member of U_t.

    The H1 sample is the members with a finite outcome; the H2 sample is the H1 sample with `complete_case`
    true. The basis constants come from every member of U_t. Control ranks may be missing outside the
    complete cases and are never imputed; a complete case with a missing rank is a data error.
    """
    sc, bins, cd = _formation_arrays(sc, bins, cd, n_bins)
    n = sc.size
    outcome = np.asarray(outcome, dtype="float64")
    complete_case = np.asarray(complete_case)
    if outcome.shape != sc.shape or complete_case.shape != sc.shape:
        raise ValueError("the outcome and complete_case must have one value per member of U_t")
    if complete_case.dtype != np.bool_:
        raise ValueError("complete_case must be strictly Boolean")
    cw = np.asarray(cw, dtype="float64")
    if cw.shape != sc.shape or not (np.isfinite(cw).all() and (cw >= 0.0).all() and (cw <= 1.0).all()):
        raise ValueError("CW must be finite and in [0, 1] for every member of U_t")
    controls = np.asarray(controls, dtype="float64")
    if controls.shape != (n, len(RANKS)):
        raise ValueError(f"the control ranks must have one row per member of U_t and {len(RANKS)} columns")
    if not np.isfinite(controls[complete_case]).all():
        raise ValueError("a complete case has a missing control rank: complete_case is inconsistent with the ranks")

    basis = BinBasis(*bin_constants(sc, bins, n_bins))
    h1 = np.isfinite(outcome)
    h2 = h1 & complete_case
    out: dict[str, MonthlyEstimate] = {}
    if h1.any():
        args = (outcome[h1], sc[h1], bins[h1], cd[h1])
        out[SPECIFICATION_C] = estimate_month(*args, basis)
        out[SPECIFICATION_B] = estimate_b(*args, basis)
        out[SPECIFICATION_H3] = estimate_h3(*args, cw[h1], basis)
    else:
        reason = "no member of U_t has a finite outcome: the H1 sample is empty"
        for spec in (SPECIFICATION_C, SPECIFICATION_B, SPECIFICATION_H3):
            out[spec] = _not_estimable(spec, reason, basis)
    if h2.any():
        args = (outcome[h2], sc[h2], bins[h2], cd[h2])
        out[SPECIFICATION_H2] = estimate_h2(*args, controls[h2], basis)
        out[SPECIFICATION_C_H2] = estimate_specification(*args, basis, SPECIFICATION_C_H2)
    else:
        reason = "no member of U_t has a finite outcome and complete controls: the H2 sample is empty"
        for spec in (SPECIFICATION_H2, SPECIFICATION_C_H2):
            out[spec] = _not_estimable(spec, reason, basis)
    return out


# ----------------------------------------------------------------------------------------------------------------
# Marginal signal slopes


def marginal_slopes(
    coefficients: Coefficients,
    basis: BinBasis,
    controls_at: np.ndarray | None = None,
    levels: tuple[float, ...] = SLOPE_DISAGREEMENT,
) -> dict[float, float]:
    """The monthly marginal signal slope g(c) of "Reporting rules for H1 and H2" for each CD^B level c.

    g(c) = (1 / n_bins) sum_b gamma~_b / h_b + c beta_3 + sum_k delta_k x_k: the equal-weight average over bins of
    the slope of the outcome on S^c at the bin centre (z = 0, where the quadratic term has no derivative), plus the
    disagreement interaction at CD^B = c, plus, for H2 only, the signal x control interactions delta_k at the
    control values x_k (`controls_at`: zero for the primary evaluation, or the monthly means of the centred ranks
    among the H2 complete cases for the descriptive one). Coefficients of H3 have no pre-specified slope.
    """
    names = tuple(coefficients.extra)
    if names not in ((), H2_EXTRAS):
        raise ValueError("marginal slopes are pre-specified for specifications B, C and H2 only")
    if names == ():
        if controls_at is not None:
            raise ValueError("control values apply to H2 coefficients only")
        control_term = 0.0
    else:
        x = np.zeros(len(RANKS)) if controls_at is None else np.asarray(controls_at, dtype="float64")
        if x.shape != (len(RANKS),) or not (np.isfinite(x).all() and (np.abs(x) <= 1.0).all()):
            raise ValueError(f"control values must be {len(RANKS)} finite numbers in [-1, 1]")
        control_term = float(sum(coefficients.extra[name] * xk for name, xk in zip(INTERACTIONS, x)))
    if len(coefficients.linear) != basis.n_bins:
        raise ValueError("the coefficients and the basis do not have the same number of bins")
    centre_slope = float(np.mean(coefficients.linear / basis.half_range))
    return {float(c): centre_slope + float(c) * coefficients.interaction + control_term for c in levels}
