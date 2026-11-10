"""Descriptive H1 outcome coverage by signal bin and within-cell disagreement group.

Pre-specified in docs/pre_analysis_plan.md, "Estimation protocol" (Samples):
before any regression, the share of U_t members with a finite evaluation
outcome is reported by signal bin and by tercile of the conditional
disagreement CD^B inside each (month, signal bin) cell. The terciles use the
rank convention of the signal bins, 1 + floor(3 (r - 1) / n) with average
ranks for ties, so tied CD^B values share a group. The diagnostic changes
neither the H1 sample, nor the estimation, nor the inference: it only
describes where outcomes are missing. Months weigh equally in the profile.

The function takes an indicator of a finite outcome, never the outcome
itself, and returns aggregates only. The indicator must be strictly Boolean
and complete: a missing indicator is rejected, never read as available. It
is not run before the estimation engine exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from disagreement.signals.conditional import SIGNAL_BINS, signal_bins

GROUPS = 3
MONTHLY_COLUMNS = ("yyyymm", "signal_bin", "cd_group", "n_stocks", "n_finite", "finite_share")
PROFILE_COLUMNS = ("signal_bin", "cd_group", "n_months", "mean_n_stocks", "mean_finite_share", "min_finite_share")


def disagreement_groups(cd: pd.Series, month: pd.Series, signal_bin: pd.Series, groups: int = GROUPS) -> pd.Series:
    """Within each (month, signal bin) cell, 1 + floor(groups (r - 1) / n) with r the average rank of CD^B."""
    cell = month.astype("int64") * 100 + signal_bin.astype("int64")
    return signal_bins(cd, cell, groups)


def _validated(yyyymm: pd.Series, signal_bin: pd.Series, cd: pd.Series, outcome_finite: pd.Series) -> pd.DataFrame:
    """The four inputs as a frame, after rejecting anything the diagnostic would otherwise misread."""
    lengths = {len(yyyymm), len(signal_bin), len(cd), len(outcome_finite)}
    if len(lengths) != 1:
        raise ValueError("yyyymm, signal_bin, CD^B and the outcome indicator must have the same length")
    cd_values = cd.to_numpy(dtype="float64", na_value=np.nan)
    if not (np.isfinite(cd_values).all() and (cd_values >= 0.0).all() and (cd_values <= 1.0).all()):
        raise ValueError("CD^B must be finite and in [0, 1] for every stock-month")
    if yyyymm.isna().any() or not pd.api.types.is_integer_dtype(yyyymm):
        raise ValueError("yyyymm must be complete integer formation months")
    months = yyyymm.to_numpy().astype("int64")
    if (((months % 100) < 1) | ((months % 100) > 12) | (months // 100 < 1900) | (months // 100 > 2100)).any():
        raise ValueError("invalid yyyymm formation month")
    if signal_bin.isna().any() or not pd.api.types.is_integer_dtype(signal_bin):
        raise ValueError("signal_bin must be complete integers")
    bins = signal_bin.to_numpy().astype("int64")
    if not ((bins >= 1) & (bins <= SIGNAL_BINS)).all():
        raise ValueError(f"signal_bin must lie in [1, {SIGNAL_BINS}]")
    if outcome_finite.isna().any() or not pd.api.types.is_bool_dtype(outcome_finite):
        raise ValueError("the outcome indicator must be strictly Boolean with no missing value")
    return pd.DataFrame(
        {
            "yyyymm": months.astype("int64"),
            "signal_bin": bins,
            "cd": cd_values,
            "finite": outcome_finite.to_numpy(dtype=bool),
        }
    )


def outcome_coverage(
    yyyymm: pd.Series, signal_bin: pd.Series, cd: pd.Series, outcome_finite: pd.Series, groups: int = GROUPS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monthly coverage by (month, signal bin, CD^B group) and its equal-month-weight profile."""
    frame = _validated(yyyymm, signal_bin, cd, outcome_finite)
    frame["cd_group"] = disagreement_groups(frame["cd"], frame["yyyymm"], frame["signal_bin"], groups).to_numpy()
    grouped = frame.groupby(["yyyymm", "signal_bin", "cd_group"], sort=True)["finite"]
    monthly = grouped.agg(n_stocks="size", n_finite="sum").reset_index()
    monthly["finite_share"] = monthly["n_finite"] / monthly["n_stocks"]
    profile = (
        monthly.groupby(["signal_bin", "cd_group"], sort=True)
        .agg(
            n_months=("yyyymm", "size"),
            mean_n_stocks=("n_stocks", "mean"),
            mean_finite_share=("finite_share", "mean"),
            min_finite_share=("finite_share", "min"),
        )
        .reset_index()
    )
    return monthly.loc[:, list(MONTHLY_COLUMNS)], profile.loc[:, list(PROFILE_COLUMNS)]
