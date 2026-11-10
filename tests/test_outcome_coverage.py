"""Descriptive H1 outcome coverage by signal bin and within-cell disagreement group (synthetic data)."""

import numpy as np
import pandas as pd
import pytest

from disagreement.inference.coverage import MONTHLY_COLUMNS, PROFILE_COLUMNS, disagreement_groups, outcome_coverage


def test_groups_follow_the_bin_rank_convention_with_ties():
    month = pd.Series([1] * 7 + [2] * 3)
    signal_bin = pd.Series([1] * 7 + [1] * 3)
    cd = pd.Series([0.0, 0.2, 0.2, 0.5, 0.7, 0.9, 1.0, 0.0, 0.5, 1.0])
    groups = disagreement_groups(cd, month, signal_bin)
    # month 1, n = 7: ranks 1, 2.5, 2.5, 4, 5, 6, 7 -> 1 + floor(3 (r - 1) / 7) = 1, 1, 1, 2, 2, 3, 3
    assert groups[:7].tolist() == [1, 1, 1, 2, 2, 3, 3]  # tied CD share a group
    assert groups[7:].tolist() == [1, 2, 3]  # month 2, n = 3


def test_hand_calculated_coverage_and_equal_month_weights():
    rows = [  # yyyymm, signal_bin, cd, finite
        (200401, 1, 0.1, True), (200401, 1, 0.4, False), (200401, 1, 0.9, True),  # groups 1, 2, 3
        (200401, 2, 0.2, True), (200401, 2, 0.6, True), (200401, 2, 0.8, False),
        (200402, 1, 0.3, False), (200402, 1, 0.5, False), (200402, 1, 0.7, True),
    ]
    frame = pd.DataFrame(rows, columns=["yyyymm", "signal_bin", "cd", "finite"])
    monthly, profile = outcome_coverage(frame["yyyymm"], frame["signal_bin"], frame["cd"], frame["finite"])
    assert list(monthly.columns) == list(MONTHLY_COLUMNS) and list(profile.columns) == list(PROFILE_COLUMNS)
    assert monthly["n_stocks"].tolist() == [1] * 9 and monthly["n_finite"].sum() == 5
    jan_bin1 = monthly[(monthly["yyyymm"] == 200401) & (monthly["signal_bin"] == 1)]
    assert jan_bin1["finite_share"].tolist() == [1.0, 0.0, 1.0]
    p = profile.set_index(["signal_bin", "cd_group"])
    assert p.loc[(1, 1), "n_months"] == 2 and p.loc[(2, 1), "n_months"] == 1
    assert p.loc[(1, 1), "mean_finite_share"] == 0.5 and p.loc[(1, 1), "min_finite_share"] == 0.0  # (1 + 0) / 2
    assert p.loc[(1, 3), "mean_finite_share"] == 1.0  # (1 + 1) / 2, months weigh equally
    assert p.loc[(2, 3), "mean_finite_share"] == 0.0
    # shuffled rows give the same tables
    shuffled = frame.sample(frac=1.0, random_state=0)
    m2, p2 = outcome_coverage(shuffled["yyyymm"], shuffled["signal_bin"], shuffled["cd"], shuffled["finite"])
    pd.testing.assert_frame_equal(m2, monthly)
    pd.testing.assert_frame_equal(p2, profile)
    with pytest.raises(ValueError):
        missing = frame["cd"].where(frame.index != 0, np.nan)
        outcome_coverage(frame["yyyymm"], frame["signal_bin"], missing, frame["finite"])


def _frame(rng, n=90):
    months = np.repeat([200401, 200402, 200403], n // 3)
    return pd.DataFrame({
        "yyyymm": months,
        "signal_bin": rng.integers(1, 4, size=n),
        "cd": rng.integers(0, 11, size=n) / 10,  # ties
        "finite": rng.random(n) < 0.8,
    })


def test_group_assignment_does_not_depend_on_outcome_availability():
    rng = np.random.default_rng(3)
    frame = _frame(rng)
    groups = disagreement_groups(frame["cd"], frame["yyyymm"], frame["signal_bin"])
    monthly_a, _ = outcome_coverage(frame["yyyymm"], frame["signal_bin"], frame["cd"], frame["finite"])
    for other in (~frame["finite"], pd.Series(True, index=frame.index), pd.Series(rng.random(len(frame)) < 0.3)):
        monthly_b, _ = outcome_coverage(frame["yyyymm"], frame["signal_bin"], frame["cd"], other)
        pd.testing.assert_frame_equal(monthly_a[["yyyymm", "signal_bin", "cd_group", "n_stocks"]],
                                      monthly_b[["yyyymm", "signal_bin", "cd_group", "n_stocks"]])
        assert monthly_b["n_finite"].sum() == other.sum()
    np.testing.assert_array_equal(
        disagreement_groups(frame["cd"], frame["yyyymm"], frame["signal_bin"]).to_numpy(), groups.to_numpy()
    )


def test_invalid_inputs_are_rejected_instead_of_being_coerced():
    rng = np.random.default_rng(4)
    frame = _frame(rng)
    ok = (frame["yyyymm"], frame["signal_bin"], frame["cd"])

    def rejected(yyyymm=None, signal_bin=None, cd=None, finite=None, message=""):
        with pytest.raises(ValueError, match=message):
            outcome_coverage(ok[0] if yyyymm is None else yyyymm, ok[1] if signal_bin is None else signal_bin,
                             ok[2] if cd is None else cd, frame["finite"] if finite is None else finite)

    with_nan = frame["finite"].astype("boolean")
    with_nan.iloc[0] = pd.NA
    rejected(finite=with_nan, message="strictly Boolean")  # a missing indicator is never read as available
    rejected(finite=pd.Series([True] * (len(frame) - 1) + [None], dtype=object), message="strictly Boolean")
    rejected(finite=frame["finite"].astype(int), message="strictly Boolean")  # 0/1 integers are not accepted
    rejected(finite=frame["finite"].astype(float), message="strictly Boolean")
    rejected(cd=frame["cd"].where(frame.index != 5, np.inf), message="finite")
    rejected(cd=frame["cd"] + 0.5, message="finite")  # above 1
    rejected(yyyymm=frame["yyyymm"].where(frame.index != 1, 200413), message="invalid yyyymm")
    rejected(yyyymm=frame["yyyymm"].astype(float), message="integer formation months")
    rejected(signal_bin=frame["signal_bin"].where(frame.index != 2, 21), message="signal_bin must lie")
    rejected(signal_bin=frame["signal_bin"].where(frame.index != 2, 0), message="signal_bin must lie")
    rejected(signal_bin=frame["signal_bin"].astype(float), message="complete integers")
    rejected(cd=frame["cd"].iloc[:-1], message="same length")
