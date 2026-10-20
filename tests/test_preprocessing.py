"""Monthly cross-sectional rank preprocessing of predictors (synthetic data)."""

import numpy as np
import pandas as pd
import pytest

from disagreement.data.validation import DataValidationError
from disagreement.preprocessing.ranks import rank_predictors


def _frame(yyyymm, **predictors):
    n = len(next(iter(predictors.values())))
    return pd.DataFrame({"permno": range(1, n + 1), "yyyymm": yyyymm, **predictors})


def test_each_predictor_is_ranked_within_its_month_and_scale_does_not_matter():
    frame = pd.concat(
        [
            _frame(200001, A=[3.0, 1.0, 2.0], B=[3e6, 1e6, 2e6], C=[1.0, 2.0, 3.0]),
            _frame(200002, A=[10.0, 30.0, 20.0], B=[-5.0, 5.0, 0.0], C=[3.0, 2.0, 1.0]),
        ],
        ignore_index=True,
    )
    out = rank_predictors(frame, ["A", "B", "C"])
    assert list(out.columns) == ["permno", "yyyymm", "A", "B", "C"]
    assert out["A"].tolist() == [1.0, -1.0, 0.0, -1.0, 1.0, 0.0]
    assert out["B"].tolist() == out["A"].tolist()  # same ordering, different scale
    assert out["C"].tolist() == [-1.0, 0.0, 1.0, 1.0, 0.0, -1.0]  # not ranked against A or B


def test_average_ranks_for_ties_and_a_constant_predictor_maps_to_zero():
    out = rank_predictors(_frame(200001, A=[1.0, 2.0, 2.0, 4.0], K=[7.0, 7.0, 7.0, 7.0]), ["A", "K"])
    assert out["A"].tolist() == pytest.approx([-1.0, 0.0, 0.0, 1.0])  # ranks 1, 2.5, 2.5, 4
    assert out["K"].tolist() == [0.0] * 4


def test_missing_and_infinite_values_do_not_take_part_and_become_zero():
    out = rank_predictors(
        _frame(200001, A=[1.0, np.nan, 3.0, np.inf, 2.0, -np.inf], S=[np.nan, 5.0, np.nan, np.nan, np.inf, np.nan]),
        ["A", "S"],
    )
    assert out["A"].tolist() == [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0]  # 1, 2, 3 ranked among themselves
    assert out["S"].tolist() == [0.0] * 6  # a single finite value maps to 0


def test_values_are_finite_and_within_minus_one_and_one():
    rng = np.random.default_rng(0)
    values = rng.standard_cauchy((500, 4)) * 1e9
    values[rng.random(values.shape) < 0.3] = np.nan
    frame = pd.concat(
        [_frame(200001 + m, **{f"P{j}": values[100 * m : 100 * (m + 1), j] for j in range(4)}) for m in range(5)],
        ignore_index=True,
    )
    out = rank_predictors(frame, [f"P{j}" for j in range(4)]).iloc[:, 2:].to_numpy()
    assert np.isfinite(out).all() and out.min() == -1.0 and out.max() == 1.0


def test_a_month_does_not_depend_on_other_months_and_input_is_not_mutated():
    month = _frame(200001, A=[0.3, np.nan, -2.0, 0.3, 9.0])
    other = _frame(200002, A=[1e9, -1e9, np.nan, 0.0, 5.0])
    combined = pd.concat([other, month], ignore_index=True)
    before = combined.copy()
    alone = rank_predictors(month, ["A"])
    together = rank_predictors(combined, ["A"]).iloc[5:].reset_index(drop=True)
    pd.testing.assert_frame_equal(alone, together)
    pd.testing.assert_frame_equal(combined, before)  # caller's frame unchanged


def test_identifier_universe_and_outcome_columns_are_never_transformed():
    frame = _frame(200001, A=[1.0, 2.0], ret_next=[0.1, -0.1], target_rank=[1.0, -1.0])
    frame["exchange"], frame["mktcap"] = "NYSE", 100.0
    assert list(rank_predictors(frame, ["A"]).columns) == ["permno", "yyyymm", "A"]
    for column in ("permno", "yyyymm", "exchange", "mktcap", "ret_next", "target_rank"):
        with pytest.raises(DataValidationError, match="not predictors"):
            rank_predictors(frame, ["A", column])
    with pytest.raises(DataValidationError, match="lacks predictors"):
        rank_predictors(frame, ["A", "Missing"])
