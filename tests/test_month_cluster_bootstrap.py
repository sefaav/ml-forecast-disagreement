"""Month-cluster bootstrap on synthetic month labels."""

from typing import Any

import numpy as np
import pytest

from disagreement.bootstrap.months import month_cluster_bootstrap
from disagreement.data.validation import DataValidationError

SIZES = {200001: 2, 200002: 4, 200003: 3, 200004: 5, 200005: 1, 200006: 6}  # stocks per month


def _sample():
    """Rows of six months of unequal size, deliberately not in month order."""
    months = np.concatenate([np.full(n, m) for m, n in SIZES.items()])
    order = np.random.default_rng(12).permutation(len(months))
    return months[order]


def test_one_draw_per_month_and_every_row_of_a_drawn_month_once_per_draw():
    yyyymm = _sample()
    sampled, rows = month_cluster_bootstrap(yyyymm, seed=3)
    assert len(sampled) == len(SIZES)  # M draws, not one per row
    assert set(sampled) <= set(SIZES)
    draws = {m: int((sampled == m).sum()) for m in SIZES}
    assert len(rows) == sum(draws[m] * n for m, n in SIZES.items())
    counts = np.bincount(rows, minlength=len(yyyymm))
    for i, month in enumerate(yyyymm):
        assert counts[i] == draws[month]  # each row appears exactly d_m times; undrawn months not at all


def test_rows_follow_the_draw_order_and_keep_the_within_month_order():
    yyyymm = _sample()
    sampled, rows = month_cluster_bootstrap(yyyymm, seed=3)
    position = 0
    for month in sampled:
        block = rows[position : position + SIZES[month]]
        np.testing.assert_array_equal(block, np.flatnonzero(yyyymm == month))
        position += SIZES[month]
    assert position == len(rows)


def test_a_month_drawn_twice_repeats_its_whole_cross_section():
    yyyymm = _sample()
    sampled, rows = month_cluster_bootstrap(yyyymm, seed=3)
    repeated = [m for m in SIZES if (sampled == m).sum() >= 2]
    assert repeated  # this seed draws at least one month twice
    for month in repeated:
        in_month = np.flatnonzero(yyyymm == month)
        assert np.isin(rows, in_month).sum() == (sampled == month).sum() * len(in_month)
        assert all((rows == i).sum() == (sampled == month).sum() for i in in_month)


def test_draws_depend_on_the_month_labels_and_seed_only():
    yyyymm = _sample()
    first = month_cluster_bootstrap(yyyymm, seed=7)
    second = month_cluster_bootstrap(yyyymm.copy(), seed=7)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    shuffled = yyyymm[::-1]  # same months, different row order
    np.testing.assert_array_equal(month_cluster_bootstrap(shuffled, seed=7)[0], first[0])


def test_global_numpy_random_state_and_input_are_untouched():
    yyyymm = _sample()
    before = yyyymm.copy()
    state = np.random.get_state()
    month_cluster_bootstrap(yyyymm, seed=5)
    after = np.random.get_state()
    assert state[0] == after[0] and np.array_equal(state[1], after[1]) and state[2:] == after[2:]
    np.testing.assert_array_equal(yyyymm, before)


def test_invalid_months_and_seeds_are_rejected():
    with pytest.raises(DataValidationError, match="non-empty"):
        month_cluster_bootstrap(np.array([], dtype="int64"), seed=0)
    with pytest.raises(DataValidationError, match="one-dimensional"):
        month_cluster_bootstrap(np.array([[200001, 200002]]), seed=0)
    with pytest.raises(DataValidationError, match="non-finite"):
        month_cluster_bootstrap(np.array([200001.0, np.nan]), seed=0)
    with pytest.raises(DataValidationError, match="non-finite"):
        month_cluster_bootstrap(np.array([200001.0, np.inf]), seed=0)
    with pytest.raises(DataValidationError, match="integer-valued"):
        month_cluster_bootstrap(np.array([200001.0, 200001.5]), seed=0)
    invalid_seeds: list[Any] = [-1, 1.5, "0", None, True]  # deliberately not valid integers
    for seed in invalid_seeds:
        with pytest.raises(ValueError, match="seed"):
            month_cluster_bootstrap(np.array([200001, 200002]), seed=seed)
