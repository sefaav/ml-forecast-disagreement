"""Walk-forward calendar of the frozen baseline (calendar objects only, no data)."""

import pytest

from disagreement.walkforward.calendar import Episode, annual_episodes, baseline_episodes, month_range

EPISODES = baseline_episodes()


def _blocks(e: Episode):
    return (
        month_range(e.train_start, e.train_end),
        month_range(e.validation_start, e.validation_end),
        month_range(e.test_start, e.test_end),
    )


def test_forty_annual_episodes_cover_480_oos_months_from_1985_to_2024():
    assert len(EPISODES) == 40
    oos = [m for e in EPISODES for m in month_range(e.test_start, e.test_end)]
    assert len(oos) == len(set(oos)) == 480  # no OOS month in two episodes
    assert oos == month_range(198501, 202412)  # continuous coverage


def test_first_and_final_episodes():
    assert EPISODES[0] == Episode(1985, 197001, 197912, 198001, 198412, 198501, 198512)
    assert EPISODES[-1] == Episode(2024, 197001, 201812, 201901, 202312, 202401, 202412)


def test_expanding_training_rolling_validation_and_one_year_test():
    for k, e in enumerate(EPISODES):
        train, validation, test = _blocks(e)
        assert e.train_start == 197001
        assert e.train_end == 197912 + 100 * k  # one more year per episode
        assert len(validation) == 60 and e.validation_start == 198001 + 100 * k
        assert len(test) == 12 and e.oos_year == 1985 + k


def test_blocks_are_chronological_contiguous_and_disjoint():
    for e in EPISODES:
        train, validation, test = _blocks(e)
        assert month_range(e.train_start, e.test_end) == train + validation + test  # no gap, no overlap
        assert not (set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test))


def test_eligibility_uses_training_only_and_history_ends_in_december_before_the_oos_year():
    for e in EPISODES:
        train, validation, test = _blocks(e)
        assert e.eligibility_cutoff == e.train_end == max(train) < min(validation)
        assert month_range(e.eligibility_cutoff, e.validation_start)[1:] == [e.validation_start]  # immediately after
        assert e.history_end == e.validation_end == (e.oos_year - 1) * 100 + 12
        assert month_range(e.history_start, e.history_end) == train + validation
        assert e.history_end < min(test)


def test_each_oos_year_becomes_the_last_validation_year_of_the_next_episode():
    for current, following in zip(EPISODES[:-1], EPISODES[1:]):
        assert following.oos_year == current.oos_year + 1
        assert month_range(following.validation_end - 11, following.validation_end) == month_range(
            current.test_start, current.test_end
        )  # e.g. 198501-198512 is OOS in episode 1985 and the last validation year in 1986
        assert following.history_end == current.test_end < following.test_start  # only realized history


def test_no_month_outside_the_baseline_sample():
    months = {m for e in EPISODES for block in _blocks(e) for m in block}
    assert min(months) == 197001 and max(months) == 202412
    with pytest.raises(ValueError):
        annual_episodes(197001, 202412, 1975, 5)  # no training year before validation
    with pytest.raises(ValueError):
        annual_episodes(197001, 202406, 1985, 5)  # incomplete final OOS year
