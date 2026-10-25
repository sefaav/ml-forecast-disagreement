"""OOS bootstrap-replica forecasting orchestration on synthetic files."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from disagreement.bootstrap.months import month_cluster_bootstrap
from disagreement.data.config import DEFAULT_CONFIG, DataPaths, load_data_config
from disagreement.models.selection import model_candidates, validation_score
from disagreement.preprocessing.ranks import rank_predictors
from disagreement.walkforward import forecast
from disagreement.walkforward.calendar import baseline_episodes, month_range

OSAP = ["A", "B", "C", "Late"]  # Late is observed only from 2003, after both training cutoffs
CRSP = ["Price", "Size", "STreversal"]
TRAIN, VALIDATION, OOS = (200001, 200112), (200201, 200312), (200401, 200412)  # episode 2004
MISSING_TARGET = (3, 200006)  # a training row without target_rank
WINNER = 5  # index of the fake winning candidate


def _setup(tmp_path):
    """Synthetic panel 2000-2005 and targets through 2003 only, with a config pinning them."""
    rng = np.random.default_rng(0)
    rows = []
    for month in month_range(200001, 200512):
        for permno in range(1, 11 + month % 3):  # unequal cross-sections
            rows.append({"permno": permno, "yyyymm": month, "exchange": "NYSE", "mktcap": 100.0})
    panel = pd.DataFrame(rows)
    for name in OSAP[:3] + CRSP:
        panel[name] = rng.standard_normal(len(panel))
    panel["Late"] = np.where(panel["yyyymm"] >= 200301, rng.standard_normal(len(panel)), np.nan)
    target = panel.loc[panel["yyyymm"] <= 200312, ["permno", "yyyymm"]].copy()  # no OOS target file
    target["ret_next"] = 0.0
    target["target_rank"] = np.tanh(panel.loc[target.index, "A"] + 0.3 * rng.standard_normal(len(target)))
    target.loc[(target["permno"] == MISSING_TARGET[0]) & (target["yyyymm"] == MISSING_TARGET[1]), "target_rank"] = np.nan

    paths = DataPaths(tmp_path / "data")
    paths.panel.mkdir(parents=True)
    paths.target.mkdir(parents=True)
    for year in range(2000, 2006):
        panel[panel["yyyymm"] // 100 == year].to_parquet(paths.panel / f"panel_{year}.parquet", index=False)
    for year in range(2000, 2004):
        target[target["yyyymm"] // 100 == year].to_parquet(paths.target / f"target_{year}.parquet", index=False)

    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["data"]["sample"] = {"start": 200001, "end": 200512}
    config["data"]["sources"]["osap"]["predictors"] = len(OSAP)
    config["walk_forward"] = {"first_oos_year": 2004, "validation_years": 2}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return paths, config_path, panel, target


class Recorder:
    """Fake families and spies recording what the orchestration passes on."""

    def __init__(self, monkeypatch, families, winner_params):
        self.events = []
        for family in families:
            monkeypatch.setitem(forecast._FAMILIES, family, (self._fit(family), self._predict))
        self.winner_params = winner_params
        monkeypatch.setattr(forecast, "month_cluster_bootstrap", self._bootstrap)
        monkeypatch.setattr(forecast, "validation_score", self._score)

    def _fit(self, family):
        def fit(X, y, params):
            self.events.append(("fit", family, dict(params), np.asarray(y).copy()))
            return dict(params)

        return fit

    def _predict(self, model, X):
        sign = 1.0 if model == self.winner_params else -1.0  # only the winner ranks like A
        return 1000.0 + sign * X[:, 0]  # raw values far outside rank space

    def _bootstrap(self, yyyymm, *, seed):
        self.events.append(("bootstrap", np.asarray(yyyymm).copy(), seed))
        return month_cluster_bootstrap(yyyymm, seed=seed)

    def _score(self, yyyymm, prediction, target):
        self.events.append(("score", np.asarray(yyyymm).copy(), np.asarray(target).copy()))
        return validation_score(yyyymm, prediction, target)


@pytest.fixture
def run(tmp_path, monkeypatch):
    paths, config_path, panel, target = _setup(tmp_path)
    candidates = model_candidates("elastic_net", config_path)

    def go(families=("elastic_net",)):
        recorder = Recorder(monkeypatch, families, candidates[WINNER])
        forecast.run_baseline_forecasts(paths, config_path, families, [2004], log=lambda _: None)
        return recorder

    return paths, config_path, panel, target, candidates, go


def test_replica_seeds():
    seeds = forecast.bootstrap_seeds(0, 2004, 5)
    assert len(seeds) == len(set(seeds)) == 5
    assert seeds == forecast.bootstrap_seeds(0, 2004, 5)
    assert not set(seeds) & set(forecast.bootstrap_seeds(0, 2005, 5))
    assert seeds != forecast.bootstrap_seeds(1, 2004, 5)
    assert forecast.bootstrap_settings() == (5, 0)


def test_selection_on_training_precedes_bootstrap_of_the_whole_history(run):
    paths, config_path, panel, target, candidates, go = run
    events = go().events
    kinds = [event[0] for event in events]
    n = len(candidates)
    assert kinds[: 2 * n] == ["fit", "score"] * n  # every candidate before any resampling
    assert kinds[2 * n :] == ["bootstrap", "fit"] * 5

    in_train = target["yyyymm"].between(*TRAIN) & target["target_rank"].notna()
    expected_train = np.sort(target.loc[in_train, "target_rank"].to_numpy())
    for _, _, params, y in events[0 : 2 * n : 2]:
        np.testing.assert_array_equal(np.sort(y), expected_train)  # original training block, finite targets
    assert [event[2] for event in events[0 : 2 * n : 2]] == candidates
    in_validation = target["yyyymm"].between(*VALIDATION)
    for _, months, scored_target in events[1 : 2 * n : 2]:
        assert set(months) == set(month_range(*VALIDATION))
        assert len(scored_target) == in_validation.sum()

    history_months = month_range(TRAIN[0], VALIDATION[1])
    for _, months, seed in events[2 * n :: 2]:
        assert set(months) == set(history_months) and len(months) == len(target)  # no OOS month
    for _, _, params, y in events[2 * n + 1 :: 2]:
        assert params == candidates[WINNER]  # the frozen selected candidate
        assert np.isfinite(y).all() and np.abs(y).max() <= 1.0

    metadata = json.loads((paths.oos_predictions / "elastic_net" / "2004.json").read_text())
    assert metadata["selected_candidate_index"] == WINNER and metadata["selected_params"] == candidates[WINNER]
    assert metadata["eligible_predictors"] == ["A", "B", "C", *CRSP]  # Late: training block only
    assert metadata["eligibility_cutoff"] == TRAIN[1]
    assert metadata["bootstrap_seeds"] == forecast.bootstrap_seeds(0, 2004, 5)
    assert metadata["model_seed"] == 0


def test_missing_targets_stay_in_the_month_draw_but_leave_the_fit(run):
    paths, config_path, panel, target, candidates, go = run
    events = go().events
    n = len(candidates)
    data = forecast.prepare_episode(paths, load_data_config(config_path), baseline_episodes(config_path)[0])
    history = np.flatnonzero(data.months <= VALIDATION[1])
    missing = np.flatnonzero(np.isnan(data.target[history]))
    assert len(missing) == 1  # the deliberately missing training target
    fit_rows = []
    for (_, months, seed), (_, _, _, y) in zip(events[2 * n :: 2], events[2 * n + 1 :: 2]):
        assert MISSING_TARGET[1] in set(months)  # its month remains drawable
        _, sample = month_cluster_bootstrap(months, seed=seed)
        drawn_missing = int(np.isin(sample, missing).sum())
        assert len(y) == len(sample) - drawn_missing and np.isfinite(y).all()  # dropped, not imputed
        fit_rows.append(len(y))
    metadata = json.loads((paths.oos_predictions / "elastic_net" / "2004.json").read_text())
    assert metadata["replica_fit_rows"] == fit_rows
    assert metadata["n_history_rows_without_target"] == 1


def test_families_share_the_bootstrap_draws(run):
    paths, config_path, panel, target, candidates, go = run
    events = go(("elastic_net", "lightgbm")).events
    draws = [(event[1], event[2]) for event in events if event[0] == "bootstrap"]
    assert len(draws) == 10
    for (months_en, seed_en), (months_lgbm, seed_lgbm) in zip(draws[:5], draws[5:]):
        assert seed_en == seed_lgbm
        np.testing.assert_array_equal(months_en, months_lgbm)
    seeds = [json.loads((paths.oos_predictions / f / "2004.json").read_text())["bootstrap_seeds"]
             for f in ("elastic_net", "lightgbm")]
    assert seeds[0] == seeds[1]


def test_every_oos_row_gets_one_raw_prediction_per_replica(run):
    paths, config_path, panel, target, candidates, go = run
    go()
    out = pd.read_parquet(paths.oos_predictions / "elastic_net" / "2004.parquet")
    oos = panel[panel["yyyymm"].between(*OOS)]
    assert list(out.columns) == ["permno", "yyyymm", "family", "replica", "prediction"]
    assert len(out) == 5 * len(oos)
    assert not out.duplicated(["permno", "yyyymm", "family", "replica"]).any()
    per_key = out.groupby(["permno", "yyyymm"])["replica"].apply(lambda replicas: sorted(replicas))
    assert len(per_key) == len(oos) and all(r == [0, 1, 2, 3, 4] for r in per_key)
    assert out.equals(out.sort_values(["yyyymm", "permno", "replica"], ignore_index=True))
    ranked = rank_predictors(oos, ["A"]).set_index(["permno", "yyyymm"])["A"]
    expected = 1000.0 + ranked.loc[list(zip(out["permno"], out["yyyymm"]))].to_numpy()
    np.testing.assert_allclose(out["prediction"].to_numpy(), expected)  # raw model output, not ranked


def test_real_elastic_net_run_is_reproducible(tmp_path):
    paths, config_path, panel, target = _setup(tmp_path)
    outputs = []
    for _ in range(2):
        forecast.run_baseline_forecasts(paths, config_path, ["elastic_net"], [2004], log=lambda _: None)
        outputs.append(
            (
                pd.read_parquet(paths.oos_predictions / "elastic_net" / "2004.parquet"),
                json.loads((paths.oos_predictions / "elastic_net" / "2004.json").read_text()),
            )
        )
    pd.testing.assert_frame_equal(outputs[0][0], outputs[1][0])
    assert outputs[0][1] == outputs[1][1]
    assert np.isfinite(outputs[0][0]["prediction"]).all()
    assert not list((paths.oos_predictions / "elastic_net").glob("*.tmp"))


def test_completion_marker_is_written_last_and_only_on_success(tmp_path):
    directory = tmp_path / "elastic_net"
    frame = pd.DataFrame(
        {"permno": [1, 2], "yyyymm": [200401, 200401], "family": "elastic_net", "replica": [0, 0], "prediction": [0.1, 0.2]}
    )
    forecast._write(directory, 2004, frame, {"oos_year": 2004})
    assert json.loads((directory / "2004.json").read_text()) == {"oos_year": 2004}

    with pytest.raises(TypeError):  # metadata that cannot be serialized fails before the final replace
        forecast._write(directory, 2004, frame, {"oos_year": object()})
    assert not (directory / "2004.json").exists()  # no completion marker after a failure
    assert sorted(p.name for p in directory.iterdir()) == ["2004.parquet"]  # no temporary file left


def test_a_historical_panel_row_without_a_target_row_is_an_error(tmp_path):
    paths, config_path, panel, target = _setup(tmp_path)
    file = paths.target / "target_2002.parquet"
    rows = pd.read_parquet(file)
    rows.iloc[1:].to_parquet(file, index=False)  # one validation-block target row deleted, panel row kept
    with pytest.raises(forecast.DataValidationError, match="1 historical panel observations have no target row"):
        forecast.prepare_episode(paths, load_data_config(config_path), baseline_episodes(config_path)[0])
