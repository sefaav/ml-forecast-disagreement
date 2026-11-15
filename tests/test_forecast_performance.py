"""Baseline OOS forecasting diagnostics: monthly rank IC of the five saved signals (synthetic data only)."""

import json
import math

import numpy as np
import pandas as pd
import pytest
import yaml

from disagreement.data.config import DEFAULT_CONFIG, DataPaths
from disagreement.data.target import TARGET_COLUMNS, rank_target
from disagreement.data.validation import DataValidationError
from disagreement.diagnostics import forecast_performance as fp
from disagreement.models.selection import FAMILIES, validation_score
from disagreement.signals.ensemble import ensemble_signals
from disagreement.signals.family import unit_rank
from disagreement.walkforward.calendar import month_range

YEAR = 2004


def test_monthly_rank_ic_by_hand_with_ties_missing_outcomes_and_constant_signals():
    month = [1] * 4 + [2] * 4 + [3] * 4 + [4] * 4 + [5] * 4
    prediction = [1, 2, 3, 4,    4, 3, 2, 1,    1, 1, 2, 3,    1, 2, 3, 4,    7, 7, 7, 7]
    target = [-1, -1 / 3, 1 / 3, 1,  -1, -1 / 3, 1 / 3, 1,  -1, -1 / 3, 1 / 3, 1,  -1, np.nan, 0.9, 1,  -1, 0, 0, 1]
    out = fp.monthly_rank_ic(month, prediction, target)
    assert out["rank_ic"].iloc[:2].tolist() == pytest.approx([1.0, -1.0])
    assert out["rank_ic"].iloc[2] == pytest.approx(3 / math.sqrt(10))  # average ranks 1.5, 1.5, 3, 4
    full_ranks = np.corrcoef([1, 3, 4], [-1, 0.9, 1])[0, 1]  # the full-universe ranks of the scored stocks
    assert out["rank_ic"].iloc[3] == pytest.approx(np.corrcoef([1, 2, 3], [-1, 0.9, 1])[0, 1])  # reranked
    assert out["rank_ic"].iloc[3] != pytest.approx(full_ranks)
    assert out["rank_ic"].iloc[4] == 0.0 and out["constant_signal"].tolist() == [False] * 4 + [True]
    assert out["n_evaluated"].tolist() == [4, 4, 4, 3, 4]
    # the exact frozen selection score: its mean, bit for bit, on months of very different sizes
    rng = np.random.default_rng(3)
    m = np.repeat([200401, 200402, 200403], [5, 400, 37])
    s, y = rng.integers(0, 6, size=len(m)).astype(float), rng.uniform(-1, 1, size=len(m))
    y[rng.random(len(m)) < 0.2] = np.nan
    monthly = fp.monthly_rank_ic(m, s, y)
    assert np.mean(monthly["rank_ic"].to_numpy()) == validation_score(m, s, y)  # equal month weights
    assert fp.summarise(monthly["rank_ic"], monthly["constant_signal"])["mean_rank_ic"] == validation_score(m, s, y)
    for bad_target, bad_prediction, message in (([np.nan] * 4, [1, 2, 3, 4], "no observation with a finite"),
                                                ([0.5] * 4, [1, 2, 3, 4], "no cross-sectional variation"),
                                                ([-1, 0, 0.5, 1], [1, np.inf, 3, 4], "non-finite signal")):
        with pytest.raises(DataValidationError, match=message):
            fp.monthly_rank_ic([1] * 4, bad_prediction, bad_target)


def _source(tmp_path, seed=0):
    """Published synthetic ensemble signals and targets of OOS year 2004, with a config whose episodes are 2004."""
    rng = np.random.default_rng(seed)
    keys = pd.DataFrame([(p, m) for m in month_range(200401, 200412) for p in range(1, 40 + m % 9)],
                        columns=["permno", "yyyymm"])
    paths = DataPaths(tmp_path / "data")
    paths.ensemble_signals.mkdir(parents=True)
    paths.target.mkdir(parents=True)
    latent = rng.standard_normal(len(keys))
    index = pd.MultiIndex.from_frame(keys)
    ranks = {f: pd.Series(unit_rank(pd.Series(latent + rng.standard_normal(len(keys))), [keys["yyyymm"]]).to_numpy(),
                          index=index) for f in FAMILIES}
    ranks["elastic_net"].loc[ranks["elastic_net"].index.get_level_values("yyyymm") == 200403] = 0.5  # constant
    ensemble = ensemble_signals(ranks)  # the saved equal-weight mean, as the ensemble stage writes it
    ensemble.sample(frac=1.0, random_state=0).to_parquet(paths.ensemble_signals / f"{YEAR}.parquet", index=False)
    (paths.ensemble_signals / f"{YEAR}.json").write_text(json.dumps({"oos_year": YEAR}))
    target = keys.copy()
    target["ret_next"] = 0.05 * latent + 0.1 * rng.standard_normal(len(keys))
    target.loc[rng.random(len(keys)) < 0.1, "ret_next"] = np.nan  # missing outcomes stay in U_t
    target["target_rank"] = rank_target(target)
    target.loc[:, list(TARGET_COLUMNS)].to_parquet(paths.target / f"target_{YEAR}.parquet", index=False)
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["data"]["sample"] = {"start": 200001, "end": 200412}
    config["walk_forward"] = {"first_oos_year": YEAR, "validation_years": 2}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return paths, config_path, ensemble, target


def _snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_execute_publishes_monthly_and_aggregate_results_of_the_five_saved_signals(tmp_path, capsys):
    paths, config, ensemble, target = _source(tmp_path)
    inputs = _snapshot(paths.root)
    argv = ["--data-dir", str(paths.root), "--config", str(config)]
    assert fp.main([*argv, "--execute"]) == 0
    out = fp.default_output_dir(paths)
    assert sorted(p.name for p in out.iterdir()) == [fp.MONTHLY, fp.SUMMARY]
    assert not [p for p in paths.diagnostics.iterdir() if p.name.startswith(".")]  # no staging left
    monthly = pd.read_csv(out / fp.MONTHLY, float_precision="round_trip")
    summary = json.loads((out / fp.SUMMARY).read_text(encoding="utf-8"))
    assert list(monthly.columns) == ["yyyymm", "signal", "rank_ic", "constant_signal", "n_universe", "n_evaluated"]
    assert len(monthly) == 12 * 5 and set(monthly["signal"]) == set(fp.SIGNALS)  # no stock-level column
    merged = ensemble.merge(target, on=["permno", "yyyymm"], validate="one_to_one")
    for name, column in fp.SIGNALS.items():  # the saved columns, the ensemble as saved (not rebuilt)
        expected = validation_score(merged["yyyymm"].to_numpy(), merged[column].to_numpy(),
                                    merged["target_rank"].to_numpy())
        rows = monthly[monthly["signal"] == name]
        assert summary["signals"][name]["mean_rank_ic"] == expected == rows["rank_ic"].mean()
        assert summary["signals"][name]["median_rank_ic"] == rows["rank_ic"].median()
        assert summary["signals"][name]["sd_rank_ic"] == pytest.approx(rows["rank_ic"].std(ddof=1), rel=1e-15)
        assert summary["signals"][name]["months_evaluated"] == 12
    assert summary["signals"]["elastic_net"]["months_constant_signal"] == 1
    assert summary["signals"]["elastic_net"]["share_zero"] == pytest.approx(1 / 12)
    universe = target.groupby("yyyymm").size()
    finite = target.dropna().groupby("yyyymm").size()
    one = monthly[monthly["signal"] == "ensemble"].set_index("yyyymm")
    assert one["n_universe"].tolist() == universe.tolist() and one["n_evaluated"].tolist() == finite.tolist()
    assert summary["coverage"]["stock_months_universe"] == len(target)
    assert summary["coverage"]["stock_months_evaluated"] == int(target["target_rank"].notna().sum())
    assert summary["grid"] == {"first": 200401, "last": 200412, "n_months": 12, "n_years": 1}
    assert set(summary["inputs"]["files"]) == {f"{YEAR}/ensemble_signals", f"{YEAR}/ensemble_signals_marker",
                                               f"{YEAR}/target"}
    target_hash = fp._sha256(paths.target / f"target_{YEAR}.parquet")
    assert summary["inputs"]["files"][f"{YEAR}/target"]["sha256"] == target_hash
    assert summary["inputs"]["config"]["sha256"] == fp._sha256(config)
    assert "permno" not in (out / fp.MONTHLY).read_text() and "Published 12 months" in capsys.readouterr().out
    assert {k: v for k, v in _snapshot(paths.root).items() if not k.startswith("diagnostics/")} == inputs
    published = _snapshot(out)
    assert fp.main([*argv, "--execute"]) == 2  # an existing result set is never replaced
    assert "already exists" in capsys.readouterr().err and _snapshot(out) == published


def test_inconsistent_inputs_are_rejected_without_any_result(tmp_path, capsys):
    paths, config, ensemble, target = _source(tmp_path)
    argv = ["--data-dir", str(paths.root), "--config", str(config), "--execute"]
    target_file = paths.target / f"target_{YEAR}.parquet"
    ensemble_file = paths.ensemble_signals / f"{YEAR}.parquet"
    cases = {
        "repeat a permno/yyyymm key": (target_file, pd.concat([target, target.iloc[:1]], ignore_index=True)),
        "not the 12 OOS months": (target_file, target[target["yyyymm"] != 200406]),
        "1 missing, 0 extra": (target_file, target.drop(index=target["ret_next"].isna().idxmax())),  # a missing outcome
        "lie outside [-1, 1]": (target_file, target.assign(target_rank=target["target_rank"] * 1.5)),
        # finite, in [-1, 1], same keys and order, yet not the frozen transformation of ret_next
        "differs from the frozen normalised rank": (target_file, target.assign(
            target_rank=target["target_rank"].mask(target["target_rank"] == 1.0, 0.999999))),
        "ranks and signal must be finite and in": (ensemble_file, ensemble.assign(mlp_rank=ensemble["mlp_rank"] + 2)),
    }
    for message, (path, frame) in cases.items():
        original = path.read_bytes()
        frame.to_parquet(path, index=False)
        assert fp.main(argv) == 2
        assert message in capsys.readouterr().err
        assert not paths.diagnostics.exists()  # nothing published, nothing staged
        path.write_bytes(original)
    assert fp.main(argv) == 0  # the restored inputs pass


def test_an_input_changed_after_its_year_was_loaded_blocks_publication(tmp_path, monkeypatch):
    paths, config, _, _ = _source(tmp_path)
    marker = paths.ensemble_signals / f"{YEAR}.json"
    real_evaluate = fp.evaluate

    def change_then_evaluate(frame, grid):  # after every per-year check, before the final one
        marker.write_text(marker.read_text(encoding="utf-8") + " ", encoding="utf-8")
        return real_evaluate(frame, grid)

    monkeypatch.setattr(fp, "evaluate", change_then_evaluate)
    changed = rf"changed since they were read: \['{YEAR}/ensemble_signals_marker'\]"
    with pytest.raises(DataValidationError, match=changed):
        fp.run(paths, config)
    assert not paths.diagnostics.exists()  # neither a result set nor a staging directory


def test_a_failure_while_publishing_leaves_no_completed_result(tmp_path, monkeypatch):
    paths, config, _, _ = _source(tmp_path)

    def failing_dumps(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(fp.json, "dumps", failing_dumps)  # the completion marker cannot be written
    with pytest.raises(OSError, match="disk full"):
        fp.run(paths, config)
    assert not fp.default_output_dir(paths).exists()
    assert not list(paths.diagnostics.iterdir())  # the staging directory is removed


def test_without_execute_no_outcome_is_read_and_nothing_is_written(tmp_path, monkeypatch, capsys):
    paths, config, _, _ = _source(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("no saved outcome may be read without --execute")

    monkeypatch.setattr(fp, "load_target", forbidden)
    monkeypatch.setattr(fp, "load_ensemble", forbidden)
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    assert fp.main(["--data-dir", str(paths.root), "--config", str(config)]) == 0
    out = capsys.readouterr().out
    assert "Dry run: 12 months" in out and "0 input files missing" in out
    assert not paths.diagnostics.exists()
