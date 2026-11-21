"""Synthetic tests of the remaining pre-specified H1 robustness analyses (disagreement.inference.robustness)."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest
import yaml

from disagreement.controls import h2
from disagreement.data.config import DEFAULT_CONFIG, DataPaths
from disagreement.data.target import TARGET_COLUMNS, rank_target
from disagreement.data.validation import KEY, DataValidationError
from disagreement.inference import output, robustness
from disagreement.inference.fama_macbeth import fama_macbeth
from disagreement.inference.inputs import CD
from disagreement.inference.monthly import BinBasis, Coefficients, MonthlyEstimate, estimate_h1_month
from disagreement.inference.robustness import (
    DESCRIPTIONS,
    FAMILY_RANKS,
    PRIMARY,
    RAW,
    REESTIMATED,
    W_RAW,
    dispersion,
    mad_measure,
    range_measure,
    run_robustness,
    series_variants,
    variant_regressors,
    verify_robustness,
)
from disagreement.inference.series import MonthlySeries
from disagreement.models.selection import FAMILIES
from disagreement.signals.between import between_family_disagreement
from disagreement.signals.conditional import SIGNAL_BINS, conditional_disagreement, signal_bins
from disagreement.signals.ensemble import ensemble_signals
from disagreement.signals.family import month_size, rank_mean, rank_numerators, unit_rank
from disagreement.walkforward.calendar import month_range

YEAR = 2004


def _derive(frame):
    """S, raw SD, the 20 bins, CD^B and S^c of a frame of family ranks, as the signal pipeline builds them."""
    size = month_size(frame["yyyymm"])
    ranks = frame[FAMILY_RANKS].to_numpy(dtype="float64")
    frame["ensemble_signal"] = rank_mean(ranks, size, "test")
    frame[RAW] = dispersion(*rank_numerators(ranks, size, "test"))
    frame["signal_bin"] = signal_bins(frame["ensemble_signal"], frame["yyyymm"], SIGNAL_BINS)
    frame[CD] = unit_rank(frame[RAW], [frame["yyyymm"], frame["signal_bin"]])
    frame["centered_signal"] = 2.0 * (frame["ensemble_signal"] - 0.5)
    return frame


def _year(rng, months, n=120):
    frame = pd.DataFrame([(p, m) for m in months for p in range(1, n + 1)], columns=list(KEY))
    for column in FAMILY_RANKS:  # average-tie unit ranks of the month's cross-section, with ties
        frame[column] = unit_rank(pd.Series(rng.integers(0, n // 3, len(frame))), [frame["yyyymm"]])
    frame["target_rank"] = np.where(rng.random(len(frame)) < 0.1, np.nan, rng.uniform(-1, 1, len(frame)))
    frame[W_RAW] = rng.uniform(0, 1, len(frame))
    return _derive(frame)


def _stub(calls):
    """A deterministic stand-in for estimate_h1_month that records its inputs."""
    def estimate(outcome, sc, bins, cd, n_bins):
        calls.append((outcome, bins, cd, n_bins))
        zeros = np.zeros(n_bins)
        coefficients = Coefficients(zeros, zeros, zeros, 0.0, float(cd.mean() + sc.std()), {})
        return MonthlyEstimate("C", True, (), int(np.isfinite(outcome).sum()), zeros.astype(int), zeros.astype(int),
                               1.0, 1.0, BinBasis(zeros, np.ones(n_bins), zeros), coefficients)
    return estimate


def test_measures_match_hand_calculations_and_keep_exact_ties():
    ranks = np.array([[0.0, 0.25, 0.5, 1.0], [0.5, 0.5, 0.5, 0.5], [1.0, 0.5, 0.0, 0.25]])  # n = 5: k / 8
    k, m = rank_numerators(ranks, np.full(3, 5), "test")
    # deviations from S = 0.4375: -0.4375, -0.1875, 0.0625, 0.5625
    assert dispersion(k, m)[0] == pytest.approx(np.sqrt(0.546875 / 4), abs=1e-16)
    assert dispersion(k, m)[0] == pytest.approx(np.std(ranks[0]), abs=1e-16)  # ddof = 0
    np.testing.assert_array_equal(range_measure(k, m), [1.0, 0.0, 1.0])
    np.testing.assert_array_equal(mad_measure(k, m), [0.3125, 0.0, 0.3125])  # 1.25 / 4, not a median deviation
    assert dispersion(k, m)[2] == dispersion(k, m)[0]  # a permutation of the ranks: bit-identical
    three = k[:, :3]  # ranks 0, 0.25, 0.5: S = 0.25
    assert dispersion(three, m)[0] == pytest.approx(np.sqrt(0.125 / 3), abs=1e-16)
    assert mad_measure(three, m)[0] == 0.5 / 3 and range_measure(three, m)[0] == 0.5


def test_variant_regressors_recompute_bins_signals_and_conditional_ranks():
    frame = _year(np.random.default_rng(1), [202001, 202002])
    i, j = 3, 40  # same month; j gets a permutation of i's family ranks, so the same S, bin and measures
    frame.loc[j, FAMILY_RANKS] = frame.loc[i, FAMILY_RANKS].to_numpy()[[2, 0, 3, 1]]
    frame = _derive(frame)
    reg = variant_regressors(frame)
    assert set(reg) == {PRIMARY, *REESTIMATED}
    sc, bins = frame["centered_signal"].to_numpy(), frame["signal_bin"].to_numpy()
    for name in (PRIMARY, "raw_disagreement", "range_disagreement", "mad_disagreement"):
        np.testing.assert_array_equal(reg[name][0], sc)
        np.testing.assert_array_equal(reg[name][1], bins)
        assert reg[name][3] == SIGNAL_BINS
    np.testing.assert_array_equal(reg[PRIMARY][2], frame[CD])
    np.testing.assert_array_equal(reg["raw_disagreement"][2], frame[RAW])  # not ranked or rescaled
    # ten bins: 1 + floor(10 (r - 1) / n) of the saved S, CD^B re-ranked within the new cells
    r = frame.groupby("yyyymm")["ensemble_signal"].rank(method="average")
    n = frame.groupby("yyyymm")["ensemble_signal"].transform("size")
    ten = (1 + np.floor(10 * (r - 1).to_numpy() / n.to_numpy())).astype(int)
    np.testing.assert_array_equal(reg["ten_bins"][1], ten)
    assert reg["ten_bins"][3] == 10 and set(ten) == set(range(1, 11))
    for (month, b), cell in frame.assign(b=ten).groupby(["yyyymm", "b"]):
        expected = (cell[RAW].rank(method="average") - 1) / (len(cell) - 1)
        np.testing.assert_allclose(reg["ten_bins"][2][cell.index], expected, rtol=0, atol=1e-15)
    for name in ("range_disagreement", "mad_disagreement", "ten_bins", "raw_disagreement"):
        assert reg[name][2][i] == reg[name][2][j]  # exact ties stay ties
        assert ((reg[name][2] >= 0) & (reg[name][2] <= 1)).all()
    # leave-one-family-out: three-family S, SD (ddof = 0), 20 bins of that S, CD^B within them
    size = month_size(frame["yyyymm"])
    for family in FAMILIES:
        rest = [c for c in FAMILY_RANKS if c != f"{family}_rank"]
        signal = rank_mean(frame[rest].to_numpy(), size, "test")
        sc3, bins3, cd3, nb = reg[f"leave_out_{family}"]
        np.testing.assert_allclose(signal, frame[rest].mean(axis=1), rtol=0, atol=1e-15)
        np.testing.assert_array_equal(sc3, 2 * (signal - 0.5))
        np.testing.assert_array_equal(bins3, signal_bins(pd.Series(signal), frame["yyyymm"], 20))
        sd = dispersion(*rank_numerators(frame[rest].to_numpy(), size, "test"))
        np.testing.assert_allclose(sd, frame[rest].std(axis=1, ddof=0), rtol=0, atol=1e-15)
        np.testing.assert_array_equal(cd3, unit_rank(pd.Series(sd), [frame["yyyymm"], pd.Series(bins3)]))
        assert nb == 20
    # the omitted family's ranks do not enter its variant; they enter every other one
    changed = frame.copy()
    changed["elastic_net_rank"] = changed["elastic_net_rank"].sample(frac=1.0, random_state=0).to_numpy()
    after = variant_regressors(_derive(changed))
    for a, b in zip(reg["leave_out_elastic_net"], after["leave_out_elastic_net"]):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(reg["leave_out_mlp"][0], after["leave_out_mlp"][0])


def test_every_analysis_uses_all_members_of_u_t_and_the_saved_target():
    rng = np.random.default_rng(2)
    months = [202011, 202012]
    frame = _year(rng, months)
    primary = variant_regressors(frame)[PRIMARY]
    values = [primary[2][idx].mean() + primary[0][idx].std() for idx in frame.groupby("yyyymm").indices.values()]
    saved = MonthlySeries(PRIMARY, np.array(months), np.array(values), np.ones(2, dtype=bool))
    calls = []
    run = run_robustness(months, [(2020, frame)], saved, _stub(calls))
    assert len(calls) == 2 * (1 + len(REESTIMATED))
    targets = [rows.to_numpy() for _, rows in frame.groupby("yyyymm")["target_rank"]]
    for outcome, bins, cd, n_bins in calls:  # the whole U_t, missing outcomes included, the saved target_rank
        assert any(np.array_equal(outcome, t, equal_nan=True) for t in targets) and np.isnan(outcome).any()
        assert bins.max() <= n_bins and len(cd) == len(outcome)
    assert sorted(c[3] for c in calls).count(10) == 2  # ten bins only for the ten-bin analysis
    assert set(run.inference) == set(DESCRIPTIONS)
    assert {name: len(e) for name, e in run.estimates.items()} == {name: 2 for name in REESTIMATED}
    with pytest.raises(DataValidationError, match="does not reproduce"):  # the reproduction check fails closed
        run_robustness(months, [(2020, frame)], MonthlySeries(PRIMARY, saved.grid, saved.values + 1e-9,
                                                               saved.estimable), _stub([]))
    with pytest.raises(DataValidationError, match="no input"):
        run_robustness(month_range(202010, 202012), [(2020, frame)],
                       MonthlySeries(PRIMARY, np.array(month_range(202010, 202012)), np.zeros(3),
                                     np.ones(3, dtype=bool)), _stub([]))


def test_2021_exclusion_and_subperiods_keep_the_frozen_calendar_convention():
    grid = np.array(month_range(198501, 202412))
    rng = np.random.default_rng(3)
    mask = np.ones(grid.size, dtype=bool)
    mask[grid == 200007] = False  # a non-estimable month keeps its position
    saved = MonthlySeries(PRIMARY, grid, np.where(mask, rng.normal(0.02, 0.1, grid.size), np.nan), mask)
    out = series_variants(saved)
    excluded = out["excluding_2021"]
    np.testing.assert_array_equal(excluded.grid, grid)
    assert sorted(grid[~excluded.estimable].tolist()) == [200007, *month_range(202101, 202112)]
    assert excluded.n_months == 480 - 13
    hac = fama_macbeth(excluded.values, excluded.estimable)
    assert hac.n_grid == 480 and hac.n_months == 467
    compressed = fama_macbeth(saved.values[mask & (grid // 100 != 2021)], np.ones(467, dtype=bool))
    assert hac.mean == compressed.mean and hac.se != compressed.se  # lags count calendar months, not rows
    first, second = out["subperiod_1985_2004"], out["subperiod_2005_2024"]
    assert first.grid.tolist() == month_range(198501, 200412) and second.grid.tolist() == month_range(200501, 202412)
    assert first.grid.size == second.grid.size == 240 and first.n_months == 239 and second.n_months == 240
    np.testing.assert_array_equal(second.values, saved.values[240:])
    np.testing.assert_array_equal(out[PRIMARY].values, saved.values)


# ----------------------------------------------------------------------------------------------------------------
# Execution and publication on a synthetic published year


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    """A synthetic OOS year 2004 built with the pipeline's functions, with the core inference published."""
    root = tmp_path_factory.mktemp("source")
    rng = np.random.default_rng(20261121)
    keys = pd.DataFrame([(p, m) for m in month_range(200401, 200412) for p in range(1, 1100 + m % 7)],
                        columns=list(KEY))
    index = pd.MultiIndex.from_frame(keys)
    ranks = {f: pd.Series(unit_rank(pd.Series(rng.integers(0, 600, len(keys))), [keys["yyyymm"]]).to_numpy(),
                          index=index) for f in FAMILIES}
    ensemble = ensemble_signals(ranks)
    signals = between_family_disagreement(ensemble).assign(
        within_family_uncertainty_raw=rng.uniform(0.0, 1.0, len(keys)))
    cond = conditional_disagreement(signals, SIGNAL_BINS)
    paths = DataPaths(root / "data")
    for directory in (paths.universe, paths.panel, paths.conditional_disagreement, paths.ensemble_signals,
                      paths.diagnostics, paths.target):
        directory.mkdir(parents=True)
    (paths.diagnostics / "summary.json").write_text(json.dumps({"validation": "PASS"}))  # a successful build
    universe = keys.assign(exchange="NYSE", mktcap=rng.integers(1, 500, size=len(keys)) * 1000.0)
    panel = universe.assign(RealizedVol=-rng.uniform(0.01, 0.1, size=len(keys)),
                            Illiquidity=rng.uniform(0.0, 1.0, size=len(keys)))
    universe.to_parquet(paths.universe / f"universe_{YEAR}.parquet", index=False)
    panel.to_parquet(paths.panel / f"panel_{YEAR}.parquet", index=False)
    for directory, frame in ((paths.conditional_disagreement, cond), (paths.ensemble_signals, ensemble)):
        frame.to_parquet(directory / f"{YEAR}.parquet", index=False)
        (directory / f"{YEAR}.json").write_text(json.dumps({"oos_year": YEAR}))
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["data"]["sample"] = {"start": 200001, "end": 200412}
    config["walk_forward"] = {"first_oos_year": YEAR, "validation_years": 2}
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert h2.main(["--data-dir", str(paths.root), "--config", str(config_path), "--year", str(YEAR)]) == 0
    sc = 2.0 * (cond.set_index(list(KEY)).loc[index, "ensemble_signal"].to_numpy() - 0.5)
    target = keys.assign(ret_next=0.02 * sc + 0.1 * rng.standard_normal(len(keys)))
    target.loc[rng.random(len(keys)) < 0.1, "ret_next"] = np.nan
    target["target_rank"] = rank_target(target)
    target.loc[:, list(TARGET_COLUMNS)].to_parquet(paths.target / f"target_{YEAR}.parquet", index=False)
    output.execute_inference(paths, config_path)
    return root


def _copy(source, tmp_path):
    shutil.copytree(source, tmp_path / "copy")
    return DataPaths(tmp_path / "copy" / "data"), tmp_path / "copy" / "config.yaml"


def _snapshot(directory):
    return {p.name: output.sha256(p) for p in sorted(directory.iterdir())}


def test_execution_publishes_a_verified_aggregate_result_set_and_never_replaces_it(source, tmp_path, capsys):
    paths, config = _copy(source, tmp_path)
    core = _snapshot(paths.diagnostics / "inference")
    argv = ["--data-dir", str(paths.root), "--config", str(config)]
    assert robustness.main([*argv, "--execute"]) == 0
    assert "Published 11 H1 robustness analyses over 12 months" in capsys.readouterr().out
    destination = paths.diagnostics / "h1_robustness"
    summary = verify_robustness(destination, paths, config)
    assert {p.name for p in destination.iterdir()} == {"inference.csv", "monthly.csv", "manifest.json",
                                                        "summary.json"}
    assert _snapshot(paths.diagnostics / "inference") == core  # the core inference is untouched
    table = pd.read_csv(destination / "inference.csv", float_precision="round_trip")
    assert table["analysis"].tolist() == list(DESCRIPTIONS) and (table["lag"] == 6).all()
    report = json.loads((paths.diagnostics / "inference" / "report.json").read_text(encoding="utf-8"))
    row = table.set_index("analysis")
    assert row.loc[PRIMARY, "mean"] == report["decisions"]["H1"]["basis"]["C.interaction"]["mean"]
    assert (row.loc[list(REESTIMATED), "n_months"] == 12).all()
    assert row.loc["subperiod_1985_2004", "n_grid"] == 12 and row.loc["subperiod_2005_2024", "status"] == "failed"
    assert row.loc["excluding_2021", "mean"] == row.loc[PRIMARY, "mean"]  # no 2021 month on this calendar
    monthly = pd.read_csv(destination / "monthly.csv")
    assert "permno" not in monthly.columns and len(monthly) == 12 * len(REESTIMATED)
    assert set(monthly.loc[monthly["analysis"] == "ten_bins", "n_bins"]) == {10} and monthly["estimable"].all()
    assert summary["non_estimable_months"] == {name: 0 for name in REESTIMATED}
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert {"inference/series.csv", f"{YEAR}/ensemble_signals", f"{YEAR}/target"} <= set(manifest["inputs"]["files"])
    # never replaced; the core result set and a destination inside it are refused
    assert robustness.main([*argv, "--execute"]) == 2 and "already exists" in capsys.readouterr().err
    for target in (paths.diagnostics / "inference", paths.diagnostics / "inference" / "inner"):
        assert robustness.main([*argv, "--execute", "--output-dir", str(target)]) == 2
    assert _snapshot(paths.diagnostics / "inference") == core
    (destination / "monthly.csv").write_text("analysis\n", encoding="utf-8")
    with pytest.raises(output.PublicationError, match="differ from the manifest"):
        verify_robustness(destination)


def test_a_non_estimable_month_keeps_its_mask_reason_and_calendar_position(source, tmp_path, monkeypatch):
    paths, config = _copy(source, tmp_path)
    real_run, ten_bin_calls = robustness.run_robustness, []

    def with_gap(grid, years, saved):  # June 2004 of the ten-bin analysis: an empty H1 sample, so not estimable
        def estimate(outcome, sc, bins, cd, n_bins):
            if n_bins == 10:
                ten_bin_calls.append(1)
                if len(ten_bin_calls) == 6:
                    outcome = np.full_like(outcome, np.nan)
            return estimate_h1_month(outcome, sc, bins, cd, n_bins)
        return real_run(grid, years, saved, estimate)

    monkeypatch.setattr(robustness, "run_robustness", with_gap)
    summary = robustness.execute(paths, config)
    destination = paths.diagnostics / "h1_robustness"
    verify_robustness(destination, paths, config)
    assert summary["non_estimable_months"]["ten_bins"] == 1
    monthly = pd.read_csv(destination / "monthly.csv", float_precision="round_trip")
    ten = monthly[monthly["analysis"] == "ten_bins"].reset_index(drop=True)
    assert ten["yyyymm"].tolist() == month_range(200401, 200412)  # the month keeps its row and position
    gap = ten.loc[5].to_dict()
    assert gap["yyyymm"] == 200406 and not gap["estimable"] and np.isnan(gap["interaction"]) and gap["n_obs"] == 0
    assert "H1 sample is empty" in json.loads(gap["reasons"])[0]
    row = pd.read_csv(destination / "inference.csv", float_precision="round_trip").set_index("analysis")
    expected = fama_macbeth(ten["interaction"].to_numpy(), ten["estimable"].to_numpy())  # zero-filled gap, N = 12
    assert (row.loc["ten_bins", "n_months"], row.loc["ten_bins", "n_grid"]) == (11, 12)
    assert (row.loc["ten_bins", "mean"], row.loc["ten_bins", "se"]) == (expected.mean, expected.se)


def test_verification_rejects_wrong_metadata_and_resealed_invalid_tables(source, tmp_path):
    paths, config = _copy(source, tmp_path)
    robustness.execute(paths, config)
    destination = paths.diagnostics / "h1_robustness"
    original = {p.name: p.read_bytes() for p in destination.iterdir()}

    def reseal(name, frame):  # a modified table with a consistently updated manifest and marker
        frame.to_csv(destination / name, index=False, float_format="%.17g", lineterminator="\n")
        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        manifest["outputs"][name] = output.sha256(destination / name)
        (destination / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
        summary["manifest_sha256"] = output.sha256(destination / "manifest.json")
        (destination / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    def restore():
        for name, content in original.items():
            (destination / name).write_bytes(content)

    for key, wrong in (("specification", "B"), ("outcome", "ret_next"), ("lag", 0), ("analyses", {PRIMARY: "x"})):
        summary = json.loads(original["summary.json"])
        summary[key] = wrong
        (destination / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        with pytest.raises(output.PublicationError, match="frozen scientific metadata"):
            verify_robustness(destination)
        restore()
    table = pd.read_csv(destination / "inference.csv", float_precision="round_trip")
    monthly = pd.read_csv(destination / "monthly.csv", float_precision="round_trip")
    for name, frame in (("inference.csv", table.iloc[:-1]), ("inference.csv", table.assign(lag=0)),
                        ("monthly.csv", monthly[monthly["yyyymm"] != 200406])):
        reseal(name, frame)
        with pytest.raises(output.PublicationError, match="inconsistent result structure"):
            verify_robustness(destination)
        restore()
    verify_robustness(destination, paths, config)


def test_failures_publish_nothing(source, tmp_path, monkeypatch):
    paths, config = _copy(source, tmp_path)
    destination = paths.diagnostics / "h1_robustness"
    load = robustness.load_year

    def changing(p, year):  # an input changes while its year is read
        frame = load(p, year)
        with open(p.target / f"target_{year}.parquet", "ab") as fh:
            fh.write(b"\0")
        return frame

    monkeypatch.setattr(robustness, "load_year", changing)
    with pytest.raises(DataValidationError, match="changed while they were read"):
        robustness.execute(paths, config)
    assert not destination.exists()
    monkeypatch.undo()
    paths, config = _copy(source, tmp_path / "second")
    real_run = robustness.run_robustness

    def then_change(*args, **kwargs):  # an input changes after the run, before publication
        result = real_run(*args, **kwargs)
        (paths.ensemble_signals / f"{YEAR}.json").write_text("{}")
        return result

    monkeypatch.setattr(robustness, "run_robustness", then_change)
    with pytest.raises(DataValidationError, match="changed since they were read"):
        robustness.execute(paths, config)
    assert not (paths.diagnostics / "h1_robustness").exists()
    assert not [p for p in paths.diagnostics.iterdir() if "staging" in p.name]
    monkeypatch.undo()
    paths, config = _copy(source, tmp_path / "third")

    def full_disk(*args, **kwargs):  # a write fails while the result set is staged
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_csv", full_disk)
    with pytest.raises(OSError):
        robustness.execute(paths, config)
    assert not (paths.diagnostics / "h1_robustness").exists()
    assert not [p for p in paths.diagnostics.iterdir() if "staging" in p.name]


def test_dry_run_reads_and_writes_nothing(source, tmp_path, monkeypatch, capsys):
    paths, config = _copy(source, tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("nothing may be loaded or estimated without --execute")

    for module, name in ((pd, "read_parquet"), (pd, "read_csv"), (robustness, "execute")):
        monkeypatch.setattr(module, name, forbidden)
    assert robustness.main(["--data-dir", str(paths.root), "--config", str(config)]) == 0
    assert "Dry run: 11 H1 robustness analyses over 12 months" in capsys.readouterr().out
    assert not (paths.diagnostics / "h1_robustness").exists()
