"""H2 control characteristics: monthly centred ranks of size, realized volatility and illiquidity (synthetic data)."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from disagreement.controls import h2
from disagreement.controls.h2 import (
    CONTROLS, DEFINITION, DEFINITION_SHA256, INTERACTIONS, OUTPUT_COLUMNS, RANKS, centered_rank, h2_controls,
    load_h2_controls, main,
)
from disagreement.data.config import DEFAULT_CONFIG, DataPaths
from disagreement.data.pipeline import remove_derived_outputs
from disagreement.data.validation import DataValidationError
from disagreement.signals.conditional import SIGNAL_BINS, conditional_disagreement
from disagreement.signals.family import remove_conditional_disagreement, unit_rank
from disagreement.walkforward.calendar import month_range

INPUT = ["permno", "yyyymm", "mktcap", "RealizedVol", "Illiquidity", "ensemble_signal"]
NAN, INF = np.nan, np.inf


def test_centered_rank_formula_ties_degenerate_groups_and_missing_values():
    rows = [  # month, value, expected centred rank
        (1, 3.0, 0.5), (1, 1.0, -1.0), (1, 2.0, -0.25), (1, 2.0, -0.25), (1, 5.0, 1.0),  # ranks 4 1 2.5 2.5 5 of 5
        (2, 7.0, 0.0), (2, 7.0, 0.0), (2, 7.0, 0.0),  # a constant cross-section
        (3, 4.0, 0.0),  # a single observation
        (4, NAN, NAN), (4, INF, NAN), (4, -INF, NAN), (4, 1.0, -1.0), (4, 2.0, 1.0),  # ranked among the two finite
        (5, NAN, NAN), (5, 10.0, 0.0),  # one finite value next to a missing one
        (6, NAN, NAN), (6, NAN, NAN),  # nothing to rank
    ]
    frame = pd.DataFrame(rows, columns=["yyyymm", "x", "expected"])
    out = centered_rank(frame["x"], frame["yyyymm"])
    np.testing.assert_array_equal(out.to_numpy(), frame["expected"].to_numpy())  # exact, NaN where missing
    assert out.index.equals(frame.index)
    # The value 2.0 ranks -0.25 in month 1 and +1.0 in month 4: months are ranked separately.
    finite = frame["x"].replace([INF, -INF], NAN)
    expected = 2.0 * unit_rank(finite, [frame["yyyymm"]]) - 1.0  # the family-signal convention, centred ...
    np.testing.assert_array_equal(out[finite.notna()].to_numpy(), expected[finite.notna()].to_numpy())
    assert (expected[finite.isna()] == 0.0).all()  # ... which would turn the missing values into 0 if reused as is

    shuffled = frame.sample(frac=1.0, random_state=1)
    np.testing.assert_array_equal(
        centered_rank(shuffled["x"], shuffled["yyyymm"]).sort_index().to_numpy(), frame["expected"].to_numpy()
    )


def test_h2_controls_formulas_signs_interactions_and_ranking_before_the_complete_case_restriction():
    e = np.exp
    rows = [  # permno, yyyymm, mktcap, RealizedVol (signed), Illiquidity (signed), S
        (1, 200401, e(1.0), -0.3, 0.5, 0.0),
        (2, 200401, e(2.0), -0.1, 0.2, 0.5),
        (3, 200401, e(3.0), -0.2, NAN, 1.0),  # no illiquidity: not a complete case, still ranked for size and vol
        (1, 200402, 5.0, NAN, INF, 0.25),  # infinite illiquidity: missing
        (2, 200402, 5.0, 0.2, 0.3, 0.75),
    ]
    frame = pd.DataFrame(rows, columns=INPUT)
    out = h2_controls(frame.sample(frac=1.0, random_state=2))  # row order must not matter
    pd.testing.assert_frame_equal(out, h2_controls(frame))
    assert list(out.columns) == list(OUTPUT_COLUMNS)
    keys = [[1, 200401], [2, 200401], [3, 200401], [1, 200402], [2, 200402]]
    assert out[["permno", "yyyymm"]].values.tolist() == keys
    assert out["complete_case"].dtype == bool

    jan = out[out["yyyymm"] == 200401]
    np.testing.assert_array_equal(jan["size"], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(jan["realized_volatility"], [0.3, 0.1, 0.2])  # sign reversed
    np.testing.assert_array_equal(jan["illiquidity"], [0.5, 0.2, NAN])  # sign kept
    np.testing.assert_array_equal(jan["size_rank"], [-1.0, 0.0, 1.0])  # among all three stocks of U_t ...
    np.testing.assert_array_equal(jan["realized_volatility_rank"], [1.0, -1.0, 0.0])
    np.testing.assert_array_equal(jan["illiquidity_rank"], [1.0, -1.0, NAN])
    assert jan["complete_case"].tolist() == [True, True, False]
    complete = jan[jan["complete_case"]]
    assert complete["size_rank"].tolist() == [-1.0, 0.0]  # ... not [-1, 1], the ranks of the two complete cases alone
    np.testing.assert_array_equal(jan["centered_signal"], [-1.0, 0.0, 1.0])
    np.testing.assert_array_equal(jan["signal_x_size_rank"], [1.0, 0.0, 1.0])
    np.testing.assert_array_equal(jan["signal_x_realized_volatility_rank"], [-1.0, 0.0, 0.0])
    np.testing.assert_array_equal(jan["signal_x_illiquidity_rank"], [-1.0, 0.0, NAN])  # missing control: missing

    feb = out[out["yyyymm"] == 200402]
    np.testing.assert_array_equal(feb["size_rank"], [0.0, 0.0])  # constant market capitalisation
    np.testing.assert_array_equal(feb["realized_volatility_rank"], [NAN, 0.0])  # one finite value
    np.testing.assert_array_equal(feb["illiquidity"], [NAN, 0.3])
    np.testing.assert_array_equal(feb["illiquidity_rank"], [NAN, 0.0])
    assert feb["complete_case"].tolist() == [False, True]
    np.testing.assert_array_equal(feb["centered_signal"], [-0.5, 0.5])
    np.testing.assert_array_equal(feb["signal_x_realized_volatility_rank"], [NAN, 0.0])
    for interaction, rank in zip(INTERACTIONS, RANKS):
        np.testing.assert_array_equal(out[interaction], out["centered_signal"] * out[rank])


def _setup(tmp_path, seed=0):
    """Universe, panel and conditional disagreement of OOS year 2004, with a config whose episodes include it."""
    rng = np.random.default_rng(seed)
    keys = pd.DataFrame(
        [(p, m) for m in month_range(200401, 200412) for p in range(1, 41 + m % 7)], columns=["permno", "yyyymm"]
    )
    universe = keys.copy()
    universe["exchange"] = "NYSE"
    universe["mktcap"] = rng.integers(1, 30, size=len(keys)) * 1000.0  # ties
    panel = universe.copy()
    panel["AM"] = rng.standard_normal(len(keys))  # an unrelated predictor
    panel["RealizedVol"] = -rng.integers(1, 9, size=len(keys)) / 100  # signed: sign -1
    panel["Illiquidity"] = rng.integers(1, 9, size=len(keys)) / 10
    panel.loc[rng.random(len(keys)) < 0.15, "Illiquidity"] = NAN
    panel.loc[rng.random(len(keys)) < 0.02, "RealizedVol"] = NAN
    panel["Size"] = -np.log(panel["mktcap"])
    signals = keys.copy()
    signals["ensemble_signal"] = rng.integers(0, 17, size=len(keys)) / 16
    signals["between_family_disagreement_raw"] = rng.uniform(0.0, 0.5, size=len(keys))
    signals["within_family_uncertainty_raw"] = rng.uniform(0.0, 1.0, size=len(keys))
    conditional = conditional_disagreement(signals, SIGNAL_BINS)

    paths = DataPaths(tmp_path / "data")
    for directory in (paths.universe, paths.panel, paths.conditional_disagreement, paths.diagnostics):
        directory.mkdir(parents=True)
    (paths.diagnostics / "summary.json").write_text(json.dumps({"validation": "PASS"}))
    universe.sample(frac=1.0, random_state=1).to_parquet(paths.universe / "universe_2004.parquet", index=False)
    panel.sample(frac=1.0, random_state=2).to_parquet(paths.panel / "panel_2004.parquet", index=False)
    shuffled = conditional.sample(frac=1.0, random_state=3)
    shuffled.to_parquet(paths.conditional_disagreement / "2004.parquet", index=False)
    (paths.conditional_disagreement / "2004.json").write_text(json.dumps({"oos_year": 2004}))
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["data"]["sample"] = {"start": 200001, "end": 200512}
    config["walk_forward"] = {"first_oos_year": 2004, "validation_years": 2}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    argv = ["--data-dir", str(paths.root), "--config", str(config_path), "--year", "2004"]
    return paths, argv


def _centered(values, month):
    finite = values.where(np.isfinite(values))
    rank = finite.groupby(month).rank(method="average")
    n = finite.groupby(month).transform("count")
    return (2.0 * (rank - 1.0) / (n - 1.0) - 1.0).where(n > 1, 0.0).where(finite.notna())


def test_output_schema_provenance_and_reproducible_completion(tmp_path, monkeypatch):
    paths, argv = _setup(tmp_path)
    directory = paths.h2_controls
    replaced, replace = [], os.replace

    def recording_replace(src, dst):
        replaced.append(os.path.basename(dst))
        replace(src, dst)

    monkeypatch.setattr(h2.os, "replace", recording_replace)
    outputs = []
    for _ in range(2):
        assert main(argv) == 0
        outputs.append(((directory / "2004.parquet").read_bytes(), (directory / "2004.json").read_bytes()))
    assert outputs[0] == outputs[1]  # identical bytes
    assert replaced[-1] == "2004.json" and replaced.count("2004.json") == 2  # the marker, written last
    assert sorted(p.name for p in directory.iterdir()) == ["2004.json", "2004.parquet"]  # no .tmp
    out, metadata = pd.read_parquet(directory / "2004.parquet"), json.loads((directory / "2004.json").read_text())

    assert list(out.columns) == list(OUTPUT_COLUMNS)
    assert not out.duplicated(["permno", "yyyymm"]).any()
    assert out.equals(out.sort_values(["yyyymm", "permno"], ignore_index=True))
    assert sorted(out["yyyymm"].unique().tolist()) == month_range(200401, 200412)
    universe = pd.read_parquet(paths.universe / "universe_2004.parquet")
    universe = universe.sort_values(["yyyymm", "permno"], ignore_index=True)
    panel = pd.read_parquet(paths.panel / "panel_2004.parquet").sort_values(["yyyymm", "permno"], ignore_index=True)
    conditional = pd.read_parquet(paths.conditional_disagreement / "2004.parquet")
    conditional = conditional.sort_values(["yyyymm", "permno"], ignore_index=True)
    pd.testing.assert_frame_equal(out[["permno", "yyyymm"]], universe[["permno", "yyyymm"]])  # every U_t stock-month

    # Recomputed independently from the inputs.
    economic = {
        "size": np.log(universe["mktcap"]),
        "realized_volatility": -panel["RealizedVol"],
        "illiquidity": panel["Illiquidity"],
    }
    for control, rank in zip(CONTROLS, RANKS):
        np.testing.assert_array_equal(out[control].to_numpy(), economic[control].to_numpy())
        np.testing.assert_array_equal(out[rank].to_numpy(), _centered(economic[control], out["yyyymm"]).to_numpy())
        values = out[rank].to_numpy()
        assert np.isnan(values).sum() == economic[control].isna().sum()  # no missing value became a rank
        assert values[np.isfinite(values)].min() >= -1.0 and values[np.isfinite(values)].max() <= 1.0
    assert out["illiquidity_rank"].isna().any() and out["realized_volatility_rank"].isna().any()
    np.testing.assert_array_equal(
        out["complete_case"].to_numpy(), out[list(RANKS)].notna().all(axis=1).to_numpy()
    )
    assert 0 < out["complete_case"].sum() < len(out)
    centered_signal = 2.0 * (conditional["ensemble_signal"].to_numpy() - 0.5)
    np.testing.assert_array_equal(out["centered_signal"].to_numpy(), centered_signal)
    for interaction, rank in zip(INTERACTIONS, RANKS):
        np.testing.assert_array_equal(out[interaction].to_numpy(), centered_signal * out[rank].to_numpy())

    sha = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in h2.input_files(paths, 2004).items()
    }
    by_month = out.groupby("yyyymm")["complete_case"].mean()
    assert metadata == {
        "oos_year": 2004,
        "n_stock_months": len(universe),
        "months": month_range(200401, 200412),
        "n_complete_case": int(out["complete_case"].sum()),
        "complete_case_share": float(out["complete_case"].mean()),
        "min_monthly_complete_case_share": float(by_month.min()),
        "coverage": {control: float(out[rank].notna().mean()) for control, rank in zip(CONTROLS, RANKS)},
        "definition": DEFINITION,
        "definition_sha256": DEFINITION_SHA256,
        "inputs_sha256": sha,
        "output_sha256": hashlib.sha256((directory / "2004.parquet").read_bytes()).hexdigest(),
    }
    assert DEFINITION_SHA256 == hashlib.sha256(json.dumps(DEFINITION, sort_keys=True).encode()).hexdigest()
    pd.testing.assert_frame_equal(load_h2_controls(paths, 2004), out)

    # The same inputs in another row order give the same output.
    for name, path in h2.input_files(paths, 2004).items():
        pd.read_parquet(path).sample(frac=1.0, random_state=7).to_parquet(path, index=False)
    assert main(argv) == 0
    pd.testing.assert_frame_equal(pd.read_parquet(directory / "2004.parquet"), out)


def test_invalid_inputs_are_rejected_and_leave_no_completion_marker(tmp_path, capsys):
    paths, argv = _setup(tmp_path)
    marker = paths.h2_controls / "2004.json"
    universe_path, panel_path = paths.universe / "universe_2004.parquet", paths.panel / "panel_2004.parquet"
    conditional_path = paths.conditional_disagreement / "2004.parquet"
    universe, panel, conditional = (pd.read_parquet(p) for p in (universe_path, panel_path, conditional_path))

    def rejected(message):
        assert main(argv) == 2
        assert message in capsys.readouterr().err
        assert not marker.exists()

    assert main(argv) == 0 and marker.exists()
    panel.iloc[1:].to_parquet(panel_path, index=False)  # a universe member without a panel row
    rejected("2004 panel covers different stock-months than the universe: 1 missing, 0 extra")
    pd.concat([panel, panel.iloc[[0]]]).to_parquet(panel_path, index=False)
    rejected("2004 panel: 1 rows repeat a permno/yyyymm key")
    changed = panel.copy()
    changed.loc[0, "mktcap"] = changed["mktcap"].to_numpy(dtype="float64")[0] + 1.0
    changed.to_parquet(panel_path, index=False)  # the panel and the universe disagree on the same stock-month
    rejected("2004 panel: mktcap differs from the universe in 1 rows")
    panel.drop(columns="Illiquidity").to_parquet(panel_path, index=False)
    rejected("2004 panel: columns missing: ['Illiquidity']")
    panel.to_parquet(panel_path, index=False)

    conditional.iloc[1:].to_parquet(conditional_path, index=False)
    rejected("2004 conditional disagreement covers different stock-months than the universe: 1 missing, 0 extra")
    conditional.to_parquet(conditional_path, index=False)
    (paths.conditional_disagreement / "2004.json").unlink()
    rejected("no complete conditional disagreement")
    (paths.conditional_disagreement / "2004.json").write_text("{}")

    pd.concat([universe, universe.iloc[[0]]]).to_parquet(universe_path, index=False)
    rejected("2004 universe: 1 rows repeat a permno/yyyymm key")
    bad = universe.copy()
    bad.loc[0, "mktcap"] = 0.0
    bad.to_parquet(universe_path, index=False)
    rejected("2004 universe: mktcap must be finite and strictly positive")
    universe[universe["yyyymm"] != 200412].to_parquet(universe_path, index=False)
    rejected("2004 universe: not the 12 OOS months; missing [200412], extra []")
    universe.drop(columns="exchange").to_parquet(universe_path, index=False)
    rejected("2004 universe: columns")
    universe.to_parquet(universe_path, index=False)
    (paths.diagnostics / "summary.json").unlink()
    rejected("no complete universe")
    (paths.diagnostics / "summary.json").write_text("{}")
    assert main(argv) == 0 and marker.exists()


@pytest.mark.parametrize("name", ["universe", "panel", "conditional_disagreement"])
def test_a_changed_upstream_file_is_detected_at_load(tmp_path, name):
    paths, argv = _setup(tmp_path)
    assert main(argv) == 0
    out = load_h2_controls(paths, 2004)
    source = h2.input_files(paths, 2004)[name]
    frame = pd.read_parquet(source)
    frame.sample(frac=1.0, random_state=5).to_parquet(source, index=False)  # same content, other bytes: still rejected
    with pytest.raises(DataValidationError, match=f"the {name} file changed since the controls were built"):
        load_h2_controls(paths, 2004)
    assert main(argv) == 0  # rebuilt from the current inputs
    pd.testing.assert_frame_equal(load_h2_controls(paths, 2004), out)


def test_a_modified_output_file_is_detected_at_load(tmp_path):
    paths, argv = _setup(tmp_path)
    assert main(argv) == 0
    path = paths.h2_controls / "2004.parquet"
    out = pd.read_parquet(path)
    out.loc[0, "illiquidity_rank"] = 0.5 if out.loc[0, "illiquidity_rank"] != 0.5 else -0.5
    out.to_parquet(path, index=False)  # the output edited, every upstream input unchanged
    with pytest.raises(DataValidationError, match="the output file does not match the SHA-256 in its marker"):
        load_h2_controls(paths, 2004)
    assert main(argv) == 0
    load_h2_controls(paths, 2004)


def test_a_failed_write_leaves_no_marker_or_temporary_file_and_can_be_rebuilt(tmp_path, monkeypatch):
    paths, argv = _setup(tmp_path)
    directory = paths.h2_controls
    assert main(argv) == 0
    out = load_h2_controls(paths, 2004)

    def failing_to_parquet(self, *args, **kwargs):
        raise OSError("disk full while writing the parquet file")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", failing_to_parquet)
    with pytest.raises(OSError, match="disk full"):
        h2.run_h2_controls(paths, Path(argv[3]), [2004], log=lambda _: None)
    assert sorted(p.name for p in directory.iterdir()) == ["2004.parquet"]  # the previous parquet, no marker, no .tmp
    monkeypatch.undo()

    replace = os.replace

    def failing_marker_replace(src, dst):
        if str(dst).endswith("2004.json"):
            raise OSError("interrupted before the marker was published")
        replace(src, dst)

    monkeypatch.setattr(h2.os, "replace", failing_marker_replace)
    with pytest.raises(OSError, match="before the marker"):
        h2.run_h2_controls(paths, Path(argv[3]), [2004], log=lambda _: None)
    assert sorted(p.name for p in directory.iterdir()) == ["2004.parquet"]  # written, but not marked complete
    with pytest.raises(DataValidationError, match="no complete H2 controls"):
        load_h2_controls(paths, 2004)
    monkeypatch.undo()

    assert main(argv) == 0  # a later run rebuilds a complete, valid year
    pd.testing.assert_frame_equal(load_h2_controls(paths, 2004), out)


def test_stale_controls_are_not_accepted_and_are_removed_with_their_sources(tmp_path):
    paths, argv = _setup(tmp_path)
    assert main(argv) == 0
    directory = paths.h2_controls
    out = load_h2_controls(paths, 2004)

    panel_path = paths.panel / "panel_2004.parquet"
    panel = pd.read_parquet(panel_path)
    panel.loc[0, "RealizedVol"] = panel["RealizedVol"].to_numpy(dtype="float64")[0] - 0.01
    panel.to_parquet(panel_path, index=False)  # an upstream input changed after the controls were built
    with pytest.raises(DataValidationError, match="the panel file changed since the controls were built"):
        load_h2_controls(paths, 2004)
    assert main(argv) == 0  # rebuilt from the changed input
    rebuilt = load_h2_controls(paths, 2004)
    assert not rebuilt.equals(out)

    marker = directory / "2004.json"
    metadata = json.loads(marker.read_text())
    metadata["definition_sha256"] = "0" * 64
    marker.write_text(json.dumps(metadata))
    with pytest.raises(DataValidationError, match="another control definition"):
        load_h2_controls(paths, 2004)
    marker.unlink()
    with pytest.raises(DataValidationError, match="no complete H2 controls"):
        load_h2_controls(paths, 2004)

    assert main(argv) == 0
    unrelated = directory / "2005.json"
    unrelated.write_text("controls of another year")
    remove_conditional_disagreement(paths, 2004)  # rerunning or removing the conditional disagreement of a year
    assert sorted(p.name for p in directory.iterdir()) == ["2005.json"]
    remove_derived_outputs(paths)  # a new build or CRSP extract
    assert not directory.exists()
