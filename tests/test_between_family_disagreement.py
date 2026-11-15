"""Raw between-family disagreement from the ensemble signals (synthetic data)."""

import hashlib
import json
import math

import numpy as np
import pandas as pd
import yaml

from disagreement.data.config import DEFAULT_CONFIG, DataPaths
from disagreement.models.selection import FAMILIES
from disagreement.signals.between import OUTPUT_COLUMNS, between_family_disagreement, load_ensemble, main
from disagreement.signals.ensemble import ENSEMBLE_COLUMNS, ensemble_signals
from disagreement.walkforward.calendar import month_range

RANKS = list(ENSEMBLE_COLUMNS[2:6])


def _ensemble(rows):
    """rows: (permno, yyyymm, four family ranks); ensemble_signal as the ensemble stage computes it."""
    frame = pd.DataFrame([(p, m, *ranks) for p, m, ranks in rows], columns=["permno", "yyyymm", *RANKS])
    frame["ensemble_signal"] = (
        frame["elastic_net_rank"] + frame["random_forest_rank"] + frame["lightgbm_rank"] + frame["mlp_rank"]
    ) / 4.0
    return frame


def test_disagreement_is_the_population_sd_of_the_four_family_ranks():
    ranks = {1: [0.5, 0.5, 0.5, 0.5], 2: [0.0, 0.0, 1.0, 1.0], 3: [0.0, 0.25, 0.5, 0.75]}  # unit ranks, n = 3
    ensemble = _ensemble([(p, 200401, values) for p, values in ranks.items()])
    out = between_family_disagreement(ensemble)
    expected = [0.0, 0.5, math.sqrt(0.078125)]  # population SD (ddof = 0)
    np.testing.assert_allclose(out["between_family_disagreement_raw"].to_numpy(), expected, rtol=0, atol=1e-15)
    sample_sd = np.std(ranks[3], ddof=1)  # 0.323..., what ddof = 1 would give for stock 3
    assert abs(out["between_family_disagreement_raw"].iloc[2] - sample_sd) > 0.01
    assert out["ensemble_signal"].tolist() == ensemble["ensemble_signal"].tolist()  # copied
    assert not np.allclose(out["between_family_disagreement_raw"].to_numpy(), [0.0, 0.5, 1.0])  # not re-ranked


def _naive(ranks):  # the former float formula, left to right (Python's sum() would round exactly)
    a, b, c, d = ranks
    s = (a + b + c + d) / 4.0
    return math.sqrt(((a - s) ** 2 + (b - s) ** 2 + (c - s) ** 2 + (d - s) ** 2) / 4.0)


def test_permuted_or_translated_family_ranks_have_exactly_the_same_disagreement():
    # Seven stocks: unit ranks are multiples of 1/12. Stock 2 permutes stock 1; stock 4 is stock 3 shifted by 2/12.
    twelfths = {1: [0, 0, 0, 2], 2: [0, 0, 2, 0], 3: [0, 0, 0, 4], 4: [2, 2, 2, 6],
                5: [12, 12, 12, 12], 6: [6, 6, 6, 6], 7: [3, 9, 3, 9]}
    rows = [(p, 200401, [k / 12 for k in values]) for p, values in twelfths.items()]
    assert _naive(rows[0][2]) != _naive(rows[1][2]) and _naive(rows[2][2]) != _naive(rows[3][2])  # the defect
    out = between_family_disagreement(_ensemble(rows))["between_family_disagreement_raw"]
    assert out.iloc[0] == out.iloc[1] and out.iloc[2] == out.iloc[3]  # bit-identical
    assert out.iloc[0] == math.sqrt(12) / 48 and out.iloc[4] == out.iloc[5] == 0.0  # sqrt(4 * 4 - 2 ** 2) / (4 * 12)


def test_exact_ensemble_signals_pass_the_loader_consistency_check(tmp_path):
    rng = np.random.default_rng(5)
    index = pd.MultiIndex.from_tuples([(p, m) for m in month_range(200401, 200412) for p in range(1, 8)],
                                      names=["permno", "yyyymm"])
    ranks = {f: pd.Series(rng.integers(0, 13, size=len(index)) / 12, index=index) for f in FAMILIES}  # n = 7
    signals = ensemble_signals(ranks)
    naive = (signals[RANKS[0]] + signals[RANKS[1]] + signals[RANKS[2]] + signals[RANKS[3]]) / 4.0  # former formula
    assert (naive != signals["ensemble_signal"]).any()  # values the former equality check would have rejected
    paths = DataPaths(tmp_path / "data")
    paths.ensemble_signals.mkdir(parents=True)
    signals.to_parquet(paths.ensemble_signals / "2004.parquet", index=False)
    (paths.ensemble_signals / "2004.json").write_text(json.dumps({"oos_year": 2004}))
    pd.testing.assert_frame_equal(load_ensemble(paths, 2004), signals)


def _setup(tmp_path):
    """A complete ensemble year 2004, with a config whose episodes include 2004."""
    rng = np.random.default_rng(0)
    rows = [
        (p, m, list(rng.integers(0, 3, size=4) / 2))  # unit ranks for any month size
        for m in month_range(200401, 200412)
        for p in range(1, 6 + m % 3)
    ]
    paths = DataPaths(tmp_path / "data")
    paths.ensemble_signals.mkdir(parents=True)
    _ensemble(rows).sample(frac=1.0, random_state=0).to_parquet(paths.ensemble_signals / "2004.parquet", index=False)
    (paths.ensemble_signals / "2004.json").write_text(json.dumps({"oos_year": 2004}))
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["data"]["sample"] = {"start": 200001, "end": 200512}
    config["walk_forward"] = {"first_oos_year": 2004, "validation_years": 2}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    argv = ["--data-dir", str(paths.root), "--config", str(config_path), "--year", "2004"]
    return paths, argv, len(rows)


def test_output_schema_coverage_and_reproducible_completion(tmp_path):
    paths, argv, n_rows = _setup(tmp_path)
    directory = paths.between_family_disagreement
    outputs = []
    for _ in range(2):
        assert main(argv) == 0
        outputs.append((pd.read_parquet(directory / "2004.parquet"), json.loads((directory / "2004.json").read_text())))
    out, metadata = outputs[0]
    pd.testing.assert_frame_equal(out, outputs[1][0])
    assert metadata == outputs[1][1]
    assert list(out.columns) == list(OUTPUT_COLUMNS) and len(out) == n_rows
    assert not out.duplicated(["permno", "yyyymm"]).any()
    assert out.equals(out.sort_values(["yyyymm", "permno"], ignore_index=True))
    disagreement = out["between_family_disagreement_raw"].to_numpy()
    assert np.isfinite(disagreement).all() and disagreement.min() >= 0.0 and disagreement.max() <= 0.5

    source = paths.ensemble_signals / "2004.parquet"
    ensemble = pd.read_parquet(source).sort_values(["yyyymm", "permno"], ignore_index=True)
    np.testing.assert_array_equal(out["ensemble_signal"].to_numpy(), ensemble["ensemble_signal"].to_numpy())
    np.testing.assert_allclose(disagreement, ensemble[RANKS].to_numpy().std(axis=1, ddof=0), rtol=0, atol=1e-15)
    assert metadata == {
        "oos_year": 2004,
        "n_stock_months": n_rows,
        "months": month_range(200401, 200412),
        "ensemble_signals_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    assert not list(directory.glob("*.tmp"))


def test_invalid_ensemble_input_leaves_no_completion_marker(tmp_path, capsys):
    paths, argv, _ = _setup(tmp_path)
    assert main(argv) == 0
    marker = paths.between_family_disagreement / "2004.json"
    source = paths.ensemble_signals / "2004.parquet"
    ensemble = pd.read_parquet(source)
    conditional = [paths.conditional_disagreement / f"2004.{suffix}" for suffix in ("parquet", "json", "json.tmp")]
    unrelated = paths.conditional_disagreement / "2005.json"
    paths.conditional_disagreement.mkdir(parents=True)
    diagnostics = [paths.mechanical_bias_diagnostics / name for name in ("summary.json", "monthly_correlations.csv")]
    paths.mechanical_bias_diagnostics.mkdir(parents=True)
    for path in conditional + [unrelated] + diagnostics:
        path.write_text("conditional disagreement of an earlier run")

    changed = ensemble.copy()
    changed.loc[changed.index[0], "ensemble_signal"] = 0.5 if changed["ensemble_signal"].iloc[0] != 0.5 else 0.25
    changed.to_parquet(source, index=False)  # saved signal no longer the mean of its family ranks
    assert main(argv) == 2
    assert "ensemble_signal is not the mean of the family ranks in 1 rows" in capsys.readouterr().err
    assert not marker.exists()
    assert not any(path.exists() for path in conditional)  # stale as soon as the rerun starts
    assert unrelated.read_text() == "conditional disagreement of an earlier run"
    assert not paths.mechanical_bias_diagnostics.exists()  # aggregate over all years: stale as a whole

    ensemble[ensemble["yyyymm"] != 200406].to_parquet(source, index=False)  # 11 months
    assert main(argv) == 2
    assert "not the 12 OOS months; missing [200406], extra []" in capsys.readouterr().err
    assert not marker.exists()

    ensemble.to_parquet(source, index=False)
    (paths.ensemble_signals / "2004.json").unlink()
    assert main(argv) == 2  # ensemble without its completion marker
    assert "no complete ensemble signals" in capsys.readouterr().err
    assert not marker.exists()
