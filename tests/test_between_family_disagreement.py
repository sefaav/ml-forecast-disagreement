"""Raw between-family disagreement from the ensemble signals (synthetic data)."""

import hashlib
import json
import math

import numpy as np
import pandas as pd
import yaml

from disagreement.data.config import DEFAULT_CONFIG, DataPaths
from disagreement.signals.between import OUTPUT_COLUMNS, between_family_disagreement, main
from disagreement.signals.ensemble import ENSEMBLE_COLUMNS
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
    ranks = {1: [0.4, 0.4, 0.4, 0.4], 2: [0.0, 0.0, 1.0, 1.0], 3: [0.1, 0.2, 0.3, 0.4]}
    ensemble = _ensemble([(p, 200401, values) for p, values in ranks.items()])
    out = between_family_disagreement(ensemble)
    expected = [0.0, 0.5, math.sqrt(0.0125)]  # population SD (ddof = 0)
    np.testing.assert_allclose(out["between_family_disagreement_raw"].to_numpy(), expected, rtol=0, atol=1e-15)
    sample_sd = np.std(ranks[3], ddof=1)  # 0.129..., what ddof = 1 would give for stock 3
    assert abs(out["between_family_disagreement_raw"].iloc[2] - sample_sd) > 0.01
    assert out["ensemble_signal"].tolist() == ensemble["ensemble_signal"].tolist()  # copied
    assert not np.allclose(out["between_family_disagreement_raw"].to_numpy(), [0.0, 0.5, 1.0])  # not re-ranked


def _setup(tmp_path):
    """A complete ensemble year 2004, with a config whose episodes include 2004."""
    rng = np.random.default_rng(0)
    rows = [
        (p, m, list(rng.integers(0, 5, size=4) / 4))
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
