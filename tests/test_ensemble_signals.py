"""Ensemble signal from the four family consensus ranks (synthetic data)."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from disagreement.data.config import DEFAULT_CONFIG, DataPaths
from disagreement.data.validation import DataValidationError
from disagreement.models.selection import FAMILIES
from disagreement.signals.ensemble import ENSEMBLE_COLUMNS, ensemble_signals, family_ranks, main
from disagreement.signals.family import SIGNAL_COLUMNS
from disagreement.walkforward.calendar import month_range

R = 5


def _signals(family, ranks):
    """ranks: {(permno, yyyymm): family_rank}. Five replica rows each; the other rank columns are unrelated."""
    rows = [(p, m, family, r, (r + 1) / 10, 0.3, rank) for (p, m), rank in ranks.items() for r in range(R)]
    return pd.DataFrame(rows, columns=list(SIGNAL_COLUMNS))


def _ranks(table):
    """table: {family: family ranks of permnos 1, 2, ... in 200401}."""
    return {
        f: family_ranks(_signals(f, {(p, 200401): v for p, v in enumerate(values, start=1)}), f, R, f)
        for f, values in table.items()
    }


def test_ensemble_is_the_plain_average_of_the_four_family_ranks_and_is_not_re_ranked():
    table = {  # stocks A, B, C
        "elastic_net": [0.1, 0.4, 0.6],
        "random_forest": [0.2, 0.4, 0.7],
        "lightgbm": [0.3, 0.4, 0.8],
        "mlp": [0.4, 0.4, 0.9],
    }
    out = ensemble_signals(_ranks(table))
    assert out["permno"].tolist() == [1, 2, 3]
    for family, values in table.items():
        assert out[f"{family}_rank"].tolist() == values  # the original family ranks
    np.testing.assert_allclose(out["ensemble_signal"].to_numpy(), [0.25, 0.40, 0.75], rtol=0, atol=1e-15)
    assert not np.allclose(out["ensemble_signal"].to_numpy(), [0.0, 0.5, 1.0])  # not a fresh cross-sectional rank


def test_replica_rows_collapse_to_one_equal_weight_per_family():
    # Stock k + 1 ranks 1 in family k only; stock 5 ranks 1 in every family.
    signals = {
        f: _signals(f, {(p, 200401): float(p in (k + 1, 5)) for p in range(1, 6)}) for k, f in enumerate(FAMILIES)
    }
    assert all(len(s) == 5 * R for s in signals.values())
    out = ensemble_signals({f: family_ranks(s, f, R, f) for f, s in signals.items()})
    assert len(out) == 5  # one row per stock-month, not per replica
    assert out["ensemble_signal"].tolist() == [0.25, 0.25, 0.25, 0.25, 1.0]  # each family weighs exactly 1/4

    broken = signals["lightgbm"].copy()
    broken.loc[broken.index[-1], "family_rank"] = 0.5  # one replica row of stock 5 disagrees
    with pytest.raises(DataValidationError, match="family_rank differs across replicas in 1 stock-months"):
        family_ranks(broken, "lightgbm", R, "lightgbm")


def _setup(tmp_path):
    """Family signals of the four families for OOS year 2004, with a config whose episodes include 2004."""
    rng = np.random.default_rng(0)
    keys = [(p, m) for m in month_range(200401, 200412) for p in range(1, 6 + m % 3)]
    paths = DataPaths(tmp_path / "data")
    for family in FAMILIES:
        directory = paths.family_signals / family
        directory.mkdir(parents=True)
        ranks = {key: float(rng.integers(0, 5)) / 4 for key in keys}
        _signals(family, ranks).sample(frac=1.0, random_state=0).to_parquet(directory / "2004.parquet", index=False)
        (directory / "2004.json").write_text(json.dumps({"family": family, "oos_year": 2004}))
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["data"]["sample"] = {"start": 200001, "end": 200512}
    config["walk_forward"] = {"first_oos_year": 2004, "validation_years": 2}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    argv = ["--data-dir", str(paths.root), "--config", str(config_path), "--year", "2004"]
    return paths, argv, keys


def test_output_schema_coverage_and_reproducible_completion(tmp_path):
    paths, argv, keys = _setup(tmp_path)
    outputs = []
    for _ in range(2):
        assert main(argv) == 0
        outputs.append(
            (
                pd.read_parquet(paths.ensemble_signals / "2004.parquet"),
                json.loads((paths.ensemble_signals / "2004.json").read_text()),
            )
        )
    out, metadata = outputs[0]
    pd.testing.assert_frame_equal(out, outputs[1][0])
    assert metadata == outputs[1][1]
    assert list(out.columns) == list(ENSEMBLE_COLUMNS) and len(out) == len(keys)
    assert not out.duplicated(["permno", "yyyymm"]).any()
    assert sorted(zip(out["permno"], out["yyyymm"])) == sorted(keys)
    assert out.equals(out.sort_values(["yyyymm", "permno"], ignore_index=True))
    values = out[list(ENSEMBLE_COLUMNS[2:])].to_numpy()
    assert np.isfinite(values).all() and values.min() >= 0.0 and values.max() <= 1.0

    indexed = out.set_index(["permno", "yyyymm"])
    for family in FAMILIES:  # copied from the source family ranks
        source = pd.read_parquet(paths.family_signals / family / "2004.parquet")
        expected = source.groupby(["permno", "yyyymm"])["family_rank"].first()
        np.testing.assert_array_equal(indexed[f"{family}_rank"].reindex(expected.index).to_numpy(), expected.to_numpy())
    total = out["elastic_net_rank"] + out["random_forest_rank"] + out["lightgbm_rank"] + out["mlp_rank"]
    np.testing.assert_array_equal(out["ensemble_signal"].to_numpy(), (total / 4.0).to_numpy())

    assert (metadata["oos_year"], metadata["families"], metadata["n_stock_months"]) == (2004, list(FAMILIES), len(keys))
    assert metadata["months"] == month_range(200401, 200412)
    assert metadata["family_signals_sha256"] == {
        f: hashlib.sha256((paths.family_signals / f / "2004.parquet").read_bytes()).hexdigest() for f in FAMILIES
    }
    assert not list(paths.ensemble_signals.glob("*.tmp"))


def test_mismatched_or_incomplete_family_signals_leave_no_completion_marker(tmp_path, capsys):
    paths, argv, keys = _setup(tmp_path)
    assert main(argv) == 0
    marker = paths.ensemble_signals / "2004.json"
    source = paths.family_signals / "mlp" / "2004.parquet"
    signals = pd.read_parquet(source)

    disagreement = paths.between_family_disagreement
    stale = [disagreement / f"2004.{suffix}" for suffix in ("parquet", "json", "parquet.tmp", "json.tmp")]
    unrelated = disagreement / "2005.json"
    disagreement.mkdir(parents=True)
    for path in stale + [unrelated]:
        path.write_text("disagreement of an earlier ensemble")

    permno, month = keys[3]
    lost = (signals["permno"] == permno) & (signals["yyyymm"] == month)
    signals[~lost].to_parquet(source, index=False)  # mlp loses one stock-month, all its replicas
    assert main(argv) == 2
    assert "mlp covers different stock-months than elastic_net: 1 missing, 0 extra" in capsys.readouterr().err
    assert not marker.exists()  # not an inner join over the other stock-months
    assert not any(path.exists() for path in stale)  # stale as soon as the ensemble rerun starts
    assert unrelated.read_text() == "disagreement of an earlier ensemble"

    extra = _signals("mlp", {(99, 200401): 0.5})
    pd.concat([signals, extra], ignore_index=True).to_parquet(source, index=False)
    assert main(argv) == 2
    assert "0 missing, 1 extra" in capsys.readouterr().err
    assert not marker.exists()

    signals.drop(columns="yyyymm").to_parquet(source, index=False)  # completion marker still present
    assert main(argv) == 2  # a validation failure, not a KeyError
    assert "mlp 2004 family signals: columns" in capsys.readouterr().err
    assert not marker.exists()

    signals.to_parquet(source, index=False)
    (paths.family_signals / "mlp" / "2004.json").unlink()
    assert main(argv) == 2  # family signals without their completion marker
    assert "no complete family signals" in capsys.readouterr().err
    assert not marker.exists()


def test_a_month_missing_from_all_four_families_is_rejected(tmp_path, capsys):
    paths, argv, _ = _setup(tmp_path)
    assert main(argv) == 0
    for family in FAMILIES:  # identical coverage across families, completion markers kept
        source = paths.family_signals / family / "2004.parquet"
        signals = pd.read_parquet(source)
        signals[signals["yyyymm"] != 200406].to_parquet(source, index=False)
    assert main(argv) == 2
    assert "2004: not the 12 OOS months; missing [200406], extra []" in capsys.readouterr().err
    assert not (paths.ensemble_signals / "2004.json").exists()
