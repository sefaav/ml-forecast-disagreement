"""Within-family estimation uncertainty from the replica ranks (synthetic data)."""

import hashlib
import json
import math

import numpy as np
import pandas as pd
import yaml

from disagreement.data.config import DEFAULT_CONFIG, DataPaths
from disagreement.models.selection import FAMILIES
from disagreement.signals.family import SIGNAL_COLUMNS
from disagreement.signals.within import OUTPUT_COLUMNS, main, within_family_uncertainty
from disagreement.walkforward.calendar import month_range

R = 5
A = [0.0, 0.0, 1.0, 1.0, 1.0]  # sample SD sqrt(0.3); population SD sqrt(0.24)
B = [1.0, 0.0, 1.0, 0.0, 1.0]  # A in another replica order
C = [0.5] * 5  # SD 0
D = [0.0, 0.5, 0.5, 0.5, 1.0]  # sample SD sqrt(0.125)


def _signals(family, replica_ranks, family_rank=0.5):
    """replica_ranks: {(permno, yyyymm): the R replica ranks}."""
    rows = [
        (p, m, family, r, q, float(np.mean(qs)), family_rank)
        for (p, m), qs in replica_ranks.items()
        for r, q in enumerate(qs)
    ]
    return pd.DataFrame(rows, columns=list(SIGNAL_COLUMNS))


def test_sample_sd_across_replicas_ranked_by_family_and_month_then_averaged():
    january = {  # stocks 1-4 in 200401; the patterns differ across families
        "elastic_net": [A, B, C, D],
        "random_forest": [C, C, A, D],
        "lightgbm": [D, A, C, C],
        "mlp": [C, C, C, C],
    }
    signals = {}
    for family, patterns in january.items():
        replica_ranks = {(p, 200401): qs for p, qs in enumerate(patterns, start=1)}
        replica_ranks |= {(1, 200402): C, (2, 200402): D}  # February: two stocks only
        signals[family] = _signals(family, replica_ranks)
    out = within_family_uncertainty(signals, R).set_index(["yyyymm", "permno"])
    jan, feb = out.loc[200401], out.loc[200402]

    en = jan["elastic_net_uncertainty_sd"]
    assert math.isclose(en[1], math.sqrt(0.3), rel_tol=0, abs_tol=1e-15)  # ddof = 1 ...
    assert abs(en[1] - np.std(A, ddof=0)) > 0.05  # ... not ddof = 0 (0.4899)
    assert en[1] == en[2]  # same replica ranks in another order: exactly the same SD, a tie
    np.testing.assert_allclose(en.to_numpy(), [math.sqrt(0.3), math.sqrt(0.3), 0.0, math.sqrt(0.125)], atol=1e-15)

    # Each family's SDs ranked on their own, with average ties; a constant month gives 0.5.
    expected = {
        "elastic_net": [5 / 6, 5 / 6, 0.0, 1 / 3],
        "random_forest": [1 / 6, 1 / 6, 1.0, 2 / 3],
        "lightgbm": [2 / 3, 1.0, 1 / 6, 1 / 6],
        "mlp": [0.5, 0.5, 0.5, 0.5],
    }
    for family, ranks in expected.items():
        np.testing.assert_allclose(jan[f"{family}_uncertainty_rank"].to_numpy(), ranks, rtol=0, atol=1e-15)
        assert feb[f"{family}_uncertainty_rank"].tolist() == [0.0, 1.0]  # February ranked on its own

    mean = np.mean(list(expected.values()), axis=0)  # 0.5417, 0.625, 0.4167, 0.4167
    np.testing.assert_allclose(jan["within_family_uncertainty_raw"].to_numpy(), mean, rtol=0, atol=1e-15)
    assert jan["within_family_uncertainty_raw"].max() < 1.0  # the mean is not ranked again


def _setup(tmp_path):
    """Family signals of the four families for OOS year 2004, with a config whose episodes include 2004."""
    rng = np.random.default_rng(0)
    keys = [(p, m) for m in month_range(200401, 200412) for p in range(1, 6 + m % 3)]
    paths = DataPaths(tmp_path / "data")
    for family in FAMILIES:
        directory = paths.family_signals / family
        directory.mkdir(parents=True)
        replica_ranks = {key: list(rng.integers(0, 5, size=R) / 4) for key in keys}
        frame = _signals(family, replica_ranks, family_rank=0.25)
        frame.sample(frac=1.0, random_state=0).to_parquet(directory / "2004.parquet", index=False)
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
    directory = paths.within_family_uncertainty
    outputs = []
    for _ in range(2):
        assert main(argv) == 0
        outputs.append((pd.read_parquet(directory / "2004.parquet"), json.loads((directory / "2004.json").read_text())))
    out, metadata = outputs[0]
    pd.testing.assert_frame_equal(out, outputs[1][0])
    assert metadata == outputs[1][1]
    assert list(out.columns) == list(OUTPUT_COLUMNS) and len(out) == len(keys)
    assert not out.duplicated(["permno", "yyyymm"]).any()
    assert sorted(zip(out["permno"], out["yyyymm"])) == sorted(keys)
    assert out.equals(out.sort_values(["yyyymm", "permno"], ignore_index=True))
    assert sorted(out["yyyymm"].unique().tolist()) == month_range(200401, 200412)

    sds = out[[f"{f}_uncertainty_sd" for f in FAMILIES]].to_numpy()
    ranks = out[[f"{f}_uncertainty_rank" for f in FAMILIES]].to_numpy()
    aggregate = out["within_family_uncertainty_raw"].to_numpy()
    assert np.isfinite(sds).all() and sds.min() >= 0.0
    assert np.isfinite(ranks).all() and ranks.min() >= 0.0 and ranks.max() <= 1.0
    assert aggregate.min() >= 0.0 and aggregate.max() <= 1.0
    np.testing.assert_array_equal(aggregate, (ranks[:, 0] + ranks[:, 1] + ranks[:, 2] + ranks[:, 3]) / 4.0)

    indexed = out.set_index(["permno", "yyyymm"])
    for family in FAMILIES:  # the SD of each stock-month's saved replica ranks
        source = pd.read_parquet(paths.family_signals / family / "2004.parquet")
        expected = source.groupby(["permno", "yyyymm"])["replica_rank"].agg(lambda q: np.std(q, ddof=1))
        got = indexed[f"{family}_uncertainty_sd"].reindex(expected.index).to_numpy()
        np.testing.assert_allclose(got, expected.to_numpy(), rtol=0, atol=1e-15)

    assert (metadata["oos_year"], metadata["replicas"], metadata["n_stock_months"]) == (2004, R, len(keys))
    assert metadata["months"] == month_range(200401, 200412)
    assert metadata["family_signals_sha256"] == {
        f: hashlib.sha256((paths.family_signals / f / "2004.parquet").read_bytes()).hexdigest() for f in FAMILIES
    }
    assert not list(directory.glob("*.tmp"))


def test_invalid_family_signals_leave_no_completion_marker(tmp_path, capsys):
    paths, argv, keys = _setup(tmp_path)
    assert main(argv) == 0
    marker = paths.within_family_uncertainty / "2004.json"
    lightgbm, mlp = (paths.family_signals / f / "2004.parquet" for f in ("lightgbm", "mlp"))
    lightgbm_signals, mlp_signals = pd.read_parquet(lightgbm), pd.read_parquet(mlp)

    lightgbm_signals.iloc[1:].to_parquet(lightgbm, index=False)  # one stock-month loses a replica
    assert main(argv) == 2
    assert "lightgbm 2004 family signals: every stock-month needs exactly replicas 0..4" in capsys.readouterr().err
    assert not marker.exists()
    lightgbm_signals.to_parquet(lightgbm, index=False)

    permno, month = keys[3]
    lost = (mlp_signals["permno"] == permno) & (mlp_signals["yyyymm"] == month)
    mlp_signals[~lost].to_parquet(mlp, index=False)  # mlp loses one stock-month, all its replicas
    assert main(argv) == 2
    assert "mlp covers different stock-months than elastic_net: 1 missing, 0 extra" in capsys.readouterr().err
    assert not marker.exists()  # not an inner join

    mlp_signals.drop(columns="replica_rank").to_parquet(mlp, index=False)
    assert main(argv) == 2
    assert "mlp 2004 family signals: columns" in capsys.readouterr().err
    assert not marker.exists()

    mlp_signals.to_parquet(mlp, index=False)
    (paths.family_signals / "mlp" / "2004.json").unlink()
    assert main(argv) == 2  # family signals without their completion marker
    assert "no complete family signals" in capsys.readouterr().err
    assert not marker.exists()
