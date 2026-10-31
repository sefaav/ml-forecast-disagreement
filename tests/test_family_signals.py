"""Family signals from raw OOS replica predictions (synthetic data)."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from disagreement.data.config import DEFAULT_CONFIG, DataPaths
from disagreement.data.validation import DataValidationError
from disagreement.signals.family import SIGNAL_COLUMNS, family_signals, main
from disagreement.walkforward.calendar import month_range

R = 5


def _predictions(rows):
    """rows: (permno, yyyymm, replica, prediction) for one family."""
    frame = pd.DataFrame(rows, columns=["permno", "yyyymm", "replica", "prediction"])
    frame["family"] = "elastic_net"
    return frame[["permno", "yyyymm", "family", "replica", "prediction"]]


def _by_key(signals, column):
    return signals.set_index(["permno", "yyyymm", "replica"])[column]


def test_replica_ranks_use_average_ties_on_the_unit_interval():
    raw = [3.0, 1.0, 2.0, 2.0, 5.0]  # ranks 4, 1, 2.5, 2.5, 5 over n = 5
    rows = [(p, 200401, r, raw[p - 1] if r == 0 else 7.0) for p in range(1, 6) for r in range(R)]
    signals = family_signals(_predictions(rows), R)
    first = signals[signals["replica"] == 0].sort_values("permno")["replica_rank"].tolist()
    assert first == [0.75, 0.0, 0.375, 0.375, 1.0]  # (rank - 1) / (n - 1)
    assert (signals.loc[signals["replica"] > 0, "replica_rank"] == 0.5).all()  # constant cross-section
    single = family_signals(_predictions([(1, 200401, r, float(r)) for r in range(R)]), R)
    assert (single[["replica_rank", "mean_replica_rank", "family_rank"]] == 0.5).all().all()  # n = 1


def test_ranks_are_computed_within_each_month_and_replica_only():
    rng = np.random.default_rng(0)
    rows = []
    for month, scale in ((200401, 1.0), (200402, 1e6)):  # very different scales across months
        for r in range(R):
            values = rng.permutation(6) * scale * (r + 1) - 3.0 * r  # different orderings and scales
            rows += [(p, month, r, float(values[p - 1])) for p in range(1, 7)]
    predictions = _predictions(rows)
    ranks = _by_key(family_signals(predictions, R), "replica_rank")
    for (month, r), group in predictions.groupby(["yyyymm", "replica"]):
        expected = (group["prediction"].rank(method="average") - 1) / (len(group) - 1)
        got = ranks.loc[list(zip(group["permno"], group["yyyymm"], group["replica"]))].to_numpy()
        np.testing.assert_array_equal(got, expected.to_numpy())


def test_family_rank_re_ranks_the_mean_of_the_replica_ranks():
    # Replica ranks of three stocks (n = 3, so ranks map to 0, 0.5, 1).
    replica_ranks = {1: [0.0, 0.0, 0.5, 0.0, 0.0], 2: [0.5, 1.0, 0.0, 1.0, 0.5], 3: [1.0, 0.5, 1.0, 0.5, 1.0]}
    rows = [(p, 200401, r, value) for p, values in replica_ranks.items() for r, value in enumerate(values)]
    signals = family_signals(_predictions(rows), R)
    stock = signals[signals["replica"] == 0].set_index("permno")
    np.testing.assert_allclose(stock["mean_replica_rank"].to_numpy(), [0.1, 0.6, 0.8])
    assert stock["family_rank"].tolist() == [0.0, 0.5, 1.0]  # a fresh ranking, not the means themselves
    assert signals.groupby("permno")["family_rank"].nunique().eq(1).all()  # repeated on every replica row

    # Same replica ranks in a different replica order: an exact tie in the consensus.
    tied = {1: [0.0, 0.5, 1.0, 0.5, 0.0], 2: [1.0, 0.0, 0.5, 0.0, 0.5], 3: [0.5, 1.0, 0.0, 1.0, 1.0]}
    rows = [(p, 200401, r, v) for p, values in tied.items() for r, v in enumerate(values)]
    stock = family_signals(_predictions(rows), R).query("replica == 0").set_index("permno")
    assert stock.loc[1, "family_rank"] == stock.loc[2, "family_rank"] == 0.25 and stock.loc[3, "family_rank"] == 1.0

    # Replicas 0-3 order A > B > C on a small scale; replica 4 reverses it on a vastly larger one.
    raw = {1: [2.0, 2.0, 2.0, 2.0, 0.0], 2: [1.0, 1.0, 1.0, 1.0, 500.0], 3: [0.0, 0.0, 0.0, 0.0, 1000.0]}
    rows = [(p, 200401, r, v) for p, values in raw.items() for r, v in enumerate(values)]
    assert [np.mean(raw[p]) for p in (1, 2, 3)] == [1.6, 100.8, 200.0]  # averaging raw values: C > B > A
    stock = family_signals(_predictions(rows), R).query("replica == 0").set_index("permno")
    np.testing.assert_allclose(stock["mean_replica_rank"].to_numpy(), [0.8, 0.5, 0.2])
    assert stock["family_rank"].tolist() == [1.0, 0.5, 0.0]  # ranks are averaged, not raw predictions: A > B > C


def _setup(tmp_path):
    """Synthetic predictions for one family and OOS year 2004, with a config whose episodes include 2004."""
    rng = np.random.default_rng(1)
    rows = [
        (p, month, r, float(np.round(rng.standard_normal(), 1)))  # rounded: many ties
        for month in month_range(200401, 200412)
        for p in range(1, 9 + month % 3)
        for r in range(R)
    ]
    paths = DataPaths(tmp_path / "data")
    directory = paths.oos_predictions / "elastic_net"
    directory.mkdir(parents=True)
    _predictions(rows).sample(frac=1.0, random_state=0).to_parquet(directory / "2004.parquet", index=False)
    (directory / "2004.json").write_text(json.dumps({"family": "elastic_net", "oos_year": 2004}))
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["data"]["sample"] = {"start": 200001, "end": 200512}
    config["walk_forward"] = {"first_oos_year": 2004, "validation_years": 2}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    argv = ["--data-dir", str(paths.root), "--config", str(config_path), "--family", "elastic_net", "--year", "2004"]
    return paths, argv, len(rows)


def test_output_schema_coverage_and_reproducible_completion(tmp_path):
    paths, argv, n_rows = _setup(tmp_path)
    outputs = []
    for _ in range(2):
        assert main(argv) == 0
        outputs.append(
            (
                pd.read_parquet(paths.family_signals / "elastic_net" / "2004.parquet"),
                json.loads((paths.family_signals / "elastic_net" / "2004.json").read_text()),
            )
        )
    signals, metadata = outputs[0]
    pd.testing.assert_frame_equal(signals, outputs[1][0])
    assert metadata == outputs[1][1]
    assert list(signals.columns) == list(SIGNAL_COLUMNS) and len(signals) == n_rows
    assert not signals.duplicated(["permno", "yyyymm", "family", "replica"]).any()
    assert signals.groupby(["permno", "yyyymm"])["replica"].apply(lambda r: sorted(r) == list(range(R))).all()
    assert signals.equals(signals.sort_values(["yyyymm", "permno", "replica"], ignore_index=True))
    ranks = signals[["replica_rank", "mean_replica_rank", "family_rank"]].to_numpy()
    assert np.isfinite(ranks).all() and ranks.min() >= 0.0 and ranks.max() <= 1.0
    assert (metadata["replicas"], metadata["n_prediction_rows"], metadata["n_stock_months"]) == (R, n_rows, n_rows // R)
    assert metadata["months"] == month_range(200401, 200412)
    assert not list((paths.family_signals / "elastic_net").glob("*.tmp"))


def test_incomplete_forecasts_are_rejected_and_leave_no_completion_marker(tmp_path, capsys):
    paths, argv, _ = _setup(tmp_path)
    assert main(argv) == 0
    source = paths.oos_predictions / "elastic_net" / "2004.parquet"
    predictions = pd.read_parquet(source)
    predictions.iloc[1:].to_parquet(source, index=False)  # one stock-month loses a replica
    assert main(argv) == 2
    assert "exactly replicas 0..4" in capsys.readouterr().err
    assert not (paths.family_signals / "elastic_net" / "2004.json").exists()  # not averaged over four

    (paths.oos_predictions / "elastic_net" / "2004.json").unlink()
    predictions.to_parquet(source, index=False)
    assert main(argv) == 2  # predictions without their completion marker
    assert "no complete OOS predictions" in capsys.readouterr().err
