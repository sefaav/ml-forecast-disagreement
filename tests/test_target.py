"""Next-month return and ranked training target (synthetic data)."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from disagreement.data import crsp
from disagreement.data.config import DataPaths, load_data_config
from disagreement.data.target import TARGET_COLUMNS, attach_next_return, build_targets, rank_target
from disagreement.data.validation import DataValidationError
from synthetic import crsp_row


def _universe(keys):
    return pd.DataFrame(keys, columns=["permno", "yyyymm"])


def test_return_comes_from_the_same_permno_in_exactly_t_plus_1():
    raw = pd.DataFrame(
        [
            crsp_row(1, "2000-01", 10.0, mthret=0.50),  # month t
            crsp_row(1, "2000-02", 10.0, mthret=0.02),  # t+1
            crsp_row(1, "2000-03", 10.0, mthret=0.90),  # t+2
            crsp_row(2, "2000-02", 10.0, mthret=0.70),  # another PERMNO in t+1
            crsp_row(3, "2001-01", 10.0, mthret=-0.10),  # t+1 across the year boundary
        ]
    )
    out = attach_next_return(_universe([(1, 200001), (3, 200012)]), raw)
    assert out["ret_next"].tolist() == [0.02, -0.10]
    assert list(out.columns) == ["permno", "yyyymm", "ret_next"]


def test_delisting_in_t_plus_1_keeps_the_observation_and_its_return():
    raw = pd.DataFrame(
        [
            crsp_row(1, "2000-02", 0.5, exch=None, mthret=-0.85, mthretflg="DE", mthdelflg="A", tradingstatusflg="S"),
            crsp_row(2, "2000-02", 0.5, exch="Q", mthret=-0.40, mthdelflg="G", sharetype="AD"),
        ]
    )
    out = attach_next_return(_universe([(1, 200001), (2, 200001)]), raw)
    assert out["ret_next"].tolist() == [-0.85, -0.40]  # t+1 status, exchange and type play no role


def test_missing_next_month_returns_stay_missing():
    raw = pd.DataFrame(
        [
            crsp_row(2, "2000-02", 10.0, mthret=None),
            crsp_row(3, "2000-02", 10.0, mthret=np.inf),
            crsp_row(4, "2000-03", 10.0, mthret=0.05),  # t+2 only
        ]
    )
    out = attach_next_return(_universe([(1, 200001), (2, 200001), (3, 200001), (4, 200001)]), raw)
    assert out["ret_next"].isna().all()
    out["target_rank"] = rank_target(out)
    assert out["target_rank"].isna().all()


def test_target_ranks_within_each_formation_month_with_average_ties():
    frame = pd.DataFrame(
        {
            "permno": [1, 2, 3, 4, 5, 1, 2, 3, 9],
            "yyyymm": [200001] * 5 + [200002] * 3 + [200003],
            "ret_next": [0.05, -0.02, 0.01, 0.01, np.nan, 0.30, 0.20, 0.10, 0.07],
        }
    )
    target = rank_target(frame).tolist()
    # 200001: ranks 4, 1, 2.5, 2.5 over n=4 -> 2*(r-1)/3 - 1
    assert target[:4] == pytest.approx([1.0, -1.0, 0.0, 0.0])
    assert np.isnan(target[4])  # missing return, missing target
    assert target[5:8] == pytest.approx([1.0, 0.0, -1.0])  # ranked apart from 200001
    assert target[8] == 0.0  # n = 1


def test_keys_stay_unique():
    raw = pd.DataFrame([crsp_row(1, "2000-02", 10.0), crsp_row(2, "2000-02", 10.0)])
    out = attach_next_return(_universe([(1, 200001), (2, 200001)]), raw)
    assert not out.duplicated(["permno", "yyyymm"]).any()
    with pytest.raises(DataValidationError, match="universe"):
        attach_next_return(_universe([(1, 200001), (1, 200001)]), raw)
    with pytest.raises(DataValidationError, match="CRSP returns"):
        attach_next_return(_universe([(1, 200001)]), pd.concat([raw, raw.iloc[[0]]]))


def _target_inputs(tmp_path):
    """Universe of a successful build and unwritten CRSP rows for 2000 and 2001."""
    cfg = dataclasses.replace(load_data_config(), sample_start=200011, sample_end=200012)
    paths = DataPaths(tmp_path / "data")
    paths.universe.mkdir(parents=True)
    paths.diagnostics.mkdir(parents=True)
    (paths.diagnostics / "summary.json").write_text("{}")
    _universe([(1, 200010), (1, 200011), (2, 200011), (1, 200012), (2, 200012)]).to_parquet(
        paths.universe / "universe_2000.parquet", index=False
    )
    crsp_dir = paths.crsp(cfg.crsp_table)
    crsp_dir.mkdir(parents=True)
    rows = {
        2000: [crsp_row(1, "2000-11", 10.0, mthret=0.03), crsp_row(1, "2000-12", 10.0, mthret=0.04),
               crsp_row(2, "2000-12", 10.0, mthret=-0.01)],
        2001: [crsp_row(1, "2001-01", 10.0, mthret=0.08)],  # permno 2 delisted: no January record
    }
    return cfg, paths, crsp_dir, rows


def _write_crsp(crsp_dir, records, year):
    crsp.coerce_raw_types(pd.DataFrame(records)).to_parquet(crsp_dir / f"crsp_monthly_{year}.parquet", index=False)


def test_build_targets_reads_t_plus_1_from_the_next_year_file(tmp_path):
    cfg, paths, crsp_dir, rows = _target_inputs(tmp_path)
    with pytest.raises(DataValidationError, match="crsp_monthly_2001"):
        build_targets(paths, cfg)
    for year, records in rows.items():
        _write_crsp(crsp_dir, records, year)

    summary = build_targets(paths, cfg)
    out = pd.read_parquet(paths.target / "target_2000.parquet")
    assert list(out.columns) == list(TARGET_COLUMNS)
    assert out[["permno", "yyyymm"]].values.tolist() == [[1, 200011], [2, 200011], [1, 200012], [2, 200012]]
    assert out["ret_next"].tolist()[:3] == [0.04, -0.01, 0.08] and np.isnan(out["ret_next"].iloc[3])
    assert out["target_rank"].tolist()[:3] == [1.0, -1.0, 0.0] and np.isnan(out["target_rank"].iloc[3])
    assert (summary["first_month"], summary["last_month"], summary["last_return_month"]) == (200011, 200012, 200101)


def test_failed_target_build_keeps_the_previous_target(tmp_path):
    cfg, paths, crsp_dir, rows = _target_inputs(tmp_path)
    for year, records in rows.items():
        _write_crsp(crsp_dir, records, year)
    build_targets(paths, cfg)
    before = (paths.target / "target_2000.parquet").read_bytes()
    downstream = [paths.oos_predictions / "elastic_net" / "2001.json", paths.family_signals / "elastic_net" / "2001.json"]
    for output in downstream:  # made from these targets
        output.parent.mkdir(parents=True)
        output.write_text("from the current targets")

    _write_crsp(crsp_dir, rows[2001] * 2, 2001)  # duplicate key: fails while building
    with pytest.raises(DataValidationError, match="CRSP returns"):
        build_targets(paths, cfg)
    assert [f.name for f in paths.target.iterdir()] == ["target_2000.parquet"]
    assert (paths.target / "target_2000.parquet").read_bytes() == before
    assert all(output.read_text() == "from the current targets" for output in downstream)  # still valid: kept
    assert sorted(d.name for d in paths.target.parent.iterdir()) == [
        "family_signals", "oos_predictions", "target", "universe",
    ]

    _write_crsp(crsp_dir, [crsp_row(1, "2001-01", 10.0, mthret=0.09)], 2001)
    build_targets(paths, cfg)  # a successful build replaces it
    assert pd.read_parquet(paths.target / "target_2000.parquet")["ret_next"].iloc[2] == 0.09
    assert sorted(d.name for d in paths.target.parent.iterdir()) == ["target", "universe"]  # stale forecasts removed
