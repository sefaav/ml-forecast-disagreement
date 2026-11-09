"""Conditional disagreement and uncertainty within monthly signal bins (synthetic data)."""

import hashlib
import json

import numpy as np
import pandas as pd
import yaml

from disagreement.data.config import DEFAULT_CONFIG, DataPaths
from disagreement.models.selection import FAMILIES
from disagreement.signals import within
from disagreement.signals.between import between_family_disagreement
from disagreement.signals.conditional import (
    OUTPUT_COLUMNS, SIGNAL_BINS, conditional_disagreement, main, signal_bins,
)
from disagreement.walkforward.calendar import month_range


def test_signal_bins_and_conditional_ranks():
    rows = [  # permno, yyyymm, ensemble_signal, between raw, within raw
        # 200401, unique signals: ranks 8 1 5 3 7 2 6 4, so with 4 bins of two stocks: 4 1 3 2 4 1 3 2
        (1, 200401, 0.8, 0.10, 0.9), (2, 200401, 0.1, 0.30, 0.1), (3, 200401, 0.5, 0.20, 0.3),
        (4, 200401, 0.3, 0.05, 0.5), (5, 200401, 0.7, 0.40, 0.2), (6, 200401, 0.2, 0.30, 0.6),
        (7, 200401, 0.6, 0.10, 0.7), (8, 200401, 0.4, 0.45, 0.5),
        # 200402, three tied signals: average rank 3 of n = 5 puts all three in bin 1 + floor(4 * 2 / 5) = 2
        (1, 200402, 0.3, 0.2, 0.5), (2, 200402, 0.3, 0.2, 0.5), (3, 200402, 0.3, 0.4, 0.5),
        (4, 200402, 0.9, 0.1, 0.1), (5, 200402, 0.1, 0.3, 0.3),
    ]
    raw = ["between_family_disagreement_raw", "within_family_uncertainty_raw"]
    frame = pd.DataFrame(rows, columns=["permno", "yyyymm", "ensemble_signal", *raw])
    shuffled = frame.sample(frac=1.0, random_state=3)  # row order must not matter
    out = conditional_disagreement(shuffled, bins=4)
    pd.testing.assert_frame_equal(out, conditional_disagreement(frame, bins=4))
    jan, feb = (out[out["yyyymm"] == m].set_index("permno") for m in (200401, 200402))

    assert jan["signal_bin"].tolist() == [4, 1, 3, 2, 4, 1, 3, 2]  # from the signal alone; 1 lowest, 4 highest
    assert feb["signal_bin"].tolist() == [2, 2, 2, 4, 1]  # ties share a bin; first-ranks would split them 1, 2, 3
    # Ranked within (month, bin) only: pairs {2, 6}, {4, 8}, {3, 7}, {1, 5}; a tie gives 0.5 each.
    assert jan["conditional_between_family_disagreement"].tolist() == [0.0, 0.5, 1.0, 0.0, 1.0, 0.5, 0.0, 1.0]
    assert jan["conditional_within_family_uncertainty"].tolist() == [1.0, 0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5]
    # Bin 2 of February: average ranks 1.5, 1.5, 3 of 3; bins 4 and 1 hold one stock each (0.5).
    assert feb["conditional_between_family_disagreement"].tolist() == [0.25, 0.25, 1.0, 0.5, 0.5]
    assert (feb["conditional_within_family_uncertainty"] == 0.5).all()  # a constant bin

    rng = np.random.default_rng(0)
    for n in (100, 43):  # unique signals: bins as even as possible
        counts = signal_bins(pd.Series(rng.permutation(n) / n), pd.Series([200401] * n), SIGNAL_BINS).value_counts()
        assert sorted(counts.index) == list(range(1, 21)) and counts.max() - counts.min() <= 1
    assert SIGNAL_BINS == 20


def _setup(tmp_path):
    """Ensemble, between-family and within-family files for OOS year 2004, with a config whose episodes include it."""
    rng = np.random.default_rng(0)
    keys = pd.DataFrame(
        [(p, m) for m in month_range(200401, 200412) for p in range(1, 41 + m % 7)], columns=["permno", "yyyymm"]
    )
    ensemble = keys.copy()
    for family in FAMILIES:
        ensemble[f"{family}_rank"] = rng.integers(0, 5, size=len(keys)) / 4  # many tied signals
    ensemble["ensemble_signal"] = (
        ensemble["elastic_net_rank"] + ensemble["random_forest_rank"] + ensemble["lightgbm_rank"] + ensemble["mlp_rank"]
    ) / 4.0
    uncertainty = keys.copy()
    for family in FAMILIES:
        uncertainty[f"{family}_uncertainty_sd"] = rng.integers(0, 4, size=len(keys)) / 8
    for family in FAMILIES:
        uncertainty[f"{family}_uncertainty_rank"] = rng.integers(0, 5, size=len(keys)) / 4
    ranks = uncertainty[[f"{f}_uncertainty_rank" for f in FAMILIES]].to_numpy()
    uncertainty["within_family_uncertainty_raw"] = (ranks[:, 0] + ranks[:, 1] + ranks[:, 2] + ranks[:, 3]) / 4.0

    paths = DataPaths(tmp_path / "data")
    sources = {
        paths.ensemble_signals: ensemble,
        paths.between_family_disagreement: between_family_disagreement(ensemble),
        paths.within_family_uncertainty: uncertainty[list(within.OUTPUT_COLUMNS)],
    }
    for directory, frame in sources.items():
        directory.mkdir(parents=True)
        frame.sample(frac=1.0, random_state=0).to_parquet(directory / "2004.parquet", index=False)
        (directory / "2004.json").write_text(json.dumps({"oos_year": 2004}))
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["data"]["sample"] = {"start": 200001, "end": 200512}
    config["walk_forward"] = {"first_oos_year": 2004, "validation_years": 2}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    argv = ["--data-dir", str(paths.root), "--config", str(config_path), "--year", "2004"]
    return paths, argv


def _unit_rank(values, groups):
    rank = values.groupby(groups).rank(method="average")
    n = values.groupby(groups).transform("count")
    return ((rank - 1) / (n - 1)).where(n > 1, 0.5)


def test_output_schema_coverage_and_reproducible_completion(tmp_path):
    paths, argv = _setup(tmp_path)
    directory = paths.conditional_disagreement
    outputs = []
    for _ in range(2):
        assert main(argv) == 0
        outputs.append((pd.read_parquet(directory / "2004.parquet"), json.loads((directory / "2004.json").read_text())))
    out, metadata = outputs[0]
    pd.testing.assert_frame_equal(out, outputs[1][0])
    assert metadata == outputs[1][1]
    assert list(out.columns) == list(OUTPUT_COLUMNS)
    assert not out.duplicated(["permno", "yyyymm"]).any()
    assert out.equals(out.sort_values(["yyyymm", "permno"], ignore_index=True))
    assert sorted(out["yyyymm"].unique().tolist()) == month_range(200401, 200412)

    sources = [
        pd.read_parquet(d / "2004.parquet").sort_values(["yyyymm", "permno"], ignore_index=True)
        for d in (paths.ensemble_signals, paths.between_family_disagreement, paths.within_family_uncertainty)
    ]
    for source in sources:  # exactly the same stock-months as every source
        pd.testing.assert_frame_equal(out[["permno", "yyyymm"]], source[["permno", "yyyymm"]])
    ensemble, disagreement, uncertainty = sources
    np.testing.assert_array_equal(out["ensemble_signal"].to_numpy(), ensemble["ensemble_signal"].to_numpy())
    np.testing.assert_array_equal(
        out["between_family_disagreement_raw"].to_numpy(), disagreement["between_family_disagreement_raw"].to_numpy()
    )
    np.testing.assert_array_equal(
        out["within_family_uncertainty_raw"].to_numpy(), uncertainty["within_family_uncertainty_raw"].to_numpy()
    )

    # Recomputed independently: 20 monthly bins from average signal ranks, then ranks within (month, bin).
    by_month = ensemble["ensemble_signal"].groupby(ensemble["yyyymm"])
    rank, n = by_month.rank(method="average"), by_month.transform("count")
    bins = 1 + np.floor(20 * (rank - 1) / n).astype("int64")
    assert out["signal_bin"].dtype == np.int64 and out["signal_bin"].between(1, 20).all()
    np.testing.assert_array_equal(out["signal_bin"].to_numpy(), np.asarray(bins))
    groups = [out["yyyymm"], out["signal_bin"]]
    for raw, conditional in (
        ("between_family_disagreement_raw", "conditional_between_family_disagreement"),
        ("within_family_uncertainty_raw", "conditional_within_family_uncertainty"),
    ):
        np.testing.assert_array_equal(out[conditional].to_numpy(), _unit_rank(out[raw], groups).to_numpy())
        values = out[conditional].to_numpy()
        assert np.isfinite(values).all() and values.min() >= 0.0 and values.max() <= 1.0

    sha = {d: hashlib.sha256((d / "2004.parquet").read_bytes()).hexdigest() for d in (
        paths.ensemble_signals, paths.between_family_disagreement, paths.within_family_uncertainty
    )}
    assert metadata == {
        "oos_year": 2004,
        "signal_bins": 20,
        "n_stock_months": len(ensemble),
        "months": month_range(200401, 200412),
        "ensemble_signals_sha256": sha[paths.ensemble_signals],
        "between_family_disagreement_sha256": sha[paths.between_family_disagreement],
        "within_family_uncertainty_sha256": sha[paths.within_family_uncertainty],
    }
    assert not list(directory.glob("*.tmp"))


def test_invalid_sources_leave_no_completion_marker(tmp_path, capsys):
    paths, argv = _setup(tmp_path)
    assert main(argv) == 0
    marker = paths.conditional_disagreement / "2004.json"
    disagreement_path = paths.between_family_disagreement / "2004.parquet"
    uncertainty_path = paths.within_family_uncertainty / "2004.parquet"
    disagreement, uncertainty = pd.read_parquet(disagreement_path), pd.read_parquet(uncertainty_path)

    def rejected(message):
        assert main(argv) == 2
        assert message in capsys.readouterr().err
        assert not marker.exists()

    paths.mechanical_bias_diagnostics.mkdir(parents=True)
    (paths.mechanical_bias_diagnostics / "summary.json").write_text("diagnostic of an earlier run")
    disagreement.iloc[1:].to_parquet(disagreement_path, index=False)  # one stock-month missing
    rejected("2004 between-family disagreement covers different stock-months than the ensemble: 1 missing, 0 extra")
    assert not paths.mechanical_bias_diagnostics.exists()  # stale as soon as a conditional year is rerun

    changed = disagreement.copy()
    changed.loc[0, "ensemble_signal"] = 0.5 if changed.loc[0, "ensemble_signal"] != 0.5 else 0.25
    changed.to_parquet(disagreement_path, index=False)  # its saved signal no longer the ensemble's
    rejected("2004 between-family disagreement: ensemble_signal differs in 1 rows")
    disagreement.to_parquet(disagreement_path, index=False)

    uncertainty.drop(columns="within_family_uncertainty_raw").to_parquet(uncertainty_path, index=False)
    rejected("2004 within-family uncertainty: columns")

    uncertainty.to_parquet(uncertainty_path, index=False)
    (paths.within_family_uncertainty / "2004.json").unlink()
    rejected("no complete within-family uncertainty")


def test_regenerating_a_year_removes_only_that_years_h2_controls(tmp_path):
    paths, argv = _setup(tmp_path)
    assert main(argv) == 0  # no H2 controls exist yet: the invalidation must be a no-op
    assert not paths.h2_controls.exists()

    suffixes = ("parquet", "json", "parquet.tmp", "json.tmp")
    stale = [paths.h2_controls / f"2004.{suffix}" for suffix in suffixes]
    unrelated = [paths.h2_controls / f"2005.{suffix}" for suffix in ("parquet", "json")]
    for path in stale + unrelated:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("controls built from an earlier conditional disagreement")
    assert main(argv) == 0  # direct regeneration of the 2004 conditional disagreement
    assert not any(path.exists() for path in stale)  # files and completion marker of 2004
    assert all(path.read_text() == "controls built from an earlier conditional disagreement" for path in unrelated)
