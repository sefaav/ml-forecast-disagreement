"""Mechanical-bias diagnostics of the conditional disagreement (synthetic data)."""

import hashlib
import json
import os

import numpy as np
import pandas as pd
import yaml

from disagreement.data.config import DEFAULT_CONFIG, DataPaths
from disagreement.diagnostics import mechanical_bias
from disagreement.diagnostics.mechanical_bias import (
    BIN_SUMMARY_COLUMNS, MONTHLY_BIN_COLUMNS, MONTHLY_COLUMNS, RELATIONSHIPS, main, monthly_bin_diagnostics,
    monthly_correlations, with_extremity,
)
from disagreement.signals.conditional import SIGNAL_BINS, conditional_disagreement
from disagreement.walkforward.calendar import month_range

USED = ["permno", "yyyymm", "ensemble_signal", "signal_bin", "between_family_disagreement_raw",
        "conditional_between_family_disagreement"]


def test_diagnostic_calculations():
    rows = [  # permno, yyyymm, S, bin, raw, conditional
        (1, 200401, 0.05, 1, 0.02, 0.0), (2, 200401, 0.10, 1, 0.05, 1.0), (3, 200401, 0.02, 1, 0.01, 0.5),
        (4, 200401, 0.95, 20, 0.03, 0.5), (5, 200401, 0.99, 20, 0.01, 0.0), (6, 200401, 0.90, 20, 0.06, 1.0),
        # February: the conditional measure is constant, and bin 2 holds a single stock.
        (1, 200402, 0.20, 1, 0.10, 0.5), (2, 200402, 0.30, 1, 0.20, 0.5), (3, 200402, 0.60, 2, 0.30, 0.5),
    ]
    frame = with_extremity(pd.DataFrame(rows, columns=USED))
    np.testing.assert_array_equal(frame["centered_signal"], 2 * (frame["ensemble_signal"] - 0.5))
    np.testing.assert_array_equal(frame["signal_extremity"], np.abs(2 * (frame["ensemble_signal"] - 0.5)))

    def corr(x, y):
        return np.corrcoef(x, y)[0, 1]

    monthly = monthly_correlations(frame).set_index("yyyymm")
    for name in ("raw_vs_signal", "raw_vs_extremity"):
        d, s = RELATIONSHIPS[name]
        expected = [corr(group[d], group[s]) for _, group in frame.groupby("yyyymm")]  # one cross-section per month
        np.testing.assert_allclose(monthly[name].to_numpy(), expected, rtol=1e-12)
    jan = frame[frame["yyyymm"] == 200401]
    np.testing.assert_allclose(
        monthly.loc[[200401], "conditional_vs_extremity"].to_numpy(),
        corr(jan["conditional_between_family_disagreement"], jan["signal_extremity"]),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        monthly.loc[[200401], "conditional_vs_signal"].to_numpy(),
        corr(
            jan["conditional_between_family_disagreement"],
            jan["ensemble_signal"],
        ),
        rtol=1e-12,
    )
    assert np.isnan(monthly.loc[200402, ["conditional_vs_signal", "conditional_vs_extremity"]]).all()  # not 0
    assert monthly["n_stocks"].tolist() == [6, 3]

    bins = monthly_bin_diagnostics(frame).set_index(["yyyymm", "signal_bin"])
    assert bins.index.tolist() == [(200401, 1), (200401, 20), (200402, 1), (200402, 2)]  # observed bins only
    top = jan[jan["signal_bin"] == 20]  # computed inside the month-bin only
    expected = corr(top["between_family_disagreement_raw"], top["ensemble_signal"])
    np.testing.assert_allclose(bins.loc[(200401, 20), ["raw_vs_signal"]].to_numpy(dtype=float), expected, rtol=1e-12)
    means = ["mean_ensemble_signal", "mean_signal_extremity", "mean_raw_disagreement", "mean_conditional_disagreement"]
    np.testing.assert_allclose(
        bins.loc[(200401, 20), means].to_numpy(dtype=float),
        [np.mean([0.95, 0.99, 0.90]), np.mean([0.9, 0.98, 0.8]), np.mean([0.03, 0.01, 0.06]), 0.5],
        rtol=1e-12,
    )
    assert bins.loc[(200402, 1), "raw_vs_signal"] == 1.0  # two stocks
    assert np.isnan(bins.loc[(200402, 1), ["conditional_vs_signal"]].to_numpy(dtype=float)).all()  # constant in the bin
    assert np.isnan(bins.loc[(200402, 2), list(RELATIONSHIPS)].to_numpy(dtype=float)).all()  # a single stock


def _setup(tmp_path):
    """Conditional disagreement for the two baseline OOS years 2004 and 2005 of a synthetic config."""
    rng = np.random.default_rng(0)
    paths = DataPaths(tmp_path / "data")
    paths.conditional_disagreement.mkdir(parents=True)
    for year in (2004, 2005):
        keys = [(p, m) for m in month_range(year * 100 + 1, year * 100 + 12) for p in range(1, 61 + m % 5)]
        frame = pd.DataFrame(keys, columns=["permno", "yyyymm"])
        frame["ensemble_signal"] = rng.integers(0, 17, size=len(frame)) / 16  # tied signals
        frame["between_family_disagreement_raw"] = rng.uniform(0.0, 0.5, size=len(frame))
        frame["within_family_uncertainty_raw"] = rng.uniform(0.0, 1.0, size=len(frame))
        out = conditional_disagreement(frame, SIGNAL_BINS)
        out.sample(frac=1.0, random_state=0).to_parquet(paths.conditional_disagreement / f"{year}.parquet", index=False)
        (paths.conditional_disagreement / f"{year}.json").write_text(json.dumps({"oos_year": year}))
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["data"]["sample"] = {"start": 200001, "end": 200512}
    config["walk_forward"] = {"first_oos_year": 2004, "validation_years": 2}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return paths, ["--data-dir", str(paths.root), "--config", str(config_path)]


FILES = ["monthly_bin_diagnostics.csv", "monthly_correlations.csv", "signal_bin_summary.csv", "summary.json"]


def test_output_contract_and_reproducible_completion(tmp_path, monkeypatch, capsys):
    paths, argv = _setup(tmp_path)
    directory = paths.mechanical_bias_diagnostics
    replaced, replace = [], os.replace

    def recording_replace(src, dst):
        replaced.append(os.path.basename(dst))
        replace(src, dst)

    monkeypatch.setattr(mechanical_bias.os, "replace", recording_replace)
    outputs = []
    for _ in range(2):
        assert main(argv) == 0
        outputs.append({name: (directory / name).read_bytes() for name in FILES})
    assert outputs[0] == outputs[1]  # identical bytes
    assert sorted(p.name for p in directory.iterdir()) == FILES  # no stock-level file, no .tmp
    assert replaced[-1] == "summary.json" and replaced.count("summary.json") == 2  # the marker, written last
    printed = capsys.readouterr().out
    assert "Descriptive diagnostic only; no methodology was changed automatically." in printed

    monthly = pd.read_csv(directory / "monthly_correlations.csv")
    monthly_bins = pd.read_csv(directory / "monthly_bin_diagnostics.csv")
    bins = pd.read_csv(directory / "signal_bin_summary.csv")
    summary = json.loads((directory / "summary.json").read_text())
    assert list(monthly.columns) == list(MONTHLY_COLUMNS)
    assert list(monthly_bins.columns) == list(MONTHLY_BIN_COLUMNS)
    assert list(bins.columns) == list(BIN_SUMMARY_COLUMNS)
    months = month_range(200401, 200512)
    assert monthly["yyyymm"].tolist() == months  # one row per month, sorted

    source = pd.concat([pd.read_parquet(paths.conditional_disagreement / f"{y}.parquet") for y in (2004, 2005)])
    observed = source.groupby(["yyyymm", "signal_bin"]).size()
    assert list(zip(monthly_bins["yyyymm"], monthly_bins["signal_bin"])) == observed.index.tolist()  # sorted
    assert monthly_bins["n_stocks"].tolist() == observed.tolist()
    assert monthly["n_stocks"].tolist() == source.groupby("yyyymm").size().tolist()

    # The bin profile weighs every month-bin equally.
    for column in ("mean_ensemble_signal", "mean_raw_disagreement", "mean_conditional_disagreement"):
        np.testing.assert_allclose(bins[column], monthly_bins.groupby("signal_bin")[column].mean(), rtol=1e-12)
    np.testing.assert_array_equal(bins["n_months"], monthly_bins.groupby("signal_bin").size())
    np.testing.assert_array_equal(bins["total_stock_months"], observed.groupby("signal_bin").sum())
    for name in RELATIONSHIPS:
        finite = monthly_bins.groupby("signal_bin")[name].agg(lambda c: c[np.isfinite(c)].abs().mean())
        np.testing.assert_allclose(bins[f"{name}_mean_abs"], finite, rtol=1e-12)

        values = monthly[name].to_numpy()
        values = values[np.isfinite(values)]
        stats = summary["monthly_correlations"][name]
        assert stats["n_valid"] == len(values)
        expected = [values.mean(), np.median(values), np.abs(values).mean(), *np.quantile(values, [0.05, 0.95])]
        got = [stats[k] for k in ("mean_correlation", "median_correlation", "mean_absolute_correlation", "q05", "q95")]
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    assert (summary["first_month"], summary["last_month"], summary["months"]) == (200401, 200512, 24)
    assert (summary["n_stock_months"], summary["signal_bins"]) == (len(source), 20)
    assert summary["conditional_disagreement_sha256"] == {
        str(y): hashlib.sha256((paths.conditional_disagreement / f"{y}.parquet").read_bytes()).hexdigest()
        for y in (2004, 2005)
    }


def test_invalid_conditional_input_leaves_no_completion_marker(tmp_path, capsys):
    paths, argv = _setup(tmp_path)
    assert main(argv) == 0
    marker = paths.mechanical_bias_diagnostics / "summary.json"
    source = paths.conditional_disagreement / "2005.parquet"
    frame = pd.read_parquet(source)

    def rejected(message):
        assert main(argv) == 2
        assert message in capsys.readouterr().err
        assert not marker.exists()

    changed = frame.copy()
    changed.loc[0, "signal_bin"] = 1 if changed.loc[0, "signal_bin"] != 1 else 2
    changed.to_parquet(source, index=False)  # a stored bin no longer that of its ensemble signal
    rejected("2005 conditional disagreement: signal_bin differs from the bins of ensemble_signal in 1 rows")

    frame.drop(columns="conditional_within_family_uncertainty").to_parquet(source, index=False)
    rejected("2005 conditional disagreement: columns")

    frame.to_parquet(source, index=False)
    (paths.conditional_disagreement / "2005.json").unlink()
    rejected("2005: no complete conditional disagreement")
