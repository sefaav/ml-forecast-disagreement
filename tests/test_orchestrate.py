"""Orchestration of the frozen inference on synthetic yearly inputs. No real data is read or estimated."""

import numpy as np
import pandas as pd
import pytest
import yaml

from disagreement.controls.h2 import RANKS, h2_controls
from disagreement.data.config import DEFAULT_CONFIG, DataPaths
from disagreement.data.target import rank_target
from disagreement.inference import orchestrate
from disagreement.inference.coverage import outcome_coverage
from disagreement.inference.inputs import CD, CW, INFERENCE_COLUMNS, load_inference_year
from disagreement.inference.monthly import (
    SPECIFICATION_C,
    SPECIFICATION_C_H2,
    SPECIFICATION_H2,
    estimate_h1_month,
    estimate_month_specifications,
)
from disagreement.inference.orchestrate import (
    OUTSTANDING_ROBUSTNESS,
    RAW_SERIES,
    SUMMARY_COLUMNS,
    baseline_calendar,
    run_baseline_inference,
    run_inference,
)
from disagreement.inference.report import build_report
from disagreement.inference.series import SPECIFICATIONS, align, infer
from disagreement.signals.conditional import SIGNAL_BINS, conditional_disagreement
from disagreement.walkforward.calendar import month_range
from test_inference_inputs import _published
from test_series import _basis, _coefficients, _estimate

GRID = month_range(202311, 202402)  # crosses a year end: two partial synthetic years


def _frame(rng, months, n=1100, empty_month=None, no_complete_month=None):
    """Validated-looking inference inputs built with the pipeline's own functions; sizes differ across months."""
    keys = pd.DataFrame([(p, m) for i, m in enumerate(months) for p in range(1, n + 37 * (i % 3) + 1)],
                        columns=["permno", "yyyymm"])
    signals = keys.copy()
    signals["ensemble_signal"] = rng.integers(0, 401, size=len(keys)) / 400  # ties in S
    signals["between_family_disagreement_raw"] = np.round(rng.uniform(0.0, 0.5, size=len(keys)), 2)  # ties in CD
    signals["within_family_uncertainty_raw"] = rng.uniform(0.0, 1.0, size=len(keys))
    cond = conditional_disagreement(signals, SIGNAL_BINS)
    raw = keys.copy()
    raw["mktcap"] = rng.integers(1, 500, size=len(keys)) * 1000.0
    raw["RealizedVol"] = -rng.uniform(0.01, 0.1, size=len(keys))
    raw["Illiquidity"] = rng.uniform(0.0, 1.0, size=len(keys))
    raw.loc[rng.random(len(keys)) < 0.08, "Illiquidity"] = np.nan
    if no_complete_month is not None:
        raw.loc[raw["yyyymm"] == no_complete_month, "Illiquidity"] = np.nan
    raw["ensemble_signal"] = signals["ensemble_signal"]
    controls = h2_controls(raw)
    out = cond.loc[:, ["permno", "yyyymm", "ensemble_signal", "signal_bin", CD, CW]].copy()
    for column in ("centered_signal", *RANKS, "complete_case"):
        out[column] = controls[column].to_numpy()
    sc = out["centered_signal"].to_numpy()
    ret = 0.02 * sc - 0.01 * sc * out[CD].to_numpy() + 0.1 * rng.standard_normal(len(out))
    missing = rng.random(len(out)) < 0.05
    missing |= (out["size_rank"].to_numpy() > 0.6) & (rng.random(len(out)) < 0.3)  # missingness tied to size
    if empty_month is not None:
        missing |= out["yyyymm"].to_numpy() == empty_month
    out["ret_next"] = np.where(missing, np.nan, ret)
    out["target_rank"] = rank_target(out)
    return out.loc[:, list(INFERENCE_COLUMNS)]


def _decision(verdict):
    """A verdict, its conditions and the means and p-values it rests on (Wald arrays as tuples)."""
    basis = {k: None if v is None else (tuple(np.atleast_1d(v.mean)), v.p_value) for k, v in verdict.basis.items()}
    return verdict.verdict, verdict.conditions, basis


def _years(frame):
    return [(year, rows.reset_index(drop=True)) for year, rows in frame.groupby(frame["yyyymm"] // 100, sort=True)]


@pytest.fixture(scope="module")
def synthetic():
    """One real run on the four-month grid: a month without outcomes and a month without complete cases."""
    rng = np.random.default_rng(20261115)
    frame = _frame(rng, GRID, empty_month=202312, no_complete_month=202401)
    calls = []

    def recording(**kwargs):
        calls.append(kwargs)
        return estimate_month_specifications(**kwargs)

    raw_calls = []

    def recording_raw(*args):
        raw_calls.append(args)
        return estimate_h1_month(*args)

    run = run_inference(GRID, _years(frame), recording, recording_raw)
    return frame, run, calls, raw_calls


def test_validated_columns_are_mapped_to_the_estimator_in_the_frozen_order(synthetic):
    frame, run, calls, raw_calls = synthetic
    assert len(calls) == len(raw_calls) == 4
    for month, kwargs, raw_args in zip(GRID, calls, raw_calls):
        rows = frame[frame["yyyymm"] == month]
        assert set(kwargs) == {"outcome", "sc", "bins", "cd", "cw", "controls", "complete_case"}
        np.testing.assert_array_equal(kwargs["outcome"], rows["target_rank"].to_numpy())
        np.testing.assert_array_equal(kwargs["sc"], rows["centered_signal"].to_numpy())
        np.testing.assert_array_equal(kwargs["bins"], rows["signal_bin"].to_numpy())
        np.testing.assert_array_equal(kwargs["cd"], rows[CD].to_numpy())
        np.testing.assert_array_equal(kwargs["cw"], rows[CW].to_numpy())
        np.testing.assert_array_equal(kwargs["controls"], rows.loc[:, list(RANKS)].to_numpy())  # RANKS order
        assert kwargs["controls"].shape[1] == 3 and kwargs["complete_case"].dtype == bool
        np.testing.assert_array_equal(raw_args[0], rows["ret_next"].to_numpy())  # raw return, untouched
        np.testing.assert_array_equal(raw_args[1], rows["centered_signal"].to_numpy())


def test_end_to_end_results_equal_direct_calls_of_the_existing_components(synthetic):
    frame, run, _, _ = synthetic
    assert run.grid.tolist() == GRID == [202311, 202312, 202401, 202402]
    direct = {}
    for month in GRID:
        rows = frame[frame["yyyymm"] == month]
        direct[month] = estimate_month_specifications(
            rows["target_rank"].to_numpy(), rows["centered_signal"].to_numpy(), rows["signal_bin"].to_numpy(),
            rows[CD].to_numpy(), rows[CW].to_numpy(), rows.loc[:, list(RANKS)].to_numpy(),
            rows["complete_case"].to_numpy())
        assert set(run.aligned.estimates) == set(SPECIFICATIONS) and len(SPECIFICATIONS) == 5
    for spec in SPECIFICATIONS:
        for i, month in enumerate(GRID):
            got, want = run.aligned.estimates[spec][i], direct[month][spec]
            assert got.specification == spec and got.estimable == want.estimable and got.n_obs == want.n_obs
            if want.coefficients is not None:
                assert got.coefficients is not None
                assert got.coefficients.interaction == want.coefficients.interaction
    expected = build_report(align(GRID, direct), h2_control_means=run.h2_control_means)
    for name in ("h1", "h2", "delta", "economic", "h3"):
        assert getattr(run.report, name).verdict == getattr(expected, name).verdict
    for name, inference in expected.coefficients.items():
        assert run.report.coefficients[name].results == inference.results
    assert set(run.report.coefficients[f"{SPECIFICATION_C}.interaction"].results) == {6, 0, 12}


def test_months_without_outcomes_or_complete_cases_stay_on_the_calendar(synthetic):
    _, run, _, _ = synthetic
    # 202312: no finite outcome -> all five primary specifications and the raw-return C are non-estimable
    for spec in SPECIFICATIONS:
        assert not run.aligned.estimates[spec][1].estimable
        assert "H1 sample is empty" in run.report.non_estimable[spec][202312][0] or (
            "H2 sample is empty" in run.report.non_estimable[spec][202312][0])
    assert 202312 in run.raw_return.non_estimable and run.raw_return.sample_sizes[1] == 0
    # 202401: no complete case -> H2 and C|H2 non-estimable, C, B and H3 estimable
    assert run.aligned.mask(SPECIFICATION_C).tolist() == [True, False, True, True]
    assert run.aligned.mask(SPECIFICATION_H2).tolist() == [True, False, False, True]
    assert run.aligned.mask(SPECIFICATION_C_H2).tolist() == [True, False, False, True]
    assert run.report.coefficients["H2.interaction"].series.missing_months == [202312, 202401]
    summary = run.monthly_summary.set_index("yyyymm")
    assert list(run.monthly_summary.columns) == list(SUMMARY_COLUMNS)
    assert summary.loc[202312, "n_finite_outcome"] == 0 and summary.loc[202312, "finite_outcome_share"] == 0.0
    assert summary.loc[202312, "n_stocks"] > 0 and summary.loc[202401, "n_h2_sample"] == 0
    december = run.coverage_monthly[run.coverage_monthly["yyyymm"] == 202312]
    assert len(december) == SIGNAL_BINS * 3 and (december["finite_share"] == 0.0).all()
    assert december["n_stocks"].sum() == summary.loc[202312, "n_stocks"]


def test_h2_control_means_use_the_h2_estimation_sample_only(synthetic):
    frame, run, _, _ = synthetic
    assert set(run.h2_control_means) == {202311, 202402}  # no invented mean for an empty H2 sample
    for month in (202311, 202402):
        rows = frame[frame["yyyymm"] == month]
        h2 = rows["target_rank"].notna() & rows["complete_case"]
        expected = rows.loc[h2, list(RANKS)].mean().to_numpy()
        np.testing.assert_allclose(run.h2_control_means[month], expected, rtol=1e-12)
        all_complete = rows.loc[rows["complete_case"], list(RANKS)].mean().to_numpy()
        assert abs(run.h2_control_means[month][0] - all_complete[0]) > 1e-3  # outcome missingness depends on size
    # the descriptive means change only the descriptive H2 slopes, never the primary ones or a verdict
    aligned = run.aligned
    without = build_report(aligned)
    for c in (0.1, 0.5, 0.9):
        assert run.report.slopes[f"H2.g({c})"].results == without.slopes[f"H2.g({c})"].results
        assert f"H2.g|controls_at_means({c})" in run.report.slopes
        assert f"H2.g|controls_at_means({c})" not in without.slopes
    for name in ("h1", "h2", "delta", "economic", "h3"):
        assert _decision(getattr(run.report, name)) == _decision(getattr(without, name))


def test_coverage_profile_is_the_equal_month_weight_profile_of_the_combined_months(synthetic):
    frame, run, _, _ = synthetic
    finite = frame["target_rank"].notna()
    monthly, profile = outcome_coverage(frame["yyyymm"], frame["signal_bin"], frame[CD], finite)
    pd.testing.assert_frame_equal(run.coverage_monthly, monthly)
    pd.testing.assert_frame_equal(run.coverage_profile, profile)
    cell = monthly[(monthly["signal_bin"] == 7) & (monthly["cd_group"] == 2)]
    assert cell["n_stocks"].nunique() > 1  # unequal cell sizes across months
    row = profile[(profile["signal_bin"] == 7) & (profile["cd_group"] == 2)].iloc[0]
    assert row["mean_finite_share"] == pytest.approx(cell["finite_share"].mean())  # months weigh equally ...
    pooled = cell["n_finite"].sum() / cell["n_stocks"].sum()
    assert row["mean_finite_share"] != pytest.approx(pooled, abs=1e-6)  # ... not stocks
    assert row["n_months"] == 4


def test_raw_return_robustness_uses_the_existing_c_estimator_and_never_changes_the_primary_report(synthetic):
    frame, run, _, _ = synthetic
    raw = run.raw_return
    assert len(raw.estimates) == 4 and all(e.specification == SPECIFICATION_C for e in raw.estimates)
    for month, estimate in zip(GRID, raw.estimates):
        rows = frame[frame["yyyymm"] == month]
        want = estimate_h1_month(rows["ret_next"].to_numpy(), rows["centered_signal"].to_numpy(),
                                 rows["signal_bin"].to_numpy(), rows[CD].to_numpy())
        assert estimate.estimable == want.estimable and estimate.n_obs == want.n_obs
        if want.coefficients is not None:
            assert estimate.coefficients is not None
            assert estimate.coefficients.interaction == want.coefficients.interaction
    assert raw.interaction.series.name == RAW_SERIES and raw.interaction.series.missing_months == [202312]
    assert raw.interaction.results == infer(raw.interaction.series).results
    # an altered raw return changes the robustness series only
    altered = frame.assign(ret_next=-5.0 * frame["ret_next"])
    other = run_inference(GRID, _years(altered))
    assert other.raw_return.interaction.series.values[0] != raw.interaction.series.values[0]
    for name in ("h1", "h2", "delta", "economic", "h3"):
        assert _decision(getattr(other.report, name)) == _decision(getattr(run.report, name))
    assert other.report.coefficients["C.interaction"].results == run.report.coefficients["C.interaction"].results


def _tiny(months):
    """Four stocks per month: enough for the wiring checks with stub estimators, far too few for a regression."""
    rows = []
    for m in months:
        for p in range(1, 5):
            rows.append((p, m, p / 5, 1 + (p - 1) * 5, 2 * (p / 5 - 0.5), p / 5, 1 - p / 5,
                         0.1 * p, -0.1 * p, 0.05 * p, True, 2 * (p - 1) / 3 - 1, 0.01 * p))
    return pd.DataFrame(rows, columns=list(INFERENCE_COLUMNS))


def _stub_estimators(events):
    rng = np.random.default_rng(3)
    basis = _basis(rng)

    def estimate(**kwargs):
        events.append(("estimate", len(kwargs["outcome"])))
        return {spec: _estimate(spec, True, basis, _coefficients(rng, spec, rng.normal(-0.01, 0.01),
                                                                 rng.normal(0.0, 0.01)))
                for spec in SPECIFICATIONS}

    def estimate_raw(*args):
        return _estimate(SPECIFICATION_C, True, basis, _coefficients(rng, SPECIFICATION_C, rng.normal(), 0.0))

    return estimate, estimate_raw


def test_calendar_errors_fail_closed_instead_of_skipping():
    grid = month_range(202311, 202402)
    first, second = _tiny(month_range(202311, 202312)), _tiny(month_range(202401, 202402))
    estimate, estimate_raw = _stub_estimators([])

    def rejected(message, years, calendar=grid):
        with pytest.raises(ValueError, match=message):
            run_inference(calendar, years, estimate, estimate_raw)

    rejected(r"no input frame for the grid years \[2024\]", [(2023, first)])
    rejected("not the next calendar year", [(2023, first), (2023, first)])
    rejected("not the next calendar year", [(2024, second), (2023, first)])
    rejected("not the next calendar year", [(2023, first), (2024, second), (2025, second)])
    rejected(r"input months \[202311\] differ from the grid months \[202311, 202312\]",
             [(2023, first[first["yyyymm"] == 202311]), (2024, second)])
    rejected("differ from the grid months", [(2023, _tiny([202310, 202311, 202312])), (2024, second)])
    rejected("exactly the validated inference columns", [(2023, first.drop(columns=CW)), (2024, second)])
    rejected("consecutive yyyymm months", [(2023, first)], calendar=[202311, 202401])
    rejected("non-empty", [], calendar=[])
    # calendar values are never truncated or parsed: fractional, string and invalid months, fractional years
    rejected("202311.9 is not an integer yyyymm month", [(2023, first), (2024, second)],
             calendar=[202311.9, 202312, 202401, 202402])
    rejected("'202311' is not an integer yyyymm month", [(2023, first), (2024, second)],
             calendar=["202311", 202312, 202401, 202402])
    rejected("202313 is not a valid yyyymm month", [(2023, first), (2024, second)],
             calendar=[202312, 202313, 202401])
    rejected("202300 is not a valid yyyymm month", [(2023, first)], calendar=[202300, 202301])
    rejected("2023.7 is not an integer OOS year", [(2023.7, first), (2024, second)])
    rejected("'2023' is not an integer OOS year", [("2023", first), (2024, second)])
    rejected("True is not an integer OOS year", [(True, first), (2024, second)])
    out = run_inference(grid, [(2023, first), (2024, second)], estimate, estimate_raw)
    assert out.grid.tolist() == grid
    numpy_years = [(np.int64(2023), first), (np.int32(2024), second)]  # NumPy integers are valid years
    numpy_grid = np.asarray(grid, dtype="int64")
    numpy = run_inference(numpy_grid, numpy_years, estimate, estimate_raw)  # type: ignore[arg-type]
    assert numpy.grid.tolist() == grid


def test_the_production_wrapper_runs_the_real_loader_on_a_published_synthetic_year(tmp_path):
    paths, target = _published(tmp_path)  # universe, panel, conditional disagreement, H2 controls and target
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["data"]["sample"] = {"start": 200001, "end": 200412}  # a shortened test calendar: OOS year 2004 only
    config["walk_forward"] = {"first_oos_year": 2004, "validation_years": 2}
    config_path = tmp_path / "short.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert baseline_calendar(config_path) == (month_range(200401, 200412), [2004])
    events, received, raw_received = [], [], []
    estimate, estimate_raw = _stub_estimators(events)

    def recording(**kwargs):
        received.append(kwargs)
        return estimate(**kwargs)

    def recording_raw(*args):
        raw_received.append(args)
        return estimate_raw(*args)

    run = run_baseline_inference(paths, config_path, estimate=recording, estimate_raw=recording_raw)
    validated = load_inference_year(paths, 2004)
    assert run.grid.tolist() == month_range(200401, 200412) and len(received) == len(raw_received) == 12
    for month, kwargs, raw_args in zip(month_range(200401, 200412), received, raw_received):
        rows = validated[validated["yyyymm"] == month]
        np.testing.assert_array_equal(kwargs["outcome"], rows["target_rank"].to_numpy())
        np.testing.assert_array_equal(kwargs["sc"], rows["centered_signal"].to_numpy())
        np.testing.assert_array_equal(kwargs["bins"], rows["signal_bin"].to_numpy())
        np.testing.assert_array_equal(kwargs["cd"], rows[CD].to_numpy())
        np.testing.assert_array_equal(kwargs["cw"], rows[CW].to_numpy())
        np.testing.assert_array_equal(kwargs["controls"], rows.loc[:, list(RANKS)].to_numpy())
        np.testing.assert_array_equal(kwargs["complete_case"], rows["complete_case"].to_numpy())
        np.testing.assert_array_equal(raw_args[0], rows["ret_next"].to_numpy())
    summary = run.monthly_summary.set_index("yyyymm")
    assert summary.index.tolist() == month_range(200401, 200412)
    assert summary["n_stocks"].sum() == len(validated) == len(target)
    assert summary["n_finite_outcome"].sum() == validated["target_rank"].notna().sum()
    h2 = validated["target_rank"].notna() & validated["complete_case"]
    assert summary["n_h2_sample"].sum() == h2.sum()
    assert run.coverage_monthly["n_stocks"].sum() == len(validated)
    assert run.report.coefficients["C.interaction"].results[6].n_grid == 12
    assert set(run.report.non_estimable) == set(SPECIFICATIONS)
    assert len(run.raw_return.estimates) == 12


def test_the_baseline_calendar_is_the_frozen_480_months_loaded_one_year_at_a_time(tmp_path, monkeypatch):
    grid, years = baseline_calendar()
    assert len(grid) == 480 and len(years) == 40 and grid[0] == 198501 and grid[-1] == 202412
    assert years == list(range(1985, 2025)) and grid == month_range(198501, 202412)
    events = []

    def loader(paths, year):  # stands in for load_inference_year: no file is read
        events.append(("load", year))
        return _tiny(month_range(year * 100 + 1, year * 100 + 12))

    monkeypatch.setattr(orchestrate, "load_inference_year", loader)
    estimate, estimate_raw = _stub_estimators(events)
    run = run_baseline_inference(DataPaths(tmp_path / "data"), estimate=estimate, estimate_raw=estimate_raw)
    assert run.grid.tolist() == grid and len(run.monthly_summary) == 480
    assert [e[1] for e in events if e[0] == "load"] == years
    # incremental: each year is loaded only after the twelve months of the previous year were estimated
    loads = [i for i, e in enumerate(events) if e[0] == "load"]
    assert all(sum(1 for e in events[a:b] if e[0] == "estimate") == 12 for a, b in zip(loads, loads[1:]))
    assert run.report.coefficients["C.interaction"].results[6].n_grid == 480
    assert len(run.raw_return.estimates) == 480 and len(run.h2_control_means) == 480
    assert not any(hasattr(run, name) for name in ("frames", "inputs", "stock_months"))
    assert len(OUTSTANDING_ROBUSTNESS) == 5 and "excluding the twelve 2021 OOS months" in OUTSTANDING_ROBUSTNESS
