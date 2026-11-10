"""Design-evidence diagnostics: forecast-only reproductions and synthetic-outcome reruns (synthetic data)."""

import hashlib
import json
import os

import numpy as np
import pandas as pd
import pytest
import yaml

from disagreement.controls.h2 import run_h2_controls
from disagreement.data.config import DEFAULT_CONFIG, DataPaths
from disagreement.data.validation import DataValidationError
from disagreement.diagnostics import design_evidence as de
from disagreement.diagnostics.design_evidence import (
    CD, CW, EXPECTED_DETERMINISTIC_CHECKS, RAW_CD, RAW_CW, RESOLVED, SCENARIOS, SIGNAL_BINS, SPECIFICATIONS,
    bartlett_hac_se, bin_constants, cell_statistics, cells_of, classify_failures, constant_family_months,
    extreme_cell_diagnostics, fama_macbeth, fwl_weights, interaction_coefficient, local_basis, main, month_designs,
    monte_carlo, natural_spline_basis, original_basis, residualise, scaled_condition, scenario_means,
    three_family_disagreement, two_step_coefficient, within_bin_table,
)
from disagreement.models.selection import FAMILIES
from disagreement.signals.conditional import conditional_disagreement, signal_bins
from disagreement.signals.family import remove_conditional_disagreement, unit_rank
from disagreement.walkforward.calendar import month_range

FILES = [
    "algebraic_identities_monthly.csv", "extreme_bin_months.csv", "geometry_bins.csv", "monte_carlo.csv",
    "specification_monthly.csv", "specification_summary.csv", "summary.json",
]


# ---------------------------------------------------------------------------------------------------- cell statistics


def test_cells_and_hand_calculated_cell_statistics():
    codes = np.array([1, 1, 1, 1, 2, 2, 3, 4, 4, 4])
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 1.0, 0.0, 1.0, 2.0])
    y = np.array([2.0, 4.0, 5.0, 9.0, 1.0, 2.0, 7.0, 1.0, 0.0, 1.0])
    cells = cells_of(codes)
    np.testing.assert_array_equal(cells.starts, [0, 4, 6, 7])
    np.testing.assert_array_equal(cells.counts, [4, 2, 1, 3])
    np.testing.assert_array_equal(cells.sum(y), [20.0, 3.0, 7.0, 2.0])
    stats = cell_statistics(x, y, cells)
    # cell 1 by hand: x = 1..4, y = 2 4 5 9: slope = 2.2, corr = 0.9707..., quadratic gain = R2(quad) - R2(lin)
    np.testing.assert_allclose(stats["slope"][0], np.polyfit(x[:4], y[:4], 1)[0], rtol=1e-12)
    np.testing.assert_allclose(stats["corr"][0], np.corrcoef(x[:4], y[:4])[0, 1], rtol=1e-12)
    fitted = np.polyval(np.polyfit(x[:4], y[:4], 2), x[:4])
    r2_quad = 1 - ((y[:4] - fitted) ** 2).sum() / ((y[:4] - y[:4].mean()) ** 2).sum()
    np.testing.assert_allclose(stats["r2_linear"][0], np.corrcoef(x[:4], y[:4])[0, 1] ** 2, rtol=1e-12)
    np.testing.assert_allclose(stats["r2_quadratic_gain"][0], r2_quad - stats["r2_linear"][0], rtol=1e-10)
    assert np.isnan(stats["corr"][1]) and np.isnan(stats["slope"][1])  # constant x: undefined, not 0
    assert np.isnan(stats["corr"][2])  # a single stock
    np.testing.assert_allclose(stats["corr"][3], np.corrcoef(x[7:], y[7:])[0, 1], rtol=1e-12)  # exactly 0 here
    np.testing.assert_allclose(stats["slope"][3], 0.0, atol=1e-15)
    # the quadratic fit needs three distinct x values
    codes2 = np.array([1, 1, 1, 1])
    stats2 = cell_statistics(np.array([0.0, 0.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0, 5.0]), cells_of(codes2))
    assert np.isfinite(stats2["corr"][0]) and np.isnan(stats2["r2_quadratic_gain"][0])
    with pytest.raises(ValueError):
        cells_of(np.array([2, 1]))


def test_within_bin_table_uses_saved_bins_and_recomputes_alternative_bins_with_the_tie_convention():
    rng = np.random.default_rng(1)
    months = month_range(200401, 200403)
    keys = [(p, m) for m in months for p in range(1, 121)]
    frame = pd.DataFrame(keys, columns=["permno", "yyyymm"])
    frame["ensemble_signal"] = rng.integers(0, 65, size=len(frame)) / 64  # ties
    frame[RAW_CD] = rng.uniform(0, 0.5, size=len(frame))
    frame[RAW_CW] = rng.uniform(0, 1, size=len(frame))
    saved = conditional_disagreement(frame, SIGNAL_BINS)
    saved["centered_signal"] = 2 * (saved["ensemble_signal"] - 0.5)

    table20 = within_bin_table(saved, SIGNAL_BINS)
    observed = saved.groupby(["yyyymm", "signal_bin"]).size()
    assert list(zip(table20["yyyymm"], table20["signal_bin"])) == observed.index.tolist()
    assert table20["n_stocks"].tolist() == observed.tolist()
    cell = saved[(saved["yyyymm"] == 200402) & (saved["signal_bin"] == 1)]
    row = table20[(table20["yyyymm"] == 200402) & (table20["signal_bin"] == 1)].iloc[0]
    np.testing.assert_allclose(row["cd_vs_signal"], np.corrcoef(cell["ensemble_signal"], cell[CD])[0, 1], rtol=1e-12)
    np.testing.assert_allclose(row["cw_slope"], np.polyfit(cell["ensemble_signal"], cell[CW], 1)[0], rtol=1e-10)
    np.testing.assert_allclose(row["cd_vs_extremity"], -row["cd_vs_signal"], rtol=1e-12)  # |S^c| = -S^c in bin 1

    table30 = within_bin_table(saved, 30)
    alt = conditional_disagreement(frame, 30)
    assert set(table30["signal_bin"]) <= set(range(1, 31))
    assert table30["n_stocks"].sum() == len(saved)
    expected_bins = signal_bins(saved["ensemble_signal"], saved["yyyymm"], 30)
    np.testing.assert_array_equal(alt["signal_bin"].to_numpy(), expected_bins.to_numpy())  # the documented formula ...
    tied = saved[saved.duplicated(["yyyymm", "ensemble_signal"], keep=False)]
    assert len(tied) > 0
    tied_keys = pd.MultiIndex.from_frame(tied[["permno", "yyyymm"]])
    alt_bins = alt.set_index(["permno", "yyyymm"]).loc[tied_keys, "signal_bin"]
    tied = tied.assign(b=alt_bins.to_numpy())
    assert (tied.groupby(["yyyymm", "ensemble_signal"])["b"].nunique() == 1).all()  # ... and ties share a bin
    assert (saved[CD] == conditional_disagreement(frame, SIGNAL_BINS)[CD]).all()  # the 20-bin outputs are not altered


# ------------------------------------------------------------------------------------------------------------ bases


def test_natural_spline_basis_is_cubic_inside_linear_beyond_the_boundary_knots_and_rejects_bad_knots():
    knots = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    x = np.linspace(0, 1, 201)
    N = natural_spline_basis(x, knots)
    assert N.shape == (201, 5)
    np.testing.assert_array_equal(N[:, 0], 1.0)
    np.testing.assert_array_equal(N[:, 1], x)
    beyond = x > 0.9
    for j in range(2, 5):  # natural: linear beyond the last knot (zero second difference)
        second = np.diff(N[beyond, j], 2)
        np.testing.assert_allclose(second, 0.0, atol=1e-12)
    inside = (x > 0.3) & (x < 0.5)
    third = np.diff(N[inside, 2], 3)  # cubic between knots: constant third difference
    np.testing.assert_allclose(third, third[0], rtol=1e-6)
    with pytest.raises(DataValidationError):
        natural_spline_basis(x, np.array([0.1, 0.1, 0.5]))
    with pytest.raises(DataValidationError):
        natural_spline_basis(x, np.array([0.1, 0.5]))


def _month(rng, n=400, n_bins=SIGNAL_BINS):
    s = rng.uniform(0, 1, n)
    sc = 2 * (s - 0.5)
    bins = signal_bins(pd.Series(s), pd.Series(np.zeros(n, dtype="int64")), n_bins).to_numpy().astype("int64")
    cd = unit_rank(pd.Series(rng.uniform(0, 0.5, n) + 0.2 * np.abs(sc)), [pd.Series(bins)]).to_numpy()
    return s, sc, bins, cd


def test_bin_constants_local_basis_and_original_basis_span_the_same_space():
    rng = np.random.default_rng(2)
    s, sc, bins, cd = _month(rng)
    c, h, z2 = bin_constants(sc, bins, SIGNAL_BINS)
    for b in range(1, SIGNAL_BINS + 1):
        v = sc[bins == b]
        np.testing.assert_allclose(c[b - 1], v.mean(), rtol=1e-12)
        np.testing.assert_allclose(h[b - 1], (v.max() - v.min()) / 2, rtol=1e-12)
        z = (v - v.mean()) / h[b - 1]
        np.testing.assert_allclose(z2[b - 1], (z * z).mean(), rtol=1e-12)
        assert z.min() == pytest.approx(-1.0 + (z.max() - 1.0), abs=1e-12) or abs(z.max() - z.min() - 2.0) < 1e-12
    Z = local_basis(sc, bins, SIGNAL_BINS, 2)
    O = original_basis(sc, bins, SIGNAL_BINS)
    assert Z.shape == O.shape == (len(s), 60)
    np.testing.assert_allclose(Z[:, SIGNAL_BINS:2 * SIGNAL_BINS].sum(axis=0), 0.0, atol=1e-10)  # z, q centred per cell
    np.testing.assert_allclose(Z[:, 2 * SIGNAL_BINS:].sum(axis=0), 0.0, atol=1e-10)
    y = rng.standard_normal(len(s))
    np.testing.assert_allclose(residualise(y, Z), residualise(y, O), atol=1e-9)  # same column space
    assert scaled_condition(Z)[0] < scaled_condition(O)[0] / 10  # and a much better conditioned basis
    with pytest.raises(DataValidationError):
        bin_constants(sc, np.where(bins == 3, 4, bins), SIGNAL_BINS)  # an empty bin


def test_scaled_condition_number():
    Q, _ = np.linalg.qr(np.random.default_rng(3).standard_normal((50, 4)))
    cond, ratio = scaled_condition(Q * np.array([1.0, 10.0, 100.0, 1000.0]))  # scaling is undone
    np.testing.assert_allclose([cond, ratio], [1.0, 1.0], rtol=1e-10)
    cond, ratio = scaled_condition(np.column_stack([Q[:, 0], Q[:, 0]]))
    assert cond > 1e15 and ratio < 1e-15  # collinear: fails the 1e-8 estimability ratio
    assert scaled_condition(np.column_stack([Q[:, 0], np.zeros(50)])) == (np.inf, 0.0)  # a zero column


# ------------------------------------------------------------------------------------------- regression identities


def test_coefficients_fwl_identities_and_basis_invariance_on_a_synthetic_month():
    rng = np.random.default_rng(4)
    s, sc, bins, cd = _month(rng)
    design = month_designs(sc, bins, cd)
    y = 0.02 * np.sin(6 * s) + 0.01 * sc * cd + 0.05 * rng.standard_normal(len(s))
    for spec in SPECIFICATIONS:
        X = np.column_stack([design.W[spec], design.x[spec]])
        beta_full = np.linalg.lstsq(X, y, rcond=None)[0][-1]
        xt = design.partialled[spec]
        np.testing.assert_allclose(xt @ y / (xt @ xt), beta_full, rtol=1e-9, atol=1e-14)  # Frisch-Waugh-Lovell
        np.testing.assert_allclose(design.W[spec].T @ xt, 0.0, atol=1e-9)  # partialled: orthogonal to the controls
    X_c = np.column_stack([design.W["C"], design.x["C"]])
    X_o = np.column_stack([original_basis(sc, bins, SIGNAL_BINS), cd, sc * cd])
    b_c, b_o = np.linalg.lstsq(X_c, y, rcond=None)[0], np.linalg.lstsq(X_o, y, rcond=None)[0]
    np.testing.assert_allclose(b_c[-2:], b_o[-2:], rtol=1e-8, atol=1e-12)  # beta_2, beta_3 do not depend on the basis
    # interaction regressor of D: the residual of CD on the spline times the centred rank
    u = (pd.Series(s).rank().to_numpy() - 1) / (len(s) - 1)
    np.testing.assert_allclose(design.x["D"], residualise(cd, design.W["D"][:, :-1]) * (2 * u - 1), rtol=1e-12)


def test_two_step_specification_e_identities_hold_only_under_the_derived_conditions():
    rng = np.random.default_rng(5)
    s, sc, bins, cd = _month(rng)
    design = month_designs(sc, bins, cd)
    W_B, W_C = design.W["B"][:, :-1], design.W["C"][:, :-1]  # the controls without CD
    V_demean, V_linear, V_quadratic = W_B[:, :SIGNAL_BINS], W_B, W_C
    y = 0.03 * np.cos(5 * s) + 0.02 * sc**3 + 0.01 * sc * cd + 0.05 * rng.standard_normal(len(s))
    b3_c = interaction_coefficient(y, design.W["C"], design.x["C"])
    b3_b = interaction_coefficient(y, design.W["B"], design.x["B"])

    def close(actual, desired):
        np.testing.assert_allclose(actual, desired, rtol=1e-8, atol=1e-13)

    # identities: V within W and S^c V within W
    close(two_step_coefficient(y, sc, cd, V_demean, W_B, True), b3_b)
    close(two_step_coefficient(y, sc, cd, V_demean, W_C, True), b3_c)
    close(two_step_coefficient(y, sc, cd, V_linear, W_C, True), b3_c)
    # counterexamples: S^c times the detrending basis leaves the span of the second-stage controls, or no level term
    scale = abs(b3_c)
    assert abs(two_step_coefficient(y, sc, cd, V_quadratic, W_C, True) - b3_c) > 1e-3 * scale
    assert abs(two_step_coefficient(y, sc, cd, V_linear, W_B, True) - b3_b) > 1e-3 * scale
    assert abs(two_step_coefficient(y, sc, cd, V_linear, W_B, True) - b3_c) > 1e-3 * scale
    assert abs(two_step_coefficient(y, sc, cd, V_linear, W_C, False) - b3_c) > 1e-3 * scale
    # the algebra behind the identity: S^c * (fitted bin-linear trend) lies in the span of C's controls, the cubic
    # part of S^c * (fitted bin-quadratic trend) does not
    fitted_linear = cd - residualise(cd, V_linear)
    fitted_quadratic = cd - residualise(cd, V_quadratic)
    np.testing.assert_allclose(residualise(sc * fitted_linear, W_C), 0.0, atol=1e-9)
    assert np.abs(residualise(sc * fitted_quadratic, W_C)).max() > 1e-6


def test_partialled_interaction_of_b_and_c_has_no_within_bin_signal_dependence_while_a_keeps_some():
    rng = np.random.default_rng(6)
    s, sc, bins, cd = _month(rng, n=1000)
    design = month_designs(sc, bins, cd)
    for spec in ("B", "C"):
        out = extreme_cell_diagnostics(design.partialled[spec], s, bins, SIGNAL_BINS)
        np.testing.assert_allclose([out["bin1_corr_vs_signal"], out["bin20_corr_vs_signal"]], 0.0, atol=1e-8)
    out_c = extreme_cell_diagnostics(design.partialled["C"], s, bins, SIGNAL_BINS)
    shares_c = [out_c["bin1_signal_variance_share"], out_c["bin20_signal_variance_share"]]
    np.testing.assert_allclose(shares_c, 0.0, atol=1e-8)
    out_a = extreme_cell_diagnostics(design.partialled["A"], s, bins, SIGNAL_BINS)
    assert abs(out_a["bin1_corr_vs_signal"]) > 0.05 or abs(out_a["bin20_corr_vs_signal"]) > 0.05  # CD rises with |S^c|
    assert out_a["all_bins_mean_noise_floor"] == pytest.approx(np.mean([2 / (n - 1) for n in np.bincount(bins)[1:]]))
    small = extreme_cell_diagnostics(design.partialled["A"][:3], s[:3], np.array([1, 1, 1]), 1)
    assert np.isnan(small["bin1_corr_vs_signal"])  # fewer than four stocks: undefined


# ------------------------------------------------------------------------------------------------- synthetic outcomes


def test_fama_macbeth_and_bartlett_hac_standard_errors_by_hand():
    rng = np.random.default_rng(7)
    series = rng.standard_normal((3, 50))
    mean, se = fama_macbeth(series)
    np.testing.assert_allclose(mean, series.mean(axis=1))
    np.testing.assert_allclose(se, series.std(axis=1, ddof=1) / np.sqrt(50))
    np.testing.assert_allclose(bartlett_hac_se(series, 0), series.std(axis=1, ddof=0) / np.sqrt(50))  # L = 0: pop. SD
    x = series[1] - series[1].mean()
    omega = (x @ x) / 50
    for lag in (1, 2, 3):
        omega += 2 * (1 - lag / 4) * (x[lag:] @ x[:-lag]) / 50
    np.testing.assert_allclose(bartlett_hac_se(series, 3)[1], np.sqrt(omega / 50), rtol=1e-12)


def _stacked(rng, months=24, n=300):
    frames = []
    for i, m in enumerate(month_range(200401, 200401 + months - 1 if months <= 12 else 200500 + months - 13)):
        s, sc, bins, cd = _month(rng, n=n + i % 3)
        frames.append(
            pd.DataFrame({"yyyymm": m, "permno": np.arange(1, len(s) + 1), "s": s, "sc": sc, "bins": bins, "cd": cd})
        )
    frame = pd.concat(frames, ignore_index=True).sort_values(["yyyymm", "permno"], ignore_index=True)
    return frame


def test_scenario_means_by_hand_and_monte_carlo_matches_direct_ols_and_is_deterministic():
    rng = np.random.default_rng(8)
    frame = _stacked(rng, months=12)
    sc, cd, s = (frame[c].to_numpy() for c in ("sc", "cd", "s"))
    cells = cells_of(frame["yyyymm"].to_numpy())
    means = scenario_means(sc, cd, cells)
    assert set(means) == set(SCENARIOS)
    np.testing.assert_allclose(means["linear"], 0.01 * sc)
    np.testing.assert_allclose(means["cubic_negative"], 0.01 * sc + 0.015 * sc**3 - 0.01 * sc * cd)
    np.testing.assert_allclose(means["tanh"], 0.012 * np.tanh(2.5 * sc))
    first = slice(0, cells.counts[0])
    q05, q95 = np.quantile(s[first], [0.05, 0.95])
    np.testing.assert_allclose(
        means["boundary"][first],
        0.01 * sc[first] + 0.05 * np.maximum(s[first] - q95, 0) - 0.05 * np.maximum(q05 - s[first], 0),
    )

    weights = {spec: np.empty(len(frame)) for spec in SPECIFICATIONS}
    designs = []
    for start, n in zip(cells.starts, cells.counts):
        sl = slice(start, start + n)
        d = month_designs(sc[sl], frame["bins"].to_numpy()[sl], cd[sl])
        designs.append(d)
        for spec in SPECIFICATIONS:
            weights[spec][sl] = d.partialled[spec] / (d.partialled[spec] @ d.partialled[spec])
    result = monte_carlo(weights, means, cells, replications=5, seed=123)
    assert len(result) == len(SCENARIOS) * len(SPECIFICATIONS)
    # the first replication's first-month coefficient equals a direct OLS on y = mu + eps, eps drawn the same way
    eps = de.NOISE_SD * np.random.default_rng(123).standard_normal(len(frame))
    y = means["boundary"] + eps
    for spec in ("A", "C", "D"):
        direct = interaction_coefficient(y[:cells.counts[0]], designs[0].W[spec], designs[0].x[spec])
        shortcut = (weights[spec] * y)[:cells.counts[0]].sum()
        np.testing.assert_allclose(shortcut, direct, rtol=1e-8, atol=1e-14)
    again = monte_carlo(weights, means, cells, replications=5, seed=123)
    pd.testing.assert_frame_equal(result, again)  # deterministic
    other = monte_carlo(weights, means, cells, replications=5, seed=124)
    assert not np.allclose(result["mean_estimate"], other["mean_estimate"])
    row = result[(result["scenario"] == "linear_negative") & (result["specification"] == "C")].iloc[0]
    assert row["true_interaction"] == -0.01 and row["replications"] == 5 and row["months"] == 12
    assert 0 <= row["rejection_rate_fm"] <= 1 and 0 <= row["rejection_rate_hac6"] <= 1


def test_null_and_strong_interaction_behaviour_of_the_rerun():
    rng = np.random.default_rng(9)
    frame = _stacked(rng, months=24, n=400)
    sc, cd = frame["sc"].to_numpy(), frame["cd"].to_numpy()
    cells = cells_of(frame["yyyymm"].to_numpy())
    weights = {spec: np.empty(len(frame)) for spec in ("B", "C")}
    for start, n in zip(cells.starts, cells.counts):
        sl = slice(start, start + n)
        d = month_designs(sc[sl], frame["bins"].to_numpy()[sl], cd[sl])
        for spec in weights:
            weights[spec][sl] = d.partialled[spec] / (d.partialled[spec] @ d.partialled[spec])
    means = scenario_means(sc, cd, cells)
    means["linear_negative"] = means["linear"] - 0.5 * sc * cd  # a huge true interaction
    result = monte_carlo(weights, means, cells, replications=200, seed=1).set_index(["scenario", "specification"])
    for spec in weights:
        null = result.loc[[("linear", spec)]].iloc[0]
        assert null["rejection_rate_fm"] < 0.15  # a linear g is spanned: size near 5% (MC SE 1.5 pp)
        assert abs(null["bias"]) < 3 * null["mean_fm_se"]
        strong = result.loc[[("linear_negative", spec)]].iloc[0]
        assert strong["rejection_rate_fm"] == 1.0 and strong["share_rejections_negative"] == 1.0
        np.testing.assert_allclose(strong["noise_free_estimate"], -0.5, rtol=1e-6)  # exact in the absence of noise


# ---------------------------------------------------------------------------------------------- constant family


def test_constant_family_months_and_three_family_disagreement():
    rng = np.random.default_rng(10)
    months = month_range(202101, 202103)
    keys = pd.DataFrame([(p, m) for m in months for p in range(1, 61)], columns=["permno", "yyyymm"])
    ranks = keys.copy()
    for family in FAMILIES:
        ranks[f"{family}_rank"] = unit_rank(pd.Series(rng.standard_normal(len(keys))), [keys["yyyymm"]])
    ranks.loc[ranks["yyyymm"] != 202102, "elastic_net_rank"] = 0.5  # constant in two of the three months
    ranks["ensemble_signal"] = sum(ranks[f"{f}_rank"] for f in FAMILIES) / 4.0
    constant = constant_family_months(ranks)
    assert constant == {"elastic_net": [202101, 202103], "random_forest": [], "lightgbm": [], "mlp": []}
    signals = keys.copy()
    signals["ensemble_signal"] = ranks["ensemble_signal"]
    signals["signal_bin"] = signal_bins(signals["ensemble_signal"], signals["yyyymm"], SIGNAL_BINS)
    cd3 = three_family_disagreement(ranks, signals)
    r = ranks[[f"{f}_rank" for f in ("random_forest", "lightgbm", "mlp")]].to_numpy()
    d3 = pd.Series(r.std(axis=1, ddof=0))
    expected = unit_rank(d3, [signals["yyyymm"], signals["signal_bin"]])
    np.testing.assert_array_equal(cd3.to_numpy(), expected.to_numpy())
    rows = ranks["yyyymm"] != 202102
    s3 = r[rows].mean(axis=1)
    np.testing.assert_allclose(ranks.loc[rows, "ensemble_signal"], (0.5 + 3 * s3) / 4, rtol=1e-12)  # one family at 0.5
    d4 = ranks.loc[rows, [f"{f}_rank" for f in FAMILIES]].to_numpy().std(axis=1, ddof=0)
    np.testing.assert_allclose(d4**2, 0.75 * d3[rows] ** 2 + 3 / 16 * (s3 - 0.5) ** 2, rtol=1e-10)


# ------------------------------------------------------------------------------------------------------ end to end


def _setup(tmp_path, constant_year: int | None = 2005, seed=0):
    """Two baseline OOS years (2004, 2005) of synthetic signals; the Elastic Net is constant in ``constant_year``."""
    rng = np.random.default_rng(seed)
    paths = DataPaths(tmp_path / "data")
    for directory in (paths.universe, paths.panel, paths.diagnostics, paths.conditional_disagreement,
                      paths.ensemble_signals, *(paths.oos_predictions / f for f in FAMILIES)):
        directory.mkdir(parents=True)
    (paths.diagnostics / "summary.json").write_text(json.dumps({"validation": "PASS"}))
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["data"]["sample"] = {"start": 200001, "end": 200512}
    config["walk_forward"] = {"first_oos_year": 2004, "validation_years": 2}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    replicas = int(config["bootstrap"]["replicas"])
    for year in (2004, 2005):
        months = month_range(year * 100 + 1, year * 100 + 12)
        keys = pd.DataFrame([(p, m) for m in months for p in range(1, 701 + m % 3)], columns=["permno", "yyyymm"])
        n = len(keys)
        ensemble = keys.copy()
        for family in FAMILIES:
            if family == "elastic_net" and year == constant_year:
                ensemble[f"{family}_rank"] = 0.5
            else:
                ensemble[f"{family}_rank"] = unit_rank(pd.Series(rng.standard_normal(n)), [keys["yyyymm"]])
        ensemble["ensemble_signal"] = (
            ensemble["elastic_net_rank"] + ensemble["random_forest_rank"] + ensemble["lightgbm_rank"]
            + ensemble["mlp_rank"]
        ) / 4.0
        ensemble.sample(frac=1.0, random_state=1).to_parquet(paths.ensemble_signals / f"{year}.parquet", index=False)
        (paths.ensemble_signals / f"{year}.json").write_text(json.dumps({"oos_year": year}))
        signals = keys.copy()
        signals["ensemble_signal"] = ensemble["ensemble_signal"]
        signals[RAW_CD] = ensemble[[f"{f}_rank" for f in FAMILIES]].to_numpy().std(axis=1, ddof=0)
        signals[RAW_CW] = rng.uniform(0.0, 1.0, size=n)
        conditional = conditional_disagreement(signals, SIGNAL_BINS)
        shuffled = conditional.sample(frac=1.0, random_state=2)
        shuffled.to_parquet(paths.conditional_disagreement / f"{year}.parquet", index=False)
        (paths.conditional_disagreement / f"{year}.json").write_text(json.dumps({"oos_year": year}))
        universe = keys.copy()
        universe["exchange"] = "NYSE"
        universe["mktcap"] = rng.integers(1, 30, size=n) * 1000.0
        panel = universe.copy()
        panel["AM"] = rng.standard_normal(n)
        panel["RealizedVol"] = -rng.integers(1, 9, size=n) / 100
        panel["Illiquidity"] = rng.integers(1, 9, size=n) / 10
        panel.loc[rng.random(n) < 0.10, "Illiquidity"] = np.nan
        panel["Size"] = -np.log(panel["mktcap"])
        universe.to_parquet(paths.universe / f"universe_{year}.parquet", index=False)
        panel.to_parquet(paths.panel / f"panel_{year}.parquet", index=False)
        for family in FAMILIES:
            n_candidates = 12 if family == "elastic_net" else 8
            if family == "elastic_net" and year == constant_year:
                scores = [-0.001 * (i + 1) for i in range(10)] + [0.0, 0.0]
                predictions = pd.DataFrame({
                    "permno": np.tile(keys["permno"].to_numpy(), replicas),
                    "yyyymm": np.tile(keys["yyyymm"].to_numpy(), replicas),
                    "family": family,
                    "replica": np.repeat(np.arange(replicas), n),
                })
                predictions["prediction"] = 0.1 * predictions["replica"] + 0.001 * (predictions["yyyymm"] % 100)
                predictions.to_parquet(paths.oos_predictions / family / f"{year}.parquet", index=False)
            else:
                scores = list(rng.uniform(-0.01, 0.05, size=n_candidates))
            best = int(np.argmax(scores))
            (paths.oos_predictions / family / f"{year}.json").write_text(json.dumps({
                "family": family, "oos_year": year, "selected_candidate_index": best,
                "selected_validation_score": scores[best], "validation_scores": scores, "replicas": replicas,
            }))
    run_h2_controls(paths, config_path, log=lambda *a: None)
    return paths, config_path, ["--data-dir", str(paths.root), "--config", str(config_path), "--replications", "12"]


def _hashes(paths):
    inputs = (paths.conditional_disagreement, paths.ensemble_signals, paths.h2_controls, paths.oos_predictions)
    files = sorted(p for d in inputs for p in d.rglob("*") if p.is_file())
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def test_end_to_end_outputs_provenance_and_reproducible_completion(tmp_path, monkeypatch, capsys):
    paths, config_path, argv = _setup(tmp_path)
    directory = paths.design_evidence_diagnostics
    before = _hashes(paths)
    replaced, replace = [], os.replace

    def recording_replace(src, dst):
        replaced.append(os.path.basename(dst))
        replace(src, dst)

    monkeypatch.setattr(de.os, "replace", recording_replace)
    outputs = []
    for _ in range(2):
        assert main(argv) == 0
        outputs.append({name: (directory / name).read_bytes() for name in FILES})
    assert _hashes(paths) == before  # the saved signals, controls and metadata are untouched
    assert sorted(p.name for p in directory.iterdir()) == FILES  # aggregates only, no .tmp
    assert replaced[-1] == "summary.json" and replaced.count("summary.json") == 2  # the marker, written last
    for name in FILES[:-1]:
        assert outputs[0][name] == outputs[1][name]  # identical bytes
    summaries = [json.loads(o["summary.json"]) for o in outputs]
    for s in summaries:
        s.pop("executed_at_utc")
    assert summaries[0] == summaries[1]
    printed = capsys.readouterr().out
    assert "new simulations" in printed and "no methodology was changed automatically" in printed

    summary = summaries[0]
    assert summary["status"] == "complete"
    assert (summary["first_month"], summary["last_month"], summary["months"]) == (200401, 200512, 24)
    source = pd.concat([pd.read_parquet(paths.conditional_disagreement / f"{y}.parquet") for y in (2004, 2005)])
    assert summary["n_stock_months"] == len(source)
    assert summary["sources_sha256"]["conditional_disagreement"] == {
        str(y): hashlib.sha256((paths.conditional_disagreement / f"{y}.parquet").read_bytes()).hexdigest()
        for y in (2004, 2005)
    }
    assert set(summary["sources_sha256"]["oos_prediction_metadata"]) == set(FAMILIES)
    mc = summary["synthetic_outcome_experiments"]
    assert (mc["seed"], mc["replications"], mc["noise_sd"], mc["months"]) == (de.SEED, 12, 0.10, 24)
    assert set(mc["scenarios"]) == set(SCENARIOS) and "new simulations" in mc["category"]
    assert set(mc["stress_scenario"]["results"]) == set(SPECIFICATIONS)
    comparison = summary["comparison_with_recorded_figures"]
    assert all(v["status"] == "new simulation" for k, v in comparison.items() if k.startswith("rejection_boundary"))
    assert set(summary["deterministic_checks_outside_tolerance"]) <= set(comparison)  # synthetic: no reproduction

    # geometry: the 20-bin profile matches a direct computation on the saved files
    profile = pd.read_csv(directory / "geometry_bins.csv")
    assert set(profile["bins"]) == {20, 30, 40}
    p20 = profile[profile["bins"] == 20].set_index("signal_bin")
    assert p20["n_months"].tolist() == [24] * 20
    cells = source.groupby(["yyyymm", "signal_bin"])[["ensemble_signal", CD, CW]]
    corr = cells.apply(lambda g: np.corrcoef(g["ensemble_signal"], g[CD])[0, 1])
    np.testing.assert_allclose(p20["cd_vs_signal_mean"], corr.groupby("signal_bin").mean(), rtol=1e-10)
    slope = cells.apply(lambda g: np.polyfit(g["ensemble_signal"], g[CW], 1)[0])
    np.testing.assert_allclose(p20["cw_slope_mean"], slope.groupby("signal_bin").mean(), rtol=1e-8)
    extremes = pd.read_csv(directory / "extreme_bin_months.csv")
    assert extremes["yyyymm"].tolist() == month_range(200401, 200512)
    np.testing.assert_allclose(extremes["bin1_cd_vs_signal"], corr.xs(1, level="signal_bin"), rtol=1e-10)

    # specifications: C keeps less identifying variance than B, the local basis is better conditioned than the original
    spec = pd.read_csv(directory / "specification_summary.csv").set_index("specification")
    assert list(spec.index) == list(SPECIFICATIONS)
    spec_b, spec_c = spec.loc[["B"]].iloc[0], spec.loc[["C"]].iloc[0]
    assert spec_b["identifying_variance_pooled_share_of_B"] == 1.0
    assert 0.5 < spec_c["identifying_variance_pooled_share_of_B"] < 1.0
    assert spec_c["se_cost_vs_B_pooled"] == pytest.approx(
        np.sqrt(1 / spec_c["identifying_variance_pooled_share_of_B"]) - 1
    )
    monthly = pd.read_csv(directory / "specification_monthly.csv")
    assert len(monthly) == 24 and monthly["estimable_full"].all()  # about 35 stocks per cell
    np.testing.assert_array_equal(
        monthly["estimable_complete_case"], (monthly["min_cell_size_complete_case"] >= de.MIN_CELL)
        & (monthly["min_distinct_signal_complete_case"] >= de.MIN_DISTINCT)
    )
    assert (monthly["C_condition_number"] < monthly["C_original_basis_condition_number"]).all()
    assert (monthly["min_cell_size_complete_case"] <= monthly["min_cell_size"]).all()
    identities = pd.read_csv(directory / "algebraic_identities_monthly.csv")
    for column in ("basis_invariance_beta2", "basis_invariance_beta3", "fwl_C", "E_demean_controlsB_vs_B",
                   "E_demean_controlsC_vs_C", "E_linear_controlsC_vs_C"):
        assert identities[column].max() < 1e-9 * max(1.0, identities["scale_beta3_C"].max())
    for column in ("E_quadratic_controlsC_vs_C", "E_linear_controlsB_vs_B", "E_linear_controlsC_no_level_vs_C"):
        assert identities[column].max() > 1e-6
    mc_table = pd.read_csv(directory / "monte_carlo.csv")
    assert len(mc_table) == len(SCENARIOS) * len(SPECIFICATIONS) and (mc_table["replications"] == 12).all()

    # the constant Elastic Net year is found from the metadata, the ranks and the raw predictions
    constant = summary["deterministic_diagnostics"]["constant_family"]
    assert [(e["family"], e["oos_year"]) for e in constant["zero_selected_score"]] == [("elastic_net", 2005)]
    entry = constant["zero_selected_score"][0]
    assert entry["n_negative_scores"] == 10 and entry["n_zero_scores"] == 2 and entry["first_zero_selected"]
    assert all(c["alpha"] == 0.1 for c in entry["zero_score_candidates"])  # the most penalised candidates
    assert max(entry["identity_max_abs_error"].values()) < 1e-12
    assert constant["constant_rank_months"] == {"elastic_net": {"2005": month_range(200501, 200512)}}
    cross_sections = constant["constant_prediction_cross_sections"]
    assert cross_sections == {"elastic_net 2005": {"cross_sections": 60, "constant": 60}}
    assert set(constant["geometry"]) == {"2004", "2005", "neighbours_pooled"}
    cells_2005 = {"cd4_vs_signal_bin1", "cd4_vs_signal_bin20", "cd3_vs_signal_bin1", "cd3_vs_signal_bin20"}
    assert set(constant["geometry"]["2005"]) == cells_2005
    pooled = constant["geometry"]["neighbours_pooled"]
    assert pooled["years"] == [2004]
    assert {k: v for k, v in pooled.items() if k != "years"} == constant["geometry"]["2004"]


def test_no_constant_family_gives_an_empty_constant_family_report(tmp_path):
    paths, config_path, argv = _setup(tmp_path, constant_year=None)
    assert main(argv) == 0
    summary = json.loads((paths.design_evidence_diagnostics / "summary.json").read_text())
    constant = summary["deterministic_diagnostics"]["constant_family"]
    assert constant["zero_selected_score"] == [] and constant["geometry"] == {}
    assert constant["constant_rank_months"] == {}


def test_invalid_or_missing_inputs_leave_no_completion_marker(tmp_path, capsys):
    paths, config_path, argv = _setup(tmp_path)
    marker = paths.design_evidence_diagnostics / "summary.json"
    assert main(argv) == 0 and marker.exists()

    def rejected(message):
        assert main(argv) == 2
        assert message in capsys.readouterr().err
        assert not marker.exists()
        assert not list(paths.design_evidence_diagnostics.glob("*.tmp"))

    source = paths.conditional_disagreement / "2005.parquet"
    original = source.read_bytes()
    frame = pd.read_parquet(source)
    changed = frame.copy()
    value = float(changed[CD].iloc[0])
    changed.loc[0, CD] = 1.0 - value if value != 0.5 else 0.25
    changed.to_parquet(source, index=False)  # a stored conditional rank that is no longer the rank of its raw measure
    rejected("conditional_between_family_disagreement differs from its recomputation in 1 rows")
    source.write_bytes(original)
    assert main(argv) == 0
    controls = paths.h2_controls / "2005.parquet"
    kept = controls.read_bytes()
    pd.read_parquet(controls).iloc[::-1].to_parquet(controls, index=False)  # same content, another file
    rejected("2005 H2 controls: the output file does not match the SHA-256 in its marker")
    controls.write_bytes(kept)
    assert main(argv) == 0
    (paths.oos_predictions / "mlp" / "2004.json").unlink()
    rejected("mlp 2004: no OOS prediction metadata")


def test_a_failed_write_leaves_no_marker_and_the_run_can_be_repeated(tmp_path, monkeypatch):
    paths, config_path, argv = _setup(tmp_path)
    directory = paths.design_evidence_diagnostics
    assert main(argv) == 0
    reference = {name: (directory / name).read_bytes() for name in FILES[:-1]}

    def failing_dumps(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(de.json, "dumps", failing_dumps)
    with pytest.raises(OSError):
        main(argv)
    assert not (directory / "summary.json").exists()  # the stale marker was removed first, the new one never written
    assert not list(directory.glob("*.tmp"))
    monkeypatch.undo()
    assert main(argv) == 0
    assert {name: (directory / name).read_bytes() for name in FILES[:-1]} == reference


def test_outputs_are_removed_with_their_sources(tmp_path):
    paths, config_path, argv = _setup(tmp_path)
    assert main(argv) == 0
    assert paths.design_evidence_diagnostics.exists()
    run_h2_controls(paths, config_path, years=[2004], log=lambda *a: None)  # new controls invalidate the evidence
    assert not paths.design_evidence_diagnostics.exists()
    assert main(argv) == 0
    remove_conditional_disagreement(paths, 2005)
    assert not paths.design_evidence_diagnostics.exists()


# ------------------------------------------------------------------------------------- guards and checkpoint


def test_fwl_weights_reject_an_interaction_in_the_span_of_its_controls():
    rng = np.random.default_rng(11)
    s, sc, bins, cd = _month(rng)
    design = month_designs(sc, bins, cd)
    w, ss = fwl_weights(design.x["C"], design.partialled["C"], "ok")
    np.testing.assert_allclose(w, design.partialled["C"] / ss)
    constant_cd = np.full_like(cd, 0.4)  # S^c x constant lies in the span of the bin-linear controls
    x = sc * constant_cd
    partialled = residualise(x, design.W["B"][:, :-1])
    with pytest.raises(DataValidationError, match="collinear"):
        fwl_weights(x, partialled, "collinear month")


def _complete_comparison():
    """Every expected check reproduced, plus the five documented exceptions at their reconstructed values."""
    comparison = {
        k: {"recorded": 0.5, "reproduced": 0.5, "tolerance": 0.005, "status": "reproduced within tolerance"}
        for k in EXPECTED_DETERMINISTIC_CHECKS
    }
    for key, (reconstructed, _) in RESOLVED.items():
        comparison[key] = {"recorded": round(reconstructed, 2), "reproduced": reconstructed, "tolerance": 0.005,
                           "status": "outside tolerance"}
    return comparison


def test_resolved_discrepancies_need_their_reconstructed_value_and_never_change_the_entry():
    comparison = _complete_comparison()
    resolved, unresolved = classify_failures(comparison)
    assert set(resolved) == set(RESOLVED) and unresolved == []
    key, (reconstructed, _) = next(iter(RESOLVED.items()))
    entry = comparison[key]
    assert entry["status"] == "outside tolerance" and entry["recorded"] == round(reconstructed, 2)  # unchanged
    assert str(reconstructed) in entry["resolution"] and "methodology_changes.md" in entry["resolution"]
    assert all("resolution" not in comparison[k] for k in EXPECTED_DETERMINISTIC_CHECKS if k not in RESOLVED)
    # a value 0.002 away from the reconstructed one is still near the recorded value (the old shared 0.03 bound
    # accepted it) but it is no longer the verified figure: unresolved, and the checkpoint blocks
    drifted = _complete_comparison()
    drifted[key]["reproduced"] = reconstructed + 0.002
    resolved, unresolved = classify_failures(drifted)
    assert key not in resolved and unresolved == [f"{key}: differs from the reconstructed value {reconstructed}"]
    assert "resolution" not in drifted[key]
    # an unexpected entry outside tolerance is unresolved as well
    extra = _complete_comparison()
    extra["other"] = {"recorded": 0.5, "reproduced": 0.6, "tolerance": 0.005, "status": "outside tolerance"}
    assert classify_failures(extra)[1] == ["other"]


def test_checkpoint_fails_closed_on_missing_not_computed_or_undefined_checks():
    missing = _complete_comparison()
    del missing["cd3_vs_signal_2021_bin1"]
    resolved, unresolved = classify_failures(missing)
    assert unresolved == ["cd3_vs_signal_2021_bin1: missing or not computed"] and set(resolved) == set(RESOLVED)
    not_computed = _complete_comparison()
    not_computed["cw_vs_signal_bin20"]["reproduced"] = None
    not_computed["cw_vs_signal_bin20"]["status"] = "not computed"
    assert classify_failures(not_computed)[1] == ["cw_vs_signal_bin20: missing or not computed"]
    undefined_value = _complete_comparison()
    undefined_value[next(iter(RESOLVED))]["reproduced"] = float("nan")
    assert len(classify_failures(undefined_value)[1]) == 1  # NaN never matches a reconstructed value
    assert classify_failures(_complete_comparison(), [202103])[1] == ["extreme-bin correlation undefined in 202103"]
    assert classify_failures({})[1] == sorted(f"{k}: missing or not computed" for k in EXPECTED_DETERMINISTIC_CHECKS)


def test_checkpoint_status_provenance_and_new_diagnostics_end_to_end(tmp_path, capsys):
    paths, config_path, argv = _setup(tmp_path)
    assert main(argv) == 0
    summary = json.loads((paths.design_evidence_diagnostics / "summary.json").read_text())
    unresolved = summary["unresolved_deterministic_failures"]
    assert summary["deterministic_checkpoint"] == "blocked"  # synthetic data: the 2021 checks cannot be computed
    assert summary["expected_deterministic_checks"] == list(EXPECTED_DETERMINISTIC_CHECKS)
    comparison = summary["comparison_with_recorded_figures"]
    missing = [k for k in EXPECTED_DETERMINISTIC_CHECKS if k not in comparison]
    assert missing and all(f"{k}: missing or not computed" in unresolved for k in missing)  # fail closed
    outside = {k for k, v in comparison.items() if v["status"] == "outside tolerance"}
    assert set(summary["resolved_discrepancies"]) <= outside
    for key in outside - set(summary["resolved_discrepancies"]):  # synthetic values do not match the reconstructed
        assert any(item == key or item.startswith(f"{key}: ") for item in unresolved), key
    assert all("resolution" in comparison[k] for k in summary["resolved_discrepancies"])
    assert "transitive_inputs" in summary["sources_sha256"]
    sources = summary["sources_sha256"]
    assert sources["config"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
    for name, directory, suffix in (
        ("conditional_disagreement", paths.conditional_disagreement, ".parquet"),
        ("conditional_disagreement_metadata", paths.conditional_disagreement, ".json"),
        ("h2_controls", paths.h2_controls, ".parquet"),
        ("h2_controls_metadata", paths.h2_controls, ".json"),
        ("ensemble_signals", paths.ensemble_signals, ".parquet"),
    ):
        assert sources[name] == {
            str(y): hashlib.sha256((directory / f"{y}{suffix}").read_bytes()).hexdigest() for y in (2004, 2005)
        }, name
    predictions = paths.oos_predictions / "elastic_net" / "2005.parquet"
    assert sources["oos_predictions"] == {"elastic_net 2005": hashlib.sha256(predictions.read_bytes()).hexdigest()}
    assert "files_read" not in summary["deterministic_diagnostics"]["constant_family"]
    cw = summary["deterministic_diagnostics"]["cw_full_cross_section"]
    assert cw["conditional_disagreement_vs_conditional_uncertainty"]["n_valid"] == 24
    assert cw["raw_disagreement_vs_raw_uncertainty"]["n_valid"] == 24
    h2 = summary["deterministic_diagnostics"]["h2_design_complete_case"]
    assert 0 < h2["min_identifying_variance_share_of_C"] <= h2["mean_identifying_variance_share_of_C"] <= 1
    assert h2["mean_variance_inflation"] >= 1 and 0 < h2["complete_case_share_of_C_identifying_variance"] <= 1
    monthly = pd.read_csv(paths.design_evidence_diagnostics / "specification_monthly.csv")
    np.testing.assert_allclose(monthly["H2_variance_inflation"], 1 / monthly["H2_identifying_variance_share_of_C"])
    assert summary["deterministic_diagnostics"]["months_with_undefined_extreme_bin_correlation"] == []
    printed = capsys.readouterr().out
    assert "resolved and documented" in printed and "checkpoint" in printed
    # metadata without the selected score is rejected cleanly
    marker = paths.oos_predictions / "lightgbm" / "2004.json"
    metadata = json.loads(marker.read_text())
    del metadata["selected_validation_score"]
    marker.write_text(json.dumps(metadata))
    assert main(argv) == 2
    assert "lightgbm 2004: metadata without validation scores or selected candidate" in capsys.readouterr().err
