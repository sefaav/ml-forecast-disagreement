"""Serialisation, provenance, atomic publication and command line of the inference (synthetic data only)."""

import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest
import yaml

from disagreement.controls import h2
from disagreement.data.config import DEFAULT_CONFIG, DataPaths
from disagreement.data.target import TARGET_COLUMNS, rank_target
from disagreement.data.validation import DataValidationError
from disagreement.inference import output, run as cli
from disagreement.inference.monthly import SPECIFICATION_B, SPECIFICATION_C, estimate_h1_month
from disagreement.inference.orchestrate import OUTSTANDING_ROBUSTNESS, RAW_SERIES, run_inference
from disagreement.inference.output import (
    MANIFEST,
    OUTPUT_FILES,
    RAW_SPECIFICATION,
    SUMMARY,
    PublicationError,
    execute_inference,
    publish,
    result_artifacts,
    run_with_provenance,
    verify_published,
)
from disagreement.inference.series import SPECIFICATIONS
from disagreement.signals.conditional import SIGNAL_BINS, conditional_disagreement
from disagreement.walkforward.calendar import month_range
from test_orchestrate import _tiny
from test_series import _basis, _coefficients, _estimate

YEAR = 2004


def _strict_json(text):
    def reject(constant):
        raise ValueError(f"non-standard JSON constant {constant}")
    return json.loads(text, parse_constant=reject)


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    """A published synthetic OOS year 2004, large enough to estimate, built with the pipeline's own commands."""
    root = tmp_path_factory.mktemp("source")
    rng = np.random.default_rng(20261116)
    keys = pd.DataFrame([(p, m) for m in month_range(200401, 200412) for p in range(1, 1100 + m % 7)],
                        columns=["permno", "yyyymm"])
    universe = keys.assign(exchange="NYSE", mktcap=rng.integers(1, 500, size=len(keys)) * 1000.0)
    panel = universe.assign(RealizedVol=-rng.uniform(0.01, 0.1, size=len(keys)),
                            Illiquidity=rng.uniform(0.0, 1.0, size=len(keys)))
    panel.loc[rng.random(len(keys)) < 0.08, "Illiquidity"] = np.nan
    signals = keys.assign(ensemble_signal=rng.integers(0, 401, size=len(keys)) / 400,
                          between_family_disagreement_raw=rng.uniform(0.0, 0.5, size=len(keys)),
                          within_family_uncertainty_raw=rng.uniform(0.0, 1.0, size=len(keys)))
    cond = conditional_disagreement(signals, SIGNAL_BINS)
    paths = DataPaths(root / "data")
    for directory in (paths.universe, paths.panel, paths.conditional_disagreement, paths.diagnostics, paths.target):
        directory.mkdir(parents=True)
    (paths.diagnostics / "summary.json").write_text(json.dumps({"validation": "PASS"}))
    universe.to_parquet(paths.universe / f"universe_{YEAR}.parquet", index=False)
    panel.to_parquet(paths.panel / f"panel_{YEAR}.parquet", index=False)
    cond.to_parquet(paths.conditional_disagreement / f"{YEAR}.parquet", index=False)
    (paths.conditional_disagreement / f"{YEAR}.json").write_text(json.dumps({"oos_year": YEAR}))
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["data"]["sample"] = {"start": 200001, "end": 200412}  # a synthetic calendar: OOS year 2004 only
    config["walk_forward"] = {"first_oos_year": YEAR, "validation_years": 2}
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert h2.main(["--data-dir", str(paths.root), "--config", str(config_path), "--year", str(YEAR)]) == 0
    target = keys.copy()
    sc = 2.0 * (cond.set_index(["permno", "yyyymm"]).loc[pd.MultiIndex.from_frame(keys), "ensemble_signal"]
                .to_numpy() - 0.5)
    target["ret_next"] = 0.02 * sc + 0.1 * rng.standard_normal(len(keys))
    target.loc[rng.random(len(keys)) < 0.1, "ret_next"] = np.nan
    target["target_rank"] = rank_target(target)
    target.loc[:, list(TARGET_COLUMNS)].to_parquet(paths.target / f"target_{YEAR}.parquet", index=False)
    return root


def _copy(source, tmp_path):
    shutil.copytree(source, tmp_path / "copy")
    root = tmp_path / "copy"
    return DataPaths(root / "data"), root / "config.yaml"


@pytest.fixture(scope="module")
def executed(source):
    """One real run with provenance on the published synthetic year."""
    return run_with_provenance(DataPaths(source / "data"), source / "config.yaml")


def test_cli_publishes_a_verified_result_set_with_scope_and_provenance(source, tmp_path, capsys):
    paths, config = _copy(source, tmp_path)
    argv = ["--data-dir", str(paths.root), "--config", str(config)]
    assert cli.main([*argv, "--execute"]) == 0
    destination = paths.diagnostics / "inference"
    summary = verify_published(destination, paths, config)
    assert {p.name for p in destination.iterdir()} == {*OUTPUT_FILES, MANIFEST, SUMMARY}
    assert not [p for p in destination.parent.iterdir() if p.name.startswith(".inference.")]  # no staging left
    out = capsys.readouterr().out
    assert "Published 12 months" in out and "supported" not in out and "beta" not in out  # administrative only
    # completion marker: technical completion, explicit scope, never a claim about outstanding analyses
    assert summary["status"] == "completed" and summary["scientific_protocol_complete"] is False
    assert "not a statement that the pre-analysis plan" in summary["meaning"]
    assert summary["grid"] == {"first": 200401, "last": 200412, "n_months": 12} and summary["n_years"] == 1
    implemented = " ".join(summary["implemented_analyses"])
    assert "H2:" in implemented and "H3:" in implemented and "raw ret_next" in implemented
    assert "specification B" in implemented and "HAC lags 0 and 12" in implemented
    outstanding = summary["outstanding_analyses"]
    assert set(OUTSTANDING_ROBUSTNESS) <= set(outstanding) and any(a.startswith("H4") for a in outstanding)
    assert not any("H4" in a or "2021" in a for a in summary["implemented_analyses"])
    assert summary["roles"]["H2"] in ("confirmatory (H1 gate passed)", "descriptive (H1 gate not passed)")
    text = json.dumps(summary).lower()
    assert "full study complete" not in text and "all robustness checks passed" not in text
    # manifest: every input file read, with portable paths and its hash
    manifest = _strict_json((destination / MANIFEST).read_text(encoding="utf-8"))
    files = manifest["inputs"]["files"]
    assert set(files) == {f"{YEAR}/{k}" for k in ("conditional_disagreement", "conditional_disagreement_marker",
                                                   "h2_controls", "h2_controls_marker", "target", "universe",
                                                   "panel")}
    assert files[f"{YEAR}/target"]["path"] == f"processed/target/target_{YEAR}.parquet"
    assert files[f"{YEAR}/target"]["sha256"] == output.sha256(paths.target / f"target_{YEAR}.parquet")
    assert manifest["inputs"]["config"]["sha256"] == output.sha256(config)
    # an existing result set is never overwritten without --replace; with it, it is superseded atomically
    before = (destination / SUMMARY).read_text(encoding="utf-8")
    assert cli.main([*argv, "--execute"]) == 2
    assert "already exists" in capsys.readouterr().err and (destination / SUMMARY).read_text() == before
    assert cli.main([*argv, "--execute", "--replace"]) == 0
    verify_published(destination, paths, config)


def test_without_execute_nothing_is_loaded_or_estimated(source, tmp_path, monkeypatch, capsys):
    paths, config = _copy(source, tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("no outcome may be read and no inference run without --execute")

    monkeypatch.setattr(cli, "execute_inference", forbidden)
    monkeypatch.setattr(output, "load_inference_year", forbidden)
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    assert cli.main(["--data-dir", str(paths.root), "--config", str(config)]) == 0
    assert "Dry run: 12 months" in capsys.readouterr().out
    assert not (paths.diagnostics / "inference").exists()
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])
    assert exit_info.value.code == 0


def test_serialised_values_reproduce_the_in_memory_report(executed, tmp_path):
    run, _ = executed
    artifacts = result_artifacts(run)
    assert set(artifacts) == set(OUTPUT_FILES)
    for name, content in artifacts.items():
        output._write_artifact(tmp_path, name, content)
    report = _strict_json((tmp_path / "report.json").read_text(encoding="utf-8"))
    for key, verdict in (("H1", run.report.h1), ("H2", run.report.h2), ("H2_paired_delta", run.report.delta),
                         ("H1_economic_reading", run.report.economic), ("H3", run.report.h3)):
        assert report["decisions"][key]["verdict"] == verdict.verdict
        assert report["decisions"][key]["conditions"] == verdict.conditions
        assert report["decisions"][key]["notes"] == list(verdict.notes)
    beta3 = run.report.coefficients["C.interaction"].results[6]
    assert report["decisions"]["H1"]["basis"]["C.interaction"]["mean"] == beta3.mean
    assert report["decisions"]["H1"]["basis"]["C.interaction"]["p_value"] == beta3.p_value
    assert report["h1_h2_gate"]["open"] == run.report.h2.conditions["gate_open"]
    assert [w["lag"] for w in report["h3_wald"]["by_lag"]] == [6, 0, 12]
    wald = run.report.wald.results[6]
    assert report["h3_wald"]["by_lag"][0]["covariance"] == wald.covariance.tolist()
    assert report["h3_wald"]["order"] == ["H3.interaction", "H3.signal_x_cw"]
    inference = pd.read_csv(tmp_path / "inference.csv", float_precision="round_trip")
    row = inference[(inference["series"] == "C.interaction") & (inference["lag"] == 6)].iloc[0]
    for field in ("mean", "se", "t", "p_value", "ci_low", "ci_high", "hac_variance"):
        assert row[field] == getattr(beta3, field)  # exact round trip with 17 significant digits
    assert row["n_months"] == beta3.n_months and row["n_grid"] == 12 and bool(row["primary"])
    groups = set(inference["group"])
    assert groups == {"coefficient", "bin_local", "paired_h2", "paired_h3", "slope", "raw_return"}
    assert {"Delta", "Delta_S", "C.g(0.1)", "H2.g|controls_at_means(0.9)", "C.linear:20", RAW_SERIES} <= set(
        inference["series"])
    assert len(inference) == 3 * inference["series"].nunique()
    raw = inference[inference["series"] == RAW_SERIES].set_index("lag")
    assert raw.loc[6, "mean"] == run.raw_return.interaction.results[6].mean
    estimates = pd.read_csv(tmp_path / "monthly_estimates.csv", float_precision="round_trip")
    assert len(estimates) == 12 * 6 and set(estimates["specification"]) == {*SPECIFICATIONS, RAW_SPECIFICATION}
    c = estimates[(estimates["specification"] == SPECIFICATION_C) & (estimates["yyyymm"] == 200405)].iloc[0]
    est = run.aligned.estimates[SPECIFICATION_C][4]
    assert est.coefficients is not None and c["interaction"] == est.coefficients.interaction
    assert c["n_obs"] == est.n_obs and c["singular_ratio"] == est.singular_ratio
    bins = pd.read_csv(tmp_path / "monthly_bins.csv", float_precision="round_trip")
    b = bins[(bins["specification"] == SPECIFICATION_C) & (bins["yyyymm"] == 200405) & (bins["bin"] == 3)].iloc[0]
    assert b["linear"] == est.coefficients.linear[2] and b["basis_half_range"] == est.basis.half_range[2]
    assert b["cell_count"] == est.cell_counts[2]
    assert bins[bins["specification"] == SPECIFICATION_B]["quadratic"].isna().all()
    profile = pd.read_csv(tmp_path / "coverage_profile.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(profile, run.coverage_profile, check_dtype=False, check_exact=True)
    means = pd.read_csv(tmp_path / "h2_control_means.csv", float_precision="round_trip")
    assert list(means.columns) == ["yyyymm", *h2.RANKS]
    np.testing.assert_array_equal(means.iloc[0, 1:].to_numpy(dtype=float), run.h2_control_means[200401])


def test_missing_values_zero_coefficients_and_calendar_positions_are_kept_apart(tmp_path):
    """Stub estimates: an exact zero, a specification estimable once, non-estimable months, across a year end."""
    grid = month_range(202311, 202402)
    rng = np.random.default_rng(4)
    basis = _basis(rng)
    calls = {"n": 0}

    def estimate(**kwargs):
        i = calls["n"]
        calls["n"] += 1
        out = {}
        for spec in SPECIFICATIONS:
            interaction = 0.0 if (spec == SPECIFICATION_C and i == 1) else rng.normal(-0.01, 0.01)
            estimable = not (spec == SPECIFICATION_B and i != 0) and not (i == 2 and spec != SPECIFICATION_C)
            out[spec] = _estimate(spec, estimable, basis, _coefficients(rng, spec, interaction, rng.normal()))
        return out

    def estimate_raw(*args):
        return _estimate(SPECIFICATION_C, False, basis, None, reason="no member of U_t has a finite outcome")

    years = [(2023, _tiny(month_range(202311, 202312))), (2024, _tiny(month_range(202401, 202402)))]
    run = run_inference(grid, years, estimate, estimate_raw)
    artifacts = result_artifacts(run)
    for name, content in artifacts.items():
        output._write_artifact(tmp_path, name, content)
    series = pd.read_csv(tmp_path / "series.csv", float_precision="round_trip")
    c = series[series["series"] == "C.interaction"]
    assert c["yyyymm"].tolist() == grid and c["estimable"].tolist() == [True] * 4
    assert c["value"].iloc[1] == 0.0  # an estimated zero stays zero
    b = series[series["series"] == "B.interaction"]
    assert b["estimable"].tolist() == [True, False, False, False] and b["value"].iloc[1:].isna().all()
    h2s = series[series["series"] == "H2.interaction"]
    assert h2s["estimable"].tolist() == [True, True, False, True] and np.isnan(h2s["value"].iloc[2])
    inference = pd.read_csv(tmp_path / "inference.csv", float_precision="round_trip")
    failed = inference[(inference["series"] == "B.interaction")]
    assert (failed["status"] == "failed").all() and failed["mean"].isna().all()
    assert failed["failure"].str.contains("at least 2 estimable months").all()
    raw = inference[inference["series"] == RAW_SERIES]
    assert (raw["status"] == "failed").all() and raw["p_value"].isna().all()
    estimates = pd.read_csv(tmp_path / "monthly_estimates.csv", float_precision="round_trip")
    bad = estimates[~estimates["estimable"]]
    assert bad["condition_number"].isna().all() and (~bad["condition_number_finite"]).all()
    assert bad["interaction"].isna().all() and (bad["n_obs"] == 0).all()
    assert json.loads(bad.iloc[0]["reasons"])[0] in ("1 cells with fewer than 30 observations",
                                                     "no member of U_t has a finite outcome")
    report = _strict_json((tmp_path / "report.json").read_text(encoding="utf-8"))  # no NaN or Infinity literal
    assert report["non_estimable"][SPECIFICATION_B] == {str(m): ["1 cells with fewer than 30 observations"]
                                                       for m in grid[1:]}
    assert set(report["non_estimable"][RAW_SPECIFICATION]) == {str(m) for m in grid}
    assert report["failures"]["B.interaction"]["6"].startswith("at least 2 estimable months")
    assert report["failures"][RAW_SERIES]["6"].startswith("at least 2 estimable months")
    assert report["grid"] == {"first": 202311, "last": 202402, "n_months": 4}


def test_publication_is_atomic_and_never_corrupts_an_existing_result(source, executed, tmp_path, monkeypatch):
    paths, config = _copy(source, tmp_path)
    run, inputs = executed
    destination = paths.diagnostics / "inference"
    publish(run, inputs, destination, paths, config)
    published = {p.name: p.read_bytes() for p in destination.iterdir()}
    original = output._write_artifact

    def failing(directory, name, content):
        if name == "series.csv":
            raise OSError("disk full")
        original(directory, name, content)

    monkeypatch.setattr(output, "_write_artifact", failing)
    with pytest.raises(OSError, match="disk full"):
        publish(run, inputs, destination, paths, config, replace=True)
    assert {p.name: p.read_bytes() for p in destination.iterdir()} == published  # untouched
    verify_published(destination, paths, config)
    fresh = paths.diagnostics / "fresh"
    with pytest.raises(OSError):
        publish(run, inputs, fresh, paths, config)
    assert not fresh.exists()
    assert not [p for p in paths.diagnostics.iterdir() if p.name.startswith(".")]  # staging removed
    monkeypatch.setattr(output, "_write_artifact", original)
    with pytest.raises(PublicationError, match="already exists"):
        publish(run, inputs, destination, paths, config)


def test_verification_rejects_modified_missing_or_extra_artifacts_and_changed_inputs(source, executed, tmp_path):
    paths, config = _copy(source, tmp_path)
    run, inputs = executed
    destination = paths.diagnostics / "inference"
    publish(run, inputs, destination, paths, config)

    def rejected(error, message):
        with pytest.raises(error, match=message):
            verify_published(destination, paths, config)

    inference_csv = destination / "inference.csv"
    content = inference_csv.read_bytes()
    inference_csv.write_bytes(content.replace(b"estimated", b"estimatee", 1))
    rejected(PublicationError, r"artifacts differ from the manifest: \['inference.csv'\]")
    inference_csv.write_bytes(content)
    inference_csv.unlink()
    rejected(PublicationError, r"missing \['inference.csv'\]")
    inference_csv.write_bytes(content)
    (destination / "notes.txt").write_text("x")
    rejected(PublicationError, r"unexpected \['notes.txt'\]")
    (destination / "notes.txt").unlink()
    manifest = destination / MANIFEST
    manifest_text = manifest.read_text(encoding="utf-8")
    manifest.write_text(manifest_text.replace('"n_months": 12', '"n_months": 13'), encoding="utf-8")
    rejected(PublicationError, "manifest does not match the completion marker")
    manifest.write_text(manifest_text, encoding="utf-8")
    verify_published(destination, paths, config)
    # inputs: a modified target file, a missing marker, a modified configuration
    target_file = paths.target / f"target_{YEAR}.parquet"
    target_bytes = target_file.read_bytes()
    frame = pd.read_parquet(target_file)
    frame.loc[frame.index[0], "ret_next"] = 0.5
    frame.to_parquet(target_file, index=False)
    rejected(DataValidationError, rf"\['{YEAR}/target'\]")
    target_file.write_bytes(target_bytes)
    marker = paths.conditional_disagreement / f"{YEAR}.json"
    marker_text = marker.read_text(encoding="utf-8")
    marker.unlink()
    rejected(DataValidationError, rf"\['{YEAR}/conditional_disagreement_marker'\]")
    marker.write_text(marker_text, encoding="utf-8")
    config.write_text(config.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    rejected(DataValidationError, "configuration file is missing or differs")


def test_sources_changing_during_the_run_block_publication(source, tmp_path, monkeypatch):
    paths, config = _copy(source, tmp_path)
    target_file = paths.target / f"target_{YEAR}.parquet"
    real_load = output.load_inference_year

    def load_then_change(p, year):
        frame = real_load(p, year)
        changed = pd.read_parquet(target_file)
        changed.loc[changed.index[0], "ret_next"] = -0.3
        changed.to_parquet(target_file, index=False)
        return frame

    monkeypatch.setattr(output, "load_inference_year", load_then_change)
    with pytest.raises(DataValidationError, match="changed while they were read"):
        execute_inference(paths, config)
    assert not (paths.diagnostics / "inference").exists()
    monkeypatch.setattr(output, "load_inference_year", real_load)
    # a change after the year was loaded, during estimation, is caught by the check before publication
    paths, config = _copy(source, tmp_path / "second")
    target_file = paths.target / f"target_{YEAR}.parquet"
    changed = []

    def raw_then_change(*args):
        if not changed:  # once, after the year was loaded and hashed, while the months are estimated
            changed.append(True)
            target_file.write_bytes(target_file.read_bytes() + b"\0")
        return estimate_h1_month(*args)

    with pytest.raises(DataValidationError, match=rf"changed since they were read: \['{YEAR}/target'\]"):
        execute_inference(paths, config, estimate_raw=raw_then_change)
    assert changed and not (paths.diagnostics / "inference").exists()
    assert not [p for p in paths.diagnostics.iterdir() if p.name.startswith(".")]


def _snapshot(root):
    """Every file and directory under root, with the bytes of each file."""
    return {p.relative_to(root).as_posix(): (p.read_bytes() if p.is_file() else None) for p in root.rglob("*")}


def _link(link, target):
    """A directory symbolic link, or a junction where symbolic links need privileges (Windows)."""
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        winapi = pytest.importorskip("_winapi")
        winapi.CreateJunction(str(target), str(link))


def test_replacement_of_unrelated_or_protected_directories_is_refused_without_touching_a_file(
        source, executed, tmp_path, monkeypatch, capsys):
    paths, config = _copy(source, tmp_path)
    run, inputs = executed
    unrelated = paths.diagnostics / "notes"  # a populated research directory under the default parent
    (unrelated / "sub").mkdir(parents=True)
    (unrelated / "draft.md").write_text("keep me")
    (unrelated / "sub" / "table.csv").write_text("a,b\n1,2\n")
    lookalike = paths.diagnostics / "lookalike"  # a genuine result set to which a research file was added
    publish(run, inputs, lookalike, paths, config)
    (lookalike / "notes.txt").write_text("keep me too")
    link = tmp_path / "link"  # outside the snapshot, pointing at the data root
    _link(link, paths.root)
    before = _snapshot(tmp_path / "copy")

    def forbidden(*args, **kwargs):
        raise AssertionError("an unsafe destination must be refused before any outcome is read")

    monkeypatch.setattr(output, "load_inference_year", forbidden)
    refusals = {
        unrelated: "not a verifiable published inference result",
        lookalike: "not a verifiable published inference result",
        paths.root: "protected research path",
        paths.root.parent: "protected research path",  # an ancestor of the data root, holding the configuration
        paths.diagnostics: "protected research path",
        paths.target.parent: "protected research path",  # data/processed, above every input directory
        paths.target: "protected research path",
        paths.target / "inference": "inside a pipeline data directory",
        link: "symbolic link or junction",
    }
    for destination, message in refusals.items():
        with pytest.raises(PublicationError, match=message):
            publish(run, inputs, destination, paths, config, replace=True)
        with pytest.raises(PublicationError, match=message):
            execute_inference(paths, config, destination, replace=True)
        argv = ["--data-dir", str(paths.root), "--config", str(config), "--output-dir", str(destination)]
        assert cli.main([*argv, "--execute", "--replace"]) == 2
        assert message in capsys.readouterr().err
    assert _snapshot(tmp_path / "copy") == before  # not a single file created, changed, moved or deleted
    assert (link / "processed").is_dir()


def test_a_valid_published_result_set_can_be_replaced_in_a_custom_directory(source, executed, tmp_path):
    paths, config = _copy(source, tmp_path)
    run, inputs = executed
    parent = tmp_path / "results"
    parent.mkdir()
    (parent / "readme.txt").write_text("unrelated sibling")
    destination = parent / "inference_v1"  # a safe new custom destination outside the data root
    first = publish(run, inputs, destination, paths, config)
    assert verify_published(destination, paths, config) == first
    second = publish(run, inputs, destination, paths, config, replace=True)
    assert verify_published(destination, paths, config) == second and second["grid"] == first["grid"]
    assert sorted(p.name for p in parent.iterdir()) == ["inference_v1", "readme.txt"]  # no staging or backup left
    assert (parent / "readme.txt").read_text() == "unrelated sibling"


def test_a_failed_final_rename_restores_the_previous_result_set(source, executed, tmp_path, monkeypatch):
    paths, config = _copy(source, tmp_path)
    run, inputs = executed
    destination = paths.diagnostics / "inference"
    publish(run, inputs, destination, paths, config)
    (paths.diagnostics / "other").mkdir()
    (paths.diagnostics / "other" / "kept.txt").write_text("unrelated")
    before = _snapshot(paths.diagnostics)
    real_replace = os.replace
    moves = []

    def failing_final_rename(src, dst):
        moves.append((os.path.basename(src), os.path.basename(dst)))
        if ".staging-" in os.path.basename(src):
            assert not destination.exists()  # the previous result set has already been moved aside
            raise OSError("rename failed")
        real_replace(src, dst)

    monkeypatch.setattr(output.os, "replace", failing_final_rename)
    with pytest.raises(OSError, match="rename failed"):
        publish(run, inputs, destination, paths, config, replace=True)
    assert moves[0][0] == "inference" and moves[0][1].startswith(".inference.previous-")
    assert moves[1][1] == "inference" and moves[2] == (moves[0][1], "inference")  # failed, then restored
    assert _snapshot(paths.diagnostics) == before  # restored byte for byte; nothing else created or deleted
    verify_published(destination, paths, config)
    # if even the restore fails, the previous result set is kept at its backup path, never deleted
    moves.clear()

    def failing_both(src, dst):
        moves.append((os.path.basename(src), os.path.basename(dst)))
        if len(moves) > 1:
            raise OSError("rename failed")
        real_replace(src, dst)

    monkeypatch.setattr(output.os, "replace", failing_both)
    with pytest.raises(PublicationError, match=r"could not be restored; it is kept at \.inference\.previous-"):
        publish(run, inputs, destination, paths, config, replace=True)
    monkeypatch.setattr(output.os, "replace", real_replace)
    backup = paths.diagnostics / moves[0][1]
    assert not destination.exists()
    assert {p.name for p in paths.diagnostics.iterdir()} == {"summary.json", "other", backup.name}  # no staging
    assert {f"inference/{k}": v for k, v in _snapshot(backup).items()} == {
        k: v for k, v in before.items() if k.startswith("inference/")}
    verify_published(backup, paths, config)


def test_link_detection_works_without_the_python_3_12_path_methods(source, tmp_path, monkeypatch):
    """As on Python 3.11: no Path.is_junction nor os.path.isjunction; links and junctions are still refused."""
    paths, config = _copy(source, tmp_path)
    for cls in type(tmp_path).__mro__:
        if "is_junction" in vars(cls):
            monkeypatch.delattr(cls, "is_junction")
    monkeypatch.delattr(os.path, "isjunction", raising=False)
    assert not hasattr(tmp_path, "is_junction") and not hasattr(os.path, "isjunction")
    plain = tmp_path / "plain"
    plain.mkdir()
    assert not output._is_link(plain) and not output._is_link(tmp_path / "missing")
    output.check_destination(tmp_path / "new", paths, config)  # an ordinary destination is accepted
    output.check_destination(paths.diagnostics / "inference", paths, config)
    links = []
    if os.name == "nt":  # a junction needs no privilege on Windows
        winapi = pytest.importorskip("_winapi")
        winapi.CreateJunction(str(paths.root), str(tmp_path / "junction"))
        links.append(tmp_path / "junction")
    try:
        (tmp_path / "symlink").symlink_to(paths.root, target_is_directory=True)
        links.append(tmp_path / "symlink")
    except OSError:
        pass  # symbolic links need a privilege on this system
    if not links:
        pytest.skip("neither a junction nor a symbolic link can be created here")
    for link in links:
        assert output._is_link(link)
        with pytest.raises(PublicationError, match="symbolic link or junction"):
            output.check_destination(link, paths, config, replace=True)


def test_a_destination_inside_a_published_result_set_is_refused(source, executed, tmp_path, monkeypatch):
    paths, config = _copy(source, tmp_path)
    run, inputs = executed
    destination = paths.diagnostics / "inference"
    publish(run, inputs, destination, paths, config)
    before = _snapshot(destination)

    def forbidden(*args, **kwargs):
        raise AssertionError("a nested destination must be refused before any outcome is read")

    monkeypatch.setattr(output, "load_inference_year", forbidden)
    for nested in (destination / "another_run", destination / "a" / "b"):
        with pytest.raises(PublicationError, match="inside a published inference result set"):
            execute_inference(paths, config, nested)
        with pytest.raises(PublicationError, match="inside a published inference result set"):
            publish(run, inputs, nested, paths, config, replace=True)
        argv = ["--data-dir", str(paths.root), "--config", str(config), "--output-dir", str(nested), "--execute"]
        assert cli.main(argv) == 2
    assert _snapshot(destination) == before  # the parent result set is untouched, byte for byte
    verify_published(destination, paths, config)
    # a sibling custom destination and a legitimate replacement are still published
    sibling = paths.diagnostics / "inference_alternative"
    publish(run, inputs, sibling, paths, config)
    verify_published(sibling, paths, config)
    publish(run, inputs, destination, paths, config, replace=True)
    verify_published(destination, paths, config)
    assert not [p for p in paths.diagnostics.iterdir() if p.name.startswith(".")]


def _reseal(destination, frame):
    """Rewrite monthly_summary.csv and make the manifest and the completion marker consistent with it again."""
    output._write_artifact(destination, "monthly_summary.csv", frame)
    manifest = json.loads((destination / MANIFEST).read_text(encoding="utf-8"))
    manifest["outputs"]["monthly_summary.csv"] = output.sha256(destination / "monthly_summary.csv")
    (destination / MANIFEST).write_text(output._dumps(manifest), encoding="utf-8")
    summary = json.loads((destination / SUMMARY).read_text(encoding="utf-8"))
    summary["manifest_sha256"] = output.sha256(destination / MANIFEST)
    (destination / SUMMARY).write_text(output._dumps(summary), encoding="utf-8")


def test_verification_rejects_a_structurally_invalid_calendar_with_consistent_hashes(source, executed, tmp_path):
    paths, config = _copy(source, tmp_path)
    run, inputs = executed
    destination = paths.diagnostics / "inference"
    publish(run, inputs, destination, paths, config)
    original = pd.read_csv(destination / "monthly_summary.csv", float_precision="round_trip")
    swapped = original.copy()
    swapped.loc[[2, 3], "yyyymm"] = [200404, 200403]
    duplicated = original.copy()
    duplicated.loc[3, "yyyymm"] = 200403
    invalid = original.copy()
    invalid.loc[5, "yyyymm"] = 200413
    fractional = original.astype({"yyyymm": "float64"})
    fractional.loc[5, "yyyymm"] = 200406.5
    # each keeps the first month, the last month and the row count, all the previous check looked at
    for frame, message in ((swapped, "not the consecutive recorded grid"), (duplicated, "consecutive"),
                           (invalid, "not a valid yyyymm month"), (fractional, "not an integer yyyymm month")):
        assert (frame["yyyymm"].iloc[0], frame["yyyymm"].iloc[-1], len(frame)) == (200401, 200412, 12)
        _reseal(destination, frame)
        with pytest.raises(PublicationError, match=rf"calendar metadata is inconsistent \(.*{message}"):
            verify_published(destination)
    _reseal(destination, original)
    verify_published(destination, paths, config)


def test_verification_rejects_a_completion_marker_that_misstates_the_scope(source, executed, tmp_path):
    paths, config = _copy(source, tmp_path)
    run, inputs = executed
    destination = paths.diagnostics / "inference"
    publish(run, inputs, destination, paths, config)
    marker = destination / SUMMARY  # not covered by the manifest: checked against the module and report.json
    original = json.loads(marker.read_text(encoding="utf-8"))
    roles = ("confirmatory (H1 gate passed)", "descriptive (H1 gate not passed)")
    other_role = roles[1] if original["roles"]["H2"] == roles[0] else roles[0]
    edits = {
        "scientific_protocol_complete": original | {"scientific_protocol_complete": True},
        "primary_outcome": original | {"primary_outcome": "ret_next"},
        "primary_lag": original | {"primary_lag": 0},
        "n_years": original | {"n_years": 40},
        "implemented_analyses": original | {"implemented_analyses": [*original["implemented_analyses"], "H4"]},
        "outstanding_analyses": original | {"outstanding_analyses": original["outstanding_analyses"][1:]},
        "roles": original | {"roles": original["roles"] | {"H2": other_role}},  # contradicts the recorded gate
        "meaning": {k: v for k, v in original.items() if k != "meaning"},  # a missing field
        "note": original | {"note": "full study complete"},  # an unexpected field
    }
    for field, summary in edits.items():
        marker.write_text(output._dumps(summary), encoding="utf-8")
        with pytest.raises(PublicationError, match=rf"misstates the result set.*'{field}'"):
            verify_published(destination)
    marker.write_text(output._dumps(original), encoding="utf-8")
    verify_published(destination, paths, config)


def test_destinations_inside_the_repository_must_be_in_its_ignored_output_areas(tmp_path, monkeypatch):
    repo = tmp_path / "repo"  # stands for the repository root, so that the real data/ is never touched
    for directory in ("src/disagreement", "docs", "configs", "experiments/results", "data"):
        (repo / directory).mkdir(parents=True)
    config = repo / "configs" / "baseline.yaml"
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(output, "REPO_ROOT", repo)
    paths = DataPaths(repo / "data")
    before = _snapshot(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("an unsafe destination must be refused before any outcome is read")

    for destination in (repo / "src" / "disagreement" / "results", repo / "docs" / "inference", repo / "inference"):
        with pytest.raises(PublicationError, match="outside its Git-ignored output areas"):
            execute_inference(paths, config, destination, runner=forbidden)
    assert _snapshot(tmp_path) == before  # refused before anything was read or created
    for destination in (output.default_output_dir(paths), repo / "experiments" / "results" / "inference",
                        tmp_path / "elsewhere" / "inference"):
        output.check_destination(destination, paths, config)  # the default, an ignored area, outside the repository
