"""Serialisation, input provenance and atomic publication of the frozen inference run.

The in-memory ``InferenceRun`` of ``disagreement.inference.orchestrate`` is the
authoritative result; this module only writes its contents. Nothing is
estimated, tested or decided here.

Published files (``data/diagnostics/inference/`` by default):

* ``report.json``: the five frozen decisions (H1, H2 behind the fixed-sequence
  gate, the paired Delta reading, the economic H1 reading, H3) with their
  conditions, notes and the L = 6 statistics they rest on, the H1 -> H2
  gate, the H3 joint Wald test at every lag, the non-estimable months of every
  specification and of the raw-return C, and every inference failure;
* ``inference.csv``: one row per (series, lag) for every scalar Fama-MacBeth
  inference of the report and of the raw-return robustness: mean, HAC
  standard error, t, p, 95% interval, T, N, lag, HAC variance, or the reason
  it could not be computed;
* ``series.csv``: every calendar-aligned monthly series (value and
  estimability mask on the full grid);
* ``monthly_estimates.csv``: one row per (month, specification), the five
  primary specifications and the raw-return C: estimability, reasons, sample
  size, singular-value diagnostics, beta_2, beta_3 and the named extra terms;
* ``monthly_bins.csv``: one row per (month, specification, bin): cell size,
  distinct S^c values, bin-local coefficients and the full-universe basis
  constants c_b, h_b, mean_b(z^2);
* ``monthly_summary.csv``, ``coverage_monthly.csv``, ``coverage_profile.csv``
  and ``h2_control_means.csv``: the descriptive sample and coverage tables;
* ``manifest.json``: the SHA-256 of every input file read (configuration;
  per year, the conditional-disagreement and H2-control Parquet files and
  markers, the target file, and the universe and panel files that the H2
  provenance check verifies) and of every artifact above;
* ``summary.json``: written last, it marks that the core inference run
  completed; it is not a statement that the pre-analysis plan was executed.

Unavailable numbers are empty in CSV and ``null`` in JSON, always next to a
status or reason; JSON never contains NaN or infinity. CSV floats are written
with 17 significant digits, which round-trips float64 exactly when read with
an exact parser (pandas: ``float_precision="round_trip"``). No stock-level row is
written. The result set is written to a staging directory beside the
destination and published by renaming it, after the input files have been
checked unchanged since they were read.

Destinations are checked fail-closed (``check_destination``): never a symbolic
link or junction, never the data root, the diagnostics directory, the
repository, the configuration file or a pipeline data directory, nor any of
their ancestors, and never inside a pipeline data directory or inside another
result set (a directory holding both a manifest and a completion marker). An
existing destination is replaced only if it is a verifiable published inference
result set made of regular files; anything else is refused without being
touched.

Replacement uses two renames: the existing result set is moved aside, then the
staging directory is renamed into place. A handled failure of the second rename
restores the previous result set; if even that fails, it is kept at its
``.previous-`` path and never deleted. Between the two renames the destination
is briefly absent, and a process crash there (or before the clean-up of the
staging directory) is not recovered automatically: the previous result set
then remains at its ``.previous-`` path for a manual restore.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import shutil
import stat
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from disagreement.controls import h2
from disagreement.controls.h2 import RANKS
from disagreement.data.config import REPO_ROOT, DataPaths
from disagreement.data.validation import DataValidationError
from disagreement.inference.fama_macbeth import FamaMacBethResult, WaldResult
from disagreement.inference.inputs import load_inference_year
from disagreement.inference.monthly import (
    H2_EXTRAS,
    H3_EXTRAS,
    SPECIFICATION_C,
    MonthlyEstimate,
    estimate_h1_month,
    estimate_month_specifications,
)
from disagreement.inference.orchestrate import (
    OUTSTANDING_ROBUSTNESS,
    PRIMARY_OUTCOME,
    RAW_OUTCOME,
    RAW_SERIES,
    InferenceRun,
    MonthEstimator,
    RawEstimator,
    baseline_calendar,
    run_inference,
)
from disagreement.inference.report import ALPHA, PRIMARY_LAG, Verdict
from disagreement.inference.series import LAGS, SPECIFICATIONS, Inference, _yyyymm
from disagreement.walkforward.calendar import month_range

SCHEMA_VERSION = 1
OUTPUT_FILES = (
    "report.json",
    "inference.csv",
    "series.csv",
    "monthly_estimates.csv",
    "monthly_bins.csv",
    "monthly_summary.csv",
    "coverage_monthly.csv",
    "coverage_profile.csv",
    "h2_control_means.csv",
)
MANIFEST = "manifest.json"
SUMMARY = "summary.json"
RAW_SPECIFICATION = f"{SPECIFICATION_C}|{RAW_OUTCOME}"
COMPLETION_MEANING = ("the core Step 22D-D3 inference run completed successfully; this is not a statement that "
                      "the pre-analysis plan, its robustness analyses or H4 were executed")
IMPLEMENTED_ANALYSES = (
    "H1: specification C on target_rank (primary, confirmatory)",
    "H2: C plus the three H2 controls on the H2 sample, behind the H1 -> H2 fixed-sequence gate",
    "H2 paired differences Delta and Delta_S on the common estimable months",
    "H3: C plus CW and S^c x CW on the H1 sample (secondary, intersection-union test, joint Wald complement)",
    "H1 and H2 marginal signal slopes at CD^B = 0.1, 0.5, 0.9 (H2 primary at X^c = 0, descriptive at means)",
    "specification B (H1 specification robustness)",
    "raw ret_next outcome for specification C (H1 outcome robustness)",
    "HAC lags 0 and 12 (sensitivity only; decisions use L = 6)",
    "outcome coverage and sample-size diagnostics (descriptive)",
)
OUTSTANDING_ANALYSES = (*OUTSTANDING_ROBUSTNESS,
                        "H4: disagreement-based filter and its secondary backtesting extension")
CHUNK = 1 << 20
NAME_SURROGATE = 0x20000000  # Windows reparse-tag bit of redirecting reparse points (IsReparseTagNameSurrogate)


class PublicationError(RuntimeError):
    """The result set could not be published; no completion marker was written."""


# ----------------------------------------------------------------------------------------------------------------
# Value conversion


def _number(value) -> float | int | None:
    """A JSON-safe number: None for missing or non-finite values, Python types otherwise."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("a Boolean is not a number")
    if isinstance(value, (int, np.integer)):
        return int(value)
    x = float(value)
    return x if math.isfinite(x) else None


def _scalar(result: FamaMacBethResult) -> dict:
    return {
        "mean": _number(result.mean), "se": _number(result.se), "t": _number(result.t),
        "p_value": _number(result.p_value), "ci_low": _number(result.ci_low), "ci_high": _number(result.ci_high),
        "n_months": result.n_months, "n_grid": result.n_grid, "lag": result.lag,
        "hac_variance": _number(result.hac_variance),
    }


def _wald(result: WaldResult) -> dict:
    return {
        "mean": [_number(v) for v in result.mean],
        "covariance": [[_number(v) for v in row] for row in result.covariance],
        "wald": _number(result.wald), "df": result.df, "p_value": _number(result.p_value),
        "n_months": result.n_months, "n_grid": result.n_grid, "lag": result.lag,
    }


def _statistic(result: FamaMacBethResult | WaldResult | None) -> dict | None:
    if result is None:
        return None
    return _wald(result) if isinstance(result, WaldResult) else _scalar(result)


def _verdict(verdict: Verdict) -> dict:
    return {
        "hypothesis": verdict.hypothesis,
        "verdict": verdict.verdict,
        "conditions": dict(verdict.conditions),
        "notes": list(verdict.notes),
        "basis": {name: _statistic(result) for name, result in verdict.basis.items()},
    }


# ----------------------------------------------------------------------------------------------------------------
# Tables


def _inference_groups(run: InferenceRun) -> dict[str, dict[str, Inference]]:
    report = run.report
    return {
        "coefficient": report.coefficients,
        "bin_local": report.bin_local,
        "paired_h2": report.paired_h2,
        "paired_h3": report.paired_h3,
        "slope": report.slopes,
        "raw_return": {RAW_SERIES: run.raw_return.interaction},
    }


def _inference_table(run: InferenceRun) -> pd.DataFrame:
    rows = []
    for group, inferences in _inference_groups(run).items():
        for name, inference in inferences.items():
            for lag in LAGS:
                row: dict = {"group": group, "series": name, "lag": lag, "primary": lag == PRIMARY_LAG}
                if lag in inference.results:
                    row |= {"status": "estimated", **_scalar(inference.results[lag]), "failure": None}
                else:
                    row |= {"status": "failed", "failure": inference.failures.get(lag, f"no result at L = {lag}")}
                rows.append(row)
    columns = ["group", "series", "lag", "primary", "status", "mean", "se", "t", "p_value", "ci_low", "ci_high",
               "n_months", "n_grid", "hac_variance", "failure"]
    table = pd.DataFrame(rows).reindex(columns=columns)
    for column in ("n_months", "n_grid"):
        table[column] = table[column].astype("Int64")
    return table


def _series_table(run: InferenceRun) -> pd.DataFrame:
    frames = []
    for group, inferences in _inference_groups(run).items():
        for name, inference in inferences.items():
            s = inference.series
            frames.append(pd.DataFrame({"group": group, "series": name, "yyyymm": s.grid,
                                        "estimable": s.estimable, "value": np.where(s.estimable, s.values, np.nan)}))
    return pd.concat(frames, ignore_index=True)


def _estimates(run: InferenceRun) -> list[tuple[int, str, str, MonthlyEstimate]]:
    out = []
    for i, month in enumerate(run.grid.tolist()):
        for spec in SPECIFICATIONS:
            out.append((month, spec, PRIMARY_OUTCOME, run.aligned.estimates[spec][i]))
        out.append((month, RAW_SPECIFICATION, RAW_OUTCOME, run.raw_return.estimates[i]))
    return out


def _monthly_tables(run: InferenceRun) -> tuple[pd.DataFrame, pd.DataFrame]:
    extras = (*H2_EXTRAS, *H3_EXTRAS)
    estimate_rows, bin_rows = [], []
    for month, spec, outcome, e in _estimates(run):
        c = e.coefficients
        row = {
            "yyyymm": month, "specification": spec, "outcome": outcome, "estimable": e.estimable,
            "reasons": json.dumps(list(e.reasons)), "n_obs": e.n_obs, "singular_ratio": e.singular_ratio,
            "condition_number": e.condition_number if math.isfinite(e.condition_number) else np.nan,
            "condition_number_finite": math.isfinite(e.condition_number),
            "disagreement": c.disagreement if c is not None else np.nan,
            "interaction": c.interaction if c is not None else np.nan,
        }
        for name in extras:
            row[name] = c.extra[name] if c is not None and name in c.extra else np.nan
        estimate_rows.append(row)
        for b in range(e.basis.n_bins):
            bin_rows.append({
                "yyyymm": month, "specification": spec, "bin": b + 1,
                "cell_count": int(e.cell_counts[b]), "cell_distinct": int(e.cell_distinct[b]),
                "intercept": c.intercept[b] if c is not None else np.nan,
                "linear": c.linear[b] if c is not None else np.nan,
                "quadratic": c.quadratic[b] if c is not None and c.quadratic.size else np.nan,
                "basis_centre": e.basis.centre[b], "basis_half_range": e.basis.half_range[b],
                "basis_mean_z2": e.basis.mean_z2[b],
            })
    return pd.DataFrame(estimate_rows), pd.DataFrame(bin_rows)


def _report_json(run: InferenceRun) -> dict:
    report = run.report
    wald = []
    for lag in LAGS:
        if lag in report.wald.results:
            wald.append({"lag": lag, "status": "estimated", **_wald(report.wald.results[lag]), "failure": None})
        else:
            wald.append({"lag": lag, "status": "failed", "failure": report.wald.failures.get(lag)})
    non_estimable = {spec: {str(m): list(r) for m, r in months.items()}
                     for spec, months in report.non_estimable.items()}
    non_estimable[RAW_SPECIFICATION] = {str(m): list(r) for m, r in run.raw_return.non_estimable.items()}
    failures = {name: {str(lag): reason for lag, reason in by_lag.items()} for name, by_lag in report.failures.items()}
    if run.raw_return.interaction.failures:
        failures[RAW_SERIES] = {str(lag): r for lag, r in run.raw_return.interaction.failures.items()}
    return {
        "schema_version": SCHEMA_VERSION,
        "primary_outcome": PRIMARY_OUTCOME,
        "primary_lag": PRIMARY_LAG,
        "sensitivity_lags": [lag for lag in LAGS if lag != PRIMARY_LAG],
        "alpha": ALPHA,
        "grid": {"first": int(run.grid[0]), "last": int(run.grid[-1]), "n_months": int(run.grid.size)},
        "decisions": {
            "H1": _verdict(report.h1),
            "H2": _verdict(report.h2),
            "H2_paired_delta": _verdict(report.delta),
            "H1_economic_reading": _verdict(report.economic),
            "H3": _verdict(report.h3),
        },
        "h1_h2_gate": {
            "open": report.h2.conditions.get("gate_open"),
            "rule": "H2 is confirmatory only if H1 is supported in the negative direction at L = 6",
        },
        "h3_wald": {"order": ["H3.interaction", "H3.signal_x_cw"], "by_lag": wald},
        "raw_return_robustness": {
            "label": "H1 outcome robustness: specification C on the raw ret_next; outside every verdict",
            "series": RAW_SERIES, "specification": RAW_SPECIFICATION,
        },
        "non_estimable": non_estimable,
        "failures": failures,
    }


def result_artifacts(run: InferenceRun) -> dict[str, pd.DataFrame | dict]:
    """Every published table and JSON document of an InferenceRun, deterministic, without recomputation."""
    estimates, bins = _monthly_tables(run)
    means = pd.DataFrame([(m, *v) for m, v in sorted(run.h2_control_means.items())], columns=["yyyymm", *RANKS])
    return {
        "report.json": _report_json(run),
        "inference.csv": _inference_table(run),
        "series.csv": _series_table(run),
        "monthly_estimates.csv": estimates,
        "monthly_bins.csv": bins,
        "monthly_summary.csv": run.monthly_summary,
        "coverage_monthly.csv": run.coverage_monthly,
        "coverage_profile.csv": run.coverage_profile,
        "h2_control_means.csv": means,
    }


def _dumps(obj) -> str:
    return json.dumps(obj, indent=2, allow_nan=False)  # NaN or infinity here is a programming error


def _write_artifact(directory: Path, name: str, content: pd.DataFrame | dict) -> None:
    path = directory / name
    if isinstance(content, dict):
        path.write_text(_dumps(content), encoding="utf-8")
    else:
        content.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")


# ----------------------------------------------------------------------------------------------------------------
# Input provenance


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def year_inputs(paths: DataPaths, oos_year: int) -> dict[str, tuple[str, Path]]:
    """The files read for one OOS year: key -> (role, path); the H2 provenance inputs included."""
    transitive = h2.input_files(paths, oos_year)  # universe, panel, conditional disagreement (verified by H2)
    return {
        f"{oos_year}/conditional_disagreement": ("conditional disagreement",
                                                 paths.conditional_disagreement / f"{oos_year}.parquet"),
        f"{oos_year}/conditional_disagreement_marker": ("conditional disagreement marker",
                                                        paths.conditional_disagreement / f"{oos_year}.json"),
        f"{oos_year}/h2_controls": ("H2 controls", paths.h2_controls / f"{oos_year}.parquet"),
        f"{oos_year}/h2_controls_marker": ("H2 controls marker", paths.h2_controls / f"{oos_year}.json"),
        f"{oos_year}/target": ("target", paths.target / f"target_{oos_year}.parquet"),
        f"{oos_year}/universe": ("H2 provenance input (universe)", transitive["universe"]),
        f"{oos_year}/panel": ("H2 provenance input (panel)", transitive["panel"]),
    }


def _portable(path: Path, root: Path) -> str:
    for base in (root, REPO_ROOT):
        try:
            return path.resolve().relative_to(base.resolve()).as_posix()
        except ValueError:
            continue
    return path.name


def _hashes(files: Mapping[str, tuple[str, Path]], label: str) -> dict[str, str]:
    out = {}
    for key, (_, path) in files.items():
        if not path.is_file():
            raise DataValidationError(f"{label}: input {key} is missing.")
        out[key] = sha256(path)
    return out


def run_with_provenance(
    paths: DataPaths,
    config_path: Path,
    estimate: MonthEstimator = estimate_month_specifications,
    estimate_raw: RawEstimator = estimate_h1_month,
) -> tuple[InferenceRun, dict]:
    """The frozen baseline run, with the SHA-256 of every input file taken before and after each year's load.

    A file that changes while its year is loaded is an error. The returned input record is checked again by
    `check_inputs` before publication.
    """
    config_hash = sha256(config_path)
    grid, years = baseline_calendar(config_path)
    record: dict = {"config": {"path": _portable(config_path, paths.root), "sha256": config_hash}, "files": {}}

    def load(oos_year: int) -> pd.DataFrame:
        files = year_inputs(paths, oos_year)
        before = _hashes(files, f"{oos_year}")
        frame = load_inference_year(paths, oos_year)
        after = _hashes(files, f"{oos_year}")
        changed = sorted(k for k in files if before[k] != after[k])
        if changed:
            raise DataValidationError(f"{oos_year}: input files changed while they were read: {changed}.")
        for key, (role, path) in files.items():
            record["files"][key] = {"role": role, "oos_year": oos_year, "path": _portable(path, paths.root),
                                    "sha256": before[key]}
        return frame

    run = run_inference(grid, ((year, load(year)) for year in years), estimate, estimate_raw)
    if sha256(config_path) != config_hash:
        raise DataValidationError("the configuration file changed during the run.")
    return run, record


def check_inputs(paths: DataPaths, config_path: Path, record: Mapping) -> None:
    """Every recorded input file still exists with its recorded SHA-256, the configuration included."""
    if not config_path.is_file() or sha256(config_path) != record["config"]["sha256"]:
        raise DataValidationError("the configuration file is missing or differs from the one recorded.")
    changed = []
    for key, entry in record["files"].items():
        path = paths.root / entry["path"]
        if not path.is_file() or sha256(path) != entry["sha256"]:
            changed.append(key)
    if changed:
        raise DataValidationError(f"input files missing or changed since they were read: {sorted(changed)}.")


# ----------------------------------------------------------------------------------------------------------------
# Publication and verification


def default_output_dir(paths: DataPaths) -> Path:
    return paths.diagnostics / "inference"


def _is_link(path: Path) -> bool:
    """A symbolic link or a redirecting Windows reparse point (junction), without Python 3.12 path methods.

    Windows reparse points are links when their tag is a name surrogate (junctions, symbolic links); an
    unknown tag is treated as a link. Non-redirecting reparse points such as cloud placeholders are not.
    """
    try:
        info = os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return False
    if stat.S_ISLNK(info.st_mode):
        return True
    if getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        tag = getattr(info, "st_reparse_tag", None)
        return tag is None or bool(tag & NAME_SURROGATE)
    return False


def _is_result_set(directory: Path) -> bool:
    """Identifiable as an inference result set: it holds both its manifest and its completion marker."""
    return (directory / MANIFEST).is_file() and (directory / SUMMARY).is_file()


def _data_directories(paths: DataPaths) -> tuple[Path, ...]:
    """Pipeline data directories: a destination is neither one of them, nor above one, nor inside one."""
    return (paths.root / "raw", paths.osap, paths.staging, paths.universe, paths.panel, paths.target,
            paths.oos_predictions, paths.family_signals, paths.ensemble_signals, paths.between_family_disagreement,
            paths.within_family_uncertainty, paths.conditional_disagreement, paths.h2_controls,
            paths.mechanical_bias_diagnostics, paths.design_evidence_diagnostics)


def _check_replaceable(output_dir: Path) -> None:
    """An existing destination may be superseded only if it is a verifiable published result of regular files."""
    try:
        if _is_link(output_dir) or not output_dir.is_dir():
            raise PublicationError("not a directory")
        if any(_is_link(p) or not p.is_file() for p in output_dir.iterdir()):
            raise PublicationError("not made of regular files only")
        verify_published(output_dir)
    except Exception as exc:  # whatever is wrong with it, it is not ours to delete
        raise PublicationError(f"{output_dir.name}: not a verifiable published inference result set; refusing to "
                               f"replace it (remove it manually if that is intended).") from exc


def check_destination(output_dir: Path, paths: DataPaths, config_path: Path, replace: bool = False) -> None:
    """Fail closed on any destination that could hold or contain files other than a published result set."""
    output_dir = Path(output_dir)
    if _is_link(output_dir):
        raise PublicationError(f"{output_dir.name}: the destination is a symbolic link or junction.")
    target = output_dir.resolve()
    data_dirs = [p.resolve() for p in _data_directories(paths)]
    anchors = [p.resolve() for p in (paths.root, paths.diagnostics, REPO_ROOT, Path(config_path))]
    for protected in (*anchors, *data_dirs):
        if target == protected or target in protected.parents:
            raise PublicationError(f"{output_dir.name}: the destination is or contains a protected research path.")
    if any(d in target.parents for d in data_dirs):
        raise PublicationError(f"{output_dir.name}: the destination is inside a pipeline data directory.")
    if any(_is_result_set(parent) for parent in target.parents):
        raise PublicationError(f"{output_dir.name}: the destination is inside a published inference result set.")
    if os.path.lexists(output_dir):
        if not replace:
            raise PublicationError(f"{output_dir.name}: a result set already exists; pass replace to supersede it.")
        _check_replaceable(output_dir)


def _scope(run: InferenceRun) -> dict:
    gate = run.report.h2.conditions.get("gate_open")
    return {
        "implemented_analyses": list(IMPLEMENTED_ANALYSES),
        "outstanding_analyses": list(OUTSTANDING_ANALYSES),
        "scientific_protocol_complete": False,
        "roles": {
            "H1": "confirmatory",
            "H2": "confirmatory (H1 gate passed)" if gate else "descriptive (H1 gate not passed)",
            "H3": "secondary, outside the H1 -> H2 sequence",
            "specification B, raw ret_next, HAC lags 0 and 12": "robustness or sensitivity, never decisional",
            "coverage and sample sizes": "descriptive",
        },
    }


def publish(
    run: InferenceRun,
    inputs: Mapping,
    output_dir: Path,
    paths: DataPaths,
    config_path: Path,
    replace: bool = False,
) -> dict:
    """Write the result set to a staging directory, check the inputs, mark it complete and publish it by renaming.

    The destination is checked by `check_destination`; an existing result set is kept unless `replace` is True.
    A handled failure at any point publishes nothing and leaves an existing result set as it was; only the staging
    and backup directories of this call are ever deleted (see the module docstring for the crash limitation).
    """
    output_dir = Path(output_dir)
    check_destination(output_dir, paths, config_path, replace)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    staging = output_dir.parent / f".{output_dir.name}.staging-{token}"
    previous = output_dir.parent / f".{output_dir.name}.previous-{token}"
    created = moved_aside = False
    try:
        staging.mkdir()
        created = True
        artifacts = result_artifacts(run)
        if set(artifacts) != set(OUTPUT_FILES):
            raise PublicationError("the result does not have the expected artifacts.")
        for name in OUTPUT_FILES:
            _write_artifact(staging, name, artifacts[name])
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "grid": {"first": int(run.grid[0]), "last": int(run.grid[-1]), "n_months": int(run.grid.size)},
            "inputs": dict(inputs),
            "outputs": {name: sha256(staging / name) for name in OUTPUT_FILES},
        }
        (staging / MANIFEST).write_text(_dumps(manifest), encoding="utf-8")
        check_inputs(paths, config_path, inputs)  # the files analysed are still the files recorded
        summary = {
            "status": "completed",
            "meaning": COMPLETION_MEANING,
            "schema_version": SCHEMA_VERSION,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "grid": manifest["grid"],
            "n_years": len({m // 100 for m in run.grid.tolist()}),
            "primary_outcome": PRIMARY_OUTCOME,
            "primary_lag": PRIMARY_LAG,
            **_scope(run),
            "manifest_sha256": sha256(staging / MANIFEST),
        }
        (staging / SUMMARY).write_text(_dumps(summary), encoding="utf-8")  # the completion marker, last
        verify_published(staging)
        if os.path.lexists(output_dir):
            check_destination(output_dir, paths, config_path, replace)  # again, just before it is moved aside
            os.replace(output_dir, previous)
            moved_aside = True
        try:
            os.replace(staging, output_dir)
        except BaseException:
            if moved_aside:
                try:
                    os.replace(previous, output_dir)  # restore the superseded result set
                except OSError as exc:
                    raise PublicationError(f"{output_dir.name}: publication failed and the previous result set could "
                                           f"not be restored; it is kept at {previous.name}.") from exc
                moved_aside = False
            raise
    finally:
        if created and staging.is_dir() and not _is_link(staging):
            shutil.rmtree(staging, ignore_errors=True)  # this call's own staging directory only
    if moved_aside:
        shutil.rmtree(previous, ignore_errors=True)  # the superseded result set, verified as such before the move
    return summary


def verify_published(output_dir: Path, paths: DataPaths | None = None, config_path: Path | None = None) -> dict:
    """Check a published result set: inventory, marker, manifest, artifact hashes, calendar; optionally the inputs.

    Returns the completion marker; raises PublicationError (or DataValidationError for inputs) otherwise.
    """
    output_dir = Path(output_dir)
    expected = {*OUTPUT_FILES, MANIFEST, SUMMARY}
    present = {p.name for p in output_dir.iterdir()} if output_dir.is_dir() else set()
    if present != expected:
        raise PublicationError(f"{output_dir.name}: file inventory differs; missing {sorted(expected - present)}, "
                               f"unexpected {sorted(present - expected)}.")
    try:
        summary = json.loads((output_dir / SUMMARY).read_text(encoding="utf-8"))
        manifest = json.loads((output_dir / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PublicationError(f"{output_dir.name}: unreadable marker or manifest.") from exc
    if summary.get("status") != "completed" or summary.get("schema_version") != SCHEMA_VERSION:
        raise PublicationError(f"{output_dir.name}: the completion marker is not a completed run of this schema.")
    if summary.get("manifest_sha256") != sha256(output_dir / MANIFEST):
        raise PublicationError(f"{output_dir.name}: the manifest does not match the completion marker.")
    if manifest.get("schema_version") != SCHEMA_VERSION or set(manifest.get("outputs", {})) != set(OUTPUT_FILES):
        raise PublicationError(f"{output_dir.name}: the manifest does not list the expected artifacts.")
    changed = sorted(name for name in OUTPUT_FILES if sha256(output_dir / name) != manifest["outputs"][name])
    if changed:
        raise PublicationError(f"{output_dir.name}: artifacts differ from the manifest: {changed}.")
    grid = manifest.get("grid")
    try:
        if not isinstance(grid, dict) or summary.get("grid") != grid:
            raise ValueError("the marker and manifest calendars differ")
        first, last = _yyyymm(grid.get("first"), "first month"), _yyyymm(grid.get("last"), "last month")
        frame = pd.read_csv(output_dir / "monthly_summary.csv", usecols=["yyyymm"])
        months = [_yyyymm(m, "monthly_summary.csv") for m in frame["yyyymm"].tolist()]
        # genuine yyyymm integers, strictly increasing, unique and consecutive from first to last
        if months != month_range(first, last) or len(months) != grid.get("n_months"):
            raise ValueError("the monthly calendar is not the consecutive recorded grid")
    except ValueError as exc:
        raise PublicationError(f"{output_dir.name}: calendar metadata is inconsistent ({exc}).") from exc
    if paths is not None and config_path is not None:
        check_inputs(paths, config_path, manifest["inputs"])
    return summary


def execute_inference(
    paths: DataPaths,
    config_path: Path,
    output_dir: Path | None = None,
    replace: bool = False,
    estimate: MonthEstimator = estimate_month_specifications,
    estimate_raw: RawEstimator = estimate_h1_month,
    runner: Callable[..., tuple[InferenceRun, dict]] = run_with_provenance,
) -> dict:
    """Run the frozen baseline with provenance and publish it; estimators and runner are replaceable in tests only."""
    destination = default_output_dir(paths) if output_dir is None else Path(output_dir)
    check_destination(destination, paths, config_path, replace)  # before any outcome is read
    run, inputs = runner(paths, config_path, estimate, estimate_raw)
    return publish(run, inputs, destination, paths, config_path, replace=replace)
