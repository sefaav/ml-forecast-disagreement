"""The H1 robustness analyses of "Robustness and subperiods" outside the core inference (descriptive only).

Pre-specified in docs/pre_analysis_plan.md, "Robustness and subperiods". Each analysis repeats the primary H1
test (specification C, ``target_rank``, H1 sample, Fama-MacBeth with the Bartlett HAC at L = 6) with one change;
none alters the H1 verdict. Specification B, the raw outcome and the HAC lags are part of the core inference.

* From the published monthly C interaction series (``diagnostics/inference/series.csv``, verified with its inputs):
  ``excluding_2021`` (the twelve 2021 months masked on the full grid, calendar positions kept) and the two fixed
  subperiods, each on its own grid. The full series is reported as ``primary`` for comparison.
* Re-estimated month by month with the frozen ``estimate_h1_month`` (basis from every member of U_t, regression on
  the members with a finite ``target_rank``, frozen estimability rule), only the disagreement measure, or the
  signal and its bins, changing:
  - ``raw_disagreement``: the saved SD of the four family ranks, not ranked;
  - ``range_disagreement``, ``mad_disagreement``: max - min and the mean absolute deviation around S of the four
    family ranks, each ranked within the saved (month, signal bin) cells like CD^B;
  - ``ten_bins``: 10 monthly bins of the saved S, CD^B re-ranked within them, ten bin-local blocks;
  - ``leave_out_<family>``: S and the raw SD (ddof = 0) of the three remaining family ranks, 20 bins of that S,
    CD^B within them, S^c = 2 (S - 0.5).
  Measures use the ranks' exact integer numerators, so mathematically tied values tie exactly. With ten bins, CW
  is recomputed in memory as specified (``conditional_disagreement``) but is not a regressor of C. The saved
  inputs are only read; nothing upstream is rerun.

As a reproduction check, C with the saved bins and CD^B is re-estimated and must match the published series
(estimability and coefficients within REPRODUCTION_TOLERANCE); it is not reported.

Published to ``data/diagnostics/h1_robustness/`` (never replaced), staged and renamed into place: ``inference.csv``
(one row per analysis at L = 6, or its failure reason), ``monthly.csv`` (re-estimated analyses: estimability,
reasons, sample size, beta_2, beta_3 per month), ``manifest.json`` (SHA-256 of the configuration, of every input
file, hashed before and after reading and again before publication, and of the outputs) and ``summary.json``,
written last. No stock-level row is written or printed.

Command line: ``python -m disagreement.inference.robustness [--execute]``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sys
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from disagreement.data.config import DEFAULT_CONFIG, DEFAULT_DATA_DIR, DataPaths
from disagreement.data.target import TARGET_COLUMNS
from disagreement.data.validation import KEY, DataValidationError
from disagreement.inference.fama_macbeth import HAC_LAG, fama_macbeth
from disagreement.inference.inputs import CD, CW, _check_conditional, _check_target, _keyed, _same_keys, load_target
from disagreement.inference.monthly import MonthlyEstimate, estimate_h1_month
from disagreement.inference.orchestrate import _validated_grid, baseline_calendar
from disagreement.inference.output import (
    MANIFEST,
    SUMMARY,
    PublicationError,
    _dumps,
    _hashes,
    _portable,
    check_destination,
    check_inputs,
    sha256,
    verify_published,
)
from disagreement.inference.output import default_output_dir as inference_dir
from disagreement.inference.series import Inference, MonthlySeries, infer
from disagreement.models.selection import FAMILIES
from disagreement.signals import conditional
from disagreement.signals.between import load_ensemble
from disagreement.signals.conditional import SIGNAL_BINS, conditional_disagreement, signal_bins
from disagreement.signals.ensemble import ENSEMBLE_COLUMNS
from disagreement.signals.family import month_size, rank_mean, rank_numerators, unit_rank
from disagreement.walkforward.calendar import month_range

RAW = "between_family_disagreement_raw"
W_RAW = "within_family_uncertainty_raw"
FAMILY_RANKS = [f"{family}_rank" for family in FAMILIES]
TEN_BINS = 10
PRIMARY = "primary"
DESCRIPTIONS = {  # analysis -> its change to the primary H1 test, as in "Robustness and subperiods"
    PRIMARY: "the published primary H1 interaction series, for comparison",
    "excluding_2021": "without the twelve 2021 formation months",
    "raw_disagreement": "raw between-family disagreement (SD of the four family ranks, not ranked) for CD^B",
    "range_disagreement": "max - min of the four family ranks, ranked within the (month, bin) cells, for CD^B",
    "mad_disagreement": "mean absolute deviation of the four family ranks around S, ranked within the cells, "
                        "for CD^B",
    "ten_bins": "CD^B and the bin-specific controls recomputed with 10 monthly bins of S",
    **{f"leave_out_{f}": f"S, raw disagreement, bins and CD^B from the three families other than {f}"
       for f in FAMILIES},
    "subperiod_1985_2004": "formation months 1985-01 to 2004-12",
    "subperiod_2005_2024": "formation months 2005-01 to 2024-12",
}
SUBPERIODS = {"subperiod_1985_2004": (198501, 200412), "subperiod_2005_2024": (200501, 202412)}
FROM_SERIES = (PRIMARY, "excluding_2021", *SUBPERIODS)
REESTIMATED = tuple(name for name in DESCRIPTIONS if name not in FROM_SERIES)
REPRODUCTION_TOLERANCE = 1e-12  # the same estimator on the same arrays: only floating-point noise is admissible
OUTPUT_FILES = ("inference.csv", "monthly.csv")
MEANING = ("descriptive H1 robustness analyses of the pre-analysis plan, each repeating the primary H1 test with "
           "one change; they do not alter the H1 verdict")

Estimator = Callable[..., MonthlyEstimate]
Regressors = tuple[np.ndarray, np.ndarray, np.ndarray, int]  # S^c, signal bins, disagreement measure, number of bins


# ----------------------------------------------------------------------------------------------------------------
# Disagreement measures from exact unit-rank numerators k (rows x families) over m (rows)


def dispersion(k: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Population SD (ddof = 0) of the family ranks k / m, as in ``disagreement.signals.between``."""
    f = k.shape[1]
    return np.sqrt(f * (k * k).sum(axis=1) - k.sum(axis=1) ** 2) / (f * m)


def range_measure(k: np.ndarray, m: np.ndarray) -> np.ndarray:
    """max - min of the family ranks."""
    return (k.max(axis=1) - k.min(axis=1)) / m


def mad_measure(k: np.ndarray, m: np.ndarray) -> np.ndarray:
    """(1 / f) sum_f |rank_f - S| with S the mean of the f ranks: sum_f |f k_f - sum k| / (f^2 m)."""
    f = k.shape[1]
    return np.abs(f * k - k.sum(axis=1, keepdims=True)).sum(axis=1) / (f * f * m)


def variant_regressors(frame: pd.DataFrame) -> dict[str, Regressors]:
    """The regressors of C, on all members of U_t, for the saved primary (PRIMARY) and every re-estimated analysis.

    `frame` holds one OOS year sorted by (yyyymm, permno) with ensemble_signal, centered_signal, signal_bin, CD^B,
    the raw SD and the four family ranks.
    """
    index, month = frame.index, frame["yyyymm"]
    ranks = frame.loc[:, FAMILY_RANKS].to_numpy(dtype="float64")
    n = month_size(month)
    k, m = rank_numerators(ranks, n, "family ranks")
    sc = frame["centered_signal"].to_numpy(dtype="float64")
    bins = frame["signal_bin"].to_numpy(dtype="int64")
    raw = frame[RAW].to_numpy(dtype="float64")

    def ranked(values: np.ndarray, cells: np.ndarray) -> np.ndarray:  # the CD^B convention within (month, bin)
        return unit_rank(pd.Series(values, index=index), [month, pd.Series(cells, index=index)]).to_numpy()

    def bins_of(signal: np.ndarray, count: int) -> np.ndarray:
        return signal_bins(pd.Series(signal, index=index), month, count).to_numpy(dtype="int64")

    out: dict[str, Regressors] = {
        PRIMARY: (sc, bins, frame[CD].to_numpy(dtype="float64"), SIGNAL_BINS),
        "raw_disagreement": (sc, bins, raw, SIGNAL_BINS),
        "range_disagreement": (sc, bins, ranked(range_measure(k, m), bins), SIGNAL_BINS),
        "mad_disagreement": (sc, bins, ranked(mad_measure(k, m), bins), SIGNAL_BINS),
    }
    ten = conditional_disagreement(frame, TEN_BINS)  # 10 bins of S, CD^B and CW within them (CW unused by C)
    if not np.array_equal(ten.loc[:, list(KEY)].to_numpy(), frame.loc[:, list(KEY)].to_numpy()):
        raise DataValidationError("the ten-bin conditional ranks are not aligned with the year's stock-months.")
    out["ten_bins"] = (sc, ten["signal_bin"].to_numpy(dtype="int64"), ten[CD].to_numpy(dtype="float64"), TEN_BINS)
    for i, family in enumerate(FAMILIES):
        rest = [j for j in range(len(FAMILIES)) if j != i]
        signal = rank_mean(ranks[:, rest], n, "family ranks")
        cells = bins_of(signal, SIGNAL_BINS)
        out[f"leave_out_{family}"] = (2.0 * (signal - 0.5), cells, ranked(dispersion(k[:, rest], m), cells),
                                      SIGNAL_BINS)
    return out


# ----------------------------------------------------------------------------------------------------------------
# Series from the published primary series


def series_variants(saved: MonthlySeries) -> dict[str, MonthlySeries]:
    """The primary series, the same without 2021 (positions kept) and the two subperiods on their own grids."""
    out = {PRIMARY: saved, "excluding_2021": saved.restricted(saved.grid // 100 != 2021, "excluding_2021")}
    for name, (first, last) in SUBPERIODS.items():
        keep = (saved.grid >= first) & (saved.grid <= last)
        out[name] = MonthlySeries(name, saved.grid[keep], saved.values[keep], saved.estimable[keep])
    return out


# ----------------------------------------------------------------------------------------------------------------
# Run


@dataclass(frozen=True)
class RobustnessRun:
    grid: np.ndarray
    estimates: dict[str, tuple[MonthlyEstimate, ...]]  # re-estimated analysis -> one estimate per grid month
    inference: dict[str, Inference]  # every analysis, at L = 6


def run_robustness(
    grid: Iterable[int],
    years: Iterable[tuple[int, pd.DataFrame]],
    saved: MonthlySeries,
    estimate: Estimator = estimate_h1_month,
) -> RobustnessRun:
    """Every analysis from validated yearly frames (calendar order) and the published primary series."""
    months = _validated_grid(grid)
    if saved.grid.tolist() != months:
        raise DataValidationError("the published primary series is not on the calendar grid.")
    monthly: dict[str, dict[int, MonthlyEstimate]] = {name: {} for name in (PRIMARY, *REESTIMATED)}
    for year, frame in years:
        outcome = frame["target_rank"].to_numpy(dtype="float64", na_value=np.nan)
        regressors = variant_regressors(frame)
        for key, rows in frame.groupby("yyyymm", sort=True).indices.items():
            month = int(key)  # type: ignore[arg-type]  # yyyymm is an integer column
            if month not in months or month // 100 != year or month in monthly[PRIMARY]:
                raise DataValidationError(f"{year}: month {month} is not an expected grid month of this year.")
            for name, (sc, bins, measure, n_bins) in regressors.items():
                monthly[name][month] = estimate(outcome[rows], sc[rows], bins[rows], measure[rows], n_bins)
    missing = sorted(set(months) - set(monthly[PRIMARY]))
    if missing:
        raise DataValidationError(f"no input for the grid months {missing[:3]}... ({len(missing)} months).")

    estimates = {name: tuple(by_month[m] for m in months) for name, by_month in monthly.items()}
    mask, values = _interaction(estimates.pop(PRIMARY))
    if not np.array_equal(mask, saved.estimable) or not (
            np.abs(values[mask] - saved.values[mask]) <= REPRODUCTION_TOLERANCE).all():
        raise DataValidationError("re-estimating C from the saved inputs does not reproduce the published series.")
    series = series_variants(saved)
    for name in REESTIMATED:
        mask, values = _interaction(estimates[name])
        series[name] = MonthlySeries(name, saved.grid, values, mask)
    return RobustnessRun(saved.grid, estimates, {name: infer(series[name], (HAC_LAG,)) for name in DESCRIPTIONS})


def _interaction(estimates: tuple[MonthlyEstimate, ...]) -> tuple[np.ndarray, np.ndarray]:
    """The estimability mask and the beta_3 values (NaN where not estimable) of a monthly estimate sequence."""
    mask = np.array([e.estimable for e in estimates], dtype=bool)
    values = np.array([e.coefficients.interaction if e.coefficients is not None else np.nan for e in estimates])
    return mask, values


# ----------------------------------------------------------------------------------------------------------------
# Inputs


def year_files(paths: DataPaths, year: int) -> dict[str, tuple[str, Path]]:
    return {
        f"{year}/conditional_disagreement": ("conditional disagreement",
                                             paths.conditional_disagreement / f"{year}.parquet"),
        f"{year}/conditional_disagreement_marker": ("conditional disagreement marker",
                                                    paths.conditional_disagreement / f"{year}.json"),
        f"{year}/ensemble_signals": ("ensemble signals", paths.ensemble_signals / f"{year}.parquet"),
        f"{year}/ensemble_signals_marker": ("ensemble signals marker", paths.ensemble_signals / f"{year}.json"),
        f"{year}/target": ("target", paths.target / f"target_{year}.parquet"),
    }


def load_year(paths: DataPaths, year: int) -> pd.DataFrame:
    """One validated OOS year: saved S, S^c, bins, CD^B, raw SD, family ranks and target_rank on the same keys."""
    labels = {name: f"{year} {name}" for name in ("conditional disagreement", "ensemble signals", "target")}
    cond = _keyed(conditional._load(paths.conditional_disagreement, year, conditional.OUTPUT_COLUMNS,
                                    "conditional disagreement"), tuple(conditional.OUTPUT_COLUMNS), year,
                  labels["conditional disagreement"])
    _check_conditional(cond, labels["conditional disagreement"])  # ranges and the frozen bin rule
    ensemble = _keyed(load_ensemble(paths, year), ENSEMBLE_COLUMNS, year, labels["ensemble signals"])
    target = _keyed(load_target(paths, year), tuple(TARGET_COLUMNS), year, labels["target"])
    _check_target(target, labels["target"])
    _same_keys(cond, ensemble, labels["ensemble signals"], "the conditional disagreement")
    _same_keys(cond, target, labels["target"], "the conditional disagreement")
    frame = cond.loc[:, [*KEY, "ensemble_signal", "signal_bin", RAW, W_RAW, CD, CW]].copy()  # all sorted by key
    if (ensemble["ensemble_signal"].to_numpy() != frame["ensemble_signal"].to_numpy()).any():
        raise DataValidationError(f"{year}: the ensemble and conditional-disagreement signals differ.")
    frame[FAMILY_RANKS] = ensemble.loc[:, FAMILY_RANKS].to_numpy(dtype="float64")
    frame["centered_signal"] = 2.0 * (frame["ensemble_signal"].to_numpy(dtype="float64") - 0.5)
    frame["target_rank"] = target["target_rank"].to_numpy(dtype="float64", na_value=np.nan)
    k, m = rank_numerators(frame[FAMILY_RANKS].to_numpy(dtype="float64"), month_size(frame["yyyymm"]),
                           labels["ensemble signals"])
    if (dispersion(k, m) != frame[RAW].to_numpy(dtype="float64")).any():
        raise DataValidationError(f"{year}: the saved raw disagreement is not the SD of the saved family ranks.")
    for raw, ranked in ((RAW, CD), (W_RAW, CW)):
        expected = unit_rank(frame[raw], [frame["yyyymm"], frame["signal_bin"]]).to_numpy()
        if (expected != frame[ranked].to_numpy(dtype="float64")).any():
            raise DataValidationError(f"{year}: the saved {ranked} is not {raw} ranked within the cells.")
    return frame


def load_published(paths: DataPaths, config_path: Path, grid: list[int], record: dict) -> MonthlySeries:
    """The verified published C interaction series; its result files are added to the input record."""
    directory = inference_dir(paths)
    files = {f"inference/{name}": ("published core inference", directory / name)
             for name in ("series.csv", "report.json", MANIFEST, SUMMARY)}
    before = _hashes(files, "published inference")
    verify_published(directory, paths, config_path)  # inventory, marker, manifest, hashes, calendar, inputs
    table = pd.read_csv(directory / "series.csv", float_precision="round_trip")
    reference = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    if _hashes(files, "published inference") != before:
        raise DataValidationError("the published inference changed while it was read.")
    rows = table[(table["group"] == "coefficient") & (table["series"] == "C.interaction")]
    if rows["yyyymm"].tolist() != grid or rows["estimable"].dtype != np.bool_:
        raise DataValidationError("the published C interaction series is not on the calendar grid.")
    mask = rows["estimable"].to_numpy()
    saved = MonthlySeries(PRIMARY, np.asarray(grid, dtype="int64"),
                          np.where(mask, rows["value"].to_numpy(dtype="float64"), np.nan), mask)
    basis = (reference["decisions"]["H1"]["basis"] or {}).get("C.interaction") or {}
    result = fama_macbeth(saved.values, saved.estimable, HAC_LAG)
    if (result.mean, result.se) != (basis.get("mean"), basis.get("se")):
        raise DataValidationError("the published series does not reproduce the published H1 statistics.")
    for key, (role, path) in files.items():
        record["files"][key] = {"role": role, "path": _portable(path, paths.root), "sha256": before[key]}
    return saved


# ----------------------------------------------------------------------------------------------------------------
# Publication


def default_output_dir(paths: DataPaths) -> Path:
    return paths.diagnostics / "h1_robustness"


def artifacts(run: RobustnessRun) -> dict[str, pd.DataFrame]:
    rows = []
    for name, inference in run.inference.items():
        row = {"analysis": name, "source": "published series" if name in FROM_SERIES else "re-estimated",
               "change": DESCRIPTIONS[name], "lag": HAC_LAG}
        if HAC_LAG in inference.results:
            r = inference.results[HAC_LAG]
            row |= {"status": "estimated", "mean": r.mean, "se": r.se, "t": r.t, "p_value": r.p_value,
                    "ci_low": r.ci_low, "ci_high": r.ci_high, "n_months": r.n_months, "n_grid": r.n_grid,
                    "hac_variance": r.hac_variance, "failure": None}
        else:
            row |= {"status": "failed", "n_grid": int(inference.series.grid.size),
                    "failure": inference.failures.get(HAC_LAG)}
        rows.append(row)
    monthly = [{"analysis": name, "yyyymm": int(month), "n_bins": e.basis.n_bins, "estimable": e.estimable,
                "reasons": json.dumps(list(e.reasons)), "n_obs": e.n_obs,
                "disagreement": e.coefficients.disagreement if e.coefficients is not None else np.nan,
                "interaction": e.coefficients.interaction if e.coefficients is not None else np.nan}
               for name, estimates in run.estimates.items() for month, e in zip(run.grid.tolist(), estimates)]
    columns = ["analysis", "source", "change", "lag", "status", "mean", "se", "t", "p_value", "ci_low", "ci_high",
               "n_months", "n_grid", "hac_variance", "failure"]
    table = pd.DataFrame(rows).reindex(columns=columns)
    table[["n_months", "n_grid"]] = table[["n_months", "n_grid"]].astype("Int64")
    return {"inference.csv": table, "monthly.csv": pd.DataFrame(monthly)}


def _grid(run: RobustnessRun) -> dict:
    return {"first": int(run.grid[0]), "last": int(run.grid[-1]), "n_months": int(run.grid.size)}


def _summary(run: RobustnessRun, manifest_sha: str) -> dict:
    return {
        "status": "completed",
        "meaning": MEANING,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "grid": _grid(run),
        "specification": "C", "outcome": "target_rank", "lag": HAC_LAG,
        "analyses": DESCRIPTIONS,
        "non_estimable_months": {name: int(sum(not e.estimable for e in es)) for name, es in run.estimates.items()},
        "inference_failures": {name: inf.failures[HAC_LAG] for name, inf in run.inference.items() if inf.failures},
        "manifest_sha256": manifest_sha,
    }


def publish(run: RobustnessRun, inputs: Mapping, output_dir: Path, paths: DataPaths, config_path: Path) -> dict:
    """Stage, check the inputs, write the marker last, verify and rename; an existing destination is never replaced."""
    check_destination(output_dir, paths, config_path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        for name, table in artifacts(run).items():
            table.to_csv(staging / name, index=False, float_format="%.17g", lineterminator="\n")
        manifest = {"grid": _grid(run), "inputs": dict(inputs),
                    "outputs": {name: sha256(staging / name) for name in OUTPUT_FILES}}
        (staging / MANIFEST).write_text(_dumps(manifest), encoding="utf-8")
        check_inputs(paths, config_path, inputs)  # the files analysed are still the files recorded
        summary = _summary(run, sha256(staging / MANIFEST))
        (staging / SUMMARY).write_text(_dumps(summary), encoding="utf-8")  # the completion marker, last
        verify_robustness(staging)
        if os.path.lexists(output_dir):
            raise PublicationError(f"{output_dir.name}: a result set appeared during the run; nothing replaced.")
        os.rename(staging, output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return summary


def verify_robustness(output_dir: Path, paths: DataPaths | None = None, config_path: Path | None = None) -> dict:
    """Inventory, marker, manifest and output hashes; with paths and configuration, every recorded input too."""
    output_dir = Path(output_dir)
    present = {p.name for p in output_dir.iterdir()} if output_dir.is_dir() else set()
    if present != {*OUTPUT_FILES, MANIFEST, SUMMARY}:
        raise PublicationError(f"{output_dir.name}: not an H1 robustness result set (file inventory differs).")
    try:
        summary = json.loads((output_dir / SUMMARY).read_text(encoding="utf-8"))
        manifest = json.loads((output_dir / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PublicationError(f"{output_dir.name}: unreadable marker or manifest.") from exc
    if (summary.get("status") != "completed" or summary.get("meaning") != MEANING
            or summary.get("manifest_sha256") != sha256(output_dir / MANIFEST)):
        raise PublicationError(f"{output_dir.name}: the completion marker does not match the manifest.")
    frozen = {"specification": "C", "outcome": "target_rank", "lag": HAC_LAG, "analyses": DESCRIPTIONS}
    if any(summary.get(key) != value for key, value in frozen.items()):
        raise PublicationError(f"{output_dir.name}: the completion marker misstates the frozen scientific metadata.")
    if manifest.get("outputs") != {name: sha256(output_dir / name) for name in OUTPUT_FILES}:
        raise PublicationError(f"{output_dir.name}: result files differ from the manifest.")
    try:  # structure: the analyses at L = 6, and every re-estimated analysis on every month of the recorded grid
        grid = manifest["grid"]
        months = month_range(grid["first"], grid["last"])
        table = pd.read_csv(output_dir / "inference.csv", usecols=["analysis", "lag"])
        monthly = pd.read_csv(output_dir / "monthly.csv", usecols=["analysis", "yyyymm"])
        if summary.get("grid") != grid or len(months) != grid["n_months"]:
            raise ValueError("the marker and manifest calendars differ")
        if table["analysis"].tolist() != list(DESCRIPTIONS) or (table["lag"] != HAC_LAG).any():
            raise ValueError("inference.csv does not hold the expected analyses at the primary lag")
        if list(zip(monthly["analysis"], monthly["yyyymm"])) != [(a, m) for a in REESTIMATED for m in months]:
            raise ValueError("monthly.csv does not cover every re-estimated analysis on every grid month")
    except (KeyError, TypeError, ValueError) as exc:
        raise PublicationError(f"{output_dir.name}: inconsistent result structure ({exc}).") from exc
    if paths is not None and config_path is not None:
        check_inputs(paths, config_path, manifest["inputs"])
    return summary


def execute(paths: DataPaths, config_path: Path, output_dir: Path | None = None) -> dict:
    """Verify the published inference, load every OOS year with provenance, run every analysis and publish."""
    destination = default_output_dir(paths) if output_dir is None else Path(output_dir)
    check_destination(destination, paths, config_path)  # before any outcome is read
    config_hash = sha256(config_path)
    grid, years = baseline_calendar(config_path)
    record: dict = {"config": {"path": _portable(config_path, paths.root), "sha256": config_hash}, "files": {}}
    saved = load_published(paths, config_path, grid, record)

    def load(year: int) -> pd.DataFrame:
        files = year_files(paths, year)
        before = _hashes(files, f"{year}")
        frame = load_year(paths, year)
        if _hashes(files, f"{year}") != before:
            raise DataValidationError(f"{year}: input files changed while they were read.")
        for key, (role, path) in files.items():
            record["files"][key] = {"role": role, "path": _portable(path, paths.root), "sha256": before[key]}
        return frame

    run = run_robustness(grid, ((year, load(year)) for year in years), saved)
    if sha256(config_path) != config_hash:
        raise DataValidationError("the configuration file changed during the run.")
    return publish(run, record, destination, paths, config_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m disagreement.inference.robustness",
                                     description="Run and publish the remaining pre-specified H1 robustness analyses.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=None, help="default: <data-dir>/diagnostics/h1_robustness")
    parser.add_argument("--execute", action="store_true", help="load the outcomes, estimate and publish")
    args = parser.parse_args(argv)
    paths = DataPaths(args.data_dir)
    destination = args.output_dir or default_output_dir(paths)
    try:
        grid, years = baseline_calendar(args.config)
        if not args.execute:
            print(f"Dry run: {len(DESCRIPTIONS) - 1} H1 robustness analyses over {len(grid)} months ({grid[0]}-"
                  f"{grid[-1]}, {len(years)} OOS years) would be published to {destination}. Nothing was loaded; "
                  f"pass --execute to run.")
            return 0
        summary = execute(paths, args.config, destination)
    except (DataValidationError, PublicationError, ValueError, KeyError, OSError) as exc:
        print(f"H1 robustness not published: {exc}", file=sys.stderr)
        return 2
    print(f"Published {len(summary['analyses']) - 1} H1 robustness analyses over {summary['grid']['n_months']} "
          f"months to {destination}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
