"""H2 control characteristics: size, realized volatility and illiquidity as monthly centred ranks.

For every stock-month of the baseline OOS sample, from formation-time data
only (definitions: docs/pre_analysis_plan.md, "H2 control characteristics"):

    size                = ln(mktcap)       universe, CRSP CIZ MthCap of formation month t
    realized_volatility = -RealizedVol     panel, signed OSAP predictor, sign reversed
    illiquidity         = +Illiquidity     panel, signed OSAP predictor, sign kept

Each economic value X is ranked separately within its formation month among
the members of U_t with a finite X (average ranks for ties) and centred,

    X_rank = 2 (rank - 1) / (n_finite - 1) - 1  in [-1, 1],

0 when the month's finite values are constant or there is only one, missing
when X is missing or infinite. The ranks use the full universe of the month,
before any complete-case restriction, and never cross months; they are the
``unit_rank`` convention of the family signals, centred like the signal.
``complete_case`` flags the stock-months with all three ranks. The centred
signal 2 (S - 0.5) uses the saved ensemble signal of the conditional
disagreement, and each interaction ``signal_x_<control>_rank`` multiplies it
with one rank, missing where the rank is missing. The ensemble signal, the
signal bins, CD and CW are read, checked and never recomputed. The output
keeps exactly the universe's stock-months; no return or target is read.

Command line: ``python -m disagreement.controls.h2 [--year Y]``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from disagreement.data.config import DEFAULT_CONFIG, DEFAULT_DATA_DIR, DataPaths
from disagreement.data.universe import UNIVERSE_COLUMNS
from disagreement.data.validation import KEY, DataValidationError, require_unique_keys
from disagreement.signals import conditional
from disagreement.signals.family import unit_rank
from disagreement.walkforward.calendar import baseline_episodes, month_range

CONTROLS = ("size", "realized_volatility", "illiquidity")
RANKS = tuple(f"{control}_rank" for control in CONTROLS)
INTERACTIONS = tuple(f"signal_x_{control}_rank" for control in CONTROLS)
PANEL_COLUMNS = ("RealizedVol", "Illiquidity")
INPUT_COLUMNS = (*KEY, "mktcap", *PANEL_COLUMNS, "ensemble_signal")
OUTPUT_COLUMNS = (*KEY, *CONTROLS, *RANKS, "complete_case", "centered_signal", *INTERACTIONS)
DEFINITION = {  # the frozen transformation, recorded with every output
    "size": "ln(mktcap): the universe's formation-month CRSP CIZ MthCap, security level",
    "realized_volatility": "-RealizedVol: the signed OSAP predictor of the formation-time panel, sign reversed",
    "illiquidity": "+Illiquidity: the signed OSAP predictor of the formation-time panel, sign kept",
    "rank": "2 * (average_rank(X) - 1) / (n_finite - 1) - 1 within each formation month among the finite values of "
    "the full universe U_t; 0 for a constant or single value; missing or infinite X stays missing; computed before "
    "any complete-case restriction; never across months",
    "complete_case": "all three ranks present",
    "centered_signal": "2 * (ensemble_signal - 0.5), the saved ensemble signal, not re-ranked",
    "interactions": "centered_signal * X_rank, missing where X_rank is missing",
    "upstream_reference": "OpenSourceAP/CrossSection tag v2.0.0, commit b4e911e69678a7424f318617a61d813f54183123",
}
DEFINITION_SHA256 = hashlib.sha256(json.dumps(DEFINITION, sort_keys=True).encode("utf-8")).hexdigest()


def _finite_or_missing(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype="float64").copy()
    out[~np.isfinite(out)] = np.nan
    return out


def centered_rank(values: pd.Series, month: pd.Series) -> pd.Series:
    """2 (rank - 1) / (n - 1) - 1 within each month among the finite values; 0 for one or constant values.

    Missing and infinite values get no rank and stay missing. ``unit_rank`` would give them the midpoint 0.5
    (centred: 0), so they are masked afterwards instead of changing the shared ranking code.
    """
    finite = pd.Series(_finite_or_missing(values.to_numpy(dtype="float64", na_value=np.nan)), index=values.index)
    centered = 2.0 * unit_rank(finite, [month]) - 1.0  # 2 * ((r - 1) / (n - 1)) is exact: a power-of-two scaling
    return centered.where(finite.notna())


def h2_controls(frame: pd.DataFrame) -> pd.DataFrame:
    """Economic values, centred monthly ranks, complete-case flag, centred signal and interactions, one row per row."""
    aligned = frame.loc[:, list(INPUT_COLUMNS)].sort_values(["yyyymm", "permno"], ignore_index=True)
    out = aligned.loc[:, list(KEY)].copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out["size"] = _finite_or_missing(np.log(aligned["mktcap"].to_numpy(dtype="float64", na_value=np.nan)))
    out["realized_volatility"] = _finite_or_missing(-aligned["RealizedVol"].to_numpy(dtype="float64", na_value=np.nan))
    out["illiquidity"] = _finite_or_missing(aligned["Illiquidity"].to_numpy(dtype="float64", na_value=np.nan))
    for control, rank in zip(CONTROLS, RANKS):
        out[rank] = centered_rank(out[control], out["yyyymm"])  # the full cross-section of the month
    out["complete_case"] = out.loc[:, list(RANKS)].notna().all(axis=1).to_numpy(dtype=bool)
    out["centered_signal"] = 2.0 * (aligned["ensemble_signal"].to_numpy(dtype="float64") - 0.5)
    for rank, interaction in zip(RANKS, INTERACTIONS):
        out[interaction] = out["centered_signal"] * out[rank]  # missing where the rank is missing
    return out.loc[:, list(OUTPUT_COLUMNS)]


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["yyyymm", "permno"], ignore_index=True)


def _require_oos_months(frame: pd.DataFrame, oos_year: int, label: str) -> None:
    expected, observed = month_range(oos_year * 100 + 1, oos_year * 100 + 12), set(frame["yyyymm"].tolist())
    if observed != set(expected):
        missing, extra = sorted(set(expected) - observed), sorted(observed - set(expected))
        raise DataValidationError(f"{label}: not the 12 OOS months; missing {missing}, extra {extra}.")


def _require_same_keys(reference: pd.DataFrame, other: pd.DataFrame, label: str, reference_name: str) -> None:
    """Both sorted by (yyyymm, permno): the same stock-months in the same order, or an error with counts only."""
    keys, other_keys = (pd.MultiIndex.from_frame(f.loc[:, list(KEY)]) for f in (reference, other))
    if not other_keys.equals(keys):
        missing, extra = len(keys.difference(other_keys)), len(other_keys.difference(keys))
        raise DataValidationError(
            f"{label} covers different stock-months than {reference_name}: {missing} missing, {extra} extra."
        )


def load_universe(paths: DataPaths, oos_year: int) -> pd.DataFrame:
    """The universe of one OOS year from a completed build: schema, unique keys, twelve months, positive mktcap."""
    path, marker = paths.universe / f"universe_{oos_year}.parquet", paths.diagnostics / "summary.json"
    if not (path.exists() and marker.exists()):
        raise DataValidationError(f"{oos_year}: no complete universe (universe file and diagnostics/summary.json).")
    label = f"{oos_year} universe"
    names = pq.read_schema(path).names
    if set(names) != set(UNIVERSE_COLUMNS):
        raise DataValidationError(f"{label}: columns {sorted(names)}, expected {sorted(UNIVERSE_COLUMNS)}.")
    frame = pd.read_parquet(path, columns=[*KEY, "mktcap"])
    require_unique_keys(frame, label)
    if not (frame["yyyymm"] // 100 == oos_year).all():
        raise DataValidationError(f"{label}: rows outside the OOS year.")
    _require_oos_months(frame, oos_year, label)
    cap = frame["mktcap"].to_numpy(dtype="float64", na_value=np.nan)
    if not (np.isfinite(cap).all() and (cap > 0.0).all()):
        raise DataValidationError(f"{label}: mktcap must be finite and strictly positive.")
    return _sorted(frame)


def load_panel_controls(paths: DataPaths, oos_year: int, universe: pd.DataFrame) -> pd.DataFrame:
    """RealizedVol and Illiquidity of the panel year, on exactly the universe's stock-months with the same mktcap."""
    path = paths.panel / f"panel_{oos_year}.parquet"
    if not path.exists():
        raise DataValidationError(f"{oos_year}: no panel file.")
    label = f"{oos_year} panel"
    required = (*KEY, "mktcap", *PANEL_COLUMNS)
    names = pq.read_schema(path).names
    missing = [c for c in required if c not in names]
    if missing:
        raise DataValidationError(f"{label}: columns missing: {missing}.")
    frame = _sorted(pd.read_parquet(path, columns=list(required)))
    require_unique_keys(frame, label)
    _require_same_keys(universe, frame, label, "the universe")
    differs = int((frame["mktcap"].to_numpy(dtype="float64") != universe["mktcap"].to_numpy(dtype="float64")).sum())
    if differs:
        raise DataValidationError(f"{label}: mktcap differs from the universe in {differs} rows.")
    return frame


def load_signal(paths: DataPaths, oos_year: int, universe: pd.DataFrame) -> pd.Series:
    """The saved ensemble signal of the complete conditional-disagreement year, on the universe's stock-months."""
    frame = conditional._load(
        paths.conditional_disagreement, oos_year, conditional.OUTPUT_COLUMNS, "conditional disagreement"
    )  # marker, exact schema, unique keys, the twelve OOS months; sorted by (yyyymm, permno)
    label = f"{oos_year} conditional disagreement"
    signal = frame["ensemble_signal"].to_numpy(dtype="float64")
    if not (np.isfinite(signal).all() and (signal >= 0.0).all() and (signal <= 1.0).all()):
        raise DataValidationError(f"{label}: ensemble_signal must be finite and in [0, 1].")
    _require_same_keys(universe, frame, label, "the universe")
    return frame["ensemble_signal"]


def load_inputs(paths: DataPaths, oos_year: int) -> pd.DataFrame:
    """The validated inputs of one OOS year, aligned on the universe's stock-months."""
    universe = load_universe(paths, oos_year)
    panel = load_panel_controls(paths, oos_year, universe)
    inputs = universe.loc[:, [*KEY, "mktcap"]].copy()
    for column in PANEL_COLUMNS:
        inputs[column] = panel[column].to_numpy(dtype="float64", na_value=np.nan)
    inputs["ensemble_signal"] = load_signal(paths, oos_year, universe).to_numpy(dtype="float64")
    return inputs


def input_files(paths: DataPaths, oos_year: int) -> dict[str, Path]:
    return {
        "universe": paths.universe / f"universe_{oos_year}.parquet",
        "panel": paths.panel / f"panel_{oos_year}.parquet",
        "conditional_disagreement": paths.conditional_disagreement / f"{oos_year}.parquet",
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(directory: Path, oos_year: int, frame: pd.DataFrame, metadata: dict) -> None:
    """Atomic writes; the JSON sidecar, replaced last, marks a complete year and records the output's SHA-256."""
    directory.mkdir(parents=True, exist_ok=True)
    output_path, metadata_path = directory / f"{oos_year}.parquet", directory / f"{oos_year}.json"
    metadata_path.unlink(missing_ok=True)
    tmp_output = output_path.with_suffix(".parquet.tmp")
    tmp_metadata = metadata_path.with_suffix(".json.tmp")
    try:
        frame.loc[:, list(OUTPUT_COLUMNS)].to_parquet(tmp_output, index=False)
        os.replace(tmp_output, output_path)
        metadata["output_sha256"] = _sha256(output_path)  # of the file as written, before the marker is published
        text = json.dumps(metadata, indent=2)
        tmp_metadata.write_text(text, encoding="utf-8")
        os.replace(tmp_metadata, metadata_path)  # the last operation: marks success
    finally:
        tmp_output.unlink(missing_ok=True)
        tmp_metadata.unlink(missing_ok=True)


def summarize(out: pd.DataFrame) -> dict:
    """Aggregate coverage of one year: counts and shares only."""
    by_month = out.groupby("yyyymm")["complete_case"].mean()
    return {
        "n_stock_months": len(out),
        "months": sorted(int(m) for m in out["yyyymm"].unique()),
        "n_complete_case": int(out["complete_case"].sum()),
        "complete_case_share": float(out["complete_case"].mean()),
        "min_monthly_complete_case_share": float(by_month.min()),
        "coverage": {control: float(out[rank].notna().mean()) for control, rank in zip(CONTROLS, RANKS)},
    }


def run_h2_controls(
    paths: DataPaths, config_path: Path = DEFAULT_CONFIG, years: Iterable[int] | None = None, log=print
) -> list[dict]:
    """The H2 controls of the selected OOS years (all baseline years by default)."""
    baseline_years = [e.oos_year for e in baseline_episodes(config_path)]
    if years is not None:
        missing = sorted(set(years) - set(baseline_years))
        if missing:
            raise ValueError(f"not baseline OOS years: {missing}")
        baseline_years = [y for y in baseline_years if y in set(years)]

    summaries = []
    for oos_year in baseline_years:
        (paths.h2_controls / f"{oos_year}.json").unlink(missing_ok=True)  # not complete until rewritten
        out = h2_controls(load_inputs(paths, oos_year))
        metadata = {
            "oos_year": oos_year,
            **summarize(out),
            "definition": DEFINITION,
            "definition_sha256": DEFINITION_SHA256,
            "inputs_sha256": {name: _sha256(path) for name, path in input_files(paths, oos_year).items()},
        }
        _write(paths.h2_controls, oos_year, out, metadata)
        log(
            f"{oos_year}: {len(out):,} stock-months, complete case {metadata['complete_case_share']:.3f} "
            f"(monthly minimum {metadata['min_monthly_complete_case_share']:.3f})"
        )
        summaries.append(metadata)
    return summaries


def load_h2_controls(paths: DataPaths, oos_year: int) -> pd.DataFrame:
    """A complete year of H2 controls: marker, definition, unchanged inputs and output, schema, keys, months, ranges.

    The provenance checks are not optional: a year whose inputs, output file or definition changed is rejected.
    """
    directory = paths.h2_controls
    path, marker = directory / f"{oos_year}.parquet", directory / f"{oos_year}.json"
    if not (path.exists() and marker.exists()):
        raise DataValidationError(f"{oos_year}: no complete H2 controls (parquet and JSON marker).")
    label = f"{oos_year} H2 controls"
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    if metadata.get("definition_sha256") != DEFINITION_SHA256:
        raise DataValidationError(f"{label}: built with another control definition; rerun the controls.")
    recorded = metadata.get("inputs_sha256", {})
    for name, source in input_files(paths, oos_year).items():
        if not source.exists() or recorded.get(name) != _sha256(source):
            raise DataValidationError(f"{label}: the {name} file changed since the controls were built; rerun.")
    if metadata.get("output_sha256") != _sha256(path):
        raise DataValidationError(f"{label}: the output file does not match the SHA-256 in its marker; rerun.")
    frame = pd.read_parquet(path)
    if list(frame.columns) != list(OUTPUT_COLUMNS):
        raise DataValidationError(f"{label}: columns {list(frame.columns)}, expected {list(OUTPUT_COLUMNS)}.")
    require_unique_keys(frame, label)
    _require_oos_months(frame, oos_year, label)
    ranks = frame.loc[:, list(RANKS)].to_numpy(dtype="float64")
    finite = np.isfinite(ranks)
    if not ((ranks[finite] >= -1.0).all() and (ranks[finite] <= 1.0).all()):
        raise DataValidationError(f"{label}: finite ranks must lie in [-1, 1].")
    if not (frame["complete_case"].to_numpy(dtype=bool) == finite.all(axis=1)).all():
        raise DataValidationError(f"{label}: complete_case does not match the presence of the three ranks.")
    return _sorted(frame)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m disagreement.controls.h2")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--year", action="append", type=int, help="OOS year, repeatable; default: all")
    args = parser.parse_args(argv)
    try:
        run_h2_controls(DataPaths(args.data_dir), args.config, args.year)
    except DataValidationError as exc:
        print(f"Data validation failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
