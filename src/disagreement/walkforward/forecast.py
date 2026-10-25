"""Strictly out-of-sample bootstrap-replica forecasts of the four model families.

For each annual episode with OOS year Y:

1. the eligible predictors are determined once, from the raw panel of the
   training block (cutoff ``train_end``), and are shared by all families;
2. they are ranked month by month for every month of the episode, before any
   resampling;
3. for each family, every frozen candidate is fitted on the training block
   (finite targets only) and scored on the validation block; the selected
   candidate is then fixed;
4. each replica draws a month-cluster bootstrap of the whole pre-OOS history
   (training and validation), drops the resampled rows without a finite
   target, refits the selected candidate and predicts every stock of the
   twelve OOS formation months.

The bootstrap seed of a replica depends only on the base seed, the OOS year
and the replica, so the four families resample the same months. Model-internal
randomness keeps the tuning seed. Targets are read only through December Y-1:
no outcome of the OOS year is loaded. Raw, unranked predictions are written
per family and OOS year, with a JSON sidecar written last as the completion
marker.

Command line: ``python -m disagreement.walkforward.forecast [--family F] [--year Y]``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from disagreement.bootstrap.months import month_cluster_bootstrap
from disagreement.data import crsp
from disagreement.data.availability import eligible_predictors
from disagreement.data.config import DEFAULT_CONFIG, DEFAULT_DATA_DIR, DataConfig, DataPaths, load_data_config
from disagreement.data.universe import UNIVERSE_COLUMNS
from disagreement.data.validation import KEY, DataValidationError, require_unique_keys
from disagreement.models.elastic_net import fit_elastic_net
from disagreement.models.lightgbm import fit_lightgbm, predict_lightgbm
from disagreement.models.mlp import fit_mlp, predict_mlp
from disagreement.models.random_forest import fit_random_forest, predict_random_forest
from disagreement.models.selection import FAMILIES, model_candidates, select_candidate, tuning_seed, validation_score
from disagreement.preprocessing.ranks import rank_predictors
from disagreement.walkforward.calendar import Episode, baseline_episodes

OUTPUT_COLUMNS = ("permno", "yyyymm", "family", "replica", "prediction")
OUTPUT_KEY = ("permno", "yyyymm", "family", "replica")


def _predict_elastic_net(model, X) -> np.ndarray:
    return model.predict(X)


_FAMILIES = {
    "elastic_net": (fit_elastic_net, _predict_elastic_net),
    "random_forest": (fit_random_forest, predict_random_forest),
    "lightgbm": (fit_lightgbm, predict_lightgbm),
    "mlp": (fit_mlp, predict_mlp),
}


def replica_seed(base_seed: int, oos_year: int, replica: int) -> int:
    """Bootstrap seed of one (OOS year, replica), the same for every family."""
    return int(np.random.SeedSequence([base_seed, oos_year, replica]).generate_state(1, dtype=np.uint32)[0])


def bootstrap_seeds(base_seed: int, oos_year: int, replicas: int) -> list[int]:
    return [replica_seed(base_seed, oos_year, replica) for replica in range(replicas)]


def bootstrap_settings(config_path: Path = DEFAULT_CONFIG) -> tuple[int, int]:
    """(replicas, base_seed) from the ``bootstrap`` section of the configuration."""
    with open(config_path, encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)["bootstrap"]
    replicas, base_seed = int(spec["replicas"]), int(spec["base_seed"])
    if replicas < 1 or base_seed < 0:
        raise ValueError("bootstrap replicas must be positive and base_seed non-negative")
    return replicas, base_seed


@dataclass(frozen=True)
class EpisodeData:
    """Model-ready rows of one episode, ordered by (yyyymm, permno)."""

    episode: Episode
    predictors: list[str]
    keys: pd.DataFrame  # permno, yyyymm
    X: np.ndarray  # monthly ranked predictors
    target: np.ndarray  # target_rank; NaN when missing and for every OOS month

    @property
    def months(self) -> np.ndarray:
        return self.keys["yyyymm"].to_numpy()


def _read_years(directory: Path, prefix: str, start: int, end: int, columns: list[str]) -> pd.DataFrame:
    frames = []
    for year in range(start // 100, end // 100 + 1):
        path = directory / f"{prefix}_{year}.parquet"
        if not path.exists():
            raise DataValidationError(f"missing {path.name}; run build and target first.")
        frame = pd.read_parquet(path, columns=columns)
        frames.append(frame[(frame["yyyymm"] >= start) & (frame["yyyymm"] <= end)])
    return pd.concat(frames, ignore_index=True)


def candidate_predictors(paths: DataPaths, cfg: DataConfig, year: int) -> list[str]:
    """Panel predictor columns: every column except the identifier and universe columns."""
    names = pq.read_schema(paths.panel / f"panel_{year}.parquet").names
    candidates = [c for c in names if c not in UNIVERSE_COLUMNS]
    expected = cfg.osap_predictors + len(crsp.CRSP_PREDICTORS)
    if len(candidates) != expected or not set(crsp.CRSP_PREDICTORS) <= set(candidates):
        raise DataValidationError(f"the panel has {len(candidates)} candidate predictors; expected {expected}.")
    return candidates


def prepare_episode(paths: DataPaths, cfg: DataConfig, episode: Episode) -> EpisodeData:
    """Eligible predictors from the training block, monthly ranks, and pre-OOS targets."""
    candidates = candidate_predictors(paths, cfg, episode.train_start // 100)
    training = _read_years(paths.panel, "panel", episode.train_start, episode.train_end, [*KEY, *candidates])
    predictors = eligible_predictors(
        training,
        candidates,
        cutoff=episode.eligibility_cutoff,
        min_stock_coverage=cfg.min_stock_coverage,
        min_month_share=cfg.min_month_share,
    )
    del training
    if not predictors:
        raise DataValidationError(f"no eligible predictor at cutoff {episode.eligibility_cutoff}.")

    panel = _read_years(paths.panel, "panel", episode.train_start, episode.test_end, [*KEY, *predictors])
    require_unique_keys(panel, "episode panel")
    panel = panel.sort_values(list(KEY)[::-1], ignore_index=True)  # yyyymm, then permno
    ranked = rank_predictors(panel, predictors)

    # Targets are read only through December Y-1: no OOS outcome is loaded.
    targets = _read_years(paths.target, "target", episode.history_start, episode.history_end, [*KEY, "target_rank"])
    merged = ranked.loc[:, list(KEY)].merge(targets, on=list(KEY), how="left", validate="one_to_one", indicator=True)
    if len(merged) != len(ranked) or int((merged["_merge"] == "both").sum()) != len(targets):
        raise DataValidationError("target rows do not match the panel's formation observations.")
    historical = merged["yyyymm"] <= episode.history_end
    n_missing = int((historical & (merged["_merge"] != "both")).sum())
    if n_missing:  # a missing target_rank is valid; a missing target row is not
        raise DataValidationError(f"{n_missing} historical panel observations have no target row.")
    return EpisodeData(
        episode=episode,
        predictors=predictors,
        keys=ranked.loc[:, list(KEY)],
        X=ranked[predictors].to_numpy(dtype="float64"),
        target=merged["target_rank"].to_numpy(dtype="float64", na_value=np.nan),
    )


def _in(months: np.ndarray, start: int, end: int) -> np.ndarray:
    return np.flatnonzero((months >= start) & (months <= end))


def run_family_episode(
    data: EpisodeData, family: str, candidates: Sequence[dict], seeds: Sequence[int]
) -> tuple[pd.DataFrame, dict]:
    """Select one candidate on training -> validation, then fit and predict one replica per seed."""
    fit, predict = _FAMILIES[family]
    e, months, X, y = data.episode, data.months, data.X, data.target
    finite = np.isfinite(y)
    train_rows = _in(months, e.train_start, e.train_end)
    fit_rows = train_rows[finite[train_rows]]
    validation_rows = _in(months, e.validation_start, e.validation_end)
    history_rows = _in(months, e.history_start, e.history_end)
    oos_rows = _in(months, e.test_start, e.test_end)
    if not (len(fit_rows) and len(validation_rows) and len(oos_rows)):
        raise DataValidationError(f"episode {e.oos_year}: empty training, validation or OOS block.")

    scores = []
    for params in candidates:  # no resampling during selection
        model = fit(X[fit_rows], y[fit_rows], params)
        prediction = predict(model, X[validation_rows])
        scores.append(validation_score(months[validation_rows], prediction, y[validation_rows]))
    best = select_candidate(scores)
    params = candidates[best]

    predictions, fit_counts = [], []
    for seed in seeds:
        _, sample = month_cluster_bootstrap(months[history_rows], seed=seed)  # every history row is drawable
        rows = history_rows[sample]
        rows = rows[finite[rows]]  # missing targets only leave the supervised fit
        model = fit(X[rows], y[rows], params)
        prediction = np.asarray(predict(model, X[oos_rows]), dtype="float64")
        if prediction.shape != (len(oos_rows),) or not np.isfinite(prediction).all():
            raise DataValidationError(f"{family} {e.oos_year}: invalid OOS predictions.")
        predictions.append(prediction)
        fit_counts.append(len(rows))

    keys = data.keys.iloc[oos_rows]
    n = len(oos_rows)
    frame = pd.DataFrame(
        {
            "permno": np.tile(keys["permno"].to_numpy(), len(seeds)),
            "yyyymm": np.tile(keys["yyyymm"].to_numpy(), len(seeds)),
            "family": family,
            "replica": np.repeat(np.arange(len(seeds), dtype="int64"), n),
            "prediction": np.concatenate(predictions),
        }
    ).sort_values(["yyyymm", "permno", "replica"], ignore_index=True)
    require_unique_keys(frame, "OOS predictions", OUTPUT_KEY)

    metadata = {
        "family": family,
        "oos_year": e.oos_year,
        "train_start": e.train_start,
        "train_end": e.train_end,
        "validation_start": e.validation_start,
        "validation_end": e.validation_end,
        "test_start": e.test_start,
        "test_end": e.test_end,
        "eligibility_cutoff": e.eligibility_cutoff,
        "n_eligible_predictors": len(data.predictors),
        "eligible_predictors": data.predictors,
        "selected_candidate_index": best,
        "selected_params": dict(params),
        "selected_validation_score": scores[best],
        "validation_scores": scores,
        "replicas": len(seeds),
        "bootstrap_seeds": list(seeds),
        "n_training_rows": len(train_rows),
        "n_training_fit_rows": len(fit_rows),
        "n_validation_rows": len(validation_rows),
        "n_history_rows": len(history_rows),
        "n_history_rows_without_target": int((~finite[history_rows]).sum()),
        "replica_fit_rows": fit_counts,
        "n_oos_rows": n,
    }
    return frame, metadata


def _completion_marker(paths: DataPaths, family: str, oos_year: int) -> Path:
    return paths.oos_predictions / family / f"{oos_year}.json"


def _write(directory: Path, oos_year: int, frame: pd.DataFrame, metadata: dict) -> None:
    """Atomic writes; the JSON sidecar, replaced last, marks a complete family-year."""
    directory.mkdir(parents=True, exist_ok=True)
    predictions_path, metadata_path = directory / f"{oos_year}.parquet", directory / f"{oos_year}.json"
    metadata_path.unlink(missing_ok=True)
    tmp_predictions = predictions_path.with_suffix(".parquet.tmp")
    tmp_metadata = metadata_path.with_suffix(".json.tmp")
    try:
        frame.loc[:, list(OUTPUT_COLUMNS)].to_parquet(tmp_predictions, index=False)
        os.replace(tmp_predictions, predictions_path)
        text = json.dumps(metadata, indent=2)  # fully serialized before anything is written
        tmp_metadata.write_text(text, encoding="utf-8")
        os.replace(tmp_metadata, metadata_path)  # the last operation: marks success
    finally:
        tmp_predictions.unlink(missing_ok=True)
        tmp_metadata.unlink(missing_ok=True)


def run_baseline_forecasts(
    paths: DataPaths,
    config_path: Path = DEFAULT_CONFIG,
    families: Iterable[str] | None = None,
    years: Iterable[int] | None = None,
    log=print,
) -> list[dict]:
    """Run the selected families and OOS years of the baseline (all by default)."""
    cfg = load_data_config(config_path)
    replicas, base_seed = bootstrap_settings(config_path)
    families = list(FAMILIES if families is None else families)
    unknown = sorted(set(families) - set(FAMILIES))
    if unknown:
        raise ValueError(f"unknown model families: {unknown}")
    episodes = baseline_episodes(config_path)
    if years is not None:
        wanted = set(years)
        missing = sorted(wanted - {e.oos_year for e in episodes})
        if missing:
            raise ValueError(f"not baseline OOS years: {missing}")
        episodes = [e for e in episodes if e.oos_year in wanted]

    model_seed = tuning_seed(config_path)
    summaries = []
    for episode in episodes:
        data = prepare_episode(paths, cfg, episode)
        seeds = bootstrap_seeds(base_seed, episode.oos_year, replicas)
        for family in families:
            # The family-year is not complete until its metadata is rewritten.
            _completion_marker(paths, family, episode.oos_year).unlink(missing_ok=True)
            frame, metadata = run_family_episode(data, family, model_candidates(family, config_path), seeds)
            metadata["model_seed"] = model_seed
            _write(paths.oos_predictions / family, episode.oos_year, frame, metadata)
            log(
                f"{family} {episode.oos_year}: candidate {metadata['selected_candidate_index']} "
                f"(validation rank IC {metadata['selected_validation_score']:.4f}), "
                f"{len(frame):,} OOS predictions"
            )
            summaries.append(metadata)
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m disagreement.walkforward.forecast")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--family", action="append", choices=FAMILIES, help="repeatable; default: all four")
    parser.add_argument("--year", action="append", type=int, help="OOS year, repeatable; default: all")
    args = parser.parse_args(argv)
    try:
        run_baseline_forecasts(DataPaths(args.data_dir), args.config, args.family, args.year)
    except DataValidationError as exc:
        print(f"Data validation failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
