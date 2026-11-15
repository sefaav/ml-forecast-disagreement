"""Validated loading and stock-month alignment of the inference inputs (synthetic data, temporary directories)."""

import io
import json

import numpy as np
import pandas as pd
import pytest

from disagreement.controls import h2
from disagreement.controls.h2 import RANKS, h2_controls
from disagreement.data.target import TARGET_COLUMNS, rank_target
from disagreement.data.validation import DataValidationError
from disagreement.inference.inputs import (
    CD,
    CW,
    INFERENCE_COLUMNS,
    assemble_inference_year,
    load_inference_year,
    load_target,
)
from disagreement.signals.conditional import SIGNAL_BINS, conditional_disagreement
from disagreement.walkforward.calendar import month_range
from test_h2_controls import _setup

YEAR = 2004


def _sources(seed=0, year=YEAR):
    """Conditional disagreement, H2 controls and targets of one synthetic year, built with the pipeline's functions."""
    rng = np.random.default_rng(seed)
    keys = pd.DataFrame([(p, m) for m in month_range(year * 100 + 1, year * 100 + 12) for p in range(1, 61 + m % 5)],
                        columns=["permno", "yyyymm"])
    signals = keys.copy()
    signals["ensemble_signal"] = rng.integers(0, 33, size=len(keys)) / 32  # ties
    signals["between_family_disagreement_raw"] = rng.uniform(0.0, 0.5, size=len(keys))
    signals["within_family_uncertainty_raw"] = rng.uniform(0.0, 1.0, size=len(keys))
    cond = conditional_disagreement(signals, SIGNAL_BINS)
    inputs = keys.copy()
    inputs["mktcap"] = rng.integers(1, 50, size=len(keys)) * 1000.0
    inputs["RealizedVol"] = -rng.integers(1, 9, size=len(keys)) / 100
    inputs["Illiquidity"] = rng.integers(1, 9, size=len(keys)) / 10
    inputs.loc[rng.random(len(keys)) < 0.15, "Illiquidity"] = np.nan
    inputs["ensemble_signal"] = signals["ensemble_signal"]
    controls = h2_controls(inputs)
    target = keys.copy()
    target["ret_next"] = np.round(rng.normal(0.01, 0.1, size=len(keys)), 3)  # ties
    target.loc[rng.random(len(keys)) < 0.1, "ret_next"] = np.nan  # delisted or missing next-month return
    target["target_rank"] = rank_target(target)
    return cond, controls, target.loc[:, list(TARGET_COLUMNS)]


def _shuffled(frame, seed):
    return frame.sample(frac=1.0, random_state=seed)


def test_sources_are_aligned_by_key_whatever_their_row_order_and_keep_every_member():
    cond, controls, target = _sources()
    out = assemble_inference_year(_shuffled(cond, 1), _shuffled(controls, 2), _shuffled(target, 3), YEAR)
    pd.testing.assert_frame_equal(out, assemble_inference_year(cond, controls, target, YEAR))
    assert list(out.columns) == list(INFERENCE_COLUMNS)
    assert len(out) == len(cond) == len(target)  # the full U_t, no inner join
    assert out[["yyyymm", "permno"]].equals(out[["yyyymm", "permno"]].sort_values(["yyyymm", "permno"],
                                                                                  ignore_index=True))
    by_key = target.set_index(["permno", "yyyymm"])
    joined = out.set_index(["permno", "yyyymm"])
    assert joined["ret_next"].isna().sum() == target["ret_next"].isna().sum() > 0  # missing outcomes stay missing
    np.testing.assert_array_equal(joined["target_rank"].to_numpy(), by_key.loc[joined.index, "target_rank"].to_numpy())
    np.testing.assert_array_equal(joined["ret_next"].to_numpy(), by_key.loc[joined.index, "ret_next"].to_numpy())
    c = cond.set_index(["permno", "yyyymm"]).loc[joined.index]
    for column in ("ensemble_signal", "signal_bin", CD, CW):
        np.testing.assert_array_equal(joined[column].to_numpy(), c[column].to_numpy())
    np.testing.assert_array_equal(joined["centered_signal"].to_numpy(), 2.0 * (c["ensemble_signal"].to_numpy() - 0.5))
    k = controls.set_index(["permno", "yyyymm"]).loc[joined.index]
    for column in (*RANKS, "complete_case"):
        np.testing.assert_array_equal(joined[column].to_numpy(), k[column].to_numpy())  # NaN ranks kept, no imputation
    assert out["complete_case"].dtype == bool and out["signal_bin"].dtype == np.int64
    assert (~out["complete_case"]).sum() > 0 and out.loc[~out["complete_case"], list(RANKS)].isna().any(axis=1).all()


def test_saved_values_are_passed_on_and_never_recomputed():
    cond, controls, target = _sources(seed=1)
    cond = cond.copy()
    rng = np.random.default_rng(5)
    cond[CD] = rng.uniform(0.0, 1.0, size=len(cond))  # not a within-cell rank: never re-ranked
    cond[CW] = rng.uniform(0.0, 1.0, size=len(cond))
    out = assemble_inference_year(cond, controls, target, YEAR).set_index(["permno", "yyyymm"])
    np.testing.assert_array_equal(out[CD].to_numpy(), cond.set_index(["permno", "yyyymm"]).loc[out.index, CD])


def test_target_rank_must_be_the_exact_frozen_normalised_rank_of_ret_next():
    cond, controls, target = _sources(seed=2)
    observed = target["ret_next"].notna()
    ties = target[observed].duplicated(["yyyymm", "ret_next"], keep=False)
    assert ties.any()  # tied returns share their average rank: the exact ranks with ties are accepted
    out = assemble_inference_year(cond, controls, target, YEAR).set_index(["permno", "yyyymm"])
    saved = target.set_index(["permno", "yyyymm"]).loc[out.index, "target_rank"].to_numpy()
    assert np.array_equal(out["target_rank"].to_numpy(), saved, equal_nan=True)  # not a single value changed
    assert np.isnan(out["target_rank"].to_numpy()).sum() == (~observed).sum()  # missing pairs kept
    # order-preserving transformations within [-1, 1] are not the frozen transformation
    cubed = target.assign(target_rank=target["target_rank"] ** 3)
    _rejected("target_rank differs from the frozen normalised rank of ret_next", target=cubed)
    affine = target.assign(target_rank=0.9 * target["target_rank"])
    _rejected("target_rank differs from the frozen normalised rank", target=affine)
    # one stock corrupted by far less than a rank step, keeping order and bounds
    one = target.copy()
    i = one.index[observed & (one["target_rank"].abs() < 0.9)][0]
    one.loc[i, "target_rank"] += 1e-9
    _rejected("target_rank differs from the frozen normalised rank of ret_next in 1 rows", target=one)
    # a month with a single finite return: target 0 is the frozen value and is accepted
    single = target.copy()
    april = single.index[single["yyyymm"] == YEAR * 100 + 4]
    single.loc[april, ["ret_next", "target_rank"]] = np.nan
    single.loc[april[0], ["ret_next", "target_rank"]] = [0.02, 0.0]
    assemble_inference_year(cond, controls, single, YEAR)


def _rejected(message, cond=None, controls=None, target=None, year=YEAR, seed=0):
    base = _sources(seed)
    with pytest.raises(DataValidationError, match=message):
        assemble_inference_year(base[0] if cond is None else cond, base[1] if controls is None else controls,
                                base[2] if target is None else target, year)


def _edit(frame, row, column, value):
    out = frame.copy()
    out.loc[out.index[row], column] = value
    return out


def test_key_errors_between_and_within_sources_are_rejected():
    cond, controls, target = _sources()
    def duplicated(frame):
        return pd.concat([frame, frame.iloc[[7]]], ignore_index=True)

    _rejected(f"{YEAR} conditional disagreement: 1 rows repeat", cond=duplicated(cond))
    _rejected(f"{YEAR} H2 controls: 1 rows repeat", controls=duplicated(controls))
    _rejected(f"{YEAR} target: 1 rows repeat", target=duplicated(target))
    _rejected(r"target covers different stock-months than the conditional disagreement: 1 missing, 0 extra",
              target=target.drop(index=target.index[5]))
    extra = controls.iloc[[0]].copy()
    extra["permno"] = 9999
    _rejected(r"H2 controls covers different stock-months than the conditional disagreement: 0 missing, 1 extra",
              controls=pd.concat([controls, extra], ignore_index=True))
    fractional = cond.copy()
    fractional["permno"] = fractional["permno"].astype(float)
    fractional.loc[fractional.index[3], "permno"] = 2.5
    _rejected("1 rows have a non-integer permno", cond=fractional)
    missing = target.copy()
    missing["permno"] = missing["permno"].astype(float)
    missing.loc[missing.index[4], "permno"] = np.nan
    _rejected("target: 1 rows have a missing permno", target=missing)
    _rejected("non-positive permno", controls=_edit(controls, 2, "permno", 0))
    _rejected("invalid yyyymm", target=_edit(target, 2, "yyyymm", YEAR * 100 + 13))
    # rows of another year, and an incomplete year, in all three sources at once
    def with_next_year(frame):
        return pd.concat([frame, frame.iloc[:3].assign(yyyymm=(YEAR + 1) * 100 + 1)], ignore_index=True)

    with pytest.raises(DataValidationError, match="3 rows outside the OOS year"):
        assemble_inference_year(with_next_year(cond), with_next_year(controls), with_next_year(target), YEAR)

    def without_june(frame):
        return frame[frame["yyyymm"] != YEAR * 100 + 6]

    with pytest.raises(DataValidationError, match=rf"not the 12 OOS months; missing \[{YEAR}06\]"):
        assemble_inference_year(without_june(cond), without_june(controls), without_june(target), YEAR)
    with pytest.raises(DataValidationError, match="rows outside the OOS year"):
        assemble_inference_year(cond, controls, target, YEAR + 1)
    with pytest.raises(ValueError, match="integer year"):
        assemble_inference_year(cond, controls, target, float(YEAR))  # type: ignore[arg-type]


def test_schema_errors_are_rejected_for_every_source():
    cond, controls, target = _sources()
    _rejected("conditional disagreement: columns", cond=cond.drop(columns=[CW]))
    _rejected("H2 controls: columns", controls=controls.assign(extra=1.0))
    _rejected("target: columns", target=target.rename(columns={"target_rank": "rank"}))


def test_malformed_signal_disagreement_and_control_values_are_detected():
    cond, controls, target = _sources()
    _rejected("1 rows have a ensemble_signal that is not finite", cond=_edit(cond, 0, "ensemble_signal", 1.2))
    _rejected("ensemble_signal that is not finite", cond=_edit(cond, 0, "ensemble_signal", np.nan))
    _rejected(f"1 rows have a {CD} that is not finite", cond=_edit(cond, 3, CD, 1.5))
    _rejected(f"1 rows have a {CW} that is not finite", cond=_edit(cond, 3, CW, np.nan))
    _rejected("signal_bin must be an integer", cond=_edit(cond, 1, "signal_bin", 21))
    _rejected("signal_bin must be an integer", cond=cond.assign(signal_bin=cond["signal_bin"].astype(float)))
    moved = cond.copy()
    first = moved.index[moved["signal_bin"] == 5][0]
    moved.loc[first, "signal_bin"] = 6
    _rejected("signal_bin differs from the bins of ensemble_signal in 1 rows", cond=moved)
    _rejected("1 finite control ranks lie outside", controls=_edit(controls, 0, "size_rank", 1.5))
    _rejected("control ranks are infinite", controls=_edit(controls, 0, "illiquidity_rank", np.inf))
    _rejected("complete_case must be strictly Boolean",
              controls=controls.assign(complete_case=controls["complete_case"].astype(int)))


def test_inconsistent_complete_case_and_centered_signal_are_detected():
    cond, controls, target = _sources()
    incomplete = controls.index[~controls["complete_case"]][0]
    complete = controls.index[controls["complete_case"]][0]
    flagged = controls.copy()
    flagged.loc[incomplete, "complete_case"] = True  # a missing rank in a complete case
    _rejected("complete_case disagrees with the presence of the three ranks in 1 rows", controls=flagged)
    unflagged = controls.copy()
    unflagged.loc[complete, "complete_case"] = False  # all three ranks present, not flagged
    _rejected("complete_case disagrees", controls=unflagged)
    shifted = controls.copy()
    shifted.loc[shifted.index[10], "centered_signal"] += 1e-6
    _rejected("centered_signal differs from 2 \\(S - 0.5\\) of the conditional disagreement in 1 rows",
              controls=shifted)
    tiny = controls.copy()
    tiny.loc[tiny.index[10], "centered_signal"] += 1e-14  # floating-point noise only: accepted
    assemble_inference_year(cond, tiny, target, YEAR)
    _rejected("centered_signal must be finite", controls=_edit(controls, 4, "centered_signal", np.nan))


def test_outcome_conventions_are_validated():
    cond, controls, target = _sources()
    observed = target.index[target["ret_next"].notna()]
    missing = target.index[target["ret_next"].isna()]
    _rejected("1 target_rank values lie outside", target=_edit(target, int(observed[0]), "target_rank", 1.5))
    _rejected("never infinite", target=_edit(target, int(observed[0]), "ret_next", np.inf))
    rank_without_return = target.copy()
    rank_without_return.loc[missing[0], "target_rank"] = 0.3
    _rejected("missing in different rows \\(1\\)", target=rank_without_return)
    return_without_rank = target.copy()
    return_without_rank.loc[observed[0], "target_rank"] = np.nan
    _rejected("missing in different rows", target=return_without_rank)
    swapped = target.copy()
    month = swapped.loc[observed, "yyyymm"] == YEAR * 100 + 3
    march = swapped.loc[observed[month.to_numpy()]].sort_values("ret_next")
    low, high = march.index[0], march.index[-1]
    swapped.loc[[low, high], "target_rank"] = swapped.loc[[high, low], "target_rank"].to_numpy()
    _rejected("target_rank is not ordered like ret_next", target=swapped)
    single = target.copy()
    april = single.index[single["yyyymm"] == YEAR * 100 + 4]
    single.loc[april, ["ret_next", "target_rank"]] = np.nan
    single.loc[april[0], ["ret_next", "target_rank"]] = [0.02, 0.4]  # the only return of the month
    _rejected("1 months with a single return do not have target_rank 0", target=single)
    single.loc[april[0], "target_rank"] = 0.0
    out = assemble_inference_year(cond, controls, single, YEAR)
    assert out.loc[out["yyyymm"] == YEAR * 100 + 4, "ret_next"].notna().sum() == 1  # the month keeps all members
    assert (out["yyyymm"] == YEAR * 100 + 4).sum() == len(april)


def test_missing_nullable_signal_bin_and_non_numeric_values_raise_aggregate_errors():
    cond, controls, target = _sources()
    nullable = cond.assign(signal_bin=cond["signal_bin"].astype("Int64"))
    assemble_inference_year(nullable, controls, target, YEAR)  # a complete nullable integer column is valid
    nullable.loc[nullable.index[11], "signal_bin"] = pd.NA
    _rejected("conditional disagreement: 1 rows have a missing signal_bin", cond=nullable)

    secret = "PRIVATE-7731"  # a malformed value that must never appear in an error message
    cases = [
        ("cond", cond, "permno", "conditional disagreement: permno must be numeric, got dtype object"),
        ("cond", cond, "ensemble_signal", "ensemble_signal must be numeric"),
        ("cond", cond, CD, f"{CD} must be numeric"),
        ("controls", controls, "size_rank", "H2 controls: size_rank must be numeric"),
        ("controls", controls, "yyyymm", "H2 controls: yyyymm must be numeric"),
        ("target", target, "ret_next", "target: ret_next must be numeric"),
        ("target", target, "target_rank", "target: target_rank must be numeric"),
    ]
    for source, frame, column, message in cases:
        malformed = frame.assign(**{column: frame[column].astype(object)})
        malformed.loc[malformed.index[3], column] = secret
        with pytest.raises(DataValidationError, match=message) as caught:
            assemble_inference_year(malformed if source == "cond" else cond,
                                    malformed if source == "controls" else controls,
                                    malformed if source == "target" else target, YEAR)
        assert secret not in str(caught.value) and caught.value.__cause__ is None
    boolean = target.assign(ret_next=target["ret_next"].notna())
    _rejected("target: ret_next must be numeric, got dtype bool", target=boolean)


def test_a_month_without_any_outcome_keeps_all_its_members():
    cond, controls, target = _sources(seed=3)
    may = YEAR * 100 + 5
    empty = target.copy()
    empty.loc[empty["yyyymm"] == may, ["ret_next", "target_rank"]] = np.nan
    out = assemble_inference_year(cond, controls, empty, YEAR)
    in_may = out["yyyymm"] == may
    assert in_may.sum() == (cond["yyyymm"] == may).sum() > 0 and may in set(out["yyyymm"])
    assert out.loc[in_may, ["ret_next", "target_rank"]].isna().all().all()
    other = out.loc[~in_may].set_index(["permno", "yyyymm"])
    saved = empty.set_index(["permno", "yyyymm"]).loc[other.index]
    assert np.array_equal(other["target_rank"].to_numpy(), saved["target_rank"].to_numpy(), equal_nan=True)


def _published(tmp_path):
    """Universe, panel, conditional disagreement and H2 controls built by the pipeline, plus a published target."""
    paths, argv = _setup(tmp_path)
    assert h2.main(argv) == 0
    universe = pd.read_parquet(paths.universe / f"universe_{YEAR}.parquet", columns=["permno", "yyyymm"])
    rng = np.random.default_rng(9)
    target = universe.sample(frac=1.0, random_state=4).reset_index(drop=True)
    target["ret_next"] = np.round(rng.normal(0.0, 0.1, size=len(target)), 3)
    target.loc[rng.random(len(target)) < 0.1, "ret_next"] = np.nan
    target["target_rank"] = rank_target(target)
    paths.target.mkdir(parents=True)
    target.loc[:, list(TARGET_COLUMNS)].to_parquet(paths.target / f"target_{YEAR}.parquet", index=False)
    return paths, target


def test_published_files_load_through_the_existing_validated_loaders(tmp_path):
    paths, target = _published(tmp_path)
    out = load_inference_year(paths, YEAR)
    assert list(out.columns) == list(INFERENCE_COLUMNS) and len(out) == len(target)
    assert out["ret_next"].isna().sum() == target["ret_next"].isna().sum()
    pd.testing.assert_frame_equal(load_target(paths, YEAR), pd.read_parquet(paths.target / f"target_{YEAR}.parquet"))
    controls = h2.load_h2_controls(paths, YEAR)
    np.testing.assert_array_equal(out["complete_case"].to_numpy(), controls["complete_case"].to_numpy())


def test_missing_or_tampered_sources_and_provenance_fail_closed(tmp_path):
    paths, _ = _published(tmp_path)
    target_file = paths.target / f"target_{YEAR}.parquet"
    # a target only in the unpublished staging directory is not read
    staged = paths.target.with_name(paths.target.name + ".partial")
    paths.target.rename(staged)
    with pytest.raises(DataValidationError, match="no published target directory"):
        load_inference_year(paths, YEAR)
    staged.rename(paths.target)
    content = target_file.read_bytes()
    target_file.unlink()
    with pytest.raises(DataValidationError, match="no target file"):
        load_inference_year(paths, YEAR)
    pd.read_parquet(io.BytesIO(content)).drop(columns="target_rank").to_parquet(target_file, index=False)
    with pytest.raises(DataValidationError, match="target: columns"):
        load_inference_year(paths, YEAR)
    target_file.write_bytes(content)
    load_inference_year(paths, YEAR)
    # the conditional-disagreement file changed after the controls were built: rejected by the H2 provenance check
    cond_file = paths.conditional_disagreement / f"{YEAR}.parquet"
    original = cond_file.read_bytes()
    frame = pd.read_parquet(cond_file)
    frame.loc[frame.index[0], CW] = 0.123
    frame.to_parquet(cond_file, index=False)
    with pytest.raises(DataValidationError, match="conditional_disagreement file changed"):
        load_inference_year(paths, YEAR)
    cond_file.write_bytes(original)
    # a missing conditional-disagreement marker, a modified H2 output, a changed control definition
    marker = paths.conditional_disagreement / f"{YEAR}.json"
    marker_text = marker.read_text(encoding="utf-8")
    marker.unlink()
    with pytest.raises(DataValidationError, match="no complete conditional disagreement"):
        load_inference_year(paths, YEAR)
    marker.write_text(marker_text, encoding="utf-8")
    controls_file = paths.h2_controls / f"{YEAR}.parquet"
    controls_original = controls_file.read_bytes()
    controls = pd.read_parquet(controls_file)
    controls.loc[controls.index[0], "size_rank"] = 0.5
    controls.to_parquet(controls_file, index=False)
    with pytest.raises(DataValidationError, match="output file does not match the SHA-256"):
        load_inference_year(paths, YEAR)
    controls_file.write_bytes(controls_original)
    controls_marker = paths.h2_controls / f"{YEAR}.json"
    metadata = json.loads(controls_marker.read_text(encoding="utf-8"))
    controls_marker.write_text(json.dumps({**metadata, "definition_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(DataValidationError, match="another control definition"):
        load_inference_year(paths, YEAR)
    controls_marker.write_text(json.dumps(metadata), encoding="utf-8")
    load_inference_year(paths, YEAR)
