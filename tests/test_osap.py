"""Streaming OSAP reader (synthetic data)."""

import zipfile

import numpy as np
import pandas as pd
import pytest

from disagreement.data import osap
from disagreement.data.validation import DataValidationError
from synthetic import write_osap_zip


def _frame(n_permnos: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        [(p, m) for p in range(1, n_permnos + 1) for m in (200001, 200002, 200003)],
        columns=["permno", "yyyymm"],
    )
    frame["SigA"] = rng.normal(size=len(frame))
    frame["SigB"] = rng.normal(size=len(frame))
    frame.loc[frame["yyyymm"] == 200001, "SigB"] = np.nan
    return frame


def test_zip_is_streamed_in_typed_blocks(tmp_path):
    frame = _frame()
    path = write_osap_zip(tmp_path / "osap.zip", frame)
    assert osap.signal_columns(osap.read_header(path)) == ["SigA", "SigB"]
    batches = list(osap.iter_frames(path, block_size=16 * 1024))
    assert len(batches) > 1
    out = pd.concat(batches, ignore_index=True)
    assert str(out["permno"].dtype) == str(out["yyyymm"].dtype) == "int64"
    pd.testing.assert_frame_equal(out, frame)


def test_float_formatted_identifiers_are_accepted(tmp_path):
    frame = _frame(5)
    frame["permno"] = frame["permno"].astype(float)  # written as "1.0"
    out = pd.concat(osap.iter_frames(write_osap_zip(tmp_path / "osap.zip", frame)))
    assert len(out) == 15 and str(out["permno"].dtype) == "int64"


def test_invalid_identifiers_are_rejected(tmp_path):
    frame = _frame(5)
    frame["permno"] = frame["permno"] + 0.5
    with pytest.raises(DataValidationError, match="non-integer permno"):
        list(osap.iter_frames(write_osap_zip(tmp_path / "a.zip", frame)))
    with pytest.raises(DataValidationError, match="yyyymm"):
        list(osap.iter_frames(write_osap_zip(tmp_path / "b.zip", _frame(5).drop(columns=["yyyymm"]))))


def test_parse_errors_name_the_column_but_not_the_value(tmp_path):
    path = tmp_path / "osap.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("x.csv", "permno,yyyymm,SigA\n10,200001,0.5\n11,200001,SECRETVALUE\n")
    with pytest.raises(DataValidationError) as excinfo:
        list(osap.iter_frames(path))
    assert "SigA" in str(excinfo.value)
    assert "SECRETVALUE" not in str(excinfo.value)


def test_archive_must_hold_exactly_one_csv(tmp_path):
    path = tmp_path / "osap.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("a.csv", "permno,yyyymm\n")
        archive.writestr("b.csv", "permno,yyyymm\n")
    with pytest.raises(DataValidationError, match="exactly one CSV"):
        osap.read_header(path)
