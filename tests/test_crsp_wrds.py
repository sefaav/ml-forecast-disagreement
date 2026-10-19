"""CRSP extraction against a fake WRDS connection (no WRDS access)."""

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from disagreement.data import crsp
from disagreement.data.cli import main
from disagreement.data.config import DataPaths
from disagreement.data.validation import DataValidationError
from synthetic import crsp_row


class FakeConnection:
    """Mimics the parts of wrds.Connection used by the extraction."""

    def __init__(self, last="2001-12-31", columns=(*crsp.REQUIRED_FIELDS, "ticker"), fail_year=None):
        self.last = pd.Timestamp(last)
        self.fail_year = fail_year
        self.columns = list(columns)
        self.queries = []

    def describe_table(self, library, table):
        return pd.DataFrame({"name": self.columns, "type": "VARCHAR"})

    def raw_sql(self, sql, date_cols=None, **_kwargs):
        self.queries.append(sql)
        if "min(mthcaldt)" in sql:
            return pd.DataFrame({"first_date": [pd.Timestamp("2000-01-31")], "last_date": [self.last]})
        year = int(re.findall(r"'(\d{4})-01-01'", sql)[0])
        if year == self.fail_year:
            raise ConnectionError("connection lost")
        last_month = min(self.last, pd.Timestamp(year, 12, 31)).strftime("%Y-%m")
        frame = pd.DataFrame([crsp_row(1, f"{year}-01", 10.0), crsp_row(2, last_month, 20.0)])
        frame["permno"] = frame["permno"].astype("Int64")  # wrds' nullable dtypes
        frame["mthcap"] = pd.array([10.0, None], dtype="Float64")
        return frame

    def close(self):
        pass

    def downloaded_years(self):
        return [int(y) for q in self.queries for y in re.findall(r"between '(\d{4})", q)]


def _extract(conn, out_dir):
    return crsp.extract_monthly(conn, "crsp", "stkmthsecuritydata", out_dir, log=lambda _: None)


def test_extract_writes_required_fields_by_year(tmp_path):
    conn = FakeConnection()
    out = tmp_path / "crsp"
    manifest = _extract(conn, out)
    data_query = next(q for q in conn.queries if "between" in q)
    assert re.findall(r"select (.*) from", data_query)[0].split(", ") == list(crsp.REQUIRED_FIELDS)
    frame = pd.read_parquet(out / "crsp_monthly_2000.parquet")
    assert str(frame["permno"].dtype) == "int64" and frame["mthcap"].isna().sum() == 1
    assert manifest["rows_by_year"] == {"2000": 2, "2001": 2}
    assert json.loads((out / "_manifest.json").read_text()) == manifest
    assert "username" not in json.dumps(manifest).lower()


def test_extraction_is_fresh_and_replaces_the_previous_extract_only_on_success(tmp_path, monkeypatch):
    out = tmp_path / "crsp"
    _extract(FakeConnection(last="2001-06-30"), out)
    before = {f.name: f.read_bytes() for f in out.iterdir()}

    with pytest.raises(ConnectionError):
        _extract(FakeConnection(last="2001-12-31", fail_year=2001), out)
    assert {f.name: f.read_bytes() for f in out.iterdir()} == before  # previous extract intact
    assert [d.name for d in tmp_path.iterdir()] == ["crsp"]  # no partial directory left

    rename = Path.rename

    def failing_rename(self, target):
        if self.name.endswith(".partial"):
            raise OSError("replacement failed")
        return rename(self, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "rename", failing_rename)
        with pytest.raises(OSError, match="replacement failed"):
            _extract(FakeConnection(last="2001-12-31"), out)
    assert {f.name: f.read_bytes() for f in out.iterdir()} == before  # previous extract restored

    fresh = FakeConnection(last="2001-12-31")
    manifest = _extract(fresh, out)
    assert fresh.downloaded_years() == [2000, 2001]  # every year again, nothing reused
    assert manifest["table_last_date"] == "2001-12-31"
    assert pd.read_parquet(out / "crsp_monthly_2001.parquet")["mthcaldt"].max() == pd.Timestamp("2001-12-31")
    assert [d.name for d in tmp_path.iterdir()] == ["crsp"]


def test_extract_stops_when_a_required_field_is_missing(tmp_path):
    conn = FakeConnection(columns=[f for f in crsp.REQUIRED_FIELDS if f != "mthdelflg"])
    with pytest.raises(DataValidationError, match="mthdelflg"):
        _extract(conn, tmp_path)
    assert not list(tmp_path.glob("*.parquet"))
    with pytest.raises(ValueError):
        crsp.extract_monthly(FakeConnection(), "crsp", "msf; drop table x", tmp_path)


def _derived_outputs(paths):
    """Placeholder outputs of an earlier build and target run."""
    outputs = [
        paths.universe / "universe_2000.parquet",
        paths.panel / "panel_2000.parquet",
        paths.target / "target_2000.parquet",
        paths.diagnostics / "summary.json",
        paths.diagnostics / "universe_by_month.csv",
    ]
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("from an earlier build")
    return outputs


def test_successful_extraction_invalidates_derived_outputs(tmp_path, monkeypatch, capsys):
    paths = DataPaths(tmp_path / "data")
    outputs = _derived_outputs(paths)
    monkeypatch.setattr(crsp, "connect", lambda _username: FakeConnection())
    assert main(["crsp-extract", "--data-dir", str(paths.root)]) == 0
    assert not any(path.exists() for path in outputs)
    assert not any(d.exists() for d in (paths.universe, paths.panel, paths.target))
    assert "run build again" in capsys.readouterr().out


def test_failed_extraction_keeps_derived_outputs(tmp_path, monkeypatch):
    paths = DataPaths(tmp_path / "data")
    outputs = _derived_outputs(paths)
    monkeypatch.setattr(crsp, "connect", lambda _username: FakeConnection(fail_year=2001))
    with pytest.raises(ConnectionError):
        main(["crsp-extract", "--data-dir", str(paths.root)])
    assert all(path.read_text() == "from an earlier build" for path in outputs)
