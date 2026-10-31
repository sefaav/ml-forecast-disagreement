"""End-to-end build on synthetic files and its input and output checks."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from disagreement.data import crsp, osap
from disagreement.data.cli import main
from disagreement.data.config import DEFAULT_CONFIG, DataPaths, load_data_config
from disagreement.data.universe import UNIVERSE_COLUMNS
from synthetic import crsp_row, write_osap_zip

PERMNOS = list(range(987001, 987041))  # synthetic identifiers, active securities
HALTED = 987041  # NYSE common stock with a non-active trading status
DELISTS = 987042  # AMEX stock alive in 2000-11 and 2000-12, delisted during 2001-01
DELFLG_G = 987043  # AMEX stock in 2001-02 only, MthDelFlg 'G' (return excluded): excluded
DELFLG_P = 987044  # AMEX stock in 2001-02 only, MthDelFlg 'P' (return included): excluded
INF_KEY = (987020, 200101)  # an OSAP value set to +infinity
CRSP_MONTHS = ["2000-11", "2000-12", "2001-01", "2001-02"]
OSAP_MONTHS = [200012, 200101, 200102, 200103]
MISSING_FROM_OSAP = {987010, 987011, 987012}
SIGNALS = ["SigA", "SigB"]
ALL_SYNTHETIC = [*PERMNOS, HALTED, DELISTS, DELFLG_G, DELFLG_P]
OSAP_FILE = load_data_config().osap_file


def _exchange_and_cap(permno: int) -> tuple[str, float]:
    k = permno - 987000
    if k <= 20:
        return "N", 1000.0 + 37.0 * k
    if k <= 25:
        return "A", 1000.0 + 37.0 * k
    return "Q", 100.0 * (k - 25)


def _osap_frame() -> pd.DataFrame:
    keys = [(p, m) for p in [*PERMNOS, HALTED, 987045, 987046] for m in OSAP_MONTHS if p not in MISSING_FROM_OSAP]
    frame = pd.DataFrame(keys, columns=["permno", "yyyymm"])
    frame["SigA"] = np.linspace(0.1, 0.9, len(frame))
    frame["SigB"] = np.where(frame["yyyymm"] == 200012, np.nan, 1.5)
    frame.loc[(frame["permno"] == INF_KEY[0]) & (frame["yyyymm"] == INF_KEY[1]), "SigA"] = np.inf
    return frame


def _setup(tmp_path, osap_frame=None, crsp_months=CRSP_MONTHS, **osap_config):
    """Synthetic CRSP extract and OSAP archive, and a config pinning that archive."""
    paths = DataPaths(tmp_path / "data")
    rows = [
        crsp_row(p, month, cap, exch=exch)
        for month in CRSP_MONTHS
        for p in PERMNOS
        for exch, cap in [_exchange_and_cap(p)]
    ]
    rows += [crsp_row(HALTED, month, 5000.0, exch="N", tradingstatusflg="H") for month in CRSP_MONTHS]
    rows += [crsp_row(DELISTS, month, 3000.0, exch="A") for month in CRSP_MONTHS[:2]]
    rows.append(crsp_row(DELISTS, CRSP_MONTHS[2], 2900.0, exch="A", mthretflg="DE", mthret=-0.6))
    rows.append(crsp_row(DELFLG_G, CRSP_MONTHS[3], 3100.0, exch="A", mthretflg="MP", mthdelflg="G"))
    rows.append(crsp_row(DELFLG_P, CRSP_MONTHS[3], 3200.0, exch="A", mthdelflg="P"))
    rows = [r for r in rows if r["mthcaldt"].strftime("%Y-%m") in crsp_months]
    raw = crsp.coerce_raw_types(pd.DataFrame(rows))
    cfg = load_data_config()
    raw_dir = paths.crsp(cfg.crsp_table)
    raw_dir.mkdir(parents=True)
    manifest = {
        "library": cfg.crsp_library,
        "table": cfg.crsp_table,
        "fields": list(crsp.REQUIRED_FIELDS),
        "table_first_date": str(raw["mthcaldt"].min().date()),
        "table_last_date": str(raw["mthcaldt"].max().date()),
        "rows_by_year": {},
    }
    for year, part in raw.groupby(raw["mthcaldt"].dt.year):
        part.to_parquet(raw_dir / f"crsp_monthly_{year}.parquet", index=False)
        manifest["rows_by_year"][str(year)] = len(part)
    (raw_dir / "_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    paths.osap.mkdir(parents=True)
    archive = write_osap_zip(paths.osap / OSAP_FILE, _osap_frame() if osap_frame is None else osap_frame)
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["data"]["sources"]["osap"].update(
        {"sha256": osap.sha256_file(archive), "predictors": len(SIGNALS), **osap_config}
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return paths, ["build", "--data-dir", str(paths.root), "--config", str(config_path)]


def _expected_universe_size() -> int:
    caps = [_exchange_and_cap(p) for p in PERMNOS]
    breakpoint = np.quantile([c for e, c in caps if e == "N"], 0.2)
    return sum(c >= breakpoint for _, c in caps)


def _concat(files):
    return pd.concat([pd.read_parquet(f) for f in sorted(files)], ignore_index=True)


def test_build_end_to_end_on_synthetic_files(tmp_path, capsys):
    paths, argv = _setup(tmp_path)
    for stale in (paths.oos_predictions / "elastic_net" / "1985.json", paths.family_signals / "elastic_net" / "1985.json"):
        stale.parent.mkdir(parents=True)
        stale.write_text("from an earlier build")
    assert main(argv) == 0
    assert not paths.oos_predictions.exists() and not paths.family_signals.exists()  # stale downstream outputs
    summary = json.loads((paths.diagnostics / "summary.json").read_text())
    captured = capsys.readouterr()
    output = (captured.out + captured.err).replace(summary["osap_sha256"], "")
    assert not any(str(p) in output for p in ALL_SYNTHETIC)  # aggregates only

    universe = _concat(paths.universe.glob("universe_*.parquet"))
    assert not {HALTED, DELFLG_G, DELFLG_P} & set(universe["permno"])
    assert universe.loc[universe["permno"] == DELISTS, "yyyymm"].tolist() == [200011, 200012]
    panel = _concat(paths.panel.glob("panel_*.parquet"))
    assert list(panel.columns) == [*UNIVERSE_COLUMNS, *SIGNALS, *crsp.CRSP_PREDICTORS]  # no target
    expected = _expected_universe_size()
    assert len(panel) == len(universe) == 4 * expected + 2
    assert not paths.staging.exists()

    indexed = panel.set_index(["permno", "yyyymm"])
    member = (987020, 200012)
    assert indexed.at[member, "Price"] == pytest.approx(-np.log(10.0))
    assert indexed.at[member, "Size"] == pytest.approx(-np.log(1000.0 + 37.0 * 20))
    assert indexed.at[member, "STreversal"] == pytest.approx(-1.0)  # MthRet 1%
    assert pd.isna(indexed.at[INF_KEY, "SigA"])  # inf -> missing
    assert all(pd.isna(indexed.at[(987010, 200101), s]) for s in SIGNALS)  # absent from OSAP: kept, no values
    assert np.isfinite(panel[[*SIGNALS, *crsp.CRSP_PREDICTORS]].fillna(0).to_numpy()).all()

    by_month = pd.read_csv(paths.diagnostics / "universe_by_month.csv")
    assert by_month["n_universe"].tolist() == [expected + 1, expected + 1, expected, expected]
    assert summary["validation"] == "PASS"
    assert (summary["first_month"], summary["last_month"], summary["months"]) == (200011, 200102, 4)
    assert summary["security_months"] == len(panel)
    assert summary["osap_join_rate"] == pytest.approx(3 * (expected - 3) / len(panel))
    assert summary["candidate_predictors"] == 5
    assert summary["predictors_without_usable_values"] == []
    assert summary["infinite_values_set_missing"] == 1


def test_failed_build_leaves_no_current_diagnostics_or_mixed_outputs(tmp_path, capsys):
    paths, argv = _setup(tmp_path)
    assert main(argv) == 0  # an earlier successful build
    frame = _osap_frame()
    archive = write_osap_zip(paths.osap / OSAP_FILE, pd.concat([frame, frame.iloc[[5]]]))
    config_path = Path(argv[-1])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["data"]["sources"]["osap"]["sha256"] = osap.sha256_file(archive)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    capsys.readouterr()

    assert main(argv) == 2  # the inputs pass their checks; fails after the new universe
    assert "OSAP: 1 rows repeat" in capsys.readouterr().err
    assert not (paths.diagnostics / "summary.json").exists()
    assert not (paths.diagnostics / "universe_by_month.csv").exists()
    assert not list(paths.panel.glob("*"))  # no earlier panel beside the new universe


def _edit_manifest(raw_dir, **changes):
    path = raw_dir / "_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(changes)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def _stale_terminal_year(raw_dir):
    _edit_manifest(raw_dir, table_last_date="2001-03-31")  # live CRSP reaches further than the files


def _row_from_another_year(raw_dir):
    early, late = (raw_dir / f"crsp_monthly_{y}.parquet" for y in (2000, 2001))
    mixed = pd.concat([pd.read_parquet(early), pd.read_parquet(late).iloc[[0]]], ignore_index=True)
    mixed.to_parquet(early, index=False)
    rows = _edit_manifest(raw_dir)["rows_by_year"]
    _edit_manifest(raw_dir, rows_by_year={**rows, "2000": len(mixed)})


def _row_count_changed(raw_dir):
    rows = _edit_manifest(raw_dir)["rows_by_year"]
    _edit_manifest(raw_dir, rows_by_year={**rows, "2001": rows["2001"] + 1})


def _other_fields(raw_dir):
    _edit_manifest(raw_dir, fields=list(crsp.REQUIRED_FIELDS[:-1]))


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (_stale_terminal_year, "differs from the manifest 200011-200103"),
        (_row_from_another_year, "CRSP 2000: 1 rows fall outside the file's year"),
        (_row_count_changed, "CRSP 2001: row count differs from the manifest"),
        (_other_fields, "unexpected source table or fields"),
    ],
)
def test_crsp_extract_must_match_its_manifest(tmp_path, capsys, corrupt, message):
    paths, argv = _setup(tmp_path)
    corrupt(paths.crsp(load_data_config().crsp_table))
    assert main(argv) == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("columns", "osap_config", "message"),
    [
        (SIGNALS, {"sha256": "0" * 64}, "does not match the SHA-256"),
        (SIGNALS, {"predictors": 3}, "expected 3 predictors, found 2"),
        ([*SIGNALS, "size"], {"predictors": 3}, "collide with panel columns: size"),
        ([*SIGNALS, "sigA"], {"predictors": 3}, "repeated or empty column names"),
    ],
)
def test_osap_input_must_be_the_pinned_release(tmp_path, capsys, columns, osap_config, message):
    frame = _osap_frame()
    for extra in columns[len(SIGNALS):]:
        frame[extra] = 0.0
    paths, argv = _setup(tmp_path, frame, **osap_config)
    assert main(argv) == 2
    assert message in capsys.readouterr().err
    assert not paths.universe.exists()  # checked before any processing


def test_missing_crsp_month_stops_the_build(tmp_path, capsys):
    _, argv = _setup(tmp_path, crsp_months=["2000-11", "2001-01", "2001-02"])
    assert main(argv) == 2
    assert "CRSP: 1 months between 200011 and 200102 have no records" in capsys.readouterr().err
