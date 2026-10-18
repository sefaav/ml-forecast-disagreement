"""Formation-time universe U_t (synthetic data)."""

import numpy as np
import pandas as pd
import pytest

from disagreement.data import crsp
from disagreement.data.universe import UNIVERSE_COLUMNS, build_universe
from disagreement.data.validation import DataValidationError
from synthetic import crsp_row, formation_row

T = 200001
NYSE_CAPS = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]  # 20th percentile: 20


def _nyse(caps=NYSE_CAPS):
    return [formation_row(p, T, "NYSE", c) for p, c in enumerate(caps, start=1)]


def _build(frame):
    members, by_month = build_universe(pd.DataFrame(frame))
    return set(members.loc[members["yyyymm"] == T, "permno"]), by_month.set_index("yyyymm").loc[[T]].iloc[0]


def test_nyse_breakpoint_from_eligible_nyse_stocks_applies_to_all_three_exchanges():
    rows = _nyse() + [
        formation_row(101, T, "NASDAQ", 0.5),  # small non-NYSE stocks do not move the breakpoint
        formation_row(102, T, "AMEX", 1e6),
        formation_row(103, T, "NYSE", 1.0, is_common=False),
        formation_row(104, T, "NYSE", np.nan),
        formation_row(201, T, "AMEX", 25.0),
        formation_row(202, T, "AMEX", 15.0),
        formation_row(203, T, "NASDAQ", 25.0),
        formation_row(204, T, "NASDAQ", 15.0),
        formation_row(205, T, None, 1e6),  # other exchange: never in U_t
    ]
    members, stats = _build(rows)
    assert stats["n_nyse_eligible"] == 6
    assert stats["nyse_breakpoint"] == pytest.approx(np.quantile(NYSE_CAPS, 0.2))
    assert members == {2, 3, 4, 5, 6, 102, 201, 203}


def test_stock_exactly_at_the_breakpoint_is_retained():
    caps = [12.5, 31.0, 47.25, 80.0, 99.0, 150.0, 400.0]
    breakpoint = float(np.quantile(caps, 0.2))  # interpolated, not an observed cap
    members, _ = _build(
        _nyse(caps)
        + [
            formation_row(301, T, "NASDAQ", breakpoint),
            formation_row(302, T, "AMEX", np.nextafter(breakpoint, -np.inf)),
        ]
    )
    assert 301 in members and 302 not in members


@pytest.mark.parametrize("dtype", ["float64", "Float64"])
def test_invalid_market_cap_is_neither_eligible_nor_in_the_breakpoint(dtype):
    invalid = [("NASDAQ", np.nan), ("NASDAQ", 0.0), ("NYSE", -5.0), ("NYSE", np.inf), ("NYSE", np.nan)]
    rows = _nyse(NYSE_CAPS[:5]) + [formation_row(400 + k, T, e, c) for k, (e, c) in enumerate(invalid)]
    frame = pd.DataFrame(rows)
    frame["mktcap"] = frame["mktcap"].astype(dtype)
    members, stats = _build(frame)
    assert stats["n_eligible"] == stats["n_nyse_eligible"] == 5
    assert stats["nyse_breakpoint"] == pytest.approx(np.quantile(NYSE_CAPS[:5], 0.2))
    assert not members & set(range(400, 405))


def test_duplicate_permno_month_is_rejected_without_quoting_rows():
    rows = [formation_row(7, T, "NYSE", 123.25), formation_row(7, T, "NYSE", 456.5)]
    with pytest.raises(DataValidationError) as excinfo:
        build_universe(pd.DataFrame(rows))
    assert "1 rows repeat" in str(excinfo.value)
    assert "123.25" not in str(excinfo.value) and "456.5" not in str(excinfo.value)


def test_month_without_eligible_nyse_stocks_has_no_breakpoint_and_no_members():
    rows = [
        formation_row(1, T, "NASDAQ", 50.0),
        formation_row(2, T, "AMEX", 70.0),
        formation_row(3, T, "NYSE", 90.0, is_common=False),
    ]
    members, stats = _build(rows)
    assert np.isnan(stats["nyse_breakpoint"])
    assert stats["n_universe"] == 0 and not members


def test_output_contains_formation_fields_only():
    rows = [formation_row(p, T, "NYSE", 10.0 * p, mthret=0.1, ret_next_month=0.2) for p in range(1, 6)]
    members, _ = build_universe(pd.DataFrame(rows))
    assert tuple(members.columns) == UNIVERSE_COLUMNS


# --- Full path from raw CIZ records ----------------------------------------

MONTH_T = [
    *(crsp_row(p, "2000-01", cap) for p, cap in enumerate(NYSE_CAPS, start=1)),
    crsp_row(7, "2000-01", 25.0, exch="Q"),
    crsp_row(8, "2000-01", 15.0, exch="A"),
]


def _universe_at_t(rows):
    members, by_month = build_universe(crsp.normalize_monthly(pd.DataFrame(rows)))
    return members.loc[members["yyyymm"] == T].reset_index(drop=True), by_month.set_index("yyyymm").loc[[T]].iloc[0]


def test_membership_depends_on_month_t_records_only():
    base_members, base_stats = _universe_at_t(MONTH_T)
    assert set(base_members["permno"]) == {2, 3, 4, 5, 6, 7}
    variants = {  # a member that delists in t+1, or disappears, stays in U_t
        "month-t returns": [dict(row, mthret=-0.5) for row in MONTH_T],
        "t+1 returns": MONTH_T + [crsp_row(p, "2000-02", 35.0, mthret=-0.9) for p in range(1, 9)],
        "t+1 delistings": MONTH_T
        + [
            crsp_row(2, "2000-02", 0.5, mthret=-0.95, mthretflg="DE", mthdelflg="A", tradingstatusflg="S"),
            crsp_row(7, "2000-02", 0.5, mthret=-0.95, mthdelflg="P"),
        ],
        "t+1 invalid records": MONTH_T
        + [crsp_row(p, "2000-02", np.nan, exch=None, mthdelflg="G") for p in range(1, 9)],
    }
    for name, rows in variants.items():
        members, stats = _universe_at_t(rows)
        pd.testing.assert_frame_equal(base_members, members, obj=name)
        pd.testing.assert_series_equal(base_stats, stats, obj=name)


def test_formation_month_exclusions_also_leave_the_breakpoint():
    excluded = [
        crsp_row(50, "2000-01", 1.0, tradingstatusflg="H"),  # NYSE, halted: would lower the breakpoint
        crsp_row(51, "2000-01", 1e6, exch="Q", tradingstatusflg="S"),  # large, but suspended
        crsp_row(70, "2000-01", 45.0, exch="Q", mthretflg="DE"),
        crsp_row(71, "2000-01", 1.0, mthretflg="DE"),  # NYSE: would lower the breakpoint
        crsp_row(72, "2000-01", 2.0, mthdelflg="G"),  # NYSE, delisting return excluded from MthRet
        *(crsp_row(73 + k, "2000-01", 45.0, exch="A", mthdelflg=flag) for k, flag in enumerate("APVGM")),
    ]
    base_members, base_stats = _universe_at_t(MONTH_T)
    members, stats = _universe_at_t(MONTH_T + excluded)
    pd.testing.assert_frame_equal(base_members, members)
    pd.testing.assert_series_equal(base_stats, stats)
