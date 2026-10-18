"""CIZ classification, normalization and CRSP predictors (synthetic data)."""

import numpy as np
import pandas as pd
import pytest

from disagreement.data import crsp
from disagreement.data.validation import DataValidationError
from synthetic import crsp_row


def test_common_stock_codes_match_legacy_share_codes_10_and_11():
    rows = [
        crsp_row(1, "2000-01", 10, issuertype="ACOR"),  # legacy SHRCD 10
        crsp_row(2, "2000-01", 10, issuertype="CORP"),  # legacy SHRCD 11
        crsp_row(3, "2000-01", 10, usincflg="N"),  # non-US incorporation
        crsp_row(4, "2000-01", 10, securitytype="FUND", securitysubtype="CEF"),
        crsp_row(5, "2000-01", 10, sharetype="AD"),
        crsp_row(6, "2000-01", 10, issuertype="REIT"),
        crsp_row(7, "2000-01", 10, sharetype=None),
    ]
    assert crsp.is_common_stock(pd.DataFrame(rows)).tolist() == [True, True] + [False] * 5


def test_exchange_requires_active_regular_way_listing_on_nyse_amex_nasdaq():
    rows = [
        crsp_row(1, "2000-01", 10, exch="N"),
        crsp_row(2, "2000-01", 10, exch="A "),  # space-padded CHAR codes
        crsp_row(3, "2000-01", 10, exch="Q", conditionaltype="RW  "),
        crsp_row(4, "2000-01", 10, exch="R"),  # NYSE Arca
        crsp_row(5, "2000-01", 10, exch="N", conditionaltype="WI"),  # when issued
        crsp_row(6, "2000-01", 10, exch="N", tradingstatusflg="H"),  # halted
        crsp_row(7, "2000-01", 10, exch="Q", tradingstatusflg="S"),  # suspended
    ]
    labels = crsp.exchange_labels(pd.DataFrame(rows))
    assert labels.iloc[:3].tolist() == ["NYSE", "AMEX", "NASDAQ"]
    assert labels.iloc[3:].isna().all()


def test_delisting_in_formation_month_from_either_crsp_indicator():
    rows = [
        crsp_row(1, "2000-01", 10),  # MthRetFlg CR, MthDelFlg N
        crsp_row(2, "2000-01", 10, mthretflg="DE", mthdelflg="A"),
        crsp_row(3, "2000-01", 10, mthdelflg="P"),  # delisting return included
        crsp_row(4, "2000-01", 10, mthdelflg="V"),
        crsp_row(5, "2000-01", 10, mthdelflg="G", mthretflg="MP"),  # delisting return excluded
        crsp_row(6, "2000-01", 10, mthdelflg="M "),
        crsp_row(7, "2000-01", 10, mthretflg="DE", mthdelflg=None),
    ]
    assert crsp.delisted_in_month(pd.DataFrame(rows)).tolist() == [False] + [True] * 6


def test_normalize_keys_by_permno_and_month_without_returns():
    rows = [crsp_row(1, "2000-01", 10), crsp_row(1, "2000-02", 11), crsp_row(2, "2000-01", 12)]
    norm = crsp.normalize_monthly(pd.DataFrame(rows))
    assert norm[["permno", "yyyymm"]].values.tolist() == [[1, 200001], [1, 200002], [2, 200001]]
    assert "mthret" not in norm.columns  # returns never reach formation data

    rows.append(crsp_row(1, "2000-01", 10, mthcaldt=pd.Timestamp("2000-01-14")))
    with pytest.raises(DataValidationError, match="1 rows repeat"):
        crsp.normalize_monthly(pd.DataFrame(rows))


@pytest.mark.parametrize("field", ["tradingstatusflg", "mthretflg", "mthdelflg"])
def test_missing_required_field_is_named_not_guessed(field):
    frame = pd.DataFrame([crsp_row(1, "2000-01", 10)]).drop(columns=[field])
    with pytest.raises(DataValidationError, match=field):
        crsp.normalize_monthly(frame)


def test_crsp_predictors_follow_the_signed_osap_definitions():
    rows = [
        crsp_row(1, "2000-01", 2500.0, mthprc=25.0, mthret=0.04),
        crsp_row(2, "2000-01", 800.0, mthprc=-8.0, mthret=None),  # bid/ask midpoint; missing return -> 0
        crsp_row(3, "2000-01", 0.0, mthprc=0.0, mthret=-0.02),  # no valid price or cap
    ]
    out = crsp.crsp_predictors(pd.DataFrame(rows))
    assert out["Price"].iloc[:2].tolist() == pytest.approx([-np.log(25.0), -np.log(8.0)])
    assert out["Size"].iloc[0] == pytest.approx(-np.log(2500.0))  # MthCap in $ thousands, = |prc| x shrout
    assert out["STreversal"].tolist() == pytest.approx([-4.0, 0.0, 2.0])  # percent
    assert np.isnan(out["Price"].iloc[2]) and np.isnan(out["Size"].iloc[2])
