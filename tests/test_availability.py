"""Predictor-availability rule of the frozen baseline (synthetic data)."""

import numpy as np
import pandas as pd
import pytest

from disagreement.data.availability import eligible_predictors
from disagreement.data.validation import DataValidationError

N_STOCKS = 10
RULE = {"min_stock_coverage": 0.60, "min_month_share": 0.95}


def _panel(months, observed):
    """U_t of N_STOCKS per month; observed[name](k) stocks have a value in the k-th month."""
    rows = []
    for k, month in enumerate(months):
        for i in range(N_STOCKS):
            row = {"permno": i + 1, "yyyymm": month, "exchange": "NYSE", "mktcap": 100.0}
            row.update({name: (1.0 if i < n_obs(k) else np.nan) for name, n_obs in observed.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def _months(start, n):
    return [int(m) for m in pd.period_range(str(start), periods=n, freq="M").strftime("%Y%m")]


def test_monthly_coverage_threshold_is_60_percent_of_u_t():
    panel = _panel(_months("2000-01", 1), {"At60": lambda k: 6, "Below": lambda k: 5})
    panel.loc[panel["permno"] == 1, "At60"] = np.inf  # not a usable value
    panel.loc[panel["permno"] == 7, "At60"] = 1.0
    assert eligible_predictors(panel, ["At60", "Below"], cutoff=200001, **RULE) == ["At60"]


def test_coverage_must_hold_in_95_percent_of_historical_months():
    months = _months("2000-01", 20)
    panel = _panel(
        months,
        {
            "In19of20": lambda k: 3 if k == 0 else 8,  # 95% of months
            "In18of20": lambda k: 3 if k < 2 else 8,  # 90% of months
        },
    )
    assert eligible_predictors(panel, ["In19of20", "In18of20"], cutoff=months[-1], **RULE) == ["In19of20"]


def test_months_after_the_cutoff_cannot_affect_eligibility():
    history = _panel(_months("2000-01", 20), {"Kept": lambda k: 8, "Dropped": lambda k: 2})
    future = _panel(_months("2001-09", 40), {"Kept": lambda k: 0, "Dropped": lambda k: 10})
    expected = eligible_predictors(history, ["Kept", "Dropped"], cutoff=200108, **RULE)
    assert expected == ["Kept"]
    combined = pd.concat([future, history], ignore_index=True)  # order does not matter either
    assert eligible_predictors(combined, ["Kept", "Dropped"], cutoff=200108, **RULE) == expected


def test_non_predictor_columns_are_never_candidates():
    panel = _panel(_months("2000-01", 12), {"Sig": lambda k: 9})
    panel["ret_next_month"] = 0.01  # fully observed, but not a candidate
    assert eligible_predictors(panel, ["Sig"], cutoff=200012, **RULE) == ["Sig"]
    for column in ("permno", "yyyymm", "exchange", "mktcap"):
        with pytest.raises(DataValidationError, match="not predictors"):
            eligible_predictors(panel, ["Sig", column], cutoff=200012, **RULE)
