"""Validated join of U_t with OSAP characteristics (synthetic data)."""

import numpy as np
import pandas as pd
import pytest

from disagreement.data.panel import join_universe_osap
from disagreement.data.validation import DataValidationError

T, T1 = 200001, 200002
SIGNALS = ["SigA", "SigB"]


def _universe():
    return pd.DataFrame(
        {
            "permno": [1, 2, 3],
            "yyyymm": [T, T, T],
            "exchange": ["NYSE", "AMEX", "NASDAQ"],
            "mktcap": [10.0, 20.0, 30.0],
        }
    )


def _osap(rows):
    return pd.DataFrame(rows, columns=["permno", "yyyymm", *SIGNALS])


def test_left_join_keeps_every_universe_row_exactly_once():
    osap = _osap(
        [
            [1, T, 0.1, 0.2],
            [3, T, np.nan, 0.4],
            [9, T, 1.0, 1.0],  # not in U_t
            [1, T1, 5.0, 5.0],  # next month: must not be attached to month t
        ]
    )
    panel = join_universe_osap(_universe(), osap, SIGNALS)
    assert list(panel.columns) == ["permno", "yyyymm", "exchange", "mktcap", "osap_matched", *SIGNALS]
    assert panel["permno"].tolist() == [1, 2, 3]
    assert panel["osap_matched"].tolist() == [True, False, True]
    assert panel.loc[panel["permno"] == 1, "SigA"].item() == 0.1
    assert panel.loc[panel["permno"] == 2, SIGNALS].isna().to_numpy().all()


def test_duplicate_osap_key_is_rejected():
    osap = _osap([[1, T, 0.1, 0.2], [1, T, 0.3, 0.4]])
    with pytest.raises(DataValidationError, match="OSAP before join"):
        join_universe_osap(_universe(), osap, SIGNALS)


def test_duplicate_universe_key_is_rejected():
    universe = pd.concat([_universe(), _universe().iloc[[0]]], ignore_index=True)
    with pytest.raises(DataValidationError, match="universe before join"):
        join_universe_osap(universe, _osap([[1, T, 0.1, 0.2]]), SIGNALS)
