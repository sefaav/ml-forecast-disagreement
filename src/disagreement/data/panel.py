"""Validated join of the formation-time universe with OSAP characteristics.

The join is a left join from U_t on (permno, yyyymm): every universe member
keeps exactly one row, whether or not OSAP has characteristics for it, and no
future return is involved. Duplicate keys on either side are rejected, so the
join can never multiply rows.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from .validation import KEY, DataValidationError, require_unique_keys


def join_universe_osap(
    universe: pd.DataFrame, osap: pd.DataFrame, signals: Sequence[str]
) -> pd.DataFrame:
    """Universe rows with their same-month OSAP characteristics attached."""
    require_unique_keys(universe, "universe before join")
    require_unique_keys(osap, "OSAP before join")
    right = osap.loc[:, [*KEY, *signals]]
    merged = universe.merge(
        right, on=list(KEY), how="left", validate="one_to_one", indicator="_merge"
    )
    if len(merged) != len(universe):
        raise DataValidationError("join changed the number of universe rows.")
    matched = merged.pop("_merge").eq("both").to_numpy(dtype=bool)
    merged.insert(len(universe.columns), "osap_matched", matched)
    return merged
