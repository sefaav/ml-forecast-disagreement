"""Chronological walk-forward calendar of the frozen baseline (formation months, yyyymm).

One episode per out-of-sample (OOS) calendar year Y, retrained annually:

- training: sample start through December Y - 1 - validation_years (expanding);
- validation: the ``validation_years`` calendar years before Y (rolling);
- OOS test: January through December of Y.

Predictor eligibility uses the training block only (cutoff: end of
training). The pre-OOS history is training followed by validation, through
December Y-1. Labels of formation months through December Y-1 are known at
the January-Y formation date: the December Y-1 target is the January-Y
return, whereas the January-Y forecast concerns the February-Y return. The
calendar is derived from the configuration alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from disagreement.data.config import DEFAULT_CONFIG, load_data_config


@dataclass(frozen=True)
class Episode:
    oos_year: int
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    test_start: int
    test_end: int

    @property
    def history_start(self) -> int:
        """First month of the pre-OOS history (training followed by validation)."""
        return self.train_start

    @property
    def history_end(self) -> int:
        """Last month of the pre-OOS history: December of the year before the OOS year."""
        return self.validation_end

    @property
    def eligibility_cutoff(self) -> int:
        """Training cutoff passed to the predictor-availability rule: the end of training."""
        return self.train_end


def month_range(start: int, end: int) -> list[int]:
    """All yyyymm months from start to end, inclusive."""
    months, month = [], start
    while month <= end:
        months.append(month)
        month = month + 89 if month % 100 == 12 else month + 1
    return months


def annual_episodes(sample_start: int, sample_end: int, first_oos_year: int, validation_years: int) -> list[Episode]:
    """One episode per OOS year from ``first_oos_year`` through the sample's last year."""
    if sample_start % 100 != 1 or sample_end % 100 != 12:
        raise ValueError("the sample must start in January and end in December")
    if validation_years < 1 or first_oos_year - validation_years - 1 < sample_start // 100:
        raise ValueError("the first episode needs at least one training year before validation")
    if first_oos_year > sample_end // 100:
        raise ValueError("the first OOS year lies after the sample")
    return [
        Episode(
            oos_year=year,
            train_start=sample_start,
            train_end=(year - validation_years - 1) * 100 + 12,
            validation_start=(year - validation_years) * 100 + 1,
            validation_end=(year - 1) * 100 + 12,
            test_start=year * 100 + 1,
            test_end=year * 100 + 12,
        )
        for year in range(first_oos_year, sample_end // 100 + 1)
    ]


def baseline_episodes(config_path: Path = DEFAULT_CONFIG) -> list[Episode]:
    """The baseline calendar from configs/baseline.yaml (data.sample and walk_forward)."""
    cfg = load_data_config(config_path)
    with open(config_path, encoding="utf-8") as fh:
        walk_forward = yaml.safe_load(fh)["walk_forward"]
    return annual_episodes(
        cfg.sample_start, cfg.sample_end, int(walk_forward["first_oos_year"]), int(walk_forward["validation_years"])
    )
