"""Local data layout and the data parameters read from configs/baseline.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "baseline.yaml"
DEFAULT_DATA_DIR = REPO_ROOT / "data"


@dataclass(frozen=True)
class DataPaths:
    """Folders under data/, all ignored by Git."""

    root: Path

    def crsp(self, table: str) -> Path:
        return self.root / "raw" / "crsp" / table

    @property
    def osap(self) -> Path:
        return self.root / "raw" / "osap"

    @property
    def staging(self) -> Path:
        return self.root / "interim" / "osap_matched"

    @property
    def universe(self) -> Path:
        return self.root / "processed" / "universe"

    @property
    def panel(self) -> Path:
        return self.root / "processed" / "panel"

    @property
    def target(self) -> Path:
        return self.root / "processed" / "target"

    @property
    def diagnostics(self) -> Path:
        return self.root / "diagnostics"


@dataclass(frozen=True)
class DataConfig:
    crsp_library: str
    crsp_table: str
    osap_release: str
    osap_file: str
    osap_sha256: str
    osap_predictors: int
    exchanges: tuple[str, ...]
    breakpoint_exchange: str
    breakpoint_quantile: float
    sample_start: int
    sample_end: int
    min_stock_coverage: float
    min_month_share: float


def load_data_config(path: Path = DEFAULT_CONFIG) -> DataConfig:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)["data"]
    crsp, osap, universe = data["sources"]["crsp"], data["sources"]["osap"], data["universe"]
    sample, predictors = data["sample"], data["predictors"]
    cfg = DataConfig(
        crsp_library=crsp["library"],
        crsp_table=crsp["monthly_table"],
        osap_release=str(osap["release"]),
        osap_file=osap["file"],
        osap_sha256=str(osap["sha256"]).lower(),
        osap_predictors=int(osap["predictors"]),
        exchanges=tuple(universe["exchanges"]),
        breakpoint_exchange=universe["breakpoint_exchange"],
        breakpoint_quantile=float(universe["breakpoint_quantile"]),
        sample_start=int(sample["start"]),
        sample_end=int(sample["end"]),
        min_stock_coverage=float(predictors["min_stock_coverage"]),
        min_month_share=float(predictors["min_month_share"]),
    )
    if not set(cfg.exchanges) <= {"NYSE", "AMEX", "NASDAQ"} or cfg.breakpoint_exchange not in cfg.exchanges:
        raise ValueError("exchanges must be NYSE/AMEX/NASDAQ and include the breakpoint exchange")
    if not 0.0 < cfg.breakpoint_quantile < 1.0:
        raise ValueError("breakpoint_quantile must lie strictly between 0 and 1")
    months = (cfg.sample_start % 100, cfg.sample_end % 100)
    if not all(1 <= m <= 12 for m in months) or cfg.sample_start > cfg.sample_end:
        raise ValueError("sample start and end must be yyyymm months, start <= end")
    if not (0.0 < cfg.min_stock_coverage <= 1.0 and 0.0 < cfg.min_month_share <= 1.0):
        raise ValueError("predictor-availability thresholds must lie in (0, 1]")
    return cfg
