"""Filesystem paths shared by maker research modules."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


MAKER_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_ROOT = MAKER_ROOT.parent


def find_hft_root() -> Path:
    """Locate the enclosing HFT project without relying on the current directory."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").exists() and (candidate / "data").is_dir():
            return candidate
    raise RuntimeError("could not locate the HFT project root")


HFT_ROOT = find_hft_root()
HFT_DATA_ROOT = HFT_ROOT / "data"
DEFAULT_OUTPUT_ROOT = MAKER_ROOT / "data" / "fair_mid"


def parse_date(date: str) -> datetime:
    return datetime.strptime(date, "%Y%m%d")


def spot_tick_path(date: str) -> Path:
    return HFT_DATA_ROOT / "tickData" / f"{date}_StockTick.parquet"


def market_data_path(date: str) -> Path:
    return HFT_DATA_ROOT / "marketData" / f"{date}_marketData.parquet"


def futures_raw_path(date: str) -> Path:
    parsed = parse_date(date)
    return (
        Path("/mnt/NAS/Parquet/Ticks")
        / parsed.strftime("%Y")
        / parsed.strftime("%m")
        / parsed.strftime("%d")
        / "stock_futures.parquet"
    )
