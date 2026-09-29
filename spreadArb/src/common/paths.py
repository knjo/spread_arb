"""Filesystem roots, time constants and the maker canonical 1 Hz grid inventory.

spreadArb reads the maker walk-forward grid as an input and writes only under
``spreadArb/data`` (not in git).
"""
from __future__ import annotations

from pathlib import Path

SPREADARB_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_ROOT = SPREADARB_ROOT.parent
MAKER_ROOT = RESEARCH_ROOT / "maker"
WALKFORWARD = MAKER_ROOT / "data" / "walkforward"
GRID_DAILY = WALKFORWARD / "daily"
DATA_ROOT = SPREADARB_ROOT / "data"
QCACHE = DATA_ROOT / "qcache"

# Session clock: seconds from 09:00 open.
CLOSE_SECOND = 15_600           # 13:20, last grid second is 15_599
QUOTE_START_SECOND = 300        # 09:05, first entry/exit quote
QUOTE_END_SECOND = 14_000       # 12:53:20, last new entry quote
MAKER_WITHDRAW_SECOND = 15_480  # 13:18, all maker orders withdrawn
SECONDS_PER_DAY = 86_400
SECOND = 1_000_000_000          # ns
OPEN_UTC = "01:00:00"           # 09:00 Taipei


def open_ns(day: str) -> int:
    from datetime import datetime, timezone
    return int(datetime.strptime(f"{day} {OPEN_UTC}", "%Y%m%d %H:%M:%S")
               .replace(tzinfo=timezone.utc).timestamp()) * SECOND


def points_path(day: str, name: str):
    return DATA_ROOT / "points" / f"Date={day}" / f"{name}.parquet"


def grid_path(day: str) -> Path:
    return GRID_DAILY / f"Date={day}" / "causal_fair.parquet"


def mapping_path(day: str) -> Path:
    return GRID_DAILY / f"Date={day}" / "mapping.parquet"


def grid_days() -> list[str]:
    """Sessions with a canonical grid, ascending YYYYMMDD."""
    return sorted(p.name[5:] for p in GRID_DAILY.glob("Date=*")
                  if (p / "causal_fair.parquet").exists() and (p / "mapping.parquet").exists())


def hist_path(day: str) -> Path:
    return QCACHE / "hist" / f"Date={day}.parquet"


def facts_path(day: str) -> Path:
    return QCACHE / "facts" / f"Date={day}.parquet"
