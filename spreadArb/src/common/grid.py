"""Per-product arrays from one session of the maker canonical 1 Hz grid."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .paths import CLOSE_SECOND, grid_path, mapping_path

GRID_COLUMNS = ["ValueCode", "QuoteCode", "seconds_from_open", "basis_mid_bp", "basis_sell_taker_bp",
                "basis_buy_taker_bp", "anchor_ewma_120s_bp", "analysis_eligible"]


@dataclass
class ProductDay:
    day: str
    vc: str
    qc: str
    expiry: str          # YYYYMMDD, from the pre-open mapping
    mid: np.ndarray      # basis_mid_bp per second, nan when absent
    buy: np.ndarray      # basis_buy_taker_bp per second (exit trigger series)
    anchor: np.ndarray   # anchor_ewma_120s_bp per second
    eligible: np.ndarray
    sell: np.ndarray | None = None   # basis_sell_taker_bp: lower bound of an S2 lock (FutBid vs SpotA1)


def load_day(day: str, products: list[str] | None = None) -> dict[str, ProductDay]:
    mapping = pl.read_parquet(mapping_path(day), columns=["ValueCode", "QuoteCode", "end_date"])
    expiry = {(r["ValueCode"], r["QuoteCode"]): r["end_date"].strftime("%Y%m%d")
              for r in mapping.iter_rows(named=True) if r["end_date"] is not None}
    frame = pl.scan_parquet(grid_path(day)).select(GRID_COLUMNS)
    if products:
        frame = frame.filter(pl.col("ValueCode").is_in(products))
    frame = frame.collect()
    result: dict[str, ProductDay] = {}
    for g in frame.partition_by("ValueCode", maintain_order=True):
        vc = g.item(0, "ValueCode")
        codes = g["QuoteCode"].unique().to_list()
        if len(codes) != 1:
            raise ValueError(f"{day} {vc}: grid mixes contracts {codes}")
        qc = codes[0]
        exp = expiry.get((vc, qc))
        if exp is None or exp < day:
            continue
        sec = g["seconds_from_open"].to_numpy()
        if sec.min() < 0 or sec.max() >= CLOSE_SECOND or len(np.unique(sec)) != len(sec):
            raise ValueError(f"{day} {vc}: invalid second index")

        def column(name: str) -> np.ndarray:
            arr = np.full(CLOSE_SECOND, np.nan)
            arr[sec] = g[name].cast(pl.Float64).fill_null(np.nan).to_numpy()
            return arr

        eligible = np.zeros(CLOSE_SECOND, dtype=bool)
        eligible[sec] = g["analysis_eligible"].fill_null(False).to_numpy()
        result[vc] = ProductDay(day, vc, qc, exp, column("basis_mid_bp"),
                                column("basis_buy_taker_bp"), column("anchor_ewma_120s_bp"), eligible,
                                column("basis_sell_taker_bp"))
    return result


def close_marks(day: str, products: list[str] | None = None) -> dict[str, tuple[int, int]]:
    """Last eligible second's spot and futures mid (TWD x 1e4) per product: the settlement marks
    used when a position is still open at the close of its expiry day."""
    cols = ["ValueCode", "seconds_from_open", "spot_bid", "spot_ask", "fut_bid", "fut_ask", "analysis_eligible"]
    frame = pl.scan_parquet(grid_path(day)).select(cols)
    if products:
        frame = frame.filter(pl.col("ValueCode").is_in(products))
    frame = (frame.filter(pl.col("analysis_eligible") & (pl.col("spot_bid") > 0) & (pl.col("fut_bid") > 0))
             .sort("seconds_from_open").group_by("ValueCode").last().collect())
    return {r["ValueCode"]: (int(round((r["spot_bid"] + r["spot_ask"]) / 2 * 10_000)),
                             int(round((r["fut_bid"] + r["fut_ask"]) / 2 * 10_000)))
            for r in frame.iter_rows(named=True)}
