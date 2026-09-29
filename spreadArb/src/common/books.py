"""Raw receive-time books and prints for one session.

Spot: SSD2 ``{D}_StockTick.parquet`` (float prices, naive-UTC microseconds, lots of 1,000 shares).
Futures: NAS ``stock_futures.parquet`` (integer prices scaled by DecimalLocator, tz-aware ns, lots).
Prices become int (TWD x 1e4), spot quantities become shares, futures quantities stay in lots.
Top-of-book merges the five displayed levels with BestBid/BestAsk the way maker ``BookSeries`` does.
"""
from __future__ import annotations

from dataclasses import dataclass
import sys

import numpy as np
import polars as pl

from .paths import CLOSE_SECOND, SECOND, open_ns

LEVELS = 5
INF = np.iinfo(np.int64).max // 4


def _maker_paths():
    """Resolve the pipeline-owned raw paths through maker's role-bound resolver."""
    from pathlib import Path
    hft = Path(__file__).resolve().parents[5]
    if str(hft) not in sys.path:
        sys.path.insert(0, str(hft))
    from src.research.futures_spot_spread.maker.src.common import paths as mp
    return mp


def raw_paths(day: str) -> tuple:
    mp = _maker_paths()
    spot = mp.resolve_input_file(mp.spot_tick_path(day), role="spot_raw")
    fut = mp.resolve_input_file(mp.futures_raw_path(day), role="future_raw")
    return spot, fut, mp.market_data_path(day)


def tick_i(p: int) -> int:
    return (100 if p < 100_000 else 500 if p < 500_000 else 1000 if p < 1_000_000
            else 5000 if p < 5_000_000 else 10_000 if p < 10_000_000 else 50_000)


def previous_tick(p: int) -> int:
    return p - tick_i(p - 1)


def next_tick(p: int) -> int:
    return p + tick_i(p)


def floor_tick(p: int) -> int:
    t = tick_i(p)
    return (p // t) * t


def ceil_tick(p: int) -> int:
    t = tick_i(max(p - 1, 1))
    return -((-p) // t) * t


@dataclass
class BookSeries:
    instrument: str
    ns: np.ndarray
    seq: np.ndarray
    formal: np.ndarray
    bid_px: np.ndarray    # (n, 5) int, 0 = absent
    bid_qty: np.ndarray
    ask_px: np.ndarray
    ask_qty: np.ndarray
    best_bid_px: np.ndarray
    best_bid_qty: np.ndarray
    best_ask_px: np.ndarray
    best_ask_qty: np.ndarray
    top_bid_px: np.ndarray = None   # merged with Best*, 0 when absent
    top_bid_qty: np.ndarray = None
    top_ask_px: np.ndarray = None   # INF when absent
    top_ask_qty: np.ndarray = None
    valid: np.ndarray = None

    def __post_init__(self):
        bpx = np.where(self.bid_qty > 0, self.bid_px, 0)
        l1 = bpx.max(axis=1)
        l1q = np.take_along_axis(self.bid_qty, bpx.argmax(axis=1)[:, None], axis=1)[:, 0]
        best_ok = (self.best_bid_px > 0) & (self.best_bid_qty > 0)
        better = best_ok & (self.best_bid_px > l1)
        same = best_ok & (self.best_bid_px == l1)
        self.top_bid_px = np.where(better, self.best_bid_px, l1)
        self.top_bid_qty = np.where(better, self.best_bid_qty, np.where(same, np.maximum(l1q, self.best_bid_qty), l1q))
        apx = np.where(self.ask_qty > 0, self.ask_px, INF)
        a1 = apx.min(axis=1)
        a1q = np.take_along_axis(self.ask_qty, apx.argmin(axis=1)[:, None], axis=1)[:, 0]
        best_ok = (self.best_ask_px > 0) & (self.best_ask_qty > 0)
        better = best_ok & (self.best_ask_px < a1)
        same = best_ok & (self.best_ask_px == a1)
        self.top_ask_px = np.where(better, self.best_ask_px, a1)
        self.top_ask_qty = np.where(better, self.best_ask_qty, np.where(same, np.maximum(a1q, self.best_ask_qty), a1q))
        self.valid = self.formal & (self.top_bid_px > 0) & (self.top_ask_px < INF) & (self.top_bid_px < self.top_ask_px)

    def index_at(self, ns: int) -> int:
        """Last book received at or before ns; -1 if none."""
        return int(np.searchsorted(self.ns, ns, side="right")) - 1

    def levels(self, i: int, side: str) -> list[tuple[int, int]]:
        """Displayed levels merged with Best*, best first, at most five."""
        px, qty = (self.bid_px[i], self.bid_qty[i]) if side == "bid" else (self.ask_px[i], self.ask_qty[i])
        d = {int(p): int(q) for p, q in zip(px, qty) if p > 0 and q > 0}
        bp, bq = ((self.best_bid_px[i], self.best_bid_qty[i]) if side == "bid"
                  else (self.best_ask_px[i], self.best_ask_qty[i]))
        bp, bq = int(bp), int(bq)
        if bp > 0 and bq > 0 and (not d or (bp >= max(d) if side == "bid" else bp <= min(d))):
            d[bp] = max(d.get(bp, 0), bq)
        return sorted(d.items(), reverse=side == "bid")[:LEVELS]

    def take(self, i: int, side: str, quantity: int, min_px: int = 1, max_px: int = INF) -> tuple[int, int] | None:
        """Sweep L1-L5 for the full quantity; (cash, levels swept) or None if depth is short."""
        left, cash, swept = quantity, 0, 0
        for px, q in self.levels(i, side):
            if not min_px <= px <= max_px:
                break
            used = min(left, q)
            cash += px * used
            left -= used
            swept += 1
            if left == 0:
                return cash, swept
        return None


@dataclass
class Prints:
    instrument: str
    ns: np.ndarray
    seq: np.ndarray
    price: np.ndarray
    qty: np.ndarray


FIELDS = ["RecvTime", "ChannelSeq", "QuoteCode", "TrialMatch", "FillPrice", "FillLots",
          "BestBidPrice", "BestAskPrice", "BestBidLots", "BestAskLots"] + \
         [f"{s}{k}{i}" for s in ("Bid", "Ask") for k in ("Price", "Lots") for i in range(1, LEVELS + 1)]


def _raw(path, codes: list[str], *, future: bool, start: int, end: int) -> pl.DataFrame:
    fields = FIELDS + (["DecimalLocator"] if future else [])
    source = pl.scan_parquet(path).filter(pl.col("QuoteCode").is_in(codes)).select(fields)
    factor = (10.0 ** (4 - pl.col("DecimalLocator"))) if future else pl.lit(10_000.0)
    prices = [c for c in FIELDS if "Price" in c]
    lots = [c for c in FIELDS if "Lots" in c]
    return (source.with_columns(
        pl.col("RecvTime").dt.epoch("ns").alias("ns"),
        pl.col("ChannelSeq").cast(pl.Int64).alias("seq"),
        (pl.col("TrialMatch") == 0).alias("formal"),
        *[(pl.col(c) * factor).round(0).fill_null(0).cast(pl.Int64).alias(c) for c in prices],
        *[(pl.col(c).fill_null(0).cast(pl.Int64) * (1 if future else 1000)).alias(c) for c in lots],
    ).filter((pl.col("ns") >= start - 1800 * SECOND) & (pl.col("ns") <= end))
        .sort(["QuoteCode", "ns", "seq"]).collect())


def _series(instrument: str, g: pl.DataFrame) -> BookSeries:
    def block(prefix: str, kind: str) -> np.ndarray:
        return g.select([f"{prefix}{kind}{i}" for i in range(1, LEVELS + 1)]).to_numpy().astype(np.int64, copy=False)
    return BookSeries(instrument, g["ns"].to_numpy().astype(np.int64), g["seq"].to_numpy().astype(np.int64),
                      g["formal"].to_numpy(), block("Bid", "Price"), block("Bid", "Lots"),
                      block("Ask", "Price"), block("Ask", "Lots"),
                      g["BestBidPrice"].to_numpy().astype(np.int64), g["BestBidLots"].to_numpy().astype(np.int64),
                      g["BestAskPrice"].to_numpy().astype(np.int64), g["BestAskLots"].to_numpy().astype(np.int64))


def load_session(day: str, spot_codes: list[str], fut_codes: list[str]):
    """Return (books, prints, spot_limits, inputs). Keys are 'S:<vc>' and 'F:<qc>'."""
    spot_path, fut_path, market_path = raw_paths(day)
    start, end = open_ns(day), open_ns(day) + CLOSE_SECOND * SECOND
    books, prints = {}, {}
    for future, path, codes in ((False, spot_path, spot_codes), (True, fut_path, fut_codes)):
        raw = _raw(path, codes, future=future, start=start, end=end)
        prefix = "F:" if future else "S:"
        trades = raw.filter(pl.col("formal") & (pl.col("FillLots") > 0) & (pl.col("FillPrice") > 0) & (pl.col("ns") >= start))
        book_rows = raw
        if future:
            # trade-only rows carry no depth; keep explicit empty/trial updates
            book_rows = raw.filter((pl.col("BidPrice1") > 0) | (pl.col("AskPrice1") > 0)
                                   | (pl.col("FillLots") <= 0) | ~pl.col("formal"))
        for g in book_rows.partition_by("QuoteCode", maintain_order=True):
            key = prefix + g.item(0, "QuoteCode")
            books[key] = _series(key, g)
        for g in trades.partition_by("QuoteCode", maintain_order=True):
            key = prefix + g.item(0, "QuoteCode")
            prints[key] = Prints(key, g["ns"].to_numpy().astype(np.int64), g["seq"].to_numpy().astype(np.int64),
                                 g["FillPrice"].to_numpy().astype(np.int64), g["FillLots"].to_numpy().astype(np.int64))
    limits = {}
    md = pl.read_parquet(market_path, columns=["quote_code", "limit_up_price", "limit_down_price", "opening_ref_price"])
    for r in md.iter_rows(named=True):
        if r["limit_up_price"] is not None and r["limit_down_price"] is not None:
            limits[r["quote_code"]] = (int(round(r["limit_down_price"] * 10_000)), int(round(r["limit_up_price"] * 10_000)))
    return books, prints, limits, [str(spot_path), str(fut_path), str(market_path)]
