"""Point-in-time metadata, raw receive-time books, and causal 1 Hz anchors."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from ..common.contracts import (_load_futures_basic_from_existing_loader, _normalise_basic_schema,
                                select_near_standard_contracts)
from ..common.paths import (MAKER_ROOT, futures_raw_path, spot_tick_path, resolve_input_file,
                            market_data_path, parse_date)
from .causal_lookup import CLOSE_SECOND, SECOND, open_ns
from .execution import Book, price_i
from .corporate_actions import known_actions
from .forecast_calendar import known_calendar

WF = MAKER_ROOT / "data/walkforward"
METADATA_ROOT = MAKER_ROOT / "data/ev_lookup_v19_metadata_20260908"
LEVELS = [f"{s}{k}{i}" for s in ("Bid", "Ask") for k in ("Price", "Lots") for i in range(1, 6)]


@dataclass(frozen=True)
class Contract:
    vc: str
    qc: str
    shares: int
    expiry: str
    spot_ref: int
    fut_ref: int


def tick_i(p: int) -> int:
    return (100 if p < 100_000 else 500 if p < 500_000 else 1000 if p < 1_000_000
            else 5000 if p < 5_000_000 else 10_000 if p < 10_000_000 else 50_000)


def previous_tick(p: int) -> int:
    # At a tier boundary, the preceding price uses the lower tier's tick.
    return p - tick_i(p - 1)


class BookSeries:
    def __init__(self, instrument: str, frame: pl.DataFrame):
        self.instrument = instrument
        self.ns = frame["ns"].to_numpy()
        self.seq = frame["seq"].to_numpy()
        self.formal = frame["formal"].to_numpy()
        self.prices = {s: frame.select([f"{s}Price{i}" for i in range(1, 6)]).to_numpy()
                       for s in ("Bid", "Ask")}
        self.lots = {s: frame.select([f"{s}Lots{i}" for i in range(1, 6)]).to_numpy()
                     for s in ("Bid", "Ask")}
        self.best = {s: frame.select(f"Best{s}Price", f"Best{s}Lots").to_numpy()
                     for s in ("Bid", "Ask")}
        self._last_index = -2
        self._last_now_ns: int | None = None
        self._last_book: Book | None = None

    def at(self, now_ns: int) -> Book | None:
        if now_ns == self._last_now_ns:
            return self._last_book
        i = int(np.searchsorted(self.ns, now_ns, side="right")) - 1
        if i < 0:
            return None
        self._last_now_ns = now_ns
        if i == self._last_index:
            return self._last_book
        levels = {}
        for side in ("Bid", "Ask"):
            d = {int(p): int(q) for p, q in zip(self.prices[side][i], self.lots[side][i])
                 if p > 0 and q > 0}
            p, q = map(int, self.best[side][i])
            # BestBid/Ask can improve the five-level book. Never sum duplicate
            # representations of the same visible quantity at one price.
            if p > 0 and q > 0 and (not d or
                    (p >= max(d) if side == "Bid" else p <= min(d))):
                d[p] = max(d.get(p, 0), q)
            levels[side] = tuple(sorted(d.items(), reverse=side == "Bid"))[:5]
        self._last_index = i
        self._last_book = Book(self.instrument, int(self.ns[i]), int(self.seq[i]),
                               levels["Bid"], levels["Ask"], bool(self.formal[i]))
        return self._last_book

    def next_ns(self, now_ns: int) -> int | None:
        i = int(np.searchsorted(self.ns, now_ns, side="right"))
        return int(self.ns[i]) if i < len(self.ns) else None


def _raw(path: Path, codes: list[str], *, future: bool, start: int, end: int) -> pl.DataFrame:
    fields = ["RecvTime", "ChannelSeq", "QuoteCode", "TrialMatch", "FillPrice", "FillLots",
              "BestBidPrice", "BestAskPrice", "BestBidLots", "BestAskLots"] + LEVELS
    if future:
        fields.append("DecimalLocator")
    source = pl.scan_parquet(path).filter(pl.col("QuoteCode").is_in(codes)).select(fields)
    factor = (10.0 ** (4 - pl.col("DecimalLocator"))) if future else pl.lit(10_000.0)
    prices = [c for c in fields if "Price" in c]
    lots = [c for c in fields if "Lots" in c]
    return (source.with_columns(
        pl.col("RecvTime").dt.epoch("ns").alias("ns"),
        pl.col("ChannelSeq").cast(pl.Int64).alias("seq"),
        (pl.col("TrialMatch") == 0).alias("formal"),
        *[(pl.col(c) * factor).round(0).fill_null(0).cast(pl.Int64).alias(c) for c in prices],
        *[(pl.col(c).fill_null(0).cast(pl.Int64) * (1 if future else 1000)).alias(c) for c in lots],
    ).filter((pl.col("ns") >= start - 1800 * SECOND) & (pl.col("ns") <= end))
        .sort(["QuoteCode", "ns", "seq"]).collect())


class MarketDay:
    def __init__(self, day: str, output: Path, products: list[str] | None = None,
                 carry: list[Contract] | None = None, *, data_outage: bool = False,
                 depth_events: bool = False):
        self.day, self.start = day, open_ns(day)
        self.end = self.start + CLOSE_SECOND * SECOND
        self.calendar = known_calendar(day)
        spot_metadata = (pl.scan_parquet(market_data_path(day)).select(
            pl.col("quote_code").alias("ValueCode"),
            pl.col("opening_ref_price").cast(pl.Float64).alias("spot_ref_price"),
            pl.col("allow_day_trade_mark").alias("day_trade_mark"),
            pl.col("limit_up_price").cast(pl.Float64),
            pl.col("limit_down_price").cast(pl.Float64)).collect())
        self.spot_limits = {r["ValueCode"]: (price_i(r["limit_down_price"]), price_i(r["limit_up_price"]))
                            for r in spot_metadata.iter_rows(named=True)
                            if r["limit_down_price"] is not None and r["limit_up_price"] is not None}
        self.inputs = [str(market_data_path(day))]
        actions = known_actions(day, METADATA_ROOT)
        self.corporate_entry_block = {r["vc"] for r in actions if day < r["effective_day"]}
        adjusted_today = {r["vc"] for r in actions if day == r["effective_day"]}
        if actions:
            self.inputs.append(str(METADATA_ROOT / "announcements/index.json"))
        basic = None

        def basic_info() -> pl.DataFrame:
            nonlocal basic
            if basic is not None:
                return basic
            cached = METADATA_ROOT / f"{day}_futures_basic.parquet"
            if not cached.exists():
                cached = output / "metadata" / f"{day}_futures_basic.parquet"
                if not cached.exists():
                    value = _load_futures_basic_from_existing_loader(day)
                    cached.parent.mkdir(exist_ok=True, parents=True)
                    value.write_parquet(cached)
            self.inputs.append(str(cached))
            basic = _normalise_basic_schema(pl.read_parquet(cached))
            return basic

        path = WF / f"daily/Date={day}/mapping.parquet"
        if path.exists():
            mapping = pl.read_parquet(path)
            self.inputs.append(str(path))
        else:
            mapping = select_near_standard_contracts(basic_info(), parse_date(day).date()).join(
                spot_metadata.filter(pl.col("day_trade_mark").is_in(["X", "Y"])),
                on="ValueCode", how="inner")
        if products:
            mapping = mapping.filter(pl.col("ValueCode").is_in(products))
        self.contracts = {}
        for r in mapping.select("ValueCode", "QuoteCode", "contract_size", "end_date",
                                "spot_ref_price", "fut_ref_price").iter_rows(named=True):
            vc = r["ValueCode"]
            if vc in self.contracts:
                raise ValueError("mapping is not one-to-one")
            expiry = r["end_date"].strftime("%Y%m%d")
            if expiry < day or abs(r["contract_size"] - 2000.0) > 1e-6:
                raise ValueError("invalid standard near-contract metadata")
            self.contracts[vc] = Contract(vc, r["QuoteCode"], round(r["contract_size"]), expiry,
                                          price_i(r["spot_ref_price"]), price_i(r["fut_ref_price"]))
        self.exact = {c.qc: c for c in self.contracts.values()}
        self.future_to_vc = {c.qc: c.vc for c in self.contracts.values()}
        self.missing_carry: set[str] = set()
        # A name losing day-trade eligibility cannot receive new quotes but its
        # existing inventory still has executable exit books. Bind by contract.
        for prior in carry or []:
            if prior.vc in adjusted_today:
                self.missing_carry.add(prior.qc)
            if prior.expiry < day or prior.qc in self.exact:
                continue
            exact = basic_info().filter((pl.col("QuoteCode") == prior.qc) &
                                        (pl.col("ValueCode") == prior.vc)).join(
                                            spot_metadata, on="ValueCode", how="inner")
            if exact.height != 1:
                self.missing_carry.add(prior.qc)
                self.exact[prior.qc] = prior
                continue
            r = exact.row(0, named=True)
            if round(r["contract_size"]) != prior.shares:
                raise ValueError(f"{day}: contract adjustment needs explicit position conversion: {prior.qc}")
            self.exact[prior.qc] = Contract(prior.vc, prior.qc, prior.shares,
                                           r["end_date"].strftime("%Y%m%d"),
                                           price_i(r["spot_ref_price"]), price_i(r["fut_ref_price"]))
        self.series: dict[str, BookSeries] = {}
        self._pair_cache: dict[tuple,tuple] = {}
        self.spot_top_events = pl.DataFrame(schema={"ns": pl.Int64, "instrument": pl.String})
        # Spot best-ask changes: the events that re-price a working S2 quote.
        self.spot_ask_events = pl.DataFrame(schema={"ns": pl.Int64, "instrument": pl.String})
        self.book_events = self.spot_ask_events.clone()
        if data_outage:
            self.inputs.append("DATA_OUTAGE: no simulated execution; inventory retained")
            self.trades = pl.DataFrame(schema={"ns": pl.Int64, "seq": pl.Int64,
                                               "instrument": pl.String, "FillPrice": pl.Int64,
                                               "FillLots": pl.Int64})
            self.anchors, self.signals = {}, {}
            return
        trade_frames, event_frames = [], []
        for future in (False, True):
            raw_path = futures_raw_path(day) if future else spot_tick_path(day)
            raw_path = resolve_input_file(raw_path, role="future_raw" if future else "spot_raw")
            self.inputs.append(str(raw_path))
            codes = sorted({c.qc if future else c.vc for c in self.exact.values()})
            raw = _raw(raw_path, codes, future=future, start=self.start, end=self.end)
            if not future:
                self.spot_top_events = (raw.with_columns(
                    pl.max_horizontal("BidPrice1", "BestBidPrice").alias("top_bid"))
                    .filter((pl.col("top_bid").diff().over("QuoteCode").fill_null(1) != 0)
                            & (pl.col("ns") >= self.start))
                    .select("ns", (pl.lit("S:") + pl.col("QuoteCode")).alias("instrument"))
                    .unique().sort(["ns", "instrument"]))
            prefix = "F:" if future else "S:"
            trade_frames.append(raw.filter(pl.col("formal") & (pl.col("FillLots") > 0)
                                           & (pl.col("FillPrice") > 0) & (pl.col("ns") >= self.start))
                                .select("ns", "seq", (pl.lit(prefix) + pl.col("QuoteCode")).alias("instrument"),
                                        "FillPrice", "FillLots"))
            # Futures trade-only rows omit the depth; retain the last actual
            # book. Explicit empty quote updates and trial transitions remain.
            book_rows = raw
            if future:
                book_rows = raw.filter((pl.col("BidPrice1") > 0) | (pl.col("AskPrice1") > 0)
                                       | (pl.col("FillLots") <= 0) | ~pl.col("formal"))
            # Wake on effective top prices and market validity, including
            # empty/trial updates and a vanished level exposing a deeper quote.
            tops = []
            for side in ("Bid", "Ask"):
                none = 0 if side == "Bid" else 10**15
                values = [pl.when((pl.col(px) > 0) & (pl.col(qty) > 0)).then(pl.col(px)).otherwise(none)
                          for px, qty in [(f"{side}Price{i}", f"{side}Lots{i}") for i in range(1, 6)]
                          + [(f"Best{side}Price", f"Best{side}Lots")]]
                fn = pl.max_horizontal if side == "Bid" else pl.min_horizontal
                tops.append(fn(values).alias("top_"+side))
            wake_fields = ["top_Bid", "top_Ask", "formal"]
            if depth_events and not future:
                wake_fields += [f"Ask{field}{i}" for i in range(1, 6) for field in ("Price", "Lots")]
                wake_fields += ["BestAskPrice", "BestAskLots"]
            events = (book_rows.with_columns(tops)
                .filter((pl.any_horizontal([
                    pl.col(k).ne(pl.col(k).shift(1).over("QuoteCode")).fill_null(True)
                    for k in wake_fields])) & (pl.col("ns") >= self.start))
                .select("ns", (pl.lit(prefix)+pl.col("QuoteCode")).alias("instrument"))
                .unique().sort(["ns", "instrument"]))
            event_frames.append(events)
            if not future:
                self.spot_ask_events = events
            for g in book_rows.partition_by("QuoteCode", maintain_order=True):
                key = prefix + g.item(0, "QuoteCode")
                self.series[key] = BookSeries(key, g)
        self.trades = pl.concat(trade_frames).sort(["ns", "instrument", "seq"])
        self.book_events = pl.concat(event_frames).sort(["ns", "instrument"])
        duplicates = self.trades.group_by("instrument", "ns", "seq").len().filter(pl.col("len") > 1)
        if duplicates.height:
            raise ValueError(f"{day}: duplicate raw trade cursors")
        self.anchors = self._anchors()
        self.signals = self._signals()

    def _signals(self) -> dict[str, dict[str, np.ndarray]]:
        times = self.start + np.arange(CLOSE_SECOND + 1, dtype=np.int64) * SECOND
        result = {}
        for vc, c in self.contracts.items():
            columns = {}
            valid = np.ones(len(times), dtype=bool)
            for prefix, instrument, ref in (("s", "S:" + vc, c.spot_ref),
                                             ("f", "F:" + c.qc, c.fut_ref)):
                b = self.series.get(instrument)
                if b is None or len(b.ns) == 0:
                    valid[:] = False
                    columns[prefix + "b"] = np.zeros(len(times))
                    columns[prefix + "a"] = np.zeros(len(times))
                    continue
                ix = np.searchsorted(b.ns, times, side="right") - 1
                valid &= ix >= 0
                ix = np.maximum(ix, 0)
                valid &= b.formal[ix]
                for side, suffix in (("Bid", "b"), ("Ask", "a")):
                    px = b.prices[side][ix, 0].copy()
                    qty = b.lots[side][ix, 0]
                    best, best_qty = b.best[side][ix, 0], b.best[side][ix, 1]
                    improved = ((best > px) if side == "Bid" else ((best < px) | (px <= 0)))
                    improved &= (best > 0) & (best_qty > 0)
                    px[improved] = best[improved]
                    valid &= (qty > 0) | improved
                    columns[prefix + suffix] = px
                bid, ask = columns[prefix + "b"], columns[prefix + "a"]
                valid &= (ref * .91 < bid) & (bid < ask) & (ask < ref * 1.08)
            an = self.anchors.get(vc, np.full(len(times), np.nan))
            fa, fb, sa = columns["fa"], columns["fb"], columns["sa"]
            tick = np.select([fa - 1 < 100_000, fa - 1 < 500_000, fa - 1 < 1_000_000,
                              fa - 1 < 5_000_000, fa - 1 < 10_000_000],
                             [100, 500, 1000, 5000, 10_000], default=50_000)
            desired = fa - tick
            ab = (desired / np.maximum(sa, 1) - 1) * 10_000
            valid &= np.isfinite(an)
            ok = valid & (desired > fb) & (ab > 0) & (ab - an >= 25.0)
            ok[:300] = False
            ok[14_000:] = False
            result[vc] = dict(columns, an=an, valid=valid, desired=desired, ab=ab, ok=ok)
        return result

    def book(self, instrument: str, ns: int) -> Book | None:
        series = self.series.get(instrument)
        return series.at(ns) if series else None

    def pair(self, c: Contract, ns: int, *, signal_buffer: bool = True) -> tuple[Book, Book] | None:
        key=(c.qc,c.spot_ref,c.fut_ref,signal_buffer)
        previous=self._pair_cache.get(key)
        if previous is not None and previous[0] == ns:
            return previous[1]
        result=self._pair_uncached(c,ns,signal_buffer=signal_buffer)
        self._pair_cache[key]=(ns,result)
        return result

    def _pair_uncached(self, c: Contract, ns: int, *, signal_buffer: bool) -> tuple[Book, Book] | None:
        spot, fut = self.book("S:" + c.vc, ns), self.book("F:" + c.qc, ns)
        if not spot or not fut or not spot.valid() or not fut.valid():
            return None
        if signal_buffer:
            for b, ref in ((spot, c.spot_ref), (fut, c.fut_ref)):
                if not ref * 0.91 < b.bids[0][0] < b.asks[0][0] < ref * 1.08:
                    return None
        return spot, fut

    def _anchors(self) -> dict[str, np.ndarray]:
        path = WF / f"daily/Date={self.day}/causal_fair.parquet"
        result = {}
        if path.exists():
            self.inputs.append(str(path))
            frame = (pl.scan_parquet(path).filter(pl.col("ValueCode").is_in(list(self.contracts)))
                     .select("ValueCode", "QuoteCode", "seconds_from_open", "timestamp",
                             "spot_recv_time", "fut_recv_time", "anchor_ewma_120s_bp").collect())
            if frame.filter((pl.col("spot_recv_time") > pl.col("timestamp")) |
                            (pl.col("fut_recv_time") > pl.col("timestamp"))).height:
                raise ValueError("canonical grid contains future book timestamps")
            for g in frame.partition_by("ValueCode"):
                vc = g.item(0, "ValueCode")
                g = g.filter(pl.col("seconds_from_open") <= CLOSE_SECOND).sort("seconds_from_open")
                if g["QuoteCode"].unique().to_list() != [self.contracts[vc].qc]:
                    raise ValueError("anchor and execution contract differ")
                if g["seconds_from_open"].to_list() != list(range(g.height)) or g.height < CLOSE_SECOND:
                    raise ValueError("incomplete canonical grid")
                values = np.full(CLOSE_SECOND + 1, np.nan)
                values[:g.height] = g["anchor_ewma_120s_bp"].to_numpy()
                result[vc] = values
            return result
        # Extension uses exactly <= second-boundary receive times. No
        # group-by-second .last(), no latest-day universe, no full-day volume.
        alpha = 1 - 0.5 ** (1 / 120)
        for vc, c in self.contracts.items():
            values = np.full(CLOSE_SECOND + 1, np.nan)
            previous = None
            for sec in range(CLOSE_SECOND + 1):
                pair = self.pair(c, self.start + sec * SECOND)
                if pair is None:
                    continue
                s, f = pair
                mid = ((f.bids[0][0] + f.asks[0][0]) /
                       (s.bids[0][0] + s.asks[0][0]) - 1) * 10_000
                previous = mid if previous is None else alpha * mid + (1 - alpha) * previous
                values[sec] = previous
            result[vc] = values
        return result
