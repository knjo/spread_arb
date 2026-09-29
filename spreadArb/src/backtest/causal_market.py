"""Causal quote opportunities and raw execution data; never reads point fill labels.

Times are receive-time nanoseconds. Equal-timestamp market messages form one
snapshot; prints at that time precede our cancels, hedges and new placements.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import polars as pl

from ..common.books import INF, load_session, tick_i, previous_tick
from ..common.grid import load_day
from ..common.paths import (CLOSE_SECOND, SECOND, QUOTE_START_SECOND, QUOTE_END_SECOND,
                            MAKER_WITHDRAW_SECOND, open_ns, mapping_path)
from ..ev import qlevel
from .gates import GateSpec, leg_series
from ..points.s2 import tick_vec
from maker.src.ev_lookup_cost.execution import Book
from .contracts import refresh_carry


@dataclass(frozen=True)
class Contract:
    vc: str
    qc: str
    shares: int
    expiry: str
    spot_ref: int
    fut_ref: int


def index(ns, t):
    return int(np.searchsorted(ns, t, side="right")) - 1


class Timeline:
    def __init__(self, market, c, grid, scale, cfg):
        self.c, self.withdraw, self.start = c, market.withdraw, market.start
        s, f = market.books["S:" + c.vc], market.books["F:" + c.qc]
        self.s, self.f = s, f
        seconds = market.start + np.arange(QUOTE_START_SECOND, CLOSE_SECOND, dtype=np.int64) * SECOND
        parts = [seconds, s.ns, f.ns]
        for instrument in ("S:" + c.vc, "F:" + c.qc):
            if instrument in market.prints:
                parts.append(market.prints[instrument].ns)
        ns = np.unique(np.concatenate(parts))
        self.ns = ns[(ns >= seconds[0]) & (ns < market.end)]
        ns = self.ns
        self.sec = ((ns - market.start) // SECOND).astype(np.int32)
        si, fi = np.searchsorted(s.ns, ns, side="right") - 1, np.searchsorted(f.ns, ns, side="right") - 1
        self.si, self.fi = np.maximum(si, 0), np.maximum(fi, 0)
        si0, fi0 = self.si, self.fi
        self.sb, self.sa = s.top_bid_px[si0], s.top_ask_px[si0]
        self.fb, self.fa = f.top_bid_px[fi0], f.top_ask_px[fi0]
        self.saq = s.top_ask_qty[si0]
        self.entry_enabled = grid is not None
        # Held inventories use their frozen absolute exit target, not a new
        # entry anchor. An absent current entry grid must not disable exits.
        self.anchor = grid.anchor[self.sec] if grid is not None else np.zeros(len(ns))
        self.resid = grid.mid[self.sec] - self.anchor if grid is not None else np.zeros(len(ns))
        self.scale, self.scale_raw = scale.get("scale"), scale.get("scale_raw")
        self.valid = ((si >= 0) & (fi >= 0) & s.valid[si0] & f.valid[fi0]
                      & np.isfinite(self.anchor) & (self.sb > c.spot_ref * .91)
                      & (self.sa < c.spot_ref * 1.08) & (self.fb > c.fut_ref * .91)
                      & (self.fa < c.fut_ref * 1.08))
        self.gates = {}
        for leg, (gn, good) in leg_series(s, f, GateSpec()).items():
            gi = np.searchsorted(gn, ns, side="right") - 1
            self.gates[leg] = self.valid & (gi >= 0) & good[np.maximum(gi, 0)] if cfg.gates_tag else self.valid.copy()
        if cfg.gates_tag not in (None, GateSpec().tag):
            raise ValueError("causal replay only supports the documented GateSpec tag")
        self.gates["S2_sell"] &= self.saq >= cfg.depth_mult * c.shares
        self.prices = {"S1": self.sb, "S2": self.fa - tick_vec(np.maximum(self.fa - 1, 1)),
                       "E1": self.sa, "E2": self.fb + tick_vec(np.maximum(self.fb, 1))}
        self.ab = {"S1": (self.fb / np.maximum(self.sb, 1) - 1) * 10000,
                   "S2": (self.prices["S2"] / np.maximum(self.sa, 1) - 1) * 10000,
                   "E1": (self.fa / np.maximum(self.sa, 1) - 1) * 10000,
                   "E2": (self.prices["E2"] / np.maximum(self.sb, 1) - 1) * 10000}
        self.quotable = {leg: good.copy() for leg, good in self.gates.items()}
        self.quotable["S2_sell"] &= self.prices["S2"] > self.fb
        self.quotable["S2_buy"] &= self.prices["E2"] < self.fa
        self.candidates, self.entry_eligible = {}, {}
        for stream, leg in (("S1", "S1_buy"), ("S2", "S2_sell")):
            ab = self.ab[stream]
            ok = (self.quotable[leg] & (self.sec < QUOTE_END_SECOND) & (ab > 0)
                  & (self.anchor >= cfg.cost.min_anchor_bp)
                  & ((ab - self.anchor >= cfg.residual_min_bp) | (ab >= 50.0)))
            if self.scale_raw is not None and self.scale_raw > 60.0:
                ok[:] = False
            bucket = np.searchsorted([0, 5, 10, 15, 20, 25, 30], ab - self.anchor)
            second = (ns - market.start) % SECOND == 0
            changed = np.r_[True, self.prices[stream][1:] != self.prices[stream][:-1]]
            changed |= second & np.r_[True, bucket[1:] != bucket[:-1]]
            changed |= np.r_[True, ok[1:] & ~ok[:-1]]
            if stream == "S1":
                ahead = s.top_bid_qty[si0]
                aq = np.floor(np.log(np.maximum(ahead / 1000, 1)) / np.log(1.10)).astype(int)
                changed |= np.r_[True, aq[1:] != aq[:-1]]
            else:
                for value in (self.sa, self.saq, self.fb, self.fa):
                    changed |= np.r_[True, value[1:] != value[:-1]]
            pr = market.prints.get(("S:" + c.vc) if stream == "S1" else ("F:" + c.qc))
            if pr is not None:
                pi = np.searchsorted(ns, pr.ns)
                mask = pi < len(ns)
                pi, px = pi[mask], pr.price[mask]
                match = px <= self.prices[stream][pi] if stream == "S1" else px >= self.prices[stream][pi]
                changed[pi[match]] = True
            self.entry_eligible[stream] = ok if self.entry_enabled else np.zeros(len(ns), dtype=bool)
            self.candidates[stream] = np.flatnonzero(ok & changed) if self.entry_enabled else np.array([], dtype=np.int64)

    def at(self, now):
        i = index(self.ns, now)
        return max(i, 0)

    def row(self, stream, i):
        if not self.entry_enabled:
            raise ValueError("held-contract exit timeline cannot create new entries")
        c = self.c
        px, ab, an = int(self.prices[stream][i]), float(self.ab[stream][i]), float(self.anchor[i])
        hedge_px = int(self.fb[i] if stream == "S1" else self.sa[i])
        scale = self.scale or 1.0
        return dict(vc=c.vc, qc=c.qc, expiry=c.expiry, stream=stream, quote_ns=int(self.ns[i]),
                    quote_second=int(self.sec[i]), price=px, level=0, quote_ab=ab, anchor=an,
                    scale=self.scale, scale_raw=self.scale_raw, resid_mid_bp=float(self.resid[i]),
                    e_norm=float(self.resid[i]) / scale, spot_b1=int(self.sb[i]), spot_a1=int(self.sa[i]),
                    tick_bp_hedge=tick_i(hedge_px) / hedge_px * 10000,
                    execution_floor_bp=tick_i(hedge_px) / hedge_px * 5000 if stream == "S2" else 0.0,
                    notional_twd=(px if stream == "S1" else int(self.sa[i])) * c.shares / 10000)

    def entry_row_at(self, stream, ns):
        """Fresh observable quote even when no ordinary candidate changed now."""
        i = index(self.ns, ns)
        if (not self.entry_enabled or i < 0 or ns >= self.withdraw
                or not self.entry_eligible[stream][i]):
            return None
        row = self.row(stream, i)
        # A timer may fall between book messages; prices are as-of the timer,
        # while order transmission and EV time start at the actual new decision.
        second = (int(ns) - self.start) // SECOND
        if second >= QUOTE_END_SECOND:
            return None
        row.update(quote_ns=int(ns), quote_second=second)
        return row

    def first_exit(self, route, target, now):
        j = int(np.searchsorted(self.ns, now, side="left"))
        leg = "S1_sell" if route == "E1" else "S2_buy"
        ok = self.quotable[leg][j:] & (self.ab[route][j:] <= target)
        good = np.flatnonzero(ok)
        if not good.size:
            return None
        i = j + int(good[0])
        if self.ns[i] >= self.withdraw:
            return None
        return int(self.ns[i])

    def guard(self, route, price, target, now, cfg):
        """Schedule the first market trigger; callers cannot act until its event time."""
        i = max(index(self.ns, now), 0)
        leg = {"S1": "S1_buy", "S2": "S2_sell", "E1": "S1_sell", "E2": "S2_buy"}[route]
        valid = self.gates[leg][i:]
        # The inside-price test belongs to the submitted limit, not a later re-peg.
        if route == "S1":
            ab = (self.fb[i:] / price - 1) * 10000
        elif route == "S2":
            ab = (price / np.maximum(self.sa[i:], 1) - 1) * 10000
        elif route == "E1":
            ab = (self.fa[i:] / price - 1) * 10000
        else:
            ab = (price / np.maximum(self.sb[i:], 1) - 1) * 10000
        bad = ~valid
        if route in ("S1", "S2"):
            bad |= (ab <= 0) | (ab - self.anchor[i:] < cfg.floor_bp)
        else:
            bad |= ab > target + cfg.exit_tol_bp
        hit = np.flatnonzero(bad)
        trigger = max(now, int(self.ns[i + hit[0]])) if hit.size else self.withdraw
        return min(trigger, self.withdraw)


class Market:
    def __init__(self, day, cfg, carry=()):
        self.day, self.start = day, open_ns(day)
        self.quote_start = self.start + QUOTE_START_SECOND * SECOND
        self.end, self.withdraw = self.start + CLOSE_SECOND * SECOND, self.start + MAKER_WITHDRAW_SECOND * SECOND
        grid = load_day(day)
        scales = {r["ValueCode"]: r for r in qlevel.table(day).iter_rows(named=True)}
        mapping = pl.read_parquet(mapping_path(day))
        self.contracts = {}
        for r in mapping.iter_rows(named=True):
            vc = r["ValueCode"]
            if vc not in grid:
                continue
            c = Contract(vc, r["QuoteCode"], int(r["contract_size"]), r["end_date"].strftime("%Y%m%d"),
                         round(r["spot_ref_price"] * 10000), round(r["fut_ref_price"] * 10000))
            if c.shares != 2000:
                raise ValueError(f"non-standard new-entry contract: {c}")
            self.contracts[vc] = c
        self.exact = {c.qc: c for c in self.contracts.values()}
        # Preload the entire mapped universe: even a product with no new-entry
        # signal can carry yesterday's exposure and require an exit today.
        needed = list(self.contracts.values())
        needed = list({c.qc: c for c in [*needed, *carry]}.values())
        self.books, self.prints, self.limits, self.inputs = load_session(
            day, sorted({c.vc for c in needed}), sorted(c.qc for c in needed))
        self.timelines = {}
        frames = []
        for vc, c in self.contracts.items():
            if "S:" + vc not in self.books or "F:" + c.qc not in self.books:
                continue
            tl = Timeline(self, c, grid[vc], scales.get(vc, {}), cfg)
            self.timelines[c.qc] = tl
            for stream in cfg.streams:
                ix = tl.candidates[stream]
                frames.append(pl.DataFrame(dict(ns=tl.ns[ix], qc=[c.qc]*len(ix), stream=[stream]*len(ix), index=ix),
                                           schema={"ns": pl.Int64, "qc": pl.String, "stream": pl.String, "index": pl.Int64}))
        self.candidates = pl.concat(frames).sort(["ns", "qc", "stream"]) if frames else pl.DataFrame()
        self._book_cache = {}
        self.marks = self.load_marks()
        self.add_carry(carry, cfg)
        metadata = Path("maker/data/ev_lookup_v19_metadata_20260908/announcements/index.json")
        self.corporate_block = {r["vc"] for r in json.loads(metadata.read_text())
                                if "error" not in r and r["announce_day"] < day <= r["effective_day"]}

    def add_carry(self, carry, cfg):
        extra, paths = refresh_carry(self.day, [c for c in carry if c.qc not in self.exact])
        self.exact.update({c.qc: c for c in extra})
        missing = [c for c in extra if "S:" + c.vc not in self.books or "F:" + c.qc not in self.books]
        if missing:
            books, prints, limits, raw = load_session(self.day, sorted({c.vc for c in missing}),
                                                      sorted({c.qc for c in missing}))
            # Keep one snapshot identity for a shared spot book.
            for ins, book in books.items():
                self.books.setdefault(ins, book)
            for ins, pr in prints.items():
                self.prints.setdefault(ins, pr)
            self.limits.update(limits)
            paths += raw
        self.inputs = sorted(set(self.inputs + paths))
        for c in extra:
            if "S:" + c.vc in self.books and "F:" + c.qc in self.books:
                self.timelines[c.qc] = Timeline(self, c, None, {}, cfg)
        self.marks = self.load_marks()

    def book(self, instrument, ns):
        s = self.books.get(instrument)
        if s is None:
            return None
        i = s.index_at(ns)
        if i < 0:
            return None
        key = (instrument, i)
        if key not in self._book_cache:
            self._book_cache[key] = Book(instrument, int(s.ns[i]), int(s.seq[i]),
                                         tuple(s.levels(i, "bid")), tuple(s.levels(i, "ask")), bool(s.formal[i]))
        return self._book_cache[key]

    def next_book(self, instrument, ns):
        s = self.books.get(instrument)
        if s is None:
            return None
        j = int(np.searchsorted(s.ns, ns, side="right"))
        return int(s.ns[j]) if j < len(s.ns) else None

    def load_marks(self):
        from ..common.books import raw_paths
        _, _, md = raw_paths(self.day)
        sp = pl.read_parquet(md, columns=["quote_code", "close_price"])
        result = {"S:" + vc: (round(px * 10000), "official_spot_close")
                  for vc, px in sp.iter_rows() if px is not None and px > 0}
        path = Path("maker/data/ev_lookup_v19_metadata_20260908/official_future_daily_marks.parquet")
        fu = (pl.scan_parquet(path).filter(pl.col("date").dt.strftime("%Y%m%d") == self.day).collect())
        result.update({"F:" + qc: (round(px * 10000), "official_future_daily")
                       for _, qc, px in fu.iter_rows() if px is not None and px > 0})
        for key, s in self.books.items():
            if key in result:
                continue
            ixs = np.flatnonzero(s.valid)
            if ixs.size:
                i = int(ixs[-1])
                result[key] = (round((int(s.top_bid_px[i]) + int(s.top_ask_px[i])) / 2), "last_valid_bbo_mid")
        return result
