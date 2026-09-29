"""S2 point table: futures maker quotes on both sides, hedged in spot at +50 ms.

side=sell (entry): ask one tick inside A1 (queue first); locked basis P / SpotA1; fills from futures prints
                   >= P; hedge buys the spot shares from the ask; t_below_* = residual floors (SpotA1 rising).
side=buy  (exit) : bid one tick inside B1 (queue first); locked basis P / SpotB1; fills from futures prints
                   <= P; hedge sells the spot shares into the bid; t_above_* = basis rising above the
                   quote-time basis (SpotB1 falling).
Policy-independent facts (independent-event, no capacity, no own-order interaction); policies filter.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import numpy as np
import polars as pl

from ..common.books import INF, BookSeries, Prints, load_session, previous_tick, tick_i
from ..common.grid import ProductDay, load_day
from ..common.paths import (CLOSE_SECOND, MAKER_WITHDRAW_SECOND, QUOTE_END_SECOND, QUOTE_START_SECOND,
                            SECOND, grid_days, open_ns, points_path)
from ..ev import qlevel
from .common import sweep_hedge

FLOORS = (0, 5, 10, 15, 20, 25, 30)      # residual floors (bp) for t_below
MOVES = (5, 10, 20)                       # quotable-price moves (bp of price) for t_up / t_down
RISES = (0, 5, 10, 20, 30)                # buy-side deterioration: basis rises this many bp above the quote-time basis
MIN_RESIDUAL_BP = 10.0                    # sell-side admission: looser than any policy floor
MAX_EXIT_RESIDUAL_BP = 10.0               # buy-side admission: quote basis at most 10 bp above the live anchor
CANCEL_NS = 50_000_000
HEDGE_NS = 50_000_000
HEDGE_TIMEOUT_NS = 5 * SECOND
REF_LO, REF_HI = 0.91, 1.08
BLOCK = 64

SCHEMA = {
    "stream": pl.String, "side": pl.String, "date": pl.String, "vc": pl.String, "qc": pl.String, "expiry": pl.String,
    "quote_ns": pl.Int64, "quote_second": pl.Int32, "price": pl.Int64, "event_kind": pl.String,
    "anchor": pl.Float64, "scale": pl.Float64, "quote_ab": pl.Float64, "eff_u": pl.Float64,
    "resid_mid_bp": pl.Float64, "e_norm": pl.Float64, "fut_a1": pl.Int64, "fut_b1": pl.Int64,
    "spot_a1": pl.Int64, "spot_b1": pl.Int64, "fut_spread_bp": pl.Float64, "tick_bp_hedge": pl.Float64, "depth_ahead": pl.Int64,
    "opp_depth_shares": pl.Int64, "notional_twd": pl.Float64,
    "t_fill_ns": pl.Int64, "fill_print_seq": pl.Int64, "fill_price": pl.Int64, "fill_kind": pl.String,
    **{f"t_below_{f}": pl.Int64 for f in FLOORS}, "t_ab0_ns": pl.Int64,
    **{f"t_above_{g}": pl.Int64 for g in RISES}, "t_gate_ns": pl.Int64,
    **{f"t_up_{d}": pl.Int64 for d in MOVES}, **{f"t_down_{d}": pl.Int64 for d in MOVES},
    "hedge_ns": pl.Int64, "hedge_vwap": pl.Int64, "hedge_levels_swept": pl.Int32, "hedge_wait_ms": pl.Float64,
    "hedge_timeout": pl.Boolean, "actual_ab": pl.Float64, "d_in_realized": pl.Float64,
    "cancel_race_20": pl.Boolean, "negative_basis": pl.Boolean,
}


def tick_vec(p: np.ndarray) -> np.ndarray:
    return np.select([p < 100_000, p < 500_000, p < 1_000_000, p < 5_000_000, p < 10_000_000],
                     [100, 500, 1000, 5000, 10_000], default=50_000)


def previous_tick_vec(p: np.ndarray) -> np.ndarray:
    return p - tick_vec(p - 1)


def next_tick_vec(p: np.ndarray) -> np.ndarray:
    return p + tick_vec(p)


@dataclass
class ProductInputs:
    day: str
    vc: str
    qc: str
    shares: int
    expiry: str
    spot_ref: int
    fut_ref: int
    spot: BookSeries
    fut: BookSeries
    fut_prints: Prints | None
    anchor: np.ndarray        # per second, nan when unknown
    resid_mid: np.ndarray     # basis_mid - anchor per second
    scale: float | None
    spot_limits: tuple[int, int]


class Timeline:
    """Forward-filled pair state on the union of spot top-ask events, futures top events and second boundaries."""

    def __init__(self, p: ProductInputs, start: int, end: int):
        secs = start + np.arange(CLOSE_SECOND + 1, dtype=np.int64) * SECOND
        s_ev = p.spot.ns[(p.spot.ns >= start) & (p.spot.ns <= end)]
        f_ev = p.fut.ns[(p.fut.ns >= start) & (p.fut.ns <= end)]
        self.t = np.unique(np.concatenate([secs, s_ev, f_ev]))
        i_s = np.searchsorted(p.spot.ns, self.t, side="right") - 1
        i_f = np.searchsorted(p.fut.ns, self.t, side="right") - 1
        sec = ((self.t - start) // SECOND).astype(np.int64)
        sec = np.clip(sec, 0, len(p.anchor) - 1)     # grid arrays cover seconds 0..CLOSE_SECOND-1
        self.anchor = p.anchor[sec]
        ok_s = (i_s >= 0)
        ok_f = (i_f >= 0)
        i_s0, i_f0 = np.maximum(i_s, 0), np.maximum(i_f, 0)
        self.a1 = np.where(ok_s, p.spot.top_ask_px[i_s0], INF)
        self.b1 = np.where(ok_s, p.spot.top_bid_px[i_s0], 0)
        sb, sa = p.spot.top_bid_px[i_s0], p.spot.top_ask_px[i_s0]
        fb, fa = p.fut.top_bid_px[i_f0], p.fut.top_ask_px[i_f0]
        ok = (ok_s & ok_f & p.spot.valid[i_s0] & p.fut.valid[i_f0] & np.isfinite(self.anchor)
              & (sb > p.spot_ref * REF_LO) & (sa < p.spot_ref * REF_HI)
              & (fb > p.fut_ref * REF_LO) & (fa < p.fut_ref * REF_HI))
        self.ok = ok
        n = len(self.t)
        nb = np.full(n + 1, n, dtype=np.int64)
        for i in range(n - 1, -1, -1):
            nb[i] = i if not ok[i] else nb[i + 1]
        self.next_bad = nb
        # eff_u < f  <=>  P / A1 - 1 < (anchor + f) / 1e4  <=>  A1 * (1 + (anchor + f) / 1e4) > P
        a1f = np.where(self.a1 < INF, self.a1.astype(float), -np.inf)
        self.g = {}
        for f in FLOORS:
            self.g[f] = self._series(a1f * (1.0 + (self.anchor + f) / 10_000.0))
        # ab <= 0  <=>  A1 >= P  <=>  A1 + 0.5 > P  (integer prices)
        self.g["ab0"] = self._series(a1f + 0.5)
        # quotable futures price after t0: a better pair appears when it rises above ours (t_up),
        # the market leaves us behind when it falls below ours (t_down)
        fa_ok = ok_f & p.fut.valid[i_f0] & (fa < INF)
        quotable = np.where(fa_ok, previous_tick_vec(np.where(fa_ok, fa, 10)).astype(float), np.nan)
        self.g["up"] = self._series(quotable)
        self.g["down"] = self._series(-quotable)
        # buy side: basis rises  <=>  SpotB1 falls: -SpotB1 + 0.5 > -P / (1 + (ab_q + g)/1e4); quotable bid = B1 + tick
        b1f = np.where(ok_s & p.spot.valid[i_s0] & (self.b1 > 0), self.b1.astype(float), -np.inf)
        self.g["bid"] = self._series(np.where(np.isfinite(b1f), -b1f + 0.5, -np.inf))
        fb_ok = ok_f & p.fut.valid[i_f0] & (fb > 0)
        quotable_bid = np.where(fb_ok, next_tick_vec(np.where(fb_ok, fb, 10)).astype(float), np.nan)
        self.g["bup"] = self._series(quotable_bid)
        self.g["bdown"] = self._series(-quotable_bid)

    def _series(self, values: np.ndarray):
        values = np.where(np.isfinite(values), values, -np.inf)
        n = len(values)
        pad = (-n) % BLOCK
        bmax = np.concatenate([values, np.full(pad, -np.inf)]).reshape(-1, BLOCK).max(axis=1)
        return values, bmax

    def first_exceed(self, key, i0: int, thr: float) -> int:
        """First timeline index >= i0 whose value exceeds thr; -1 if none."""
        values, bmax = self.g[key]
        n = len(values)
        if i0 >= n:
            return -1
        b = i0 // BLOCK
        seg = values[i0:min((b + 1) * BLOCK, n)]
        hit = np.flatnonzero(seg > thr)
        if hit.size:
            return i0 + int(hit[0])
        later = np.flatnonzero(bmax[b + 1:] > thr)
        if later.size == 0:
            return -1
        bb = b + 1 + int(later[0])
        seg = values[bb * BLOCK:min((bb + 1) * BLOCK, n)]
        return bb * BLOCK + int(np.flatnonzero(seg > thr)[0])


def candidate_events(p: ProductInputs, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    """Event times (ns) and kinds at which a fresh S2 quote could be sent."""
    lo, hi = start + QUOTE_START_SECOND * SECOND, start + QUOTE_END_SECOND * SECOND
    secs = start + np.arange(QUOTE_START_SECOND, QUOTE_END_SECOND, dtype=np.int64) * SECOND
    parts = [(secs, "second"),
             (p.fut.ns[(p.fut.ns >= lo) & (p.fut.ns < hi)], "fut_book"),
             (p.spot.ns[(p.spot.ns >= lo) & (p.spot.ns < hi)], "spot_book")]
    if p.fut_prints is not None:
        parts.append((p.fut_prints.ns[(p.fut_prints.ns >= lo) & (p.fut_prints.ns < hi)], "fut_print"))
    t = np.concatenate([a for a, _ in parts])
    kinds = np.concatenate([np.full(len(a), k) for a, k in parts])
    order = np.lexsort((kinds, t))   # stable by time; kind order irrelevant for identical ns
    t, kinds = t[order], kinds[order]
    keep = np.ones(len(t), dtype=bool)
    keep[1:] = t[1:] != t[:-1]
    return t[keep], kinds[keep]


def product_rows(p: ProductInputs) -> list[dict]:
    start = open_ns(p.day)
    end = start + CLOSE_SECOND * SECOND
    withdraw = start + MAKER_WITHDRAW_SECOND * SECOND
    if len(p.spot.ns) == 0 or len(p.fut.ns) == 0:
        return []
    tl = Timeline(p, start, end)
    t, kinds = candidate_events(p, start, end)
    if len(t) == 0:
        return []
    i_s = np.searchsorted(p.spot.ns, t, side="right") - 1
    i_f = np.searchsorted(p.fut.ns, t, side="right") - 1
    sec = ((t - start) // SECOND).astype(np.int64)
    anchor = p.anchor[sec]
    ok = (i_s >= 0) & (i_f >= 0)
    i_s0, i_f0 = np.maximum(i_s, 0), np.maximum(i_f, 0)
    ok &= p.spot.valid[i_s0] & p.fut.valid[i_f0] & np.isfinite(anchor)
    sb, sa, sq = p.spot.top_bid_px[i_s0], p.spot.top_ask_px[i_s0], p.spot.top_ask_qty[i_s0]
    fb, fa = p.fut.top_bid_px[i_f0], p.fut.top_ask_px[i_f0]
    ok &= (sb > p.spot_ref * REF_LO) & (sa < p.spot_ref * REF_HI) & (fb > p.fut_ref * REF_LO) & (fa < p.fut_ref * REF_HI)
    fa_safe = np.where(fa < INF, fa, 10)
    price = previous_tick_vec(fa_safe)
    ok &= (fa < INF) & (price > fb) & (price > 0)
    sa_safe = np.where((sa < INF) & (sa > 0), sa, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ab = (price / sa_safe - 1.0) * 10_000.0
        eff = ab - anchor
    ok &= (ab > 0) & (eff >= MIN_RESIDUAL_BP)
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return buy_rows(p, tl, t, kinds, i_s, i_f, sec, anchor, start, end, withdraw)
    # de-duplicate unchanged quote states; prints at/above the quote change the fill outcome for later quotes
    state = np.stack([price[idx], sa[idx], sq[idx], fa[idx], fb[idx]], axis=1)
    changed = np.ones(len(idx), dtype=bool)
    changed[1:] = np.any(state[1:] != state[:-1], axis=1)
    is_print = kinds[idx] == "fut_print"
    print_hits = np.zeros(len(idx), dtype=bool)
    if p.fut_prints is not None and is_print.any():
        pos = np.searchsorted(p.fut_prints.ns, t[idx[is_print]], side="left")
        pos = np.clip(pos, 0, len(p.fut_prints.ns) - 1)
        print_hits[is_print] = p.fut_prints.price[pos] >= price[idx[is_print]]
    # a row is a quote-state segment valid until the next row: emit on state change, on prints that
    # would fill the segment's quote, and when the anchor moves eff_u across a floor (second boundaries)
    floor_bucket = np.searchsorted(np.asarray(FLOORS, dtype=float), eff[idx], side="right")
    is_second = kinds[idx] == "second"
    emit = np.zeros(len(idx), dtype=bool)
    last_bucket = -1
    for k in range(len(idx)):
        if changed[k] or print_hits[k] or (is_second[k] and floor_bucket[k] != last_bucket):
            emit[k] = True
            last_bucket = floor_bucket[k]
    idx = idx[emit]
    rows = []
    # fill lookup per distinct price: first print at or above the quote strictly after quote time
    fill_cache = {}
    prints = p.fut_prints
    limit_up = p.spot_limits[1]
    tick_hedge = tick_vec(sa[idx]) / np.maximum(sa[idx], 1) * 10_000.0
    for n, k in enumerate(idx):
        t0, P = int(t[k]), int(price[k])
        if prints is not None:
            if P not in fill_cache:
                m = prints.price >= P
                fill_cache[P] = (prints.ns[m], prints.seq[m], prints.price[m])
            ns_p, seq_p, px_p = fill_cache[P]
            j = int(np.searchsorted(ns_p, t0, side="right"))
            filled = j < len(ns_p) and ns_p[j] <= withdraw
        else:
            filled = False
        t_fill = int(ns_p[j]) if filled else None
        # deterioration and gate from the forward-filled timeline
        i_prev = int(np.searchsorted(tl.t, t0, side="right")) - 1
        below = {}
        for key in (*FLOORS, "ab0"):
            values, _ = tl.g[key]
            if i_prev >= 0 and values[i_prev] > P:
                below[key] = t0
            else:
                jx = tl.first_exceed(key, i_prev + 1, P)
                below[key] = int(tl.t[jx]) if jx >= 0 else None
        jb = tl.next_bad[i_prev + 1] if i_prev + 1 < len(tl.t) else len(tl.t)
        t_gate = int(tl.t[jb]) if jb < len(tl.t) else None
        moves = {}
        for d in MOVES:
            ju = tl.first_exceed("up", i_prev + 1, P * (1.0 + d / 10_000.0))
            jd = tl.first_exceed("down", i_prev + 1, -P * (1.0 - d / 10_000.0))
            moves[f"t_up_{d}"] = int(tl.t[ju]) if ju >= 0 else None
            moves[f"t_down_{d}"] = int(tl.t[jd]) if jd >= 0 else None
        # mandatory spot hedge from +50 ms, L1-L5 full quantity, retry on later books
        hedge = None
        if filled:
            th = t_fill + HEDGE_NS
            i = p.spot.index_at(th)
            timeout = False
            while 0 <= i < len(p.spot.ns):
                if p.spot.ns[i] > th:
                    th = int(p.spot.ns[i])
                if th - t_fill > HEDGE_TIMEOUT_NS:
                    timeout = True          # flagged, never abandoned: the exposure stays until depth appears
                if p.spot.valid[i]:
                    taken = p.spot.take(i, "ask", p.shares, 1, limit_up)
                    if taken is not None:
                        hedge = (th, taken[0], taken[1])
                        break
                i += 1
                if i < len(p.spot.ns) and p.spot.ns[i] > end:
                    break
        quote_ab = float(ab[k])
        row = dict(stream="S2", side="sell", date=p.day, vc=p.vc, qc=p.qc, expiry=p.expiry, quote_ns=t0, quote_second=int(sec[k]),
                   price=P, event_kind=str(kinds[k]), anchor=float(anchor[k]), scale=p.scale, quote_ab=quote_ab,
                   eff_u=float(eff[k]), resid_mid_bp=float(p.resid_mid[sec[k]]),
                   e_norm=(float(p.resid_mid[sec[k]]) / p.scale) if p.scale else None,
                   fut_a1=int(fa[k]), fut_b1=int(fb[k]), spot_a1=int(sa[k]), spot_b1=int(sb[k]),
                   fut_spread_bp=float((fa[k] - fb[k]) / fb[k] * 10_000.0), tick_bp_hedge=float(tick_hedge[n]),
                   depth_ahead=0, opp_depth_shares=int(sq[k]), notional_twd=float(sa[k]) * p.shares / 10_000.0,
                   t_fill_ns=t_fill, fill_print_seq=int(seq_p[j]) if filled else None,
                   fill_price=int(px_p[j]) if filled else None,
                   fill_kind=("at_price" if px_p[j] == P else "through") if filled else None,
                   **{f"t_below_{f}": below[f] for f in FLOORS}, t_ab0_ns=below["ab0"],
                   **{f"t_above_{g}": None for g in RISES}, t_gate_ns=t_gate, **moves,
                   hedge_ns=hedge[0] if hedge else None, hedge_vwap=int(round(hedge[1] / p.shares)) if hedge else None,
                   hedge_levels_swept=hedge[2] if hedge else None,
                   hedge_wait_ms=(hedge[0] - t_fill) / 1e6 if hedge else None,
                   hedge_timeout=(timeout if filled else None),
                   actual_ab=(P / (hedge[1] / p.shares) - 1.0) * 10_000.0 if hedge else None)
        row["d_in_realized"] = quote_ab - row["actual_ab"] if hedge else None
        tb20 = below[20]
        row["cancel_race_20"] = bool(filled and tb20 is not None and tb20 < t_fill <= tb20 + CANCEL_NS)
        row["negative_basis"] = bool(hedge and row["actual_ab"] <= 0)
        rows.append(row)
    rows.extend(buy_rows(p, tl, t, kinds, i_s, i_f, sec, anchor, start, end, withdraw))
    return rows


def buy_rows(p: ProductInputs, tl: Timeline, t, kinds, i_s, i_f, sec, anchor, start, end, withdraw) -> list[dict]:
    """side=buy: futures bid one tick inside B1; the exit-side quote of a short-basis position."""
    ok = (i_s >= 0) & (i_f >= 0)
    i_s0, i_f0 = np.maximum(i_s, 0), np.maximum(i_f, 0)
    ok &= p.spot.valid[i_s0] & p.fut.valid[i_f0] & np.isfinite(anchor)
    sb, sa, sbq = p.spot.top_bid_px[i_s0], p.spot.top_ask_px[i_s0], p.spot.top_bid_qty[i_s0]
    fb, fa = p.fut.top_bid_px[i_f0], p.fut.top_ask_px[i_f0]
    ok &= (sb > p.spot_ref * REF_LO) & (sa < p.spot_ref * REF_HI) & (fb > p.fut_ref * REF_LO) & (fa < p.fut_ref * REF_HI)
    ok &= (sb > 0) & (sa < INF) & (fb > 0) & (fa < INF)
    fb_safe = np.where(fb > 0, fb, 10)
    price = next_tick_vec(fb_safe)
    ok &= (price < fa)
    with np.errstate(divide="ignore", invalid="ignore"):
        ab = (price / np.where(sb > 0, sb, 1) - 1.0) * 10_000.0
        eff = ab - anchor
    ok &= eff <= MAX_EXIT_RESIDUAL_BP
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return []
    rises = np.asarray(RISES, dtype=float)
    bucket = np.searchsorted(-rises[::-1], -eff[idx], side="right")
    is_second = kinds[idx] == "second"
    emit = np.zeros(len(idx), dtype=bool)
    last_p, last_bucket = None, -1
    for k in range(len(idx)):
        if price[idx[k]] != last_p or (is_second[k] and bucket[k] != last_bucket):
            emit[k] = True
            last_p, last_bucket = price[idx[k]], bucket[k]
    idx = idx[emit]
    rows, fill_cache, hedge_cache = [], {}, {}
    prints = p.fut_prints
    limit_down = p.spot_limits[0]
    for k in idx:
        t0, P = int(t[k]), int(price[k])
        filled, t_fill, j = False, None, None
        if prints is not None:
            if P not in fill_cache:
                m = prints.price <= P
                fill_cache[P] = (prints.ns[m], prints.seq[m], prints.price[m])
            ns_p, seq_p, px_p = fill_cache[P]
            j = int(np.searchsorted(ns_p, t0, side="right"))
            filled = j < len(ns_p) and ns_p[j] <= withdraw
            t_fill = int(ns_p[j]) if filled else None
        quote_ab = float(ab[k])
        i_prev = int(np.searchsorted(tl.t, t0, side="right")) - 1
        above = {}
        for g in RISES:
            thr = -P / (1.0 + (quote_ab + g) / 10_000.0)
            values, _ = tl.g["bid"]
            if i_prev >= 0 and values[i_prev] > thr:
                above[g] = t0
            else:
                jx = tl.first_exceed("bid", i_prev + 1, thr)
                above[g] = int(tl.t[jx]) if jx >= 0 else None
        jb = tl.next_bad[i_prev + 1] if i_prev + 1 < len(tl.t) else len(tl.t)
        t_gate = int(tl.t[jb]) if jb < len(tl.t) else None
        moves = {}
        for d in MOVES:
            ju = tl.first_exceed("bup", i_prev + 1, P * (1.0 + d / 10_000.0))
            jd = tl.first_exceed("bdown", i_prev + 1, -P * (1.0 - d / 10_000.0))
            moves[f"t_up_{d}"] = int(tl.t[ju]) if ju >= 0 else None
            moves[f"t_down_{d}"] = int(tl.t[jd]) if jd >= 0 else None
        hedge = None
        if filled:
            if t_fill in hedge_cache:
                hedge = hedge_cache[t_fill]
            else:
                hedge = hedge_cache[t_fill] = sweep_hedge(p.spot, "bid", p.shares, t_fill, HEDGE_NS, HEDGE_TIMEOUT_NS,
                                                         limit_down, INF, end)
        actual_ab = (P / (hedge[1] / p.shares) - 1.0) * 10_000.0 if hedge else None
        rows.append(dict(
            stream="S2", side="buy", date=p.day, vc=p.vc, qc=p.qc, expiry=p.expiry, quote_ns=t0, quote_second=int(sec[k]),
            price=P, event_kind=str(kinds[k]), anchor=float(anchor[k]), scale=p.scale, quote_ab=quote_ab,
            eff_u=float(eff[k]), resid_mid_bp=float(p.resid_mid[sec[k]]),
            e_norm=(float(p.resid_mid[sec[k]]) / p.scale) if p.scale else None,
            fut_a1=int(fa[k]), fut_b1=int(fb[k]), spot_a1=int(sa[k]), spot_b1=int(sb[k]),
            fut_spread_bp=float((fa[k] - fb[k]) / fb[k] * 10_000.0),
            tick_bp_hedge=float(tick_vec(np.array([sb[k]]))[0] / sb[k] * 10_000.0),
            depth_ahead=0, opp_depth_shares=int(sbq[k]), notional_twd=float(sb[k]) * p.shares / 10_000.0,
            t_fill_ns=t_fill, fill_print_seq=int(seq_p[j]) if filled else None, fill_price=int(px_p[j]) if filled else None,
            fill_kind=("at_price" if px_p[j] == P else "through") if filled else None,
            **{f"t_below_{f}": None for f in FLOORS}, t_ab0_ns=None,
            **{f"t_above_{g}": above[g] for g in RISES}, t_gate_ns=t_gate, **moves,
            hedge_ns=hedge[0] if hedge else None, hedge_vwap=int(round(hedge[1] / p.shares)) if hedge else None,
            hedge_levels_swept=hedge[2] if hedge else None, hedge_wait_ms=(hedge[0] - t_fill) / 1e6 if hedge else None,
            hedge_timeout=(hedge[3] if hedge else None) if filled else None,
            actual_ab=actual_ab, d_in_realized=(quote_ab - actual_ab) if hedge else None,
            cancel_race_20=False, negative_basis=False))
    return rows


def build_day(day: str, products: list[str] | None = None, *, write: bool = True) -> pl.DataFrame:
    started = time.time()
    grid = load_day(day, products)
    levels = qlevel.table(day)
    scale = {r["ValueCode"]: r["scale"] for r in levels.iter_rows(named=True)}
    mapping = pl.read_parquet(points_mapping(day))
    meta = {r["ValueCode"]: r for r in mapping.iter_rows(named=True)}
    vcs = [vc for vc in grid if vc in meta]
    qcs = [grid[vc].qc for vc in vcs]
    books, prints, limits, inputs = load_session(day, vcs, qcs)
    frames, counts = [], {}
    for vc in vcs:
        g, m = grid[vc], meta[vc]
        spot, fut = books.get("S:" + vc), books.get("F:" + g.qc)
        if spot is None or fut is None or vc not in limits:
            counts[vc] = "missing_books"
            continue
        inp = ProductInputs(day, vc, g.qc, int(round(m["contract_size"])), g.expiry,
                            int(round(m["spot_ref_price"] * 10_000)), int(round(m["fut_ref_price"] * 10_000)),
                            spot, fut, prints.get("F:" + g.qc), g.anchor, g.mid - g.anchor, scale.get(vc), limits[vc])
        rows = product_rows(inp)
        counts[vc] = len(rows)
        if rows:
            frames.append(pl.from_dicts(rows, schema=SCHEMA))
    frame = pl.concat(frames) if frames else pl.DataFrame(schema=SCHEMA)
    if write:
        path = points_path(day, "s2_entries")
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)
        manifest = dict(day=day, rows=frame.height, products=len(vcs), inputs=inputs, elapsed_s=round(time.time() - started, 1),
                        params=dict(floors=FLOORS, rises=RISES, moves=MOVES, min_residual_bp=MIN_RESIDUAL_BP,
                                    max_exit_residual_bp=MAX_EXIT_RESIDUAL_BP,
                                    cancel_ms=CANCEL_NS / 1e6, hedge_ms=HEDGE_NS / 1e6, hedge_timeout_s=HEDGE_TIMEOUT_NS / SECOND),
                        counts=counts)
        (path.parent / "s2_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return frame


def points_mapping(day: str):
    from ..common.paths import mapping_path
    return mapping_path(day)


def sequential_fills(frame: pl.DataFrame, *, residual_bp: float = 25.0, floor_bp: float = 20.0,
                     depth_mult: int = 5, cancel_ns: int = CANCEL_NS, repeg_bp: int | None = None,
                     patience_ns: int = 0) -> pl.DataFrame:
    """Reference walker: one live quote per product; the next quote goes out the instant the previous
    one is cancelled (effective) or its hedge completes. No cooldown.

    Rows are quote-state segments: a row's outcomes hold for any quote time between its quote_ns and
    the next row's. A quote is filled when t_fill <= first cancel trigger + latency; cancel triggers are
    the floor / gate / ab<=0 times and, when repeg_bp is set, a better pair (t_up_{repeg_bp}) appearing
    after `patience_ns` of waiting.
    """
    out = []
    floor_col, up_col = f"t_below_{int(floor_bp)}", f"t_up_{repeg_bp}" if repeg_bp else None
    if "side" in frame.columns:
        frame = frame.filter(pl.col("side") == "sell")
    for vc, g in frame.sort(["vc", "quote_ns"]).group_by("vc", maintain_order=True):
        vc = vc[0] if isinstance(vc, tuple) else vc
        rows = g.to_dicts()
        starts = [r["quote_ns"] for r in rows]
        cursor = -1
        shares = None
        for i, r in enumerate(rows):
            seg_end = starts[i + 1] if i + 1 < len(rows) else None
            if seg_end is not None and seg_end <= cursor:
                continue
            q_t = max(r["quote_ns"], cursor)
            if r["eff_u"] < residual_bp:
                continue
            if shares is None:
                shares = int(round(r["notional_twd"] * 10_000 / r["spot_a1"]))
            if r["opp_depth_shares"] < depth_mult * shares:
                continue
            triggers = [v for v in (r[floor_col], r["t_gate_ns"], r["t_ab0_ns"]) if v is not None]
            if any(v <= q_t for v in triggers) or (r["t_fill_ns"] is not None and r["t_fill_ns"] <= q_t):
                continue      # state already stale at this quote time (anchor-driven crossing inside the segment)
            if up_col and r[up_col] is not None and r[up_col] >= q_t + patience_ns:
                triggers.append(r[up_col])
            cancel_at = min(triggers) if triggers else None
            filled = r["t_fill_ns"] is not None and (cancel_at is None or r["t_fill_ns"] <= cancel_at + cancel_ns)
            if filled:
                out.append(dict(vc=vc, quote_ns=q_t, t_fill_ns=r["t_fill_ns"], hedge_ns=r["hedge_ns"],
                                quote_ab=r["quote_ab"], actual_ab=r["actual_ab"], eff_u=r["eff_u"],
                                wait_s=(r["t_fill_ns"] - q_t) / 1e9,
                                cancel_race=(cancel_at is not None and r["t_fill_ns"] > cancel_at),
                                within_50ms=(r["t_fill_ns"] - q_t <= 50_000_000)))
                cursor = r["hedge_ns"] if r["hedge_ns"] is not None else r["t_fill_ns"] + HEDGE_TIMEOUT_NS
            elif cancel_at is not None:
                cursor = cancel_at + cancel_ns
            else:
                break   # quote stays working until withdrawal
            # every cancel/fill trigger is itself a book, print or floor-crossing event, so the
            # cursor lands in a later segment; the loop simply continues from there
    return pl.DataFrame(out) if out else pl.DataFrame()


def fill_wait_quantiles(frame: pl.DataFrame, floor_bp: float = 20.0) -> dict:
    """Seconds from quote to fill for candidates that fill before any cancel trigger."""
    floor_col = f"t_below_{int(floor_bp)}"
    if "side" in frame.columns:
        frame = frame.filter(pl.col("side") == "sell")
    g = frame.filter(pl.col("t_fill_ns").is_not_null()).with_columns(
        pl.min_horizontal(floor_col, "t_gate_ns", "t_ab0_ns").alias("cancel_at"))
    clean = g.filter(pl.col("cancel_at").is_null() | (pl.col("t_fill_ns") <= pl.col("cancel_at")))
    wait = (clean["t_fill_ns"] - clean["quote_ns"]) / 1e9
    return {"n": clean.height, **{f"p{q}": round(float(wait.quantile(q / 100)), 2) for q in (10, 25, 50, 75, 90, 95)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day", required=True)
    parser.add_argument("--products", help="comma-separated ValueCodes")
    parser.add_argument("--walk", action="store_true", help="run the reference sequential walker")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    if args.day not in grid_days():
        raise SystemExit("day needs the canonical grid (anchor)")
    frame = build_day(args.day, args.products.split(",") if args.products else None, write=not args.no_write)
    summary = dict(day=args.day, rows=frame.height, products=frame["vc"].n_unique())
    for side in ("sell", "buy"):
        f = frame.filter(pl.col("side") == side)
        filled = f.filter(pl.col("t_fill_ns").is_not_null())
        summary[side] = dict(rows=f.height, with_fill=filled.height,
                             hedge_timeouts=int(filled["hedge_timeout"].sum()) if filled.height else 0,
                             cancel_race_20=int(f["cancel_race_20"].sum()), negative_basis=int(f["negative_basis"].sum()))
    print(json.dumps(summary))
    print(json.dumps(dict(fill_wait_seconds=fill_wait_quantiles(frame))))
    if args.walk:
        w = sequential_fills(frame)
        if w.height:
            print(json.dumps(dict(walk_fills=w.height, products=w["vc"].n_unique(),
                                  mean_actual_ab=round(float(w["actual_ab"].mean()), 2),
                                  mean_quote_ab=round(float(w["quote_ab"].mean()), 2),
                                  mean_decay=round(float((w["quote_ab"] - w["actual_ab"]).mean()), 2),
                                  cancel_race=int(w["cancel_race"].sum()), within_50ms=int(w["within_50ms"].sum()),
                                  negative_basis=int((w["actual_ab"] <= 0).sum()))))
        else:
            print(json.dumps(dict(walk_fills=0)))


if __name__ == "__main__":
    main()
