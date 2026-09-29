"""S1 point table: spot maker quotes on both sides, hedged in futures at +50 ms.

side=buy  (entry): bid ladder inside / B1 / B1-1; locked basis FutBid / P; fills from spot prints <= P
                   after the displayed queue ahead; hedge sells one futures lot into the bid.
side=sell (exit) : ask ladder inside / A1; locked basis FutAsk / P; fills from spot prints >= P after the
                   displayed queue ahead; hedge buys one futures lot from the ask.
Every row also carries the user's precomputed makerFill seconds (Bid1/Bid2/Ask1/Ask2) at that spot tick.
Policy-independent rows; policies filter. Buy rows store t_below_* (residual floors, futures bid falling),
sell rows store t_above_* (basis rising above the quote-time basis, futures ask rising).
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import numpy as np
import polars as pl

from ..common.books import INF, BookSeries, Prints, load_session, next_tick, previous_tick, tick_i
from ..common.grid import load_day
from ..common.paths import (CLOSE_SECOND, MAKER_WITHDRAW_SECOND, QUOTE_END_SECOND, QUOTE_START_SECOND,
                            SECOND, grid_days, mapping_path, open_ns, points_path)
from ..ev import qlevel
from .common import FirstPassage, next_false, sweep_hedge
from .s2 import CANCEL_NS, FLOORS, HEDGE_NS, HEDGE_TIMEOUT_NS, MIN_RESIDUAL_BP, MOVES, REF_HI, REF_LO

RISES = (0, 5, 10, 20, 30)        # sell-side deterioration: basis rises this many bp above the quote-time basis
MAX_EXIT_RESIDUAL_BP = 10.0       # sell-side admission: quote basis at most 10 bp above the live anchor
LADDER_BELOW = 1                  # buy: B1 and B1-1 (+ inside); sell: A1 (+ inside)
BOARD_LOT = 1000
AHEAD_STEP = {"buy": 1.10, "sell": 1.20}   # queue-ahead segment buckets (relative steps, never finer than one lot)

SCHEMA = {
    "stream": pl.String, "side": pl.String, "date": pl.String, "vc": pl.String, "qc": pl.String, "expiry": pl.String,
    "quote_ns": pl.Int64, "quote_second": pl.Int32, "price": pl.Int64, "level": pl.Int32, "event_kind": pl.String,
    "anchor": pl.Float64, "scale": pl.Float64, "quote_ab": pl.Float64, "eff_u": pl.Float64,
    "resid_mid_bp": pl.Float64, "e_norm": pl.Float64, "spot_b1": pl.Int64, "spot_a1": pl.Int64,
    "fut_bid": pl.Int64, "fut_ask": pl.Int64, "fut_spread_bp": pl.Float64, "tick_bp_hedge": pl.Float64,
    "depth_ahead": pl.Int64, "opp_depth_lots": pl.Int64, "notional_twd": pl.Float64,
    "mf_bid1_s": pl.Float32, "mf_bid2_s": pl.Float32, "mf_ask1_s": pl.Float32, "mf_ask2_s": pl.Float32,
    "mf_fill_ns": pl.Int64,
    "t_partial_ns": pl.Int64, "t_fill_ns": pl.Int64, "fill_print_seq": pl.Int64, "fill_price": pl.Int64,
    "fill_kind": pl.String,
    **{f"t_below_{f}": pl.Int64 for f in FLOORS}, "t_ab0_ns": pl.Int64,
    **{f"t_above_{g}": pl.Int64 for g in RISES}, "t_gate_ns": pl.Int64,
    **{f"t_up_{d}": pl.Int64 for d in MOVES}, **{f"t_down_{d}": pl.Int64 for d in MOVES},
    "hedge_ns": pl.Int64, "hedge_vwap": pl.Int64, "hedge_levels_swept": pl.Int32, "hedge_wait_ms": pl.Float64,
    "hedge_timeout": pl.Boolean, "actual_ab": pl.Float64, "d_in_realized": pl.Float64,
    "cancel_race_20": pl.Boolean, "negative_basis": pl.Boolean,
}


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
    spot_prints: Prints | None
    anchor: np.ndarray
    resid_mid: np.ndarray
    scale: float | None
    spot_limits: tuple[int, int]
    makerfill: np.ndarray | None = None   # (len(spot.ns), 4): Bid1/Bid2/Ask1/Ask2 seconds aligned to spot book rows


def ahead_bucket(ahead: int, step: float) -> int:
    if ahead < BOARD_LOT:
        return 0
    return int(np.log(ahead / BOARD_LOT) / np.log(step)) + 1


class Timeline:
    """Deterioration/move series on futures events + second boundaries; the legality gate on the union."""

    def __init__(self, p: ProductInputs, start: int, end: int):
        secs = start + np.arange(CLOSE_SECOND + 1, dtype=np.int64) * SECOND
        s_ev = p.spot.ns[(p.spot.ns >= start) & (p.spot.ns <= end)]
        f_ev = p.fut.ns[(p.fut.ns >= start) & (p.fut.ns <= end)]
        self.tg = np.unique(np.concatenate([secs, s_ev, f_ev]))
        i_s = np.searchsorted(p.spot.ns, self.tg, side="right") - 1
        i_f = np.searchsorted(p.fut.ns, self.tg, side="right") - 1
        secg = np.clip(((self.tg - start) // SECOND).astype(np.int64), 0, len(p.anchor) - 1)
        ok_s, ok_f = i_s >= 0, i_f >= 0
        i_s0, i_f0 = np.maximum(i_s, 0), np.maximum(i_f, 0)
        sb, sa = p.spot.top_bid_px[i_s0], p.spot.top_ask_px[i_s0]
        fb, fa = p.fut.top_bid_px[i_f0], p.fut.top_ask_px[i_f0]
        self.ok = (ok_s & ok_f & p.spot.valid[i_s0] & p.fut.valid[i_f0] & np.isfinite(p.anchor[secg])
                   & (sb > p.spot_ref * REF_LO) & (sa < p.spot_ref * REF_HI)
                   & (fb > p.fut_ref * REF_LO) & (fa < p.fut_ref * REF_HI))
        self.next_bad = next_false(self.ok)
        self.t = np.unique(np.concatenate([secs, f_ev]))
        i_f = np.searchsorted(p.fut.ns, self.t, side="right") - 1
        sec = np.clip(((self.t - start) // SECOND).astype(np.int64), 0, len(p.anchor) - 1)
        self.anchor = p.anchor[sec]
        ok_f = i_f >= 0
        i_f0 = np.maximum(i_f, 0)
        fb, fa = p.fut.top_bid_px[i_f0], p.fut.top_ask_px[i_f0]
        valid = ok_f & p.fut.valid[i_f0]
        fbf = np.where(valid & (fb > 0), fb.astype(float), np.nan)
        faf = np.where(valid & (fa < INF), fa.astype(float), np.nan)
        self.g = {}
        # buy: eff_u < f  <=>  FutBid < P (1 + (anchor + f)/1e4)  <=>  -FutBid / (1 + (anchor + f)/1e4) > -P
        for f in FLOORS:
            self.g[("buy", f)] = FirstPassage(-fbf / (1.0 + (self.anchor + f) / 10_000.0))
        self.g[("buy", "ab0")] = FirstPassage(-fbf + 0.5)
        self.g[("buy", "up")] = FirstPassage(fbf)
        self.g[("buy", "down")] = FirstPassage(-fbf)
        # sell: basis rises g bp above the quote-time basis  <=>  FutAsk > P (1 + (ab_q + g)/1e4); threshold per row
        self.g[("sell", "ask")] = FirstPassage(faf)
        self.g[("sell", "down")] = FirstPassage(-faf)


def ladder_prices(side: str, b1: int, a1: int, limits: tuple[int, int]) -> list[tuple[int, int]]:
    """(level, price): -1 = one tick inside the spread; 0 = L1; k = k ticks away from L1 (buy side only)."""
    out = []
    if side == "buy":
        inside = next_tick(b1)
        if inside < a1:
            out.append((-1, inside))
        p = b1
        for k in range(LADDER_BELOW + 1):
            if p < limits[0] or p <= 0:
                break
            out.append((k, p))
            p = previous_tick(p)
    else:
        inside = previous_tick(a1)
        if inside > b1:
            out.append((-1, inside))
        if a1 <= limits[1]:
            out.append((0, a1))
    return out


def displayed_at(book: BookSeries, i: int, price: int, side: str = "bid") -> int:
    return sum(q for px, q in book.levels(i, side) if px == price)


def candidate_events(p: ProductInputs, start: int) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = start + QUOTE_START_SECOND * SECOND, start + QUOTE_END_SECOND * SECOND
    secs = start + np.arange(QUOTE_START_SECOND, QUOTE_END_SECOND, dtype=np.int64) * SECOND
    parts = [(secs, "second"),
             (p.fut.ns[(p.fut.ns >= lo) & (p.fut.ns < hi)], "fut_book"),
             (p.spot.ns[(p.spot.ns >= lo) & (p.spot.ns < hi)], "spot_book")]
    t = np.concatenate([a for a, _ in parts])
    kinds = np.concatenate([np.full(len(a), k) for a, k in parts])
    order = np.lexsort((kinds, t))
    t, kinds = t[order], kinds[order]
    keep = np.ones(len(t), dtype=bool)
    keep[1:] = t[1:] != t[:-1]
    return t[keep], kinds[keep]


def _fill(prints: Prints | None, cache: dict, side: str, P: int, t0: int, ahead: int, shares: int, withdraw: int):
    """(t_partial, t_full, seq, price) for a resting order at P placed at t0 with `ahead` shares in front."""
    if prints is None:
        return None, None, None, None
    key = (side, P)
    if key not in cache:
        m = prints.price <= P if side == "buy" else prints.price >= P
        cache[key] = (prints.ns[m], prints.seq[m], prints.price[m], np.cumsum(prints.qty[m]))
    ns_p, seq_p, px_p, cum = cache[key]
    j0 = int(np.searchsorted(ns_p, t0, side="right"))
    if j0 >= len(ns_p):
        return None, None, None, None
    base = cum[j0 - 1] if j0 > 0 else 0
    jp = int(np.searchsorted(cum, base + ahead + BOARD_LOT, side="left"))
    jf = int(np.searchsorted(cum, base + ahead + shares, side="left"))
    t_partial = int(ns_p[jp]) if jp < len(ns_p) and ns_p[jp] <= withdraw else None
    if jf < len(ns_p) and ns_p[jf] <= withdraw:
        return t_partial, int(ns_p[jf]), int(seq_p[jf]), int(px_p[jf])
    return t_partial, None, None, None


def product_rows(p: ProductInputs) -> list[dict]:
    start = open_ns(p.day)
    end = start + CLOSE_SECOND * SECOND
    withdraw = start + MAKER_WITHDRAW_SECOND * SECOND
    if len(p.spot.ns) == 0 or len(p.fut.ns) == 0:
        return []
    tl = Timeline(p, start, end)
    t, kinds = candidate_events(p, start)
    if len(t) == 0:
        return []
    i_s = np.searchsorted(p.spot.ns, t, side="right") - 1
    i_f = np.searchsorted(p.fut.ns, t, side="right") - 1
    sec = ((t - start) // SECOND).astype(np.int64)
    anchor = p.anchor[sec]
    ok = (i_s >= 0) & (i_f >= 0)
    i_s0, i_f0 = np.maximum(i_s, 0), np.maximum(i_f, 0)
    ok &= p.spot.valid[i_s0] & p.fut.valid[i_f0] & np.isfinite(anchor)
    sb, sa = p.spot.top_bid_px[i_s0], p.spot.top_ask_px[i_s0]
    fb, fa, fbq, faq = (p.fut.top_bid_px[i_f0], p.fut.top_ask_px[i_f0], p.fut.top_bid_qty[i_f0], p.fut.top_ask_qty[i_f0])
    ok &= (sb > p.spot_ref * REF_LO) & (sa < p.spot_ref * REF_HI) & (fb > p.fut_ref * REF_LO) & (fa < p.fut_ref * REF_HI)
    ok &= (sb > 0) & (sa < INF) & (fb > 0) & (fa < INF)
    idx_all = np.flatnonzero(ok)
    if idx_all.size == 0:
        return []
    last_state: dict[tuple, tuple] = {}
    last_bucket: dict[tuple, int] = {}
    fill_cache: dict = {}
    query_cache: dict = {}
    hedge_cache: dict = {}
    rows = []
    fut_lo, fut_hi = (p.fut_ref * 90 + 99) // 100, p.fut_ref * 110 // 100
    floors = np.asarray(FLOORS, dtype=float)
    rises = np.asarray(RISES, dtype=float)
    for k in idx_all:
        t0, kind = int(t[k]), str(kinds[k])
        b1, a1, fbid, fask, an = int(sb[k]), int(sa[k]), int(fb[k]), int(fa[k]), float(anchor[k])
        isp = int(i_s[k])
        mf = p.makerfill[isp] if p.makerfill is not None else None
        lv_bid = lv_ask = None
        for side in ("buy", "sell"):
            opp = fbid if side == "buy" else fask
            for level, P in ladder_prices(side, b1, a1, p.spot_limits):
                quote_ab = (opp / P - 1.0) * 10_000.0
                eff = quote_ab - an
                if side == "buy" and (quote_ab <= 0 or eff < MIN_RESIDUAL_BP):
                    continue
                if side == "sell" and eff > MAX_EXIT_RESIDUAL_BP:
                    continue
                ahead = 0 if level < 0 else displayed_at(p.spot, isp, P, "bid" if side == "buy" else "ask")
                state = (P, ahead_bucket(ahead, AHEAD_STEP[side]))
                bucket = int(np.searchsorted(floors if side == "buy" else -rises[::-1], eff if side == "buy" else -eff, side="right"))
                skey = (side, level)
                changed = last_state.get(skey) != state
                second_cross = kind == "second" and last_bucket.get(skey) != bucket
                if not (changed or second_cross):
                    continue
                last_state[skey], last_bucket[skey] = state, bucket
                t_partial, t_full, fill_seq, fill_px = _fill(p.spot_prints, fill_cache, side, P, t0, ahead, p.shares, withdraw)
                i_prev = int(np.searchsorted(tl.t, t0, side="right")) - 1
                below, above, moves = {}, {}, {}
                if side == "buy":
                    key_q = (side, P, i_prev)
                    cached = query_cache.get(key_q)
                    if cached is None:
                        for key in (*FLOORS, "ab0"):
                            fp = tl.g[("buy", key)]
                            if i_prev >= 0 and fp.values[i_prev] > -P:
                                below[key] = "now"
                            else:
                                jx = fp.first(i_prev + 1, -P)
                                below[key] = int(tl.t[jx]) if jx >= 0 else None
                        for d in MOVES:
                            ju = tl.g[("buy", "up")].first(i_prev + 1, fbid * (1.0 + d / 10_000.0))
                            jd = tl.g[("buy", "down")].first(i_prev + 1, -fbid * (1.0 - d / 10_000.0))
                            moves[f"t_up_{d}"] = int(tl.t[ju]) if ju >= 0 else None
                            moves[f"t_down_{d}"] = int(tl.t[jd]) if jd >= 0 else None
                        cached = query_cache[key_q] = (below, moves)
                    below = {kk: (t0 if v == "now" else v) for kk, v in cached[0].items()}
                    moves = cached[1]
                else:
                    fp = tl.g[("sell", "ask")]
                    for g in RISES:
                        thr = P * (1.0 + (quote_ab + g) / 10_000.0)
                        if i_prev >= 0 and fp.values[i_prev] > thr:
                            above[g] = t0
                        else:
                            jx = fp.first(i_prev + 1, thr)
                            above[g] = int(tl.t[jx]) if jx >= 0 else None
                    for d in MOVES:
                        ju = fp.first(i_prev + 1, fask * (1.0 + d / 10_000.0))
                        jd = tl.g[("sell", "down")].first(i_prev + 1, -fask * (1.0 - d / 10_000.0))
                        moves[f"t_up_{d}"] = int(tl.t[ju]) if ju >= 0 else None
                        moves[f"t_down_{d}"] = int(tl.t[jd]) if jd >= 0 else None
                ig = int(np.searchsorted(tl.tg, t0, side="right")) - 1
                jb = tl.next_bad[ig + 1] if ig + 1 < len(tl.tg) else len(tl.tg)
                t_gate = int(tl.tg[jb]) if jb < len(tl.tg) else None
                hedge = None
                if t_full:
                    hkey = (side, t_full)
                    if hkey in hedge_cache:
                        hedge = hedge_cache[hkey]
                    else:
                        hedge = hedge_cache[hkey] = sweep_hedge(p.fut, "bid" if side == "buy" else "ask", 1, t_full,
                                                                HEDGE_NS, HEDGE_TIMEOUT_NS, fut_lo, fut_hi, end)
                actual_ab = (hedge[1] / P - 1.0) * 10_000.0 if hedge else None
                # the user's makerFill at this spot tick, and the seconds for this exact level when it is L1/L2
                mf_fill_ns = None
                if mf is not None:
                    if lv_bid is None:
                        lv_bid = [px for px, _ in p.spot.levels(isp, "bid")]
                        lv_ask = [px for px, _ in p.spot.levels(isp, "ask")]
                    lv = lv_bid if side == "buy" else lv_ask
                    col = None
                    if level == 0:
                        col = 0 if side == "buy" else 2
                    elif level == 1 and len(lv) > 1 and lv[1] == P:
                        col = 1
                    if col is not None and np.isfinite(mf[col]):
                        mf_fill_ns = t0 + int(round(float(mf[col]) * SECOND))
                tb20 = below.get(20)
                rows.append(dict(
                    stream="S1", side=side, date=p.day, vc=p.vc, qc=p.qc, expiry=p.expiry, quote_ns=t0,
                    quote_second=int(sec[k]), price=P, level=level, event_kind=kind, anchor=an, scale=p.scale,
                    quote_ab=quote_ab, eff_u=eff, resid_mid_bp=float(p.resid_mid[sec[k]]),
                    e_norm=(float(p.resid_mid[sec[k]]) / p.scale) if p.scale else None,
                    spot_b1=b1, spot_a1=a1, fut_bid=fbid, fut_ask=fask,
                    fut_spread_bp=float((fask - fbid) / fbid * 10_000.0),
                    tick_bp_hedge=float(tick_i(opp) / opp * 10_000.0),
                    depth_ahead=int(ahead), opp_depth_lots=int(fbq[k] if side == "buy" else faq[k]),
                    notional_twd=P * p.shares / 10_000.0,
                    mf_bid1_s=float(mf[0]) if mf is not None else None, mf_bid2_s=float(mf[1]) if mf is not None else None,
                    mf_ask1_s=float(mf[2]) if mf is not None else None, mf_ask2_s=float(mf[3]) if mf is not None else None,
                    mf_fill_ns=mf_fill_ns,
                    t_partial_ns=t_partial, t_fill_ns=t_full, fill_print_seq=fill_seq, fill_price=fill_px,
                    fill_kind=(("through" if (fill_px < P if side == "buy" else fill_px > P) else "at_price") if t_full else None),
                    **{f"t_below_{f}": below.get(f) for f in FLOORS}, t_ab0_ns=below.get("ab0"),
                    **{f"t_above_{g}": above.get(g) for g in RISES}, t_gate_ns=t_gate, **moves,
                    hedge_ns=hedge[0] if hedge else None, hedge_vwap=int(hedge[1]) if hedge else None,
                    hedge_levels_swept=hedge[2] if hedge else None,
                    hedge_wait_ms=(hedge[0] - t_full) / 1e6 if hedge else None,
                    hedge_timeout=(hedge[3] if hedge else None) if t_full else None,
                    actual_ab=actual_ab, d_in_realized=(quote_ab - actual_ab) if hedge else None,
                    cancel_race_20=bool(side == "buy" and t_full and tb20 is not None and tb20 < t_full <= tb20 + CANCEL_NS),
                    negative_basis=bool(hedge and side == "buy" and actual_ab <= 0)))
    return rows


def load_makerfill(day: str, vcs: list[str], books: dict) -> dict[str, np.ndarray]:
    """Align the user's makerFill (per spot tick, keyed by ChannelSeq) to each product's book rows."""
    try:
        from ..common.books import _maker_paths
        mp = _maker_paths()
        path = mp.resolve_input_file(mp.MAKERFILL_ROOT / f"{day}_makerFill.parquet", role="makerfill")
    except Exception:
        return {}
    cols = ["Bid1_FillSeconds", "Bid2_FillSeconds", "Ask1_FillSeconds", "Ask2_FillSeconds"]
    frame = (pl.scan_parquet(path).filter(pl.col("QuoteCode").is_in(vcs))
             .select("QuoteCode", pl.col("ChannelSeq").cast(pl.Int64), *cols).collect())
    out = {}
    for g in frame.partition_by("QuoteCode", maintain_order=True):
        vc = g.item(0, "QuoteCode")
        book = books.get("S:" + vc)
        if book is None:
            continue
        seq = g["ChannelSeq"].to_numpy()
        order = np.argsort(seq)
        seq = seq[order]
        vals = g.select(cols).to_numpy().astype(np.float32)[order]
        pos = np.searchsorted(seq, book.seq)
        pos = np.clip(pos, 0, len(seq) - 1)
        hit = seq[pos] == book.seq
        aligned = np.full((len(book.seq), 4), np.nan, dtype=np.float32)
        aligned[hit] = vals[pos[hit]]
        out[vc] = aligned
    return out


def build_day(day: str, products: list[str] | None = None, *, write: bool = True) -> pl.DataFrame:
    started = time.time()
    grid = load_day(day, products)
    levels = qlevel.table(day)
    scale = {r["ValueCode"]: r["scale"] for r in levels.iter_rows(named=True)}
    meta = {r["ValueCode"]: r for r in pl.read_parquet(mapping_path(day)).iter_rows(named=True)}
    vcs = [vc for vc in grid if vc in meta]
    qcs = [grid[vc].qc for vc in vcs]
    books, prints, limits, inputs = load_session(day, vcs, qcs)
    makerfill = load_makerfill(day, vcs, books)
    frames, counts = [], {}
    for vc in vcs:
        g, m = grid[vc], meta[vc]
        spot, fut = books.get("S:" + vc), books.get("F:" + g.qc)
        if spot is None or fut is None or vc not in limits:
            counts[vc] = "missing_books"
            continue
        inp = ProductInputs(day, vc, g.qc, int(round(m["contract_size"])), g.expiry,
                            int(round(m["spot_ref_price"] * 10_000)), int(round(m["fut_ref_price"] * 10_000)),
                            spot, fut, prints.get("S:" + vc), g.anchor, g.mid - g.anchor, scale.get(vc), limits[vc],
                            makerfill.get(vc))
        rows = product_rows(inp)
        counts[vc] = len(rows)
        if rows:
            frames.append(pl.from_dicts(rows, schema=SCHEMA))
    frame = pl.concat(frames) if frames else pl.DataFrame(schema=SCHEMA)
    if write:
        path = points_path(day, "s1_entries")
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)
        (path.parent / "s1_manifest.json").write_text(json.dumps(dict(
            day=day, rows=frame.height, products=len(vcs), inputs=inputs, makerfill_products=len(makerfill),
            elapsed_s=round(time.time() - started, 1),
            params=dict(floors=FLOORS, rises=RISES, moves=MOVES, min_residual_bp=MIN_RESIDUAL_BP,
                        max_exit_residual_bp=MAX_EXIT_RESIDUAL_BP, ladder_below=LADDER_BELOW, board_lot=BOARD_LOT,
                        ahead_step=AHEAD_STEP, cancel_ms=CANCEL_NS / 1e6, hedge_ms=HEDGE_NS / 1e6,
                        hedge_timeout_s=HEDGE_TIMEOUT_NS / SECOND), counts=counts), indent=1) + "\n")
    return frame


def sequential_fills(frame: pl.DataFrame, *, residual_bp: float = 25.0, floor_bp: float = 20.0,
                     level_max: int = 0, cancel_ns: int = CANCEL_NS) -> pl.DataFrame:
    """Reference walker over buy rows: one live quote per product; take the highest-priced level
    (<= level_max ticks below B1) whose eff_u clears residual_bp; partials that never complete are rollbacks."""
    out = []
    floor_col = f"t_below_{int(floor_bp)}"
    buys = frame.filter(pl.col("side") == "buy")
    for vc, g in buys.sort(["vc", "quote_ns", "level"]).group_by("vc", maintain_order=True):
        vc = vc[0] if isinstance(vc, tuple) else vc
        by_t: dict[int, list] = {}
        for r in g.to_dicts():
            by_t.setdefault(r["quote_ns"], []).append(r)
        times = sorted(by_t)
        cursor = -1
        for i, t0 in enumerate(times):
            seg_end = times[i + 1] if i + 1 < len(times) else None
            if seg_end is not None and seg_end <= cursor:
                continue
            q_t = max(t0, cursor)
            cands = [r for r in by_t[t0] if r["level"] <= level_max and r["eff_u"] >= residual_bp]
            if not cands:
                continue
            r = min(cands, key=lambda x: x["level"])
            triggers = [v for v in (r[floor_col], r["t_gate_ns"], r["t_ab0_ns"]) if v is not None]
            if any(v <= q_t for v in triggers) or (r["t_fill_ns"] is not None and r["t_fill_ns"] <= q_t):
                continue
            cancel_at = min(triggers) if triggers else None
            deadline = cancel_at + cancel_ns if cancel_at is not None else None
            full = r["t_fill_ns"] is not None and (deadline is None or r["t_fill_ns"] <= deadline)
            partial = (not full and r["t_partial_ns"] is not None and r["t_partial_ns"] > q_t
                       and (deadline is None or r["t_partial_ns"] <= deadline))
            if full:
                out.append(dict(vc=vc, quote_ns=q_t, level=r["level"], price=r["price"], t_fill_ns=r["t_fill_ns"],
                                hedge_ns=r["hedge_ns"], quote_ab=r["quote_ab"], actual_ab=r["actual_ab"], eff_u=r["eff_u"],
                                depth_ahead=r["depth_ahead"], wait_s=(r["t_fill_ns"] - q_t) / 1e9, rollback=False,
                                mf_fill_ns=r["mf_fill_ns"],
                                cancel_race=(cancel_at is not None and r["t_fill_ns"] > cancel_at)))
                cursor = r["hedge_ns"] if r["hedge_ns"] is not None else r["t_fill_ns"] + HEDGE_TIMEOUT_NS
            elif partial:
                out.append(dict(vc=vc, quote_ns=q_t, level=r["level"], price=r["price"], t_fill_ns=r["t_partial_ns"],
                                hedge_ns=None, quote_ab=r["quote_ab"], actual_ab=None, eff_u=r["eff_u"],
                                depth_ahead=r["depth_ahead"], wait_s=(r["t_partial_ns"] - q_t) / 1e9, rollback=True,
                                mf_fill_ns=r["mf_fill_ns"], cancel_race=False))
                cursor = deadline + HEDGE_NS if deadline is not None else r["t_partial_ns"] + HEDGE_NS
            elif cancel_at is not None:
                cursor = cancel_at + cancel_ns
            else:
                break
    return pl.DataFrame(out) if out else pl.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day", required=True)
    parser.add_argument("--products")
    parser.add_argument("--walk", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    if args.day not in grid_days():
        raise SystemExit("day needs the canonical grid (anchor)")
    frame = build_day(args.day, args.products.split(",") if args.products else None, write=not args.no_write)
    summary = dict(day=args.day, rows=frame.height, products=frame["vc"].n_unique())
    for side in ("buy", "sell"):
        f = frame.filter(pl.col("side") == side)
        filled = f.filter(pl.col("t_fill_ns").is_not_null())
        summary[side] = dict(rows=f.height, with_full_fill=filled.height,
                             partial_only=f.filter(pl.col("t_partial_ns").is_not_null() & pl.col("t_fill_ns").is_null()).height,
                             by_level={str(k): v for k, v in sorted(f.group_by("level").len().rows())},
                             mf_rows=int(f["mf_fill_ns"].is_not_null().sum()),
                             hedge_timeouts=int(filled["hedge_timeout"].sum()) if filled.height else 0)
    print(json.dumps(summary))
    if args.walk:
        w = sequential_fills(frame)
        if w.height:
            full = w.filter(~pl.col("rollback"))
            both = full.filter(pl.col("mf_fill_ns").is_not_null())
            print(json.dumps(dict(walk_fills=full.height, rollbacks=int(w["rollback"].sum()), products=w["vc"].n_unique(),
                                  mean_quote_ab=round(float(full["quote_ab"].mean()), 2) if full.height else None,
                                  mean_actual_ab=round(float(full["actual_ab"].mean()), 2) if full.height else None,
                                  mean_decay=round(float((full["quote_ab"] - full["actual_ab"]).mean()), 2) if full.height else None,
                                  by_level={str(k): v for k, v in sorted(full.group_by("level").len().rows())} if full.height else {},
                                  wait_p50_s=round(float(full["wait_s"].median()), 1) if full.height else None,
                                  makerfill_vs_queue_p50_s=round(float(((both["mf_fill_ns"] - both["t_fill_ns"]) / 1e9).median()), 2) if both.height else None)))
        else:
            print(json.dumps(dict(walk_fills=0)))


if __name__ == "__main__":
    main()
