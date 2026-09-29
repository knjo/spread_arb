"""Slippage factor sample: every maker fill that the v1 guards would let through, with the state of both
books and the user's spot tickFeature 50 ms BEFORE the fill (the last moment a cancel could still work).

Four legs (maker leg -> hedge leg):
    S1_buy   spot bid maker    -> sell futures bid   (entry)     adverse = prices falling
    S1_sell  spot ask maker    -> buy futures ask    (exit E1)   adverse = prices rising
    S2_sell  futures ask maker -> buy spot ask       (entry)     adverse = prices rising
    S2_buy   futures bid maker -> sell spot bid      (exit E2)   adverse = prices falling

slip_bp  = adverse basis change from the quote to the hedge (entry: quote_ab - actual_ab; exit: the reverse)
jump_bp  = hedge execution vs the hedge-side L1 at the snapshot (what moved in the last ~100 ms)
drift_bp = slip_bp - jump_bp (what we let drift inside the guard while the quote was resting)

    python -m spreadArb.src.slip.build --start 20260126 --end 20260813 [--force]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import polars as pl

from ..common.books import INF, load_session, tick_i
from ..common.paths import DATA_ROOT, SECOND, grid_days, open_ns, points_path

MS = 1_000_000
PLACE_NS = CANCEL_NS = SNAP_NS = 50 * MS
LEGS = {
    "S1_buy": dict(table="s1", side="buy", maker="S", maker_side="bid", hedge="F", hedge_side="bid", adverse=-1, exit=False),
    "S1_sell": dict(table="s1", side="sell", maker="S", maker_side="ask", hedge="F", hedge_side="ask", adverse=1, exit=True),
    "S2_sell": dict(table="s2", side="sell", maker="F", maker_side="ask", hedge="S", hedge_side="ask", adverse=1, exit=False),
    "S2_buy": dict(table="s2", side="buy", maker="F", maker_side="bid", hedge="S", hedge_side="bid", adverse=-1, exit=True),
}
LABEL_PREFIXES = ("FutureAsk1_", "FutureBid1_", "TakerSell_CloseBP", "TakerBuy_CloseBP", "midEdge_")
TICK = np.vectorize(tick_i, otypes=[np.int64])


def out_path(day: str):
    return DATA_ROOT / "slip" / f"Date={day}" / "fills.parquet"


def feature_path(day: str):
    from ..common.books import raw_paths
    return raw_paths(day)[0].parent.parent / "tickFeature" / f"{day}_tickFeature.parquet"


def select_fills(day: str) -> pl.DataFrame:
    frames = []
    for leg, m in LEGS.items():
        f = pl.read_parquet(points_path(day, f"{m['table']}_entries")).filter(pl.col("side") == m["side"])
        if m["table"] == "s1":
            f = f.filter(pl.col("level") == 0).with_columns(pl.col("opp_depth_lots").cast(pl.Float64).alias("opp_depth"))
        else:
            f = f.with_columns((pl.col("opp_depth_shares") / 1000.0).alias("opp_depth"))     # spot lots
        if m["exit"]:
            guard = pl.min_horizontal("t_above_5", "t_gate_ns")
        else:
            f = f.filter((pl.col("eff_u") >= 25.0) | (pl.col("quote_ab") >= 50.0))
            guard = pl.min_horizontal("t_below_20", "t_gate_ns", "t_ab0_ns")
        live = pl.col("quote_ns") + PLACE_NS
        f = f.with_columns(guard.alias("guard_ns"))
        f = f.filter(pl.col("t_fill_ns").is_not_null() & pl.col("hedge_ns").is_not_null() & (pl.col("t_fill_ns") > live)
                     & (pl.col("guard_ns").is_null() | ((pl.col("guard_ns") > live) & (pl.col("t_fill_ns") <= pl.col("guard_ns") + CANCEL_NS))))
        sign = -1.0 if m["exit"] else 1.0
        f = f.sort("quote_ns").unique(subset=["vc", "t_fill_ns"], keep="first", maintain_order=True)
        frames.append(f.select(
            pl.lit(leg).alias("leg"), pl.lit(day).alias("day"), "vc", "qc", "quote_ns", "quote_second", "price", "anchor", "scale",
            "quote_ab", "eff_u", "tick_bp_hedge", "fut_spread_bp", pl.col("depth_ahead").cast(pl.Float64), "opp_depth",
            "t_fill_ns", "fill_kind", "guard_ns", "hedge_ns", "hedge_vwap", "hedge_levels_swept", "hedge_timeout", "actual_ab",
            (sign * (pl.col("quote_ab") - pl.col("actual_ab"))).alias("slip_bp"),
            (pl.col("guard_ns").is_not_null() & (pl.col("t_fill_ns") > pl.col("guard_ns"))).alias("race"),
            ((pl.col("t_fill_ns") - pl.col("quote_ns")) / 1e9).alias("wait_s")))
    return pl.concat(frames, how="vertical_relaxed")


def _second_level(px: np.ndarray, qty: np.ndarray, top: np.ndarray, side: str) -> np.ndarray:
    """Second displayed price behind the merged L1 (0 / INF when absent)."""
    if side == "bid":
        v = np.where((qty > 0) & (px < top[:, None]), px, 0)
        return v.max(axis=1)
    v = np.where((qty > 0) & (px > top[:, None]), px, INF)
    return v.min(axis=1)


def _book_state(book, t: np.ndarray, near: str, adverse: int, prefix: str) -> dict[str, np.ndarray]:
    i = np.searchsorted(book.ns, t, side="right") - 1
    ok = i >= 0
    i = np.clip(i, 0, len(book.ns) - 1)
    bid, ask, bq, aq = book.top_bid_px[i], book.top_ask_px[i], book.top_bid_qty[i], book.top_ask_qty[i]
    valid = ok & book.valid[i]
    mid = (bid + ask) / 2.0
    tick = TICK(np.where(bid > 0, bid, 1))
    b2 = _second_level(book.bid_px[i], book.bid_qty[i], bid, "bid")
    a2 = _second_level(book.ask_px[i], book.ask_qty[i], ask, "ask")
    near_px, near_q, far_q = (bid, bq, aq) if near == "bid" else (ask, aq, bq)
    gap = np.where(b2 > 0, (bid - b2) / tick, np.nan) if near == "bid" else np.where(a2 < INF, (a2 - ask) / tick, np.nan)
    near5 = (book.bid_qty[i] if near == "bid" else book.ask_qty[i]).sum(axis=1)
    out = {f"{prefix}_valid": valid, f"{prefix}_near_px": near_px.astype(np.float64), f"{prefix}_spread_ticks": (ask - bid) / tick,
           f"{prefix}_near_qty": near_q.astype(np.float64), f"{prefix}_far_qty": far_q.astype(np.float64),
           f"{prefix}_near_share": near_q / np.maximum(near_q + far_q, 1), f"{prefix}_gap12_ticks": gap,
           f"{prefix}_near5_qty": near5.astype(np.float64), f"{prefix}_seq": book.seq[i],
           f"{prefix}_stale_ms": (t - book.ns[i]) / 1e6}
    for label, back in (("1s", SECOND), ("5s", 5 * SECOND), ("30s", 30 * SECOND)):
        j = np.clip(np.searchsorted(book.ns, t - back, side="right") - 1, 0, len(book.ns) - 1)
        mid_then = (book.top_bid_px[j] + book.top_ask_px[j]) / 2.0
        good = valid & book.valid[j]
        out[f"{prefix}_mom_{label}_bp"] = np.where(good, adverse * (mid - mid_then) / np.where(mid_then > 0, mid_then, 1) * 1e4, np.nan)
        if label != "30s":
            out[f"{prefix}_updates_{label}"] = (i - j).astype(np.float64)
    return out


def _print_counts(pr, t: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
    if pr is None:
        return {f"{prefix}_prints_1s": np.zeros(len(t)), f"{prefix}_prints_5s": np.zeros(len(t))}
    hi = np.searchsorted(pr.ns, t, side="right")
    return {f"{prefix}_prints_{lab}": (hi - np.searchsorted(pr.ns, t - back, side="right")).astype(np.float64)
            for lab, back in (("1s", SECOND), ("5s", 5 * SECOND))}


def build_day(day: str) -> dict:
    started = time.time()
    fills = select_fills(day)
    pairs = fills.select("vc", "qc").unique()
    books, prints, _, _ = load_session(day, pairs["vc"].to_list(), pairs["qc"].to_list())
    t_load = time.time() - started
    parts = []
    for (leg, vc, qc), g in fills.group_by(["leg", "vc", "qc"], maintain_order=True):
        m = LEGS[leg]
        keys = {"S": f"S:{vc}", "F": f"F:{qc}"}
        mk, hk = keys[m["maker"]], keys[m["hedge"]]
        if mk not in books or hk not in books:
            continue
        t = g["t_fill_ns"].to_numpy() - SNAP_NS
        cols = {}
        cols.update(_book_state(books[mk], t, m["maker_side"], m["adverse"], "maker"))
        cols.update(_book_state(books[hk], t, m["hedge_side"], m["adverse"], "hedge"))
        cols.update(_print_counts(prints.get(mk), t, "maker"))
        cols.update(_print_counts(prints.get(hk), t, "hedge"))
        spot_seq = cols["maker_seq"] if m["maker"] == "S" else cols["hedge_seq"]
        hv = g["hedge_vwap"].to_numpy().astype(np.float64)
        near = cols["hedge_near_px"]
        jump = m["adverse"] * (hv - near) / np.where(near > 0, near, 1) * 1e4
        parts.append(g.with_columns([pl.Series(k, v) for k, v in cols.items()]
                                    + [pl.Series("spot_seq", spot_seq.astype(np.int64)), pl.Series("jump_bp", jump)]))
    out = pl.concat(parts, how="vertical_relaxed").with_columns((pl.col("slip_bp") - pl.col("jump_bp")).alias("drift_bp"))
    # the user's per-spot-tick features at the snapshot tick (as-of on ChannelSeq within the product)
    fp = feature_path(day)
    schema = pl.scan_parquet(fp).collect_schema()
    feats = [c for c in schema if c not in ("QuoteCode", "ChannelSeq") and not c.startswith(LABEL_PREFIXES)]
    tf = (pl.scan_parquet(fp).filter(pl.col("QuoteCode").is_in(pairs["vc"].to_list()))
          .select(pl.col("QuoteCode").alias("vc"), pl.col("ChannelSeq").cast(pl.Int64).alias("feat_seq"), *feats)
          .sort(["vc", "feat_seq"]).collect())
    out = (out.sort(["vc", "spot_seq"])
           .join_asof(tf, left_on="spot_seq", right_on="feat_seq", by="vc", strategy="backward")
           .with_columns((pl.col("spot_seq") - pl.col("feat_seq")).alias("feat_lag_ticks")))
    path = out_path(day)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(path)
    info = dict(day=day, fills=out.height, by_leg={k: int(v) for k, v in out.group_by("leg").len().iter_rows()},
                feat_exact=float((out["feat_lag_ticks"] == 0).mean()), load_s=round(t_load, 1), elapsed_s=round(time.time() - started, 1))
    (path.parent / "manifest.json").write_text(json.dumps(info) + "\n")
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--shard", default="0/1", help="i/n: build only days with index % n == i")
    args = ap.parse_args()
    i, n = (int(x) for x in args.shard.split("/"))
    days = [d for d in grid_days() if (not args.start or d >= args.start) and (not args.end or d <= args.end)]
    for k, day in enumerate(days):
        if k % n != i or (out_path(day).exists() and not args.force):
            continue
        try:
            print(json.dumps(build_day(day)), flush=True)
        except Exception as exc:  # keep the batch going; the day is reported
            print(json.dumps(dict(day=day, error=repr(exc))), flush=True)


if __name__ == "__main__":
    main()
