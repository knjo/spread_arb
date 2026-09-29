"""Replay maker's S0 q95 spot-bid commands through the spreadArb spot queue model and compare fills
with the maker cost_guard shadow (same quotes, two engines).

    uv run --project /home/kevin/Project/HFT --no-sync python -m spreadArb.src.points.compare_maker_s1 --day 20260706
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from ..common.books import load_session
from ..common.grid import load_day
from ..common.paths import MAKER_ROOT, QUOTE_END_SECOND, SECOND, open_ns
from .s1 import BOARD_LOT, CANCEL_NS, displayed_at

COMMANDS = MAKER_ROOT / "data/walkforward/august_attribution_s0_20260824_v2/raw_order_facts.parquet"
SHADOW = str(MAKER_ROOT / "data/ev_lookup_cost_full_20260909/guard/Date={day}/shadow/positions.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day", required=True)
    args = parser.parse_args()
    day = args.day
    start = open_ns(day)
    cutoff = start + QUOTE_END_SECOND * SECOND
    cmd = (pl.scan_parquet(COMMANDS).filter(pl.col("Date") == day)
           .select("raw_order_fact_id", "ValueCode", "QuoteCode", "target_price", "nominal_new_time_ns", "nominal_stop_time_ns")
           .collect())
    shadow = pl.read_parquet(SHADOW.format(day=day)).filter(pl.col("stream") == "S1")
    maker_fill = {r["source_intent"]: r["entry_fill_ns"] for r in shadow.iter_rows(named=True) if r["entry_fill_ns"] is not None}
    maker_quoted = set(shadow["source_intent"].to_list())
    grid = load_day(day)
    vcs = sorted(set(cmd["ValueCode"].to_list()) & set(grid))
    books, prints, limits, _ = load_session(day, vcs, [grid[v].qc for v in vcs])
    rows = []
    for r in cmd.iter_rows(named=True):
        vc = r["ValueCode"]
        spot, pr = books.get("S:" + vc), prints.get("S:" + vc)
        t0, stop = r["nominal_new_time_ns"], min(r["nominal_stop_time_ns"], cutoff)
        if spot is None or t0 is None or stop is None or stop <= t0 or not start <= t0 < cutoff:
            continue
        P = int(round(r["target_price"] * 10_000))
        i = spot.index_at(t0)
        legal = i >= 0 and spot.valid[i] and P < spot.top_ask_px[i]
        ahead = displayed_at(spot, i, P) if legal else None
        t_full = None
        if legal and pr is not None:
            m = pr.price <= P
            ns_p, cum = pr.ns[m], np.cumsum(pr.qty[m])
            j0 = int(np.searchsorted(ns_p, t0, side="right"))
            if j0 < len(ns_p):
                base = cum[j0 - 1] if j0 > 0 else 0
                jf = int(np.searchsorted(cum, base + ahead + 2 * BOARD_LOT, side="left"))
                if jf < len(ns_p):
                    t_full = int(ns_p[jf])
        ours_filled = t_full is not None and t_full <= stop + CANCEL_NS
        rows.append(dict(id=r["raw_order_fact_id"], vc=vc, price=P, t0=t0, stop=stop, legal=legal, ahead=ahead,
                         ours_fill_ns=t_full if ours_filled else None,
                         maker_quoted=r["raw_order_fact_id"] in maker_quoted,
                         maker_fill_ns=maker_fill.get(r["raw_order_fact_id"])))
    f = pl.DataFrame(rows)
    both = f.filter(pl.col("ours_fill_ns").is_not_null() & pl.col("maker_fill_ns").is_not_null())
    dt = (both["ours_fill_ns"] - both["maker_fill_ns"]) / 1e9 if both.height else pl.Series([])
    with pl.Config(tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True):
        print(f"day {day}: commands {f.height} (maker quoted {int(f['maker_quoted'].sum())}, legal for us {int(f['legal'].sum())})")
        print(f"fills: ours {int(f['ours_fill_ns'].is_not_null().sum())}, maker {int(f['maker_fill_ns'].is_not_null().sum())}, "
              f"both {both.height}, ours-only {int((f['ours_fill_ns'].is_not_null() & f['maker_fill_ns'].is_null()).sum())}, "
              f"maker-only {int((f['ours_fill_ns'].is_null() & f['maker_fill_ns'].is_not_null()).sum())}")
        if both.height:
            print(f"fill-time difference ours-maker (s): p10 {dt.quantile(0.1):.3f} p50 {dt.quantile(0.5):.3f} p90 {dt.quantile(0.9):.3f}; "
                  f"|dt|<=1s share {(dt.abs() <= 1).mean():.3f}")
        mo = f.filter(pl.col("ours_fill_ns").is_null() & pl.col("maker_fill_ns").is_not_null())
        if mo.height:
            print("maker-only examples:")
            print(mo.select("vc", "price", "t0", "stop", "legal", "ahead", "maker_fill_ns").head(8))
        oo = f.filter(pl.col("ours_fill_ns").is_not_null() & pl.col("maker_fill_ns").is_null())
        if oo.height:
            print("ours-only examples:")
            print(oo.select("vc", "price", "t0", "stop", "legal", "ahead", "ours_fill_ns", "maker_quoted").head(8))


if __name__ == "__main__":
    main()
