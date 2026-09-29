"""Read-only raw-tick checks for the four documented fixed audit sessions.

Checks recorded fills and hedge prices against raw prints and books; this is
not a shared-resource portfolio replay and does not certify queue priority.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl

from ..common.books import load_session
from ..common.paths import DATA_ROOT, grid_path
from .audit import load

FIXED_DAYS = ("20260511", "20260601", "20260706", "20260803")
NS_TOL = 128  # nullable NumPy float64 timestamps in the frozen exit replay


def requests_for_day(positions: dict, day: str) -> list[dict]:
    requests = []
    for name, pos in positions.items():
        for r in pos.filter((pl.col("quote_day") == day) | (pl.col("close_day") == day)).iter_rows(named=True):
            if r["quote_day"] == day:
                s1 = r["stream"] == "S1"
                requests.append(dict(run=name, id=r["id"], phase="entry", vc=r["vc"], qc=r["qc"],
                                     maker_spot=s1, maker_buy=s1, price=r["entry_price"], live=r["quote_ns"],
                                     fill=r["fill_ns"], hedge=r["hedge_ns"], shares=r["shares"],
                                     hedge_px=r["fut_sell_px"] if s1 else r["spot_buy_cash"] // r["shares"],
                                     pnl_net=r["pnl_net"]))
            if r["close_day"] == day and r["close_kind"] == "maker_exit":
                e1 = r["exit_route"] == "E1"
                requests.append(dict(run=name, id=r["id"], phase="exit", vc=r["vc"], qc=r["qc"],
                                     maker_spot=e1, maker_buy=not e1, price=r["exit_price"], live=r["exit_quote_ns"],
                                     fill=r["exit_fill_ns"], hedge=r["exit_hedge_ns"], shares=r["shares"],
                                     hedge_px=r["fut_buy_px"] if e1 else r["spot_sell_cash"] // r["shares"],
                                     pnl_net=r["pnl_net"]))
    return requests


def audit_day(positions: dict, day: str) -> tuple[dict, list[dict]]:
    requests = requests_for_day(positions, day)
    vcs = sorted({r["vc"] for r in requests})
    qcs = sorted({r["qc"] for r in requests})
    books, prints, _, paths = load_session(day, vcs, qcs)
    outcomes = []
    for r in requests:
        own = "S:" + r["vc"] if r["maker_spot"] else "F:" + r["qc"]
        hedge_key = "F:" + r["qc"] if r["maker_spot"] else "S:" + r["vc"]
        pr = prints.get(own)
        b = books.get(hedge_key)
        x = dict(day=day, run=r["run"], id=r["id"], phase=r["phase"],
                 maker_instrument=own, hedge_instrument=hedge_key, pnl_net=r["pnl_net"],
                 raw_print_match=False, hedge_price_match=False, post_live_print_volume=None,
                 insufficient_post_live_print_volume=False, hedge_snapshot_ns=None,
                 hedge_side="ask" if r["maker_buy"] is False else "bid",
                 s2_hold_a1_below_one_contract=False, s2_hold_a1_below_five_contracts=False)
        if pr is not None:
            lo = int(np.searchsorted(pr.ns, r["fill"] - NS_TOL, side="left"))
            hi = int(np.searchsorted(pr.ns, r["fill"] + NS_TOL, side="right"))
            px = pr.price[lo:hi]
            eligible = px <= r["price"] if r["maker_buy"] else px >= r["price"]
            x["raw_print_match"] = bool(np.any(eligible & (pr.qty[lo:hi] > 0)))
            if r["maker_spot"]:
                lo = int(np.searchsorted(pr.ns, r["live"] + NS_TOL, side="right"))
                eligible = pr.price[lo:hi] <= r["price"] if r["maker_buy"] else pr.price[lo:hi] >= r["price"]
                qty = int(pr.qty[lo:hi][eligible].sum())
                x["post_live_print_volume"] = qty
                x["insufficient_post_live_print_volume"] = qty < r["shares"]
        if b is not None:
            idx = int(np.searchsorted(b.ns, r["hedge"] + NS_TOL, side="right")) - 1
            if idx >= 0 and b.valid[idx]:
                # Independent L1-L5 summation; does not call BookSeries.take.
                qty = 1 if r["maker_spot"] else r["shares"]
                remaining, cash = qty, 0
                for px, depth in b.levels(idx, x["hedge_side"]):
                    taken = min(remaining, depth)
                    remaining -= taken
                    cash += taken * px
                    if remaining == 0:
                        break
                x["hedge_snapshot_ns"] = int(b.ns[idx])
                x["hedge_price_match"] = remaining == 0 and round(cash / qty) == r["hedge_px"]
        if r["phase"] == "entry" and not r["maker_spot"]:
            spot = books.get("S:" + r["vc"])
            if spot is not None:
                lo = max(0, int(np.searchsorted(spot.ns, r["live"], side="right")) - 1)
                hi = int(np.searchsorted(spot.ns, r["fill"] - 50_000_000 - NS_TOL, side="left"))
                if hi > lo:
                    depth = spot.top_ask_qty[lo:hi]
                    x["s2_hold_a1_below_one_contract"] = bool(np.any(depth < r["shares"]))
                    x["s2_hold_a1_below_five_contracts"] = bool(np.any(depth < 5 * r["shares"]))
        outcomes.append(x)
    grid = (pl.scan_parquet(grid_path(day)).filter(pl.col("ValueCode").is_in(vcs))
            .select(pl.len().alias("rows"),
                    (pl.col("spot_recv_time").dt.epoch("ns") > pl.col("timestamp").dt.epoch("ns")).sum().alias("future_spot_timestamps"),
                    (pl.col("fut_recv_time").dt.epoch("ns") > pl.col("timestamp").dt.epoch("ns")).sum().alias("future_fut_timestamps"))
            .collect().row(0, named=True))
    metadata = dict(day=day, products=len(vcs), recorded_legs=len(requests), raw_sources=paths, grid=grid)
    return metadata, outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", nargs="+", default=list(FIXED_DAYS))
    args = parser.parse_args()
    out = DATA_ROOT / "backtest" / "logic_audit_20260922"
    out.mkdir(parents=True, exist_ok=True)
    positions = {name: load(name)[0] for name in ("A_fixed", "B_dyn")}
    days, outcomes = [], []
    for day in args.days:
        meta, rows = audit_day(positions, day)
        days.append(meta)
        outcomes.extend(rows)
        print(json.dumps(meta), flush=True)
    frame = pl.from_dicts(outcomes, infer_schema_length=None)
    frame.write_csv(out / "raw_leg_checks.csv")
    summary = frame.group_by("run", "phase").agg(
        pl.len().alias("legs"), (~pl.col("raw_print_match")).sum().alias("missing_print"),
        (~pl.col("hedge_price_match")).sum().alias("hedge_mismatch"),
        pl.col("insufficient_post_live_print_volume").sum(),
        pl.col("s2_hold_a1_below_one_contract").sum(), pl.col("s2_hold_a1_below_five_contracts").sum()).sort("run", "phase")
    payload = dict(days=days, summary=summary.to_dicts(),
                   scope="All recorded A/B maker/hedge legs on the requested days; independent raw print and cash matching. Not shared queue/depth or full-tape opportunity validation.")
    (out / "raw_verification.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
