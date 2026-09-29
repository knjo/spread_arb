"""Tag the entry-leg fills of the slippage sample with the as-of EV decision of the v1 policy, so the screen can
run on the population the strategy would actually trade (not on every guarded fill).

    python -m spreadArb.src.slip.admit [--hurdle 8.5] [--force]
"""
from __future__ import annotations

import argparse
import json
import time

import polars as pl

from ..backtest.policy import Decider, PolicyConfig
from ..common.paths import DATA_ROOT, grid_days, points_path
from ..ev import abs_reach, reach
from .build import out_path

ENTRY = {"S1_buy": ("s1", "buy", "S1"), "S2_sell": ("s2", "sell", "S2")}


def admit_path(day: str):
    return DATA_ROOT / "slip" / f"Date={day}" / "admit.parquet"


def build_day(day: str, samples: pl.DataFrame, cfg: PolicyConfig, window: int = 20) -> dict:
    started = time.time()
    fills = pl.read_parquet(out_path(day)).filter(pl.col("leg").is_in(list(ENTRY)))
    reach_table = reach.fit(day, window)
    reach_table = reach_table if reach_table.days else None
    decider = Decider(day, reach_table, abs_reach.AbsTable.fit(samples, as_of=day), cfg)
    out = []
    for leg, (table, side, stream) in ENTRY.items():
        pt = pl.read_parquet(points_path(day, f"{table}_entries"), columns=(["vc", "quote_ns", "side", "expiry", "e_norm", "resid_mid_bp"]
                                                                             + (["level"] if table == "s1" else [])))
        pt = pt.filter(pl.col("side") == side)
        if table == "s1":
            pt = pt.filter(pl.col("level") == 0)
        pt = pt.select("vc", "quote_ns", "expiry", "e_norm", "resid_mid_bp").unique(subset=["vc", "quote_ns"], keep="first")
        f = fills.filter(pl.col("leg") == leg).select("leg", "vc", "quote_ns", "t_fill_ns", "quote_ab", "anchor", "scale", "quote_second",
                                                      "tick_bp_hedge").join(pt, on=["vc", "quote_ns"], how="left")
        for r in f.iter_rows(named=True):
            r["stream"] = stream
            if r["expiry"] is None:
                out.append(dict(leg=leg, vc=r["vc"], t_fill_ns=r["t_fill_ns"], ev_admit=False, ev_route=None, ev_bp=None, ev_score=None))
                continue
            d = decider.decide(r)
            b = d.best
            out.append(dict(leg=leg, vc=r["vc"], t_fill_ns=r["t_fill_ns"], ev_admit=bool(d.admit), ev_route=b.route if b else None,
                            ev_bp=float(b.ev_bp) if b else None, ev_score=float(b.score) if b else None))
    frame = pl.from_dicts(out, schema={"leg": pl.String, "vc": pl.String, "t_fill_ns": pl.Int64, "ev_admit": pl.Boolean,
                                       "ev_route": pl.String, "ev_bp": pl.Float64, "ev_score": pl.Float64})
    frame.write_parquet(admit_path(day))
    return dict(day=day, entry_fills=frame.height, admitted=int(frame["ev_admit"].sum()), ev_calls=decider.calls,
                elapsed_s=round(time.time() - started, 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hurdle", type=float, default=8.5)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = PolicyConfig(hurdle_bp_per_trading_day=args.hurdle)
    samples = abs_reach.load_samples()
    for day in grid_days():
        if not out_path(day).exists() or (admit_path(day).exists() and not args.force):
            continue
        print(json.dumps(build_day(day, samples, cfg)), flush=True)


if __name__ == "__main__":
    main()
