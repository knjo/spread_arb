"""EV surface for one decision day: reach cells with n/SE/CI, and EV/T/score per (e, x).

    uv run --project /home/kevin/Project/HFT --no-sync python -m spreadArb.src.ev.surface --day 20260706 [--coord residual_bp]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl

from ..common.grid import load_day
from ..common.paths import DATA_ROOT, grid_days
from . import abs_reach, qlevel, reach
from .config import CostConfig
from .ev import Quote, choose, evaluate, horizon


def illustrative_surface(table: reach.ReachTable, cfg: CostConfig, *, stream: str, anchor: float,
                         unit: float, t_sec: int, decision_day: str, expiry: str, e_points) -> pl.DataFrame:
    h = horizon(decision_day, expiry)
    rows = []
    for e in e_points:
        quote = Quote(stream, anchor + e * unit, anchor, unit, e, t_sec)
        for v in evaluate(quote, h, table, cfg, table.x_grid):
            rows.append(dict(e=e, quote_ab=quote.quote_ab, x=v.x if v.x is not None else "settle", b_x=v.b_x_bp,
                             c0=v.c0, c1=v.c1, q=v.q, p_sd=v.p_sd, p_on=v.p_on, p_never=v.p_never,
                             ev=v.ev_bp, t_days=v.t_days, score=v.score, surplus=v.surplus,
                             n_min=v.n_min, level=v.level_max, fallback=v.fallback))
    return pl.from_dicts(rows, infer_schema_length=None)


def product_decisions(table: reach.ReachTable, cfg: CostConfig, decision_day: str, t_sec: int,
                      stream: str, products: dict, levels: pl.DataFrame, coord: str,
                      lock_mode: str = "mid", abs_table=None) -> pl.DataFrame:
    scale = {r["ValueCode"]: r["scale"] for r in levels.iter_rows(named=True)}
    rows = []
    for vc, p in products.items():
        if vc not in scale or not p.eligible[t_sec] or not np.isfinite(p.mid[t_sec]) or not np.isfinite(p.anchor[t_sec]):
            continue
        anchor, resid = float(p.anchor[t_sec]), float(p.mid[t_sec] - p.anchor[t_sec])
        unit = scale[vc] if coord == "residual_scaled" else 1.0
        # S2 lock bounds until Stage 2 supplies books: mid basis (upper) / sell-taker basis (lower)
        lock = anchor + resid
        if lock_mode == "sell" and p.sell is not None and np.isfinite(p.sell[t_sec]):
            lock = float(p.sell[t_sec])
        quote = Quote(stream, lock, anchor, unit, resid / unit, t_sec)
        evals = evaluate(quote, horizon(decision_day, p.expiry), table, cfg, table.x_grid, abs_table=abs_table)
        d = choose(evals, cfg, quote)
        best = d.best
        abs_eval = next((v for v in evals if v.route == "absolute"), None)
        rows.append(dict(ValueCode=vc, expiry=p.expiry, anchor=round(anchor, 1), mid_ab=round(anchor + resid, 1),
                         lock_ab=round(lock, 1), scale=round(scale[vc], 1), e=round(quote.e, 2),
                         admit=d.admit, reason=d.reason, route=best.route if best else None,
                         x=(str(best.x) if best.x is not None else "settle") if best else None,
                         abs_score=round(abs_eval.score, 2) if abs_eval else None,
                         ev=round(best.ev_bp, 1) if best else None, t_days=round(best.t_days, 2) if best else None,
                         score=round(best.score, 2) if best else None, p_sd=round(best.p_sd, 2) if best else None,
                         n_min=best.n_min if best else None))
    return pl.from_dicts(rows, infer_schema_length=None).sort("score", descending=True, nulls_last=True)


def pivot(frame: pl.DataFrame, index: str, on: str, value: str, count: str | None = "n") -> pl.DataFrame:
    f = frame.with_columns(pl.col(on).cast(pl.String))
    if count:
        f = f.with_columns((pl.col(value).round(3).cast(pl.String) + " (" + pl.col(count).cast(pl.String) + ")").alias("cell"))
    else:
        f = f.with_columns(pl.col(value).round(2).cast(pl.String).alias("cell"))
    return f.pivot(on=on, index=index, values="cell", aggregate_function="first").sort(index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day", required=True)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--stream", default="S2")
    parser.add_argument("--t", type=int, default=1800)
    parser.add_argument("--coord", default="residual_scaled", choices=list(reach.COORDS))
    parser.add_argument("--anchor", type=float, default=30.0)
    parser.add_argument("--min-days", type=int, default=5)
    parser.add_argument("--full", action="store_true",
                        help="pool every cached session (structure study; not causal for the product example)")
    args = parser.parse_args()
    if args.day not in grid_days():
        raise SystemExit("decision day needs a grid for the per-product example")
    cfg = CostConfig()
    table = reach.fit(None if args.full else args.day, args.window, coord=args.coord, min_days=args.min_days)
    try:
        abs_table = abs_reach.AbsTable.fit(abs_reach.load_samples(), as_of=None if args.full else args.day)
    except FileNotFoundError:
        abs_table = None
    out = DATA_ROOT / "ev_surface" / f"Date={args.day}" / (f"{args.coord}_full" if args.full else f"{args.coord}_w{args.window}")
    out.mkdir(parents=True, exist_ok=True)
    frames = table.frames()
    for name, f in frames.items():
        f.write_csv(out / f"{name}.csv")
    levels = qlevel.table(args.day).filter(pl.col("scale").is_not_null())
    median_scale = float(levels["scale"].median())
    products = load_day(args.day)
    expiries = sorted(p.expiry for p in products.values())
    expiry_example = expiries[len(expiries) // 2]
    unit = median_scale if args.coord == "residual_scaled" else 1.0
    e_points = (0.75, 1.25, 2.0, 3.0) if args.coord == "residual_scaled" else (10.0, 20.0, 35.0, 60.0)
    surf = illustrative_surface(table, cfg, stream=args.stream, anchor=args.anchor, unit=unit, t_sec=args.t,
                                decision_day=args.day, expiry=expiry_example, e_points=e_points)
    surf.write_csv(out / "surface.csv")
    prods = product_decisions(table, cfg, args.day, args.t, args.stream, products, levels, args.coord, "mid", abs_table)
    prods.write_csv(out / "products.csv")
    prods_sell = product_decisions(table, cfg, args.day, args.t, args.stream, products, levels, args.coord, "sell", abs_table)
    prods_sell.write_csv(out / "products_sell_lock.csv")
    if abs_table is not None:
        abs_table.frame().write_csv(out / "abs_table.csv")
    h = horizon(args.day, expiry_example)
    t_b, k_b = int(reach.bucket([args.t], reach.T_EDGES)[0]), int(reach.k_bucket([h.k_days], table.k_edges)[0])
    (out / "meta.json").write_text(json.dumps(dict(
        day=args.day, window=args.window, train_days=table.days, coord=table.coord, x_grid=table.x_grid,
        e_edges=table.e_edges, k_edges=table.k_edges, min_n=table.min_n, min_days=table.min_days,
        median_scale=median_scale, full=args.full,
        expiry_example=expiry_example,
        t=args.t, t_b=t_b, k_b=k_b, stream=args.stream, anchor=args.anchor,
        cfg=dict(fee_sd=cfg.fee_same_day_bp, fee_on=cfg.fee_overnight_bp, ev_min=cfg.ev_min_bp,
                 hurdle=cfg.hurdle_bp_per_day, d_in=dict(cfg.d_in_base), d_settle=cfg.d_settle_bp(),
                 margin=cfg.margin_bp, min_anchor=cfg.min_anchor_bp, max_scale=qlevel.MAX_SCALE_BP)), indent=1) + "\n")
    with pl.Config(tbl_rows=60, tbl_cols=20, fmt_str_lengths=30, tbl_width_chars=220, tbl_hide_dataframe_shape=True,
                   tbl_hide_column_data_types=True):
        print(f"[{args.coord}{' FULL' if args.full else ''}] sessions {table.days[0]}..{table.days[-1]} ({len(table.days)} d); "
              f"products with scale {levels.height}; median scale {median_scale:.1f} bp; example expiry {expiry_example} "
              f"(K={h.K}, settle +{h.settle_offset}d, k_b={k_b}); t={args.t} (t_b={t_b}); e edges {table.e_edges}; "
              f"k edges {table.k_edges}")
        k_labels = {i: f"{lo}-{hi}" for i, (lo, hi) in enumerate(zip((0,) + tuple(e + 1 for e in table.k_edges),
                                                                       table.k_edges + ("inf",)))}
        print("\nk profile — rows k bucket (calendar days to expiry), fixed t_b; for each e_b: C0 / C1 / q at x=0 and x=x_min")
        x_min = min(table.x_grid)
        for e_b in (1, 2, 3):
            rows = []
            for kb in range(len(table.k_edges) + 1):
                def cell(tab, names, vals):
                    c = tab.get((names, vals))
                    return (f"{c.p:.2f}({c.n_days})" if c else "-")
                rows.append(dict(k=k_labels[kb],
                                 c0_x0=cell(table.c0, ("e_b", "x", "t_b", "k_b"), (e_b, 0.0, t_b, kb)),
                                 c0_xmin=cell(table.c0, ("e_b", "x", "t_b", "k_b"), (e_b, x_min, t_b, kb)),
                                 c1_x0=cell(table.c1, ("e_b", "x", "k_b"), (e_b, 0.0, kb)),
                                 c1_xmin=cell(table.c1, ("e_b", "x", "k_b"), (e_b, x_min, kb)),
                                 q_x0=cell(table.q, ("e_b", "x", "k_b"), (e_b, 0.0, kb)),
                                 q_xmin=cell(table.q, ("e_b", "x", "k_b"), (e_b, x_min, kb))))
            print(f"e_b={e_b}  (p(n_days))")
            print(pl.DataFrame(rows))
        c0 = frames["c0"].filter((pl.col("level") == 0) & (pl.col("t_b") == t_b) & (pl.col("k_b") == k_b))
        print(f"\nC0  P(reach x today | e bucket)  rows e_b, cols x  — cell = p (n); days per cell: "
              f"{sorted(set(c0['n_days'].to_list()))}")
        print(pivot(c0, "e_b", "x", "p"))
        print("C0 day-block 95% CI half-width (rows e_b, cols x)")
        print(pivot(c0.with_columns(((pl.col("ci_hi") - pl.col("ci_lo")) / 2).alias("hw")), "e_b", "x", "hw", count=None))
        c1 = frames["c1"].filter((pl.col("level") == 0) & (pl.col("k_b") == k_b))
        print(f"\nC1  P(reach x by tomorrow | e bucket)")
        print(pivot(c1, "e_b", "x", "p"))
        q = frames["q"].filter((pl.col("level") == 0) & (pl.col("k_b") == k_b)).sort(["e_b", "x"])
        print(f"\nq  per-session reach rate for sessions after tomorrow (rows e_b, cols x; k_b={k_b})")
        print(pivot(q, "e_b", "x", "p"))
        qx = frames["q"].filter(pl.col("level") == 1).sort(["k_b", "x"])   # (x, k_b): pooled over e
        print("q pooled over e (rows k_b; n_days in brackets)")
        print(pivot(qx.with_columns(pl.col("n_days").alias("n")), "k_b", "x", "p"))
        print(f"\nEV surface — {args.stream}, anchor {args.anchor}, unit {unit:.1f} bp, expiry {expiry_example}; cell = score bp/day")
        print(pivot(surf, "e", "x", "score", count=None))
        print("cell = EV bp")
        print(pivot(surf, "e", "x", "ev", count=None))
        print("cell = expected days")
        print(pivot(surf, "e", "x", "t_days", count=None))
        if abs_table is not None:
            af = abs_table.frame().filter(pl.col("level") == 0).sort(["thr_bp", "k_b"])
            print(f"\nabsolute convergence table (taker first-crossing samples{' as of ' + args.day if not args.full else ', full'}): "
                  f"rows thr_bp x k_b (0 / 1-2 / 3-5 / 6-10 / 11-15 / 16+ td); P first convergence day 0/1/2/3-5/6-15, later, settle")
            print(af.select("thr_bp", "k_b", "n", pl.col("p0").round(2), pl.col("p1").round(2), pl.col("p2").round(2),
                            pl.col("p3_5").round(2), pl.col("p6_15").round(2), pl.col("p_later").round(2),
                            pl.col("p_settle").round(2), pl.col("mean_cal_days").round(1)))
        print(f"\nper-product at t={args.t} on {args.day}: lock_ab = mid basis (upper bound of an S2 lock); top 15 by score")
        print(prods.head(15))
        print("reasons (mid lock):", {r["reason"]: r["len"] for r in prods.group_by("reason").len().sort("reason").to_dicts()})
        print("reasons (sell-taker lock):", {r["reason"]: r["len"] for r in prods_sell.group_by("reason").len().sort("reason").to_dicts()})
        adm = prods.filter(pl.col("admit"))
        if adm.height:
            print("admitted route/x:", {f"{r['route']}:{r['x']}": r["len"] for r in adm.group_by("route", "x").len().to_dicts()},
                  f"| mean score {adm['score'].mean():.2f}, mean ev {adm['ev'].mean():.1f}")


if __name__ == "__main__":
    main()
