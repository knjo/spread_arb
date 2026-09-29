"""Screen "when not to quote" conditions against the slippage sample built by slip.build (+ slip.admit).

Populations:
    all     every fill the v1 guards let through
    policy  entry legs: fills of quotes the as-of EV admits; exit legs: quotes at or below the live anchor (eff_u <= 0)
    v1      the fills the v1 replay actually took (small, used as a cross-check of rules found on `policy`)

For every feature: choose the worse tail (low or high, `--tail` of the sample) on TRAIN days (even index), then on
TEST days (odd index) report what cancelling in that state would have removed. A greedy pass stacks up to three
rules. Named conditions get full bucket tables.

    python -m spreadArb.src.slip.screen [--leg S2_sell] [--tail 0.2] [--population policy] [--v1 v1]
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from ..common.paths import DATA_ROOT
from .build import LEGS

META = {"leg", "day", "vc", "qc", "quote_ns", "price", "t_fill_ns", "fill_kind", "guard_ns", "hedge_ns", "hedge_vwap",
        "hedge_levels_swept", "hedge_timeout", "actual_ab", "slip_bp", "race", "jump_bp", "drift_bp", "maker_valid",
        "hedge_valid", "maker_near_px", "hedge_near_px", "maker_seq", "hedge_seq", "spot_seq", "feat_seq", "feat_lag_ticks",
        "SpreadPairID", "SpreadPairSeq", "MidPrice", "MicroPrice", "SpreadPairAsk", "SpreadPairBid", "TickSize",
        "ev_admit", "ev_route", "ev_bp", "ev_score", "v1_slip",
        # quote-level covariates: they belong in the cost model (d_in as a function of them), not in a cancel rule
        "eff_u", "quote_ab", "anchor", "scale", "quote_second",
        # leaky in the source definitions (broadcast / backward-filled from later rows): never screen these
        "SpreadNarrowOrderTime", "SpreadNarrowSide"}
NAMED = ["hedge_gap12_ticks", "hedge_spread_ticks", "maker_spread_ticks", "maker_near_qty", "maker_far_qty", "maker_near_share",
         "hedge_near_qty", "hedge_near_share", "depth_ahead", "opp_depth", "MD_L1Rate_10", "MD_L1Rate_30", "MD_L1Rate_100",
         "B1_A1B1", "B1_B1B5", "B12_B1B5", "A1_A1A5", "A12_A1A5", "tick_bp_hedge", "fut_spread_bp", "hedge_mom_1s_bp",
         "maker_mom_1s_bp", "wait_s", "eff_u", "quote_ab"]


def load_leg(leg: str, population: str, v1_run: str = "v1") -> pl.DataFrame:
    root = DATA_ROOT / "slip"
    f = pl.scan_parquet(str(root / "Date=*" / "fills.parquet")).filter(pl.col("leg") == leg)
    f = f.filter(pl.col("maker_valid") & pl.col("hedge_valid") & ~pl.col("hedge_timeout").fill_null(False))
    exit_leg = LEGS[leg]["exit"]
    if population in ("policy", "episode"):
        if exit_leg:
            f = f.filter(pl.col("eff_u") <= 0.0)
        else:
            adm = pl.scan_parquet(str(root / "Date=*" / "admit.parquet")).filter(pl.col("leg") == leg)
            f = f.join(adm, on=["leg", "vc", "t_fill_ns"], how="inner").filter(pl.col("ev_admit"))
            if leg == "S2_sell":
                f = f.filter(pl.col("opp_depth") >= 10.0)      # v1: spot A1 >= 5 x 2000 shares
    d = f.collect()
    if population == "episode":
        # one position per product: the strategy takes the first admissible fill of a product, not every fill event
        d = d.sort("t_fill_ns").unique(subset=["vc", "day"], keep="first", maintain_order=True)
    if population == "v1":
        pos = pl.read_parquet(DATA_ROOT / "backtest" / v1_run / "positions.parquet")
        if leg in ("S1_buy", "S2_sell"):
            k = pos.filter(pl.col("stream") == leg[:2]).select("vc", pl.col("fill_ns").alias("t_fill_ns"),
                                                                (pl.col("quote_ab") - pl.col("actual_ab")).alias("v1_slip"))
        elif leg == "S1_sell":
            k = pos.filter(pl.col("close_kind") == "maker_exit").select("vc", pl.col("exit_fill_ns").alias("t_fill_ns"),
                                                                        (pl.col("exit_realized_ab") - pl.col("exit_quote_ab")).alias("v1_slip"))
        else:
            return d.clear()
        d = d.join(k.unique(subset=["vc", "t_fill_ns"]), on=["vc", "t_fill_ns"], how="inner").with_columns(pl.col("v1_slip").alias("slip_bp"))
    return d


def base_line(d: pl.DataFrame) -> str:
    if not d.height:
        return "n 0"
    s = d["slip_bp"]
    ticks = (d["slip_bp"] / d["tick_bp_hedge"])
    return (f"n {d.height:,} | slip mean {s.mean():.1f} med {s.median():.1f} p90 {s.quantile(.9):.1f} | in hedge ticks mean {ticks.mean():.2f} | "
            f"zero-slip {float((s <= 0.5).mean()):.2f} | >=30bp {float((s >= 30).mean()):.2f} | race {float(d['race'].mean()):.2f} | "
            f"jump {d['jump_bp'].mean():.1f} drift {d['drift_bp'].mean():.1f}")


def bucket_table(d: pl.DataFrame, col: str, q: int = 5) -> str:
    x = d[col].cast(pl.Float64).to_numpy()
    ok = np.isfinite(x)
    x, slip, race, jump = x[ok], d["slip_bp"].to_numpy()[ok], d["race"].to_numpy()[ok], d["jump_bp"].to_numpy()[ok]
    if x.size < 100:
        return f"  {col}: too few"
    uniq = np.unique(x)
    if col.endswith("_ticks"):
        edges = [1.5, 2.5, 4.5]
    else:
        edges = list((uniq[:-1] + uniq[1:]) / 2) if len(uniq) <= 6 else sorted(set(np.quantile(x, np.linspace(0, 1, q + 1)[1:-1]).tolist()))
    b = np.searchsorted(np.asarray(edges), x, side="right")
    lines = [f"  {col}  (n {ok.sum():,}, missing {1 - ok.mean():.1%})"]
    lo, hi = [-np.inf] + list(edges), list(edges) + [np.inf]
    for k in range(len(edges) + 1):
        m = b == k
        if m.sum() < 20:
            continue
        lines.append(f"    [{lo[k]:>10.3g}, {hi[k]:>10.3g})  share {m.mean():5.2f}  slip {slip[m].mean():6.1f}  med {np.median(slip[m]):6.1f}  "
                     f"race {race[m].mean():4.2f}  jump {jump[m].mean():6.1f}  >=30bp {np.mean(slip[m] >= 30):4.2f}")
    return "\n".join(lines)


def features(d: pl.DataFrame) -> list[str]:
    return [c for c, dt in d.schema.items() if c not in META and dt.is_numeric()]


def best_rules(d: pl.DataFrame, tail: float, keep: np.ndarray | None = None) -> tuple[list[dict], np.ndarray]:
    days = sorted(d["day"].unique().to_list())
    is_train = d["day"].is_in(days[0::2]).to_numpy()
    month = d["day"].str.slice(0, 6).to_numpy()
    slip, race = d["slip_bp"].to_numpy(), d["race"].to_numpy()
    keep = np.ones(d.height, bool) if keep is None else keep
    rows = []
    for col in features(d):
        x = d[col].cast(pl.Float64).to_numpy()
        ok = np.isfinite(x) & keep
        tr, te = ok & is_train, ok & ~is_train
        if tr.sum() < 200 or te.sum() < 200 or np.unique(x[tr][:200_000]).size < 3:
            continue
        lo_thr, hi_thr = np.quantile(x[tr], tail), np.quantile(x[tr], 1 - tail)
        cands = {"low": x <= lo_thr, "high": x >= hi_thr}
        cands = {s: m for s, m in cands.items() if 0.02 <= (tr & m).sum() / tr.sum() <= 0.6}
        if not cands:
            continue
        side = max(cands, key=lambda s: slip[tr & cands[s]].mean())
        bad = cands[side]
        rem, kp = te & bad, te & ~bad
        if rem.sum() < 30 or kp.sum() < 30:
            continue
        months = [m for m in np.unique(month) if (ok & (month == m) & bad).sum() >= 15 and (ok & (month == m) & ~bad).sum() >= 15]
        agree = sum(slip[ok & (month == m) & bad].mean() > slip[ok & (month == m) & ~bad].mean() for m in months)
        rows.append(dict(feature=col, side=side, thr=float(lo_thr if side == "low" else hi_thr),
                         removed_share=float(rem.sum() / te.sum()), removed_slip=float(slip[rem].mean()), kept_slip=float(slip[kp].mean()),
                         gain_bp=float(slip[te].mean() - slip[kp].mean()), removed_race=float(race[rem].mean()),
                         slip_share_removed=float(np.clip(slip[rem], 0, None).sum() / max(np.clip(slip[te], 0, None).sum(), 1e-9)),
                         months=f"{agree}/{len(months)}"))
    rows.sort(key=lambda r: -r["gain_bp"])
    return rows, is_train


def rule_mask(d: pl.DataFrame, r: dict) -> np.ndarray:
    x = d[r["feature"]].cast(pl.Float64).to_numpy()
    return (x <= r["thr"]) if r["side"] == "low" else (x >= r["thr"])


def fmt(r: dict) -> str:
    return (f"  {r['feature']:24s} {r['side']:4s} thr {r['thr']:>10.4g} | removes {r['removed_share']:4.2f} of fills at {r['removed_slip']:6.1f} bp "
            f"(race {r['removed_race']:4.2f}, {r['slip_share_removed']:4.2f} of slip) | kept {r['kept_slip']:6.1f} | gain {r['gain_bp']:5.1f} | months {r['months']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leg", default="all")
    ap.add_argument("--tail", type=float, default=0.2)
    ap.add_argument("--population", default="episode", choices=["all", "policy", "episode"])
    ap.add_argument("--baselines", action="store_true", help="only print the baseline line of every population")
    ap.add_argument("--v1", default="v1", help="replay run used for the cross-check")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()
    for leg in (LEGS if args.leg == "all" else [args.leg]):
        d = load_leg(leg, args.population)
        print(f"\n################ {leg} | population={args.population} | {d['day'].n_unique()} days\n{base_line(d)}")
        v1 = load_leg(leg, "v1", args.v1)
        print(f"   v1 replay fills  : {base_line(v1)}")
        if args.baselines:
            for pop in ("all", "policy", "episode"):
                if pop != args.population:
                    print(f"   {pop:17s}: {base_line(load_leg(leg, pop))}")
            continue
        rules, is_train = best_rules(d, args.tail)
        pl.from_dicts(rules).write_csv(DATA_ROOT / "slip" / f"screen_{leg}_{args.population}_tail{int(args.tail * 100)}.csv")
        slip = d["slip_bp"].to_numpy()
        print(f"-- single rules: cancel while the feature is in its worse {args.tail:.0%} tail (side+threshold from train days; test-day numbers, base {slip[~is_train].mean():.1f} bp)")
        for r in rules[:args.top]:
            line = fmt(r)
            if v1.height >= 50:
                m = rule_mask(v1, r)
                vs = v1["slip_bp"].to_numpy()
                ok = np.isfinite(v1[r["feature"]].cast(pl.Float64).to_numpy())
                if (m & ok).sum() >= 5:
                    line += f" || v1: removes {np.mean(m & ok):4.2f} at {vs[m & ok].mean():6.1f}, kept {vs[~m & ok].mean():6.1f}"
            print(line)
        print("-- greedy stack (each next rule is chosen on what the previous ones kept):")
        keep = np.ones(d.height, bool)
        chosen = []
        for step in range(3):
            rs, _ = best_rules(d, args.tail, keep)
            rs = [r for r in rs if r["feature"] not in {c["feature"] for c in chosen}]
            if not rs or rs[0]["gain_bp"] < 0.3:
                break
            chosen.append(rs[0])
            keep &= ~(rule_mask(d, rs[0]) & np.isfinite(d[rs[0]["feature"]].cast(pl.Float64).to_numpy()))
            te = ~is_train
            line = (f"  +{rs[0]['feature']} {rs[0]['side']} {rs[0]['thr']:.4g}: test fills kept {np.mean(keep[te]):4.2f}, slip {slip[te & keep].mean():5.1f} "
                    f"(from {slip[te].mean():5.1f}), race {d['race'].to_numpy()[te & keep].mean():4.2f}")
            if v1.height >= 50:
                vk = np.ones(v1.height, bool)
                for c in chosen:
                    vk &= ~(rule_mask(v1, c) & np.isfinite(v1[c["feature"]].cast(pl.Float64).to_numpy()))
                line += f" || v1: kept {vk.mean():4.2f}, slip {v1['slip_bp'].to_numpy()[vk].mean():5.1f} (from {v1['slip_bp'].mean():5.1f})"
            print(line)
        print("-- named conditions (population sample)")
        for col in NAMED:
            if col in d.columns:
                print(bucket_table(d, col))


if __name__ == "__main__":
    main()
