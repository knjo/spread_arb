"""Stage 3 report: read a replay run directory and print the performance faces the user asked for.

    uv run --project /home/kevin/Project/HFT --no-sync python -m spreadArb.src.backtest.report --run v1
"""
from __future__ import annotations

import argparse
import json

import polars as pl

from ..common.paths import DATA_ROOT


def load_run(name: str) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame | None, dict]:
    d = DATA_ROOT / "backtest" / name
    pos = pl.read_parquet(d / "positions.parquet")
    daily = pl.read_csv(d / "daily.csv", infer_schema_length=None)
    rb = pl.read_csv(d / "rollbacks.csv") if (d / "rollbacks.csv").exists() else None
    cfg = json.loads((d / "config.json").read_text())
    return pos, daily, rb, cfg


def _bp(col: str = "pnl_net") -> pl.Expr:
    return (pl.col(col) / (pl.col("spot_buy_cash") / 1e4) * 1e4)


def group_lines(pos: pl.DataFrame, key: str) -> list[str]:
    g = (pos.group_by(key).agg(pl.len().alias("n"), pl.col("pnl_net").sum().alias("pnl"),
                               _bp().mean().alias("bp_mean"), _bp().median().alias("bp_med"),
                               pl.col("holding_days").mean().alias("hold"), pl.col("ev_bp").mean().alias("ev"),
                               pl.col("t_days_pred").mean().alias("t_pred"),
                               (pl.col("close_day") == pl.col("quote_day")).mean().alias("same_day"),
                               (pl.col("spot_buy_cash") / 1e4).sum().alias("notional"))
         .sort(key))
    out = [f"by {key}:"]
    for r in g.iter_rows(named=True):
        out.append(f"  {str(r[key]):12s} n {r['n']:5d} pnl {r['pnl'] or 0:>12,.0f} | net {r['bp_mean'] or 0:6.1f} bp (med {r['bp_med'] or 0:6.1f}) "
                   f"vs EV {r['ev']:6.1f} | hold {r['hold'] or 0:5.2f} d vs pred {r['t_pred']:5.2f} | same-day {r['same_day']:.2f} | "
                   f"notional {r['notional'] / 1e6:7.1f} M")
    return out


def report(name: str) -> str:
    pos, daily, rb, cfg = load_run(name)
    lines = [f"run {name}: {cfg['cap_twd'] / 1e6:.0f} M cap, hurdle {cfg['hurdle_bp_per_trading_day']} bp/trading day, "
             f"streams {cfg['streams']}, residual_min {cfg['residual_min_bp']}, floor {cfg['floor_bp']}, "
             f"S1 levels {cfg['s1_levels']}, exit tol {cfg['exit_tol_bp']}, place/cancel {cfg['place_ns'] / 1e6:.0f}/{cfg['cancel_ns'] / 1e6:.0f} ms, "
             f"exit routes {cfg.get('exit_routes', ['E1'])}, gates {cfg.get('gates_tag')}"]
    n_days = daily.height
    days_with_table = daily.filter(pl.col("reach_days") > 0).height
    total = float(pos["pnl_net"].fill_null(0).sum()) + (float(rb["cost_twd"].sum()) if rb is not None else 0.0)
    closed = pos.filter(pl.col("close_kind").is_in(["maker_exit", "settlement"]))
    marked = pos.filter(pl.col("close_kind") == "open_marked")
    lines.append(f"days {n_days} ({daily['day'][0]}..{daily['day'][-1]}; Q table available on {days_with_table}) | pairs {pos.height} "
                 f"(closed {closed.height}, marked open at end {marked.height}) | rollbacks {0 if rb is None else rb.height}")
    lines.append(f"net PnL {total:,.0f} TWD = {total / max(n_days, 1):,.0f}/day on {cfg['cap_twd'] / 1e6:.0f} M "
                 f"= {total / n_days * 250 / cfg['cap_twd'] * 100:.1f}% planned annual (250 sessions) | "
                 f"realized part {float(closed['pnl_net'].sum()):,.0f}, marked-open part {float(marked['pnl_net'].sum()):,.0f}")
    core = daily.filter((pl.col("day") >= 20260401) & (pl.col("day") <= 20260731))
    if core.height:
        cp = pos.filter((pl.col("quote_day") >= "20260401") & (pl.col("quote_day") <= "20260731"))
        lines.append(f"core window 4/1-7/31 ({core.height} days, tables warmed up): {float(core['pnl_net'].mean()):,.0f}/day = "
                     f"{float(core['pnl_net'].mean()) * 250 / cfg['cap_twd'] * 100:.1f}% planned annual | {cp.height / core.height:.1f} pairs/day | "
                     f"committed mean {float(core['committed_end'].mean()) / 1e6:.1f} M, at cap {core.filter(pl.col('committed_end') >= 0.975 * cfg['cap_twd']).height} days | "
                     f"daily Sharpe {float(core['pnl_net'].mean() / core['pnl_net'].std()) * 250 ** 0.5:.1f}")
    cum = daily["pnl_net"].cum_sum()
    lines.append(f"whole run: max drawdown of cumulative daily pnl {float((cum - cum.cum_max()).min()):,.0f} | daily Sharpe {float(daily['pnl_net'].mean() / daily['pnl_net'].std()) * 250 ** 0.5:.1f}"
                 + (f" | hurdle raised on {int((daily['hurdle_used'] > cfg['cost']['hurdle_bp_per_day'] + 0.01).sum())} days (mean {float(daily['hurdle_used'].mean()):.1f} bp/day)" if "hurdle_used" in daily.columns else ""))
    # per-trade
    lines.append(f"per pair: net {pos.select(_bp().mean()).item():.1f} bp mean, "
                 f"{pos.select(_bp().median()).item():.1f} bp median, {float(pos['pnl_net'].mean()):,.0f} TWD/pair | "
                 f"notional/pair {float(pos['spot_buy_cash'].mean()) / 1e4 / 1e3:,.0f} k | "
                 f"win rate {float((pos['pnl_net'] > 0).mean()):.2f}")
    # volume / turnover / day-trade
    notional = float(pos["spot_buy_cash"].sum()) / 1e4
    lines.append(f"volume: {pos.height / n_days:.1f} pairs/day, {notional / 1e6:.0f} M spot notional bought = "
                 f"{notional / n_days / 1e6:.2f} M/day = {notional / n_days / cfg['cap_twd']:.3f} turns/day of cap")
    same_day = float((closed["close_day"] == closed["quote_day"]).mean()) if closed.height else float("nan")
    lines.append(f"day-trade rate (closed pairs): {same_day:.3f} | maker exits {pos.filter(pl.col('close_kind') == 'maker_exit').height}, "
                 f"settlements {pos.filter(pl.col('close_kind') == 'settlement').height}")
    # position / holding
    lines.append(f"positions: mean open at close {float(daily['open_end'].mean()):.1f}, max {int(daily['open_end'].max())} | "
                 f"committed at close mean {float(daily['committed_end'].mean()) / 1e6:.1f} M, max {float(daily['committed_end'].max()) / 1e6:.1f} M | "
                 f"holding mean {float(closed['holding_days'].mean()) if closed.height else float('nan'):.2f} d (median "
                 f"{float(closed['holding_days'].median()) if closed.height else float('nan'):.2f}) vs predicted {float(closed['t_days_pred'].mean()) if closed.height else float('nan'):.2f}")
    conc = (pos.group_by("quote_day", "vc").agg(pl.len().alias("n"), (pl.col("spot_buy_cash") / 1e4).sum().alias("notional")))
    lines.append(f"per product: max positions opened in one day {int(conc['n'].max())}, mean {float(conc['n'].mean()):.2f} | "
                 f"top-10 products' share of pairs {pos.group_by('vc').len().sort('len', descending=True).head(10)['len'].sum() / pos.height:.2f} | "
                 f"product-cap rejects {int(daily['product_cap_rejects'].sum()) if 'product_cap_rejects' in daily.columns else 0:,}")
    lines.append(f"admission: candidates {int(daily['candidates'].sum()):,}, gate rejects {int(daily['gate_rejects'].sum()) if 'gate_rejects' in daily.columns else 0:,}, evaluated {int(daily['evaluated'].sum()):,}, "
                 f"admitted {int(daily['admitted'].sum()):,}, filled {int(daily['filled'].sum()):,}, cap rejects {int(daily['cap_rejects'].sum()):,}, "
                 f"hurdle rejects {int(daily['hurdle_rejects'].sum()):,}, no estimate {int(daily['no_estimate'].sum()):,}")
    for key in ("stream", "route", "close_kind"):
        lines += group_lines(pos, key)
    if "exit_route" in pos.columns and pos["exit_route"].drop_nulls().n_unique() > 1:
        ex_ = pos.filter(pl.col("close_kind") == "maker_exit")
        lines += group_lines(ex_, "exit_route")
        lines.append(f"  exits where the other route would also have filled inside the cancel latency: {int(ex_['exit_double_risk'].sum())}")
    # calibration: predicted EV vs realized by EV bucket (closed pairs only)
    if closed.height:
        b = closed.with_columns(pl.col("ev_bp").cut([20, 40, 60, 100], labels=["<20", "20-40", "40-60", "60-100", ">=100"]).alias("ev_bucket"))
        g = b.group_by("ev_bucket").agg(pl.len().alias("n"), pl.col("ev_bp").mean().alias("ev"), _bp().mean().alias("net"),
                                        pl.col("holding_days").mean().alias("hold"), pl.col("t_days_pred").mean().alias("t_pred"),
                                        pl.col("p_sd_pred").mean().alias("p_sd"),
                                        (pl.col("close_day") == pl.col("quote_day")).mean().alias("sd")).sort("ev_bucket")
        lines.append("calibration (closed pairs) by predicted EV bucket:")
        for r in g.iter_rows(named=True):
            lines.append(f"  EV {str(r['ev_bucket']):7s} n {r['n']:5d} pred {r['ev']:6.1f} realized {r['net']:6.1f} bp | "
                         f"hold {r['hold']:5.2f} vs {r['t_pred']:5.2f} d | P_sd pred {r['p_sd']:.2f} vs same-day {r['sd']:.2f}")
    # slippage
    lines.append("entry slippage quote_ab - actual_ab (bp) by stream x race:")
    g = (pos.with_columns((pl.col("quote_ab") - pl.col("actual_ab")).alias("slip"))
         .group_by("stream", "entry_race").agg(pl.len().alias("n"), pl.col("slip").mean().alias("mean"),
                                                 pl.col("slip").median().alias("med"), pl.col("slip").quantile(0.9).alias("p90"))
         .sort("stream", "entry_race"))
    for r in g.iter_rows(named=True):
        lines.append(f"  {r['stream']} race={str(r['entry_race']):5s} n {r['n']:5d} mean {r['mean']:6.1f} med {r['med']:6.1f} p90 {r['p90']:6.1f}")
    for s in ("S1", "S2"):
        sub = pos.filter(pl.col("stream") == s)
        if sub.height:
            lines.append(f"  {s} all: mean {sub.select((pl.col('quote_ab') - pl.col('actual_ab')).mean()).item():.1f} bp "
                         f"(config d_in_base {cfg['cost']['d_in_base'].get(s)} + margin {cfg['cost']['margin_bp']})")
    ex = pos.filter(pl.col("close_kind") == "maker_exit")
    if ex.height:
        lines.append("exit slippage realized - quote basis (bp) by race:")
        g = (ex.with_columns((pl.col("exit_realized_ab") - pl.col("exit_quote_ab")).alias("slip"),
                             (pl.col("exit_realized_ab") - pl.col("target_bp")).alias("vs_target"),
                             ((pl.col("exit_fill_ns") - pl.col("exit_quote_ns")) / 1e9).alias("wait_s"))
             .group_by("exit_race").agg(pl.len().alias("n"), pl.col("slip").mean().alias("mean"), pl.col("slip").median().alias("med"),
                                        pl.col("vs_target").mean().alias("vs_target"), pl.col("wait_s").median().alias("wait_med"),
                                        pl.col("wait_s").mean().alias("wait_mean")).sort("exit_race"))
        for r in g.iter_rows(named=True):
            lines.append(f"  race={str(r['exit_race']):5s} n {r['n']:5d} mean {r['mean']:6.1f} med {r['med']:6.1f} | vs target {r['vs_target']:6.1f} | "
                         f"wait med {r['wait_med']:7.1f} s mean {r['wait_mean']:7.1f} s")
        if "exit_route" in ex.columns:
            for r in (ex.with_columns((pl.col("exit_realized_ab") - pl.col("exit_quote_ab")).alias("slip")).group_by("exit_route")
                      .agg(pl.len().alias("n"), pl.col("slip").mean().alias("mean"), pl.col("slip").median().alias("med"),
                           pl.col("exit_race").mean().alias("race")).sort("exit_route").iter_rows(named=True)):
                lines.append(f"  route {r['exit_route']}: n {r['n']:5d} mean {r['mean']:6.1f} med {r['med']:6.1f} race {r['race']:.2f}")
        lines.append(f"  all exits: mean {ex.select((pl.col('exit_realized_ab') - pl.col('exit_quote_ab')).mean()).item():.1f} bp "
                     f"(config d_out_base {cfg['cost']['d_out_base']} + margin {cfg['cost']['margin_bp']})")
    # monthly
    m = (pos.with_columns(pl.col("quote_day").str.slice(0, 6).alias("month"))
         .group_by("month").agg(pl.len().alias("n"), pl.col("pnl_net").sum().alias("pnl"), _bp().mean().alias("bp"),
                                (pl.col("close_kind") == "open_marked").sum().alias("marked")).sort("month"))
    lines.append("by entry month:")
    for r in m.iter_rows(named=True):
        lines.append(f"  {r['month']} n {r['n']:5d} pnl {r['pnl']:>12,.0f} | net {r['bp']:6.1f} bp | marked open {r['marked']}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="v1")
    print(report(ap.parse_args().run))


if __name__ == "__main__":
    main()
