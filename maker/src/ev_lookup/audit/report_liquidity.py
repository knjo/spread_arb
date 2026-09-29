"""Summarize the completed inside/depth/buffer experiment without selecting a winner."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path

import polars as pl

from ..analyze_full_study import analyze
from ..verify_full_study import read_rows
from .report_v21 import maker_message_load


VARIANTS = ["inside", "depth5", "buffer50", "depth5_buffer50"]


def report(root):
    summaries, all_fills, messages, reasons, months = [], [], [], [], []
    for variant in VARIANTS:
        manifest = json.loads((root/variant/"manifest.json").read_text())
        if manifest.get("planned_days", manifest["days"]) != manifest["days"]:
            raise AssertionError("The planned full continuation has not finished")
        verification = json.loads((root/variant/"verification_liquidity.json").read_text())
        if not verification["passed"] or verification["base"]["base"]["days"] != len(manifest["days"]):
            raise AssertionError("liquidity verification missing or failed")
    # Each analysis writes only its own variant directory. Spawn avoids
    # inheriting a parent process's initialized Polars thread pool.
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
        analyses = dict(zip(VARIANTS, pool.map(analyze, [root/v for v in VARIANTS]), strict=True))
    for variant in VARIANTS:
        folder = root/variant
        manifest = json.loads((folder/"manifest.json").read_text())
        analysis = analyses[variant]
        name = manifest["configurations"][0]["name"]
        summary = analysis["summaries"][0]
        for row in analysis["monthly"]:
            months.append(dict(variant=variant, **row))
        positions, quotes, at_fill = {}, {}, {}
        for day in manifest["days"]:
            current = folder/f"Date={day}"/name
            decisions = read_rows(current/"decisions.parquet")
            quotes.update({d["intent_id"]:d for d in decisions if d["admit"]})
            positions.update({p["id"]:p for p in read_rows(current/"positions.parquet") if p["entry_fill_ns"] is not None})
            tape_path = current/"execution.parquet"
            tape = pl.scan_parquet(tape_path) if tape_path.exists() else None
            # Millions of rejected, never-sent proposals need aggregate counts,
            # not Python dictionaries. Every actual order/fill/hedge is retained.
            trace = (tape.filter(pl.col("kind") != "s2_liquidity_reject").collect().to_dicts()
                     if tape is not None else [])
            at_fill.update({t["position_id"]:t for t in trace if t["kind"] == "s2_fill_liquidity"})
            messages.append(dict(variant=variant, **maker_message_load(day, trace)))
            categories = {}
            for d in decisions:
                key = d["stream"], "admission", d["reason"]
                categories[key] = categories.get(key, 0)+1
            if tape is not None:
                grouped = tape.filter(pl.col("kind").is_in(["s2_liquidity_reject", "event_cancel"]))
                for t in grouped.group_by("kind", "reason").len().collect().to_dicts():
                    key = "S2", t["kind"], t["reason"]
                    categories[key] = categories.get(key, 0)+t["len"]
            reasons.extend(dict(variant=variant, day=day, stream=k[0], stage=k[1], reason=k[2], count=n)
                           for k, n in categories.items())
        final_marks = read_rows(folder/f'Date={manifest["days"][-1]}'/name/"marks.parquet")
        s2 = [p for p in positions.values() if p["stream"] == "S2"]
        final_s2_marks = [m["official_mark_twd"] for m in final_marks if positions[m["position_id"]]["stream"] == "S2"]
        if any(x is None for x in final_s2_marks):
            s2_total = None
        else:
            s2_total = sum(p["pnl_twd"] for p in s2 if p["state"] == "closed")+sum(final_s2_marks)
        count_days = len(manifest["available_days"])
        daily = pl.read_csv(folder/f"{name}_daily.csv", schema_overrides={"day":pl.String})
        shadow_daily = pl.read_csv(folder/"shadow_daily.csv")
        row = dict(variant=variant, **summary,
            after_funding_2pct_per_day=(summary["final_official_equity_twd"]-summary["funding_2pct_twd"])/count_days
                if summary["final_official_equity_twd"] is not None else None,
            s2_fills=len(s2), s2_fills_per_day=len(s2)/count_days,
            s2_total_twd=s2_total, s2_total_per_day=s2_total/count_days if s2_total is not None else None,
            s2_nonpositive_basis=sum(p["actual_ab"] is not None and p["actual_ab"] <= 0 for p in s2),
            shadow_s2_fills=shadow_daily["fills_s2"].sum(),
            active_entry_days=daily.filter(pl.col("fills") > 0).height,
            last_entry_day=daily.filter(pl.col("fills") > 0)["day"].max())
        summaries.append(row)
        for p in s2:
            d = quotes[p["id"]]
            t = at_fill[p["id"]]
            actual = p["spot_buy_cash"]/p["spot_buy_qty"]/10000 if p["spot_buy_qty"] else None
            before = t.get("depth_vwap")
            lag = (p["hedged_ns"]-p["entry_fill_ns"])/1e6 if p["hedged_ns"] is not None else None
            all_fills.append(dict(variant=variant, position_id=p["id"], day=p["entry_day"],
                vc=p["contract"]["vc"], hedge_shares=p["contract"]["shares"],
                quote_a1_multiple=d.get("liquidity_a1_multiple"), preprint_a1_multiple=t["a1_multiple"],
                preprint_depth_reason=t["reason"], preprint_depth_vwap=before,
                preprint_adverse_vwap=t.get("adverse_vwap"), actual_hedge_price=actual, hedge_ms=lag,
                actual_basis=p["actual_ab"], quote_basis=p["quote_ab"], quote_ev_bp=d["est_bp"],
                quote_execution_buffer_bp=d["execution_cost_bp"],
                quote_to_hedge_basis_loss_bp=p["quote_ab"]-p["actual_ab"] if p["actual_ab"] is not None else None,
                hedge_move_bp=(actual/before-1)*10000 if before and actual else None,
                closed=p["state"] == "closed", close_kind=p["close_kind"],
                pnl_twd=p["pnl_twd"] if p["state"] == "closed" else None,
                realized_bp=p["pnl_bp"] if p["state"] == "closed" else None,
                expired=p["contract"]["expiry"] < manifest["days"][-1]))
    summary = pl.from_dicts(summaries, infer_schema_length=None)
    summary.write_csv(root/"comparison.csv")
    fills = pl.from_dicts(all_fills, infer_schema_length=None)
    fills.write_parquet(root/"s2_fills.parquet")
    pl.from_dicts(months).write_csv(root/"monthly_comparison.csv")
    pl.from_dicts(messages).write_csv(root/"message_load.csv")
    pl.from_dicts(reasons).write_csv(root/"decision_reasons_daily.csv")
    diagnostic = []
    for variant in VARIANTS:
        sample = fills.filter(pl.col("variant") == variant)
        eligible = sample.filter(pl.col("hedge_move_bp").is_not_null())
        for tag, f in [("all_priceable", eligible), ("exact_50ms", eligible.filter((pl.col("hedge_ms")-50).abs() < 1e-6))]:
            worse = f.filter(pl.col("hedge_move_bp") > 1e-7)
            diagnostic.append(dict(variant=variant, sample=tag, all_s2_fills=sample.height, priceable=f.height,
                worse_fraction=worse.height/f.height if f.height else None,
                more_than_one_tick_fraction=f.filter(pl.col("actual_hedge_price") > pl.col("preprint_adverse_vwap")+1e-8).height/f.height if f.height else None,
                mean_move_bp=f["hedge_move_bp"].mean(), median_move_bp=f["hedge_move_bp"].median(),
                p95_move_bp=f["hedge_move_bp"].quantile(.95),
                mean_move_bp_given_worse=worse["hedge_move_bp"].mean(),
                median_preprint_a1_multiple=f["preprint_a1_multiple"].median()))
    pl.from_dicts(diagnostic, infer_schema_length=None).write_csv(root/"s2_hedge_diagnostic.csv")
    fills.filter(pl.col("expired")).group_by("variant").agg(
        pl.len().alias("matured_s2"), pl.col("closed").sum().alias("closed"),
        pl.col("quote_ev_bp").mean().alias("mean_quote_ev_bp"),
        pl.col("realized_bp").mean().alias("mean_realized_bp"),
        pl.col("quote_execution_buffer_bp").mean().alias("mean_quote_buffer_bp")
    ).write_csv(root/"s2_forecast_calibration.csv")
    result = dict(summaries=summaries, hedge_diagnostics=diagnostic,
        report_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note="Whole S1+S2 PnL includes final official marks and C8. S2 totals include S2 final marks. The fixed buffer changes quotes only. Hedge diagnostics condition on actual futures fills and full-quantity preprint prices; missing quotes and delayed hedges stay in total PnL.")
    (root/"comparison.json").write_text(json.dumps(result, indent=2)+"\n")
    plot(root)
    return summary.select("variant", "equity_per_available_day", "after_funding_2pct_per_day", "s2_fills", "s2_total_per_day", "mean_paired_twd")


def plot(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime
    labels = {"inside":"First priority", "depth5":"A1 >= 5x hedge", "buffer50":"50% x 1-tick buffer", "depth5_buffer50":"Depth + buffer"}
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, layout="constrained")
    for variant in VARIANTS:
        d = pl.read_csv(root/variant/f"{variant}_20M_daily.csv", schema_overrides={"day":pl.String})
        dates = [datetime.strptime(x, "%Y%m%d") for x in d["day"]]
        eq = [x/1000 if x is not None else float("nan") for x in d["official_equity_twd"]]
        axes[0].plot(dates, eq, label=labels[variant], lw=1.5)
        axes[1].plot(dates, d["fills_s2"].cum_sum().to_list(), lw=1.5)
    axes[0].set(ylabel="Equity (TWD thousands)", title="S1 + S2, 20M reserved cap; transaction costs included, before funding")
    axes[0].legend(ncol=2)
    axes[1].set(ylabel="Cumulative S2 fills", xlabel="Session date (2026)")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(root/"liquidity_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    a = p.parse_args()
    print(report(a.root))
