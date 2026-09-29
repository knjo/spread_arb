"""Full-period cash, funding, cost and mature-cohort EV calibration report."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl

from ...ev_lookup.audit.execution_costs import audit as audit_costs, filled_positions
from ..analyze_full_study import analyze
from .equivalence import check as check_control
from .execution_breakdown import breakdown


def decisions(root, actor, days):
    frames = []
    cols = ["intent_id","ns","stream","admit","reason","est_bp","quote_ab","eff_u",
            "p_sd","p_overnight","p_expiry","p_other","d_in","d_sd","d_on","execution_cost_bp"]
    for day in days:
        path = root/f"Date={day}"/actor/"decisions.parquet"
        if path.exists():
            frames.append(pl.scan_parquet(path).select(cols).collect())
    return pl.concat(frames,how="diagonal_relaxed").unique(["intent_id","ns"],keep="last")


def forecast_rows(positions, quotes, final_day):
    return (positions.filter(pl.col("actual_ab").is_not_null() &
                              (pl.col("contract").struct.field("expiry") < final_day))
            .select("id","quote_ns","stream","entry_day","close_day","close_kind","pnl_bp","pnl_twd")
            .join(quotes.drop("stream"),left_on=["id","quote_ns"],right_on=["intent_id","ns"],how="left",validate="1:1")
            .with_columns(pl.when(pl.col("close_day").is_null()).then(pl.lit("unresolved"))
                .when(pl.col("entry_day") == pl.col("close_day")).then(pl.lit("sd"))
                .when(pl.col("close_kind") == "maker_exit").then(pl.lit("overnight"))
                .when(pl.col("close_kind") == "expiry_basis_zero_accounting").then(pl.lit("expiry"))
                .otherwise(pl.lit("other")).alias("actual_kind"),
                pl.when(pl.col("close_day").is_null()).then(None).otherwise(pl.col("pnl_bp")).alias("realized_bp")))


def generate(root: Path):
    plan=json.loads((root/"plain/manifest.json").read_text())
    control=None
    if len(plan["days"]) == 86:
        prior=Path(__file__).resolve().parents[3]/"data/ev_lookup_v22_full_20260909/depth5_buffer50"
        control=check_control(root,prior)
    summaries, streams, cohorts, selection, source_audits, monthly = [], [], [], [], [], []
    for mode in ("plain","guard"):
        folder = root/mode
        m = json.loads((folder/"manifest.json").read_text())
        if m["status"] != "completed":
            raise AssertionError("refuse partial-period headline")
        if not json.loads((folder/"verification_cost.json").read_text())["passed"]:
            raise AssertionError("independent full audit is required")
        result = analyze(folder)
        monthly.extend(dict(mode=mode,**r) for r in result["monthly"])
        source_audits.append(audit_costs(folder,folder/"cost_audit"))
        days, n = m["days"],len(m["available_days"])
        shadow = filled_positions(folder,"shadow",days)
        for row in result["summaries"]:
            name = row["portfolio"]
            row.update(mode=mode,after_funding_2pct_per_day=(row["final_official_equity_twd"]-row["funding_2pct_twd"])/n)
            row["bidask_per_day"] = ((row["realized_twd"]+row["final_marked_open_bidask_twd"])/n
                                      if row["final_unmarked_bidask"] == 0 else None)
            row["bidask_after_funding_2pct_per_day"] = (row["bidask_per_day"]-row["funding_2pct_twd"]/n
                                                        if row["bidask_per_day"] is not None else None)
            summaries.append(row)
            pos = filled_positions(folder,name,days)
            quotes = decisions(folder,name,days)
            marks_path = folder/f"Date={days[-1]}"/name/"marks.parquet"
            final_marks = (pl.read_parquet(marks_path) if marks_path.exists() else
                           pl.DataFrame(schema={"position_id":pl.String,"official_mark_twd":pl.Float64}))
            marked = pos.join(final_marks.select("position_id","official_mark_twd"),
                              left_on="id",right_on="position_id",how="left",validate="1:1")
            for stream in ("S1","S2"):
                part = marked.filter(pl.col("stream") == stream)
                realized = part.filter(pl.col("close_ns").is_not_null())["pnl_twd"].sum()
                opened = part.filter(pl.col("close_ns").is_null())
                if opened["official_mark_twd"].null_count():
                    raise AssertionError("final open inventory lacks official valuation")
                mark = opened["official_mark_twd"].sum()
                streams.append(dict(mode=mode,portfolio=name,stream=stream,fills=part.height,
                    realized_twd=realized,final_open_twd=mark,total_twd=realized+mark,per_day=(realized+mark)/n))
            independent_total = sum(r["total_twd"] for r in streams if r["portfolio"] == name)
            if abs(independent_total-row["final_official_equity_twd"])>1e-5:
                raise AssertionError("independent S1+S2 cash and inventory do not match total equity")
            own = forecast_rows(pos,quotes,days[-1]).with_columns(pl.lit(mode).alias("mode"),pl.lit(name).alias("portfolio"))
            if own["est_bp"].null_count():
                raise AssertionError("filled entry has no recorded quote decision")
            cohorts.append(own)
            # Match the original causal decision to a common shadow opportunity.
            # A missing actor decision remains unmatched, never an invented EV rejection.
            matched = forecast_rows(shadow,quotes,days[-1]).with_columns(
                pl.col("reason").fill_null("unmatched").alias("decision_reason"),
                pl.lit(mode).alias("mode"),pl.lit(name).alias("portfolio"))
            selection.append(matched)
    pl.from_dicts(summaries,infer_schema_length=None).write_csv(root/"comparison.csv")
    pl.from_dicts(streams,infer_schema_length=None).write_csv(root/"streams.csv")
    pl.from_dicts(monthly,infer_schema_length=None).write_csv(root/"monthly_comparison.csv")
    cohort = pl.concat(cohorts,how="diagonal_relaxed")
    cohort.write_parquet(root/"matured_forecasts.parquet")
    calibration = cohort.group_by("mode","portfolio","stream").agg(pl.len().alias("pairs"),
        pl.col("est_bp").mean().alias("quote_ev_bp"),pl.col("realized_bp").mean(),
        *[pl.col("p_"+k).mean().alias("predicted_"+k) for k in ("sd","overnight","expiry","other")],
        *[(pl.col("actual_kind")==k).mean().alias("actual_"+k) for k in ("sd","overnight","expiry","other","unresolved")])
    calibration.write_csv(root/"forecast_calibration.csv")
    selected = pl.concat(selection,how="diagonal_relaxed")
    selected.write_parquet(root/"shadow_selection.parquet")
    selected.group_by("mode","portfolio","stream","decision_reason").agg(pl.len().alias("pairs"),
        pl.col("realized_bp").mean(),pl.col("est_bp").mean().alias("quote_ev_bp"),
        (pl.col("actual_kind")=="unresolved").sum().alias("unresolved")).write_csv(root/"shadow_selection.csv")
    selected.with_columns(pl.col("entry_day").str.slice(0,6).alias("entry_month"))\
        .group_by("mode","portfolio","stream","entry_month","decision_reason").agg(
            pl.len().alias("pairs"),pl.col("realized_bp").mean(),pl.col("est_bp").mean().alias("quote_ev_bp"),
            (pl.col("actual_kind")=="unresolved").sum().alias("unresolved"))\
        .write_csv(root/"shadow_selection_monthly.csv")
    # Store fixed-bin ranking separately from capacity-constrained actual PnL.
    binned = selected.filter(pl.col("est_bp").is_not_null()).with_columns(
        pl.col("est_bp").cut([0,10,25,50],labels=["<0","0-10","10-25","25-50",">=50"],left_closed=True)
        .alias("ev_bucket"))
    ranking = binned.group_by("mode","portfolio","stream","ev_bucket").agg(pl.len().alias("pairs"),
        pl.col("realized_bp").mean(),pl.col("est_bp").mean(),(pl.col("realized_bp")>0).mean().alias("win_fraction"))
    ranking.write_csv(root/"shadow_ev_ranking.csv")
    binned.with_columns(pl.col("entry_day").str.slice(0,6).alias("entry_month"))\
        .group_by("mode","portfolio","stream","entry_month","ev_bucket").agg(
            pl.len().alias("pairs"),pl.col("realized_bp").mean(),pl.col("est_bp").mean(),
            (pl.col("realized_bp")>0).mean().alias("win_fraction"))\
        .write_csv(root/"shadow_ev_ranking_monthly.csv")
    execution=breakdown(root)
    report = dict(summaries=summaries,streams=streams,calibration=calibration.to_dicts(),control_equivalence=control,
                  execution_diagnostics=execution,
                  audits=source_audits,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/"comparison.json").write_text(json.dumps(report,indent=2)+"\n")
    plot(root,summaries)
    return report


def plot(root, summaries):
    import os
    os.environ.setdefault("MPLCONFIGDIR","/tmp/hft-matplotlib-cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker
    from datetime import datetime
    import numpy as np

    fig, axes = plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
    colors={"fixed_20M":"#0072B2","cost_20M":"#D55E00","cost_guard_20M":"#009E73"}
    labels={"fixed_20M":"Fixed S2 cost floor","cost_20M":"All-leg cost EV","cost_guard_20M":"Cost EV + exit/quote guards"}
    for row in summaries:
        name,mode=row["portfolio"],row["mode"]
        daily=pl.read_csv(root/mode/f"{name}_daily.csv",schema_overrides={"day":pl.String})
        dates=[datetime.strptime(d,"%Y%m%d") for d in daily["day"]]
        axes[0].plot(dates,[v if v is not None else np.nan for v in daily["official_equity_twd"]],
                     color=colors[name],label=labels[name],lw=1.5)
        usage=pl.read_csv(root/mode/"capacity_usage_daily.csv",schema_overrides={"day":pl.String})
        usage=usage.filter(pl.col("portfolio")==name)
        axes[1].plot(dates,usage["mean_paired_twd"]/1e6,color=colors[name],label=labels[name],lw=1.2)
    axes[0].set(title="Cumulative PnL incl. open inventory",ylabel="TWD")
    axes[0].yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    axes[1].set(title="Average paired capital during session",ylabel="TWD million")
    axes[1].axhline(20,color="#777777",ls="--",lw=.8)
    for ax in axes:
        ax.grid(alpha=.2);ax.legend(fontsize=8)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.suptitle("S1 + S2 | First-priority S2 | 20M reserved cap | 2026")
    fig.supxlabel("Before funding; gaps retain unavailable official marks; cancel/hedge 50ms, immediate new queue admission",fontsize=8)
    fig.savefig(root/"cost_equity_comparison.png",dpi=160)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    args=parser.parse_args()
    report=generate(args.root)
    print(json.dumps([dict(portfolio=r["portfolio"],per_day=r["equity_per_available_day"],
                          after_2pct=r["after_funding_2pct_per_day"]) for r in report["summaries"]],indent=2))


if __name__ == "__main__":
    main()
