"""Full-period realized, inventory-marked, and capacity-usage reporting."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path

import polars as pl

from .causal_lookup import SECOND, open_ns
from .verify_full_study import read_rows


def interval_statistics(events: list[tuple[int,float]], start: int, end: int,
                        opening: float=0., limit: float=float("inf")) -> tuple[float,float,float]:
    current, last, area, peak, over = opening, start, 0., opening, 0.
    merged=defaultdict(float)
    for ns,delta in events:
        if not start <= ns <= end:
            raise AssertionError("usage interval outside day")
        merged[ns] += delta
    for ns,delta in sorted(merged.items()):
        area += current*(ns-last)
        if current > limit+1e-6:
            over += (ns-last)/SECOND
        current += delta
        peak = max(peak,current)
        last = ns
    area += current*(end-last)
    if current > limit+1e-6:
        over += (end-last)/SECOND
    return area/(end-start),peak,over


def analyze(root: Path) -> dict:
    manifest=json.loads((root/"manifest.json").read_text())
    if manifest["status"] != "completed":
        raise AssertionError("incomplete full-period study")
    days=manifest["days"]
    day_index={d:i for i,d in enumerate(days)}
    final_ns=open_ns(days[-1])+15600*SECOND
    summaries=[]; monthly=[]; stream_summary=[]; usage_rows=[]; trade_rows=[]
    for config in manifest["configurations"]:
        name=config["name"]
        daily=pl.read_csv(root/f"{name}_daily.csv",schema_overrides={"day":pl.String})
        entries={}; expiry=set(); corporate=set(); rollbacks=set(); prior_committed=0.
        usage=[]; times_10=[]; times_11=[]
        for day in days:
            folder=root/f"Date={day}"/name
            start=open_ns(day); end=start+15600*SECOND
            inventory_events=[]
            for p in read_rows(folder/"positions.parquet"):
                if p["entry_fill_ns"] is None:
                    continue
                entries[p["id"]]=p
                if p["hedged_ns"] is not None and p["future_sell_qty"]:
                    left=max(start,p["hedged_ns"])
                    right=min(end,p["close_ns"] or end)
                    if left < right:
                        nominal=p["spot_buy_cash"]/10_000
                        inventory_events += [(left,nominal),(right,-nominal)]
            ledger=read_rows(folder/"ledger.parquet")
            changes=[(r["ns"],r["delta_cents"]/100) for r in ledger]
            reserved_mean,reserved_peak,over_seconds=interval_statistics(changes,start,end,prior_committed,config["cap_twd"])
            paired_mean,paired_peak,_=interval_statistics(inventory_events,start,end)
            nominal10=sum(delta for ns,delta in inventory_events if ns<=start+3600*SECOND)
            nominal11=sum(delta for ns,delta in inventory_events if ns<=start+7200*SECOND)
            times_10.append(nominal10);times_11.append(nominal11)
            prior_committed += sum(delta for _,delta in changes)
            row=dict(portfolio=name,day=day,mean_committed_twd=reserved_mean,peak_committed_twd=reserved_peak,
                     mean_paired_twd=paired_mean,peak_paired_twd=paired_peak,overrun_seconds=over_seconds,
                     paired_1000_twd=nominal10,paired_1100_twd=nominal11)
            usage.append(row);usage_rows.append(row)
            for t in read_rows(folder/"execution.parquet"):
                if t["kind"] == "expiry_basis_zero_accounting": expiry.add(t["position_id"])
                if t["kind"] == "corporate_risk_exit": corporate.add(t["position_id"])
                if t["kind"] == "partial_entry_rollback": rollbacks.add(t["position_id"])
        paired=[p for p in entries.values() if p["hedged_ns"] is not None and p["future_sell_qty"]]
        closed=[p for p in entries.values() if p["state"] == "closed"]
        capital_days=sum(p["spot_buy_cash"]/10_000*(min(final_ns,p["close_ns"] or final_ns)-p["hedged_ns"])
                         /SECOND/86400 for p in paired)
        realized=sum(p["pnl_twd"] for p in closed)
        if abs(realized-daily["realized_twd"].sum()) > 1e-5:
            raise AssertionError("trade and daily realized totals differ")
        final=daily.row(-1,named=True)
        eq=daily["official_equity_twd"].to_list()
        peak=0.;drawdown=0.
        for value in eq:
            if value is not None:
                peak=max(peak,value);drawdown=max(drawdown,peak-value)
        complete=all(v is not None for v in eq)
        s=dict(portfolio=name,sessions=len(days),available_sessions=len(manifest["available_days"]),
               realized_twd=realized,realized_per_available_day=realized/len(manifest["available_days"]),
               realized_per_calendar_session=realized/len(days),
               final_official_equity_twd=final["official_equity_twd"],
               equity_per_available_day=final["official_equity_twd"]/len(manifest["available_days"])
               if final["official_equity_twd"] is not None else None,
               final_official_marked_open_twd=final["official_marked_open_twd"],
               final_marked_open_bidask_twd=final["marked_open_twd"],
               final_unmarked_bidask=final["unmarked_positions"],
               final_unmarked_official=final["official_unmarked_positions"],
               final_carry_twd=final["carry_twd"],final_open_positions=final["open_positions"],
               final_unhedged=final["unhedged"],final_continuity_blocked=final["continuity_blocked"],
               maker_triggered_entries=len(entries),paired_entries=len(paired),partial_rollbacks=len(rollbacks),
               actual_nonpositive_basis=sum(p["actual_ab"] is not None and p["actual_ab"]<=0 for p in entries.values()),
               same_day_paired_fraction=sum(p["close_day"] == p["entry_day"] for p in paired)/len(paired) if paired else None,
               by_next_session_paired_fraction=sum(p["close_day"] is not None and
                   day_index[p["close_day"]]-day_index[p["entry_day"]]<=1 for p in paired)/len(paired) if paired else None,
               mean_paired_twd=sum(r["mean_paired_twd"] for r in usage)/len(days),
               mean_committed_twd=sum(r["mean_committed_twd"] for r in usage)/len(days),
               peak_committed_twd=max(r["peak_committed_twd"] for r in usage),
               peak_paired_twd=max(r["peak_paired_twd"] for r in usage),
               mean_paired_1000_twd=sum(times_10)/len(days),mean_paired_1100_twd=sum(times_11)/len(days),
               peak_eod_carry_twd=daily["carry_twd"].max(),days_eod_over_20M=daily.filter(pl.col("carry_twd")>20_000_000+1e-6).height,
               unreserved_fills=daily["unreserved_fills"].sum(),cap_overrun_events=daily["cap_overrun_events"].sum(),
               cap_overrun_seconds=sum(r["overrun_seconds"] for r in usage),
               max_equity_drawdown_twd=drawdown if complete else None,complete_daily_official_marks=complete,
               expiry_closed=len(expiry),expiry_realized_twd=sum(p["pnl_twd"] for p in closed if p["id"] in expiry),
               corporate_closed=sum(p["id"] in corporate for p in closed),
               corporate_realized_twd=sum(p["pnl_twd"] for p in closed if p["id"] in corporate),
               capital_days_twd=capital_days,funding_1pct_twd=capital_days*.01/365,
               funding_2pct_twd=capital_days*.02/365,funding_3pct_twd=capital_days*.03/365)
        summaries.append(s)
        previous_equity=0.
        for month in sorted({d[:6] for d in days}):
            subset=daily.filter(pl.col("day").str.starts_with(month))
            ending=subset["official_equity_twd"][-1]
            monthly.append(dict(portfolio=name,month=month,sessions=subset.height,
                                realized_twd=subset["realized_twd"].sum(),
                                equity_change_twd=ending-previous_equity if ending is not None and previous_equity is not None else None,
                                ending_equity_twd=ending,ending_carry_twd=subset["carry_twd"][-1]))
            previous_equity=ending
        for stream in ["S1","S2"]:
            selected=[p for p in entries.values() if p["stream"] == stream]
            stream_summary.append(dict(portfolio=name,stream=stream,entries=len(selected),
                                       realized_twd=sum(p["pnl_twd"] for p in selected),
                                       open_positions=sum(p["state"] != "closed" for p in selected)))
        for p in entries.values():
            trade_rows.append(dict(portfolio=name,position_id=p["id"],source_intent=p["source_intent"],stream=p["stream"],
                                   vc=p["contract"]["vc"],qc=p["contract"]["qc"],entry_day=p["entry_day"],
                                   close_day=p["close_day"],state=p["state"],nominal_twd=p["spot_buy_cash"]/10_000,
                                   pnl_twd=p["pnl_twd"],actual_ab=p["actual_ab"],expiry=p["id"] in expiry,
                                   corporate_exit=p["id"] in corporate,partial_rollback=p["id"] in rollbacks))
    for name,rows in [("summary",summaries),("monthly",monthly),("streams",stream_summary),("capacity_usage_daily",usage_rows)]:
        pl.from_dicts(rows,infer_schema_length=None).write_csv(root/f"{name}.csv")
    pl.from_dicts(trade_rows,infer_schema_length=None).write_parquet(root/"trades_summary.parquet")
    result=dict(summaries=summaries,monthly=monthly,streams=stream_summary)
    (root/"analysis.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    args=parser.parse_args()
    print(json.dumps(analyze(args.root)["summaries"],indent=2))
