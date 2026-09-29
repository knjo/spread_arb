"""Separate observable pre-print drift from post-fill hedge movement."""
import json

import polars as pl

from ...ev_lookup.audit.execution_costs import filled_positions
from ...ev_lookup.audit.report_v21 import maker_message_load


KINDS=["s2_fill_liquidity","maker_fill","taker_fill","cancel","quote","event_cancel"]


def breakdown(root):
    rows, latencies, messages=[] ,[],[]
    for mode in ("plain","guard"):
        folder=root/mode
        manifest=json.loads((folder/"manifest.json").read_text())
        for config in manifest["configurations"]:
            name=config["name"]
            positions=filled_positions(folder,name,manifest["days"])
            at_fill,ready={},{}
            for day in manifest["days"]:
                path=folder/f"Date={day}"/name/"execution.parquet"
                trace=(pl.scan_parquet(path).filter(pl.col("kind").is_in(KINDS)).collect().to_dicts()
                       if path.exists() else [])
                messages.append(dict(mode=mode,portfolio=name,**maker_message_load(day,trace)))
                for t in trace:
                    pid=t["position_id"]
                    if t["kind"]=="s2_fill_liquidity":
                        at_fill[pid]=t
                    elif t["kind"]=="maker_fill":
                        purpose=(("entry_spot" if t["stream"]=="S2" else "entry_future")
                                 if t["purpose"]=="entry" else "exit_future")
                        ready[pid,purpose]=t["ns"]
                    elif t["kind"]=="cancel" and t["purpose"]=="entry":
                        ready[pid,"entry_rollback"]=t["ns"]
                    elif t["kind"]=="taker_fill":
                        start=ready.get((pid,t["purpose"]))
                        if start is not None:
                            latencies.append(dict(mode=mode,portfolio=name,stream=t["stream"],day=day,
                                position_id=pid,purpose=t["purpose"],ready_ns=start,fill_ns=t["ns"],
                                latency_ms=(t["ns"]-start)/1e6))
                        if t["purpose"]=="exit_spot_remainder":
                            ready[pid,"exit_future"]=t["ns"]
            for p in positions.filter(pl.col("stream")=="S2").iter_rows(named=True):
                t=at_fill[p["id"]]
                before=t.get("depth_vwap")
                actual=p["spot_buy_cash"]/p["spot_buy_qty"]/10_000 if p["spot_buy_qty"] else None
                price=p["future_sell_cash"]/p["future_sell_qty"]/p["contract"]["shares"]/10_000
                pre_basis=(price/before-1)*10_000 if before else None
                if before and t["book_ns"]>=p["entry_fill_ns"]:
                    raise AssertionError("pre-print diagnostic used simultaneous or later market data")
                pre_decay=p["quote_ab"]-pre_basis if pre_basis is not None else None
                post_decay=pre_basis-p["actual_ab"] if pre_basis is not None and p["actual_ab"] is not None else None
                total=p["quote_ab"]-p["actual_ab"] if p["actual_ab"] is not None else None
                if pre_decay is not None and post_decay is not None and abs(pre_decay+post_decay-total)>1e-8:
                    raise AssertionError("entry cost decomposition does not reconcile")
                rows.append(dict(mode=mode,portfolio=name,position_id=p["id"],day=p["entry_day"],
                    preprint_reason=t["reason"],preprint_depth_vwap=before,preprint_adverse_vwap=t.get("adverse_vwap"),
                    actual_hedge_price=actual,hedge_ms=(p["hedged_ns"]-p["entry_fill_ns"])/1e6 if p["hedged_ns"] else None,
                    prefill_decay_bp=pre_decay,postfill_basis_decay_bp=post_decay,total_entry_decay_bp=total,
                    spot_move_bp=(actual/before-1)*10_000 if before and actual else None))
    fills=pl.from_dicts(rows,infer_schema_length=None)
    fills.write_parquet(root/"s2_execution_breakdown.parquet")
    diagnostics=[]
    for (mode,name),sample in fills.partition_by(["mode","portfolio"],as_dict=True).items():
        eligible=sample.filter(pl.col("spot_move_bp").is_not_null())
        for label,frame in [("all_priceable",eligible),("exact_50ms",eligible.filter((pl.col("hedge_ms")-50).abs()<1e-6))]:
            worse=frame.filter(pl.col("spot_move_bp")>1e-7)
            diagnostics.append(dict(mode=mode,portfolio=name,sample=label,total_s2_fills=sample.height,
                priceable=frame.height,unpriceable_s2_fills=sample.height-eligible.height,
                worse_fraction=worse.height/frame.height if frame.height else None,
                more_than_one_tick_fraction=frame.filter(pl.col("actual_hedge_price")>pl.col("preprint_adverse_vwap")+1e-8).height/frame.height if frame.height else None,
                prefill_decay_bp=frame["prefill_decay_bp"].mean(),postfill_basis_decay_bp=frame["postfill_basis_decay_bp"].mean(),
                total_entry_decay_bp=frame["total_entry_decay_bp"].mean(),mean_spot_move_bp=frame["spot_move_bp"].mean(),
                p95_spot_move_bp=frame["spot_move_bp"].quantile(.95),mean_spot_move_given_worse=worse["spot_move_bp"].mean()))
    pl.from_dicts(diagnostics,infer_schema_length=None).write_csv(root/"s2_hedge_diagnostic.csv")
    latency=pl.from_dicts(latencies,infer_schema_length=None)
    latency.write_parquet(root/"hedge_latencies.parquet")
    summary=latency.group_by("mode","portfolio","stream","purpose").agg(pl.len().alias("fills"),
        pl.col("latency_ms").median().alias("median_ms"),pl.col("latency_ms").quantile(.99).alias("p99_ms"),
        pl.col("latency_ms").max().alias("max_ms"),(pl.col("latency_ms")>50.000001).sum().alias("later_than_50ms"))
    summary.write_csv(root/"hedge_latency_summary.csv")
    pl.from_dicts(messages).write_csv(root/"maker_message_load.csv")
    return diagnostics
