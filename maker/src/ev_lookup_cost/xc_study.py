"""Execution-cost-aware EV study (ev_lookup_xc, 2026-09-09).

One process per execution mode; every mode owns an independent causal shadow
because the decay / hazard tables are learned from the shadow's own execution
rule. All portfolios: 20M cap, EV admission (est >= 0.6 lambda), no bpday gate,
hazard-split EV, hedge 50ms, cancel 50ms, taker cross disabled.

  second : legacy 1 Hz maintenance. base_20M (no execution cost, = Codex v21
           split_ev) vs xc_ev_20M (measured entry/exit decay + margin in EV).
  event  : xc_ev + relative re-peg (locked basis fell 10 bp below quote value
           -> event cancel + re-quote), no exit quotes in the first 5 minutes,
           event-driven exit guard on futures updates; with and without S1.
  expiry : event rules + carry positions ride to expiry (C8 accounting) instead
           of exiting at the frozen anchor-5 target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from time import monotonic

import polars as pl

from .forecast_calendar import calendar_spec
from .full_study import OUTAGES, checkpoint, restore
from .market import MarketDay, METADATA_ROOT
from .portfolio import Portfolio
from .replay import Replay

EVENT_RULES = dict(event_quotes=True, repeg_drop_bp=10.0, exit_open_delay_s=300, exit_event_guard=True)
MODES = {
    "second": dict(execution=dict(event_quotes=False),
                   portfolios=[("base_20M", dict(exec_cost=False)),
                               ("xc_ev_20M", dict(exec_cost=True))]),
    "event": dict(execution=EVENT_RULES,
                  portfolios=[("xc_full_20M", dict(exec_cost=True)),
                              ("xc_full_s2_20M", dict(exec_cost=True, use_s1=False))]),
    "expiry": dict(execution=dict(EVENT_RULES, carry_exit="expiry"),
                   portfolios=[("xc_expiry_20M", dict(exec_cost=True)),
                               ("xc_expiry_s2_20M", dict(exec_cost=True, use_s1=False))]),
}
COMMON = dict(cap_twd=20_000_000, use_ev=True, use_bpday=False, split_ev=True,
              enable_cross=False, hedge_ms=50, cancel_ms=50)


def build(mode: str, margin_bp: float) -> tuple[list[Portfolio], Portfolio]:
    spec = MODES[mode]
    execution = spec["execution"]
    actors = [Portfolio(name, **COMMON, **execution, decay_margin_bp=margin_bp, **flags)
              for name, flags in spec["portfolios"]]
    shadow = Portfolio("shadow", 10**12, use_ev=False, use_bpday=False, shadow=True,
                       split_ev=True, enable_cross=False, hedge_ms=50, cancel_ms=50, **execution)
    return actors, shadow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=list(MODES), required=True)
    parser.add_argument("--margin-bp", type=float, default=3.0)
    parser.add_argument("--days", nargs="+", help="Optional contiguous subset (smoke tests)")
    parser.add_argument("--end", default="20260902")
    parser.add_argument("--products", nargs="+")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    calendar = METADATA_ROOT / "calendar_20260504_20260902.parquet"
    days = (pl.read_parquet(calendar).filter(pl.col("DayType") == "TradeDay").sort("Date")
            ["Date"].dt.strftime("%Y%m%d").to_list())
    days = [d for d in days if d <= args.end]
    if args.days:
        days = [d for d in days if d in set(args.days)]
    if not days:
        parser.error("no sessions selected")
    actors, shadow = build(args.mode, args.margin_bp)
    replay = Replay(args.output, actors, products=args.products, shadow=shadow)
    manifest_path = args.output / "manifest.json"
    if args.resume:
        manifest = json.loads(manifest_path.read_text())
        for name, digest in manifest["sources"].items():
            if hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"source changed since checkpoint: {name}")
        restore(replay)
        if days[:len(replay.history.sessions)] != replay.history.sessions:
            raise ValueError("calendar changed since checkpoint")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "source_snapshot").mkdir()
        sources = {}
        for p in Path(__file__).parent.glob("*.py"):
            shutil.copy2(p, args.output / "source_snapshot" / p.name)
            sources[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        spec = MODES[args.mode]
        manifest = dict(version="xc_execution_cost", mode=args.mode, days=days,
                        available_days=[d for d in days if d not in OUTAGES],
                        data_outage_days=[d for d in days if d in OUTAGES],
                        execution=spec["execution"], portfolios=spec["portfolios"], common=COMMON,
                        decay_margin_bp=args.margin_bp, sources=sources, products=args.products,
                        status="running", finance_cost_included=False,
                        decay_tables="walk-forward shadow: entry = quote_ab - actual_ab by (stream, time, futures spread); "
                                     "exit = realized exit basis - (anchor-5) by (stream, same-day/carry); prior 20 sessions, "
                                     "n>=30 with pooled fallback, priors 15/5 bp, floored at 0, plus margin",
                        ev="p_sd*(eff_u-d_in+5-d_sd-20) + p_overnight*(eff_u-d_in+5-d_on-34) + p_expiry*(ab-d_in-34) + p_other*other",
                        forecast_calendar=calendar_spec(),
                        expiry="C8 basis-zero accounting after exact expiry",
                        s1="S0 commands through 20260813; S2 only thereafter")
    manifest["status"] = "running"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        for day in days[len(replay.history.sessions):]:
            start = monotonic()
            rows = replay.day(day, data_outage=day in OUTAGES)
            checkpoint(replay)
            print(json.dumps(dict(day=day, seconds=round(monotonic() - start, 2), results=[
                {k: row.get(k) for k in ("portfolio", "fills", "fills_s1", "fills_s2", "negative_basis",
                                         "same_day", "realized_twd", "official_equity_twd", "carry_twd",
                                         "event_cancels", "drift_cancels", "requotes", "exit_guard_cancels")}
                for row in rows])), flush=True)
    except Exception as error:
        manifest.update(status="failed", failure_type=type(error).__name__,
                        completed_days=replay.history.sessions)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        raise
    manifest.update(status="completed",
                    peak_committed_twd={a.name: a.ledger.peak_cents / 100 for a in actors},
                    unhedged_final={a.name: a.daily[-1]["unhedged"] for a in actors})
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
