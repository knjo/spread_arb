"""First-priority, cost-aware EV replay with an independent shadow per execution mode."""
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
from .liquidity_portfolio import LiquidityPortfolio
from .market import METADATA_ROOT, MarketDay
from .replay import Replay


RULE = dict(enabled=True, min_a1_multiple=5.0, adverse_probability=0.5, adverse_ticks=1)
COMMON = dict(cap_twd=20_000_000, use_ev=True, use_bpday=False, split_ev=True,
              event_quotes=True, enable_cross=False, hedge_ms=50, cancel_ms=50,
              liquidity_rule=RULE, carry_exit="target", decay_margin_bp=3.0)


def configurations(mode: str) -> list[dict]:
    if mode == "plain":
        return [dict(COMMON, name="fixed_20M", exec_cost=False),
                dict(COMMON, name="cost_20M", exec_cost=True)]
    if mode == "guard":
        return [dict(COMMON, name="cost_guard_20M", exec_cost=True,
                     repeg_drop_bp=10.0, exit_open_delay_s=300, exit_event_guard=True)]
    raise ValueError("unknown execution mode")


def build(mode: str, output: Path, products=None) -> Replay:
    configs = configurations(mode)
    actors = [LiquidityPortfolio(**c) for c in configs]
    shadow_config = dict(configs[-1], name="shadow", cap_twd=10**12,
                         use_ev=False, use_bpday=False, shadow=True, exec_cost=False)
    return Replay(output, actors, products=products, shadow=LiquidityPortfolio(**shadow_config))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    temp = path.with_suffix(".tmp.json")
    temp.write_text(json.dumps(value, indent=2)+"\n")
    temp.replace(path)


def validate_resume(manifest: dict, spec: dict, source: Path) -> None:
    for key, value in spec.items():
        if manifest[key] != value:
            raise ValueError(f"resume configuration changed: {key}")
    for name, expected in manifest["sources"].items():
        if digest(source/name) != expected:
            raise ValueError(f"source changed: {name}")
    for name, expected in manifest["inputs_sha256"].items():
        if digest(Path(name)) != expected:
            raise ValueError(f"input metadata changed: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", required=True, choices=["plain", "guard"])
    parser.add_argument("--days", nargs="+", help="Contiguous smoke-test subset; never a warm checkpoint substitution")
    parser.add_argument("--products", nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-days", type=int, help="Operational limit only; preserve the full manifest plan")
    args = parser.parse_args()
    calendar = METADATA_ROOT/"calendar_20260504_20260902.parquet"
    days = (pl.read_parquet(calendar).filter(pl.col("DayType") == "TradeDay").sort("Date")
            ["Date"].dt.strftime("%Y%m%d").to_list())
    if args.days:
        selected = [d for d in days if d in set(args.days)]
        if not selected or selected != days[days.index(selected[0]):days.index(selected[-1])+1]:
            parser.error("smoke days must form a contiguous calendar interval")
        days = selected
    if args.max_days is not None and args.max_days <= 0:
        parser.error("max-days must be positive")
    source = Path(__file__).parent
    spec = dict(version="v23_inside_execution_cost", mode=args.mode, days=days,
                products=args.products, configurations=configurations(args.mode))
    replay = build(args.mode, args.output, args.products)
    manifest_path = args.output/"manifest.json"
    if args.resume:
        manifest = json.loads(manifest_path.read_text())
        validate_resume(manifest, spec, source)
        restore(replay)
        if replay.history.sessions != days[:len(replay.history.sessions)]:
            raise ValueError("checkpoint calendar differs")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/"source_snapshot").mkdir()
        sources = {}
        for p in sorted(source.glob("*.py")):
            shutil.copy2(p, args.output/"source_snapshot"/p.name)
            sources[p.name] = digest(p)
        (args.output/"input_snapshot").mkdir()
        inputs = {}
        for p in (calendar, METADATA_ROOT/"announcements/index.json",
                  METADATA_ROOT/"official_future_daily_marks.parquet"):
            shutil.copy2(p, args.output/"input_snapshot"/p.name)
            inputs[str(p)] = digest(p)
        manifest = dict(spec, status="running", available_days=[d for d in days if d not in OUTAGES],
            data_outage_days=[d for d in days if d in OUTAGES], sources=sources, inputs_sha256=inputs,
            hedge_ms=50, cancel_ms=50, finance_cost_included=False, depth_events=True,
            forecast_calendar=calendar_spec(),
            s2="Every new futures ask improves public A1 by one legal tick. Insufficient depth/EV waits. No backward repricing.",
            execution="Immediate new queue entry, unthrottled; cancel and mandatory hedge 50ms, races retained.",
            costs="Prior 20 completed shadow sessions. Entry=max(mean decay floored at zero + 3bp, fixed depth/50% tick floor); exit=(F buy-S sell)/S entry cash*1e4-(anchor-5), mean floored at zero+3bp. Priors 15/5bp, n>=30 pooled fallback.",
            other="Historical other-exit net already includes execution costs; no second subtraction.",
            exit_policy="Frozen entry anchor minus 5bp; expiry basis zero only after actual original expiry.",
            s1="Causal S0 commands through 20260813; S2 new entries only thereafter.",
            selection="Exploratory historical study: thresholds/priors chosen after inspecting the same period, not untouched validation.",
            limitations="P_sd still counts same-day partial rollbacks as same-day closure; measured separately. No live acknowledgment/rate calibration.")
    manifest["status"] = "running"
    write_json(manifest_path, manifest)
    pending = days[len(replay.history.sessions):]
    if args.max_days is not None:
        pending = pending[:args.max_days]
    try:
        for day in pending:
            start = monotonic()
            carry = list({p.contract.qc:p.contract for a in replay.actors
                          for p in a.positions.values() if p.id in a.active}.values())
            market = MarketDay(day, args.output, args.products, carry,
                               data_outage=day in OUTAGES, depth_events=True)
            rows = replay.day(day, data_outage=day in OUTAGES, market=market)
            checkpoint(replay)
            print(json.dumps(dict(day=day, seconds=round(monotonic()-start, 2), results=[
                {k:r.get(k) for k in ("portfolio", "fills", "fills_s1", "fills_s2", "realized_twd",
                                      "official_equity_twd", "carry_twd", "drift_cancels", "exit_guard_cancels")}
                for r in rows if r["portfolio"] != "shadow"])), flush=True)
            del market
    except Exception as error:
        manifest.update(status="failed", failure_type=type(error).__name__, completed_days=replay.history.sessions)
        write_json(manifest_path, manifest)
        raise
    manifest.update(status="completed" if replay.history.sessions == days else "checkpointed",
                    completed_days=replay.history.sessions,
                    peak_committed_twd={a.name:a.ledger.peak_cents/100 for a in replay.portfolios})
    write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
