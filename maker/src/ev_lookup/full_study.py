"""Causal full-period capacity study: fixed, parked, and historical-release admission."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
from time import monotonic

import polars as pl

from .causal_lookup import (CapacityObservation, DayEntryOutcome, ExitRiskObservation,
                            ResolvedOutcome)
from .market import Contract, METADATA_ROOT
from .portfolio import Portfolio, Position
from .replay import Replay

CONFIGS = [
    dict(name="ev_bpday_20M", cap_twd=20_000_000, use_ev=True, use_bpday=True),
    dict(name="ev_20M", cap_twd=20_000_000, use_ev=True, use_bpday=False),
    dict(name="fcfs_20M", cap_twd=20_000_000, use_ev=False, use_bpday=False),
    dict(name="ev_bpday_park_20M", cap_twd=20_000_000, use_ev=True, use_bpday=True, park_spot=True),
    dict(name="ev_bpday_adaptive_25M", cap_twd=25_000_000, use_ev=True, use_bpday=True,
         park_spot=True, overnight_target_twd=20_000_000),
    dict(name="ev_bpday_adaptive_30M", cap_twd=30_000_000, use_ev=True, use_bpday=True,
         park_spot=True, overnight_target_twd=20_000_000),
]
OUTAGES = ["20260828"]


def checkpoint(replay: Replay) -> None:
    actors = []
    for a in replay.actors:
        actors.append(dict(name=a.name, positions=[asdict(p) for p in a.positions.values()],
                           amounts=a.ledger.amounts, committed_cents=a.ledger.committed_cents,
                           peak_cents=a.ledger.peak_cents, last_ns=a.ledger.last_ns,
                           daily=a.daily, rejected_by_day=a.rejected_by_day, counter=a._counter))
    state = dict(sessions=replay.history.sessions,
                 resolutions=[asdict(r) for r in replay.history.resolutions],
                 entry_days=[asdict(r) for r in replay.history.entry_days], actors=actors,
                 capacity_observations=[asdict(r) for r in replay.history.capacity_observations],
                 exit_risks=[asdict(r) for r in replay.history.exit_risks])
    temp = replay.output / "checkpoint.tmp.json"
    temp.write_text(json.dumps(state) + "\n")
    temp.replace(replay.output / "checkpoint.json")


def restore(replay: Replay) -> None:
    state = json.loads((replay.output / "checkpoint.json").read_text())
    replay.history.sessions = state["sessions"]
    replay.history.resolutions = [ResolvedOutcome(**r) for r in state["resolutions"]]
    replay.history.entry_days = [DayEntryOutcome(**r) for r in state["entry_days"]]
    replay.history.capacity_observations = [CapacityObservation(**r) for r in state["capacity_observations"]]
    replay.history._resolved_ids = {r.position_id for r in replay.history.resolutions}
    replay.history._entry_ids = {r.position_id for r in replay.history.entry_days}
    replay.history.exit_risks = [ExitRiskObservation(**r) for r in state.get("exit_risks", [])]
    replay.history._exit_ids = {r.position_id for r in replay.history.exit_risks}
    replay.snapshots = json.loads((replay.output / "snapshots.json").read_text())
    for a, saved in zip(replay.actors, state["actors"], strict=True):
        if a.name != saved["name"]:
            raise ValueError("checkpoint policy order mismatch")
        for row in saved["positions"]:
            row["contract"] = Contract(**row["contract"])
            p = Position(**row)
            a.positions[p.id] = p
        a.active = set(a.positions)
        a.ledger.amounts = saved["amounts"]
        for key in ("committed_cents", "peak_cents", "last_ns"):
            setattr(a.ledger, key, saved[key])
        a.daily, a.rejected_by_day, a._counter = saved["daily"], saved["rejected_by_day"], saved["counter"]
        if sum(a.ledger.amounts.values()) != a.ledger.committed_cents:
            raise AssertionError("checkpoint capacity does not reconcile")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--configs", nargs="+", help="Subset of configuration names to run")
    args = parser.parse_args()
    if args.configs:
        unknown = set(args.configs) - {c["name"] for c in CONFIGS}
        if unknown:
            parser.error(f"unknown configs: {sorted(unknown)}")
        CONFIGS[:] = [c for c in CONFIGS if c["name"] in args.configs]
    calendar = METADATA_ROOT / "calendar_20260504_20260902.parquet"
    days = (pl.read_parquet(calendar).filter(pl.col("DayType") == "TradeDay")
            .sort("Date")["Date"].dt.strftime("%Y%m%d").to_list())
    actors = [Portfolio(**config, hedge_ms=50, cancel_ms=50, split_ev=False,
                        event_quotes=False, enable_cross=False) for config in CONFIGS]
    replay = Replay(args.output, actors)
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
        source_dir = args.output / "source_snapshot"
        source_dir.mkdir()
        sources = {}
        for source in Path(__file__).parent.glob("*.py"):
            shutil.copy2(source, source_dir / source.name)
            sources[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
        inputs = {}
        input_dir = args.output / "input_snapshot"
        input_dir.mkdir()
        for source in [calendar, METADATA_ROOT / "announcements/index.json",
                       METADATA_ROOT / "official_future_daily_marks.parquet"]:
            shutil.copy2(source, input_dir / source.name)
            inputs[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        manifest = dict(version="v20_capacity_full", days=days, available_days=[d for d in days if d not in OUTAGES],
                        data_outage_days=OUTAGES, calendar=str(calendar), configurations=CONFIGS,
                        caps=[20_000_000, 25_000_000, 30_000_000], products=None, hedge_ms=50, cancel_ms=50,
                        sources=sources, inputs_sha256=inputs, status="running", finance_cost_included=False,
                        expiry="C8 basis-zero accounting after exact expiry; blocked identities retain capital",
                        missing_contract="quarantine until explicit corporate-action conversion; no synthetic release",
                        corporate_risk="after official publication, stop new entries and taker-close both legs before adjustment",
                        adaptive_capacity="20M target plus Wilson lower-bound expected releases, prior 20 completed sessions only; 25M/30M research ceilings",
                        parking="S1 below B1 keeps queue without reservation; raw top changes activate or cancel; jump/race fills may exceed ceiling and remain booked",
                        valuation="end-of-session bid/ask marks, full round-trip cost, missing marks remain null",
                        s1="S0 commands through 20260813; S2 only thereafter")
    manifest["status"] = "running"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        for day in days[len(replay.history.sessions):]:
            start = monotonic()
            rows = replay.day(day, data_outage=day in OUTAGES)
            checkpoint(replay)
            print(json.dumps(dict(day=day, seconds=round(monotonic() - start, 2), results=rows)), flush=True)
    except Exception as error:
        manifest.update(status="failed", failure_type=type(error).__name__,
                        completed_days=replay.history.sessions)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        raise
    manifest.update(status="completed", peak_committed_twd={a.name: a.ledger.peak_cents / 100 for a in actors},
                    unhedged_final={a.name: a.daily[-1]["unhedged"] for a in actors})
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
