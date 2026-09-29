"""Matched 20M EV/cancellation ablation, independent shadows per execution rule."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from time import monotonic

import polars as pl

from .full_study import OUTAGES, checkpoint, restore
from .market import MarketDay, METADATA_ROOT
from .portfolio import Portfolio
from .replay import Replay
from .forecast_calendar import calendar_spec


def configs(event_quotes: bool) -> list[dict]:
    return [dict(name=name, cap_twd=20_000_000, use_ev=True, use_bpday=bp,
                 split_ev=split, event_quotes=event_quotes, enable_cross=False)
            for name, split, bp in [("legacy_ev_20M", False, False),
                                    ("split_ev_20M", True, False),
                                    ("split_bpday_20M", True, True)]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--end", default="20260902")
    parser.add_argument("--products", nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--modes", nargs="+", choices=["second", "event"], default=["second", "event"])
    args = parser.parse_args()
    calendar = METADATA_ROOT / "calendar_20260504_20260902.parquet"
    days = (pl.read_parquet(calendar).filter(pl.col("DayType") == "TradeDay")
            .sort("Date")["Date"].dt.strftime("%Y%m%d").to_list())
    days = [d for d in days if d <= args.end]
    if not args.resume:
        args.output.mkdir(parents=True, exist_ok=True)
    replays, manifests = [], []
    for mode in args.modes:
        root = args.output / mode
        cs = configs(mode == "event")
        replay = Replay(root, [Portfolio(**c, hedge_ms=50, cancel_ms=50) for c in cs], products=args.products)
        if args.resume:
            manifest = json.loads((root / "manifest.json").read_text())
            if manifest["days"] != days or manifest["products"] != args.products:
                raise ValueError("resume calendar/products differ")
            for name, digest in manifest["sources"].items():
                if hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() != digest:
                    raise ValueError(f"source changed: {name}")
            restore(replay)
        else:
            root.mkdir()
            (root / "source_snapshot").mkdir()
            sources = {}
            for p in Path(__file__).parent.glob("*.py"):
                shutil.copy2(p, root / "source_snapshot" / p.name)
                sources[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
            (root / "input_snapshot").mkdir()
            inputs = {}
            for p in (calendar, METADATA_ROOT / "announcements/index.json",
                      METADATA_ROOT / "official_future_daily_marks.parquet"):
                shutil.copy2(p, root / "input_snapshot" / p.name)
                inputs[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
            manifest = dict(version="v21_ev_event", days=days, available_days=[d for d in days if d not in OUTAGES],
                data_outage_days=[d for d in days if d in OUTAGES], configurations=cs,
                sources=sources, inputs_sha256=inputs, products=args.products, status="running",
                hedge_ms=50, cancel_ms=50, finance_cost_included=False,
                expiry="C8 zero-basis accounting only after exact expiry; no change to ordinary exits",
                exit_policy="fixed quote-time anchor minus 5bp; experimental taker cross disabled",
                hazard="prior 20 completed sessions; carry paired at open; unresolved exposures retained in denominator",
                hazard_fallback=">=30 exposures by stream/expiry bucket, stream, pooled; otherwise fixed 0.5 target-exit hazard/session",
                forecast_calendar=calendar_spec(),
                s2="raw book top/validity event cancel, 50ms effective; reprice back to anchor+25 if inside no longer qualifies"
                   if mode == "event" else "legacy 1Hz decisions; event invalidation audit has no execution effect",
                shadow="independent by execution mode; shared replacement intent emitted after all same-time cancels",
                s1="original S0 commands end 20260813; thereafter S2 new entries only")
        manifests.append(manifest)
        replays.append(replay)
    if len({tuple(r.history.sessions) for r in replays}) != 1:
        raise ValueError("mode checkpoints differ; recover the interrupted day before resuming")
    for replay, manifest in zip(replays, manifests):
        (replay.output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    for day in days[len(replays[0].history.sessions):]:
        start = monotonic()
        carry = list({p.contract.qc: p.contract for r in replays for a in r.actors
                      for p in a.positions.values() if p.id in a.active}.values())
        market = MarketDay(day, args.output, args.products, carry, data_outage=day in OUTAGES)
        result = {}
        for replay in replays:
            rows = replay.day(day, data_outage=day in OUTAGES, market=market)
            checkpoint(replay)
            result[replay.output.name] = [{k: row[k] for k in
                ("portfolio", "fills", "realized_twd", "official_equity_twd", "carry_twd", "event_cancels")}
                for row in rows if row["portfolio"] != "shadow"]
        print(json.dumps(dict(day=day, seconds=round(monotonic()-start, 2), results=result)), flush=True)
        del market
    for replay, manifest in zip(replays, manifests):
        manifest.update(status="completed", peak_committed_twd={a.name: a.ledger.peak_cents/100 for a in replay.portfolios})
        (replay.output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
