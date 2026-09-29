"""Fixed S2 inside-only depth/buffer ablation, with independent causal shadows."""
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
from .market import MarketDay, METADATA_ROOT
from .replay import Replay


RULES = {
    "inside": dict(enabled=False, min_a1_multiple=0.0, adverse_probability=0.0, adverse_ticks=1),
    "depth5": dict(enabled=True, min_a1_multiple=5.0, adverse_probability=0.0, adverse_ticks=1),
    "buffer50": dict(enabled=True, min_a1_multiple=0.0, adverse_probability=0.5, adverse_ticks=1),
    "depth5_buffer50": dict(enabled=True, min_a1_multiple=5.0, adverse_probability=0.5, adverse_ticks=1),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--variants", nargs="+", choices=list(RULES), default=list(RULES))
    parser.add_argument("--end", default="20260902")
    parser.add_argument("--products", nargs="+")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    calendar = METADATA_ROOT/"calendar_20260504_20260902.parquet"
    days = (pl.read_parquet(calendar).filter(pl.col("DayType") == "TradeDay").sort("Date")
            ["Date"].dt.strftime("%Y%m%d").to_list())
    days = [d for d in days if d <= args.end]
    if not days:
        raise ValueError("empty study calendar")
    replays, manifests = [], []
    for variant in args.variants:
        root = args.output/variant
        config = dict(name=variant+"_20M", cap_twd=20_000_000, use_ev=True,
            use_bpday=False, split_ev=True, event_quotes=True, enable_cross=False,
            liquidity_rule=RULES[variant])
        actor = LiquidityPortfolio(**config, hedge_ms=50, cancel_ms=50)
        shadow = LiquidityPortfolio("shadow", 10**12, use_ev=False, use_bpday=False,
            shadow=True, hedge_ms=50, cancel_ms=50, liquidity_rule=RULES[variant])
        replay = Replay(root, [actor], products=args.products, shadow=shadow)
        if args.resume:
            manifest = json.loads((root/"manifest.json").read_text())
            if (manifest["days"] != days or manifest["products"] != args.products
                    or manifest["configurations"] != [config]):
                raise ValueError("resume configuration changed")
            for name, digest in manifest["sources"].items():
                if hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() != digest:
                    raise ValueError(f"source changed: {name}")
            restore(replay)
        else:
            root.mkdir(parents=True, exist_ok=False)
            (root/"source_snapshot").mkdir()
            sources = {}
            for p in Path(__file__).parent.glob("*.py"):
                shutil.copy2(p, root/"source_snapshot"/p.name)
                sources[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
            (root/"input_snapshot").mkdir()
            inputs = {}
            for p in (calendar, METADATA_ROOT/"announcements/index.json",
                      METADATA_ROOT/"official_future_daily_marks.parquet"):
                shutil.copy2(p, root/"input_snapshot"/p.name)
                inputs[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
            manifest = dict(version="v22_s2_liquidity", status="running", days=days,
                available_days=[d for d in days if d not in OUTAGES],
                data_outage_days=[d for d in days if d in OUTAGES], configurations=[config],
                products=args.products, sources=sources, inputs_sha256=inputs,
                hedge_ms=50, cancel_ms=50, finance_cost_included=False,
                forecast_calendar=calendar_spec(), depth_events=True,
                s2="new orders A1 minus one legal tick; later external asks at our price stay behind; cancel on a lower ask or crossed bid; no back-of-book quotes",
                liquidity="available A1 shares / contract hedge shares; full five-level preview without consuming depth",
                buffer="raw EV minus known depth basis loss and fixed 50% one-spot-tick scenario; no synthetic PnL deduction",
                shadow="independent per variant; no EV/cap/bpday gate, same inside/depth rule; common shadow opportunity scheduler retained",
                exit_policy="v21 frozen quote anchor minus 5bp, unchanged actual four-leg accounting; C8 only after expiry",
                execution="unthrottled, immediate new-order queue entry; cancel and necessary hedge delays 50ms; races retained",
                selection="four prespecified hypotheses after inspecting v21; exploratory historical replay, not untouched validation",
                s1="original S0 new commands end 20260813; thereafter S2 new only")
        replays.append(replay)
        manifests.append(manifest)
        (root/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    if len({tuple(r.history.sessions) for r in replays}) != 1:
        raise ValueError("variant checkpoints have different completed dates")
    for day in days[len(replays[0].history.sessions):]:
        start = monotonic()
        carry = list({p.contract.qc:p.contract for r in replays for a in r.actors
                      for p in a.positions.values() if p.id in a.active}.values())
        market = MarketDay(day, args.output, args.products, carry,
                           data_outage=day in OUTAGES, depth_events=True)
        result = {}
        for replay in replays:
            rows = replay.day(day, data_outage=day in OUTAGES, market=market)
            checkpoint(replay)
            result[replay.output.name] = [{k:r[k] for k in ("portfolio", "fills", "fills_s2",
                "realized_twd", "official_equity_twd", "carry_twd", "event_cancels")}
                for r in rows if r["portfolio"] != "shadow"]
        print(json.dumps(dict(day=day, seconds=round(monotonic()-start, 2), results=result)), flush=True)
        del market
    for replay, manifest in zip(replays, manifests):
        manifest.update(status="completed", peak_committed_twd={a.name:a.ledger.peak_cents/100 for a in replay.portfolios})
        (replay.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
