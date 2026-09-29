"""v19 entry point. Historical v18 is preserved under archive_v18/."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from time import monotonic

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[6]))
    __package__ = "src.research.futures_spot_spread.maker.src.ev_lookup"

from .market import MAKER_ROOT, WF
from .portfolio import Portfolio
from .replay import Replay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", nargs="+", help="Contiguous trading sessions; default all canonical days")
    parser.add_argument("--products", nargs="+", help="Optional diagnostic subset")
    parser.add_argument("--caps", nargs="+", type=int, default=[20_000_000])
    parser.add_argument("--policies", nargs="+", choices=["ev_bpday", "ev", "fcfs"], default=["ev_bpday"])
    parser.add_argument("--output", type=Path, default=MAKER_ROOT / "data/ev_lookup_v19_20260908")
    parser.add_argument("--hedge-ms", type=int, default=50)
    parser.add_argument("--cancel-ms", type=int, default=0, help="V0 default: immediate observed cancellation")
    args = parser.parse_args()
    if args.hedge_ms < 0 or not 0 <= args.cancel_ms < 1000:
        parser.error("invalid hedge/cancel latency")
    if any(c <= 0 or c % 1_000_000 for c in args.caps) or len(set(args.caps)) != len(args.caps):
        parser.error("caps must be distinct positive integer millions of TWD")
    if len(set(args.policies)) != len(args.policies):
        parser.error("policies must be distinct")
    days = args.days or sorted(p.name[5:] for p in (WF / "daily").glob("Date=*")
                               if "20260504" <= p.name[5:] <= "20260813")
    if days != sorted(set(days)):
        parser.error("days must be unique and chronological")
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in Path(__file__).parent.glob("*.py")}
    source_dir = args.output / "source_snapshot"
    source_dir.mkdir()
    for name in sources:
        shutil.copy2(Path(__file__).parent / name, source_dir / name)
    manifest = dict(version="v19", days=days, products=args.products, caps=args.caps,
                    policies=args.policies, hedge_ms=args.hedge_ms, cancel_ms=args.cancel_ms,
                    sources=sources, status="running",
                    capacity="spot entry nominal plus pending maximum nominal; cents",
                    expiry="C8 basis-zero accounting after exact contract expiry",
                    maker_fill="shared printed-volume FIFO, no queue cancellation credit",
                    exit_quote="first causal touch; recheck frozen maker price against current future hedge every second",
                    s1="all S0 nominal new/cancel commands; no outcome labels",
                    signal_frequency_hz=1, finance_cost_included=False)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    actors = [Portfolio(f"{policy}_{cap // 1_000_000}M", cap,
                        use_ev=policy != "fcfs", use_bpday=policy == "ev_bpday",
                        hedge_ms=args.hedge_ms, cancel_ms=args.cancel_ms,
                        split_ev=False, event_quotes=False, enable_cross=False)
              for cap in args.caps for policy in args.policies]
    replay = Replay(args.output, actors, products=args.products)
    try:
        for day in days:
            start = monotonic()
            rows = replay.day(day)
            print(json.dumps(dict(day=day, seconds=round(monotonic() - start, 2), results=rows)), flush=True)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failure_type"] = type(error).__name__
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        raise
    manifest["status"] = "completed"
    manifest["peak_committed_twd"] = {p.name: p.ledger.peak_cents / 100 for p in actors}
    manifest["unhedged_final"] = {p.name: p.daily[-1]["unhedged"] for p in actors}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
