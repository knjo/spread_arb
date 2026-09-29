"""Build the Stage 1 cache: per-session level histograms and reach facts.

    uv run --project /home/kevin/Project/HFT --no-sync python -m spreadArb.src.ev.build --before 20260706 --days 22
"""
from __future__ import annotations

import argparse
import json
import time

from ..common.paths import QCACHE, facts_path, grid_days, hist_path
from .reach import GridCache, build_day


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--before", help="build the N sessions before this day")
    parser.add_argument("--days", type=int, default=22)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    days = grid_days()
    if args.before:
        todo = [d for d in days if d < args.before][-args.days:]
    else:
        todo = [d for d in days if (not args.start or d >= args.start) and (not args.end or d <= args.end)]
    cache = GridCache(keep=3)
    manifest_path = QCACHE / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for day in todo:
        if not args.force and facts_path(day).exists() and hist_path(day).exists():
            continue
        started = time.time()
        facts = build_day(day, cache, days)
        manifest[day] = {"facts_rows": facts.height, "elapsed_s": round(time.time() - started, 1)}
        print(json.dumps({"day": day, **manifest[day]}), flush=True)
        QCACHE.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
