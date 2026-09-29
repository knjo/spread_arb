"""Build point tables for every canonical-grid session (skips days already built).

    uv run --project /home/kevin/Project/HFT --no-sync python -m spreadArb.src.points.build_all --stream s2 [--start D --end D] [--force]
"""
from __future__ import annotations

import argparse
import json
import time
import traceback

from ..common.paths import grid_days, points_path
from . import s1, s2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", choices=["s1", "s2"], required=True)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    module = {"s1": s1, "s2": s2}[args.stream]
    name = f"{args.stream}_entries"
    days = [d for d in grid_days() if (not args.start or d >= args.start) and (not args.end or d <= args.end)]
    for day in days:
        if not args.force and points_path(day, name).exists():
            continue
        started = time.time()
        try:
            frame = module.build_day(day)
            print(json.dumps(dict(stream=args.stream, day=day, rows=frame.height, products=frame["vc"].n_unique(),
                                  elapsed_s=round(time.time() - started, 1))), flush=True)
        except Exception as error:   # keep the batch going; the manifest of a failed day is absent
            print(json.dumps(dict(stream=args.stream, day=day, error=repr(error))), flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
