"""CLI for the source-bound current-ladder D+1 fresh-book benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .current_ladder_overnight import (
    DEFAULT_ENTRY_ROOT,
    DEFAULT_EXIT_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PREREQUISITE_ROOT,
    build_current_ladder_overnight_plan,
    run_current_ladder_overnight,
    verify_current_ladder_overnight_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the manifest-derived 60-session current-ladder strict D+1 "
            "first-joint spot-bid/future-ask benchmark with book age <=1s"
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--preflight-only",
        action="store_true",
        help="verify immutable sources and print the derived cohort without raw replay",
    )
    mode.add_argument(
        "--verify-only",
        type=Path,
        metavar="OUTPUT",
        help="source-rebind and verify an existing completed output",
    )
    parser.add_argument("--entry-root", type=Path, default=DEFAULT_ENTRY_ROOT)
    parser.add_argument("--exit-root", type=Path, default=DEFAULT_EXIT_ROOT)
    parser.add_argument(
        "--prerequisite-root", type=Path, default=DEFAULT_PREREQUISITE_ROOT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    common = {
        "entry_root": args.entry_root,
        "exit_root": args.exit_root,
        "prerequisite_root": args.prerequisite_root,
    }
    if args.preflight_only:
        plan = build_current_ladder_overnight_plan(**common)
        print(
            json.dumps(
                {
                    "entry_dates": list(plan.entry_dates),
                    "products": list(plan.products),
                    "available_product_days": len(plan.product_days),
                    "grid_product_days": plan.universe.height,
                    "missing_product_days": int(
                        plan.universe.filter(
                            ~plan.universe["selected_for_replay"]
                        ).height
                    ),
                    "candidate_sessions": len(plan.candidate_sessions),
                    "source_binding": plan.source_binding,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.verify_only is not None:
        marker = verify_current_ladder_overnight_bundle(
            args.verify_only,
            **common,
        )
    else:
        marker = run_current_ladder_overnight(
            output_root=args.output,
            resume=not args.no_resume,
            **common,
        )
    print(json.dumps(marker, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
