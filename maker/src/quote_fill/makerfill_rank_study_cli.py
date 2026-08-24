"""CLI for the bounded legacy makerFill B1--B5 candidate study."""

from __future__ import annotations

import argparse
from pathlib import Path

from .makerfill_rank_study_runner import (
    DEFAULT_SAMPLE_DATES,
    MakerFillRankStudyConfig,
    publish_makerfill_rank_study,
    verify_makerfill_rank_study,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-root", type=Path)
    parser.add_argument("--tick-root", type=Path)
    parser.add_argument("--makerfill-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dates",
        nargs="+",
        default=list(DEFAULT_SAMPLE_DATES),
        help="sorted YYYYMMDD sample dates",
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser


def run(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.verify_only:
        marker = verify_makerfill_rank_study(args.output)
        print(
            f"verified {args.output} "
            f"config_sha256={marker['config_sha256']}"
        )
        return 0
    if any(
        value is None
        for value in (args.execution_root, args.tick_root, args.makerfill_root)
    ):
        raise ValueError(
            "--execution-root, --tick-root, and --makerfill-root are required"
        )
    config = MakerFillRankStudyConfig(
        execution_root=str(args.execution_root),
        tick_root=str(args.tick_root),
        makerfill_root=str(args.makerfill_root),
        dates=tuple(args.dates),
    )
    published = publish_makerfill_rank_study(args.output, config)
    verify_makerfill_rank_study(published)
    print(f"published and verified {published}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
