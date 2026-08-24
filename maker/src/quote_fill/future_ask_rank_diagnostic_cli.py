"""CLI for the fixed five-day FUTURE-maker ASK1--ASK5 diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path

from .future_ask_rank_diagnostic import (
    DEFAULT_SAMPLE_DATES,
    FutureAskRankDiagnosticConfig,
    publish_future_ask_rank_diagnostic,
    verify_future_ask_rank_diagnostic,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-root", type=Path)
    parser.add_argument("--bid-study-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dates",
        nargs="+",
        default=list(DEFAULT_SAMPLE_DATES),
        help="sorted YYYYMMDD dates; formal run uses the frozen five-day cohort",
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser


def run(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.verify_only:
        marker = verify_future_ask_rank_diagnostic(args.output)
        print(
            f"verified {args.output} "
            f"config_sha256={marker['config_sha256']}"
        )
        return 0
    if args.execution_root is None or args.bid_study_root is None:
        raise ValueError(
            "--execution-root and --bid-study-root are required for publish"
        )
    if tuple(args.dates) != DEFAULT_SAMPLE_DATES:
        raise ValueError(
            "formal CLI publish is fixed to the representative five-day cohort"
        )
    config = FutureAskRankDiagnosticConfig(
        execution_root=str(args.execution_root),
        bid_study_root=str(args.bid_study_root),
        dates=tuple(args.dates),
    )
    published = publish_future_ask_rank_diagnostic(args.output, config)
    verify_future_ask_rank_diagnostic(published)
    print(f"published and verified {published}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
