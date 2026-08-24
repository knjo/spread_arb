"""CLI for the per-submitted-entry-quote nominal-V0 lookup checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .submitted_entry_lookup import SubmittedEntryLookupConfig
from .submitted_entry_lookup_runner import (
    DEFAULT_CONDITIONAL_CHECKPOINT_ROOT,
    DEFAULT_ENTRY_EXECUTION_ROOT,
    DEFAULT_EXIT_MAKER_ROOT,
    DEFAULT_OUTPUT_ROOT,
    run_submitted_entry_lookup_checkpoint,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exit-root", type=Path, default=DEFAULT_EXIT_MAKER_ROOT)
    parser.add_argument("--entry-root", type=Path, default=DEFAULT_ENTRY_EXECUTION_ROOT)
    parser.add_argument(
        "--conditional-checkpoint-root",
        type=Path,
        default=DEFAULT_CONDITIONAL_CHECKPOINT_ROOT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sessions", type=int, default=60)
    parser.add_argument("--value-code", action="append", dest="value_codes")
    parser.add_argument("--allow-fewer-sessions", action="store_true")
    parser.add_argument("--require-balanced-product-days", action="store_true")
    parser.add_argument("--skip-hash-validation", action="store_true")
    parser.add_argument("--include-causal-state", action="store_true")
    parser.add_argument("--lookback-sessions", type=int, default=60)
    parser.add_argument("--min-history-sessions", type=int, default=40)
    parser.add_argument("--min-group-sessions", type=int, default=20)
    parser.add_argument("--min-known-paths", type=int, default=100)
    parser.add_argument("--flat-completed-cycle-cost-bp", type=float, default=19.0)
    parser.add_argument("--same-day-completed-cycle-cost-bp", type=float, default=19.0)
    parser.add_argument("--overnight-completed-cycle-cost-bp", type=float, default=34.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = SubmittedEntryLookupConfig(
        lookback_sessions=args.lookback_sessions,
        min_history_sessions=args.min_history_sessions,
        min_group_sessions=args.min_group_sessions,
        min_known_paths=args.min_known_paths,
        assumed_flat_completed_cycle_cost_bp=args.flat_completed_cycle_cost_bp,
        assumed_same_day_completed_cycle_cost_bp=(
            args.same_day_completed_cycle_cost_bp
        ),
        assumed_overnight_completed_cycle_cost_bp=(
            args.overnight_completed_cycle_cost_bp
        ),
    )
    checkpoint = run_submitted_entry_lookup_checkpoint(
        args.exit_root,
        args.entry_root,
        args.conditional_checkpoint_root,
        output_dir=args.output,
        sessions=args.sessions,
        value_codes=args.value_codes,
        require_exact_sessions=not args.allow_fewer_sessions,
        require_balanced_product_days=args.require_balanced_product_days,
        validate_hashes=not args.skip_hash_validation,
        include_causal_state=args.include_causal_state,
        lookup_config=config,
    )
    print(
        "submitted entry lookup checkpoint complete: "
        f"labels={checkpoint.labels.height}, lookup={checkpoint.lookup.height}, "
        f"state_lookup={checkpoint.state_lookup.height}, output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
