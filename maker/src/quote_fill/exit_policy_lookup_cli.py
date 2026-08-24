"""Command line entrypoint for the conditional exit-policy lookup checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .exit_policy_lookup import ExitPolicyLookupConfig
from .exit_policy_lookup_runner import (
    DEFAULT_ENTRY_EXECUTION_ROOT,
    DEFAULT_EXIT_MAKER_ROOT,
    DEFAULT_OUTPUT_ROOT,
    run_exit_policy_lookup_checkpoint,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exit-root", type=Path, default=DEFAULT_EXIT_MAKER_ROOT)
    parser.add_argument(
        "--entry-root", type=Path, default=DEFAULT_ENTRY_EXECUTION_ROOT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overnight-root", type=Path)
    parser.add_argument("--sessions", type=int, default=60)
    parser.add_argument("--value-code", action="append", dest="value_codes")
    parser.add_argument("--allow-fewer-sessions", action="store_true")
    parser.add_argument("--require-balanced-product-days", action="store_true")
    parser.add_argument("--skip-hash-validation", action="store_true")
    parser.add_argument("--skip-causal-state", action="store_true")
    parser.add_argument("--lookback-sessions", type=int, default=60)
    parser.add_argument("--min-history-sessions", type=int, default=40)
    parser.add_argument("--min-group-sessions", type=int, default=20)
    parser.add_argument("--min-known-paths", type=int, default=100)
    parser.add_argument("--assumed-non-price-cost-bp", type=float, default=19.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = ExitPolicyLookupConfig(
        lookback_sessions=args.lookback_sessions,
        min_history_sessions=args.min_history_sessions,
        min_group_sessions=args.min_group_sessions,
        min_known_paths=args.min_known_paths,
        assumed_non_price_cycle_cost_bp=args.assumed_non_price_cost_bp,
    )
    checkpoint = run_exit_policy_lookup_checkpoint(
        args.exit_root,
        args.entry_root,
        output_dir=args.output,
        overnight_root=args.overnight_root,
        sessions=args.sessions,
        value_codes=args.value_codes,
        require_exact_sessions=not args.allow_fewer_sessions,
        require_balanced_product_days=args.require_balanced_product_days,
        validate_hashes=not args.skip_hash_validation,
        include_causal_state=not args.skip_causal_state,
        lookup_config=config,
    )
    print(
        "exit policy lookup checkpoint complete: "
        f"strict_labels={checkpoint.labels.strict.height}, "
        f"nominal_v0_labels={checkpoint.labels.nominal_v0.height}, "
        f"strict_lookup={checkpoint.strict_lookup.height}, "
        f"nominal_v0_lookup={checkpoint.nominal_v0_lookup.height}, "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
