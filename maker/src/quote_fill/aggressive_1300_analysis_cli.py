"""CLI for the source-bound 13:00 challenger analysis bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from .aggressive_1300_analysis import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SEED_TEMPLATE,
    DEFAULT_SOURCE_ROOT,
    DEFAULT_SUPPLEMENTAL_ROOT,
    DEFAULT_UNIVERSE_ROOT,
    run_analysis_bundle,
    verify_analysis_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--universe-root", type=Path, default=DEFAULT_UNIVERSE_ROOT)
    parser.add_argument("--supplemental-root", type=Path, default=DEFAULT_SUPPLEMENTAL_ROOT)
    parser.add_argument("--seed-template", type=Path, default=DEFAULT_SEED_TEMPLATE)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--no-source-rehash", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify_analysis_bundle(
            args.output_root,
            verify_sources=not args.no_source_rehash,
        )
        return
    if args.no_source_rehash:
        parser.error("--no-source-rehash is only valid with --verify-only")
    run_analysis_bundle(
        args.output_root,
        source_root=args.source_root,
        universe_root=args.universe_root,
        supplemental_root=args.supplemental_root,
        seed_template_path=args.seed_template,
        cache_root=args.cache_root,
    )


if __name__ == "__main__":
    main()
