"""CLI for the tick-ladder-bound v2 liquidity research-universe manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .universe_manifest import (
    DEFAULT_LIQUIDITY_ROOT,
    DEFAULT_OUTPUT_DIR,
    UniverseManifestConfig,
    build_research_universe_manifest,
    load_verified_liquidity_sources,
    publish_research_universe_manifest,
)


def run(
    *,
    liquidity_root: Path = DEFAULT_LIQUIDITY_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    sources = load_verified_liquidity_sources(liquidity_root)
    config = UniverseManifestConfig()
    result = build_research_universe_manifest(
        sources.rolling_screen,
        sources.stable_core_products,
        sources.lineage,
        config,
    )
    return publish_research_universe_manifest(
        result,
        sources.lineage,
        output_dir,
        config,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the hash- and price-ladder-bound retrospective raw-replay "
            "universe"
        )
    )
    parser.add_argument(
        "--liquidity-root",
        type=Path,
        default=DEFAULT_LIQUIDITY_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    args = parser.parse_args(argv)
    destination = run(
        liquidity_root=args.liquidity_root,
        output_dir=args.output_dir,
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
