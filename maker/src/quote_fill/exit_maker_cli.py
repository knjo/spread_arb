"""Command-line entry point for the standalone exit maker replay."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
from typing import Sequence

from .exit_maker_allocator import validate_exit_maker_allocator_launch

import polars as pl

from ..common.paths import HFT_DATA_ROOT, MAKER_ROOT
from .execution_runner import (
    DEFAULT_ROLLING_BOUNDARY_PATH,
    DEFAULT_WALKFORWARD_DAILY_ROOT,
)
from .exit_maker_runner import (
    DEFAULT_ENTRY_EXECUTION_ROOT,
    DEFAULT_EXIT_MAKER_OUTPUT_ROOT,
    ExitMakerRunnerConfig,
    _validate_disjoint_roots,
    discover_entry_product_days,
    run_exit_maker_replay,
)


DEFAULT_SESSIONS_PATH = MAKER_ROOT / "data" / "walkforward" / "sessions.txt"


def _parse_csv(value: str) -> tuple[str, ...]:
    parsed = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parsed or len(parsed) != len(set(parsed)):
        raise argparse.ArgumentTypeError("values must be nonempty and unique")
    return parsed


def _load_sessions(path: Path, last_sessions: int) -> tuple[str, ...]:
    sessions = tuple(
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if last_sessions <= 0 or len(sessions) < last_sessions:
        raise ValueError("last-sessions must be positive and available")
    return sessions[-last_sessions:]


def run(
    *,
    sessions: Sequence[str],
    symbols: Sequence[str],
    entry_execution_root: Path = DEFAULT_ENTRY_EXECUTION_ROOT,
    output_root: Path = DEFAULT_EXIT_MAKER_OUTPUT_ROOT,
    daily_root: Path = DEFAULT_WALKFORWARD_DAILY_ROOT,
    boundary_snapshot_path: Path = DEFAULT_ROLLING_BOUNDARY_PATH,
    data_root: Path = HFT_DATA_ROOT,
    quantiles: Sequence[int] = (50, 80, 95),
    hedge_delay_ns: int = 50_000_000,
    max_book_age_ns: int | None = None,
    resume: bool = True,
) -> pl.DataFrame:
    """Discover the requested entry grid, record support, and replay it."""

    validate_exit_maker_allocator_launch()
    entry_root = Path(entry_execution_root)
    destination = Path(output_root)
    _validate_disjoint_roots(entry_root, destination)
    keys, availability = discover_entry_product_days(
        sessions,
        symbols,
        entry_execution_root=entry_root,
    )
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(
        availability,
        destination / "product_day_universe.csv",
    )
    if not keys:
        raise ValueError(
            "none of the requested product-days has a complete entry partition"
        )
    config = ExitMakerRunnerConfig(
        boundary_quantiles=tuple(int(value) for value in quantiles),
        hedge_delay_ns=hedge_delay_ns,
        max_book_age_ns=max_book_age_ns,
    )
    return run_exit_maker_replay(
        keys,
        entry_execution_root=entry_root,
        output_root=destination,
        config=config,
        daily_root=Path(daily_root),
        boundary_snapshot_path=Path(boundary_snapshot_path),
        data_root=Path(data_root),
        resume=resume,
    )


def _atomic_write_csv(frame: pl.DataFrame, destination: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    try:
        Path(temporary_name).unlink()
        frame.write_csv(temporary_name)
        Path(temporary_name).replace(destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay exit maker-then-taker paths from entry partitions"
    )
    parser.add_argument("--sessions", type=Path, default=DEFAULT_SESSIONS_PATH)
    parser.add_argument("--last-sessions", type=int, default=60)
    parser.add_argument("--symbols", type=_parse_csv, required=True)
    parser.add_argument("--quantiles", type=_parse_csv, default=("50", "80", "95"))
    parser.add_argument(
        "--entry-root", type=Path, default=DEFAULT_ENTRY_EXECUTION_ROOT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_EXIT_MAKER_OUTPUT_ROOT)
    parser.add_argument(
        "--daily-root", type=Path, default=DEFAULT_WALKFORWARD_DAILY_ROOT
    )
    parser.add_argument(
        "--boundary-path", type=Path, default=DEFAULT_ROLLING_BOUNDARY_PATH
    )
    parser.add_argument("--data-root", type=Path, default=HFT_DATA_ROOT)
    parser.add_argument("--hedge-delay-ms", type=int, default=50)
    parser.add_argument("--max-book-age-ms", type=int)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sessions = _load_sessions(args.sessions, args.last_sessions)
    manifest = run(
        sessions=sessions,
        symbols=args.symbols,
        entry_execution_root=args.entry_root,
        output_root=args.output,
        daily_root=args.daily_root,
        boundary_snapshot_path=args.boundary_path,
        data_root=args.data_root,
        quantiles=tuple(int(value) for value in args.quantiles),
        hedge_delay_ns=args.hedge_delay_ms * 1_000_000,
        max_book_age_ns=(
            None
            if args.max_book_age_ms is None
            else args.max_book_age_ms * 1_000_000
        ),
        resume=not args.no_resume,
    )
    print(manifest.select("Date", "ValueCode").sort(["Date", "ValueCode"]))


if __name__ == "__main__":
    main()
