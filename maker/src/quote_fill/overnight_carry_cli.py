"""CLI for strict overnight carry labels and their gross-only report."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
from typing import Sequence

import polars as pl

from ..common.paths import HFT_DATA_ROOT, MAKER_ROOT
from .overnight_carry import OvernightCarryConfig
from .overnight_carry_runner import (
    DEFAULT_ENTRY_EXECUTION_ROOT,
    DEFAULT_EXIT_MAKER_ROOT,
    DEFAULT_FUTURES_RAW_ROOT,
    DEFAULT_OVERNIGHT_CARRY_ROOT,
    OvernightCarryRunnerConfig,
    discover_overnight_product_days,
    run_overnight_carry_replay,
)


DEFAULT_SESSIONS_PATH = MAKER_ROOT / "data" / "walkforward" / "sessions.txt"


def _parse_csv(value: str) -> tuple[str, ...]:
    parsed = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parsed or len(parsed) != len(set(parsed)):
        raise argparse.ArgumentTypeError("values must be nonempty and unique")
    return parsed


def _load_sessions(path: Path) -> tuple[str, ...]:
    sessions = tuple(
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not sessions or sessions != tuple(sorted(sessions)) or len(sessions) != len(
        set(sessions)
    ):
        raise ValueError("sessions file must be nonempty, unique and ascending")
    return sessions


def _read_table(path: Path) -> pl.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pl.read_csv(path)
    raise ValueError(f"unsupported table format for {path}; use parquet or csv")


def run(
    *,
    sessions: Sequence[str],
    entry_sessions: Sequence[str],
    symbols: Sequence[str],
    contract_calendar: pl.DataFrame,
    settlement_facts: pl.DataFrame | None = None,
    exit_maker_root: Path = DEFAULT_EXIT_MAKER_ROOT,
    entry_execution_root: Path = DEFAULT_ENTRY_EXECUTION_ROOT,
    output_root: Path = DEFAULT_OVERNIGHT_CARRY_ROOT,
    max_carry_sessions: int = 1,
    max_book_age_ns: int | None = None,
    data_root: Path = HFT_DATA_ROOT,
    futures_raw_root: Path = DEFAULT_FUTURES_RAW_ROOT,
    resume: bool = True,
    build_report: bool = True,
    report_output: Path | None = None,
) -> pl.DataFrame:
    """Discover complete inputs, replay strict labels, then report atomically."""

    keys, availability = discover_overnight_product_days(
        entry_sessions,
        symbols,
        exit_maker_root=Path(exit_maker_root),
        entry_execution_root=Path(entry_execution_root),
    )
    destination = Path(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(availability, destination / "product_day_universe.csv")
    if not keys:
        raise ValueError("none of the requested product-days has both complete inputs")
    config = OvernightCarryRunnerConfig(
        carry=OvernightCarryConfig(
            max_carry_sessions=max_carry_sessions,
            max_book_age_ns=max_book_age_ns,
        )
    )
    manifest = run_overnight_carry_replay(
        keys,
        sessions=sessions,
        contract_calendar=contract_calendar,
        settlement_facts=settlement_facts,
        exit_maker_root=Path(exit_maker_root),
        entry_execution_root=Path(entry_execution_root),
        output_root=destination,
        config=config,
        data_root=Path(data_root),
        futures_raw_root=Path(futures_raw_root),
        resume=resume,
    )
    if build_report:
        from .overnight_carry_report import run_overnight_carry_report

        run_overnight_carry_report(
            destination,
            output_dir=(Path(report_output) if report_output is not None else None),
        )
    return manifest


def _atomic_write_csv(frame: pl.DataFrame, destination: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.unlink()
        frame.write_csv(temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Label strict overnight carry branches on exact-contract raw tape; "
            "outputs remain gross-only and not EV-ready"
        )
    )
    parser.add_argument("--sessions", type=Path, default=DEFAULT_SESSIONS_PATH)
    parser.add_argument("--last-entry-sessions", type=int, default=60)
    parser.add_argument("--symbols", type=_parse_csv, required=True)
    parser.add_argument("--contract-calendar", type=Path, required=True)
    parser.add_argument("--settlements", type=Path)
    parser.add_argument("--exit-root", type=Path, default=DEFAULT_EXIT_MAKER_ROOT)
    parser.add_argument("--entry-root", type=Path, default=DEFAULT_ENTRY_EXECUTION_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OVERNIGHT_CARRY_ROOT)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--max-carry-sessions", type=int, default=1)
    parser.add_argument("--max-book-age-ms", type=int)
    parser.add_argument("--data-root", type=Path, default=HFT_DATA_ROOT)
    parser.add_argument("--futures-root", type=Path, default=DEFAULT_FUTURES_RAW_ROOT)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sessions = _load_sessions(args.sessions)
    if args.last_entry_sessions <= 0 or len(sessions) < args.last_entry_sessions:
        raise ValueError("last-entry-sessions must be positive and available")
    manifest = run(
        sessions=sessions,
        entry_sessions=sessions[-args.last_entry_sessions :],
        symbols=args.symbols,
        contract_calendar=_read_table(args.contract_calendar),
        settlement_facts=(
            None if args.settlements is None else _read_table(args.settlements)
        ),
        exit_maker_root=args.exit_root,
        entry_execution_root=args.entry_root,
        output_root=args.output,
        max_carry_sessions=args.max_carry_sessions,
        max_book_age_ns=(
            None
            if args.max_book_age_ms is None
            else args.max_book_age_ms * 1_000_000
        ),
        data_root=args.data_root,
        futures_raw_root=args.futures_root,
        resume=not args.no_resume,
        build_report=not args.skip_report,
        report_output=args.report_output,
    )
    print(manifest.select("Date", "ValueCode").sort(["Date", "ValueCode"]))


if __name__ == "__main__":
    main()
