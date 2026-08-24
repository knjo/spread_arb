"""CLI for the narrowed walk-forward maker execution replay.

The command is intentionally a research runner rather than a production
strategy.  It reads only validated D-1 rolling boundary snapshots, replays
one product-day at a time, and can attach two causal same-day taker-exit
benchmarks: the fair value frozen at entry and the D-1 empirical lower band.
An unhit exit remains ``carry_at_eod`` and is never imputed as zero PnL.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import polars as pl

from ..quote_width.rolling import load_rolling_boundary_snapshots
from ..common.paths import MAKER_ROOT
from .execution_runner import (
    DEFAULT_WALKFORWARD_DAILY_ROOT,
    DEFAULT_ROLLING_BOUNDARY_PATH,
    ExecutionRunnerConfig,
    load_walkforward_execution_day_batch,
    run_day_batched_execution_replay,
)


DEFAULT_SESSIONS_PATH = MAKER_ROOT / "data" / "walkforward" / "sessions.txt"


def build_frozen_exit_rules(
    action_facts: pl.DataFrame,
    boundary_rows: pl.DataFrame,
) -> pl.DataFrame:
    """Build D-1-safe frozen-center and frozen-lower exit rules.

    ``threshold_basis_bp`` is the causal entry fair plus the D-1 upper
    distance.  Subtracting that upper distance therefore recovers the fair
    observed at the submit event; subtracting the D-1 lower distance adds the
    asymmetric full-band benchmark.  Both thresholds are frozen after entry.
    """

    if action_facts.is_empty():
        return pl.DataFrame(
            schema={
                "policy_generation_id": pl.String,
                "exit_rule_id": pl.String,
                "exit_threshold_basis_bp": pl.Float64,
                "source_asof_date": pl.String,
                "contains_target_day_outcome": pl.Boolean,
            }
        )
    action_required = {
        "Date",
        "ValueCode",
        "QuoteCode",
        "boundary_quantile",
        "policy_generation_id",
        "threshold_basis_bp",
        "source_asof_date",
    }
    boundary_required = {
        "Date",
        "ValueCode",
        "QuoteCode",
        "boundary_quantile",
        "upper_distance_bp",
        "lower_distance_bp",
        "source_asof_date",
        "execution_safe_snapshot",
        "contains_target_day_outcome",
    }
    missing_action = sorted(action_required - set(action_facts.columns))
    missing_boundary = sorted(boundary_required - set(boundary_rows.columns))
    if missing_action or missing_boundary:
        raise ValueError(
            "missing exit-rule inputs: "
            f"action={missing_action}, boundary={missing_boundary}"
        )
    keys = ["Date", "ValueCode", "QuoteCode", "boundary_quantile"]
    boundaries = boundary_rows.select(
        *keys,
        "upper_distance_bp",
        "lower_distance_bp",
        pl.col("source_asof_date").alias("_boundary_source_asof_date"),
        "execution_safe_snapshot",
        "contains_target_day_outcome",
    )
    if boundaries.select(keys).n_unique() != boundaries.height:
        raise ValueError("boundary rows duplicate an exact product-day quantile")
    if boundaries.filter(
        ~pl.col("execution_safe_snapshot").fill_null(False)
        | pl.col("contains_target_day_outcome").fill_null(True)
    ).height:
        raise ValueError("exit-rule boundary input is not execution-safe")
    base = action_facts.select(
        *keys,
        "policy_generation_id",
        "threshold_basis_bp",
        "source_asof_date",
    ).join(boundaries, on=keys, how="left", validate="m:1")
    bad = base.filter(
        pl.col("upper_distance_bp").is_null()
        | pl.col("lower_distance_bp").is_null()
        | pl.col("source_asof_date").is_null()
        | pl.col("_boundary_source_asof_date").is_null()
        | (
            pl.col("source_asof_date").cast(pl.String)
            != pl.col("_boundary_source_asof_date").cast(pl.String)
        ).fill_null(True)
        | (
            pl.col("source_asof_date").cast(pl.String)
            >= pl.col("Date").cast(pl.String)
        ).fill_null(True)
    )
    if bad.height:
        raise ValueError("exit-rule boundary lineage is missing, unsafe, or mismatched")
    base = base.with_columns(
        (pl.col("threshold_basis_bp") - pl.col("upper_distance_bp")).alias(
            "_frozen_center_bp"
        )
    )
    center = base.select(
        "policy_generation_id",
        pl.lit("frozen_center").alias("exit_rule_id"),
        pl.col("_frozen_center_bp").alias("exit_threshold_basis_bp"),
        "source_asof_date",
        pl.lit(False).alias("contains_target_day_outcome"),
    )
    lower = base.select(
        "policy_generation_id",
        pl.lit("frozen_lower").alias("exit_rule_id"),
        (
            pl.col("_frozen_center_bp") - pl.col("lower_distance_bp")
        ).alias("exit_threshold_basis_bp"),
        "source_asof_date",
        pl.lit(False).alias("contains_target_day_outcome"),
    )
    return pl.concat([center, lower]).sort(
        ["policy_generation_id", "exit_rule_id"]
    )


def _parse_csv(value: str) -> tuple[str, ...]:
    parsed = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parsed or len(parsed) != len(set(parsed)):
        raise argparse.ArgumentTypeError("values must be nonempty and unique")
    return parsed


def _load_sessions(path: Path, last_sessions: int) -> tuple[str, ...]:
    sessions = tuple(
        line.strip() for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if last_sessions <= 0 or len(sessions) < last_sessions:
        raise ValueError("last-sessions must be positive and available")
    return sessions[-last_sessions:]


def run(
    *,
    sessions: Sequence[str],
    symbols: Sequence[str],
    quantiles: Sequence[int],
    output_dir: Path,
    daily_root: Path = DEFAULT_WALKFORWARD_DAILY_ROOT,
    boundary_path: Path = DEFAULT_ROLLING_BOUNDARY_PATH,
    with_taker_exit: bool = True,
    resume: bool = True,
    workers: int = 1,
    worker_index: int = 0,
) -> pl.DataFrame:
    if workers <= 0 or not 0 <= worker_index < workers:
        raise ValueError("worker_index must be in [0, workers)")
    config = ExecutionRunnerConfig(
        boundary_quantiles=tuple(int(value) for value in quantiles)
    )
    selected_boundaries = (
        load_rolling_boundary_snapshots(
            boundary_path,
            expected_daily_root=daily_root,
        )
        .filter(
            pl.col("Date").cast(pl.String).is_in(list(sessions))
            & pl.col("ValueCode").cast(pl.String).is_in(list(symbols))
            & pl.col("boundary_quantile").cast(pl.Int64).is_in(list(quantiles))
        )
    )
    product_days, availability = _available_product_days(
        sessions, symbols, daily_root=daily_root
    )
    product_days = product_days[worker_index::workers]
    output_dir.mkdir(parents=True, exist_ok=True)
    if worker_index == 0:
        universe_path = output_dir / "product_day_universe.csv"
        temporary = output_dir / ".product_day_universe.tmp.csv"
        availability.write_csv(temporary)
        temporary.replace(universe_path)

    def exit_loader(
        date: str, value_code: str, action_facts: pl.DataFrame
    ) -> pl.DataFrame | None:
        if not with_taker_exit:
            return None
        rows = selected_boundaries.filter(
            (pl.col("Date").cast(pl.String) == date)
            & (pl.col("ValueCode").cast(pl.String) == value_code)
        )
        return build_frozen_exit_rules(action_facts, rows)

    def loader(date: str, value_codes: Sequence[str]):
        return load_walkforward_execution_day_batch(
            date,
            value_codes,
            daily_root=daily_root,
            boundary_snapshot_path=boundary_path,
            quantiles=config.boundary_quantiles,
        )

    return run_day_batched_execution_replay(
        product_days,
        loader,
        output_dir,
        config,
        exit_rule_loader=exit_loader if with_taker_exit else None,
        resume=resume,
    )


def _available_product_days(
    sessions: Sequence[str],
    symbols: Sequence[str],
    *,
    daily_root: Path,
) -> tuple[tuple[tuple[str, str], ...], pl.DataFrame]:
    """Use one common calendar while retaining only actually mapped products."""

    requested = {(str(date), str(symbol)) for date in sessions for symbol in symbols}
    available: set[tuple[str, str]] = set()
    for date in sessions:
        mapping_path = Path(daily_root) / f"Date={date}" / "mapping.parquet"
        if not mapping_path.is_file():
            raise FileNotFoundError(mapping_path)
        values = (
            pl.scan_parquet(mapping_path)
            .filter(pl.col("ValueCode").cast(pl.String).is_in(list(symbols)))
            .select(pl.col("ValueCode").cast(pl.String))
            .unique()
            .collect(engine="streaming")["ValueCode"]
            .to_list()
        )
        available.update((str(date), str(value)) for value in values)
    unexpected = available - requested
    if unexpected:
        raise ValueError(f"daily mappings contain unrequested product-days: {unexpected}")
    rows = [
        {
            "Date": date,
            "ValueCode": symbol,
            "requested": True,
            "available_daily_mapping": (date, symbol) in available,
            "availability_status": (
                "available" if (date, symbol) in available else "missing_daily_mapping"
            ),
        }
        for date in sessions
        for symbol in symbols
    ]
    audit = pl.from_dicts(rows, infer_schema_length=None).sort(
        ["Date", "ValueCode"]
    )
    keys = tuple(
        (row["Date"], row["ValueCode"])
        for row in audit.filter(pl.col("available_daily_mapping")).iter_rows(
            named=True
        )
    )
    return keys, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a bounded walk-forward maker execution replay"
    )
    parser.add_argument("--sessions", type=Path, default=DEFAULT_SESSIONS_PATH)
    parser.add_argument("--last-sessions", type=int, default=60)
    parser.add_argument("--symbols", type=_parse_csv, required=True)
    parser.add_argument("--quantiles", type=_parse_csv, default=("50", "80", "95"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--without-taker-exit", action="store_true", help="skip same-day exit facts"
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sessions = _load_sessions(args.sessions, args.last_sessions)
    manifest = run(
        sessions=sessions,
        symbols=args.symbols,
        quantiles=tuple(int(value) for value in args.quantiles),
        output_dir=args.output,
        with_taker_exit=not args.without_taker_exit,
        resume=not args.no_resume,
        workers=args.workers,
        worker_index=args.worker_index,
    )
    print(manifest.select("Date", "ValueCode").sort(["Date", "ValueCode"]))


if __name__ == "__main__":
    main()
