"""Bounded real product-day benchmark for the compact evaluator prototype."""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import gc
import hashlib
import json
from pathlib import Path
import resource
import shutil
import tempfile
import time
from typing import Callable

import polars as pl

from .compact_evaluator import (
    COMPACT_EVALUATOR_VERSION,
    CompactEvaluationResult,
    CompactEvaluatorConfig,
    evaluate_compact_product_day,
    load_makerfill_fast_adapter,
    summarize_compact_q,
)
from .execution_runner import (
    load_walkforward_execution_product_day,
    replay_execution_product_day,
)


DEFAULT_ENTRY_ROOT = Path("maker/data/walkforward/execution_narrow_60d")
DEFAULT_POSITION_ROOT = Path("maker/data/walkforward/exit_maker_narrow_60d")
_ACTION_KEYS = [
    "route",
    "boundary_quantile",
    "spread_pair_epoch",
    "target_price_tick",
    "submit_recv_time_ns",
    "submit_event_sequence",
    "submit_row_index",
]


def benchmark_product_day(
    date: str,
    value_code: str,
    *,
    entry_root: Path = DEFAULT_ENTRY_ROOT,
    position_root: Path = DEFAULT_POSITION_ROOT,
) -> tuple[dict[str, object], CompactEvaluationResult, pl.DataFrame]:
    """Run one read-only real benchmark and return metrics plus compact rows."""

    date = str(date)
    value_code = str(value_code)
    partition = Path(entry_root) / f"Date={date}" / f"ValueCode={value_code}"
    position_partition = (
        Path(position_root) / f"Date={date}" / f"ValueCode={value_code}"
    )
    action_path = partition / "execution_action_facts.parquet"
    position_path = position_partition / "exit_maker_position_policy_facts.parquet"
    if not action_path.is_file():
        raise FileNotFoundError(action_path)
    if not position_path.is_file():
        raise FileNotFoundError(position_path)

    load_started = time.perf_counter()
    merged = load_walkforward_execution_product_day(date, value_code)
    load_seconds = time.perf_counter() - load_started
    adapter_started = time.perf_counter()
    adapter = load_makerfill_fast_adapter(date, value_code)
    adapter_seconds = time.perf_counter() - adapter_started

    hybrid_seconds, hybrid = _timed(
        lambda: evaluate_compact_product_day(
            merged,
            CompactEvaluatorConfig(fill_backend="hybrid_makerfill"),
            makerfill_adapter=adapter,
        )
    )
    indexed_seconds, indexed = _timed(
        lambda: evaluate_compact_product_day(
            merged,
            CompactEvaluatorConfig(fill_backend="indexed"),
            makerfill_adapter=adapter,
        )
    )
    formal_seconds, formal_live = _timed(
        lambda: replay_execution_product_day(merged)
    )

    formal_actions = pl.read_parquet(action_path)
    positions = pl.read_parquet(position_path)
    indexed_parity = _indexed_formal_parity(
        indexed.order_outcomes, formal_actions
    )
    makerfill_parity = _makerfill_indexed_parity(
        hybrid.order_outcomes, indexed.order_outcomes
    )
    position_comparison = _position_comparison(
        hybrid.order_outcomes,
        formal_actions,
        positions,
    )
    q_summary = summarize_compact_q(hybrid.order_outcomes)
    persisted_files = [path for path in partition.iterdir() if path.is_file()]
    metrics: dict[str, object] = {
        "benchmark_version": "compact_evaluator_benchmark_v1",
        "compact_evaluator_version": COMPACT_EVALUATOR_VERSION,
        "Date": date,
        "ValueCode": value_code,
        "scope": "one_loaded_product_day_read_only",
        "timing_seconds": {
            "raw_and_merged_load": load_seconds,
            "makerfill_adapter_load": adapter_seconds,
            "compact_hybrid_evaluation": hybrid_seconds,
            "compact_indexed_evaluation": indexed_seconds,
            "formal_evaluation": formal_seconds,
            "hybrid_speedup_vs_formal_evaluation_only": (
                formal_seconds / hybrid_seconds if hybrid_seconds else None
            ),
            "end_to_end_speedup_claimed": False,
            "note": (
                "raw product-day loading dominates and is shared; timings are "
                "single-run warm-cache diagnostics, not a capacity promise"
            ),
        },
        "memory": {
            "process_peak_rss_kib_after_all_modes": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
            "compact_hybrid_output_estimated_bytes": _result_size(hybrid),
            "compact_indexed_output_estimated_bytes": _result_size(indexed),
            "formal_live_output_estimated_bytes": _result_size(formal_live),
            "persisted_formal_partition_bytes": sum(
                path.stat().st_size for path in persisted_files
            ),
            "note": (
                "peak RSS includes the shared raw loader; DataFrame estimated "
                "bytes compare retained result surfaces"
            ),
        },
        "rows": {
            "compact_hybrid_order_outcomes": hybrid.order_outcomes.height,
            "compact_hybrid_state_changes": hybrid.state_changes.height,
            "compact_hybrid_audit": hybrid.audit.height,
            "compact_q_summary": q_summary.height,
            "formal_actions": formal_actions.height,
            "formal_positions": positions.height,
        },
        "indexed_vs_formal": indexed_parity,
        "makerfill_approx_vs_indexed": makerfill_parity,
        "position_comparison": position_comparison,
        "safety": {
            "makerfill_outcome_exact": False,
            "makerfill_hedge_is_estimate_only": True,
            "unsupported_spot_ranks_imputed_as_no_fill": False,
            "pathwise_ev_ready": False,
            "joint_volume_allocated": False,
            "formal_roots_mutated": False,
        },
    }
    return metrics, hybrid, q_summary


def publish_benchmark(
    output_dir: Path,
    metrics: dict[str, object],
    result: CompactEvaluationResult,
    q_summary: pl.DataFrame,
) -> None:
    """Publish a new analysis-only bundle; existing paths are never reused."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp.", dir=output_dir.parent
        )
    )
    try:
        result.order_outcomes.write_parquet(temporary / "order_outcomes.parquet")
        result.state_changes.write_parquet(temporary / "state_changes.parquet")
        result.audit.write_parquet(temporary / "audit.parquet")
        q_summary.write_csv(temporary / "q_summary.csv")
        (temporary / "benchmark.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifacts = {}
        for path in sorted(temporary.iterdir()):
            artifacts[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        marker = {
            "schema_version": "compact_evaluator_benchmark_bundle_v1",
            "analysis_only": True,
            "formal_roots_mutated": False,
            "pathwise_ev_ready": False,
            "joint_volume_allocated": False,
            "artifacts": artifacts,
        }
        (temporary / "complete.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _indexed_formal_parity(
    compact: pl.DataFrame, formal: pl.DataFrame
) -> dict[str, object]:
    parity_fields = [
        "raw_order_fact_id",
        "nominal_stop_recv_time_ns",
        "nominal_stop_reason",
        "known_filled_quantity",
        "any_fill",
        "full_fill",
        "partial_fill",
        "first_fill_recv_time_ns",
        "first_fill_event_sequence",
        "first_fill_row_index",
        "full_fill_recv_time_ns",
        "full_fill_event_sequence",
        "full_fill_row_index",
    ]
    _require(compact, {*_ACTION_KEYS, *parity_fields}, "compact indexed")
    _require(formal, {*_ACTION_KEYS, *parity_fields}, "formal actions")
    renamed = formal.select(
        *_ACTION_KEYS,
        *[pl.col(column).alias(f"formal_{column}") for column in parity_fields],
    )
    joined = compact.join(
        renamed,
        on=_ACTION_KEYS,
        how="full",
        coalesce=True,
        validate="1:1",
    )
    field_equal = {
        column: int(
            joined.select(
                pl.col(column)
                .eq_missing(pl.col(f"formal_{column}"))
                .sum()
            ).item()
        )
        for column in parity_fields
    }
    return {
        "compact_rows": compact.height,
        "formal_rows": formal.height,
        "joined_rows": joined.height,
        "exact_key_set_parity": joined.height == compact.height == formal.height,
        "field_equal_rows": field_equal,
        "all_fields_exact_parity": all(
            value == joined.height for value in field_equal.values()
        ),
    }


def _makerfill_indexed_parity(
    hybrid: pl.DataFrame, indexed: pl.DataFrame
) -> dict[str, object]:
    direct = hybrid.filter(pl.col("fill_backend") == "makerfill_fast")
    exact = indexed.select(
        *_ACTION_KEYS,
        pl.col("any_fill").alias("indexed_any_fill"),
        pl.col("full_fill").alias("indexed_full_fill"),
        pl.col("full_fill_recv_time_ns").alias("indexed_full_fill_recv_time_ns"),
    )
    joined = direct.join(exact, on=_ACTION_KEYS, validate="1:1")
    both_full = (pl.col("full_fill") == True) & (  # noqa: E712
        pl.col("indexed_full_fill") == True  # noqa: E712
    )
    return {
        "direct_mapping_aliases": direct.height,
        "direct_mapping_unique_physical_orders": direct.select(
            "raw_order_fact_id"
        ).n_unique(),
        "excluded_spot_aliases": hybrid.filter(
            pl.col("fill_backend") == "excluded_fail_closed"
        ).height,
        "full_status_equal_aliases": int(
            joined.select(
                pl.col("full_fill")
                .eq_missing(pl.col("indexed_full_fill"))
                .sum()
            ).item()
        ),
        "makerfill_full_indexed_full": joined.filter(both_full).height,
        "makerfill_full_indexed_not_full": joined.filter(
            (pl.col("full_fill") == True)  # noqa: E712
            & (pl.col("indexed_full_fill") != True)  # noqa: E712
        ).height,
        "makerfill_not_full_indexed_full": joined.filter(
            (pl.col("full_fill") != True)  # noqa: E712
            & (pl.col("indexed_full_fill") == True)  # noqa: E712
        ).height,
        "both_full_exact_timestamp_matches": joined.filter(both_full).filter(
            pl.col("full_fill_recv_time_ns")
            == pl.col("indexed_full_fill_recv_time_ns")
        ).height,
        "interpretation": (
            "makerFill is an EOD Float32/displayed-queue approximation; "
            "status agreement is diagnostic and not an exact fill label"
        ),
    }


def _position_comparison(
    hybrid: pl.DataFrame,
    formal: pl.DataFrame,
    positions: pl.DataFrame,
) -> dict[str, object]:
    formal_keyed = formal.select(
        *_ACTION_KEYS,
        "policy_generation_id",
        (
            pl.col("full_fill").fill_null(False)
            & pl.col("entry_hedge_label_observed")
            & pl.col("entry_hedge_executable")
        ).alias("formal_position_established"),
    )
    position_counts = positions.group_by("entry_policy_generation_id").agg(
        pl.len().alias("formal_position_rows"),
        pl.col("branch_status").sort().alias("formal_branch_statuses"),
    )
    formal_keyed = formal_keyed.join(
        position_counts,
        left_on="policy_generation_id",
        right_on="entry_policy_generation_id",
        how="left",
        validate="1:1",
    )
    compared = hybrid.join(
        formal_keyed,
        on=_ACTION_KEYS,
        how="inner",
        validate="1:1",
    ).with_columns(
        (
            pl.col("full_fill").fill_null(False)
            & pl.col("hedge_label_observed")
            & pl.col("hedge_executable")
        ).alias("compact_exact_position_established"),
        (
            pl.col("full_fill").fill_null(False)
            & pl.col("hedge_estimate_available")
            & (pl.col("hedge_estimate_status") == "executable")
        ).fill_null(False).alias("compact_estimated_position_established"),
    )
    branches = {
        str(row["branch_status"]): int(row["len"])
        for row in positions.group_by("branch_status")
        .len()
        .sort("branch_status")
        .iter_rows(named=True)
    }
    return {
        "formal_established_entry_aliases": int(
            compared.select(pl.col("formal_position_established").sum()).item()
        ),
        "compact_exact_established_aliases": int(
            compared.select(
                pl.col("compact_exact_position_established").sum()
            ).item()
        ),
        "compact_estimated_established_aliases": int(
            compared.select(
                pl.col("compact_estimated_position_established").sum()
            ).item()
        ),
        "exact_established_status_equal_aliases": int(
            compared.select(
                (
                    pl.col("compact_exact_position_established")
                    == pl.col("formal_position_established")
                ).sum()
            ).item()
        ),
        "estimated_established_status_equal_aliases": int(
            compared.select(
                (
                    pl.col("compact_estimated_position_established")
                    == pl.col("formal_position_established")
                ).sum()
            ).item()
        ),
        "formal_position_rows": positions.height,
        "formal_position_rows_linked_to_compared_actions": int(
            compared.select(
                pl.col("formal_position_rows").fill_null(0).sum()
            ).item()
        ),
        "formal_branch_status_counts": branches,
        "exit_outcome_parity_claimed": False,
        "note": (
            "compact v1 labels entry plus 50ms opposite-leg execution; "
            "formal same-day exit branches are joined only as downstream "
            "coverage and are not regenerated here"
        ),
    }


def _timed(function: Callable[[], object]) -> tuple[float, object]:
    gc.collect()
    started = time.perf_counter()
    value = function()
    return time.perf_counter() - started, value


def _result_size(value: object) -> int:
    if not is_dataclass(value):
        return 0
    total = 0
    for field in fields(value):
        item = getattr(value, field.name)
        if isinstance(item, pl.DataFrame):
            total += item.estimated_size()
    return total


def _require(frame: pl.DataFrame, columns: set[str], source: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--value-code", required=True)
    parser.add_argument("--entry-root", type=Path, default=DEFAULT_ENTRY_ROOT)
    parser.add_argument(
        "--position-root", type=Path, default=DEFAULT_POSITION_ROOT
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metrics, result, q_summary = benchmark_product_day(
        args.date,
        args.value_code,
        entry_root=args.entry_root,
        position_root=args.position_root,
    )
    if args.output is not None:
        publish_benchmark(args.output, metrics, result, q_summary)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
