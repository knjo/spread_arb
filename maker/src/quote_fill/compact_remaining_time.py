"""Source-bound remaining-session-time and nominal-stop analysis.

The analysis projects the immutable 60-session execution action facts.  It
does not reopen raw tapes or recompute fills.  Every source partition marker,
artifact hash, schema, config, and frozen inventory digest is verified by the
compact frozen-fact loader before the supported A/B1-2 rows are aggregated.

Rows are policy alternatives.  Quantiles, routes, ranks, and stop categories
must never be summed into a portfolio or interpreted as pathwise EV.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, time, timedelta
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Iterable

import polars as pl

from . import compact_frozen_facts as frozen


REMAINING_TIME_VERSION = "compact_remaining_time_stop_reason_v1"
BUNDLE_SCHEMA_VERSION = "compact_remaining_time_stop_reason_bundle_v1"
DEFAULT_OUTPUT_ROOT = Path(
    "maker/data/walkforward/compact_remaining_time_stop_reason_60d_v1"
)
SESSION_CUTOFF_LOCAL = time(13, 20)
SESSION_TIMEZONE = "Asia/Taipei"
HEDGE_DELAY_NS = 50_000_000
SECOND_NS = 1_000_000_000
MINUTE_NS = 60 * SECOND_NS
MAX_SESSION_REMAINING_NS = 255 * MINUTE_NS

_ROUTE_RANKS = (
    ("future_ask_spot_taker", "future", "ask", "ASK1"),
    ("future_ask_spot_taker", "future", "ask", "ASK2"),
    ("spot_bid_future_taker", "spot", "bid", "BID1"),
    ("spot_bid_future_taker", "spot", "bid", "BID2"),
)
_TIME_BUCKETS = (
    (0, "lt_5m", 0, 5 * MINUTE_NS),
    (1, "5_to_lt_15m", 5 * MINUTE_NS, 15 * MINUTE_NS),
    (2, "15_to_lt_30m", 15 * MINUTE_NS, 30 * MINUTE_NS),
    (3, "30_to_lt_60m", 30 * MINUTE_NS, 60 * MINUTE_NS),
    (4, "ge_60m", 60 * MINUTE_NS, None),
)
_TARGET_RETREAT_REASONS = ("target_retreat",)
_GATE_REASONS = (
    "future_book_gate",
    "future_exec_book_gate",
    "future_ref_gate",
    "missing_anchor",
    "spot_ref_gate",
    "spot_trial_match",
    "target_not_passive",
    "target_ref_gate",
)
_CUTOFF_REASONS = ("session_cutoff",)
_OTHER_REASONS: tuple[str, ...] = ()
_STOP_CATEGORIES = (
    (0, "target_retreat", _TARGET_RETREAT_REASONS),
    (1, "gate", _GATE_REASONS),
    (2, "cutoff", _CUTOFF_REASONS),
    (3, "other", _OTHER_REASONS),
)
_KNOWN_STOP_REASONS = frozenset(
    reason
    for _, _, reasons in _STOP_CATEGORIES
    for reason in reasons
)
_WAIT_QUANTILES = (0.50, 0.90, 0.95)

_ANALYSIS_COLUMNS = {
    "Date": pl.String,
    "ValueCode": pl.String,
    "route": pl.String,
    "maker_market": pl.String,
    "maker_side": pl.String,
    "boundary_quantile": pl.Int64,
    "raw_order_fact_id": pl.String,
    "target_rank_at_submit": pl.String,
    "submit_recv_time_ns": pl.Int64,
    "nominal_stop_recv_time_ns": pl.Int64,
    "nominal_stop_reason": pl.String,
    "first_fill_recv_time_ns": pl.Int64,
    "full_fill_recv_time_ns": pl.Int64,
    "any_fill": pl.Boolean,
    "full_fill": pl.Boolean,
    "partial_fill": pl.Boolean,
    "cancel_required": pl.Boolean,
    "entry_hedge_decision_time_ns": pl.Int64,
    "entry_hedge_label_observed": pl.Boolean,
    "entry_hedge_executable": pl.Boolean,
}

_COUNT_COLUMNS = (
    "policy_aliases",
    "unique_physical_orders",
    "full_fill_aliases",
    "full_fill_unique_physical_orders",
    "partial_only_aliases",
    "partial_only_unique_physical_orders",
    "no_fill_aliases",
    "no_fill_unique_physical_orders",
    "any_fill_aliases",
    "any_fill_unique_physical_orders",
    "cancel_required_aliases",
    "cancel_required_unique_physical_orders",
    "hedge_label_observed_aliases",
    "hedge_label_observed_unique_physical_orders",
    "hedge_executable_aliases",
    "hedge_executable_unique_physical_orders",
)
_RATE_COLUMNS = (
    "full_fill_rate",
    "partial_only_rate",
    "no_fill_rate",
    "any_fill_rate",
    "cancel_required_rate",
    "hedge_observed_per_full_rate",
    "hedge_executable_per_full_rate",
)
_WAIT_COLUMNS = (
    "wait_to_first_seconds_p50",
    "wait_to_first_seconds_p90",
    "wait_to_first_seconds_p95",
    "wait_to_full_seconds_p50",
    "wait_to_full_seconds_p90",
    "wait_to_full_seconds_p95",
)

SUMMARY_SCHEMA: dict[str, pl.DataType] = {
    "boundary_quantile": pl.Int64,
    "route": pl.String,
    "maker_market": pl.String,
    "maker_side": pl.String,
    "target_rank_at_submit": pl.String,
    "remaining_session_bucket_order": pl.Int64,
    "remaining_session_bucket": pl.String,
    "remaining_session_lower_bound_ns": pl.Int64,
    "remaining_session_upper_bound_ns_exclusive": pl.Int64,
    "nominal_stop_category_order": pl.Int64,
    "nominal_stop_category": pl.String,
    "hedge_delay_ns": pl.Int64,
    **{name: pl.Int64 for name in _COUNT_COLUMNS},
    **{name: pl.Float64 for name in _RATE_COLUMNS},
    **{name: pl.Float64 for name in _WAIT_COLUMNS},
    "cell_populated": pl.Boolean,
    "supported_denominator_only": pl.Boolean,
    "alternative_q_rows_additive": pl.Boolean,
    "route_rows_additive": pl.Boolean,
    "rank_rows_additive": pl.Boolean,
    "stop_category_rows_additive": pl.Boolean,
    "independent_event_labels": pl.Boolean,
    "joint_volume_allocated": pl.Boolean,
    "pathwise_ev_ready": pl.Boolean,
    "realized_nominal_stop_reason_future_label": pl.Boolean,
    "online_stop_reason_feature_ready": pl.Boolean,
}

AUDIT_SCHEMA: dict[str, pl.DataType] = {
    "Date": pl.String,
    "ValueCode": pl.String,
    "source_action_rows": pl.Int64,
    "supported_aliases": pl.Int64,
    "supported_unique_physical_orders": pl.Int64,
    "unsupported_aliases": pl.Int64,
    "full_fill_aliases": pl.Int64,
    "partial_only_aliases": pl.Int64,
    "no_fill_aliases": pl.Int64,
    "any_fill_aliases": pl.Int64,
    "cancel_required_aliases": pl.Int64,
    "hedge_label_observed_aliases": pl.Int64,
    "hedge_executable_aliases": pl.Int64,
    "target_retreat_aliases": pl.Int64,
    "gate_aliases": pl.Int64,
    "cutoff_aliases": pl.Int64,
    "other_stop_aliases": pl.Int64,
    "minimum_remaining_session_ns": pl.Int64,
    "maximum_remaining_session_ns": pl.Int64,
    "source_contract_validated": pl.Boolean,
    "unsupported_outcomes_imputed": pl.Boolean,
    "independent_event_labels": pl.Boolean,
    "joint_volume_allocated": pl.Boolean,
    "pathwise_ev_ready": pl.Boolean,
}

_EXPECTED_ARTIFACTS = {
    "remaining_time_stop_reason_summary.parquet",
    "product_day_audit.parquet",
}
_MARKER_KEYS = {
    "schema_version",
    "complete",
    "analysis_only",
    "formal_roots_mutated",
    "execution_root",
    "source",
    "selection",
    "selection_sha256",
    "implementation_sources",
    "metrics",
    "safety",
    "artifacts",
}


@dataclass(frozen=True)
class RemainingTimeResult:
    summary: pl.DataFrame
    product_day_audit: pl.DataFrame
    source_partitions: int
    source_action_rows: int


def summarize_remaining_time_universe(
    execution_root: Path = frozen.DEFAULT_EXECUTION_ROOT,
) -> RemainingTimeResult:
    """Verify the frozen 60-session source and stream the compact tables."""

    root = Path(execution_root)
    if root.is_symlink():
        raise ValueError("execution root must not be a symlink")
    if root.resolve() != Path(frozen.DEFAULT_EXECUTION_ROOT).resolve():
        raise ValueError("remaining-time v1 requires the canonical execution root")
    manifest_path = root / "execution_partition_manifest.parquet"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("remaining-time execution manifest is absent or a symlink")
    preflight_manifest = pl.read_parquet(manifest_path)
    _validate_source_paths_are_canonical(root, preflight_manifest)
    manifest, paths = frozen._selected_manifest_paths(  # noqa: SLF001
        root, dates=None, value_codes=None
    )
    actions = frozen._scan_actions(paths)  # noqa: SLF001
    frozen._validate_action_contract_lazy(actions)  # noqa: SLF001
    result = build_remaining_time_tables(actions, manifest)
    if (
        result.source_partitions != frozen.EXPECTED_EXECUTION_PARTITIONS
        or result.source_action_rows != frozen.EXPECTED_EXECUTION_ACTION_ROWS
    ):
        raise ValueError("remaining-time source inventory is incomplete")
    return result


def _validate_source_paths_are_canonical(
    execution_root: Path, manifest: pl.DataFrame
) -> None:
    """Reject symlink/repointing even when cloned bytes still hash identically."""

    lexical_root = Path(execution_root)
    resolved_root = lexical_root.resolve()
    root_manifest = lexical_root / "execution_partition_manifest.parquet"
    universe = lexical_root / "product_day_universe.csv"
    for path in (lexical_root, root_manifest, universe):
        if path.is_symlink():
            raise ValueError(f"remaining-time source path is a symlink: {path}")
    for row in manifest.select("Date", "ValueCode", "partition").iter_rows(
        named=True
    ):
        date = str(row["Date"])
        value_code = str(row["ValueCode"])
        lexical_partition = (
            lexical_root / f"Date={date}" / f"ValueCode={value_code}"
        )
        canonical_partition = (
            resolved_root / f"Date={date}" / f"ValueCode={value_code}"
        )
        canonical_relative = (
            Path(frozen.DEFAULT_EXECUTION_ROOT)
            / f"Date={date}"
            / f"ValueCode={value_code}"
        )
        declared = Path(str(row["partition"]))
        allowed_declared = {
            lexical_partition,
            canonical_relative,
            canonical_partition,
        }
        if declared not in allowed_declared:
            raise ValueError(
                "remaining-time manifest partition is not the canonical lexical path: "
                f"{declared}"
            )
        paths = (
            lexical_partition.parent,
            lexical_partition,
            lexical_partition / "complete.json",
            lexical_partition / "execution_action_facts.parquet",
        )
        if any(path.is_symlink() for path in paths):
            raise ValueError(
                f"remaining-time partition lineage contains a symlink: {lexical_partition}"
            )
        if lexical_partition.resolve() != canonical_partition:
            raise ValueError(
                f"remaining-time partition escaped canonical root: {lexical_partition}"
            )


def build_remaining_time_tables(
    actions: pl.LazyFrame | pl.DataFrame,
    manifest: pl.DataFrame,
) -> RemainingTimeResult:
    """Aggregate an already source-validated action projection.

    This function still enforces the analysis-specific denominator, lifecycle,
    cutoff, stop taxonomy, and 50ms hedge invariants.  The formal entry point
    additionally runs the complete frozen action validator above.
    """

    lazy = actions.lazy() if isinstance(actions, pl.DataFrame) else actions
    normalized = _normalize_analysis_projection(lazy)
    manifest_grid = _normalize_manifest_grid(manifest)
    cutoffs = _cutoff_frame(manifest_grid["Date"].unique().to_list())
    enriched = _enrich(normalized, cutoffs.lazy())
    _validate_analysis_projection(enriched)

    aggregated = _aggregate_summary(enriched).collect(engine="streaming")
    summary = _complete_summary_grid(aggregated)
    action_audit = _aggregate_audit(enriched).collect(engine="streaming")
    audit = _complete_product_day_audit(manifest_grid, action_audit)
    result = RemainingTimeResult(
        summary=summary,
        product_day_audit=audit,
        source_partitions=manifest_grid.height,
        source_action_rows=int(manifest_grid["source_action_rows"].sum()),
    )
    _validate_result(result)
    return result


def publish_remaining_time_bundle(
    output: Path = DEFAULT_OUTPUT_ROOT,
    *,
    execution_root: Path = frozen.DEFAULT_EXECUTION_ROOT,
) -> dict[str, object]:
    """Atomically publish and source-recompute-verify the analysis bundle."""

    output = Path(output)
    _validate_output_target(output, Path(execution_root))
    result = summarize_remaining_time_universe(execution_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp.", dir=output.parent)
    )
    try:
        frames = {
            "remaining_time_stop_reason_summary.parquet": result.summary,
            "product_day_audit.parquet": result.product_day_audit,
        }
        for filename, frame in frames.items():
            frame.write_parquet(temporary / filename)
        marker = _build_marker(
            frames,
            temporary,
            result,
            execution_root=Path(execution_root),
        )
        (temporary / "complete.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        verify_remaining_time_bundle(temporary)
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return marker


def _validate_output_target(output: Path, execution_root: Path) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    candidate = output.parent.resolve() / output.name
    source = Path(execution_root).resolve()
    if candidate == source or source in candidate.parents:
        raise ValueError("remaining-time output must not mutate the execution root")


def verify_remaining_time_bundle(output: Path) -> dict[str, object]:
    """Rehash the exact bundle and rebuild both tables from frozen sources."""

    output = Path(output)
    if output.is_symlink() or not output.is_dir():
        raise ValueError("remaining-time bundle must be a real directory")
    marker_path = output / "complete.json"
    marker = _read_json(marker_path)
    if set(marker) != _MARKER_KEYS:
        raise ValueError("remaining-time marker keys mismatch")
    if (
        marker.get("schema_version") != BUNDLE_SCHEMA_VERSION
        or marker.get("complete") is not True
        or marker.get("analysis_only") is not True
        or marker.get("formal_roots_mutated") is not False
    ):
        raise ValueError("remaining-time marker state is invalid")
    actual_entries = {path.name for path in output.iterdir()}
    if actual_entries != _EXPECTED_ARTIFACTS | {"complete.json"}:
        raise ValueError("remaining-time bundle inventory mismatch")
    if any(path.is_symlink() or not path.is_file() for path in output.iterdir()):
        raise ValueError("remaining-time bundle contains a non-file or symlink")

    execution_root = Path(str(marker.get("execution_root", "")))
    if execution_root.resolve() != Path(frozen.DEFAULT_EXECUTION_ROOT).resolve():
        raise ValueError("remaining-time execution-root identity mismatch")
    if marker.get("source") != _source_payload(execution_root):
        raise ValueError("remaining-time frozen source anchors mismatch")
    if marker.get("selection") != _selection_payload():
        raise ValueError("remaining-time selection/config mismatch")
    if marker.get("selection_sha256") != _canonical_sha256(
        _selection_payload()
    ):
        raise ValueError("remaining-time selection hash mismatch")
    if marker.get("implementation_sources") != _implementation_sources():
        raise ValueError("remaining-time implementation lineage mismatch")
    if marker.get("safety") != _safety_payload():
        raise ValueError("remaining-time safety flags mismatch")

    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != _EXPECTED_ARTIFACTS:
        raise ValueError("remaining-time artifact metadata inventory mismatch")
    frames = {
        "remaining_time_stop_reason_summary.parquet": pl.read_parquet(
            output / "remaining_time_stop_reason_summary.parquet"
        ),
        "product_day_audit.parquet": pl.read_parquet(
            output / "product_day_audit.parquet"
        ),
    }
    expected_schemas = {
        "remaining_time_stop_reason_summary.parquet": SUMMARY_SCHEMA,
        "product_day_audit.parquet": AUDIT_SCHEMA,
    }
    for filename, frame in frames.items():
        path = output / filename
        metadata = artifacts.get(filename)
        _verify_artifact_metadata(path, frame, metadata, expected_schemas[filename])

    rebuilt = summarize_remaining_time_universe(execution_root)
    if not frames["remaining_time_stop_reason_summary.parquet"].equals(
        rebuilt.summary, null_equal=True
    ):
        raise ValueError("remaining-time summary differs from source rebuild")
    if not frames["product_day_audit.parquet"].equals(
        rebuilt.product_day_audit, null_equal=True
    ):
        raise ValueError("remaining-time product-day audit differs from source rebuild")
    if marker.get("metrics") != _metrics_payload(rebuilt):
        raise ValueError("remaining-time marker metrics differ from source rebuild")
    return marker


def _normalize_analysis_projection(actions: pl.LazyFrame) -> pl.LazyFrame:
    schema = actions.collect_schema()
    missing = sorted(set(_ANALYSIS_COLUMNS) - set(schema.names()))
    if missing:
        raise ValueError(f"remaining-time actions missing columns: {missing}")
    return actions.select(
        *[
            pl.col(name).cast(dtype, strict=True).alias(name)
            for name, dtype in _ANALYSIS_COLUMNS.items()
        ]
    )


def _normalize_manifest_grid(manifest: pl.DataFrame) -> pl.DataFrame:
    required = {"Date", "ValueCode", "execution_action_facts_rows"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"remaining-time manifest missing columns: {missing}")
    grid = manifest.select(
        pl.col("Date").cast(pl.String, strict=True),
        pl.col("ValueCode").cast(pl.String, strict=True),
        pl.col("execution_action_facts_rows")
        .cast(pl.Int64, strict=True)
        .alias("source_action_rows"),
    ).sort("Date", "ValueCode")
    if (
        grid.is_empty()
        or grid.select("Date", "ValueCode").n_unique() != grid.height
        or grid.filter(
            pl.col("Date").is_null()
            | pl.col("ValueCode").is_null()
            | pl.col("source_action_rows").is_null()
            | (pl.col("Date").str.len_chars() != 8)
            | (pl.col("ValueCode").str.len_chars() == 0)
            | (pl.col("source_action_rows") < 0)
        ).height
    ):
        raise ValueError("remaining-time manifest grid is invalid")
    return grid


def _cutoff_frame(dates: Iterable[str]) -> pl.DataFrame:
    rows = [
        {
            "Date": str(date),
            "session_cutoff_recv_time_ns": _session_cutoff_ns(str(date)),
        }
        for date in sorted({str(value) for value in dates})
    ]
    return pl.DataFrame(
        rows,
        schema={"Date": pl.String, "session_cutoff_recv_time_ns": pl.Int64},
    )


def _session_cutoff_ns(date: str) -> int:
    parsed = datetime.strptime(str(date), "%Y%m%d")
    local_naive = datetime.combine(parsed.date(), SESSION_CUTOFF_LOCAL)
    utc_naive = local_naive - timedelta(hours=8)
    delta = utc_naive - datetime(1970, 1, 1)
    return (
        (delta.days * 86_400 + delta.seconds) * SECOND_NS
        + delta.microseconds * 1_000
    )


def _stop_category_expr() -> pl.Expr:
    reason = pl.col("nominal_stop_reason")
    return (
        pl.when(reason.is_in(list(_TARGET_RETREAT_REASONS)))
        .then(pl.lit("target_retreat"))
        .when(reason.is_in(list(_GATE_REASONS)))
        .then(pl.lit("gate"))
        .when(reason.is_in(list(_CUTOFF_REASONS)))
        .then(pl.lit("cutoff"))
        .when(reason.is_in(list(_OTHER_REASONS)))
        .then(pl.lit("other"))
        .otherwise(pl.lit(None, dtype=pl.String))
    )


def _stop_order_expr() -> pl.Expr:
    return (
        pl.when(pl.col("nominal_stop_category") == "target_retreat")
        .then(pl.lit(0))
        .when(pl.col("nominal_stop_category") == "gate")
        .then(pl.lit(1))
        .when(pl.col("nominal_stop_category") == "cutoff")
        .then(pl.lit(2))
        .when(pl.col("nominal_stop_category") == "other")
        .then(pl.lit(3))
        .otherwise(pl.lit(None, dtype=pl.Int64))
        .cast(pl.Int64)
    )


def _bucket_expr() -> pl.Expr:
    remaining = pl.col("remaining_session_ns")
    return (
        pl.when(remaining < 5 * MINUTE_NS)
        .then(pl.lit("lt_5m"))
        .when(remaining < 15 * MINUTE_NS)
        .then(pl.lit("5_to_lt_15m"))
        .when(remaining < 30 * MINUTE_NS)
        .then(pl.lit("15_to_lt_30m"))
        .when(remaining < 60 * MINUTE_NS)
        .then(pl.lit("30_to_lt_60m"))
        .otherwise(pl.lit("ge_60m"))
    )


def _bucket_order_expr() -> pl.Expr:
    return (
        pl.when(pl.col("remaining_session_ns") < 5 * MINUTE_NS)
        .then(pl.lit(0))
        .when(pl.col("remaining_session_ns") < 15 * MINUTE_NS)
        .then(pl.lit(1))
        .when(pl.col("remaining_session_ns") < 30 * MINUTE_NS)
        .then(pl.lit(2))
        .when(pl.col("remaining_session_ns") < 60 * MINUTE_NS)
        .then(pl.lit(3))
        .otherwise(pl.lit(4))
        .cast(pl.Int64)
    )


def _enrich(actions: pl.LazyFrame, cutoffs: pl.LazyFrame) -> pl.LazyFrame:
    joined = actions.join(cutoffs, on="Date", how="left", validate="m:1")
    supported = frozen._SUPPORTED_ACTION  # noqa: SLF001
    return (
        joined.with_columns(
            supported.alias("_supported"),
            (
                pl.col("session_cutoff_recv_time_ns")
                - pl.col("submit_recv_time_ns")
            ).alias("remaining_session_ns"),
            _stop_category_expr().alias("nominal_stop_category"),
        )
        .with_columns(
            _bucket_expr().alias("remaining_session_bucket"),
            _bucket_order_expr().alias("remaining_session_bucket_order"),
            _stop_order_expr().alias("nominal_stop_category_order"),
            (
                pl.col("first_fill_recv_time_ns")
                - pl.col("submit_recv_time_ns")
            ).alias("_wait_to_first_ns"),
            (
                pl.col("full_fill_recv_time_ns")
                - pl.col("submit_recv_time_ns")
            ).alias("_wait_to_full_ns"),
        )
    )


def _validate_analysis_projection(actions: pl.LazyFrame) -> None:
    supported = pl.col("_supported")
    any_true = pl.col("any_fill") == True  # noqa: E712
    full_true = pl.col("full_fill") == True  # noqa: E712
    partial_true = pl.col("partial_fill") == True  # noqa: E712
    required_null = pl.any_horizontal(
        [
            pl.col(name).is_null()
            for name in (
                "Date",
                "ValueCode",
                "route",
                "maker_market",
                "maker_side",
                "boundary_quantile",
                "raw_order_fact_id",
                "target_rank_at_submit",
                "submit_recv_time_ns",
                "nominal_stop_recv_time_ns",
                "nominal_stop_reason",
                "session_cutoff_recv_time_ns",
                "entry_hedge_label_observed",
                "entry_hedge_executable",
            )
        ]
    )
    supported_outcome_null = supported & pl.any_horizontal(
        [
            pl.col(name).is_null()
            for name in (
                "any_fill",
                "full_fill",
                "partial_fill",
                "cancel_required",
            )
        ]
    )
    invalid = (
        required_null
        | (pl.col("raw_order_fact_id").str.len_chars() == 0).fill_null(True)
        | ~pl.col("boundary_quantile")
        .is_in(list(frozen.SUPPORTED_QUANTILES))
        .fill_null(False)
        | pl.col("nominal_stop_category").is_null()
        | (pl.col("remaining_session_ns") <= 0).fill_null(True)
        | (
            pl.col("remaining_session_ns") > MAX_SESSION_REMAINING_NS
        ).fill_null(True)
        | (
            pl.col("nominal_stop_recv_time_ns")
            < pl.col("submit_recv_time_ns")
        ).fill_null(True)
        | (
            pl.col("nominal_stop_recv_time_ns")
            > pl.col("session_cutoff_recv_time_ns")
        ).fill_null(True)
        | supported_outcome_null
        | (
            supported & (any_true != (full_true | partial_true)).fill_null(True)
        )
        | (supported & (full_true & partial_true).fill_null(True))
        | (
            supported
            & (pl.col("cancel_required") != ~full_true).fill_null(True)
        )
        | (
            supported
            & (
                pl.col("first_fill_recv_time_ns").is_not_null() != any_true
            ).fill_null(True)
        )
        | (
            supported
            & (
                pl.col("full_fill_recv_time_ns").is_not_null() != full_true
            ).fill_null(True)
        )
        | (
            supported
            & any_true
            & (
                (pl.col("_wait_to_first_ns") < 0)
                | (
                    pl.col("first_fill_recv_time_ns")
                    > pl.col("nominal_stop_recv_time_ns")
                )
            ).fill_null(True)
        )
        | (
            supported
            & full_true
            & (
                (pl.col("_wait_to_full_ns") < 0)
                | (
                    pl.col("full_fill_recv_time_ns")
                    > pl.col("nominal_stop_recv_time_ns")
                )
                | (
                    pl.col("full_fill_recv_time_ns")
                    < pl.col("first_fill_recv_time_ns")
                )
            ).fill_null(True)
        )
        | (
            supported
            & (
                pl.col("entry_hedge_label_observed") != full_true
            ).fill_null(True)
        )
        | (
            supported
            & pl.col("entry_hedge_executable")
            & ~pl.col("entry_hedge_label_observed")
        )
        | (
            supported
            & full_true
            & (
                pl.col("entry_hedge_decision_time_ns")
                != pl.col("full_fill_recv_time_ns") + HEDGE_DELAY_NS
            ).fill_null(True)
        )
    )
    invalid_rows = int(
        actions.select(invalid.sum().alias("invalid_rows"))
        .collect(engine="streaming")
        .item()
    )
    if invalid_rows:
        raise ValueError(
            f"remaining-time projection has {invalid_rows} invalid action rows"
        )


def _aggregate_summary(actions: pl.LazyFrame) -> pl.LazyFrame:
    selected = actions.filter(pl.col("_supported"))
    full = pl.col("full_fill")
    partial = pl.col("partial_fill")
    any_fill = pl.col("any_fill")
    no_fill = ~pl.col("any_fill")
    cancel = pl.col("cancel_required")
    observed = pl.col("entry_hedge_label_observed")
    executable = pl.col("entry_hedge_executable")
    groups = [
        "boundary_quantile",
        "route",
        "maker_market",
        "maker_side",
        "target_rank_at_submit",
        "remaining_session_bucket_order",
        "remaining_session_bucket",
        "nominal_stop_category_order",
        "nominal_stop_category",
    ]

    def unique_when(predicate: pl.Expr, alias: str) -> pl.Expr:
        return (
            pl.col("raw_order_fact_id")
            .filter(predicate)
            .n_unique()
            .cast(pl.Int64)
            .alias(alias)
        )

    first_wait = pl.col("_wait_to_first_ns").filter(any_fill) / SECOND_NS
    full_wait = pl.col("_wait_to_full_ns").filter(full) / SECOND_NS
    return selected.group_by(groups).agg(
        pl.len().cast(pl.Int64).alias("policy_aliases"),
        pl.col("raw_order_fact_id")
        .n_unique()
        .cast(pl.Int64)
        .alias("unique_physical_orders"),
        full.sum().cast(pl.Int64).alias("full_fill_aliases"),
        unique_when(full, "full_fill_unique_physical_orders"),
        partial.sum().cast(pl.Int64).alias("partial_only_aliases"),
        unique_when(partial, "partial_only_unique_physical_orders"),
        no_fill.sum().cast(pl.Int64).alias("no_fill_aliases"),
        unique_when(no_fill, "no_fill_unique_physical_orders"),
        any_fill.sum().cast(pl.Int64).alias("any_fill_aliases"),
        unique_when(any_fill, "any_fill_unique_physical_orders"),
        cancel.sum().cast(pl.Int64).alias("cancel_required_aliases"),
        unique_when(cancel, "cancel_required_unique_physical_orders"),
        observed.sum().cast(pl.Int64).alias("hedge_label_observed_aliases"),
        unique_when(observed, "hedge_label_observed_unique_physical_orders"),
        executable.sum().cast(pl.Int64).alias("hedge_executable_aliases"),
        unique_when(executable, "hedge_executable_unique_physical_orders"),
        first_wait.quantile(0.50, interpolation="nearest")
        .cast(pl.Float64)
        .alias("wait_to_first_seconds_p50"),
        first_wait.quantile(0.90, interpolation="nearest")
        .cast(pl.Float64)
        .alias("wait_to_first_seconds_p90"),
        first_wait.quantile(0.95, interpolation="nearest")
        .cast(pl.Float64)
        .alias("wait_to_first_seconds_p95"),
        full_wait.quantile(0.50, interpolation="nearest")
        .cast(pl.Float64)
        .alias("wait_to_full_seconds_p50"),
        full_wait.quantile(0.90, interpolation="nearest")
        .cast(pl.Float64)
        .alias("wait_to_full_seconds_p90"),
        full_wait.quantile(0.95, interpolation="nearest")
        .cast(pl.Float64)
        .alias("wait_to_full_seconds_p95"),
    )


def _summary_grid() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for quantile in frozen.SUPPORTED_QUANTILES:
        for route, market, side, rank in _ROUTE_RANKS:
            for bucket_order, bucket, lower, upper in _TIME_BUCKETS:
                for stop_order, stop_category, _ in _STOP_CATEGORIES:
                    rows.append(
                        {
                            "boundary_quantile": int(quantile),
                            "route": route,
                            "maker_market": market,
                            "maker_side": side,
                            "target_rank_at_submit": rank,
                            "remaining_session_bucket_order": bucket_order,
                            "remaining_session_bucket": bucket,
                            "remaining_session_lower_bound_ns": lower,
                            "remaining_session_upper_bound_ns_exclusive": upper,
                            "nominal_stop_category_order": stop_order,
                            "nominal_stop_category": stop_category,
                        }
                    )
    return pl.from_dicts(rows, infer_schema_length=None).with_columns(
        pl.col("remaining_session_upper_bound_ns_exclusive").cast(pl.Int64)
    )


def _complete_summary_grid(aggregated: pl.DataFrame) -> pl.DataFrame:
    keys = [
        "boundary_quantile",
        "route",
        "maker_market",
        "maker_side",
        "target_rank_at_submit",
        "remaining_session_bucket_order",
        "remaining_session_bucket",
        "nominal_stop_category_order",
        "nominal_stop_category",
    ]
    frame = _summary_grid().join(aggregated, on=keys, how="left", validate="1:1")
    frame = frame.with_columns(
        *[pl.col(name).fill_null(0).cast(pl.Int64) for name in _COUNT_COLUMNS]
    ).with_columns(
        _rate("full_fill_aliases", "policy_aliases", "full_fill_rate"),
        _rate("partial_only_aliases", "policy_aliases", "partial_only_rate"),
        _rate("no_fill_aliases", "policy_aliases", "no_fill_rate"),
        _rate("any_fill_aliases", "policy_aliases", "any_fill_rate"),
        _rate(
            "cancel_required_aliases", "policy_aliases", "cancel_required_rate"
        ),
        _rate(
            "hedge_label_observed_aliases",
            "full_fill_aliases",
            "hedge_observed_per_full_rate",
        ),
        _rate(
            "hedge_executable_aliases",
            "full_fill_aliases",
            "hedge_executable_per_full_rate",
        ),
        (pl.col("policy_aliases") > 0).alias("cell_populated"),
        pl.lit(True).alias("supported_denominator_only"),
        pl.lit(False).alias("alternative_q_rows_additive"),
        pl.lit(False).alias("route_rows_additive"),
        pl.lit(False).alias("rank_rows_additive"),
        pl.lit(False).alias("stop_category_rows_additive"),
        pl.lit(True).alias("independent_event_labels"),
        pl.lit(False).alias("joint_volume_allocated"),
        pl.lit(False).alias("pathwise_ev_ready"),
        pl.lit(True).alias("realized_nominal_stop_reason_future_label"),
        pl.lit(False).alias("online_stop_reason_feature_ready"),
        pl.lit(HEDGE_DELAY_NS, dtype=pl.Int64).alias("hedge_delay_ns"),
    )
    return _cast_select(
        frame.sort(
            "boundary_quantile",
            "route",
            "target_rank_at_submit",
            "remaining_session_bucket_order",
            "nominal_stop_category_order",
        ),
        SUMMARY_SCHEMA,
    )


def _rate(numerator: str, denominator: str, alias: str) -> pl.Expr:
    return (
        pl.when(pl.col(denominator) > 0)
        .then(pl.col(numerator).cast(pl.Float64) / pl.col(denominator))
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias(alias)
    )


def _aggregate_audit(actions: pl.LazyFrame) -> pl.LazyFrame:
    supported = pl.col("_supported")
    full = supported & pl.col("full_fill").fill_null(False)
    partial = supported & pl.col("partial_fill").fill_null(False)
    any_fill = supported & pl.col("any_fill").fill_null(False)
    no_fill = supported & ~pl.col("any_fill").fill_null(False)
    cancel = supported & pl.col("cancel_required").fill_null(False)
    observed = supported & pl.col("entry_hedge_label_observed")
    executable = supported & pl.col("entry_hedge_executable")
    return actions.group_by("Date", "ValueCode").agg(
        pl.len().cast(pl.Int64).alias("observed_source_action_rows"),
        supported.sum().cast(pl.Int64).alias("supported_aliases"),
        pl.col("raw_order_fact_id")
        .filter(supported)
        .n_unique()
        .cast(pl.Int64)
        .alias("supported_unique_physical_orders"),
        (~supported).sum().cast(pl.Int64).alias("unsupported_aliases"),
        full.sum().cast(pl.Int64).alias("full_fill_aliases"),
        partial.sum().cast(pl.Int64).alias("partial_only_aliases"),
        no_fill.sum().cast(pl.Int64).alias("no_fill_aliases"),
        any_fill.sum().cast(pl.Int64).alias("any_fill_aliases"),
        cancel.sum().cast(pl.Int64).alias("cancel_required_aliases"),
        observed.sum().cast(pl.Int64).alias("hedge_label_observed_aliases"),
        executable.sum().cast(pl.Int64).alias("hedge_executable_aliases"),
        (
            supported & (pl.col("nominal_stop_category") == "target_retreat")
        )
        .sum()
        .cast(pl.Int64)
        .alias("target_retreat_aliases"),
        (supported & (pl.col("nominal_stop_category") == "gate"))
        .sum()
        .cast(pl.Int64)
        .alias("gate_aliases"),
        (supported & (pl.col("nominal_stop_category") == "cutoff"))
        .sum()
        .cast(pl.Int64)
        .alias("cutoff_aliases"),
        (supported & (pl.col("nominal_stop_category") == "other"))
        .sum()
        .cast(pl.Int64)
        .alias("other_stop_aliases"),
        pl.col("remaining_session_ns")
        .filter(supported)
        .min()
        .cast(pl.Int64)
        .alias("minimum_remaining_session_ns"),
        pl.col("remaining_session_ns")
        .filter(supported)
        .max()
        .cast(pl.Int64)
        .alias("maximum_remaining_session_ns"),
    )


def _complete_product_day_audit(
    manifest_grid: pl.DataFrame, action_audit: pl.DataFrame
) -> pl.DataFrame:
    count_columns = [
        name
        for name, dtype in AUDIT_SCHEMA.items()
        if dtype == pl.Int64
        and name
        not in {"minimum_remaining_session_ns", "maximum_remaining_session_ns"}
    ]
    frame = manifest_grid.join(
        action_audit, on=["Date", "ValueCode"], how="left", validate="1:1"
    )
    frame = frame.with_columns(
        *[
            pl.col(name).fill_null(0).cast(pl.Int64)
            for name in count_columns
            if name != "source_action_rows"
        ],
        pl.lit(True).alias("source_contract_validated"),
        pl.lit(False).alias("unsupported_outcomes_imputed"),
        pl.lit(True).alias("independent_event_labels"),
        pl.lit(False).alias("joint_volume_allocated"),
        pl.lit(False).alias("pathwise_ev_ready"),
    )
    bad_observed = frame.filter(
        pl.col("observed_source_action_rows").fill_null(0)
        != pl.col("source_action_rows")
    )
    if bad_observed.height:
        raise ValueError("remaining-time per-partition source row count mismatch")
    frame = frame.drop("observed_source_action_rows")
    return _cast_select(frame.sort("Date", "ValueCode"), AUDIT_SCHEMA)


def _cast_select(
    frame: pl.DataFrame, schema: dict[str, pl.DataType]
) -> pl.DataFrame:
    missing = sorted(set(schema) - set(frame.columns))
    if missing:
        raise ValueError(f"remaining-time derived frame missing columns: {missing}")
    return frame.select(
        *[
            pl.col(name).cast(dtype, strict=True).alias(name)
            for name, dtype in schema.items()
        ]
    )


def _validate_result(result: RemainingTimeResult) -> None:
    summary = result.summary
    audit = result.product_day_audit
    if summary.schema != SUMMARY_SCHEMA or summary.height != 240:
        raise ValueError("remaining-time summary schema/grid mismatch")
    if audit.schema != AUDIT_SCHEMA or audit.height != result.source_partitions:
        raise ValueError("remaining-time audit schema/grid mismatch")
    invalid_summary = summary.filter(
        (pl.col("policy_aliases") < 0)
        | (pl.col("unique_physical_orders") > pl.col("policy_aliases"))
        | (
            pl.col("full_fill_aliases")
            + pl.col("partial_only_aliases")
            + pl.col("no_fill_aliases")
            != pl.col("policy_aliases")
        )
        | (
            pl.col("any_fill_aliases")
            != pl.col("full_fill_aliases") + pl.col("partial_only_aliases")
        )
        | (
            pl.col("cancel_required_aliases")
            != pl.col("partial_only_aliases") + pl.col("no_fill_aliases")
        )
        | (
            pl.col("hedge_label_observed_aliases")
            != pl.col("full_fill_aliases")
        )
        | (
            pl.col("hedge_executable_aliases")
            > pl.col("hedge_label_observed_aliases")
        )
        | (pl.col("cell_populated") != (pl.col("policy_aliases") > 0))
        | ~pl.col("supported_denominator_only")
        | pl.col("alternative_q_rows_additive")
        | pl.col("route_rows_additive")
        | pl.col("rank_rows_additive")
        | pl.col("stop_category_rows_additive")
        | ~pl.col("independent_event_labels")
        | pl.col("joint_volume_allocated")
        | pl.col("pathwise_ev_ready")
        | ~pl.col("realized_nominal_stop_reason_future_label")
        | pl.col("online_stop_reason_feature_ready")
        | (pl.col("hedge_delay_ns") != HEDGE_DELAY_NS)
    )
    if invalid_summary.height:
        raise ValueError("remaining-time summary invariants failed")
    _validate_rates_and_waits(summary)

    audit_nonnull = [
        name
        for name in AUDIT_SCHEMA
        if name
        not in {"minimum_remaining_session_ns", "maximum_remaining_session_ns"}
    ]
    invalid_audit = audit.filter(
        pl.any_horizontal([pl.col(name).is_null() for name in audit_nonnull])
        | (pl.col("source_action_rows") < 0)
        | (
            pl.col("supported_aliases") + pl.col("unsupported_aliases")
            != pl.col("source_action_rows")
        )
        | (
            pl.col("full_fill_aliases")
            + pl.col("partial_only_aliases")
            + pl.col("no_fill_aliases")
            != pl.col("supported_aliases")
        )
        | (
            pl.col("any_fill_aliases")
            != pl.col("full_fill_aliases") + pl.col("partial_only_aliases")
        )
        | (
            pl.col("cancel_required_aliases")
            != pl.col("partial_only_aliases") + pl.col("no_fill_aliases")
        )
        | (
            pl.col("target_retreat_aliases")
            + pl.col("gate_aliases")
            + pl.col("cutoff_aliases")
            + pl.col("other_stop_aliases")
            != pl.col("supported_aliases")
        )
        | (
            pl.col("hedge_label_observed_aliases")
            != pl.col("full_fill_aliases")
        )
        | (
            pl.col("hedge_executable_aliases")
            > pl.col("hedge_label_observed_aliases")
        )
        | (
            (pl.col("supported_aliases") > 0)
            & (
                pl.col("minimum_remaining_session_ns").is_null()
                | pl.col("maximum_remaining_session_ns").is_null()
                | (pl.col("minimum_remaining_session_ns") <= 0)
                | (
                    pl.col("maximum_remaining_session_ns")
                    > MAX_SESSION_REMAINING_NS
                )
                | (
                    pl.col("minimum_remaining_session_ns")
                    > pl.col("maximum_remaining_session_ns")
                )
            )
        )
        | (
            (pl.col("supported_aliases") == 0)
            & (
                pl.col("minimum_remaining_session_ns").is_not_null()
                | pl.col("maximum_remaining_session_ns").is_not_null()
            )
        )
        | ~pl.col("source_contract_validated")
        | pl.col("unsupported_outcomes_imputed")
        | ~pl.col("independent_event_labels")
        | pl.col("joint_volume_allocated")
        | pl.col("pathwise_ev_ready")
    )
    if invalid_audit.height:
        raise ValueError("remaining-time product-day audit invariants failed")
    if int(audit["source_action_rows"].sum()) != result.source_action_rows:
        raise ValueError("remaining-time audit source total mismatch")
    for name in (
        "policy_aliases",
        "full_fill_aliases",
        "partial_only_aliases",
        "no_fill_aliases",
        "any_fill_aliases",
        "cancel_required_aliases",
        "hedge_label_observed_aliases",
        "hedge_executable_aliases",
    ):
        audit_name = "supported_aliases" if name == "policy_aliases" else name
        if int(summary[name].sum()) != int(audit[audit_name].sum()):
            raise ValueError(f"remaining-time summary/audit total mismatch: {name}")


def _validate_rates_and_waits(summary: pl.DataFrame) -> None:
    rate_specs = {
        "full_fill_rate": ("full_fill_aliases", "policy_aliases"),
        "partial_only_rate": ("partial_only_aliases", "policy_aliases"),
        "no_fill_rate": ("no_fill_aliases", "policy_aliases"),
        "any_fill_rate": ("any_fill_aliases", "policy_aliases"),
        "cancel_required_rate": ("cancel_required_aliases", "policy_aliases"),
        "hedge_observed_per_full_rate": (
            "hedge_label_observed_aliases",
            "full_fill_aliases",
        ),
        "hedge_executable_per_full_rate": (
            "hedge_executable_aliases",
            "full_fill_aliases",
        ),
    }
    for rate, (numerator, denominator) in rate_specs.items():
        bad = summary.filter(
            pl.when(pl.col(denominator) > 0)
            .then(
                (pl.col(rate) - pl.col(numerator) / pl.col(denominator)).abs()
                > 1e-15
            )
            .otherwise(pl.col(rate).is_not_null())
        )
        if bad.height:
            raise ValueError(f"remaining-time rate is incoherent: {rate}")
    first_bad = summary.filter(
        pl.when(pl.col("any_fill_aliases") > 0)
        .then(
            pl.any_horizontal(
                [pl.col(name).is_null() | (pl.col(name) < 0) for name in _WAIT_COLUMNS[:3]]
            )
            | (pl.col(_WAIT_COLUMNS[0]) > pl.col(_WAIT_COLUMNS[1]))
            | (pl.col(_WAIT_COLUMNS[1]) > pl.col(_WAIT_COLUMNS[2]))
        )
        .otherwise(pl.any_horizontal([pl.col(name).is_not_null() for name in _WAIT_COLUMNS[:3]]))
    )
    full_bad = summary.filter(
        pl.when(pl.col("full_fill_aliases") > 0)
        .then(
            pl.any_horizontal(
                [pl.col(name).is_null() | (pl.col(name) < 0) for name in _WAIT_COLUMNS[3:]]
            )
            | (pl.col(_WAIT_COLUMNS[3]) > pl.col(_WAIT_COLUMNS[4]))
            | (pl.col(_WAIT_COLUMNS[4]) > pl.col(_WAIT_COLUMNS[5]))
        )
        .otherwise(pl.any_horizontal([pl.col(name).is_not_null() for name in _WAIT_COLUMNS[3:]]))
    )
    if first_bad.height or full_bad.height:
        raise ValueError("remaining-time wait quantiles are incoherent")


def _selection_payload() -> dict[str, object]:
    return {
        "analysis_version": REMAINING_TIME_VERSION,
        "boundary_quantiles": list(frozen.SUPPORTED_QUANTILES),
        "route_rank_contract": [
            {
                "route": route,
                "maker_market": market,
                "maker_side": side,
                "target_rank_at_submit": rank,
            }
            for route, market, side, rank in _ROUTE_RANKS
        ],
        "session_cutoff": {
            "local_time": SESSION_CUTOFF_LOCAL.isoformat(),
            "timezone": SESSION_TIMEZONE,
            "exclusive_submit_bound": True,
        },
        "remaining_session_definition": (
            "session_cutoff_recv_time_ns - submit_recv_time_ns"
        ),
        "maximum_remaining_session_ns": MAX_SESSION_REMAINING_NS,
        "remaining_session_buckets": [
            {
                "order": order,
                "name": name,
                "lower_bound_ns_inclusive": lower,
                "upper_bound_ns_exclusive": upper,
            }
            for order, name, lower, upper in _TIME_BUCKETS
        ],
        "stop_reason_mapping": {
            category: list(reasons)
            for _, category, reasons in _STOP_CATEGORIES
        },
        "nominal_stop_reason_timing": "realized_post_submit_future_label",
        "wait_quantiles": list(_WAIT_QUANTILES),
        "wait_quantile_interpolation": "nearest",
        "rate_and_wait_weighting": "supported_policy_alias_rows",
        "physical_count_key": "raw_order_fact_id",
        "physical_counts_safe_to_sum_across_q_route_rank_or_stop": False,
        "hedge_delay_ns": HEDGE_DELAY_NS,
        "full_partial_no_fill_partition": (
            "full + partial_only + no_fill = supported"
        ),
        "cancel_required_overlap": "partial_only + no_fill",
    }


def _source_payload(execution_root: Path) -> dict[str, object]:
    root = Path(execution_root).resolve()
    return {
        "execution_root": str(root),
        "execution_manifest": str(root / "execution_partition_manifest.parquet"),
        "execution_manifest_sha256": frozen.EXPECTED_EXECUTION_MANIFEST_SHA256,
        "partition_source_inventory_sha256": (
            frozen.EXPECTED_PARTITION_SOURCE_INVENTORY_SHA256
        ),
        "execution_config_sha256": frozen.EXPECTED_EXECUTION_CONFIG_SHA256,
        "execution_runner_version": frozen.EXPECTED_EXECUTION_RUNNER_VERSION,
        "expected_partitions": frozen.EXPECTED_EXECUTION_PARTITIONS,
        "expected_action_rows": frozen.EXPECTED_EXECUTION_ACTION_ROWS,
        "expected_sessions": frozen.EXPECTED_SESSIONS,
    }


def _safety_payload() -> dict[str, object]:
    return {
        "supported_denominator_only": True,
        "unsupported_outcomes_imputed": False,
        "raw_tapes_opened": False,
        "fills_or_hedges_recomputed": False,
        "independent_event_labels": True,
        "joint_volume_allocated": False,
        "alternative_q_rows_additive": False,
        "route_rows_additive": False,
        "rank_rows_additive": False,
        "stop_category_rows_additive": False,
        "pathwise_ev_ready": False,
        "realized_nominal_stop_reason_future_label": True,
        "online_stop_reason_feature_ready": False,
        "cost_profile_complete": False,
        "production_strategy_go": False,
    }


def _metrics_payload(result: RemainingTimeResult) -> dict[str, object]:
    summary = result.summary
    audit = result.product_day_audit
    return {
        "summary_rows": summary.height,
        "populated_cells": int(summary["cell_populated"].sum()),
        "source_partitions": result.source_partitions,
        "source_action_rows": result.source_action_rows,
        "supported_policy_alias_inventory": int(summary["policy_aliases"].sum()),
        "unsupported_policy_alias_inventory": int(
            audit["unsupported_aliases"].sum()
        ),
        "full_fill_alias_inventory": int(summary["full_fill_aliases"].sum()),
        "partial_only_alias_inventory": int(
            summary["partial_only_aliases"].sum()
        ),
        "no_fill_alias_inventory": int(summary["no_fill_aliases"].sum()),
        "cancel_required_alias_inventory": int(
            summary["cancel_required_aliases"].sum()
        ),
        "hedge_label_observed_alias_inventory": int(
            summary["hedge_label_observed_aliases"].sum()
        ),
        "hedge_executable_alias_inventory": int(
            summary["hedge_executable_aliases"].sum()
        ),
        "inventory_counts_are_not_cross_q_or_route_performance_claims": True,
    }


def _build_marker(
    frames: dict[str, pl.DataFrame],
    directory: Path,
    result: RemainingTimeResult,
    *,
    execution_root: Path,
) -> dict[str, object]:
    artifacts: dict[str, dict[str, object]] = {}
    for filename, frame in frames.items():
        path = directory / filename
        artifacts[filename] = {
            "kind": "parquet",
            "rows": frame.height,
            "columns": frame.width,
            "schema": [[name, str(dtype)] for name, dtype in frame.schema.items()],
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "complete": True,
        "analysis_only": True,
        "formal_roots_mutated": False,
        "execution_root": str(Path(execution_root).resolve()),
        "source": _source_payload(Path(execution_root)),
        "selection": _selection_payload(),
        "selection_sha256": _canonical_sha256(_selection_payload()),
        "implementation_sources": _implementation_sources(),
        "metrics": _metrics_payload(result),
        "safety": _safety_payload(),
        "artifacts": artifacts,
    }


def _verify_artifact_metadata(
    path: Path,
    frame: pl.DataFrame,
    metadata: object,
    schema: dict[str, pl.DataType],
) -> None:
    if not isinstance(metadata, dict) or set(metadata) != {
        "kind",
        "rows",
        "columns",
        "schema",
        "bytes",
        "sha256",
    }:
        raise ValueError(f"remaining-time artifact metadata invalid: {path.name}")
    expected = {
        "kind": "parquet",
        "rows": frame.height,
        "columns": frame.width,
        "schema": [[name, str(dtype)] for name, dtype in schema.items()],
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }
    if metadata != expected or frame.schema != schema:
        raise ValueError(f"remaining-time artifact mismatch: {path.name}")


def _implementation_sources() -> dict[str, str]:
    paths = (Path(__file__), Path(frozen.__file__))
    return {str(path.resolve()): _file_sha256(path) for path in paths}


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid remaining-time marker: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"invalid remaining-time marker: {path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execution-root", type=Path, default=frozen.DEFAULT_EXECUTION_ROOT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--verify-only", type=Path)
    args = parser.parse_args(argv)
    if args.verify_only is not None:
        marker = verify_remaining_time_bundle(args.verify_only)
        print(json.dumps(marker["metrics"], sort_keys=True))
        return 0
    marker = publish_remaining_time_bundle(
        args.output, execution_root=args.execution_root
    )
    print(json.dumps(marker["metrics"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
