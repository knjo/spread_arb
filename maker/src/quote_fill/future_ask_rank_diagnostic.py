"""Source-bound FUTURE-maker ASK1--ASK5 indexed-truth diagnostic.

This module deliberately does *not* extend the historical stock ``makerFill``
files to futures.  FUTURE-maker orders are evaluated from the already-frozen
execution action facts, whose fills were produced by the exact receive-cursor
``IndexedTradeReplay`` path.  The selected cohort uses the same five fixed
representative dates and complete product-day universe as the existing BID
rank diagnostic.

The unit of every q statistic is ``(Date, ValueCode, boundary_quantile,
raw_order_fact_id)``.  Policy aliases are collapsed only after all physical
submit and outcome fields agree.  A second inventory contains exactly one row
per physical ``raw_order_fact_id`` across q and never chooses one q-dependent
stop outcome as the physical truth.

The existing BID bundle is projected through the same all-date rank-band
metrics for a descriptive, route-separated comparison.  ASK and BID rows are
never pooled because their maker markets, quantities, and sampling contracts
differ.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, Mapping

import polars as pl


FUTURE_ASK_DIAGNOSTIC_VERSION = (
    "future_ask_l1_l5_indexed_truth_v1_physical_q_dedupe"
)
DEFAULT_SAMPLE_DATES = (
    "20260603",  # trade/epoch stress
    "20260703",  # before futures tick-ladder change
    "20260706",  # after futures tick-ladder change
    "20260731",  # low workload
    "20260806",  # median workload
)
DEFAULT_PRODUCT_DAY_COUNTS = (
    ("20260603", 45),
    ("20260703", 44),
    ("20260706", 44),
    ("20260731", 45),
    ("20260806", 45),
)
QUANTILES = (50, 80, 95)
ASK_RANKS = ("ASK1", "ASK2", "ASK3", "ASK4", "ASK5")
HEDGE_DELAY_NS = 50_000_000
ASK_ROUTE = "future_ask_spot_taker"
BID_ROUTE = "spot_bid_future_taker"


ACTION_COLUMNS = (
    "Date",
    "ValueCode",
    "QuoteCode",
    "route",
    "maker_market",
    "maker_side",
    "boundary_quantile",
    "raw_order_fact_id",
    "policy_generation_id",
    "target_rank_at_submit",
    "target_price",
    "initial_queue_ahead",
    "queue_known",
    "intended_quantity",
    "submit_recv_time_ns",
    "submit_event_sequence",
    "submit_row_index",
    "nominal_stop_recv_time_ns",
    "nominal_stop_reason",
    "first_fill_recv_time_ns",
    "first_fill_event_sequence",
    "first_fill_row_index",
    "full_fill_recv_time_ns",
    "full_fill_event_sequence",
    "full_fill_row_index",
    "known_filled_quantity",
    "any_fill",
    "full_fill",
    "partial_fill",
    "trade_through_fill",
    "fill_reason",
    "terminal_recv_time_ns",
    "terminal_reason",
    "cancel_required",
    "lifetime_ms",
    "time_first_to_full_ms",
    "independent_event_label",
    "joint_volume_allocated",
    "entry_hedge_status",
    "entry_hedge_decision_time_ns",
    "entry_hedge_decision_snapshot_recv_time_ns",
    "entry_hedge_decision_snapshot_event_sequence",
    "entry_hedge_decision_snapshot_row_index",
    "entry_hedge_arrival_reference_price",
    "entry_hedge_decision_best_price",
    "entry_hedge_executable_vwap_price",
    "entry_hedge_available_quantity",
    "entry_hedge_executed_quantity",
    "entry_hedge_depth_shortfall",
    "entry_hedge_levels_swept",
    "entry_hedge_signed_latency_slippage_bp",
    "entry_hedge_signed_depth_slippage_bp",
    "entry_hedge_signed_total_slippage_bp",
    "entry_hedge_decision_book_age_ms",
    "entry_hedge_contract_size_shares",
    "entry_hedge_label_observed",
    "entry_hedge_executable",
)

_FLOAT_COLUMNS = {
    "target_price",
    "lifetime_ms",
    "time_first_to_full_ms",
    "entry_hedge_arrival_reference_price",
    "entry_hedge_decision_best_price",
    "entry_hedge_executable_vwap_price",
    "entry_hedge_signed_latency_slippage_bp",
    "entry_hedge_signed_depth_slippage_bp",
    "entry_hedge_signed_total_slippage_bp",
    "entry_hedge_decision_book_age_ms",
}
_STRING_COLUMNS = {
    "Date",
    "ValueCode",
    "QuoteCode",
    "route",
    "maker_market",
    "maker_side",
    "raw_order_fact_id",
    "policy_generation_id",
    "target_rank_at_submit",
    "nominal_stop_reason",
    "fill_reason",
    "terminal_reason",
    "entry_hedge_status",
}
_BOOL_COLUMNS = {
    "queue_known",
    "any_fill",
    "full_fill",
    "partial_fill",
    "trade_through_fill",
    "cancel_required",
    "independent_event_label",
    "joint_volume_allocated",
    "entry_hedge_label_observed",
    "entry_hedge_executable",
}

_PHYSICAL_KEY = (
    "Date",
    "ValueCode",
    "boundary_quantile",
    "raw_order_fact_id",
)
_STABLE_ALIAS_COLUMNS = tuple(
    column for column in ACTION_COLUMNS if column != "policy_generation_id"
)
_RAW_STABLE_COLUMNS = (
    "Date",
    "ValueCode",
    "QuoteCode",
    "route",
    "maker_market",
    "maker_side",
    "raw_order_fact_id",
    "target_rank_at_submit",
    "target_price",
    "initial_queue_ahead",
    "queue_known",
    "intended_quantity",
    "submit_recv_time_ns",
    "submit_event_sequence",
    "submit_row_index",
)

_BID_REQUIRED_COLUMNS = {
    "Date",
    "ValueCode",
    "route",
    "maker_market",
    "maker_side",
    "boundary_quantile",
    "raw_order_fact_id",
    "target_rank_at_submit",
    "rank_group",
    "full_fill",
    "entry_hedge_executable",
    "entry_hedge_signed_total_slippage_bp",
    "submit_event_sequence",
}


@dataclass(frozen=True)
class FutureAskRankDiagnosticConfig:
    execution_root: str
    bid_study_root: str
    dates: tuple[str, ...] = DEFAULT_SAMPLE_DATES
    expected_product_day_counts: tuple[tuple[str, int], ...] = (
        DEFAULT_PRODUCT_DAY_COUNTS
    )
    analysis_version: str = FUTURE_ASK_DIAGNOSTIC_VERSION

    def validate(self) -> None:
        if self.analysis_version != FUTURE_ASK_DIAGNOSTIC_VERSION:
            raise ValueError("unsupported future ASK diagnostic version")
        if self.dates != tuple(sorted(self.dates)) or len(set(self.dates)) != len(
            self.dates
        ):
            raise ValueError("dates must be sorted and unique")
        if not self.dates:
            raise ValueError("dates cannot be empty")
        for value in self.dates:
            try:
                datetime.strptime(value, "%Y%m%d")
            except ValueError as exc:
                raise ValueError(f"invalid study date: {value}") from exc
        expected = dict(self.expected_product_day_counts)
        if tuple(expected) != self.dates:
            raise ValueError(
                "expected_product_day_counts must contain each date in order"
            )
        if any(count <= 0 for count in expected.values()):
            raise ValueError("expected product-day counts must be positive")


@dataclass(frozen=True)
class FutureAskRankDiagnosticResult:
    q_physical_orders: pl.DataFrame
    physical_raw_inventory: pl.DataFrame
    rank_summary: pl.DataFrame
    stop_reason_summary: pl.DataFrame
    coverage: pl.DataFrame
    symmetric_route_comparison: pl.DataFrame
    audit: pl.DataFrame


def run_future_ask_rank_diagnostic(
    config: FutureAskRankDiagnosticConfig,
) -> FutureAskRankDiagnosticResult:
    """Recompute all diagnostic facts from frozen execution partitions."""

    config.validate()
    manifest = _load_selected_manifest(config)
    selected_parts: list[pl.DataFrame] = []
    alias_counts: list[pl.DataFrame] = []

    for row in manifest.iter_rows(named=True):
        date = str(row["Date"])
        value_code = str(row["ValueCode"])
        partition = _resolved_partition(config, row)
        action_path = partition / "execution_action_facts.parquet"
        marker_path = partition / "complete.json"
        _validate_execution_partition(
            action_path,
            marker_path,
            row,
            date=date,
            value_code=value_code,
        )
        selected = _read_selected_actions(
            action_path, date=date, value_code=value_code
        )
        if selected.height:
            selected_parts.append(selected)
            alias_counts.append(
                selected.group_by(*_PHYSICAL_KEY)
                .len()
                .rename({"len": "collapsed_policy_aliases"})
            )

    if not selected_parts:
        raise ValueError("fixed cohort contains no FUTURE-maker ASK1--ASK5 actions")
    aliases = pl.concat(selected_parts, how="vertical_relaxed")
    _validate_selected_actions(aliases)
    _validate_alias_consistency(aliases)
    alias_count = pl.concat(alias_counts, how="vertical_relaxed")
    q_physical = (
        aliases.sort(
            [*_PHYSICAL_KEY, "policy_generation_id"]
        )
        .unique(list(_PHYSICAL_KEY), keep="first")
        .join(alias_count, on=list(_PHYSICAL_KEY), how="left", validate="1:1")
        .rename(
            {"policy_generation_id": "representative_policy_generation_id"}
        )
        .with_columns(
            pl.col("target_rank_at_submit")
            .str.slice(-1)
            .cast(pl.Int64)
            .alias("rank_level"),
            pl.when(pl.col("target_rank_at_submit").is_in(["ASK1", "ASK2"]))
            .then(pl.lit("ASK1_2"))
            .otherwise(pl.lit("ASK3_5"))
            .alias("rank_group"),
            pl.when(pl.col("full_fill"))
            .then(pl.lit("full_fill"))
            .when(pl.col("partial_fill"))
            .then(pl.lit("partial_fill"))
            .otherwise(pl.lit("no_fill"))
            .alias("outcome_class"),
            pl.when(pl.col("submit_event_sequence") == 0)
            .then(pl.lit("anchor"))
            .when(pl.col("submit_event_sequence") == 1)
            .then(pl.lit("future"))
            .when(pl.col("submit_event_sequence") == 2)
            .then(pl.lit("spot"))
            .otherwise(pl.lit("unknown"))
            .alias("submit_event_source"),
            pl.lit(True).alias("indexed_outcome_exact"),
            pl.lit(True).alias("fill_cursor_exact"),
            pl.lit(HEDGE_DELAY_NS).cast(pl.Int64).alias("hedge_delay_ns"),
            pl.lit("spot").alias("opposite_hedge_market"),
            pl.lit("not_used_future_maker_unsupported")
            .alias("legacy_makerfill_status"),
            (
                pl.col("entry_hedge_decision_time_ns").is_not_null()
                | pl.col("entry_hedge_signed_total_slippage_bp").is_not_null()
            ).alias("entry_hedge_payload_present"),
            (
                pl.col("entry_hedge_label_observed")
                & pl.col("entry_hedge_executable")
            ).alias("hedge_metric_eligible"),
            (
                (
                    pl.col("entry_hedge_decision_time_ns").is_not_null()
                    | pl.col("entry_hedge_signed_total_slippage_bp").is_not_null()
                )
                & (~pl.col("entry_hedge_label_observed"))
            ).alias("unobserved_raw_hedge_payload_excluded"),
            pl.lit(True).alias("analysis_only"),
        )
        .sort(
            [
                "Date",
                "ValueCode",
                "boundary_quantile",
                "submit_recv_time_ns",
                "submit_event_sequence",
                "submit_row_index",
                "raw_order_fact_id",
            ]
        )
    )
    if q_physical.select(*_PHYSICAL_KEY).n_unique() != q_physical.height:
        raise ValueError("q physical dedupe key is not unique")
    _validate_q_physical(q_physical)

    raw_inventory = _physical_raw_inventory(q_physical)
    summary = _rank_summary(q_physical)
    stops = _stop_reason_summary(q_physical)
    coverage = _coverage(manifest, q_physical)
    bid = _load_bid_projection(Path(config.bid_study_root), config.dates)
    comparison = _symmetric_route_comparison(q_physical, bid)
    audit = _audit(manifest, aliases, q_physical, raw_inventory)
    return FutureAskRankDiagnosticResult(
        q_physical_orders=q_physical,
        physical_raw_inventory=raw_inventory,
        rank_summary=summary,
        stop_reason_summary=stops,
        coverage=coverage,
        symmetric_route_comparison=comparison,
        audit=audit,
    )


def publish_future_ask_rank_diagnostic(
    output_root: Path,
    config: FutureAskRankDiagnosticConfig,
) -> Path:
    """Atomically publish a fail-closed, source-bound diagnostic bundle."""

    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.tmp-", dir=output_root.parent
        )
    )
    try:
        inventory = _build_source_inventory(config)
        result = run_future_ask_rank_diagnostic(config)
        frames = {
            "q_physical_orders.parquet": result.q_physical_orders,
            "physical_raw_inventory.parquet": result.physical_raw_inventory,
            "rank_summary.parquet": result.rank_summary,
            "stop_reason_summary.parquet": result.stop_reason_summary,
            "coverage.parquet": result.coverage,
            "symmetric_route_comparison.parquet": (
                result.symmetric_route_comparison
            ),
            "audit.parquet": result.audit,
            "input_inventory.parquet": inventory,
        }
        artifacts: dict[str, dict[str, object]] = {}
        for name, frame in frames.items():
            path = stage / name
            frame.write_parquet(path, compression="zstd")
            artifacts[name] = _artifact_record(path, frame)

        config_payload = asdict(config)
        implementation = _implementation_identity()
        marker = {
            "status": "complete",
            "analysis_version": FUTURE_ASK_DIAGNOSTIC_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": config_payload,
            "config_sha256": _canonical_sha256(config_payload),
            "implementation_files": implementation,
            "implementation_identity_sha256": _canonical_sha256(
                implementation
            ),
            "artifacts": artifacts,
            "safety": _safety_contract(),
        }
        (stage / "complete.json").write_text(
            json.dumps(marker, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        stage.rename(output_root)
        return output_root
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def verify_future_ask_rank_diagnostic(
    output_root: Path,
) -> dict[str, object]:
    """Verify hashes, schemas, lineage, invariants, and full recomputation."""

    output_root = Path(output_root)
    marker_path = output_root / "complete.json"
    if not marker_path.is_file():
        raise FileNotFoundError(marker_path)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("status") != "complete":
        raise ValueError("future ASK marker is incomplete")
    if marker.get("analysis_version") != FUTURE_ASK_DIAGNOSTIC_VERSION:
        raise ValueError("future ASK analysis version mismatch")
    if marker.get("safety") != _safety_contract():
        raise ValueError("future ASK safety contract mismatch")

    expected_files = {
        "q_physical_orders.parquet",
        "physical_raw_inventory.parquet",
        "rank_summary.parquet",
        "stop_reason_summary.parquet",
        "coverage.parquet",
        "symmetric_route_comparison.parquet",
        "audit.parquet",
        "input_inventory.parquet",
    }
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != expected_files:
        raise ValueError("future ASK artifact declaration mismatch")
    actual_files = {path.name for path in output_root.iterdir() if path.is_file()}
    if actual_files != expected_files | {"complete.json"}:
        raise ValueError("future ASK root file set mismatch")
    for name in sorted(expected_files):
        path = output_root / name
        frame = pl.read_parquet(path)
        if _artifact_record(path, frame) != artifacts[name]:
            raise ValueError(f"future ASK artifact mismatch: {name}")

    raw_config = marker.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError("future ASK config is not an object")
    if _canonical_sha256(raw_config) != marker.get("config_sha256"):
        raise ValueError("future ASK config hash mismatch")
    config = FutureAskRankDiagnosticConfig(
        execution_root=str(raw_config["execution_root"]),
        bid_study_root=str(raw_config["bid_study_root"]),
        dates=tuple(str(value) for value in raw_config["dates"]),
        expected_product_day_counts=tuple(
            (str(date), int(count))
            for date, count in raw_config["expected_product_day_counts"]
        ),
        analysis_version=str(raw_config["analysis_version"]),
    )
    config.validate()
    implementation = marker.get("implementation_files")
    if (
        not isinstance(implementation, dict)
        or implementation != _implementation_identity()
        or _canonical_sha256(implementation)
        != marker.get("implementation_identity_sha256")
    ):
        raise ValueError("future ASK implementation identity mismatch")

    expected_inventory = _build_source_inventory(config)
    _assert_frame_exact(
        pl.read_parquet(output_root / "input_inventory.parquet"),
        expected_inventory,
        "input_inventory.parquet",
    )
    recomputed = run_future_ask_rank_diagnostic(config)
    expected_frames = {
        "q_physical_orders.parquet": recomputed.q_physical_orders,
        "physical_raw_inventory.parquet": recomputed.physical_raw_inventory,
        "rank_summary.parquet": recomputed.rank_summary,
        "stop_reason_summary.parquet": recomputed.stop_reason_summary,
        "coverage.parquet": recomputed.coverage,
        "symmetric_route_comparison.parquet": (
            recomputed.symmetric_route_comparison
        ),
        "audit.parquet": recomputed.audit,
    }
    for name, expected in expected_frames.items():
        _assert_frame_exact(pl.read_parquet(output_root / name), expected, name)
    _verify_persisted_invariants(output_root)
    return marker


def _load_selected_manifest(
    config: FutureAskRankDiagnosticConfig,
) -> pl.DataFrame:
    path = Path(config.execution_root) / "execution_partition_manifest.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pl.read_parquet(path).with_columns(
        pl.col("Date").cast(pl.String), pl.col("ValueCode").cast(pl.String)
    )
    required = {
        "Date",
        "ValueCode",
        "partition",
        "config_sha256",
        "complete",
        "execution_action_facts_rows",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"execution manifest missing columns: {missing}")
    if frame.select("Date", "ValueCode").n_unique() != frame.height:
        raise ValueError("execution manifest keys are duplicated")
    selected = frame.filter(pl.col("Date").is_in(config.dates)).sort(
        "Date", "ValueCode"
    )
    expected = dict(config.expected_product_day_counts)
    counts = dict(
        selected.group_by("Date")
        .len()
        .select("Date", "len")
        .iter_rows()
    )
    if counts != expected:
        raise ValueError(
            f"fixed product-day cohort mismatch: expected={expected}, actual={counts}"
        )
    if selected.filter(pl.col("complete") != True).height:  # noqa: E712
        raise ValueError("fixed product-day cohort contains incomplete partitions")
    return selected


def _resolved_partition(
    config: FutureAskRankDiagnosticConfig, row: Mapping[str, object]
) -> Path:
    partition = Path(str(row["partition"]))
    if partition.is_absolute():
        return partition
    direct = Path(config.execution_root).parent.parent.parent / partition
    if direct.exists():
        return direct
    return Path.cwd() / partition


def _validate_execution_partition(
    action_path: Path,
    marker_path: Path,
    manifest_row: Mapping[str, object],
    *,
    date: str,
    value_code: str,
) -> None:
    from .execution_runner import EXECUTION_RUNNER_VERSION

    if not action_path.is_file() or not marker_path.is_file():
        raise FileNotFoundError(f"{date}/{value_code}: missing action or marker")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if (
        marker.get("complete") is not True
        or str(marker.get("Date")) != date
        or str(marker.get("ValueCode")) != value_code
        or marker.get("runner_version") != EXECUTION_RUNNER_VERSION
        or marker.get("config_sha256") != manifest_row["config_sha256"]
    ):
        raise ValueError(f"{date}/{value_code}: execution marker identity drift")
    config = marker.get("config")
    if (
        not isinstance(config, dict)
        or _canonical_sha256(config) != marker.get("config_sha256")
        or int(config.get("hedge_delay_ns", -1)) != HEDGE_DELAY_NS
        or ASK_ROUTE not in config.get("routes", [])
    ):
        raise ValueError(f"{date}/{value_code}: execution config drift")
    semantics = marker.get("fact_semantics")
    if (
        not isinstance(semantics, dict)
        or int(semantics.get("hedge_delay_ns", -1)) != HEDGE_DELAY_NS
        or semantics.get("independent_event_label") is not True
        or semantics.get("joint_volume_allocated") is not False
    ):
        raise ValueError(f"{date}/{value_code}: execution semantics drift")
    artifacts = marker.get("artifacts")
    meta = (
        artifacts.get("execution_action_facts.parquet")
        if isinstance(artifacts, dict)
        else None
    )
    if not isinstance(meta, dict):
        raise ValueError(f"{date}/{value_code}: missing action artifact metadata")
    schema = pl.read_parquet_schema(action_path)
    rows = _parquet_rows(action_path)
    if (
        _file_sha256(action_path) != meta.get("sha256")
        or action_path.stat().st_size != int(meta.get("bytes", -1))
        or rows != int(meta.get("rows", -1))
        or len(schema) != int(meta.get("columns", -1))
        or rows != int(manifest_row["execution_action_facts_rows"])
    ):
        raise ValueError(f"{date}/{value_code}: execution action lineage drift")


def _read_selected_actions(
    action_path: Path, *, date: str, value_code: str
) -> pl.DataFrame:
    schema = pl.read_parquet_schema(action_path)
    missing = sorted(set(ACTION_COLUMNS) - set(schema))
    rows = _parquet_rows(action_path)
    if missing:
        if rows == 0:
            return _empty_actions()
        raise ValueError(
            f"{date}/{value_code}: action facts missing columns {missing}"
        )
    expressions = []
    for column in ACTION_COLUMNS:
        expression = pl.col(column)
        if column in _FLOAT_COLUMNS:
            expression = expression.cast(pl.Float64)
        elif column in _STRING_COLUMNS:
            expression = expression.cast(pl.String)
        elif column in _BOOL_COLUMNS:
            expression = expression.cast(pl.Boolean)
        else:
            expression = expression.cast(pl.Int64)
        expressions.append(expression.alias(column))
    selected = (
        pl.scan_parquet(action_path)
        .filter(
            (pl.col("route") == ASK_ROUTE)
            & (pl.col("maker_market") == "future")
            & (pl.col("maker_side") == "ask")
            & pl.col("target_rank_at_submit").is_in(ASK_RANKS)
            & pl.col("boundary_quantile").is_in(QUANTILES)
        )
        .select(*expressions)
        .collect(engine="streaming")
    )
    if selected.filter(
        (pl.col("Date") != date) | (pl.col("ValueCode") != value_code)
    ).height:
        raise ValueError(f"{date}/{value_code}: partition key drift")
    return selected


def _empty_actions() -> pl.DataFrame:
    schema: dict[str, pl.DataType] = {}
    for column in ACTION_COLUMNS:
        if column in _FLOAT_COLUMNS:
            schema[column] = pl.Float64
        elif column in _STRING_COLUMNS:
            schema[column] = pl.String
        elif column in _BOOL_COLUMNS:
            schema[column] = pl.Boolean
        else:
            schema[column] = pl.Int64
    return pl.DataFrame(schema=schema)


def _validate_selected_actions(actions: pl.DataFrame) -> None:
    if actions.is_empty():
        raise ValueError("selected ASK actions cannot be empty")
    invalid = actions.filter(
        pl.col("raw_order_fact_id").is_null()
        | pl.col("policy_generation_id").is_null()
        | pl.col("target_price").is_null()
        | (~pl.col("target_price").is_finite())
        | (pl.col("target_price") <= 0)
        | pl.col("initial_queue_ahead").is_null()
        | (pl.col("initial_queue_ahead") < 0)
        | (pl.col("queue_known") != True).fill_null(True)  # noqa: E712
        | (pl.col("intended_quantity") != 1).fill_null(True)
        | pl.col("submit_recv_time_ns").is_null()
        | pl.col("submit_event_sequence").is_null()
        | pl.col("submit_row_index").is_null()
        | pl.col("nominal_stop_recv_time_ns").is_null()
        | (
            pl.col("nominal_stop_recv_time_ns")
            < pl.col("submit_recv_time_ns")
        )
        | pl.col("nominal_stop_reason").is_null()
        | pl.col("known_filled_quantity").is_null()
        | ~pl.col("known_filled_quantity").is_in([0, 1])
        | pl.col("any_fill").is_null()
        | pl.col("full_fill").is_null()
        | pl.col("partial_fill").is_null()
        | (
            pl.col("any_fill")
            != (pl.col("known_filled_quantity") > 0)
        ).fill_null(True)
        | (
            pl.col("full_fill")
            != (pl.col("known_filled_quantity") == 1)
        ).fill_null(True)
        | pl.col("partial_fill")
        | (pl.col("any_fill") != pl.col("full_fill")).fill_null(True)
        | (pl.col("independent_event_label") != True).fill_null(True)  # noqa: E712
        | (pl.col("joint_volume_allocated") != False).fill_null(True)  # noqa: E712
        | (
            pl.col("full_fill")
            & pl.col("full_fill_recv_time_ns").is_null()
        )
        | (
            (~pl.col("full_fill"))
            & pl.col("full_fill_recv_time_ns").is_not_null()
        )
        | (
            pl.col("full_fill")
            & (pl.col("terminal_reason") != "full_fill")
        ).fill_null(True)
        | (
            (~pl.col("full_fill"))
            & (pl.col("terminal_reason") == "full_fill")
        ).fill_null(True)
        | (
            pl.col("entry_hedge_label_observed")
            != pl.col("full_fill")
        ).fill_null(True)
        | (
            pl.col("full_fill")
            & (
                pl.col("entry_hedge_decision_time_ns")
                != pl.col("full_fill_recv_time_ns") + HEDGE_DELAY_NS
            )
        ).fill_null(True)
        | (
            pl.col("entry_hedge_executable")
            & (~pl.col("entry_hedge_label_observed"))
        ).fill_null(True)
        | (
            pl.col("entry_hedge_executable")
            & pl.col("entry_hedge_signed_total_slippage_bp").is_null()
        )
    )
    if invalid.height:
        raise ValueError(
            f"selected ASK actions contain {invalid.height} incoherent rows"
        )
    if actions.select("policy_generation_id").n_unique() != actions.height:
        raise ValueError("selected ASK policy_generation_id is duplicated")


def _validate_alias_consistency(actions: pl.DataFrame) -> None:
    check_columns = [
        column for column in _STABLE_ALIAS_COLUMNS if column not in _PHYSICAL_KEY
    ]
    grouped = actions.group_by(*_PHYSICAL_KEY).agg(
        *(pl.col(column).n_unique().alias(column) for column in check_columns)
    )
    invalid = grouped.filter(
        pl.any_horizontal(
            *(pl.col(column) != 1 for column in check_columns)
        )
    )
    if invalid.height:
        raise ValueError(
            "policy aliases disagree within q physical raw order: "
            f"{invalid.height} keys"
        )


def _validate_q_physical(frame: pl.DataFrame) -> None:
    if set(frame.get_column("boundary_quantile").unique()) != set(QUANTILES):
        raise ValueError("ASK diagnostic is missing a required q cohort")
    if set(frame.get_column("target_rank_at_submit").unique()) != set(ASK_RANKS):
        raise ValueError("ASK diagnostic is missing a required visible rank")
    if frame.filter(pl.col("outcome_class") == "partial_fill").height:
        raise ValueError("one-contract FUTURE maker cannot have a partial label")
    if frame.filter(
        (pl.col("legacy_makerfill_status") != "not_used_future_maker_unsupported")
        | (pl.col("indexed_outcome_exact") != True)  # noqa: E712
        | (pl.col("hedge_delay_ns") != HEDGE_DELAY_NS)
        | (
            pl.col("hedge_metric_eligible")
            != (
                pl.col("entry_hedge_label_observed")
                & pl.col("entry_hedge_executable")
            )
        ).fill_null(True)
    ).height:
        raise ValueError("ASK diagnostic truth/legacy/hedge contract drift")


def _physical_raw_inventory(q_physical: pl.DataFrame) -> pl.DataFrame:
    raw_key = {"Date", "ValueCode", "raw_order_fact_id"}
    check_columns = [
        column for column in _RAW_STABLE_COLUMNS if column not in raw_key
    ]
    stable = q_physical.group_by("Date", "ValueCode", "raw_order_fact_id").agg(
        *(pl.col(column).n_unique().alias(column) for column in check_columns)
    )
    if stable.filter(
        pl.any_horizontal(*(pl.col(column) != 1 for column in check_columns))
    ).height:
        raise ValueError("physical raw submit state varies across q")

    first = q_physical.sort(
        "Date", "ValueCode", "raw_order_fact_id", "boundary_quantile"
    ).unique(["Date", "ValueCode", "raw_order_fact_id"], keep="first")
    membership = q_physical.group_by(
        "Date", "ValueCode", "raw_order_fact_id"
    ).agg(
        pl.col("boundary_quantile").n_unique().alias("q_membership_count"),
        (pl.col("boundary_quantile") == 50).any().alias("in_q50"),
        (pl.col("boundary_quantile") == 80).any().alias("in_q80"),
        (pl.col("boundary_quantile") == 95).any().alias("in_q95"),
        pl.col("full_fill").n_unique().alias("full_fill_variants_across_q"),
        pl.col("terminal_reason")
        .n_unique()
        .alias("terminal_reason_variants_across_q"),
        pl.col("nominal_stop_recv_time_ns")
        .n_unique()
        .alias("stop_time_variants_across_q"),
        pl.col("collapsed_policy_aliases")
        .sum()
        .cast(pl.Int64)
        .alias("source_policy_aliases_across_q"),
    ).with_columns(
        (
            (pl.col("full_fill_variants_across_q") > 1)
            | (pl.col("terminal_reason_variants_across_q") > 1)
            | (pl.col("stop_time_variants_across_q") > 1)
        ).alias("policy_outcome_varies_across_q")
    )
    return (
        first.select(
            *_RAW_STABLE_COLUMNS,
            "rank_level",
            "rank_group",
            "submit_event_source",
            "legacy_makerfill_status",
            "indexed_outcome_exact",
        )
        .join(
            membership,
            on=["Date", "ValueCode", "raw_order_fact_id"],
            how="left",
            validate="1:1",
        )
        .with_columns(
            pl.lit("no_q_outcome_selected_in_physical_inventory")
            .alias("outcome_semantics")
        )
        .sort("Date", "ValueCode", "submit_recv_time_ns", "raw_order_fact_id")
    )


def _rank_summary(frame: pl.DataFrame) -> pl.DataFrame:
    rows: list[pl.DataFrame] = []
    for date_scope, scoped in (
        ("by_date", frame),
        ("all_dates", frame.with_columns(pl.lit("__all__").alias("Date"))),
    ):
        for rank_scope, keys in (
            (
                "individual_rank",
                [
                    "Date",
                    "boundary_quantile",
                    "target_rank_at_submit",
                    "rank_group",
                ],
            ),
            ("rank_group", ["Date", "boundary_quantile", "rank_group"]),
        ):
            grouped = scoped.group_by(*keys).agg(*_metric_expressions())
            if rank_scope == "rank_group":
                grouped = grouped.with_columns(
                    pl.col("rank_group").alias("target_rank_at_submit")
                )
            grouped = _metric_rates(grouped).with_columns(
                pl.lit(date_scope).alias("date_scope"),
                pl.lit(rank_scope).alias("rank_scope"),
                pl.lit(ASK_ROUTE).alias("route"),
                pl.lit("future").alias("maker_market"),
                pl.lit("ask").alias("maker_side"),
                pl.lit("q_physical_raw_order").alias("population_unit"),
                pl.lit(True).alias("indexed_outcome_exact"),
                pl.lit(False).alias("legacy_makerfill_used"),
                pl.lit(False).alias("joint_volume_allocated"),
            )
            prefix = [
                "date_scope",
                "Date",
                "route",
                "maker_market",
                "maker_side",
                "population_unit",
                "rank_scope",
                "boundary_quantile",
                "target_rank_at_submit",
                "rank_group",
            ]
            rows.append(
                grouped.select(
                    *prefix,
                    *(column for column in grouped.columns if column not in prefix),
                )
            )
    return pl.concat(rows, how="vertical_relaxed").sort(
        "date_scope",
        "Date",
        "rank_scope",
        "boundary_quantile",
        "target_rank_at_submit",
    )


def _metric_expressions() -> tuple[pl.Expr, ...]:
    executable_slip = pl.col("entry_hedge_signed_total_slippage_bp").filter(
        pl.col("hedge_metric_eligible")
    )
    return (
        pl.len().cast(pl.Int64).alias("n"),
        pl.col("full_fill").sum().cast(pl.Int64).alias("full_fill_count"),
        pl.col("partial_fill").sum().cast(pl.Int64).alias("partial_fill_count"),
        (~pl.col("any_fill")).sum().cast(pl.Int64).alias("no_fill_count"),
        pl.col("any_fill").sum().cast(pl.Int64).alias("any_fill_count"),
        pl.col("cancel_required").sum().cast(pl.Int64).alias("cancel_required_count"),
        pl.col("trade_through_fill")
        .sum()
        .cast(pl.Int64)
        .alias("trade_through_fill_count"),
        pl.col("entry_hedge_label_observed")
        .sum()
        .cast(pl.Int64)
        .alias("hedge_label_observed_count"),
        pl.col("entry_hedge_executable")
        .sum()
        .cast(pl.Int64)
        .alias("hedge_executable_count"),
        pl.col("unobserved_raw_hedge_payload_excluded")
        .sum()
        .cast(pl.Int64)
        .alias("unobserved_raw_hedge_payload_excluded_count"),
        executable_slip.count().cast(pl.Int64).alias("hedge_slippage_n"),
        executable_slip.mean().round(12).alias("hedge_slippage_bp_mean"),
        executable_slip.median().round(12).alias("hedge_slippage_bp_p50"),
        executable_slip
        .quantile(0.9, interpolation="nearest")
        .round(12)
        .alias("hedge_slippage_bp_p90"),
        pl.col("lifetime_ms").median().round(6).alias("lifetime_ms_p50"),
        pl.col("lifetime_ms")
        .quantile(0.9, interpolation="nearest")
        .round(6)
        .alias("lifetime_ms_p90"),
        pl.col("time_first_to_full_ms")
        .filter(pl.col("full_fill"))
        .median()
        .round(6)
        .alias("time_to_full_ms_p50"),
        pl.col("time_first_to_full_ms")
        .filter(pl.col("full_fill"))
        .quantile(0.9, interpolation="nearest")
        .round(6)
        .alias("time_to_full_ms_p90"),
    )


def _metric_rates(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        (100.0 * pl.col("full_fill_count") / pl.col("n"))
        .round(12)
        .alias("full_fill_pct"),
        (100.0 * pl.col("partial_fill_count") / pl.col("n"))
        .round(12)
        .alias("partial_fill_pct"),
        (100.0 * pl.col("no_fill_count") / pl.col("n"))
        .round(12)
        .alias("no_fill_pct"),
        pl.when(pl.col("full_fill_count") > 0)
        .then(
            100.0
            * pl.col("hedge_executable_count")
            / pl.col("full_fill_count")
        )
        .otherwise(None)
        .round(12)
        .alias("hedge_executable_given_full_pct"),
    )


def _stop_reason_summary(frame: pl.DataFrame) -> pl.DataFrame:
    rows: list[pl.DataFrame] = []
    for date_scope, scoped in (
        ("by_date", frame),
        ("all_dates", frame.with_columns(pl.lit("__all__").alias("Date"))),
    ):
        grouped = scoped.group_by(
            "Date",
            "boundary_quantile",
            "rank_group",
            "outcome_class",
            "nominal_stop_reason",
            "terminal_reason",
        ).agg(pl.len().cast(pl.Int64).alias("n"))
        denominators = scoped.group_by(
            "Date", "boundary_quantile", "rank_group"
        ).agg(pl.len().cast(pl.Int64).alias("rank_group_n"))
        rows.append(
            grouped.join(
                denominators,
                on=["Date", "boundary_quantile", "rank_group"],
                how="left",
                validate="m:1",
            ).with_columns(
                (100.0 * pl.col("n") / pl.col("rank_group_n"))
                .round(12)
                .alias("pct_of_rank_group"),
                (pl.col("outcome_class") != "full_fill").alias("stop_applied"),
                pl.lit(date_scope).alias("date_scope"),
                pl.lit(ASK_ROUTE).alias("route"),
            )
        )
    prefix = [
        "date_scope",
        "Date",
        "route",
        "boundary_quantile",
        "rank_group",
        "outcome_class",
        "nominal_stop_reason",
        "terminal_reason",
    ]
    result = pl.concat(rows, how="vertical_relaxed")
    return result.select(
        *prefix, *(column for column in result.columns if column not in prefix)
    ).sort(*prefix)


def _coverage(manifest: pl.DataFrame, frame: pl.DataFrame) -> pl.DataFrame:
    base = manifest.select(
        "Date",
        "ValueCode",
        pl.col("execution_action_facts_rows").cast(pl.Int64),
        pl.col("complete").alias("execution_partition_complete"),
    ).join(pl.DataFrame({"boundary_quantile": QUANTILES}), how="cross")
    grouped = frame.group_by("Date", "ValueCode", "boundary_quantile").agg(
        pl.len().cast(pl.Int64).alias("q_physical_orders"),
        *(
            (pl.col("target_rank_at_submit") == rank)
            .sum()
            .cast(pl.Int64)
            .alias(f"{rank.lower()}_orders")
            for rank in ASK_RANKS
        ),
        pl.col("full_fill").sum().cast(pl.Int64).alias("full_fill_count"),
        pl.col("partial_fill").sum().cast(pl.Int64).alias("partial_fill_count"),
        (~pl.col("any_fill")).sum().cast(pl.Int64).alias("no_fill_count"),
    )
    count_columns = [
        "q_physical_orders",
        *(f"{rank.lower()}_orders" for rank in ASK_RANKS),
        "full_fill_count",
        "partial_fill_count",
        "no_fill_count",
    ]
    return (
        base.join(
            grouped,
            on=["Date", "ValueCode", "boundary_quantile"],
            how="left",
            validate="1:1",
        )
        .with_columns(*(pl.col(column).fill_null(0) for column in count_columns))
        .with_columns(
            (pl.col("q_physical_orders") > 0).alias("has_selected_ask_rank"),
            pl.lit(True).alias("in_fixed_product_day_cohort"),
            pl.lit(False).alias("legacy_makerfill_read"),
        )
        .sort("Date", "ValueCode", "boundary_quantile")
    )


def _load_bid_projection(root: Path, dates: tuple[str, ...]) -> pl.DataFrame:
    marker_path = root / "complete.json"
    alias_path = root / "alias_comparison.parquet"
    if not marker_path.is_file() or not alias_path.is_file():
        raise FileNotFoundError("existing BID diagnostic marker/alias is missing")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    artifacts = marker.get("artifacts")
    alias_meta = (
        artifacts.get("alias_comparison.parquet")
        if isinstance(artifacts, dict)
        else None
    )
    if (
        marker.get("status") != "complete"
        or not isinstance(alias_meta, dict)
        or _file_sha256(alias_path) != alias_meta.get("sha256")
        or alias_path.stat().st_size != int(alias_meta.get("bytes", -1))
    ):
        raise ValueError("existing BID diagnostic lineage mismatch")
    marker_config = marker.get("config")
    if (
        not isinstance(marker_config, dict)
        or tuple(str(value) for value in marker_config.get("dates", [])) != dates
    ):
        raise ValueError("existing BID diagnostic date cohort mismatch")
    schema = pl.read_parquet_schema(alias_path)
    missing = sorted(_BID_REQUIRED_COLUMNS - set(schema))
    if missing:
        raise ValueError(f"existing BID projection missing columns: {missing}")
    bid = pl.read_parquet(alias_path, columns=sorted(_BID_REQUIRED_COLUMNS))
    invalid = bid.filter(
        (pl.col("route") != BID_ROUTE)
        | (pl.col("maker_market") != "spot")
        | (pl.col("maker_side") != "bid")
        | ~pl.col("target_rank_at_submit").is_in(
            ["BID1", "BID2", "BID3", "BID4", "BID5"]
        )
        | ~pl.col("rank_group").is_in(["L1_2", "L3_5"])
        | (pl.col("submit_event_sequence") != 2)
        | pl.col("full_fill").is_null()
        | pl.col("entry_hedge_executable").is_null()
        | (
            pl.col("entry_hedge_executable")
            & pl.col("entry_hedge_signed_total_slippage_bp").is_null()
        )
    )
    if invalid.height:
        raise ValueError("existing BID projection contains mixed/invalid routes")
    key = ["Date", "ValueCode", "boundary_quantile", "raw_order_fact_id"]
    if bid.select(*key).n_unique() != bid.height:
        raise ValueError("existing BID q physical key is duplicated")
    return bid.sort(*key)


def _symmetric_route_comparison(
    ask: pl.DataFrame, bid: pl.DataFrame
) -> pl.DataFrame:
    ask_rows = _comparison_metric_rows(
        ask.with_columns(
            pl.when(pl.col("rank_group") == "ASK1_2")
            .then(pl.lit("L1_2"))
            .otherwise(pl.lit("L3_5"))
            .alias("level_band")
        ),
        route=ASK_ROUTE,
        maker_market="future",
        maker_side="ask",
        indexed_truth_source="execution_action_facts",
        sampling_contract="all_sparse_policy_starts_anchor_or_spot",
        legacy_component_present=False,
    )
    bid_rows = _comparison_metric_rows(
        bid.with_columns(pl.col("rank_group").alias("level_band")),
        route=BID_ROUTE,
        maker_market="spot",
        maker_side="bid",
        indexed_truth_source="existing_bid_bundle_alias_projection",
        sampling_contract="direct_spot_snapshot_starts_only",
        legacy_component_present=True,
    )
    result = pl.concat([ask_rows, bid_rows], how="vertical_relaxed").sort(
        "boundary_quantile", "level_band", "route"
    )
    if set(result.get_column("route")) != {ASK_ROUTE, BID_ROUTE}:
        raise ValueError("symmetric comparison route coverage mismatch")
    return result


def _comparison_metric_rows(
    frame: pl.DataFrame,
    *,
    route: str,
    maker_market: str,
    maker_side: str,
    indexed_truth_source: str,
    sampling_contract: str,
    legacy_component_present: bool,
) -> pl.DataFrame:
    slip = pl.col("entry_hedge_signed_total_slippage_bp").filter(
        pl.col("entry_hedge_executable")
    )
    grouped = frame.group_by("boundary_quantile", "level_band").agg(
        pl.len().cast(pl.Int64).alias("q_physical_orders"),
        pl.col("full_fill").sum().cast(pl.Int64).alias("full_fill_count"),
        pl.col("entry_hedge_executable")
        .sum()
        .cast(pl.Int64)
        .alias("hedge_executable_count"),
        slip.count().cast(pl.Int64).alias("hedge_slippage_n"),
        slip.mean().round(12).alias("hedge_slippage_bp_mean"),
        slip.median().round(12).alias("hedge_slippage_bp_p50"),
        slip.quantile(0.9, interpolation="nearest")
        .round(12)
        .alias("hedge_slippage_bp_p90"),
    ).with_columns(
        (100.0 * pl.col("full_fill_count") / pl.col("q_physical_orders"))
        .round(12)
        .alias("full_fill_pct"),
        pl.when(pl.col("full_fill_count") > 0)
        .then(
            100.0
            * pl.col("hedge_executable_count")
            / pl.col("full_fill_count")
        )
        .otherwise(None)
        .round(12)
        .alias("hedge_executable_given_full_pct"),
        pl.lit(route).alias("route"),
        pl.lit(maker_market).alias("maker_market"),
        pl.lit(maker_side).alias("maker_side"),
        pl.lit(indexed_truth_source).alias("indexed_truth_source"),
        pl.lit(sampling_contract).alias("sampling_contract"),
        pl.lit(legacy_component_present).alias("legacy_component_present"),
        pl.lit(False).alias("routes_pooled"),
        pl.lit(False).alias("sampling_contracts_exchangeable"),
        pl.lit("descriptive_only_no_cross_route_causal_delta")
        .alias("comparison_status"),
    )
    prefix = [
        "route",
        "maker_market",
        "maker_side",
        "indexed_truth_source",
        "sampling_contract",
        "boundary_quantile",
        "level_band",
    ]
    return grouped.select(
        *prefix, *(column for column in grouped.columns if column not in prefix)
    )


def _audit(
    manifest: pl.DataFrame,
    aliases: pl.DataFrame,
    q_physical: pl.DataFrame,
    raw_inventory: pl.DataFrame,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for date in [*sorted(set(q_physical.get_column("Date"))), "__all__"]:
        action_slice = aliases if date == "__all__" else aliases.filter(
            pl.col("Date") == date
        )
        q_slice = q_physical if date == "__all__" else q_physical.filter(
            pl.col("Date") == date
        )
        raw_slice = raw_inventory if date == "__all__" else raw_inventory.filter(
            pl.col("Date") == date
        )
        manifest_slice = manifest if date == "__all__" else manifest.filter(
            pl.col("Date") == date
        )
        rows.append(
            {
                "Date": date,
                "product_days": manifest_slice.height,
                "selected_policy_aliases": action_slice.height,
                "q_physical_orders": q_slice.height,
                "collapsed_duplicate_aliases": (
                    action_slice.height - q_slice.height
                ),
                "unique_physical_raw_orders": raw_slice.height,
                "raw_orders_reused_across_q": raw_slice.filter(
                    pl.col("q_membership_count") > 1
                ).height,
                "ask1_2_q_orders": q_slice.filter(
                    pl.col("rank_group") == "ASK1_2"
                ).height,
                "ask3_5_q_orders": q_slice.filter(
                    pl.col("rank_group") == "ASK3_5"
                ).height,
                "full_fill_count": q_slice.filter(pl.col("full_fill")).height,
                "partial_fill_count": q_slice.filter(
                    pl.col("partial_fill")
                ).height,
                "no_fill_count": q_slice.filter(~pl.col("any_fill")).height,
                "anchor_start_count": q_slice.filter(
                    pl.col("submit_event_source") == "anchor"
                ).height,
                "future_start_count": q_slice.filter(
                    pl.col("submit_event_source") == "future"
                ).height,
                "spot_start_count": q_slice.filter(
                    pl.col("submit_event_source") == "spot"
                ).height,
                "unobserved_raw_hedge_payload_excluded": q_slice.filter(
                    pl.col("unobserved_raw_hedge_payload_excluded")
                ).height,
                "legacy_makerfill_files_read": 0,
                "indexed_outcome_exact": True,
                "physical_q_dedupe": True,
                "routes_pooled": False,
                "pathwise_ev_ready": False,
                "joint_volume_allocated": False,
            }
        )
    return pl.from_dicts(rows, infer_schema_length=None).sort("Date")


def _build_source_inventory(
    config: FutureAskRankDiagnosticConfig,
) -> pl.DataFrame:
    config.validate()
    manifest = _load_selected_manifest(config)
    root_manifest_path = (
        Path(config.execution_root) / "execution_partition_manifest.parquet"
    )
    records = [
        _source_record("execution_root_manifest", "__all__", None, root_manifest_path)
    ]
    for row in manifest.iter_rows(named=True):
        date = str(row["Date"])
        value_code = str(row["ValueCode"])
        partition = _resolved_partition(config, row)
        action_path = partition / "execution_action_facts.parquet"
        marker_path = partition / "complete.json"
        _validate_execution_partition(
            action_path,
            marker_path,
            row,
            date=date,
            value_code=value_code,
        )
        records.append(
            _source_record(
                "execution_action_facts", date, value_code, action_path
            )
        )
        records.append(
            _source_record(
                "execution_partition_marker", date, value_code, marker_path
            )
        )
    bid_root = Path(config.bid_study_root)
    _load_bid_projection(bid_root, config.dates)
    records.append(
        _source_record(
            "existing_bid_alias_projection",
            "__all__",
            None,
            bid_root / "alias_comparison.parquet",
        )
    )
    records.append(
        _source_record(
            "existing_bid_complete_marker",
            "__all__",
            None,
            bid_root / "complete.json",
        )
    )
    return pl.from_dicts(
        records,
        schema={
            "source_kind": pl.String,
            "Date": pl.String,
            "ValueCode": pl.String,
            "path": pl.String,
            "bytes": pl.Int64,
            "sha256": pl.String,
            "rows": pl.Int64,
            "columns": pl.Int64,
            "schema_sha256": pl.String,
        },
        infer_schema_length=None,
    ).sort("source_kind", "Date", "ValueCode", "path")


def _source_record(
    source_kind: str,
    date: str,
    value_code: str | None,
    path: Path,
) -> dict[str, object]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        schema = pl.read_parquet_schema(path)
        rows: int | None = _parquet_rows(path)
        columns: int | None = len(schema)
        schema_hash: str | None = _canonical_sha256(_schema_payload(schema))
    else:
        rows = None
        columns = None
        schema_hash = None
    return {
        "source_kind": source_kind,
        "Date": date,
        "ValueCode": value_code,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
        "rows": rows,
        "columns": columns,
        "schema_sha256": schema_hash,
    }


def _implementation_identity() -> dict[str, dict[str, object]]:
    module = Path(__file__).resolve()
    paths = {
        "future_ask_diagnostic": module,
        "future_ask_cli": module.with_name("future_ask_rank_diagnostic_cli.py"),
        "execution_runner": module.with_name("execution_runner.py"),
        "execution_study": module.with_name("study.py"),
        "indexed_replay": module.with_name("indexed_replay.py"),
        "hedge_study": module.with_name("hedge_study.py"),
    }
    identity: dict[str, dict[str, object]] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        identity[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
    return identity


def _safety_contract() -> dict[str, object]:
    return {
        "analysis_only": True,
        "ask_fill_truth": "indexed_execution_action_facts",
        "ask_outcome_exact": True,
        "ask_fill_cursor_exact": True,
        "ask_legacy_makerfill_used": False,
        "ask_legacy_makerfill_imputed": False,
        "future_maker_partial_structurally_zero_at_quantity_one": True,
        "hedge_delay_ns": HEDGE_DELAY_NS,
        "opposite_hedge_market": "spot",
        "physical_dedupe_key": list(_PHYSICAL_KEY),
        "q_outcomes_may_vary_for_same_raw_order": True,
        "unobserved_raw_hedge_payload_excluded_from_metrics": True,
        "routes_pooled": False,
        "bid_comparison_descriptive_only": True,
        "sampling_contracts_exchangeable": False,
        "pathwise_ev_ready": False,
        "joint_volume_allocated": False,
        "cost_profile_complete": False,
    }


def _verify_persisted_invariants(output_root: Path) -> None:
    q = pl.read_parquet(output_root / "q_physical_orders.parquet")
    raw = pl.read_parquet(output_root / "physical_raw_inventory.parquet")
    summary = pl.read_parquet(output_root / "rank_summary.parquet")
    comparison = pl.read_parquet(
        output_root / "symmetric_route_comparison.parquet"
    )
    if q.select(*_PHYSICAL_KEY).n_unique() != q.height:
        raise ValueError("persisted q physical key is duplicated")
    if (
        raw.select("Date", "ValueCode", "raw_order_fact_id").n_unique()
        != raw.height
    ):
        raise ValueError("persisted physical raw inventory key is duplicated")
    all_groups = summary.filter(
        (pl.col("date_scope") == "all_dates")
        & (pl.col("rank_scope") == "rank_group")
    )
    if int(all_groups.get_column("n").sum()) != q.height:
        raise ValueError("persisted ASK summary denominator does not reconcile")
    if all_groups.filter(
        pl.col("full_fill_count")
        + pl.col("partial_fill_count")
        + pl.col("no_fill_count")
        != pl.col("n")
    ).height:
        raise ValueError("persisted ASK outcome cells do not partition n")
    if (
        set(comparison.get_column("route")) != {ASK_ROUTE, BID_ROUTE}
        or comparison.filter(pl.col("routes_pooled") != False).height  # noqa: E712
    ):
        raise ValueError("persisted comparison mixed route contract")


def _artifact_record(path: Path, frame: pl.DataFrame) -> dict[str, object]:
    return {
        "rows": frame.height,
        "columns": frame.width,
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
        "schema": _schema_payload(frame.schema),
    }


def _schema_payload(schema: pl.Schema | Mapping[str, pl.DataType]) -> list[dict[str, str]]:
    return [{"name": name, "dtype": str(dtype)} for name, dtype in schema.items()]


def _assert_frame_exact(
    actual: pl.DataFrame, expected: pl.DataFrame, name: str
) -> None:
    if actual.schema != expected.schema or not actual.equals(expected):
        raise ValueError(f"future ASK semantic recomputation mismatch: {name}")


def _parquet_rows(path: Path) -> int:
    return int(
        pl.scan_parquet(path)
        .select(pl.len().cast(pl.Int64).alias("rows"))
        .collect(engine="streaming")
        .item()
    )


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "ASK_RANKS",
    "DEFAULT_PRODUCT_DAY_COUNTS",
    "DEFAULT_SAMPLE_DATES",
    "FUTURE_ASK_DIAGNOSTIC_VERSION",
    "FutureAskRankDiagnosticConfig",
    "FutureAskRankDiagnosticResult",
    "publish_future_ask_rank_diagnostic",
    "run_future_ask_rank_diagnostic",
    "verify_future_ask_rank_diagnostic",
]
