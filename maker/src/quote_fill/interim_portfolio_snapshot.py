"""Build a frozen, analysis-only portfolio snapshot from a running replay.

This module never writes below the formal entry or same-day exit roots.  The
caller supplies an exact completed-marker count and the digest of that marker
prefix, which makes a snapshot reproducible even while the formal replay keeps
publishing later partitions.

The output deliberately stops short of an equity curve.  It records physical
filled-entry volume, nominal and strict EOD exposure diagnostics, and cashflow
from *completed same-day cycles only*.  Cross-session terminal cashflow and a
complete cost profile are prerequisites for strategy return, drawdown, win
rate, or Sharpe statistics and are therefore not emitted here.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import polars as pl

from .filled_entry_report import (
    _established_entry_expr,
    _normalise_actions,
    _normalise_positions,
    _validate_complete_policy_population,
)
from .hedge_study import SPOT_LOT_SHARES


SNAPSHOT_VERSION = "exit_maker_interim_portfolio_snapshot_v2"
DAILY_ARTIFACT = "daily_metrics.parquet"
DAILY_CSV_ARTIFACT = "daily_metrics.csv"
SUMMARY_ARTIFACT = "policy_summary.csv"
ENTRY_ROUTE_DAILY_ARTIFACT = "entry_route_daily_metrics.parquet"
ENTRY_ROUTE_DAILY_CSV_ARTIFACT = "entry_route_daily_metrics.csv"
ENTRY_ROUTE_SUMMARY_ARTIFACT = "entry_route_summary.csv"
INVENTORY_ARTIFACT = "input_marker_inventory.csv"
CHART_ARTIFACT = "portfolio_diagnostics.png"

ENTRY_ACTION_ARTIFACT = "execution_action_facts.parquet"
POSITION_ARTIFACT = "exit_maker_position_policy_facts.parquet"
ENTRY_MANIFEST = "execution_partition_manifest.parquet"

CELL_KEY = ["boundary_quantile", "exit_rule_id", "exit_route"]
ENTRY_ROUTE_CELL_KEY = ["entry_route", *CELL_KEY]
ENTRY_ROUTES = ("future_ask_spot_taker", "spot_bid_future_taker")
PNL_SCOPE = "completed_same_day_only_unresolved_excluded"
NOTIONAL_SEMANTICS = "one_way_spot_leg_entry_price_notional_twd"
EOD_NOTIONAL_SCOPE = "positions_classified_open_at_eod"
COST_SCOPE = "completed_same_day_19bp_sensitivity_only"
FIXED_COST_SENSITIVITY_BP = 19.0

# These declarations are deliberately exact and fail closed in verify-only mode.
# A bundle that changes even one of them is not the artifact published here.
SAFETY_SEMANTICS: Mapping[str, object] = {
    "analysis_only": True,
    "strategy_defensible": False,
    "pathwise_ev_ready": False,
    "cross_session_terminal_complete": False,
    "full_cost_profile_complete": False,
    "joint_volume_allocated": False,
    "alternative_policy_rows_additive": False,
    "entry_routes_pooled_descriptive_only": True,
    "unresolved_cashflow_imputed": False,
    "unresolved_paths_excluded_from_pnl_curve": True,
    "completed_only_curve_is_equity_curve": False,
    "strategy_win_rate_mdd_sharpe_suppressed": True,
    "cost_sensitivity_analysis_only": True,
    "formal_roots_mutated": False,
    "pnl_scope": PNL_SCOPE,
    "cost_scope": COST_SCOPE,
    "notional_semantics": NOTIONAL_SEMANTICS,
    "eod_notional_scope": EOD_NOTIONAL_SCOPE,
    "notional_is_eod_mark": False,
    "notional_is_two_leg_gross_exposure": False,
    "notional_is_futures_margin": False,
    "notional_is_capital_requirement": False,
}

# Artifact schemas are part of the frozen meaning of this bundle.  They are
# intentionally independent of both the files being verified and their marker
# metadata, so a coordinated file+marker dtype rewrite cannot verify.
_CANONICAL_DAILY_PARQUET_SCHEMA = (
    ("Date", "String"),
    ("boundary_quantile", "Int64"),
    ("exit_rule_id", "String"),
    ("exit_route", "String"),
    ("entry_positions", "UInt32"),
    ("entry_spot_shares", "Int64"),
    ("entry_future_contracts", "Int64"),
    ("cumulative_new_entry_one_way_spot_leg_entry_price_notional_twd", "Float64"),
    ("same_day_completed", "UInt32"),
    ("nominal_eod_carry_positions", "UInt32"),
    ("nominal_unknown_positions", "UInt32"),
    ("strict_proven_flat", "UInt32"),
    ("strict_carry_positions", "UInt32"),
    ("strict_unresolved_positions", "UInt32"),
    ("cancel_race_unknown", "UInt32"),
    ("fill_unknown_at_eod", "UInt32"),
    ("partial_fill_carry_at_eod", "UInt32"),
    ("hedge_incomplete_residual", "UInt32"),
    ("nominal_eod_spot_shares", "Int64"),
    ("nominal_eod_future_contracts", "Int64"),
    ("nominal_eod_open_one_way_spot_leg_entry_price_notional_twd", "Float64"),
    ("strict_unresolved_spot_shares", "Int64"),
    ("strict_unresolved_future_contracts", "Int64"),
    ("strict_unresolved_eod_one_way_spot_leg_entry_price_notional_twd", "Float64"),
    ("completed_gross_twd", "Float64"),
    ("completed_after19_twd", "Float64"),
    ("completed_one_way_spot_leg_entry_price_notional_twd", "Float64"),
    ("completed_gross_weighted_bp", "Float64"),
    ("completed_after19_weighted_bp", "Float64"),
    ("analysis_only", "Boolean"),
    ("strategy_defensible", "Boolean"),
    ("alternative_policy_rows_additive", "Boolean"),
    ("entry_routes_pooled_descriptive_only", "Boolean"),
    ("entry_route_specific_descriptive_only", "Boolean"),
    ("pnl_scope", "String"),
    ("cost_scope", "String"),
    ("notional_semantics", "String"),
    ("eod_notional_scope", "String"),
    ("notional_is_eod_mark", "Boolean"),
    ("notional_is_two_leg_gross_exposure", "Boolean"),
    ("notional_is_futures_margin", "Boolean"),
    ("notional_is_capital_requirement", "Boolean"),
)
_CANONICAL_ROUTE_DAILY_PARQUET_SCHEMA = (
    _CANONICAL_DAILY_PARQUET_SCHEMA[0],
    ("entry_route", "String"),
    *_CANONICAL_DAILY_PARQUET_SCHEMA[1:],
)


def _canonical_csv_schema(
    parquet_schema: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            name,
            "Int64"
            if dtype == "UInt32" or (name == "Date" and dtype == "String")
            else dtype,
        )
        for name, dtype in parquet_schema
    )


_CANONICAL_SUMMARY_CSV_SCHEMA = (
    ("boundary_quantile", "Int64"),
    ("exit_rule_id", "String"),
    ("exit_route", "String"),
    ("days", "Int64"),
    ("entries_total", "Int64"),
    ("entries_daily_mean", "Float64"),
    ("entries_daily_p90", "Float64"),
    ("entries_daily_max", "Int64"),
    ("entry_spot_shares_total", "Int64"),
    ("entry_future_contracts_total", "Int64"),
    ("new_entry_one_way_spot_leg_entry_price_notional_daily_mean_twd", "Float64"),
    ("new_entry_one_way_spot_leg_entry_price_notional_daily_p90_twd", "Float64"),
    ("new_entry_one_way_spot_leg_entry_price_notional_daily_max_twd", "Float64"),
    ("same_day_completed_total", "Int64"),
    ("nominal_carry_total", "Int64"),
    ("nominal_carry_daily_p50", "Float64"),
    ("nominal_carry_daily_p90", "Float64"),
    ("nominal_carry_daily_max", "Int64"),
    ("nominal_unknown_total", "Int64"),
    ("carry_spot_shares_daily_mean", "Float64"),
    ("carry_future_contracts_daily_mean", "Float64"),
    ("nominal_eod_open_one_way_spot_leg_entry_price_notional_daily_mean_twd", "Float64"),
    ("nominal_eod_open_one_way_spot_leg_entry_price_notional_daily_p90_twd", "Float64"),
    ("nominal_eod_open_one_way_spot_leg_entry_price_notional_daily_max_twd", "Float64"),
    ("strict_unresolved_daily_p50", "Float64"),
    ("strict_unresolved_daily_p90", "Float64"),
    ("strict_unresolved_daily_max", "Int64"),
    ("cancel_race_unknown_total", "Int64"),
    ("fill_unknown_total", "Int64"),
    ("partial_total", "Int64"),
    ("residual_total", "Int64"),
    ("completed_gross_total_twd", "Float64"),
    ("completed_after19_total_twd", "Float64"),
    ("completed_one_way_spot_leg_entry_price_notional_total_twd", "Float64"),
    ("nominal_completion_rate", "Float64"),
    ("completed_gross_weighted_bp", "Float64"),
    ("completed_after19_weighted_bp", "Float64"),
    ("strategy_daily_win_rate", "String"),
    ("strategy_mdd_twd", "String"),
    ("strategy_annualized_sharpe", "String"),
    ("strategy_metric_status", "String"),
    ("analysis_only", "Boolean"),
    ("strategy_defensible", "Boolean"),
    ("alternative_policy_rows_additive", "Boolean"),
    ("entry_routes_pooled_descriptive_only", "Boolean"),
    ("entry_route_specific_descriptive_only", "Boolean"),
    ("pnl_scope", "String"),
    ("cost_scope", "String"),
    ("notional_semantics", "String"),
    ("eod_notional_scope", "String"),
    ("notional_is_eod_mark", "Boolean"),
    ("notional_is_two_leg_gross_exposure", "Boolean"),
    ("notional_is_futures_margin", "Boolean"),
    ("notional_is_capital_requirement", "Boolean"),
)
_CANONICAL_ROUTE_SUMMARY_CSV_SCHEMA = (
    ("entry_route", "String"),
    *_CANONICAL_SUMMARY_CSV_SCHEMA,
)
_CANONICAL_INVENTORY_CSV_SCHEMA = (
    ("ordinal", "Int64"),
    ("Date", "Int64"),
    ("ValueCode", "Int64"),
    ("marker_path", "String"),
    ("marker_sha256", "String"),
    ("marker_bytes", "Int64"),
    ("runner_version", "String"),
    ("runner_config_sha256", "String"),
    ("partition_config_sha256", "String"),
    ("entry_action_sha256", "String"),
    ("position_policy_sha256", "String"),
    ("position_policy_bytes", "Int64"),
    ("position_policy_rows", "Int64"),
    ("position_policy_columns", "Int64"),
)
_CANONICAL_ARTIFACT_SCHEMAS: Mapping[
    str,
    tuple[tuple[str, str], ...],
] = {
    DAILY_ARTIFACT: _CANONICAL_DAILY_PARQUET_SCHEMA,
    DAILY_CSV_ARTIFACT: _canonical_csv_schema(_CANONICAL_DAILY_PARQUET_SCHEMA),
    SUMMARY_ARTIFACT: _CANONICAL_SUMMARY_CSV_SCHEMA,
    ENTRY_ROUTE_DAILY_ARTIFACT: _CANONICAL_ROUTE_DAILY_PARQUET_SCHEMA,
    ENTRY_ROUTE_DAILY_CSV_ARTIFACT: _canonical_csv_schema(
        _CANONICAL_ROUTE_DAILY_PARQUET_SCHEMA
    ),
    ENTRY_ROUTE_SUMMARY_ARTIFACT: _CANONICAL_ROUTE_SUMMARY_CSV_SCHEMA,
    INVENTORY_ARTIFACT: _CANONICAL_INVENTORY_CSV_SCHEMA,
}
_CANONICAL_CHART_METADATA = {
    "kind": "png",
    "channels": 4,
}
PHYSICAL_CELL_KEY = [
    "Date",
    "ValueCode",
    "QuoteCode",
    "entry_route",
    "boundary_quantile",
    "entry_raw_order_fact_id",
    "position_established_ns",
    "exit_rule_id",
    "exit_route",
]
NOMINAL_CARRY_BRANCHES = (
    "carry_at_eod_cancel_unconfirmed",
    "carry_at_eod_no_admission",
    "no_fill_before_cancel_request",
)
STRICT_CARRY_BRANCHES = (
    "carry_at_eod_cancel_unconfirmed",
    "carry_at_eod_no_admission",
)

ACTION_COLUMNS = (
    "Date",
    "ValueCode",
    "QuoteCode",
    "route",
    "boundary_quantile",
    "raw_order_fact_id",
    "policy_generation_id",
    "full_fill",
    "partial_fill",
    "any_fill",
    "cancel_required",
    "entry_hedge_status",
    "full_fill_recv_time_ns",
    "entry_hedge_decision_time_ns",
    "entry_hedge_label_observed",
    "entry_hedge_executable",
    "entry_spot_price",
    "entry_future_price",
    "entry_hedge_contract_size_shares",
    "known_filled_quantity",
    "entry_hedge_executed_quantity",
)
POSITION_COLUMNS = (
    "Date",
    "ValueCode",
    "QuoteCode",
    "entry_route",
    "entry_policy_generation_id",
    "entry_raw_order_fact_id",
    "exit_rule_id",
    "exit_route",
    "exit_policy_trial_id",
    "exit_threshold_basis_bp",
    "exit_rule_source_asof_date",
    "position_status",
    "position_established_ns",
    "nominal_instant_cancel_v0_branch",
    "branch_status",
    "terminal_outcome",
    "exit_decision_time_ns",
    "gross_cycle_pnl_twd",
)


@dataclass(frozen=True)
class SnapshotConfig:
    marker_count: int
    snapshot_time: str
    expected_marker_inventory_sha256: str
    expected_last_key: str
    cost_sensitivity_bp: float = FIXED_COST_SENSITIVITY_BP

    def validate(self) -> None:
        if self.marker_count <= 0:
            raise ValueError("marker_count must be positive")
        if len(self.expected_marker_inventory_sha256) != 64:
            raise ValueError("expected marker inventory digest must be SHA-256")
        if self.expected_last_key.count("/") != 1:
            raise ValueError("expected_last_key must be Date/ValueCode")
        if not self.snapshot_time:
            raise ValueError("snapshot_time is required")
        if self.cost_sensitivity_bp != FIXED_COST_SENSITIVITY_BP:
            raise ValueError("this snapshot contract requires exactly 19 bp")


@dataclass(frozen=True)
class FrozenInputs:
    inventory: pl.DataFrame
    action_paths: tuple[Path, ...]
    position_paths: tuple[Path, ...]
    expected_product_days: tuple[tuple[str, str], ...]
    snapshot_product_days: tuple[tuple[str, str], ...]
    marker_inventory_sha256: str
    runner_config_sha256: str


@dataclass(frozen=True)
class SnapshotTables:
    daily: pl.DataFrame
    summary: pl.DataFrame
    entry_route_daily: pl.DataFrame
    entry_route_summary: pl.DataFrame
    full_dates: tuple[str, ...]
    excluded_partial_dates: tuple[str, ...]
    entry_action_aliases: int
    established_entry_aliases: int
    physical_entry_dependencies: int
    physical_policy_cells: int
    physical_policy_aliases: int
    complete_date_physical_policy_cells: int
    population_audit: Mapping[str, object]


def build_interim_portfolio_snapshot(
    *,
    exit_root: Path,
    entry_root: Path,
    output_root: Path,
    config: SnapshotConfig,
) -> Mapping[str, object]:
    """Validate a marker prefix and atomically publish its diagnostic bundle."""

    config.validate()
    exit_root = Path(exit_root).resolve()
    entry_root = Path(entry_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    if output_root == exit_root or exit_root in output_root.parents:
        raise ValueError("analysis output must be outside the formal exit root")
    if output_root == entry_root or entry_root in output_root.parents:
        raise ValueError("analysis output must be outside the formal entry root")

    frozen = _freeze_inputs(exit_root, entry_root, config)
    tables = _build_tables(frozen, config)
    return _publish(
        frozen=frozen,
        tables=tables,
        exit_root=exit_root,
        entry_root=entry_root,
        output_root=output_root,
        config=config,
    )


def _freeze_inputs(
    exit_root: Path,
    entry_root: Path,
    config: SnapshotConfig,
) -> FrozenInputs:
    manifest_path = entry_root / ENTRY_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = pl.read_parquet(manifest_path).sort(["Date", "ValueCode"])
    expected = tuple(
        tuple(map(str, row))
        for row in manifest.select("Date", "ValueCode").iter_rows()
    )
    if config.marker_count > len(expected):
        raise ValueError("snapshot marker count exceeds entry manifest")

    all_markers = tuple(sorted(exit_root.glob("Date=*/ValueCode=*/complete.json")))
    all_keys = tuple(_marker_path_key(path) for path in all_markers)
    if all_keys != expected[: len(all_keys)]:
        raise ValueError("formal exit markers are not a strict entry-manifest prefix")
    if len(all_markers) < config.marker_count:
        raise ValueError("formal replay has not reached the requested snapshot prefix")

    snapshot_keys = expected[: config.marker_count]
    if "/".join(snapshot_keys[-1]) != config.expected_last_key:
        raise ValueError("requested marker prefix ends at an unexpected product-day")

    inventory_digest = hashlib.sha256()
    inventory_rows: list[dict[str, object]] = []
    action_paths: list[Path] = []
    position_paths: list[Path] = []
    runner_config_sha256: str | None = None
    for ordinal, (date, value_code) in enumerate(snapshot_keys, start=1):
        partition = exit_root / f"Date={date}" / f"ValueCode={value_code}"
        marker_path = partition / "complete.json"
        marker_bytes = marker_path.read_bytes()
        marker_sha256 = hashlib.sha256(marker_bytes).hexdigest()
        inventory_digest.update(str(marker_path.relative_to(exit_root)).encode())
        inventory_digest.update(b"\0")
        inventory_digest.update(bytes.fromhex(marker_sha256))
        payload = json.loads(marker_bytes)
        if (
            payload.get("complete") is not True
            or str(payload.get("Date")) != date
            or str(payload.get("ValueCode")) != value_code
        ):
            raise ValueError(f"invalid exit marker identity: {marker_path}")
        current_runner_sha = str(payload.get("runner_config_sha256"))
        if runner_config_sha256 is None:
            runner_config_sha256 = current_runner_sha
        if current_runner_sha != runner_config_sha256:
            raise ValueError("snapshot mixes exit runner configurations")

        artifacts = payload.get("artifacts")
        marker_config = payload.get("config")
        if not isinstance(artifacts, dict) or not isinstance(marker_config, dict):
            raise ValueError(f"invalid exit marker payload: {marker_path}")
        position_meta = artifacts.get(POSITION_ARTIFACT)
        source = marker_config.get("source")
        if not isinstance(position_meta, dict) or not isinstance(source, dict):
            raise ValueError(f"exit marker lacks snapshot inputs: {marker_path}")
        action_source = source.get("action_source")
        if not isinstance(action_source, dict):
            raise ValueError(f"exit marker lacks action source: {marker_path}")

        position_path = partition / POSITION_ARTIFACT
        action_path = (
            entry_root
            / f"Date={date}"
            / f"ValueCode={value_code}"
            / ENTRY_ACTION_ARTIFACT
        )
        if position_path.stat().st_size != int(position_meta["bytes"]):
            raise ValueError(f"position byte size mismatch: {position_path}")
        if _file_sha256(position_path) != str(position_meta["sha256"]):
            raise ValueError(f"position hash mismatch: {position_path}")
        if _file_sha256(action_path) != str(action_source.get("sha256")):
            raise ValueError(f"entry action source hash mismatch: {action_path}")
        actual_position_rows = int(
            pl.scan_parquet(position_path)
            .select(pl.len())
            .collect(engine="streaming")
            .item()
        )
        actual_position_columns = len(pl.read_parquet_schema(position_path))
        if (
            actual_position_rows != int(position_meta["rows"])
            or actual_position_columns != int(position_meta["columns"])
        ):
            raise ValueError(f"position metadata mismatch: {position_path}")

        inventory_rows.append(
            {
                "ordinal": ordinal,
                "Date": date,
                "ValueCode": value_code,
                "marker_path": str(marker_path),
                "marker_sha256": marker_sha256,
                "marker_bytes": len(marker_bytes),
                "runner_version": str(payload.get("runner_version")),
                "runner_config_sha256": current_runner_sha,
                "partition_config_sha256": str(payload.get("config_sha256")),
                "entry_action_sha256": str(action_source.get("sha256")),
                "position_policy_sha256": str(position_meta["sha256"]),
                "position_policy_bytes": int(position_meta["bytes"]),
                "position_policy_rows": int(position_meta["rows"]),
                "position_policy_columns": int(position_meta["columns"]),
            }
        )
        action_paths.append(action_path)
        position_paths.append(position_path)

    observed_digest = inventory_digest.hexdigest()
    if observed_digest != config.expected_marker_inventory_sha256:
        raise ValueError(
            "snapshot marker inventory digest mismatch: "
            f"expected {config.expected_marker_inventory_sha256}, "
            f"observed {observed_digest}"
        )
    assert runner_config_sha256 is not None
    return FrozenInputs(
        inventory=pl.from_dicts(inventory_rows, infer_schema_length=None),
        action_paths=tuple(action_paths),
        position_paths=tuple(position_paths),
        expected_product_days=expected,
        snapshot_product_days=snapshot_keys,
        marker_inventory_sha256=observed_digest,
        runner_config_sha256=runner_config_sha256,
    )


def _build_tables(frozen: FrozenInputs, config: SnapshotConfig) -> SnapshotTables:
    actions = _normalise_actions(
        pl.scan_parquet([str(path) for path in frozen.action_paths])
        .select(ACTION_COLUMNS)
        .collect(engine="streaming")
    )
    positions = _normalise_positions(
        pl.scan_parquet([str(path) for path in frozen.position_paths])
        .select(POSITION_COLUMNS)
        .collect(engine="streaming")
    )
    population = _validate_complete_policy_population(actions, positions)
    established = actions.filter(_established_entry_expr())
    action_meta = established.select(
        pl.col("policy_generation_id").alias("entry_policy_generation_id"),
        "boundary_quantile",
        pl.col("route").alias("_action_entry_route"),
        pl.col("raw_order_fact_id").alias("_action_entry_raw_order_fact_id"),
        pl.col("entry_hedge_decision_time_ns").alias(
            "_action_position_established_ns"
        ),
        (
            pl.when(pl.col("route") == "future_ask_spot_taker")
            .then(pl.col("entry_hedge_executed_quantity"))
            .otherwise(pl.col("known_filled_quantity"))
            * SPOT_LOT_SHARES
        )
        .cast(pl.Int64)
        .alias("spot_shares"),
        pl.when(pl.col("route") == "future_ask_spot_taker")
        .then(pl.col("known_filled_quantity"))
        .otherwise(pl.col("entry_hedge_executed_quantity"))
        .cast(pl.Int64)
        .alias("future_contracts"),
        "entry_spot_price",
        "entry_hedge_contract_size_shares",
    )
    joined = positions.join(
        action_meta,
        on="entry_policy_generation_id",
        how="left",
        validate="m:1",
    )
    if joined["boundary_quantile"].null_count():
        raise ValueError("position facts contain a non-established entry")
    identity_mismatch = joined.filter(
        (pl.col("entry_route") != pl.col("_action_entry_route")).fill_null(True)
        | (
            pl.col("entry_raw_order_fact_id")
            != pl.col("_action_entry_raw_order_fact_id")
        ).fill_null(True)
        | (
            pl.col("position_established_ns")
            != pl.col("_action_position_established_ns")
        ).fill_null(True)
    )
    if identity_mismatch.height:
        raise ValueError("entry and exit physical identities disagree")
    invalid_quantity = joined.filter(
        (pl.col("spot_shares") <= 0)
        | (pl.col("future_contracts") <= 0)
        | (
            pl.col("spot_shares")
            != pl.col("future_contracts")
            * pl.col("entry_hedge_contract_size_shares")
        )
    )
    if invalid_quantity.height:
        raise ValueError("entry spot/futures quantities do not hedge one contract")
    joined = joined.with_columns(
        (pl.col("spot_shares") * pl.col("entry_spot_price")).alias(
            "one_way_spot_leg_entry_price_notional_twd"
        )
    )
    physical = _collapse_physical_policy_cells(joined)

    expected_counts = _counts_by_date(frozen.expected_product_days)
    snapshot_counts = _counts_by_date(frozen.snapshot_product_days)
    full_dates = tuple(
        sorted(
            date
            for date, count in snapshot_counts.items()
            if count == expected_counts[date]
        )
    )
    excluded_partial_dates = tuple(
        sorted(
            date
            for date, count in snapshot_counts.items()
            if count != expected_counts[date]
        )
    )
    if not full_dates:
        raise ValueError("snapshot has no fully completed sessions")
    complete_physical = physical.filter(pl.col("Date").is_in(full_dates))
    daily = _daily_metrics(
        complete_physical,
        full_dates,
        config,
        split_entry_route=False,
    )
    summary = _policy_summary(daily, split_entry_route=False)
    entry_route_daily = _daily_metrics(
        complete_physical,
        full_dates,
        config,
        split_entry_route=True,
    )
    entry_route_summary = _policy_summary(
        entry_route_daily,
        split_entry_route=True,
    )

    dependency_key = [
        "Date",
        "ValueCode",
        "QuoteCode",
        "route",
        "raw_order_fact_id",
        "entry_hedge_decision_time_ns",
    ]
    physical_dependencies = established.select(dependency_key).unique().height
    return SnapshotTables(
        daily=daily,
        summary=summary,
        entry_route_daily=entry_route_daily,
        entry_route_summary=entry_route_summary,
        full_dates=full_dates,
        excluded_partial_dates=excluded_partial_dates,
        entry_action_aliases=actions.height,
        established_entry_aliases=established.height,
        physical_entry_dependencies=physical_dependencies,
        physical_policy_cells=physical.height,
        physical_policy_aliases=int(physical["policy_aliases"].sum()),
        complete_date_physical_policy_cells=complete_physical.height,
        population_audit=population,
    )


def _collapse_physical_policy_cells(joined: pl.DataFrame) -> pl.DataFrame:
    consistency_columns = [
        "nominal_instant_cancel_v0_branch",
        "branch_status",
        "terminal_outcome",
        "exit_decision_time_ns",
        "gross_cycle_pnl_twd",
        "spot_shares",
        "future_contracts",
        "one_way_spot_leg_entry_price_notional_twd",
    ]
    conflicts = joined.group_by(PHYSICAL_CELL_KEY).agg(
        *(pl.col(column).n_unique().alias(column) for column in consistency_columns)
    ).filter(
        pl.any_horizontal(
            *(pl.col(column) > 1 for column in consistency_columns)
        )
    )
    if conflicts.height:
        raise ValueError("aliases disagree within a physical policy cell")
    return joined.group_by(PHYSICAL_CELL_KEY, maintain_order=True).agg(
        pl.len().alias("policy_aliases"),
        *(
            pl.col(column).first().alias(column)
            for column in consistency_columns
        ),
    )


def _daily_metrics(
    physical: pl.DataFrame,
    full_dates: Sequence[str],
    config: SnapshotConfig,
    *,
    split_entry_route: bool,
) -> pl.DataFrame:
    nominal = pl.col("nominal_instant_cancel_v0_branch")
    strict = pl.col("branch_status")
    same_day = nominal == "flat_same_day"
    nominal_carry = nominal.is_in(NOMINAL_CARRY_BRANCHES)
    strict_flat = strict == "flat_same_day"
    strict_carry = strict.is_in(STRICT_CARRY_BRANCHES)
    sensitivity_rate = config.cost_sensitivity_bp / 10_000.0
    grouping_key = ["Date", *(ENTRY_ROUTE_CELL_KEY if split_entry_route else CELL_KEY)]
    daily = physical.group_by(grouping_key).agg(
        pl.len().alias("entry_positions"),
        pl.col("spot_shares").sum().alias("entry_spot_shares"),
        pl.col("future_contracts").sum().alias("entry_future_contracts"),
        pl.col("one_way_spot_leg_entry_price_notional_twd")
        .sum()
        .alias(
            "cumulative_new_entry_one_way_spot_leg_entry_price_notional_twd"
        ),
        same_day.sum().alias("same_day_completed"),
        nominal_carry.sum().alias("nominal_eod_carry_positions"),
        (~(same_day | nominal_carry)).sum().alias("nominal_unknown_positions"),
        strict_flat.sum().alias("strict_proven_flat"),
        strict_carry.sum().alias("strict_carry_positions"),
        (~strict_flat).sum().alias("strict_unresolved_positions"),
        (strict == "cancel_race_unknown").sum().alias("cancel_race_unknown"),
        (strict == "fill_unknown_at_eod").sum().alias("fill_unknown_at_eod"),
        (strict == "partial_fill_carry_at_eod")
        .sum()
        .alias("partial_fill_carry_at_eod"),
        (strict == "hedge_incomplete_residual")
        .sum()
        .alias("hedge_incomplete_residual"),
        pl.col("spot_shares")
        .filter(nominal_carry)
        .sum()
        .alias("nominal_eod_spot_shares"),
        pl.col("future_contracts")
        .filter(nominal_carry)
        .sum()
        .alias("nominal_eod_future_contracts"),
        pl.col("one_way_spot_leg_entry_price_notional_twd")
        .filter(nominal_carry)
        .sum()
        .alias(
            "nominal_eod_open_one_way_spot_leg_entry_price_notional_twd"
        ),
        pl.col("spot_shares")
        .filter(~strict_flat)
        .sum()
        .alias("strict_unresolved_spot_shares"),
        pl.col("future_contracts")
        .filter(~strict_flat)
        .sum()
        .alias("strict_unresolved_future_contracts"),
        pl.col("one_way_spot_leg_entry_price_notional_twd")
        .filter(~strict_flat)
        .sum()
        .alias(
            "strict_unresolved_eod_one_way_spot_leg_entry_price_notional_twd"
        ),
        pl.col("gross_cycle_pnl_twd")
        .filter(same_day)
        .sum()
        .alias("completed_gross_twd"),
        (
            pl.col("gross_cycle_pnl_twd")
            - pl.col("one_way_spot_leg_entry_price_notional_twd")
            * sensitivity_rate
        )
        .filter(same_day)
        .sum()
        .alias("completed_after19_twd"),
        pl.col("one_way_spot_leg_entry_price_notional_twd")
        .filter(same_day)
        .sum()
        .alias("completed_one_way_spot_leg_entry_price_notional_twd"),
    )
    scenarios = _scenario_frame()
    if split_entry_route:
        scenarios = pl.DataFrame({"entry_route": list(ENTRY_ROUTES)}).join(
            scenarios,
            how="cross",
        )
    grid = pl.DataFrame({"Date": list(full_dates)}).join(scenarios, how="cross")
    result = grid.join(
        daily,
        on=grouping_key,
        how="left",
        validate="1:1",
    )
    numeric = [
        column
        for column, dtype in result.schema.items()
        if column not in {"Date", *CELL_KEY} and dtype.is_numeric()
    ]
    result = result.with_columns(
        *(pl.col(column).fill_null(0) for column in numeric)
    ).with_columns(
        pl.when(
            pl.col("completed_one_way_spot_leg_entry_price_notional_twd") > 0
        )
        .then(
            pl.col("completed_gross_twd")
            / pl.col("completed_one_way_spot_leg_entry_price_notional_twd")
            * 10_000.0
        )
        .otherwise(None)
        .alias("completed_gross_weighted_bp"),
        pl.when(
            pl.col("completed_one_way_spot_leg_entry_price_notional_twd") > 0
        )
        .then(
            pl.col("completed_after19_twd")
            / pl.col("completed_one_way_spot_leg_entry_price_notional_twd")
            * 10_000.0
        )
        .otherwise(None)
        .alias("completed_after19_weighted_bp"),
        pl.lit(True).alias("analysis_only"),
        pl.lit(False).alias("strategy_defensible"),
        pl.lit(False).alias("alternative_policy_rows_additive"),
        pl.lit(not split_entry_route).alias(
            "entry_routes_pooled_descriptive_only"
        ),
        pl.lit(split_entry_route).alias(
            "entry_route_specific_descriptive_only"
        ),
        pl.lit(PNL_SCOPE).alias("pnl_scope"),
        pl.lit(COST_SCOPE).alias("cost_scope"),
        pl.lit(NOTIONAL_SEMANTICS).alias("notional_semantics"),
        pl.lit(EOD_NOTIONAL_SCOPE).alias("eod_notional_scope"),
        pl.lit(False).alias("notional_is_eod_mark"),
        pl.lit(False).alias("notional_is_two_leg_gross_exposure"),
        pl.lit(False).alias("notional_is_futures_margin"),
        pl.lit(False).alias("notional_is_capital_requirement"),
    )
    sort_key = ENTRY_ROUTE_CELL_KEY if split_entry_route else CELL_KEY
    return result.sort([*sort_key, "Date"])


def _policy_summary(
    daily: pl.DataFrame,
    *,
    split_entry_route: bool,
) -> pl.DataFrame:
    grouping_key = ENTRY_ROUTE_CELL_KEY if split_entry_route else CELL_KEY
    summary = daily.group_by(grouping_key).agg(
        pl.len().alias("days"),
        pl.col("entry_positions").sum().alias("entries_total"),
        pl.col("entry_positions").mean().alias("entries_daily_mean"),
        pl.col("entry_positions").quantile(0.90).alias("entries_daily_p90"),
        pl.col("entry_positions").max().alias("entries_daily_max"),
        pl.col("entry_spot_shares").sum().alias("entry_spot_shares_total"),
        pl.col("entry_future_contracts")
        .sum()
        .alias("entry_future_contracts_total"),
        pl.col("cumulative_new_entry_one_way_spot_leg_entry_price_notional_twd")
        .mean()
        .alias(
            "new_entry_one_way_spot_leg_entry_price_notional_daily_mean_twd"
        ),
        pl.col("cumulative_new_entry_one_way_spot_leg_entry_price_notional_twd")
        .quantile(0.90)
        .alias(
            "new_entry_one_way_spot_leg_entry_price_notional_daily_p90_twd"
        ),
        pl.col("cumulative_new_entry_one_way_spot_leg_entry_price_notional_twd")
        .max()
        .alias(
            "new_entry_one_way_spot_leg_entry_price_notional_daily_max_twd"
        ),
        pl.col("same_day_completed").sum().alias("same_day_completed_total"),
        pl.col("nominal_eod_carry_positions")
        .sum()
        .alias("nominal_carry_total"),
        pl.col("nominal_eod_carry_positions")
        .median()
        .alias("nominal_carry_daily_p50"),
        pl.col("nominal_eod_carry_positions")
        .quantile(0.90)
        .alias("nominal_carry_daily_p90"),
        pl.col("nominal_eod_carry_positions")
        .max()
        .alias("nominal_carry_daily_max"),
        pl.col("nominal_unknown_positions")
        .sum()
        .alias("nominal_unknown_total"),
        pl.col("nominal_eod_spot_shares")
        .mean()
        .alias("carry_spot_shares_daily_mean"),
        pl.col("nominal_eod_future_contracts")
        .mean()
        .alias("carry_future_contracts_daily_mean"),
        pl.col("nominal_eod_open_one_way_spot_leg_entry_price_notional_twd")
        .mean()
        .alias(
            "nominal_eod_open_one_way_spot_leg_entry_price_notional_daily_mean_twd"
        ),
        pl.col("nominal_eod_open_one_way_spot_leg_entry_price_notional_twd")
        .quantile(0.90)
        .alias(
            "nominal_eod_open_one_way_spot_leg_entry_price_notional_daily_p90_twd"
        ),
        pl.col("nominal_eod_open_one_way_spot_leg_entry_price_notional_twd")
        .max()
        .alias(
            "nominal_eod_open_one_way_spot_leg_entry_price_notional_daily_max_twd"
        ),
        pl.col("strict_unresolved_positions")
        .median()
        .alias("strict_unresolved_daily_p50"),
        pl.col("strict_unresolved_positions")
        .quantile(0.90)
        .alias("strict_unresolved_daily_p90"),
        pl.col("strict_unresolved_positions")
        .max()
        .alias("strict_unresolved_daily_max"),
        pl.col("cancel_race_unknown").sum().alias("cancel_race_unknown_total"),
        pl.col("fill_unknown_at_eod").sum().alias("fill_unknown_total"),
        pl.col("partial_fill_carry_at_eod").sum().alias("partial_total"),
        pl.col("hedge_incomplete_residual").sum().alias("residual_total"),
        pl.col("completed_gross_twd")
        .sum()
        .alias("completed_gross_total_twd"),
        pl.col("completed_after19_twd")
        .sum()
        .alias("completed_after19_total_twd"),
        pl.col("completed_one_way_spot_leg_entry_price_notional_twd")
        .sum()
        .alias("completed_one_way_spot_leg_entry_price_notional_total_twd"),
    ).with_columns(
        (
            pl.col("same_day_completed_total") / pl.col("entries_total")
        ).alias("nominal_completion_rate"),
        (
            pl.col("completed_gross_total_twd")
            / pl.col("completed_one_way_spot_leg_entry_price_notional_total_twd")
            * 10_000.0
        ).alias("completed_gross_weighted_bp"),
        (
            pl.col("completed_after19_total_twd")
            / pl.col("completed_one_way_spot_leg_entry_price_notional_total_twd")
            * 10_000.0
        ).alias("completed_after19_weighted_bp"),
        pl.lit(None, dtype=pl.Float64).alias("strategy_daily_win_rate"),
        pl.lit(None, dtype=pl.Float64).alias("strategy_mdd_twd"),
        pl.lit(None, dtype=pl.Float64).alias("strategy_annualized_sharpe"),
        pl.lit("suppressed_cross_terminal_and_full_costs_incomplete").alias(
            "strategy_metric_status"
        ),
        pl.lit(True).alias("analysis_only"),
        pl.lit(False).alias("strategy_defensible"),
        pl.lit(False).alias("alternative_policy_rows_additive"),
        pl.lit(not split_entry_route).alias(
            "entry_routes_pooled_descriptive_only"
        ),
        pl.lit(split_entry_route).alias(
            "entry_route_specific_descriptive_only"
        ),
        pl.lit(PNL_SCOPE).alias("pnl_scope"),
        pl.lit(COST_SCOPE).alias("cost_scope"),
        pl.lit(NOTIONAL_SEMANTICS).alias("notional_semantics"),
        pl.lit(EOD_NOTIONAL_SCOPE).alias("eod_notional_scope"),
        pl.lit(False).alias("notional_is_eod_mark"),
        pl.lit(False).alias("notional_is_two_leg_gross_exposure"),
        pl.lit(False).alias("notional_is_futures_margin"),
        pl.lit(False).alias("notional_is_capital_requirement"),
    )
    return summary.sort(grouping_key)


def _publish(
    *,
    frozen: FrozenInputs,
    tables: SnapshotTables,
    exit_root: Path,
    entry_root: Path,
    output_root: Path,
    config: SnapshotConfig,
) -> Mapping[str, object]:
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    try:
        frozen.inventory.write_csv(stage / INVENTORY_ARTIFACT)
        tables.daily.write_parquet(stage / DAILY_ARTIFACT)
        tables.daily.write_csv(stage / DAILY_CSV_ARTIFACT)
        tables.summary.write_csv(stage / SUMMARY_ARTIFACT)
        tables.entry_route_daily.write_parquet(stage / ENTRY_ROUTE_DAILY_ARTIFACT)
        tables.entry_route_daily.write_csv(stage / ENTRY_ROUTE_DAILY_CSV_ARTIFACT)
        tables.entry_route_summary.write_csv(stage / ENTRY_ROUTE_SUMMARY_ARTIFACT)
        _plot_diagnostics(tables.daily, stage / CHART_ARTIFACT, config)
        _verify_tabular_round_trip(stage, frozen.inventory, tables)

        artifact_names = (
            INVENTORY_ARTIFACT,
            DAILY_ARTIFACT,
            DAILY_CSV_ARTIFACT,
            SUMMARY_ARTIFACT,
            ENTRY_ROUTE_DAILY_ARTIFACT,
            ENTRY_ROUTE_DAILY_CSV_ARTIFACT,
            ENTRY_ROUTE_SUMMARY_ARTIFACT,
            CHART_ARTIFACT,
        )
        artifacts = {
            name: _artifact_metadata(stage / name)
            for name in artifact_names
        }
        _verify_canonical_artifact_contract(stage, artifacts)
        implementation_sha256 = _file_sha256(Path(__file__))
        config_payload = asdict(config) | {
            "snapshot_version": SNAPSHOT_VERSION,
            "implementation_sha256": implementation_sha256,
            "exit_root": str(exit_root.resolve()),
            "entry_root": str(entry_root.resolve()),
            "output_root": str(output_root.resolve()),
        }
        marker: dict[str, object] = {
            "complete": True,
            "snapshot_version": SNAPSHOT_VERSION,
            "config": config_payload,
            "config_sha256": _canonical_sha256(config_payload),
            "snapshot_time": config.snapshot_time,
            "input_marker_count": config.marker_count,
            "input_marker_inventory_sha256": frozen.marker_inventory_sha256,
            "input_last_key": config.expected_last_key,
            "entry_product_day_count": len(frozen.expected_product_days),
            "exit_runner_config_sha256": frozen.runner_config_sha256,
            "full_dates": list(tables.full_dates),
            "full_date_count": len(tables.full_dates),
            "excluded_partial_dates": list(tables.excluded_partial_dates),
            "entry_action_aliases": tables.entry_action_aliases,
            "established_entry_aliases": tables.established_entry_aliases,
            "physical_entry_dependencies": tables.physical_entry_dependencies,
            "physical_policy_cells": tables.physical_policy_cells,
            "physical_policy_aliases": tables.physical_policy_aliases,
            "complete_date_physical_policy_cells": (
                tables.complete_date_physical_policy_cells
            ),
            "population_audit": dict(tables.population_audit),
            "daily_rows": tables.daily.height,
            "policy_summary_rows": tables.summary.height,
            "entry_route_daily_rows": tables.entry_route_daily.height,
            "entry_route_summary_rows": tables.entry_route_summary.height,
            **SAFETY_SEMANTICS,
            "artifacts": artifacts,
        }
        (stage / "complete.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        stage.replace(output_root)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return verify_interim_portfolio_snapshot(output_root)


def verify_interim_portfolio_snapshot(root: Path) -> Mapping[str, object]:
    """Fail-closed verification of a published analysis-only bundle."""

    root = Path(root)
    marker_path = root / "complete.json"
    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    expected_top_level = {
        "complete",
        "snapshot_version",
        "config",
        "config_sha256",
        "snapshot_time",
        "input_marker_count",
        "input_marker_inventory_sha256",
        "input_last_key",
        "entry_product_day_count",
        "exit_runner_config_sha256",
        "full_dates",
        "full_date_count",
        "excluded_partial_dates",
        "entry_action_aliases",
        "established_entry_aliases",
        "physical_entry_dependencies",
        "physical_policy_cells",
        "physical_policy_aliases",
        "complete_date_physical_policy_cells",
        "population_audit",
        "daily_rows",
        "policy_summary_rows",
        "entry_route_daily_rows",
        "entry_route_summary_rows",
        "artifacts",
        *SAFETY_SEMANTICS,
    }
    if set(payload) != expected_top_level:
        raise ValueError("snapshot marker field set mismatch")
    if payload["complete"] is not True:
        raise ValueError("snapshot completion marker is not complete")
    if payload["snapshot_version"] != SNAPSHOT_VERSION:
        raise ValueError("snapshot version mismatch")
    invalid_semantics = {
        key: payload.get(key)
        for key, expected in SAFETY_SEMANTICS.items()
        if payload.get(key) != expected or type(payload.get(key)) is not type(expected)
    }
    if invalid_semantics:
        raise ValueError(f"snapshot safety/semantic flags are invalid: {invalid_semantics}")

    config = payload["config"]
    expected_config_fields = {
        "marker_count",
        "snapshot_time",
        "expected_marker_inventory_sha256",
        "expected_last_key",
        "cost_sensitivity_bp",
        "snapshot_version",
        "implementation_sha256",
        "exit_root",
        "entry_root",
        "output_root",
    }
    if not isinstance(config, dict) or set(config) != expected_config_fields:
        raise ValueError("snapshot config field set mismatch")
    if _canonical_sha256(config) != payload["config_sha256"]:
        raise ValueError("snapshot config hash mismatch")
    snapshot_config = SnapshotConfig(
        marker_count=int(config["marker_count"]),
        snapshot_time=str(config["snapshot_time"]),
        expected_marker_inventory_sha256=str(
            config["expected_marker_inventory_sha256"]
        ),
        expected_last_key=str(config["expected_last_key"]),
        cost_sensitivity_bp=float(config["cost_sensitivity_bp"]),
    )
    snapshot_config.validate()
    if (
        config["snapshot_version"] != SNAPSHOT_VERSION
        or config["implementation_sha256"] != _file_sha256(Path(__file__))
        or Path(str(config["output_root"])).resolve() != root.resolve()
        or config["marker_count"] != payload["input_marker_count"]
        or config["snapshot_time"] != payload["snapshot_time"]
        or config["expected_marker_inventory_sha256"]
        != payload["input_marker_inventory_sha256"]
        or config["expected_last_key"] != payload["input_last_key"]
    ):
        raise ValueError("snapshot config identity mismatch")
    exit_root = Path(str(config["exit_root"])).resolve()
    entry_root = Path(str(config["entry_root"])).resolve()
    if (
        root.resolve() in {exit_root, entry_root}
        or exit_root in root.resolve().parents
        or entry_root in root.resolve().parents
    ):
        raise ValueError("analysis snapshot is inside a formal root")

    artifact_names = {
        INVENTORY_ARTIFACT,
        DAILY_ARTIFACT,
        DAILY_CSV_ARTIFACT,
        SUMMARY_ARTIFACT,
        ENTRY_ROUTE_DAILY_ARTIFACT,
        ENTRY_ROUTE_DAILY_CSV_ARTIFACT,
        ENTRY_ROUTE_SUMMARY_ARTIFACT,
        CHART_ARTIFACT,
    }
    artifacts = payload["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != artifact_names:
        raise ValueError("snapshot artifact manifest field set mismatch")
    if {path.name for path in root.iterdir()} != {*artifact_names, "complete.json"}:
        raise ValueError("snapshot artifact set mismatch")
    for name in sorted(artifact_names):
        path = root / name
        metadata = artifacts[name]
        if not path.is_file() or not isinstance(metadata, dict):
            raise ValueError(f"missing snapshot artifact: {path}")
        if _artifact_metadata(path) != metadata:
            raise ValueError(f"snapshot artifact metadata mismatch: {path}")
    _verify_canonical_artifact_contract(root, artifacts)

    daily = pl.read_parquet(root / DAILY_ARTIFACT)
    daily_csv = pl.read_csv(
        root / DAILY_CSV_ARTIFACT,
        schema_overrides=daily.schema,
    )
    route_daily = pl.read_parquet(root / ENTRY_ROUTE_DAILY_ARTIFACT)
    route_daily_csv = pl.read_csv(
        root / ENTRY_ROUTE_DAILY_CSV_ARTIFACT,
        schema_overrides=route_daily.schema,
    )
    if not daily.equals(daily_csv, null_equal=True):
        raise ValueError("daily Parquet and CSV contents differ")
    if not route_daily.equals(route_daily_csv, null_equal=True):
        raise ValueError("entry-route daily Parquet and CSV contents differ")
    _verify_daily_semantics(daily, payload, split_entry_route=False)
    _verify_daily_semantics(route_daily, payload, split_entry_route=True)
    _verify_route_rollup(daily, route_daily)

    expected_summary = _policy_summary(daily, split_entry_route=False)
    summary = pl.read_csv(
        root / SUMMARY_ARTIFACT,
        schema_overrides=expected_summary.schema,
    )
    if not summary.equals(expected_summary, null_equal=True):
        raise ValueError("policy summary does not reproduce from daily metrics")
    expected_route_summary = _policy_summary(
        route_daily,
        split_entry_route=True,
    )
    route_summary = pl.read_csv(
        root / ENTRY_ROUTE_SUMMARY_ARTIFACT,
        schema_overrides=expected_route_summary.schema,
    )
    if not route_summary.equals(expected_route_summary, null_equal=True):
        raise ValueError("entry-route summary does not reproduce from daily metrics")
    _verify_summary_semantics(summary, split_entry_route=False)
    _verify_summary_semantics(route_summary, split_entry_route=True)

    count_fields = (
        "input_marker_count",
        "entry_product_day_count",
        "full_date_count",
        "entry_action_aliases",
        "established_entry_aliases",
        "physical_entry_dependencies",
        "physical_policy_cells",
        "physical_policy_aliases",
        "complete_date_physical_policy_cells",
        "daily_rows",
        "policy_summary_rows",
        "entry_route_daily_rows",
        "entry_route_summary_rows",
    )
    if any(type(payload[name]) is not int or payload[name] < 0 for name in count_fields):
        raise ValueError("snapshot declared counts have invalid types/values")
    if (
        daily.height != int(payload["daily_rows"])
        or summary.height != int(payload["policy_summary_rows"])
        or route_daily.height != int(payload["entry_route_daily_rows"])
        or route_summary.height != int(payload["entry_route_summary_rows"])
    ):
        raise ValueError("snapshot declared table row count mismatch")
    if (
        int(daily["entry_positions"].sum())
        != int(payload["complete_date_physical_policy_cells"])
        or int(payload["complete_date_physical_policy_cells"])
        > int(payload["physical_policy_cells"])
        or int(payload["physical_policy_cells"])
        > int(payload["physical_policy_aliases"])
        or int(payload["physical_entry_dependencies"])
        > int(payload["established_entry_aliases"])
        or int(payload["established_entry_aliases"])
        > int(payload["entry_action_aliases"])
    ):
        raise ValueError("snapshot declared population counts are inconsistent")
    inventory = pl.read_csv(root / INVENTORY_ARTIFACT)
    _verify_inventory(inventory, payload, config)
    for name, frame in (
        (DAILY_ARTIFACT, daily),
        (DAILY_CSV_ARTIFACT, daily_csv),
        (SUMMARY_ARTIFACT, summary),
        (ENTRY_ROUTE_DAILY_ARTIFACT, route_daily),
        (ENTRY_ROUTE_DAILY_CSV_ARTIFACT, route_daily_csv),
        (ENTRY_ROUTE_SUMMARY_ARTIFACT, route_summary),
        (INVENTORY_ARTIFACT, inventory),
    ):
        metadata = artifacts[name]
        if frame.height != metadata["rows"] or frame.width != metadata["columns"]:
            raise ValueError(f"snapshot artifact shape mismatch: {name}")
    plt.imread(root / CHART_ARTIFACT)
    return payload


_DAILY_VALUE_COLUMNS = {
    "entry_positions",
    "entry_spot_shares",
    "entry_future_contracts",
    "cumulative_new_entry_one_way_spot_leg_entry_price_notional_twd",
    "same_day_completed",
    "nominal_eod_carry_positions",
    "nominal_unknown_positions",
    "strict_proven_flat",
    "strict_carry_positions",
    "strict_unresolved_positions",
    "cancel_race_unknown",
    "fill_unknown_at_eod",
    "partial_fill_carry_at_eod",
    "hedge_incomplete_residual",
    "nominal_eod_spot_shares",
    "nominal_eod_future_contracts",
    "nominal_eod_open_one_way_spot_leg_entry_price_notional_twd",
    "strict_unresolved_spot_shares",
    "strict_unresolved_future_contracts",
    "strict_unresolved_eod_one_way_spot_leg_entry_price_notional_twd",
    "completed_gross_twd",
    "completed_after19_twd",
    "completed_one_way_spot_leg_entry_price_notional_twd",
    "completed_gross_weighted_bp",
    "completed_after19_weighted_bp",
}
_TABLE_SEMANTIC_COLUMNS = {
    "analysis_only",
    "strategy_defensible",
    "alternative_policy_rows_additive",
    "entry_routes_pooled_descriptive_only",
    "entry_route_specific_descriptive_only",
    "pnl_scope",
    "cost_scope",
    "notional_semantics",
    "eod_notional_scope",
    "notional_is_eod_mark",
    "notional_is_two_leg_gross_exposure",
    "notional_is_futures_margin",
    "notional_is_capital_requirement",
}
_ADDITIVE_DAILY_COLUMNS = sorted(
    _DAILY_VALUE_COLUMNS
    - {"completed_gross_weighted_bp", "completed_after19_weighted_bp"}
)


def _verify_daily_semantics(
    frame: pl.DataFrame,
    payload: Mapping[str, object],
    *,
    split_entry_route: bool,
) -> None:
    key = ["Date", *(ENTRY_ROUTE_CELL_KEY if split_entry_route else CELL_KEY)]
    expected_columns = {
        *key,
        *_DAILY_VALUE_COLUMNS,
        *_TABLE_SEMANTIC_COLUMNS,
    }
    if set(frame.columns) != expected_columns:
        raise ValueError("daily metric schema field set mismatch")

    full_dates = tuple(map(str, payload["full_dates"]))
    excluded_dates = tuple(map(str, payload["excluded_partial_dates"]))
    if (
        not full_dates
        or tuple(sorted(set(full_dates))) != full_dates
        or tuple(sorted(set(excluded_dates))) != excluded_dates
        or set(full_dates) & set(excluded_dates)
        or len(full_dates) != int(payload["full_date_count"])
    ):
        raise ValueError("snapshot full/partial date declarations are invalid")
    scenarios = _scenario_frame()
    if split_entry_route:
        scenarios = pl.DataFrame({"entry_route": list(ENTRY_ROUTES)}).join(
            scenarios,
            how="cross",
        )
    expected_keys = (
        pl.DataFrame({"Date": list(full_dates)})
        .join(scenarios, how="cross")
        .select(key)
        .sort(key)
    )
    actual_keys = frame.select(key).sort(key)
    if not actual_keys.equals(expected_keys, null_equal=True):
        raise ValueError("daily metric scenario/date grid mismatch")

    table_semantics: Mapping[str, object] = {
        "analysis_only": True,
        "strategy_defensible": False,
        "alternative_policy_rows_additive": False,
        "entry_routes_pooled_descriptive_only": not split_entry_route,
        "entry_route_specific_descriptive_only": split_entry_route,
        "pnl_scope": PNL_SCOPE,
        "cost_scope": COST_SCOPE,
        "notional_semantics": NOTIONAL_SEMANTICS,
        "eod_notional_scope": EOD_NOTIONAL_SCOPE,
        "notional_is_eod_mark": False,
        "notional_is_two_leg_gross_exposure": False,
        "notional_is_futures_margin": False,
        "notional_is_capital_requirement": False,
    }
    for column, expected in table_semantics.items():
        values = frame[column].unique().to_list()
        if values != [expected]:
            raise ValueError(f"daily semantic column mismatch: {column}")

    nonnegative = sorted(
        _DAILY_VALUE_COLUMNS
        - {
            "completed_gross_twd",
            "completed_after19_twd",
            "completed_gross_weighted_bp",
            "completed_after19_weighted_bp",
        }
    )
    if frame.filter(pl.any_horizontal(*(pl.col(name) < 0 for name in nonnegative))).height:
        raise ValueError("daily counts/quantities/notional must be nonnegative")
    invalid_counts = frame.filter(
        pl.col("same_day_completed")
        + pl.col("nominal_eod_carry_positions")
        + pl.col("nominal_unknown_positions")
        != pl.col("entry_positions")
    )
    if invalid_counts.height:
        raise ValueError("daily nominal branch counts do not partition entries")
    if frame.filter(
        pl.col("strict_proven_flat") + pl.col("strict_unresolved_positions")
        != pl.col("entry_positions")
    ).height:
        raise ValueError("daily strict branch counts do not partition entries")

    expected_after_cost = (
        pl.col("completed_gross_twd")
        - pl.col("completed_one_way_spot_leg_entry_price_notional_twd")
        * (float(payload["config"]["cost_sensitivity_bp"]) / 10_000.0)
    )
    if frame.filter(
        (pl.col("completed_after19_twd") - expected_after_cost).abs() > 1e-6
    ).height:
        raise ValueError("daily completed-only cost sensitivity mismatch")
    expected_gross_bp = pl.when(
        pl.col("completed_one_way_spot_leg_entry_price_notional_twd") > 0
    ).then(
        pl.col("completed_gross_twd")
        / pl.col("completed_one_way_spot_leg_entry_price_notional_twd")
        * 10_000.0
    ).otherwise(None)
    expected_after_bp = pl.when(
        pl.col("completed_one_way_spot_leg_entry_price_notional_twd") > 0
    ).then(
        pl.col("completed_after19_twd")
        / pl.col("completed_one_way_spot_leg_entry_price_notional_twd")
        * 10_000.0
    ).otherwise(None)
    has_completed_notional = (
        pl.col("completed_one_way_spot_leg_entry_price_notional_twd") > 0
    )
    if frame.filter(
        (pl.col("completed_gross_weighted_bp").is_not_null() != has_completed_notional)
        | (
            pl.col("completed_after19_weighted_bp").is_not_null()
            != has_completed_notional
        )
    ).height:
        raise ValueError("daily completed-only weighted-bp nullability mismatch")
    mismatch = frame.select(
        (
            pl.col("completed_gross_weighted_bp") - expected_gross_bp
        ).abs().fill_null(0).max().alias("gross"),
        (
            pl.col("completed_after19_weighted_bp") - expected_after_bp
        ).abs().fill_null(0).max().alias("after"),
    ).row(0)
    if any(float(value or 0.0) > 1e-9 for value in mismatch):
        raise ValueError("daily completed-only weighted-bp mismatch")

    entry_columns = [
        "entry_positions",
        "entry_spot_shares",
        "entry_future_contracts",
        "cumulative_new_entry_one_way_spot_leg_entry_price_notional_twd",
    ]
    identity_key = ["Date", "boundary_quantile"]
    if split_entry_route:
        identity_key.append("entry_route")
    if frame.group_by(identity_key).agg(
        *(pl.col(name).n_unique().alias(name) for name in entry_columns)
    ).filter(
        pl.any_horizontal(*(pl.col(name) != 1 for name in entry_columns))
    ).height:
        raise ValueError("entry volume differs across alternative exit policies")


def _verify_route_rollup(
    pooled: pl.DataFrame,
    route_daily: pl.DataFrame,
) -> None:
    rollup = route_daily.group_by(["Date", *CELL_KEY]).agg(
        *(pl.col(name).sum().alias(name) for name in _ADDITIVE_DAILY_COLUMNS)
    )
    joined = pooled.join(
        rollup,
        on=["Date", *CELL_KEY],
        how="left",
        suffix="_route_sum",
        validate="1:1",
    )
    mismatches = []
    for name in _ADDITIVE_DAILY_COLUMNS:
        left = pl.col(name)
        right = pl.col(f"{name}_route_sum")
        mismatches.append(
            (left - right).abs() > (1e-6 if pooled.schema[name].is_float() else 0)
        )
    if joined.filter(pl.any_horizontal(*mismatches)).height:
        raise ValueError("pooled daily metrics do not equal entry-route rollup")


def _verify_summary_semantics(
    frame: pl.DataFrame,
    *,
    split_entry_route: bool,
) -> None:
    for column, expected in {
        "analysis_only": True,
        "strategy_defensible": False,
        "alternative_policy_rows_additive": False,
        "entry_routes_pooled_descriptive_only": not split_entry_route,
        "entry_route_specific_descriptive_only": split_entry_route,
        "pnl_scope": PNL_SCOPE,
        "cost_scope": COST_SCOPE,
        "notional_semantics": NOTIONAL_SEMANTICS,
        "eod_notional_scope": EOD_NOTIONAL_SCOPE,
        "notional_is_eod_mark": False,
        "notional_is_two_leg_gross_exposure": False,
        "notional_is_futures_margin": False,
        "notional_is_capital_requirement": False,
        "strategy_metric_status": (
            "suppressed_cross_terminal_and_full_costs_incomplete"
        ),
    }.items():
        if frame[column].unique().to_list() != [expected]:
            raise ValueError(f"summary semantic column mismatch: {column}")
    suppressed = (
        "strategy_daily_win_rate",
        "strategy_mdd_twd",
        "strategy_annualized_sharpe",
    )
    if any(frame[name].null_count() != frame.height for name in suppressed):
        raise ValueError("strategy metrics must remain fully suppressed")


def _verify_inventory(
    inventory: pl.DataFrame,
    payload: Mapping[str, object],
    config: Mapping[str, object],
) -> None:
    expected_columns = {
        "ordinal",
        "Date",
        "ValueCode",
        "marker_path",
        "marker_sha256",
        "marker_bytes",
        "runner_version",
        "runner_config_sha256",
        "partition_config_sha256",
        "entry_action_sha256",
        "position_policy_sha256",
        "position_policy_bytes",
        "position_policy_rows",
        "position_policy_columns",
    }
    if set(inventory.columns) != expected_columns:
        raise ValueError("input inventory schema field set mismatch")
    marker_count = int(payload["input_marker_count"])
    if inventory.height != marker_count or marker_count > int(
        payload["entry_product_day_count"]
    ):
        raise ValueError("input inventory row count mismatch")
    if inventory["ordinal"].to_list() != list(range(1, marker_count + 1)):
        raise ValueError("input inventory ordinals are not contiguous")
    keys = [
        (str(date), str(value_code))
        for date, value_code in inventory.select("Date", "ValueCode").iter_rows()
    ]
    if keys != sorted(keys) or len(set(keys)) != marker_count:
        raise ValueError("input inventory product-day order/uniqueness mismatch")
    if "/".join(keys[-1]) != payload["input_last_key"]:
        raise ValueError("input inventory last-key mismatch")
    digest = hashlib.sha256()
    exit_root = Path(str(config["exit_root"]))
    entry_root = Path(str(config["entry_root"]))
    entry_manifest_path = entry_root / ENTRY_MANIFEST
    if not entry_manifest_path.is_file():
        raise ValueError("formal entry manifest is missing")
    entry_manifest = pl.read_parquet(entry_manifest_path).sort(["Date", "ValueCode"])
    expected_keys = [
        (str(date), str(value_code))
        for date, value_code in entry_manifest.select(
            "Date",
            "ValueCode",
        ).iter_rows()
    ]
    if (
        len(expected_keys) != int(payload["entry_product_day_count"])
        or keys != expected_keys[:marker_count]
    ):
        raise ValueError("input inventory is not a strict entry-manifest prefix")
    expected_counts = _counts_by_date(expected_keys)
    snapshot_counts = _counts_by_date(keys)
    expected_full_dates = tuple(
        sorted(
            date
            for date, count in snapshot_counts.items()
            if count == expected_counts[date]
        )
    )
    expected_partial_dates = tuple(
        sorted(
            date
            for date, count in snapshot_counts.items()
            if count != expected_counts[date]
        )
    )
    if (
        expected_full_dates != tuple(map(str, payload["full_dates"]))
        or expected_partial_dates
        != tuple(map(str, payload["excluded_partial_dates"]))
    ):
        raise ValueError("snapshot full/partial date lineage mismatch")
    for row, (date, value_code) in zip(inventory.iter_rows(named=True), keys):
        relative_marker = f"Date={date}/ValueCode={value_code}/complete.json"
        marker_path = exit_root / relative_marker
        if Path(str(row["marker_path"])) != marker_path:
            raise ValueError("input inventory marker path identity mismatch")
        for field in (
            "marker_sha256",
            "runner_config_sha256",
            "partition_config_sha256",
            "entry_action_sha256",
            "position_policy_sha256",
        ):
            value = str(row[field])
            try:
                decoded = bytes.fromhex(value)
            except ValueError as error:
                raise ValueError(f"input inventory invalid SHA-256: {field}") from error
            if len(decoded) != 32 or value != value.lower():
                raise ValueError(f"input inventory invalid SHA-256: {field}")
        if row["runner_config_sha256"] != payload["exit_runner_config_sha256"]:
            raise ValueError("input inventory runner config lineage mismatch")
        if any(
            int(row[field]) <= 0
            for field in (
                "marker_bytes",
                "position_policy_bytes",
                "position_policy_columns",
            )
        ) or int(row["position_policy_rows"]) < 0:
            raise ValueError("input inventory artifact metadata is invalid")
        marker_bytes = marker_path.read_bytes()
        if (
            len(marker_bytes) != int(row["marker_bytes"])
            or hashlib.sha256(marker_bytes).hexdigest() != row["marker_sha256"]
        ):
            raise ValueError("formal exit marker no longer matches frozen inventory")
        marker = json.loads(marker_bytes)
        marker_config = marker.get("config")
        marker_artifacts = marker.get("artifacts")
        if (
            marker.get("complete") is not True
            or marker.get("Date") != date
            or marker.get("ValueCode") != value_code
            or marker.get("runner_version") != row["runner_version"]
            or marker.get("runner_config_sha256")
            != row["runner_config_sha256"]
            or marker.get("config_sha256") != row["partition_config_sha256"]
            or not isinstance(marker_config, dict)
            or _canonical_sha256(marker_config) != marker.get("config_sha256")
            or not isinstance(marker_artifacts, dict)
        ):
            raise ValueError("formal exit marker identity/lineage mismatch")
        position_meta = marker_artifacts.get(POSITION_ARTIFACT)
        marker_source = marker_config.get("source")
        action_source = (
            marker_source.get("action_source")
            if isinstance(marker_source, dict)
            else None
        )
        if (
            not isinstance(position_meta, dict)
            or not isinstance(action_source, dict)
            or action_source.get("sha256") != row["entry_action_sha256"]
            or position_meta.get("sha256") != row["position_policy_sha256"]
            or position_meta.get("bytes") != row["position_policy_bytes"]
            or position_meta.get("rows") != row["position_policy_rows"]
            or position_meta.get("columns") != row["position_policy_columns"]
        ):
            raise ValueError("formal exit source/artifact declaration mismatch")
        position_path = marker_path.parent / POSITION_ARTIFACT
        action_path = (
            entry_root
            / f"Date={date}"
            / f"ValueCode={value_code}"
            / ENTRY_ACTION_ARTIFACT
        )
        if (
            position_path.stat().st_size != int(row["position_policy_bytes"])
            or _file_sha256(position_path) != row["position_policy_sha256"]
            or _file_sha256(action_path) != row["entry_action_sha256"]
            or pl.scan_parquet(position_path).select(pl.len()).collect().item()
            != int(row["position_policy_rows"])
            or len(pl.read_parquet_schema(position_path))
            != int(row["position_policy_columns"])
        ):
            raise ValueError("formal snapshot source artifact no longer matches inventory")
        digest.update(relative_marker.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(str(row["marker_sha256"])))
    if digest.hexdigest() != payload["input_marker_inventory_sha256"]:
        raise ValueError("input inventory digest mismatch")

    population = payload["population_audit"]
    expected_population_constants = {
        "expected_boundary_quantiles": [50, 80, 95],
        "expected_entry_routes": list(ENTRY_ROUTES),
        "expected_exit_routes": [
            "future_bid_spot_taker",
            "spot_ask_future_taker",
        ],
        "expected_rules": ["frozen_center", "frozen_lower"],
        "policy_rows_per_established_entry": 4,
    }
    expected_population_fields = {
        *expected_population_constants,
        "established_entry_aliases",
        "validated_position_rows",
    }
    if (
        not isinstance(population, dict)
        or set(population) != expected_population_fields
        or any(
            population.get(name) != value
            for name, value in expected_population_constants.items()
        )
    ):
        raise ValueError("population audit semantic identity mismatch")
    if (
        population.get("established_entry_aliases")
        != payload["established_entry_aliases"]
        or population.get("validated_position_rows")
        != payload["physical_policy_aliases"]
        or int(payload["established_entry_aliases"]) * 4
        != int(payload["physical_policy_aliases"])
        or int(payload["physical_policy_cells"])
        > int(payload["physical_policy_aliases"])
    ):
        raise ValueError("population audit count lineage mismatch")


def _verify_canonical_artifact_contract(
    root: Path,
    artifacts: Mapping[str, object],
) -> None:
    for name, expected_schema in _CANONICAL_ARTIFACT_SCHEMAS.items():
        metadata = artifacts.get(name)
        if not isinstance(metadata, dict) or set(metadata) != {
            "bytes",
            "sha256",
            "kind",
            "rows",
            "columns",
            "schema",
        }:
            raise ValueError(f"canonical tabular metadata field set mismatch: {name}")
        path = root / name
        frame = (
            pl.read_parquet(path)
            if path.suffix == ".parquet"
            else pl.read_csv(path)
        )
        actual_schema = tuple(
            (column, str(dtype)) for column, dtype in frame.schema.items()
        )
        expected_kind = "parquet" if path.suffix == ".parquet" else "csv"
        if (
            actual_schema != expected_schema
            or metadata["schema"] != dict(expected_schema)
            or metadata["kind"] != expected_kind
            or metadata["columns"] != len(expected_schema)
        ):
            raise ValueError(f"canonical artifact schema mismatch: {name}")
    chart = artifacts.get(CHART_ARTIFACT)
    if not isinstance(chart, dict) or set(chart) != {
        "bytes",
        "sha256",
        "pixel_height",
        "pixel_width",
        *_CANONICAL_CHART_METADATA,
    }:
        raise ValueError("canonical chart metadata field set mismatch")
    if (
        any(
            chart.get(name) != value
            for name, value in _CANONICAL_CHART_METADATA.items()
        )
        or not isinstance(chart.get("pixel_height"), int)
        or not isinstance(chart.get("pixel_width"), int)
        or int(chart["pixel_height"]) <= 0
        or int(chart["pixel_width"]) <= 0
    ):
        raise ValueError("canonical chart metadata mismatch")


def _plot_diagnostics(
    daily: pl.DataFrame,
    destination: Path,
    config: SnapshotConfig,
) -> None:
    ordered = daily.sort([*CELL_KEY, "Date"])
    figure, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True)
    colors = plt.get_cmap("tab20").colors[:12]
    scenarios = list(_scenario_frame().iter_rows(named=True))

    # Entry volume is identical across the four alternative exit policies.
    entry_check = ordered.group_by(["Date", "boundary_quantile"]).agg(
        pl.col(
            "cumulative_new_entry_one_way_spot_leg_entry_price_notional_twd"
        )
        .n_unique()
        .alias("values")
    )
    if entry_check.filter(pl.col("values") != 1).height:
        raise ValueError("entry notional differs across exit alternatives")
    entry = ordered.group_by(["Date", "boundary_quantile"]).agg(
        pl.col(
            "cumulative_new_entry_one_way_spot_leg_entry_price_notional_twd"
        )
        .first()
        .alias("notional")
    )
    for quantile, color in zip((50, 80, 95), ("#1f77b4", "#ff7f0e", "#2ca02c")):
        frame = entry.filter(pl.col("boundary_quantile") == quantile).sort("Date")
        axes[0].plot(
            frame["Date"].to_list(),
            (frame["notional"] / 1_000_000.0).to_list(),
            label=f"q{quantile}",
            color=color,
            linewidth=1.8,
        )
    axes[0].set_title(
        "Daily pooled entry-route candidates: one-way spot-leg entry-price "
        "notional (descriptive upper envelope)"
    )
    axes[0].set_ylabel("One-way spot-leg TWD million")
    axes[0].legend(ncol=3, loc="upper left")

    scenario_handles: list[Line2D] = []
    for scenario, color in zip(scenarios, colors):
        frame = _select_scenario(ordered, scenario)
        label = _scenario_label(scenario)
        x = frame["Date"].to_list()
        axes[1].plot(
            x,
            (
                frame[
                    "nominal_eod_open_one_way_spot_leg_entry_price_notional_twd"
                ]
                / 1_000_000.0
            ).to_list(),
            color=color,
            linewidth=1.25,
            alpha=0.95,
        )
        axes[1].plot(
            x,
            (
                frame[
                    "strict_unresolved_eod_one_way_spot_leg_entry_price_notional_twd"
                ]
                / 1_000_000.0
            ).to_list(),
            color=color,
            linewidth=1.0,
            linestyle="--",
            alpha=0.55,
        )
        gross = frame["completed_gross_twd"].cum_sum() / 1_000_000.0
        sensitivity = frame["completed_after19_twd"].cum_sum() / 1_000_000.0
        axes[2].plot(
            x,
            gross.to_list(),
            color=color,
            linewidth=1.0,
            linestyle=":",
            alpha=0.55,
        )
        axes[2].plot(
            x,
            sensitivity.to_list(),
            color=color,
            linewidth=1.4,
            alpha=0.95,
        )
        scenario_handles.append(Line2D([0], [0], color=color, label=label))

    axes[1].set_title(
        "Positions classified open at EOD: one-way spot-leg entry-price notional "
        "(nominal solid / strict unresolved dashed)"
    )
    axes[1].set_ylabel("One-way spot-leg TWD million")
    axes[2].set_title(
        "Completed same-day cashflow only: gross (dotted) vs 19 bp sensitivity (solid)"
    )
    axes[2].set_ylabel("Cumulative TWD million")
    axes[2].set_xlabel("Entry session")
    axes[2].axhline(0.0, color="black", linewidth=0.7)
    axes[2].text(
        0.5,
        0.50,
        "NOT EQUITY CURVE\nUNRESOLVED / CROSS-SESSION CASHFLOW EXCLUDED",
        transform=axes[2].transAxes,
        ha="center",
        va="center",
        color="#b00020",
        fontsize=18,
        fontweight="bold",
        alpha=0.30,
        rotation=8,
    )
    for axis in axes:
        axis.grid(True, linestyle="--", alpha=0.25)
        axis.tick_params(axis="x", labelrotation=45, labelsize=8)
    style_handles = [
        Line2D([0], [0], color="black", linestyle="-", label="primary / 19 bp"),
        Line2D([0], [0], color="black", linestyle="--", label="strict unresolved"),
        Line2D([0], [0], color="black", linestyle=":", label="completed gross"),
    ]
    figure.legend(
        handles=[*scenario_handles, *style_handles],
        loc="center right",
        bbox_to_anchor=(0.995, 0.5),
        fontsize=8,
        title="Alternative policy (non-additive)",
    )
    figure.suptitle(
        f"Analysis-only exit-maker snapshot | {config.snapshot_time} | "
        f"{config.marker_count} markers",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.01,
        0.005,
        "Selection-biased interim prefix; entry routes are pooled independent candidates; "
        "notional is not an EOD mark, two-leg exposure, margin, or capital. Cancel ACK, "
        "cross-session cashflow, full costs, and joint allocation are incomplete.",
        fontsize=9,
        color="#b00020",
    )
    figure.tight_layout(rect=(0.0, 0.02, 0.82, 0.97))
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _verify_tabular_round_trip(
    stage: Path,
    inventory: pl.DataFrame,
    tables: SnapshotTables,
) -> None:
    loaded_inventory = pl.read_csv(
        stage / INVENTORY_ARTIFACT,
        schema_overrides=inventory.schema,
    )
    loaded_daily_parquet = pl.read_parquet(stage / DAILY_ARTIFACT)
    loaded_daily_csv = pl.read_csv(
        stage / DAILY_CSV_ARTIFACT,
        schema_overrides=tables.daily.schema,
    )
    loaded_summary = pl.read_csv(
        stage / SUMMARY_ARTIFACT,
        schema_overrides=tables.summary.schema,
    )
    loaded_entry_route_daily_parquet = pl.read_parquet(
        stage / ENTRY_ROUTE_DAILY_ARTIFACT
    )
    loaded_entry_route_daily_csv = pl.read_csv(
        stage / ENTRY_ROUTE_DAILY_CSV_ARTIFACT,
        schema_overrides=tables.entry_route_daily.schema,
    )
    loaded_entry_route_summary = pl.read_csv(
        stage / ENTRY_ROUTE_SUMMARY_ARTIFACT,
        schema_overrides=tables.entry_route_summary.schema,
    )
    if not inventory.equals(loaded_inventory, null_equal=True):
        raise ValueError("marker inventory CSV round-trip mismatch")
    if not tables.daily.equals(loaded_daily_parquet, null_equal=True):
        raise ValueError("daily Parquet round-trip mismatch")
    if not loaded_daily_parquet.equals(loaded_daily_csv, null_equal=True):
        raise ValueError("daily CSV round-trip mismatch")
    if not tables.summary.equals(loaded_summary, null_equal=True):
        raise ValueError("policy summary CSV round-trip mismatch")
    if not tables.entry_route_daily.equals(
        loaded_entry_route_daily_parquet,
        null_equal=True,
    ):
        raise ValueError("entry-route daily Parquet round-trip mismatch")
    if not loaded_entry_route_daily_parquet.equals(
        loaded_entry_route_daily_csv,
        null_equal=True,
    ):
        raise ValueError("entry-route daily CSV round-trip mismatch")
    if not tables.entry_route_summary.equals(
        loaded_entry_route_summary,
        null_equal=True,
    ):
        raise ValueError("entry-route summary CSV round-trip mismatch")
    plt.imread(stage / CHART_ARTIFACT)


def _artifact_metadata(path: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }
    if path.suffix == ".parquet":
        frame = pl.read_parquet(path)
        result.update(
            {
                "kind": "parquet",
                "rows": frame.height,
                "columns": frame.width,
                "schema": {name: str(dtype) for name, dtype in frame.schema.items()},
            }
        )
    elif path.suffix == ".csv":
        frame = pl.read_csv(path)
        result.update(
            {
                "kind": "csv",
                "rows": frame.height,
                "columns": frame.width,
                "schema": {name: str(dtype) for name, dtype in frame.schema.items()},
            }
        )
    elif path.suffix == ".png":
        image = plt.imread(path)
        result.update(
            {
                "kind": "png",
                "pixel_height": int(image.shape[0]),
                "pixel_width": int(image.shape[1]),
                "channels": int(image.shape[2]) if image.ndim == 3 else 1,
            }
        )
    return result


def _scenario_frame() -> pl.DataFrame:
    rows = [
        {
            "boundary_quantile": quantile,
            "exit_rule_id": rule,
            "exit_route": route,
        }
        for quantile in (50, 80, 95)
        for rule in ("frozen_center", "frozen_lower")
        for route in ("future_bid_spot_taker", "spot_ask_future_taker")
    ]
    return pl.from_dicts(rows, infer_schema_length=None)


def _select_scenario(
    frame: pl.DataFrame, scenario: Mapping[str, object]
) -> pl.DataFrame:
    return frame.filter(
        (pl.col("boundary_quantile") == int(scenario["boundary_quantile"]))
        & (pl.col("exit_rule_id") == str(scenario["exit_rule_id"]))
        & (pl.col("exit_route") == str(scenario["exit_route"]))
    ).sort("Date")


def _scenario_label(scenario: Mapping[str, object]) -> str:
    rule = "C" if scenario["exit_rule_id"] == "frozen_center" else "L"
    route = (
        "F-M"
        if scenario["exit_route"] == "future_bid_spot_taker"
        else "S-M"
    )
    return f"q{scenario['boundary_quantile']} {rule} {route}"


def _counts_by_date(
    product_days: Iterable[tuple[str, str]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for date, _ in product_days:
        counts[date] = counts.get(date, 0) + 1
    return counts


def _marker_path_key(path: Path) -> tuple[str, str]:
    return (
        path.parent.parent.name.removeprefix("Date="),
        path.parent.name.removeprefix("ValueCode="),
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exit-root", type=Path, required=True)
    parser.add_argument("--entry-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--marker-count", type=int, required=True)
    parser.add_argument("--snapshot-time", required=True)
    parser.add_argument("--marker-inventory-sha256", required=True)
    parser.add_argument("--last-key", required=True)
    parser.add_argument("--cost-sensitivity-bp", type=float, default=19.0)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.verify_only:
        marker = verify_interim_portfolio_snapshot(args.output)
    else:
        marker = build_interim_portfolio_snapshot(
            exit_root=args.exit_root,
            entry_root=args.entry_root,
            output_root=args.output,
            config=SnapshotConfig(
                marker_count=args.marker_count,
                snapshot_time=args.snapshot_time,
                expected_marker_inventory_sha256=args.marker_inventory_sha256,
                expected_last_key=args.last_key,
                cost_sensitivity_bp=args.cost_sensitivity_bp,
            ),
        )
    print(json.dumps(marker, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
