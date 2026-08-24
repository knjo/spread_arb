"""Source-bound post-cross policy, capital, and position-limit evaluation.

This module deliberately starts from the *formal* filled-entry population.  It
does not replay ticks and it never treats q/rule/route alternatives as
additive.  Its post-cross-only narrow loader reuses the frozen hash and lineage
validators before any output directory is created.  Every same-day artifact is
still hash/row/schema verified, but only projected entry actions, established
positions, and their exact bound exit rules are materialised; the enormous
candidate/alias replay tables are never loaded.  The terminal overlay remains
the nominal-instant-cancel-V0 formal cross-session artifact.

The report has two distinct layers:

* descriptive terminal-date cashflow and outstanding-position diagnostics;
* D-safe challenger/risk-control diagnostics.

Neither layer is called EV unless every label in the relevant prequential
window is mature and point-priced and an exact, source-bound component cost
profile covers every physical policy path.  A missing cost profile, censored
or unknown terminal mass, or the nominal cancel assumption therefore leaves
the selected action null and emits an explicit NO-GO reason.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Iterable, Mapping, Sequence

import polars as pl

from . import exit_maker_report as _exit_report
from . import exit_maker_study as _exit_study
from . import filled_entry_report as _filled_report
from .exit_maker_report import ExitMakerPartitionInputs
from .filled_entry_report import (
    FROZEN_BOUNDARY_QUANTILES,
    FROZEN_ENTRY_ROUTES,
    FROZEN_EXIT_ROUTES,
    FROZEN_EXIT_RULE_IDS,
    FilledEntryPrimaryReport,
    _expected_formal_prerequisite_identity,
    _prerequisite_session_calendar,
    _report_implementation_sources,
    build_filled_entry_primary_report,
    load_cross_session_nominal_terminal_facts,
)
from .finite_horizon_policy import _student_t_quantile


EVALUATOR_VERSION = (
    "post_cross_position_evaluator_v4_cache_bounded_preaggregated_actions"
)
BUNDLE_SCHEMA_VERSION = (
    "post_cross_position_evaluation_bundle_v4_cache_bounded_rebuild_verified"
)
COST_PROFILE_SCHEMA_VERSION = "post_cross_path_cost_profile_v1"
POLICY_KEY: tuple[str, ...] = (
    "boundary_quantile",
    "entry_route",
    "exit_rule_id",
    "exit_route",
)
PHYSICAL_PATH_KEY: tuple[str, ...] = (
    "Date",
    "ValueCode",
    "QuoteCode",
    "entry_route",
    "boundary_quantile",
    "entry_raw_order_fact_id",
    "position_established_ns",
    "exit_rule_id",
    "exit_route",
)
COST_COMPONENT_COLUMNS: tuple[str, ...] = (
    "fee_cost_bp",
    "tax_cost_bp",
    "commission_cost_bp",
    "financing_cost_bp",
    "overnight_cost_bp",
    "cancel_cost_bp",
    "emergency_cost_bp",
    "other_risk_cost_bp",
)

_FORMAL_CROSS_AUDIT_COLUMNS: tuple[str, ...] = (
    "last_observed_session_date",
    "exit_decision_time_ns",
    "outcome_type",
    "outcome_status",
    "terminal_cashflow_priced",
    "cancel_semantics",
    "nominal_cancel_model_assumption",
    "pathwise_ev_ready",
    "joint_volume_allocated",
)

_NARROW_ACTION_SCHEMA: dict[str, pl.DataType] = {
    "Date": pl.String,
    "ValueCode": pl.String,
    "QuoteCode": pl.String,
    "route": pl.String,
    "boundary_quantile": pl.Int64,
    "raw_order_fact_id": pl.String,
    "policy_generation_id": pl.String,
    "full_fill": pl.Boolean,
    "partial_fill": pl.Boolean,
    "any_fill": pl.Boolean,
    "cancel_required": pl.Boolean,
    "entry_hedge_status": pl.String,
    "full_fill_recv_time_ns": pl.Int64,
    "entry_hedge_decision_time_ns": pl.Int64,
    "entry_hedge_label_observed": pl.Boolean,
    "entry_hedge_executable": pl.Boolean,
    "entry_spot_price": pl.Float64,
    "entry_hedge_contract_size_shares": pl.Int64,
}

_NARROW_POSITION_SCHEMA: dict[str, pl.DataType] = {
    "Date": pl.String,
    "ValueCode": pl.String,
    "QuoteCode": pl.String,
    "entry_route": pl.String,
    "entry_policy_generation_id": pl.String,
    "entry_raw_order_fact_id": pl.String,
    "exit_rule_id": pl.String,
    "exit_route": pl.String,
    "exit_policy_trial_id": pl.String,
    "exit_threshold_basis_bp": pl.Float64,
    "exit_rule_source_asof_date": pl.String,
    "position_status": pl.String,
    "position_established_ns": pl.Int64,
    "nominal_instant_cancel_v0_branch": pl.String,
    "gross_cycle_pnl_twd": pl.Float64,
}

_NARROW_EXIT_RULE_SCHEMA: dict[str, pl.DataType] = {
    "Date": pl.String,
    "ValueCode": pl.String,
    "QuoteCode": pl.String,
    "route": pl.String,
    "policy_generation_id": pl.String,
    "raw_order_fact_id": pl.String,
    "exit_rule_id": pl.String,
    "exit_threshold_basis_bp": pl.Float64,
    "exit_rule_source_asof_date": pl.String,
}

# The execution producer intentionally emits these two minimal typed schemas
# for a legitimate zero-action product-day.  They predate the derived
# fill/hedge and threshold/source columns, so a zero-row projection may be
# synthesized only from these exact ordered schemas (or from a source that
# already contains every projected column with the exact dtype).
_ENTRY_ZERO_ACTION_SCHEMA: dict[str, pl.DataType] = {
    "Date": pl.String,
    "ValueCode": pl.String,
    "QuoteCode": pl.String,
    "route": pl.String,
    "boundary_quantile": pl.Int64,
    "lookup_action_id": pl.String,
    "raw_order_fact_id": pl.String,
    "policy_generation_id": pl.String,
    "entry_execution_outcome": pl.String,
    "pathwise_ev_ready": pl.Boolean,
}

_ENTRY_ZERO_EXIT_RULE_SCHEMA: dict[str, pl.DataType] = {
    "Date": pl.String,
    "ValueCode": pl.String,
    "QuoteCode": pl.String,
    "route": pl.String,
    "raw_order_fact_id": pl.String,
    "policy_generation_id": pl.String,
    "exit_rule_id": pl.String,
    "branch_status": pl.String,
    "same_day_exit": pl.Boolean,
    "overnight_carry": pl.Boolean,
    "terminal_outcome": pl.Boolean,
    "needs_next_session_label": pl.Boolean,
    "gross_cycle_pnl_twd": pl.Float64,
    "eod_liquidation_gross_pnl_twd": pl.Float64,
    "pathwise_ev_ready": pl.Boolean,
}

# Nonempty execution artifacts are produced from Polars expressions rather
# than an explicit terminal cast.  The frozen 60-day root therefore contains
# five *exact* ordered schemas: the canonical schema plus four legitimate
# inference variants.  Most importantly, a product-day with no full fills has
# an all-null ``full_fill_recv_time_ns`` physical column, which Parquet records
# as ``Null`` rather than ``Int64``.  Keep the complete producer schema here so
# accepting that nullable inference cannot turn into a projection-only
# fail-open.  Only the exact frozen variants below are admitted.
_ENTRY_CANONICAL_NONZERO_ACTION_SCHEMA: dict[str, pl.DataType] = {
    "Date": pl.String,
    "ValueCode": pl.String,
    "QuoteCode": pl.String,
    "route": pl.String,
    "maker_market": pl.String,
    "maker_side": pl.String,
    "boundary_quantile": pl.Int64,
    "boundary_role": pl.String,
    "parameter_version": pl.String,
    "source_asof_date": pl.String,
    "spread_pair_epoch": pl.Int64,
    "raw_order_fact_id": pl.String,
    "policy_generation_id": pl.String,
    "target_price_tick": pl.Int64,
    "target_price": pl.Float64,
    "target_rank_at_submit": pl.String,
    "threshold_basis_bp": pl.Float64,
    "effective_basis_bp": pl.Float64,
    "initial_queue_ahead": pl.Int64,
    "queue_known": pl.Boolean,
    "intended_quantity": pl.Int64,
    "submit_recv_time_ns": pl.Int64,
    "submit_event_sequence": pl.Int64,
    "submit_row_index": pl.Int64,
    "nominal_stop_recv_time_ns": pl.Int64,
    "nominal_stop_reason": pl.String,
    "first_fill_recv_time_ns": pl.Int64,
    "first_fill_event_sequence": pl.Int64,
    "first_fill_row_index": pl.Int64,
    "full_fill_recv_time_ns": pl.Int64,
    "full_fill_event_sequence": pl.Int64,
    "full_fill_row_index": pl.Int64,
    "known_filled_quantity": pl.Int64,
    "any_fill": pl.Boolean,
    "full_fill": pl.Boolean,
    "partial_fill": pl.Boolean,
    "trade_through_fill": pl.Boolean,
    "fill_reason": pl.String,
    "terminal_recv_time_ns": pl.Int64,
    "terminal_reason": pl.String,
    "cancel_required": pl.Boolean,
    "lifetime_ms": pl.Float64,
    "time_first_to_full_ms": pl.Float64,
    "spot_book_age_ms_at_submit": pl.Float64,
    "future_book_age_ms_at_submit": pl.Float64,
    "peak_nominal_layers_policy_day": pl.Int64,
    "independent_event_label": pl.Boolean,
    "joint_volume_allocated": pl.Boolean,
    "shadow_touch_10ms": pl.Boolean,
    "shadow_touch_50ms": pl.Boolean,
    "shadow_touch_100ms": pl.Boolean,
    "shadow_touch_500ms": pl.Boolean,
    "shadow_through_10ms": pl.Boolean,
    "shadow_through_50ms": pl.Boolean,
    "shadow_through_100ms": pl.Boolean,
    "shadow_through_500ms": pl.Boolean,
    "lookup_action_id": pl.String,
    "same_price_policy_aliases": pl.UInt32,
    "entry_hedge_status": pl.String,
    "entry_hedge_decision_time_ns": pl.Int64,
    "entry_hedge_decision_snapshot_recv_time_ns": pl.Int64,
    "entry_hedge_decision_snapshot_event_sequence": pl.Int64,
    "entry_hedge_decision_snapshot_row_index": pl.Int64,
    "entry_hedge_arrival_reference_price": pl.Float64,
    "entry_hedge_decision_best_price": pl.Float64,
    "entry_hedge_executable_vwap_price": pl.Float64,
    "entry_hedge_partial_vwap_price": pl.Float64,
    "entry_hedge_available_quantity": pl.Int64,
    "entry_hedge_executed_quantity": pl.Int64,
    "entry_hedge_depth_shortfall": pl.Int64,
    "entry_hedge_levels_swept": pl.Int64,
    "entry_hedge_signed_latency_slippage_bp": pl.Float64,
    "entry_hedge_signed_depth_slippage_bp": pl.Float64,
    "entry_hedge_signed_total_slippage_bp": pl.Float64,
    "entry_hedge_decision_book_age_ms": pl.Float64,
    "entry_hedge_contract_size_shares": pl.Int64,
    "same_absolute_price_alias": pl.Boolean,
    "rank_bucket": pl.String,
    "queue_bucket": pl.String,
    "tod_bucket": pl.String,
    "freshness_bucket": pl.String,
    "entry_hedge_label_observed": pl.Boolean,
    "entry_hedge_executable": pl.Boolean,
    "entry_execution_outcome": pl.String,
    "entry_future_price": pl.Float64,
    "entry_spot_price": pl.Float64,
    "entry_locked_basis_bp": pl.Float64,
    "entry_gross_cash_edge_twd": pl.Float64,
    "same_day_exit_status": pl.String,
    "overnight_branch_status": pl.String,
    "fees_tax_status": pl.String,
    "executable_maker_exit_included": pl.Boolean,
    "fees_tax_included": pl.Boolean,
    "overnight_cost_included": pl.Boolean,
    "pathwise_ev_ready": pl.Boolean,
    "contains_target_day_outcome": pl.Boolean,
}


def _entry_action_schema_variant(
    changes: Mapping[str, pl.DataType],
) -> dict[str, pl.DataType]:
    if not set(changes).issubset(_ENTRY_CANONICAL_NONZERO_ACTION_SCHEMA):
        raise RuntimeError("entry action schema variant changes an unknown column")
    return {
        name: changes.get(name, dtype)
        for name, dtype in _ENTRY_CANONICAL_NONZERO_ACTION_SCHEMA.items()
    }


_ENTRY_NO_FILL_NONZERO_ACTION_SCHEMA = _entry_action_schema_variant(
    {
        "first_fill_recv_time_ns": pl.Null,
        "first_fill_event_sequence": pl.Null,
        "first_fill_row_index": pl.Null,
        "full_fill_recv_time_ns": pl.Null,
        "full_fill_event_sequence": pl.Null,
        "full_fill_row_index": pl.Null,
        "fill_reason": pl.Null,
        "time_first_to_full_ms": pl.Null,
    }
)
_ENTRY_NO_FULL_FILL_NONZERO_ACTION_SCHEMA = _entry_action_schema_variant(
    {
        "full_fill_recv_time_ns": pl.Null,
        "full_fill_event_sequence": pl.Null,
        "full_fill_row_index": pl.Null,
        "time_first_to_full_ms": pl.Null,
    }
)
_ENTRY_INTEGER_TARGET_PRICE_NONZERO_ACTION_SCHEMA = _entry_action_schema_variant(
    {"target_price": pl.Int64}
)
_ENTRY_NO_EXECUTABLE_HEDGE_PRICE_NONZERO_ACTION_SCHEMA = (
    _entry_action_schema_variant(
        {
            "entry_hedge_decision_best_price": pl.Null,
            "entry_hedge_executable_vwap_price": pl.Null,
            "entry_hedge_partial_vwap_price": pl.Null,
            "entry_hedge_signed_latency_slippage_bp": pl.Null,
            "entry_hedge_signed_depth_slippage_bp": pl.Null,
            "entry_hedge_signed_total_slippage_bp": pl.Null,
        }
    )
)
_ENTRY_NONZERO_ACTION_SCHEMAS: tuple[Mapping[str, pl.DataType], ...] = (
    _ENTRY_CANONICAL_NONZERO_ACTION_SCHEMA,
    _ENTRY_NO_FILL_NONZERO_ACTION_SCHEMA,
    _ENTRY_NO_FULL_FILL_NONZERO_ACTION_SCHEMA,
    _ENTRY_INTEGER_TARGET_PRICE_NONZERO_ACTION_SCHEMA,
    _ENTRY_NO_EXECUTABLE_HEDGE_PRICE_NONZERO_ACTION_SCHEMA,
)
_EXPECTED_ENTRY_NONZERO_ACTION_SCHEMA_SHA256: frozenset[str] = frozenset(
    {
        "4ef33d6abee3e8d5915762a3e86122672a833f378e1c1cbbfdf2d0ff2cd9eb67",
        "d4462644570e6095fc171d71412f43a3b2dec461146bab0eb38f38bd1fa8dbcc",
        "5f85a509c38d7955f8dacdb1d85a19e88962c751aa6107cdf80bf90afff36c3e",
        "b93ab2e3aa7273d3b92ecf93c49198a50ca151d082658c0413778a24d73b675d",
        "e137b7c6e905e84c730eda45c2464dea6b9bb9d74706dd85470710cd8e5dee0c",
    }
)
_EXPECTED_ENTRY_ACTION_SCHEMA_INVENTORY_SHA256 = (
    "70891dd500743ce2fcf5d29ff7f8f7b8b1f36c4c4f53deb0e2cff3867502d13d"
)
_EXPECTED_ENTRY_RAW_ORDER_PARTITION_INVENTORY_SHA256 = (
    "d28fbc246482ed97628f24bd94b75d5ee3a06ced6184fdbb8eb0575b6ab1fb17"
)
_EXPECTED_ENTRY_RAW_ORDER_PARTITION_UNIQUE_SUM = 2_760_254


def _ordered_schema_sha256(schema: Mapping[str, pl.DataType]) -> str:
    payload = [(name, str(dtype)) for name, dtype in schema.items()]
    return hashlib.sha256(
        json.dumps(
            payload,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


if (
    frozenset(
        _ordered_schema_sha256(schema)
        for schema in _ENTRY_NONZERO_ACTION_SCHEMAS
    )
    != _EXPECTED_ENTRY_NONZERO_ACTION_SCHEMA_SHA256
):
    raise RuntimeError("frozen nonempty entry action schema constants changed")

_EXIT_MAKER_AUDIT_SCHEMA: dict[str, pl.DataType] = {
    "Date": pl.String,
    "ValueCode": pl.String,
    "QuoteCode": pl.String,
    "all_entry_aliases": pl.Int64,
    "all_entry_policy_aliases": pl.Int64,
    "entry_policy_aliases": pl.Int64,
    "expected_exit_rules": pl.Int64,
    "expected_exit_routes": pl.Int64,
    "expected_policy_trials": pl.Int64,
    "materialized_policy_trials": pl.Int64,
    "position_established_trials": pl.Int64,
    "no_admission_trials": pl.Int64,
    "unique_physical_entry_positions": pl.Int64,
    "physical_exit_policy_replays": pl.Int64,
    "raw_candidate_facts": pl.Int64,
    "candidate_alias_rows": pl.Int64,
    "oco_winner_trials": pl.Int64,
    "flat_same_day_trials": pl.Int64,
    "cancel_ack_observed_rows": pl.Int64,
    "cancel_race_modeled_rows": pl.Int64,
    "strict_ev_ready_rows": pl.Int64,
    "instant_cancel_v0": pl.Boolean,
    "d_minus_one_lineage_validated": pl.Boolean,
    "joint_volume_allocated": pl.Boolean,
}

_EXIT_MAKER_MANIFEST_SCHEMA: dict[str, pl.DataType] = {
    "Date": pl.String,
    "ValueCode": pl.String,
    "partition": pl.String,
    "runner_config_sha256": pl.String,
    "config_sha256": pl.String,
    "action_source_sha256": pl.String,
    "exit_rule_source_kind": pl.String,
    "exit_rule_source_sha256": pl.String,
    "complete": pl.Boolean,
    "exit_maker_audit_rows": pl.Int64,
    "exit_maker_candidate_aliases_rows": pl.Int64,
    "exit_maker_observations_rows": pl.Int64,
    "exit_maker_policy_support_rows": pl.Int64,
    "exit_maker_position_policy_facts_rows": pl.Int64,
    "exit_maker_raw_candidate_facts_rows": pl.Int64,
    "exit_maker_transitions_rows": pl.Int64,
}

_EXIT_ARTIFACT_ROW_COLUMNS: dict[str, str] = {
    "exit_maker_audit.parquet": "exit_maker_audit_rows",
    "exit_maker_candidate_aliases.parquet": (
        "exit_maker_candidate_aliases_rows"
    ),
    "exit_maker_observations.parquet": "exit_maker_observations_rows",
    "exit_maker_policy_support.parquet": "exit_maker_policy_support_rows",
    "exit_maker_position_policy_facts.parquet": (
        "exit_maker_position_policy_facts_rows"
    ),
    "exit_maker_raw_candidate_facts.parquet": (
        "exit_maker_raw_candidate_facts_rows"
    ),
    "exit_maker_transitions.parquet": "exit_maker_transitions_rows",
}

_ENTRY_TRANSITIVE_SOURCE_ARTIFACTS: tuple[str, ...] = (
    "execution_action_facts.parquet",
    "exit_facts.parquet",
    "target_audit.parquet",
)
_SAME_DAY_TRANSITIVE_SOURCE_ARTIFACTS: tuple[str, ...] = tuple(
    _EXIT_ARTIFACT_ROW_COLUMNS
)


@dataclass(frozen=True)
class CostSensitivity:
    """Completed-cycle cost assumption; never a full cost profile."""

    scenario_id: str
    same_day_bp: float
    overnight_bp: float

    def validate(self) -> None:
        if not self.scenario_id:
            raise ValueError("cost sensitivity scenario_id must be non-empty")
        if not all(
            character.isalnum() or character == "_"
            for character in self.scenario_id
        ):
            raise ValueError(
                "cost sensitivity scenario_id may contain only letters, digits, and _"
            )
        for name in ("same_day_bp", "overnight_bp"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


DEFAULT_COST_SENSITIVITIES: tuple[CostSensitivity, ...] = (
    CostSensitivity("gross_0bp", 0.0, 0.0),
    CostSensitivity("completed_flat_19bp", 19.0, 19.0),
    CostSensitivity("completed_same19_overnight34bp", 19.0, 34.0),
)


@dataclass(frozen=True)
class PositionLimit:
    """One independent fixed-policy capacity scenario."""

    limit_id: str
    max_concurrent_positions: int | None = None
    max_outstanding_notional_twd: float | None = None
    scope: str = "portfolio"

    def validate(self) -> None:
        if not self.limit_id:
            raise ValueError("position limit_id must be non-empty")
        if self.scope not in {"portfolio", "per_value_code"}:
            raise ValueError("position limit scope must be portfolio or per_value_code")
        if self.max_concurrent_positions is not None:
            value = self.max_concurrent_positions
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("max_concurrent_positions must be a positive integer")
        if self.max_outstanding_notional_twd is not None:
            value = float(self.max_outstanding_notional_twd)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    "max_outstanding_notional_twd must be finite and positive"
                )
        if (
            self.max_concurrent_positions is None
            and self.max_outstanding_notional_twd is None
        ):
            raise ValueError("a position-limit scenario must constrain something")


DEFAULT_POSITION_LIMITS: tuple[PositionLimit, ...] = (
    PositionLimit("positions_25", max_concurrent_positions=25),
    PositionLimit("positions_50", max_concurrent_positions=50),
    PositionLimit("positions_100", max_concurrent_positions=100),
    PositionLimit("positions_250", max_concurrent_positions=250),
    PositionLimit("notional_10m", max_outstanding_notional_twd=10_000_000.0),
    PositionLimit("notional_25m", max_outstanding_notional_twd=25_000_000.0),
    PositionLimit("notional_50m", max_outstanding_notional_twd=50_000_000.0),
    PositionLimit("notional_100m", max_outstanding_notional_twd=100_000_000.0),
    PositionLimit(
        "per_product_positions_1",
        max_concurrent_positions=1,
        scope="per_value_code",
    ),
    PositionLimit(
        "per_product_positions_2",
        max_concurrent_positions=2,
        scope="per_value_code",
    ),
    PositionLimit(
        "per_product_positions_5",
        max_concurrent_positions=5,
        scope="per_value_code",
    ),
    PositionLimit(
        "per_product_notional_10m",
        max_outstanding_notional_twd=10_000_000.0,
        scope="per_value_code",
    ),
    PositionLimit(
        "per_product_notional_25m",
        max_outstanding_notional_twd=25_000_000.0,
        scope="per_value_code",
    ),
)


@dataclass(frozen=True)
class PostCrossEvaluationConfig:
    """Prequential support and analysis-only risk-control contract."""

    lookback_sessions: int = 60
    min_training_dates: int = 20
    min_policy_origins: int = 100
    min_completed_cycles_for_diagnostic_rank: int = 30
    confidence_level: float = 0.95
    minimum_lcb_bp: float = 0.0
    expected_product_days: int = 2_687

    def validate(self) -> None:
        for name in (
            "lookback_sessions",
            "min_training_dates",
            "min_policy_origins",
            "min_completed_cycles_for_diagnostic_rank",
            "expected_product_days",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_training_dates > self.lookback_sessions:
            raise ValueError("min_training_dates cannot exceed lookback_sessions")
        if not 0.5 < float(self.confidence_level) < 1.0:
            raise ValueError("confidence_level must be in (0.5, 1)")
        if not math.isfinite(float(self.minimum_lcb_bp)):
            raise ValueError("minimum_lcb_bp must be finite")


@dataclass(frozen=True)
class FormalPostCrossSources:
    """Fully verified nominal terminal paths and their source identity."""

    report: FilledEntryPrimaryReport
    policy_paths: pl.DataFrame
    session_calendar: tuple[str, ...]
    entry_sessions: tuple[str, ...]
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class PostCrossPositionEvaluation:
    """Tables published by the post-cross evaluator."""

    policy_paths: pl.DataFrame
    policy_summary: pl.DataFrame
    product_policy_summary: pl.DataFrame
    daily_terminal_cashflows: pl.DataFrame
    daily_outstanding: pl.DataFrame
    prequential_rankings: pl.DataFrame
    prequential_decisions: pl.DataFrame
    position_limit_sweep: pl.DataFrame
    readiness: pl.DataFrame
    metadata: Mapping[str, object]


def _formal_exit_artifact_schemas() -> dict[str, Mapping[str, pl.DataType]]:
    """Return the producer-owned schemas without materialising any artifact."""

    return {
        "exit_maker_policy_support.parquet": _exit_study._support_schema(),
        "exit_maker_observations.parquet": _exit_study._observation_schema(),
        "exit_maker_transitions.parquet": _exit_study._transition_schema(),
        "exit_maker_candidate_aliases.parquet": (
            _exit_study._candidate_alias_schema()
        ),
        "exit_maker_raw_candidate_facts.parquet": (
            _exit_study._candidate_schema()
        ),
        "exit_maker_position_policy_facts.parquet": (
            _exit_study._position_policy_schema()
        ),
        "exit_maker_audit.parquet": _EXIT_MAKER_AUDIT_SCHEMA,
    }


def _validate_exact_exit_artifact_schemas(partition: Path) -> None:
    for name, expected in _formal_exit_artifact_schemas().items():
        path = partition / name
        _require_regular_nonsymlink_file(path, "formal exit-maker artifact")
        actual = pl.read_parquet_schema(path)
        if list(actual.items()) != list(expected.items()):
            raise ValueError(
                f"formal exit-maker artifact schema mismatch: {path}"
            )


def _require_regular_nonsymlink_file(path: Path, source: str) -> None:
    if path.is_symlink():
        raise ValueError(f"{source} must not be a symlink: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)


def _discard_clean_file_cache(path: Path, *, source: str) -> None:
    """Release clean formal-source pages after their bytes were validated.

    The formal Linux run reads roughly 15 GiB solely to recompute immutable
    SHA-256 declarations.  Those pages are not inputs after a partition's
    projection is materialised, but cgroup-v2 otherwise keeps them charged to
    the evaluator alongside the report heap.  This is an advisory cache drop;
    it never changes bytes or bypasses the frozen validators that ran first.
    The formal runner fails closed if the required Linux primitive is absent
    or fails, because continuing would violate its 8 GiB soft-memory gate.
    """

    _require_regular_nonsymlink_file(path, source)
    if not hasattr(os, "posix_fadvise") or not hasattr(
        os, "POSIX_FADV_DONTNEED"
    ):
        raise RuntimeError(
            "formal post-cross cache control requires POSIX_FADV_DONTNEED"
        )
    try:
        with path.open("rb", buffering=0) as stream:
            os.posix_fadvise(  # type: ignore[attr-defined]
                stream.fileno(),
                0,
                0,
                os.POSIX_FADV_DONTNEED,  # type: ignore[attr-defined]
            )
    except OSError as error:
        raise RuntimeError(
            f"failed to discard validated formal-source cache: {path}"
        ) from error


def _discard_partition_artifact_cache(
    partition: Path,
    artifact_names: Iterable[str],
    *,
    source: str,
) -> int:
    count = 0
    for name in artifact_names:
        if Path(name).name != name:
            raise ValueError(f"invalid {source} artifact name: {name}")
        _discard_clean_file_cache(partition / name, source=source)
        count += 1
    return count


def _emit_memory_phase(phase: str, **details: object) -> None:
    """Emit a bounded diagnostic snapshot without changing bundle semantics."""

    payload: dict[str, object] = {
        "phase": phase,
        "pid": os.getpid(),
        **details,
    }
    try:
        status: dict[str, str] = {}
        for line in Path("/proc/self/status").read_text().splitlines():
            if ":" in line:
                name, value = line.split(":", 1)
                if name in {"VmRSS", "VmHWM"}:
                    status[name] = value.strip()
        payload.update(
            {
                "process_rss": status.get("VmRSS"),
                "process_hwm": status.get("VmHWM"),
            }
        )
        cgroup_relative = next(
            line.split("::", 1)[1]
            for line in Path("/proc/self/cgroup").read_text().splitlines()
            if "::" in line
        )
        cgroup = Path("/sys/fs/cgroup") / cgroup_relative.lstrip("/")
        for name in ("memory.current", "memory.peak"):
            path = cgroup / name
            if path.is_file():
                payload[name.replace(".", "_")] = int(path.read_text().strip())
        events_path = cgroup / "memory.events"
        if events_path.is_file():
            payload["memory_events"] = {
                name: int(value)
                for name, value in (
                    line.split() for line in events_path.read_text().splitlines()
                )
            }
    except (OSError, StopIteration, ValueError) as error:
        payload["diagnostic_error"] = type(error).__name__
    print(
        json.dumps(
            {"post_cross_memory_phase": payload},
            sort_keys=True,
            separators=(",", ":"),
        ),
        file=sys.stderr,
        flush=True,
    )


def _canonical_product_day_partition(
    root: Path,
    record: Mapping[str, object],
) -> Path:
    """Return the lexical canonical partition and reject symlink escapes."""

    date = str(record["Date"])
    value_code = str(record["ValueCode"])
    root = Path(root)
    if root.is_symlink():
        raise ValueError(f"formal source root must not be a symlink: {root}")
    root_resolved = root.resolve(strict=True)
    expected = root_resolved / f"Date={date}" / f"ValueCode={value_code}"
    actual = Path(record["partition"])
    if actual.absolute() != expected:
        raise ValueError(
            "formal product-day partition is not the canonical root/Date/Value path: "
            f"{date}/{value_code}"
        )
    if expected.parent.is_symlink() or expected.is_symlink():
        raise ValueError(f"formal product-day partition must not be a symlink: {expected}")
    if not expected.is_dir() or expected.resolve(strict=True) != expected:
        raise ValueError(
            f"formal product-day partition escapes its canonical root: {expected}"
        )
    marker = expected / "complete.json"
    _require_regular_nonsymlink_file(marker, "formal product-day marker")
    record_marker = record.get("marker")
    if record_marker is not None and Path(record_marker).absolute() != marker:
        raise ValueError(
            f"formal product-day marker is not at its canonical path: {marker}"
        )
    return expected


def _read_projected_parquet(
    path: Path,
    schema: Mapping[str, pl.DataType],
    *,
    declared_rows: int,
    source: str,
    allowed_zero_schemas: Sequence[Mapping[str, pl.DataType]] = (),
    allowed_nonzero_schemas: Sequence[Mapping[str, pl.DataType]] = (),
    allowed_null_projection_casts: Mapping[str, pl.DataType] | None = None,
) -> pl.DataFrame:
    """Read only a bounded projection; typed narrow empties remain legal."""

    _require_regular_nonsymlink_file(path, source)
    actual = pl.read_parquet_schema(path)
    if declared_rows == 0:
        exact_allowed_zero = any(
            list(actual.items()) == list(allowed.items())
            for allowed in allowed_zero_schemas
        )
        if not exact_allowed_zero:
            raise ValueError(
                f"{source} zero-row producer schema mismatch"
            )
        return pl.DataFrame(schema=schema)

    if allowed_nonzero_schemas:
        exact_allowed_nonzero = any(
            list(actual.items()) == list(allowed.items())
            for allowed in allowed_nonzero_schemas
        )
        if not exact_allowed_nonzero:
            raise ValueError(f"{source} nonzero producer schema mismatch")

    normalized_nulls = {
        name
        for name, dtype in (allowed_null_projection_casts or {}).items()
        if schema.get(name) == dtype and actual.get(name) == pl.Null
    }
    bad = [
        name
        for name, dtype in schema.items()
        if actual.get(name) != dtype and name not in normalized_nulls
    ]
    if bad:
        raise ValueError(f"{source} projected schema mismatch: {bad}")
    projected = pl.read_parquet(path, columns=list(schema))
    if normalized_nulls:
        projected = projected.with_columns(
            *(
                pl.col(name).cast(schema[name], strict=True)
                for name in sorted(normalized_nulls)
            )
        )
    return projected


def _validate_partition_identity(
    frame: pl.DataFrame,
    *,
    date: str,
    value_code: str,
    source: str,
) -> str | None:
    if frame.is_empty():
        return None
    if set(frame["Date"].cast(pl.String).unique().to_list()) != {date}:
        raise ValueError(f"{source} Date disagrees with its partition")
    if set(frame["ValueCode"].cast(pl.String).unique().to_list()) != {
        value_code
    }:
        raise ValueError(f"{source} ValueCode disagrees with its partition")
    quote_codes = frame["QuoteCode"].cast(pl.String).drop_nulls().unique().to_list()
    if len(quote_codes) != 1:
        raise ValueError(f"{source} must contain one non-null QuoteCode")
    return str(quote_codes[0])


_EXECUTION_DIAGNOSTIC_COUNT_COLUMNS: tuple[str, ...] = (
    "submitted_entry_policy_aliases",
    "submitted_physical_raw_orders",
    "known_no_fill_policy_aliases",
    "entry_fill_unknown_policy_aliases",
    "entry_partial_fill_policy_aliases",
    "entry_full_fill_policy_aliases",
    "entry_full_fill_hedge_executable_policy_aliases",
    "entry_full_fill_hedge_not_executable_policy_aliases",
    "entry_cancel_request_policy_aliases",
)


def _record_partition_raw_order_inventory(
    actions: pl.DataFrame,
    *,
    partition: str,
    seen_raw_order_fact_ids: set[str] | None,
) -> dict[str, object]:
    """Bind the raw-order namespace needed for additive diagnostics.

    The frozen full-action validator groups physical hedge facts globally by
    ``raw_order_fact_id`` and its diagnostic table computes a per-cell
    ``n_unique``.  Per-partition validation/aggregation is exactly equivalent
    only when raw IDs never cross a product-day boundary.  Small/subset loads
    prove that directly with ``seen_raw_order_fact_ids``.  The 60-day formal
    load instead compares the complete partitioned inventory to an immutable
    audited digest, avoiding a multi-million-element Python set in the capped
    process.
    """

    values = actions["raw_order_fact_id"].unique().sort().to_list()
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("entry raw_order_fact_id inventory is not canonical")
    raw_ids = tuple(str(value) for value in values)
    if seen_raw_order_fact_ids is not None:
        duplicate = seen_raw_order_fact_ids.intersection(raw_ids)
        if duplicate:
            raise ValueError(
                "entry raw_order_fact_id crosses a product-day partition"
            )
        seen_raw_order_fact_ids.update(raw_ids)
    return {
        "partition": partition,
        "rows": actions.height,
        "unique_raw_order_fact_ids": len(raw_ids),
        "raw_order_fact_ids_sha256": _canonical_sha256(raw_ids),
    }


def _merge_partition_execution_diagnostics(
    frames: Sequence[pl.DataFrame],
    *,
    expected_action_rows: int,
    expected_established_rows: int,
    raw_order_ids_partition_disjoint: bool,
) -> pl.DataFrame:
    """Merge additive per-partition diagnostics without retaining 4m actions."""

    if raw_order_ids_partition_disjoint is not True:
        raise ValueError(
            "execution diagnostics require partition-disjoint raw order IDs"
        )

    if not frames:
        if expected_action_rows != 0 or expected_established_rows != 0:
            raise ValueError("formal execution diagnostics lost nonempty actions")
        return _filled_report._build_execution_diagnostics(
            pl.DataFrame(schema=_NARROW_ACTION_SCHEMA)
        )
    key = list(_filled_report.EXECUTION_DIAGNOSTIC_KEY)
    merged = (
        pl.concat(frames, how="vertical", rechunk=False)
        .group_by(key)
        .agg(
            *(
                pl.col(name).sum().alias(name)
                for name in _EXECUTION_DIAGNOSTIC_COUNT_COLUMNS
            )
        )
        .with_columns(
            (
                pl.col("entry_cancel_request_policy_aliases")
                / pl.col("submitted_entry_policy_aliases")
            ).alias("entry_cancel_request_rate"),
            (
                pl.col("entry_full_fill_hedge_executable_policy_aliases")
                / pl.col("submitted_entry_policy_aliases")
            ).alias("full_hedged_rate_per_submitted_alias"),
            pl.lit("execution_diagnostic_only_not_profit_denominator").alias(
                "table_semantics"
            ),
            pl.lit(True).alias("cancel_rate_is_request_only"),
            pl.lit(False).alias("boundary_quantile_rows_safe_to_sum"),
        )
        .select(frames[0].columns)
        .sort(key)
    )
    if (
        int(merged["submitted_entry_policy_aliases"].sum())
        != expected_action_rows
        or int(
            merged[
                "entry_full_fill_hedge_executable_policy_aliases"
            ].sum()
        )
        != expected_established_rows
        or merged.select(key).n_unique() != merged.height
    ):
        raise ValueError("preaggregated execution diagnostics do not conserve actions")
    mutually_exclusive_outcomes = sum(
        int(merged[name].sum())
        for name in (
            "known_no_fill_policy_aliases",
            "entry_fill_unknown_policy_aliases",
            "entry_partial_fill_policy_aliases",
            "entry_full_fill_policy_aliases",
        )
    )
    full_accounting = int(
        merged["entry_full_fill_hedge_executable_policy_aliases"].sum()
    ) + int(
        merged["entry_full_fill_hedge_not_executable_policy_aliases"].sum()
    )
    if (
        mutually_exclusive_outcomes != expected_action_rows
        or full_accounting
        != int(merged["entry_full_fill_policy_aliases"].sum())
    ):
        raise ValueError(
            "preaggregated execution diagnostics violate outcome accounting"
        )
    return merged


def _load_narrow_entry_sources(
    partition: Path,
    exit_source: Mapping[str, object],
    *,
    date: str,
    value_code: str,
) -> tuple[
    pl.DataFrame,
    pl.DataFrame,
    str,
    Mapping[str, object],
    Mapping[str, object],
]:
    if partition.parent.is_symlink() or partition.is_symlink():
        raise ValueError(
            f"entry source partition must not be a symlink: {partition}"
        )
    if not partition.is_dir() or partition.resolve(strict=True) != partition.absolute():
        raise ValueError(
            f"entry source partition escapes its canonical root: {partition}"
        )
    marker = partition / "complete.json"
    _require_regular_nonsymlink_file(marker, "entry source marker")
    payload = _exit_report._read_json(marker)
    if payload.get("complete") is not True:
        raise ValueError(f"entry source marker is incomplete: {marker}")
    if str(payload.get("Date")) != date or str(payload.get("ValueCode")) != value_code:
        raise ValueError(f"entry source marker identity mismatch: {marker}")
    config = payload.get("config")
    config_sha = payload.get("config_sha256")
    if not isinstance(config, dict) or not isinstance(config_sha, str):
        raise ValueError(f"entry source config is invalid: {marker}")
    if _exit_report._canonical_sha256(config) != config_sha:
        raise ValueError(f"entry source config hash mismatch: {marker}")
    if str(exit_source.get("upstream_config_sha256")) != config_sha:
        raise ValueError(f"exit-maker source binds a different entry config: {marker}")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"entry source artifact manifest is invalid: {marker}")
    target_declaration = artifacts.get("target_audit.parquet")
    if not isinstance(target_declaration, dict):
        raise ValueError(f"entry source marker lacks target_audit.parquet: {marker}")
    target_path = partition / "target_audit.parquet"
    _require_regular_nonsymlink_file(target_path, "entry source artifact")
    _exit_report._validate_one_artifact(
        target_path, target_declaration, validate_hashes=True
    )

    frames: list[pl.DataFrame] = []
    action_schema_record: dict[str, object] | None = None
    for source_name, artifact_name, schema, zero_schema in (
        (
            "action_source",
            "execution_action_facts.parquet",
            _NARROW_ACTION_SCHEMA,
            _ENTRY_ZERO_ACTION_SCHEMA,
        ),
        (
            "exit_rule_source",
            "exit_facts.parquet",
            _NARROW_EXIT_RULE_SCHEMA,
            _ENTRY_ZERO_EXIT_RULE_SCHEMA,
        ),
    ):
        declared_source = exit_source.get(source_name)
        declared = artifacts.get(artifact_name)
        if not isinstance(declared_source, dict) or not isinstance(declared, dict):
            raise ValueError(f"entry source binding is incomplete: {marker}")
        if declared_source.get("artifact") != artifact_name:
            raise ValueError(
                f"unexpected bound entry artifact for {source_name}: "
                f"{declared_source.get('artifact')}"
            )
        if str(declared_source.get("sha256")) != str(declared.get("sha256")):
            raise ValueError(
                f"exit-maker source hash disagrees with entry marker: {artifact_name}"
            )
        path = partition / artifact_name
        _require_regular_nonsymlink_file(path, "entry source artifact")
        _exit_report._validate_one_artifact(path, declared, validate_hashes=True)
        rows = declared.get("rows")
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
            raise ValueError(f"entry artifact row declaration is invalid: {path}")
        frames.append(
            _read_projected_parquet(
                path,
                schema,
                declared_rows=rows,
                source=f"entry {artifact_name}",
                allowed_zero_schemas=(zero_schema,),
                allowed_nonzero_schemas=(
                    _ENTRY_NONZERO_ACTION_SCHEMAS
                    if artifact_name == "execution_action_facts.parquet"
                    else ()
                ),
                allowed_null_projection_casts=(
                    {"full_fill_recv_time_ns": pl.Int64}
                    if artifact_name == "execution_action_facts.parquet"
                    else {}
                ),
            )
        )
        if artifact_name == "execution_action_facts.parquet":
            actual_schema = pl.read_parquet_schema(path)
            action_schema_record = {
                "partition": f"Date={date}/ValueCode={value_code}",
                "rows": rows,
                "schema_sha256": _ordered_schema_sha256(actual_schema),
                "full_fill_recv_time_ns": str(
                    actual_schema.get("full_fill_recv_time_ns")
                ),
            }

    actions = _exit_report._validate_entry_action_execution_contract(
        frames[0], expected_hedge_delay_ns=_exit_report.EXPECTED_HEDGE_DELAY_NS
    ).select(list(_NARROW_ACTION_SCHEMA))
    exit_rules = frames[1]
    action_quote = _validate_partition_identity(
        actions, date=date, value_code=value_code, source="entry actions"
    )
    exit_quote = _validate_partition_identity(
        exit_rules, date=date, value_code=value_code, source="entry exit facts"
    )
    if action_quote is not None and exit_quote is not None and action_quote != exit_quote:
        raise ValueError("entry action and exit-rule QuoteCode disagree")
    if not actions.is_empty():
        if actions.select("policy_generation_id").n_unique() != actions.height:
            raise ValueError(
                "entry actions must be unique within their product-day partition"
            )
        invalid_universe = actions.filter(
            pl.col("policy_generation_id").is_null()
            | ~pl.col("policy_generation_id").str.starts_with(
                f"{date}/{value_code}/"
            )
            | ~pl.col("route").is_in(FROZEN_ENTRY_ROUTES)
            | ~pl.col("boundary_quantile").is_in(
                FROZEN_BOUNDARY_QUANTILES
            )
        )
        if invalid_universe.height:
            raise ValueError(
                "entry action partition has noncanonical identity, route, or q"
            )
    if action_schema_record is None:
        raise RuntimeError("entry action schema inventory was not recorded")
    return actions, exit_rules, config_sha, config, action_schema_record


def _current_partition_source_identity(
    partition: Path,
    payload: Mapping[str, object],
    artifact_names: Sequence[str],
    *,
    source: str,
) -> dict[str, object]:
    marker = partition / "complete.json"
    _require_regular_nonsymlink_file(marker, f"{source} marker")
    artifacts = payload.get("artifacts")
    config_sha = payload.get("config_sha256")
    runner_version = payload.get("runner_version")
    if (
        not isinstance(artifacts, Mapping)
        or not isinstance(config_sha, str)
        or not isinstance(runner_version, str)
        or not runner_version
    ):
        raise ValueError(f"{source} marker source identity is incomplete")
    artifact_hashes: dict[str, str] = {}
    for name in artifact_names:
        declaration = artifacts.get(name)
        if not isinstance(declaration, Mapping):
            raise ValueError(f"{source} marker lacks transitive artifact {name}")
        artifact_hashes[name] = _validate_sha256(
            declaration.get("sha256"), f"{source}/{name} sha256"
        )
    return {
        "partition": str(partition),
        "marker_sha256": _file_sha256(marker),
        "config_sha256": _validate_sha256(config_sha, f"{source} config sha256"),
        "runner_version": runner_version,
        "artifacts": artifact_hashes,
    }


def _current_formal_source_binding_record(
    *,
    date: str,
    value_code: str,
    entry_partition: Path,
    same_day_partition: Path,
    same_day_payload: Mapping[str, object],
) -> dict[str, object]:
    entry_payload = _exit_report._read_json(entry_partition / "complete.json")
    return {
        "Date": date,
        "ValueCode": value_code,
        "entry_execution": _current_partition_source_identity(
            entry_partition,
            entry_payload,
            _ENTRY_TRANSITIVE_SOURCE_ARTIFACTS,
            source="entry execution",
        ),
        "same_day_exit_maker": _current_partition_source_identity(
            same_day_partition,
            same_day_payload,
            _SAME_DAY_TRANSITIVE_SOURCE_ARTIFACTS,
            source="same-day exit-maker",
        ),
    }


def _validate_and_bind_position_exit_rules(
    positions: pl.DataFrame,
    exit_rules: pl.DataFrame,
    *,
    expected_predecessor: str,
) -> pl.DataFrame:
    """Validate all projected rules, then retain exactly the position FK set."""

    exit_key = [
        "Date",
        "ValueCode",
        "QuoteCode",
        "route",
        "policy_generation_id",
        "raw_order_fact_id",
        "exit_rule_id",
    ]
    for frame, label in (
        (positions, "same-day position"),
        (exit_rules, "bound entry exit fact"),
    ):
        if frame.is_empty():
            continue
        if frame.filter(
            pl.col("exit_rule_source_asof_date").is_null()
            | (pl.col("exit_rule_source_asof_date") != expected_predecessor)
            | pl.col("exit_threshold_basis_bp").is_null()
            | ~pl.col("exit_threshold_basis_bp").is_finite()
        ).height:
            raise ValueError(
                f"{label} threshold/source is not finite and exact-D-1"
            )
    if exit_rules.select(exit_key).n_unique() != exit_rules.height:
        raise ValueError("bound entry exit facts duplicate a frozen rule key")

    required = positions.select(
        "Date",
        "ValueCode",
        "QuoteCode",
        pl.col("entry_route").alias("route"),
        pl.col("entry_policy_generation_id").alias("policy_generation_id"),
        pl.col("entry_raw_order_fact_id").alias("raw_order_fact_id"),
        "exit_rule_id",
        pl.col("exit_threshold_basis_bp").alias(
            "_position_exit_threshold_basis_bp"
        ),
        pl.col("exit_rule_source_asof_date").alias(
            "_position_exit_rule_source_asof_date"
        ),
    ).unique()
    if required.select(exit_key).n_unique() != required.height:
        raise ValueError("position policy facts disagree on one entry exit-rule key")
    bound = required.join(
        exit_rules,
        on=exit_key,
        how="left",
        validate="1:1",
    )
    if bound.height != required.height or bound.filter(
        pl.col("exit_threshold_basis_bp").is_null()
        | (
            pl.col("exit_threshold_basis_bp")
            != pl.col("_position_exit_threshold_basis_bp")
        ).fill_null(True)
        | (
            pl.col("exit_rule_source_asof_date")
            != pl.col("_position_exit_rule_source_asof_date")
        ).fill_null(True)
    ).height:
        raise ValueError(
            "same-day position identity/threshold/source differs from bound entry exit_facts"
        )
    result = bound.select(list(_NARROW_EXIT_RULE_SCHEMA))
    if result.select(exit_key).n_unique() != result.height:
        raise AssertionError("narrow bound exit-rule projection contains extras")
    return result


def _validate_exact_exit_root_manifest(
    root: Path,
    selected: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    root = Path(root)
    path = root / "exit_maker_partition_manifest.parquet"
    _require_regular_nonsymlink_file(path, "formal exit-maker root manifest")
    metadata = _exit_report._validate_root_manifest(
        root, selected, validate_hashes=True, required=True
    )
    if list(pl.read_parquet_schema(path).items()) != list(
        _EXIT_MAKER_MANIFEST_SCHEMA.items()
    ):
        raise ValueError("formal exit-maker root manifest schema mismatch")
    manifest = pl.read_parquet(path)
    if not manifest.equals(manifest.sort(["Date", "ValueCode"]), null_equal=True):
        raise ValueError("formal exit-maker root manifest is not canonically sorted")
    expected_keys = {
        (str(row["Date"]), str(row["ValueCode"])) for row in selected
    }
    actual_keys = {
        (str(row["Date"]), str(row["ValueCode"]))
        for row in manifest.iter_rows(named=True)
    }
    if actual_keys != expected_keys or manifest.height != len(selected):
        raise ValueError("formal exit-maker root manifest inventory is not exact")
    lookup = {
        (str(row["Date"]), str(row["ValueCode"])): row
        for row in manifest.iter_rows(named=True)
    }
    for record in selected:
        key = (str(record["Date"]), str(record["ValueCode"]))
        row = lookup[key]
        payload = record["payload"]
        partition = _canonical_product_day_partition(root, record)
        if not isinstance(payload, dict) or row["complete"] is not True:
            raise ValueError(f"invalid complete manifest row: {key}")
        config = payload.get("config")
        artifacts = payload.get("artifacts")
        if not isinstance(config, dict) or not isinstance(artifacts, dict):
            raise ValueError(f"invalid exit-maker marker payload: {key}")
        source = config.get("source")
        if not isinstance(source, dict):
            raise ValueError(f"exit-maker marker lacks source lineage: {key}")
        action_source = source.get("action_source")
        exit_source = source.get("exit_rule_source")
        if not isinstance(action_source, dict) or not isinstance(exit_source, dict):
            raise ValueError(f"exit-maker marker source lineage is incomplete: {key}")
        manifest_partition = row["partition"]
        expected_partition_paths = {partition.resolve().as_posix()}
        try:
            expected_partition_paths.add(
                partition.resolve().relative_to(Path.cwd().resolve()).as_posix()
            )
        except ValueError:
            pass
        if (
            not isinstance(manifest_partition, str)
            or Path(manifest_partition).as_posix() not in expected_partition_paths
        ):
            raise ValueError(
                f"formal exit-maker manifest disagrees on partition: {key}"
            )
        expected_scalars = {
            "runner_config_sha256": payload.get("runner_config_sha256"),
            "config_sha256": payload.get("config_sha256"),
            "action_source_sha256": action_source.get("sha256"),
            "exit_rule_source_kind": exit_source.get("kind"),
            "exit_rule_source_sha256": exit_source.get("sha256"),
        }
        for column, expected in expected_scalars.items():
            if row[column] != expected:
                raise ValueError(
                    f"formal exit-maker manifest disagrees on {column}: {key}"
                )
        for artifact, column in _EXIT_ARTIFACT_ROW_COLUMNS.items():
            declaration = artifacts.get(artifact)
            if not isinstance(declaration, dict) or row[column] != declaration.get(
                "rows"
            ):
                raise ValueError(
                    f"formal exit-maker manifest row count mismatch: {key}/{artifact}"
                )
    return {
        **metadata,
        "same_day_root_manifest_exact_inventory": True,
        "same_day_root_manifest_exact_schema": True,
    }


def _load_narrow_formal_partition_inputs(
    exit_maker_root: Path,
    entry_execution_root: Path,
    *,
    sessions: int,
    value_codes: Iterable[str] | None,
    session_calendar: Sequence[str],
) -> ExitMakerPartitionInputs:
    """Hash-validate every formal artifact while materialising only report inputs."""

    raw_root = Path(exit_maker_root)
    raw_entry_root = Path(entry_execution_root)
    if raw_root.is_symlink() or raw_entry_root.is_symlink():
        raise ValueError("formal same-day and entry roots must not be symlinks")
    root = raw_root.resolve(strict=True)
    entry_root = raw_entry_root.resolve(strict=True)
    _exit_report._validate_session_request(sessions)
    requested_products = _exit_report._normalise_products(value_codes)
    records = _exit_report._discover_markers(root, requested_products)
    if not records:
        raise FileNotFoundError(f"no complete exit-maker partitions under {root}")
    available_dates = sorted({str(record["Date"]) for record in records})
    if len(available_dates) < sessions:
        raise ValueError(
            f"requested {sessions} complete sessions, found {len(available_dates)}"
        )
    selected_dates = available_dates[-sessions:]
    selected = [row for row in records if str(row["Date"]) in selected_dates]
    products = list(
        requested_products
        if requested_products is not None
        else sorted({str(record["ValueCode"]) for record in selected})
    )
    selected_by_key = {
        (str(row["Date"]), str(row["ValueCode"])): row for row in selected
    }
    if len(selected_by_key) != len(selected):
        raise ValueError("duplicate complete exit-maker marker for a product-day")
    coverage = pl.from_dicts(
        [
            {
                "Date": date,
                "ValueCode": value_code,
                "partition_complete": (date, value_code) in selected_by_key,
                "partition": (
                    str(selected_by_key[(date, value_code)]["partition"])
                    if (date, value_code) in selected_by_key
                    else None
                ),
            }
            for date in selected_dates
            for value_code in products
        ],
        infer_schema_length=None,
    ).sort(["Date", "ValueCode"])
    missing = coverage.filter(~pl.col("partition_complete"))
    full_frozen_action_inventory = (
        len(selected) == 2_687
        and len(selected_dates) == 60
        and len(products) == 45
    )
    predecessors = _exit_report._expected_session_predecessors(
        session_calendar, selected_dates
    )
    if predecessors is None:
        raise ValueError("formal narrow loader requires an exact session calendar")

    action_frames: list[pl.DataFrame] = []
    execution_diagnostic_frames: list[pl.DataFrame] = []
    position_frames: list[pl.DataFrame] = []
    bound_exit_frames: list[pl.DataFrame] = []
    runner_hashes: set[str] = set()
    runner_payloads: set[str] = set()
    runner_versions: set[str] = set()
    entry_config_hashes: set[str] = set()
    entry_hedge_delay_values: set[int] = set()
    exit_rule_id_sets: set[tuple[str, ...]] = set()
    exit_route_sets: set[tuple[str, ...]] = set()
    taker_exit_grid_values: set[int] = set()
    taker_exit_max_book_age_values: set[int] = set()
    full_hedged_rows = 0
    validated_artifacts = 0
    source_binding_records: list[dict[str, object]] = []
    entry_action_schema_records: list[Mapping[str, object]] = []
    entry_raw_order_inventory_records: list[Mapping[str, object]] = []
    subset_seen_raw_order_fact_ids: set[str] | None = (
        None if full_frozen_action_inventory else set()
    )
    discarded_source_cache_files = 0
    all_action_rows = 0
    established_action_rows = 0

    for partition_index, record in enumerate(selected, start=1):
        payload = record["payload"]
        if not isinstance(payload, dict):
            raise ValueError("exit-maker marker payload must be an object")
        partition = _canonical_product_day_partition(root, record)
        config = _exit_report._validate_exit_marker(
            payload, partition, validate_hashes=True
        )
        _validate_exact_exit_artifact_schemas(partition)
        validated_artifacts += len(_formal_exit_artifact_schemas())
        runner = config["runner"]
        source = config["source"]
        assert isinstance(runner, dict) and isinstance(source, dict)
        runner_hashes.add(str(payload["runner_config_sha256"]))
        runner_payloads.add(_exit_report._canonical_json(runner))
        runner_versions.add(str(payload.get("runner_version")))
        hedge_delay = runner.get("hedge_delay_ns")
        if hedge_delay != _exit_report.EXPECTED_HEDGE_DELAY_NS:
            raise ValueError("formal exit-maker hedge delay is not exactly 50 ms")
        exit_rule_id_sets.add(
            tuple(sorted(str(value) for value in runner.get("exit_rule_ids", ())))
        )
        exit_route_sets.add(
            tuple(sorted(str(value) for value in runner.get("routes", ())))
        )

        artifacts = payload.get("artifacts")
        assert isinstance(artifacts, dict)
        position_declaration = artifacts[
            "exit_maker_position_policy_facts.parquet"
        ]
        assert isinstance(position_declaration, dict)
        position_rows = position_declaration.get("rows")
        if (
            isinstance(position_rows, bool)
            or not isinstance(position_rows, int)
            or position_rows < 0
        ):
            raise ValueError("position artifact row declaration is invalid")
        positions = _read_projected_parquet(
            partition / "exit_maker_position_policy_facts.parquet",
            _NARROW_POSITION_SCHEMA,
            declared_rows=position_rows,
            source="same-day position policy facts",
            allowed_zero_schemas=(_exit_study._position_policy_schema(),),
        )
        date = str(record["Date"])
        value_code = str(record["ValueCode"])
        position_quote = _validate_partition_identity(
            positions,
            date=date,
            value_code=value_code,
            source="same-day position policy facts",
        )

        (
            actions,
            exit_rules,
            entry_config_hash,
            entry_config,
            action_schema_record,
        ) = (
            _load_narrow_entry_sources(
                entry_root / f"Date={date}" / f"ValueCode={value_code}",
                source,
                date=date,
                value_code=value_code,
            )
        )
        entry_action_schema_records.append(action_schema_record)
        action_quote = (
            None
            if actions.is_empty()
            else str(actions.item(0, "QuoteCode"))
        )
        if position_quote is not None and action_quote is not None and (
            position_quote != action_quote
        ):
            raise ValueError("same-day position and entry action QuoteCode disagree")
        bound_exit = _validate_and_bind_position_exit_rules(
            positions,
            exit_rules,
            expected_predecessor=predecessors[date],
        )
        source_binding_records.append(
            _current_formal_source_binding_record(
                date=date,
                value_code=value_code,
                entry_partition=(
                    entry_root / f"Date={date}" / f"ValueCode={value_code}"
                ),
                same_day_partition=partition,
                same_day_payload=payload,
            )
        )

        all_action_rows += actions.height
        entry_raw_order_inventory_records.append(
            _record_partition_raw_order_inventory(
                actions,
                partition=f"Date={date}/ValueCode={value_code}",
                seen_raw_order_fact_ids=subset_seen_raw_order_fact_ids,
            )
        )
        if not actions.is_empty():
            execution_diagnostic_frames.append(
                _filled_report._build_execution_diagnostics(actions)
            )
        established_actions = actions.filter(
            _filled_report._established_entry_expr()
        )
        established_action_rows += established_actions.height
        action_frames.append(established_actions)
        position_frames.append(positions)
        bound_exit_frames.append(bound_exit)
        full_hedged_rows += actions.filter(pl.col("full_fill") == True).height  # noqa: E712
        entry_config_hashes.add(entry_config_hash)
        entry_hedge_delay = entry_config.get("hedge_delay_ns")
        if entry_hedge_delay != _exit_report.EXPECTED_HEDGE_DELAY_NS:
            raise ValueError("entry action marker hedge delay is not exactly 50 ms")
        entry_hedge_delay_values.add(int(entry_hedge_delay))
        exit_path = entry_config.get("exit_path")
        if not isinstance(exit_path, dict):
            raise ValueError("entry runner config is missing the T/T exit_path")
        grid_ns = exit_path.get("grid_ns")
        max_book_age_ns = exit_path.get("max_book_age_ns")
        if (
            isinstance(grid_ns, bool)
            or not isinstance(grid_ns, int)
            or grid_ns <= 0
            or isinstance(max_book_age_ns, bool)
            or not isinstance(max_book_age_ns, int)
            or max_book_age_ns < 0
        ):
            raise ValueError("entry T/T exit grid/book-age config is invalid")
        taker_exit_grid_values.add(grid_ns)
        taker_exit_max_book_age_values.add(max_book_age_ns)
        discarded_source_cache_files += _discard_partition_artifact_cache(
            partition,
            _formal_exit_artifact_schemas(),
            source="validated same-day exit-maker artifact",
        )
        discarded_source_cache_files += _discard_partition_artifact_cache(
            entry_root / f"Date={date}" / f"ValueCode={value_code}",
            _ENTRY_TRANSITIVE_SOURCE_ARTIFACTS,
            source="validated entry artifact",
        )
        if len(selected) >= 500 and (
            partition_index % 500 == 0 or partition_index == len(selected)
        ):
            _emit_memory_phase(
                "narrow_partition_validation",
                completed_partitions=partition_index,
                total_partitions=len(selected),
                discarded_source_cache_files=discarded_source_cache_files,
            )

    if len(runner_hashes) != 1 or len(runner_payloads) != 1 or len(runner_versions) != 1:
        raise ValueError("selected exit-maker partitions do not share one runner config")
    if len(entry_config_hashes) != 1:
        raise ValueError("selected entry partitions do not share one runner config")
    if entry_hedge_delay_values != {_exit_report.EXPECTED_HEDGE_DELAY_NS}:
        raise ValueError("selected entry partitions do not share the formal hedge delay")
    if exit_rule_id_sets != {tuple(FROZEN_EXIT_RULE_IDS)}:
        raise ValueError("formal exit-maker runner must declare Center and Lower")
    if exit_route_sets != {tuple(sorted(FROZEN_EXIT_ROUTES))}:
        raise ValueError("formal exit-maker runner must declare both exit routes")
    if len(taker_exit_grid_values) != 1 or len(taker_exit_max_book_age_values) != 1:
        raise ValueError("selected T/T baselines do not share one exit-grid config")
    manifest_metadata = _validate_exact_exit_root_manifest(root, selected)
    entry_action_schema_inventory_sha256 = _canonical_sha256(
        sorted(
            entry_action_schema_records,
            key=lambda row: str(row["partition"]),
        )
    )
    full_frozen_schema_inventory = full_frozen_action_inventory
    if full_frozen_schema_inventory and (
        entry_action_schema_inventory_sha256
        != _EXPECTED_ENTRY_ACTION_SCHEMA_INVENTORY_SHA256
    ):
        raise ValueError("entry action full schema inventory digest mismatch")
    entry_raw_order_inventory_sha256 = _canonical_sha256(
        sorted(
            entry_raw_order_inventory_records,
            key=lambda row: str(row["partition"]),
        )
    )
    entry_raw_order_partition_unique_sum = sum(
        int(row["unique_raw_order_fact_ids"])
        for row in entry_raw_order_inventory_records
    )
    if full_frozen_action_inventory and (
        entry_raw_order_inventory_sha256
        != _EXPECTED_ENTRY_RAW_ORDER_PARTITION_INVENTORY_SHA256
        or entry_raw_order_partition_unique_sum
        != _EXPECTED_ENTRY_RAW_ORDER_PARTITION_UNIQUE_SUM
    ):
        raise ValueError("entry raw-order partition inventory digest mismatch")

    def combine(
        frames: Sequence[pl.DataFrame], schema: Mapping[str, pl.DataType]
    ) -> pl.DataFrame:
        return (
            pl.concat(frames, how="vertical", rechunk=False)
            if frames
            else pl.DataFrame(schema=schema)
        )

    actions = combine(action_frames, _NARROW_ACTION_SCHEMA)
    positions = combine(position_frames, _NARROW_POSITION_SCHEMA)
    bound_exits = combine(bound_exit_frames, _NARROW_EXIT_RULE_SCHEMA)
    execution_diagnostics = _merge_partition_execution_diagnostics(
        execution_diagnostic_frames,
        expected_action_rows=all_action_rows,
        expected_established_rows=established_action_rows,
        raw_order_ids_partition_disjoint=True,
    )
    if actions.height != established_action_rows:
        raise ValueError("retained established action count changed during concat")
    position_lineage = _exit_report._validate_position_rule_lineage(
        positions,
        bound_exits,
        expected_session_predecessors=predecessors,
    )
    metadata: dict[str, object] = {
        "report_version": _exit_report.REPORT_VERSION,
        "exit_maker_root": str(root),
        "entry_execution_root": str(entry_root),
        "selected_session_count": len(selected_dates),
        "selected_dates": selected_dates,
        "selected_product_count": len(products),
        "selected_product_day_count": len(selected),
        "expected_product_day_count": len(selected_dates) * len(products),
        "missing_product_day_count": missing.height,
        "balanced_product_day_grid": missing.is_empty(),
        "value_codes": products,
        "runner_config_sha256": next(iter(runner_hashes)),
        "entry_config_sha256": next(iter(entry_config_hashes)),
        "runner_version": next(iter(runner_versions)),
        "hedge_delay_ns": _exit_report.EXPECTED_HEDGE_DELAY_NS,
        "entry_action_hedge_delay_ns": next(iter(entry_hedge_delay_values)),
        "exit_maker_hedge_delay_ns": _exit_report.EXPECTED_HEDGE_DELAY_NS,
        "entry_exit_hedge_delay_match": True,
        "entry_action_hedge_delay_rows_validated": full_hedged_rows,
        "_post_cross_execution_diagnostics": execution_diagnostics,
        "_post_cross_all_action_rows": all_action_rows,
        "post_cross_retained_established_action_rows": (
            established_action_rows
        ),
        "post_cross_full_actions_preaggregated": True,
        "exit_rule_ids": list(next(iter(exit_rule_id_sets))),
        "exit_routes": list(next(iter(exit_route_sets))),
        "session_predecessors": predecessors,
        "session_predecessor_validation": "exact_calendar",
        "position_rule_lineage": position_lineage,
        "position_policy_to_bound_entry_exit_facts_crossvalidated": True,
        "entry_action_and_exit_facts_hash_lineage_validated": True,
        "entry_action_nonzero_full_schemas_exact": True,
        "entry_action_schema_inventory_sha256": (
            entry_action_schema_inventory_sha256
        ),
        "entry_action_full_schema_inventory_bound": (
            full_frozen_schema_inventory
        ),
        "entry_raw_order_partition_inventory_sha256": (
            entry_raw_order_inventory_sha256
        ),
        "entry_raw_order_partition_unique_sum": (
            entry_raw_order_partition_unique_sum
        ),
        "entry_raw_order_partition_disjoint_bound": True,
        "entry_raw_order_full_inventory_bound": (
            full_frozen_action_inventory
        ),
        **manifest_metadata,
        "taker_taker_exit_grid_ns": next(iter(taker_exit_grid_values)),
        "taker_taker_exit_max_book_age_ns": next(
            iter(taker_exit_max_book_age_values)
        ),
        "partition_hashes_validated": True,
        "formal_narrow_loader": True,
        "formal_narrow_loader_version": "post_cross_narrow_formal_loader_v3",
        "formal_exit_artifact_schemas_exact": True,
        "formal_exit_artifacts_hash_validated": validated_artifacts,
        "formal_source_cache_discard_required": True,
        "formal_source_cache_discard_files": discarded_source_cache_files,
        "unused_exit_artifacts_materialized": False,
        "bound_entry_exit_rules_only": True,
        "formal_partition_source_binding_rows": len(source_binding_records),
        "formal_partition_source_binding_sha256": _canonical_sha256(
            sorted(
                source_binding_records,
                key=lambda row: (str(row["Date"]), str(row["ValueCode"])),
            )
        ),
    }
    return ExitMakerPartitionInputs(
        policy_support=pl.DataFrame(),
        candidate_aliases=pl.DataFrame(),
        raw_candidate_facts=pl.DataFrame(),
        position_policy_facts=positions,
        action_facts=actions,
        taker_exit_facts=bound_exits,
        coverage=coverage,
        metadata=metadata,
    )


def _validate_prerequisite_source_root_binding(
    expected_prerequisite: Mapping[str, object],
    *,
    entry_root: Path,
    prerequisite_root: Path,
) -> None:
    """Bind supplied roots to the exact paths signed by the prerequisite."""

    declared_prerequisite = expected_prerequisite.get("root")
    source_identity = expected_prerequisite.get("source_identity")
    if not isinstance(declared_prerequisite, str) or not isinstance(
        source_identity, Mapping
    ):
        raise ValueError("formal prerequisite source identity is incomplete")
    if Path(declared_prerequisite).resolve(strict=True) != prerequisite_root:
        raise ValueError("supplied prerequisite root differs from its signed identity")

    declared_entry = source_identity.get("entry_execution_root")
    product_days = source_identity.get("product_days")
    if not isinstance(declared_entry, str) or not isinstance(product_days, Mapping):
        raise ValueError("formal prerequisite lacks its entry source root binding")
    declared_entry_path = Path(declared_entry)
    if declared_entry_path.is_symlink() or (
        declared_entry_path.resolve(strict=True) != entry_root
    ):
        raise ValueError(
            "supplied entry execution root differs from prerequisite source identity"
        )

    declared_manifest = product_days.get("path")
    expected_manifest = entry_root / "execution_partition_manifest.parquet"
    if not isinstance(declared_manifest, str):
        raise ValueError("formal prerequisite lacks its entry manifest path binding")
    declared_manifest_path = Path(declared_manifest)
    _require_regular_nonsymlink_file(
        declared_manifest_path, "prerequisite-bound entry manifest"
    )
    if declared_manifest_path.resolve(strict=True) != expected_manifest:
        raise ValueError(
            "prerequisite product-day manifest does not belong to supplied entry root"
        )


def _normalise_embedded_partition_source_identity(
    value: object,
    *,
    expected_partition: Path,
    artifact_names: Sequence[str],
    source: str,
) -> dict[str, object]:
    expected_keys = {
        "partition",
        "marker_sha256",
        "config_sha256",
        "runner_version",
        "artifacts",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ValueError(f"cross marker {source} identity schema is not exact")
    if value.get("partition") != str(expected_partition):
        raise ValueError(
            f"cross marker {source} path differs from supplied formal root"
        )
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(artifact_names):
        raise ValueError(f"cross marker {source} artifact identity is not exact")
    runner_version = value.get("runner_version")
    if not isinstance(runner_version, str) or not runner_version:
        raise ValueError(f"cross marker {source} runner version is invalid")
    return {
        "partition": str(expected_partition),
        "marker_sha256": _validate_sha256(
            value.get("marker_sha256"), f"cross {source} marker sha256"
        ),
        "config_sha256": _validate_sha256(
            value.get("config_sha256"), f"cross {source} config sha256"
        ),
        "runner_version": runner_version,
        "artifacts": {
            name: _validate_sha256(
                artifacts.get(name), f"cross {source}/{name} sha256"
            )
            for name in artifact_names
        },
    }


def _validate_cross_embedded_source_bindings(
    cross_root: Path,
    entry_root: Path,
    same_day_root: Path,
    coverage: pl.DataFrame,
    *,
    expected_rows: int,
    expected_sha256: str,
) -> dict[str, object]:
    """Reconcile every v8 cross marker with the current formal source bytes."""

    expected_sha256 = _validate_sha256(
        expected_sha256, "formal partition source binding sha256"
    )
    complete = coverage.filter(
        pl.col("partition_complete") == True  # noqa: E712
    ).select(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
    ).sort(["Date", "ValueCode"])
    if complete.height != expected_rows:
        raise ValueError("formal source binding row count differs from coverage")

    records: list[dict[str, object]] = []
    for row in complete.iter_rows(named=True):
        date = str(row["Date"])
        value_code = str(row["ValueCode"])
        partition = _canonical_product_day_partition(
            cross_root,
            {
                "Date": date,
                "ValueCode": value_code,
                "partition": (
                    cross_root / f"Date={date}" / f"ValueCode={value_code}"
                ),
                "marker": (
                    cross_root
                    / f"Date={date}"
                    / f"ValueCode={value_code}"
                    / "complete.json"
                ),
            },
        )
        payload = _exit_report._read_json(partition / "complete.json")
        if (
            payload.get("complete") is not True
            or str(payload.get("Date")) != date
            or str(payload.get("ValueCode")) != value_code
        ):
            raise ValueError("cross marker product-day identity is invalid")
        config = payload.get("config")
        if not isinstance(config, Mapping) or _canonical_sha256(config) != payload.get(
            "config_sha256"
        ):
            raise ValueError("cross marker config identity is invalid")
        source = config.get("source")
        expected_source_keys = {
            "Date",
            "ValueCode",
            "entry_execution",
            "same_day_exit_maker",
        }
        if (
            not isinstance(source, Mapping)
            or set(source) != expected_source_keys
            or str(source.get("Date")) != date
            or str(source.get("ValueCode")) != value_code
        ):
            raise ValueError("cross marker embedded source identity is not exact")
        entry_partition = entry_root / f"Date={date}" / f"ValueCode={value_code}"
        same_day_partition = (
            same_day_root / f"Date={date}" / f"ValueCode={value_code}"
        )
        records.append(
            {
                "Date": date,
                "ValueCode": value_code,
                "entry_execution": _normalise_embedded_partition_source_identity(
                    source.get("entry_execution"),
                    expected_partition=entry_partition,
                    artifact_names=_ENTRY_TRANSITIVE_SOURCE_ARTIFACTS,
                    source="entry execution",
                ),
                "same_day_exit_maker": (
                    _normalise_embedded_partition_source_identity(
                        source.get("same_day_exit_maker"),
                        expected_partition=same_day_partition,
                        artifact_names=_SAME_DAY_TRANSITIVE_SOURCE_ARTIFACTS,
                        source="same-day exit-maker",
                    )
                ),
            }
        )
    actual_sha256 = _canonical_sha256(records)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "cross embedded entry/same-day source hashes differ from current roots"
        )
    return {
        "cross_embedded_formal_sources_reconciled": True,
        "cross_embedded_formal_source_rows": len(records),
        "cross_embedded_formal_source_sha256": actual_sha256,
    }


def _discard_cross_session_artifact_cache(
    cross_root: Path,
    coverage: pl.DataFrame,
) -> int:
    """Drop only the exact v8 outputs already verified by the frozen loader."""

    from .exit_maker_cross_session_runner import (
        CROSS_SESSION_MANIFEST_NAME,
        _OUTPUT_ARTIFACTS,
    )

    count = 0
    complete = coverage.filter(
        pl.col("partition_complete") == True  # noqa: E712
    ).select("Date", "ValueCode")
    for row in complete.sort("Date", "ValueCode").iter_rows(named=True):
        partition = (
            cross_root
            / f"Date={str(row['Date'])}"
            / f"ValueCode={str(row['ValueCode'])}"
        )
        if partition.parent.is_symlink() or partition.is_symlink():
            raise ValueError(
                f"cross-session partition must not be a symlink: {partition}"
            )
        count += _discard_partition_artifact_cache(
            partition,
            _OUTPUT_ARTIFACTS,
            source="validated cross-session artifact",
        )
    _discard_clean_file_cache(
        cross_root / CROSS_SESSION_MANIFEST_NAME,
        source="validated cross-session manifest",
    )
    return count + 1


def load_formal_post_cross_sources(
    *,
    exit_maker_root: Path,
    entry_execution_root: Path,
    cross_session_root: Path,
    prerequisite_root: Path,
    sessions: int = 60,
    expected_product_days: int = 2_687,
    expected_product_count: int = 45,
    expected_grid_product_days: int = 2_700,
    expected_missing_product_days: int = 13,
    value_codes: Iterable[str] | None = None,
) -> FormalPostCrossSources:
    """Verify all formal roots and return only nominal-V0 terminal paths.

    This function performs no writes.  It intentionally has no
    ``skip_hash_validation`` or ``allow_fewer_sessions`` escape hatch.
    """

    raw_roots = {
        "entry execution": Path(entry_execution_root),
        "same-day exit": Path(exit_maker_root),
        "cross-session": Path(cross_session_root),
        "cross prerequisite": Path(prerequisite_root),
    }
    symlink_roots = [label for label, root in raw_roots.items() if root.is_symlink()]
    if symlink_roots:
        raise ValueError(
            "formal source roots must not be symlinks: " + ", ".join(symlink_roots)
        )
    entry_root = raw_roots["entry execution"].resolve()
    exit_root = raw_roots["same-day exit"].resolve()
    cross_root = raw_roots["cross-session"].resolve()
    prerequisite = raw_roots["cross prerequisite"].resolve()
    for label, root in (
        ("entry execution", entry_root),
        ("same-day exit", exit_root),
        ("cross-session", cross_root),
        ("cross prerequisite", prerequisite),
    ):
        if not Path(root).is_dir():
            raise FileNotFoundError(f"formal {label} root does not exist: {root}")

    expected_prerequisite = _expected_formal_prerequisite_identity(
        prerequisite
    )
    _validate_prerequisite_source_root_binding(
        expected_prerequisite,
        entry_root=entry_root,
        prerequisite_root=prerequisite,
    )
    calendar = _prerequisite_session_calendar(expected_prerequisite)
    inputs = _load_narrow_formal_partition_inputs(
        exit_root,
        entry_root,
        sessions=sessions,
        value_codes=value_codes,
        session_calendar=calendar,
    )
    _emit_memory_phase(
        "formal_narrow_inputs_loaded",
        action_rows=inputs.action_facts.height,
        position_rows=inputs.position_policy_facts.height,
        bound_exit_rule_rows=inputs.taker_exit_facts.height,
    )
    complete_coverage, missing_coverage = _validate_formal_coverage(
        inputs.coverage,
        inputs.metadata,
        expected_sessions=sessions,
        expected_product_count=expected_product_count,
        expected_complete_product_days=expected_product_days,
        expected_grid_product_days=expected_grid_product_days,
        expected_missing_product_days=expected_missing_product_days,
    )
    binding_rows = inputs.metadata.get("formal_partition_source_binding_rows")
    binding_sha = inputs.metadata.get("formal_partition_source_binding_sha256")
    if (
        isinstance(binding_rows, bool)
        or not isinstance(binding_rows, int)
        or not isinstance(binding_sha, str)
    ):
        raise ValueError("formal narrow loader source binding metadata is invalid")
    cross_source_binding = _validate_cross_embedded_source_bindings(
        cross_root,
        entry_root,
        exit_root,
        inputs.coverage,
        expected_rows=binding_rows,
        expected_sha256=binding_sha,
    )

    overlay = load_cross_session_nominal_terminal_facts(
        cross_root,
        expected_coverage=inputs.coverage,
        expected_position_policy_facts=inputs.position_policy_facts,
        expected_prerequisite_identity=expected_prerequisite,
    )
    cross_cache_files = _discard_cross_session_artifact_cache(
        cross_root, inputs.coverage
    )
    _emit_memory_phase(
        "formal_cross_overlay_loaded",
        terminal_rows=overlay.terminal_policy_facts.height,
        discarded_cross_cache_files=cross_cache_files,
    )
    execution_diagnostics = inputs.metadata.get(
        "_post_cross_execution_diagnostics"
    )
    all_action_rows = inputs.metadata.get("_post_cross_all_action_rows")
    if not isinstance(execution_diagnostics, pl.DataFrame) or (
        isinstance(all_action_rows, bool)
        or not isinstance(all_action_rows, int)
        or all_action_rows < inputs.action_facts.height
    ):
        raise ValueError("formal preaggregated action diagnostics are invalid")
    report_input_metadata = dict(inputs.metadata)
    del report_input_metadata["_post_cross_execution_diagnostics"]
    del report_input_metadata["_post_cross_all_action_rows"]
    report_inputs = ExitMakerPartitionInputs(
        policy_support=inputs.policy_support,
        candidate_aliases=inputs.candidate_aliases,
        raw_candidate_facts=inputs.raw_candidate_facts,
        position_policy_facts=inputs.position_policy_facts,
        action_facts=inputs.action_facts,
        taker_exit_facts=inputs.taker_exit_facts,
        coverage=inputs.coverage,
        metadata=report_input_metadata,
    )
    report = build_filled_entry_primary_report(
        report_inputs,
        terminal_policy_facts=overlay.terminal_policy_facts,
    )
    _emit_memory_phase(
        "filled_entry_report_built",
        policy_path_rows=report.filled_entry_policy_paths.height,
    )
    metadata = dict(report.metadata)
    metadata.update(overlay.metadata)
    metadata.update(cross_source_binding)
    metadata["entry_policy_aliases_all"] = all_action_rows
    metadata["post_cross_full_action_diagnostics_preaggregated"] = True
    metadata["post_cross_execution_diagnostic_rows"] = (
        execution_diagnostics.height
    )
    metadata["formal_entry_root_prerequisite_bound"] = True
    metadata["cross_session_source_cache_discard_files"] = cross_cache_files
    metadata["formal_source_cache_discard_complete"] = True
    report = FilledEntryPrimaryReport(
        coverage=report.coverage,
        filled_entry_policy_paths=report.filled_entry_policy_paths,
        policy_summary=report.policy_summary,
        pooled_12_cell_summary=report.pooled_12_cell_summary,
        product_policy_summary=report.product_policy_summary,
        execution_diagnostics=execution_diagnostics,
        metadata=metadata,
    )
    required_metadata = {
        "cross_session_partition_hashes_validated": True,
        "cross_session_policy_lineage_crossvalidated": True,
        "cross_session_formal_prerequisite_bound": True,
        "formal_entry_root_prerequisite_bound": True,
        "cross_embedded_formal_sources_reconciled": True,
        "cross_session_d_minus_one_lineage_validated": True,
        "strict_cross_session_outcomes_in_primary": False,
        "complete_center_lower_x_two_exit_routes_per_established_entry": True,
        "position_policy_to_bound_entry_exit_facts_crossvalidated": True,
        "entry_exit_hedge_delay_match": True,
        "gross_zero_imputation": False,
        "formal_narrow_loader": True,
        "formal_exit_artifact_schemas_exact": True,
        "entry_action_nonzero_full_schemas_exact": True,
        "entry_action_full_schema_inventory_bound": True,
        "entry_raw_order_partition_disjoint_bound": True,
        "entry_raw_order_full_inventory_bound": True,
        "formal_source_cache_discard_required": True,
        "formal_source_cache_discard_complete": True,
        "post_cross_full_actions_preaggregated": True,
        "post_cross_full_action_diagnostics_preaggregated": True,
        "unused_exit_artifacts_materialized": False,
        "bound_entry_exit_rules_only": True,
        "same_day_root_manifest_exact_inventory": True,
        "same_day_root_manifest_exact_schema": True,
    }
    for key, expected in required_metadata.items():
        if metadata.get(key) is not expected:
            raise ValueError(
                f"formal post-cross source readiness check failed: {key}={metadata.get(key)!r}"
            )
    if (
        metadata.get("entry_action_schema_inventory_sha256")
        != _EXPECTED_ENTRY_ACTION_SCHEMA_INVENTORY_SHA256
    ):
        raise ValueError("formal entry action schema inventory identity mismatch")
    expected_report_sources = _report_implementation_sources()
    if metadata.get("report_implementation_sources") != expected_report_sources:
        raise ValueError(
            "formal filled-entry report implementation identity is not current"
        )
    if metadata.get("report_implementation_sources_sha256") != _canonical_sha256(
        expected_report_sources
    ):
        raise ValueError("formal filled-entry report implementation hash mismatch")

    terminal = overlay.terminal_policy_facts
    _require(
        terminal,
        {"exit_policy_trial_id", *_FORMAL_CROSS_AUDIT_COLUMNS},
        "formal nominal cross outcomes",
    )
    audit = terminal.select("exit_policy_trial_id", *_FORMAL_CROSS_AUDIT_COLUMNS)
    if audit.select("exit_policy_trial_id").n_unique() != audit.height:
        raise ValueError("formal nominal outcomes duplicate exit_policy_trial_id")
    action_quantiles = inputs.action_facts.select(
        pl.col("policy_generation_id").alias("entry_policy_generation_id"),
        "boundary_quantile",
    )
    if (
        action_quantiles.select("entry_policy_generation_id").n_unique()
        != action_quantiles.height
    ):
        raise ValueError("entry actions duplicate policy_generation_id")
    physical_audit = (
        inputs.position_policy_facts.join(
            action_quantiles,
            on="entry_policy_generation_id",
            how="left",
            validate="m:1",
        )
        .join(audit, on="exit_policy_trial_id", how="left", validate="1:1")
    )
    _require(
        physical_audit,
        {*PHYSICAL_PATH_KEY, *_FORMAL_CROSS_AUDIT_COLUMNS},
        "alias-level formal terminal audit",
    )
    conflicts = physical_audit.group_by(*PHYSICAL_PATH_KEY).agg(
        *(
            pl.col(column).n_unique().alias(column)
            for column in _FORMAL_CROSS_AUDIT_COLUMNS
        )
    ).filter(
        pl.any_horizontal(
            *(pl.col(column) != 1 for column in _FORMAL_CROSS_AUDIT_COLUMNS)
        )
    )
    if conflicts.height:
        raise ValueError(
            "policy aliases disagree on terminal interval/readiness audit fields"
        )
    physical_audit = physical_audit.group_by(
        *PHYSICAL_PATH_KEY, maintain_order=True
    ).agg(
        *(
            pl.col(column).first().alias(column)
            for column in _FORMAL_CROSS_AUDIT_COLUMNS
        )
    )
    paths = report.filled_entry_policy_paths.join(
        physical_audit,
        on=list(PHYSICAL_PATH_KEY),
        how="left",
        validate="1:1",
    )
    if paths["cancel_semantics"].null_count():
        raise ValueError("collapsed physical paths are missing cross outcome lineage")
    if set(paths["cancel_semantics"].unique().to_list()) != {
        "nominal_instant_cancel_v0"
    }:
        raise ValueError("post-cross evaluator may load nominal outcomes only")
    if paths.filter(
        (pl.col("nominal_cancel_model_assumption") != True).fill_null(True)  # noqa: E712
    ).height:
        raise ValueError("nominal cross paths do not declare the model assumption")
    if paths.filter(
        (pl.col("pathwise_ev_ready") != False).fill_null(True)  # noqa: E712
    ).height:
        raise ValueError("gross nominal paths incorrectly advertise EV readiness")

    selected_dates = sorted(
        set(inputs.coverage["Date"].cast(pl.String).to_list())
    )
    if len(selected_dates) != sessions:
        raise ValueError(
            f"formal entry cohort must span exactly {sessions} sessions; "
            f"found {len(selected_dates)}"
        )
    if not set(paths["Date"].cast(pl.String).to_list()).issubset(selected_dates):
        raise ValueError("filled-entry paths fall outside formal entry coverage")
    manifest_path = cross_root / "cross_session_partition_manifest.parquet"
    source_metadata = {
        **metadata,
        "exit_maker_root": str(exit_root),
        "entry_execution_root": str(entry_root),
        "cross_session_root": str(cross_root),
        "cross_session_prerequisite_root": str(prerequisite),
        "formal_entry_product_days": complete_coverage.height,
        "formal_entry_grid_product_days": inputs.coverage.height,
        "formal_entry_missing_product_days": missing_coverage.height,
        "formal_entry_session_count": len(selected_dates),
        "cross_session_manifest_path": str(manifest_path.resolve()),
        "cross_session_manifest_sha256": _file_sha256(manifest_path),
        "formal_hash_validation_skippable": False,
        "formal_roots_mutated": False,
    }
    return FormalPostCrossSources(
        report=report,
        policy_paths=paths,
        session_calendar=tuple(calendar),
        entry_sessions=tuple(selected_dates),
        metadata=source_metadata,
    )


def build_post_cross_position_evaluation(
    sources: FormalPostCrossSources,
    *,
    config: PostCrossEvaluationConfig = PostCrossEvaluationConfig(),
    cost_sensitivities: Sequence[CostSensitivity] = DEFAULT_COST_SENSITIVITIES,
    position_limits: Sequence[PositionLimit] = DEFAULT_POSITION_LIMITS,
    full_cost_profile_root: Path | None = None,
) -> PostCrossPositionEvaluation:
    """Build descriptive and fail-closed prequential post-cross tables."""

    config.validate()
    sensitivities = tuple(cost_sensitivities)
    limits = tuple(position_limits)
    if not sensitivities:
        raise ValueError("at least one cost sensitivity is required")
    if not limits:
        raise ValueError("at least one position limit is required")
    for item in sensitivities:
        item.validate()
    for item in limits:
        item.validate()
    if len({item.scenario_id for item in sensitivities}) != len(sensitivities):
        raise ValueError("cost sensitivity scenario_id values must be unique")
    if len({item.limit_id for item in limits}) != len(limits):
        raise ValueError("position limit_id values must be unique")

    _validate_source_bundle(sources)

    paths = _prepare_policy_paths(sources.policy_paths, sources.session_calendar)
    universe_sha = _policy_path_universe_sha256(paths)
    paths, cost_metadata = _attach_full_cost_profile(
        paths,
        full_cost_profile_root,
        expected_universe_sha256=universe_sha,
        expected_cross_manifest_sha256=str(
            sources.metadata["cross_session_manifest_sha256"]
        ),
    )
    policy = _policy_summary(paths, sensitivities, POLICY_KEY)
    product_policy = _policy_summary(
        paths, sensitivities, ("ValueCode", *POLICY_KEY)
    )
    daily_terminal = _daily_terminal_cashflows(
        paths, sources.session_calendar, sources.entry_sessions, sensitivities
    )
    daily_outstanding = _daily_outstanding(
        paths, sources.session_calendar, sources.entry_sessions
    )
    rankings, decisions = _prequential_rankings(
        paths,
        sources.session_calendar,
        sources.entry_sessions,
        config,
        sensitivities[-1],
    )
    sweep = _position_limit_sweep(paths, limits, sensitivities)
    readiness, readiness_metadata = _readiness(
        paths,
        rankings,
        decisions,
        cost_metadata,
        sources.metadata,
    )

    implementation_sources = _implementation_sources()
    evaluator_config_payload = {
        "config": asdict(config),
        "cost_sensitivities": [asdict(item) for item in sensitivities],
        "position_limits": [asdict(item) for item in limits],
        "full_cost_profile_hash": cost_metadata.get("full_cost_profile_hash"),
        "policy_path_universe_sha256": universe_sha,
        "cross_session_manifest_sha256": sources.metadata[
            "cross_session_manifest_sha256"
        ],
    }
    metadata: dict[str, object] = {
        **sources.metadata,
        **cost_metadata,
        **readiness_metadata,
        "evaluator_version": EVALUATOR_VERSION,
        "evaluator_implementation_sources": implementation_sources,
        "evaluator_implementation_sources_sha256": _canonical_sha256(
            implementation_sources
        ),
        "evaluator_config_sha256": _canonical_sha256(evaluator_config_payload),
        "config": asdict(config),
        "cost_sensitivities": [asdict(item) for item in sensitivities],
        "position_limits": [asdict(item) for item in limits],
        "physical_policy_paths": paths.height,
        "policy_path_universe_sha256": universe_sha,
        "policy_cells_expected": 24,
        "policy_cells_reported": policy.height,
        "alternative_policy_rows_additive": False,
        "boundary_quantile_rows_safe_to_sum": False,
        "entry_route_rows_safe_to_sum": False,
        "exit_rule_rows_safe_to_sum": False,
        "exit_route_rows_safe_to_sum": False,
        "alias_rows_double_counted": False,
        "daily_terminal_zero_means_no_cashflow_not_unresolved_imputation": True,
        "unresolved_cashflow_imputed": False,
        "cost_sensitivity_is_full_cost_profile": False,
        "gross_includes_observed_entry_exit_price_and_latency_slippage": True,
        "observed_slippage_subtracted_again": False,
        "completed_only_terminal_cashflow_is_equity_curve": False,
        "win_rate_mdd_sharpe_published": False,
        "notional_semantics": "one_way_spot_leg_entry_price_notional_twd",
        "notional_is_eod_mark": False,
        "notional_is_two_leg_gross_exposure": False,
        "notional_is_futures_margin": False,
        "notional_is_capital_requirement": False,
        "position_limit_sweep_semantics": (
            "independent_fixed_policy_recv_time_ns_conservative_entry_before_exit_on_tie"
        ),
        "position_limit_sweeps_joint_volume_allocated": False,
        "position_limit_sweeps_production_ready": False,
        "best_q_claimed": bool(
            decisions.height
            and decisions.sort("asof_date").tail(1).item(
                0, "selected_action_ev_ready"
            )
        ),
        "formal_roots_mutated": False,
    }
    _validate_evaluation(
        paths,
        policy,
        product_policy,
        daily_terminal,
        daily_outstanding,
        rankings,
        decisions,
        sweep,
        readiness,
    )
    return PostCrossPositionEvaluation(
        policy_paths=paths,
        policy_summary=policy,
        product_policy_summary=product_policy,
        daily_terminal_cashflows=daily_terminal,
        daily_outstanding=daily_outstanding,
        prequential_rankings=rankings,
        prequential_decisions=decisions,
        position_limit_sweep=sweep,
        readiness=readiness,
        metadata=metadata,
    )


def run_post_cross_position_evaluation(
    *,
    exit_maker_root: Path,
    entry_execution_root: Path,
    cross_session_root: Path,
    prerequisite_root: Path,
    output: Path,
    full_cost_profile_root: Path | None = None,
    config: PostCrossEvaluationConfig = PostCrossEvaluationConfig(),
    cost_sensitivities: Sequence[CostSensitivity] = DEFAULT_COST_SENSITIVITIES,
    position_limits: Sequence[PositionLimit] = DEFAULT_POSITION_LIMITS,
) -> PostCrossPositionEvaluation:
    """Run after all formal roots exist; publish only after every check passes."""

    _emit_memory_phase("formal_run_start")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if full_cost_profile_root is not None:
        raise ValueError(
            "caller-supplied path cost roots are not a formal source-bound component "
            "producer and cannot unlock EV; omit --full-cost-profile-root"
        )
    _validate_output_disjoint(
        output,
        (
            Path(exit_maker_root),
            Path(entry_execution_root),
            Path(cross_session_root),
            Path(prerequisite_root),
            *((Path(full_cost_profile_root),) if full_cost_profile_root else ()),
        ),
    )
    # Nothing above or in the source/build functions creates the output path.
    sources = load_formal_post_cross_sources(
        exit_maker_root=Path(exit_maker_root),
        entry_execution_root=Path(entry_execution_root),
        cross_session_root=Path(cross_session_root),
        prerequisite_root=Path(prerequisite_root),
        sessions=60,
        expected_product_days=config.expected_product_days,
    )
    _emit_memory_phase(
        "formal_sources_ready",
        policy_path_rows=sources.policy_paths.height,
    )
    evaluation = build_post_cross_position_evaluation(
        sources,
        config=config,
        cost_sensitivities=cost_sensitivities,
        position_limits=position_limits,
        full_cost_profile_root=full_cost_profile_root,
    )
    _emit_memory_phase(
        "post_cross_evaluation_built",
        policy_path_rows=evaluation.policy_paths.height,
    )
    _publish_evaluation(evaluation, output)
    _emit_memory_phase("post_cross_bundle_published")
    return evaluation


def verify_post_cross_position_evaluation(root: Path) -> dict[str, object]:
    """Rebuild from immutable formal roots and verify the exact output bundle."""

    return _verify_post_cross_position_evaluation(root, source_override=None)


def _verify_post_cross_position_evaluation_with_sources(
    root: Path,
    sources: FormalPostCrossSources,
) -> dict[str, object]:
    """Bounded-fixture verifier; production callers cannot override formal roots."""

    return _verify_post_cross_position_evaluation(root, source_override=sources)


def _verify_post_cross_position_evaluation(
    root: Path,
    *,
    source_override: FormalPostCrossSources | None,
) -> dict[str, object]:
    root = Path(root)
    marker_path = root / "complete.json"
    if not marker_path.is_file():
        raise FileNotFoundError(f"post-cross evaluation is incomplete: {root}")
    expected_entries = {"complete.json", *_ARTIFACTS}
    actual_entries = {path.name for path in root.iterdir()}
    if actual_entries != expected_entries or any(
        path.is_symlink() or not path.is_file() for path in root.iterdir()
    ):
        raise ValueError("post-cross evaluation root inventory is not exact")

    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker_keys = {
        "complete",
        "schema_version",
        "evaluator_version",
        "metadata",
        "artifacts",
    }
    if not isinstance(marker, dict) or set(marker) != marker_keys:
        raise ValueError("post-cross evaluation marker envelope is not exact")
    if marker["complete"] is not True:
        raise ValueError("post-cross evaluation marker is not complete")
    if marker["schema_version"] != BUNDLE_SCHEMA_VERSION:
        raise ValueError("post-cross evaluation bundle schema mismatch")
    if marker["evaluator_version"] != EVALUATOR_VERSION:
        raise ValueError("post-cross evaluation version mismatch")
    metadata = marker["metadata"]
    artifacts = marker["artifacts"]
    if not isinstance(metadata, dict):
        raise ValueError("post-cross evaluation metadata must be an object")
    if not isinstance(artifacts, dict) or set(artifacts) != set(_ARTIFACTS):
        raise ValueError("post-cross evaluation artifact inventory mismatch")

    frames: dict[str, pl.DataFrame] = {}
    declaration_keys = {"rows", "columns", "schema", "bytes", "sha256"}
    for name in _ARTIFACTS:
        declared = artifacts[name]
        if not isinstance(declared, dict) or set(declared) != declaration_keys:
            raise ValueError(f"invalid artifact declaration: {name}")
        path = root / name
        if _file_sha256(path) != declared["sha256"]:
            raise ValueError(f"post-cross artifact hash mismatch: {name}")
        frame = pl.read_parquet(path)
        if (
            isinstance(declared["rows"], bool)
            or not isinstance(declared["rows"], int)
            or isinstance(declared["columns"], bool)
            or not isinstance(declared["columns"], int)
            or frame.height != declared["rows"]
            or frame.width != declared["columns"]
        ):
            raise ValueError(f"post-cross artifact shape mismatch: {name}")
        if (
            isinstance(declared["bytes"], bool)
            or not isinstance(declared["bytes"], int)
            or path.stat().st_size != declared["bytes"]
        ):
            raise ValueError(f"post-cross artifact byte count mismatch: {name}")
        schema = {column: str(dtype) for column, dtype in frame.schema.items()}
        if schema != declared["schema"]:
            raise ValueError(f"post-cross artifact schema mismatch: {name}")
        frames[name] = frame

    _assert_safe_bundle_metadata(metadata)
    _validate_evaluation(
        frames["physical_policy_paths.parquet"],
        frames["policy_summary.parquet"],
        frames["product_policy_summary.parquet"],
        frames["daily_terminal_cashflows.parquet"],
        frames["daily_outstanding.parquet"],
        frames["prequential_policy_rankings.parquet"],
        frames["prequential_decisions.parquet"],
        frames["position_limit_sweep.parquet"],
        frames["readiness.parquet"],
    )

    config = _config_from_metadata(metadata)
    sensitivities = _cost_sensitivities_from_metadata(metadata)
    limits = _position_limits_from_metadata(metadata)
    if source_override is None:
        root_keys = {
            "exit_maker_root",
            "entry_execution_root",
            "cross_session_root",
            "cross_session_prerequisite_root",
        }
        if not root_keys.issubset(metadata):
            raise ValueError("published bundle lacks immutable formal source roots")
        sources = load_formal_post_cross_sources(
            exit_maker_root=Path(str(metadata["exit_maker_root"])),
            entry_execution_root=Path(str(metadata["entry_execution_root"])),
            cross_session_root=Path(str(metadata["cross_session_root"])),
            prerequisite_root=Path(
                str(metadata["cross_session_prerequisite_root"])
            ),
            sessions=60,
            expected_product_days=config.expected_product_days,
        )
    else:
        sources = source_override
    rebuilt = build_post_cross_position_evaluation(
        sources,
        config=config,
        cost_sensitivities=sensitivities,
        position_limits=limits,
        full_cost_profile_root=None,
    )
    expected_metadata = json.loads(
        json.dumps(dict(rebuilt.metadata), sort_keys=True, default=str)
    )
    if metadata != expected_metadata:
        raise ValueError("published metadata does not exactly match source rebuild")
    for filename, attribute in _ARTIFACTS.items():
        expected = getattr(rebuilt, attribute)
        actual = frames[filename]
        if actual.schema != expected.schema or not actual.equals(
            expected, null_equal=True
        ):
            raise ValueError(
                f"post-cross artifact does not exactly match source rebuild: {filename}"
            )
    return marker


def _assert_safe_bundle_metadata(metadata: Mapping[str, object]) -> None:
    current_sources = _implementation_sources()
    if metadata.get("evaluator_version") != EVALUATOR_VERSION:
        raise ValueError("post-cross evaluator metadata version mismatch")
    if metadata.get("evaluator_implementation_sources") != current_sources:
        raise ValueError("post-cross evaluator implementation identity changed")
    if metadata.get(
        "evaluator_implementation_sources_sha256"
    ) != _canonical_sha256(current_sources):
        raise ValueError("post-cross evaluator implementation hash mismatch")
    required_false = (
        "alternative_policy_rows_additive",
        "unresolved_cashflow_imputed",
        "formal_roots_mutated",
        "pathwise_ev_ready",
        "strategy_defensible",
        "best_q_claimed",
        "nominal_post_fill_fixed_policy_ev_ready",
        "full_cost_profile_complete",
        "formal_component_cost_source_bound",
        "d_safe_prequential_ev_action_exists",
        "best_q_selection_go",
        "position_limit_deployment_go",
        "production_strategy_go",
    )
    bad = [name for name in required_false if metadata.get(name) is not False]
    if bad:
        raise ValueError(f"published bundle violates fail-closed flags: {bad}")
    if metadata.get("source_integrity_go") is not True:
        raise ValueError("published bundle did not pass complete source integrity")
    config_payload = {
        "config": metadata.get("config"),
        "cost_sensitivities": metadata.get("cost_sensitivities"),
        "position_limits": metadata.get("position_limits"),
        "full_cost_profile_hash": metadata.get("full_cost_profile_hash"),
        "policy_path_universe_sha256": metadata.get(
            "policy_path_universe_sha256"
        ),
        "cross_session_manifest_sha256": metadata.get(
            "cross_session_manifest_sha256"
        ),
    }
    if metadata.get("evaluator_config_sha256") != _canonical_sha256(
        config_payload
    ):
        raise ValueError("post-cross evaluator config identity mismatch")


def _config_from_metadata(
    metadata: Mapping[str, object],
) -> PostCrossEvaluationConfig:
    payload = metadata.get("config")
    expected = {
        "lookback_sessions",
        "min_training_dates",
        "min_policy_origins",
        "min_completed_cycles_for_diagnostic_rank",
        "confidence_level",
        "minimum_lcb_bp",
        "expected_product_days",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("published evaluator config schema is not exact")
    try:
        config = PostCrossEvaluationConfig(**payload)
        config.validate()
    except (TypeError, ValueError) as error:
        raise ValueError("published evaluator config is invalid") from error
    return config


def _cost_sensitivities_from_metadata(
    metadata: Mapping[str, object],
) -> tuple[CostSensitivity, ...]:
    payload = metadata.get("cost_sensitivities")
    expected = {"scenario_id", "same_day_bp", "overnight_bp"}
    if not isinstance(payload, list) or not payload:
        raise ValueError("published cost sensitivities are invalid")
    values: list[CostSensitivity] = []
    for row in payload:
        if not isinstance(row, dict) or set(row) != expected:
            raise ValueError("published cost sensitivity schema is not exact")
        try:
            item = CostSensitivity(**row)
            item.validate()
        except (TypeError, ValueError) as error:
            raise ValueError("published cost sensitivity is invalid") from error
        values.append(item)
    if len({item.scenario_id for item in values}) != len(values):
        raise ValueError("published cost sensitivity identifiers are not unique")
    return tuple(values)


def _position_limits_from_metadata(
    metadata: Mapping[str, object],
) -> tuple[PositionLimit, ...]:
    payload = metadata.get("position_limits")
    expected = {
        "limit_id",
        "max_concurrent_positions",
        "max_outstanding_notional_twd",
        "scope",
    }
    if not isinstance(payload, list) or not payload:
        raise ValueError("published position limits are invalid")
    values: list[PositionLimit] = []
    for row in payload:
        if not isinstance(row, dict) or set(row) != expected:
            raise ValueError("published position-limit schema is not exact")
        try:
            item = PositionLimit(**row)
            item.validate()
        except (TypeError, ValueError) as error:
            raise ValueError("published position limit is invalid") from error
        values.append(item)
    if len({item.limit_id for item in values}) != len(values):
        raise ValueError("published position-limit identifiers are not unique")
    return tuple(values)


def _prepare_policy_paths(
    frame: pl.DataFrame, session_calendar: Sequence[str]
) -> pl.DataFrame:
    required = {
        *PHYSICAL_PATH_KEY,
        "exit_policy_trial_id",
        "physical_entry_dependency_id",
        "filled_entry_policy_aliases",
        "normalization_notional_twd",
        "filled_entry_outcome_category",
        "terminal_date",
        "terminal_reason",
        "gross_cycle_pnl_twd",
        "gross_cycle_bp",
        *_FORMAL_CROSS_AUDIT_COLUMNS,
    }
    _require(frame, required, "post-cross policy paths")
    if frame.is_empty():
        raise ValueError("post-cross policy paths are empty")
    if frame.select(*PHYSICAL_PATH_KEY).n_unique() != frame.height:
        raise ValueError("post-cross physical policy paths are not de-aliased")
    if frame.select("exit_policy_trial_id").n_unique() != frame.height:
        raise ValueError("post-cross physical paths reuse exit_policy_trial_id")
    if frame["physical_entry_dependency_id"].null_count() or frame.filter(
        pl.col("physical_entry_dependency_id").cast(pl.String).str.len_chars() == 0
    ).height:
        raise ValueError("physical entry dependency identity is null or empty")
    if (
        frame.select(*POLICY_KEY, "physical_entry_dependency_id").n_unique()
        != frame.height
    ):
        raise ValueError(
            "physical_entry_dependency_id must be unique within every policy cell"
        )
    dependency_population = frame.group_by(
        "physical_entry_dependency_id", "boundary_quantile"
    ).agg(
        pl.len().alias("rows"),
        pl.struct("exit_rule_id", "exit_route").n_unique().alias("actions"),
        pl.col("exit_rule_id").n_unique().alias("rules"),
        pl.col("exit_route").n_unique().alias("routes"),
        *(
            pl.col(column).n_unique().alias(f"identity__{column}")
            for column in (
                "Date",
                "ValueCode",
                "QuoteCode",
                "entry_route",
                "entry_raw_order_fact_id",
                "position_established_ns",
            )
        ),
    )
    if dependency_population.filter(
        (pl.col("rows") != 4)
        | (pl.col("actions") != 4)
        | (pl.col("rules") != 2)
        | (pl.col("routes") != 2)
        | pl.any_horizontal(
            *(
                pl.col(f"identity__{column}") != 1
                for column in (
                    "Date",
                    "ValueCode",
                    "QuoteCode",
                    "entry_route",
                    "entry_raw_order_fact_id",
                    "position_established_ns",
                )
            )
        )
    ).height:
        raise ValueError(
            "physical entry dependency does not have one immutable four-action exit grid"
        )
    expected_q = set(FROZEN_BOUNDARY_QUANTILES)
    expected_entry = set(FROZEN_ENTRY_ROUTES)
    expected_rules = set(FROZEN_EXIT_RULE_IDS)
    expected_exit = set(FROZEN_EXIT_ROUTES)
    for column, expected in (
        ("boundary_quantile", expected_q),
        ("entry_route", expected_entry),
        ("exit_rule_id", expected_rules),
        ("exit_route", expected_exit),
    ):
        actual = set(frame[column].drop_nulls().unique().to_list())
        if actual != expected:
            raise ValueError(
                f"formal policy dimension {column} differs: {sorted(actual)}"
            )
    if frame.filter(
        pl.col("filled_entry_policy_aliases").is_null()
        | (pl.col("filled_entry_policy_aliases") <= 0)
    ).height:
        raise ValueError("physical paths have invalid alias accounting")
    if set(frame["cancel_semantics"].drop_nulls().unique().to_list()) != {
        "nominal_instant_cancel_v0"
    } or frame["cancel_semantics"].null_count():
        raise ValueError("post-cross evaluator accepts nominal terminal paths only")
    if frame.filter(
        (pl.col("nominal_cancel_model_assumption") != True).fill_null(True)  # noqa: E712
        | (pl.col("pathwise_ev_ready") != False).fill_null(True)  # noqa: E712
        | (pl.col("joint_volume_allocated") != False).fill_null(True)  # noqa: E712
    ).height:
        raise ValueError("nominal path readiness declarations are inconsistent")
    notional = pl.col("normalization_notional_twd")
    if frame.filter(notional.is_null() | ~notional.is_finite() | (notional <= 0)).height:
        raise ValueError("physical paths require finite positive notional")

    calendar = tuple(str(value) for value in session_calendar)
    if len(calendar) != len(set(calendar)) or list(calendar) != sorted(calendar):
        raise ValueError("session calendar must be unique and increasing")
    index = {value: position for position, value in enumerate(calendar)}
    dates = frame["Date"].cast(pl.String).to_list()
    if missing := sorted(set(dates) - set(index)):
        raise ValueError(f"entry dates absent from session calendar: {missing[:5]}")

    completed = pl.col("filled_entry_outcome_category") == "completed"
    unresolved = ~completed
    calendar_lookup = pl.DataFrame(
        {
            "Date": calendar,
            "_entry_calendar_index": list(range(len(calendar))),
        }
    )
    label_lookup = pl.DataFrame(
        {
            "_label_date": calendar,
            "_label_calendar_index": list(range(len(calendar))),
            "_next_calendar_date": [*calendar[1:], None],
        }
    )
    result = frame.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("terminal_date").cast(pl.String),
        pl.col("last_observed_session_date").cast(pl.String),
    ).join(calendar_lookup, on="Date", how="left", validate="m:1")
    if result.filter(
        ~pl.col("filled_entry_outcome_category").is_in(
            ["completed", "censored", "unknown", "still_open"]
        )
        | pl.col("_entry_calendar_index").is_null()
    ).height:
        raise ValueError("invalid terminal category or entry session")
    if result.filter(
        pl.col("position_established_ns").is_null()
        | (pl.col("position_established_ns") < 0)
    ).height:
        raise ValueError("position establishment cursor is invalid")
    result = result.with_columns(
        pl.when(completed)
        .then(pl.col("terminal_date"))
        .otherwise(pl.col("last_observed_session_date"))
        .alias("_label_date")
    ).join(label_lookup, on="_label_date", how="left", validate="m:1")

    reconstructed = (
        pl.col("gross_cycle_pnl_twd")
        / pl.col("normalization_notional_twd")
        * 10_000.0
    )
    gross_tolerance = pl.max_horizontal(
        pl.lit(1e-9),
        pl.max_horizontal(
            reconstructed.abs(), pl.col("gross_cycle_bp").abs()
        )
        * 1e-10,
    )
    if result.filter(
        completed
        & (
            pl.col("terminal_date").is_null()
            | pl.col("_label_calendar_index").is_null()
        )
    ).height:
        raise ValueError("completed path terminal_date is outside the calendar")
    if result.filter(
        completed
        & (
            pl.col("gross_cycle_pnl_twd").is_null()
            | ~pl.col("gross_cycle_pnl_twd").is_finite()
            | pl.col("gross_cycle_bp").is_null()
            | ~pl.col("gross_cycle_bp").is_finite()
        )
    ).height:
        raise ValueError("completed path requires finite gross")
    if result.filter(
        completed
        & (pl.col("terminal_cashflow_priced") != True).fill_null(True)  # noqa: E712
    ).height:
        raise ValueError("completed path is not terminal-cashflow priced")
    if result.filter(
        completed & (pl.col("outcome_type") != "terminal").fill_null(True)
    ).height:
        raise ValueError("completed path outcome_type must be terminal")
    if result.filter(
        completed
        & (
            pl.col("last_observed_session_date")
            != pl.col("terminal_date")
        ).fill_null(True)
    ).height:
        raise ValueError(
            "completed path last-observed date must equal terminal date"
        )
    if result.filter(
        completed
        & (
            pl.col("exit_decision_time_ns").is_null()
            | (pl.col("exit_decision_time_ns") < 0)
        )
    ).height:
        raise ValueError("completed path exit decision cursor is invalid")
    if result.filter(
        completed
        & (pl.col("terminal_date") == pl.col("Date"))
        & (
            pl.col("exit_decision_time_ns")
            < pl.col("position_established_ns")
        )
    ).height:
        raise ValueError("same-day completed exit precedes position establishment")
    if result.filter(
        completed
        & ((reconstructed - pl.col("gross_cycle_bp")).abs() > gross_tolerance)
    ).height:
        raise ValueError("completed path gross TWD/bp decomposition disagrees")
    if result.filter(
        unresolved
        & (
            pl.col("terminal_date").is_not_null()
            | pl.col("gross_cycle_pnl_twd").is_not_null()
        )
    ).height:
        raise ValueError("unresolved path carries a terminal cashflow")
    if result.filter(
        unresolved & pl.col("_label_calendar_index").is_null()
    ).height:
        raise ValueError(
            "unresolved path requires last_observed_session_date in calendar"
        )
    if result.filter(
        unresolved
        & (pl.col("terminal_cashflow_priced") != False).fill_null(True)  # noqa: E712
    ).height:
        raise ValueError("unresolved path claims a priced terminal cashflow")
    if result.filter(
        unresolved & (pl.col("outcome_type") != "censored").fill_null(True)
    ).height:
        raise ValueError("unresolved path outcome_type must be censored")
    if result.filter(
        pl.col("_label_calendar_index") < pl.col("_entry_calendar_index")
    ).height:
        raise ValueError("path label availability precedes entry")

    def path_identifier(value: Mapping[str, object]) -> str:
        return _canonical_sha256(
            {name: value[name] for name in PHYSICAL_PATH_KEY}
        )

    result = result.with_columns(
        pl.struct(PHYSICAL_PATH_KEY)
        .map_elements(path_identifier, return_dtype=pl.String)
        .alias("policy_path_id"),
        pl.col("_label_date").alias("label_availability_date"),
        pl.when(completed)
        .then(pl.col("terminal_date"))
        .otherwise(pl.col("_next_calendar_date"))
        .alias("outstanding_interval_end_exclusive"),
        (
            pl.col("_label_calendar_index")
            - pl.col("_entry_calendar_index")
        ).alias("holding_session_boundaries"),
        (completed & (pl.col("terminal_date") == pl.col("Date"))).alias(
            "completed_same_day"
        ),
        (completed & (pl.col("terminal_date") > pl.col("Date"))).alias(
            "completed_overnight"
        ),
        pl.lit(
            "entry_eod_through_before_terminal_date;unresolved_through_last_observed_eod"
        ).alias("outstanding_interval_semantics"),
    ).drop(
        "_entry_calendar_index",
        "_label_date",
        "_label_calendar_index",
        "_next_calendar_date",
    )
    if result.select("policy_path_id").n_unique() != result.height:
        raise AssertionError("derived policy_path_id is not unique")
    return result.sort([*POLICY_KEY, "Date", "position_established_ns", "policy_path_id"])


def _attach_full_cost_profile(
    paths: pl.DataFrame,
    root: Path | None,
    *,
    expected_universe_sha256: str,
    expected_cross_manifest_sha256: str,
) -> tuple[pl.DataFrame, dict[str, object]]:
    del expected_universe_sha256, expected_cross_manifest_sha256
    if root is not None:
        raise ValueError(
            "caller-supplied path cost roots are not a formal source-bound component "
            "producer and cannot unlock EV; omit --full-cost-profile-root"
        )
    result = paths.with_columns(
        *(pl.lit(None, dtype=pl.Float64).alias(name) for name in COST_COMPONENT_COLUMNS),
        pl.lit(None, dtype=pl.Float64).alias("full_non_price_cost_bp"),
        pl.lit(None, dtype=pl.Float64).alias("full_cost_net_path_value_bp"),
        pl.lit(False).alias("full_cost_components_complete"),
        pl.lit(False).alias("formal_component_cost_source_bound"),
        pl.lit(None, dtype=pl.String).alias("cost_profile_source_asof_date"),
        pl.lit(None, dtype=pl.String).alias("cost_profile_hash"),
    )
    return result, {
        "full_cost_profile_supplied": False,
        "full_cost_profile_complete": False,
        "formal_component_cost_source_bound": False,
        "full_cost_profile_id": None,
        "full_cost_profile_version": None,
        "full_cost_profile_hash": None,
        "full_cost_profile_source_asof_date": None,
        "full_cost_profile_contains_target_day_outcome": None,
        "full_cost_profile_readiness_reason": (
            "formal_source_bound_component_cost_producer_not_available"
        ),
    }


def _policy_summary(
    paths: pl.DataFrame,
    sensitivities: Sequence[CostSensitivity],
    keys: Sequence[str],
) -> pl.DataFrame:
    category = pl.col("filled_entry_outcome_category")
    completed = category == "completed"
    base = paths.group_by(list(keys)).agg(
        pl.len().alias("filled_entry_positions"),
        pl.col("filled_entry_policy_aliases").sum().alias(
            "source_policy_aliases_collapsed"
        ),
        pl.col("physical_entry_dependency_id").n_unique().alias(
            "unique_physical_entry_dependencies"
        ),
        pl.col("Date").n_unique().alias("entry_sessions"),
        completed.sum().alias("completed_cycles"),
        (category == "censored").sum().alias("censored_positions"),
        (category == "unknown").sum().alias("unknown_positions"),
        (category == "still_open").sum().alias("still_open_positions"),
        pl.col("normalization_notional_twd").sum().alias(
            "new_entry_one_way_notional_twd"
        ),
        pl.col("gross_cycle_pnl_twd").sum().alias("completed_gross_twd"),
        pl.col("gross_cycle_bp").mean().alias("completed_gross_mean_bp"),
        pl.col("gross_cycle_bp").median().alias("completed_gross_p50_bp"),
        pl.col("gross_cycle_bp").quantile(0.05).alias("completed_gross_p05_bp"),
        pl.col("gross_cycle_bp").quantile(0.95).alias("completed_gross_p95_bp"),
        pl.col("holding_session_boundaries").filter(completed).mean().alias(
            "completed_mean_overnight_boundaries"
        ),
    ).with_columns(
        (pl.col("completed_cycles") / pl.col("filled_entry_positions")).alias(
            "completion_rate"
        ),
        (pl.col("censored_positions") / pl.col("filled_entry_positions")).alias(
            "censor_rate"
        ),
        (
            (pl.col("unknown_positions") + pl.col("still_open_positions"))
            / pl.col("filled_entry_positions")
        ).alias("unknown_or_open_rate"),
        (
            (pl.col("completed_cycles") == pl.col("filled_entry_positions"))
            & (pl.col("censored_positions") == 0)
            & (pl.col("unknown_positions") == 0)
            & (pl.col("still_open_positions") == 0)
        ).alias("terminal_cashflow_point_identified"),
        pl.lit(False).alias("alternative_policy_rows_additive"),
        pl.lit(False).alias("unresolved_cashflow_imputed"),
    )
    for scenario in sensitivities:
        cost_bp = _sensitivity_cost_expr(scenario)
        values = paths.filter(completed).with_columns(
            cost_bp.alias("_sensitivity_bp"),
            (
                pl.col("gross_cycle_pnl_twd")
                - cost_bp
                * pl.col("normalization_notional_twd")
                / 10_000.0
            ).alias("_after_twd"),
            (pl.col("gross_cycle_bp") - cost_bp).alias("_after_bp"),
        ).group_by(list(keys)).agg(
            pl.col("_after_twd").sum().alias(
                f"{scenario.scenario_id}__completed_after_twd"
            ),
            pl.col("_after_bp").mean().alias(
                f"{scenario.scenario_id}__completed_after_mean_bp"
            ),
        )
        base = base.join(values, on=list(keys), how="left", validate="1:1")
    if "ValueCode" in keys:
        grid = paths.select("ValueCode").unique().join(_policy_grid(), how="cross")
    else:
        grid = _policy_grid()
    base = grid.join(base, on=list(keys), how="left", validate="1:1")
    zero_columns = (
        "filled_entry_positions",
        "source_policy_aliases_collapsed",
        "unique_physical_entry_dependencies",
        "entry_sessions",
        "completed_cycles",
        "censored_positions",
        "unknown_positions",
        "still_open_positions",
        "new_entry_one_way_notional_twd",
        "completed_gross_twd",
    )
    base = base.with_columns(
        *(pl.col(name).fill_null(0).alias(name) for name in zero_columns),
        pl.col("terminal_cashflow_point_identified").fill_null(False),
        pl.col("alternative_policy_rows_additive").fill_null(False),
        pl.col("unresolved_cashflow_imputed").fill_null(False),
    ).with_columns(
        (pl.col("filled_entry_positions") > 0).alias(
            "policy_cell_has_filled_entry_support"
        )
    )
    return base.sort(list(keys))


def _daily_terminal_cashflows(
    paths: pl.DataFrame,
    session_calendar: Sequence[str],
    entry_sessions: Sequence[str],
    sensitivities: Sequence[CostSensitivity],
) -> pl.DataFrame:
    completed = paths.filter(
        pl.col("filled_entry_outcome_category") == "completed"
    )
    cells = _policy_grid()
    first = min(entry_sessions)
    last = max(paths["label_availability_date"].to_list())
    terminal_dates = [date for date in session_calendar if first <= date <= last]
    grid = pl.DataFrame({"terminal_date": terminal_dates}).join(
        cells, how="cross"
    )
    aggregates = completed.group_by("terminal_date", *POLICY_KEY).agg(
        pl.len().alias("terminal_completed_cycles"),
        pl.col("gross_cycle_pnl_twd").sum().alias("terminal_realized_gross_twd"),
        pl.col("normalization_notional_twd").sum().alias(
            "terminal_completed_normalization_notional_twd"
        ),
    )
    result = grid.join(
        aggregates,
        on=["terminal_date", *POLICY_KEY],
        how="left",
        validate="1:1",
    ).with_columns(
        pl.col("terminal_completed_cycles").fill_null(0),
        pl.col("terminal_realized_gross_twd").fill_null(0.0),
        pl.col("terminal_completed_normalization_notional_twd").fill_null(0.0),
    )
    for scenario in sensitivities:
        adjusted = completed.with_columns(
            _sensitivity_cost_expr(scenario).alias("_cost_bp")
        ).with_columns(
            (
                pl.col("gross_cycle_pnl_twd")
                - pl.col("_cost_bp")
                * pl.col("normalization_notional_twd")
                / 10_000.0
            ).alias("_after")
        ).group_by("terminal_date", *POLICY_KEY).agg(
            pl.col("_after").sum().alias(
                f"{scenario.scenario_id}__terminal_after_twd"
            )
        )
        result = result.join(
            adjusted,
            on=["terminal_date", *POLICY_KEY],
            how="left",
            validate="1:1",
        ).with_columns(
            pl.col(f"{scenario.scenario_id}__terminal_after_twd").fill_null(0.0)
        )
    return result.with_columns(
        pl.lit("terminal_date_realized_cashflow").alias("date_semantics"),
        pl.lit(False).alias("unresolved_cashflow_imputed"),
        pl.lit(True).alias(
            "zero_daily_cashflow_means_no_completed_terminal_that_day"
        ),
    ).sort(["terminal_date", *POLICY_KEY])


def _daily_outstanding(
    paths: pl.DataFrame,
    session_calendar: Sequence[str],
    entry_sessions: Sequence[str],
) -> pl.DataFrame:
    calendar = tuple(session_calendar)
    first = min(entry_sessions)
    last_label = max(paths["label_availability_date"].to_list())
    dates = [date for date in calendar if first <= date <= last_label]
    cells = [dict(row) for row in _policy_grid().iter_rows(named=True)]
    rows: list[dict[str, object]] = []
    grouped = paths.partition_by(list(POLICY_KEY), as_dict=True, maintain_order=True)
    for cell in cells:
        key = tuple(cell[name] for name in POLICY_KEY)
        # Polars partition keys are tuples even for multiple columns.
        cell_paths = grouped.get(key, pl.DataFrame())
        date_index = {date: idx for idx, date in enumerate(dates)}
        count_delta = [0] * (len(dates) + 1)
        notional_delta = [0.0] * (len(dates) + 1)
        completed_delta = [0] * (len(dates) + 1)
        censored_delta = [0] * (len(dates) + 1)
        unknown_delta = [0] * (len(dates) + 1)
        new_count = [0] * len(dates)
        new_notional = [0.0] * len(dates)
        for item in cell_paths.iter_rows(named=True):
            start = date_index[str(item["Date"])]
            end_date = item["outstanding_interval_end_exclusive"]
            end = date_index.get(str(end_date), len(dates)) if end_date else len(dates)
            notional = float(item["normalization_notional_twd"])
            new_count[start] += 1
            new_notional[start] += notional
            # A same-day completion has start == end and contributes no EOD risk.
            if end > start:
                count_delta[start] += 1
                count_delta[end] -= 1
                notional_delta[start] += notional
                notional_delta[end] -= notional
                category = str(item["filled_entry_outcome_category"])
                target = (
                    completed_delta
                    if category == "completed"
                    else censored_delta
                    if category == "censored"
                    else unknown_delta
                )
                target[start] += 1
                target[end] -= 1
        live_count = live_completed = live_censored = live_unknown = 0
        live_notional = 0.0
        for idx, date in enumerate(dates):
            live_count += count_delta[idx]
            live_notional += notional_delta[idx]
            live_completed += completed_delta[idx]
            live_censored += censored_delta[idx]
            live_unknown += unknown_delta[idx]
            rows.append(
                {
                    **cell,
                    "Date": date,
                    "new_positions": new_count[idx],
                    "new_entry_one_way_notional_twd": new_notional[idx],
                    "outstanding_eod_positions": live_count,
                    "outstanding_eod_one_way_entry_notional_twd": live_notional,
                    "eventually_completed_outstanding_positions": live_completed,
                    "eventually_censored_outstanding_positions": live_censored,
                    "eventually_unknown_or_open_outstanding_positions": live_unknown,
                    "outstanding_category_breakdown_uses_final_outcome": True,
                    "notional_is_eod_mark": False,
                    "notional_is_two_leg_gross_exposure": False,
                    "notional_is_futures_margin": False,
                    "notional_is_capital_requirement": False,
                }
            )
    return pl.from_dicts(rows, infer_schema_length=None).sort(
        ["Date", *POLICY_KEY]
    )


def _prequential_rankings(
    paths: pl.DataFrame,
    session_calendar: Sequence[str],
    entry_sessions: Sequence[str],
    config: PostCrossEvaluationConfig,
    diagnostic_sensitivity: CostSensitivity,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    calendar = tuple(session_calendar)
    index = {date: idx for idx, date in enumerate(calendar)}
    rows: list[dict[str, object]] = []
    for asof_date in entry_sessions:
        asof_index = index[asof_date]
        history_dates = calendar[max(0, asof_index - config.lookback_sessions):asof_index]
        history = paths.filter(pl.col("Date").is_in(history_dates))
        history_by_policy = history.partition_by(
            list(POLICY_KEY), as_dict=True, maintain_order=True
        )
        for cell in _policy_grid().iter_rows(named=True):
            policy_key = tuple(cell[name] for name in POLICY_KEY)
            subset = history_by_policy.get(policy_key, paths.head(0))
            n = subset.height
            mature = subset.filter(pl.col("label_availability_date") < asof_date)
            pending = n - mature.height
            completed = mature.filter(
                pl.col("filled_entry_outcome_category") == "completed"
            )
            censored = mature.filter(
                pl.col("filled_entry_outcome_category") == "censored"
            ).height
            unknown = mature.filter(
                pl.col("filled_entry_outcome_category").is_in(
                    ["unknown", "still_open"]
                )
            ).height
            training_dates = subset["Date"].n_unique() if n else 0
            complete_cost = bool(
                n
                and subset["full_cost_components_complete"].fill_null(False).all()
            )
            formal_cost_source_bound = bool(
                n
                and subset["formal_component_cost_source_bound"]
                .fill_null(False)
                .all()
            )
            cost_asof_safe = formal_cost_source_bound
            if (
                formal_cost_source_bound
                and "cost_profile_source_asof_date" in subset.columns
            ):
                # Reserved for a future row-varying profile contract.
                cost_asof_safe = bool(
                    subset["cost_profile_source_asof_date"].drop_nulls().len() == 0
                    or (
                        subset["cost_profile_source_asof_date"].drop_nulls()
                        < asof_date
                    ).all()
                )
            point_identified = bool(
                n > 0
                and pending == 0
                and mature.height == n
                and completed.height == n
                and censored == 0
                and unknown == 0
            )
            ev_ready = bool(
                point_identified
                and complete_cost
                and formal_cost_source_bound
                and cost_asof_safe
                and training_dates >= config.min_training_dates
                and n >= config.min_policy_origins
            )
            diagnostic_mean = None
            if completed.height:
                cost_expr = _sensitivity_cost_expr(diagnostic_sensitivity)
                diagnostic_mean = completed.select(
                    (pl.col("gross_cycle_bp") - cost_expr).mean()
                ).item()
            ev_mean = lcb = standard_error = None
            clusters = 0
            if ev_ready:
                values = completed.select(
                    "Date", "full_cost_net_path_value_bp"
                )
                ev_mean = float(values["full_cost_net_path_value_bp"].mean())
                cluster = values.group_by("Date").agg(
                    (
                        pl.col("full_cost_net_path_value_bp") - ev_mean
                    ).sum().alias("score")
                )
                clusters = cluster.height
                if clusters >= 2:
                    standard_error = math.sqrt(
                        clusters
                        / (clusters - 1)
                        * float(cluster["score"].pow(2).sum())
                        / (n * n)
                    )
                    critical = _student_t_quantile(
                        config.confidence_level, clusters - 1
                    )
                    lcb = ev_mean - critical * standard_error
                else:
                    ev_ready = False
            if not n:
                status = "no_prior_policy_origins"
            elif training_dates < config.min_training_dates:
                status = "insufficient_training_dates"
            elif n < config.min_policy_origins:
                status = "insufficient_physical_policy_origins"
            elif pending:
                status = "pending_terminal_labels"
            elif censored or unknown:
                status = "unpriced_censored_or_unknown_terminal_mass"
            elif not formal_cost_source_bound:
                status = "formal_source_bound_component_cost_producer_not_available"
            elif not complete_cost:
                status = "incomplete_formal_component_cost_profile"
            elif not cost_asof_safe:
                status = "cost_profile_not_d_safe_for_asof_date"
            elif clusters < 2:
                status = "insufficient_date_clusters_for_lcb"
            elif ev_ready:
                status = "ready_nominal_post_fill_fixed_policy"
            else:
                status = "not_ready"
            rows.append(
                {
                    "asof_date": asof_date,
                    **cell,
                    "history_start_date": history_dates[0] if history_dates else None,
                    "history_end_date": history_dates[-1] if history_dates else None,
                    "n_policy_origins": n,
                    "n_training_dates": training_dates,
                    "n_labels_matured": mature.height,
                    "n_labels_pending": pending,
                    "n_completed_cycles": completed.height,
                    "n_censored": censored,
                    "n_unknown_or_open": unknown,
                    "label_maturity_rate": mature.height / n if n else None,
                    "terminal_cashflow_point_identified": point_identified,
                    "full_cost_components_complete": complete_cost,
                    "formal_component_cost_source_bound": formal_cost_source_bound,
                    "cost_profile_d_safe": cost_asof_safe,
                    "diagnostic_cost_sensitivity_id": diagnostic_sensitivity.scenario_id,
                    "diagnostic_completed_only_after_cost_mean_bp": diagnostic_mean,
                    "diagnostic_rank_support": bool(
                        completed.height
                        >= config.min_completed_cycles_for_diagnostic_rank
                    ),
                    "nominal_post_fill_ev_mean_bp": ev_mean,
                    "date_clustered_standard_error_bp": standard_error,
                    "one_sided_lcb_bp": lcb,
                    "n_date_clusters_for_lcb": clusters,
                    "nominal_post_fill_ev_ready": ev_ready,
                    "ev_status": status,
                    "contains_target_day_outcome": False,
                    "alternative_policy_rows_additive": False,
                    "unresolved_cashflow_imputed": False,
                }
            )
    rankings = pl.from_dicts(rows, infer_schema_length=None)
    rankings = rankings.with_columns(
        pl.when(pl.col("diagnostic_rank_support"))
        .then(
            pl.col("diagnostic_completed_only_after_cost_mean_bp")
            .rank("ordinal", descending=True)
            .over("asof_date")
        )
        .otherwise(None)
        .alias("diagnostic_challenger_rank"),
        pl.when(pl.col("nominal_post_fill_ev_ready"))
        .then(
            pl.col("one_sided_lcb_bp")
            .rank("ordinal", descending=True)
            .over("asof_date")
        )
        .otherwise(None)
        .alias("ev_lcb_rank"),
    ).sort(["asof_date", *POLICY_KEY])

    decisions: list[dict[str, object]] = []
    for asof_date, group in rankings.partition_by(
        "asof_date", as_dict=True, maintain_order=True
    ).items():
        date_value = asof_date[0] if isinstance(asof_date, tuple) else asof_date
        ready = group.filter(
            pl.col("nominal_post_fill_ev_ready")
            & (pl.col("one_sided_lcb_bp") >= config.minimum_lcb_bp)
        ).sort("one_sided_lcb_bp", descending=True)
        challenger = group.filter(pl.col("diagnostic_rank_support")).sort(
            "diagnostic_completed_only_after_cost_mean_bp", descending=True
        )
        selected = ready.row(0, named=True) if ready.height else None
        diagnostic = challenger.row(0, named=True) if challenger.height else None
        decisions.append(
            {
                "asof_date": str(date_value),
                "selected_action_ev_ready": selected is not None,
                "decision_status": (
                    "SELECT_NOMINAL_POST_FILL_FIXED_POLICY"
                    if selected is not None
                    else "NO_GO_NO_EV_READY_ACTION"
                ),
                **{
                    f"selected_{name}": None if selected is None else selected[name]
                    for name in POLICY_KEY
                },
                "selected_nominal_post_fill_ev_mean_bp": (
                    None if selected is None else selected["nominal_post_fill_ev_mean_bp"]
                ),
                "selected_one_sided_lcb_bp": (
                    None if selected is None else selected["one_sided_lcb_bp"]
                ),
                **{
                    f"diagnostic_challenger_{name}": (
                        None if diagnostic is None else diagnostic[name]
                    )
                    for name in POLICY_KEY
                },
                "diagnostic_challenger_completed_only_after_cost_mean_bp": (
                    None
                    if diagnostic is None
                    else diagnostic[
                        "diagnostic_completed_only_after_cost_mean_bp"
                    ]
                ),
                "diagnostic_challenger_is_selected_action": False,
                "selected_action_scope": "nominal_post_fill_fixed_policy_only",
                "ranking_uses_only_Date_less_than_asof": True,
                "label_visible_only_if_label_availability_date_less_than_asof": True,
                "nominal_cancel_model_assumption": True,
                "production_strategy_go": False,
            }
        )
    return rankings, pl.from_dicts(decisions, infer_schema_length=None).sort(
        "asof_date"
    )


def _position_limit_sweep(
    paths: pl.DataFrame,
    limits: Sequence[PositionLimit],
    sensitivities: Sequence[CostSensitivity],
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = paths.partition_by(list(POLICY_KEY), as_dict=True, maintain_order=True)
    for key, cell_paths in grouped.items():
        cell = dict(zip(POLICY_KEY, key))
        ordered = cell_paths.sort(
            ["Date", "position_established_ns", "ValueCode", "policy_path_id"]
        )
        for limit in limits:
            active: list[tuple[str, int, str, float, str]] = []
            active_count = 0
            active_notional = 0.0
            active_count_by_product: dict[str, int] = {}
            active_notional_by_product: dict[str, float] = {}
            accepted = rejected_count = rejected_notional = 0
            accepted_completed = accepted_censored = accepted_unknown = 0
            accepted_gross = 0.0
            accepted_sensitivity_twd = {
                scenario.scenario_id: 0.0 for scenario in sensitivities
            }
            accepted_full_cost_net = 0.0
            accepted_full_cost_complete = True
            accepted_new_notional = 0.0
            peak_count = 0
            peak_notional = 0.0
            peak_single_product_count = 0
            peak_single_product_notional = 0.0
            for item in ordered.iter_rows(named=True):
                entry_key = (
                    str(item["Date"]),
                    int(item["position_established_ns"]),
                )
                # Strictly earlier exits release capacity.  Equal timestamps
                # retain the old position, a conservative tie convention.
                while active and (active[0][0], active[0][1]) < entry_key:
                    _, _, _, released, released_product = heapq.heappop(active)
                    active_count -= 1
                    active_notional -= released
                    active_count_by_product[released_product] -= 1
                    active_notional_by_product[released_product] -= released
                notional = float(item["normalization_notional_twd"])
                value_code = str(item["ValueCode"])
                scoped_count = (
                    active_count
                    if limit.scope == "portfolio"
                    else active_count_by_product.get(value_code, 0)
                )
                scoped_notional = (
                    active_notional
                    if limit.scope == "portfolio"
                    else active_notional_by_product.get(value_code, 0.0)
                )
                count_block = bool(
                    limit.max_concurrent_positions is not None
                    and scoped_count + 1 > limit.max_concurrent_positions
                )
                notional_block = bool(
                    limit.max_outstanding_notional_twd is not None
                    and scoped_notional + notional
                    > limit.max_outstanding_notional_twd + 1e-9
                )
                if count_block or notional_block:
                    rejected_count += int(count_block)
                    rejected_notional += int(notional_block)
                    continue
                accepted += 1
                accepted_new_notional += notional
                active_count += 1
                active_notional += notional
                active_count_by_product[value_code] = (
                    active_count_by_product.get(value_code, 0) + 1
                )
                active_notional_by_product[value_code] = (
                    active_notional_by_product.get(value_code, 0.0) + notional
                )
                peak_count = max(peak_count, active_count)
                peak_notional = max(peak_notional, active_notional)
                peak_single_product_count = max(
                    peak_single_product_count,
                    active_count_by_product[value_code],
                )
                peak_single_product_notional = max(
                    peak_single_product_notional,
                    active_notional_by_product[value_code],
                )
                category = str(item["filled_entry_outcome_category"])
                if category == "completed":
                    accepted_completed += 1
                    accepted_gross += float(item["gross_cycle_pnl_twd"])
                    for scenario in sensitivities:
                        cost_bp = (
                            scenario.same_day_bp
                            if bool(item["completed_same_day"])
                            else scenario.overnight_bp
                        )
                        accepted_sensitivity_twd[scenario.scenario_id] += (
                            float(item["gross_cycle_pnl_twd"])
                            - cost_bp * notional / 10_000.0
                        )
                    net_bp = item.get("full_cost_net_path_value_bp")
                    if _finite(net_bp):
                        accepted_full_cost_net += (
                            float(net_bp) * notional / 10_000.0
                        )
                    else:
                        accepted_full_cost_complete = False
                    exit_time = item.get("exit_decision_time_ns")
                    if exit_time is None:
                        exit_time = 9_223_372_036_854_775_807
                    heapq.heappush(
                        active,
                        (
                            str(item["terminal_date"]),
                            int(exit_time),
                            str(item["policy_path_id"]),
                            notional,
                            value_code,
                        ),
                    )
                elif category == "censored":
                    accepted_censored += 1
                    accepted_full_cost_complete = False
                else:
                    accepted_unknown += 1
                    accepted_full_cost_complete = False
            point_identified = bool(
                accepted
                and accepted_completed == accepted
                and accepted_censored == 0
                and accepted_unknown == 0
            )
            rows.append(
                {
                    **cell,
                    **asdict(limit),
                    "candidate_positions": ordered.height,
                    "accepted_positions": accepted,
                    "rejected_positions": ordered.height - accepted,
                    "rejected_by_count_cap": rejected_count,
                    "rejected_by_notional_cap": rejected_notional,
                    "acceptance_rate": accepted / ordered.height,
                    "accepted_completed_cycles": accepted_completed,
                    "accepted_censored_positions": accepted_censored,
                    "accepted_unknown_or_open_positions": accepted_unknown,
                    "accepted_completed_gross_twd": accepted_gross,
                    **{
                        f"{scenario.scenario_id}__accepted_completed_after_twd": (
                            accepted_sensitivity_twd[scenario.scenario_id]
                        )
                        for scenario in sensitivities
                    },
                    "accepted_new_entry_one_way_notional_twd": accepted_new_notional,
                    "peak_concurrent_positions": peak_count,
                    "peak_outstanding_one_way_entry_notional_twd": peak_notional,
                    "peak_single_product_concurrent_positions": (
                        peak_single_product_count
                    ),
                    "peak_single_product_outstanding_one_way_entry_notional_twd": (
                        peak_single_product_notional
                    ),
                    "accepted_terminal_cashflow_point_identified": point_identified,
                    "full_cost_net_pnl_twd": (
                        accepted_full_cost_net
                        if point_identified and accepted_full_cost_complete
                        else None
                    ),
                    "position_limit_sweep_analysis_only": True,
                    "position_limit_sweep_production_ready": False,
                    "joint_volume_allocated": False,
                    "alternative_policy_rows_additive": False,
                    "unresolved_cashflow_imputed": False,
                    "inventory_unit_semantics": (
                        "one_paired_long_spot_short_future_established_position"
                    ),
                    "gross_and_sensitivity_cashflow_conditioning": "accepted_completed_cycles_only",
                    "tie_handling": "entry_before_exit_at_equal_recv_time_ns",
                }
            )
    return pl.from_dicts(rows, infer_schema_length=None).sort(
        [*POLICY_KEY, "limit_id"]
    )


def _readiness(
    paths: pl.DataFrame,
    rankings: pl.DataFrame,
    decisions: pl.DataFrame,
    cost_metadata: Mapping[str, object],
    source_metadata: Mapping[str, object],
) -> tuple[pl.DataFrame, dict[str, object]]:
    counts = {
        str(row["filled_entry_outcome_category"]): int(row["len"])
        for row in paths.group_by("filled_entry_outcome_category").len().iter_rows(
            named=True
        )
    }
    unresolved = sum(
        counts.get(name, 0) for name in ("censored", "unknown", "still_open")
    )
    latest_selected = bool(
        decisions.height
        and decisions.sort("asof_date").tail(1).item(
            0, "selected_action_ev_ready"
        )
    )
    expected_report_sources = _report_implementation_sources()
    source_checks = [
        (
            "source_coverage_contract_go",
            all(
                source_metadata.get(name) == expected
                for name, expected in (
                    ("formal_entry_product_days", 2_687),
                    ("formal_entry_grid_product_days", 2_700),
                    ("formal_entry_missing_product_days", 13),
                    ("formal_entry_session_count", 60),
                )
            ),
            "formal 60x45 grid must retain 2,687 complete and 13 unavailable",
        ),
        (
            "source_cross_partition_hashes_go",
            source_metadata.get("cross_session_partition_hashes_validated") is True
            and all(
                _is_sha256(source_metadata.get(name))
                for name in (
                    "cross_session_manifest_sha256",
                    "cross_session_runner_config_sha256",
                    "cross_session_prerequisite_marker_sha256",
                    "cross_session_prerequisite_marker_payload_sha256",
                    "cross_session_prerequisite_config_sha256",
                    "cross_session_prerequisite_source_identity_sha256",
                )
            ),
            "cross manifest, partitions, runner, and prerequisite identities are hashed",
        ),
        (
            "source_cross_policy_lineage_go",
            source_metadata.get("cross_session_policy_lineage_crossvalidated")
            is True
            and source_metadata.get("cross_embedded_formal_sources_reconciled")
            is True,
            "cross terminal outcomes map exactly to frozen same-day policy facts",
        ),
        (
            "source_prerequisite_binding_go",
            source_metadata.get("cross_session_formal_prerequisite_bound") is True
            and source_metadata.get("formal_entry_root_prerequisite_bound") is True,
            "cross partitions bind the verified prerequisite root identity",
        ),
        (
            "source_d_minus_one_lineage_go",
            source_metadata.get("cross_session_d_minus_one_lineage_validated")
            is True,
            "every frozen rule source is the exact prior session",
        ),
        (
            "source_entry_exit_foreign_key_go",
            source_metadata.get(
                "position_policy_to_bound_entry_exit_facts_crossvalidated"
            )
            is True,
            "position facts retain immutable entry/exit foreign keys",
        ),
        (
            "source_hedge_50ms_go",
            source_metadata.get("entry_exit_hedge_delay_match") is True
            and source_metadata.get("entry_action_hedge_delay_ns") == 50_000_000
            and source_metadata.get("exit_maker_hedge_delay_ns") == 50_000_000,
            "entry and exit hedge labels are both exactly +50 ms",
        ),
        (
            "source_complete_four_action_grid_go",
            source_metadata.get(
                "complete_center_lower_x_two_exit_routes_per_established_entry"
            )
            is True,
            "every established entry has Center/Lower x two exit routes",
        ),
        (
            "source_no_gross_imputation_go",
            source_metadata.get("gross_zero_imputation") is False
            and source_metadata.get("unknown_or_censored_cashflow_imputed")
            is False,
            "unresolved terminal cashflow is never zero-imputed",
        ),
        (
            "source_nominal_only_go",
            source_metadata.get("strict_cross_session_outcomes_in_primary") is False
            and source_metadata.get("terminal_overlay_cancel_semantics")
            == "nominal_instant_cancel_v0"
            and paths.filter(
                (pl.col("cancel_semantics") != "nominal_instant_cancel_v0")
                .fill_null(True)
                | (
                    pl.col("nominal_cancel_model_assumption") != True  # noqa: E712
                ).fill_null(True)
                | (pl.col("pathwise_ev_ready") != False).fill_null(True)  # noqa: E712
            ).is_empty(),
            "primary paths contain nominal instant-cancel V0 only",
        ),
        (
            "source_implementation_identity_go",
            source_metadata.get("report_implementation_sources")
            == expected_report_sources
            and source_metadata.get("report_implementation_sources_sha256")
            == _canonical_sha256(expected_report_sources)
            and source_metadata.get("formal_hash_validation_skippable") is False
            and source_metadata.get("formal_narrow_loader") is True
            and source_metadata.get("formal_exit_artifact_schemas_exact") is True
            and source_metadata.get("entry_action_nonzero_full_schemas_exact")
            is True
            and source_metadata.get("entry_action_full_schema_inventory_bound")
            is True
            and source_metadata.get("entry_action_schema_inventory_sha256")
            == _EXPECTED_ENTRY_ACTION_SCHEMA_INVENTORY_SHA256
            and source_metadata.get("entry_raw_order_partition_disjoint_bound")
            is True
            and source_metadata.get("entry_raw_order_full_inventory_bound")
            is True
            and source_metadata.get(
                "entry_raw_order_partition_inventory_sha256"
            )
            == _EXPECTED_ENTRY_RAW_ORDER_PARTITION_INVENTORY_SHA256
            and source_metadata.get("entry_raw_order_partition_unique_sum")
            == _EXPECTED_ENTRY_RAW_ORDER_PARTITION_UNIQUE_SUM
            and source_metadata.get("formal_source_cache_discard_required")
            is True
            and source_metadata.get("formal_source_cache_discard_complete")
            is True
            and source_metadata.get("post_cross_full_actions_preaggregated")
            is True
            and source_metadata.get(
                "post_cross_full_action_diagnostics_preaggregated"
            )
            is True
            and source_metadata.get("unused_exit_artifacts_materialized") is False
            and source_metadata.get("bound_entry_exit_rules_only") is True
            and source_metadata.get("same_day_root_manifest_exact_inventory") is True
            and source_metadata.get("same_day_root_manifest_exact_schema") is True,
            (
                "filled-entry validators and the bounded-memory formal loader "
                "match current sources"
            ),
        ),
    ]
    source_integrity = all(bool(go) for _, go, _ in source_checks)
    checks = [
        *source_checks,
        (
            "source_integrity_go",
            source_integrity,
            "all formal coverage, hash, lineage, semantics, hedge, and implementation gates",
        ),
        (
            "descriptive_terminal_report_go",
            source_integrity,
            "requires source integrity; unresolved gross remains un-imputed",
        ),
        (
            "terminal_cashflow_point_identified",
            unresolved == 0,
            f"unpriced terminal paths={unresolved}",
        ),
        (
            "full_cost_profile_go",
            cost_metadata.get("full_cost_profile_complete") is True
            and cost_metadata.get("formal_component_cost_source_bound") is True,
            str(cost_metadata.get("full_cost_profile_readiness_reason")),
        ),
        (
            "d_safe_prequential_ev_action_exists",
            bool(
                rankings.height
                and rankings["nominal_post_fill_ev_ready"].fill_null(False).any()
            ),
            "requires mature point-priced labels, full costs, and support",
        ),
        (
            "best_q_selection_go",
            latest_selected,
            "latest-asof nominal post-fill selection; diagnostic challenger is never promoted",
        ),
        (
            "position_limit_sweep_analysis_go",
            source_integrity,
            "requires source integrity; independent conservative recv-time sweep",
        ),
        (
            "position_limit_deployment_go",
            False,
            "independent alternatives lack joint volume allocation and exact cursor ties",
        ),
        (
            "production_strategy_go",
            False,
            "nominal cancel V0 is a model assumption; strict cancel ACK remains unidentified",
        ),
    ]
    frame = pl.from_dicts(
        [
            {
                "gate": name,
                "go": bool(go),
                "status": "GO" if go else "NO_GO",
                "reason": reason,
            }
            for name, go, reason in checks
        ],
        infer_schema_length=None,
    )
    metadata = {
        name: bool(go) for name, go, _ in checks
    }
    metadata.update(
        outcome_category_counts=counts,
        unresolved_terminal_paths=unresolved,
        nominal_cancel_model_assumption=True,
        strict_cancel_ack_identified=False,
        nominal_post_fill_fixed_policy_ev_ready=metadata[
            "d_safe_prequential_ev_action_exists"
        ],
        pathwise_ev_ready=False,
        strategy_defensible=False,
    )
    return frame, metadata


def _validate_evaluation(
    paths: pl.DataFrame,
    policy: pl.DataFrame,
    product_policy: pl.DataFrame,
    daily_terminal: pl.DataFrame,
    daily_outstanding: pl.DataFrame,
    rankings: pl.DataFrame,
    decisions: pl.DataFrame,
    sweep: pl.DataFrame,
    readiness: pl.DataFrame,
) -> None:
    if policy.height != 24:
        raise AssertionError(f"policy summary must contain 24 cells, found {policy.height}")
    if policy.select(*POLICY_KEY).n_unique() != 24:
        raise AssertionError("policy summary duplicates a fixed-policy cell")
    if paths.select(*PHYSICAL_PATH_KEY).n_unique() != paths.height:
        raise AssertionError("policy paths contain aliases")
    if paths.select("exit_policy_trial_id").n_unique() != paths.height:
        raise AssertionError("policy paths duplicate exit-policy foreign keys")
    if (
        paths.select(*POLICY_KEY, "physical_entry_dependency_id").n_unique()
        != paths.height
    ):
        raise AssertionError("policy cells duplicate physical entry dependencies")
    if product_policy.select("ValueCode", *POLICY_KEY).n_unique() != product_policy.height:
        raise AssertionError("product-policy summary duplicates a cell")
    if daily_terminal.select("terminal_date", *POLICY_KEY).n_unique() != daily_terminal.height:
        raise AssertionError("daily terminal table duplicates a cell-date")
    if daily_outstanding.select("Date", *POLICY_KEY).n_unique() != daily_outstanding.height:
        raise AssertionError("daily outstanding table duplicates a cell-date")
    if rankings.select("asof_date", *POLICY_KEY).n_unique() != rankings.height:
        raise AssertionError("prequential ranking duplicates a cell-date")
    if decisions.select("asof_date").n_unique() != decisions.height:
        raise AssertionError("prequential decisions duplicate an as-of date")
    for name, frame in (
        ("policy summary", policy),
        ("product-policy summary", product_policy),
    ):
        if frame.filter(
            (pl.col("alternative_policy_rows_additive") != False).fill_null(True)  # noqa: E712
            | (pl.col("unresolved_cashflow_imputed") != False).fill_null(True)  # noqa: E712
        ).height:
            raise AssertionError(f"{name} violates alternative/unresolved semantics")
    if daily_terminal.filter(
        (pl.col("unresolved_cashflow_imputed") != False).fill_null(True)  # noqa: E712
    ).height:
        raise AssertionError("daily terminal cashflow imputes unresolved paths")

    outstanding_counts = (
        "new_positions",
        "outstanding_eod_positions",
        "eventually_completed_outstanding_positions",
        "eventually_censored_outstanding_positions",
        "eventually_unknown_or_open_outstanding_positions",
    )
    outstanding_notionals = (
        "new_entry_one_way_notional_twd",
        "outstanding_eod_one_way_entry_notional_twd",
    )
    if daily_outstanding.filter(
        pl.any_horizontal(
            *(pl.col(name).is_null() | (pl.col(name) < 0) for name in outstanding_counts),
            *(
                pl.col(name).is_null()
                | ~pl.col(name).is_finite()
                | (pl.col(name) < -1e-9)
                for name in outstanding_notionals
            ),
        )
        | (
            pl.col("outstanding_eod_positions")
            != pl.col("eventually_completed_outstanding_positions")
            + pl.col("eventually_censored_outstanding_positions")
            + pl.col("eventually_unknown_or_open_outstanding_positions")
        )
    ).height:
        raise AssertionError("daily outstanding counts/notional are impossible")
    if daily_outstanding.filter(
        (pl.col("notional_is_capital_requirement") != False).fill_null(True)  # noqa: E712
    ).height:
        raise AssertionError("daily outstanding misstates notional as capital")

    if rankings.filter(
        (pl.col("contains_target_day_outcome") != False).fill_null(True)  # noqa: E712
        | (pl.col("alternative_policy_rows_additive") != False).fill_null(True)  # noqa: E712
        | (pl.col("unresolved_cashflow_imputed") != False).fill_null(True)  # noqa: E712
        | (pl.col("formal_component_cost_source_bound") != False).fill_null(True)  # noqa: E712
        | (pl.col("full_cost_components_complete") != False).fill_null(True)  # noqa: E712
        | (pl.col("cost_profile_d_safe") != False).fill_null(True)  # noqa: E712
        | (pl.col("nominal_post_fill_ev_ready") != False).fill_null(True)  # noqa: E712
    ).height:
        raise AssertionError("prequential rankings violate D-safe fail-closed gates")
    if decisions.filter(
        ~pl.col("selected_action_ev_ready")
        & pl.any_horizontal(
            *(pl.col(f"selected_{name}").is_not_null() for name in POLICY_KEY)
        )
    ).height:
        raise AssertionError("NO-GO decision leaked a selected action")
    selected_value_columns = (
        *(f"selected_{name}" for name in POLICY_KEY),
        "selected_nominal_post_fill_ev_mean_bp",
        "selected_one_sided_lcb_bp",
    )
    if decisions.filter(
        (pl.col("selected_action_ev_ready") != False).fill_null(True)  # noqa: E712
        | (pl.col("production_strategy_go") != False).fill_null(True)  # noqa: E712
        | (
            pl.col("diagnostic_challenger_is_selected_action") != False  # noqa: E712
        ).fill_null(True)
        | pl.any_horizontal(
            *(pl.col(name).is_not_null() for name in selected_value_columns)
        )
    ).height:
        raise AssertionError("prequential decision leaks an EV action")
    if sweep.filter(
        (pl.col("position_limit_sweep_production_ready") != False).fill_null(True)  # noqa: E712
        | (pl.col("joint_volume_allocated") != False).fill_null(True)  # noqa: E712
        | (pl.col("alternative_policy_rows_additive") != False).fill_null(True)  # noqa: E712
        | (pl.col("unresolved_cashflow_imputed") != False).fill_null(True)  # noqa: E712
        | pl.col("full_cost_net_pnl_twd").is_not_null()
        | (pl.col("candidate_positions") < 0)
        | (pl.col("accepted_positions") < 0)
        | (pl.col("rejected_positions") < 0)
        | (
            pl.col("candidate_positions")
            != pl.col("accepted_positions") + pl.col("rejected_positions")
        )
        | (
            pl.col("accepted_positions")
            != pl.col("accepted_completed_cycles")
            + pl.col("accepted_censored_positions")
            + pl.col("accepted_unknown_or_open_positions")
        )
    ).height:
        raise AssertionError("analysis-only position sweep claims production readiness")
    required_gates = {
        "source_coverage_contract_go",
        "source_cross_partition_hashes_go",
        "source_cross_policy_lineage_go",
        "source_prerequisite_binding_go",
        "source_d_minus_one_lineage_go",
        "source_entry_exit_foreign_key_go",
        "source_hedge_50ms_go",
        "source_complete_four_action_grid_go",
        "source_no_gross_imputation_go",
        "source_nominal_only_go",
        "source_implementation_identity_go",
        "source_integrity_go",
        "descriptive_terminal_report_go",
        "terminal_cashflow_point_identified",
        "full_cost_profile_go",
        "d_safe_prequential_ev_action_exists",
        "best_q_selection_go",
        "position_limit_sweep_analysis_go",
        "position_limit_deployment_go",
        "production_strategy_go",
    }
    if (
        readiness.select("gate").n_unique() != readiness.height
        or set(readiness["gate"].to_list()) != required_gates
    ):
        raise AssertionError("readiness gate inventory is not exact")
    if readiness.filter(
        pl.col("gate").is_in(
            [
                "full_cost_profile_go",
                "d_safe_prequential_ev_action_exists",
                "best_q_selection_go",
                "position_limit_deployment_go",
                "production_strategy_go",
            ]
        )
        & pl.col("go")
    ).height:
        raise AssertionError("nominal evaluation claims production strategy GO")


def _validate_source_bundle(sources: FormalPostCrossSources) -> None:
    calendar = tuple(str(value) for value in sources.session_calendar)
    entry_sessions = tuple(str(value) for value in sources.entry_sessions)
    if not calendar or len(calendar) != len(set(calendar)) or list(calendar) != sorted(calendar):
        raise ValueError("source session calendar must be non-empty, unique, and increasing")
    if (
        not entry_sessions
        or len(entry_sessions) != len(set(entry_sessions))
        or list(entry_sessions) != sorted(entry_sessions)
    ):
        raise ValueError("source entry sessions must be non-empty, unique, and increasing")
    if not set(entry_sessions).issubset(calendar):
        raise ValueError("source entry sessions are absent from the candidate calendar")
    _require(sources.policy_paths, {"Date"}, "source policy paths")
    path_dates = set(sources.policy_paths["Date"].cast(pl.String).to_list())
    if not path_dates.issubset(entry_sessions):
        raise ValueError("source policy paths fall outside the entry-session cohort")
    manifest_digest = sources.metadata.get("cross_session_manifest_sha256")
    _validate_sha256(manifest_digest, "cross_session_manifest_sha256")


def _validate_formal_coverage(
    coverage: pl.DataFrame,
    metadata: Mapping[str, object],
    *,
    expected_sessions: int,
    expected_product_count: int,
    expected_complete_product_days: int,
    expected_grid_product_days: int,
    expected_missing_product_days: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    _require(
        coverage,
        {"Date", "ValueCode", "partition_complete"},
        "formal same-day coverage",
    )
    if coverage["partition_complete"].null_count():
        raise ValueError("formal same-day coverage has null completion status")
    complete = coverage.filter(
        pl.col("partition_complete") == True  # noqa: E712
    )
    missing = coverage.filter(
        pl.col("partition_complete") == False  # noqa: E712
    )
    expected_contract = {
        "selected_session_count": expected_sessions,
        "selected_product_count": expected_product_count,
        "selected_product_day_count": expected_complete_product_days,
        "expected_product_day_count": expected_grid_product_days,
        "missing_product_day_count": expected_missing_product_days,
    }
    for name, expected in expected_contract.items():
        value = metadata.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise ValueError(
                f"formal same-day coverage contract mismatch: {name}={value!r}, "
                f"expected {expected}"
            )
    if complete.height != expected_complete_product_days:
        raise ValueError(
            "formal same-day complete coverage is not the frozen product-day universe: "
            f"expected {expected_complete_product_days}, found {complete.height}"
        )
    if (
        coverage.height != expected_grid_product_days
        or missing.height != expected_missing_product_days
        or coverage.select("Date", "ValueCode").n_unique() != coverage.height
        or coverage["Date"].cast(pl.String).n_unique() != expected_sessions
        or coverage["ValueCode"].cast(pl.String).n_unique()
        != expected_product_count
    ):
        raise ValueError("formal same-day date-by-product coverage grid is inconsistent")
    return complete, missing


def _sensitivity_cost_expr(scenario: CostSensitivity) -> pl.Expr:
    return (
        pl.when(pl.col("completed_same_day"))
        .then(pl.lit(float(scenario.same_day_bp)))
        .otherwise(pl.lit(float(scenario.overnight_bp)))
    )


def _policy_grid() -> pl.DataFrame:
    return pl.from_dicts(
        [
            {
                "boundary_quantile": q,
                "entry_route": entry,
                "exit_rule_id": rule,
                "exit_route": exit_route,
            }
            for q in FROZEN_BOUNDARY_QUANTILES
            for entry in FROZEN_ENTRY_ROUTES
            for rule in FROZEN_EXIT_RULE_IDS
            for exit_route in FROZEN_EXIT_ROUTES
        ],
        infer_schema_length=None,
    )


def _policy_path_universe_sha256(paths: pl.DataFrame) -> str:
    columns = [
        "policy_path_id",
        "exit_policy_trial_id",
        "filled_entry_outcome_category",
        "terminal_date",
        "label_availability_date",
        "gross_cycle_pnl_twd",
        "normalization_notional_twd",
    ]
    return _canonical_frame_records_sha256(
        paths, columns=columns, sort_by=("policy_path_id",)
    )


def _canonical_frame_records_sha256(
    frame: pl.DataFrame,
    *,
    columns: Sequence[str],
    sort_by: Sequence[str],
) -> str:
    """Hash a canonical JSON list without a list/dict/bytes full-frame copy."""

    selected = frame.select(columns).sort(list(sort_by))
    digest = hashlib.sha256()
    digest.update(b"[")
    first = True
    for row in selected.iter_rows(named=True):
        if not first:
            digest.update(b",")
        first = False
        digest.update(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                default=str,
            ).encode("utf-8")
        )
    digest.update(b"]")
    return digest.hexdigest()


_ARTIFACTS: Mapping[str, str] = {
    "physical_policy_paths.parquet": "policy_paths",
    "policy_summary.parquet": "policy_summary",
    "product_policy_summary.parquet": "product_policy_summary",
    "daily_terminal_cashflows.parquet": "daily_terminal_cashflows",
    "daily_outstanding.parquet": "daily_outstanding",
    "prequential_policy_rankings.parquet": "prequential_rankings",
    "prequential_decisions.parquet": "prequential_decisions",
    "position_limit_sweep.parquet": "position_limit_sweep",
    "readiness.parquet": "readiness",
}


def _publish_evaluation(
    evaluation: PostCrossPositionEvaluation, destination: Path
) -> None:
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        artifacts: dict[str, dict[str, object]] = {}
        for filename, attribute in _ARTIFACTS.items():
            frame = getattr(evaluation, attribute)
            path = stage / filename
            frame.write_parquet(path)
            artifacts[filename] = {
                "rows": frame.height,
                "columns": frame.width,
                "schema": {name: str(dtype) for name, dtype in frame.schema.items()},
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        marker = {
            "complete": True,
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "evaluator_version": EVALUATOR_VERSION,
            "metadata": json.loads(
                json.dumps(dict(evaluation.metadata), sort_keys=True, default=str)
            ),
            "artifacts": artifacts,
        }
        (stage / "complete.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        stage.replace(destination)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _validate_output_disjoint(output: Path, sources: Sequence[Path]) -> None:
    target = output.resolve()
    for source in sources:
        resolved = source.resolve()
        if target == resolved or target in resolved.parents or resolved in target.parents:
            raise ValueError(
                f"output must be disjoint from every formal/source root: {output} vs {source}"
            )


def _implementation_sources() -> dict[str, str]:
    root = Path(__file__).parent
    names = (
        "post_cross_position_evaluator.py",
        "post_cross_position_cli.py",
        "filled_entry_report.py",
        "exit_maker_report.py",
        "exit_maker_study.py",
        "exit_maker_cross_session_runner.py",
        "cross_session_prerequisite.py",
        "finite_horizon_policy.py",
    )
    return {name: _file_sha256(root / name) for name in names}


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: object, name: str) -> str:
    rendered = str(value)
    if len(rendered) != 64 or any(ch not in "0123456789abcdef" for ch in rendered):
        raise ValueError(f"{name} must be a lower-case SHA-256 digest")
    return rendered


def _is_sha256(value: object) -> bool:
    rendered = str(value)
    return len(rendered) == 64 and all(
        character in "0123456789abcdef" for character in rendered
    )


def _finite(value: object) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _require(frame: pl.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def _parse_sensitivity(value: str) -> CostSensitivity:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "cost sensitivity must be ID:SAME_DAY_BP:OVERNIGHT_BP"
        )
    try:
        result = CostSensitivity(parts[0], float(parts[1]), float(parts[2]))
        result.validate()
        return result
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _parse_position_limit(value: str) -> PositionLimit:
    parts = value.split(":")
    if len(parts) not in {3, 4}:
        raise argparse.ArgumentTypeError(
            "position limit must be ID:SCOPE:MAX_POSITIONS_OR_NONE:MAX_NOTIONAL_OR_NONE"
        )
    try:
        if len(parts) == 3:
            identifier, count_value, notional_value = parts
            scope = "portfolio"
        else:
            identifier, scope, count_value, notional_value = parts
        count = None if count_value.lower() == "none" else int(count_value)
        notional = (
            None if notional_value.lower() == "none" else float(notional_value)
        )
        result = PositionLimit(identifier, count, notional, scope)
        result.validate()
        return result
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exit-root", type=Path, required=True)
    parser.add_argument("--entry-root", type=Path, required=True)
    parser.add_argument("--cross-session-root", type=Path, required=True)
    parser.add_argument("--prerequisite-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--full-cost-profile-root",
        type=Path,
        help=(
            "reserved for a future formal source-bound cost producer; any caller "
            "root is currently rejected"
        ),
    )
    parser.add_argument(
        "--cost-sensitivity",
        type=_parse_sensitivity,
        action="append",
        help="repeatable ID:SAME_DAY_BP:OVERNIGHT_BP; defaults to 0/0, 19/19, 19/34",
    )
    parser.add_argument(
        "--position-limit",
        type=_parse_position_limit,
        action="append",
        help=(
            "repeatable ID:SCOPE:MAX_POSITIONS_OR_NONE:MAX_NOTIONAL_TWD_OR_NONE; "
            "SCOPE is portfolio or per_value_code"
        ),
    )
    parser.add_argument("--lookback-sessions", type=int, default=60)
    parser.add_argument("--min-training-dates", type=int, default=20)
    parser.add_argument("--min-policy-origins", type=int, default=100)
    parser.add_argument(
        "--min-completed-cycles-for-diagnostic-rank", type=int, default=30
    )
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--minimum-lcb-bp", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = PostCrossEvaluationConfig(
        lookback_sessions=args.lookback_sessions,
        min_training_dates=args.min_training_dates,
        min_policy_origins=args.min_policy_origins,
        min_completed_cycles_for_diagnostic_rank=(
            args.min_completed_cycles_for_diagnostic_rank
        ),
        confidence_level=args.confidence_level,
        minimum_lcb_bp=args.minimum_lcb_bp,
    )
    result = run_post_cross_position_evaluation(
        exit_maker_root=args.exit_root,
        entry_execution_root=args.entry_root,
        cross_session_root=args.cross_session_root,
        prerequisite_root=args.prerequisite_root,
        output=args.output,
        full_cost_profile_root=args.full_cost_profile_root,
        config=config,
        cost_sensitivities=(
            tuple(args.cost_sensitivity)
            if args.cost_sensitivity
            else DEFAULT_COST_SENSITIVITIES
        ),
        position_limits=(
            tuple(args.position_limit)
            if args.position_limit
            else DEFAULT_POSITION_LIMITS
        ),
    )
    print(json.dumps(dict(result.metadata), sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
