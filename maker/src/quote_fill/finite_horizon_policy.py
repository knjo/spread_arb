"""Causal contextual EV lookup for frozen maker-policy alternatives.

The directly identified decision in the current replay is deliberately narrow:
at one *position-establishment* opportunity, choose one mutually-exclusive
frozen continuation policy (for example Center/Lower x exit route).  One row in
the terminal label population is one policy origin.  The same eventual terminal
cashflow must never be pasted onto every intraday observation; doing that would
length-weight long paths and manufacture an unsupported Bellman controller.

This module therefore provides four fail-closed layers:

* canonicalise aliases that are the same physical legal-tick action;
* join one origin action to one fixed-policy terminal path and preserve a
  no-fill transition as zero immediate price PnL plus an explicit continuation;
* build prequential Date < D lookups while retaining labels with
  ``label_end_date >= D`` (or null) as pending mass in the D denominator; and
* rank mutually-exclusive, legal actions subject to inventory and active-OCO
  constraints, falling back exact -> product -> peer -> global.  ``NO_TRADE``
  is permitted only while flat; open risk requires a priced WAIT baseline.

Only point-identified, fully costed four-leg paths receive an EV and a one-sided
date-clustered lower confidence bound.  Pending, censored, or unknown terminal
mass has no invented finite value bound.  ``audit_intraday_transition_readiness``
defines the additional common-clock/WAIT/next-state facts required before a
sequential intraday controller can be fitted; this v1 never calls the contextual
lookup an intraday switching policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
import hashlib
import json
import math
from statistics import NormalDist
from typing import Iterable, Mapping, Sequence

import polars as pl


CONTEXTUAL_ESTIMAND = "fixed_policy_choice_at_position_establishment_v1"
LOOKUP_SCHEMA_VERSION = "frozen_contextual_policy_lookup_v1"
TRANSITION_SCHEMA_VERSION = "intraday_common_clock_transition_contract_v1"
POOLED = "__POOLED__"

ORIGIN_EXECUTION_OUTCOMES = frozenset(
    {"fill", "no_fill", "unknown", "censored"}
)
TERMINAL_OUTCOME_STATUSES = frozenset({"known", "unknown", "censored"})
ACTION_EFFECTS = frozenset(
    {"select_policy", "submit_order", "cancel_order", "wait"}
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

ACTION_REQUIRED_COLUMNS = frozenset(
    {
        "policy_path_id",
        "action_alias",
        "decision_id",
        "candidate_set_id",
        "candidate_set_expected_count",
        "candidate_set_hash",
        "candidate_set_complete",
        "alternative_set_id",
        "physical_entry_id",
        "Date",
        "ValueCode",
        "QuoteCode",
        "decision_phase",
        "common_decision_clock_id",
        "decision_time_ns",
        "decision_event_sequence",
        "decision_row_index",
        "entry_route",
        "entry_q",
        "entry_basis_bucket",
        "locked_entry_state_bucket",
        "route",
        "legal_tick",
        "relative_tick_offset",
        "action_bucket",
        "policy_family",
        "lifecycle_policy_version",
        "queue_scenario",
        "intended_maker_quantity",
        "hedge_delay_ns",
        "remaining_minutes",
        "holding_age_minutes",
        "sessions_to_expiry",
        "state_bucket",
        "peer_group",
        "source_asof_date",
        "contains_target_day_outcome",
        "finite_horizon_sessions",
        "terminal_policy_id",
        "terminal_policy_executable",
        "cost_profile_id",
        "cost_profile_version",
        "cost_profile_hash",
        "cost_profile_source_asof_date",
        "cost_profile_contains_target_day_outcome",
        "legal_action",
    }
)

OUTCOME_REQUIRED_COLUMNS = frozenset(
    {
        "path_label_id",
        "policy_path_id",
        "alternative_set_id",
        "physical_entry_id",
        "Date",
        "ValueCode",
        "QuoteCode",
        "decision_phase",
        "entry_route",
        "entry_q",
        "entry_basis_bucket",
        "locked_entry_state_bucket",
        "route",
        "legal_tick",
        "relative_tick_offset",
        "action_bucket",
        "policy_family",
        "lifecycle_policy_version",
        "queue_scenario",
        "intended_maker_quantity",
        "hedge_delay_ns",
        "finite_horizon_sessions",
        "label_horizon_end_date",
        "terminal_policy_id",
        "terminal_policy_executable",
        "label_end_date",
        "origin_execution_outcome",
        "terminal_outcome_status",
        "terminal_branch",
        "terminal_fill_observed",
        "actual_four_leg_gross_bp",
        "actual_four_leg_complete",
        "gross_includes_observed_slippage",
        "immediate_price_pnl_bp",
        "continuation_gross_value_bp",
        "continuation_state_id",
        "continuation_state_bucket",
        "known_accrued_cost_bp",
        "terminal_cost_remainder_bp",
        "terminal_cost_remainder_known",
        "cost_profile_id",
        "cost_profile_version",
        "cost_profile_hash",
        "cost_profile_source_asof_date",
        "cost_profile_contains_target_day_outcome",
        "cost_components_complete",
        *COST_COMPONENT_COLUMNS,
    }
)

RANK_CONSTRAINT_COLUMNS = frozenset(
    {
        "action_effect",
        "inventory_position_units",
        "reserved_inventory_units",
        "action_inventory_delta_units",
        "inventory_min_units",
        "inventory_max_units",
        "active_oco_sibling_count",
        "replaces_active_oco",
        "is_risk_baseline",
    }
)

# A shared state/action clock is the minimum input for a future sequential
# controller.  Current fixed-policy terminal outcomes do not satisfy this
# contract merely because they contain per-session diagnostic observations.
INTRADAY_TRANSITION_REQUIRED_COLUMNS = frozenset(
    {
        "transition_id",
        "Date",
        "physical_entry_id",
        "shared_market_path_id",
        "decision_clock_id",
        "decision_time_ns",
        "decision_event_sequence",
        "decision_row_index",
        "state_id",
        "action_id",
        "action_effect",
        "entry_route",
        "entry_q",
        "entry_basis_bucket",
        "locked_entry_state_bucket",
        "route",
        "legal_tick",
        "relative_tick_offset",
        "lifecycle_policy_version",
        "queue_scenario",
        "intended_maker_quantity",
        "hedge_delay_ns",
        "remaining_minutes",
        "holding_age_minutes",
        "sessions_to_expiry",
        "state_bucket",
        "finite_horizon_sessions",
        "horizon_end_date",
        "terminal_policy_id",
        "terminal_policy_executable",
        "common_decision_clock",
        "complete_legal_action_set",
        "outcome_class",
        "immediate_price_pnl_bp",
        "known_accrued_cost_bp",
        "terminal_cost_remainder_bp",
        "terminal_cost_remainder_known",
        "cost_profile_id",
        "cost_profile_version",
        "cost_profile_hash",
        "cost_profile_source_asof_date",
        "cost_profile_contains_target_day_outcome",
        "cost_components_complete",
        *COST_COMPONENT_COLUMNS,
        "next_state_id",
        "next_Date",
        "next_physical_entry_id",
        "next_shared_market_path_id",
        "next_decision_clock_id",
        "next_decision_time_ns",
        "next_decision_event_sequence",
        "next_decision_row_index",
        "terminal_transition",
        "inventory_feasible",
        "oco_feasible",
        "sequential_policy_replay_complete",
    }
)

# Fields that make one candidate executable and replay-identifiable.  The set
# hash deliberately excludes aliases and the candidate-set metadata itself, so
# aliases cannot change it and the hash has no circular dependency.
CANDIDATE_ACTION_HASH_FIELDS: tuple[str, ...] = tuple(
    sorted(
        (
            ACTION_REQUIRED_COLUMNS
            | RANK_CONSTRAINT_COLUMNS
            | {"oco_group_id"}
        )
        - {
            "action_alias",
            "candidate_set_expected_count",
            "candidate_set_hash",
            "candidate_set_complete",
        }
    )
)

LOOKUP_DIMENSIONS: tuple[str, ...] = (
    "ValueCode",
    "peer_group",
    "decision_phase",
    "entry_route",
    "entry_q",
    "entry_basis_bucket",
    "locked_entry_state_bucket",
    "route",
    "legal_tick_bucket",
    "relative_tick_offset_bucket",
    "action_bucket",
    "policy_family",
    "lifecycle_policy_version",
    "queue_scenario",
    "intended_maker_quantity_bucket",
    "hedge_delay_bucket",
    "remaining_minutes_bucket",
    "holding_age_bucket",
    "sessions_to_expiry_bucket",
    "state_bucket",
    "finite_horizon_bucket",
    "terminal_policy_id",
    "cost_profile_hash",
)

FALLBACK_LEVELS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("exact", LOOKUP_DIMENSIONS),
    (
        "product_primary",
        (
            "ValueCode",
            "peer_group",
            "decision_phase",
            "entry_route",
            "entry_q",
            "entry_basis_bucket",
            "locked_entry_state_bucket",
            "route",
            "relative_tick_offset_bucket",
            "action_bucket",
            "policy_family",
            "lifecycle_policy_version",
            "queue_scenario",
            "intended_maker_quantity_bucket",
            "hedge_delay_bucket",
            "remaining_minutes_bucket",
            "holding_age_bucket",
            "sessions_to_expiry_bucket",
            "state_bucket",
            "finite_horizon_bucket",
            "terminal_policy_id",
            "cost_profile_hash",
        ),
    ),
    (
        "peer_primary",
        (
            "peer_group",
            "decision_phase",
            "entry_route",
            "entry_q",
            "entry_basis_bucket",
            "locked_entry_state_bucket",
            "route",
            "relative_tick_offset_bucket",
            "action_bucket",
            "policy_family",
            "lifecycle_policy_version",
            "queue_scenario",
            "intended_maker_quantity_bucket",
            "hedge_delay_bucket",
            "remaining_minutes_bucket",
            "holding_age_bucket",
            "sessions_to_expiry_bucket",
            "state_bucket",
            "finite_horizon_bucket",
            "terminal_policy_id",
            "cost_profile_hash",
        ),
    ),
    (
        "global_primary",
        (
            "decision_phase",
            "entry_route",
            "entry_q",
            "entry_basis_bucket",
            "locked_entry_state_bucket",
            "route",
            "relative_tick_offset_bucket",
            "action_bucket",
            "policy_family",
            "lifecycle_policy_version",
            "queue_scenario",
            "intended_maker_quantity_bucket",
            "hedge_delay_bucket",
            "remaining_minutes_bucket",
            "holding_age_bucket",
            "sessions_to_expiry_bucket",
            "state_bucket",
            "finite_horizon_bucket",
            "terminal_policy_id",
            "cost_profile_hash",
        ),
    ),
)


@dataclass(frozen=True)
class ContextualPolicyEVConfig:
    """Bucketing, support, and confidence contract for the lookup."""

    lookback_sessions: int = 60
    min_history_sessions: int = 40
    min_training_dates: int = 20
    min_policy_origins: int = 200
    min_terminal_fills: int = 30
    min_priced_terminals: int = 100
    min_label_maturity_coverage: float = 1.0
    max_unknown_rate: float = 0.0
    max_censor_rate: float = 0.0
    confidence_level: float = 0.95
    finite_horizon_sessions: int = 5
    embargo_sessions: int = 5
    production_like_window_sessions: int = 60
    shrinkage_kappa: float = 50.0
    min_shrinkage_child_origins: int = 30
    min_peer_global_origins: int = 500
    min_peer_global_products: int = 5
    remaining_minute_edges: tuple[float, ...] = (5.0, 15.0, 30.0, 60.0, 120.0)
    holding_age_minute_edges: tuple[float, ...] = (
        1.0,
        5.0,
        15.0,
        30.0,
        60.0,
        180.0,
    )
    expiry_session_edges: tuple[int, ...] = (1, 2, 4, 6, 11)
    legal_tick_bucket_width: int = 1
    estimator_version: str = LOOKUP_SCHEMA_VERSION

    def validate(self) -> None:
        if self.lookback_sessions <= 0:
            raise ValueError("lookback_sessions must be positive")
        if not 1 <= self.min_history_sessions <= self.lookback_sessions:
            raise ValueError(
                "min_history_sessions must be in [1, lookback_sessions]"
            )
        for name in (
            "min_training_dates",
            "min_policy_origins",
            "min_terminal_fills",
            "min_priced_terminals",
            "finite_horizon_sessions",
            "embargo_sessions",
            "production_like_window_sessions",
            "min_shrinkage_child_origins",
            "min_peer_global_origins",
            "min_peer_global_products",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in (
            "min_label_maturity_coverage",
            "max_unknown_rate",
            "max_censor_rate",
            "confidence_level",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0.5 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0.5, 1)")
        if self.finite_horizon_sessions <= 0:
            raise ValueError("finite_horizon_sessions must be positive")
        if self.embargo_sessions < self.finite_horizon_sessions:
            raise ValueError("embargo_sessions must cover finite_horizon_sessions")
        if self.production_like_window_sessions < self.min_history_sessions:
            raise ValueError(
                "production_like_window_sessions cannot be below burn-in"
            )
        if self.min_peer_global_origins < 500:
            raise ValueError("peer/global fallback requires at least 500 origins")
        if self.min_peer_global_products < 5:
            raise ValueError("peer/global fallback requires at least 5 products")
        if (
            not math.isfinite(float(self.shrinkage_kappa))
            or self.shrinkage_kappa <= 0
        ):
            raise ValueError("shrinkage_kappa must be finite and positive")
        _validate_edges(self.remaining_minute_edges, "remaining_minute_edges")
        _validate_edges(self.holding_age_minute_edges, "holding_age_minute_edges")
        _validate_edges(self.expiry_session_edges, "expiry_session_edges")
        if (
            isinstance(self.legal_tick_bucket_width, bool)
            or not isinstance(self.legal_tick_bucket_width, int)
            or self.legal_tick_bucket_width <= 0
        ):
            raise ValueError("legal_tick_bucket_width must be a positive integer")
        if not self.estimator_version:
            raise ValueError("estimator_version must be non-empty")


@dataclass(frozen=True)
class ActionRankingConfig:
    """Execution-time admission threshold for contextual policy choices."""

    minimum_lcb_bp: float = 0.0
    require_position_establishment_phase: bool = True

    def validate(self) -> None:
        if not math.isfinite(float(self.minimum_lcb_bp)):
            raise ValueError("minimum_lcb_bp must be finite")


@dataclass(frozen=True)
class ContextualPolicyRanking:
    """Action-level audit plus one decision-level selection/disposition row."""

    scored_actions: pl.DataFrame
    decisions: pl.DataFrame


def canonicalize_policy_actions(action_facts: pl.DataFrame) -> pl.DataFrame:
    """Collapse threshold aliases onto one physical fixed-policy action.

    ``policy_path_id`` is the canonical policy identity.  Aliases sharing that
    ID must agree on every state/action field.  Conversely, two different IDs
    cannot claim the same decision, legal tick, action bucket, and frozen
    policy family: such rows would double-count one physical alternative.
    """

    _require(action_facts, ACTION_REQUIRED_COLUMNS, "policy action facts")
    if action_facts.is_empty():
        return action_facts
    actions = action_facts.with_columns(
        *(
            pl.col(name).cast(pl.String)
            for name in (
                "policy_path_id",
                "action_alias",
                "decision_id",
                "candidate_set_id",
                "candidate_set_hash",
                "alternative_set_id",
                "physical_entry_id",
                "Date",
                "ValueCode",
                "QuoteCode",
                "decision_phase",
                "common_decision_clock_id",
                "entry_route",
                "entry_q",
                "entry_basis_bucket",
                "locked_entry_state_bucket",
                "route",
                "action_bucket",
                "policy_family",
                "lifecycle_policy_version",
                "queue_scenario",
                "state_bucket",
                "peer_group",
                "source_asof_date",
                "terminal_policy_id",
                "cost_profile_id",
                "cost_profile_version",
                "cost_profile_hash",
                "cost_profile_source_asof_date",
            )
        ),
        pl.col("candidate_set_expected_count").cast(pl.Int64, strict=True),
        pl.col("candidate_set_complete").cast(pl.Boolean, strict=True),
        pl.col("decision_time_ns").cast(pl.Int64, strict=True),
        pl.col("decision_event_sequence").cast(pl.Int64, strict=True),
        pl.col("decision_row_index").cast(pl.Int64, strict=True),
        pl.col("legal_tick").cast(pl.Int64, strict=True),
        pl.col("relative_tick_offset").cast(pl.Int64, strict=True),
        pl.col("intended_maker_quantity").cast(pl.Int64, strict=True),
        pl.col("hedge_delay_ns").cast(pl.Int64, strict=True),
        pl.col("remaining_minutes").cast(pl.Float64, strict=True),
        pl.col("holding_age_minutes").cast(pl.Float64, strict=True),
        pl.col("sessions_to_expiry").cast(pl.Int64, strict=True),
        pl.col("finite_horizon_sessions").cast(pl.Int64, strict=True),
        pl.col("legal_action").cast(pl.Boolean, strict=True),
        pl.col("contains_target_day_outcome").cast(pl.Boolean, strict=True),
        pl.col("terminal_policy_executable").cast(pl.Boolean, strict=True),
        pl.col("cost_profile_contains_target_day_outcome").cast(
            pl.Boolean, strict=True
        ),
    )
    if "action_effect" in actions.columns:
        actions = actions.with_columns(pl.col("action_effect").cast(pl.String))
    _validate_action_rows(actions)

    fixed = sorted(ACTION_REQUIRED_COLUMNS - {"action_alias"})
    optional_fixed = sorted(
        (RANK_CONSTRAINT_COLUMNS | {"oco_group_id"}) & set(actions.columns)
    )
    fixed.extend(name for name in optional_fixed if name not in fixed)
    records: list[dict[str, object]] = []
    for key, group in actions.group_by("policy_path_id", maintain_order=True):
        policy_path_id = key[0] if isinstance(key, tuple) else key
        ordered = group.sort("action_alias")
        for name in fixed:
            if not _series_all_equal(ordered[name]):
                raise ValueError(
                    f"aliases for policy_path_id {policy_path_id} disagree on {name}"
                )
        first = dict(ordered.row(0, named=True))
        aliases = sorted({str(value) for value in ordered["action_alias"].to_list()})
        first["action_alias"] = aliases[0]
        first["action_aliases"] = aliases
        first["action_alias_count"] = len(aliases)
        records.append(first)
    canonical = pl.from_dicts(records, infer_schema_length=None)
    _validate_candidate_sets(canonical)

    physical_identity = [
        "decision_id",
        "alternative_set_id",
        "decision_phase",
        "common_decision_clock_id",
        "decision_time_ns",
        "decision_event_sequence",
        "decision_row_index",
        "entry_route",
        "entry_q",
        "entry_basis_bucket",
        "locked_entry_state_bucket",
        "route",
        "legal_tick",
        "relative_tick_offset",
        "action_bucket",
        "policy_family",
        "lifecycle_policy_version",
        "queue_scenario",
        "intended_maker_quantity",
        "hedge_delay_ns",
        "remaining_minutes",
        "holding_age_minutes",
        "sessions_to_expiry",
        "state_bucket",
        "finite_horizon_sessions",
        "terminal_policy_id",
        "cost_profile_hash",
    ]
    duplicates = canonical.group_by(physical_identity).agg(
        pl.col("policy_path_id").n_unique().alias("_policy_ids")
    ).filter(pl.col("_policy_ids") > 1)
    if duplicates.height:
        raise ValueError(
            "physically identical legal-tick action is split across policy_path_id; "
            "canonicalize its aliases"
        )
    return canonical.sort(["Date", "decision_id", "policy_path_id"])


def bucket_policy_actions(
    canonical_actions: pl.DataFrame,
    config: ContextualPolicyEVConfig = ContextualPolicyEVConfig(),
) -> pl.DataFrame:
    """Attach deterministic remaining-time/age/expiry/legal-tick buckets."""

    config.validate()
    _require(canonical_actions, ACTION_REQUIRED_COLUMNS, "canonical actions")
    if canonical_actions.is_empty():
        return canonical_actions
    return canonical_actions.with_columns(
        pl.col("remaining_minutes")
        .map_elements(
            lambda value: _remaining_minutes_bucket(
                value, config.remaining_minute_edges
            ),
            return_dtype=pl.String,
        )
        .alias("remaining_minutes_bucket"),
        pl.col("holding_age_minutes")
        .map_elements(
            lambda value: _continuous_bucket(
                value, config.holding_age_minute_edges, "holding"
            ),
            return_dtype=pl.String,
        )
        .alias("holding_age_bucket"),
        pl.col("sessions_to_expiry")
        .map_elements(
            lambda value: _continuous_bucket(
                value, config.expiry_session_edges, "expiry"
            ),
            return_dtype=pl.String,
        )
        .alias("sessions_to_expiry_bucket"),
        pl.col("legal_tick")
        .map_elements(
            lambda value: _legal_tick_bucket(
                value, config.legal_tick_bucket_width
            ),
            return_dtype=pl.String,
        )
        .alias("legal_tick_bucket"),
        pl.col("relative_tick_offset")
        .map_elements(
            lambda value: f"relative_tick:{int(value)}",
            return_dtype=pl.String,
        )
        .alias("relative_tick_offset_bucket"),
        pl.col("intended_maker_quantity")
        .map_elements(
            lambda value: f"qty:{int(value)}", return_dtype=pl.String
        )
        .alias("intended_maker_quantity_bucket"),
        pl.col("hedge_delay_ns")
        .map_elements(
            lambda value: f"hedge_ns:{int(value)}", return_dtype=pl.String
        )
        .alias("hedge_delay_bucket"),
        pl.col("finite_horizon_sessions")
        .map_elements(
            lambda value: f"horizon:{int(value)}", return_dtype=pl.String
        )
        .alias("finite_horizon_bucket"),
    )


def build_contextual_policy_paths(
    action_facts: pl.DataFrame,
    terminal_path_facts: pl.DataFrame,
    config: ContextualPolicyEVConfig = ContextualPolicyEVConfig(),
) -> pl.DataFrame:
    """Join canonical origin actions to one mutually-exclusive terminal path.

    A known origin fill carries the actual eventual four-leg gross as immediate
    price PnL and zero continuation.  A known origin no-fill carries exactly
    zero immediate price PnL plus the same fixed policy's explicit continuation
    gross and next-state identity.  Unknown/censored paths retain null terminal
    value.  Non-price cost is subtracted once and only when its versioned
    profile is complete.
    """

    config.validate()
    actions = bucket_policy_actions(canonicalize_policy_actions(action_facts), config)
    if actions.filter(
        (pl.col("decision_phase") != "position_establishment")
        | (pl.col("holding_age_minutes").abs() > 1e-12)
    ).height:
        raise ValueError(
            "current terminal paths identify only position_establishment with "
            "zero holding age; intraday states require transition facts"
        )
    if actions.filter(
        pl.col("finite_horizon_sessions") != config.finite_horizon_sessions
    ).height:
        raise ValueError("action finite horizon disagrees with lookup configuration")
    _require(
        terminal_path_facts,
        OUTCOME_REQUIRED_COLUMNS,
        "fixed-policy terminal path facts",
    )
    if actions.is_empty() and terminal_path_facts.is_empty():
        return pl.DataFrame()
    if actions.is_empty() or terminal_path_facts.is_empty():
        raise ValueError("origin actions and terminal paths must both be present")

    outcomes = terminal_path_facts.with_columns(
        *(
            pl.col(name).cast(pl.String)
            for name in (
                "path_label_id",
                "policy_path_id",
                "alternative_set_id",
                "physical_entry_id",
                "Date",
                "ValueCode",
                "QuoteCode",
                "decision_phase",
                "entry_route",
                "entry_q",
                "entry_basis_bucket",
                "locked_entry_state_bucket",
                "route",
                "action_bucket",
                "policy_family",
                "lifecycle_policy_version",
                "queue_scenario",
                "label_horizon_end_date",
                "label_end_date",
                "terminal_policy_id",
                "origin_execution_outcome",
                "terminal_outcome_status",
                "terminal_branch",
                "continuation_state_id",
                "continuation_state_bucket",
                "cost_profile_id",
                "cost_profile_version",
                "cost_profile_hash",
                "cost_profile_source_asof_date",
            )
        ),
        pl.col("legal_tick").cast(pl.Int64, strict=True),
        pl.col("relative_tick_offset").cast(pl.Int64, strict=True),
        pl.col("intended_maker_quantity").cast(pl.Int64, strict=True),
        pl.col("hedge_delay_ns").cast(pl.Int64, strict=True),
        pl.col("finite_horizon_sessions").cast(pl.Int64, strict=True),
        *(
            pl.col(name).cast(pl.Float64, strict=True)
            for name in (
                "actual_four_leg_gross_bp",
                "immediate_price_pnl_bp",
                "continuation_gross_value_bp",
                "known_accrued_cost_bp",
                "terminal_cost_remainder_bp",
                *COST_COMPONENT_COLUMNS,
            )
        ),
        *(
            pl.col(name).cast(pl.Boolean, strict=True)
            for name in (
                "terminal_fill_observed",
                "actual_four_leg_complete",
                "gross_includes_observed_slippage",
                "terminal_policy_executable",
                "terminal_cost_remainder_known",
                "cost_profile_contains_target_day_outcome",
                "cost_components_complete",
            )
        ),
    )
    _validate_outcome_rows(outcomes)
    if outcomes.select("path_label_id").n_unique() != outcomes.height:
        raise ValueError("terminal paths duplicate path_label_id")
    if outcomes.select("policy_path_id").n_unique() != outcomes.height:
        raise ValueError("terminal paths must contain one label per policy_path_id")

    action_ids = set(actions["policy_path_id"].to_list())
    outcome_ids = set(outcomes["policy_path_id"].to_list())
    if missing := sorted(action_ids - outcome_ids):
        raise ValueError(f"origin actions are missing terminal paths: {missing[:5]}")
    if orphan := sorted(outcome_ids - action_ids):
        raise ValueError(f"terminal paths have no origin action: {orphan[:5]}")

    identity = [
        "alternative_set_id",
        "physical_entry_id",
        "Date",
        "ValueCode",
        "QuoteCode",
        "decision_phase",
        "entry_route",
        "entry_q",
        "entry_basis_bucket",
        "locked_entry_state_bucket",
        "route",
        "legal_tick",
        "relative_tick_offset",
        "action_bucket",
        "policy_family",
        "lifecycle_policy_version",
        "queue_scenario",
        "intended_maker_quantity",
        "hedge_delay_ns",
        "finite_horizon_sessions",
        "terminal_policy_id",
        "terminal_policy_executable",
        "cost_profile_id",
        "cost_profile_version",
        "cost_profile_hash",
        "cost_profile_source_asof_date",
        "cost_profile_contains_target_day_outcome",
    ]
    outcome_select = [
        "policy_path_id",
        *(
            pl.col(name).alias(f"_outcome_{name}")
            for name in identity
        ),
        *(
            name
            for name in outcomes.columns
            if name not in {"policy_path_id", *identity}
        ),
    ]
    joined = actions.join(
        outcomes.select(*outcome_select),
        on="policy_path_id",
        how="inner",
        validate="1:1",
    )
    mismatch = None
    for name in identity:
        term = (
            pl.col(name).cast(pl.String) != pl.col(f"_outcome_{name}").cast(pl.String)
        ).fill_null(True)
        mismatch = term if mismatch is None else mismatch | term
    assert mismatch is not None
    if joined.filter(mismatch).height:
        raise ValueError("origin action and terminal path physical identities disagree")

    joined = joined.drop(*(f"_outcome_{name}" for name in identity))
    component_complete = pl.all_horizontal(
        *(
            pl.col(name).is_finite() & (pl.col(name) >= 0)
            for name in COST_COMPONENT_COLUMNS
        )
    )
    full_cost_known = (
        component_complete
        & pl.col("cost_components_complete")
        & pl.col("terminal_cost_remainder_known")
    )
    joined = joined.with_columns(
        pl.when(full_cost_known)
        .then(pl.sum_horizontal(*(pl.col(name) for name in COST_COMPONENT_COLUMNS)))
        .otherwise(None)
        .alias("non_price_cost_bp"),
        component_complete.alias("observed_cost_components_finite"),
    )
    known = pl.col("terminal_outcome_status") == "known"
    priced = (
        known
        & pl.col("actual_four_leg_complete")
        & pl.col("gross_includes_observed_slippage")
        & pl.col("cost_components_complete")
        & pl.col("observed_cost_components_finite")
        & pl.col("terminal_cost_remainder_known")
        & pl.col("actual_four_leg_gross_bp").is_finite()
        & pl.col("non_price_cost_bp").is_finite()
    )
    result = joined.with_columns(
        priced.alias("terminal_cashflow_priced"),
        pl.when(priced)
        .then(
            pl.col("actual_four_leg_gross_bp") - pl.col("non_price_cost_bp")
        )
        .otherwise(None)
        .alias("realized_path_value_bp"),
        pl.when(priced)
        .then(
            pl.col("immediate_price_pnl_bp")
            + pl.col("continuation_gross_value_bp")
            - pl.col("non_price_cost_bp")
        )
        .otherwise(None)
        .alias("decomposed_path_value_bp"),
        pl.lit(CONTEXTUAL_ESTIMAND).alias("identified_decision_scope"),
        pl.lit(False).alias("intraday_policy_switching_identified"),
        pl.lit(True).alias("one_origin_per_fixed_policy_path"),
        pl.lit(False).alias("terminal_value_duplicated_across_observations"),
        pl.lit(True).alias("price_slippage_already_in_four_leg_gross"),
        pl.col("known_accrued_cost_bp").alias(
            "known_accrued_cost_retained_when_terminal_unresolved_bp"
        ),
        pl.when(pl.col("terminal_outcome_status") == "censored")
        .then(pl.lit("censored"))
        .when(pl.col("terminal_outcome_status") == "unknown")
        .then(pl.lit("unknown"))
        .otherwise(pl.col("origin_execution_outcome"))
        .alias("analysis_outcome_class"),
    )
    bad_decomposition = result.filter(
        pl.col("terminal_cashflow_priced")
        & (
            pl.col("realized_path_value_bp")
            - pl.col("decomposed_path_value_bp")
        )
        .abs()
        .gt(1e-9)
    )
    if bad_decomposition.height:
        raise ValueError("immediate plus continuation does not reconstruct path value")

    # One physical alternative in one exact contextual cell is one sampling
    # origin.  This catches alias/policy duplication before any fallback pool.
    exact_unit = list(
        dict.fromkeys(
            [
                "Date",
                "ValueCode",
                "QuoteCode",
                "physical_entry_id",
                "alternative_set_id",
                *LOOKUP_DIMENSIONS,
            ]
        )
    )
    if result.select(exact_unit).n_unique() != result.height:
        raise ValueError(
            "fixed-policy paths duplicate a physical alternative within an exact cell"
        )
    return result.sort(["Date", "alternative_set_id", "policy_path_id"])


def build_prequential_contextual_lookup(
    policy_paths: pl.DataFrame,
    sessions: Sequence[str],
    config: ContextualPolicyEVConfig = ContextualPolicyEVConfig(),
    *,
    asof_dates: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Build D-safe exact/product/peer/global fixed-policy EV tables.

    The admission cohort uses only ``Date < D``.  Every origin in the trailing
    Date window remains in the denominator.  Its label is visible only when
    ``label_end_date < D``; a null/future/same-day maturity is pending.  Thus a
    future terminal value cannot affect even the denominator classification,
    and unresolved mass prevents a point EV/LCB rather than being assigned 0.
    """

    config.validate()
    _require(
        policy_paths,
        {
            "path_label_id",
            "policy_path_id",
            "alternative_set_id",
            "physical_entry_id",
            "Date",
            "label_end_date",
            *LOOKUP_DIMENSIONS,
            "origin_execution_outcome",
            "analysis_outcome_class",
            "terminal_outcome_status",
            "terminal_fill_observed",
            "terminal_cashflow_priced",
            "realized_path_value_bp",
            "actual_four_leg_gross_bp",
            "immediate_price_pnl_bp",
            "continuation_gross_value_bp",
            "non_price_cost_bp",
            "known_accrued_cost_bp",
            "cost_components_complete",
            "terminal_policy_executable",
            "finite_horizon_sessions",
            "label_horizon_end_date",
        },
        "contextual policy paths",
    )
    if policy_paths.is_empty():
        return pl.DataFrame()
    calendar = _normalise_sessions(sessions)
    selected = _normalise_sessions(asof_dates or calendar)
    if missing := sorted(set(selected) - set(calendar)):
        raise ValueError(f"asof dates absent from session calendar: {missing[:5]}")
    path_dates = set(policy_paths["Date"].cast(pl.String).to_list())
    if missing := sorted(path_dates - set(calendar)):
        raise ValueError(f"policy path Dates absent from session calendar: {missing[:5]}")
    _validate_prepared_paths(policy_paths)
    _validate_finite_horizon_contract(policy_paths, calendar, config)

    index = {date: offset for offset, date in enumerate(calendar)}
    outputs: list[pl.DataFrame] = []
    for asof_date in selected:
        asof_index = index[asof_date]
        eligible_end = asof_index - config.embargo_sessions
        if eligible_end <= 0:
            continue
        prior_dates = calendar[
            max(0, eligible_end - config.lookback_sessions) : eligible_end
        ]
        if not prior_dates:
            continue
        cohort = policy_paths.filter(pl.col("Date").is_in(prior_dates))
        if cohort.is_empty():
            continue
        embargo_dates = calendar[eligible_end:asof_index]
        embargoed = policy_paths.filter(pl.col("Date").is_in(embargo_dates))
        level_outputs: list[pl.DataFrame] = []
        for level_index, (level_name, retained_keys) in enumerate(FALLBACK_LEVELS):
            summary = _aggregate_lookup_level(
                cohort,
                embargoed=embargoed,
                asof_date=asof_date,
                retained_keys=retained_keys,
                level_index=level_index,
                level_name=level_name,
                window_sessions=len(prior_dates),
                config=config,
            )
            if summary.height:
                level_outputs.append(
                    summary.with_columns(
                        pl.lit(prior_dates[0]).alias("window_start_date"),
                        pl.lit(prior_dates[-1]).alias("window_end_date"),
                        pl.lit(len(prior_dates)).alias("window_sessions"),
                        pl.lit(config.lookback_sessions).alias("lookback_sessions"),
                        pl.lit(config.finite_horizon_sessions).alias(
                            "finite_horizon_sessions"
                        ),
                        pl.lit(config.embargo_sessions).alias("embargo_sessions"),
                        pl.lit(calendar[eligible_end - 1]).alias(
                            "embargo_training_cutoff_date"
                        ),
                        pl.lit(
                            len(prior_dates)
                            >= config.production_like_window_sessions
                        ).alias("production_like_window"),
                        pl.lit(config.production_like_window_sessions).alias(
                            "production_like_required_sessions"
                        ),
                        pl.lit(config.estimator_version).alias("estimator_version"),
                        pl.lit(LOOKUP_SCHEMA_VERSION).alias("lookup_schema_version"),
                        pl.lit(CONTEXTUAL_ESTIMAND).alias("identified_decision_scope"),
                        pl.lit(True).alias("execution_safe_snapshot"),
                        pl.lit(False).alias("contains_target_day_outcome"),
                        pl.lit(True).alias("pending_origins_retained_in_denominator"),
                        pl.lit(False).alias("unresolved_mass_imputed_zero"),
                        pl.lit(False).alias("finite_unresolved_value_bound_assumed"),
                        pl.lit(False).alias("intraday_policy_switching_identified"),
                    )
                )
        if level_outputs:
            outputs.append(
                _attach_hierarchical_shrinkage(
                    pl.concat(level_outputs, how="diagonal_relaxed", rechunk=True),
                    config,
                )
            )
    if not outputs:
        return pl.DataFrame()
    result = pl.concat(outputs, how="diagonal_relaxed", rechunk=True)
    key = ["asof_date", "fallback_level", *LOOKUP_DIMENSIONS]
    if result.select(key).n_unique() != result.height:
        raise AssertionError("prequential lookup contains duplicate fallback keys")
    return result.sort(key)


def rank_contextual_policy_actions(
    decision_action_facts: pl.DataFrame,
    lookup: pl.DataFrame,
    ev_config: ContextualPolicyEVConfig = ContextualPolicyEVConfig(),
    ranking_config: ActionRankingConfig = ActionRankingConfig(),
) -> ContextualPolicyRanking:
    """Rank one policy; emit NO_TRADE only for a flat pre-entry admission."""

    ev_config.validate()
    ranking_config.validate()
    _require(
        decision_action_facts,
        ACTION_REQUIRED_COLUMNS | RANK_CONSTRAINT_COLUMNS,
        "decision action facts",
    )
    _require(
        lookup,
        {
            "asof_date",
            "fallback_level",
            "fallback_name",
            *LOOKUP_DIMENSIONS,
            "ev_ready",
            "ev_status",
            "posterior_ev_mean_bp",
            "posterior_one_sided_lcb_bp",
            "execution_safe_snapshot",
            "contains_target_day_outcome",
            "intraday_policy_switching_identified",
        },
        "contextual EV lookup",
    )
    if decision_action_facts.is_empty():
        return ContextualPolicyRanking(pl.DataFrame(), pl.DataFrame())
    if lookup.filter(
        ~pl.col("execution_safe_snapshot").fill_null(False)
        | pl.col("contains_target_day_outcome").fill_null(True)
        | pl.col("intraday_policy_switching_identified").fill_null(True)
    ).height:
        raise ValueError("lookup is not a safe fixed-policy prior snapshot")

    canonical = bucket_policy_actions(
        canonicalize_policy_actions(decision_action_facts), ev_config
    )
    _validate_ranking_constraints(canonical)
    _validate_decision_sets(canonical, ranking_config)

    lookup_key = ["asof_date", "fallback_level", *LOOKUP_DIMENSIONS]
    if lookup.select(lookup_key).n_unique() != lookup.height:
        raise ValueError("lookup contains duplicate fallback keys")
    lookup_rows: dict[tuple[object, ...], dict[str, object]] = {}
    for row in lookup.iter_rows(named=True):
        key = (
            str(row["asof_date"]),
            int(row["fallback_level"]),
            *(row[name] for name in LOOKUP_DIMENSIONS),
        )
        lookup_rows[key] = row

    scored_records: list[dict[str, object]] = []
    lookup_stat_columns = [
        name
        for name in lookup.columns
        if name not in {"asof_date", *LOOKUP_DIMENSIONS}
    ]
    for action in canonical.iter_rows(named=True):
        projected = (
            int(action["inventory_position_units"])
            + int(action["reserved_inventory_units"])
            + int(action["action_inventory_delta_units"])
        )
        inventory_ok = (
            int(action["inventory_min_units"])
            <= projected
            <= int(action["inventory_max_units"])
        )
        effect = str(action["action_effect"])
        active_siblings = int(action["active_oco_sibling_count"])
        oco_ok = (
            effect not in {"submit_order", "select_policy"}
            or active_siblings == 0
            or action["replaces_active_oco"] is True
        )
        legal = action["legal_action"] is True
        constraints_ok = legal and inventory_ok and oco_ok

        ready_match: dict[str, object] | None = None
        diagnostic_match: dict[str, object] | None = None
        for level_index, (_, retained) in enumerate(FALLBACK_LEVELS):
            values = tuple(
                action[name] if name in retained else POOLED
                for name in LOOKUP_DIMENSIONS
            )
            candidate = lookup_rows.get((str(action["Date"]), level_index, *values))
            if candidate is None:
                continue
            if diagnostic_match is None:
                diagnostic_match = candidate
            if candidate["ev_ready"] is True:
                ready_match = candidate
                break
        chosen = ready_match or diagnostic_match
        record = dict(action)
        record.update(
            {
                "projected_inventory_units": projected,
                "inventory_constraint_satisfied": inventory_ok,
                "oco_constraint_satisfied": oco_ok,
                "all_execution_constraints_satisfied": constraints_ok,
                "lookup_found": chosen is not None,
                "ready_lookup_found": ready_match is not None,
                "lookup_fallback_name": (
                    None if chosen is None else chosen["fallback_name"]
                ),
                "lookup_fallback_level": (
                    None if chosen is None else chosen["fallback_level"]
                ),
                "lookup_ev_ready": (
                    False if ready_match is None else bool(ready_match["ev_ready"])
                ),
                "lookup_ev_status": (
                    "missing_lookup" if chosen is None else chosen["ev_status"]
                ),
                "action_rank": None,
                "selected": False,
            }
        )
        if chosen is not None:
            for name in lookup_stat_columns:
                if name in {"fallback_level", "fallback_name"}:
                    continue
                record[f"lookup_{name}"] = chosen[name]
        record["ranking_score_lcb_bp"] = (
            None
            if ready_match is None
            else ready_match["posterior_one_sided_lcb_bp"]
        )
        record["ranking_expected_ev_bp"] = (
            None if ready_match is None else ready_match["posterior_ev_mean_bp"]
        )
        scored_records.append(record)

    scored = pl.from_dicts(scored_records, infer_schema_length=None)
    ranks: dict[str, tuple[int, bool]] = {}
    decision_records: list[dict[str, object]] = []
    for key, group in scored.group_by("decision_id", maintain_order=True):
        decision_id = str(key[0] if isinstance(key, tuple) else key)
        inventory_position = int(group.item(0, "inventory_position_units"))
        eligible = group.filter(
            pl.col("all_execution_constraints_satisfied")
            & pl.col("lookup_ev_ready")
            & pl.col("ranking_score_lcb_bp").is_finite()
        ).sort(
            [
                "ranking_score_lcb_bp",
                "ranking_expected_ev_bp",
                "lookup_fallback_level",
                "policy_path_id",
            ],
            descending=[True, True, False, False],
        )
        selected_id: str | None = None
        selected_row: Mapping[str, object] | None = None
        if inventory_position != 0:
            baseline = group.filter(pl.col("is_risk_baseline"))
            if baseline.height != 1:
                raise ValueError(
                    "open inventory requires exactly one explicit risk baseline"
                )
            baseline_row = baseline.row(0, named=True)
            if baseline_row["action_effect"] != "wait":
                raise ValueError("open-inventory risk baseline must be a priced WAIT")
            if not (
                baseline_row["all_execution_constraints_satisfied"] is True
                and baseline_row["lookup_ev_ready"] is True
                and _optional_finite(baseline_row["ranking_score_lcb_bp"])
                is not None
            ):
                raise ValueError(
                    "open inventory has no executable, priced WAIT baseline"
                )
        for rank, row in enumerate(eligible.iter_rows(named=True), 1):
            policy_path_id = str(row["policy_path_id"])
            choose = rank == 1 and (
                inventory_position != 0
                or float(row["ranking_score_lcb_bp"])
                >= float(ranking_config.minimum_lcb_bp)
            )
            ranks[policy_path_id] = (rank, choose)
            if choose:
                selected_id = policy_path_id
                selected_row = row
        if selected_row is not None:
            disposition = (
                "selected_open_inventory_risk_policy"
                if inventory_position != 0
                else "selected_contextual_fixed_policy"
            )
        elif eligible.height:
            disposition = "no_trade_lcb_below_threshold"
        elif group.filter(pl.col("lookup_ev_ready")).height:
            disposition = "no_trade_execution_constraints"
        else:
            disposition = "no_trade_no_ready_lookup"
        first = group.row(0, named=True)
        decision_records.append(
            {
                "Date": first["Date"],
                "decision_id": decision_id,
                "alternative_set_id": first["alternative_set_id"],
                "physical_entry_id": first["physical_entry_id"],
                "ValueCode": first["ValueCode"],
                "QuoteCode": first["QuoteCode"],
                "decision_phase": first["decision_phase"],
                "inventory_position_units": inventory_position,
                "candidate_action_count": group.height,
                "constraint_eligible_action_count": group.filter(
                    pl.col("all_execution_constraints_satisfied")
                ).height,
                "ready_action_count": eligible.height,
                "selected_policy_path_id": selected_id,
                "selected_policy_family": (
                    None if selected_row is None else selected_row["policy_family"]
                ),
                "selected_route": (
                    None if selected_row is None else selected_row["route"]
                ),
                "selected_lcb_bp": (
                    None
                    if selected_row is None
                    else selected_row["ranking_score_lcb_bp"]
                ),
                "decision_disposition": disposition,
                "no_trade": selected_row is None,
                "mutually_exclusive_single_selection": True,
                "intraday_policy_switching_identified": False,
            }
        )

    if ranks:
        rank_frame = pl.from_dicts(
            [
                {
                    "policy_path_id": policy_path_id,
                    "action_rank": values[0],
                    "selected": values[1],
                }
                for policy_path_id, values in ranks.items()
            ],
            infer_schema_length=None,
        )
        scored = scored.drop("action_rank", "selected").join(
            rank_frame,
            on="policy_path_id",
            how="left",
            validate="1:1",
        ).with_columns(
            pl.col("action_rank").cast(pl.Int64),
            pl.col("selected").fill_null(False),
        )
    else:
        scored = scored.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("action_rank"),
            pl.lit(False).alias("selected"),
        )
    if scored.filter(pl.col("selected")).group_by("decision_id").len().filter(
        pl.col("len") > 1
    ).height:
        raise AssertionError("more than one mutually-exclusive action was selected")
    decisions = pl.from_dicts(decision_records, infer_schema_length=None).sort(
        ["Date", "decision_id"]
    )
    return ContextualPolicyRanking(
        scored.sort(["Date", "decision_id", "action_rank", "policy_path_id"]),
        decisions,
    )


def audit_intraday_transition_readiness(
    transition_facts: pl.DataFrame | None,
) -> pl.DataFrame:
    """Audit whether common-clock transitions exist for a future controller.

    Passing this audit means the *input transition table* is structurally ready
    for a sequential replay/estimator.  It does not retroactively turn fixed
    terminal paths into decision-level labels, and this module still reports
    ``intraday_controller_ready=False`` because no Bellman/sequential policy
    fit is implemented here.
    """

    base: dict[str, object] = {
        "transition_schema_version": TRANSITION_SCHEMA_VERSION,
        "transition_facts_supplied": transition_facts is not None,
        "transition_data_ready": False,
        "intraday_controller_ready": False,
        "contextual_fixed_policy_lookup_ready_in_principle": True,
        "current_terminal_paths_can_be_reused_per_observation": False,
        "missing_columns": sorted(INTRADAY_TRANSITION_REQUIRED_COLUMNS),
        "readiness_status": "missing_transition_facts",
        "transition_rows": 0,
        "decision_states": 0,
        "decision_dates": 0,
    }
    if transition_facts is None or transition_facts.is_empty():
        return pl.from_dicts([base], infer_schema_length=None)
    missing = sorted(
        INTRADAY_TRANSITION_REQUIRED_COLUMNS - set(transition_facts.columns)
    )
    base.update(
        transition_facts_supplied=True,
        missing_columns=missing,
        transition_rows=transition_facts.height,
    )
    if missing:
        base["readiness_status"] = "missing_required_transition_columns"
        return pl.from_dicts([base], infer_schema_length=None)

    facts = transition_facts
    state_keys = [
        "Date",
        "physical_entry_id",
        "shared_market_path_id",
        "decision_clock_id",
        "state_id",
    ]
    base["decision_states"] = facts.select(state_keys).n_unique()
    base["decision_dates"] = facts["Date"].n_unique()
    if facts.select("transition_id").n_unique() != facts.height:
        base["readiness_status"] = "duplicate_transition_identity"
        return pl.from_dicts([base], infer_schema_length=None)
    if facts.select([*state_keys, "action_id"]).n_unique() != facts.height:
        base["readiness_status"] = "duplicate_state_action_transition"
        return pl.from_dicts([base], infer_schema_length=None)
    if facts.filter(
        ~pl.col("common_decision_clock").fill_null(False)
        | ~pl.col("complete_legal_action_set").fill_null(False)
    ).height:
        base["readiness_status"] = "incomplete_or_noncommon_decision_clock"
        return pl.from_dicts([base], infer_schema_length=None)
    if facts.filter(
        ~pl.col("sequential_policy_replay_complete").fill_null(False)
    ).height:
        base["readiness_status"] = "sequential_policy_replay_incomplete"
        return pl.from_dicts([base], infer_schema_length=None)
    state_context = [
        "decision_time_ns",
        "decision_event_sequence",
        "decision_row_index",
        "entry_route",
        "entry_q",
        "entry_basis_bucket",
        "locked_entry_state_bucket",
        "remaining_minutes",
        "holding_age_minutes",
        "sessions_to_expiry",
        "state_bucket",
        "finite_horizon_sessions",
        "horizon_end_date",
        "terminal_policy_id",
        "terminal_policy_executable",
    ]
    inconsistent = facts.group_by(state_keys).agg(
        *(pl.col(name).n_unique().alias(name) for name in state_context)
    ).filter(pl.any_horizontal(*(pl.col(name) != 1 for name in state_context)))
    if inconsistent.height:
        base["readiness_status"] = "state_actions_disagree_on_full_decision_context"
        return pl.from_dicts([base], infer_schema_length=None)
    replay_fixed = (
        "finite_horizon_sessions",
        "horizon_end_date",
        "terminal_policy_id",
        "terminal_policy_executable",
    )
    inconsistent_replay = facts.group_by(
        ["physical_entry_id", "shared_market_path_id"]
    ).agg(*(pl.col(name).n_unique().alias(name) for name in replay_fixed)).filter(
        pl.any_horizontal(*(pl.col(name) != 1 for name in replay_fixed))
    )
    if inconsistent_replay.height:
        base["readiness_status"] = "replay_changes_finite_horizon_or_terminal_policy"
        return pl.from_dicts([base], infer_schema_length=None)
    wait_counts = facts.group_by(state_keys).agg(
        (pl.col("action_effect") == "wait").sum().alias("_wait_count")
    )
    if wait_counts.filter(pl.col("_wait_count") != 1).height:
        base["readiness_status"] = "each_state_requires_exactly_one_wait_action"
        return pl.from_dicts([base], infer_schema_length=None)
    invalid_outcome = facts.filter(
        ~pl.col("outcome_class")
        .is_in(sorted(ORIGIN_EXECUTION_OUTCOMES))
        .fill_null(False)
    )
    if invalid_outcome.height:
        base["readiness_status"] = "invalid_transition_outcome_class"
        return pl.from_dicts([base], infer_schema_length=None)
    if facts.filter(
        ~pl.col("action_effect").is_in(sorted(ACTION_EFFECTS)).fill_null(False)
    ).height:
        base["readiness_status"] = "invalid_transition_action_effect"
        return pl.from_dicts([base], infer_schema_length=None)
    if facts.filter(
        ~pl.col("inventory_feasible").fill_null(False)
        | ~pl.col("oco_feasible").fill_null(False)
    ).height:
        base["readiness_status"] = "transition_contains_infeasible_action"
        return pl.from_dicts([base], infer_schema_length=None)
    next_fields = (
        "next_state_id",
        "next_Date",
        "next_physical_entry_id",
        "next_shared_market_path_id",
        "next_decision_clock_id",
        "next_decision_time_ns",
        "next_decision_event_sequence",
        "next_decision_row_index",
    )
    state_rows: dict[tuple[str, str, str, str, str], Mapping[str, object]] = {}
    has_unresolved_transition = False
    for row in facts.iter_rows(named=True):
        try:
            date = _date(row["Date"], "Date")
            horizon_end = _date(row["horizon_end_date"], "horizon_end_date")
            if horizon_end < date:
                raise ValueError("horizon_end_date precedes transition Date")
            _positive_int(
                row["finite_horizon_sessions"], "finite_horizon_sessions"
            )
            current_cursor = (
                _non_negative_int(row["decision_time_ns"], "decision_time_ns"),
                _non_negative_int(
                    row["decision_event_sequence"], "decision_event_sequence"
                ),
                _non_negative_int(row["decision_row_index"], "decision_row_index"),
            )
            _non_negative_int(row["legal_tick"], "legal_tick")
            _strict_int(row["relative_tick_offset"], "relative_tick_offset")
            _positive_int(
                row["intended_maker_quantity"], "intended_maker_quantity"
            )
            _non_negative_int(row["hedge_delay_ns"], "hedge_delay_ns")
            _non_negative_float(row["remaining_minutes"], "remaining_minutes")
            _non_negative_float(
                row["holding_age_minutes"], "holding_age_minutes"
            )
            _non_negative_int(row["sessions_to_expiry"], "sessions_to_expiry")
            if not str(row["terminal_policy_id"]).strip():
                raise ValueError("missing terminal policy")
            if row["terminal_policy_executable"] is not True:
                raise ValueError("terminal policy is not executable")
            cost_source = _date(
                row["cost_profile_source_asof_date"],
                "cost_profile_source_asof_date",
            )
            if cost_source >= date:
                raise ValueError("transition cost profile is not prequential")
            if row["cost_profile_contains_target_day_outcome"] is not False:
                raise ValueError("transition cost profile contains target outcome")
            for name in (
                "cost_profile_id",
                "cost_profile_version",
                "entry_route",
                "entry_q",
                "entry_basis_bucket",
                "locked_entry_state_bucket",
                "lifecycle_policy_version",
                "queue_scenario",
                "transition_id",
                "physical_entry_id",
                "shared_market_path_id",
                "decision_clock_id",
                "state_id",
                "action_id",
                "route",
                "state_bucket",
            ):
                if row[name] is None or not str(row[name]).strip():
                    raise ValueError(f"{name} must be non-empty")
            _validate_sha256(row["cost_profile_hash"], "cost_profile_hash")
            accrued = _non_negative_float(
                row["known_accrued_cost_bp"], "known_accrued_cost_bp"
            )
            if row["outcome_class"] == "no_fill":
                no_fill_immediate = _optional_finite(
                    row["immediate_price_pnl_bp"]
                )
                if no_fill_immediate is None or abs(no_fill_immediate) > 1e-12:
                    base["readiness_status"] = (
                        "no_fill_transition_loses_zero_pnl_continuation"
                    )
                    return pl.from_dicts([base], infer_schema_length=None)
            components = [
                None
                if row[name] is None
                else _non_negative_float(row[name], name)
                for name in COST_COMPONENT_COLUMNS
            ]
            if row["outcome_class"] in {"unknown", "censored"}:
                observed = [value for value in components if value is not None]
                if (
                    row["cost_components_complete"] is True
                    or row["terminal_cost_remainder_known"] is True
                    or row["terminal_cost_remainder_bp"] is not None
                    or (observed and abs(sum(observed) - accrued) > 1e-9)
                ):
                    raise ValueError(
                        "unresolved transition misstates unknown terminal cost"
                    )
                has_unresolved_transition = True
            else:
                immediate = _optional_finite(row["immediate_price_pnl_bp"])
                if immediate is None:
                    raise ValueError("transition lacks finite immediate price PnL")
                if row["outcome_class"] == "no_fill" and abs(immediate) > 1e-12:
                    base["readiness_status"] = (
                        "no_fill_transition_loses_zero_pnl_continuation"
                    )
                    return pl.from_dicts([base], infer_schema_length=None)
                remainder = _optional_finite(row["terminal_cost_remainder_bp"])
                if (
                    row["cost_components_complete"] is not True
                    or row["terminal_cost_remainder_known"] is not True
                    or remainder is None
                    or remainder < 0
                    or any(value is None for value in components)
                    or abs(
                        sum(value for value in components if value is not None)
                        - (accrued + remainder)
                    )
                    > 1e-9
                ):
                    raise ValueError("incomplete component-level transition cost")
            if not isinstance(row["terminal_transition"], bool):
                raise ValueError("terminal_transition must be boolean")
        except (ArithmeticError, TypeError, ValueError):
            base["readiness_status"] = "invalid_transition_context_or_cost_contract"
            return pl.from_dicts([base], infer_schema_length=None)
        key = (
            date,
            str(row["physical_entry_id"]),
            str(row["shared_market_path_id"]),
            str(row["decision_clock_id"]),
            str(row["state_id"]),
        )
        state_rows.setdefault(key, row)

        terminal = row["terminal_transition"] is True
        present = [row[name] is not None for name in next_fields]
        if terminal and any(present):
            base["readiness_status"] = "terminal_transition_must_have_null_next_tuple"
            return pl.from_dicts([base], infer_schema_length=None)
        if not terminal and not all(present):
            base["readiness_status"] = "nonterminal_transition_missing_full_next_tuple"
            return pl.from_dicts([base], infer_schema_length=None)
        if terminal:
            if row["outcome_class"] == "no_fill":
                base["readiness_status"] = (
                    "terminal_no_fill_violates_executable_terminal_policy"
                )
                return pl.from_dicts([base], infer_schema_length=None)
            continue
        try:
            next_date = _date(row["next_Date"], "next_Date")
            next_cursor = (
                _non_negative_int(
                    row["next_decision_time_ns"], "next_decision_time_ns"
                ),
                _non_negative_int(
                    row["next_decision_event_sequence"],
                    "next_decision_event_sequence",
                ),
                _non_negative_int(
                    row["next_decision_row_index"], "next_decision_row_index"
                ),
            )
        except (TypeError, ValueError):
            base["readiness_status"] = "invalid_next_decision_cursor"
            return pl.from_dicts([base], infer_schema_length=None)
        if next_date < date or (next_date == date and next_cursor <= current_cursor):
            base["readiness_status"] = "next_decision_cursor_is_not_strictly_forward"
            return pl.from_dicts([base], infer_schema_length=None)
        if next_date > horizon_end:
            base["readiness_status"] = "next_decision_exceeds_finite_horizon"
            return pl.from_dicts([base], infer_schema_length=None)
        if (
            str(row["next_physical_entry_id"]) != str(row["physical_entry_id"])
            or str(row["next_shared_market_path_id"])
            != str(row["shared_market_path_id"])
        ):
            base["readiness_status"] = "next_transition_changes_replay_identity"
            return pl.from_dicts([base], infer_schema_length=None)

    for row in facts.iter_rows(named=True):
        if row["terminal_transition"] is True:
            continue
        next_key = (
            str(row["next_Date"]),
            str(row["next_physical_entry_id"]),
            str(row["next_shared_market_path_id"]),
            str(row["next_decision_clock_id"]),
            str(row["next_state_id"]),
        )
        target = state_rows.get(next_key)
        if target is None:
            base["readiness_status"] = "exact_next_state_absent_from_transition_table"
            return pl.from_dicts([base], infer_schema_length=None)
        for next_name, current_name in (
            ("next_decision_time_ns", "decision_time_ns"),
            ("next_decision_event_sequence", "decision_event_sequence"),
            ("next_decision_row_index", "decision_row_index"),
        ):
            if row[next_name] != target[current_name]:
                base["readiness_status"] = "next_cursor_disagrees_with_joined_state"
                return pl.from_dicts([base], infer_schema_length=None)
    if has_unresolved_transition:
        base["readiness_status"] = "unresolved_transition_cashflow"
        return pl.from_dicts([base], infer_schema_length=None)
    base.update(
        transition_data_ready=True,
        intraday_controller_ready=False,
        missing_columns=[],
        readiness_status="transition_data_ready_sequential_estimator_not_implemented",
    )
    return pl.from_dicts([base], infer_schema_length=None)


def audit_current_fact_adapter_contract(
    *,
    action_columns: Iterable[str] = (),
    terminal_path_columns: Iterable[str] = (),
    transition_columns: Iterable[str] = (),
) -> pl.DataFrame:
    """Report whether upstream artifacts can populate the normalized schemas.

    This is intentionally a schema/lineage contract, not a best-effort rename
    adapter.  Legacy scalar ``non_price_cost_bp`` fields, diagnostic order-book
    observations, or fixed terminal labels cannot synthesize the missing
    component costs, common clocks, candidate alternatives, or transitions.
    """

    action_available = set(action_columns)
    terminal_available = set(terminal_path_columns)
    transition_available = set(transition_columns)
    missing_action = sorted(ACTION_REQUIRED_COLUMNS - action_available)
    missing_terminal = sorted(OUTCOME_REQUIRED_COLUMNS - terminal_available)
    missing_transition = sorted(
        INTRADAY_TRANSITION_REQUIRED_COLUMNS - transition_available
    )
    return pl.from_dicts(
        [
            {
                "adapter_contract_version": "finite_horizon_fact_adapter_v1",
                "fixed_policy_adapter_ready": not missing_action
                and not missing_terminal,
                "intraday_transition_adapter_ready": not missing_transition,
                "intraday_controller_implemented": False,
                "missing_action_fields": missing_action,
                "missing_terminal_path_fields": missing_terminal,
                "missing_transition_fields": missing_transition,
                "legacy_scalar_cost_is_sufficient": False,
                "fixed_terminal_path_can_be_expanded_per_observation": False,
                "requires_component_cost_profile_lineage": True,
                "requires_complete_candidate_set_identity": True,
                "requires_common_clock_wait_and_exact_next_state": True,
            }
        ],
        infer_schema_length=None,
    )


def config_payload(config: ContextualPolicyEVConfig) -> dict[str, object]:
    """Stable JSON-compatible configuration payload for atomic publishers."""

    config.validate()
    payload = asdict(config)
    return {key: list(value) if isinstance(value, tuple) else value for key, value in payload.items()}


def _aggregate_lookup_level(
    cohort: pl.DataFrame,
    *,
    embargoed: pl.DataFrame,
    asof_date: str,
    retained_keys: Sequence[str],
    level_index: int,
    level_name: str,
    window_sessions: int,
    config: ContextualPolicyEVConfig,
) -> pl.DataFrame:
    keys = list(retained_keys)
    mature = pl.col("label_end_date").is_not_null() & (
        pl.col("label_end_date") < asof_date
    )
    priced = mature & pl.col("terminal_cashflow_priced").fill_null(False)
    outcome = pl.col("analysis_outcome_class")
    unit_keys = list(
        dict.fromkeys(
            [
                *keys,
                "Date",
                "ValueCode",
                "QuoteCode",
                "physical_entry_id",
                "alternative_set_id",
            ]
        )
    )
    units = cohort.group_by(unit_keys).agg(
        pl.len().alias("_unit_policy_paths"),
        mature.sum().alias("_unit_matured_paths"),
        (~mature).sum().alias("_unit_pending_paths"),
        (mature & (outcome == "fill")).sum().alias("_unit_fill_paths"),
        (mature & (outcome == "no_fill")).sum().alias("_unit_no_fill_paths"),
        (mature & (outcome == "unknown")).sum().alias("_unit_unknown_paths"),
        (mature & (outcome == "censored")).sum().alias("_unit_censored_paths"),
        priced.sum().alias("_unit_priced_paths"),
        (priced & pl.col("terminal_fill_observed"))
        .any()
        .alias("_unit_has_terminal_fill"),
        pl.col("terminal_policy_executable").all().alias(
            "_unit_terminal_policy_executable"
        ),
        pl.col("cost_components_complete")
        .filter(mature)
        .all()
        .fill_null(False)
        .alias("_unit_cost_components_complete"),
        pl.col("path_label_id").n_unique().alias("_unit_path_labels"),
        pl.col("realized_path_value_bp").filter(priced).sum().alias(
            "_unit_value_sum"
        ),
        pl.col("actual_four_leg_gross_bp").filter(priced).sum().alias(
            "_unit_gross_sum"
        ),
        pl.col("immediate_price_pnl_bp").filter(priced).sum().alias(
            "_unit_immediate_sum"
        ),
        pl.col("continuation_gross_value_bp").filter(priced).sum().alias(
            "_unit_continuation_sum"
        ),
        pl.col("non_price_cost_bp").filter(priced).sum().alias("_unit_cost_sum"),
        pl.col("known_accrued_cost_bp").filter(mature).mean().alias(
            "_unit_known_accrued_cost_mean"
        ),
        pl.col("continuation_gross_value_bp")
        .filter(priced & (pl.col("origin_execution_outcome") == "no_fill"))
        .mean()
        .alias("_unit_no_fill_continuation_mean"),
        pl.col("label_end_date").filter(mature).max().alias("_unit_label_cutoff"),
    ).with_columns(
        (pl.col("_unit_matured_paths") / pl.col("_unit_policy_paths")).alias(
            "_unit_maturity_share"
        ),
        (pl.col("_unit_pending_paths") / pl.col("_unit_policy_paths")).alias(
            "_unit_pending_share"
        ),
        (pl.col("_unit_fill_paths") / pl.col("_unit_policy_paths")).alias(
            "_unit_fill_share"
        ),
        (pl.col("_unit_no_fill_paths") / pl.col("_unit_policy_paths")).alias(
            "_unit_no_fill_share"
        ),
        (pl.col("_unit_unknown_paths") / pl.col("_unit_policy_paths")).alias(
            "_unit_unknown_share"
        ),
        (pl.col("_unit_censored_paths") / pl.col("_unit_policy_paths")).alias(
            "_unit_censored_share"
        ),
        (
            (pl.col("_unit_priced_paths") == pl.col("_unit_policy_paths"))
            & (pl.col("_unit_pending_paths") == 0)
            & (pl.col("_unit_unknown_paths") == 0)
            & (pl.col("_unit_censored_paths") == 0)
        ).alias("_unit_point_identified"),
        (pl.col("_unit_priced_paths") > 0).alias("_unit_has_priced_terminal"),
        pl.when(pl.col("_unit_priced_paths") > 0)
        .then(pl.col("_unit_value_sum") / pl.col("_unit_priced_paths"))
        .otherwise(None)
        .alias("_unit_conditional_value"),
    ).with_columns(
        pl.when(pl.col("_unit_point_identified"))
        .then(pl.col("_unit_value_sum") / pl.col("_unit_policy_paths"))
        .otherwise(None)
        .alias("_unit_identified_value"),
        *(
            pl.when(pl.col("_unit_point_identified"))
            .then(pl.col(source) / pl.col("_unit_policy_paths"))
            .otherwise(None)
            .alias(target)
            for source, target in (
                ("_unit_gross_sum", "_unit_identified_gross"),
                ("_unit_immediate_sum", "_unit_identified_immediate"),
                ("_unit_continuation_sum", "_unit_identified_continuation"),
                ("_unit_cost_sum", "_unit_identified_cost"),
            )
        ),
    )

    grouped = units.group_by(keys).agg(
        pl.len().alias("n_policy_origins"),
        pl.col("ValueCode").n_unique().alias("n_products"),
        pl.col("_unit_policy_paths").sum().alias("n_policy_paths_diagnostic"),
        pl.col("Date").n_unique().alias("n_training_dates"),
        pl.col("Date").min().alias("training_start_date"),
        pl.col("Date").max().alias("training_end_date"),
        pl.col("_unit_label_cutoff").max().alias("label_cutoff_date"),
        pl.col("_unit_path_labels").sum().alias("n_path_labels_diagnostic"),
        pl.col("_unit_maturity_share").sum().alias("n_labels_matured"),
        pl.col("_unit_pending_share").sum().alias("n_labels_pending"),
        pl.col("_unit_fill_share").sum().alias("n_outcome_fill"),
        pl.col("_unit_no_fill_share").sum().alias("n_outcome_no_fill"),
        pl.col("_unit_unknown_share").sum().alias("n_outcome_unknown"),
        pl.col("_unit_censored_share").sum().alias("n_outcome_censored"),
        pl.col("_unit_maturity_share").mean().alias("label_maturity_coverage"),
        pl.col("_unit_pending_share").mean().alias("pending_rate"),
        pl.col("_unit_fill_share").mean().alias("p_outcome_fill"),
        pl.col("_unit_no_fill_share").mean().alias("p_outcome_no_fill"),
        pl.col("_unit_unknown_share").mean().alias("unknown_rate"),
        pl.col("_unit_censored_share").mean().alias("censor_rate"),
        pl.col("_unit_has_terminal_fill").sum().alias(
            "n_terminal_fill_origins"
        ),
        pl.col("_unit_has_priced_terminal").sum().alias(
            "n_priced_terminal_origins"
        ),
        pl.col("_unit_point_identified").sum().alias(
            "n_point_identified_origins"
        ),
        pl.col("_unit_terminal_policy_executable").sum().alias(
            "n_executable_terminal_policy_origins"
        ),
        pl.col("_unit_cost_components_complete").sum().alias(
            "n_cost_complete_origins"
        ),
        pl.col("_unit_conditional_value").drop_nulls().mean().alias(
            "conditional_known_ev_mean_bp"
        ),
        pl.col("_unit_known_accrued_cost_mean").mean().alias(
            "known_accrued_cost_mean_bp"
        ),
        pl.col("_unit_no_fill_continuation_mean").drop_nulls().mean().alias(
            "conditional_no_fill_continuation_mean_bp"
        ),
        *(
            pl.col(source).drop_nulls().mean().alias(target)
            for source, target in (
                ("_unit_identified_value", "_identified_mean_candidate"),
                ("_unit_identified_gross", "_identified_gross_candidate"),
                ("_unit_identified_immediate", "_identified_immediate_candidate"),
                (
                    "_unit_identified_continuation",
                    "_identified_continuation_candidate",
                ),
                ("_unit_identified_cost", "_identified_cost_candidate"),
            )
        ),
    ).with_columns(
        pl.col("n_policy_origins").alias("n_physical_origins"),
        pl.col("n_terminal_fill_origins").alias("n_terminal_fills"),
        pl.col("n_priced_terminal_origins").alias("n_terminal_priced"),
        (
            pl.col("n_point_identified_origins") == pl.col("n_policy_origins")
        ).alias("terminal_cashflow_point_identified"),
        (
            pl.col("p_outcome_fill")
            + pl.col("p_outcome_no_fill")
            + pl.col("unknown_rate")
            + pl.col("censor_rate")
        ).alias("matured_outcome_probability_sum"),
        (
            pl.col("p_outcome_fill")
            + pl.col("p_outcome_no_fill")
            + pl.col("unknown_rate")
            + pl.col("censor_rate")
            + pl.col("pending_rate")
        ).alias("origin_probability_sum"),
    ).with_columns(
        *(
            pl.when(pl.col("terminal_cashflow_point_identified"))
            .then(pl.col(source))
            .otherwise(None)
            .alias(target)
            for source, target in (
                ("_identified_mean_candidate", "identified_ev_mean_bp"),
                (
                    "_identified_gross_candidate",
                    "identified_four_leg_gross_mean_bp",
                ),
                (
                    "_identified_immediate_candidate",
                    "identified_immediate_price_pnl_mean_bp",
                ),
                (
                    "_identified_continuation_candidate",
                    "identified_continuation_gross_mean_bp",
                ),
                (
                    "_identified_cost_candidate",
                    "identified_non_price_cost_mean_bp",
                ),
            )
        ),
    )

    if embargoed.is_empty():
        grouped = grouped.with_columns(
            pl.lit(0).cast(pl.Int64).alias("n_embargoed_recent_origins")
        )
    else:
        recent = embargoed.group_by(keys).agg(
            pl.struct(
                "Date",
                "ValueCode",
                "QuoteCode",
                "physical_entry_id",
                "alternative_set_id",
            )
            .n_unique()
            .alias("n_embargoed_recent_origins")
        )
        grouped = grouped.join(recent, on=keys, how="left", validate="1:1").with_columns(
            pl.col("n_embargoed_recent_origins").fill_null(0)
        )

    means = grouped.select(*keys, "identified_ev_mean_bp")
    clusters = units.filter(pl.col("_unit_identified_value").is_not_null()).join(
        means, on=keys, how="left", validate="m:1"
    ).group_by([*keys, "Date"]).agg(
        (
            pl.col("_unit_identified_value") - pl.col("identified_ev_mean_bp")
        ).sum().alias("_cluster_score"),
        pl.len().alias("_cluster_units"),
    ).group_by(keys).agg(
        pl.len().alias("n_date_clusters_for_ci"),
        pl.col("_cluster_score").pow(2).sum().alias("_cluster_score_sq_sum"),
        pl.col("_cluster_units").sum().alias("_cluster_unit_count"),
    )
    grouped = grouped.join(clusters, on=keys, how="left", validate="1:1")
    grouped = grouped.with_columns(
        pl.when(
            pl.col("terminal_cashflow_point_identified")
            & (pl.col("n_date_clusters_for_ci") >= 2)
            & (pl.col("_cluster_unit_count") > 0)
        ).then(
            (
                pl.col("n_date_clusters_for_ci")
                / (pl.col("n_date_clusters_for_ci") - 1)
                * pl.col("_cluster_score_sq_sum")
                / pl.col("_cluster_unit_count").pow(2)
            ).sqrt()
        ).otherwise(None).alias("date_clustered_standard_error_bp"),
        pl.col("n_date_clusters_for_ci")
        .map_elements(
            lambda count: (
                _student_t_quantile(config.confidence_level, int(count) - 1)
                if count is not None and int(count) >= 2
                else None
            ),
            return_dtype=pl.Float64,
        )
        .alias("one_sided_critical_value"),
        pl.lit(config.confidence_level).alias("one_sided_confidence_level"),
        pl.lit("student_t_df_G_minus_1").alias("lcb_reference_distribution"),
    ).with_columns(
        (
            pl.col("identified_ev_mean_bp")
            - pl.col("one_sided_critical_value")
            * pl.col("date_clustered_standard_error_bp")
        ).alias("one_sided_lcb_bp")
    )

    is_pool = level_index >= 2
    grouped = grouped.with_columns(
        pl.lit(window_sessions >= config.min_history_sessions).alias(
            "history_session_support_gate"
        ),
        (pl.col("n_training_dates") >= config.min_training_dates).alias(
            "training_date_support_gate"
        ),
        (pl.col("n_policy_origins") >= config.min_policy_origins).alias(
            "policy_origin_support_gate"
        ),
        (pl.col("n_terminal_fill_origins") >= config.min_terminal_fills).alias(
            "terminal_fill_support_gate"
        ),
        (pl.col("n_priced_terminal_origins") >= config.min_priced_terminals).alias(
            "priced_terminal_support_gate"
        ),
        (pl.col("label_maturity_coverage") >= config.min_label_maturity_coverage).alias(
            "label_maturity_support_gate"
        ),
        (pl.col("unknown_rate") <= config.max_unknown_rate).alias(
            "unknown_rate_gate"
        ),
        (pl.col("censor_rate") <= config.max_censor_rate).alias(
            "censor_rate_gate"
        ),
        (pl.col("n_date_clusters_for_ci") >= 2).fill_null(False).alias(
            "cluster_ci_support_gate"
        ),
        (
            pl.col("n_executable_terminal_policy_origins")
            == pl.col("n_policy_origins")
        ).alias("executable_terminal_policy_gate"),
        (
            pl.lit(not is_pool)
            |
            (pl.col("n_policy_origins") >= config.min_peer_global_origins)
            & (pl.col("n_products") >= config.min_peer_global_products)
        ).alias("peer_global_pool_support_gate"),
    )
    statistical = pl.all_horizontal(
        pl.col("history_session_support_gate"),
        pl.col("training_date_support_gate"),
        pl.col("policy_origin_support_gate"),
        pl.col("terminal_fill_support_gate"),
        pl.col("priced_terminal_support_gate"),
        pl.col("label_maturity_support_gate"),
        pl.col("unknown_rate_gate"),
        pl.col("censor_rate_gate"),
        pl.col("cluster_ci_support_gate"),
        pl.col("executable_terminal_policy_gate"),
        pl.col("peer_global_pool_support_gate"),
    )
    grouped = grouped.with_columns(
        statistical.alias("statistical_support_ready"),
        (
            statistical
            & pl.col("terminal_cashflow_point_identified")
            & pl.col("one_sided_lcb_bp").is_finite()
        ).alias("direct_ev_ready"),
    ).with_columns(
        pl.when(~pl.col("history_session_support_gate"))
        .then(pl.lit("insufficient_history_sessions"))
        .when(~pl.col("training_date_support_gate"))
        .then(pl.lit("insufficient_training_dates"))
        .when(~pl.col("policy_origin_support_gate"))
        .then(pl.lit("insufficient_physical_policy_origins"))
        .when(~pl.col("terminal_fill_support_gate"))
        .then(pl.lit("insufficient_physical_terminal_fill_origins"))
        .when(~pl.col("priced_terminal_support_gate"))
        .then(pl.lit("insufficient_physical_priced_terminal_origins"))
        .when(~pl.col("peer_global_pool_support_gate"))
        .then(pl.lit("peer_global_requires_500_origins_and_5_products"))
        .when(~pl.col("executable_terminal_policy_gate"))
        .then(pl.lit("finite_horizon_terminal_policy_not_executable"))
        .when(~pl.col("label_maturity_support_gate"))
        .then(pl.lit("insufficient_label_maturity_coverage"))
        .when(~pl.col("unknown_rate_gate"))
        .then(pl.lit("excess_unknown_terminal_mass"))
        .when(~pl.col("censor_rate_gate"))
        .then(pl.lit("excess_censored_terminal_mass"))
        .when(~pl.col("terminal_cashflow_point_identified"))
        .then(pl.lit("unpriced_pending_unknown_or_censored_terminal_mass"))
        .when(~pl.col("cluster_ci_support_gate"))
        .then(pl.lit("insufficient_date_clusters_for_student_t_lcb"))
        .otherwise(pl.lit("ready"))
        .alias("direct_ev_status"),
        pl.when(pl.col("terminal_cashflow_point_identified"))
        .then(pl.lit("point_identified"))
        .otherwise(pl.lit("unbounded_without_terminal_value_limits"))
        .alias("cashflow_identification_status"),
    )

    result = grouped
    for dimension in LOOKUP_DIMENSIONS:
        if dimension not in keys:
            result = result.with_columns(pl.lit(POOLED).alias(dimension))
    return result.with_columns(
        pl.lit(asof_date).alias("asof_date"),
        pl.lit(level_index).alias("fallback_level"),
        pl.lit(level_name).alias("fallback_name"),
        pl.lit(json.dumps(list(keys), separators=(",", ":"))).alias(
            "fallback_retained_keys_json"
        ),
    ).drop(
        "_identified_mean_candidate",
        "_identified_gross_candidate",
        "_identified_immediate_candidate",
        "_identified_continuation_candidate",
        "_identified_cost_candidate",
        "_cluster_score_sq_sum",
        "_cluster_unit_count",
    ).select(
        "asof_date",
        "fallback_level",
        "fallback_name",
        *LOOKUP_DIMENSIONS,
        pl.exclude(
            "asof_date",
            "fallback_level",
            "fallback_name",
            *LOOKUP_DIMENSIONS,
        ),
    )


def _attach_hierarchical_shrinkage(
    frame: pl.DataFrame, config: ContextualPolicyEVConfig
) -> pl.DataFrame:
    """Attach a transparent κ prior from product -> peer -> global."""

    rows = [dict(row) for row in frame.iter_rows(named=True)]
    by_key: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        by_key[
            (
                int(row["fallback_level"]),
                *(row[name] for name in LOOKUP_DIMENSIONS),
            )
        ] = row
    for level in reversed(range(len(FALLBACK_LEVELS))):
        _, child_keys = FALLBACK_LEVELS[level]
        for row in (item for item in rows if int(item["fallback_level"]) == level):
            child_mean = _optional_finite(row.get("identified_ev_mean_bp"))
            child_lcb = _optional_finite(row.get("one_sided_lcb_bp"))
            child_point_identified = (
                row.get("terminal_cashflow_point_identified") is True
                and row.get("executable_terminal_policy_gate") is True
            )
            child_usable = (
                child_point_identified
                and int(row.get("n_policy_origins") or 0)
                >= config.min_shrinkage_child_origins
                and child_mean is not None
                and child_lcb is not None
            )
            parent: dict[str, object] | None = None
            if level + 1 < len(FALLBACK_LEVELS):
                _, parent_keys = FALLBACK_LEVELS[level + 1]
                parent_dims = tuple(
                    row[name] if name in parent_keys else POOLED
                    for name in LOOKUP_DIMENSIONS
                )
                parent = by_key.get((level + 1, *parent_dims))
            parent_ready = parent is not None and parent.get("ev_ready") is True
            direct_ready = row.get("direct_ev_ready") is True
            if parent_ready and child_usable:
                n = float(row["n_policy_origins"])
                weight = n / (n + config.shrinkage_kappa)
                parent_mean = float(parent["posterior_ev_mean_bp"])
                parent_lcb = float(parent["posterior_one_sided_lcb_bp"])
                posterior_mean = weight * float(child_mean) + (1.0 - weight) * parent_mean
                posterior_lcb = weight * float(child_lcb) + (1.0 - weight) * parent_lcb
                ready = True
                status = "ready_kappa_hierarchically_shrunk"
                source = "child_plus_parent"
            elif direct_ready:
                weight = 1.0
                posterior_mean = child_mean
                posterior_lcb = child_lcb
                ready = True
                status = "ready_direct_no_eligible_parent"
                source = "direct"
            elif parent_ready and child_point_identified:
                weight = 0.0
                posterior_mean = float(parent["posterior_ev_mean_bp"])
                posterior_lcb = float(parent["posterior_one_sided_lcb_bp"])
                ready = True
                status = "ready_parent_only_fallback"
                source = "parent_only"
            else:
                weight = None
                posterior_mean = None
                posterior_lcb = None
                ready = False
                status = str(row.get("direct_ev_status") or "no_supported_parent")
                source = "none"
            row.update(
                shrinkage_kappa=config.shrinkage_kappa,
                shrinkage_weight=weight,
                shrinkage_source=source,
                parent_fallback_name=(None if parent is None else parent["fallback_name"]),
                parent_ev_ready=parent_ready,
                posterior_ev_mean_bp=posterior_mean,
                posterior_one_sided_lcb_bp=posterior_lcb,
                hierarchical_ev_ready=ready,
                hierarchical_ev_status=status,
                ev_ready=ready,
                ev_status=status,
            )
    return pl.from_dicts(rows, infer_schema_length=None)


def _validate_action_rows(actions: pl.DataFrame) -> None:
    if actions.select(["policy_path_id", "action_alias"]).n_unique() != actions.height:
        raise ValueError("policy action facts duplicate policy_path_id/action_alias")
    for row in actions.iter_rows(named=True):
        date = _date(row["Date"], "Date")
        source = _date(row["source_asof_date"], "source_asof_date")
        cost_source = _date(
            row["cost_profile_source_asof_date"],
            "cost_profile_source_asof_date",
        )
        if source >= date:
            raise ValueError("source_asof_date must strictly precede Date")
        if cost_source >= date:
            raise ValueError(
                "cost_profile_source_asof_date must strictly precede Date"
            )
        if row["contains_target_day_outcome"] is not False:
            raise ValueError("decision action contains target-day outcome")
        if row["cost_profile_contains_target_day_outcome"] is not False:
            raise ValueError("decision cost profile contains target-day outcome")
        for name in (
            "policy_path_id",
            "action_alias",
            "decision_id",
            "candidate_set_id",
            "alternative_set_id",
            "physical_entry_id",
            "ValueCode",
            "QuoteCode",
            "decision_phase",
            "common_decision_clock_id",
            "entry_route",
            "entry_q",
            "entry_basis_bucket",
            "locked_entry_state_bucket",
            "route",
            "action_bucket",
            "policy_family",
            "lifecycle_policy_version",
            "queue_scenario",
            "state_bucket",
            "peer_group",
            "terminal_policy_id",
            "cost_profile_id",
            "cost_profile_version",
        ):
            if row[name] is None or not str(row[name]).strip():
                raise ValueError(f"{name} must be non-empty")
        for name in (
            "ValueCode",
            "peer_group",
            "decision_phase",
            "entry_route",
            "entry_q",
            "entry_basis_bucket",
            "locked_entry_state_bucket",
            "route",
            "action_bucket",
            "policy_family",
            "lifecycle_policy_version",
            "queue_scenario",
            "state_bucket",
            "terminal_policy_id",
            "cost_profile_hash",
        ):
            if str(row[name]) == POOLED:
                raise ValueError(f"{name} uses reserved pooled sentinel")
        _validate_sha256(row["candidate_set_hash"], "candidate_set_hash")
        _validate_sha256(row["cost_profile_hash"], "cost_profile_hash")
        for name in (
            "candidate_set_complete",
            "terminal_policy_executable",
            "legal_action",
        ):
            if not isinstance(row[name], bool):
                raise ValueError(f"{name} must be boolean")
        _positive_int(
            row["candidate_set_expected_count"],
            "candidate_set_expected_count",
        )
        _non_negative_int(row["decision_time_ns"], "decision_time_ns")
        _non_negative_int(
            row["decision_event_sequence"], "decision_event_sequence"
        )
        _non_negative_int(row["decision_row_index"], "decision_row_index")
        _non_negative_int(row["legal_tick"], "legal_tick")
        _strict_int(row["relative_tick_offset"], "relative_tick_offset")
        _positive_int(
            row["intended_maker_quantity"], "intended_maker_quantity"
        )
        _non_negative_int(row["hedge_delay_ns"], "hedge_delay_ns")
        _non_negative_float(row["remaining_minutes"], "remaining_minutes")
        _non_negative_float(row["holding_age_minutes"], "holding_age_minutes")
        _non_negative_int(row["sessions_to_expiry"], "sessions_to_expiry")
        _positive_int(row["finite_horizon_sessions"], "finite_horizon_sessions")


def _validate_outcome_rows(outcomes: pl.DataFrame) -> None:
    for row in outcomes.iter_rows(named=True):
        start = _date(row["Date"], "Date")
        horizon_end = _date(
            row["label_horizon_end_date"], "label_horizon_end_date"
        )
        end = (
            None
            if row["label_end_date"] is None
            else _date(row["label_end_date"], "label_end_date")
        )
        cost_source = _date(
            row["cost_profile_source_asof_date"],
            "cost_profile_source_asof_date",
        )
        if horizon_end < start:
            raise ValueError("label_horizon_end_date cannot precede origin Date")
        if end is not None and end < start:
            raise ValueError("label_end_date cannot precede origin Date")
        if end is not None and end > horizon_end:
            raise ValueError("label_end_date cannot exceed finite label horizon")
        if cost_source >= start:
            raise ValueError(
                "cost_profile_source_asof_date must strictly precede Date"
            )
        if row["cost_profile_contains_target_day_outcome"] is not False:
            raise ValueError("terminal cost profile contains target-day outcome")
        for name in (
            "path_label_id",
            "policy_path_id",
            "alternative_set_id",
            "physical_entry_id",
            "ValueCode",
            "QuoteCode",
            "decision_phase",
            "entry_route",
            "entry_q",
            "entry_basis_bucket",
            "locked_entry_state_bucket",
            "route",
            "action_bucket",
            "policy_family",
            "lifecycle_policy_version",
            "queue_scenario",
            "terminal_policy_id",
            "cost_profile_id",
            "cost_profile_version",
        ):
            if row[name] is None or not str(row[name]).strip():
                raise ValueError(f"{name} must be non-empty")
        _validate_sha256(row["cost_profile_hash"], "cost_profile_hash")
        _non_negative_int(row["legal_tick"], "legal_tick")
        _strict_int(row["relative_tick_offset"], "relative_tick_offset")
        _positive_int(
            row["intended_maker_quantity"], "intended_maker_quantity"
        )
        _non_negative_int(row["hedge_delay_ns"], "hedge_delay_ns")
        _positive_int(row["finite_horizon_sessions"], "finite_horizon_sessions")
        for name in (
            "terminal_fill_observed",
            "actual_four_leg_complete",
            "gross_includes_observed_slippage",
            "terminal_policy_executable",
            "terminal_cost_remainder_known",
            "cost_components_complete",
        ):
            if not isinstance(row[name], bool):
                raise ValueError(f"{name} must be boolean")
        origin = str(row["origin_execution_outcome"])
        terminal = str(row["terminal_outcome_status"])
        if origin not in ORIGIN_EXECUTION_OUTCOMES:
            raise ValueError("invalid origin_execution_outcome")
        if terminal not in TERMINAL_OUTCOME_STATUSES:
            raise ValueError("invalid terminal_outcome_status")
        accrued = _optional_finite(row["known_accrued_cost_bp"])
        if accrued is None or accrued < 0:
            raise ValueError("known_accrued_cost_bp must be finite and non-negative")
        components = [_optional_finite(row[name]) for name in COST_COMPONENT_COLUMNS]
        if any(
            raw is not None and (value is None or value < 0)
            for raw, value in zip(
                (row[name] for name in COST_COMPONENT_COLUMNS),
                components,
                strict=True,
            )
        ):
            raise ValueError("cost components must be finite and non-negative")
        if origin == "unknown" and terminal != "unknown":
            raise ValueError("unknown origin execution must remain terminal unknown")
        if origin == "censored" and terminal != "censored":
            raise ValueError("censored origin execution must remain terminal censored")
        if origin == "no_fill":
            immediate = _optional_finite(row["immediate_price_pnl_bp"])
            if immediate is None or abs(immediate) > 1e-12:
                raise ValueError("no-fill immediate price PnL must be exactly zero")
            if not row["continuation_state_id"] or not row["continuation_state_bucket"]:
                raise ValueError("no-fill must retain its continuation state identity")
        if terminal == "known":
            if end is None:
                raise ValueError("known terminal path requires label_end_date")
            if origin not in {"fill", "no_fill"}:
                raise ValueError("known terminal path requires fill or no_fill origin")
            if row["terminal_branch"] is None:
                raise ValueError("known terminal path requires terminal_branch")
            gross = _optional_finite(row["actual_four_leg_gross_bp"])
            immediate = _optional_finite(row["immediate_price_pnl_bp"])
            continuation = _optional_finite(row["continuation_gross_value_bp"])
            if gross is None or immediate is None or continuation is None:
                raise ValueError("known path lacks finite four-leg decomposition")
            if row["actual_four_leg_complete"] is not True:
                raise ValueError("known path is not an actual complete four-leg path")
            if row["gross_includes_observed_slippage"] is not True:
                raise ValueError("four-leg gross must use observed execution prices")
            if row["cost_components_complete"] is not True or any(
                value is None for value in components
            ):
                raise ValueError("known path requires complete component-level costs")
            remainder = _optional_finite(row["terminal_cost_remainder_bp"])
            if (
                row["terminal_cost_remainder_known"] is not True
                or remainder is None
                or remainder < 0
            ):
                raise ValueError("known path requires known terminal cost remainder")
            component_sum = sum(float(value) for value in components if value is not None)
            if abs(component_sum - (accrued + remainder)) > 1e-9:
                raise ValueError(
                    "component costs must equal accrued plus terminal remainder"
                )
            if abs(immediate + continuation - gross) > 1e-9:
                raise ValueError("immediate plus continuation must equal four-leg gross")
            if origin == "fill" and abs(continuation) > 1e-12:
                raise ValueError("origin fill cannot duplicate terminal value in continuation")
            if origin == "no_fill" and abs(continuation - gross) > 1e-9:
                raise ValueError("no-fill continuation must retain the fixed-policy gross")
        else:
            for name in (
                "actual_four_leg_gross_bp",
                "continuation_gross_value_bp",
            ):
                if row[name] is not None:
                    raise ValueError(
                        f"unresolved terminal path must not claim priced {name}"
                    )
            if row["actual_four_leg_complete"] is True:
                raise ValueError("unresolved path cannot claim complete four-leg cashflow")
            if row["gross_includes_observed_slippage"] is True:
                raise ValueError("unresolved path cannot claim complete observed slippage")
            if row["cost_components_complete"] is True:
                raise ValueError(
                    "unresolved path cannot claim complete terminal cost components"
                )
            if row["terminal_cost_remainder_known"] is True:
                raise ValueError(
                    "unresolved path cannot claim known terminal cost remainder"
                )
            if row["terminal_cost_remainder_bp"] is not None:
                raise ValueError(
                    "unresolved path terminal cost remainder must remain unknown"
                )
            observed_components = [value for value in components if value is not None]
            if observed_components and abs(sum(observed_components) - accrued) > 1e-9:
                raise ValueError(
                    "unresolved observed cost components must equal known accrued cost"
                )


def _validate_prepared_paths(paths: pl.DataFrame) -> None:
    if paths.select("path_label_id").n_unique() != paths.height:
        raise ValueError("contextual paths duplicate path_label_id")
    if paths.select("policy_path_id").n_unique() != paths.height:
        raise ValueError("contextual paths duplicate policy_path_id")
    if paths.filter(
        pl.col("terminal_cashflow_priced")
        & (
            pl.col("realized_path_value_bp").is_null()
            | ~pl.col("realized_path_value_bp").is_finite()
        )
    ).height:
        raise ValueError("priced contextual path lacks finite realized value")


def _validate_finite_horizon_contract(
    paths: pl.DataFrame,
    sessions: Sequence[str],
    config: ContextualPolicyEVConfig,
) -> None:
    """Verify that every origin has a calendar-defined H and terminal rule."""

    calendar_index = {date: index for index, date in enumerate(sessions)}
    for row in paths.iter_rows(named=True):
        origin = _date(row["Date"], "Date")
        horizon = _positive_int(
            row["finite_horizon_sessions"], "finite_horizon_sessions"
        )
        if horizon != config.finite_horizon_sessions:
            raise ValueError("path finite horizon disagrees with lookup configuration")
        end_index = calendar_index[origin] + horizon
        if end_index >= len(sessions):
            raise ValueError(
                "session calendar does not extend through each path label horizon"
            )
        expected_end = sessions[end_index]
        observed_end = _date(
            row["label_horizon_end_date"], "label_horizon_end_date"
        )
        if observed_end != expected_end:
            raise ValueError(
                "label_horizon_end_date does not equal origin plus finite H sessions"
            )
        label_end = row["label_end_date"]
        if label_end is not None and _date(label_end, "label_end_date") > expected_end:
            raise ValueError("label_end_date exceeds the executable terminal horizon")
        if not str(row["terminal_policy_id"]).strip():
            raise ValueError("finite horizon requires an explicit terminal_policy_id")


@lru_cache(maxsize=512)
def _student_t_quantile(probability: float, degrees_of_freedom: int) -> float:
    """Numerically invert Student's t CDF without a SciPy dependency."""

    probability = float(probability)
    if not 0.5 < probability < 1.0:
        raise ValueError("Student t upper quantile probability must be in (0.5, 1)")
    if degrees_of_freedom <= 0:
        raise ValueError("Student t degrees_of_freedom must be positive")
    lower = 0.0
    upper = max(1.0, NormalDist().inv_cdf(probability))
    while _student_t_cdf(upper, degrees_of_freedom) < probability:
        upper *= 2.0
        if upper > 1e12:
            raise ArithmeticError("failed to bracket Student t quantile")
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        if _student_t_cdf(midpoint, degrees_of_freedom) < probability:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def _student_t_cdf(value: float, degrees_of_freedom: int) -> float:
    if value == 0.0:
        return 0.5
    v = float(degrees_of_freedom)
    x = v / (v + value * value)
    tail_twice = _regularized_incomplete_beta(x, v / 2.0, 0.5)
    return 1.0 - 0.5 * tail_twice if value > 0.0 else 0.5 * tail_twice


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    max_iterations = 300
    epsilon = 3e-14
    tiny = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d
    for iteration in range(1, max_iterations + 1):
        twice = 2 * iteration
        numerator = (
            iteration * (b - iteration) * x
            / ((qam + twice) * (a + twice))
        )
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c
        numerator = -(
            (a + iteration)
            * (qab + iteration)
            * x
            / ((a + twice) * (qap + twice))
        )
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= epsilon:
            return result
    raise ArithmeticError("incomplete beta continued fraction did not converge")


def _validate_ranking_constraints(actions: pl.DataFrame) -> None:
    invalid_effect = actions.filter(
        ~pl.col("action_effect").is_in(sorted(ACTION_EFFECTS)).fill_null(False)
    )
    if invalid_effect.height:
        raise ValueError("decision actions contain invalid action_effect")
    for row in actions.iter_rows(named=True):
        for name in (
            "inventory_position_units",
            "reserved_inventory_units",
            "action_inventory_delta_units",
            "inventory_min_units",
            "inventory_max_units",
            "active_oco_sibling_count",
        ):
            value = row[name]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if row["active_oco_sibling_count"] < 0:
            raise ValueError("active_oco_sibling_count must be non-negative")
        if row["inventory_min_units"] > row["inventory_max_units"]:
            raise ValueError("inventory bounds are inverted")
        if not isinstance(row["replaces_active_oco"], bool):
            raise ValueError("replaces_active_oco must be boolean")
        if not isinstance(row["is_risk_baseline"], bool):
            raise ValueError("is_risk_baseline must be boolean")


def _validate_decision_sets(
    actions: pl.DataFrame, ranking_config: ActionRankingConfig
) -> None:
    fixed = [
        "Date",
        "candidate_set_id",
        "candidate_set_expected_count",
        "candidate_set_hash",
        "candidate_set_complete",
        "alternative_set_id",
        "physical_entry_id",
        "ValueCode",
        "QuoteCode",
        "peer_group",
        "decision_phase",
        "common_decision_clock_id",
        "decision_time_ns",
        "decision_event_sequence",
        "decision_row_index",
        "entry_route",
        "entry_q",
        "entry_basis_bucket",
        "locked_entry_state_bucket",
        "remaining_minutes",
        "holding_age_minutes",
        "sessions_to_expiry",
        "state_bucket",
        "finite_horizon_sessions",
        "terminal_policy_id",
        "terminal_policy_executable",
        "inventory_position_units",
        "reserved_inventory_units",
        "inventory_min_units",
        "inventory_max_units",
        "active_oco_sibling_count",
    ]
    inconsistent = actions.group_by("decision_id").agg(
        *(pl.col(name).n_unique().alias(name) for name in fixed)
    ).filter(pl.any_horizontal(*(pl.col(name) != 1 for name in fixed)))
    if inconsistent.height:
        raise ValueError("mutually-exclusive action rows disagree on decision state")
    baseline_counts = actions.group_by("decision_id").agg(
        pl.col("inventory_position_units").first().alias("_inventory"),
        pl.col("is_risk_baseline").sum().alias("_baseline_count"),
    )
    if baseline_counts.filter(
        (pl.col("_inventory") != 0) & (pl.col("_baseline_count") != 1)
    ).height:
        raise ValueError(
            "open inventory requires exactly one explicit priced WAIT baseline"
        )
    if actions.filter(
        pl.col("is_risk_baseline") & (pl.col("action_effect") != "wait")
    ).height:
        raise ValueError("risk baseline action_effect must be wait")
    if ranking_config.require_position_establishment_phase and actions.filter(
        (pl.col("decision_phase") != "position_establishment")
        | (pl.col("holding_age_minutes").abs() > 1e-12)
    ).height:
        raise ValueError(
            "v1 identifies only fixed-policy choice at position_establishment "
            "with zero holding age"
        )


def _continuous_bucket(
    value: object, edges: Sequence[float | int], prefix: str
) -> str:
    number = _non_negative_float(value, prefix)
    lower = 0.0
    for edge in edges:
        upper = float(edge)
        if number < upper:
            return f"{prefix}:[{_format_bound(lower)},{_format_bound(upper)})"
        lower = upper
    return f"{prefix}:[{_format_bound(lower)},inf)"


def _remaining_minutes_bucket(
    value: object, edges: Sequence[float | int]
) -> str:
    """Right-closed countdown buckets: <=5, (5,15], ..., >120."""

    number = _non_negative_float(value, "remaining_minutes")
    lower = 0.0
    for index, edge in enumerate(edges):
        upper = float(edge)
        if number <= upper:
            left = "[" if index == 0 else "("
            return (
                f"remaining:{left}{_format_bound(lower)},"
                f"{_format_bound(upper)}]"
            )
        lower = upper
    return f"remaining:({_format_bound(lower)},inf)"


def _legal_tick_bucket(value: object, width: int) -> str:
    tick = _non_negative_int(value, "legal_tick")
    lower = (tick // width) * width
    upper = lower + width - 1
    return f"tick:{lower}" if width == 1 else f"tick:[{lower},{upper}]"


def _validate_edges(values: Sequence[float | int], name: str) -> None:
    converted = [float(value) for value in values]
    if not converted or any(not math.isfinite(value) or value <= 0 for value in converted):
        raise ValueError(f"{name} must contain positive finite bounds")
    if converted != sorted(set(converted)):
        raise ValueError(f"{name} must be strictly increasing and unique")


def _normalise_sessions(values: Iterable[str]) -> list[str]:
    sessions = [str(value) for value in values]
    if not sessions or len(sessions) != len(set(sessions)):
        raise ValueError("sessions must be non-empty and unique")
    if sessions != sorted(sessions):
        raise ValueError("sessions must be sorted ascending")
    for value in sessions:
        _date(value, "session")
    return sessions


def _date(value: object, name: str) -> str:
    text = str(value)
    try:
        parsed = datetime.strptime(text, "%Y%m%d")
    except ValueError as error:
        raise ValueError(f"{name} must be valid YYYYMMDD") from error
    return parsed.strftime("%Y%m%d")


def _non_negative_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and non-negative")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and non-negative") from error
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _non_negative_int(value: object, name: str) -> int:
    integer = _strict_int(value, name)
    if integer < 0:
        raise ValueError(f"{name} must be non-negative")
    return integer


def _positive_int(value: object, name: str) -> int:
    integer = _strict_int(value, name)
    if integer <= 0:
        raise ValueError(f"{name} must be positive")
    return integer


def _strict_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if not math.isfinite(number) or number != int(number):
        raise ValueError(f"{name} must be an integer")
    return int(number)


def _optional_finite(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _format_bound(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:g}"


def _series_all_equal(series: pl.Series) -> bool:
    values = series.to_list()
    if not values:
        return True
    first = values[0]
    return all(_same_value(first, value) for value in values[1:])


def _same_value(left: object, right: object) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, float) or isinstance(right, float):
        try:
            return math.isclose(
                float(left), float(right), rel_tol=0.0, abs_tol=1e-12
            )
        except (TypeError, ValueError):
            pass
    return left == right


def _ratio(numerator: str, denominator: str) -> pl.Expr:
    return (
        pl.when(pl.col(denominator) > 0)
        .then(pl.col(numerator) / pl.col(denominator))
        .otherwise(None)
    )


def stable_sha256(payload: Mapping[str, object]) -> str:
    """Hash a JSON-compatible manifest/config payload deterministically."""

    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compute_candidate_set_hash(
    rows: Sequence[Mapping[str, object]],
) -> str:
    """Hash the complete canonical candidate set for one decision.

    Callers should stamp this value on every row in the decision.  Aliases of
    one ``policy_path_id`` collapse to one payload; disagreements between such
    aliases are rejected instead of being hidden by the hash.
    """

    if not rows:
        raise ValueError("candidate action set cannot be empty")
    available = set.intersection(*(set(row) for row in rows))
    fields = [name for name in CANDIDATE_ACTION_HASH_FIELDS if name in available]
    required_hash_fields = sorted(
        ACTION_REQUIRED_COLUMNS
        - {
            "action_alias",
            "candidate_set_expected_count",
            "candidate_set_hash",
            "candidate_set_complete",
        }
    )
    if missing := sorted(set(required_hash_fields) - available):
        raise ValueError(f"candidate hash rows missing fields: {missing}")
    by_policy: dict[str, dict[str, object]] = {}
    for raw in rows:
        policy_path_id = str(raw["policy_path_id"])
        payload = {name: _json_scalar(raw.get(name)) for name in fields}
        prior = by_policy.get(policy_path_id)
        if prior is not None and prior != payload:
            raise ValueError(
                f"aliases for policy_path_id {policy_path_id} disagree in "
                "candidate action hash payload"
            )
        by_policy[policy_path_id] = payload
    ordered = [by_policy[key] for key in sorted(by_policy)]
    return stable_sha256(
        {
            "candidate_hash_schema": "complete_canonical_action_set_v1",
            "actions": ordered,
        }
    )


def _validate_candidate_sets(actions: pl.DataFrame) -> None:
    decision_fixed = (
        "candidate_set_id",
        "candidate_set_expected_count",
        "candidate_set_hash",
        "candidate_set_complete",
        "Date",
        "decision_phase",
        "common_decision_clock_id",
        "decision_time_ns",
        "decision_event_sequence",
        "decision_row_index",
        "alternative_set_id",
        "physical_entry_id",
        "ValueCode",
        "QuoteCode",
        "peer_group",
        "entry_route",
        "entry_q",
        "entry_basis_bucket",
        "locked_entry_state_bucket",
        "remaining_minutes",
        "holding_age_minutes",
        "sessions_to_expiry",
        "state_bucket",
        "finite_horizon_sessions",
        "terminal_policy_id",
        "terminal_policy_executable",
        "source_asof_date",
        "contains_target_day_outcome",
    )
    reused = actions.group_by("candidate_set_id").agg(
        pl.col("decision_id").n_unique().alias("_decisions")
    ).filter(pl.col("_decisions") != 1)
    if reused.height:
        raise ValueError("candidate_set_id must identify exactly one decision")
    for key, group in actions.group_by("decision_id", maintain_order=True):
        decision_id = str(key[0] if isinstance(key, tuple) else key)
        for name in decision_fixed:
            if not _series_all_equal(group[name]):
                raise ValueError(
                    f"candidate rows for decision {decision_id} disagree on {name}"
                )
        if group.item(0, "candidate_set_complete") is not True:
            raise ValueError("candidate set must be explicitly complete")
        expected = _non_negative_int(
            group.item(0, "candidate_set_expected_count"),
            "candidate_set_expected_count",
        )
        if expected <= 0 or expected != group.height:
            raise ValueError(
                "canonical candidate count does not match "
                "candidate_set_expected_count"
            )
        claimed = str(group.item(0, "candidate_set_hash"))
        _validate_sha256(claimed, "candidate_set_hash")
        actual = compute_candidate_set_hash(list(group.iter_rows(named=True)))
        if claimed != actual:
            raise ValueError("candidate_set_hash does not match complete action set")


def _json_scalar(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_scalar(item) for item in value]
    return str(value)


def _validate_sha256(value: object, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return text


def _require(frame: pl.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")
