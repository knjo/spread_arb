"""Strict V2 bridge for maker/taker exit-policy observations.

The entry execution replay and the exit-maker replay deliberately have
different sampling units:

* an entry ``raw_order_fact_id`` is one physical candidate which may be
  shared by several q aliases;
* an exit policy is an *alternative* ``exit_rule_id x exit_route`` attached
  to one entry alias; and
* a taker/taker close is one shared counterfactual baseline for both exit
  maker routes, not two independent observations.

This module keeps those units separate.  It maps one normalized exit-policy
fact to one mutually-exclusive terminal-path row, builds unique taker/taker
baseline references and route-specific matched pairs, and provides an
analysis-only flat-cost sensitivity.  It does not change the V1 adapter in
``terminal_paths.py`` and it never promotes nominal cancellation to an
observed terminal outcome.

All actual entry and exit prices are already reflected in
``gross_cycle_pnl_twd``.  Hedge/exit slippage columns are diagnostics and are
therefore never subtracted again by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable, Mapping

import polars as pl

from .ev_surface import DEFAULT_COST_COLUMNS
from .hedge_study import (
    FUTURE_ASK_ROUTE,
    FUTURE_HEDGE_CONTRACTS,
    SPOT_BID_ROUTE,
    SPOT_HEDGE_LOTS,
)


FUTURE_BID_EXIT_ROUTE = "future_bid_spot_taker"
SPOT_ASK_EXIT_ROUTE = "spot_ask_future_taker"
SUPPORTED_EXIT_ROUTES = (FUTURE_BID_EXIT_ROUTE, SPOT_ASK_EXIT_ROUTE)
MAKER_TAKER_EXIT_STYLE = "maker_taker"

EXIT_MAKER_PATH_LOOKUP_KEYS_V2: tuple[str, ...] = (
    "ValueCode",
    "entry_route",
    "entry_lookup_action_id",
    "entry_parameter_version",
    "exit_rule_id",
    "exit_style",
    "exit_route",
    "state_family",
    "state_bucket",
    "lifecycle_policy_version",
    "exit_lifecycle_policy_version",
    "queue_scenario",
    "exit_queue_scenario",
    "intended_maker_quantity",
    "hedge_quantity",
    "exit_maker_quantity",
    "exit_hedge_quantity",
)

# Primitive exit-maker outcomes which can be normalized without inventing an
# exchange acknowledgement, a fractional futures hedge, or an overnight PnL.
KNOWN_TARGET_EXIT_BRANCH = "flat_same_day"
CENSORED_CARRY_BRANCH = "carry_at_eod_cancel_unconfirmed"
CENSORED_CARRY_BRANCHES = frozenset(
    {CENSORED_CARRY_BRANCH, "carry_at_eod_no_admission"}
)
UNKNOWN_EXIT_BRANCHES = frozenset(
    {
        "no_fill_before_cancel_request",
        "partial_fill_then_cancel",
        "partial_fill_carry_at_eod",
        "hedge_incomplete_residual",
        "hedge_incomplete_carry_at_eod",
        "fill_unknown_then_cancel",
        "fill_unknown_at_eod",
        "cancel_race_unknown",
        "exit_policy_unassigned",
        "not_opened_or_unhedged",
    }
)
SUPPORTED_POLICY_BRANCHES = frozenset(
    {
        KNOWN_TARGET_EXIT_BRANCH,
        *CENSORED_CARRY_BRANCHES,
        *UNKNOWN_EXIT_BRANCHES,
    }
)

NON_PRICE_COST_SCOPE = "non_price_roundtrip_fees_tax_commission_only"

_ACTION_REQUIRED = {
    "Date",
    "ValueCode",
    "QuoteCode",
    "route",
    "parameter_version",
    "lookup_action_id",
    "raw_order_fact_id",
    "policy_generation_id",
    "rank_bucket",
    "queue_bucket",
    "tod_bucket",
    "freshness_bucket",
    "intended_quantity",
    "submit_recv_time_ns",
    "terminal_recv_time_ns",
    "queue_known",
    "any_fill",
    "full_fill",
    "partial_fill",
    "cancel_required",
    "entry_hedge_status",
    "entry_hedge_signed_total_slippage_bp",
    "entry_hedge_contract_size_shares",
    "entry_spot_price",
}

# This is the normalized boundary between a raw exit-maker study and terminal
# accounting.  There must be one row for every entry alias/rule/route policy
# trial, including zero-admission and unresolved trials.
EXIT_POLICY_FACT_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
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
        "exit_lifecycle_policy_version",
        "exit_queue_scenario",
        "exit_style",
        "exit_maker_quantity",
        "exit_hedge_quantity",
        "oco_winner_raw_candidate_fact_id",
        "oco_winner_spread_pair_epoch",
        "oco_winner_target_price_tick",
        "oco_winner_submit_recv_time_ns",
        "oco_winner_submit_event_sequence",
        "oco_winner_submit_row_index",
        "oco_winner_raw_identity_excludes_rule",
        "oco_position_projection_safe",
        "oco_active_sibling_cancel_count",
        "prior_unacked_cancel_count_before_winner",
        "branch_status",
        "terminal_outcome",
        "needs_next_session_label",
        "exit_decision_time_ns",
        "gross_cycle_pnl_twd",
        "exit_hedge_status",
        "exit_hedge_signed_total_slippage_bp",
        "cancel_ack_observed",
        "cancel_race_modeled",
        "joint_volume_allocated",
    }
)

_TAKER_BASELINE_REQUIRED = {
    "Date",
    "ValueCode",
    "QuoteCode",
    "route",
    "raw_order_fact_id",
    "policy_generation_id",
    "exit_rule_id",
    "exit_threshold_basis_bp",
    "exit_rule_source_asof_date",
    "branch_status",
    "terminal_outcome",
    "needs_next_session_label",
    "exit_decision_time_ns",
    "gross_cycle_pnl_twd",
}

_TAKER_BASELINE_BRANCHES = frozenset(
    {
        "partial_entry_unhedged",
        "entry_fill_unknown",
        "no_entry_fill",
        "entry_hedge_unpriceable",
        "same_day_taker_exit",
        "carry_at_eod",
        "exit_policy_unassigned",
    }
)


@dataclass(frozen=True)
class ExitMakerTerminalConfig:
    """Frozen entry-side semantics for the V2 terminal adapter."""

    lifecycle_policy_version: str
    queue_scenario: str
    state_family: str = "execution_exit_maker_policy_v2"
    mapping_version: str = "exit_maker_terminal_adapter_v2"

    def validate(self) -> None:
        for name in (
            "lifecycle_policy_version",
            "queue_scenario",
            "state_family",
            "mapping_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class FlatCostSensitivityConfig:
    """Analysis-only flat non-price round-trip cost assumption."""

    assumed_cycle_cost_bp: float = 19.0
    sensitivity_id: str = "flat_non_price_roundtrip_19bp_v1"
    cost_scope: str = NON_PRICE_COST_SCOPE

    def validate(self) -> None:
        if (
            isinstance(self.assumed_cycle_cost_bp, bool)
            or not isinstance(self.assumed_cycle_cost_bp, (int, float))
            or not math.isfinite(float(self.assumed_cycle_cost_bp))
            or float(self.assumed_cycle_cost_bp) < 0
        ):
            raise ValueError("assumed_cycle_cost_bp must be finite and non-negative")
        if not isinstance(self.sensitivity_id, str) or not self.sensitivity_id.strip():
            raise ValueError("sensitivity_id must be a non-empty string")
        if self.cost_scope != NON_PRICE_COST_SCOPE:
            raise ValueError(
                "flat sensitivity cost_scope must explicitly exclude price slippage"
            )


@dataclass(frozen=True)
class MatchedExitStyleComparison:
    """Unique T/T references, route alternatives, and an outcome matrix."""

    baseline_refs: pl.DataFrame
    pairs: pl.DataFrame
    outcome_matrix: pl.DataFrame


def canonical_exit_raw_candidate_fact_id(
    *,
    entry_raw_order_fact_id: str,
    exit_route: str,
    spread_pair_epoch: int,
    target_price_tick: int,
    submit_recv_time_ns: int,
    submit_event_sequence: int,
    submit_row_index: int,
) -> str:
    """Return the rule-free physical identity used by the exit-maker study.

    The key intentionally excludes q alias, ``exit_rule_id``, threshold and
    any policy version.  Thus center/lower policies which submit the exact
    same physical order resolve to one raw fact instead of two fills.
    """

    entry_raw = _required_text(
        entry_raw_order_fact_id, "entry_raw_order_fact_id"
    )
    route = _required_text(exit_route, "exit_route")
    if route not in SUPPORTED_EXIT_ROUTES:
        raise ValueError(f"unsupported exit route: {route}")
    components = (
        entry_raw,
        route,
        _nonnegative_integer(spread_pair_epoch, "spread_pair_epoch"),
        _integer(target_price_tick, "target_price_tick"),
        _nonnegative_integer(submit_recv_time_ns, "submit_recv_time_ns"),
        _integer(submit_event_sequence, "submit_event_sequence"),
        _nonnegative_integer(submit_row_index, "submit_row_index"),
    )
    digest = hashlib.sha256(
        "|".join(map(str, components)).encode("utf-8")
    ).hexdigest()[:24]
    return f"exit-raw-{digest}"


def build_exit_maker_terminal_paths_v2(
    action_facts: pl.DataFrame,
    exit_policy_facts: pl.DataFrame,
    config: ExitMakerTerminalConfig,
) -> pl.DataFrame:
    """Build one strict path per entry alias x exit rule x exit route.

    ``exit_policy_facts`` is already a policy-level OCO projection.  Raw
    maker-order generations must be collapsed by the exit study before this
    function is called; accepting several rows for one joint policy would
    turn competing exit orders into fabricated independent terminal paths.

    A nominal cancel request is never a known terminal no-fill outcome.  The
    only known V2 maker-exit branch is a quantity-flat same-day maker/taker
    close.  An open full-sized position at EOD is censored, and every partial,
    residual, unknown-fill, hedge-incomplete, and cancel-race branch remains
    unknown.
    """

    config.validate()
    # Exit-maker support is conditional on an already-established entry.
    # Therefore a product-day with no established entries correctly has no
    # exit policy rows even when the entry action table itself is non-empty.
    if exit_policy_facts.is_empty():
        return pl.DataFrame(schema=_empty_terminal_schema())
    if action_facts.is_empty():
        raise ValueError("exit policy facts cannot exist without entry action facts")
    _require(action_facts, _ACTION_REQUIRED, "execution action facts")
    _require(
        exit_policy_facts,
        EXIT_POLICY_FACT_REQUIRED_COLUMNS,
        "exit maker policy facts",
    )
    if action_facts.select("policy_generation_id").n_unique() != action_facts.height:
        raise ValueError("action facts must be unique by policy_generation_id")

    policy_key = [
        "entry_policy_generation_id",
        "exit_rule_id",
        "exit_route",
    ]
    if exit_policy_facts.select(policy_key).n_unique() != exit_policy_facts.height:
        raise ValueError(
            "exit policy facts must be unique by entry alias, exit rule, and route"
        )
    if (
        exit_policy_facts.select("exit_policy_trial_id").n_unique()
        != exit_policy_facts.height
    ):
        raise ValueError("exit_policy_trial_id must be globally unique")

    _validate_exit_policy_facts(exit_policy_facts)
    action = _action_for_join(action_facts)
    joined = exit_policy_facts.join(
        action,
        on="entry_policy_generation_id",
        how="left",
        validate="m:1",
    )
    if joined.filter(pl.col("_action_Date").is_null()).height:
        raise ValueError("exit policy fact does not resolve to an entry action alias")
    _validate_action_identity(joined)
    _validate_established_entry_aliases(joined)

    records = [
        _map_policy_path(row, config)
        for row in joined.iter_rows(named=True)
    ]
    paths = pl.from_dicts(records, infer_schema_length=None).with_columns(
        pl.col("filled_cashflow_before_cost_bp").cast(pl.Float64),
        pl.col("hedge_slippage_bp_50ms").cast(pl.Float64),
        pl.col("exit_slippage_bp").cast(pl.Float64),
        pl.col("capital_time_seconds").cast(pl.Float64),
        pl.col("normalization_notional_twd").cast(pl.Float64),
        *(pl.col(name).cast(pl.Float64) for name in DEFAULT_COST_COLUMNS),
        pl.col("cost_profile_version").cast(pl.String),
    )
    if paths.select("path_id").n_unique() != paths.height:
        raise ValueError("V2 adapter generated duplicate path_id values")
    expected = exit_policy_facts.select(policy_key).n_unique()
    if paths.height != expected:
        raise ValueError("V2 adapter did not preserve one path per joint policy")
    _validate_terminal_partition(paths)
    return paths.sort(
        [
            "Date",
            "ValueCode",
            "entry_route",
            "entry_lookup_action_id",
            "exit_rule_id",
            "exit_route",
            "path_id",
        ]
    )


def build_taker_taker_baseline_refs(
    action_facts: pl.DataFrame,
    taker_exit_facts: pl.DataFrame,
) -> pl.DataFrame:
    """Normalize current T/T facts to one reference per entry alias/rule.

    Exit route is intentionally absent from the reference key.  The same T/T
    observation can be joined to both maker-exit routes for paired analysis,
    but it remains one baseline observation in ``baseline_refs``.
    """

    if taker_exit_facts.is_empty():
        return pl.DataFrame(schema=_empty_baseline_schema())
    if action_facts.is_empty():
        raise ValueError("taker/taker exit facts cannot exist without entry actions")
    _require(action_facts, _ACTION_REQUIRED, "execution action facts")
    _require(taker_exit_facts, _TAKER_BASELINE_REQUIRED, "taker/taker exit facts")
    if action_facts.select("policy_generation_id").n_unique() != action_facts.height:
        raise ValueError("action facts must be unique by policy_generation_id")
    baseline_key = ["policy_generation_id", "exit_rule_id"]
    if taker_exit_facts.select(baseline_key).n_unique() != taker_exit_facts.height:
        raise ValueError("taker/taker facts must be unique by entry alias and rule")
    invalid = taker_exit_facts.filter(
        ~pl.col("branch_status").is_in(_TAKER_BASELINE_BRANCHES).fill_null(False)
    )
    if invalid.height:
        raise ValueError("taker/taker facts contain an unsupported branch_status")

    action = _action_for_baseline_join(action_facts)
    joined = taker_exit_facts.join(
        action,
        on="policy_generation_id",
        how="left",
        validate="m:1",
    )
    if joined.filter(pl.col("_action_Date").is_null()).height:
        raise ValueError("taker/taker fact does not resolve to an entry action alias")
    _validate_baseline_identity(joined)
    _validate_prior_lineage(
        joined,
        source_column="exit_rule_source_asof_date",
        allow_null_for_branch="exit_policy_unassigned",
    )
    # The M/T exit study is conditional on a fully established position, so
    # its paired T/T baseline must use the exact same entry aliases.
    joined = joined.filter(
        (pl.col("_action_full_fill") == True)  # noqa: E712
        & (pl.col("_action_entry_hedge_status") == "executable")
    )
    if joined.is_empty():
        return pl.DataFrame(schema=_empty_baseline_schema())

    records = [_map_taker_baseline(row) for row in joined.iter_rows(named=True)]
    refs = pl.from_dicts(records, infer_schema_length=None).with_columns(
        pl.col("baseline_gross_cycle_pnl_twd").cast(pl.Float64),
        pl.col("baseline_filled_cashflow_before_cost_bp").cast(pl.Float64),
        pl.col("normalization_notional_twd").cast(pl.Float64),
    )
    if refs.select("taker_taker_baseline_ref_id").n_unique() != refs.height:
        raise ValueError("duplicate taker/taker baseline reference id")
    physical_entry_key = [
        "Date",
        "ValueCode",
        "QuoteCode",
        "entry_route",
        "entry_raw_order_fact_id",
    ]
    exit_policy_signature_key = [
        *physical_entry_key,
        "entry_parameter_version",
        "exit_rule_id",
        "exit_rule_source_asof_date",
        "exit_threshold_basis_bp",
    ]
    _validate_physical_baseline_aliases(refs, exit_policy_signature_key)
    alias_counts = refs.group_by(exit_policy_signature_key).agg(
        pl.len().alias("physical_entry_alias_count")
    )
    refs = refs.join(
        alias_counts,
        on=exit_policy_signature_key,
        how="left",
        validate="m:1",
    ).with_columns(
        pl.struct(physical_entry_key)
        .map_elements(
            lambda value: _stable_id(
                "physical_entry_dependency",
                *(value[name] for name in physical_entry_key),
            ),
            return_dtype=pl.String,
        )
        .alias("physical_entry_dependency_id"),
        pl.struct(exit_policy_signature_key)
        .map_elements(
            lambda value: _stable_id(
                "taker_taker_exit_policy_signature",
                *(value[name] for name in exit_policy_signature_key),
            ),
            return_dtype=pl.String,
        )
        .alias("baseline_exit_policy_signature_id"),
        pl.struct(exit_policy_signature_key)
        .map_elements(
            lambda value: _stable_id(
                "physical_taker_taker_baseline",
                *(value[name] for name in exit_policy_signature_key),
            ),
            return_dtype=pl.String,
        )
        .alias("physical_taker_taker_baseline_id"),
        (1.0 / pl.col("physical_entry_alias_count")).alias(
            "physical_entry_alias_weight"
        ),
        (pl.col("physical_entry_alias_count") > 1).alias(
            "baseline_shared_across_entry_aliases"
        ),
        pl.lit(False).alias("entry_alias_is_independent_observation"),
        pl.lit(False).alias(
            "entry_q_alias_rows_safe_to_sum_as_independent"
        ),
        pl.lit(True).alias("exit_rule_is_alternative_policy"),
        pl.lit(False).alias(
            "exit_policy_signature_rows_safe_to_sum_as_independent"
        ),
        pl.lit(False).alias(
            "baseline_exit_rule_rows_safe_to_sum_as_independent"
        ),
    )
    refs = _conform_to_schema(
        refs, _empty_baseline_schema(), "taker/taker baseline refs"
    )
    return refs.sort(
        [
            "Date",
            "ValueCode",
            "entry_route",
            "entry_lookup_action_id",
            "exit_rule_id",
        ]
    )


def build_matched_exit_style_comparison(
    maker_paths: pl.DataFrame,
    baseline_refs: pl.DataFrame,
) -> MatchedExitStyleComparison:
    """Pair maker routes to unique T/T references without duplicating weight."""

    maker_required = {
        "path_id",
        "physical_path_id",
        "Date",
        "ValueCode",
        "QuoteCode",
        "entry_route",
        "entry_policy_generation_id",
        "entry_lookup_action_id",
        "exit_rule_id",
        "exit_threshold_basis_bp",
        "exit_rule_source_asof_date",
        "exit_route",
        "exit_style",
        "outcome_status",
        "terminal_branch",
        "filled_cashflow_before_cost_bp",
        "physical_exit_raw_order_fact_id",
        "exit_rule_rows_safe_to_sum_as_independent",
    }
    baseline_required = set(_empty_baseline_schema())
    _require(maker_paths, maker_required, "maker terminal paths")
    _require(baseline_refs, baseline_required, "taker/taker baseline refs")
    if maker_paths.is_empty() and baseline_refs.is_empty():
        return MatchedExitStyleComparison(
            baseline_refs,
            pl.DataFrame(schema=_empty_pair_schema()),
            pl.DataFrame(schema=_empty_outcome_matrix_schema()),
        )
    if maker_paths.is_empty() or baseline_refs.is_empty():
        raise ValueError("maker paths and baseline refs must both be present")
    if maker_paths.filter(pl.col("exit_style") != MAKER_TAKER_EXIT_STYLE).height:
        raise ValueError("matched comparison accepts maker_taker paths only")
    maker_key = [
        "entry_policy_generation_id",
        "exit_rule_id",
        "exit_route",
    ]
    if maker_paths.select(maker_key).n_unique() != maker_paths.height:
        raise ValueError("maker paths duplicate an entry/rule/route policy")
    baseline_key = ["entry_policy_generation_id", "exit_rule_id"]
    if baseline_refs.select(baseline_key).n_unique() != baseline_refs.height:
        raise ValueError("baseline refs duplicate an entry/rule observation")
    maker_baseline_keys = maker_paths.select(baseline_key).unique()
    reference_keys = baseline_refs.select(baseline_key)
    if maker_baseline_keys.join(
        reference_keys, on=baseline_key, how="anti"
    ).height:
        raise ValueError("maker paths are missing matched T/T baseline refs")
    if reference_keys.join(
        maker_baseline_keys, on=baseline_key, how="anti"
    ).height:
        raise ValueError("T/T baseline refs contain unmatched entry/rule aliases")

    right = baseline_refs.rename(
        {
            "Date": "_baseline_Date",
            "ValueCode": "_baseline_ValueCode",
            "QuoteCode": "_baseline_QuoteCode",
            "entry_route": "_baseline_entry_route",
            "entry_raw_order_fact_id": "_baseline_entry_raw_order_fact_id",
            "entry_lookup_action_id": "_baseline_entry_lookup_action_id",
        }
    )
    pairs = maker_paths.select(
        "path_id",
        "physical_path_id",
        "Date",
        "ValueCode",
        "QuoteCode",
        "entry_route",
        "entry_policy_generation_id",
        "entry_raw_order_fact_id",
        "entry_lookup_action_id",
        "exit_rule_id",
        "exit_threshold_basis_bp",
        "exit_rule_source_asof_date",
        "exit_route",
        "outcome_status",
        "terminal_branch",
        "filled_cashflow_before_cost_bp",
        "physical_exit_raw_order_fact_id",
        "exit_rule_rows_safe_to_sum_as_independent",
    ).rename(
        {
            "outcome_status": "maker_outcome_status",
            "terminal_branch": "maker_terminal_branch",
            "filled_cashflow_before_cost_bp": "maker_gross_cycle_bp",
            "exit_threshold_basis_bp": "maker_exit_threshold_basis_bp",
            "exit_rule_source_asof_date": "maker_exit_rule_source_asof_date",
        }
    ).join(
        right,
        on=baseline_key,
        how="left",
        validate="m:1",
    )
    if pairs.filter(pl.col("taker_taker_baseline_ref_id").is_null()).height:
        raise ValueError("maker path is missing its T/T baseline reference")
    _validate_pair_identity(pairs)

    match_counts = pairs.group_by("taker_taker_baseline_ref_id").agg(
        pl.len().alias("maker_route_matches_for_baseline"),
        pl.col("exit_route").n_unique().alias("maker_routes_for_baseline"),
    )
    pairs = pairs.join(
        match_counts,
        on="taker_taker_baseline_ref_id",
        how="left",
        validate="m:1",
    ).with_columns(
        (
            1.0 / pl.col("maker_route_matches_for_baseline")
        ).alias("baseline_reference_weight"),
        (
            pl.col("physical_entry_alias_weight")
            / pl.col("maker_route_matches_for_baseline")
        ).alias("physical_baseline_observation_weight"),
        (pl.col("maker_route_matches_for_baseline") > 1).alias(
            "baseline_shared_across_exit_routes"
        ),
        pl.lit(False).alias("route_copy_is_independent_baseline"),
        pl.lit(True).alias("alternative_policy_comparison"),
    )
    maker_same_day = (
        (pl.col("maker_outcome_status") == "known")
        & pl.col("maker_terminal_branch").is_in(
            ["same_day_target_exit", "same_day_aggressive_exit"]
        )
        & pl.col("maker_gross_cycle_bp").is_not_null()
    )
    baseline_same_day = (
        (pl.col("baseline_outcome_status") == "known")
        & pl.col("baseline_terminal_branch").is_in(
            ["same_day_target_exit", "same_day_aggressive_exit"]
        )
        & pl.col("baseline_filled_cashflow_before_cost_bp").is_not_null()
    )
    pairs = pairs.with_columns(
        (maker_same_day & baseline_same_day).alias("both_same_day_known"),
        pl.when(maker_same_day & baseline_same_day)
        .then(
            pl.col("maker_gross_cycle_bp")
            - pl.col("baseline_filled_cashflow_before_cost_bp")
        )
        .otherwise(None)
        .alias("paired_gross_delta_bp"),
    ).sort(
        [
            "Date",
            "ValueCode",
            "entry_route",
            "entry_lookup_action_id",
            "exit_rule_id",
            "exit_route",
        ]
    )

    bad_weight = pairs.group_by("taker_taker_baseline_ref_id").agg(
        pl.col("baseline_reference_weight").sum().alias("weight_sum")
    ).filter((pl.col("weight_sum") - 1.0).abs() > 1e-12)
    if bad_weight.height:
        raise ValueError("T/T baseline reference weights do not sum to one")
    bad_physical_weight = pairs.group_by(
        "physical_taker_taker_baseline_id"
    ).agg(
        pl.col("physical_baseline_observation_weight")
        .sum()
        .alias("weight_sum")
    ).filter((pl.col("weight_sum") - 1.0).abs() > 1e-12)
    if bad_physical_weight.height:
        raise ValueError("physical T/T observation weights do not sum to one")

    pairs = _conform_to_schema(
        pairs, _empty_pair_schema(), "matched maker/taker vs taker/taker pairs"
    )

    matrix_keys = [
        "entry_lookup_action_id",
        "exit_rule_id",
        "exit_route",
        "maker_outcome_status",
        "maker_terminal_branch",
        "baseline_outcome_status",
        "baseline_terminal_branch",
    ]
    outcome_matrix = pairs.group_by(matrix_keys).agg(
        pl.len().alias("matched_policy_pairs"),
        pl.col("taker_taker_baseline_ref_id")
        .n_unique()
        .alias("unique_taker_taker_baseline_refs"),
        pl.col("baseline_reference_weight")
        .sum()
        .alias("baseline_alias_effective_observations"),
        pl.col("physical_baseline_observation_weight")
        .sum()
        .alias("baseline_effective_observations"),
        pl.col("both_same_day_known").sum().alias("both_same_day_known_pairs"),
        pl.col("paired_gross_delta_bp")
        .drop_nulls()
        .median()
        .alias("paired_gross_delta_bp_p50"),
    ).with_columns(
        pl.lit(False).alias("exit_route_rows_safe_to_sum_as_independent_baselines")
    ).sort(matrix_keys)
    return MatchedExitStyleComparison(
        baseline_refs=baseline_refs,
        pairs=pairs,
        outcome_matrix=outcome_matrix,
    )


def build_flat_non_price_cost_sensitivity(
    terminal_paths: pl.DataFrame,
    config: FlatCostSensitivityConfig = FlatCostSensitivityConfig(),
) -> pl.DataFrame:
    """Subtract one flat non-price cycle cost in a separate diagnostic table.

    This output intentionally omits the standard EV cost columns.  It cannot
    satisfy :func:`ev_surface.build_daily_ev_lookup` and must not be merged
    back into strict terminal facts.  The 19 bp assumption is applied once
    only to known completed same-day cycles; unresolved paths remain null.
    """

    config.validate()
    required = {
        "path_id",
        "physical_path_id",
        "Date",
        "ValueCode",
        "entry_route",
        "entry_policy_generation_id",
        "entry_lookup_action_id",
        "exit_rule_id",
        "exit_style",
        "exit_route",
        "outcome_status",
        "terminal_branch",
        "filled_cashflow_before_cost_bp",
        "hedge_slippage_bp_50ms",
        "exit_slippage_bp",
        "cost_profile_version",
        *DEFAULT_COST_COLUMNS,
    }
    _require(terminal_paths, required, "terminal paths")
    if terminal_paths.is_empty():
        return pl.DataFrame(schema=_empty_sensitivity_schema())
    if terminal_paths.select("path_id").n_unique() != terminal_paths.height:
        raise ValueError("terminal paths must be unique by path_id")
    completed = (
        (pl.col("outcome_status") == "known")
        & pl.col("terminal_branch").is_in(
            ["same_day_target_exit", "same_day_aggressive_exit"]
        )
    )
    malformed = terminal_paths.filter(
        completed
        & (
            pl.col("filled_cashflow_before_cost_bp").is_null()
            | ~pl.col("filled_cashflow_before_cost_bp").is_finite()
        )
    )
    if malformed.height:
        raise ValueError("known completed cycle is missing finite gross cashflow")

    # Strict V2 facts are unpriced.  Reject a caller accidentally passing a
    # formally costed table, rather than layering a flat assumption on top.
    already_costed = terminal_paths.filter(
        pl.col("cost_profile_version").is_not_null()
        | pl.any_horizontal(
            *(pl.col(name).is_not_null() for name in DEFAULT_COST_COLUMNS)
        )
    )
    if already_costed.height:
        raise ValueError(
            "flat sensitivity cannot be layered on paths with strict cost inputs"
        )

    assumed = float(config.assumed_cycle_cost_bp)
    return terminal_paths.select(
        "path_id",
        "physical_path_id",
        "Date",
        "ValueCode",
        "entry_route",
        "entry_policy_generation_id",
        "entry_lookup_action_id",
        "exit_rule_id",
        "exit_style",
        "exit_route",
        "outcome_status",
        "terminal_branch",
        pl.col("filled_cashflow_before_cost_bp").alias("gross_cycle_bp"),
        "hedge_slippage_bp_50ms",
        "exit_slippage_bp",
    ).with_columns(
        pl.lit(config.sensitivity_id).alias("cost_sensitivity_id"),
        pl.lit(config.cost_scope).alias("cost_scope"),
        pl.lit(assumed).alias("assumed_non_price_cycle_cost_bp"),
        completed.alias("sensitivity_applicable"),
        pl.when(completed).then(pl.lit(1)).otherwise(pl.lit(0)).cast(pl.Int64).alias(
            "applied_cycle_cost_count"
        ),
        pl.when(completed)
        .then(pl.col("gross_cycle_bp") - assumed)
        .otherwise(None)
        .alias("conditional_net_after_assumed_cost_bp"),
        pl.when(completed)
        .then(pl.col("gross_cycle_bp") - assumed > 0.0)
        .otherwise(None)
        .alias("conditional_positive_after_assumed_cost"),
        pl.lit(True).alias("price_slippage_already_in_gross"),
        pl.lit(False).alias("hedge_slippage_subtracted_again"),
        pl.lit(False).alias("exit_slippage_subtracted_again"),
        pl.lit(True).alias("analysis_only"),
        pl.lit(False).alias("production_eligible"),
        pl.lit(False).alias("eligible_for_strict_ev_lookup"),
        pl.lit(False).alias("strict_cost_columns_modified"),
    ).sort(
        [
            "Date",
            "ValueCode",
            "entry_route",
            "entry_lookup_action_id",
            "exit_rule_id",
            "exit_route",
        ]
    )


def build_nominal_instant_cancel_v0_sensitivity(
    position_policy_facts: pl.DataFrame,
    action_facts: pl.DataFrame,
    cost_bp: float = 19.0,
) -> pl.DataFrame:
    """Evaluate the explicit instantaneous-cancel V0 counterfactual.

    This is intentionally separate from strict terminal paths.  In
    particular, a strict ``cancel_race_unknown`` row may use its preserved
    ``nominal_instant_cancel_v0_branch=flat_same_day`` under the user's V0
    assumption, but the result remains analysis-only and cannot enter the
    strict EV lookup.  Actual maker and delayed taker fill prices are already
    inside ``gross_cycle_pnl_twd``; the flat cost subtracts only non-price
    fees/tax/commission once.
    """

    sensitivity_config = FlatCostSensitivityConfig(
        assumed_cycle_cost_bp=cost_bp,
        sensitivity_id="nominal_instant_cancel_v0_non_price_cost_v1",
    )
    sensitivity_config.validate()
    if position_policy_facts.is_empty():
        return pl.DataFrame(schema=_empty_nominal_cancel_sensitivity_schema())
    if action_facts.is_empty():
        raise ValueError("nominal cancel facts require entry action facts")
    _require(action_facts, _ACTION_REQUIRED, "execution action facts")
    _require(
        position_policy_facts,
        {
            *EXIT_POLICY_FACT_REQUIRED_COLUMNS,
            "nominal_instant_cancel_v0_branch",
            "instant_cancel_v0",
            "raw_candidate_count",
        },
        "exit maker position policy facts",
    )
    if action_facts.select("policy_generation_id").n_unique() != action_facts.height:
        raise ValueError("action facts must be unique by policy_generation_id")
    policy_key = ["entry_policy_generation_id", "exit_rule_id", "exit_route"]
    if position_policy_facts.select(policy_key).n_unique() != position_policy_facts.height:
        raise ValueError(
            "position policy facts must be unique by entry alias, rule and route"
        )
    if position_policy_facts.schema["instant_cancel_v0"] != pl.Boolean:
        raise ValueError("instant_cancel_v0 must be boolean")
    if position_policy_facts.filter(
        (pl.col("instant_cancel_v0") != True).fill_null(True)  # noqa: E712
    ).height:
        raise ValueError(
            "nominal cancel sensitivity requires the explicit instant_cancel_v0 model"
        )
    invalid_nominal = position_policy_facts.filter(
        ~pl.col("nominal_instant_cancel_v0_branch")
        .is_in(SUPPORTED_POLICY_BRANCHES)
        .fill_null(False)
    )
    if invalid_nominal.height:
        raise ValueError("unsupported nominal_instant_cancel_v0_branch")
    _validate_exit_policy_facts(position_policy_facts)

    joined = position_policy_facts.join(
        _action_for_join(action_facts),
        on="entry_policy_generation_id",
        how="left",
        validate="m:1",
    )
    if joined.filter(pl.col("_action_Date").is_null()).height:
        raise ValueError("position policy fact does not resolve to an entry alias")
    _validate_action_identity(joined)
    _validate_established_entry_aliases(joined)
    records = [
        _map_nominal_cancel_sensitivity(
            row,
            assumed_cost_bp=float(sensitivity_config.assumed_cycle_cost_bp),
            sensitivity_id=sensitivity_config.sensitivity_id,
        )
        for row in joined.iter_rows(named=True)
    ]
    result = pl.from_dicts(records, infer_schema_length=None).with_columns(
        pl.col("gross_cycle_bp").cast(pl.Float64),
        pl.col("conditional_net_after_assumed_cost_bp").cast(pl.Float64),
        pl.col("normalization_notional_twd").cast(pl.Float64),
    )
    if result.select("nominal_cancel_sensitivity_path_id").n_unique() != result.height:
        raise ValueError("duplicate nominal instant-cancel sensitivity path id")
    return result.sort(
        [
            "Date",
            "ValueCode",
            "entry_route",
            "entry_lookup_action_id",
            "exit_rule_id",
            "exit_route",
        ]
    )


def _map_nominal_cancel_sensitivity(
    row: Mapping[str, object],
    *,
    assumed_cost_bp: float,
    sensitivity_id: str,
) -> dict[str, object]:
    nominal_branch = str(row["nominal_instant_cancel_v0_branch"])
    strict_branch = str(row["branch_status"])
    if nominal_branch == KNOWN_TARGET_EXIT_BRANCH:
        modeled_outcome = "known"
        modeled_terminal_branch: str | None = "same_day_target_exit"
        if row["exit_hedge_status"] != "executable":
            raise ValueError(
                "nominal flat_same_day requires an actual executable exit hedge"
            )
        gross_bp, notional = _normalise_gross(
            row["gross_cycle_pnl_twd"],
            row["_action_entry_spot_price"],
            row["_action_entry_hedge_contract_size_shares"],
            True,
        )
        if gross_bp is None or notional is None:
            raise ValueError(
                "nominal flat_same_day requires finite gross PnL and entry notional"
            )
        net_bp: float | None = gross_bp - assumed_cost_bp
        applied_cost_count = 1
    elif nominal_branch in {
        *CENSORED_CARRY_BRANCHES,
        "no_fill_before_cancel_request",
    }:
        modeled_outcome = "censored"
        modeled_terminal_branch = None
        gross_bp = None
        notional = None
        net_bp = None
        applied_cost_count = 0
    else:
        modeled_outcome = "unknown"
        modeled_terminal_branch = None
        gross_bp = None
        notional = None
        net_bp = None
        applied_cost_count = 0

    active_sibling_count = _nonnegative_integer(
        row["oco_active_sibling_cancel_count"],
        "oco_active_sibling_cancel_count",
    )
    prior_unacked_count = _nonnegative_integer(
        row["prior_unacked_cancel_count_before_winner"],
        "prior_unacked_cancel_count_before_winner",
    )
    any_unacked_count = active_sibling_count + prior_unacked_count
    raw_candidate_count = _nonnegative_integer(
        row["raw_candidate_count"], "raw_candidate_count"
    )
    trial_id = str(row["exit_policy_trial_id"])
    return {
        "nominal_cancel_sensitivity_path_id": _stable_id(
            "nominal_instant_cancel_v0", trial_id
        ),
        "exit_policy_trial_id": trial_id,
        "physical_entry_id": str(row["entry_raw_order_fact_id"]),
        "physical_exit_raw_order_fact_id": row[
            "oco_winner_raw_candidate_fact_id"
        ],
        "Date": str(row["Date"]),
        "ValueCode": str(row["ValueCode"]),
        "QuoteCode": str(row["QuoteCode"]),
        "entry_route": str(row["entry_route"]),
        "entry_policy_generation_id": str(row["entry_policy_generation_id"]),
        "entry_lookup_action_id": str(row["_action_lookup_action_id"]),
        "exit_rule_id": str(row["exit_rule_id"]),
        "exit_route": str(row["exit_route"]),
        "exit_style": MAKER_TAKER_EXIT_STYLE,
        "strict_branch_status": strict_branch,
        "strict_terminal_outcome": bool(row["terminal_outcome"]),
        "strict_needs_next_session_label": bool(
            row["needs_next_session_label"]
        ),
        "strict_projection_safe": bool(row["oco_position_projection_safe"]),
        "strict_cancel_ack_observed": bool(row["cancel_ack_observed"]),
        "strict_cancel_race_modeled": bool(row["cancel_race_modeled"]),
        "nominal_instant_cancel_v0_branch": nominal_branch,
        "modeled_outcome_status": modeled_outcome,
        "modeled_terminal_branch": modeled_terminal_branch,
        "raw_candidate_count": raw_candidate_count,
        "oco_active_sibling_cancel_count": active_sibling_count,
        "prior_unacked_cancel_count_before_winner": prior_unacked_count,
        "any_unacked_cancel_count": any_unacked_count,
        "has_active_sibling_cancel_request": active_sibling_count > 0,
        "has_prior_unacked_cancel_request": prior_unacked_count > 0,
        "has_any_unacked_cancel_request": any_unacked_count > 0,
        "cancel_rate_numerator": int(any_unacked_count > 0),
        "cancel_rate_denominator": 1,
        "strict_cancel_ambiguity_overridden": (
            strict_branch == "cancel_race_unknown"
            and nominal_branch == KNOWN_TARGET_EXIT_BRANCH
        ),
        "gross_cycle_bp": gross_bp,
        "normalization_notional_twd": notional,
        "cost_sensitivity_id": sensitivity_id,
        "cost_scope": NON_PRICE_COST_SCOPE,
        "assumed_non_price_cycle_cost_bp": assumed_cost_bp,
        "applied_cycle_cost_count": applied_cost_count,
        "conditional_net_after_assumed_cost_bp": net_bp,
        "conditional_positive_after_assumed_cost": (
            net_bp > 0.0 if net_bp is not None else None
        ),
        "price_slippage_already_in_gross": True,
        "hedge_slippage_subtracted_again": False,
        "exit_slippage_subtracted_again": False,
        "model_version": "nominal_instant_cancel_v0",
        "model_assumption": True,
        "analysis_only": True,
        "production_eligible": False,
        "eligible_for_strict_ev_lookup": False,
        "pathwise_ev_ready": False,
        "exit_rule_rows_safe_to_sum_as_independent": False,
    }


def _action_for_join(action_facts: pl.DataFrame) -> pl.DataFrame:
    selected = action_facts.select(
        "Date",
        "ValueCode",
        "QuoteCode",
        "route",
        "parameter_version",
        "lookup_action_id",
        "raw_order_fact_id",
        "policy_generation_id",
        "rank_bucket",
        "queue_bucket",
        "tod_bucket",
        "freshness_bucket",
        "intended_quantity",
        "submit_recv_time_ns",
        "terminal_recv_time_ns",
        "queue_known",
        "any_fill",
        "full_fill",
        "partial_fill",
        "cancel_required",
        "entry_hedge_status",
        "entry_hedge_signed_total_slippage_bp",
        "entry_hedge_contract_size_shares",
        "entry_spot_price",
    )
    return selected.rename(
        {
            "policy_generation_id": "entry_policy_generation_id",
            **{
                column: f"_action_{column}"
                for column in selected.columns
                if column != "policy_generation_id"
            },
        }
    )


def _action_for_baseline_join(action_facts: pl.DataFrame) -> pl.DataFrame:
    selected = action_facts.select(
        "Date",
        "ValueCode",
        "QuoteCode",
        "route",
        "parameter_version",
        "lookup_action_id",
        "raw_order_fact_id",
        "policy_generation_id",
        "entry_spot_price",
        "entry_hedge_contract_size_shares",
        "full_fill",
        "entry_hedge_status",
    )
    return selected.rename(
        {
            column: f"_action_{column}"
            for column in selected.columns
            if column != "policy_generation_id"
        }
    )


def _validate_exit_policy_facts(frame: pl.DataFrame) -> None:
    invalid_branch = frame.filter(
        ~pl.col("branch_status").is_in(SUPPORTED_POLICY_BRANCHES).fill_null(False)
    )
    if invalid_branch.height:
        raise ValueError("exit maker policy facts contain unsupported branch_status")
    if frame.filter(
        (pl.col("exit_style") != MAKER_TAKER_EXIT_STYLE).fill_null(True)
    ).height:
        raise ValueError("V2 adapter accepts maker_taker exit policy facts only")
    invalid_route = frame.filter(
        (~pl.col("exit_route").is_in(SUPPORTED_EXIT_ROUTES)).fill_null(True)
    )
    if invalid_route.height:
        raise ValueError("exit maker policy facts contain an unsupported exit route")
    _validate_prior_lineage(frame, source_column="exit_rule_source_asof_date")

    malformed_text = frame.filter(
        pl.any_horizontal(
            *(
                pl.col(column).is_null()
                | (pl.col(column).cast(pl.String).str.len_chars() == 0)
                for column in (
                    "entry_policy_generation_id",
                    "entry_raw_order_fact_id",
                    "exit_policy_trial_id",
                    "exit_rule_id",
                    "exit_lifecycle_policy_version",
                    "exit_queue_scenario",
                )
            )
        )
    )
    if malformed_text.height:
        raise ValueError("exit maker policy identity/version fields must be present")
    invalid_threshold = frame.filter(
        pl.col("exit_threshold_basis_bp").is_null()
        | ~pl.col("exit_threshold_basis_bp").is_finite()
    )
    if invalid_threshold.height:
        raise ValueError("exit threshold must be finite")
    boolean_columns = (
        "oco_winner_raw_identity_excludes_rule",
        "oco_position_projection_safe",
        "terminal_outcome",
        "needs_next_session_label",
        "cancel_ack_observed",
        "cancel_race_modeled",
        "joint_volume_allocated",
    )
    if any(frame.schema[column] != pl.Boolean for column in boolean_columns):
        raise ValueError("terminal/cancel/race/volume flags must be booleans")
    if frame.filter(
        pl.any_horizontal(*(pl.col(column).is_null() for column in boolean_columns))
    ).height:
        raise ValueError("terminal/cancel/race/volume flags must be explicit booleans")

    canonical_signatures: dict[str, tuple[object, ...]] = {}
    for row in frame.iter_rows(named=True):
        canonical_id = _validate_canonical_exit_raw_identity(row)
        if canonical_id is not None:
            signature = _canonical_exit_raw_signature(row)
            previous = canonical_signatures.setdefault(canonical_id, signature)
            if previous != signature:
                raise ValueError(
                    "canonical exit raw id maps to inconsistent physical signatures"
                )
        route = str(row["exit_route"])
        maker_quantity = _positive_integer(
            row["exit_maker_quantity"], "exit_maker_quantity"
        )
        hedge_quantity = _positive_integer(
            row["exit_hedge_quantity"], "exit_hedge_quantity"
        )
        expected = (
            (FUTURE_HEDGE_CONTRACTS, SPOT_HEDGE_LOTS)
            if route == FUTURE_BID_EXIT_ROUTE
            else (SPOT_HEDGE_LOTS, FUTURE_HEDGE_CONTRACTS)
        )
        if (maker_quantity, hedge_quantity) != expected:
            raise ValueError(
                f"unexpected exit quantities for {route}: "
                f"{maker_quantity}/{hedge_quantity}"
            )
        sibling_cancel_count = _nonnegative_integer(
            row["oco_active_sibling_cancel_count"],
            "oco_active_sibling_cancel_count",
        )
        prior_unacked_cancel_count = _nonnegative_integer(
            row["prior_unacked_cancel_count_before_winner"],
            "prior_unacked_cancel_count_before_winner",
        )
        branch = str(row["branch_status"])
        terminal = row["terminal_outcome"] is True
        needs_next = row["needs_next_session_label"] is True
        if branch == KNOWN_TARGET_EXIT_BRANCH:
            if canonical_id is None:
                raise ValueError(
                    "flat_same_day requires a validated canonical exit raw identity"
                )
            if row["oco_position_projection_safe"] is not True:
                raise ValueError(
                    "flat_same_day requires a projection-safe OCO position"
                )
            if (
                sibling_cancel_count + prior_unacked_cancel_count > 0
                and row["cancel_ack_observed"] is not True
            ):
                raise ValueError(
                    "flat_same_day cannot retain unresolved sibling cancel or prior cancel "
                    "requests; study must emit cancel_race_unknown"
                )
            if not terminal or needs_next:
                raise ValueError("flat_same_day must be terminal and not need next day")
            if row["exit_hedge_status"] != "executable":
                raise ValueError("flat_same_day requires an executable exit hedge")
            if _finite_or_none(row["gross_cycle_pnl_twd"]) is None:
                raise ValueError("flat_same_day requires finite gross cycle PnL")
            if _integer_or_none(row["exit_decision_time_ns"]) is None:
                raise ValueError("flat_same_day requires an exit decision time")
        elif branch in CENSORED_CARRY_BRANCHES:
            if terminal or not needs_next:
                raise ValueError("EOD carry must be nonterminal and need next day")
        elif terminal:
            raise ValueError(
                "unresolved/cancel-request branch cannot claim terminal_outcome"
            )


def _canonical_exit_raw_signature(
    row: Mapping[str, object],
) -> tuple[object, ...]:
    return (
        str(row["entry_raw_order_fact_id"]),
        str(row["exit_route"]),
        row["oco_winner_spread_pair_epoch"],
        row["oco_winner_target_price_tick"],
        row["oco_winner_submit_recv_time_ns"],
        row["oco_winner_submit_event_sequence"],
        row["oco_winner_submit_row_index"],
    )


def _validate_canonical_exit_raw_identity(
    row: Mapping[str, object],
) -> str | None:
    """Validate the study's raw ID and prove rule/q exclusion by recomputing it."""

    id_column = "oco_winner_raw_candidate_fact_id"
    component_columns = (
        "oco_winner_spread_pair_epoch",
        "oco_winner_target_price_tick",
        "oco_winner_submit_recv_time_ns",
        "oco_winner_submit_event_sequence",
        "oco_winner_submit_row_index",
    )
    values = (row[id_column], *(row[name] for name in component_columns))
    present = tuple(value is not None for value in values)
    excludes_rule = row["oco_winner_raw_identity_excludes_rule"] is True
    if not any(present):
        if excludes_rule:
            raise ValueError(
                "winner raw identity cannot claim rule exclusion without an identity"
            )
        return None
    if not all(present):
        raise ValueError("winner canonical raw identity components must be all-or-none")
    if not excludes_rule:
        raise ValueError(
            "winner raw identity must explicitly exclude q alias, rule and threshold"
        )
    expected = canonical_exit_raw_candidate_fact_id(
        entry_raw_order_fact_id=str(row["entry_raw_order_fact_id"]),
        exit_route=str(row["exit_route"]),
        spread_pair_epoch=_integer(
            row["oco_winner_spread_pair_epoch"], "oco_winner_spread_pair_epoch"
        ),
        target_price_tick=_integer(
            row["oco_winner_target_price_tick"], "oco_winner_target_price_tick"
        ),
        submit_recv_time_ns=_integer(
            row["oco_winner_submit_recv_time_ns"],
            "oco_winner_submit_recv_time_ns",
        ),
        submit_event_sequence=_integer(
            row["oco_winner_submit_event_sequence"],
            "oco_winner_submit_event_sequence",
        ),
        submit_row_index=_integer(
            row["oco_winner_submit_row_index"], "oco_winner_submit_row_index"
        ),
    )
    actual = str(row[id_column])
    if actual != expected:
        raise ValueError(
            "winner raw candidate id is not the canonical rule-free physical id"
        )
    return actual


def _validate_established_entry_aliases(frame: pl.DataFrame) -> None:
    """Require every referenced action to have opened and hedged the position."""

    invalid = frame.filter(
        (pl.col("_action_full_fill") != True).fill_null(True)  # noqa: E712
        | (
            pl.col("_action_entry_hedge_status") != "executable"
        ).fill_null(True)
    )
    if invalid.height:
        raise ValueError(
            "exit maker policy facts are conditional on established entry aliases "
            "(full entry fill plus executable entry hedge)"
        )


def _validate_action_identity(frame: pl.DataFrame) -> None:
    pairs = (
        ("Date", "_action_Date"),
        ("ValueCode", "_action_ValueCode"),
        ("QuoteCode", "_action_QuoteCode"),
        ("entry_route", "_action_route"),
        ("entry_raw_order_fact_id", "_action_raw_order_fact_id"),
    )
    for left, right in pairs:
        mismatch = frame.filter(
            (
                pl.col(left).cast(pl.String)
                != pl.col(right).cast(pl.String)
            ).fill_null(True)
        )
        if mismatch.height:
            raise ValueError(f"entry action and exit policy disagree on {left}")


def _validate_baseline_identity(frame: pl.DataFrame) -> None:
    for left, right in (
        ("Date", "_action_Date"),
        ("ValueCode", "_action_ValueCode"),
        ("QuoteCode", "_action_QuoteCode"),
        ("route", "_action_route"),
        ("raw_order_fact_id", "_action_raw_order_fact_id"),
    ):
        if frame.filter(
            (
                pl.col(left).cast(pl.String)
                != pl.col(right).cast(pl.String)
            ).fill_null(True)
        ).height:
            raise ValueError(f"entry action and T/T baseline disagree on {left}")


def _validate_physical_baseline_aliases(
    frame: pl.DataFrame,
    exit_policy_signature_key: list[str],
) -> None:
    """Ensure exact duplicate aliases do not relabel one T/T policy path."""

    shared_outcome_columns = (
        "baseline_branch_status",
        "baseline_outcome_status",
        "baseline_terminal_branch",
        "baseline_exit_decision_time_ns",
        "baseline_gross_cycle_pnl_twd",
        "baseline_filled_cashflow_before_cost_bp",
        "normalization_notional_twd",
    )
    inconsistent = frame.group_by(exit_policy_signature_key).agg(
        *(pl.col(name).n_unique().alias(name) for name in shared_outcome_columns)
    ).filter(
        pl.any_horizontal(
            *(pl.col(name) != 1 for name in shared_outcome_columns)
        )
    )
    if inconsistent.height:
        raise ValueError(
            "aliases disagree on an identical T/T exit-policy signature"
        )


def _map_policy_path(
    row: Mapping[str, object],
    config: ExitMakerTerminalConfig,
) -> dict[str, object]:
    entry_status = _entry_fill_status(row)
    entry_hedge_status = _entry_hedge_status(row, entry_status)
    branch = str(row["branch_status"])
    _validate_entry_branch(entry_status, entry_hedge_status, branch)

    if branch == KNOWN_TARGET_EXIT_BRANCH:
        outcome_status = "known"
        terminal_branch: str | None = "same_day_target_exit"
        mapping_reason = "maker_exit_full_and_delayed_taker_hedge_flat"
    elif branch in CENSORED_CARRY_BRANCHES:
        outcome_status = "censored"
        terminal_branch = None
        mapping_reason = "full_position_open_at_eod_requires_overnight_label"
    else:
        outcome_status = "unknown"
        terminal_branch = None
        mapping_reason = _unknown_mapping_reason(branch)

    gross_bp, notional = _gross_cashflow_bp(row, terminal_branch)
    capital_time = _capital_time_seconds(row, terminal_branch)
    statuses = _execution_statuses(
        entry_status=entry_status,
        entry_hedge_status=entry_hedge_status,
        outcome_status=outcome_status,
        terminal_branch=terminal_branch,
    )
    state_bucket = "|".join(
        f"{name}={_required_text(row[f'_action_{name}'], name)}"
        for name in (
            "rank_bucket",
            "queue_bucket",
            "tod_bucket",
            "freshness_bucket",
        )
    )
    date = str(row["Date"])
    entry_policy_id = str(row["entry_policy_generation_id"])
    exit_rule_id = str(row["exit_rule_id"])
    exit_route = str(row["exit_route"])
    path_id = "|".join(
        (
            date,
            str(row["ValueCode"]),
            str(row["QuoteCode"]),
            str(row["entry_route"]),
            entry_policy_id,
            exit_rule_id,
            exit_route,
            MAKER_TAKER_EXIT_STYLE,
        )
    )
    entry_intended = _positive_integer(
        row["_action_intended_quantity"], "intended_quantity"
    )
    entry_hedge_quantity = _entry_hedge_quantity(str(row["entry_route"]))
    return {
        "path_id": path_id,
        # q aliases which share one entry raw fact retain the same physical
        # dependency id, while their policy path ids remain distinct.
        "physical_path_id": str(row["entry_raw_order_fact_id"]),
        "physical_entry_id": str(row["entry_raw_order_fact_id"]),
        "entry_raw_order_fact_id": str(row["entry_raw_order_fact_id"]),
        "physical_exit_raw_order_fact_id": row[
            "oco_winner_raw_candidate_fact_id"
        ],
        "canonical_exit_raw_identity_validated": row[
            "oco_winner_raw_candidate_fact_id"
        ] is not None,
        # Center/lower and q aliases are alternative policies even when the
        # canonical raw order is shared.  Never expose their rows as IID.
        "exit_rule_row_is_independent_physical_observation": False,
        "exit_rule_rows_safe_to_sum_as_independent": False,
        "entry_policy_generation_id": entry_policy_id,
        "exit_policy_trial_id": str(row["exit_policy_trial_id"]),
        "Date": date,
        "label_end_date": date,
        "ValueCode": str(row["ValueCode"]),
        "QuoteCode": str(row["QuoteCode"]),
        "route": str(row["entry_route"]),
        "entry_route": str(row["entry_route"]),
        "lookup_action_id": str(row["_action_lookup_action_id"]),
        "entry_lookup_action_id": str(row["_action_lookup_action_id"]),
        "parameter_version": str(row["_action_parameter_version"]),
        "entry_parameter_version": str(row["_action_parameter_version"]),
        "exit_rule_id": exit_rule_id,
        "exit_rule_source_asof_date": str(row["exit_rule_source_asof_date"]),
        "exit_threshold_basis_bp": float(row["exit_threshold_basis_bp"]),
        "exit_style": str(row["exit_style"]),
        "exit_route": exit_route,
        "state_family": config.state_family,
        "state_bucket": state_bucket,
        "lifecycle_policy_version": config.lifecycle_policy_version,
        "exit_lifecycle_policy_version": str(
            row["exit_lifecycle_policy_version"]
        ),
        "queue_scenario": config.queue_scenario,
        "exit_queue_scenario": str(row["exit_queue_scenario"]),
        "intended_maker_quantity": entry_intended,
        "hedge_quantity": entry_hedge_quantity,
        "exit_maker_quantity": int(row["exit_maker_quantity"]),
        "exit_hedge_quantity": int(row["exit_hedge_quantity"]),
        "outcome_status": outcome_status,
        "terminal_branch": terminal_branch,
        **statuses,
        "filled_cashflow_before_cost_bp": gross_bp,
        "hedge_slippage_bp_50ms": _finite_or_none(
            row["_action_entry_hedge_signed_total_slippage_bp"]
        ),
        "exit_slippage_bp": _finite_or_none(
            row["exit_hedge_signed_total_slippage_bp"]
        ),
        "capital_time_seconds": capital_time,
        "cost_profile_version": None,
        **{name: None for name in DEFAULT_COST_COLUMNS},
        "normalization_notional_twd": notional,
        "source_exit_branch_status": branch,
        "terminal_mapping_reason": mapping_reason,
        "terminal_mapping_version": config.mapping_version,
        "exit_cancel_ack_observed": bool(row["cancel_ack_observed"]),
        "exit_cancel_race_modeled": bool(row["cancel_race_modeled"]),
        "exit_active_sibling_cancel_count": int(
            row["oco_active_sibling_cancel_count"]
        ),
        "exit_prior_unacked_cancel_count_before_winner": int(
            row["prior_unacked_cancel_count_before_winner"]
        ),
        "cost_components_complete": False,
        "strict_ev_ready": False,
        "pathwise_ev_ready": False,
        "joint_volume_allocated": bool(row["joint_volume_allocated"]),
        "contains_target_day_outcome": True,
    }


def _map_taker_baseline(row: Mapping[str, object]) -> dict[str, object]:
    branch = str(row["branch_status"])
    if branch == "same_day_taker_exit" and row["terminal_outcome"] is True:
        outcome = "known"
        terminal: str | None = "same_day_aggressive_exit"
    elif branch == "carry_at_eod" and row["needs_next_session_label"] is True:
        outcome = "censored"
        terminal = None
    else:
        outcome = "unknown"
        terminal = None
    gross, notional = _normalise_gross(
        row["gross_cycle_pnl_twd"],
        row["_action_entry_spot_price"],
        row["_action_entry_hedge_contract_size_shares"],
        terminal is not None,
    )
    date = str(row["Date"])
    policy = str(row["policy_generation_id"])
    rule = str(row["exit_rule_id"])
    ref_id = _stable_id(
        "taker_taker_baseline",
        date,
        str(row["ValueCode"]),
        str(row["QuoteCode"]),
        str(row["route"]),
        policy,
        rule,
    )
    return {
        "taker_taker_baseline_ref_id": ref_id,
        "Date": date,
        "ValueCode": str(row["ValueCode"]),
        "QuoteCode": str(row["QuoteCode"]),
        "entry_route": str(row["route"]),
        "entry_policy_generation_id": policy,
        "entry_raw_order_fact_id": str(row["raw_order_fact_id"]),
        "entry_lookup_action_id": str(row["_action_lookup_action_id"]),
        "entry_parameter_version": str(row["_action_parameter_version"]),
        "exit_rule_id": rule,
        "exit_threshold_basis_bp": _finite_or_none(
            row["exit_threshold_basis_bp"]
        ),
        "exit_rule_source_asof_date": row["exit_rule_source_asof_date"],
        "baseline_exit_style": "taker_taker",
        "baseline_branch_status": branch,
        "baseline_outcome_status": outcome,
        "baseline_terminal_branch": terminal,
        "baseline_exit_decision_time_ns": _integer_or_none(
            row["exit_decision_time_ns"]
        ),
        "baseline_gross_cycle_pnl_twd": _finite_or_none(
            row["gross_cycle_pnl_twd"]
        ),
        "baseline_filled_cashflow_before_cost_bp": gross,
        "normalization_notional_twd": notional,
        "baseline_reference_is_unique": True,
        "baseline_is_route_independent": True,
        "baseline_costs_included": False,
        "baseline_pathwise_ev_ready": False,
    }


def _validate_entry_branch(
    entry_status: str,
    entry_hedge_status: str,
    branch: str,
) -> None:
    if entry_status == "no_fill":
        allowed = {"no_entry_fill", "not_opened_or_unhedged"}
    elif entry_status == "partial":
        allowed = {"partial_entry_unhedged", "not_opened_or_unhedged"}
    elif entry_status == "unknown":
        allowed = {"entry_fill_unknown", "not_opened_or_unhedged"}
    elif entry_hedge_status != "executable":
        allowed = {"entry_hedge_unpriceable", "not_opened_or_unhedged"}
    else:
        allowed = {
            KNOWN_TARGET_EXIT_BRANCH,
            *CENSORED_CARRY_BRANCHES,
            *UNKNOWN_EXIT_BRANCHES,
        }
    if branch not in allowed:
        raise ValueError(
            "entry execution state disagrees with exit maker policy branch: "
            f"{entry_status}/{entry_hedge_status}/{branch}"
        )


def _entry_fill_status(row: Mapping[str, object]) -> str:
    if row["_action_full_fill"] is True:
        return "full"
    if row["_action_partial_fill"] is True:
        return "partial"
    if row["_action_any_fill"] is False and row["_action_queue_known"] is True:
        return "no_fill"
    return "unknown"


def _entry_hedge_status(row: Mapping[str, object], entry_status: str) -> str:
    if entry_status == "no_fill":
        return "not_applicable"
    if entry_status != "full":
        return "unknown"
    status = row["_action_entry_hedge_status"]
    if status == "executable":
        return "executable"
    if status == "insufficient_depth":
        return "failed"
    if status is None:
        return "unknown"
    return "failed"


def _execution_statuses(
    *,
    entry_status: str,
    entry_hedge_status: str,
    outcome_status: str,
    terminal_branch: str | None,
) -> dict[str, str]:
    if terminal_branch == "same_day_target_exit":
        return {
            "cancel_status": "not_cancelled",
            "entry_fill_status": "full",
            "topup_status": "not_applicable",
            "hedge_50ms_status": "executable",
            "same_day_exit_status": "target_exit",
            "overnight_status": "not_applicable",
            "expiry_status": "not_applicable",
            "emergency_status": "not_triggered",
        }
    if outcome_status == "censored":
        return {
            "cancel_status": "not_cancelled",
            "entry_fill_status": "full",
            "topup_status": "not_applicable",
            "hedge_50ms_status": "executable",
            "same_day_exit_status": "no_exit",
            "overnight_status": "carried_open",
            "expiry_status": "unknown",
            "emergency_status": "unknown",
        }
    return {
        "cancel_status": "not_cancelled" if entry_status == "full" else "unknown",
        "entry_fill_status": entry_status,
        "topup_status": "unknown" if entry_status == "partial" else "not_applicable",
        "hedge_50ms_status": entry_hedge_status,
        "same_day_exit_status": "unknown",
        "overnight_status": "unknown",
        "expiry_status": "unknown",
        "emergency_status": "unknown",
    }


def _gross_cashflow_bp(
    row: Mapping[str, object], terminal_branch: str | None
) -> tuple[float | None, float | None]:
    return _normalise_gross(
        row["gross_cycle_pnl_twd"],
        row["_action_entry_spot_price"],
        row["_action_entry_hedge_contract_size_shares"],
        terminal_branch is not None,
    )


def _normalise_gross(
    gross_value: object,
    spot_value: object,
    shares_value: object,
    required: bool,
) -> tuple[float | None, float | None]:
    gross = _finite_or_none(gross_value)
    spot = _finite_or_none(spot_value)
    shares = _finite_or_none(shares_value)
    if not required:
        return None, None
    if gross is None or spot is None or shares is None:
        return None, None
    notional = spot * shares
    if notional <= 0:
        return None, None
    return gross / notional * 10_000.0, notional


def _capital_time_seconds(
    row: Mapping[str, object], terminal_branch: str | None
) -> float | None:
    if terminal_branch is None:
        return None
    submit = _integer_or_none(row["_action_submit_recv_time_ns"])
    terminal = _integer_or_none(row["exit_decision_time_ns"])
    if submit is None or terminal is None or terminal < submit:
        return None
    return (terminal - submit) / 1_000_000_000.0


def _entry_hedge_quantity(route: str) -> int:
    if route == FUTURE_ASK_ROUTE:
        return SPOT_HEDGE_LOTS
    if route == SPOT_BID_ROUTE:
        return FUTURE_HEDGE_CONTRACTS
    raise ValueError(f"unsupported entry route: {route}")


def _unknown_mapping_reason(branch: str) -> str:
    reasons = {
        "no_entry_fill": "entry_cancel_ack_and_race_unobserved",
        "partial_entry_unhedged": "partial_entry_requires_topup_or_emergency_close",
        "entry_fill_unknown": "entry_fill_or_queue_state_unknown",
        "entry_hedge_unpriceable": "entry_hedge_completion_unknown",
        "no_fill_before_cancel_request": "exit_cancel_request_is_not_cancel_ack",
        "partial_fill_then_cancel": "partial_exit_and_cancel_race_unresolved",
        "partial_fill_carry_at_eod": "partial_exit_residual_requires_terminal_label",
        "hedge_incomplete_residual": "exit_hedge_residual_unpriced",
        "hedge_incomplete_carry_at_eod": "exit_hedge_and_overnight_residual_unpriced",
        "fill_unknown_then_cancel": "exit_fill_and_cancel_race_unknown",
        "fill_unknown_at_eod": "exit_fill_state_unknown_at_eod",
        "cancel_race_unknown": "exit_cancel_ack_and_race_unobserved",
        "exit_policy_unassigned": "exit_policy_fact_unassigned",
        "not_opened_or_unhedged": "entry_position_not_established",
    }
    return reasons.get(branch, "unresolved_exit_maker_path")


def _validate_terminal_partition(paths: pl.DataFrame) -> None:
    key = ["entry_policy_generation_id", "exit_rule_id", "exit_route"]
    counts = paths.group_by(key).len(name="n")
    if counts.filter(pl.col("n") != 1).height:
        raise ValueError("joint exit policy does not have exactly one terminal row")
    bad_known = paths.filter(
        (pl.col("outcome_status") == "known")
        & (pl.col("terminal_branch") != "same_day_target_exit")
    )
    bad_unresolved = paths.filter(
        (pl.col("outcome_status") != "known")
        & pl.col("terminal_branch").is_not_null()
    )
    if bad_known.height or bad_unresolved.height:
        raise ValueError("exit policy terminal branches are not mutually exclusive")


def _validate_pair_identity(frame: pl.DataFrame) -> None:
    for left, right in (
        ("Date", "_baseline_Date"),
        ("ValueCode", "_baseline_ValueCode"),
        ("QuoteCode", "_baseline_QuoteCode"),
        ("entry_route", "_baseline_entry_route"),
        ("entry_raw_order_fact_id", "_baseline_entry_raw_order_fact_id"),
        ("entry_lookup_action_id", "_baseline_entry_lookup_action_id"),
    ):
        if frame.filter(
            (
                pl.col(left).cast(pl.String)
                != pl.col(right).cast(pl.String)
            ).fill_null(True)
        ).height:
            raise ValueError(f"maker path and T/T baseline disagree on {left}")
    threshold_mismatch = frame.filter(
        (
            pl.col("maker_exit_threshold_basis_bp")
            != pl.col("exit_threshold_basis_bp")
        ).fill_null(True)
    )
    source_mismatch = frame.filter(
        (
            pl.col("maker_exit_rule_source_asof_date").cast(pl.String)
            != pl.col("exit_rule_source_asof_date").cast(pl.String)
        ).fill_null(True)
    )
    if threshold_mismatch.height or source_mismatch.height:
        raise ValueError(
            "maker path and T/T baseline disagree on the exit-policy signature"
        )


def _validate_prior_lineage(
    frame: pl.DataFrame,
    *,
    source_column: str,
    allow_null_for_branch: str | None = None,
) -> None:
    invalid = pl.col(source_column).is_null() | (
        pl.col(source_column).cast(pl.String) >= pl.col("Date").cast(pl.String)
    )
    if allow_null_for_branch is not None:
        invalid = (pl.col("branch_status") != allow_null_for_branch) & invalid
    if frame.filter(invalid).height:
        raise ValueError(f"{source_column} must be strictly before Date")


def _stable_id(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]


def _finite_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _integer_or_none(value: object) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _required_text(value: object, name: str) -> str:
    if value is None or not str(value).strip():
        raise ValueError(f"{name} must be present")
    return str(value)


def _require(frame: pl.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def _conform_to_schema(
    frame: pl.DataFrame,
    schema: Mapping[str, pl.DataType],
    source: str,
) -> pl.DataFrame:
    expected = set(schema)
    actual = set(frame.columns)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{source} schema mismatch: missing={missing}, extra={extra}"
        )
    return frame.select(
        *(pl.col(name).cast(dtype).alias(name) for name, dtype in schema.items())
    )


def _empty_terminal_schema() -> Mapping[str, pl.DataType]:
    return {
        "path_id": pl.String,
        "physical_path_id": pl.String,
        "physical_entry_id": pl.String,
        "entry_raw_order_fact_id": pl.String,
        "physical_exit_raw_order_fact_id": pl.String,
        "canonical_exit_raw_identity_validated": pl.Boolean,
        "exit_rule_row_is_independent_physical_observation": pl.Boolean,
        "exit_rule_rows_safe_to_sum_as_independent": pl.Boolean,
        "entry_policy_generation_id": pl.String,
        "exit_policy_trial_id": pl.String,
        "Date": pl.String,
        "label_end_date": pl.String,
        "ValueCode": pl.String,
        "QuoteCode": pl.String,
        "route": pl.String,
        "entry_route": pl.String,
        "lookup_action_id": pl.String,
        "entry_lookup_action_id": pl.String,
        "parameter_version": pl.String,
        "entry_parameter_version": pl.String,
        "exit_rule_id": pl.String,
        "exit_rule_source_asof_date": pl.String,
        "exit_threshold_basis_bp": pl.Float64,
        "exit_style": pl.String,
        "exit_route": pl.String,
        "state_family": pl.String,
        "state_bucket": pl.String,
        "lifecycle_policy_version": pl.String,
        "exit_lifecycle_policy_version": pl.String,
        "queue_scenario": pl.String,
        "exit_queue_scenario": pl.String,
        "intended_maker_quantity": pl.Int64,
        "hedge_quantity": pl.Int64,
        "exit_maker_quantity": pl.Int64,
        "exit_hedge_quantity": pl.Int64,
        "outcome_status": pl.String,
        "terminal_branch": pl.String,
        "cancel_status": pl.String,
        "entry_fill_status": pl.String,
        "topup_status": pl.String,
        "hedge_50ms_status": pl.String,
        "same_day_exit_status": pl.String,
        "overnight_status": pl.String,
        "expiry_status": pl.String,
        "emergency_status": pl.String,
        "filled_cashflow_before_cost_bp": pl.Float64,
        "hedge_slippage_bp_50ms": pl.Float64,
        "exit_slippage_bp": pl.Float64,
        "capital_time_seconds": pl.Float64,
        "cost_profile_version": pl.String,
        **{name: pl.Float64 for name in DEFAULT_COST_COLUMNS},
        "normalization_notional_twd": pl.Float64,
        "source_exit_branch_status": pl.String,
        "terminal_mapping_reason": pl.String,
        "terminal_mapping_version": pl.String,
        "exit_cancel_ack_observed": pl.Boolean,
        "exit_cancel_race_modeled": pl.Boolean,
        "exit_active_sibling_cancel_count": pl.Int64,
        "exit_prior_unacked_cancel_count_before_winner": pl.Int64,
        "cost_components_complete": pl.Boolean,
        "strict_ev_ready": pl.Boolean,
        "pathwise_ev_ready": pl.Boolean,
        "joint_volume_allocated": pl.Boolean,
        "contains_target_day_outcome": pl.Boolean,
    }


def _empty_baseline_schema() -> Mapping[str, pl.DataType]:
    return {
        "taker_taker_baseline_ref_id": pl.String,
        "Date": pl.String,
        "ValueCode": pl.String,
        "QuoteCode": pl.String,
        "entry_route": pl.String,
        "entry_policy_generation_id": pl.String,
        "entry_raw_order_fact_id": pl.String,
        "entry_lookup_action_id": pl.String,
        "entry_parameter_version": pl.String,
        "exit_rule_id": pl.String,
        "exit_threshold_basis_bp": pl.Float64,
        "exit_rule_source_asof_date": pl.String,
        "baseline_exit_style": pl.String,
        "baseline_branch_status": pl.String,
        "baseline_outcome_status": pl.String,
        "baseline_terminal_branch": pl.String,
        "baseline_exit_decision_time_ns": pl.Int64,
        "baseline_gross_cycle_pnl_twd": pl.Float64,
        "baseline_filled_cashflow_before_cost_bp": pl.Float64,
        "normalization_notional_twd": pl.Float64,
        "baseline_reference_is_unique": pl.Boolean,
        "baseline_is_route_independent": pl.Boolean,
        "baseline_costs_included": pl.Boolean,
        "baseline_pathwise_ev_ready": pl.Boolean,
        "physical_entry_dependency_id": pl.String,
        "baseline_exit_policy_signature_id": pl.String,
        "physical_taker_taker_baseline_id": pl.String,
        "physical_entry_alias_count": pl.UInt32,
        "physical_entry_alias_weight": pl.Float64,
        "baseline_shared_across_entry_aliases": pl.Boolean,
        "entry_alias_is_independent_observation": pl.Boolean,
        "entry_q_alias_rows_safe_to_sum_as_independent": pl.Boolean,
        "exit_rule_is_alternative_policy": pl.Boolean,
        "exit_policy_signature_rows_safe_to_sum_as_independent": pl.Boolean,
        "baseline_exit_rule_rows_safe_to_sum_as_independent": pl.Boolean,
    }


def _empty_pair_schema() -> Mapping[str, pl.DataType]:
    return {
        "path_id": pl.String,
        "physical_path_id": pl.String,
        "Date": pl.String,
        "ValueCode": pl.String,
        "QuoteCode": pl.String,
        "entry_route": pl.String,
        "entry_policy_generation_id": pl.String,
        "entry_raw_order_fact_id": pl.String,
        "entry_lookup_action_id": pl.String,
        "exit_rule_id": pl.String,
        "exit_route": pl.String,
        "maker_exit_threshold_basis_bp": pl.Float64,
        "maker_exit_rule_source_asof_date": pl.String,
        "maker_outcome_status": pl.String,
        "maker_terminal_branch": pl.String,
        "maker_gross_cycle_bp": pl.Float64,
        "physical_exit_raw_order_fact_id": pl.String,
        "exit_rule_rows_safe_to_sum_as_independent": pl.Boolean,
        "taker_taker_baseline_ref_id": pl.String,
        "_baseline_Date": pl.String,
        "_baseline_ValueCode": pl.String,
        "_baseline_QuoteCode": pl.String,
        "_baseline_entry_route": pl.String,
        "_baseline_entry_raw_order_fact_id": pl.String,
        "_baseline_entry_lookup_action_id": pl.String,
        "entry_parameter_version": pl.String,
        "exit_threshold_basis_bp": pl.Float64,
        "exit_rule_source_asof_date": pl.String,
        "baseline_exit_style": pl.String,
        "baseline_branch_status": pl.String,
        "baseline_outcome_status": pl.String,
        "baseline_terminal_branch": pl.String,
        "baseline_exit_decision_time_ns": pl.Int64,
        "baseline_gross_cycle_pnl_twd": pl.Float64,
        "baseline_filled_cashflow_before_cost_bp": pl.Float64,
        "normalization_notional_twd": pl.Float64,
        "baseline_reference_is_unique": pl.Boolean,
        "baseline_is_route_independent": pl.Boolean,
        "baseline_costs_included": pl.Boolean,
        "baseline_pathwise_ev_ready": pl.Boolean,
        "physical_entry_dependency_id": pl.String,
        "baseline_exit_policy_signature_id": pl.String,
        "physical_taker_taker_baseline_id": pl.String,
        "physical_entry_alias_count": pl.UInt32,
        "physical_entry_alias_weight": pl.Float64,
        "baseline_shared_across_entry_aliases": pl.Boolean,
        "entry_alias_is_independent_observation": pl.Boolean,
        "entry_q_alias_rows_safe_to_sum_as_independent": pl.Boolean,
        "exit_rule_is_alternative_policy": pl.Boolean,
        "exit_policy_signature_rows_safe_to_sum_as_independent": pl.Boolean,
        "baseline_exit_rule_rows_safe_to_sum_as_independent": pl.Boolean,
        "maker_route_matches_for_baseline": pl.UInt32,
        "maker_routes_for_baseline": pl.UInt32,
        "baseline_reference_weight": pl.Float64,
        "physical_baseline_observation_weight": pl.Float64,
        "baseline_shared_across_exit_routes": pl.Boolean,
        "route_copy_is_independent_baseline": pl.Boolean,
        "alternative_policy_comparison": pl.Boolean,
        "both_same_day_known": pl.Boolean,
        "paired_gross_delta_bp": pl.Float64,
    }


def _empty_outcome_matrix_schema() -> Mapping[str, pl.DataType]:
    return {
        "entry_lookup_action_id": pl.String,
        "exit_rule_id": pl.String,
        "exit_route": pl.String,
        "maker_outcome_status": pl.String,
        "maker_terminal_branch": pl.String,
        "baseline_outcome_status": pl.String,
        "baseline_terminal_branch": pl.String,
        "matched_policy_pairs": pl.UInt32,
        "unique_taker_taker_baseline_refs": pl.UInt32,
        "baseline_alias_effective_observations": pl.Float64,
        "baseline_effective_observations": pl.Float64,
        "both_same_day_known_pairs": pl.UInt32,
        "paired_gross_delta_bp_p50": pl.Float64,
        "exit_route_rows_safe_to_sum_as_independent_baselines": pl.Boolean,
    }


def _empty_sensitivity_schema() -> Mapping[str, pl.DataType]:
    return {
        "path_id": pl.String,
        "physical_path_id": pl.String,
        "Date": pl.String,
        "ValueCode": pl.String,
        "entry_route": pl.String,
        "entry_policy_generation_id": pl.String,
        "entry_lookup_action_id": pl.String,
        "exit_rule_id": pl.String,
        "exit_style": pl.String,
        "exit_route": pl.String,
        "outcome_status": pl.String,
        "terminal_branch": pl.String,
        "gross_cycle_bp": pl.Float64,
        "hedge_slippage_bp_50ms": pl.Float64,
        "exit_slippage_bp": pl.Float64,
        "cost_sensitivity_id": pl.String,
        "cost_scope": pl.String,
        "assumed_non_price_cycle_cost_bp": pl.Float64,
        "sensitivity_applicable": pl.Boolean,
        "applied_cycle_cost_count": pl.Int64,
        "conditional_net_after_assumed_cost_bp": pl.Float64,
        "conditional_positive_after_assumed_cost": pl.Boolean,
        "price_slippage_already_in_gross": pl.Boolean,
        "hedge_slippage_subtracted_again": pl.Boolean,
        "exit_slippage_subtracted_again": pl.Boolean,
        "analysis_only": pl.Boolean,
        "production_eligible": pl.Boolean,
        "eligible_for_strict_ev_lookup": pl.Boolean,
        "strict_cost_columns_modified": pl.Boolean,
    }


def _empty_nominal_cancel_sensitivity_schema() -> Mapping[str, pl.DataType]:
    return {
        "nominal_cancel_sensitivity_path_id": pl.String,
        "exit_policy_trial_id": pl.String,
        "physical_entry_id": pl.String,
        "physical_exit_raw_order_fact_id": pl.String,
        "Date": pl.String,
        "ValueCode": pl.String,
        "QuoteCode": pl.String,
        "entry_route": pl.String,
        "entry_policy_generation_id": pl.String,
        "entry_lookup_action_id": pl.String,
        "exit_rule_id": pl.String,
        "exit_route": pl.String,
        "exit_style": pl.String,
        "strict_branch_status": pl.String,
        "strict_terminal_outcome": pl.Boolean,
        "strict_needs_next_session_label": pl.Boolean,
        "strict_projection_safe": pl.Boolean,
        "strict_cancel_ack_observed": pl.Boolean,
        "strict_cancel_race_modeled": pl.Boolean,
        "nominal_instant_cancel_v0_branch": pl.String,
        "modeled_outcome_status": pl.String,
        "modeled_terminal_branch": pl.String,
        "oco_active_sibling_cancel_count": pl.Int64,
        "prior_unacked_cancel_count_before_winner": pl.Int64,
        "any_unacked_cancel_count": pl.Int64,
        "raw_candidate_count": pl.Int64,
        "has_active_sibling_cancel_request": pl.Boolean,
        "has_prior_unacked_cancel_request": pl.Boolean,
        "has_any_unacked_cancel_request": pl.Boolean,
        "cancel_rate_numerator": pl.Int64,
        "cancel_rate_denominator": pl.Int64,
        "strict_cancel_ambiguity_overridden": pl.Boolean,
        "gross_cycle_bp": pl.Float64,
        "normalization_notional_twd": pl.Float64,
        "cost_sensitivity_id": pl.String,
        "cost_scope": pl.String,
        "assumed_non_price_cycle_cost_bp": pl.Float64,
        "applied_cycle_cost_count": pl.Int64,
        "conditional_net_after_assumed_cost_bp": pl.Float64,
        "conditional_positive_after_assumed_cost": pl.Boolean,
        "price_slippage_already_in_gross": pl.Boolean,
        "hedge_slippage_subtracted_again": pl.Boolean,
        "exit_slippage_subtracted_again": pl.Boolean,
        "model_version": pl.String,
        "model_assumption": pl.Boolean,
        "analysis_only": pl.Boolean,
        "production_eligible": pl.Boolean,
        "eligible_for_strict_ev_lookup": pl.Boolean,
        "pathwise_ev_ready": pl.Boolean,
        "exit_rule_rows_safe_to_sum_as_independent": pl.Boolean,
    }
