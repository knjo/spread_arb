"""Daily prequential diagnostics per submitted entry-policy quote.

This adapter widens the denominator of :mod:`exit_policy_lookup` from an
already-established entry position to every submitted entry-policy alias.
It is deliberately a nominal V0 analysis checkpoint, not a production EV:

* an observed no-fill is assigned modeled gross cashflow zero under an
  instantaneous-cancel assumption, while its cancel cost remains missing;
* a fully filled and hedged entry inherits the validated same-day/overnight
  nominal-V0 exit-policy label;
* entry-fill ambiguity, partial fills, hedge failures, and unresolved exits
  remain null/unknown in the denominator; and
* aliases without a frozen exit rule are retained once as an unassigned
  unknown path instead of being dropped or duplicated across invented rules.

Two non-price cost sensitivities are emitted.  The first charges a flat cost
once per completed cycle.  The second can charge different same-day and
overnight costs.  Neither charges a no-fill row; both explicitly omit the
unknown no-fill cancel cost and therefore are never called EV.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import polars as pl

from .exit_policy_lookup import (
    OPTIONAL_CAUSAL_STATE_KEYS,
    OVERNIGHT_TERMINAL_BRANCHES,
    PRIMARY_LOOKUP_KEYS,
    SAME_DAY_TERMINAL_BRANCHES,
    ExitPolicyLookupConfig,
    build_daily_prequential_lookup,
)


EXIT_ROUTES: tuple[str, ...] = (
    "future_bid_spot_taker",
    "spot_ask_future_taker",
)
UNASSIGNED_EXIT_RULE_ID = "__unassigned_exit_policy__"
UNASSIGNED_EXIT_ROUTE = "__unassigned_exit_route__"
NO_FILL_TERMINAL_BRANCH = "entry_no_fill_instant_cancel_v0"
SUBMITTED_SCENARIO = "per_submitted_entry_quote_nominal_v0"

ENTRY_OUTCOME_CATEGORIES = frozenset(
    {
        "known_no_fill_v0",
        "entry_fill_unknown",
        "full_hedged",
        "partial_unknown",
        "full_hedge_unknown",
    }
)


@dataclass(frozen=True)
class SubmittedEntryLookupConfig:
    """Support and explicitly analysis-only cost assumptions."""

    lookback_sessions: int = 60
    min_history_sessions: int = 40
    min_group_sessions: int = 20
    min_known_paths: int = 100
    min_outcome_label_coverage: float = 0.80
    max_censor_rate: float = 0.10
    max_unknown_rate: float = 0.20
    assumed_flat_completed_cycle_cost_bp: float = 19.0
    assumed_same_day_completed_cycle_cost_bp: float = 19.0
    assumed_overnight_completed_cycle_cost_bp: float = 34.0
    estimator_version: str = "submitted_entry_quote_prequential_checkpoint_v1"

    def validate(self) -> None:
        lookup = self.as_conditional_lookup_config()
        lookup.validate()
        for name in (
            "assumed_same_day_completed_cycle_cost_bp",
            "assumed_overnight_completed_cycle_cost_bp",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative")

    def as_conditional_lookup_config(self) -> ExitPolicyLookupConfig:
        return ExitPolicyLookupConfig(
            lookback_sessions=self.lookback_sessions,
            min_history_sessions=self.min_history_sessions,
            min_group_sessions=self.min_group_sessions,
            min_known_paths=self.min_known_paths,
            min_outcome_label_coverage=self.min_outcome_label_coverage,
            max_censor_rate=self.max_censor_rate,
            max_unknown_rate=self.max_unknown_rate,
            assumed_non_price_cycle_cost_bp=(
                self.assumed_flat_completed_cycle_cost_bp
            ),
            estimator_version=self.estimator_version,
        )


def build_submitted_entry_policy_labels(
    action_facts: pl.DataFrame,
    taker_exit_facts: pl.DataFrame,
    conditional_nominal_v0_labels: pl.DataFrame,
    config: SubmittedEntryLookupConfig = SubmittedEntryLookupConfig(),
) -> pl.DataFrame:
    """Build the natural entry-alias x available-exit-policy denominator."""

    config.validate()
    _validate_inputs(action_facts, taker_exit_facts, conditional_nominal_v0_labels)
    if action_facts.is_empty():
        return pl.DataFrame()

    actions = action_facts.select(
        "Date",
        "ValueCode",
        "QuoteCode",
        pl.col("route").alias("entry_route"),
        pl.col("policy_generation_id").alias("entry_policy_generation_id"),
        pl.col("raw_order_fact_id").alias("physical_entry_id"),
        pl.col("lookup_action_id").alias("entry_q"),
        "boundary_quantile",
        "source_asof_date",
        pl.col("parameter_version").alias("entry_parameter_version_dimension"),
        *OPTIONAL_CAUSAL_STATE_KEYS,
        "entry_execution_outcome",
        "queue_known",
        "any_fill",
        "full_fill",
        "partial_fill",
        "entry_hedge_status",
    )
    bad_entry_lineage = actions.filter(
        pl.col("source_asof_date").is_null()
        | (pl.col("source_asof_date") >= pl.col("Date"))
    )
    if bad_entry_lineage.height:
        raise ValueError("entry action source_asof_date must strictly precede Date")

    rules = taker_exit_facts.select(
        "policy_generation_id",
        pl.col("Date").alias("_exit_rule_Date"),
        "exit_rule_id",
        "exit_rule_source_asof_date",
        "exit_threshold_basis_bp",
    ).unique()
    if rules.select(["policy_generation_id", "exit_rule_id"]).n_unique() != rules.height:
        raise ValueError("exit-rule dimensions disagree within entry policy")
    bad_exit_lineage = rules.filter(
        pl.col("exit_rule_source_asof_date").is_null()
        | (pl.col("exit_rule_source_asof_date") >= pl.col("_exit_rule_Date"))
    )
    if bad_exit_lineage.height:
        raise ValueError("exit rule source_asof_date must strictly precede Date")

    assigned = actions.join(
        rules,
        left_on="entry_policy_generation_id",
        right_on="policy_generation_id",
        how="inner",
        validate="1:m",
    ).join(pl.DataFrame({"exit_route": list(EXIT_ROUTES)}), how="cross")
    if assigned.filter(pl.col("Date") != pl.col("_exit_rule_Date")).height:
        raise ValueError("entry action and exit rule dates disagree")
    assigned = assigned.drop("_exit_rule_Date")
    assigned = assigned.with_columns(pl.lit(True).alias("exit_policy_assigned"))

    rule_policy_ids = rules.select(
        pl.col("policy_generation_id").alias("entry_policy_generation_id")
    ).unique()
    unassigned = actions.join(
        rule_policy_ids,
        on="entry_policy_generation_id",
        how="anti",
    ).with_columns(
        pl.lit(UNASSIGNED_EXIT_RULE_ID).alias("exit_rule_id"),
        pl.lit(UNASSIGNED_EXIT_ROUTE).alias("exit_route"),
        pl.lit(None, dtype=pl.String).alias("exit_rule_source_asof_date"),
        pl.lit(None, dtype=pl.Float64).alias("exit_threshold_basis_bp"),
        pl.lit(False).alias("exit_policy_assigned"),
    )
    paths = pl.concat([assigned, unassigned], how="diagonal_relaxed")
    paths = paths.with_columns(
        pl.when(pl.col("entry_execution_outcome") == "no_fill_then_cancel")
        .then(pl.lit("known_no_fill_v0"))
        .when(
            pl.col("entry_execution_outcome") == "full_fill_hedge_executable"
        )
        .then(pl.lit("full_hedged"))
        .when(pl.col("partial_fill").fill_null(False))
        .then(pl.lit("partial_unknown"))
        .when(pl.col("full_fill").fill_null(False))
        .then(pl.lit("full_hedge_unknown"))
        .otherwise(pl.lit("entry_fill_unknown"))
        .alias("entry_outcome_category"),
        pl.concat_str(
            [
                pl.lit("submitted-v0/"),
                pl.col("entry_policy_generation_id"),
                pl.lit("/"),
                pl.col("exit_rule_id"),
                pl.lit("/"),
                pl.col("exit_route"),
            ]
        ).alias("exit_policy_trial_id"),
    )

    conditional = conditional_nominal_v0_labels.select(
        "entry_policy_generation_id",
        "exit_rule_id",
        "exit_route",
        pl.col("exit_policy_trial_id").alias("conditional_exit_policy_trial_id"),
        pl.col("Date").alias("_conditional_Date"),
        pl.col("ValueCode").alias("_conditional_ValueCode"),
        pl.col("QuoteCode").alias("_conditional_QuoteCode"),
        pl.col("entry_route").alias("_conditional_entry_route"),
        pl.col("label_end_date").alias("_conditional_label_end_date"),
        pl.col("outcome_status").alias("_conditional_outcome_status"),
        pl.col("terminal_branch").alias("_conditional_terminal_branch"),
        pl.col("actual_four_leg_gross_bp").alias("_conditional_gross_bp"),
        pl.col("source_exit_branch_status").alias(
            "_conditional_source_exit_branch_status"
        ),
        pl.col("strict_cancel_ambiguity_overridden").alias(
            "_conditional_cancel_ambiguity_overridden"
        ),
        pl.col("any_unacked_cancel_count").alias(
            "_conditional_unacked_cancel_count"
        ),
        pl.col("overnight_label_attached").alias(
            "_conditional_overnight_label_attached"
        ),
        pl.col("terminal_label_source").alias("_conditional_terminal_label_source"),
        pl.col("unresolved_reason").alias("_conditional_unresolved_reason"),
    )
    joined = paths.join(
        conditional,
        on=["entry_policy_generation_id", "exit_rule_id", "exit_route"],
        how="left",
        validate="m:1",
    )
    full_hedged = pl.col("entry_outcome_category") == "full_hedged"
    missing_conditional = joined.filter(
        full_hedged & pl.col("conditional_exit_policy_trial_id").is_null()
    )
    if missing_conditional.height:
        raise ValueError("full-hedged submitted path lacks conditional exit label")
    unexpected_conditional = joined.filter(
        pl.col("conditional_exit_policy_trial_id").is_not_null()
        & ~full_hedged
    )
    if unexpected_conditional.height:
        raise ValueError("conditional exit label attached to non-established entry")
    identity_mismatch = joined.filter(
        full_hedged
        & (
            (pl.col("Date") != pl.col("_conditional_Date"))
            | (pl.col("ValueCode") != pl.col("_conditional_ValueCode"))
            | (pl.col("QuoteCode") != pl.col("_conditional_QuoteCode"))
            | (pl.col("entry_route") != pl.col("_conditional_entry_route"))
        )
    )
    if identity_mismatch.height:
        raise ValueError("submitted and conditional exit identities disagree")

    no_fill = pl.col("entry_outcome_category") == "known_no_fill_v0"
    unresolved_entry = ~no_fill & ~full_hedged
    labeled = joined.with_columns(
        pl.concat_str([pl.lit("submitted-v0-label/"), pl.col("exit_policy_trial_id")])
        .alias("policy_label_id"),
        pl.when(no_fill)
        .then(pl.lit("known"))
        .when(full_hedged)
        .then(pl.col("_conditional_outcome_status"))
        .otherwise(pl.lit("unknown"))
        .alias("outcome_status"),
        pl.when(no_fill)
        .then(pl.lit(NO_FILL_TERMINAL_BRANCH))
        .when(full_hedged)
        .then(pl.col("_conditional_terminal_branch"))
        .otherwise(None)
        .cast(pl.String)
        .alias("terminal_branch"),
        pl.when(no_fill)
        .then(pl.lit(0.0))
        .when(full_hedged)
        .then(pl.col("_conditional_gross_bp"))
        .otherwise(None)
        .cast(pl.Float64)
        .alias("submitted_quote_gross_cashflow_bp"),
        pl.when(full_hedged)
        .then(pl.col("_conditional_label_end_date"))
        .otherwise(pl.col("Date"))
        .alias("label_end_date"),
        pl.when(no_fill)
        .then(pl.lit("entry_no_fill_then_nominal_instant_cancel_v0"))
        .when(full_hedged)
        .then(pl.col("_conditional_source_exit_branch_status"))
        .otherwise(pl.concat_str([pl.lit("entry_"), pl.col("entry_outcome_category")]))
        .alias("source_exit_branch_status"),
        pl.when(full_hedged)
        .then(pl.col("_conditional_cancel_ambiguity_overridden"))
        .otherwise(False)
        .alias("strict_cancel_ambiguity_overridden"),
        pl.when(full_hedged)
        .then(pl.col("_conditional_unacked_cancel_count"))
        .otherwise(0)
        .cast(pl.Int64)
        .alias("any_unacked_cancel_count"),
        pl.when(full_hedged)
        .then(pl.col("_conditional_overnight_label_attached"))
        .otherwise(False)
        .alias("overnight_label_attached"),
        pl.when(no_fill)
        .then(pl.lit("entry_execution_no_fill_v0"))
        .when(full_hedged)
        .then(pl.col("_conditional_terminal_label_source"))
        .otherwise(pl.lit("entry_execution_unresolved"))
        .alias("terminal_label_source"),
        pl.when(unresolved_entry)
        .then(pl.col("entry_outcome_category"))
        .when(full_hedged)
        .then(pl.col("_conditional_unresolved_reason"))
        .otherwise(None)
        .alias("unresolved_reason"),
    )
    known = pl.col("outcome_status") == "known"
    completed = known & pl.col("terminal_branch").is_in(
        [*SAME_DAY_TERMINAL_BRANCHES, *OVERNIGHT_TERMINAL_BRANCHES]
    )
    same_day = known & pl.col("terminal_branch").is_in(
        list(SAME_DAY_TERMINAL_BRANCHES)
    )
    overnight = known & pl.col("terminal_branch").is_in(
        list(OVERNIGHT_TERMINAL_BRANCHES)
    )
    labeled = labeled.with_columns(
        pl.col("submitted_quote_gross_cashflow_bp").alias(
            "actual_four_leg_gross_bp"
        ),
        pl.when(no_fill)
        .then(0)
        .when(completed)
        .then(1)
        .otherwise(None)
        .cast(pl.Int64)
        .alias("completed_cycle_cost_applied_count"),
        pl.when(no_fill)
        .then(pl.lit(0.0))
        .when(completed)
        .then(
            pl.col("submitted_quote_gross_cashflow_bp")
            - config.assumed_flat_completed_cycle_cost_bp
        )
        .otherwise(None)
        .alias("cashflow_after_flat_completed_cycle_cost_sensitivity_bp"),
        pl.when(no_fill)
        .then(pl.lit(0.0))
        .when(same_day)
        .then(
            pl.col("submitted_quote_gross_cashflow_bp")
            - config.assumed_same_day_completed_cycle_cost_bp
        )
        .when(overnight)
        .then(
            pl.col("submitted_quote_gross_cashflow_bp")
            - config.assumed_overnight_completed_cycle_cost_bp
        )
        .otherwise(None)
        .alias("cashflow_after_branch_completed_cycle_cost_sensitivity_bp"),
        no_fill.alias("entry_no_fill_v0_model_assumption"),
        no_fill.alias("no_fill_cancel_cost_missing"),
        pl.lit(False).alias("complete_production_cost_profile"),
        pl.lit(config.assumed_flat_completed_cycle_cost_bp).alias(
            "assumed_non_price_cycle_cost_bp"
        ),
        pl.lit(config.assumed_same_day_completed_cycle_cost_bp).alias(
            "assumed_same_day_completed_cycle_cost_bp"
        ),
        pl.lit(config.assumed_overnight_completed_cycle_cost_bp).alias(
            "assumed_overnight_completed_cycle_cost_bp"
        ),
        pl.lit(SUBMITTED_SCENARIO).alias("scenario"),
        pl.lit(True).alias("terminal_model_assumption"),
        pl.lit(False).alias("ev_ready"),
        pl.lit(False).alias("conditional_on_established_entry"),
        pl.lit(True).alias("entry_fill_probability_included"),
        pl.lit(False).alias("policy_rows_safe_to_sum_across_q_rule_route"),
        pl.lit(True).alias("price_slippage_already_in_gross"),
        pl.lit(True).alias("analysis_only"),
        pl.lit(False).alias("production_eligible"),
    )
    _validate_submitted_labels(labeled)
    return labeled.drop(
        column for column in labeled.columns if column.startswith("_conditional_")
    ).sort(["Date", *PRIMARY_LOOKUP_KEYS[1:], "exit_policy_trial_id"])


def build_daily_submitted_entry_lookup(
    submitted_labels: pl.DataFrame,
    sessions: Sequence[str],
    config: SubmittedEntryLookupConfig = SubmittedEntryLookupConfig(),
    *,
    include_causal_state: bool = False,
    asof_dates: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Build D-safe per-submitted-quote lookup diagnostics."""

    config.validate()
    _validate_submitted_labels(submitted_labels)
    if submitted_labels.is_empty():
        return pl.DataFrame()
    compatibility = submitted_labels.with_columns(
        pl.col("cashflow_after_flat_completed_cycle_cost_sensitivity_bp").alias(
            "conditional_net_after_assumed_cost_bp"
        )
    )
    base = build_daily_prequential_lookup(
        compatibility,
        sessions,
        config.as_conditional_lookup_config(),
        include_causal_state=include_causal_state,
        asof_dates=asof_dates,
    )
    if base.is_empty():
        return base
    base = base.rename(
        {
            "conditional_known_net_after_assumed_cost_mean_bp": (
                "conditional_known_cashflow_after_flat_completed_cycle_cost_mean_bp"
            ),
            "conditional_known_net_after_assumed_cost_p50_bp": (
                "conditional_known_cashflow_after_flat_completed_cycle_cost_p50_bp"
            ),
            "known_net_after_assumed_cost_zero_for_unpriced_sensitivity_bp": (
                "known_cashflow_after_flat_completed_cycle_cost_zero_for_unpriced_sensitivity_bp"
            ),
            "identified_net_after_assumed_cost_sensitivity_mean_bp": (
                "identified_cashflow_after_flat_completed_cycle_cost_sensitivity_mean_bp"
            ),
            "assumed_non_price_cycle_cost_bp": (
                "assumed_flat_completed_cycle_cost_bp"
            ),
            "flat_cost_scope": "flat_completed_cycle_cost_scope",
        }
    )
    calendar = [str(value) for value in sessions]
    selected = [str(value) for value in (asof_dates or calendar)]
    keys = [*PRIMARY_LOOKUP_KEYS]
    if include_causal_state:
        keys.extend(OPTIONAL_CAUSAL_STATE_KEYS)
    supplements: list[pl.DataFrame] = []
    index_by_date = {date: index for index, date in enumerate(calendar)}
    for asof_date in selected:
        index = index_by_date[asof_date]
        prior = calendar[max(0, index - config.lookback_sessions) : index]
        if not prior:
            continue
        training = submitted_labels.filter(pl.col("Date").is_in(prior))
        if training.is_empty():
            continue
        matured = pl.col("label_end_date").is_not_null() & (
            pl.col("label_end_date") < asof_date
        )
        known = matured & (pl.col("outcome_status") == "known")
        branch_priced = known & pl.col(
            "cashflow_after_branch_completed_cycle_cost_sensitivity_bp"
        ).is_finite()
        supplement = training.group_by(keys).agg(
            pl.col("entry_policy_generation_id")
            .n_unique()
            .cast(pl.Int64)
            .alias("n_entry_policy_aliases"),
            pl.col("physical_entry_id")
            .n_unique()
            .cast(pl.Int64)
            .alias("n_submitted_raw_orders"),
            pl.col("exit_policy_assigned")
            .sum()
            .cast(pl.Int64)
            .alias("n_paths_with_assigned_exit_policy"),
            (~pl.col("exit_policy_assigned"))
            .sum()
            .cast(pl.Int64)
            .alias("n_unassigned_exit_policy_paths"),
            (pl.col("entry_outcome_category") == "known_no_fill_v0")
            .sum()
            .cast(pl.Int64)
            .alias("n_entry_no_fill_v0_paths"),
            (pl.col("entry_outcome_category") == "entry_fill_unknown")
            .sum()
            .cast(pl.Int64)
            .alias("n_entry_fill_unknown_paths"),
            (pl.col("entry_outcome_category") == "full_hedged")
            .sum()
            .cast(pl.Int64)
            .alias("n_full_hedged_entry_paths"),
            (pl.col("entry_outcome_category") == "partial_unknown")
            .sum()
            .cast(pl.Int64)
            .alias("n_partial_entry_unknown_paths"),
            (pl.col("entry_outcome_category") == "full_hedge_unknown")
            .sum()
            .cast(pl.Int64)
            .alias("n_full_entry_hedge_unknown_paths"),
            (
                matured
                & pl.col("no_fill_cancel_cost_missing")
                & (pl.col("outcome_status") == "known")
            )
            .sum()
            .cast(pl.Int64)
            .alias("n_known_no_fill_paths_with_cancel_cost_missing"),
            (
                matured
                & (pl.col("completed_cycle_cost_applied_count") == 1)
            )
            .sum()
            .cast(pl.Int64)
            .alias("n_completed_cycles_costed_once"),
            pl.col("cashflow_after_branch_completed_cycle_cost_sensitivity_bp")
            .filter(branch_priced)
            .sum()
            .alias("_known_branch_cost_cashflow_sum_bp"),
            pl.col("cashflow_after_branch_completed_cycle_cost_sensitivity_bp")
            .filter(branch_priced)
            .mean()
            .alias(
                "conditional_known_cashflow_after_branch_completed_cycle_cost_mean_bp"
            ),
            pl.col("cashflow_after_branch_completed_cycle_cost_sensitivity_bp")
            .filter(branch_priced)
            .median()
            .alias(
                "conditional_known_cashflow_after_branch_completed_cycle_cost_p50_bp"
            ),
        ).with_columns(pl.lit(asof_date).alias("asof_date"))
        supplements.append(supplement)
    extra = pl.concat(supplements, how="diagonal_relaxed", rechunk=True)
    result = base.join(extra, on=["asof_date", *keys], how="left", validate="1:1")
    result = result.with_columns(
        (
            pl.col("_known_branch_cost_cashflow_sum_bp")
            / pl.col("n_policy_paths")
        ).alias(
            "known_cashflow_after_branch_completed_cycle_cost_zero_for_unpriced_sensitivity_bp"
        ),
        pl.when(pl.col("terminal_cashflow_point_identified"))
        .then(
            pl.col(
                "conditional_known_cashflow_after_branch_completed_cycle_cost_mean_bp"
            )
        )
        .otherwise(None)
        .alias(
            "identified_cashflow_after_branch_completed_cycle_cost_sensitivity_mean_bp"
        ),
        pl.lit(config.assumed_same_day_completed_cycle_cost_bp).alias(
            "assumed_same_day_completed_cycle_cost_bp"
        ),
        pl.lit(config.assumed_overnight_completed_cycle_cost_bp).alias(
            "assumed_overnight_completed_cycle_cost_bp"
        ),
        pl.lit("same_day_or_overnight_completed_cycle_once").alias(
            "branch_completed_cycle_cost_scope"
        ),
        pl.lit(False).alias("conditional_on_established_entry"),
        pl.lit(True).alias("entry_fill_probability_included"),
        pl.lit("entry_policy_alias_x_available_exit_policy_path").alias(
            "sampling_unit"
        ),
        pl.lit(True).alias("unassigned_exit_policy_retained_once"),
        pl.lit(True).alias("no_fill_gross_zero_is_model_assumption"),
        pl.lit(False).alias("no_fill_cancel_cost_included"),
        pl.lit(True).alias("branch_cost_sensitivity_analysis_only"),
        pl.lit(False).alias("complete_production_cost_profile"),
        pl.lit(False).alias("ev_ready"),
        pl.lit(None, dtype=pl.Float64).alias("expected_net_cashflow_bp"),
    ).drop("_known_branch_cost_cashflow_sum_bp")
    _validate_submitted_lookup(result)
    return result.sort(["asof_date", *keys])


def submitted_label_category_counts(labels: pl.DataFrame) -> pl.DataFrame:
    """Return an auditable whole-population category/assignment cross-tab."""

    _validate_submitted_labels(labels)
    return labels.group_by(
        ["entry_outcome_category", "exit_policy_assigned"]
    ).agg(
        pl.len().cast(pl.Int64).alias("policy_paths"),
        pl.col("entry_policy_generation_id")
        .n_unique()
        .cast(pl.Int64)
        .alias("entry_policy_aliases"),
        pl.col("physical_entry_id")
        .n_unique()
        .cast(pl.Int64)
        .alias("physical_entry_orders"),
    ).sort(["entry_outcome_category", "exit_policy_assigned"])


def _validate_inputs(
    action_facts: pl.DataFrame,
    taker_exit_facts: pl.DataFrame,
    conditional_labels: pl.DataFrame,
) -> None:
    _require(
        action_facts,
        {
            "Date",
            "ValueCode",
            "QuoteCode",
            "route",
            "policy_generation_id",
            "raw_order_fact_id",
            "lookup_action_id",
            "boundary_quantile",
            "source_asof_date",
            "parameter_version",
            *OPTIONAL_CAUSAL_STATE_KEYS,
            "entry_execution_outcome",
            "queue_known",
            "any_fill",
            "full_fill",
            "partial_fill",
            "entry_hedge_status",
        },
        "entry action facts",
    )
    _require(
        taker_exit_facts,
        {
            "Date",
            "policy_generation_id",
            "exit_rule_id",
            "exit_rule_source_asof_date",
            "exit_threshold_basis_bp",
        },
        "taker exit facts",
    )
    _require(
        conditional_labels,
        {
            "entry_policy_generation_id",
            "exit_rule_id",
            "exit_route",
            "exit_policy_trial_id",
            "Date",
            "ValueCode",
            "QuoteCode",
            "entry_route",
            "label_end_date",
            "outcome_status",
            "terminal_branch",
            "actual_four_leg_gross_bp",
            "source_exit_branch_status",
            "strict_cancel_ambiguity_overridden",
            "any_unacked_cancel_count",
            "overnight_label_attached",
            "terminal_label_source",
            "unresolved_reason",
        },
        "conditional nominal V0 labels",
    )
    if action_facts.select("policy_generation_id").n_unique() != action_facts.height:
        raise ValueError("entry action facts duplicate policy_generation_id")
    conditional_key = [
        "entry_policy_generation_id",
        "exit_rule_id",
        "exit_route",
    ]
    if conditional_labels.select(conditional_key).n_unique() != conditional_labels.height:
        raise ValueError("conditional labels duplicate exact entry/exit policy key")


def _validate_submitted_labels(labels: pl.DataFrame) -> None:
    if labels.is_empty():
        return
    _require(
        labels,
        {
            "policy_label_id",
            "exit_policy_trial_id",
            "physical_entry_id",
            "entry_policy_generation_id",
            "Date",
            "label_end_date",
            *PRIMARY_LOOKUP_KEYS,
            *OPTIONAL_CAUSAL_STATE_KEYS,
            "entry_outcome_category",
            "exit_policy_assigned",
            "outcome_status",
            "terminal_branch",
            "actual_four_leg_gross_bp",
            "cashflow_after_flat_completed_cycle_cost_sensitivity_bp",
            "cashflow_after_branch_completed_cycle_cost_sensitivity_bp",
            "completed_cycle_cost_applied_count",
            "entry_no_fill_v0_model_assumption",
            "no_fill_cancel_cost_missing",
            "assumed_non_price_cycle_cost_bp",
            "scenario",
            "terminal_model_assumption",
            "source_exit_branch_status",
            "strict_cancel_ambiguity_overridden",
            "any_unacked_cancel_count",
        },
        "submitted entry labels",
    )
    if labels.select("policy_label_id").n_unique() != labels.height:
        raise ValueError("submitted labels duplicate policy_label_id")
    if labels.select("exit_policy_trial_id").n_unique() != labels.height:
        raise ValueError("submitted labels duplicate exit_policy_trial_id")
    invalid_category = labels.filter(
        ~pl.col("entry_outcome_category")
        .is_in(list(ENTRY_OUTCOME_CATEGORIES))
        .fill_null(False)
    )
    if invalid_category.height:
        raise ValueError("submitted labels contain invalid entry outcome category")
    malformed_unresolved = labels.filter(
        (pl.col("outcome_status") != "known")
        & (
            pl.col("terminal_branch").is_not_null()
            | pl.col("actual_four_leg_gross_bp").is_not_null()
            | pl.col(
                "cashflow_after_flat_completed_cycle_cost_sensitivity_bp"
            ).is_not_null()
            | pl.col(
                "cashflow_after_branch_completed_cycle_cost_sensitivity_bp"
            ).is_not_null()
        )
    )
    if malformed_unresolved.height:
        raise ValueError("unresolved submitted label claims priced cashflow")
    no_fill = labels.filter(
        pl.col("entry_outcome_category") == "known_no_fill_v0"
    )
    malformed_no_fill = no_fill.filter(
        (pl.col("outcome_status") != "known")
        | (pl.col("actual_four_leg_gross_bp") != 0.0)
        | (pl.col("completed_cycle_cost_applied_count") != 0)
        | ~pl.col("entry_no_fill_v0_model_assumption")
        | ~pl.col("no_fill_cancel_cost_missing")
    )
    if malformed_no_fill.height:
        raise ValueError("no-fill V0 labels violate zero-gross/cancel-cost contract")
    completed = labels.filter(pl.col("completed_cycle_cost_applied_count") == 1)
    if completed.filter(pl.col("entry_outcome_category") != "full_hedged").height:
        raise ValueError("completed cycle is not a full-hedged entry path")
    invalid_dates = labels.filter(
        pl.col("Date").is_null()
        | (
            pl.col("label_end_date").is_not_null()
            & (pl.col("label_end_date") < pl.col("Date"))
        )
        | (
            (pl.col("outcome_status") == "known")
            & pl.col("label_end_date").is_null()
        )
    )
    if invalid_dates.height:
        raise ValueError("submitted labels contain invalid maturity dates")


def _validate_submitted_lookup(lookup: pl.DataFrame) -> None:
    if lookup.is_empty():
        return
    if lookup.filter(pl.col("ev_ready")).height:
        raise ValueError("submitted lookup unexpectedly marks EV ready")
    if lookup.filter(pl.col("expected_net_cashflow_bp").is_not_null()).height:
        raise ValueError("submitted lookup unexpectedly prices production EV")
    if lookup.filter(pl.col("conditional_on_established_entry")).height:
        raise ValueError("submitted lookup reverted to conditional-entry semantics")
    if lookup.filter(~pl.col("entry_fill_probability_included")).height:
        raise ValueError("submitted lookup omits entry fill outcomes")
    if lookup.filter(pl.col("contains_target_day_outcome")).height:
        raise ValueError("submitted lookup contains target-day outcomes")
    if lookup.filter(pl.col("label_cutoff_date") >= pl.col("asof_date")).height:
        raise ValueError("submitted lookup violates D-safe maturity cutoff")


def _require(frame: pl.DataFrame, columns: set[str], source: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")
