"""Causal daily lookup checkpoint for conditional exit-maker policies.

This module is intentionally an analysis checkpoint, not a production EV
estimator.  It adapts the validated exit-maker policy facts into two separate
label populations:

* ``strict`` keeps every unobserved cancel race unknown; and
* ``nominal_instant_cancel_v0`` applies the explicitly named instantaneous
  cancel counterfactual from :mod:`exit_maker_terminal`.

Both populations retain censored and unknown rows in their denominator.  A
daily table for decision day ``D`` keeps every admission in the trailing
window, while only outcomes whose ``label_end_date < D`` are visible.  Future
and null-maturity labels remain pending in the denominator.  Known cashflow
statistics are conditional diagnostics.  A zero-for-unpriced number is
emitted only as a named sensitivity; it is never called EV.  Without explicit
loss/profit caps, unresolved terminal mass has no finite cashflow bound.

The sampling unit is conditional on an already-established entry position.
These tables therefore cannot be interpreted as EV per submitted entry quote.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import polars as pl

from .ev_surface import EVLookupConfig
from .exit_maker_terminal import (
    ExitMakerTerminalConfig,
    build_exit_maker_terminal_paths_v2,
    build_nominal_instant_cancel_v0_sensitivity,
)


PRIMARY_LOOKUP_KEYS: tuple[str, ...] = (
    "ValueCode",
    "entry_route",
    "entry_q",
    "exit_rule_id",
    "exit_route",
)

OPTIONAL_CAUSAL_STATE_KEYS: tuple[str, ...] = (
    "rank_bucket",
    "queue_bucket",
    "tod_bucket",
    "freshness_bucket",
)

OUTCOME_STATUSES = frozenset({"known", "censored", "unknown"})
STRICT_CARRY_BRANCHES = frozenset(
    {
        "carry_at_eod_cancel_unconfirmed",
        "carry_at_eod_no_admission",
    }
)
SAME_DAY_TERMINAL_BRANCHES = frozenset(
    {"same_day_target_exit", "same_day_aggressive_exit"}
)
OVERNIGHT_TERMINAL_BRANCHES = frozenset(
    {"overnight_exit", "expiry_forced_flat", "expiry_settlement"}
)

_BASE_SUPPORT = EVLookupConfig()


@dataclass(frozen=True)
class ExitPolicyLookupConfig:
    """Support and analysis-only cost contract for the checkpoint."""

    lookback_sessions: int = _BASE_SUPPORT.lookback_sessions
    min_history_sessions: int = _BASE_SUPPORT.min_history_sessions
    min_group_sessions: int = _BASE_SUPPORT.min_group_sessions
    min_known_paths: int = _BASE_SUPPORT.min_known_paths
    min_outcome_label_coverage: float = _BASE_SUPPORT.min_outcome_label_coverage
    max_censor_rate: float = _BASE_SUPPORT.max_censor_rate
    max_unknown_rate: float = _BASE_SUPPORT.max_unknown_rate
    assumed_non_price_cycle_cost_bp: float = 19.0
    estimator_version: str = "conditional_exit_policy_prequential_checkpoint_v1"

    def validate(self) -> None:
        if self.lookback_sessions <= 0:
            raise ValueError("lookback_sessions must be positive")
        if not 1 <= self.min_history_sessions <= self.lookback_sessions:
            raise ValueError(
                "min_history_sessions must be in [1, lookback_sessions]"
            )
        if self.min_group_sessions <= 0 or self.min_known_paths <= 0:
            raise ValueError("support thresholds must be positive")
        for name in (
            "min_outcome_label_coverage",
            "max_censor_rate",
            "max_unknown_rate",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        cost = self.assumed_non_price_cycle_cost_bp
        if (
            isinstance(cost, bool)
            or not isinstance(cost, (int, float))
            or not math.isfinite(float(cost))
            or float(cost) < 0.0
        ):
            raise ValueError(
                "assumed_non_price_cycle_cost_bp must be finite and non-negative"
            )
        if not isinstance(self.estimator_version, str) or not self.estimator_version:
            raise ValueError("estimator_version must be non-empty")


@dataclass(frozen=True)
class ExitPolicyLabelTables:
    """Strict and nominal-V0 path labels with identical audit columns."""

    strict: pl.DataFrame
    nominal_v0: pl.DataFrame


def build_exit_policy_label_tables(
    action_facts: pl.DataFrame,
    position_policy_facts: pl.DataFrame,
    *,
    terminal_config: ExitMakerTerminalConfig,
    assumed_non_price_cycle_cost_bp: float = 19.0,
    overnight_labels: pl.DataFrame | None = None,
) -> ExitPolicyLabelTables:
    """Adapt exit-maker facts while keeping strict and V0 semantics separate."""

    cost = _validate_cost(assumed_non_price_cycle_cost_bp)
    dimensions = _entry_dimensions(action_facts)
    strict_raw = build_exit_maker_terminal_paths_v2(
        action_facts,
        position_policy_facts,
        terminal_config,
    )
    nominal_raw = build_nominal_instant_cancel_v0_sensitivity(
        position_policy_facts,
        action_facts,
        cost_bp=cost,
    )
    if strict_raw.is_empty():
        empty = _empty_label_frame()
        return ExitPolicyLabelTables(empty, empty.clone())

    strict = _build_strict_labels(strict_raw, dimensions, cost)
    nominal = _build_nominal_labels(strict, nominal_raw, cost)
    if overnight_labels is not None:
        strict = attach_overnight_labels(strict, overnight_labels, cost)
        nominal = attach_overnight_labels(nominal, overnight_labels, cost)
    _validate_policy_labels(strict, "strict policy labels")
    _validate_policy_labels(nominal, "nominal V0 policy labels")
    return ExitPolicyLabelTables(strict=strict, nominal_v0=nominal)


def attach_overnight_labels(
    policy_labels: pl.DataFrame,
    overnight_labels: pl.DataFrame,
    assumed_non_price_cycle_cost_bp: float = 19.0,
) -> pl.DataFrame:
    """Replace strict-carry censor rows only when an exact overnight label exists.

    A partial overnight root is allowed: unmatched carry rows remain censored.
    Identity disagreements fail closed.  The caller is responsible for loading
    marker/hash-validated overnight partitions.
    """

    cost = _validate_cost(assumed_non_price_cycle_cost_bp)
    if policy_labels.is_empty() or overnight_labels.is_empty():
        return policy_labels
    _require(
        overnight_labels,
        {
            "exit_policy_trial_id",
            "Date",
            "ValueCode",
            "QuoteCode",
            "entry_route",
            "exit_rule_id",
            "exit_route",
            "source_branch_status",
            "outcome_status",
            "terminal_branch",
            "label_end_date",
            "filled_cashflow_before_cost_bp",
            "label_status",
            "unresolved_reason",
        },
        "overnight labels",
    )
    if (
        overnight_labels.select("exit_policy_trial_id").n_unique()
        != overnight_labels.height
    ):
        raise ValueError("overnight labels duplicate exit_policy_trial_id")
    invalid = overnight_labels.filter(
        ~pl.col("outcome_status").is_in(OUTCOME_STATUSES).fill_null(False)
    )
    if invalid.height:
        raise ValueError("overnight labels contain invalid outcome_status")

    overnight = overnight_labels.select(
        "exit_policy_trial_id",
        pl.col("Date").alias("_overnight_Date"),
        pl.col("ValueCode").alias("_overnight_ValueCode"),
        pl.col("QuoteCode").alias("_overnight_QuoteCode"),
        pl.col("entry_route").alias("_overnight_entry_route"),
        pl.col("exit_rule_id").alias("_overnight_exit_rule_id"),
        pl.col("exit_route").alias("_overnight_exit_route"),
        pl.col("source_branch_status").alias("_overnight_source_branch_status"),
        pl.col("outcome_status").alias("_overnight_outcome_status"),
        pl.col("terminal_branch")
        .cast(pl.String)
        .alias("_overnight_terminal_branch"),
        pl.col("label_end_date").alias("_overnight_label_end_date"),
        pl.col("filled_cashflow_before_cost_bp")
        .cast(pl.Float64)
        .alias("_overnight_gross_bp"),
        pl.col("label_status").alias("_overnight_label_status"),
        pl.col("unresolved_reason").alias("_overnight_unresolved_reason"),
    )
    joined = policy_labels.join(
        overnight,
        on="exit_policy_trial_id",
        how="left",
        validate="1:1",
    )
    found = pl.col("_overnight_Date").is_not_null()
    expected_carry = pl.col("source_exit_branch_status").is_in(
        list(STRICT_CARRY_BRANCHES)
    )
    unexpected = joined.filter(found & ~expected_carry)
    if unexpected.height:
        raise ValueError("overnight label attached to a non-carry exit policy")
    identity_mismatch = joined.filter(
        found
        & (
            (pl.col("Date") != pl.col("_overnight_Date"))
            | (pl.col("ValueCode") != pl.col("_overnight_ValueCode"))
            | (pl.col("QuoteCode") != pl.col("_overnight_QuoteCode"))
            | (pl.col("entry_route") != pl.col("_overnight_entry_route"))
            | (pl.col("exit_rule_id") != pl.col("_overnight_exit_rule_id"))
            | (pl.col("exit_route") != pl.col("_overnight_exit_route"))
            | (
                pl.col("source_exit_branch_status")
                != pl.col("_overnight_source_branch_status")
            )
        )
    )
    if identity_mismatch.height:
        raise ValueError("overnight and exit policy identities disagree")
    malformed_known = joined.filter(
        found
        & (pl.col("_overnight_outcome_status") == "known")
        & (
            pl.col("_overnight_terminal_branch").is_null()
            | pl.col("_overnight_gross_bp").is_null()
            | ~pl.col("_overnight_gross_bp").is_finite()
        )
    )
    if malformed_known.height:
        raise ValueError("known overnight label is missing terminal cashflow")

    result = joined.with_columns(
        found.alias("overnight_label_attached"),
        pl.when(found)
        .then(pl.col("_overnight_outcome_status"))
        .otherwise(pl.col("outcome_status"))
        .alias("outcome_status"),
        pl.when(found)
        .then(pl.col("_overnight_terminal_branch"))
        .otherwise(pl.col("terminal_branch"))
        .cast(pl.String)
        .alias("terminal_branch"),
        pl.when(found)
        .then(pl.col("_overnight_label_end_date"))
        .otherwise(pl.col("label_end_date"))
        .alias("label_end_date"),
        pl.when(found)
        .then(pl.col("_overnight_gross_bp"))
        .otherwise(pl.col("actual_four_leg_gross_bp"))
        .alias("actual_four_leg_gross_bp"),
        pl.when(found)
        .then(pl.col("_overnight_label_status"))
        .otherwise(pl.col("terminal_label_source"))
        .alias("terminal_label_source"),
        pl.when(found)
        .then(pl.col("_overnight_unresolved_reason"))
        .otherwise(pl.col("unresolved_reason"))
        .alias("unresolved_reason"),
    ).with_columns(
        pl.when(
            (pl.col("outcome_status") == "known")
            & pl.col("actual_four_leg_gross_bp").is_finite()
        )
        .then(pl.col("actual_four_leg_gross_bp") - cost)
        .otherwise(None)
        .alias("conditional_net_after_assumed_cost_bp"),
        (
            (pl.col("outcome_status") == "known")
            & pl.col("actual_four_leg_gross_bp").is_finite()
        ).alias("terminal_cashflow_priced"),
    )
    return result.drop(
        column for column in result.columns if column.startswith("_overnight_")
    ).sort(["Date", *PRIMARY_LOOKUP_KEYS[1:], "exit_policy_trial_id"])


def build_daily_prequential_lookup(
    policy_labels: pl.DataFrame,
    sessions: Sequence[str],
    config: ExitPolicyLookupConfig = ExitPolicyLookupConfig(),
    *,
    include_causal_state: bool = False,
    asof_dates: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Build one D-safe rolling diagnostic table per requested decision day."""

    config.validate()
    _validate_policy_labels(policy_labels, "policy labels")
    if policy_labels.is_empty():
        return pl.DataFrame()
    cost_values = policy_labels["assumed_non_price_cycle_cost_bp"].unique().to_list()
    if len(cost_values) != 1 or not math.isclose(
        float(cost_values[0]),
        float(config.assumed_non_price_cycle_cost_bp),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "policy-label assumed cost does not match lookup configuration"
        )
    calendar = _normalise_sessions(sessions)
    selected = _normalise_sessions(asof_dates or calendar)
    missing_dates = sorted(set(selected) - set(calendar))
    if missing_dates:
        raise ValueError(f"asof dates absent from calendar: {missing_dates[:5]}")
    keys = [*PRIMARY_LOOKUP_KEYS]
    if include_causal_state:
        keys.extend(OPTIONAL_CAUSAL_STATE_KEYS)
    _require(policy_labels, set(keys), "policy labels")

    calendar_index = {date: index for index, date in enumerate(calendar)}
    outputs: list[pl.DataFrame] = []
    for asof_date in selected:
        index = calendar_index[asof_date]
        prior_dates = calendar[max(0, index - config.lookback_sessions) : index]
        if not prior_dates:
            continue
        # Preserve the full prior admission cohort.  A future or null label
        # maturity is represented as pending at D; filtering it away would
        # create a hindsight-selected denominator.
        training = policy_labels.filter(pl.col("Date").is_in(prior_dates))
        if training.is_empty():
            continue
        summary = _aggregate_lookup(
            training,
            keys=keys,
            window_sessions=len(prior_dates),
            asof_date=asof_date,
            config=config,
        ).with_columns(
            pl.lit(asof_date).alias("asof_date"),
            pl.lit(prior_dates[0]).alias("window_start_date"),
            pl.lit(prior_dates[-1]).alias("window_end_date"),
            pl.lit(len(prior_dates)).cast(pl.Int64).alias("window_sessions"),
            pl.lit(config.lookback_sessions)
            .cast(pl.Int64)
            .alias("lookback_sessions"),
            pl.lit(include_causal_state).alias("causal_state_in_lookup_key"),
            pl.lit(config.estimator_version).alias("estimator_version"),
            pl.lit(True).alias("execution_safe_snapshot"),
            pl.lit(False).alias("contains_target_day_outcome"),
        )
        outputs.append(summary)
    if not outputs:
        return pl.DataFrame()
    return pl.concat(outputs, how="diagonal_relaxed", rechunk=True).sort(
        ["asof_date", *keys]
    )


def _entry_dimensions(action_facts: pl.DataFrame) -> pl.DataFrame:
    required = {
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
    }
    _require(action_facts, required, "entry action facts")
    if action_facts.is_empty():
        return pl.DataFrame()
    if action_facts.select("policy_generation_id").n_unique() != action_facts.height:
        raise ValueError("entry action facts duplicate policy_generation_id")
    bad_lineage = action_facts.filter(
        pl.col("source_asof_date").is_null()
        | (pl.col("source_asof_date") >= pl.col("Date"))
    )
    if bad_lineage.height:
        raise ValueError("entry action source_asof_date must strictly precede Date")
    return action_facts.select(
        pl.col("policy_generation_id").alias("entry_policy_generation_id"),
        pl.col("lookup_action_id").alias("entry_q"),
        "boundary_quantile",
        "source_asof_date",
        pl.col("parameter_version").alias("entry_parameter_version_dimension"),
        *OPTIONAL_CAUSAL_STATE_KEYS,
    )


def _build_strict_labels(
    strict_raw: pl.DataFrame,
    dimensions: pl.DataFrame,
    cost: float,
) -> pl.DataFrame:
    labels = strict_raw.select(
        pl.col("path_id").alias("policy_label_id"),
        "exit_policy_trial_id",
        "physical_path_id",
        "physical_entry_id",
        "physical_exit_raw_order_fact_id",
        "Date",
        "label_end_date",
        "ValueCode",
        "QuoteCode",
        "entry_route",
        "entry_policy_generation_id",
        "entry_lookup_action_id",
        "exit_rule_id",
        "exit_route",
        "exit_rule_source_asof_date",
        "exit_threshold_basis_bp",
        "outcome_status",
        pl.col("terminal_branch").cast(pl.String).alias("terminal_branch"),
        pl.col("filled_cashflow_before_cost_bp")
        .cast(pl.Float64)
        .alias("actual_four_leg_gross_bp"),
        "normalization_notional_twd",
        "source_exit_branch_status",
        "terminal_mapping_reason",
        "exit_cancel_ack_observed",
        "exit_cancel_race_modeled",
        "exit_active_sibling_cancel_count",
        "exit_prior_unacked_cancel_count_before_winner",
        "joint_volume_allocated",
    ).join(
        dimensions,
        on="entry_policy_generation_id",
        how="left",
        validate="m:1",
    )
    if labels.filter(pl.col("entry_q").is_null()).height:
        raise ValueError("strict path did not resolve to entry q/state dimensions")
    if labels.filter(pl.col("entry_q") != pl.col("entry_lookup_action_id")).height:
        raise ValueError("strict entry lookup id disagrees with entry q")
    if labels.filter(
        pl.col("exit_rule_source_asof_date").is_null()
        | (pl.col("exit_rule_source_asof_date") >= pl.col("Date"))
    ).height:
        raise ValueError("exit rule source_asof_date must strictly precede Date")
    priced = (
        (pl.col("outcome_status") == "known")
        & pl.col("actual_four_leg_gross_bp").is_finite()
    )
    return labels.with_columns(
        pl.col("outcome_status").alias("strict_outcome_status"),
        pl.col("terminal_branch").alias("strict_terminal_branch"),
        pl.lit("strict_observed_cancel_semantics_v1").alias("scenario"),
        pl.lit(False).alias("terminal_model_assumption"),
        pl.lit(False).alias("strict_cancel_ambiguity_overridden"),
        (
            pl.col("exit_active_sibling_cancel_count")
            + pl.col("exit_prior_unacked_cancel_count_before_winner")
        ).alias("any_unacked_cancel_count"),
        pl.lit(cost).alias("assumed_non_price_cycle_cost_bp"),
        pl.when(priced)
        .then(pl.col("actual_four_leg_gross_bp") - cost)
        .otherwise(None)
        .alias("conditional_net_after_assumed_cost_bp"),
        priced.alias("terminal_cashflow_priced"),
        pl.lit(False).alias("overnight_label_attached"),
        pl.lit("same_day_exit_maker_replay_or_unresolved")
        .alias("terminal_label_source"),
        pl.when(pl.col("outcome_status") == "known")
        .then(None)
        .otherwise(pl.col("terminal_mapping_reason"))
        .alias("unresolved_reason"),
        pl.lit(True).alias("flat_cost_sensitivity_analysis_only"),
        pl.lit(False).alias("production_eligible"),
        pl.lit(False).alias("ev_ready"),
        pl.lit(True).alias("conditional_on_established_entry"),
        pl.lit(False).alias("policy_rows_safe_to_sum_across_q_rule_route"),
        pl.lit(True).alias("price_slippage_already_in_gross"),
    ).sort(["Date", *PRIMARY_LOOKUP_KEYS[1:], "exit_policy_trial_id"])


def _build_nominal_labels(
    strict: pl.DataFrame,
    nominal_raw: pl.DataFrame,
    cost: float,
) -> pl.DataFrame:
    nominal = nominal_raw.select(
        "exit_policy_trial_id",
        pl.col("modeled_outcome_status").alias("_v0_outcome_status"),
        pl.col("modeled_terminal_branch")
        .cast(pl.String)
        .alias("_v0_terminal_branch"),
        pl.col("gross_cycle_bp").cast(pl.Float64).alias("_v0_gross_bp"),
        "strict_cancel_ambiguity_overridden",
        "any_unacked_cancel_count",
        "model_version",
        "model_assumption",
        "analysis_only",
        "production_eligible",
        "eligible_for_strict_ev_lookup",
    )
    if nominal.height != strict.height:
        raise ValueError("nominal V0 and strict path populations differ in size")
    joined = strict.drop(
        "strict_cancel_ambiguity_overridden",
        "any_unacked_cancel_count",
        "production_eligible",
    ).join(
        nominal,
        on="exit_policy_trial_id",
        how="left",
        validate="1:1",
    )
    if joined.filter(pl.col("_v0_outcome_status").is_null()).height:
        raise ValueError("strict policy path is missing nominal V0 projection")
    priced = (
        (pl.col("_v0_outcome_status") == "known")
        & pl.col("_v0_gross_bp").is_finite()
    )
    return joined.with_columns(
        pl.concat_str(
            [pl.lit("nominal-v0-"), pl.col("exit_policy_trial_id")]
        ).alias("policy_label_id"),
        pl.col("Date").alias("label_end_date"),
        pl.col("_v0_outcome_status").alias("outcome_status"),
        pl.col("_v0_terminal_branch")
        .cast(pl.String)
        .alias("terminal_branch"),
        pl.col("_v0_gross_bp").alias("actual_four_leg_gross_bp"),
        pl.lit("nominal_instant_cancel_v0").alias("scenario"),
        pl.col("model_assumption").alias("terminal_model_assumption"),
        pl.when(priced)
        .then(pl.col("_v0_gross_bp") - cost)
        .otherwise(None)
        .alias("conditional_net_after_assumed_cost_bp"),
        priced.alias("terminal_cashflow_priced"),
        pl.lit(False).alias("overnight_label_attached"),
        pl.lit("nominal_instant_cancel_v0_same_day_or_unresolved")
        .alias("terminal_label_source"),
        pl.when(pl.col("_v0_outcome_status") == "known")
        .then(None)
        .otherwise(pl.col("terminal_mapping_reason"))
        .alias("unresolved_reason"),
        pl.lit(True).alias("flat_cost_sensitivity_analysis_only"),
        pl.lit(False).alias("ev_ready"),
    ).drop("_v0_outcome_status", "_v0_terminal_branch", "_v0_gross_bp").sort(
        ["Date", *PRIMARY_LOOKUP_KEYS[1:], "exit_policy_trial_id"]
    )


def _aggregate_lookup(
    training: pl.DataFrame,
    *,
    keys: Sequence[str],
    window_sessions: int,
    asof_date: str,
    config: ExitPolicyLookupConfig,
) -> pl.DataFrame:
    matured = (
        pl.col("label_end_date").is_not_null()
        & (pl.col("label_end_date") < asof_date)
    )
    pending = ~matured
    known = matured & (pl.col("outcome_status") == "known")
    censored = matured & (pl.col("outcome_status") == "censored")
    unknown = matured & (pl.col("outcome_status") == "unknown")
    priced = known & pl.col("actual_four_leg_gross_bp").is_finite()
    known_net = known & pl.col("conditional_net_after_assumed_cost_bp").is_finite()
    same_day = known & pl.col("terminal_branch").is_in(
        list(SAME_DAY_TERMINAL_BRANCHES)
    )
    overnight = known & pl.col("terminal_branch").is_in(
        list(OVERNIGHT_TERMINAL_BRANCHES)
    )
    grouped = training.group_by(list(keys)).agg(
        pl.len().cast(pl.Int64).alias("n_policy_paths"),
        pl.col("Date").n_unique().cast(pl.Int64).alias("n_training_sessions"),
        pl.col("Date").min().alias("train_start_date"),
        pl.col("Date").max().alias("train_end_date"),
        pl.col("label_end_date")
        .filter(matured)
        .max()
        .alias("label_cutoff_date"),
        pl.col("physical_entry_id")
        .n_unique()
        .cast(pl.Int64)
        .alias("n_physical_entries"),
        known.sum().cast(pl.Int64).alias("n_outcome_known"),
        censored.sum().cast(pl.Int64).alias("n_outcome_censored"),
        unknown.sum().cast(pl.Int64).alias("n_outcome_unknown"),
        matured.sum().cast(pl.Int64).alias("n_labels_matured"),
        pending.sum().cast(pl.Int64).alias("n_labels_pending"),
        priced.sum().cast(pl.Int64).alias("n_terminal_cashflow_priced"),
        same_day.sum().cast(pl.Int64).alias("n_same_day_terminal_known"),
        overnight.sum().cast(pl.Int64).alias("n_overnight_terminal_known"),
        (pl.col("source_exit_branch_status") == "cancel_race_unknown")
        .sum()
        .cast(pl.Int64)
        .alias("n_strict_cancel_race_unknown"),
        pl.col("strict_cancel_ambiguity_overridden")
        .sum()
        .cast(pl.Int64)
        .alias("n_cancel_ambiguity_overridden_by_model"),
        (pl.col("any_unacked_cancel_count") > 0)
        .sum()
        .cast(pl.Int64)
        .alias("n_paths_with_unacked_cancel_request"),
        pl.col("actual_four_leg_gross_bp")
        .filter(priced)
        .sum()
        .alias("_known_gross_sum_bp"),
        pl.col("actual_four_leg_gross_bp")
        .filter(priced)
        .mean()
        .alias("conditional_known_gross_mean_bp"),
        pl.col("actual_four_leg_gross_bp")
        .filter(priced)
        .median()
        .alias("conditional_known_gross_p50_bp"),
        pl.col("actual_four_leg_gross_bp")
        .filter(priced)
        .quantile(0.05, interpolation="linear")
        .alias("conditional_known_gross_p05_bp"),
        pl.col("actual_four_leg_gross_bp")
        .filter(priced)
        .quantile(0.95, interpolation="linear")
        .alias("conditional_known_gross_p95_bp"),
        pl.col("conditional_net_after_assumed_cost_bp")
        .filter(known_net)
        .sum()
        .alias("_known_net_after_assumed_cost_sum_bp"),
        pl.col("conditional_net_after_assumed_cost_bp")
        .filter(known_net)
        .mean()
        .alias("conditional_known_net_after_assumed_cost_mean_bp"),
        pl.col("conditional_net_after_assumed_cost_bp")
        .filter(known_net)
        .median()
        .alias("conditional_known_net_after_assumed_cost_p50_bp"),
        pl.col("scenario").n_unique().alias("_scenario_count"),
        pl.col("scenario").first().alias("scenario"),
        pl.col("terminal_model_assumption").any().alias("terminal_model_assumption"),
    )
    if grouped.filter(pl.col("_scenario_count") != 1).height:
        raise ValueError("one lookup cell mixes strict and nominal scenarios")

    result = grouped.with_columns(
        _ratio("n_outcome_known", "n_policy_paths").alias("outcome_label_coverage"),
        _ratio("n_labels_matured", "n_policy_paths").alias(
            "label_maturity_coverage"
        ),
        _ratio("n_labels_pending", "n_policy_paths").alias("label_pending_rate"),
        _ratio("n_outcome_censored", "n_policy_paths").alias("censor_rate"),
        _ratio("n_outcome_unknown", "n_policy_paths").alias("unknown_rate"),
        _ratio("n_terminal_cashflow_priced", "n_outcome_known").alias(
            "cashflow_coverage_given_known"
        ),
        _ratio("n_paths_with_unacked_cancel_request", "n_policy_paths").alias(
            "unacked_cancel_request_rate"
        ),
        (pl.col("n_policy_paths") - pl.col("n_outcome_known")).alias(
            "n_unpriced_terminal_mass"
        ),
        (
            (pl.col("n_outcome_known") == pl.col("n_policy_paths"))
            & (
                pl.col("n_terminal_cashflow_priced")
                == pl.col("n_outcome_known")
            )
        ).alias("terminal_cashflow_point_identified"),
        (pl.col("_known_gross_sum_bp") / pl.col("n_policy_paths")).alias(
            "known_gross_zero_for_unpriced_sensitivity_bp"
        ),
        (
            pl.col("_known_net_after_assumed_cost_sum_bp")
            / pl.col("n_policy_paths")
        ).alias(
            "known_net_after_assumed_cost_zero_for_unpriced_sensitivity_bp"
        ),
    ).with_columns(
        pl.when(pl.col("terminal_cashflow_point_identified"))
        .then(pl.col("conditional_known_gross_mean_bp"))
        .otherwise(None)
        .alias("identified_gross_mean_bp"),
        pl.when(pl.col("terminal_cashflow_point_identified"))
        .then(pl.col("conditional_known_gross_mean_bp"))
        .otherwise(None)
        .alias("gross_mean_identification_lower_bp"),
        pl.when(pl.col("terminal_cashflow_point_identified"))
        .then(pl.col("conditional_known_gross_mean_bp"))
        .otherwise(None)
        .alias("gross_mean_identification_upper_bp"),
        pl.when(pl.col("terminal_cashflow_point_identified"))
        .then(pl.col("conditional_known_net_after_assumed_cost_mean_bp"))
        .otherwise(None)
        .alias("identified_net_after_assumed_cost_sensitivity_mean_bp"),
        pl.when(pl.col("terminal_cashflow_point_identified"))
        .then(pl.lit("point_identified"))
        .otherwise(pl.lit("unbounded_without_unresolved_cashflow_limits"))
        .alias("cashflow_identification_status"),
        pl.lit(window_sessions >= config.min_history_sessions).alias(
            "history_support_gate"
        ),
        (pl.col("n_training_sessions") >= config.min_group_sessions).alias(
            "group_session_support_gate"
        ),
        (pl.col("n_outcome_known") >= config.min_known_paths).alias(
            "known_path_support_gate"
        ),
        (pl.col("outcome_label_coverage") >= config.min_outcome_label_coverage).alias(
            "outcome_label_coverage_gate"
        ),
        (pl.col("censor_rate") <= config.max_censor_rate).alias("censor_rate_gate"),
        (pl.col("unknown_rate") <= config.max_unknown_rate).alias("unknown_rate_gate"),
    )
    statistical_support = (
        pl.col("history_support_gate")
        & pl.col("group_session_support_gate")
        & pl.col("known_path_support_gate")
    )
    reasons = pl.concat_str(
        [
            pl.when(~pl.col("history_support_gate"))
            .then(pl.lit("insufficient_history_sessions;"))
            .otherwise(pl.lit("")),
            pl.when(~pl.col("group_session_support_gate"))
            .then(pl.lit("insufficient_group_sessions;"))
            .otherwise(pl.lit("")),
            pl.when(~pl.col("known_path_support_gate"))
            .then(pl.lit("insufficient_known_paths;"))
            .otherwise(pl.lit("")),
            pl.when(~pl.col("terminal_cashflow_point_identified"))
            .then(pl.lit("unpriced_censored_unknown_or_pending;"))
            .otherwise(pl.lit("")),
            pl.when(pl.col("terminal_model_assumption"))
            .then(pl.lit("nominal_instant_cancel_model_assumption;"))
            .otherwise(pl.lit("")),
            pl.lit("incomplete_production_cost_profile"),
        ],
        separator="",
    )
    return result.with_columns(
        statistical_support.alias("statistical_support_ready"),
        reasons.alias("not_ready_reasons"),
        pl.when(~pl.col("history_support_gate"))
        .then(pl.lit("insufficient_history_sessions"))
        .when(~pl.col("group_session_support_gate"))
        .then(pl.lit("insufficient_group_sessions"))
        .when(~pl.col("known_path_support_gate"))
        .then(pl.lit("insufficient_known_paths"))
        .when(~pl.col("terminal_cashflow_point_identified"))
        .then(pl.lit("unpriced_censored_unknown_or_pending"))
        .when(pl.col("terminal_model_assumption"))
        .then(pl.lit("nominal_instant_cancel_model_assumption"))
        .otherwise(pl.lit("incomplete_production_cost_profile"))
        .alias("ev_status"),
        pl.lit(False).alias("ev_ready"),
        pl.lit(None, dtype=pl.Float64).alias("expected_net_cashflow_bp"),
        pl.lit(None, dtype=pl.Float64).alias("action_score_bp"),
        pl.lit(config.assumed_non_price_cycle_cost_bp).alias(
            "assumed_non_price_cycle_cost_bp"
        ),
        pl.lit("non_price_roundtrip_fees_tax_commission_only").alias(
            "flat_cost_scope"
        ),
        pl.lit(True).alias("analysis_only"),
        pl.lit(False).alias("production_eligible"),
        pl.lit(True).alias("conditional_on_established_entry"),
        pl.lit(False).alias("entry_fill_probability_included"),
        pl.lit(True).alias("unknown_and_censored_retained"),
        pl.lit(True).alias("unmatured_labels_retained_as_pending"),
        pl.lit(True).alias("zero_imputation_sensitivity_only"),
        pl.lit(False).alias("zero_imputation_used_for_ev"),
        pl.lit(False).alias("finite_unresolved_cashflow_limits_assumed"),
        pl.lit(False).alias("policy_rows_safe_to_sum_across_q_rule_route"),
        pl.lit(True).alias("price_slippage_already_in_gross"),
    ).drop(
        "_known_gross_sum_bp",
        "_known_net_after_assumed_cost_sum_bp",
        "_scenario_count",
    )


def _validate_policy_labels(frame: pl.DataFrame, source: str) -> None:
    if frame.is_empty():
        return
    required = {
        "policy_label_id",
        "exit_policy_trial_id",
        "physical_entry_id",
        "Date",
        "label_end_date",
        *PRIMARY_LOOKUP_KEYS,
        *OPTIONAL_CAUSAL_STATE_KEYS,
        "outcome_status",
        "terminal_branch",
        "actual_four_leg_gross_bp",
        "conditional_net_after_assumed_cost_bp",
        "assumed_non_price_cycle_cost_bp",
        "scenario",
        "terminal_model_assumption",
        "source_exit_branch_status",
        "strict_cancel_ambiguity_overridden",
        "any_unacked_cancel_count",
    }
    _require(frame, required, source)
    if frame.select("policy_label_id").n_unique() != frame.height:
        raise ValueError(f"{source} duplicate policy_label_id")
    if frame.select("exit_policy_trial_id").n_unique() != frame.height:
        raise ValueError(f"{source} duplicate exit_policy_trial_id")
    invalid_outcome = frame.filter(
        ~pl.col("outcome_status").is_in(OUTCOME_STATUSES).fill_null(False)
    )
    if invalid_outcome.height:
        raise ValueError(f"{source} invalid outcome_status")
    invalid_date = frame.filter(
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
    if invalid_date.height:
        raise ValueError(f"{source} invalid label maturity date")
    malformed_known = frame.filter(
        (pl.col("outcome_status") == "known")
        & (
            pl.col("terminal_branch").is_null()
            | pl.col("actual_four_leg_gross_bp").is_null()
            | ~pl.col("actual_four_leg_gross_bp").is_finite()
        )
    )
    if malformed_known.height:
        raise ValueError(f"{source} known outcome lacks priced cashflow")
    malformed_unresolved = frame.filter(
        (pl.col("outcome_status") != "known")
        & (
            pl.col("terminal_branch").is_not_null()
            | pl.col("actual_four_leg_gross_bp").is_not_null()
            | pl.col("conditional_net_after_assumed_cost_bp").is_not_null()
        )
    )
    if malformed_unresolved.height:
        raise ValueError(f"{source} unresolved outcome claims terminal cashflow")


def _normalise_sessions(values: Sequence[str]) -> list[str]:
    sessions = [str(value) for value in values]
    if not sessions:
        raise ValueError("sessions must be non-empty")
    if len(sessions) != len(set(sessions)):
        raise ValueError("sessions contain duplicates")
    if sessions != sorted(sessions):
        raise ValueError("sessions must be sorted ascending")
    if any(len(value) != 8 or not value.isdigit() for value in sessions):
        raise ValueError("sessions must use YYYYMMDD strings")
    return sessions


def _ratio(numerator: str, denominator: str) -> pl.Expr:
    return (
        pl.when(pl.col(denominator) > 0)
        .then(pl.col(numerator) / pl.col(denominator))
        .otherwise(None)
    )


def _validate_cost(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError("assumed non-price cost must be finite and non-negative")
    return float(value)


def _empty_label_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "policy_label_id": pl.String,
            "exit_policy_trial_id": pl.String,
            "physical_entry_id": pl.String,
            "Date": pl.String,
            "label_end_date": pl.String,
            "ValueCode": pl.String,
            "entry_route": pl.String,
            "entry_q": pl.String,
            "exit_rule_id": pl.String,
            "exit_route": pl.String,
            **{name: pl.String for name in OPTIONAL_CAUSAL_STATE_KEYS},
            "outcome_status": pl.String,
            "terminal_branch": pl.String,
            "actual_four_leg_gross_bp": pl.Float64,
            "conditional_net_after_assumed_cost_bp": pl.Float64,
            "scenario": pl.String,
        }
    )


def _require(frame: pl.DataFrame, columns: set[str], source: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")
