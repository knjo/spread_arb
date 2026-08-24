"""Causal lookup-table EV and legal-tick action surfaces.

This module is deliberately a small, pure bridge between the existing maker
target geometry and the eventual four-leg replay.  It does three things:

* turns threshold aliases into legal absolute maker-price actions and de-dups
  aliases that round to the same price at the same decision state;
* builds a daily lookup table from *terminal path* facts whose labels mature
  strictly before the decision day; and
* joins the two tables and marks at most one positive-EV action per decision.

The probability unit is one mutually-exclusive terminal path.  Fill, hedge,
exit, and overnight marginals are reported as diagnostics only; they are never
multiplied to manufacture an EV.  Unknown and censored paths remain explicit
and are excluded from the known-terminal denominator.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import math
from typing import Iterable, Mapping, Sequence

import polars as pl

from .targets import (
    ROUTE_SPECS,
    absolute_price_tick,
    effective_basis_bp,
    is_passive_target,
    target_price_for_basis,
    tick_index_to_price,
)


TERMINAL_BRANCHES: tuple[str, ...] = (
    "no_fill_cancel",
    "no_fill_expire",
    "partial_unresolved",
    "hedge_failure_emergency",
    "same_day_target_exit",
    "same_day_aggressive_exit",
    "overnight_exit",
    "overnight_unresolved",
    "expiry_forced_flat",
    "expiry_settlement",
    "emergency_exit",
)

OUTCOME_STATUSES = frozenset({"known", "censored", "unknown"})
EXECUTION_STATUS_VALUES: Mapping[str, frozenset[str]] = {
    "cancel_status": frozenset({"cancelled", "not_cancelled", "unknown"}),
    "entry_fill_status": frozenset({"no_fill", "partial", "full", "unknown"}),
    "topup_status": frozenset(
        {"not_applicable", "completed", "failed", "pending", "unknown"}
    ),
    "hedge_50ms_status": frozenset(
        {"not_applicable", "executable", "partial", "failed", "unknown"}
    ),
    "same_day_exit_status": frozenset(
        {
            "not_applicable",
            "target_exit",
            "aggressive_exit",
            "no_exit",
            "failed",
            "unknown",
        }
    ),
    "overnight_status": frozenset(
        {"not_applicable", "carried_open", "closed", "unresolved", "unknown"}
    ),
    "expiry_status": frozenset(
        {"not_applicable", "forced_flat", "settled", "unresolved", "unknown"}
    ),
    "emergency_status": frozenset(
        {"not_applicable", "triggered", "not_triggered", "unknown"}
    ),
}

DEFAULT_COST_COLUMNS: tuple[str, ...] = (
    "fee_cost_bp",
    "tax_cost_bp",
    "financing_cost_bp",
    "overnight_cost_bp",
    "cancel_cost_bp",
    "emergency_cost_bp",
)

SLIPPAGE_DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "hedge_slippage_bp_50ms",
    "exit_slippage_bp",
)

DEFAULT_LOOKUP_KEYS: tuple[str, ...] = (
    "ValueCode",
    "route",
    "target_offset_ticks",
    "state_family",
    "state_bucket",
    "lifecycle_policy_version",
    "queue_scenario",
    "intended_maker_quantity",
    "hedge_quantity",
)

ACTION_ALIAS_DECISION_KEYS: tuple[str, ...] = (
    "Date",
    "ValueCode",
    "QuoteCode",
    "route",
    "decision_id",
)

ACTION_SELECTION_KEYS: tuple[str, ...] = (
    "Date",
    "ValueCode",
    "QuoteCode",
    "route",
    "spread_pair_epoch",
)

DECISION_SCORE_KEYS: tuple[str, ...] = (
    *ACTION_ALIAS_DECISION_KEYS,
    "decision_sequence",
)

ACTION_IDENTITY_KEYS: tuple[str, ...] = (
    "Date",
    "ValueCode",
    "QuoteCode",
    "route",
    "stage",
    "spread_pair_epoch",
    "absolute_target_tick",
)


@dataclass(frozen=True)
class LegalActionSurface:
    """Canonical raw actions, threshold aliases, and causal decision actions."""

    actions: pl.DataFrame
    aliases: pl.DataFrame
    decision_actions: pl.DataFrame


@dataclass(frozen=True)
class EVLookupConfig:
    """Support and cost contract for the empirical pathwise lookup table."""

    lookback_sessions: int = 60
    min_history_sessions: int = 40
    min_group_sessions: int = 20
    min_known_paths: int = 100
    min_outcome_label_coverage: float = 0.90
    max_censor_rate: float = 0.10
    max_unknown_rate: float = 0.05
    capital_charge_bp_per_hour: float = 0.0
    downside_penalty_weight: float = 0.0
    emergency_penalty_bp_per_event: float = 0.0
    lookup_keys: tuple[str, ...] = DEFAULT_LOOKUP_KEYS
    required_cost_columns: tuple[str, ...] = DEFAULT_COST_COLUMNS
    estimator_version: str = "pathwise_lookup_ev_v1"

    def validate(self) -> None:
        if self.lookback_sessions <= 0:
            raise ValueError("lookback_sessions must be positive")
        if not 1 <= self.min_history_sessions <= self.lookback_sessions:
            raise ValueError(
                "min_history_sessions must be in [1, lookback_sessions]"
            )
        if self.min_group_sessions <= 0 or self.min_known_paths <= 0:
            raise ValueError("group support thresholds must be positive")
        for name, value in (
            ("min_outcome_label_coverage", self.min_outcome_label_coverage),
            ("max_censor_rate", self.max_censor_rate),
            ("max_unknown_rate", self.max_unknown_rate),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if (
            not math.isfinite(self.capital_charge_bp_per_hour)
            or self.capital_charge_bp_per_hour < 0
        ):
            raise ValueError("capital charge must be finite and non-negative")
        if (
            not math.isfinite(self.downside_penalty_weight)
            or self.downside_penalty_weight < 0
        ):
            raise ValueError("downside penalty weight must be finite and non-negative")
        if (
            not math.isfinite(self.emergency_penalty_bp_per_event)
            or self.emergency_penalty_bp_per_event < 0
        ):
            raise ValueError(
                "emergency penalty must be finite and non-negative"
            )
        if not self.lookup_keys or len(set(self.lookup_keys)) != len(
            self.lookup_keys
        ):
            raise ValueError("lookup_keys must be non-empty and unique")
        if not self.required_cost_columns or len(
            set(self.required_cost_columns)
        ) != len(self.required_cost_columns):
            raise ValueError("required_cost_columns must be non-empty and unique")
        if any(
            name in {"filled_cashflow_before_cost_bp", "capital_time_seconds"}
            for name in self.required_cost_columns
        ):
            raise ValueError(
                "filled cashflow and capital time are separate from cost columns"
            )


def build_legal_action_surface(candidates: pl.DataFrame) -> LegalActionSurface:
    """Resolve threshold aliases to legal, passive absolute-price actions.

    ``current_state_eligible`` is the upstream TrialMatch/book/ref/freshness
    gate.  A false or null gate stays in ``aliases`` with an explicit status,
    but cannot produce an action.  For a legal row, aliases that round to the
    same absolute tick in the same SpreadPair epoch share one ``action_id`` and
    therefore one future raw queue replay fact.  ``decision_id`` is audit-only;
    it is never part of canonical action identity.
    """

    required = {
        "Date",
        "ValueCode",
        "QuoteCode",
        "route",
        "decision_id",
        "decision_sequence",
        "spread_pair_epoch",
        "action_alias",
        "threshold_basis_bp",
        "spot_bid",
        "spot_ask",
        "fut_bid",
        "fut_ask",
        "fut_exec_bid",
        "fut_exec_ask",
        "current_state_eligible",
        "source_asof_date",
        "contains_target_day_outcome",
        "parameter_version",
        "lifecycle_policy_version",
        "queue_scenario",
        "intended_maker_quantity",
        "hedge_quantity",
    }
    _require(candidates, required, "action candidates")
    if candidates.is_empty():
        return LegalActionSurface(pl.DataFrame(), pl.DataFrame(), pl.DataFrame())

    alias_key = [*ACTION_ALIAS_DECISION_KEYS, "action_alias"]
    if candidates.select(alias_key).n_unique() != candidates.height:
        raise ValueError("action candidates contain duplicate decision aliases")
    _validate_prior_lineage(candidates)
    _validate_policy_lineage(candidates)
    if candidates.filter(
        pl.col("parameter_version").is_null()
        | (pl.col("parameter_version").cast(pl.String).str.len_chars() == 0)
    ).height:
        raise ValueError("action candidate parameter_version must be present")

    alias_rows: list[dict[str, object]] = []
    for row in candidates.iter_rows(named=True):
        record = dict(row)
        record.update(
            {
                "action_id": None,
                "absolute_target_price": None,
                "absolute_target_tick": None,
                "effective_basis_bp": None,
                "target_offset_ticks": None,
                "stage": None,
                "action_status": None,
            }
        )
        eligible = row["current_state_eligible"]
        if eligible is not True:
            record["action_status"] = (
                "current_state_ineligible"
                if eligible is False
                else "current_state_unknown"
            )
            alias_rows.append(record)
            continue

        try:
            route = str(row["route"])
            if route not in ROUTE_SPECS:
                raise ValueError("unknown route")
            spec = ROUTE_SPECS[route]
            session_date = str(row["Date"])
            threshold = _finite_number(
                row["threshold_basis_bp"], "threshold_basis_bp"
            )
            quotes = {
                name: _optional_number(row[name])
                for name in (
                    "spot_bid",
                    "spot_ask",
                    "fut_exec_bid",
                    "fut_exec_ask",
                )
            }
            target = target_price_for_basis(
                route,
                threshold,
                session_date=session_date,
                **quotes,
            )
            target_tick = absolute_price_tick(
                target,
                market=spec.maker_market,
                session_date=session_date,
            )
            effective = effective_basis_bp(route, target, **quotes)
            passive = is_passive_target(route, target, **quotes)
            maker_bbo = _maker_same_side_bbo(row, route)
            maker_bbo_tick = absolute_price_tick(
                maker_bbo,
                market=spec.maker_market,
                session_date=session_date,
            )
            offset = (
                target_tick - maker_bbo_tick
                if spec.maker_side == "ask"
                else maker_bbo_tick - target_tick
            )
        except (KeyError, TypeError, ValueError):
            record["action_status"] = "invalid_target_inputs"
            alias_rows.append(record)
            continue

        record.update(
            {
                "absolute_target_price": target,
                "absolute_target_tick": target_tick,
                "effective_basis_bp": effective,
                "target_offset_ticks": offset,
                "stage": spec.stage,
                "action_status": "legal" if passive else "target_not_passive",
            }
        )
        alias_rows.append(record)

    aliases = pl.from_dicts(alias_rows, infer_schema_length=None)
    legal = aliases.filter(pl.col("action_status") == "legal")
    if legal.is_empty():
        return LegalActionSurface(pl.DataFrame(), aliases, pl.DataFrame())

    # Validate aliases observed at one decision state, then canonicalise raw
    # order identity across repeated decisions in the same SpreadPair epoch.
    decision_identity = [
        *ACTION_ALIAS_DECISION_KEYS,
        "decision_sequence",
        "spread_pair_epoch",
        "absolute_target_tick",
    ]
    fixed_within_action = [
        "absolute_target_price",
        "effective_basis_bp",
        "target_offset_ticks",
        "source_asof_date",
        "parameter_version",
        "lifecycle_policy_version",
        "queue_scenario",
        "intended_maker_quantity",
        "hedge_quantity",
    ]
    inconsistent = legal.group_by(decision_identity).agg(
        *(pl.col(name).n_unique().alias(name) for name in fixed_within_action)
    ).filter(pl.any_horizontal(pl.col(name) != 1 for name in fixed_within_action))
    if inconsistent.height:
        raise ValueError("aliases for one legal action disagree on market state")

    action_rows: list[dict[str, object]] = []
    action_ids: list[dict[str, object]] = []
    identity = list(ACTION_IDENTITY_KEYS)
    for key, group in legal.group_by(identity, maintain_order=True):
        key_values = key if isinstance(key, tuple) else (key,)
        key_map = dict(zip(identity, key_values, strict=True))
        ordered = group.sort(["decision_sequence", "decision_id", "action_alias"])
        first = ordered.row(0, named=True)
        action_id = _action_id(key_map)
        action_rows.append(
            {
                "action_id": action_id,
                **key_map,
                "maker_market": ROUTE_SPECS[str(first["route"])].maker_market,
                "maker_side": ROUTE_SPECS[str(first["route"])].maker_side,
                "submit_decision_id": first["decision_id"],
                "submit_decision_sequence": first["decision_sequence"],
                "absolute_target_price": first["absolute_target_price"],
                "effective_basis_bp": first["effective_basis_bp"],
                "target_offset_ticks": first["target_offset_ticks"],
                "action_alias_count": group["action_alias"].n_unique(),
                "decision_alias_count": group.height,
                "action_aliases": sorted(
                    {str(value) for value in group["action_alias"].to_list()}
                ),
                "threshold_basis_bp_min": min(
                    float(value) for value in group["threshold_basis_bp"].to_list()
                ),
                "threshold_basis_bp_max": max(
                    float(value) for value in group["threshold_basis_bp"].to_list()
                ),
                "source_asof_date": first["source_asof_date"],
                "parameter_version": first["parameter_version"],
                "lifecycle_policy_version": first["lifecycle_policy_version"],
                "queue_scenario": first["queue_scenario"],
                "intended_maker_quantity": first["intended_maker_quantity"],
                "hedge_quantity": first["hedge_quantity"],
                "execution_safe_snapshot": True,
                "contains_target_day_outcome": False,
            }
        )
        action_ids.append({**key_map, "action_id": action_id})

    actions = pl.from_dicts(action_rows, infer_schema_length=None).sort(
        [*ACTION_SELECTION_KEYS, "absolute_target_tick"]
    )
    alias_id_frame = pl.from_dicts(action_ids, infer_schema_length=None)
    aliases = aliases.join(
        alias_id_frame,
        on=identity,
        how="left",
        suffix="_resolved",
        validate="m:1",
    ).with_columns(
        pl.coalesce("action_id_resolved", "action_id").alias("action_id")
    ).drop("action_id_resolved")
    aliases = aliases.sort([*ACTION_ALIAS_DECISION_KEYS, "action_alias"])
    decision_key = [
        *ACTION_ALIAS_DECISION_KEYS,
        "decision_sequence",
        "spread_pair_epoch",
        "stage",
        "absolute_target_tick",
        "action_id",
    ]
    decision_actions = aliases.filter(pl.col("action_status") == "legal").group_by(
        decision_key, maintain_order=True
    ).agg(
        pl.col("absolute_target_price").first(),
        pl.col("effective_basis_bp").first(),
        pl.col("target_offset_ticks").first(),
        pl.col("action_alias").n_unique().alias("action_alias_count"),
        pl.col("action_alias").unique().sort().alias("action_aliases"),
        pl.col("source_asof_date").first(),
        pl.col("parameter_version").first(),
        pl.col("lifecycle_policy_version").first(),
        pl.col("queue_scenario").first(),
        pl.col("intended_maker_quantity").first(),
        pl.col("hedge_quantity").first(),
    ).with_columns(
        pl.lit(True).alias("execution_safe_snapshot"),
        pl.lit(False).alias("contains_target_day_outcome"),
    ).sort([*ACTION_ALIAS_DECISION_KEYS, "absolute_target_tick"])
    return LegalActionSurface(actions, aliases, decision_actions)


def enumerate_legal_tick_actions(
    decisions: pl.DataFrame,
    *,
    max_ticks_per_decision: int = 128,
) -> LegalActionSurface:
    """Enumerate every legal passive tick in an explicitly supplied range.

    The caller derives ``min_absolute_target_tick`` and
    ``max_absolute_target_tick`` causally (normally from D-1-safe normal/tail
    knots).  Unlike :func:`build_legal_action_surface`, this function does not
    treat those knots as the action set: every exchange tick in the inclusive
    range is evaluated.  Repeated appearances of the same absolute price in
    one SpreadPair epoch collapse to one canonical raw action, with its first
    decision retained as the submit state.
    """

    required = {
        "Date",
        "ValueCode",
        "QuoteCode",
        "route",
        "decision_id",
        "decision_sequence",
        "spread_pair_epoch",
        "min_absolute_target_tick",
        "max_absolute_target_tick",
        "spot_bid",
        "spot_ask",
        "fut_bid",
        "fut_ask",
        "fut_exec_bid",
        "fut_exec_ask",
        "current_state_eligible",
        "source_asof_date",
        "contains_target_day_outcome",
        "parameter_version",
        "lifecycle_policy_version",
        "queue_scenario",
        "intended_maker_quantity",
        "hedge_quantity",
    }
    _require(decisions, required, "tick-enumeration decisions")
    if max_ticks_per_decision <= 0:
        raise ValueError("max_ticks_per_decision must be positive")
    if decisions.is_empty():
        return LegalActionSurface(pl.DataFrame(), pl.DataFrame(), pl.DataFrame())
    decision_key = [*ACTION_ALIAS_DECISION_KEYS, "decision_sequence"]
    if decisions.select(decision_key).n_unique() != decisions.height:
        raise ValueError("tick-enumeration decisions contain duplicate states")
    _validate_prior_lineage(decisions)
    _validate_policy_lineage(decisions)

    rows: list[dict[str, object]] = []
    for row in decisions.sort("decision_sequence").iter_rows(named=True):
        if row["current_state_eligible"] is not True:
            continue
        route = str(row["route"])
        if route not in ROUTE_SPECS:
            raise ValueError(f"unknown route: {route}")
        spec = ROUTE_SPECS[route]
        session_date = str(row["Date"])
        lower = _non_negative_integer(
            row["min_absolute_target_tick"], "min_absolute_target_tick"
        )
        upper = _non_negative_integer(
            row["max_absolute_target_tick"], "max_absolute_target_tick"
        )
        if upper < lower:
            raise ValueError("max_absolute_target_tick must be >= minimum")
        if upper - lower + 1 > max_ticks_per_decision:
            raise ValueError("tick enumeration exceeds max_ticks_per_decision")
        quotes = {
            name: _optional_number(row[name])
            for name in (
                "spot_bid",
                "spot_ask",
                "fut_exec_bid",
                "fut_exec_ask",
            )
        }
        maker_bbo_tick = absolute_price_tick(
            _maker_same_side_bbo(row, route),
            market=spec.maker_market,
            session_date=session_date,
        )
        for tick in range(lower, upper + 1):
            price = tick_index_to_price(
                tick,
                market=spec.maker_market,
                session_date=session_date,
            )
            if not is_passive_target(route, price, **quotes):
                continue
            offset = (
                tick - maker_bbo_tick
                if spec.maker_side == "ask"
                else maker_bbo_tick - tick
            )
            rows.append(
                {
                    **row,
                    "stage": spec.stage,
                    "absolute_target_tick": tick,
                    "absolute_target_price": price,
                    "effective_basis_bp": effective_basis_bp(
                        route, price, **quotes
                    ),
                    "target_offset_ticks": offset,
                }
            )
    if not rows:
        return LegalActionSurface(pl.DataFrame(), pl.DataFrame(), pl.DataFrame())
    expanded = pl.from_dicts(rows, infer_schema_length=None)
    action_rows: list[dict[str, object]] = []
    for key, group in expanded.group_by(ACTION_IDENTITY_KEYS, maintain_order=True):
        key_values = key if isinstance(key, tuple) else (key,)
        key_map = dict(zip(ACTION_IDENTITY_KEYS, key_values, strict=True))
        ordered = group.sort(["decision_sequence", "decision_id"])
        first = ordered.row(0, named=True)
        action_rows.append(
            {
                "action_id": _action_id(key_map),
                **key_map,
                "maker_market": ROUTE_SPECS[str(first["route"])].maker_market,
                "maker_side": ROUTE_SPECS[str(first["route"])].maker_side,
                "submit_decision_id": first["decision_id"],
                "submit_decision_sequence": first["decision_sequence"],
                "absolute_target_price": first["absolute_target_price"],
                "effective_basis_bp": first["effective_basis_bp"],
                "target_offset_ticks": first["target_offset_ticks"],
                "decision_alias_count": group.height,
                "action_alias_count": 0,
                "action_aliases": [],
                "source_asof_date": first["source_asof_date"],
                "parameter_version": first["parameter_version"],
                "lifecycle_policy_version": first["lifecycle_policy_version"],
                "queue_scenario": first["queue_scenario"],
                "intended_maker_quantity": first["intended_maker_quantity"],
                "hedge_quantity": first["hedge_quantity"],
                "enumeration_source": "inclusive_legal_tick_range",
                "execution_safe_snapshot": True,
                "contains_target_day_outcome": False,
            }
        )
    actions = pl.from_dicts(action_rows, infer_schema_length=None).sort(
        [*ACTION_SELECTION_KEYS, "absolute_target_tick"]
    )
    action_ids = actions.select(*ACTION_IDENTITY_KEYS, "action_id")
    decision_actions = expanded.join(
        action_ids,
        on=list(ACTION_IDENTITY_KEYS),
        how="left",
        validate="m:1",
    ).select(
        "action_id",
        *ACTION_ALIAS_DECISION_KEYS,
        "decision_sequence",
        "spread_pair_epoch",
        "stage",
        "absolute_target_tick",
        "absolute_target_price",
        "effective_basis_bp",
        "target_offset_ticks",
        "source_asof_date",
        "parameter_version",
        "lifecycle_policy_version",
        "queue_scenario",
        "intended_maker_quantity",
        "hedge_quantity",
    ).with_columns(
        pl.lit(0).alias("action_alias_count"),
        pl.lit([], dtype=pl.List(pl.String)).alias("action_aliases"),
        pl.lit(True).alias("execution_safe_snapshot"),
        pl.lit(False).alias("contains_target_day_outcome"),
    ).unique(
        subset=[
            *ACTION_ALIAS_DECISION_KEYS,
            "decision_sequence",
            "absolute_target_tick",
        ],
        maintain_order=True,
    ).sort([*ACTION_ALIAS_DECISION_KEYS, "absolute_target_tick"])
    counts = decision_actions.group_by(
        [*ACTION_ALIAS_DECISION_KEYS, "decision_sequence"]
    ).len(name="enumerated_legal_action_count")
    audit = decisions.join(
        counts,
        on=[*ACTION_ALIAS_DECISION_KEYS, "decision_sequence"],
        how="left",
        validate="1:1",
    ).with_columns(
        pl.col("enumerated_legal_action_count").fill_null(0),
        pl.when(pl.col("current_state_eligible") == True)  # noqa: E712
        .then(pl.lit("range_enumerated"))
        .when(pl.col("current_state_eligible") == False)  # noqa: E712
        .then(pl.lit("current_state_ineligible"))
        .otherwise(pl.lit("current_state_unknown"))
        .alias("action_status"),
    )
    return LegalActionSurface(actions, audit, decision_actions)


def build_daily_ev_lookup(
    path_facts: pl.DataFrame,
    sessions: Sequence[str],
    config: EVLookupConfig = EVLookupConfig(),
    *,
    asof_dates: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Build empirical, pathwise EV tables using only labels mature before D.

    One row of ``path_facts`` is one raw admitted quote path, not a boundary
    alias.  A known path has exactly one value from ``TERMINAL_BRANCHES``.
    Censored and unknown rows have a null branch.  EV is the mean of complete
    path cashflows, equivalently the sum of mutually-exclusive branch
    probability times branch conditional mean; no fill/hedge/exit marginals
    are multiplied.  ``filled_cashflow_before_cost_bp`` is computed from the
    actual fill prices, so the separately reported hedge/exit slippage fields
    are diagnostics and are not subtracted a second time.
    """

    config.validate()
    calendar = _normalise_sessions(sessions)
    selected = _normalise_sessions(asof_dates or calendar)
    unknown_dates = sorted(set(selected) - set(calendar))
    if unknown_dates:
        raise ValueError(
            f"asof dates absent from session calendar: {unknown_dates[:5]}"
        )
    facts = _prepare_path_facts(path_facts, config)
    if facts.is_empty():
        return pl.DataFrame()

    calendar_index = {date: index for index, date in enumerate(calendar)}
    outputs: list[pl.DataFrame] = []
    for asof_date in selected:
        target_index = calendar_index[asof_date]
        prior_dates = calendar[
            max(0, target_index - config.lookback_sessions) : target_index
        ]
        if not prior_dates:
            continue
        training = facts.filter(
            pl.col("Date").is_in(prior_dates)
            & (pl.col("label_end_date") < asof_date)
        )
        if training.is_empty():
            continue
        summary = _aggregate_pathwise(
            training, config, window_sessions=len(prior_dates)
        ).with_columns(
            pl.lit(asof_date).alias("asof_date"),
            pl.lit(prior_dates[0]).alias("window_start_date"),
            pl.lit(prior_dates[-1]).alias("window_end_date"),
            pl.lit(len(prior_dates)).alias("window_sessions"),
            pl.lit(config.lookback_sessions).alias("lookback_sessions"),
            pl.lit(config.estimator_version).alias("estimator_version"),
            pl.lit("|".join(config.required_cost_columns)).alias(
                "required_cost_columns"
            ),
            pl.lit(True).alias("execution_safe_snapshot"),
            pl.lit(False).alias("contains_target_day_outcome"),
        )
        outputs.append(summary)
    if not outputs:
        return pl.DataFrame()
    return pl.concat(outputs, how="diagonal_relaxed").sort(
        ["asof_date", *config.lookup_keys]
    )


def score_action_surface(
    actions: pl.DataFrame,
    lookup: pl.DataFrame,
    *,
    lookup_keys: Sequence[str] = DEFAULT_LOOKUP_KEYS,
    state_family: str = "all",
    state_bucket: str = "all",
    min_action_score_bp: float = 0.0,
) -> pl.DataFrame:
    """Attach the D-safe lookup and select the best positive action per decision.

    Pass ``LegalActionSurface.decision_actions`` rather than canonical
    ``actions``: the latter intentionally removes repeated within-epoch
    decisions and therefore cannot be used as a causal decision clock.  This
    is quote admission only.  Position limits, reservations, and existing live
    layers remain the responsibility of the sequential portfolio replay and
    :class:`quote_fill.layered.LayeredSampler`.
    """

    if not math.isfinite(min_action_score_bp):
        raise ValueError("min_action_score_bp must be finite")
    _require(
        actions,
        {
            "action_id",
            *DECISION_SCORE_KEYS,
            "spread_pair_epoch",
            "absolute_target_tick",
            "target_offset_ticks",
        },
        "legal actions",
    )
    _require(
        lookup,
        {
            "asof_date",
            *lookup_keys,
            "ev_ready",
            "ev_status",
            "expected_net_cashflow_bp",
            "action_score_bp",
            "execution_safe_snapshot",
            "contains_target_day_outcome",
        },
        "EV lookup",
    )
    if actions.is_empty():
        return actions
    if lookup.filter(
        ~pl.col("execution_safe_snapshot").fill_null(False)
        | pl.col("contains_target_day_outcome").fill_null(True)
    ).height:
        raise ValueError("EV lookup is not an execution-safe prior snapshot")

    enriched = actions
    if "state_family" in lookup_keys and "state_family" not in enriched.columns:
        enriched = enriched.with_columns(pl.lit(state_family).alias("state_family"))
    if "state_bucket" in lookup_keys and "state_bucket" not in enriched.columns:
        enriched = enriched.with_columns(pl.lit(state_bucket).alias("state_bucket"))
    _require(enriched, set(lookup_keys), "legal actions")

    right_key = ["asof_date", *lookup_keys]
    if lookup.select(right_key).n_unique() != lookup.height:
        raise ValueError("EV lookup contains duplicate action-state keys")
    right = lookup.with_columns(pl.lit(True).alias("_lookup_found"))
    scored = enriched.join(
        right,
        left_on=["Date", *lookup_keys],
        right_on=right_key,
        how="left",
        validate="m:1",
        suffix="_lookup",
    ).with_columns(
        pl.col("_lookup_found").fill_null(False).alias("lookup_found"),
        pl.col("ev_ready").fill_null(False),
        pl.when(pl.col("_lookup_found").fill_null(False))
        .then(pl.col("ev_status"))
        .otherwise(pl.lit("missing_lookup"))
        .alias("ev_status"),
    ).drop("_lookup_found")

    eligible = scored.filter(
        pl.col("ev_ready")
        & pl.col("action_score_bp").is_not_null()
        & (pl.col("action_score_bp") >= min_action_score_bp)
    )
    rank_rows: list[dict[str, object]] = []
    if not eligible.is_empty():
        for _, group in eligible.group_by(DECISION_SCORE_KEYS, maintain_order=True):
            ordered = group.sort(
                [
                    "action_score_bp",
                    "expected_net_cashflow_bp",
                    "target_offset_ticks",
                    "action_id",
                ],
                descending=[True, True, True, False],
            )
            for rank, action_id in enumerate(ordered["action_id"].to_list(), 1):
                rank_rows.append(
                    {
                        "action_id": action_id,
                        **{
                            name: ordered[name][rank - 1]
                            for name in DECISION_SCORE_KEYS
                        },
                        "action_ev_rank": rank,
                        "quote_selected": rank == 1,
                    }
                )
    if rank_rows:
        ranked = pl.from_dicts(rank_rows, infer_schema_length=None)
        scored = scored.join(
            ranked,
            on=["action_id", *DECISION_SCORE_KEYS],
            how="left",
            validate="1:1",
        )
    else:
        scored = scored.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("action_ev_rank"),
            pl.lit(False).alias("quote_selected"),
        )
    return scored.with_columns(
        pl.col("action_ev_rank").cast(pl.Int64),
        pl.col("quote_selected").fill_null(False),
    ).sort([*DECISION_SCORE_KEYS, "absolute_target_tick"])


def _prepare_path_facts(
    path_facts: pl.DataFrame, config: EVLookupConfig
) -> pl.DataFrame:
    required = {
        "path_id",
        "Date",
        "label_end_date",
        *config.lookup_keys,
        "lifecycle_policy_version",
        "queue_scenario",
        "intended_maker_quantity",
        "hedge_quantity",
        "outcome_status",
        "terminal_branch",
        *EXECUTION_STATUS_VALUES,
        "filled_cashflow_before_cost_bp",
        *config.required_cost_columns,
        *SLIPPAGE_DIAGNOSTIC_COLUMNS,
        "capital_time_seconds",
        "cost_profile_version",
    }
    _require(path_facts, required, "terminal path facts")
    if path_facts.is_empty():
        return path_facts
    facts = path_facts.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("label_end_date").cast(pl.String),
        pl.col("path_id").cast(pl.String),
        pl.col("outcome_status").cast(pl.String),
        pl.col("terminal_branch").cast(pl.String),
        pl.col("filled_cashflow_before_cost_bp").cast(pl.Float64, strict=False),
        pl.col("capital_time_seconds").cast(pl.Float64, strict=False),
        *(pl.col(name).cast(pl.String) for name in EXECUTION_STATUS_VALUES),
        *(
            pl.col(name).cast(pl.Float64, strict=False)
            for name in config.required_cost_columns
        ),
        *(
            pl.col(name).cast(pl.Float64, strict=False)
            for name in SLIPPAGE_DIAGNOSTIC_COLUMNS
        ),
    )
    _validate_fact_dates(facts)
    _validate_policy_lineage(facts)
    unique_key = ["path_id", *config.lookup_keys]
    if facts.select(unique_key).n_unique() != facts.height:
        raise ValueError("terminal path facts duplicate a raw path within a lookup state")
    bad_outcome = facts.filter(~pl.col("outcome_status").is_in(OUTCOME_STATUSES))
    if bad_outcome.height:
        raise ValueError("invalid outcome_status in terminal path facts")
    bad_known = facts.filter(
        (pl.col("outcome_status") == "known")
        & ~pl.col("terminal_branch").is_in(TERMINAL_BRANCHES).fill_null(False)
    )
    if bad_known.height:
        raise ValueError("known path must have exactly one valid terminal_branch")
    bad_unresolved = facts.filter(
        (pl.col("outcome_status") != "known")
        & pl.col("terminal_branch").is_not_null()
    )
    if bad_unresolved.height:
        raise ValueError("censored/unknown path must not claim a terminal_branch")
    bad_expiry_branch = facts.filter(
        (
            (pl.col("terminal_branch") == "expiry_forced_flat")
            & (pl.col("expiry_status") != "forced_flat")
        )
        | (
            (pl.col("terminal_branch") == "expiry_settlement")
            & (pl.col("expiry_status") != "settled")
        )
        | (
            (pl.col("expiry_status") == "forced_flat")
            & (pl.col("terminal_branch") != "expiry_forced_flat")
        )
        | (
            (pl.col("expiry_status") == "settled")
            & (pl.col("terminal_branch") != "expiry_settlement")
        )
    )
    if bad_expiry_branch.height:
        raise ValueError("expiry status and terminal branch disagree")
    for column, allowed in EXECUTION_STATUS_VALUES.items():
        invalid = facts.filter(~pl.col(column).is_in(allowed).fill_null(False))
        if invalid.height:
            raise ValueError(f"invalid or null {column} in terminal path facts")
    for column in config.required_cost_columns:
        invalid_cost = facts.filter(
            pl.col(column).is_not_null()
            & (~pl.col(column).is_finite() | (pl.col(column) < 0))
        )
        if invalid_cost.height:
            raise ValueError(f"{column} must be null or finite and non-negative")
    for column in SLIPPAGE_DIAGNOSTIC_COLUMNS:
        invalid_slippage = facts.filter(
            pl.col(column).is_not_null() & ~pl.col(column).is_finite()
        )
        if invalid_slippage.height:
            raise ValueError(f"{column} must be null or finite")
    invalid_capital = facts.filter(
        pl.col("capital_time_seconds").is_not_null()
        & (
            ~pl.col("capital_time_seconds").is_finite()
            | (pl.col("capital_time_seconds") < 0)
        )
    )
    if invalid_capital.height:
        raise ValueError("capital_time_seconds must be null or finite and non-negative")

    known = pl.col("outcome_status") == "known"
    costs_complete = pl.all_horizontal(
        pl.col("filled_cashflow_before_cost_bp").is_not_null()
        & pl.col("filled_cashflow_before_cost_bp").is_finite(),
        pl.col("capital_time_seconds").is_not_null()
        & pl.col("capital_time_seconds").is_finite(),
        pl.col("cost_profile_version").is_not_null()
        & (pl.col("cost_profile_version").cast(pl.String).str.len_chars() > 0),
        *(
            pl.col(name).is_not_null() & pl.col(name).is_finite()
            for name in config.required_cost_columns
        ),
    )
    execution_complete = pl.all_horizontal(
        *(pl.col(name) != "unknown" for name in EXECUTION_STATUS_VALUES)
    )
    hedge_slippage_required = pl.col("hedge_50ms_status").is_in(
        ["executable", "partial"]
    )
    exit_slippage_required = pl.col("terminal_branch").is_in(
        [
            "same_day_target_exit",
            "same_day_aggressive_exit",
            "overnight_exit",
            "expiry_forced_flat",
            "emergency_exit",
        ]
    )
    slippage_complete = pl.all_horizontal(
        ~hedge_slippage_required
        | (
            pl.col("hedge_slippage_bp_50ms").is_not_null()
            & pl.col("hedge_slippage_bp_50ms").is_finite()
        ),
        ~exit_slippage_required
        | (
            pl.col("exit_slippage_bp").is_not_null()
            & pl.col("exit_slippage_bp").is_finite()
        ),
    )
    complete = known & costs_complete & execution_complete & slippage_complete
    total_cost = pl.sum_horizontal(
        *(pl.col(name) for name in config.required_cost_columns)
    )
    return facts.with_columns(
        known.alias("_outcome_known"),
        (pl.col("outcome_status") == "censored").alias("_outcome_censored"),
        (pl.col("outcome_status") == "unknown").alias("_outcome_unknown"),
        (known & costs_complete).alias("_cost_complete_known"),
        (known & execution_complete).alias("_execution_complete_known"),
        (known & slippage_complete).alias("_slippage_complete_known"),
        complete.alias("_ev_complete_known"),
        pl.when(complete)
        .then(pl.col("filled_cashflow_before_cost_bp") - total_cost)
        .otherwise(None)
        .alias("_path_net_cashflow_bp"),
    ).with_columns(
        pl.when(pl.col("_ev_complete_known"))
        .then((-pl.col("_path_net_cashflow_bp")).clip(lower_bound=0.0))
        .otherwise(None)
        .alias("_path_downside_bp")
    )


def _aggregate_pathwise(
    training: pl.DataFrame,
    config: EVLookupConfig,
    *,
    window_sessions: int,
) -> pl.DataFrame:
    known = pl.col("_outcome_known")
    ev_complete = pl.col("_ev_complete_known")
    expressions: list[pl.Expr] = [
        pl.len().alias("n_paths"),
        pl.col("Date").n_unique().alias("n_training_sessions"),
        pl.col("Date").min().alias("train_start_date"),
        pl.col("Date").max().alias("train_end_date"),
        pl.col("label_end_date").max().alias("label_cutoff_date"),
        known.sum().alias("n_outcome_known"),
        pl.col("_outcome_censored").sum().alias("n_outcome_censored"),
        pl.col("_outcome_unknown").sum().alias("n_outcome_unknown"),
        pl.col("_cost_complete_known").sum().alias("n_cost_complete_known"),
        pl.col("_execution_complete_known")
        .sum()
        .alias("n_execution_complete_known"),
        pl.col("_slippage_complete_known")
        .sum()
        .alias("n_slippage_complete_known"),
        ev_complete.sum().alias("n_ev_complete_known"),
        pl.col("_path_net_cashflow_bp")
        .filter(ev_complete)
        .sum()
        .alias("_sum_path_net_cashflow_bp"),
        pl.col("capital_time_seconds")
        .filter(ev_complete)
        .sum()
        .alias("_sum_capital_time_seconds"),
        pl.col("_path_downside_bp")
        .filter(ev_complete)
        .sum()
        .alias("_sum_path_downside_bp"),
        pl.col("cost_profile_version")
        .filter(known & pl.col("cost_profile_version").is_not_null())
        .n_unique()
        .alias("n_cost_profile_versions"),
        (pl.col("cancel_status") != "unknown").sum().alias("n_cancel_observed"),
        (pl.col("cancel_status") == "unknown").sum().alias("n_cancel_unknown"),
        (pl.col("cancel_status") == "cancelled").sum().alias("n_cancelled"),
        (pl.col("entry_fill_status") != "unknown")
        .sum()
        .alias("n_entry_fill_observed"),
        (pl.col("entry_fill_status") == "unknown")
        .sum()
        .alias("n_entry_fill_unknown"),
        (pl.col("entry_fill_status") == "no_fill").sum().alias("n_no_fill"),
        (pl.col("entry_fill_status") == "partial")
        .sum()
        .alias("n_partial_fill"),
        (pl.col("entry_fill_status") == "full").sum().alias("n_full_fill"),
        pl.col("topup_status")
        .is_in(["completed", "failed", "pending"])
        .sum()
        .alias("n_topup_observed"),
        (pl.col("topup_status") == "unknown").sum().alias("n_topup_unknown"),
        (pl.col("topup_status") == "completed").sum().alias("n_topup_completed"),
        pl.col("hedge_50ms_status")
        .is_in(["executable", "partial", "failed"])
        .sum()
        .alias("n_hedge_50ms_observed"),
        (pl.col("hedge_50ms_status") == "unknown")
        .sum()
        .alias("n_hedge_50ms_unknown"),
        (pl.col("hedge_50ms_status") == "executable")
        .sum()
        .alias("n_hedge_50ms_executable"),
        (pl.col("hedge_50ms_status") == "partial")
        .sum()
        .alias("n_hedge_50ms_partial"),
        pl.col("hedge_slippage_bp_50ms")
        .is_finite()
        .fill_null(False)
        .sum()
        .alias("n_hedge_slippage_observed"),
        pl.col("hedge_slippage_bp_50ms")
        .filter(pl.col("hedge_slippage_bp_50ms").is_finite())
        .mean()
        .alias("mean_hedge_slippage_bp_50ms"),
        pl.col("same_day_exit_status")
        .is_in(["target_exit", "aggressive_exit", "no_exit", "failed"])
        .sum()
        .alias("n_same_day_exit_observed"),
        (pl.col("same_day_exit_status") == "unknown")
        .sum()
        .alias("n_same_day_exit_unknown"),
        (pl.col("same_day_exit_status") == "target_exit")
        .sum()
        .alias("n_same_day_target_exit"),
        (pl.col("same_day_exit_status") == "aggressive_exit")
        .sum()
        .alias("n_same_day_aggressive_exit"),
        pl.col("exit_slippage_bp")
        .is_finite()
        .fill_null(False)
        .sum()
        .alias("n_exit_slippage_observed"),
        pl.col("exit_slippage_bp")
        .filter(pl.col("exit_slippage_bp").is_finite())
        .mean()
        .alias("mean_exit_slippage_bp"),
        (pl.col("overnight_status") != "unknown")
        .sum()
        .alias("n_overnight_status_observed"),
        (pl.col("overnight_status") == "unknown")
        .sum()
        .alias("n_overnight_unknown"),
        pl.col("overnight_status")
        .is_in(["carried_open", "closed", "unresolved"])
        .sum()
        .alias("n_overnight_paths"),
        (pl.col("expiry_status") != "unknown")
        .sum()
        .alias("n_expiry_status_observed"),
        (pl.col("expiry_status") == "unknown")
        .sum()
        .alias("n_expiry_status_unknown"),
        (pl.col("expiry_status") == "forced_flat")
        .sum()
        .alias("n_expiry_forced_flat"),
        (pl.col("expiry_status") == "settled")
        .sum()
        .alias("n_expiry_settlement"),
        (pl.col("expiry_status") == "unresolved")
        .sum()
        .alias("n_expiry_unresolved"),
        (pl.col("emergency_status") != "unknown")
        .sum()
        .alias("n_emergency_observed"),
        (pl.col("emergency_status") == "unknown")
        .sum()
        .alias("n_emergency_unknown"),
        (pl.col("emergency_status") == "triggered")
        .sum()
        .alias("n_emergency_triggered"),
    ]
    for branch in TERMINAL_BRANCHES:
        in_branch = known & (pl.col("terminal_branch") == branch)
        expressions.extend(
            [
                in_branch.sum().alias(f"n_branch_{branch}"),
                pl.col("_path_net_cashflow_bp")
                .filter(in_branch & ev_complete)
                .mean()
                .alias(f"mean_net_cashflow_bp_branch_{branch}"),
            ]
        )
    result = training.group_by(list(config.lookup_keys)).agg(*expressions)

    result = result.with_columns(
        _ratio("n_outcome_known", "n_paths", "outcome_label_coverage"),
        _ratio("n_outcome_censored", "n_paths", "censor_rate"),
        _ratio("n_outcome_unknown", "n_paths", "unknown_rate"),
        _ratio(
            "n_cost_complete_known",
            "n_outcome_known",
            "cost_coverage_given_known",
        ),
        _ratio(
            "n_execution_complete_known",
            "n_outcome_known",
            "execution_field_coverage_given_known",
        ),
        _ratio(
            "n_slippage_complete_known",
            "n_outcome_known",
            "slippage_diagnostic_coverage_given_known",
        ),
        _ratio("n_cancelled", "n_cancel_observed", "cancel_rate_given_observed"),
        _ratio(
            "n_partial_fill",
            "n_entry_fill_observed",
            "partial_fill_rate_given_observed",
        ),
        _ratio(
            "n_full_fill",
            "n_entry_fill_observed",
            "full_fill_rate_given_observed",
        ),
        _ratio(
            "n_topup_completed",
            "n_topup_observed",
            "topup_complete_rate_given_observed",
        ),
        _ratio(
            "n_hedge_50ms_executable",
            "n_hedge_50ms_observed",
            "hedge_50ms_executable_rate_given_observed",
        ),
        (
            pl.when(pl.col("n_same_day_exit_observed") > 0)
            .then(
                (
                    pl.col("n_same_day_target_exit")
                    + pl.col("n_same_day_aggressive_exit")
                )
                / pl.col("n_same_day_exit_observed")
            )
            .otherwise(None)
            .alias("same_day_exit_rate_given_observed")
        ),
        _ratio(
            "n_overnight_paths",
            "n_overnight_status_observed",
            "overnight_path_rate_given_status_observed",
        ),
        (
            pl.when(pl.col("n_expiry_status_observed") > 0)
            .then(
                (
                    pl.col("n_expiry_forced_flat")
                    + pl.col("n_expiry_settlement")
                    + pl.col("n_expiry_unresolved")
                )
                / pl.col("n_expiry_status_observed")
            )
            .otherwise(None)
            .alias("expiry_path_rate_given_status_observed")
        ),
        _ratio(
            "n_emergency_triggered",
            "n_emergency_observed",
            "emergency_rate_given_observed",
        ),
        pl.when(
            (pl.col("n_paths") > 0)
            & (pl.col("n_outcome_known") == pl.col("n_paths"))
            & (pl.col("n_ev_complete_known") == pl.col("n_paths"))
        )
        .then(
            pl.col("_sum_path_net_cashflow_bp") / pl.col("n_paths")
        )
        .otherwise(None)
        .alias("expected_net_cashflow_bp"),
        pl.when(
            (pl.col("n_paths") > 0)
            & (
                pl.col("n_ev_complete_known") == pl.col("n_outcome_known")
            )
        )
        .then(pl.col("_sum_path_net_cashflow_bp") / pl.col("n_paths"))
        .otherwise(None)
        .alias("known_cashflow_contribution_per_admitted_quote_bp"),
        pl.when(
            (pl.col("n_paths") > 0)
            & (pl.col("n_outcome_known") == pl.col("n_paths"))
            & (pl.col("n_ev_complete_known") == pl.col("n_paths"))
        )
        .then(
            pl.col("_sum_capital_time_seconds") / pl.col("n_paths")
        )
        .otherwise(None)
        .alias("expected_capital_time_seconds"),
        pl.when(
            (pl.col("n_paths") > 0)
            & (pl.col("n_outcome_known") == pl.col("n_paths"))
            & (pl.col("n_ev_complete_known") == pl.col("n_paths"))
        )
        .then(pl.col("_sum_path_downside_bp") / pl.col("n_paths"))
        .otherwise(None)
        .alias("expected_downside_bp"),
        *(
            _ratio(
                f"n_branch_{branch}",
                "n_paths",
                f"p_branch_{branch}",
            )
            for branch in TERMINAL_BRANCHES
        ),
        _ratio("n_outcome_censored", "n_paths", "p_outcome_censored"),
        _ratio("n_outcome_unknown", "n_paths", "p_outcome_unknown"),
    )
    probability_columns = [f"p_branch_{branch}" for branch in TERMINAL_BRANCHES]
    contribution_columns = [
        f"ev_contribution_bp_branch_{branch}" for branch in TERMINAL_BRANCHES
    ]
    result = result.with_columns(
        *(
            pl.when(pl.col(f"n_branch_{branch}") > 0)
            .then(
                pl.col(f"p_branch_{branch}")
                * pl.col(f"mean_net_cashflow_bp_branch_{branch}")
            )
            .otherwise(0.0)
            .alias(f"ev_contribution_bp_branch_{branch}")
            for branch in TERMINAL_BRANCHES
        )
    )
    result = result.with_columns(
        pl.when(pl.col("n_paths") > 0)
        .then(pl.sum_horizontal(*(pl.col(name) for name in probability_columns)))
        .otherwise(None)
        .alias("branch_probability_sum"),
        pl.when(pl.col("n_paths") > 0)
        .then(
            pl.sum_horizontal(*(pl.col(name) for name in probability_columns))
            + pl.col("p_outcome_censored")
            + pl.col("p_outcome_unknown")
        )
        .otherwise(None)
        .alias("admitted_path_probability_sum"),
        (pl.col("p_outcome_censored") + pl.col("p_outcome_unknown")).alias(
            "unpriced_terminal_mass"
        ),
        pl.when(
            (pl.col("n_outcome_known") == pl.col("n_paths"))
            & (pl.col("n_ev_complete_known") == pl.col("n_paths"))
        )
        .then(
            pl.sum_horizontal(*(pl.col(name) for name in contribution_columns))
        )
        .otherwise(None)
        .alias("branch_reconstructed_ev_bp"),
        (
            config.capital_charge_bp_per_hour
            * pl.col("expected_capital_time_seconds")
            / 3600.0
        ).alias("capital_time_penalty_bp"),
        (
            config.downside_penalty_weight * pl.col("expected_downside_bp")
        ).alias("downside_penalty_bp"),
        (
            config.emergency_penalty_bp_per_event
            * pl.col("emergency_rate_given_observed")
        ).alias("emergency_risk_penalty_bp"),
    ).with_columns(
        (
            pl.col("expected_net_cashflow_bp")
            - pl.col("capital_time_penalty_bp")
            - pl.col("downside_penalty_bp")
            - pl.col("emergency_risk_penalty_bp")
        ).alias("action_score_bp")
    )

    support = (
        pl.lit(window_sessions >= config.min_history_sessions)
        & (pl.col("n_training_sessions") >= config.min_group_sessions)
        & (pl.col("n_outcome_known") >= config.min_known_paths)
        & (pl.col("n_outcome_known") == pl.col("n_paths"))
        & (
            pl.col("outcome_label_coverage")
            >= config.min_outcome_label_coverage
        )
        & (pl.col("censor_rate") <= config.max_censor_rate)
        & (pl.col("unknown_rate") <= config.max_unknown_rate)
        & (pl.col("n_cost_complete_known") == pl.col("n_outcome_known"))
        & (
            pl.col("n_execution_complete_known") == pl.col("n_outcome_known")
        )
        & (pl.col("n_slippage_complete_known") == pl.col("n_outcome_known"))
        & (pl.col("n_ev_complete_known") == pl.col("n_outcome_known"))
        & (pl.col("branch_probability_sum") - 1.0).abs().le(1e-12)
        & (pl.col("admitted_path_probability_sum") - 1.0).abs().le(1e-12)
        & pl.col("expected_net_cashflow_bp").is_not_null()
    )
    result = result.with_columns(
        support.alias("_group_support_ready"),
        pl.lit(window_sessions >= config.min_history_sessions).alias(
            "history_support_gate"
        ),
        pl.lit(False).alias("hierarchical_fallback_included"),
        pl.lit("none_product_state_only").alias("fallback_level"),
    ).with_columns(
        pl.col("_group_support_ready").alias("ev_ready"),
        pl.when(~pl.col("history_support_gate"))
        .then(pl.lit("insufficient_history_sessions"))
        .when(pl.col("n_training_sessions") < config.min_group_sessions)
        .then(pl.lit("insufficient_group_sessions"))
        .when(pl.col("n_outcome_known") < config.min_known_paths)
        .then(pl.lit("insufficient_known_paths"))
        .when(pl.col("n_outcome_known") != pl.col("n_paths"))
        .then(pl.lit("unpriced_censored_or_unknown"))
        .when(
            pl.col("outcome_label_coverage")
            < config.min_outcome_label_coverage
        )
        .then(pl.lit("insufficient_label_coverage"))
        .when(pl.col("censor_rate") > config.max_censor_rate)
        .then(pl.lit("excess_censor_rate"))
        .when(pl.col("unknown_rate") > config.max_unknown_rate)
        .then(pl.lit("excess_unknown_rate"))
        .when(pl.col("n_cost_complete_known") != pl.col("n_outcome_known"))
        .then(pl.lit("incomplete_cost_inputs"))
        .when(
            pl.col("n_execution_complete_known") != pl.col("n_outcome_known")
        )
        .then(pl.lit("incomplete_execution_fields"))
        .when(
            pl.col("n_slippage_complete_known") != pl.col("n_outcome_known")
        )
        .then(pl.lit("incomplete_slippage_diagnostics"))
        .when((pl.col("branch_probability_sum") - 1.0).abs() > 1e-12)
        .then(pl.lit("non_exhaustive_terminal_branches"))
        .otherwise(pl.lit("ready"))
        .alias("ev_status"),
    ).drop(
        "_group_support_ready",
        "_sum_path_net_cashflow_bp",
        "_sum_capital_time_seconds",
        "_sum_path_downside_bp",
    )
    return result


def _ratio(numerator: str, denominator: str, output: str) -> pl.Expr:
    return (
        pl.when(pl.col(denominator) > 0)
        .then(pl.col(numerator) / pl.col(denominator))
        .otherwise(None)
        .alias(output)
    )


def _maker_same_side_bbo(row: Mapping[str, object], route: str) -> float:
    spec = ROUTE_SPECS[route]
    column = f"{'fut' if spec.maker_market == 'future' else 'spot'}_{spec.maker_side}"
    return _finite_positive(row.get(column), column)


def _action_id(key: Mapping[str, object]) -> str:
    identity = "|".join(
        str(key[name]) for name in ACTION_IDENTITY_KEYS
    )
    return hashlib.sha1(identity.encode("utf-8")).hexdigest()[:20]


def _validate_prior_lineage(frame: pl.DataFrame) -> None:
    parsed = frame.select(
        pl.col("Date").cast(pl.String).alias("Date"),
        pl.col("source_asof_date").cast(pl.String).alias("source_asof_date"),
        pl.col("contains_target_day_outcome"),
    )
    for row in parsed.iter_rows(named=True):
        target = _parse_date(row["Date"], "Date")
        source = _parse_date(row["source_asof_date"], "source_asof_date")
        if source >= target:
            raise ValueError("source_asof_date must be strictly before Date")
        if row["contains_target_day_outcome"] is not False:
            raise ValueError("action candidate contains target-day outcome")


def _validate_fact_dates(frame: pl.DataFrame) -> None:
    for row in frame.select("Date", "label_end_date").iter_rows(named=True):
        start = _parse_date(row["Date"], "Date")
        end = _parse_date(row["label_end_date"], "label_end_date")
        if end < start:
            raise ValueError("label_end_date must not be before path Date")


def _validate_policy_lineage(frame: pl.DataFrame) -> None:
    required = {
        "lifecycle_policy_version",
        "queue_scenario",
        "intended_maker_quantity",
        "hedge_quantity",
    }
    _require(frame, required, "policy lineage")
    for row in frame.select(*sorted(required)).iter_rows(named=True):
        for name in ("lifecycle_policy_version", "queue_scenario"):
            value = row[name]
            if value is None or not str(value).strip():
                raise ValueError(f"{name} must be present")
        if _non_negative_integer(
            row["intended_maker_quantity"], "intended_maker_quantity"
        ) <= 0:
            raise ValueError("intended_maker_quantity must be positive")
        if _non_negative_integer(row["hedge_quantity"], "hedge_quantity") <= 0:
            raise ValueError("hedge_quantity must be positive")


def _normalise_sessions(values: Iterable[str]) -> list[str]:
    result = sorted({str(value) for value in values})
    if not result:
        raise ValueError("session calendar must not be empty")
    for value in result:
        _parse_date(value, "session date")
    return result


def _parse_date(value: object, name: str) -> datetime:
    text = str(value)
    if len(text) != 8:
        raise ValueError(f"{name} must be valid YYYYMMDD")
    try:
        return datetime.strptime(text, "%Y%m%d")
    except ValueError as error:
        raise ValueError(f"{name} must be valid YYYYMMDD") from error


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_positive(value: object, name: str) -> float:
    result = _finite_number(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _non_negative_integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        result = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a non-negative integer") from error
    if result < 0 or str(result) != str(value):
        # Accept integer-valued numeric scalars such as numpy.int64, but reject
        # lossy float/string coercion.
        try:
            if float(value) != result:  # type: ignore[arg-type]
                raise ValueError
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a non-negative integer") from error
    return result


def _optional_number(value: object) -> float | None:
    if value is None:
        return None
    result = _finite_number(value, "market quote")
    return result


def _require(frame: pl.DataFrame, columns: set[str], source: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")
