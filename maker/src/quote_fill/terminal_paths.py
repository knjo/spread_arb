"""Honest bridge from execution observations to terminal-path EV facts.

``execution_action_facts`` and ``exit_facts`` are not, by themselves, a
complete PnL path.  This adapter preserves that distinction while producing
the schema consumed by :func:`quote_fill.ev_surface.build_daily_ev_lookup`.

The current raw replay can establish only one economic terminal outcome
without an additional execution model: a full maker entry, an executable
50 ms hedge, and a same-day raw taker/taker close.  A position still open at
EOD is right-censored.  Partial entries, an unfinished hedge, an unknown fill,
and a requested-but-unacknowledged cancel remain unknown.  In particular,
``cancel_required`` is never silently converted into ``cancelled``.

The adapter deliberately leaves every cost component, the cost-profile
version, and exit-slippage diagnostic null.  It can therefore build an audit
lookup, but its output cannot make an EV cell ready.  A later enrichment must
attach a versioned account/date cost profile, an independently defined exit
slippage diagnostic, and terminal overnight/expiry paths.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import polars as pl

from .ev_surface import DEFAULT_COST_COLUMNS
from .hedge_study import (
    FUTURE_ASK_ROUTE,
    FUTURE_HEDGE_CONTRACTS,
    SPOT_BID_ROUTE,
    SPOT_HEDGE_LOTS,
)


EXECUTION_PATH_LOOKUP_KEYS_V1: tuple[str, ...] = (
    "ValueCode",
    "route",
    "lookup_action_id",
    "parameter_version",
    "exit_rule_id",
    "state_family",
    "state_bucket",
    "lifecycle_policy_version",
    "queue_scenario",
    "intended_maker_quantity",
    "hedge_quantity",
)

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
    "nominal_stop_reason",
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

_EXIT_REQUIRED = {
    "Date",
    "ValueCode",
    "QuoteCode",
    "route",
    "raw_order_fact_id",
    "policy_generation_id",
    "exit_rule_id",
    "exit_rule_source_asof_date",
    "branch_status",
    "terminal_outcome",
    "needs_next_session_label",
    "exit_decision_time_ns",
    "gross_cycle_pnl_twd",
}

_EXIT_BRANCHES = frozenset(
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

_IDENTITY_COLUMNS = (
    "Date",
    "ValueCode",
    "QuoteCode",
    "route",
    "raw_order_fact_id",
)


@dataclass(frozen=True)
class TerminalPathAdapterConfig:
    """Frozen lineage for the observation-to-path mapping.

    ``nominal_cancel_model_version=None`` is the safe default: a requested
    cancel has no exchange ACK or cancel-race replay and remains unknown.  A
    caller may opt into the current instantaneous nominal-stop simulation by
    naming that model explicitly.  The model name is appended to the
    lifecycle key so observed and simulated cancellation semantics cannot be
    pooled accidentally.
    """

    lifecycle_policy_version: str
    queue_scenario: str
    nominal_cancel_model_version: str | None = None
    state_family: str = "execution_policy_alias_v1"
    mapping_version: str = "execution_terminal_path_adapter_v1"

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
        if self.nominal_cancel_model_version is not None and (
            not isinstance(self.nominal_cancel_model_version, str)
            or not self.nominal_cancel_model_version.strip()
        ):
            raise ValueError(
                "nominal_cancel_model_version must be None or a non-empty string"
            )


def build_execution_terminal_paths_v1(
    action_facts: pl.DataFrame,
    exit_facts: pl.DataFrame,
    config: TerminalPathAdapterConfig,
) -> pl.DataFrame:
    """Map current execution observations to EV terminal-path facts.

    The unit is one ``policy_generation_id x exit_rule_id`` joint policy
    path.  It is intentionally *not* deduplicated by ``raw_order_fact_id``:
    entry aliases and exit rules are alternative policies.  The physical raw
    id is retained so dependence and shared-volume allocation remain visible.

    Use :data:`EXECUTION_PATH_LOOKUP_KEYS_V1` as ``EVLookupConfig.lookup_keys``.
    This bridge evaluates the current q-alias runner; it does not manufacture
    ``target_offset_ticks`` that are absent from the artifacts and is not a
    substitute for the future legal-tick replay.
    """

    config.validate()
    if action_facts.is_empty() and exit_facts.is_empty():
        return pl.DataFrame()
    if action_facts.is_empty() or exit_facts.is_empty():
        raise ValueError(
            "action and exit facts must both be present; an unassigned exit "
            "policy is not a terminal path"
        )
    _require(action_facts, _ACTION_REQUIRED, "execution action facts")
    _require(exit_facts, _EXIT_REQUIRED, "exit facts")
    if (
        action_facts.select("policy_generation_id").n_unique()
        != action_facts.height
    ):
        raise ValueError("action facts must be unique by policy_generation_id")
    exit_key = ["policy_generation_id", "exit_rule_id"]
    if exit_facts.select(exit_key).n_unique() != exit_facts.height:
        raise ValueError("exit facts must be unique by policy generation and rule")
    invalid_branch = exit_facts.filter(
        ~pl.col("branch_status").is_in(_EXIT_BRANCHES).fill_null(False)
    )
    if invalid_branch.height:
        raise ValueError("exit facts contain an unsupported branch_status")

    action_ids = set(str(value) for value in action_facts["policy_generation_id"])
    exit_ids = set(str(value) for value in exit_facts["policy_generation_id"])
    if missing := sorted(action_ids - exit_ids):
        raise ValueError(f"actions are missing exit-rule facts: {missing[:5]}")
    if orphan := sorted(exit_ids - action_ids):
        raise ValueError(
            f"exit facts contain unknown policy generations: {orphan[:5]}"
        )

    joined = exit_facts.join(
        action_facts,
        on="policy_generation_id",
        how="left",
        suffix="_action",
        validate="m:1",
    )
    _validate_join_identity(joined)
    _validate_exit_rule_lineage(joined)

    lifecycle_version = config.lifecycle_policy_version
    if config.nominal_cancel_model_version is not None:
        lifecycle_version = (
            f"{lifecycle_version}|nominal_cancel="
            f"{config.nominal_cancel_model_version}"
        )

    records: list[dict[str, object]] = []
    for row in joined.iter_rows(named=True):
        record = _map_one_path(row, config, lifecycle_version)
        records.append(record)
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
        raise ValueError("adapter generated duplicate path_id values")
    return paths.sort(
        [
            "Date",
            "ValueCode",
            "route",
            "lookup_action_id",
            "exit_rule_id",
            "path_id",
        ]
    )


def _map_one_path(
    row: Mapping[str, object],
    config: TerminalPathAdapterConfig,
    lifecycle_version: str,
) -> dict[str, object]:
    date = str(row["Date"])
    route = str(row["route"])
    intended = _positive_integer(
        row.get("intended_quantity"), "intended_quantity"
    )
    hedge_quantity = _hedge_quantity(route)
    expected_maker_quantity = 1 if route == FUTURE_ASK_ROUTE else 2
    if intended != expected_maker_quantity:
        raise ValueError(
            f"unexpected intended maker quantity for {route}: {intended}"
        )
    branch_status = str(row["branch_status"])
    entry_fill_status = _entry_fill_status(row)
    hedge_status = _hedge_status(row, entry_fill_status)
    _validate_execution_branch(
        entry_fill_status=entry_fill_status,
        hedge_status=hedge_status,
        branch_status=branch_status,
    )

    outcome_status = "unknown"
    terminal_branch: str | None = None
    mapping_reason = "unresolved_execution_path"

    if branch_status == "exit_policy_unassigned":
        mapping_reason = "exit_rule_fact_missing_or_policy_unassigned"
    elif (
        entry_fill_status == "full"
        and hedge_status == "executable"
        and branch_status == "same_day_taker_exit"
        and row.get("terminal_outcome") is True
    ):
        outcome_status = "known"
        terminal_branch = "same_day_aggressive_exit"
        mapping_reason = "observed_full_entry_hedge_and_raw_taker_exit"
    elif (
        entry_fill_status == "full"
        and hedge_status == "executable"
        and branch_status == "carry_at_eod"
        and row.get("needs_next_session_label") is True
    ):
        outcome_status = "censored"
        mapping_reason = "position_open_at_eod_requires_overnight_label"
    elif entry_fill_status == "partial":
        mapping_reason = "partial_entry_requires_topup_or_emergency_close"
    elif entry_fill_status == "unknown":
        mapping_reason = "entry_fill_or_queue_state_unknown"
    elif entry_fill_status == "full" and hedge_status != "executable":
        mapping_reason = "hedge_completion_or_emergency_close_unknown"
    elif entry_fill_status == "no_fill":
        if (
            row.get("cancel_required") is True
            and config.nominal_cancel_model_version is not None
        ):
            outcome_status = "known"
            terminal_branch = "no_fill_cancel"
            mapping_reason = "explicit_instantaneous_nominal_cancel_model"
        else:
            mapping_reason = "cancel_ack_and_cancel_race_unobserved"

    state_bucket = "|".join(
        f"{name}={_required_text(row.get(name), name)}"
        for name in (
            "rank_bucket",
            "queue_bucket",
            "tod_bucket",
            "freshness_bucket",
        )
    )
    gross_bp, normalization_notional = _gross_cashflow_bp(
        row, terminal_branch
    )
    capital_time = _capital_time_seconds(row, terminal_branch)
    statuses = _execution_statuses(
        row,
        outcome_status=outcome_status,
        terminal_branch=terminal_branch,
        entry_fill_status=entry_fill_status,
        hedge_status=hedge_status,
        branch_status=branch_status,
    )
    policy_generation = str(row["policy_generation_id"])
    exit_rule_id = str(row["exit_rule_id"])
    path_id = "|".join(
        (
            date,
            str(row["ValueCode"]),
            str(row["QuoteCode"]),
            route,
            policy_generation,
            exit_rule_id,
        )
    )
    return {
        "path_id": path_id,
        "physical_path_id": str(row["raw_order_fact_id"]),
        "policy_generation_id": policy_generation,
        "Date": date,
        # Same-day facts have matured at D.  Censored/unknown rows record only
        # the last observed date; a future enrichment must replace this when
        # an overnight/expiry label matures.
        "label_end_date": date,
        "ValueCode": str(row["ValueCode"]),
        "QuoteCode": str(row["QuoteCode"]),
        "route": route,
        "lookup_action_id": str(row["lookup_action_id"]),
        "parameter_version": str(row["parameter_version"]),
        "exit_rule_id": exit_rule_id,
        "exit_rule_fact_observed": branch_status != "exit_policy_unassigned",
        "state_family": config.state_family,
        "state_bucket": state_bucket,
        "lifecycle_policy_version": lifecycle_version,
        "queue_scenario": config.queue_scenario,
        "intended_maker_quantity": intended,
        "hedge_quantity": hedge_quantity,
        "outcome_status": outcome_status,
        "terminal_branch": terminal_branch,
        **statuses,
        "filled_cashflow_before_cost_bp": gross_bp,
        "hedge_slippage_bp_50ms": _finite_or_none(
            row.get("entry_hedge_signed_total_slippage_bp")
        ),
        # The raw exit facts preserve executed VWAP but no independently
        # defined arrival/reference price.  Threshold overshoot is not an
        # execution-slippage measure, so it must remain missing.
        "exit_slippage_bp": None,
        "capital_time_seconds": capital_time,
        "cost_profile_version": None,
        **{name: None for name in DEFAULT_COST_COLUMNS},
        "normalization_notional_twd": normalization_notional,
        "terminal_mapping_reason": mapping_reason,
        "terminal_mapping_version": config.mapping_version,
        "nominal_cancel_model_version": config.nominal_cancel_model_version,
        "cost_components_complete": False,
        "exit_slippage_diagnostic_complete": (
            terminal_branch == "no_fill_cancel"
        ),
        "overnight_terminal_label_complete": terminal_branch is not None,
        "joint_volume_allocated": bool(row.get("joint_volume_allocated", False)),
        "contains_target_day_outcome": True,
    }


def _execution_statuses(
    row: Mapping[str, object],
    *,
    outcome_status: str,
    terminal_branch: str | None,
    entry_fill_status: str,
    hedge_status: str,
    branch_status: str,
) -> dict[str, str]:
    if terminal_branch == "no_fill_cancel":
        return {
            "cancel_status": "cancelled",
            "entry_fill_status": "no_fill",
            "topup_status": "not_applicable",
            "hedge_50ms_status": "not_applicable",
            "same_day_exit_status": "not_applicable",
            "overnight_status": "not_applicable",
            "expiry_status": "not_applicable",
            "emergency_status": "not_triggered",
        }
    if terminal_branch == "same_day_aggressive_exit":
        return {
            "cancel_status": "not_cancelled",
            "entry_fill_status": "full",
            "topup_status": "not_applicable",
            "hedge_50ms_status": "executable",
            "same_day_exit_status": "aggressive_exit",
            "overnight_status": "not_applicable",
            "expiry_status": "not_applicable",
            "emergency_status": "not_triggered",
        }
    if outcome_status == "censored" and branch_status == "carry_at_eod":
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
        "cancel_status": "unknown",
        "entry_fill_status": entry_fill_status,
        "topup_status": (
            "unknown" if entry_fill_status == "partial" else "not_applicable"
        ),
        "hedge_50ms_status": hedge_status,
        "same_day_exit_status": "unknown",
        "overnight_status": "unknown",
        "expiry_status": "unknown",
        "emergency_status": "unknown",
    }


def _entry_fill_status(row: Mapping[str, object]) -> str:
    if row.get("full_fill") is True:
        return "full"
    if row.get("partial_fill") is True:
        return "partial"
    if row.get("any_fill") is False and row.get("queue_known") is True:
        return "no_fill"
    return "unknown"


def _hedge_status(row: Mapping[str, object], entry_fill_status: str) -> str:
    if entry_fill_status == "no_fill":
        return "not_applicable"
    if entry_fill_status != "full":
        return "unknown"
    status = row.get("entry_hedge_status")
    if status == "executable":
        return "executable"
    if status == "insufficient_depth":
        executed = row.get("entry_hedge_executed_quantity")
        return "partial" if isinstance(executed, int) and executed > 0 else "failed"
    if status is None:
        return "unknown"
    return "failed"


def _validate_execution_branch(
    *, entry_fill_status: str, hedge_status: str, branch_status: str
) -> None:
    if branch_status == "exit_policy_unassigned":
        return
    if entry_fill_status == "no_fill":
        allowed = {"no_entry_fill"}
    elif entry_fill_status == "partial":
        allowed = {"partial_entry_unhedged"}
    elif entry_fill_status == "unknown":
        allowed = {"entry_fill_unknown"}
    elif hedge_status == "executable":
        allowed = {"same_day_taker_exit", "carry_at_eod"}
    else:
        allowed = {"entry_hedge_unpriceable"}
    if branch_status not in allowed:
        raise ValueError(
            "action fill/hedge state disagrees with exit branch_status: "
            f"{entry_fill_status}/{hedge_status}/{branch_status}"
        )


def _gross_cashflow_bp(
    row: Mapping[str, object], terminal_branch: str | None
) -> tuple[float | None, float | None]:
    if terminal_branch == "no_fill_cancel":
        return 0.0, None
    if terminal_branch != "same_day_aggressive_exit":
        return None, None
    gross = _finite_or_none(row.get("gross_cycle_pnl_twd"))
    spot = _finite_or_none(row.get("entry_spot_price"))
    shares = _finite_or_none(row.get("entry_hedge_contract_size_shares"))
    if gross is None or spot is None or shares is None:
        return None, None
    notional = spot * shares
    if notional <= 0:
        return None, None
    return gross / notional * 10_000.0, notional


def _capital_time_seconds(
    row: Mapping[str, object], terminal_branch: str | None
) -> float | None:
    submit = _integer_or_none(row.get("submit_recv_time_ns"))
    if submit is None:
        return None
    if terminal_branch == "same_day_aggressive_exit":
        terminal = _integer_or_none(row.get("exit_decision_time_ns"))
    elif terminal_branch == "no_fill_cancel":
        terminal = _integer_or_none(row.get("terminal_recv_time_ns"))
    else:
        return None
    if terminal is None or terminal < submit:
        return None
    return (terminal - submit) / 1_000_000_000.0


def _hedge_quantity(route: str) -> int:
    if route == FUTURE_ASK_ROUTE:
        return SPOT_HEDGE_LOTS
    if route == SPOT_BID_ROUTE:
        return FUTURE_HEDGE_CONTRACTS
    raise ValueError(f"unsupported entry route: {route}")


def _validate_join_identity(frame: pl.DataFrame) -> None:
    for column in _IDENTITY_COLUMNS:
        action_column = f"{column}_action"
        if action_column not in frame.columns:
            raise ValueError(f"joined facts are missing {action_column}")
        mismatch = frame.filter(
            pl.col(column).cast(pl.String)
            != pl.col(action_column).cast(pl.String)
        )
        if mismatch.height:
            raise ValueError(f"action and exit facts disagree on {column}")


def _validate_exit_rule_lineage(frame: pl.DataFrame) -> None:
    malformed = frame.filter(
        (pl.col("branch_status") != "exit_policy_unassigned")
        & (
            pl.col("exit_rule_source_asof_date").is_null()
            | (
                pl.col("exit_rule_source_asof_date").cast(pl.String)
                >= pl.col("Date").cast(pl.String)
            )
        )
    )
    if malformed.height:
        raise ValueError("exit rule source_asof_date must be strictly before Date")


def _finite_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _integer_or_none(value: object) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _positive_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _required_text(value: object, name: str) -> str:
    if value is None or not str(value).strip():
        raise ValueError(f"{name} must be present")
    return str(value)


def _require(frame: pl.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")
