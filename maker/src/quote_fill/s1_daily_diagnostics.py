"""Compact, resume-safe diagnostics for one S1 policy/date replay.

The chronological loop is intentionally discarded after each partition.  This
module extracts the distribution samples and exact counters needed by the S1
comparison report before that happens.  Accounting facts remain the sole PnL
source; these diagnostics never synthesize cashflow or terminal outcomes.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from .capacity_ledger import BUCKET_NAMES, CapacityLedger
from .s1_entry_day_runner import S1EntryDayRun
from .s1_event_loop import RiskEvent
from .s1_hedge import HEDGE_DELAY_NS

DIAGNOSTICS_SCHEMA_VERSION: Final = "s1_daily_diagnostics_v2_economic_gate"
_UNRESOLVED_STATES: Final = frozenset(
    {
        "entry_hedge_timeout_unresolved",
        "exit_rollback_failed_unresolved",
    }
)
_TOP_LEVEL_KEYS: Final = frozenset(
    {
        "schema_version",
        "date",
        "policy_id",
        "target_rank_counts",
        "makerfill_outcome_counts",
        "economic_gate_decision_status_counts",
        "economic_gate_actual_send_status_counts",
        "economic_gate_dispatch_outcome_counts",
        "economic_gate_sent_expected_margin_bp",
        "economic_gate_sent_modeled_cost_twd",
        "candidate_terminal_counts",
        "order_terminal_counts",
        "admission_status_counts",
        "request_counts",
        "execution_role_counts",
        "position_state_counts",
        "active_fill_count",
        "active_fill_latency_ms",
        "exit_physical_fill_count",
        "exit_physical_fill_shares",
        "exit_physical_fill_reason_counts",
        "risk_groups",
        "global_committed_peak_twd",
        "global_committed_end_twd",
        "carry_out_positions",
        "carry_out_notional_twd",
        "naked_unresolved_positions",
        "naked_unresolved_notional_twd",
    }
)


class S1DailyDiagnosticsError(ValueError):
    """One replay cannot be reduced to a valid diagnostic record."""


@dataclass(frozen=True, slots=True)
class _RiskLifecycle:
    created: RiskEvent
    terminal: RiskEvent
    evaluations: tuple[RiskEvent, ...]


def build_s1_daily_diagnostics(
    run: S1EntryDayRun,
    ledger: CapacityLedger,
) -> dict[str, object]:
    """Return one canonical JSON-safe diagnostic object.

    The input ledger must be the exact ledger used by ``run``.  Naked terminal
    states are reported here but are rejected by the production orchestrator;
    paired carry is kept separate so ``open`` never silently means both.
    """

    if not isinstance(run, S1EntryDayRun):
        raise TypeError("run must be an S1EntryDayRun")
    if not isinstance(ledger, CapacityLedger):
        raise TypeError("ledger must be a CapacityLedger")
    result = run.result
    if result.capacity_transitions != ledger.transitions:
        raise S1DailyDiagnosticsError("run and ledger transition deltas differ")
    ledger.verify()

    orders_by_raw_id = {order.raw_order_fact_id: order for order in result.orders}
    if len(orders_by_raw_id) != len(result.orders):
        raise S1DailyDiagnosticsError("entry raw order IDs are duplicated")
    active_fills = tuple(
        event for event in result.fill_events if event.status == "filled"
    )
    fill_latency_ms: list[float] = []
    for event in active_fills:
        try:
            order = orders_by_raw_id[event.raw_order_fact_id]
        except KeyError as error:
            raise S1DailyDiagnosticsError(
                "active fill has no sent entry order"
            ) from error
        delay_ns = event.cursor.recv_time_ns - order.actual_start_cursor.recv_time_ns
        if delay_ns < 0:
            raise S1DailyDiagnosticsError("entry fill precedes actual new")
        fill_latency_ms.append(delay_ns / 1_000_000.0)

    risk_groups = _risk_group_records(result.risk_events, result.executions)
    state_counts = Counter(position.state for position in result.positions)
    naked = tuple(
        position
        for position in result.positions
        if position.state in _UNRESOLVED_STATES
    )
    naked_capacity_ids = {position.capacity_id for position in naked}
    carry_capacity_ids = {position.capacity_id for position in result.carry_out}
    if naked_capacity_ids & carry_capacity_ids:
        raise S1DailyDiagnosticsError("naked and paired carry capacity IDs overlap")

    peak_by_bucket = {
        bucket: max(
            (
                getattr(transition.global_after, bucket)
                for transition in result.capacity_transitions
            ),
            default=getattr(ledger.global_balances, bucket),
        )
        for bucket in BUCKET_NAMES
    }
    peak_by_bucket["total_committed_notional_twd"] = max(
        (
            transition.global_after.total_committed_notional_twd
            for transition in result.capacity_transitions
        ),
        default=ledger.global_balances.total_committed_notional_twd,
    )
    end_by_bucket = {
        **{bucket: getattr(ledger.global_balances, bucket) for bucket in BUCKET_NAMES},
        "total_committed_notional_twd": (
            ledger.global_balances.total_committed_notional_twd
        ),
    }
    physical_fill_reasons = Counter(
        fill.fill_reason for fill in result.exit_physical_fills
    )
    decision_estimates = tuple(
        value
        for value in run.economic_gate_estimates
        if value.evaluation_stage == "decision_observation"
    )
    actual_estimates = tuple(
        value
        for value in run.economic_gate_estimates
        if value.evaluation_stage == "actual_send_refresh"
    )
    sent_audits = tuple(
        value
        for value in run.economic_gate_audits
        if value.dispatch_outcome == "sent"
    )
    if len(actual_estimates) != len(run.economic_gate_audits):
        raise S1DailyDiagnosticsError(
            "actual-send economic estimate/audit counts differ"
        )
    if len(sent_audits) != len(result.orders):
        raise S1DailyDiagnosticsError("sent economic audits differ from orders")
    record: dict[str, object] = {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "date": run.summary.date,
        "policy_id": run.summary.policy_id,
        "target_rank_counts": _counter_record(
            Counter(
                event.exact_target_rank
                if event.exact_target_rank is not None
                else "not_BID1_or_BID2"
                for event in run.makerfill_assessments
            )
        ),
        "makerfill_outcome_counts": _counter_record(
            Counter(
                event.potential_outcome_status for event in run.makerfill_assessments
            )
        ),
        "economic_gate_decision_status_counts": _counter_record(
            Counter(value.status for value in decision_estimates)
        ),
        "economic_gate_actual_send_status_counts": _counter_record(
            Counter(value.status for value in actual_estimates)
        ),
        "economic_gate_dispatch_outcome_counts": _counter_record(
            Counter(value.dispatch_outcome for value in run.economic_gate_audits)
        ),
        "economic_gate_sent_expected_margin_bp": sorted(
            float(value.estimate.selected_expected_margin_bp)
            for value in sent_audits
            if value.estimate.selected_expected_margin_bp is not None
        ),
        "economic_gate_sent_modeled_cost_twd": sorted(
            float(cost)
            for value in sent_audits
            if (
                cost := (
                    value.estimate.same_day_modeled_cost_twd
                    if value.estimate.cost_horizon == "same_day"
                    else value.estimate.overnight_modeled_cost_twd
                    if value.estimate.cost_horizon == "overnight"
                    else None
                )
            )
            is not None
        ),
        "candidate_terminal_counts": _counter_record(
            Counter(event.status for event in result.candidate_intent_audit)
        ),
        "order_terminal_counts": _counter_record(
            Counter(event.event_type for event in result.order_events)
        ),
        "admission_status_counts": _counter_record(
            Counter(event.status for event in result.admission_events)
        ),
        "request_counts": _counter_record(
            Counter(
                f"{event.stage}:{event.venue}:{event.request_class}:{event.event_type}"
                for event in result.request_events
            )
        ),
        "execution_role_counts": _counter_record(
            Counter(execution.role for execution in result.executions)
        ),
        "position_state_counts": _counter_record(state_counts),
        "active_fill_count": len(active_fills),
        "active_fill_latency_ms": sorted(fill_latency_ms),
        "exit_physical_fill_count": len(result.exit_physical_fills),
        "exit_physical_fill_shares": sum(
            fill.fill_shares for fill in result.exit_physical_fills
        ),
        "exit_physical_fill_reason_counts": _counter_record(physical_fill_reasons),
        "risk_groups": risk_groups,
        "global_committed_peak_twd": peak_by_bucket,
        "global_committed_end_twd": end_by_bucket,
        "carry_out_positions": len(result.carry_out),
        "carry_out_notional_twd": _capacity_notional(ledger, carry_capacity_ids),
        "naked_unresolved_positions": len(naked),
        "naked_unresolved_notional_twd": _capacity_notional(ledger, naked_capacity_ids),
    }
    return validate_s1_daily_diagnostics(record)


def validate_s1_daily_diagnostics(record: Mapping[str, object]) -> dict[str, object]:
    """Strictly validate and copy one decoded diagnostic JSON object."""

    if not isinstance(record, Mapping) or set(record) != _TOP_LEVEL_KEYS:
        raise S1DailyDiagnosticsError("daily diagnostics schema mismatch")
    result = dict(record)
    if result["schema_version"] != DIAGNOSTICS_SCHEMA_VERSION:
        raise S1DailyDiagnosticsError("daily diagnostics version mismatch")
    _yyyymmdd(result["date"], "date")
    _identifier(result["policy_id"], "policy_id")
    for name in (
        "target_rank_counts",
        "makerfill_outcome_counts",
        "economic_gate_decision_status_counts",
        "economic_gate_actual_send_status_counts",
        "economic_gate_dispatch_outcome_counts",
        "candidate_terminal_counts",
        "order_terminal_counts",
        "admission_status_counts",
        "request_counts",
        "execution_role_counts",
        "position_state_counts",
        "exit_physical_fill_reason_counts",
    ):
        result[name] = _validated_counter(result[name], name)
    for name in (
        "economic_gate_sent_expected_margin_bp",
        "economic_gate_sent_modeled_cost_twd",
    ):
        result[name] = _float_samples(result[name], name)
    for name in (
        "active_fill_count",
        "exit_physical_fill_count",
        "exit_physical_fill_shares",
        "carry_out_positions",
        "carry_out_notional_twd",
        "naked_unresolved_positions",
        "naked_unresolved_notional_twd",
    ):
        _nonnegative_int(result[name], name)
    result["active_fill_latency_ms"] = _float_samples(
        result["active_fill_latency_ms"], "active_fill_latency_ms"
    )
    if result["active_fill_count"] != len(result["active_fill_latency_ms"]):
        raise S1DailyDiagnosticsError("active fill count/sample mismatch")
    result["global_committed_peak_twd"] = _bucket_record(
        result["global_committed_peak_twd"],
        "global_committed_peak_twd",
        simultaneous=False,
    )
    result["global_committed_end_twd"] = _bucket_record(
        result["global_committed_end_twd"],
        "global_committed_end_twd",
        simultaneous=True,
    )
    if not isinstance(result["risk_groups"], list):
        raise S1DailyDiagnosticsError("risk_groups must be a JSON array")
    groups = [
        _validated_risk_group(value, index)
        for index, value in enumerate(result["risk_groups"])
    ]
    keys = [(group["stage"], group["risk_kind"]) for group in groups]
    if keys != sorted(set(keys)):
        raise S1DailyDiagnosticsError("risk groups must be unique and sorted")
    result["risk_groups"] = groups
    return result


def _risk_group_records(
    events: Sequence[RiskEvent],
    executions: Sequence[object],
) -> list[dict[str, object]]:
    lifecycles = _risk_lifecycles(events)
    execution_by_request: dict[str, object] = {}
    for execution in executions:
        request_id = getattr(execution, "request_id", None)
        role = getattr(execution, "role", None)
        if request_id is None or role not in (
            "entry_hedge",
            "entry_rollback",
            "exit_hedge",
            "exit_rollback",
        ):
            continue
        if request_id in execution_by_request:
            raise S1DailyDiagnosticsError("risk request has multiple executions")
        execution_by_request[request_id] = execution

    grouped: dict[tuple[str, str], list[_RiskLifecycle]] = {}
    for lifecycle in lifecycles:
        key = (lifecycle.created.stage, lifecycle.created.risk_kind)
        grouped.setdefault(key, []).append(lifecycle)
    records: list[dict[str, object]] = []
    for (stage, risk_kind), rows in sorted(grouped.items()):
        delays: list[float] = []
        slips: list[float] = []
        actual_send = 0
        timeout = 0
        arrival_available = 0
        gate_reasons: Counter[str] = Counter()
        for lifecycle in rows:
            created = lifecycle.created
            arrival_available += int(created.arrival_reference.available)
            first_gate = next(
                (
                    event.gate_reason
                    for event in lifecycle.evaluations
                    if event.gate_reason is not None
                ),
                None,
            )
            gate_reasons[first_gate or "none"] += 1
            if lifecycle.terminal.event_type == "timeout":
                timeout += 1
                continue
            actual_send += 1
            target_ns = created.arrival_reference.reference_cursor.recv_time_ns
            if risk_kind == "hedge":
                target_ns += HEDGE_DELAY_NS
            delay_ns = lifecycle.terminal.cursor.recv_time_ns - target_ns
            if delay_ns < 0:
                raise S1DailyDiagnosticsError("risk actual send precedes target")
            delays.append(delay_ns / 1_000_000.0)
            try:
                execution = execution_by_request[created.request_id]
            except KeyError as error:
                raise S1DailyDiagnosticsError(
                    "risk actual send has no matching execution"
                ) from error
            slip = created.arrival_reference.adverse_slippage_bp(execution.price)
            if slip is not None:
                slips.append(slip)
        records.append(
            {
                "stage": stage,
                "risk_kind": risk_kind,
                "created": len(rows),
                "actual_send": actual_send,
                "timeout": timeout,
                "on_time": sum(delay == 0.0 for delay in delays),
                "delayed": sum(delay > 0.0 for delay in delays),
                "delay_ms": sorted(delays),
                "arrival_reference_available": arrival_available,
                "arrival_reference_denominator": len(rows),
                "adverse_slippage_bp": sorted(slips),
                "initial_gate_reason_counts": _counter_record(gate_reasons),
            }
        )
    unmatched = set(execution_by_request).difference(
        lifecycle.created.request_id for lifecycle in lifecycles
    )
    if unmatched:
        raise S1DailyDiagnosticsError("risk execution has no lifecycle")
    return records


def _risk_lifecycles(events: Sequence[RiskEvent]) -> tuple[_RiskLifecycle, ...]:
    by_id: dict[str, list[RiskEvent]] = {}
    last_cursor = None
    for event in events:
        if not isinstance(event, RiskEvent):
            raise TypeError("risk events must contain RiskEvent values")
        if last_cursor is not None and event.cursor < last_cursor:
            raise S1DailyDiagnosticsError("risk events move backwards")
        last_cursor = event.cursor
        by_id.setdefault(event.risk_id, []).append(event)
    result: list[_RiskLifecycle] = []
    for risk_id, rows in sorted(by_id.items()):
        created = tuple(event for event in rows if event.event_type == "created")
        terminal = tuple(
            event for event in rows if event.event_type in ("actual_send", "timeout")
        )
        if len(created) != 1 or len(terminal) != 1 or rows[0] != created[0]:
            raise S1DailyDiagnosticsError(
                f"risk lifecycle is incomplete or unordered: {risk_id}"
            )
        identity = (
            created[0].request_id,
            created[0].position_id,
            created[0].product_id,
            created[0].risk_kind,
            created[0].stage,
        )
        if any(
            (
                event.request_id,
                event.position_id,
                event.product_id,
                event.risk_kind,
                event.stage,
            )
            != identity
            for event in rows
        ):
            raise S1DailyDiagnosticsError("risk lifecycle identity drifted")
        result.append(
            _RiskLifecycle(
                created=created[0],
                terminal=terminal[0],
                evaluations=tuple(
                    event for event in rows if event.event_type == "evaluated"
                ),
            )
        )
    return tuple(result)


def _validated_risk_group(value: object, index: int) -> dict[str, object]:
    keys = {
        "stage",
        "risk_kind",
        "created",
        "actual_send",
        "timeout",
        "on_time",
        "delayed",
        "delay_ms",
        "arrival_reference_available",
        "arrival_reference_denominator",
        "adverse_slippage_bp",
        "initial_gate_reason_counts",
    }
    if not isinstance(value, Mapping) or set(value) != keys:
        raise S1DailyDiagnosticsError(f"risk_groups[{index}] schema mismatch")
    result = dict(value)
    if result["stage"] not in ("entry", "exit"):
        raise S1DailyDiagnosticsError("risk group stage is invalid")
    if result["risk_kind"] not in ("hedge", "rollback"):
        raise S1DailyDiagnosticsError("risk group kind is invalid")
    for name in (
        "created",
        "actual_send",
        "timeout",
        "on_time",
        "delayed",
        "arrival_reference_available",
        "arrival_reference_denominator",
    ):
        _nonnegative_int(result[name], name)
    if result["created"] != result["actual_send"] + result["timeout"]:
        raise S1DailyDiagnosticsError("risk terminal counts do not close")
    if result["actual_send"] != result["on_time"] + result["delayed"]:
        raise S1DailyDiagnosticsError("risk send-delay counts do not close")
    if result["created"] != result["arrival_reference_denominator"]:
        raise S1DailyDiagnosticsError("risk arrival denominator does not close")
    if result["arrival_reference_available"] > result["created"]:
        raise S1DailyDiagnosticsError("risk arrival numerator exceeds denominator")
    result["delay_ms"] = _float_samples(result["delay_ms"], "delay_ms")
    result["adverse_slippage_bp"] = _float_samples(
        result["adverse_slippage_bp"], "adverse_slippage_bp"
    )
    if len(result["delay_ms"]) != result["actual_send"]:
        raise S1DailyDiagnosticsError("risk delay sample count mismatch")
    result["initial_gate_reason_counts"] = _validated_counter(
        result["initial_gate_reason_counts"], "initial_gate_reason_counts"
    )
    return result


def _capacity_notional(ledger: CapacityLedger, capacity_ids: set[str]) -> int:
    return sum(
        ledger.account_balances(capacity_id).total_committed_notional_twd
        for capacity_id in capacity_ids
    )


def _counter_record(counter: Counter[str]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def _validated_counter(value: object, name: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise S1DailyDiagnosticsError(f"{name} must be a JSON object")
    result: dict[str, int] = {}
    for key, count in value.items():
        _identifier(key, f"{name} key")
        result[str(key)] = _nonnegative_int(count, f"{name}.{key}")
    if list(result) != sorted(result):
        raise S1DailyDiagnosticsError(f"{name} keys must be sorted")
    return result


def _bucket_record(
    value: object,
    name: str,
    *,
    simultaneous: bool,
) -> dict[str, int]:
    expected = {*BUCKET_NAMES, "total_committed_notional_twd"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise S1DailyDiagnosticsError(f"{name} bucket schema mismatch")
    result = {key: _nonnegative_int(value[key], f"{name}.{key}") for key in expected}
    bucket_sum = sum(result[bucket] for bucket in BUCKET_NAMES)
    total = result["total_committed_notional_twd"]
    if simultaneous and total != bucket_sum:
        raise S1DailyDiagnosticsError(f"{name} bucket total mismatch")
    if not simultaneous and (
        total > bucket_sum or total < max(result[bucket] for bucket in BUCKET_NAMES)
    ):
        raise S1DailyDiagnosticsError(f"{name} peak envelope is inconsistent")
    return {key: result[key] for key in sorted(result)}


def _float_samples(value: object, name: str) -> list[float]:
    if not isinstance(value, list):
        raise S1DailyDiagnosticsError(f"{name} must be a JSON array")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise S1DailyDiagnosticsError(f"{name} must contain numbers")
        number = float(item)
        if not math.isfinite(number):
            raise S1DailyDiagnosticsError(f"{name} contains a non-finite number")
        result.append(number)
    if result != sorted(result):
        raise S1DailyDiagnosticsError(f"{name} samples must be sorted")
    return result


def _nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise S1DailyDiagnosticsError(f"{name} must be a nonnegative integer")
    return value


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise S1DailyDiagnosticsError(f"{name} must be a canonical string")
    return value


def _yyyymmdd(value: object, name: str) -> str:
    result = _identifier(value, name)
    if len(result) != 8 or not result.isdigit():
        raise S1DailyDiagnosticsError(f"{name} must be YYYYMMDD")
    return result


__all__ = [
    "DIAGNOSTICS_SCHEMA_VERSION",
    "S1DailyDiagnosticsError",
    "build_s1_daily_diagnostics",
    "validate_s1_daily_diagnostics",
]
