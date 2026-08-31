"""Causal actual-send economics for S1 Spot-Bid entry admission.

The calculation is deliberately a dispatch gate, not a realized-PnL model.
It prices both legs of a hypothetical complete cycle from the same causal raw
book snapshot, applies the user's exact modeled fees/taxes, and compares the
result with a pre-registered safety floor.  The later maker fill, +50 ms hedge,
exit queue and terminal accounting remain authoritative for realized results.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Final, Literal

from .layered import EventCursor
from .s1_target import S1SpotBidTarget
from .targets import absolute_price_tick, is_passive_target, price_in_ref_band
from .transaction_costs import TransactionCostProfile

CostHorizon = Literal["ungated", "same_day", "overnight"]
EvaluationStage = Literal["decision_observation", "actual_send_refresh"]
DispatchOutcome = Literal[
    "sent",
    "economic_gate_blocked",
    "not_sent_after_gate_pass",
]
GateStatus = Literal[
    "ungated_priced",
    "eligible",
    "below_floor",
    "route_ineligible",
]
ECONOMIC_GATE_RECORD_VERSION: Final = "s1_economic_gate_event_v2_frozen_exit"
ECONOMIC_GATE_AUDIT_VERSION: Final = "s1_economic_gate_actual_send_audit_v1"


@dataclass(frozen=True, slots=True)
class S1EconomicGateRule:
    """One immutable actual-send cost policy."""

    cost_horizon: CostHorizon
    safety_floor_bp: float | None
    economic_gate_enabled: bool
    deployment_shortlist_eligible: bool

    def __post_init__(self) -> None:
        if self.cost_horizon not in ("ungated", "same_day", "overnight"):
            raise ValueError("unsupported cost_horizon")
        if not isinstance(self.economic_gate_enabled, bool) or not isinstance(
            self.deployment_shortlist_eligible,
            bool,
        ):
            raise TypeError("economic gate flags must be boolean")
        if self.cost_horizon == "ungated":
            if self.economic_gate_enabled or self.safety_floor_bp is not None:
                raise ValueError("ungated control cannot carry a cost floor")
            if self.deployment_shortlist_eligible:
                raise ValueError("ungated control cannot enter deployment shortlist")
            return
        if not self.economic_gate_enabled:
            raise ValueError("priced cost horizon requires an enabled gate")
        floor = _nonnegative_finite(self.safety_floor_bp, "safety_floor_bp")
        object.__setattr__(self, "safety_floor_bp", floor)


@dataclass(frozen=True, slots=True)
class S1EconomicGateEstimate:
    """Auditable snapshot estimate used by one actual-send decision."""

    date: str
    policy_id: str
    product_id: str
    observation_cursor: EventCursor
    evaluation_stage: EvaluationStage
    cost_profile_id: str
    cost_horizon: CostHorizon
    safety_floor_bp: float | None
    economic_gate_enabled: bool
    deployment_shortlist_eligible: bool
    status: GateStatus
    reason: str
    gate_open: bool
    shares: int
    contracts: int
    entry_spot_target: float
    entry_future_sell_vwap: float
    future_buy_vwap: float
    hypothetical_exit_spot_target: float
    frozen_exit_spot_target_tick: int
    normalization_notional_twd: float
    gross_expected_twd: float | None
    spot_round_trip_commission_twd: float | None
    same_day_spot_sell_tax_twd: float | None
    overnight_spot_sell_tax_twd: float | None
    futures_round_trip_tax_twd: float | None
    futures_round_trip_commission_twd: float | None
    same_day_modeled_cost_twd: float | None
    overnight_modeled_cost_twd: float | None
    same_day_expected_margin_twd: float | None
    overnight_expected_margin_twd: float | None
    same_day_expected_margin_bp: float | None
    overnight_expected_margin_bp: float | None
    selected_expected_margin_twd: float | None
    selected_expected_margin_bp: float | None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.date, str)
            or len(self.date) != 8
            or not self.date.isdigit()
        ):
            raise ValueError("date must be YYYYMMDD")
        for name in ("policy_id", "product_id", "cost_profile_id", "reason"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.observation_cursor, EventCursor):
            raise TypeError("observation_cursor must be an EventCursor")
        if self.evaluation_stage not in (
            "decision_observation",
            "actual_send_refresh",
        ):
            raise ValueError("unsupported economic-gate evaluation stage")
        S1EconomicGateRule(
            cost_horizon=self.cost_horizon,
            safety_floor_bp=self.safety_floor_bp,
            economic_gate_enabled=self.economic_gate_enabled,
            deployment_shortlist_eligible=self.deployment_shortlist_eligible,
        )
        if self.status not in (
            "ungated_priced",
            "eligible",
            "below_floor",
            "route_ineligible",
        ):
            raise ValueError("unsupported economic gate status")
        if not isinstance(self.gate_open, bool):
            raise TypeError("gate_open must be boolean")
        if type(self.shares) is not int or self.shares <= 0:
            raise ValueError("shares must be a positive integer")
        if type(self.contracts) is not int or self.contracts <= 0:
            raise ValueError("contracts must be a positive integer")
        for name in (
            "entry_spot_target",
            "entry_future_sell_vwap",
            "future_buy_vwap",
            "hypothetical_exit_spot_target",
            "normalization_notional_twd",
        ):
            _positive_finite(getattr(self, name), name)
        if (
            isinstance(self.frozen_exit_spot_target_tick, bool)
            or not isinstance(self.frozen_exit_spot_target_tick, int)
            or self.frozen_exit_spot_target_tick < 0
        ):
            raise ValueError("frozen_exit_spot_target_tick must be non-negative")
        if self.frozen_exit_spot_target_tick != absolute_price_tick(
            self.hypothetical_exit_spot_target,
            market="spot",
            session_date=self.date,
        ):
            raise ValueError("frozen economic exit target price/tick mismatch")
        nullable_names = (
            "gross_expected_twd",
            "spot_round_trip_commission_twd",
            "same_day_spot_sell_tax_twd",
            "overnight_spot_sell_tax_twd",
            "futures_round_trip_tax_twd",
            "futures_round_trip_commission_twd",
            "same_day_modeled_cost_twd",
            "overnight_modeled_cost_twd",
            "same_day_expected_margin_twd",
            "overnight_expected_margin_twd",
            "same_day_expected_margin_bp",
            "overnight_expected_margin_bp",
            "selected_expected_margin_twd",
            "selected_expected_margin_bp",
        )
        for name in nullable_names:
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be finite or None")
        if self.status == "ungated_priced":
            if self.cost_horizon != "ungated" or not self.gate_open:
                raise ValueError("ungated status/gate fields are inconsistent")
        elif self.status == "eligible":
            if not self.economic_gate_enabled or not self.gate_open:
                raise ValueError("eligible status requires an open enabled gate")
        elif self.gate_open:
            raise ValueError("closed economic-gate status cannot be gate_open")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_record(self) -> dict[str, object]:
        """Return a strict versioned JSON-safe audit record."""

        return {
            "schema_version": ECONOMIC_GATE_RECORD_VERSION,
            **self.to_dict(),
        }

    @classmethod
    def from_record(cls, value: Mapping[str, object]) -> S1EconomicGateEstimate:
        """Decode one exact audit record without accepting silent defaults."""

        field_names = tuple(field.name for field in fields(cls))
        expected = {"schema_version", *field_names}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("economic gate record schema mismatch")
        if value["schema_version"] != ECONOMIC_GATE_RECORD_VERSION:
            raise ValueError("economic gate record version mismatch")
        cursor_record = value["observation_cursor"]
        if not isinstance(cursor_record, Mapping) or set(cursor_record) != {
            "recv_time_ns",
            "event_sequence",
            "row_index",
        }:
            raise ValueError("economic gate cursor schema mismatch")
        cursor = EventCursor(
            cursor_record["recv_time_ns"],
            cursor_record["event_sequence"],
            cursor_record["row_index"],
        )
        payload = {name: value[name] for name in field_names}
        payload["observation_cursor"] = cursor
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class S1EconomicGateAudit:
    """One actual-send refresh joined to its final dispatch outcome."""

    estimate: S1EconomicGateEstimate
    dispatch_outcome: DispatchOutcome
    raw_order_fact_id: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.estimate, S1EconomicGateEstimate):
            raise TypeError("estimate must be an S1EconomicGateEstimate")
        if self.estimate.evaluation_stage != "actual_send_refresh":
            raise ValueError("economic-gate audit requires an actual-send refresh")
        if self.dispatch_outcome not in (
            "sent",
            "economic_gate_blocked",
            "not_sent_after_gate_pass",
        ):
            raise ValueError("unsupported economic-gate dispatch outcome")
        if self.dispatch_outcome == "sent":
            if not self.estimate.gate_open:
                raise ValueError("sent economic-gate audit requires an open gate")
            if (
                not isinstance(self.raw_order_fact_id, str)
                or not self.raw_order_fact_id
            ):
                raise ValueError("sent economic-gate audit requires a raw order ID")
            return
        if self.raw_order_fact_id is not None:
            raise ValueError("non-sent economic-gate audit cannot carry a raw order ID")
        if self.dispatch_outcome == "economic_gate_blocked" and self.estimate.gate_open:
            raise ValueError("economic-gate-blocked audit requires a closed gate")
        if (
            self.dispatch_outcome == "not_sent_after_gate_pass"
            and not self.estimate.gate_open
        ):
            raise ValueError("post-gate non-send audit requires an open gate")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": ECONOMIC_GATE_AUDIT_VERSION,
            "dispatch_outcome": self.dispatch_outcome,
            "raw_order_fact_id": self.raw_order_fact_id,
            "estimate": self.estimate.to_record(),
        }

    @classmethod
    def from_record(cls, value: Mapping[str, object]) -> S1EconomicGateAudit:
        if not isinstance(value, Mapping) or set(value) != {
            "schema_version",
            "dispatch_outcome",
            "raw_order_fact_id",
            "estimate",
        }:
            raise ValueError("economic gate audit schema mismatch")
        if value["schema_version"] != ECONOMIC_GATE_AUDIT_VERSION:
            raise ValueError("economic gate audit version mismatch")
        estimate = value["estimate"]
        if not isinstance(estimate, Mapping):
            raise TypeError("economic gate audit estimate must be an object")
        return cls(
            estimate=S1EconomicGateEstimate.from_record(estimate),
            dispatch_outcome=value["dispatch_outcome"],
            raw_order_fact_id=value["raw_order_fact_id"],
        )


UNGATED_CONTROL_RULE = S1EconomicGateRule(
    cost_horizon="ungated",
    safety_floor_bp=None,
    economic_gate_enabled=False,
    deployment_shortlist_eligible=False,
)


def evaluate_s1_entry_economics(
    target: S1SpotBidTarget,
    *,
    observation_cursor: EventCursor,
    spot_reference_price: float,
    rule: S1EconomicGateRule,
    evaluation_stage: EvaluationStage = "actual_send_refresh",
    cost_profile: TransactionCostProfile | None = None,
) -> S1EconomicGateEstimate:
    """Price the fully frozen actual-send cycle and apply one strict floor."""

    if not isinstance(target, S1SpotBidTarget):
        raise TypeError("target must be an S1SpotBidTarget")
    if not isinstance(observation_cursor, EventCursor):
        raise TypeError("observation_cursor must be an EventCursor")
    if target.actual_new_send_cursor != observation_cursor:
        raise ValueError("economic gate cursor must equal target actual-send cursor")
    if not isinstance(rule, S1EconomicGateRule):
        raise TypeError("rule must be an S1EconomicGateRule")
    if evaluation_stage not in ("decision_observation", "actual_send_refresh"):
        raise ValueError("unsupported economic-gate evaluation stage")
    reference = _positive_finite(spot_reference_price, "spot_reference_price")
    profile = cost_profile or TransactionCostProfile()
    if not isinstance(profile, TransactionCostProfile):
        raise TypeError("cost_profile must be a TransactionCostProfile or None")
    profile.validate()

    base = {
        "date": target.Date,
        "policy_id": target.policy_id,
        "product_id": target.ValueCode,
        "observation_cursor": observation_cursor,
        "evaluation_stage": evaluation_stage,
        "cost_profile_id": profile.profile_id,
        "cost_horizon": rule.cost_horizon,
        "safety_floor_bp": rule.safety_floor_bp,
        "economic_gate_enabled": rule.economic_gate_enabled,
        "deployment_shortlist_eligible": rule.deployment_shortlist_eligible,
        "shares": target.contract_size_shares,
        "contracts": 1,
        "entry_spot_target": target.target_price,
        "entry_future_sell_vwap": target.fut_exec_bid_at_actual_new,
        "future_buy_vwap": target.fut_exec_ask_at_actual_new,
        "hypothetical_exit_spot_target": target.frozen_exit_target_price,
        "frozen_exit_spot_target_tick": (target.frozen_exit_absolute_price_tick),
        "normalization_notional_twd": (
            target.target_price * target.contract_size_shares
        ),
    }
    exit_spot = target.frozen_exit_target_price
    future_buy = target.fut_exec_ask_at_actual_new
    passive_exit = is_passive_target(
        "spot_ask_future_taker",
        exit_spot,
        spot_bid=target.spot_bid_at_actual_new,
    )
    in_band = price_in_ref_band(exit_spot, reference)
    shares = target.contract_size_shares
    gross = shares * (
        (exit_spot - target.target_price)
        + (target.fut_exec_bid_at_actual_new - future_buy)
    )
    same_day = profile.paired_cycle_cost_breakdown(
        entry_spot_price=target.target_price,
        exit_spot_price=exit_spot,
        entry_future_price=target.fut_exec_bid_at_actual_new,
        exit_future_price=future_buy,
        shares=shares,
        contracts=1.0,
        same_day=True,
    )
    overnight = profile.paired_cycle_cost_breakdown(
        entry_spot_price=target.target_price,
        exit_spot_price=exit_spot,
        entry_future_price=target.fut_exec_bid_at_actual_new,
        exit_future_price=future_buy,
        shares=shares,
        contracts=1.0,
        same_day=False,
    )
    normalization = float(base["normalization_notional_twd"])
    same_day_margin = gross - same_day.total_twd
    overnight_margin = gross - overnight.total_twd
    same_day_margin_bp = 10_000.0 * same_day_margin / normalization
    overnight_margin_bp = 10_000.0 * overnight_margin / normalization
    selected_margin: float | None = None
    selected_margin_bp: float | None = None
    if rule.cost_horizon == "same_day":
        selected_margin = same_day_margin
        selected_margin_bp = same_day_margin_bp
    elif rule.cost_horizon == "overnight":
        selected_margin = overnight_margin
        selected_margin_bp = overnight_margin_bp

    route_reason = None
    if not passive_exit:
        route_reason = "frozen_exit_target_not_passive"
    elif not in_band:
        route_reason = "frozen_exit_target_outside_reference_band"
    if not rule.economic_gate_enabled:
        status: GateStatus = "ungated_priced"
        reason = "ungated_control"
        gate_open = True
    elif route_reason is not None:
        status = "route_ineligible"
        reason = route_reason
        gate_open = False
    else:
        assert selected_margin_bp is not None
        assert rule.safety_floor_bp is not None
        gate_open = selected_margin_bp > rule.safety_floor_bp
        status = "eligible" if gate_open else "below_floor"
        reason = "eligible" if gate_open else "expected_margin_not_above_floor"

    return S1EconomicGateEstimate(
        **base,
        status=status,
        reason=reason,
        gate_open=gate_open,
        gross_expected_twd=gross,
        spot_round_trip_commission_twd=(
            same_day.spot_entry_commission_twd + same_day.spot_exit_commission_twd
        ),
        same_day_spot_sell_tax_twd=same_day.spot_exit_tax_twd,
        overnight_spot_sell_tax_twd=overnight.spot_exit_tax_twd,
        futures_round_trip_tax_twd=(
            same_day.futures_entry_tax_twd + same_day.futures_exit_tax_twd
        ),
        futures_round_trip_commission_twd=(
            same_day.futures_entry_commission_twd + same_day.futures_exit_commission_twd
        ),
        same_day_modeled_cost_twd=same_day.total_twd,
        overnight_modeled_cost_twd=overnight.total_twd,
        same_day_expected_margin_twd=same_day_margin,
        overnight_expected_margin_twd=overnight_margin,
        same_day_expected_margin_bp=same_day_margin_bp,
        overnight_expected_margin_bp=overnight_margin_bp,
        selected_expected_margin_twd=selected_margin,
        selected_expected_margin_bp=selected_margin_bp,
    )


def _positive_finite(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _nonnegative_finite(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


__all__ = [
    "ECONOMIC_GATE_AUDIT_VERSION",
    "ECONOMIC_GATE_RECORD_VERSION",
    "UNGATED_CONTROL_RULE",
    "CostHorizon",
    "DispatchOutcome",
    "EvaluationStage",
    "GateStatus",
    "S1EconomicGateAudit",
    "S1EconomicGateEstimate",
    "S1EconomicGateRule",
    "evaluate_s1_entry_economics",
]
