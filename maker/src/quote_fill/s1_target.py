"""Actual-new target freezing for the S1 Spot Bid maker route."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Final, Literal
from zoneinfo import ZoneInfo

from .layered import EventCursor
from .policy_spec import TOD_BUCKETS, PolicySpec
from .targets import (
    ROUTE_SPECS,
    absolute_price_tick,
    effective_basis_bp,
    is_passive_target,
    target_price_for_basis,
)

S1_ROUTE: Final = "spot_bid_future_taker"
SESSION_TIMEZONE: Final = ZoneInfo("Asia/Taipei")
NANOSECONDS_PER_SECOND: Final = 1_000_000_000
PRICE_EPSILON: Final = 1e-8

TargetLocation = Literal["not_passive", "inside_spread", "at_bid1", "below_bid1"]

_TOD_INTERVALS: Final = (
    (300, 3_600, "0905_1000"),
    (3_600, 7_200, "1000_1100"),
    (7_200, 10_800, "1100_1200"),
    (10_800, 14_400, "1200_1300"),
)
if tuple(value[2] for value in _TOD_INTERVALS) != TOD_BUCKETS:
    raise RuntimeError("S1 target TOD intervals disagree with PolicySpec")


@dataclass(frozen=True, slots=True)
class S1SpotBidTarget:
    """One immutable absolute target frozen at the actual new-send cursor."""

    Date: str
    ValueCode: str
    QuoteCode: str
    route: str
    stage: str
    maker_market: str
    maker_side: str
    actual_new_send_cursor: EventCursor
    actual_new_seconds_from_open: int
    entry_tod_bucket: str
    policy_id: str
    policy_kind: str
    anchor_model_id: str
    causal_anchor_basis_bp: float
    upper_distance_bp: float
    lower_distance_bp: float
    entry_threshold_basis_bp: float
    frozen_exit_threshold_basis_bp: float
    fut_exec_bid_at_actual_new: float
    spot_bid_at_actual_new: float
    spot_ask_at_actual_new: float
    target_price: float
    absolute_price_tick: int
    effective_entry_basis_bp: float
    passive_target: bool
    target_location: TargetLocation
    contract_size_shares: int
    reservation_notional_twd: int
    upper_source_id: str
    lower_source_id: str
    upper_source_asof_date: str | None
    lower_source_asof_date: str | None
    combined_source_asof_date: str | None
    boundary_quantile: int | None
    fallback_reason: str | None
    policy_spec_version: str
    contains_target_day_outcome: bool


def actual_send_tod_bucket(date: str, cursor: EventCursor) -> tuple[int, str]:
    """Return the actual-send session second and its frozen S1 TOD bucket."""

    _validate_date(date)
    if not isinstance(cursor, EventCursor):
        raise TypeError("actual new-send cursor must be an EventCursor")
    session_start_ns = _local_time_ns(date, hour=9, minute=0)
    session_second = (cursor.recv_time_ns - session_start_ns) // NANOSECONDS_PER_SECOND
    for lower, upper, bucket in _TOD_INTERVALS:
        if lower <= session_second < upper:
            return int(session_second), bucket
    raise ValueError("actual new-send cursor is outside the S1 entry session")


def build_s1_spot_bid_target(
    spec: PolicySpec,
    *,
    date: str,
    value_code: str,
    quote_code: str,
    route: str,
    actual_new_send_cursor: EventCursor,
    causal_anchor_basis_bp: float,
    fut_exec_bid: float,
    spot_bid: float,
    spot_ask: float,
    contract_size_shares: float,
) -> S1SpotBidTarget:
    """Freeze one Spot Bid entry and exit target at actual new-send time.

    No nominal, fill, or cancellation cursor is accepted.  Consequently the
    returned anchor and absolute price cannot be recomputed later in the
    order lifecycle.
    """

    if not isinstance(spec, PolicySpec):
        raise TypeError("spec must be a PolicySpec")
    _validate_identity(
        spec,
        date=date,
        value_code=value_code,
        quote_code=quote_code,
        route=route,
    )
    seconds_from_open, tod_bucket = actual_send_tod_bucket(date, actual_new_send_cursor)
    if tod_bucket != spec.entry_tod_bucket:
        raise ValueError("actual new-send TOD bucket does not match PolicySpec")

    anchor = _finite(causal_anchor_basis_bp, "causal_anchor_basis_bp")
    future_bid = _positive(fut_exec_bid, "fut_exec_bid")
    current_spot_bid = _positive(spot_bid, "spot_bid")
    current_spot_ask = _positive(spot_ask, "spot_ask")
    if current_spot_bid > current_spot_ask:
        raise ValueError("spot bid cannot exceed spot ask")
    raw_contract_size = _positive(contract_size_shares, "contract_size_shares")
    rounded_contract_size = round(raw_contract_size)
    if not math.isclose(
        raw_contract_size,
        rounded_contract_size,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("contract_size_shares must be an integer share count")
    contract_size = int(rounded_contract_size)
    entry_threshold = anchor + spec.upper_distance_bp
    frozen_exit_threshold = anchor - spec.lower_distance_bp
    if not math.isfinite(entry_threshold) or not math.isfinite(frozen_exit_threshold):
        raise ValueError("derived entry/exit thresholds must be finite")

    target_price = target_price_for_basis(
        S1_ROUTE,
        entry_threshold,
        session_date=date,
        fut_exec_bid=future_bid,
    )
    target_tick = absolute_price_tick(
        target_price,
        market="spot",
        session_date=date,
    )
    effective_basis = effective_basis_bp(
        S1_ROUTE,
        target_price,
        fut_exec_bid=future_bid,
    )
    passive = is_passive_target(
        S1_ROUTE,
        target_price,
        spot_ask=current_spot_ask,
    )
    if not passive:
        target_location: TargetLocation = "not_passive"
    elif target_price > current_spot_bid + PRICE_EPSILON:
        target_location = "inside_spread"
    elif math.isclose(
        target_price,
        current_spot_bid,
        rel_tol=0.0,
        abs_tol=PRICE_EPSILON,
    ):
        target_location = "at_bid1"
    else:
        target_location = "below_bid1"
    reservation = math.ceil(target_price * contract_size)
    if reservation <= 0:
        raise ValueError("reservation notional must be positive")

    route_spec = ROUTE_SPECS[S1_ROUTE]
    return S1SpotBidTarget(
        Date=date,
        ValueCode=value_code,
        QuoteCode=quote_code,
        route=S1_ROUTE,
        stage=route_spec.stage,
        maker_market=route_spec.maker_market,
        maker_side=route_spec.maker_side,
        actual_new_send_cursor=actual_new_send_cursor,
        actual_new_seconds_from_open=seconds_from_open,
        entry_tod_bucket=tod_bucket,
        policy_id=spec.policy_id,
        policy_kind=spec.kind,
        anchor_model_id=spec.anchor_model_id,
        causal_anchor_basis_bp=anchor,
        upper_distance_bp=spec.upper_distance_bp,
        lower_distance_bp=spec.lower_distance_bp,
        entry_threshold_basis_bp=entry_threshold,
        frozen_exit_threshold_basis_bp=frozen_exit_threshold,
        fut_exec_bid_at_actual_new=future_bid,
        spot_bid_at_actual_new=current_spot_bid,
        spot_ask_at_actual_new=current_spot_ask,
        target_price=target_price,
        absolute_price_tick=target_tick,
        effective_entry_basis_bp=effective_basis,
        passive_target=passive,
        target_location=target_location,
        contract_size_shares=contract_size,
        reservation_notional_twd=reservation,
        upper_source_id=spec.upper_source_id,
        lower_source_id=spec.lower_source_id,
        upper_source_asof_date=spec.upper_source_asof_date,
        lower_source_asof_date=spec.lower_source_asof_date,
        combined_source_asof_date=spec.combined_source_asof_date,
        boundary_quantile=spec.boundary_quantile,
        fallback_reason=spec.fallback_reason,
        policy_spec_version=spec.spec_version,
        contains_target_day_outcome=spec.contains_target_day_outcome,
    )


def _validate_identity(
    spec: PolicySpec,
    *,
    date: str,
    value_code: str,
    quote_code: str,
    route: str,
) -> None:
    _validate_date(date)
    for name, value in (
        ("value_code", value_code),
        ("quote_code", quote_code),
        ("route", route),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
    if route != S1_ROUTE:
        raise ValueError("S1 target builder only supports Spot Bid maker entry")
    if (date, value_code, quote_code) != (
        spec.Date,
        spec.ValueCode,
        spec.QuoteCode,
    ):
        raise ValueError("Date/product identity does not match PolicySpec")


@lru_cache(maxsize=None)
def _validate_date(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"\d{8}", value):
        raise ValueError("date must be YYYYMMDD")
    try:
        datetime.strptime(value, "%Y%m%d").replace(tzinfo=SESSION_TIMEZONE)
    except ValueError as error:
        raise ValueError("date must be a valid YYYYMMDD date") from error


@lru_cache(maxsize=None)
def _local_time_ns(date: str, *, hour: int, minute: int) -> int:
    value = datetime.strptime(date, "%Y%m%d").replace(
        hour=hour,
        minute=minute,
        tzinfo=SESSION_TIMEZONE,
    )
    return int(value.timestamp()) * NANOSECONDS_PER_SECOND


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: float, name: str) -> float:
    result = _finite(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


__all__ = [
    "S1_ROUTE",
    "S1SpotBidTarget",
    "actual_send_tod_bucket",
    "build_s1_spot_bid_target",
]
