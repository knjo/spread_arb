"""Causal Spot Ask maker target for the normal S1 lower exit route."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from .layered import EventCursor
from .s1_hedge import CausalBookState, executable_book
from .targets import (
    absolute_price_tick,
    effective_basis_bp,
    is_passive_target,
    price_in_ref_band,
)

ROUTE = "spot_ask_future_taker"
PRICE_EPSILON = 1e-8

ExitTargetLocation = Literal[
    "not_passive",
    "inside_spread",
    "at_ask1",
    "displayed_deeper",
    "undisplayed_deeper",
]


@dataclass(frozen=True, slots=True)
class S1SpotAskTarget:
    date: str
    value_code: str
    quote_code: str
    position_id: str
    scenario_id: str
    observation_cursor: EventCursor
    frozen_exit_threshold_basis_bp: float
    frozen_exit_target_price: float
    frozen_exit_absolute_price_tick: int
    future_buy_vwap: float | None
    target_price: float
    absolute_price_tick: int
    effective_exit_basis_bp: float | None
    passive_target: bool
    target_in_reference_band: bool
    target_location: ExitTargetLocation | None
    initial_queue_ahead_shares: int | None
    queue_observable: bool
    gate_open: bool
    gate_reason: str
    spot_book_cursor: EventCursor | None
    future_book_cursor: EventCursor | None


def build_s1_spot_ask_target(
    *,
    date: str,
    value_code: str,
    quote_code: str,
    position_id: str,
    scenario_id: str,
    observation_cursor: EventCursor,
    frozen_exit_threshold_basis_bp: float,
    frozen_exit_target_price: float,
    frozen_exit_absolute_price_tick: int,
    spot_book: CausalBookState | None,
    future_book: CausalBookState | None,
    future_contracts: int = 1,
) -> S1SpotAskTarget:
    """Observe one exit without moving the position's frozen absolute ask."""

    for name, value in (
        ("date", date),
        ("value_code", value_code),
        ("quote_code", quote_code),
        ("position_id", position_id),
        ("scenario_id", scenario_id),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
    if len(date) != 8 or not date.isdigit():
        raise ValueError("date must be YYYYMMDD")
    if not isinstance(observation_cursor, EventCursor):
        raise TypeError("observation_cursor must be an EventCursor")
    threshold = _finite(
        frozen_exit_threshold_basis_bp,
        "frozen_exit_threshold_basis_bp",
    )
    frozen_target = _positive(
        frozen_exit_target_price,
        "frozen_exit_target_price",
    )
    if (
        isinstance(frozen_exit_absolute_price_tick, bool)
        or not isinstance(frozen_exit_absolute_price_tick, int)
        or frozen_exit_absolute_price_tick < 0
    ):
        raise ValueError("frozen_exit_absolute_price_tick must be non-negative")
    derived_tick = absolute_price_tick(
        frozen_target,
        market="spot",
        session_date=date,
    )
    if derived_tick != frozen_exit_absolute_price_tick:
        raise ValueError("frozen exit target price/tick mismatch")
    if (
        isinstance(future_contracts, bool)
        or not isinstance(future_contracts, int)
        or future_contracts <= 0
    ):
        raise ValueError("future_contracts must be a positive integer")

    spot_reason = _spot_book_reason(spot_book, observation_cursor)
    future_exec = None
    future_reason: str | None = None
    if future_book is not None and future_book.book_cursor.cursor > observation_cursor:
        raise ValueError("future book cannot follow the observation cursor")
    if future_book is None:
        future_reason = "missing_future_book"
    else:
        future_exec, raw_future_reason = executable_book(
            future_book,
            side="buy",
            quantity=future_contracts,
            quantity_unit="future_contracts",
            send_eligible_cursor=observation_cursor,
        )
        if future_exec is None:
            future_reason = f"future_{raw_future_reason or 'not_executable'}"

    effective_basis = (
        None
        if future_exec is None
        else effective_basis_bp(
            ROUTE,
            frozen_target,
            fut_exec_ask=future_exec.executable_vwap,
        )
    )
    passive = False
    in_band = False
    location: ExitTargetLocation | None = None
    queue_ahead: int | None = None
    queue_observable = False
    target_reason: str | None = None
    if spot_reason is None:
        assert spot_book is not None
        passive = is_passive_target(
            ROUTE,
            frozen_target,
            spot_bid=spot_book.bids[0].price,
        )
        reference = spot_book.reference_price
        assert reference is not None
        in_band = price_in_ref_band(frozen_target, float(reference))
        location, queue_ahead, queue_observable = _locate_target(
            frozen_target,
            spot_book,
        )
        if not passive:
            target_reason = "target_not_passive"
        elif not in_band:
            target_reason = "target_outside_reference_band"

    # A passive Spot first leg must not remain exposed while its immediate
    # Future-buy hedge is already known to be unexecutable.  B6 still performs
    # the independent t0/+5s execution decision after an actual maker fill;
    # this is a pre-fill risk gate, not a substitute execution price.
    reason = spot_reason or future_reason or target_reason or "eligible"
    return S1SpotAskTarget(
        date=date,
        value_code=value_code,
        quote_code=quote_code,
        position_id=position_id,
        scenario_id=scenario_id,
        observation_cursor=observation_cursor,
        frozen_exit_threshold_basis_bp=threshold,
        frozen_exit_target_price=frozen_target,
        frozen_exit_absolute_price_tick=frozen_exit_absolute_price_tick,
        future_buy_vwap=(None if future_exec is None else future_exec.executable_vwap),
        target_price=frozen_target,
        absolute_price_tick=frozen_exit_absolute_price_tick,
        effective_exit_basis_bp=effective_basis,
        passive_target=passive,
        target_in_reference_band=in_band,
        target_location=location,
        initial_queue_ahead_shares=queue_ahead,
        queue_observable=queue_observable,
        gate_open=reason == "eligible",
        gate_reason=reason,
        spot_book_cursor=(None if spot_book is None else spot_book.book_cursor.cursor),
        future_book_cursor=(
            None if future_book is None else future_book.book_cursor.cursor
        ),
    )


def _spot_book_reason(
    state: CausalBookState | None,
    observation_cursor: EventCursor,
) -> str | None:
    if state is None:
        return "missing_spot_book"
    if state.book_cursor.cursor > observation_cursor:
        raise ValueError("spot book cannot follow the observation cursor")
    if not state.gate_open:
        return state.gate_reason or "spot_gate_closed"
    reference = state.reference_price
    if (
        reference is None
        or not math.isfinite(float(reference))
        or float(reference) <= 0
    ):
        return "invalid_spot_reference"
    if not state.bids or not state.asks:
        return "empty_spot_book"
    if state.bids[0].price > state.asks[0].price:
        return "crossed_spot_book"
    if not (
        price_in_ref_band(state.bids[0].price, float(reference))
        and price_in_ref_band(state.asks[0].price, float(reference))
    ):
        return "spot_bbo_outside_reference_band"
    return None


def _locate_target(
    target_price: float,
    state: CausalBookState,
) -> tuple[ExitTargetLocation, int | None, bool]:
    best_bid = state.bids[0].price
    best_ask = state.asks[0].price
    if target_price <= best_bid + PRICE_EPSILON:
        return "not_passive", None, False
    if target_price < best_ask - PRICE_EPSILON:
        return "inside_spread", 0, True
    for index, level in enumerate(state.asks):
        if math.isclose(
            target_price,
            level.price,
            rel_tol=0.0,
            abs_tol=PRICE_EPSILON,
        ):
            return (
                "at_ask1" if index == 0 else "displayed_deeper",
                level.quantity,
                True,
            )
    return "undisplayed_deeper", None, False


def _finite(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _positive(value: object, name: str) -> float:
    result = _finite(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


__all__ = ["S1SpotAskTarget", "build_s1_spot_ask_target"]
