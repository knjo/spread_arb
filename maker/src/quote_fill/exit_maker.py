"""Pure maker-then-taker replay primitives for closing a paired position.

The entry replay and the immediate taker/taker exit benchmark deliberately do
not answer whether an already-open position can be closed with one passive
leg.  This module fills that narrow gap without loading data frames or making
portfolio decisions.  It supports the two exit routes for a position which is
long spot and short its stock future:

``future_bid_spot_taker``
    Buy the future passively, then sell the corresponding spot lots 50 ms
    after each completed futures fill unit.

``spot_ask_future_taker``
    Sell spot passively, then buy one future 50 ms after every two completed
    spot board lots.  An odd spot-lot fill is retained as explicit residual
    inventory; the replay never invents a fractional futures contract.

Orders remain independent research candidates.  Displayed/traded volume is
not jointly allocated across overlapping generations or policies.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal, Mapping, Sequence

from .engine import (
    LayeredWindowBuildResult,
    TargetObservation,
    build_layered_order_windows,
)
from .hedge import (
    DEFAULT_HEDGE_DELAY_NS,
    HedgeExecutionLabel,
    MakerFillHedgeRequest,
    OppositeBookSnapshot,
    _IndexedOppositeBookSnapshots,
    _index_validated_opposite_snapshots,
    _label_delayed_taker_hedge_indexed,
)
from .indexed_replay import IndexedTradeReplay, QuantityFillLabel
from .layered import EventCursor
from .replay import IndependentOrderWindow
from .targets import (
    ROUTE_SPECS,
    absolute_price_tick,
    effective_basis_bp,
    is_passive_target,
    price_in_ref_band,
    target_price_for_basis,
    tick_index_to_price,
)


ExitMakerRoute = Literal[
    "future_bid_spot_taker",
    "spot_ask_future_taker",
]
ExitBranch = Literal[
    "flat_same_day",
    "no_fill_before_cancel_request",
    "carry_at_eod_cancel_unconfirmed",
    "partial_fill_then_cancel",
    "partial_fill_carry_at_eod",
    "hedge_incomplete_residual",
    "hedge_incomplete_carry_at_eod",
    "fill_unknown_then_cancel",
    "fill_unknown_at_eod",
]
OcoDisposition = Literal[
    "winner",
    "sibling_cancel_required",
    "sibling_full_fill_race",
    "ended_before_winner",
    "not_active_at_winner",
    "no_full_fill_winner",
]

FUTURE_BID_EXIT_ROUTE: ExitMakerRoute = "future_bid_spot_taker"
SPOT_ASK_EXIT_ROUTE: ExitMakerRoute = "spot_ask_future_taker"
SUPPORTED_EXIT_MAKER_ROUTES: tuple[ExitMakerRoute, ...] = (
    FUTURE_BID_EXIT_ROUTE,
    SPOT_ASK_EXIT_ROUTE,
)


@dataclass(frozen=True)
class ExitMakerRouteContract:
    """Quantity and side contract for one maker-exit route."""

    route: ExitMakerRoute
    maker_market: Literal["future", "spot"]
    maker_side: Literal["bid", "ask"]
    opposite_market: Literal["future", "spot"]
    hedge_side: Literal["buy", "sell"]
    maker_units_per_hedge: int
    opposite_units_per_hedge: int


EXIT_MAKER_ROUTE_CONTRACTS: Mapping[
    ExitMakerRoute, ExitMakerRouteContract
] = {
    FUTURE_BID_EXIT_ROUTE: ExitMakerRouteContract(
        route=FUTURE_BID_EXIT_ROUTE,
        maker_market="future",
        maker_side="bid",
        opposite_market="spot",
        hedge_side="sell",
        maker_units_per_hedge=1,
        opposite_units_per_hedge=2,
    ),
    SPOT_ASK_EXIT_ROUTE: ExitMakerRouteContract(
        route=SPOT_ASK_EXIT_ROUTE,
        maker_market="spot",
        maker_side="ask",
        opposite_market="future",
        hedge_side="buy",
        maker_units_per_hedge=2,
        opposite_units_per_hedge=1,
    ),
}


@dataclass(frozen=True)
class PairedExitPosition:
    """One or more matched long-spot/short-future units.

    Quantities use native book units: spot board lots and futures contracts.
    The current research contract is two spot board lots per future contract.
    """

    spot_long_lots: int = 2
    future_short_contracts: int = 1
    spot_lots_per_future_contract: int = 2

    def __post_init__(self) -> None:
        for name, value in (
            ("spot_long_lots", self.spot_long_lots),
            ("future_short_contracts", self.future_short_contracts),
            ("spot_lots_per_future_contract", self.spot_lots_per_future_contract),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            self.spot_long_lots
            != self.future_short_contracts * self.spot_lots_per_future_contract
        ):
            raise ValueError("starting exit position must be exactly paired")
        if self.spot_lots_per_future_contract != 2:
            raise ValueError("maker-exit V0 requires two spot lots per future")


@dataclass(frozen=True)
class ResidualExitPosition:
    """Known position left after the maker fill and delayed taker attempts."""

    spot_long_lots: int
    future_short_contracts: int

    def __post_init__(self) -> None:
        for name, value in (
            ("spot_long_lots", self.spot_long_lots),
            ("future_short_contracts", self.future_short_contracts),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    @property
    def flat(self) -> bool:
        return self.spot_long_lots == 0 and self.future_short_contracts == 0


@dataclass(frozen=True)
class ExitMakerObservation:
    """Auditable legal target derived from one causal BBO observation."""

    route: ExitMakerRoute
    cursor: EventCursor
    spread_pair_epoch: int
    threshold_basis_bp: float
    spot_sell_exec_price: float
    future_buy_exec_price: float
    unclamped_target_price: float
    target_price: float
    absolute_target_tick: int
    effective_basis_bp: float
    passive_clamped: bool
    threshold_already_taker_executable: bool
    passive: bool
    inside_reference_band: bool
    gate_open: bool
    gate_reason: str
    initial_queue_ahead: int | None
    target_rank: str

    def as_target_observation(self) -> TargetObservation:
        return TargetObservation(
            cursor=self.cursor,
            spread_pair_epoch=self.spread_pair_epoch,
            absolute_target_tick=self.absolute_target_tick,
            gate_open=self.gate_open,
            gate_reason=self.gate_reason,
            initial_queue_ahead=self.initial_queue_ahead,
            target_rank=self.target_rank,
            source="exit_maker_legal_target",
        )


@dataclass(frozen=True)
class ExitMakerHedgeAttempt:
    """One 50 ms opposite-leg attempt for completed maker hedge units."""

    generation_id: str
    unit_start: int
    unit_count: int
    maker_fill_cursor: EventCursor
    maker_fill_quantity: int
    hedge_quantity: int
    decision_time_ns: int
    status: str
    execution: HedgeExecutionLabel | None

    @property
    def executed_hedge_quantity(self) -> int:
        return 0 if self.execution is None else self.execution.executed_quantity

    @property
    def hedge_complete(self) -> bool:
        return self.execution is not None and self.execution.hedge_complete


@dataclass(frozen=True)
class ExitMakerOrderOutcome:
    """Independent physical outcome for one maker-exit generation."""

    generation_id: str
    route: ExitMakerRoute
    intended_maker_quantity: int
    quantity_fill: QuantityFillLabel
    hedge_attempts: tuple[ExitMakerHedgeAttempt, ...]
    branch_status: ExitBranch
    cancel_required: bool
    eod_carry: bool
    known_filled_maker_quantity: int | None
    remaining_maker_quantity: int | None
    unhedgeable_maker_quantity: int | None
    residual_position: ResidualExitPosition | None
    cancel_ack_observed: bool = False
    cancel_race_modeled: bool = False
    cancel_model: str = "nominal_instant_cancel_v0"
    joint_volume_allocated: bool = False
    pathwise_ev_ready: bool = False

    @property
    def position_flat(self) -> bool:
        return self.residual_position is not None and self.residual_position.flat

    @property
    def any_fill(self) -> bool | None:
        return self.quantity_fill.any_fill

    @property
    def full_fill(self) -> bool | None:
        return self.quantity_fill.full_fill

    @property
    def partial_fill(self) -> bool:
        return self.quantity_fill.partial_fill


@dataclass(frozen=True)
class ExitMakerOcoMember:
    """One independent fact projected onto a single-position OCO policy."""

    generation_id: str
    route: ExitMakerRoute
    disposition: OcoDisposition
    independent_full_fill_cursor: EventCursor | None
    oco_cancel_request_cursor: EventCursor | None
    independent_full_fill_after_cancel_request: bool
    partial_fill_before_winner: bool
    selected_for_position: bool
    cancel_ack_observed: bool = False
    cancel_race_modeled: bool = False


@dataclass(frozen=True)
class ExitMakerOcoProjection:
    """Earliest-full-fill-wins projection over independent candidates.

    This is an inventory projection, not a cancel execution model.  All
    sibling cancellations are request timestamps only.  Exact ACK and fills
    racing those requests remain unobserved, so ``strict_ev_ready`` is false.
    """

    winner_generation_id: str | None
    winner_full_fill_cursor: EventCursor | None
    members: tuple[ExitMakerOcoMember, ...]
    same_cursor_winner_candidates: tuple[str, ...]
    position_projection_safe: bool
    cancel_ack_observed: bool = False
    cancel_race_modeled: bool = False
    strict_ev_ready: bool = False


def make_exit_maker_observation(
    route: str,
    cursor: EventCursor,
    spread_pair_epoch: int,
    threshold_basis_bp: float,
    *,
    session_date: str | None = None,
    spot_bid: float,
    spot_ask: float,
    future_bid: float,
    future_ask: float,
    spot_sell_exec_price: float,
    future_buy_exec_price: float,
    maker_reference_price: float,
    initial_queue_ahead: int | None,
    raw_gate_open: bool = True,
    raw_gate_reason: str = "raw_gate_closed",
) -> ExitMakerObservation:
    """Derive and gate a conservative legal exit-maker target.

    The target uses the required-quantity opposite-leg executable VWAP, not
    B1/A1, then rounds in the conservative direction.  If that raw target
    crosses, it is clamped to the most aggressive passive legal tick.  BBO is
    used only for passivity and rank.
    """

    contract = _route_contract(route)
    if not isinstance(cursor, EventCursor):
        raise TypeError("cursor must be an EventCursor")
    if (
        isinstance(spread_pair_epoch, bool)
        or not isinstance(spread_pair_epoch, int)
        or spread_pair_epoch < 0
    ):
        raise ValueError("spread_pair_epoch must be a non-negative integer")
    if not math.isfinite(threshold_basis_bp):
        raise ValueError("threshold_basis_bp must be finite")
    if not isinstance(raw_gate_open, bool):
        raise ValueError("raw_gate_open must be boolean")
    if not raw_gate_open and not raw_gate_reason:
        raise ValueError("a closed raw gate requires raw_gate_reason")
    if initial_queue_ahead is not None and (
        isinstance(initial_queue_ahead, bool)
        or not isinstance(initial_queue_ahead, int)
        or initial_queue_ahead < 0
    ):
        raise ValueError("initial_queue_ahead must be non-negative or None")

    bbo = {
        "spot_bid": _positive(spot_bid, "spot_bid"),
        "spot_ask": _positive(spot_ask, "spot_ask"),
        "future_bid": _positive(future_bid, "future_bid"),
        "future_ask": _positive(future_ask, "future_ask"),
    }
    spot_exec = _positive(spot_sell_exec_price, "spot_sell_exec_price")
    future_exec = _positive(future_buy_exec_price, "future_buy_exec_price")
    reference = _positive(maker_reference_price, "maker_reference_price")
    if bbo["spot_bid"] >= bbo["spot_ask"]:
        raise ValueError("spot BBO must satisfy bid < ask")
    if bbo["future_bid"] >= bbo["future_ask"]:
        raise ValueError("future BBO must satisfy bid < ask")

    unclamped_target = target_price_for_basis(
        contract.route,
        float(threshold_basis_bp),
        session_date=session_date,
        spot_bid=spot_exec,
        fut_exec_ask=future_exec,
    )
    target = unclamped_target
    if contract.route == FUTURE_BID_EXIT_ROUTE and target >= bbo["future_ask"]:
        ask_tick = absolute_price_tick(
            bbo["future_ask"],
            market=contract.maker_market,
            session_date=session_date,
        )
        if ask_tick == 0:
            raise ValueError("future ask has no positive passive predecessor tick")
        target = tick_index_to_price(
            ask_tick - 1,
            market=contract.maker_market,
            session_date=session_date,
        )
    elif contract.route == SPOT_ASK_EXIT_ROUTE and target <= bbo["spot_bid"]:
        target = tick_index_to_price(
            absolute_price_tick(
                bbo["spot_bid"],
                market=contract.maker_market,
                session_date=session_date,
            )
            + 1,
            market=contract.maker_market,
            session_date=session_date,
        )
    passive_clamped = not math.isclose(
        target, unclamped_target, rel_tol=0.0, abs_tol=1e-8
    )
    target_tick = absolute_price_tick(
        target,
        market=contract.maker_market,
        session_date=session_date,
    )
    # Round-trip through absolute_price_tick is the legal-ladder assertion.
    passive = is_passive_target(
        contract.route,
        target,
        spot_bid=bbo["spot_bid"],
        fut_exec_ask=bbo["future_ask"],
    )
    if not passive:
        raise AssertionError("passive clamp failed to produce a passive target")
    in_band = price_in_ref_band(target, reference)
    effective = effective_basis_bp(
        contract.route,
        target,
        spot_bid=spot_exec,
        fut_exec_ask=future_exec,
    )
    if effective > float(threshold_basis_bp) + 1e-7:
        raise AssertionError("legal passive target violates exit basis threshold")
    threshold_already_taker_executable = (
        (future_exec / spot_exec - 1.0) * 10_000.0
        <= float(threshold_basis_bp) + 1e-9
    )
    if not raw_gate_open:
        gate_open = False
        gate_reason = raw_gate_reason
    elif not in_band:
        gate_open = False
        gate_reason = "outside_reference_band"
    else:
        gate_open = True
        gate_reason = "open"

    if contract.maker_market == "future":
        maker_bid, maker_ask = bbo["future_bid"], bbo["future_ask"]
    else:
        maker_bid, maker_ask = bbo["spot_bid"], bbo["spot_ask"]
    target_rank = _target_rank(contract.maker_side, target, maker_bid, maker_ask)
    return ExitMakerObservation(
        route=contract.route,
        cursor=cursor,
        spread_pair_epoch=spread_pair_epoch,
        threshold_basis_bp=float(threshold_basis_bp),
        spot_sell_exec_price=spot_exec,
        future_buy_exec_price=future_exec,
        unclamped_target_price=unclamped_target,
        target_price=target,
        absolute_target_tick=target_tick,
        effective_basis_bp=effective,
        passive_clamped=passive_clamped,
        threshold_already_taker_executable=threshold_already_taker_executable,
        passive=passive,
        inside_reference_band=in_band,
        gate_open=gate_open,
        gate_reason=gate_reason,
        initial_queue_ahead=initial_queue_ahead,
        target_rank=target_rank,
    )


def build_exit_maker_order_windows(
    observations: Iterable[ExitMakerObservation],
    *,
    route: str,
    policy_id: str,
    cutoff_cursor: EventCursor,
) -> LayeredWindowBuildResult:
    """Apply SpreadPair admission and retreat cancellation to exit targets."""

    contract = _route_contract(route)
    ordered = tuple(observations)
    for observation in ordered:
        if not isinstance(observation, ExitMakerObservation):
            raise TypeError("observations must contain ExitMakerObservation values")
        if observation.route != contract.route:
            raise ValueError("every observation must match the requested exit route")
    return build_layered_order_windows(
        (observation.as_target_observation() for observation in ordered),
        route=contract.route,
        policy_id=policy_id,
        cutoff_cursor=cutoff_cursor,
    )


def replay_exit_maker_window(
    window: IndependentOrderWindow,
    *,
    route: str,
    maker_replay: IndexedTradeReplay,
    opposite_snapshots: Sequence[OppositeBookSnapshot]
    | Iterable[OppositeBookSnapshot],
    eod_cursor: EventCursor,
    starting_position: PairedExitPosition = PairedExitPosition(),
    hedge_delay_ns: int = DEFAULT_HEDGE_DELAY_NS,
    max_book_age_ns: int | None = None,
) -> ExitMakerOrderOutcome:
    """Replay one independent maker exit and its delayed taker hedge(s).

    A futures maker fill is hedgeable contract by contract.  A spot maker
    fill becomes hedgeable only at each two-lot completion cursor.  Multiple
    hedge units completed by the same raw print are grouped into one depth
    sweep, so a trade-through cannot reuse one displayed opposite book level
    for each unit.
    """

    snapshots = tuple(opposite_snapshots)
    _validate_snapshot_order(snapshots)
    return _replay_exit_maker_window_indexed_snapshots(
        window,
        route=route,
        maker_replay=maker_replay,
        opposite_snapshot_index=_index_validated_opposite_snapshots(snapshots),
        eod_cursor=eod_cursor,
        starting_position=starting_position,
        hedge_delay_ns=hedge_delay_ns,
        max_book_age_ns=max_book_age_ns,
    )


def _replay_exit_maker_window_indexed_snapshots(
    window: IndependentOrderWindow,
    *,
    route: str,
    maker_replay: IndexedTradeReplay,
    opposite_snapshot_index: _IndexedOppositeBookSnapshots,
    eod_cursor: EventCursor,
    starting_position: PairedExitPosition,
    hedge_delay_ns: int,
    max_book_age_ns: int | None,
) -> ExitMakerOrderOutcome:
    """Replay one window after the batch has validated snapshot ordering."""

    contract = _route_contract(route)
    if not isinstance(window, IndependentOrderWindow):
        raise TypeError("window must be an IndependentOrderWindow")
    if not isinstance(maker_replay, IndexedTradeReplay):
        raise TypeError("maker_replay must be an IndexedTradeReplay")
    if not isinstance(eod_cursor, EventCursor):
        raise TypeError("eod_cursor must be an EventCursor")
    if window.maker_side != contract.maker_side:
        raise ValueError("window maker side does not match exit route")
    if window.stop_cursor > eod_cursor:
        raise ValueError("window cannot extend beyond eod_cursor")
    if (
        isinstance(hedge_delay_ns, bool)
        or not isinstance(hedge_delay_ns, int)
        or hedge_delay_ns < 0
    ):
        raise ValueError("hedge_delay_ns must be a non-negative integer")
    if max_book_age_ns is not None and (
        isinstance(max_book_age_ns, bool)
        or not isinstance(max_book_age_ns, int)
        or max_book_age_ns < 0
    ):
        raise ValueError("max_book_age_ns must be non-negative or None")
    intended = (
        starting_position.future_short_contracts
        if contract.maker_market == "future"
        else starting_position.spot_long_lots
    )
    quantity_fill = maker_replay.label_quantity(window, intended)
    known_filled = quantity_fill.known_filled_quantity_before_stop
    attempts: list[ExitMakerHedgeAttempt] = []
    if known_filled is not None:
        for unit_start, unit_count, fill_cursor in _hedge_completion_groups(
            maker_replay,
            window,
            known_filled,
            contract.maker_units_per_hedge,
        ):
            maker_quantity = unit_count * contract.maker_units_per_hedge
            hedge_quantity = unit_count * contract.opposite_units_per_hedge
            decision_ns = fill_cursor.recv_time_ns + hedge_delay_ns
            attempt_id = f"{window.generation_id}/exit-hedge-{unit_start}"
            if decision_ns > eod_cursor.recv_time_ns:
                attempts.append(
                    ExitMakerHedgeAttempt(
                        generation_id=attempt_id,
                        unit_start=unit_start,
                        unit_count=unit_count,
                        maker_fill_cursor=fill_cursor,
                        maker_fill_quantity=maker_quantity,
                        hedge_quantity=hedge_quantity,
                        decision_time_ns=decision_ns,
                        status="decision_after_eod",
                        execution=None,
                    )
                )
                continue
            execution = _label_delayed_taker_hedge_indexed(
                MakerFillHedgeRequest(
                    generation_id=attempt_id,
                    fill_cursor=fill_cursor,
                    maker_fill_quantity=maker_quantity,
                    hedge_side=contract.hedge_side,
                    hedge_quantity=hedge_quantity,
                    delay_ns=hedge_delay_ns,
                ),
                opposite_snapshot_index,
                max_book_age_ns=max_book_age_ns,
            )
            attempts.append(
                ExitMakerHedgeAttempt(
                    generation_id=attempt_id,
                    unit_start=unit_start,
                    unit_count=unit_count,
                    maker_fill_cursor=fill_cursor,
                    maker_fill_quantity=maker_quantity,
                    hedge_quantity=hedge_quantity,
                    decision_time_ns=decision_ns,
                    status=execution.status,
                    execution=execution,
                )
            )

    residual = _residual_position(
        contract,
        starting_position,
        known_filled,
        tuple(attempts),
    )
    at_eod = window.stop_reason == "session_cutoff"
    branch = _branch_status(quantity_fill, residual, at_eod)
    unhedgeable = (
        None
        if known_filled is None
        else known_filled % contract.maker_units_per_hedge
    )
    return ExitMakerOrderOutcome(
        generation_id=window.generation_id,
        route=contract.route,
        intended_maker_quantity=intended,
        quantity_fill=quantity_fill,
        hedge_attempts=tuple(attempts),
        branch_status=branch,
        cancel_required=quantity_fill.full_fill is not True,
        eod_carry=at_eod and (residual is None or not residual.flat),
        known_filled_maker_quantity=known_filled,
        remaining_maker_quantity=(
            None if known_filled is None else intended - known_filled
        ),
        unhedgeable_maker_quantity=unhedgeable,
        residual_position=residual,
    )


def replay_exit_maker_windows(
    windows: Iterable[IndependentOrderWindow],
    *,
    route: str,
    maker_replay: IndexedTradeReplay,
    opposite_snapshots: Sequence[OppositeBookSnapshot]
    | Iterable[OppositeBookSnapshot],
    eod_cursor: EventCursor,
    starting_position: PairedExitPosition = PairedExitPosition(),
    hedge_delay_ns: int = DEFAULT_HEDGE_DELAY_NS,
    max_book_age_ns: int | None = None,
) -> tuple[ExitMakerOrderOutcome, ...]:
    """Batch wrapper which reuses one maker-trade index and opposite tape."""

    snapshot_index = _index_exit_maker_opposite_snapshots(opposite_snapshots)
    return _replay_exit_maker_windows_indexed(
        windows,
        route=route,
        maker_replay=maker_replay,
        opposite_snapshot_index=snapshot_index,
        eod_cursor=eod_cursor,
        starting_position=starting_position,
        hedge_delay_ns=hedge_delay_ns,
        max_book_age_ns=max_book_age_ns,
    )


def _index_exit_maker_opposite_snapshots(
    opposite_snapshots: Sequence[OppositeBookSnapshot]
    | Iterable[OppositeBookSnapshot],
) -> _IndexedOppositeBookSnapshots:
    snapshots = tuple(opposite_snapshots)
    _validate_snapshot_order(snapshots)
    return _index_validated_opposite_snapshots(snapshots)


def _replay_exit_maker_windows_indexed(
    windows: Iterable[IndependentOrderWindow],
    *,
    route: str,
    maker_replay: IndexedTradeReplay,
    opposite_snapshot_index: _IndexedOppositeBookSnapshots,
    eod_cursor: EventCursor,
    starting_position: PairedExitPosition = PairedExitPosition(),
    hedge_delay_ns: int = DEFAULT_HEDGE_DELAY_NS,
    max_book_age_ns: int | None = None,
) -> tuple[ExitMakerOrderOutcome, ...]:
    """Batch replay against one product-day-local validated snapshot index."""

    return tuple(
        _replay_exit_maker_window_indexed_snapshots(
            window,
            route=route,
            maker_replay=maker_replay,
            opposite_snapshot_index=opposite_snapshot_index,
            eod_cursor=eod_cursor,
            starting_position=starting_position,
            hedge_delay_ns=hedge_delay_ns,
            max_book_age_ns=max_book_age_ns,
        )
        for window in windows
    )


def project_earliest_full_fill_oco(
    candidates: Iterable[
        tuple[IndependentOrderWindow, ExitMakerOrderOutcome]
    ],
) -> ExitMakerOcoProjection:
    """Project independent facts onto one earliest-full-fill-wins position.

    The winning full-fill cursor emits sibling *cancel requests* for orders
    active at that cursor.  The function intentionally does not erase later
    independent fills: those are retained as cancel-race diagnostics.  A
    sibling partial before the winner, or multiple candidates sharing the
    exact earliest cursor, makes the simple position projection unsafe.
    """

    pairs = tuple(candidates)
    ids: set[str] = set()
    for window, outcome in pairs:
        if not isinstance(window, IndependentOrderWindow):
            raise TypeError("OCO windows must be IndependentOrderWindow values")
        if not isinstance(outcome, ExitMakerOrderOutcome):
            raise TypeError("OCO outcomes must be ExitMakerOrderOutcome values")
        if window.generation_id != outcome.generation_id:
            raise ValueError("OCO window/outcome generation IDs must match")
        if window.generation_id in ids:
            raise ValueError("OCO candidate generation IDs must be unique")
        ids.add(window.generation_id)

    full_candidates = sorted(
        (
            outcome.quantity_fill.full_fill_cursor,
            outcome.generation_id,
        )
        for _, outcome in pairs
        if outcome.full_fill is True
        and outcome.quantity_fill.full_fill_cursor is not None
    )
    if not full_candidates:
        return ExitMakerOcoProjection(
            winner_generation_id=None,
            winner_full_fill_cursor=None,
            members=tuple(
                ExitMakerOcoMember(
                    generation_id=outcome.generation_id,
                    route=outcome.route,
                    disposition="no_full_fill_winner",
                    independent_full_fill_cursor=(
                        outcome.quantity_fill.full_fill_cursor
                    ),
                    oco_cancel_request_cursor=None,
                    independent_full_fill_after_cancel_request=False,
                    partial_fill_before_winner=False,
                    selected_for_position=False,
                )
                for _, outcome in pairs
            ),
            same_cursor_winner_candidates=(),
            position_projection_safe=False,
        )

    winner_cursor, winner_id = full_candidates[0]
    assert winner_cursor is not None
    same_cursor = tuple(
        generation_id
        for cursor, generation_id in full_candidates
        if cursor == winner_cursor
    )
    members: list[ExitMakerOcoMember] = []
    unsafe_partial = False
    for window, outcome in pairs:
        full_cursor = outcome.quantity_fill.full_fill_cursor
        first_cursor = outcome.quantity_fill.first_fill_cursor
        partial_before = (
            outcome.generation_id != winner_id
            and first_cursor is not None
            and first_cursor < winner_cursor
            and (full_cursor is None or full_cursor >= winner_cursor)
        )
        unsafe_partial = unsafe_partial or partial_before
        if outcome.generation_id == winner_id:
            disposition: OcoDisposition = "winner"
            cancel_cursor = None
        elif full_cursor == winner_cursor:
            disposition = "sibling_full_fill_race"
            cancel_cursor = winner_cursor
        elif window.stop_cursor < winner_cursor:
            disposition = "ended_before_winner"
            cancel_cursor = None
        elif window.start_cursor >= winner_cursor:
            disposition = "not_active_at_winner"
            cancel_cursor = None
        else:
            disposition = "sibling_cancel_required"
            cancel_cursor = winner_cursor
        after_request = (
            cancel_cursor is not None
            and full_cursor is not None
            and full_cursor >= cancel_cursor
        )
        members.append(
            ExitMakerOcoMember(
                generation_id=outcome.generation_id,
                route=outcome.route,
                disposition=disposition,
                independent_full_fill_cursor=full_cursor,
                oco_cancel_request_cursor=cancel_cursor,
                independent_full_fill_after_cancel_request=after_request,
                partial_fill_before_winner=partial_before,
                selected_for_position=outcome.generation_id == winner_id,
            )
        )
    return ExitMakerOcoProjection(
        winner_generation_id=winner_id,
        winner_full_fill_cursor=winner_cursor,
        members=tuple(members),
        same_cursor_winner_candidates=same_cursor,
        position_projection_safe=(len(same_cursor) == 1 and not unsafe_partial),
    )


def _route_contract(route: str) -> ExitMakerRouteContract:
    try:
        contract = EXIT_MAKER_ROUTE_CONTRACTS[route]  # type: ignore[index]
    except KeyError as error:
        raise ValueError(f"unsupported exit maker route: {route}") from error
    spec = ROUTE_SPECS[route]
    if spec.stage != "exit":
        raise AssertionError("exit maker route contract points at an entry route")
    return contract


def _hedge_completion_groups(
    replay: IndexedTradeReplay,
    window: IndependentOrderWindow,
    known_filled: int,
    maker_units_per_hedge: int,
) -> tuple[tuple[int, int, EventCursor], ...]:
    by_cursor: dict[EventCursor, list[int]] = {}
    hedge_units = known_filled // maker_units_per_hedge
    for zero_based in range(hedge_units):
        threshold = (zero_based + 1) * maker_units_per_hedge
        cursor = replay.label_quantity(window, threshold).full_fill_cursor
        if cursor is None:
            raise AssertionError("known maker quantity is missing a completion cursor")
        by_cursor.setdefault(cursor, []).append(zero_based + 1)
    return tuple(
        (indices[0], len(indices), cursor)
        for cursor, indices in sorted(by_cursor.items())
    )


def _residual_position(
    contract: ExitMakerRouteContract,
    starting: PairedExitPosition,
    known_filled: int | None,
    attempts: tuple[ExitMakerHedgeAttempt, ...],
) -> ResidualExitPosition | None:
    if known_filled is None:
        return None
    executed_hedge = sum(item.executed_hedge_quantity for item in attempts)
    if contract.route == FUTURE_BID_EXIT_ROUTE:
        spot = starting.spot_long_lots - executed_hedge
        future = starting.future_short_contracts - known_filled
    else:
        spot = starting.spot_long_lots - known_filled
        future = starting.future_short_contracts - executed_hedge
    if spot < 0 or future < 0:
        raise AssertionError("exit execution exceeded starting inventory")
    return ResidualExitPosition(spot, future)


def _branch_status(
    fill: QuantityFillLabel,
    residual: ResidualExitPosition | None,
    at_eod: bool,
) -> ExitBranch:
    if fill.known_filled_quantity_before_stop is None:
        return "fill_unknown_at_eod" if at_eod else "fill_unknown_then_cancel"
    if residual is not None and residual.flat:
        return "flat_same_day"
    if fill.any_fill is False:
        return (
            "carry_at_eod_cancel_unconfirmed"
            if at_eod
            else "no_fill_before_cancel_request"
        )
    if fill.partial_fill:
        return (
            "partial_fill_carry_at_eod"
            if at_eod
            else "partial_fill_then_cancel"
        )
    return (
        "hedge_incomplete_carry_at_eod"
        if at_eod
        else "hedge_incomplete_residual"
    )


def _target_rank(
    side: Literal["bid", "ask"],
    target: float,
    maker_bid: float,
    maker_ask: float,
) -> str:
    if side == "bid":
        if math.isclose(target, maker_bid, rel_tol=0.0, abs_tol=1e-8):
            return "B1"
        return "inside" if target > maker_bid else "behind"
    if math.isclose(target, maker_ask, rel_tol=0.0, abs_tol=1e-8):
        return "A1"
    return "inside" if target < maker_ask else "behind"


def _validate_snapshot_order(
    snapshots: tuple[OppositeBookSnapshot, ...],
) -> None:
    for snapshot in snapshots:
        if not isinstance(snapshot, OppositeBookSnapshot):
            raise TypeError("opposite_snapshots must contain OppositeBookSnapshot")
    for previous, current in zip(snapshots, snapshots[1:]):
        if current.cursor <= previous.cursor:
            raise ValueError("opposite_snapshots must be strictly cursor-sorted")


def _positive(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)
