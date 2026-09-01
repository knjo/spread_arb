"""Synthetic acceptance contracts for the integrated S1 domain clock."""

from __future__ import annotations

import json
import unittest
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from unittest.mock import patch

from ..quote_fill.capacity_ledger import (
    CapacityIdentityRegistryReceipt,
    CapacityLedger,
)
from ..quote_fill.layered import EventCursor
from ..quote_fill.s1_accounting import (
    ExpiryAccountingMark,
    PositionEstablishedFact,
    TerminalRealizedAccounting,
    encode_accounting_fact,
)
from ..quote_fill.s1_accounting_bridge import (
    S1AccountingBridge,
    S1AccountingProduct,
)
from ..quote_fill.s1_cross_ledger_verifier import (
    verify_s1_accounting_capacity_links,
)
from ..quote_fill.s1_event_loop import (
    ActualSendMakerSnapshot,
    ContractExpiry,
    EntryObservation,
    ExitCutoff,
    ExitDrainBarrier,
    PotentialEntryFill,
    S1CarryCodecError,
    S1CarryContractBinding,
    S1CarryPosition,
    S1EventLoop,
    S1ExecutionFact,
    S1ExternalEvent,
    S1LoopConfig,
    S1Product,
    S1ReplayResult,
    SentEntryOrder,
    SessionExpiry,
    VenueBookUpdate,
    decode_s1_carry_contract_binding,
    decode_s1_carry_position,
    encode_s1_carry_contract_binding,
    encode_s1_carry_position,
)
from ..quote_fill.s1_exit_target import S1SpotAskTarget
from ..quote_fill.s1_hedge import (
    HEDGE_DELAY_NS,
    HEDGE_RETRY_NS,
    CausalBookState,
    RawBookCursor,
    RawBookEvent,
    RawBookLevel,
    RawBookStateMachine,
)
from ..quote_fill.s1_spot_close_adapter import (
    SPOT_CLOSE_EVENT_SEQUENCE,
    OfficialSpotClose,
    official_spot_close_source_id,
)
from ..quote_fill.s1_spot_trade_adapter import (
    PhysicalSpotTrade,
    physical_spot_trade_source_id,
)
from ..quote_fill.targets import absolute_price_tick

DATE = "20260505"
NEXT_DATE = "20260506"
P1 = "2330"
P2 = "2317"
P3 = "2454"
D1_OPEN_NS = 1_000_000_000
D2_OPEN_NS = 10_000_000_000


class TimelineStateAdapter:
    def __init__(self, states: Mapping[str, Sequence[EntryObservation]]) -> None:
        self._states = {
            product_id: tuple(sorted(values, key=lambda value: value.source_cursor))
            for product_id, values in states.items()
        }
        self.calls: list[tuple[str, EventCursor]] = []

    def current_state(
        self,
        product_id: str,
        assignment_cursor: EventCursor,
    ) -> EntryObservation | None:
        self.calls.append((product_id, assignment_cursor))
        causal = [
            value
            for value in self._states.get(product_id, ())
            if value.source_cursor <= assignment_cursor
        ]
        return causal[-1] if causal else None


class RelativeFillAdapter:
    def __init__(self, delays_ns: Mapping[str, int]) -> None:
        self._delays = dict(delays_ns)

    def potential_fill(
        self,
        order: SentEntryOrder,
        snapshot: ActualSendMakerSnapshot,
    ) -> PotentialEntryFill | None:
        self.asserted_snapshot = snapshot
        delay = self._delays.get(order.product_id)
        if delay is None:
            return None
        return PotentialEntryFill(
            order.actual_start_cursor.recv_time_ns + delay,
            f"makerfill/{order.product_id}/{order.raw_order_fact_id}",
        )


class RecordingExecutionAdapter:
    def __init__(self) -> None:
        self.facts: list[S1ExecutionFact] = []

    def record_execution(self, fact: S1ExecutionFact) -> None:
        self.facts.append(fact)


class SinglePassEvents:
    def __init__(self, events: Sequence[S1ExternalEvent]) -> None:
        self._events = tuple(events)
        self.iterations = 0

    def __iter__(self) -> Iterator[S1ExternalEvent]:
        self.iterations += 1
        if self.iterations != 1:
            raise RuntimeError("external stream was iterated more than once")
        yield from self._events


class QueryRiskBooks:
    def __init__(
        self,
        changes: Mapping[tuple[str, str], Sequence[RawBookEvent]],
    ) -> None:
        self._states: dict[tuple[str, str], tuple[CausalBookState, ...]] = {}
        for key, events in changes.items():
            machine = RawBookStateMachine()
            states: list[CausalBookState] = []
            for event in sorted(events, key=lambda value: value.book_cursor):
                state = machine.ingest(event)
                if state is not None:
                    states.append(state)
            self._states[key] = tuple(states)
        self.state_calls: list[tuple[str, str, EventCursor]] = []
        self.next_calls: list[tuple[str, str, EventCursor, int]] = []

    def state_as_of(
        self,
        venue: str,
        product_id: str,
        cursor: EventCursor,
    ) -> CausalBookState | None:
        self.state_calls.append((venue, product_id, cursor))
        causal = [
            state
            for state in self._states.get((venue, product_id), ())
            if state.book_cursor.cursor <= cursor
        ]
        return causal[-1] if causal else None

    def next_change_cursor(
        self,
        venue: str,
        product_id: str,
        after_cursor: EventCursor,
        deadline_ns: int,
    ) -> EventCursor | None:
        self.next_calls.append((venue, product_id, after_cursor, deadline_ns))
        for state in self._states.get((venue, product_id), ()):
            cursor = state.book_cursor.cursor
            if cursor > after_cursor:
                return cursor if cursor.recv_time_ns <= deadline_ns else None
        return None


class ExitQuoteQueryRiskBooks(QueryRiskBooks):
    def __init__(
        self,
        changes: Mapping[tuple[str, str], Sequence[RawBookEvent]],
    ) -> None:
        super().__init__(changes)
        self.exit_quote_calls: list[tuple[str, str, EventCursor, int]] = []

    def next_exit_quote_change_cursor(
        self,
        venue: str,
        product_id: str,
        after_cursor: EventCursor,
        deadline_ns: int,
    ) -> EventCursor | None:
        self.exit_quote_calls.append((venue, product_id, after_cursor, deadline_ns))
        for state in self._states.get((venue, product_id), ()):
            cursor = state.book_cursor.cursor
            if cursor > after_cursor:
                return cursor if cursor.recv_time_ns <= deadline_ns else None
        return None


class InvalidNextChangeAdapter:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def state_as_of(
        self,
        venue: str,
        product_id: str,
        cursor: EventCursor,
    ) -> None:
        del venue, product_id, cursor

    def next_change_cursor(
        self,
        venue: str,
        product_id: str,
        after_cursor: EventCursor,
        deadline_ns: int,
    ) -> EventCursor:
        del venue, product_id
        if self.mode == "regression":
            return after_cursor
        return EventCursor(deadline_ns + 1, 20, 0)


class SyntheticSpotTrades:
    def __init__(self, trades: Sequence[PhysicalSpotTrade]) -> None:
        self._trades = tuple(sorted(trades, key=lambda value: value.cursor))

    def next_trade_cursor(
        self,
        product_id: str,
        after_cursor: EventCursor,
        deadline_ns: int,
    ) -> EventCursor | None:
        for trade in self._trades:
            if (
                trade.product_id == product_id
                and trade.cursor > after_cursor
                and trade.cursor.recv_time_ns <= deadline_ns
            ):
                return trade.cursor
        return None

    def trades_at(
        self,
        product_id: str,
        timestamp_ns: int,
    ) -> tuple[PhysicalSpotTrade, ...]:
        return tuple(
            trade
            for trade in self._trades
            if trade.product_id == product_id
            and trade.cursor.recv_time_ns == timestamp_ns
        )


class PriceAwareSpotTrades(SyntheticSpotTrades):
    def __init__(self, trades: Sequence[PhysicalSpotTrade]) -> None:
        super().__init__(trades)
        self.price_aware_calls: list[tuple[str, EventCursor, int, float]] = []
        self.generic_calls = 0
        self.trades_at_times: list[int] = []
        self.trades_at_calls: list[tuple[str, int]] = []

    def next_trade_cursor(
        self,
        product_id: str,
        after_cursor: EventCursor,
        deadline_ns: int,
    ) -> EventCursor | None:
        self.generic_calls += 1
        return super().next_trade_cursor(product_id, after_cursor, deadline_ns)

    def next_trade_cursor_at_or_above(
        self,
        product_id: str,
        after_cursor: EventCursor,
        deadline_ns: int,
        minimum_trade_price: float,
    ) -> EventCursor | None:
        self.price_aware_calls.append(
            (product_id, after_cursor, deadline_ns, minimum_trade_price)
        )
        for trade in self._trades:
            if (
                trade.product_id == product_id
                and trade.cursor > after_cursor
                and trade.cursor.recv_time_ns <= deadline_ns
                and trade.trade_price >= minimum_trade_price
            ):
                return trade.cursor
        return None

    def trades_at(
        self,
        product_id: str,
        timestamp_ns: int,
    ) -> tuple[PhysicalSpotTrade, ...]:
        self.trades_at_times.append(timestamp_ns)
        self.trades_at_calls.append((product_id, timestamp_ns))
        return super().trades_at(product_id, timestamp_ns)


def spot_trade(
    product_id: str,
    time_ns: int,
    *,
    price: float = 102.0,
    quantity: int = 2_000,
    row: int = 0,
) -> PhysicalSpotTrade:
    quote_code = product_id
    cursor = EventCursor(time_ns, 1, row)
    return PhysicalSpotTrade(
        product_id=product_id,
        quote_code=quote_code,
        cursor=cursor,
        trade_price=price,
        quantity_shares=quantity,
        source_id=physical_spot_trade_source_id(
            product_id,
            quote_code,
            time_ns,
            1,
            1,
            row,
        ),
        channel_sequence=1,
        packet_sequence=1,
        source_row=row,
        trial_match=False,
    )


def official_close(
    product_id: str,
    time_ns: int,
    *,
    date: str = DATE,
    quote_code: str | None = None,
    close_price: float = 101.0,
    row: int = 0,
) -> OfficialSpotClose:
    quote = quote_code or {P1: "CDFE6", P2: "HCFD6", P3: "QFFD6"}[product_id]
    cursor = EventCursor(time_ns, SPOT_CLOSE_EVENT_SEQUENCE, row)
    return OfficialSpotClose(
        date=date,
        product_id=product_id,
        quote_code=quote,
        close_price=close_price,
        source_cursor=cursor,
        source_id=official_spot_close_source_id(
            date,
            product_id,
            quote,
            time_ns,
            1,
            1,
            row,
        ),
        channel_sequence=1,
        packet_sequence=1,
        source_row=row,
    )


def product(
    product_id: str,
    *,
    session_end: int = 20_000_000_000,
    end_date: str | None = None,
) -> S1Product:
    quote = {P1: "CDFE6", P2: "HCFD6", P3: "QFFD6"}[product_id]
    return S1Product(
        product_id,
        product_id,
        quote,
        contract_size_shares=2_000,
        future_contracts=1,
        future_session_end_time_ns=session_end,
        spot_session_end_time_ns=session_end,
        end_date=end_date,
    )


def observation(
    product_id: str,
    time_ns: int,
    *,
    row: int = 0,
    tick: int = 1_000,
    price: float = 100.0,
    notional: int = 5_000,
    gate: bool = True,
    admission: bool = True,
    snapshot: int = 1,
    lower_bp: float = 12.5,
    exit_price: float = 101.0,
) -> EntryObservation:
    return EntryObservation(
        EventCursor(time_ns, 10, row),
        product_id,
        tick if gate else None,
        price if gate else None,
        notional if admission else None,
        gate,
        admission,
        "eligible" if gate else "gate_closed",
        ActualSendMakerSnapshot(
            product_id,
            snapshot,
            time_ns,
            100.0,
            99.5,
            10,
            10,
        ),
        lower_bp,
        frozen_exit_target_price=exit_price,
        frozen_exit_absolute_price_tick=absolute_price_tick(
            exit_price,
            market="spot",
            session_date=DATE,
        ),
    )


def book(
    venue: str,
    product_id: str,
    time_ns: int,
    *,
    packet: int = 1,
    row: int = 0,
    legal: bool = True,
    trial: bool = False,
    bid_price: float = 99.0,
    ask_price: float = 101.0,
) -> VenueBookUpdate:
    quantity = 3_000 if venue == "spot" else 10
    event = RawBookEvent(
        RawBookCursor(EventCursor(time_ns, 20, row), packet),
        trial_match=trial,
        formal_book=not trial,
        reference_price=100.0 if not trial else None,
        l1_bids=(RawBookLevel(bid_price, quantity),) if legal and not trial else (),
        l1_asks=(RawBookLevel(ask_price, quantity),) if legal and not trial else (),
    )
    return VenueBookUpdate(venue, product_id, event)


def config(
    *,
    date: str = DATE,
    global_cap: int = 20_000,
    product_cap: int = 10_000,
    spot_request_cap: int = 100,
    future_request_cap: int = 5,
    window_ns: int = 1_000_000_000,
) -> S1LoopConfig:
    return S1LoopConfig(
        date,
        "q95",
        global_cap_twd=global_cap,
        product_cap_twd=product_cap,
        spot_request_cap=spot_request_cap,
        future_request_cap=future_request_cap,
        request_window_ns=window_ns,
    )


class S1EventLoopAcceptanceTest(unittest.TestCase):
    def _day_one_carry(
        self,
        *,
        accounting: S1AccountingBridge | None = None,
    ) -> tuple[S1ReplayResult, CapacityLedger]:
        opened = observation(P1, D1_OPEN_NS)
        ledger = CapacityLedger(
            global_cap_twd=20_000,
            product_cap_twd=10_000,
        )
        result = S1EventLoop(
            config(),
            (product(P1, session_end=D2_OPEN_NS - 1),),
            entry_state_adapter=TimelineStateAdapter({P1: (opened,)}),
            fill_adapter=RelativeFillAdapter({P1: 100}),
            normal_exit_enabled=True,
            spot_trade_adapter=SyntheticSpotTrades(()),
            accounting_adapter=accounting,
            capacity_ledger=ledger,
        ).run(
            (
                book("future", P1, D1_OPEN_NS - 10, packet=900),
                book("spot", P1, D1_OPEN_NS - 10, packet=901),
                opened,
            )
        )
        self.assertEqual(len(result.carry_out), 1)
        self.assertEqual(result.carry_out[0].product_id, P1)
        return result, ledger

    def test_carry_position_json_codec_round_trip_is_exact(self) -> None:
        day_one, _ = self._day_one_carry()
        carry = day_one.carry_out[0]
        record = json.loads(
            json.dumps(encode_s1_carry_position(carry), allow_nan=False)
        )

        self.assertEqual(
            record["position_established_fact"],
            encode_accounting_fact(carry.position_established_fact),
        )
        decoded = decode_s1_carry_position(record)
        self.assertEqual(decoded, carry)
        self.assertEqual(encode_s1_carry_position(decoded), record)
        self.assertIsInstance(decoded, S1CarryPosition)

    def test_frozen_exit_price_tick_consistency_is_enforced_at_boundaries(
        self,
    ) -> None:
        day_one, _ = self._day_one_carry()
        order = day_one.orders[0]
        self.assertIsNotNone(order.frozen_exit_absolute_price_tick)
        assert order.frozen_exit_absolute_price_tick is not None
        with self.assertRaisesRegex(ValueError, "sent-order.*price/tick mismatch"):
            replace(
                order,
                frozen_exit_absolute_price_tick=(
                    order.frozen_exit_absolute_price_tick + 1
                ),
            )
        with self.assertRaisesRegex(ValueError, "legal spot ladder"):
            replace(order, frozen_exit_target_price=100.1)

        carry = day_one.carry_out[0]
        with self.assertRaisesRegex(ValueError, "carry.*price/tick mismatch"):
            replace(
                carry,
                frozen_exit_absolute_price_tick=(
                    carry.frozen_exit_absolute_price_tick + 1
                ),
            )

        mismatched_tick = encode_s1_carry_position(carry)
        mismatched_tick["frozen_exit_absolute_price_tick"] = (
            carry.frozen_exit_absolute_price_tick + 1
        )
        with self.assertRaisesRegex(S1CarryCodecError, "price/tick mismatch"):
            decode_s1_carry_position(mismatched_tick)

        off_ladder_price = encode_s1_carry_position(carry)
        off_ladder_price["frozen_exit_target_price"] = 100.1
        with self.assertRaisesRegex(S1CarryCodecError, "legal spot ladder"):
            decode_s1_carry_position(off_ladder_price)

    def test_carry_position_json_codec_rejects_schema_type_and_fact_tamper(
        self,
    ) -> None:
        day_one, _ = self._day_one_carry()
        carry = day_one.carry_out[0]
        original = encode_s1_carry_position(carry)

        missing = dict(original)
        missing.pop("quote_code")
        with self.assertRaisesRegex(S1CarryCodecError, "schema mismatch"):
            decode_s1_carry_position(missing)

        extra = dict(original)
        extra["unexpected"] = None
        with self.assertRaisesRegex(S1CarryCodecError, "schema mismatch"):
            decode_s1_carry_position(extra)

        for field_name, invalid_value in (
            ("record_type", "s1_carry_contract_binding"),
            ("schema_version", True),
            ("frozen_exit_threshold_basis_bp", 1),
            ("frozen_exit_threshold_basis_bp", float("nan")),
            ("frozen_exit_target_price", 1),
            ("frozen_exit_target_price", float("nan")),
            ("frozen_exit_absolute_price_tick", 1.0),
            ("frozen_exit_absolute_price_tick", 0),
            ("execution_truth", "approx"),
        ):
            malformed = dict(original)
            malformed[field_name] = invalid_value
            with (
                self.subTest(field_name=field_name, invalid_value=invalid_value),
                self.assertRaises(S1CarryCodecError),
            ):
                decode_s1_carry_position(malformed)

        fact_not_object = dict(original)
        fact_not_object["position_established_fact"] = []
        with self.assertRaisesRegex(S1CarryCodecError, "JSON object"):
            decode_s1_carry_position(fact_not_object)

        fact_extra = json.loads(json.dumps(original))
        fact_extra["position_established_fact"]["unexpected"] = 1
        with self.assertRaisesRegex(S1CarryCodecError, "schema mismatch"):
            decode_s1_carry_position(fact_extra)

        fact_capacity_tamper = json.loads(json.dumps(original))
        fact_capacity_tamper["position_established_fact"]["capacity_id"] = "other"
        with self.assertRaisesRegex(S1CarryCodecError, "capacity_id"):
            decode_s1_carry_position(fact_capacity_tamper)

        fact_quantity_tamper = json.loads(json.dumps(original))
        fact_quantity_tamper["position_established_fact"]["spot_shares"] = 0
        with self.assertRaises(S1CarryCodecError):
            decode_s1_carry_position(fact_quantity_tamper)

        fact_time_tamper = json.loads(json.dumps(original))
        fact_time_tamper["position_established_fact"]["position_established_ns"] += 1
        with self.assertRaisesRegex(S1CarryCodecError, "establishment time"):
            decode_s1_carry_position(fact_time_tamper)

        with self.assertRaises(S1CarryCodecError):
            encode_s1_carry_position(replace(carry, frozen_exit_threshold_basis_bp=1))
        with self.assertRaisesRegex(ValueError, "price/tick mismatch"):
            replace(carry, frozen_exit_target_price=1)

    def test_carry_contract_binding_json_codec_is_strict_and_exact(self) -> None:
        binding = S1CarryContractBinding.from_product(product(P1, end_date="20260529"))
        record = json.loads(
            json.dumps(
                encode_s1_carry_contract_binding(binding),
                allow_nan=False,
            )
        )
        decoded = decode_s1_carry_contract_binding(record)
        self.assertEqual(decoded, binding)
        self.assertEqual(encode_s1_carry_contract_binding(decoded), record)

        no_expiry = S1CarryContractBinding.from_product(product(P1))
        no_expiry_record = json.loads(
            json.dumps(encode_s1_carry_contract_binding(no_expiry), allow_nan=False)
        )
        self.assertIsNone(no_expiry_record["end_date"])
        self.assertEqual(
            decode_s1_carry_contract_binding(no_expiry_record),
            no_expiry,
        )

        missing = dict(record)
        missing.pop("value_code")
        with self.assertRaisesRegex(S1CarryCodecError, "schema mismatch"):
            decode_s1_carry_contract_binding(missing)

        extra = dict(record)
        extra["unexpected"] = 1
        with self.assertRaisesRegex(S1CarryCodecError, "schema mismatch"):
            decode_s1_carry_contract_binding(extra)

        for field_name, invalid_value in (
            ("record_type", "s1_carry_position"),
            ("schema_version", True),
            ("contract_size_shares", True),
            ("future_contracts", float("inf")),
            ("end_date", 20260529),
            ("end_date", "20260230"),
        ):
            malformed = dict(record)
            malformed[field_name] = invalid_value
            with (
                self.subTest(field_name=field_name, invalid_value=invalid_value),
                self.assertRaises(S1CarryCodecError),
            ):
                decode_s1_carry_contract_binding(malformed)

    def test_query_book_path_matches_stream_trial_and_reopen_facts(self) -> None:
        opened = observation(P1, 100)
        fill_time = 200
        target = fill_time + HEDGE_DELAY_NS
        trial = book("future", P1, target, packet=501, trial=True)
        reopened = book("future", P1, target + 10, packet=502)

        stream = S1EventLoop(
            config(),
            (product(P1),),
            entry_state_adapter=TimelineStateAdapter({P1: (opened,)}),
            fill_adapter=RelativeFillAdapter({P1: 100}),
        ).run((opened, trial, reopened))

        queried_books = QueryRiskBooks({("future", P1): (trial.event, reopened.event)})
        queried = S1EventLoop(
            config(),
            (product(P1),),
            entry_state_adapter=TimelineStateAdapter({P1: (opened,)}),
            risk_book_adapter=queried_books,
            fill_adapter=RelativeFillAdapter({P1: 100}),
        ).run((opened,))

        self.assertEqual(queried.positions, stream.positions)
        self.assertEqual(queried.executions, stream.executions)
        self.assertEqual(queried.position_events, stream.position_events)
        self.assertEqual(queried.risk_events, stream.risk_events)
        self.assertEqual(
            [
                event
                for event in queried.request_events
                if event.event_type == "actual_send"
            ],
            [
                event
                for event in stream.request_events
                if event.event_type == "actual_send"
            ],
        )
        hedge_evaluations = [
            event
            for event in queried.risk_events
            if event.risk_kind == "hedge" and event.event_type == "evaluated"
        ]
        self.assertEqual(hedge_evaluations[0].gate_reason, "trial_match")
        self.assertEqual(queried.executions[-1].book_packet_sequence, 502)
        self.assertEqual(len(queried_books.state_calls), 4)
        self.assertEqual(len(queried_books.next_calls), 1)

    def test_query_book_adapter_is_idle_without_active_risk(self) -> None:
        opened = observation(P1, 100)
        future = book("future", P1, 1_000, packet=601)
        queried_books = QueryRiskBooks({("future", P1): (future.event,)})
        result = S1EventLoop(
            config(),
            (product(P1),),
            entry_state_adapter=TimelineStateAdapter({P1: (opened,)}),
            risk_book_adapter=queried_books,
        ).run((opened,))

        self.assertEqual(result.positions, ())
        self.assertEqual(queried_books.state_calls, [])
        self.assertEqual(queried_books.next_calls, [])

    def test_query_next_change_contract_fails_closed(self) -> None:
        opened = observation(P1, 100)
        for mode, message in (
            ("regression", "must follow"),
            ("past_deadline", "exceeds"),
        ):
            with self.subTest(mode=mode):
                loop = S1EventLoop(
                    config(),
                    (product(P1),),
                    entry_state_adapter=TimelineStateAdapter({P1: (opened,)}),
                    risk_book_adapter=InvalidNextChangeAdapter(mode),
                    fill_adapter=RelativeFillAdapter({P1: 100}),
                )
                with self.assertRaisesRegex(ValueError, message):
                    loop.run((opened,))

    def test_query_future_quota_delay_uses_latest_reopened_book(self) -> None:
        p1 = observation(P1, 100)
        p2 = observation(P2, 101)
        fill_time = 200
        target = fill_time + HEDGE_DELAY_NS
        changes = {
            ("future", product_id): (
                book("future", product_id, target, packet=1).event,
                book(
                    "future",
                    product_id,
                    target + 50,
                    packet=2,
                    trial=True,
                ).event,
                book("future", product_id, target + 110, packet=3).event,
            )
            for product_id in (P1, P2)
        }
        queried_books = QueryRiskBooks(changes)
        result = S1EventLoop(
            config(future_request_cap=1, window_ns=100),
            (product(P1), product(P2)),
            entry_state_adapter=TimelineStateAdapter({P1: (p1,), P2: (p2,)}),
            risk_book_adapter=queried_books,
            fill_adapter=RelativeFillAdapter({P1: 100, P2: 99}),
        ).run((p1, p2))

        hedges = sorted(
            (fact.cursor.recv_time_ns, fact.book_packet_sequence)
            for fact in result.executions
            if fact.role == "entry_hedge"
        )
        self.assertEqual(hedges, [(target, 1), (target + 110, 3)])
        self.assertTrue(
            any(
                call[2].recv_time_ns == target + 100
                for call in queried_books.state_calls
            )
        )
        self.assertEqual(len(queried_books.state_calls), 9)
        self.assertEqual(len(queried_books.next_calls), 3)

    def test_query_book_rollback_path_reverses_spot_first_leg(self) -> None:
        opened = observation(P1, 100)
        spot = book("spot", P1, 150, packet=701)
        queried_books = QueryRiskBooks({("spot", P1): (spot.event,)})
        result = S1EventLoop(
            config(),
            (product(P1),),
            entry_state_adapter=TimelineStateAdapter({P1: (opened,)}),
            risk_book_adapter=queried_books,
            fill_adapter=RelativeFillAdapter({P1: 100}),
        ).run((opened,))

        rollback = next(
            fact for fact in result.executions if fact.role == "entry_rollback"
        )
        self.assertEqual(rollback.quantity, 2_000)
        self.assertEqual(rollback.quantity_unit, "spot_shares")
        self.assertEqual(rollback.book_packet_sequence, 701)
        self.assertEqual(result.positions[0].state, "entry_emergency_rollback_flat")
        self.assertEqual(len(queried_books.state_calls), 6)
        self.assertEqual(len(queried_books.next_calls), 2)

    def test_streaming_merge_is_single_pass_and_combines_equal_time_wake(self) -> None:
        p1 = observation(P1, 100)
        stale = observation(P2, 101)
        refreshed = observation(P2, 200, tick=1_001, price=100.5, snapshot=9)
        events = SinglePassEvents((p1, stale, refreshed))
        state = TimelineStateAdapter({P1: (p1,), P2: (stale, refreshed)})
        loop = S1EventLoop(
            config(spot_request_cap=1, window_ns=100),
            (product(P1), product(P2)),
            entry_state_adapter=state,
        )

        result = loop.run(events)

        self.assertEqual(events.iterations, 1)
        p2_order = next(order for order in result.orders if order.product_id == P2)
        self.assertEqual(p2_order.actual_start_cursor.recv_time_ns, 200)
        self.assertEqual(p2_order.absolute_price_tick, 1_001)
        self.assertEqual(p2_order.maker_snapshot_channel_seq, 9)

    def test_external_cursor_regression_fails_closed(self) -> None:
        later = observation(P1, 200)
        earlier = observation(P1, 100)
        state = TimelineStateAdapter({P1: (earlier, later)})
        loop = S1EventLoop(
            config(),
            (product(P1),),
            entry_state_adapter=state,
        )

        with self.assertRaisesRegex(ValueError, "non-decreasing"):
            loop.run(SinglePassEvents((later, earlier)))

    def test_actual_cursor_ineligible_request_uses_no_token_or_capacity(self) -> None:
        intent_state = observation(P1, 100)
        actual_state = observation(P1, 100, admission=False, snapshot=2)
        state = TimelineStateAdapter({P1: (actual_state,)})
        loop = S1EventLoop(
            config(),
            (product(P1),),
            entry_state_adapter=state,
        )

        result = loop.run((intent_state, SessionExpiry(EventCursor(200, 30, 0))))

        self.assertEqual(result.orders, ())
        self.assertEqual(result.admission_events, ())
        self.assertEqual(result.capacity_transitions, ())
        self.assertEqual(result.spot_requests_sent, 0)

    def test_economic_gate_reopens_on_book_change_without_pending_new(self) -> None:
        blocked = replace(
            observation(P1, 100, admission=False),
            gate_reason="economic_gate:expected_margin_not_above_floor",
        )
        reopened = observation(P1, 150, snapshot=3)
        state = TimelineStateAdapter({P1: (blocked, reopened)})
        future_change = book("future", P1, 150, packet=44)
        books = QueryRiskBooks({("future", P1): (future_change.event,)})
        loop = S1EventLoop(
            config(),
            (product(P1),),
            entry_state_adapter=state,
            risk_book_adapter=books,
        )

        result = loop.run((blocked, SessionExpiry(EventCursor(300, 30, 0))))

        self.assertEqual(len(result.orders), 1)
        self.assertEqual(result.orders[0].actual_start_cursor.recv_time_ns, 150)
        self.assertEqual(result.spot_requests_sent, 1)
        self.assertTrue(
            any(
                venue == "future" and after.recv_time_ns == 100
                for venue, _, after, _ in books.next_calls
            )
        )

    def test_missing_future_ask_reopens_on_future_book_recovery(self) -> None:
        blocked = replace(
            observation(P1, 100, gate=False, admission=False),
            gate_reason="empty_future_book",
            base_gate_book_wake_venues=frozenset(("future",)),
        )
        reopened = observation(P1, 150, snapshot=3)
        state = TimelineStateAdapter({P1: (blocked, reopened)})
        future_recovery = book("future", P1, 150, packet=45)
        books = QueryRiskBooks({("future", P1): (future_recovery.event,)})
        loop = S1EventLoop(
            config(),
            (product(P1),),
            entry_state_adapter=state,
            risk_book_adapter=books,
        )

        result = loop.run((blocked, SessionExpiry(EventCursor(300, 30, 0))))

        self.assertEqual(len(result.orders), 1)
        self.assertEqual(result.orders[0].actual_start_cursor.recv_time_ns, 150)
        self.assertEqual(result.spot_requests_sent, 1)
        self.assertEqual(
            [
                venue
                for venue, _, after, _ in books.next_calls
                if after.recv_time_ns == 100
            ],
            ["future"],
        )

    def test_delayed_new_refetches_current_state_and_cannot_fill_at_start(self) -> None:
        first = observation(P1, 100, snapshot=11)
        stale = observation(P2, 101, snapshot=21)
        refreshed = observation(
            P2,
            150,
            tick=1_001,
            price=100.5,
            notional=6_000,
            snapshot=22,
        )
        state = TimelineStateAdapter({P1: (first,), P2: (stale, refreshed)})
        fill_adapter = RelativeFillAdapter({P2: 0})
        loop = S1EventLoop(
            config(spot_request_cap=1, window_ns=100),
            (product(P1), product(P2)),
            entry_state_adapter=state,
            fill_adapter=fill_adapter,
        )

        result = loop.run((first, stale))

        self.assertEqual(len(result.orders), 2)
        delayed = next(order for order in result.orders if order.product_id == P2)
        self.assertEqual(delayed.actual_start_cursor.recv_time_ns, 200)
        self.assertEqual(delayed.absolute_price_tick, 1_001)
        self.assertEqual(delayed.target_price, 100.5)
        self.assertEqual(delayed.reservation_notional_twd, 6_000)
        self.assertEqual(delayed.maker_snapshot_channel_seq, 22)
        self.assertIs(fill_adapter.asserted_snapshot, delayed.maker_snapshot)
        self.assertTrue(
            any(audit.status == "coalesced" for audit in result.candidate_intent_audit)
        )
        self.assertEqual(
            result.fill_events[-1].status, "suppressed_not_after_actual_start"
        )
        self.assertEqual(result.executions, ())
        self.assertIsNone(result.mean_daily_net_twd)

    def test_shadow_cap_blocks_siblings_and_expiry_never_creates_raw_order(
        self,
    ) -> None:
        p1 = observation(P1, 100, notional=6_000)
        p2 = observation(P2, 100, row=1, notional=6_000)
        expiry = SessionExpiry(EventCursor(1_000, 30, 0))
        state = TimelineStateAdapter({P1: (p1,), P2: (p2,)})
        loop = S1EventLoop(
            config(global_cap=10_000),
            (product(P1), product(P2)),
            entry_state_adapter=state,
        )

        result = loop.run((p1, p2, expiry))

        self.assertEqual(len(result.orders), 1)
        self.assertEqual(
            sorted((event.admitted, event.status) for event in result.admission_events),
            [(False, "blocked_global_cap"), (True, "admitted")],
        )
        self.assertEqual(
            result.capacity_transitions[-1].global_after.total_committed_notional_twd,
            0,
        )
        self.assertTrue(
            any(audit.status == "expired" for audit in result.candidate_intent_audit)
        )

    def test_redundant_blocked_admission_probes_sleep_with_economic_equivalence(
        self,
    ) -> None:
        admitted = observation(P2, 100, notional=5_000)
        blocked = observation(P1, 100, row=1, notional=5_000)
        p3_initial = observation(P3, 100, row=2, notional=5_000)
        p3_states = (
            p3_initial,
            *(
                replace(
                    p3_initial,
                    source_cursor=EventCursor(timestamp, 10, 0),
                )
                for timestamp in range(200, 1_200, 100)
            ),
        )
        events = (
            admitted,
            blocked,
            *p3_states,
            SessionExpiry(EventCursor(2_000, 30, 0)),
        )

        def replay(*, suppress: bool) -> S1ReplayResult:
            return S1EventLoop(
                config(global_cap=5_000, product_cap=5_000),
                (product(P1), product(P2), product(P3)),
                entry_state_adapter=TimelineStateAdapter(
                    {P1: (blocked,), P2: (admitted,), P3: p3_states}
                ),
                suppress_redundant_blocked_admission_probes=suppress,
            ).run(events)

        optimized = replay(suppress=True)
        legacy = replay(suppress=False)

        self.assertEqual(
            replace(
                optimized,
                admission_events=(),
                capacity_transitions=(),
                suppressed_redundant_blocked_admission_probes=0,
            ),
            replace(
                legacy,
                admission_events=(),
                capacity_transitions=(),
                suppressed_redundant_blocked_admission_probes=0,
            ),
        )
        optimized_blocked = [
            event for event in optimized.admission_events if not event.admitted
        ]
        legacy_blocked = [
            event for event in legacy.admission_events if not event.admitted
        ]
        self.assertEqual(len(optimized_blocked), 2)
        self.assertEqual(len(legacy_blocked), 22)
        self.assertEqual(
            optimized.suppressed_redundant_blocked_admission_probes,
            20,
        )
        self.assertEqual(optimized.orders, legacy.orders)
        self.assertEqual(optimized.executions, legacy.executions)
        self.assertEqual(optimized.positions, legacy.positions)
        self.assertEqual(
            optimized.capacity_transitions[-1].global_after,
            legacy.capacity_transitions[-1].global_after,
        )

    def test_own_snapshot_change_rechecks_sleeping_blocked_candidate(self) -> None:
        admitted = observation(P2, 100, notional=5_000)
        initial = observation(P1, 100, row=1, notional=5_000, snapshot=801)
        refreshed = observation(P1, 200, notional=5_000, snapshot=802)
        unchanged = replace(
            refreshed,
            source_cursor=EventCursor(300, 10, 0),
        )
        result = S1EventLoop(
            config(global_cap=5_000, product_cap=5_000),
            (product(P1), product(P2)),
            entry_state_adapter=TimelineStateAdapter(
                {P1: (initial, refreshed, unchanged), P2: (admitted,)}
            ),
        ).run((admitted, initial, refreshed, unchanged))

        p1_attempts = [
            event for event in result.admission_events if event.product_id == P1
        ]
        self.assertEqual(
            [event.cursor.recv_time_ns for event in p1_attempts],
            [100, 200],
        )
        self.assertEqual(
            result.suppressed_redundant_blocked_admission_probes,
            1,
        )

    def test_cross_product_probe_preserves_venue_priority_and_send_stream(self) -> None:
        p2_intent = observation(P2, 100, snapshot=201)
        p2_actual = observation(P2, 200, snapshot=202)
        p3_trigger = observation(P3, 200, row=1, snapshot=301)
        events = (p2_intent, p3_trigger, SessionExpiry(EventCursor(300, 30, 0)))

        def replay(*, suppress: bool) -> S1ReplayResult:
            return S1EventLoop(
                config(),
                (product(P2), product(P3)),
                entry_state_adapter=TimelineStateAdapter(
                    {P2: (p2_actual,), P3: (p3_trigger,)}
                ),
                suppress_redundant_blocked_admission_probes=suppress,
            ).run(events)

        optimized = replay(suppress=True)
        legacy = replay(suppress=False)

        self.assertEqual(optimized.orders, legacy.orders)
        self.assertEqual(optimized.request_events, legacy.request_events)
        self.assertEqual(optimized.order_events, legacy.order_events)
        self.assertEqual(optimized.executions, legacy.executions)
        self.assertEqual(optimized.fill_events, legacy.fill_events)
        self.assertEqual(optimized.position_events, legacy.position_events)
        self.assertEqual(optimized.positions, legacy.positions)
        self.assertEqual(optimized.risk_events, legacy.risk_events)
        self.assertEqual(optimized.exit_inventory_facts, legacy.exit_inventory_facts)
        self.assertEqual(optimized.exit_physical_fills, legacy.exit_physical_fills)
        self.assertEqual(
            optimized.capacity_transitions[-1].global_after,
            legacy.capacity_transitions[-1].global_after,
        )
        self.assertEqual(
            [
                (order.product_id, order.actual_start_cursor.recv_time_ns)
                for order in optimized.orders
            ],
            [(P2, 200), (P3, 200)],
        )

    def test_same_cursor_admission_cannot_use_later_cancel_release(self) -> None:
        open_p1 = observation(P1, 100, notional=10_000)
        close_p1 = observation(P1, 200, gate=False, admission=False)
        open_p2 = observation(P2, 200, row=1, notional=10_000)
        state = TimelineStateAdapter({P1: (open_p1, close_p1), P2: (open_p2,)})
        loop = S1EventLoop(
            config(global_cap=10_000),
            (product(P1), product(P2)),
            entry_state_adapter=state,
        )

        result = loop.run((open_p1, close_p1, open_p2))

        p2_attempts = [
            event for event in result.admission_events if event.product_id == P2
        ]
        self.assertEqual(
            [(event.cursor.recv_time_ns, event.admitted) for event in p2_attempts],
            [(200, False), (201, True)],
        )
        p2_order = next(order for order in result.orders if order.product_id == P2)
        self.assertEqual(p2_order.actual_start_cursor.recv_time_ns, 201)
        release = next(
            row for row in result.capacity_transitions if row.status == "actual_cancel"
        )
        self.assertEqual(release.event_sequence, 500)

    def test_product_cap_release_wakes_sleeping_candidate_at_t_plus_one(self) -> None:
        opened = observation(P1, 100, tick=1_000, notional=5_000)
        admission_closed = observation(
            P1,
            200,
            row=0,
            tick=999,
            notional=5_000,
            admission=False,
        )
        reopened = observation(
            P1,
            200,
            row=1,
            tick=999,
            notional=5_000,
            admission=True,
            snapshot=901,
        )
        result = S1EventLoop(
            config(global_cap=10_000, product_cap=5_000),
            (product(P1),),
            entry_state_adapter=TimelineStateAdapter(
                {P1: (opened, admission_closed, reopened)}
            ),
        ).run((opened, admission_closed, reopened))

        self.assertEqual(
            [
                (event.cursor.recv_time_ns, event.status, event.admitted)
                for event in result.admission_events
            ],
            [
                (100, "admitted", True),
                (200, "blocked_product_cap", False),
                (201, "admitted", True),
            ],
        )
        self.assertEqual(
            [order.actual_start_cursor.recv_time_ns for order in result.orders],
            [100, 201],
        )
        release = next(
            transition
            for transition in result.capacity_transitions
            if transition.status == "actual_cancel"
        )
        self.assertEqual(release.timestamp_ns, 200)
        self.assertEqual(release.event_sequence, 500)

    def test_frozen_cancel_loses_to_existing_fill_then_hedge_pairs(self) -> None:
        opened = observation(P1, 100)
        closed = observation(P1, 200, gate=False, admission=False)
        state = TimelineStateAdapter({P1: (opened, closed)})
        recorder = RecordingExecutionAdapter()
        loop = S1EventLoop(
            config(),
            (product(P1),),
            entry_state_adapter=state,
            fill_adapter=RelativeFillAdapter({P1: 100}),
            execution_adapter=recorder,
        )

        result = loop.run((opened, book("future", P1, 150, packet=71), closed))

        events = [event.event_type for event in result.order_events]
        self.assertIn("filled", events)
        self.assertIn("cancel_noop_after_fill", events)
        self.assertNotIn("actual_cancelled", events)
        self.assertEqual(result.positions[0].state, "paired_open")
        self.assertEqual(
            [fact.quantity for fact in result.executions],
            [2_000, 1],
        )
        self.assertEqual(result.executions[-1].book_packet_sequence, 71)
        self.assertEqual(tuple(recorder.facts), result.executions)
        self.assertEqual(
            result.capacity_transitions[-1].global_after.paired_open,
            5_000,
        )

    def test_trial_match_at_t0_retries_with_latest_legal_book(self) -> None:
        opened = observation(P1, 100)
        fill_time = 200
        target = fill_time + HEDGE_DELAY_NS
        state = TimelineStateAdapter({P1: (opened,)})
        loop = S1EventLoop(
            config(),
            (product(P1),),
            entry_state_adapter=state,
            fill_adapter=RelativeFillAdapter({P1: fill_time - 100}),
        )

        result = loop.run(
            (
                opened,
                book("future", P1, target, packet=80, trial=True),
                book("future", P1, target + 10, packet=81),
            )
        )

        hedge_evaluations = [
            event
            for event in result.risk_events
            if event.risk_kind == "hedge" and event.event_type == "evaluated"
        ]
        self.assertEqual(hedge_evaluations[0].gate_reason, "trial_match")
        hedge_execution = next(
            fact for fact in result.executions if fact.role == "entry_hedge"
        )
        self.assertEqual(hedge_execution.cursor.recv_time_ns, target + 10)
        self.assertEqual(hedge_execution.book_packet_sequence, 81)
        self.assertEqual(result.positions[0].state, "paired_open")

    def test_inclusive_deadline_send_beats_timeout(self) -> None:
        opened = observation(P1, 100)
        fill_time = 200
        deadline = fill_time + HEDGE_DELAY_NS + HEDGE_RETRY_NS
        state = TimelineStateAdapter({P1: (opened,)})
        loop = S1EventLoop(
            config(),
            (product(P1),),
            entry_state_adapter=state,
            fill_adapter=RelativeFillAdapter({P1: 100}),
        )

        result = loop.run((opened, book("future", P1, deadline, packet=91)))

        self.assertEqual(result.positions[0].state, "paired_open")
        self.assertFalse(
            any(event.event_type == "timeout" for event in result.risk_events)
        )
        hedge = next(fact for fact in result.executions if fact.role == "entry_hedge")
        self.assertEqual(hedge.cursor.recv_time_ns, deadline)

    def test_future_quota_delay_revalidates_the_latest_book(self) -> None:
        p1 = observation(P1, 100)
        p2 = observation(P2, 101)
        fill_time = 200
        target = fill_time + HEDGE_DELAY_NS
        state = TimelineStateAdapter({P1: (p1,), P2: (p2,)})
        loop = S1EventLoop(
            config(future_request_cap=1, window_ns=100),
            (product(P1), product(P2)),
            entry_state_adapter=state,
            fill_adapter=RelativeFillAdapter({P1: 100, P2: 99}),
        )

        result = loop.run(
            (
                p1,
                p2,
                book("future", P1, target, packet=1),
                book("future", P2, target, packet=1),
                book("future", P1, target + 50, packet=2, trial=True),
                book("future", P2, target + 50, packet=2, trial=True),
                book("future", P1, target + 110, packet=3),
                book("future", P2, target + 110, packet=3),
            )
        )

        hedges = sorted(
            (fact.cursor.recv_time_ns, fact.book_packet_sequence)
            for fact in result.executions
            if fact.role == "entry_hedge"
        )
        self.assertEqual(hedges, [(target, 1), (target + 110, 3)])
        self.assertEqual(result.future_requests_sent, 2)
        self.assertTrue(
            all(position.state == "paired_open" for position in result.positions)
        )

    def test_hedge_timeout_rolls_back_2000_spot_shares_and_releases_cap(self) -> None:
        opened = observation(P1, 100)
        state = TimelineStateAdapter({P1: (opened,)})
        loop = S1EventLoop(
            config(),
            (product(P1),),
            entry_state_adapter=state,
            fill_adapter=RelativeFillAdapter({P1: 100}),
        )

        result = loop.run((opened, book("spot", P1, 150, packet=101)))

        rollback = next(
            fact for fact in result.executions if fact.role == "entry_rollback"
        )
        self.assertEqual(rollback.side, "sell")
        self.assertEqual(rollback.quantity, 2_000)
        self.assertEqual(rollback.quantity_unit, "spot_shares")
        self.assertEqual(result.positions[0].state, "entry_emergency_rollback_flat")
        self.assertEqual(
            result.capacity_transitions[-1].global_after.total_committed_notional_twd,
            0,
        )
        self.assertIsNone(result.approx_screen_completion_rate_20m)
        self.assertIsNone(result.mean_daily_net_twd)

    def test_rollback_timeout_is_terminal_and_late_book_cannot_revive(self) -> None:
        opened = observation(P1, 100)
        fill_time = 200
        rollback_deadline = fill_time + HEDGE_DELAY_NS + HEDGE_RETRY_NS + HEDGE_RETRY_NS
        state = TimelineStateAdapter({P1: (opened,)})
        loop = S1EventLoop(
            config(),
            (product(P1),),
            entry_state_adapter=state,
            fill_adapter=RelativeFillAdapter({P1: 100}),
        )

        result = loop.run((opened, book("spot", P1, rollback_deadline + 1, packet=111)))

        self.assertEqual(result.positions[0].state, "entry_hedge_timeout_unresolved")
        self.assertFalse(
            any(fact.role == "entry_rollback" for fact in result.executions)
        )
        rollback_timeouts = [
            event
            for event in result.risk_events
            if event.risk_kind == "rollback" and event.event_type == "timeout"
        ]
        self.assertEqual(len(rollback_timeouts), 1)
        self.assertEqual(
            result.capacity_transitions[-1].global_after.hedge_pending,
            5_000,
        )

    def test_existing_rollback_precedes_same_time_cancel_and_new(self) -> None:
        p1 = observation(P1, 100)
        fill_time = 200
        hedge_deadline = fill_time + HEDGE_DELAY_NS + HEDGE_RETRY_NS
        p2_open = observation(P2, hedge_deadline - 50)
        p2_close = observation(
            P2,
            hedge_deadline + 50,
            gate=False,
            admission=False,
        )
        p3_open = observation(P3, hedge_deadline + 50, row=1)
        state = TimelineStateAdapter(
            {P1: (p1,), P2: (p2_open, p2_close), P3: (p3_open,)}
        )
        loop = S1EventLoop(
            config(spot_request_cap=1, window_ns=100),
            (product(P1), product(P2), product(P3)),
            entry_state_adapter=state,
            fill_adapter=RelativeFillAdapter({P1: 100}),
        )

        result = loop.run(
            (
                p1,
                book("spot", P1, hedge_deadline - 60, packet=201),
                p2_open,
                p2_close,
                p3_open,
            )
        )

        competing_time = hedge_deadline + 50
        sent_at_competing_time = [
            event
            for event in result.request_events
            if event.event_type == "actual_send"
            and event.event_cursor.recv_time_ns == competing_time
        ]
        self.assertEqual(len(sent_at_competing_time), 1)
        self.assertEqual(
            sent_at_competing_time[0].risk_subtype,
            "emergency_rollback",
        )
        later_spot = [
            event
            for event in result.request_events
            if event.event_type == "actual_send"
            and event.event_cursor.recv_time_ns > competing_time
        ]
        self.assertEqual(
            [event.request_class for event in later_spot],
            ["cancel", "new"],
        )
        rollback = next(
            fact for fact in result.executions if fact.role == "entry_rollback"
        )
        self.assertEqual(rollback.quantity, 2_000)

    def test_normal_exit_joint_clock_fills_hedges_flattens_and_releases_cap(
        self,
    ) -> None:
        opened = observation(P1, 100)
        entry_fill_time = 200
        entry_hedge_time = entry_fill_time + HEDGE_DELAY_NS
        exit_trade_time = entry_hedge_time + 10_000_000
        exit_hedge_time = exit_trade_time + HEDGE_DELAY_NS
        trade = spot_trade(P1, exit_trade_time)

        result = S1EventLoop(
            config(),
            (product(P1),),
            entry_state_adapter=TimelineStateAdapter({P1: (opened,)}),
            fill_adapter=RelativeFillAdapter({P1: entry_fill_time - 100}),
            normal_exit_enabled=True,
            spot_trade_adapter=SyntheticSpotTrades((trade,)),
        ).run(
            (
                book("future", P1, 90, packet=10),
                book("spot", P1, 90, packet=11),
                opened,
            )
        )

        self.assertEqual(
            [fact.role for fact in result.executions],
            ["entry_maker", "entry_hedge", "exit_maker", "exit_hedge"],
        )
        self.assertEqual(result.executions[-1].cursor.recv_time_ns, exit_hedge_time)
        self.assertEqual(result.positions[0].state, "exit_maker_flat")
        self.assertEqual(result.positions[0].exit_resolution_state, "flat")
        self.assertEqual(result.positions[0].exit_unresolved_shares, 0)
        self.assertEqual(
            result.capacity_transitions[-1].event_type,
            "exit_hedge_complete",
        )
        self.assertEqual(
            result.capacity_transitions[-1].global_after.total_committed_notional_twd,
            0,
        )
        self.assertEqual(len(result.exit_physical_fills), 1)
        self.assertEqual(result.exit_physical_fills[0].fill_reason, "trade_through")
        exit_new = next(
            event
            for event in result.request_events
            if event.stage == "exit"
            and event.request_class == "new"
            and event.event_type == "actual_send"
        )
        self.assertEqual(exit_new.event_cursor.recv_time_ns, entry_hedge_time + 1)
        self.assertEqual(result.executions[2].domain_allocation_id is not None, True)
        self.assertEqual(
            result.executions[3].initiating_execution_allocations[0].execution_id,
            result.executions[2].execution_id,
        )

    def test_exit_quote_clock_uses_specialized_query_with_exact_fallback_result(
        self,
    ) -> None:
        opened = observation(P1, 100)
        entry_fill_time = 200
        entry_hedge_t0 = entry_fill_time + HEDGE_DELAY_NS
        future_reopen_time = entry_hedge_t0 + 1_000_000
        trade_time = future_reopen_time + 10_000_000
        changes = {
            ("spot", P1): (book("spot", P1, 90, packet=980).event,),
            ("future", P1): (
                book(
                    "future",
                    P1,
                    90,
                    packet=981,
                    legal=False,
                ).event,
                book(
                    "future",
                    P1,
                    future_reopen_time,
                    packet=982,
                ).event,
            ),
        }
        fallback = QueryRiskBooks(changes)
        specialized = ExitQuoteQueryRiskBooks(changes)

        def replay(adapter: QueryRiskBooks) -> S1ReplayResult:
            return S1EventLoop(
                config(),
                (product(P1),),
                entry_state_adapter=TimelineStateAdapter({P1: (opened,)}),
                fill_adapter=RelativeFillAdapter({P1: 100}),
                risk_book_adapter=adapter,
                normal_exit_enabled=True,
                spot_trade_adapter=SyntheticSpotTrades((spot_trade(P1, trade_time),)),
            ).run((opened, SessionExpiry(EventCursor(20_000_000_000, 30, 0))))

        fallback_result = replay(fallback)
        specialized_result = replay(specialized)

        self.assertEqual(specialized_result, fallback_result)
        self.assertTrue(specialized.exit_quote_calls)
        self.assertEqual(
            {call[0] for call in specialized.exit_quote_calls},
            {"spot", "future"},
        )
        self.assertTrue(
            any(
                venue == "future" and after.recv_time_ns == entry_hedge_t0
                for venue, _, after, _ in specialized.next_calls
            )
        )
        exit_new = next(
            event
            for event in specialized_result.request_events
            if event.stage == "exit"
            and event.request_class == "new"
            and event.event_type == "actual_send"
        )
        self.assertEqual(exit_new.event_cursor.recv_time_ns, future_reopen_time + 1)
        self.assertEqual(
            specialized_result.exit_physical_fills[0].fill_cursor.recv_time_ns,
            trade_time,
        )
        self.assertEqual(specialized_result.positions[0].state, "exit_maker_flat")

    def test_future_exit_book_closure_cancels_passive_spot_before_later_trade(
        self,
    ) -> None:
        opened = observation(P1, 100)
        entry_fill_time = 200
        entry_hedge_time = entry_fill_time + HEDGE_DELAY_NS
        future_closes = entry_hedge_time + 10_000_000
        later_trade = future_closes + 10_000_000
        books = ExitQuoteQueryRiskBooks(
            {
                ("spot", P1): (book("spot", P1, 90, packet=990).event,),
                ("future", P1): (
                    book("future", P1, 90, packet=991).event,
                    book(
                        "future",
                        P1,
                        future_closes,
                        packet=992,
                        legal=False,
                    ).event,
                ),
            }
        )

        result = S1EventLoop(
            config(),
            (product(P1),),
            entry_state_adapter=TimelineStateAdapter({P1: (opened,)}),
            fill_adapter=RelativeFillAdapter({P1: 100}),
            risk_book_adapter=books,
            normal_exit_enabled=True,
            spot_trade_adapter=SyntheticSpotTrades((spot_trade(P1, later_trade),)),
        ).run((opened, SessionExpiry(EventCursor(20_000_000_000, 30, 0))))

        self.assertNotIn("exit_maker", [fact.role for fact in result.executions])
        self.assertEqual(result.positions[0].state, "paired_open")
        self.assertEqual(len(result.carry_out), 1)
        self.assertEqual(
            result.exit_desired_withdrawal_reason_counts,
            (("gate:future_empty_book_side", 1),),
        )
        self.assertTrue(
            any(
                event.stage == "exit"
                and event.request_class == "cancel"
                and event.event_type == "actual_send"
                for event in result.request_events
            )
        )
        self.assertTrue(
            any(call[0] == "future" for call in books.exit_quote_calls)
        )

    def test_direct_future_book_update_wakes_and_cancels_passive_spot(self) -> None:
        opened = observation(P1, 100)
        entry_hedge_time = 200 + HEDGE_DELAY_NS
        future_closes = entry_hedge_time + 10_000_000
        later_trade = future_closes + 10_000_000

        result = S1EventLoop(
            config(),
            (product(P1),),
            entry_state_adapter=TimelineStateAdapter({P1: (opened,)}),
            fill_adapter=RelativeFillAdapter({P1: 100}),
            normal_exit_enabled=True,
            spot_trade_adapter=SyntheticSpotTrades((spot_trade(P1, later_trade),)),
        ).run(
            (
                book("future", P1, 90, packet=993),
                book("spot", P1, 90, packet=994),
                opened,
                book(
                    "future",
                    P1,
                    future_closes,
                    packet=995,
                    legal=False,
                ),
                SessionExpiry(EventCursor(20_000_000_000, 30, 0)),
            )
        )

        self.assertNotIn("exit_maker", [fact.role for fact in result.executions])
        self.assertEqual(result.positions[0].state, "paired_open")
        self.assertEqual(
            result.exit_desired_withdrawal_reason_counts,
            (("gate:future_empty_book_side", 1),),
        )

    def test_future_last_legal_ask_wakes_and_cancels_passive_spot(self) -> None:
        opened = observation(P1, 100)
        entry_hedge_time = 200 + HEDGE_DELAY_NS
        future_reaches_last_legal_tick = entry_hedge_time + 10_000_000
        later_trade = future_reaches_last_legal_tick + 10_000_000

        result = S1EventLoop(
            config(),
            (product(P1),),
            entry_state_adapter=TimelineStateAdapter({P1: (opened,)}),
            fill_adapter=RelativeFillAdapter({P1: 100}),
            normal_exit_enabled=True,
            spot_trade_adapter=SyntheticSpotTrades((spot_trade(P1, later_trade),)),
        ).run(
            (
                # With ref=100 and the pre-2026-07-06 futures ladder, 107.5
                # is the final legal tick below the strict 108.0 upper band.
                book("future", P1, 90, packet=996, ask_price=107.0),
                book("spot", P1, 90, packet=997),
                opened,
                book(
                    "future",
                    P1,
                    future_reaches_last_legal_tick,
                    packet=998,
                    ask_price=107.5,
                ),
                SessionExpiry(EventCursor(20_000_000_000, 30, 0)),
            )
        )

        self.assertNotIn("exit_maker", [fact.role for fact in result.executions])
        self.assertEqual(result.positions[0].state, "paired_open")
        self.assertEqual(
            result.exit_desired_withdrawal_reason_counts,
            (("gate:future_upper_band_headroom_lt_1_tick", 1),),
        )
        self.assertTrue(
            any(
                event.stage == "exit"
                and event.request_class == "cancel"
                and event.event_type == "actual_send"
                for event in result.request_events
            )
        )

    def test_exit_cutoff_cancels_passive_order_but_same_cursor_fill_wins(self) -> None:
        opened = observation(P1, 100)
        entry_fill_time = 200
        entry_hedge_time = entry_fill_time + HEDGE_DELAY_NS
        cutoff_time = entry_hedge_time + 10_000_000

        result = S1EventLoop(
            config(),
            (product(P1),),
            entry_state_adapter=TimelineStateAdapter({P1: (opened,)}),
            fill_adapter=RelativeFillAdapter({P1: 100}),
            normal_exit_enabled=True,
            spot_trade_adapter=SyntheticSpotTrades((spot_trade(P1, cutoff_time),)),
        ).run(
            (
                book("future", P1, 90, packet=993),
                book("spot", P1, 90, packet=994),
                opened,
                ExitCutoff(EventCursor(cutoff_time, 30, 0)),
                ExitDrainBarrier(EventCursor(cutoff_time + 100_000_000, 30, 0)),
                SessionExpiry(EventCursor(20_000_000_000, 30, 0)),
            )
        )

        self.assertEqual(result.positions[0].state, "exit_maker_flat")
        self.assertIn("exit_maker", [fact.role for fact in result.executions])
        self.assertIn("exit_hedge", [fact.role for fact in result.executions])
        cancel = next(
            event
            for event in result.request_events
            if event.stage == "exit"
            and event.request_class == "cancel"
            and event.event_type == "actual_send"
        )
        self.assertEqual(cancel.event_cursor.recv_time_ns, cutoff_time)
        self.assertEqual(
            result.exit_desired_withdrawal_reason_counts,
            (("safety_cutoff", 1),),
        )
        self.assertTrue(result.exit_cutoff_applied)
        self.assertTrue(result.exit_drain_barrier_applied)

    def test_exit_drain_barrier_fails_if_cancel_has_not_actually_sent(self) -> None:
        opened = observation(P1, 100)
        exit_new_time = 1_000_000_100
        cutoff_time = exit_new_time + 10_000_000
        barrier_time = cutoff_time + 10_000_000

        loop = S1EventLoop(
            config(spot_request_cap=1),
            (product(P1),),
            entry_state_adapter=TimelineStateAdapter({P1: (opened,)}),
            fill_adapter=RelativeFillAdapter({P1: 100}),
            normal_exit_enabled=True,
            spot_trade_adapter=SyntheticSpotTrades(()),
        )
        with self.assertRaisesRegex(RuntimeError, "exit drain barrier violated"):
            loop.run(
                (
                    book("future", P1, 90, packet=996),
                    book("spot", P1, 90, packet=997),
                    opened,
                    ExitCutoff(EventCursor(cutoff_time, 30, 0)),
                    ExitDrainBarrier(EventCursor(barrier_time, 30, 0)),
                    SessionExpiry(EventCursor(20_000_000_000, 30, 0)),
                )
            )


    def test_exit_refresh_builds_one_target_per_cursor_and_frozen_lower(self) -> None:
        first = observation(P1, 100, snapshot=601)
        second_time = HEDGE_DELAY_NS + 10_000_000
        second = observation(P1, second_time, snapshot=602)

        class CountingEventLoop(S1EventLoop):
            target_builds: list[tuple[EventCursor, float, str]]

            def _build_exit_target(
                self,
                position_id: str,
                cursor: EventCursor,
                spot_book: CausalBookState | None,
                future_book: CausalBookState | None,
            ) -> S1SpotAskTarget:
                threshold = self._positions[position_id].frozen_exit_threshold_basis_bp
                assert threshold is not None
                self.target_builds.append((cursor, threshold, position_id))
                return super()._build_exit_target(
                    position_id,
                    cursor,
                    spot_book,
                    future_book,
                )

        def replay(loop_type: type[S1EventLoop]) -> tuple[S1ReplayResult, S1EventLoop]:
            loop = loop_type(
                config(),
                (product(P1),),
                entry_state_adapter=TimelineStateAdapter({P1: (first, second)}),
                fill_adapter=RelativeFillAdapter({P1: 100}),
                normal_exit_enabled=True,
                spot_trade_adapter=SyntheticSpotTrades(()),
            )
            if isinstance(loop, CountingEventLoop):
                loop.target_builds = []
            result = loop.run(
                (
                    book("future", P1, 90, packet=983),
                    book("spot", P1, 90, packet=984),
                    first,
                    second,
                    SessionExpiry(EventCursor(200_000_000, 30, 0)),
                )
            )
            return result, loop

        expected, _ = replay(S1EventLoop)
        actual, counted = replay(CountingEventLoop)
        assert isinstance(counted, CountingEventLoop)

        self.assertEqual(actual, expected)
        self.assertEqual(len(actual.positions), 1)
        build_keys = [
            (cursor, threshold) for cursor, threshold, _ in counted.target_builds
        ]
        self.assertEqual(len(build_keys), len(set(build_keys)))
        self.assertLess(len(counted.target_builds), len(actual.exit_inventory_facts))

    def test_day_one_checkpoint_rebuilds_carry_and_releases_only_day_two_delta(
        self,
    ) -> None:
        accounting = S1AccountingBridge(
            default_date=DATE,
            scenario_id="q95",
            products=(S1AccountingProduct(P1, P1, 2_000),),
            execution_date_resolver=(
                lambda cursor: DATE if cursor.recv_time_ns < D2_OPEN_NS else NEXT_DATE
            ),
        )
        day_one, day_one_ledger = self._day_one_carry(accounting=accounting)
        seed = day_one_ledger.to_seed(DATE)
        restored = CapacityLedger.from_seed(seed)
        trade_time = D2_OPEN_NS + 10_000_000

        day_two = S1EventLoop(
            config(date=NEXT_DATE),
            (product(P1),),
            entry_state_adapter=TimelineStateAdapter({}),
            normal_exit_enabled=True,
            spot_trade_adapter=SyntheticSpotTrades((spot_trade(P1, trade_time),)),
            accounting_adapter=accounting,
            capacity_ledger=restored,
            carry_in=day_one.carry_out,
            day_open_time_ns=D2_OPEN_NS,
        ).run(
            (
                book("future", P1, D2_OPEN_NS, packet=910),
                book("spot", P1, D2_OPEN_NS, packet=911),
            )
        )

        self.assertEqual(day_two.carry_in, day_one.carry_out)
        self.assertEqual(day_two.carry_out, ())
        self.assertEqual(day_two.entry_disabled_product_ids, frozenset({P1}))
        self.assertEqual(day_two.orders, ())
        self.assertEqual(day_two.positions[0].state, "exit_maker_flat")
        self.assertEqual(
            [row.event_type for row in day_two.capacity_transitions],
            ["exit_started", "exit_hedge_complete"],
        )
        self.assertEqual(
            day_two.capacity_transitions[0].sequence,
            seed.transition_sequence_offset + 1,
        )
        self.assertEqual(day_two.capacity_transitions, restored.transitions)
        self.assertEqual(restored.global_balances.total_committed_notional_twd, 0)
        day_one_exit_raw = next(
            event.raw_order_fact_id
            for event in day_one.order_events
            if event.stage == "exit" and event.event_type == "actual_new_working"
        )
        day_two_exit_raw = next(
            event.raw_order_fact_id
            for event in day_two.order_events
            if event.stage == "exit" and event.event_type == "actual_new_working"
        )
        self.assertNotEqual(day_one_exit_raw, day_two_exit_raw)
        all_execution_ids = [
            fact.execution_id for fact in (*day_one.executions, *day_two.executions)
        ]
        self.assertEqual(len(all_execution_ids), len(set(all_execution_ids)))
        terminal = next(
            fact
            for fact in accounting.facts
            if isinstance(fact, TerminalRealizedAccounting)
        )
        self.assertEqual(terminal.terminal_date, NEXT_DATE)
        accounting.verify()

    def test_compact_checkpoint_validates_carry_through_accounting_provenance(
        self,
    ) -> None:
        accounting = S1AccountingBridge(
            default_date=DATE,
            scenario_id="q95",
            products=(S1AccountingProduct(P1, P1, 2_000),),
            execution_date_resolver=(
                lambda cursor: DATE if cursor.recv_time_ns < D2_OPEN_NS else NEXT_DATE
            ),
        )
        day_one, ledger = self._day_one_carry(accounting=accounting)
        receipt = CapacityIdentityRegistryReceipt(
            schema_version=1,
            transition_count=len(ledger.transitions),
            admitted_count=sum(
                transition.event_type == "new_reservation_attempt"
                and transition.admitted is True
                for transition in ledger.transitions
            ),
            registry_sha256="1" * 64,
        )
        restored = CapacityLedger.from_compact_checkpoint(
            ledger.to_compact_checkpoint(DATE, identity_registry_receipt=receipt)
        )

        loop = S1EventLoop(
            config(date=NEXT_DATE),
            (product(P1),),
            entry_state_adapter=TimelineStateAdapter({}),
            normal_exit_enabled=True,
            spot_trade_adapter=SyntheticSpotTrades(()),
            accounting_adapter=accounting,
            capacity_ledger=restored,
            carry_in=day_one.carry_out,
            day_open_time_ns=D2_OPEN_NS,
        )
        self.assertTrue(restored.historical_transition_identities_omitted)
        self.assertEqual(loop.carry_in, day_one.carry_out)
        self.assertEqual(restored.transitions, ())

        carry = day_one.carry_out[0]
        with self.assertRaisesRegex(ValueError, "replayed accounting facts"):
            S1EventLoop(
                config(date=NEXT_DATE),
                (product(P1),),
                entry_state_adapter=TimelineStateAdapter({}),
                normal_exit_enabled=True,
                spot_trade_adapter=SyntheticSpotTrades(()),
                capacity_ledger=CapacityLedger.from_compact_checkpoint(
                    ledger.to_compact_checkpoint(
                        DATE, identity_registry_receipt=receipt
                    )
                ),
                carry_in=day_one.carry_out,
                day_open_time_ns=D2_OPEN_NS,
            )

        tampered_fact = replace(
            carry.position_established_fact,
            sequence=carry.position_established_fact.sequence + 1,
        )
        with self.assertRaisesRegex(ValueError, "differs from accounting history"):
            S1EventLoop(
                config(date=NEXT_DATE),
                (product(P1),),
                entry_state_adapter=TimelineStateAdapter({}),
                normal_exit_enabled=True,
                spot_trade_adapter=SyntheticSpotTrades(()),
                accounting_adapter=accounting,
                capacity_ledger=CapacityLedger.from_compact_checkpoint(
                    ledger.to_compact_checkpoint(
                        DATE, identity_registry_receipt=receipt
                    )
                ),
                carry_in=(replace(carry, position_established_fact=tampered_fact),),
                day_open_time_ns=D2_OPEN_NS,
            )

    def test_contract_expiry_marks_exact_pair_without_execution_or_cost(self) -> None:
        close_time = D2_OPEN_NS + 100
        accounting = S1AccountingBridge(
            default_date=DATE,
            scenario_id="q95",
            products=(S1AccountingProduct(P1, P1, 2_000),),
            execution_date_resolver=(
                lambda cursor: DATE if cursor.recv_time_ns < D2_OPEN_NS else NEXT_DATE
            ),
        )
        day_one, day_one_ledger = self._day_one_carry(accounting=accounting)
        restored = CapacityLedger.from_seed(day_one_ledger.to_seed(DATE))
        close = official_close(P1, close_time, date=NEXT_DATE, close_price=103.5)

        result = S1EventLoop(
            config(date=NEXT_DATE),
            (product(P1, end_date=NEXT_DATE),),
            entry_state_adapter=TimelineStateAdapter({}),
            normal_exit_enabled=True,
            spot_trade_adapter=SyntheticSpotTrades(()),
            accounting_adapter=accounting,
            capacity_ledger=restored,
            carry_in=day_one.carry_out,
            day_open_time_ns=D2_OPEN_NS,
        ).run((ContractExpiry(close),))

        self.assertEqual(result.executions, ())
        self.assertEqual(result.request_events, ())
        self.assertEqual(result.carry_out, ())
        self.assertEqual(result.positions[0].state, "expiry_basis_zero_accounting")
        self.assertEqual(result.expiry_mark_count, 1)
        self.assertEqual(result.expiry_marks[0].spot_close_source_id, close.source_id)
        release = result.capacity_transitions[0]
        self.assertEqual(release.event_type, "expiry_basis_zero_release")
        release_cursor = EventCursor(
            release.timestamp_ns,
            release.event_sequence,
            release.row_index,
        )
        self.assertLess(close.source_cursor, release_cursor)
        self.assertLess(release_cursor, result.expiry_marks[0].cursor)
        mark = next(
            fact for fact in accounting.facts if isinstance(fact, ExpiryAccountingMark)
        )
        self.assertTrue(mark.non_executable)
        self.assertIsNone(mark.request_id)
        self.assertIsNone(mark.execution_id)
        self.assertEqual(mark.synthetic_commission_twd, 0)
        self.assertEqual(mark.synthetic_tax_twd, 0)
        self.assertEqual(mark.spot_close_source_cursor, close.source_cursor)
        report = verify_s1_accounting_capacity_links(
            (mark,),
            result.capacity_transitions,
        )
        self.assertEqual(report.expiry_marks, 1)
        self.assertEqual(report.linked_capacity_transitions, 1)
        accounting.verify()

    def test_contract_expiry_one_close_marks_multiple_positions_uniquely(self) -> None:
        first = observation(P1, D1_OPEN_NS, tick=1_000, snapshot=51)
        second = observation(P1, D1_OPEN_NS + 1_000, tick=1_001, snapshot=52)
        accounting = S1AccountingBridge(
            default_date=DATE,
            scenario_id="q95",
            products=(S1AccountingProduct(P1, P1, 2_000),),
            execution_date_resolver=(
                lambda cursor: DATE if cursor.recv_time_ns < D2_OPEN_NS else NEXT_DATE
            ),
        )
        ledger = CapacityLedger(
            global_cap_twd=20_000,
            product_cap_twd=10_000,
        )
        day_one = S1EventLoop(
            config(),
            (product(P1, session_end=D2_OPEN_NS - 1),),
            entry_state_adapter=TimelineStateAdapter({P1: (first, second)}),
            fill_adapter=RelativeFillAdapter({P1: 100}),
            normal_exit_enabled=True,
            spot_trade_adapter=SyntheticSpotTrades(()),
            accounting_adapter=accounting,
            capacity_ledger=ledger,
        ).run(
            (
                book("future", P1, D1_OPEN_NS - 10, packet=950),
                book("spot", P1, D1_OPEN_NS - 10, packet=951),
                first,
                second,
            )
        )
        self.assertEqual(len(day_one.carry_out), 2)
        restored = CapacityLedger.from_seed(ledger.to_seed(DATE))
        close = official_close(P1, D2_OPEN_NS + 100, date=NEXT_DATE)

        day_two = S1EventLoop(
            config(date=NEXT_DATE),
            (product(P1, end_date=NEXT_DATE),),
            entry_state_adapter=TimelineStateAdapter({}),
            normal_exit_enabled=True,
            spot_trade_adapter=SyntheticSpotTrades(()),
            accounting_adapter=accounting,
            capacity_ledger=restored,
            carry_in=day_one.carry_out,
            day_open_time_ns=D2_OPEN_NS,
        ).run((ContractExpiry(close),))

        self.assertEqual(day_two.expiry_mark_count, 2)
        self.assertEqual(
            len({mark.mark_id for mark in day_two.expiry_marks}),
            2,
        )
        self.assertEqual(
            len({mark.capacity_release_transition_id for mark in day_two.expiry_marks}),
            2,
        )
        self.assertLess(day_two.expiry_marks[0].cursor, day_two.expiry_marks[1].cursor)
        marks = tuple(
            fact for fact in accounting.facts if isinstance(fact, ExpiryAccountingMark)
        )
        report = verify_s1_accounting_capacity_links(
            marks,
            day_two.capacity_transitions,
        )
        self.assertEqual(report.expiry_marks, 2)
        self.assertEqual(restored.global_balances.total_committed_notional_twd, 0)

    def test_same_timestamp_contract_expiries_share_one_capacity_replay(self) -> None:
        first = observation(P1, D1_OPEN_NS, row=0, snapshot=61)
        second = observation(P2, D1_OPEN_NS, row=1, snapshot=62)
        third = observation(P3, D1_OPEN_NS, row=2, snapshot=63)
        accounting_products = (
            S1AccountingProduct(P1, P1, 2_000),
            S1AccountingProduct(P2, P2, 2_000),
            S1AccountingProduct(P3, P3, 2_000),
        )
        execution_date_resolver = (
            lambda cursor: DATE if cursor.recv_time_ns < D2_OPEN_NS else NEXT_DATE
        )
        accounting = S1AccountingBridge(
            default_date=DATE,
            scenario_id="q95",
            products=accounting_products,
            execution_date_resolver=execution_date_resolver,
        )
        ledger = CapacityLedger(
            global_cap_twd=20_000,
            product_cap_twd=10_000,
        )
        day_one = S1EventLoop(
            config(),
            (
                product(P1, session_end=D2_OPEN_NS - 1),
                product(P2, session_end=D2_OPEN_NS - 1),
                product(P3, session_end=D2_OPEN_NS - 1),
            ),
            entry_state_adapter=TimelineStateAdapter(
                {P1: (first,), P2: (second,), P3: (third,)}
            ),
            fill_adapter=RelativeFillAdapter({P1: 100, P2: 100, P3: 100}),
            normal_exit_enabled=True,
            spot_trade_adapter=SyntheticSpotTrades(()),
            accounting_adapter=accounting,
            capacity_ledger=ledger,
        ).run(
            (
                book("future", P1, D1_OPEN_NS - 10, row=0, packet=970),
                book("spot", P1, D1_OPEN_NS - 10, row=1, packet=971),
                book("future", P2, D1_OPEN_NS - 10, row=2, packet=972),
                book("spot", P2, D1_OPEN_NS - 10, row=3, packet=973),
                book("future", P3, D1_OPEN_NS - 10, row=4, packet=974),
                book("spot", P3, D1_OPEN_NS - 10, row=5, packet=975),
                first,
                second,
                third,
            )
        )
        self.assertEqual(len(day_one.carry_out), 3)

        seed = ledger.to_seed(DATE)
        day_one_facts = accounting.facts
        reference_accounting = S1AccountingBridge.from_facts(
            default_date=DATE,
            scenario_id="q95",
            products=accounting_products,
            facts=day_one_facts,
            execution_date_resolver=execution_date_resolver,
        )
        restored = CapacityLedger.from_seed(seed)
        loop = S1EventLoop(
            config(date=NEXT_DATE),
            (
                product(P1, end_date=NEXT_DATE),
                product(P2, end_date=NEXT_DATE),
                product(P3, end_date=NEXT_DATE),
            ),
            entry_state_adapter=TimelineStateAdapter({}),
            normal_exit_enabled=True,
            spot_trade_adapter=SyntheticSpotTrades(()),
            accounting_adapter=accounting,
            capacity_ledger=restored,
            carry_in=day_one.carry_out,
            day_open_time_ns=D2_OPEN_NS,
        )
        close_time = D2_OPEN_NS + 100
        expiry_events = (
            ContractExpiry(
                official_close(
                    P1,
                    close_time,
                    date=NEXT_DATE,
                    row=0,
                )
            ),
            ContractExpiry(
                official_close(
                    P2,
                    close_time,
                    date=NEXT_DATE,
                    row=1,
                )
            ),
            ContractExpiry(
                official_close(
                    P3,
                    close_time,
                    date=NEXT_DATE,
                    row=2,
                )
            ),
        )
        with patch.object(restored, "verify", wraps=restored.verify) as verify:
            day_two = loop.run(expiry_events)

        reference_restored = CapacityLedger.from_seed(seed)
        reference_loop = S1EventLoop(
            config(date=NEXT_DATE),
            (
                product(P1, end_date=NEXT_DATE),
                product(P2, end_date=NEXT_DATE),
                product(P3, end_date=NEXT_DATE),
            ),
            entry_state_adapter=TimelineStateAdapter({}),
            normal_exit_enabled=True,
            spot_trade_adapter=SyntheticSpotTrades(()),
            accounting_adapter=reference_accounting,
            capacity_ledger=reference_restored,
            carry_in=day_one.carry_out,
            day_open_time_ns=D2_OPEN_NS,
        )
        reference_handler = reference_loop._process_contract_expiry

        def process_with_per_event_verify(
            event: ContractExpiry,
            timestamp_ns: int,
            *,
            replay: object,
        ) -> object:
            del replay
            return reference_handler(
                event,
                timestamp_ns,
                replay=reference_restored.verify(),
            )

        with patch.object(
            reference_loop,
            "_process_contract_expiry",
            side_effect=process_with_per_event_verify,
        ):
            reference_day_two = reference_loop.run(expiry_events)

        self.assertEqual(verify.call_count, 3)
        self.assertEqual(day_two, reference_day_two)
        self.assertEqual(accounting.facts, reference_accounting.facts)
        self.assertEqual(
            restored.verify().transition_chain_sha256,
            reference_restored.verify().transition_chain_sha256,
        )
        self.assertEqual(day_two.expiry_mark_count, 3)
        self.assertEqual(day_two.carry_out, ())
        self.assertEqual(
            [row.event_type for row in day_two.capacity_transitions],
            [
                "expiry_basis_zero_release",
                "expiry_basis_zero_release",
                "expiry_basis_zero_release",
            ],
        )
        self.assertEqual(
            {mark.product_id for mark in day_two.expiry_marks},
            {P1, P2, P3},
        )
        accounting.verify()

    def test_contract_expiry_rejects_metadata_or_missing_accounting(self) -> None:
        day_one, ledger = self._day_one_carry()
        seed = ledger.to_seed(DATE)

        cases = (
            (
                "date differs",
                product(P1, end_date=NEXT_DATE),
                official_close(P1, D2_OPEN_NS + 100, date=DATE),
            ),
            (
                "end_date",
                product(P1, end_date=DATE),
                official_close(P1, D2_OPEN_NS + 100, date=NEXT_DATE),
            ),
            (
                "end_date",
                product(P1),
                official_close(P1, D2_OPEN_NS + 100, date=NEXT_DATE),
            ),
            (
                "quote differs",
                product(P1, end_date=NEXT_DATE),
                official_close(
                    P1,
                    D2_OPEN_NS + 100,
                    date=NEXT_DATE,
                    quote_code="HCFD6",
                ),
            ),
        )
        for message, expiry_product, close in cases:
            with self.subTest(message=message):
                restored = CapacityLedger.from_seed(seed)
                loop = S1EventLoop(
                    config(date=NEXT_DATE),
                    (expiry_product,),
                    entry_state_adapter=TimelineStateAdapter({}),
                    normal_exit_enabled=True,
                    spot_trade_adapter=SyntheticSpotTrades(()),
                    capacity_ledger=restored,
                    carry_in=day_one.carry_out,
                    day_open_time_ns=D2_OPEN_NS,
                )
                with self.assertRaisesRegex(ValueError, message):
                    loop.run((ContractExpiry(close),))
                self.assertEqual(restored.transitions, ())

        restored = CapacityLedger.from_seed(seed)
        loop = S1EventLoop(
            config(date=NEXT_DATE),
            (product(P1, end_date=NEXT_DATE),),
            entry_state_adapter=TimelineStateAdapter({}),
            normal_exit_enabled=True,
            spot_trade_adapter=SyntheticSpotTrades(()),
            capacity_ledger=restored,
            carry_in=day_one.carry_out,
            day_open_time_ns=D2_OPEN_NS,
        )
        with self.assertRaisesRegex(ValueError, "accounting_adapter"):
            loop.run(
                (ContractExpiry(official_close(P1, D2_OPEN_NS + 100, date=NEXT_DATE)),)
            )
        self.assertEqual(restored.transitions, ())

    def test_contract_expiry_rejects_exit_in_progress_before_release(self) -> None:
        opened = observation(P1, 100)
        trade_time = 60_000_000
        close_time = trade_time + 1
        ledger = CapacityLedger(
            global_cap_twd=20_000,
            product_cap_twd=10_000,
        )
        accounting = S1AccountingBridge(
            default_date=DATE,
            scenario_id="q95",
            products=(S1AccountingProduct(P1, P1, 2_000),),
        )
        loop = S1EventLoop(
            config(),
            (product(P1, end_date=DATE),),
            entry_state_adapter=TimelineStateAdapter({P1: (opened,)}),
            fill_adapter=RelativeFillAdapter({P1: 100}),
            normal_exit_enabled=True,
            spot_trade_adapter=SyntheticSpotTrades((spot_trade(P1, trade_time),)),
            accounting_adapter=accounting,
            capacity_ledger=ledger,
        )

        with self.assertRaisesRegex(ValueError, "entirely paired-open"):
            loop.run(
                (
                    book("future", P1, 90, packet=960),
                    book("spot", P1, 90, packet=961),
                    opened,
                    ContractExpiry(official_close(P1, close_time)),
                )
            )
        self.assertEqual(ledger.global_balances.exit_in_progress, 5_000)
        self.assertNotIn(
            "expiry_basis_zero_release",
            [row.event_type for row in ledger.transitions],
        )
        self.assertFalse(
            any(isinstance(fact, ExpiryAccountingMark) for fact in accounting.facts)
        )

    def test_day_open_trade_cannot_backfill_a_fresh_carry_order(self) -> None:
        day_one, day_one_ledger = self._day_one_carry()
        restored = CapacityLedger.from_seed(day_one_ledger.to_seed(DATE))
        day_two = S1EventLoop(
            config(date=NEXT_DATE),
            (product(P1),),
            entry_state_adapter=TimelineStateAdapter({}),
            normal_exit_enabled=True,
            spot_trade_adapter=SyntheticSpotTrades((spot_trade(P1, D2_OPEN_NS),)),
            capacity_ledger=restored,
            carry_in=day_one.carry_out,
            day_open_time_ns=D2_OPEN_NS,
        ).run(
            (
                book("future", P1, D2_OPEN_NS, packet=920),
                book("spot", P1, D2_OPEN_NS, packet=921),
            )
        )

        self.assertEqual(day_two.exit_physical_fills, ())
        self.assertEqual(day_two.capacity_transitions, ())
        self.assertEqual(day_two.positions[0].state, "paired_open")
        self.assertEqual(day_two.carry_out, day_two.carry_in)
        exit_new = next(
            event
            for event in day_two.request_events
            if event.stage == "exit" and event.event_type == "actual_send"
        )
        self.assertEqual(exit_new.event_cursor.recv_time_ns, D2_OPEN_NS)

    def test_opening_carry_product_stays_entry_disabled_after_it_flattens(
        self,
    ) -> None:
        day_one, day_one_ledger = self._day_one_carry()
        restored = CapacityLedger.from_seed(day_one_ledger.to_seed(DATE))
        trade_time = D2_OPEN_NS + 10_000_000
        later = trade_time + HEDGE_DELAY_NS + 10_000_000
        p1_entry = observation(P1, later, snapshot=31)
        p2_entry = observation(P2, later + 1, snapshot=32)

        day_two = S1EventLoop(
            config(date=NEXT_DATE),
            (product(P1), product(P2)),
            entry_state_adapter=TimelineStateAdapter(
                {P1: (p1_entry,), P2: (p2_entry,)}
            ),
            fill_adapter=RelativeFillAdapter({P1: 100, P2: 100}),
            normal_exit_enabled=True,
            spot_trade_adapter=SyntheticSpotTrades((spot_trade(P1, trade_time),)),
            capacity_ledger=restored,
            carry_in=day_one.carry_out,
            day_open_time_ns=D2_OPEN_NS,
        ).run(
            (
                book("future", P1, D2_OPEN_NS, packet=930),
                book("spot", P1, D2_OPEN_NS, packet=931),
                book("future", P2, D2_OPEN_NS, packet=932),
                book("spot", P2, D2_OPEN_NS, packet=933),
                p1_entry,
                p2_entry,
            )
        )

        self.assertEqual(day_two.entry_disabled_product_ids, frozenset({P1}))
        self.assertEqual([order.product_id for order in day_two.orders], [P2])
        states = {position.product_id: position.state for position in day_two.positions}
        self.assertEqual(states[P1], "exit_maker_flat")
        self.assertEqual(states[P2], "paired_open")
        self.assertEqual([carry.product_id for carry in day_two.carry_out], [P2])

    def test_carry_constructor_rejects_incomplete_or_mismatched_restart(self) -> None:
        day_one, day_one_ledger = self._day_one_carry()
        restored = CapacityLedger.from_seed(day_one_ledger.to_seed(DATE))
        common = {
            "config": config(date=NEXT_DATE),
            "products": (product(P1),),
            "entry_state_adapter": TimelineStateAdapter({}),
            "normal_exit_enabled": True,
            "spot_trade_adapter": SyntheticSpotTrades(()),
            "day_open_time_ns": D2_OPEN_NS,
        }
        with self.assertRaisesRegex(ValueError, "match carry_in exactly"):
            S1EventLoop(capacity_ledger=restored, carry_in=(), **common)
        with self.assertRaisesRegex(ValueError, "quote_code"):
            S1EventLoop(
                capacity_ledger=restored,
                carry_in=(replace(day_one.carry_out[0], quote_code="wrong"),),
                **common,
            )
        with self.assertRaisesRegex(ValueError, "empty daily transition delta"):
            S1EventLoop(
                capacity_ledger=day_one_ledger,
                carry_in=day_one.carry_out,
                **common,
            )
        with self.assertRaisesRegex(ValueError, "normal_exit_enabled"):
            S1EventLoop(
                config(date=NEXT_DATE),
                (product(P1),),
                entry_state_adapter=TimelineStateAdapter({}),
                capacity_ledger=restored,
                carry_in=day_one.carry_out,
                day_open_time_ns=D2_OPEN_NS,
            )
        with self.assertRaisesRegex(ValueError, "day_open_time_ns"):
            S1EventLoop(
                config(date=NEXT_DATE),
                (product(P1),),
                entry_state_adapter=TimelineStateAdapter({}),
                normal_exit_enabled=True,
                spot_trade_adapter=SyntheticSpotTrades(()),
                capacity_ledger=restored,
                carry_in=day_one.carry_out,
            )

    def test_multi_product_exit_identities_are_globally_unique(self) -> None:
        first = observation(P1, 100, snapshot=41)
        second = observation(P2, 101, snapshot=42)
        trade_time = 60_000_000
        accounting = S1AccountingBridge(
            default_date=DATE,
            scenario_id="q95",
            products=(
                S1AccountingProduct(P1, P1, 2_000),
                S1AccountingProduct(P2, P2, 2_000),
            ),
        )
        trades = PriceAwareSpotTrades(
            (
                spot_trade(P1, trade_time, row=1),
                spot_trade(P2, trade_time, row=2),
            )
        )
        result = S1EventLoop(
            config(),
            (product(P1), product(P2)),
            entry_state_adapter=TimelineStateAdapter({P1: (first,), P2: (second,)}),
            fill_adapter=RelativeFillAdapter({P1: 100, P2: 99}),
            normal_exit_enabled=True,
            spot_trade_adapter=trades,
            accounting_adapter=accounting,
        ).run(
            (
                book("future", P1, 90, packet=940),
                book("spot", P1, 90, packet=941),
                book("future", P2, 90, packet=942),
                book("spot", P2, 90, packet=943),
                first,
                second,
            )
        )

        execution_ids = [fact.execution_id for fact in result.executions]
        physical_ids = [fact.fill_id for fact in result.exit_physical_fills]
        allocation_ids = [
            fact.domain_allocation_id
            for fact in result.executions
            if fact.role == "exit_maker"
        ]
        self.assertEqual(len(execution_ids), len(set(execution_ids)))
        self.assertEqual(len(physical_ids), len(set(physical_ids)))
        self.assertEqual(len(allocation_ids), len(set(allocation_ids)))
        self.assertEqual(len(physical_ids), 2)
        self.assertEqual(
            trades.trades_at_calls,
            [(product_id, trade_time) for product_id in sorted((P1, P2))],
        )
        accounting.verify()

    def test_normal_exit_rejects_multi_contract_position_contract(self) -> None:
        multi = S1Product(
            P1,
            P1,
            "CDFE6",
            contract_size_shares=2_000,
            future_contracts=2,
            future_session_end_time_ns=20_000_000_000,
            spot_session_end_time_ns=20_000_000_000,
        )
        with self.assertRaisesRegex(ValueError, "one futures contract"):
            S1EventLoop(
                config(),
                (multi,),
                entry_state_adapter=TimelineStateAdapter({}),
                normal_exit_enabled=True,
                spot_trade_adapter=SyntheticSpotTrades(()),
            )

    def test_exit_new_cannot_backfill_a_same_timestamp_trade(self) -> None:
        opened = observation(P1, 100)
        entry_fill_time = 200
        entry_hedge_time = entry_fill_time + HEDGE_DELAY_NS
        exit_new_time = entry_hedge_time + 1
        trade = spot_trade(P1, exit_new_time)

        result = S1EventLoop(
            config(),
            (product(P1),),
            entry_state_adapter=TimelineStateAdapter({P1: (opened,)}),
            fill_adapter=RelativeFillAdapter({P1: 100}),
            normal_exit_enabled=True,
            spot_trade_adapter=SyntheticSpotTrades((trade,)),
        ).run(
            (
                book("future", P1, 90, packet=20),
                book("spot", P1, 90, packet=21),
                opened,
            )
        )

        self.assertEqual(result.exit_physical_fills, ())
        self.assertEqual(result.positions[0].state, "paired_open")
        self.assertEqual(result.positions[0].exit_unresolved_shares, 2_000)
        exit_new = next(
            event
            for event in result.request_events
            if event.stage == "exit" and event.event_type == "actual_send"
        )
        self.assertEqual(exit_new.event_cursor.recv_time_ns, trade.cursor.recv_time_ns)

    def test_price_aware_trade_wake_skips_prints_below_active_exit_target(
        self,
    ) -> None:
        opened = observation(P1, 100)
        entry_fill_time = 200
        entry_hedge_time = entry_fill_time + HEDGE_DELAY_NS
        low_time = entry_hedge_time + 5_000_000
        high_time = entry_hedge_time + 10_000_000
        trades = PriceAwareSpotTrades(
            (
                spot_trade(P1, low_time, price=100.0),
                spot_trade(P1, high_time, price=102.0),
            )
        )

        result = S1EventLoop(
            config(),
            (product(P1),),
            entry_state_adapter=TimelineStateAdapter({P1: (opened,)}),
            fill_adapter=RelativeFillAdapter({P1: 100}),
            normal_exit_enabled=True,
            spot_trade_adapter=trades,
        ).run(
            (
                book("future", P1, 90, packet=970),
                book("spot", P1, 90, packet=971),
                opened,
            )
        )

        self.assertTrue(trades.price_aware_calls)
        self.assertEqual(trades.generic_calls, 0)
        self.assertGreater(trades.price_aware_calls[0][3], 100.0)
        self.assertNotIn(low_time, trades.trades_at_times)
        self.assertIn(high_time, trades.trades_at_times)
        self.assertEqual(result.positions[0].state, "exit_maker_flat")

    def test_exit_trade_wake_queries_only_bound_product_and_matches_legacy_poll(
        self,
    ) -> None:
        opened = observation(P1, 100)
        entry_hedge_time = 200 + HEDGE_DELAY_NS
        trade_time = entry_hedge_time + 10_000_000
        noise_times = (
            entry_hedge_time + 2_000_000,
            entry_hedge_time + 4_000_000,
            entry_hedge_time + 6_000_000,
        )

        class LegacyTradePollingLoop(S1EventLoop):
            def _process_fills(self, timestamp_ns: int) -> None:
                for product_id, allocator in self._exit_fill_allocators.items():
                    if allocator.active_raw_order_ids:
                        self._exit_trade_probe_products[timestamp_ns].add(product_id)
                super()._process_fills(timestamp_ns)

        def replay(
            loop_type: type[S1EventLoop],
        ) -> tuple[S1ReplayResult, PriceAwareSpotTrades]:
            trades = PriceAwareSpotTrades((spot_trade(P1, trade_time),))
            loop = loop_type(
                config(),
                (product(P1),),
                entry_state_adapter=TimelineStateAdapter({P1: (opened,)}),
                fill_adapter=RelativeFillAdapter({P1: 100}),
                normal_exit_enabled=True,
                spot_trade_adapter=trades,
            )
            result = loop.run(
                (
                    book("future", P1, 90, packet=985),
                    book("spot", P1, 90, packet=986),
                    opened,
                    *(
                        book("spot", P1, wake, packet=987 + index)
                        for index, wake in enumerate(noise_times)
                    ),
                )
            )
            return result, trades

        optimized_result, optimized_trades = replay(S1EventLoop)
        legacy_result, legacy_trades = replay(LegacyTradePollingLoop)

        self.assertEqual(optimized_result, legacy_result)
        self.assertEqual(optimized_trades.trades_at_calls, [(P1, trade_time)])
        self.assertGreater(
            len(legacy_trades.trades_at_calls),
            len(optimized_trades.trades_at_calls),
        )
        self.assertLess(
            len(optimized_trades.price_aware_calls),
            len(legacy_trades.price_aware_calls),
        )

    def test_frozen_exit_price_does_not_cancel_when_future_ask_moves(self) -> None:
        opened = observation(P1, 100)
        entry_fill_time = 200
        entry_hedge_time = entry_fill_time + HEDGE_DELAY_NS
        trade_time = entry_hedge_time + 10_000_000
        trade = spot_trade(P1, trade_time, price=102.0)

        result = S1EventLoop(
            config(),
            (product(P1),),
            entry_state_adapter=TimelineStateAdapter({P1: (opened,)}),
            fill_adapter=RelativeFillAdapter({P1: 100}),
            normal_exit_enabled=True,
            spot_trade_adapter=SyntheticSpotTrades((trade,)),
        ).run(
            (
                book("future", P1, 90, packet=30),
                book("spot", P1, 90, packet=31),
                opened,
                book(
                    "future",
                    P1,
                    trade_time,
                    packet=32,
                    bid_price=100.0,
                    ask_price=102.0,
                ),
            )
        )

        cancels = [
            event
            for event in result.request_events
            if event.stage == "exit"
            and event.request_class == "cancel"
            and event.event_type == "actual_send"
        ]
        self.assertEqual(cancels, [])
        exit_fill = next(
            fact for fact in result.executions if fact.role == "exit_maker"
        )
        self.assertEqual(exit_fill.cursor.event_sequence, 300)
        self.assertEqual(result.positions[0].state, "exit_maker_flat")

    def test_exit_future_quota_delay_revalidates_and_uses_shared_token_clock(
        self,
    ) -> None:
        opened = observation(P1, 100)
        entry_fill_time = 200
        entry_hedge_time = entry_fill_time + HEDGE_DELAY_NS
        trade_time = entry_hedge_time + 10_000_000
        exit_t0 = trade_time + HEDGE_DELAY_NS
        expected_send = entry_hedge_time + 1_000_000_000

        result = S1EventLoop(
            config(future_request_cap=1),
            (product(P1),),
            entry_state_adapter=TimelineStateAdapter({P1: (opened,)}),
            fill_adapter=RelativeFillAdapter({P1: 100}),
            normal_exit_enabled=True,
            spot_trade_adapter=SyntheticSpotTrades((spot_trade(P1, trade_time),)),
        ).run(
            (
                book("future", P1, 90, packet=40),
                book("spot", P1, 90, packet=41),
                opened,
            )
        )

        exit_hedge = next(
            fact for fact in result.executions if fact.role == "exit_hedge"
        )
        self.assertGreater(exit_hedge.cursor.recv_time_ns, exit_t0)
        self.assertEqual(exit_hedge.cursor.recv_time_ns, expected_send)
        future_actual = [
            event.event_cursor.recv_time_ns
            for event in result.request_events
            if event.venue == "future" and event.event_type == "actual_send"
        ]
        self.assertEqual(future_actual, [entry_hedge_time, expected_send])

    def test_exit_hedge_timeout_rolls_back_same_sources_and_restores_cap(self) -> None:
        opened = observation(P1, 100)
        entry_fill_time = 200
        entry_hedge_time = entry_fill_time + HEDGE_DELAY_NS
        trade_time = entry_hedge_time + 10_000_000
        future_closes = trade_time + 10_000_000

        result = S1EventLoop(
            config(),
            (product(P1),),
            entry_state_adapter=TimelineStateAdapter({P1: (opened,)}),
            fill_adapter=RelativeFillAdapter({P1: 100}),
            normal_exit_enabled=True,
            spot_trade_adapter=SyntheticSpotTrades((spot_trade(P1, trade_time),)),
        ).run(
            (
                book("future", P1, 90, packet=50),
                book("spot", P1, 90, packet=51),
                opened,
                book("future", P1, future_closes, packet=52, trial=True),
            )
        )

        maker = next(fact for fact in result.executions if fact.role == "exit_maker")
        rollback = next(
            fact for fact in result.executions if fact.role == "exit_rollback"
        )
        self.assertEqual(
            rollback.initiating_execution_allocations,
            (
                type(rollback.initiating_execution_allocations[0])(
                    maker.execution_id,
                    2_000,
                ),
            ),
        )
        self.assertNotIn("exit_hedge", [fact.role for fact in result.executions])
        self.assertEqual(result.positions[0].state, "paired_open")
        self.assertEqual(result.positions[0].exit_unresolved_shares, 2_000)
        final = result.capacity_transitions[-1].global_after
        self.assertEqual(final.paired_open, 5_000)
        self.assertEqual(final.exit_in_progress, 0)
        timeout = next(
            event
            for event in result.risk_events
            if event.stage == "exit"
            and event.risk_kind == "hedge"
            and event.event_type == "timeout"
        )
        self.assertEqual(timeout.cursor.event_sequence, 800)

    def test_maker_fill_at_session_end_reaches_timeout_and_stays_unresolved(
        self,
    ) -> None:
        opened = observation(P1, 100)
        session_end = 200
        result = S1EventLoop(
            config(),
            (product(P1, session_end=session_end),),
            entry_state_adapter=TimelineStateAdapter({P1: (opened,)}),
            fill_adapter=RelativeFillAdapter({P1: session_end - 100}),
        ).run(
            (
                book("future", P1, 90, packet=60),
                book("spot", P1, 90, packet=61),
                opened,
                SessionExpiry(EventCursor(session_end, 30, 0)),
            )
        )

        self.assertEqual(
            result.positions[0].state,
            "entry_hedge_timeout_unresolved",
        )
        self.assertEqual(
            result.capacity_transitions[-1].global_after.hedge_pending,
            5_000,
        )
        hedge_timeout = next(
            event
            for event in result.risk_events
            if event.risk_kind == "hedge" and event.event_type == "timeout"
        )
        self.assertEqual(
            hedge_timeout.cursor.recv_time_ns,
            session_end + HEDGE_DELAY_NS,
        )
        rollback_timeout = next(
            event
            for event in result.risk_events
            if event.risk_kind == "rollback" and event.event_type == "timeout"
        )
        self.assertEqual(
            rollback_timeout.cursor.recv_time_ns,
            session_end + HEDGE_DELAY_NS,
        )

    def test_exit_rollback_failure_keeps_capacity_and_never_rehangs(self) -> None:
        opened = observation(P1, 100)
        entry_fill_time = 200
        entry_hedge_time = entry_fill_time + HEDGE_DELAY_NS
        trade_time = entry_hedge_time + 10_000_000
        future_closes = trade_time + 10_000_000
        spot_closes = future_closes + 10_000_000

        result = S1EventLoop(
            config(),
            (product(P1),),
            entry_state_adapter=TimelineStateAdapter({P1: (opened,)}),
            fill_adapter=RelativeFillAdapter({P1: 100}),
            normal_exit_enabled=True,
            spot_trade_adapter=SyntheticSpotTrades((spot_trade(P1, trade_time),)),
        ).run(
            (
                book("future", P1, 90, packet=70),
                book("spot", P1, 90, packet=71),
                opened,
                book("future", P1, future_closes, packet=72, trial=True),
                book("spot", P1, spot_closes, packet=73, trial=True),
            )
        )

        self.assertNotIn("exit_rollback", [fact.role for fact in result.executions])
        self.assertEqual(
            result.positions[0].state,
            "exit_rollback_failed_unresolved",
        )
        self.assertEqual(result.positions[0].exit_resolution_state, "unresolved")
        self.assertEqual(result.positions[0].exit_unresolved_shares, 2_000)
        self.assertEqual(result.carry_out, ())
        final = result.capacity_transitions[-1]
        self.assertEqual(final.event_type, "exit_rollback_failed")
        self.assertEqual(final.global_after.exit_in_progress, 5_000)
        exit_news = [
            event
            for event in result.request_events
            if event.stage == "exit"
            and event.request_class == "new"
            and event.event_type == "actual_send"
        ]
        self.assertEqual(len(exit_news), 1)

    def test_one_trade_aggregates_two_fifo_positions_with_unique_execution_cursors(
        self,
    ) -> None:
        first = observation(P1, 100, tick=1_000, snapshot=1)
        second = observation(P1, 300, tick=1_001, snapshot=2)
        trade_time = 60_000_000
        loop = S1EventLoop(
            config(),
            (product(P1),),
            entry_state_adapter=TimelineStateAdapter({P1: (first, second)}),
            fill_adapter=RelativeFillAdapter({P1: 100}),
            normal_exit_enabled=True,
            spot_trade_adapter=SyntheticSpotTrades(
                (spot_trade(P1, trade_time, quantity=4_000),)
            ),
        )
        result = loop.run(
            (
                book("future", P1, 90, packet=80),
                book("spot", P1, 90, packet=81),
                first,
                second,
            )
        )

        maker_legs = [fact for fact in result.executions if fact.role == "exit_maker"]
        self.assertEqual(len(result.exit_physical_fills), 1)
        self.assertEqual(result.exit_physical_fills[0].fill_shares, 4_000)
        self.assertEqual(len(maker_legs), 2)
        self.assertLess(maker_legs[0].cursor, maker_legs[1].cursor)
        self.assertNotEqual(
            maker_legs[0].domain_allocation_id,
            maker_legs[1].domain_allocation_id,
        )
        for execution in maker_legs:
            allocation_id = execution.domain_allocation_id
            assert allocation_id is not None
            transition = next(
                value
                for value in result.capacity_transitions
                if allocation_id in value.transition_id
            )
            transition_cursor = EventCursor(
                transition.timestamp_ns,
                transition.event_sequence,
                transition.row_index,
            )
            self.assertLess(transition_cursor, execution.cursor)
        self.assertTrue(
            all(position.state == "exit_maker_flat" for position in result.positions)
        )

    def test_fifo_coordinator_blocks_newer_different_target_position(self) -> None:
        first = observation(
            P1,
            100,
            tick=1_000,
            snapshot=1,
            lower_bp=12.5,
        )
        second = observation(
            P1,
            300,
            tick=1_001,
            snapshot=2,
            lower_bp=100.0,
            exit_price=102.0,
        )
        loop = S1EventLoop(
            config(),
            (product(P1),),
            entry_state_adapter=TimelineStateAdapter({P1: (first, second)}),
            fill_adapter=RelativeFillAdapter({P1: 100}),
            normal_exit_enabled=True,
            spot_trade_adapter=SyntheticSpotTrades(()),
        )
        result = loop.run(
            (
                book("future", P1, 90, packet=90),
                book("spot", P1, 90, packet=91),
                first,
                second,
            )
        )

        controller = loop.exit_controllers[P1]
        self.assertEqual(len(controller.working_orders), 1)
        self.assertEqual(len(controller.working_orders[0].members), 1)
        oldest = min(controller.positions, key=lambda value: value.fifo_key)
        newest = max(controller.positions, key=lambda value: value.fifo_key)
        self.assertEqual(
            controller.working_orders[0].members[0].position_id,
            oldest.position_id,
        )
        self.assertEqual(newest.desired_shares, 0)
        self.assertIsNone(newest.allocation_id)
        exit_news = [
            event
            for event in result.request_events
            if event.stage == "exit"
            and event.request_class == "new"
            and event.event_type == "actual_send"
        ]
        self.assertEqual(len(exit_news), 1)

    def test_partial_exit_cancel_rolls_back_exact_subunit_and_restores_cap(
        self,
    ) -> None:
        opened = observation(P1, 100)
        entry_fill_time = 200
        entry_hedge_time = entry_fill_time + HEDGE_DELAY_NS
        trade_time = entry_hedge_time + 10_000_000
        result = S1EventLoop(
            config(),
            (product(P1),),
            entry_state_adapter=TimelineStateAdapter({P1: (opened,)}),
            fill_adapter=RelativeFillAdapter({P1: 100}),
            normal_exit_enabled=True,
            spot_trade_adapter=SyntheticSpotTrades(
                (
                    spot_trade(
                        P1,
                        trade_time,
                        price=101.0,
                        quantity=4_000,
                    ),
                )
            ),
        ).run(
            (
                book("future", P1, 90, packet=100),
                book("spot", P1, 90, packet=101),
                opened,
            )
        )

        physical = result.exit_physical_fills[0]
        self.assertEqual(physical.fill_shares, 1_000)
        self.assertEqual(physical.fill_reason, "same_price_queue_depletion")
        rollback = next(
            fact for fact in result.executions if fact.role == "exit_rollback"
        )
        self.assertEqual(rollback.quantity, 1_000)
        self.assertEqual(
            rollback.initiating_execution_allocations[0].share_equivalent,
            1_000,
        )
        self.assertEqual(result.positions[0].state, "paired_open")
        final = result.capacity_transitions[-1].global_after
        self.assertEqual(final.paired_open, 5_000)
        self.assertEqual(final.exit_in_progress, 0)
        exit_new_times = [
            event.event_cursor.recv_time_ns
            for event in result.request_events
            if event.stage == "exit"
            and event.request_class == "new"
            and event.event_type == "actual_send"
        ]
        self.assertEqual(len(exit_new_times), 2)
        self.assertEqual(exit_new_times[-1], rollback.cursor.recv_time_ns + 1)

    def test_accounting_bridge_owns_canonical_establishment_and_terminal(self) -> None:
        opened = observation(P1, 100, lower_bp=12.5)
        entry_fill_time = 200
        entry_hedge_time = entry_fill_time + HEDGE_DELAY_NS
        trade_time = entry_hedge_time + 10_000_000
        accounting = S1AccountingBridge(
            default_date=DATE,
            scenario_id="q95",
            products=(S1AccountingProduct(P1, P1, 2_000),),
        )

        result = S1EventLoop(
            config(),
            (product(P1),),
            entry_state_adapter=TimelineStateAdapter({P1: (opened,)}),
            fill_adapter=RelativeFillAdapter({P1: 100}),
            normal_exit_enabled=True,
            spot_trade_adapter=SyntheticSpotTrades((spot_trade(P1, trade_time),)),
            accounting_adapter=accounting,
        ).run(
            (
                book("future", P1, 90, packet=110),
                book("spot", P1, 90, packet=111),
                opened,
            )
        )

        establishment = next(
            fact
            for fact in accounting.facts
            if isinstance(fact, PositionEstablishedFact)
        )
        terminal = next(
            fact
            for fact in accounting.facts
            if isinstance(fact, TerminalRealizedAccounting)
        )
        self.assertEqual(establishment.sequence, 3)
        self.assertEqual(establishment.cursor.event_sequence, 450)
        self.assertEqual(terminal.sequence, 6)
        self.assertEqual(terminal.cursor.event_sequence, 450)
        self.assertEqual(terminal.terminal_outcome, "exit_maker_flat")
        self.assertEqual(result.positions[0].state, "exit_maker_flat")
        accounting.verify()


if __name__ == "__main__":
    unittest.main()
