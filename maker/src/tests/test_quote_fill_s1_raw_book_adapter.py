"""Contracts for the immutable S1 raw-book day index."""

from __future__ import annotations

import random
import unittest
from datetime import UTC, datetime, timedelta

import polars as pl

from ..quote_fill import s1_raw_book_adapter as raw_book_adapter
from ..quote_fill.layered import EventCursor
from ..quote_fill.s1_exit_target import build_s1_spot_ask_target
from ..quote_fill.s1_hedge import (
    CausalBookState,
    RawBookCursor,
    RawBookEvent,
    RawBookLevel,
    executable_book,
)
from ..quote_fill.s1_raw_book_adapter import (
    RawBookDayIndex,
    RawBookKey,
    build_raw_book_day_index,
)
from ..quote_fill.targets import absolute_price_tick, target_price_for_basis

BASE = datetime(2026, 5, 5, 1, 0, 0)  # noqa: DTZ001


def _mapping() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ValueCode": ["2317"],
            "QuoteCode": ["DHFB6"],
            "spot_ref_price": [100.0],
            "fut_ref_price": [100.5],
            "contract_size": [2_000],
        }
    )


def _row(
    market: str,
    *,
    code: str,
    second: int,
    channel: int,
    packet: int,
    trial: int = 0,
    **updates: object,
) -> dict[str, object]:
    recv_time = BASE + timedelta(seconds=second)
    if market == "future":
        recv_time = recv_time.replace(tzinfo=UTC)
    row: dict[str, object] = {
        "RecvTime": recv_time,
        "QuoteCode": code,
        "ChannelSeq": channel,
        "PacketSeq": packet,
        "TrialMatch": trial,
        "BestBidPrice": 0,
        "BestBidLots": 0,
        "BestAskPrice": 0,
        "BestAskLots": 0,
    }
    for side in ("Bid", "Ask"):
        for level in range(1, 6):
            row[f"{side}Price{level}"] = 0
            row[f"{side}Lots{level}"] = 0
    if market == "future":
        row["DecimalLocator"] = 2
    row.update(updates)
    return row


def _spot_rows() -> pl.DataFrame:
    rows = [
        _row(
            "spot",
            code="2317",
            second=1,
            channel=11,
            packet=2,
            BestBidPrice=100.0,
            BestBidLots=7,
            BestAskPrice=101.0,
            BestAskLots=4,
        ),
        _row(
            "spot",
            code="2317",
            second=0,
            channel=10,
            packet=1,
            BidPrice1=100.0,
            BidLots1=2,
            AskPrice1=101.0,
            AskLots1=3,
            BestBidPrice=100.0,
            BestBidLots=5,
            BestAskPrice=101.0,
            BestAskLots=1,
        ),
        _row(
            "spot",
            code="2317",
            second=2,
            channel=12,
            packet=3,
            trial=1,
            BidPrice1=88.0,
            BidLots1=99,
            AskPrice1=111.0,
            AskLots1=99,
        ),
        _row(
            "spot",
            code="2317",
            second=3,
            channel=13,
            packet=4,
        ),
        _row(
            "spot",
            code="2317",
            second=4,
            channel=14,
            packet=5,
            BestBidPrice=99.0,
            BestBidLots=1,
            BestAskPrice=102.0,
            BestAskLots=2,
        ),
        _row(
            "spot",
            code="2317",
            second=5,
            channel=15,
            packet=6,
            BidPrice1=98.0,
            BidLots1=3,
            AskPrice1=103.0,
            AskLots1=4,
        ),
        _row(
            "spot",
            code="9999",
            second=0,
            channel=1,
            packet=1,
            BidPrice1=1.0,
            BidLots1=1,
            AskPrice1=2.0,
            AskLots1=1,
        ),
    ]
    return pl.from_dicts(rows, infer_schema_length=None)


def _future_rows() -> pl.DataFrame:
    return pl.from_dicts(
        [
            _row(
                "future",
                code="DHFB6",
                second=0,
                channel=20,
                packet=10,
                BidPrice1=10_000,
                BidLots1=2,
                AskPrice1=10_100,
                AskLots1=3,
                BestBidPrice=10_000,
                BestBidLots=5,
                BestAskPrice=10_050,
                BestAskLots=4,
            ),
            _row(
                "future",
                code="OTHER",
                second=0,
                channel=1,
                packet=1,
                BidPrice1=1,
                BidLots1=1,
                AskPrice1=2,
                AskLots1=1,
            ),
        ],
        infer_schema_length=None,
    )


def _ns(second: int) -> int:
    value = BASE + timedelta(seconds=second)
    return int(pl.Series([value]).dt.timestamp("ns")[0])


class RawBookDayIndexTest(unittest.TestCase):
    def test_packed_boundaries_match_legacy_randomized_queries(self) -> None:
        index = build_raw_book_day_index(_spot_rows(), _future_rows(), _mapping())
        spot = RawBookKey("spot", "2317", "DHFB6")
        series = index._require_series(spot)
        randomizer = random.Random(20260827)
        boundaries: list[int | EventCursor | RawBookCursor] = []
        for _ in range(500):
            timestamp = _ns(randomizer.randrange(-1, 7))
            row_index = randomizer.randrange(0, index.retained_event_count + 2)
            event = EventCursor(timestamp, randomizer.randrange(0, 3), row_index)
            boundary_kind = randomizer.randrange(3)
            if boundary_kind == 0:
                boundaries.append(timestamp)
            elif boundary_kind == 1:
                boundaries.append(event)
            else:
                boundaries.append(RawBookCursor(event, randomizer.randrange(0, 10)))

        for boundary in boundaries:
            expected = 0
            while expected < series.length:
                position = series.position_start + expected
                if (
                    raw_book_adapter._clock_boundary(index._clock, position, boundary)
                    > boundary
                ):
                    break
                expected += 1
            self.assertEqual(index._right_index(series, boundary), expected)

            effective_expected = 0
            while effective_expected < series.effective_length:
                derived_position = series.effective_position_start + effective_expected
                position = int(
                    index._effective_clock["clock_position"][derived_position]
                )
                if (
                    raw_book_adapter._clock_boundary(index._clock, position, boundary)
                    > boundary
                ):
                    break
                effective_expected += 1
            self.assertEqual(
                index._effective_right_index(series, boundary),
                effective_expected,
            )

            exit_expected = 0
            while exit_expected < series.exit_quote_length:
                derived_position = series.exit_quote_position_start + exit_expected
                position = int(
                    index._exit_quote_clock["clock_position"][derived_position]
                )
                if (
                    raw_book_adapter._clock_boundary(index._clock, position, boundary)
                    > boundary
                ):
                    break
                exit_expected += 1
            self.assertEqual(
                index._exit_quote_right_index(series, boundary),
                exit_expected,
            )

        with self.assertRaisesRegex(TypeError, "query boundary"):
            index.event_as_of(spot, True)

    def test_exact_cache_preserves_same_time_cursor_and_trial_formal_semantics(
        self,
    ) -> None:
        same_time_rows = pl.from_dicts(
            [
                _row(
                    "spot",
                    code="2317",
                    second=0,
                    channel=1,
                    packet=10,
                    BidPrice1=99.0,
                    BidLots1=1,
                    AskPrice1=101.0,
                    AskLots1=1,
                ),
                _row(
                    "spot",
                    code="2317",
                    second=0,
                    channel=2,
                    packet=11,
                    trial=1,
                ),
                _row(
                    "spot",
                    code="2317",
                    second=0,
                    channel=3,
                    packet=12,
                    BidPrice1=98.0,
                    BidLots1=2,
                    AskPrice1=102.0,
                    AskLots1=2,
                ),
            ],
            infer_schema_length=None,
        )
        index = build_raw_book_day_index(same_time_rows, _future_rows(), _mapping())
        spot = RawBookKey("spot", "2317", "DHFB6")
        rows = tuple(index.iter_indexed_changes(spot, _ns(0) - 1, _ns(0)))
        self.assertEqual(len(rows), 3)
        formal_before, trial, formal_after = rows

        first_boundary = formal_before.event.book_cursor
        trial_boundary = trial.event.book_cursor
        final_boundary = EventCursor(_ns(0), 1, 0)
        self.assertTrue(index.event_as_of(spot, first_boundary).formal_book)
        self.assertTrue(index.event_as_of(spot, trial_boundary).trial_match)
        final = index.indexed_event_as_of(spot, final_boundary)
        repeated = index.indexed_event_as_of(spot, final_boundary)
        assert final is not None
        self.assertIs(repeated, final)
        self.assertEqual(final, formal_after)
        self.assertEqual(final.source_cursor.channel_sequence, 3)
        self.assertEqual(final.event.book_cursor.packet_sequence, 12)

        state = index.state_as_of(spot, final_boundary)
        repeated_state = index.state_as_of(spot, final_boundary)
        self.assertIs(repeated_state, state)
        assert state is not None
        self.assertTrue(state.gate_open)
        self.assertEqual(state.bids[0], RawBookLevel(98.0, 2_000))
        later_phase_state = index.state_as_of(
            spot,
            EventCursor(_ns(0), 600, 999),
        )
        self.assertIs(later_phase_state, state)

        trial_state = index.state_as_of(spot, trial_boundary)
        assert trial_state is not None
        self.assertFalse(trial_state.gate_open)
        self.assertEqual((trial_state.bids, trial_state.asks), ((), ()))
        restored = index.state_as_of(spot, final_boundary)
        self.assertEqual(restored, state)
        counts = index.query_cache_counts
        self.assertGreaterEqual(counts["indexed_hits"], 2)
        self.assertGreaterEqual(counts["state_hits"], 1)
        self.assertGreaterEqual(counts["state_misses"], 3)

    def test_trial_clear_is_immediate_and_only_formal_book_reopens(self) -> None:
        index = build_raw_book_day_index(_spot_rows(), _future_rows(), _mapping())
        spot = RawBookKey("spot", "2317", "DHFB6")
        # Compact-v2 drops the second=3 non-book status row.
        self.assertEqual(index.event_count(spot), 5)

        initial = index.state_as_of(spot, _ns(0))
        assert initial is not None
        self.assertTrue(initial.gate_open)
        self.assertEqual(initial.reference_price, 100.0)
        self.assertEqual(
            [(level.price, level.quantity) for level in initial.bids], [(100.0, 5_000)]
        )
        self.assertEqual(
            [(level.price, level.quantity) for level in initial.asks], [(101.0, 3_000)]
        )

        best_only = index.state_as_of(spot, _ns(1))
        assert best_only is not None
        self.assertEqual(best_only.bids[0].quantity, 7_000)
        self.assertEqual(best_only.asks[0].quantity, 4_000)

        trial_event = index.event_as_of(spot, _ns(2))
        assert trial_event is not None
        self.assertTrue(trial_event.trial_match)
        self.assertFalse(trial_event.formal_book)
        cleared = index.state_as_of(spot, _ns(2))
        assert cleared is not None
        self.assertFalse(cleared.gate_open)
        self.assertEqual((cleared.bids, cleared.asks), ((), ()))

        last_change = index.event_as_of(spot, _ns(3))
        assert last_change is not None
        self.assertTrue(last_change.trial_match)
        self.assertEqual(last_change.book_cursor.cursor.recv_time_ns, _ns(2))
        still_cleared = index.state_as_of(spot, _ns(3))
        self.assertEqual(still_cleared, cleared)

        next_cursor = index.next_change_cursor(spot, _ns(2))
        assert next_cursor is not None
        self.assertEqual(next_cursor.cursor.recv_time_ns, _ns(4))

        reopened = index.state_as_of(spot, _ns(4))
        assert reopened is not None
        self.assertTrue(reopened.gate_open)
        self.assertEqual(
            [(level.price, level.quantity) for level in reopened.bids], [(99.0, 1_000)]
        )
        self.assertEqual(
            [(level.price, level.quantity) for level in reopened.asks], [(102.0, 2_000)]
        )

        l1_after_best = index.state_as_of(spot, _ns(5))
        assert l1_after_best is not None
        self.assertEqual(
            [(level.price, level.quantity) for level in l1_after_best.bids],
            [(99.0, 1_000), (98.0, 3_000)],
        )
        self.assertEqual(
            [(level.price, level.quantity) for level in l1_after_best.asks],
            [(102.0, 2_000), (103.0, 4_000)],
        )

    def test_future_decimal_locator_and_contract_quantities_are_preserved(self) -> None:
        index = RawBookDayIndex.from_selected_rows(
            _spot_rows(), _future_rows(), _mapping()
        )
        future = RawBookKey("future", "2317", "DHFB6")
        self.assertEqual(index.event_count(future), 1)
        state = index.state_as_of(future, _ns(0))
        assert state is not None
        self.assertEqual(
            [(level.price, level.quantity) for level in state.bids],
            [(100.0, 5)],
        )
        self.assertEqual(
            [(level.price, level.quantity) for level in state.asks],
            [(100.5, 4), (101.0, 3)],
        )
        self.assertEqual(state.reference_price, 100.5)
        self.assertEqual(state.book_cursor.cursor.recv_time_ns, _ns(0))
        self.assertEqual(state.book_cursor.cursor.event_sequence, 0)
        self.assertEqual(state.book_cursor.cursor.row_index, 1)
        self.assertEqual(state.book_cursor.packet_sequence, 10)
        source = index.source_cursor_as_of(future, _ns(0))
        assert source is not None
        self.assertEqual(source.recv_time_ns, _ns(0))
        self.assertEqual(source.channel_sequence, 20)
        self.assertEqual(source.packet_sequence, 10)
        self.assertEqual(source.source_row, 0)

    def test_asof_and_open_closed_interval_queries_use_logical_boundaries(self) -> None:
        index = build_raw_book_day_index(_spot_rows(), _future_rows(), _mapping())
        spot = RawBookKey("spot", "2317", "DHFB6")
        self.assertIsNone(index.event_as_of(spot, _ns(0) - 1))
        self.assertIsNone(index.state_as_of(spot, _ns(0) - 1))

        changes = tuple(index.iter_raw_changes(spot, _ns(2), _ns(5)))
        self.assertEqual(
            [event.book_cursor.cursor.recv_time_ns for event in changes],
            [_ns(4), _ns(5)],
        )
        self.assertTrue(changes[0].formal_book)
        self.assertTrue(changes[-1].formal_book)

        trial = index.event_as_of(spot, _ns(2))
        deadline = index.event_as_of(spot, _ns(4))
        assert trial is not None and deadline is not None
        cursor_changes = tuple(
            index.iter_raw_changes(
                spot,
                trial.book_cursor,
                deadline.book_cursor,
            )
        )
        self.assertEqual(
            [event.book_cursor.cursor.recv_time_ns for event in cursor_changes],
            [_ns(4)],
        )

    def test_one_sided_snapshots_replace_bundle_and_other_bundle_persists(self) -> None:
        spot_rows = pl.from_dicts(
            [
                _row(
                    "spot",
                    code="2317",
                    second=0,
                    channel=1,
                    packet=1,
                    BidPrice1=99.0,
                    BidLots1=2,
                    AskPrice1=101.0,
                    AskLots1=3,
                    BestBidPrice=100.0,
                    BestBidLots=1,
                    BestAskPrice=100.5,
                    BestAskLots=1,
                ),
                _row(
                    "spot",
                    code="2317",
                    second=1,
                    channel=2,
                    packet=2,
                    BidPrice1=98.0,
                    BidLots1=4,
                ),
                _row(
                    "spot",
                    code="2317",
                    second=2,
                    channel=3,
                    packet=3,
                    BestAskPrice=102.0,
                    BestAskLots=5,
                ),
            ],
            infer_schema_length=None,
        )
        index = build_raw_book_day_index(spot_rows, _future_rows(), _mapping())
        key = RawBookKey("spot", "2317", "DHFB6")

        bid_only_l1 = index.state_as_of(key, _ns(1))
        assert bid_only_l1 is not None
        self.assertEqual(
            [(level.price, level.quantity) for level in bid_only_l1.bids],
            [(100.0, 1_000), (98.0, 4_000)],
        )
        self.assertEqual(
            [(level.price, level.quantity) for level in bid_only_l1.asks],
            [(100.5, 1_000)],
        )

        ask_only_best = index.state_as_of(key, _ns(2))
        assert ask_only_best is not None
        self.assertEqual(
            [(level.price, level.quantity) for level in ask_only_best.bids],
            [(98.0, 4_000)],
        )
        self.assertEqual(
            [(level.price, level.quantity) for level in ask_only_best.asks],
            [(102.0, 5_000)],
        )

    def test_spot_source_projection_is_exact_and_lazy_constructor_matches(self) -> None:
        index = RawBookDayIndex.from_selected_scans(
            _spot_rows().lazy(), _future_rows().lazy(), _mapping()
        )
        snapshots = index.spot_source_snapshots()
        self.assertEqual(index.spot_source_snapshot_count, 4)
        self.assertEqual(snapshots.height, 4)
        self.assertEqual(
            snapshots["ChannelSeq"].to_list(),
            [10, 11, 14, 15],
        )
        first = snapshots.row(0, named=True)
        self.assertEqual(first["ValueCode"], "2317")
        self.assertEqual(first["QuoteCode"], "DHFB6")
        self.assertEqual(first["PacketSeq"], 1)
        self.assertEqual(first["source_row"], 1)
        self.assertEqual(first["BidLots1"], 2)
        scalar = index.spot_source_snapshot(
            RawBookKey("spot", "2317", "DHFB6"),
            10,
            recv_time_ns=_ns(0),
        )
        assert scalar is not None
        self.assertEqual(scalar.source_cursor.packet_sequence, 1)
        self.assertEqual(scalar.source_cursor.source_row, 1)
        self.assertEqual(scalar.bid_price1, 100.0)
        self.assertEqual(scalar.bid_price2, 0.0)
        self.assertEqual(scalar.bid_lots1, 2)
        self.assertIsNone(
            index.spot_source_snapshot(
                RawBookKey("spot", "2317", "DHFB6"),
                10,
                recv_time_ns=_ns(1),
            )
        )
        self.assertGreater(index.estimated_size_bytes, 0)
        self.assertEqual(
            set(index.storage_frames()),
            {
                "clock",
                "effective_clock",
                "exit_quote_clock",
                "spot_lookup",
                "spot_l1",
                "spot_best",
                "future_l1",
                "future_best",
                "keys",
            },
        )

    def test_effective_clock_keeps_clears_and_skips_identical_snapshots(self) -> None:
        spot_rows = pl.from_dicts(
            [
                _row(
                    "spot",
                    code="2317",
                    second=0,
                    channel=1,
                    packet=1,
                    BidPrice1=99.0,
                    BidLots1=2,
                    AskPrice1=101.0,
                    AskLots1=3,
                    BestBidPrice=100.0,
                    BestBidLots=1,
                    BestAskPrice=100.5,
                    BestAskLots=1,
                ),
                _row(
                    "spot",
                    code="2317",
                    second=1,
                    channel=2,
                    packet=2,
                    BidPrice1=99.0,
                    BidLots1=2,
                ),
                _row(
                    "spot",
                    code="2317",
                    second=2,
                    channel=3,
                    packet=3,
                    BidPrice1=99.0,
                    BidLots1=2,
                ),
                _row(
                    "spot",
                    code="2317",
                    second=3,
                    channel=4,
                    packet=4,
                    BestBidPrice=100.0,
                    BestBidLots=1,
                ),
                _row(
                    "spot",
                    code="2317",
                    second=4,
                    channel=5,
                    packet=5,
                    BestBidPrice=100.0,
                    BestBidLots=1,
                ),
                _row(
                    "spot",
                    code="2317",
                    second=5,
                    channel=6,
                    packet=6,
                    trial=1,
                ),
                _row(
                    "spot",
                    code="2317",
                    second=6,
                    channel=7,
                    packet=7,
                    trial=1,
                ),
                _row(
                    "spot",
                    code="2317",
                    second=7,
                    channel=8,
                    packet=8,
                    BidPrice1=99.0,
                    BidLots1=2,
                    AskPrice1=101.0,
                    AskLots1=3,
                    BestBidPrice=100.0,
                    BestBidLots=1,
                    BestAskPrice=100.5,
                    BestAskLots=1,
                ),
            ],
            infer_schema_length=None,
        )
        index = build_raw_book_day_index(spot_rows, _future_rows(), _mapping())
        key = RawBookKey("spot", "2317", "DHFB6")

        self.assertEqual(index.event_count(key), 8)
        self.assertEqual(index.effective_event_count(key), 5)
        self.assertEqual(index.effective_change_count, 6)  # plus one future state

        raw_repeat = index.next_change_cursor(key, _ns(1))
        assert raw_repeat is not None
        self.assertEqual(raw_repeat.cursor.recv_time_ns, _ns(2))
        effective = index.next_effective_change_cursor(key, _ns(1))
        assert effective is not None
        self.assertEqual(effective.cursor.recv_time_ns, _ns(3))

        repeated_state = index.state_as_of(key, _ns(2))
        assert repeated_state is not None
        self.assertEqual(repeated_state.book_cursor.cursor.recv_time_ns, _ns(2))
        self.assertEqual(
            [(level.price, level.quantity) for level in repeated_state.asks],
            [(100.5, 1_000)],
        )
        repeated_source = index.source_cursor_as_of(key, _ns(2))
        assert repeated_source is not None
        self.assertEqual(repeated_source.channel_sequence, 3)

        best_ask_cleared = index.state_as_of(key, _ns(4))
        assert best_ask_cleared is not None
        self.assertEqual(best_ask_cleared.asks, ())
        after_first_trial = index.next_effective_change_cursor(key, _ns(5))
        assert after_first_trial is not None
        self.assertEqual(after_first_trial.cursor.recv_time_ns, _ns(7))

        adapter = index.as_risk_book_adapter()
        after_explicit_clear = EventCursor(_ns(1), 0, 0)
        self.assertIsNone(
            adapter.next_change_cursor("spot", "2317", after_explicit_clear, _ns(2))
        )
        through_change = adapter.next_change_cursor(
            "spot", "2317", after_explicit_clear, _ns(3)
        )
        assert through_change is not None
        self.assertEqual(through_change.recv_time_ns, _ns(3))

    def test_effective_clock_matches_brute_force_forward_filled_states(self) -> None:
        rng = random.Random(20260827)
        templates: tuple[dict[str, object], ...] = (
            {"trial": 1},
            {"BidPrice1": 99.0, "BidLots1": 2},
            {"AskPrice1": 101.0, "AskLots1": 3},
            {"BestBidPrice": 100.0, "BestBidLots": 1},
            {"BestAskPrice": 100.5, "BestAskLots": 4},
            {
                "BidPrice1": 98.0,
                "BidLots1": 5,
                "AskPrice1": 102.0,
                "AskLots1": 6,
                "BestBidPrice": 99.0,
                "BestBidLots": 2,
                "BestAskPrice": 101.0,
                "BestAskLots": 3,
            },
        )
        rows = []
        for second in range(250):
            template = dict(rng.choice(templates))
            trial = int(template.pop("trial", 0))
            rows.append(
                _row(
                    "spot",
                    code="2317",
                    second=second,
                    channel=second + 1,
                    packet=second + 1,
                    trial=trial,
                    **template,
                )
            )
        index = build_raw_book_day_index(
            pl.from_dicts(rows, infer_schema_length=None),
            _future_rows(),
            _mapping(),
        )
        key = RawBookKey("spot", "2317", "DHFB6")
        raw_events = tuple(index.iter_raw_changes(key, _ns(0) - 1, _ns(len(rows))))

        def level_signature(level: RawBookLevel) -> tuple[str, int]:
            return (float(level.price).hex(), int(level.quantity))

        def state_signature(event: RawBookEvent) -> tuple[object, ...]:
            reference = event.reference_price
            return (
                event.trial_match,
                event.formal_book,
                None if reference is None else float(reference).hex(),
                tuple(level_signature(level) for level in event.l1_bids),
                tuple(level_signature(level) for level in event.l1_asks),
                None if event.best_bid is None else level_signature(event.best_bid),
                None if event.best_ask is None else level_signature(event.best_ask),
            )

        expected = []
        previous: tuple[object, ...] | None = None
        for event in raw_events:
            signature = state_signature(event)
            if signature != previous:
                expected.append(event.book_cursor)
            previous = signature

        actual = []
        boundary: int | RawBookCursor = _ns(0) - 1
        while True:
            change = index.next_effective_change_cursor(key, boundary)
            if change is None:
                break
            actual.append(change)
            boundary = change
        self.assertEqual(actual, expected)
        self.assertEqual(index.effective_event_count(key), len(expected))

    def test_spot_exit_clock_uses_bbo_prices_not_depth_or_positive_quantity(
        self,
    ) -> None:
        rows = [
            _row(
                "spot",
                code="2317",
                second=0,
                channel=1,
                packet=1,
                BidPrice1=99.0,
                BidLots1=2,
                BidPrice2=98.0,
                BidLots2=3,
                AskPrice1=101.0,
                AskLots1=4,
                AskPrice2=102.0,
                AskLots2=5,
                BestBidPrice=100.0,
                BestBidLots=1,
                BestAskPrice=100.5,
                BestAskLots=1,
            ),
            _row(
                "spot",
                code="2317",
                second=1,
                channel=2,
                packet=2,
                BidPrice1=99.0,
                BidLots1=20,
                BidPrice2=98.0,
                BidLots2=30,
                AskPrice1=101.0,
                AskLots1=40,
                AskPrice2=102.0,
                AskLots2=50,
                BestBidPrice=100.0,
                BestBidLots=10,
                BestAskPrice=100.5,
                BestAskLots=10,
            ),
            _row(
                "spot",
                code="2317",
                second=2,
                channel=3,
                packet=3,
                BidPrice1=99.0,
                BidLots1=20,
                BidPrice2=97.0,
                BidLots2=30,
                AskPrice1=101.0,
                AskLots1=40,
                AskPrice2=103.0,
                AskLots2=50,
            ),
            _row(
                "spot",
                code="2317",
                second=3,
                channel=4,
                packet=4,
                BestBidPrice=100.0,
                BestBidLots=10,
            ),
            _row(
                "spot",
                code="2317",
                second=4,
                channel=5,
                packet=5,
                BestBidPrice=100.0,
                BestBidLots=99,
            ),
            _row(
                "spot",
                code="2317",
                second=5,
                channel=6,
                packet=6,
                BidPrice1=99.0,
                BidLots1=1,
            ),
            _row(
                "spot",
                code="2317",
                second=6,
                channel=7,
                packet=7,
                BidPrice1=99.0,
                BidLots1=99,
            ),
            _row(
                "spot",
                code="2317",
                second=7,
                channel=8,
                packet=8,
                trial=1,
            ),
            _row(
                "spot",
                code="2317",
                second=8,
                channel=9,
                packet=9,
                trial=1,
            ),
            _row(
                "spot",
                code="2317",
                second=9,
                channel=10,
                packet=10,
                BestBidPrice=100.0,
                BestBidLots=1,
                BestAskPrice=100.5,
                BestAskLots=1,
            ),
            _row(
                "spot",
                code="2317",
                second=10,
                channel=11,
                packet=11,
                BidPrice1=100.6,
                BidLots1=1,
                AskPrice1=100.4,
                AskLots1=1,
            ),
            _row(
                "spot",
                code="2317",
                second=11,
                channel=12,
                packet=12,
                BidPrice1=100.6,
                BidLots1=1,
                BidPrice2=95.0,
                BidLots2=5,
                AskPrice1=100.4,
                AskLots1=1,
                AskPrice2=105.0,
                AskLots2=5,
            ),
        ]
        index = build_raw_book_day_index(
            pl.from_dicts(rows, infer_schema_length=None),
            _future_rows(),
            _mapping(),
        )
        key = RawBookKey("spot", "2317", "DHFB6")
        self.assertEqual(index.event_count(key), 12)
        self.assertEqual(index.exit_quote_event_count(key), 6)

        actual = []
        boundary: int | RawBookCursor = _ns(0) - 1
        while True:
            change = index.next_exit_quote_change_cursor(key, boundary)
            if change is None:
                break
            actual.append(change.cursor.recv_time_ns)
            boundary = change
        self.assertEqual(actual, [_ns(value) for value in (0, 3, 5, 7, 9, 10)])

        unchanged_quantity_state = index.state_as_of(key, _ns(1))
        assert unchanged_quantity_state is not None
        self.assertEqual(
            unchanged_quantity_state.book_cursor.cursor.recv_time_ns,
            _ns(1),
        )
        source = index.source_cursor_as_of(key, _ns(1))
        assert source is not None
        self.assertEqual(source.channel_sequence, 2)

        adapter = index.as_risk_book_adapter()
        after_initial = EventCursor(_ns(0), 0, 0)
        self.assertIsNone(
            adapter.next_exit_quote_change_cursor("spot", "2317", after_initial, _ns(2))
        )
        through_clear = adapter.next_exit_quote_change_cursor(
            "spot", "2317", after_initial, _ns(3)
        )
        assert through_clear is not None
        self.assertEqual(through_clear.recv_time_ns, _ns(3))

    def test_future_exit_clock_matches_buy_one_status_and_vwap(self) -> None:
        future_rows = pl.from_dicts(
            [
                _row(
                    "future",
                    code="DHFB6",
                    second=0,
                    channel=1,
                    packet=1,
                    BidPrice1=9_900,
                    BidLots1=5,
                    AskPrice1=10_100,
                    AskLots1=1,
                    BestBidPrice=10_000,
                    BestBidLots=1,
                    BestAskPrice=10_050,
                    BestAskLots=1,
                ),
                _row(
                    "future",
                    code="DHFB6",
                    second=1,
                    channel=2,
                    packet=2,
                    BestBidPrice=1_000,
                    BestBidLots=10,
                    BestAskPrice=1_005,
                    BestAskLots=10,
                    DecimalLocator=1,
                ),
                _row(
                    "future",
                    code="DHFB6",
                    second=2,
                    channel=3,
                    packet=3,
                    BidPrice1=9_800,
                    BidLots1=9,
                    AskPrice1=10_100,
                    AskLots1=9,
                ),
                _row(
                    "future",
                    code="DHFB6",
                    second=3,
                    channel=4,
                    packet=4,
                    BestBidPrice=10_000,
                    BestBidLots=10,
                ),
                _row(
                    "future",
                    code="DHFB6",
                    second=4,
                    channel=5,
                    packet=5,
                    BidPrice1=9_900,
                    BidLots1=1,
                    AskPrice1=10_000,
                    AskLots1=0,
                    AskPrice2=10_200,
                    AskLots2=1,
                ),
                _row(
                    "future",
                    code="DHFB6",
                    second=5,
                    channel=6,
                    packet=6,
                    BidPrice1=9_900,
                    BidLots1=10,
                    AskPrice1=10_000,
                    AskLots1=0,
                    AskPrice2=10_200,
                    AskLots2=10,
                ),
                _row(
                    "future",
                    code="DHFB6",
                    second=6,
                    channel=7,
                    packet=7,
                    BidPrice1=10_300,
                    BidLots1=1,
                    AskPrice1=10_200,
                    AskLots1=1,
                ),
                _row(
                    "future",
                    code="DHFB6",
                    second=7,
                    channel=8,
                    packet=8,
                    BidPrice1=10_400,
                    BidLots1=1,
                    AskPrice1=10_200,
                    AskLots1=1,
                ),
                _row(
                    "future",
                    code="DHFB6",
                    second=8,
                    channel=9,
                    packet=9,
                    trial=1,
                ),
                _row(
                    "future",
                    code="DHFB6",
                    second=9,
                    channel=10,
                    packet=10,
                    trial=1,
                ),
                _row(
                    "future",
                    code="DHFB6",
                    second=10,
                    channel=11,
                    packet=11,
                    BidPrice1=9_900,
                    BidLots1=1,
                    AskPrice1=10_200,
                    AskLots1=1,
                ),
            ],
            infer_schema_length=None,
        )
        index = build_raw_book_day_index(_spot_rows(), future_rows, _mapping())
        key = RawBookKey("future", "2317", "DHFB6")
        raw_events = tuple(index.iter_raw_changes(key, _ns(0) - 1, _ns(11)))
        expected: list[RawBookCursor] = []
        previous: tuple[bool, str | None] | None = None
        for event in raw_events:
            state = index.state_as_of(key, event.book_cursor)
            assert state is not None
            executable, _ = executable_book(
                state,
                side="buy",
                quantity=1,
                quantity_unit="future_contracts",
                send_eligible_cursor=event.book_cursor.cursor,
            )
            signature = (
                state.gate_open,
                None if executable is None else float(executable.executable_vwap).hex(),
            )
            if signature != previous:
                expected.append(event.book_cursor)
            previous = signature

        actual: list[RawBookCursor] = []
        boundary: int | RawBookCursor = _ns(0) - 1
        while True:
            change = index.next_exit_quote_change_cursor(key, boundary)
            if change is None:
                break
            actual.append(change)
            boundary = change
        self.assertEqual(actual, expected)
        self.assertEqual(
            [value.cursor.recv_time_ns for value in actual],
            [_ns(value) for value in (0, 3, 4, 6, 8, 10)],
        )
        self.assertEqual(index.exit_quote_event_count(key), len(expected))

    def test_spot_exit_clock_is_conservative_for_frozen_thresholds(self) -> None:
        rng = random.Random(20260828)
        templates: tuple[dict[str, object], ...] = (
            {"trial": 1},
            {
                "BidPrice1": 99.0,
                "BidLots1": 1,
                "AskPrice1": 101.0,
                "AskLots1": 1,
            },
            {
                "BidPrice1": 99.0,
                "BidLots1": 20,
                "BidPrice2": 95.0,
                "BidLots2": 50,
                "AskPrice1": 101.0,
                "AskLots1": 30,
                "AskPrice2": 105.0,
                "AskLots2": 60,
            },
            {
                "BestBidPrice": 100.0,
                "BestBidLots": 1,
                "BestAskPrice": 100.5,
                "BestAskLots": 1,
            },
            {
                "BestBidPrice": 100.0,
                "BestBidLots": 100,
                "BestAskPrice": 100.5,
                "BestAskLots": 100,
            },
            {
                "BidPrice1": 102.0,
                "BidLots1": 1,
                "AskPrice1": 101.0,
                "AskLots1": 1,
            },
            {
                "BidPrice1": 90.0,
                "BidLots1": 1,
                "AskPrice1": 110.0,
                "AskLots1": 1,
            },
            {"BidPrice1": 99.0, "BidLots1": 1},
            {"AskPrice1": 101.0, "AskLots1": 1},
        )
        rows = []
        for second in range(300):
            template = dict(rng.choice(templates))
            trial = int(template.pop("trial", 0))
            rows.append(
                _row(
                    "spot",
                    code="2317",
                    second=second,
                    channel=second + 1,
                    packet=second + 1,
                    trial=trial,
                    **template,
                )
            )
        index = build_raw_book_day_index(
            pl.from_dicts(rows, infer_schema_length=None),
            _future_rows(),
            _mapping(),
        )
        key = RawBookKey("spot", "2317", "DHFB6")
        raw_events = tuple(index.iter_raw_changes(key, _ns(0) - 1, _ns(300)))
        future_book = CausalBookState(
            RawBookCursor(EventCursor(_ns(0) - 2), 1),
            True,
            None,
            100.5,
            (RawBookLevel(100.0, 10),),
            (RawBookLevel(100.5, 10),),
        )
        thresholds = (-80.0, -30.0, 0.0, 30.0, 80.0)
        scheduler_transitions: set[RawBookCursor] = set()
        previous: tuple[tuple[bool, int | None], ...] | None = None
        for event in raw_events:
            state = index.state_as_of(key, event.book_cursor)
            assert state is not None
            outcomes = []
            for threshold in thresholds:
                frozen_price = target_price_for_basis(
                    "spot_ask_future_taker",
                    threshold,
                    session_date="20260505",
                    fut_exec_ask=100.5,
                )
                target = build_s1_spot_ask_target(
                    date="20260505",
                    value_code="2317",
                    quote_code="DHFB6",
                    position_id=f"position/{threshold}",
                    scenario_id="randomized-exit-clock",
                    observation_cursor=event.book_cursor.cursor,
                    frozen_exit_threshold_basis_bp=threshold,
                    frozen_exit_target_price=frozen_price,
                    frozen_exit_absolute_price_tick=absolute_price_tick(
                        frozen_price,
                        market="spot",
                        session_date="20260505",
                    ),
                    spot_book=state,
                    future_book=future_book,
                )
                outcomes.append(
                    (
                        target.gate_open,
                        target.absolute_price_tick if target.gate_open else None,
                    )
                )
            signature = tuple(outcomes)
            if signature != previous:
                scheduler_transitions.add(event.book_cursor)
            previous = signature

        derived: set[RawBookCursor] = set()
        boundary: int | RawBookCursor = _ns(0) - 1
        while True:
            change = index.next_exit_quote_change_cursor(key, boundary)
            if change is None:
                break
            derived.add(change)
            boundary = change
        self.assertTrue(scheduler_transitions.issubset(derived))
        self.assertLess(len(derived), index.event_count(key))

    def test_future_exit_clock_randomized_brute_force_equivalence(self) -> None:
        rng = random.Random(20260829)
        templates: tuple[dict[str, object], ...] = (
            {"trial": 1},
            {
                "BidPrice1": 9_900,
                "BidLots1": 1,
                "AskPrice1": 10_100,
                "AskLots1": 1,
            },
            {
                "BidPrice1": 9_900,
                "BidLots1": 10,
                "BidPrice2": 9_800,
                "BidLots2": 20,
                "AskPrice1": 10_100,
                "AskLots1": 10,
                "AskPrice2": 10_200,
                "AskLots2": 20,
            },
            {
                "BestBidPrice": 10_000,
                "BestBidLots": 1,
                "BestAskPrice": 10_050,
                "BestAskLots": 1,
            },
            {
                "BestBidPrice": 10_000,
                "BestBidLots": 100,
                "BestAskPrice": 10_050,
                "BestAskLots": 100,
            },
            {
                "BidPrice1": 10_200,
                "BidLots1": 1,
                "AskPrice1": 10_100,
                "AskLots1": 1,
            },
            {
                "BidPrice1": 9_000,
                "BidLots1": 1,
                "AskPrice1": 11_000,
                "AskLots1": 1,
            },
            {
                "BidPrice1": 9_900,
                "BidLots1": 1,
                "AskPrice1": 10_000,
                "AskLots1": 0,
                "AskPrice2": 10_200,
                "AskLots2": 1,
            },
            {"BidPrice1": 9_900, "BidLots1": 1},
            {"AskPrice1": 10_100, "AskLots1": 1},
        )
        rows = []
        for second in range(300):
            template = dict(rng.choice(templates))
            trial = int(template.pop("trial", 0))
            rows.append(
                _row(
                    "future",
                    code="DHFB6",
                    second=second,
                    channel=second + 1,
                    packet=second + 1,
                    trial=trial,
                    **template,
                )
            )
        index = build_raw_book_day_index(
            _spot_rows(),
            pl.from_dicts(rows, infer_schema_length=None),
            _mapping(),
        )
        key = RawBookKey("future", "2317", "DHFB6")
        raw_events = tuple(index.iter_raw_changes(key, _ns(0) - 1, _ns(300)))
        expected: list[RawBookCursor] = []
        previous: tuple[bool, str | None] | None = None
        for event in raw_events:
            state = index.state_as_of(key, event.book_cursor)
            assert state is not None
            executable, _ = executable_book(
                state,
                side="buy",
                quantity=1,
                quantity_unit="future_contracts",
                send_eligible_cursor=event.book_cursor.cursor,
            )
            signature = (
                state.gate_open,
                None if executable is None else float(executable.executable_vwap).hex(),
            )
            if signature != previous:
                expected.append(event.book_cursor)
            previous = signature

        actual: list[RawBookCursor] = []
        boundary: int | RawBookCursor = _ns(0) - 1
        while True:
            change = index.next_exit_quote_change_cursor(key, boundary)
            if change is None:
                break
            actual.append(change)
            boundary = change
        self.assertEqual(actual, expected)
        self.assertLess(len(actual), index.event_count(key))

    def test_risk_book_bridge_returns_earliest_change_through_deadline(self) -> None:
        index = build_raw_book_day_index(_spot_rows(), _future_rows(), _mapping())
        adapter = index.as_risk_book_adapter()
        state = adapter.state_as_of("spot", "2317", EventCursor(_ns(1), 99, 0))
        assert state is not None
        self.assertTrue(state.gate_open)

        after_trial = EventCursor(_ns(2), 0, 0)
        self.assertIsNone(
            adapter.next_change_cursor("spot", "2317", after_trial, _ns(3))
        )
        next_change = adapter.next_change_cursor("spot", "2317", after_trial, _ns(4))
        assert next_change is not None
        self.assertEqual(next_change.recv_time_ns, _ns(4))
        self.assertEqual(next_change.event_sequence, 0)

    def test_same_time_rows_get_merged_loop_order_and_keep_source_cursor(self) -> None:
        same_time = [
            _row(
                "spot",
                code="2317",
                second=0,
                channel=11,
                packet=9,
                BestBidPrice=104.0,
                BestBidLots=1,
                BestAskPrice=105.0,
                BestAskLots=1,
            ),
            _row(
                "spot",
                code="2317",
                second=0,
                channel=10,
                packet=8,
                BestBidPrice=102.0,
                BestBidLots=1,
                BestAskPrice=103.0,
                BestAskLots=1,
            ),
            _row(
                "spot",
                code="2317",
                second=0,
                channel=11,
                packet=1,
                BestBidPrice=103.0,
                BestBidLots=1,
                BestAskPrice=104.0,
                BestAskLots=1,
            ),
        ]
        spot_rows = pl.from_dicts(same_time, infer_schema_length=None)
        index = build_raw_book_day_index(spot_rows, _future_rows(), _mapping())
        key = RawBookKey("spot", "2317", "DHFB6")
        indexed = tuple(index.iter_indexed_changes(key, _ns(0) - 1, _ns(0)))
        events = tuple(value.event for value in indexed)
        self.assertEqual(
            [event.book_cursor.cursor.event_sequence for event in events],
            [0, 0, 0],
        )
        self.assertEqual(
            [event.book_cursor.cursor.row_index for event in events],
            [0, 1, 2],
        )
        self.assertEqual(
            [value.source_cursor.channel_sequence for value in indexed],
            [10, 11, 11],
        )
        self.assertEqual(
            [value.source_cursor.packet_sequence for value in indexed],
            [8, 1, 9],
        )
        self.assertEqual(
            [value.source_cursor.source_row for value in indexed],
            [1, 2, 0],
        )
        future_key = RawBookKey("future", "2317", "DHFB6")
        future = index.indexed_event_as_of(future_key, _ns(0))
        assert future is not None
        self.assertEqual(future.event.book_cursor.cursor, EventCursor(_ns(0), 0, 3))
        self.assertEqual(future.source_cursor.channel_sequence, 20)

        at_second_spot_row = index.event_as_of(key, EventCursor(_ns(0), 0, 1))
        assert at_second_spot_row is not None
        self.assertEqual(at_second_spot_row.best_bid.price, 103.0)
        after_first = tuple(
            index.iter_raw_changes(
                key,
                EventCursor(_ns(0), 0, 0),
                EventCursor(_ns(0), 0, 2),
            )
        )
        self.assertEqual(
            [event.best_bid.price for event in after_first],
            [103.0, 104.0],
        )
        final = index.state_as_of(key, _ns(0))
        assert final is not None
        self.assertEqual(final.bids[0].price, 104.0)

        effective_after_first = index.next_effective_change_cursor(
            key, EventCursor(_ns(0), 0, 0)
        )
        assert effective_after_first is not None
        self.assertEqual(effective_after_first.cursor, EventCursor(_ns(0), 0, 1))
        effective_after_second = index.next_effective_change_cursor(
            key, effective_after_first
        )
        assert effective_after_second is not None
        self.assertEqual(effective_after_second.cursor, EventCursor(_ns(0), 0, 2))
        self.assertIsNone(index.next_effective_change_cursor(key, _ns(0)))

        exit_after_first = index.next_exit_quote_change_cursor(
            key, EventCursor(_ns(0), 0, 0)
        )
        assert exit_after_first is not None
        self.assertEqual(exit_after_first.cursor, EventCursor(_ns(0), 0, 1))
        exit_after_second = index.next_exit_quote_change_cursor(key, exit_after_first)
        assert exit_after_second is not None
        self.assertEqual(exit_after_second.cursor, EventCursor(_ns(0), 0, 2))
        self.assertIsNone(index.next_exit_quote_change_cursor(key, _ns(0)))
        adapter = index.as_risk_book_adapter()
        deadline_inclusive = adapter.next_exit_quote_change_cursor(
            "spot",
            "2317",
            EventCursor(_ns(0), 0, 0),
            _ns(0),
        )
        self.assertEqual(deadline_inclusive, EventCursor(_ns(0), 0, 1))

        duplicated = pl.from_dicts([*same_time, same_time[0]], infer_schema_length=None)
        with self.assertRaisesRegex(ValueError, "duplicate cursor"):
            build_raw_book_day_index(duplicated, _future_rows(), _mapping())

    def test_only_exact_mapping_keys_are_queryable(self) -> None:
        index = build_raw_book_day_index(_spot_rows(), _future_rows(), _mapping())
        self.assertEqual(
            index.keys,
            (
                RawBookKey("future", "2317", "DHFB6"),
                RawBookKey("spot", "2317", "DHFB6"),
            ),
        )
        with self.assertRaisesRegex(ValueError, "exact mapping"):
            index.state_as_of(RawBookKey("spot", "9999", "DHFB6"), _ns(0))

        duplicate_mapping = pl.concat([_mapping(), _mapping()])
        with self.assertRaisesRegex(ValueError, "one-to-one"):
            build_raw_book_day_index(
                _spot_rows(),
                _future_rows(),
                duplicate_mapping,
            )


if __name__ == "__main__":
    unittest.main()
