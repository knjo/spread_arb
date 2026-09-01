"""Actual-cursor entry-state adapter contracts."""

from __future__ import annotations

import random
import unittest
from datetime import datetime, timedelta
from time import perf_counter
from zoneinfo import ZoneInfo

import polars as pl

from ..quote_fill.layered import EventCursor
from ..quote_fill.policy_spec import TOD_BUCKETS, PolicySpec
from ..quote_fill.s1_day_state import S1_POLICY_DECISION_SIGNATURE_COLUMNS
from ..quote_fill.s1_entry_state_adapter import S1EntryStateAdapter
from ..quote_fill.s1_event_loop import ActualSendMakerSnapshot
from ..quote_fill.s1_hedge import CausalBookState, RawBookCursor, RawBookLevel

DATE = "20260505"
TZ = ZoneInfo("Asia/Taipei")


def _time_ns(second: int) -> int:
    value = datetime(2026, 5, 5, 9, 0, tzinfo=TZ) + timedelta(seconds=second)
    return int(value.timestamp() * 1_000_000_000)


def _book(cursor_ns: int, *, future: bool = False) -> CausalBookState:
    if future:
        bids = (RawBookLevel(101.0, 5),)
        asks = (RawBookLevel(101.5, 5),)
        reference = 101.0
    else:
        bids = (RawBookLevel(100.5, 10), RawBookLevel(100.0, 20))
        asks = (RawBookLevel(101.0, 10),)
        reference = 100.5
    return CausalBookState(
        RawBookCursor(EventCursor(cursor_ns), 1),
        True,
        None,
        reference,
        bids,
        asks,
    )


class Provider:
    def __init__(self) -> None:
        self.spot = _book(_time_ns(299))
        self.future = _book(_time_ns(299), future=True)
        self.snapshot = ActualSendMakerSnapshot(
            "2330",
            17,
            _time_ns(299),
            100.5,
            100.0,
            10,
            20,
        )

    def state_as_of(
        self,
        venue: str,
        product_id: str,
        cursor: EventCursor,
    ) -> CausalBookState | None:
        del product_id, cursor
        return self.spot if venue == "spot" else self.future

    def maker_snapshot_as_of(
        self,
        product_id: str,
        cursor: EventCursor,
    ) -> ActualSendMakerSnapshot | None:
        del product_id, cursor
        return self.snapshot


def _common() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Date": [DATE, DATE],
            "ValueCode": ["2330", "2330"],
            "QuoteCode": ["CDFE6", "CDFE6"],
            "decision_time_ns": [_time_ns(300), _time_ns(301)],
            "analysis_eligible": [True, True],
            "selected_anchor_bp": [0.0, 0.0],
            "contract_size": [2_000.0, 2_000.0],
        }
    )


def _specs() -> tuple[PolicySpec, ...]:
    return tuple(
        PolicySpec(
            Date=DATE,
            ValueCode="2330",
            QuoteCode="CDFE6",
            entry_tod_bucket=bucket,
            policy_id="fixed20",
            kind="fixed",
            upper_distance_bp=20.0,
            lower_distance_bp=20.0,
            upper_source_id="constant_bp:20",
            lower_source_id="constant_bp:20",
            upper_source_asof_date=None,
            lower_source_asof_date=None,
            combined_source_asof_date=None,
        )
        for bucket in TOD_BUCKETS
    )


class S1EntryStateAdapterTest(unittest.TestCase):
    def test_sparse_observation_carries_exact_panel_policy_signature(self) -> None:
        adapter = S1EntryStateAdapter(
            _common(),
            _specs(),
            date=DATE,
            policy_id="fixed20",
            book_provider=Provider(),
        )
        signature_values = tuple(
            f"signature-{index}"
            for index, _ in enumerate(S1_POLICY_DECISION_SIGNATURE_COLUMNS)
        )
        changes = pl.DataFrame(
            {
                "Date": [DATE],
                "ValueCode": ["2330"],
                "decision_time_ns": [_time_ns(300)],
                **{
                    column: [value]
                    for column, value in zip(
                        S1_POLICY_DECISION_SIGNATURE_COLUMNS,
                        signature_values,
                        strict=True,
                    )
                },
            }
        )

        observations = tuple(adapter.iter_observations(changes))

        self.assertEqual(len(observations), 1)
        self.assertEqual(
            observations[0].policy_state_signature,
            signature_values,
        )
        with self.assertRaisesRegex(ValueError, "partial policy signature"):
            tuple(
                adapter.iter_observations(
                    changes.drop(S1_POLICY_DECISION_SIGNATURE_COLUMNS[-1])
                )
            )

    def test_packed_decision_lookup_matches_legacy_and_caches_interval(self) -> None:
        adapter = S1EntryStateAdapter(
            _common(),
            _specs(),
            date=DATE,
            policy_id="fixed20",
            book_provider=Provider(),
        )
        table = adapter._decisions
        offset, length = table.spans["2330"]
        times = table.frame["decision_time_ns"].slice(offset, length)
        randomizer = random.Random(20260827)
        for _ in range(1_000):
            timestamp_ns = _time_ns(299) + randomizer.randrange(0, 3_000_000_000)
            cursor = EventCursor(
                timestamp_ns,
                randomizer.randrange(0, 900),
                randomizer.randrange(0, 20),
            )
            legacy_position = times.search_sorted(timestamp_ns, side="right") - 1
            legacy = (
                None
                if legacy_position < 0
                else table.frame.row(offset + legacy_position)
            )
            self.assertEqual(table.row_as_of("2330", cursor), legacy)

        before = dict(adapter.decision_lookup_counts)
        started = perf_counter()
        for index in range(10_000):
            table.row_as_of(
                "2330",
                EventCursor(
                    _time_ns(300) + index % 1_000_000_000,
                    index % 900,
                    index % 20,
                ),
            )
        elapsed = perf_counter() - started
        after = adapter.decision_lookup_counts
        self.assertGreaterEqual(after["hits"] - before["hits"], 9_999)
        self.assertLess(elapsed, 1.0)

    def test_actual_cursor_uses_raw_ab12_snapshot_and_freezes_lower(self) -> None:
        provider = Provider()
        adapter = S1EntryStateAdapter(
            _common(),
            _specs(),
            date=DATE,
            policy_id="fixed20",
            book_provider=provider,
        )
        state = adapter.current_state("2330", EventCursor(_time_ns(300), 200, 1))
        self.assertIsNotNone(state)
        assert state is not None
        self.assertTrue(state.base_gate_open)
        self.assertTrue(state.admission_open)
        self.assertEqual(state.target_price, 100.5)
        self.assertEqual(state.maker_snapshot, provider.snapshot)
        self.assertEqual(state.reservation_notional_twd, 201_000)
        self.assertEqual(state.frozen_exit_threshold_basis_bp, -20.0)

    def test_non_ab12_target_is_base_open_but_not_admitted(self) -> None:
        provider = Provider()
        provider.snapshot = ActualSendMakerSnapshot(
            "2330",
            18,
            _time_ns(299),
            99.5,
            99.0,
            10,
            20,
        )
        adapter = S1EntryStateAdapter(
            _common(),
            _specs(),
            date=DATE,
            policy_id="fixed20",
            book_provider=provider,
        )
        state = adapter.current_state("2330", EventCursor(_time_ns(300), 200, 1))
        assert state is not None
        self.assertTrue(state.base_gate_open)
        self.assertFalse(state.admission_open)
        self.assertIsNone(state.reservation_notional_twd)
        self.assertEqual(state.gate_reason, "target_not_exact_raw_bid1_bid2")

    def test_trial_match_and_sparse_observation_order_fail_closed(self) -> None:
        provider = Provider()
        provider.future = CausalBookState(
            RawBookCursor(EventCursor(_time_ns(299)), 1),
            False,
            "trial_match",
            None,
            (),
            (),
        )
        adapter = S1EntryStateAdapter(
            _common(),
            _specs(),
            date=DATE,
            policy_id="fixed20",
            book_provider=provider,
        )
        state = adapter.current_state("2330", EventCursor(_time_ns(300), 200, 1))
        assert state is not None
        self.assertFalse(state.base_gate_open)
        self.assertEqual(state.gate_reason, "trial_match")
        self.assertEqual(state.base_gate_book_wake_venues, frozenset(("future",)))

        changes = pl.DataFrame(
            {
                "Date": [DATE, DATE],
                "ValueCode": ["2330", "2330"],
                "decision_time_ns": [_time_ns(300), _time_ns(300)],
            }
        )
        with self.assertRaisesRegex(ValueError, "duplicated"):
            tuple(adapter.iter_observations(changes))

    def test_unavailable_future_ask_requests_only_future_book_wake(self) -> None:
        provider = Provider()
        assert provider.future is not None
        provider.future = CausalBookState(
            provider.future.book_cursor,
            True,
            None,
            provider.future.reference_price,
            provider.future.bids,
            (),
        )
        adapter = S1EntryStateAdapter(
            _common(),
            _specs(),
            date=DATE,
            policy_id="fixed20",
            book_provider=provider,
        )

        state = adapter.current_state("2330", EventCursor(_time_ns(300), 200, 1))

        assert state is not None
        self.assertFalse(state.base_gate_open)
        self.assertEqual(state.gate_reason, "empty_future_book")
        self.assertEqual(state.base_gate_book_wake_venues, frozenset(("future",)))


if __name__ == "__main__":
    unittest.main()
