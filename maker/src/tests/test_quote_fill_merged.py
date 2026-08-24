from __future__ import annotations

from datetime import datetime
import unittest

import polars as pl

from maker.src.quote_fill.merged import (
    _MarketState,
    _refresh_maker_rank_and_queue_from_full_state,
    _rank_and_queue,
    _route_state,
    _thin_market_events,
    observations_for_policy,
    session_cutoff_cursor,
)


def _state(*, market: str, trial: bool = False, raw_book: bool = True) -> dict:
    row = {
        "market": market,
        "trial_match": trial,
        "raw_has_book": raw_book,
        "ref_price": 100.0,
        "recv_time_ns": 100,
        "book_recv_time_ns": 90,
        "bid_price_1": 99.0,
        "bid_lots_1": 5,
        "ask_price_1": 101.0,
        "ask_lots_1": 6,
        "exec_bid_price": 99.0,
        "exec_bid_lots": 5,
        "exec_ask_price": 101.0,
        "exec_ask_lots": 6,
        "best_bid_price": None,
        "best_bid_lots": None,
        "best_ask_price": None,
        "best_ask_lots": None,
    }
    for level in range(2, 6):
        row[f"bid_price_{level}"] = 99.0 - level
        row[f"bid_lots_{level}"] = 5 + level
        row[f"ask_price_{level}"] = 101.0 + level
        row[f"ask_lots_{level}"] = 6 + level
    return row


class MergedStateTest(unittest.TestCase):
    def test_formal_book_requires_new_quote_after_trial(self) -> None:
        state = _MarketState()
        state.update(_state(market="future"))
        self.assertTrue(state.formal_after_trial)
        state.update(_state(market="future", trial=True, raw_book=False))
        self.assertFalse(state.formal_after_trial)
        state.update(_state(market="future", raw_book=False))
        self.assertFalse(state.formal_after_trial)
        state.update(_state(market="future", raw_book=True))
        self.assertTrue(state.formal_after_trial)

    def test_raw_execution_price_controls_passivity(self) -> None:
        spot = _MarketState()
        future = _MarketState()
        spot.update(_state(market="spot"))
        raw_future = _state(market="future")
        raw_future["exec_bid_price"] = 100.0
        raw_future["exec_ask_price"] = 101.0
        future.update(raw_future)
        result = _route_state(
            "future_ask_spot_taker",
            0.0,
            {
                "upper_distance_bp": 0.0,
                "adaptive_parameter_valid": True,
            },
            spot,
            future,
        )
        self.assertEqual(result["absolute_target_price"], 101.0)
        self.assertTrue(result["gate_open"])

        raw_future["exec_bid_price"] = 101.0
        future.update(raw_future)
        blocked = _route_state(
            "future_ask_spot_taker",
            0.0,
            {
                "upper_distance_bp": 0.0,
                "adaptive_parameter_valid": True,
            },
            spot,
            future,
        )
        self.assertFalse(blocked["gate_open"])
        self.assertEqual(blocked["admission_reason"], "target_not_passive")

    def test_queue_uses_exact_visible_level_and_inside_is_zero(self) -> None:
        row = _state(market="spot")
        self.assertEqual(_rank_and_queue(99.0, "bid", row), ("BID1", 5))
        self.assertEqual(_rank_and_queue(100.0, "bid", row), ("inside", 0))
        self.assertEqual(
            _rank_and_queue(90.0, "bid", row), ("behind_visible", None)
        )

    def test_queue_is_rejoined_from_full_lots_only_state_causally(self) -> None:
        rows = []
        for recv_ns, sequence, lots in (
            (10, 1, 5),
            (20, 2, 11),  # lots-only update omitted by sparse target events
            (30, 3, 99),  # same ns but spot priority follows future trigger
        ):
            row = _state(market="spot")
            row.update(
                {
                    "recv_time_ns": recv_ns,
                    "sequence": sequence,
                    "packet_sequence": sequence,
                    "bid_lots_1": lots,
                    "exec_bid_lots": lots,
                }
            )
            rows.append(row)
        spot_full = pl.from_dicts(rows, infer_schema_length=None)
        records = [
            {
                "maker_market": "spot",
                "maker_side": "bid",
                "absolute_target_price": 99.0,
                "recv_time_ns": 30,
                "cursor_event_sequence": 1,  # opposite future event
                "cursor_row_index": 100,
                "target_rank": "stale",
                "initial_queue_ahead": -1,
            }
        ]

        _refresh_maker_rank_and_queue_from_full_state(
            records,
            spot_full,
            pl.DataFrame(),
        )

        self.assertEqual(records[0]["target_rank"], "BID1")
        self.assertEqual(records[0]["initial_queue_ahead"], 11)

    def test_thinning_retains_delayed_formal_book_after_trial(self) -> None:
        rows = []
        for recv_ns, sequence, trial, raw_book in (
            (10, 1, False, True),
            (20, 2, True, False),
            (30, 3, False, False),
            (40, 4, False, True),
        ):
            row = _state(market="future", trial=trial, raw_book=raw_book)
            row.update(
                {
                    "Date": "20260128",
                    "ValueCode": "2317",
                    "recv_time": datetime(2026, 1, 28, 1, 5, 0, recv_ns),
                    "recv_time_ns": recv_ns,
                    "sequence": sequence,
                    "packet_sequence": sequence,
                }
            )
            rows.append(row)

        thinned = _thin_market_events(
            pl.from_dicts(rows, infer_schema_length=None),
            include_base_epoch=False,
        )

        self.assertIn(4, thinned["sequence"].to_list())
        state = _MarketState()
        for row in thinned.iter_rows(named=True):
            state.update(row)
        self.assertTrue(state.formal_after_trial)

    def test_cutoff_is_1320_taipei_in_utc_nanoseconds(self) -> None:
        cursor = session_cutoff_cursor("20260128")
        expected = datetime(2026, 1, 28, 5, 20) - datetime(1970, 1, 1)
        self.assertEqual(
            cursor.recv_time_ns,
            int(expected.total_seconds() * 1_000_000_000),
        )

    def test_empty_observation_day_preserves_zero_order_support(self) -> None:
        self.assertEqual(
            observations_for_policy(
                pl.DataFrame(),
                date="20260622",
                value_code="2303",
                route="future_ask_spot_taker",
                boundary_quantile=95,
            ),
            (),
        )


if __name__ == "__main__":
    unittest.main()
