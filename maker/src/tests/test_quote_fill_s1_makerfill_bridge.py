"""Contracts for the audited S1 legacy makerFill bridge."""

from __future__ import annotations

import math
import unittest

import polars as pl

from ..quote_fill.layered import EventCursor
from ..quote_fill.makerfill_adapter import MakerFillLabelIndex
from ..quote_fill.s1_event_loop import ActualSendMakerSnapshot, SentEntryOrder
from ..quote_fill.s1_makerfill_bridge import S1MakerFillBridge
from ..quote_fill.targets import absolute_price_tick


def _index(fill_seconds: float) -> MakerFillLabelIndex:
    return MakerFillLabelIndex.from_frame(
        pl.DataFrame(
            {
                "QuoteCode": ["2330"],
                "ChannelSeq": [17],
                "Bid1_FillSeconds": [fill_seconds],
                "Bid2_FillSeconds": [2.0],
            },
            schema_overrides={
                "ChannelSeq": pl.UInt64,
                "Bid1_FillSeconds": pl.Float32,
                "Bid2_FillSeconds": pl.Float32,
            },
        )
    )


def _order(
    snapshot: ActualSendMakerSnapshot,
    *,
    target_price: float = 100.0,
) -> SentEntryOrder:
    return SentEntryOrder(
        date="20260505",
        policy_id="q95",
        product_id="2330",
        value_code="2330",
        quote_code="CDFE6",
        request_id="request-1",
        candidate_intent_id="candidate-1",
        raw_order_fact_id="raw-1",
        policy_alias_id="alias-1",
        capacity_id="candidate-1",
        actual_start_cursor=EventCursor(1_100_000_000, 700, 1),
        absolute_price_tick=absolute_price_tick(target_price),
        target_price=target_price,
        reservation_notional_twd=200_000,
        contract_size_shares=2_000,
        maker_snapshot=snapshot,
        frozen_exit_threshold_basis_bp=5.0,
    )


class S1MakerFillBridgeTest(unittest.TestCase):
    def test_supported_fill_is_scheduled_and_audited(self) -> None:
        snapshot = ActualSendMakerSnapshot(
            "2330",
            17,
            1_000_000_000,
            100.0,
            99.9,
            10,
            20,
        )
        order = _order(snapshot)
        bridge = S1MakerFillBridge(_index(1.0))

        potential = bridge.potential_fill(order, snapshot)

        self.assertIsNotNone(potential)
        assert potential is not None
        self.assertEqual(potential.fill_time_ns, 2_000_000_000)
        self.assertEqual(potential.execution_truth, "approximate")
        self.assertFalse(potential.fill_cursor_exact)
        self.assertIn("/2330/17/BID1", potential.source_id)
        self.assertEqual(len(bridge.assessments), 1)
        event = bridge.assessments[0]
        self.assertTrue(event.outcome_supported)
        self.assertEqual(event.potential_outcome_status, "approx_potential_fill")
        self.assertEqual(
            bridge.assessments_by_raw_id["raw-1"],
            event,
        )
        self.assertEqual(bridge.assessment_rows()[0]["raw_order_fact_id"], "raw-1")

    def test_eod_no_fill_and_unsupported_rank_remain_distinct(self) -> None:
        snapshot = ActualSendMakerSnapshot(
            "2330",
            17,
            1_000_000_000,
            100.0,
            99.9,
            10,
            20,
        )
        no_fill = S1MakerFillBridge(_index(math.nan))
        self.assertIsNone(no_fill.potential_fill(_order(snapshot), snapshot))
        self.assertEqual(
            no_fill.assessments[0].potential_outcome_status,
            "approx_no_fill_through_eod",
        )
        self.assertTrue(no_fill.assessments[0].outcome_supported)

        unsupported = S1MakerFillBridge(_index(1.0))
        self.assertIsNone(
            unsupported.potential_fill(
                _order(snapshot, target_price=99.8),
                snapshot,
            )
        )
        self.assertEqual(
            unsupported.assessments[0].potential_outcome_status,
            "unsupported_target_not_displayed_l1_l2",
        )
        self.assertFalse(unsupported.assessments[0].outcome_supported)

    def test_duplicate_or_mismatched_assessment_fails_closed(self) -> None:
        snapshot = ActualSendMakerSnapshot(
            "2330",
            17,
            1_000_000_000,
            100.0,
            99.9,
            10,
            20,
        )
        order = _order(snapshot)
        bridge = S1MakerFillBridge(_index(1.0))
        bridge.potential_fill(order, snapshot)
        with self.assertRaisesRegex(ValueError, "already"):
            bridge.potential_fill(order, snapshot)

        other = ActualSendMakerSnapshot(
            "2330",
            18,
            1_000_000_001,
            100.0,
            99.9,
            10,
            20,
        )
        with self.assertRaisesRegex(ValueError, "differs"):
            S1MakerFillBridge(_index(1.0)).potential_fill(order, other)


if __name__ == "__main__":
    unittest.main()
