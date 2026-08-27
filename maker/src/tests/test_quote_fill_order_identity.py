from __future__ import annotations

import unittest

from ..quote_fill.layered import EventCursor
from ..quote_fill.order_identity import (
    CANDIDATE_INTENT_ID_FIELDS,
    RAW_ORDER_FACT_ID_FIELDS,
    candidate_intent_id,
    forbidden_raw_order_identity_fields,
    policy_alias_id,
    raw_order_fact_id,
    verify_candidate_intent_record,
    verify_policy_alias_record,
    verify_raw_order_fact_record,
)

COMMON = {
    "Date": "20260826",
    "ValueCode": "2330",
    "QuoteCode": "2330F",
    "route": "spot_bid",
    "stage": "entry",
    "maker_side": "bid",
    "absolute_price_tick": 1045,
}


class OrderIdentityTest(unittest.TestCase):
    def test_raw_id_is_policy_independent_and_requires_actual_send(self) -> None:
        cursor = EventCursor(100, 4, 9)
        first = raw_order_fact_id(**COMMON, actual_start_cursor=cursor)
        second = raw_order_fact_id(
            **dict(reversed(list(COMMON.items()))), actual_start_cursor=cursor
        )
        self.assertEqual(first, second)
        self.assertNotIn("policy", first)
        with self.assertRaisesRegex(ValueError, "actual send cursor"):
            raw_order_fact_id(**COMMON, actual_start_cursor=None)

    def test_every_physical_field_changes_raw_id(self) -> None:
        baseline = raw_order_fact_id(
            **COMMON, actual_start_cursor=EventCursor(100, 4, 9)
        )
        variants = [
            {**COMMON, "Date": "20260827"},
            {**COMMON, "ValueCode": "2317"},
            {**COMMON, "QuoteCode": "2317F"},
            {**COMMON, "route": "future_ask"},
            {**COMMON, "stage": "exit"},
            {**COMMON, "maker_side": "ask"},
            {**COMMON, "absolute_price_tick": 1046},
        ]
        ids = {
            raw_order_fact_id(**variant, actual_start_cursor=EventCursor(100, 4, 9))
            for variant in variants
        }
        ids.update(
            raw_order_fact_id(**COMMON, actual_start_cursor=cursor)
            for cursor in (
                EventCursor(101, 4, 9),
                EventCursor(100, 5, 9),
                EventCursor(100, 4, 10),
            )
        )
        self.assertEqual(len(ids), len(variants) + 3)
        self.assertNotIn(baseline, ids)

    def test_policy_alias_changes_without_changing_raw_id(self) -> None:
        raw = raw_order_fact_id(**COMMON, actual_start_cursor=EventCursor(100, 4, 9))
        self.assertNotEqual(
            policy_alias_id(raw_order_fact_id=raw, policy_id="A"),
            policy_alias_id(raw_order_fact_id=raw, policy_id="B"),
        )

    def test_candidate_has_pre_send_cursor_and_no_policy(self) -> None:
        intent = candidate_intent_id(**COMMON, intent_cursor=EventCursor(80, 2, 3))
        self.assertEqual(
            intent,
            candidate_intent_id(**COMMON, intent_cursor=EventCursor(80, 2, 3)),
        )
        self.assertIn("policy_id", forbidden_raw_order_identity_fields())
        self.assertIn("generation", forbidden_raw_order_identity_fields())

    def test_record_verifiers_detect_mutation(self) -> None:
        candidate_record = {
            **COMMON,
            "intent_recv_time_ns": 80,
            "intent_event_sequence": 2,
            "intent_row_index": 3,
        }
        candidate_record["candidate_intent_id"] = candidate_intent_id(
            **COMMON, intent_cursor=EventCursor(80, 2, 3)
        )
        verify_candidate_intent_record(candidate_record)
        self.assertEqual(
            CANDIDATE_INTENT_ID_FIELDS[-3:],
            (
                "intent_recv_time_ns",
                "intent_event_sequence",
                "intent_row_index",
            ),
        )

        raw_record = {
            **COMMON,
            "actual_start_recv_time_ns": 100,
            "actual_start_event_sequence": 4,
            "actual_start_row_index": 9,
        }
        raw_record["raw_order_fact_id"] = raw_order_fact_id(
            **COMMON, actual_start_cursor=EventCursor(100, 4, 9)
        )
        verify_raw_order_fact_record(raw_record)
        mutated = {**raw_record, "actual_start_row_index": 10}
        with self.assertRaisesRegex(ValueError, "does not match"):
            verify_raw_order_fact_record(mutated)
        self.assertEqual(len(RAW_ORDER_FACT_ID_FIELDS), 10)

        alias = {
            "raw_order_fact_id": raw_record["raw_order_fact_id"],
            "policy_id": "Q2",
        }
        alias["policy_alias_id"] = policy_alias_id(**alias)
        verify_policy_alias_record(alias)
        with self.assertRaisesRegex(ValueError, "does not match"):
            verify_policy_alias_record({**alias, "policy_id": "fixed"})

    def test_validation_rejects_ambiguous_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive integer"):
            raw_order_fact_id(
                **{**COMMON, "absolute_price_tick": True},
                actual_start_cursor=EventCursor(1),
            )
        with self.assertRaisesRegex(ValueError, "missing fields"):
            verify_raw_order_fact_record({"raw_order_fact_id": "x"})
        with self.assertRaisesRegex(ValueError, "policy_id"):
            verify_policy_alias_record(
                {
                    "policy_alias_id": "x",
                    "raw_order_fact_id": "raw_order_fact_v1_x",
                    "policy_id": 1,
                }
            )


if __name__ == "__main__":
    unittest.main()
