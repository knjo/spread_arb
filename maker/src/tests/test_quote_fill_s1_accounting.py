from __future__ import annotations

import json
import unittest
from dataclasses import replace

from ..quote_fill.layered import EventCursor
from ..quote_fill.s1_accounting import (
    AccountingCodecError,
    AccountingError,
    AccountingReplayError,
    ExpiryAccountingMark,
    InitiatingExecutionAllocation,
    InventoryNotFlatError,
    S1AccountingLedger,
    decode_accounting_fact,
    decode_accounting_facts,
    encode_accounting_fact,
    encode_accounting_facts,
    replay_accounting_facts,
)


def cursor(value: int) -> EventCursor:
    return EventCursor(value, value % 3, value % 2)


def execution_common(
    execution_id: str,
    position_id: str,
    cursor_value: int,
    *,
    execution_date: str = "20260825",
    role: str = "normal",
    truth: str = "exact",
    initiating_execution_id: str | None = None,
) -> dict[str, object]:
    linked = role in ("hedge", "rollback")
    return {
        "execution_id": execution_id,
        "position_id": position_id,
        "value_code": "2330",
        "scenario_id": "spot_bid_primary",
        "capacity_id": f"cap-{position_id}",
        "role": role,
        "execution_truth": truth,
        "execution_source_id": (
            "legacy_makerfill" if truth == "approximate" else "indexed_replay"
        ),
        "request_id": f"request-{execution_id}",
        "hedge_intent_id": f"hedge-{execution_id}" if linked else None,
        "initiating_execution_id": initiating_execution_id,
        "execution_date": execution_date,
        "cursor": cursor(cursor_value),
    }


class S1AccountingLedgerTest(unittest.TestCase):
    def _open_pair(
        self,
        ledger: S1AccountingLedger,
        *,
        position_id: str = "p",
        start: int = 1,
        execution_date: str = "20260825",
        truth: str = "exact",
        shares: int = 2000,
    ) -> int:
        ledger.record_spot_execution(
            **execution_common(
                f"{position_id}-spot-open",
                position_id,
                start,
                execution_date=execution_date,
                truth=truth,
            ),
            side="buy",
            price=100,
            shares=shares,
        )
        ledger.record_future_execution(
            **execution_common(
                f"{position_id}-future-open",
                position_id,
                start + 1,
                execution_date=execution_date,
                role="hedge",
                initiating_execution_id=f"{position_id}-spot-open",
            ),
            side="sell",
            price=103,
            contracts=1,
            share_equivalent=shares,
        )
        ledger.establish_position(
            establishment_id=f"{position_id}-established",
            position_id=position_id,
            establishment_date=execution_date,
            cursor=cursor(start + 2),
            position_established_ns=start + 2,
            capacity_transition_id=f"{position_id}-paired-capacity",
        )
        return start + 3

    def _mixed_fact_ledger(self) -> S1AccountingLedger:
        ledger = S1AccountingLedger()
        self._open_pair(ledger, position_id="flat", start=1)
        ledger.record_spot_execution(
            **execution_common("flat-spot-close", "flat", 4),
            side="sell",
            price=101,
            shares=2000,
        )
        ledger.record_future_execution(
            **execution_common(
                "flat-future-close",
                "flat",
                5,
                role="hedge",
                initiating_execution_id="flat-spot-close",
            ),
            side="buy",
            price=101,
            contracts=1,
            share_equivalent=2000,
        )
        ledger.seal_terminal(
            terminal_id="flat-terminal",
            position_id="flat",
            terminal_date="20260825",
            cursor=cursor(6),
            terminal_outcome="exit_maker_flat",
            capacity_release_transition_id="flat-release",
        )
        self._open_pair(ledger, position_id="expiry", start=7)
        ledger.record_expiry_accounting_mark(
            mark_id="expiry-mark",
            position_id="expiry",
            expiry_date="20260825",
            cursor=cursor(10),
            spot_close_source_id="official-close",
            spot_close_source_cursor=EventCursor(9, 2, 0),
            spot_close_price=101,
            capacity_release_transition_id="expiry-release",
        )
        return ledger.verify()

    def test_json_fact_codec_roundtrips_mixed_verified_stream(self) -> None:
        ledger = self._mixed_fact_ledger()
        records = encode_accounting_facts(ledger.facts)

        self.assertEqual(
            {record["fact_type"] for record in records},
            {
                "executed_leg",
                "position_established",
                "terminal_realized_accounting",
                "expiry_accounting_mark",
            },
        )
        self.assertTrue(
            all(isinstance(record.get("allocations", []), list) for record in records)
        )
        wire_records = json.loads(json.dumps(records, allow_nan=False, sort_keys=True))
        decoded = decode_accounting_facts(wire_records)

        self.assertEqual(decoded, ledger.facts)
        self.assertEqual(replay_accounting_facts(decoded).verify().facts, ledger.facts)
        for fact in ledger.facts:
            self.assertEqual(
                decode_accounting_fact(encode_accounting_fact(fact)),
                fact,
            )

    def test_json_fact_codec_rejects_schema_and_type_tampering(self) -> None:
        records = json.loads(
            json.dumps(encode_accounting_facts(self._mixed_fact_ledger().facts))
        )
        base = records[0]

        unknown_type = {**base, "fact_type": "unknown"}
        with self.assertRaisesRegex(AccountingCodecError, "unknown fact_type"):
            decode_accounting_fact(unknown_type)

        missing = dict(base)
        missing.pop("sequence")
        with self.assertRaisesRegex(AccountingCodecError, "missing=.*sequence"):
            decode_accounting_fact(missing)

        unknown = {**base, "unexpected": 1}
        with self.assertRaisesRegex(AccountingCodecError, "unknown=.*unexpected"):
            decode_accounting_fact(unknown)

        scalar_type_drift = {**base, "sequence": "1"}
        with self.assertRaisesRegex(AccountingCodecError, "sequence.*type"):
            decode_accounting_fact(scalar_type_drift)

        cursor_type_drift = {**base, "recv_time_ns": "1"}
        with self.assertRaisesRegex(AccountingCodecError, "cursor fields"):
            decode_accounting_fact(cursor_type_drift)

        linked = next(
            record
            for record in records
            if record.get("initiating_execution_allocations")
        )
        nested = json.loads(json.dumps(linked))
        nested["initiating_execution_allocations"][0]["share_equivalent"] = "2000"
        with self.assertRaisesRegex(AccountingCodecError, "share_equivalent.*type"):
            decode_accounting_fact(nested)

        allocated = next(record for record in records if record.get("allocations"))
        nested_schema = json.loads(json.dumps(allocated))
        nested_schema["allocations"][0].pop("same_day")
        with self.assertRaisesRegex(AccountingCodecError, "schema mismatch"):
            decode_accounting_fact(nested_schema)

    def test_clean_pair_seals_terminal_and_replays(self) -> None:
        ledger = S1AccountingLedger()
        self._open_pair(ledger)
        ledger.record_spot_execution(
            **execution_common("spot-close", "p", 4),
            side="sell",
            price=101,
            shares=2000,
        )
        future_close = ledger.record_future_execution(
            **execution_common(
                "future-close",
                "p",
                5,
                role="hedge",
                initiating_execution_id="spot-close",
            ),
            side="buy",
            price=101,
            contracts=1,
            share_equivalent=2000,
        )
        self.assertAlmostEqual(future_close.realized_pnl_twd, 4000)
        terminal = ledger.seal_terminal(
            terminal_id="terminal-p",
            position_id="p",
            terminal_date="20260825",
            cursor=cursor(6),
            terminal_outcome="exit_maker_flat",
            capacity_release_transition_id="capacity-release-p",
        )
        self.assertAlmostEqual(terminal.spot_cashflow_twd, 2000)
        self.assertAlmostEqual(
            terminal.realized_net_twd,
            6000 - terminal.commission_twd - terminal.tax_twd,
        )
        self.assertEqual(terminal.execution_truth, "exact")
        replayed = ledger.verify()
        self.assertEqual(len(replayed.facts), len(ledger.facts))

    def test_approximate_execution_truth_propagates_to_terminal(self) -> None:
        ledger = S1AccountingLedger()
        ledger.record_spot_execution(
            **execution_common("entry", "p", 1, truth="approximate"),
            side="buy",
            price=100,
            shares=1000,
        )
        ledger.record_spot_execution(
            **execution_common(
                "rollback",
                "p",
                2,
                role="rollback",
                initiating_execution_id="entry",
            ),
            side="sell",
            price=99,
            shares=1000,
        )
        terminal = ledger.seal_terminal(
            terminal_id="terminal",
            position_id="p",
            terminal_date="20260825",
            cursor=cursor(3),
            terminal_outcome="entry_emergency_rollback_flat",
            capacity_release_transition_id="release",
        )
        self.assertEqual(terminal.execution_truth, "approximate")

    def test_spot_fifo_splits_same_day_and_overnight_tax(self) -> None:
        ledger = S1AccountingLedger()
        ledger.record_spot_execution(
            **execution_common("old", "p", 1, execution_date="20260824"),
            side="buy",
            price=99,
            shares=1000,
        )
        ledger.record_spot_execution(
            **execution_common("new", "p", 2),
            side="buy",
            price=100,
            shares=1000,
        )
        ledger.record_future_execution(
            **execution_common("future", "p", 3),
            side="sell",
            price=103,
            contracts=1,
            share_equivalent=2000,
        )
        ledger.establish_position(
            establishment_id="established",
            position_id="p",
            establishment_date="20260825",
            cursor=cursor(4),
            position_established_ns=4,
            capacity_transition_id="paired-capacity",
        )
        close = ledger.record_spot_execution(
            **execution_common("sell", "p", 5),
            side="sell",
            price=101,
            shares=1500,
        )
        self.assertEqual(
            [allocation.quantity for allocation in close.allocations],
            [1000, 500],
        )
        self.assertEqual(
            [allocation.same_day for allocation in close.allocations],
            [False, True],
        )
        expected = ledger.profile.spot_sell_tax_twd(
            101, 1000, same_day=False
        ) + ledger.profile.spot_sell_tax_twd(101, 500, same_day=True)
        self.assertAlmostEqual(close.tax_twd, expected)

    def test_product_fifo_rejects_skipping_older_position(self) -> None:
        ledger = S1AccountingLedger()
        self._open_pair(
            ledger,
            position_id="old-position",
            start=1,
            execution_date="20260824",
            shares=1000,
        )
        self._open_pair(
            ledger,
            position_id="new-position",
            start=4,
            shares=1000,
        )
        with self.assertRaisesRegex(AccountingError, "canonical"):
            ledger.record_spot_execution(
                **execution_common("bad-close", "new-position", 7),
                side="sell",
                price=101,
                shares=1000,
            )

    def test_future_fifo_rejects_skipping_older_position(self) -> None:
        ledger = S1AccountingLedger()
        self._open_pair(
            ledger,
            position_id="old-position",
            start=1,
        )
        self._open_pair(
            ledger,
            position_id="new-position",
            start=4,
        )
        with self.assertRaisesRegex(AccountingError, "canonical"):
            ledger.record_future_execution(
                **execution_common("bad-close", "new-position", 7),
                side="buy",
                price=101,
                contracts=1,
                share_equivalent=2000,
            )

    def test_aggregate_spot_fifo_can_reach_p2_before_p1_future_hedge(self) -> None:
        ledger = S1AccountingLedger()
        self._open_pair(ledger, position_id="p1", start=1)
        self._open_pair(ledger, position_id="p2", start=4)

        p1_spot_exit = execution_common("p1-spot-exit", "p1", 7)
        p1_spot_exit["cursor"] = EventCursor(7, 0, 0)
        ledger.record_spot_execution(
            **p1_spot_exit,
            side="sell",
            price=101,
            shares=2000,
        )
        p2_spot_exit = execution_common("p2-spot-exit", "p2", 7)
        p2_spot_exit["cursor"] = EventCursor(7, 0, 1)
        ledger.record_spot_execution(
            **p2_spot_exit,
            side="sell",
            price=101,
            shares=2000,
        )
        ledger.record_future_execution(
            **execution_common(
                "p1-future-exit",
                "p1",
                8,
                role="hedge",
                initiating_execution_id="p1-spot-exit",
            ),
            side="buy",
            price=101,
            contracts=1,
            share_equivalent=2000,
        )
        ledger.record_future_execution(
            **execution_common(
                "p2-future-exit",
                "p2",
                9,
                role="hedge",
                initiating_execution_id="p2-spot-exit",
            ),
            side="buy",
            price=101,
            contracts=1,
            share_equivalent=2000,
        )

        self.assertEqual(ledger.inventory("p1"), (0, 0, 0))
        self.assertEqual(ledger.inventory("p2"), (0, 0, 0))
        ledger.verify()

    def test_aggregate_spot_fifo_still_rejects_p2_before_p1_spot(self) -> None:
        ledger = S1AccountingLedger()
        self._open_pair(ledger, position_id="p1", start=1)
        self._open_pair(ledger, position_id="p2", start=4)

        with self.assertRaisesRegex(AccountingError, "spot-leg FIFO"):
            ledger.record_spot_execution(
                **execution_common("p2-spot-exit", "p2", 7),
                side="sell",
                price=101,
                shares=2000,
            )

    def test_aggregate_future_fifo_can_reach_p2_before_p1_spot_hedge(self) -> None:
        ledger = S1AccountingLedger()
        self._open_pair(ledger, position_id="p1", start=1)
        self._open_pair(ledger, position_id="p2", start=4)

        p1_future_exit = execution_common("p1-future-exit", "p1", 7)
        p1_future_exit["cursor"] = EventCursor(7, 0, 0)
        ledger.record_future_execution(
            **p1_future_exit,
            side="buy",
            price=101,
            contracts=1,
            share_equivalent=2000,
        )
        p2_future_exit = execution_common("p2-future-exit", "p2", 7)
        p2_future_exit["cursor"] = EventCursor(7, 0, 1)
        ledger.record_future_execution(
            **p2_future_exit,
            side="buy",
            price=101,
            contracts=1,
            share_equivalent=2000,
        )
        ledger.record_spot_execution(
            **execution_common(
                "p1-spot-exit",
                "p1",
                8,
                role="hedge",
                initiating_execution_id="p1-future-exit",
            ),
            side="sell",
            price=101,
            shares=2000,
        )
        ledger.record_spot_execution(
            **execution_common(
                "p2-spot-exit",
                "p2",
                9,
                role="hedge",
                initiating_execution_id="p2-future-exit",
            ),
            side="sell",
            price=101,
            shares=2000,
        )

        self.assertEqual(ledger.inventory("p1"), (0, 0, 0))
        self.assertEqual(ledger.inventory("p2"), (0, 0, 0))
        ledger.verify()

    def test_forced_first_leg_keeps_whole_position_fifo(self) -> None:
        ledger = S1AccountingLedger()
        self._open_pair(ledger, position_id="p1", start=1)
        self._open_pair(ledger, position_id="p2", start=4)

        ledger.record_spot_execution(
            **execution_common("p1-forced-spot", "p1", 7, role="forced"),
            side="sell",
            price=100,
            shares=2000,
        )
        with self.assertRaisesRegex(AccountingError, "canonical"):
            ledger.record_spot_execution(
                **execution_common("p2-forced-spot", "p2", 8, role="forced"),
                side="sell",
                price=100,
                shares=2000,
            )
        ledger.record_future_execution(
            **execution_common(
                "p1-forced-future",
                "p1",
                8,
                role="hedge",
                initiating_execution_id="p1-forced-spot",
            ),
            side="buy",
            price=100,
            contracts=1,
            share_equivalent=2000,
        )
        ledger.record_spot_execution(
            **execution_common("p2-forced-spot", "p2", 9, role="forced"),
            side="sell",
            price=100,
            shares=2000,
        )

        self.assertEqual(ledger.inventory("p1"), (0, 0, 0))
        self.assertEqual(ledger.inventory("p2"), (0, -1, -2000))
        ledger.verify()

    def test_position_establishment_is_causal_unique_and_required_for_exit(
        self,
    ) -> None:
        ledger = S1AccountingLedger()
        ledger.record_spot_execution(
            **execution_common("spot-open", "p", 1),
            side="buy",
            price=100,
            shares=2000,
        )
        with self.assertRaisesRegex(AccountingError, "exactly paired"):
            ledger.establish_position(
                establishment_id="too-early",
                position_id="p",
                establishment_date="20260825",
                cursor=cursor(2),
                position_established_ns=2,
                capacity_transition_id="paired-capacity",
            )
        ledger.record_future_execution(
            **execution_common(
                "future-open",
                "p",
                2,
                role="hedge",
                initiating_execution_id="spot-open",
            ),
            side="sell",
            price=103,
            contracts=1,
            share_equivalent=2000,
        )
        with self.assertRaisesRegex(AccountingError, "unestablished"):
            ledger.record_spot_execution(
                **execution_common("exit-before-establishment", "p", 3),
                side="sell",
                price=101,
                shares=2000,
            )
        with self.assertRaisesRegex(AccountingError, "backfilled"):
            ledger.establish_position(
                establishment_id="backfilled",
                position_id="p",
                establishment_date="20260825",
                cursor=cursor(3),
                position_established_ns=2,
                capacity_transition_id="paired-capacity",
            )
        established = ledger.establish_position(
            establishment_id="established",
            position_id="p",
            establishment_date="20260825",
            cursor=cursor(3),
            position_established_ns=3,
            capacity_transition_id="paired-capacity",
        )
        self.assertEqual(established.fifo_key, (3, "p"))
        with self.assertRaisesRegex(AccountingError, "cannot be repeated"):
            ledger.establish_position(
                establishment_id="duplicate",
                position_id="p",
                establishment_date="20260825",
                cursor=cursor(4),
                position_established_ns=4,
                capacity_transition_id="paired-capacity",
            )
        with self.assertRaises(AccountingReplayError):
            replay_accounting_facts(
                (
                    *ledger.rows,
                    replace(established, position_established_ns=2),
                )
            )

    def test_position_fifo_tie_breaks_by_position_id(self) -> None:
        ledger = S1AccountingLedger()

        def at_cursor(
            execution_id: str,
            position_id: str,
            event_cursor: EventCursor,
            *,
            role: str = "normal",
            initiating_execution_id: str | None = None,
        ) -> dict[str, object]:
            values = execution_common(
                execution_id,
                position_id,
                1,
                role=role,
                initiating_execution_id=initiating_execution_id,
            )
            values["cursor"] = event_cursor
            return values

        ledger.record_spot_execution(
            **at_cursor("b-spot", "p-b", EventCursor(1, 0, 0)),
            side="buy",
            price=100,
            shares=1000,
        )
        ledger.record_future_execution(
            **at_cursor(
                "b-future",
                "p-b",
                EventCursor(1, 0, 1),
                role="hedge",
                initiating_execution_id="b-spot",
            ),
            side="sell",
            price=103,
            contracts=1,
            share_equivalent=1000,
        )
        ledger.record_spot_execution(
            **at_cursor("a-spot", "p-a", EventCursor(1, 0, 2)),
            side="buy",
            price=100,
            shares=1000,
        )
        ledger.record_future_execution(
            **at_cursor(
                "a-future",
                "p-a",
                EventCursor(1, 0, 3),
                role="hedge",
                initiating_execution_id="a-spot",
            ),
            side="sell",
            price=103,
            contracts=1,
            share_equivalent=1000,
        )
        ledger.establish_position(
            establishment_id="b-established",
            position_id="p-b",
            establishment_date="20260825",
            cursor=EventCursor(2, 0, 0),
            position_established_ns=2,
            capacity_transition_id="b-paired-capacity",
        )
        ledger.establish_position(
            establishment_id="a-established",
            position_id="p-a",
            establishment_date="20260825",
            cursor=EventCursor(2, 0, 1),
            position_established_ns=2,
            capacity_transition_id="a-paired-capacity",
        )
        with self.assertRaisesRegex(AccountingError, "canonical"):
            ledger.record_spot_execution(
                **at_cursor("b-exit", "p-b", EventCursor(3, 0, 0)),
                side="sell",
                price=101,
                shares=1000,
            )
        first = ledger.record_spot_execution(
            **at_cursor("a-exit", "p-a", EventCursor(3, 0, 0)),
            side="sell",
            price=101,
            shares=1000,
        )
        self.assertEqual(first.position_id, "p-a")

    def test_rollback_requires_prior_same_market_opposite_exact_quantity(self) -> None:
        ledger = S1AccountingLedger()
        ledger.record_spot_execution(
            **execution_common("entry", "p", 1),
            side="buy",
            price=100,
            shares=2000,
        )
        with self.assertRaisesRegex(AccountingError, "quantity must match"):
            ledger.record_spot_execution(
                **execution_common(
                    "bad-rollback",
                    "p",
                    2,
                    role="rollback",
                    initiating_execution_id="entry",
                ),
                side="sell",
                price=99,
                shares=1000,
            )
        rollback = ledger.record_spot_execution(
            **execution_common(
                "rollback",
                "p",
                2,
                role="rollback",
                initiating_execution_id="entry",
            ),
            side="sell",
            price=99,
            shares=2000,
        )
        self.assertEqual(rollback.allocations[0].lot_id, "entry:spot")

    def test_multi_source_spot_fills_fund_one_future_hedge(self) -> None:
        ledger = S1AccountingLedger()
        ledger.record_spot_execution(
            **execution_common("spot-600", "p", 1),
            side="buy",
            price=100,
            shares=600,
        )
        ledger.record_spot_execution(
            **execution_common("spot-1400", "p", 2),
            side="buy",
            price=100,
            shares=1400,
        )
        hedge = ledger.record_future_execution(
            **execution_common("future-hedge", "p", 3, role="hedge"),
            initiating_execution_allocations=(
                InitiatingExecutionAllocation("spot-600", 600),
                InitiatingExecutionAllocation("spot-1400", 1400),
            ),
            side="sell",
            price=103,
            contracts=1,
            share_equivalent=2000,
        )
        self.assertEqual(
            hedge.initiating_execution_allocations,
            (
                InitiatingExecutionAllocation("spot-600", 600),
                InitiatingExecutionAllocation("spot-1400", 1400),
            ),
        )
        self.assertIsNone(hedge.initiating_execution_id)
        self.assertEqual(ledger.inventory("p"), (2000, -1, -2000))
        ledger.verify()

    def test_one_4000_share_source_funds_two_hedges_without_overuse(self) -> None:
        ledger = S1AccountingLedger()
        ledger.record_spot_execution(
            **execution_common("spot-4000", "p", 1),
            side="buy",
            price=100,
            shares=4000,
        )
        allocation = (InitiatingExecutionAllocation("spot-4000", 2000),)
        first = ledger.record_future_execution(
            **execution_common("future-hedge-1", "p", 2, role="hedge"),
            initiating_execution_allocations=allocation,
            side="sell",
            price=103,
            contracts=1,
            share_equivalent=2000,
        )
        self.assertEqual(first.initiating_execution_id, "spot-4000")
        with self.assertRaisesRegex(AccountingError, "exceeds unconsumed"):
            ledger.record_future_execution(
                **execution_common("future-overuse", "p", 3, role="hedge"),
                initiating_execution_allocations=(
                    InitiatingExecutionAllocation("spot-4000", 2001),
                ),
                side="sell",
                price=103,
                contracts=1,
                share_equivalent=2001,
            )
        second = ledger.record_future_execution(
            **execution_common("future-hedge-2", "p", 3, role="hedge"),
            initiating_execution_allocations=allocation,
            side="sell",
            price=103,
            contracts=1,
            share_equivalent=2000,
        )
        self.assertEqual(second.initiating_execution_id, "spot-4000")
        with self.assertRaisesRegex(AccountingError, "exceeds unconsumed"):
            ledger.record_future_execution(
                **execution_common("future-double-use", "p", 4, role="hedge"),
                initiating_execution_allocations=(
                    InitiatingExecutionAllocation("spot-4000", 1),
                ),
                side="sell",
                price=103,
                contracts=1,
                share_equivalent=1,
            )
        self.assertEqual(ledger.inventory("p"), (4000, -2, -4000))
        ledger.verify()

    def test_multi_source_partial_entry_rollback_replays(self) -> None:
        ledger = S1AccountingLedger()
        ledger.record_spot_execution(
            **execution_common("spot-600", "p", 1),
            side="buy",
            price=100,
            shares=600,
        )
        ledger.record_spot_execution(
            **execution_common("spot-1400", "p", 2),
            side="buy",
            price=101,
            shares=1400,
        )
        rollback = ledger.record_spot_execution(
            **execution_common("spot-rollback", "p", 3, role="rollback"),
            initiating_execution_allocations=(
                InitiatingExecutionAllocation("spot-600", 600),
                InitiatingExecutionAllocation("spot-1400", 1400),
            ),
            side="sell",
            price=99,
            shares=2000,
        )
        self.assertEqual(
            tuple((row.lot_id, row.share_equivalent) for row in rollback.allocations),
            (("spot-600:spot", 600), ("spot-1400:spot", 1400)),
        )
        self.assertEqual(ledger.inventory("p"), (0, 0, 0))
        ledger.seal_terminal(
            terminal_id="entry-partial-flat",
            position_id="p",
            terminal_date="20260825",
            cursor=cursor(4),
            terminal_outcome="entry_partial_rollback_flat",
            capacity_release_transition_id="release",
        )
        ledger.verify()

    def test_linked_execution_rejects_hedge_or_rollback_as_source(self) -> None:
        hedge_source = S1AccountingLedger()
        hedge_source.record_spot_execution(
            **execution_common("spot", "p", 1),
            side="buy",
            price=100,
            shares=2000,
        )
        hedge_source.record_future_execution(
            **execution_common(
                "future-hedge",
                "p",
                2,
                role="hedge",
                initiating_execution_id="spot",
            ),
            side="sell",
            price=103,
            contracts=1,
            share_equivalent=2000,
        )
        with self.assertRaisesRegex(AccountingError, "normal/forced first leg"):
            hedge_source.record_spot_execution(
                **execution_common(
                    "hedge-of-hedge",
                    "p",
                    3,
                    role="hedge",
                    initiating_execution_id="future-hedge",
                ),
                side="buy",
                price=100,
                shares=2000,
            )

        rollback_source = S1AccountingLedger()
        rollback_source.record_spot_execution(
            **execution_common("entry", "p", 1),
            side="buy",
            price=100,
            shares=1000,
        )
        rollback_source.record_spot_execution(
            **execution_common(
                "rollback",
                "p",
                2,
                role="rollback",
                initiating_execution_id="entry",
            ),
            side="sell",
            price=99,
            shares=1000,
        )
        with self.assertRaisesRegex(AccountingError, "normal/forced first leg"):
            rollback_source.record_spot_execution(
                **execution_common(
                    "rollback-of-rollback",
                    "p",
                    3,
                    role="rollback",
                    initiating_execution_id="rollback",
                ),
                side="buy",
                price=100,
                shares=1000,
            )

        forced_source = S1AccountingLedger()
        forced_source.record_spot_execution(
            **execution_common("forced-first-leg", "p", 1, role="forced"),
            side="buy",
            price=100,
            shares=1000,
        )
        forced_source.record_future_execution(
            **execution_common(
                "forced-second-leg",
                "p",
                2,
                role="hedge",
                initiating_execution_id="forced-first-leg",
            ),
            side="sell",
            price=103,
            contracts=1,
            share_equivalent=1000,
        )
        self.assertEqual(forced_source.inventory("p"), (1000, -1, -1000))

    def test_rollback_source_must_match_fifo_lot_actually_closed(self) -> None:
        ledger = S1AccountingLedger()
        ledger.record_spot_execution(
            **execution_common("old-entry", "p", 1),
            side="buy",
            price=100,
            shares=1000,
        )
        ledger.record_future_execution(
            **execution_common(
                "old-hedge",
                "p",
                2,
                role="hedge",
                initiating_execution_id="old-entry",
            ),
            side="sell",
            price=103,
            contracts=1,
            share_equivalent=1000,
        )
        ledger.record_spot_execution(
            **execution_common("new-entry", "p", 3),
            side="buy",
            price=200,
            shares=1000,
        )
        with self.assertRaisesRegex(AccountingError, "FIFO lot closure"):
            ledger.record_spot_execution(
                **execution_common("new-rollback", "p", 4, role="rollback"),
                initiating_execution_allocations=(
                    InitiatingExecutionAllocation("new-entry", 1000),
                ),
                side="sell",
                price=199,
                shares=1000,
            )
        self.assertEqual(ledger.inventory("p"), (2000, -1, -1000))

    def test_replay_rejects_coherent_source_identity_and_split_tamper(self) -> None:
        identity = S1AccountingLedger()
        identity.record_spot_execution(
            **execution_common("source-a", "p", 1),
            side="buy",
            price=100,
            shares=1000,
        )
        identity.record_spot_execution(
            **execution_common("source-b", "p", 2),
            side="buy",
            price=100,
            shares=1000,
        )
        hedge = identity.record_future_execution(
            **execution_common("hedge", "p", 3, role="hedge"),
            initiating_execution_allocations=(
                InitiatingExecutionAllocation("source-a", 1000),
            ),
            side="sell",
            price=103,
            contracts=1,
            share_equivalent=1000,
        )
        identity_tamper = replace(
            hedge,
            initiating_execution_id="source-b",
            initiating_execution_allocations=(
                InitiatingExecutionAllocation("source-b", 1000),
            ),
        )
        with self.assertRaisesRegex(AccountingReplayError, "canonical earliest"):
            replay_accounting_facts((*identity.rows[:2], identity_tamper))

        split = S1AccountingLedger()
        split.record_spot_execution(
            **execution_common("source-a", "p", 1),
            side="buy",
            price=100,
            shares=1000,
        )
        split.record_spot_execution(
            **execution_common("source-b", "p", 2),
            side="buy",
            price=100,
            shares=1000,
        )
        split_hedge = split.record_future_execution(
            **execution_common("hedge", "p", 3, role="hedge"),
            initiating_execution_allocations=(
                InitiatingExecutionAllocation("source-a", 1000),
                InitiatingExecutionAllocation("source-b", 500),
            ),
            side="sell",
            price=103,
            contracts=1,
            share_equivalent=1500,
        )
        split_tamper = replace(
            split_hedge,
            initiating_execution_allocations=(
                InitiatingExecutionAllocation("source-a", 900),
                InitiatingExecutionAllocation("source-b", 600),
            ),
        )
        with self.assertRaisesRegex(AccountingReplayError, "canonical earliest"):
            replay_accounting_facts((*split.rows[:2], split_tamper))

    def test_request_link_is_nullable_only_when_role_does_not_require_it(self) -> None:
        normal = S1AccountingLedger()
        normal_common = execution_common("entry", "p", 1)
        normal_common["request_id"] = None
        recorded = normal.record_spot_execution(
            **normal_common,
            side="buy",
            price=100,
            shares=1000,
        )
        self.assertIsNone(recorded.request_id)

        rollback = S1AccountingLedger()
        rollback.record_spot_execution(
            **execution_common("entry", "p", 1),
            side="buy",
            price=100,
            shares=1000,
        )
        rollback_common = execution_common(
            "rollback",
            "p",
            2,
            role="rollback",
            initiating_execution_id="entry",
        )
        rollback_common["request_id"] = None
        with self.assertRaisesRegex(AccountingError, "request_id"):
            rollback.record_spot_execution(
                **rollback_common,
                side="sell",
                price=99,
                shares=1000,
            )

    def test_exit_rollback_opens_new_acquisition_lots(self) -> None:
        ledger = S1AccountingLedger()
        self._open_pair(ledger, execution_date="20260824")
        ledger.record_spot_execution(
            **execution_common("spot-exit", "p", 4),
            side="sell",
            price=101,
            shares=2000,
        )
        spot_rollback = ledger.record_spot_execution(
            **execution_common(
                "spot-rollback",
                "p",
                5,
                role="rollback",
                initiating_execution_id="spot-exit",
            ),
            side="buy",
            price=102,
            shares=2000,
        )
        ledger.record_future_execution(
            **execution_common("future-exit", "p", 6),
            side="buy",
            price=101,
            contracts=1,
            share_equivalent=2000,
        )
        future_rollback = ledger.record_future_execution(
            **execution_common(
                "future-rollback",
                "p",
                7,
                role="rollback",
                initiating_execution_id="future-exit",
            ),
            side="sell",
            price=102,
            contracts=1,
            share_equivalent=2000,
        )
        self.assertEqual(spot_rollback.opened_acquisition_date, "20260825")
        self.assertEqual(future_rollback.opened_acquisition_date, "20260825")
        self.assertEqual(ledger.inventory("p"), (2000, -1, -2000))

    def test_integer_quantity_and_integral_contract_size_are_required(self) -> None:
        ledger = S1AccountingLedger()
        with self.assertRaises(AccountingError):
            ledger.record_spot_execution(
                **execution_common("spot", "p", 1),
                side="buy",
                price=100,
                shares=1.0,
            )
        with self.assertRaises(AccountingError):
            ledger.record_future_execution(
                **execution_common("future", "p", 1),
                side="sell",
                price=100,
                contracts=2,
                share_equivalent=2001,
            )

    def test_dates_are_valid_monotone_and_may_bind_to_cursor_calendar(self) -> None:
        ledger = S1AccountingLedger()
        with self.assertRaises(AccountingError):
            ledger.record_spot_execution(
                **execution_common("invalid", "p", 1, execution_date="20261399"),
                side="buy",
                price=100,
                shares=1,
            )
        ledger.record_spot_execution(
            **execution_common("first", "p", 1),
            side="buy",
            price=100,
            shares=1,
        )
        with self.assertRaisesRegex(AccountingError, "date is out of order"):
            ledger.record_spot_execution(
                **execution_common("backward", "p", 2, execution_date="20260824"),
                side="buy",
                price=100,
                shares=1,
            )
        resolved = S1AccountingLedger(cursor_date_resolver=lambda _: "20260826")
        with self.assertRaisesRegex(AccountingError, "calendar resolver"):
            resolved.record_spot_execution(
                **execution_common("mismatch", "p", 1),
                side="buy",
                price=100,
                shares=1,
            )

    def test_full_event_cursor_must_be_strictly_increasing(self) -> None:
        ledger = S1AccountingLedger()
        ledger.record_spot_execution(
            **execution_common("first", "p", 1),
            side="buy",
            price=100,
            shares=1,
        )
        with self.assertRaisesRegex(AccountingError, "strictly increasing"):
            ledger.record_spot_execution(
                **execution_common("duplicate-cursor", "p", 1),
                side="buy",
                price=100,
                shares=1,
            )

    def test_terminal_requires_existing_exactly_flat_position_and_seals_it(
        self,
    ) -> None:
        ledger = S1AccountingLedger()
        with self.assertRaisesRegex(AccountingError, "unknown position"):
            ledger.terminal_realized("missing")
        with self.assertRaisesRegex(AccountingError, "unknown position"):
            ledger.seal_terminal(
                terminal_id="missing",
                position_id="missing",
                terminal_date="20260825",
                cursor=cursor(1),
                terminal_outcome="exit_maker_flat",
                capacity_release_transition_id="release",
            )
        ledger.record_spot_execution(
            **execution_common("entry", "p", 1),
            side="buy",
            price=100,
            shares=1,
        )
        with self.assertRaises(InventoryNotFlatError):
            ledger.seal_terminal(
                terminal_id="nonflat",
                position_id="p",
                terminal_date="20260825",
                cursor=cursor(2),
                terminal_outcome="exit_maker_flat",
                capacity_release_transition_id="release",
            )
        ledger.record_spot_execution(
            **execution_common(
                "rollback",
                "p",
                2,
                role="rollback",
                initiating_execution_id="entry",
            ),
            side="sell",
            price=99,
            shares=1,
        )
        terminal = ledger.seal_terminal(
            terminal_id="flat",
            position_id="p",
            terminal_date="20260825",
            cursor=cursor(3),
            terminal_outcome="entry_emergency_rollback_flat",
            capacity_release_transition_id="release",
        )
        self.assertIs(ledger.terminal_realized("p"), terminal)
        with self.assertRaisesRegex(AccountingError, "sealed position"):
            ledger.record_spot_execution(
                **execution_common("reopen", "p", 4),
                side="buy",
                price=100,
                shares=1,
            )

    def test_expiry_mark_is_basis_zero_non_executable_and_cost_free(self) -> None:
        ledger = S1AccountingLedger()
        self._open_pair(ledger, truth="approximate")
        source_cursor = EventCursor(4, 0, 0)
        mark = ledger.record_expiry_accounting_mark(
            mark_id="expiry-p",
            position_id="p",
            expiry_date="20260826",
            cursor=cursor(4),
            spot_close_source_id="spot-close/20260826/2330/close",
            spot_close_source_cursor=source_cursor,
            spot_close_price=101,
            capacity_release_transition_id="expiry-release-p",
        )
        self.assertIsInstance(mark, ExpiryAccountingMark)
        self.assertTrue(mark.non_executable)
        self.assertIsNone(mark.request_id)
        self.assertIsNone(mark.execution_id)
        self.assertAlmostEqual(mark.synthetic_commission_twd, 0)
        self.assertAlmostEqual(mark.synthetic_tax_twd, 0)
        self.assertAlmostEqual(mark.spot_cashflow_twd, 2000)
        self.assertAlmostEqual(mark.futures_realized_pnl_twd, 4000)
        self.assertAlmostEqual(
            mark.realized_net_twd,
            6000 - mark.actual_commission_twd - mark.actual_tax_twd,
        )
        self.assertEqual(mark.execution_truth, "approximate")
        self.assertEqual(
            mark.spot_close_source_id,
            "spot-close/20260826/2330/close",
        )
        self.assertEqual(mark.spot_close_source_cursor, source_cursor)
        digest = mark.as_dict()
        self.assertEqual(
            digest["spot_close_source_id"],
            "spot-close/20260826/2330/close",
        )
        self.assertEqual(digest["spot_close_source_recv_time_ns"], 4)
        self.assertEqual(digest["spot_close_source_event_sequence"], 0)
        self.assertEqual(digest["spot_close_source_row_index"], 0)
        self.assertNotIn("spot_close_source_cursor", digest)
        self.assertEqual(ledger.inventory("p"), (0, 0, 0))
        self.assertIs(ledger.terminal_realized("p"), mark)
        replayed = ledger.verify()
        self.assertEqual(
            replayed.expiry_marks[0].spot_close_source_cursor,
            source_cursor,
        )

        with self.assertRaisesRegex(AccountingReplayError, "source cursor"):
            replay_accounting_facts(
                (
                    *ledger.facts[:-1],
                    replace(
                        mark,
                        spot_close_source_cursor=EventCursor(5, 0, 0),
                    ),
                )
            )
        with self.assertRaisesRegex(AccountingReplayError, "canonical"):
            replay_accounting_facts(
                (
                    *ledger.facts[:-1],
                    replace(mark, spot_close_source_id=" noncanonical "),
                )
            )

    def test_expiry_spot_close_source_must_share_resolved_mark_date(self) -> None:
        def resolve_date(value: EventCursor) -> str:
            return "20260825" if value.recv_time_ns < 10 else "20260826"

        ledger = S1AccountingLedger(cursor_date_resolver=resolve_date)
        self._open_pair(ledger)
        mark_cursor = EventCursor(10, 1, 0)
        with self.assertRaisesRegex(AccountingError, "source date disagrees"):
            ledger.record_expiry_accounting_mark(
                mark_id="expiry-p",
                position_id="p",
                expiry_date="20260826",
                cursor=mark_cursor,
                spot_close_source_id="spot-close/20260826/2330/close",
                spot_close_source_cursor=EventCursor(9, 2, 0),
                spot_close_price=101,
                capacity_release_transition_id="expiry-release-p",
            )

        mark = ledger.record_expiry_accounting_mark(
            mark_id="expiry-p",
            position_id="p",
            expiry_date="20260826",
            cursor=mark_cursor,
            spot_close_source_id="spot-close/20260826/2330/close",
            spot_close_source_cursor=EventCursor(10, 0, 0),
            spot_close_price=101,
            capacity_release_transition_id="expiry-release-p",
        )
        self.assertEqual(mark.spot_close_source_cursor, EventCursor(10, 0, 0))
        ledger.verify()

    def test_expiry_mark_rejects_naked_or_mismatched_inventory(self) -> None:
        naked = S1AccountingLedger()
        naked.record_spot_execution(
            **execution_common("spot", "p", 1),
            side="buy",
            price=100,
            shares=1000,
        )
        with self.assertRaises(InventoryNotFlatError):
            naked.record_expiry_accounting_mark(
                mark_id="naked",
                position_id="p",
                expiry_date="20260826",
                cursor=cursor(2),
                spot_close_source_id="spot-close/naked",
                spot_close_source_cursor=EventCursor(2, 0, 0),
                spot_close_price=101,
                capacity_release_transition_id="release",
            )
        mismatched = S1AccountingLedger()
        mismatched.record_spot_execution(
            **execution_common("spot", "p", 1),
            side="buy",
            price=100,
            shares=1000,
        )
        mismatched.record_future_execution(
            **execution_common("future", "p", 2),
            side="sell",
            price=103,
            contracts=1,
            share_equivalent=2000,
        )
        with self.assertRaises(InventoryNotFlatError):
            mismatched.record_expiry_accounting_mark(
                mark_id="mismatch",
                position_id="p",
                expiry_date="20260826",
                cursor=cursor(3),
                spot_close_source_id="spot-close/mismatch",
                spot_close_source_cursor=cursor(3),
                spot_close_price=101,
                capacity_release_transition_id="release",
            )
        long_future = S1AccountingLedger()
        long_future.record_spot_execution(
            **execution_common("spot", "p", 1),
            side="buy",
            price=100,
            shares=1000,
        )
        long_future.record_future_execution(
            **execution_common("future", "p", 2),
            side="buy",
            price=103,
            contracts=1,
            share_equivalent=1000,
        )
        with self.assertRaises(InventoryNotFlatError):
            long_future.record_expiry_accounting_mark(
                mark_id="long-future",
                position_id="p",
                expiry_date="20260826",
                cursor=cursor(3),
                spot_close_source_id="spot-close/long-future",
                spot_close_source_cursor=cursor(3),
                spot_close_price=101,
                capacity_release_transition_id="release",
            )

    def test_replay_rejects_derived_terminal_and_sequence_tamper(self) -> None:
        ledger = S1AccountingLedger()
        ledger.record_spot_execution(
            **execution_common("entry", "p", 1),
            side="buy",
            price=100,
            shares=100,
        )
        ledger.record_spot_execution(
            **execution_common(
                "rollback",
                "p",
                2,
                role="rollback",
                initiating_execution_id="entry",
            ),
            side="sell",
            price=99,
            shares=100,
        )
        ledger.seal_terminal(
            terminal_id="terminal",
            position_id="p",
            terminal_date="20260825",
            cursor=cursor(3),
            terminal_outcome="entry_emergency_rollback_flat",
            capacity_release_transition_id="release",
        )
        leg = ledger.rows[0]
        rollback = ledger.rows[1]
        terminal = ledger.terminal_rows[0]
        tampered_fact_sets = (
            (replace(leg, commission_twd=leg.commission_twd + 1),),
            (
                *ledger.rows,
                replace(
                    terminal,
                    realized_net_twd=terminal.realized_net_twd + 1,
                ),
            ),
            (
                leg,
                replace(rollback, initiating_execution_id="not-the-entry"),
            ),
            (replace(leg, sequence=2),),
        )
        for facts in tampered_fact_sets:
            with self.assertRaises(AccountingReplayError):
                replay_accounting_facts(facts)


if __name__ == "__main__":
    unittest.main()
