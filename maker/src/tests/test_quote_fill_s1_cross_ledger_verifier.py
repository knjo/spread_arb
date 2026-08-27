from __future__ import annotations

import unittest
from dataclasses import replace

from maker.src.quote_fill.capacity_ledger import CapacityLedger
from maker.src.quote_fill.layered import EventCursor
from maker.src.quote_fill.s1_accounting import InitiatingExecutionAllocation
from maker.src.quote_fill.s1_accounting_bridge import (
    S1AccountingBridge,
    S1AccountingProduct,
)
from maker.src.quote_fill.s1_cross_ledger_verifier import (
    S1CrossLedgerError,
    verify_s1_accounting_capacity_links,
)
from maker.src.tests.test_quote_fill_s1_accounting_bridge import execution


def complete_portfolio():  # type: ignore[no-untyped-def]
    capacity = CapacityLedger(global_cap_twd=1_000, product_cap_twd=1_000)
    capacity.attempt_new_reservation(
        transition_id="reserve",
        timestamp_ns=1,
        capacity_id="capacity-1",
        product_id="1101",
        requested_notional_twd=800,
    )
    capacity.record_entry_full_fill(
        transition_id="fill",
        timestamp_ns=2,
        capacity_id="capacity-1",
    )
    capacity.complete_entry_hedge(
        transition_id="paired",
        timestamp_ns=3,
        event_sequence=0,
        row_index=0,
        capacity_id="capacity-1",
        notional_twd=800,
    )

    bridge = S1AccountingBridge(
        default_date="20260825",
        scenario_id="q95",
        products=(S1AccountingProduct("product-1", "1101", 2000),),
    )
    bridge.record_execution(
        execution(
            "entry-maker",
            1,
            role="entry_maker",
            market="spot",
            side="buy",
            quantity=2000,
            unit="spot_shares",
        )
    )
    entry_hedge = execution(
        "entry-hedge",
        3,
        role="entry_hedge",
        market="future",
        side="sell",
        quantity=1,
        unit="future_contracts",
        request_id="entry-risk",
        sources=(InitiatingExecutionAllocation("entry-maker", 2000),),
    )
    entry_hedge = replace(entry_hedge, cursor=EventCursor(3, 1, 0))
    bridge.record_execution(entry_hedge)
    bridge.establish_position(
        establishment_id="established",
        position_id="position-1",
        product_id="product-1",
        capacity_id="capacity-1",
        cursor=EventCursor(3, 2, 0),
        capacity_transition_id="paired",
        execution_truth="exact",
    )
    bridge.record_execution(
        replace(
            execution(
                "exit-maker",
                4,
                role="exit_maker",
                market="spot",
                side="sell",
                quantity=2000,
                unit="spot_shares",
            ),
            cursor=EventCursor(4, 1, 0),
        )
    )
    capacity.begin_exit(
        transition_id="begin-exit",
        timestamp_ns=4,
        event_sequence=2,
        row_index=0,
        capacity_id="capacity-1",
        notional_twd=800,
    )
    capacity.complete_exit_hedge(
        transition_id="release",
        timestamp_ns=5,
        event_sequence=0,
        row_index=0,
        capacity_id="capacity-1",
        notional_twd=800,
    )
    bridge.record_execution(
        replace(
            execution(
                "exit-hedge",
                5,
                role="exit_hedge",
                market="future",
                side="buy",
                quantity=1,
                unit="future_contracts",
                request_id="exit-risk",
                sources=(InitiatingExecutionAllocation("exit-maker", 2000),),
            ),
            cursor=EventCursor(5, 1, 0),
        )
    )
    bridge.seal_terminal(
        terminal_id="terminal",
        position_id="position-1",
        cursor=EventCursor(5, 2, 0),
        terminal_outcome="exit_maker_flat",
        capacity_release_transition_id="release",
    )
    return bridge, capacity


class S1CrossLedgerVerifierTest(unittest.TestCase):
    def test_complete_lifecycle_links_exactly_once(self) -> None:
        bridge, capacity = complete_portfolio()
        report = verify_s1_accounting_capacity_links(
            bridge.facts,
            capacity.transitions,
        )
        self.assertEqual(report.establishments, 1)
        self.assertEqual(report.executable_terminals, 1)
        self.assertEqual(report.linked_capacity_transitions, 2)

    def test_unknown_wrong_type_and_capacity_identity_are_rejected(self) -> None:
        bridge, capacity = complete_portfolio()
        facts = list(bridge.facts)
        facts[-1] = replace(facts[-1], capacity_release_transition_id="missing")
        with self.assertRaisesRegex(S1CrossLedgerError, "unknown"):
            verify_s1_accounting_capacity_links(facts, capacity.transitions)

        facts = list(bridge.facts)
        facts[-1] = replace(
            facts[-1],
            capacity_release_transition_id="begin-exit",
        )
        with self.assertRaisesRegex(S1CrossLedgerError, "wrong capacity event"):
            verify_s1_accounting_capacity_links(facts, capacity.transitions)

        transitions = list(capacity.transitions)
        release_index = next(
            index
            for index, row in enumerate(transitions)
            if row.transition_id == "release"
        )
        transitions[release_index] = replace(
            transitions[release_index],
            capacity_id="another",
        )
        with self.assertRaisesRegex(S1CrossLedgerError, "capacity_id"):
            verify_s1_accounting_capacity_links(bridge.facts, transitions)

    def test_orphaned_release_and_noncausal_settlement_are_rejected(self) -> None:
        bridge, capacity = complete_portfolio()
        with self.assertRaisesRegex(S1CrossLedgerError, "one-to-one"):
            verify_s1_accounting_capacity_links(
                bridge.facts[:-1],
                capacity.transitions,
            )

        facts = list(bridge.facts)
        facts[-1] = replace(facts[-1], cursor=EventCursor(5, 0, 0))
        with self.assertRaisesRegex(S1CrossLedgerError, "causally precede"):
            verify_s1_accounting_capacity_links(facts, capacity.transitions)

    def test_invalid_fact_object_is_rejected_before_joining(self) -> None:
        _bridge, capacity = complete_portfolio()
        with self.assertRaisesRegex(TypeError, "accounting_facts"):
            verify_s1_accounting_capacity_links(  # type: ignore[arg-type]
                [object()],
                capacity.transitions,
            )


if __name__ == "__main__":
    unittest.main()
