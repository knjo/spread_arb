from __future__ import annotations

import json
import unittest
from dataclasses import dataclass, replace

from maker.src.quote_fill.layered import EventCursor
from maker.src.quote_fill.s1_accounting import (
    AccountingReplayError,
    InitiatingExecutionAllocation,
    decode_accounting_facts,
    encode_accounting_facts,
    replay_accounting_facts,
)
from maker.src.quote_fill.s1_accounting_bridge import (
    S1AccountingBridge,
    S1AccountingProduct,
)
from maker.src.quote_fill.transaction_costs import TransactionCostProfile


@dataclass(frozen=True)
class FakeExecution:
    execution_id: str
    position_id: str
    product_id: str
    capacity_id: str
    request_id: str | None
    role: str
    market: str
    side: str
    cursor: EventCursor
    price: float
    quantity: int
    quantity_unit: str
    execution_truth: str
    execution_source_id: str
    initiating_execution_allocations: tuple[InitiatingExecutionAllocation, ...] = ()
    hedge_intent_id: str | None = None


def execution(
    execution_id: str,
    sequence: int,
    *,
    role: str,
    market: str,
    side: str,
    quantity: int,
    unit: str,
    request_id: str | None = None,
    sources: tuple[InitiatingExecutionAllocation, ...] = (),
    position_id: str = "position-1",
    product_id: str = "product-1",
    capacity_id: str = "capacity-1",
) -> FakeExecution:
    return FakeExecution(
        execution_id=execution_id,
        position_id=position_id,
        product_id=product_id,
        capacity_id=capacity_id,
        request_id=request_id,
        role=role,
        market=market,
        side=side,
        cursor=EventCursor(sequence, 0, 0),
        price=100.0 + sequence,
        quantity=quantity,
        quantity_unit=unit,
        execution_truth="exact",
        execution_source_id=f"source-{sequence}",
        initiating_execution_allocations=sources,
        hedge_intent_id=request_id,
    )


class S1AccountingBridgeTest(unittest.TestCase):
    def bridge(self) -> S1AccountingBridge:
        return S1AccountingBridge(
            default_date="20260825",
            scenario_id="q95",
            products=(S1AccountingProduct("product-1", "1101", 2000),),
        )

    def test_fact_checkpoint_and_suffix_do_not_require_full_tuple_copy(self) -> None:
        bridge = self.bridge()
        self.assertEqual(bridge.fact_count, 0)
        self.assertEqual(bridge.facts_since(0), ())
        checkpoint = bridge.fact_count
        first = bridge.record_execution(
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
        self.assertEqual(bridge.fact_count, 1)
        self.assertEqual(bridge.facts_since(checkpoint), (first,))
        self.assertEqual(bridge.facts_since(bridge.fact_count), ())
        with self.assertRaises(TypeError):
            bridge.facts_since(True)
        with self.assertRaises(IndexError):
            bridge.facts_since(2)

    def test_full_entry_exit_lifecycle_is_replayable(self) -> None:
        bridge = self.bridge()
        entry = execution(
            "entry-maker",
            1,
            role="entry_maker",
            market="spot",
            side="buy",
            quantity=2000,
            unit="spot_shares",
        )
        bridge.record_execution(entry)
        bridge.record_execution(
            execution(
                "entry-hedge",
                2,
                role="entry_hedge",
                market="future",
                side="sell",
                quantity=1,
                unit="future_contracts",
                request_id="entry-risk",
                sources=(InitiatingExecutionAllocation("entry-maker", 2000),),
            )
        )
        established = bridge.establish_position(
            establishment_id="established-1",
            position_id="position-1",
            product_id="product-1",
            capacity_id="capacity-1",
            cursor=EventCursor(3, 0, 0),
            capacity_transition_id="entry-hedge-capacity",
            execution_truth="exact",
        )
        self.assertEqual(established.fifo_key, (3, "position-1"))

        bridge.record_execution(
            execution(
                "exit-maker-a",
                4,
                role="exit_maker",
                market="spot",
                side="sell",
                quantity=600,
                unit="spot_shares",
            )
        )
        bridge.record_execution(
            execution(
                "exit-maker-b",
                5,
                role="exit_maker",
                market="spot",
                side="sell",
                quantity=1400,
                unit="spot_shares",
            )
        )
        bridge.record_execution(
            execution(
                "exit-hedge",
                6,
                role="exit_hedge",
                market="future",
                side="buy",
                quantity=1,
                unit="future_contracts",
                request_id="exit-risk",
                sources=(
                    InitiatingExecutionAllocation("exit-maker-a", 600),
                    InitiatingExecutionAllocation("exit-maker-b", 1400),
                ),
            )
        )
        terminal = bridge.seal_terminal(
            terminal_id="terminal-1",
            position_id="position-1",
            cursor=EventCursor(7, 0, 0),
            terminal_outcome="exit_maker_flat",
            capacity_release_transition_id="exit-release",
        )

        self.assertEqual(terminal.executed_leg_count, 5)
        self.assertEqual(bridge.ledger.inventory("position-1"), (0, 0, 0))
        self.assertIs(bridge.verify(), bridge)

    def test_entry_rollback_seals_without_establishment(self) -> None:
        bridge = self.bridge()
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
        bridge.record_execution(
            execution(
                "entry-rollback",
                2,
                role="entry_rollback",
                market="spot",
                side="sell",
                quantity=2000,
                unit="spot_shares",
                request_id="rollback-risk",
                sources=(InitiatingExecutionAllocation("entry-maker", 2000),),
            )
        )
        terminal = bridge.seal_terminal(
            terminal_id="rollback-terminal",
            position_id="position-1",
            cursor=EventCursor(3, 0, 0),
            terminal_outcome="entry_emergency_rollback_flat",
            capacity_release_transition_id="rollback-release",
        )
        self.assertEqual(terminal.executed_leg_count, 2)
        bridge.verify()

    def test_expiry_mark_forwards_exact_spot_close_provenance(self) -> None:
        bridge = self.bridge()
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
        bridge.record_execution(
            execution(
                "entry-hedge",
                2,
                role="entry_hedge",
                market="future",
                side="sell",
                quantity=1,
                unit="future_contracts",
                request_id="entry-risk",
                sources=(InitiatingExecutionAllocation("entry-maker", 2000),),
            )
        )
        bridge.establish_position(
            establishment_id="established-1",
            position_id="position-1",
            product_id="product-1",
            capacity_id="capacity-1",
            cursor=EventCursor(3, 0, 0),
            capacity_transition_id="entry-hedge-capacity",
            execution_truth="exact",
        )
        source_cursor = EventCursor(4, 0, 0)
        mark = bridge.record_expiry_mark(
            mark_id="expiry-1",
            position_id="position-1",
            cursor=EventCursor(4, 1, 0),
            capacity_release_transition_id="expiry-release",
            spot_close_source_id="spot-close/20260825/1101/close",
            spot_close_source_cursor=source_cursor,
            spot_close_price=101.0,
        )

        self.assertEqual(
            mark.spot_close_source_id,
            "spot-close/20260825/1101/close",
        )
        self.assertEqual(mark.spot_close_source_cursor, source_cursor)
        self.assertEqual(bridge.ledger.inventory("position-1"), (0, 0, 0))
        self.assertIs(bridge.verify(), bridge)

    def test_linked_execution_requires_exact_source_allocations(self) -> None:
        bridge = self.bridge()
        with self.assertRaisesRegex(ValueError, "source allocations"):
            bridge.record_execution(
                execution(
                    "bad-hedge",
                    1,
                    role="entry_hedge",
                    market="future",
                    side="sell",
                    quantity=1,
                    unit="future_contracts",
                    request_id="risk",
                )
            )

    def test_unknown_product_and_duplicate_execution_are_rejected(self) -> None:
        bridge = self.bridge()
        row = execution(
            "entry-maker",
            1,
            role="entry_maker",
            market="spot",
            side="buy",
            quantity=2000,
            unit="spot_shares",
        )
        bridge.record_execution(row)
        with self.assertRaisesRegex(ValueError, "already recorded"):
            bridge.record_execution(row)
        unknown = FakeExecution(
            **{**row.__dict__, "execution_id": "unknown", "product_id": "missing"}
        )
        with self.assertRaisesRegex(ValueError, "unknown product"):
            bridge.record_execution(unknown)

    def test_position_product_and_capacity_binding_cannot_drift(self) -> None:
        bridge = S1AccountingBridge(
            default_date="20260825",
            scenario_id="q95",
            products=(
                S1AccountingProduct("product-1", "1101", 2000),
                S1AccountingProduct("product-2", "1101", 2000),
            ),
        )
        first = execution(
            "entry-maker",
            1,
            role="entry_maker",
            market="spot",
            side="buy",
            quantity=2000,
            unit="spot_shares",
        )
        bridge.record_execution(first)
        drifted = FakeExecution(
            **{
                **first.__dict__,
                "execution_id": "drifted",
                "product_id": "product-2",
            }
        )
        with self.assertRaisesRegex(ValueError, "binding changed"):
            bridge.record_execution(drifted)
        with self.assertRaisesRegex(ValueError, "establishment disagrees"):
            bridge.establish_position(
                establishment_id="bad-establishment",
                position_id="position-1",
                product_id="product-1",
                capacity_id="wrong-capacity",
                cursor=EventCursor(2, 0, 0),
                capacity_transition_id="transition",
                execution_truth="exact",
            )

    def test_establishment_truth_mismatch_is_rejected_before_commit(self) -> None:
        bridge = self.bridge()
        approximate = execution(
            "entry-maker",
            1,
            role="entry_maker",
            market="spot",
            side="buy",
            quantity=2000,
            unit="spot_shares",
        )
        approximate = FakeExecution(
            **{**approximate.__dict__, "execution_truth": "approximate"}
        )
        bridge.record_execution(approximate)
        bridge.record_execution(
            FakeExecution(
                **{
                    **execution(
                        "entry-hedge",
                        2,
                        role="entry_hedge",
                        market="future",
                        side="sell",
                        quantity=1,
                        unit="future_contracts",
                        request_id="risk",
                        sources=(InitiatingExecutionAllocation("entry-maker", 2000),),
                    ).__dict__,
                    "execution_truth": "exact",
                }
            )
        )
        with self.assertRaisesRegex(ValueError, "truth disagrees"):
            bridge.establish_position(
                establishment_id="bad-truth",
                position_id="position-1",
                product_id="product-1",
                capacity_id="capacity-1",
                cursor=EventCursor(3, 0, 0),
                capacity_transition_id="transition",
                execution_truth="exact",
            )
        self.assertEqual(bridge.ledger.establishment_rows, ())

    def test_terminal_outcome_cannot_relabel_normal_exit_as_entry_rollback(
        self,
    ) -> None:
        bridge = self.bridge()
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
        bridge.record_execution(
            execution(
                "entry-hedge",
                2,
                role="entry_hedge",
                market="future",
                side="sell",
                quantity=1,
                unit="future_contracts",
                request_id="entry-risk",
                sources=(InitiatingExecutionAllocation("entry-maker", 2000),),
            )
        )
        bridge.establish_position(
            establishment_id="established",
            position_id="position-1",
            product_id="product-1",
            capacity_id="capacity-1",
            cursor=EventCursor(3, 0, 0),
            capacity_transition_id="paired",
            execution_truth="exact",
        )
        bridge.record_execution(
            execution(
                "exit-maker",
                4,
                role="exit_maker",
                market="spot",
                side="sell",
                quantity=2000,
                unit="spot_shares",
            )
        )
        bridge.record_execution(
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
            )
        )
        with self.assertRaisesRegex(ValueError, "cannot follow establishment"):
            bridge.seal_terminal(
                terminal_id="mislabelled",
                position_id="position-1",
                cursor=EventCursor(6, 0, 0),
                terminal_outcome="entry_emergency_rollback_flat",
                capacity_release_transition_id="release",
            )
        self.assertEqual(bridge.ledger.terminal_rows, ())

    def test_route_role_terminal_gate_survives_serialized_replay(self) -> None:
        bridge = self.bridge()
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
        bridge.record_execution(
            execution(
                "entry-hedge",
                2,
                role="entry_hedge",
                market="future",
                side="sell",
                quantity=1,
                unit="future_contracts",
                request_id="entry-risk",
                sources=(InitiatingExecutionAllocation("entry-maker", 2000),),
            )
        )
        bridge.establish_position(
            establishment_id="established",
            position_id="position-1",
            product_id="product-1",
            capacity_id="capacity-1",
            cursor=EventCursor(3, 0, 0),
            capacity_transition_id="paired",
            execution_truth="exact",
        )
        bridge.record_execution(
            execution(
                "exit-maker",
                4,
                role="exit_maker",
                market="spot",
                side="sell",
                quantity=2000,
                unit="spot_shares",
            )
        )
        bridge.record_execution(
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
            )
        )
        bridge.seal_terminal(
            terminal_id="terminal",
            position_id="position-1",
            cursor=EventCursor(6, 0, 0),
            terminal_outcome="exit_maker_flat",
            capacity_release_transition_id="release",
        )
        facts = list(bridge.facts)
        facts[4] = replace(facts[4], route_role="entry_hedge")
        with self.assertRaisesRegex(AccountingReplayError, "route roles"):
            replay_accounting_facts(facts, require_route_roles=True)

        stripped = tuple(
            replace(fact, route_role=None) if hasattr(fact, "route_role") else fact
            for fact in bridge.facts
        )
        with self.assertRaisesRegex(AccountingReplayError, "requires route_role"):
            replay_accounting_facts(stripped, require_route_roles=True)

    def test_resume_open_day_then_cross_day_exit_matches_uninterrupted_facts(
        self,
    ) -> None:
        def date_for_cursor(cursor: EventCursor) -> str:
            return "20260825" if cursor.recv_time_ns < 100 else "20260826"

        products = (S1AccountingProduct("product-1", "1101", 2000),)
        uninterrupted = S1AccountingBridge(
            default_date="20260825",
            scenario_id="q95",
            products=products,
            execution_date_resolver=date_for_cursor,
        )
        for offset, position_id, capacity_id in (
            (0, "position-1", "capacity-1"),
            (30, "position-2", "capacity-2"),
        ):
            maker_id = f"{position_id}/entry-maker"
            uninterrupted.record_execution(
                execution(
                    maker_id,
                    10 + offset,
                    role="entry_maker",
                    market="spot",
                    side="buy",
                    quantity=2000,
                    unit="spot_shares",
                    position_id=position_id,
                    capacity_id=capacity_id,
                )
            )
            uninterrupted.record_execution(
                execution(
                    f"{position_id}/entry-hedge",
                    20 + offset,
                    role="entry_hedge",
                    market="future",
                    side="sell",
                    quantity=1,
                    unit="future_contracts",
                    request_id=f"{position_id}/entry-risk",
                    sources=(InitiatingExecutionAllocation(maker_id, 2000),),
                    position_id=position_id,
                    capacity_id=capacity_id,
                )
            )
            uninterrupted.establish_position(
                establishment_id=f"{position_id}/established",
                position_id=position_id,
                product_id="product-1",
                capacity_id=capacity_id,
                cursor=EventCursor(30 + offset, 0, 0),
                capacity_transition_id=f"{position_id}/paired",
                execution_truth="exact",
            )

        day_one_facts = decode_accounting_facts(
            json.loads(json.dumps(encode_accounting_facts(uninterrupted.facts)))
        )
        resumed = S1AccountingBridge.from_facts(
            default_date="20260825",
            scenario_id="q95",
            products=products,
            facts=day_one_facts,
            execution_date_resolver=date_for_cursor,
        )
        self.assertEqual(resumed.facts, uninterrupted.facts)
        day_one_fact_count = len(day_one_facts)
        self.assertEqual(resumed.fact_count, day_one_fact_count)

        for bridge in (uninterrupted, resumed):
            bridge.record_execution(
                execution(
                    "position-1/exit-maker",
                    110,
                    role="exit_maker",
                    market="spot",
                    side="sell",
                    quantity=2000,
                    unit="spot_shares",
                )
            )
            bridge.record_execution(
                execution(
                    "position-1/exit-hedge",
                    120,
                    role="exit_hedge",
                    market="future",
                    side="buy",
                    quantity=1,
                    unit="future_contracts",
                    request_id="position-1/exit-risk",
                    sources=(
                        InitiatingExecutionAllocation(
                            "position-1/exit-maker",
                            2000,
                        ),
                    ),
                )
            )
            bridge.seal_terminal(
                terminal_id="position-1/terminal",
                position_id="position-1",
                cursor=EventCursor(130, 0, 0),
                terminal_outcome="exit_maker_flat",
                capacity_release_transition_id="position-1/released",
            )

        self.assertEqual(resumed.facts, uninterrupted.facts)
        self.assertEqual(
            resumed.facts_since(day_one_fact_count),
            uninterrupted.facts[day_one_fact_count:],
        )
        self.assertEqual(
            tuple(fact.sequence for fact in resumed.facts),
            tuple(range(1, len(resumed.facts) + 1)),
        )
        exit_row = next(
            row
            for row in resumed.ledger.rows
            if row.execution_id == "position-1/exit-maker"
        )
        self.assertEqual(exit_row.execution_date, "20260826")
        self.assertFalse(exit_row.allocations[0].same_day)
        self.assertAlmostEqual(
            exit_row.tax_twd,
            TransactionCostProfile().spot_sell_tax_twd(
                exit_row.price,
                2000,
                same_day=False,
            ),
        )

        terminal_and_open = S1AccountingBridge.from_facts(
            default_date="20260825",
            scenario_id="q95",
            products=products,
            facts=decode_accounting_facts(
                json.loads(json.dumps(encode_accounting_facts(resumed.facts)))
            ),
            execution_date_resolver=date_for_cursor,
        )
        self.assertEqual(terminal_and_open.facts, uninterrupted.facts)
        self.assertEqual(terminal_and_open.ledger.inventory("position-1"), (0, 0, 0))
        self.assertEqual(
            terminal_and_open.ledger.inventory("position-2"),
            (2000, -1, -2000),
        )
        with self.assertRaisesRegex(ValueError, "already recorded"):
            terminal_and_open.record_execution(
                execution(
                    "position-1/entry-maker",
                    140,
                    role="entry_maker",
                    market="spot",
                    side="buy",
                    quantity=2000,
                    unit="spot_shares",
                )
            )
        for bridge in (uninterrupted, terminal_and_open):
            bridge.record_execution(
                execution(
                    "position-2/exit-maker",
                    210,
                    role="exit_maker",
                    market="spot",
                    side="sell",
                    quantity=2000,
                    unit="spot_shares",
                    position_id="position-2",
                    capacity_id="capacity-2",
                )
            )
            bridge.record_execution(
                execution(
                    "position-2/exit-hedge",
                    220,
                    role="exit_hedge",
                    market="future",
                    side="buy",
                    quantity=1,
                    unit="future_contracts",
                    request_id="position-2/exit-risk",
                    sources=(
                        InitiatingExecutionAllocation(
                            "position-2/exit-maker",
                            2000,
                        ),
                    ),
                    position_id="position-2",
                    capacity_id="capacity-2",
                )
            )
            bridge.seal_terminal(
                terminal_id="position-2/terminal",
                position_id="position-2",
                cursor=EventCursor(230, 0, 0),
                terminal_outcome="exit_maker_flat",
                capacity_release_transition_id="position-2/released",
            )
        self.assertEqual(terminal_and_open.facts, uninterrupted.facts)

    def test_resume_rejects_context_mismatch_and_tampered_facts(self) -> None:
        bridge = self.bridge()
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
        facts = bridge.facts
        kwargs = {
            "default_date": "20260825",
            "scenario_id": "q95",
            "products": (S1AccountingProduct("product-1", "1101", 2000),),
            "facts": facts,
        }

        with self.assertRaisesRegex(ValueError, "scenario_id"):
            S1AccountingBridge.from_facts(**{**kwargs, "scenario_id": "q80"})
        with self.assertRaisesRegex(ValueError, "product mapping"):
            S1AccountingBridge.from_facts(
                **{
                    **kwargs,
                    "products": (S1AccountingProduct("product-1", "9999", 2000),),
                }
            )
        with self.assertRaisesRegex(ValueError, "one-to-one"):
            S1AccountingBridge.from_facts(
                **{
                    **kwargs,
                    "products": (
                        S1AccountingProduct("product-1", "1101", 2000),
                        S1AccountingProduct("product-2", "1101", 2000),
                    ),
                }
            )

        tampered_money = (replace(facts[0], signed_cashflow_twd=1.0),)
        with self.assertRaises(AccountingReplayError):
            S1AccountingBridge.from_facts(**{**kwargs, "facts": tampered_money})
        tampered_profile = (replace(facts[0], cost_profile_id="tampered"),)
        with self.assertRaisesRegex(ValueError, "cost profile"):
            S1AccountingBridge.from_facts(**{**kwargs, "facts": tampered_profile})
        changed_profile = replace(
            TransactionCostProfile(),
            spot_commission_multiplier=0.2,
        )
        with self.assertRaises(AccountingReplayError):
            S1AccountingBridge.from_facts(**kwargs, profile=changed_profile)
        with self.assertRaisesRegex(AccountingReplayError, "cursor calendar"):
            S1AccountingBridge.from_facts(
                **kwargs,
                execution_date_resolver=lambda _cursor: "20260826",
            )

    def test_resume_rejects_future_contract_size_mismatch(self) -> None:
        bridge = self.bridge()
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
        bridge.record_execution(
            execution(
                "entry-hedge",
                2,
                role="entry_hedge",
                market="future",
                side="sell",
                quantity=1,
                unit="future_contracts",
                request_id="entry-risk",
                sources=(InitiatingExecutionAllocation("entry-maker", 2000),),
            )
        )
        with self.assertRaisesRegex(ValueError, "contract size"):
            S1AccountingBridge.from_facts(
                default_date="20260825",
                scenario_id="q95",
                products=(S1AccountingProduct("product-1", "1101", 1000),),
                facts=bridge.facts,
            )


if __name__ == "__main__":
    unittest.main()
