from __future__ import annotations

import itertools
import unittest
from dataclasses import replace

from maker.src.quote_fill.layered import EventCursor
from maker.src.quote_fill.s1_accounting import PositionEstablishedFact
from maker.src.quote_fill.s1_exit_fifo_coordinator import (
    ExitFifoCoordinatorError,
    build_exit_fifo_desired_updates,
)
from maker.src.quote_fill.s1_exit_inventory import (
    ExitDesiredUpdate,
    ExitPositionView,
    S1ExitInventoryController,
)
from maker.src.quote_fill.s1_exit_target import S1SpotAskTarget

OBSERVATION_CURSOR = EventCursor(100, 2, 0)
BOOK_CURSOR = EventCursor(100, 0, 0)


def position(
    position_id: str,
    established_ns: int,
    *,
    value_code: str = "2330",
    scenario_id: str = "scenario-a",
    contract_size_shares: int = 2000,
    available_spot_shares: int = 2000,
    desired_absolute_target_tick: int | None = None,
    desired_shares: int = 0,
    allocation_id: str | None = None,
    allocation_kind: str | None = None,
    unhedged_fill_shares: int = 0,
    hedge_pending_shares: int = 0,
    hedged_exit_shares: int = 0,
    rollback_pending_shares: int = 0,
    rollback_failed_shares: int = 0,
) -> ExitPositionView:
    unresolved_shares = (
        available_spot_shares
        + unhedged_fill_shares
        + hedge_pending_shares
        + rollback_pending_shares
        + rollback_failed_shares
    )
    unresolved_reasons = tuple(
        reason
        for quantity, reason in (
            (available_spot_shares, "available_inventory"),
            (unhedged_fill_shares, "unhedged_fill"),
            (hedge_pending_shares, "hedge_pending"),
            (rollback_pending_shares, "rollback_pending"),
            (rollback_failed_shares, "rollback_failed"),
        )
        if quantity
    )
    return ExitPositionView(
        position_id=position_id,
        value_code=value_code,
        scenario_id=scenario_id,
        capacity_id=f"capacity-{position_id}",
        position_established_ns=established_ns,
        contract_size_shares=contract_size_shares,
        available_spot_shares=available_spot_shares,
        desired_absolute_target_tick=desired_absolute_target_tick,
        desired_shares=desired_shares,
        allocation_id=allocation_id,
        allocation_kind=allocation_kind,  # type: ignore[arg-type]
        unhedged_fill_shares=unhedged_fill_shares,
        hedge_pending_shares=hedge_pending_shares,
        hedged_exit_shares=hedged_exit_shares,
        rollback_pending_shares=rollback_pending_shares,
        rollback_failed_shares=rollback_failed_shares,
        unresolved_shares=unresolved_shares,
        resolution_state="flat" if unresolved_shares == 0 else "unresolved",
        unresolved_reasons=unresolved_reasons,  # type: ignore[arg-type]
        capacity_releasable=unresolved_shares == 0,
    )


def target(
    position_id: str,
    tick: int | None,
    *,
    gate_open: bool = True,
    gate_reason: str | None = None,
    value_code: str = "2330",
    scenario_id: str = "scenario-a",
    observation_cursor: EventCursor = OBSERVATION_CURSOR,
) -> S1SpotAskTarget:
    reason = gate_reason or ("eligible" if gate_open else "missing_spot_book")
    return S1SpotAskTarget(
        date="20260827",
        value_code=value_code,
        quote_code="2330F",
        position_id=position_id,
        scenario_id=scenario_id,
        observation_cursor=observation_cursor,
        frozen_exit_threshold_basis_bp=0.0,
        future_buy_vwap=100.0 if gate_open else None,
        target_price=float(tick) if tick is not None else None,
        absolute_price_tick=tick,
        effective_exit_basis_bp=0.0 if gate_open else None,
        passive_target=gate_open,
        target_in_reference_band=gate_open,
        target_location="at_ask1" if gate_open else None,
        initial_queue_ahead_shares=1000 if gate_open else None,
        queue_observable=gate_open,
        gate_open=gate_open,
        gate_reason=reason,
        spot_book_cursor=BOOK_CURSOR if gate_open else None,
        future_book_cursor=BOOK_CURSOR if gate_open else None,
    )


def build(
    positions: tuple[ExitPositionView, ...],
    targets: tuple[S1SpotAskTarget, ...],
) -> tuple[ExitDesiredUpdate, ...]:
    return build_exit_fifo_desired_updates(
        positions,
        targets,
        observation_cursor=OBSERVATION_CURSOR,
    )


class ExitFifoCoordinatorTests(unittest.TestCase):
    def test_same_tick_integral_positions_form_maximal_prefix(self) -> None:
        positions = (position("p1", 1), position("p2", 2))
        updates = build(
            positions,
            (target("p1", 100), target("p2", 100)),
        )
        self.assertEqual(
            updates,
            (
                ExitDesiredUpdate("p1", 100, 2000),
                ExitDesiredUpdate("p2", 100, 2000),
            ),
        )

    def test_first_different_tick_stops_all_newer_positions(self) -> None:
        positions = tuple(position(f"p{index}", index) for index in range(1, 5))
        updates = build(
            positions,
            (
                target("p1", 100),
                target("p2", 100),
                target("p3", 101),
                target("p4", 100),
            ),
        )
        self.assertEqual(
            updates,
            (
                ExitDesiredUpdate("p1", 100, 2000),
                ExitDesiredUpdate("p2", 100, 2000),
                ExitDesiredUpdate("p3", None, 0),
                ExitDesiredUpdate("p4", None, 0),
            ),
        )

    def test_oldest_closed_gate_blocks_every_newer_position(self) -> None:
        updates = build(
            (position("p1", 1), position("p2", 2)),
            (
                target("p1", None, gate_open=False),
                target("p2", 100),
            ),
        )
        self.assertEqual(
            updates,
            (
                ExitDesiredUpdate("p1", None, 0),
                ExitDesiredUpdate("p2", None, 0),
            ),
        )

    def test_oldest_execution_state_blocks_every_newer_position(self) -> None:
        blocked_states = {
            "unhedged": {
                "available_spot_shares": 1400,
                "unhedged_fill_shares": 600,
            },
            "hedge_pending": {
                "available_spot_shares": 0,
                "hedge_pending_shares": 2000,
            },
            "rollback_pending": {
                "available_spot_shares": 0,
                "rollback_pending_shares": 2000,
            },
            "rollback_failed": {
                "available_spot_shares": 0,
                "rollback_failed_shares": 2000,
            },
        }
        for name, state in blocked_states.items():
            with self.subTest(name=name):
                updates = build(
                    (position("p1", 1, **state), position("p2", 2)),
                    (target("p1", 100), target("p2", 100)),
                )
                self.assertEqual(
                    updates,
                    (
                        ExitDesiredUpdate("p1", None, 0),
                        ExitDesiredUpdate("p2", None, 0),
                    ),
                )

    def test_nonintegral_oldest_inventory_is_a_fifo_stop(self) -> None:
        updates = build(
            (
                position("p1", 1, available_spot_shares=1400),
                position("p2", 2),
            ),
            (target("p1", 100), target("p2", 100)),
        )
        self.assertEqual(
            updates,
            (
                ExitDesiredUpdate("p1", None, 0),
                ExitDesiredUpdate("p2", None, 0),
            ),
        )

    def test_rollback_restored_old_position_reclaims_fifo_priority(self) -> None:
        restored_old = position(
            "p1",
            1,
            available_spot_shares=2000,
            desired_shares=0,
            desired_absolute_target_tick=None,
        )
        newer = position(
            "p2",
            2,
            desired_shares=2000,
            desired_absolute_target_tick=101,
            allocation_id="working-p2",
            allocation_kind="working",
        )
        updates = build(
            (newer, restored_old),
            (target("p2", 101), target("p1", 100)),
        )
        self.assertEqual(
            updates,
            (
                ExitDesiredUpdate("p1", 100, 2000),
                ExitDesiredUpdate("p2", None, 0),
            ),
        )

    def test_flat_prefix_is_transparent_and_always_desired_zero(self) -> None:
        flat = position(
            "p0",
            1,
            available_spot_shares=0,
            hedged_exit_shares=2000,
        )
        positions = (flat, position("p1", 2), position("p2", 3))
        updates = build(
            positions,
            (
                target("p0", None, gate_open=False),
                target("p1", 100),
                target("p2", 100),
            ),
        )
        self.assertEqual(
            updates,
            (
                ExitDesiredUpdate("p0", None, 0),
                ExitDesiredUpdate("p1", 100, 2000),
                ExitDesiredUpdate("p2", 100, 2000),
            ),
        )

    def test_input_permutations_have_one_canonical_replay_result(self) -> None:
        positions = (
            position("p1", 1),
            position("p2", 2),
            position("p3", 3),
        )
        targets = (
            target("p1", 100),
            target("p2", 100),
            target("p3", 101),
        )
        expected = build(positions, targets)
        for position_order in itertools.permutations(positions):
            for target_order in itertools.permutations(targets):
                self.assertEqual(build(position_order, target_order), expected)

    def test_output_batch_is_accepted_directly_by_inventory_controller(self) -> None:
        state = S1ExitInventoryController(
            Date="20260827",
            ValueCode="2330",
            QuoteCode="2330F",
            scenario_id="scenario-a",
        )
        for sequence, (position_id, established_ns, add_ns) in enumerate(
            (("p1", 1, 2), ("p2", 3, 4)),
            start=1,
        ):
            state.add_paired_position(
                PositionEstablishedFact(
                    sequence=sequence,
                    establishment_id=f"establishment-{position_id}",
                    position_id=position_id,
                    value_code="2330",
                    scenario_id="scenario-a",
                    capacity_id=f"capacity-{position_id}",
                    establishment_date="20260827",
                    cursor=EventCursor(established_ns),
                    position_established_ns=established_ns,
                    capacity_transition_id=f"paired-{position_id}",
                    spot_shares=2000,
                    short_future_contracts=1,
                    short_future_share_equivalent=2000,
                    execution_truth="exact",
                ),
                absolute_target_tick=99,
                cursor=EventCursor(add_ns),
            )
        observation = EventCursor(5)
        outputs = build_exit_fifo_desired_updates(
            state.positions,
            (
                replace(
                    target("p1", 100),
                    observation_cursor=observation,
                    spot_book_cursor=observation,
                    future_book_cursor=observation,
                ),
                replace(
                    target("p2", 100),
                    observation_cursor=observation,
                    spot_book_cursor=observation,
                    future_book_cursor=observation,
                ),
            ),
            observation_cursor=observation,
        )
        fact = state.set_positions_desired(outputs, cursor=EventCursor(6))
        self.assertEqual(fact.desired_updates, outputs)
        self.assertEqual(
            tuple((item.position_id, item.desired_shares) for item in state.positions),
            (("p1", 2000), ("p2", 2000)),
        )

    def test_rejects_scope_identity_coverage_and_cursor_corruption(self) -> None:
        p1 = position("p1", 1)
        p2 = position("p2", 2)
        t1 = target("p1", 100)
        t2 = target("p2", 100)
        cases = (
            (
                "value_code",
                (p1, replace(p2, value_code="2317")),
                (t1, t2),
            ),
            (
                "scenario_id",
                (p1, replace(p2, scenario_id="scenario-b")),
                (t1, t2),
            ),
            ("repeat position_id", (p1, p1), (t1, t2)),
            ("repeat position_id", (p1, p2), (t1, t1)),
            ("cover positions exactly", (p1, p2), (t1,)),
            (
                "value_code differs",
                (p1, p2),
                (replace(t1, value_code="2317"), t2),
            ),
            (
                "scenario_id differs",
                (p1, p2),
                (replace(t1, scenario_id="scenario-b"), t2),
            ),
            (
                "target cursor differs",
                (p1, p2),
                (
                    replace(t1, observation_cursor=EventCursor(99)),
                    t2,
                ),
            ),
        )
        for message, positions, targets in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ExitFifoCoordinatorError, message),
            ):
                build(positions, targets)

    def test_rejects_internally_inconsistent_views_and_targets(self) -> None:
        valid_position = position("p1", 1)
        valid_target = target("p1", 100)
        with self.assertRaisesRegex(
            ExitFifoCoordinatorError,
            "unresolved quantity buckets",
        ):
            build(
                (replace(valid_position, unresolved_shares=1),),
                (valid_target,),
            )
        with self.assertRaisesRegex(
            ExitFifoCoordinatorError,
            "capacity_releasable",
        ):
            build(
                (replace(valid_position, capacity_releasable=True),),
                (valid_target,),
            )
        with self.assertRaisesRegex(
            ExitFifoCoordinatorError,
            "open target gate requires an absolute tick",
        ):
            build((valid_position,), (replace(valid_target, absolute_price_tick=None),))
        with self.assertRaisesRegex(
            ExitFifoCoordinatorError,
            "closed target gate cannot have eligible reason",
        ):
            build(
                (valid_position,),
                (replace(valid_target, gate_open=False),),
            )


if __name__ == "__main__":
    unittest.main()
