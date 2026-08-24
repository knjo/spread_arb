from __future__ import annotations

import unittest
from unittest.mock import patch

from maker.src.quote_fill import exit_maker as exit_maker_module
from maker.src.quote_fill.exit_maker import (
    FUTURE_BID_EXIT_ROUTE,
    SPOT_ASK_EXIT_ROUTE,
    PairedExitPosition,
    build_exit_maker_order_windows,
    make_exit_maker_observation,
    project_earliest_full_fill_oco,
    replay_exit_maker_window,
    replay_exit_maker_windows,
)
from maker.src.quote_fill.hedge import BookLevel, OppositeBookSnapshot
from maker.src.quote_fill.indexed_replay import IndexedTradeReplay
from maker.src.quote_fill.layered import EventCursor
from maker.src.quote_fill.replay import IndependentOrderWindow, TradeEvent
from maker.src.quote_fill.targets import absolute_price_tick


MS = 1_000_000


def _cursor(ms: int, sequence: int = 0) -> EventCursor:
    return EventCursor(ms * MS, sequence, 0)


def _book(
    ms: int,
    *,
    bids: tuple[tuple[float, int], ...] = ((100.0, 20),),
    asks: tuple[tuple[float, int], ...] = ((101.0, 20),),
) -> OppositeBookSnapshot:
    return OppositeBookSnapshot(
        _cursor(ms),
        tuple(BookLevel(*level) for level in bids),
        tuple(BookLevel(*level) for level in asks),
    )


def _observation(
    route: str,
    ms: int,
    epoch: int,
    *,
    spot_bid: float,
    spot_ask: float,
    future_bid: float,
    future_ask: float,
    spot_sell_exec_price: float | None = None,
    future_buy_exec_price: float | None = None,
    threshold: float = 0.0,
    queue: int | None = 0,
    session_date: str | None = None,
):
    reference = future_bid if route == FUTURE_BID_EXIT_ROUTE else spot_ask
    return make_exit_maker_observation(
        route,
        _cursor(ms),
        epoch,
        threshold,
        session_date=session_date,
        spot_bid=spot_bid,
        spot_ask=spot_ask,
        future_bid=future_bid,
        future_ask=future_ask,
        spot_sell_exec_price=(
            spot_bid if spot_sell_exec_price is None else spot_sell_exec_price
        ),
        future_buy_exec_price=(
            future_ask if future_buy_exec_price is None else future_buy_exec_price
        ),
        maker_reference_price=reference,
        initial_queue_ahead=queue,
    )


class ExitMakerTargetTest(unittest.TestCase):
    def test_future_exit_target_uses_session_ladder_regime(self) -> None:
        pre_change = _observation(
            FUTURE_BID_EXIT_ROUTE,
            10,
            1,
            spot_bid=2130.0,
            spot_ask=2135.0,
            future_bid=2130.0,
            future_ask=2135.0,
            threshold=5.0,
            session_date="20260703",
        )
        post_change = _observation(
            FUTURE_BID_EXIT_ROUTE,
            10,
            1,
            spot_bid=2130.0,
            spot_ask=2135.0,
            future_bid=2130.0,
            future_ask=2135.0,
            threshold=5.0,
            session_date="20260706",
        )
        self.assertEqual(pre_change.target_price, 2130.0)
        self.assertEqual(post_change.target_price, 2131.0)
        self.assertEqual(
            post_change.absolute_target_tick,
            absolute_price_tick(
                2131.0,
                market="future",
                session_date="20260706",
            ),
        )

    def test_both_routes_emit_legal_passive_ticks(self) -> None:
        future = _observation(
            FUTURE_BID_EXIT_ROUTE,
            10,
            1,
            spot_bid=100.0,
            spot_ask=100.5,
            future_bid=100.0,
            future_ask=100.5,
            threshold=30.0,
        )
        spot = _observation(
            SPOT_ASK_EXIT_ROUTE,
            10,
            1,
            spot_bid=100.0,
            spot_ask=100.5,
            future_bid=100.0,
            future_ask=100.5,
            threshold=30.0,
        )
        self.assertTrue(future.gate_open)
        self.assertTrue(spot.gate_open)
        self.assertEqual(absolute_price_tick(future.target_price), future.absolute_target_tick)
        self.assertEqual(absolute_price_tick(spot.target_price), spot.absolute_target_tick)
        self.assertLessEqual(future.effective_basis_bp, 30.0)
        self.assertLessEqual(spot.effective_basis_bp, 30.0)

    def test_crossing_target_is_clamped_to_passive_tick_not_rejected(self) -> None:
        open_observation = _observation(
            FUTURE_BID_EXIT_ROUTE,
            10,
            1,
            spot_bid=100.0,
            spot_ask=100.5,
            future_bid=99.5,
            future_ask=100.5,
        )
        crossed = _observation(
            FUTURE_BID_EXIT_ROUTE,
            20,
            1,
            spot_bid=100.0,
            spot_ask=100.5,
            future_bid=99.0,
            future_ask=100.0,
        )
        self.assertTrue(crossed.gate_open)
        self.assertTrue(crossed.passive_clamped)
        self.assertTrue(crossed.passive)
        self.assertAlmostEqual(crossed.target_price, 99.9)
        self.assertTrue(crossed.threshold_already_taker_executable)
        spot_crossed = _observation(
            SPOT_ASK_EXIT_ROUTE,
            20,
            1,
            spot_bid=100.0,
            spot_ask=101.0,
            future_bid=99.5,
            future_ask=100.0,
        )
        self.assertTrue(spot_crossed.gate_open)
        self.assertTrue(spot_crossed.passive_clamped)
        self.assertAlmostEqual(spot_crossed.target_price, 100.5)
        self.assertTrue(spot_crossed.passive)
        built = build_exit_maker_order_windows(
            [open_observation, crossed],
            route=FUTURE_BID_EXIT_ROUTE,
            policy_id="exit/future/center",
            cutoff_cursor=_cursor(30),
        )
        self.assertEqual(len(built.windows), 1)
        self.assertEqual(built.windows[0].stop_cursor, _cursor(20))
        self.assertEqual(built.windows[0].stop_reason, "target_retreat")

    def test_required_quantity_vwap_not_bbo_drives_target(self) -> None:
        bbo_target = _observation(
            FUTURE_BID_EXIT_ROUTE,
            10,
            1,
            spot_bid=100.0,
            spot_ask=100.5,
            future_bid=99.0,
            future_ask=101.0,
            threshold=0.0,
        )
        l2_vwap_target = _observation(
            FUTURE_BID_EXIT_ROUTE,
            20,
            2,
            spot_bid=100.0,
            spot_ask=100.5,
            future_bid=99.0,
            future_ask=101.0,
            spot_sell_exec_price=99.8,
            threshold=0.0,
        )
        self.assertEqual(bbo_target.target_price, 100.0)
        self.assertAlmostEqual(l2_vwap_target.target_price, 99.8)
        self.assertNotEqual(
            bbo_target.absolute_target_tick,
            l2_vwap_target.absolute_target_tick,
        )

        spot_l2 = _observation(
            SPOT_ASK_EXIT_ROUTE,
            30,
            3,
            spot_bid=100.0,
            spot_ask=102.0,
            future_bid=100.0,
            future_ask=100.5,
            future_buy_exec_price=101.0,
            threshold=0.0,
        )
        self.assertEqual(spot_l2.target_price, 101.0)

    def test_forward_layer_and_retreat_use_absolute_exit_ticks(self) -> None:
        observations = [
            _observation(
                FUTURE_BID_EXIT_ROUTE,
                10,
                1,
                spot_bid=100.0,
                spot_ask=100.5,
                future_bid=99.5,
                future_ask=101.0,
            ),
            _observation(
                FUTURE_BID_EXIT_ROUTE,
                20,
                1,
                spot_bid=101.0,
                spot_ask=101.5,
                future_bid=100.5,
                future_ask=102.0,
            ),
            _observation(
                FUTURE_BID_EXIT_ROUTE,
                30,
                1,
                spot_bid=100.0,
                spot_ask=100.5,
                future_bid=99.5,
                future_ask=101.0,
            ),
        ]
        built = build_exit_maker_order_windows(
            observations,
            route=FUTURE_BID_EXIT_ROUTE,
            policy_id="exit/future/lower",
            cutoff_cursor=_cursor(40),
        )
        by_tick = {window.target_price_tick: window for window in built.windows}
        aggressive = observations[1].absolute_target_tick
        base = observations[0].absolute_target_tick
        self.assertEqual(by_tick[aggressive].stop_cursor, _cursor(30))
        self.assertEqual(by_tick[aggressive].stop_reason, "target_retreat")
        self.assertEqual(by_tick[base].stop_reason, "session_cutoff")


class ExitMakerReplayTest(unittest.TestCase):
    def test_batch_replay_matches_scalar_and_validates_snapshots_once(self) -> None:
        target = absolute_price_tick(100.0)
        windows = (
            IndependentOrderWindow(
                "future-exit/batch-1",
                "bid",
                target,
                _cursor(1),
                _cursor(40),
                0,
                "target_retreat",
            ),
            IndependentOrderWindow(
                "future-exit/batch-2",
                "bid",
                target,
                _cursor(50),
                _cursor(100),
                0,
                "session_cutoff",
            ),
        )
        maker_replay = IndexedTradeReplay(
            [TradeEvent(_cursor(10, 1), target, 1)]
        )
        snapshots = (_book(9), _book(60))
        expected = tuple(
            replay_exit_maker_window(
                window,
                route=FUTURE_BID_EXIT_ROUTE,
                maker_replay=maker_replay,
                opposite_snapshots=snapshots,
                eod_cursor=_cursor(100),
            )
            for window in windows
        )

        validator = exit_maker_module._validate_snapshot_order
        with patch.object(
            exit_maker_module,
            "_validate_snapshot_order",
            wraps=validator,
        ) as validate:
            actual = replay_exit_maker_windows(
                windows,
                route=FUTURE_BID_EXIT_ROUTE,
                maker_replay=maker_replay,
                opposite_snapshots=snapshots,
                eod_cursor=_cursor(100),
            )

        self.assertEqual(actual, expected)
        self.assertEqual(validate.call_count, 1)

    def test_batch_replay_rejects_invalid_snapshot_order_before_replay(self) -> None:
        target = absolute_price_tick(100.0)
        window = IndependentOrderWindow(
            "future-exit/invalid-snapshots",
            "bid",
            target,
            _cursor(1),
            _cursor(100),
            0,
            "session_cutoff",
        )
        with self.assertRaisesRegex(ValueError, "strictly cursor-sorted"):
            replay_exit_maker_windows(
                (window,),
                route=FUTURE_BID_EXIT_ROUTE,
                maker_replay=IndexedTradeReplay([]),
                opposite_snapshots=(_book(60), _book(9)),
                eod_cursor=_cursor(100),
            )

    def test_future_bid_full_fill_sells_spot_after_50ms_and_flattens(self) -> None:
        target = absolute_price_tick(100.0)
        window = IndependentOrderWindow(
            "future-exit/1",
            "bid",
            target,
            _cursor(1),
            _cursor(100),
            0,
            "session_cutoff",
        )
        outcome = replay_exit_maker_window(
            window,
            route=FUTURE_BID_EXIT_ROUTE,
            maker_replay=IndexedTradeReplay(
                [TradeEvent(_cursor(10, 1), target, 1)]
            ),
            opposite_snapshots=[
                _book(9, bids=((100.0, 10),)),
                _book(60, bids=((99.5, 1), (99.0, 10))),
            ],
            eod_cursor=_cursor(100),
        )
        self.assertEqual(outcome.branch_status, "flat_same_day")
        self.assertTrue(outcome.position_flat)
        self.assertFalse(outcome.eod_carry)
        self.assertFalse(outcome.cancel_required)
        self.assertEqual(len(outcome.hedge_attempts), 1)
        attempt = outcome.hedge_attempts[0]
        self.assertEqual(attempt.decision_time_ns, 60 * MS)
        self.assertEqual(attempt.execution.hedge_side, "sell")
        self.assertEqual(attempt.execution.executable_vwap_price, 99.25)
        self.assertEqual(attempt.execution.levels_swept, 2)
        self.assertEqual(outcome.residual_position.spot_long_lots, 0)
        self.assertEqual(outcome.residual_position.future_short_contracts, 0)

    def test_partial_future_fill_is_hedged_and_remainder_is_canceled(self) -> None:
        target = absolute_price_tick(100.0)
        window = IndependentOrderWindow(
            "future-exit/scale",
            "bid",
            target,
            _cursor(1),
            _cursor(100),
            0,
            "target_retreat",
        )
        outcome = replay_exit_maker_window(
            window,
            route=FUTURE_BID_EXIT_ROUTE,
            maker_replay=IndexedTradeReplay(
                [TradeEvent(_cursor(10, 1), target, 1)]
            ),
            opposite_snapshots=[
                _book(9, bids=((100.0, 10),)),
                _book(60, bids=((99.5, 10),)),
            ],
            eod_cursor=_cursor(200),
            starting_position=PairedExitPosition(4, 2),
        )
        self.assertEqual(outcome.branch_status, "partial_fill_then_cancel")
        self.assertTrue(outcome.partial_fill)
        self.assertTrue(outcome.cancel_required)
        self.assertEqual(len(outcome.hedge_attempts), 1)
        self.assertTrue(outcome.hedge_attempts[0].hedge_complete)
        self.assertEqual(outcome.residual_position.spot_long_lots, 2)
        self.assertEqual(outcome.residual_position.future_short_contracts, 1)

    def test_spot_ask_full_fill_buys_future_after_second_lot(self) -> None:
        target = absolute_price_tick(100.5)
        window = IndependentOrderWindow(
            "spot-exit/1",
            "ask",
            target,
            _cursor(1),
            _cursor(150),
            0,
            "session_cutoff",
        )
        outcome = replay_exit_maker_window(
            window,
            route=SPOT_ASK_EXIT_ROUTE,
            maker_replay=IndexedTradeReplay(
                [
                    TradeEvent(_cursor(10, 2), target, 1),
                    TradeEvent(_cursor(40, 2), target, 1),
                ]
            ),
            opposite_snapshots=[
                _book(39, asks=((101.0, 10),)),
                _book(90, asks=((101.5, 1), (102.0, 10))),
            ],
            eod_cursor=_cursor(150),
        )
        self.assertEqual(outcome.branch_status, "flat_same_day")
        self.assertEqual(outcome.hedge_attempts[0].maker_fill_cursor, _cursor(40, 2))
        self.assertEqual(outcome.hedge_attempts[0].decision_time_ns, 90 * MS)
        self.assertEqual(outcome.hedge_attempts[0].execution.hedge_side, "buy")
        self.assertEqual(outcome.hedge_attempts[0].execution.executable_vwap_price, 101.5)

    def test_odd_spot_partial_is_not_rounded_to_a_fractional_future(self) -> None:
        target = absolute_price_tick(100.5)
        window = IndependentOrderWindow(
            "spot-exit/partial",
            "ask",
            target,
            _cursor(1),
            _cursor(50),
            0,
            "target_retreat",
        )
        outcome = replay_exit_maker_window(
            window,
            route=SPOT_ASK_EXIT_ROUTE,
            maker_replay=IndexedTradeReplay(
                [TradeEvent(_cursor(10, 2), target, 1)]
            ),
            opposite_snapshots=[_book(10), _book(60)],
            eod_cursor=_cursor(100),
        )
        self.assertEqual(outcome.branch_status, "partial_fill_then_cancel")
        self.assertEqual(outcome.unhedgeable_maker_quantity, 1)
        self.assertEqual(outcome.hedge_attempts, ())
        self.assertEqual(outcome.residual_position.spot_long_lots, 1)
        self.assertEqual(outcome.residual_position.future_short_contracts, 1)

    def test_no_fill_and_post_eod_hedge_are_explicit_carry_branches(self) -> None:
        target = absolute_price_tick(100.0)
        no_fill_window = IndependentOrderWindow(
            "future-exit/no-fill",
            "bid",
            target,
            _cursor(1),
            _cursor(100),
            0,
            "session_cutoff",
        )
        no_fill = replay_exit_maker_window(
            no_fill_window,
            route=FUTURE_BID_EXIT_ROUTE,
            maker_replay=IndexedTradeReplay([]),
            opposite_snapshots=[],
            eod_cursor=_cursor(100),
        )
        self.assertEqual(
            no_fill.branch_status, "carry_at_eod_cancel_unconfirmed"
        )
        self.assertTrue(no_fill.eod_carry)
        self.assertFalse(no_fill.cancel_ack_observed)
        self.assertFalse(no_fill.cancel_race_modeled)
        self.assertEqual(no_fill.cancel_model, "nominal_instant_cancel_v0")
        self.assertFalse(no_fill.pathwise_ev_ready)

        cancel_window = IndependentOrderWindow(
            "future-exit/cancel-no-fill",
            "bid",
            target,
            _cursor(1),
            _cursor(50),
            0,
            "target_retreat",
        )
        cancel_no_fill = replay_exit_maker_window(
            cancel_window,
            route=FUTURE_BID_EXIT_ROUTE,
            maker_replay=IndexedTradeReplay([]),
            opposite_snapshots=[],
            eod_cursor=_cursor(100),
        )
        self.assertEqual(
            cancel_no_fill.branch_status, "no_fill_before_cancel_request"
        )
        self.assertTrue(cancel_no_fill.cancel_required)
        self.assertFalse(cancel_no_fill.cancel_ack_observed)
        self.assertFalse(cancel_no_fill.cancel_race_modeled)
        self.assertFalse(cancel_no_fill.pathwise_ev_ready)

        late_window = IndependentOrderWindow(
            "future-exit/late-fill",
            "bid",
            target,
            _cursor(1),
            _cursor(100),
            0,
            "session_cutoff",
        )
        late = replay_exit_maker_window(
            late_window,
            route=FUTURE_BID_EXIT_ROUTE,
            maker_replay=IndexedTradeReplay(
                [TradeEvent(_cursor(80, 1), target, 1)]
            ),
            opposite_snapshots=[_book(79)],
            eod_cursor=_cursor(100),
        )
        self.assertEqual(late.hedge_attempts[0].status, "decision_after_eod")
        self.assertEqual(late.branch_status, "hedge_incomplete_carry_at_eod")
        self.assertTrue(late.eod_carry)
        self.assertEqual(late.residual_position.spot_long_lots, 2)
        self.assertEqual(late.residual_position.future_short_contracts, 0)

    def test_oco_projects_earliest_full_fill_without_inventing_cancel_ack(self) -> None:
        first_tick = absolute_price_tick(100.0)
        second_tick = absolute_price_tick(99.5)
        third_tick = absolute_price_tick(99.0)
        windows = (
            IndependentOrderWindow(
                "oco/first", "bid", first_tick, _cursor(1), _cursor(100), 0,
                "session_cutoff",
            ),
            IndependentOrderWindow(
                "oco/second", "bid", second_tick, _cursor(1), _cursor(100), 0,
                "session_cutoff",
            ),
            IndependentOrderWindow(
                "oco/later", "bid", third_tick, _cursor(25), _cursor(100), 0,
                "session_cutoff",
            ),
        )
        replay = IndexedTradeReplay(
            [
                TradeEvent(_cursor(20, 1), first_tick, 1),
                TradeEvent(_cursor(30, 1), second_tick, 1),
            ]
        )
        snapshots = [_book(19), _book(70), _book(80)]
        outcomes = tuple(
            replay_exit_maker_window(
                window,
                route=FUTURE_BID_EXIT_ROUTE,
                maker_replay=replay,
                opposite_snapshots=snapshots,
                eod_cursor=_cursor(100),
            )
            for window in windows
        )
        projection = project_earliest_full_fill_oco(zip(windows, outcomes))
        self.assertEqual(projection.winner_generation_id, "oco/first")
        by_id = {member.generation_id: member for member in projection.members}
        self.assertEqual(by_id["oco/first"].disposition, "winner")
        self.assertEqual(
            by_id["oco/second"].disposition, "sibling_cancel_required"
        )
        self.assertEqual(
            by_id["oco/second"].oco_cancel_request_cursor, _cursor(20, 1)
        )
        self.assertTrue(
            by_id["oco/second"].independent_full_fill_after_cancel_request
        )
        self.assertEqual(
            by_id["oco/later"].disposition, "not_active_at_winner"
        )
        self.assertTrue(projection.position_projection_safe)
        self.assertFalse(projection.cancel_ack_observed)
        self.assertFalse(projection.cancel_race_modeled)
        self.assertFalse(projection.strict_ev_ready)


if __name__ == "__main__":
    unittest.main()
