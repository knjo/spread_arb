"""Tests for the pure one-second buy-side quote controller."""

from __future__ import annotations

import unittest

from ..quote_fill.one_second_message_load import OneSecondBidQuoteController


def _shape(actions: tuple[object, ...]) -> list[tuple[str, str, int, int]]:
    return [
        (
            action.kind,
            action.reason,
            action.absolute_price_tick,
            action.submit_point_offset,
        )
        for action in actions
    ]


class OneSecondBidQuoteControllerTest(unittest.TestCase):
    def test_resampling_or_current_point_change_does_not_replace_same_price(self) -> None:
        controller = OneSecondBidQuoteController()

        initial = controller.reconcile(
            1,
            100,
            True,
            True,
            0,
        )
        self.assertEqual(_shape(initial), [
            ("submit", "initial_eligible", 100, 0)
        ])
        self.assertEqual(
            controller.reconcile(
                2,
                100,
                True,
                True,
                -1,
            ),
            (),
        )
        self.assertEqual(len(controller.active_quotes), 1)
        self.assertEqual(controller.active_quotes[0].submit_point_offset, 0)

    def test_forward_adds_layer_and_retreat_only_cancels_higher_prices(self) -> None:
        controller = OneSecondBidQuoteController()
        first = controller.reconcile(1, 100, True, True, -1)[0]
        forward = controller.reconcile(2, 101, True, True, 0)

        self.assertEqual(_shape(forward), [
            ("submit", "forward_new_price", 101, 0)
        ])
        self.assertEqual(
            [quote.absolute_price_tick for quote in controller.active_quotes],
            [100, 101],
        )

        retreat = controller.reconcile(3, 99, True, True, -1)
        self.assertEqual(_shape(retreat), [
            ("cancel", "target_retreat", 100, -1),
            ("cancel", "target_retreat", 101, 0),
        ])
        self.assertEqual(controller.active_quotes, ())
        self.assertEqual(controller.reconcile(4, 99, True, True, -1), ())
        self.assertEqual(first.generation, retreat[0].generation)

    def test_canceled_price_can_submit_as_a_new_generation_on_forward_revisit(self) -> None:
        controller = OneSecondBidQuoteController()
        controller.reconcile(1, 100, True, True, -1)
        old = controller.reconcile(2, 101, True, True, 0)[0]
        controller.reconcile(3, 100, True, True, -1)

        revisit = controller.reconcile(4, 101, True, True, 0)
        self.assertEqual(_shape(revisit), [
            ("submit", "forward_new_price", 101, 0)
        ])
        self.assertNotEqual(revisit[0].generation, old.generation)
        self.assertEqual(
            [quote.absolute_price_tick for quote in controller.active_quotes],
            [100, 101],
        )

    def test_gate_close_cancels_all_and_reopen_submits_same_target_once(self) -> None:
        controller = OneSecondBidQuoteController()
        controller.reconcile(1, 100, True, True, -1)
        controller.reconcile(2, 101, True, True, 0)

        closed = controller.reconcile(
            3,
            None,
            False,
            False,
            None,
            gate_reason="book_invalid",
        )
        self.assertEqual(
            [(action.kind, action.reason) for action in closed],
            [("cancel", "book_invalid"), ("cancel", "book_invalid")],
        )

        reopened = controller.reconcile(4, 101, True, True, -1)
        self.assertEqual(_shape(reopened), [
            ("submit", "gate_reopen", 101, -1)
        ])
        self.assertEqual(controller.reconcile(5, 101, True, True, 0), ())

    def test_unchanged_previously_unquoted_target_can_become_eligible(self) -> None:
        controller = OneSecondBidQuoteController()
        self.assertEqual(
            controller.reconcile(
                1,
                100,
                True,
                False,
                -2,
            ),
            (),
        )
        self.assertEqual(
            controller.reconcile(
                2,
                100,
                True,
                False,
                -2,
            ),
            (),
        )

        admitted = controller.reconcile(
            3,
            100,
            True,
            True,
            -1,
        )
        self.assertEqual(_shape(admitted), [
            ("submit", "became_admission_eligible", 100, -1)
        ])

        # Eligibility is a submit filter, not a cancellation gate.
        self.assertEqual(
            controller.reconcile(
                4,
                100,
                True,
                False,
                -2,
            ),
            (),
        )
        self.assertEqual(len(controller.active_quotes), 1)

    def test_cutoff_cancels_all_with_each_original_submit_offset(self) -> None:
        controller = OneSecondBidQuoteController()
        controller.reconcile(1, 100, True, True, -1)
        controller.reconcile(2, 101, True, True, 0)

        cutoff = controller.close(3)
        self.assertEqual(_shape(cutoff), [
            ("cancel", "session_cutoff", 100, -1),
            ("cancel", "session_cutoff", 101, 0),
        ])
        self.assertTrue(controller.cutoff_reached)
        self.assertEqual(controller.active_quotes, ())
        with self.assertRaisesRegex(RuntimeError, "after session cutoff"):
            controller.reconcile(4, 100, True, True, 0)

    def test_validation_is_strict_and_does_not_consume_bad_second(self) -> None:
        controller = OneSecondBidQuoteController()
        with self.assertRaisesRegex(ValueError, "requires submit_point_offset"):
            controller.reconcile(1, 100, True, True, None)
        self.assertEqual(
            controller.reconcile(1, 100, True, True, 0)[0].kind,
            "submit",
        )
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            controller.reconcile(1, 100, True, True, 0)


if __name__ == "__main__":
    unittest.main()
