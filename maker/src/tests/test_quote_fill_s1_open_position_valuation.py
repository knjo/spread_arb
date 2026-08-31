from __future__ import annotations

import json
import unittest
from decimal import Decimal

from ..quote_fill.layered import EventCursor
from ..quote_fill.s1_accounting import S1AccountingLedger
from ..quote_fill.s1_hedge import (
    CausalBookState,
    RawBookCursor,
    RawBookLevel,
)
from ..quote_fill.s1_open_position_valuation import (
    COMMON_HORIZON_CURSOR,
    S1CommonHorizonBooks,
    S1OpenPositionValuationError,
    decode_s1_open_position_valuation,
    encode_s1_open_position_valuation,
    value_s1_common_horizon_open_positions,
)

SCENARIO = "q95_C0_sd_f5"


def _cursor(value: int) -> EventCursor:
    return EventCursor(value, 0, 0)


def _common(
    execution_id: str,
    cursor_value: int,
    *,
    execution_date: str,
    role: str = "normal",
    initiating_execution_id: str | None = None,
    position_id: str = "p",
) -> dict[str, object]:
    linked = role in ("hedge", "rollback")
    return {
        "execution_id": execution_id,
        "position_id": position_id,
        "value_code": "2330",
        "scenario_id": SCENARIO,
        "capacity_id": f"cap-{position_id}",
        "role": role,
        "execution_truth": "exact",
        "execution_source_id": "indexed-replay",
        "request_id": f"request-{execution_id}",
        "hedge_intent_id": f"hedge-{execution_id}" if linked else None,
        "initiating_execution_id": initiating_execution_id,
        "execution_date": execution_date,
        "cursor": _cursor(cursor_value),
    }


def _rollback_reopened_ledger() -> S1AccountingLedger:
    ledger = S1AccountingLedger()
    ledger.record_spot_execution(
        **_common("spot-open", 1, execution_date="20260812"),
        side="buy",
        price=100,
        shares=2000,
    )
    ledger.record_future_execution(
        **_common(
            "future-open",
            2,
            execution_date="20260812",
            role="hedge",
            initiating_execution_id="spot-open",
        ),
        side="sell",
        price=103,
        contracts=1,
        share_equivalent=2000,
    )
    ledger.establish_position(
        establishment_id="established",
        position_id="p",
        establishment_date="20260812",
        cursor=_cursor(3),
        position_established_ns=3,
        capacity_transition_id="paired-capacity",
    )
    ledger.record_spot_execution(
        **_common("spot-exit", 4, execution_date="20260813"),
        side="sell",
        price=101,
        shares=2000,
    )
    ledger.record_spot_execution(
        **_common(
            "spot-rollback",
            5,
            execution_date="20260813",
            role="rollback",
            initiating_execution_id="spot-exit",
        ),
        side="buy",
        price=102,
        shares=2000,
    )
    ledger.record_future_execution(
        **_common("future-exit", 6, execution_date="20260813"),
        side="buy",
        price=101,
        contracts=1,
        share_equivalent=2000,
    )
    ledger.record_future_execution(
        **_common(
            "future-rollback",
            7,
            execution_date="20260813",
            role="rollback",
            initiating_execution_id="future-exit",
        ),
        side="sell",
        price=102,
        contracts=1,
        share_equivalent=2000,
    )
    return ledger.verify()


def _two_open_position_ledger() -> S1AccountingLedger:
    ledger = S1AccountingLedger()
    for position_id, start, spot_price, future_price in (
        ("p1", 1, 100, 103),
        ("p2", 4, 101, 104),
    ):
        spot_id = f"{position_id}-spot-open"
        ledger.record_spot_execution(
            **_common(
                spot_id,
                start,
                execution_date="20260812",
                position_id=position_id,
            ),
            side="buy",
            price=spot_price,
            shares=1000,
        )
        ledger.record_future_execution(
            **_common(
                f"{position_id}-future-open",
                start + 1,
                execution_date="20260812",
                role="hedge",
                initiating_execution_id=spot_id,
                position_id=position_id,
            ),
            side="sell",
            price=future_price,
            contracts=1,
            share_equivalent=1000,
        )
        ledger.establish_position(
            establishment_id=f"{position_id}-established",
            position_id=position_id,
            establishment_date="20260812",
            cursor=_cursor(start + 2),
            position_established_ns=start + 2,
            capacity_transition_id=f"{position_id}-paired-capacity",
        )
    return ledger.verify()


def _book(
    *,
    bid_price: float,
    bid_quantity: int,
    ask_price: float,
    ask_quantity: int,
    packet: int,
) -> CausalBookState:
    return CausalBookState(
        book_cursor=RawBookCursor(
            EventCursor(COMMON_HORIZON_CURSOR.recv_time_ns - packet, 10, 0),
            packet,
        ),
        gate_open=True,
        gate_reason=None,
        reference_price=(bid_price + ask_price) / 2,
        bids=(RawBookLevel(bid_price, bid_quantity),),
        asks=(RawBookLevel(ask_price, ask_quantity),),
    )


def _books(*, spot_bid_quantity: int = 2000) -> S1CommonHorizonBooks:
    return S1CommonHorizonBooks(
        spot=_book(
            bid_price=103,
            bid_quantity=spot_bid_quantity,
            ask_price=104,
            ask_quantity=2000,
            packet=2,
        ),
        future=_book(
            bid_price=100,
            bid_quantity=2,
            ask_price=101,
            ask_quantity=2,
            packet=1,
        ),
    )


class S1OpenPositionValuationTest(unittest.TestCase):
    def test_same_product_positions_share_aggregate_depth_fail_closed(self) -> None:
        ledger = _two_open_position_ledger()
        result = value_s1_common_horizon_open_positions(
            ledger.facts,
            scenario_id=SCENARIO,
            books_by_value_code={"2330": _books(spot_bid_quantity=1500)},
        )

        self.assertEqual(len(result.rows), 2)
        self.assertTrue(all(not row.priced for row in result.rows))
        self.assertEqual(
            {row.unpriced_reason for row in result.rows},
            {"spot_insufficient_depth"},
        )
        self.assertIsNone(result.aggregate)

    def test_same_product_aggregate_vwap_conserves_position_totals(self) -> None:
        ledger = _two_open_position_ledger()
        result = value_s1_common_horizon_open_positions(
            ledger.facts,
            scenario_id=SCENARIO,
            books_by_value_code={"2330": _books(spot_bid_quantity=2000)},
        )

        self.assertTrue(result.publishable)
        self.assertEqual(len(result.rows), 2)
        self.assertEqual({row.spot_exit_vwap for row in result.rows}, {Decimal("103.0")})
        self.assertEqual(
            {row.future_exit_vwap for row in result.rows},
            {Decimal("101.0")},
        )
        assert result.aggregate is not None
        self.assertEqual(
            result.aggregate.gross_mark_pnl_twd,
            sum(
                (row.gross_mark_pnl_twd for row in result.rows),
                start=Decimal(0),
            ),
        )
        self.assertEqual(
            result.aggregate.remaining_exit_cost_twd,
            sum(
                (row.remaining_exit_cost_twd for row in result.rows),
                start=Decimal(0),
            ),
        )
        self.assertEqual(
            result.aggregate.net_mark_pnl_twd,
            sum(
                (row.net_mark_pnl_twd for row in result.rows),
                start=Decimal(0),
            ),
        )

    def test_rollback_reopened_lots_drive_gross_and_same_day_exit_tax(self) -> None:
        ledger = _rollback_reopened_ledger()
        snapshot = ledger.open_inventory_snapshot("p")
        self.assertEqual(snapshot.spot_lots[0].opening_price, 102)
        self.assertEqual(snapshot.spot_lots[0].acquisition_date, "20260813")
        self.assertEqual(snapshot.future_lots[0].opening_price, 102)

        result = value_s1_common_horizon_open_positions(
            ledger.facts,
            scenario_id=SCENARIO,
            books_by_value_code={"2330": _books()},
        )

        self.assertTrue(result.publishable)
        self.assertIsNotNone(result.aggregate)
        row = result.rows[0]
        self.assertTrue(row.priced)
        self.assertEqual(row.prior_spot_cashflow_twd, Decimal("-202000.0"))
        self.assertEqual(row.prior_futures_realized_pnl_twd, Decimal("4000.0"))
        self.assertEqual(row.spot_liquidation_cashflow_twd, Decimal("206000.0"))
        self.assertEqual(row.futures_unrealized_pnl_twd, Decimal("2000.0"))
        self.assertEqual(row.gross_mark_pnl_twd, Decimal("10000.0"))
        expected_tax = Decimal(
            str(
                ledger.profile.spot_sell_tax_twd(
                    103,
                    2000,
                    same_day=True,
                )
            )
        )
        overnight_tax = Decimal(
            str(
                ledger.profile.spot_sell_tax_twd(
                    103,
                    2000,
                    same_day=False,
                )
            )
        )
        self.assertEqual(row.remaining_spot_sell_tax_twd, expected_tax)
        self.assertNotEqual(row.remaining_spot_sell_tax_twd, overnight_tax)
        self.assertEqual(
            row.net_mark_pnl_twd,
            row.gross_mark_pnl_twd
            - row.incurred_commission_twd
            - row.incurred_tax_twd
            - row.remaining_exit_cost_twd,
        )
        assert result.aggregate is not None
        self.assertEqual(result.aggregate.gross_mark_pnl_twd, Decimal("10000.0"))
        self.assertEqual(result.aggregate.net_mark_pnl_twd, row.net_mark_pnl_twd)

    def test_depth_short_book_is_explicitly_unpriced_and_blocks_aggregate(
        self,
    ) -> None:
        ledger = _rollback_reopened_ledger()
        result = value_s1_common_horizon_open_positions(
            ledger.facts,
            scenario_id=SCENARIO,
            books_by_value_code={"2330": _books(spot_bid_quantity=1000)},
        )

        self.assertFalse(result.publishable)
        self.assertIsNone(result.aggregate)
        self.assertEqual(len(result.rows), 1)
        row = result.rows[0]
        self.assertFalse(row.priced)
        self.assertEqual(row.unpriced_reason, "spot_insufficient_depth")
        self.assertIsNone(row.gross_mark_pnl_twd)
        self.assertGreater(result.incurred_open_execution_cost_twd, 0)

        wire = json.loads(
            json.dumps(
                encode_s1_open_position_valuation(row),
                allow_nan=False,
                sort_keys=True,
            )
        )
        self.assertEqual(decode_s1_open_position_valuation(wire), row)

    def test_missing_and_illegal_books_are_explicitly_unpriced(self) -> None:
        ledger = _rollback_reopened_ledger()
        missing = value_s1_common_horizon_open_positions(
            ledger.facts,
            scenario_id=SCENARIO,
            books_by_value_code={},
        )
        self.assertEqual(
            missing.rows[0].unpriced_reason,
            "missing_spot_and_future_books",
        )
        self.assertIsNone(missing.aggregate)

        valid = _books()
        crossed_spot = _book(
            bid_price=105,
            bid_quantity=2000,
            ask_price=104,
            ask_quantity=2000,
            packet=3,
        )
        illegal = value_s1_common_horizon_open_positions(
            ledger.facts,
            scenario_id=SCENARIO,
            books_by_value_code={
                "2330": S1CommonHorizonBooks(
                    spot=crossed_spot,
                    future=valid.future,
                )
            },
        )
        self.assertEqual(illegal.rows[0].unpriced_reason, "spot_crossed_book")
        self.assertIsNone(illegal.aggregate)

    def test_post_horizon_expiry_terminal_is_not_open_valued(self) -> None:
        ledger = _rollback_reopened_ledger()
        expiry_ns = COMMON_HORIZON_CURSOR.recv_time_ns + 600_000_000_000
        ledger.record_expiry_accounting_mark(
            mark_id="p-expiry",
            position_id="p",
            expiry_date="20260813",
            cursor=EventCursor(expiry_ns, 600, 0),
            spot_close_source_id="official-close/20260813/2330",
            spot_close_source_cursor=EventCursor(expiry_ns - 1, 20, 0),
            spot_close_price=101,
            capacity_release_transition_id="p-expiry-release",
        )

        result = value_s1_common_horizon_open_positions(
            ledger.facts,
            scenario_id=SCENARIO,
            books_by_value_code={},
        )

        self.assertEqual(result.rows, ())
        self.assertTrue(result.publishable)
        assert result.aggregate is not None
        self.assertEqual(result.aggregate.position_count, 0)
        self.assertEqual(result.aggregate.net_mark_pnl_twd, 0)

    def test_record_codec_is_strict_and_checks_cost_conservation(self) -> None:
        ledger = _rollback_reopened_ledger()
        row = value_s1_common_horizon_open_positions(
            ledger.facts,
            scenario_id=SCENARIO,
            books_by_value_code={"2330": _books()},
        ).rows[0]
        record = encode_s1_open_position_valuation(row)
        self.assertEqual(decode_s1_open_position_valuation(record), row)

        missing = dict(record)
        missing.pop("position_id")
        with self.assertRaisesRegex(S1OpenPositionValuationError, "schema mismatch"):
            decode_s1_open_position_valuation(missing)

        extra = {**record, "unexpected": None}
        with self.assertRaisesRegex(S1OpenPositionValuationError, "schema mismatch"):
            decode_s1_open_position_valuation(extra)

        wrong_type = {**record, "gross_mark_pnl_twd": 10000.0}
        with self.assertRaisesRegex(S1OpenPositionValuationError, "decimal text"):
            decode_s1_open_position_valuation(wrong_type)

        drifted = {**record, "net_mark_pnl_twd": "1"}
        with self.assertRaisesRegex(S1OpenPositionValuationError, "does not conserve"):
            decode_s1_open_position_valuation(drifted)


if __name__ == "__main__":
    unittest.main()
