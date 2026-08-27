"""Focused actual-new target-freezing contracts for S1 Spot Bid."""

from __future__ import annotations

import math
import unittest
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from ..quote_fill.layered import EventCursor
from ..quote_fill.policy_spec import PolicySpec
from ..quote_fill.s1_target import (
    S1_ROUTE,
    actual_send_tod_bucket,
    build_s1_spot_bid_target,
)
from ..quote_fill.targets import absolute_price_tick

DATE = "20260505"
TZ = ZoneInfo("Asia/Taipei")


def _cursor(second: int, nanoseconds: int = 0) -> EventCursor:
    start = datetime(2026, 5, 5, 9, 0, tzinfo=TZ)
    recv_time_ns = (
        int((start + timedelta(seconds=second)).timestamp()) * 1_000_000_000
        + nanoseconds
    )
    return EventCursor(recv_time_ns, 4, 17)


def _q_spec(tod: str = "0905_1000") -> PolicySpec:
    return PolicySpec(
        Date=DATE,
        ValueCode="2330",
        QuoteCode="CDFE6",
        entry_tod_bucket=tod,
        policy_id="q95",
        kind="quantile",
        upper_distance_bp=30.0,
        lower_distance_bp=0.0,
        upper_source_id="Q2_trail20_date_equal",
        lower_source_id="C0_center",
        upper_source_asof_date="20260504",
        lower_source_asof_date="20260504",
        combined_source_asof_date="20260504",
        boundary_quantile=95,
    )


def _fixed_spec(tod: str = "0905_1000") -> PolicySpec:
    return PolicySpec(
        Date=DATE,
        ValueCode="2330",
        QuoteCode="CDFE6",
        entry_tod_bucket=tod,
        policy_id="fixed20",
        kind="fixed",
        upper_distance_bp=20.0,
        lower_distance_bp=20.0,
        upper_source_id="constant_bp:20",
        lower_source_id="constant_bp:20",
        upper_source_asof_date=None,
        lower_source_asof_date=None,
        combined_source_asof_date=None,
    )


def _build(spec: PolicySpec, cursor: EventCursor, **overrides: object):
    values: dict[str, object] = {
        "date": DATE,
        "value_code": "2330",
        "quote_code": "CDFE6",
        "route": S1_ROUTE,
        "actual_new_send_cursor": cursor,
        "causal_anchor_basis_bp": 10.0,
        "fut_exec_bid": 101.0,
        "spot_bid": 100.0,
        "spot_ask": 100.8,
        "contract_size_shares": 1000.0,
    }
    values.update(overrides)
    return build_s1_spot_bid_target(spec, **values)  # type: ignore[arg-type]


class S1TargetTest(unittest.TestCase):
    def test_q_c0_freezes_center_exit_and_full_cursor_provenance(self) -> None:
        cursor = _cursor(300, 123)
        target = _build(_q_spec(), cursor)

        self.assertEqual(target.actual_new_send_cursor, cursor)
        self.assertEqual(target.actual_new_seconds_from_open, 300)
        self.assertEqual(target.entry_tod_bucket, "0905_1000")
        self.assertTrue(
            math.isclose(target.entry_threshold_basis_bp, 40.0, abs_tol=1e-12)
        )
        self.assertTrue(
            math.isclose(
                target.frozen_exit_threshold_basis_bp,
                10.0,
                abs_tol=1e-12,
            )
        )
        self.assertEqual(target.lower_source_id, "C0_center")
        self.assertEqual(
            target.absolute_price_tick,
            absolute_price_tick(target.target_price),
        )
        self.assertGreaterEqual(target.effective_entry_basis_bp, 40.0)
        self.assertEqual(
            target.reservation_notional_twd,
            math.ceil(target.target_price * 1000.0),
        )
        self.assertEqual(target.target_location, "inside_spread")

    def test_fixed_policy_is_symmetric_and_target_passivity_is_observed(self) -> None:
        target = _build(_fixed_spec(), _cursor(301))
        self.assertTrue(
            math.isclose(target.entry_threshold_basis_bp, 30.0, abs_tol=1e-12)
        )
        self.assertTrue(
            math.isclose(
                target.frozen_exit_threshold_basis_bp,
                -10.0,
                abs_tol=1e-12,
            )
        )
        self.assertTrue(target.passive_target)
        self.assertIsNone(target.upper_source_asof_date)

        non_passive = _build(
            _fixed_spec(),
            _cursor(301),
            spot_bid=target.target_price - 0.1,
            spot_ask=target.target_price,
        )
        self.assertFalse(non_passive.passive_target)

    def test_tod_boundaries_are_half_open_and_must_match_spec(self) -> None:
        expectations = {
            300: "0905_1000",
            3599: "0905_1000",
            3600: "1000_1100",
            7199: "1000_1100",
            7200: "1100_1200",
            10799: "1100_1200",
            10800: "1200_1300",
            14399: "1200_1300",
        }
        for second, expected in expectations.items():
            with self.subTest(second=second):
                self.assertEqual(
                    actual_send_tod_bucket(DATE, _cursor(second))[1],
                    expected,
                )
        for second in (299, 14400):
            with self.assertRaisesRegex(ValueError, "outside"):
                actual_send_tod_bucket(DATE, _cursor(second))
        with self.assertRaisesRegex(ValueError, "TOD bucket"):
            _build(_q_spec("0905_1000"), _cursor(3600))

    def test_bad_identity_route_cursor_and_market_values_fail_closed(self) -> None:
        spec = _q_spec()
        with self.assertRaisesRegex(ValueError, "identity"):
            _build(spec, _cursor(300), value_code="2317")
        with self.assertRaisesRegex(ValueError, "Spot Bid"):
            _build(spec, _cursor(300), route="future_ask_spot_taker")
        wrong_day = EventCursor(_cursor(300).recv_time_ns + 86_400_000_000_000)
        with self.assertRaisesRegex(ValueError, "outside"):
            _build(spec, wrong_day)
        for name, value in (
            ("causal_anchor_basis_bp", math.nan),
            ("fut_exec_bid", 0.0),
            ("spot_bid", 0.0),
            ("spot_ask", -1.0),
            ("contract_size_shares", math.inf),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                _build(spec, _cursor(300), **{name: value})

    def test_frozen_target_is_immutable_and_spec_change_builds_new_target(self) -> None:
        target = _build(_q_spec(), _cursor(300))
        with self.assertRaises(FrozenInstanceError):
            target.causal_anchor_basis_bp = 99.0  # type: ignore[misc]

        rebuilt = _build(
            replace(_q_spec(), upper_distance_bp=100.0),
            _cursor(300),
        )
        self.assertFalse(
            math.isclose(
                target.target_price,
                rebuilt.target_price,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        self.assertTrue(
            math.isclose(target.causal_anchor_basis_bp, 10.0, abs_tol=1e-12)
        )

    def test_nonintegral_contract_and_crossed_spot_book_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "integer share"):
            _build(
                _q_spec(),
                _cursor(300),
                contract_size_shares=1000.5,
            )
        with self.assertRaisesRegex(ValueError, "bid cannot exceed"):
            _build(
                _q_spec(),
                _cursor(300),
                spot_bid=101.0,
                spot_ask=100.0,
            )


if __name__ == "__main__":
    unittest.main()
