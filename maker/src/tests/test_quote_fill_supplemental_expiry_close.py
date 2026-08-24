from __future__ import annotations

from datetime import time
import unittest

import polars as pl

from maker.src.quote_fill.portfolio_cap_backtester import local_session_timestamp_ns
from maker.src.quote_fill.supplemental_expiry_close import (
    EXACT_PRICE_SOURCE,
    OVERLAY_VERSION,
    TERMINAL_REASON,
    TERMINAL_RESOLUTION,
    apply_expiry_daily_close_overlay,
)


D1 = "20260102"
D2 = "20260105"
IDENTITY = "a" * 64


def _path(identifier: str, *, expiry: bool) -> dict[str, object]:
    resolution = "expiry_last_valid_session_mark" if expiry else "source_completed"
    gross = 10.0 if expiry else 5.0
    return {
        "Date": D1,
        "ValueCode": "2330" if expiry else "2317",
        "QuoteCode": "A1234" if expiry else "B1234",
        "policy_path_id": identifier,
        "position_established_ns": local_session_timestamp_ns(D1, time(9, 10)),
        "entry_spot_price": 100.0,
        "entry_future_price": 105.0,
        "entry_contract_size_shares": 10,
        "normalization_notional_twd": 1_000.0,
        "terminal_date": D2 if expiry else D1,
        "exit_decision_time_ns": local_session_timestamp_ns(
            D2 if expiry else D1, time(13, 20)
        ),
        "exit_spot_price": 99.0 if expiry else 100.0,
        "exit_future_price": 103.0 if expiry else 104.5,
        "gross_cycle_pnl_twd": gross,
        "gross_cycle_bp": gross / 1_000.0 * 10_000.0,
        "supplemental_terminal_resolution": resolution,
        "supplemental_terminal_source_identity_sha256": "b" * 64,
        "supplemental_overlay_version": "v2",
        "outcome_status": resolution,
        "terminal_reason": resolution,
        "completed_same_day": not expiry,
        "completed_overnight": expiry,
        "terminal_cashflow_priced": True,
        "unresolved_cashflow_imputed": False,
        "exact_price_source": "legacy",
        "expiry_uses_last_valid_session_mark": expiry,
        "expiry_uses_last_observed_session_mark": False,
        "expiry_mark_uses_trade_fallback": False,
        "spot_expiry_mark_is_executable_bbo": True if expiry else None,
        "future_expiry_mark_is_executable_bbo": True if expiry else None,
        "expiry_mark_is_official_close": False if expiry else None,
        "expiry_mark_is_official_settlement": False if expiry else None,
    }


def _mark() -> dict[str, object]:
    return {
        "Date": D2,
        "expiry_session": D2,
        "calendar_version": "v1",
        "ValueCode": "2330",
        "QuoteCode": "A1234",
        "spot_close_price": 99.0,
        "future_close_price": 103.0,
        "spot_close_time_ns": local_session_timestamp_ns(D2, time(13, 20)),
        "future_close_time_ns": local_session_timestamp_ns(D2, time(13, 20)),
        "spot_close_source": "legacy_bid",
        "future_close_source": "legacy_ask",
        "source_identity_sha256": "b" * 64,
        "mark_is_official_close": False,
        "mark_is_official_settlement": False,
        "spot_mark_is_executable_bbo": True,
        "future_mark_is_executable_bbo": True,
        "mark_uses_trade_fallback": False,
        "mark_role": "legacy",
    }


class SupplementalExpiryCloseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.paths = pl.from_dicts(
            [_path("expiry", expiry=True), _path("normal", expiry=False)],
            infer_schema_length=None,
        )
        self.marks = pl.from_dicts([_mark()], infer_schema_length=None)
        self.audit = pl.from_dicts(
            [
                {
                    "Date": D1,
                    "ValueCode": "2330",
                    "policy_path_id": "expiry",
                    "terminal_resolution": "expiry_last_valid_session_mark",
                }
            ],
            infer_schema_length=None,
        )
        self.facts = pl.from_dicts(
            [
                {
                    "Date": D2,
                    "ValueCode": "2330",
                    "QuoteCode": "A1234",
                    "spot_close_price": 101.0,
                    "future_close_price": 104.0,
                    "source_identity_sha256": IDENTITY,
                }
            ],
            infer_schema_length=None,
        )

    def test_only_expiry_path_is_repriced_from_paired_daily_closes(self) -> None:
        result = apply_expiry_daily_close_overlay(
            self.paths, self.marks, self.audit, self.facts
        )
        expiry = result.supplemental_paths.filter(
            pl.col("policy_path_id") == "expiry"
        ).row(0, named=True)
        normal_before = self.paths.filter(pl.col("policy_path_id") == "normal").row(
            0, named=True
        )
        normal_after = result.supplemental_paths.filter(
            pl.col("policy_path_id") == "normal"
        ).row(0, named=True)
        self.assertEqual(normal_before, normal_after)
        self.assertEqual(expiry["exit_spot_price"], 101.0)
        self.assertEqual(expiry["exit_future_price"], 104.0)
        self.assertEqual(expiry["gross_cycle_pnl_twd"], 20.0)
        self.assertEqual(expiry["gross_cycle_bp"], 200.0)
        self.assertEqual(expiry["supplemental_terminal_resolution"], TERMINAL_RESOLUTION)
        self.assertEqual(expiry["terminal_reason"], TERMINAL_REASON)
        self.assertEqual(expiry["supplemental_overlay_version"], OVERLAY_VERSION)
        self.assertEqual(expiry["exact_price_source"], EXACT_PRICE_SOURCE)
        self.assertTrue(expiry["expiry_mark_is_official_close"])
        self.assertFalse(expiry["expiry_mark_is_official_settlement"])
        self.assertFalse(expiry["expiry_mark_uses_trade_fallback"])
        self.assertEqual(
            expiry["exit_decision_time_ns"],
            local_session_timestamp_ns(D2, time(13, 30)),
        )
        mark = result.expiry_marks.row(0, named=True)
        self.assertEqual(mark["spot_close_price"], 101.0)
        self.assertEqual(mark["future_close_price"], 104.0)
        self.assertTrue(mark["mark_is_official_close"])
        self.assertFalse(mark["mark_is_official_settlement"])
        self.assertFalse(mark["spot_mark_is_executable_bbo"])
        self.assertEqual(
            result.continuation_audit.item(0, "terminal_resolution"),
            TERMINAL_RESOLUTION,
        )
        self.assertEqual(result.overlay_audit.item(0, "gross_pnl_delta_twd"), 10.0)

    def test_requires_exact_fact_coverage(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            apply_expiry_daily_close_overlay(
                self.paths, self.marks, self.audit, self.facts.clear()
            )


if __name__ == "__main__":
    unittest.main()
