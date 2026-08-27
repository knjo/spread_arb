"""Contracts for compact S1 daily report diagnostics."""

from __future__ import annotations

import unittest

from ..quote_fill.capacity_ledger import CapacityLedger
from ..quote_fill.s1_accounting_bridge import (
    S1AccountingBridge,
    S1AccountingProduct,
)
from ..quote_fill.s1_daily_diagnostics import (
    S1DailyDiagnosticsError,
    build_s1_daily_diagnostics,
    validate_s1_daily_diagnostics,
)
from ..quote_fill.s1_entry_day_runner import run_s1_entry_policy
from .test_quote_fill_s1_portfolio_runner import D1, VALUE, _prepared


class S1DailyDiagnosticsTest(unittest.TestCase):
    def test_builds_json_safe_closed_counters_and_distributions(self) -> None:
        prepared = _prepared(D1)
        ledger = CapacityLedger()
        accounting = S1AccountingBridge(
            default_date=D1,
            scenario_id="q95",
            products=(S1AccountingProduct(VALUE, VALUE, 2_000),),
        )
        run = run_s1_entry_policy(
            prepared,
            "q95",
            normal_exit_enabled=True,
            accounting_adapter=accounting,
            capacity_ledger=ledger,
        )

        record = build_s1_daily_diagnostics(run, ledger)

        self.assertEqual(record["date"], D1)
        self.assertEqual(record["policy_id"], "q95")
        self.assertEqual(record["active_fill_count"], 1)
        self.assertEqual(record["position_state_counts"], {"paired_open": 1})
        self.assertEqual(record["carry_out_positions"], 1)
        self.assertGreater(record["carry_out_notional_twd"], 0)
        self.assertEqual(record["naked_unresolved_positions"], 0)
        self.assertEqual(validate_s1_daily_diagnostics(record), record)

    def test_rejects_schema_counter_and_sample_tamper(self) -> None:
        prepared = _prepared(D1)
        ledger = CapacityLedger()
        run = run_s1_entry_policy(
            prepared,
            "fixed20",
            normal_exit_enabled=True,
            capacity_ledger=ledger,
        )
        record = build_s1_daily_diagnostics(run, ledger)

        with (
            self.subTest("extra"),
            self.assertRaisesRegex(S1DailyDiagnosticsError, "schema"),
        ):
            validate_s1_daily_diagnostics({**record, "extra": 1})
        with (
            self.subTest("bool count"),
            self.assertRaisesRegex(S1DailyDiagnosticsError, "nonnegative integer"),
        ):
            validate_s1_daily_diagnostics({**record, "active_fill_count": True})
        with (
            self.subTest("sample count"),
            self.assertRaisesRegex(S1DailyDiagnosticsError, "count/sample"),
        ):
            validate_s1_daily_diagnostics({**record, "active_fill_latency_ms": []})


if __name__ == "__main__":
    unittest.main()
