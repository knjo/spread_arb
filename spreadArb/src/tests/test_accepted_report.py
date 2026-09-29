"""A final-performance report must reject partial or mismatched evidence."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from ..backtest.accepted_report import report


class ReportAcceptanceTest(unittest.TestCase):
    def fixture(self, root):
        manifest = dict(days=["20260706", "20260707"], configs={
            "A": dict(reserve_on_submit=False, max_positions_per_product=None),
            "B": dict(reserve_on_submit=False, max_positions_per_product=None)})
        verification = dict(complete=True, status="PASS", official_valuation=True,
                            metrics={"raw": {"dates": manifest["days"]}})
        ev = dict(status="PASS", complete=True)
        return manifest, verification, ev

    def rejected(self, field, value, message):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, verification, ev = self.fixture(root)
            if field == "raw_dates":
                verification["metrics"]["raw"]["dates"] = value
            elif field == "ev_complete":
                ev["complete"] = value
            elif field == "reservation":
                manifest["configs"]["A"]["reserve_on_submit"] = value
            else:
                verification[field] = value
            for name, data in (("manifest", manifest), ("verification", verification), ("ev_validation", ev)):
                (root/f"{name}.json").write_text(json.dumps(data))
            with patch("spreadArb.src.backtest.accepted_report.grid_days", return_value=manifest["days"]):
                with self.assertRaisesRegex(ValueError, message):
                    report(root)

    def test_incomplete_period_is_rejected(self):
        self.rejected("complete", False, "complete canonical period")

    def test_partial_raw_scope_is_rejected(self):
        self.rejected("raw_dates", ["20260706"], "every session")

    def test_missing_official_valuation_is_rejected(self):
        self.rejected("official_valuation", False, "official equity")

    def test_skipped_signal_check_is_rejected(self):
        self.rejected("ev_complete", False, "submitted-signal")

    def test_full_reservation_is_rejected(self):
        self.rejected("reservation", True, "user-confirmed")

    def test_b_only_report_does_not_require_or_link_a_outputs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, verification, ev = self.fixture(root)
            manifest["configs"] = {"B": dict(reserve_on_submit=False,
                max_positions_per_product=None, quote_refresh_ns=60_000_000_000)}
            verification["metrics"]["raw"].update(checked_legs=1,
                raw_quote_prices_checked=1, raw_decision_inputs_checked=1)
            fields = ("pnl_twd", "annual_simple_pct", "closed_pnl_twd", "terminal_mark_twd", "daily_twd",
                "paired_entries", "rollback_cycles", "capital_peak_twd", "mtm_drawdown_twd", "raised_days",
                "mandatory_capacity_cancels_checked", "quote_lifetimes_checked")
            verification["metrics"]["B"] = {key: 1 for key in fields}
            ev.update(signals=dict(submitted_signals=1, refresh_checks=1, checked_signals=1),
                      calibration=dict(cohorts=[]))
            for name, data in (("manifest", manifest), ("verification", verification), ("ev_validation", ev)):
                (root/f"{name}.json").write_text(json.dumps(data))
            (root/"B_daily.csv").write_text("peak_rss_gib\n1.0\n")
            with patch("spreadArb.src.backtest.accepted_report.grid_days", return_value=manifest["days"]), \
                    patch("spreadArb.src.backtest.accepted_report.plot_equity"):
                result = report(root).read_text()
            self.assertIn("| 項目 | B 動態 hurdle |", result)
            self.assertIn("B_reconciled.csv", result)
            self.assertNotIn("A_reconciled.csv", result)


if __name__ == "__main__":
    unittest.main()
