"""Mature-cohort diagnostics must not select only fast completed positions."""
from pathlib import Path
from dataclasses import replace
from tempfile import TemporaryDirectory
import unittest

import polars as pl

from ..backtest.ev_validation import calibration, decision_errors
from .test_causal_replay import decision


class EVCalibrationTest(unittest.TestCase):
    def positions(self):
        base = dict(id="mature", stream="S1", route="residual", hedge_ns=1_000_000_000,
                    quote_day="20260126", close_day="20260129", expiry="20260131", state="closed",
                    close_ns=1_000_000_000+3*86400*1_000_000_000, entry_spot_cash=1_000_000_000,
                    pnl_net=100., ev_bp=20., t_days_pred=2., p_sd_pred=.25)
        return [base, {**base, "id": "fast_but_unmature", "expiry": "20260228", "pnl_net": 9999.},
                {**base, "id": "still_open", "expiry": "20260228", "state": "paired", "close_ns": None,
                 "close_day": None, "pnl_net": None},
                {**base, "id": "failed_entry", "hedge_ns": None, "pnl_net": -100.}]

    def test_maturity_excludes_even_closed_trades_from_unmature_cohort(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            pl.from_dicts(self.positions()).write_parquet(root/"A_positions_all.parquet")
            result = calibration(root, dict(days=["20260203"], configs={"A": {}}))
            self.assertEqual(result["scope"]["A"], dict(paired=3, mature=1, unresolved_mature=0,
                excluded_unmature=2, failed_entry_cycles=1))
            cohort = next(r for r in result["cohorts"] if r["dimension"] == "all")
            self.assertAlmostEqual(cohort["realized_net_bp"], 10.)
            self.assertAlmostEqual(cohort["predicted_net_twd"], 200.)
            self.assertAlmostEqual(cohort["realized_days"], 3.)
            self.assertAlmostEqual(cohort["brier_same_day"], .0625)

    def test_unresolved_expired_inventory_blocks_final_diagnostic(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            rows = self.positions()
            rows[0]["state"] = "paired"
            pl.from_dicts(rows).write_parquet(root/"A_positions_all.parquet")
            with self.assertRaisesRegex(ValueError, "mature inventory remains unresolved"):
                calibration(root, dict(days=["20260203"], configs={"A": {}}))

    def test_refresh_rejection_and_portfolio_hurdle_are_recomputed(self):
        d = decision()
        row = {k:getattr(d.best,k) for k in ("score","ev_bp","t_days","p_sd","route")}
        row.update(base_admit=True,A="submitted",B="hurdle")
        self.assertFalse(decision_errors(row,d,{"A":5.,"B":100.}))
        self.assertIn("B_below_hurdle_admission", decision_errors({**row,"B":"submitted"},d,{"B":100.}))
        self.assertIn("A_false_hurdle_rejection", decision_errors({**row,"A":"hurdle"},d,{"A":5.}))

    def test_rejected_refresh_without_estimate_is_valid_but_false_send_fails(self):
        d = replace(decision(),admit=False,reason="no_estimate",best=None,evals=())
        row = dict(base_admit=False,route=None,score=None,ev_bp=None,t_days=None,p_sd=None,
                   A="no_estimate",B="not_requested")
        self.assertFalse(decision_errors(row,d,{"A":5.,"B":5.}))
        self.assertIn("A_invalid_rejection", decision_errors({**row,"A":"submitted"},d,{"A":5.}))


if __name__ == "__main__":
    unittest.main()
