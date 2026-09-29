"""Integration regressions for inside-only orders and causal cash-normalized EV."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from .causal_lookup import CLOSE_SECOND, SECOND, LookupHistory, open_ns
from .cost_observations import observe_costs
from .cost_study import build, validate_resume
from .decide import EntryDecider
from .execution import price_i, realized_pnl
from .full_study import checkpoint, restore
from .liquidity_portfolio import LiquidityPortfolio
from .test_xc import XcPortfolioTests


class CostTests(unittest.TestCase):
    def test_exit_cost_identity_uses_entry_cash_even_after_price_change(self):
        day = "20260724"
        h = LookupHistory()
        p = SimpleNamespace(id="x", stream="S2", continuity_blocked=None, actual_ab=100.0,
            quote_ab=115.0, hedged_ns=open_ns("20260723")+400*SECOND, quote_second=300,
            quote_spread_bp=90.0, entry_day="20260723", close_day=day, close_kind="maker_exit",
            spot_buy_cash=1_000_000_000, spot_sell_cash=1_100_000_000,
            future_buy_cash=1_111_000_000, anchor=90.0)
        rows = observe_costs(h, [p], day, open_ns(day)+CLOSE_SECOND*SECOND)
        entry, exit_cost = rows
        _, actual = realized_pnl(p.spot_buy_cash, p.spot_sell_cash, 1_010_000_000,
                                p.future_buy_cash, entry_day=p.entry_day, exit_day=day)
        estimated = p.quote_ab-(p.anchor-5)-entry.bp-exit_cost.bp-34
        self.assertAlmostEqual(estimated, actual)
        self.assertAlmostEqual(exit_cost.bp, 25.0)
        self.assertEqual(entry.day, day)  # late pairing enters today's close, never yesterday's table
        self.assertFalse(h.freeze(day, 20e6, {}).decay)
        h.finish_session(day)
        self.assertTrue(h.freeze("20260727", 20e6, {}).decay)

    def test_entry_floor_does_not_double_charge_empirical_decay(self):
        day = "20260724"
        decider = EntryDecider(2_000_000_000, LookupHistory().freeze(day, 20e6, {}),
                               use_bpday=False, exec_cost=True)
        args = dict(now_ns=open_ns(day)+300*SECOND, stream="S2", quote_second=300,
                    eff_u=80., ab=100., reservation_cents=1000, committed_cents=0, expiry="20260819")
        plain = decider.decide(**args)
        small = decider.decide(**args, execution_cost_bp=10.)
        large = decider.decide(**args, execution_cost_bp=23.)
        self.assertAlmostEqual(small.est_bp, plain.est_bp)
        self.assertAlmostEqual(large.est_bp, plain.est_bp-5.)
        self.assertAlmostEqual(large.d_in, 23.)

    def test_snapshot_costs_survive_atomic_checkpoint(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            replay = build("plain", root)
            replay.snapshots = [{"day":"20260724", "decay":{"entry|S2|0|2":[31,123.456789,500.123]}}]
            checkpoint(replay)
            (root/"snapshots.json").write_text('[]')
            restored = build("plain", root)
            restore(restored)
            self.assertEqual(restored.snapshots, replay.snapshots)

    def test_resume_rejects_changed_execution_or_margin(self):
        with TemporaryDirectory() as td:
            for key in ("mode", "configurations", "days", "products"):
                manifest = dict(mode="plain", configurations=[{"decay_margin_bp":3}], days=["20260724"], products=None)
                spec = dict(manifest)
                spec[key] = "different"
                with self.assertRaises(ValueError):
                    validate_resume(manifest, spec, Path(td))


class FirstPriorityIntegrationTests(unittest.TestCase):
    def setUp(self):
        XcPortfolioTests.setUp(self)
        self.market.future_to_vc = {"XHF6":"X"}

    def make(self, **flags):
        p = LiquidityPortfolio("test", 20_000_000, use_ev=True, use_bpday=False,
            cancel_ms=50, liquidity_rule=dict(enabled=False), **flags)
        p.begin(self.market, 0, LookupHistory().freeze(self.day, 20e6, {}), lambda *e:self.tasks.append(e))
        return p

    def test_relative_cancel_and_first_priority_rules_run_together(self):
        p = self.make(repeg_drop_bp=10.)
        p.submit(intent_id="x", c=self.c, stream="S2", price=price_i(100.5), ns=self.now,
                 anchor=0., ab=50., eff_u=50.)
        self.spot_ask = 100.15
        p.book_update("S:X", self.now+1)
        self.assertEqual(p.trace[-1]["reason"], "drift")
        p.submit(intent_id="back", c=self.c, stream="S2", price=price_i(101.), ns=self.now+2,
                 anchor=0., ab=80., eff_u=80.)
        self.assertNotIn("back", p.positions)

    def test_exit_guard_does_not_require_a_working_s2_entry(self):
        p = self.make(exit_event_guard=True)
        pos = XcPortfolioTests.paired(self, p)
        self.spot_ask, self.future_ask = 100.6, 100.0
        p.second(401)
        self.assertIsNotNone(pos.exit_order)
        self.assertFalse(p.s2_live)
        self.future_ask = 101.0
        p.book_update("F:XHF6", self.start+401*SECOND+1)
        self.assertEqual(p.exit_guard_cancels, 1)


if __name__ == "__main__":
    unittest.main()
