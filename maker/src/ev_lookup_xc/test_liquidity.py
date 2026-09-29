"""Liquidity is known at quote time; quantity changes cancel, races remain."""
from dataclasses import replace
import unittest

from .causal_lookup import LookupHistory
from .decide import EntryDecider
from .execution import Book, TakerDepth, price_i
from .liquidity_portfolio import LiquidityPortfolio
from .s2_liquidity import LiquidityRule, hedge_risk
from .test_ev_exit_events import EventPortfolioTests


class DepthTests(unittest.TestCase):
    def setUp(self):
        self.depth = TakerDepth()
        self.book = Book("S:X", 100, 1, ((price_i(99), 10000),),
                         ((price_i(100), 1000), (price_i(100.5), 3000)), True)

    def test_depth_preview_and_probability_use_hedge_quantity(self):
        r = hedge_risk(self.book, 2000, price_i(101), 100, LiquidityRule(), self.depth)
        self.assertAlmostEqual(r["a1_multiple"], .5)
        self.assertAlmostEqual(r["depth_vwap"], 100.25)
        self.assertAlmostEqual(r["adverse_vwap"], 100.75)
        raw = (101/100-1)*10000
        expected = .5*(101/100.25-1)*10000+.5*(101/100.75-1)*10000
        self.assertAlmostEqual(r["execution_cost_bp"], raw-expected)
        self.assertFalse(self.depth.used)

    def test_already_spent_depth_cannot_pass_quote_gate(self):
        self.depth.take(self.book, "buy", 3500, 100)
        r = hedge_risk(self.book, 2000, price_i(101), 100, LiquidityRule(), self.depth)
        self.assertEqual(r["reason"], "hedge_depth")

    def test_five_times_uses_shares_not_futures_lots(self):
        b = replace(self.book, asks=((price_i(100), 9999),))
        rule = LiquidityRule(min_a1_multiple=5, adverse_probability=0)
        self.assertEqual(hedge_risk(b, 2000, price_i(101), 100, rule, self.depth)["reason"], "a1_depth")
        b = replace(b, asks=((price_i(100), 10000),))
        self.assertEqual(hedge_risk(b, 2000, price_i(101), 100, rule, self.depth)["reason"], "ok")

    def test_tick_tier_transition_and_future_book_rejection(self):
        b = replace(self.book, bids=((price_i(99.8), 10000),), asks=((price_i(99.9), 2000),))
        r = hedge_risk(b, 2000, price_i(101), 100, LiquidityRule(adverse_ticks=2), self.depth)
        self.assertAlmostEqual(r["adverse_vwap"], 100.5)
        with self.assertRaises(ValueError):
            hedge_risk(b, 2000, price_i(101), 99, LiquidityRule(), self.depth)


class LiquidityOrderTests(unittest.TestCase):
    def setUp(self):
        EventPortfolioTests.setUp(self)
        self.p = LiquidityPortfolio("test", 20_000_000, use_ev=True, use_bpday=False,
            cancel_ms=50, hedge_ms=50,
            liquidity_rule=dict(min_a1_multiple=5, adverse_probability=.5))
        self.p.begin(self.market, 0, LookupHistory().freeze(self.day, 20e6, {}),
                     lambda *event:self.tasks.append(event))
        self.market.future_to_vc = {"XHF6":"X"}
        self.original_book = self.market.book
        self.shown = 10000
        def book(instrument, ns):
            b = self.original_book(instrument, ns)
            return replace(b, asks=((price_i(self.spot_ask), self.shown),)) if instrument.startswith("S:") else b
        self.market.book = book
        self.market.pair = lambda c, ns, signal_buffer=True:(book("S:X", ns), book("F:XHF6", ns))

    def quote(self):
        self.p.submit(intent_id="x", c=self.c, stream="S2", price=price_i(100.5), ns=self.now,
                      anchor=0., ab=50., eff_u=50.)

    def test_quantity_only_fade_sends_delayed_cancel(self):
        self.quote()
        self.shown = 9999
        invalid = self.now+100_000_000
        self.p.book_update("S:X", invalid)
        self.assertEqual(self.p.queue.orders["x"].cancel_ns, invalid+50_000_000)
        self.assertEqual(self.p.trace[-1]["reason"], "a1_depth")

    def test_depth_cancel_race_must_hedge_actual_price(self):
        self.quote()
        self.shown = 9999
        invalid = self.now+100_000_000
        self.p.book_update("S:X", invalid)
        fill = invalid+1_000_000
        self.p.trade("F:XHF6", fill, 1, price_i(100.5), 1)
        self.spot_ask = 101.0
        self.p.cancel("x", invalid+50_000_000)
        self.p.hedge("x", fill+50_000_000)
        self.assertEqual(self.p.positions["x"].state, "paired")
        self.assertLess(self.p.positions["x"].actual_ab, 0.)
        self.assertEqual(self.p.positions["x"].spot_buy_cash, price_i(101)*2000)
        self.assertGreater(self.p.ledger.committed_cents, 0)

    def test_no_back_of_book_fallback(self):
        self.spot_ask = 100.4
        self.p._requote_s2("X", self.now)
        self.assertFalse(self.p.queue.orders)

    def test_price_priority_loss_cancels(self):
        self.quote()
        self.future_ask = 100.0
        self.p.book_update("F:XHF6", self.now+1)
        self.assertEqual(self.p.trace[-1]["reason"], "first_priority")

    def test_later_external_ask_at_same_price_does_not_lose_fifo_priority(self):
        self.quote()
        self.future_ask = 100.5
        self.p.book_update("F:XHF6", self.now+1)
        self.assertIsNone(self.p.queue.orders["x"].cancel_ns)

    def test_cost_changes_admission_without_changing_hazard_or_basis_cell(self):
        decider = EntryDecider(2_000_000_000, self.p.decider.snapshot, use_bpday=False)
        args = dict(now_ns=self.now, stream="S2", quote_second=300, eff_u=25., ab=25.,
                    reservation_cents=1_000_000, committed_cents=0, expiry=self.c.expiry)
        before = decider.decide(**args)
        after = decider.decide(**args, execution_cost_bp=before.est_bp+1.)
        self.assertTrue(before.admit)
        self.assertEqual(after.reason, "ev")
        self.assertAlmostEqual(after.est_bp, -1.)
        self.assertEqual(before.cell, after.cell)
        self.assertAlmostEqual(before.p_expiry, after.p_expiry)


if __name__ == "__main__":
    unittest.main()
