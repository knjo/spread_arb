import unittest

from ...ev_lookup_cost import test_liquidity as fixtures
from ...ev_lookup_cost.causal_lookup import LookupHistory
from ...ev_lookup_cost.execution import price_i
from ..portfolio import TargetPortfolio
from .s2_supply import SupplyPortfolio


class SupplyChronologyTests(unittest.TestCase):
    def setUp(self):
        fixtures.LiquidityOrderTests.setUp(self)

    def actor(self,event=True,cooldown=0,residual=25):
        p=SupplyPortfolio(name='supply',event_entry=event,cooldown_seconds=cooldown,
                          depth_multiple=5,residual_bp=residual)
        p.begin(self.market,0,LookupHistory().freeze(self.day,10**12,{}),lambda *e:self.tasks.append(e))
        return p

    def test_idle_event_can_quote_between_seconds_without_a_shadow_intent(self):
        p=self.actor();other=self.actor(event=False)
        ns=self.now+100_000_000
        p.book_update('F:XHF6',ns);other.book_update('F:XHF6',ns)
        self.assertEqual(len(p.queue.orders),1)
        self.assertFalse(other.queue.orders)
        order=next(iter(p.queue.orders.values()))
        self.assertEqual(order.placed_ns,ns)
        self.assertEqual(order.price,price_i(100.5))

    def test_existing_persistent_wait_still_misses_an_idle_book_opportunity(self):
        original=TargetPortfolio('original',20_000_000,use_ev=False,use_bpday=False,event_quotes=True,
            persistent_wait=True,liquidity_rule=dict(min_a1_multiple=5,adverse_probability=.5))
        original.begin(self.market,0,LookupHistory().freeze(self.day,20_000_000,{}),lambda *e:self.tasks.append(e))
        original.waiting_s2['X']=self.now
        event=self.now+100_000_000
        original.book_update('F:XHF6',event)
        candidate=self.actor();candidate.book_update('F:XHF6',event)
        self.assertFalse(original.queue.orders)
        self.assertEqual(len(candidate.queue.orders),1)

    def test_zero_cooldown_waits_for_actual_hedge_completion(self):
        p=self.actor();p._requote_s2('X',self.now)
        pid=p.s2_live['X'];fill=self.now+100_000_000
        p.trade('F:XHF6',fill,1,price_i(100.5),1)
        p.book_update('S:X',fill+1)
        self.assertNotIn('X',p.s2_live)
        p.hedge(pid,fill+50_000_000)
        self.shown=20000
        p.book_update('S:X',fill+60_000_000)
        self.assertIn('X',p.s2_live)

    def test_cancel_race_retains_negative_basis_and_required_stock_purchase(self):
        p=self.actor();p._requote_s2('X',self.now)
        pid=p.s2_live['X'];invalid=self.now+100_000_000
        self.shown=9999;p.book_update('S:X',invalid)
        p.trade('F:XHF6',invalid+1,1,price_i(100.5),1)
        self.spot_ask=101.
        p.cancel(pid,invalid+50_000_000)
        p.hedge(pid,invalid+50_000_001)
        self.assertLess(p.positions[pid].actual_ab,0.)
        self.assertEqual(p.positions[pid].spot_buy_qty,2000)
        self.assertGreater(p.ledger.committed_cents,0)

    def test_residual_sensitivity_does_not_back_off_first_priority(self):
        self.spot_ask=100.4
        normal=self.actor();loose=self.actor(residual=0)
        normal._requote_s2('X',self.now);loose._requote_s2('X',self.now)
        self.assertFalse(normal.queue.orders)
        self.assertEqual(next(iter(loose.queue.orders.values())).price,price_i(100.5))

    def test_same_timestamp_print_cannot_fill_a_new_quote(self):
        p=self.actor();p._requote_s2('X',self.now)
        pid=p.s2_live['X']
        p.trade('F:XHF6',self.now,1,price_i(100.5),1)
        self.assertIsNone(p.positions[pid].entry_fill_ns)


if __name__=='__main__':
    unittest.main()
