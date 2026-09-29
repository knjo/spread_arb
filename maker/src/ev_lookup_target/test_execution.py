"""Unreserved live orders retain cancel races and only reuse released capacity."""
from dataclasses import replace
import unittest

from ..ev_lookup_cost import test_liquidity as fixtures
from ..ev_lookup_cost.causal_lookup import LookupHistory
from ..ev_lookup_cost.execution import price_i
from .portfolio import TargetPortfolio


class TargetExecutionTests(unittest.TestCase):
    def setUp(self):
        fixtures.LiquidityOrderTests.setUp(self)
        self.p=TargetPortfolio('target',450_000,use_ev=False,use_bpday=False,cancel_ms=50,hedge_ms=50,
            reserve_quotes=False,persistent_wait=False,
            liquidity_rule=dict(min_a1_multiple=5,adverse_probability=.5))
        self.p.begin(self.market,0,LookupHistory().freeze(self.day,450_000,{}),lambda *e:self.tasks.append(e))

    def submit(self,pid='x',contract=None,ns=None):
        self.p.submit(intent_id=pid,c=contract or self.c,stream='S2',price=price_i(100.5),ns=ns or self.now,
                      anchor=0.,ab=50.,eff_u=50.)

    def test_quote_does_not_consume_capital_and_maintenance_handles_unreserved_order(self):
        self.submit()
        self.assertEqual(self.p.ledger.committed_cents,0)
        self.assertIn('x',self.p.unreserved)
        self.p.book_update('F:XHF6',self.now+1)
        self.assertIsNone(self.p.queue.orders['x'].cancel_ns)

    def test_first_fill_takes_capacity_and_later_cancel_race_is_retained(self):
        self.p.ledger.reserve('carry',19_000_000,self.now-1)
        other=replace(self.c,vc='Y',qc='YHF6')
        self.market.future_to_vc['YHF6']='Y'
        self.submit('x')
        self.submit('y',other,self.now+1)
        first=self.now+1_000_000
        self.p.trade('F:XHF6',first,1,price_i(100.5),1)
        self.assertEqual(self.p.ledger.committed_cents,41_000_000)
        self.assertEqual(self.p.queue.orders['y'].cancel_ns,first+50_000_000)
        self.p.trade('F:YHF6',first+10_000_000,2,price_i(100.5),1)
        self.assertEqual(self.p.positions['y'].future_sell_qty,1)
        self.assertEqual(self.p.ledger.committed_cents,63_000_000)
        self.assertTrue(any(t['kind']=='unreserved_fill' and t['overrun_cents']>0 for t in self.p.trace))
        self.assertEqual(self.p.positions['y'].hedge_kind,'entry_spot')

    def test_unreserved_spot_fill_books_its_buy_limit_not_future_upper_bound(self):
        self.p.submit(intent_id='stock',c=self.c,stream='S1',price=price_i(99.),ns=self.now,
                      anchor=0.,ab=50.,eff_u=50.)
        self.assertEqual(self.p.ledger.committed_cents,0)
        self.p.trade('S:X',self.now+1,1,price_i(99.),2000)
        self.assertEqual(self.p.ledger.amounts['stock'],19_800_000)
        self.assertEqual(self.p.positions['stock'].hedge_kind,'entry_future')

    def test_full_book_keeps_intent_waiting_and_release_creates_fresh_queue(self):
        self.p.persistent_wait=True
        self.p.ledger.reserve('carry',45_000_000,self.now-1)
        self.p._requote_s2('X',self.now)
        self.assertIn('X',self.p.waiting_s2)
        self.assertFalse(self.p.queue.orders)
        later=self.now+100_000_000
        self.p.ledger.release('carry',later,terminal=True)
        self.p.retry_waiting(later)
        pid=self.p.s2_live['X']
        self.assertEqual(self.p.queue.orders[pid].placed_ns,later)
        # A print already observed at the release timestamp cannot fill the new order.
        self.p.trade('F:XHF6',later,1,price_i(100.5),1)
        self.assertIsNone(self.p.positions[pid].entry_fill_ns)
        self.p.trade('F:XHF6',later+1,2,price_i(100.5),1)
        self.assertEqual(self.p.positions[pid].entry_fill_ns,later+1)

    def test_depth_or_ev_loss_never_reprices_future_behind_first_priority(self):
        self.spot_ask=100.4
        self.p._requote_s2('X',self.now)
        self.assertFalse(self.p.queue.orders)


if __name__ == '__main__':
    unittest.main()
