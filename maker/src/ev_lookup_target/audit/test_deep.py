import unittest

from ...ev_lookup_cost import test_liquidity as fixtures
from ...ev_lookup_cost.causal_lookup import LookupHistory
from ...ev_lookup_cost.execution import price_i
from .deep_policy import DeepSpotPortfolio


class DeepQueueTests(unittest.TestCase):
    def setUp(self):
        fixtures.LiquidityOrderTests.setUp(self)
        self.p=DeepSpotPortfolio('deep',450_000,use_ev=False,use_bpday=False,cancel_ms=50,hedge_ms=50,
            reserve_quotes=False,persistent_wait=False,deep_shared=True,
            liquidity_rule=dict(min_a1_multiple=5,adverse_probability=.5))
        self.p.begin(self.market,0,LookupHistory().freeze(self.day,450_000,{}),lambda *e:self.tasks.append(e))
        self.p.ledger.reserve('carry',45_000_000,self.now-1)

    def quote(self):
        self.p.submit(intent_id='stock',c=self.c,stream='S1',price=price_i(99.),ns=self.now,
                      anchor=0.,ab=50.,eff_u=50.)

    def test_full_capacity_keeps_deep_order_without_reserving(self):
        self.quote()
        self.assertIn('stock',self.p.queue.orders)
        self.assertFalse(self.p.decisions[-1]['slot_free'])
        self.assertTrue(self.p.decisions[-1]['admit'])
        self.assertEqual(self.p.ledger.committed_cents,45_000_000)
        self.assertFalse(self.p.parked)
        self.p._enforce_capacity(self.now+1)
        self.assertIsNone(self.p.queue.orders['stock'].cancel_ns)

    def test_top_approach_cancels_and_late_fill_is_kept(self):
        self.quote()
        self.spot_ask=99.5
        now=self.now+1_000_000
        self.p.book_update('S:X',now)
        self.assertEqual(self.p.queue.orders['stock'].cancel_ns,now+50_000_000)
        self.p.trade('S:X',now+10_000_000,1,price_i(99.),2000)
        self.assertEqual(self.p.ledger.committed_cents,64_800_000)
        self.assertEqual(self.p.positions['stock'].hedge_kind,'entry_future')

    def test_natural_release_preserves_original_queue_timestamp(self):
        self.quote()
        later=self.now+1_000_000
        self.p.ledger.release('carry',later,terminal=True)
        self.spot_ask=99.5
        self.p.book_update('S:X',later)
        self.assertIsNone(self.p.queue.orders['stock'].cancel_ns)
        self.assertEqual(self.p.queue.orders['stock'].placed_ns,self.now)
        self.p.trade('S:X',later+1,1,price_i(99.),2000)
        self.assertEqual(self.p.positions['stock'].spot_buy_qty,2000)

    def test_direct_jump_cannot_discard_fill_for_lack_of_capacity(self):
        self.quote()
        self.p.trade('S:X',self.now+1,1,price_i(98.5),2000)
        self.assertEqual(self.p.positions['stock'].spot_buy_qty,2000)
        self.assertGreater(self.p.ledger.committed_cents,self.p.ledger.cap_cents)

    def test_s2_still_needs_observable_room(self):
        self.p.submit(intent_id='future',c=self.c,stream='S2',price=price_i(100.5),ns=self.now,
                      anchor=0.,ab=50.,eff_u=50.)
        self.assertNotIn('future',self.p.queue.orders)
        self.assertEqual(self.p.decisions[-1]['reason'],'cap')

    def test_actual_overrun_cancels_even_other_deep_orders_without_erasing_races(self):
        self.quote()
        self.p.submit(intent_id='lower',c=self.c,stream='S1',price=price_i(98.5),ns=self.now,
                      anchor=0.,ab=50.,eff_u=50.)
        now=self.now+1_000_000
        self.p.trade('S:X',now,1,price_i(99.),2000)
        self.assertEqual(self.p.queue.orders['lower'].cancel_ns,now+50_000_000)
        self.p.trade('S:X',now+1,2,price_i(98.5),2000)
        self.assertEqual(self.p.positions['lower'].spot_buy_qty,2000)
        self.assertTrue(any(t.get('capacity_emergency') for t in self.p.trace))

    def test_deep_parking_does_not_bypass_the_value_gate(self):
        self.p.decider.use_ev=True
        self.p.submit(intent_id='bad_value',c=self.c,stream='S1',price=price_i(99.),ns=self.now,
                      anchor=200.,ab=50.,eff_u=-150.)
        self.assertNotIn('bad_value',self.p.queue.orders)
        self.assertEqual(self.p.decisions[-1]['reason'],'ev')

    def test_actual_overrun_pauses_new_deep_quotes(self):
        self.p.ledger.book_unreserved_fill('prior_race',5_000_000,self.now)
        self.quote()
        self.assertNotIn('stock',self.p.queue.orders)
        self.assertEqual(self.p.decisions[-1]['reason'],'cap')


if __name__=='__main__':
    unittest.main()
