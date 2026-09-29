import unittest
from dataclasses import replace
from unittest.mock import patch

from ...ev_lookup_cost import test_liquidity as fixtures
from ...ev_lookup_cost.causal_lookup import LookupHistory
from ...ev_lookup_cost.execution import price_i
from .event_s2_policy import EventS2Portfolio


class EventS2Tests(unittest.TestCase):
    def setUp(self):
        fixtures.LiquidityOrderTests.setUp(self)
        self.shown=20000
        self.p=EventS2Portfolio('event',20_000_000,use_ev=False,use_bpday=False,
            event_s2=True,event_quotes=True,reserve_quotes=False,deep_shared=True,
            liquidity_rule=dict(min_a1_multiple=5,adverse_probability=.5),cancel_ms=50,hedge_ms=50)
        self.p.begin(self.market,0,LookupHistory().freeze(self.day,20_000_000,{}),lambda *e:self.tasks.append(e))

    def fill(self):
        self.p.book_update('F:XHF6',self.now)
        pid=self.p.s2_live['X'];fill=self.now+100_000_000
        self.p.trade('F:XHF6',fill,1,price_i(100.5),1)
        return pid,fill

    def test_immediate_requote_after_hedge_without_waiting_for_a_new_book_or_second(self):
        pid,fill=self.fill()
        self.p.book_update('S:X',fill+1)
        self.assertNotIn('X',self.p.s2_live)
        ready=fill+50_000_000
        self.p.hedge(pid,ready)
        new=self.p.queue.orders[self.p.s2_live['X']]
        self.assertEqual(new.placed_ns,ready)
        self.assertNotEqual(new.position_id,pid)
        self.assertEqual(self.p.positions[pid].state,'paired')
        self.assertEqual(self.p.ledger.amounts[pid],20_000_000)
        self.p.trade('F:XHF6',ready,2,price_i(100.5),1)
        self.assertIsNone(self.p.positions[new.position_id].entry_fill_ns)

    def test_real_capacity_still_blocks_requote_after_hedge(self):
        self.p.ledger.reserve('carry',1_965_000_000,self.now-1)
        pid,fill=self.fill()
        self.p.hedge(pid,fill+50_000_000)
        self.assertNotIn('X',self.p.s2_live)
        self.assertEqual(self.p.decisions[-1]['reason'],'cap')
        self.assertEqual(self.p.positions[pid].state,'paired')

    def test_depth_spent_by_previous_hedge_is_unavailable_for_new_quote(self):
        self.shown=10000
        pid,fill=self.fill()
        self.p.hedge(pid,fill+50_000_000)
        self.assertNotIn('X',self.p.s2_live)
        self.assertEqual(self.p.trace[-1]['kind'],'s2_liquidity_reject')

    def test_pending_hedge_is_rebuilt_at_the_next_begin(self):
        pid,_=self.fill()
        self.p.entry_hedges.clear()
        self.p.begin(self.market,1,LookupHistory().freeze(self.day,20_000_000,{}),lambda *e:self.tasks.append(e))
        self.assertEqual(self.p.entry_hedges['X'],pid)
        self.p.book_update('F:XHF6',self.now+200_000_000)
        self.assertNotIn('X',self.p.s2_live)

    def test_event_entry_still_honors_the_value_gate(self):
        original=self.p.decider.decide
        with patch.object(self.p.decider,'decide',side_effect=lambda **kw:
                          replace(original(**kw),admit=False,reason='q_value')):
            self.p.book_update('F:XHF6',self.now)
        self.assertFalse(self.p.queue.orders)
        self.assertEqual(self.p.decisions[-1]['reason'],'q_value')

    def test_event_adapter_disabled_keeps_the_shadow_idle_and_its_cooldown(self):
        self.p.event_s2=False
        self.p.book_update('F:XHF6',self.now)
        self.assertNotIn('X',self.p.s2_live)
        self.p.submit(intent_id='legacy',c=self.c,stream='S2',price=price_i(100.5),ns=self.now,
                      anchor=0.,ab=50.,eff_u=50.)
        fill=self.now+100_000_000
        self.p.trade('F:XHF6',fill,1,price_i(100.5),1)
        self.assertEqual(self.p.cooldown['X'],fill+60_000_000_000)


if __name__=='__main__':
    unittest.main()
