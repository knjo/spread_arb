"""User's exact TWD 19.8M/20M scenarios, funded by actual simulated fills."""
from dataclasses import asdict
import unittest

from maker.src.ev_lookup_cost.execution import Book
from ..backtest.causal_market import Contract
from ..backtest.unreserved_audit import UnreservedAudit
from .test_causal_replay import T, DELAY, FakeMarket, engine, decision, row
from .test_unreserved_audit import facts


def product(market, vc, price=1_000_000):
    c = Contract(vc, "F"+vc, 2000, "20260715", price, price*102//100)
    market.contracts[vc] = c
    market.timelines[c.qc] = next(iter(market.timelines.values()))
    market.limits[vc] = (price*90//100, price*110//100)
    market.b["S:"+vc] = Book("S:"+vc, 0, 1,
        ((price*99//100, 1_000_000),), ((price*101//100, 1_000_000),), True)
    market.b["F:"+c.qc] = Book("F:"+c.qc, 0, 1,
        ((price*102//100, 1000),), ((price*103//100, 1000),), True)
    return c


def quote(c, ns, price=None):
    price = c.spot_ref if price is None else price
    return {**row(ns=ns), "vc":c.vc, "qc":c.qc, "price":price,
            "spot_a1":c.spot_ref*101//100, "quote_second":ns//T}


def funded_scenario():
    trades = [("S:"+str(8000+i%5), i*T+60_000_000, 1_000_000, 2000) for i in range(1,100)]
    trades.append(("S:9000",101*T,1_000_000,2000))
    market = FakeMarket(trades, guard=499*T)
    market.end, market.withdraw = 500*T, 499*T
    contracts = {str(8000+i):product(market,str(8000+i)) for i in range(5)}
    pending = [product(market,str(9000+i)) for i in range(25)]
    large = product(market,"9998",price=25_000_000)  # exactly TWD 5M per pair
    too_large = product(market,"9999")
    actor, replay = engine(market,reserve_on_submit=False,cap_twd=20_000_000)
    for i in range(1,100):
        if actor.offer(quote(contracts[str(8000+i%5)],i*T),decision()) != "submitted":
            raise AssertionError("could not build actual funded inventory")
        replay.drain(i*T+2*DELAY+10_000_000)
    return actor,replay,pending,large,too_large


class UserCapacityScenarioTest(unittest.TestCase):
    def test_1980w_can_quote_500w_but_no_individual_above_20w(self):
        a,replay,pending,_,too_large = funded_scenario()
        self.assertEqual(sum(p.state == "paired" for p in a.cycles.values()),99)
        self.assertEqual(a.ledger.committed_cents,1_980_000_000)
        for c in pending:
            self.assertEqual(a.offer(quote(c,100*T),decision()),"submitted")
        replay.drain(100*T+DELAY)
        working = [o for o in a.orders.values() if not o["done"]]
        self.assertEqual(len(working),25)
        self.assertEqual(sum(o["price"]*o["qty"]//100 for o in working),500_000_000)
        self.assertTrue(all(o["price"]*o["qty"]//100 <= 20_000_000 for o in working))
        self.assertEqual(a.ledger.committed_cents,1_980_000_000)
        self.assertEqual(a.offer(quote(too_large,100*T+DELAY+1,price=1_005_000),decision()),"cap")
        self.assertFalse(UnreservedAudit(asdict(a.cfg)).check_day(**facts(a)))

    def test_full_2000w_cannot_spend_future_500w_exit(self):
        a,replay,pending,large,_ = funded_scenario()
        for c in pending:
            a.offer(quote(c,100*T),decision())
        replay.drain(101*T)
        self.assertEqual(a.ledger.committed_cents,2_000_000_000)
        remainder = [o for o in a.orders.values() if not o["done"]]
        self.assertEqual(len(remainder),24)
        self.assertTrue(all(o["cancel_at"] == 101*T for o in remainder))
        paired = [p for p in a.cycles.values() if p.state == "paired"][:25]
        self.assertEqual(len(paired),25)
        for p in paired:
            a.force_close(p,110*T,"scheduled_fixture_exit")
        self.assertEqual(a.offer(quote(large,101*T+1),decision()),"cap")
        replay.drain(110*T-1)
        self.assertEqual(a.ledger.committed_cents,2_000_000_000)
        self.assertEqual(a.offer(quote(large,110*T-1),decision()),"cap")
        replay.drain(110*T)
        self.assertEqual(a.ledger.committed_cents,1_500_000_000)
        self.assertEqual(a.offer(quote(large,110*T+1),decision()),"submitted")
        self.assertEqual(a.ledger.committed_cents,1_500_000_000)
        self.assertFalse(UnreservedAudit(asdict(a.cfg)).check_day(**facts(a)))


if __name__ == "__main__":
    unittest.main()
