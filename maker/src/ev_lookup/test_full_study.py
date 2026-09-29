"""Inventory continuity, missing marks, and full-study restart regressions."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import numpy as np

from .causal_lookup import CapacityObservation, LookupHistory, SECOND, open_ns
from .capacity_policy import activate_parked, admission_limit
from .execution import Book
from .full_study import checkpoint, restore
from .market import Contract
from .portfolio import Portfolio, Position
from .replay import Replay
from .valuation import mark_positions


class FullStudyTests(unittest.TestCase):
    def test_simultaneous_park_activation_allocates_to_earlier_quote(self):
        a=Portfolio("park",200_000,use_ev=False,use_bpday=False,park_spot=True,cancel_ms=50)
        c=Contract("3152","NBFG6",2000,"20260715",1_000_000,1_010_000)
        day="20260622"; start=open_ns(day)
        books=[Book("S:3152",start,1,((1_000_000,2000),),((1_010_000,2000),),True)]
        market=SimpleNamespace(day=day,start=start,contracts={c.vc:c},book=lambda *args:books[0])
        a.begin(market,0,LookupHistory().freeze(day,200_000,{}),lambda *args:None)
        for intent,second in [("z_earlier",1),("a_later",2)]:
            a.submit(intent_id=intent,c=c,stream="S1",price=990_000,ns=start+second*SECOND,
                     anchor=0.,ab=50.,eff_u=50.)
        books[0]=Book("S:3152",start+3*SECOND,2,((990_000,2000),),((1_000_000,2000),),True)
        activate_parked(a,start+3*SECOND)
        self.assertIn("z_earlier",a.ledger.amounts)
        self.assertNotIn("a_later",a.ledger.amounts)
        self.assertIsNotNone(a.queue.orders["a_later"].cancel_ns)

    def test_capacity_release_reissues_a_still_valid_stock_intent(self):
        a=Portfolio("park",200_000,use_ev=False,use_bpday=False,park_spot=True,cancel_ms=50)
        c=Contract("3152","NBFG6",2000,"20260715",1_000_000,1_010_000)
        day="20260622"; start=open_ns(day)
        books=[Book("S:3152",start,1,((1_000_000,2000),),((1_010_000,2000),),True)]
        signals={c.vc:dict(valid=np.ones(15601,dtype=bool),an=np.zeros(15601),fb=np.full(15601,1_020_000))}
        market=SimpleNamespace(day=day,start=start,end=start+15600*SECOND,contracts={c.vc:c},
                               book=lambda *args:books[0],signals=signals)
        a.begin(market,0,LookupHistory().freeze(day,200_000,{}),lambda *args:None)
        a.ledger.reserve("other",19_000_000,start)
        row=dict(ValueCode=c.vc,QuoteCode=c.qc,raw_order_fact_id="original",target_price=99.,
                 nominal_stop_time_ns=start+100*SECOND)
        a.offer_s1(row,start+SECOND)
        books[0]=Book("S:3152",start+2*SECOND,2,((990_000,2000),),((1_000_000,2000),),True)
        activate_parked(a,start+2*SECOND)
        a.cancel("original",start+2*SECOND+50_000_000)
        a.ledger.release("other",start+3*SECOND,terminal=True)
        a.second(3)
        current=a.s1_attempts["original"]
        self.assertIn("/retry/",current)
        self.assertIn(current,a.queue.orders)
        self.assertEqual(a.ledger.committed_cents,19_800_000)
        a.stop_s1("original",start+4*SECOND)
        self.assertNotIn("original",a.desired_s1)
        self.assertEqual(a.queue.orders[current].cancel_ns,start+4*SECOND+50_000_000)

    def test_parked_jump_fill_is_kept_even_if_capacity_is_exceeded(self):
        a = Portfolio("park", 200_000, use_ev=False, use_bpday=False, park_spot=True, cancel_ms=50)
        c = Contract("3152", "NBFG6", 2000, "20260715", 1_000_000, 1_010_000)
        day = "20260622"
        start = open_ns(day)
        book = Book("S:3152", start, 1, ((1_000_000, 2000),), ((1_010_000, 2000),), True)
        tasks = []
        market = SimpleNamespace(day=day, start=start, end=start+15600*SECOND,
                                 contracts={c.vc: c}, book=lambda *args: book)
        a.begin(market, 0, LookupHistory().freeze(day, 200_000, {}), lambda *args: tasks.append(args))
        a.ledger.reserve("other_inventory", 19_000_000, start)
        a.submit(intent_id="park", c=c, stream="S1", price=990_000, ns=start+SECOND,
                 anchor=0., ab=50., eff_u=50.)
        self.assertEqual(a.ledger.committed_cents, 19_000_000)
        self.assertIn("park", a.parked)
        # Price jumps through the resting bid before a cancellation is possible.
        a.trade("S:3152", start+2*SECOND, 3, 990_000, 2000)
        self.assertEqual(a.positions["park"].spot_buy_qty, 2000)
        self.assertEqual(a.positions["park"].hedge_kind, "entry_future")
        self.assertEqual(a.ledger.committed_cents, 38_800_000)
        self.assertEqual(a.ledger.events[-1]["kind"], "unreserved_fill")

    def test_activation_preserves_queue_and_starts_cancel_when_full(self):
        a = Portfolio("park", 200_000, use_ev=False, use_bpday=False, park_spot=True, cancel_ms=50)
        c = Contract("3152", "NBFG6", 2000, "20260715", 1_000_000, 1_010_000)
        day="20260622"; start=open_ns(day)
        books = [Book("S:3152", start, 1, ((1_000_000, 2000),), ((1_010_000, 2000),), True)]
        market = SimpleNamespace(day=day, start=start, contracts={c.vc:c}, book=lambda *args: books[0])
        a.begin(market, 0, LookupHistory().freeze(day, 200_000, {}), lambda *args: None)
        a.ledger.reserve("other",19_000_000,start)
        a.submit(intent_id="park",c=c,stream="S1",price=990_000,ns=start+SECOND,anchor=0.,ab=50.,eff_u=50.)
        books[0]=Book("S:3152",start+2*SECOND,2,((990_000,2000),),((1_000_000,2000),),True)
        activate_parked(a,start+2*SECOND,"S:3152")
        self.assertEqual(a.queue.orders["park"].cancel_ns,start+2*SECOND+50_000_000)
        self.assertIn("park",a.queue.orders)
        self.assertNotIn("park",a.ledger.amounts)

    def test_adaptive_headroom_uses_only_prior_observed_survival(self):
        a,p=self.actor()
        a.overnight_target_cents=20_000_000
        a.ledger.cap_cents=30_000_000
        day="20260623"; start=open_ns(day)
        h=LookupHistory(); h.finish_session("20260622")
        h.capacity_observations=[CapacityObservation("20260622","S2",True,0.,5000.,True,start-1)
                                 for _ in range(50)]
        market=SimpleNamespace(day=day,start=start,contracts={p.contract.vc:p.contract})
        a.begin(market,1,h.freeze(day,200_000,{}),lambda *args:None)
        self.assertEqual(admission_limit(a,start+3600*SECOND),30_000_000)
        self.assertEqual(admission_limit(a,start+6000*SECOND),20_000_000)
        h.capacity_observations.append(CapacityObservation(day,"S2",True,0.,14000.,True,start+15600*SECOND))
        self.assertEqual(len(a.decider.snapshot.capacity_observations),50)

    def actor(self):
        a = Portfolio("ev_bpday_20M", 20_000_000, use_ev=True, use_bpday=True)
        c = Contract("3152", "NBFG6", 2000, "20260715", 1_000_000, 1_010_000)
        p = Position("carry", c, "S2", open_ns("20260622"), 300, 100., 0., "20260622", 0,
                     state="paired", entry_fill_ns=open_ns("20260622") + 1,
                     hedged_ns=open_ns("20260622") + 2,
                     spot_buy_qty=2000, spot_buy_cash=2_000_000_000,
                     future_sell_qty=1, future_sell_cash=2_020_000_000)
        a.positions[p.id] = p
        a.active.add(p.id)
        a.ledger.reserve(p.id, 20_000_000, p.quote_ns)
        return a, p

    def test_missing_identity_cannot_release_at_expiry_or_bind_reused_code(self):
        a, p = self.actor()
        h = LookupHistory()
        for day, missing in [("20260623", {p.contract.qc}), ("20260716", set())]:
            market = SimpleNamespace(day=day, start=open_ns(day), missing_carry=missing,
                                     contracts={p.contract.vc: p.contract}, exact={p.contract.qc: p.contract})
            a.begin(market, len(h.sessions), h.freeze(day, 20e6, {}), lambda *args: None)
            h.finish_session(day)
            self.assertEqual(a.ledger.committed_cents, 20_000_000)
            self.assertEqual(p.state, "paired")
            self.assertIsNone(p.close_day)
        self.assertEqual(p.continuity_blocked, "20260623")

    def test_end_mark_uses_two_liquidation_sides_and_missing_is_not_zero(self):
        a, p = self.actor()
        a.day = "20260623"
        end = open_ns(a.day) + 15_600_000_000_000
        books = {"S:3152": Book("S:3152", end, 1, ((1_020_000, 2000),), ((1_030_000, 2000),), True),
                 "F:NBFG6": Book("F:NBFG6", end, 2, ((1_020_000, 1),), ((1_025_000, 1),), True)}
        a.market = SimpleNamespace(end=end, book=lambda instrument, now: books.get(instrument))
        a.daily = [{"realized_twd": 123.}]
        marks = mark_positions(a)
        self.assertAlmostEqual(marks["equity_twd"], 443.)
        books.clear()
        marks = mark_positions(a)
        self.assertIsNone(marks["equity_twd"])
        self.assertEqual(marks["unmarked_positions"], 1)

    def test_checkpoint_keeps_carry_and_previous_day_lookup(self):
        a, p = self.actor()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            replay = Replay(root, [a])
            replay.history.finish_session("20260622")
            a.daily = [{"day": "20260622", "realized_twd": 0.}]
            (root / "snapshots.json").write_text(json.dumps([]))
            checkpoint(replay)
            b = Portfolio(a.name, 20_000_000, use_ev=True, use_bpday=True)
            resumed = Replay(root, [b])
            restore(resumed)
            self.assertEqual(b.positions[p.id], p)
            self.assertEqual(b.active, {p.id})
            self.assertEqual(b.ledger.amounts, a.ledger.amounts)
            self.assertEqual(resumed.history.sessions, ["20260622"])


if __name__ == "__main__":
    unittest.main()
