"""One-minute route refresh, cancel races, and independent negative controls."""
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ..backtest.policy import PolicyConfig, StreamingDecider
from ..backtest.refresh_audit import audit_refresh, audit_quote_prices, audit_decision_inputs
from ..backtest.runtime import require_qcache
from ..common.paths import QUOTE_END_SECOND
from .test_causal_replay import C, T, DELAY, FakeMarket, engine, decision, row
from .test_ev import FakeLookup
from .test_points_s2 import books


class RefreshTest(unittest.TestCase):
    def setup_replay(self, trades=(), **cfg):
        market = FakeMarket(trades, guard=179*T)
        market.end, market.withdraw = 180*T, 179*T
        a, replay = engine(market, reserve_on_submit=False, **cfg)
        tl = market.timelines[C.qc]
        tl.entry_row_at = lambda stream, ns: {**row(stream, ns), "price": 502000 if stream == "S1" else 524000}
        replay.refresh_decider = SimpleNamespace(decide=lambda r: decision())
        return a, replay, market

    def test_routes_expire_independently_and_reprice_after_ack(self):
        a, replay, _ = self.setup_replay()
        a.offer(row("S1", T), decision())
        a.offer(row("S2", 2*T), decision())
        first = next(iter(a.orders))
        deadline = T+DELAY+60*T
        replay.drain(deadline)
        self.assertEqual(a.orders[first]["cancel_at"], deadline)
        self.assertFalse(a.orders[first]["done"])
        self.assertEqual(a.stats["entry_refresh_submitted"], 0)
        replay.drain(deadline+DELAY)
        self.assertTrue(a.orders[first]["done"])
        replay.drain(deadline+DELAY+1)
        replacement = list(a.orders.values())[-1]
        self.assertEqual(replacement["route"], "S1")
        self.assertEqual(replacement["price"], 502000)
        self.assertEqual(replacement["send_ns"], deadline+DELAY+1)
        self.assertEqual(replacement["live_ns"], deadline+2*DELAY+1)
        self.assertEqual(a.stats["quote_expirations"], 1)
        self.assertEqual(replay.decisions[0]["origin"], "refresh")
        replay.drain(2*T+DELAY+60*T)
        self.assertEqual(a.stats["quote_expirations"], 2)

    def test_fill_during_expiry_cancel_remains_real_and_no_replacement(self):
        deadline = T+DELAY+60*T
        a, replay, _ = self.setup_replay([("S:2330", deadline+1, 505000, 2000)])
        a.offer(row(), decision())
        replay.drain(deadline+2*DELAY)
        p = next(iter(a.cycles.values()))
        self.assertEqual(p.state, "paired")
        self.assertEqual(a.ledger.committed_cents, 10100000)
        self.assertEqual(a.stats["entry_refresh_submitted"], 0)
        self.assertEqual(a.stats["cancel_race_fills"], 1)

    def test_partial_remains_funded_until_rollback_then_refresh(self):
        deadline = T+DELAY+60*T
        a, replay, _ = self.setup_replay([("S:2330", 2*T, 505000, 1000)])
        a.offer(row(), decision())
        replay.drain(deadline+DELAY)
        self.assertEqual(a.ledger.committed_cents, 5050000)
        self.assertEqual(a.stats["entry_refresh_submitted"], 0)
        replay.drain(deadline+2*DELAY)
        self.assertEqual(a.ledger.committed_cents, 0)
        self.assertEqual(a.stats["entry_refresh_submitted"], 0)
        replay.drain(deadline+2*DELAY+1)
        self.assertEqual(a.stats["entry_refresh_submitted"], 1)
        checks, errors = audit_refresh(a.trace, a.cash, asdict(a.cfg), deadline+2*DELAY+1)
        self.assertFalse(errors)
        self.assertEqual(checks["entry_replacements_checked"], 1)

    def test_already_pending_cancel_keeps_original_ack(self):
        deadline = T+DELAY+60*T
        a, replay, _ = self.setup_replay()
        a.offer(row(), decision())
        replay.drain(deadline-10)
        oid = next(iter(a.orders))
        a.request_cancel(oid, deadline-10)
        replay.drain(deadline+DELAY-10)
        self.assertTrue(a.orders[oid]["done"])
        self.assertEqual(a.orders[oid]["cancel_at"], deadline-10)

    def test_stale_timer_does_not_cancel_new_route_order(self):
        a, replay, _ = self.setup_replay()
        a.offer(row(), decision())
        replay.drain(2*T)
        a.request_cancel(next(iter(a.orders)), 2*T)
        replay.drain(3*T)
        a.offer(row(ns=3*T), decision())
        replay.drain(T+DELAY+60*T)
        self.assertEqual(a.stats["quote_expirations"], 0)
        self.assertIsNone(list(a.orders.values())[-1]["cancel_at"])

    def test_refresh_rechecks_current_capacity_hurdle_and_market(self):
        for rejection in ("cap", "hurdle", "market_ineligible"):
            with self.subTest(rejection=rejection):
                a, replay, market = self.setup_replay(cap_twd=110000, product_cap_frac=1)
                a.offer(row(), decision())
                replay.drain(60*T)
                if rejection == "cap":
                    a.ledger.exposure("unrelated", 10000000, 60*T, "fixture")
                elif rejection == "hurdle":
                    a.hurdle = 1000.
                else:
                    market.timelines[C.qc].entry_row_at = lambda *args: None
                replay.drain(62*T)
                self.assertEqual(a.stats["entry_refresh_submitted"], 0)
                self.assertEqual(a.trace[-1]["outcome"], rejection)

    def test_exit_routes_refresh_independently_using_latest_price(self):
        a, replay, market = self.setup_replay([("S:2330", T+DELAY+1,505000,2000)],
                                             exit_routes=("E1", "E2"))
        a.offer(row(), decision())
        replay.drain(2*T)
        p = next(iter(a.cycles.values()))
        p.target_bp = 100.
        market.timelines[C.qc] = SimpleNamespace(first_exit=lambda route,target,ns:ns,
            guard=lambda *args:179*T, at=lambda ns:0,
            prices={"E1":[515000],"E2":[525000]})
        a.exit_offers(p, 3*T)
        replay.drain(4*T)
        market.timelines[C.qc].prices = {"E1":[516000],"E2":[524000]}
        replay.drain(64*T)
        latest = {o["route"]:o for o in a.orders.values() if not o["done"]}
        self.assertEqual(set(latest), {"E1", "E2"})
        self.assertEqual(latest["E1"]["price"], 516000)
        self.assertEqual(latest["E2"]["price"], 524000)
        self.assertEqual(a.stats["quote_expirations"], 2)
        self.assertEqual(a.ledger.committed_cents, 10100000)

    def test_timer_uses_asof_prices_but_current_decision_time_and_cutoff(self):
        from .test_causal_replay import TimelineRegressionTest
        tl, start, second = TimelineRegressionTest().timeline()
        now = start+650*second+123
        r = tl.entry_row_at("S2", now)
        self.assertEqual(r["quote_ns"], now)
        self.assertEqual(r["quote_second"], 650)
        self.assertEqual(r["price"], int(tl.prices["S2"][tl.at(now)]))
        self.assertIsNone(tl.entry_row_at("S2", start+QUOTE_END_SECOND*second))

    def test_refresh_evaluations_do_not_enter_dynamic_market_population(self):
        market = StreamingDecider("20260706", FakeLookup(), None, PolicyConfig())
        refresh = StreamingDecider("20260706", FakeLookup(), None, PolicyConfig())
        market.decide(row())
        scores = market.admitted_scores()
        refresh.decide({**row(), "quote_second": 1900})
        self.assertEqual(market.admitted_scores(), scores)
        self.assertEqual(len(refresh.admitted_scores()), 1)


class IndependentRefreshTest(unittest.TestCase):
    def input_fixture(self):
        from .test_points_s2 import T0
        data = {"S:2330": books("S:2330", [(0,490000,1000,491000,1000,True),
                                              (60,500000,1000,501000,1000,True)]),
                "F:FUT6": books("F:FUT6", [(0,505000,10,507000,10,True),
                                             (60,515000,10,517000,10,True)])}
        r = dict(vc="2330", qc="FUT6", ns=T0+60*T+1, quote_second=60, stream="S1", price=500000,
                 spot_a1=501000, quote_ab=300., tick_bp_hedge=1000/515000*10000,
                 execution_floor_bp=0., scale=20., anchor=30., resid_mid_bp=20., e_norm=1.)
        return r, data, {("2330","FUT6",60):(30.,50.)}, T0

    def test_raw_decision_input_check_uses_current_books_and_grid(self):
        r, data, grid, start = self.input_fixture()
        checked, errors = audit_decision_inputs([r], data, grid, start)
        self.assertEqual(checked["raw_decision_inputs_checked"], 1)
        self.assertFalse(errors)
        s2 = {**r,"stream":"S2","price":516000,"quote_ab":(516000/501000-1)*10000,
              "tick_bp_hedge":1000/501000*10000,"execution_floor_bp":1000/501000*5000}
        self.assertFalse(audit_decision_inputs([s2], data, grid, start)[1])

    def test_raw_decision_input_check_detects_stale_ev_fields(self):
        r, data, grid, start = self.input_fixture()
        for field, value in (("quote_ab",400.),("anchor",25.),("spot_a1",491000),("quote_second",59)):
            with self.subTest(field=field):
                errors = audit_decision_inputs([{**r,field:value}], data, grid, start)[1]
                self.assertTrue(any(e.get("field") == field for e in errors))

    def test_absolute_quote_may_have_no_normalized_residual(self):
        r, data, grid, start = self.input_fixture()
        r.update(resid_mid_bp=float("nan"),e_norm=float("nan"),scale=None)
        grid[("2330","FUT6",60)] = (30.,None)
        self.assertFalse(audit_decision_inputs([r],data,grid,start)[1])
        errors = audit_decision_inputs([{**r,"e_norm":1.}],data,grid,start)[1]
        self.assertTrue(any(e.get("field") == "e_norm" for e in errors))

    def trace(self):
        return [dict(kind="submit", ns=T, id="p", order_id="o", route="S1", qty=2000),
                dict(kind="live", ns=T+DELAY, id="p", order_id="o"),
                dict(kind="cancel_request", ns=61*T+DELAY, id="p", order_id="o"),
                dict(kind="cancel_effective", ns=61*T+2*DELAY, id="p", order_id="o")]

    def test_audit_detects_missing_deadline_cancel(self):
        cfg = asdict(PolicyConfig())
        self.assertFalse(audit_refresh(self.trace(), [], cfg, 180*T)[1])
        checks, errors = audit_refresh(self.trace()[:2], [], cfg, 180*T)
        self.assertEqual(checks["quote_lifetimes_checked"], 1)
        self.assertIn("missing_quote_deadline_cancel", {e["check"] for e in errors})

    def test_audit_detects_replacement_before_old_entry_flat(self):
        events = self.trace()+[dict(kind="submit",ns=62*T,id="new",order_id="neworder",route="S1",
                                   qty=2000,replaces_order_id="o")]
        errors = audit_refresh(events, [], asdict(PolicyConfig()), 180*T)[1]
        self.assertIn("refresh_before_old_entry_flat", {e["check"] for e in errors})

    def test_audit_detects_fill_after_deadline_and_wrong_expiry(self):
        events = self.trace()+[dict(kind="quote_expired",ns=62*T,id="p",order_id="o",deadline_ns=62*T)]
        cash = [dict(liquidity="maker",order_id="o",ns=62*T,qty=2000)]
        errors = audit_refresh(events, cash, asdict(PolicyConfig()), 180*T)[1]
        self.assertIn("fill_after_quote_deadline", {e["check"] for e in errors})
        self.assertIn("wrong_quote_deadline", {e["check"] for e in errors})

    def test_raw_price_audit_rejects_stale_price(self):
        b = books("S:2330", [(0,500000,1000,501000,1000,True),
                             (60,502000,1000,503000,1000,True)])
        e = dict(kind="submit", ns=int(b.ns[-1])+1, order_id="o", route="S1", price=500000)
        checked, errors = audit_quote_prices([e], {"o":{"instrument":"S:2330"}}, {"S:2330":b})
        self.assertEqual(checked["raw_quote_prices_checked"], 1)
        self.assertEqual(errors[0]["check"], "quote_not_current_price")
        self.assertFalse(audit_quote_prices([{**e,"price":502000}],
                         {"o":{"instrument":"S:2330"}}, {"S:2330":b})[1])

    def test_cold_cache_missing_prior_day_fails(self):
        with TemporaryDirectory() as temp, \
             patch("spreadArb.src.backtest.runtime.grid_days", return_value=["20260126","20260127"]), \
             patch("spreadArb.src.backtest.runtime.hist_path", return_value=Path(temp)/"hist"), \
             patch("spreadArb.src.backtest.runtime.facts_path", return_value=Path(temp)/"facts"):
            self.assertEqual(require_qcache(["20260126"]), {})
            with self.assertRaisesRegex(ValueError, "incomplete Q cache"):
                require_qcache(["20260127"])

    def test_invalid_ttl_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            replace(PolicyConfig(), quote_refresh_ns=0)


if __name__ == "__main__":
    unittest.main()
