"""Causal exit survival, event cancellation races, and unchanged hard capacity."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

from .causal_lookup import CapacityObservation, CLOSE_SECOND, SECOND, LookupHistory, open_ns
from .exit_model import forecast_exits, policy_ev
from .forecast_calendar import known_calendar
from .execution import Book, price_i
from . import test_ev_exit_events as fixtures


class ExitHazardTests(unittest.TestCase):
    def test_typhoon_closure_unavailable_before_announcement(self):
        before = known_calendar("20260618")
        self.assertFalse(before["20260619"])
        self.assertTrue(before["20260710"])
        self.assertTrue(known_calendar("20260709")["20260710"])
        self.assertFalse(known_calendar("20260710")["20260710"])
        f = forecast_exits("20260709", "20260715", "S2", .2, {}, known_calendar("20260709"))
        self.assertEqual(f.remaining_sessions, 4)

    def test_unclosed_carry_stays_in_denominator_and_other_is_not_expiry(self):
        h = LookupHistory()
        day = "20260723"
        for i in range(40):
            kind = "maker_exit" if i < 20 else "corporate_close" if i < 30 else None
            h.capacity_observations.append(CapacityObservation(day, "S2", True, 0., 1000.,
                kind is not None, open_ns(day)+CLOSE_SECOND*SECOND, "20260727", kind, -10.))
        self.assertFalse(h.freeze(day, 20e6, {}).exit_hazards)
        h.finish_session(day)
        snap = h.freeze("20260724", 20e6, {})
        self.assertEqual(snap.exit_hazards["S2"][:3], (40, 20, 10))
        f = forecast_exits("20260724", "20260727", "S2", 0., snap.exit_hazards, {})
        self.assertAlmostEqual(f.p_overnight, .5)
        self.assertAlmostEqual(f.p_other, .25)
        self.assertAlmostEqual(f.p_expiry, .25)
        self.assertAlmostEqual(f.other_net_bp, -10.)
        self.assertAlmostEqual(policy_ev(30., 80., f), .5*1+.25*46-.25*10)

    def test_expiry_day_keeps_zero_basis_branch_and_calendar_holidays(self):
        f = forecast_exits("20260724", "20260724", "S2", .2, {}, {})
        self.assertAlmostEqual(f.p_expiry, .8)
        self.assertAlmostEqual(f.p_overnight, 0.)
        self.assertAlmostEqual(policy_ev(30., 80., f), .2*15+.8*46)
        f = forecast_exits("20260724", "20260727", "S2", .2, {}, {"20260727": False})
        self.assertEqual(f.remaining_sessions, 0)

    def test_future_observation_cannot_change_frozen_probabilities(self):
        h = LookupHistory()
        h.finish_session("20260723")
        before = h.freeze("20260724", 20e6, {})
        for i in range(40):
            h.capacity_observations.append(CapacityObservation("20260723", "S2", True, 0., 1000.,
                True, open_ns("20260724"), "20260727", "maker_exit", 10.))
        self.assertEqual(h.freeze("20260724", 20e6, {}).exit_hazards, before.exit_hazards)


class EventRaceTests(unittest.TestCase):
    setUp = fixtures.EventPortfolioTests.setUp

    def test_requote_outside_visible_depth_cannot_assume_zero_queue(self):
        self.spot_ask = 101.0
        self.p._requote_s2("X", self.now)
        self.assertFalse(self.p.queue.orders)
    def test_cancel_with_100ms_head_start_prevents_original_fill(self):
        self.p.submit(intent_id="x", c=self.c, stream="S2", price=price_i(100.5), ns=self.now,
                      anchor=0., ab=50., eff_u=50.)
        self.spot_ask = 100.4
        invalid = self.now+100_000_000
        self.p.book_update("S:X", invalid)
        self.p.cancel("x", invalid+50_000_000)
        self.p.trade("F:XHF6", invalid+100_000_000, 1, price_i(100.5), 1)
        self.assertIsNone(self.p.positions["x"].entry_fill_ns)
        self.assertEqual(self.p.ledger.committed_cents, 0)

    def test_one_ms_race_stays_booked_and_requires_hedge(self):
        self.p.submit(intent_id="x", c=self.c, stream="S2", price=price_i(100.5), ns=self.now,
                      anchor=0., ab=50., eff_u=50.)
        self.spot_ask = 100.4
        invalid = self.now+100_000_000
        self.p.book_update("S:X", invalid)
        self.p.trade("F:XHF6", invalid+1_000_000, 1, price_i(100.5), 1)
        self.p.cancel("x", invalid+50_000_000)
        pos = self.p.positions["x"]
        self.assertEqual(pos.hedge_kind, "entry_spot")
        self.assertGreater(self.p.ledger.committed_cents, 0)
        event = next(t for t in self.p.trace if t["kind"] == "s2_invalid_fill")
        self.assertFalse(event["cancel_could_arrive"])

    def test_legacy_audit_does_not_retroactively_cancel(self):
        self.p.event_quotes = False
        self.p.submit(intent_id="x", c=self.c, stream="S2", price=price_i(100.5), ns=self.now,
                      anchor=0., ab=50., eff_u=50.)
        self.spot_ask = 100.4
        self.p.book_update("S:X", self.now+1)
        self.assertIsNone(self.p.queue.orders["x"].cancel_ns)
        self.p.trade("F:XHF6", self.now+100_000_000, 1, price_i(100.5), 1)
        self.assertTrue(next(t for t in self.p.trace if t["kind"] == "s2_invalid_fill")["cancel_could_arrive"])

    def test_invalid_future_update_cancels_spot_hedge_dependent_quote(self):
        self.p.submit(intent_id="x", c=self.c, stream="S2", price=price_i(100.5), ns=self.now,
                      anchor=0., ab=50., eff_u=50.)
        self.market.future_to_vc = {"XHF6": "X"}
        self.market.pair = lambda *a, **k: None
        self.p.book_update("F:XHF6", self.now+1)
        self.assertIsNotNone(self.p.queue.orders["x"].cancel_ns)


if __name__ == "__main__":
    unittest.main()
