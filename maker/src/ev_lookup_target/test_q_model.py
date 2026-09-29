"""Competing-risk and calendar-duration invariants for the new Q estimates."""
from dataclasses import replace
import unittest

from ..ev_lookup_cost.forecast_calendar import known_calendar
from .q_model import QSnapshot, empty_stats
from .target import ReturnTarget


def stats(normal=0, survive=0, rollback=0, other=0):
    row = empty_stats()
    row.update(n=normal+survive+rollback+other, normal=normal, survive=survive,
               rollback=rollback, other=other, normal_seconds=normal*7200,
               rollback_seconds=rollback*7200, other_seconds=other*7200,
               normal_days=normal*.02, rollback_days=rollback*.02, other_days=other*.02,
               rollback_net=-50.*rollback, other_net=-100.*other)
    return row


class QModelTests(unittest.TestCase):
    def estimate(self, snapshot, **kwargs):
        args = dict(stream='S1', quote_second=3600, ab=90., eff_u=60., expiry='20260520',
                    spread_bp=40., quote_notional_twd=200_000., calendar=known_calendar(snapshot.day))
        args.update(kwargs)
        return snapshot.estimate(**args)

    def snapshot(self, entry, carry=None):
        return QSnapshot('20260504', ['20260501'], {'entry|S1': entry, 'carry|S1': carry or stats(survive=100)}, {})

    def test_all_surviving_carry_goes_to_true_expiry_and_pays_calendar_cost(self):
        q = self.estimate(self.snapshot(stats(survive=100)))
        self.assertAlmostEqual(q.p_sd, 0.)
        self.assertAlmostEqual(q.p_expiry, 1.)
        self.assertGreater(q.expected_calendar_days, 16.)
        self.assertAlmostEqual(q.before_funding_bp, 90-18-34)
        self.assertAlmostEqual(q.funding_bp, q.expected_calendar_days*200/365)

    def test_rollback_never_gets_normal_exit_profit(self):
        q = self.estimate(self.snapshot(stats(rollback=100)))
        self.assertAlmostEqual(q.p_rollback, 1.)
        self.assertAlmostEqual(q.p_sd, 0.)
        self.assertAlmostEqual(q.before_funding_bp, -50.)

    def test_normal_overnight_exit_is_not_basis_zero_settlement(self):
        q = self.estimate(self.snapshot(stats(survive=100), stats(normal=100)))
        self.assertAlmostEqual(q.p_overnight, 1.)
        self.assertAlmostEqual(q.p_expiry, 0.)
        self.assertAlmostEqual(q.before_funding_bp, 60+5-18-8-34)

    def test_adverse_floor_is_max_with_entry_cost_not_a_second_charge(self):
        source = self.snapshot(stats(normal=100))
        low, high = self.estimate(source, execution_cost_bp=10.), self.estimate(source, execution_cost_bp=30.)
        self.assertAlmostEqual(low.d_in, 18.)
        self.assertAlmostEqual(high.d_in, 30.)
        self.assertAlmostEqual(low.before_funding_bp-high.before_funding_bp, 12.)

    def test_target_changes_hurdle_without_inventing_profit_or_future_releases(self):
        source = self.snapshot(stats(normal=30, survive=60, rollback=10), stats(normal=30, survive=60, other=10))
        low = self.estimate(source, target=ReturnTarget(annual_net_return=.10))
        high = self.estimate(source, target=ReturnTarget(annual_net_return=.30))
        self.assertAlmostEqual(low.net_bp, high.net_bp)
        self.assertAlmostEqual(low.expected_calendar_days, high.expected_calendar_days)
        self.assertGreater(low.surplus_bp, high.surplus_bp)
        self.assertAlmostEqual(sum((high.p_sd,high.p_overnight,high.p_expiry,high.p_rollback,high.p_other)), 1.)

    def test_known_calendar_does_not_anticipate_typhoon_closure(self):
        self.assertTrue(known_calendar('20260708')['20260710'])
        self.assertFalse(known_calendar('20260710')['20260710'])


if __name__ == '__main__':
    unittest.main()
