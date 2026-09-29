"""Boundary events must not become an earlier capacity release."""
import unittest

from ...ev_lookup_cost.causal_lookup import SECOND, open_ns
from .event_capacity_timing import day_row


class CapacityTimingTests(unittest.TestCase):
    def test_10_clock_boundary_and_expiry_are_separate_from_earlier_market_release(self):
        start=open_ns('20260723');cut=start+3600*SECOND
        positions={
            'expiry':dict(state='closed',close_kind='expiry_basis_zero_accounting',entry_day='20260722'),
            'new':dict(state='paired',close_kind=None,entry_day='20260723'),
            'carry':dict(state='closed',close_kind='maker_exit',entry_day='20260722'),
            'today':dict(state='closed',close_kind='maker_exit',entry_day='20260723'),
        }
        ledger=[
            dict(id='expiry',ns=start,kind='release',delta_cents=-200_000_000),
            dict(id='new',ns=start+SECOND,kind='unreserved_fill',delta_cents=500_000_000),
            dict(id='carry',ns=cut,kind='release',delta_cents=-300_000_000),
            dict(id='today',ns=cut+SECOND,kind='release',delta_cents=-100_000_000),
        ]
        r=day_row('20260723',ledger,positions,23_000_000.)
        self.assertAlmostEqual(r['committed_at10_twd'],23_000_000.)
        self.assertAlmostEqual(r['closing_committed_twd'],22_000_000.)
        self.assertAlmostEqual(r['actual_market_release_before10_twd'],0.)
        self.assertAlmostEqual(r['actual_carry_market_release_before10_twd'],0.)
        self.assertAlmostEqual(r['expiry_accounting_release_twd'],2_000_000.)
        self.assertAlmostEqual(r['over25_seconds_before10'],3599.)
        self.assertAlmostEqual(r['over25_seconds_from10'],0.)


if __name__=='__main__':
    unittest.main()
