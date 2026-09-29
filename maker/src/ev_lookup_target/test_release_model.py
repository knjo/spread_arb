"""Release forecasts use surviving historical positions at the observed clock."""
import unittest

import polars as pl

from ..ev_lookup_cost.causal_lookup import CLOSE_SECOND, SECOND, open_ns
from ..ev_lookup_cost.forecast_calendar import known_calendar
from .release_model import SCHEMA, ReleaseSnapshot


class ReleaseTests(unittest.TestCase):
    def rows(self):
        return pl.from_dicts([dict(day='20260504',available_ns=open_ns('20260504')+CLOSE_SECOND*SECOND,
            id=str(i),stream='S2',carry=True,start_second=0.,end_second=3600. if i<60 else float(CLOSE_SECOND),
            closed=i<60,days_to_expiry=16,premium=40.,nominal_twd=200_000.) for i in range(100)],schema=SCHEMA)

    def test_past_morning_exits_are_not_still_in_late_risk_set(self):
        snapshot=ReleaseSnapshot.fit('20260505',self.rows())
        early=snapshot.probability(stream='S2',carry=True,second=300,days_to_expiry=15,premium=40.)
        late=snapshot.probability(stream='S2',carry=True,second=7200,days_to_expiry=15,premium=40.)
        self.assertAlmostEqual(early['mean'],.6)
        self.assertLess(early['discounted'],.6)
        self.assertAlmostEqual(late['discounted'],0.)

    def test_today_results_are_forbidden(self):
        with self.assertRaisesRegex(AssertionError,'unavailable'):
            ReleaseSnapshot.fit('20260504',self.rows())

    def test_credit_does_not_edit_or_release_current_positions(self):
        snapshot=ReleaseSnapshot.fit('20260505',self.rows())
        positions=[dict(id='live',nominal_twd=20_000_000.,stream='S2',entry_day='20260504',expiry='20260520',premium=40.)]
        value=snapshot.capacity(positions,second=300,calendar=known_calendar('20260505'))
        self.assertAlmostEqual(positions[0]['nominal_twd'],20_000_000.)
        self.assertGreater(value['admission_twd'],20_000_000.)
        self.assertLessEqual(value['admission_twd'],25_000_000.)
        self.assertLess(value['forecast_next_session_twd'],20_000_000.)


if __name__ == '__main__':
    unittest.main()
