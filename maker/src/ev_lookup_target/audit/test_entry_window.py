import unittest

from ...ev_lookup_cost.forecast_calendar import known_calendar
from ..q_model import QSnapshot
from ..test_q_model import stats
from .entry_window import DAILY_HURDLE_BP, ENTRY_SECONDS, sessions


class WindowClockTests(unittest.TestCase):
    def args(self,day='20260724',expiry='20260819'):
        return dict(stream='S2',quote_second=500,ab=100.,eff_u=70.,expiry=expiry,spread_bp=50.,
                    quote_notional_twd=200_000.,calendar=known_calendar(day))

    def test_one_hour_same_day_uses_one_hour_of_the_entry_budget(self):
        entry=stats(normal=100)
        entry['normal_days']=100*3600/86400
        q=QSnapshot('20260724',[],{'entry|S2':entry},{})
        result=sessions(q,**self.args())
        self.assertAlmostEqual(result,3600/ENTRY_SECONDS)
        self.assertAlmostEqual(result*DAILY_HURDLE_BP,12*3600/14000)

    def test_weekend_costs_calendar_funding_but_only_blocks_available_entry_time(self):
        carry=stats(normal=100)
        carry['normal_seconds']=100*300
        q=QSnapshot('20260724',[],{'entry|S2':stats(survive=100),'carry|S2':carry},{})
        args=self.args()
        before=q.estimate(**args)
        window=sessions(q,**args)
        after=q.estimate(**args)
        self.assertAlmostEqual(window,(14000-500+300)/14000)
        self.assertGreater(before.expected_calendar_days,2.99)
        self.assertAlmostEqual(before.net_bp,after.net_bp)
        self.assertAlmostEqual(before.funding_bp,after.funding_bp)
        self.assertLess(window*DAILY_HURDLE_BP,before.capital_hurdle_bp)

    def test_survival_occupies_every_predicted_entry_window_through_expiry(self):
        q=QSnapshot('20260724',[],{'entry|S2':stats(survive=100),'carry|S2':stats(survive=100)},{})
        # Friday remainder plus Monday and Tuesday; C8 on Wednesday's opening.
        self.assertAlmostEqual(sessions(q,**self.args(expiry='20260728')),(13500+2*14000)/14000)


if __name__=='__main__':
    unittest.main()
