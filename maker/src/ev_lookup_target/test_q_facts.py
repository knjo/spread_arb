"""Causal observations, right-censored carry, and return-unit regressions."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import polars as pl

from ..ev_lookup_cost.causal_lookup import CLOSE_SECOND, SECOND, open_ns
from ..ev_lookup_cost.market import Contract
from ..ev_lookup_cost.portfolio import Position
from .q_facts import RISK_SCHEMA, position_facts, training_window
from .target import ReturnTarget


class QFactsTests(unittest.TestCase):
    def fixture(self, entry_day='20260504'):
        start = open_ns(entry_day)
        contract = Contract('X', 'XFE6', 2000, '20260520', 1_000_000, 1_010_000)
        p = Position('sample', contract, 'S1', start+300*SECOND, 300, 60., 25., entry_day, 0)
        p.entry_fill_ns = start+301*SECOND
        p.hedged_ns = start+302*SECOND
        p.state = 'paired'
        p.spot_buy_qty, p.spot_buy_cash = 2000, 2_000_000_000
        p.future_sell_qty, p.future_sell_cash = 1, 2_008_000_000
        p.actual_ab = 40.
        decisions = pl.DataFrame([dict(intent_id=p.id, ns=p.quote_ns, reservation_cents=20_000_000, execution_cost_bp=0.)])
        return p, decisions

    def frames(self, day, p, decisions, **kwargs):
        return position_facts(day, pl.from_dicts([asdict(p)]), decisions, **kwargs)

    def test_unresolved_carry_stays_in_denominator_without_future_pnl(self):
        p, decisions = self.fixture()
        quotes, risk, costs = self.frames('20260505', p, decisions)
        self.assertEqual(quotes.height, 0)
        self.assertEqual(risk['event'].to_list(), ['survive'])
        self.assertEqual(risk['phase'].to_list(), ['carry'])
        self.assertIsNone(risk['outcome_net_bp'][0])
        self.assertEqual(costs.height, 0)

    def test_rollback_is_not_a_normal_same_day_exit(self):
        p, decisions = self.fixture()
        p.hedged_ns, p.actual_ab = None, None
        p.future_sell_qty, p.future_sell_cash = 0, 0
        p.spot_buy_qty, p.spot_buy_cash = 100, 100_000_000
        p.close_ns, p.close_day = open_ns(p.entry_day)+400*SECOND, p.entry_day
        p.state, p.close_kind, p.pnl_twd = 'closed', 'partial_entry_rollback', -40.
        _, risk, costs = self.frames(p.entry_day, p, decisions)
        self.assertEqual(risk['event'].to_list(), ['rollback'])
        self.assertAlmostEqual(risk['outcome_net_bp_on_quote_capital'][0], -2.)
        self.assertEqual(costs.height, 0)

    def test_outage_keeps_accounting_but_does_not_train_zero_exit(self):
        p, decisions = self.fixture()
        _, risk, costs = self.frames('20260505', p, decisions, outage=True)
        self.assertEqual(risk.height, 0)
        self.assertEqual(costs.height, 0)

    def test_future_close_in_daily_snapshot_is_rejected(self):
        p, decisions = self.fixture()
        p.close_ns, p.close_day = open_ns('20260505'), '20260505'
        with self.assertRaisesRegex(AssertionError, 'future close_ns'):
            self.frames('20260504', p, decisions)

    def test_asof_training_never_reads_today_or_future_partition(self):
        p, decisions = self.fixture()
        _, risk, _ = self.frames('20260504', p, decisions)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'Date=20260504').mkdir()
            risk.write_parquet(root/'Date=20260504/risk.parquet')
            # Today/future are deliberately absent: reading either is a failure.
            (root/'manifest.json').write_text(json.dumps(dict(status='completed', days=['20260504','20260505','20260506'])))
            read = training_window(root, '20260505')
            self.assertEqual(read.height, 1)
            self.assertTrue(read['available_ns'][0] < open_ns('20260505'))
            self.assertTrue(training_window(root, '20260504').is_empty())

    def test_today_available_timestamp_is_rejected_in_prior_partition(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'Date=20260504').mkdir()
            p, decisions = self.fixture()
            _, risk, _ = self.frames('20260504', p, decisions)
            risk.with_columns(pl.lit(open_ns('20260505')).alias('available_ns')).write_parquet(root/'Date=20260504/risk.parquet')
            (root/'manifest.json').write_text(json.dumps(dict(status='completed', days=['20260504'])))
            with self.assertRaisesRegex(AssertionError, 'unavailable'):
                training_window(root, '20260505')

    def test_target_uses_total_capital_and_does_not_double_count_funding(self):
        target = ReturnTarget()
        self.assertAlmostEqual(target.annual_net_twd, 6_000_000.)
        self.assertAlmostEqual(target.daily_net_twd, 24_000.)
        self.assertAlmostEqual(target.per_trade_twd(20), 1200.)
        self.assertAlmostEqual(target.capital_day_hurdle_bp(), 3000/365)
        self.assertAlmostEqual(target.capital_day_hurdle_bp(.5), 6000/365)
        with self.assertRaises(ValueError):
            target.capital_day_hurdle_bp(0)


if __name__ == '__main__':
    unittest.main()
