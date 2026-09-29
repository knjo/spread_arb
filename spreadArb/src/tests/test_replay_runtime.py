"""Carry exits, independent route handoff and immutable replay provenance."""
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import polars as pl

from ..backtest.causal_market import Contract, Market, Timeline
from ..backtest.runtime import init_run, assert_sources
from ..backtest.replay import parse_args
from ..backtest.policy import PolicyConfig
from ..common.paths import SECOND, CLOSE_SECOND, MAKER_WITHDRAW_SECOND
from .test_causal_replay import C, T, DELAY, FakeMarket, engine, decision, row
from .test_points_s2 import books, prints, T0


class CarryAndRouteTest(unittest.TestCase):
    def test_existing_raw_books_still_get_missing_held_contract_exit_timeline(self):
        old = Contract("2330", "OLD", 2000, "20260715", 390000, 390000)
        current = replace(old, spot_ref=400000, fut_ref=400000)
        m = Market.__new__(Market)
        m.day, m.start, m.end = "20260706", T0, T0+CLOSE_SECOND*SECOND
        m.withdraw = T0+MAKER_WITHDRAW_SECOND*SECOND
        m.exact = {"NEW": replace(current, qc="NEW")}
        m.books = {"S:2330": books("S:2330", [(0,400000,5000,400500,20000,True)]),
                   "F:OLD": books("F:OLD", [(0,403000,3,404500,2,True)])}
        m.prints, m.timelines, m.inputs = {}, {}, []
        m.candidates = pl.DataFrame({"qc": ["NEW"]})
        with patch("spreadArb.src.backtest.causal_market.refresh_carry", return_value=([current], ["daily_metadata"])), \
             patch("spreadArb.src.backtest.causal_market.load_session") as raw, \
             patch.object(m, "load_marks", return_value={}):
            m.add_carry([old], PolicyConfig())
        raw.assert_not_called()
        self.assertEqual(m.exact["OLD"].fut_ref, 400000)
        tl = m.timelines["OLD"]
        now = T0+300*SECOND
        self.assertEqual(tl.first_exit("E2", 100., now), now)
        self.assertTrue(all(len(ix) == 0 for ix in tl.candidates.values()))
        self.assertEqual(m.candidates["qc"].to_list(), ["NEW"])
        with self.assertRaisesRegex(ValueError, "cannot create new entries"):
            tl.row("S2", 0)

    def test_fifo_handoff_waits_for_each_route_fill_or_cancel(self):
        trades = [("S:2330", T+60_000_000,505000,2000),
                  ("S:2330", 2*T+60_000_000,505000,2000),
                  ("S:2330", 4*T+60_000_000,505000,2000)]
        m = FakeMarket(trades, guard=18*T)
        a, replay = engine(m, reserve_on_submit=False, exit_routes=("E1", "E2"))
        for k in (1, 2):
            a.offer(row(ns=k*T), decision())
            replay.drain(k*T+200_000_000)
        p1, p2 = a.cycles.values()
        m.timelines[C.qc] = SimpleNamespace(first_exit=lambda route,target,ns:ns,
            guard=lambda *args:18*T, at=lambda ns:0, prices={"E1":[505000],"E2":[525000]})
        p1.target_bp = p2.target_bp = 100.
        a.exit_offers(p1, 4*T)
        a.exit_offers(p2, 4*T)
        replay.drain(4*T+60_000_001)
        self.assertEqual(a.orders[a.exit_working[(C.vc, "E1")]]["pid"], p2.id)
        self.assertEqual(a.orders[a.exit_working[(C.vc, "E2")]]["pid"], p1.id)
        self.assertEqual(a.ledger.committed_cents, 20200000)
        replay.drain(4*T+110_000_001)
        self.assertEqual(a.orders[a.exit_working[(C.vc, "E2")]]["pid"], p2.id)
        self.assertEqual(a.ledger.committed_cents, 10100000)

    def test_post_only_quote_cannot_cross_our_existing_opposite_order(self):
        a, replay = engine(FakeMarket(), reserve_on_submit=False)
        a.offer(row(), decision())
        replay.drain(T+DELAY)
        p = next(iter(a.cycles.values()))
        a.new_order(p, "E1", 505000, T+DELAY+1)
        replay.drain(T+2*DELAY+1)
        self.assertEqual(a.stats["marketable_rejects"], 1)
        self.assertEqual(len(a.queue.orders), 1)


class ReplayProvenanceTest(unittest.TestCase):
    def test_cli_defaults_to_no_reservation_and_no_prefetch(self):
        options, cfg = parse_args([])
        self.assertEqual(options["prefetch"], 0)
        self.assertFalse(cfg.reserve_on_submit)
        self.assertTrue(parse_args(["--reserve-on-submit"])[1].reserve_on_submit)
        self.assertFalse(parse_args(["--no-reserve-on-submit"])[1].reserve_on_submit)

    def test_resume_rejects_changed_source_policy_or_period(self):
        a, _ = engine(FakeMarket(), reserve_on_submit=False)
        with TemporaryDirectory() as temp:
            out = Path(temp)/"run"
            days = ["20260706", "20260707"]
            init_run(out, days, [a], False)
            init_run(out, days, [a], True)
            assert_sources(out)
            with self.assertRaisesRegex(ValueError, "date range and policy"):
                init_run(out, days[:1], [a], True)
            with patch("spreadArb.src.backtest.runtime.execution_sources", return_value={"changed": "sha"}):
                with self.assertRaisesRegex(ValueError, "source changed"):
                    init_run(out, days, [a], True)
                with self.assertRaisesRegex(RuntimeError, "source changed"):
                    assert_sources(out)
            manifest = json.loads((out/"manifest.json").read_text())
            manifest["configs"]["A"]["cap_twd"] = 1
            (out/"manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "date range and policy"):
                init_run(out, days, [a], True)


if __name__ == "__main__":
    unittest.main()
