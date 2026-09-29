"""User-confirmed product/route limits, independent of held position count."""
from collections import Counter
from types import SimpleNamespace
import unittest

from ..backtest.policy import PolicyConfig, PRESETS
from .test_causal_replay import C, DELAY, T, FakeMarket, decision, engine, row


class QuoteConcurrencyTest(unittest.TestCase):
    def test_both_presets_allow_unlimited_position_count(self):
        for name, preset in PRESETS.items():
            with self.subTest(preset=name):
                self.assertIsNone(PolicyConfig(**preset).max_positions_per_product)

    def test_s1_and_s2_can_rest_together_but_each_rejects_a_duplicate(self):
        a, replay = engine(FakeMarket(), reserve_on_submit=False)
        for stream in ("S1", "S2"):
            self.assertEqual(a.offer(row(stream), decision()), "submitted")
            self.assertEqual(a.offer(row(stream), decision()), "busy")
        replay.drain(T + DELAY)
        self.assertEqual(len(a.queue.orders), 2)
        self.assertEqual(a.ledger.committed_cents, 0)

    def test_cancel_pending_keeps_only_its_own_entry_slot(self):
        a, replay = engine(FakeMarket(), reserve_on_submit=False)
        a.offer(row(), decision())
        replay.drain(T + DELAY)
        oid = next(iter(a.orders))
        a.request_cancel(oid, T + 60_000_000)
        self.assertEqual(a.offer(row(ns=T + 70_000_000), decision()), "busy")
        self.assertEqual(a.offer(row("S2", T + 70_000_000), decision()), "submitted")
        replay.drain(T + 110_000_000)
        self.assertEqual(a.offer(row(ns=T + 110_000_001), decision()), "submitted")

    def test_unfinished_hedge_blocks_same_stream_only(self):
        market = FakeMarket([("S:2330", T + 60_000_000, 505000, 2000)], fut_qty=0)
        a, replay = engine(market, reserve_on_submit=False)
        a.offer(row(), decision())
        replay.drain(T + 200_000_000)
        self.assertEqual(next(iter(a.cycles.values())).state, "entry_hedge")
        self.assertEqual(a.offer(row(ns=2*T), decision()), "busy")
        self.assertEqual(a.offer(row("S2", 2*T), decision()), "submitted")

    def test_many_positions_can_coexist_with_all_four_quote_routes(self):
        trades = [("S:2330", k*T + 60_000_000, 505000, 2000) for k in range(1, 5)]
        for name, preset in PRESETS.items():
            with self.subTest(preset=name):
                market = FakeMarket(trades, guard=18*T)
                a, replay = engine(market, reserve_on_submit=False, **preset)
                # Four real fills/hedges accumulate four held positions.
                for k in range(1, 5):
                    self.assertEqual(a.offer(row(ns=k*T), decision()), "submitted")
                    replay.drain(k*T + 200_000_000)
                held = list(a.cycles.values())
                self.assertEqual([p.state for p in held], ["paired"] * 4)
                market.timelines[C.qc] = SimpleNamespace(
                    first_exit=lambda route, target, ns: ns,
                    guard=lambda *args: 18*T,
                    at=lambda ns: 0, prices={"E1": [510000], "E2": [520000]})
                for p in held:
                    p.target_bp = 100.0
                for p in held:
                    a.exit_offers(p, 5*T)
                replay.drain(5*T)
                for stream in ("S1", "S2"):
                    self.assertEqual(a.offer(row(stream, 5*T), decision()), "submitted")
                replay.drain(5*T + DELAY)
                live = [o for o in a.orders.values() if not o["done"]]
                self.assertEqual(Counter(o["route"] for o in live),
                                 Counter({"S1": 1, "S2": 1, "E1": 1, "E2": 1}))
                self.assertEqual(len(a.queue.orders), 4)
                exits = [o for o in live if o["route"] in ("E1", "E2")]
                self.assertEqual({o["pid"] for o in exits}, {held[0].id})
                self.assertEqual(sum(p.state == "paired" for p in a.cycles.values()), 4)
                self.assertEqual(a.ledger.committed_cents, 4 * 10100000)
                for route, price in (("E1", 510000), ("E2", 520000)):
                    with self.assertRaisesRegex(AssertionError, "multiple working exits"):
                        a.new_order(held[1], route, price, 6*T)


if __name__ == "__main__":
    unittest.main()
