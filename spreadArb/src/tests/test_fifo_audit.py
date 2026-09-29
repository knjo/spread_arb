"""Raw queue negative controls independent of the execution engine."""
from copy import deepcopy
import unittest

from ..backtest.fifo_audit import audit_fifo
from ..common.paths import SECOND
from .test_points_s2 import books, prints, T0


class RawQueueAuditTest(unittest.TestCase):
    def fixture(self):
        pid, oid = "S1/20260706/2330/1", "order"
        events = [dict(kind="submit", id=pid, order_id=oid, route="S1", price=505000, qty=2000, ns=T0+SECOND),
                  dict(kind="live", id=pid, order_id=oid, ahead=1000, ns=T0+1_050_000_000),
                  dict(kind="cancel_effective", id=pid, order_id=oid, ns=T0+2*SECOND)]
        cash = [dict(order_id=oid, ns=T0+1_060_000_000, sequence=1, qty=1000, liquidity="maker")]
        b = books("S:2330", [(0,505000,1000,510000,10000,True)])
        pr = prints("S:2330", [(1.01,505000,10000),(1.06,505000,2000),(2.01,505000,2000)])
        return dict(events=events, cash=cash, positions={pid: {"qc": "FUT6"}}, mapping={"2330": "FUT6"},
                    books={"S:2330": b}, prints={"S:2330": pr})

    def test_raw_ahead_and_only_live_volume_support_partial_fill(self):
        counts, failures = audit_fifo(**self.fixture())
        self.assertEqual(failures, [])
        self.assertEqual(counts["raw_queue_fills_checked"], 1)

    def test_claiming_full_fill_with_insufficient_live_print_volume_fails(self):
        data = self.fixture()
        data["cash"][0]["qty"] = 2000
        checks = {f["check"] for f in audit_fifo(**data)[1]}
        self.assertIn("maker_fill_not_supported_by_raw_queue", checks)

    def test_omitting_actual_fill_fails(self):
        data = self.fixture()
        data["cash"] = []
        self.assertIn("raw_queue_fill_missing_from_replay", {f["check"] for f in audit_fifo(**data)[1]})

    def test_invented_ahead_and_fill_after_cancel_fail(self):
        data = self.fixture()
        data["events"][1]["ahead"] = 0
        extra = deepcopy(data["cash"][0])
        extra.update(ns=T0+2_010_000_000, sequence=2)
        data["cash"].append(extra)
        checks = {f["check"] for f in audit_fifo(**data)[1]}
        self.assertIn("raw_queue_ahead", checks)
        self.assertIn("maker_fill_not_supported_by_raw_queue", checks)


if __name__ == "__main__":
    unittest.main()
