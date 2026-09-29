"""Days with only settlement events must still pass through full accounting."""
import unittest

import polars as pl

from ..backtest.causal_audit import entry_submissions


class AuditEventSchemaTest(unittest.TestCase):
    def test_settlement_only_day_has_no_submission_route(self):
        events = pl.DataFrame([
            dict(ns=1, kind="execution", id="old", purpose="settlement"),
            dict(ns=1, kind="closed", id="old", purpose=None)])
        self.assertEqual(list(entry_submissions(events)), [])

    def test_submission_without_route_is_not_silently_ignored(self):
        with self.assertRaises(KeyError):
            list(entry_submissions(pl.DataFrame([dict(ns=1, kind="submit", id="bad")])))

    def test_both_entry_routes_remain_audited(self):
        events = pl.DataFrame([dict(ns=i, kind="submit", id=r, route=r)
                               for i, r in enumerate(("S1", "S2", "E1", "E2"))])
        self.assertEqual([e["id"] for e in entry_submissions(events)], ["S1", "S2"])


if __name__ == "__main__":
    unittest.main()
