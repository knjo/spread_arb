from __future__ import annotations

import unittest

import polars as pl

from maker.src.quote_fill.partial import summarize_spot_partial_completion


class PartialSummaryTest(unittest.TestCase):
    def test_horizon_denominator_keeps_early_cancel_as_competing_failure(self) -> None:
        frame = pl.DataFrame(
            {
                "route": ["spot_bid_future_taker"] * 3,
                "boundary_quantile": [50, 50, 50],
                "raw_order_fact_id": ["a", "b", "c"],
                "any_fill": [True, True, True],
                "full_fill": [True, True, False],
                "partial_fill": [False, False, True],
                "first_fill_recv_time_ns": [0, 0, 0],
                "full_fill_recv_time_ns": [40_000_000, 2_000_000_000, None],
                "nominal_stop_recv_time_ns": [100_000_000, 3_000_000_000, 25_000_000],
            }
        )
        result = summarize_spot_partial_completion(frame, horizons_ms=(50, 1000))
        h50 = result.filter(pl.col("horizon_ms") == 50).row(0, named=True)
        self.assertEqual(h50["complete_two_lots"], 1)
        self.assertEqual(h50["any_fill_orders"], 3)
        self.assertEqual(h50["p_complete_two_lots_given_any_fill"], 1 / 3)
        h1000 = result.filter(pl.col("horizon_ms") == 1000).row(0, named=True)
        self.assertEqual(h1000["complete_two_lots"], 1)
        self.assertEqual(h1000["any_fill_orders"], 3)
        self.assertEqual(h1000["p_complete_two_lots_given_any_fill"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
