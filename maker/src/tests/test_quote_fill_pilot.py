"""Tests for the sparse WP02 base-intent loader."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import unittest

import polars as pl

from ..common.paths import HFT_DATA_ROOT, MAKER_ROOT
from ..quote_fill.pilot import (
    DEFAULT_FAIR_PANEL_PATH,
    DEFAULT_SNAPSHOT_PATH,
    build_base_epoch_intents,
    build_base_epoch_intents_from_frames,
)


BANNED_FORWARD_LABELS = {
    "Close",
    "FutureHigh",
    "FutureLow",
    "SpreadNarrowOrderTime",
    "SpreadNarrowSide",
    "TakerSell_CloseBP",
    "TakerBuy_CloseBP",
    "midEdge_60sBP",
    "midEdge_300sBP",
    "midEdge_1800sBP",
    "future_center_bp",
    "future_center_coverage",
    "future_30s_bp",
    "future_30s_coverage",
    "future_60s_bp",
    "future_60s_coverage",
    "future_180s_bp",
    "future_180s_coverage",
    "future_300s_bp",
    "future_300s_coverage",
}


def _dt(hour: int, minute: int, second: int) -> datetime:
    return datetime(2026, 1, 28, hour, minute, second)


class BaseEpochIntentSyntheticTest(unittest.TestCase):
    def test_only_epoch_transitions_expand_to_two_quantiles_and_routes(self) -> None:
        spot = pl.DataFrame(
            {
                "Date": ["20260128"] * 4,
                "ValueCode": ["2303"] * 4,
                "spot_channel_seq": [1, 2, 3, 4],
                "event_recv_time": [
                    _dt(1, 4, 59),
                    _dt(1, 5, 1),
                    _dt(1, 5, 2),
                    _dt(1, 5, 3),
                ],
                "event_trans_time": [
                    _dt(9, 4, 59),
                    _dt(9, 5, 1),
                    _dt(9, 5, 2),
                    _dt(9, 5, 3),
                ],
                "spot_trial_match": [0] * 4,
                "spot_ref_price_current": [100.0] * 4,
                "spot_bid": [99.9] * 4,
                "spot_bid_2": [99.8] * 4,
                "spot_ask": [100.0] * 4,
                "spot_ask_2": [100.5] * 4,
                "spot_bid_lots": [100] * 4,
                "spot_ask_lots": [100] * 4,
                "spread_pair_id": [1, 2, 2, 3],
                "spread_pair_seq": [1, 1, 1, 1],
                "spread_pair_epoch": [1, 2, 2, 3],
                "spread_count_at_same_count": [1, 1, 2, 1],
            }
        )
        fair = pl.DataFrame(
            {
                "Date": ["20260128", "20260128"],
                "ValueCode": ["2303", "2303"],
                "QuoteCode": ["CCFB6", "CCFB6"],
                "fair_timestamp": [_dt(1, 5, 0), _dt(1, 5, 2)],
                "fut_exec_bid": [100.5, 100.5],
                "fut_exec_ask": [101.0, 101.0],
                "fut_bid": [100.5, 100.5],
                "fut_ask": [101.0, 101.0],
                "fut_ref_price": [100.0, 100.0],
                "analysis_eligible": [True, True],
                "fut_formal": [True, True],
                "fut_book_ok": [True, True],
                "fut_exec_book_ok": [True, True],
                "fut_ref_ok": [True, True],
                "anchor_ewma_120s_bp": [50.0, 50.0],
            }
        )
        snapshot = pl.DataFrame(
            {
                "Date": ["20260128", "20260128"],
                "ValueCode": ["2303", "2303"],
                "QuoteCode": ["CCFB6", "CCFB6"],
                "boundary_quantile": [50, 80],
                "upper_distance_bp": [20.0, 30.0],
                "adaptive_parameter_valid": [True, True],
                "source_asof_date": ["20260127", "20260127"],
            }
        )

        result = build_base_epoch_intents_from_frames(spot, fair, snapshot)

        self.assertEqual(result.height, 8)
        self.assertEqual(result["base_event_id"].n_unique(), 2)
        self.assertEqual(set(result["route"]), {
            "future_ask_spot_taker",
            "spot_bid_future_taker",
        })
        self.assertEqual(set(result["boundary_quantile"]), {50, 80})
        self.assertTrue(result["gate_open"].all())
        self.assertTrue(
            result.select(
                (pl.col("fair_timestamp") <= pl.col("event_recv_time")).all()
            ).item()
        )
        self.assertTrue(BANNED_FORWARD_LABELS.isdisjoint(result.columns))
        self.assertEqual(
            result.schema["event_recv_time"], pl.Datetime("ns")
        )
        self.assertEqual(result.schema["fair_timestamp"], pl.Datetime("ns"))


class BaseEpochIntentDataSmokeTest(unittest.TestCase):
    @unittest.skipUnless(
        (
            HFT_DATA_ROOT / "tickData" / "20260128_StockTick.parquet"
        ).exists()
        and (
            HFT_DATA_ROOT / "tickFeature" / "20260128_tickFeature.parquet"
        ).exists()
        and DEFAULT_FAIR_PANEL_PATH.exists()
        and DEFAULT_SNAPSHOT_PATH.exists(),
        "local 20260128 pilot data unavailable",
    )
    def test_one_day_four_symbol_smoke(self) -> None:
        result = build_base_epoch_intents(
            "20260128", ["2303", "2317", "2603", "2881"]
        )

        self.assertGreater(result.height, 0)
        self.assertEqual(result.height, result["base_event_id"].n_unique() * 4)
        self.assertEqual(set(result["boundary_quantile"]), {50, 80})
        self.assertTrue(BANNED_FORWARD_LABELS.isdisjoint(result.columns))
        self.assertTrue(
            result.select(
                pl.col("source_asof_date")
                .drop_nulls()
                .str.strptime(pl.Date, "%Y%m%d")
                .lt(pl.col("Date").drop_nulls().str.strptime(pl.Date, "%Y%m%d"))
                .all()
            ).item()
        )


if __name__ == "__main__":
    unittest.main()
