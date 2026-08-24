from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import polars as pl

from maker.src.quote_fill.dynamic_future_hedge import (
    DynamicFutureHedgeConfig,
    label_dynamic_future_hedges,
    load_relevant_future_book_hits,
    select_manifest_validated_fills,
    summarize_dynamic_future_hedges,
)
from maker.src.quote_fill.dynamic_future_hedge_provenance_migration import (
    migrate_legacy_provenance,
)

DATE = "20260128"
VALUE_CODE = "2317"
QUOTE_CODE = "DHFB6"
BASE_NS = 1_769_562_300_000_000_000  # 2026-01-28 01:05:00 UTC


def _manifest() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Date": [DATE],
            "ValueCode": [VALUE_CODE],
            "QuoteCode": [QUOTE_CODE],
            "selector_version": ["monthly_selector_causal_v2"],
        }
    )


def _outcomes(*, value_code: str = VALUE_CODE) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "physical_order_id": ["order-1", "order-2", "order-3"],
            "Date": [DATE] * 3,
            "ValueCode": [value_code] * 3,
            "QuoteCode": [QUOTE_CODE] * 3,
            "makerfill_implied_fill_time_ns": [
                BASE_NS + 10_000_000,
                None,
                BASE_NS + 20_000_000,
            ],
            "approximate_fill_before_nominal_stop": [True, False, None],
            "outcome_supported": [True, True, False],
            "contract_size": [2000.0] * 3,
            "target_price": [99.0, 99.0, 99.0],
        }
    )


def _mapping() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Date": [DATE],
            "ValueCode": [VALUE_CODE],
            "QuoteCode": [QUOTE_CODE],
            "spot_ref_price": [100.0],
            "fut_ref_price": [100.0],
            "contract_size": [2000.0],
        }
    )


def _future_raw() -> pl.DataFrame:
    recv = [
        datetime(2026, 1, 28, 1, 5, 0, tzinfo=UTC),
        datetime(2026, 1, 28, 1, 5, 0, 20_000, tzinfo=UTC),
        datetime(2026, 1, 28, 1, 5, 0, 40_000, tzinfo=UTC),
    ]
    data: dict[str, object] = {
        "RecvTime": recv,
        "TransTime": [
            datetime(2026, 1, 28, 9, 5, 0),  # noqa: DTZ001 - feed is naive
            datetime(2026, 1, 28, 9, 5, 0, 20_000),  # noqa: DTZ001
            datetime(2026, 1, 28, 9, 5, 0, 40_000),  # noqa: DTZ001
        ],
        "QuoteCode": [QUOTE_CODE] * 3,
        "PacketSeq": [1, 2, 3],
        "ChannelSeq": [10, 11, 12],
        "TrialMatch": [0, 1, 0],
        "DecimalLocator": [2, 2, 2],
        "TotalFillLots": [0, 1, 1],
        "FillPrice": [0, 10000, 0],
        "FillLots": [0, 1, 0],
        "BestBidPrice": [10100, 0, 9900],
        "BestBidLots": [2, 0, 3],
        "BestAskPrice": [10100, 0, 10300],
        "BestAskLots": [2, 0, 3],
    }
    for side, prices, lots in (
        ("Bid", [100, 99, 98, 97, 96], [5, 4, 3, 2, 1]),
        ("Ask", [102, 103, 104, 105, 106], [5, 4, 3, 2, 1]),
    ):
        for level in range(1, 6):
            data[f"{side}Price{level}"] = [prices[level - 1] * 100, 0, 0]
            data[f"{side}Lots{level}"] = [lots[level - 1], 0, 0]
    return pl.DataFrame(data)


class DynamicFutureHedgeTest(unittest.TestCase):
    def test_legacy_provenance_migration_changes_no_prices(self) -> None:
        legacy = pl.DataFrame(
            {
                "physical_order_id": ["a", "b", "c"],
                "decision_best_price": [100.0, 101.0, 102.0],
                "signed_total_slippage_bp": [0.0, 1.0, 2.0],
                "status": ["executable"] * 3,
                "future_asof_backend": [
                    None,
                    "pyarrow_sorted_stream",
                    "polars_global_sort_fallback",
                ],
                "raw_receive_time_regression_detected": [None, False, True],
            }
        )
        migrated = migrate_legacy_provenance(legacy)
        self.assertTrue(
            legacy.select(
                "physical_order_id",
                "decision_best_price",
                "signed_total_slippage_bp",
                "status",
            ).equals(
                migrated.select(
                    "physical_order_id",
                    "decision_best_price",
                    "signed_total_slippage_bp",
                    "status",
                ),
                null_equal=True,
            )
        )
        self.assertEqual(
            migrated["future_asof_backend"].to_list(),
            [
                "pyarrow_sorted_stream",
                "pyarrow_sorted_stream",
                "polars_global_sort_forced",
            ],
        )
        self.assertEqual(
            migrated["raw_receive_time_regression_detected"].to_list(),
            [False, False, False],
        )
        self.assertEqual(
            migrated["regression_check_performed"].to_list(),
            [True, True, False],
        )

    def test_manifest_filter_selects_only_supported_fill(self) -> None:
        selected, counters = select_manifest_validated_fills(
            _outcomes(), _manifest()
        )
        self.assertEqual(selected.height, 1)
        self.assertEqual(selected.item(0, "physical_order_id"), "order-1")
        self.assertEqual(counters["candidate_outcomes"], 3)
        self.assertEqual(counters["approximate_fills"], 1)

    def test_manifest_rejects_product_outside_dynamic_universe(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the causal daily manifest"):
            select_manifest_validated_fills(
                _outcomes(value_code="2330"), _manifest()
            )

    def test_sparse_loader_preserves_l1_best_and_ignores_zero_book_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "future.parquet"
            _future_raw().write_parquet(path)
            states = load_relevant_future_book_hits(
                DATE,
                select_manifest_validated_fills(_outcomes(), _manifest())[0],
                _mapping(),
                future_path=path,
            )
        self.assertEqual(states.height, 2)  # arrival and +50 ms only
        self.assertEqual(
            states["future_asof_backend"].unique().to_list(),
            ["pyarrow_sorted_stream"],
        )
        self.assertFalse(states["raw_receive_time_regression_detected"].any())
        self.assertTrue(states["regression_check_performed"].all())
        arrival = states.filter(pl.col("query_kind") == "arrival").row(
            0, named=True
        )
        self.assertEqual(arrival["l1_bid_price_1"], 100.0)
        self.assertEqual(arrival["best_bid_price"], 101.0)
        decision = states.filter(pl.col("query_kind") == "decision").row(
            0, named=True
        )
        self.assertEqual(decision["l1_bid_price_1"], 100.0)
        self.assertEqual(decision["best_bid_price"], 99.0)

    def test_sell_one_contract_labels_latency_and_zero_depth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "future.parquet"
            _future_raw().write_parquet(path)
            states = load_relevant_future_book_hits(
                DATE,
                select_manifest_validated_fills(_outcomes(), _manifest())[0],
                _mapping(),
                future_path=path,
            )
        selected, _ = select_manifest_validated_fills(_outcomes(), _manifest())
        facts = label_dynamic_future_hedges(
            selected,
            states,
            DynamicFutureHedgeConfig(hedge_delay_ns=50_000_000),
        )
        row = facts.row(0, named=True)
        self.assertEqual(row["status"], "executable")
        self.assertEqual(row["arrival_reference_price"], 101.0)
        self.assertEqual(row["decision_best_price"], 100.0)
        self.assertEqual(row["executable_vwap_price"], 100.0)
        self.assertAlmostEqual(
            row["signed_latency_slippage_bp"], 1.0 / 101.0 * 10_000.0
        )
        self.assertEqual(row["signed_depth_slippage_bp"], 0.0)
        self.assertAlmostEqual(
            row["signed_total_slippage_bp"], 1.0 / 101.0 * 10_000.0
        )
        self.assertEqual(row["decision_book_age_ms"], 20.0)
        self.assertFalse(row["entry_fill_time_exact"])
        self.assertTrue(row["future_zero_book_nonprice_events_ignored"])
        summary = summarize_dynamic_future_hedges(facts)
        overall = summary.filter(pl.col("scope") == "overall")
        self.assertEqual(overall.item(0, "executable_rate"), 1.0)

    def test_unsorted_across_batches_uses_global_sort_fallback(self) -> None:
        raw = (
            _future_raw()
            .with_row_index("_row")
            .with_columns(
                pl.when(pl.col("_row") == 1)
                .then(10200)
                .otherwise(pl.col("BestBidPrice"))
                .alias("BestBidPrice"),
                pl.when(pl.col("_row") == 1)
                .then(2)
                .otherwise(pl.col("BestBidLots"))
                .alias("BestBidLots"),
                pl.when(pl.col("_row") == 1)
                .then(10300)
                .otherwise(pl.col("BestAskPrice"))
                .alias("BestAskPrice"),
                pl.when(pl.col("_row") == 1)
                .then(2)
                .otherwise(pl.col("BestAskLots"))
                .alias("BestAskLots"),
            )
            .drop("_row")
        )
        raw = pl.concat(
            [raw.slice(index, 1) for index in (0, 2, 1)],
            how="vertical_relaxed",
        )
        selected, _ = select_manifest_validated_fills(_outcomes(), _manifest())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "future.parquet"
            raw.write_parquet(path)
            with patch(
                "maker.src.quote_fill.dynamic_future_hedge._STREAM_BATCH_SIZE",
                2,
            ):
                hits = load_relevant_future_book_hits(
                    DATE,
                    selected,
                    _mapping(),
                    future_path=path,
                )
        self.assertEqual(
            hits["future_asof_backend"].unique().to_list(),
            ["polars_global_sort_fallback"],
        )
        self.assertTrue(hits["raw_receive_time_regression_detected"].all())
        self.assertTrue(hits["regression_check_performed"].all())
        arrival = hits.filter(pl.col("query_kind") == "arrival")
        decision = hits.filter(pl.col("query_kind") == "decision")
        self.assertEqual(arrival.item(0, "best_bid_price"), 101.0)
        self.assertEqual(decision.item(0, "best_bid_price"), 99.0)
        facts = label_dynamic_future_hedges(selected, hits)
        self.assertEqual(facts.item(0, "status"), "executable")
        self.assertAlmostEqual(
            facts.item(0, "signed_total_slippage_bp"),
            1.0 / 101.0 * 10_000.0,
        )

    def test_forced_global_sort_has_distinct_provenance(self) -> None:
        selected, _ = select_manifest_validated_fills(_outcomes(), _manifest())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "future.parquet"
            _future_raw().write_parquet(path)
            hits = load_relevant_future_book_hits(
                DATE,
                selected,
                _mapping(),
                config=DynamicFutureHedgeConfig(force_global_sort_backend=True),
                future_path=path,
            )
        self.assertEqual(
            hits["future_asof_backend"].unique().to_list(),
            ["polars_global_sort_forced"],
        )
        self.assertFalse(hits["raw_receive_time_regression_detected"].any())
        self.assertFalse(hits["regression_check_performed"].any())
        facts = label_dynamic_future_hedges(selected, hits)
        self.assertEqual(
            facts.item(0, "future_asof_backend"),
            "polars_global_sort_forced",
        )
        self.assertFalse(facts.item(0, "regression_check_performed"))


if __name__ == "__main__":
    unittest.main()
