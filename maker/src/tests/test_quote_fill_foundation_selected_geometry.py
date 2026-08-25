"""Tests for selected-lookup S0.5 known-cost geometry."""

from __future__ import annotations

import unittest

import polars as pl

from maker.src.quote_fill.foundation_cohort_selection import TOD_BUCKETS
from maker.src.quote_fill.foundation_selected_geometry import (
    SELECTED_GEOMETRY_VERSION,
    build_selected_lookup_geometry,
)


class FoundationSelectedGeometryTest(unittest.TestCase):
    def test_builds_seven_policies_for_every_primary_tod_cell(self) -> None:
        result = build_selected_lookup_geometry(
            _mother(),
            _mapping(),
            _boundaries(),
        )
        geometry = result.geometry_long

        self.assertEqual(geometry.height, 2 * len(TOD_BUCKETS) * 7)
        coverage = geometry.group_by(
            ["Date", "ValueCode", "QuoteCode", "tod_bucket"]
        ).agg(pl.col("policy_id").n_unique().alias("policies"))
        self.assertEqual(coverage.height, 2 * len(TOD_BUCKETS))
        self.assertTrue((coverage["policies"] == 7).all())
        self.assertTrue(geometry["development_only"].all())
        self.assertTrue(geometry["reference_diagnostic"].all())
        self.assertFalse(geometry["actionable_execution"].any())
        self.assertFalse(geometry["ev_ready"].any())
        self.assertFalse(geometry["contains_target_day_outcome"].any())
        self.assertEqual(
            geometry["selected_geometry_version"].unique().to_list(),
            [SELECTED_GEOMETRY_VERSION],
        )

        q95 = geometry.filter(
            (pl.col("Date") == "20260505")
            & (pl.col("tod_bucket") == TOD_BUCKETS[0])
            & (pl.col("policy_id") == "q95")
        ).row(0, named=True)
        self.assertAlmostEqual(q95["upper_distance_bp"], 30.0)
        self.assertAlmostEqual(q95["lower_distance_bp"], 25.0)
        self.assertAlmostEqual(q95["nominal_band_bp"], 55.0)
        self.assertAlmostEqual(
            q95["nominal_same_day_known_cost_margin_bp"],
            q95["nominal_band_bp"] - q95["same_day_reference_cost_bp"],
        )
        self.assertEqual(q95["anchor_model_id"], "time_ewma_15s")
        self.assertEqual(q95["candidate_id"], "Q1_trail60_date_equal")
        self.assertEqual(q95["upper_source_asof_date"], "20260504")
        self.assertEqual(q95["lower_source_asof_date"], "20260501")
        self.assertEqual(q95["source_asof_date"], "20260504")

        fixed = geometry.filter(
            (pl.col("Date") == "20260505")
            & (pl.col("tod_bucket") == TOD_BUCKETS[0])
            & (pl.col("policy_id") == "fixed30")
        ).row(0, named=True)
        self.assertAlmostEqual(fixed["nominal_band_bp"], 60.0)
        self.assertEqual(fixed["anchor_model_id"], "time_ewma_15s")
        self.assertEqual(fixed["candidate_id"], "Q1_trail60_date_equal")

    def test_summaries_distinguish_product_days_from_tod_rows(self) -> None:
        result = build_selected_lookup_geometry(
            _mother(),
            _mapping(),
            _boundaries(),
        )

        overall = result.summary_overall.filter(pl.col("policy_id") == "q95").row(
            0, named=True
        )
        self.assertEqual(overall["product_days"], 2)
        self.assertEqual(overall["product_day_tod_rows"], 8)
        self.assertEqual(overall["tod_buckets"], 4)
        self.assertEqual(overall["months"], 2)
        self.assertEqual(overall["summary_scope"], "overall")
        self.assertEqual(overall["anchor_model_id"], "time_ewma_15s")
        self.assertEqual(overall["candidate_id"], "Q1_trail60_date_equal")
        self.assertTrue(overall["development_only"])
        self.assertFalse(overall["actionable_execution"])
        self.assertFalse(overall["ev_ready"])

        self.assertEqual(result.summary_by_month.height, 2 * 7)
        may = result.summary_by_month.filter(
            (pl.col("month") == "202605") & (pl.col("policy_id") == "q95")
        ).row(0, named=True)
        self.assertEqual(may["product_days"], 1)
        self.assertEqual(may["product_day_tod_rows"], 4)
        self.assertEqual(may["summary_scope"], "month")

        self.assertEqual(result.summary_by_tod.height, len(TOD_BUCKETS) * 7)
        first_tod = result.summary_by_tod.filter(
            (pl.col("tod_bucket") == TOD_BUCKETS[0]) & (pl.col("policy_id") == "q95")
        ).row(0, named=True)
        self.assertEqual(first_tod["product_days"], 2)
        self.assertEqual(first_tod["product_day_tod_rows"], 2)
        self.assertEqual(first_tod["tod_buckets"], 1)
        self.assertEqual(first_tod["summary_scope"], "tod_bucket")

    def test_rejects_incomplete_or_unsafe_selected_lookup(self) -> None:
        incomplete = _boundaries().head(_boundaries().height - 1)
        with self.assertRaisesRegex(ValueError, "24-cell"):
            build_selected_lookup_geometry(_mother(), _mapping(), incomplete)

        unsafe = _boundaries().with_columns(
            pl.when(
                (pl.col("Date") == "20260505")
                & (pl.col("tod_bucket") == TOD_BUCKETS[0])
                & (pl.col("boundary_quantile") == 95)
                & (pl.col("side") == "positive")
            )
            .then(pl.col("Date"))
            .otherwise(pl.col("source_asof_date"))
            .alias("source_asof_date")
        )
        with self.assertRaisesRegex(ValueError, "invalid or unsafe"):
            build_selected_lookup_geometry(_mother(), _mapping(), unsafe)

    def test_rejects_unsafe_mother_lineage(self) -> None:
        unsafe = _mother().with_columns(
            pl.when(pl.col("s1_primary"))
            .then(pl.col("Date"))
            .otherwise(pl.col("boundary_source_asof_date"))
            .alias("boundary_source_asof_date")
        )

        with self.assertRaisesRegex(ValueError, "unsafe boundary_source_asof_date"):
            build_selected_lookup_geometry(unsafe, _mapping(), _boundaries())


def _mother() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Date": ["20260505", "20260601", "20260601"],
            "ValueCode": ["2330", "2317", "1101"],
            "QuoteCode": ["TXFA", "DHFN6", "DFFN6"],
            "anchor_model_id": ["time_ewma_15s"] * 3,
            "candidate_id": ["Q1_trail60_date_equal"] * 3,
            "s1_primary": [True, True, False],
            "boundary_source_asof_date": [
                "20260504",
                "20260529",
                "20260529",
            ],
            "liquidity_source_asof_date": [
                "20260504",
                "20260529",
                "20260529",
            ],
            "contains_target_day_outcome": [False, False, False],
        }
    )


def _mapping() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Date": ["20260505", "20260601", "20260601"],
            "ValueCode": ["2330", "2317", "1101"],
            "QuoteCode": ["TXFA", "DHFN6", "DFFN6"],
            "spot_ref_price": [100.0, 101.0, 50.0],
            "fut_ref_price": [100.0, 101.0, 50.0],
            "contract_size": [2_000.0, 2_000.0, 2_000.0],
            "decimal_locator": [2, 2, 2],
            "end_date": ["20260617", "20260617", "20260617"],
            "day_trade_mark": ["X", "X", "X"],
            "trading_turnover": [1.0, 2.0, 3.0],
            "ins_type": ["stock", "stock", "stock"],
        }
    )


def _boundaries() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    product_days = (
        ("20260505", "2330", "TXFA", "20260504", "20260501"),
        ("20260601", "2317", "DHFN6", "20260529", "20260528"),
    )
    distances = {
        50: (10.0, 5.0),
        80: (20.0, 15.0),
        95: (30.0, 25.0),
    }
    for date, value_code, quote_code, upper_source, lower_source in product_days:
        for tod_index, tod_bucket in enumerate(TOD_BUCKETS):
            for quantile, (upper, lower) in distances.items():
                for side, distance, source in (
                    ("positive", upper + tod_index, upper_source),
                    ("negative", lower + tod_index, lower_source),
                ):
                    rows.append(
                        {
                            "Date": date,
                            "ValueCode": value_code,
                            "QuoteCode": quote_code,
                            "anchor_model_id": "time_ewma_15s",
                            "candidate_id": "Q1_trail60_date_equal",
                            "tod_bucket": tod_bucket,
                            "boundary_quantile": quantile,
                            "side": side,
                            "boundary_distance_bp": distance,
                            "source_asof_date": source,
                            "native_supported": True,
                            "effective_supported": True,
                            "contains_target_day_outcome": False,
                        }
                    )
    return pl.from_dicts(rows, infer_schema_length=None)


if __name__ == "__main__":
    unittest.main()
