from __future__ import annotations

import unittest
from datetime import UTC, date, datetime, timedelta

import polars as pl

from ..quote_fill.policy_spec import ANCHOR_MODEL_ID, TOD_BUCKETS, PolicySpec
from ..quote_fill.s1_day_state import (
    ENTRY_STOP_SECOND,
    SESSION_START_SECOND,
    build_s1_policy_day_state,
    build_s1_policy_state_changes,
    thin_s1_policy_state_changes,
    weight_s1_policy_state_changes,
)


def _day() -> pl.DataFrame:
    seconds = list(range(15_600))
    start = datetime(2026, 5, 5, 1, 0, tzinfo=UTC).replace(tzinfo=None)
    timestamps = [start + timedelta(seconds=value) for value in seconds]
    return pl.DataFrame(
        {
            "Date": ["20260505"] * len(seconds),
            "ValueCode": ["2330"] * len(seconds),
            "QuoteCode": ["CDFE6"] * len(seconds),
            "timestamp": timestamps,
            "seconds_from_open": seconds,
            "spot_recv_time": timestamps,
            "spot_sequence": list(range(1, len(seconds) + 1)),
            "spot_ref_price": [50.0] * len(seconds),
            "fut_ref_price": [50.0] * len(seconds),
            "contract_size": [2_000.0] * len(seconds),
            "end_date": [date(2026, 6, 17)] * len(seconds),
            "spot_bid": [50.0] * len(seconds),
            "spot_ask": [50.1] * len(seconds),
            "spot_bid_lots": [10] * len(seconds),
            "spot_ask_lots": [10] * len(seconds),
            "fut_exec_bid": [50.2] * len(seconds),
            "fut_exec_bid_lots": [10] * len(seconds),
            "fut_exec_ask": [50.3] * len(seconds),
            "fut_exec_ask_lots": [10] * len(seconds),
            "basis_mid_bp": [20.0] * len(seconds),
            "eligible_base": [True] * len(seconds),
        }
    )


def _specs() -> tuple[PolicySpec, ...]:
    distances = (20.0, 30.0, 40.0, 50.0)
    return tuple(
        PolicySpec(
            Date="20260505",
            ValueCode="2330",
            QuoteCode="CDFE6",
            entry_tod_bucket=bucket,
            policy_id="q50",
            kind="quantile",
            upper_distance_bp=distance,
            lower_distance_bp=0.0,
            upper_source_id="Q2_trail20_date_equal",
            lower_source_id="C0_center",
            upper_source_asof_date="20260504",
            lower_source_asof_date="20260504",
            combined_source_asof_date="20260504",
            anchor_model_id=ANCHOR_MODEL_ID,
            boundary_quantile=50,
        )
        for bucket, distance in zip(TOD_BUCKETS, distances, strict=True)
    )


class S1DayStateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.state = build_s1_policy_day_state(_day(), _specs(), policy_id="q50")

    def test_complete_grid_and_frozen_policy_provenance(self) -> None:
        self.assertEqual(
            self.state.height,
            ENTRY_STOP_SECOND - SESSION_START_SECOND,
        )
        self.assertEqual(self.state["policy_id"].unique().to_list(), ["q50"])
        self.assertNotIn("contains_target_day_outcome", self.state.columns)
        first = self.state.filter(pl.col("seconds_from_open") == 300).row(0, named=True)
        self.assertAlmostEqual(float(first["selected_anchor_bp"]), 20.0, places=10)
        self.assertAlmostEqual(float(first["entry_threshold_basis_bp"]), 40.0)
        self.assertAlmostEqual(
            float(first["frozen_exit_threshold_basis_bp_at_observation"]),
            20.0,
        )

    def test_ab12_geometry_changes_with_tod_spec(self) -> None:
        first = self.state.filter(pl.col("seconds_from_open") == 300).row(0, named=True)
        second = self.state.filter(pl.col("seconds_from_open") == 3_600).row(
            0, named=True
        )
        self.assertEqual(first["target_location"], "BID1")
        self.assertEqual(int(first["point_offset"]), 0)
        self.assertTrue(bool(first["ab12_admission_open"]))
        self.assertEqual(second["target_location"], "BID2_BY_TICK")
        self.assertEqual(int(second["point_offset"]), -1)
        self.assertTrue(bool(second["ab12_admission_open"]))

    def test_sparse_changes_keep_bucket_target_transitions(self) -> None:
        changes = thin_s1_policy_state_changes(self.state)
        direct = build_s1_policy_state_changes(_day(), _specs(), policy_id="q50")
        seconds = set(changes["seconds_from_open"].to_list())
        self.assertIn(300, seconds)
        self.assertIn(3_600, seconds)
        self.assertLess(changes.height, self.state.height)
        self.assertTrue(direct.equals(changes))
        weighted = weight_s1_policy_state_changes(direct)
        self.assertEqual(
            int(weighted["represented_product_seconds"].sum()),
            ENTRY_STOP_SECOND - SESSION_START_SECOND,
        )

    def test_missing_tod_spec_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "exact four S1 TOD"):
            build_s1_policy_day_state(_day(), _specs()[:-1], policy_id="q50")

    def test_nonpassive_target_is_not_admitted(self) -> None:
        day = _day().with_columns(pl.lit(49.95).alias("spot_ask"))
        result = build_s1_policy_day_state(day, _specs(), policy_id="q50")
        row = result.filter(pl.col("seconds_from_open") == 300).row(0, named=True)
        self.assertEqual(row["gate_reason"], "target_not_passive")
        self.assertFalse(bool(row["base_gate_open"]))
        self.assertFalse(bool(row["ab12_admission_open"]))


if __name__ == "__main__":
    unittest.main()
