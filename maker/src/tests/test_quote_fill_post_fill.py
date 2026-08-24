from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest

import polars as pl

from maker.src.quote_fill.post_fill import (
    run_post_fill_opportunity_study,
    session_cutoff_ns,
)


DATE = "20260128"
BASE = datetime(2026, 1, 28, 5, 19, 0)  # 13:19 Asia/Taipei on UTC clock
NS = 1_000_000_000


def _timestamp(second: int) -> datetime:
    return BASE + timedelta(seconds=second)


def _ns(second: int, milliseconds: int = 0) -> int:
    epoch = datetime(1970, 1, 1)
    return int((_timestamp(second) - epoch).total_seconds() * NS) + milliseconds * 1_000_000


def _aliases(*, fill_second: int, fill_ms: int = 0) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Date": [DATE],
            "ValueCode": ["2317"],
            "QuoteCode": ["DHFB6"],
            "route": ["future_ask_spot_taker"],
            "boundary_quantile": [50],
            "raw_order_fact_id": ["same-raw-fact"],
            "policy_generation_id": ["q50-policy-generation"],
            "full_fill": [True],
            "full_fill_recv_time_ns": [_ns(fill_second, fill_ms)],
            "full_fill_event_sequence": [1],
            "full_fill_row_index": [123],
        }
    )


def _adaptive() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Date": [DATE],
            "ValueCode": ["2317"],
            "QuoteCode": ["DHFB6"],
            "boundary_quantile": [50],
            "lower_distance_bp": [10.0],
            "adaptive_parameter_valid": [True],
            "source_asof_date": ["20260127"],
            "contains_target_day_outcome": [False],
        }
    )


def _fair(
    seconds: list[int],
    basis: list[float],
    anchors: list[float],
    eligible: list[bool] | None = None,
) -> pl.DataFrame:
    eligible = eligible or [True] * len(seconds)
    return pl.DataFrame(
        {
            "Date": [DATE] * len(seconds),
            "ValueCode": ["2317"] * len(seconds),
            "QuoteCode": ["DHFB6"] * len(seconds),
            "timestamp": [_timestamp(second) for second in seconds],
            "basis_mid_bp": basis,
            "anchor_ewma_120s_bp": anchors,
            "analysis_eligible": eligible,
            "eligible_base": eligible,
        }
    )


class PostFillOpportunityTest(unittest.TestCase):
    def test_causal_frozen_m0_delay_and_moving_anchor_apparent_hit(self) -> None:
        # Fill at 50.200s: latest eligible fair state is t=50 (M0=50), not
        # the post-fill t=51 row (M=100).  A 1s delay starts at ceil(51.2)=52.
        seconds = list(range(50, 61))
        basis = [70.0, 40.0, 60.0] + [60.0] * 8
        anchors = [50.0, 100.0, 65.0, 75.0] + [75.0] * 7
        result = run_post_fill_opportunity_study(
            _aliases(fill_second=50, fill_ms=200),
            _fair(seconds, basis, anchors),
            _adaptive(),
        )
        self.assertEqual(result.labels.height, 8)
        one_second = result.labels.filter(
            pl.col("observation_delay_seconds") == 1
        )

        dynamic = one_second.filter(
            pl.col("target_id") == "dynamic_center"
        ).row(0, named=True)
        self.assertEqual(dynamic["frozen_anchor_m0_bp"], 50.0)
        self.assertEqual(dynamic["entry_fair_timestamp_ns"], _ns(50))
        self.assertEqual(dynamic["evaluation_start_grid_ns"], _ns(52))
        self.assertEqual(dynamic["status"], "hit")
        self.assertEqual(dynamic["terminal_timestamp_ns"], _ns(52))
        self.assertTrue(dynamic["anchor_only_apparent_hit"])
        self.assertAlmostEqual(dynamic["time_to_latent_hit_seconds"], 1.8)

        # t=51 would have hit the frozen center, but it is before the first
        # eligible observation grid and cannot leak through the delay.
        frozen = one_second.filter(
            pl.col("target_id") == "frozen_center"
        ).row(0, named=True)
        self.assertEqual(frozen["status"], "session_no_hit")
        self.assertFalse(frozen["latent_exit_opportunity_hit"])

        dynamic_lower = one_second.filter(
            pl.col("target_id") == "dynamic_adaptive_lower"
        ).row(0, named=True)
        self.assertEqual(dynamic_lower["status"], "hit")
        self.assertEqual(dynamic_lower["terminal_timestamp_ns"], _ns(53))
        self.assertTrue(dynamic_lower["anchor_only_apparent_hit"])

        thirty = result.labels.filter(
            pl.col("observation_delay_seconds") == 30
        )
        self.assertTrue((thirty["status"] == "session_no_hit").all())
        self.assertTrue((thirty["observed_eligible_grids"] == 0).all())

    def test_first_eligibility_gap_censors_and_does_not_bridge_to_later_hit(self) -> None:
        fair = _fair(
            [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
            [70.0, 70.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
            [50.0] * 11,
            [True, False, True, True, True, True, True, True, True, True, True],
        )
        labels = run_post_fill_opportunity_study(
            _aliases(fill_second=50), fair, _adaptive()
        ).labels.filter(pl.col("observation_delay_seconds") == 1)
        self.assertTrue((labels["status"] == "censor").all())
        self.assertTrue((labels["terminal_timestamp_ns"] == _ns(51)).all())
        self.assertTrue(
            labels["censor_reason"].str.contains("analysis_ineligible").all()
        )
        self.assertTrue((labels["observed_eligible_grids"] == 0).all())

    def test_1320_cutoff_is_exclusive(self) -> None:
        # The target is first met exactly at 13:20.  That grid is outside the
        # research session and cannot be called a same-day opportunity hit.
        fair = _fair(
            [57, 58, 59, 60],
            [70.0, 60.0, 60.0, 30.0],
            [50.0, 50.0, 50.0, 50.0],
        )
        labels = run_post_fill_opportunity_study(
            _aliases(fill_second=57), fair, _adaptive()
        ).labels.filter(pl.col("observation_delay_seconds") == 1)
        self.assertTrue((labels["status"] == "session_no_hit").all())
        self.assertTrue((labels["observed_eligible_grids"] == 2).all())
        self.assertEqual(labels.item(0, "session_cutoff_ns"), session_cutoff_ns(DATE))

    def test_missing_entry_state_is_explicit_and_summary_preserves_support(self) -> None:
        fair = _fair(
            [51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
            [30.0] * 10,
            [50.0] * 10,
        )
        result = run_post_fill_opportunity_study(
            _aliases(fill_second=50), fair, _adaptive()
        )
        self.assertTrue((result.labels["status"] == "no_entry_state").all())
        self.assertEqual(result.summary.height, 8)
        row = result.summary.row(0, named=True)
        self.assertEqual(row["full_fill_policy_aliases"], 1)
        self.assertEqual(row["entry_state_available"], 0)
        self.assertEqual(row["no_entry_state_aliases"], 1)
        self.assertFalse(row["display_support_valid"])
        self.assertFalse(row["actionable_execution"])
        self.assertFalse(row["pnl_ready"])
        self.assertFalse(row["ev_ready"])

    def test_latest_ineligible_state_blocks_older_eligible_entry_state(self) -> None:
        fair = _fair(
            [49, 50, 51, 52],
            [70.0, 70.0, 70.0, 30.0],
            [50.0, 50.0, 50.0, 50.0],
            [True, True, False, True],
        )
        labels = run_post_fill_opportunity_study(
            _aliases(fill_second=51, fill_ms=200), fair, _adaptive()
        ).labels
        self.assertTrue((labels["status"] == "no_entry_state").all())
        self.assertTrue(labels["entry_fair_timestamp_ns"].is_null().all())

    def test_target_day_adaptive_outcome_is_rejected(self) -> None:
        unsafe = _adaptive().with_columns(
            pl.lit(True).alias("contains_target_day_outcome")
        )
        with self.assertRaisesRegex(ValueError, "target-day outcomes"):
            run_post_fill_opportunity_study(
                _aliases(fill_second=50),
                _fair([50, 51], [70.0, 60.0], [50.0, 50.0]),
                unsafe,
            )

    def test_writer_names_outputs_as_latent_opportunities(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run_post_fill_opportunity_study(
                _aliases(fill_second=57),
                _fair([57, 58, 59], [70.0, 60.0, 60.0], [50.0] * 3),
                _adaptive(),
                output_dir=output,
            )
            self.assertTrue(
                (output / "latent_exit_opportunity_labels.parquet").exists()
            )
            self.assertTrue(
                (output / "latent_exit_opportunity_summary.csv").exists()
            )
            self.assertFalse((output / "exit_pnl.csv").exists())


if __name__ == "__main__":
    unittest.main()
