"""Synthetic tests for the non-overlapping latent basis-cycle state machine."""

from __future__ import annotations

from datetime import datetime, timedelta
import unittest

import polars as pl

from ..quote_width.cycle import (
    _outcome_by_horizon,
    build_cycle_policy_universe,
    build_latent_cycles,
    summarize_cycles_by_day_symbol,
)


def _panel(
    basis: list[float],
    anchor: list[float] | None = None,
    *,
    seconds: list[int] | None = None,
    valid: list[bool] | None = None,
    fresh: list[bool] | None = None,
) -> pl.DataFrame:
    rows = len(basis)
    seconds = seconds or list(range(300, 300 + rows))
    anchor = anchor or [0.0] * rows
    valid = valid or [True] * rows
    fresh = fresh or [True] * rows
    start = datetime(2026, 1, 28, 1, 5)
    return pl.DataFrame(
        {
            "Date": ["20260128"] * rows,
            "ValueCode": ["2303"] * rows,
            "QuoteCode": ["CCFB6"] * rows,
            "timestamp": [
                start + timedelta(seconds=value - seconds[0]) for value in seconds
            ],
            "seconds_from_open": seconds,
            "basis_mid_bp": basis,
            "anchor_ewma_120s_bp": anchor,
            "analysis_eligible": valid,
            "eligible_base": valid,
            "eligible_1000ms": fresh,
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("ns")))


def _universe(
    *,
    anchor_modes: tuple[str, ...] = ("dynamic",),
    exit_width_ratios: tuple[float, ...] = (1.0,),
    exit_delays_seconds: tuple[int, ...] = (1,),
    samples: tuple[str, ...] = ("base",),
) -> pl.DataFrame:
    candidates = pl.DataFrame(
        {
            "Date": ["20260128"],
            "ValueCode": ["2303"],
            "QuoteCode": ["CCFB6"],
            "width_family": ["fixed_bp"],
            "width_policy": ["fixed_10bp"],
            "candidate_width_bp": [10.0],
        }
    )
    return build_cycle_policy_universe(
        candidates,
        anchor_modes=anchor_modes,
        exit_width_ratios=exit_width_ratios,
        exit_delays_seconds=exit_delays_seconds,
        samples=samples,
    )


class LatentCycleStateMachineTest(unittest.TestCase):
    def test_policy_universe_rejects_undefined_ttl_and_invalid_dimensions(self) -> None:
        candidates = _universe().select(
            "Date",
            "ValueCode",
            "QuoteCode",
            "width_family",
            "width_policy",
            "candidate_width_bp",
        ).unique()
        with self.assertRaisesRegex(ValueError, "force-flat"):
            build_cycle_policy_universe(candidates, max_horizon_seconds=60)
        with self.assertRaisesRegex(ValueError, "finite"):
            build_cycle_policy_universe(candidates, exit_width_ratios=(float("nan"),))
        with self.assertRaisesRegex(ValueError, "whole seconds"):
            build_cycle_policy_universe(candidates, exit_delays_seconds=(1.5,))

    def test_repeated_upper_states_do_not_overlap_and_exact_bounds_count(self) -> None:
        result = build_latent_cycles(
            _panel([-1.0, 10.0, 12.0, 11.0, -10.0, -1.0, 10.0, -10.0]),
            _universe(),
        )
        self.assertEqual(result.height, 2)
        self.assertEqual(result["status"].to_list(), ["complete", "complete"])
        self.assertEqual(result["entry_seconds_from_open"].to_list(), [301, 306])
        self.assertEqual(result["end_seconds_from_open"].to_list(), [304, 307])
        self.assertTrue(
            result.item(1, "entry_seconds_from_open")
            > result.item(0, "end_seconds_from_open")
        )

    def test_invalid_gap_censors_open_cycle_and_stops_that_policy_day(self) -> None:
        result = build_latent_cycles(
            _panel(
                [-1.0, 10.0, 12.0, 12.0, -1.0, 10.0, -10.0],
                valid=[True, True, False, True, True, True, True],
            ),
            _universe(),
        )
        self.assertEqual(result.height, 1)
        self.assertEqual(result.item(0, "status"), "censored")
        self.assertEqual(result.item(0, "end_reason"), "eligibility_gap")
        self.assertTrue(result.item(0, "policy_day_stopped_after_event"))

    def test_flat_invalid_gap_can_rearm_only_after_a_new_below_to_above_cross(self) -> None:
        result = build_latent_cycles(
            _panel(
                [-1.0, 12.0, 12.0, -1.0, 10.0, -10.0],
                valid=[True, False, True, True, True, True],
            ),
            _universe(),
        )
        self.assertEqual(result.height, 1)
        self.assertEqual(result.item(0, "entry_seconds_from_open"), 304)
        self.assertEqual(result.item(0, "status"), "complete")

    def test_timestamp_gap_does_not_bridge_an_upper_crossing(self) -> None:
        result = build_latent_cycles(
            _panel(
                [-1.0, 10.0, 12.0, -1.0, 10.0, -10.0],
                seconds=[300, 301, 303, 304, 305, 306],
            ),
            _universe(),
        )
        self.assertEqual(result.height, 1)
        self.assertEqual(result.item(0, "end_reason"), "timestamp_gap")
        self.assertEqual(result.item(0, "end_seconds_from_open"), 301)
        self.assertEqual(result.item(0, "censor_seconds_from_open"), 303)
        self.assertEqual(result.schema["entry_timestamp"], pl.Datetime("ns"))
        self.assertEqual(result.schema["end_timestamp"], pl.Datetime("ns"))
        self.assertEqual(result.schema["censor_timestamp"], pl.Datetime("ns"))

    def test_first_row_above_upper_is_left_censored(self) -> None:
        result = build_latent_cycles(
            _panel([12.0, 11.0, -1.0, 10.0, -10.0]),
            _universe(),
        )
        self.assertEqual(result.height, 1)
        self.assertEqual(result.item(0, "entry_seconds_from_open"), 303)

    def test_frozen_exit_does_not_rearm_while_dynamic_residual_is_above_upper(self) -> None:
        result = build_latent_cycles(
            _panel(
                [-1.0, 10.0, -1.0, 0.0, -21.0, -10.0, -30.0],
                anchor=[0.0, 0.0, -20.0, -20.0, -20.0, -20.0, -20.0],
            ),
            _universe(anchor_modes=("frozen_entry",), exit_width_ratios=(0.0,)),
        )
        self.assertEqual(result.height, 2)
        self.assertEqual(result["entry_seconds_from_open"].to_list(), [301, 305])
        self.assertEqual(result.item(0, "end_seconds_from_open"), 302)

    def test_dynamic_and_frozen_exit_separate_anchor_motion_from_basis_capture(self) -> None:
        result = build_latent_cycles(
            _panel(
                [-1.0, 10.0, 9.0, -10.0],
                anchor=[0.0, 0.0, 20.0, 20.0],
            ),
            _universe(anchor_modes=("dynamic", "frozen_entry")),
        )
        dynamic = result.filter(pl.col("anchor_mode") == "dynamic").row(
            0, named=True
        )
        frozen = result.filter(pl.col("anchor_mode") == "frozen_entry").row(
            0, named=True
        )
        self.assertEqual(dynamic["end_seconds_from_open"], 302)
        self.assertEqual(frozen["end_seconds_from_open"], 303)
        self.assertAlmostEqual(dynamic["basis_capture_bp"], 1.0)
        self.assertAlmostEqual(frozen["basis_capture_bp"], 20.0)
        self.assertAlmostEqual(dynamic["capture_identity_error_bp"], 0.0)
        self.assertAlmostEqual(frozen["capture_identity_error_bp"], 0.0)

    def test_center_half_and_symmetric_lower_boundaries_have_correct_sign(self) -> None:
        result = build_latent_cycles(
            _panel([-1.0, 10.0, 0.0, -5.0, -10.0]),
            _universe(exit_width_ratios=(0.0, 0.5, 1.0)),
        ).sort("exit_width_ratio")
        self.assertEqual(result["end_seconds_from_open"].to_list(), [302, 303, 304])
        self.assertEqual(result["exit_width_bp"].to_list(), [0.0, 5.0, 10.0])

    def test_actual_timestamp_gap_is_detected_when_second_index_looks_continuous(self) -> None:
        panel = _panel([-1.0, 10.0, -10.0])
        panel = panel.with_columns(
            pl.when(pl.col("seconds_from_open") == 302)
            .then(pl.col("timestamp") + pl.duration(seconds=1))
            .otherwise(pl.col("timestamp"))
            .alias("timestamp")
        )
        result = build_latent_cycles(panel, _universe())
        self.assertEqual(result.height, 1)
        self.assertEqual(result.item(0, "status"), "censored")
        self.assertEqual(result.item(0, "end_reason"), "timestamp_gap")

    def test_freshness_gate_censors_instead_of_skipping_then_reconnecting(self) -> None:
        result = build_latent_cycles(
            _panel(
                [-1.0, 10.0, 12.0],
                fresh=[True, True, False],
            ),
            _universe(samples=("fresh_1000ms",)),
        )
        self.assertEqual(result.height, 1)
        self.assertEqual(result.item(0, "status"), "censored")
        self.assertEqual(result.item(0, "end_reason"), "freshness_gate")

    def test_base_sample_requires_eligible_base_even_if_analysis_flag_is_true(self) -> None:
        panel = _panel([-1.0, 10.0, -10.0]).with_columns(
            pl.lit(False).alias("eligible_base")
        )
        with self.assertRaisesRegex(ValueError, "no latent entry events"):
            build_latent_cycles(panel, _universe())

    def test_nonfinite_state_reason_is_consistent(self) -> None:
        panel = _panel([-1.0, 10.0, 12.0]).with_columns(
            pl.when(pl.col("seconds_from_open") == 302)
            .then(None)
            .otherwise(pl.col("anchor_ewma_120s_bp"))
            .alias("anchor_ewma_120s_bp")
        )
        result = build_latent_cycles(panel, _universe())
        self.assertEqual(result.item(0, "end_reason"), "nonfinite_state")
        self.assertEqual(result.item(0, "all_end_reasons"), "nonfinite_state")

    def test_horizon_outcome_is_tristate(self) -> None:
        horizons = (5,)
        self.assertIs(
            _outcome_by_horizon("complete", 2, 2, horizons)[
                "complete_within_5s"
            ],
            True,
        )
        self.assertIs(
            _outcome_by_horizon("complete", 8, 8, horizons)[
                "complete_within_5s"
            ],
            False,
        )
        self.assertIsNone(
            _outcome_by_horizon("censored", None, 3, horizons)[
                "complete_within_5s"
            ]
        )
        self.assertIs(
            _outcome_by_horizon("censored", None, 8, horizons)[
                "complete_within_5s"
            ],
            False,
        )

    def test_all_censored_summary_keeps_nullable_metric_schema(self) -> None:
        universe = _universe()
        cycles = build_latent_cycles(
            _panel([-1.0, 10.0, 12.0], valid=[True, True, False]),
            universe,
            horizons_seconds=(1, 2, 3),
        )
        summary = summarize_cycles_by_day_symbol(
            cycles,
            universe,
            horizons_seconds=(1, 2, 3),
        )
        self.assertEqual(summary.item(0, "censored_cycles"), 1)
        self.assertIsNone(summary.item(0, "basis_capture_bp_p50"))
        self.assertIsNone(summary.item(0, "p_complete_within_2s"))


if __name__ == "__main__":
    unittest.main()
