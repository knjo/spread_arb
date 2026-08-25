"""Focused contracts for causal boundary candidate orchestration."""

from __future__ import annotations

import unittest
from datetime import date, timedelta

import polars as pl

from maker.src.quote_fill.foundation_boundary_adaptation import SCORE_SCHEMA
from maker.src.quote_fill.foundation_boundary_orchestration import (
    ORCHESTRATION_MULTIPLIER_SCHEMA,
    Q1_ID,
    Q3_ID,
    Q4_ID,
    Q6_ID,
    BoundaryCandidatePanel,
    BoundaryOrchestrationConfig,
    add_tod_variants,
    build_boundary_candidate_panel,
)

ANCHOR_ID = "time_ewma_30s"
MINIMA_ONE = {50: 1, 80: 1, 95: 1}
TOD_STARTS = {
    "0905_1000": 301,
    "1000_1100": 3_601,
    "1100_1200": 7_201,
    "1200_1300": 10_801,
}


def _session_dates(count: int = 56) -> list[date]:
    first = date(2026, 1, 2)
    return [first + timedelta(days=offset) for offset in range(count)]


def _canonical_panel(
    count: int = 56,
) -> tuple[list[str], pl.DataFrame, pl.DataFrame]:
    dates = _session_dates(count)
    expiry = date(2026, 12, 16)
    episodes: list[dict[str, object]] = []
    mapping: list[dict[str, object]] = []
    for date_index, trading_date in enumerate(dates):
        date_text = trading_date.strftime("%Y%m%d")
        quote_code = f"Q{date_index:03d}"
        mapping.append(
            {
                "Date": date_text,
                "ValueCode": "1111",
                "QuoteCode": quote_code,
                "expiry_date": expiry,
                "calendar_dte": (expiry - trading_date).days,
            }
        )
        for bucket_index, (tod_bucket, start_second) in enumerate(TOD_STARTS.items()):
            for side, side_shift in (("positive", 0.0), ("negative", 1.0)):
                episodes.append(
                    {
                        "episode_id": (f"{date_text}/{quote_code}/{tod_bucket}/{side}"),
                        "Date": date_text,
                        "ValueCode": "1111",
                        "QuoteCode": quote_code,
                        "anchor_model_id": ANCHOR_ID,
                        "side": side,
                        "observed_amplitude_bp": (
                            8.0 + date_index / 20.0 + bucket_index + side_shift
                        ),
                        "left_censored": False,
                        "right_censored": False,
                        "completed_center_return": True,
                        "start_seconds_from_open": start_second,
                        "end_seconds_from_open": start_second + 1,
                        "right_censor_reason": None,
                        "tod_bucket": tod_bucket,
                        "expiry_date": expiry,
                        "calendar_dte": (expiry - trading_date).days,
                    }
                )
    return (
        [value.strftime("%Y%m%d") for value in dates],
        pl.from_dicts(episodes, infer_schema_length=None),
        pl.from_dicts(mapping, infer_schema_length=None),
    )


def _test_config(*, minimum_cells: int = 25) -> BoundaryOrchestrationConfig:
    return BoundaryOrchestrationConfig(
        q3_level_sessions=5,
        q3_minimum_level_dates=4,
        q3_minimum_observable_cells_per_side=minimum_cells,
        q4_level_sessions=10,
        q4_minimum_level_dates=8,
        q4_minimum_observable_cells_per_side=minimum_cells,
        tod_level_sessions=10,
        tod_minimum_level_dates=8,
        tod_minimum_observable_cells_per_bucket_side=minimum_cells,
    )


def _manual_q1_predictions(
    target_dates: list[str],
    sessions: list[str],
    mapping: pl.DataFrame,
) -> pl.DataFrame:
    session_index = {value: index for index, value in enumerate(sessions)}
    quote_by_date = dict(mapping.select("Date", "QuoteCode").iter_rows())
    rows: list[dict[str, object]] = []
    for date_text in target_dates:
        source_date = sessions[session_index[date_text] - 1]
        for tod_bucket in TOD_STARTS:
            for side, shift in (("positive", 0.0), ("negative", 1.0)):
                for quantile, boundary in ((50, 10.0), (80, 15.0), (95, 20.0)):
                    rows.append(
                        {
                            "Date": date_text,
                            "ValueCode": "1111",
                            "QuoteCode": quote_by_date[date_text],
                            "anchor_model_id": ANCHOR_ID,
                            "candidate_id": Q1_ID,
                            "candidate_kind": "date_equal_censor_identified",
                            "tod_bucket": tod_bucket,
                            "boundary_quantile": quantile,
                            "side": side,
                            "boundary_distance_bp": boundary + shift,
                            "source_asof_date": source_date,
                            "history_start_date": sessions[0],
                            "history_end_date": source_date,
                            "native_supported": True,
                            "effective_supported": True,
                            "effective_candidate_id": Q1_ID,
                            "effective_source_asof_date": source_date,
                            "fallback_used": False,
                            "fallback_reason": None,
                            "contains_target_day_outcome": False,
                        }
                    )
    return pl.from_dicts(rows, infer_schema_length=None)


def _manual_support_audit(target_dates: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Date": target_dates,
            "anchor_model_id": [ANCHOR_ID] * len(target_dates),
            "candidate_id": [Q1_ID] * len(target_dates),
            "mapped_product_day_tod_units": [4] * len(target_dates),
            "native_all_q_units": [4] * len(target_dates),
            "effective_all_q_units": [4] * len(target_dates),
            "fallback_effective_units": [0] * len(target_dates),
            "product_day_tod_units_with_rows": [4] * len(target_dates),
            "native_all_q_coverage": [1.0] * len(target_dates),
            "effective_all_q_coverage": [1.0] * len(target_dates),
            "contains_target_day_outcome": [False] * len(target_dates),
        }
    )


class BoundaryOrchestrationTests(unittest.TestCase):
    def test_builds_effective_q6_q3_and_q4_panel(self) -> None:
        sessions, episodes, mapping = _canonical_panel()
        target_date = sessions[-1]

        panel = build_boundary_candidate_panel(
            episodes,
            sessions,
            mapping,
            target_dates=[target_date],
            anchor_model_id=ANCHOR_ID,
            minimum_completed_per_side=MINIMA_ONE,
            config=_test_config(),
        )

        expected = {
            "Q0_trail60_event_pooled",
            Q1_ID,
            "Q2_trail20_date_equal",
            Q3_ID,
            Q4_ID,
            Q6_ID,
        }
        self.assertEqual(set(panel.predictions["candidate_id"].unique()), expected)
        self.assertEqual(panel.predictions["Date"].unique().to_list(), [target_date])
        self.assertTrue(panel.predictions["effective_supported"].all())
        self.assertTrue(
            panel.predictions.filter(
                pl.col("source_asof_date") >= pl.col("Date")
            ).is_empty()
        )
        self.assertTrue((~panel.predictions["contains_target_day_outcome"]).all())

        q1 = panel.predictions.filter(pl.col("candidate_id") == Q1_ID).sort(
            ["tod_bucket", "boundary_quantile", "side"]
        )
        q6 = panel.predictions.filter(pl.col("candidate_id") == Q6_ID).sort(
            ["tod_bucket", "boundary_quantile", "side"]
        )
        self.assertEqual(q1.height, 24)
        self.assertEqual(q6.height, q1.height)
        self.assertTrue((~q6["native_supported"]).all())
        self.assertTrue(q6["fallback_used"].all())
        self.assertEqual(q6["effective_candidate_id"].unique().to_list(), [Q1_ID])
        for actual, expected_value in zip(
            q6["boundary_distance_bp"],
            q1["boundary_distance_bp"],
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected_value)

        for candidate_id in (Q3_ID, Q4_ID):
            candidate = panel.predictions.filter(pl.col("candidate_id") == candidate_id)
            self.assertEqual(candidate.height, 24)
            self.assertTrue(candidate["native_supported"].all())
            self.assertEqual(
                candidate["effective_candidate_id"].unique().to_list(),
                [candidate_id],
            )

        self.assertEqual(
            set(panel.calibration["candidate_id"].unique()),
            expected,
        )
        self.assertTrue(panel.calibration["score_contains_same_day_outcome"].all())
        self.assertEqual(
            set(panel.multipliers["output_candidate_id"].unique()),
            {Q3_ID, Q4_ID},
        )
        self.assertTrue(panel.multipliers["scale_native_supported"].all())
        q3_scale = panel.multipliers.filter(
            (pl.col("output_candidate_id") == Q3_ID) & (pl.col("side") == "positive")
        ).row(0, named=True)
        self.assertEqual(q3_scale["observable_score_rows"], 60)
        self.assertEqual(q3_scale["observable_episode_cells"], 20)
        self.assertTrue(
            panel.multipliers.filter(
                pl.col("source_asof_date") >= pl.col("Date")
            ).is_empty()
        )
        self.assertTrue((~panel.multipliers["contains_target_day_outcome"]).all())
        q5_support = panel.support_audit.filter(
            (pl.col("Date") == target_date)
            & (pl.col("candidate_id") == "Q5_prev1_expiry_dte5")
        ).row(0, named=True)
        self.assertEqual(q5_support["mapped_product_day_tod_units"], 4)
        self.assertEqual(q5_support["native_all_q_units"], 0)
        self.assertEqual(q5_support["effective_all_q_units"], 0)
        q6_support = panel.support_audit.filter(
            (pl.col("Date") == target_date) & (pl.col("candidate_id") == Q6_ID)
        ).row(0, named=True)
        self.assertEqual(q6_support["native_all_q_units"], 0)
        self.assertEqual(q6_support["effective_all_q_units"], 4)

    def test_incremental_tod_pass_reuses_base_panel(self) -> None:
        sessions, episodes, mapping = _canonical_panel()
        target_dates = sessions[-11:]
        predictions = _manual_q1_predictions(target_dates, sessions, mapping)
        base_panel = BoundaryCandidatePanel(
            raw_predictions=predictions.head(0),
            predictions=predictions,
            calibration=pl.DataFrame(schema=SCORE_SCHEMA),
            multipliers=pl.DataFrame(schema=ORCHESTRATION_MULTIPLIER_SCHEMA),
            support_audit=_manual_support_audit(target_dates),
            build_dates=tuple(target_dates),
            target_dates=tuple(target_dates),
            anchor_model_id=ANCHOR_ID,
        )

        panel = add_tod_variants(
            base_panel,
            episodes,
            sessions,
            tod_source_candidates=(Q1_ID,),
            config=_test_config(),
        )

        self.assertTrue(panel.raw_predictions.equals(base_panel.raw_predictions))
        self.assertEqual(
            panel.predictions.filter(pl.col("candidate_id") == Q1_ID).height,
            predictions.height,
        )
        tod_id = f"{Q1_ID}__tod10"
        final_tod = panel.predictions.filter(
            (pl.col("Date") == target_dates[-1]) & (pl.col("candidate_id") == tod_id)
        )
        self.assertEqual(final_tod.height, 24)
        self.assertTrue(final_tod["native_supported"].all())
        self.assertEqual(
            panel.multipliers.filter(
                (pl.col("Date") == target_dates[-1])
                & (pl.col("output_candidate_id") == tod_id)
            ).height,
            8,
        )
        final_support = panel.support_audit.filter(
            (pl.col("Date") == target_dates[-1]) & (pl.col("candidate_id") == tod_id)
        ).row(0, named=True)
        self.assertEqual(final_support["mapped_product_day_tod_units"], 4)
        self.assertEqual(final_support["native_all_q_units"], 4)
        self.assertAlmostEqual(final_support["native_all_q_coverage"], 1.0)

    def test_insufficient_scale_support_falls_back_to_q1(self) -> None:
        sessions, episodes, mapping = _canonical_panel()
        target_date = sessions[-1]
        panel = build_boundary_candidate_panel(
            episodes,
            sessions,
            mapping,
            target_dates=[target_date],
            minimum_completed_per_side=MINIMA_ONE,
            config=_test_config(minimum_cells=10_000),
        )

        q1 = panel.predictions.filter(pl.col("candidate_id") == Q1_ID).sort(
            ["tod_bucket", "boundary_quantile", "side"]
        )
        for candidate_id in (Q3_ID, Q4_ID):
            candidate = panel.predictions.filter(
                pl.col("candidate_id") == candidate_id
            ).sort(["tod_bucket", "boundary_quantile", "side"])
            self.assertTrue((~candidate["native_supported"]).all())
            self.assertTrue(candidate["fallback_multiplier_used"].all())
            self.assertEqual(
                candidate["effective_candidate_id"].unique().to_list(),
                [Q1_ID],
            )
            for actual, expected_value in zip(
                candidate["boundary_distance_bp"],
                q1["boundary_distance_bp"],
                strict=True,
            ):
                self.assertAlmostEqual(actual, expected_value)

        target_scales = panel.multipliers.filter(pl.col("Date") == target_date)
        self.assertTrue((~target_scales["scale_native_supported"]).all())

    def test_rejects_anchor_mismatch_and_nonfrozen_tod_source(self) -> None:
        sessions, episodes, mapping = _canonical_panel()
        with self.assertRaisesRegex(ValueError, "anchor_model_id does not match"):
            build_boundary_candidate_panel(
                episodes,
                sessions,
                mapping,
                target_dates=[sessions[-1]],
                anchor_model_id="different_anchor",
                minimum_completed_per_side=MINIMA_ONE,
            )
        with self.assertRaisesRegex(ValueError, "unsupported TOD source"):
            build_boundary_candidate_panel(
                episodes,
                sessions,
                mapping,
                target_dates=[sessions[-1]],
                tod_source_candidates=("Q0_trail60_event_pooled",),
                minimum_completed_per_side=MINIMA_ONE,
            )


if __name__ == "__main__":
    unittest.main()
