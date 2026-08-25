from __future__ import annotations

import unittest

import polars as pl

from maker.src.quote_fill.foundation_cohort_selection import (
    FORBIDDEN_OUTPUT_FIELDS,
    build_anchor_product_day_support,
    build_s1_mother_cohort,
)


class FoundationCohortSelectionTest(unittest.TestCase):
    @staticmethod
    def _anchor_support() -> pl.DataFrame:
        return pl.DataFrame(
            {
                "Date": ["20260505"],
                "ValueCode": ["2330"],
                "QuoteCode": ["2330F"],
                "anchor_model_id": ["time_ewma_30s"],
                "selected_anchor_support": [True],
                "selected_anchor_native_seconds": [10],
                "analysis_eligible_seconds": [10],
                "selected_anchor_coverage_rate": [1.0],
            }
        )

    @staticmethod
    def _boundaries(*, omit_last: bool = False) -> pl.DataFrame:
        rows = []
        for bucket in (
            "0905_1000",
            "1000_1100",
            "1100_1200",
            "1200_1300",
        ):
            for quantile in (50, 80, 95):
                for side in ("positive", "negative"):
                    rows.append(
                        {
                            "Date": "20260505",
                            "ValueCode": "2330",
                            "QuoteCode": "2330F",
                            "anchor_model_id": "time_ewma_30s",
                            "candidate_id": "Q1_trail60_date_equal",
                            "tod_bucket": bucket,
                            "boundary_quantile": quantile,
                            "side": side,
                            "boundary_distance_bp": float(quantile),
                            "native_supported": True,
                            "effective_supported": True,
                            "source_asof_date": "20260504",
                            "contains_target_day_outcome": False,
                        }
                    )
        return pl.from_dicts(rows[:-1] if omit_last else rows)

    @staticmethod
    def _liquidity(*, mismatch: bool = False) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "Date": ["20260505", "20260505"],
                "ValueCode": ["2330", "2330"],
                "QuoteCode": ["2330F", "2330F"],
                "route": ["spot_bid_future_taker"] * 2,
                "boundary_quantile": [50, 80],
                "source_asof_date": ["20260504", "20260504"],
                "long_history_gate": [True, not mismatch],
                "recent_history_gate": [True, True],
                "hard_data_gate": [True, True],
                # These incumbent-boundary gates must not affect the new mother.
                "boundary_parameter_gate": [False, False],
                "support_gate": [False, False],
                "pre_replay_candidate": [False, False],
            }
        )

    def test_anchor_support_uses_any_causal_finite_entry_second(self) -> None:
        day = pl.DataFrame(
            {
                "Date": ["20260505", "20260505", "20260505"],
                "ValueCode": ["2330"] * 3,
                "QuoteCode": ["2330F"] * 3,
                "seconds_from_open": [299, 300, 301],
                "analysis_eligible": [True, True, True],
                "selected_anchor_bp": [None, None, 3.0],
            }
        )

        result = build_anchor_product_day_support(
            day,
            anchor_column="selected_anchor_bp",
            anchor_model_id="time_ewma_30s",
        ).row(0, named=True)

        self.assertTrue(result["selected_anchor_support"])
        self.assertEqual(result["selected_anchor_native_seconds"], 1)
        self.assertEqual(result["analysis_eligible_seconds"], 2)
        self.assertAlmostEqual(result["selected_anchor_coverage_rate"], 0.5)

    def test_legacy_q_dependent_gates_do_not_select_the_mother(self) -> None:
        result = build_s1_mother_cohort(
            pl.DataFrame(
                {
                    "Date": ["20260505"],
                    "ValueCode": ["2330"],
                    "QuoteCode": ["2330F"],
                }
            ),
            self._anchor_support(),
            self._boundaries(),
            self._liquidity(),
            selected_anchor_model_id="time_ewma_30s",
            selected_candidate_id="Q1_trail60_date_equal",
        )

        row = result.mother.row(0, named=True)
        self.assertTrue(row["selected_all_q_support"])
        self.assertTrue(row["q_independent_liquidity_gate"])
        self.assertTrue(row["s1_primary"])
        self.assertFalse(FORBIDDEN_OUTPUT_FIELDS & set(result.mother.columns))
        self.assertTrue(result.liquidity_invariance_audit["q50_q80_invariant"].all())

    def test_incomplete_boundary_grid_fails_selected_support(self) -> None:
        result = build_s1_mother_cohort(
            pl.DataFrame(
                {
                    "Date": ["20260505"],
                    "ValueCode": ["2330"],
                    "QuoteCode": ["2330F"],
                }
            ),
            self._anchor_support(),
            self._boundaries(omit_last=True),
            self._liquidity(),
            selected_anchor_model_id="time_ewma_30s",
            selected_candidate_id="Q1_trail60_date_equal",
        )

        self.assertFalse(result.mother["selected_all_q_support"].item())
        self.assertFalse(result.mother["s1_primary"].item())

    def test_q50_q80_liquidity_gate_mismatch_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "q50/q80"):
            build_s1_mother_cohort(
                pl.DataFrame(
                    {
                        "Date": ["20260505"],
                        "ValueCode": ["2330"],
                        "QuoteCode": ["2330F"],
                    }
                ),
                self._anchor_support(),
                self._boundaries(),
                self._liquidity(mismatch=True),
                selected_anchor_model_id="time_ewma_30s",
                selected_candidate_id="Q1_trail60_date_equal",
            )

    def test_null_source_dates_fail_closed(self) -> None:
        mapping = pl.DataFrame(
            {
                "Date": ["20260505"],
                "ValueCode": ["2330"],
                "QuoteCode": ["2330F"],
            }
        )
        cases = (
            (
                "boundary",
                self._boundaries().with_columns(
                    pl.lit(None, dtype=pl.String).alias("source_asof_date")
                ),
                self._liquidity(),
                "boundary source date",
            ),
            (
                "liquidity",
                self._boundaries(),
                self._liquidity().with_columns(
                    pl.lit(None, dtype=pl.String).alias("source_asof_date")
                ),
                "liquidity source date",
            ),
        )
        for name, boundaries, liquidity, message in cases:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                build_s1_mother_cohort(
                    mapping,
                    self._anchor_support(),
                    boundaries,
                    liquidity,
                    selected_anchor_model_id="time_ewma_30s",
                    selected_candidate_id="Q1_trail60_date_equal",
                )


if __name__ == "__main__":
    unittest.main()
