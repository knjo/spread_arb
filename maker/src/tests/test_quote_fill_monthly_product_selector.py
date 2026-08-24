"""Tests for the leakage-safe M-1 to M product selector."""

from __future__ import annotations

import unittest

import polars as pl

from maker.src.quote_fill.monthly_product_selector import (
    MonthlyProductSelectorConfig,
    build_daily_allowlist,
    build_dynamic_full_band_events,
    build_monthly_membership,
    build_monthly_product_metrics,
)


def _config() -> MonthlyProductSelectorConfig:
    return MonthlyProductSelectorConfig(
        minimum_pass_days=1,
        minimum_upper_events=1,
        minimum_signal_days=1,
        minimum_lcb80=0.0,
        maximum_wait_upper_bound_p90_seconds=1_000,
    )


def _boundary(
    date: str,
    value_code: str,
    quote_code: str,
) -> dict[str, object]:
    return {
        "Date": date,
        "ValueCode": value_code,
        "QuoteCode": quote_code,
        "boundary_quantile": 95,
        "upper_distance_bp": 10.0,
        "lower_distance_bp": 10.0,
        "adaptive_parameter_valid": True,
        "positive_completed_per_session": 2.0,
        "negative_completed_per_session": 2.0,
        "source_asof_date": "20260102" if date.startswith("202601") else "20260202",
        "execution_safe_snapshot": True,
        "contains_target_day_outcome": False,
    }


def _liquidity(
    date: str,
    value_code: str,
    quote_code: str,
    *,
    candidate: bool = True,
) -> dict[str, object]:
    return {
        "Date": date,
        "ValueCode": value_code,
        "QuoteCode": quote_code,
        "boundary_quantile": 80,
        "route": "spot_bid_future_taker",
        "pre_replay_candidate": candidate,
        "liquidity_gate_status": "pass" if candidate else "known_fail",
        "source_asof_date": "20260102" if date.startswith("202601") else "20260202",
        "execution_safe_snapshot": True,
        "contains_target_day_outcome": False,
    }


def _excursion(
    date: str,
    value_code: str,
    quote_code: str,
    side: str,
    amplitude_bp: float,
    start: int,
) -> dict[str, object]:
    return {
        "Date": date,
        "ValueCode": value_code,
        "QuoteCode": quote_code,
        "side": side,
        "amplitude_bp": amplitude_bp,
        "start_seconds_from_open": start,
        "end_seconds_from_open": start + 20,
    }


class MonthlyProductSelectorTest(unittest.TestCase):
    def setUp(self) -> None:
        products = (("1111", "QA"), ("2222", "QB"))
        self.boundaries = pl.from_dicts(
            [
                _boundary(date, value_code, quote_code)
                for date in ("20260105", "20260203")
                for value_code, quote_code in products
            ],
            infer_schema_length=None,
        )
        self.liquidity = pl.from_dicts(
            [
                _liquidity(date, value_code, quote_code)
                for date in ("20260105", "20260203")
                for value_code, quote_code in products
            ],
            infer_schema_length=None,
        )
        self.excursions = pl.from_dicts(
            [
                _excursion("20260105", "1111", "QA", "positive", 12.0, 100),
                _excursion("20260105", "1111", "QA", "negative", 11.0, 300),
                _excursion("20260105", "2222", "QB", "positive", 12.0, 110),
                _excursion("20260203", "1111", "QA", "positive", 12.0, 150),
                _excursion("20260203", "1111", "QA", "negative", 11.0, 250),
                _excursion("20260203", "2222", "QB", "positive", 12.0, 160),
            ],
            infer_schema_length=None,
        )

    def test_previous_month_selects_next_month_without_target_labels(self) -> None:
        config = _config()
        events, eligible = build_dynamic_full_band_events(
            self.boundaries,
            self.liquidity,
            self.excursions,
            config=config,
        )
        january = events.filter(pl.col("observation_month") == "202601")
        hit = january.filter(pl.col("ValueCode") == "1111").row(0, named=True)
        miss = january.filter(pl.col("ValueCode") == "2222").row(0, named=True)
        self.assertTrue(hit["same_day_dynamic_lower_hit"])
        self.assertEqual(hit["start_to_start_seconds"], 200)
        self.assertEqual(hit["wait_lower_bound_seconds"], 180)
        self.assertEqual(hit["wait_upper_bound_seconds"], 220)
        self.assertFalse(miss["same_day_dynamic_lower_hit"])

        metrics = build_monthly_product_metrics(
            events,
            eligible,
            self.boundaries,
            config=config,
        )
        membership = build_monthly_membership(metrics, config=config)
        self.assertEqual(
            membership["effective_month"].unique().to_list(), ["202602"]
        )
        february = membership.filter(pl.col("effective_month") == "202602")
        selected = february.filter(pl.col("monthly_primary_selected"))
        self.assertEqual(selected["ValueCode"].to_list(), ["1111"])
        row = selected.row(0, named=True)
        self.assertEqual(row["source_month"], "202601")
        self.assertLess(row["source_month_last_date"], "20260201")
        self.assertFalse(row["selection_contains_effective_month_outcome"])

        daily = build_daily_allowlist(membership, self.liquidity, config=config)
        feb_daily = daily.filter(pl.col("Date") == "20260203")
        allowed = feb_daily.filter(pl.col("new_entry_allowed"))
        self.assertEqual(allowed["ValueCode"].to_list(), ["1111"])
        removed = feb_daily.filter(pl.col("ValueCode") == "2222").row(
            0, named=True
        )
        self.assertTrue(removed["exit_only_if_inventory"])
        self.assertFalse(removed["force_flat_on_membership_removal"])

    def test_unsupported_or_slow_score_does_not_consume_primary_rank(self) -> None:
        metrics = pl.DataFrame(
            {
                "source_month": ["202601"] * 4,
                "ValueCode": ["unsupported", "too_slow", "winner", "runner_up"],
                "source_month_last_date": ["20260131"] * 4,
                "prior_pass_days": [0, 10, 10, 10],
                "prior_upper_events": [100, 20, 20, 20],
                "prior_signal_days": [0, 10, 10, 10],
                "prior_daily_same_day_lower_lcb80": [0.999, 0.950, 0.900, 0.800],
                "prior_wait_upper_bound_p90_seconds": [10.0, 2_000.0, 100.0, 100.0],
            }
        )
        membership = build_monthly_membership(
            metrics,
            config=MonthlyProductSelectorConfig(
                minimum_pass_days=1,
                minimum_upper_events=1,
                minimum_signal_days=1,
                minimum_lcb80=0.0,
                maximum_wait_upper_bound_p90_seconds=1_000,
            ),
            completed_source_months=("202601",),
        )
        ranked = membership.filter(pl.col("monthly_primary_selected")).sort(
            "primary_lcb_rank_in_effective_month"
        )
        self.assertEqual(ranked["ValueCode"].to_list(), ["winner", "runner_up"])
        unsupported = membership.filter(
            pl.col("ValueCode") == "unsupported"
        ).row(0, named=True)
        self.assertIsNone(unsupported["lcb_rank_in_effective_month"])
        too_slow = membership.filter(pl.col("ValueCode") == "too_slow").row(
            0, named=True
        )
        self.assertFalse(too_slow["monthly_primary_selected"])
        self.assertIsNone(too_slow["primary_lcb_rank_in_effective_month"])

    def test_rejects_target_day_liquidity_as_non_causal(self) -> None:
        unsafe = self.liquidity.with_columns(
            pl.when(pl.col("Date") == "20260203")
            .then(pl.lit("20260203"))
            .otherwise(pl.col("source_asof_date"))
            .alias("source_asof_date")
        )
        with self.assertRaisesRegex(ValueError, "not D-safe"):
            build_dynamic_full_band_events(
                self.boundaries,
                unsafe,
                self.excursions,
                config=_config(),
            )

    def test_rejects_null_source_date_instead_of_kleene_dropping_it(self) -> None:
        unsafe = self.boundaries.with_columns(
            pl.when(pl.col("Date") == "20260203")
            .then(pl.lit(None, dtype=pl.String))
            .otherwise(pl.col("source_asof_date"))
            .alias("source_asof_date")
        )
        with self.assertRaisesRegex(ValueError, "not D-safe"):
            build_dynamic_full_band_events(
                unsafe,
                self.liquidity,
                self.excursions,
                config=_config(),
            )

    def test_tied_scores_use_value_code_as_deterministic_tie_break(self) -> None:
        rows = []
        for value_code in ("BBBB", "AAAA"):
            rows.append(
                {
                    "source_month": "202601",
                    "ValueCode": value_code,
                    "source_month_last_date": "20260131",
                    "prior_pass_days": 10,
                    "prior_upper_events": 20,
                    "prior_signal_days": 10,
                    "prior_daily_same_day_lower_lcb80": 0.9,
                    "prior_wait_upper_bound_p90_seconds": 100.0,
                }
            )
        config = MonthlyProductSelectorConfig(
            minimum_pass_days=1,
            minimum_upper_events=1,
            minimum_signal_days=1,
            minimum_lcb80=0.0,
            maximum_wait_upper_bound_p90_seconds=1_000,
        )
        forward = build_monthly_membership(
            pl.from_dicts(rows),
            config=config,
            completed_source_months=("202601",),
        )
        reverse = build_monthly_membership(
            pl.from_dicts(list(reversed(rows))),
            config=config,
            completed_source_months=("202601",),
        )
        for frame in (forward, reverse):
            ranks = {
                row["ValueCode"]: row["primary_lcb_rank_in_effective_month"]
                for row in frame.iter_rows(named=True)
            }
            self.assertEqual(ranks, {"AAAA": 1, "BBBB": 2})

    def test_target_month_label_changes_cannot_change_its_membership(self) -> None:
        config = _config()
        events, eligible = build_dynamic_full_band_events(
            self.boundaries,
            self.liquidity,
            self.excursions,
            config=config,
        )
        original = build_monthly_membership(
            build_monthly_product_metrics(
                events, eligible, self.boundaries, config=config
            ),
            config=config,
        ).filter(pl.col("effective_month") == "202602")

        mutated_excursions = self.excursions.filter(
            ~pl.col("Date").str.starts_with("202602")
        )
        mutated_events, mutated_eligible = build_dynamic_full_band_events(
            self.boundaries,
            self.liquidity,
            mutated_excursions,
            config=config,
        )
        mutated = build_monthly_membership(
            build_monthly_product_metrics(
                mutated_events,
                mutated_eligible,
                self.boundaries,
                config=config,
            ),
            config=config,
            completed_source_months=("202601",),
        ).filter(pl.col("effective_month") == "202602")
        columns = [
            "ValueCode",
            "monthly_primary_selected",
            "primary_lcb_rank_in_effective_month",
        ]
        self.assertTrue(original.select(columns).equals(mutated.select(columns)))


if __name__ == "__main__":
    unittest.main()
