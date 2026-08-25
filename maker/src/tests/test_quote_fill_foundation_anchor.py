"""Tests for the single-session S0.5 anchor validation functions."""

from __future__ import annotations

import unittest
from datetime import date

import polars as pl

from maker.src.quote_fill.foundation_anchor import (
    add_freshness_labels,
    build_anchor_candidates,
    evaluate_anchor_day,
    summarize_prior_day,
)


def _day_frame(
    seconds: list[int],
    basis: list[float],
    *,
    session_date: str = "20260813",
    value_code: str = "2303",
    quote_code: str = "CCFH6",
    eligible_base: list[bool] | None = None,
    eligible_1000ms: list[bool] | None = None,
    eligible_100ms: list[bool] | None = None,
    leg_skew_ms: list[float] | None = None,
) -> pl.DataFrame:
    rows = len(seconds)
    base = eligible_base if eligible_base is not None else [True] * rows
    age_1000 = eligible_1000ms if eligible_1000ms is not None else list(base)
    age_100 = eligible_100ms if eligible_100ms is not None else list(base)
    skew = leg_skew_ms if leg_skew_ms is not None else [0.0] * rows
    return pl.DataFrame(
        {
            "Date": [session_date] * rows,
            "ValueCode": [value_code] * rows,
            "QuoteCode": [quote_code] * rows,
            "seconds_from_open": seconds,
            "end_date": [date(2026, 8, 19)] * rows,
            "basis_mid_bp": basis,
            "eligible_base": base,
            "eligible_1000ms": age_1000,
            "eligible_100ms": age_100,
            "leg_skew_ms": skew,
        }
    )


class FoundationAnchorTest(unittest.TestCase):
    def test_future_center_recomputes_the_entire_window_for_each_gate(self) -> None:
        seconds = list(range(601))
        basis = [float(value) for value in seconds]
        age_1000 = [True] * len(seconds)
        for second in range(330, 358):
            age_1000[second] = False
        candidates = build_anchor_candidates(
            _day_frame(seconds, basis, eligible_1000ms=age_1000)
        )

        base = add_freshness_labels(candidates, "base").filter(
            pl.col("seconds_from_open") == 300
        )
        strict = add_freshness_labels(candidates, "age_1000ms").filter(
            pl.col("seconds_from_open") == 300
        )

        self.assertAlmostEqual(base.item(0, "future_center_bp"), 465.0)
        self.assertAlmostEqual(base.item(0, "future_center_coverage"), 1.0)
        self.assertIsNone(strict.item(0, "future_center_bp"))
        self.assertAlmostEqual(
            strict.item(0, "future_center_coverage"),
            243 / 271,
        )

    def test_exact_prior_seed_and_invalid_prior_fallback(self) -> None:
        prior_a = _day_frame(
            list(range(300, 305)),
            [1.0, 2.0, 3.0, 4.0, 5.0],
            session_date="20260812",
            eligible_base=[True, True, True, True, False],
        )
        prior_b = _day_frame(
            list(range(300, 305)),
            [1.0, 2.0, 3.0, 4.0, 5.0],
            session_date="20260812",
            value_code="2317",
            quote_code="CDFH6",
            eligible_base=[True, True, True, False, False],
        )
        priors = summarize_prior_day(
            pl.concat([prior_a, prior_b]),
            target_date="20260813",
            prior_date="20260812",
        )
        prior_flags = dict(
            priors.select("ValueCode", "prior_valid").iter_rows()
        )
        self.assertTrue(prior_flags["2303"])
        self.assertFalse(prior_flags["2317"])
        self.assertAlmostEqual(
            priors.filter(pl.col("ValueCode") == "2303").item(
                0, "prior_coverage"
            ),
            0.8,
        )

        current_a = _day_frame([0, 1, 2], [10.0, 10.0, 10.0])
        current_b = _day_frame(
            [0, 1, 2],
            [10.0, 10.0, 10.0],
            value_code="2317",
            quote_code="CDFH6",
        )
        candidates = build_anchor_candidates(
            pl.concat([current_a, current_b]), priors
        )
        valid = candidates.filter(
            (pl.col("ValueCode") == "2303")
            & (pl.col("seconds_from_open") == 0)
        )
        fallback = candidates.filter(
            (pl.col("ValueCode") == "2317")
            & (pl.col("seconds_from_open") == 0)
        )
        self.assertTrue(valid.item(0, "prior_valid"))
        self.assertGreater(
            abs(
                valid.item(0, "anchor_prior_seeded_ewma_120s_bp")
                - valid.item(0, "anchor_ewma_120s_bp")
            ),
            1e-6,
        )
        self.assertFalse(fallback.item(0, "prior_valid"))
        self.assertAlmostEqual(
            fallback.item(0, "anchor_prior_seeded_ewma_120s_bp"),
            fallback.item(0, "anchor_ewma_120s_bp"),
        )

    def test_last_five_minutes_are_horizon_censored(self) -> None:
        seconds = list(range(15_299, 15_600))
        result = evaluate_anchor_day(
            _day_frame(seconds, [10.0] * len(seconds)),
            priors=None,
            stage="pseudo_holdout",
        )
        sample = result.coverage.filter(
            (pl.col("model") == "ewma_120s")
            & (pl.col("freshness_sample") == "base")
        )
        family_counts = dict(
            sample.group_by("stratum_family")
            .len()
            .select("stratum_family", "len")
            .iter_rows()
        )
        self.assertEqual(family_counts, {"overall": 1, "tod": 2, "dte": 1})
        final_bucket = result.coverage.filter(
            (pl.col("model") == "ewma_120s")
            & (pl.col("freshness_sample") == "base")
            & (pl.col("stratum_family") == "tod")
            & (pl.col("stratum_value") == "13:15-13:20_horizon_censored")
        )
        self.assertEqual(final_bucket.item(0, "n_grid"), 300)
        self.assertEqual(final_bucket.item(0, "n_current_eligible"), 300)
        self.assertEqual(final_bucket.item(0, "n_horizon_possible"), 0)
        self.assertEqual(
            final_bucket.item(0, "censor_horizon_after_session"), 300
        )
        self.assertEqual(final_bucket.item(0, "evaluable"), 0)

    def test_delayed_positive_has_occupancy_and_300s_nonoverlap(self) -> None:
        seconds = list(range(1_201))
        basis = [0.0 if second < 300 else 10.0 for second in seconds]
        result = evaluate_anchor_day(
            _day_frame(seconds, basis),
            priors=None,
            stage="pseudo_holdout",
        )
        delayed = result.delayed_daily.filter(
            (pl.col("model") == "open_5m")
            & (pl.col("freshness_sample") == "base")
            & (pl.col("stratum_family") == "overall")
        )
        self.assertEqual(delayed["n_positive_occupancy"].sum(), 601)
        self.assertEqual(delayed["n_positive_nonoverlap"].sum(), 3)

    def test_tv_does_not_bridge_an_eligibility_gap(self) -> None:
        seconds = list(range(298, 305))
        basis = [0.0, 0.0, 10.0, 999.0, 30.0, 31.0, 999.0]
        gate = [False, False, True, False, True, True, False]
        result = evaluate_anchor_day(
            _day_frame(seconds, basis, eligible_base=gate),
            priors=None,
            stage="pseudo_holdout",
        )
        persistence = result.anchor_daily.filter(
            (pl.col("model") == "persistence")
            & (pl.col("freshness_sample") == "base")
            & (pl.col("stratum_family") == "tod")
            & (pl.col("stratum_value") == "09:05-09:15")
        )
        self.assertEqual(persistence.item(0, "n_adjacent_legal_pairs"), 1)
        self.assertAlmostEqual(persistence.item(0, "anchor_tv_bp"), 1.0)
        self.assertAlmostEqual(persistence.item(0, "basis_tv_bp"), 1.0)

    def test_locked_forward_date_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "locked development cutoff"):
            build_anchor_candidates(
                _day_frame(
                    [0, 1],
                    [1.0, 1.0],
                    session_date="20260814",
                )
            )


if __name__ == "__main__":
    unittest.main()
