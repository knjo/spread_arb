"""Tests for daily rolling quote-fill state/probability tables."""

from __future__ import annotations

import unittest

import polars as pl

from maker.src.quote_fill.walkforward import (
    WalkForwardProbabilityConfig,
    WalkForwardSplit,
    build_action_outcome_facts,
    build_walkforward_probability_tables,
)


class WalkForwardActionFactTest(unittest.TestCase):
    def test_locked_edge_and_state_are_event_facts_not_ev(self) -> None:
        aliases = pl.DataFrame(
            {
                "Date": ["20260102", "20260102"],
                "ValueCode": ["2317", "2317"],
                "QuoteCode": ["DHFA6", "DHFA6"],
                "route": ["spot_bid_future_taker"] * 2,
                "boundary_quantile": [50, 50],
                "raw_order_fact_id": ["raw-a", "raw-b"],
                "policy_generation_id": ["a", "b"],
                "target_rank_at_submit": ["BID1", "BID2"],
                "initial_queue_ahead": [1, 8],
                "submit_recv_time_ns": [
                    1_767_313_000_000_000_000,
                    1_767_323_000_000_000_000,
                ],
                "threshold_basis_bp": [60.0, 60.0],
                "effective_basis_bp": [62.0, 62.0],
                "full_fill": [True, False],
                "cancel_required": [False, True],
                "spot_book_age_ms_at_submit": [20.0, 2000.0],
                "future_book_age_ms_at_submit": [30.0, 100.0],
                "source_asof_date": ["20260101", "20260101"],
                "parameter_version": ["test-parameter", "test-parameter"],
            }
        )
        snapshot = pl.DataFrame(
            {
                "Date": ["20260102"],
                "ValueCode": ["2317"],
                "QuoteCode": ["DHFA6"],
                "boundary_quantile": [50],
                "upper_distance_bp": [10.0],
                "lower_distance_bp": [8.0],
                "adaptive_parameter_valid": [True],
                "source_asof_date": ["20260101"],
                "contains_target_day_outcome": [False],
                "parameter_version": ["test-parameter"],
            }
        )
        hedge = pl.DataFrame(
            {
                "raw_order_fact_id": ["raw-a"],
                "status": ["executable"],
                "signed_total_slippage_bp": [3.0],
                "decision_book_age_ms": [50.0],
                "depth_shortfall": [0],
            }
        )
        labels = pl.DataFrame(
            {
                "policy_generation_id": ["a"],
                "target_id": ["frozen_adaptive_lower"],
                "observation_delay_seconds": [30],
                "status": ["hit"],
                "time_to_latent_hit_seconds": [90.0],
                "Date": ["20260102"],
                "ValueCode": ["2317"],
                "QuoteCode": ["DHFA6"],
                "route": ["spot_bid_future_taker"],
                "boundary_quantile": [50],
                "lower_distance_bp": [8.0],
                "adaptive_source_asof_date": ["20260101"],
                "adaptive_parameter_version": ["test-parameter"],
            }
        )
        facts = build_action_outcome_facts(aliases, snapshot, hedge, labels)
        filled = facts.filter(pl.col("policy_generation_id") == "a").row(
            0, named=True
        )
        # Fair=60-10=50, rounded open=12.  After 3 bp adverse hedge move,
        # locked basis=59 and therefore misses the original 60 bp threshold.
        self.assertEqual(filled["effective_open_distance_bp"], 12.0)
        self.assertEqual(filled["locked_basis_bp_50ms"], 59.0)
        self.assertEqual(filled["locked_open_distance_bp_50ms"], 9.0)
        self.assertFalse(filled["locked_threshold_qualified_50ms"])
        self.assertTrue(filled["lower_hit_30s"])
        self.assertEqual(filled["rank_bucket"], "at_bbo")
        self.assertEqual(filled["queue_bucket"], "00_0to1")
        self.assertFalse(filled["ev_ready"])

        bad_snapshot = snapshot.with_columns(
            pl.lit("different-version").alias("parameter_version")
        )
        with self.assertRaisesRegex(ValueError, "lineage"):
            build_action_outcome_facts(aliases, bad_snapshot, hedge, labels)
        null_lineage = aliases.with_columns(
            pl.lit(None, dtype=pl.String).alias("source_asof_date")
        )
        with self.assertRaisesRegex(ValueError, "lineage"):
            build_action_outcome_facts(
                null_lineage, snapshot, hedge, labels
            )


def _fact(
    date: str,
    *,
    fill: bool,
    label_end_date: str | None = None,
) -> dict[str, object]:
    return {
        "Date": date,
        "ValueCode": "2317",
        "route": "spot_bid_future_taker",
        "boundary_quantile": 50,
        "label_end_date": label_end_date or date,
        "entry_full_fill": fill,
        "hedge_label_observed": fill,
        "hedge_priceable_50ms": fill,
        "locked_threshold_qualified_50ms": fill,
        "lower_label_known_30s": fill,
        "lower_hit_30s": fill,
        "hedge_slippage_bp_50ms": 2.0 if fill else None,
        "nominal_latent_band_bp": 20.0,
        "rank_bucket": "at_bbo",
        "queue_bucket": "00_0to1",
        "tod_bucket": "mid_0930_1200",
        "freshness_bucket": "fresh_le100ms",
    }


class WalkForwardProbabilityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sessions = [
            "20260101",
            "20260102",
            "20260103",
            "20260104",
            "20260105",
        ]
        self.config = WalkForwardProbabilityConfig(
            lookback_sessions=2,
            min_history_sessions=2,
            beta_prior_strength=0.0,
            min_product_orders=1,
            min_product_fills=1,
            min_product_dates=1,
            min_peer_orders=1,
            min_peer_fills=1,
            min_peer_dates=1,
            estimator_version="test",
        )

    def test_target_day_outcome_never_changes_target_snapshot(self) -> None:
        facts = pl.DataFrame(
            [
                _fact("20260102", fill=False),
                _fact("20260103", fill=True),
                _fact("20260104", fill=True),
            ]
        )
        result = build_walkforward_probability_tables(
            facts,
            self.sessions,
            self.config,
            asof_dates=["20260104"],
        ).product_state_probability
        all_state = result.filter(pl.col("state_family") == "all").row(
            0, named=True
        )
        self.assertEqual(all_state["train_start_date"], "20260102")
        self.assertEqual(all_state["train_end_date"], "20260103")
        self.assertEqual(all_state["n_orders"], 2)
        self.assertEqual(all_state["reported_p_full_fill"], 0.5)
        self.assertTrue(all_state["execution_safe_snapshot"])
        self.assertFalse(all_state["contains_target_day_outcome"])

        changed = facts.with_columns(
            pl.when(pl.col("Date") == "20260104")
            .then(pl.lit(False))
            .otherwise(pl.col("entry_full_fill"))
            .alias("entry_full_fill")
        )
        changed_result = build_walkforward_probability_tables(
            changed,
            self.sessions,
            self.config,
            asof_dates=["20260104"],
        ).product_state_probability
        changed_all = changed_result.filter(
            pl.col("state_family") == "all"
        ).row(0, named=True)
        self.assertEqual(
            all_state["reported_p_full_fill"],
            changed_all["reported_p_full_fill"],
        )

    def test_immature_terminal_label_is_excluded(self) -> None:
        facts = pl.DataFrame(
            [
                _fact("20260102", fill=False),
                _fact(
                    "20260103",
                    fill=True,
                    label_end_date="20260104",
                ),
            ]
        )
        result = build_walkforward_probability_tables(
            facts,
            self.sessions,
            self.config,
            asof_dates=["20260104"],
        ).product_state_probability
        all_state = result.filter(pl.col("state_family") == "all").row(
            0, named=True
        )
        self.assertEqual(all_state["n_orders"], 1)
        self.assertIsNone(all_state["reported_p_full_fill"])
        self.assertFalse(all_state["product_fill_support_gate"])
        self.assertEqual(all_state["label_cutoff_date"], "20260102")

    def test_unknown_post_fill_labels_are_not_reported_as_zero(self) -> None:
        rows = [
            _fact("20260102", fill=True),
            _fact("20260103", fill=True),
        ]
        for row in rows:
            row["hedge_label_observed"] = False
            row["hedge_priceable_50ms"] = False
            row["locked_threshold_qualified_50ms"] = False
            row["lower_label_known_30s"] = False
            row["lower_hit_30s"] = False
            row["hedge_slippage_bp_50ms"] = None
        result = build_walkforward_probability_tables(
            pl.DataFrame(rows),
            self.sessions,
            self.config,
            asof_dates=["20260104"],
        ).product_state_probability
        all_state = result.filter(pl.col("state_family") == "all").row(
            0, named=True
        )
        self.assertIsNone(all_state["reported_p_locked_given_fill"])
        self.assertIsNone(all_state["reported_p_lower_hit_given_fill"])
        self.assertEqual(all_state["locked_fallback_level"], "insufficient_support")

    def test_unsupported_peer_does_not_distort_supported_product(self) -> None:
        rows = [
            _fact("20260102", fill=True),
            _fact("20260103", fill=False),
        ]
        peer = _fact("20260103", fill=True)
        peer["ValueCode"] = "2303"
        rows.append(peer)
        config = WalkForwardProbabilityConfig(
            lookback_sessions=2,
            min_history_sessions=2,
            beta_prior_strength=50.0,
            min_product_orders=2,
            min_product_fills=1,
            min_product_dates=2,
            min_peer_orders=2,
            min_peer_fills=2,
            min_peer_dates=2,
            estimator_version="peer-test",
        )
        result = build_walkforward_probability_tables(
            pl.DataFrame(rows),
            self.sessions,
            config,
            asof_dates=["20260104"],
        ).product_state_probability
        product = result.filter(
            (pl.col("state_family") == "all")
            & (pl.col("ValueCode") == "2317")
        ).row(0, named=True)
        self.assertFalse(product["peer_fill_support_gate"])
        self.assertEqual(product["reported_p_full_fill"], 0.5)
        self.assertEqual(product["fill_fallback_level"], "product_empirical")

    def test_locked_forward_fails_without_verified_manifest(self) -> None:
        sessions = ["20260812", "20260813", "20260814"]
        config = WalkForwardProbabilityConfig(
            lookback_sessions=2,
            min_history_sessions=2,
            beta_prior_strength=0,
            min_product_orders=1,
            min_product_fills=1,
            min_product_dates=1,
            min_peer_orders=1,
            min_peer_fills=1,
            min_peer_dates=1,
            estimator_version="freeze-test",
        )
        with self.assertRaisesRegex(ValueError, "verified frozen manifest"):
            build_walkforward_probability_tables(
                pl.DataFrame(
                    [
                        _fact("20260812", fill=False),
                        _fact("20260813", fill=True),
                    ]
                ),
                sessions,
                config,
                WalkForwardSplit(),
                asof_dates=["20260814"],
            )


if __name__ == "__main__":
    unittest.main()
