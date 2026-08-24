"""Tests for the causal pre-replay liquidity screen."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import polars as pl

from maker.src.quote_fill.liquidity import (
    LIQUIDITY_PUBLICATION_SCHEMA_VERSION,
    LiquidityScreenConfig,
    build_rolling_liquidity_screen,
    run_liquidity_screen_study,
    summarize_daily_liquidity,
    summarize_pseudo_validation_stability,
)
from maker.src.quote_fill.targets import (
    FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE,
    PRICE_LADDER_VERSION,
)
from maker.src.quote_width.rolling import (
    RollingBoundaryConfig,
    _partitioned_daily_source_provenance,
    write_rolling_boundary_snapshots,
)


class DailyLiquidityTest(unittest.TestCase):
    def test_spreads_depth_freshness_and_activity(self) -> None:
        panel = pl.DataFrame(
            {
                "Date": ["20260101"] * 3,
                "ValueCode": ["2317"] * 3,
                "QuoteCode": ["DHFA6"] * 3,
                "seconds_from_open": [300, 301, 302],
                "spot_bid": [50.0] * 3,
                "spot_ask": [50.1] * 3,
                "fut_bid": [50.0] * 3,
                # Regular L1 is wider than executable Best; screen must use
                # the price actually available to the future taker route.
                "fut_ask": [50.3] * 3,
                "fut_exec_bid": [50.0] * 3,
                "fut_exec_ask": [50.2] * 3,
                "spot_ask_lots": [2, 1, 3],
                "fut_exec_bid_lots": [1, 1, 0],
                "contract_size": [2000.0] * 3,
                "spot_sequence": [1, 1, 2],
                "fut_sequence": [10, 11, 11],
                "eligible_base": [True] * 3,
                "eligible_1000ms": [True, False, True],
                "eligible_5000ms": [True, True, True],
                "basis_sell_taker_bp": [10.0] * 3,
                "basis_buy_taker_bp": [40.0] * 3,
            }
        )
        row = summarize_daily_liquidity(panel).row(0, named=True)
        self.assertEqual(row["spot_spread_ticks_p50"], 1.0)
        self.assertEqual(row["fut_spread_ticks_p50"], 2.0)
        self.assertEqual(row["tt_band_bp_p50"], 30.0)
        self.assertAlmostEqual(row["fresh_1000ms_rate"], 2 / 3)
        self.assertAlmostEqual(row["p_spot_buy_one_unit_l1_depth"], 2 / 3)
        self.assertAlmostEqual(row["p_fut_bid_depth_ge1contract"], 2 / 3)
        self.assertEqual(row["p_spot_one_unit_given_fresh1000"], 1.0)
        self.assertEqual(row["p_fut_one_unit_given_fresh1000"], 0.5)
        self.assertTrue(row["contains_target_day_outcome"])

    def test_null_sequences_do_not_create_fake_activity(self) -> None:
        panel = pl.DataFrame(
            {
                "Date": ["20260101"] * 2,
                "ValueCode": ["2317"] * 2,
                "QuoteCode": ["DHFA6"] * 2,
                "seconds_from_open": [300, 301],
                "spot_bid": [50.0] * 2,
                "spot_ask": [50.1] * 2,
                "fut_bid": [50.0] * 2,
                "fut_ask": [50.1] * 2,
                "fut_exec_bid": [50.0] * 2,
                "fut_exec_ask": [50.1] * 2,
                "spot_ask_lots": [2] * 2,
                "fut_exec_bid_lots": [1] * 2,
                "contract_size": [2000.0] * 2,
                "spot_sequence": [None, None],
                "fut_sequence": [None, None],
                "eligible_base": [False] * 2,
                "eligible_1000ms": [False] * 2,
                "eligible_5000ms": [False] * 2,
                "basis_sell_taker_bp": [None, None],
                "basis_buy_taker_bp": [None, None],
            }
        )
        row = summarize_daily_liquidity(panel).row(0, named=True)
        self.assertEqual(row["spot_changed_seconds"], 0)
        self.assertEqual(row["fut_changed_seconds"], 0)

    def test_high_price_future_spread_uses_effective_date_ladder(self) -> None:
        panel = pl.DataFrame(
            {
                "Date": ["20260703", "20260706"],
                "ValueCode": ["2308", "2308"],
                "QuoteCode": ["FRFG6", "FRFG6"],
                "seconds_from_open": [300, 300],
                "spot_bid": [2130.0, 2130.0],
                "spot_ask": [2135.0, 2135.0],
                "fut_bid": [2130.0, 2130.0],
                "fut_ask": [2135.0, 2131.0],
                "fut_exec_bid": [2130.0, 2130.0],
                "fut_exec_ask": [2135.0, 2131.0],
                "spot_ask_lots": [2, 2],
                "fut_exec_bid_lots": [1, 1],
                "contract_size": [2000.0, 2000.0],
                "spot_sequence": [1, 1],
                "fut_sequence": [1, 1],
                "eligible_base": [True, True],
                "eligible_1000ms": [True, True],
                "eligible_5000ms": [True, True],
                "basis_sell_taker_bp": [10.0, 10.0],
                "basis_buy_taker_bp": [20.0, 20.0],
            }
        )
        rows = summarize_daily_liquidity(panel).sort("Date")
        self.assertEqual(rows["spot_spread_ticks_p50"].to_list(), [1.0, 1.0])
        self.assertEqual(rows["fut_spread_ticks_p50"].to_list(), [1.0, 1.0])


def _daily(date: str, future_spread: float) -> dict[str, object]:
    return {
        "Date": date,
        "ValueCode": "2317",
        "eligible_grid_rate": 0.9,
        "fresh_1000ms_rate": 0.8,
        "fresh_5000ms_rate": 0.9,
        "spot_spread_ticks_p50": 1.0,
        "spot_spread_ticks_p95": 1.0,
        "fut_spread_ticks_p50": future_spread,
        "fut_spread_ticks_p95": future_spread,
        "tt_band_bp_p50": 30.0,
        "tt_band_bp_p95": 50.0,
        "tt_band_fresh1000_bp_p50": 30.0,
        "tt_band_fresh5000_bp_p50": 30.0,
        "fresh_1000ms_rows": 100,
        "fresh_5000ms_rows": 200,
        "p_spot_buy_one_unit_l1_depth": 0.9,
        "p_spot_one_unit_given_fresh1000": 0.9,
        "p_fut_bid_depth_ge1contract": 0.9,
        "p_fut_one_unit_given_fresh1000": 0.9,
        "spot_changed_seconds_per_hour": 100.0,
        "fut_changed_seconds_per_hour": 100.0,
        "spot_changed_second_rate": 0.1,
        "fut_changed_second_rate": 0.1,
    }


class RollingLiquidityScreenTest(unittest.TestCase):
    def test_wide_taker_leg_is_route_specific_and_target_day_is_excluded(self) -> None:
        sessions = ["20260101", "20260102", "20260103"]
        daily = pl.DataFrame(
            [
                _daily("20260101", 5.0),
                _daily("20260102", 5.0),
                # Target-day liquidity must not affect the target snapshot.
                _daily("20260103", 1.0),
            ]
        )
        boundaries = pl.DataFrame(
            {
                "Date": ["20260103"],
                "ValueCode": ["2317"],
                "QuoteCode": ["DHFA6"],
                "boundary_quantile": [50],
                "upper_distance_bp": [20.0],
                "lower_distance_bp": [20.0],
                "adaptive_parameter_valid": [True],
                "source_asof_date": ["20260102"],
                "contains_target_day_outcome": [False],
                "boundary_role": ["rolling_latent_candidate"],
                "execution_safe_snapshot": [True],
                "parameter_version": ["test-boundary"],
                "target_day_realized_pnl": [999.0],
            }
        )
        config = LiquidityScreenConfig(
            lookback_sessions=2,
            recent_sessions=2,
            min_history_sessions=2,
            min_recent_sessions=2,
            eligible_median_warning_threshold=0.5,
            eligible_q10_warning_threshold=0.5,
            min_fresh_1000ms_rate_median=0.1,
            min_fresh_5000ms_rate_median=0.1,
            min_unit_depth_rate_median=0.8,
            min_maker_changed_second_rate_median=0.01,
            wide_hedge_spread_p50_ticks=2.0,
            wide_hedge_spread_p95_ticks=4.0,
        )
        result = build_rolling_liquidity_screen(
            daily, boundaries, sessions, config
        )
        future_maker = result.filter(
            pl.col("route") == "future_ask_spot_taker"
        ).row(0, named=True)
        spot_maker = result.filter(
            pl.col("route") == "spot_bid_future_taker"
        ).row(0, named=True)

        self.assertFalse(future_maker["wide_hedge_spread_flag"])
        self.assertTrue(future_maker["pre_replay_candidate"])
        self.assertTrue(spot_maker["wide_hedge_spread_flag"])
        # Wide taker spread is a diagnostic stratum, not a hard rejection:
        # maker/taker EV must be measured in raw replay.
        self.assertTrue(spot_maker["pre_replay_candidate"])
        self.assertEqual(spot_maker["liquidity_train_end_date"], "20260102")
        self.assertFalse(spot_maker["contains_target_day_outcome"])
        self.assertNotIn("target_day_realized_pnl", result.columns)

    def test_q10_warning_never_rejects_an_otherwise_valid_product(self) -> None:
        sessions = ["20260101", "20260102", "20260103"]
        first = _daily("20260101", 1.0)
        second = _daily("20260102", 1.0)
        first["eligible_grid_rate"] = 0.6
        boundaries = pl.DataFrame(
            {
                "Date": ["20260103"],
                "ValueCode": ["2317"],
                "QuoteCode": ["DHFA6"],
                "boundary_quantile": [50],
                "upper_distance_bp": [20.0],
                "lower_distance_bp": [20.0],
                "adaptive_parameter_valid": [True],
                "source_asof_date": ["20260102"],
                "contains_target_day_outcome": [False],
                "boundary_role": ["rolling_latent_candidate"],
                "execution_safe_snapshot": [True],
                "parameter_version": ["test-boundary"],
            }
        )
        config = LiquidityScreenConfig(
            lookback_sessions=2,
            recent_sessions=2,
            min_history_sessions=2,
            min_recent_sessions=2,
            eligible_median_warning_threshold=0.5,
            eligible_q10_warning_threshold=0.7,
            min_fresh_1000ms_rate_median=0.1,
            min_fresh_5000ms_rate_median=0.1,
            min_unit_depth_rate_median=0.8,
            min_maker_changed_second_rate_median=0.01,
        )
        result = build_rolling_liquidity_screen(
            pl.DataFrame([first, second]), boundaries, sessions, config
        )
        self.assertTrue(result["eligible_q10_only_warning"].all())
        self.assertTrue(result["eligible_q10_stress_flag"].all())
        self.assertTrue((result["replay_tier"] == "core_candidate").all())
        self.assertTrue(result["exploration_replay_eligible"].all())
        self.assertTrue(result["pre_replay_candidate"].all())
        self.assertTrue(
            (result["liquidity_gate_status"] == "pass").all()
        )

        median_warning = build_rolling_liquidity_screen(
            pl.DataFrame([first, second]),
            boundaries,
            sessions,
            LiquidityScreenConfig(
                lookback_sessions=2,
                recent_sessions=2,
                min_history_sessions=2,
                min_recent_sessions=2,
                eligible_median_warning_threshold=0.8,
                eligible_q10_warning_threshold=0.1,
                min_fresh_1000ms_rate_median=0.1,
                min_fresh_5000ms_rate_median=0.1,
                min_unit_depth_rate_median=0.8,
                min_maker_changed_second_rate_median=0.01,
            ),
        )
        self.assertTrue(median_warning["eligible_median_stress_flag"].all())
        self.assertTrue(median_warning["pre_replay_candidate"].all())
        self.assertTrue(
            (median_warning["liquidity_gate_status"] == "pass").all()
        )

    def test_pseudo_stable_core_requires_both_routes(self) -> None:
        rows: list[dict[str, object]] = []
        for date in ("20260701", "20260702"):
            for route in ("future_ask_spot_taker", "spot_bid_future_taker"):
                rows.append(
                    {
                        "Date": date,
                        "ValueCode": "2317",
                        "route": route,
                        "boundary_quantile": 50,
                        "support_gate": True,
                        "liquidity_gate_status": "pass",
                        "replay_tier": "core_candidate",
                        "eligible_q10_only_warning": False,
                        "eligible_median_stress_flag": False,
                        "maker_spread_ticks_p50": 1.0,
                        "maker_spread_ticks_p95": 1.0,
                        "hedge_spread_ticks_p50": 1.0,
                        "hedge_spread_ticks_p95": 1.0,
                        "fresh_1000ms_rate_day_median": 0.5,
                        "unit_hedge_depth_rate": 1.0,
                    }
                )
        boundaries = pl.DataFrame(
            [
                {
                    "Date": "20260702",
                    "ValueCode": "2317",
                    "QuoteCode": "DHFG6",
                    "boundary_quantile": quantile,
                    "upper_distance_bp": float(quantile) / 10,
                    "lower_distance_bp": float(quantile) / 10,
                    "upper_distance_future_ticks": 0.5,
                    "lower_distance_future_ticks": 0.5,
                }
                for quantile in (50, 80, 95)
            ]
        )
        route, stable = summarize_pseudo_validation_stability(
            pl.DataFrame(rows),
            boundaries,
            LiquidityScreenConfig(
                pseudo_validation_start_date="20260701",
                min_stability_sessions=2,
                stable_core_rate=0.8,
            ),
        )
        self.assertEqual(route.height, 2)
        self.assertEqual(stable["ValueCode"].to_list(), ["2317"])
        self.assertEqual(stable["qualifying_routes"].item(), 2)
        self.assertFalse(stable["production_universe_approved"].item())


class LiquidityPublicationTest(unittest.TestCase):
    def _write_valid_sources(self, root: Path) -> tuple[Path, Path, pl.DataFrame]:
        daily_root = root / "daily"
        for date in ("20260101", "20260102", "20260103"):
            partition = daily_root / f"Date={date}"
            partition.mkdir(parents=True)
            (partition / "complete.json").write_text(
                json.dumps({"date": date}) + "\n",
                encoding="utf-8",
            )
        boundary_dir = root / "rolling_boundaries"
        boundary_config = RollingBoundaryConfig(
            lookback_sessions=2,
            min_history_sessions=2,
            min_excursion_history_sessions_per_side=1,
            min_completed_excursions_per_side=1,
            parameter_version="rolling-test-v1",
        )
        boundaries = pl.DataFrame(
            {
                "Date": ["20260103"] * 3,
                "ValueCode": ["2317"] * 3,
                "QuoteCode": ["DHFA6"] * 3,
                "boundary_quantile": [50, 80, 95],
                "boundary_role": [
                    "rolling_latent_candidate",
                    "rolling_latent_candidate",
                    "tail_diagnostic",
                ],
                "upper_distance_bp": [20.0, 30.0, 40.0],
                "lower_distance_bp": [20.0, 30.0, 40.0],
                "adaptive_parameter_valid": [True] * 3,
                "source_asof_date": ["20260102"] * 3,
                "contains_target_day_outcome": [False] * 3,
                "execution_safe_snapshot": [True] * 3,
                "parameter_version": [boundary_config.parameter_version] * 3,
                "lookback_sessions": [boundary_config.lookback_sessions] * 3,
                "price_ladder_version": [PRICE_LADDER_VERSION] * 3,
                "future_one_dollar_tick_effective_date": [
                    FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE
                ] * 3,
            }
        )
        write_rolling_boundary_snapshots(
            boundaries,
            boundary_dir,
            boundary_config,
            source_provenance=_partitioned_daily_source_provenance(daily_root),
        )
        daily = pl.DataFrame(
            [
                _daily("20260101", 1.0),
                _daily("20260102", 1.0),
                _daily("20260103", 1.0),
            ]
        ).with_columns(
            pl.lit(PRICE_LADDER_VERSION).alias("price_ladder_version"),
            pl.lit(FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE).alias(
                "future_one_dollar_tick_effective_date"
            ),
        )
        return (
            daily_root,
            boundary_dir / "rolling_boundary_snapshots.parquet",
            daily,
        )

    def test_reuse_requires_current_atomic_cache_lineage(self) -> None:
        config = LiquidityScreenConfig(
            lookback_sessions=2,
            recent_sessions=2,
            min_history_sessions=2,
            min_recent_sessions=2,
            eligible_median_warning_threshold=0.5,
            eligible_q10_warning_threshold=0.5,
            min_fresh_1000ms_rate_median=0.1,
            min_fresh_5000ms_rate_median=0.1,
            min_unit_depth_rate_median=0.5,
            min_maker_changed_second_rate_median=0.01,
        )
        module = "maker.src.quote_fill.liquidity"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            daily_root, boundary_path, daily = self._write_valid_sources(root)
            output = root / "liquidity"
            with patch(
                f"{module}.summarize_partitioned_daily_liquidity",
                return_value=daily,
            ) as source_loader:
                first_daily, first_screen = run_liquidity_screen_study(
                    daily_root=daily_root,
                    rolling_boundary_path=boundary_path,
                    output_dir=output,
                    config=config,
                )
            source_loader.assert_called_once()
            marker = json.loads(
                (output / "complete.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                marker["schema_version"],
                LIQUIDITY_PUBLICATION_SCHEMA_VERSION,
            )
            self.assertEqual(
                marker["artifacts"]["daily_liquidity.parquet"]["rows"],
                first_daily.height,
            )

            with patch(
                f"{module}.summarize_partitioned_daily_liquidity",
                side_effect=AssertionError("raw daily source must not open"),
            ) as source_loader:
                reused_daily, reused_screen = run_liquidity_screen_study(
                    daily_root=daily_root,
                    rolling_boundary_path=boundary_path,
                    output_dir=output,
                    config=config,
                    reuse_daily_liquidity=True,
                )
            source_loader.assert_not_called()
            self.assertEqual(reused_daily.shape, first_daily.shape)
            self.assertEqual(reused_screen.shape, first_screen.shape)

            # A v1/unversioned marker must fail instead of silently reading the
            # schema-compatible old parquet or falling back to daily raw facts.
            (output / "complete.json").write_text(
                json.dumps({"publication_version": "legacy-v1"}) + "\n",
                encoding="utf-8",
            )
            with patch(
                f"{module}.summarize_partitioned_daily_liquidity",
                side_effect=AssertionError("raw daily source must not open"),
            ) as source_loader:
                with self.assertRaisesRegex(
                    ValueError,
                    "incomplete|unsupported liquidity cache schema",
                ):
                    run_liquidity_screen_study(
                        daily_root=daily_root,
                        rolling_boundary_path=boundary_path,
                        output_dir=output,
                        config=config,
                        reuse_daily_liquidity=True,
                    )
            source_loader.assert_not_called()

    def test_unmarked_rolling_rejected_before_daily_source_io(self) -> None:
        module = "maker.src.quote_fill.liquidity"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            boundary = root / "rolling_boundary_snapshots.parquet"
            pl.DataFrame({"legacy": [1]}).write_parquet(boundary)
            with patch(
                f"{module}.summarize_partitioned_daily_liquidity",
                side_effect=AssertionError("daily source must not open"),
            ) as source_loader:
                with self.assertRaisesRegex(
                    FileNotFoundError,
                    "publication is incomplete",
                ):
                    run_liquidity_screen_study(
                        daily_root=root / "daily",
                        rolling_boundary_path=boundary,
                        output_dir=root / "liquidity",
                    )
            source_loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
