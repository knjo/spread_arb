from __future__ import annotations

import unittest
from datetime import UTC, date, datetime

import polars as pl

from maker.src.quote_fill.dynamic_estimated_path_portfolio import (
    backtest_full_population_inventory,
    build_dynamic_entry_positions,
    label_dynamic_frozen_lower_paths,
    price_dynamic_terminal_paths,
    run_completed_only_cap_backtest,
    summarize_dynamic_path_coverage,
)


def _ns(value: str) -> int:
    return int(
        datetime.fromisoformat(value)
        .replace(tzinfo=UTC)
        .timestamp()
        * 1_000_000_000
    )


def _entries(rows: list[dict[str, object]]) -> pl.DataFrame:
    base: list[dict[str, object]] = []
    for row in rows:
        day = str(row["Date"])
        identifier = str(row["physical_order_id"])
        base.append(
            {
                "Date": day,
                "ValueCode": str(row["ValueCode"]),
                "QuoteCode": str(row["QuoteCode"]),
                "physical_order_id": identifier,
                "boundary_quantile": 95,
                "target_price": float(row.get("target_price", 100.0)),
                "submit_decision_time_ns": int(
                    row.get("submit_ns", _ns("2026-01-02T01:05:00"))
                ),
                "makerfill_implied_fill_time_ns": int(
                    row.get("fill_ns", _ns("2026-01-02T01:05:00.100"))
                ),
                "full_fill": True,
                "outcome_supported": True,
                "anchor_ewma_120s_bp": float(row.get("anchor", 20.0)),
                "upper_distance_bp": 10.0,
                "lower_distance_bp": float(row.get("lower", 10.0)),
                "contract_size": 2000.0,
                "end_date": row.get("end_date", date(2026, 1, 6)),
                "fill_outcome_exact_within_model": False,
                "fill_cursor_exact": False,
                "fill_backend": "makerfill_test",
            }
        )
    return pl.DataFrame(base)


def _hedges(entries: pl.DataFrame) -> pl.DataFrame:
    return entries.select(
        "physical_order_id",
        (
            pl.col("makerfill_implied_fill_time_ns") + 50_000_000
        ).alias("entry_hedge_decision_time_ns"),
        pl.lit(101.0).alias("entry_future_price"),
        pl.lit(True).alias("entry_hedge_executable"),
        pl.lit(0).alias("entry_hedge_depth_shortfall"),
        pl.lit(True).alias("entry_hedge_price_approximate"),
        pl.lit("raw_tape_plus_50ms_estimate").alias("entry_hedge_source"),
    )


def _fair(
    day: str,
    value_code: str,
    quote_code: str,
    rows: list[tuple[str, float, float, float]],
) -> pl.DataFrame:
    count = len(rows)
    return pl.DataFrame(
        {
            "Date": [day] * count,
            "ValueCode": [value_code] * count,
            "QuoteCode": [quote_code] * count,
            "timestamp": [datetime.fromisoformat(row[0]) for row in rows],
            "contract_size": [2000.0] * count,
            "spot_bid": [row[1] for row in rows],
            "spot_bid_lots": [2] * count,
            "fut_exec_ask": [row[2] for row in rows],
            "fut_exec_ask_lots": [1] * count,
            "basis_buy_taker_bp": [row[3] for row in rows],
            "analysis_eligible": [True] * count,
        },
        schema_overrides={"timestamp": pl.Datetime("ns")},
    )


class DynamicEstimatedPathPortfolioTest(unittest.TestCase):
    def test_actual_fast_schema_aliases_and_manifest_are_supported(self) -> None:
        source = _entries(
            [
                {
                    "Date": "20260102",
                    "ValueCode": "2330",
                    "QuoteCode": "CDFB6",
                    "physical_order_id": "p1",
                }
            ]
        )
        manifest = pl.DataFrame(
            {
                "Date": ["20260102"],
                "ValueCode": ["2330"],
                "QuoteCode": ["CDFB6"],
            }
        )
        result = build_dynamic_entry_positions(
            source, hedge_facts=_hedges(source), manifest=manifest
        )
        row = result.row(0, named=True)
        self.assertEqual(row["position_established_ns"], row["fill_ns"] + 50_000_000)
        self.assertEqual(row["exit_threshold_basis_bp"], 10.0)
        self.assertTrue(row["entry_pricing_supported"])
        self.assertTrue(row["entry_outcome_approximate"])
        self.assertFalse(row["fixed45_universe_used"])

    def test_missing_plus_50ms_price_fails_closed(self) -> None:
        source = _entries(
            [
                {
                    "Date": "20260102",
                    "ValueCode": "2330",
                    "QuoteCode": "CDFB6",
                    "physical_order_id": "p1",
                }
            ]
        )
        hedge = _hedges(source).with_columns(
            pl.lit(None, dtype=pl.Float64).alias("entry_future_price")
        )
        result = build_dynamic_entry_positions(source, hedge_facts=hedge)
        self.assertFalse(result.item(0, "entry_pricing_supported"))
        self.assertEqual(
            result.item(0, "entry_pricing_unsupported_reason"),
            "missing_plus_50ms_hedge_price",
        )

    def test_actual_dynamic_hedge_schema_is_adapted_with_slippage(self) -> None:
        source = _entries(
            [
                {
                    "Date": "20260102",
                    "ValueCode": "2330",
                    "QuoteCode": "CDFB6",
                    "physical_order_id": "p1",
                }
            ]
        )
        hedge = pl.DataFrame(
            {
                "physical_order_id": ["p1"],
                "status": ["executable"],
                "decision_time_ns": [
                    source.item(0, "makerfill_implied_fill_time_ns") + 50_000_000
                ],
                "executable_vwap_price": [101.0],
                "depth_shortfall": [0],
                "signed_latency_slippage_bp": [2.0],
                "signed_depth_slippage_bp": [3.0],
                "signed_total_slippage_bp": [5.0],
                "decision_book_age_ms": [12.5],
                "entry_fill_time_exact": [False],
                "hedge_version": ["dynamic_future_hedge_v1_analysis_only"],
            }
        )
        row = build_dynamic_entry_positions(
            source, hedge_facts=hedge
        ).row(0, named=True)
        self.assertTrue(row["entry_pricing_supported"])
        self.assertEqual(row["entry_hedge_status"], "executable")
        self.assertEqual(row["entry_hedge_signed_total_slippage_bp"], 5.0)
        self.assertTrue(row["entry_hedge_price_approximate"])
        self.assertEqual(
            row["entry_hedge_source"], "dynamic_future_hedge_v1_analysis_only"
        )

    def test_same_day_first_passage_starts_on_next_full_second(self) -> None:
        source = _entries(
            [
                {
                    "Date": "20260102",
                    "ValueCode": "2330",
                    "QuoteCode": "CDFB6",
                    "physical_order_id": "p1",
                }
            ]
        )
        positions = build_dynamic_entry_positions(source, hedge_facts=_hedges(source))
        fair = _fair(
            "20260102",
            "2330",
            "CDFB6",
            [
                ("2026-01-02T01:05:00", 100.5, 100.5, 0.0),
                ("2026-01-02T01:05:01", 100.4, 100.6, 20.0),
                ("2026-01-02T01:05:02", 100.6, 100.7, 9.0),
            ],
        )
        paths = label_dynamic_frozen_lower_paths(
            positions, fair, ["20260102", "20260105", "20260106"]
        )
        row = paths.row(0, named=True)
        self.assertEqual(row["path_status"], "same_day_frozen_lower_hit")
        self.assertEqual(row["exit_decision_time_ns"], _ns("2026-01-02T01:05:02"))
        self.assertAlmostEqual(row["gross_cycle_pnl_twd"], 1800.0)

    def test_cross_day_search_ignores_next_day_entry_universe(self) -> None:
        source = _entries(
            [
                {
                    "Date": "20260102",
                    "ValueCode": "2330",
                    "QuoteCode": "CDFB6",
                    "physical_order_id": "p1",
                }
            ]
        )
        # The manifest contains the product only on its entry day.  The carry
        # search must still use its exact contract on 20260105.
        manifest = pl.DataFrame(
            {
                "Date": ["20260102"],
                "ValueCode": ["2330"],
                "QuoteCode": ["CDFB6"],
            }
        )
        positions = build_dynamic_entry_positions(
            source, hedge_facts=_hedges(source), manifest=manifest
        )
        fair = pl.concat(
            [
                _fair(
                    "20260102",
                    "2330",
                    "CDFB6",
                    [("2026-01-02T05:20:00", 100.0, 101.0, 100.0)],
                ),
                _fair(
                    "20260105",
                    "2330",
                    "CDFB6",
                    [("2026-01-05T01:05:00", 100.4, 100.5, 9.0)],
                ),
            ]
        )
        row = label_dynamic_frozen_lower_paths(
            positions, fair, ["20260102", "20260105", "20260106"]
        ).row(0, named=True)
        self.assertEqual(row["path_status"], "cross_session_frozen_lower_hit")
        self.assertEqual(row["terminal_date"], "20260105")

    def test_expiry_uses_last_joint_grid_then_paired_proxy_mark(self) -> None:
        source = _entries(
            [
                {
                    "Date": "20260102",
                    "ValueCode": "2330",
                    "QuoteCode": "CDFB6",
                    "physical_order_id": "grid",
                    "end_date": date(2026, 1, 5),
                },
                {
                    "Date": "20260102",
                    "ValueCode": "2317",
                    "QuoteCode": "DHFB6",
                    "physical_order_id": "close",
                    "end_date": date(2026, 1, 5),
                },
            ]
        )
        positions = build_dynamic_entry_positions(source, hedge_facts=_hedges(source))
        fair = _fair(
            "20260105",
            "2330",
            "CDFB6",
            [
                ("2026-01-05T05:19:58", 99.5, 101.0, 150.0),
                ("2026-01-05T05:19:59", 99.6, 101.0, 140.0),
            ],
        )
        closes = pl.DataFrame(
            {
                "Date": ["20260105"],
                "ValueCode": ["2317"],
                "QuoteCode": ["DHFB6"],
                "spot_close_price": [101.0],
                "future_close_price": [100.5],
                "spot_close_is_official_daily_close_field": [True],
                "future_close_is_official_daily_close": [False],
                "future_close_is_official_settlement": [False],
                "future_close_is_last_trade_proxy": [True],
                "paired_mark_is_fully_official_close": [False],
            }
        )
        paths = label_dynamic_frozen_lower_paths(
            positions,
            fair,
            ["20260102", "20260105"],
            expiry_close_facts=closes,
        )
        by_id = {row["physical_order_id"]: row for row in paths.iter_rows(named=True)}
        self.assertEqual(
            by_id["grid"]["path_status"],
            "expiry_last_joint_executable_grid_estimated_nonofficial",
        )
        self.assertTrue(by_id["grid"]["expiry_last_joint_grid_fallback"])
        self.assertEqual(
            by_id["close"]["path_status"],
            "expiry_spot_close_future_last_trade_proxy_"
            "accounting_mark_non_executable",
        )
        self.assertTrue(by_id["close"]["expiry_spot_mark_is_official_close"])
        self.assertTrue(by_id["close"]["expiry_future_mark_is_last_trade_proxy"])
        self.assertFalse(
            by_id["close"]["expiry_paired_mark_is_fully_official_close"]
        )
        self.assertFalse(by_id["close"]["expiry_mark_is_executable"])

    def test_costs_coverage_and_cap_replay_are_explicitly_conditional(self) -> None:
        source = _entries(
            [
                {
                    "Date": "20260102",
                    "ValueCode": "2330",
                    "QuoteCode": "CDFB6",
                    "physical_order_id": "p1",
                }
            ]
        )
        positions = build_dynamic_entry_positions(source, hedge_facts=_hedges(source))
        fair = _fair(
            "20260102",
            "2330",
            "CDFB6",
            [("2026-01-02T01:05:01", 100.6, 100.7, 9.0)],
        )
        terminals = label_dynamic_frozen_lower_paths(
            positions, fair, ["20260102", "20260105", "20260106"]
        )
        priced = price_dynamic_terminal_paths(terminals)
        row = priced.row(0, named=True)
        expected_cost = (
            2000 * 100.0 * 0.000171
            + 2000 * 100.6 * 0.000171
            + 2000 * 100.6 * 0.0015
            + 2000 * 101.0 * 0.00002
            + 2000 * 100.7 * 0.00002
            + 40.0
        )
        self.assertAlmostEqual(row["total_transaction_cost_twd"], expected_cost)
        self.assertAlmostEqual(
            row["net_cycle_pnl_twd"], row["gross_cycle_pnl_twd"] - expected_cost
        )
        coverage = summarize_dynamic_path_coverage(priced).row(0, named=True)
        self.assertEqual(coverage["same_day_close_rate_all_filled_entries"], 1.0)
        cap = run_completed_only_cap_backtest(
            priced, ["20260102", "20260105", "20260106"]
        )
        assert cap is not None
        self.assertEqual(cap.summary.height, 5)
        self.assertTrue(all(cap.summary["completed_only_conditional_replay"]))
        self.assertFalse(any(cap.summary["full_population_portfolio_ev"]))

    def test_full_population_keeps_unresolved_inventory_and_blocks_next_day(self) -> None:
        source = _entries(
            [
                {
                    "Date": "20260102",
                    "ValueCode": "2330",
                    "QuoteCode": "CDFB6",
                    "physical_order_id": "open",
                    "end_date": date(2026, 1, 6),
                },
                {
                    "Date": "20260105",
                    "ValueCode": "2330",
                    "QuoteCode": "CDFB6",
                    "physical_order_id": "blocked",
                    "submit_ns": _ns("2026-01-05T01:05:00"),
                    "fill_ns": _ns("2026-01-05T01:05:00.100"),
                    "end_date": date(2026, 1, 6),
                },
            ]
        )
        positions = build_dynamic_entry_positions(source, hedge_facts=_hedges(source))
        # Both paths remain above frozen-lower through the observed horizon.
        fair = pl.concat(
            [
                _fair(
                    "20260102",
                    "2330",
                    "CDFB6",
                    [("2026-01-02T05:20:00", 100.0, 101.0, 100.0)],
                ),
                _fair(
                    "20260105",
                    "2330",
                    "CDFB6",
                    [("2026-01-05T05:20:00", 100.0, 101.0, 100.0)],
                ),
            ]
        )
        terminals = label_dynamic_frozen_lower_paths(
            positions, fair, ["20260102", "20260105", "20260106"]
        )
        self.assertTrue(all(terminals["path_status"] == "observation_horizon_open"))
        all_costed = price_dynamic_terminal_paths(terminals)
        result = backtest_full_population_inventory(
            all_costed, ["20260102", "20260105", "20260106"]
        )
        assert result is not None
        self.assertEqual(result.summary.height, 5)
        self.assertTrue(
            all(result.summary["accepted_unresolved_open_paths"] == 1)
        )
        self.assertTrue(
            all(result.summary["rejected_opening_carry_exit_only"] == 1)
        )
        day = result.daily.filter(
            (pl.col("hard_intraday_cap_twd") == 10_000_000.0)
            & (pl.col("Date") == "20260105")
        ).row(0, named=True)
        self.assertEqual(day["unresolved_eod_positions"], 1)
        self.assertEqual(day["new_spot_notional_twd"], 0.0)
        self.assertFalse(result.summary["full_population_portfolio_ev"].any())

    def test_cutoff_uses_fill_time_not_plus_50ms_establishment(self) -> None:
        source = _entries(
            [
                {
                    "Date": "20260102",
                    "ValueCode": "2330",
                    "QuoteCode": "CDFB6",
                    "physical_order_id": "cross-cutoff",
                    "submit_ns": _ns("2026-01-02T04:59:59.900"),
                    "fill_ns": _ns("2026-01-02T04:59:59.980"),
                }
            ]
        )
        positions = build_dynamic_entry_positions(source, hedge_facts=_hedges(source))
        self.assertGreater(
            positions.item(0, "position_established_ns"),
            _ns("2026-01-02T05:00:00"),
        )
        fair = _fair(
            "20260102",
            "2330",
            "CDFB6",
            [("2026-01-02T05:00:01", 100.6, 100.7, 9.0)],
        )
        terminals = label_dynamic_frozen_lower_paths(
            positions, fair, ["20260102", "20260105", "20260106"]
        )
        priced = price_dynamic_terminal_paths(terminals)
        full = backtest_full_population_inventory(
            priced, ["20260102", "20260105", "20260106"]
        )
        assert full is not None
        self.assertTrue(all(full.summary["accepted_filled_entries"] == 1))
        self.assertTrue(all(full.summary["rejected_entry_cutoff"] == 0))
        completed = run_completed_only_cap_backtest(
            priced, ["20260102", "20260105", "20260106"]
        )
        assert completed is not None
        self.assertTrue(all(completed.summary["accepted_paths"] == 1))
        self.assertTrue(all(completed.summary["rejected_entry_cutoff"] == 0))


if __name__ == "__main__":
    unittest.main()
