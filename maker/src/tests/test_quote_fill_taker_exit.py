from __future__ import annotations

from datetime import date, datetime, timezone
import unittest

import polars as pl

from maker.src.quote_fill.taker_exit import (
    ExecutionCostProfile,
    TakerExitConfig,
    apply_execution_cost_profile,
    build_physical_entry_positions,
    label_taker_exit_paths,
    summarize_taker_exit_paths,
)


def _ns(value: str) -> int:
    return int(
        datetime.fromisoformat(value)
        .replace(tzinfo=timezone.utc)
        .timestamp()
        * 1_000_000_000
    )


class TakerExitTest(unittest.TestCase):
    def test_aliases_collapse_and_route_prices_are_correct(self) -> None:
        aliases = pl.DataFrame(
            {
                "Date": ["20260102", "20260102"],
                "ValueCode": ["2330", "2330"],
                "QuoteCode": ["CDFB6", "CDFB6"],
                "route": ["spot_bid_future_taker"] * 2,
                "raw_order_fact_id": ["r1", "r1"],
                "boundary_quantile": [50, 80],
                "target_price": [100.0, 100.0],
                "full_fill": [True, True],
            }
        )
        hedge = pl.DataFrame(
            {
                "raw_order_fact_id": ["r1"],
                "status": ["executable"],
                "executable_vwap_price": [101.0],
                "decision_time_ns": [_ns("2026-01-02T01:05:00.100")],
                "contract_size_shares": [2000],
                "depth_shortfall": [0],
            }
        )
        result = build_physical_entry_positions(aliases, hedge)
        self.assertEqual(result.height, 1)
        self.assertEqual(
            result["boundary_quantile_aliases"][0].to_list(), [50, 80]
        )
        self.assertEqual(result.item(0, "entry_spot_price"), 100.0)
        self.assertEqual(result.item(0, "entry_future_price"), 101.0)

    def test_same_day_first_passage_starts_at_next_full_second(self) -> None:
        positions = self._positions("spot_bid_future_taker")
        fair = self._fair(
            "20260102",
            [
                ("2026-01-02T01:05:00", 100.5, 100.5),
                ("2026-01-02T01:05:01", 100.1, 100.8),
                ("2026-01-02T01:05:02", 100.6, 100.7),
            ],
        )
        labels = label_taker_exit_paths(
            positions,
            fair,
            ["20260102", "20260105"],
            TakerExitConfig(
                gross_profit_targets_twd=(1000.0,),
                samples=("base",),
                carry_to_next_session=False,
            ),
        )
        row = labels.row(0, named=True)
        # t=00 is before the position's 100ms hedge decision and cannot hit.
        self.assertEqual(row["status"], "same_day_target_exit")
        self.assertEqual(row["exit_timestamp_ns"], _ns("2026-01-02T01:05:02"))
        self.assertAlmostEqual(row["gross_cash_pnl_twd"], 1800.0)

    def test_no_same_day_hit_carries_exact_contract_to_next_session(self) -> None:
        positions = self._positions("future_ask_spot_taker")
        fair = pl.concat(
            [
                self._fair(
                    "20260102",
                    [("2026-01-02T05:19:59", 100.0, 101.0)],
                ),
                self._fair(
                    "20260105",
                    [("2026-01-05T01:05:00", 100.2, 100.8)],
                ),
            ]
        )
        labels = label_taker_exit_paths(
            positions,
            fair,
            ["20260102", "20260105"],
            TakerExitConfig(
                gross_profit_targets_twd=(2000.0,), samples=("base",)
            ),
        )
        row = labels.row(0, named=True)
        self.assertEqual(row["status"], "next_session_forced_exit")
        self.assertFalse(row["entry_contract_expiry_day"])
        self.assertEqual(row["exit_date"], "20260105")
        self.assertEqual(row["overnight_sessions"], 1)
        self.assertAlmostEqual(row["gross_cash_pnl_twd"], 800.0)

    def test_expiry_day_forces_same_day_exit_instead_of_carrying_contract(self) -> None:
        positions = self._positions("future_ask_spot_taker")
        fair = pl.concat(
            [
                self._fair(
                    "20260102",
                    [
                        ("2026-01-02T05:19:58", 100.1, 101.0),
                        ("2026-01-02T05:19:59", 100.2, 101.0),
                    ],
                    end_date=date(2026, 1, 2),
                ),
                # The same QuoteCode must never be silently carried through
                # expiry, even if a synthetic next-session row is present.
                self._fair(
                    "20260105",
                    [("2026-01-05T01:05:00", 101.0, 100.0)],
                    end_date=date(2026, 1, 2),
                ),
            ]
        )
        labels = label_taker_exit_paths(
            positions,
            fair,
            ["20260102", "20260105"],
            TakerExitConfig(
                gross_profit_targets_twd=(5000.0,), samples=("base",)
            ),
        )
        row = labels.row(0, named=True)
        self.assertEqual(row["status"], "expiry_day_forced_exit")
        self.assertTrue(row["entry_contract_expiry_day"])
        self.assertEqual(row["exit_date"], "20260102")
        self.assertEqual(row["overnight_sessions"], 0)
        self.assertEqual(row["exit_timestamp_ns"], _ns("2026-01-02T05:19:59"))

    def test_cost_profile_uses_daytrade_only_on_same_day_branch(self) -> None:
        labels = pl.DataFrame(
            {
                "status": ["same_day_target_exit", "next_session_forced_exit"],
                "contract_size_shares": [2000, 2000],
                "entry_spot_price": [100.0, 100.0],
                "entry_future_price": [101.0, 101.0],
                "exit_spot_price": [100.5, 100.5],
                "exit_future_price": [100.5, 100.5],
                "gross_cash_pnl_twd": [2000.0, 2000.0],
                "overnight_sessions": [0, 1],
            }
        )
        priced = apply_execution_cost_profile(
            labels,
            ExecutionCostProfile(
                "test",
                spot_commission_rate=0.0,
                spot_daytrade_sell_tax_rate=0.001,
                spot_regular_sell_tax_rate=0.002,
                futures_tax_rate=0.0,
                futures_commission_twd_per_contract_side=0.0,
                overnight_carry_cost_twd_per_session=50.0,
            ),
        )
        self.assertAlmostEqual(priced.item(0, "spot_tax_twd"), 201.0)
        self.assertAlmostEqual(priced.item(1, "spot_tax_twd"), 402.0)
        self.assertAlmostEqual(priced.item(1, "overnight_carry_cost_twd"), 50.0)
        self.assertAlmostEqual(priced.item(0, "net_cash_pnl_twd"), 1799.0)
        self.assertAlmostEqual(priced.item(1, "net_cash_pnl_twd"), 1548.0)

    def test_summary_keeps_mutually_exclusive_branch_denominator(self) -> None:
        labels = pl.DataFrame(
            {
                "ValueCode": ["2330"] * 3,
                "route": ["spot_bid_future_taker"] * 3,
                "sample": ["base"] * 3,
                "gross_profit_target_twd": [500.0] * 3,
                "status": [
                    "same_day_target_exit",
                    "next_session_forced_exit",
                    "unresolved_no_next_book",
                ],
                "gross_cash_pnl_twd": [800.0, -200.0, None],
                "holding_seconds": [10.0, 90_000.0, None],
                "max_same_day_taker_gross_twd": [800.0, 100.0, None],
            }
        )
        summary = summarize_taker_exit_paths(labels).row(0, named=True)
        self.assertEqual(summary["filled_hedged_positions"], 3)
        self.assertAlmostEqual(
            summary["p_same_day_target_exit_given_filled_hedged"], 1 / 3
        )
        self.assertAlmostEqual(
            summary["p_next_session_forced_exit_given_filled_hedged"], 1 / 3
        )
        self.assertAlmostEqual(summary["p_unresolved_given_filled_hedged"], 1 / 3)
        self.assertFalse(summary["conditional_exit_ev_complete"])

    @staticmethod
    def _positions(route: str) -> pl.DataFrame:
        if route == "future_ask_spot_taker":
            spot, future = 100.0, 101.0
        else:
            spot, future = 100.0, 101.0
        return pl.DataFrame(
            {
                "raw_order_fact_id": ["r1"],
                "Date": ["20260102"],
                "ValueCode": ["2330"],
                "QuoteCode": ["CDFB6"],
                "route": [route],
                "boundary_quantile_aliases": [[50]],
                "policy_alias_count": [1],
                "position_established_ns": [_ns("2026-01-02T01:05:00.100")],
                "contract_size_shares": [2000],
                "entry_spot_price": [spot],
                "entry_future_price": [future],
            },
            schema_overrides={"policy_alias_count": pl.UInt32},
        )

    @staticmethod
    def _fair(
        date_value: str,
        rows: list[tuple[str, float, float]],
        *,
        end_date: date = date(2026, 1, 21),
    ) -> pl.DataFrame:
        count = len(rows)
        return pl.DataFrame(
            {
                "Date": [date_value] * count,
                "ValueCode": ["2330"] * count,
                "QuoteCode": ["CDFB6"] * count,
                "timestamp": [datetime.fromisoformat(row[0]) for row in rows],
                "seconds_from_open": [300 + index for index in range(count)],
                "spot_bid": [row[1] for row in rows],
                "spot_bid_lots": [2] * count,
                "fut_exec_ask": [row[2] for row in rows],
                "fut_exec_ask_lots": [1] * count,
                "contract_size": [2000.0] * count,
                "end_date": [end_date] * count,
                "eligible_base": [True] * count,
                "eligible_1000ms": [True] * count,
            },
            schema_overrides={"timestamp": pl.Datetime("ns")},
        )


if __name__ == "__main__":
    unittest.main()
