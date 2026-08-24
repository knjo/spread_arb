from __future__ import annotations

from datetime import time
import unittest

import polars as pl

from maker.src.quote_fill.portfolio_cap_backtester import (
    DEFAULT_HARD_INTRADAY_CAPS_TWD,
    PortfolioCapBacktestConfig,
    PortfolioCapScenario,
    backtest_priced_paths,
    local_session_timestamp_ns,
)


D1 = "20260102"
D2 = "20260105"
D3 = "20260106"


def _ns(date: str, hour: int, minute: int = 0) -> int:
    return local_session_timestamp_ns(date, time(hour, minute))


def _path(
    identifier: str,
    *,
    date: str = D1,
    entry_ns: int | None = None,
    terminal_date: str | None = None,
    exit_ns: int | None = None,
    product: str = "A",
    notional: float = 40.0,
    gross: float = 10.0,
    cost: float = 2.0,
) -> dict[str, object]:
    entry_ns = _ns(date, 9, 0) if entry_ns is None else entry_ns
    terminal_date = date if terminal_date is None else terminal_date
    exit_ns = _ns(terminal_date, 10, 0) if exit_ns is None else exit_ns
    return {
        "Date": date,
        "ValueCode": product,
        "policy_path_id": identifier,
        "position_established_ns": entry_ns,
        "normalization_notional_twd": notional,
        "terminal_date": terminal_date,
        "exit_decision_time_ns": exit_ns,
        "entry_spot_price": notional,
        "entry_future_price": notional * 1.01,
        "entry_contract_size_shares": 1,
        "exit_spot_price": notional,
        "exit_future_price": notional * 1.01,
        "gross_cycle_pnl_twd": gross,
        "total_transaction_cost_twd": cost,
        "net_cycle_pnl_twd": gross - cost,
        "terminal_cashflow_priced": True,
    }


def _frame(*rows: dict[str, object]) -> pl.DataFrame:
    return pl.from_dicts(
        rows,
        infer_schema_length=None,
        schema_overrides={
            "position_established_ns": pl.Int64,
            "exit_decision_time_ns": pl.Int64,
            "normalization_notional_twd": pl.Float64,
            "entry_spot_price": pl.Float64,
            "entry_future_price": pl.Float64,
            "entry_contract_size_shares": pl.Int64,
            "exit_spot_price": pl.Float64,
            "exit_future_price": pl.Float64,
            "gross_cycle_pnl_twd": pl.Float64,
            "total_transaction_cost_twd": pl.Float64,
            "net_cycle_pnl_twd": pl.Float64,
        },
    )


def _config(
    hard_cap: float,
    *,
    product_fraction: float = 1.0,
    eod_limit: float | None = None,
) -> PortfolioCapBacktestConfig:
    return PortfolioCapBacktestConfig(
        scenarios=(
            PortfolioCapScenario(
                "test",
                hard_intraday_cap_twd=hard_cap,
                eod_overnight_limit_twd=eod_limit,
            ),
        ),
        per_product_fraction=product_fraction,
    )


class PortfolioCapBacktesterTest(unittest.TestCase):
    def test_equal_timestamp_entry_precedes_release_and_later_entry_reuses_cap(
        self,
    ) -> None:
        ten = _ns(D1, 10, 0)
        paths = _frame(
            _path("old", notional=70.0, exit_ns=ten),
            _path(
                "equal-rejected",
                product="B",
                notional=40.0,
                entry_ns=ten,
                exit_ns=_ns(D1, 10, 30),
            ),
            _path(
                "later-accepted",
                product="C",
                notional=40.0,
                entry_ns=ten + 1,
                exit_ns=_ns(D1, 11, 0),
            ),
        )

        result = backtest_priced_paths(
            paths, config=_config(100.0), session_dates=(D1,)
        )
        entries = {
            row["policy_path_id"]: row
            for row in result.events.filter(
                pl.col("event_type") == "entry_candidate"
            ).iter_rows(named=True)
        }
        self.assertEqual(entries["old"]["entry_admission_status"], "accepted")
        self.assertEqual(
            entries["equal-rejected"]["entry_admission_status"],
            "rejected_portfolio_cap",
        )
        self.assertEqual(
            entries["later-accepted"]["entry_admission_status"], "accepted"
        )
        event_order = [
            (row["event_type"], row["policy_path_id"])
            for row in result.events.iter_rows(named=True)
        ]
        self.assertEqual(
            event_order,
            [
                ("entry_candidate", "old"),
                ("entry_candidate", "equal-rejected"),
                ("position_exit", "old"),
                ("entry_candidate", "later-accepted"),
                ("position_exit", "later-accepted"),
            ],
        )
        summary = result.summary.row(0, named=True)
        self.assertEqual(summary["accepted_paths"], 2)
        self.assertEqual(summary["rejected_portfolio_cap"], 1)
        self.assertEqual(summary["peak_intraday_one_way_notional_twd"], 70.0)

    def test_cutoff_is_exclusive_and_eod_limit_does_not_reject_entry(self) -> None:
        paths = _frame(
            _path(
                "overnight",
                entry_ns=_ns(D1, 12, 59),
                terminal_date=D2,
                exit_ns=_ns(D2, 10, 0),
                notional=60.0,
            ),
            _path(
                "product-rejected",
                entry_ns=_ns(D1, 12, 59) + 1,
                terminal_date=D2,
                exit_ns=_ns(D2, 10, 1),
                notional=10.0,
            ),
            _path(
                "cutoff-rejected",
                product="B",
                entry_ns=_ns(D1, 13, 0),
                terminal_date=D2,
                exit_ns=_ns(D2, 10, 2),
                notional=10.0,
            ),
        )
        result = backtest_priced_paths(
            paths,
            config=_config(200.0, product_fraction=0.30, eod_limit=50.0),
            session_dates=(D1, D2),
        )
        entries = {
            row["policy_path_id"]: row
            for row in result.events.filter(
                pl.col("event_type") == "entry_candidate"
            ).iter_rows(named=True)
        }
        self.assertTrue(entries["overnight"]["entry_admitted"])
        self.assertEqual(
            entries["product-rejected"]["entry_admission_status"],
            "rejected_product_cap",
        )
        self.assertEqual(
            entries["cutoff-rejected"]["entry_admission_status"],
            "rejected_entry_cutoff",
        )
        day = result.daily.filter(pl.col("Date") == D1).row(0, named=True)
        self.assertEqual(day["intraday_peak_one_way_notional_twd"], 60.0)
        self.assertEqual(day["eod_outstanding_one_way_notional_twd"], 60.0)
        self.assertEqual(day["overnight_carried_out_one_way_notional_twd"], 60.0)
        self.assertTrue(day["eod_overnight_limit_breached"])
        self.assertEqual(day["eod_overnight_limit_overage_twd"], 10.0)
        self.assertFalse(day["eod_overnight_limit_enforced_on_admission"])

    def test_daily_turnover_pnl_losses_drawdown_and_time_weighted_usage(
        self,
    ) -> None:
        paths = _frame(
            _path(
                "winner",
                notional=40.0,
                gross=10.0,
                cost=2.0,
                exit_ns=_ns(D1, 10, 0),
            ),
            _path(
                "overnight-loss",
                entry_ns=_ns(D1, 9, 0) + 1,
                terminal_date=D2,
                exit_ns=_ns(D2, 10, 0),
                product="B",
                notional=50.0,
                gross=-5.0,
                cost=2.0,
            ),
            _path(
                "same-day-loss",
                date=D2,
                entry_ns=_ns(D2, 10, 0) + 1,
                terminal_date=D2,
                exit_ns=_ns(D2, 11, 0),
                product="C",
                notional=40.0,
                gross=-1.0,
                cost=2.0,
            ),
        )
        result = backtest_priced_paths(
            paths, config=_config(100.0), session_dates=(D1, D2)
        )
        day1 = result.daily.filter(pl.col("Date") == D1).row(0, named=True)
        day2 = result.daily.filter(pl.col("Date") == D2).row(0, named=True)
        self.assertEqual(day1["intraday_peak_one_way_notional_twd"], 90.0)
        self.assertAlmostEqual(
            day1["intraday_time_weighted_mean_one_way_notional_twd"],
            (40.0 * 1.0 + 50.0 * 4.5) / 4.5,
            places=8,
        )
        self.assertEqual(day1["eod_outstanding_one_way_notional_twd"], 50.0)
        self.assertEqual(day1["accepted_entry_one_way_turnover_twd"], 90.0)
        self.assertEqual(day1["realized_net_pnl_twd"], 8.0)
        self.assertEqual(day1["cumulative_realized_net_pnl_twd"], 8.0)
        self.assertEqual(day1["realized_drawdown_twd"], 0.0)
        self.assertEqual(day2["opening_outstanding_one_way_notional_twd"], 50.0)
        self.assertEqual(day2["realized_net_pnl_twd"], -10.0)
        self.assertEqual(day2["cumulative_realized_net_pnl_twd"], -2.0)
        self.assertEqual(day2["realized_drawdown_twd"], 10.0)

        summary = result.summary.row(0, named=True)
        self.assertEqual(summary["accepted_paths"], 3)
        self.assertEqual(summary["completed_exits"], 3)
        self.assertEqual(summary["winning_exits"], 1)
        self.assertEqual(summary["losing_exits"], 2)
        self.assertEqual(summary["loss_sum_twd"], 10.0)
        self.assertEqual(summary["loss_magnitude_p50_twd"], 5.0)
        self.assertAlmostEqual(summary["loss_magnitude_p90_twd"], 6.6)
        self.assertEqual(summary["largest_loss_twd"], 7.0)
        self.assertEqual(summary["max_realized_drawdown_twd"], 10.0)
        self.assertEqual(summary["entry_notional_cap_turns"], 1.3)
        self.assertEqual(summary["exit_notional_cap_turns"], 1.3)
        self.assertEqual(summary["daily_win_rate_including_zero_days"], 0.5)
        self.assertEqual(summary["total_net_to_max_realized_drawdown"], -0.2)
        self.assertAlmostEqual(summary["same_day_close_ratio"], 2.0 / 3.0)
        self.assertEqual(
            summary["mean_distinct_accepted_products_per_session"], 1.5
        )
        self.assertAlmostEqual(
            summary["mean_max_single_product_entry_notional_share"],
            ((50.0 / 90.0) + 1.0) / 2.0,
        )
        expected_sample_std = ((9.0**2 + (-9.0) ** 2) / 1.0) ** 0.5
        self.assertAlmostEqual(
            summary["annualized_sharpe_daily_net_252"],
            -1.0 / expected_sample_std * (252.0**0.5),
        )

    def test_explicit_calendar_keeps_no_event_carry_session(self) -> None:
        paths = _frame(
            _path(
                "carry",
                terminal_date=D3,
                exit_ns=_ns(D3, 10, 0),
                notional=40.0,
            )
        )
        result = backtest_priced_paths(
            paths, config=_config(100.0), session_dates=(D1, D2, D3)
        )
        middle = result.daily.filter(pl.col("Date") == D2).row(0, named=True)
        self.assertEqual(middle["entry_candidates"], 0)
        self.assertEqual(middle["completed_exits"], 0)
        self.assertEqual(middle["opening_outstanding_one_way_notional_twd"], 40.0)
        self.assertEqual(middle["intraday_peak_one_way_notional_twd"], 40.0)
        self.assertEqual(
            middle["intraday_time_weighted_mean_one_way_notional_twd"], 40.0
        )
        self.assertEqual(middle["eod_outstanding_one_way_notional_twd"], 40.0)
        self.assertTrue(middle["session_calendar_explicitly_supplied"])

    def test_product_held_at_open_is_exit_only_for_the_entire_session(self) -> None:
        paths = _frame(
            _path(
                "carry-a",
                product="A",
                terminal_date=D2,
                exit_ns=_ns(D2, 9, 30),
                notional=40.0,
            ),
            _path(
                "new-a-after-exit",
                date=D2,
                product="A",
                entry_ns=_ns(D2, 10, 0),
                exit_ns=_ns(D2, 11, 0),
                notional=40.0,
            ),
            _path(
                "new-b",
                date=D2,
                product="B",
                entry_ns=_ns(D2, 10, 0) + 1,
                exit_ns=_ns(D2, 11, 0),
                notional=40.0,
            ),
        )
        config = PortfolioCapBacktestConfig(
            scenarios=(PortfolioCapScenario("test", hard_intraday_cap_twd=200.0),),
            per_product_fraction=1.0,
            block_new_entries_for_products_held_at_session_open=True,
        )
        result = backtest_priced_paths(
            paths, config=config, session_dates=(D1, D2)
        )
        entries = {
            row["policy_path_id"]: row
            for row in result.events.filter(
                pl.col("event_type") == "entry_candidate"
            ).iter_rows(named=True)
        }
        self.assertTrue(entries["carry-a"]["entry_admitted"])
        self.assertEqual(
            entries["new-a-after-exit"]["entry_admission_status"],
            "rejected_opening_carry_exit_only",
        )
        self.assertTrue(
            entries["new-a-after-exit"]["opening_carry_exit_only_blocked"]
        )
        self.assertTrue(entries["new-b"]["entry_admitted"])
        day2 = result.daily.filter(pl.col("Date") == D2).row(0, named=True)
        self.assertEqual(day2["opening_carry_products"], 1)
        self.assertEqual(day2["rejected_opening_carry_exit_only"], 1)
        summary = result.summary.row(0, named=True)
        self.assertEqual(summary["rejected_opening_carry_exit_only"], 1)
        self.assertTrue(summary["opening_carry_product_exit_only_policy"])

    def test_default_grid_is_10_through_50_million(self) -> None:
        result = backtest_priced_paths(_frame(_path("one")), session_dates=(D1,))
        self.assertEqual(
            result.summary["hard_intraday_cap_twd"].to_list(),
            list(DEFAULT_HARD_INTRADAY_CAPS_TWD),
        )
        self.assertEqual(result.summary["accepted_paths"].to_list(), [1] * 5)
        self.assertEqual(result.events.height, 10)

    def test_rejects_unpriced_or_internally_inconsistent_path(self) -> None:
        unpriced = _path("unpriced")
        unpriced["terminal_cashflow_priced"] = False
        with self.assertRaisesRegex(ValueError, "point-identified"):
            backtest_priced_paths(_frame(unpriced), config=_config(100.0))

        inconsistent = _path("bad-net")
        inconsistent["net_cycle_pnl_twd"] = 999.0
        with self.assertRaisesRegex(ValueError, "gross - cost"):
            backtest_priced_paths(_frame(inconsistent), config=_config(100.0))


if __name__ == "__main__":
    unittest.main()
