from __future__ import annotations

import unittest

import polars as pl

from maker.src.quote_fill.combined_cost_cap_sweep import (
    CombinedCapConfig,
    build_combined_cost_cap_sweep,
)


D1 = "20260102"
D2 = "20260105"


def _path(
    identifier: str,
    *,
    date: str = D1,
    established_ns: int = 100,
    value_code: str = "A",
    quote_code: str = "QA1",
    notional: float = 100.0,
    category: str = "unknown",
    same_day: bool = False,
    terminal_date: str | None = None,
    exit_ns: int | None = None,
    gross_twd: float = 10.0,
    gross_bp: float = 100.0,
    entry_spot: float | None = None,
    entry_future: float | None = None,
    contract_size: int | None = None,
    exit_spot: float | None = None,
    exit_future: float | None = None,
) -> dict[str, object]:
    completed = category == "completed"
    if completed:
        contract_size = 1 if contract_size is None else contract_size
        entry_spot = (
            float(notional) / float(contract_size)
            if entry_spot is None
            else entry_spot
        )
        entry_future = entry_spot if entry_future is None else entry_future
        exit_spot = entry_spot if exit_spot is None else exit_spot
        exit_future = entry_future if exit_future is None else exit_future
        terminal_date = terminal_date or (date if same_day else D2)
        exit_ns = 200 if exit_ns is None else exit_ns
    else:
        gross_twd = None
        gross_bp = None
        terminal_date = None
        exit_ns = None
        entry_spot = None
        entry_future = None
        contract_size = None
        exit_spot = None
        exit_future = None
    return {
        "Date": date,
        "ValueCode": value_code,
        "QuoteCode": quote_code,
        "entry_route": "spot_bid_future_taker",
        "boundary_quantile": 95,
        "entry_policy_generation_id": f"generation-{identifier}",
        "entry_raw_order_fact_id": f"raw-{identifier}",
        "exit_policy_trial_id": f"exit-{identifier}",
        "position_established_ns": established_ns,
        "filled_entry_outcome_category": category,
        "terminal_date": terminal_date,
        "exit_decision_time_ns": exit_ns,
        "gross_cycle_pnl_twd": gross_twd,
        "gross_cycle_bp": gross_bp,
        "normalization_notional_twd": notional,
        "physical_entry_dependency_id": f"physical-{identifier}",
        "policy_path_id": f"path-{identifier}",
        "completed_same_day": completed and same_day,
        "completed_overnight": completed and not same_day,
        "terminal_cashflow_priced": completed,
        "nominal_cancel_model_assumption": True,
        "pathwise_ev_ready": False,
        "joint_volume_allocated": False,
        "unresolved_cashflow_imputed": False,
        "diagnostic_challenger_is_selected_action": False,
        "production_strategy_go": False,
        "entry_spot_price": entry_spot,
        "entry_future_price": entry_future,
        "entry_contract_size_shares": contract_size,
        "exit_spot_price": exit_spot,
        "exit_future_price": exit_future,
        "exact_price_source": "synthetic_exact_prices" if completed else None,
    }


def _frame(*rows: dict[str, object]) -> pl.DataFrame:
    return pl.from_dicts(
        rows,
        infer_schema_length=None,
        schema_overrides={
            "position_established_ns": pl.Int64,
            "exit_decision_time_ns": pl.Int64,
            "gross_cycle_pnl_twd": pl.Float64,
            "gross_cycle_bp": pl.Float64,
            "normalization_notional_twd": pl.Float64,
            "entry_spot_price": pl.Float64,
            "entry_future_price": pl.Float64,
            "entry_contract_size_shares": pl.Int64,
            "exit_spot_price": pl.Float64,
            "exit_future_price": pl.Float64,
        },
    )


def _config(
    cap: float = 1_000_000.0, *, fraction: float = 1.0
) -> CombinedCapConfig:
    return CombinedCapConfig(
        portfolio_caps_twd=(cap,),
        per_product_fraction=fraction,
    )


class CombinedCostCapSweepTest(unittest.TestCase):
    def test_exact_four_price_costs_and_unresolved_remain_null(self) -> None:
        paths = _frame(
            _path(
                "same",
                category="completed",
                same_day=True,
                notional=100_000.0,
                gross_twd=1_000.0,
                gross_bp=100.0,
                entry_spot=100.0,
                entry_future=120.0,
                contract_size=1_000,
                exit_spot=110.0,
                exit_future=130.0,
            ),
            _path(
                "overnight",
                category="completed",
                same_day=False,
                established_ns=101,
                notional=100_000.0,
                gross_twd=1_000.0,
                gross_bp=100.0,
                entry_spot=100.0,
                entry_future=120.0,
                contract_size=1_000,
                exit_spot=110.0,
                exit_future=130.0,
            ),
            _path("unknown", established_ns=102, notional=50_000.0),
        )

        result = build_combined_cost_cap_sweep(paths, config=_config())
        costs = {
            row["policy_path_id"]: row
            for row in result.path_costs.iter_rows(named=True)
        }
        same = costs["path-same"]
        overnight = costs["path-overnight"]

        expected_common = {
            "spot_buy_commission_twd": 17.10,
            "spot_sell_commission_twd": 18.81,
            "futures_entry_tax_twd": 2.40,
            "futures_exit_tax_twd": 2.60,
            "futures_entry_commission_twd": 20.0,
            "futures_exit_commission_twd": 20.0,
        }
        for column, expected in expected_common.items():
            self.assertAlmostEqual(same[column], expected, places=10)
            self.assertAlmostEqual(overnight[column], expected, places=10)
        self.assertAlmostEqual(same["spot_sell_tax_twd"], 165.0, places=10)
        self.assertAlmostEqual(
            overnight["spot_sell_tax_twd"], 330.0, places=10
        )
        self.assertAlmostEqual(
            same["near_flat_price_reference_variable_cost_bp"], 18.82
        )
        self.assertAlmostEqual(
            overnight["near_flat_price_reference_variable_cost_bp"], 33.82
        )
        self.assertAlmostEqual(same["total_transaction_cost_twd"], 245.91)
        self.assertAlmostEqual(
            overnight["total_transaction_cost_twd"], 410.91
        )
        self.assertAlmostEqual(same["effective_transaction_cost_bp"], 24.591)
        self.assertAlmostEqual(
            overnight["effective_transaction_cost_bp"], 41.091
        )
        self.assertAlmostEqual(same["net_cycle_pnl_twd"], 754.09)
        self.assertAlmostEqual(overnight["net_cycle_pnl_twd"], 589.09)
        self.assertAlmostEqual(same["net_cycle_pnl_bp"], 75.409)
        self.assertAlmostEqual(overnight["net_cycle_pnl_bp"], 58.909)

        unresolved = costs["path-unknown"]
        nullable_costs = (
            "spot_buy_commission_twd",
            "spot_sell_commission_twd",
            "spot_sell_tax_twd",
            "futures_entry_tax_twd",
            "futures_exit_tax_twd",
            "futures_entry_commission_twd",
            "futures_exit_commission_twd",
            "total_transaction_cost_twd",
            "effective_transaction_cost_bp",
            "net_cycle_pnl_twd",
            "net_cycle_pnl_bp",
        )
        for column in nullable_costs:
            self.assertIsNone(unresolved[column])
        self.assertFalse(unresolved["transaction_cost_point_identified"])
        unresolved_summary = result.cost_summary.filter(
            pl.col("summary_group") == "unresolved_null"
        ).row(0, named=True)
        self.assertEqual(unresolved_summary["positions"], 1)
        self.assertIsNone(unresolved_summary["total_transaction_cost_twd"])
        self.assertIsNone(unresolved_summary["net_cycle_pnl_twd"])

    def test_simultaneous_cap_rejection_categories_and_state_immutability(
        self,
    ) -> None:
        paths = _frame(
            _path(
                "a-accepted",
                value_code="A",
                quote_code="A-old",
                established_ns=100,
                notional=25.0,
            ),
            _path(
                "a-product-only",
                value_code="A",
                quote_code="A-new",
                established_ns=101,
                notional=10.0,
            ),
            _path(
                "b-accepted",
                value_code="B",
                established_ns=102,
                notional=30.0,
            ),
            _path(
                "c-accepted",
                value_code="C",
                established_ns=103,
                notional=30.0,
            ),
            _path(
                "a-both",
                value_code="A",
                quote_code="A-third",
                established_ns=104,
                notional=20.0,
            ),
            _path(
                "d-portfolio-only",
                value_code="D",
                established_ns=105,
                notional=20.0,
            ),
        )
        result = build_combined_cost_cap_sweep(
            paths, config=_config(100.0, fraction=0.30)
        )
        events = {
            row["policy_path_id"]: row
            for row in result.cap_events.iter_rows(named=True)
        }

        self.assertEqual(
            events["path-a-product-only"]["admission_status"], "product_only"
        )
        self.assertEqual(events["path-a-both"]["admission_status"], "both")
        self.assertEqual(
            events["path-d-portfolio-only"]["admission_status"],
            "portfolio_only",
        )
        # ValueCode, rather than QuoteCode, is the product-cap aggregation key.
        self.assertEqual(
            events["path-a-product-only"][
                "active_same_product_notional_before_entry_twd"
            ],
            25.0,
        )
        self.assertNotEqual(
            events["path-a-accepted"]["QuoteCode"],
            events["path-a-product-only"]["QuoteCode"],
        )
        for identifier in (
            "path-a-product-only",
            "path-a-both",
            "path-d-portfolio-only",
        ):
            event = events[identifier]
            self.assertFalse(event["accepted"])
            self.assertEqual(
                event["active_portfolio_notional_after_decision_twd"],
                event["active_portfolio_notional_before_entry_twd"],
            )
            self.assertEqual(
                event["active_same_product_notional_after_decision_twd"],
                event["active_same_product_notional_before_entry_twd"],
            )

        summary = result.cap_summary.row(0, named=True)
        self.assertEqual(summary["accepted_positions"], 3)
        self.assertEqual(summary["rejected_positions"], 3)
        self.assertEqual(summary["rejected_portfolio_only"], 1)
        self.assertEqual(summary["rejected_product_only"], 1)
        self.assertEqual(summary["rejected_both_caps"], 1)
        self.assertEqual(
            summary["peak_outstanding_one_way_entry_notional_twd"], 85.0
        )
        self.assertEqual(
            summary[
                "peak_single_product_outstanding_one_way_entry_notional_twd"
            ],
            30.0,
        )
        self.assertTrue(summary["portfolio_cap_compliant"])
        self.assertTrue(summary["per_product_cap_compliant"])
        self.assertEqual(
            result.cap_events.filter(
                pl.col("active_portfolio_notional_after_decision_twd")
                > pl.col("portfolio_cap_twd")
            ).height,
            0,
        )
        self.assertEqual(
            result.cap_events.filter(
                pl.col("active_same_product_notional_after_decision_twd")
                > pl.col("per_product_cap_twd")
            ).height,
            0,
        )

    def test_equal_time_entry_precedes_exit_but_later_entry_sees_release(
        self,
    ) -> None:
        paths = _frame(
            _path(
                "completed",
                category="completed",
                same_day=True,
                value_code="A",
                established_ns=100,
                exit_ns=200,
                notional=60.0,
                gross_twd=1.0,
                gross_bp=100.0,
            ),
            _path(
                "equal-time",
                value_code="B",
                established_ns=200,
                notional=50.0,
            ),
            _path(
                "later",
                value_code="C",
                established_ns=201,
                notional=50.0,
            ),
        )
        result = build_combined_cost_cap_sweep(
            paths, config=_config(100.0, fraction=1.0)
        )
        events = {
            row["policy_path_id"]: row
            for row in result.cap_events.iter_rows(named=True)
        }
        equal_time = events["path-equal-time"]
        self.assertEqual(equal_time["admission_status"], "portfolio_only")
        self.assertEqual(equal_time["released_completed_positions_before_entry"], 0)
        self.assertEqual(
            equal_time["tie_handling"],
            "entry_before_exit_at_equal_recv_time_ns",
        )
        later = events["path-later"]
        self.assertTrue(later["accepted"])
        self.assertEqual(later["released_completed_positions_before_entry"], 1)
        self.assertEqual(
            later["released_completed_notional_before_entry_twd"], 60.0
        )
        self.assertEqual(later["active_portfolio_notional_before_entry_twd"], 0.0)

    def test_unknown_and_censored_never_release_capacity(self) -> None:
        paths = _frame(
            _path(
                "unknown",
                category="unknown",
                value_code="A",
                established_ns=100,
                notional=60.0,
            ),
            _path(
                "censored",
                category="censored",
                value_code="B",
                established_ns=101,
                notional=40.0,
            ),
            _path(
                "next-session",
                category="unknown",
                date=D2,
                value_code="C",
                established_ns=100,
                notional=20.0,
            ),
        )
        result = build_combined_cost_cap_sweep(
            paths, config=_config(100.0, fraction=1.0)
        )
        last = result.cap_events.filter(
            pl.col("policy_path_id") == "path-next-session"
        ).row(0, named=True)
        self.assertEqual(last["released_completed_positions_before_entry"], 0)
        self.assertEqual(last["active_portfolio_notional_before_entry_twd"], 100.0)
        self.assertEqual(last["admission_status"], "portfolio_only")
        summary = result.cap_summary.row(0, named=True)
        self.assertEqual(summary["accepted_unknown_or_open_positions"], 1)
        self.assertEqual(summary["accepted_censored_positions"], 1)
        self.assertFalse(summary["accepted_terminal_cashflow_point_identified"])
        self.assertTrue(summary["unresolved_positions_never_release_capacity"])

    def test_invalid_notional_and_exact_prices_fail_closed(self) -> None:
        for invalid in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(notional=invalid):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    build_combined_cost_cap_sweep(
                        _frame(_path("bad-notional", notional=invalid)),
                        config=_config(),
                    )

        invalid_price_cases = (
            ("entry_spot_price", 0.0),
            ("entry_future_price", -1.0),
            ("exit_spot_price", float("nan")),
            ("exit_future_price", float("inf")),
            ("exit_future_price", None),
        )
        for column, invalid in invalid_price_cases:
            row = _path(
                f"bad-{column}-{invalid}",
                category="completed",
                same_day=True,
                notional=100.0,
                gross_twd=1.0,
                gross_bp=100.0,
            )
            row[column] = invalid
            with self.subTest(column=column, value=invalid):
                with self.assertRaisesRegex(
                    ValueError, "reconciled exact leg prices"
                ):
                    build_combined_cost_cap_sweep(
                        _frame(row), config=_config()
                    )

        mismatched = _path(
            "mismatched-notional",
            category="completed",
            same_day=True,
            notional=100.0,
            entry_spot=99.0,
            contract_size=1,
            gross_twd=1.0,
            gross_bp=100.0,
        )
        with self.assertRaisesRegex(ValueError, "reconciled exact leg prices"):
            build_combined_cost_cap_sweep(
                _frame(mismatched), config=_config()
            )


if __name__ == "__main__":
    unittest.main()
