from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import polars as pl

from maker.src.quote_fill.aggressive_1300_analysis import (
    Aggressive1300AnalysisConfig,
    EXPECTED_CACHE_ARTIFACTS,
    _cache_inventory_schema,
    _cycle_transaction_cost,
    _entry_price_source_inventory_schema,
    _market_template_schema,
    _normalize_template_record,
    _validate_controller_results,
    build_analysis_result,
    build_controller_contract,
    build_daily_open_inventory,
    build_hard_cap_controller_results,
    build_position_template_overlay,
    publish_analysis_bundle,
    validate_market_templates,
    verify_analysis_bundle,
)
from maker.src.quote_fill.aggressive_1300_exit import aggressive_1300_start_cursor


D1 = "20260102"
D2 = "20260105"
D3 = "20260106"


_PATH_SCHEMA = {
    "Date": pl.String,
    "ValueCode": pl.String,
    "QuoteCode": pl.String,
    "entry_raw_order_fact_id": pl.String,
    "position_established_ns": pl.Int64,
    "policy_path_id": pl.String,
    "exit_policy_trial_id": pl.String,
    "terminal_date": pl.String,
    "exit_decision_time_ns": pl.Int64,
    "terminal_cashflow_priced": pl.Boolean,
    "outcome_type": pl.String,
    "outcome_status": pl.String,
    "last_observed_session_date": pl.String,
    "outstanding_interval_end_exclusive": pl.String,
    "supplemental_terminal_resolution": pl.String,
    "model_imputed_full_carry_on_unknown": pl.Boolean,
    "double_exit_bias_possible": pl.Boolean,
    "expiry_uses_last_valid_session_mark": pl.Boolean,
    "expiry_uses_last_observed_session_mark": pl.Boolean,
    "expiry_mark_uses_trade_fallback": pl.Boolean,
    "spot_expiry_mark_is_executable_bbo": pl.Boolean,
    "future_expiry_mark_is_executable_bbo": pl.Boolean,
    "expiry_mark_is_official_close": pl.Boolean,
    "expiry_mark_is_official_settlement": pl.Boolean,
    "entry_no_fill_included": pl.Boolean,
    "gross_zero_imputation": pl.Boolean,
    "entry_policy_generation_id": pl.String,
    "entry_route": pl.String,
    "boundary_quantile": pl.Int64,
    "normalization_notional_twd": pl.Float64,
    "entry_spot_price": pl.Float64,
    "entry_future_price": pl.Float64,
    "entry_contract_size_shares": pl.Int64,
    "exit_spot_price": pl.Float64,
    "exit_future_price": pl.Float64,
    "gross_cycle_pnl_twd": pl.Float64,
    "gross_cycle_bp": pl.Float64,
}


def _path(
    name: str,
    *,
    origin: str,
    established_ns: int,
    terminal_date: str | None = None,
    terminal_ns: int | None = None,
    last_observed: str = D2,
    entry_spot_price: float = 100.0,
    value_code: str = "2330",
    quote_code: str = "CDF1",
) -> dict[str, object]:
    priced = terminal_date is not None
    entry_future_price = entry_spot_price + 1.0
    exit_spot = entry_spot_price + 2.0 if priced else None
    exit_future = entry_spot_price if priced else None
    gross = 6_000.0 if priced else None
    return {
        "Date": origin,
        "ValueCode": value_code,
        "QuoteCode": quote_code,
        "entry_raw_order_fact_id": name,
        "position_established_ns": established_ns,
        "policy_path_id": f"path/{name}",
        "exit_policy_trial_id": f"trial/{name}",
        "terminal_date": terminal_date,
        "exit_decision_time_ns": terminal_ns,
        "terminal_cashflow_priced": priced,
        "outcome_type": "terminal" if priced else "censored",
        "outcome_status": "same_day_target_exit" if priced else "maker_fill_state_unknown",
        "last_observed_session_date": last_observed,
        "outstanding_interval_end_exclusive": last_observed,
        "supplemental_terminal_resolution": (
            "synthetic_priced_terminal" if priced else "synthetic_unpriced_terminal"
        ),
        "model_imputed_full_carry_on_unknown": False,
        "double_exit_bias_possible": False,
        "expiry_uses_last_valid_session_mark": False,
        "expiry_uses_last_observed_session_mark": False,
        "expiry_mark_uses_trade_fallback": False,
        "spot_expiry_mark_is_executable_bbo": None,
        "future_expiry_mark_is_executable_bbo": None,
        "expiry_mark_is_official_close": None,
        "expiry_mark_is_official_settlement": None,
        "entry_no_fill_included": False,
        "gross_zero_imputation": False,
        "entry_policy_generation_id": f"entry/{name}",
        "entry_route": "spot_bid_future_taker",
        "boundary_quantile": 95,
        "normalization_notional_twd": entry_spot_price * 2_000,
        "entry_spot_price": entry_spot_price,
        "entry_future_price": entry_future_price,
        "entry_contract_size_shares": 2_000,
        "exit_spot_price": exit_spot,
        "exit_future_price": exit_future,
        "gross_cycle_pnl_twd": gross,
        "gross_cycle_bp": 300.0 if priced else None,
    }


def _paths() -> pl.DataFrame:
    return pl.from_dicts(
        [
            _path(
                "carry",
                origin=D1,
                established_ns=aggressive_1300_start_cursor(D1).recv_time_ns - 1,
            ),
            _path(
                "late-origin",
                origin=D2,
                established_ns=aggressive_1300_start_cursor(D2).recv_time_ns,
                last_observed=D3,
            ),
            _path(
                "terminal-at-start",
                origin=D1,
                established_ns=aggressive_1300_start_cursor(D1).recv_time_ns - 2,
                terminal_date=D2,
                terminal_ns=aggressive_1300_start_cursor(D2).recv_time_ns,
            ),
            _path(
                "date-anchor",
                origin=D3,
                established_ns=aggressive_1300_start_cursor(D3).recv_time_ns - 1,
                terminal_date=D3,
                terminal_ns=aggressive_1300_start_cursor(D3).recv_time_ns,
                last_observed=D3,
            ),
        ],
        schema=_PATH_SCHEMA,
        strict=True,
    )


def _template() -> pl.DataFrame:
    record = _normalize_template_record(
        {
            "Date": D1,
            "ValueCode": "2330",
            "QuoteCode": "CDF1",
            "policy_version": "aggressive_1300_dynamic_b1a1_oco_v1",
            "scheduled_start_recv_time_ns": aggressive_1300_start_cursor(D1).recv_time_ns,
            "actual_first_submit_recv_time_ns": None,
            "entry_after_1300_allowed": False,
            "dynamic_b1a1_peg": True,
            "future_candidate_generations": 0,
            "spot_candidate_generations": 0,
            "winner_generation_id": None,
            "winner_route": None,
            "winner_full_fill_recv_time_ns": None,
            "winner_hedge_delay_ns": None,
            "winner_hedge_status": None,
            "active_sibling_cancel_count": 0,
            "prior_unacked_cancel_count_before_winner": 0,
            "sibling_full_fill_after_cancel_request_count": 0,
            "sibling_partial_before_winner_count": 0,
            "oco_position_projection_safe": True,
            "nominal_branch_status": "carry_at_eod_no_admission",
            "strict_branch_status": "carry_at_eod_no_admission",
            "nominal_close_count": 0,
            "duplicate_close_prevented": True,
            "cancel_ack_observed": False,
            "cancel_model": "nominal_instant_cancel_v0",
            "joint_volume_allocated": False,
            "strict_ev_ready": False,
            "winner_maker_price": None,
            "winner_taker_vwap_price": None,
            "exit_decision_time_ns": None,
            "exit_spot_price": None,
            "exit_future_price": None,
        },
        template_source="synthetic_test",
    )
    return pl.from_dicts([record], schema=_market_template_schema(), strict=True)


def _flat_template(
    *,
    date: str = D1,
    value_code: str = "2330",
    quote_code: str = "CDF1",
    decision_offset_ns: int = 10_000_000_000,
) -> pl.DataFrame:
    start = aggressive_1300_start_cursor(date).recv_time_ns
    fill = start + decision_offset_ns - 50_000_000
    record = _normalize_template_record(
        {
            "Date": date,
            "ValueCode": value_code,
            "QuoteCode": quote_code,
            "policy_version": "aggressive_1300_dynamic_b1a1_oco_v1",
            "scheduled_start_recv_time_ns": start,
            "actual_first_submit_recv_time_ns": start + 1,
            "entry_after_1300_allowed": False,
            "dynamic_b1a1_peg": True,
            "future_candidate_generations": 1,
            "spot_candidate_generations": 1,
            "winner_generation_id": (
                f"prefix/aggressive_1300_dynamic_b1a1_oco_v1/"
                "spot_ask_future_taker/generation"
            ),
            "winner_route": "spot_ask_future_taker",
            "winner_full_fill_recv_time_ns": fill,
            "winner_hedge_delay_ns": 50_000_000,
            "winner_hedge_status": "executable",
            "active_sibling_cancel_count": 1,
            "prior_unacked_cancel_count_before_winner": 0,
            "sibling_full_fill_after_cancel_request_count": 0,
            "sibling_partial_before_winner_count": 0,
            "oco_position_projection_safe": True,
            "nominal_branch_status": "flat_same_day",
            "strict_branch_status": "cancel_race_unknown",
            "nominal_close_count": 1,
            "duplicate_close_prevented": True,
            "cancel_ack_observed": False,
            "cancel_model": "nominal_instant_cancel_v0",
            "joint_volume_allocated": False,
            "strict_ev_ready": False,
            "winner_maker_price": 102.0,
            "winner_taker_vwap_price": 100.0,
            "exit_decision_time_ns": start + decision_offset_ns,
            "exit_spot_price": 102.0,
            "exit_future_price": 100.0,
        },
        template_source="synthetic_flat_test",
    )
    return pl.from_dicts([record], schema=_market_template_schema(), strict=True)


def _cache_inventory() -> pl.DataFrame:
    rows = []
    for name in EXPECTED_CACHE_ARTIFACTS:
        rows.append(
            {
                "Date": D1,
                "ValueCode": "2330",
                "QuoteCode": "CDF1",
                "cache_key_sha256": "a" * 64,
                "cache_complete_path": "/tmp/cache/complete.json",
                "cache_complete_sha256": "b" * 64,
                "artifact_name": name,
                "artifact_path": f"/tmp/cache/{name}",
                "artifact_sha256": "c" * 64,
                "artifact_bytes": 1,
                "artifact_rows": 1,
                "artifact_columns": 1,
                "raw_source_fingerprint_json": "{}",
                "raw_source_fingerprint_sha256": "d" * 64,
                "cache_artifact_content_hash_verified": True,
                "upstream_raw_content_integrity_bound": False,
            }
        )
    return pl.from_dicts(rows, schema=_cache_inventory_schema(), strict=True)


def _entry_inventory() -> pl.DataFrame:
    return pl.from_dicts(
        [
            {
                "source_kind": "aggressive_1300_entry_execution_action",
                "Date": D1,
                "ValueCode": "2330",
                "partition_complete_path": "/tmp/entry/complete.json",
                "partition_complete_sha256": "e" * 64,
                "artifact_path": "/tmp/entry/execution_action_facts.parquet",
                "artifact_sha256": "f" * 64,
                "artifact_bytes": 1,
                "artifact_rows": 4,
                "artifact_columns": 9,
            }
        ],
        schema=_entry_price_source_inventory_schema(),
        strict=True,
    )


class Aggressive1300AnalysisTest(unittest.TestCase):
    def test_daily_inventory_preserves_carry_and_rejects_origin_day_late_entry(self) -> None:
        inventory, audit = build_daily_open_inventory(_paths())
        self.assertEqual(
            inventory.select("Date", "position_id").rows(),
            [(D1, "terminal-at-start"), (D1, "carry"), (D2, "carry")],
        )
        row = audit.row(0, named=True)
        self.assertEqual(row["origin_day_entries_at_or_after_1300_excluded"], 1)
        self.assertEqual(row["daily_inventory_rows_excluded_from_late_origin_entries"], 1)
        self.assertEqual(row["daily_open_position_rows"], 3)
        self.assertEqual(row["carried_position_rows"], 1)

    def test_template_and_overlay_fail_closed_without_joint_volume(self) -> None:
        templates = _template()
        validate_market_templates(templates)
        bad = templates.with_columns(pl.lit(True).alias("joint_volume_allocated"))
        with self.assertRaisesRegex(ValueError, "lifecycle"):
            validate_market_templates(bad)
        inventory, _ = build_daily_open_inventory(_paths())
        overlay = build_position_template_overlay(inventory, templates)
        self.assertEqual(overlay.height, 3)
        self.assertEqual(overlay["market_template_available"].sum(), 2)
        self.assertFalse(any(overlay["controller_nominal_close_eligible"]))
        self.assertFalse(any(overlay["position_close_is_actual_execution_claim"]))

    def test_market_template_rejects_full_fill_before_first_submit(self) -> None:
        template = _flat_template()
        bad = template.with_columns(
            (pl.col("winner_full_fill_recv_time_ns") + 1).alias(
                "actual_first_submit_recv_time_ns"
            )
        )
        with self.assertRaisesRegex(ValueError, "lifecycle"):
            validate_market_templates(bad)

    def test_daily_release_counter_includes_normal_exit_before_later_entry(self) -> None:
        start2 = aggressive_1300_start_cursor(D2).recv_time_ns
        paths = pl.from_dicts(
            [
                _path(
                    "morning-exit",
                    origin=D1,
                    established_ns=aggressive_1300_start_cursor(D1).recv_time_ns - 1,
                    terminal_date=D2,
                    terminal_ns=start2 - 3 * 60 * 60 * 1_000_000_000,
                ),
                _path(
                    "later-entry",
                    origin=D2,
                    established_ns=start2 - 2 * 60 * 60 * 1_000_000_000,
                    value_code="2317",
                    quote_code="CDF2",
                ),
            ],
            schema=_PATH_SCHEMA,
            strict=True,
        )
        _, daily, _ = build_hard_cap_controller_results(
            paths,
            _template(),
            session_dates=(D1, D2),
        )
        row = daily.filter(
            (pl.col("Date") == D2)
            & pl.col("primary_conservative_result")
            & (pl.col("portfolio_cap_twd") == 50_000_000.0)
        ).row(0, named=True)
        self.assertEqual(row["normal_releases_before_1300"], 1)
        self.assertEqual(
            row["normal_release_notional_before_1300_twd"], 200_000.0
        )

    def test_forbidden_post1300_entry_never_advances_normal_exit_clock(self) -> None:
        start1 = aggressive_1300_start_cursor(D1).recv_time_ns
        start2 = aggressive_1300_start_cursor(D2).recv_time_ns
        aggressive_decision_offset = 6 * 60 * 1_000_000_000 + 50_000_000
        template = _flat_template(
            date=D2,
            decision_offset_ns=aggressive_decision_offset,
        )
        config = Aggressive1300AnalysisConfig(
            portfolio_caps_twd=(1_000_000.0,),
            eod_target_ceiling_twd=100_000.0,
            per_product_cap_fraction=1.0,
        )
        for normal_minutes, aggressive_expected in ((5, False), (15, True)):
            with self.subTest(normal_minutes=normal_minutes):
                paths = pl.from_dicts(
                    [
                        _path(
                            "carry",
                            origin=D1,
                            established_ns=start1 - 1,
                            terminal_date=D2,
                            terminal_ns=(
                                start2
                                + normal_minutes * 60 * 1_000_000_000
                            ),
                        ),
                        _path(
                            "forbidden-late",
                            origin=D2,
                            established_ns=start2 + 10 * 60 * 1_000_000_000,
                        ),
                    ],
                    schema=_PATH_SCHEMA,
                    strict=True,
                )
                positions, daily, _ = build_hard_cap_controller_results(
                    paths,
                    template,
                    session_dates=(D1, D2),
                    config=config,
                )
                scenario = positions.filter(
                    pl.col("primary_conservative_result")
                )
                carry = scenario.filter(pl.col("position_id") == "carry").row(
                    0, named=True
                )
                late = scenario.filter(
                    pl.col("position_id") == "forbidden-late"
                ).row(0, named=True)
                day2 = daily.filter(
                    (pl.col("Date") == D2)
                    & pl.col("primary_conservative_result")
                ).row(0, named=True)
                self.assertEqual(day2["active_notional_at_1300_twd"], 200_000.0)
                self.assertEqual(
                    late["admission_status"],
                    "forbidden_new_entry_at_or_after_1300",
                )
                self.assertEqual(late["active_notional_before_admission_twd"], 200_000.0)
                self.assertEqual(late["active_notional_after_admission_twd"], 200_000.0)
                self.assertEqual(
                    bool(carry["aggressive_close_allocated"]), aggressive_expected
                )
                self.assertEqual(
                    carry["scenario_terminal_source"],
                    (
                        "aggressive_1300_template_capacity"
                        if aggressive_expected
                        else "selected_normal_continuation"
                    ),
                )

    def test_controller_contract_exposes_conservative_and_optimistic_bounds(self) -> None:
        contract = build_controller_contract()
        self.assertEqual(contract.height, 10)
        self.assertEqual(contract["intraday_notional_cap_twd"].unique().sort().to_list(), [
            10_000_000.0,
            20_000_000.0,
            30_000_000.0,
            40_000_000.0,
            50_000_000.0,
        ])
        self.assertTrue(all(contract["accepted_active_inventory_source_bound"]))
        self.assertTrue(all(contract["actual_target_attainment_metrics_published"]))
        self.assertEqual(contract["primary_conservative_result"].sum(), 5)
        self.assertEqual(contract["joint_volume_overcount_upper_bound"].sum(), 5)
        self.assertFalse(any(contract["controller_formal_go"]))

    def test_controller_normal_exit_competes_then_capacity_bound_and_releases_next_day(self) -> None:
        start1 = aggressive_1300_start_cursor(D1).recv_time_ns
        start2 = aggressive_1300_start_cursor(D2).recv_time_ns
        rows = [
            _path(
                "normal-first",
                origin=D1,
                established_ns=start1 - 4,
                terminal_date=D1,
                terminal_ns=start1 + 5_000_000_000,
                last_observed=D2,
            ),
            _path("p1", origin=D1, established_ns=start1 - 3, last_observed=D2),
            _path("p2", origin=D1, established_ns=start1 - 2, last_observed=D2),
            _path("p3", origin=D1, established_ns=start1 - 1, last_observed=D2),
            _path(
                "next-day",
                origin=D2,
                established_ns=start2 - 1,
                last_observed=D2,
                entry_spot_price=150.0,
                value_code="2317",
                quote_code="CDF2",
            ),
        ]
        paths = pl.from_dicts(rows, schema=_PATH_SCHEMA, strict=True)
        config = Aggressive1300AnalysisConfig(
            portfolio_caps_twd=(800_000.0,),
            eod_target_ceiling_twd=200_000.0,
            per_product_cap_fraction=1.0,
        )
        positions, daily, summary = build_hard_cap_controller_results(
            paths, _flat_template(), config=config
        )
        conservative = summary.filter(pl.col("primary_conservative_result")).row(
            0, named=True
        )
        optimistic = summary.filter(~pl.col("primary_conservative_result")).row(
            0, named=True
        )
        self.assertEqual(conservative["aggressive_closed_positions"], 1)
        self.assertEqual(optimistic["aggressive_closed_positions"], 2)
        day1 = daily.filter(pl.col("Date") == D1).sort(
            "primary_conservative_result", descending=True
        )
        self.assertEqual(day1["normal_releases_after_controller_to_eod"].sum(), 0)
        self.assertFalse(day1.row(0, named=True)["controller_target_attained"])
        self.assertTrue(day1.row(1, named=True)["controller_target_attained"])
        normal_first = positions.filter(pl.col("position_id") == "normal-first")
        self.assertFalse(any(normal_first["aggressive_close_allocated"]))
        self.assertTrue(
            all(
                normal_first["scenario_terminal_source"]
                == "selected_normal_continuation"
            )
        )
        next_day = positions.filter(pl.col("position_id") == "next-day")
        self.assertTrue(all(next_day["accepted"]))
        self.assertTrue(all(next_day["admission_status"] == "accepted_before_1300"))

    def test_controller_uses_full_calendar_when_no_new_entry_on_carry_day(self) -> None:
        start1 = aggressive_1300_start_cursor(D1).recv_time_ns
        start3 = aggressive_1300_start_cursor(D3).recv_time_ns
        paths = pl.from_dicts(
            [
                _path(
                    "carry-through-empty-day",
                    origin=D1,
                    established_ns=start1 - 1,
                    terminal_date=D3,
                    terminal_ns=start3 + 1,
                    last_observed=D3,
                )
            ],
            schema=_PATH_SCHEMA,
            strict=True,
        )
        templates = pl.concat(
            [_template(), _flat_template(date=D2)], how="vertical"
        ).sort("Date")
        config = Aggressive1300AnalysisConfig(
            portfolio_caps_twd=(400_000.0,),
            eod_target_ceiling_twd=100_000.0,
            per_product_cap_fraction=1.0,
        )
        positions, daily, summary = build_hard_cap_controller_results(
            paths,
            templates,
            session_dates=(D1, D2, D3),
            config=config,
        )
        self.assertEqual(daily["Date"].unique().sort().to_list(), [D1, D2, D3])
        empty_entry_day = daily.filter(pl.col("Date") == D2)
        self.assertEqual(empty_entry_day["accepted_entries_before_1300"].sum(), 0)
        self.assertEqual(empty_entry_day["aggressive_closes_allocated"].sum(), 2)
        self.assertTrue(all(positions["aggressive_close_allocated"]))
        self.assertTrue(all(summary["aggressive_closed_positions"] == 1))

    def test_controller_enforces_product_cap_before_portfolio_cap(self) -> None:
        start = aggressive_1300_start_cursor(D1).recv_time_ns
        paths = pl.from_dicts(
            [
                _path("product-1", origin=D1, established_ns=start - 2, last_observed=D1),
                _path("product-2", origin=D1, established_ns=start - 1, last_observed=D1),
            ],
            schema=_PATH_SCHEMA,
            strict=True,
        )
        config = Aggressive1300AnalysisConfig(
            portfolio_caps_twd=(1_000_000.0,),
            eod_target_ceiling_twd=1_000_000.0,
            per_product_cap_fraction=0.30,
        )
        positions, daily, summary = build_hard_cap_controller_results(
            paths, _template(), config=config
        )
        second = positions.filter(pl.col("position_id") == "product-2")
        self.assertTrue(all(~second["accepted"]))
        self.assertTrue(
            all(second["admission_status"] == "per_product_hard_cap_blocked")
        )
        self.assertEqual(summary["product_only_blocked_entries"].sum(), 2)
        self.assertEqual(daily["product_only_blocked_entries_before_1300"].sum(), 2)

    def test_held_product_is_exit_only_for_whole_session(self) -> None:
        start1 = aggressive_1300_start_cursor(D1).recv_time_ns
        start2 = aggressive_1300_start_cursor(D2).recv_time_ns
        paths = pl.from_dicts(
            [
                _path(
                    "overnight-carry",
                    origin=D1,
                    established_ns=start1 - 1,
                    terminal_date=D2,
                    terminal_ns=start2 - 2,
                    last_observed=D2,
                ),
                _path(
                    "same-product-new-entry",
                    origin=D2,
                    established_ns=start2 - 1,
                    last_observed=D2,
                ),
            ],
            schema=_PATH_SCHEMA,
            strict=True,
        )
        config = Aggressive1300AnalysisConfig(
            portfolio_caps_twd=(1_000_000.0,),
            eod_target_ceiling_twd=1_000_000.0,
            per_product_cap_fraction=1.0,
        )
        positions, daily, summary = build_hard_cap_controller_results(
            paths, _template(), session_dates=(D1, D2), config=config
        )
        new_entry = positions.filter(
            pl.col("position_id") == "same-product-new-entry"
        )
        self.assertTrue(all(~new_entry["accepted"]))
        self.assertTrue(
            all(new_entry["admission_status"] == "held_product_exit_only_blocked")
        )
        day2 = daily.filter(pl.col("Date") == D2)
        self.assertTrue(all(day2["held_product_exit_only_products"] == 1))
        self.assertTrue(all(day2["held_product_exit_only_blocked_entries"] == 1))
        self.assertTrue(all(summary["held_product_exit_only_blocked_entries"] == 1))

    def test_10m_20m_are_identity_controls_under_same_admission_rules(self) -> None:
        start = aggressive_1300_start_cursor(D1).recv_time_ns
        paths = pl.from_dicts(
            [_path("control", origin=D1, established_ns=start - 1, last_observed=D1)],
            schema=_PATH_SCHEMA,
            strict=True,
        )
        config = Aggressive1300AnalysisConfig(
            portfolio_caps_twd=(10_000_000.0, 20_000_000.0),
            eod_target_ceiling_twd=20_000_000.0,
            per_product_cap_fraction=0.30,
        )
        positions, daily, summary = build_hard_cap_controller_results(
            paths, _flat_template(), config=config
        )
        self.assertTrue(all(~positions["aggressive_close_allocated"]))
        self.assertTrue(all(summary["aggressive_closed_positions"] == 0))
        for cap in config.portfolio_caps_twd:
            group = positions.filter(pl.col("portfolio_cap_twd") == cap)
            self.assertEqual(group["accepted"].to_list(), [True, True])
            day = daily.filter(pl.col("portfolio_cap_twd") == cap)
            self.assertEqual(day["eod_active_notional_twd"].n_unique(), 1)

    def test_transaction_cost_matches_formal_profile(self) -> None:
        same_day = _cycle_transaction_cost(
            entry_date=D1,
            terminal_date=D1,
            entry_spot_price=100.0,
            exit_spot_price=102.0,
            entry_future_price=101.0,
            exit_future_price=100.0,
            shares=2_000,
        )
        overnight = _cycle_transaction_cost(
            entry_date=D1,
            terminal_date=D2,
            entry_spot_price=100.0,
            exit_spot_price=102.0,
            entry_future_price=101.0,
            exit_future_price=100.0,
            shares=2_000,
        )
        self.assertAlmostEqual(same_day, 423.124, places=9)
        self.assertAlmostEqual(overnight, 729.124, places=9)

    def test_post_expiry_close_accounting_keeps_1320_primary_snapshot(self) -> None:
        start = aggressive_1300_start_cursor(D1).recv_time_ns
        row = _path(
            "official-expiry",
            origin=D1,
            established_ns=start - 1,
            terminal_date=D1,
            terminal_ns=start + 30 * 60 * 1_000_000_000,
            last_observed=D1,
        )
        row.update(
            {
                "supplemental_terminal_resolution": (
                    "expiry_same_day_two_leg_close_price"
                ),
                "expiry_uses_last_valid_session_mark": False,
                "expiry_uses_last_observed_session_mark": False,
                "expiry_mark_uses_trade_fallback": False,
                "spot_expiry_mark_is_executable_bbo": False,
                "future_expiry_mark_is_executable_bbo": False,
                "expiry_mark_is_official_close": True,
                "expiry_mark_is_official_settlement": False,
            }
        )
        paths = pl.from_dicts([row], schema=_PATH_SCHEMA, strict=True)
        config = Aggressive1300AnalysisConfig(
            portfolio_caps_twd=(1_000_000.0,),
            eod_target_ceiling_twd=1_000_000.0,
            per_product_cap_fraction=1.0,
        )
        _, daily, summary = build_hard_cap_controller_results(
            paths, _template(), config=config
        )
        self.assertTrue(all(daily["eod_active_positions"] == 1))
        self.assertTrue(all(daily["post_expiry_close_positions"] == 1))
        self.assertTrue(all(daily["post_expiry_close_active_positions"] == 0))
        self.assertTrue(all(daily["eod_active_notional_twd"] == 200_000.0))
        self.assertTrue(
            all(daily["post_expiry_close_active_notional_twd"] == 0.0)
        )
        self.assertTrue(
            all(summary["post_expiry_close_is_full_1330_market_replay"] == False)  # noqa: E712
        )

    def test_normal_exit_wins_equal_aggressive_decision_timestamp(self) -> None:
        start = aggressive_1300_start_cursor(D1).recv_time_ns
        paths = pl.from_dicts(
            [
                _path(
                    "equal-timestamp",
                    origin=D1,
                    established_ns=start - 1,
                    terminal_date=D1,
                    terminal_ns=start + 10_000_000_000,
                    last_observed=D1,
                )
            ],
            schema=_PATH_SCHEMA,
            strict=True,
        )
        config = Aggressive1300AnalysisConfig(
            portfolio_caps_twd=(400_000.0,),
            eod_target_ceiling_twd=100_000.0,
            per_product_cap_fraction=1.0,
        )
        positions, daily, summary = build_hard_cap_controller_results(
            paths, _flat_template(), config=config
        )
        self.assertTrue(all(~positions["aggressive_close_allocated"]))
        self.assertTrue(
            all(positions["scenario_terminal_source"] == "selected_normal_continuation")
        )
        self.assertTrue(
            all(daily["normal_wins_equal_aggressive_timestamp_positions"] == 1)
        )
        self.assertTrue(
            all(summary["normal_wins_equal_aggressive_timestamp_positions"] == 1)
        )

    def test_controller_rejects_null_supplemental_provenance(self) -> None:
        paths = _paths().with_columns(
            pl.when(pl.col("entry_raw_order_fact_id") == "carry")
            .then(None)
            .otherwise(pl.col("expiry_mark_uses_trade_fallback"))
            .cast(pl.Boolean)
            .alias("expiry_mark_uses_trade_fallback")
        )
        with self.assertRaisesRegex(ValueError, "source invariant"):
            build_hard_cap_controller_results(paths, _template())

    def test_controller_validator_recomputes_cost_daily_and_summary(self) -> None:
        start = aggressive_1300_start_cursor(D1).recv_time_ns
        paths = pl.from_dicts(
            [
                _path(
                    "priced-control",
                    origin=D1,
                    established_ns=start - 1,
                    terminal_date=D1,
                    terminal_ns=start + 1,
                    last_observed=D1,
                )
            ],
            schema=_PATH_SCHEMA,
            strict=True,
        )
        config = Aggressive1300AnalysisConfig(
            portfolio_caps_twd=(1_000_000.0,),
            eod_target_ceiling_twd=1_000_000.0,
            per_product_cap_fraction=1.0,
        )
        positions, daily, summary = build_hard_cap_controller_results(
            paths, _template(), config=config
        )
        bad_cost = positions.with_columns(
            pl.when(pl.col("scenario_terminal_cashflow_priced"))
            .then(pl.col("scenario_transaction_cost_twd") + 1.0)
            .otherwise(pl.col("scenario_transaction_cost_twd"))
            .alias("scenario_transaction_cost_twd")
        )
        with self.assertRaisesRegex(ValueError, "lifecycle|transaction cost"):
            _validate_controller_results(bad_cost, daily, summary, config)
        bad_daily = daily.with_columns(
            (pl.col("realized_net_pnl_twd") + 1.0).alias("realized_net_pnl_twd")
        )
        with self.assertRaisesRegex(ValueError, "daily realized"):
            _validate_controller_results(positions, bad_daily, summary, config)
        bad_summary = summary.with_columns(
            (pl.col("completed_scenario_net_pnl_twd") + 1.0).alias(
                "completed_scenario_net_pnl_twd"
            )
        )
        with self.assertRaisesRegex(ValueError, "summary"):
            _validate_controller_results(positions, daily, bad_summary, config)

    def test_atomic_bundle_round_trip_and_semantic_tamper_rejection(self) -> None:
        inventory, audit = build_daily_open_inventory(_paths())
        templates = _template()
        result = build_analysis_result(
            _paths(),
            templates,
            _cache_inventory(),
            _entry_inventory(),
            audit,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bundle"
            publish_analysis_bundle(output, result, {"synthetic": True})
            marker = verify_analysis_bundle(output, verify_sources=False)
            self.assertTrue(marker["complete"])
            payload = __import__("json").loads((output / "complete.json").read_text())
            payload["fact_semantics"]["formal_ev_ready"] = True
            unhashed = dict(payload)
            unhashed.pop("marker_payload_sha256")
            from maker.src.quote_fill.aggressive_1300_analysis import _canonical_sha256

            payload["marker_payload_sha256"] = _canonical_sha256(unhashed)
            (output / "complete.json").write_text(
                __import__("json").dumps(payload, sort_keys=True)
            )
            with self.assertRaisesRegex(ValueError, "safety semantics"):
                verify_analysis_bundle(output, verify_sources=False)


if __name__ == "__main__":
    unittest.main()
