"""Tests for strict Exit Maker V2 terminal and comparison adapters."""

from __future__ import annotations

import unittest

import polars as pl

from maker.src.quote_fill.ev_surface import EVLookupConfig, build_daily_ev_lookup
from maker.src.quote_fill.exit_maker_terminal import (
    EXIT_MAKER_PATH_LOOKUP_KEYS_V2,
    FUTURE_BID_EXIT_ROUTE,
    NON_PRICE_COST_SCOPE,
    SPOT_ASK_EXIT_ROUTE,
    ExitMakerTerminalConfig,
    build_exit_maker_terminal_paths_v2,
    build_flat_non_price_cost_sensitivity,
    build_matched_exit_style_comparison,
    build_nominal_instant_cancel_v0_sensitivity,
    build_taker_taker_baseline_refs,
    canonical_exit_raw_candidate_fact_id,
)


DATE = "20260521"


def _action(
    policy_id: str,
    *,
    action_id: str = "q80",
    raw_id: str | None = None,
    kind: str = "opened",
) -> dict[str, object]:
    full = kind in {"opened", "hedge_failed"}
    partial = kind == "partial"
    any_fill: bool | None = True if full or partial else False
    hedge_status: str | None = "executable" if kind == "opened" else None
    if kind == "hedge_failed":
        hedge_status = "insufficient_depth"
    return {
        "Date": DATE,
        "ValueCode": "2303",
        "QuoteCode": "CCFF6",
        "route": "future_ask_spot_taker",
        "parameter_version": "boundary-v1",
        "lookup_action_id": action_id,
        "raw_order_fact_id": raw_id or f"raw-{policy_id}",
        "policy_generation_id": policy_id,
        "rank_bucket": "at_bbo",
        "queue_bucket": "00_0to1",
        "tod_bucket": "mid_0930_1200",
        "freshness_bucket": "fresh_le100ms",
        "intended_quantity": 1,
        "submit_recv_time_ns": 1_000_000_000,
        "terminal_recv_time_ns": 2_000_000_000,
        "queue_known": True,
        "any_fill": any_fill,
        "full_fill": full,
        "partial_fill": partial,
        "cancel_required": not full,
        "entry_hedge_status": hedge_status,
        "entry_hedge_signed_total_slippage_bp": (
            1.25 if hedge_status == "executable" else None
        ),
        "entry_hedge_contract_size_shares": 2_000,
        "entry_spot_price": 115.0,
    }


def _policy(
    entry_policy_id: str,
    *,
    exit_rule_id: str = "frozen_center",
    exit_route: str = FUTURE_BID_EXIT_ROUTE,
    branch: str = "flat_same_day",
    gross_twd: float | None = 2_000.0,
    projection_safe: bool | None = None,
    active_sibling_cancel_count: int = 0,
    prior_unacked_cancel_count: int = 0,
    cancel_ack_observed: bool = False,
    entry_raw_id: str | None = None,
    nominal_branch: str | None = None,
    exit_threshold_bp: float = 129.95,
) -> dict[str, object]:
    maker_quantity, hedge_quantity = (
        (1, 2) if exit_route == FUTURE_BID_EXIT_ROUTE else (2, 1)
    )
    if projection_safe is None:
        projection_safe = branch == "flat_same_day"
    raw_id = entry_raw_id or f"raw-{entry_policy_id}"
    winner_present = branch in {"flat_same_day", "cancel_race_unknown"}
    winner_epoch = 42 if winner_present else None
    winner_tick = 2_340 if winner_present else None
    winner_recv = 3_000_000_000 if winner_present else None
    winner_sequence = 17 if winner_present else None
    winner_row = 23 if winner_present else None
    winner_raw_id = (
        canonical_exit_raw_candidate_fact_id(
            entry_raw_order_fact_id=raw_id,
            exit_route=exit_route,
            spread_pair_epoch=winner_epoch,
            target_price_tick=winner_tick,
            submit_recv_time_ns=winner_recv,
            submit_event_sequence=winner_sequence,
            submit_row_index=winner_row,
        )
        if winner_present
        else None
    )
    terminal = branch == "flat_same_day"
    carry = branch in {
        "carry_at_eod_cancel_unconfirmed",
        "carry_at_eod_no_admission",
    }
    return {
        "Date": DATE,
        "ValueCode": "2303",
        "QuoteCode": "CCFF6",
        "entry_route": "future_ask_spot_taker",
        "entry_policy_generation_id": entry_policy_id,
        "entry_raw_order_fact_id": raw_id,
        "exit_rule_id": exit_rule_id,
        "exit_route": exit_route,
        "exit_policy_trial_id": (
            f"trial-{entry_policy_id}-{exit_rule_id}-{exit_route}"
        ),
        "exit_threshold_basis_bp": exit_threshold_bp,
        "exit_rule_source_asof_date": "20260520",
        "exit_lifecycle_policy_version": "static-one-order-v1",
        "exit_queue_scenario": "displayed-independent-v1",
        "exit_style": "maker_taker",
        "exit_maker_quantity": maker_quantity,
        "exit_hedge_quantity": hedge_quantity,
        "raw_candidate_count": 1 if winner_present else 0,
        "oco_winner_raw_candidate_fact_id": winner_raw_id,
        "oco_winner_spread_pair_epoch": winner_epoch,
        "oco_winner_target_price_tick": winner_tick,
        "oco_winner_submit_recv_time_ns": winner_recv,
        "oco_winner_submit_event_sequence": winner_sequence,
        "oco_winner_submit_row_index": winner_row,
        "oco_winner_raw_identity_excludes_rule": winner_present,
        "oco_position_projection_safe": projection_safe,
        "oco_active_sibling_cancel_count": active_sibling_cancel_count,
        "prior_unacked_cancel_count_before_winner": prior_unacked_cancel_count,
        "nominal_instant_cancel_v0_branch": nominal_branch or branch,
        "branch_status": branch,
        "terminal_outcome": terminal,
        "needs_next_session_label": carry,
        "exit_decision_time_ns": 5_000_000_000 if terminal else None,
        "gross_cycle_pnl_twd": gross_twd if terminal else None,
        "exit_hedge_status": "executable" if terminal else None,
        "exit_hedge_signed_total_slippage_bp": 2.5 if terminal else None,
        "cancel_ack_observed": cancel_ack_observed,
        "cancel_race_modeled": False,
        "joint_volume_allocated": False,
        "instant_cancel_v0": True,
    }


def _taker_exit(
    policy_id: str,
    *,
    exit_rule_id: str = "frozen_center",
    branch: str = "same_day_taker_exit",
    raw_id: str | None = None,
    exit_threshold_bp: float = 129.95,
    gross_twd: float = 1_000.0,
    exit_decision_time_ns: int = 5_000_000_000,
) -> dict[str, object]:
    same_day = branch == "same_day_taker_exit"
    carry = branch == "carry_at_eod"
    return {
        "Date": DATE,
        "ValueCode": "2303",
        "QuoteCode": "CCFF6",
        "route": "future_ask_spot_taker",
        "raw_order_fact_id": raw_id or f"raw-{policy_id}",
        "policy_generation_id": policy_id,
        "exit_rule_id": exit_rule_id,
        "exit_threshold_basis_bp": exit_threshold_bp,
        "exit_rule_source_asof_date": "20260520",
        "branch_status": branch,
        "terminal_outcome": same_day,
        "needs_next_session_label": carry,
        "exit_decision_time_ns": exit_decision_time_ns if same_day else None,
        "gross_cycle_pnl_twd": gross_twd if same_day else None,
    }


class ExitMakerTerminalV2Test(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ExitMakerTerminalConfig(
            lifecycle_policy_version="entry-layered-v1",
            queue_scenario="entry-displayed-independent-v1",
        )

    def test_mutually_exclusive_routes_and_ev_surface_contract(self) -> None:
        action = pl.DataFrame([_action("entry-1")])
        policies = pl.from_dicts(
            [
                _policy("entry-1", exit_route=FUTURE_BID_EXIT_ROUTE),
                _policy(
                    "entry-1",
                    exit_route=SPOT_ASK_EXIT_ROUTE,
                    branch="carry_at_eod_cancel_unconfirmed",
                    gross_twd=None,
                ),
            ],
            infer_schema_length=None,
        )
        paths = build_exit_maker_terminal_paths_v2(
            action, policies, self.config
        )
        self.assertEqual(paths.height, 2)
        by_route = {
            row["exit_route"]: row for row in paths.iter_rows(named=True)
        }

        closed = by_route[FUTURE_BID_EXIT_ROUTE]
        self.assertEqual(closed["outcome_status"], "known")
        self.assertEqual(closed["terminal_branch"], "same_day_target_exit")
        self.assertEqual(closed["same_day_exit_status"], "target_exit")
        self.assertAlmostEqual(
            closed["filled_cashflow_before_cost_bp"],
            2_000.0 / 230_000.0 * 10_000.0,
        )
        self.assertAlmostEqual(closed["capital_time_seconds"], 4.0)
        self.assertEqual(closed["exit_slippage_bp"], 2.5)
        self.assertIsNone(closed["fee_cost_bp"])
        self.assertFalse(closed["strict_ev_ready"])

        carry = by_route[SPOT_ASK_EXIT_ROUTE]
        self.assertEqual(carry["outcome_status"], "censored")
        self.assertIsNone(carry["terminal_branch"])
        self.assertEqual(carry["overnight_status"], "carried_open")
        self.assertIsNone(carry["filled_cashflow_before_cost_bp"])

        lookup = build_daily_ev_lookup(
            paths.filter(pl.col("exit_route") == FUTURE_BID_EXIT_ROUTE),
            [DATE, "20260522"],
            EVLookupConfig(
                lookback_sessions=1,
                min_history_sessions=1,
                min_group_sessions=1,
                min_known_paths=1,
                lookup_keys=EXIT_MAKER_PATH_LOOKUP_KEYS_V2,
            ),
            asof_dates=["20260522"],
        )
        self.assertEqual(lookup.height, 1)
        self.assertFalse(lookup.item(0, "ev_ready"))
        self.assertEqual(lookup.item(0, "ev_status"), "incomplete_cost_inputs")

    def test_q_aliases_share_physical_entry_but_not_policy_paths(self) -> None:
        actions = pl.from_dicts(
            [
                _action("entry-q50", action_id="q50", raw_id="shared-raw"),
                _action("entry-q80", action_id="q80", raw_id="shared-raw"),
            ],
            infer_schema_length=None,
        )
        policies = pl.from_dicts(
            [
                {
                    **_policy("entry-q50", entry_raw_id="shared-raw"),
                },
                {
                    **_policy("entry-q80", entry_raw_id="shared-raw"),
                },
            ],
            infer_schema_length=None,
        )
        paths = build_exit_maker_terminal_paths_v2(
            actions, policies, self.config
        )
        self.assertEqual(paths.height, 2)
        self.assertEqual(paths.select("path_id").n_unique(), 2)
        self.assertEqual(paths.select("physical_path_id").n_unique(), 1)
        self.assertEqual(
            paths.select("physical_exit_raw_order_fact_id").n_unique(), 1
        )
        self.assertTrue(paths["canonical_exit_raw_identity_validated"].all())
        self.assertEqual(
            set(paths["entry_lookup_action_id"].to_list()), {"q50", "q80"}
        )

    def test_center_lower_and_exit_routes_remain_alternative_policies(self) -> None:
        actions = pl.DataFrame([_action("entry-1")])
        policies = pl.from_dicts(
            [
                _policy(
                    "entry-1",
                    exit_rule_id=rule,
                    exit_route=route,
                )
                for rule in ("frozen_center", "frozen_lower")
                for route in (FUTURE_BID_EXIT_ROUTE, SPOT_ASK_EXIT_ROUTE)
            ],
            infer_schema_length=None,
        )
        paths = build_exit_maker_terminal_paths_v2(
            actions, policies, self.config
        )
        self.assertEqual(paths.height, 4)
        self.assertEqual(
            paths.select(
                "entry_policy_generation_id", "exit_rule_id", "exit_route"
            ).n_unique(),
            4,
        )
        self.assertEqual(paths.select("physical_path_id").n_unique(), 1)
        # The two routes are different physical orders, while center/lower
        # share the same canonical raw order within each route.
        self.assertEqual(
            paths.select("physical_exit_raw_order_fact_id").n_unique(), 2
        )
        self.assertFalse(
            paths["exit_rule_rows_safe_to_sum_as_independent"].any()
        )

        refs = build_taker_taker_baseline_refs(
            actions,
            pl.from_dicts(
                [
                    _taker_exit("entry-1", exit_rule_id=rule)
                    for rule in ("frozen_center", "frozen_lower")
                ]
            ),
        )
        comparison = build_matched_exit_style_comparison(paths, refs)
        self.assertEqual(comparison.baseline_refs.height, 2)
        self.assertEqual(comparison.pairs.height, 4)
        weight_sums = comparison.pairs.group_by(
            "taker_taker_baseline_ref_id"
        ).agg(pl.col("baseline_reference_weight").sum().alias("weight"))
        self.assertTrue(
            weight_sums.select(
                ((pl.col("weight") - 1.0).abs() <= 1e-12).all()
            ).item()
        )

    def test_nominal_exit_cancel_is_unknown_not_no_fill_terminal(self) -> None:
        action = pl.DataFrame([_action("entry-1")])
        policy = pl.DataFrame(
            [
                _policy(
                    "entry-1",
                    branch="no_fill_before_cancel_request",
                    gross_twd=None,
                )
            ]
        )
        path = build_exit_maker_terminal_paths_v2(
            action, policy, self.config
        ).row(0, named=True)
        self.assertEqual(path["outcome_status"], "unknown")
        self.assertIsNone(path["terminal_branch"])
        self.assertEqual(
            path["terminal_mapping_reason"],
            "exit_cancel_request_is_not_cancel_ack",
        )
        self.assertFalse(path["exit_cancel_ack_observed"])
        self.assertFalse(path["exit_cancel_race_modeled"])

    def test_no_admission_is_censored_carry(self) -> None:
        action = pl.DataFrame([_action("entry-1")])
        policy = pl.DataFrame(
            [
                _policy(
                    "entry-1",
                    branch="carry_at_eod_no_admission",
                    gross_twd=None,
                )
            ]
        )
        path = build_exit_maker_terminal_paths_v2(
            action, policy, self.config
        ).row(0, named=True)
        self.assertEqual(path["outcome_status"], "censored")
        self.assertIsNone(path["terminal_branch"])
        self.assertEqual(path["overnight_status"], "carried_open")

    def test_flat_requires_safe_projection_and_resolved_sibling_cancels(self) -> None:
        action = pl.DataFrame([_action("entry-1")])
        unsafe_projection = pl.DataFrame(
            [_policy("entry-1", projection_safe=False)]
        )
        with self.assertRaisesRegex(ValueError, "projection-safe"):
            build_exit_maker_terminal_paths_v2(
                action, unsafe_projection, self.config
            )

        pending_sibling = pl.DataFrame(
            [_policy("entry-1", active_sibling_cancel_count=1)]
        )
        with self.assertRaisesRegex(ValueError, "sibling cancel"):
            build_exit_maker_terminal_paths_v2(
                action, pending_sibling, self.config
            )

        resolved_sibling = pl.DataFrame(
            [
                _policy(
                    "entry-1",
                    active_sibling_cancel_count=1,
                    cancel_ack_observed=True,
                )
            ]
        )
        path = build_exit_maker_terminal_paths_v2(
            action, resolved_sibling, self.config
        ).row(0, named=True)
        self.assertEqual(path["outcome_status"], "known")

    def test_entry_state_and_policy_branch_must_agree(self) -> None:
        action = pl.DataFrame([_action("entry-1", kind="no_fill")])
        bad = pl.DataFrame([_policy("entry-1")])
        with self.assertRaisesRegex(ValueError, "established"):
            build_exit_maker_terminal_paths_v2(action, bad, self.config)

        # No established entry means no conditional exit denominator.  It is
        # valid for the study to emit no policy facts at all.
        empty = build_exit_maker_terminal_paths_v2(
            action,
            pl.DataFrame(schema={name: pl.String for name in ()}),
            self.config,
        )
        self.assertTrue(empty.is_empty())

    def test_zero_established_entry_keeps_empty_adapter_schemas_composable(self) -> None:
        action = pl.DataFrame([_action("entry-1", kind="no_fill")])
        paths = build_exit_maker_terminal_paths_v2(
            action, pl.DataFrame(), self.config
        )
        refs = build_taker_taker_baseline_refs(
            action,
            pl.DataFrame([_taker_exit("entry-1", branch="no_entry_fill")]),
        )
        comparison = build_matched_exit_style_comparison(paths, refs)
        self.assertTrue(paths.is_empty())
        self.assertTrue(refs.is_empty())
        self.assertTrue(comparison.pairs.is_empty())
        self.assertTrue(comparison.outcome_matrix.is_empty())
        self.assertTrue(build_flat_non_price_cost_sensitivity(paths).is_empty())
        self.assertTrue(
            build_nominal_instant_cancel_v0_sensitivity(
                pl.DataFrame(), action
            ).is_empty()
        )

        opened_action = pl.DataFrame([_action("opened-entry")])
        opened_paths = build_exit_maker_terminal_paths_v2(
            opened_action,
            pl.DataFrame([_policy("opened-entry")]),
            self.config,
        )
        opened_refs = build_taker_taker_baseline_refs(
            opened_action,
            pl.DataFrame([_taker_exit("opened-entry")]),
        )
        opened_comparison = build_matched_exit_style_comparison(
            opened_paths, opened_refs
        )
        self.assertEqual(paths.schema, opened_paths.schema)
        self.assertEqual(refs.schema, opened_refs.schema)
        self.assertEqual(comparison.pairs.schema, opened_comparison.pairs.schema)
        self.assertEqual(
            comparison.outcome_matrix.schema,
            opened_comparison.outcome_matrix.schema,
        )
        self.assertEqual(pl.concat([refs, opened_refs]).height, 1)
        self.assertEqual(
            pl.concat([comparison.pairs, opened_comparison.pairs]).height, 1
        )

    def test_taker_baseline_is_one_reference_shared_with_fractional_weight(self) -> None:
        actions = pl.DataFrame([_action("entry-1")])
        policies = pl.from_dicts(
            [
                _policy("entry-1", exit_route=FUTURE_BID_EXIT_ROUTE),
                _policy(
                    "entry-1",
                    exit_route=SPOT_ASK_EXIT_ROUTE,
                    branch="carry_at_eod_cancel_unconfirmed",
                    gross_twd=None,
                ),
            ],
            infer_schema_length=None,
        )
        paths = build_exit_maker_terminal_paths_v2(
            actions, policies, self.config
        )
        refs = build_taker_taker_baseline_refs(
            actions,
            pl.DataFrame([_taker_exit("entry-1")]),
        )
        comparison = build_matched_exit_style_comparison(paths, refs)

        self.assertEqual(comparison.baseline_refs.height, 1)
        self.assertEqual(comparison.pairs.height, 2)
        self.assertEqual(
            comparison.pairs.select("taker_taker_baseline_ref_id").n_unique(),
            1,
        )
        weights = comparison.pairs["baseline_reference_weight"].to_list()
        self.assertEqual(weights, [0.5, 0.5])
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertTrue(
            comparison.pairs["baseline_shared_across_exit_routes"].all()
        )
        self.assertFalse(
            comparison.pairs["route_copy_is_independent_baseline"].any()
        )

        future_pair = comparison.pairs.filter(
            pl.col("exit_route") == FUTURE_BID_EXIT_ROUTE
        ).row(0, named=True)
        self.assertTrue(future_pair["both_same_day_known"])
        # Maker future bid @117 vs T/T future ask @117.5 saves 1,000 TWD,
        # or 43.47826087 bp on the same 230,000 TWD entry notional.
        self.assertAlmostEqual(
            future_pair["paired_gross_delta_bp"],
            1_000.0 / 230_000.0 * 10_000.0,
        )
        self.assertAlmostEqual(
            comparison.outcome_matrix[
                "baseline_effective_observations"
            ].sum(),
            1.0,
        )

    def test_q_alias_and_route_copies_share_one_physical_baseline_weight(self) -> None:
        actions = pl.from_dicts(
            [
                _action("entry-q50", action_id="q50", raw_id="shared-raw"),
                _action("entry-q80", action_id="q80", raw_id="shared-raw"),
            ],
            infer_schema_length=None,
        )
        policies = pl.from_dicts(
            [
                _policy(
                    policy_id,
                    entry_raw_id="shared-raw",
                    exit_route=route,
                )
                for policy_id in ("entry-q50", "entry-q80")
                for route in (FUTURE_BID_EXIT_ROUTE, SPOT_ASK_EXIT_ROUTE)
            ],
            infer_schema_length=None,
        )
        paths = build_exit_maker_terminal_paths_v2(
            actions, policies, self.config
        )
        refs = build_taker_taker_baseline_refs(
            actions,
            pl.from_dicts(
                [
                    _taker_exit(policy_id, raw_id="shared-raw")
                    for policy_id in ("entry-q50", "entry-q80")
                ],
                infer_schema_length=None,
            ),
        )
        self.assertEqual(refs.height, 2)
        self.assertEqual(refs.select("physical_taker_taker_baseline_id").n_unique(), 1)
        self.assertEqual(refs["physical_entry_alias_weight"].to_list(), [0.5, 0.5])
        comparison = build_matched_exit_style_comparison(paths, refs)
        self.assertEqual(comparison.pairs.height, 4)
        self.assertEqual(
            comparison.pairs["physical_baseline_observation_weight"].to_list(),
            [0.25, 0.25, 0.25, 0.25],
        )
        self.assertAlmostEqual(
            comparison.pairs["physical_baseline_observation_weight"].sum(),
            1.0,
        )

    def test_q_aliases_with_different_thresholds_are_distinct_policy_baselines(
        self,
    ) -> None:
        actions = pl.from_dicts(
            [
                _action("entry-q50", action_id="q50", raw_id="shared-raw"),
                _action("entry-q80", action_id="q80", raw_id="shared-raw"),
            ],
            infer_schema_length=None,
        )
        policies = pl.from_dicts(
            [
                _policy(
                    policy_id,
                    entry_raw_id="shared-raw",
                    exit_route=route,
                    exit_threshold_bp=threshold,
                )
                for policy_id, threshold in (
                    ("entry-q50", 100.0),
                    ("entry-q80", 80.0),
                )
                for route in (FUTURE_BID_EXIT_ROUTE, SPOT_ASK_EXIT_ROUTE)
            ],
            infer_schema_length=None,
        )
        paths = build_exit_maker_terminal_paths_v2(
            actions, policies, self.config
        )
        refs = build_taker_taker_baseline_refs(
            actions,
            pl.from_dicts(
                [
                    _taker_exit(
                        "entry-q50",
                        raw_id="shared-raw",
                        exit_threshold_bp=100.0,
                        gross_twd=1_000.0,
                        exit_decision_time_ns=5_000_000_000,
                    ),
                    _taker_exit(
                        "entry-q80",
                        raw_id="shared-raw",
                        exit_threshold_bp=80.0,
                        gross_twd=1_500.0,
                        exit_decision_time_ns=6_000_000_000,
                    ),
                ],
                infer_schema_length=None,
            ),
        )

        self.assertEqual(refs.height, 2)
        self.assertEqual(
            refs.select("physical_entry_dependency_id").n_unique(), 1
        )
        self.assertEqual(
            refs.select("baseline_exit_policy_signature_id").n_unique(), 2
        )
        self.assertEqual(
            refs.select("physical_taker_taker_baseline_id").n_unique(), 2
        )
        self.assertEqual(refs["physical_entry_alias_weight"].to_list(), [1.0, 1.0])
        self.assertFalse(
            refs["entry_q_alias_rows_safe_to_sum_as_independent"].any()
        )

        comparison = build_matched_exit_style_comparison(paths, refs)
        self.assertEqual(comparison.pairs.height, 4)
        weights = comparison.pairs.group_by(
            "baseline_exit_policy_signature_id"
        ).agg(
            pl.col("physical_baseline_observation_weight")
            .sum()
            .alias("weight")
        )
        self.assertEqual(weights.height, 2)
        self.assertTrue(
            weights.select(
                ((pl.col("weight") - 1.0).abs() <= 1e-12).all()
            ).item()
        )

        mismatched_paths = paths.with_columns(
            pl.when(pl.col("entry_lookup_action_id") == "q80")
            .then(pl.lit(81.0))
            .otherwise(pl.col("exit_threshold_basis_bp"))
            .alias("exit_threshold_basis_bp")
        )
        with self.assertRaisesRegex(ValueError, "exit-policy signature"):
            build_matched_exit_style_comparison(mismatched_paths, refs)

    def test_flat_19bp_sensitivity_is_separate_and_does_not_double_slippage(self) -> None:
        action = pl.DataFrame([_action("entry-1")])
        policies = pl.from_dicts(
            [
                _policy("entry-1", exit_route=FUTURE_BID_EXIT_ROUTE),
                _policy(
                    "entry-1",
                    exit_route=SPOT_ASK_EXIT_ROUTE,
                    branch="carry_at_eod_cancel_unconfirmed",
                    gross_twd=None,
                ),
            ],
            infer_schema_length=None,
        )
        paths = build_exit_maker_terminal_paths_v2(
            action, policies, self.config
        )
        sensitivity = build_flat_non_price_cost_sensitivity(paths)
        self.assertEqual(sensitivity.height, 2)
        self.assertNotIn("fee_cost_bp", sensitivity.columns)
        self.assertTrue(sensitivity["analysis_only"].all())
        self.assertFalse(sensitivity["production_eligible"].any())
        self.assertFalse(sensitivity["eligible_for_strict_ev_lookup"].any())
        self.assertEqual(set(sensitivity["cost_scope"]), {NON_PRICE_COST_SCOPE})

        closed = sensitivity.filter(
            pl.col("exit_route") == FUTURE_BID_EXIT_ROUTE
        ).row(0, named=True)
        gross = 2_000.0 / 230_000.0 * 10_000.0
        self.assertEqual(closed["applied_cycle_cost_count"], 1)
        self.assertAlmostEqual(
            closed["conditional_net_after_assumed_cost_bp"], gross - 19.0
        )
        self.assertFalse(closed["hedge_slippage_subtracted_again"])
        self.assertFalse(closed["exit_slippage_subtracted_again"])

        carry = sensitivity.filter(
            pl.col("exit_route") == SPOT_ASK_EXIT_ROUTE
        ).row(0, named=True)
        self.assertEqual(carry["applied_cycle_cost_count"], 0)
        self.assertIsNone(carry["conditional_net_after_assumed_cost_bp"])
        self.assertEqual(paths["fee_cost_bp"].null_count(), paths.height)

    def test_nominal_instant_cancel_v0_is_analysis_only_and_keeps_strict_unknown(self) -> None:
        action = pl.DataFrame([_action("entry-1")])
        policy_row = {
            **_policy(
                "entry-1",
                branch="cancel_race_unknown",
                nominal_branch="flat_same_day",
                projection_safe=False,
                active_sibling_cancel_count=2,
                prior_unacked_cancel_count=1,
            ),
            # The nominal V0 projection keeps the actual maker/hedge prices
            # and PnL, while the strict branch refuses to call it terminal.
            "needs_next_session_label": True,
            "exit_decision_time_ns": 5_000_000_000,
            "gross_cycle_pnl_twd": 2_000.0,
            "exit_hedge_status": "executable",
            "exit_hedge_signed_total_slippage_bp": 2.5,
        }
        policy = pl.DataFrame([policy_row])
        strict = build_exit_maker_terminal_paths_v2(
            action, policy, self.config
        ).row(0, named=True)
        self.assertEqual(strict["outcome_status"], "unknown")
        self.assertIsNone(strict["filled_cashflow_before_cost_bp"])

        modeled = build_nominal_instant_cancel_v0_sensitivity(
            policy, action
        ).row(0, named=True)
        gross = 2_000.0 / 230_000.0 * 10_000.0
        self.assertEqual(modeled["strict_branch_status"], "cancel_race_unknown")
        self.assertEqual(modeled["modeled_outcome_status"], "known")
        self.assertEqual(modeled["modeled_terminal_branch"], "same_day_target_exit")
        self.assertEqual(modeled["oco_active_sibling_cancel_count"], 2)
        self.assertEqual(
            modeled["prior_unacked_cancel_count_before_winner"], 1
        )
        self.assertEqual(modeled["any_unacked_cancel_count"], 3)
        self.assertEqual(modeled["cancel_rate_numerator"], 1)
        self.assertTrue(modeled["strict_cancel_ambiguity_overridden"])
        self.assertAlmostEqual(modeled["gross_cycle_bp"], gross)
        self.assertAlmostEqual(
            modeled["conditional_net_after_assumed_cost_bp"], gross - 19.0
        )
        self.assertTrue(modeled["model_assumption"])
        self.assertTrue(modeled["analysis_only"])
        self.assertFalse(modeled["pathwise_ev_ready"])
        self.assertFalse(modeled["eligible_for_strict_ev_lookup"])
        self.assertFalse(modeled["exit_slippage_subtracted_again"])

        bad_model = pl.DataFrame([{**policy_row, "instant_cancel_v0": False}])
        with self.assertRaisesRegex(ValueError, "explicit instant_cancel_v0"):
            build_nominal_instant_cancel_v0_sensitivity(bad_model, action)

    def test_duplicate_policy_and_unsafe_lineage_fail_closed(self) -> None:
        action = pl.DataFrame([_action("entry-1")])
        row = _policy("entry-1")
        duplicates = pl.from_dicts([row, {**row, "exit_policy_trial_id": "other"}])
        with self.assertRaisesRegex(ValueError, "unique by entry alias"):
            build_exit_maker_terminal_paths_v2(
                action, duplicates, self.config
            )

        unsafe = pl.DataFrame(
            [{**row, "exit_rule_source_asof_date": DATE}]
        )
        with self.assertRaisesRegex(ValueError, "strictly before"):
            build_exit_maker_terminal_paths_v2(action, unsafe, self.config)

        rule_specific_raw_id = pl.DataFrame(
            [{**row, "oco_winner_raw_candidate_fact_id": "rule-specific-id"}]
        )
        with self.assertRaisesRegex(ValueError, "canonical rule-free"):
            build_exit_maker_terminal_paths_v2(
                action, rule_specific_raw_id, self.config
            )


if __name__ == "__main__":
    unittest.main()
