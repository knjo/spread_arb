"""Synthetic contracts for the contextual finite-horizon policy lookup."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import tempfile
import unittest

import polars as pl

from maker.src.quote_fill.finite_horizon_policy import (
    ActionRankingConfig,
    ContextualPolicyEVConfig,
    audit_current_fact_adapter_contract,
    audit_intraday_transition_readiness,
    build_contextual_policy_paths,
    build_prequential_contextual_lookup,
    bucket_policy_actions,
    canonicalize_policy_actions,
    compute_candidate_set_hash,
    rank_contextual_policy_actions,
)
from maker.src.quote_fill.finite_horizon_policy_cli import (
    run_finite_horizon_policy_publish,
)


def _prior(date: str) -> str:
    return {
        "20260101": "20251231",
        "20260102": "20260101",
        "20260103": "20260102",
        "20260104": "20260103",
        "20260105": "20260104",
        "20260106": "20260105",
        "20260107": "20260106",
    }[date]


def _horizon_end(date: str) -> str:
    return {
        "20260101": "20260102",
        "20260102": "20260103",
        "20260103": "20260104",
        "20260104": "20260105",
        "20260105": "20260106",
        "20260106": "20260107",
    }[date]


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stamp_candidates(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    if not rows:
        return rows
    for row in rows:
        row["candidate_set_expected_count"] = len(
            {str(item["policy_path_id"]) for item in rows}
        )
        row["candidate_set_complete"] = True
    digest = compute_candidate_set_hash(rows)
    for row in rows:
        row["candidate_set_hash"] = digest
    return rows


def _action(
    policy: str,
    date: str,
    *,
    value_code: str = "2317",
    alias: str = "q95-center",
    family: str = "center",
    legal_tick: int = 2301,
    alternative_set: str | None = None,
    **changes: object,
) -> dict[str, object]:
    alternative = alternative_set or f"alt-{policy}"
    row: dict[str, object] = {
        "policy_path_id": policy,
        "action_alias": alias,
        "decision_id": f"decision-{alternative}",
        "candidate_set_id": f"candidates-{alternative}",
        "candidate_set_expected_count": 1,
        "candidate_set_hash": "0" * 64,
        "candidate_set_complete": True,
        "alternative_set_id": alternative,
        "physical_entry_id": f"entry-{alternative}",
        "Date": date,
        "ValueCode": value_code,
        "QuoteCode": "DHFA6",
        "decision_phase": "position_establishment",
        "common_decision_clock_id": f"clock-{alternative}",
        "decision_time_ns": 10_000_000_000,
        "decision_event_sequence": 1,
        "decision_row_index": 1,
        "entry_route": "future_maker_spot_taker",
        "entry_q": "q95",
        "entry_basis_bucket": "entry_basis_tail",
        "locked_entry_state_bucket": "entry_locked",
        "route": "future_bid_spot_taker",
        "legal_tick": legal_tick,
        "relative_tick_offset": 0,
        "action_bucket": "maker_exit_at_legal_tick",
        "policy_family": family,
        "lifecycle_policy_version": "lifecycle-v1",
        "queue_scenario": "queue-base-v1",
        "intended_maker_quantity": 1,
        "hedge_delay_ns": 50_000_000,
        "remaining_minutes": 75.0,
        "holding_age_minutes": 0.0,
        "sessions_to_expiry": 8,
        "state_bucket": "basis_tail_queue_low",
        "peer_group": "electronics",
        "source_asof_date": _prior(date),
        "contains_target_day_outcome": False,
        "finite_horizon_sessions": 1,
        "terminal_policy_id": "forced-marketable-exit-v1",
        "terminal_policy_executable": True,
        "cost_profile_id": "tw-market-costs",
        "cost_profile_version": "cost-v1",
        "cost_profile_hash": _digest("cost-v1"),
        "cost_profile_source_asof_date": _prior(date),
        "cost_profile_contains_target_day_outcome": False,
        "legal_action": True,
        "action_effect": "select_policy",
        "inventory_position_units": 1,
        "reserved_inventory_units": 0,
        "action_inventory_delta_units": 0,
        "inventory_min_units": 0,
        "inventory_max_units": 1,
        "active_oco_sibling_count": 0,
        "replaces_active_oco": False,
        "is_risk_baseline": False,
        "oco_group_id": f"oco-{alternative}",
    }
    row.update(changes)
    return _stamp_candidates([row])[0]


_UNSET = object()


def _outcome(
    action: dict[str, object],
    *,
    label_end: str | None | object = _UNSET,
    gross: float = 20.0,
    cost: float = 2.0,
    origin: str = "fill",
    terminal: str = "known",
    **changes: object,
) -> dict[str, object]:
    known = terminal == "known"
    no_fill = origin == "no_fill"
    observed_label_end = action["Date"] if label_end is _UNSET else label_end
    accrued = cost * 0.4 if known else max(cost, 0.0)
    remainder = cost - accrued if known else None
    row: dict[str, object] = {
        "path_label_id": f"label-{action['policy_path_id']}",
        "policy_path_id": action["policy_path_id"],
        "alternative_set_id": action["alternative_set_id"],
        "physical_entry_id": action["physical_entry_id"],
        "Date": action["Date"],
        "ValueCode": action["ValueCode"],
        "QuoteCode": action["QuoteCode"],
        "decision_phase": action["decision_phase"],
        "entry_route": action["entry_route"],
        "entry_q": action["entry_q"],
        "entry_basis_bucket": action["entry_basis_bucket"],
        "locked_entry_state_bucket": action["locked_entry_state_bucket"],
        "route": action["route"],
        "legal_tick": action["legal_tick"],
        "relative_tick_offset": action["relative_tick_offset"],
        "action_bucket": action["action_bucket"],
        "policy_family": action["policy_family"],
        "lifecycle_policy_version": action["lifecycle_policy_version"],
        "queue_scenario": action["queue_scenario"],
        "intended_maker_quantity": action["intended_maker_quantity"],
        "hedge_delay_ns": action["hedge_delay_ns"],
        "finite_horizon_sessions": action["finite_horizon_sessions"],
        "label_horizon_end_date": _horizon_end(str(action["Date"])),
        "terminal_policy_id": action["terminal_policy_id"],
        "terminal_policy_executable": action["terminal_policy_executable"],
        "label_end_date": observed_label_end,
        "origin_execution_outcome": origin,
        "terminal_outcome_status": terminal,
        "terminal_branch": "cross_session_maker_exit" if known else None,
        "terminal_fill_observed": known,
        "actual_four_leg_gross_bp": gross if known else None,
        "actual_four_leg_complete": known,
        "gross_includes_observed_slippage": known,
        "immediate_price_pnl_bp": 0.0 if no_fill else (gross if known else None),
        "continuation_gross_value_bp": gross if no_fill and known else (0.0 if known else None),
        "continuation_state_id": f"next-{action['policy_path_id']}" if no_fill else None,
        "continuation_state_bucket": "carried_open" if no_fill else None,
        "known_accrued_cost_bp": accrued,
        "terminal_cost_remainder_bp": remainder,
        "terminal_cost_remainder_known": known,
        "cost_profile_id": action["cost_profile_id"],
        "cost_profile_version": action["cost_profile_version"],
        "cost_profile_hash": action["cost_profile_hash"],
        "cost_profile_source_asof_date": action[
            "cost_profile_source_asof_date"
        ],
        "cost_profile_contains_target_day_outcome": False,
        "cost_components_complete": known,
        "fee_cost_bp": cost if known else accrued,
        "tax_cost_bp": 0.0 if known else None,
        "commission_cost_bp": 0.0 if known else None,
        "financing_cost_bp": 0.0 if known else None,
        "overnight_cost_bp": 0.0 if known else None,
        "cancel_cost_bp": 0.0 if known else None,
        "emergency_cost_bp": 0.0 if known else None,
        "other_risk_cost_bp": 0.0 if known else None,
    }
    row.update(changes)
    return row


def _small_config(**changes: object) -> ContextualPolicyEVConfig:
    values: dict[str, object] = {
        "lookback_sessions": 4,
        "min_history_sessions": 2,
        "min_training_dates": 2,
        "min_policy_origins": 2,
        "min_terminal_fills": 2,
        "min_priced_terminals": 2,
        "min_label_maturity_coverage": 1.0,
        "max_unknown_rate": 0.0,
        "max_censor_rate": 0.0,
        "confidence_level": 0.95,
        "finite_horizon_sessions": 1,
        "embargo_sessions": 1,
        "production_like_window_sessions": 2,
        "min_shrinkage_child_origins": 2,
        "estimator_version": "synthetic-contextual-test-v1",
    }
    values.update(changes)
    return ContextualPolicyEVConfig(**values)


class ContextualPathIdentityTest(unittest.TestCase):
    def test_remaining_time_buckets_are_right_closed_countdowns(self) -> None:
        rows = []
        for index, value in enumerate([5.0, 15.0, 30.0, 60.0, 120.0, 121.0]):
            rows.append(
                _action(
                    f"bucket-{index}",
                    "20260101",
                    alternative_set=f"bucket-alt-{index}",
                    legal_tick=2301 + index,
                    remaining_minutes=value,
                )
            )
        bucketed = bucket_policy_actions(
            canonicalize_policy_actions(pl.DataFrame(rows)), _small_config()
        )
        self.assertEqual(
            bucketed["remaining_minutes_bucket"].to_list(),
            [
                "remaining:[0,5]",
                "remaining:(5,15]",
                "remaining:(15,30]",
                "remaining:(30,60]",
                "remaining:(60,120]",
                "remaining:(120,inf)",
            ],
        )

    def test_same_tick_aliases_are_one_sampling_path(self) -> None:
        first = _action("policy-1", "20260101", alias="q80")
        second = dict(first)
        second["action_alias"] = "q95"
        actions = pl.DataFrame([first, second])

        canonical = canonicalize_policy_actions(actions)
        self.assertEqual(canonical.height, 1)
        self.assertEqual(canonical.item(0, "action_alias_count"), 2)
        self.assertEqual(
            canonical.row(0, named=True)["action_aliases"], ["q80", "q95"]
        )

        paths = build_contextual_policy_paths(
            actions,
            pl.DataFrame([_outcome(first)]),
            _small_config(),
        )
        self.assertEqual(paths.height, 1)
        self.assertEqual(paths.item(0, "realized_path_value_bp"), 18.0)

    def test_split_physical_identity_and_join_mismatch_fail_closed(self) -> None:
        first = _action("policy-1", "20260101")
        split = dict(first)
        split.update(policy_path_id="policy-2", action_alias="same-tick-other-id")
        _stamp_candidates([first, split])
        with self.assertRaisesRegex(ValueError, "split across policy_path_id"):
            canonicalize_policy_actions(pl.DataFrame([first, split]))

        single = _action("policy-single", "20260101")
        mismatched = _outcome(single)
        mismatched["legal_tick"] = 2302
        with self.assertRaisesRegex(ValueError, "identities disagree"):
            build_contextual_policy_paths(
                pl.DataFrame([single]),
                pl.DataFrame([mismatched]),
                _small_config(),
            )

    def test_no_fill_is_zero_immediate_pnl_plus_continuation(self) -> None:
        action = _action("policy-no-fill", "20260101")
        outcome = _outcome(action, gross=14.0, cost=3.0, origin="no_fill")
        path = build_contextual_policy_paths(
            pl.DataFrame([action]), pl.DataFrame([outcome]), _small_config()
        ).row(0, named=True)
        self.assertEqual(path["immediate_price_pnl_bp"], 0.0)
        self.assertEqual(path["continuation_gross_value_bp"], 14.0)
        self.assertEqual(path["realized_path_value_bp"], 11.0)
        self.assertEqual(path["decomposed_path_value_bp"], 11.0)

        outcome["immediate_price_pnl_bp"] = 1.0
        with self.assertRaisesRegex(ValueError, "exactly zero"):
            build_contextual_policy_paths(
                pl.DataFrame([action]), pl.DataFrame([outcome]), _small_config()
            )

    def test_terminal_path_cannot_be_pasted_onto_later_holding_state(self) -> None:
        action = _action(
            "later-state",
            "20260101",
            decision_phase="intraday_manage",
            holding_age_minutes=30.0,
        )
        with self.assertRaisesRegex(ValueError, "require transition facts"):
            build_contextual_policy_paths(
                pl.DataFrame([action]),
                pl.DataFrame([_outcome(action)]),
                _small_config(),
            )

    def test_complete_candidate_hash_count_and_common_clock_are_mandatory(self) -> None:
        first = _action(
            "candidate-center",
            "20260101",
            alternative_set="candidate-alt",
            family="center",
        )
        second = _action(
            "candidate-lower",
            "20260101",
            alternative_set="candidate-alt",
            family="lower",
            legal_tick=2300,
        )
        _stamp_candidates([first, second])
        self.assertEqual(canonicalize_policy_actions(pl.DataFrame([first, second])).height, 2)

        missing = dict(first)
        with self.assertRaisesRegex(ValueError, "candidate count"):
            canonicalize_policy_actions(pl.DataFrame([missing]))

        bad_hash = [dict(first), dict(second)]
        for row in bad_hash:
            row["candidate_set_hash"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "candidate_set_hash does not match"):
            canonicalize_policy_actions(pl.DataFrame(bad_hash))

        different_clock = [dict(first), dict(second)]
        different_clock[1]["decision_event_sequence"] = 2
        _stamp_candidates(different_clock)
        with self.assertRaisesRegex(ValueError, "decision_event_sequence"):
            canonicalize_policy_actions(pl.DataFrame(different_clock))

    def test_component_costs_are_strict_but_unresolved_accrual_is_retained(self) -> None:
        action = _action("strict-cost", "20260101")
        inconsistent = _outcome(action, gross=10.0, cost=2.0)
        inconsistent["fee_cost_bp"] = 3.0
        with self.assertRaisesRegex(ValueError, "accrued plus terminal remainder"):
            build_contextual_policy_paths(
                pl.DataFrame([action]),
                pl.DataFrame([inconsistent]),
                _small_config(),
            )

        unresolved = _outcome(
            action,
            terminal="unknown",
            origin="unknown",
            label_end=None,
            cost=1.75,
        )
        path = build_contextual_policy_paths(
            pl.DataFrame([action]), pl.DataFrame([unresolved]), _small_config()
        ).row(0, named=True)
        self.assertEqual(path["known_accrued_cost_bp"], 1.75)
        self.assertEqual(
            path["known_accrued_cost_retained_when_terminal_unresolved_bp"],
            1.75,
        )
        self.assertIsNone(path["non_price_cost_bp"])
        self.assertFalse(path["terminal_cashflow_priced"])


class PrequentialContextualLookupTest(unittest.TestCase):
    sessions = [
        "20260101",
        "20260102",
        "20260103",
        "20260104",
        "20260105",
        "20260106",
        "20260107",
    ]

    def test_target_day_and_unmatured_origin_do_not_leak(self) -> None:
        a1 = _action("p1", "20260101", alternative_set="alt-1")
        a2 = _action("p2", "20260102", alternative_set="alt-2")
        target = _action("target", "20260103", alternative_set="alt-target")
        o1 = _outcome(a1, label_end="20260102", gross=12.0, cost=2.0)
        # A missing old terminal classification remains in the origin
        # denominator and blocks a finite EV; it is never silently dropped.
        o2 = _outcome(
            a2,
            label_end=None,
            terminal="censored",
            origin="censored",
            cost=1.25,
        )
        ot = _outcome(target, label_end="20260103", gross=99999.0, cost=0.0)
        paths = build_contextual_policy_paths(
            pl.DataFrame([a1, a2, target]),
            pl.DataFrame([o1, o2, ot]),
            _small_config(),
        )
        lookup = build_prequential_contextual_lookup(
            paths,
            self.sessions,
            _small_config(),
            asof_dates=["20260104"],
        )
        exact = lookup.filter(pl.col("fallback_name") == "exact").row(
            0, named=True
        )
        self.assertEqual(exact["n_policy_origins"], 2)
        self.assertEqual(exact["n_labels_matured"], 1)
        self.assertEqual(exact["n_labels_pending"], 1)
        self.assertEqual(exact["n_outcome_fill"], 1)
        self.assertEqual(exact["n_outcome_no_fill"], 0)
        self.assertAlmostEqual(exact["origin_probability_sum"], 1.0)
        self.assertEqual(exact["conditional_known_ev_mean_bp"], 10.0)
        self.assertIsNone(exact["identified_ev_mean_bp"])
        self.assertIsNone(exact["one_sided_lcb_bp"])
        self.assertFalse(exact["ev_ready"])
        self.assertEqual(
            exact["cashflow_identification_status"],
            "unbounded_without_terminal_value_limits",
        )

        changed = dict(ot)
        changed["actual_four_leg_gross_bp"] = -1000.0
        changed["immediate_price_pnl_bp"] = -1000.0
        changed_paths = build_contextual_policy_paths(
            pl.DataFrame([a1, a2, target]),
            pl.DataFrame([o1, o2, changed]),
            _small_config(),
        )
        changed_exact = build_prequential_contextual_lookup(
            changed_paths,
            self.sessions,
            _small_config(),
            asof_dates=["20260104"],
        ).filter(pl.col("fallback_name") == "exact").row(0, named=True)
        for column in (
            "n_policy_origins",
            "n_labels_matured",
            "n_labels_pending",
            "conditional_known_ev_mean_bp",
            "identified_ev_mean_bp",
            "one_sided_lcb_bp",
            "ev_ready",
            "ev_status",
        ):
            self.assertEqual(exact[column], changed_exact[column])

    def test_null_label_end_is_retained_as_pending_not_rejected(self) -> None:
        mature_action = _action("mature", "20260101", alternative_set="mature-alt")
        pending_action = _action(
            "pending", "20260102", alternative_set="pending-alt"
        )
        mature = _outcome(mature_action, gross=10.0, cost=0.0)
        pending = _outcome(
            pending_action,
            terminal="censored",
            origin="censored",
            label_end_date=None,
        )
        # The helper defaults a missing label_end to Date, so set null after it
        # constructs the otherwise valid unresolved path.
        pending["label_end_date"] = None
        paths = build_contextual_policy_paths(
            pl.DataFrame([mature_action, pending_action]),
            pl.DataFrame([mature, pending]),
            _small_config(),
        )
        exact = build_prequential_contextual_lookup(
            paths,
            self.sessions,
            _small_config(),
            asof_dates=["20260104"],
        ).filter(pl.col("fallback_name") == "exact").row(0, named=True)
        self.assertEqual(exact["n_policy_origins"], 2)
        self.assertEqual(exact["n_labels_pending"], 1)
        self.assertEqual(exact["n_outcome_censored"], 0)
        self.assertFalse(exact["terminal_cashflow_point_identified"])

    def test_mean_and_date_clustered_one_sided_lcb_not_completed_median(self) -> None:
        actions = [
            _action(f"p{index}", date, alternative_set=f"alt-{index}")
            for index, date in enumerate(
                ["20260101", "20260102", "20260103"], 1
            )
        ]
        nets = [100.0, -10.0, -10.0]
        outcomes = [
            _outcome(action, gross=net, cost=0.0)
            for action, net in zip(actions, nets, strict=True)
        ]
        config = _small_config(
            min_training_dates=3,
            min_policy_origins=3,
            min_terminal_fills=3,
            min_priced_terminals=3,
        )
        paths = build_contextual_policy_paths(
            pl.DataFrame(actions), pl.DataFrame(outcomes), config
        )
        exact = build_prequential_contextual_lookup(
            paths,
            self.sessions,
            config,
            asof_dates=["20260105"],
        ).filter(pl.col("fallback_name") == "exact").row(0, named=True)
        self.assertAlmostEqual(exact["identified_ev_mean_bp"], 80.0 / 3.0)
        self.assertNotEqual(exact["identified_ev_mean_bp"], -10.0)
        self.assertLess(exact["one_sided_lcb_bp"], exact["identified_ev_mean_bp"])
        self.assertEqual(exact["n_date_clusters_for_ci"], 3)
        self.assertGreater(exact["one_sided_critical_value"], 1.6448536269514722)
        self.assertEqual(
            exact["lcb_reference_distribution"], "student_t_df_G_minus_1"
        )
        self.assertTrue(exact["ev_ready"])

    def test_physical_origin_support_is_not_inflated_by_two_tick_paths(self) -> None:
        actions: list[dict[str, object]] = []
        outcomes: list[dict[str, object]] = []
        for origin_index, date in enumerate(["20260101", "20260102"], 1):
            pair = [
                _action(
                    f"origin-{origin_index}-tick-{tick}",
                    date,
                    alternative_set=f"origin-alt-{origin_index}",
                    family="center",
                    legal_tick=tick,
                    relative_tick_offset=0,
                )
                for tick in (2301, 2302)
            ]
            _stamp_candidates(pair)
            actions.extend(pair)
            outcomes.extend(_outcome(action, gross=10.0, cost=1.0) for action in pair)
        config = _small_config(
            min_policy_origins=2,
            min_terminal_fills=3,
            min_priced_terminals=3,
        )
        paths = build_contextual_policy_paths(
            pl.DataFrame(actions), pl.DataFrame(outcomes), config
        )
        row = build_prequential_contextual_lookup(
            paths, self.sessions, config, asof_dates=["20260104"]
        ).filter(pl.col("fallback_name") == "product_primary").row(0, named=True)
        self.assertEqual(row["n_physical_origins"], 2)
        self.assertEqual(row["n_policy_paths_diagnostic"], 4)
        self.assertEqual(row["n_terminal_fills"], 2)
        self.assertEqual(row["n_terminal_priced"], 2)
        self.assertFalse(row["terminal_fill_support_gate"])
        self.assertFalse(row["priced_terminal_support_gate"])
        self.assertFalse(row["ev_ready"])

    def test_finite_horizon_calendar_and_terminal_policy_fail_closed(self) -> None:
        first = _action("horizon-1", "20260101", alternative_set="horizon-alt-1")
        bad_horizon = _outcome(first)
        bad_horizon["label_horizon_end_date"] = "20260103"
        paths = build_contextual_policy_paths(
            pl.DataFrame([first]), pl.DataFrame([bad_horizon]), _small_config()
        )
        with self.assertRaisesRegex(ValueError, "origin plus finite H"):
            build_prequential_contextual_lookup(
                paths, self.sessions, _small_config(), asof_dates=["20260104"]
            )

        actions = [
            _action(
                f"terminal-{index}",
                date,
                alternative_set=f"terminal-alt-{index}",
                terminal_policy_executable=False,
            )
            for index, date in enumerate(["20260101", "20260102"], 1)
        ]
        outcomes = [_outcome(action, gross=8.0, cost=1.0) for action in actions]
        config = _small_config()
        blocked_paths = build_contextual_policy_paths(
            pl.DataFrame(actions), pl.DataFrame(outcomes), config
        )
        exact = build_prequential_contextual_lookup(
            blocked_paths, self.sessions, config, asof_dates=["20260104"]
        ).filter(pl.col("fallback_name") == "exact").row(0, named=True)
        self.assertFalse(exact["executable_terminal_policy_gate"])
        self.assertFalse(exact["ev_ready"])
        self.assertEqual(
            exact["direct_ev_status"],
            "finite_horizon_terminal_policy_not_executable",
        )

    def test_kappa_50_shrinkage_uses_supported_product_parent(self) -> None:
        actions: list[dict[str, object]] = []
        outcomes: list[dict[str, object]] = []
        for index in range(200):
            action = _action(
                f"shrink-{index}",
                "20260101" if index % 2 == 0 else "20260102",
                alternative_set=f"shrink-alt-{index}",
                legal_tick=2301 if index < 30 else 2302,
                relative_tick_offset=0,
            )
            actions.append(action)
            outcomes.append(
                _outcome(action, gross=30.0 if index < 30 else 10.0, cost=0.0)
            )
        config = _small_config(
            min_policy_origins=200,
            min_terminal_fills=200,
            min_priced_terminals=200,
            min_shrinkage_child_origins=30,
            shrinkage_kappa=50.0,
        )
        paths = build_contextual_policy_paths(
            pl.DataFrame(actions), pl.DataFrame(outcomes), config
        )
        lookup = build_prequential_contextual_lookup(
            paths, self.sessions, config, asof_dates=["20260104"]
        )
        child = lookup.filter(
            (pl.col("fallback_name") == "exact")
            & (pl.col("legal_tick_bucket") == "tick:2301")
        ).row(0, named=True)
        self.assertFalse(child["direct_ev_ready"])
        self.assertTrue(child["ev_ready"])
        self.assertEqual(child["shrinkage_source"], "child_plus_parent")
        self.assertAlmostEqual(child["shrinkage_weight"], 30.0 / 80.0)
        self.assertEqual(
            child["ev_status"], "ready_kappa_hierarchically_shrunk"
        )
        self.assertLess(child["posterior_ev_mean_bp"], 30.0)
        self.assertGreater(child["posterior_ev_mean_bp"], 13.0)

    def test_default_burn_in_and_production_window_are_explicit(self) -> None:
        config = ContextualPolicyEVConfig()
        self.assertEqual(config.min_history_sessions, 40)
        self.assertEqual(config.production_like_window_sessions, 60)
        self.assertEqual(config.shrinkage_kappa, 50.0)
        self.assertEqual(config.min_peer_global_origins, 500)
        self.assertEqual(config.min_peer_global_products, 5)


class ContextualActionRankingTest(unittest.TestCase):
    sessions = [
        "20260101",
        "20260102",
        "20260103",
        "20260104",
        "20260105",
        "20260106",
    ]

    _cached_lookup: tuple[pl.DataFrame, ContextualPolicyEVConfig] | None = None

    @classmethod
    def _lookup_for_families(cls) -> tuple[pl.DataFrame, ContextualPolicyEVConfig]:
        if cls._cached_lookup is not None:
            return cls._cached_lookup
        actions: list[dict[str, object]] = []
        outcomes: list[dict[str, object]] = []
        family_gross = {"center": 30.0, "lower": 20.0, "safe": 10.0}
        for family, gross in family_gross.items():
            for index in range(500):
                date = ["20260101", "20260102", "20260103"][index % 3]
                product = ["2317", "2303", "2454", "2382", "2881"][
                    index % 5
                ]
                action = _action(
                    f"hist-{family}-{index}",
                    date,
                    value_code=product,
                    family=family,
                    alternative_set=f"hist-alt-{family}-{index}",
                )
                actions.append(action)
                outcomes.append(_outcome(action, gross=gross, cost=2.0))
        config = _small_config()
        paths = build_contextual_policy_paths(
            pl.DataFrame(actions), pl.DataFrame(outcomes), config
        )
        lookup = build_prequential_contextual_lookup(
            paths,
            cls.sessions,
            config,
            asof_dates=["20260105"],
        )
        cls._cached_lookup = (lookup, config)
        return cls._cached_lookup

    def test_global_fallback_then_inventory_and_oco_constraints(self) -> None:
        lookup, config = self._lookup_for_families()
        product_parent = lookup.filter(
            (pl.col("fallback_name") == "product_primary")
            & (pl.col("ValueCode") == "2317")
            & (pl.col("policy_family") == "center")
        ).row(0, named=True)
        self.assertEqual(product_parent["parent_fallback_name"], "peer_primary")
        self.assertEqual(product_parent["shrinkage_source"], "child_plus_parent")
        base = _action(
            "live-center",
            "20260105",
            value_code="9999",
            family="center",
            alternative_set="live-alt",
            peer_group="unseen-peer",
            action_inventory_delta_units=1,
            active_oco_sibling_count=1,
        )
        # Same decision/alternative set, different frozen policies.
        lower = _action(
            "live-lower",
            "20260105",
            value_code="9999",
            family="lower",
            alternative_set="live-alt",
            peer_group="unseen-peer",
            active_oco_sibling_count=1,
            action_effect="submit_order",
            replaces_active_oco=False,
        )
        safe = _action(
            "live-safe",
            "20260105",
            value_code="9999",
            family="safe",
            alternative_set="live-alt",
            peer_group="unseen-peer",
            active_oco_sibling_count=1,
            action_effect="wait",
            is_risk_baseline=True,
        )
        _stamp_candidates([base, lower, safe])
        result = rank_contextual_policy_actions(
            pl.DataFrame([base, lower, safe]),
            lookup,
            config,
            ActionRankingConfig(minimum_lcb_bp=0.0),
        )
        rows = {
            row["policy_family"]: row
            for row in result.scored_actions.iter_rows(named=True)
        }
        self.assertEqual(rows["center"]["lookup_fallback_name"], "global_primary")
        self.assertFalse(rows["center"]["inventory_constraint_satisfied"])
        self.assertFalse(rows["center"]["oco_constraint_satisfied"])
        self.assertFalse(rows["lower"]["oco_constraint_satisfied"])
        self.assertTrue(rows["safe"]["selected"])
        decision = result.decisions.row(0, named=True)
        self.assertEqual(decision["selected_policy_family"], "safe")
        self.assertFalse(decision["no_trade"])
        self.assertEqual(
            result.scored_actions.filter(pl.col("selected")).height, 1
        )

    def test_cost_profile_mismatch_is_no_trade(self) -> None:
        lookup, config = self._lookup_for_families()
        live = _action(
            "live-cost-v2",
            "20260105",
            value_code="9999",
            family="center",
            alternative_set="live-cost-alt",
            cost_profile_version="cost-v2",
            cost_profile_hash=_digest("cost-v2"),
            inventory_position_units=0,
        )
        result = rank_contextual_policy_actions(
            pl.DataFrame([live]), lookup, config
        )
        self.assertFalse(result.scored_actions.item(0, "lookup_found"))
        self.assertTrue(result.decisions.item(0, "no_trade"))
        self.assertEqual(
            result.decisions.item(0, "decision_disposition"),
            "no_trade_no_ready_lookup",
        )

    def test_open_inventory_requires_priced_wait_and_ignores_admission_threshold(self) -> None:
        lookup, config = self._lookup_for_families()
        unsafe = _action(
            "live-no-baseline",
            "20260105",
            value_code="9999",
            family="safe",
            alternative_set="live-no-baseline-alt",
            peer_group="unseen-peer",
            action_effect="select_policy",
            is_risk_baseline=False,
        )
        with self.assertRaisesRegex(ValueError, "explicit priced WAIT baseline"):
            rank_contextual_policy_actions(pl.DataFrame([unsafe]), lookup, config)

        wait = _action(
            "live-priced-wait",
            "20260105",
            value_code="9999",
            family="safe",
            alternative_set="live-priced-wait-alt",
            peer_group="unseen-peer",
            action_effect="wait",
            is_risk_baseline=True,
        )
        result = rank_contextual_policy_actions(
            pl.DataFrame([wait]),
            lookup,
            config,
            ActionRankingConfig(minimum_lcb_bp=1_000_000.0),
        )
        self.assertTrue(result.scored_actions.item(0, "selected"))
        self.assertFalse(result.decisions.item(0, "no_trade"))
        self.assertEqual(
            result.decisions.item(0, "decision_disposition"),
            "selected_open_inventory_risk_policy",
        )

    def test_tiny_peer_and_global_pool_cannot_price_unseen_product(self) -> None:
        actions: list[dict[str, object]] = []
        outcomes: list[dict[str, object]] = []
        for index, (date, product) in enumerate(
            [("20260101", "2317"), ("20260102", "2303")], 1
        ):
            action = _action(
                f"tiny-pool-{index}",
                date,
                value_code=product,
                alternative_set=f"tiny-pool-alt-{index}",
            )
            actions.append(action)
            outcomes.append(_outcome(action, gross=20.0, cost=1.0))
        config = _small_config()
        lookup = build_prequential_contextual_lookup(
            build_contextual_policy_paths(
                pl.DataFrame(actions), pl.DataFrame(outcomes), config
            ),
            self.sessions,
            config,
            asof_dates=["20260104"],
        )
        global_row = lookup.filter(
            pl.col("fallback_name") == "global_primary"
        ).row(0, named=True)
        self.assertFalse(global_row["peer_global_pool_support_gate"])
        self.assertFalse(global_row["ev_ready"])
        live = _action(
            "unseen-small-pool",
            "20260104",
            value_code="9999",
            alternative_set="unseen-small-pool-alt",
            peer_group="unseen-peer",
            inventory_position_units=0,
        )
        result = rank_contextual_policy_actions(pl.DataFrame([live]), lookup, config)
        self.assertTrue(result.decisions.item(0, "no_trade"))
        self.assertFalse(result.scored_actions.item(0, "ready_lookup_found"))


def _transition(
    transition_id: str,
    *,
    state: str,
    clock: str,
    action: str,
    outcome: str,
    terminal: bool,
    next_state: str | None = None,
    next_clock: str | None = None,
    decision_time: int = 100,
    **changes: object,
) -> dict[str, object]:
    event_sequence = 1 if decision_time == 100 else 2
    row_index = event_sequence
    row: dict[str, object] = {
        "transition_id": transition_id,
        "Date": "20260101",
        "physical_entry_id": "entry-1",
        "shared_market_path_id": "market-path-1",
        "decision_clock_id": clock,
        "decision_time_ns": decision_time,
        "decision_event_sequence": event_sequence,
        "decision_row_index": row_index,
        "state_id": state,
        "action_id": f"{state}-{action}",
        "action_effect": action,
        "entry_route": "future_maker_spot_taker",
        "entry_q": "q95",
        "entry_basis_bucket": "entry_basis_tail",
        "locked_entry_state_bucket": "entry_locked",
        "route": "future_bid_spot_taker",
        "legal_tick": 2301,
        "relative_tick_offset": 0,
        "lifecycle_policy_version": "lifecycle-v1",
        "queue_scenario": "queue-base-v1",
        "intended_maker_quantity": 1,
        "hedge_delay_ns": 50_000_000,
        "remaining_minutes": 60.0,
        "holding_age_minutes": 0.0,
        "sessions_to_expiry": 8,
        "state_bucket": "tail",
        "finite_horizon_sessions": 1,
        "horizon_end_date": "20260102",
        "terminal_policy_id": "forced-marketable-exit-v1",
        "terminal_policy_executable": True,
        "common_decision_clock": True,
        "complete_legal_action_set": True,
        "outcome_class": outcome,
        "immediate_price_pnl_bp": 0.0 if outcome == "no_fill" else 5.0,
        "known_accrued_cost_bp": 0.0,
        "terminal_cost_remainder_bp": 0.0,
        "terminal_cost_remainder_known": True,
        "cost_profile_id": "tw-market-costs",
        "cost_profile_version": "cost-v1",
        "cost_profile_hash": _digest("cost-v1"),
        "cost_profile_source_asof_date": "20251231",
        "cost_profile_contains_target_day_outcome": False,
        "cost_components_complete": True,
        "fee_cost_bp": 0.0,
        "tax_cost_bp": 0.0,
        "commission_cost_bp": 0.0,
        "financing_cost_bp": 0.0,
        "overnight_cost_bp": 0.0,
        "cancel_cost_bp": 0.0,
        "emergency_cost_bp": 0.0,
        "other_risk_cost_bp": 0.0,
        "next_state_id": next_state,
        "next_Date": "20260101" if next_state else None,
        "next_physical_entry_id": "entry-1" if next_state else None,
        "next_shared_market_path_id": "market-path-1" if next_state else None,
        "next_decision_clock_id": next_clock,
        "next_decision_time_ns": decision_time + 100 if next_state else None,
        "next_decision_event_sequence": event_sequence + 1 if next_state else None,
        "next_decision_row_index": row_index + 1 if next_state else None,
        "terminal_transition": terminal,
        "inventory_feasible": True,
        "oco_feasible": True,
        "sequential_policy_replay_complete": True,
    }
    row.update(changes)
    return row


def _valid_transitions() -> list[dict[str, object]]:
    return [
        _transition(
            "t0-wait",
            state="s0",
            clock="c0",
            action="wait",
            outcome="no_fill",
            terminal=False,
            next_state="s1",
            next_clock="c1",
        ),
        _transition(
            "t0-submit",
            state="s0",
            clock="c0",
            action="submit_order",
            outcome="fill",
            terminal=True,
        ),
        _transition(
            "t1-wait",
            state="s1",
            clock="c1",
            action="wait",
            outcome="fill",
            terminal=True,
            decision_time=200,
        ),
        _transition(
            "t1-submit",
            state="s1",
            clock="c1",
            action="submit_order",
            outcome="fill",
            terminal=True,
            decision_time=200,
        ),
    ]


class IntradayTransitionReadinessTest(unittest.TestCase):
    def test_missing_and_valid_transition_contract_are_explicit(self) -> None:
        missing = audit_intraday_transition_readiness(None).row(0, named=True)
        self.assertFalse(missing["transition_data_ready"])
        self.assertEqual(missing["readiness_status"], "missing_transition_facts")

        rows = _valid_transitions()
        ready = audit_intraday_transition_readiness(pl.DataFrame(rows)).row(
            0, named=True
        )
        self.assertTrue(ready["transition_data_ready"])
        self.assertFalse(ready["intraday_controller_ready"])
        self.assertEqual(
            ready["readiness_status"],
            "transition_data_ready_sequential_estimator_not_implemented",
        )

        rows[0]["immediate_price_pnl_bp"] = 1.0
        invalid = audit_intraday_transition_readiness(pl.DataFrame(rows)).row(
            0, named=True
        )
        self.assertFalse(invalid["transition_data_ready"])
        self.assertEqual(
            invalid["readiness_status"],
            "no_fill_transition_loses_zero_pnl_continuation",
        )

    def test_every_nonterminal_branch_including_fill_requires_full_next_tuple(self) -> None:
        rows = _valid_transitions()
        rows[1]["terminal_transition"] = False
        result = audit_intraday_transition_readiness(pl.DataFrame(rows)).row(
            0, named=True
        )
        self.assertFalse(result["transition_data_ready"])
        self.assertEqual(
            result["readiness_status"],
            "nonterminal_transition_missing_full_next_tuple",
        )

    def test_next_state_join_is_exact_composite_not_bare_state_id(self) -> None:
        rows = _valid_transitions()
        rows[0]["next_Date"] = "20260102"
        result = audit_intraday_transition_readiness(pl.DataFrame(rows)).row(
            0, named=True
        )
        self.assertFalse(result["transition_data_ready"])
        self.assertEqual(
            result["readiness_status"],
            "exact_next_state_absent_from_transition_table",
        )

    def test_terminal_has_null_next_tuple_and_cursor_moves_forward(self) -> None:
        rows = _valid_transitions()
        for name in (
            "next_state_id",
            "next_Date",
            "next_physical_entry_id",
            "next_shared_market_path_id",
            "next_decision_clock_id",
            "next_decision_time_ns",
            "next_decision_event_sequence",
            "next_decision_row_index",
        ):
            rows[1][name] = rows[0][name]
        terminal = audit_intraday_transition_readiness(pl.DataFrame(rows)).row(
            0, named=True
        )
        self.assertEqual(
            terminal["readiness_status"],
            "terminal_transition_must_have_null_next_tuple",
        )

        backward = _valid_transitions()
        backward[0]["next_decision_time_ns"] = 50
        result = audit_intraday_transition_readiness(pl.DataFrame(backward)).row(
            0, named=True
        )
        self.assertFalse(result["transition_data_ready"])
        self.assertEqual(
            result["readiness_status"],
            "next_decision_cursor_is_not_strictly_forward",
        )


class CurrentFactAdapterContractTest(unittest.TestCase):
    def test_adapter_never_invents_component_costs_or_transitions(self) -> None:
        missing = audit_current_fact_adapter_contract(
            action_columns={"policy_path_id", "non_price_cost_bp"},
            terminal_path_columns={"path_label_id", "non_price_cost_bp"},
        ).row(0, named=True)
        self.assertFalse(missing["fixed_policy_adapter_ready"])
        self.assertFalse(missing["intraday_transition_adapter_ready"])
        self.assertFalse(missing["legacy_scalar_cost_is_sufficient"])
        self.assertIn("fee_cost_bp", missing["missing_terminal_path_fields"])
        self.assertIn("common_decision_clock_id", missing["missing_action_fields"])

        action = _action("adapter", "20260101")
        outcome = _outcome(action)
        transition = _valid_transitions()[0]
        complete = audit_current_fact_adapter_contract(
            action_columns=action.keys(),
            terminal_path_columns=outcome.keys(),
            transition_columns=transition.keys(),
        ).row(0, named=True)
        self.assertTrue(complete["fixed_policy_adapter_ready"])
        self.assertTrue(complete["intraday_transition_adapter_ready"])
        self.assertFalse(complete["intraday_controller_implemented"])


class AtomicPublisherTest(unittest.TestCase):
    def test_tiny_publish_is_atomic_and_refuses_existing_destination(self) -> None:
        actions = [
            _action("p1", "20260101", alternative_set="alt-1"),
            _action("p2", "20260102", alternative_set="alt-2"),
        ]
        outcomes = [_outcome(action, gross=10.0, cost=1.0) for action in actions]
        config = _small_config()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            action_path = root / "actions.parquet"
            outcome_path = root / "outcomes.parquet"
            sessions_path = root / "sessions.txt"
            destination = root / "published"
            pl.DataFrame(actions).write_parquet(action_path)
            pl.DataFrame(outcomes).write_parquet(outcome_path)
            sessions_path.write_text(
                "20260101\n20260102\n20260103\n20260104\n",
                encoding="utf-8",
            )
            result = run_finite_horizon_policy_publish(
                historical_actions_path=action_path,
                terminal_paths_path=outcome_path,
                sessions_path=sessions_path,
                output_root=destination,
                config=config,
                asof_dates=["20260104"],
            )
            self.assertEqual(result, destination.resolve())
            self.assertTrue((destination / "complete.json").is_file())
            self.assertTrue((destination / "prequential_ev_lookup.parquet").is_file())
            self.assertTrue((destination / "fact_adapter_readiness.parquet").is_file())
            marker = json.loads(
                (destination / "complete.json").read_text(encoding="utf-8")
            )
            self.assertEqual(marker["status"], "complete")
            self.assertEqual(marker["artifact_count"], 4)
            with self.assertRaises(FileExistsError):
                run_finite_horizon_policy_publish(
                    historical_actions_path=action_path,
                    terminal_paths_path=outcome_path,
                    sessions_path=sessions_path,
                    output_root=destination,
                    config=config,
                    asof_dates=["20260104"],
                )


if __name__ == "__main__":
    unittest.main()
