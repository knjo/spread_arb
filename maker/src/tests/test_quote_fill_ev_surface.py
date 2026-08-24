"""Tests for causal pathwise EV lookup and legal-tick action surfaces."""

from __future__ import annotations

import unittest

import polars as pl

from maker.src.quote_fill.ev_surface import (
    DEFAULT_COST_COLUMNS,
    EVLookupConfig,
    build_daily_ev_lookup,
    build_legal_action_surface,
    enumerate_legal_tick_actions,
    score_action_surface,
)


def _candidate(alias: str, threshold: float, **changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "Date": "20260103",
        "ValueCode": "2317",
        "QuoteCode": "DHFA6",
        "route": "future_ask_spot_taker",
        "decision_id": "decision-1",
        "decision_sequence": 1,
        "spread_pair_epoch": 7,
        "action_alias": alias,
        "threshold_basis_bp": threshold,
        "spot_bid": 99.5,
        "spot_ask": 100.0,
        "fut_bid": 100.0,
        "fut_ask": 100.5,
        "fut_exec_bid": 100.0,
        "fut_exec_ask": 100.5,
        "current_state_eligible": True,
        "source_asof_date": "20260102",
        "contains_target_day_outcome": False,
        "parameter_version": "test-boundary-v1",
        "lifecycle_policy_version": "layered-v1",
        "queue_scenario": "base_queue",
        "intended_maker_quantity": 1,
        "hedge_quantity": 1,
    }
    row.update(changes)
    return row


def _path(
    path_id: str,
    date: str,
    branch: str | None,
    gross: float | None,
    **changes: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "path_id": path_id,
        "Date": date,
        "label_end_date": date,
        "ValueCode": "2317",
        "route": "future_ask_spot_taker",
        "target_offset_ticks": 0,
        "state_family": "all",
        "state_bucket": "all",
        "lifecycle_policy_version": "layered-v1",
        "queue_scenario": "base_queue",
        "intended_maker_quantity": 1,
        "hedge_quantity": 1,
        "outcome_status": "known",
        "terminal_branch": branch,
        "cancel_status": "not_cancelled",
        "entry_fill_status": "full",
        "topup_status": "not_applicable",
        "hedge_50ms_status": "executable",
        "same_day_exit_status": "target_exit",
        "overnight_status": "not_applicable",
        "expiry_status": "not_applicable",
        "emergency_status": "not_triggered",
        "filled_cashflow_before_cost_bp": gross,
        "hedge_slippage_bp_50ms": None,
        "exit_slippage_bp": None,
        "capital_time_seconds": 100.0,
        "cost_profile_version": "cost-v1",
        **{column: 0.0 for column in DEFAULT_COST_COLUMNS},
    }
    row.update(changes)
    return row


class LegalActionSurfaceTest(unittest.TestCase):
    def test_rounded_aliases_share_one_absolute_price_action(self) -> None:
        candidates = pl.DataFrame(
            [
                _candidate("q50", 5.0),
                _candidate("q80", 20.0),
                _candidate("q95", 30.0, current_state_eligible=False),
            ]
        )
        result = build_legal_action_surface(candidates)

        self.assertEqual(result.actions.height, 1)
        action = result.actions.row(0, named=True)
        self.assertEqual(action["absolute_target_price"], 100.5)
        self.assertEqual(action["target_offset_ticks"], 0)
        self.assertEqual(action["action_alias_count"], 2)
        self.assertEqual(action["action_aliases"], ["q50", "q80"])
        self.assertEqual(result.decision_actions.height, 1)

        aliases = {
            row["action_alias"]: row
            for row in result.aliases.iter_rows(named=True)
        }
        self.assertEqual(aliases["q50"]["action_id"], aliases["q80"]["action_id"])
        self.assertIsNone(aliases["q95"]["action_id"])
        self.assertEqual(
            aliases["q95"]["action_status"], "current_state_ineligible"
        )

    def test_candidate_snapshot_must_be_strictly_prior(self) -> None:
        candidate = _candidate("q50", 5.0, source_asof_date="20260103")
        with self.assertRaisesRegex(ValueError, "strictly before"):
            build_legal_action_surface(pl.DataFrame([candidate]))

    def test_same_epoch_price_is_one_action_across_decision_aliases(self) -> None:
        rows = [
            _candidate("q50", 5.0),
            _candidate(
                "q50",
                5.0,
                decision_id="decision-2",
                decision_sequence=2,
            ),
        ]
        result = build_legal_action_surface(pl.DataFrame(rows))
        self.assertEqual(result.actions.height, 1)
        action = result.actions.row(0, named=True)
        self.assertEqual(action["submit_decision_id"], "decision-1")
        self.assertEqual(action["decision_alias_count"], 2)
        self.assertEqual(result.aliases["action_id"].n_unique(), 1)
        self.assertEqual(result.decision_actions.height, 2)

    def test_explicit_tick_range_enumerates_and_epoch_dedups(self) -> None:
        first = _candidate("unused", 5.0)
        first.update(
            min_absolute_target_tick=2300,
            max_absolute_target_tick=2302,
        )
        second = dict(first)
        second.update(decision_id="decision-2", decision_sequence=2)
        columns = {
            key: value
            for key, value in first.items()
            if key not in {"action_alias", "threshold_basis_bp"}
        }
        columns_2 = {
            key: value
            for key, value in second.items()
            if key not in {"action_alias", "threshold_basis_bp"}
        }
        surface = enumerate_legal_tick_actions(pl.DataFrame([columns, columns_2]))
        actions = surface.actions
        # Tick 2300 (100.0) crosses the future bid; 2301/2302 are passive.
        self.assertEqual(actions["absolute_target_tick"].to_list(), [2301, 2302])
        self.assertEqual(actions["decision_alias_count"].to_list(), [2, 2])
        self.assertEqual(actions["submit_decision_id"].to_list(), ["decision-1"] * 2)
        self.assertEqual(surface.decision_actions.height, 4)


class DailyEVLookupTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sessions = ["20260101", "20260102", "20260103", "20260104"]
        self.config = EVLookupConfig(
            lookback_sessions=3,
            min_history_sessions=3,
            min_group_sessions=2,
            min_known_paths=4,
            min_outcome_label_coverage=1.0,
            max_censor_rate=0.0,
            max_unknown_rate=0.0,
            downside_penalty_weight=0.5,
            emergency_penalty_bp_per_event=4.0,
            estimator_version="test-pathwise",
        )

    def _complete_paths(self) -> list[dict[str, object]]:
        return [
            _path(
                "cancel",
                "20260101",
                "no_fill_cancel",
                0.0,
                cancel_status="cancelled",
                entry_fill_status="no_fill",
                hedge_50ms_status="not_applicable",
                same_day_exit_status="not_applicable",
                cancel_cost_bp=0.5,
                capital_time_seconds=10.0,
            ),
            _path(
                "same-day",
                "20260101",
                "same_day_target_exit",
                10.0,
                fee_cost_bp=1.0,
                tax_cost_bp=1.0,
                hedge_slippage_bp_50ms=1.0,
                exit_slippage_bp=1.0,
                capital_time_seconds=100.0,
            ),
            _path(
                "overnight",
                "20260102",
                "overnight_exit",
                12.0,
                same_day_exit_status="no_exit",
                overnight_status="closed",
                fee_cost_bp=1.0,
                tax_cost_bp=1.0,
                hedge_slippage_bp_50ms=1.0,
                exit_slippage_bp=1.0,
                financing_cost_bp=1.0,
                overnight_cost_bp=1.0,
                capital_time_seconds=3600.0,
            ),
            _path(
                "emergency",
                "20260102",
                "emergency_exit",
                -3.0,
                hedge_50ms_status="failed",
                same_day_exit_status="failed",
                emergency_status="triggered",
                fee_cost_bp=1.0,
                emergency_cost_bp=2.0,
                exit_slippage_bp=2.0,
                capital_time_seconds=20.0,
            ),
        ]

    def test_ev_is_reconstructed_from_exclusive_complete_paths(self) -> None:
        rows = self._complete_paths()
        # Neither a target-day path nor a label that matures on D may leak in.
        rows.extend(
            [
                _path("target-day", "20260104", "emergency_exit", -999.0),
                _path(
                    "immature",
                    "20260103",
                    "emergency_exit",
                    -999.0,
                    label_end_date="20260104",
                ),
            ]
        )
        lookup = build_daily_ev_lookup(
            pl.DataFrame(rows),
            self.sessions,
            self.config,
            asof_dates=["20260104"],
        )
        row = lookup.row(0, named=True)

        self.assertTrue(row["execution_safe_snapshot"])
        self.assertFalse(row["contains_target_day_outcome"])
        self.assertEqual(row["label_cutoff_date"], "20260102")
        self.assertEqual(row["n_paths"], 4)
        self.assertEqual(row["n_outcome_known"], 4)
        self.assertAlmostEqual(row["branch_probability_sum"], 1.0)
        self.assertEqual(row["p_branch_no_fill_cancel"], 0.25)
        self.assertEqual(row["p_branch_same_day_target_exit"], 0.25)
        self.assertEqual(row["p_branch_overnight_exit"], 0.25)
        self.assertEqual(row["p_branch_emergency_exit"], 0.25)
        # Actual-fill path nets are -0.5, 8, 8, -6 bp.  Slippage diagnostics
        # are already embodied by fill prices and are not subtracted twice.
        self.assertAlmostEqual(row["expected_net_cashflow_bp"], 2.375)
        self.assertAlmostEqual(row["branch_reconstructed_ev_bp"], 2.375)
        self.assertAlmostEqual(row["expected_downside_bp"], 1.625)
        self.assertAlmostEqual(row["downside_penalty_bp"], 0.8125)
        self.assertAlmostEqual(row["emergency_risk_penalty_bp"], 1.0)
        self.assertAlmostEqual(row["action_score_bp"], 0.5625)
        self.assertEqual(row["cancel_rate_given_observed"], 0.25)
        self.assertEqual(row["n_full_fill"], 3)
        self.assertEqual(row["n_hedge_50ms_executable"], 2)
        self.assertEqual(row["n_hedge_slippage_observed"], 2)
        self.assertEqual(row["n_same_day_target_exit"], 1)
        self.assertEqual(row["n_overnight_paths"], 1)
        self.assertEqual(row["n_emergency_triggered"], 1)
        self.assertTrue(row["ev_ready"])
        self.assertEqual(row["ev_status"], "ready")

        changed = [dict(item) for item in rows]
        changed[-2]["filled_cashflow_before_cost_bp"] = 999_999.0
        changed_lookup = build_daily_ev_lookup(
            pl.DataFrame(changed),
            self.sessions,
            self.config,
            asof_dates=["20260104"],
        )
        self.assertEqual(
            row["expected_net_cashflow_bp"],
            changed_lookup.row(0, named=True)["expected_net_cashflow_bp"],
        )

    def test_censor_unknown_and_missing_cost_are_not_zero(self) -> None:
        known = _path("known", "20260101", "same_day_target_exit", 10.0)
        known["tax_cost_bp"] = None
        censored = _path(
            "censored",
            "20260101",
            None,
            None,
            outcome_status="censored",
            cancel_status="unknown",
            entry_fill_status="partial",
            topup_status="unknown",
            hedge_50ms_status="unknown",
            same_day_exit_status="unknown",
            overnight_status="unknown",
            emergency_status="unknown",
            capital_time_seconds=None,
            cost_profile_version=None,
        )
        unknown = dict(censored)
        unknown.update(path_id="unknown", outcome_status="unknown")
        for row in (censored, unknown):
            for column in DEFAULT_COST_COLUMNS:
                row[column] = None
        config = EVLookupConfig(
            lookback_sessions=2,
            min_history_sessions=1,
            min_group_sessions=1,
            min_known_paths=1,
            min_outcome_label_coverage=0.0,
            max_censor_rate=1.0,
            max_unknown_rate=1.0,
            estimator_version="missing-input-test",
        )
        lookup = build_daily_ev_lookup(
            pl.DataFrame([known, censored, unknown]),
            ["20260101", "20260102"],
            config,
            asof_dates=["20260102"],
        )
        row = lookup.row(0, named=True)
        self.assertEqual(row["n_outcome_known"], 1)
        self.assertEqual(row["n_outcome_censored"], 1)
        self.assertEqual(row["n_outcome_unknown"], 1)
        self.assertEqual(row["n_cost_complete_known"], 0)
        self.assertIsNone(row["expected_net_cashflow_bp"])
        self.assertFalse(row["ev_ready"])
        self.assertEqual(row["ev_status"], "unpriced_censored_or_unknown")

    def test_minimum_history_sessions_is_a_hard_gate(self) -> None:
        path = _path("only", "20260101", "same_day_target_exit", 1.0)
        config = EVLookupConfig(
            lookback_sessions=3,
            min_history_sessions=2,
            min_group_sessions=1,
            min_known_paths=1,
        )
        lookup = build_daily_ev_lookup(
            pl.DataFrame([path]),
            ["20260101", "20260102"],
            config,
            asof_dates=["20260102"],
        )
        row = lookup.row(0, named=True)
        self.assertFalse(row["history_support_gate"])
        self.assertFalse(row["ev_ready"])
        self.assertEqual(row["ev_status"], "insufficient_history_sessions")

    def test_executable_hedge_requires_slippage_diagnostic(self) -> None:
        path = _path("missing-hedge", "20260101", "same_day_target_exit", 2.0)
        path["exit_slippage_bp"] = 0.5
        lookup = build_daily_ev_lookup(
            pl.DataFrame([path]),
            ["20260101", "20260102"],
            EVLookupConfig(
                lookback_sessions=1,
                min_history_sessions=1,
                min_group_sessions=1,
                min_known_paths=1,
            ),
            asof_dates=["20260102"],
        )
        row = lookup.row(0, named=True)
        self.assertEqual(row["n_slippage_complete_known"], 0)
        self.assertIsNone(row["expected_net_cashflow_bp"])
        self.assertFalse(row["ev_ready"])
        self.assertEqual(row["ev_status"], "incomplete_slippage_diagnostics")

    def test_unknown_mass_remains_in_admitted_quote_denominator(self) -> None:
        known = _path(
            "known-cancel",
            "20260101",
            "no_fill_cancel",
            0.0,
            cancel_status="cancelled",
            entry_fill_status="no_fill",
            hedge_50ms_status="not_applicable",
            same_day_exit_status="not_applicable",
            cancel_cost_bp=0.5,
        )
        unresolved = _path(
            "unknown-terminal",
            "20260101",
            None,
            None,
            outcome_status="unknown",
            cancel_status="unknown",
            entry_fill_status="unknown",
            topup_status="unknown",
            hedge_50ms_status="unknown",
            same_day_exit_status="unknown",
            overnight_status="unknown",
            expiry_status="unknown",
            emergency_status="unknown",
            capital_time_seconds=None,
            cost_profile_version=None,
        )
        for column in DEFAULT_COST_COLUMNS:
            unresolved[column] = None
        lookup = build_daily_ev_lookup(
            pl.DataFrame([known, unresolved]),
            ["20260101", "20260102"],
            EVLookupConfig(
                lookback_sessions=1,
                min_history_sessions=1,
                min_group_sessions=1,
                min_known_paths=1,
                min_outcome_label_coverage=0.0,
                max_unknown_rate=1.0,
            ),
            asof_dates=["20260102"],
        )
        row = lookup.row(0, named=True)
        self.assertEqual(row["p_branch_no_fill_cancel"], 0.5)
        self.assertEqual(row["p_outcome_unknown"], 0.5)
        self.assertEqual(row["unpriced_terminal_mass"], 0.5)
        self.assertEqual(row["admitted_path_probability_sum"], 1.0)
        self.assertAlmostEqual(
            row["known_cashflow_contribution_per_admitted_quote_bp"], -0.25
        )
        self.assertIsNone(row["expected_net_cashflow_bp"])
        self.assertFalse(row["ev_ready"])

    def test_expiry_fill_is_not_mislabeled_as_overnight(self) -> None:
        expiry = _path(
            "expiry",
            "20260101",
            "expiry_forced_flat",
            3.0,
            same_day_exit_status="no_exit",
            overnight_status="not_applicable",
            expiry_status="forced_flat",
            hedge_slippage_bp_50ms=0.2,
            exit_slippage_bp=0.5,
        )
        lookup = build_daily_ev_lookup(
            pl.DataFrame([expiry]),
            ["20260101", "20260102"],
            EVLookupConfig(
                lookback_sessions=1,
                min_history_sessions=1,
                min_group_sessions=1,
                min_known_paths=1,
            ),
            asof_dates=["20260102"],
        )
        row = lookup.row(0, named=True)
        self.assertEqual(row["n_expiry_forced_flat"], 1)
        self.assertEqual(row["n_overnight_paths"], 0)
        self.assertEqual(row["p_branch_expiry_forced_flat"], 1.0)
        self.assertTrue(row["ev_ready"])

    def test_default_support_cannot_mark_eight_day_pilot_ready(self) -> None:
        sessions = [f"202601{day:02d}" for day in range(1, 10)]
        paths = [
            _path(
                f"pilot-{day}",
                f"202601{day:02d}",
                "no_fill_expire",
                0.0,
                entry_fill_status="no_fill",
                hedge_50ms_status="not_applicable",
                same_day_exit_status="not_applicable",
            )
            for day in range(1, 9)
        ]
        lookup = build_daily_ev_lookup(
            pl.DataFrame(paths),
            sessions,
            asof_dates=["20260109"],
        )
        row = lookup.row(0, named=True)
        self.assertEqual(row["window_sessions"], 8)
        self.assertFalse(row["history_support_gate"])
        self.assertFalse(row["ev_ready"])
        self.assertEqual(row["ev_status"], "insufficient_history_sessions")

    def test_nonterminal_path_cannot_claim_a_branch(self) -> None:
        bad = _path(
            "bad",
            "20260101",
            "same_day_target_exit",
            None,
            outcome_status="censored",
        )
        with self.assertRaisesRegex(ValueError, "must not claim"):
            build_daily_ev_lookup(
                pl.DataFrame([bad]),
                ["20260101", "20260102"],
                EVLookupConfig(
                    lookback_sessions=1,
                    min_history_sessions=1,
                    min_group_sessions=1,
                    min_known_paths=1,
                ),
                asof_dates=["20260102"],
            )


class ScoreActionSurfaceTest(unittest.TestCase):
    @staticmethod
    def _lookup(offset: int, score: float) -> dict[str, object]:
        return {
            "asof_date": "20260103",
            "ValueCode": "2317",
            "route": "future_ask_spot_taker",
            "target_offset_ticks": offset,
            "state_family": "all",
            "state_bucket": "all",
            "lifecycle_policy_version": "layered-v1",
            "queue_scenario": "base_queue",
            "intended_maker_quantity": 1,
            "hedge_quantity": 1,
            "ev_ready": True,
            "ev_status": "ready",
            "expected_net_cashflow_bp": score,
            "action_score_bp": score,
            "execution_safe_snapshot": True,
            "contains_target_day_outcome": False,
        }

    def test_only_best_ready_positive_action_is_selected(self) -> None:
        candidates = pl.DataFrame(
            [_candidate("near", 5.0), _candidate("far", 60.0)]
        )
        actions = build_legal_action_surface(candidates).decision_actions
        lookup = pl.DataFrame([self._lookup(0, 1.0), self._lookup(1, 2.0)])
        scored = score_action_surface(actions, lookup)
        selected = scored.filter(pl.col("quote_selected")).row(0, named=True)
        self.assertEqual(scored.height, 2)
        self.assertEqual(selected["target_offset_ticks"], 1)
        self.assertEqual(selected["action_ev_rank"], 1)

    def test_scoring_is_per_causal_decision_not_whole_epoch(self) -> None:
        candidates = pl.DataFrame(
            [
                _candidate("near", 5.0),
                _candidate(
                    "near",
                    5.0,
                    decision_id="decision-2",
                    decision_sequence=2,
                ),
                _candidate(
                    "far",
                    60.0,
                    decision_id="decision-2",
                    decision_sequence=2,
                ),
            ]
        )
        surface = build_legal_action_surface(candidates)
        self.assertEqual(surface.actions.height, 2)
        self.assertEqual(surface.decision_actions.height, 3)
        scored = score_action_surface(
            surface.decision_actions,
            pl.DataFrame([self._lookup(0, 1.0), self._lookup(1, 2.0)]),
        )
        selected = scored.filter(pl.col("quote_selected")).sort(
            "decision_sequence"
        )
        self.assertEqual(selected.height, 2)
        self.assertEqual(selected["target_offset_ticks"].to_list(), [0, 1])


if __name__ == "__main__":
    unittest.main()
