"""Tests for the selected-q95 D-safe universe audit."""

from __future__ import annotations

import unittest

import polars as pl

from maker.src.quote_fill.d_safe_universe_audit import (
    Q95_GATE_METHOD,
    build_d_safe_universe_audit,
)


def _paths() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Date": ["20260103", "20260103"],
            "ValueCode": ["1101", "1101"],
            "QuoteCode": ["TX1101", "TX1101"],
            "entry_route": [
                "spot_bid_future_taker",
                "future_ask_spot_taker",
            ],
            "boundary_quantile": [95, 95],
            "policy_path_id": ["p1", "p2"],
            "position_established_ns": [1, 2],
            "normalization_notional_twd": [1_000_000.0, 1_000_000.0],
            "filled_entry_outcome_category": ["completed", "unknown"],
            "terminal_date": ["20260103", None],
            "terminal_cashflow_priced": [True, False],
        }
    )


def _screen(*, stale_spot_route: bool = False, disagree: bool = False) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    statuses = {
        "spot_bid_future_taker": "pass",
        "future_ask_spot_taker": "known_fail",
    }
    for value_code in ("1101", "9999"):
        for route in ("spot_bid_future_taker", "future_ask_spot_taker"):
            for quantile in (50, 80):
                status = "pass" if value_code == "9999" else statuses[route]
                if disagree and value_code == "1101" and route == "spot_bid_future_taker" and quantile == 80:
                    status = "known_fail"
                source = (
                    "20260103"
                    if stale_spot_route
                    and value_code == "1101"
                    and route == "spot_bid_future_taker"
                    else "20260102"
                )
                rows.append(
                    {
                        "Date": "20260103",
                        "ValueCode": value_code,
                        "QuoteCode": f"TX{value_code}",
                        "route": route,
                        "boundary_quantile": quantile,
                        "source_asof_date": source,
                        "liquidity_gate_status": status,
                        "pre_replay_candidate": status == "pass",
                        "replay_tier": (
                            "core_candidate" if status == "pass" else "known_fail"
                        ),
                        "execution_safe_snapshot": True,
                        "contains_target_day_outcome": False,
                    }
                )
    return pl.from_dicts(rows, infer_schema_length=None)


def _manifest() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ValueCode": ["1101", "2412", "9999"],
            "execution_cli_member": [True, True, False],
            "retrospective_research_selection": [True, True, True],
            "selection_contains_target_day_outcomes": [True, True, True],
            "production_universe_approved": [False, False, False],
            "runtime_daily_liquidity_gate_required": [True, True, True],
        }
    )


class DSafeUniverseAuditTest(unittest.TestCase):
    def test_causal_consensus_gate_and_exit_only_semantics(self) -> None:
        result = build_d_safe_universe_audit(
            _paths(), _screen(), _manifest()
        )
        paths = result.path_gate_audit.sort("policy_path_id")
        self.assertEqual(paths["new_entry_gate_pass"].to_list(), [True, False])
        self.assertEqual(
            paths["new_entry_gate_reason"].to_list(),
            ["pass", "liquidity_known_fail"],
        )
        blocked = paths.row(1, named=True)
        self.assertEqual(
            blocked["blocked_product_if_already_held_action"],
            "exit_only_continue_existing_exit_policy",
        )
        self.assertFalse(blocked["forced_liquidation_due_to_gate"])
        product_day = result.product_day_gate_audit.row(0, named=True)
        self.assertEqual(product_day["product_day_gate_status"], "mixed_selected_routes")
        self.assertEqual(result.coverage_gap.height, 2)
        self.assertTrue(
            result.coverage_gap[
                "retrospective_universe_selection_leak_evidence"
            ].all()
        )
        summary = result.overall_summary.row(0, named=True)
        self.assertEqual(summary["liquidity_gate_method"], Q95_GATE_METHOD)
        self.assertFalse(summary["exact_q95_liquidity_row_available"])
        self.assertEqual(summary["raw_replay_cohort_product_count"], 2)
        self.assertEqual(summary["raw_replay_products_without_selected_paths"], "2412")

    def test_same_day_source_is_blocked_not_used(self) -> None:
        result = build_d_safe_universe_audit(
            _paths(), _screen(stale_spot_route=True), _manifest()
        )
        spot = result.path_gate_audit.filter(
            pl.col("entry_route") == "spot_bid_future_taker"
        ).row(0, named=True)
        self.assertFalse(spot["new_entry_gate_pass"])
        self.assertEqual(
            spot["new_entry_gate_reason"], "unsafe_or_noncausal_route_gate"
        )

    def test_q50_q80_disagreement_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "q50/q80 route gate rows disagree"):
            build_d_safe_universe_audit(
                _paths(), _screen(disagree=True), _manifest()
            )

    def test_selected_q_must_be_q95(self) -> None:
        paths = _paths().with_columns(pl.lit(80).alias("boundary_quantile"))
        with self.assertRaisesRegex(ValueError, "only configured q95"):
            build_d_safe_universe_audit(paths, _screen(), _manifest())


if __name__ == "__main__":
    unittest.main()
