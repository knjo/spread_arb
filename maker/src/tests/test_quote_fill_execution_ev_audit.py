"""Tests for the strict execution-root to EV-audit bridge."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import polars as pl

from maker.src.quote_fill.ev_surface import EVLookupConfig
from maker.src.quote_fill.execution_ev_audit import (
    UNASSIGNED_EXIT_RULE_ID,
    build_execution_ev_audit,
    run_execution_ev_audit,
)
from maker.src.quote_fill.execution_report import PartitionedExecutionInputs
from maker.src.quote_fill.terminal_paths import (
    EXECUTION_PATH_LOOKUP_KEYS_V1,
    TerminalPathAdapterConfig,
)


PRIOR_DATE = "20260101"
ASOF_DATE = "20260102"
ROUTE = "future_ask_spot_taker"


def _action(policy_id: str, date: str, kind: str) -> dict[str, object]:
    full = kind in {"same_day", "carry"}
    return {
        "Date": date,
        "ValueCode": "2317",
        "QuoteCode": "DHFA6",
        "route": ROUTE,
        "parameter_version": "boundary-v1",
        "lookup_action_id": "q80",
        "raw_order_fact_id": f"raw-{policy_id}",
        "policy_generation_id": policy_id,
        "rank_bucket": "at_bbo",
        "queue_bucket": "00_0to1",
        "tod_bucket": "mid_0930_1200",
        "freshness_bucket": "fresh_le100ms",
        "intended_quantity": 1,
        "submit_recv_time_ns": 1_000_000_000,
        "terminal_recv_time_ns": 2_000_000_000,
        "nominal_stop_reason": "target_retreat",
        "queue_known": True,
        "any_fill": full,
        "full_fill": full,
        "partial_fill": False,
        "cancel_required": not full,
        "entry_hedge_status": "executable" if full else None,
        "entry_hedge_signed_total_slippage_bp": 1.25 if full else None,
        "entry_hedge_executed_quantity": 2 if full else None,
        "entry_hedge_contract_size_shares": 2_000 if full else None,
        "entry_spot_price": 100.0 if full else None,
        "joint_volume_allocated": False,
    }


def _exit(policy_id: str, date: str, kind: str) -> dict[str, object]:
    branch = {
        "same_day": "same_day_taker_exit",
        "carry": "carry_at_eod",
        "no_fill": "no_entry_fill",
    }[kind]
    return {
        "Date": date,
        "ValueCode": "2317",
        "QuoteCode": "DHFA6",
        "route": ROUTE,
        "raw_order_fact_id": f"raw-{policy_id}",
        "policy_generation_id": policy_id,
        "exit_rule_id": "frozen_center",
        "exit_rule_source_asof_date": "20251231",
        "branch_status": branch,
        "terminal_outcome": branch in {"same_day_taker_exit", "no_entry_fill"},
        "needs_next_session_label": branch == "carry_at_eod",
        "exit_decision_time_ns": (
            5_000_000_000 if branch == "same_day_taker_exit" else None
        ),
        "gross_cycle_pnl_twd": (
            200.0 if branch == "same_day_taker_exit" else None
        ),
    }


def _inputs() -> PartitionedExecutionInputs:
    specs = [
        ("prior-known", PRIOR_DATE, "same_day"),
        ("prior-unknown", PRIOR_DATE, "no_fill"),
        ("prior-carry", PRIOR_DATE, "carry"),
        ("target-known", ASOF_DATE, "same_day"),
    ]
    action = pl.from_dicts(
        [_action(policy, date, kind) for policy, date, kind in specs],
        infer_schema_length=None,
    )
    exits = pl.from_dicts(
        [_exit(policy, date, kind) for policy, date, kind in specs],
        infer_schema_length=None,
    )
    coverage = pl.DataFrame(
        {
            "Date": [PRIOR_DATE, PRIOR_DATE, ASOF_DATE, ASOF_DATE],
            "ValueCode": ["2317", "2603", "2317", "2603"],
            "partition_complete": [True, False, True, True],
            "partition": ["p1", None, "p2", "p3"],
        }
    )
    return PartitionedExecutionInputs(
        action_facts=action,
        raw_order_facts=pl.DataFrame(),
        hedge_facts=pl.DataFrame(),
        daily_support=pl.DataFrame(),
        exit_facts=exits,
        coverage=coverage,
        metadata={
            "execution_root": "/tmp/execution",
            "selected_dates": [PRIOR_DATE, ASOF_DATE],
            "selected_session_count": 2,
            "selected_product_count": 2,
            "selected_product_day_count": 3,
            "balanced_product_day_grid": False,
            "partition_hashes_validated": True,
        },
    )


def _lookup_config() -> EVLookupConfig:
    return EVLookupConfig(
        lookback_sessions=2,
        min_history_sessions=1,
        min_group_sessions=1,
        min_known_paths=1,
        min_outcome_label_coverage=0.0,
        max_censor_rate=1.0,
        max_unknown_rate=1.0,
        lookup_keys=EXECUTION_PATH_LOOKUP_KEYS_V1,
    )


class ExecutionEVAuditTest(unittest.TestCase):
    def test_last_date_is_visible_in_paths_but_excluded_from_lookup(self) -> None:
        audit = build_execution_ev_audit(
            _inputs(),
            state_collapse="all",
            lookup_config=_lookup_config(),
        )
        paths = audit.terminal_paths
        self.assertEqual(paths.height, 4)
        self.assertEqual(paths["state_family"].unique().to_list(), ["all"])
        self.assertEqual(paths["state_bucket"].unique().to_list(), ["all"])

        target = paths.filter(pl.col("Date") == ASOF_DATE).row(0, named=True)
        self.assertTrue(target["contains_asof_target_day_outcome"])
        self.assertFalse(target["included_in_asof_lookup"])
        self.assertEqual(
            target["asof_lookup_role"], "excluded_asof_target_day_outcome"
        )
        self.assertEqual(
            paths.filter(pl.col("included_in_asof_lookup")).height, 3
        )

        statuses = paths.group_by("outcome_status").len()
        counts = dict(
            zip(statuses["outcome_status"], statuses["len"], strict=True)
        )
        self.assertEqual(counts, {"known": 2, "unknown": 1, "censored": 1})
        unresolved = paths.filter(pl.col("outcome_status") != "known")
        self.assertTrue(
            unresolved["filled_cashflow_before_cost_bp"].is_null().all()
        )
        self.assertTrue(paths["fee_cost_bp"].is_null().all())

        lookup = audit.ev_lookup.row(0, named=True)
        self.assertEqual(lookup["asof_date"], ASOF_DATE)
        self.assertEqual(lookup["train_end_date"], PRIOR_DATE)
        self.assertEqual(lookup["n_paths"], 3)
        self.assertEqual(lookup["n_outcome_known"], 1)
        self.assertEqual(lookup["n_outcome_unknown"], 1)
        self.assertEqual(lookup["n_outcome_censored"], 1)
        self.assertTrue(lookup["execution_safe_snapshot"])
        self.assertFalse(lookup["contains_target_day_outcome"])
        self.assertFalse(lookup["ev_ready"])
        self.assertEqual(lookup["ev_status"], "unpriced_censored_or_unknown")
        self.assertIsNone(lookup["expected_net_cashflow_bp"])

    def test_runner_allows_incomplete_grid_and_writes_reproducible_artifacts(
        self,
    ) -> None:
        inputs = _inputs()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "audit"
            with patch(
                "maker.src.quote_fill.execution_ev_audit."
                "load_partitioned_execution_inputs",
                return_value=inputs,
            ) as loader:
                audit = run_execution_ev_audit(
                    root / "execution",
                    output_dir=output,
                    sessions=2,
                    allow_incomplete_product_grid=True,
                    state_collapse="all",
                    lookup_config=_lookup_config(),
                )
            self.assertFalse(
                loader.call_args.kwargs["require_balanced_product_days"]
            )
            self.assertFalse(audit.coverage["partition_complete"].all())

            expected = {
                "strict_terminal_paths.parquet",
                "asof_ev_lookup.parquet",
                "terminal_path_summary.csv",
                "ev_lookup_status_summary.csv",
                "coverage.csv",
                "audit_config.json",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            payload = json.loads((output / "audit_config.json").read_text())
            self.assertTrue(payload["complete"])
            self.assertEqual(payload["asof_date"], ASOF_DATE)
            self.assertEqual(payload["state_collapse"], "all")
            self.assertFalse(payload["asof_target_day_outcomes_in_lookup"])
            self.assertFalse(payload["unknown_imputed_as_zero"])
            self.assertFalse(payload["censored_imputed_as_zero"])
            self.assertFalse(payload["null_costs_imputed_as_zero"])
            self.assertTrue(
                payload["loader_options"]["allow_incomplete_product_grid"]
            )
            self.assertEqual(payload["ev_ready_cells"], 0)
            self.assertEqual(payload["priced_expected_ev_cells"], 0)
            self.assertEqual(len(payload["artifacts"]), 5)

            persisted = pl.read_parquet(output / "strict_terminal_paths.parquet")
            self.assertTrue(
                persisted.filter(pl.col("outcome_status") != "known")[
                    "filled_cashflow_before_cost_bp"
                ]
                .is_null()
                .all()
            )
            first_config = (output / "audit_config.json").read_bytes()
            with patch(
                "maker.src.quote_fill.execution_ev_audit."
                "load_partitioned_execution_inputs",
                return_value=inputs,
            ):
                run_execution_ev_audit(
                    root / "execution",
                    output_dir=output,
                    sessions=2,
                    allow_incomplete_product_grid=True,
                    state_collapse="all",
                    lookup_config=_lookup_config(),
                )
            self.assertEqual(
                first_config, (output / "audit_config.json").read_bytes()
            )

    def test_strict_runner_rejects_cancel_completion_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not permit"):
            build_execution_ev_audit(
                _inputs(),
                adapter_config=TerminalPathAdapterConfig(
                    lifecycle_policy_version="layered-v1",
                    queue_scenario="base-queue",
                    nominal_cancel_model_version="instant-cancel-v1",
                ),
                lookup_config=_lookup_config(),
            )

    def test_missing_exit_fact_is_retained_as_one_unknown_sentinel_path(self) -> None:
        inputs = _inputs()
        incomplete = replace(
            inputs,
            exit_facts=inputs.exit_facts.filter(
                pl.col("policy_generation_id") != "prior-unknown"
            ),
        )
        audit = build_execution_ev_audit(
            incomplete,
            state_collapse="all",
            lookup_config=_lookup_config(),
        )
        sentinel = audit.terminal_paths.filter(
            pl.col("exit_rule_id") == UNASSIGNED_EXIT_RULE_ID
        )
        self.assertEqual(sentinel.height, 1)
        row = sentinel.row(0, named=True)
        self.assertEqual(row["outcome_status"], "unknown")
        self.assertIsNone(row["terminal_branch"])
        self.assertFalse(row["exit_rule_fact_observed"])
        self.assertIsNone(row["filled_cashflow_before_cost_bp"])
        self.assertEqual(
            row["terminal_mapping_reason"],
            "exit_rule_fact_missing_or_policy_unassigned",
        )
        self.assertEqual(audit.metadata["unassigned_exit_rule_path_rows"], 1)
        self.assertEqual(audit.terminal_paths.height, 4)


if __name__ == "__main__":
    unittest.main()
