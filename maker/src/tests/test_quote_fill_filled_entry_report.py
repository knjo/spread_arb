from __future__ import annotations

from contextlib import redirect_stderr
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

import polars as pl

from maker.src.quote_fill.exit_maker_report import (
    ExitMakerPartitionInputs,
    _file_sha256,
)
from maker.src.quote_fill.filled_entry_report import (
    OUTCOME_CENSORED,
    OUTCOME_COMPLETED,
    OUTCOME_UNKNOWN,
    REPORT_VERSION,
    _canonical_sha256 as _report_canonical_sha256,
    _parse_args,
    _publish_report,
    build_filled_entry_primary_report,
    load_cross_session_nominal_terminal_facts,
)
from maker.src.quote_fill.exit_maker_cross_session_runner import (
    CROSS_SESSION_PREREQUISITE_BINDING_VERSION,
    _RUNNER_AUDIT_SCHEMA,
    _canonical_sha256,
    _publish_partition as _publish_cross_partition,
    _rebuild_root_manifest as _rebuild_cross_root_manifest,
)
from maker.src.quote_fill.exit_maker_cross_session import (
    _attempt_schema,
    _outcome_schema,
)


FUTURE_ENTRY = "future_ask_spot_taker"
SPOT_ENTRY = "spot_bid_future_taker"
FUTURE_EXIT = "future_bid_spot_taker"
SPOT_EXIT = "spot_ask_future_taker"
VALUE = "2330"
QUOTE = "CDF1"
DATE = "20260102"


def _inputs() -> ExitMakerPartitionInputs:
    actions: list[dict[str, object]] = []
    positions: list[dict[str, object]] = []

    def add_action(
        *,
        policy: str,
        raw: str,
        route: str,
        q: int,
        full: bool | None = True,
        any_fill: bool | None = True,
        partial: bool | None = False,
        hedge: str | None = "executable",
        cancel: bool = False,
    ) -> None:
        actions.append(
            {
                "Date": DATE,
                "ValueCode": VALUE,
                "QuoteCode": QUOTE,
                "route": route,
                "boundary_quantile": q,
                "raw_order_fact_id": raw,
                "policy_generation_id": policy,
                "full_fill": full,
                "partial_fill": partial,
                "any_fill": any_fill,
                "cancel_required": cancel,
                "entry_hedge_status": hedge,
                "full_fill_recv_time_ns": (
                    950_000_000 if full is True else None
                ),
                "entry_hedge_decision_time_ns": (
                    1_000_000_000 if hedge is not None else None
                ),
                "entry_hedge_label_observed": full is True,
                "entry_hedge_executable": (
                    full is True and hedge == "executable"
                ),
                "entry_spot_price": 100.0,
                "entry_hedge_contract_size_shares": 2_000,
            }
        )

    # The future-entry q aliases share one physical raw fill.  The duplicate
    # q50 alias is also intentionally identical inside its policy cell.
    for q in (50, 80, 95):
        add_action(policy=f"future-q{q}", raw="shared-future-raw", route=FUTURE_ENTRY, q=q)
        add_action(policy=f"spot-q{q}", raw=f"spot-raw-{q}", route=SPOT_ENTRY, q=q)
    add_action(
        policy="future-q50-alias",
        raw="shared-future-raw",
        route=FUTURE_ENTRY,
        q=50,
    )
    # A shorter-lived policy alias can stop as a no-fill even though a
    # longer-lived alias sharing the canonical raw order later establishes the
    # physical hedge.  Raw hedge status/cursor remain populated; alias-local
    # booleans keep this row out of the primary denominator.
    add_action(
        policy="future-q80-shared-raw-no-fill",
        raw="shared-future-raw",
        route=FUTURE_ENTRY,
        q=80,
        full=False,
        any_fill=False,
        partial=False,
        hedge="executable",
        cancel=True,
    )


    add_action(
        policy="known-no-fill",
        raw="no-fill-raw",
        route=FUTURE_ENTRY,
        q=50,
        full=False,
        any_fill=False,
        partial=False,
        hedge=None,
        cancel=True,
    )
    add_action(
        policy="partial-entry",
        raw="partial-raw",
        route=SPOT_ENTRY,
        q=80,
        full=False,
        any_fill=True,
        partial=True,
        hedge=None,
        cancel=True,
    )
    add_action(
        policy="hedge-unavailable",
        raw="hedge-fail-raw",
        route=SPOT_ENTRY,
        q=95,
        full=True,
        any_fill=True,
        partial=False,
        hedge="insufficient_depth",
        cancel=False,
    )

    established = [
        row
        for row in actions
        if row["full_fill"] is True
        and row["entry_hedge_label_observed"] is True
        and row["entry_hedge_executable"] is True
    ]
    for action in established:
        for rule in ("frozen_center", "frozen_lower"):
            for exit_route in (FUTURE_EXIT, SPOT_EXIT):
                if rule == "frozen_center" and exit_route == FUTURE_EXIT:
                    branch = "flat_same_day"
                    gross = 1_000.0
                elif rule == "frozen_lower" and exit_route == SPOT_EXIT:
                    branch = "flat_same_day"
                    gross = 2_000.0
                elif rule == "frozen_center":
                    branch = "carry_at_eod_no_admission"
                    gross = None
                else:
                    branch = "fill_unknown_at_eod"
                    gross = None
                trial = f"{action['policy_generation_id']}/{rule}/{exit_route}"
                positions.append(
                    {
                        "Date": DATE,
                        "ValueCode": VALUE,
                        "QuoteCode": QUOTE,
                        "entry_route": action["route"],
                        "entry_policy_generation_id": action[
                            "policy_generation_id"
                        ],
                        "entry_raw_order_fact_id": action["raw_order_fact_id"],
                        "exit_rule_id": rule,
                        "exit_route": exit_route,
                        "exit_policy_trial_id": trial,
                        "exit_threshold_basis_bp": (
                            100.0 if rule == "frozen_center" else 80.0
                        ),
                        "exit_rule_source_asof_date": "20260101",
                        "position_status": "position_established",
                        "position_established_ns": 1_000_000_000,
                        "nominal_instant_cancel_v0_branch": branch,
                        "gross_cycle_pnl_twd": gross,
                    }
                )

    exit_rows = []
    for action in actions:
        for rule in ("frozen_center", "frozen_lower"):
            exit_rows.append(
                {
                    "Date": DATE,
                    "ValueCode": VALUE,
                    "QuoteCode": QUOTE,
                    "route": action["route"],
                    "raw_order_fact_id": action["raw_order_fact_id"],
                    "policy_generation_id": action["policy_generation_id"],
                    "exit_rule_id": rule,
                    "exit_threshold_basis_bp": (
                        100.0 if rule == "frozen_center" else 80.0
                    ),
                    "exit_rule_source_asof_date": "20260101",
                    "branch_status": "synthetic",
                    "terminal_outcome": False,
                    "exit_decision_time_ns": None,
                    "gross_cycle_pnl_twd": None,
                    "exit_added_latency_ns": 50_000_000,
                }
            )
    empty = pl.DataFrame()
    return ExitMakerPartitionInputs(
        policy_support=empty,
        candidate_aliases=empty,
        raw_candidate_facts=empty,
        position_policy_facts=pl.from_dicts(positions, infer_schema_length=None),
        action_facts=pl.from_dicts(actions, infer_schema_length=None),
        taker_exit_facts=pl.from_dicts(exit_rows, infer_schema_length=None),
        coverage=pl.DataFrame(
            {
                "Date": [DATE],
                "ValueCode": [VALUE],
                "partition_complete": [True],
                "partition": ["synthetic"],
            }
        ),
        metadata={
            "selected_session_count": 1,
            "hedge_delay_ns": 50_000_000,
            "entry_action_hedge_delay_ns": 50_000_000,
            "exit_maker_hedge_delay_ns": 50_000_000,
            "session_predecessors": {DATE: "20260101"},
        },
    )


def _synthetic_prerequisite_identity(root: Path) -> dict[str, object]:
    prerequisite_root = (root / "synthetic-prerequisite").resolve()
    prerequisite_root.mkdir(parents=True, exist_ok=True)
    candidate_sessions = prerequisite_root / "candidate_sessions.txt"
    candidate_sessions.write_text("20260101\n20260102\n", encoding="utf-8")
    source_identity = {
        "candidate_sessions": {
            "path": str(candidate_sessions),
            "sha256": _file_sha256(candidate_sessions),
        },
        "product_days": {"sha256": "2" * 64},
    }
    return {
        "binding_version": CROSS_SESSION_PREREQUISITE_BINDING_VERSION,
        "root": str(prerequisite_root),
        "schema_version": "cross_session_prerequisites_v1",
        "marker_sha256": "3" * 64,
        "marker_payload_sha256": "4" * 64,
        "config_sha256": "5" * 64,
        "source_identity": source_identity,
        "source_identity_sha256": _canonical_sha256(source_identity),
    }


def _write_cross_root(
    root: Path,
    inputs: ExitMakerPartitionInputs,
    *,
    nominal_cancel_semantics: str = "nominal_instant_cancel_v0",
    frozen_threshold_offset_bp: float = 0.0,
    completed_outcome_type: str = "terminal",
    bind_prerequisite: bool = True,
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for row in inputs.position_policy_facts.iter_rows(named=True):
        branch = str(row["nominal_instant_cancel_v0_branch"])
        if branch == "flat_same_day":
            category = OUTCOME_COMPLETED
            terminal_date = str(row["Date"])
        elif branch in {
            "carry_at_eod_cancel_unconfirmed",
            "carry_at_eod_no_admission",
            "no_fill_before_cancel_request",
        }:
            category = "still_open"
            terminal_date = None
        else:
            category = OUTCOME_UNKNOWN
            terminal_date = None
        records.append(
            {
                "Date": row["Date"],
                "ValueCode": row["ValueCode"],
                "QuoteCode": row["QuoteCode"],
                "entry_route": row["entry_route"],
                "entry_policy_generation_id": row[
                    "entry_policy_generation_id"
                ],
                "entry_raw_order_fact_id": row["entry_raw_order_fact_id"],
                "exit_rule_id": row["exit_rule_id"],
                "exit_route": row["exit_route"],
                "exit_policy_trial_id": row["exit_policy_trial_id"],
                "exit_rule_source_asof_date": row[
                    "exit_rule_source_asof_date"
                ],
                "frozen_exit_threshold_basis_bp": row[
                    "exit_threshold_basis_bp"
                ]
                + frozen_threshold_offset_bp,
                "filled_entry_outcome_category": category,
                "gross_cycle_pnl_twd": row["gross_cycle_pnl_twd"],
                "terminal_date": terminal_date,
                "terminal_reason": branch,
                "outcome_type": (
                    completed_outcome_type
                    if category == OUTCOME_COMPLETED
                    else "censored"
                ),
                "terminal_cashflow_priced": category == OUTCOME_COMPLETED,
                "cancel_semantics": nominal_cancel_semantics,
                "nominal_cancel_model_assumption": (
                    nominal_cancel_semantics == "nominal_instant_cancel_v0"
                ),
                "entry_no_fill_in_primary": False,
                "pathwise_ev_ready": False,
            }
        )
    nominal = pl.from_dicts(
        records,
        schema=_outcome_schema(),
        strict=True,
    )
    strict = nominal.with_columns(
        pl.lit("strict").alias("cancel_semantics"),
        pl.lit(False).alias("nominal_cancel_model_assumption"),
    )
    empty_attempts = pl.DataFrame(schema=_attempt_schema())
    audit = pl.from_dicts(
        [
            {
                "Date": DATE,
                "ValueCode": VALUE,
                "QuoteCode": QUOTE,
                "cancel_semantics": semantics,
                "policy_outcome_rows": nominal.height,
                "session_attempt_rows": 0,
                "completed_rows": 0,
                "same_day_completed_rows": 0,
                "cross_session_completed_rows": 0,
                "still_open_rows": 0,
                "censored_rows": 0,
                "unknown_rows": 0,
                "candidate_session_count": 0,
                "loaded_candidate_session_count": 0,
                "loaded_candidate_sessions": [],
                "load_failure_status": None,
                "load_failure_detail": None,
                "old_forced_taker_taker_used": False,
                "pathwise_ev_ready": False,
            }
            for semantics in ("strict", "nominal_instant_cancel_v0")
        ],
        schema=_RUNNER_AUDIT_SCHEMA,
        strict=True,
    )
    partition = root / f"Date={DATE}" / f"ValueCode={VALUE}"
    prerequisite_identity = _synthetic_prerequisite_identity(root)
    embedded_prerequisite = prerequisite_identity if bind_prerequisite else None
    embedded_runner = {
        "test_fixture": "filled_entry_cross_root",
        "prerequisite_identity": embedded_prerequisite,
        "global_inputs": {"prerequisite": embedded_prerequisite},
    }
    runner_hash = _canonical_sha256(embedded_runner)
    partition_config = {
        "runner": embedded_runner,
        "prerequisite": embedded_prerequisite,
        "test_fixture": "filled_entry_cross_root",
    }
    _publish_cross_partition(
        partition,
        date=DATE,
        value_code=VALUE,
        frames={
            "cross_session_strict_policy_outcomes.parquet": strict,
            "cross_session_strict_session_attempts.parquet": empty_attempts,
            "cross_session_nominal_policy_outcomes.parquet": nominal,
            "cross_session_nominal_session_attempts.parquet": empty_attempts,
            "cross_session_runner_audit.parquet": audit,
        },
        partition_config=partition_config,
        config_sha256=_canonical_sha256(partition_config),
        runner_config_sha256=runner_hash,
    )
    _rebuild_cross_root_manifest(root, runner_hash)
    return prerequisite_identity


class FilledEntryPrimaryReportTest(unittest.TestCase):
    def test_rule_source_must_be_exact_session_predecessor(self) -> None:
        inputs = _inputs()
        bad_source = "20251231"
        bad = replace(
            inputs,
            position_policy_facts=inputs.position_policy_facts.with_columns(
                pl.lit(bad_source).alias("exit_rule_source_asof_date")
            ),
            taker_exit_facts=inputs.taker_exit_facts.with_columns(
                pl.lit(bad_source).alias("exit_rule_source_asof_date")
            ),
        )
        with self.assertRaisesRegex(ValueError, "session predecessor"):
            build_filled_entry_primary_report(bad)

    def test_primary_cannot_downgrade_to_strictly_prior_rule_source(self) -> None:
        inputs = _inputs()
        metadata = dict(inputs.metadata)
        metadata.pop("session_predecessors")
        stale_source = "20251231"
        bad = replace(
            inputs,
            metadata=metadata,
            position_policy_facts=inputs.position_policy_facts.with_columns(
                pl.lit(stale_source).alias("exit_rule_source_asof_date")
            ),
            taker_exit_facts=inputs.taker_exit_facts.with_columns(
                pl.lit(stale_source).alias("exit_rule_source_asof_date")
            ),
        )
        with self.assertRaisesRegex(ValueError, "exact session_predecessors"):
            build_filled_entry_primary_report(bad)

    def test_position_rule_must_match_bound_entry_exit_facts(self) -> None:
        inputs = _inputs()
        bad = replace(
            inputs,
            position_policy_facts=inputs.position_policy_facts.with_columns(
                (pl.col("exit_threshold_basis_bp") + 7.0).alias(
                    "exit_threshold_basis_bp"
                )
            ),
        )
        with self.assertRaisesRegex(ValueError, "bound entry exit_facts"):
            build_filled_entry_primary_report(bad)

    def test_every_established_entry_requires_complete_four_action_grid(self) -> None:
        inputs = _inputs()
        bad = replace(
            inputs,
            position_policy_facts=inputs.position_policy_facts.filter(
                pl.col("exit_rule_id") == "frozen_center"
            ),
        )
        with self.assertRaisesRegex(ValueError, "exact frozen Center/Lower"):
            build_filled_entry_primary_report(bad)

    def test_contradictory_entry_fill_flags_fail_closed(self) -> None:
        inputs = _inputs()
        bad_actions = inputs.action_facts.with_columns(
            pl.when(pl.col("policy_generation_id") == "spot-q95")
            .then(pl.lit(False))
            .otherwise(pl.col("any_fill"))
            .alias("any_fill")
        )
        with self.assertRaisesRegex(ValueError, "fill flags are contradictory"):
            build_filled_entry_primary_report(
                replace(inputs, action_facts=bad_actions)
            )

    def test_shared_raw_no_fill_alias_keeps_physical_hedge_but_is_excluded(self) -> None:
        inputs = _inputs()
        shared = inputs.action_facts.filter(
            pl.col("raw_order_fact_id") == "shared-future-raw"
        )
        no_fill = shared.filter(
            pl.col("policy_generation_id")
            == "future-q80-shared-raw-no-fill"
        ).row(0, named=True)
        self.assertFalse(no_fill["full_fill"])
        self.assertEqual(no_fill["entry_hedge_status"], "executable")
        self.assertEqual(no_fill["entry_hedge_decision_time_ns"], 1_000_000_000)
        self.assertFalse(no_fill["entry_hedge_label_observed"])
        self.assertFalse(no_fill["entry_hedge_executable"])

        report = build_filled_entry_primary_report(inputs)
        self.assertNotIn(
            no_fill["policy_generation_id"],
            report.filled_entry_policy_paths[
                "entry_policy_generation_id"
            ].to_list(),
        )
        diagnostic = report.execution_diagnostics.filter(
            (pl.col("entry_route") == FUTURE_ENTRY)
            & (pl.col("boundary_quantile") == 80)
        ).row(0, named=True)
        self.assertEqual(diagnostic["submitted_entry_policy_aliases"], 2)
        self.assertEqual(diagnostic["known_no_fill_policy_aliases"], 1)
        self.assertEqual(
            diagnostic["entry_full_fill_hedge_executable_policy_aliases"], 1
        )

    def test_same_day_terminal_tuple_cannot_be_rewritten(self) -> None:
        inputs = _inputs()
        row = inputs.position_policy_facts.filter(
            pl.col("nominal_instant_cancel_v0_branch") == "flat_same_day"
        ).row(0, named=True)
        overlay = pl.DataFrame(
            {
                "exit_policy_trial_id": [row["exit_policy_trial_id"]],
                "filled_entry_outcome_category": [OUTCOME_COMPLETED],
                "gross_cycle_pnl_twd": [row["gross_cycle_pnl_twd"]],
                "terminal_date": ["20260105"],
                "terminal_reason": ["cross_session_rewrite"],
            }
        )
        with self.assertRaisesRegex(ValueError, "same-day terminal identity"):
            build_filled_entry_primary_report(
                inputs,
                terminal_policy_facts=overlay,
            )

    def test_same_day_overlay_reason_is_ignored_and_baseline_is_preserved(self) -> None:
        inputs = _inputs()
        row = inputs.position_policy_facts.filter(
            pl.col("nominal_instant_cancel_v0_branch") == "flat_same_day"
        ).row(0, named=True)
        overlay = pl.DataFrame(
            {
                "exit_policy_trial_id": [row["exit_policy_trial_id"]],
                "filled_entry_outcome_category": [OUTCOME_COMPLETED],
                "gross_cycle_pnl_twd": [row["gross_cycle_pnl_twd"]],
                "terminal_date": [DATE],
                "terminal_reason": ["same_day_target_exit"],
            }
        )
        report = build_filled_entry_primary_report(
            inputs,
            terminal_policy_facts=overlay,
        )
        path = report.filled_entry_policy_paths.filter(
            pl.col("exit_policy_trial_id") == row["exit_policy_trial_id"]
        ).row(0, named=True)
        self.assertEqual(path["terminal_date"], DATE)
        self.assertEqual(path["terminal_reason"], "flat_same_day")
        self.assertEqual(
            path["gross_cycle_pnl_twd"], row["gross_cycle_pnl_twd"]
        )

    def test_primary_cross_root_rejects_unbound_prerequisite(self) -> None:
        inputs = _inputs()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cross"
            root.mkdir()
            expected = _write_cross_root(
                root,
                inputs,
                bind_prerequisite=False,
            )
            with self.assertRaisesRegex(ValueError, "prerequisite identity mismatch"):
                load_cross_session_nominal_terminal_facts(
                    root,
                    expected_coverage=inputs.coverage,
                    expected_position_policy_facts=inputs.position_policy_facts,
                    expected_prerequisite_identity=expected,
                )
            with self.assertRaisesRegex(ValueError, "explicit prerequisite identity"):
                load_cross_session_nominal_terminal_facts(
                    root,
                    expected_coverage=inputs.coverage,
                    expected_position_policy_facts=inputs.position_policy_facts,
                    expected_prerequisite_identity=None,
                )

    def test_primary_is_filled_only_dealiased_and_reproduces_12_cell_shape(self) -> None:
        report = build_filled_entry_primary_report(_inputs())

        self.assertEqual(report.policy_summary.height, 24)
        self.assertEqual(report.pooled_12_cell_summary.height, 12)
        self.assertFalse(report.metadata["gross_zero_imputation"])
        self.assertFalse(report.metadata["unknown_or_censored_cashflow_imputed"])
        self.assertFalse(report.metadata["net_or_after_cost_statistics_present"])
        self.assertFalse(report.metadata["non_price_cost_profile_applied"])
        self.assertEqual(report.metadata["entry_no_fill_rows_in_primary"], 0)
        self.assertEqual(
            report.metadata["cross_q_shared_physical_entry_dependencies"], 1
        )

        q50_future_entry = report.policy_summary.filter(
            (pl.col("entry_route") == FUTURE_ENTRY)
            & (pl.col("boundary_quantile") == 50)
            & (pl.col("exit_rule_id") == "frozen_center")
            & (pl.col("exit_route") == FUTURE_EXIT)
        ).row(0, named=True)
        self.assertEqual(q50_future_entry["filled_entry_positions"], 1)
        self.assertEqual(q50_future_entry["filled_entry_policy_aliases"], 2)
        self.assertEqual(q50_future_entry["completed_cycles"], 1)
        self.assertAlmostEqual(q50_future_entry["gross_cycle_bp_p50"], 50.0)

        pooled_completed = report.pooled_12_cell_summary.filter(
            (pl.col("boundary_quantile") == 50)
            & (pl.col("exit_rule_id") == "frozen_center")
            & (pl.col("exit_route") == FUTURE_EXIT)
        ).row(0, named=True)
        self.assertEqual(pooled_completed["filled_entry_positions"], 2)
        self.assertEqual(pooled_completed["completed_cycles"], 2)
        self.assertEqual(pooled_completed["completion_rate_given_filled_entry"], 1.0)
        self.assertAlmostEqual(pooled_completed["gross_cycle_bp_mean"], 50.0)

        pooled_open = report.pooled_12_cell_summary.filter(
            (pl.col("boundary_quantile") == 50)
            & (pl.col("exit_rule_id") == "frozen_center")
            & (pl.col("exit_route") == SPOT_EXIT)
        ).row(0, named=True)
        self.assertEqual(pooled_open["still_open_positions"], 2)
        self.assertEqual(pooled_open["completed_cycles"], 0)
        self.assertIsNone(pooled_open["gross_cycle_bp_mean"])

        pooled_unknown = report.pooled_12_cell_summary.filter(
            (pl.col("boundary_quantile") == 50)
            & (pl.col("exit_rule_id") == "frozen_lower")
            & (pl.col("exit_route") == FUTURE_EXIT)
        ).row(0, named=True)
        self.assertEqual(pooled_unknown["unknown_positions"], 2)
        self.assertIsNone(pooled_unknown["gross_cycle_bp_p50"])
        self.assertFalse(
            report.filled_entry_policy_paths["entry_no_fill_included"].any()
        )
        self.assertFalse(
            report.pooled_12_cell_summary["boundary_quantile_rows_safe_to_sum"].any()
        )

    def test_execution_diagnostics_are_separate_and_have_no_profit_columns(self) -> None:
        report = build_filled_entry_primary_report(_inputs())
        future_q50 = report.execution_diagnostics.filter(
            (pl.col("entry_route") == FUTURE_ENTRY)
            & (pl.col("boundary_quantile") == 50)
        ).row(0, named=True)
        self.assertEqual(future_q50["submitted_entry_policy_aliases"], 3)
        self.assertEqual(future_q50["submitted_physical_raw_orders"], 2)
        self.assertEqual(future_q50["known_no_fill_policy_aliases"], 1)
        self.assertEqual(
            future_q50["entry_full_fill_hedge_executable_policy_aliases"], 2
        )
        self.assertEqual(future_q50["entry_cancel_request_policy_aliases"], 1)
        self.assertTrue(future_q50["cancel_rate_is_request_only"])
        self.assertFalse(
            any(
                token in column
                for column in report.execution_diagnostics.columns
                for token in ("gross", "net_pnl", "profit")
            )
        )

    def test_cross_session_overlay_prices_open_path_without_zero_imputation(self) -> None:
        inputs = _inputs()
        target = inputs.position_policy_facts.filter(
            (pl.col("entry_policy_generation_id") == "spot-q95")
            & (pl.col("exit_rule_id") == "frozen_center")
            & (pl.col("exit_route") == SPOT_EXIT)
        ).item(0, "exit_policy_trial_id")
        censored = inputs.position_policy_facts.filter(
            (pl.col("entry_policy_generation_id") == "spot-q95")
            & (pl.col("exit_rule_id") == "frozen_lower")
            & (pl.col("exit_route") == FUTURE_EXIT)
        ).item(0, "exit_policy_trial_id")
        terminal = pl.from_dicts(
            [
                {
                    "exit_policy_trial_id": target,
                    "filled_entry_outcome_category": OUTCOME_COMPLETED,
                    "gross_cycle_pnl_twd": 3_000.0,
                    "terminal_date": "20260105",
                    "terminal_reason": "cross_session_maker_fill_and_50ms_hedge",
                },
                {
                    "exit_policy_trial_id": censored,
                    "filled_entry_outcome_category": OUTCOME_CENSORED,
                    "gross_cycle_pnl_twd": None,
                    "terminal_date": None,
                    "terminal_reason": "observation_horizon",
                },
            ],
            infer_schema_length=None,
        )
        report = build_filled_entry_primary_report(
            inputs,
            terminal_policy_facts=terminal,
        )
        target_path = report.filled_entry_policy_paths.filter(
            pl.col("exit_policy_trial_id") == target
        ).row(0, named=True)
        self.assertEqual(target_path["filled_entry_outcome_category"], OUTCOME_COMPLETED)
        self.assertEqual(target_path["terminal_date"], "20260105")
        self.assertAlmostEqual(target_path["gross_cycle_bp"], 150.0)
        censored_path = report.filled_entry_policy_paths.filter(
            pl.col("exit_policy_trial_id") == censored
        ).row(0, named=True)
        self.assertEqual(censored_path["filled_entry_outcome_category"], OUTCOME_CENSORED)
        self.assertIsNone(censored_path["gross_cycle_bp"])

        invalid = terminal.with_columns(
            pl.when(pl.col("exit_policy_trial_id") == censored)
            .then(pl.lit(1.0))
            .otherwise(pl.col("gross_cycle_pnl_twd"))
            .alias("gross_cycle_pnl_twd")
        )
        with self.assertRaisesRegex(ValueError, "must not carry imputed gross"):
            build_filled_entry_primary_report(
                inputs,
                terminal_policy_facts=invalid,
            )

    def test_overlay_cannot_rewrite_same_day_completion(self) -> None:
        inputs = _inputs()
        trial = inputs.position_policy_facts.filter(
            pl.col("nominal_instant_cancel_v0_branch") == "flat_same_day"
        ).item(0, "exit_policy_trial_id")
        overlay = pl.DataFrame(
            {
                "exit_policy_trial_id": [trial],
                "filled_entry_outcome_category": [OUTCOME_UNKNOWN],
                "gross_cycle_pnl_twd": [None],
            },
            schema_overrides={"gross_cycle_pnl_twd": pl.Float64},
        )
        with self.assertRaisesRegex(ValueError, "contradicts"):
            build_filled_entry_primary_report(
                inputs,
                terminal_policy_facts=overlay,
            )

    def test_partitioned_cross_root_loader_is_nominal_only_and_lineage_exact(self) -> None:
        inputs = _inputs()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cross"
            root.mkdir()
            prerequisite_identity = _write_cross_root(root, inputs)
            overlay = load_cross_session_nominal_terminal_facts(
                root,
                expected_coverage=inputs.coverage,
                expected_position_policy_facts=inputs.position_policy_facts,
                expected_prerequisite_identity=prerequisite_identity,
            )
            self.assertEqual(
                overlay.terminal_policy_facts.height,
                inputs.position_policy_facts.height,
            )
            self.assertEqual(
                set(overlay.terminal_policy_facts["cancel_semantics"]),
                {"nominal_instant_cancel_v0"},
            )
            self.assertFalse(
                overlay.metadata["strict_cross_session_outcomes_in_primary"]
            )
            self.assertTrue(
                overlay.metadata[
                    "cross_session_terminal_and_censored_outcomes_separate"
                ]
            )
            report = build_filled_entry_primary_report(
                inputs,
                terminal_policy_facts=overlay.terminal_policy_facts,
            )
            self.assertFalse(report.metadata["gross_zero_imputation"])

    def test_partitioned_cross_root_loader_rejects_strict_as_primary(self) -> None:
        inputs = _inputs()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cross"
            root.mkdir()
            prerequisite_identity = _write_cross_root(
                root, inputs, nominal_cancel_semantics="strict"
            )
            with self.assertRaisesRegex(
                ValueError, "must be nominal_instant_cancel_v0"
            ):
                load_cross_session_nominal_terminal_facts(
                    root,
                    expected_coverage=inputs.coverage,
                    expected_position_policy_facts=inputs.position_policy_facts,
                    expected_prerequisite_identity=prerequisite_identity,
                )

    def test_partitioned_cross_root_loader_rejects_threshold_lineage_drift(self) -> None:
        inputs = _inputs()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cross"
            root.mkdir()
            prerequisite_identity = _write_cross_root(
                root,
                inputs,
                frozen_threshold_offset_bp=0.01,
            )
            with self.assertRaisesRegex(ValueError, "frozen same-day policy lineage"):
                load_cross_session_nominal_terminal_facts(
                    root,
                    expected_coverage=inputs.coverage,
                    expected_position_policy_facts=inputs.position_policy_facts,
                    expected_prerequisite_identity=prerequisite_identity,
                )

    def test_partitioned_cross_root_loader_keeps_terminal_and_censored_separate(self) -> None:
        inputs = _inputs()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cross"
            root.mkdir()
            prerequisite_identity = _write_cross_root(
                root,
                inputs,
                completed_outcome_type="censored",
            )
            with self.assertRaisesRegex(
                ValueError, "priced completed cycles terminal"
            ):
                load_cross_session_nominal_terminal_facts(
                    root,
                    expected_coverage=inputs.coverage,
                    expected_position_policy_facts=inputs.position_policy_facts,
                    expected_prerequisite_identity=prerequisite_identity,
                )

    def test_atomic_publish_and_cli_defaults(self) -> None:
        report = build_filled_entry_primary_report(_inputs())
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "report"
            _publish_report(report, output)
            marker = json.loads((output / "report_complete.json").read_text())
            self.assertTrue(marker["complete"])
            self.assertEqual(marker["report_version"], REPORT_VERSION)
            sources = marker["report_implementation_sources"]
            self.assertEqual(
                sources["filled_entry_report.py"],
                _file_sha256(
                    Path(
                        __import__(
                            "maker.src.quote_fill.filled_entry_report",
                            fromlist=["__file__"],
                        ).__file__
                    )
                ),
            )
            self.assertEqual(
                marker["report_implementation_sources_sha256"],
                _report_canonical_sha256(sources),
            )
            self.assertEqual(len(marker["artifacts"]), 6)
            self.assertTrue((output / "filled_entry_policy_paths.parquet").is_file())
            self.assertTrue(
                (output / "filled_entry_primary_pooled_12_cells.csv").is_file()
            )
            with self.assertRaises(FileExistsError):
                _publish_report(report, output)

        parsed = _parse_args(
            ["--exit-root", "/tmp/exit", "--entry-root", "/tmp/entry"]
        )
        self.assertEqual(parsed.sessions, 60)
        self.assertFalse(parsed.skip_hash_validation)
        self.assertIsNone(parsed.terminal_policy_facts)
        self.assertIsNone(parsed.cross_session_root)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            _parse_args(
                [
                    "--exit-root",
                    "/tmp/exit",
                    "--entry-root",
                    "/tmp/entry",
                    "--terminal-policy-facts",
                    "/tmp/terminal.parquet",
                    "--cross-session-root",
                    "/tmp/cross",
                ]
            )


if __name__ == "__main__":
    unittest.main()
