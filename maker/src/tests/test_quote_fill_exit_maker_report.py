from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import polars as pl

from maker.src.quote_fill.exit_maker_report import (
    ExitMakerPartitionInputs,
    _canonical_sha256,
    _file_sha256,
    _parse_args,
    _validate_entry_action_execution_contract,
    build_exit_maker_report,
    load_exit_maker_partition_inputs,
    run_exit_maker_report,
)


VALUE_CODE = "2330"
QUOTE_CODE = "CDF1"
ENTRY_ROUTE = "future_ask_spot_taker"
FUTURE_EXIT = "future_bid_spot_taker"
SPOT_EXIT = "spot_ask_future_taker"
RULE = "frozen_center"


def _canonical_raw_id(entry_raw: str, route: str) -> str:
    components = (entry_raw, route, 1, 200, 1_200_000_000, 10, 0)
    digest = hashlib.sha256(
        "|".join(map(str, components)).encode("utf-8")
    ).hexdigest()[:24]
    return f"exit-raw-{digest}"


def _frames(date: str) -> dict[str, pl.DataFrame]:
    source_date = {
        "20260102": "20260101",
        "20260105": "20260102",
    }.get(date, "20260101")
    entry_raw = f"entry-raw-{date}"
    policies = [f"{date}/q50/a", f"{date}/q50/b"]
    actions = pl.from_dicts(
        [
            {
                "Date": date,
                "ValueCode": VALUE_CODE,
                "QuoteCode": QUOTE_CODE,
                "route": ENTRY_ROUTE,
                "parameter_version": "boundary-v1",
                "boundary_quantile": 50,
                "lookup_action_id": "q50",
                "raw_order_fact_id": entry_raw,
                "policy_generation_id": policy,
                "full_fill": True,
                "any_fill": True,
                "partial_fill": False,
                "full_fill_recv_time_ns": 950_000_000,
                "entry_hedge_status": "executable",
                "entry_hedge_decision_time_ns": 1_000_000_000,
                "entry_hedge_label_observed": True,
                "entry_hedge_executable": True,
                "submit_recv_time_ns": 0,
                "entry_spot_price": 100.0,
                "entry_hedge_contract_size_shares": 2_000,
                "entry_hedge_signed_latency_slippage_bp": 1.0,
                "entry_hedge_signed_depth_slippage_bp": 2.0,
                "entry_hedge_signed_total_slippage_bp": 3.0,
                "entry_hedge_decision_book_age_ms": 50.0,
            }
            for policy in policies
        ],
        infer_schema_length=None,
    )
    support_rows: list[dict[str, object]] = []
    alias_rows: list[dict[str, object]] = []
    position_rows: list[dict[str, object]] = []
    raw_id = _canonical_raw_id(entry_raw, FUTURE_EXIT)
    for policy in policies:
        for route in (FUTURE_EXIT, SPOT_EXIT):
            trial = f"{policy}/exit/{RULE}/{route}"
            admitted = route == FUTURE_EXIT
            support_rows.append(
                {
                    "Date": date,
                    "ValueCode": VALUE_CODE,
                    "QuoteCode": QUOTE_CODE,
                    "entry_route": ENTRY_ROUTE,
                    "entry_policy_generation_id": policy,
                    "entry_raw_order_fact_id": entry_raw,
                    "exit_rule_id": RULE,
                    "exit_route": route,
                    "exit_policy_trial_id": trial,
                    "exit_threshold_basis_bp": 100.0,
                    "exit_rule_source_asof_date": source_date,
                    "position_status": "position_established",
                    "admission_status": "admitted" if admitted else "no_admission",
                    "raw_candidate_count": 1 if admitted else 0,
                }
            )
            if admitted:
                alias_rows.append(
                    {
                        "Date": date,
                        "ValueCode": VALUE_CODE,
                        "QuoteCode": QUOTE_CODE,
                        "entry_route": ENTRY_ROUTE,
                        "entry_policy_generation_id": policy,
                        "entry_raw_order_fact_id": entry_raw,
                        "exit_rule_id": RULE,
                        "exit_route": route,
                        "exit_policy_trial_id": trial,
                        "exit_raw_candidate_fact_id": raw_id,
                        "exit_candidate_alias_id": f"{trial}/candidate",
                        "cancel_required": True,
                        "any_fill": True,
                        "full_fill": True,
                        "partial_fill": False,
                        "exit_hedge_status": "executable",
                    }
                )
            position_rows.append(
                {
                    "Date": date,
                    "ValueCode": VALUE_CODE,
                    "QuoteCode": QUOTE_CODE,
                    "entry_route": ENTRY_ROUTE,
                    "entry_policy_generation_id": policy,
                    "entry_raw_order_fact_id": entry_raw,
                    "exit_rule_id": RULE,
                    "exit_route": route,
                    "exit_policy_trial_id": trial,
                    "exit_threshold_basis_bp": 100.0,
                    "exit_rule_source_asof_date": source_date,
                    "position_status": "position_established",
                    "position_established_ns": 1_000_000_000,
                    "nominal_instant_cancel_v0_branch": (
                        "flat_same_day" if admitted else "carry_at_eod_no_admission"
                    ),
                    "branch_status": (
                        "cancel_race_unknown"
                        if admitted
                        else "carry_at_eod_no_admission"
                    ),
                    "terminal_outcome": False,
                    "exit_decision_time_ns": 2_000_000_000 if admitted else None,
                    "gross_cycle_pnl_twd": 1_000.0 if admitted else None,
                    "exit_hedge_status": "executable" if admitted else None,
                    "exit_hedge_signed_latency_slippage_bp": 2.0 if admitted else None,
                    "exit_hedge_signed_depth_slippage_bp": 3.0 if admitted else None,
                    "exit_hedge_signed_total_slippage_bp": 5.0 if admitted else None,
                    "oco_active_sibling_cancel_count": 1 if admitted else 0,
                    "prior_unacked_cancel_count_before_winner": 0,
                    "cancel_ack_observed": False,
                    "joint_volume_allocated": False,
                }
            )
    raw = pl.from_dicts(
        [
            {
                "Date": date,
                "ValueCode": VALUE_CODE,
                "QuoteCode": QUOTE_CODE,
                "entry_raw_order_fact_id": entry_raw,
                "exit_route": FUTURE_EXIT,
                "exit_raw_candidate_fact_id": raw_id,
                "spread_pair_epoch": 1,
                "target_price_tick": 200,
                "submit_recv_time_ns": 1_200_000_000,
                "submit_event_sequence": 10,
                "submit_row_index": 0,
                "cancel_required": True,
                "cancel_ack_observed": False,
                "any_fill": True,
                "full_fill": True,
                "partial_fill": False,
                "exit_hedge_status": "executable",
            }
        ],
        infer_schema_length=None,
    )
    exits = pl.from_dicts(
        [
            {
                "Date": date,
                "ValueCode": VALUE_CODE,
                "QuoteCode": QUOTE_CODE,
                "route": ENTRY_ROUTE,
                "raw_order_fact_id": entry_raw,
                "policy_generation_id": policy,
                "exit_rule_id": RULE,
                "exit_threshold_basis_bp": 100.0,
                "exit_rule_source_asof_date": source_date,
                "branch_status": (
                    "same_day_taker_exit" if index == 0 else "carry_at_eod"
                ),
                "terminal_outcome": index == 0,
                "exit_decision_time_ns": 1_800_000_000 if index == 0 else None,
                "gross_cycle_pnl_twd": 600.0 if index == 0 else None,
                "exit_added_latency_ns": 0,
            }
            for index, policy in enumerate(policies)
        ],
        infer_schema_length=None,
    )
    return {
        "actions": actions,
        "exits": exits,
        "support": pl.from_dicts(support_rows, infer_schema_length=None),
        "aliases": pl.from_dicts(alias_rows, infer_schema_length=None),
        "raw": raw,
        "positions": pl.from_dicts(position_rows, infer_schema_length=None),
    }


def _inputs(date: str = "20260102") -> ExitMakerPartitionInputs:
    frames = _frames(date)
    return ExitMakerPartitionInputs(
        policy_support=frames["support"],
        candidate_aliases=frames["aliases"],
        raw_candidate_facts=frames["raw"],
        position_policy_facts=frames["positions"],
        action_facts=frames["actions"],
        taker_exit_facts=frames["exits"],
        coverage=pl.DataFrame(
            {
                "Date": [date],
                "ValueCode": [VALUE_CODE],
                "partition_complete": [True],
                "partition": ["synthetic"],
            }
        ),
        metadata={
            "selected_session_count": 1,
            "hedge_delay_ns": 50_000_000,
            "taker_taker_exit_grid_ns": 1_000_000_000,
        },
    )


def _artifact_meta(path: Path, frame: pl.DataFrame) -> dict[str, object]:
    frame.write_parquet(path)
    return {
        "rows": frame.height,
        "columns": frame.width,
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _publish_synthetic_partition(exit_root: Path, entry_root: Path, date: str) -> None:
    frames = _frames(date)
    entry_partition = entry_root / f"Date={date}" / f"ValueCode={VALUE_CODE}"
    entry_partition.mkdir(parents=True)
    entry_artifacts = {
        "execution_action_facts.parquet": _artifact_meta(
            entry_partition / "execution_action_facts.parquet", frames["actions"]
        ),
        "exit_facts.parquet": _artifact_meta(
            entry_partition / "exit_facts.parquet", frames["exits"]
        ),
    }
    entry_config = {
        "runner_version": "synthetic-entry-v1",
        "hedge_delay_ns": 50_000_000,
        "exit_path": {
            "grid_ns": 1_000_000_000,
            "max_book_age_ns": 1_000_000_000,
        },
    }
    entry_config_sha = _canonical_sha256(entry_config)
    (entry_partition / "complete.json").write_text(
        json.dumps(
            {
                "complete": True,
                "Date": date,
                "ValueCode": VALUE_CODE,
                "runner_version": "synthetic-entry-v1",
                "config": entry_config,
                "config_sha256": entry_config_sha,
                "artifacts": entry_artifacts,
            },
            sort_keys=True,
        )
    )

    exit_partition = exit_root / f"Date={date}" / f"ValueCode={VALUE_CODE}"
    exit_partition.mkdir(parents=True)
    materialized = {
        "exit_maker_policy_support.parquet": frames["support"],
        "exit_maker_observations.parquet": pl.DataFrame({"placeholder": []}, schema={"placeholder": pl.Int64}),
        "exit_maker_transitions.parquet": pl.DataFrame({"placeholder": []}, schema={"placeholder": pl.Int64}),
        "exit_maker_candidate_aliases.parquet": frames["aliases"],
        "exit_maker_raw_candidate_facts.parquet": frames["raw"],
        "exit_maker_position_policy_facts.parquet": frames["positions"],
        "exit_maker_audit.parquet": pl.DataFrame({"placeholder": []}, schema={"placeholder": pl.Int64}),
    }
    exit_artifacts = {
        name: _artifact_meta(exit_partition / name, frame)
        for name, frame in materialized.items()
    }
    runner = {
        "runner_version": "exit_maker_product_day_v1",
        "hedge_delay_ns": 50_000_000,
        "routes": [FUTURE_EXIT, SPOT_EXIT],
        "exit_rule_ids": [RULE, "frozen_lower"],
    }
    source = {
        "Date": date,
        "ValueCode": VALUE_CODE,
        "upstream_config_sha256": entry_config_sha,
        "action_source": {
            "artifact": "execution_action_facts.parquet",
            "sha256": entry_artifacts["execution_action_facts.parquet"]["sha256"],
        },
        "exit_rule_source": {
            "kind": "entry_execution_exit_facts_v1",
            "artifact": "exit_facts.parquet",
            "sha256": entry_artifacts["exit_facts.parquet"]["sha256"],
        },
    }
    config = {"runner": runner, "source": source}
    (exit_partition / "complete.json").write_text(
        json.dumps(
            {
                "complete": True,
                "Date": date,
                "ValueCode": VALUE_CODE,
                "runner_version": "exit_maker_product_day_v1",
                "runner_config_sha256": _canonical_sha256(runner),
                "config": config,
                "config_sha256": _canonical_sha256(config),
                "artifacts": exit_artifacts,
            },
            sort_keys=True,
        )
    )
    manifest_rows = []
    for marker_path in sorted(
        exit_root.glob("Date=*/ValueCode=*/complete.json")
    ):
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
        manifest_rows.append(
            {
                "Date": payload["Date"],
                "ValueCode": payload["ValueCode"],
                "runner_config_sha256": payload["runner_config_sha256"],
                "config_sha256": payload["config_sha256"],
                "complete": True,
            }
        )
    pl.from_dicts(manifest_rows, infer_schema_length=None).sort(
        ["Date", "ValueCode"]
    ).write_parquet(exit_root / "exit_maker_partition_manifest.parquet")


class ExitMakerReportTest(unittest.TestCase):
    def test_shared_raw_no_fill_alias_may_retain_physical_hedge_fields(self) -> None:
        actions = _frames("20260102")["actions"]
        no_fill = actions.head(1).with_columns(
            pl.lit("20260102/q80/no-fill").alias("policy_generation_id"),
            pl.lit(80, dtype=pl.Int64).alias("boundary_quantile"),
            pl.lit(False).alias("any_fill"),
            pl.lit(False).alias("full_fill"),
            pl.lit(False).alias("partial_fill"),
            pl.lit(None, dtype=pl.Int64).alias("full_fill_recv_time_ns"),
            # Physical raw-order hedge fields intentionally remain populated.
            pl.lit(False).alias("entry_hedge_label_observed"),
            pl.lit(False).alias("entry_hedge_executable"),
        )
        validated = _validate_entry_action_execution_contract(
            pl.concat([actions, no_fill], how="vertical")
        )
        row = validated.filter(
            pl.col("policy_generation_id") == "20260102/q80/no-fill"
        ).row(0, named=True)
        self.assertEqual(row["entry_hedge_status"], "executable")
        self.assertIsNotNone(row["entry_hedge_decision_time_ns"])
        self.assertIsNone(row["entry_action_hedge_delay_ns"])

    def test_shared_raw_full_aliases_must_agree_on_physical_hedge(self) -> None:
        actions = _frames("20260102")["actions"]
        second_policy = actions.item(1, "policy_generation_id")
        tampered_status = actions.with_columns(
            pl.when(pl.col("policy_generation_id") == second_policy)
            .then(pl.lit("insufficient_depth"))
            .otherwise(pl.col("entry_hedge_status"))
            .alias("entry_hedge_status"),
            pl.when(pl.col("policy_generation_id") == second_policy)
            .then(pl.lit(False))
            .otherwise(pl.col("entry_hedge_executable"))
            .alias("entry_hedge_executable"),
        )
        with self.assertRaisesRegex(ValueError, "canonical physical hedge"):
            _validate_entry_action_execution_contract(tampered_status)

        tampered_cursor = actions.with_columns(
            pl.when(pl.col("policy_generation_id") == second_policy)
            .then(pl.col("full_fill_recv_time_ns") + 1_000_000)
            .otherwise(pl.col("full_fill_recv_time_ns"))
            .alias("full_fill_recv_time_ns"),
            pl.when(pl.col("policy_generation_id") == second_policy)
            .then(pl.col("entry_hedge_decision_time_ns") + 1_000_000)
            .otherwise(pl.col("entry_hedge_decision_time_ns"))
            .alias("entry_hedge_decision_time_ns"),
        )
        with self.assertRaisesRegex(ValueError, "canonical physical hedge"):
            _validate_entry_action_execution_contract(tampered_cursor)

    def test_entry_marker_hedge_delay_is_bound_to_exit_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exit_root = root / "exit"
            entry_root = root / "entry"
            _publish_synthetic_partition(exit_root, entry_root, "20260102")
            entry_marker = (
                entry_root
                / "Date=20260102"
                / f"ValueCode={VALUE_CODE}"
                / "complete.json"
            )
            entry_payload = json.loads(entry_marker.read_text(encoding="utf-8"))
            entry_payload["config"]["hedge_delay_ns"] = 100_000_000
            entry_payload["config_sha256"] = _canonical_sha256(
                entry_payload["config"]
            )
            entry_marker.write_text(
                json.dumps(entry_payload, sort_keys=True), encoding="utf-8"
            )
            exit_marker = (
                exit_root
                / "Date=20260102"
                / f"ValueCode={VALUE_CODE}"
                / "complete.json"
            )
            exit_payload = json.loads(exit_marker.read_text(encoding="utf-8"))
            exit_payload["config"]["source"]["upstream_config_sha256"] = (
                entry_payload["config_sha256"]
            )
            exit_payload["config_sha256"] = _canonical_sha256(
                exit_payload["config"]
            )
            exit_marker.write_text(
                json.dumps(exit_payload, sort_keys=True), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "entry action marker hedge delay"):
                load_exit_maker_partition_inputs(
                    exit_root,
                    entry_root,
                    sessions=1,
                )

    def test_entry_action_observed_hedge_delay_must_be_50ms(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exit_root = root / "exit"
            entry_root = root / "entry"
            _publish_synthetic_partition(exit_root, entry_root, "20260102")
            entry_partition = (
                entry_root / "Date=20260102" / f"ValueCode={VALUE_CODE}"
            )
            action_path = entry_partition / "execution_action_facts.parquet"
            actions = pl.read_parquet(action_path).with_columns(
                (pl.col("entry_hedge_decision_time_ns") + 50_000_000).alias(
                    "entry_hedge_decision_time_ns"
                )
            )
            actions.write_parquet(action_path)
            entry_marker = entry_partition / "complete.json"
            entry_payload = json.loads(entry_marker.read_text(encoding="utf-8"))
            entry_payload["artifacts"][action_path.name] = {
                **entry_payload["artifacts"][action_path.name],
                "bytes": action_path.stat().st_size,
                "sha256": _file_sha256(action_path),
            }
            entry_marker.write_text(
                json.dumps(entry_payload, sort_keys=True), encoding="utf-8"
            )
            exit_marker = (
                exit_root
                / "Date=20260102"
                / f"ValueCode={VALUE_CODE}"
                / "complete.json"
            )
            exit_payload = json.loads(exit_marker.read_text(encoding="utf-8"))
            exit_payload["config"]["source"]["action_source"]["sha256"] = (
                _file_sha256(action_path)
            )
            exit_payload["config_sha256"] = _canonical_sha256(
                exit_payload["config"]
            )
            exit_marker.write_text(
                json.dumps(exit_payload, sort_keys=True), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "incoherent 50 ms hedge facts"):
                load_exit_maker_partition_inputs(
                    exit_root,
                    entry_root,
                    sessions=1,
                )

    def test_formal_loader_requires_same_day_root_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exit_root = root / "exit"
            entry_root = root / "entry"
            _publish_synthetic_partition(exit_root, entry_root, "20260102")
            (exit_root / "exit_maker_partition_manifest.parquet").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "root manifest"):
                load_exit_maker_partition_inputs(
                    exit_root,
                    entry_root,
                    sessions=1,
                )

    def test_sampling_units_nominal_cost_and_matched_baseline(self) -> None:
        report = build_exit_maker_report(_inputs())
        future = report.product_policy.filter(pl.col("exit_route") == FUTURE_EXIT)
        spot = report.product_policy.filter(pl.col("exit_route") == SPOT_EXIT)
        self.assertEqual(future.height, 1)
        self.assertEqual(future.item(0, "established_policy_trials"), 2)
        self.assertEqual(future.item(0, "unique_canonical_raw_candidates"), 1)
        self.assertEqual(future.item(0, "policy_candidate_aliases"), 2)
        self.assertEqual(future.item(0, "raw_cancel_required_rate"), 1.0)
        self.assertEqual(
            future.item(0, "raw_full_fill_rate_given_fill_known"), 1.0
        )
        self.assertEqual(future.item(0, "nominal_same_day_rate"), 1.0)
        self.assertEqual(future.item(0, "strict_unknown_rate"), 1.0)
        self.assertAlmostEqual(future.item(0, "nominal_same_day_gross_bp_p50"), 50.0)
        self.assertAlmostEqual(
            future.item(0, "nominal_same_day_gross_minus_cost_bp_p50"), 31.0
        )
        self.assertEqual(future.item(0, "nominal_same_day_positive_after_cost_rate"), 1.0)
        self.assertEqual(future.item(0, "nominal_same_day_holding_ms_p50"), 1_000.0)
        self.assertEqual(future.item(0, "nominal_same_day_capital_seconds_p50"), 2.0)
        self.assertEqual(future.item(0, "exit_threshold_basis_bp_exact"), 100.0)
        self.assertEqual(future.item(0, "exit_threshold_basis_bp_p10"), 100.0)
        self.assertEqual(future.item(0, "exit_threshold_basis_bp_p50"), 100.0)
        self.assertEqual(future.item(0, "exit_threshold_basis_bp_p90"), 100.0)
        self.assertEqual(
            future.item(0, "exit_rule_source_asof_date_exact"), "20260101"
        )
        self.assertEqual(future.item(0, "d_minus_one_lineage_valid_rate"), 1.0)
        self.assertEqual(future.item(0, "entry_hedge_total_slippage_bp_p50"), 3.0)
        self.assertEqual(future.item(0, "entry_hedge_total_slippage_bp_p95"), 3.0)
        self.assertEqual(future.item(0, "entry_hedge_fresh_le_100ms_rate"), 1.0)
        self.assertEqual(
            future.item(0, "nominal_same_day_exit_hedge_total_slippage_bp_p50"),
            5.0,
        )
        self.assertEqual(
            future.item(0, "nominal_same_day_exit_hedge_total_slippage_bp_p95"),
            5.0,
        )
        self.assertFalse(future.item(0, "cancel_ack_observation_available"))
        self.assertFalse(future.item(0, "exit_hedge_book_age_available"))
        self.assertEqual(spot.item(0, "unique_canonical_raw_candidates"), 0)
        self.assertEqual(spot.item(0, "nominal_carry_rate"), 1.0)

        future_pairs = report.matched_pairs.filter(pl.col("exit_route") == FUTURE_EXIT)
        self.assertEqual(future_pairs.height, 2)
        self.assertEqual(future_pairs["taker_taker_baseline_ref_id"].n_unique(), 2)
        self.assertTrue(future_pairs["baseline_reference_weight"].eq(0.5).all())
        self.assertEqual(future_pairs["both_same_day_completed"].sum(), 1)
        paired = future_pairs.filter(pl.col("both_same_day_completed"))
        self.assertAlmostEqual(paired.item(0, "paired_maker_minus_taker_gross_bp"), 20.0)
        self.assertEqual(paired.item(0, "exit_hedge_signed_total_slippage_bp"), 5.0)
        self.assertEqual(paired.item(0, "taker_taker_exit_added_latency_ns"), 0)
        self.assertEqual(paired.item(0, "maker_exit_hedge_delay_ns"), 50_000_000)
        self.assertEqual(paired.item(0, "taker_taker_exit_grid_ns"), 1_000_000_000)
        self.assertFalse(paired.item(0, "latency_matched"))
        self.assertEqual(
            paired.item(0, "taker_baseline_role"),
            "optimistic_zero_added_latency_benchmark",
        )
        self.assertFalse(future_pairs["route_copy_is_independent_baseline"].any())
        self.assertFalse(report.metadata["latency_matched"])
        self.assertEqual(report.policy_threshold_lineage.height, 4)
        self.assertTrue(
            report.policy_threshold_lineage[
                "source_strictly_precedes_target_date"
            ].all()
        )
        daily_future = report.daily_policy.filter(
            pl.col("exit_route") == FUTURE_EXIT
        )
        self.assertEqual(
            daily_future.item(0, "exit_threshold_basis_bp_exact"), 100.0
        )
        self.assertEqual(
            daily_future.item(0, "exit_rule_source_asof_date_exact"),
            "20260101",
        )

    def test_threshold_distribution_and_source_lineage_are_not_collapsed(self) -> None:
        inputs = _inputs()
        second_policy = inputs.action_facts.item(1, "policy_generation_id")

        def vary(frame: pl.DataFrame, policy_column: str) -> pl.DataFrame:
            return frame.with_columns(
                pl.when(pl.col(policy_column) == second_policy)
                .then(pl.lit(80.0))
                .otherwise(pl.col("exit_threshold_basis_bp"))
                .alias("exit_threshold_basis_bp"),
                pl.when(pl.col(policy_column) == second_policy)
                .then(pl.lit("20251231"))
                .otherwise(pl.col("exit_rule_source_asof_date"))
                .alias("exit_rule_source_asof_date"),
            )

        report = build_exit_maker_report(
            ExitMakerPartitionInputs(
                vary(inputs.policy_support, "entry_policy_generation_id"),
                inputs.candidate_aliases,
                inputs.raw_candidate_facts,
                vary(inputs.position_policy_facts, "entry_policy_generation_id"),
                inputs.action_facts,
                vary(inputs.taker_exit_facts, "policy_generation_id"),
                inputs.coverage,
                inputs.metadata,
            )
        )
        future = report.product_policy.filter(pl.col("exit_route") == FUTURE_EXIT)
        self.assertEqual(future.item(0, "exit_threshold_support_trials"), 2)
        self.assertEqual(future.item(0, "exit_threshold_unique_values"), 2)
        self.assertEqual(future.item(0, "exit_threshold_basis_bp_min"), 80.0)
        self.assertEqual(future.item(0, "exit_threshold_basis_bp_max"), 100.0)
        self.assertIsNone(future.item(0, "exit_threshold_basis_bp_exact"))
        self.assertTrue(future.item(0, "exit_threshold_varies_within_cell"))
        self.assertEqual(future.item(0, "exit_rule_source_unique_dates"), 2)
        self.assertIsNone(future.item(0, "exit_rule_source_asof_date_exact"))
        self.assertEqual(future.item(0, "frozen_exit_policy_signatures"), 2)
        self.assertEqual(future.item(0, "d_minus_one_lineage_valid_rate"), 1.0)
        lineage = report.policy_threshold_lineage.filter(
            pl.col("exit_route") == FUTURE_EXIT
        )
        self.assertEqual(set(lineage["exit_threshold_basis_bp"]), {80.0, 100.0})
        self.assertEqual(
            set(lineage["exit_rule_source_asof_date"]),
            {"20251231", "20260101"},
        )
        self.assertEqual(lineage["frozen_exit_policy_signature"].n_unique(), 2)

        mismatched_taker = vary(inputs.taker_exit_facts, "policy_generation_id")
        with self.assertRaisesRegex(ValueError, "threshold/source disagree"):
            build_exit_maker_report(
                ExitMakerPartitionInputs(
                    inputs.policy_support,
                    inputs.candidate_aliases,
                    inputs.raw_candidate_facts,
                    inputs.position_policy_facts,
                    inputs.action_facts,
                    mismatched_taker,
                    inputs.coverage,
                    inputs.metadata,
                )
            )

    def test_policy_specific_horizon_outcome_does_not_use_raw_representative(self) -> None:
        inputs = _inputs()
        first_policy = inputs.action_facts.item(0, "policy_generation_id")
        second_policy = inputs.action_facts.item(1, "policy_generation_id")
        center_trial = inputs.policy_support.filter(
            (pl.col("entry_policy_generation_id") == first_policy)
            & (pl.col("exit_route") == FUTURE_EXIT)
        ).item(0, "exit_policy_trial_id")
        lower_trial = inputs.policy_support.filter(
            (pl.col("entry_policy_generation_id") == second_policy)
            & (pl.col("exit_route") == FUTURE_EXIT)
        ).item(0, "exit_policy_trial_id")

        support = inputs.policy_support.with_columns(
            pl.when(pl.col("entry_policy_generation_id") == second_policy)
            .then(pl.lit("frozen_lower"))
            .otherwise(pl.col("exit_rule_id"))
            .alias("exit_rule_id")
        )
        positions = inputs.position_policy_facts.with_columns(
            pl.when(pl.col("entry_policy_generation_id") == second_policy)
            .then(pl.lit("frozen_lower"))
            .otherwise(pl.col("exit_rule_id"))
            .alias("exit_rule_id"),
            pl.when(pl.col("exit_policy_trial_id") == center_trial)
            .then(pl.lit("carry_at_eod_cancel_unconfirmed"))
            .otherwise(pl.col("nominal_instant_cancel_v0_branch"))
            .alias("nominal_instant_cancel_v0_branch"),
            pl.when(pl.col("exit_policy_trial_id") == center_trial)
            .then(pl.lit("carry_at_eod_cancel_unconfirmed"))
            .otherwise(pl.col("branch_status"))
            .alias("branch_status"),
            pl.when(pl.col("exit_policy_trial_id") == center_trial)
            .then(pl.lit(None).cast(pl.Int64))
            .otherwise(pl.col("exit_decision_time_ns"))
            .alias("exit_decision_time_ns"),
            pl.when(pl.col("exit_policy_trial_id") == center_trial)
            .then(pl.lit(None).cast(pl.Float64))
            .otherwise(pl.col("gross_cycle_pnl_twd"))
            .alias("gross_cycle_pnl_twd"),
            pl.when(pl.col("exit_policy_trial_id") == center_trial)
            .then(pl.lit(None).cast(pl.String))
            .otherwise(pl.col("exit_hedge_status"))
            .alias("exit_hedge_status"),
            *(
                pl.when(pl.col("exit_policy_trial_id") == center_trial)
                .then(pl.lit(None).cast(pl.Float64))
                .otherwise(pl.col(column))
                .alias(column)
                for column in (
                    "exit_hedge_signed_latency_slippage_bp",
                    "exit_hedge_signed_depth_slippage_bp",
                    "exit_hedge_signed_total_slippage_bp",
                )
            ),
        )
        aliases = inputs.candidate_aliases.with_columns(
            pl.when(pl.col("entry_policy_generation_id") == second_policy)
            .then(pl.lit("frozen_lower"))
            .otherwise(pl.col("exit_rule_id"))
            .alias("exit_rule_id"),
            pl.when(pl.col("exit_policy_trial_id") == center_trial)
            .then(pl.lit(False))
            .otherwise(pl.col("any_fill"))
            .alias("any_fill"),
            pl.when(pl.col("exit_policy_trial_id") == center_trial)
            .then(pl.lit(False))
            .otherwise(pl.col("full_fill"))
            .alias("full_fill"),
            pl.when(pl.col("exit_policy_trial_id") == center_trial)
            .then(pl.lit(None).cast(pl.String))
            .otherwise(pl.col("exit_hedge_status"))
            .alias("exit_hedge_status"),
            pl.when(pl.col("exit_policy_trial_id") == lower_trial)
            .then(pl.lit(False))
            .otherwise(pl.col("cancel_required"))
            .alias("cancel_required"),
        )
        taker = inputs.taker_exit_facts.with_columns(
            pl.when(pl.col("policy_generation_id") == second_policy)
            .then(pl.lit("frozen_lower"))
            .otherwise(pl.col("exit_rule_id"))
            .alias("exit_rule_id")
        )
        report = build_exit_maker_report(
            ExitMakerPartitionInputs(
                support,
                aliases,
                inputs.raw_candidate_facts,
                positions,
                inputs.action_facts,
                taker,
                inputs.coverage,
                inputs.metadata,
            )
        )
        future = report.product_policy.filter(pl.col("exit_route") == FUTURE_EXIT)
        center = future.filter(pl.col("exit_rule_id") == RULE)
        lower = future.filter(pl.col("exit_rule_id") == "frozen_lower")
        # The canonical raw representative follows the longer lower horizon
        # and says full_fill=True.  The shorter center policy stopped before
        # that fill and must retain its policy-specific no-fill result.
        self.assertEqual(
            center.item(0, "raw_full_fill_rate_given_fill_known"), 0.0
        )
        self.assertEqual(
            lower.item(0, "raw_full_fill_rate_given_fill_known"), 1.0
        )
        self.assertEqual(center.item(0, "raw_cancel_required_rate"), 1.0)
        self.assertEqual(lower.item(0, "raw_cancel_required_rate"), 0.0)

    def test_nullable_fill_label_is_unknown_not_zero(self) -> None:
        inputs = _inputs()
        aliases = inputs.candidate_aliases.with_columns(
            pl.lit(None).cast(pl.Boolean).alias("any_fill"),
            pl.lit(None).cast(pl.Boolean).alias("full_fill"),
            pl.lit(None).cast(pl.Boolean).alias("partial_fill"),
            pl.lit(None).cast(pl.String).alias("exit_hedge_status"),
        )
        report = build_exit_maker_report(
            ExitMakerPartitionInputs(
                inputs.policy_support,
                aliases,
                inputs.raw_candidate_facts,
                inputs.position_policy_facts,
                inputs.action_facts,
                inputs.taker_exit_facts,
                inputs.coverage,
                inputs.metadata,
            )
        )
        future = report.product_policy.filter(pl.col("exit_route") == FUTURE_EXIT)
        self.assertEqual(future.item(0, "raw_fill_known_candidates"), 0)
        self.assertEqual(future.item(0, "raw_fill_unknown_candidates"), 1)
        self.assertEqual(future.item(0, "raw_fill_unknown_rate"), 1.0)
        self.assertIsNone(future.item(0, "raw_full_fill_rate_given_fill_known"))
        self.assertEqual(
            future.item(0, "raw_full_fill_rate_all_candidates_lower_bound"),
            0.0,
        )

    def test_partition_lineage_hashes_coverage_and_atomic_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exit_root = root / "exit"
            entry_root = root / "entry"
            for date in ("20260102", "20260105"):
                _publish_synthetic_partition(exit_root, entry_root, date)
            session_calendar = root / "sessions.txt"
            session_calendar.write_text(
                "20260101\n20260102\n20260105\n", encoding="utf-8"
            )

            inputs = load_exit_maker_partition_inputs(
                exit_root,
                entry_root,
                sessions=2,
                require_balanced_product_days=True,
                session_calendar=("20260101", "20260102", "20260105"),
            )
            self.assertEqual(inputs.metadata["selected_session_count"], 2)
            self.assertEqual(inputs.coverage.height, 2)
            output = root / "report"
            report = run_exit_maker_report(
                exit_root,
                entry_root,
                sessions=2,
                require_balanced_product_days=True,
                output_dir=output,
                session_calendar_path=session_calendar,
            )
            self.assertEqual(report.product_policy.height, 2)
            marker = json.loads((output / "report_complete.json").read_text())
            self.assertTrue(marker["complete"])
            self.assertEqual(len(marker["artifacts"]), 8)
            self.assertTrue((output / "exit_policy_threshold_lineage.csv").is_file())
            with self.assertRaises(FileExistsError):
                run_exit_maker_report(
                    exit_root,
                    entry_root,
                    sessions=2,
                    output_dir=output,
                    session_calendar_path=session_calendar,
                )
            with self.assertRaisesRegex(ValueError, "requested 3 complete sessions"):
                load_exit_maker_partition_inputs(exit_root, entry_root, sessions=3)

            corrupt = (
                exit_root
                / "Date=20260105"
                / f"ValueCode={VALUE_CODE}"
                / "exit_maker_policy_support.parquet"
            )
            corrupt.write_bytes(corrupt.read_bytes() + b"corrupt")
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                load_exit_maker_partition_inputs(exit_root, entry_root, sessions=2)

    def test_cli_defaults_to_strict_60_session_hash_validation(self) -> None:
        parsed = _parse_args(
            ["--exit-root", "/tmp/exit", "--entry-root", "/tmp/entry"]
        )
        self.assertEqual(parsed.sessions, 60)
        self.assertFalse(parsed.allow_fewer_sessions)
        self.assertFalse(parsed.skip_hash_validation)
        self.assertEqual(parsed.assumed_non_price_cost_bp, 19.0)


if __name__ == "__main__":
    unittest.main()
