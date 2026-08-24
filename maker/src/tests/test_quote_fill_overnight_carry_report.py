from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import polars as pl

from maker.src.quote_fill.ev_surface import DEFAULT_COST_COLUMNS
from maker.src.quote_fill.overnight_carry_report import (
    INPUT_ARTIFACTS,
    REPORT_ARTIFACTS,
    OvernightCarryPartitionInputs,
    _canonical_sha256,
    _file_sha256,
    build_overnight_carry_report,
    load_overnight_carry_partition_inputs,
    run_overnight_carry_report,
)


VALUE_CODE = "2303"
QUOTE_CODE = "CDF1"
RUNNER = {
    "runner_version": "overnight_carry_product_day_v1",
    "max_carry_sessions": 1,
}
RUNNER_SHA = _canonical_sha256(RUNNER)


def _label(
    date: str,
    alias: int,
    dependency: str,
    *,
    gross_bp: float = 50.0,
) -> dict[str, object]:
    rule = f"rule-{alias}"
    route = "future_bid_spot_taker" if alias == 1 else "spot_ask_future_taker"
    row: dict[str, object] = {
        "carry_label_id": f"carry-{date}-{alias}",
        "Date": date,
        "ValueCode": VALUE_CODE,
        "QuoteCode": QUOTE_CODE,
        "entry_route": "future_ask_spot_taker",
        "entry_policy_generation_id": f"entry-policy-{date}-{alias}",
        "entry_raw_order_fact_id": f"entry-raw-{date}",
        "exit_rule_id": rule,
        "exit_route": route,
        "exit_policy_trial_id": f"exit-policy-{date}-{alias}",
        "source_branch_status": (
            "carry_at_eod_cancel_unconfirmed"
            if alias == 1
            else "carry_at_eod_no_admission"
        ),
        "position_established_ns": 1_000_000_000,
        "entry_spot_price": 100.0,
        "entry_future_price": 101.0,
        "contract_size_shares": 2_000,
        "physical_entry_dependency_id": dependency,
        "physical_entry_strict_carry_policy_alias_count": 2,
        "physical_entry_coverage_weight": 0.5,
        "policy_observation_weight": 1.0,
        "physical_entry_alias_nonindependent": True,
        "cross_q_rule_route_additive": False,
        "sampling_unit": "exit_policy_trial_alternative",
        "expiry_session": "20260131",
        "calendar_version": "calendar-v1",
        "carry_policy_version": "carry-v1",
        "max_carry_sessions": 1,
        "max_book_age_ns": None,
        "max_book_age_enforced": False,
        "added_exit_latency_ns": 0,
        "latency_matched_to_exit_maker": False,
        "exact_quote_code_required": True,
        "roll_attempted": False,
        "roll_policy_version": None,
        "new_quote_code": None,
        "label_status": "overnight_exit",
        "terminal_branch": "overnight_exit",
        "transition_branch": "same_exact_contract",
        "outcome_status": "known",
        "unresolved_reason": None,
        "exit_date": "20260105",
        "label_end_date": "20260105",
        "exit_decision_time_ns": 2_000_000_000,
        "exit_spot_snapshot_time_ns": 2_000_000_000,
        "exit_future_snapshot_time_ns": 2_000_000_000,
        "exit_spot_price": 101.0,
        "exit_future_price": 101.5,
        "exit_spot_levels_swept": 1,
        "exit_future_levels_swept": 1,
        "exit_spot_book_age_ms": 0.0,
        "exit_future_book_age_ms": 0.0,
        "settlement_price": None,
        "settlement_time_ns": None,
        "settlement_source_version": None,
        "gross_cycle_pnl_twd": gross_bp / 10_000.0 * 200_000.0,
        "normalization_notional_twd": 200_000.0,
        "filled_cashflow_before_cost_bp": gross_bp,
        "holding_seconds": 1.0,
        "terminal_cashflow_priced": True,
        "counterfactual_executable": True,
        "execution_benchmark_role": (
            "optimistic_zero_added_latency_same_cursor_taker_taker"
        ),
        "book_freshness_status": "ungated_age_recorded",
        "needs_next_session_label": False,
        "cost_profile_version": None,
        "fees_tax_included": False,
        "pathwise_ev_ready": False,
    }
    row.update({column: None for column in DEFAULT_COST_COLUMNS})
    return row


def _frames(date: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    dependency = f"physical-{date}"
    labels = pl.from_dicts(
        [_label(date, 1, dependency), _label(date, 2, dependency)],
        infer_schema_length=None,
    )
    audit = pl.DataFrame(
        {
            "position_policy_rows": [3],
            "strict_carry_policy_rows": [2],
            "excluded_noncarry_or_unknown_rows": [1],
            "label_rows": [2],
            "priced_terminal_rows": [2],
            "overnight_exit_rows": [2],
            "expiry_settlement_rows": [0],
            "expiry_settlement_unpriced_rows": [0],
            "roll_substitution_forbidden_rows": [0],
            "unresolved_rows": [0],
            "fees_tax_complete_rows": [0],
            "pathwise_ev_ready_rows": [0],
        }
    )
    return labels, audit


def _artifact(path: Path, frame: pl.DataFrame) -> dict[str, object]:
    frame.write_parquet(path)
    return {
        "rows": frame.height,
        "columns": frame.width,
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _publish_partition(root: Path, date: str) -> dict[str, object]:
    labels, audit = _frames(date)
    partition = root / f"Date={date}" / f"ValueCode={VALUE_CODE}"
    partition.mkdir(parents=True)
    artifacts = {
        INPUT_ARTIFACTS[0]: _artifact(partition / INPUT_ARTIFACTS[0], labels),
        INPUT_ARTIFACTS[1]: _artifact(partition / INPUT_ARTIFACTS[1], audit),
    }
    config = {
        "runner": RUNNER,
        "source": {"Date": date, "ValueCode": VALUE_CODE},
    }
    marker: dict[str, object] = {
        "complete": True,
        "Date": date,
        "ValueCode": VALUE_CODE,
        "runner_version": RUNNER["runner_version"],
        "runner_config_sha256": RUNNER_SHA,
        "config": config,
        "config_sha256": _canonical_sha256(config),
        "artifacts": artifacts,
        "fact_semantics": {
            "gross_only": True,
            "strict_carry_branches_only": True,
            "cancel_race_unknown_excluded": True,
            "exact_quote_code_no_roll": True,
            "first_missing_candidate_censors": True,
            "expiry_requires_final_settlement": True,
            "cost_columns_null": True,
            "pathwise_ev_ready": False,
        },
    }
    marker["marker_payload_sha256"] = _canonical_sha256(marker)
    (partition / "complete.json").write_text(
        json.dumps(marker, sort_keys=True), encoding="utf-8"
    )
    return marker


def _publish_manifest(root: Path, markers: list[dict[str, object]]) -> None:
    pl.from_dicts(
        [
            {
                "Date": marker["Date"],
                "ValueCode": marker["ValueCode"],
                "runner_config_sha256": marker["runner_config_sha256"],
                "config_sha256": marker["config_sha256"],
                "complete": True,
            }
            for marker in markers
        ],
        infer_schema_length=None,
    ).write_parquet(root / "overnight_carry_partition_manifest.parquet")


class OvernightCarryReportTest(unittest.TestCase):
    def test_load_deduplicate_gross_only_and_publish_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "overnight"
            markers = [
                _publish_partition(source, "20260102"),
                _publish_partition(source, "20260105"),
            ]
            _publish_manifest(source, markers)

            inputs = load_overnight_carry_partition_inputs(
                source,
                sessions=2,
                value_codes=[VALUE_CODE],
                require_balanced_product_days=True,
            )
            self.assertEqual(inputs.labels.height, 4)
            self.assertEqual(inputs.audits.height, 2)
            self.assertTrue(inputs.metadata["root_manifest_validated"])

            report = build_overnight_carry_report(inputs)
            self.assertEqual(report.physical_facts.height, 2)
            self.assertTrue(
                (report.physical_facts["policy_alias_rows"] == 2).all()
            )
            self.assertEqual(report.policy_summary.height, 2)
            self.assertEqual(report.status_summary.height, 1)
            self.assertEqual(
                report.status_summary.item(0, "unique_physical_entries"), 2
            )
            self.assertEqual(
                report.status_summary.item(0, "gross_before_cost_bp_p50"), 50.0
            )
            for frame in (
                report.combined_labels,
                report.combined_audits,
                report.physical_facts,
                report.policy_summary,
                report.status_summary,
            ):
                self.assertFalse(frame["pathwise_ev_ready"].any())
                self.assertFalse(frame["cross_q_rule_route_additive"].any())
                for column in DEFAULT_COST_COLUMNS:
                    self.assertEqual(frame[column].null_count(), frame.height)
                self.assertFalse(any("net" in name for name in frame.columns))

            output = root / "report"
            published = run_overnight_carry_report(
                source,
                output_dir=output,
                sessions=2,
                value_codes=[VALUE_CODE],
                require_balanced_product_days=True,
            )
            self.assertEqual(published.physical_facts.height, 2)
            marker = json.loads((output / "report_complete.json").read_text())
            self.assertTrue(marker["complete"])
            self.assertEqual(set(marker["artifacts"]), set(REPORT_ARTIFACTS))
            declared_sha = marker.pop("marker_payload_sha256")
            self.assertEqual(declared_sha, _canonical_sha256(marker))
            for name in REPORT_ARTIFACTS:
                self.assertTrue((output / name).is_file())
            resumed = run_overnight_carry_report(source, output_dir=output)
            self.assertEqual(resumed.physical_facts.height, 2)
            newer = _publish_partition(source, "20260106")
            _publish_manifest(source, [*markers, newer])
            with self.assertRaisesRegex(FileExistsError, "different partitions"):
                run_overnight_carry_report(source, output_dir=output)

    def test_rejects_artifact_corruption_and_root_manifest_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "overnight"
            marker = _publish_partition(source, "20260102")
            _publish_manifest(source, [marker])
            labels_path = (
                source
                / "Date=20260102"
                / f"ValueCode={VALUE_CODE}"
                / INPUT_ARTIFACTS[0]
            )
            labels_path.write_bytes(labels_path.read_bytes() + b"corrupt")
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                load_overnight_carry_partition_inputs(source)

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "overnight"
            marker = _publish_partition(source, "20260102")
            _publish_manifest(source, [marker])
            manifest_path = source / "overnight_carry_partition_manifest.parquet"
            manifest = pl.read_parquet(manifest_path).with_columns(
                pl.lit("wrong").alias("config_sha256")
            )
            manifest.write_parquet(manifest_path)
            with self.assertRaisesRegex(ValueError, "root manifest hash mismatch"):
                load_overnight_carry_partition_inputs(source)

    def test_rejects_cost_or_ev_ready_and_conflicting_physical_alias(self) -> None:
        labels, audit = _frames("20260102")
        coverage = pl.DataFrame(
            {
                "Date": ["20260102"],
                "ValueCode": [VALUE_CODE],
                "partition_complete": [True],
                "partition": ["synthetic"],
            }
        )
        inputs = OvernightCarryPartitionInputs(labels, audit, coverage, {})

        bad_cost = labels.with_columns(pl.lit(1.0).alias("fee_cost_bp"))
        with self.assertRaisesRegex(ValueError, "fee_cost_bp entirely null"):
            build_overnight_carry_report(
                OvernightCarryPartitionInputs(bad_cost, audit, coverage, {})
            )

        bad_ev = labels.with_columns(pl.lit(True).alias("pathwise_ev_ready"))
        with self.assertRaisesRegex(ValueError, "pathwise_ev_ready=false"):
            build_overnight_carry_report(
                OvernightCarryPartitionInputs(bad_ev, audit, coverage, {})
            )

        bad_strict = labels.with_columns(
            pl.lit("cancel_race_unknown").alias("source_branch_status"),
            pl.lit(False).alias("exact_quote_code_required"),
            pl.lit(True).alias("roll_attempted"),
            pl.lit("roll-v1").alias("roll_policy_version"),
            pl.lit("CDF2").alias("new_quote_code"),
        )
        with self.assertRaisesRegex(ValueError, "strict carry labels"):
            build_overnight_carry_report(
                OvernightCarryPartitionInputs(bad_strict, audit, coverage, {})
            )

        conflicting = labels.with_row_index("_row").with_columns(
            pl.when(pl.col("_row") == 1)
            .then(pl.lit(51.0))
            .otherwise(pl.col("filled_cashflow_before_cost_bp"))
            .alias("filled_cashflow_before_cost_bp"),
            pl.when(pl.col("_row") == 1)
            .then(pl.lit(1_020.0))
            .otherwise(pl.col("gross_cycle_pnl_twd"))
            .alias("gross_cycle_pnl_twd"),
        ).drop("_row")
        with self.assertRaisesRegex(ValueError, "disagree on physical outcome"):
            build_overnight_carry_report(
                OvernightCarryPartitionInputs(conflicting, audit, coverage, {})
            )

        self.assertEqual(build_overnight_carry_report(inputs).physical_facts.height, 1)


if __name__ == "__main__":
    unittest.main()
