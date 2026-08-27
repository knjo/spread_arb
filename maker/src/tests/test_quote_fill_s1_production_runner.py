from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from maker.src.quote_fill.capacity_ledger import (
    CapacityLedger,
    CapacityLedgerCompactCheckpoint,
    CapacityTransition,
    decode_capacity_ledger_compact_checkpoint,
    decode_capacity_transition,
    encode_capacity_ledger_compact_checkpoint,
    encode_capacity_transition,
)
from maker.src.quote_fill.policy_spec import POLICY_IDS
from maker.src.quote_fill.s1_accounting_bridge import S1AccountingProduct
from maker.src.quote_fill.s1_bundle_artifacts import (
    read_s1_bundle_partition,
    write_s1_bundle_partition,
)
from maker.src.quote_fill.s1_capacity_identity_registry import (
    S1CapacityIdentityRegistry,
)
from maker.src.quote_fill.s1_production_runner import (
    CAPACITY_REGISTRY_FILENAME,
    GENESIS_PARTITION_SHA256,
    S1_DEVELOPMENT_DATES,
    S1ProductionConfig,
    S1ProductionRunError,
    _lineage_record,
    _load_resume_prefix,
    _partition_marker_sha256,
    _partition_path,
    _verify_resumed_capacity_partition,
    verify_s1_production_bundle,
)

_RUN_FINGERPRINT = "a" * 64
_RUN_CONFIG_SHA256 = "b" * 64
_INPUT_MANIFEST_SHA256 = "c" * 64


class S1ProductionRunnerIntegrationTest(unittest.TestCase):
    def test_capacity_receipts_round_trip_across_two_artifact_partitions(self) -> None:
        date_one, date_two = S1_DEVELOPMENT_DATES[:2]
        policy_id = POLICY_IDS[0]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = S1CapacityIdentityRegistry(root / CAPACITY_REGISTRY_FILENAME)
            ledger_one = CapacityLedger(
                global_cap_twd=20_000_000,
                product_cap_twd=10_000_000,
            )
            ledger_one.attempt_new_reservation(
                transition_id="reservation-1",
                timestamp_ns=1,
                capacity_id="capacity-1",
                product_id="product-1",
                requested_notional_twd=100,
            )
            receipt_one = registry.commit_partition(
                policy_id,
                date_one,
                ledger_one.transitions,
                None,
                verify_full_history_before_commit=False,
            )
            checkpoint_one = ledger_one.to_compact_checkpoint(
                through_date=date_one,
                identity_registry_receipt=receipt_one,
            )
            lineage_one = self._lineage(
                checkpoint=None,
                previous_policy_sha256=GENESIS_PARTITION_SHA256,
                previous_global_sha256=GENESIS_PARTITION_SHA256,
            )
            partition_one = _partition_path(root, date_one, policy_id)
            self._write_partition(
                partition_one,
                date=date_one,
                policy_id=policy_id,
                lineage=lineage_one,
                transitions=ledger_one.transitions,
                checkpoint=checkpoint_one,
            )
            marker_one = _partition_marker_sha256(partition_one)
            restored_one, transitions_one = self._read_partition(
                partition_one,
                date=date_one,
                policy_id=policy_id,
                lineage=lineage_one,
            )
            self.assertEqual(restored_one, checkpoint_one)
            self.assertEqual(restored_one.identity_registry_receipt, receipt_one)
            self.assertEqual(transitions_one, ledger_one.transitions)

            ledger_two = CapacityLedger.from_compact_checkpoint(checkpoint_one)
            ledger_two.record_entry_full_fill(
                transition_id="fill-1",
                timestamp_ns=2,
                capacity_id="capacity-1",
            )
            receipt_two = registry.commit_partition(
                policy_id,
                date_two,
                ledger_two.transitions,
                receipt_one,
                verify_full_history_before_commit=False,
            )
            checkpoint_two = ledger_two.to_compact_checkpoint(
                through_date=date_two,
                identity_registry_receipt=receipt_two,
            )
            lineage_two = self._lineage(
                checkpoint=checkpoint_one,
                previous_policy_sha256=marker_one,
                previous_global_sha256=marker_one,
            )
            partition_two = _partition_path(root, date_two, policy_id)
            self._write_partition(
                partition_two,
                date=date_two,
                policy_id=policy_id,
                lineage=lineage_two,
                transitions=ledger_two.transitions,
                checkpoint=checkpoint_two,
            )
            restored_two, transitions_two = self._read_partition(
                partition_two,
                date=date_two,
                policy_id=policy_id,
                lineage=lineage_two,
            )

            self.assertEqual(restored_two, checkpoint_two)
            self.assertEqual(transitions_two, ledger_two.transitions)
            self.assertEqual(registry.verify()[policy_id], receipt_two)
            self.assertEqual(
                lineage_two["previous_capacity_registry_receipt"],
                encode_capacity_ledger_compact_checkpoint(checkpoint_one)[
                    "identity_registry_receipt"
                ],
            )

            self.assertEqual(
                restored_two.identity_registry_receipt.transition_count,
                restored_two.transition_sequence_offset,
            )

    def test_resume_capacity_helper_accepts_a_valid_nonempty_partition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = S1CapacityIdentityRegistry(Path(temporary) / "registry.sqlite")
            ledger = CapacityLedger(
                global_cap_twd=20_000_000,
                product_cap_twd=10_000_000,
            )
            ledger.attempt_new_reservation(
                transition_id="reservation-1",
                timestamp_ns=1,
                capacity_id="capacity-1",
                product_id="product-1",
                requested_notional_twd=100,
            )
            ledger.release_working_leaves(
                transition_id="release-1",
                timestamp_ns=2,
                capacity_id="capacity-1",
                reason="actual_cancel",
            )
            receipt = registry.commit_partition(
                POLICY_IDS[0],
                S1_DEVELOPMENT_DATES[0],
                ledger.transitions,
                None,
            )
            checkpoint = ledger.to_compact_checkpoint(
                through_date=S1_DEVELOPMENT_DATES[0],
                identity_registry_receipt=receipt,
            )
            _verify_resumed_capacity_partition(
                prior=None,
                transitions=ledger.transitions,
                checkpoint=checkpoint,
            )

    def test_verify_requires_complete_marker_without_creating_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "missing-bundle"
            config = S1ProductionConfig(
                output_root=root,
                report_path=Path(temporary) / "missing-report.md",
            )

            with self.assertRaisesRegex(
                S1ProductionRunError,
                "bundle root must be an existing real directory",
            ):
                verify_s1_production_bundle(config)

            self.assertFalse(root.exists())

    def test_resume_rejects_noncanonical_date_major_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            later_partition = _partition_path(
                root,
                S1_DEVELOPMENT_DATES[0],
                POLICY_IDS[1],
            )
            later_partition.mkdir(parents=True)
            config = S1ProductionConfig(
                output_root=root,
                report_path=root / "report.md",
                source_commit="1" * 40,
            )
            registry = S1CapacityIdentityRegistry(root / CAPACITY_REGISTRY_FILENAME)

            with self.assertRaisesRegex(
                S1ProductionRunError,
                "not one canonical date-major prefix",
            ):
                _load_resume_prefix(
                    config,
                    run_config_fingerprint=_RUN_FINGERPRINT,
                    run_config_sha256=_RUN_CONFIG_SHA256,
                    accounting_products=(
                        S1AccountingProduct(
                            product_id="product-1",
                            value_code="value-1",
                            contract_size_shares=1,
                        ),
                    ),
                    registry=registry,
                )

    def _lineage(
        self,
        *,
        checkpoint: CapacityLedgerCompactCheckpoint | None,
        previous_policy_sha256: str,
        previous_global_sha256: str,
    ) -> dict[str, object]:
        state = SimpleNamespace(
            checkpoint=checkpoint,
            previous_partition_sha256=previous_policy_sha256,
            accounting_fact_count=0,
            carry=(),
        )
        return _lineage_record(
            date_input_manifest_sha256=_INPUT_MANIFEST_SHA256,
            previous_global_partition_sha256=previous_global_sha256,
            state=state,
        )

    def _write_partition(
        self,
        partition: Path,
        *,
        date: str,
        policy_id: str,
        lineage: dict[str, object],
        transitions: tuple[CapacityTransition, ...],
        checkpoint: CapacityLedgerCompactCheckpoint,
    ) -> None:
        bound_summary = {"date": date, "policy_id": policy_id}
        write_s1_bundle_partition(
            partition,
            run_config_fingerprint=_RUN_FINGERPRINT,
            run_config_sha256=_RUN_CONFIG_SHA256,
            date=date,
            policy_id=policy_id,
            lineage=lineage,
            daily_summary=bound_summary,
            daily_risk_summary=bound_summary,
            daily_diagnostics=bound_summary,
            accounting_fact_records=(),
            capacity_transition_records=(
                encode_capacity_transition(transition) for transition in transitions
            ),
            compact_checkpoint_record=encode_capacity_ledger_compact_checkpoint(
                checkpoint
            ),
            carry_records=(),
            carry_binding_records=(),
        )

    def _read_partition(
        self,
        partition: Path,
        *,
        date: str,
        policy_id: str,
        lineage: dict[str, object],
    ) -> tuple[CapacityLedgerCompactCheckpoint, tuple[CapacityTransition, ...]]:
        records = read_s1_bundle_partition(
            partition,
            expected_run_config_fingerprint=_RUN_FINGERPRINT,
            expected_run_config_sha256=_RUN_CONFIG_SHA256,
            expected_date=date,
            expected_policy_id=policy_id,
            expected_lineage=lineage,
        )
        transitions = tuple(
            decode_capacity_transition(record)
            for record in records.capacity_transition_records
        )
        checkpoint = decode_capacity_ledger_compact_checkpoint(
            records.compact_checkpoint_record
        )
        return checkpoint, transitions


if __name__ == "__main__":
    unittest.main()
