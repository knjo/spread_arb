from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from maker.src.quote_fill.s1_bundle_artifacts import (
    S1BundleArtifactError,
    read_s1_bundle_partition,
    write_s1_bundle_partition,
)


class S1BundleArtifactsTest(unittest.TestCase):
    def payload(self) -> dict[str, object]:
        return {
            "run_config_fingerprint": "a" * 64,
            "run_config_sha256": "b" * 64,
            "date": "20260825",
            "policy_id": "q95_lower",
            "lineage": {"previous_checkpoint_sha256": "genesis", "sequence": 0},
            "daily_summary": {
                "date": "20260825",
                "policy_id": "q95_lower",
                "entry_positions": 2,
            },
            "daily_risk_summary": {
                "date": "20260825",
                "policy_id": "q95_lower",
                "created": 3,
                "actual_send": 2,
            },
            "daily_diagnostics": {
                "date": "20260825",
                "policy_id": "q95_lower",
                "entry_latency_ns": [100, 200],
            },
            "accounting_fact_records": [
                {"fact_type": "execution", "position_id": "p1"},
                {"fact_type": "terminal", "position_id": "p1"},
            ],
            "capacity_transition_records": [
                {"transition": "reserve", "position_id": "p1", "quantity": 1}
            ],
            "economic_gate_estimate_records": [
                {
                    "schema_version": "s1_economic_gate_event_v2_frozen_exit",
                    "evaluation_stage": "decision_observation",
                    "status": "below_floor",
                },
                {
                    "schema_version": "s1_economic_gate_event_v2_frozen_exit",
                    "evaluation_stage": "actual_send_refresh",
                    "status": "eligible",
                },
            ],
            "economic_gate_event_records": [
                {
                    "schema_version": "s1_economic_gate_actual_send_audit_v1",
                    "dispatch_outcome": "sent",
                    "raw_order_fact_id": "order-1",
                    "estimate": {
                        "schema_version": "s1_economic_gate_event_v2_frozen_exit",
                        "evaluation_stage": "actual_send_refresh",
                        "status": "eligible",
                    },
                },
            ],
            "compact_checkpoint_record": {
                "accounting_fact_count": 2,
                "capacity_transition_count": 1,
            },
            "carry_records": [{"position_id": "p2", "quantity": 1}],
            "carry_binding_records": [
                {"position_id": "p2", "checkpoint_sha256": "c" * 64}
            ],
        }

    @staticmethod
    def expected(payload: dict[str, object]) -> dict[str, object]:
        return {
            "expected_run_config_fingerprint": payload["run_config_fingerprint"],
            "expected_run_config_sha256": payload["run_config_sha256"],
            "expected_date": payload["date"],
            "expected_policy_id": payload["policy_id"],
            "expected_lineage": payload["lineage"],
        }

    def test_roundtrip_is_atomic_lazy_canonical_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = self.payload()
            first = root / "first"
            second = root / "second"

            written = write_s1_bundle_partition(first, **payload)
            write_s1_bundle_partition(second, **self.payload())
            read = read_s1_bundle_partition(first, **self.expected(payload))

            self.assertEqual(read.date, "20260825")
            self.assertEqual(written.complete_marker, read.complete_marker)
            self.assertEqual(read.accounting_fact_records.record_count, 2)
            self.assertEqual(
                tuple(read.accounting_fact_records),
                tuple(payload["accounting_fact_records"]),
            )
            self.assertEqual(
                tuple(read.economic_gate_estimate_records),
                tuple(payload["economic_gate_estimate_records"]),
            )
            self.assertEqual(
                tuple(read.economic_gate_event_records),
                tuple(payload["economic_gate_event_records"]),
            )
            self.assertEqual(read.economic_gate_estimate_records.record_count, 2)
            estimate_metadata = read.complete_marker["artifacts"][
                "economic_gate_estimates.jsonl.gz"
            ]
            self.assertEqual(estimate_metadata["record_count"], 2)
            self.assertEqual(estimate_metadata["format"], "gzip_jsonl")
            self.assertEqual(
                (first / "accounting_facts.jsonl.gz").read_bytes(),
                (second / "accounting_facts.jsonl.gz").read_bytes(),
            )
            self.assertEqual(
                (first / "economic_gate_estimates.jsonl.gz").read_bytes(),
                (second / "economic_gate_estimates.jsonl.gz").read_bytes(),
            )
            self.assertEqual(
                (first / "capacity_transitions.jsonl.gz").read_bytes()[4:8],
                b"\0\0\0\0",
            )
            self.assertEqual(
                (first / "economic_gate_estimates.jsonl.gz").read_bytes()[4:8],
                b"\0\0\0\0",
            )
            self.assertEqual(
                (first / "daily_summary.json").read_bytes(),
                b'{"date":"20260825","entry_positions":2,"policy_id":"q95_lower"}\n',
            )
            with self.assertRaises(FileExistsError):
                write_s1_bundle_partition(first, **self.payload())

    def test_large_capacity_generator_is_streamed_and_verified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = self.payload()
            payload["capacity_transition_records"] = (
                {"position_id": f"p{index}", "transition": "reserve"}
                for index in range(5_000)
            )
            partition = Path(directory) / "partition"

            write_s1_bundle_partition(partition, **payload)
            read = read_s1_bundle_partition(partition, **self.expected(payload))

            self.assertEqual(read.capacity_transition_records.record_count, 5_000)
            self.assertEqual(sum(1 for _ in read.capacity_transition_records), 5_000)

    def test_rejects_wrong_binding_artifact_drift_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = self.payload()
            intact = root / "intact"
            write_s1_bundle_partition(intact, **payload)

            wrong_values = {
                "expected_policy_id": "another_policy",
                "expected_date": "20260826",
                "expected_run_config_fingerprint": "d" * 64,
                "expected_run_config_sha256": "e" * 64,
                "expected_lineage": {
                    "previous_checkpoint_sha256": "different",
                    "sequence": 0,
                },
            }
            for key, value in wrong_values.items():
                expected = self.expected(payload)
                expected[key] = value
                with (
                    self.subTest(key=key),
                    self.assertRaisesRegex(S1BundleArtifactError, "identity|lineage"),
                ):
                    read_s1_bundle_partition(intact, **expected)

            extra = root / "extra"
            write_s1_bundle_partition(extra, **self.payload())
            (extra / "unexpected.json").write_text("{}\n")
            with self.assertRaisesRegex(S1BundleArtifactError, "artifact set"):
                read_s1_bundle_partition(extra, **self.expected(payload))

            missing = root / "missing"
            write_s1_bundle_partition(missing, **self.payload())
            (missing / "economic_gate_estimates.jsonl.gz").unlink()
            with self.assertRaisesRegex(S1BundleArtifactError, "artifact set"):
                read_s1_bundle_partition(missing, **self.expected(payload))

            drift = root / "drift"
            write_s1_bundle_partition(drift, **self.payload())
            (drift / "daily_summary.json").write_text("{}\n")
            with self.assertRaisesRegex(S1BundleArtifactError, "content drift"):
                read_s1_bundle_partition(drift, **self.expected(payload))

            gzip_drift = root / "gzip-drift"
            write_s1_bundle_partition(gzip_drift, **self.payload())
            gzip_path = gzip_drift / "economic_gate_estimates.jsonl.gz"
            damaged = bytearray(gzip_path.read_bytes())
            damaged[-1] ^= 1
            gzip_path.write_bytes(damaged)
            with self.assertRaisesRegex(S1BundleArtifactError, "content drift"):
                read_s1_bundle_partition(gzip_drift, **self.expected(payload))

            symlinked = root / "symlinked"
            write_s1_bundle_partition(symlinked, **self.payload())
            target = root / "outside.json"
            target.write_text("[]\n")
            (symlinked / "carry.json").unlink()
            (symlinked / "carry.json").symlink_to(target)
            with self.assertRaisesRegex(S1BundleArtifactError, "symlink"):
                read_s1_bundle_partition(symlinked, **self.expected(payload))

    def test_invalid_payload_never_publishes_partition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            partition = Path(directory) / "partition"
            payload = self.payload()
            payload["daily_summary"] = {
                "date": "20260825",
                "policy_id": "q95_lower",
                "net": math.nan,
            }

            with self.assertRaises(S1BundleArtifactError):
                write_s1_bundle_partition(partition, **payload)
            self.assertFalse(partition.exists())

    def test_invalid_economic_gate_estimate_never_publishes_partition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            partition = Path(directory) / "partition"
            payload = self.payload()
            payload["economic_gate_estimate_records"] = [
                {
                    "schema_version": "s1_economic_gate_event_v2_frozen_exit",
                    "selected_expected_margin_bp": math.nan,
                }
            ]

            with self.assertRaisesRegex(
                S1BundleArtifactError, "economic_gate_estimate_records"
            ):
                write_s1_bundle_partition(partition, **payload)
            self.assertFalse(partition.exists())


if __name__ == "__main__":
    unittest.main()
