from __future__ import annotations

import gzip
import json
import subprocess
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import polars as pl

from ..quote_fill.s1_clock_differential import (
    CheckpointIdentity,
    S1ClockDifferentialError,
    S1ClockDifferentialMismatch,
    _bounded_prepared,
    _checkpoint_name,
    _full_1hz_prepared,
    _git_source_commit,
    _parse_args,
    _run_full_1hz_gate,
    _validate_cli_args,
    assert_exact_material_equal,
    load_material_checkpoint,
    write_material_checkpoint,
)


def _material() -> dict[str, object]:
    return {
        "sent_orders": [
            {
                "raw_order_fact_id": "raw-1",
                "product_id": "2330",
                "target_price": 100.5,
            }
        ],
        "executions": [
            {
                "execution_id": "execution-1",
                "price": 100.5,
                "quantity": 1000,
            }
        ],
        "positions": [{"position_id": "position-1", "state": "paired_open"}],
        "carry_in": [],
        "carry_out": [],
        "accounting_facts": [],
        "sent_economic_estimates": [{"expected_net_twd": 123.0}],
        "final_capacity_balances": {"global": {"active_paired": 1}},
    }


@dataclass(frozen=True)
class _PreparedStub:
    day_open_time_ns: int
    entry_cutoff_time_ns: int
    entry_product_ids: tuple[str, ...]
    state_changes_by_policy: dict[str, pl.DataFrame]


class S1ClockDifferentialTest(unittest.TestCase):
    def setUp(self) -> None:
        self.identity = CheckpointIdentity(
            date="20260505",
            policy_id="q95_C0_sd_f5",
            horizon_minutes=10,
            enabled_product_ids=("2330",),
            source_commit="a" * 40,
        )

    def test_checkpoint_is_deterministic_and_round_trips(self) -> None:
        material = _material()
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json.gz"
            second = Path(directory) / "second.json.gz"
            first_meta = write_material_checkpoint(
                first,
                label="A",
                role="generic_effective",
                identity=self.identity,
                material=material,
            )
            second_meta = write_material_checkpoint(
                second,
                label="A",
                role="generic_effective",
                identity=self.identity,
                material=material,
            )

            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(
                first_meta["document_sha256"], second_meta["document_sha256"]
            )
            self.assertEqual(
                load_material_checkpoint(
                    first,
                    expected_identity=self.identity,
                    expected_role="generic_effective",
                ),
                material,
            )

    def test_checkpoint_rejects_component_metadata_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "material.json.gz"
            write_material_checkpoint(
                path,
                label="A",
                role="generic_effective",
                identity=self.identity,
                material=_material(),
            )
            with gzip.open(path, "rt", encoding="utf-8") as source:
                document = json.load(source)
            document["components"]["executions"]["count"] = 99
            with gzip.open(path, "wt", encoding="utf-8") as target:
                json.dump(document, target)

            with self.assertRaisesRegex(S1ClockDifferentialError, "component metadata"):
                load_material_checkpoint(path, expected_identity=self.identity)

    def test_checkpoint_rejects_wrong_role_and_differential_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "material.json.gz"
            write_material_checkpoint(
                path,
                label="B",
                role="sparse_entry_route_derived",
                identity=self.identity,
                material=_material(),
            )
            with self.assertRaisesRegex(S1ClockDifferentialError, "role"):
                load_material_checkpoint(
                    path,
                    expected_identity=self.identity,
                    expected_role="generic_effective",
                )

            with gzip.open(path, "rt", encoding="utf-8") as source:
                document = json.load(source)
            document["differential_runner_version"] = "stale-runner"
            with gzip.open(path, "wt", encoding="utf-8") as target:
                json.dump(document, target)
            with self.assertRaisesRegex(
                S1ClockDifferentialError, "differential runner version"
            ):
                load_material_checkpoint(
                    path,
                    expected_identity=self.identity,
                    expected_role="sparse_entry_route_derived",
                )

    def test_checkpoint_rejects_different_source_commit_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "material.json.gz"
            write_material_checkpoint(
                path,
                label="A",
                role="generic_effective",
                identity=self.identity,
                material=_material(),
            )
            other_commit = CheckpointIdentity(
                date=self.identity.date,
                policy_id=self.identity.policy_id,
                horizon_minutes=self.identity.horizon_minutes,
                enabled_product_ids=self.identity.enabled_product_ids,
                source_commit="b" * 40,
            )
            with self.assertRaisesRegex(S1ClockDifferentialError, "identity"):
                load_material_checkpoint(
                    path,
                    expected_identity=other_commit,
                    expected_role="generic_effective",
                )

    def test_git_source_commit_rejects_dirty_source_and_expected_mismatch(self) -> None:
        dirty = subprocess.CompletedProcess(
            args=("git",),
            returncode=0,
            stdout=" M src/quote_fill/example.py\n",
            stderr="",
        )
        with (
            patch(
                "maker.src.quote_fill.s1_clock_differential.subprocess.run",
                return_value=dirty,
            ),
            self.assertRaisesRegex(S1ClockDifferentialError, "must be clean"),
        ):
            _git_source_commit()

        clean = subprocess.CompletedProcess(
            args=("git",), returncode=0, stdout="", stderr=""
        )
        head = subprocess.CompletedProcess(
            args=("git",), returncode=0, stdout=f"{'a' * 40}\n", stderr=""
        )
        with (
            patch(
                "maker.src.quote_fill.s1_clock_differential.subprocess.run",
                side_effect=(clean, head),
            ),
            self.assertRaisesRegex(S1ClockDifferentialError, "differs"),
        ):
            _git_source_commit("b" * 40)

    def test_exact_comparison_reports_component_hash_and_first_difference(self) -> None:
        left = _material()
        right = _material()
        right["executions"] = [
            {
                "execution_id": "execution-1",
                "price": 100.5,
                "quantity": 2000,
            }
        ]

        with self.assertRaises(S1ClockDifferentialMismatch) as raised:
            assert_exact_material_equal(left, right)

        report = raised.exception.report
        self.assertEqual(report["mismatched_components"], ["executions"])
        self.assertEqual(report["first_difference"]["path"], "$[0].quantity")
        execution = next(
            row
            for row in report["component_assertions"]
            if row["component"] == "executions"
        )
        self.assertEqual(execution["left_count"], 1)
        self.assertEqual(execution["right_count"], 1)
        self.assertNotEqual(execution["left_sha256"], execution["right_sha256"])

    def test_exact_comparison_accepts_identical_material(self) -> None:
        assertions = assert_exact_material_equal(_material(), _material())
        self.assertTrue(assertions)
        self.assertTrue(all(row["exact"] is True for row in assertions))

    def test_bounded_prepared_filters_policy_rows_and_moves_cutoff(self) -> None:
        minute = 60 * 1_000_000_000
        open_ns = 1_000_000_000_000
        source = pl.DataFrame(
            {
                "ValueCode": ["2330", "2330", "2317", "2317"],
                "decision_time_ns": [
                    open_ns + 5 * minute,
                    open_ns + 20 * minute,
                    open_ns + 5 * minute,
                    open_ns + 20 * minute,
                ],
            }
        )
        prepared = _PreparedStub(
            day_open_time_ns=open_ns,
            entry_cutoff_time_ns=open_ns + 4 * 60 * minute,
            entry_product_ids=("2317", "2330"),
            state_changes_by_policy={"policy": source},
        )

        bounded = _bounded_prepared(prepared, "policy", 10)  # type: ignore[arg-type]

        self.assertEqual(bounded.state_changes_by_policy["policy"].height, 2)
        self.assertEqual(
            bounded.entry_cutoff_time_ns,
            open_ns + (5 + 10) * minute,
        )

    def test_checkpoint_name_records_scope_and_horizon(self) -> None:
        name = _checkpoint_name(self.identity, "generic_effective")
        self.assertEqual(
            name,
            "s1_clock_diff_20260505_q95_C0_sd_f5_10m_2330_generic_effective.json.gz",
        )

    def test_full_1hz_builder_uses_true_non_sparse_policy_state(self) -> None:
        causal = pl.DataFrame({"unused": [1]})
        common = pl.DataFrame({"common": [1, 2]})
        full_state = pl.DataFrame(
            {
                "ValueCode": ["2317", "2330"],
                "decision_time_ns": [1, 1],
            }
        )
        scanner = MagicMock()
        scanner.join.return_value.select.return_value.collect.return_value = causal
        prepared = SimpleNamespace(
            specs_by_policy={
                "policy": (
                    SimpleNamespace(Date="20260505", ValueCode="2317", QuoteCode="A"),
                    SimpleNamespace(Date="20260505", ValueCode="2330", QuoteCode="B"),
                )
            },
            source_paths={"causal_fair": Path("causal.parquet")},
            common_decisions=pl.DataFrame({"row": [1, 2]}),
            state_changes_by_policy={"policy": pl.DataFrame()},
        )
        with (
            patch(
                "maker.src.quote_fill.s1_clock_differential.pl.scan_parquet",
                return_value=scanner,
            ),
            patch(
                "maker.src.quote_fill.s1_clock_differential.materialize_s1_common_day",
                return_value=common,
            ) as materialize,
            patch(
                "maker.src.quote_fill.s1_clock_differential.build_s1_policy_day_state",
                return_value=full_state,
            ) as build,
            patch(
                "maker.src.quote_fill.s1_clock_differential.replace",
                side_effect=lambda value, **changes: SimpleNamespace(
                    **{**vars(value), **changes}
                ),
            ),
        ):
            rebuilt = _full_1hz_prepared(prepared, "policy")  # type: ignore[arg-type]

        materialize.assert_called_once_with(causal)
        self.assertFalse(build.call_args.kwargs["_sparse_only"])
        self.assertIs(rebuilt.state_changes_by_policy["policy"], full_state)

    def test_full_1hz_cli_rejects_bounded_or_generic_resume(self) -> None:
        bounded = _parse_args(
            [
                "--date",
                "20260505",
                "--policy-id",
                "policy",
                "--horizon-minutes",
                "10",
                "--compare-full-1hz",
            ]
        )
        with self.assertRaisesRegex(ValueError, "full-day"):
            _validate_cli_args(bounded)

        generic_resume = _parse_args(
            [
                "--date",
                "20260505",
                "--policy-id",
                "policy",
                "--compare-full-1hz",
                "--resume-generic",
                "generic.json.gz",
            ]
        )
        with self.assertRaisesRegex(ValueError, "A/B"):
            _validate_cli_args(generic_resume)

    def test_full_1hz_gate_checkpoints_sparse_and_full_then_compares(self) -> None:
        material = _material()
        prepared = SimpleNamespace(name="sparse")
        full_prepared = SimpleNamespace(name="full")
        with tempfile.TemporaryDirectory() as directory:
            arguments = SimpleNamespace(
                policy_id=self.identity.policy_id,
                checkpoint_dir=Path(directory),
                resume_sparse_derived=None,
            )
            with (
                patch(
                    "maker.src.quote_fill.s1_clock_differential._run_material",
                    side_effect=[
                        (material, {"clock_mode": "entry_route_derived"}),
                        (material, {"clock_mode": "entry_route_derived"}),
                    ],
                ) as run_material,
                patch(
                    "maker.src.quote_fill.s1_clock_differential._full_1hz_prepared",
                    return_value=full_prepared,
                ) as build_full,
                patch("maker.src.quote_fill.s1_clock_differential._emit"),
            ):
                status = _run_full_1hz_gate(
                    arguments,  # type: ignore[arg-type]
                    prepared,  # type: ignore[arg-type]
                    self.identity,
                    frozenset(("2330",)),
                )

            self.assertEqual(status, 0)
            build_full.assert_called_once_with(prepared, self.identity.policy_id)
            self.assertEqual(run_material.call_count, 2)
            self.assertTrue(
                (
                    Path(directory)
                    / _checkpoint_name(self.identity, "entry_route_derived")
                ).is_file()
            )
            self.assertTrue(
                (
                    Path(directory)
                    / _checkpoint_name(self.identity, "full_1hz_entry_route_derived")
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
