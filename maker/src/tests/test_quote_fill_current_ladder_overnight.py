from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import polars as pl

from maker.src.quote_fill import current_ladder_overnight as subject
from maker.src.quote_fill.exit_maker_runner import EXIT_MAKER_RUNNER_VERSION


@dataclass
class _Fixture:
    root: Path
    entry_root: Path
    exit_root: Path
    prerequisite_root: Path
    data_root: Path
    futures_raw_root: Path
    prerequisite_payload: dict[str, object]
    contract: subject.CurrentLadderSourceContract


class CurrentLadderOvernightTest(unittest.TestCase):
    def test_manifest_row_partition_parity_canonicalizes_only_the_path(self) -> None:
        absolute = Path.cwd() / "maker"
        subject._verify_manifest_row_parity(
            {"partition": str(absolute), "complete": True},
            {"partition": "maker", "complete": True},
            {"partition": pl.String, "complete": pl.Boolean},
            source="fixture",
        )
        with self.assertRaisesRegex(ValueError, "differs from root manifest: partition"):
            subject._verify_manifest_row_parity(
                {"partition": str(absolute), "complete": True},
                {"partition": "maker/not-the-same", "complete": True},
                {"partition": pl.String, "complete": pl.Boolean},
                source="fixture",
            )

    def test_output_manifest_partition_parity_canonicalizes_only_the_path(self) -> None:
        absolute = Path.cwd() / "maker" / "data" / "fixture-partition"
        declared = {name: "same" for name in subject._OUTPUT_MANIFEST_SCHEMA}
        declared["partition"] = str(absolute)
        rebuilt = dict(declared)
        rebuilt["partition"] = "maker/data/fixture-partition"
        subject._verify_output_manifest_row_parity(
            rebuilt,
            declared,
            source="fixture",
        )
        rebuilt["partition"] = "maker/data/different-partition"
        with self.assertRaisesRegex(ValueError, "fixture/partition"):
            subject._verify_output_manifest_row_parity(
                rebuilt,
                declared,
                source="fixture",
            )
        rebuilt = dict(declared)
        rebuilt["complete"] = "different"
        with self.assertRaisesRegex(ValueError, "fixture/complete"):
            subject._verify_output_manifest_row_parity(
                rebuilt,
                declared,
                source="fixture",
            )

    def test_plan_uses_manifest_dates_not_tail_of_candidate_calendar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _fixture(Path(temporary))
            with patch.object(
                subject,
                "verify_cross_session_prerequisites",
                return_value=fixture.prerequisite_payload,
            ):
                common = {
                    "entry_root": fixture.entry_root,
                    "exit_root": fixture.exit_root,
                    "prerequisite_root": fixture.prerequisite_root,
                    "data_root": fixture.data_root,
                    "futures_raw_root": fixture.futures_raw_root,
                    "source_contract": fixture.contract,
                }
                plan = subject.build_current_ladder_overnight_plan(
                    **common,
                )

            self.assertEqual(plan.entry_dates, ("20260102", "20260105"))
            self.assertEqual(
                plan.candidate_sessions,
                ("20251231", "20260102", "20260105", "20260106"),
            )
            self.assertNotEqual(
                plan.entry_dates, plan.candidate_sessions[-len(plan.entry_dates) :]
            )
            self.assertEqual(
                dict(plan.next_session_by_entry_date),
                {"20260102": "20260105", "20260105": "20260106"},
            )
            self.assertEqual(plan.products, ("1101", "2301"))
            self.assertEqual(len(plan.product_days), 3)
            self.assertEqual(plan.universe.height, 4)
            missing = plan.universe.filter(~pl.col("selected_for_replay"))
            self.assertEqual(missing.select("Date", "ValueCode").row(0), ("20260102", "2301"))

    def test_entry_and_exit_key_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _fixture(
                Path(temporary),
                exit_keys=(
                    ("20260102", "1101"),
                    ("20260102", "2301"),
                    ("20260105", "2301"),
                ),
            )
            with patch.object(
                subject,
                "verify_cross_session_prerequisites",
                return_value=fixture.prerequisite_payload,
            ):
                with self.assertRaisesRegex(ValueError, "manifest keys differ"):
                    subject.build_current_ladder_overnight_plan(
                        entry_root=fixture.entry_root,
                        exit_root=fixture.exit_root,
                        prerequisite_root=fixture.prerequisite_root,
                        source_contract=fixture.contract,
                    )

    def test_source_file_hash_mutation_fails_before_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _fixture(Path(temporary))
            sessions = fixture.prerequisite_root / subject.SESSION_ARTIFACT_NAME
            sessions.write_text(sessions.read_text() + "20260107\n", encoding="utf-8")
            with patch.object(
                subject,
                "verify_cross_session_prerequisites",
                return_value=fixture.prerequisite_payload,
            ):
                with self.assertRaisesRegex(ValueError, "formal source hash mismatch"):
                    subject.build_current_ladder_overnight_plan(
                        entry_root=fixture.entry_root,
                        exit_root=fixture.exit_root,
                        prerequisite_root=fixture.prerequisite_root,
                        source_contract=fixture.contract,
                    )

    def test_frozen_upstream_marker_inventory_rejects_self_consistent_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _fixture(Path(temporary))
            marker = (
                fixture.entry_root
                / "Date=20260102"
                / "ValueCode=1101"
                / "complete.json"
            )
            payload = json.loads(marker.read_text(encoding="utf-8"))
            payload["post_manifest_replacement"] = True
            marker.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with patch.object(
                subject,
                "verify_cross_session_prerequisites",
                return_value=fixture.prerequisite_payload,
            ):
                with self.assertRaisesRegex(
                    ValueError, "formal upstream partition inventory mismatch"
                ):
                    subject.build_current_ladder_overnight_plan(
                        entry_root=fixture.entry_root,
                        exit_root=fixture.exit_root,
                        prerequisite_root=fixture.prerequisite_root,
                        source_contract=fixture.contract,
                    )

    def test_upstream_artifact_tamper_fails_before_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _fixture(Path(temporary))
            artifact = (
                fixture.entry_root
                / "Date=20260102"
                / "ValueCode=1101"
                / "execution_action_facts.parquet"
            )
            artifact.write_bytes(artifact.read_bytes() + b"tamper")
            with patch.object(
                subject,
                "verify_cross_session_prerequisites",
                return_value=fixture.prerequisite_payload,
            ):
                with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                    subject.build_current_ladder_overnight_plan(
                        entry_root=fixture.entry_root,
                        exit_root=fixture.exit_root,
                        prerequisite_root=fixture.prerequisite_root,
                        source_contract=fixture.contract,
                    )

    def test_upstream_extra_partition_fails_exact_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _fixture(Path(temporary))
            (
                fixture.entry_root
                / "Date=20260102"
                / "ValueCode=9999"
            ).mkdir()
            with patch.object(
                subject,
                "verify_cross_session_prerequisites",
                return_value=fixture.prerequisite_payload,
            ):
                with self.assertRaisesRegex(
                    ValueError, "entry execution partition inventory mismatch"
                ):
                    subject.build_current_ladder_overnight_plan(
                        entry_root=fixture.entry_root,
                        exit_root=fixture.exit_root,
                        prerequisite_root=fixture.prerequisite_root,
                        source_contract=fixture.contract,
                    )

    def test_prerequisite_entry_manifest_binding_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _fixture(Path(temporary))
            payload = json.loads(json.dumps(fixture.prerequisite_payload))
            assert isinstance(payload["config"], dict)
            payload["config"]["product_days_sha256"] = "f" * 64
            with patch.object(
                subject,
                "verify_cross_session_prerequisites",
                return_value=payload,
            ):
                with self.assertRaisesRegex(ValueError, "manifest hash binding"):
                    subject.build_current_ladder_overnight_plan(
                        entry_root=fixture.entry_root,
                        exit_root=fixture.exit_root,
                        prerequisite_root=fixture.prerequisite_root,
                        source_contract=fixture.contract,
                    )

    def test_runner_policy_is_non_overridable_d1_and_fresh_le_one_second(self) -> None:
        config = subject.current_ladder_runner_config()
        self.assertEqual(config.carry.max_carry_sessions, 1)
        self.assertEqual(config.carry.max_book_age_ns, 1_000_000_000)
        self.assertEqual(config.carry.added_exit_latency_ns, 0)
        self.assertEqual(config.carry.policy_version, subject.POLICY_VERSION)
        self.assertTrue(config.carry.calendar_required)

    def test_formal_raw_root_overrides_fail_before_source_scan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            alternate_data = Path(temporary) / "alternate-data"
            alternate_futures = Path(temporary) / "alternate-futures"
            alternate_data.mkdir()
            alternate_futures.mkdir()
            with self.assertRaisesRegex(ValueError, "formal data root override"):
                subject.build_current_ladder_overnight_plan(
                    data_root=alternate_data,
                )
            with self.assertRaisesRegex(ValueError, "formal futures root override"):
                subject.build_current_ladder_overnight_plan(
                    futures_raw_root=alternate_futures,
                )

    def test_public_verifier_accepts_a_tiny_exact_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _fixture(Path(temporary))
            output = fixture.root / "output"
            output.mkdir()
            with patch.object(
                subject,
                "verify_cross_session_prerequisites",
                return_value=fixture.prerequisite_payload,
            ):
                common = {
                    "entry_root": fixture.entry_root,
                    "exit_root": fixture.exit_root,
                    "prerequisite_root": fixture.prerequisite_root,
                    "data_root": fixture.data_root,
                    "futures_raw_root": fixture.futures_raw_root,
                    "source_contract": fixture.contract,
                }
                plan = subject.build_current_ladder_overnight_plan(
                    **common,
                )
                runner_payload = subject._expected_runner_payload(
                    plan, subject.current_ladder_runner_config()
                )
                runner_sha = subject._canonical_sha256(runner_payload)
                manifest = _output_tree(output, plan, runner_payload, runner_sha)
                for date, value_code in plan.product_days:
                    partition = output / f"Date={date}" / f"ValueCode={value_code}"
                    payload = json.loads((partition / "complete.json").read_text())
                    source = subject.verify_overnight_sources(
                        date,
                        value_code,
                        exit_maker_root=fixture.exit_root,
                        entry_execution_root=fixture.entry_root,
                        config=subject.current_ladder_runner_config(),
                    )
                    payload["config"]["source"] = source.payload(
                        subject.current_ladder_runner_config()
                    )
                    (partition / "complete.json").write_text(
                        json.dumps(payload, indent=2, sort_keys=True) + "\n"
                    )
                _report_tree(output, plan, runner_sha)
                _top_marker(output, plan, runner_sha)

                rows = {
                    (str(row["Date"]), str(row["ValueCode"])): row
                    for row in manifest.iter_rows(named=True)
                }

                def verify_partition(path: Path, **_: object) -> dict[str, object]:
                    rebuilt = dict(rows[
                        (
                            path.parent.name.removeprefix("Date="),
                            path.name.removeprefix("ValueCode="),
                        )
                    ])
                    rebuilt["partition"] = str(path)
                    return rebuilt

                with (
                    patch.object(
                        subject,
                        "verify_overnight_output_partition",
                        side_effect=verify_partition,
                    ),
                    patch.object(
                        subject, "run_overnight_carry_report"
                    ) as report_mock,
                ):
                    marker = subject.verify_current_ladder_overnight_bundle(
                        output,
                        **common,
                    )
                    relative_output = Path(os.path.relpath(output, Path.cwd()))
                    relative_marker = subject.verify_current_ladder_overnight_bundle(
                        relative_output,
                        **common,
                    )
                    self.assertTrue(relative_marker["complete"])
                    self.assertEqual(relative_marker, marker)
                    for report_call in report_mock.call_args_list[:2]:
                        self.assertEqual(report_call.args[0], output.resolve())
                        self.assertEqual(
                            report_call.kwargs["output_dir"],
                            (output / subject.REPORT_DIRECTORY_NAME).resolve(),
                        )
                    root_extra = output / "foreign-root.txt"
                    root_extra.write_text("foreign\n", encoding="utf-8")
                    with self.assertRaisesRegex(
                        ValueError, "overnight output root inventory mismatch"
                    ):
                        subject.verify_current_ladder_overnight_bundle(
                            output, **common
                        )
                    root_extra.unlink()
                    date_extra = output / "Date=20260102" / "foreign-date.txt"
                    date_extra.write_text("foreign\n", encoding="utf-8")
                    with self.assertRaisesRegex(
                        ValueError, "overnight output date inventory mismatch"
                    ):
                        subject.verify_current_ladder_overnight_bundle(
                            output, **common
                        )
                    date_extra.unlink()
                    partition_extra = (
                        output
                        / "Date=20260102"
                        / "ValueCode=1101"
                        / "foreign-partition.txt"
                    )
                    partition_extra.write_text("foreign\n", encoding="utf-8")
                    with self.assertRaisesRegex(
                        ValueError, "overnight output partition file inventory mismatch"
                    ):
                        subject.verify_current_ladder_overnight_bundle(
                            output, **common
                        )
                    partition_extra.unlink()
                    report_extra = (
                        output / subject.REPORT_DIRECTORY_NAME / "foreign-report.txt"
                    )
                    report_extra.write_text("foreign\n", encoding="utf-8")
                    with self.assertRaisesRegex(
                        ValueError, "overnight report file inventory mismatch"
                    ):
                        subject.verify_current_ladder_overnight_bundle(
                            output, **common
                        )
                    report_extra.unlink()
                    extra = output / "Date=20260102" / "ValueCode=9999"
                    extra.mkdir()
                    with self.assertRaisesRegex(
                        ValueError, "overnight output date inventory mismatch"
                    ):
                        subject.verify_current_ladder_overnight_bundle(
                            output, **common
                        )
                    extra.rmdir()
                    missing = output / "Date=20260102" / "ValueCode=1101"
                    hidden = missing.parent / "missing-fixture-partition"
                    missing.rename(hidden)
                    with self.assertRaisesRegex(
                        ValueError, "overnight output date inventory mismatch"
                    ):
                        subject.verify_current_ladder_overnight_bundle(
                            output, **common
                        )
                    hidden.rename(missing)
                    raw = (
                        fixture.data_root
                        / "tickData"
                        / "20260105_StockTick.parquet"
                    )
                    raw.write_bytes(raw.read_bytes() + b"drift")
                    with self.assertRaisesRegex(
                        ValueError, "partition raw source fingerprint drift"
                    ):
                        subject.verify_current_ladder_overnight_bundle(
                            output, **common
                        )
                self.assertEqual(
                    report_mock.call_args.kwargs["sessions"], len(plan.entry_dates)
                )
            self.assertTrue(marker["complete"])

    def test_public_verifier_rejects_missing_completion_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(FileNotFoundError, "completion marker"):
                subject.verify_current_ladder_overnight_bundle(Path(temporary))

    def test_public_verifier_rejects_output_root_symlink_before_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.mkdir()
            alias = root / "alias"
            alias.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                subject.verify_current_ladder_overnight_bundle(alias)

    def test_no_resume_fails_closed_when_bundle_marker_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / subject.COMPLETION_MARKER_NAME).write_text(
                "{}\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(FileExistsError, "resume is disabled"):
                subject.run_current_ladder_overnight(
                    output_root=output,
                    resume=False,
                )

    def test_output_must_be_disjoint_from_all_input_trees(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _fixture(Path(temporary))
            common = {
                "entry_root": fixture.entry_root,
                "exit_root": fixture.exit_root,
                "prerequisite_root": fixture.prerequisite_root,
                "data_root": fixture.data_root,
                "futures_raw_root": fixture.futures_raw_root,
                "source_contract": fixture.contract,
            }
            nested_outputs = (
                fixture.prerequisite_root / "forbidden-output",
                fixture.data_root / "forbidden-output",
                fixture.futures_raw_root / "forbidden-output",
            )
            for output in nested_outputs:
                with self.subTest(output=output):
                    with self.assertRaisesRegex(ValueError, "fully disjoint"):
                        subject.run_current_ladder_overnight(
                            output_root=output,
                            **common,
                        )
                    self.assertFalse(output.exists())
                    output.mkdir()
                    with self.assertRaisesRegex(ValueError, "fully disjoint"):
                        subject.verify_current_ladder_overnight_bundle(
                            output,
                            **common,
                        )
                    output.rmdir()
            with self.assertRaisesRegex(ValueError, "fully disjoint"):
                subject.run_current_ladder_overnight(
                    output_root=fixture.root,
                    **common,
                )

    def test_foreign_partial_output_fails_before_runner_and_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _fixture(Path(temporary))
            output = fixture.root / "poisoned-output"
            output.mkdir()
            foreign = output / "do-not-touch.txt"
            foreign.write_text("preserve-me\n", encoding="utf-8")
            before = foreign.read_bytes()
            with (
                patch.object(
                    subject,
                    "verify_cross_session_prerequisites",
                    return_value=fixture.prerequisite_payload,
                ),
                patch.object(subject, "run_overnight_carry_replay") as runner,
            ):
                with self.assertRaisesRegex(ValueError, "foreign root entries"):
                    subject.run_current_ladder_overnight(
                        entry_root=fixture.entry_root,
                        exit_root=fixture.exit_root,
                        prerequisite_root=fixture.prerequisite_root,
                        data_root=fixture.data_root,
                        futures_raw_root=fixture.futures_raw_root,
                        output_root=output,
                        source_contract=fixture.contract,
                    )
            runner.assert_not_called()
            self.assertEqual(foreign.read_bytes(), before)
            self.assertEqual({path.name for path in output.iterdir()}, {foreign.name})


def _fixture(
    root: Path,
    *,
    exit_keys: tuple[tuple[str, str], ...] | None = None,
) -> _Fixture:
    entry_root = root / "entry"
    exit_root = root / "exit"
    prerequisite_root = root / "prerequisite"
    data_root = root / "raw-data"
    futures_raw_root = root / "raw-futures"
    for path in (
        entry_root,
        exit_root,
        prerequisite_root,
        data_root,
        futures_raw_root,
    ):
        path.mkdir(parents=True)
    for date in ("20260105", "20260106"):
        spot = data_root / "tickData" / f"{date}_StockTick.parquet"
        future = (
            futures_raw_root
            / date[:4]
            / date[4:6]
            / date[6:8]
            / "stock_futures.parquet"
        )
        spot.parent.mkdir(parents=True, exist_ok=True)
        future.parent.mkdir(parents=True, exist_ok=True)
        spot.write_bytes(f"spot-{date}".encode())
        future.write_bytes(f"future-{date}".encode())
    entry_keys = (
        ("20260102", "1101"),
        ("20260105", "1101"),
        ("20260105", "2301"),
    )
    exit_keys = entry_keys if exit_keys is None else exit_keys
    entry_config_payload = {"fixture": "entry-current-ladder"}
    entry_config = subject._canonical_sha256(entry_config_payload)
    exit_runner_payload = {"fixture": "exit-current-ladder"}
    exit_runner = subject._canonical_sha256(exit_runner_payload)

    entry_rows = []
    for date, value_code in entry_keys:
        partition = entry_root / f"Date={date}" / f"ValueCode={value_code}"
        partition.mkdir(parents=True)
        artifacts = _fixture_artifacts(
            partition,
            tuple(
                f"{name.removesuffix('_rows')}.parquet"
                for name in subject._ENTRY_MANIFEST_SCHEMA
                if name.endswith("_rows")
            ),
        )
        (partition / "complete.json").write_text(
            json.dumps(
                {
                    "Date": date,
                    "ValueCode": value_code,
                    "complete": True,
                    "runner_version": "fixture-entry-runner",
                    "config": entry_config_payload,
                    "config_sha256": entry_config,
                    "artifacts": artifacts,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        entry_rows.append(
            {
                "Date": date,
                "ValueCode": value_code,
                "partition": str(partition),
                "config_sha256": entry_config,
                "complete": True,
                **{
                    name: 0
                    for name in subject._ENTRY_MANIFEST_SCHEMA
                    if name.endswith("_rows")
                },
            }
        )
    entry_manifest = pl.from_dicts(
        entry_rows, schema=subject._ENTRY_MANIFEST_SCHEMA, infer_schema_length=None
    ).sort(["Date", "ValueCode"])
    entry_manifest_path = entry_root / subject.ENTRY_MANIFEST_NAME
    entry_manifest.write_parquet(entry_manifest_path)

    exit_rows = []
    for date, value_code in exit_keys:
        partition = exit_root / f"Date={date}" / f"ValueCode={value_code}"
        partition.mkdir(parents=True)
        artifacts = _fixture_artifacts(
            partition,
            tuple(
                f"{name.removesuffix('_rows')}.parquet"
                for name in subject._EXIT_MANIFEST_SCHEMA
                if name.endswith("_rows")
            ),
        )
        exit_config_payload = {
            "runner": exit_runner_payload,
            "source": {
                "action_source": {"sha256": "4" * 64},
                "exit_rule_source": {
                    "kind": "fixture",
                    "sha256": "5" * 64,
                },
            },
        }
        exit_config = subject._canonical_sha256(exit_config_payload)
        (partition / "complete.json").write_text(
            json.dumps(
                {
                    "Date": date,
                    "ValueCode": value_code,
                    "complete": True,
                    "runner_version": EXIT_MAKER_RUNNER_VERSION,
                    "runner_config_sha256": exit_runner,
                    "config": exit_config_payload,
                    "config_sha256": exit_config,
                    "artifacts": artifacts,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        exit_rows.append(
            {
                "Date": date,
                "ValueCode": value_code,
                "partition": str(partition),
                "runner_config_sha256": exit_runner,
                "config_sha256": exit_config,
                "action_source_sha256": "4" * 64,
                "exit_rule_source_kind": "fixture",
                "exit_rule_source_sha256": "5" * 64,
                "complete": True,
                **{
                    name: 0
                    for name in subject._EXIT_MANIFEST_SCHEMA
                    if name.endswith("_rows")
                },
            }
        )
    exit_manifest = pl.from_dicts(
        exit_rows, schema=subject._EXIT_MANIFEST_SCHEMA, infer_schema_length=None
    ).sort(["Date", "ValueCode"])
    exit_manifest_path = exit_root / subject.EXIT_MANIFEST_NAME
    exit_manifest.write_parquet(exit_manifest_path)

    sessions_path = prerequisite_root / subject.SESSION_ARTIFACT_NAME
    sessions_path.write_text(
        "20251231\n20260102\n20260105\n20260106\n", encoding="utf-8"
    )
    calendar = pl.DataFrame(
        {
            "QuoteCode": ["DFFA6"],
            "expiry_session": ["20260121"],
            "calendar_version": ["fixture-current-ladder-v1"],
        },
        schema={
            "QuoteCode": pl.String,
            "expiry_session": pl.String,
            "calendar_version": pl.String,
        },
    )
    calendar_path = prerequisite_root / subject.CALENDAR_ARTIFACT_NAME
    calendar.write_parquet(calendar_path)
    entry_sha = subject._file_sha256(entry_manifest_path)
    sessions_sha = subject._file_sha256(sessions_path)
    calendar_sha = subject._file_sha256(calendar_path)
    prerequisite_payload: dict[str, object] = {
        "complete": True,
        "schema_version": "cross_session_prerequisites_v1",
        "config": {
            "entry_execution_root": str(entry_root),
            "product_days_path": str(entry_manifest_path),
            "product_days_sha256": entry_sha,
            "contract_calendar_sha256": calendar_sha,
        },
        "config_sha256": "6" * 64,
        "artifacts": {
            subject.SESSION_ARTIFACT_NAME: {
                "bytes": sessions_path.stat().st_size,
                "sha256": sessions_sha,
            },
            subject.CALENDAR_ARTIFACT_NAME: {
                "bytes": calendar_path.stat().st_size,
                "sha256": calendar_sha,
            },
        },
    }
    prerequisite_marker = prerequisite_root / subject.PREREQUISITE_MARKER_NAME
    prerequisite_marker.write_text(
        json.dumps(prerequisite_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if tuple(exit_keys) == entry_keys:
        inventory = subject._verify_upstream_partition_inventories(
            entry_manifest,
            exit_manifest,
            entry_root=entry_root,
            exit_root=exit_root,
            entry_config_sha256=entry_config,
            exit_runner_config_sha256=exit_runner,
            overnight_config=subject.current_ladder_runner_config(),
        )
    else:
        inventory = {
            "entry_partition_marker_inventory_sha256": "a" * 64,
            "exit_partition_marker_inventory_sha256": "b" * 64,
            "consumed_upstream_source_inventory_sha256": "c" * 64,
        }
    contract = subject.CurrentLadderSourceContract(
        entry_manifest_sha256=entry_sha,
        exit_manifest_sha256=subject._file_sha256(exit_manifest_path),
        prerequisite_marker_sha256=subject._file_sha256(prerequisite_marker),
        candidate_sessions_sha256=sessions_sha,
        contract_calendar_sha256=calendar_sha,
        entry_config_sha256=entry_config,
        exit_runner_config_sha256=exit_runner,
        entry_partition_marker_inventory_sha256=str(
            inventory["entry_partition_marker_inventory_sha256"]
        ),
        exit_partition_marker_inventory_sha256=str(
            inventory["exit_partition_marker_inventory_sha256"]
        ),
        consumed_upstream_source_inventory_sha256=str(
            inventory["consumed_upstream_source_inventory_sha256"]
        ),
        expected_entry_dates=2,
        expected_products=2,
        expected_available_product_days=3,
        expected_grid_product_days=4,
        expected_missing_product_days=1,
        expected_candidate_sessions=4,
    )
    return _Fixture(
        root=root,
        entry_root=entry_root,
        exit_root=exit_root,
        prerequisite_root=prerequisite_root,
        data_root=data_root,
        futures_raw_root=futures_raw_root,
        prerequisite_payload=prerequisite_payload,
        contract=contract,
    )


def _fixture_artifacts(
    partition: Path,
    names: tuple[str, ...],
) -> dict[str, dict[str, object]]:
    artifacts: dict[str, dict[str, object]] = {}
    for name in names:
        path = partition / name
        pl.DataFrame(schema={"fixture": pl.Int64}).write_parquet(path)
        artifacts[name] = {
            "rows": 0,
            "columns": 1,
            "bytes": path.stat().st_size,
            "sha256": subject._file_sha256(path),
        }
    return artifacts


def _output_tree(
    output: Path,
    plan: subject.CurrentLadderOvernightPlan,
    runner_payload: dict[str, object],
    runner_sha: str,
) -> pl.DataFrame:
    rows = []
    for date, value_code in plan.product_days:
        partition = output / f"Date={date}" / f"ValueCode={value_code}"
        partition.mkdir(parents=True)
        payload = {
            "config": {
                "runner": runner_payload,
                "candidate_sessions": [plan.next_session_by_entry_date[date]],
                "raw_source_fingerprints": subject._candidate_raw_fingerprints(
                    (plan.next_session_by_entry_date[date],),
                    data_root=plan.data_root,
                    futures_raw_root=plan.futures_raw_root,
                    custom_loader=False,
                ),
                "source": {},
            }
        }
        (partition / "complete.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for name in subject._OVERNIGHT_OUTPUT_ARTIFACTS:
            pl.DataFrame(schema={"fixture": pl.Int64}).write_parquet(
                partition / name
            )
        rows.append(
            {
                "Date": date,
                "ValueCode": value_code,
                "partition": str(partition),
                "runner_config_sha256": runner_sha,
                "config_sha256": "7" * 64,
                "overnight_carry_labels_rows": 0,
                "overnight_carry_audit_rows": 0,
                "complete": True,
            }
        )
    manifest = pl.from_dicts(
        rows, schema=subject._OUTPUT_MANIFEST_SCHEMA, infer_schema_length=None
    ).sort(["Date", "ValueCode"])
    manifest.write_parquet(output / subject.OVERNIGHT_CARRY_MANIFEST_NAME)
    plan.universe.write_parquet(output / subject.UNIVERSE_ARTIFACT_NAME)
    return manifest


def _report_tree(
    output: Path,
    plan: subject.CurrentLadderOvernightPlan,
    runner_sha: str,
) -> None:
    report = output / subject.REPORT_DIRECTORY_NAME
    report.mkdir()
    artifacts: dict[str, dict[str, object]] = {}
    for name in subject._carry_report.REPORT_ARTIFACTS:
        path = report / name
        pl.DataFrame(schema={"fixture": pl.Int64}).write_parquet(path)
        artifacts[name] = {
            "rows": 0,
            "columns": 1,
            "bytes": path.stat().st_size,
            "sha256": subject._file_sha256(path),
        }
    marker: dict[str, object] = {
        "complete": True,
        "report_version": subject._carry_report.REPORT_VERSION,
        "selected_session_count": len(plan.entry_dates),
        "selected_dates": list(plan.entry_dates),
        "selected_product_count": len(plan.products),
        "selected_product_day_count": len(plan.product_days),
        "expected_product_day_count": plan.universe.height,
        "missing_product_day_count": plan.universe.height - len(plan.product_days),
        "value_codes": list(plan.products),
        "runner_config_sha256": runner_sha,
        "gross_only": True,
        "pathwise_ev_ready": False,
        "artifacts": artifacts,
    }
    marker["marker_payload_sha256"] = subject._canonical_sha256(marker)
    (report / "report_complete.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _top_marker(
    output: Path,
    plan: subject.CurrentLadderOvernightPlan,
    runner_sha: str,
) -> None:
    marker = {
        "schema_version": subject.BUNDLE_SCHEMA_VERSION,
        "orchestrator_version": subject.ORCHESTRATOR_VERSION,
        "complete": True,
        "analysis_only": True,
        "gross_only": True,
        "pathwise_ev_ready": False,
        "joint_volume_allocated": False,
        "exact_contract_no_roll": True,
        "d1_only": True,
        "first_joint_taker_taker": True,
        "fresh_book_max_age_ns": subject.FRESH_BOOK_MAX_AGE_NS,
        "entry_dates_derived_from_manifests": True,
        "candidate_calendar_used_as_entry_selector": False,
        "source_binding": dict(plan.source_binding),
        "runner_config_sha256": runner_sha,
        "implementation_sources": subject._implementation_sources(),
        "artifacts": {
            subject.OVERNIGHT_CARRY_MANIFEST_NAME: subject._parquet_metadata(
                output / subject.OVERNIGHT_CARRY_MANIFEST_NAME
            ),
            subject.UNIVERSE_ARTIFACT_NAME: subject._parquet_metadata(
                output / subject.UNIVERSE_ARTIFACT_NAME
            ),
            f"{subject.REPORT_DIRECTORY_NAME}/report_complete.json": subject._file_metadata(
                output / subject.REPORT_DIRECTORY_NAME / "report_complete.json"
            ),
        },
    }
    (output / subject.COMPLETION_MARKER_NAME).write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    unittest.main()
