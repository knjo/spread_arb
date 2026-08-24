from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import polars as pl

from maker.src.quote_fill.execution_runner import (
    EXECUTION_RUNNER_VERSION,
    ExecutionRunnerConfig,
    WalkForwardExecutionDayBatch,
    _config_payload as _execution_config_payload,
)
from maker.src.quote_fill import exit_maker_allocator
from maker.src.quote_fill.exit_maker_runner import (
    EXIT_MAKER_ALLOCATOR_ENV_NAME,
    EXIT_MAKER_ALLOCATOR_ENV_VALUE,
    EXIT_MAKER_ALLOCATOR_RUNTIME_VALUE,
    EXIT_MAKER_MANIFEST_NAME,
    ExitMakerRunnerConfig,
    _canonical_sha256,
    _file_sha256,
    _runner_config_payload,
    _validate_allocator_runtime,
    discover_entry_product_days,
    run_exit_maker_replay,
)
from maker.src.quote_fill.raw_tape import RawTapeDay
from maker.src.quote_fill.targets import (
    FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE,
    PRICE_LADDER_VERSION,
)


DATE = "20260102"
PRODUCTS = ("2317", "2603")
QUOTES = ("DHFA6", "DKFA6")
QUANTILES = (50, 80, 95)
RESULT_NAMES = (
    "policy_support",
    "observations",
    "transitions",
    "candidate_aliases",
    "raw_candidate_facts",
    "position_policy_facts",
    "audit",
)


def _write_entry_partition(root: Path, value_code: str) -> Path:
    partition = root / f"Date={DATE}" / f"ValueCode={value_code}"
    partition.mkdir(parents=True)
    action = pl.DataFrame(
        {"Date": [DATE], "ValueCode": [value_code], "entry": [1]}
    )
    exits = pl.DataFrame(
        {
            "Date": [DATE],
            "ValueCode": [value_code],
            "exit_rule_id": ["frozen_center"],
        }
    )
    action_path = partition / "execution_action_facts.parquet"
    exit_path = partition / "exit_facts.parquet"
    action.write_parquet(action_path)
    exits.write_parquet(exit_path)
    upstream_config = _execution_config_payload(ExecutionRunnerConfig())
    artifacts = {
        action_path.name: _artifact_metadata(action_path, action),
        exit_path.name: _artifact_metadata(exit_path, exits),
    }
    marker = {
        "complete": True,
        "Date": DATE,
        "ValueCode": value_code,
        "runner_version": EXECUTION_RUNNER_VERSION,
        "config": upstream_config,
        "config_sha256": _canonical_sha256(upstream_config),
        "artifacts": artifacts,
        "fact_semantics": {
            "price_ladder_version": PRICE_LADDER_VERSION,
            "future_one_dollar_tick_effective_date": (
                FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE
            ),
        },
    }
    (partition / "complete.json").write_text(
        json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8"
    )
    return partition


def _artifact_metadata(path: Path, frame: pl.DataFrame) -> dict[str, object]:
    return {
        "rows": frame.height,
        "columns": frame.width,
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _batch(value_codes: tuple[str, ...]) -> WalkForwardExecutionDayBatch:
    quote_by_value = dict(zip(PRODUCTS, QUOTES, strict=True))
    mapping = pl.DataFrame(
        {
            "ValueCode": list(value_codes),
            "QuoteCode": [quote_by_value[value] for value in value_codes],
        }
    )
    states = mapping.with_columns(
        pl.lit(DATE).alias("Date"),
        pl.int_range(1, pl.len() + 1).alias("sequence"),
    )
    trades = pl.DataFrame(schema={"ValueCode": pl.String})
    audit = mapping.with_columns(pl.lit(DATE).alias("Date"))
    tape = RawTapeDay(
        date=DATE,
        mapping=mapping,
        spot_states=states,
        future_states=states,
        spot_trades=trades,
        future_trades=trades,
        audit=audit,
    )
    clock = states.select("Date", "ValueCode", "sequence").rename(
        {"sequence": "spread_pair_epoch"}
    )
    return WalkForwardExecutionDayBatch(
        date=DATE,
        value_codes=value_codes,
        sources=(),
        raw_tape=tape,
        spot_feature_state=clock,
        boundary_quantiles=QUANTILES,
    )


def _result(value_code: str) -> SimpleNamespace:
    frames = {
        name: pl.DataFrame(
            {"Date": [DATE], "ValueCode": [value_code], "artifact": [name]}
        )
        for name in RESULT_NAMES
    }
    return SimpleNamespace(
        **frames,
        frames=lambda: {
            f"exit_maker_{name}": frame for name, frame in frames.items()
        },
    )


@patch.dict(
    os.environ,
    {EXIT_MAKER_ALLOCATOR_ENV_NAME: EXIT_MAKER_ALLOCATOR_RUNTIME_VALUE},
)
@patch.object(
    exit_maker_allocator,
    "_RAW_EXIT_MAKER_ALLOCATOR_ENV_VALUE",
    EXIT_MAKER_ALLOCATOR_ENV_VALUE,
)
class ExitMakerRunnerTest(unittest.TestCase):
    def test_allocator_launch_missing_wrong_or_late_fails_before_output_or_raw(
        self,
    ) -> None:
        config = ExitMakerRunnerConfig(boundary_quantiles=QUANTILES)
        loader = Mock(side_effect=AssertionError("raw IO must not run"))
        captured_values = (None, "dirty_decay_ms:1000")
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "must-not-exist"
            for captured in captured_values:
                with self.subTest(captured=captured), patch.object(
                    exit_maker_allocator,
                    "_RAW_EXIT_MAKER_ALLOCATOR_ENV_VALUE",
                    captured,
                ), patch.dict(
                    os.environ,
                    {
                        EXIT_MAKER_ALLOCATOR_ENV_NAME: (
                            EXIT_MAKER_ALLOCATOR_RUNTIME_VALUE
                        )
                    },
                    clear=True,
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "raw process-launch allocator environment"
                    ):
                        run_exit_maker_replay(
                            ((DATE, PRODUCTS[0]),),
                            output_root=output_root,
                            config=config,
                            day_batch_loader=loader,
                        )
                self.assertFalse(output_root.exists())
                loader.assert_not_called()

    def test_allocator_runtime_missing_or_wrong_fails_before_output_or_raw(self) -> None:
        config = ExitMakerRunnerConfig(boundary_quantiles=QUANTILES)
        loader = Mock(side_effect=AssertionError("raw IO must not run"))
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "must-not-exist"
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(
                    RuntimeError, "process-start allocator environment"
                ):
                    run_exit_maker_replay(
                        ((DATE, PRODUCTS[0]),),
                        output_root=output_root,
                        config=config,
                        day_batch_loader=loader,
                    )
            self.assertFalse(output_root.exists())
            loader.assert_not_called()

            with patch.dict(
                os.environ,
                {EXIT_MAKER_ALLOCATOR_ENV_NAME: "dirty_decay_ms:1000"},
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "process-start allocator environment"
                ):
                    run_exit_maker_replay(
                        ((DATE, PRODUCTS[0]),),
                        output_root=output_root,
                        config=config,
                        day_batch_loader=loader,
                    )
            self.assertFalse(output_root.exists())
            loader.assert_not_called()

    def test_allocator_runtime_exact_value_is_accepted_and_hash_bound(self) -> None:
        config = ExitMakerRunnerConfig()
        _validate_allocator_runtime(config)
        payload = _runner_config_payload(config)
        self.assertEqual(
            payload["allocator_env_name"], EXIT_MAKER_ALLOCATOR_ENV_NAME
        )
        self.assertEqual(
            payload["allocator_env_value"], EXIT_MAKER_ALLOCATOR_ENV_VALUE
        )
        self.assertEqual(
            payload["allocator_runtime_value"],
            EXIT_MAKER_ALLOCATOR_RUNTIME_VALUE,
        )

    def test_default_artifact_replayer_publishes_atomic_marker_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            entry_root = temporary / "entry"
            output_root = temporary / "exit-maker"
            _write_entry_partition(entry_root, PRODUCTS[0])
            loader = Mock(side_effect=lambda _date, values, **_kwargs: _batch(values))

            def write_artifacts(
                actions,
                _exits,
                _tape,
                _clock,
                _config,
                *,
                artifact_directory,
            ):
                paths = {}
                for name, frame in _result(
                    str(actions.item(0, "ValueCode"))
                ).frames().items():
                    path = Path(artifact_directory) / f"{name}.parquet"
                    frame.write_parquet(path)
                    paths[name] = path
                return SimpleNamespace(artifact_paths=lambda: paths)

            with patch.object(
                ExitMakerRunnerConfig,
                "study_config",
                return_value=object(),
            ), patch(
                "maker.src.quote_fill.exit_maker_study."
                "replay_exit_maker_product_day_to_artifacts",
                side_effect=write_artifacts,
            ) as replayer:
                manifest = run_exit_maker_replay(
                    ((DATE, PRODUCTS[0]),),
                    entry_execution_root=entry_root,
                    output_root=output_root,
                    config=ExitMakerRunnerConfig(boundary_quantiles=QUANTILES),
                    day_batch_loader=loader,
                )

            self.assertEqual(manifest.height, 1)
            self.assertEqual(replayer.call_count, 1)
            partition = output_root / f"Date={DATE}" / f"ValueCode={PRODUCTS[0]}"
            expected_files = {
                "complete.json",
                *(f"exit_maker_{name}.parquet" for name in RESULT_NAMES),
            }
            self.assertEqual(
                {path.name for path in partition.iterdir()}, expected_files
            )
            marker = json.loads(
                (partition / "complete.json").read_text(encoding="utf-8")
            )
            for filename, metadata in marker["artifacts"].items():
                path = partition / filename
                frame = pl.read_parquet(path)
                self.assertEqual(metadata["rows"], frame.height)
                self.assertEqual(metadata["columns"], frame.width)
                self.assertEqual(metadata["bytes"], path.stat().st_size)
                self.assertEqual(metadata["sha256"], _file_sha256(path))
            self.assertEqual(
                marker["config"]["runner"]["allocator_env_name"],
                EXIT_MAKER_ALLOCATOR_ENV_NAME,
            )
            self.assertEqual(
                marker["config"]["runner"]["allocator_env_value"],
                EXIT_MAKER_ALLOCATOR_ENV_VALUE,
            )
            self.assertEqual(
                marker["config"]["runner"]["allocator_runtime_value"],
                EXIT_MAKER_ALLOCATOR_RUNTIME_VALUE,
            )
            self.assertEqual(
                list(partition.parent.glob(f".{partition.name}.tmp-*")), []
            )

    def test_default_artifact_replay_base_exception_cleans_atomic_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            entry_root = temporary / "entry"
            output_root = temporary / "exit-maker"
            _write_entry_partition(entry_root, PRODUCTS[0])
            loader = Mock(side_effect=lambda _date, values, **_kwargs: _batch(values))

            def fail_after_partial_write(
                _actions,
                _exits,
                _tape,
                _clock,
                _config,
                *,
                artifact_directory,
            ):
                chunks = Path(artifact_directory) / ".chunks"
                chunks.mkdir()
                pl.DataFrame({"partial": [1]}).write_parquet(
                    chunks / "partial.parquet"
                )
                raise KeyboardInterrupt("synthetic artifact replay failure")

            with patch.object(
                ExitMakerRunnerConfig,
                "study_config",
                return_value=object(),
            ), patch(
                "maker.src.quote_fill.exit_maker_study."
                "replay_exit_maker_product_day_to_artifacts",
                side_effect=fail_after_partial_write,
            ):
                with self.assertRaisesRegex(
                    KeyboardInterrupt, "synthetic artifact replay failure"
                ):
                    run_exit_maker_replay(
                        ((DATE, PRODUCTS[0]),),
                        entry_execution_root=entry_root,
                        output_root=output_root,
                        config=ExitMakerRunnerConfig(
                            boundary_quantiles=QUANTILES
                        ),
                        day_batch_loader=loader,
                    )
            date_root = output_root / f"Date={DATE}"
            self.assertFalse(
                (date_root / f"ValueCode={PRODUCTS[0]}").exists()
            )
            self.assertEqual(list(date_root.glob(".*.tmp-*")), [])

    def test_day_batch_atomic_publish_and_full_resume_skip_raw_io(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            entry_root = temporary / "entry"
            output_root = temporary / "exit-maker"
            for product in PRODUCTS:
                _write_entry_partition(entry_root, product)

            loader = Mock(side_effect=lambda _date, values, **_kwargs: _batch(values))
            replayer = Mock(
                side_effect=lambda actions, _exits, _tape, _clock, _config: _result(
                    str(actions.item(0, "ValueCode"))
                )
            )
            config = ExitMakerRunnerConfig(boundary_quantiles=QUANTILES)
            keys = tuple((DATE, product) for product in PRODUCTS)
            with patch.object(
                ExitMakerRunnerConfig,
                "study_config",
                return_value=object(),
            ):
                first = run_exit_maker_replay(
                    keys,
                    entry_execution_root=entry_root,
                    output_root=output_root,
                    config=config,
                    day_batch_loader=loader,
                    product_day_replayer=replayer,
                )
                resumed = run_exit_maker_replay(
                    keys,
                    entry_execution_root=entry_root,
                    output_root=output_root,
                    config=config,
                    day_batch_loader=loader,
                    product_day_replayer=replayer,
                )

            self.assertEqual(first.height, 2)
            self.assertEqual(resumed.height, 2)
            self.assertTrue(first["exit_rule_source_sha256"].is_not_null().all())
            self.assertEqual(loader.call_count, 1)
            self.assertEqual(replayer.call_count, 2)
            loaded_args = loader.call_args
            self.assertEqual(loaded_args.args[:2], (DATE, PRODUCTS))
            self.assertEqual(loaded_args.kwargs["quantiles"], QUANTILES)
            self.assertTrue((output_root / EXIT_MAKER_MANIFEST_NAME).is_file())
            expected_files = {
                "complete.json",
                *(f"exit_maker_{name}.parquet" for name in RESULT_NAMES),
            }
            for product in PRODUCTS:
                partition = output_root / f"Date={DATE}" / f"ValueCode={product}"
                self.assertEqual(
                    {path.name for path in partition.iterdir()}, expected_files
                )
                marker = json.loads(
                    (partition / "complete.json").read_text(encoding="utf-8")
                )
                source = marker["config"]["source"]
                self.assertEqual(
                    source["exit_rule_source"]["artifact"], "exit_facts.parquet"
                )
                self.assertEqual(
                    source["exit_rule_source"]["sha256"],
                    _file_sha256(
                        entry_root
                        / f"Date={DATE}"
                        / f"ValueCode={product}"
                        / "exit_facts.parquet"
                    ),
                )
                self.assertEqual(
                    marker["config"]["runner"]["allocation_semantics"],
                    "independent_candidates_no_joint_volume_allocation_v0",
                )
                self.assertEqual(
                    source["upstream_runner_version"],
                    EXECUTION_RUNNER_VERSION,
                )
                self.assertEqual(
                    marker["config"]["runner"]["upstream_entry_contract"][
                        "config_sha256"
                    ],
                    _canonical_sha256(
                        _execution_config_payload(ExecutionRunnerConfig())
                    ),
                )

    def test_changed_exit_rule_hash_invalidates_completed_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            entry_root = temporary / "entry"
            output_root = temporary / "exit-maker"
            partition = _write_entry_partition(entry_root, PRODUCTS[0])
            loader = Mock(side_effect=lambda _date, values, **_kwargs: _batch(values))
            replayer = Mock(return_value=_result(PRODUCTS[0]))
            config = ExitMakerRunnerConfig(boundary_quantiles=QUANTILES)
            with patch.object(
                ExitMakerRunnerConfig,
                "study_config",
                return_value=object(),
            ):
                run_exit_maker_replay(
                    ((DATE, PRODUCTS[0]),),
                    entry_execution_root=entry_root,
                    output_root=output_root,
                    config=config,
                    day_batch_loader=loader,
                    product_day_replayer=replayer,
                )

                exit_path = partition / "exit_facts.parquet"
                changed = pl.DataFrame(
                    {
                        "Date": [DATE],
                        "ValueCode": [PRODUCTS[0]],
                        "exit_rule_id": ["frozen_lower"],
                    }
                )
                changed.write_parquet(exit_path)
                marker_path = partition / "complete.json"
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                marker["artifacts"][exit_path.name] = _artifact_metadata(
                    exit_path, changed
                )
                marker_path.write_text(
                    json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8"
                )

                with self.assertRaisesRegex(ValueError, "partition config mismatch"):
                    run_exit_maker_replay(
                        ((DATE, PRODUCTS[0]),),
                        entry_execution_root=entry_root,
                        output_root=output_root,
                        config=config,
                        day_batch_loader=loader,
                        product_day_replayer=replayer,
                    )
            self.assertEqual(loader.call_count, 1)

    def test_replay_failure_leaves_no_partition_or_temp_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            entry_root = temporary / "entry"
            output_root = temporary / "exit-maker"
            _write_entry_partition(entry_root, PRODUCTS[0])
            loader = Mock(side_effect=lambda _date, values, **_kwargs: _batch(values))
            replayer = Mock(side_effect=RuntimeError("synthetic replay failure"))
            with patch.object(
                ExitMakerRunnerConfig,
                "study_config",
                return_value=object(),
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic replay failure"):
                    run_exit_maker_replay(
                        ((DATE, PRODUCTS[0]),),
                        entry_execution_root=entry_root,
                        output_root=output_root,
                        config=ExitMakerRunnerConfig(boundary_quantiles=QUANTILES),
                        day_batch_loader=loader,
                        product_day_replayer=replayer,
                    )
            date_root = output_root / f"Date={DATE}"
            self.assertFalse(
                (date_root / f"ValueCode={PRODUCTS[0]}").exists()
            )
            self.assertEqual(list(date_root.glob(".*.tmp-*")), [])

    def test_embedded_runner_semantics_cannot_forge_global_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            entry_root = temporary / "entry"
            output_root = temporary / "exit-maker"
            _write_entry_partition(entry_root, PRODUCTS[0])
            loader = Mock(side_effect=lambda _date, values, **_kwargs: _batch(values))
            config = ExitMakerRunnerConfig(boundary_quantiles=QUANTILES)
            with patch.object(
                ExitMakerRunnerConfig,
                "study_config",
                return_value=object(),
            ):
                run_exit_maker_replay(
                    ((DATE, PRODUCTS[0]),),
                    entry_execution_root=entry_root,
                    output_root=output_root,
                    config=config,
                    day_batch_loader=loader,
                    product_day_replayer=Mock(return_value=_result(PRODUCTS[0])),
                )

                marker_path = (
                    output_root
                    / f"Date={DATE}"
                    / f"ValueCode={PRODUCTS[0]}"
                    / "complete.json"
                )
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                marker["config"]["runner"]["allocation_semantics"] = "forged"
                marker["config_sha256"] = _canonical_sha256(marker["config"])
                marker_path.write_text(
                    json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "embedded runner config"):
                    run_exit_maker_replay(
                        ((DATE, PRODUCTS[0]),),
                        entry_execution_root=entry_root,
                        output_root=output_root,
                        config=config,
                        day_batch_loader=loader,
                        product_day_replayer=Mock(return_value=_result(PRODUCTS[0])),
                    )

    def test_grid_discovery_retains_missing_and_rejects_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "entry"
            _write_entry_partition(root, PRODUCTS[0])
            keys, audit = discover_entry_product_days(
                (DATE,), PRODUCTS, entry_execution_root=root
            )
            self.assertEqual(keys, ((DATE, PRODUCTS[0]),))
            self.assertEqual(
                audit.filter(~pl.col("available_entry_partition")).item(
                    0, "availability_status"
                ),
                "missing_entry_partition",
            )

            incomplete = root / f"Date={DATE}" / f"ValueCode={PRODUCTS[1]}"
            incomplete.mkdir(parents=True)
            with self.assertRaisesRegex(FileExistsError, "without complete"):
                discover_entry_product_days(
                    (DATE,), PRODUCTS, entry_execution_root=root
                )

    def test_stale_entry_runner_is_rejected_before_artifact_or_raw_io(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            entry_root = temporary / "entry"
            output_root = temporary / "exit-maker"
            partition = _write_entry_partition(entry_root, PRODUCTS[0])
            marker_path = partition / "complete.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            marker["runner_version"] = "product_day_execution_facts_v2_walkforward"
            marker["config"]["runner_version"] = (
                "product_day_execution_facts_v2_walkforward"
            )
            marker["config"].pop("price_ladder_version")
            marker["config"].pop("future_one_dollar_tick_effective_date")
            marker["config_sha256"] = _canonical_sha256(marker["config"])
            marker_path.write_text(
                json.dumps(marker, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            # Prove the v2 marker is rejected before this stale artifact is
            # opened and before the day-batch loader can touch raw tape.
            (partition / "execution_action_facts.parquet").write_bytes(b"stale")
            loader = Mock(side_effect=AssertionError("raw IO must not run"))
            with self.assertRaisesRegex(ValueError, "runner version is not current"):
                run_exit_maker_replay(
                    ((DATE, PRODUCTS[0]),),
                    entry_execution_root=entry_root,
                    output_root=output_root,
                    config=ExitMakerRunnerConfig(boundary_quantiles=QUANTILES),
                    day_batch_loader=loader,
                )
            loader.assert_not_called()

    def test_stale_entry_ladder_fields_are_rejected_before_raw_io(self) -> None:
        stale_values = {
            "price_ladder_version": "legacy-ladder",
            "future_one_dollar_tick_effective_date": "20990101",
        }
        for field, stale_value in stale_values.items():
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as directory:
                    temporary = Path(directory)
                    entry_root = temporary / "entry"
                    output_root = temporary / "exit-maker"
                    partition = _write_entry_partition(entry_root, PRODUCTS[0])
                    marker_path = partition / "complete.json"
                    marker = json.loads(
                        marker_path.read_text(encoding="utf-8")
                    )
                    marker["config"][field] = stale_value
                    marker["config_sha256"] = _canonical_sha256(
                        marker["config"]
                    )
                    marker_path.write_text(
                        json.dumps(marker, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    loader = Mock(
                        side_effect=AssertionError("raw IO must not run")
                    )
                    with self.assertRaisesRegex(
                        ValueError, "current v3 ladder contract"
                    ):
                        run_exit_maker_replay(
                            ((DATE, PRODUCTS[0]),),
                            entry_execution_root=entry_root,
                            output_root=output_root,
                            config=ExitMakerRunnerConfig(
                                boundary_quantiles=QUANTILES
                            ),
                            day_batch_loader=loader,
                        )
                    loader.assert_not_called()

    def test_hash_payload_binds_execution_semantics(self) -> None:
        base = ExitMakerRunnerConfig()
        payload = _runner_config_payload(base)
        self.assertEqual(payload["routes"], list(base.routes))
        self.assertEqual(
            set(payload["implementation_sources"]),
            {
                "execution_runner.py",
                "execution_facts.py",
                "exit_maker_allocator.py",
                "exit_maker_runner.py",
                "exit_maker.py",
                "exit_maker_study.py",
                "hedge.py",
                "targets.py",
            },
        )
        self.assertEqual(
            payload["upstream_entry_contract"]["runner_version"],
            EXECUTION_RUNNER_VERSION,
        )
        self.assertEqual(
            payload["upstream_entry_contract"]["price_ladder_version"],
            PRICE_LADDER_VERSION,
        )
        self.assertEqual(
            payload["upstream_entry_contract"][
                "future_one_dollar_tick_effective_date"
            ],
            FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE,
        )
        self.assertEqual(
            payload["upstream_entry_contract"]["targets_source_sha256"],
            payload["implementation_sources"]["targets.py"],
        )
        self.assertEqual(
            payload["upstream_entry_contract"][
                "exit_maker_runner_source_sha256"
            ],
            payload["implementation_sources"]["exit_maker_runner.py"],
        )
        for field in (
            "hedge_delay_ns",
            "exit_rule_source_kind",
            "exit_rule_source_artifact",
            "lifecycle_policy_version",
            "cancel_model",
            "oco_semantics",
            "allocation_semantics",
            "allocator_env_name",
            "allocator_env_value",
            "allocator_runtime_value",
        ):
            self.assertIn(field, payload)
        changed = replace(base, hedge_delay_ns=base.hedge_delay_ns + 1)
        self.assertNotEqual(
            _canonical_sha256(payload),
            _canonical_sha256(_runner_config_payload(changed)),
        )


if __name__ == "__main__":
    unittest.main()
