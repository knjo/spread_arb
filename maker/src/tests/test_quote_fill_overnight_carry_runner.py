from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import polars as pl

from maker.src.quote_fill.overnight_carry import OvernightCarryConfig
from maker.src.quote_fill.overnight_carry_runner import (
    OVERNIGHT_CARRY_MANIFEST_NAME,
    OvernightCarryRunnerConfig,
    _canonical_sha256,
    _file_sha256,
    discover_overnight_product_days,
    run_overnight_carry_replay,
)
from maker.src.quote_fill.raw_tape import RawTapeDay


DATE = "20260601"
NEXT = "20260602"
LATER = "20260603"
VALUE = "2303"
QUOTE = "CCFF6"


def _position_facts() -> pl.DataFrame:
    base = {
        "Date": DATE,
        "ValueCode": VALUE,
        "QuoteCode": QUOTE,
        "entry_route": "future_ask_spot_taker",
        "entry_policy_generation_id": "entry-policy-1",
        "entry_raw_order_fact_id": "entry-raw-1",
        "exit_rule_id": "frozen_center",
        "exit_route": "future_bid_spot_taker",
        "position_status": "position_established",
        "position_established_ns": 1_000_000_000,
        "needs_next_session_label": True,
    }
    carry = {
        **base,
        "exit_policy_trial_id": "exit-policy-carry",
        "branch_status": "carry_at_eod_cancel_unconfirmed",
    }
    unknown = {
        **base,
        "exit_policy_trial_id": "exit-policy-unknown",
        "branch_status": "cancel_race_unknown",
    }
    return pl.from_dicts([carry, unknown], infer_schema_length=None)


def _actions() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Date": [DATE],
            "ValueCode": [VALUE],
            "QuoteCode": [QUOTE],
            "route": ["future_ask_spot_taker"],
            "policy_generation_id": ["entry-policy-1"],
            "raw_order_fact_id": ["entry-raw-1"],
            "full_fill": [True],
            "entry_hedge_status": ["executable"],
            "entry_hedge_contract_size_shares": [2_000],
            "entry_spot_price": [100.0],
            "entry_future_price": [101.0],
        }
    )


def _write_source_partition(
    root: Path, artifact_name: str, frame: pl.DataFrame, runner: str
) -> Path:
    partition = root / f"Date={DATE}" / f"ValueCode={VALUE}"
    partition.mkdir(parents=True)
    artifact = partition / artifact_name
    frame.write_parquet(artifact)
    config = {"runner": runner}
    marker = {
        "complete": True,
        "Date": DATE,
        "ValueCode": VALUE,
        "runner_version": runner,
        "config": config,
        "config_sha256": _canonical_sha256(config),
        "artifacts": {
            artifact_name: {
                "rows": frame.height,
                "columns": frame.width,
                "bytes": artifact.stat().st_size,
                "sha256": _file_sha256(artifact),
            }
        },
    }
    (partition / "complete.json").write_text(
        json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8"
    )
    return partition


def _state(
    market: str,
    *,
    recv: int,
    sequence: int,
    bid1: float,
    bid1_lots: int,
    bid2: float,
    bid2_lots: int,
    ask1: float,
    ask1_lots: int,
) -> pl.DataFrame:
    row: dict[str, object] = {
        "Date": NEXT,
        "market": market,
        "ValueCode": VALUE,
        "QuoteCode": QUOTE,
        "recv_time_ns": recv,
        "sequence": sequence,
        "trial_match": False,
        "raw_has_book": True,
        "book_state_available": True,
        "book_recv_time_ns": recv,
    }
    for side in ("bid", "ask"):
        for level in range(1, 6):
            row[f"{side}_price_{level}"] = None
            row[f"{side}_lots_{level}"] = None
    row.update(
        {
            "bid_price_1": bid1,
            "bid_lots_1": bid1_lots,
            "bid_price_2": bid2,
            "bid_lots_2": bid2_lots,
            "ask_price_1": ask1,
            "ask_lots_1": ask1_lots,
        }
    )
    return pl.from_dicts([row], infer_schema_length=None)


def _tape(date: str, mapping: pl.DataFrame) -> RawTapeDay:
    spot = _state(
        "spot",
        recv=10_000_000_000,
        sequence=2,
        bid1=100.5,
        bid1_lots=1,
        bid2=100.0,
        bid2_lots=2,
        ask1=101.0,
        ask1_lots=10,
    ).with_columns(pl.lit(date).alias("Date"))
    future = _state(
        "future",
        recv=10_000_000_000,
        sequence=1,
        bid1=100.0,
        bid1_lots=5,
        bid2=99.5,
        bid2_lots=5,
        ask1=100.5,
        ask1_lots=4,
    ).with_columns(pl.lit(date).alias("Date"))
    empty = pl.DataFrame(schema={"ValueCode": pl.String})
    audit = mapping.select("ValueCode", "QuoteCode").with_columns(
        pl.lit(date).alias("Date")
    )
    return RawTapeDay(date, mapping, spot, future, empty, empty, audit)


def _calendar() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "QuoteCode": [QUOTE],
            "expiry_session": ["20260617"],
            "calendar_version": ["official-test-v1"],
        }
    )


class OvernightCarryRunnerTests(unittest.TestCase):
    def _roots(self, temporary: Path) -> tuple[Path, Path, Path]:
        exit_root = temporary / "exit"
        entry_root = temporary / "entry"
        output_root = temporary / "overnight"
        _write_source_partition(
            exit_root,
            "exit_maker_position_policy_facts.parquet",
            _position_facts(),
            "exit-test-v1",
        )
        _write_source_partition(
            entry_root,
            "execution_action_facts.parquet",
            _actions(),
            "entry-test-v1",
        )
        return exit_root, entry_root, output_root

    def test_atomic_publish_strict_only_and_resume_skips_raw_io(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            exit_root, entry_root, output_root = self._roots(Path(directory))
            loader = Mock(side_effect=lambda date, mapping: _tape(date, mapping))
            kwargs = {
                "sessions": [DATE, NEXT],
                "contract_calendar": _calendar(),
                "exit_maker_root": exit_root,
                "entry_execution_root": entry_root,
                "output_root": output_root,
                "raw_tape_loader": loader,
            }
            first = run_overnight_carry_replay(((DATE, VALUE),), **kwargs)
            resumed = run_overnight_carry_replay(((DATE, VALUE),), **kwargs)

            self.assertEqual(first.height, 1)
            self.assertEqual(resumed.height, 1)
            self.assertEqual(loader.call_count, 1)
            self.assertTrue((output_root / OVERNIGHT_CARRY_MANIFEST_NAME).is_file())
            partition = output_root / f"Date={DATE}" / f"ValueCode={VALUE}"
            self.assertEqual(
                {path.name for path in partition.iterdir()},
                {
                    "complete.json",
                    "overnight_carry_labels.parquet",
                    "overnight_carry_audit.parquet",
                },
            )
            labels = pl.read_parquet(partition / "overnight_carry_labels.parquet")
            self.assertEqual(labels.height, 1)
            row = labels.row(0, named=True)
            self.assertEqual(row["exit_policy_trial_id"], "exit-policy-carry")
            self.assertEqual(row["label_status"], "overnight_exit")
            self.assertAlmostEqual(row["filled_cashflow_before_cost_bp"], 75.0)
            self.assertIsNone(row["fee_cost_bp"])
            self.assertFalse(row["pathwise_ev_ready"])
            audit = pl.read_parquet(partition / "overnight_carry_audit.parquet")
            self.assertEqual(audit.item(0, "excluded_noncarry_or_unknown_rows"), 1)
            marker = json.loads(
                (partition / "complete.json").read_text(encoding="utf-8")
            )
            self.assertTrue(marker["fact_semantics"]["exact_quote_code_no_roll"])
            self.assertFalse(marker["fact_semantics"]["pathwise_ev_ready"])

    def test_zero_positions_accept_minimal_empty_actions_without_raw_io(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            exit_root = temporary / "exit"
            entry_root = temporary / "entry"
            output_root = temporary / "overnight"
            _write_source_partition(
                exit_root,
                "exit_maker_position_policy_facts.parquet",
                _position_facts().head(0),
                "exit-test-v1",
            )
            minimal_actions = pl.DataFrame(
                schema={
                    "Date": pl.String,
                    "ValueCode": pl.String,
                    "QuoteCode": pl.String,
                    "route": pl.String,
                    "boundary_quantile": pl.Int64,
                    "lookup_action_id": pl.String,
                    "raw_order_fact_id": pl.String,
                    "policy_generation_id": pl.String,
                    "entry_execution_outcome": pl.String,
                    "pathwise_ev_ready": pl.Boolean,
                }
            )
            _write_source_partition(
                entry_root,
                "execution_action_facts.parquet",
                minimal_actions,
                "entry-test-v1",
            )
            loader = Mock(side_effect=AssertionError("raw tape must not be read"))
            arguments = {
                "sessions": [DATE, NEXT],
                "contract_calendar": _calendar(),
                "exit_maker_root": exit_root,
                "entry_execution_root": entry_root,
                "output_root": output_root,
                "raw_tape_loader": loader,
            }
            first = run_overnight_carry_replay(((DATE, VALUE),), **arguments)
            partition = output_root / f"Date={DATE}" / f"ValueCode={VALUE}"
            before = {
                path.name: _file_sha256(path)
                for path in partition.iterdir()
                if path.is_file()
            }
            resumed = run_overnight_carry_replay(((DATE, VALUE),), **arguments)
            after = {
                path.name: _file_sha256(path)
                for path in partition.iterdir()
                if path.is_file()
            }

            self.assertEqual(loader.call_count, 0)
            self.assertEqual(first.item(0, "overnight_carry_labels_rows"), 0)
            self.assertEqual(resumed.item(0, "overnight_carry_labels_rows"), 0)
            self.assertEqual(before, after)
            labels = pl.read_parquet(partition / "overnight_carry_labels.parquet")
            audit = pl.read_parquet(partition / "overnight_carry_audit.parquet")
            self.assertEqual(labels.height, 0)
            self.assertEqual(labels.width, 71)
            self.assertEqual(audit.item(0, "position_policy_rows"), 0)
            self.assertEqual(audit.item(0, "strict_carry_policy_rows"), 0)
            self.assertEqual(audit.item(0, "label_rows"), 0)

    def test_first_missing_candidate_censors_and_never_loads_later(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            exit_root, entry_root, output_root = self._roots(Path(directory))
            calls: list[str] = []

            def missing(date: str, mapping: pl.DataFrame) -> RawTapeDay:
                calls.append(date)
                if date == NEXT:
                    raise FileNotFoundError(date)
                return _tape(date, mapping)

            run_overnight_carry_replay(
                ((DATE, VALUE),),
                sessions=[DATE, NEXT, LATER],
                contract_calendar=_calendar(),
                exit_maker_root=exit_root,
                entry_execution_root=entry_root,
                output_root=output_root,
                config=OvernightCarryRunnerConfig(
                    carry=OvernightCarryConfig(max_carry_sessions=2)
                ),
                raw_tape_loader=missing,
            )
            self.assertEqual(calls, [NEXT])
            labels = pl.read_parquet(
                output_root
                / f"Date={DATE}"
                / f"ValueCode={VALUE}"
                / "overnight_carry_labels.parquet"
            )
            self.assertEqual(labels.item(0, "label_status"), "unresolved_missing_raw_tape")
            self.assertIn("later_sessions_not_examined", labels.item(0, "unresolved_reason"))

    def test_changed_entry_source_hash_invalidates_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            exit_root, entry_root, output_root = self._roots(Path(directory))
            loader = Mock(side_effect=lambda date, mapping: _tape(date, mapping))
            arguments = {
                "sessions": [DATE, NEXT],
                "contract_calendar": _calendar(),
                "exit_maker_root": exit_root,
                "entry_execution_root": entry_root,
                "output_root": output_root,
                "raw_tape_loader": loader,
            }
            run_overnight_carry_replay(((DATE, VALUE),), **arguments)

            partition = entry_root / f"Date={DATE}" / f"ValueCode={VALUE}"
            artifact = partition / "execution_action_facts.parquet"
            changed = _actions().with_columns(
                pl.lit(100.25).alias("entry_spot_price")
            )
            changed.write_parquet(artifact)
            marker_path = partition / "complete.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            marker["artifacts"][artifact.name] = {
                "rows": changed.height,
                "columns": changed.width,
                "bytes": artifact.stat().st_size,
                "sha256": _file_sha256(artifact),
            }
            marker_path.write_text(
                json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "partition config mismatch"):
                run_overnight_carry_replay(((DATE, VALUE),), **arguments)

    def test_changed_global_calendar_cannot_mix_one_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            exit_root, entry_root, output_root = self._roots(Path(directory))
            loader = Mock(side_effect=lambda date, mapping: _tape(date, mapping))
            arguments = {
                "sessions": [DATE, NEXT],
                "contract_calendar": _calendar(),
                "exit_maker_root": exit_root,
                "entry_execution_root": entry_root,
                "output_root": output_root,
                "raw_tape_loader": loader,
            }
            run_overnight_carry_replay(((DATE, VALUE),), **arguments)
            changed_calendar = _calendar().with_columns(
                pl.lit("official-test-v2").alias("calendar_version")
            )
            with self.assertRaisesRegex(
                ValueError, "runner (config mismatch|is not an allowlisted)"
            ):
                run_overnight_carry_replay(
                    ((DATE, VALUE),),
                    **{**arguments, "contract_calendar": changed_calendar},
                )

    def test_discovery_requires_both_complete_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            exit_root = temporary / "exit"
            entry_root = temporary / "entry"
            _write_source_partition(
                exit_root,
                "exit_maker_position_policy_facts.parquet",
                _position_facts(),
                "exit-test-v1",
            )
            keys, audit = discover_overnight_product_days(
                [DATE],
                [VALUE],
                exit_maker_root=exit_root,
                entry_execution_root=entry_root,
            )
            self.assertEqual(keys, ())
            self.assertEqual(audit.item(0, "availability_status"), "missing_entry_partition")

    def test_output_root_cannot_be_nested_inside_an_input_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            exit_root, entry_root, _ = self._roots(Path(directory))
            with self.assertRaisesRegex(ValueError, "must be disjoint"):
                run_overnight_carry_replay(
                    ((DATE, VALUE),),
                    sessions=[DATE, NEXT],
                    contract_calendar=_calendar(),
                    exit_maker_root=exit_root,
                    entry_execution_root=entry_root,
                    output_root=exit_root / "overnight",
                    raw_tape_loader=lambda date, mapping: _tape(date, mapping),
                )


if __name__ == "__main__":
    unittest.main()
