from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import polars as pl

from maker.src.quote_fill.execution_runner import EXECUTION_RUNNER_VERSION
from maker.src.quote_fill.future_ask_rank_diagnostic import (
    FutureAskRankDiagnosticConfig,
    publish_future_ask_rank_diagnostic,
    run_future_ask_rank_diagnostic,
    verify_future_ask_rank_diagnostic,
)


DATE = "20260731"
VALUE_CODE = "1513"
QUANTILES = (50, 80, 95)
RANKS = ("ASK1", "ASK2", "ASK3", "ASK4", "ASK5")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _schema(frame: pl.DataFrame) -> list[dict[str, str]]:
    return [
        {"name": name, "dtype": str(dtype)}
        for name, dtype in frame.schema.items()
    ]


def _action_row(
    q: int,
    rank: str,
    *,
    policy_suffix: str = "a",
    full: bool | None = None,
    raw_hedge_payload: bool = False,
) -> dict[str, object]:
    level = int(rank[-1])
    if full is None:
        full = level in {1, 3} and not (q == 95 and level == 1)
    submit = 1_000_000_000 + level * 1_000
    stop = submit + q * 10_000_000
    fill = submit + 5_000_000 if full else None
    decision = (
        fill + 50_000_000
        if fill is not None
        else (submit + 75_000_000 if raw_hedge_payload else None)
    )
    return {
        "Date": DATE,
        "ValueCode": VALUE_CODE,
        "QuoteCode": "SDFG6",
        "route": "future_ask_spot_taker",
        "maker_market": "future",
        "maker_side": "ask",
        "boundary_quantile": q,
        "raw_order_fact_id": f"raw-{rank}",
        "policy_generation_id": f"policy-{q}-{rank}-{policy_suffix}",
        "target_rank_at_submit": rank,
        "target_price": 100.0 + level,
        "initial_queue_ahead": level,
        "queue_known": True,
        "intended_quantity": 1,
        "submit_recv_time_ns": submit,
        "submit_event_sequence": 2,
        "submit_row_index": 100 + level,
        "nominal_stop_recv_time_ns": stop,
        "nominal_stop_reason": "target_retreat",
        "first_fill_recv_time_ns": fill,
        "first_fill_event_sequence": 2 if full else None,
        "first_fill_row_index": 200 + level if full else None,
        "full_fill_recv_time_ns": fill,
        "full_fill_event_sequence": 2 if full else None,
        "full_fill_row_index": 200 + level if full else None,
        "known_filled_quantity": 1 if full else 0,
        "any_fill": full,
        "full_fill": full,
        "partial_fill": False,
        "trade_through_fill": full,
        "fill_reason": "trade_through" if full else None,
        "terminal_recv_time_ns": fill if full else stop,
        "terminal_reason": "full_fill" if full else "target_retreat",
        "cancel_required": not full,
        "lifetime_ms": 5.0 if full else float(q * 10),
        "time_first_to_full_ms": 0.0 if full else None,
        "independent_event_label": True,
        "joint_volume_allocated": False,
        "entry_hedge_status": "executable" if full or raw_hedge_payload else None,
        "entry_hedge_decision_time_ns": decision,
        "entry_hedge_decision_snapshot_recv_time_ns": decision,
        "entry_hedge_decision_snapshot_event_sequence": (
            2 if full or raw_hedge_payload else None
        ),
        "entry_hedge_decision_snapshot_row_index": (
            300 + level if full or raw_hedge_payload else None
        ),
        "entry_hedge_arrival_reference_price": (
            50.0 if full or raw_hedge_payload else None
        ),
        "entry_hedge_decision_best_price": (
            50.01 if full or raw_hedge_payload else None
        ),
        "entry_hedge_executable_vwap_price": (
            50.02 if full or raw_hedge_payload else None
        ),
        "entry_hedge_available_quantity": (
            10 if full or raw_hedge_payload else None
        ),
        "entry_hedge_executed_quantity": (
            2 if full or raw_hedge_payload else None
        ),
        "entry_hedge_depth_shortfall": (
            0 if full or raw_hedge_payload else None
        ),
        "entry_hedge_levels_swept": (
            1 if full or raw_hedge_payload else None
        ),
        "entry_hedge_signed_latency_slippage_bp": (
            2.0 if full or raw_hedge_payload else None
        ),
        "entry_hedge_signed_depth_slippage_bp": (
            2.0 if full or raw_hedge_payload else None
        ),
        "entry_hedge_signed_total_slippage_bp": (
            float(level) if full or raw_hedge_payload else None
        ),
        "entry_hedge_decision_book_age_ms": (
            1.0 if full or raw_hedge_payload else None
        ),
        "entry_hedge_contract_size_shares": (
            2000 if full or raw_hedge_payload else None
        ),
        "entry_hedge_label_observed": full,
        "entry_hedge_executable": full,
    }


def _write_execution(root: Path, actions: pl.DataFrame) -> Path:
    execution = root / "execution"
    partition = execution / f"Date={DATE}" / f"ValueCode={VALUE_CODE}"
    partition.mkdir(parents=True)
    action_path = partition / "execution_action_facts.parquet"
    actions.write_parquet(action_path)
    config = {
        "boundary_quantiles": [50, 80, 95],
        "hedge_delay_ns": 50_000_000,
        "routes": ["future_ask_spot_taker", "spot_bid_future_taker"],
        "runner_version": EXECUTION_RUNNER_VERSION,
    }
    config_sha = _canonical(config)
    marker = {
        "complete": True,
        "Date": DATE,
        "ValueCode": VALUE_CODE,
        "runner_version": EXECUTION_RUNNER_VERSION,
        "config": config,
        "config_sha256": config_sha,
        "fact_semantics": {
            "hedge_delay_ns": 50_000_000,
            "independent_event_label": True,
            "joint_volume_allocated": False,
        },
        "artifacts": {
            "execution_action_facts.parquet": {
                "bytes": action_path.stat().st_size,
                "rows": actions.height,
                "columns": actions.width,
                "sha256": _sha(action_path),
            }
        },
    }
    (partition / "complete.json").write_text(
        json.dumps(marker, sort_keys=True), encoding="utf-8"
    )
    pl.DataFrame(
        {
            "Date": [DATE],
            "ValueCode": [VALUE_CODE],
            "partition": [str(partition.resolve())],
            "config_sha256": [config_sha],
            "complete": [True],
            "execution_action_facts_rows": [actions.height],
        }
    ).write_parquet(execution / "execution_partition_manifest.parquet")
    return execution


def _write_bid(root: Path, *, mixed_route: bool = False) -> Path:
    bid = root / "bid"
    bid.mkdir()
    rows = []
    for q in QUANTILES:
        for rank, group, full in (
            ("BID1", "L1_2", True),
            ("BID3", "L3_5", False),
        ):
            rows.append(
                {
                    "Date": DATE,
                    "ValueCode": VALUE_CODE,
                    "route": (
                        "future_ask_spot_taker" if mixed_route else "spot_bid_future_taker"
                    ),
                    "maker_market": "spot",
                    "maker_side": "bid",
                    "boundary_quantile": q,
                    "raw_order_fact_id": f"bid-{q}-{rank}",
                    "target_rank_at_submit": rank,
                    "rank_group": group,
                    "full_fill": full,
                    "entry_hedge_executable": full,
                    "entry_hedge_signed_total_slippage_bp": 3.0 if full else None,
                    "submit_event_sequence": 2,
                }
            )
    frame = pl.from_dicts(rows, infer_schema_length=None)
    path = bid / "alias_comparison.parquet"
    frame.write_parquet(path)
    marker = {
        "status": "complete",
        "config": {"dates": [DATE]},
        "artifacts": {
            "alias_comparison.parquet": {
                "bytes": path.stat().st_size,
                "rows": frame.height,
                "columns": frame.width,
                "sha256": _sha(path),
                "schema": _schema(frame),
            }
        },
    }
    (bid / "complete.json").write_text(
        json.dumps(marker, sort_keys=True), encoding="utf-8"
    )
    return bid


def _fixture(root: Path) -> FutureAskRankDiagnosticConfig:
    rows = [_action_row(q, rank) for q in QUANTILES for rank in RANKS]
    rows = [
        (
            _action_row(80, "ASK2", raw_hedge_payload=True)
            if row["boundary_quantile"] == 80
            and row["target_rank_at_submit"] == "ASK2"
            else row
        )
        for row in rows
    ]
    # A literal same-q physical alias must collapse without changing outcomes.
    rows.append(_action_row(50, "ASK1", policy_suffix="b"))
    actions = pl.from_dicts(rows, infer_schema_length=None)
    execution = _write_execution(root, actions)
    bid = _write_bid(root)
    return FutureAskRankDiagnosticConfig(
        execution_root=str(execution),
        bid_study_root=str(bid),
        dates=(DATE,),
        expected_product_day_counts=((DATE, 1),),
    )


class FutureAskRankDiagnosticTest(unittest.TestCase):
    def test_q_physical_dedupe_outcomes_and_route_separation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _fixture(Path(tmp))
            result = run_future_ask_rank_diagnostic(config)
            self.assertEqual(result.q_physical_orders.height, 15)
            self.assertEqual(result.physical_raw_inventory.height, 5)
            ask1_q50 = result.q_physical_orders.filter(
                (pl.col("boundary_quantile") == 50)
                & (pl.col("target_rank_at_submit") == "ASK1")
            ).row(0, named=True)
            self.assertEqual(ask1_q50["collapsed_policy_aliases"], 2)
            self.assertEqual(ask1_q50["outcome_class"], "full_fill")
            self.assertEqual(ask1_q50["hedge_delay_ns"], 50_000_000)
            self.assertEqual(
                ask1_q50["legacy_makerfill_status"],
                "not_used_future_maker_unsupported",
            )
            ask1_raw = result.physical_raw_inventory.filter(
                pl.col("raw_order_fact_id") == "raw-ASK1"
            ).row(0, named=True)
            self.assertEqual(ask1_raw["q_membership_count"], 3)
            self.assertTrue(ask1_raw["policy_outcome_varies_across_q"])
            self.assertEqual(
                set(result.symmetric_route_comparison.get_column("route")),
                {"future_ask_spot_taker", "spot_bid_future_taker"},
            )
            self.assertEqual(
                result.symmetric_route_comparison.filter(
                    pl.col("routes_pooled")
                ).height,
                0,
            )
            raw_payload = result.q_physical_orders.filter(
                (pl.col("boundary_quantile") == 80)
                & (pl.col("target_rank_at_submit") == "ASK2")
            ).row(0, named=True)
            self.assertTrue(raw_payload["entry_hedge_payload_present"])
            self.assertTrue(raw_payload["unobserved_raw_hedge_payload_excluded"])
            self.assertFalse(raw_payload["hedge_metric_eligible"])

    def test_rank_summary_partitions_full_partial_no_fill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_future_ask_rank_diagnostic(_fixture(Path(tmp)))
            rows = result.rank_summary.filter(
                (pl.col("date_scope") == "all_dates")
                & (pl.col("rank_scope") == "rank_group")
            )
            self.assertEqual(rows.get_column("n").sum(), 15)
            self.assertEqual(rows.get_column("partial_fill_count").sum(), 0)
            self.assertEqual(
                (
                    rows.get_column("full_fill_count")
                    + rows.get_column("partial_fill_count")
                    + rows.get_column("no_fill_count")
                ).to_list(),
                rows.get_column("n").to_list(),
            )
            self.assertEqual(result.coverage.height, 3)
            self.assertTrue(
                result.coverage.get_column("execution_partition_complete").all()
            )

    def test_divergent_same_q_alias_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [_action_row(q, rank) for q in QUANTILES for rank in RANKS]
            rows.append(_action_row(50, "ASK1", policy_suffix="bad", full=False))
            execution = _write_execution(
                root, pl.from_dicts(rows, infer_schema_length=None)
            )
            bid = _write_bid(root)
            config = FutureAskRankDiagnosticConfig(
                execution_root=str(execution),
                bid_study_root=str(bid),
                dates=(DATE,),
                expected_product_day_counts=((DATE, 1),),
            )
            with self.assertRaisesRegex(ValueError, "aliases disagree"):
                run_future_ask_rank_diagnostic(config)

    def test_wrong_hedge_delay_is_rejected_at_row_level(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [_action_row(q, rank) for q in QUANTILES for rank in RANKS]
            rows[0]["entry_hedge_decision_time_ns"] = int(
                rows[0]["full_fill_recv_time_ns"]
            ) + 49_000_000
            execution = _write_execution(
                root, pl.from_dicts(rows, infer_schema_length=None)
            )
            bid = _write_bid(root)
            config = FutureAskRankDiagnosticConfig(
                execution_root=str(execution),
                bid_study_root=str(bid),
                dates=(DATE,),
                expected_product_day_counts=((DATE, 1),),
            )
            with self.assertRaisesRegex(ValueError, "incoherent"):
                run_future_ask_rank_diagnostic(config)

    def test_existing_bid_mixed_route_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            actions = pl.from_dicts(
                [_action_row(q, rank) for q in QUANTILES for rank in RANKS],
                infer_schema_length=None,
            )
            execution = _write_execution(root, actions)
            bid = _write_bid(root, mixed_route=True)
            config = FutureAskRankDiagnosticConfig(
                execution_root=str(execution),
                bid_study_root=str(bid),
                dates=(DATE,),
                expected_product_day_counts=((DATE, 1),),
            )
            with self.assertRaisesRegex(ValueError, "mixed/invalid routes"):
                run_future_ask_rank_diagnostic(config)

    def test_publish_verify_and_atomic_existing_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _fixture(root)
            output = root / "bundle"
            publish_future_ask_rank_diagnostic(output, config)
            marker = verify_future_ask_rank_diagnostic(output)
            self.assertEqual(marker["status"], "complete")
            self.assertFalse(marker["safety"]["ask_legacy_makerfill_used"])
            with self.assertRaises(FileExistsError):
                publish_future_ask_rank_diagnostic(output, config)

    def test_plain_artifact_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "bundle"
            publish_future_ask_rank_diagnostic(output, _fixture(root))
            path = output / "rank_summary.parquet"
            frame = pl.read_parquet(path).with_columns(
                (pl.col("n") + 1).alias("n")
            )
            frame.write_parquet(path)
            with self.assertRaisesRegex(ValueError, "artifact mismatch"):
                verify_future_ask_rank_diagnostic(output)

    def test_coordinated_artifact_and_marker_tamper_hits_recompute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "bundle"
            publish_future_ask_rank_diagnostic(output, _fixture(root))
            path = output / "rank_summary.parquet"
            frame = pl.read_parquet(path).with_columns(
                (pl.col("n") + 1).alias("n")
            )
            frame.write_parquet(path)
            marker_path = output / "complete.json"
            marker = json.loads(marker_path.read_text())
            marker["artifacts"]["rank_summary.parquet"] = {
                "rows": frame.height,
                "columns": frame.width,
                "bytes": path.stat().st_size,
                "sha256": _sha(path),
                "schema": _schema(frame),
            }
            marker_path.write_text(json.dumps(marker, sort_keys=True))
            with self.assertRaisesRegex(ValueError, "semantic recomputation"):
                verify_future_ask_rank_diagnostic(output)

    def test_source_mutation_after_publish_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _fixture(root)
            output = root / "bundle"
            publish_future_ask_rank_diagnostic(output, config)
            action = next(
                (Path(config.execution_root) / f"Date={DATE}").glob(
                    "ValueCode=*/execution_action_facts.parquet"
                )
            )
            frame = pl.read_parquet(action).with_columns(
                pl.when(pl.col("policy_generation_id") == "policy-50-ASK2-a")
                .then(pl.lit("mutated"))
                .otherwise(pl.col("policy_generation_id"))
                .alias("policy_generation_id")
            )
            frame.write_parquet(action)
            with self.assertRaisesRegex(ValueError, "lineage drift"):
                verify_future_ask_rank_diagnostic(output)

    def test_extra_root_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "bundle"
            publish_future_ask_rank_diagnostic(output, _fixture(root))
            shutil.copy2(output / "audit.parquet", output / "hidden-extra.parquet")
            with self.assertRaisesRegex(ValueError, "root file set"):
                verify_future_ask_rank_diagnostic(output)


if __name__ == "__main__":
    unittest.main()
