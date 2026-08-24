from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import polars as pl

from maker.src.quote_fill.compact_remaining_time import (
    HEDGE_DELAY_NS,
    MINUTE_NS,
    SUMMARY_SCHEMA,
    _ANALYSIS_COLUMNS,
    _GATE_REASONS,
    _file_sha256,
    _session_cutoff_ns,
    _validate_output_target,
    _validate_source_paths_are_canonical,
    build_remaining_time_tables,
    main,
    publish_remaining_time_bundle,
    verify_remaining_time_bundle,
)


DATE = "20260714"
VALUE_CODE = "1513"


def _action(
    *,
    remaining_ns: int,
    reason: str = "target_retreat",
    outcome: str = "no_fill",
    quantile: int = 50,
    route: str = "spot_bid_future_taker",
    rank: str = "BID1",
    raw_id: str = "raw-1",
    wait_ns: int = 1_000_000_000,
    supported: bool = True,
) -> dict[str, object]:
    cutoff = _session_cutoff_ns(DATE)
    submit = cutoff - remaining_ns
    if reason == "session_cutoff":
        nominal_stop = cutoff
    else:
        nominal_stop = min(submit + 30_000_000_000, cutoff)
    full = outcome == "full"
    partial = outcome == "partial"
    any_fill = full or partial
    first = submit + wait_ns if any_fill else None
    full_time = first if full else None
    market = "spot" if route == "spot_bid_future_taker" else "future"
    side = "bid" if market == "spot" else "ask"
    if not supported:
        any_value: bool | None = None
        full_value: bool | None = None
        partial_value: bool | None = None
        cancel_value: bool | None = None
        first = None
        full_time = None
    else:
        any_value = any_fill
        full_value = full
        partial_value = partial
        cancel_value = not full
    return {
        "Date": DATE,
        "ValueCode": VALUE_CODE,
        "route": route,
        "maker_market": market,
        "maker_side": side,
        "boundary_quantile": quantile,
        "raw_order_fact_id": raw_id,
        "target_rank_at_submit": rank,
        "submit_recv_time_ns": submit,
        "nominal_stop_recv_time_ns": nominal_stop,
        "nominal_stop_reason": reason,
        "first_fill_recv_time_ns": first,
        "full_fill_recv_time_ns": full_time,
        "any_fill": any_value,
        "full_fill": full_value,
        "partial_fill": partial_value,
        "cancel_required": cancel_value,
        "entry_hedge_decision_time_ns": (
            full_time + HEDGE_DELAY_NS if full else None
        ),
        "entry_hedge_label_observed": full,
        "entry_hedge_executable": full,
    }


def _frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    frame = pl.from_dicts(rows, infer_schema_length=None)
    return frame.select(
        *[
            pl.col(name).cast(dtype, strict=True).alias(name)
            for name, dtype in _ANALYSIS_COLUMNS.items()
        ]
    )


def _manifest(rows: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Date": [DATE],
            "ValueCode": [VALUE_CODE],
            "execution_action_facts_rows": [rows],
        }
    )


class RemainingTimeAggregationTest(unittest.TestCase):
    def test_fixed_grid_bucket_edges_and_mutually_exclusive_outcomes(self) -> None:
        rows = [
            _action(
                remaining_ns=5 * MINUTE_NS - 1,
                reason="target_retreat",
                outcome="full",
                raw_id="r1",
            ),
            _action(
                remaining_ns=5 * MINUTE_NS,
                reason="future_ref_gate",
                outcome="partial",
                quantile=80,
                route="future_ask_spot_taker",
                rank="ASK1",
                raw_id="r2",
            ),
            _action(
                remaining_ns=15 * MINUTE_NS,
                reason="session_cutoff",
                outcome="no_fill",
                quantile=95,
                raw_id="r3",
            ),
            _action(
                remaining_ns=30 * MINUTE_NS,
                reason="target_not_passive",
                outcome="no_fill",
                raw_id="r4",
            ),
            _action(
                remaining_ns=60 * MINUTE_NS,
                reason="missing_anchor",
                outcome="full",
                raw_id="r5",
            ),
            _action(
                remaining_ns=10 * MINUTE_NS,
                reason="target_retreat",
                rank="BID3",
                raw_id="unsupported",
                supported=False,
            ),
        ]
        result = build_remaining_time_tables(_frame(rows), _manifest(len(rows)))
        self.assertEqual(result.summary.shape, (240, len(SUMMARY_SCHEMA)))
        populated = result.summary.filter(pl.col("cell_populated"))
        self.assertEqual(
            set(populated["remaining_session_bucket"].to_list()),
            {
                "lt_5m",
                "5_to_lt_15m",
                "15_to_lt_30m",
                "30_to_lt_60m",
                "ge_60m",
            },
        )
        self.assertEqual(int(result.summary["policy_aliases"].sum()), 5)
        self.assertEqual(int(result.summary["full_fill_aliases"].sum()), 2)
        self.assertEqual(int(result.summary["partial_only_aliases"].sum()), 1)
        self.assertEqual(int(result.summary["no_fill_aliases"].sum()), 2)
        self.assertEqual(int(result.summary["cancel_required_aliases"].sum()), 3)
        audit = result.product_day_audit.row(0, named=True)
        self.assertEqual(audit["source_action_rows"], 6)
        self.assertEqual(audit["supported_aliases"], 5)
        self.assertEqual(audit["unsupported_aliases"], 1)
        self.assertEqual(audit["gate_aliases"], 3)
        self.assertFalse(audit["unsupported_outcomes_imputed"])

    def test_all_frozen_gate_reasons_are_gate_and_unknown_fails_closed(self) -> None:
        rows = [
            _action(
                remaining_ns=(index + 1) * MINUTE_NS,
                reason=reason,
                raw_id=f"gate-{index}",
            )
            for index, reason in enumerate(_GATE_REASONS)
        ]
        result = build_remaining_time_tables(_frame(rows), _manifest(len(rows)))
        self.assertEqual(int(result.summary["policy_aliases"].sum()), len(rows))
        self.assertEqual(
            int(
                result.summary.filter(pl.col("nominal_stop_category") == "gate")[
                    "policy_aliases"
                ].sum()
            ),
            len(rows),
        )
        self.assertEqual(
            int(
                result.summary.filter(pl.col("nominal_stop_category") == "other")[
                    "policy_aliases"
                ].sum()
            ),
            0,
        )
        bad = _frame(
            [_action(remaining_ns=MINUTE_NS, reason="new_unfrozen_reason")]
        )
        with self.assertRaisesRegex(ValueError, "invalid action rows"):
            build_remaining_time_tables(bad, _manifest(1))

    def test_supported_null_and_invalid_remaining_time_fail_closed(self) -> None:
        row = _action(remaining_ns=MINUTE_NS)
        row["full_fill"] = None
        with self.assertRaisesRegex(ValueError, "invalid action rows"):
            build_remaining_time_tables(_frame([row]), _manifest(1))
        for remaining in (0, 255 * MINUTE_NS + 1):
            with self.subTest(remaining=remaining):
                with self.assertRaisesRegex(ValueError, "invalid action rows"):
                    build_remaining_time_tables(
                        _frame([_action(remaining_ns=remaining)]), _manifest(1)
                    )

    def test_nearest_conditional_wait_quantiles_and_zero_cells(self) -> None:
        rows = [
            _action(
                remaining_ns=20 * MINUTE_NS,
                outcome="full",
                raw_id=f"r{wait}",
                wait_ns=wait * 1_000_000_000,
            )
            for wait in (1, 2, 3, 4)
        ]
        result = build_remaining_time_tables(_frame(rows), _manifest(len(rows)))
        cell = result.summary.filter(
            (pl.col("boundary_quantile") == 50)
            & (pl.col("route") == "spot_bid_future_taker")
            & (pl.col("target_rank_at_submit") == "BID1")
            & (pl.col("remaining_session_bucket") == "15_to_lt_30m")
            & (pl.col("nominal_stop_category") == "target_retreat")
        ).row(0, named=True)
        self.assertEqual(cell["wait_to_first_seconds_p50"], 3.0)
        self.assertEqual(cell["wait_to_first_seconds_p90"], 4.0)
        self.assertEqual(cell["wait_to_first_seconds_p95"], 4.0)
        self.assertEqual(cell["wait_to_full_seconds_p50"], 3.0)
        empty = result.summary.filter(~pl.col("cell_populated")).row(0, named=True)
        self.assertIsNone(empty["full_fill_rate"])
        self.assertIsNone(empty["wait_to_first_seconds_p50"])
        self.assertIsNone(empty["wait_to_full_seconds_p50"])

    def test_shared_raw_q_aliases_are_not_claimed_additive(self) -> None:
        rows = [
            _action(
                remaining_ns=20 * MINUTE_NS,
                quantile=quantile,
                raw_id="same-physical",
            )
            for quantile in (50, 80, 95)
        ]
        result = build_remaining_time_tables(_frame(rows), _manifest(len(rows)))
        selected = result.summary.filter(pl.col("policy_aliases") > 0)
        self.assertEqual(selected.height, 3)
        self.assertEqual(int(selected["unique_physical_orders"].sum()), 3)
        self.assertTrue(
            selected["alternative_q_rows_additive"].eq(False).all()  # noqa: E712
        )
        self.assertTrue(selected["route_rows_additive"].eq(False).all())  # noqa: E712

    def test_typed_empty_partition_is_zero_support_not_no_fill(self) -> None:
        empty = pl.DataFrame(schema=_ANALYSIS_COLUMNS)
        result = build_remaining_time_tables(empty, _manifest(0))
        self.assertEqual(result.summary.height, 240)
        self.assertEqual(int(result.summary["policy_aliases"].sum()), 0)
        self.assertEqual(int(result.summary["no_fill_aliases"].sum()), 0)
        audit = result.product_day_audit.row(0, named=True)
        self.assertEqual(audit["source_action_rows"], 0)
        self.assertEqual(audit["supported_aliases"], 0)
        self.assertEqual(audit["unsupported_aliases"], 0)

    def test_null_manifest_row_count_fails_closed(self) -> None:
        manifest = pl.DataFrame(
            {
                "Date": [DATE],
                "ValueCode": [VALUE_CODE],
                "execution_action_facts_rows": [None],
            },
            schema_overrides={"execution_action_facts_rows": pl.Int64},
        )
        with self.assertRaisesRegex(ValueError, "manifest grid is invalid"):
            build_remaining_time_tables(
                pl.DataFrame(schema=_ANALYSIS_COLUMNS), manifest
            )

    def test_cutoff_epoch_is_exact_for_multiple_dates(self) -> None:
        self.assertEqual(_session_cutoff_ns("20260714"), 1784006400000000000)
        self.assertEqual(
            _session_cutoff_ns("20260820") - _session_cutoff_ns("20260714"),
            37 * 86_400 * 1_000_000_000,
        )

    def test_source_partition_or_artifact_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "execution"
            partition = root / f"Date={DATE}" / f"ValueCode={VALUE_CODE}"
            partition.mkdir(parents=True)
            (root / "execution_partition_manifest.parquet").write_bytes(b"m")
            (root / "product_day_universe.csv").write_bytes(b"u")
            (partition / "complete.json").write_bytes(b"{}")
            action = partition / "execution_action_facts.parquet"
            action.write_bytes(b"a")
            manifest = pl.DataFrame(
                {
                    "Date": [DATE],
                    "ValueCode": [VALUE_CODE],
                    "partition": [str(partition.resolve())],
                }
            )
            _validate_source_paths_are_canonical(root, manifest)
            action.unlink()
            target = Path(temporary) / "outside.parquet"
            target.write_bytes(b"a")
            action.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "symlink"):
                _validate_source_paths_are_canonical(root, manifest)


class RemainingTimeBundleTest(unittest.TestCase):
    def setUp(self) -> None:
        rows = [
            _action(
                remaining_ns=10 * MINUTE_NS,
                outcome="full",
                raw_id="bundle-full",
            ),
            _action(
                remaining_ns=20 * MINUTE_NS,
                outcome="no_fill",
                reason="session_cutoff",
                raw_id="bundle-no-fill",
            ),
        ]
        self.result = build_remaining_time_tables(_frame(rows), _manifest(2))
        self.temporary = tempfile.TemporaryDirectory()
        self.output = Path(self.temporary.name) / "bundle"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _publish(self) -> None:
        with mock.patch(
            "maker.src.quote_fill.compact_remaining_time."
            "summarize_remaining_time_universe",
            return_value=self.result,
        ):
            publish_remaining_time_bundle(self.output)

    def _verify(self) -> dict[str, object]:
        with mock.patch(
            "maker.src.quote_fill.compact_remaining_time."
            "summarize_remaining_time_universe",
            return_value=self.result,
        ):
            return verify_remaining_time_bundle(self.output)

    def test_atomic_publish_verify_and_exact_inventory(self) -> None:
        self._publish()
        marker = self._verify()
        self.assertEqual(set(marker["artifacts"]), {
            "remaining_time_stop_reason_summary.parquet",
            "product_day_audit.parquet",
        })
        self.assertFalse(marker["safety"]["online_stop_reason_feature_ready"])
        self.assertTrue(
            marker["safety"]["realized_nominal_stop_reason_future_label"]
        )
        self.assertFalse(marker["safety"]["alternative_q_rows_additive"])

    def test_coordinated_artifact_and_marker_tamper_is_rejected(self) -> None:
        self._publish()
        path = self.output / "remaining_time_stop_reason_summary.parquet"
        frame = pl.read_parquet(path).with_columns(
            (pl.col("policy_aliases") + 1).alias("policy_aliases")
        )
        frame.write_parquet(path)
        marker_path = self.output / "complete.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        metadata = marker["artifacts"][path.name]
        metadata["bytes"] = path.stat().st_size
        metadata["sha256"] = _file_sha256(path)
        marker_path.write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "differs from source rebuild"):
            self._verify()

    def test_extra_file_and_source_anchor_tamper_are_rejected(self) -> None:
        self._publish()
        (self.output / "undeclared.parquet").write_bytes(b"x")
        with self.assertRaisesRegex(ValueError, "inventory mismatch"):
            self._verify()
        (self.output / "undeclared.parquet").unlink()
        marker_path = self.output / "complete.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["source"]["execution_manifest_sha256"] = "0" * 64
        marker_path.write_text(json.dumps(marker), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source anchors"):
            self._verify()

    def test_cli_verify_only_uses_same_strict_verifier(self) -> None:
        self._publish()
        with mock.patch(
            "maker.src.quote_fill.compact_remaining_time."
            "summarize_remaining_time_universe",
            return_value=self.result,
        ), redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--verify-only", str(self.output)]), 0)

    def test_atomic_failure_leaves_no_output_or_stage(self) -> None:
        with mock.patch(
            "maker.src.quote_fill.compact_remaining_time."
            "summarize_remaining_time_universe",
            return_value=self.result,
        ), mock.patch(
            "maker.src.quote_fill.compact_remaining_time."
            "verify_remaining_time_bundle",
            side_effect=ValueError("forced verifier failure"),
        ):
            with self.assertRaisesRegex(ValueError, "forced verifier failure"):
                publish_remaining_time_bundle(self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(
            list(Path(self.temporary.name).glob(".bundle.tmp.*")), []
        )

    def test_dangling_symlink_and_source_descendant_output_are_rejected(self) -> None:
        dangling = Path(self.temporary.name) / "dangling"
        dangling.symlink_to(
            Path(self.temporary.name) / "absent", target_is_directory=True
        )
        with self.assertRaises(FileExistsError):
            _validate_output_target(dangling, Path(self.temporary.name) / "source")
        source = Path(self.temporary.name) / "source"
        source.mkdir()
        with self.assertRaisesRegex(ValueError, "must not mutate"):
            _validate_output_target(source / "analysis", source)
        linked_parent = Path(self.temporary.name) / "linked-source"
        linked_parent.symlink_to(source, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "must not mutate"):
            _validate_output_target(linked_parent / "analysis", source)


if __name__ == "__main__":
    unittest.main()
