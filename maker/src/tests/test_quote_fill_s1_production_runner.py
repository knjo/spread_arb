from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from maker.src.quote_fill.capacity_ledger import (
    CapacityLedger,
    CapacityLedgerCompactCheckpoint,
    CapacityTransition,
    decode_capacity_ledger_compact_checkpoint,
    decode_capacity_transition,
    encode_capacity_ledger_compact_checkpoint,
    encode_capacity_transition,
)
from maker.src.quote_fill.s1_accounting_bridge import S1AccountingProduct
from maker.src.quote_fill.s1_bundle_artifacts import (
    read_s1_bundle_partition,
    write_s1_bundle_partition,
)
from maker.src.quote_fill.s1_capacity_identity_registry import (
    S1CapacityIdentityRegistry,
)
from maker.src.quote_fill.s1_open_position_valuation import (
    COMMON_HORIZON_CURSOR,
    COMMON_HORIZON_DATE,
)
from maker.src.quote_fill.s1_production_runner import (
    CAPACITY_REGISTRY_FILENAME,
    COMMON_POPULATION_SCHEMA_VERSION,
    FINAL_BUNDLE_SCHEMA_VERSION,
    GENESIS_PARTITION_SHA256,
    S1_DEVELOPMENT_DATES,
    VERIFICATION_FILENAME,
    VERIFICATION_SCHEMA_VERSION,
    VERIFIER_VERSION,
    S1ProductionConfig,
    S1ProductionRunError,
    _build_common_horizon_valuations,
    _bundle_complete_event_record,
    _ensure_date_input_manifest,
    _ensure_run_config,
    _lineage_record,
    _load_resume_prefix,
    _open_valuation_result_record,
    _partition_marker_sha256,
    _partition_path,
    _performance_record,
    _publish_or_verify_canonical_json,
    _render_report,
    _validate_common_horizon_accounting,
    _verify_resumed_capacity_partition,
    verify_s1_production_bundle,
)
from maker.src.quote_fill.s1_publication_gate import (
    S1OpenPositionValuation,
    S1PublicationDecision,
)
from maker.src.quote_fill.s1_scenario_spec import SCENARIO_IDS

_RUN_FINGERPRINT = "a" * 64
_RUN_CONFIG_SHA256 = "b" * 64
_INPUT_MANIFEST_SHA256 = "c" * 64


@dataclass(frozen=True)
class _PerformanceRecordFixture:
    scenario_id: str
    reporting_sessions: int
    approx_screen_completion_numerator: int
    approx_screen_completion_denominator: int
    approx_screen_completion_rate: Decimal
    mean_daily_net_twd: Decimal
    total_net_twd: Decimal
    terminal_net_bp_of_turnover: Decimal
    open_or_unresolved: int
    hedge_priced_numerator: int
    hedge_priced_denominator: int
    hedge_pricing_coverage_source: str


class S1ProductionRunnerIntegrationTest(unittest.TestCase):
    def test_blocked_report_omits_optional_rankings_and_prints_gate_reason(
        self,
    ) -> None:
        decision = self._publication_decision(
            economic_allowed=False,
            s2_allowed=False,
            economic_blockers=("scenario[x].open_valuation.missing",),
        )

        report = _render_report(
            run_config=self._run_config(),
            run_config_sha256="b" * 64,
            performance=self._performance_rows(),
            open_valuations=self._open_valuations(),
            shortlist=None,
            publication_decision=decision,
            diagnostics=self._diagnostics(),
            entry_month={scenario_id: () for scenario_id in SCENARIO_IDS},
        )

        self.assertIn("economic ranking allowed：`false`", report.lower())
        self.assertIn("scenario[x].open_valuation.missing", report)
        self.assertIn("未發布", report)
        self.assertNotIn("Completion champion：`", report)
        self.assertIn("ever-capacity-blocked candidates", report)

    def test_report_exposes_cost_aware_location_risk_and_capacity_metrics(
        self,
    ) -> None:
        report = _render_report(
            run_config=self._run_config(),
            run_config_sha256="b" * 64,
            performance=self._performance_rows(),
            open_valuations=self._open_valuations(),
            shortlist=None,
            publication_decision=self._publication_decision(
                economic_allowed=False,
                s2_allowed=False,
                economic_blockers=("scenario[x].open_valuation.missing",),
            ),
            diagnostics=self._diagnostics(),
            entry_month={scenario_id: () for scenario_id in SCENARIO_IDS},
        )

        self.assertIn("mean daily terminal realized net TWD", report)
        self.assertIn("`BID1`=2, `BID2`=3, `not_BID1_or_BID2`=4 | 9", report)
        self.assertIn("entry:hedge | 5 | 3 | 2 | 8/10 | 9/10", report)
        self.assertIn("7/8", report)
        self.assertIn("101 | 102 | 103 | 104 | 105 | 106", report)
        self.assertIn("不可相加", report)

    def test_performance_result_record_names_terminal_only_mean_daily_net(
        self,
    ) -> None:
        performance = _PerformanceRecordFixture(
            scenario_id=SCENARIO_IDS[0],
            reporting_sessions=71,
            approx_screen_completion_numerator=3,
            approx_screen_completion_denominator=5,
            approx_screen_completion_rate=Decimal("0.6"),
            mean_daily_net_twd=Decimal("123.45"),
            total_net_twd=Decimal("8764.95"),
            terminal_net_bp_of_turnover=Decimal("4.25"),
            open_or_unresolved=2,
            hedge_priced_numerator=8,
            hedge_priced_denominator=10,
            hedge_pricing_coverage_source="all_risk_lifecycle_events",
        )
        record = _performance_record(
            performance,
            self._open_valuation(
                performance.scenario_id,
                open_count=2,
                publishable=False,
            ),
        )

        self.assertEqual(
            record["mean_daily_terminal_realized_net_twd_20m"],
            "123.45",
        )
        self.assertEqual(
            record["mean_daily_terminal_realized_net_scope"],
            "executable_terminals_and_expiry_marks_only;"
            "open_positions_reported_in_separate_common_horizon_mark",
        )
        self.assertFalse(record["open_positions_comparably_valued"])
        self.assertIsNone(record["economic_ranking_net_twd"])

    def test_common_horizon_conserves_final_paired_naked_and_open_costs(
        self,
    ) -> None:
        performance = SimpleNamespace(
            scenario_id=SCENARIO_IDS[0],
            open_or_unresolved=2,
            open_execution_actual_commission_twd=Decimal(3),
            open_execution_actual_tax_twd=Decimal(7),
            open_execution_actual_cost_twd=Decimal(10),
        )
        valuation = SimpleNamespace(
            scenario_id=SCENARIO_IDS[0],
            rows=(
                SimpleNamespace(
                    incurred_commission_twd=Decimal(1),
                    incurred_tax_twd=Decimal(2),
                ),
                SimpleNamespace(
                    incurred_commission_twd=Decimal(2),
                    incurred_tax_twd=Decimal(5),
                ),
            ),
            incurred_open_execution_cost_twd=Decimal(10),
        )
        diagnostics = {
            "final_carry_positions": 2,
            "final_naked_unresolved_positions": 0,
        }

        self.assertEqual(
            _validate_common_horizon_accounting(
                performance,
                valuation,
                diagnostics,
            ),
            (2, 0),
        )

        with self.assertRaisesRegex(S1ProductionRunError, "naked unresolved"):
            _validate_common_horizon_accounting(
                SimpleNamespace(
                    **{
                        **performance.__dict__,
                        "open_or_unresolved": 3,
                    }
                ),
                valuation,
                {
                    **diagnostics,
                    "final_naked_unresolved_positions": 1,
                },
            )

        with self.assertRaisesRegex(S1ProductionRunError, "open commission"):
            _validate_common_horizon_accounting(
                SimpleNamespace(
                    **{
                        **performance.__dict__,
                        "open_execution_actual_commission_twd": Decimal(4),
                    }
                ),
                valuation,
                diagnostics,
            )

    def test_common_horizon_builder_uses_exact_8_13_cursor_and_final_facts(
        self,
    ) -> None:
        class _Adapter:
            def __init__(self) -> None:
                self.calls: list[tuple[str, str, object]] = []

            def state_as_of(
                self,
                venue: str,
                product_id: str,
                cursor: object,
            ) -> None:
                self.calls.append((venue, product_id, cursor))

        adapter = _Adapter()
        prepared = SimpleNamespace(
            session_expiry_time_ns=COMMON_HORIZON_CURSOR.recv_time_ns,
            raw_books=SimpleNamespace(as_risk_book_adapter=lambda: adapter),
            products=(
                SimpleNamespace(product_id="2330", value_code="2330"),
                SimpleNamespace(product_id="2317", value_code="2317"),
            ),
            source_paths={},
        )
        states = {
            scenario_id: SimpleNamespace(
                accounting=SimpleNamespace(
                    verify=lambda scenario_id=scenario_id: SimpleNamespace(
                        facts=(f"final-fact-{scenario_id}",)
                    )
                )
            )
            for scenario_id in SCENARIO_IDS
        }
        config = S1ProductionConfig(
            output_root=Path("/tmp/s1-builder-test"),
            report_path=Path("/tmp/s1-builder-test.md"),
        )

        def _mark(facts: object, *, scenario_id: str, **_: object) -> object:
            self.assertEqual(facts, (f"final-fact-{scenario_id}",))
            return SimpleNamespace(scenario_id=scenario_id)

        module = "maker.src.quote_fill.s1_production_runner"
        with (
            patch(f"{module}._merge_common_horizon_bindings", return_value=()),
            patch(
                f"{module}._ensure_date_input_manifest",
                return_value=({"input_records": []}, "a" * 64),
            ) as manifest,
            patch(f"{module}._validated_file_records", return_value=[]),
            patch(f"{module}._verify_file_records_stable"),
            patch(f"{module}.prepare_s1_entry_day", return_value=prepared) as prepare,
            patch(f"{module}._validate_prepared_source_paths"),
            patch(f"{module}._validate_prepared_day"),
            patch(
                f"{module}.value_s1_common_horizon_open_positions",
                side_effect=_mark,
            ),
        ):
            result = _build_common_horizon_valuations(
                config=config,
                run_config_sha256="b" * 64,
                states=states,
                catalog={},
                verification_only=True,
            )

        self.assertEqual(
            tuple(row.scenario_id for row in result),
            SCENARIO_IDS,
        )
        manifest.assert_called_once_with(
            config,
            date=COMMON_HORIZON_DATE,
            run_config_sha256="b" * 64,
            require_existing=True,
        )
        prepare.assert_called_once_with(
            COMMON_HORIZON_DATE,
            paths=config.paths,
            required_exit_only_bindings=(),
        )
        self.assertEqual(
            adapter.calls,
            [
                (venue, product_id, COMMON_HORIZON_CURSOR)
                for product_id in ("2330", "2317")
                for venue in ("spot", "future")
            ],
        )

    def test_open_valuation_result_record_handles_priced_unpriced_and_zero(
        self,
    ) -> None:
        aggregate = S1OpenPositionValuation(
            position_count=1,
            valuation_method_id="method",
            valuation_asof_id="asof",
            comparable_across_scenarios=True,
            gross_mark_pnl_twd=Decimal(12),
            remaining_exit_cost_twd=Decimal(2),
            net_mark_pnl_twd=Decimal(7),
        )
        priced_row = SimpleNamespace(
            priced=True,
            unpriced_reason=None,
            spot_liquidation_cashflow_twd=Decimal(100),
            future_exit_vwap=Decimal(50),
            future_share_equivalent=2,
        )
        module = "maker.src.quote_fill.s1_production_runner"
        with patch(
            f"{module}.encode_s1_open_position_valuations",
            return_value=[{"position_id": "p"}],
        ):
            priced = _open_valuation_result_record(
                SimpleNamespace(
                    scenario_id=SCENARIO_IDS[0],
                    rows=(priced_row,),
                    incurred_open_execution_cost_twd=Decimal(3),
                    aggregate=aggregate,
                )
            )
        self.assertEqual(priced["priced_position_count"], 1)
        self.assertEqual(priced["spot_liquidation_notional_twd"], "100")
        self.assertEqual(priced["future_liquidation_notional_twd"], "100")
        self.assertEqual(priced["rows"], [{"position_id": "p"}])

        unpriced_row = SimpleNamespace(
            priced=False,
            unpriced_reason="spot_insufficient_depth",
        )
        with patch(
            f"{module}.encode_s1_open_position_valuations",
            return_value=[{"position_id": "u"}],
        ):
            unpriced = _open_valuation_result_record(
                SimpleNamespace(
                    scenario_id=SCENARIO_IDS[0],
                    rows=(unpriced_row,),
                    incurred_open_execution_cost_twd=Decimal(3),
                    aggregate=None,
                )
            )
        self.assertEqual(unpriced["unpriced_position_count"], 1)
        self.assertEqual(
            unpriced["unpriced_reason_counts"],
            {"spot_insufficient_depth": 1},
        )
        self.assertIsNone(unpriced["spot_liquidation_notional_twd"])

        zero = _open_valuation_result_record(
            SimpleNamespace(
                scenario_id=SCENARIO_IDS[0],
                rows=(),
                incurred_open_execution_cost_twd=Decimal(0),
                aggregate=S1OpenPositionValuation(
                    position_count=0,
                    valuation_method_id="method",
                    valuation_asof_id="asof",
                    comparable_across_scenarios=True,
                    gross_mark_pnl_twd=Decimal(0),
                    remaining_exit_cost_twd=Decimal(0),
                    net_mark_pnl_twd=Decimal(0),
                ),
            )
        )
        self.assertEqual(zero["rows"], [])
        self.assertEqual(zero["spot_liquidation_notional_twd"], "0")

    def test_allowed_report_and_cli_event_publish_safe_champions(self) -> None:
        decision = self._publication_decision(
            economic_allowed=True,
            s2_allowed=True,
        )
        ranked = tuple(SimpleNamespace(scenario_id=value) for value in SCENARIO_IDS[1:])
        shortlist = SimpleNamespace(
            completion_champion=ranked[0],
            net_champion=ranked[1],
            selected=ranked[:2],
            completion_ranking=ranked,
            net_ranking=tuple(reversed(ranked)),
            pareto_frontier=ranked[:2],
        )
        report = _render_report(
            run_config=self._run_config(),
            run_config_sha256="b" * 64,
            performance=self._performance_rows(),
            open_valuations=self._open_valuations(),
            shortlist=shortlist,
            publication_decision=decision,
            diagnostics=self._diagnostics(),
            entry_month={scenario_id: () for scenario_id in SCENARIO_IDS},
        )
        event = _bundle_complete_event_record(
            SimpleNamespace(
                output_root=Path("out"),
                report_path=Path("report.md"),
                run_config_sha256="b" * 64,
                complete_sha256="c" * 64,
                shortlist=shortlist,
                publication_decision=decision,
            )
        )

        self.assertIn(f"Completion champion：`{ranked[0].scenario_id}`", report)
        self.assertTrue(event["shortlist_available"])
        self.assertEqual(event["completion_champion"], ranked[0].scenario_id)

    def test_cli_event_is_safe_without_a_shortlist(self) -> None:
        decision = self._publication_decision(
            economic_allowed=False,
            s2_allowed=False,
            economic_blockers=("scenario[x].open_valuation.missing",),
        )
        event = _bundle_complete_event_record(
            SimpleNamespace(
                output_root=Path("out"),
                report_path=Path("report.md"),
                run_config_sha256="b" * 64,
                complete_sha256="c" * 64,
                shortlist=None,
                publication_decision=decision,
            )
        )

        self.assertFalse(event["shortlist_available"])
        self.assertIsNone(event["completion_champion"])
        self.assertIsNone(event["net_champion"])
        self.assertEqual(
            event["economic_ranking_blockers"],
            ["scenario[x].open_valuation.missing"],
        )

    def test_capacity_receipts_round_trip_across_two_artifact_partitions(self) -> None:
        date_one, date_two = S1_DEVELOPMENT_DATES[:2]
        policy_id = SCENARIO_IDS[0]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = S1CapacityIdentityRegistry(root / CAPACITY_REGISTRY_FILENAME)
            self.addCleanup(registry.close)
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
            self.addCleanup(registry.close)
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
                SCENARIO_IDS[0],
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

    def test_verify_success_atomically_persists_canonical_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "bundle"
            root.mkdir()
            report_path = Path(temporary) / "report.md"
            source_commit = "1" * 40
            run_config = {"source_commit": source_commit}
            run_config_payload = self._canonical_json_bytes(run_config)
            (root / "run_config.json").write_bytes(run_config_payload)
            results_payload = self._canonical_json_bytes({"result": "verified"})
            (root / "results.json").write_bytes(results_payload)
            report_path.write_text("verified report\n", encoding="utf-8")
            run_config_sha256 = hashlib.sha256(run_config_payload).hexdigest()
            results_sha256 = hashlib.sha256(results_payload).hexdigest()
            report_sha256 = hashlib.sha256(report_path.read_bytes()).hexdigest()
            complete = {
                "schema_version": FINAL_BUNDLE_SCHEMA_VERSION,
                "complete": True,
                "run_config_sha256": run_config_sha256,
                "results_sha256": results_sha256,
                "report_sha256": report_sha256,
                "partition_count": len(S1_DEVELOPMENT_DATES) * len(SCENARIO_IDS),
            }
            complete_payload = self._canonical_json_bytes(complete)
            (root / "complete.json").write_bytes(complete_payload)
            complete_sha256 = hashlib.sha256(complete_payload).hexdigest()
            result = SimpleNamespace(
                output_root=root,
                report_path=report_path,
                run_config_sha256=run_config_sha256,
                complete_sha256=complete_sha256,
                verifier_source_commit=source_commit,
                resumed_partitions=(
                    len(S1_DEVELOPMENT_DATES) * len(SCENARIO_IDS)
                ),
                executed_partitions=0,
            )
            config = S1ProductionConfig(
                output_root=root,
                report_path=report_path,
                source_commit=source_commit,
            )

            with (
                patch(
                    "maker.src.quote_fill.s1_production_runner._run_s1_production_bundle",
                    return_value=result,
                ) as replay,
                patch(
                    "maker.src.quote_fill.s1_production_runner._git_source_commit",
                    return_value=source_commit,
                ) as source_guard,
            ):
                self.assertIs(verify_s1_production_bundle(config), result)
                self.assertIs(verify_s1_production_bundle(config), result)
            self.assertEqual(source_guard.call_count, 2)
            source_guard.assert_called_with(expected=source_commit)

            verified_config = replay.call_args.args[0]
            self.assertTrue(verified_config.verify_completed_input_content)
            self.assertTrue(replay.call_args.kwargs["verification_only"])
            verification_path = root / VERIFICATION_FILENAME
            self.assertEqual((root / "complete.json").read_bytes(), complete_payload)
            payload = verification_path.read_bytes()
            record = json.loads(payload)
            self.assertEqual(payload, self._canonical_json_bytes(record))
            self.assertEqual(
                record,
                {
                    "schema_version": VERIFICATION_SCHEMA_VERSION,
                    "verifier_version": VERIFIER_VERSION,
                    "verification_status": "verified",
                    "verify_inputs": True,
                    "complete_sha256": complete_sha256,
                    "run_config_sha256": run_config_sha256,
                    "bundle_source_commit": source_commit,
                    "verifier_source_commit": source_commit,
                    "partition_count": (
                        len(S1_DEVELOPMENT_DATES) * len(SCENARIO_IDS)
                    ),
                    "results_sha256": results_sha256,
                    "report_sha256": report_sha256,
                },
            )

    def test_verify_failure_never_creates_verification_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "bundle"
            root.mkdir()
            config = S1ProductionConfig(
                output_root=root,
                report_path=Path(temporary) / "report.md",
            )

            with patch(
                "maker.src.quote_fill.s1_production_runner._run_s1_production_bundle",
                side_effect=S1ProductionRunError("deep verification failed"),
            ), self.assertRaisesRegex(
                S1ProductionRunError,
                "deep verification failed",
            ):
                verify_s1_production_bundle(config)

            self.assertFalse((root / VERIFICATION_FILENAME).exists())

    def test_verification_guards_never_recreate_missing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_config_path = root / "run_config.json"
            with self.assertRaisesRegex(
                S1ProductionRunError,
                "requires existing exact run_config",
            ):
                _ensure_run_config(
                    run_config_path,
                    {"schema_version": "test"},
                    require_existing_exact=True,
                )
            self.assertFalse(run_config_path.exists())

            results_path = root / "results.json"
            with self.assertRaisesRegex(
                S1ProductionRunError,
                "existing real file",
            ):
                _publish_or_verify_canonical_json(
                    results_path,
                    {"result": "recomputed"},
                    verification_only=True,
                )
            self.assertFalse(results_path.exists())

            config = S1ProductionConfig(
                output_root=root,
                report_path=root / "report.md",
            )
            with self.assertRaisesRegex(
                S1ProductionRunError,
                "requires existing date input manifest",
            ):
                _ensure_date_input_manifest(
                    config,
                    date=S1_DEVELOPMENT_DATES[0],
                    run_config_sha256="a" * 64,
                    require_existing=True,
                )
            manifest = (
                root
                / "input_manifests"
                / f"Date={S1_DEVELOPMENT_DATES[0]}.json"
            )
            self.assertFalse(manifest.exists())

    def test_resume_rejects_noncanonical_date_major_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            later_partition = _partition_path(
                root,
                S1_DEVELOPMENT_DATES[0],
                SCENARIO_IDS[1],
            )
            later_partition.mkdir(parents=True)
            config = S1ProductionConfig(
                output_root=root,
                report_path=root / "report.md",
                source_commit="1" * 40,
            )
            registry = S1CapacityIdentityRegistry(root / CAPACITY_REGISTRY_FILENAME)
            self.addCleanup(registry.close)

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
                    verification_only=False,
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

    @staticmethod
    def _canonical_json_bytes(value: object) -> bytes:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")

    @staticmethod
    def _publication_decision(
        *,
        economic_allowed: bool,
        s2_allowed: bool,
        economic_blockers: tuple[str, ...] = (),
    ) -> S1PublicationDecision:
        return S1PublicationDecision(
            publication_allowed=True,
            descriptive_publication_allowed=True,
            completion_ranking_allowed=True,
            economic_ranking_allowed=economic_allowed,
            s2_shortlist_allowed=s2_allowed,
            integrity_failures=(),
            completion_ranking_blockers=(),
            economic_ranking_blockers=economic_blockers,
            shortlist_blockers=(),
            failure_reasons=(),
            unavailable_cost_disclosures=(
                "scenario[x].unmodeled_cost[overnight_financing]=not_modeled",
            ),
        )

    @staticmethod
    def _run_config() -> dict[str, object]:
        return {
            "source_commit": "1" * 40,
            "common_population": {
                "schema_version": COMMON_POPULATION_SCHEMA_VERSION,
                "key_columns": [
                    "Date",
                    "ValueCode",
                    "QuoteCode",
                    "entry_tod_bucket",
                ],
                "cell_count": 15_638 * 4,
                "hash_semantics": "sha256_concatenated_canonical_json_lines_v1",
                "sha256": "a" * 64,
            },
        }

    @staticmethod
    def _performance_rows() -> tuple[SimpleNamespace, ...]:
        return tuple(
            SimpleNamespace(
                scenario_id=scenario_id,
                reporting_sessions=71,
                entry_fill_approximate=1,
                same_day_exit_maker_flat=1,
                approx_screen_completion_numerator=1,
                approx_screen_completion_denominator=1,
                open_or_unresolved=0,
                terminal_coverage_numerator=1,
                terminal_coverage_denominator=1,
                terminal_gross_pnl_twd=Decimal(10),
                terminal_modeled_direct_cost_twd=Decimal(2),
                total_net_twd=Decimal(8),
                mean_daily_net_twd=Decimal("0.112676056338028169"),
                terminal_executed_turnover_twd=Decimal(1000),
                terminal_net_bp_of_turnover=Decimal(80),
                open_execution_actual_cost_twd=Decimal(0),
                hedge_priced_numerator=1,
                hedge_priced_denominator=1,
                monthly_net=(),
            )
            for scenario_id in SCENARIO_IDS
        )

    @staticmethod
    def _open_valuation(
        scenario_id: str,
        *,
        open_count: int = 0,
        publishable: bool = True,
    ) -> SimpleNamespace:
        rows = tuple(
            SimpleNamespace(
                priced=publishable,
                unpriced_reason=(None if publishable else "spot_insufficient_depth"),
            )
            for _ in range(open_count)
        )
        aggregate = (
            S1OpenPositionValuation(
                position_count=open_count,
                valuation_method_id="s1_causal_executable_fifo_common_horizon_v1",
                valuation_asof_id="20260813T132000+0800/test",
                comparable_across_scenarios=True,
                gross_mark_pnl_twd=Decimal(0),
                remaining_exit_cost_twd=Decimal(0),
                net_mark_pnl_twd=Decimal(0),
            )
            if publishable
            else None
        )
        return SimpleNamespace(
            scenario_id=scenario_id,
            rows=rows,
            incurred_open_execution_cost_twd=Decimal(0),
            aggregate=aggregate,
        )

    @classmethod
    def _open_valuations(cls) -> dict[str, SimpleNamespace]:
        return {
            scenario_id: cls._open_valuation(scenario_id)
            for scenario_id in SCENARIO_IDS
        }

    @staticmethod
    def _diagnostics() -> dict[str, dict[str, object]]:
        return {
            scenario_id: {
                "lookup_supported_cells": 62_552,
                "common_lookup_cells": 62_552,
                "decision_economic_gate_open": 1,
                "decision_economic_checks": 1,
                "actual_send_economic_gate_open": 1,
                "actual_send_economic_checks": 1,
                "candidate_intents": 1,
                "blocked_candidate_intents": 0,
                "reservation_attempts": 1,
                "cap_blocked_attempts": 0,
                "admitted_orders": 1,
                "sent_entry_orders": 1,
                "makerfill_supported_orders": 1,
                "actual_active_entry_fills": 1,
                "economic_gate_sent_expected_margin_bp_p50": 1.0,
                "economic_gate_sent_expected_margin_bp_p95": 2.0,
                "economic_gate_sent_modeled_cost_twd_p50": 100.0,
                "economic_gate_sent_modeled_cost_twd_p95": 120.0,
                "active_fill_latency_ms_p50": 1.0,
                "active_fill_latency_ms_p95": 2.0,
                "target_rank_counts": {
                    "BID1": 2,
                    "BID2": 3,
                    "not_BID1_or_BID2": 4,
                },
                "exit_desired_withdrawal_reason_counts": {
                    "gate:future_empty_book_side": 2,
                    "safety_cutoff": 1,
                },
                "exit_cutoff_sessions": 71,
                "exit_drain_barrier_sessions": 71,
                "risk_groups": [
                    {
                        "stage": "entry",
                        "risk_kind": "hedge",
                        "created": 10,
                        "actual_send": 8,
                        "timeout": 2,
                        "on_time": 5,
                        "delayed": 3,
                        "arrival_reference_available": 9,
                        "arrival_reference_denominator": 10,
                        "delay_ms_p50": 0.0,
                        "delay_ms_p95": 25.0,
                        "delay_ms_max": 40.0,
                        "adverse_slippage_bp_p50": 0.1,
                        "adverse_slippage_bp_p95": 0.4,
                        "adverse_slippage_bp_max": 0.5,
                        "adverse_slippage_sample_count": 7,
                        "initial_gate_reason_counts": {"none": 10},
                    }
                ],
                "mean_daily_spot_requests_sent": 1.0,
                "mean_daily_future_requests_sent": 1.0,
                "spot_rolling_request_peak": 1,
                "future_rolling_request_peak": 1,
                "global_cap_peak_twd": 1,
                "product_cap_peak_twd": 1,
                "carry_notional_days_twd": 0,
                "final_carry_notional_twd": 0,
                "final_carry_positions": 0,
                "final_naked_unresolved_positions": 0,
                "final_naked_unresolved_notional_twd": 0,
                "naked_unresolved_notional_twd_max": 0,
                "global_committed_bucket_peaks_twd": {
                    "working_unfilled": 101,
                    "entry_partial": 102,
                    "hedge_pending": 103,
                    "paired_open": 104,
                    "exit_in_progress": 105,
                    "total_committed_notional_twd": 106,
                },
            }
            for scenario_id in SCENARIO_IDS
        }

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
            economic_gate_estimate_records=(),
            economic_gate_event_records=(),
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
