from __future__ import annotations

import unittest
from dataclasses import replace
from decimal import Decimal, localcontext

from maker.src.quote_fill.layered import EventCursor
from maker.src.quote_fill.s1_accounting import (
    InitiatingExecutionAllocation,
    S1AccountingLedger,
)
from maker.src.quote_fill.s1_event_loop import RiskEvent
from maker.src.quote_fill.s1_performance import (
    S1DailyReplaySummary,
    S1PerformanceError,
    aggregate_s1_scenario_metrics,
    build_s1_daily_replay_summary_from_risk_events,
)


class FactBuilder:
    def __init__(self) -> None:
        self.ledger = S1AccountingLedger(require_route_roles=True)
        self.sequence = 0
        self.terminals: dict[str, object] = {}

    def cursor(self) -> EventCursor:
        self.sequence += 1
        return EventCursor(self.sequence, 0, 0)

    def common(
        self,
        position_id: str,
        execution_id: str,
        execution_date: str,
        *,
        role: str,
        route_role: str,
        truth: str,
        source_execution_id: str | None = None,
    ) -> dict[str, object]:
        linked = role in ("hedge", "rollback")
        allocations = (
            ()
            if source_execution_id is None
            else (InitiatingExecutionAllocation(source_execution_id, 2000),)
        )
        return {
            "execution_id": execution_id,
            "position_id": position_id,
            "value_code": f"value-{position_id}",
            "scenario_id": "q95",
            "capacity_id": f"capacity-{position_id}",
            "role": role,
            "execution_truth": truth,
            "execution_source_id": f"source-{execution_id}",
            "request_id": f"request-{execution_id}",
            "hedge_intent_id": f"risk-{execution_id}" if linked else None,
            "initiating_execution_id": source_execution_id,
            "initiating_execution_allocations": allocations,
            "execution_date": execution_date,
            "cursor": self.cursor(),
            "route_role": route_role,
        }

    def open_pair(
        self,
        position_id: str,
        execution_date: str,
        *,
        truth: str = "exact",
    ) -> None:
        maker_id = f"{position_id}-entry-maker"
        self.ledger.record_spot_execution(
            **self.common(
                position_id,
                maker_id,
                execution_date,
                role="normal",
                route_role="entry_maker",
                truth=truth,
            ),
            side="buy",
            price=100.1,
            shares=2000,
        )
        hedge = self.ledger.record_future_execution(
            **self.common(
                position_id,
                f"{position_id}-entry-hedge",
                execution_date,
                role="hedge",
                route_role="entry_hedge",
                truth="exact",
                source_execution_id=maker_id,
            ),
            side="sell",
            price=103.2,
            contracts=1,
            share_equivalent=2000,
        )
        establishment_cursor = self.cursor()
        self.ledger.establish_position(
            establishment_id=f"{position_id}-established",
            position_id=position_id,
            establishment_date=execution_date,
            cursor=establishment_cursor,
            position_established_ns=establishment_cursor.recv_time_ns,
            capacity_transition_id=f"{position_id}-paired-capacity",
        )
        self.assert_after(hedge.cursor, establishment_cursor)

    def close_pair(
        self,
        position_id: str,
        execution_date: str,
        *,
        spot_price: float = 101.1,
        future_price: float = 101.0,
        terminal_outcome: str = "exit_maker_flat",
    ) -> None:
        maker_id = f"{position_id}-exit-maker"
        self.ledger.record_spot_execution(
            **self.common(
                position_id,
                maker_id,
                execution_date,
                role="normal",
                route_role="exit_maker",
                truth="exact",
            ),
            side="sell",
            price=spot_price,
            shares=2000,
        )
        self.ledger.record_future_execution(
            **self.common(
                position_id,
                f"{position_id}-exit-hedge",
                execution_date,
                role="hedge",
                route_role="exit_hedge",
                truth="exact",
                source_execution_id=maker_id,
            ),
            side="buy",
            price=future_price,
            contracts=1,
            share_equivalent=2000,
        )
        terminal = self.ledger.seal_terminal(
            terminal_id=f"{position_id}-terminal",
            position_id=position_id,
            terminal_date=execution_date,
            cursor=self.cursor(),
            terminal_outcome=terminal_outcome,
            capacity_release_transition_id=f"{position_id}-exit-release",
        )
        self.terminals[position_id] = terminal

    def rollback(self, position_id: str, execution_date: str) -> None:
        maker_id = f"{position_id}-entry-maker"
        self.ledger.record_spot_execution(
            **self.common(
                position_id,
                maker_id,
                execution_date,
                role="normal",
                route_role="entry_maker",
                truth="approximate",
            ),
            side="buy",
            price=100.1,
            shares=2000,
        )
        self.ledger.record_spot_execution(
            **self.common(
                position_id,
                f"{position_id}-entry-rollback",
                execution_date,
                role="rollback",
                route_role="entry_rollback",
                truth="exact",
                source_execution_id=maker_id,
            ),
            side="sell",
            price=99.8,
            shares=2000,
        )
        terminal = self.ledger.seal_terminal(
            terminal_id=f"{position_id}-terminal",
            position_id=position_id,
            terminal_date=execution_date,
            cursor=self.cursor(),
            terminal_outcome="entry_emergency_rollback_flat",
            capacity_release_transition_id=f"{position_id}-rollback-release",
        )
        self.terminals[position_id] = terminal

    def unresolved_entry(self, position_id: str, execution_date: str) -> None:
        self.ledger.record_spot_execution(
            **self.common(
                position_id,
                f"{position_id}-entry-maker",
                execution_date,
                role="normal",
                route_role="entry_maker",
                truth="approximate",
            ),
            side="buy",
            price=100.1,
            shares=2000,
        )

    def expiry(self, position_id: str, expiry_date: str) -> None:
        source_cursor = self.cursor()
        mark = self.ledger.record_expiry_accounting_mark(
            mark_id=f"{position_id}-expiry",
            position_id=position_id,
            expiry_date=expiry_date,
            cursor=self.cursor(),
            spot_close_source_id=f"{position_id}-spot-close",
            spot_close_source_cursor=source_cursor,
            spot_close_price=101.4,
            capacity_release_transition_id=f"{position_id}-expiry-release",
        )
        self.terminals[position_id] = mark

    def assert_after(self, earlier: EventCursor, later: EventCursor) -> None:
        if later <= earlier:
            raise AssertionError("test fact cursor failed to advance")


def complete_portfolio() -> FactBuilder:
    builder = FactBuilder()
    builder.open_pair("same", "20260825", truth="exact")
    builder.close_pair("same", "20260825")
    builder.open_pair("cross", "20260825", truth="approximate")
    builder.open_pair("expiry", "20260825", truth="exact")
    builder.rollback("rollback", "20260825")
    builder.open_pair("open", "20260825", truth="exact")
    builder.unresolved_entry("unresolved", "20260825")
    builder.close_pair("cross", "20260826", spot_price=102.3, future_price=100.7)
    builder.expiry("expiry", "20260827")
    builder.ledger.verify()
    return builder


def risk_event(
    risk_id: str,
    event_type: str,
    cursor: EventCursor,
    *,
    stage: str,
    kind: str,
    status: str,
    position_id: str | None = None,
    product_id: str | None = None,
    request_id: str | None = None,
) -> RiskEvent:
    return RiskEvent(
        risk_id=risk_id,
        request_id=request_id or f"request-{risk_id}",
        position_id=position_id or f"position-{risk_id}",
        product_id=product_id or f"product-{risk_id}",
        risk_kind=kind,
        event_type=event_type,
        cursor=cursor,
        status=status,
        gate_reason=None,
        arrival_reference=object(),
        stage=stage,
    )


class S1PerformanceTest(unittest.TestCase):
    def test_daily_risk_summary_counts_entry_exit_rollback_and_timeout(self) -> None:
        events = (
            risk_event(
                "entry-hedge",
                "created",
                EventCursor(1),
                stage="entry",
                kind="hedge",
                status="waiting",
            ),
            risk_event(
                "entry-hedge",
                "evaluated",
                EventCursor(2),
                stage="entry",
                kind="hedge",
                status="send_eligible",
            ),
            risk_event(
                "entry-hedge",
                "actual_send",
                EventCursor(2),
                stage="entry",
                kind="hedge",
                status="hedge_sent",
            ),
            risk_event(
                "exit-hedge",
                "created",
                EventCursor(3),
                stage="exit",
                kind="hedge",
                status="waiting",
            ),
            risk_event(
                "exit-hedge",
                "evaluated",
                EventCursor(4),
                stage="exit",
                kind="hedge",
                status="waiting",
            ),
            risk_event(
                "exit-hedge",
                "timeout",
                EventCursor(5),
                stage="exit",
                kind="hedge",
                status="hedge_retry_timeout",
            ),
            risk_event(
                "exit-rollback",
                "created",
                EventCursor(5),
                stage="exit",
                kind="rollback",
                status="waiting",
            ),
            risk_event(
                "exit-rollback",
                "evaluated",
                EventCursor(6),
                stage="exit",
                kind="rollback",
                status="send_eligible",
            ),
            risk_event(
                "exit-rollback",
                "actual_send",
                EventCursor(6),
                stage="exit",
                kind="rollback",
                status="rollback_sent",
            ),
        )

        summary = build_s1_daily_replay_summary_from_risk_events(
            events,
            date="20260825",
            scenario_id="q95",
        )

        self.assertEqual(summary.date, "20260825")
        self.assertEqual(summary.scenario_id, "q95")
        self.assertEqual(summary.all_risk_hedge_priced_numerator, 2)
        self.assertEqual(summary.all_risk_hedge_priced_denominator, 3)

    def test_daily_risk_summary_accepts_empty_day(self) -> None:
        summary = build_s1_daily_replay_summary_from_risk_events(
            (),
            date="20260825",
            scenario_id="q95",
        )

        self.assertEqual(summary.all_risk_hedge_priced_numerator, 0)
        self.assertEqual(summary.all_risk_hedge_priced_denominator, 0)

    def test_daily_risk_summary_rejects_lifecycle_tampering(self) -> None:
        created = risk_event(
            "entry-hedge",
            "created",
            EventCursor(1),
            stage="entry",
            kind="hedge",
            status="waiting",
        )
        evaluated = risk_event(
            "entry-hedge",
            "evaluated",
            EventCursor(2),
            stage="entry",
            kind="hedge",
            status="send_eligible",
        )
        sent = risk_event(
            "entry-hedge",
            "actual_send",
            EventCursor(2),
            stage="entry",
            kind="hedge",
            status="hedge_sent",
        )
        cases = {
            "orphan": ((evaluated, sent), "orphaned"),
            "duplicate created": (
                (created, replace(created, cursor=EventCursor(2)), sent),
                "multiple created",
            ),
            "double terminal": (
                (
                    created,
                    evaluated,
                    sent,
                    replace(
                        sent,
                        event_type="timeout",
                        cursor=EventCursor(3),
                        status="hedge_retry_timeout",
                    ),
                ),
                "multiple terminal",
            ),
            "missing terminal": ((created, evaluated), "missing a terminal"),
            "cursor regression": (
                (created, replace(evaluated, cursor=EventCursor(0)), sent),
                "cursor moved backwards",
            ),
            "stage drift": (
                (created, replace(evaluated, stage="exit"), sent),
                "stage drifted",
            ),
            "position drift": (
                (created, replace(evaluated, position_id="different"), sent),
                "position_id drifted",
            ),
            "product drift": (
                (created, replace(evaluated, product_id="different"), sent),
                "product_id drifted",
            ),
            "kind drift": (
                (created, replace(evaluated, risk_kind="rollback"), sent),
                "risk_kind drifted",
            ),
        }
        for name, (events, message) in cases.items():
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(S1PerformanceError, message),
            ):
                build_s1_daily_replay_summary_from_risk_events(
                    events,
                    date="20260825",
                    scenario_id="q95",
                )

    def test_full_scenario_partitions_completion_net_and_open_costs(self) -> None:
        builder = complete_portfolio()
        metrics = aggregate_s1_scenario_metrics(
            builder.ledger.facts,
            ("20260901", "20260827", "20260825", "20260826"),
        )

        self.assertEqual(metrics.scenario_id, "q95")
        self.assertEqual(metrics.reporting_sessions, 4)
        self.assertEqual(metrics.entry_positions, 6)
        self.assertEqual(metrics.entry_fill_exact, 3)
        self.assertEqual(metrics.entry_fill_approximate, 3)
        self.assertEqual(metrics.same_day_exit_maker_flat, 1)
        self.assertEqual(metrics.cross_day_exit_maker_flat, 1)
        self.assertEqual(metrics.expiry_marks, 1)
        self.assertEqual(metrics.entry_rollbacks, 1)
        self.assertEqual(metrics.other_executable_terminals, 0)
        self.assertEqual(metrics.open_or_unresolved, 2)
        self.assertEqual(metrics.terminal_coverage_numerator, 4)
        self.assertEqual(metrics.terminal_coverage_denominator, 6)
        with localcontext() as context:
            context.prec = 50
            self.assertEqual(metrics.completion_rate, Decimal(1) / Decimal(6))
            self.assertEqual(metrics.terminal_coverage, Decimal(4) / Decimal(6))
        self.assertEqual(metrics.approx_screen_completion_numerator, 0)
        self.assertEqual(metrics.approx_screen_completion_denominator, 3)
        self.assertEqual(metrics.approx_screen_completion_rate, Decimal(0))
        self.assertEqual(metrics.entry_hedge_success_numerator, 4)
        self.assertEqual(metrics.entry_hedge_success_denominator, 6)
        self.assertEqual(
            metrics.hedge_pricing_coverage_source,
            "entry_hedge_success_proxy",
        )

        executable = sum(
            (
                Decimal(str(builder.terminals[position].realized_net_twd))
                for position in ("same", "cross", "rollback")
            ),
            Decimal(0),
        )
        expiry = Decimal(str(builder.terminals["expiry"].realized_net_twd))
        self.assertEqual(metrics.executable_terminal_net_twd, executable)
        self.assertEqual(metrics.expiry_mark_net_twd, expiry)
        self.assertEqual(metrics.total_net_twd, executable + expiry)
        self.assertEqual(metrics.mean_daily_net_twd, (executable + expiry) / 4)

        open_rows = [
            row
            for row in builder.ledger.rows
            if row.position_id in {"open", "unresolved"}
        ]
        expected_open_cost = sum(
            (Decimal(str(row.total_cost_twd)) for row in open_rows), Decimal(0)
        )
        open_cashflow = sum(
            (Decimal(str(row.signed_cashflow_twd)) for row in open_rows),
            Decimal(0),
        )
        self.assertEqual(metrics.open_execution_actual_cost_twd, expected_open_cost)
        self.assertLess(expected_open_cost, abs(open_cashflow))
        self.assertNotEqual(metrics.total_net_twd, executable + expiry + open_cashflow)

        self.assertEqual(
            tuple(row.calendar_key for row in metrics.daily_net),
            ("20260825", "20260826", "20260827", "20260901"),
        )
        self.assertEqual(metrics.daily_net[-1].total_net_twd, Decimal(0))
        self.assertEqual(
            tuple(row.calendar_key for row in metrics.monthly_net),
            ("202608", "202609"),
        )
        self.assertEqual(metrics.monthly_net[1].total_net_twd, Decimal(0))
        self.assertEqual(
            sum((row.total_net_twd for row in metrics.daily_net), Decimal(0)),
            metrics.total_net_twd,
        )

        exact, approximate = metrics.by_entry_fill_truth
        self.assertEqual(exact.entry_fill_truth, "exact")
        self.assertEqual(exact.entry_positions, 3)
        self.assertEqual(exact.same_day_exit_maker_flat, 1)
        self.assertEqual(exact.expiry_marks, 1)
        self.assertEqual(exact.open_or_unresolved, 1)
        self.assertEqual(approximate.entry_fill_truth, "approximate")
        self.assertEqual(approximate.entry_positions, 3)
        self.assertEqual(approximate.cross_day_exit_maker_flat, 1)
        self.assertEqual(approximate.entry_rollbacks, 1)
        self.assertEqual(approximate.open_or_unresolved, 1)

        ranking = metrics.to_scenario_metrics()
        self.assertEqual(ranking.completion_numerator, 1)
        self.assertEqual(ranking.completion_denominator, 6)
        self.assertEqual(ranking.total_net_twd, metrics.total_net_twd)
        self.assertEqual(ranking.reporting_sessions, 4)
        self.assertEqual(ranking.hedge_priced_numerator, 4)
        self.assertEqual(ranking.hedge_priced_denominator, 6)

        approx_ranking = metrics.to_approx_screen_scenario_metrics()
        self.assertEqual(approx_ranking.completion_numerator, 0)
        self.assertEqual(approx_ranking.completion_denominator, 3)
        self.assertEqual(approx_ranking.total_net_twd, metrics.total_net_twd)
        self.assertEqual(approx_ranking.reporting_sessions, 4)
        self.assertEqual(approx_ranking.hedge_priced_numerator, 4)
        self.assertEqual(approx_ranking.hedge_priced_denominator, 6)

    def test_explicit_daily_all_risk_coverage_overrides_entry_proxy(self) -> None:
        builder = complete_portfolio()
        summaries = tuple(
            S1DailyReplaySummary(date, "q95", numerator, denominator)
            for date, numerator, denominator in (
                ("20260825", 5, 7),
                ("20260826", 2, 2),
                ("20260827", 1, 1),
                ("20260901", 0, 0),
            )
        )
        metrics = aggregate_s1_scenario_metrics(
            builder.ledger.facts,
            ("20260825", "20260826", "20260827", "20260901"),
            daily_replay_summaries=summaries,
        )
        self.assertEqual(metrics.entry_hedge_success_numerator, 4)
        self.assertEqual(metrics.entry_hedge_success_denominator, 6)
        self.assertEqual(metrics.hedge_priced_numerator, 8)
        self.assertEqual(metrics.hedge_priced_denominator, 10)
        self.assertEqual(
            metrics.hedge_pricing_coverage_source,
            "daily_all_risk_pricing",
        )
        ranking = metrics.to_scenario_metrics()
        self.assertEqual(ranking.hedge_priced_numerator, 8)
        self.assertEqual(ranking.hedge_priced_denominator, 10)

    def test_other_executable_terminal_contributes_net_not_completion(self) -> None:
        builder = FactBuilder()
        builder.open_pair("hard-flat", "20260825")
        builder.close_pair(
            "hard-flat",
            "20260825",
            terminal_outcome="aggressive_hard_flat",
        )
        terminal = builder.terminals["hard-flat"]
        metrics = aggregate_s1_scenario_metrics(
            builder.ledger.facts,
            ("20260825",),
        )
        self.assertEqual(metrics.same_day_exit_maker_flat, 0)
        self.assertEqual(metrics.other_executable_terminals, 1)
        self.assertEqual(metrics.completion_rate, Decimal(0))
        self.assertEqual(
            metrics.executable_terminal_net_twd,
            Decimal(str(terminal.realized_net_twd)),
        )
        self.assertEqual(metrics.expiry_mark_net_twd, Decimal(0))

    def test_zero_entries_has_defined_daily_net_and_undefined_rates(self) -> None:
        metrics = aggregate_s1_scenario_metrics(
            (),
            ("20260825", "20260826"),
            scenario_id="fixed30",
        )
        self.assertEqual(metrics.entry_positions, 0)
        self.assertIsNone(metrics.completion_rate)
        self.assertIsNone(metrics.terminal_coverage)
        self.assertIsNone(metrics.entry_hedge_success_rate)
        self.assertEqual(metrics.mean_daily_net_twd, Decimal(0))
        self.assertEqual(metrics.total_net_twd, Decimal(0))
        ranking = metrics.to_scenario_metrics()
        self.assertFalse(ranking.completion_defined)
        self.assertFalse(ranking.hedge_pricing_coverage_defined)

    def test_exact_only_entries_leave_approx_screen_rate_undefined(self) -> None:
        builder = FactBuilder()
        builder.open_pair("exact", "20260825", truth="exact")
        builder.close_pair("exact", "20260825")
        metrics = aggregate_s1_scenario_metrics(
            builder.ledger.facts,
            ("20260825",),
        )

        self.assertEqual(metrics.completion_rate, Decimal(1))
        self.assertEqual(metrics.approx_screen_completion_numerator, 0)
        self.assertEqual(metrics.approx_screen_completion_denominator, 0)
        self.assertIsNone(metrics.approx_screen_completion_rate)
        ranking = metrics.to_approx_screen_scenario_metrics()
        self.assertEqual(ranking.completion_numerator, 0)
        self.assertEqual(ranking.completion_denominator, 0)
        self.assertFalse(ranking.completion_defined)

    def test_replay_and_scenario_contracts_fail_closed(self) -> None:
        builder = complete_portfolio()
        facts = builder.ledger.facts
        with self.assertRaisesRegex(S1PerformanceError, "replay verification"):
            aggregate_s1_scenario_metrics(
                (replace(facts[0], sequence=99), *facts[1:]),
                ("20260825", "20260826", "20260827"),
            )
        with self.assertRaisesRegex(S1PerformanceError, "one scenario"):
            aggregate_s1_scenario_metrics(
                facts,
                ("20260825", "20260826", "20260827"),
                scenario_id="fixed30",
            )
        with self.assertRaisesRegex(S1PerformanceError, "outside reporting_dates"):
            aggregate_s1_scenario_metrics(
                facts,
                ("20260825", "20260826"),
            )

    def test_unique_entry_and_complete_daily_summary_are_enforced(self) -> None:
        builder = FactBuilder()
        builder.ledger.record_spot_execution(
            **builder.common(
                "duplicate",
                "duplicate-entry-a",
                "20260825",
                role="normal",
                route_role="entry_maker",
                truth="exact",
            ),
            side="buy",
            price=100.0,
            shares=2000,
        )
        builder.ledger.record_spot_execution(
            **builder.common(
                "duplicate",
                "duplicate-entry-b",
                "20260825",
                role="normal",
                route_role="entry_maker",
                truth="exact",
            ),
            side="buy",
            price=100.0,
            shares=2000,
        )
        with self.assertRaisesRegex(S1PerformanceError, "exactly one entry_maker"):
            aggregate_s1_scenario_metrics(
                builder.ledger.facts,
                ("20260825",),
            )

        complete = complete_portfolio()
        incomplete = (
            S1DailyReplaySummary("20260825", "q95", 1, 1),
            S1DailyReplaySummary("20260826", "q95", 1, 1),
        )
        with self.assertRaisesRegex(S1PerformanceError, "cover every reporting"):
            aggregate_s1_scenario_metrics(
                complete.ledger.facts,
                ("20260825", "20260826", "20260827"),
                daily_replay_summaries=incomplete,
            )

        omitted_entry_risks = tuple(
            S1DailyReplaySummary(date, "q95", 0, 0)
            for date in ("20260825", "20260826", "20260827")
        )
        with self.assertRaisesRegex(S1PerformanceError, "omits entry hedge risks"):
            aggregate_s1_scenario_metrics(
                complete.ledger.facts,
                ("20260825", "20260826", "20260827"),
                daily_replay_summaries=omitted_entry_risks,
            )

    def test_terminal_date_cannot_precede_entry_date(self) -> None:
        builder = complete_portfolio()
        facts = list(builder.ledger.facts)
        terminal_index = next(
            index
            for index, fact in enumerate(facts)
            if getattr(fact, "terminal_id", None) == "cross-terminal"
        )
        facts[terminal_index] = replace(facts[terminal_index], terminal_date="20260824")
        with self.assertRaisesRegex(S1PerformanceError, "replay verification"):
            aggregate_s1_scenario_metrics(
                tuple(facts),
                ("20260824", "20260825", "20260826", "20260827"),
            )


if __name__ == "__main__":
    unittest.main()
