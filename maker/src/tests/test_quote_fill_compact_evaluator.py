from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import polars as pl

from maker.src.quote_fill.compact_evaluator import (
    CompactEvaluatorConfig,
    MakerFillFastAdapter,
    _OUTCOME_BASE_SCHEMA,
    _empty_outcomes,
    _normalize_outcomes,
    _select_fill,
    build_compact_candidates,
)
from maker.src.quote_fill.compact_frozen_facts import (
    build_compact_from_execution_actions,
    build_compact_state_changes,
    load_frozen_compact_product_days,
    summarize_frozen_product_days,
    _verify_action_partition,
)
from maker.src.quote_fill.compact_exit import (
    CAPACITY_BASIS,
    allocate_fifo_exact_capacity,
    build_independent_close_opportunities,
    canonical_capacity_event_id,
    canonical_source_trade_event_id,
    simulate_fifo_residual_capacity_contract,
)
from maker.src.quote_fill.indexed_replay import IndexedTradeReplay
from maker.src.quote_fill.layered import EventCursor
from maker.src.quote_fill.replay import IndependentOrderWindow, TradeEvent


def _observations(
    route: str,
    rows: list[tuple[int, int, int | None, bool, bool]],
) -> pl.DataFrame:
    maker_market = "spot" if route.startswith("spot_") else "future"
    maker_side = "bid" if "_bid_" in route else "ask"
    records = []
    for recv_ns, epoch, tick, gate, base in rows:
        records.append(
            {
                "Date": "20260714",
                "ValueCode": "1513",
                "QuoteCode": "SDFG6",
                "route": route,
                "boundary_quantile": 50,
                "spread_pair_epoch": epoch,
                "base_epoch_transition": base,
                "event_source": "spot" if base else "future",
                "recv_time_ns": recv_ns,
                "cursor_event_sequence": 2 if base else 1,
                "cursor_row_index": recv_ns,
                "absolute_target_price": float(tick) if tick else None,
                "absolute_target_tick": tick,
                "target_rank": "BID1" if maker_side == "bid" else "ASK1",
                "initial_queue_ahead": 1 if gate else None,
                "gate_open": gate,
                "admission_reason": "eligible" if gate else "trial_match",
                "maker_market": maker_market,
                "maker_side": maker_side,
            }
        )
    return pl.from_dicts(records, infer_schema_length=None)


class CompactFirstPassageTest(unittest.TestCase):
    def test_bid_next_smaller_gate_and_epoch_candidate_contract(self) -> None:
        frame = _observations(
            "spot_bid_future_taker",
            [
                (10, 1, 100, True, True),
                (20, 1, 101, True, False),
                (30, 1, 100, True, False),
                (40, 1, 102, True, False),
                (50, 1, None, False, False),
                (60, 2, 100, True, True),
            ],
        )
        candidates = build_compact_candidates(
            frame,
            date="20260714",
            value_code="1513",
            route="spot_bid_future_taker",
            boundary_quantile=50,
            cutoff_cursor=EventCursor(100, 3, 0),
        )
        self.assertEqual(
            [item.decision.observation.absolute_target_tick for item in candidates],
            [100, 101, 102, 100],
        )
        self.assertEqual(
            [
                (item.stop_cursor.recv_time_ns, item.stop_type, item.stop_reason)
                for item in candidates
            ],
            [
                (50, "gate_invalid", "trial_match"),
                (30, "target_retreat", "target_retreat"),
                (50, "gate_invalid", "trial_match"),
                (100, "session_cutoff", "session_cutoff"),
            ],
        )
        self.assertEqual(
            [item.decision.decision_reason for item in candidates],
            [
                "spread_pair_epoch_transition",
                "forward_new_absolute_tick",
                "forward_new_absolute_tick",
                "spread_pair_epoch_transition",
            ],
        )

    def test_ask_next_greater_is_less_aggressive_stop(self) -> None:
        frame = _observations(
            "future_ask_spot_taker",
            [
                (10, 1, 105, True, True),
                (20, 1, 104, True, False),
                (30, 1, 105, True, False),
                (40, 1, None, False, False),
            ],
        )
        candidates = build_compact_candidates(
            frame,
            date="20260714",
            value_code="1513",
            route="future_ask_spot_taker",
            boundary_quantile=50,
            cutoff_cursor=EventCursor(100, 3, 0),
        )
        self.assertEqual(
            [item.decision.observation.absolute_target_tick for item in candidates],
            [105, 104],
        )
        self.assertEqual(
            [
                (item.stop_cursor.recv_time_ns, item.stop_type, item.stop_reason)
                for item in candidates
            ],
            [
                (40, "gate_invalid", "trial_match"),
                (30, "target_retreat", "target_retreat"),
            ],
        )


class MakerFillFastAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = MakerFillFastAdapter(
            pl.DataFrame(
                {
                    "QuoteCode": ["1513"],
                    "ChannelSeq": [7],
                    "Ask1_FillSeconds": [float("nan")],
                    "Ask2_FillSeconds": [4.0],
                    "Bid1_FillSeconds": [2.0],
                    "Bid2_FillSeconds": [5.0],
                }
            )
        )

    def test_mapping_is_exact_but_outcome_and_cursor_are_approximate(self) -> None:
        start = 1_500_000_000
        window = IndependentOrderWindow(
            "g1",
            "bid",
            100,
            EventCursor(start, 1, 0),
            EventCursor(3_500_000_000, 1, 0),
            5,
            "target_retreat",
        )
        label = self.adapter.label(
            window,
            {
                "maker_market": "spot",
                "target_rank": "BID1",
                "maker_snapshot_sequence": 7,
                "maker_snapshot_recv_time_ns": 1_000_000_000,
            },
            value_code="1513",
            intended_quantity=2,
        )
        self.assertIsNotNone(label)
        assert label is not None
        self.assertTrue(label.makerfill_mapping_exact)
        self.assertFalse(label.outcome_exact)
        self.assertFalse(label.fill_cursor_exact)
        self.assertTrue(label.full_fill)
        self.assertEqual(label.full_fill_cursor.recv_time_ns, 3_000_000_000)
        self.assertEqual(label.indexed_query_count, 0)

    def test_spot_level_three_fails_closed_without_index_query(self) -> None:
        window = IndependentOrderWindow(
            "g2",
            "bid",
            100,
            EventCursor(1_000_000_000, 2, 7),
            EventCursor(4_000_000_000, 1, 0),
            5,
            "target_retreat",
        )
        fill = _select_fill(
            window,
            {
                "maker_market": "spot",
                "target_rank": "BID3",
                "maker_snapshot_sequence": 7,
                "maker_snapshot_recv_time_ns": 1_000_000_000,
            },
            value_code="1513",
            intended_quantity=2,
            config=CompactEvaluatorConfig(),
            makerfill_adapter=self.adapter,
            replay=IndexedTradeReplay(()),
        )
        self.assertEqual(fill.backend, "excluded_fail_closed")
        self.assertIsNone(fill.full_fill)
        self.assertEqual(fill.indexed_query_count, 0)

    def test_future_maker_uses_one_indexed_quantity_query(self) -> None:
        window = IndependentOrderWindow(
            "g3",
            "ask",
            100,
            EventCursor(10, 1, 0),
            EventCursor(30, 1, 0),
            0,
            "target_retreat",
        )
        replay = IndexedTradeReplay(
            (TradeEvent(EventCursor(20, 1, 1), 101, 1),)
        )
        fill = _select_fill(
            window,
            {"maker_market": "future", "target_rank": "ASK1"},
            value_code="1513",
            intended_quantity=1,
            config=CompactEvaluatorConfig(),
            makerfill_adapter=self.adapter,
            replay=replay,
        )
        self.assertEqual(fill.backend, "indexed_trade_replay")
        self.assertEqual(fill.indexed_query_count, 1)
        self.assertTrue(fill.full_fill)
        self.assertTrue(fill.fill_cursor_exact)

    def test_nan_is_no_eod_fill_but_null_and_infinity_fail_closed(self) -> None:
        window = IndependentOrderWindow(
            "g4",
            "ask",
            100,
            EventCursor(1_000_000_000, 2, 7),
            EventCursor(9_000_000_000, 3, 0),
            1,
            "session_cutoff",
        )
        metadata = {
            "maker_market": "spot",
            "target_rank": "ASK1",
            "maker_snapshot_sequence": 7,
            "maker_snapshot_recv_time_ns": 1_000_000_000,
        }
        nan_label = self.adapter.label(
            window, metadata, value_code="1513", intended_quantity=2
        )
        self.assertIsNotNone(nan_label)
        assert nan_label is not None
        self.assertFalse(nan_label.full_fill)
        for invalid in (None, float("inf"), float("-inf")):
            adapter = MakerFillFastAdapter(
                pl.DataFrame(
                    {
                        "QuoteCode": ["1513"],
                        "ChannelSeq": [7],
                        "Ask1_FillSeconds": [invalid],
                        "Ask2_FillSeconds": [1.0],
                        "Bid1_FillSeconds": [1.0],
                        "Bid2_FillSeconds": [1.0],
                    },
                    schema_overrides={"Ask1_FillSeconds": pl.Float64},
                )
            )
            self.assertIsNone(
                adapter.label(
                    window, metadata, value_code="1513", intended_quantity=2
                )
            )

    def test_makerfill_same_ns_after_stop_priority_is_not_a_fill(self) -> None:
        window = IndependentOrderWindow(
            "g5",
            "bid",
            100,
            EventCursor(1_500_000_000, 1, 0),
            EventCursor(3_000_000_000, 1, 0),
            1,
            "target_retreat",
        )
        label = self.adapter.label(
            window,
            {
                "maker_market": "spot",
                "target_rank": "BID1",
                "maker_snapshot_sequence": 7,
                "maker_snapshot_recv_time_ns": 1_000_000_000,
            },
            value_code="1513",
            intended_quantity=2,
        )
        self.assertIsNotNone(label)
        assert label is not None
        self.assertFalse(label.full_fill)


def _formal_action(rank: str, *, full: bool, generation: str) -> dict[str, object]:
    full_ns = 2_000_000_000 if full else None
    return {
        "Date": "20260714",
        "ValueCode": "1513",
        "QuoteCode": "SDFG6",
        "route": "spot_bid_future_taker",
        "maker_market": "spot",
        "maker_side": "bid",
        "boundary_quantile": 50,
        "spread_pair_epoch": 1,
        "raw_order_fact_id": f"raw-{generation}",
        "policy_generation_id": generation,
        "target_price_tick": 100,
        "target_price": 100.0,
        "target_rank_at_submit": rank,
        "initial_queue_ahead": 1,
        "queue_known": True,
        "intended_quantity": 2,
        "submit_recv_time_ns": 1_000_000_000,
        "submit_event_sequence": 2,
        "submit_row_index": 7,
        "nominal_stop_recv_time_ns": 3_000_000_000,
        "nominal_stop_reason": "target_retreat",
        "first_fill_recv_time_ns": full_ns,
        "first_fill_event_sequence": 2 if full else None,
        "first_fill_row_index": 8 if full else None,
        "full_fill_recv_time_ns": full_ns,
        "full_fill_event_sequence": 2 if full else None,
        "full_fill_row_index": 8 if full else None,
        "known_filled_quantity": 2 if full else 0,
        "any_fill": full,
        "full_fill": full,
        "partial_fill": False,
        "trade_through_fill": False,
        "terminal_recv_time_ns": full_ns or 3_000_000_000,
        "terminal_reason": "full_fill" if full else "target_retreat",
        "cancel_required": not full,
        "entry_hedge_status": "executable" if full else None,
        "entry_hedge_decision_time_ns": (
            full_ns + 50_000_000 if full_ns is not None else None
        ),
        "entry_hedge_executable_vwap_price": 101.0 if full else None,
        "entry_hedge_executed_quantity": 1 if full else None,
        "entry_hedge_depth_shortfall": 0 if full else None,
        "entry_hedge_decision_book_age_ms": 5.0 if full else None,
        "entry_hedge_signed_latency_slippage_bp": 0.0 if full else None,
        "entry_hedge_signed_depth_slippage_bp": 0.0 if full else None,
        "entry_hedge_signed_total_slippage_bp": 0.0 if full else None,
        "entry_hedge_label_observed": full,
        "entry_hedge_executable": full,
    }


class FrozenCompactFactsTest(unittest.TestCase):
    def test_compact_benchmark_cli_has_no_removed_position_root_dependency(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "maker.src.quote_fill.compact_frozen_benchmark",
                "--help",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertNotIn("position-root", completed.stdout)

    def test_l1_l2_supported_and_deeper_rank_fails_closed(self) -> None:
        actions = pl.from_dicts(
            [
                _formal_action("BID1", full=True, generation="g1"),
                _formal_action("BID2", full=False, generation="g2"),
                _formal_action("BID3", full=True, generation="g3"),
            ],
            infer_schema_length=None,
        )
        outcomes = build_compact_from_execution_actions(actions)
        unsupported = outcomes.filter(
            pl.col("target_rank_at_submit") == "BID3"
        ).row(0, named=True)
        self.assertFalse(unsupported["outcome_supported"])
        self.assertIsNone(unsupported["full_fill"])
        self.assertIsNone(unsupported["cancel_required"])
        self.assertFalse(unsupported["hedge_label_observed"])
        supported = outcomes.filter(pl.col("outcome_supported"))
        self.assertEqual(supported.height, 2)
        self.assertEqual(
            build_compact_state_changes(outcomes).height,
            4,
        )
        audit = summarize_frozen_product_days(outcomes)
        self.assertEqual(int(audit["candidate_policy_aliases"].sum()), 3)
        self.assertEqual(int(audit["supported_outcome_aliases"].sum()), 2)
        self.assertEqual(int(audit["unsupported_unknown_aliases"].sum()), 1)

    def test_route_market_side_rank_support_is_canonical(self) -> None:
        corrupt = _formal_action("ASK1", full=True, generation="g1")
        with self.assertRaisesRegex(ValueError, "incoherent"):
            build_compact_from_execution_actions(
                pl.from_dicts([corrupt], infer_schema_length=None)
            )

    def test_supported_null_and_chronology_corruption_fail_closed(self) -> None:
        corrupt_null = _formal_action("BID1", full=False, generation="g1")
        corrupt_null["full_fill"] = None
        with self.assertRaisesRegex(ValueError, "incoherent"):
            build_compact_from_execution_actions(
                pl.from_dicts([corrupt_null], infer_schema_length=None)
            )
        corrupt_time = _formal_action("BID1", full=True, generation="g2")
        corrupt_time["first_fill_recv_time_ns"] = 500_000_000
        corrupt_time["full_fill_recv_time_ns"] = 500_000_000
        corrupt_time["terminal_recv_time_ns"] = 500_000_000
        corrupt_time["entry_hedge_decision_time_ns"] = 550_000_000
        with self.assertRaisesRegex(ValueError, "incoherent"):
            build_compact_from_execution_actions(
                pl.from_dicts([corrupt_time], infer_schema_length=None)
            )

    def test_shared_raw_physical_identity_must_be_coherent(self) -> None:
        left = _formal_action("BID1", full=False, generation="g1")
        right = _formal_action("BID1", full=False, generation="g2")
        left["raw_order_fact_id"] = "shared"
        right["raw_order_fact_id"] = "shared"
        right["target_price_tick"] = 101
        right["target_price"] = 101.0
        with self.assertRaisesRegex(ValueError, "physical hedge facts"):
            build_compact_from_execution_actions(
                pl.from_dicts([left, right], infer_schema_length=None)
            )

    def test_shared_raw_nonfill_alias_may_retain_physical_hedge_cursor(self) -> None:
        full = _formal_action("BID1", full=True, generation="q50")
        no_fill = _formal_action("BID1", full=False, generation="q80")
        full["raw_order_fact_id"] = "shared"
        no_fill["raw_order_fact_id"] = "shared"
        no_fill["boundary_quantile"] = 80
        no_fill["entry_hedge_status"] = full["entry_hedge_status"]
        no_fill["entry_hedge_decision_time_ns"] = full[
            "entry_hedge_decision_time_ns"
        ]
        no_fill["entry_hedge_executable_vwap_price"] = full[
            "entry_hedge_executable_vwap_price"
        ]
        no_fill["entry_hedge_executed_quantity"] = full[
            "entry_hedge_executed_quantity"
        ]
        no_fill["entry_hedge_depth_shortfall"] = full[
            "entry_hedge_depth_shortfall"
        ]
        result = build_compact_from_execution_actions(
            pl.from_dicts([full, no_fill], infer_schema_length=None)
        )
        self.assertEqual(result.height, 2)
        self.assertEqual(int(result["full_fill"].sum()), 1)
        self.assertEqual(int(result["hedge_label_observed"].sum()), 1)

    def test_real_narrow_zero_partition_has_stable_typed_outputs(self) -> None:
        result = load_frozen_compact_product_days(
            dates=["20260522"], value_codes=["2412"]
        )
        self.assertTrue(result.order_outcomes.is_empty())
        self.assertGreater(result.order_outcomes.width, 60)
        self.assertGreater(result.state_changes.width, 10)
        self.assertGreater(result.audit.width, 10)
        self.assertGreater(result.q_summary.width, 30)

    def test_partition_marker_hash_is_verified_before_projection(self) -> None:
        source_root = Path("maker/data/walkforward/execution_narrow_60d")
        source_partition = (
            source_root / "Date=20260522" / "ValueCode=2412"
        )
        source_manifest = pl.read_parquet(
            source_root / "execution_partition_manifest.parquet"
        ).filter(
            (pl.col("Date") == "20260522")
            & (pl.col("ValueCode") == "2412")
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "execution"
            partition = root / "Date=20260522" / "ValueCode=2412"
            partition.mkdir(parents=True)
            shutil.copy2(
                source_partition / "execution_action_facts.parquet",
                partition / "execution_action_facts.parquet",
            )
            shutil.copy2(
                source_partition / "complete.json",
                partition / "complete.json",
            )
            manifest_row = source_manifest.row(0, named=True)
            _verify_action_partition(
                partition,
                manifest_row,
                partition / "execution_action_facts.parquet",
            )
            with (partition / "execution_action_facts.parquet").open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "byte count|hash mismatch"):
                _verify_action_partition(
                    partition,
                    manifest_row,
                    partition / "execution_action_facts.parquet",
                )

    def test_truncated_root_manifest_cannot_masquerade_as_full_or_subset(self) -> None:
        source = pl.read_parquet(
            "maker/data/walkforward/execution_narrow_60d/"
            "execution_partition_manifest.parquet"
        ).head(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "execution"
            root.mkdir()
            source.write_parquet(root / "execution_partition_manifest.parquet")
            with self.assertRaisesRegex(ValueError, "frozen full inventory"):
                load_frozen_compact_product_days(root)
            with self.assertRaisesRegex(ValueError, "frozen full inventory"):
                load_frozen_compact_product_days(
                    root,
                    dates=[str(source.item(0, "Date"))],
                    value_codes=[str(source.item(0, "ValueCode"))],
                )

    def test_canonical_empty_outcome_schema_is_concat_stable(self) -> None:
        empty = _normalize_outcomes(_empty_outcomes())
        self.assertEqual(empty.schema, dict(_OUTCOME_BASE_SCHEMA))
        one = pl.DataFrame(
            [
                {
                    column: None
                    for column in _OUTCOME_BASE_SCHEMA
                }
            ],
            schema=_OUTCOME_BASE_SCHEMA,
        )
        combined = pl.concat([empty, one], how="vertical")
        self.assertEqual(combined.schema, empty.schema)
        self.assertEqual(combined.height, 1)


def _exit_row(
    position: str,
    policy: str,
    physical: str,
    established_ns: int,
    *,
    exact: bool = True,
    submit_ns: int = 10,
    submit_sequence: int = 3,
    submit_row: int | None = None,
    stop_ns: int = 40,
    full_ns: int = 30,
    full_sequence: int = 5,
    full_row: int = 100,
    scenario: str = "scenario-1",
    requested_quantity: int = 2,
    physical_order_quantity: int = 2,
    hedge_observed: bool | None = None,
    hedge_executable: bool | None = None,
) -> dict[str, object]:
    hedge_observed = exact if hedge_observed is None else hedge_observed
    hedge_executable = exact if hedge_executable is None else hedge_executable
    return {
        "scenario_id": scenario,
        "Date": "20260714",
        "ValueCode": "1513",
        "QuoteCode": "TXF202607",
        "position_id": position,
        "position_established_recv_time_ns": established_ns,
        "position_established_event_sequence": 1,
        "position_established_row_index": established_ns,
        "exit_route": "spot_ask_future_taker",
        "maker_market": "spot",
        "maker_side": "ask",
        "exit_policy_generation_id": policy,
        "physical_exit_order_id": physical,
        "absolute_price_tick": 100,
        "submit_recv_time_ns": submit_ns,
        "submit_event_sequence": submit_sequence,
        "submit_row_index": (
            10 + established_ns if submit_row is None else submit_row
        ),
        "nominal_stop_recv_time_ns": stop_ns,
        "nominal_stop_event_sequence": 9,
        "nominal_stop_row_index": 999,
        "requested_quantity": requested_quantity,
        "physical_order_quantity": physical_order_quantity,
        "outcome_supported": True,
        "full_fill": True,
        "full_fill_recv_time_ns": full_ns,
        "full_fill_event_sequence": full_sequence,
        "full_fill_row_index": full_row,
        "fill_cursor_exact": exact,
        "hedge_label_observed": hedge_observed,
        "hedge_executable": hedge_executable,
        "hedge_decision_time_ns": (
            full_ns + 50_000_000 if hedge_observed else None
        ),
        "hedge_input_fill_cursor_exact": exact,
    }


def _capacity_event(
    *,
    scenario: str = "scenario-1",
    fill_ns: int = 30,
    fill_sequence: int = 5,
    fill_row: int = 100,
    traded: int = 2,
    queue_before: int = 0,
) -> dict[str, object]:
    row: dict[str, object] = {
        "scenario_id": scenario,
        "capacity_event_id": "pending",
        "source_trade_event_id": "pending",
        "Date": "20260714",
        "ValueCode": "1513",
        "exit_route": "spot_ask_future_taker",
        "maker_market": "spot",
        "maker_side": "ask",
        "absolute_price_tick": 100,
        "fill_recv_time_ns": fill_ns,
        "fill_event_sequence": fill_sequence,
        "fill_row_index": fill_row,
        "source_trade_quantity": traded,
        "external_queue_ahead_before": queue_before,
        "external_queue_ahead_after": max(0, queue_before - traded),
        "available_quantity": max(0, traded - queue_before),
        "capacity_basis": CAPACITY_BASIS,
        "source_replay_claimed_exact": True,
        "source_replay_version": "indexed_trade_replay_test_v1",
        "source_tape_sha256": "a" * 64,
    }
    row["source_trade_event_id"] = canonical_source_trade_event_id(row)
    row["capacity_event_id"] = canonical_capacity_event_id(row)
    return row


class CompactExitAllocationTest(unittest.TestCase):
    def test_independent_first_then_unique_physical_fifo_without_alias_double_count(self) -> None:
        outcomes = pl.from_dicts(
            [
                _exit_row("p1", "a1", "physical-1", 1),
                _exit_row("p1", "a2", "physical-1", 1),
                _exit_row("p2", "b1", "physical-2", 2),
                _exit_row("p3", "c1", "physical-3", 3, exact=False),
            ],
            infer_schema_length=None,
        )
        independent = build_independent_close_opportunities(outcomes)
        self.assertEqual(
            independent.filter(
                pl.col("independent_close_status")
                == "exact_independent_intraday_close"
            ).height,
            3,
        )
        self.assertEqual(
            independent.filter(
                pl.col("independent_close_status")
                == "approx_close_unallocatable"
            ).height,
            1,
        )
        capacity = pl.from_dicts([_capacity_event()], infer_schema_length=None)
        physical, aliases = simulate_fifo_residual_capacity_contract(
            independent, capacity
        )
        self.assertEqual(physical.height, 2)
        self.assertEqual(int(physical["fifo_allocated_quantity"].sum()), 2)
        first = physical.filter(
            pl.col("physical_exit_order_id") == "physical-1"
        ).row(0, named=True)
        second = physical.filter(
            pl.col("physical_exit_order_id") == "physical-2"
        ).row(0, named=True)
        self.assertEqual(first["policy_alias_count"], 2)
        self.assertEqual(first["fifo_allocated_quantity"], 2)
        self.assertTrue(first["fifo_cursor_matches_independent"])
        self.assertFalse(first["fifo_close_terminal_ready"])
        self.assertEqual(second["fifo_allocated_quantity"], 0)
        self.assertEqual(
            aliases.filter(
                pl.col("physical_exit_order_id") == "physical-1"
            ).height,
            2,
        )
        excluded = aliases.filter(
            pl.col("physical_exit_order_id") == "physical-3"
        ).row(0, named=True)
        self.assertFalse(excluded["fifo_cursor_matches_independent"])
        self.assertFalse(excluded["capacity_provenance_verified"])
        self.assertTrue(excluded["experimental_contract_only"])
        self.assertTrue(excluded["actual_fifo_metrics_absent"])
        self.assertTrue(excluded["mutually_exclusive_scenario"])

    def test_alias_local_outcomes_cannot_inherit_physical_allocation(self) -> None:
        full = _exit_row("p1", "a1", "physical-1", 1)
        no_fill = _exit_row("p1", "a2", "physical-1", 1)
        no_fill.update(
            {
                "full_fill": False,
                "full_fill_recv_time_ns": None,
                "full_fill_event_sequence": None,
                "full_fill_row_index": None,
                "hedge_label_observed": False,
                "hedge_executable": False,
                "hedge_decision_time_ns": None,
            }
        )
        with self.assertRaisesRegex(ValueError, "aliases disagree"):
            build_independent_close_opportunities(
                pl.from_dicts([full, no_fill], infer_schema_length=None)
            )

    def test_logical_position_quantity_cannot_exceed_physical_order(self) -> None:
        first = _exit_row("p1", "a", "aggregate", 1, submit_row=10)
        second = _exit_row("p2", "b", "aggregate", 2, submit_row=10)
        with self.assertRaisesRegex(ValueError, "logical position demand"):
            build_independent_close_opportunities(
                pl.from_dicts([first, second], infer_schema_length=None)
            )

    def test_unknown_full_flag_is_not_classified_as_no_fill(self) -> None:
        unknown = _exit_row("p1", "a", "physical-1", 1, exact=False)
        unknown.update(
            {
                "full_fill": None,
                "full_fill_recv_time_ns": None,
                "full_fill_event_sequence": None,
                "full_fill_row_index": None,
                "hedge_label_observed": False,
                "hedge_executable": False,
                "hedge_decision_time_ns": None,
            }
        )
        independent = build_independent_close_opportunities(
            pl.from_dicts([unknown], infer_schema_length=None)
        )
        self.assertEqual(
            independent.item(0, "independent_close_status"),
            "unknown_outcome_unallocatable",
        )
        self.assertEqual(
            independent.schema["independent_close_quantity"], pl.Int64
        )

    def test_unsupported_exit_cannot_carry_exact_fill_or_hedge_truth(self) -> None:
        unsupported = _exit_row("p1", "a", "physical-1", 1)
        unsupported["outcome_supported"] = False
        unsupported["full_fill"] = None
        unsupported["full_fill_recv_time_ns"] = None
        unsupported["full_fill_event_sequence"] = None
        unsupported["full_fill_row_index"] = None
        with self.assertRaisesRegex(ValueError, "exact fill/hedge truth"):
            build_independent_close_opportunities(
                pl.from_dicts([unsupported], infer_schema_length=None)
            )

    def test_fifo_recomputes_and_rejects_tampered_derived_eligibility(self) -> None:
        no_fill = _exit_row("p1", "a", "physical-1", 1)
        no_fill.update(
            {
                "full_fill": False,
                "full_fill_recv_time_ns": None,
                "full_fill_event_sequence": None,
                "full_fill_row_index": None,
                "hedge_label_observed": False,
                "hedge_executable": False,
                "hedge_decision_time_ns": None,
            }
        )
        genuine = _exit_row("p2", "b", "physical-2", 2)
        independent = build_independent_close_opportunities(
            pl.from_dicts([no_fill, genuine], infer_schema_length=None)
        ).with_columns(
            pl.when(pl.col("position_id") == "p1")
            .then(pl.lit(True))
            .otherwise(pl.col("fifo_capacity_eligible"))
            .alias("fifo_capacity_eligible")
        )
        with self.assertRaisesRegex(ValueError, "tampered"):
            simulate_fifo_residual_capacity_contract(
                independent,
                pl.from_dicts([_capacity_event()], infer_schema_length=None),
            )

    def test_exchange_fifo_uses_submit_cursor_not_position_age(self) -> None:
        older_position_later_order = _exit_row(
            "old", "a", "late-order", 1, submit_ns=20
        )
        newer_position_earlier_order = _exit_row(
            "new", "b", "early-order", 2, submit_ns=10
        )
        independent = build_independent_close_opportunities(
            pl.from_dicts(
                [older_position_later_order, newer_position_earlier_order],
                infer_schema_length=None,
            )
        )
        physical, _ = simulate_fifo_residual_capacity_contract(
            independent,
            pl.from_dicts([_capacity_event()], infer_schema_length=None),
        )
        allocation = {
            row["physical_exit_order_id"]: row["fifo_allocated_quantity"]
            for row in physical.to_dicts()
        }
        self.assertEqual(allocation, {"early-order": 2, "late-order": 0})

    def test_capacity_after_active_stop_is_not_allocated(self) -> None:
        row = _exit_row(
            "p1",
            "a",
            "physical-1",
            1,
            full_ns=20,
            stop_ns=25,
        )
        independent = build_independent_close_opportunities(
            pl.from_dicts([row], infer_schema_length=None)
        )
        physical, aliases = simulate_fifo_residual_capacity_contract(
            independent,
            pl.from_dicts([_capacity_event(fill_ns=30)], infer_schema_length=None),
        )
        self.assertEqual(physical.item(0, "fifo_allocated_quantity"), 0)
        self.assertFalse(aliases.item(0, "fifo_close_terminal_ready"))

    def test_unobserved_hedge_is_not_fifo_eligible(self) -> None:
        row = _exit_row(
            "p1",
            "a",
            "physical-1",
            1,
            hedge_observed=False,
            hedge_executable=False,
        )
        independent = build_independent_close_opportunities(
            pl.from_dicts([row], infer_schema_length=None)
        )
        self.assertFalse(independent.item(0, "fifo_capacity_eligible"))
        physical, aliases = simulate_fifo_residual_capacity_contract(
            independent,
            pl.from_dicts([_capacity_event()], infer_schema_length=None),
        )
        self.assertTrue(physical.is_empty())
        self.assertFalse(aliases.item(0, "fifo_close_terminal_ready"))

    def test_capacity_must_be_canonical_residual_not_raw_print(self) -> None:
        independent = build_independent_close_opportunities(
            pl.from_dicts(
                [_exit_row("p1", "a", "physical-1", 1)],
                infer_schema_length=None,
            )
        )
        corrupt = _capacity_event(traded=2, queue_before=1)
        corrupt["available_quantity"] = 2
        corrupt["capacity_event_id"] = canonical_capacity_event_id(corrupt)
        with self.assertRaisesRegex(ValueError, "arithmetic"):
            simulate_fifo_residual_capacity_contract(
                independent, pl.from_dicts([corrupt], infer_schema_length=None)
            )
        relabeled = _capacity_event()
        relabeled["source_trade_event_id"] = "b" * 64
        relabeled["capacity_event_id"] = canonical_capacity_event_id(relabeled)
        with self.assertRaisesRegex(ValueError, "source_trade_event_id"):
            simulate_fifo_residual_capacity_contract(
                independent,
                pl.from_dicts([relabeled], infer_schema_length=None),
            )

    def test_one_raw_cursor_cannot_mint_capacity_by_changing_qty_or_target(self) -> None:
        independent = build_independent_close_opportunities(
            pl.from_dicts(
                [_exit_row("p1", "a", "physical-1", 1)],
                infer_schema_length=None,
            )
        )
        first = _capacity_event(traded=2)
        second = _capacity_event(traded=4)
        second["absolute_price_tick"] = 101
        second["source_trade_event_id"] = canonical_source_trade_event_id(
            second
        )
        second["capacity_event_id"] = canonical_capacity_event_id(second)
        self.assertEqual(
            first["source_trade_event_id"], second["source_trade_event_id"]
        )
        with self.assertRaisesRegex(ValueError, "reused"):
            simulate_fifo_residual_capacity_contract(
                independent,
                pl.from_dicts([first, second], infer_schema_length=None),
            )

    def test_same_source_capacity_may_replay_only_across_explicit_scenarios(self) -> None:
        first_order = _exit_row(
            "p1", "a", "physical-1", 1, scenario="scenario-1"
        )
        second_order = _exit_row(
            "p1", "b", "physical-1", 1, scenario="scenario-2"
        )
        independent = build_independent_close_opportunities(
            pl.from_dicts([first_order, second_order], infer_schema_length=None)
        )
        first = _capacity_event(scenario="scenario-1")
        second = _capacity_event(scenario="scenario-2")
        self.assertEqual(
            first["source_trade_event_id"], second["source_trade_event_id"]
        )
        physical, _ = simulate_fifo_residual_capacity_contract(
            independent,
            pl.from_dicts([first, second], infer_schema_length=None),
        )
        self.assertEqual(physical.height, 2)
        self.assertFalse(bool(physical["joint_volume_allocated"].any()))
        self.assertTrue(bool(physical["mutually_exclusive_scenario"].all()))

    def test_same_ns_cursor_order_and_two_stage_position_fifo(self) -> None:
        early = _exit_row(
            "p1",
            "a",
            "aggregate",
            1,
            submit_ns=10,
            submit_row=10,
            requested_quantity=1,
            physical_order_quantity=2,
            full_ns=30,
            full_sequence=5,
            full_row=100,
        )
        late = _exit_row(
            "p2",
            "b",
            "aggregate",
            2,
            submit_ns=10,
            submit_row=10,
            requested_quantity=1,
            physical_order_quantity=2,
            full_ns=30,
            full_sequence=6,
            full_row=101,
        )
        independent = build_independent_close_opportunities(
            pl.from_dicts([early, late], infer_schema_length=None)
        )
        first = _capacity_event(
            fill_ns=30, fill_sequence=5, fill_row=100, traded=1
        )
        second = _capacity_event(
            fill_ns=30, fill_sequence=6, fill_row=101, traded=1
        )
        physical, aliases = simulate_fifo_residual_capacity_contract(
            independent,
            pl.from_dicts([first, second], infer_schema_length=None),
        )
        self.assertEqual(physical.item(0, "fifo_allocated_quantity"), 2)
        projected = {
            row["position_id"]: (
                row["fifo_allocated_quantity"],
                row["fifo_full_fill_event_sequence"],
                row["fifo_cursor_matches_independent"],
            )
            for row in aliases.to_dicts()
        }
        self.assertEqual(projected["p1"], (1, 5, True))
        self.assertEqual(projected["p2"], (1, 6, True))

    def test_capacity_cursor_strings_are_cast_before_chronological_sort(self) -> None:
        order = _exit_row(
            "p1",
            "a",
            "physical-1",
            1,
            requested_quantity=2,
            physical_order_quantity=2,
            stop_ns=200,
            full_ns=100,
            full_sequence=5,
            full_row=100,
        )
        independent = build_independent_close_opportunities(
            pl.from_dicts([order], infer_schema_length=None)
        )
        early = _capacity_event(
            fill_ns=20, fill_sequence=4, fill_row=90, traded=1
        )
        late = _capacity_event(
            fill_ns=100, fill_sequence=5, fill_row=100, traded=1
        )
        capacity = pl.from_dicts([early, late], infer_schema_length=None).with_columns(
            pl.col("fill_recv_time_ns").cast(pl.String)
        )
        physical, _ = simulate_fifo_residual_capacity_contract(
            independent, capacity
        )
        self.assertEqual(physical.item(0, "fifo_first_fill_recv_time_ns"), 20)
        self.assertEqual(physical.item(0, "fifo_full_fill_recv_time_ns"), 100)

    def test_independent_cursor_strings_are_cast_before_fifo_projection(self) -> None:
        independent = build_independent_close_opportunities(
            pl.from_dicts(
                [_exit_row("p1", "a", "physical-1", 1)],
                infer_schema_length=None,
            )
        ).with_columns(pl.col("submit_recv_time_ns").cast(pl.String))
        capacity = pl.from_dicts(
            [_capacity_event()], infer_schema_length=None
        )
        physical, aliases = simulate_fifo_residual_capacity_contract(
            independent, capacity
        )
        self.assertEqual(physical.schema["submit_recv_time_ns"], pl.Int64)
        self.assertEqual(aliases.schema["submit_recv_time_ns"], pl.Int64)
        self.assertEqual(physical.item(0, "submit_recv_time_ns"), 10)
        self.assertEqual(aliases.item(0, "submit_recv_time_ns"), 10)

    def test_exact_fifo_api_fails_closed_without_verified_builder(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "no verified source-bound"):
            allocate_fifo_exact_capacity(pl.DataFrame(), pl.DataFrame())


if __name__ == "__main__":
    unittest.main()
