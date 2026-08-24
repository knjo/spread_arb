from __future__ import annotations

from bisect import bisect_right
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import polars as pl

from maker.src.quote_fill import exit_maker_study as exit_maker_study_module
from maker.src.quote_fill.exit_maker import (
    FUTURE_BID_EXIT_ROUTE,
    SPOT_ASK_EXIT_ROUTE,
)
from maker.src.quote_fill.exit_maker_study import (
    ExitMakerProductDayReplayCache,
    ExitMakerReplayTuning,
    ExitMakerStudyConfig,
    _build_observations,
    _build_timeline,
    _index_observation_timeline,
    _observation_at_point,
    _thin_target_timeline,
    replay_exit_maker_product_day,
    replay_exit_maker_product_day_to_artifacts,
)
from maker.src.quote_fill.layered import EventCursor
from maker.src.quote_fill.raw_tape import RawTapeDay


MS = 1_000_000
DATE = "20260102"
VALUE_CODE = "2330"
QUOTE_CODE = "CDF1"


def _state(
    market: str,
    ms: int,
    sequence: int,
    *,
    bid: float,
    ask: float,
    bid_lots: int = 10,
    ask_lots: int = 10,
    raw_has_book: bool = True,
    trial_match: bool = False,
) -> dict[str, object]:
    row: dict[str, object] = {
        "Date": DATE,
        "market": market,
        "ValueCode": VALUE_CODE,
        "QuoteCode": QUOTE_CODE,
        "instrument_code": VALUE_CODE if market == "spot" else QUOTE_CODE,
        "recv_time_ns": ms * MS,
        "sequence": sequence,
        "packet_sequence": sequence,
        "trial_match": trial_match,
        "ref_price": 100.0,
        "contract_size": 2_000.0,
        "raw_has_book": raw_has_book,
        "book_state_available": True,
        "book_recv_time_ns": ms * MS if raw_has_book else 0,
        "exec_bid_price": bid,
        "exec_bid_lots": bid_lots,
        "exec_ask_price": ask,
        "exec_ask_lots": ask_lots,
        "best_bid_price": bid if market == "future" else None,
        "best_bid_lots": bid_lots if market == "future" else None,
        "best_ask_price": ask if market == "future" else None,
        "best_ask_lots": ask_lots if market == "future" else None,
    }
    for side, price, lots in (
        ("bid", bid, bid_lots),
        ("ask", ask, ask_lots),
    ):
        for level in range(1, 6):
            row[f"{side}_price_{level}"] = price if level == 1 else None
            row[f"{side}_lots_{level}"] = lots if level == 1 else None
    return row


def _trade(market: str, ms: int, sequence: int, price: float, lots: int) -> dict[str, object]:
    return {
        "Date": DATE,
        "market": market,
        "ValueCode": VALUE_CODE,
        "QuoteCode": QUOTE_CODE,
        "instrument_code": VALUE_CODE if market == "spot" else QUOTE_CODE,
        "recv_time_ns": ms * MS,
        "sequence": sequence,
        "packet_sequence": sequence,
        "trade_price": price,
        "trade_lots": lots,
    }


def _raw_tape() -> tuple[RawTapeDay, pl.DataFrame]:
    spot_states = pl.from_dicts(
        [
            _state("spot", 0, 1, bid=100.0, ask=100.5),
            _state(
                "spot",
                200,
                2,
                bid=100.0,
                ask=100.5,
                raw_has_book=False,
            ),
            _state("spot", 250, 3, bid=99.5, ask=100.5),
        ],
        infer_schema_length=None,
    )
    future_states = pl.from_dicts(
        [
            _state("future", 0, 10, bid=99.5, ask=100.5),
            _state(
                "future",
                200,
                11,
                bid=99.5,
                ask=100.5,
                raw_has_book=False,
            ),
            _state("future", 250, 12, bid=100.0, ask=101.0),
        ],
        infer_schema_length=None,
    )
    mapping = pl.DataFrame(
        {
            "ValueCode": [VALUE_CODE],
            "QuoteCode": [QUOTE_CODE],
            "spot_ref_price": [100.0],
            "fut_ref_price": [100.0],
            "contract_size": [2_000.0],
        }
    )
    tape = RawTapeDay(
        DATE,
        mapping,
        spot_states,
        future_states,
        pl.from_dicts([_trade("spot", 200, 2, 101.0, 2)]),
        pl.from_dicts([_trade("future", 200, 11, 99.5, 1)]),
        pl.DataFrame(),
    )
    clock = pl.DataFrame(
        {
            "Date": [DATE, DATE, DATE],
            "ValueCode": [VALUE_CODE, VALUE_CODE, VALUE_CODE],
            "spot_channel_seq": [1, 2, 3],
            "spread_pair_id": [1, 1, 2],
            "spread_pair_epoch": [1, 1, 2],
        }
    )
    return tape, clock


def _prior_unacked_cancel_tape() -> tuple[RawTapeDay, pl.DataFrame]:
    tape, clock = _raw_tape()
    extra_spot = pl.from_dicts(
        [_state("spot", 350, 4, bid=99.5, ask=100.5)],
        infer_schema_length=None,
    )
    tape = RawTapeDay(
        tape.date,
        tape.mapping,
        pl.concat([tape.spot_states, extra_spot], how="vertical_relaxed"),
        tape.future_states,
        pl.DataFrame(schema=tape.spot_trades.schema),
        pl.from_dicts([_trade("future", 300, 13, 99.0, 1)]),
        tape.audit,
    )
    clock = pl.concat(
        [
            clock,
            pl.DataFrame(
                {
                    "Date": [DATE],
                    "ValueCode": [VALUE_CODE],
                    "spot_channel_seq": [4],
                    "spread_pair_id": [2],
                    "spread_pair_epoch": [2],
                }
            ),
        ],
        how="vertical_relaxed",
    )
    return tape, clock


def _actions(*, full_fill: bool = True) -> pl.DataFrame:
    rows = []
    for policy in ("entry/q50/1", "entry/q80/1"):
        rows.append(
            {
                "Date": DATE,
                "ValueCode": VALUE_CODE,
                "QuoteCode": QUOTE_CODE,
                "route": "future_ask_spot_taker",
                "raw_order_fact_id": "shared-entry-physical",
                "policy_generation_id": policy,
                "full_fill": full_fill,
                "entry_hedge_status": "executable" if full_fill else None,
                "entry_hedge_decision_time_ns": 100 * MS if full_fill else None,
                "entry_future_price": 102.0 if full_fill else None,
                "entry_spot_price": 100.0 if full_fill else None,
                "entry_hedge_contract_size_shares": 2_000 if full_fill else None,
            }
        )
    return pl.from_dicts(rows, infer_schema_length=None)


def _exit_facts(actions: pl.DataFrame, *, source_asof: str = "20260101") -> pl.DataFrame:
    rows = []
    for action in actions.iter_rows(named=True):
        for rule in ("frozen_center", "frozen_lower"):
            rows.append(
                {
                    "Date": DATE,
                    "ValueCode": VALUE_CODE,
                    "QuoteCode": QUOTE_CODE,
                    "route": action["route"],
                    "raw_order_fact_id": action["raw_order_fact_id"],
                    "policy_generation_id": action["policy_generation_id"],
                    "exit_rule_id": rule,
                    "exit_threshold_basis_bp": 0.0,
                    "exit_rule_source_asof_date": source_asof,
                }
            )
    return pl.from_dicts(rows, infer_schema_length=None)


def _legacy_build_observations(
    timeline,
    *,
    seed_timeline,
    route: str,
    threshold_basis_bp: float,
    start_time_ns: int,
    cutoff_cursor: EventCursor,
):
    """Reference implementation retained to prove the cursor-index rewrite."""

    start_cursor = EventCursor(start_time_ns, 3, 0)
    if start_cursor >= cutoff_cursor or not seed_timeline:
        return ()
    seed_cursors = [point.cursor for point in seed_timeline]
    seed_index = bisect_right(seed_cursors, start_cursor) - 1
    relevant = []
    if seed_index >= 0:
        relevant.append((start_cursor, seed_timeline[seed_index]))
    relevant.extend(
        (point.cursor, point)
        for point in timeline
        if start_cursor < point.cursor < cutoff_cursor
    )
    observations = []
    last = None
    last_signature = None
    for cursor, point in relevant:
        observation, failure = _observation_at_point(
            point,
            cursor=cursor,
            route=route,
            threshold_basis_bp=threshold_basis_bp,
        )
        if observation is None:
            if last is not None:
                epoch = (
                    point.spot.spread_pair_epoch
                    if point.spot is not None
                    and point.spot.spread_pair_epoch is not None
                    else last.spread_pair_epoch
                )
                observation = replace(
                    last,
                    cursor=cursor,
                    spread_pair_epoch=epoch,
                    gate_open=False,
                    gate_reason=failure or "raw_state_unavailable",
                    initial_queue_ahead=None,
                    target_rank="unavailable",
                )
            else:
                continue
        last = observation
        signature = (
            observation.spread_pair_epoch,
            observation.absolute_target_tick,
            observation.gate_open,
            observation.gate_reason,
        )
        if signature == last_signature:
            continue
        observations.append(observation)
        last_signature = signature
    return tuple(observations)


class ExitMakerProductDayStudyTest(unittest.TestCase):
    def test_spooled_artifacts_are_value_exact_and_leave_no_chunks(self) -> None:
        tape, clock = _raw_tape()
        actions = _actions()
        exits = _exit_facts(actions)
        arguments = {
            "cutoff_cursor": EventCursor(500 * MS, 3, 0),
            "replay_tuning": ExitMakerReplayTuning(record_chunk_rows=2),
        }
        reference = replay_exit_maker_product_day(
            actions,
            exits,
            tape,
            clock,
            **arguments,
        )
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory) / "stage"
            artifacts = replay_exit_maker_product_day_to_artifacts(
                actions,
                exits,
                tape,
                clock,
                artifact_directory=stage,
                **arguments,
            )
            paths = artifacts.artifact_paths()
            self.assertEqual(set(paths), set(reference.frames()))
            self.assertEqual(
                {path.name for path in stage.iterdir()},
                {f"{name}.parquet" for name in reference.frames()},
            )
            for name, expected in reference.frames().items():
                observed = pl.read_parquet(paths[name])
                self.assertEqual(observed.schema, expected.schema, name)
                self.assertTrue(observed.equals(expected), name)

    def test_spooled_empty_and_zero_established_paths_are_exact(self) -> None:
        tape, clock = _raw_tape()
        opened_actions = _actions()
        cases = (
            (
                "empty_actions",
                opened_actions.head(0),
                _exit_facts(opened_actions).head(0),
            ),
            (
                "zero_established",
                _actions(full_fill=False),
                pl.DataFrame(),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            for name, actions, exits in cases:
                with self.subTest(name=name):
                    reference = replay_exit_maker_product_day(
                        actions,
                        exits,
                        tape,
                        clock,
                        cutoff_cursor=EventCursor(500 * MS, 3, 0),
                    )
                    artifacts = replay_exit_maker_product_day_to_artifacts(
                        actions,
                        exits,
                        tape,
                        clock,
                        artifact_directory=Path(directory) / name,
                        cutoff_cursor=EventCursor(500 * MS, 3, 0),
                    )
                    for frame_name, expected in reference.frames().items():
                        observed = pl.read_parquet(
                            artifacts.artifact_paths()[frame_name]
                        )
                        self.assertEqual(
                            observed.schema, expected.schema, frame_name
                        )
                        self.assertTrue(
                            observed.equals(expected), frame_name
                        )

    def test_record_accumulator_never_exceeds_hard_chunk_bound(self) -> None:
        accumulator = exit_maker_study_module._RecordChunkAccumulator(
            {"value": pl.Int64},
            chunk_rows=2,
        )
        accumulator.extend({"value": value} for value in range(7))
        self.assertEqual([chunk.height for chunk in accumulator.chunks], [2, 2, 2])
        self.assertEqual(len(accumulator.pending), 1)
        observed = accumulator.finish()
        self.assertTrue(
            observed.equals(pl.DataFrame({"value": list(range(7))}))
        )

    def test_same_day_replay_without_external_cache_skips_content_hash(self) -> None:
        tape, clock = _raw_tape()
        actions = _actions()
        with patch.object(
            exit_maker_study_module,
            "_physical_replay_input_signature",
            side_effect=AssertionError("unexpected cross-call fingerprint"),
        ):
            result = replay_exit_maker_product_day(
                actions,
                _exit_facts(actions),
                tape,
                clock,
                cutoff_cursor=EventCursor(500 * MS, 3, 0),
            )
        self.assertEqual(result.policy_support.height, 8)

    def test_external_cache_rejects_stale_raw_and_clock_content(self) -> None:
        tape, clock = _raw_tape()
        actions = _actions()
        exits = _exit_facts(actions)
        arguments = {"cutoff_cursor": EventCursor(500 * MS, 3, 0)}

        raw_cache = ExitMakerProductDayReplayCache()
        replay_exit_maker_product_day(
            actions,
            exits,
            tape,
            clock,
            replay_cache=raw_cache,
            **arguments,
        )
        changed_tape = RawTapeDay(
            tape.date,
            tape.mapping,
            tape.spot_states,
            tape.future_states,
            tape.spot_trades,
            tape.future_trades.with_columns(
                (pl.col("trade_lots") + 1).alias("trade_lots")
            ),
            tape.audit,
        )
        with self.assertRaisesRegex(ValueError, "cache input mismatch"):
            replay_exit_maker_product_day(
                actions,
                exits,
                changed_tape,
                clock,
                replay_cache=raw_cache,
                **arguments,
            )

        clock_cache = ExitMakerProductDayReplayCache()
        replay_exit_maker_product_day(
            actions,
            exits,
            tape,
            clock,
            replay_cache=clock_cache,
            **arguments,
        )
        changed_clock = clock.with_columns(
            pl.when(pl.col("spot_channel_seq") == 3)
            .then(pl.lit(3))
            .otherwise(pl.col("spread_pair_epoch"))
            .alias("spread_pair_epoch")
        )
        with self.assertRaisesRegex(ValueError, "cache input mismatch"):
            replay_exit_maker_product_day(
                actions,
                exits,
                tape,
                changed_clock,
                replay_cache=clock_cache,
                **arguments,
            )

    def test_zero_cache_bounds_clear_existing_hits_before_lookup(self) -> None:
        tape, clock = _raw_tape()
        actions = _actions()
        exits = _exit_facts(actions)
        cache = ExitMakerProductDayReplayCache()
        arguments = {
            "cutoff_cursor": EventCursor(500 * MS, 3, 0),
            "replay_cache": cache,
        }
        first = replay_exit_maker_product_day(
            actions,
            exits,
            tape,
            clock,
            **arguments,
        )
        self.assertGreater(len(cache.physical), 0)
        self.assertEqual(len(cache.outcomes), 0)

        original = exit_maker_study_module._replay_physical_policy
        with patch.object(
            exit_maker_study_module,
            "_replay_physical_policy",
            wraps=original,
        ) as replay:
            second = replay_exit_maker_product_day(
                actions,
                exits,
                tape,
                clock,
                replay_tuning=ExitMakerReplayTuning(
                    record_chunk_rows=2,
                    physical_cache_max_entries=0,
                    outcome_cache_max_entries=0,
                ),
                **arguments,
            )
        self.assertGreater(replay.call_count, 0)
        self.assertEqual(len(cache.physical), 0)
        self.assertEqual(len(cache.outcomes), 0)
        for name, expected in first.frames().items():
            self.assertTrue(second.frames()[name].equals(expected), name)

    def test_chunk_size_and_cache_retention_are_value_exact(self) -> None:
        tape, clock = _raw_tape()
        actions = _actions()
        exits = _exit_facts(actions)
        arguments = {
            "cutoff_cursor": EventCursor(500 * MS, 3, 0),
        }
        reference = replay_exit_maker_product_day(
            actions,
            exits,
            tape,
            clock,
            replay_tuning=ExitMakerReplayTuning(
                record_chunk_rows=100_000,
                physical_cache_max_entries=128,
                outcome_cache_max_entries=128,
            ),
            **arguments,
        )
        bounded_cache = ExitMakerProductDayReplayCache()
        tiny_chunks = replay_exit_maker_product_day(
            actions,
            exits,
            tape,
            clock,
            replay_cache=bounded_cache,
            replay_tuning=ExitMakerReplayTuning(
                record_chunk_rows=1,
                physical_cache_max_entries=0,
                outcome_cache_max_entries=1,
            ),
            **arguments,
        )
        for name, expected in reference.frames().items():
            observed = tiny_chunks.frames()[name]
            self.assertEqual(observed.schema, expected.schema, name)
            self.assertTrue(observed.equals(expected), name)
        self.assertEqual(len(bounded_cache.physical), 0)
        self.assertEqual(len(bounded_cache.outcomes), 0)

    def test_identical_rule_windows_reuse_trade_replay_and_rebind_ids(self) -> None:
        tape, clock = _raw_tape()
        actions = _actions()
        original_replay = (
            exit_maker_study_module._replay_exit_maker_windows_indexed
        )
        with patch.object(
            exit_maker_study_module,
            "_replay_exit_maker_windows_indexed",
            wraps=original_replay,
        ) as replay:
            result = replay_exit_maker_product_day(
                actions,
                _exit_facts(actions),
                tape,
                clock,
                cutoff_cursor=EventCursor(500 * MS, 3, 0),
            )

        # q50/q80 already share the exact physical cache.  The two distinct
        # exit-rule policies have identical threshold/window structures here,
        # so only one maker-trade replay per route is required.
        self.assertEqual(replay.call_count, 2)
        identities = result.candidate_aliases.select(
            "physical_exit_policy_id",
            "exit_policy_candidate_id",
        ).unique()
        self.assertTrue(
            all(
                str(candidate_id).startswith(f"{physical_id}/")
                for physical_id, candidate_id in identities.iter_rows()
            )
        )

    def test_indexed_observations_are_value_exact_to_legacy_scan(self) -> None:
        tape, clock = _raw_tape()
        full_timeline = _build_timeline(tape, clock, VALUE_CODE)
        target_timeline = _thin_target_timeline(full_timeline)
        timeline_index = _index_observation_timeline(full_timeline)

        for route in (FUTURE_BID_EXIT_ROUTE, SPOT_ASK_EXIT_ROUTE):
            for threshold in (-25.0, 0.0, 25.0):
                for start_ms, cutoff_ms in (
                    (0, 500),
                    (1, 500),
                    (100, 500),
                    (200, 250),
                    (200, 500),
                    (250, 500),
                    (500, 500),
                ):
                    with self.subTest(
                        route=route,
                        threshold=threshold,
                        start_ms=start_ms,
                        cutoff_ms=cutoff_ms,
                    ):
                        expected = _legacy_build_observations(
                            target_timeline,
                            seed_timeline=full_timeline,
                            route=route,
                            threshold_basis_bp=threshold,
                            start_time_ns=start_ms * MS,
                            cutoff_cursor=EventCursor(cutoff_ms * MS, 3, 0),
                        )
                        actual = _build_observations(
                            timeline_index,
                            route=route,
                            threshold_basis_bp=threshold,
                            start_time_ns=start_ms * MS,
                            cutoff_cursor=EventCursor(cutoff_ms * MS, 3, 0),
                        )
                        self.assertEqual(actual, expected)

    def test_full_denominator_physical_alias_linkage_and_both_hedges(self) -> None:
        tape, clock = _raw_tape()
        actions = _actions()
        result = replay_exit_maker_product_day(
            actions,
            _exit_facts(actions),
            tape,
            clock,
            ExitMakerStudyConfig(),
            cutoff_cursor=EventCursor(500 * MS, 3, 0),
        )

        # Two entry aliases x two D-1 rules x two maker-exit routes.
        self.assertEqual(result.policy_support.height, 8)
        self.assertEqual(result.position_policy_facts.height, 8)
        self.assertEqual(
            set(result.policy_support["exit_route"].to_list()),
            {FUTURE_BID_EXIT_ROUTE, SPOT_ASK_EXIT_ROUTE},
        )
        self.assertTrue(
            result.policy_support["position_status"]
            .eq("position_established")
            .all()
        )

        # q50/q80 share one physical position.  Each physical rule/route gets
        # an epoch-1 base order and an epoch-2 base order, not duplicate raw
        # facts per entry alias.
        self.assertEqual(
            result.position_policy_facts["physical_exit_policy_id"].n_unique(), 4
        )
        self.assertEqual(result.raw_candidate_facts.height, 4)
        self.assertEqual(
            result.raw_candidate_facts["exit_raw_candidate_fact_id"].n_unique(), 4
        )
        self.assertEqual(result.candidate_aliases.height, 16)
        zero = replay_exit_maker_product_day(
            _actions(full_fill=False),
            _exit_facts(_actions(full_fill=False)),
            tape,
            clock,
            cutoff_cursor=EventCursor(500 * MS, 3, 0),
        )
        self.assertEqual(result.observations.schema, zero.observations.schema)
        self.assertEqual(result.transitions.schema, zero.transitions.schema)
        self.assertIn("Date", result.observations.columns)
        self.assertIn("ValueCode", result.transitions.columns)
        no_fill_tape = RawTapeDay(
            tape.date,
            tape.mapping,
            tape.spot_states,
            tape.future_states,
            tape.spot_trades.head(0),
            tape.future_trades.head(0),
            tape.audit,
        )
        no_fill = replay_exit_maker_product_day(
            actions,
            _exit_facts(actions),
            no_fill_tape,
            clock,
            ExitMakerStudyConfig(),
            cutoff_cursor=EventCursor(500 * MS, 3, 0),
        )
        self.assertEqual(
            result.candidate_aliases.schema, no_fill.candidate_aliases.schema
        )
        self.assertEqual(
            result.raw_candidate_facts.schema, no_fill.raw_candidate_facts.schema
        )
        self.assertEqual(
            result.position_policy_facts.schema,
            no_fill.position_policy_facts.schema,
        )
        aliases_per_raw = result.candidate_aliases.group_by(
            "exit_raw_candidate_fact_id"
        ).len()
        self.assertTrue(aliases_per_raw["len"].eq(4).all())

        policies = result.position_policy_facts
        self.assertTrue(policies["oco_winner_generation_id"].is_not_null().all())
        self.assertTrue(policies["oco_position_projection_safe"].all())
        self.assertTrue(policies["branch_status"].eq("flat_same_day").all())
        self.assertTrue(policies["terminal_outcome"].all())
        self.assertFalse(policies["cancel_ack_observed"].any())
        self.assertFalse(policies["cancel_race_modeled"].any())
        self.assertFalse(policies["strict_ev_ready"].any())
        self.assertTrue(
            policies["gross_cycle_pnl_twd"].is_not_null().all()
        )

        future = policies.filter(pl.col("exit_route") == FUTURE_BID_EXIT_ROUTE)
        spot = policies.filter(pl.col("exit_route") == SPOT_ASK_EXIT_ROUTE)
        self.assertTrue(future["exit_maker_quantity"].eq(1).all())
        self.assertTrue(future["exit_hedge_quantity"].eq(2).all())
        self.assertTrue(future["exit_hedge_status"].eq("executable").all())
        self.assertTrue(spot["exit_maker_quantity"].eq(2).all())
        self.assertTrue(spot["exit_hedge_quantity"].eq(1).all())
        self.assertTrue(spot["exit_hedge_status"].eq("executable").all())

    def test_active_policy_subset_is_exactly_the_full_replay_slice(self) -> None:
        tape, clock = _raw_tape()
        actions = _actions()
        exits = _exit_facts(actions)
        arguments = {
            "config": ExitMakerStudyConfig(),
            "cutoff_cursor": EventCursor(500 * MS, 3, 0),
        }
        full = replay_exit_maker_product_day(
            actions, exits, tape, clock, **arguments
        )
        selected = tuple(
            full.position_policy_facts.sort("exit_policy_trial_id")
            .head(3)["exit_policy_trial_id"]
            .to_list()
        )
        active = replay_exit_maker_product_day(
            actions,
            exits,
            tape,
            clock,
            active_exit_policy_trial_ids=selected,
            **arguments,
        )
        for field in ("policy_support", "position_policy_facts"):
            expected = getattr(full, field).filter(
                pl.col("exit_policy_trial_id").is_in(selected)
            ).sort("exit_policy_trial_id")
            observed = getattr(active, field).sort("exit_policy_trial_id")
            self.assertTrue(observed.equals(expected), field)
        audit = active.audit.row(0, named=True)
        self.assertEqual(audit["expected_policy_trials"], len(selected))
        self.assertEqual(audit["materialized_policy_trials"], len(selected))
        with self.assertRaisesRegex(ValueError, "absent from established"):
            replay_exit_maker_product_day(
                actions,
                exits,
                tape,
                clock,
                active_exit_policy_trial_ids=("not-a-real-policy",),
                **arguments,
            )

    def test_equivalent_classifiers_share_physical_replays(self) -> None:
        tape, clock = _raw_tape()
        actions = _actions()
        exits = _exit_facts(actions)
        cache = ExitMakerProductDayReplayCache()
        original = exit_maker_study_module._replay_physical_policy
        with patch.object(
            exit_maker_study_module,
            "_replay_physical_policy",
            wraps=original,
        ) as replay:
            first = replay_exit_maker_product_day(
                actions,
                exits,
                tape,
                clock,
                ExitMakerStudyConfig(exit_lifecycle_policy_version="strict"),
                cutoff_cursor=EventCursor(500 * MS, 3, 0),
                replay_cache=cache,
            )
            first_calls = replay.call_count
            second = replay_exit_maker_product_day(
                actions,
                exits,
                tape,
                clock,
                ExitMakerStudyConfig(exit_lifecycle_policy_version="nominal"),
                cutoff_cursor=EventCursor(500 * MS, 3, 0),
                replay_cache=cache,
            )
        self.assertGreater(first_calls, 0)
        self.assertEqual(replay.call_count, first_calls)
        comparable = [
            column
            for column in first.position_policy_facts.columns
            if column != "exit_lifecycle_policy_version"
        ]
        self.assertTrue(
            first.position_policy_facts.select(comparable).equals(
                second.position_policy_facts.select(comparable)
            )
        )

    def test_unopened_aliases_are_only_retained_in_all_entry_audit(self) -> None:
        tape, clock = _raw_tape()
        actions = _actions(full_fill=False)
        result = replay_exit_maker_product_day(
            actions,
            _exit_facts(actions),
            tape,
            clock,
            cutoff_cursor=EventCursor(500 * MS, 3, 0),
        )
        self.assertEqual(result.policy_support.height, 0)
        self.assertEqual(result.position_policy_facts.height, 0)
        self.assertEqual(result.raw_candidate_facts.height, 0)
        self.assertEqual(result.candidate_aliases.height, 0)
        self.assertEqual(result.audit.item(0, "all_entry_aliases"), 2)
        self.assertEqual(result.audit.item(0, "entry_policy_aliases"), 0)
        self.assertEqual(result.audit.item(0, "expected_policy_trials"), 0)

    def test_d_minus_one_lineage_fails_closed(self) -> None:
        tape, clock = _raw_tape()
        actions = _actions()
        with self.assertRaisesRegex(ValueError, "strictly before Date"):
            replay_exit_maker_product_day(
                actions,
                _exit_facts(actions, source_asof=DATE),
                tape,
                clock,
                cutoff_cursor=EventCursor(500 * MS, 3, 0),
            )

    def test_spread_pair_clock_must_cover_every_spot_state(self) -> None:
        tape, clock = _raw_tape()
        actions = _actions()
        with self.assertRaisesRegex(ValueError, "cover every raw spot state"):
            replay_exit_maker_product_day(
                actions,
                _exit_facts(actions),
                tape,
                clock.head(2),
                cutoff_cursor=EventCursor(500 * MS, 3, 0),
            )

    def test_prior_nominal_cancel_without_ack_is_strict_cancel_race(self) -> None:
        tape, clock = _prior_unacked_cancel_tape()
        actions = _actions()
        result = replay_exit_maker_product_day(
            actions,
            _exit_facts(actions),
            tape,
            clock,
            cutoff_cursor=EventCursor(500 * MS, 3, 0),
        )
        selected = result.position_policy_facts.filter(
            (pl.col("entry_policy_generation_id") == "entry/q50/1")
            & (pl.col("exit_rule_id") == "frozen_center")
            & (pl.col("exit_route") == FUTURE_BID_EXIT_ROUTE)
        )
        self.assertEqual(selected.height, 1)
        self.assertEqual(
            selected.item(0, "nominal_instant_cancel_v0_branch"),
            "flat_same_day",
        )
        self.assertEqual(
            selected.item(0, "prior_unacked_cancel_count_before_winner"), 1
        )
        self.assertEqual(selected.item(0, "oco_active_sibling_cancel_count"), 0)
        self.assertEqual(selected.item(0, "branch_status"), "cancel_race_unknown")
        self.assertFalse(selected.item(0, "terminal_outcome"))
        self.assertFalse(selected.item(0, "strict_ev_ready"))

    def test_empty_exit_frame_is_valid_when_no_position_was_established(self) -> None:
        tape, clock = _raw_tape()
        result = replay_exit_maker_product_day(
            _actions(full_fill=False),
            pl.DataFrame(),
            tape,
            clock,
            cutoff_cursor=EventCursor(500 * MS, 3, 0),
        )
        self.assertEqual(result.policy_support.height, 0)
        self.assertEqual(result.position_policy_facts.height, 0)
        self.assertEqual(result.audit.item(0, "all_entry_aliases"), 2)


if __name__ == "__main__":
    unittest.main()
