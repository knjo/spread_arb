from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import polars as pl

from maker.src.quote_fill import exit_maker_cross_session as cross_session_module
from maker.src.quote_fill.exit_maker import (
    FUTURE_BID_EXIT_ROUTE,
    SPOT_ASK_EXIT_ROUTE,
)
from maker.src.quote_fill.exit_maker_cross_session import (
    CrossSessionExitMakerConfig,
    CrossSessionExitMakerSession,
    _classify_session_policy,
    classify_exit_maker_session,
    replay_cross_session_exit_maker,
    replay_exit_maker_session_policy,
)
from maker.src.quote_fill.exit_maker_study import (
    ExitMakerStudyConfig,
    replay_exit_maker_product_day,
)
from maker.src.quote_fill.exit_maker_report import ExitMakerPartitionInputs
from maker.src.quote_fill.filled_entry_report import (
    build_filled_entry_primary_report,
)
from maker.src.quote_fill.layered import EventCursor
from maker.src.quote_fill.raw_tape import RawTapeDay


MS = 1_000_000
D1 = "20260601"
D2 = "20260602"
D3 = "20260603"
VALUE = "2303"
QUOTE = "CCFF6"


def _state(
    date: str,
    market: str,
    ms: int,
    sequence: int,
    *,
    bid: float,
    ask: float,
    bid_lots: int = 10,
    ask_lots: int = 10,
    ref_price: float = 100.0,
    raw_has_book: bool = True,
) -> dict[str, object]:
    row: dict[str, object] = {
        "Date": date,
        "market": market,
        "ValueCode": VALUE,
        "QuoteCode": QUOTE,
        "instrument_code": VALUE if market == "spot" else QUOTE,
        "recv_time_ns": ms * MS,
        "sequence": sequence,
        "packet_sequence": sequence,
        "trial_match": False,
        "ref_price": ref_price,
        "contract_size": 2_000.0,
        "raw_has_book": raw_has_book,
        "book_state_available": True,
        "book_recv_time_ns": ms * MS,
        "exec_bid_price": bid,
        "exec_bid_lots": bid_lots,
        "exec_ask_price": ask,
        "exec_ask_lots": ask_lots,
        "best_bid_price": bid if market == "future" else None,
        "best_bid_lots": bid_lots if market == "future" else None,
        "best_ask_price": ask if market == "future" else None,
        "best_ask_lots": ask_lots if market == "future" else None,
    }
    for side, price, lots in (("bid", bid, bid_lots), ("ask", ask, ask_lots)):
        for level in range(1, 6):
            row[f"{side}_price_{level}"] = price if level == 1 else None
            row[f"{side}_lots_{level}"] = lots if level == 1 else None
    return row


def _trade(
    date: str,
    market: str,
    ms: int,
    sequence: int,
    *,
    price: float,
    lots: int,
    ref_price: float = 100.0,
) -> dict[str, object]:
    return {
        "Date": date,
        "market": market,
        "ValueCode": VALUE,
        "QuoteCode": QUOTE,
        "instrument_code": VALUE if market == "spot" else QUOTE,
        "recv_time_ns": ms * MS,
        "sequence": sequence,
        "packet_sequence": sequence,
        "ref_price": ref_price,
        "trade_price": price,
        "trade_lots": lots,
    }


def _trade_frame(rows: list[dict[str, object]], market: str, date: str) -> pl.DataFrame:
    if rows:
        return pl.from_dicts(rows, infer_schema_length=None)
    template = pl.from_dicts(
        [_trade(date, market, 1, 1, price=100.0, lots=1)],
        infer_schema_length=None,
    )
    return template.head(0)


def _tape(
    date: str,
    *,
    future_trades: list[tuple[int, float, int]] = (),
    spot_trades: list[tuple[int, float, int]] = (),
    spot_updates: list[dict[str, object]] = (),
    quote: str = QUOTE,
    raw_ref_price: float = 100.0,
) -> tuple[RawTapeDay, pl.DataFrame]:
    spot_rows = [
        _state(
            date,
            "spot",
            100,
            2,
            bid=100.0,
            ask=100.5,
            bid_lots=2,
            ref_price=raw_ref_price,
        ),
        *spot_updates,
    ]
    future_rows = [
        _state(
            date,
            "future",
            100,
            1,
            bid=99.5,
            ask=100.5,
            ref_price=raw_ref_price,
        )
    ]
    future_trade_rows = [
        _trade(
            date,
            "future",
            ms,
            10 + index,
            price=price,
            lots=lots,
            ref_price=raw_ref_price,
        )
        for index, (ms, price, lots) in enumerate(future_trades)
    ]
    spot_trade_rows = [
        _trade(
            date,
            "spot",
            ms,
            20 + index,
            price=price,
            lots=lots,
            ref_price=raw_ref_price,
        )
        for index, (ms, price, lots) in enumerate(spot_trades)
    ]
    mapping = pl.DataFrame(
        {
            "ValueCode": [VALUE],
            "QuoteCode": [quote],
            "spot_ref_price": [raw_ref_price],
            "fut_ref_price": [raw_ref_price],
            "contract_size": [2_000.0],
        }
    )
    if quote != QUOTE:
        for rows in (spot_rows, future_rows, future_trade_rows, spot_trade_rows):
            for row in rows:
                row["QuoteCode"] = quote
                if row["market"] == "future":
                    row["instrument_code"] = quote
    tape = RawTapeDay(
        date,
        mapping,
        pl.from_dicts(spot_rows, infer_schema_length=None),
        pl.from_dicts(future_rows, infer_schema_length=None),
        _trade_frame(spot_trade_rows, "spot", date),
        _trade_frame(future_trade_rows, "future", date),
        pl.DataFrame(),
    )
    clock = pl.DataFrame(
        {
            "Date": [date for _ in spot_rows],
            "ValueCode": [VALUE for _ in spot_rows],
            "spot_channel_seq": [int(row["sequence"]) for row in spot_rows],
            "spread_pair_id": [1 for _ in spot_rows],
            "spread_pair_epoch": [1 for _ in spot_rows],
        }
    )
    return tape, clock


def _session(
    date: str,
    *,
    future_trades: list[tuple[int, float, int]] = (),
    spot_trades: list[tuple[int, float, int]] = (),
    spot_updates: list[dict[str, object]] = (),
    quote: str = QUOTE,
    raw_ref_price: float = 100.0,
    candidate_ref_price: float = 100.0,
) -> CrossSessionExitMakerSession:
    tape, clock = _tape(
        date,
        future_trades=future_trades,
        spot_trades=spot_trades,
        spot_updates=spot_updates,
        quote=quote,
        raw_ref_price=raw_ref_price,
    )
    return CrossSessionExitMakerSession(
        date=date,
        raw_tape=tape,
        spread_pair_clock=clock,
        spot_ref_price=candidate_ref_price,
        future_ref_price=candidate_ref_price,
        ref_price_source_date=date,
        ref_price_source_version=f"candidate-ref-{date}",
        cutoff_cursor=EventCursor(1_000 * MS, 3, 0),
    )


def _actions(
    *,
    include_no_fill: bool = False,
    include_unsupported_null: bool = False,
) -> pl.DataFrame:
    rows = [
        {
            "Date": D1,
            "ValueCode": VALUE,
            "QuoteCode": QUOTE,
            "route": "future_ask_spot_taker",
            "raw_order_fact_id": "entry-raw-established",
            "policy_generation_id": "entry-policy-established",
            "full_fill": True,
            "entry_hedge_status": "executable",
            "entry_hedge_decision_time_ns": 50 * MS,
            "entry_hedge_label_observed": True,
            "entry_hedge_executable": True,
            "entry_future_price": 102.0,
            "entry_spot_price": 100.0,
            "entry_hedge_contract_size_shares": 2_000,
        }
    ]
    if include_no_fill:
        rows.append(
            {
                "Date": D1,
                "ValueCode": VALUE,
                "QuoteCode": QUOTE,
                "route": "future_ask_spot_taker",
                "raw_order_fact_id": "entry-raw-established",
                "policy_generation_id": "entry-policy-no-fill",
                "full_fill": False,
                "entry_hedge_status": "executable",
                "entry_hedge_decision_time_ns": 50 * MS,
                "entry_hedge_label_observed": False,
                "entry_hedge_executable": False,
                "entry_future_price": 102.0,
                "entry_spot_price": 100.0,
                "entry_hedge_contract_size_shares": 2_000,
            }
        )
    if include_unsupported_null:
        rows.append(
            {
                "Date": D1,
                "ValueCode": VALUE,
                "QuoteCode": QUOTE,
                "route": "future_ask_spot_taker",
                "raw_order_fact_id": "entry-raw-unsupported-null",
                "policy_generation_id": "entry-policy-unsupported-null",
                "full_fill": None,
                "entry_hedge_status": None,
                "entry_hedge_decision_time_ns": None,
                "entry_hedge_label_observed": False,
                "entry_hedge_executable": False,
                "entry_future_price": None,
                "entry_spot_price": None,
                "entry_hedge_contract_size_shares": None,
            }
        )
    return pl.from_dicts(rows, infer_schema_length=None)


def _exits(actions: pl.DataFrame) -> pl.DataFrame:
    records = []
    for row in actions.iter_rows(named=True):
        for rule, threshold in (("frozen_center", 0.0), ("frozen_lower", -10.0)):
            records.append(
                {
                    "Date": D1,
                    "ValueCode": VALUE,
                    "QuoteCode": QUOTE,
                    "route": row["route"],
                    "raw_order_fact_id": row["raw_order_fact_id"],
                    "policy_generation_id": row["policy_generation_id"],
                    "exit_rule_id": rule,
                    "exit_threshold_basis_bp": threshold,
                    "exit_rule_source_asof_date": "20260529",
                }
            )
    return pl.from_dicts(records, infer_schema_length=None)


def _calendar(expiry: str = D3) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "QuoteCode": [QUOTE],
            "expiry_session": [expiry],
            "calendar_version": ["official-test-v1"],
        }
    )


def _policy_row(
    frame: pl.DataFrame,
    *,
    rule: str = "frozen_center",
    route: str = FUTURE_BID_EXIT_ROUTE,
) -> dict[str, object]:
    selected = frame.filter(
        (pl.col("exit_rule_id") == rule) & (pl.col("exit_route") == route)
    )
    assert selected.height == 1
    return selected.row(0, named=True)


class CrossSessionExitMakerTests(unittest.TestCase):
    def test_unsupported_null_fill_is_excluded_from_entry_admission(self) -> None:
        actions = _actions(include_unsupported_null=True)
        result = replay_cross_session_exit_maker(
            actions,
            _exits(actions),
            (_session(D1), _session(D2)),
            (D1, D2, D3),
            _calendar(),
        )
        self.assertEqual(result.policy_outcomes.height, 4)
        audit = result.audit.row(0, named=True)
        self.assertEqual(audit["established_entry_policy_alias_rows"], 1)
        self.assertEqual(
            audit["excluded_entry_no_fill_or_unhedged_rows"], 1
        )

    def test_full_fill_with_null_hedge_status_remains_fail_closed(self) -> None:
        actions = _actions().with_columns(
            pl.lit(None, dtype=pl.String).alias("entry_hedge_status"),
            pl.lit(False).alias("entry_hedge_label_observed"),
            pl.lit(False).alias("entry_hedge_executable"),
        )
        with self.assertRaisesRegex(
            ValueError, "entry alias-local hedge admission facts"
        ):
            replay_cross_session_exit_maker(
                actions,
                _exits(actions),
                (_session(D1),),
                (D1, D2, D3),
                _calendar(),
            )

    def test_policy_only_spool_replays_once_for_strict_and_nominal(self) -> None:
        actions = _actions()
        exits = _exits(actions)
        entry_tape, entry_clock = _tape(D1)
        same_day = replay_exit_maker_product_day(
            actions,
            exits,
            entry_tape,
            entry_clock,
            ExitMakerStudyConfig(max_book_age_ns=None),
            cutoff_cursor=EventCursor(1_000 * MS, 3, 0),
        )
        candidate = _session(D2, future_trades=[(200, 100.0, 1)])

        def replay(semantics: str, **kwargs: object):
            return replay_cross_session_exit_maker(
                actions,
                exits,
                (candidate,),
                (D1, D2),
                _calendar(),
                CrossSessionExitMakerConfig(
                    cancel_semantics=semantics,  # type: ignore[arg-type]
                    lifecycle_policy_version="shared-physical-test-v1",
                ),
                same_day_result=same_day,
                **kwargs,
            )

        baseline = {
            semantics: replay(semantics)
            for semantics in ("strict", "nominal_instant_cancel_v0")
        }
        with tempfile.TemporaryDirectory() as directory:
            shared: dict[tuple[object, ...], pl.DataFrame] = {}
            original = (
                cross_session_module.replay_exit_maker_product_day_to_artifacts
            )
            with patch.object(
                cross_session_module,
                "replay_exit_maker_product_day_to_artifacts",
                wraps=original,
            ) as spooled:
                bounded = {
                    semantics: replay(
                        semantics,
                        shared_day_policy_facts=shared,
                        day_policy_spool_root=Path(directory),
                        retain_session_artifacts=False,
                    )
                    for semantics in (
                        "strict",
                        "nominal_instant_cancel_v0",
                    )
                }
            self.assertEqual(spooled.call_count, 1)
            self.assertEqual(len(shared), 1)
            self.assertEqual(
                list(Path(directory).glob(".cross-day-policy.tmp-*")), []
            )

        for semantics in baseline:
            self.assertTrue(
                baseline[semantics].policy_outcomes.equals(
                    bounded[semantics].policy_outcomes
                ),
                semantics,
            )
            self.assertTrue(
                baseline[semantics].session_attempts.equals(
                    bounded[semantics].session_attempts
                ),
                semantics,
            )
            self.assertTrue(bounded[semantics].candidate_aliases.is_empty())
            self.assertTrue(bounded[semantics].observations.is_empty())
            self.assertTrue(bounded[semantics].transitions.is_empty())

    def test_existing_same_day_terminal_keeps_identical_gross_for_overlay(self) -> None:
        actions = _actions()
        exits = _exits(actions)
        tape, clock = _tape(D1, future_trades=[(200, 100.0, 1)])
        same_day = replay_exit_maker_product_day(
            actions,
            exits,
            tape,
            clock,
            ExitMakerStudyConfig(max_book_age_ns=None),
            cutoff_cursor=EventCursor(1_000 * MS, 3, 0),
        )
        cross = replay_cross_session_exit_maker(
            actions,
            exits,
            [],
            [D1, D2, D3],
            _calendar(),
            CrossSessionExitMakerConfig(
                cancel_semantics="nominal_instant_cancel_v0"
            ),
            same_day_result=same_day,
            same_day_book_age_gate_enforced=False,
        )
        source = _policy_row(same_day.position_policy_facts)
        overlay = _policy_row(cross.policy_outcomes)
        self.assertEqual(overlay["outcome_type"], "terminal")
        self.assertEqual(overlay["terminal_branch"], "same_day_target_exit")
        self.assertEqual(overlay["filled_entry_outcome_category"], "completed")
        self.assertAlmostEqual(
            overlay["gross_cycle_pnl_twd"],
            source["gross_cycle_pnl_twd"],
        )
        self.assertEqual(overlay["terminal_date"], D1)
        self.assertEqual(overlay["terminal_reason"], "same_day_target_exit")

        report_actions = actions.with_columns(
            pl.lit(95).alias("boundary_quantile"),
            pl.lit(False).alias("partial_fill"),
            pl.lit(True).alias("any_fill"),
            pl.lit(False).alias("cancel_required"),
            (
                pl.col("entry_hedge_decision_time_ns") - 50_000_000
            ).alias("full_fill_recv_time_ns"),
            pl.lit(True).alias("entry_hedge_label_observed"),
            pl.lit(True).alias("entry_hedge_executable"),
        )
        empty = pl.DataFrame()
        report_inputs = ExitMakerPartitionInputs(
            policy_support=empty,
            candidate_aliases=empty,
            raw_candidate_facts=empty,
            position_policy_facts=same_day.position_policy_facts,
            action_facts=report_actions,
            taker_exit_facts=exits,
            coverage=pl.DataFrame(
                {
                    "Date": [D1],
                    "ValueCode": [VALUE],
                    "partition_complete": [True],
                    "partition": ["synthetic-cross-session"],
                }
            ),
            metadata={
                "selected_session_count": 1,
                "hedge_delay_ns": 50_000_000,
                "entry_action_hedge_delay_ns": 50_000_000,
                "exit_maker_hedge_delay_ns": 50_000_000,
                "session_predecessors": {D1: "20260529"},
            },
        )
        report = build_filled_entry_primary_report(
            report_inputs,
            terminal_policy_facts=cross.policy_outcomes,
        )
        path = report.filled_entry_policy_paths.filter(
            pl.col("exit_policy_trial_id") == overlay["exit_policy_trial_id"]
        ).row(0, named=True)
        self.assertEqual(path["filled_entry_outcome_category"], "completed")
        self.assertEqual(path["terminal_date"], D1)
        self.assertAlmostEqual(
            path["gross_cycle_pnl_twd"],
            source["gross_cycle_pnl_twd"],
        )

    def test_pure_session_adapter_never_needs_a_candidate_day_entry_row(self) -> None:
        tape, clock = _tape(
            D1,
            future_trades=[(200, 100.0, 1)],
        )
        replay = replay_exit_maker_session_policy(
            tape,
            clock,
            value_code=VALUE,
            exact_quote_code=QUOTE,
            route=FUTURE_BID_EXIT_ROUTE,
            threshold_basis_bp=0.0,
            start_time_ns=50 * MS,
            physical_policy_id="pure-session-policy",
            cutoff_cursor=EventCursor(1_000 * MS, 3, 0),
        )
        disposition = classify_exit_maker_session(replay)
        self.assertEqual(disposition.status, "flat_maker_taker")
        self.assertTrue(disposition.terminal)
        self.assertTrue(replay.candidate_session_keys)
        self.assertTrue(
            all(key.startswith(f"{D1}:") for key in replay.candidate_session_keys)
        )
        self.assertTrue(replay.day_order_queue_reset)
        self.assertFalse(replay.book_age_gate_enforced)

    def test_pure_session_adapter_rejects_a_roll_contract(self) -> None:
        tape, clock = _tape(D1, quote="CCFG6")
        with self.assertRaisesRegex(ValueError, "exact entry QuoteCode"):
            replay_exit_maker_session_policy(
                tape,
                clock,
                value_code=VALUE,
                exact_quote_code=QUOTE,
                route=FUTURE_BID_EXIT_ROUTE,
                threshold_basis_bp=0.0,
                start_time_ns=50 * MS,
                physical_policy_id="pure-session-policy",
                cutoff_cursor=EventCursor(1_000 * MS, 3, 0),
            )

    def test_carries_position_but_resets_day_order_queue_and_epoch_key(self) -> None:
        actions = _actions(include_no_fill=True)
        result = replay_cross_session_exit_maker(
            actions,
            _exits(actions),
            [
                _session(D1),
                _session(D2, future_trades=[(200, 100.0, 1)]),
            ],
            [D1, D2, D3],
            _calendar(),
        )
        # Only the actually established entry contributes four alternatives.
        self.assertEqual(result.policy_outcomes.height, 4)
        row = _policy_row(result.policy_outcomes)
        self.assertEqual(row["outcome_type"], "terminal")
        self.assertEqual(row["terminal_session_date"], D2)
        self.assertEqual(row["overnight_boundaries_crossed"], 1)
        self.assertEqual(row["sessions_attempted"], 2)
        self.assertAlmostEqual(row["gross_cycle_pnl_twd"], 4_000.0)
        attempts = result.session_attempts.filter(
            pl.col("exit_policy_trial_id") == row["exit_policy_trial_id"]
        ).sort("session_date")
        self.assertEqual(attempts["session_transition"].to_list(), ["carry", "terminal"])
        self.assertTrue(attempts["day_order_queue_reset"].all())
        self.assertTrue(attempts["frozen_threshold_unchanged"].all())
        candidates = result.candidate_aliases.filter(
            (pl.col("exit_rule_id") == "frozen_center")
            & (pl.col("exit_route") == FUTURE_BID_EXIT_ROUTE)
        )
        self.assertEqual(set(candidates["session_epoch_key"].to_list()), {f"{D1}:1", f"{D2}:1"})
        self.assertEqual(
            candidates.select("cross_session_candidate_id").n_unique(),
            candidates.height,
        )
        audit = result.audit.row(0, named=True)
        self.assertEqual(audit["excluded_entry_no_fill_or_unhedged_rows"], 1)
        self.assertTrue(audit["no_fill_entries_excluded_from_primary"])

    def test_later_day_replay_receives_only_still_active_policies(self) -> None:
        actions = _actions()
        observed: list[tuple[str, tuple[str, ...]]] = []
        original = cross_session_module.replay_exit_maker_product_day

        def wrapped(*args: object, **kwargs: object):
            frame = args[0]
            assert isinstance(frame, pl.DataFrame)
            observed.append(
                (
                    str(frame.item(0, "Date")),
                    tuple(kwargs["active_exit_policy_trial_ids"]),
                )
            )
            return original(*args, **kwargs)

        with patch.object(
            cross_session_module,
            "replay_exit_maker_product_day",
            side_effect=wrapped,
        ):
            result = replay_cross_session_exit_maker(
                actions,
                _exits(actions),
                [
                    _session(D1),
                    _session(D2, future_trades=[(200, 100.0, 1)]),
                    _session(D3),
                ],
                [D1, D2, D3],
                _calendar(),
            )
        self.assertEqual(
            [(date, len(policy_ids)) for date, policy_ids in observed],
            [(D1, 4), (D2, 3), (D3, 2)],
        )
        terminal = set(
            result.policy_outcomes.filter(
                pl.col("outcome_type") == "terminal"
            )["exit_policy_trial_id"].to_list()
        )
        d3_active = set(dict(observed)[D3])
        self.assertTrue(terminal)
        self.assertTrue(terminal.isdisjoint(d3_active))
        attempts = result.session_attempts.group_by("session_date").len().sort(
            "session_date"
        )
        self.assertEqual(attempts["len"].to_list(), [4, 3, 2])

    def test_candidate_day_reference_overrides_stale_entry_day_reference(self) -> None:
        actions = _actions()
        result = replay_cross_session_exit_maker(
            actions,
            _exits(actions),
            [
                _session(D1),
                # Raw rows still say 100, but the explicit D2 PIT reference is
                # 200.  Prices near 100 therefore fail the D2 reference gate.
                _session(
                    D2,
                    future_trades=[(200, 100.0, 1)],
                    raw_ref_price=100.0,
                    candidate_ref_price=200.0,
                ),
            ],
            [D1, D2, D3],
            _calendar(),
        )
        row = _policy_row(result.policy_outcomes)
        self.assertEqual(row["outcome_type"], "censored")
        self.assertEqual(row["outcome_status"], "right_censored_observation_end")
        d2 = result.observations.filter(
            (pl.col("session_date") == D2)
            & (pl.col("exit_rule_id") == "frozen_center")
            & (pl.col("exit_route") == FUTURE_BID_EXIT_ROUTE)
        )
        self.assertGreater(d2.height, 0)
        self.assertTrue(
            {"spot_ref_gate", "future_ref_gate"}
            & set(d2["gate_reason"].to_list())
        )
        self.assertEqual(d2["ref_price_source_date"].unique().to_list(), [D2])

    def test_old_book_is_diagnostic_not_a_hard_exit_gate(self) -> None:
        actions = _actions()
        result = replay_cross_session_exit_maker(
            actions,
            _exits(actions),
            [_session(D1, future_trades=[(200, 100.0, 1)])],
            [D1, D2, D3],
            _calendar(),
            config=CrossSessionExitMakerConfig(
                book_age_diagnostic_threshold_ns=1 * MS
            ),
        )
        row = _policy_row(result.policy_outcomes)
        self.assertEqual(row["outcome_type"], "terminal")
        self.assertEqual(row["book_age_diagnostic_status"], "diagnostic_over_threshold")
        self.assertGreater(row["max_book_age_ns"], 1 * MS)
        self.assertFalse(row["book_age_gate_enforced"])
        self.assertEqual(row["filled_entry_outcome_category"], "completed")
        self.assertEqual(row["filled_entry_terminal_date"], D1)
        self.assertAlmostEqual(row["gross_cycle_pnl_twd"], 4_000.0)

    def test_partial_spot_maker_exit_is_censored_and_never_restarted_full(self) -> None:
        actions = _actions()
        result = replay_cross_session_exit_maker(
            actions,
            _exits(actions),
            [
                # Ten displayed ask lots are ahead; eleven traded lots leave
                # one known maker lot filled out of the required two.
                _session(D1, spot_trades=[(200, 100.5, 11)]),
                _session(D2, spot_trades=[(200, 100.5, 2)]),
            ],
            [D1, D2, D3],
            _calendar(),
        )
        row = _policy_row(
            result.policy_outcomes,
            route=SPOT_ASK_EXIT_ROUTE,
        )
        self.assertEqual(row["outcome_type"], "censored")
        self.assertEqual(row["outcome_status"], "partial_exit_fill_unknown_position")
        attempts = result.session_attempts.filter(
            pl.col("exit_policy_trial_id") == row["exit_policy_trial_id"]
        )
        self.assertEqual(attempts.height, 1)
        self.assertEqual(attempts.item(0, "session_date"), D1)

    def test_incomplete_delayed_hedge_is_censored(self) -> None:
        actions = _actions()
        thin_update = _state(
            D1,
            "spot",
            225,
            3,
            bid=100.0,
            ask=100.5,
            bid_lots=1,
        )
        result = replay_cross_session_exit_maker(
            actions,
            _exits(actions),
            [
                _session(
                    D1,
                    future_trades=[(200, 100.0, 1)],
                    spot_updates=[thin_update],
                ),
                _session(D2),
            ],
            [D1, D2, D3],
            _calendar(),
        )
        row = _policy_row(result.policy_outcomes)
        self.assertEqual(row["outcome_type"], "censored")
        self.assertEqual(row["outcome_status"], "exit_hedge_incomplete")
        self.assertEqual(row["sessions_attempted"], 1)

    def test_missing_intermediate_session_fails_closed_before_later_day(self) -> None:
        actions = _actions()
        result = replay_cross_session_exit_maker(
            actions,
            _exits(actions),
            [
                _session(D1),
                _session(D3, future_trades=[(200, 100.0, 1)]),
            ],
            [D1, D2, D3],
            _calendar(),
        )
        row = _policy_row(result.policy_outcomes)
        self.assertEqual(row["outcome_type"], "censored")
        self.assertEqual(row["outcome_status"], "right_censored_missing_raw_session")
        self.assertEqual(row["last_observed_session_date"], D1)
        self.assertFalse(
            result.session_attempts["session_date"].eq(D3).any()
        )

    def test_incomplete_spread_clock_is_a_named_right_censor(self) -> None:
        actions = _actions()
        d2 = _session(D2)
        bad_d2 = CrossSessionExitMakerSession(
            **{
                **d2.__dict__,
                "spread_pair_clock": d2.spread_pair_clock.head(0),
            }
        )
        result = replay_cross_session_exit_maker(
            actions,
            _exits(actions),
            [_session(D1), bad_d2],
            [D1, D2, D3],
            _calendar(),
        )
        row = _policy_row(result.policy_outcomes)
        self.assertEqual(
            row["outcome_status"],
            "right_censored_missing_spread_clock",
        )
        self.assertEqual(row["filled_entry_outcome_category"], "censored")

    def test_expiry_without_settlement_is_unpriced_not_forced_taker(self) -> None:
        actions = _actions()
        result = replay_cross_session_exit_maker(
            actions,
            _exits(actions),
            [_session(D1)],
            [D1, D2, D3],
            _calendar(D1),
        )
        row = _policy_row(result.policy_outcomes)
        self.assertEqual(row["outcome_type"], "censored")
        self.assertEqual(
            row["outcome_status"],
            "right_censored_expiry_settlement_unpriced",
        )
        self.assertIsNone(row["gross_cycle_pnl_twd"])
        self.assertFalse(row["terminal_cashflow_priced"])
        self.assertFalse(result.audit.item(0, "forced_next_session_taker_taker_used"))

    def test_explicit_cancel_race_classifier_fails_closed(self) -> None:
        row = {
            "branch_status": "cancel_race_unknown",
            "nominal_instant_cancel_v0_branch": "flat_same_day",
            "needs_next_session_label": True,
            "terminal_outcome": False,
            "exit_hedge_status": "executable",
            "exit_spot_price": 100.0,
            "exit_future_price": 100.0,
            "gross_cycle_pnl_twd": 4_000.0,
        }
        strict = _classify_session_policy(row, cancel_semantics="strict")
        nominal = _classify_session_policy(
            row,
            cancel_semantics="nominal_instant_cancel_v0",
        )
        self.assertEqual(strict[0], "censored")
        self.assertEqual(strict[1], "cancel_race_unknown")
        self.assertEqual(nominal[0], "terminal")

    def test_exact_contract_is_not_rolled_and_can_resume_after_zero_event_day(self) -> None:
        actions = _actions()
        result = replay_cross_session_exit_maker(
            actions,
            _exits(actions),
            [
                _session(D1),
                _session(D2, quote="CCFG6"),
                _session(D3, future_trades=[(200, 100.0, 1)]),
            ],
            [D1, D2, D3],
            _calendar(),
        )
        row = _policy_row(result.policy_outcomes)
        self.assertEqual(row["outcome_type"], "terminal")
        self.assertEqual(row["terminal_session_date"], D3)
        attempts = result.session_attempts.filter(
            pl.col("exit_policy_trial_id") == row["exit_policy_trial_id"]
        ).sort("session_date")
        self.assertEqual(
            attempts["branch_status"].to_list(),
            [
                "carry_at_eod_cancel_unconfirmed",
                "carry_no_exact_contract_events",
                "flat_same_day",
            ],
        )

    def test_expiry_after_observed_calendar_is_an_observation_end_censor(self) -> None:
        actions = _actions()
        result = replay_cross_session_exit_maker(
            actions,
            _exits(actions),
            [_session(D1)],
            [D1, D2, D3],
            _calendar("20260617"),
        )
        row = _policy_row(result.policy_outcomes)
        self.assertEqual(row["outcome_status"], "right_censored_observation_end")
        self.assertEqual(row["filled_entry_outcome_category"], "still_open")
        self.assertEqual(row["expiry_session"], "20260617")

    def test_ref_lineage_must_be_candidate_session_date(self) -> None:
        session = _session(D2)
        invalid = CrossSessionExitMakerSession(
            **{
                **session.__dict__,
                "ref_price_source_date": D1,
            }
        )
        with self.assertRaisesRegex(ValueError, "source date"):
            invalid.validate()


if __name__ == "__main__":
    unittest.main()
