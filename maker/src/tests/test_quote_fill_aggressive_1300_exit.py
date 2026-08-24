from __future__ import annotations

from datetime import datetime
import unittest

import polars as pl

from maker.src.quote_fill.aggressive_1300_exit import (
    Aggressive1300ProductDayReplayCache,
    aggressive_1300_start_cursor,
    replay_aggressive_1300_inventory_batch,
    replay_aggressive_1300_product_day,
    select_aggressive_1300_inventory,
    select_open_positions_at_1300,
)
from maker.src.quote_fill.exit_maker import (
    FUTURE_BID_EXIT_ROUTE,
    SPOT_ASK_EXIT_ROUTE,
)
from maker.src.quote_fill.merged import session_cutoff_cursor
from maker.src.quote_fill.raw_tape import RawTapeDay
from maker.src.quote_fill.targets import absolute_price_tick


DATE = "20260102"
VALUE_CODE = "2330"
QUOTE_CODE = "CDF1"
MS = 1_000_000


def _start_ns() -> int:
    return aggressive_1300_start_cursor(DATE).recv_time_ns


def _state(
    market: str,
    offset_ms: int,
    sequence: int,
    *,
    bid: float,
    ask: float,
    bid_lots: int = 3,
    ask_lots: int = 3,
    raw_has_book: bool = True,
    trial_match: bool = False,
) -> dict[str, object]:
    recv_ns = _start_ns() + offset_ms * MS
    row: dict[str, object] = {
        "Date": DATE,
        "market": market,
        "ValueCode": VALUE_CODE,
        "QuoteCode": QUOTE_CODE,
        "instrument_code": VALUE_CODE if market == "spot" else QUOTE_CODE,
        "recv_time_ns": recv_ns,
        "sequence": sequence,
        "packet_sequence": sequence,
        "trial_match": trial_match,
        "ref_price": 100.0,
        "contract_size": 2_000.0,
        "raw_has_book": raw_has_book,
        "book_state_available": True,
        "book_recv_time_ns": recv_ns,
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


def _trade(
    market: str,
    offset_ms: int,
    sequence: int,
    *,
    price: float,
    lots: int,
) -> dict[str, object]:
    return {
        "Date": DATE,
        "market": market,
        "ValueCode": VALUE_CODE,
        "QuoteCode": QUOTE_CODE,
        "instrument_code": VALUE_CODE if market == "spot" else QUOTE_CODE,
        "recv_time_ns": _start_ns() + offset_ms * MS,
        "sequence": sequence,
        "packet_sequence": sequence,
        "trade_price": price,
        "trade_lots": lots,
    }


def _empty_trades() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "Date": pl.String,
            "market": pl.String,
            "ValueCode": pl.String,
            "QuoteCode": pl.String,
            "instrument_code": pl.String,
            "recv_time_ns": pl.Int64,
            "sequence": pl.Int64,
            "packet_sequence": pl.Int64,
            "trade_price": pl.Float64,
            "trade_lots": pl.Int64,
        }
    )


def _tape(
    spot_states: list[dict[str, object]],
    future_states: list[dict[str, object]],
    *,
    spot_trades: list[dict[str, object]] | None = None,
    future_trades: list[dict[str, object]] | None = None,
) -> tuple[RawTapeDay, pl.DataFrame]:
    mapping = pl.DataFrame(
        {
            "ValueCode": [VALUE_CODE],
            "QuoteCode": [QUOTE_CODE],
            "spot_ref_price": [100.0],
            "fut_ref_price": [100.0],
            "contract_size": [2_000.0],
        }
    )
    spot = pl.from_dicts(spot_states, infer_schema_length=None)
    future = pl.from_dicts(future_states, infer_schema_length=None)
    spot_trade_frame = (
        _empty_trades()
        if not spot_trades
        else pl.from_dicts(spot_trades, infer_schema_length=None)
    )
    future_trade_frame = (
        _empty_trades()
        if not future_trades
        else pl.from_dicts(future_trades, infer_schema_length=None)
    )
    tape = RawTapeDay(
        DATE,
        mapping,
        spot,
        future,
        spot_trade_frame,
        future_trade_frame,
        pl.DataFrame(),
    )
    clock = pl.DataFrame(
        {
            "Date": [DATE] * spot.height,
            "ValueCode": [VALUE_CODE] * spot.height,
            "spot_channel_seq": spot["sequence"],
            "spread_pair_id": [1] * spot.height,
            "spread_pair_epoch": [1] * spot.height,
        }
    )
    return tape, clock


def _constant_books_with_trades(
    *, future_fill_ms: int, spot_fill_ms: int
) -> tuple[RawTapeDay, pl.DataFrame]:
    spot_states = [
        _state("spot", -1, 1, bid=100.0, ask=101.0, bid_lots=20, ask_lots=3),
        _state("spot", 60, 2, bid=99.5, ask=101.0, bid_lots=20, ask_lots=3),
        _state("spot", 100, 3, bid=99.5, ask=101.0, bid_lots=20, ask_lots=3),
    ]
    future_states = [
        _state("future", -1, 10, bid=100.0, ask=101.0, bid_lots=1, ask_lots=20),
        _state("future", 60, 11, bid=100.0, ask=101.5, bid_lots=1, ask_lots=20),
        _state("future", 100, 12, bid=100.0, ask=101.5, bid_lots=1, ask_lots=20),
    ]
    return _tape(
        spot_states,
        future_states,
        spot_trades=[
            _trade("spot", spot_fill_ms, 20, price=101.0, lots=5),
        ],
        future_trades=[
            _trade("future", future_fill_ms, 30, price=100.0, lots=2),
        ],
    )


def _action(
    policy_id: str,
    raw_id: str,
    *,
    full: bool,
    full_fill_ns: int | None,
    decision_ns: int | None,
    status: str | None,
    observed: bool,
    executable: bool,
) -> dict[str, object]:
    return {
        "Date": DATE,
        "ValueCode": VALUE_CODE,
        "QuoteCode": QUOTE_CODE,
        "route": "future_ask_spot_taker",
        "raw_order_fact_id": raw_id,
        "policy_generation_id": policy_id,
        "any_fill": full,
        "full_fill": full,
        "partial_fill": False,
        "full_fill_recv_time_ns": full_fill_ns,
        "entry_hedge_status": status,
        "entry_hedge_decision_time_ns": decision_ns,
        "entry_hedge_label_observed": observed,
        "entry_hedge_executable": executable,
        "entry_future_price": 102.0,
        "entry_spot_price": 100.0,
        "entry_hedge_contract_size_shares": 2_000,
    }


_SELECTED_PATH_SCHEMA = {
    "Date": pl.String,
    "ValueCode": pl.String,
    "QuoteCode": pl.String,
    "entry_raw_order_fact_id": pl.String,
    "position_established_ns": pl.Int64,
    "policy_path_id": pl.String,
    "exit_policy_trial_id": pl.String,
    "terminal_date": pl.String,
    "exit_decision_time_ns": pl.Int64,
    "terminal_cashflow_priced": pl.Boolean,
    "outcome_type": pl.String,
    "outcome_status": pl.String,
    "last_observed_session_date": pl.String,
    "outstanding_interval_end_exclusive": pl.String,
    "entry_no_fill_included": pl.Boolean,
    "gross_zero_imputation": pl.Boolean,
}


def _selected_path(
    physical_id: str,
    *,
    origin_date: str = DATE,
    established_ns: int | None = None,
    terminal_date: str | None = None,
    terminal_ns: int | None = None,
    last_observed: str = DATE,
) -> dict[str, object]:
    priced = terminal_date is not None
    if established_ns is None:
        established_ns = aggressive_1300_start_cursor(origin_date).recv_time_ns - MS
    return {
        "Date": origin_date,
        "ValueCode": VALUE_CODE,
        "QuoteCode": QUOTE_CODE,
        "entry_raw_order_fact_id": physical_id,
        "position_established_ns": established_ns,
        "policy_path_id": f"path/{physical_id}",
        "exit_policy_trial_id": f"trial/{physical_id}",
        "terminal_date": terminal_date,
        "exit_decision_time_ns": terminal_ns,
        "terminal_cashflow_priced": priced,
        "outcome_type": "terminal" if priced else "censored",
        "outcome_status": "same_day_target_exit" if priced else "maker_fill_state_unknown",
        "last_observed_session_date": last_observed,
        "outstanding_interval_end_exclusive": last_observed,
        "entry_no_fill_included": False,
        "gross_zero_imputation": False,
    }


def _selected_paths(*rows: dict[str, object]) -> pl.DataFrame:
    return pl.from_dicts(list(rows), schema=_SELECTED_PATH_SCHEMA, strict=True)


class Aggressive1300ExitTest(unittest.TestCase):
    def test_start_cursor_reuses_existing_taipei_session_axis(self) -> None:
        start = aggressive_1300_start_cursor(DATE)
        cutoff = session_cutoff_cursor(DATE)
        expected = datetime(2026, 1, 2, 5, 0) - datetime(1970, 1, 1)
        self.assertEqual(start.recv_time_ns, int(expected.total_seconds() * 1e9))
        self.assertEqual(cutoff.recv_time_ns - start.recv_time_ns, 20 * 60 * 10**9)

    def test_dynamic_pegs_keep_forward_layer_and_cancel_it_on_retreat(self) -> None:
        tape, clock = _tape(
            [
                _state("spot", -1, 1, bid=99.0, ask=101.0),
                _state("spot", 110, 2, bid=99.0, ask=100.0),
                _state("spot", 210, 3, bid=99.0, ask=101.0),
            ],
            [
                _state("future", -1, 10, bid=100.0, ask=102.0),
                _state("future", 100, 11, bid=101.0, ask=102.0),
                _state("future", 200, 12, bid=100.0, ask=102.0),
            ],
        )
        result = replay_aggressive_1300_product_day(
            tape,
            clock,
            position_id="physical-entry-1",
            position_established_recv_time_ns=_start_ns() - 1,
            position_open_at_start=True,
        )

        future = result.builds_by_route[FUTURE_BID_EXIT_ROUTE]
        spot = result.builds_by_route[SPOT_ASK_EXIT_ROUTE]
        self.assertEqual(
            [(row.kind, row.reason) for row in future.transitions[:3]],
            [
                ("submit", "new_epoch"),
                ("submit", "forward_new_price"),
                ("cancel", "target_retreat"),
            ],
        )
        self.assertEqual(
            [(row.kind, row.reason) for row in spot.transitions[:3]],
            [
                ("submit", "new_epoch"),
                ("submit", "forward_new_price"),
                ("cancel", "target_retreat"),
            ],
        )
        self.assertEqual(len(future.windows), 2)
        self.assertEqual(len(spot.windows), 2)
        old_future = min(future.windows, key=lambda row: row.start_cursor)
        self.assertEqual(old_future.initial_queue_ahead, 3)
        self.assertEqual(old_future.stop_reason, "session_cutoff")
        advanced_future = max(future.windows, key=lambda row: row.start_cursor)
        self.assertEqual(advanced_future.stop_reason, "target_retreat")
        self.assertEqual(
            advanced_future.target_price_tick,
            absolute_price_tick(101.0, market="future", session_date=DATE),
        )
        self.assertEqual(result.nominal_close_count, 0)

    def test_future_winner_hedges_at_50ms_and_sibling_never_closes_twice(self) -> None:
        tape, clock = _constant_books_with_trades(future_fill_ms=10, spot_fill_ms=20)
        result = replay_aggressive_1300_product_day(
            tape,
            clock,
            position_id="physical-entry-1",
            position_established_recv_time_ns=_start_ns() - 1,
            position_open_at_start=True,
        )

        self.assertEqual(result.winner_route, FUTURE_BID_EXIT_ROUTE)
        self.assertEqual(result.nominal_branch_status, "flat_same_day")
        self.assertEqual(result.strict_branch_status, "cancel_race_unknown")
        self.assertEqual(result.nominal_close_count, 1)
        self.assertEqual(result.active_sibling_cancel_count, 1)
        self.assertTrue(result.projection.position_projection_safe)
        winner = result.winner_outcome
        assert winner is not None
        self.assertEqual(len(winner.hedge_attempts), 1)
        attempt = winner.hedge_attempts[0]
        self.assertEqual(
            attempt.decision_time_ns - attempt.maker_fill_cursor.recv_time_ns,
            50_000_000,
        )
        self.assertEqual(attempt.status, "executable")
        summary = result.summary_record()
        self.assertEqual(summary["sibling_full_fill_after_cancel_request_count"], 1)
        self.assertTrue(summary["duplicate_close_prevented"])

    def test_spot_winner_uses_future_buy_hedge(self) -> None:
        tape, clock = _constant_books_with_trades(future_fill_ms=20, spot_fill_ms=10)
        result = replay_aggressive_1300_product_day(
            tape,
            clock,
            position_id="physical-entry-1",
            position_established_recv_time_ns=_start_ns() - 1,
            position_open_at_start=True,
        )
        self.assertEqual(result.winner_route, SPOT_ASK_EXIT_ROUTE)
        winner = result.winner_outcome
        assert winner is not None
        self.assertTrue(winner.position_flat)
        self.assertEqual(winner.hedge_attempts[0].status, "executable")
        self.assertEqual(
            winner.hedge_attempts[0].execution.hedge_side,  # type: ignore[union-attr]
            "buy",
        )
        self.assertEqual(result.nominal_close_count, 1)

    def test_post_1300_position_is_rejected(self) -> None:
        tape, clock = _constant_books_with_trades(future_fill_ms=10, spot_fill_ms=20)
        with self.assertRaisesRegex(ValueError, "entry cutoff"):
            replay_aggressive_1300_product_day(
                tape,
                clock,
                position_id="late-entry",
                position_established_recv_time_ns=_start_ns() + 1,
                position_open_at_start=True,
            )

        with self.assertRaisesRegex(ValueError, "entry cutoff"):
            replay_aggressive_1300_product_day(
                tape,
                clock,
                position_id="exactly-at-cutoff",
                position_established_recv_time_ns=_start_ns(),
                position_open_at_start=True,
            )
        with self.assertRaisesRegex(ValueError, "proven open"):
            replay_aggressive_1300_product_day(
                tape,
                clock,
                position_id="not-proven-open",
                position_established_recv_time_ns=_start_ns() - 1,
                position_open_at_start=False,
            )

    def test_inventory_is_alias_local_cutoff_bound_and_physically_deduped(self) -> None:
        before_fill = _start_ns() - 100 * MS
        before_hedge = before_fill + 50 * MS
        after_fill = _start_ns() + 100 * MS
        after_hedge = after_fill + 50 * MS
        rows = [
            _action(
                "q50",
                "shared-before",
                full=True,
                full_fill_ns=before_fill,
                decision_ns=before_hedge,
                status="executable",
                observed=True,
                executable=True,
            ),
            _action(
                "q80",
                "shared-before",
                full=True,
                full_fill_ns=before_fill,
                decision_ns=before_hedge,
                status="executable",
                observed=True,
                executable=True,
            ),
            _action(
                "q95-no-fill",
                "shared-before",
                full=False,
                full_fill_ns=None,
                decision_ns=before_hedge,
                status="executable",
                observed=False,
                executable=False,
            ),
            _action(
                "late-q50",
                "late-physical",
                full=True,
                full_fill_ns=after_fill,
                decision_ns=after_hedge,
                status="executable",
                observed=True,
                executable=True,
            ),
        ]
        actions = pl.from_dicts(rows, infer_schema_length=None).with_columns(
            pl.col("full_fill_recv_time_ns").cast(pl.Int64),
            pl.col("entry_hedge_decision_time_ns").cast(pl.Int64),
        )
        selected = select_aggressive_1300_inventory(actions)
        self.assertEqual(selected.inventory.height, 1)
        self.assertEqual(
            selected.inventory.item(0, "entry_raw_order_fact_id"), "shared-before"
        )
        self.assertEqual(selected.inventory.item(0, "eligible_policy_alias_count"), 2)
        self.assertEqual(
            selected.inventory.item(
                0, "eligible_policy_generation_ids"
            ).to_list(),
            ["q50", "q80"],
        )
        audit = selected.audit.row(0, named=True)
        self.assertEqual(audit["physical_paired_positions_at_1300"], 1)
        self.assertEqual(audit["established_after_1300_policy_aliases_excluded"], 1)
        self.assertEqual(audit["post_1300_new_entries_admitted"], 0)

    def test_open_inventory_uses_normal_terminal_state_and_preserves_carry(self) -> None:
        prior = "20260101"
        later = "20260105"
        paths = _selected_paths(
            _selected_path(
                "same-day-terminal-before",
                terminal_date=DATE,
                terminal_ns=_start_ns(),
            ),
            _selected_path(
                "same-day-terminal-after",
                terminal_date=DATE,
                terminal_ns=_start_ns() + MS,
            ),
            _selected_path(
                "same-day-late-entry",
                established_ns=_start_ns(),
                last_observed=DATE,
            ),
            _selected_path(
                "carry-unresolved",
                origin_date=prior,
                last_observed=DATE,
            ),
            _selected_path(
                "carry-future-terminal",
                origin_date=prior,
                terminal_date=later,
                terminal_ns=aggressive_1300_start_cursor(later).recv_time_ns + MS,
                last_observed=later,
            ),
            _selected_path(
                "already-flat-prior",
                origin_date=prior,
                terminal_date=prior,
                terminal_ns=aggressive_1300_start_cursor(prior).recv_time_ns + MS,
                last_observed=prior,
            ),
            _selected_path(
                "stale-unresolved",
                origin_date=prior,
                last_observed=prior,
            ),
            _selected_path(
                "not-yet-originated",
                origin_date=later,
                last_observed=later,
            ),
        )
        selected = select_open_positions_at_1300(paths, target_date=DATE)
        self.assertEqual(
            selected.inventory["position_id"].to_list(),
            [
                "carry-future-terminal",
                "carry-unresolved",
                "same-day-terminal-after",
            ],
        )
        by_id = {
            row["position_id"]: row
            for row in selected.inventory.iter_rows(named=True)
        }
        self.assertEqual(
            by_id["carry-unresolved"]["inventory_origin"],
            "carried_from_prior_session",
        )
        self.assertEqual(
            by_id["same-day-terminal-after"]["normal_path_status_at_1300"],
            "normal_terminal_after_1300",
        )
        self.assertTrue(all(selected.inventory["open_at_1300"]))
        audit = selected.audit.row(0, named=True)
        self.assertEqual(audit["selected_physical_paths"], 8)
        self.assertEqual(audit["open_physical_positions_at_1300"], 3)
        self.assertEqual(audit["normal_terminal_by_1300_paths_excluded"], 1)
        self.assertEqual(
            audit["target_day_entries_at_or_after_1300_excluded"], 1
        )
        self.assertEqual(audit["carried_open_positions"], 2)
        self.assertEqual(audit["new_entries_at_or_after_1300_admitted"], 0)

    def test_open_inventory_rejects_duplicate_or_incoherent_selected_path(self) -> None:
        duplicate = _selected_path("same-physical")
        with self.assertRaisesRegex(ValueError, "one row per physical"):
            select_open_positions_at_1300(
                _selected_paths(duplicate, {**duplicate, "policy_path_id": "path/other"}),
                target_date=DATE,
            )
        incoherent = _selected_paths(_selected_path("bad-terminal")).with_columns(
            pl.lit(True).alias("terminal_cashflow_priced")
        )
        with self.assertRaisesRegex(ValueError, "coherent terminal"):
            select_open_positions_at_1300(incoherent, target_date=DATE)

    def test_batch_cache_builds_market_replay_once_and_never_claims_joint_volume(self) -> None:
        prior = "20260101"
        inventory = select_open_positions_at_1300(
            _selected_paths(
                _selected_path("same-day-open", last_observed=DATE),
                _selected_path(
                    "carried-open",
                    origin_date=prior,
                    last_observed=DATE,
                ),
            ),
            target_date=DATE,
        ).inventory
        tape, clock = _constant_books_with_trades(future_fill_ms=10, spot_fill_ms=20)
        cache = Aggressive1300ProductDayReplayCache()
        first = replay_aggressive_1300_inventory_batch(
            tape,
            clock,
            inventory,
            source_cache_key="immutable-synth-tape-v1",
            cache=cache,
        )
        self.assertEqual(cache.misses, 1)
        self.assertEqual(cache.hits, 0)
        self.assertEqual(first.position_outcomes.height, 2)
        self.assertEqual(first.position_outcomes["position_id"].n_unique(), 2)
        self.assertEqual(first.position_outcomes["nominal_close_count"].to_list(), [1, 1])
        self.assertEqual(first.position_outcomes["winner_generation_id"].n_unique(), 2)
        self.assertFalse(any(first.position_outcomes["joint_volume_allocated"]))
        self.assertFalse(any(first.position_outcomes["position_outcomes_safe_to_sum"]))
        self.assertTrue(all(first.position_outcomes["duplicate_close_prevented"]))
        first_audit = first.audit.row(0, named=True)
        self.assertEqual(first_audit["market_templates_built_this_call"], 1)
        self.assertFalse(first_audit["market_template_cache_hit"])
        self.assertFalse(first_audit["portfolio_terminal_metric_available"])

        second = replay_aggressive_1300_inventory_batch(
            tape,
            clock,
            inventory,
            source_cache_key="immutable-synth-tape-v1",
            cache=cache,
        )
        self.assertEqual(cache.misses, 1)
        self.assertEqual(cache.hits, 1)
        self.assertIs(first.market_template, second.market_template)
        self.assertTrue(second.audit.item(0, "market_template_cache_hit"))
        self.assertEqual(second.audit.item(0, "market_templates_built_this_call"), 0)

    def test_batch_recomputes_open_state_before_touching_market_cache(self) -> None:
        inventory = select_open_positions_at_1300(
            _selected_paths(_selected_path("open-before-cutoff", last_observed=DATE)),
            target_date=DATE,
        ).inventory
        tape, clock = _constant_books_with_trades(future_fill_ms=10, spot_fill_ms=20)
        cache = Aggressive1300ProductDayReplayCache()

        late = inventory.with_columns(
            pl.lit(_start_ns()).alias("position_established_recv_time_ns")
        )
        with self.assertRaisesRegex(ValueError, "at or after 13:00"):
            replay_aggressive_1300_inventory_batch(
                tape,
                clock,
                late,
                source_cache_key="immutable-synth-tape-v1",
                cache=cache,
            )
        self.assertEqual(cache.misses, 0)

        already_flat = inventory.with_columns(
            pl.lit(True).alias("normal_terminal_cashflow_priced"),
            pl.lit(DATE).alias("normal_terminal_date"),
            pl.lit(_start_ns()).cast(pl.Int64).alias("normal_exit_decision_time_ns"),
            pl.lit("terminal").alias("normal_outcome_type"),
            pl.lit("same_day_target_exit").alias("normal_outcome_status"),
            pl.lit("normal_terminal_after_1300").alias(
                "normal_path_status_at_1300"
            ),
        )
        with self.assertRaisesRegex(ValueError, "already flat"):
            replay_aggressive_1300_inventory_batch(
                tape,
                clock,
                already_flat,
                source_cache_key="immutable-synth-tape-v1",
                cache=cache,
            )
        self.assertEqual(cache.misses, 0)


if __name__ == "__main__":
    unittest.main()
