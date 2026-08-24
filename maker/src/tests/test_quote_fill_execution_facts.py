from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import polars as pl

from maker.src.quote_fill.execution_facts import (
    ExecutableExitPathConfig,
    build_executable_exit_facts,
    build_executable_taker_exit_path,
    build_execution_action_facts,
    summarize_executable_exit_daily,
    summarize_execution_daily,
)
from maker.src.quote_fill.execution_runner import (
    ExecutionRunnerConfig,
    load_walkforward_execution_product_day,
    load_walkforward_product_day_sources,
    replay_execution_product_day,
    run_partitioned_execution_replay,
)
from maker.src.quote_fill.merged import (
    MergedTargetStudyInput,
    session_cutoff_cursor,
)
from maker.src.quote_fill.raw_tape import RawTapeDay
from maker.src.quote_fill.targets import (
    FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE,
    PRICE_LADDER_VERSION,
)
from maker.src.quote_width.rolling import (
    RollingBoundaryConfig,
    _partitioned_daily_source_provenance,
    write_rolling_boundary_snapshots,
)


DATE = "20260102"
VALUE_CODE = "2317"
QUOTE_CODE = "DHFA6"
MS = 1_000_000


def _mapping() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ValueCode": [VALUE_CODE],
            "QuoteCode": [QUOTE_CODE],
            "spot_ref_price": [100.0],
            "fut_ref_price": [101.0],
            "contract_size": [2000.0],
        }
    )


def _write_walkforward_source_fixture(
    root: Path,
    *,
    boundary_quote_code: str = QUOTE_CODE,
    source_asof_date: str = "20260101",
) -> tuple[Path, Path]:
    daily_root = root / "daily"
    partition = daily_root / f"Date={DATE}"
    partition.mkdir(parents=True)
    pl.DataFrame(
        {
            "Date": [DATE],
            "ValueCode": [VALUE_CODE],
            "QuoteCode": [QUOTE_CODE],
            "spot_ref_price": [100.0],
            "fut_ref_price": [101.0],
            "contract_size": [2000.0],
            # This realised field must never survive the loader allowlist.
            "trading_turnover": [999_999.0],
        }
    ).write_parquet(partition / "mapping.parquet")
    pl.DataFrame(
        {
            "Date": [DATE, DATE],
            "ValueCode": [VALUE_CODE, VALUE_CODE],
            "QuoteCode": [QUOTE_CODE, QUOTE_CODE],
            "timestamp": [
                datetime(2026, 1, 2, 1, 5, 0),
                datetime(2026, 1, 2, 1, 5, 1),
            ],
            "spot_ref_price": [100.0, 100.0],
            "fut_ref_price": [101.0, 101.0],
            "contract_size": [2000.0, 2000.0],
            "anchor_ewma_120s_bp": [10.0, 11.0],
            # This outcome-like extra column is deliberately not selected.
            "future_excursion_bp": [123.0, 456.0],
        }
    ).write_parquet(partition / "causal_fair.parquet")
    (partition / "complete.json").write_text("{}\n", encoding="utf-8")

    boundary_path = root / "rolling_boundary_snapshots.parquet"
    boundary_config = RollingBoundaryConfig(
        parameter_version="rolling-test-v1"
    )
    boundaries = pl.DataFrame(
        {
            "Date": [DATE] * 3,
            "ValueCode": [VALUE_CODE] * 3,
            "QuoteCode": [boundary_quote_code] * 3,
            "contract_size": [2000.0] * 3,
            "fut_ref_price": [101.0] * 3,
            "spot_ref_price": [100.0] * 3,
            "boundary_quantile": [50, 80, 95],
            "boundary_role": [
                "rolling_latent_candidate",
                "rolling_latent_candidate",
                "tail_diagnostic",
            ],
            "upper_distance_bp": [10.0, 20.0, 30.0],
            "lower_distance_bp": [9.0, 19.0, 29.0],
            "adaptive_parameter_valid": [True] * 3,
            "source_asof_date": [source_asof_date] * 3,
            "parameter_version": ["rolling-test-v1"] * 3,
            "price_ladder_version": [PRICE_LADDER_VERSION] * 3,
            "future_one_dollar_tick_effective_date": [
                FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE
            ] * 3,
            "lookback_sessions": [boundary_config.lookback_sessions] * 3,
            "probability_layer": ["rolling_latent_price_path_prior"] * 3,
            "execution_safe_snapshot": [True] * 3,
            "contains_target_day_outcome": [False] * 3,
        }
    )
    write_rolling_boundary_snapshots(
        boundaries,
        root,
        boundary_config,
        source_provenance=_partitioned_daily_source_provenance(daily_root),
    )
    return daily_root, boundary_path


def _validated_daily_marker_payload() -> dict[str, object]:
    return {
        "schema_version": "daily_latent_facts_v2_migrated_nonatomic",
        "migrated_legacy_marker": True,
        "artifacts": {
            "causal_fair.parquet": {
                "execution_safe_causal_panel": True,
                "contains_target_day_outcome": False,
            },
            "mapping.parquet": {
                "consumer_must_use_preopen_allowlist": True,
                "contains_target_day_trading_turnover": True,
            },
        },
    }


def _state(
    market: str,
    timestamp_ns: int,
    sequence: int,
    *,
    bid: float,
    ask: float,
    bid_lots: int = 5,
    ask_lots: int = 5,
    trial_match: bool = False,
    raw_has_book: bool = True,
) -> dict[str, object]:
    row: dict[str, object] = {
        "Date": DATE,
        "market": market,
        "ValueCode": VALUE_CODE,
        "QuoteCode": QUOTE_CODE,
        "recv_time_ns": timestamp_ns,
        "sequence": sequence,
        "packet_sequence": sequence,
        "trial_match": trial_match,
        "ref_price": 100.0 if market == "spot" else 101.0,
        "book_state_available": True,
        "book_recv_time_ns": timestamp_ns,
        "raw_has_book": raw_has_book,
        "best_bid_price": None,
        "best_bid_lots": None,
        "best_ask_price": None,
        "best_ask_lots": None,
    }
    for level in range(1, 6):
        populated = level <= 2
        row[f"bid_price_{level}"] = (
            bid - (level - 1) * 0.5 if populated else None
        )
        row[f"bid_lots_{level}"] = bid_lots if populated else None
        row[f"ask_price_{level}"] = (
            ask + (level - 1) * 0.5 if populated else None
        )
        row[f"ask_lots_{level}"] = ask_lots if populated else None
    return row


def _raw_tape_for_exit() -> RawTapeDay:
    spot = pl.from_dicts(
        [
            _state("spot", 100, 1, bid=99.0, ask=100.0),
            _state("spot", 200, 2, bid=100.0, ask=101.0),
        ],
        infer_schema_length=None,
    )
    future = pl.from_dicts(
        [
            _state("future", 100, 1, bid=100.0, ask=101.0),
            _state("future", 200, 2, bid=100.0, ask=100.5),
        ],
        infer_schema_length=None,
    )
    empty = pl.DataFrame()
    return RawTapeDay(
        DATE,
        _mapping(),
        spot,
        future,
        empty,
        empty,
        pl.DataFrame({"Date": [DATE], "ValueCode": [VALUE_CODE]}),
    )


def _alias_frame() -> pl.DataFrame:
    common: dict[str, object] = {
        "Date": DATE,
        "ValueCode": VALUE_CODE,
        "QuoteCode": QUOTE_CODE,
        "route": "future_ask_spot_taker",
        "spread_pair_epoch": 10,
        "target_price_tick": 2400,
        "target_price": 102.0,
        "target_rank_at_submit": "ASK1",
        "initial_queue_ahead": 1,
        "queue_known": True,
        "intended_quantity": 1,
        "submit_recv_time_ns": 100,
        "submit_event_sequence": 0,
        "submit_row_index": 1,
        "nominal_stop_recv_time_ns": 500,
        "nominal_stop_reason": "target_retreat",
        "first_fill_recv_time_ns": 150,
        "first_fill_event_sequence": 1,
        "first_fill_row_index": 3,
        "full_fill_recv_time_ns": 150,
        "full_fill_event_sequence": 1,
        "full_fill_row_index": 3,
        "known_filled_quantity": 1,
        "any_fill": True,
        "full_fill": True,
        "partial_fill": False,
        "cancel_required": False,
        "spot_book_age_ms_at_submit": 10.0,
        "future_book_age_ms_at_submit": 20.0,
        "source_asof_date": "20260101",
    }
    rows = [
        {
            **common,
            "boundary_quantile": quantile,
            "raw_order_fact_id": "shared-raw",
            "policy_generation_id": f"full-q{quantile}",
        }
        for quantile in (50, 80)
    ]
    rows.append(
        {
            **common,
            "boundary_quantile": 50,
            "raw_order_fact_id": "partial-raw",
            "policy_generation_id": "partial-q50",
            "spread_pair_epoch": 11,
            "submit_recv_time_ns": 600,
            "submit_row_index": 2,
            "nominal_stop_recv_time_ns": 900,
            "first_fill_recv_time_ns": 700,
            "first_fill_row_index": 4,
            "full_fill_recv_time_ns": None,
            "full_fill_event_sequence": None,
            "full_fill_row_index": None,
            "known_filled_quantity": 1,
            "full_fill": False,
            "partial_fill": True,
            "cancel_required": True,
        }
    )
    return pl.from_dicts(rows, infer_schema_length=None)


def _hedge_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "raw_order_fact_id": ["shared-raw"],
            "status": ["executable"],
            "decision_time_ns": [200],
            "decision_snapshot_recv_time_ns": [200],
            "decision_snapshot_event_sequence": [2],
            "decision_snapshot_row_index": [2],
            "arrival_reference_price": [100.0],
            "decision_best_price": [100.0],
            "executable_vwap_price": [100.0],
            "available_quantity": [5],
            "executed_quantity": [2],
            "depth_shortfall": [0],
            "levels_swept": [1],
            "signed_latency_slippage_bp": [1.0],
            "signed_depth_slippage_bp": [2.0],
            "signed_total_slippage_bp": [3.0],
            "decision_book_age_ms": [0.0],
            "contract_size_shares": [2000],
        }
    )


def _merged_product_day() -> MergedTargetStudyInput:
    cutoff = session_cutoff_cursor(DATE).recv_time_ns
    submit = cutoff - 2_000 * MS
    fill = submit + 100 * MS
    spot = pl.from_dicts(
        [
            _state("spot", submit - 1, 1, bid=99.0, ask=100.0),
            _state("spot", fill + 50 * MS, 2, bid=99.5, ask=100.5),
        ],
        infer_schema_length=None,
    )
    future = pl.from_dicts(
        [
            _state("future", submit - 1, 1, bid=100.0, ask=101.0),
            _state("future", fill + 50 * MS, 2, bid=100.0, ask=101.0),
        ],
        infer_schema_length=None,
    )
    trade = pl.DataFrame(
        {
            "ValueCode": [VALUE_CODE],
            "recv_time_ns": [fill],
            "sequence": [10],
            "packet_sequence": [10],
            "trade_price": [102.0],
            "trade_lots": [1],
        }
    )
    empty_trade = pl.DataFrame(schema=trade.schema)
    tape = RawTapeDay(
        DATE,
        _mapping(),
        spot,
        future,
        empty_trade,
        trade,
        pl.DataFrame({"Date": [DATE], "ValueCode": [VALUE_CODE]}),
    )
    observations: list[dict[str, object]] = []
    for quantile in (50, 80):
        observations.append(
            {
                "Date": DATE,
                "ValueCode": VALUE_CODE,
                "QuoteCode": QUOTE_CODE,
                "route": "future_ask_spot_taker",
                "boundary_quantile": quantile,
                "recv_time_ns": submit,
                "cursor_event_sequence": 0,
                "cursor_row_index": 0,
                "spread_pair_epoch": 1,
                "absolute_target_tick": 2302,
                "gate_open": True,
                "admission_reason": "admitted",
                "initial_queue_ahead": 0,
                "target_rank": "ASK1",
                "event_source": "anchor",
                "boundary_role": "upper",
                "parameter_version": "test-v1",
                "source_asof_date": "20260101",
                "absolute_target_price": 101.0,
                "threshold_basis_bp": 100.0,
                "effective_basis_bp": 100.0,
                "spot_book_age_ms": 0.0,
                "future_book_age_ms": 0.0,
            }
        )
    return MergedTargetStudyInput(
        DATE,
        _mapping(),
        tape,
        pl.from_dicts(observations, infer_schema_length=None),
        pl.DataFrame({"Date": [DATE], "ValueCode": [VALUE_CODE]}),
    )


class ExecutionActionFactTest(unittest.TestCase):
    def test_aliases_share_one_physical_hedge_and_keep_partial_cancel(self) -> None:
        facts = build_execution_action_facts(_alias_frame(), _hedge_frame())
        shared = facts.filter(pl.col("raw_order_fact_id") == "shared-raw")
        self.assertEqual(shared.height, 2)
        self.assertEqual(shared["same_price_policy_aliases"].to_list(), [2, 2])
        self.assertTrue(shared["entry_hedge_executable"].all())
        self.assertEqual(
            shared["full_fill_recv_time_ns"].to_list(), [150, 150]
        )

        partial = facts.filter(
            pl.col("policy_generation_id") == "partial-q50"
        ).row(0, named=True)
        self.assertEqual(partial["entry_execution_outcome"], "partial_fill_then_cancel")
        self.assertEqual(partial["first_fill_recv_time_ns"], 700)
        self.assertTrue(partial["cancel_required"])
        self.assertFalse(partial["pathwise_ev_ready"])

        daily = summarize_execution_daily(facts)
        q50 = daily.filter(pl.col("lookup_action_id") == "q50").row(
            0, named=True
        )
        self.assertEqual(q50["policy_alias_orders"], 2)
        self.assertEqual(q50["unique_raw_order_facts"], 2)
        self.assertEqual(q50["full_fills"], 1)
        self.assertEqual(q50["partial_fills"], 1)
        self.assertEqual(q50["cancel_required_rate"], 0.5)

    def test_raw_identity_and_exact_fill_cursor_are_guarded(self) -> None:
        aliases = _alias_frame().with_columns(
            pl.when(pl.col("policy_generation_id") == "full-q80")
            .then(pl.lit(99))
            .otherwise(pl.col("spread_pair_epoch"))
            .alias("spread_pair_epoch")
        )
        with self.assertRaisesRegex(ValueError, "physical identity"):
            build_execution_action_facts(aliases, _hedge_frame())

        missing_cursor = _alias_frame().with_columns(
            pl.when(pl.col("policy_generation_id") == "full-q50")
            .then(None)
            .otherwise(pl.col("full_fill_event_sequence"))
            .alias("full_fill_event_sequence")
        )
        with self.assertRaisesRegex(ValueError, "three-part cursor"):
            build_execution_action_facts(missing_cursor, _hedge_frame())


class ExecutableExitFactTest(unittest.TestCase):
    def test_first_executable_threshold_hit_and_carry_are_separate(self) -> None:
        path = build_executable_taker_exit_path(
            _raw_tape_for_exit(),
            VALUE_CODE,
            start_time_ns=100,
            cutoff_time_ns=301,
            config=ExecutableExitPathConfig(grid_ns=100, max_book_age_ns=1000),
        )
        self.assertAlmostEqual(path.item(0, "exit_basis_bp"), 202.020202)
        self.assertAlmostEqual(path.item(1, "exit_basis_bp"), 50.0)

        action = build_execution_action_facts(
            _alias_frame().filter(
                pl.col("policy_generation_id") == "full-q50"
            ),
            _hedge_frame(),
        )
        rules = pl.DataFrame(
            {
                "policy_generation_id": ["full-q50", "full-q50"],
                "exit_rule_id": ["close_100bp", "close_0bp"],
                "exit_threshold_basis_bp": [100.0, 0.0],
                "source_asof_date": ["20260101", "20260101"],
                "contains_target_day_outcome": [False, False],
            }
        )
        facts = build_executable_exit_facts(action, rules, path)
        hit = facts.filter(pl.col("exit_rule_id") == "close_100bp").row(
            0, named=True
        )
        self.assertEqual(hit["branch_status"], "same_day_taker_exit")
        self.assertEqual(hit["exit_decision_time_ns"], 200)
        self.assertEqual(hit["gross_cycle_pnl_twd"], 3000.0)
        carry = facts.filter(pl.col("exit_rule_id") == "close_0bp").row(
            0, named=True
        )
        self.assertEqual(carry["branch_status"], "carry_at_eod")
        self.assertTrue(carry["eod_mark_available"])
        self.assertFalse(carry["terminal_outcome"])
        self.assertTrue(carry["needs_next_session_label"])
        self.assertIsNone(carry["net_cycle_pnl_twd"])
        self.assertFalse(carry["pathwise_ev_ready"])

        daily = summarize_executable_exit_daily(action, facts)
        self.assertEqual(daily.height, 2)
        self.assertEqual(daily["same_day_taker_exits"].sum(), 1)
        self.assertEqual(daily["overnight_carry_branches"].sum(), 1)

    def test_exit_rule_must_be_strictly_prior_and_target_day_safe(self) -> None:
        path = build_executable_taker_exit_path(
            _raw_tape_for_exit(),
            VALUE_CODE,
            start_time_ns=100,
            cutoff_time_ns=301,
            config=ExecutableExitPathConfig(grid_ns=100, max_book_age_ns=1000),
        )
        action = build_execution_action_facts(
            _alias_frame().head(1), _hedge_frame()
        )
        unsafe = pl.DataFrame(
            {
                "policy_generation_id": ["full-q50"],
                "exit_rule_id": ["bad"],
                "exit_threshold_basis_bp": [100.0],
                "source_asof_date": [DATE],
                "contains_target_day_outcome": [True],
            }
        )
        with self.assertRaisesRegex(ValueError, "target-day"):
            build_executable_exit_facts(action, unsafe, path)

        safe = unsafe.with_columns(
            pl.lit("20260101").alias("source_asof_date"),
            pl.lit(False).alias("contains_target_day_outcome"),
        )
        mismatched_path = path.with_columns(
            pl.lit("OTHER").alias("QuoteCode")
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            build_executable_exit_facts(action, safe, mismatched_path)

    def test_trial_match_requires_a_new_formal_book(self) -> None:
        rows = [
            _state("spot", 100, 1, bid=99.0, ask=100.0),
            _state(
                "spot",
                150,
                2,
                bid=99.0,
                ask=100.0,
                trial_match=True,
                raw_has_book=False,
            ),
            _state(
                "spot",
                200,
                3,
                bid=99.0,
                ask=100.0,
                raw_has_book=False,
            ),
            _state("spot", 250, 4, bid=100.0, ask=101.0),
        ]
        future = [
            _state("future", timestamp, index, bid=100.0, ask=100.5)
            for index, timestamp in enumerate((100, 200, 250), 1)
        ]
        empty = pl.DataFrame()
        tape = RawTapeDay(
            DATE,
            _mapping(),
            pl.from_dicts(rows, infer_schema_length=None),
            pl.from_dicts(future, infer_schema_length=None),
            empty,
            empty,
            pl.DataFrame(),
        )
        path = build_executable_taker_exit_path(
            tape,
            VALUE_CODE,
            start_time_ns=100,
            cutoff_time_ns=301,
            config=ExecutableExitPathConfig(grid_ns=100, max_book_age_ns=1000),
        )
        self.assertEqual(path.item(1, "status"), "spot_awaiting_formal_book")
        self.assertEqual(path.item(2, "status"), "executable")


class ExecutionRunnerTest(unittest.TestCase):
    def test_legacy_unmarked_boundary_fails_before_raw_io(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            daily_root, boundary_path = _write_walkforward_source_fixture(root)
            (boundary_path.parent / "complete.json").unlink()
            module = "maker.src.quote_fill.execution_runner"
            with (
                patch(
                    f"{module}.validate_completion_marker",
                    return_value=_validated_daily_marker_payload(),
                ),
                patch(f"{module}.load_raw_tape_day") as raw_loader,
            ):
                with self.assertRaisesRegex(
                    FileNotFoundError,
                    "publication is incomplete",
                ):
                    load_walkforward_execution_product_day(
                        DATE,
                        VALUE_CODE,
                        daily_root=daily_root,
                        boundary_snapshot_path=boundary_path,
                        data_root=root / "raw-root",
                    )
            raw_loader.assert_not_called()

    def test_walkforward_path_loader_is_safe_and_uses_exact_raw_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            daily_root, boundary_path = _write_walkforward_source_fixture(
                Path(directory)
            )
            feature = pl.DataFrame(
                {"Date": [DATE], "ValueCode": [VALUE_CODE]}
            )
            merged_audit = pl.DataFrame(
                {"Date": [DATE], "ValueCode": [VALUE_CODE]}
            )
            module = "maker.src.quote_fill.execution_runner"
            with (
                patch(
                    f"{module}.validate_completion_marker",
                    return_value=_validated_daily_marker_payload(),
                ),
                patch(
                    f"{module}.load_raw_tape_day",
                    return_value=_raw_tape_for_exit(),
                ) as raw_loader,
                patch(
                    f"{module}.load_spot_feature_state",
                    return_value=feature,
                ),
                patch(
                    f"{module}.build_merged_target_observations_from_frames",
                    return_value=(pl.DataFrame(), merged_audit),
                ) as merged_builder,
            ):
                merged = load_walkforward_execution_product_day(
                    DATE,
                    VALUE_CODE,
                    daily_root=daily_root,
                    boundary_snapshot_path=boundary_path,
                    data_root=Path(directory) / "raw-root",
                )

            self.assertEqual(merged.mapping["QuoteCode"].to_list(), [QUOTE_CODE])
            self.assertNotIn("Date", merged.mapping.columns)
            self.assertNotIn("trading_turnover", merged.mapping.columns)
            raw_mapping = raw_loader.call_args.args[1]
            self.assertNotIn("Date", raw_mapping.columns)
            self.assertNotIn("trading_turnover", raw_mapping.columns)
            fair = merged_builder.call_args.args[4]
            boundaries = merged_builder.call_args.args[5]
            self.assertEqual(fair.columns, [
                "Date",
                "ValueCode",
                "QuoteCode",
                "spot_ref_price",
                "fut_ref_price",
                "contract_size",
                "anchor_ewma_120s_bp",
                "fair_timestamp",
            ])
            self.assertNotIn("future_excursion_bp", fair.columns)
            self.assertEqual(
                boundaries["boundary_quantile"].to_list(), [50, 80, 95]
            )
            audit = merged.audit.row(0, named=True)
            self.assertTrue(audit["execution_safe_source_validated"])
            self.assertTrue(audit["daily_source_migrated_nonatomic"])
            self.assertEqual(audit["boundary_source_asof_date"], "20260101")

    def test_walkforward_sources_fail_closed_on_asof_and_contract(self) -> None:
        module = "maker.src.quote_fill.execution_runner"
        marker = _validated_daily_marker_payload()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            daily_root, boundary_path = _write_walkforward_source_fixture(
                root, source_asof_date=DATE
            )
            with patch(
                f"{module}.validate_completion_marker", return_value=marker
            ):
                with self.assertRaisesRegex(ValueError, "strictly before"):
                    load_walkforward_product_day_sources(
                        DATE,
                        VALUE_CODE,
                        daily_root=daily_root,
                        boundary_snapshot_path=boundary_path,
                    )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            daily_root, boundary_path = _write_walkforward_source_fixture(
                root, boundary_quote_code="WRONG"
            )
            with patch(
                f"{module}.validate_completion_marker", return_value=marker
            ):
                with self.assertRaisesRegex(ValueError, "must be exactly"):
                    load_walkforward_product_day_sources(
                        DATE,
                        VALUE_CODE,
                        daily_root=daily_root,
                        boundary_snapshot_path=boundary_path,
                    )

    def test_product_day_reuses_raw_fact_and_emits_cancel_denominators(self) -> None:
        result = replay_execution_product_day(
            _merged_product_day(),
            ExecutionRunnerConfig(boundary_quantiles=(50, 80)),
        )
        self.assertEqual(result.order_aliases.height, 2)
        self.assertEqual(result.raw_order_facts.height, 1)
        self.assertEqual(result.hedge_facts.height, 1)
        self.assertEqual(
            result.order_aliases["raw_order_fact_id"].n_unique(), 1
        )
        self.assertTrue(result.order_aliases["full_fill"].all())
        self.assertFalse(result.order_aliases["cancel_required"].any())
        future_rows = result.execution_daily_facts.filter(
            pl.col("route") == "future_ask_spot_taker"
        )
        self.assertEqual(future_rows.height, 2)
        self.assertEqual(future_rows["policy_alias_orders"].to_list(), [1, 1])

    def test_atomic_partition_resume_does_not_reload_raw_tape(self) -> None:
        calls: list[tuple[str, str]] = []

        def loader(date: str, value_code: str) -> MergedTargetStudyInput:
            calls.append((date, value_code))
            return _merged_product_day()

        config = ExecutionRunnerConfig(boundary_quantiles=(50, 80))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = run_partitioned_execution_replay(
                [(DATE, VALUE_CODE)], loader, root, config
            )
            self.assertEqual(manifest.height, 1)
            self.assertEqual(calls, [(DATE, VALUE_CODE)])
            partition = root / f"Date={DATE}" / f"ValueCode={VALUE_CODE}"
            marker = json.loads(
                (partition / "complete.json").read_text(encoding="utf-8")
            )
            self.assertTrue(marker["complete"])
            self.assertEqual(
                marker["fact_semantics"]["spread_pair_clock"],
                "SpreadPairTotalCount",
            )
            self.assertIn(
                "spot_partial_incremental_quantity_path",
                marker["missing_for_net_ev"],
            )
            self.assertTrue(
                (partition / "execution_action_facts.parquet").exists()
            )

            resumed = run_partitioned_execution_replay(
                [(DATE, VALUE_CODE)], loader, root, config
            )
            self.assertEqual(resumed.height, 1)
            self.assertEqual(calls, [(DATE, VALUE_CODE)])


if __name__ == "__main__":
    unittest.main()
