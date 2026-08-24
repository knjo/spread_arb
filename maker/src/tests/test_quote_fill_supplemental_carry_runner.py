from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import polars as pl

from maker.src.quote_fill.exit_maker_cross_session import (
    CrossSessionExitMakerSession,
)
from maker.src.quote_fill.raw_tape import RawTapeDay
from maker.src.quote_fill.supplemental_carry_runner import (
    _extract_unresolved_execution_facts,
    replay_supplemental_continuations,
    thin_candidate_session_one_second,
)


D1 = "20260601"
D2 = "20260602"
D3 = "20260603"
VALUE = "A"
QUOTE = "QAF6"
SHA = "a" * 64


def _path(
    identifier: str,
    *,
    status: str = "maker_fill_state_unknown",
    last_observed: str = D1,
) -> dict[str, object]:
    return {
        "Date": D1,
        "ValueCode": VALUE,
        "QuoteCode": QUOTE,
        "policy_path_id": f"path-{identifier}",
        "entry_policy_generation_id": f"entry-{identifier}",
        "exit_policy_trial_id": f"trial-{identifier}",
        "outcome_status": status,
        "last_observed_session_date": last_observed,
    }


def _market_state(
    date: str,
    time_ns: int,
    *,
    bid: float,
    ask: float,
) -> dict[str, object]:
    return {
        "Date": date,
        "ValueCode": VALUE,
        "QuoteCode": QUOTE,
        "recv_time_ns": time_ns,
        "sequence": time_ns,
        "packet_sequence": time_ns,
        "ref_price": 100.0,
        "raw_has_book": True,
        "book_state_available": True,
        "exec_bid_price": bid,
        "exec_ask_price": ask,
    }


def _session(date: str) -> CrossSessionExitMakerSession:
    tape = RawTapeDay(
        date=date,
        mapping=pl.DataFrame(
            {
                "ValueCode": [VALUE],
                "QuoteCode": [QUOTE],
                "spot_ref_price": [100.0],
                "fut_ref_price": [100.0],
            }
        ),
        spot_states=pl.from_dicts(
            [
                _market_state(date, 100, bid=99.0, ask=100.0),
                _market_state(date, 300, bid=101.0, ask=102.0),
            ],
            infer_schema_length=None,
        ),
        future_states=pl.from_dicts(
            [
                _market_state(date, 110, bid=102.0, ask=103.0),
                _market_state(date, 250, bid=103.0, ask=104.0),
            ],
            infer_schema_length=None,
        ),
        spot_trades=pl.DataFrame(),
        future_trades=pl.DataFrame(),
        audit=pl.DataFrame(),
    )
    return CrossSessionExitMakerSession(
        date=date,
        raw_tape=tape,
        spread_pair_clock=pl.DataFrame(),
        spot_ref_price=100.0,
        future_ref_price=100.0,
        ref_price_source_date=date,
        ref_price_source_version="synthetic-ref-v1",
        session_start_time_ns=99,
    )


def _policy_row(
    trial: str,
    branch: str,
    *,
    spot: float | None = None,
    future: float | None = None,
) -> dict[str, object]:
    terminal = branch == "flat_same_day"
    return {
        "exit_policy_trial_id": trial,
        "branch_status": branch,
        "nominal_instant_cancel_v0_branch": branch,
        "terminal_outcome": terminal,
        "needs_next_session_label": branch.startswith("carry_"),
        "exit_hedge_status": "executable" if terminal else None,
        "exit_spot_price": spot,
        "exit_future_price": future,
        "exit_decision_time_ns": 200 if terminal else None,
        "gross_cycle_pnl_twd": 1.0 if terminal else None,
    }


def _replay_source(ids: list[str]) -> tuple[pl.DataFrame, pl.DataFrame]:
    actions = pl.DataFrame(
        {
            "policy_generation_id": [f"entry-{item}" for item in ids],
            "Date": [D1] * len(ids),
            "entry_hedge_decision_time_ns": [50] * len(ids),
        }
    )
    exits = pl.DataFrame(
        {
            "policy_generation_id": [f"entry-{item}" for item in ids],
            "Date": [D1] * len(ids),
        }
    )
    return actions, exits


class SupplementalCarryRunnerTest(unittest.TestCase):
    def test_one_second_mode_retains_spread_epochs_and_last_valid_marks(
        self,
    ) -> None:
        base = _session(D1)
        spot_rows = [
            _market_state(D1, 100, bid=99.0, ask=100.0),
            _market_state(D1, 200, bid=100.0, ask=101.0),
            _market_state(D1, 300, bid=101.0, ask=102.0),
            _market_state(D1, 900, bid=999.0, ask=1_000.0),
        ]
        future_rows = [
            _market_state(D1, 110, bid=102.0, ask=103.0),
            _market_state(D1, 210, bid=103.0, ask=104.0),
            _market_state(D1, 910, bid=998.0, ask=999.0),
        ]
        for sequence, row in enumerate(spot_rows, start=1):
            row["sequence"] = sequence
            row["packet_sequence"] = sequence
        for sequence, row in enumerate(future_rows, start=1):
            row["sequence"] = sequence
            row["packet_sequence"] = sequence
        spot_rows[-1]["raw_has_book"] = False
        future_rows[-1]["raw_has_book"] = False
        raw = replace(
            base.raw_tape,
            spot_states=pl.from_dicts(spot_rows, infer_schema_length=None),
            future_states=pl.from_dicts(future_rows, infer_schema_length=None),
        )
        session = replace(
            base,
            raw_tape=raw,
            spread_pair_clock=pl.DataFrame(
                {
                    "Date": [D1] * 4,
                    "ValueCode": [VALUE] * 4,
                    "spot_channel_seq": [1, 2, 3, 4],
                    "spread_pair_epoch": [1, 1, 2, 2],
                    "spread_pair_id": [1, 1, 2, 2],
                }
            ),
        )
        thinned, audit = thin_candidate_session_one_second(
            session, value_code=VALUE, quote_code=QUOTE
        )
        self.assertEqual(thinned.raw_tape.spot_states["sequence"].to_list(), [1, 3, 4])
        self.assertEqual(thinned.spread_pair_clock["spot_channel_seq"].to_list(), [1, 3, 4])
        self.assertEqual(thinned.raw_tape.future_states["sequence"].to_list(), [2, 3])
        self.assertTrue(audit["state_sampling_approximate"])
        self.assertTrue(audit["raw_trade_events_unchanged"])
        self.assertFalse(audit["target_cancel_clock_exact"])
        self.assertFalse(audit["delayed_hedge_snapshot_exact"])

    def test_product_session_is_loaded_and_replayed_once_for_all_open_paths(
        self,
    ) -> None:
        paths = pl.from_dicts(
            [
                _path("terminal"),
                _path("unknown"),
                _path("carry"),
                _path(
                    "expiry",
                    status="right_censored_expiry_settlement_unpriced",
                    last_observed=D3,
                ),
            ],
            infer_schema_length=None,
        )
        actions, exits = _replay_source(
            ["terminal", "unknown", "carry", "expiry"]
        )
        load_calls: list[tuple[str, str, str]] = []
        replay_calls: list[tuple[str, tuple[str, ...], int]] = []

        def loader(
            date: str, value: str, quote: str
        ) -> tuple[CrossSessionExitMakerSession, str]:
            load_calls.append((date, value, quote))
            return _session(date), SHA

        def replayer(
            day_actions: pl.DataFrame,
            day_exits: pl.DataFrame,
            session: CrossSessionExitMakerSession,
            trials: tuple[str, ...] | list[str],
            spool_root: Path,
        ) -> pl.DataFrame:
            del day_exits, spool_root
            replay_calls.append(
                (session.date, tuple(trials), day_actions.height)
            )
            if session.date == D2:
                rows = {
                    "trial-terminal": _policy_row(
                        "trial-terminal", "flat_same_day", spot=105.0, future=101.0
                    ),
                    "trial-unknown": _policy_row(
                        "trial-unknown", "maker_fill_unknown_at_cutoff"
                    ),
                    "trial-carry": _policy_row(
                        "trial-carry", "carry_at_eod_no_admission"
                    ),
                }
            else:
                rows = {
                    "trial-unknown": _policy_row(
                        "trial-unknown", "carry_at_eod_no_admission"
                    ),
                    "trial-carry": _policy_row(
                        "trial-carry", "flat_same_day", spot=106.0, future=100.0
                    ),
                }
            return pl.from_dicts(
                [rows[trial] for trial in trials], infer_schema_length=None
            )

        with tempfile.TemporaryDirectory() as directory:
            result = replay_supplemental_continuations(
                paths,
                actions,
                exits,
                (D1, D2, D3),
                pl.DataFrame(
                    {
                        "QuoteCode": [QUOTE],
                        "expiry_session": [D3],
                        "calendar_version": ["synthetic-calendar-v1"],
                    }
                ),
                session_loader=loader,
                policy_day_replayer=replayer,
                spool_root=Path(directory),
            )

        self.assertEqual(result.pair_sessions_replayed, 2)
        self.assertEqual(
            load_calls,
            [(D2, VALUE, QUOTE), (D3, VALUE, QUOTE)],
        )
        self.assertEqual(len(replay_calls), 2)
        self.assertEqual(replay_calls[0][2], 3)
        self.assertEqual(replay_calls[1][2], 2)
        self.assertEqual(result.continuation_terminals.height, 2)
        terminals = {
            row["policy_path_id"]: row
            for row in result.continuation_terminals.iter_rows(named=True)
        }
        self.assertEqual(terminals["path-terminal"]["terminal_date"], D2)
        self.assertEqual(terminals["path-carry"]["terminal_date"], D3)
        self.assertEqual(result.expiry_marks.height, 1)
        mark = result.expiry_marks.row(0, named=True)
        self.assertEqual(mark["spot_close_price"], 101.0)
        self.assertEqual(mark["future_close_price"], 104.0)
        self.assertFalse(mark["mark_is_official_settlement"])
        audit = {
            row["policy_path_id"]: row
            for row in result.continuation_audit.iter_rows(named=True)
        }
        self.assertEqual(
            audit["path-unknown"]["terminal_resolution"],
            "expiry_last_valid_session_mark",
        )
        self.assertEqual(audit["path-unknown"]["imputed_unknown_sessions"], 2)
        self.assertEqual(
            audit["path-expiry"]["terminal_resolution"],
            "expiry_last_valid_session_mark",
        )

    def test_unknown_first_observed_on_expiry_is_mark_only_not_replayed(
        self,
    ) -> None:
        paths = pl.from_dicts(
            [_path("unknown-expiry", last_observed=D3)],
            infer_schema_length=None,
        )
        actions, exits = _replay_source(["unknown-expiry"])
        load_calls: list[str] = []

        def loader(
            date: str, value: str, quote: str
        ) -> tuple[CrossSessionExitMakerSession, str]:
            del value, quote
            load_calls.append(date)
            return _session(date), SHA

        def must_not_replay(*args: object) -> pl.DataFrame:
            raise AssertionError("expiry unknown must not replay the source day")

        with tempfile.TemporaryDirectory() as directory:
            result = replay_supplemental_continuations(
                paths,
                actions,
                exits,
                (D1, D2, D3),
                pl.DataFrame(
                    {
                        "QuoteCode": [QUOTE],
                        "expiry_session": [D3],
                        "calendar_version": ["synthetic-calendar-v1"],
                    }
                ),
                session_loader=loader,
                policy_day_replayer=must_not_replay,
                spool_root=Path(directory),
            )
        self.assertEqual(load_calls, [D3])
        self.assertEqual(result.continuation_terminals.height, 0)
        self.assertEqual(result.expiry_marks.height, 1)
        audit = result.continuation_audit.row(0, named=True)
        self.assertEqual(
            audit["terminal_resolution"], "expiry_last_valid_session_mark"
        )
        self.assertEqual(audit["sessions_replayed"], 0)

    def test_extracts_exact_entry_prices_and_two_frozen_rules_per_entry(
        self,
    ) -> None:
        unresolved = pl.DataFrame(
            {
                "Date": [D1, D1],
                "ValueCode": [VALUE, VALUE],
                "entry_policy_generation_id": ["entry-a", "entry-b"],
                "normalization_notional_twd": [200.0, 600.0],
            }
        )
        action_rows = [
            _action_fact("entry-a", spot=100.0, future=102.0, shares=2),
            _action_fact("entry-b", spot=200.0, future=203.0, shares=3),
            _action_fact("entry-unused", spot=50.0, future=51.0, shares=1),
        ]
        exit_rows = [
            _exit_fact(entry, rule)
            for entry in ("entry-a", "entry-b", "entry-unused")
            for rule in ("frozen_center", "frozen_lower")
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partition = root / "partition"
            partition.mkdir()
            action_path = partition / "execution_action_facts.parquet"
            exit_path = partition / "exit_facts.parquet"
            pl.from_dicts(action_rows, infer_schema_length=None).write_parquet(
                action_path
            )
            pl.from_dicts(exit_rows, infer_schema_length=None).write_parquet(
                exit_path
            )
            marker = {
                "complete": True,
                "Date": D1,
                "ValueCode": VALUE,
                "artifacts": {
                    action_path.name: _declaration(action_path, len(action_rows)),
                    exit_path.name: _declaration(exit_path, len(exit_rows)),
                },
            }
            (partition / "complete.json").write_text(
                json.dumps(marker), encoding="utf-8"
            )
            manifest_path = root / "execution_partition_manifest.parquet"
            pl.DataFrame(
                {
                    "Date": [D1],
                    "ValueCode": [VALUE],
                    "partition": [str(partition)],
                    "complete": [True],
                }
            ).write_parquet(manifest_path)
            actions, exits, prices, inventory = (
                _extract_unresolved_execution_facts(
                    unresolved,
                    execution_manifest_path=manifest_path,
                )
            )
        self.assertEqual(actions.height, 2)
        self.assertEqual(exits.height, 4)
        self.assertEqual(prices.height, 2)
        self.assertEqual(inventory.height, 2)
        by_id = {
            row["entry_policy_generation_id"]: row
            for row in prices.iter_rows(named=True)
        }
        self.assertEqual(by_id["entry-a"]["entry_spot_price"], 100.0)
        self.assertEqual(by_id["entry-b"]["entry_contract_size_shares"], 3)
        self.assertEqual(
            by_id["entry-a"]["entry_price_source"],
            "execution_action_facts.parquet",
        )


def _action_fact(
    entry_id: str,
    *,
    spot: float,
    future: float,
    shares: int,
) -> dict[str, object]:
    return {
        "Date": D1,
        "ValueCode": VALUE,
        "QuoteCode": QUOTE,
        "route": "spot_bid_future_taker",
        "raw_order_fact_id": f"raw-{entry_id}",
        "policy_generation_id": entry_id,
        "full_fill": True,
        "entry_hedge_status": "executable",
        "entry_hedge_decision_time_ns": 50,
        "entry_hedge_label_observed": True,
        "entry_hedge_executable": True,
        "entry_future_price": future,
        "entry_spot_price": spot,
        "entry_hedge_contract_size_shares": shares,
    }


def _exit_fact(entry_id: str, rule: str) -> dict[str, object]:
    return {
        "Date": D1,
        "ValueCode": VALUE,
        "QuoteCode": QUOTE,
        "route": "spot_bid_future_taker",
        "raw_order_fact_id": f"raw-{entry_id}",
        "policy_generation_id": entry_id,
        "exit_rule_id": rule,
        "exit_threshold_basis_bp": 5.0,
        "exit_rule_source_asof_date": D1,
    }


def _declaration(path: Path, rows: int) -> dict[str, object]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "sha256": digest,
        "bytes": path.stat().st_size,
        "rows": rows,
        "columns": len(pl.read_parquet_schema(path)),
    }


if __name__ == "__main__":
    unittest.main()
