from __future__ import annotations

import unittest

import polars as pl

from maker.src.quote_fill.combined_cost_cap_sweep import (
    CombinedCapConfig,
    build_combined_cost_cap_sweep,
)
from maker.src.quote_fill.raw_tape import RawTapeDay
from maker.src.quote_fill.supplemental_carry_terminal import (
    apply_supplemental_carry_terminal_overlay,
    build_last_observed_session_liquidation_mark,
    build_last_valid_session_bbo_mark,
)


D1 = "20260616"
D2 = "20260617"
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _path(
    identifier: str,
    *,
    date: str = D1,
    value_code: str = "A",
    quote_code: str = "QAF6",
    established_ns: int = 100,
    notional: float = 100.0,
    category: str = "unknown",
    outcome_status: str = "maker_fill_state_unknown",
) -> dict[str, object]:
    return {
        "Date": date,
        "ValueCode": value_code,
        "QuoteCode": quote_code,
        "entry_route": "spot_bid_future_taker",
        "boundary_quantile": 95,
        "entry_policy_generation_id": f"entry-{identifier}",
        "entry_raw_order_fact_id": f"raw-{identifier}",
        "exit_policy_trial_id": f"exit-{identifier}",
        "position_established_ns": established_ns,
        "filled_entry_outcome_category": category,
        "outcome_status": outcome_status,
        "terminal_date": None,
        "exit_decision_time_ns": None,
        "gross_cycle_pnl_twd": None,
        "gross_cycle_bp": None,
        "normalization_notional_twd": notional,
        "physical_entry_dependency_id": f"physical-{identifier}",
        "policy_path_id": f"path-{identifier}",
        "completed_same_day": False,
        "completed_overnight": False,
        "terminal_cashflow_priced": False,
        "nominal_cancel_model_assumption": True,
        "pathwise_ev_ready": False,
        "joint_volume_allocated": False,
        "unresolved_cashflow_imputed": False,
        "diagnostic_challenger_is_selected_action": False,
        "production_strategy_go": False,
        "entry_spot_price": None,
        "entry_future_price": None,
        "entry_contract_size_shares": None,
        "exit_spot_price": None,
        "exit_future_price": None,
        "exact_price_source": None,
    }


def _paths(*rows: dict[str, object]) -> pl.DataFrame:
    return pl.from_dicts(
        rows,
        infer_schema_length=None,
        schema_overrides={
            "position_established_ns": pl.Int64,
            "exit_decision_time_ns": pl.Int64,
            "gross_cycle_pnl_twd": pl.Float64,
            "gross_cycle_bp": pl.Float64,
            "normalization_notional_twd": pl.Float64,
            "entry_spot_price": pl.Float64,
            "entry_future_price": pl.Float64,
            "entry_contract_size_shares": pl.Int64,
            "exit_spot_price": pl.Float64,
            "exit_future_price": pl.Float64,
        },
    )


def _entry(
    identifier: str,
    *,
    spot: float,
    future: float,
    shares: int = 1,
) -> dict[str, object]:
    return {
        "entry_policy_generation_id": f"entry-{identifier}",
        "entry_spot_price": spot,
        "entry_future_price": future,
        "entry_contract_size_shares": shares,
        "entry_price_source": "synthetic_source_bound_entry",
        "source_identity_sha256": SHA_A,
    }


def _continuation(
    identifier: str,
    *,
    date: str = D2,
    time_ns: int = 200,
    spot: float,
    future: float,
) -> dict[str, object]:
    return {
        "policy_path_id": f"path-{identifier}",
        "terminal_date": date,
        "exit_decision_time_ns": time_ns,
        "exit_spot_price": spot,
        "exit_future_price": future,
        "exact_price_source": "extended_normal_exit_replay",
        "source_identity_sha256": SHA_B,
    }


def _expiry_mark(
    *,
    value_code: str = "A",
    quote_code: str = "QAF6",
    spot: float = 90.0,
    future: float = 100.0,
    official_close: bool = False,
    official_settlement: bool = False,
) -> dict[str, object]:
    return {
        "Date": D2,
        "expiry_session": D2,
        "calendar_version": "synthetic-official-calendar-v1",
        "ValueCode": value_code,
        "QuoteCode": quote_code,
        "spot_close_price": spot,
        "future_close_price": future,
        "spot_close_time_ns": 300,
        "future_close_time_ns": 250,
        "spot_close_source": "last_valid_session_spot_bid",
        "future_close_source": "last_valid_session_future_ask",
        "source_identity_sha256": SHA_C,
        "mark_is_official_close": official_close,
        "mark_is_official_settlement": official_settlement,
    }


class SupplementalCarryTerminalTest(unittest.TestCase):
    def test_unknown_full_pair_continues_to_normal_replay_terminal(self) -> None:
        source = _paths(
            _path("resolved", notional=100.0),
            _path(
                "still-unresolved",
                value_code="B",
                quote_code="QBF6",
                established_ns=101,
                notional=80.0,
            ),
        )
        overlay = apply_supplemental_carry_terminal_overlay(
            source,
            pl.from_dicts(
                [_entry("resolved", spot=100.0, future=105.0)],
                infer_schema_length=None,
            ),
            pl.from_dicts(
                [_continuation("resolved", spot=110.0, future=103.0)],
                infer_schema_length=None,
            ),
            None,
        )
        resolved = overlay.filter(
            pl.col("policy_path_id") == "path-resolved"
        ).row(0, named=True)
        self.assertEqual(resolved["filled_entry_outcome_category"], "completed")
        self.assertEqual(resolved["terminal_date"], D2)
        self.assertTrue(resolved["terminal_cashflow_priced"])
        self.assertEqual(resolved["gross_cycle_pnl_twd"], 12.0)
        self.assertEqual(resolved["gross_cycle_bp"], 1_200.0)
        self.assertEqual(
            resolved["supplemental_terminal_resolution"],
            "normal_continuation_replay",
        )
        self.assertTrue(resolved["model_imputed_full_carry_on_unknown"])
        self.assertTrue(resolved["double_exit_bias_possible"])
        self.assertFalse(resolved["unresolved_cashflow_imputed"])
        self.assertFalse(
            resolved["production_strategy_go_after_supplemental_overlay"]
        )

        unresolved = overlay.filter(
            pl.col("policy_path_id") == "path-still-unresolved"
        ).row(0, named=True)
        self.assertFalse(unresolved["terminal_cashflow_priced"])
        self.assertIsNone(unresolved["gross_cycle_pnl_twd"])
        self.assertIsNone(unresolved["entry_spot_price"])
        self.assertEqual(
            unresolved["supplemental_terminal_resolution"],
            "imputed_full_carry_still_unresolved",
        )
        self.assertTrue(unresolved["double_exit_bias_possible"])

    def test_expiry_last_valid_mark_forces_nonnull_terminal_not_settlement(
        self,
    ) -> None:
        source = _paths(
            _path(
                "expiry",
                category="censored",
                outcome_status="right_censored_expiry_settlement_unpriced",
                notional=100.0,
            )
        )
        overlay = apply_supplemental_carry_terminal_overlay(
            source,
            pl.from_dicts(
                [_entry("expiry", spot=100.0, future=105.0)],
                infer_schema_length=None,
            ),
            None,
            pl.from_dicts([_expiry_mark()], infer_schema_length=None),
        )
        row = overlay.row(0, named=True)
        self.assertEqual(row["filled_entry_outcome_category"], "completed")
        self.assertEqual(row["terminal_date"], D2)
        self.assertEqual(row["exit_decision_time_ns"], 300)
        self.assertEqual(row["gross_cycle_pnl_twd"], -5.0)
        self.assertEqual(row["gross_cycle_bp"], -500.0)
        self.assertTrue(row["terminal_cashflow_priced"])
        self.assertTrue(row["expiry_uses_last_valid_session_mark"])
        self.assertFalse(row["expiry_mark_is_official_close"])
        self.assertFalse(row["expiry_mark_is_official_settlement"])
        self.assertFalse(row["model_imputed_full_carry_on_unknown"])
        self.assertIn("not_settlement", row["terminal_reason"])
        self.assertIn("last_valid_session", row["exact_price_source"])

    def test_imputed_terminal_releases_capacity_before_later_entry(self) -> None:
        source = _paths(
            _path(
                "carried",
                value_code="A",
                quote_code="QAF6",
                date=D1,
                established_ns=100,
                notional=60.0,
            ),
            _path(
                "next-entry",
                value_code="B",
                quote_code="QBF6",
                date=D2,
                established_ns=101,
                notional=60.0,
            ),
        )
        overlay = apply_supplemental_carry_terminal_overlay(
            source,
            pl.from_dicts(
                [_entry("carried", spot=60.0, future=65.0)],
                infer_schema_length=None,
            ),
            pl.from_dicts(
                [
                    _continuation(
                        "carried", time_ns=100, spot=61.0, future=64.0
                    )
                ],
                infer_schema_length=None,
            ),
            None,
        )
        result = build_combined_cost_cap_sweep(
            overlay,
            config=CombinedCapConfig(
                portfolio_caps_twd=(100.0,), per_product_fraction=1.0
            ),
        )
        events = {
            row["policy_path_id"]: row
            for row in result.cap_events.iter_rows(named=True)
        }
        self.assertTrue(events["path-carried"]["accepted"])
        self.assertTrue(events["path-next-entry"]["accepted"])
        self.assertEqual(
            events["path-next-entry"][
                "released_completed_positions_before_entry"
            ],
            1,
        )
        self.assertEqual(
            events["path-next-entry"][
                "released_completed_notional_before_entry_twd"
            ],
            60.0,
        )
        summary = result.cap_summary.row(0, named=True)
        self.assertEqual(summary["accepted_positions"], 2)
        self.assertEqual(summary["accepted_completed_cycles"], 1)
        self.assertEqual(summary["accepted_unknown_or_open_positions"], 1)
        self.assertEqual(
            summary["peak_outstanding_one_way_entry_notional_twd"], 60.0
        )

    def test_last_valid_bbo_mark_is_source_bound_and_not_official(self) -> None:
        spot = pl.from_dicts(
            [
                _state(D2, "A", "QAF6", 100, 1, bid=99.0, ask=100.0),
                _state(D2, "A", "QAF6", 200, 2, bid=101.0, ask=102.0),
                _state(
                    D2,
                    "A",
                    "QAF6",
                    300,
                    3,
                    bid=500.0,
                    ask=501.0,
                    raw_has_book=False,
                ),
            ],
            infer_schema_length=None,
        )
        future = pl.from_dicts(
            [
                _state(D2, "A", "QAF6", 150, 1, bid=101.0, ask=102.0),
                _state(D2, "A", "QAF6", 250, 2, bid=102.0, ask=103.0),
            ],
            infer_schema_length=None,
        )
        tape = RawTapeDay(
            date=D2,
            mapping=pl.DataFrame({"ValueCode": ["A"], "QuoteCode": ["QAF6"]}),
            spot_states=spot,
            future_states=future,
            spot_trades=pl.DataFrame(),
            future_trades=pl.DataFrame(),
            audit=pl.DataFrame(),
        )
        mark = build_last_valid_session_bbo_mark(
            tape,
            value_code="A",
            quote_code="QAF6",
            expiry_session=D2,
            calendar_version="synthetic-official-calendar-v1",
            source_identity_sha256=SHA_C,
        ).row(0, named=True)
        self.assertEqual(mark["spot_close_price"], 101.0)
        self.assertEqual(mark["future_close_price"], 103.0)
        self.assertEqual(mark["spot_close_time_ns"], 200)
        self.assertEqual(mark["future_close_time_ns"], 250)
        self.assertFalse(mark["mark_is_official_close"])
        self.assertFalse(mark["mark_is_official_settlement"])
        self.assertIn("not_official_close_or_settlement", mark["mark_role"])
        self.assertEqual(mark["source_identity_sha256"], SHA_C)

    def test_incomplete_or_mislabeled_sources_fail_closed(self) -> None:
        source = _paths(
            _path(
                "expiry",
                category="censored",
                outcome_status="right_censored_expiry_settlement_unpriced",
            )
        )
        entries = pl.from_dicts(
            [_entry("expiry", spot=100.0, future=105.0)],
            infer_schema_length=None,
        )
        bad_mark = _expiry_mark(official_settlement=True)
        with self.assertRaisesRegex(ValueError, "not official settlement"):
            apply_supplemental_carry_terminal_overlay(
                source,
                entries,
                None,
                pl.from_dicts([bad_mark], infer_schema_length=None),
            )

        bad_mark = _expiry_mark()
        bad_mark["future_close_price"] = None
        with self.assertRaisesRegex(ValueError, "invalid future_close_price"):
            apply_supplemental_carry_terminal_overlay(
                source,
                entries,
                None,
                pl.from_dicts([bad_mark], infer_schema_length=None),
            )

        bad_entry = _entry("expiry", spot=100.0, future=105.0)
        bad_entry["source_identity_sha256"] = "not-a-hash"
        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            apply_supplemental_carry_terminal_overlay(
                source,
                pl.from_dicts([bad_entry], infer_schema_length=None),
                None,
                pl.from_dicts([_expiry_mark()], infer_schema_length=None),
            )

    def test_liquidation_mark_names_spot_trade_fallback(self) -> None:
        tape = RawTapeDay(
            date=D2,
            mapping=pl.DataFrame({"ValueCode": ["A"], "QuoteCode": ["QAF6"]}),
            spot_states=pl.from_dicts(
                [_state(D2, "A", "QAF6", 100, 1, bid=None, ask=None)],
                infer_schema_length=None,
            ),
            future_states=pl.from_dicts(
                [_state(D2, "A", "QAF6", 250, 1, bid=101.0, ask=103.0)],
                infer_schema_length=None,
            ),
            spot_trades=pl.DataFrame(
                {
                    "Date": [D2, D2],
                    "ValueCode": ["A", "A"],
                    "QuoteCode": ["QAF6", "QAF6"],
                    "recv_time_ns": [200, 300],
                    "sequence": [2, 3],
                    "packet_sequence": [2, 3],
                    "trade_price": [101.0, 102.0],
                    "trade_lots": [1, 1],
                }
            ),
            future_trades=pl.DataFrame(),
            audit=pl.DataFrame(),
        )
        mark_frame = build_last_observed_session_liquidation_mark(
            tape,
            value_code="A",
            quote_code="QAF6",
            expiry_session=D2,
            calendar_version="synthetic-calendar-v1",
            source_identity_sha256=SHA_C,
        )
        mark = mark_frame.row(0, named=True)
        self.assertEqual(mark["spot_close_price"], 102.0)
        self.assertEqual(mark["spot_close_time_ns"], 300)
        self.assertEqual(mark["future_close_price"], 103.0)
        self.assertEqual(
            mark["spot_close_source"],
            "last_observed_session_spot_trade_not_official_close",
        )
        self.assertFalse(mark["spot_mark_is_executable_bbo"])
        self.assertTrue(mark["future_mark_is_executable_bbo"])
        self.assertTrue(mark["mark_uses_trade_fallback"])
        self.assertFalse(mark["mark_is_official_close"])
        self.assertFalse(mark["mark_is_official_settlement"])
        overlay = apply_supplemental_carry_terminal_overlay(
            _paths(
                _path(
                    "expiry-trade",
                    category="censored",
                    outcome_status="right_censored_expiry_settlement_unpriced",
                )
            ),
            pl.from_dicts(
                [_entry("expiry-trade", spot=100.0, future=105.0)],
                infer_schema_length=None,
            ),
            None,
            mark_frame,
        ).row(0, named=True)
        self.assertEqual(
            overlay["supplemental_terminal_resolution"],
            "expiry_last_observed_session_liquidation_mark",
        )
        self.assertTrue(overlay["expiry_uses_last_observed_session_mark"])
        self.assertFalse(overlay["expiry_uses_last_valid_session_mark"])
        self.assertTrue(overlay["expiry_mark_uses_trade_fallback"])
        self.assertIn("not_official_close_or_settlement", overlay["terminal_reason"])


def _state(
    date: str,
    value_code: str,
    quote_code: str,
    recv_time_ns: int,
    sequence: int,
    *,
    bid: float | None,
    ask: float | None,
    raw_has_book: bool = True,
) -> dict[str, object]:
    return {
        "Date": date,
        "ValueCode": value_code,
        "QuoteCode": quote_code,
        "recv_time_ns": recv_time_ns,
        "sequence": sequence,
        "packet_sequence": sequence,
        "raw_has_book": raw_has_book,
        "book_state_available": True,
        "exec_bid_price": bid,
        "exec_ask_price": ask,
    }


if __name__ == "__main__":
    unittest.main()
