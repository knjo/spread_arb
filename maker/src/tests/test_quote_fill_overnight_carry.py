from __future__ import annotations

import unittest

import polars as pl

from maker.src.quote_fill.overnight_carry import (
    OvernightCarryConfig,
    build_strict_overnight_carry_labels,
)
from maker.src.quote_fill.raw_tape import RawTapeDay


DATE = "20260601"
NEXT = "20260602"
LATER = "20260603"
VALUE = "2303"
QUOTE = "CCFF6"


def _policy(branch: str = "carry_at_eod_cancel_unconfirmed") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Date": [DATE],
            "ValueCode": [VALUE],
            "QuoteCode": [QUOTE],
            "entry_route": ["future_ask_spot_taker"],
            "entry_policy_generation_id": ["entry-policy-1"],
            "entry_raw_order_fact_id": ["entry-raw-1"],
            "exit_rule_id": ["frozen_center"],
            "exit_route": ["future_bid_spot_taker"],
            "exit_policy_trial_id": ["exit-policy-1"],
            "position_status": ["position_established"],
            "position_established_ns": [1_000_000_000],
            "branch_status": [branch],
            "needs_next_session_label": [True],
        }
    )


def _entry() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Date": [DATE],
            "ValueCode": [VALUE],
            "QuoteCode": [QUOTE],
            "route": ["future_ask_spot_taker"],
            "policy_generation_id": ["entry-policy-1"],
            "raw_order_fact_id": ["entry-raw-1"],
            "full_fill": [True],
            "entry_hedge_status": ["executable"],
            "entry_hedge_contract_size_shares": [2_000],
            "entry_spot_price": [100.0],
            "entry_future_price": [101.0],
        }
    )


def _calendar(expiry: str = "20260617") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "QuoteCode": [QUOTE],
            "expiry_session": [expiry],
            "calendar_version": ["official-test-calendar-v1"],
        }
    )


def _state_row(
    *,
    date: str = NEXT,
    market: str,
    quote: str,
    recv: int,
    sequence: int,
    bid1: float,
    bid1_lots: int,
    bid2: float,
    bid2_lots: int,
    ask1: float,
    ask1_lots: int,
) -> dict[str, object]:
    row: dict[str, object] = {
        "Date": date,
        "market": market,
        "ValueCode": VALUE,
        "QuoteCode": quote,
        "recv_time_ns": recv,
        "sequence": sequence,
        "packet_sequence": sequence,
        "trial_match": False,
        "raw_has_book": True,
        "book_state_available": True,
        "book_recv_time_ns": recv,
        "best_bid_price": None,
        "best_bid_lots": None,
        "best_ask_price": None,
        "best_ask_lots": None,
    }
    for side in ("bid", "ask"):
        for level in range(1, 6):
            row[f"{side}_price_{level}"] = None
            row[f"{side}_lots_{level}"] = None
    row.update(
        {
            "bid_price_1": bid1,
            "bid_lots_1": bid1_lots,
            "bid_price_2": bid2,
            "bid_lots_2": bid2_lots,
            "ask_price_1": ask1,
            "ask_lots_1": ask1_lots,
        }
    )
    return row


def _tape(
    quote: str = QUOTE,
    include_future: bool = True,
    *,
    date: str = NEXT,
) -> RawTapeDay:
    # Future arrives first at the same receive timestamp.  The first causal
    # joint state is therefore the later spot event (priority 2).
    future = pl.from_dicts(
        [
            _state_row(
                market="future",
                date=date,
                quote=quote,
                recv=10_000_000_000,
                sequence=1,
                bid1=100.0,
                bid1_lots=5,
                bid2=99.5,
                bid2_lots=5,
                ask1=100.5,
                ask1_lots=4,
            )
        ]
    ) if include_future else pl.DataFrame(schema={name: dtype for name, dtype in _empty_state_schema().items()})
    spot = pl.from_dicts(
        [
            _state_row(
                market="spot",
                date=date,
                quote=quote,
                recv=10_000_000_000,
                sequence=2,
                bid1=100.5,
                bid1_lots=1,
                bid2=100.0,
                bid2_lots=2,
                ask1=101.0,
                ask1_lots=10,
            )
        ]
    )
    mapping = pl.DataFrame(
        {
            "ValueCode": [VALUE],
            "QuoteCode": [quote],
            "spot_ref_price": [100.0],
            "fut_ref_price": [100.0],
            "contract_size": [2_000.0],
        }
    )
    return RawTapeDay(date, mapping, spot, future, pl.DataFrame(), pl.DataFrame(), pl.DataFrame())


def _empty_state_schema() -> dict[str, pl.DataType]:
    row = _state_row(
        market="future",
        quote=QUOTE,
        recv=1,
        sequence=1,
        bid1=1.0,
        bid1_lots=1,
        bid2=0.5,
        bid2_lots=1,
        ask1=1.5,
        ask1_lots=1,
    )
    return pl.from_dicts([row]).schema


class OvernightCarryTests(unittest.TestCase):
    def test_exact_contract_first_joint_raw_close_prices_gross_cashflow(self) -> None:
        result = build_strict_overnight_carry_labels(
            _policy(),
            _entry(),
            _tape(),
            [DATE, NEXT],
            _calendar(),
        )
        self.assertEqual(result.labels.height, 1)
        row = result.labels.row(0, named=True)
        # Spot sells two board lots: (100.5 + 100.0) / 2 = 100.25.
        self.assertAlmostEqual(row["exit_spot_price"], 100.25)
        self.assertAlmostEqual(row["exit_future_price"], 100.5)
        self.assertAlmostEqual(row["gross_cycle_pnl_twd"], 1_500.0)
        self.assertEqual(row["terminal_branch"], "overnight_exit")
        self.assertEqual(row["transition_branch"], "same_exact_contract")
        self.assertTrue(row["terminal_cashflow_priced"])
        self.assertFalse(row["pathwise_ev_ready"])
        self.assertIsNone(row["fee_cost_bp"])
        self.assertIsNone(row["tax_cost_bp"])
        self.assertEqual(row["exit_decision_time_ns"], 10_000_000_000)
        self.assertEqual(row["exit_future_snapshot_time_ns"], 10_000_000_000)
        self.assertEqual(
            row["execution_benchmark_role"],
            "optimistic_zero_added_latency_same_cursor_taker_taker",
        )
        self.assertEqual(row["added_exit_latency_ns"], 0)
        self.assertFalse(row["latency_matched_to_exit_maker"])
        self.assertFalse(row["max_book_age_enforced"])
        self.assertEqual(row["book_freshness_status"], "ungated_age_recorded")

    def test_unknown_cancel_race_is_excluded_not_promoted_to_carry(self) -> None:
        result = build_strict_overnight_carry_labels(
            _policy("cancel_race_unknown"),
            _entry(),
            _tape(),
            [DATE, NEXT],
            _calendar(),
        )
        self.assertEqual(result.labels.height, 0)
        audit = result.audit.row(0, named=True)
        self.assertEqual(audit["strict_carry_policy_rows"], 0)
        self.assertEqual(audit["excluded_noncarry_or_unknown_rows"], 1)

    def test_different_next_contract_is_never_substituted(self) -> None:
        result = build_strict_overnight_carry_labels(
            _policy(),
            _entry(),
            _tape("CCFG6"),
            [DATE, NEXT],
            _calendar(),
        )
        row = result.labels.row(0, named=True)
        self.assertEqual(row["label_status"], "roll_substitution_forbidden")
        self.assertIsNone(row["terminal_branch"])
        self.assertEqual(row["transition_branch"], "roll_substitution_forbidden")
        self.assertFalse(row["roll_attempted"])
        self.assertIsNone(row["new_quote_code"])
        self.assertIsNone(row["gross_cycle_pnl_twd"])

    def test_expiry_stays_unpriced_without_final_settlement_fact(self) -> None:
        result = build_strict_overnight_carry_labels(
            _policy(),
            _entry(),
            _tape("CCFG6"),
            [DATE, NEXT],
            _calendar(DATE),
        )
        row = result.labels.row(0, named=True)
        self.assertEqual(row["label_status"], "expiry_settlement_unpriced")
        self.assertEqual(row["transition_branch"], "expiry_settlement_required")
        self.assertFalse(row["terminal_cashflow_priced"])
        self.assertFalse(row["pathwise_ev_ready"])

    def test_final_settlement_plus_raw_spot_unwind_prices_expiry_branch(self) -> None:
        settlements = pl.DataFrame(
            {
                "QuoteCode": [QUOTE],
                "expiry_session": [DATE],
                "settlement_price": [100.75],
                "settlement_time_ns": [8_000_000_000],
                "settlement_source_version": ["official-final-v1"],
                "settlement_final": [True],
            }
        )
        result = build_strict_overnight_carry_labels(
            _policy(),
            _entry(),
            _tape("CCFG6"),
            [DATE, NEXT],
            _calendar(DATE),
            settlement_facts=settlements,
        )
        row = result.labels.row(0, named=True)
        self.assertEqual(row["terminal_branch"], "expiry_settlement")
        self.assertEqual(row["transition_branch"], "expiry_settlement")
        self.assertAlmostEqual(row["exit_spot_price"], 100.25)
        self.assertAlmostEqual(row["exit_future_price"], 100.75)
        self.assertAlmostEqual(row["gross_cycle_pnl_twd"], 1_000.0)
        self.assertEqual(row["settlement_source_version"], "official-final-v1")
        self.assertTrue(row["terminal_cashflow_priced"])
        self.assertFalse(row["pathwise_ev_ready"])

    def test_calendar_is_mandatory_and_missing_exact_contract_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing contract calendar"):
            build_strict_overnight_carry_labels(
                _policy(),
                _entry(),
                _tape(),
                [DATE, NEXT],
                _calendar().head(0),
                config=OvernightCarryConfig(),
            )

    def test_missing_first_candidate_session_censors_before_later_tape(self) -> None:
        result = build_strict_overnight_carry_labels(
            _policy(),
            _entry(),
            [_tape(date=LATER)],
            [DATE, NEXT, LATER],
            _calendar(),
            config=OvernightCarryConfig(max_carry_sessions=2),
        )
        row = result.labels.row(0, named=True)
        self.assertEqual(row["label_status"], "unresolved_missing_raw_tape")
        self.assertIn("later_sessions_not_examined", row["unresolved_reason"])
        self.assertFalse(row["terminal_cashflow_priced"])
        self.assertIsNone(row["exit_date"])

    def test_expiry_spot_book_must_be_newer_than_final_settlement(self) -> None:
        settlement_time = 11_000_000_000
        settlements = pl.DataFrame(
            {
                "QuoteCode": [QUOTE],
                "expiry_session": [DATE],
                "settlement_price": [100.75],
                "settlement_time_ns": [settlement_time],
                "settlement_source_version": ["official-final-v1"],
                "settlement_final": [True],
            }
        )
        base = _tape("CCFG6")
        old_book = base.spot_states.row(0, named=True)
        forwarded = dict(old_book)
        forwarded.update(
            {
                "recv_time_ns": 12_000_000_000,
                "sequence": 3,
                "packet_sequence": 3,
                "raw_has_book": False,
                "book_recv_time_ns": 10_000_000_000,
            }
        )
        stale_only = RawTapeDay(
            NEXT,
            base.mapping,
            pl.from_dicts([old_book, forwarded]),
            base.future_states,
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
        )
        unresolved = build_strict_overnight_carry_labels(
            _policy(),
            _entry(),
            stale_only,
            [DATE, NEXT],
            _calendar(DATE),
            settlement_facts=settlements,
        ).labels.row(0, named=True)
        self.assertEqual(
            unresolved["label_status"], "expiry_settlement_missing_spot_exit"
        )
        self.assertFalse(unresolved["terminal_cashflow_priced"])

        fresh = _state_row(
            market="spot",
            quote="CCFG6",
            recv=13_000_000_000,
            sequence=4,
            bid1=99.5,
            bid1_lots=1,
            bid2=99.0,
            bid2_lots=2,
            ask1=100.0,
            ask1_lots=10,
        )
        fresh_tape = RawTapeDay(
            NEXT,
            base.mapping,
            pl.from_dicts([old_book, forwarded, fresh]),
            base.future_states,
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
        )
        priced = build_strict_overnight_carry_labels(
            _policy(),
            _entry(),
            fresh_tape,
            [DATE, NEXT],
            _calendar(DATE),
            settlement_facts=settlements,
        ).labels.row(0, named=True)
        self.assertEqual(priced["label_status"], "expiry_settlement")
        self.assertEqual(priced["exit_spot_snapshot_time_ns"], 13_000_000_000)
        self.assertGreaterEqual(
            priced["exit_spot_snapshot_time_ns"], priced["settlement_time_ns"]
        )
        self.assertAlmostEqual(priced["exit_spot_price"], 99.25)

    def test_v1_rejects_multi_contract_quantity(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one futures contract"):
            build_strict_overnight_carry_labels(
                _policy(),
                _entry(),
                _tape(),
                [DATE, NEXT],
                _calendar(),
                config=OvernightCarryConfig(futures_contracts=2),
            )

    def test_policy_rows_expose_shared_physical_entry_dependency(self) -> None:
        second = _policy().with_columns(
            pl.lit("frozen_lower").alias("exit_rule_id"),
            pl.lit("spot_ask_future_taker").alias("exit_route"),
            pl.lit("exit-policy-2").alias("exit_policy_trial_id"),
        )
        result = build_strict_overnight_carry_labels(
            pl.concat([_policy(), second]),
            _entry(),
            _tape(),
            [DATE, NEXT],
            _calendar(),
            config=OvernightCarryConfig(max_book_age_ns=0),
        )
        self.assertEqual(result.labels.height, 2)
        self.assertEqual(result.labels["physical_entry_dependency_id"].n_unique(), 1)
        self.assertEqual(
            set(result.labels["physical_entry_strict_carry_policy_alias_count"]),
            {2},
        )
        self.assertAlmostEqual(
            result.labels["physical_entry_coverage_weight"].sum(), 1.0
        )
        self.assertTrue(result.labels["physical_entry_alias_nonindependent"].all())
        self.assertFalse(result.labels["cross_q_rule_route_additive"].any())
        self.assertEqual(set(result.labels["policy_observation_weight"]), {1.0})
        self.assertTrue(result.labels["max_book_age_enforced"].all())
        self.assertEqual(
            set(result.labels["book_freshness_status"]),
            {"configured_limit_passed"},
        )


if __name__ == "__main__":
    unittest.main()
