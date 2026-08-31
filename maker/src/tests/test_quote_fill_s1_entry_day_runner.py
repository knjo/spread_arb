"""Wiring contracts for the production S1 entry-day runner."""

from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from inspect import signature
from types import MappingProxyType

import polars as pl

from ..quote_fill.capacity_ledger import CapacityLedger
from ..quote_fill.makerfill_adapter import MakerFillLabelIndex
from ..quote_fill.policy_spec import POLICY_IDS, TOD_BUCKETS, PolicySpec
from ..quote_fill.s1_accounting_bridge import (
    S1AccountingBridge,
    S1AccountingProduct,
)
from ..quote_fill.s1_entry_day_runner import (
    SPOT_CLOSE_DELAY_SECONDS,
    PreparedS1EntryDay,
    _extend_identities_with_required_exit_only_bindings,
    _products_from_identities,
    _specs_by_policy,
    _validate_development_date,
    _validate_policy_ids,
    _validated_product_identities,
    prepare_s1_entry_day,
    run_s1_entry_policies,
    run_s1_entry_policy,
)
from ..quote_fill.s1_event_loop import S1CarryContractBinding, S1Product
from ..quote_fill.s1_raw_book_adapter import build_raw_book_day_index
from ..quote_fill.s1_scenario_spec import (
    SCENARIO_IDS,
    S1ScenarioSpec,
    build_s1_scenario_spec_table,
)
from ..quote_fill.s1_spot_close_adapter import SpotCloseDayIndex
from ..quote_fill.s1_spot_trade_adapter import (
    SpotTradeDayIndex,
    build_spot_trade_day_index,
)
from .test_quote_fill_s1_scenario_spec import (
    _convergence,
    _entry_lookup,
    _mother,
)

DATE = "20260505"
BASE = datetime(2026, 5, 5, 1, 5, tzinfo=UTC).replace(tzinfo=None)


def _specs(policy_id: str = "fixed20") -> tuple[PolicySpec, ...]:
    distance = float(policy_id.removeprefix("fixed"))
    return tuple(
        PolicySpec(
            Date=DATE,
            ValueCode="2330",
            QuoteCode="CDFE6",
            entry_tod_bucket=bucket,
            policy_id=policy_id,
            kind="fixed",
            upper_distance_bp=distance,
            lower_distance_bp=distance,
            upper_source_id=f"constant_bp:{int(distance)}",
            lower_source_id=f"constant_bp:{int(distance)}",
            upper_source_asof_date=None,
            lower_source_asof_date=None,
            combined_source_asof_date=None,
        )
        for bucket in TOD_BUCKETS
    )


def _row(
    *,
    code: str,
    channel: int,
    packet: int,
    future: bool = False,
    recv_time: datetime = BASE,
    fill_price: float = 0.0,
    fill_lots: int = 0,
    close_price: float = 0.0,
) -> dict[str, object]:
    row: dict[str, object] = {
        "RecvTime": recv_time,
        "QuoteCode": code,
        "ChannelSeq": channel,
        "PacketSeq": packet,
        "TrialMatch": 0,
        "FillPrice": fill_price,
        "FillLots": fill_lots,
        "Close": close_price,
        "BestBidPrice": 0,
        "BestBidLots": 0,
        "BestAskPrice": 0,
        "BestAskLots": 0,
    }
    for side in ("Bid", "Ask"):
        for level in range(1, 6):
            row[f"{side}Price{level}"] = 0
            row[f"{side}Lots{level}"] = 0
    if future:
        row.update(
            {
                "DecimalLocator": 2,
                "BidPrice1": 10_100,
                "BidLots1": 5,
                "AskPrice1": 10_200,
                "AskLots1": 5,
            }
        )
    else:
        row.update(
            {
                "BidPrice1": 100.5,
                "BidLots1": 5,
                "BidPrice2": 100.0,
                "BidLots2": 10,
                "AskPrice1": 101.0,
                "AskLots1": 5,
            }
        )
    return row


def _prepared(
    *,
    entry_fill_seconds: float | None = None,
    include_exit_trade: bool = False,
    include_exit_only_product: bool = False,
    expiry_today: bool = False,
) -> PreparedS1EntryDay:
    recv_ns = int(pl.Series([BASE]).dt.timestamp("ns")[0])
    expiry = recv_ns + 15_000_000_000
    value_codes = ["2330"]
    quote_codes = ["CDFE6"]
    spot_refs = [100.5]
    future_refs = [101.0]
    if include_exit_only_product:
        value_codes.append("2603")
        quote_codes.append("CZFE6")
        spot_refs.append(50.0)
        future_refs.append(50.5)
    mapping = pl.DataFrame(
        {
            "ValueCode": value_codes,
            "QuoteCode": quote_codes,
            "spot_ref_price": spot_refs,
            "fut_ref_price": future_refs,
        }
    )
    spot_rows = [_row(code="2330", channel=17, packet=1)]
    future_rows = [_row(code="CDFE6", channel=18, packet=2, future=True)]
    if include_exit_only_product:
        spot_rows.append(_row(code="2603", channel=27, packet=1))
        future_rows.append(_row(code="CZFE6", channel=28, packet=2, future=True))
    if include_exit_trade:
        spot_rows.append(
            _row(
                code="2330",
                channel=19,
                packet=3,
                recv_time=BASE + timedelta(seconds=1),
                fill_price=103.0,
                fill_lots=2,
            )
        )
    if expiry_today:
        spot_rows.append(
            _row(
                code="2330",
                channel=29,
                packet=4,
                recv_time=BASE + timedelta(seconds=15 + SPOT_CLOSE_DELAY_SECONDS + 1),
                close_price=101.5,
            )
        )
    spot_frame = pl.from_dicts(spot_rows)
    raw = build_raw_book_day_index(
        spot_frame,
        pl.from_dicts(future_rows),
        mapping,
    )
    spot_trades = build_spot_trade_day_index(spot_frame, mapping)
    spot_closes = (
        SpotCloseDayIndex.from_selected_rows(
            spot_frame,
            mapping.filter(pl.col("ValueCode") == "2330"),
            date=DATE,
            close_not_before_ns=(expiry + SPOT_CLOSE_DELAY_SECONDS * 1_000_000_000),
        )
        if expiry_today
        else None
    )
    fill_seconds = float("nan") if entry_fill_seconds is None else entry_fill_seconds
    labels = MakerFillLabelIndex.from_frame(
        pl.DataFrame(
            {
                "QuoteCode": ["2330"],
                "ChannelSeq": pl.Series([17], dtype=pl.UInt64),
                "Bid1_FillSeconds": pl.Series([fill_seconds], dtype=pl.Float32),
                "Bid2_FillSeconds": pl.Series([float("nan")], dtype=pl.Float32),
            }
        )
    )
    common = pl.DataFrame(
        {
            "Date": [DATE],
            "ValueCode": ["2330"],
            "QuoteCode": ["CDFE6"],
            "decision_time_ns": [recv_ns],
            "analysis_eligible": [True],
            "selected_anchor_bp": [0.0],
            "contract_size": [2_000.0],
        }
    )
    changes = pl.DataFrame(
        {
            "Date": [DATE],
            "ValueCode": ["2330"],
            "decision_time_ns": [recv_ns],
        }
    )
    product = S1Product(
        "2330",
        "2330",
        "CDFE6",
        2_000,
        1,
        expiry,
        expiry,
        DATE if expiry_today else "20260520",
    )
    products = [product]
    if include_exit_only_product:
        products.append(
            S1Product(
                "2603",
                "2603",
                "CZFE6",
                2_000,
                1,
                expiry,
                expiry,
                "20260520",
            )
        )
    return PreparedS1EntryDay(
        date=DATE,
        policy_ids=("fixed20",),
        specs_by_policy=MappingProxyType({"fixed20": _specs()}),
        common_decisions=common,
        state_changes_by_policy=MappingProxyType({"fixed20": changes}),
        raw_books=raw,
        spot_trades=spot_trades,
        spot_closes=spot_closes,
        makerfill_labels=labels,
        products=tuple(products),
        entry_product_ids=("2330",),
        day_open_time_ns=recv_ns - 1_000_000_000,
        entry_cutoff_time_ns=recv_ns + 10_000_000_000,
        session_expiry_time_ns=expiry,
    )


class S1EntryDayRunnerTest(unittest.TestCase):
    def test_missing_daily_mapping_adds_exact_raw_backed_carry_pair(self) -> None:
        identities = pl.DataFrame(
            {
                "ValueCode": ["2330"],
                "QuoteCode": ["CDFE6"],
                "spot_ref_price": [100.0],
                "fut_ref_price": [101.0],
                "contract_size": [2_000.0],
                "end_date": ["20260520"],
            }
        )
        binding = S1CarryContractBinding("2603", "2603", "CZFE6", 2_000, 1, "20260520")
        extended = _extend_identities_with_required_exit_only_bindings(
            identities,
            (binding,),
            date=DATE,
            contracts=pl.DataFrame(
                {
                    "ValueCode": ["2603"],
                    "QuoteCode": ["CZFE6"],
                    "contract_size": [2_000],
                    "decimal_locator": [2],
                    "end_date": ["2026-05-20"],
                    "fut_ref_price": [50.5],
                }
            ).lazy(),
            spot_raw=pl.DataFrame(
                {"QuoteCode": ["2603", "2603"], "RefPrice": [50.0, 50.0]}
            ).lazy(),
            future_raw=pl.DataFrame(
                {"QuoteCode": ["CZFE6"], "DecimalLocator": [2]}
            ).lazy(),
        )

        self.assertEqual(extended["ValueCode"].to_list(), ["2330", "2603"])
        carry = extended.filter(pl.col("ValueCode") == "2603").row(0, named=True)
        self.assertEqual(carry["QuoteCode"], "CZFE6")
        self.assertAlmostEqual(carry["spot_ref_price"], 50.0)
        self.assertAlmostEqual(carry["fut_ref_price"], 50.5)

    def test_missing_carry_pair_fails_closed_on_raw_or_identity_drift(self) -> None:
        identities = pl.DataFrame(
            {
                "ValueCode": ["2603"],
                "QuoteCode": ["CZFG6"],
                "spot_ref_price": [50.0],
                "fut_ref_price": [50.5],
                "contract_size": [2_000.0],
                "end_date": ["20260520"],
            }
        )
        binding = S1CarryContractBinding("2603", "2603", "CZFE6", 2_000, 1, "20260520")
        kwargs = {
            "date": DATE,
            "contracts": pl.DataFrame(
                {
                    "ValueCode": ["2603"],
                    "QuoteCode": ["CZFE6"],
                    "contract_size": [2_000],
                    "decimal_locator": [2],
                    "end_date": ["20260520"],
                    "fut_ref_price": [50.5],
                }
            ).lazy(),
            "spot_raw": pl.DataFrame(
                {"QuoteCode": ["2603"], "RefPrice": [50.0]}
            ).lazy(),
            "future_raw": pl.DataFrame(
                {"QuoteCode": ["CZFE6"], "DecimalLocator": [2]}
            ).lazy(),
        }
        with self.assertRaisesRegex(ValueError, "conflicts with frozen"):
            _extend_identities_with_required_exit_only_bindings(
                identities, (binding,), **kwargs
            )

        missing = identities.clear()
        bad_raw = dict(kwargs)
        bad_raw["future_raw"] = pl.DataFrame(
            {"QuoteCode": ["CZFE6"], "DecimalLocator": [3]}
        ).lazy()
        with self.assertRaisesRegex(ValueError, "DecimalLocator"):
            _extend_identities_with_required_exit_only_bindings(
                missing, (binding,), **bad_raw
            )

    def test_prepared_day_runs_shared_clock_and_reports_no_pnl(self) -> None:
        prepared = _prepared()
        self.assertIsInstance(prepared.spot_trades, SpotTradeDayIndex)
        self.assertEqual(prepared.spot_trades.product_ids, ("2330",))
        self.assertEqual(prepared.spot_trades.trade_count, 0)

        run = run_s1_entry_policy(prepared, "fixed20")
        summary = run.summary
        self.assertEqual(summary.sent_entry_orders, 1)
        self.assertEqual(summary.makerfill_supported_orders, 1)
        self.assertEqual(summary.makerfill_eod_positive_orders, 0)
        self.assertEqual(summary.actual_active_entry_fills, 0)
        self.assertGreater(summary.decision_economic_checks, 0)
        self.assertEqual(summary.actual_send_economic_checks, 1)
        self.assertEqual(summary.actual_send_economic_gate_open, 1)
        self.assertEqual(summary.actual_send_economic_gate_closed, 0)
        self.assertEqual(summary.actual_send_not_sent_after_gate_pass, 0)
        self.assertEqual(len(run.economic_gate_audits), 1)
        self.assertEqual(run.economic_gate_audits[0].dispatch_outcome, "sent")
        self.assertEqual(
            run.economic_gate_audits[0].raw_order_fact_id,
            run.result.orders[0].raw_order_fact_id,
        )
        self.assertAlmostEqual(summary.makerfill_support_rate_of_sent or 0.0, 1.0)
        self.assertAlmostEqual(summary.actual_active_fill_rate_of_sent or 0.0, 0.0)
        self.assertEqual(summary.spot_requests_sent, 2)
        self.assertFalse(summary.performance_available)
        self.assertEqual(
            summary.performance_unavailable_reason,
            "normal_exit_route_not_integrated",
        )
        self.assertIsNone(run.result.mean_daily_net_twd)

    def test_opt_in_normal_exit_uses_shared_trade_index_and_accounting(self) -> None:
        prepared = _prepared(entry_fill_seconds=0.1, include_exit_trade=True)
        self.assertEqual(prepared.spot_trades.trade_count, 1)

        run = run_s1_entry_policy(
            prepared,
            "fixed20",
            normal_exit_enabled=True,
        )

        self.assertTrue(run.result.normal_exit_enabled)
        self.assertEqual(
            [fact.role for fact in run.result.executions],
            ["entry_maker", "entry_hedge", "exit_maker", "exit_hedge"],
        )
        self.assertEqual(run.result.positions[0].state, "exit_maker_flat")
        self.assertEqual(run.summary.entry_hedge_executions, 1)
        self.assertIsNotNone(run.summary.entry_hedge_success_rate)
        assert run.summary.entry_hedge_success_rate is not None
        self.assertAlmostEqual(run.summary.entry_hedge_success_rate, 1.0)
        self.assertEqual(run.summary.eod_paired_open_positions, 0)
        self.assertEqual(run.summary.paired_open_positions, 0)
        self.assertFalse(run.summary.performance_available)
        self.assertEqual(
            run.summary.performance_unavailable_reason,
            "portfolio_accounting_aggregation_required",
        )
        self.assertEqual(len(run.result.exit_physical_fills), 1)
        self.assertEqual(run.result.exit_physical_fills[0].fill_reason, "trade_through")

    def test_full_mapping_keeps_exit_only_product_but_never_enables_entry(self) -> None:
        prepared = _prepared(include_exit_only_product=True)
        self.assertEqual(prepared.entry_product_ids, ("2330",))
        self.assertEqual(
            tuple(product.product_id for product in prepared.products),
            ("2330", "2603"),
        )
        self.assertEqual(prepared.spot_trades.product_ids, ("2330", "2603"))
        self.assertEqual(
            {key.value_code for key in prepared.raw_books.keys},
            {"2330", "2603"},
        )

        run = run_s1_entry_policy(prepared, "fixed20")
        self.assertEqual({order.product_id for order in run.result.orders}, {"2330"})
        self.assertEqual(run.result.entry_disabled_product_ids, frozenset({"2603"}))
        self.assertEqual(run.summary.product_count, 1)
        with self.assertRaisesRegex(ValueError, "non-entry products"):
            run_s1_entry_policy(
                prepared,
                "fixed20",
                entry_enabled_product_ids=frozenset({"2330", "2603"}),
            )

    def test_expiry_close_event_uses_injected_accounting_and_capacity(self) -> None:
        prepared = _prepared(entry_fill_seconds=0.1, expiry_today=True)
        self.assertIsNotNone(prepared.spot_closes)
        accounting = S1AccountingBridge(
            default_date=DATE,
            scenario_id="fixed20",
            products=(S1AccountingProduct("2330", "2330", 2_000),),
        )
        capacity = CapacityLedger()

        run = run_s1_entry_policy(
            prepared,
            "fixed20",
            normal_exit_enabled=True,
            accounting_adapter=accounting,
            capacity_ledger=capacity,
            carry_in=(),
            entry_enabled_product_ids=frozenset({"2330"}),
        )

        self.assertEqual(run.result.expiry_mark_count, 1)
        self.assertEqual(len(run.result.expiry_marks), 1)
        self.assertAlmostEqual(run.result.expiry_marks[0].spot_close_price, 101.5)
        self.assertEqual(
            run.result.positions[0].state,
            "expiry_basis_zero_accounting",
        )
        self.assertGreater(len(capacity.transitions), 0)
        self.assertTrue(
            any(
                getattr(fact, "terminal_outcome", None)
                == "expiry_basis_zero_accounting"
                for fact in accounting.facts
            )
        )
        accounting.verify()

    def test_entry_only_expiry_day_remains_accounting_free(self) -> None:
        prepared = _prepared(expiry_today=True)
        run = run_s1_entry_policy(prepared, "fixed20")
        self.assertEqual(run.result.expiry_mark_count, 0)
        self.assertEqual(run.result.expiry_marks, ())
        self.assertFalse(run.result.normal_exit_enabled)

    def test_full_mapping_identity_validation_only_requires_entry_subset_in_common(
        self,
    ) -> None:
        mapping = pl.DataFrame(
            {
                "Date": [DATE, DATE],
                "ValueCode": ["2330", "2603"],
                "QuoteCode": ["CDFE6", "CZFE6"],
                "spot_ref_price": [100.5, 50.0],
                "fut_ref_price": [101.0, 50.5],
                "contract_size": [2_000.0, 2_000.0],
                "end_date": [
                    datetime(2026, 5, 20, tzinfo=UTC).date(),
                    datetime(2026, 5, 20, tzinfo=UTC).date(),
                ],
            }
        )
        common = mapping.filter(pl.col("ValueCode") == "2330")
        product_keys = common.select("Date", "ValueCode", "QuoteCode")

        identities = _validated_product_identities(common, mapping, product_keys)
        self.assertEqual(identities["ValueCode"].to_list(), ["2330", "2603"])
        self.assertEqual(identities["end_date"].to_list(), ["20260520", "20260520"])
        products = _products_from_identities(identities, 123)
        self.assertEqual([product.end_date for product in products], ["20260520"] * 2)

        missing_entry = mapping.filter(pl.col("ValueCode") == "2603")
        with self.assertRaisesRegex(ValueError, "cover every PolicySpec"):
            _validated_product_identities(common, missing_entry, product_keys)

    def test_prepared_expiry_product_requires_complete_close_index(self) -> None:
        prepared = _prepared(expiry_today=True)
        assert prepared.spot_closes is not None
        empty = SpotCloseDayIndex(
            DATE,
            prepared.spot_closes.close_not_before_ns,
            ("2330",),
            {},
        )
        with self.assertRaisesRegex(ValueError, "lacks an official spot close"):
            replace(prepared, spot_closes=empty)

    def test_multiple_policies_cannot_share_stateful_injections(self) -> None:
        prepared = _prepared()
        multi = replace(
            prepared,
            policy_ids=("fixed20", "fixed25"),
            specs_by_policy=MappingProxyType(
                {
                    "fixed20": _specs(),
                    "fixed25": _specs("fixed25"),
                }
            ),
            state_changes_by_policy=MappingProxyType(
                {
                    "fixed20": prepared.state_changes_by_policy["fixed20"],
                    "fixed25": prepared.state_changes_by_policy["fixed20"],
                }
            ),
        )

        with self.assertRaisesRegex(ValueError, "cannot share"):
            list(run_s1_entry_policies(multi, capacity_ledger=CapacityLedger()))

    def test_scenario_grid_is_default_and_cannot_mix_with_legacy_ids(self) -> None:
        default_ids = signature(prepare_s1_entry_day).parameters[
            "policy_ids"
        ].default
        self.assertEqual(default_ids, SCENARIO_IDS)
        self.assertEqual(_validate_policy_ids(SCENARIO_IDS), SCENARIO_IDS)
        self.assertEqual(_validate_policy_ids(POLICY_IDS), POLICY_IDS)
        with self.assertRaisesRegex(ValueError, "cannot be mixed"):
            _validate_policy_ids((POLICY_IDS[0], SCENARIO_IDS[0]))

    def test_unsupported_scenario_spec_is_retained_as_explicit_no_trade(
        self,
    ) -> None:
        scenario_id = "q95_C2_sd_f5"
        table = build_s1_scenario_spec_table(
            _mother(),
            _entry_lookup(),
            _convergence(),
        ).filter(pl.col("scenario_id") == scenario_id)
        specs_by_policy = _specs_by_policy(table, (scenario_id,))
        specs = specs_by_policy[scenario_id]

        self.assertEqual(len(specs), len(TOD_BUCKETS))
        self.assertTrue(all(isinstance(spec, S1ScenarioSpec) for spec in specs))
        first = next(
            spec for spec in specs if spec.entry_tod_bucket == TOD_BUCKETS[0]
        )
        self.assertFalse(first.lookup_supported)
        self.assertEqual(first.lookup_support_reason, "lower_unsupported")

        legacy = _prepared()
        prepared = replace(
            legacy,
            policy_ids=(scenario_id,),
            specs_by_policy=MappingProxyType({scenario_id: specs}),
            state_changes_by_policy=MappingProxyType(
                {scenario_id: legacy.state_changes_by_policy["fixed20"]}
            ),
        )
        run = run_s1_entry_policy(prepared, scenario_id)
        self.assertEqual(run.summary.sent_entry_orders, 0)
        self.assertEqual(run.result.orders, ())
        self.assertEqual(run.result.spot_requests_sent, 0)

    def test_unprepared_policy_and_forward_window_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "not present"):
            run_s1_entry_policy(_prepared(), "q95")
        with self.assertRaisesRegex(TypeError, "boolean"):
            run_s1_entry_policy(
                _prepared(),
                "fixed20",
                normal_exit_enabled=1,  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(ValueError, "protected"):
            _validate_development_date("20260814")
        with self.assertRaisesRegex(ValueError, "valid"):
            _validate_development_date("20260230")


if __name__ == "__main__":
    unittest.main()
