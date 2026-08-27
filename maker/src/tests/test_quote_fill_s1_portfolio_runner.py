"""Focused contracts for the date-major seven-policy S1 orchestrator."""

from __future__ import annotations

import gc
import unittest
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from weakref import ReferenceType, ref

import polars as pl

from ..quote_fill.capacity_ledger import (
    CapacityLedger,
    replay_capacity_transitions,
)
from ..quote_fill.makerfill_adapter import MakerFillLabelIndex
from ..quote_fill.policy_spec import (
    ENTRY_CANDIDATE_ID,
    LOWER_CANDIDATE_ID,
    POLICY_IDS,
    TOD_BUCKETS,
    PolicySpec,
)
from ..quote_fill.s1_accounting import (
    ExecutedLeg,
    PositionEstablishedFact,
    TerminalRealizedAccounting,
)
from ..quote_fill.s1_accounting_bridge import S1AccountingProduct
from ..quote_fill.s1_entry_day_runner import (
    PreparedS1EntryDay,
    run_s1_entry_policy,
)
from ..quote_fill.s1_event_loop import S1Product
from ..quote_fill.s1_portfolio_runner import (
    S1PortfolioDayPartition,
    run_s1_portfolio,
    run_s1_portfolio_from_dates,
)
from ..quote_fill.s1_raw_book_adapter import build_raw_book_day_index
from ..quote_fill.s1_spot_trade_adapter import build_spot_trade_day_index

D1 = "20260505"
D2 = "20260506"
VALUE = "2330"
QUOTE = "CDFE6"
CATALOG = (S1AccountingProduct(VALUE, VALUE, 2_000),)
EXTENDED_CATALOG = (
    *CATALOG,
    S1AccountingProduct("2603", "2603", 2_000),
)


def _base(date: str) -> datetime:
    return (
        datetime.strptime(date, "%Y%m%d")
        .replace(
            hour=1,
            minute=5,
            tzinfo=UTC,
        )
        .replace(tzinfo=None)
    )


def _raw_row(
    *,
    quote_code: str,
    channel: int,
    packet: int,
    recv_time: datetime,
    future: bool = False,
    fill_price: float = 0.0,
    fill_lots: int = 0,
) -> dict[str, object]:
    row: dict[str, object] = {
        "RecvTime": recv_time,
        "QuoteCode": quote_code,
        "ChannelSeq": channel,
        "PacketSeq": packet,
        "TrialMatch": 0,
        "FillPrice": fill_price,
        "FillLots": fill_lots,
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


def _policy_specs(
    date: str,
    *,
    value_code: str,
    quote_code: str,
) -> MappingProxyType[str, tuple[PolicySpec, ...]]:
    prior = (
        datetime.strptime(date, "%Y%m%d").replace(tzinfo=UTC) - timedelta(days=1)
    ).strftime("%Y%m%d")
    result: dict[str, tuple[PolicySpec, ...]] = {}
    for policy_id in POLICY_IDS:
        if policy_id.startswith("q"):
            quantile = int(policy_id[1:])
            values = tuple(
                PolicySpec(
                    Date=date,
                    ValueCode=value_code,
                    QuoteCode=quote_code,
                    entry_tod_bucket=bucket,
                    policy_id=policy_id,
                    kind="quantile",
                    upper_distance_bp=20.0,
                    lower_distance_bp=0.0,
                    upper_source_id=ENTRY_CANDIDATE_ID,
                    lower_source_id=LOWER_CANDIDATE_ID,
                    upper_source_asof_date=prior,
                    lower_source_asof_date=prior,
                    combined_source_asof_date=prior,
                    boundary_quantile=quantile,
                )
                for bucket in TOD_BUCKETS
            )
        else:
            distance = float(policy_id.removeprefix("fixed"))
            source_id = f"constant_bp:{int(distance)}"
            values = tuple(
                PolicySpec(
                    Date=date,
                    ValueCode=value_code,
                    QuoteCode=quote_code,
                    entry_tod_bucket=bucket,
                    policy_id=policy_id,
                    kind="fixed",
                    upper_distance_bp=distance,
                    lower_distance_bp=distance,
                    upper_source_id=source_id,
                    lower_source_id=source_id,
                    upper_source_asof_date=None,
                    lower_source_asof_date=None,
                    combined_source_asof_date=None,
                )
                for bucket in TOD_BUCKETS
            )
        result[policy_id] = values
    return MappingProxyType(result)


def _prepared(
    date: str,
    *,
    value_code: str = VALUE,
    quote_code: str = QUOTE,
    contract_size: int = 2_000,
    include_exit_trade: bool = False,
) -> PreparedS1EntryDay:
    base = _base(date)
    recv_ns = int(pl.Series([base]).dt.timestamp("ns")[0])
    session_expiry_ns = recv_ns + 15_000_000_000
    mapping = pl.DataFrame(
        {
            "ValueCode": [value_code],
            "QuoteCode": [quote_code],
            "spot_ref_price": [100.5],
            "fut_ref_price": [101.0],
        }
    )
    spot_rows = [
        _raw_row(
            quote_code=value_code,
            channel=17,
            packet=1,
            recv_time=base,
        )
    ]
    if include_exit_trade:
        spot_rows.append(
            _raw_row(
                quote_code=value_code,
                channel=18,
                packet=2,
                recv_time=base + timedelta(seconds=1),
                fill_price=110.0,
                fill_lots=2,
            )
        )
    spot = pl.from_dicts(spot_rows)
    future = pl.from_dicts(
        [
            _raw_row(
                quote_code=quote_code,
                channel=27,
                packet=1,
                recv_time=base,
                future=True,
            )
        ]
    )
    common = pl.DataFrame(
        {
            "Date": [date],
            "ValueCode": [value_code],
            "QuoteCode": [quote_code],
            "decision_time_ns": [recv_ns],
            "analysis_eligible": [True],
            "selected_anchor_bp": [0.0],
            "contract_size": [float(contract_size)],
        }
    )
    changes = pl.DataFrame(
        {
            "Date": [date],
            "ValueCode": [value_code],
            "decision_time_ns": [recv_ns],
        }
    )
    labels = MakerFillLabelIndex.from_frame(
        pl.DataFrame(
            {
                "QuoteCode": [value_code],
                "ChannelSeq": pl.Series([17], dtype=pl.UInt64),
                "Bid1_FillSeconds": pl.Series([0.1], dtype=pl.Float32),
                "Bid2_FillSeconds": pl.Series([0.1], dtype=pl.Float32),
            }
        )
    )
    raw_books = build_raw_book_day_index(spot, future, mapping)
    specs = _policy_specs(
        date,
        value_code=value_code,
        quote_code=quote_code,
    )
    return PreparedS1EntryDay(
        date=date,
        policy_ids=POLICY_IDS,
        specs_by_policy=specs,
        common_decisions=common,
        state_changes_by_policy=MappingProxyType(
            {policy_id: changes for policy_id in POLICY_IDS}
        ),
        raw_books=raw_books,
        spot_trades=build_spot_trade_day_index(spot, mapping),
        spot_closes=None,
        makerfill_labels=labels,
        products=(
            S1Product(
                product_id=value_code,
                value_code=value_code,
                quote_code=quote_code,
                contract_size_shares=contract_size,
                future_contracts=1,
                future_session_end_time_ns=session_expiry_ns,
                spot_session_end_time_ns=session_expiry_ns,
                end_date="20260520",
            ),
        ),
        entry_product_ids=(value_code,),
        day_open_time_ns=recv_ns - 1_000_000_000,
        entry_cutoff_time_ns=recv_ns + 10_000_000_000,
        session_expiry_time_ns=session_expiry_ns,
    )


class S1PortfolioRunnerTest(unittest.TestCase):
    def test_single_day_exposes_final_carry_and_committed_checkpoint(self) -> None:
        portfolio = run_s1_portfolio((_prepared(D1),))

        for result in portfolio.policy_results:
            self.assertEqual(len(result.final_carry), 1)
            capacity_id = result.final_carry[0].capacity_id
            self.assertGreater(
                result.capacity_seed.account_balances[capacity_id].paired_open,
                0,
            )
            self.assertEqual(result.cross_ledger_report.establishments, 1)
            self.assertEqual(result.cross_ledger_report.executable_terminals, 0)

    def test_date_major_seven_policy_carry_exit_and_checkpoint_isolation(self) -> None:
        calls: list[tuple[str, str]] = []
        accounting_objects: dict[tuple[str, str], object] = {}
        capacity_objects: dict[tuple[str, str], object] = {}

        def spy_runner(
            prepared: PreparedS1EntryDay,
            policy_id: str,
            **kwargs: object,
        ):
            calls.append((prepared.date, policy_id))
            accounting_objects[(prepared.date, policy_id)] = kwargs[
                "accounting_adapter"
            ]
            capacity_objects[(prepared.date, policy_id)] = kwargs["capacity_ledger"]
            return run_s1_entry_policy(prepared, policy_id, **kwargs)

        portfolio = run_s1_portfolio(
            (
                _prepared(D1),
                _prepared(D2, include_exit_trade=True),
            ),
            policy_runner=spy_runner,
        )

        self.assertEqual(
            calls,
            [
                *((D1, policy_id) for policy_id in POLICY_IDS),
                *((D2, policy_id) for policy_id in POLICY_IDS),
            ],
        )
        self.assertEqual(portfolio.dates, (D1, D2))
        self.assertEqual(tuple(portfolio.by_policy), POLICY_IDS)
        first_accounting_objects = [
            accounting_objects[(D1, policy_id)] for policy_id in POLICY_IDS
        ]
        self.assertEqual(len({id(value) for value in first_accounting_objects}), 7)
        self.assertEqual(
            len(
                {
                    id(capacity_objects[(date, policy_id)])
                    for date in (D1, D2)
                    for policy_id in POLICY_IDS
                }
            ),
            14,
        )
        for policy_id in POLICY_IDS:
            self.assertIs(
                accounting_objects[(D1, policy_id)],
                accounting_objects[(D2, policy_id)],
            )
            self.assertIsNot(
                capacity_objects[(D1, policy_id)],
                capacity_objects[(D2, policy_id)],
            )
            result = portfolio.by_policy[policy_id]
            day_one, day_two = result.daily_runs
            self.assertEqual(len(day_one.result.carry_out), 1)
            self.assertEqual(day_two.result.carry_in, day_one.result.carry_out)
            self.assertEqual(day_two.result.carry_out, ())
            self.assertEqual(result.final_carry, ())
            self.assertEqual(
                day_two.result.entry_disabled_product_ids,
                frozenset({VALUE}),
            )
            self.assertEqual(day_two.result.orders, ())
            self.assertEqual(
                {position.state for position in day_two.result.positions},
                {"exit_maker_flat"},
            )
            self.assertTrue(
                all(fact.scenario_id == policy_id for fact in result.accounting_facts)
            )
            self.assertEqual(
                {
                    fact.execution_date
                    for fact in result.accounting_facts
                    if isinstance(fact, ExecutedLeg)
                },
                {D1, D2},
            )
            self.assertEqual(
                [
                    fact.establishment_date
                    for fact in result.accounting_facts
                    if isinstance(fact, PositionEstablishedFact)
                ],
                [D1],
            )
            self.assertEqual(
                [
                    fact.terminal_date
                    for fact in result.accounting_facts
                    if isinstance(fact, TerminalRealizedAccounting)
                ],
                [D2],
            )
            self.assertEqual(result.cross_ledger_report.establishments, 1)
            self.assertEqual(result.cross_ledger_report.executable_terminals, 1)

            first_delta = day_one.result.capacity_transitions
            second_delta = day_two.result.capacity_transitions
            self.assertGreater(len(first_delta), 0)
            self.assertGreater(len(second_delta), 0)
            self.assertEqual(
                result.capacity_transitions,
                (*first_delta, *second_delta),
            )
            self.assertEqual(first_delta[0].sequence, 1)
            self.assertEqual(
                second_delta[0].sequence,
                first_delta[-1].sequence + 1,
            )
            full = replay_capacity_transitions(result.capacity_transitions)
            restored = CapacityLedger.from_seed(result.capacity_seed).verify()
            self.assertEqual(full.account_balances, restored.account_balances)
            self.assertEqual(full.global_balances, restored.global_balances)
            self.assertEqual(
                full.transition_chain_sha256,
                restored.transition_chain_sha256,
            )
            self.assertEqual(
                full.last_sequence,
                result.capacity_seed.transition_sequence_offset,
            )

    def test_streaming_source_is_date_major_and_releases_each_prepared_day(
        self,
    ) -> None:
        events: list[tuple[str, ...]] = []
        prior_common: ReferenceType[pl.DataFrame] | None = None

        def prepare(date: str) -> PreparedS1EntryDay:
            nonlocal prior_common
            gc.collect()
            if prior_common is not None:
                self.assertIsNone(prior_common())
            events.append(("prepare", date))
            prepared = _prepared(date)
            prior_common = ref(prepared.common_decisions)
            return prepared

        def spy_runner(
            prepared: PreparedS1EntryDay,
            policy_id: str,
            **kwargs: object,
        ):
            events.append(("run", prepared.date, policy_id))
            return run_s1_entry_policy(prepared, policy_id, **kwargs)

        portfolio = run_s1_portfolio_from_dates(
            (D1, D2),
            prepare,
            EXTENDED_CATALOG,
            policy_runner=spy_runner,
            retain_daily_runs=False,
        )

        self.assertEqual(
            events,
            [
                ("prepare", D1),
                *(("run", D1, policy_id) for policy_id in POLICY_IDS),
                ("prepare", D2),
                *(("run", D2, policy_id) for policy_id in POLICY_IDS),
            ],
        )
        self.assertEqual(portfolio.dates, (D1, D2))

    def test_streaming_sink_gets_exact_deltas_and_summaries_survive_drop(
        self,
    ) -> None:
        sunk: list[tuple[str, str, object, tuple[object, ...], tuple[object, ...]]] = []

        def sink(partition: S1PortfolioDayPartition) -> None:
            sunk.append(
                (
                    partition.date,
                    partition.policy_id,
                    partition.run.summary,
                    partition.accounting_facts,
                    partition.capacity_transitions,
                )
            )

        portfolio = run_s1_portfolio_from_dates(
            (D1, D2),
            lambda date: _prepared(date, include_exit_trade=date == D2),
            CATALOG,
            retain_daily_runs=False,
            sink=sink,
        )

        self.assertEqual(
            [(date, policy_id) for date, policy_id, *_rest in sunk],
            [
                *((D1, policy_id) for policy_id in POLICY_IDS),
                *((D2, policy_id) for policy_id in POLICY_IDS),
            ],
        )
        for policy_id in POLICY_IDS:
            result = portfolio.by_policy[policy_id]
            self.assertFalse(result.daily_runs_retained)
            self.assertEqual(result.daily_runs, ())
            self.assertEqual(
                tuple(summary.date for summary in result.daily_summaries),
                (D1, D2),
            )
            selected = [row for row in sunk if row[1] == policy_id]
            self.assertEqual(
                tuple(fact for row in selected for fact in row[3]),
                result.accounting_facts,
            )
            self.assertEqual(
                tuple(transition for row in selected for transition in row[4]),
                result.capacity_transitions,
            )
            self.assertEqual(result.cross_ledger_report.executable_terminals, 1)

    def test_streaming_catalog_subset_and_identity_are_fail_closed(self) -> None:
        calls: list[tuple[str, str]] = []

        def should_not_run(
            prepared: PreparedS1EntryDay,
            policy_id: str,
            **_kwargs: object,
        ) -> object:
            calls.append((prepared.date, policy_id))
            raise AssertionError("catalog validation must precede daily replay")

        with (
            self.subTest("unknown product"),
            self.assertRaisesRegex(ValueError, "absent from frozen"),
        ):
            run_s1_portfolio_from_dates(
                (D1,),
                lambda date: _prepared(
                    date,
                    value_code="2603",
                    quote_code="CZFE6",
                ),
                CATALOG,
                policy_runner=should_not_run,
            )
        with (
            self.subTest("contract drift"),
            self.assertRaisesRegex(ValueError, "identity/contract"),
        ):
            run_s1_portfolio_from_dates(
                (D1,),
                lambda date: _prepared(date, contract_size=1_000),
                CATALOG,
                policy_runner=should_not_run,
            )
        self.assertEqual(calls, [])

    def test_streaming_dates_and_callback_identity_are_strict(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            run_s1_portfolio_from_dates(
                (D2, D1),
                _prepared,
                CATALOG,
            )
        with self.assertRaisesRegex(ValueError, "different date"):
            run_s1_portfolio_from_dates(
                (D1,),
                lambda _date: _prepared(D2),
                CATALOG,
            )

    def test_carry_product_missing_from_later_daily_mapping_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "absent from daily mapping"):
            run_s1_portfolio(
                (
                    _prepared(D1),
                    _prepared(D2, value_code="2603", quote_code="CZFE6"),
                )
            )

    def test_carried_quote_change_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "contract/quote mapping changed"):
            run_s1_portfolio(
                (
                    _prepared(D1),
                    _prepared(D2, quote_code="CDFM6"),
                )
            )

    def test_contract_size_drift_fails_before_any_policy_runs(self) -> None:
        calls: list[tuple[str, str]] = []

        def should_not_run(
            prepared: PreparedS1EntryDay,
            policy_id: str,
            **_kwargs: object,
        ):
            calls.append((prepared.date, policy_id))
            raise AssertionError("preflight validation should run first")

        with self.assertRaisesRegex(ValueError, "contract size changed"):
            run_s1_portfolio(
                (
                    _prepared(D1),
                    _prepared(D2, contract_size=1_000),
                ),
                policy_runner=should_not_run,
            )
        self.assertEqual(calls, [])

    def test_days_must_be_strictly_increasing_and_complete(self) -> None:
        first = _prepared(D1)
        second = _prepared(D2)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            run_s1_portfolio((second, first))
        with self.assertRaisesRegex(ValueError, "all seven policies"):
            run_s1_portfolio(
                (
                    PreparedS1EntryDay(
                        date=first.date,
                        policy_ids=("fixed20",),
                        specs_by_policy=MappingProxyType(
                            {"fixed20": first.specs_by_policy["fixed20"]}
                        ),
                        common_decisions=first.common_decisions,
                        state_changes_by_policy=MappingProxyType(
                            {"fixed20": first.state_changes_by_policy["fixed20"]}
                        ),
                        raw_books=first.raw_books,
                        spot_trades=first.spot_trades,
                        spot_closes=first.spot_closes,
                        makerfill_labels=first.makerfill_labels,
                        products=first.products,
                        entry_product_ids=first.entry_product_ids,
                        day_open_time_ns=first.day_open_time_ns,
                        entry_cutoff_time_ns=first.entry_cutoff_time_ns,
                        session_expiry_time_ns=first.session_expiry_time_ns,
                    ),
                )
            )


if __name__ == "__main__":
    unittest.main()
