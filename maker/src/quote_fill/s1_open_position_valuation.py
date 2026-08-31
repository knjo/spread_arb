"""Pure common-horizon executable valuation for open S1 paired positions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import date, datetime
from decimal import Decimal
from typing import Final
from zoneinfo import ZoneInfo

from .layered import EventCursor
from .s1_accounting import (
    AccountingFact,
    ExecutedLeg,
    PositionEstablishedFact,
    S1OpenInventorySnapshot,
    replay_accounting_facts,
)
from .s1_hedge import CausalBookState, RawBookCursor, executable_book
from .s1_publication_gate import S1OpenPositionValuation
from .transaction_costs import TransactionCostProfile

COMMON_HORIZON_DATE: Final = "20260813"
COMMON_HORIZON_TIME: Final = "13:20:00"
COMMON_HORIZON_EVENT_SEQUENCE: Final = 600
VALUATION_METHOD_ID: Final = "s1_causal_executable_fifo_common_horizon_v1"
VALUATION_RECORD_TYPE: Final = "s1_open_position_valuation"
VALUATION_RECORD_SCHEMA_VERSION: Final = 1
_TAIPEI: Final = ZoneInfo("Asia/Taipei")
_MONEY_TOLERANCE: Final = Decimal("0.000001")


def _common_horizon_recv_time_ns() -> int:
    value = datetime(2026, 8, 13, 13, 20, tzinfo=_TAIPEI)
    return int(value.timestamp()) * 1_000_000_000


COMMON_HORIZON_CURSOR: Final = EventCursor(
    _common_horizon_recv_time_ns(),
    COMMON_HORIZON_EVENT_SEQUENCE,
    0,
)
COMMON_HORIZON_ASOF_ID: Final = (
    "20260813T132000+0800/"
    f"{COMMON_HORIZON_CURSOR.recv_time_ns}/"
    f"{COMMON_HORIZON_CURSOR.event_sequence}/"
    f"{COMMON_HORIZON_CURSOR.row_index}"
)


class S1OpenPositionValuationError(ValueError):
    """Accounting or market inputs cannot produce a valid valuation record."""


@dataclass(frozen=True, slots=True)
class S1CommonHorizonBooks:
    """Latest causal Spot and Future states as-of the common cursor."""

    spot: CausalBookState | None
    future: CausalBookState | None

    def __post_init__(self) -> None:
        for name in ("spot", "future"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, CausalBookState):
                raise TypeError(f"{name} must be CausalBookState or None")


@dataclass(frozen=True, slots=True)
class S1OpenPositionValuationRecord:
    """Persistable per-position valuation or explicit unpriced outcome."""

    scenario_id: str
    position_id: str
    value_code: str
    capacity_id: str
    establishment_date: str
    valuation_method_id: str
    valuation_asof_id: str
    horizon_cursor: EventCursor
    spot_book_cursor: RawBookCursor | None
    future_book_cursor: RawBookCursor | None
    priced: bool
    unpriced_reason: str | None
    spot_shares: int
    future_contracts: int
    future_share_equivalent: int
    spot_exit_vwap: Decimal | None
    future_exit_vwap: Decimal | None
    prior_spot_cashflow_twd: Decimal | None
    prior_futures_realized_pnl_twd: Decimal | None
    spot_liquidation_cashflow_twd: Decimal | None
    futures_unrealized_pnl_twd: Decimal | None
    gross_mark_pnl_twd: Decimal | None
    incurred_commission_twd: Decimal
    incurred_tax_twd: Decimal
    remaining_spot_commission_twd: Decimal | None
    remaining_spot_sell_tax_twd: Decimal | None
    remaining_futures_commission_twd: Decimal | None
    remaining_futures_tax_twd: Decimal | None
    remaining_exit_cost_twd: Decimal | None
    net_mark_pnl_twd: Decimal | None

    def __post_init__(self) -> None:
        for name in (
            "scenario_id",
            "position_id",
            "value_code",
            "capacity_id",
            "valuation_method_id",
            "valuation_asof_id",
        ):
            _canonical_text(getattr(self, name), name)
        _valid_date(self.establishment_date, "establishment_date")
        if not isinstance(self.horizon_cursor, EventCursor):
            raise TypeError("horizon_cursor must be EventCursor")
        if self.horizon_cursor != COMMON_HORIZON_CURSOR:
            raise S1OpenPositionValuationError("horizon cursor is not the S1 common horizon")
        for name in ("spot_book_cursor", "future_book_cursor"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, RawBookCursor):
                raise TypeError(f"{name} must be RawBookCursor or None")
        if not isinstance(self.priced, bool):
            raise TypeError("priced must be boolean")
        for name in ("spot_shares", "future_contracts", "future_share_equivalent"):
            _nonnegative_int(getattr(self, name), name)
        _nonnegative_money(self.incurred_commission_twd, "incurred_commission_twd")
        _nonnegative_money(self.incurred_tax_twd, "incurred_tax_twd")
        optional_money = (
            "spot_exit_vwap",
            "future_exit_vwap",
            "prior_spot_cashflow_twd",
            "prior_futures_realized_pnl_twd",
            "spot_liquidation_cashflow_twd",
            "futures_unrealized_pnl_twd",
            "gross_mark_pnl_twd",
            "remaining_spot_commission_twd",
            "remaining_spot_sell_tax_twd",
            "remaining_futures_commission_twd",
            "remaining_futures_tax_twd",
            "remaining_exit_cost_twd",
            "net_mark_pnl_twd",
        )
        for name in optional_money:
            value = getattr(self, name)
            if value is not None:
                _finite_decimal(value, name)
        if self.priced:
            if self.unpriced_reason is not None:
                raise S1OpenPositionValuationError(
                    "priced valuation cannot have an unpriced reason"
                )
            if any(getattr(self, name) is None for name in optional_money):
                raise S1OpenPositionValuationError(
                    "priced valuation requires every financial field"
                )
            if min(
                self.spot_shares,
                self.future_contracts,
                self.future_share_equivalent,
            ) <= 0:
                raise S1OpenPositionValuationError(
                    "priced valuation requires positive paired quantities"
                )
            if any(
                cursor is not None and cursor.cursor > self.horizon_cursor
                for cursor in (self.spot_book_cursor, self.future_book_cursor)
            ):
                raise S1OpenPositionValuationError(
                    "priced valuation book follows the common horizon"
                )
            self._validate_conservation()
        else:
            _canonical_text(self.unpriced_reason, "unpriced_reason")
            if any(getattr(self, name) is not None for name in optional_money):
                raise S1OpenPositionValuationError(
                    "unpriced valuation cannot contain financial fields"
                )

    def _validate_conservation(self) -> None:
        assert self.gross_mark_pnl_twd is not None
        assert self.remaining_exit_cost_twd is not None
        assert self.net_mark_pnl_twd is not None
        assert self.remaining_spot_commission_twd is not None
        assert self.remaining_spot_sell_tax_twd is not None
        assert self.remaining_futures_commission_twd is not None
        assert self.remaining_futures_tax_twd is not None
        remaining = (
            self.remaining_spot_commission_twd
            + self.remaining_spot_sell_tax_twd
            + self.remaining_futures_commission_twd
            + self.remaining_futures_tax_twd
        )
        if abs(remaining - self.remaining_exit_cost_twd) > _MONEY_TOLERANCE:
            raise S1OpenPositionValuationError("remaining exit costs do not conserve")
        expected_net = (
            self.gross_mark_pnl_twd
            - self.incurred_commission_twd
            - self.incurred_tax_twd
            - self.remaining_exit_cost_twd
        )
        if abs(expected_net - self.net_mark_pnl_twd) > _MONEY_TOLERANCE:
            raise S1OpenPositionValuationError("gross-cost-net does not conserve")


@dataclass(frozen=True, slots=True)
class S1CommonHorizonValuationResult:
    scenario_id: str
    rows: tuple[S1OpenPositionValuationRecord, ...]
    incurred_open_execution_cost_twd: Decimal
    aggregate: S1OpenPositionValuation | None

    def __post_init__(self) -> None:
        _canonical_text(self.scenario_id, "scenario_id")
        if not isinstance(self.rows, tuple) or any(
            not isinstance(row, S1OpenPositionValuationRecord) for row in self.rows
        ):
            raise TypeError("rows must be valuation-record tuple")
        if any(row.scenario_id != self.scenario_id for row in self.rows):
            raise S1OpenPositionValuationError("result contains another scenario")
        expected_incurred = sum(
            (
                row.incurred_commission_twd + row.incurred_tax_twd
                for row in self.rows
            ),
            start=Decimal(0),
        )
        if expected_incurred != self.incurred_open_execution_cost_twd:
            raise S1OpenPositionValuationError("result incurred costs do not conserve")
        if any(not row.priced for row in self.rows):
            if self.aggregate is not None:
                raise S1OpenPositionValuationError(
                    "unpriced row forbids a publishable aggregate"
                )
            return
        expected = _aggregate(self.rows)
        if self.aggregate != expected:
            raise S1OpenPositionValuationError("result aggregate does not conserve")

    @property
    def publishable(self) -> bool:
        return self.aggregate is not None


@dataclass(frozen=True, slots=True)
class _OpenPositionInput:
    establishment: PositionEstablishedFact
    snapshot: S1OpenInventorySnapshot
    executions: tuple[ExecutedLeg, ...]


def value_s1_common_horizon_open_positions(
    facts: Sequence[AccountingFact],
    *,
    scenario_id: str,
    books_by_value_code: Mapping[str, S1CommonHorizonBooks],
    profile: TransactionCostProfile | None = None,
) -> S1CommonHorizonValuationResult:
    """Mark final unsealed inventory with books frozen at the S1 horizon.

    The complete accounting stream determines which positions remain open;
    only the market-book pricing cursor is frozen at 2026-08-13 13:20.
    """

    _canonical_text(scenario_id, "scenario_id")
    if not isinstance(books_by_value_code, Mapping):
        raise TypeError("books_by_value_code must be a mapping")
    for key, value in books_by_value_code.items():
        _canonical_text(key, "books_by_value_code key")
        if not isinstance(value, S1CommonHorizonBooks):
            raise TypeError("books_by_value_code values must be S1CommonHorizonBooks")
    cost_profile = profile or TransactionCostProfile()
    cost_profile.validate()
    ledger = replay_accounting_facts(tuple(facts), profile=cost_profile)
    terminal_ids = {
        row.position_id for row in (*ledger.terminal_rows, *ledger.expiry_marks)
    }
    establishments = tuple(
        row
        for row in ledger.establishment_rows
        if row.scenario_id == scenario_id and row.position_id not in terminal_ids
    )
    inputs = tuple(
        _OpenPositionInput(
            establishment=establishment,
            snapshot=ledger.open_inventory_snapshot(establishment.position_id),
            executions=tuple(
                row
                for row in ledger.rows
                if row.position_id == establishment.position_id
            ),
        )
        for establishment in establishments
    )
    by_value_code: dict[str, list[_OpenPositionInput]] = {}
    for value in inputs:
        by_value_code.setdefault(value.establishment.value_code, []).append(value)
    rows_by_position: dict[str, S1OpenPositionValuationRecord] = {}
    for value_code, group in by_value_code.items():
        for row in _value_product_group(
            tuple(group),
            books=books_by_value_code.get(value_code),
            profile=cost_profile,
        ):
            rows_by_position[row.position_id] = row
    rows = tuple(
        rows_by_position[establishment.position_id]
        for establishment in establishments
    )
    incurred = sum(
        (row.incurred_commission_twd + row.incurred_tax_twd for row in rows),
        start=Decimal(0),
    )
    aggregate = _aggregate(rows)
    return S1CommonHorizonValuationResult(
        scenario_id=scenario_id,
        rows=rows,
        incurred_open_execution_cost_twd=incurred,
        aggregate=aggregate,
    )


def _value_product_group(
    inputs: tuple[_OpenPositionInput, ...],
    *,
    books: S1CommonHorizonBooks | None,
    profile: TransactionCostProfile,
) -> tuple[S1OpenPositionValuationRecord, ...]:
    commons = tuple(_position_common(value) for value in inputs)
    if any(not _snapshot_is_exact_pair(value.snapshot) for value in inputs):
        return tuple(
            _unpriced(common, books, "inventory_not_exactly_paired")
            for common in commons
        )
    if books is None:
        return tuple(
            _unpriced(common, None, "missing_spot_and_future_books")
            for common in commons
        )
    aggregate_spot_shares = sum(
        int(common["spot_shares"]) for common in commons
    )
    aggregate_future_contracts = sum(
        int(common["future_contracts"]) for common in commons
    )
    spot_execution, spot_reason = _executable_at_horizon(
        books.spot,
        side="sell",
        quantity=aggregate_spot_shares,
        quantity_unit="spot_shares",
        venue="spot",
    )
    if spot_execution is None:
        return tuple(
            _unpriced(common, books, spot_reason or "spot_book_not_executable")
            for common in commons
        )
    future_execution, future_reason = _executable_at_horizon(
        books.future,
        side="buy",
        quantity=aggregate_future_contracts,
        quantity_unit="future_contracts",
        venue="future",
    )
    if future_execution is None:
        return tuple(
            _unpriced(common, books, future_reason or "future_book_not_executable")
            for common in commons
        )
    spot_exit = _decimal(spot_execution.executable_vwap)
    future_exit = _decimal(future_execution.executable_vwap)
    return tuple(
        _value_position_at_common_vwap(
            value,
            common=common,
            books=books,
            profile=profile,
            spot_exit=spot_exit,
            future_exit=future_exit,
        )
        for value, common in zip(inputs, commons, strict=True)
    )


def _position_common(value: _OpenPositionInput) -> dict[str, object]:
    establishment = value.establishment
    snapshot = value.snapshot
    executions = value.executions
    incurred_commission = _sum_decimal(row.commission_twd for row in executions)
    incurred_tax = _sum_decimal(row.tax_twd for row in executions)
    spot_shares = sum(lot.shares for lot in snapshot.spot_lots)
    future_contracts = sum(lot.contracts for lot in snapshot.future_lots)
    future_equivalent = sum(lot.share_equivalent for lot in snapshot.future_lots)
    return {
        "scenario_id": establishment.scenario_id,
        "position_id": establishment.position_id,
        "value_code": establishment.value_code,
        "capacity_id": establishment.capacity_id,
        "establishment_date": establishment.establishment_date,
        "valuation_method_id": VALUATION_METHOD_ID,
        "valuation_asof_id": COMMON_HORIZON_ASOF_ID,
        "horizon_cursor": COMMON_HORIZON_CURSOR,
        "spot_shares": spot_shares,
        "future_contracts": future_contracts,
        "future_share_equivalent": future_equivalent,
        "incurred_commission_twd": incurred_commission,
        "incurred_tax_twd": incurred_tax,
    }


def _snapshot_is_exact_pair(snapshot: S1OpenInventorySnapshot) -> bool:
    spot_shares = sum(lot.shares for lot in snapshot.spot_lots)
    future_contracts = sum(lot.contracts for lot in snapshot.future_lots)
    future_equivalent = sum(lot.share_equivalent for lot in snapshot.future_lots)
    return not (
        spot_shares <= 0
        or future_contracts <= 0
        or future_equivalent <= 0
        or spot_shares != future_equivalent
        or any(lot.side != "sell" for lot in snapshot.future_lots)
    )


def _value_position_at_common_vwap(
    value: _OpenPositionInput,
    *,
    common: Mapping[str, object],
    books: S1CommonHorizonBooks,
    profile: TransactionCostProfile,
    spot_exit: Decimal,
    future_exit: Decimal,
) -> S1OpenPositionValuationRecord:
    snapshot = value.snapshot
    executions = value.executions
    spot_shares = sum(lot.shares for lot in snapshot.spot_lots)
    future_contracts = sum(lot.contracts for lot in snapshot.future_lots)
    future_equivalent = sum(lot.share_equivalent for lot in snapshot.future_lots)
    incurred_commission = _sum_decimal(row.commission_twd for row in executions)
    incurred_tax = _sum_decimal(row.tax_twd for row in executions)
    prior_spot_cashflow = _sum_decimal(
        row.signed_cashflow_twd for row in executions if row.market == "spot"
    )
    prior_future_realized = _sum_decimal(
        row.realized_pnl_twd for row in executions if row.market == "future"
    )
    spot_liquidation = spot_exit * spot_shares
    future_unrealized = sum(
        (
            (_decimal(lot.opening_price) - future_exit) * lot.share_equivalent
            for lot in snapshot.future_lots
        ),
        start=Decimal(0),
    )
    gross = (
        prior_spot_cashflow
        + prior_future_realized
        + spot_liquidation
        + future_unrealized
    )
    remaining_spot_commission = _decimal(
        profile.spot_commission_twd(float(spot_exit), spot_shares)
    )
    remaining_spot_tax = _sum_decimal(
        profile.spot_sell_tax_twd(
            float(spot_exit),
            lot.shares,
            same_day=lot.acquisition_date == COMMON_HORIZON_DATE,
        )
        for lot in snapshot.spot_lots
    )
    remaining_future_commission = _decimal(
        profile.futures_commission_twd(future_contracts)
    )
    remaining_future_tax = _decimal(
        profile.futures_tax_twd(float(future_exit), future_equivalent)
    )
    remaining = (
        remaining_spot_commission
        + remaining_spot_tax
        + remaining_future_commission
        + remaining_future_tax
    )
    net = gross - incurred_commission - incurred_tax - remaining
    return S1OpenPositionValuationRecord(
        **common,
        spot_book_cursor=books.spot.book_cursor,
        future_book_cursor=books.future.book_cursor,
        priced=True,
        unpriced_reason=None,
        spot_exit_vwap=spot_exit,
        future_exit_vwap=future_exit,
        prior_spot_cashflow_twd=prior_spot_cashflow,
        prior_futures_realized_pnl_twd=prior_future_realized,
        spot_liquidation_cashflow_twd=spot_liquidation,
        futures_unrealized_pnl_twd=future_unrealized,
        gross_mark_pnl_twd=gross,
        remaining_spot_commission_twd=remaining_spot_commission,
        remaining_spot_sell_tax_twd=remaining_spot_tax,
        remaining_futures_commission_twd=remaining_future_commission,
        remaining_futures_tax_twd=remaining_future_tax,
        remaining_exit_cost_twd=remaining,
        net_mark_pnl_twd=net,
    )


def _unpriced(
    common: Mapping[str, object],
    books: S1CommonHorizonBooks | None,
    reason: str,
) -> S1OpenPositionValuationRecord:
    return S1OpenPositionValuationRecord(
        **common,
        spot_book_cursor=(None if books is None or books.spot is None else books.spot.book_cursor),
        future_book_cursor=(
            None if books is None or books.future is None else books.future.book_cursor
        ),
        priced=False,
        unpriced_reason=reason,
        spot_exit_vwap=None,
        future_exit_vwap=None,
        prior_spot_cashflow_twd=None,
        prior_futures_realized_pnl_twd=None,
        spot_liquidation_cashflow_twd=None,
        futures_unrealized_pnl_twd=None,
        gross_mark_pnl_twd=None,
        remaining_spot_commission_twd=None,
        remaining_spot_sell_tax_twd=None,
        remaining_futures_commission_twd=None,
        remaining_futures_tax_twd=None,
        remaining_exit_cost_twd=None,
        net_mark_pnl_twd=None,
    )


def _executable_at_horizon(
    state: CausalBookState | None,
    *,
    side: str,
    quantity: int,
    quantity_unit: str,
    venue: str,
):
    if state is None:
        return None, f"missing_{venue}_book"
    if state.book_cursor.cursor > COMMON_HORIZON_CURSOR:
        return None, f"{venue}_book_after_common_horizon"
    execution, reason = executable_book(
        state,
        side=side,
        quantity=quantity,
        quantity_unit=quantity_unit,
        send_eligible_cursor=COMMON_HORIZON_CURSOR,
    )
    return execution, None if reason is None else f"{venue}_{reason}"


def _aggregate(
    rows: tuple[S1OpenPositionValuationRecord, ...],
) -> S1OpenPositionValuation | None:
    if any(not row.priced for row in rows):
        return None
    gross = sum((row.gross_mark_pnl_twd for row in rows), start=Decimal(0))
    remaining = sum((row.remaining_exit_cost_twd for row in rows), start=Decimal(0))
    net = sum((row.net_mark_pnl_twd for row in rows), start=Decimal(0))
    assert all(value is not None for value in (gross, remaining, net))
    return S1OpenPositionValuation(
        position_count=len(rows),
        valuation_method_id=VALUATION_METHOD_ID,
        valuation_asof_id=COMMON_HORIZON_ASOF_ID,
        comparable_across_scenarios=True,
        gross_mark_pnl_twd=gross,
        remaining_exit_cost_twd=remaining,
        net_mark_pnl_twd=net,
    )


_MONEY_FIELDS: Final = frozenset(
    {
        field.name
        for field in fields(S1OpenPositionValuationRecord)
        if field.name
        not in {
            "scenario_id",
            "position_id",
            "value_code",
            "capacity_id",
            "establishment_date",
            "valuation_method_id",
            "valuation_asof_id",
            "horizon_cursor",
            "spot_book_cursor",
            "future_book_cursor",
            "priced",
            "unpriced_reason",
            "spot_shares",
            "future_contracts",
            "future_share_equivalent",
        }
    }
)
_CURSOR_NAMES: Final = ("recv_time_ns", "event_sequence", "row_index")


def encode_s1_open_position_valuation(
    row: S1OpenPositionValuationRecord,
) -> dict[str, object]:
    if not isinstance(row, S1OpenPositionValuationRecord):
        raise TypeError("row must be S1OpenPositionValuationRecord")
    record: dict[str, object] = {
        "record_type": VALUATION_RECORD_TYPE,
        "schema_version": VALUATION_RECORD_SCHEMA_VERSION,
    }
    for field in fields(row):
        name = field.name
        value = getattr(row, name)
        if name == "horizon_cursor":
            _encode_cursor(record, "horizon", value)
        elif name in ("spot_book_cursor", "future_book_cursor"):
            prefix = name.removesuffix("_cursor")
            _encode_raw_book_cursor(record, prefix, value)
        elif name in _MONEY_FIELDS:
            record[name] = None if value is None else str(value)
        else:
            record[name] = value
    return record


def decode_s1_open_position_valuation(
    record: Mapping[str, object],
) -> S1OpenPositionValuationRecord:
    if not isinstance(record, Mapping):
        raise TypeError("valuation record must be a mapping")
    expected = _valuation_record_keys()
    actual = set(record)
    if actual != expected:
        raise S1OpenPositionValuationError(
            "valuation record schema mismatch: "
            f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
        )
    if record["record_type"] != VALUATION_RECORD_TYPE:
        raise S1OpenPositionValuationError("valuation record_type is invalid")
    if type(record["schema_version"]) is not int or record["schema_version"] != 1:
        raise S1OpenPositionValuationError("valuation schema_version is unsupported")
    values: dict[str, object] = {}
    for field in fields(S1OpenPositionValuationRecord):
        name = field.name
        if name == "horizon_cursor":
            values[name] = _decode_cursor(record, "horizon")
        elif name in ("spot_book_cursor", "future_book_cursor"):
            values[name] = _decode_raw_book_cursor(
                record,
                name.removesuffix("_cursor"),
            )
        elif name in _MONEY_FIELDS:
            values[name] = _decode_decimal(record[name], name)
        else:
            values[name] = record[name]
    try:
        return S1OpenPositionValuationRecord(**values)
    except (TypeError, ValueError) as error:
        if isinstance(error, S1OpenPositionValuationError):
            raise
        raise S1OpenPositionValuationError(str(error)) from error


def encode_s1_open_position_valuations(
    rows: Sequence[S1OpenPositionValuationRecord],
) -> list[dict[str, object]]:
    return [encode_s1_open_position_valuation(row) for row in rows]


def decode_s1_open_position_valuations(
    records: Sequence[Mapping[str, object]],
) -> tuple[S1OpenPositionValuationRecord, ...]:
    return tuple(decode_s1_open_position_valuation(record) for record in records)


def _valuation_record_keys() -> set[str]:
    result = {"record_type", "schema_version"}
    for field in fields(S1OpenPositionValuationRecord):
        name = field.name
        if name == "horizon_cursor":
            result.update(f"horizon_{part}" for part in _CURSOR_NAMES)
        elif name in ("spot_book_cursor", "future_book_cursor"):
            prefix = name.removesuffix("_cursor")
            result.update(f"{prefix}_{part}" for part in _CURSOR_NAMES)
            result.add(f"{prefix}_packet_sequence")
        else:
            result.add(name)
    return result


def _encode_cursor(
    record: dict[str, object],
    prefix: str,
    cursor: EventCursor,
) -> None:
    for name in _CURSOR_NAMES:
        record[f"{prefix}_{name}"] = getattr(cursor, name)


def _encode_raw_book_cursor(
    record: dict[str, object],
    prefix: str,
    cursor: RawBookCursor | None,
) -> None:
    for name in _CURSOR_NAMES:
        record[f"{prefix}_{name}"] = (
            None if cursor is None else getattr(cursor.cursor, name)
        )
    record[f"{prefix}_packet_sequence"] = (
        None if cursor is None else cursor.packet_sequence
    )


def _decode_cursor(record: Mapping[str, object], prefix: str) -> EventCursor:
    values = tuple(record[f"{prefix}_{name}"] for name in _CURSOR_NAMES)
    if any(type(value) is not int or value < 0 for value in values):
        raise S1OpenPositionValuationError(f"{prefix} cursor fields are invalid")
    return EventCursor(*values)


def _decode_raw_book_cursor(
    record: Mapping[str, object],
    prefix: str,
) -> RawBookCursor | None:
    values = tuple(record[f"{prefix}_{name}"] for name in _CURSOR_NAMES)
    packet = record[f"{prefix}_packet_sequence"]
    if all(value is None for value in (*values, packet)):
        return None
    if any(type(value) is not int or value < 0 for value in (*values, packet)):
        raise S1OpenPositionValuationError(
            f"{prefix} raw-book cursor fields are invalid"
        )
    return RawBookCursor(EventCursor(*values), packet)


def _decode_decimal(value: object, name: str) -> Decimal | None:
    if value is None:
        return None
    if type(value) is not str:
        raise S1OpenPositionValuationError(f"{name} must be decimal text or null")
    try:
        result = Decimal(value)
    except Exception as error:
        raise S1OpenPositionValuationError(f"{name} is invalid decimal text") from error
    if not result.is_finite() or str(result) != value:
        raise S1OpenPositionValuationError(f"{name} is not canonical decimal text")
    return result


def _sum_decimal(values) -> Decimal:
    return sum((_decimal(value) for value in values), start=Decimal(0))


def _decimal(value: object) -> Decimal:
    return Decimal(str(value))


def _canonical_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise S1OpenPositionValuationError(f"{name} must be canonical text")
    return value


def _valid_date(value: object, name: str) -> str:
    _canonical_text(value, name)
    try:
        parsed = date.fromisoformat(f"{value[:4]}-{value[4:6]}-{value[6:]}")
    except ValueError as error:
        raise S1OpenPositionValuationError(f"{name} must be YYYYMMDD") from error
    if parsed.strftime("%Y%m%d") != value:
        raise S1OpenPositionValuationError(f"{name} must be YYYYMMDD")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise S1OpenPositionValuationError(f"{name} must be non-negative integer")
    return value


def _finite_decimal(value: object, name: str) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise S1OpenPositionValuationError(f"{name} must be finite Decimal")
    return value


def _nonnegative_money(value: object, name: str) -> Decimal:
    result = _finite_decimal(value, name)
    if result < 0:
        raise S1OpenPositionValuationError(f"{name} must be non-negative")
    return result


__all__ = [
    "COMMON_HORIZON_ASOF_ID",
    "COMMON_HORIZON_CURSOR",
    "COMMON_HORIZON_DATE",
    "COMMON_HORIZON_TIME",
    "VALUATION_METHOD_ID",
    "S1CommonHorizonBooks",
    "S1CommonHorizonValuationResult",
    "S1OpenPositionValuationError",
    "S1OpenPositionValuationRecord",
    "decode_s1_open_position_valuation",
    "decode_s1_open_position_valuations",
    "encode_s1_open_position_valuation",
    "encode_s1_open_position_valuations",
    "value_s1_common_horizon_open_positions",
]
