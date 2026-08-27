"""Production preparation and replay for one S1 Spot-Bid entry day.

This runner is the file/data boundary around :class:`S1EventLoop`.  A prepared
day owns the common causal decision table, sparse states for the requested
policies, one shared compact raw-book index, one shared physical spot-trade
index, and one shared makerFill label index.  Seven policies can therefore
reuse physical market data without replaying nine million raw changes as
Python events seven times.

Entry-only replay remains the default.  Normal Spot-Ask-maker exit is an
explicit opt-in and is always wired to the exact physical trade index and the
replayable accounting bridge.
"""

from __future__ import annotations

import gc
import math
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Final

import polars as pl

from ..common.paths import HFT_DATA_ROOT, MAKER_ROOT
from .capacity_ledger import CapacityLedger
from .layered import EventCursor
from .makerfill_adapter import MakerFillLabelIndex, MakerFillScalarEvent
from .policy_spec import (
    DEFAULT_CONVERGENCE_PATH,
    DEFAULT_ENTRY_LOOKUP_PATH,
    DEFAULT_MOTHER_PATH,
    POLICY_IDS,
    PolicySpec,
    build_policy_spec_table,
)
from .s1_accounting_bridge import S1AccountingBridge, S1AccountingProduct
from .s1_day_state import (
    ENTRY_STOP_SECOND,
    SESSION_END_SECOND,
    SESSION_START_SECOND,
    build_s1_policy_state_changes,
    materialize_s1_common_day,
)
from .s1_entry_state_adapter import S1EntryStateAdapter, merge_entry_events
from .s1_event_loop import (
    ContractExpiry,
    EntryCutoff,
    S1CarryContractBinding,
    S1CarryPosition,
    S1EventLoop,
    S1LoopConfig,
    S1Product,
    S1ReplayResult,
    SessionExpiry,
)
from .s1_hedge import HEDGE_DELAY_NS
from .s1_makerfill_bridge import S1MakerFillBridge
from .s1_raw_book_adapter import (
    RawBookDayIndex,
    build_raw_book_day_index_from_scans,
)
from .s1_raw_entry_provider import S1RawEntryBookProvider
from .s1_spot_close_adapter import SpotCloseDayIndex
from .s1_spot_trade_adapter import (
    SpotTradeDayIndex,
    build_spot_trade_day_index_from_scan,
)

RUNNER_VERSION: Final = "s1_entry_day_joint_clock_v2_multiday_expiry"
DEVELOPMENT_END_DATE: Final = "20260813"
ONE_SECOND_NS: Final = 1_000_000_000
SPOT_CLOSE_DELAY_SECONDS: Final = 600
DEFAULT_DAILY_ROOT: Final = MAKER_ROOT / "data" / "walkforward" / "daily"
DEFAULT_MAKERFILL_ROOT: Final = HFT_DATA_ROOT / "makerFill"
DEFAULT_SPOT_TICK_ROOT: Final = HFT_DATA_ROOT / "tickData"
DEFAULT_FUTURE_TICK_ROOT: Final = Path("/mnt/NAS/Parquet/Ticks")

CAUSAL_INPUT_COLUMNS: Final = (
    "Date",
    "ValueCode",
    "QuoteCode",
    "timestamp",
    "seconds_from_open",
    "spot_recv_time",
    "spot_sequence",
    "spot_ref_price",
    "fut_ref_price",
    "contract_size",
    "end_date",
    "spot_bid",
    "spot_ask",
    "fut_exec_bid",
    "fut_exec_ask",
    "analysis_eligible",
    "basis_mid_bp",
    "eligible_base",
)

ENTRY_DECISION_COLUMNS: Final = (
    "Date",
    "ValueCode",
    "QuoteCode",
    "decision_time_ns",
    "analysis_eligible",
    "selected_anchor_bp",
    "contract_size",
)


@dataclass(frozen=True, slots=True)
class S1EntryRunnerPaths:
    daily_root: Path = DEFAULT_DAILY_ROOT
    makerfill_root: Path = DEFAULT_MAKERFILL_ROOT
    spot_tick_root: Path = DEFAULT_SPOT_TICK_ROOT
    future_tick_root: Path = DEFAULT_FUTURE_TICK_ROOT
    mother_path: Path = DEFAULT_MOTHER_PATH
    entry_lookup_path: Path = DEFAULT_ENTRY_LOOKUP_PATH
    convergence_path: Path = DEFAULT_CONVERGENCE_PATH

    def causal_path(self, date: str) -> Path:
        return self.daily_root / f"Date={date}" / "causal_fair.parquet"

    def makerfill_path(self, date: str) -> Path:
        return self.makerfill_root / f"{date}_makerFill.parquet"

    def mapping_path(self, date: str) -> Path:
        return self.daily_root / f"Date={date}" / "mapping.parquet"

    def contracts_path(self, date: str) -> Path:
        return self.daily_root / "metadata" / f"{date}_contracts.parquet"

    def spot_raw_path(self, date: str) -> Path:
        return self.spot_tick_root / f"{date}_StockTick.parquet"

    def future_raw_path(self, date: str) -> Path:
        return (
            self.future_tick_root
            / date[:4]
            / date[4:6]
            / date[6:8]
            / "stock_futures.parquet"
        )


@dataclass(frozen=True, slots=True)
class PreparedS1EntryDay:
    date: str
    policy_ids: tuple[str, ...]
    specs_by_policy: Mapping[str, tuple[PolicySpec, ...]] = field(repr=False)
    common_decisions: pl.DataFrame = field(repr=False)
    state_changes_by_policy: Mapping[str, pl.DataFrame] = field(repr=False)
    raw_books: RawBookDayIndex = field(repr=False)
    spot_trades: SpotTradeDayIndex = field(repr=False)
    spot_closes: SpotCloseDayIndex | None = field(repr=False)
    makerfill_labels: MakerFillLabelIndex = field(repr=False)
    products: tuple[S1Product, ...]
    entry_product_ids: tuple[str, ...]
    day_open_time_ns: int
    entry_cutoff_time_ns: int
    session_expiry_time_ns: int
    source_paths: Mapping[str, Path] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        _validate_development_date(self.date)
        policy_ids = _validate_policy_ids(self.policy_ids)
        if set(self.specs_by_policy) != set(policy_ids):
            raise ValueError("spec map does not match prepared policy_ids")
        if set(self.state_changes_by_policy) != set(policy_ids):
            raise ValueError("state-change map does not match prepared policy_ids")
        if not isinstance(self.common_decisions, pl.DataFrame):
            raise TypeError("common_decisions must be a Polars DataFrame")
        _require_columns(
            self.common_decisions,
            set(ENTRY_DECISION_COLUMNS),
            "prepared common decisions",
        )
        dates = self.common_decisions["Date"].cast(pl.String).unique().to_list()
        if dates != [self.date]:
            raise ValueError("prepared common decisions contain another date")
        if not isinstance(self.raw_books, RawBookDayIndex):
            raise TypeError("raw_books must be a RawBookDayIndex")
        if not isinstance(self.spot_trades, SpotTradeDayIndex):
            raise TypeError("spot_trades must be a SpotTradeDayIndex")
        if self.spot_closes is not None and not isinstance(
            self.spot_closes, SpotCloseDayIndex
        ):
            raise TypeError("spot_closes must be a SpotCloseDayIndex or None")
        if not isinstance(self.makerfill_labels, MakerFillLabelIndex):
            raise TypeError("makerfill_labels must be a MakerFillLabelIndex")
        if not self.products or any(
            not isinstance(product, S1Product) for product in self.products
        ):
            raise TypeError("products must contain S1Product values")
        if any(product.end_date is None for product in self.products):
            raise ValueError("prepared products must preserve contract end_date")
        product_ids = [product.product_id for product in self.products]
        if len(product_ids) != len(set(product_ids)):
            raise ValueError("prepared product ids are duplicated")
        if tuple(sorted(product_ids)) != self.spot_trades.product_ids:
            raise ValueError("spot-trade mapping does not match prepared products")
        entry_ids = tuple(self.entry_product_ids)
        if (
            not entry_ids
            or tuple(sorted(set(entry_ids))) != entry_ids
            or not set(entry_ids).issubset(product_ids)
        ):
            raise ValueError(
                "entry_product_ids must be a sorted nonempty product subset"
            )
        common_ids = tuple(
            sorted(self.common_decisions["ValueCode"].cast(pl.String).unique())
        )
        if common_ids != entry_ids:
            raise ValueError("prepared common decisions do not match entry_product_ids")
        raw_product_ids = {key.value_code for key in self.raw_books.keys}
        if raw_product_ids != set(product_ids):
            raise ValueError("raw-book mapping does not match prepared products")
        if self.day_open_time_ns <= 0:
            raise ValueError("day open must be a positive timestamp")
        if self.entry_cutoff_time_ns <= 0:
            raise ValueError("entry cutoff must be a positive timestamp")
        if self.entry_cutoff_time_ns <= self.day_open_time_ns:
            raise ValueError("entry cutoff must follow day open")
        if self.session_expiry_time_ns <= self.entry_cutoff_time_ns:
            raise ValueError("session expiry must follow entry cutoff")
        expiry_product_ids = tuple(
            sorted(
                product.product_id
                for product in self.products
                if product.end_date == self.date
            )
        )
        if expiry_product_ids:
            if self.spot_closes is None:
                raise ValueError("expiry products require official spot closes")
            if self.spot_closes.date != self.date:
                raise ValueError("spot-close date differs from prepared date")
            if self.spot_closes.product_ids != expiry_product_ids:
                raise ValueError("spot-close products differ from expiry products")
            if self.spot_closes.close_not_before_ns != (
                self.session_expiry_time_ns + SPOT_CLOSE_DELAY_SECONDS * ONE_SECOND_NS
            ):
                raise ValueError("spot-close publication boundary is not 13:30")
            if self.spot_closes.available_count != len(expiry_product_ids):
                raise ValueError("an expiry product lacks an official spot close")
        elif self.spot_closes is not None:
            raise ValueError("spot_closes supplied without an expiring product")
        for policy_id in policy_ids:
            specs = tuple(self.specs_by_policy[policy_id])
            if not specs or any(spec.policy_id != policy_id for spec in specs):
                raise ValueError("prepared PolicySpec values do not match policy")
            changes = self.state_changes_by_policy[policy_id]
            if not isinstance(changes, pl.DataFrame) or changes.is_empty():
                raise ValueError("prepared policy state changes cannot be empty")
        object.__setattr__(self, "policy_ids", policy_ids)
        object.__setattr__(
            self,
            "specs_by_policy",
            MappingProxyType(
                {key: tuple(value) for key, value in self.specs_by_policy.items()}
            ),
        )
        object.__setattr__(
            self,
            "state_changes_by_policy",
            MappingProxyType(dict(self.state_changes_by_policy)),
        )
        object.__setattr__(
            self, "source_paths", MappingProxyType(dict(self.source_paths))
        )


@dataclass(frozen=True, slots=True)
class S1EntryDaySummary:
    date: str
    policy_id: str
    product_count: int
    sparse_state_changes: int
    sent_entry_orders: int
    makerfill_supported_orders: int
    makerfill_eod_positive_orders: int
    actual_active_entry_fills: int
    entry_hedge_executions: int
    eod_paired_open_positions: int
    rollback_flat_positions: int
    unresolved_positions: int
    spot_requests_sent: int
    future_requests_sent: int
    admission_checks: int
    blocked_admission_checks: int
    suppressed_redundant_blocked_admission_probes: int
    blocked_candidate_intents: int
    candidate_intents: int
    suppressed_after_terminal_fills: int
    actual_cancelled_orders: int
    session_expired_orders: int
    global_cap_peak_twd: int
    product_cap_peak_twd: int
    spot_rolling_request_peak: int
    future_rolling_request_peak: int
    entry_hedges_at_t0: int
    entry_hedges_delayed: int
    max_entry_hedge_retry_delay_ms: float | None
    makerfill_eod_positive_rate: float | None
    actual_active_fill_rate: float | None
    entry_hedge_success_rate: float | None
    performance_available: bool
    performance_unavailable_reason: str
    runner_version: str = RUNNER_VERSION

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def paired_open_positions(self) -> int:
        """Backward-compatible alias for the explicitly EOD state count."""

        return self.eod_paired_open_positions


@dataclass(frozen=True, slots=True)
class S1EntryDayRun:
    result: S1ReplayResult = field(repr=False)
    makerfill_assessments: tuple[MakerFillScalarEvent, ...] = field(repr=False)
    summary: S1EntryDaySummary


def prepare_s1_entry_day(
    date: str,
    *,
    policy_ids: Sequence[str] = POLICY_IDS,
    paths: S1EntryRunnerPaths | None = None,
    required_exit_only_bindings: Sequence[S1CarryContractBinding] = (),
) -> PreparedS1EntryDay:
    """Load and compact one development day for one or more policy replays."""

    _validate_development_date(date)
    requested = _validate_policy_ids(policy_ids)
    required_bindings = _validate_required_exit_only_bindings(
        required_exit_only_bindings
    )
    selected_paths = paths or S1EntryRunnerPaths()
    if not isinstance(selected_paths, S1EntryRunnerPaths):
        raise TypeError("paths must be S1EntryRunnerPaths or None")

    source_paths = {
        "causal_fair": selected_paths.causal_path(date),
        "mapping": selected_paths.mapping_path(date),
        "contracts": selected_paths.contracts_path(date),
        "spot_raw": selected_paths.spot_raw_path(date),
        "future_raw": selected_paths.future_raw_path(date),
        "makerfill": selected_paths.makerfill_path(date),
        "mother": selected_paths.mother_path,
        "entry_lookup": selected_paths.entry_lookup_path,
        "convergence": selected_paths.convergence_path,
    }
    for role, path in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {role} input: {path}")

    policy_table = build_policy_spec_table(
        _read_date_partition(selected_paths.mother_path, date),
        _read_date_partition(selected_paths.entry_lookup_path, date),
        _read_date_partition(selected_paths.convergence_path, date),
    )
    day_policy_table = policy_table.filter(
        (pl.col("Date") == date) & pl.col("policy_id").is_in(requested)
    )
    specs_by_policy = _specs_by_policy(day_policy_table, requested)
    product_codes = _common_product_codes(specs_by_policy)
    product_keys = _product_keys(specs_by_policy)
    del policy_table, day_policy_table

    causal_day = (
        pl.scan_parquet(source_paths["causal_fair"])
        .join(
            product_keys.lazy(),
            on=["Date", "ValueCode", "QuoteCode"],
            how="inner",
            validate="m:1",
        )
        .select(CAUSAL_INPUT_COLUMNS)
        .collect(engine="streaming")
    )
    common = materialize_s1_common_day(causal_day)
    state_changes = {
        policy_id: build_s1_policy_state_changes(
            common,
            specs_by_policy[policy_id],
            policy_id=policy_id,
        )
        for policy_id in requested
    }
    common_decisions = common.select(ENTRY_DECISION_COLUMNS).rechunk()
    mapping = (
        pl.scan_parquet(source_paths["mapping"])
        .filter(pl.col("Date").cast(pl.String) == date)
        .select(
            "Date",
            "ValueCode",
            "QuoteCode",
            "spot_ref_price",
            "fut_ref_price",
            "contract_size",
            "end_date",
        )
        .collect(engine="streaming")
    )
    identities = _validated_product_identities(common, mapping, product_keys)
    open_time_ns = _session_open_time_ns(common)
    entry_cutoff_time_ns = open_time_ns + ENTRY_STOP_SECOND * ONE_SECOND_NS
    session_expiry_time_ns = open_time_ns + SESSION_END_SECOND * ONE_SECOND_NS
    spot_raw_scan = pl.scan_parquet(source_paths["spot_raw"])
    future_raw_scan = pl.scan_parquet(source_paths["future_raw"])
    if required_bindings:
        identities = _extend_identities_with_required_exit_only_bindings(
            identities,
            required_bindings,
            date=date,
            contracts=pl.scan_parquet(source_paths["contracts"]),
            spot_raw=spot_raw_scan,
            future_raw=future_raw_scan,
        )
    products = _products_from_identities(identities, session_expiry_time_ns)
    raw_mapping = identities.select(
        "ValueCode",
        "QuoteCode",
        "spot_ref_price",
        "fut_ref_price",
    )
    del causal_day, common
    gc.collect()

    raw_books = build_raw_book_day_index_from_scans(
        spot_raw_scan,
        future_raw_scan,
        raw_mapping,
    )
    spot_trades = build_spot_trade_day_index_from_scan(
        spot_raw_scan,
        raw_mapping,
    )
    expiry_mapping = identities.filter(pl.col("end_date") == date)
    spot_closes = None
    if not expiry_mapping.is_empty():
        close_boundary_ns = (
            session_expiry_time_ns + SPOT_CLOSE_DELAY_SECONDS * ONE_SECOND_NS
        )
        spot_closes = SpotCloseDayIndex.from_selected_scan(
            spot_raw_scan,
            expiry_mapping.select("ValueCode", "QuoteCode"),
            date=date,
            close_not_before_ns=close_boundary_ns,
        )
        missing_closes = tuple(
            product_id
            for product_id in spot_closes.product_ids
            if spot_closes.close_fact(product_id) is None
        )
        if missing_closes:
            raise ValueError(
                f"expiry products lack official spot closes: {missing_closes}"
            )
    makerfill = (
        pl.scan_parquet(source_paths["makerfill"])
        .filter(pl.col("QuoteCode").cast(pl.String).is_in(product_codes))
        .select(
            "QuoteCode",
            "ChannelSeq",
            "Bid1_FillSeconds",
            "Bid2_FillSeconds",
        )
        .collect(engine="streaming")
    )
    makerfill_labels = MakerFillLabelIndex.from_frame(makerfill)
    del makerfill
    gc.collect()
    return PreparedS1EntryDay(
        date=date,
        policy_ids=requested,
        specs_by_policy=specs_by_policy,
        common_decisions=common_decisions,
        state_changes_by_policy=state_changes,
        raw_books=raw_books,
        spot_trades=spot_trades,
        spot_closes=spot_closes,
        makerfill_labels=makerfill_labels,
        products=products,
        entry_product_ids=tuple(product_codes),
        day_open_time_ns=open_time_ns,
        entry_cutoff_time_ns=entry_cutoff_time_ns,
        session_expiry_time_ns=session_expiry_time_ns,
        source_paths=source_paths,
    )


def run_s1_entry_policy(
    prepared: PreparedS1EntryDay,
    policy_id: str,
    *,
    normal_exit_enabled: bool = False,
    accounting_adapter: S1AccountingBridge | None = None,
    capacity_ledger: CapacityLedger | None = None,
    carry_in: Sequence[S1CarryPosition] = (),
    entry_enabled_product_ids: frozenset[str] | None = None,
) -> S1EntryDayRun:
    """Run one prepared policy on the shared chronological joint clock."""

    if not isinstance(prepared, PreparedS1EntryDay):
        raise TypeError("prepared must be a PreparedS1EntryDay")
    if policy_id not in prepared.policy_ids:
        raise ValueError("policy_id is not present in the prepared day")
    if not isinstance(normal_exit_enabled, bool):
        raise TypeError("normal_exit_enabled must be boolean")
    enabled_ids = (
        frozenset(prepared.entry_product_ids)
        if entry_enabled_product_ids is None
        else entry_enabled_product_ids
    )
    if not isinstance(enabled_ids, frozenset) or any(
        not isinstance(product_id, str) or not product_id for product_id in enabled_ids
    ):
        raise TypeError("entry_enabled_product_ids must be a frozenset or None")
    unexpected_entry_ids = enabled_ids.difference(prepared.entry_product_ids)
    if unexpected_entry_ids:
        raise ValueError(
            "entry_enabled_product_ids contains non-entry products: "
            f"{sorted(unexpected_entry_ids)}"
        )
    provider = S1RawEntryBookProvider(prepared.raw_books)
    entry_state = S1EntryStateAdapter(
        prepared.common_decisions,
        prepared.specs_by_policy[policy_id],
        date=prepared.date,
        policy_id=policy_id,
        book_provider=provider,
    )
    makerfill = S1MakerFillBridge(prepared.makerfill_labels)
    session_events = merge_entry_events(
        entry_state.iter_observations(prepared.state_changes_by_policy[policy_id]),
        (
            EntryCutoff(EventCursor(prepared.entry_cutoff_time_ns, 10, 1)),
            SessionExpiry(EventCursor(prepared.session_expiry_time_ns, 10, 1)),
        ),
    )
    accounting = accounting_adapter
    if accounting is None and normal_exit_enabled:
        accounting = S1AccountingBridge(
            default_date=prepared.date,
            scenario_id=policy_id,
            products=tuple(
                S1AccountingProduct(
                    product_id=product.product_id,
                    value_code=product.value_code,
                    contract_size_shares=product.contract_size_shares,
                )
                for product in prepared.products
            ),
        )
    expiry_events = (
        ()
        if accounting is None or prepared.spot_closes is None
        else tuple(ContractExpiry(close) for close in prepared.spot_closes.facts)
    )
    external = chain(session_events, expiry_events)
    loop = S1EventLoop(
        S1LoopConfig(prepared.date, policy_id),
        prepared.products,
        entry_state_adapter=entry_state,
        risk_book_adapter=prepared.raw_books.as_risk_book_adapter(),
        fill_adapter=makerfill,
        accounting_adapter=accounting,
        normal_exit_enabled=normal_exit_enabled,
        spot_trade_adapter=(prepared.spot_trades if normal_exit_enabled else None),
        capacity_ledger=capacity_ledger,
        carry_in=carry_in,
        entry_enabled_product_ids=enabled_ids,
        day_open_time_ns=prepared.day_open_time_ns,
    )
    result = loop.run(external)
    assessments = makerfill.assessments
    return S1EntryDayRun(
        result=result,
        makerfill_assessments=assessments,
        summary=_summarize(prepared, policy_id, result, assessments),
    )


def run_s1_entry_policies(
    prepared: PreparedS1EntryDay,
    *,
    normal_exit_enabled: bool = False,
    accounting_adapter: S1AccountingBridge | None = None,
    capacity_ledger: CapacityLedger | None = None,
    carry_in: Sequence[S1CarryPosition] = (),
    entry_enabled_product_ids: frozenset[str] | None = None,
) -> Iterator[S1EntryDayRun]:
    """Yield policies one at a time so callers can persist and release each run."""

    carry_values = tuple(carry_in)
    if len(prepared.policy_ids) > 1 and (
        accounting_adapter is not None or capacity_ledger is not None or carry_values
    ):
        raise ValueError(
            "multiple policies cannot share accounting, capacity, or carry state"
        )
    for policy_id in prepared.policy_ids:
        yield run_s1_entry_policy(
            prepared,
            policy_id,
            normal_exit_enabled=normal_exit_enabled,
            accounting_adapter=accounting_adapter,
            capacity_ledger=capacity_ledger,
            carry_in=carry_values,
            entry_enabled_product_ids=entry_enabled_product_ids,
        )


def _summarize(
    prepared: PreparedS1EntryDay,
    policy_id: str,
    result: S1ReplayResult,
    assessments: tuple[MakerFillScalarEvent, ...],
) -> S1EntryDaySummary:
    supported = sum(event.outcome_supported for event in assessments)
    eod_positive = sum(event.makerfill_potential_fill is True for event in assessments)
    filled_events = sum(event.status == "filled" for event in result.fill_events)
    entry_maker_executions = sum(
        execution.role == "entry_maker" for execution in result.executions
    )
    entry_hedge_executions = sum(
        execution.role == "entry_hedge" for execution in result.executions
    )
    if filled_events != entry_maker_executions:
        raise RuntimeError("entry maker executions disagree with active fill events")
    if entry_hedge_executions > entry_maker_executions:
        raise RuntimeError("entry hedge executions exceed entry maker executions")
    position_states = Counter(position.state for position in result.positions)
    paired = position_states["paired_open"]
    rollback_flat = position_states["entry_emergency_rollback_flat"]
    unresolved = position_states["entry_hedge_timeout_unresolved"]
    blocked = sum(not event.admitted for event in result.admission_events)
    blocked_ids = {
        event.capacity_id for event in result.admission_events if not event.admitted
    }
    candidate_ids = {
        *(order.candidate_intent_id for order in result.orders),
        *(audit.candidate_intent_id for audit in result.candidate_intent_audit),
    }
    order_events = Counter(event.event_type for event in result.order_events)
    cap_peak = max(
        (
            transition.global_after.total_committed_notional_twd
            for transition in result.capacity_transitions
        ),
        default=0,
    )
    product_cap_peak = max(
        (
            transition.product_after.total_committed_notional_twd
            for transition in result.capacity_transitions
        ),
        default=0,
    )
    hedge_delays = _entry_hedge_retry_delays_ns(result)
    return S1EntryDaySummary(
        date=prepared.date,
        policy_id=policy_id,
        product_count=len(prepared.entry_product_ids),
        sparse_state_changes=prepared.state_changes_by_policy[policy_id].height,
        sent_entry_orders=len(result.orders),
        makerfill_supported_orders=supported,
        makerfill_eod_positive_orders=eod_positive,
        actual_active_entry_fills=entry_maker_executions,
        entry_hedge_executions=entry_hedge_executions,
        eod_paired_open_positions=paired,
        rollback_flat_positions=rollback_flat,
        unresolved_positions=unresolved,
        spot_requests_sent=result.spot_requests_sent,
        future_requests_sent=result.future_requests_sent,
        admission_checks=len(result.admission_events),
        blocked_admission_checks=blocked,
        suppressed_redundant_blocked_admission_probes=(
            result.suppressed_redundant_blocked_admission_probes
        ),
        blocked_candidate_intents=len(blocked_ids),
        candidate_intents=len(candidate_ids),
        suppressed_after_terminal_fills=sum(
            event.status == "suppressed_after_terminal" for event in result.fill_events
        ),
        actual_cancelled_orders=order_events["actual_cancelled"],
        session_expired_orders=order_events["session_expired"],
        global_cap_peak_twd=cap_peak,
        product_cap_peak_twd=product_cap_peak,
        spot_rolling_request_peak=_rolling_request_peak(result, "spot"),
        future_rolling_request_peak=_rolling_request_peak(result, "future"),
        entry_hedges_at_t0=sum(delay == 0 for delay in hedge_delays),
        entry_hedges_delayed=sum(delay > 0 for delay in hedge_delays),
        max_entry_hedge_retry_delay_ms=(
            None if not hedge_delays else max(hedge_delays) / 1_000_000.0
        ),
        makerfill_eod_positive_rate=_rate(eod_positive, supported),
        actual_active_fill_rate=_rate(entry_maker_executions, supported),
        entry_hedge_success_rate=_rate(
            entry_hedge_executions,
            entry_maker_executions,
        ),
        performance_available=result.mean_daily_net_twd is not None,
        performance_unavailable_reason=(
            ""
            if result.mean_daily_net_twd is not None
            else result.performance_unavailable_reason
        ),
    )


def _specs_by_policy(
    frame: pl.DataFrame,
    policy_ids: tuple[str, ...],
) -> dict[str, tuple[PolicySpec, ...]]:
    if frame.is_empty():
        raise ValueError("date has no selected PolicySpec rows")
    result: dict[str, tuple[PolicySpec, ...]] = {}
    for policy_id in policy_ids:
        selected = frame.filter(pl.col("policy_id") == policy_id).sort(
            "ValueCode", "entry_tod_bucket"
        )
        if selected.is_empty():
            raise ValueError(f"date has no PolicySpec rows for {policy_id}")
        result[policy_id] = tuple(
            PolicySpec.from_dict(row) for row in selected.iter_rows(named=True)
        )
    return result


def _common_product_codes(
    specs_by_policy: Mapping[str, tuple[PolicySpec, ...]],
) -> list[str]:
    product_sets = {
        policy_id: {spec.ValueCode for spec in specs}
        for policy_id, specs in specs_by_policy.items()
    }
    first = next(iter(product_sets.values()))
    if not first or any(values != first for values in product_sets.values()):
        raise ValueError("prepared policies do not have a common product universe")
    return sorted(first)


def _product_keys(
    specs_by_policy: Mapping[str, tuple[PolicySpec, ...]],
) -> pl.DataFrame:
    first = next(iter(specs_by_policy.values()))
    keys = pl.from_dicts(
        [
            {
                "Date": spec.Date,
                "ValueCode": spec.ValueCode,
                "QuoteCode": spec.QuoteCode,
            }
            for spec in first
        ]
    ).unique()
    if keys.select("ValueCode").n_unique() != keys.height:
        raise ValueError("PolicySpec ValueCode/QuoteCode mapping is not one-to-one")
    return keys.sort("ValueCode")


def _validate_required_exit_only_bindings(
    values: Sequence[S1CarryContractBinding],
) -> tuple[S1CarryContractBinding, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("required_exit_only_bindings must be a sequence")
    bindings = tuple(values)
    if any(not isinstance(value, S1CarryContractBinding) for value in bindings):
        raise TypeError(
            "required_exit_only_bindings must contain S1CarryContractBinding values"
        )
    if any(value.future_contracts != 1 for value in bindings):
        raise ValueError("required exit-only bindings must use one future contract")
    if any(value.product_id != value.value_code for value in bindings):
        raise ValueError("required exit-only product_id must equal value_code")
    if any(value.end_date is None for value in bindings):
        raise ValueError("required exit-only bindings must preserve end_date")
    if len({value.product_id for value in bindings}) != len(bindings) or len(
        {value.quote_code for value in bindings}
    ) != len(bindings):
        raise ValueError("required exit-only bindings must be one-to-one")
    return tuple(sorted(bindings, key=lambda value: value.product_id))


def _extend_identities_with_required_exit_only_bindings(
    identities: pl.DataFrame,
    bindings: Sequence[S1CarryContractBinding],
    *,
    date: str,
    contracts: pl.LazyFrame,
    spot_raw: pl.LazyFrame,
    future_raw: pl.LazyFrame,
) -> pl.DataFrame:
    """Add exact frozen carry pairs absent from the active daily mapping."""

    required = _validate_required_exit_only_bindings(bindings)
    if not required:
        return identities
    _require_columns(
        identities,
        {
            "ValueCode",
            "QuoteCode",
            "spot_ref_price",
            "fut_ref_price",
            "contract_size",
            "end_date",
        },
        "daily identities",
    )
    contract_schema = contracts.collect_schema()
    _require_schema_columns(
        contract_schema,
        {
            "ValueCode",
            "QuoteCode",
            "contract_size",
            "decimal_locator",
            "end_date",
            "fut_ref_price",
        },
        "daily contracts metadata",
    )
    _require_schema_columns(
        spot_raw.collect_schema(), {"QuoteCode", "RefPrice"}, "spot raw"
    )
    _require_schema_columns(
        future_raw.collect_schema(),
        {"QuoteCode", "DecimalLocator"},
        "future raw",
    )
    existing_by_product = {
        str(row["ValueCode"]): row for row in identities.iter_rows(named=True)
    }
    existing_by_quote = {
        str(row["QuoteCode"]): row for row in identities.iter_rows(named=True)
    }
    additions: list[dict[str, object]] = []
    for binding in required:
        if binding.end_date is None or binding.end_date < date:
            raise ValueError("required exit-only binding is expired before replay date")
        existing = existing_by_product.get(binding.product_id)
        if existing is not None:
            if (
                str(existing["QuoteCode"]) != binding.quote_code
                or round(float(existing["contract_size"]))
                != binding.contract_size_shares
                or str(existing["end_date"]) != binding.end_date
            ):
                raise ValueError("daily mapping conflicts with frozen carry binding")
            continue
        quote_owner = existing_by_quote.get(binding.quote_code)
        if quote_owner is not None:
            raise ValueError("frozen carry QuoteCode belongs to another daily product")

        metadata = (
            contracts.filter(
                (pl.col("ValueCode").cast(pl.String) == binding.value_code)
                & (pl.col("QuoteCode").cast(pl.String) == binding.quote_code)
            )
            .select(
                pl.col("ValueCode").cast(pl.String),
                pl.col("QuoteCode").cast(pl.String),
                pl.col("contract_size").cast(pl.Float64),
                pl.col("decimal_locator").cast(pl.Int64),
                pl.col("end_date")
                .cast(pl.String)
                .str.replace_all("-", "")
                .alias("end_date"),
                pl.col("fut_ref_price").cast(pl.Float64),
            )
            .collect(engine="streaming")
        )
        if metadata.height != 1:
            raise ValueError("frozen carry pair lacks one exact contracts metadata row")
        meta = metadata.row(0, named=True)
        contract_size = float(meta["contract_size"])
        future_reference = float(meta["fut_ref_price"])
        decimal_locator = int(meta["decimal_locator"])
        if (
            not math.isfinite(contract_size)
            or abs(contract_size - binding.contract_size_shares) >= 1e-9
            or str(meta["end_date"]) != binding.end_date
            or not math.isfinite(future_reference)
            or future_reference <= 0
            or decimal_locator < 0
        ):
            raise ValueError("contracts metadata conflicts with frozen carry binding")

        spot = (
            spot_raw.filter(pl.col("QuoteCode").cast(pl.String) == binding.value_code)
            .select(pl.col("RefPrice").cast(pl.Float64).alias("reference"))
            .filter(
                pl.col("reference").is_not_null()
                & pl.col("reference").is_finite()
                & (pl.col("reference") > 0)
            )
            .unique()
            .collect(engine="streaming")
        )
        if spot.height != 1:
            raise ValueError("frozen carry spot raw lacks one unique positive RefPrice")
        future_decimal = (
            future_raw.filter(pl.col("QuoteCode").cast(pl.String) == binding.quote_code)
            .select(pl.col("DecimalLocator").cast(pl.Int64).alias("decimal_locator"))
            .drop_nulls()
            .unique()
            .collect(engine="streaming")
        )
        if (
            future_decimal.height != 1
            or int(future_decimal.item(0, "decimal_locator")) != decimal_locator
        ):
            raise ValueError(
                "frozen carry future raw/metadata DecimalLocator differs or is absent"
            )
        addition = {
            "ValueCode": binding.value_code,
            "QuoteCode": binding.quote_code,
            "spot_ref_price": float(spot.item(0, "reference")),
            "fut_ref_price": future_reference,
            "contract_size": contract_size,
            "end_date": binding.end_date,
        }
        additions.append(addition)
        existing_by_product[binding.product_id] = addition
        existing_by_quote[binding.quote_code] = addition
    if not additions:
        return identities
    return pl.concat(
        (identities, pl.from_dicts(additions, infer_schema_length=None)),
        how="vertical_relaxed",
    ).sort("ValueCode")


def _validated_product_identities(
    common: pl.DataFrame,
    mapping: pl.DataFrame,
    product_keys: pl.DataFrame,
) -> pl.DataFrame:
    mapping_columns = (
        "Date",
        "ValueCode",
        "QuoteCode",
        "spot_ref_price",
        "fut_ref_price",
        "contract_size",
        "end_date",
    )
    _require_columns(mapping, set(mapping_columns), "daily mapping")
    normalized_mapping = _normalize_identity_frame(mapping)
    invalid_mapping = normalized_mapping.filter(_invalid_identity_expr())
    if not invalid_mapping.is_empty():
        raise ValueError("daily active mapping has invalid product identities")
    for end_date in normalized_mapping["end_date"]:
        _valid_yyyymmdd(str(end_date), "mapping end_date")
    if normalized_mapping.is_empty() or any(
        normalized_mapping.select(column).n_unique() != normalized_mapping.height
        for column in ("ValueCode", "QuoteCode")
    ):
        raise ValueError("daily active mapping is not one-to-one")
    if (
        normalized_mapping.select("Date", "ValueCode", "QuoteCode").n_unique()
        != normalized_mapping.height
    ):
        raise ValueError("daily active mapping keys are duplicated")

    normalized_keys = product_keys.select(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
    )
    if set(normalized_mapping["Date"].unique()) != set(
        normalized_keys["Date"].unique()
    ):
        raise ValueError("daily active mapping date differs from PolicySpec date")
    entry_mapping = normalized_mapping.join(
        normalized_keys,
        on=["Date", "ValueCode", "QuoteCode"],
        how="inner",
        validate="1:1",
    )
    if entry_mapping.height != normalized_keys.height:
        raise ValueError("daily mapping does not cover every PolicySpec product")

    observed = _normalize_identity_frame(common.select(mapping_columns)).unique()
    if not observed.filter(_invalid_identity_expr()).is_empty():
        raise ValueError("causal entry subset has invalid product identities")
    for end_date in observed["end_date"]:
        _valid_yyyymmdd(str(end_date), "causal end_date")
    if observed.height != product_keys.height:
        raise ValueError("causal identity/reference/contract values change within day")
    compared = observed.join(
        entry_mapping,
        on=["Date", "ValueCode", "QuoteCode"],
        how="left",
        suffix="_mapping",
        validate="1:1",
    )
    invalid = compared.filter(
        pl.col("spot_ref_price_mapping").is_null()
        | pl.col("fut_ref_price_mapping").is_null()
        | pl.col("contract_size_mapping").is_null()
        | ((pl.col("spot_ref_price") - pl.col("spot_ref_price_mapping")).abs() >= 1e-9)
        | ((pl.col("fut_ref_price") - pl.col("fut_ref_price_mapping")).abs() >= 1e-9)
        | ((pl.col("contract_size") - pl.col("contract_size_mapping")).abs() >= 1e-9)
        | (pl.col("end_date") != pl.col("end_date_mapping"))
    )
    if not invalid.is_empty():
        raise ValueError("causal product identity disagrees with daily mapping")
    return normalized_mapping.select(
        "ValueCode",
        "QuoteCode",
        "spot_ref_price",
        "fut_ref_price",
        "contract_size",
        "end_date",
    ).sort("ValueCode")


def _invalid_identity_expr() -> pl.Expr:
    return (
        pl.col("Date").is_null()
        | (pl.col("Date").str.len_chars() != 8)
        | pl.col("ValueCode").is_null()
        | (pl.col("ValueCode").str.len_chars() == 0)
        | pl.col("QuoteCode").is_null()
        | (pl.col("QuoteCode").str.len_chars() == 0)
        | pl.col("spot_ref_price").is_null()
        | ~pl.col("spot_ref_price").is_finite()
        | (pl.col("spot_ref_price") <= 0)
        | pl.col("fut_ref_price").is_null()
        | ~pl.col("fut_ref_price").is_finite()
        | (pl.col("fut_ref_price") <= 0)
        | pl.col("contract_size").is_null()
        | ~pl.col("contract_size").is_finite()
        | (pl.col("contract_size") <= 0)
        | pl.col("end_date").is_null()
        | (pl.col("end_date").str.len_chars() != 8)
    )


def _products_from_identities(
    identities: pl.DataFrame,
    session_expiry_time_ns: int,
) -> tuple[S1Product, ...]:
    products: list[S1Product] = []
    for row in identities.iter_rows(named=True):
        contract_size = row["contract_size"]
        if (
            isinstance(contract_size, bool)
            or not isinstance(contract_size, (int, float))
            or contract_size <= 0
            or abs(float(contract_size) - round(float(contract_size))) >= 1e-9
        ):
            raise ValueError("contract_size must be a positive whole share count")
        value_code = str(row["ValueCode"])
        products.append(
            S1Product(
                product_id=value_code,
                value_code=value_code,
                quote_code=str(row["QuoteCode"]),
                contract_size_shares=round(float(contract_size)),
                future_contracts=1,
                future_session_end_time_ns=session_expiry_time_ns,
                spot_session_end_time_ns=session_expiry_time_ns,
                end_date=_valid_yyyymmdd(str(row["end_date"]), "end_date"),
            )
        )
    return tuple(products)


def _normalize_identity_frame(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.select(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("spot_ref_price").cast(pl.Float64),
        pl.col("fut_ref_price").cast(pl.Float64),
        pl.col("contract_size").cast(pl.Float64),
        pl.col("end_date").cast(pl.String).str.replace_all("-", "").alias("end_date"),
    )


def _session_open_time_ns(common: pl.DataFrame) -> int:
    starts = (
        common.filter(pl.col("seconds_from_open") == SESSION_START_SECOND)
        .select(
            (pl.col("decision_time_ns") - SESSION_START_SECOND * ONE_SECOND_NS).alias(
                "open_time_ns"
            )
        )["open_time_ns"]
        .unique()
        .to_list()
    )
    if len(starts) != 1 or not isinstance(starts[0], int) or starts[0] <= 0:
        raise ValueError("causal day has no unique session-open clock")
    return starts[0]


def _read_date_partition(path: Path, date: str) -> pl.DataFrame:
    frame = (
        pl.scan_parquet(path)
        .filter(pl.col("Date").cast(pl.String) == date)
        .collect(engine="streaming")
    )
    if frame.is_empty():
        raise ValueError(f"input has no rows for {date}: {path}")
    return frame


def _entry_hedge_retry_delays_ns(result: S1ReplayResult) -> tuple[int, ...]:
    trigger_by_position = {
        event.position_id: event.cursor.recv_time_ns
        for event in result.position_events
        if event.to_state == "hedge_pending" and event.reason == "entry_maker_fill"
    }
    delays: list[int] = []
    for execution in result.executions:
        if execution.role != "entry_hedge":
            continue
        try:
            trigger_ns = trigger_by_position[execution.position_id]
        except KeyError as error:
            raise RuntimeError("entry hedge has no maker-fill trigger") from error
        delay = execution.cursor.recv_time_ns - trigger_ns - HEDGE_DELAY_NS
        if delay < 0:
            raise RuntimeError("entry hedge execution precedes its B6 t0")
        delays.append(delay)
    return tuple(delays)


def _rolling_request_peak(result: S1ReplayResult, venue: str) -> int:
    if venue not in ("spot", "future"):
        raise ValueError("venue must be spot or future")
    times = sorted(
        event.event_cursor.recv_time_ns
        for event in result.request_events
        if event.event_type == "actual_send" and event.venue == venue
    )
    expected = (
        result.spot_requests_sent if venue == "spot" else result.future_requests_sent
    )
    if len(times) != expected:
        raise RuntimeError("request-event count disagrees with scheduler total")
    left = 0
    peak = 0
    for right, timestamp_ns in enumerate(times):
        boundary = timestamp_ns - ONE_SECOND_NS
        while left <= right and times[left] <= boundary:
            left += 1
        peak = max(peak, right - left + 1)
    return peak


def _validate_policy_ids(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("policy_ids must be a sequence of policy names")
    result = tuple(values)
    if not result or any(value not in POLICY_IDS for value in result):
        raise ValueError("policy_ids contain an unknown or empty policy")
    if len(result) != len(set(result)):
        raise ValueError("policy_ids cannot contain duplicates")
    return result


def _validate_development_date(value: str) -> None:
    if not isinstance(value, str) or len(value) != 8 or not value.isdigit():
        raise ValueError("date must be YYYYMMDD")
    try:
        datetime.strptime(value, "%Y%m%d").replace(tzinfo=UTC)
    except ValueError as error:
        raise ValueError("date must be a valid YYYYMMDD date") from error
    if value > DEVELOPMENT_END_DATE:
        raise ValueError(
            "S1 development replay cannot read the protected 20260814+ window"
        )


def _valid_yyyymmdd(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 8 or not value.isdigit():
        raise ValueError(f"{name} must be valid YYYYMMDD")
    try:
        datetime.strptime(value, "%Y%m%d").replace(tzinfo=UTC)
    except ValueError as error:
        raise ValueError(f"{name} must be valid YYYYMMDD") from error
    return value


def _require_columns(frame: pl.DataFrame, required: set[str], source: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def _require_schema_columns(
    schema: Mapping[str, object], required: set[str], source: str
) -> None:
    missing = sorted(required - set(schema))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


__all__ = [
    "DEVELOPMENT_END_DATE",
    "RUNNER_VERSION",
    "SPOT_CLOSE_DELAY_SECONDS",
    "PreparedS1EntryDay",
    "S1EntryDayRun",
    "S1EntryDaySummary",
    "S1EntryRunnerPaths",
    "prepare_s1_entry_day",
    "run_s1_entry_policies",
    "run_s1_entry_policy",
]
