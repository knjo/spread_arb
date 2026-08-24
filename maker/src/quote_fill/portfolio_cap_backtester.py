"""Chronological portfolio-cap replay for fully priced paired paths.

This module deliberately starts *after* entry/exit execution labelling.  Every
input row must already have one point-identified terminal cashflow, exact entry
and exit leg prices, and transaction costs.  The replay only decides whether a
candidate can be admitted under a hard intraday portfolio/product cap and then
releases that capacity at the supplied terminal timestamp.

An optional EOD overnight limit is reporting-only.  It never rejects an entry
and never invents an exit.  This distinction is important for experiments that
allow a larger intraday book and use a separate exit policy to reduce overnight
inventory before the close.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone
import heapq
import math
from typing import Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import polars as pl


DEFAULT_HARD_INTRADAY_CAPS_TWD = (
    10_000_000.0,
    20_000_000.0,
    30_000_000.0,
    40_000_000.0,
    50_000_000.0,
)

_REQUIRED_COLUMNS = {
    "Date",
    "ValueCode",
    "policy_path_id",
    "position_established_ns",
    "normalization_notional_twd",
    "terminal_date",
    "exit_decision_time_ns",
    "entry_spot_price",
    "entry_future_price",
    "entry_contract_size_shares",
    "exit_spot_price",
    "exit_future_price",
    "gross_cycle_pnl_twd",
    "total_transaction_cost_twd",
    "net_cycle_pnl_twd",
}

_FLOAT_COLUMNS = (
    "normalization_notional_twd",
    "entry_spot_price",
    "entry_future_price",
    "exit_spot_price",
    "exit_future_price",
    "gross_cycle_pnl_twd",
    "total_transaction_cost_twd",
    "net_cycle_pnl_twd",
)


@dataclass(frozen=True)
class PortfolioCapScenario:
    """One independently replayed capacity scenario.

    ``hard_intraday_cap_twd`` and its per-product fraction are enforced at
    every accepted entry.  ``eod_overnight_limit_twd`` is intentionally only a
    diagnostic target; upstream terminal paths must encode any 13:00 unwind
    policy used to reach it.
    """

    scenario_id: str
    hard_intraday_cap_twd: float
    eod_overnight_limit_twd: float | None = None

    def validate(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id must be non-empty")
        if not _positive_finite(self.hard_intraday_cap_twd):
            raise ValueError("hard_intraday_cap_twd must be finite and positive")
        if self.eod_overnight_limit_twd is not None and not _positive_finite(
            self.eod_overnight_limit_twd
        ):
            raise ValueError(
                "eod_overnight_limit_twd must be finite and positive or None"
            )


def default_cap_scenarios() -> tuple[PortfolioCapScenario, ...]:
    return tuple(
        PortfolioCapScenario(
            scenario_id=f"hard_intraday_{int(cap)}_eod_reporting_only",
            hard_intraday_cap_twd=cap,
        )
        for cap in DEFAULT_HARD_INTRADAY_CAPS_TWD
    )


@dataclass(frozen=True)
class PortfolioCapBacktestConfig:
    scenarios: tuple[PortfolioCapScenario, ...] = default_cap_scenarios()
    per_product_fraction: float = 0.30
    entry_cutoff_local_time: time = time(13, 0)
    session_open_local_time: time = time(9, 0)
    session_close_local_time: time = time(13, 30)
    timezone_name: str = "Asia/Taipei"
    block_new_entries_for_products_held_at_session_open: bool = False

    def validate(self) -> None:
        if not self.scenarios:
            raise ValueError("at least one cap scenario is required")
        identifiers: set[str] = set()
        for scenario in self.scenarios:
            scenario.validate()
            if scenario.scenario_id in identifiers:
                raise ValueError("cap scenario_id values must be unique")
            identifiers.add(scenario.scenario_id)
        if not _positive_finite(self.per_product_fraction) or (
            self.per_product_fraction > 1.0
        ):
            raise ValueError("per_product_fraction must be in (0, 1]")
        if not self.session_open_local_time < self.entry_cutoff_local_time:
            raise ValueError("entry cutoff must be after the session open")
        if not self.entry_cutoff_local_time <= self.session_close_local_time:
            raise ValueError("entry cutoff must not be after the session close")
        try:
            ZoneInfo(self.timezone_name)
        except (KeyError, ValueError) as error:
            raise ValueError("timezone_name is invalid") from error
        if not isinstance(
            self.block_new_entries_for_products_held_at_session_open, bool
        ):
            raise ValueError("opening-carry exit-only policy flag must be boolean")


@dataclass(frozen=True)
class PortfolioCapBacktestResult:
    events: pl.DataFrame
    daily: pl.DataFrame
    summary: pl.DataFrame


def backtest_priced_paths(
    paths: pl.DataFrame,
    *,
    config: PortfolioCapBacktestConfig = PortfolioCapBacktestConfig(),
    session_dates: Sequence[str] | None = None,
) -> PortfolioCapBacktestResult:
    """Replay priced paths under each configured hard intraday cap.

    Entries at or after the configured 13:00 cutoff are rejected.  At equal
    timestamps all entries are processed before exits, a conservative capacity
    convention.  If ``session_dates`` is omitted, the daily panel uses the
    sorted union of entry and terminal dates; callers should supply a complete
    trading calendar when no-event carry days must appear explicitly.
    """

    config.validate()
    source = _normalise_priced_paths(paths, config)
    dates, calendar_supplied = _normalise_session_dates(source, session_dates)

    all_event_rows: list[dict[str, object]] = []
    all_daily_rows: list[dict[str, object]] = []
    all_summary_rows: list[dict[str, object]] = []
    for scenario in config.scenarios:
        scenario_events, accepted = _replay_scenario(source, scenario, config)
        daily = _build_daily_rows(
            scenario_events,
            accepted,
            dates,
            scenario,
            config,
            calendar_supplied=calendar_supplied,
        )
        summary = _build_summary_row(
            scenario_events,
            accepted,
            daily,
            scenario,
            config,
            calendar_supplied=calendar_supplied,
        )
        all_event_rows.extend(scenario_events)
        all_daily_rows.extend(daily)
        all_summary_rows.append(summary)

    result = PortfolioCapBacktestResult(
        events=_event_frame(all_event_rows),
        daily=_daily_frame(all_daily_rows),
        summary=_summary_frame(all_summary_rows),
    )
    _validate_result(result, source, config, dates)
    return result


def local_session_timestamp_ns(
    date: str, local_time: time, timezone_name: str = "Asia/Taipei"
) -> int:
    """Convert a local wall-clock session timestamp to epoch nanoseconds."""

    day = _parse_date(date)
    local = datetime.combine(day.date(), local_time, tzinfo=ZoneInfo(timezone_name))
    utc = local.astimezone(timezone.utc)
    delta = utc - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (
        (delta.days * 86_400 + delta.seconds) * 1_000_000_000
        + delta.microseconds * 1_000
    )


def _normalise_priced_paths(
    paths: pl.DataFrame, config: PortfolioCapBacktestConfig
) -> pl.DataFrame:
    if not isinstance(paths, pl.DataFrame):
        raise TypeError("paths must be a polars DataFrame")
    missing = sorted(_REQUIRED_COLUMNS - set(paths.columns))
    if missing:
        raise ValueError(f"priced paths missing columns: {missing}")
    if paths.is_empty():
        raise ValueError("priced paths must be non-empty")

    try:
        result = paths.with_columns(
            pl.col("Date").cast(pl.String),
            pl.col("terminal_date").cast(pl.String),
            pl.col("ValueCode").cast(pl.String),
            pl.col("policy_path_id").cast(pl.String),
            pl.col("position_established_ns").cast(pl.Int64),
            pl.col("exit_decision_time_ns").cast(pl.Int64),
            pl.col("entry_contract_size_shares").cast(pl.Int64),
            *(pl.col(column).cast(pl.Float64) for column in _FLOAT_COLUMNS),
        ).sort(
            ["Date", "position_established_ns", "ValueCode", "policy_path_id"]
        )
    except (TypeError, ValueError, pl.exceptions.PolarsError) as error:
        raise ValueError(
            "priced path columns cannot be cast to canonical types"
        ) from error

    if "entry_admission_time_ns" in result.columns:
        result = result.with_columns(pl.col("entry_admission_time_ns").cast(pl.Int64))
    else:
        result = result.with_columns(
            pl.col("position_established_ns").alias("entry_admission_time_ns")
        )

    if result["policy_path_id"].n_unique() != result.height:
        raise ValueError("policy_path_id values must be unique")
    if "terminal_cashflow_priced" in result.columns and result.filter(
        (pl.col("terminal_cashflow_priced") != True).fill_null(True)  # noqa: E712
    ).height:
        raise ValueError("every path must have a point-identified terminal cashflow")

    zone = ZoneInfo(config.timezone_name)
    for row in result.iter_rows(named=True):
        identifier = str(row["policy_path_id"])
        entry_date = _canonical_date(str(row["Date"]))
        terminal_date = _canonical_date(str(row["terminal_date"]))
        entry_ns = _integer_ns(row["position_established_ns"], "entry", identifier)
        admission_ns = _integer_ns(
            row["entry_admission_time_ns"], "entry admission", identifier
        )
        exit_ns = _integer_ns(row["exit_decision_time_ns"], "exit", identifier)
        if (terminal_date, exit_ns) < (entry_date, entry_ns):
            raise ValueError(f"{identifier}: terminal timestamp precedes entry")
        if _local_date_for_ns(entry_ns, zone) != entry_date:
            raise ValueError(f"{identifier}: entry timestamp does not belong to Date")
        if _local_date_for_ns(admission_ns, zone) != entry_date:
            raise ValueError(
                f"{identifier}: entry admission timestamp does not belong to Date"
            )
        if admission_ns > entry_ns:
            raise ValueError(
                f"{identifier}: entry admission timestamp follows establishment"
            )
        if _local_date_for_ns(exit_ns, zone) != terminal_date:
            raise ValueError(
                f"{identifier}: exit timestamp does not belong to terminal_date"
            )

        for column in _FLOAT_COLUMNS:
            value = row[column]
            if value is None or not math.isfinite(float(value)):
                raise ValueError(f"{identifier}: {column} must be finite and non-null")
        for column in (
            "normalization_notional_twd",
            "entry_spot_price",
            "entry_future_price",
            "exit_spot_price",
            "exit_future_price",
        ):
            if float(row[column]) <= 0:
                raise ValueError(f"{identifier}: {column} must be positive")
        shares = row["entry_contract_size_shares"]
        if isinstance(shares, bool) or not isinstance(shares, int) or shares <= 0:
            raise ValueError(
                f"{identifier}: entry_contract_size_shares must be positive"
            )
        expected_notional = float(row["entry_spot_price"]) * shares
        if not math.isclose(
            expected_notional,
            float(row["normalization_notional_twd"]),
            rel_tol=1e-10,
            abs_tol=1e-6,
        ):
            raise ValueError(
                f"{identifier}: one-way entry notional differs from "
                "spot price × shares"
            )
        cost = float(row["total_transaction_cost_twd"])
        if cost < 0:
            raise ValueError(f"{identifier}: transaction cost must be non-negative")
        if not math.isclose(
            float(row["gross_cycle_pnl_twd"]) - cost,
            float(row["net_cycle_pnl_twd"]),
            rel_tol=1e-10,
            abs_tol=1e-6,
        ):
            raise ValueError(f"{identifier}: gross - cost does not equal net")

    return result.with_columns(
        (
            pl.col("entry_contract_size_shares")
            * (pl.col("entry_spot_price") + pl.col("entry_future_price"))
        ).alias("entry_two_leg_turnover_twd"),
        (
            pl.col("entry_contract_size_shares") * pl.col("exit_spot_price")
        ).alias("exit_one_way_turnover_twd"),
        (
            pl.col("entry_contract_size_shares")
            * (pl.col("exit_spot_price") + pl.col("exit_future_price"))
        ).alias("exit_two_leg_turnover_twd"),
    )


def _normalise_session_dates(
    paths: pl.DataFrame, session_dates: Sequence[str] | None
) -> tuple[tuple[str, ...], bool]:
    required_dates = set(paths["Date"].to_list()) | set(
        paths["terminal_date"].to_list()
    )
    if session_dates is None:
        values = tuple(sorted(required_dates))
        return values, False
    values = tuple(_canonical_date(str(value)) for value in session_dates)
    if not values or tuple(sorted(set(values))) != values:
        raise ValueError("session_dates must be non-empty, ascending, and unique")
    missing = sorted(required_dates - set(values))
    if missing:
        raise ValueError(f"session_dates omit path event dates: {missing}")
    return values, True


def _replay_scenario(
    paths: pl.DataFrame,
    scenario: PortfolioCapScenario,
    config: PortfolioCapBacktestConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    product_cap = scenario.hard_intraday_cap_twd * config.per_product_fraction
    active: dict[str, dict[str, object]] = {}
    active_count_by_product: dict[str, int] = {}
    active_notional_by_product: dict[str, float] = {}
    exit_heap: list[tuple[str, int, str, dict[str, object]]] = []
    event_rows: list[dict[str, object]] = []
    accepted: list[dict[str, object]] = []
    active_notional = 0.0
    current_entry_date: str | None = None
    opening_carry_products: set[str] = set()

    def common_event(
        row: Mapping[str, object], event_type: str, event_date: str, event_ns: int
    ) -> dict[str, object]:
        return {
            "scenario_id": scenario.scenario_id,
            "hard_intraday_cap_twd": scenario.hard_intraday_cap_twd,
            "per_product_cap_fraction": config.per_product_fraction,
            "per_product_hard_intraday_cap_twd": product_cap,
            "eod_overnight_limit_twd": scenario.eod_overnight_limit_twd,
            "event_sequence": len(event_rows) + 1,
            "event_type": event_type,
            "event_date": event_date,
            "event_timestamp_ns": event_ns,
            "ValueCode": str(row["ValueCode"]),
            "policy_path_id": str(row["policy_path_id"]),
            "capacity_notional_twd": float(row["normalization_notional_twd"]),
            "notional_basis": "one_way_spot_entry_notional_twd",
            "hard_intraday_cap_enforced_on_admission": True,
            "eod_overnight_limit_enforced_on_admission": False,
            "opening_carry_product_exit_only_policy": (
                config.block_new_entries_for_products_held_at_session_open
            ),
            "equal_timestamp_tie_handling": "entry_before_exit",
        }

    def emit_exit(row: dict[str, object]) -> None:
        nonlocal active_notional
        identifier = str(row["policy_path_id"])
        product = str(row["ValueCode"])
        if identifier not in active:
            raise AssertionError("accepted path is not active at terminal event")
        notional = float(row["normalization_notional_twd"])
        before_count = len(active)
        before_notional = active_notional
        before_product_count = active_count_by_product[product]
        before_product_notional = active_notional_by_product[product]
        del active[identifier]
        active_notional -= notional
        active_count_by_product[product] -= 1
        active_notional_by_product[product] -= notional
        if abs(active_notional) < 1e-8:
            active_notional = 0.0
        if abs(active_notional_by_product[product]) < 1e-8:
            active_notional_by_product[product] = 0.0

        event = common_event(
            row,
            "position_exit",
            str(row["terminal_date"]),
            int(row["exit_decision_time_ns"]),
        )
        event.update(
            {
                "entry_date": str(row["Date"]),
                "terminal_date": str(row["terminal_date"]),
                "entry_admission_status": None,
                "entry_admitted": None,
                "entry_cutoff_blocked": None,
                "opening_carry_exit_only_blocked": None,
                "portfolio_cap_blocked": None,
                "per_product_cap_blocked": None,
                "requested_entry_one_way_notional_twd": None,
                "accepted_entry_one_way_turnover_twd": None,
                "accepted_entry_two_leg_turnover_twd": None,
                "released_capacity_notional_twd": notional,
                "exit_one_way_turnover_twd": float(
                    row["exit_one_way_turnover_twd"]
                ),
                "exit_two_leg_turnover_twd": float(
                    row["exit_two_leg_turnover_twd"]
                ),
                "gross_cycle_pnl_twd": float(row["gross_cycle_pnl_twd"]),
                "total_transaction_cost_twd": float(
                    row["total_transaction_cost_twd"]
                ),
                "net_cycle_pnl_twd": float(row["net_cycle_pnl_twd"]),
                "completed_same_day": str(row["Date"])
                == str(row["terminal_date"]),
                "completed_overnight": str(row["Date"])
                < str(row["terminal_date"]),
                "active_positions_before_event": before_count,
                "active_portfolio_notional_before_event_twd": before_notional,
                "active_same_product_positions_before_event": before_product_count,
                "active_same_product_notional_before_event_twd": (
                    before_product_notional
                ),
                "active_positions_after_event": len(active),
                "active_portfolio_notional_after_event_twd": active_notional,
                "active_same_product_positions_after_event": (
                    active_count_by_product[product]
                ),
                "active_same_product_notional_after_event_twd": (
                    active_notional_by_product[product]
                ),
            }
        )
        event_rows.append(event)

    for row in paths.iter_rows(named=True):
        entry_date = str(row["Date"])
        entry_ns = int(row["position_established_ns"])
        if entry_date != current_entry_date:
            open_ns = local_session_timestamp_ns(
                entry_date,
                config.session_open_local_time,
                config.timezone_name,
            )
            while exit_heap and (exit_heap[0][0], exit_heap[0][1]) < (
                entry_date,
                open_ns,
            ):
                _, _, _, exiting = heapq.heappop(exit_heap)
                emit_exit(exiting)
            opening_carry_products = {
                str(active_row["ValueCode"])
                for active_row in active.values()
                if str(active_row["Date"]) < entry_date
            }
            current_entry_date = entry_date
        entry_key = (entry_date, entry_ns)
        while exit_heap and (exit_heap[0][0], exit_heap[0][1]) < entry_key:
            _, _, _, exiting = heapq.heappop(exit_heap)
            emit_exit(exiting)

        product = str(row["ValueCode"])
        notional = float(row["normalization_notional_twd"])
        before_count = len(active)
        before_notional = active_notional
        before_product_count = active_count_by_product.get(product, 0)
        before_product_notional = active_notional_by_product.get(product, 0.0)
        prospective_notional = before_notional + notional
        prospective_product_notional = before_product_notional + notional
        cutoff_ns = local_session_timestamp_ns(
            entry_date,
            config.entry_cutoff_local_time,
            config.timezone_name,
        )
        admission_ns = int(row["entry_admission_time_ns"])
        cutoff_blocked = admission_ns >= cutoff_ns
        opening_carry_blocked = (
            config.block_new_entries_for_products_held_at_session_open
            and product in opening_carry_products
        )
        portfolio_blocked = (
            prospective_notional > scenario.hard_intraday_cap_twd + 1e-9
        )
        product_blocked = prospective_product_notional > product_cap + 1e-9
        admitted = (
            not cutoff_blocked
            and not opening_carry_blocked
            and not portfolio_blocked
            and not product_blocked
        )
        if cutoff_blocked:
            status = "rejected_entry_cutoff"
        elif opening_carry_blocked:
            status = "rejected_opening_carry_exit_only"
        elif portfolio_blocked and product_blocked:
            status = "rejected_both_caps"
        elif portfolio_blocked:
            status = "rejected_portfolio_cap"
        elif product_blocked:
            status = "rejected_product_cap"
        else:
            status = "accepted"

        if admitted:
            identifier = str(row["policy_path_id"])
            active[identifier] = row
            active_notional += notional
            active_count_by_product[product] = before_product_count + 1
            active_notional_by_product[product] = prospective_product_notional
            accepted.append(row)
            heapq.heappush(
                exit_heap,
                (
                    str(row["terminal_date"]),
                    int(row["exit_decision_time_ns"]),
                    identifier,
                    row,
                ),
            )

        event = common_event(row, "entry_candidate", entry_date, entry_ns)
        event.update(
            {
                "entry_date": entry_date,
                "terminal_date": str(row["terminal_date"]),
                "entry_cutoff_ns": cutoff_ns,
                "entry_admission_time_ns": admission_ns,
                "entry_cutoff_uses_admission_not_establishment": True,
                "entry_admission_status": status,
                "entry_admitted": admitted,
                "entry_cutoff_blocked": cutoff_blocked,
                "opening_carry_exit_only_blocked": opening_carry_blocked,
                "portfolio_cap_blocked": portfolio_blocked,
                "per_product_cap_blocked": product_blocked,
                "requested_entry_one_way_notional_twd": notional,
                "accepted_entry_one_way_turnover_twd": notional if admitted else None,
                "accepted_entry_two_leg_turnover_twd": (
                    float(row["entry_two_leg_turnover_twd"])
                    if admitted
                    else None
                ),
                "released_capacity_notional_twd": None,
                "exit_one_way_turnover_twd": None,
                "exit_two_leg_turnover_twd": None,
                "gross_cycle_pnl_twd": None,
                "total_transaction_cost_twd": None,
                "net_cycle_pnl_twd": None,
                "completed_same_day": None,
                "completed_overnight": None,
                "active_positions_before_event": before_count,
                "active_portfolio_notional_before_event_twd": before_notional,
                "active_same_product_positions_before_event": before_product_count,
                "active_same_product_notional_before_event_twd": (
                    before_product_notional
                ),
                "prospective_portfolio_notional_twd": prospective_notional,
                "prospective_same_product_notional_twd": (
                    prospective_product_notional
                ),
                "active_positions_after_event": len(active),
                "active_portfolio_notional_after_event_twd": active_notional,
                "active_same_product_positions_after_event": (
                    active_count_by_product.get(product, 0)
                ),
                "active_same_product_notional_after_event_twd": (
                    active_notional_by_product.get(product, 0.0)
                ),
            }
        )
        event_rows.append(event)

    while exit_heap:
        _, _, _, exiting = heapq.heappop(exit_heap)
        emit_exit(exiting)
    if active or abs(active_notional) > 1e-8:
        raise AssertionError("priced-path replay ended with active inventory")
    return event_rows, accepted


def _build_daily_rows(
    events: list[dict[str, object]],
    accepted: list[dict[str, object]],
    dates: tuple[str, ...],
    scenario: PortfolioCapScenario,
    config: PortfolioCapBacktestConfig,
    *,
    calendar_supplied: bool,
) -> list[dict[str, object]]:
    events_by_date: dict[str, list[dict[str, object]]] = {date: [] for date in dates}
    for event in events:
        events_by_date[str(event["event_date"])].append(event)

    cumulative_net = 0.0
    running_peak = 0.0
    rows: list[dict[str, object]] = []
    for date in dates:
        open_ns = local_session_timestamp_ns(
            date, config.session_open_local_time, config.timezone_name
        )
        close_ns = local_session_timestamp_ns(
            date, config.session_close_local_time, config.timezone_name
        )
        duration_ns = close_ns - open_ns
        day_events = events_by_date[date]
        entry_events = [
            event for event in day_events if event["event_type"] == "entry_candidate"
        ]
        exit_events = [
            event for event in day_events if event["event_type"] == "position_exit"
        ]

        opening = [
            row
            for row in accepted
            if int(row["position_established_ns"]) < open_ns
            and int(row["exit_decision_time_ns"]) >= open_ns
        ]
        eod = [
            row
            for row in accepted
            if int(row["position_established_ns"]) <= close_ns
            and int(row["exit_decision_time_ns"]) > close_ns
        ]
        overnight_out = [row for row in eod if str(row["terminal_date"]) > date]

        mean_notional_numerator = 0.0
        mean_count_numerator = 0
        for row in accepted:
            overlap_start = max(open_ns, int(row["position_established_ns"]))
            overlap_end = min(close_ns, int(row["exit_decision_time_ns"]))
            if overlap_end > overlap_start:
                overlap = overlap_end - overlap_start
                mean_notional_numerator += (
                    overlap * float(row["normalization_notional_twd"])
                )
                mean_count_numerator += overlap
        mean_notional = mean_notional_numerator / duration_ns
        mean_positions = mean_count_numerator / duration_ns

        active_products: dict[str, float] = {}
        active_counts: dict[str, int] = {}
        for row in opening:
            product = str(row["ValueCode"])
            active_products[product] = active_products.get(product, 0.0) + float(
                row["normalization_notional_twd"]
            )
            active_counts[product] = active_counts.get(product, 0) + 1
        current_notional = sum(active_products.values())
        current_count = sum(active_counts.values())
        peak_notional = current_notional
        peak_count = current_count
        peak_product_notional = max(active_products.values(), default=0.0)
        peak_product_count = max(active_counts.values(), default=0)
        for event in day_events:
            event_ns = int(event["event_timestamp_ns"])
            if not open_ns <= event_ns <= close_ns:
                continue
            product = str(event["ValueCode"])
            notional = float(event["capacity_notional_twd"])
            if event["event_type"] == "entry_candidate" and event["entry_admitted"]:
                current_count += 1
                current_notional += notional
                active_counts[product] = active_counts.get(product, 0) + 1
                active_products[product] = active_products.get(product, 0.0) + notional
            elif event["event_type"] == "position_exit":
                current_count -= 1
                current_notional -= notional
                active_counts[product] -= 1
                active_products[product] -= notional
            peak_count = max(peak_count, current_count)
            peak_notional = max(peak_notional, current_notional)
            peak_product_count = max(
                peak_product_count, max(active_counts.values(), default=0)
            )
            peak_product_notional = max(
                peak_product_notional, max(active_products.values(), default=0.0)
            )

        accepted_entries = [event for event in entry_events if event["entry_admitted"]]
        cutoff_rejects = [
            event
            for event in entry_events
            if event["entry_admission_status"] == "rejected_entry_cutoff"
        ]
        portfolio_rejects = [
            event
            for event in entry_events
            if event["entry_admission_status"] == "rejected_portfolio_cap"
        ]
        product_rejects = [
            event
            for event in entry_events
            if event["entry_admission_status"] == "rejected_product_cap"
        ]
        both_rejects = [
            event
            for event in entry_events
            if event["entry_admission_status"] == "rejected_both_caps"
        ]
        opening_carry_rejects = [
            event
            for event in entry_events
            if event["entry_admission_status"]
            == "rejected_opening_carry_exit_only"
        ]
        gross = _sum_event(exit_events, "gross_cycle_pnl_twd")
        cost = _sum_event(exit_events, "total_transaction_cost_twd")
        net = _sum_event(exit_events, "net_cycle_pnl_twd")
        exit_net = [float(event["net_cycle_pnl_twd"]) for event in exit_events]
        losses = [-value for value in exit_net if value < 0]
        cumulative_net += net
        running_peak = max(running_peak, cumulative_net)
        drawdown = running_peak - cumulative_net
        eod_notional = _sum_rows(eod, "normalization_notional_twd")
        overnight_notional = _sum_rows(
            overnight_out, "normalization_notional_twd"
        )
        overnight_limit = scenario.eod_overnight_limit_twd
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "Date": date,
                "hard_intraday_cap_twd": scenario.hard_intraday_cap_twd,
                "per_product_hard_intraday_cap_twd": (
                    scenario.hard_intraday_cap_twd * config.per_product_fraction
                ),
                "eod_overnight_limit_twd": overnight_limit,
                "session_open_ns": open_ns,
                "entry_cutoff_ns": local_session_timestamp_ns(
                    date, config.entry_cutoff_local_time, config.timezone_name
                ),
                "session_close_ns": close_ns,
                "session_duration_ns": duration_ns,
                "entry_candidates": len(entry_events),
                "accepted_entries": len(accepted_entries),
                "rejected_entries": len(entry_events) - len(accepted_entries),
                "rejected_entry_cutoff": len(cutoff_rejects),
                "rejected_opening_carry_exit_only": len(opening_carry_rejects),
                "rejected_portfolio_cap": len(portfolio_rejects),
                "rejected_product_cap": len(product_rejects),
                "rejected_both_caps": len(both_rejects),
                "rejected_requested_one_way_notional_twd": _sum_event(
                    [event for event in entry_events if not event["entry_admitted"]],
                    "requested_entry_one_way_notional_twd",
                ),
                "accepted_entry_one_way_turnover_twd": _sum_event(
                    accepted_entries, "accepted_entry_one_way_turnover_twd"
                ),
                "accepted_entry_two_leg_turnover_twd": _sum_event(
                    accepted_entries, "accepted_entry_two_leg_turnover_twd"
                ),
                "completed_exits": len(exit_events),
                "same_day_exits": sum(
                    event["completed_same_day"] is True for event in exit_events
                ),
                "overnight_exits": sum(
                    event["completed_overnight"] is True for event in exit_events
                ),
                "released_capacity_notional_twd": _sum_event(
                    exit_events, "released_capacity_notional_twd"
                ),
                "exit_one_way_turnover_twd": _sum_event(
                    exit_events, "exit_one_way_turnover_twd"
                ),
                "exit_two_leg_turnover_twd": _sum_event(
                    exit_events, "exit_two_leg_turnover_twd"
                ),
                "opening_outstanding_positions": len(opening),
                "opening_carry_products": len(
                    {str(row["ValueCode"]) for row in opening}
                ),
                "opening_outstanding_one_way_notional_twd": _sum_rows(
                    opening, "normalization_notional_twd"
                ),
                "intraday_peak_positions": peak_count,
                "intraday_peak_one_way_notional_twd": peak_notional,
                "intraday_time_weighted_mean_positions": mean_positions,
                "intraday_time_weighted_mean_one_way_notional_twd": mean_notional,
                "peak_single_product_positions": peak_product_count,
                "peak_single_product_one_way_notional_twd": peak_product_notional,
                "eod_outstanding_positions": len(eod),
                "eod_outstanding_one_way_notional_twd": eod_notional,
                "overnight_carried_out_positions": len(overnight_out),
                "overnight_carried_out_one_way_notional_twd": overnight_notional,
                "intraday_peak_cap_utilization": (
                    peak_notional / scenario.hard_intraday_cap_twd
                ),
                "intraday_mean_cap_utilization": (
                    mean_notional / scenario.hard_intraday_cap_twd
                ),
                "eod_hard_cap_utilization": (
                    eod_notional / scenario.hard_intraday_cap_twd
                ),
                "entry_capacity_turnover": (
                    _sum_event(
                        accepted_entries, "accepted_entry_one_way_turnover_twd"
                    )
                    / scenario.hard_intraday_cap_twd
                ),
                "exit_capacity_turnover": (
                    _sum_event(exit_events, "exit_one_way_turnover_twd")
                    / scenario.hard_intraday_cap_twd
                ),
                "eod_overnight_limit_usage": (
                    overnight_notional / overnight_limit
                    if overnight_limit is not None
                    else None
                ),
                "eod_overnight_limit_overage_twd": (
                    max(0.0, overnight_notional - overnight_limit)
                    if overnight_limit is not None
                    else None
                ),
                "eod_overnight_limit_breached": (
                    overnight_notional > overnight_limit + 1e-9
                    if overnight_limit is not None
                    else None
                ),
                "realized_gross_pnl_twd": gross,
                "realized_transaction_cost_twd": cost,
                "realized_net_pnl_twd": net,
                "winning_exits": sum(value > 0 for value in exit_net),
                "losing_exits": len(losses),
                "flat_exits": sum(value == 0 for value in exit_net),
                "loss_sum_twd": sum(losses),
                "loss_magnitude_p50_twd": _quantile(losses, 0.50),
                "loss_magnitude_p90_twd": _quantile(losses, 0.90),
                "largest_loss_twd": max(losses) if losses else None,
                "cumulative_realized_net_pnl_twd": cumulative_net,
                "realized_drawdown_twd": drawdown,
                "session_calendar_explicitly_supplied": calendar_supplied,
                "hard_intraday_cap_enforced_on_admission": True,
                "eod_overnight_limit_enforced_on_admission": False,
                "opening_carry_product_exit_only_policy": (
                    config.block_new_entries_for_products_held_at_session_open
                ),
                "notional_basis": "one_way_spot_entry_notional_twd",
            }
        )
    return rows


def _build_summary_row(
    events: list[dict[str, object]],
    accepted: list[dict[str, object]],
    daily: list[dict[str, object]],
    scenario: PortfolioCapScenario,
    config: PortfolioCapBacktestConfig,
    *,
    calendar_supplied: bool,
) -> dict[str, object]:
    entries = [event for event in events if event["event_type"] == "entry_candidate"]
    accepted_entries = [event for event in entries if event["entry_admitted"]]
    exits = [event for event in events if event["event_type"] == "position_exit"]
    net_values = [float(event["net_cycle_pnl_twd"]) for event in exits]
    losses = [-value for value in net_values if value < 0]
    daily_net = [float(row["realized_net_pnl_twd"]) for row in daily]
    daily_net_mean = _mean(daily_net)
    daily_net_std = _sample_std(daily_net)
    entry_turnover = _sum_event(
        accepted_entries, "accepted_entry_one_way_turnover_twd"
    )
    exit_turnover = _sum_event(exits, "exit_one_way_turnover_twd")
    sessions = len(daily)
    mean_active_notional = _mean(
        [
            float(row["intraday_time_weighted_mean_one_way_notional_twd"])
            for row in daily
        ]
    )
    max_drawdown = max(float(row["realized_drawdown_twd"]) for row in daily)
    accepted_products_by_date: dict[str, dict[str, float]] = {
        str(row["Date"]): {} for row in daily
    }
    for event in accepted_entries:
        date = str(event["entry_date"])
        product = str(event["ValueCode"])
        by_product = accepted_products_by_date[date]
        by_product[product] = by_product.get(product, 0.0) + float(
            event["accepted_entry_one_way_turnover_twd"]
        )
    distinct_products_all_days = [
        len(accepted_products_by_date[str(row["Date"])]) for row in daily
    ]
    active_product_days = [
        by_product
        for by_product in accepted_products_by_date.values()
        if by_product
    ]
    max_product_entry_shares = [
        max(by_product.values()) / sum(by_product.values())
        for by_product in active_product_days
    ]
    annualized_net_twd_240 = daily_net_mean * 240.0
    return {
        "scenario_id": scenario.scenario_id,
        "hard_intraday_cap_twd": scenario.hard_intraday_cap_twd,
        "per_product_cap_fraction": config.per_product_fraction,
        "per_product_hard_intraday_cap_twd": (
            scenario.hard_intraday_cap_twd * config.per_product_fraction
        ),
        "eod_overnight_limit_twd": scenario.eod_overnight_limit_twd,
        "session_count": sessions,
        "candidate_paths": len(entries),
        "accepted_paths": len(accepted_entries),
        "rejected_paths": len(entries) - len(accepted_entries),
        "acceptance_rate": len(accepted_entries) / len(entries),
        "rejected_entry_cutoff": _count_status(entries, "rejected_entry_cutoff"),
        "rejected_opening_carry_exit_only": _count_status(
            entries, "rejected_opening_carry_exit_only"
        ),
        "rejected_portfolio_cap": _count_status(entries, "rejected_portfolio_cap"),
        "rejected_product_cap": _count_status(entries, "rejected_product_cap"),
        "rejected_both_caps": _count_status(entries, "rejected_both_caps"),
        "completed_exits": len(exits),
        "same_day_completed_paths": sum(
            event["completed_same_day"] is True for event in exits
        ),
        "overnight_completed_paths": sum(
            event["completed_overnight"] is True for event in exits
        ),
        "accepted_entry_one_way_turnover_twd": entry_turnover,
        "accepted_entry_two_leg_turnover_twd": _sum_event(
            accepted_entries, "accepted_entry_two_leg_turnover_twd"
        ),
        "exit_one_way_turnover_twd": exit_turnover,
        "exit_two_leg_turnover_twd": _sum_event(
            exits, "exit_two_leg_turnover_twd"
        ),
        "entry_notional_cap_turns": entry_turnover
        / scenario.hard_intraday_cap_twd,
        "exit_notional_cap_turns": exit_turnover / scenario.hard_intraday_cap_twd,
        "mean_daily_entry_capacity_turnover": entry_turnover
        / scenario.hard_intraday_cap_twd
        / sessions,
        "mean_daily_exit_capacity_turnover": exit_turnover
        / scenario.hard_intraday_cap_twd
        / sessions,
        "peak_concurrent_positions": max(
            int(row["intraday_peak_positions"]) for row in daily
        ),
        "peak_intraday_one_way_notional_twd": max(
            float(row["intraday_peak_one_way_notional_twd"]) for row in daily
        ),
        "mean_daily_intraday_peak_one_way_notional_twd": _mean(
            [float(row["intraday_peak_one_way_notional_twd"]) for row in daily]
        ),
        "mean_time_weighted_intraday_one_way_notional_twd": _mean(
            [
                float(row["intraday_time_weighted_mean_one_way_notional_twd"])
                for row in daily
            ]
        ),
        "mean_intraday_peak_cap_utilization": _mean(
            [float(row["intraday_peak_cap_utilization"]) for row in daily]
        ),
        "mean_time_weighted_intraday_cap_utilization": _mean(
            [float(row["intraday_mean_cap_utilization"]) for row in daily]
        ),
        "mean_daily_active_one_way_notional_twd": mean_active_notional,
        "max_daily_active_one_way_notional_twd": max(
            float(row["intraday_peak_one_way_notional_twd"]) for row in daily
        ),
        "peak_eod_outstanding_one_way_notional_twd": max(
            float(row["eod_outstanding_one_way_notional_twd"]) for row in daily
        ),
        "mean_eod_outstanding_one_way_notional_twd": _mean(
            [float(row["eod_outstanding_one_way_notional_twd"]) for row in daily]
        ),
        "p50_eod_outstanding_one_way_notional_twd": _quantile(
            [float(row["eod_outstanding_one_way_notional_twd"]) for row in daily],
            0.50,
        ),
        "p90_eod_outstanding_one_way_notional_twd": _quantile(
            [float(row["eod_outstanding_one_way_notional_twd"]) for row in daily],
            0.90,
        ),
        "peak_overnight_carried_one_way_notional_twd": max(
            float(row["overnight_carried_out_one_way_notional_twd"])
            for row in daily
        ),
        "overnight_notional_days_twd": sum(
            float(row["overnight_carried_out_one_way_notional_twd"])
            for row in daily
        ),
        "days_with_overnight_carry": sum(
            int(row["overnight_carried_out_positions"]) > 0 for row in daily
        ),
        "days_breaching_eod_overnight_limit": (
            sum(row["eod_overnight_limit_breached"] is True for row in daily)
            if scenario.eod_overnight_limit_twd is not None
            else None
        ),
        "realized_gross_pnl_twd": _sum_event(exits, "gross_cycle_pnl_twd"),
        "realized_transaction_cost_twd": _sum_event(
            exits, "total_transaction_cost_twd"
        ),
        "realized_net_pnl_twd": _sum_event(exits, "net_cycle_pnl_twd"),
        "total_net_to_max_realized_drawdown": (
            _sum_event(exits, "net_cycle_pnl_twd") / max_drawdown
            if max_drawdown > 0
            else None
        ),
        "daily_win_rate_including_zero_days": (
            sum(value > 0 for value in daily_net) / sessions
        ),
        "positive_realized_days": sum(value > 0 for value in daily_net),
        "negative_realized_days": sum(value < 0 for value in daily_net),
        "zero_realized_days": sum(value == 0 for value in daily_net),
        "annualized_sharpe_daily_net_252": (
            daily_net_mean / daily_net_std * math.sqrt(252.0)
            if daily_net_std > 0
            else 0.0
        ),
        "annualized_net_pnl_twd_240": annualized_net_twd_240,
        "annualized_return_on_mean_intraday_notional_240": (
            annualized_net_twd_240 / mean_active_notional
            if mean_active_notional > 0
            else None
        ),
        "annualized_return_on_hard_intraday_cap_240": (
            annualized_net_twd_240 / scenario.hard_intraday_cap_twd
        ),
        "net_pnl_on_entry_turnover_bp": (
            _sum_event(exits, "net_cycle_pnl_twd") / entry_turnover * 10_000.0
            if entry_turnover
            else None
        ),
        "winning_exits": sum(value > 0 for value in net_values),
        "losing_exits": len(losses),
        "flat_exits": sum(value == 0 for value in net_values),
        "same_day_close_ratio": (
            sum(event["completed_same_day"] is True for event in exits) / len(exits)
            if exits
            else None
        ),
        "mean_distinct_accepted_products_per_session": _mean(
            distinct_products_all_days
        ),
        "mean_distinct_accepted_products_per_active_entry_day": (
            _mean([len(value) for value in active_product_days])
            if active_product_days
            else 0.0
        ),
        "mean_max_single_product_entry_notional_share": (
            _mean(max_product_entry_shares) if max_product_entry_shares else None
        ),
        "win_rate_excluding_flat": (
            sum(value > 0 for value in net_values)
            / (sum(value > 0 for value in net_values) + len(losses))
            if any(value != 0 for value in net_values)
            else None
        ),
        "loss_sum_twd": sum(losses),
        "loss_magnitude_p50_twd": _quantile(losses, 0.50),
        "loss_magnitude_p90_twd": _quantile(losses, 0.90),
        "loss_magnitude_p95_twd": _quantile(losses, 0.95),
        "largest_loss_twd": max(losses) if losses else None,
        "daily_realized_net_p05_twd": _quantile(daily_net, 0.05),
        "daily_realized_net_p50_twd": _quantile(daily_net, 0.50),
        "daily_realized_net_p95_twd": _quantile(daily_net, 0.95),
        "worst_daily_realized_net_twd": min(daily_net),
        "max_realized_drawdown_twd": max_drawdown,
        "session_calendar_explicitly_supplied": calendar_supplied,
        "terminal_cashflows_point_identified": len(exits) == len(accepted),
        "hard_intraday_cap_enforced_on_admission": True,
        "eod_overnight_limit_enforced_on_admission": False,
        "opening_carry_product_exit_only_policy": (
            config.block_new_entries_for_products_held_at_session_open
        ),
        "entry_cutoff_is_exclusive": True,
        "equal_timestamp_tie_handling": "entry_before_exit",
        "notional_basis": "one_way_spot_entry_notional_twd",
    }


def _validate_result(
    result: PortfolioCapBacktestResult,
    source: pl.DataFrame,
    config: PortfolioCapBacktestConfig,
    dates: tuple[str, ...],
) -> None:
    if result.summary.height != len(config.scenarios):
        raise AssertionError("summary scenario count differs")
    if result.daily.height != len(config.scenarios) * len(dates):
        raise AssertionError("daily scenario-date grid is incomplete")
    entry_events = result.events.filter(pl.col("event_type") == "entry_candidate")
    if entry_events.height != source.height * len(config.scenarios):
        raise AssertionError("entry event grid is incomplete")
    invalid_caps = result.events.filter(
        pl.col("active_portfolio_notional_after_event_twd")
        > pl.col("hard_intraday_cap_twd") + 1e-6
    )
    if invalid_caps.height:
        raise AssertionError("hard intraday portfolio cap was exceeded")
    invalid_products = result.events.filter(
        pl.col("active_same_product_notional_after_event_twd")
        > pl.col("per_product_hard_intraday_cap_twd") + 1e-6
    )
    if invalid_products.height:
        raise AssertionError("hard intraday product cap was exceeded")
    accepted = entry_events.filter(pl.col("entry_admitted"))
    exits = result.events.filter(pl.col("event_type") == "position_exit")
    if exits.height != accepted.height:
        raise AssertionError("accepted priced paths were not all released")
    if result.summary.filter(
        (pl.col("hard_intraday_cap_enforced_on_admission") != True)  # noqa: E712
        | (pl.col("eod_overnight_limit_enforced_on_admission") != False)  # noqa: E712
    ).height:
        raise AssertionError("intraday/EOD cap semantics changed")
    if result.summary.filter(
        pl.col("opening_carry_product_exit_only_policy")
        != config.block_new_entries_for_products_held_at_session_open
    ).height:
        raise AssertionError("opening-carry exit-only policy metadata changed")
    if config.block_new_entries_for_products_held_at_session_open and entry_events.filter(
        pl.col("entry_admitted") & pl.col("opening_carry_exit_only_blocked")
    ).height:
        raise AssertionError("opening-carry product admitted a new entry")


def _event_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.from_dicts(
        rows,
        infer_schema_length=None,
        schema_overrides={
            "eod_overnight_limit_twd": pl.Float64,
            "entry_cutoff_ns": pl.Int64,
            "entry_admitted": pl.Boolean,
            "entry_cutoff_blocked": pl.Boolean,
            "opening_carry_exit_only_blocked": pl.Boolean,
            "portfolio_cap_blocked": pl.Boolean,
            "per_product_cap_blocked": pl.Boolean,
            "requested_entry_one_way_notional_twd": pl.Float64,
            "accepted_entry_one_way_turnover_twd": pl.Float64,
            "accepted_entry_two_leg_turnover_twd": pl.Float64,
            "released_capacity_notional_twd": pl.Float64,
            "exit_one_way_turnover_twd": pl.Float64,
            "exit_two_leg_turnover_twd": pl.Float64,
            "gross_cycle_pnl_twd": pl.Float64,
            "total_transaction_cost_twd": pl.Float64,
            "net_cycle_pnl_twd": pl.Float64,
            "completed_same_day": pl.Boolean,
            "completed_overnight": pl.Boolean,
            "prospective_portfolio_notional_twd": pl.Float64,
            "prospective_same_product_notional_twd": pl.Float64,
        },
    ).sort(["scenario_id", "event_sequence"])


def _daily_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.from_dicts(
        rows,
        infer_schema_length=None,
        schema_overrides={
            "eod_overnight_limit_twd": pl.Float64,
            "eod_overnight_limit_usage": pl.Float64,
            "eod_overnight_limit_overage_twd": pl.Float64,
            "eod_overnight_limit_breached": pl.Boolean,
            "loss_magnitude_p50_twd": pl.Float64,
            "loss_magnitude_p90_twd": pl.Float64,
            "largest_loss_twd": pl.Float64,
        },
    ).sort(["scenario_id", "Date"])


def _summary_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.from_dicts(
        rows,
        infer_schema_length=None,
        schema_overrides={
            "eod_overnight_limit_twd": pl.Float64,
            "days_breaching_eod_overnight_limit": pl.Int64,
            "net_pnl_on_entry_turnover_bp": pl.Float64,
            "total_net_to_max_realized_drawdown": pl.Float64,
            "annualized_return_on_mean_intraday_notional_240": pl.Float64,
            "same_day_close_ratio": pl.Float64,
            "mean_max_single_product_entry_notional_share": pl.Float64,
            "win_rate_excluding_flat": pl.Float64,
            "loss_magnitude_p50_twd": pl.Float64,
            "loss_magnitude_p90_twd": pl.Float64,
            "loss_magnitude_p95_twd": pl.Float64,
            "largest_loss_twd": pl.Float64,
        },
    ).sort("hard_intraday_cap_twd")


def _parse_date(value: str) -> datetime:
    try:
        parsed = datetime.strptime(str(value), "%Y%m%d")
    except ValueError as error:
        raise ValueError("dates must be valid YYYYMMDD") from error
    return parsed


def _canonical_date(value: str) -> str:
    parsed = _parse_date(value)
    text = parsed.strftime("%Y%m%d")
    if text != value:
        raise ValueError("dates must use canonical YYYYMMDD")
    return text


def _local_date_for_ns(timestamp_ns: int, zone: ZoneInfo) -> str:
    seconds = timestamp_ns // 1_000_000_000
    return datetime.fromtimestamp(seconds, timezone.utc).astimezone(zone).strftime(
        "%Y%m%d"
    )


def _integer_ns(value: object, label: str, identifier: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{identifier}: {label} timestamp must be non-negative int")
    return value


def _positive_finite(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _sum_event(rows: Iterable[Mapping[str, object]], column: str) -> float:
    return sum(float(row[column]) for row in rows if row.get(column) is not None)


def _sum_rows(rows: Iterable[Mapping[str, object]], column: str) -> float:
    return sum(float(row[column]) for row in rows)


def _count_status(rows: Iterable[Mapping[str, object]], status: str) -> int:
    return sum(row["entry_admission_status"] == status for row in rows)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(
        sum((float(value) - mean) ** 2 for value in values) / (len(values) - 1)
    )


def _quantile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


__all__ = [
    "DEFAULT_HARD_INTRADAY_CAPS_TWD",
    "PortfolioCapBacktestConfig",
    "PortfolioCapBacktestResult",
    "PortfolioCapScenario",
    "backtest_priced_paths",
    "default_cap_scenarios",
    "local_session_timestamp_ns",
]
