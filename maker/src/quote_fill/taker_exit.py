"""Conservative executable exit baseline after an entry maker fill.

The entry studies establish the same economic position through two routes:
long spot and short one stock-futures contract.  This module deliberately
avoids assuming that a hypothetical exit maker order filled.  Instead it
labels the first *one-second decision epoch* where the whole position can be
closed by selling spot at the causal bid and buying futures at the causal
executable ask.

The result is a conservative taker/taker exit benchmark.  It is not a maker
exit model and it does not turn the one-second fair grid into raw sub-second
execution.  A position that does not reach its same-day cash target remains a
real position: it is carried to the next session (when requested), never
silently treated as a censor or a zero-PnL outcome.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import polars as pl

from ..common.paths import MAKER_ROOT
from ..quote_width.daily_facts import (
    DEFAULT_DAILY_ROOT,
    completed_artifact_paths,
)


ENTRY_ROUTES = (
    "future_ask_spot_taker",
    "spot_bid_future_taker",
)
SAMPLE_COLUMNS: Mapping[str, str] = {
    "base": "eligible_base",
    "fresh_1000ms": "eligible_1000ms",
}
STATUS_VALUES = (
    "same_day_target_exit",
    "expiry_day_forced_exit",
    "next_session_forced_exit",
    "unresolved_no_next_book",
)
_NS_PER_SECOND = 1_000_000_000
DEFAULT_QUOTE_FILL_DIR = MAKER_ROOT / "data" / "quote_fill"


@dataclass(frozen=True)
class TakerExitStudyResult:
    positions: pl.DataFrame
    labels: pl.DataFrame
    summary: pl.DataFrame


@dataclass(frozen=True)
class TakerExitConfig:
    """Frozen policy grid for the conservative exit benchmark."""

    gross_profit_targets_twd: tuple[float, ...] = (0.0, 200.0, 500.0, 1000.0)
    samples: tuple[str, ...] = ("base", "fresh_1000ms")
    carry_to_next_session: bool = True
    force_exit_on_expiry: bool = True
    next_session_start_seconds: int = 300
    policy_version: str = "one_second_taker_exit_v1"

    def validate(self) -> None:
        if not self.gross_profit_targets_twd:
            raise ValueError("gross_profit_targets_twd must not be empty")
        if len(set(self.gross_profit_targets_twd)) != len(
            self.gross_profit_targets_twd
        ):
            raise ValueError("gross profit targets must be unique")
        if any(
            isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in self.gross_profit_targets_twd
        ):
            raise ValueError("gross profit targets must be finite numbers")
        unknown = sorted(set(self.samples) - set(SAMPLE_COLUMNS))
        if unknown:
            raise ValueError(f"unknown taker-exit samples: {unknown}")
        if not self.samples or len(set(self.samples)) != len(self.samples):
            raise ValueError("samples must be non-empty and unique")
        if self.next_session_start_seconds < 0:
            raise ValueError("next_session_start_seconds must be non-negative")


@dataclass(frozen=True)
class ExecutionCostProfile:
    """Versioned cash-cost assumptions applied once to actual fills.

    Rates are deliberately caller-supplied.  Brokerage discounts, tax
    eligibility and financing terms are account/date specific and must not be
    hidden in the label builder.
    """

    profile_version: str
    spot_commission_rate: float
    spot_daytrade_sell_tax_rate: float
    spot_regular_sell_tax_rate: float
    futures_tax_rate: float
    futures_commission_twd_per_contract_side: float
    overnight_carry_cost_twd_per_session: float = 0.0
    complete_for_net_ev: bool = False

    def validate(self) -> None:
        if not self.profile_version:
            raise ValueError("profile_version must not be empty")
        for name, value in asdict(self).items():
            if name in {"profile_version", "complete_for_net_ev"}:
                continue
            if (
                isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")


def build_physical_entry_positions(
    order_aliases: pl.DataFrame,
    hedge_facts: pl.DataFrame,
) -> pl.DataFrame:
    """Collapse policy aliases to one established physical entry position."""

    _require(
        order_aliases,
        {
            "Date",
            "ValueCode",
            "QuoteCode",
            "route",
            "raw_order_fact_id",
            "boundary_quantile",
            "target_price",
            "full_fill",
        },
        "order aliases",
    )
    _require(
        hedge_facts,
        {
            "raw_order_fact_id",
            "status",
            "executable_vwap_price",
            "decision_time_ns",
            "contract_size_shares",
            "depth_shortfall",
        },
        "hedge facts",
    )
    if hedge_facts.select("raw_order_fact_id").n_unique() != hedge_facts.height:
        raise ValueError("hedge facts must be unique by raw_order_fact_id")

    full = order_aliases.filter(pl.col("full_fill").fill_null(False))
    if full.is_empty():
        return pl.DataFrame(schema=_position_schema())
    consistency = full.group_by("raw_order_fact_id").agg(
        *[
            pl.col(column).n_unique().alias(f"_{column}_n")
            for column in (
                "Date",
                "ValueCode",
                "QuoteCode",
                "route",
                "target_price",
            )
        ]
    )
    bad_consistency = consistency.select(
        pl.any_horizontal(
            *[
                pl.col(name) != 1
                for name in consistency.columns
                if name != "raw_order_fact_id"
            ]
        ).alias("bad")
    )
    if bad_consistency["bad"].any():
        raise ValueError("policy aliases disagree on physical entry identity")

    physical = full.group_by("raw_order_fact_id").agg(
        pl.col("Date").first(),
        pl.col("ValueCode").first(),
        pl.col("QuoteCode").first(),
        pl.col("route").first(),
        pl.col("target_price").first().alias("maker_fill_price"),
        pl.col("boundary_quantile").sort().unique().alias(
            "boundary_quantile_aliases"
        ),
        pl.len().alias("policy_alias_count"),
    ).join(
        hedge_facts,
        on="raw_order_fact_id",
        how="left",
        validate="1:1",
    )
    missing = physical.filter(
        pl.col("status").is_null()
        | pl.col("decision_time_ns").is_null()
        | pl.col("contract_size_shares").is_null()
    )
    if missing.height:
        raise ValueError("every physical full fill must resolve to a hedge fact")
    supported = physical.filter(
        (pl.col("status") == "executable")
        & (pl.col("depth_shortfall").fill_null(0) == 0)
        & pl.col("executable_vwap_price").is_finite()
        & (pl.col("executable_vwap_price") > 0)
        & (pl.col("contract_size_shares") > 0)
        & pl.col("route").is_in(ENTRY_ROUTES)
    ).with_columns(
        pl.when(pl.col("route") == "future_ask_spot_taker")
        .then(pl.col("executable_vwap_price"))
        .otherwise(pl.col("maker_fill_price"))
        .alias("entry_spot_price"),
        pl.when(pl.col("route") == "future_ask_spot_taker")
        .then(pl.col("maker_fill_price"))
        .otherwise(pl.col("executable_vwap_price"))
        .alias("entry_future_price"),
    )
    return supported.select(
        "raw_order_fact_id",
        "Date",
        "ValueCode",
        "QuoteCode",
        "route",
        "boundary_quantile_aliases",
        "policy_alias_count",
        pl.col("decision_time_ns").alias("position_established_ns"),
        pl.col("contract_size_shares").cast(pl.Int64).alias(
            "contract_size_shares"
        ),
        "entry_spot_price",
        "entry_future_price",
    ).sort(["Date", "ValueCode", "position_established_ns"])


def label_taker_exit_paths(
    positions: pl.DataFrame,
    causal_fair: pl.DataFrame,
    sessions: Sequence[str] | Iterable[str],
    config: TakerExitConfig = TakerExitConfig(),
) -> pl.DataFrame:
    """Label mutually exclusive same-day/next-session taker exit branches."""

    config.validate()
    sessions = _normalise_sessions(sessions)
    _require(
        positions,
        set(_position_schema()),
        "physical entry positions",
    )
    _require(
        causal_fair,
        {
            "Date",
            "ValueCode",
            "QuoteCode",
            "timestamp",
            "seconds_from_open",
            "spot_bid",
            "spot_bid_lots",
            "fut_exec_ask",
            "fut_exec_ask_lots",
            "contract_size",
            "end_date",
            *SAMPLE_COLUMNS.values(),
        },
        "causal fair",
    )
    if positions.is_empty():
        return pl.DataFrame(schema=_label_schema())
    session_index = {date: index for index, date in enumerate(sessions)}
    unknown = sorted(set(positions["Date"].to_list()) - set(sessions))
    if unknown:
        raise ValueError(f"position dates absent from session calendar: {unknown[:5]}")

    fair = causal_fair.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("timestamp").dt.epoch("ns").alias("_timestamp_ns"),
    ).sort(["Date", "ValueCode", "_timestamp_ns"])
    paths = {
        (str(date), str(value_code), str(quote_code)): group
        for (date, value_code, quote_code), group in fair.group_by(
            "Date", "ValueCode", "QuoteCode"
        )
    }
    records: list[dict[str, object]] = []
    for position in positions.iter_rows(named=True):
        date = str(position["Date"])
        key = (date, str(position["ValueCode"]), str(position["QuoteCode"]))
        same_day = paths.get(key, pl.DataFrame())
        next_date = _next_session(date, sessions, session_index)
        next_key = (
            next_date,
            str(position["ValueCode"]),
            str(position["QuoteCode"]),
        ) if next_date is not None else None
        next_day = paths.get(next_key, pl.DataFrame()) if next_key else pl.DataFrame()
        expiry_day = _is_expiry_day(same_day, date)
        for sample in config.samples:
            day_path = _executable_close_path(
                same_day,
                position,
                eligibility_column=SAMPLE_COLUMNS[sample],
                earliest_ns=_next_full_second_ns(
                    int(position["position_established_ns"])
                ),
                minimum_seconds_from_open=0,
            )
            carry_path = _executable_close_path(
                next_day,
                position,
                eligibility_column=SAMPLE_COLUMNS[sample],
                earliest_ns=0,
                minimum_seconds_from_open=config.next_session_start_seconds,
            )
            max_row = _max_gross_row(day_path)
            eod_row = day_path.row(-1, named=True) if day_path.height else None
            for target in config.gross_profit_targets_twd:
                hit = day_path.filter(pl.col("gross_cash_pnl_twd") >= float(target))
                if hit.height:
                    terminal = hit.row(0, named=True)
                    status = "same_day_target_exit"
                    exit_date = date
                    overnight_sessions = 0
                elif config.force_exit_on_expiry and expiry_day and eod_row is not None:
                    terminal = eod_row
                    status = "expiry_day_forced_exit"
                    exit_date = date
                    overnight_sessions = 0
                elif config.carry_to_next_session and carry_path.height:
                    terminal = carry_path.row(0, named=True)
                    status = "next_session_forced_exit"
                    exit_date = next_date
                    overnight_sessions = 1
                else:
                    terminal = None
                    status = "unresolved_no_next_book"
                    exit_date = next_date
                    overnight_sessions = None
                if terminal is not None:
                    unresolved_reason = None
                elif not config.carry_to_next_session:
                    unresolved_reason = "carry_disabled"
                elif expiry_day:
                    unresolved_reason = "expiry_day_no_eligible_force_exit_book"
                elif next_date is None:
                    unresolved_reason = "no_next_session_in_calendar"
                elif next_key not in paths:
                    unresolved_reason = "exact_contract_absent_next_session"
                else:
                    unresolved_reason = "no_eligible_next_session_close"
                records.append(
                    _label_record(
                        position,
                        sample=sample,
                        target=float(target),
                        entry_contract_expiry_day=expiry_day,
                        status=status,
                        terminal=terminal,
                        exit_date=exit_date,
                        overnight_sessions=overnight_sessions,
                        unresolved_reason=unresolved_reason,
                        max_row=max_row,
                        eod_row=eod_row,
                        config=config,
                    )
                )
    return pl.from_dicts(
        records, schema=_label_schema(), infer_schema_length=None
    ).sort(
        [
            "Date",
            "ValueCode",
            "route",
            "raw_order_fact_id",
            "sample",
            "gross_profit_target_twd",
        ]
    )


def apply_execution_cost_profile(
    labels: pl.DataFrame,
    profile: ExecutionCostProfile,
) -> pl.DataFrame:
    """Apply fees/taxes to resolved paths without altering path labels."""

    profile.validate()
    _require(
        labels,
        {
            "status",
            "contract_size_shares",
            "entry_spot_price",
            "entry_future_price",
            "exit_spot_price",
            "exit_future_price",
            "gross_cash_pnl_twd",
            "overnight_sessions",
        },
        "taker exit labels",
    )
    resolved = pl.col("status").is_in(
        [
            "same_day_target_exit",
            "expiry_day_forced_exit",
            "next_session_forced_exit",
        ]
    )
    shares = pl.col("contract_size_shares").cast(pl.Float64)
    spot_commission = (
        shares
        * (pl.col("entry_spot_price") + pl.col("exit_spot_price"))
        * profile.spot_commission_rate
    )
    spot_tax_rate = pl.when(
        pl.col("status").is_in(
            ["same_day_target_exit", "expiry_day_forced_exit"]
        )
    ).then(
        pl.lit(profile.spot_daytrade_sell_tax_rate)
    ).otherwise(pl.lit(profile.spot_regular_sell_tax_rate))
    spot_tax = shares * pl.col("exit_spot_price") * spot_tax_rate
    futures_tax = (
        shares
        * (pl.col("entry_future_price") + pl.col("exit_future_price"))
        * profile.futures_tax_rate
    )
    futures_commission = pl.lit(
        2.0 * profile.futures_commission_twd_per_contract_side
    )
    carry = (
        pl.col("overnight_sessions").cast(pl.Float64)
        * profile.overnight_carry_cost_twd_per_session
    )
    return labels.with_columns(
        pl.when(resolved).then(spot_commission).otherwise(None).alias(
            "spot_commission_twd"
        ),
        pl.when(resolved).then(spot_tax).otherwise(None).alias("spot_tax_twd"),
        pl.when(resolved).then(futures_tax).otherwise(None).alias(
            "futures_tax_twd"
        ),
        pl.when(resolved).then(futures_commission).otherwise(None).alias(
            "futures_commission_twd"
        ),
        pl.when(resolved).then(carry).otherwise(None).alias(
            "overnight_carry_cost_twd"
        ),
        pl.lit(profile.profile_version).alias("cost_profile_version"),
        pl.lit(profile.complete_for_net_ev).alias("cost_profile_complete"),
    ).with_columns(
        pl.when(resolved)
        .then(
            pl.sum_horizontal(
                "spot_commission_twd",
                "spot_tax_twd",
                "futures_tax_twd",
                "futures_commission_twd",
                "overnight_carry_cost_twd",
            )
        )
        .otherwise(None)
        .alias("total_execution_cost_twd")
    ).with_columns(
        pl.when(resolved)
        .then(pl.col("gross_cash_pnl_twd") - pl.col("total_execution_cost_twd"))
        .otherwise(None)
        .alias("net_cash_pnl_twd"),
        resolved.alias("terminal_cashflow_known"),
    )


def summarize_taker_exit_paths(labels: pl.DataFrame) -> pl.DataFrame:
    """Summarize branch probabilities conditional on filled+hedged positions."""

    _require(
        labels,
        {
            "ValueCode",
            "route",
            "sample",
            "gross_profit_target_twd",
            "status",
            "gross_cash_pnl_twd",
        },
        "taker exit labels",
    )
    keys = ["ValueCode", "route", "sample", "gross_profit_target_twd"]
    has_cost = "net_cash_pnl_twd" in labels.columns
    aggregations: list[pl.Expr] = [
        pl.len().alias("filled_hedged_positions"),
        (pl.col("status") == "same_day_target_exit").sum().alias(
            "same_day_target_exits"
        ),
        (pl.col("status") == "expiry_day_forced_exit").sum().alias(
            "expiry_day_forced_exits"
        ),
        (pl.col("status") == "next_session_forced_exit").sum().alias(
            "next_session_forced_exits"
        ),
        (pl.col("status") == "unresolved_no_next_book").sum().alias(
            "unresolved_positions"
        ),
        pl.col("gross_cash_pnl_twd").mean().alias("mean_gross_cash_pnl_twd"),
        pl.col("gross_cash_pnl_twd").median().alias("median_gross_cash_pnl_twd"),
        pl.col("holding_seconds").median().alias("holding_seconds_p50"),
        pl.col("max_same_day_taker_gross_twd").median().alias(
            "max_same_day_taker_gross_twd_p50"
        ),
    ]
    if has_cost:
        aggregations.extend(
            [
                pl.col("net_cash_pnl_twd").mean().alias(
                    "mean_net_cash_pnl_twd"
                ),
                pl.col("net_cash_pnl_twd").median().alias(
                    "median_net_cash_pnl_twd"
                ),
                pl.col("terminal_cashflow_known").sum().alias(
                    "terminal_cashflow_known_positions"
                ),
                pl.col("cost_profile_version").n_unique().alias(
                    "cost_profile_versions"
                ),
                pl.col("cost_profile_complete").all().alias(
                    "cost_profile_complete"
                ),
            ]
        )
    result = labels.group_by(keys).agg(*aggregations).with_columns(
        (
            pl.col("same_day_target_exits")
            / pl.col("filled_hedged_positions")
        ).alias("p_same_day_target_exit_given_filled_hedged"),
        (
            pl.col("next_session_forced_exits")
            / pl.col("filled_hedged_positions")
        ).alias("p_next_session_forced_exit_given_filled_hedged"),
        (
            pl.col("expiry_day_forced_exits")
            / pl.col("filled_hedged_positions")
        ).alias("p_expiry_day_forced_exit_given_filled_hedged"),
        (
            pl.col("unresolved_positions")
            / pl.col("filled_hedged_positions")
        ).alias("p_unresolved_given_filled_hedged"),
    )
    if has_cost:
        result = result.with_columns(
            (
                (pl.col("unresolved_positions") == 0)
                & (
                    pl.col("terminal_cashflow_known_positions")
                    == pl.col("filled_hedged_positions")
                )
                & (pl.col("cost_profile_versions") == 1)
                & pl.col("cost_profile_complete")
            ).alias("conditional_exit_ev_complete")
        )
    else:
        result = result.with_columns(
            pl.lit(False).alias("conditional_exit_ev_complete")
        )
    return result.sort(keys)


def run_taker_exit_study(
    *,
    quote_fill_dir: Path = DEFAULT_QUOTE_FILL_DIR,
    daily_root: Path = DEFAULT_DAILY_ROOT,
    sessions_path: Path | None = None,
    config: TakerExitConfig = TakerExitConfig(),
    cost_profile: ExecutionCostProfile | None = None,
    output_dir: Path | None = None,
) -> TakerExitStudyResult:
    """Run the benchmark from persisted entry/hedge facts.

    Only product-days containing an established position and their immediate
    next trading sessions are read from the large daily fact store.
    """

    quote_fill_dir = Path(quote_fill_dir)
    sessions_path = sessions_path or (Path(daily_root).parent / "sessions.txt")
    sessions = [
        line.strip()
        for line in Path(sessions_path).read_text().splitlines()
        if line.strip()
    ]
    positions = build_physical_entry_positions(
        pl.read_parquet(quote_fill_dir / "order_aliases.parquet"),
        pl.read_parquet(quote_fill_dir / "hedge_facts.parquet"),
    )
    required_dates = set(positions["Date"].to_list())
    index = {date: offset for offset, date in enumerate(sessions)}
    for date in tuple(required_dates):
        if date not in index:
            raise ValueError(f"position date absent from sessions: {date}")
        offset = index[date] + 1
        if offset < len(sessions):
            required_dates.add(sessions[offset])
    value_codes = positions["ValueCode"].unique().to_list()
    paths = {
        path.parent.name.removeprefix("Date="): path
        for path in completed_artifact_paths(
            Path(daily_root), "causal_fair.parquet"
        )
    }
    missing = sorted(required_dates - set(paths))
    if missing:
        raise FileNotFoundError(f"missing causal fair partitions: {missing[:5]}")
    columns = [
        "Date",
        "ValueCode",
        "QuoteCode",
        "timestamp",
        "seconds_from_open",
        "spot_bid",
        "spot_bid_lots",
        "fut_exec_ask",
        "fut_exec_ask_lots",
        "contract_size",
        "end_date",
        *SAMPLE_COLUMNS.values(),
    ]
    causal = pl.concat(
        [
            pl.scan_parquet(paths[date])
            .filter(pl.col("ValueCode").is_in(value_codes))
            .select(columns)
            .collect(engine="streaming")
            for date in sorted(required_dates)
        ],
        how="vertical",
    )
    labels = label_taker_exit_paths(positions, causal, sessions, config)
    if cost_profile is not None:
        labels = apply_execution_cost_profile(labels, cost_profile)
    summary = summarize_taker_exit_paths(labels)
    result = TakerExitStudyResult(positions, labels, summary)
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        positions.write_parquet(destination / "taker_exit_positions.parquet")
        labels.write_parquet(destination / "taker_exit_paths.parquet")
        summary.write_csv(destination / "taker_exit_summary.csv")
        payload = {
            **asdict(config),
            "cost_profile": asdict(cost_profile) if cost_profile else None,
            "input_positions": positions.height,
            "entry_contract_expiry_day_positions": labels.select(
                "raw_order_fact_id", "entry_contract_expiry_day"
            )
            .unique()
            .filter(pl.col("entry_contract_expiry_day"))
            .height,
            "label_rows": labels.height,
            "conditional_on_entry_full_fill_and_executable_hedge": True,
            "exit_maker_fill_included": False,
            "full_action_ev_ready": False,
        }
        (destination / "taker_exit_config.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    return result


def _executable_close_path(
    frame: pl.DataFrame,
    position: dict[str, object],
    *,
    eligibility_column: str,
    earliest_ns: int,
    minimum_seconds_from_open: int,
) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame()
    shares = float(position["contract_size_shares"])
    entry_spot = float(position["entry_spot_price"])
    entry_future = float(position["entry_future_price"])
    return frame.filter(
        (pl.col("_timestamp_ns") >= earliest_ns)
        & (pl.col("seconds_from_open") >= minimum_seconds_from_open)
        & pl.col(eligibility_column).fill_null(False)
        & pl.col("spot_bid").is_finite()
        & pl.col("fut_exec_ask").is_finite()
        & (pl.col("spot_bid_lots") * 1000 >= shares)
        & (pl.col("fut_exec_ask_lots") >= 1)
    ).select(
        pl.col("Date").alias("exit_date"),
        pl.col("_timestamp_ns").alias("exit_timestamp_ns"),
        pl.col("spot_bid").alias("exit_spot_price"),
        pl.col("fut_exec_ask").alias("exit_future_price"),
        (
            shares
            * (
                (pl.col("spot_bid") - entry_spot)
                + (entry_future - pl.col("fut_exec_ask"))
            )
        ).alias("gross_cash_pnl_twd"),
    )


def _max_gross_row(frame: pl.DataFrame) -> dict[str, object] | None:
    if frame.is_empty():
        return None
    return frame.sort(
        ["gross_cash_pnl_twd", "exit_timestamp_ns"],
        descending=[True, False],
    ).row(0, named=True)


def _label_record(
    position: dict[str, object],
    *,
    sample: str,
    target: float,
    entry_contract_expiry_day: bool,
    status: str,
    terminal: dict[str, object] | None,
    exit_date: str | None,
    overnight_sessions: int | None,
    unresolved_reason: str | None,
    max_row: dict[str, object] | None,
    eod_row: dict[str, object] | None,
    config: TakerExitConfig,
) -> dict[str, object]:
    resolved = terminal is not None
    exit_ns = int(terminal["exit_timestamp_ns"]) if resolved else None
    established = int(position["position_established_ns"])
    holding_seconds = (
        (exit_ns - established) / _NS_PER_SECOND if exit_ns is not None else None
    )
    return {
        **position,
        "sample": sample,
        "gross_profit_target_twd": target,
        "entry_contract_expiry_day": entry_contract_expiry_day,
        "status": status,
        "exit_date": exit_date,
        "exit_timestamp_ns": exit_ns,
        "exit_spot_price": terminal["exit_spot_price"] if resolved else None,
        "exit_future_price": terminal["exit_future_price"] if resolved else None,
        "gross_cash_pnl_twd": terminal["gross_cash_pnl_twd"] if resolved else None,
        "holding_seconds": holding_seconds,
        "overnight_sessions": overnight_sessions,
        "unresolved_reason": unresolved_reason,
        "max_same_day_taker_gross_twd": (
            max_row["gross_cash_pnl_twd"] if max_row is not None else None
        ),
        "max_same_day_taker_timestamp_ns": (
            max_row["exit_timestamp_ns"] if max_row is not None else None
        ),
        "eod_taker_mark_gross_twd": (
            eod_row["gross_cash_pnl_twd"] if eod_row is not None else None
        ),
        "same_day_decision_clock": "one_second_causal_grid",
        "exit_execution": "simultaneous_spot_bid_and_future_exec_ask_l1",
        "exit_maker_fill_included": False,
        "policy_version": config.policy_version,
        "contains_target_day_outcome": True,
        "execution_safe_snapshot": False,
    }


def _normalise_sessions(sessions: Sequence[str] | Iterable[str]) -> list[str]:
    result = sorted({str(value) for value in sessions})
    if not result:
        raise ValueError("sessions must not be empty")
    parsed = pl.DataFrame({"Date": result}).with_columns(
        pl.col("Date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("_d")
    )
    if parsed.filter(
        pl.col("_d").is_null() | (pl.col("Date").str.len_chars() != 8)
    ).height:
        raise ValueError("sessions must contain valid YYYYMMDD values")
    return result


def _next_session(
    date: str,
    sessions: list[str],
    index: dict[str, int],
) -> str | None:
    offset = index[date] + 1
    return sessions[offset] if offset < len(sessions) else None


def _is_expiry_day(frame: pl.DataFrame, date: str) -> bool:
    if frame.is_empty() or "end_date" not in frame.columns:
        return False
    values = frame["end_date"].drop_nulls().unique()
    if values.len() != 1:
        return False
    value = values[0]
    return value.strftime("%Y%m%d") == date


def _next_full_second_ns(value: int) -> int:
    return (value // _NS_PER_SECOND + 1) * _NS_PER_SECOND


def _require(frame: pl.DataFrame, columns: set[str], source: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def _position_schema() -> dict[str, pl.DataType]:
    return {
        "raw_order_fact_id": pl.String,
        "Date": pl.String,
        "ValueCode": pl.String,
        "QuoteCode": pl.String,
        "route": pl.String,
        "boundary_quantile_aliases": pl.List(pl.Int64),
        "policy_alias_count": pl.UInt32,
        "position_established_ns": pl.Int64,
        "contract_size_shares": pl.Int64,
        "entry_spot_price": pl.Float64,
        "entry_future_price": pl.Float64,
    }


def _label_schema() -> dict[str, pl.DataType]:
    return {
        **_position_schema(),
        "sample": pl.String,
        "gross_profit_target_twd": pl.Float64,
        "entry_contract_expiry_day": pl.Boolean,
        "status": pl.String,
        "exit_date": pl.String,
        "exit_timestamp_ns": pl.Int64,
        "exit_spot_price": pl.Float64,
        "exit_future_price": pl.Float64,
        "gross_cash_pnl_twd": pl.Float64,
        "holding_seconds": pl.Float64,
        "overnight_sessions": pl.Int64,
        "unresolved_reason": pl.String,
        "max_same_day_taker_gross_twd": pl.Float64,
        "max_same_day_taker_timestamp_ns": pl.Int64,
        "eod_taker_mark_gross_twd": pl.Float64,
        "same_day_decision_clock": pl.String,
        "exit_execution": pl.String,
        "exit_maker_fill_included": pl.Boolean,
        "policy_version": pl.String,
        "contains_target_day_outcome": pl.Boolean,
        "execution_safe_snapshot": pl.Boolean,
    }
