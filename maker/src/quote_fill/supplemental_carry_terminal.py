"""Supplemental full-carry continuation and expiry-mark terminal overlay.

This module is intentionally separate from the immutable formal
cross-session replay.  It implements a user-requested *scenario* in which an
exit-maker ``maker_fill_state_unknown`` is treated as a known zero fill and
the original full long-spot/short-futures pair is carried forward.  That
assumption can double-exit a position which actually filled, so every affected
row is explicitly flagged and the result is never promoted to an exact or
production path.

Normal continuation terminals must be supplied by an extended exit replay.
If no normal terminal is observed before expiry, a complete pair of
source-bound expiry-session marks may force a terminal cashflow.  A last valid
spot bid and futures ask from the supplied session tape is called a
``last_valid_session_mark``; it is not described as an official close or a
futures settlement.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import polars as pl

from .raw_tape import RawTapeDay


OVERLAY_VERSION = "imputed_full_carry_expiry_mark_v1"

_PATH_REQUIRED = {
    "Date",
    "ValueCode",
    "QuoteCode",
    "entry_policy_generation_id",
    "policy_path_id",
    "position_established_ns",
    "filled_entry_outcome_category",
    "outcome_status",
    "terminal_date",
    "exit_decision_time_ns",
    "gross_cycle_pnl_twd",
    "gross_cycle_bp",
    "normalization_notional_twd",
    "completed_same_day",
    "completed_overnight",
    "terminal_cashflow_priced",
    "unresolved_cashflow_imputed",
}
_EXACT_PRICE_COLUMNS = {
    "entry_spot_price": pl.Float64,
    "entry_future_price": pl.Float64,
    "entry_contract_size_shares": pl.Int64,
    "exit_spot_price": pl.Float64,
    "exit_future_price": pl.Float64,
    "exact_price_source": pl.String,
}
_ENTRY_PRICE_REQUIRED = {
    "entry_policy_generation_id",
    "entry_spot_price",
    "entry_future_price",
    "entry_contract_size_shares",
    "entry_price_source",
    "source_identity_sha256",
}
_CONTINUATION_REQUIRED = {
    "policy_path_id",
    "terminal_date",
    "exit_decision_time_ns",
    "exit_spot_price",
    "exit_future_price",
    "exact_price_source",
    "source_identity_sha256",
}
_EXPIRY_MARK_REQUIRED = {
    "Date",
    "expiry_session",
    "calendar_version",
    "ValueCode",
    "QuoteCode",
    "spot_close_price",
    "future_close_price",
    "spot_close_time_ns",
    "future_close_time_ns",
    "spot_close_source",
    "future_close_source",
    "source_identity_sha256",
    "mark_is_official_close",
    "mark_is_official_settlement",
}
_STATE_REQUIRED = {
    "Date",
    "ValueCode",
    "QuoteCode",
    "recv_time_ns",
    "sequence",
    "raw_has_book",
    "book_state_available",
    "exec_bid_price",
    "exec_ask_price",
}
_TRADE_REQUIRED = {
    "Date",
    "ValueCode",
    "QuoteCode",
    "recv_time_ns",
    "sequence",
    "trade_price",
    "trade_lots",
}


@dataclass(frozen=True)
class SupplementalCarryConfig:
    """Frozen assumptions for the non-formal carry scenario."""

    eligible_unknown_statuses: tuple[str, ...] = (
        "maker_fill_state_unknown",
        "right_censored_observation_end",
    )
    expiry_unpriced_status: str = "right_censored_expiry_settlement_unpriced"
    model_imputed_full_carry_on_unknown: bool = True
    overlay_version: str = OVERLAY_VERSION

    def validate(self) -> None:
        if (
            not self.eligible_unknown_statuses
            or len(set(self.eligible_unknown_statuses))
            != len(self.eligible_unknown_statuses)
            or any(not value for value in self.eligible_unknown_statuses)
        ):
            raise ValueError("eligible unknown statuses must be nonempty and unique")
        if not self.expiry_unpriced_status:
            raise ValueError("expiry_unpriced_status must be nonempty")
        if self.model_imputed_full_carry_on_unknown is not True:
            raise ValueError("supplemental scenario must disclose imputed full carry")
        if not self.overlay_version:
            raise ValueError("overlay_version must be nonempty")


def build_last_valid_session_bbo_mark(
    raw_tape: RawTapeDay,
    *,
    value_code: str,
    quote_code: str,
    expiry_session: str,
    calendar_version: str,
    source_identity_sha256: str,
) -> pl.DataFrame:
    """Return the last valid spot-bid/futures-ask mark in a supplied tape.

    The two legs are selected independently from the final real book update
    for the exact pair.  This is an executable-side BBO mark, not proof of
    executable depth, an official close, or an official settlement.
    """

    if not isinstance(raw_tape, RawTapeDay):
        raise TypeError("raw_tape must be a RawTapeDay")
    if str(raw_tape.date) != str(expiry_session):
        raise ValueError("last-valid expiry mark tape date must equal expiry_session")
    if not calendar_version:
        raise ValueError("calendar_version must be nonempty")
    _validate_sha256(source_identity_sha256, "source_identity_sha256")
    spot = _last_valid_book(
        raw_tape.spot_states,
        date=str(raw_tape.date),
        value_code=str(value_code),
        quote_code=str(quote_code),
        side="bid",
    )
    future = _last_valid_book(
        raw_tape.future_states,
        date=str(raw_tape.date),
        value_code=str(value_code),
        quote_code=str(quote_code),
        side="ask",
    )
    return pl.from_dicts(
        [
            {
                "Date": str(raw_tape.date),
                "expiry_session": str(expiry_session),
                "calendar_version": str(calendar_version),
                "ValueCode": str(value_code),
                "QuoteCode": str(quote_code),
                "spot_close_price": float(spot["exec_bid_price"]),
                "future_close_price": float(future["exec_ask_price"]),
                "spot_close_time_ns": int(spot["recv_time_ns"]),
                "future_close_time_ns": int(future["recv_time_ns"]),
                "spot_close_source": "last_valid_session_spot_bid",
                "future_close_source": "last_valid_session_future_ask",
                "source_identity_sha256": source_identity_sha256,
                "mark_is_official_close": False,
                "mark_is_official_settlement": False,
                "mark_role": (
                    "last_valid_session_executable_bbo_mark_not_official_close_"
                    "or_settlement"
                ),
            }
        ],
        infer_schema_length=None,
    )


def build_last_observed_session_liquidation_mark(
    raw_tape: RawTapeDay,
    *,
    value_code: str,
    quote_code: str,
    expiry_session: str,
    calendar_version: str,
    source_identity_sha256: str,
) -> pl.DataFrame:
    """Return a two-leg expiry mark with an explicitly named trade fallback.

    Each leg first uses its liquidation-side final valid BBO (spot bid for
    the long spot and futures ask for the short future).  Only when that side
    has no valid BBO anywhere in the supplied session does it use the final
    positive observed trade.  Such a trade is not an executable quote,
    official close, or futures settlement.
    """

    if not isinstance(raw_tape, RawTapeDay):
        raise TypeError("raw_tape must be a RawTapeDay")
    if str(raw_tape.date) != str(expiry_session):
        raise ValueError("last-observed mark tape date must equal expiry_session")
    if not calendar_version:
        raise ValueError("calendar_version must be nonempty")
    _validate_sha256(source_identity_sha256, "source_identity_sha256")
    spot, spot_source, spot_bbo = _last_liquidation_leg(
        raw_tape.spot_states,
        raw_tape.spot_trades,
        date=str(raw_tape.date),
        value_code=str(value_code),
        quote_code=str(quote_code),
        market="spot",
        side="bid",
    )
    future, future_source, future_bbo = _last_liquidation_leg(
        raw_tape.future_states,
        raw_tape.future_trades,
        date=str(raw_tape.date),
        value_code=str(value_code),
        quote_code=str(quote_code),
        market="future",
        side="ask",
    )
    return pl.from_dicts(
        [
            {
                "Date": str(raw_tape.date),
                "expiry_session": str(expiry_session),
                "calendar_version": str(calendar_version),
                "ValueCode": str(value_code),
                "QuoteCode": str(quote_code),
                "spot_close_price": float(spot["price"]),
                "future_close_price": float(future["price"]),
                "spot_close_time_ns": int(spot["recv_time_ns"]),
                "future_close_time_ns": int(future["recv_time_ns"]),
                "spot_close_source": spot_source,
                "future_close_source": future_source,
                "source_identity_sha256": source_identity_sha256,
                "mark_is_official_close": False,
                "mark_is_official_settlement": False,
                "spot_mark_is_executable_bbo": spot_bbo,
                "future_mark_is_executable_bbo": future_bbo,
                "mark_uses_trade_fallback": not (spot_bbo and future_bbo),
                "mark_role": (
                    "last_observed_session_liquidation_mark_not_official_close_"
                    "or_settlement"
                ),
            }
        ],
        infer_schema_length=None,
    )


def apply_supplemental_carry_terminal_overlay(
    paths: pl.DataFrame,
    entry_prices: pl.DataFrame,
    continuation_terminals: pl.DataFrame | None,
    expiry_marks: pl.DataFrame | None,
    *,
    config: SupplementalCarryConfig = SupplementalCarryConfig(),
) -> pl.DataFrame:
    """Overlay normal continuation or expiry-mark terminals onto paths.

    ``paths`` should be the exact-price-enriched challenger paths consumed by
    the cost/cap sweep: source-completed rows already have six exact-price
    fields, while unresolved rows have those fields null.  Entry prices for
    unresolved rows are supplied separately so a missing terminal never
    leaks partial pricing into the cost producer.
    """

    config.validate()
    source = _normalise_paths(paths)
    entries = _entry_price_lookup(entry_prices)
    continuations = _continuation_lookup(continuation_terminals)
    marks = _expiry_mark_lookup(expiry_marks)
    path_ids = set(source["policy_path_id"].to_list())
    extra_continuations = set(continuations) - path_ids
    if extra_continuations:
        raise ValueError("continuation terminals contain unknown policy_path_id")

    rows: list[dict[str, object]] = []
    for item in source.iter_rows(named=True):
        row = dict(item)
        category = str(row["filled_entry_outcome_category"])
        source_status = str(row["outcome_status"])
        row.update(
            {
                "source_filled_entry_outcome_category": category,
                "source_outcome_status": source_status,
                "supplemental_terminal_resolution": (
                    "source_completed" if category == "completed" else "unresolved"
                ),
                "model_imputed_full_carry_on_unknown": False,
                "double_exit_bias_possible": False,
                "expiry_uses_last_valid_session_mark": False,
                "expiry_uses_last_observed_session_mark": False,
                "expiry_mark_uses_trade_fallback": False,
                "spot_expiry_mark_is_executable_bbo": None,
                "future_expiry_mark_is_executable_bbo": None,
                "expiry_mark_is_official_close": None,
                "expiry_mark_is_official_settlement": None,
                "supplemental_terminal_source_identity_sha256": None,
                "supplemental_overlay_version": config.overlay_version,
                "supplemental_scenario_analysis_only": True,
                "production_strategy_go_after_supplemental_overlay": False,
            }
        )
        if category == "completed":
            rows.append(row)
            continue

        unknown_full_carry = (
            category in {"unknown", "still_open"}
            and source_status in config.eligible_unknown_statuses
        )
        expiry_unpriced = source_status == config.expiry_unpriced_status
        if not unknown_full_carry and not expiry_unpriced:
            rows.append(row)
            continue

        continuation = continuations.get(str(row["policy_path_id"]))
        mark = marks.get(
            (str(row["ValueCode"]), str(row["QuoteCode"]))
        )
        mark_kind = (
            "expiry_last_observed_session_liquidation_mark"
            if mark is not None and mark.get("mark_uses_trade_fallback") is True
            else "expiry_last_valid_session_mark"
        )
        chosen_kind: str | None = None
        chosen: Mapping[str, object] | None = None
        if continuation is not None and mark is not None:
            continuation_key = (
                str(continuation["terminal_date"]),
                int(continuation["exit_decision_time_ns"]),
            )
            mark_key = (
                str(mark["Date"]),
                max(
                    int(mark["spot_close_time_ns"]),
                    int(mark["future_close_time_ns"]),
                ),
            )
            if continuation_key <= mark_key:
                chosen_kind, chosen = "normal_continuation_replay", continuation
            else:
                chosen_kind, chosen = mark_kind, mark
        elif continuation is not None:
            chosen_kind, chosen = "normal_continuation_replay", continuation
        elif mark is not None:
            chosen_kind, chosen = mark_kind, mark
        if chosen is None:
            if unknown_full_carry:
                row["model_imputed_full_carry_on_unknown"] = True
                row["double_exit_bias_possible"] = True
                row["supplemental_terminal_resolution"] = (
                    "imputed_full_carry_still_unresolved"
                )
            rows.append(row)
            continue

        entry_id = str(row["entry_policy_generation_id"])
        try:
            entry = entries[entry_id]
        except KeyError as error:
            raise ValueError(
                f"terminal overlay lacks entry prices for {entry_id}"
            ) from error
        if chosen_kind == "normal_continuation_replay":
            terminal_date = str(chosen["terminal_date"])
            terminal_ns = int(chosen["exit_decision_time_ns"])
            exit_spot = float(chosen["exit_spot_price"])
            exit_future = float(chosen["exit_future_price"])
            exact_source = str(chosen["exact_price_source"])
            source_identity = str(chosen["source_identity_sha256"])
            terminal_reason = (
                "supplemental_normal_exit_replay_after_imputed_full_carry"
            )
        else:
            terminal_date = str(chosen["Date"])
            terminal_ns = max(
                int(chosen["spot_close_time_ns"]),
                int(chosen["future_close_time_ns"]),
            )
            exit_spot = float(chosen["spot_close_price"])
            exit_future = float(chosen["future_close_price"])
            exact_source = (
                f"{chosen['spot_close_source']}+{chosen['future_close_source']}"
            )
            source_identity = str(chosen["source_identity_sha256"])
            terminal_reason = (
                "expiry_last_observed_session_liquidation_mark_not_official_"
                "close_or_settlement"
                if chosen_kind
                == "expiry_last_observed_session_liquidation_mark"
                else "expiry_last_valid_session_mark_not_settlement"
            )
        _validate_terminal_after_entry(row, terminal_date, terminal_ns)
        shares = int(entry["entry_contract_size_shares"])
        entry_spot = float(entry["entry_spot_price"])
        entry_future = float(entry["entry_future_price"])
        notional = float(row["normalization_notional_twd"])
        if not math.isclose(
            shares * entry_spot, notional, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError("entry prices do not reconcile to path notional")
        gross = shares * (
            (exit_spot - entry_spot) + (entry_future - exit_future)
        )
        row.update(
            {
                "filled_entry_outcome_category": "completed",
                "outcome_status": terminal_reason,
                "terminal_date": terminal_date,
                "terminal_reason": terminal_reason,
                "exit_decision_time_ns": terminal_ns,
                "gross_cycle_pnl_twd": gross,
                "gross_cycle_bp": gross / notional * 10_000.0,
                "completed_same_day": terminal_date == str(row["Date"]),
                "completed_overnight": terminal_date > str(row["Date"]),
                "terminal_cashflow_priced": True,
                "entry_spot_price": entry_spot,
                "entry_future_price": entry_future,
                "entry_contract_size_shares": shares,
                "exit_spot_price": exit_spot,
                "exit_future_price": exit_future,
                "exact_price_source": exact_source,
                "unresolved_cashflow_imputed": False,
                "supplemental_terminal_resolution": chosen_kind,
                "model_imputed_full_carry_on_unknown": unknown_full_carry,
                "double_exit_bias_possible": unknown_full_carry,
                "expiry_uses_last_valid_session_mark": (
                    chosen_kind == "expiry_last_valid_session_mark"
                ),
                "expiry_uses_last_observed_session_mark": (
                    chosen_kind
                    == "expiry_last_observed_session_liquidation_mark"
                ),
                "expiry_mark_uses_trade_fallback": (
                    bool(chosen.get("mark_uses_trade_fallback", False))
                    if chosen_kind != "normal_continuation_replay"
                    else False
                ),
                "spot_expiry_mark_is_executable_bbo": (
                    chosen.get("spot_mark_is_executable_bbo")
                    if chosen_kind != "normal_continuation_replay"
                    else None
                ),
                "future_expiry_mark_is_executable_bbo": (
                    chosen.get("future_mark_is_executable_bbo")
                    if chosen_kind != "normal_continuation_replay"
                    else None
                ),
                "expiry_mark_is_official_close": (
                    bool(chosen["mark_is_official_close"])
                    if chosen_kind != "normal_continuation_replay"
                    else None
                ),
                "expiry_mark_is_official_settlement": (
                    bool(chosen["mark_is_official_settlement"])
                    if chosen_kind != "normal_continuation_replay"
                    else None
                ),
                "supplemental_terminal_source_identity_sha256": source_identity,
            }
        )
        if "outstanding_interval_end_exclusive" in row:
            row["outstanding_interval_end_exclusive"] = terminal_date
        rows.append(row)
    result = pl.from_dicts(rows, infer_schema_length=None).sort(
        ["Date", "position_established_ns", "ValueCode", "policy_path_id"]
    )
    _validate_overlay_result(result, source)
    return result


def _normalise_paths(paths: pl.DataFrame) -> pl.DataFrame:
    missing = sorted(_PATH_REQUIRED - set(paths.columns))
    if missing:
        raise ValueError(f"paths missing columns: {missing}")
    result = paths
    expressions: list[pl.Expr] = []
    for column, dtype in _EXACT_PRICE_COLUMNS.items():
        if column not in result.columns:
            expressions.append(pl.lit(None, dtype=dtype).alias(column))
    if expressions:
        result = result.with_columns(*expressions)
    result = result.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("terminal_date").cast(pl.String),
        pl.col("normalization_notional_twd").cast(pl.Float64),
    )
    if result.is_empty() or result["policy_path_id"].n_unique() != result.height:
        raise ValueError("paths must be nonempty and unique by policy_path_id")
    invalid_notional = result.filter(
        pl.col("normalization_notional_twd").is_null()
        | ~pl.col("normalization_notional_twd").is_finite()
        | (pl.col("normalization_notional_twd") <= 0)
    )
    if invalid_notional.height:
        raise ValueError("path notional must be finite and positive")
    return result.sort(
        ["Date", "position_established_ns", "ValueCode", "policy_path_id"]
    )


def _entry_price_lookup(frame: pl.DataFrame) -> dict[str, dict[str, object]]:
    _require(frame, _ENTRY_PRICE_REQUIRED, "entry prices")
    if frame.select("entry_policy_generation_id").n_unique() != frame.height:
        raise ValueError("entry prices must be unique by generation ID")
    result: dict[str, dict[str, object]] = {}
    for row in frame.iter_rows(named=True):
        for name in (
            "entry_spot_price",
            "entry_future_price",
            "entry_contract_size_shares",
        ):
            if not _positive(row[name]):
                raise ValueError(f"entry prices contain invalid {name}")
        _validate_sha256(row["source_identity_sha256"], "entry source identity")
        if not row["entry_price_source"]:
            raise ValueError("entry_price_source must be nonempty")
        result[str(row["entry_policy_generation_id"])] = dict(row)
    return result


def _continuation_lookup(
    frame: pl.DataFrame | None,
) -> dict[str, dict[str, object]]:
    if frame is None:
        return {}
    _require(frame, _CONTINUATION_REQUIRED, "continuation terminals")
    if frame.select("policy_path_id").n_unique() != frame.height:
        raise ValueError("continuation terminals must be unique by policy_path_id")
    result: dict[str, dict[str, object]] = {}
    for row in frame.iter_rows(named=True):
        if not _positive(row["exit_spot_price"]) or not _positive(
            row["exit_future_price"]
        ):
            raise ValueError("continuation terminal prices must be positive")
        if not _nonnegative_int(row["exit_decision_time_ns"]):
            raise ValueError("continuation terminal time must be non-negative")
        if not row["exact_price_source"]:
            raise ValueError("continuation exact_price_source must be nonempty")
        _validate_sha256(
            row["source_identity_sha256"], "continuation source identity"
        )
        result[str(row["policy_path_id"])] = dict(row)
    return result


def _expiry_mark_lookup(
    frame: pl.DataFrame | None,
) -> dict[tuple[str, str], dict[str, object]]:
    if frame is None:
        return {}
    _require(frame, _EXPIRY_MARK_REQUIRED, "expiry marks")
    if frame.select("ValueCode", "QuoteCode").n_unique() != frame.height:
        raise ValueError("expiry marks must be unique by ValueCode and QuoteCode")
    result: dict[tuple[str, str], dict[str, object]] = {}
    for row in frame.iter_rows(named=True):
        if str(row["Date"]) != str(row["expiry_session"]):
            raise ValueError("expiry mark Date must equal expiry_session")
        if not row["calendar_version"]:
            raise ValueError("expiry marks require calendar_version")
        for name in ("spot_close_price", "future_close_price"):
            if not _positive(row[name]):
                raise ValueError(f"expiry marks contain invalid {name}")
        for name in ("spot_close_time_ns", "future_close_time_ns"):
            if not _nonnegative_int(row[name]):
                raise ValueError(f"expiry marks contain invalid {name}")
        for name in ("spot_close_source", "future_close_source"):
            if not row[name]:
                raise ValueError(f"expiry marks require {name}")
        _validate_sha256(row["source_identity_sha256"], "expiry source identity")
        if row["mark_is_official_settlement"] is True:
            raise ValueError(
                "this overlay accepts close/mark prices, not official settlement claims"
            )
        key = (str(row["ValueCode"]), str(row["QuoteCode"]))
        result[key] = dict(row)
    return result


def _last_valid_book(
    frame: pl.DataFrame,
    *,
    date: str,
    value_code: str,
    quote_code: str,
    side: str,
) -> dict[str, object]:
    _require(frame, _STATE_REQUIRED, "normalized market states")
    price_column = "exec_bid_price" if side == "bid" else "exec_ask_price"
    selected = (
        frame.filter(
            (pl.col("Date").cast(pl.String) == date)
            & (pl.col("ValueCode").cast(pl.String) == value_code)
            & (pl.col("QuoteCode").cast(pl.String) == quote_code)
            & pl.col("raw_has_book").fill_null(False)
            & pl.col("book_state_available").fill_null(False)
            & pl.col(price_column).is_not_null()
            & pl.col(price_column).is_finite()
            & (pl.col(price_column) > 0)
        )
        .sort(
            [
                column
                for column in ("recv_time_ns", "sequence", "packet_sequence")
                if column in frame.columns
            ]
        )
    )
    if selected.is_empty():
        raise ValueError(f"no last valid session {side} mark for exact pair")
    return selected.row(-1, named=True)


def _last_liquidation_leg(
    states: pl.DataFrame,
    trades: pl.DataFrame,
    *,
    date: str,
    value_code: str,
    quote_code: str,
    market: str,
    side: str,
) -> tuple[dict[str, object], str, bool]:
    try:
        book = _last_valid_book(
            states,
            date=date,
            value_code=value_code,
            quote_code=quote_code,
            side=side,
        )
    except ValueError:
        trade = _last_valid_trade(
            trades,
            date=date,
            value_code=value_code,
            quote_code=quote_code,
        )
        return (
            {
                "price": trade["trade_price"],
                "recv_time_ns": trade["recv_time_ns"],
            },
            f"last_observed_session_{market}_trade_not_official_close",
            False,
        )
    price_column = "exec_bid_price" if side == "bid" else "exec_ask_price"
    return (
        {"price": book[price_column], "recv_time_ns": book["recv_time_ns"]},
        f"last_valid_session_{market}_{side}",
        True,
    )


def _last_valid_trade(
    frame: pl.DataFrame,
    *,
    date: str,
    value_code: str,
    quote_code: str,
) -> dict[str, object]:
    _require(frame, _TRADE_REQUIRED, "normalized market trades")
    selected = (
        frame.filter(
            (pl.col("Date").cast(pl.String) == date)
            & (pl.col("ValueCode").cast(pl.String) == value_code)
            & (pl.col("QuoteCode").cast(pl.String) == quote_code)
            & pl.col("trade_price").is_not_null()
            & pl.col("trade_price").is_finite()
            & (pl.col("trade_price") > 0)
            & pl.col("trade_lots").is_not_null()
            & (pl.col("trade_lots") > 0)
        )
        .sort(
            [
                column
                for column in ("recv_time_ns", "sequence", "packet_sequence")
                if column in frame.columns
            ]
        )
    )
    if selected.is_empty():
        raise ValueError("no positive observed trade exists for liquidation fallback")
    return selected.row(-1, named=True)


def _validate_terminal_after_entry(
    row: Mapping[str, object], terminal_date: str, terminal_ns: int
) -> None:
    entry_key = (str(row["Date"]), int(row["position_established_ns"]))
    terminal_key = (str(terminal_date), int(terminal_ns))
    if terminal_key < entry_key:
        raise ValueError("supplemental terminal precedes position establishment")


def _validate_overlay_result(result: pl.DataFrame, source: pl.DataFrame) -> None:
    if result.height != source.height or result["policy_path_id"].n_unique() != result.height:
        raise ValueError("supplemental overlay changed the path population")
    priced = result.filter(pl.col("terminal_cashflow_priced"))
    if priced.filter(
        pl.col("gross_cycle_pnl_twd").is_null()
        | pl.col("gross_cycle_bp").is_null()
        | pl.col("terminal_date").is_null()
        | pl.col("exit_decision_time_ns").is_null()
    ).height:
        raise ValueError("supplemental priced terminal is incomplete")
    imputed = result.filter(pl.col("model_imputed_full_carry_on_unknown"))
    if imputed.filter(
        (pl.col("double_exit_bias_possible") != True).fill_null(True)  # noqa: E712
        | (pl.col("supplemental_scenario_analysis_only") != True).fill_null(True)  # noqa: E712
        | (
            pl.col("production_strategy_go_after_supplemental_overlay") != False
        ).fill_null(True)  # noqa: E712
    ).height:
        raise ValueError("imputed carry disclosure flags are incomplete")
    expiry = result.filter(pl.col("expiry_uses_last_valid_session_mark"))
    if expiry.filter(
        (pl.col("expiry_mark_is_official_settlement") != False).fill_null(True)  # noqa: E712
    ).height:
        raise ValueError("last-valid expiry mark was mislabeled as settlement")
    observed = result.filter(pl.col("expiry_uses_last_observed_session_mark"))
    if observed.filter(
        (pl.col("expiry_mark_uses_trade_fallback") != True).fill_null(True)  # noqa: E712
        | (pl.col("expiry_mark_is_official_close") != False).fill_null(True)  # noqa: E712
        | (pl.col("expiry_mark_is_official_settlement") != False).fill_null(True)  # noqa: E712
    ).height:
        raise ValueError("last-observed expiry trade fallback labels are incoherent")


def _require(frame: pl.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _positive(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value)) and float(value) > 0


def _nonnegative_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _validate_sha256(value: object, label: str) -> None:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase SHA-256")
