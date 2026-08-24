"""Replace legacy expiry BBO marks with same-day two-leg close prices.

The supplemental carry replay v2 used the last executable-side book state on
the expiry session.  This module is a narrow, deterministic overlay: normal
continuation and source-completed paths remain unchanged, while paths whose
selected terminal is an expiry mark are repriced from a source-bound pair of
daily ``close_price`` facts.

The daily close is an accounting/forced-flat mark.  It is not an executable
BBO and the futures close must not be mislabeled as settlement.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
import math
import re
from typing import Mapping

import polars as pl

from .portfolio_cap_backtester import local_session_timestamp_ns


OVERLAY_VERSION = "supplemental_full_carry_expiry_daily_close_overlay_v3"
TERMINAL_RESOLUTION = "expiry_same_day_two_leg_close_price"
TERMINAL_REASON = "expiry_same_day_two_leg_close_price_forced_flat"
SPOT_CLOSE_SOURCE = "MarketInfo.twse_security_trades_daily.close_price"
FUTURE_CLOSE_SOURCE = "MarketInfo.taifex_futures_trades_daily.close_price"
EXACT_PRICE_SOURCE = f"{SPOT_CLOSE_SOURCE}+{FUTURE_CLOSE_SOURCE}"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FACT_REQUIRED = {
    "Date",
    "ValueCode",
    "QuoteCode",
    "spot_close_price",
    "future_close_price",
    "source_identity_sha256",
}
_PATH_REQUIRED = {
    "Date",
    "ValueCode",
    "QuoteCode",
    "policy_path_id",
    "entry_spot_price",
    "entry_future_price",
    "entry_contract_size_shares",
    "normalization_notional_twd",
    "terminal_date",
    "exit_decision_time_ns",
    "exit_spot_price",
    "exit_future_price",
    "gross_cycle_pnl_twd",
    "gross_cycle_bp",
    "supplemental_terminal_resolution",
    "supplemental_terminal_source_identity_sha256",
    "supplemental_overlay_version",
    "outcome_status",
    "terminal_reason",
    "completed_same_day",
    "completed_overnight",
    "terminal_cashflow_priced",
    "unresolved_cashflow_imputed",
    "exact_price_source",
    "expiry_uses_last_valid_session_mark",
    "expiry_uses_last_observed_session_mark",
    "expiry_mark_uses_trade_fallback",
    "spot_expiry_mark_is_executable_bbo",
    "future_expiry_mark_is_executable_bbo",
    "expiry_mark_is_official_close",
    "expiry_mark_is_official_settlement",
}
_MARK_REQUIRED = {
    "Date",
    "expiry_session",
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
    "spot_mark_is_executable_bbo",
    "future_mark_is_executable_bbo",
    "mark_uses_trade_fallback",
    "mark_role",
}


@dataclass(frozen=True)
class ExpiryCloseOverlayResult:
    supplemental_paths: pl.DataFrame
    expiry_marks: pl.DataFrame
    continuation_audit: pl.DataFrame
    overlay_audit: pl.DataFrame


def apply_expiry_daily_close_overlay(
    paths: pl.DataFrame,
    legacy_expiry_marks: pl.DataFrame,
    continuation_audit: pl.DataFrame,
    daily_close_facts: pl.DataFrame,
    *,
    close_local_time: time = time(13, 30),
    timezone_name: str = "Asia/Taipei",
) -> ExpiryCloseOverlayResult:
    """Reprice only expiry-selected terminals from paired daily closes."""

    _require(paths, _PATH_REQUIRED, "supplemental paths")
    _require(legacy_expiry_marks, _MARK_REQUIRED, "legacy expiry marks")
    _require(daily_close_facts, _FACT_REQUIRED, "daily close facts")
    if paths.is_empty() or paths["policy_path_id"].n_unique() != paths.height:
        raise ValueError("supplemental paths must be nonempty and path-unique")

    facts = _normalise_facts(daily_close_facts)
    marks = _normalise_marks(legacy_expiry_marks)
    fact_lookup = {
        (str(row["Date"]), str(row["ValueCode"]), str(row["QuoteCode"])): row
        for row in facts.iter_rows(named=True)
    }
    mark_keys = {
        (str(row["Date"]), str(row["ValueCode"]), str(row["QuoteCode"]))
        for row in marks.iter_rows(named=True)
    }
    if set(fact_lookup) != mark_keys:
        missing = sorted(mark_keys - set(fact_lookup))
        extra = sorted(set(fact_lookup) - mark_keys)
        raise ValueError(
            f"daily close facts do not exactly cover expiry marks: "
            f"missing={missing}, extra={extra}"
        )

    expiry_rows = paths.filter(
        pl.col("supplemental_terminal_resolution").str.starts_with("expiry_")
    )
    if expiry_rows.is_empty():
        raise ValueError("supplemental paths contain no expiry-selected terminals")
    expiry_ids = set(expiry_rows["policy_path_id"].to_list())
    audit_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    for source in paths.iter_rows(named=True):
        row = dict(source)
        identifier = str(row["policy_path_id"])
        if identifier not in expiry_ids:
            result_rows.append(row)
            continue
        key = (str(row["terminal_date"]), str(row["ValueCode"]), str(row["QuoteCode"]))
        try:
            fact = fact_lookup[key]
        except KeyError as error:
            raise ValueError(f"expiry path lacks a paired daily close fact: {key}") from error

        shares = _positive_int(row["entry_contract_size_shares"], "entry shares")
        entry_spot = _positive_float(row["entry_spot_price"], "entry spot")
        entry_future = _positive_float(row["entry_future_price"], "entry future")
        notional = _positive_float(row["normalization_notional_twd"], "notional")
        if not math.isclose(shares * entry_spot, notional, rel_tol=1e-10, abs_tol=1e-6):
            raise ValueError(f"{identifier}: entry notional does not reconcile")
        spot_close = float(fact["spot_close_price"])
        future_close = float(fact["future_close_price"])
        gross = shares * (
            (spot_close - entry_spot) + (entry_future - future_close)
        )
        close_ns = local_session_timestamp_ns(
            str(fact["Date"]), close_local_time, timezone_name
        )
        old_gross = float(row["gross_cycle_pnl_twd"])
        old_spot = float(row["exit_spot_price"])
        old_future = float(row["exit_future_price"])
        old_resolution = str(row["supplemental_terminal_resolution"])
        row.update(
            {
                "terminal_date": str(fact["Date"]),
                "exit_decision_time_ns": close_ns,
                "exit_spot_price": spot_close,
                "exit_future_price": future_close,
                "gross_cycle_pnl_twd": gross,
                "gross_cycle_bp": gross / notional * 10_000.0,
                "supplemental_terminal_resolution": TERMINAL_RESOLUTION,
                "supplemental_terminal_source_identity_sha256": str(
                    fact["source_identity_sha256"]
                ),
                "supplemental_overlay_version": OVERLAY_VERSION,
                "outcome_status": TERMINAL_REASON,
                "terminal_reason": TERMINAL_REASON,
                "completed_same_day": str(row["Date"]) == str(fact["Date"]),
                "completed_overnight": str(row["Date"]) < str(fact["Date"]),
                "terminal_cashflow_priced": True,
                "unresolved_cashflow_imputed": False,
                "exact_price_source": EXACT_PRICE_SOURCE,
                "expiry_uses_last_valid_session_mark": False,
                "expiry_uses_last_observed_session_mark": False,
                "expiry_mark_uses_trade_fallback": False,
                "spot_expiry_mark_is_executable_bbo": False,
                "future_expiry_mark_is_executable_bbo": False,
                "expiry_mark_is_official_close": True,
                "expiry_mark_is_official_settlement": False,
            }
        )
        result_rows.append(row)
        audit_rows.append(
            {
                "policy_path_id": identifier,
                "Date": str(row["Date"]),
                "terminal_date": str(fact["Date"]),
                "ValueCode": str(row["ValueCode"]),
                "QuoteCode": str(row["QuoteCode"]),
                "legacy_terminal_resolution": old_resolution,
                "overlay_terminal_resolution": TERMINAL_RESOLUTION,
                "legacy_exit_spot_price": old_spot,
                "daily_spot_close_price": spot_close,
                "legacy_exit_future_price": old_future,
                "daily_future_close_price": future_close,
                "legacy_gross_cycle_pnl_twd": old_gross,
                "daily_close_gross_cycle_pnl_twd": gross,
                "gross_pnl_delta_twd": gross - old_gross,
                "daily_close_source_identity_sha256": str(
                    fact["source_identity_sha256"]
                ),
            }
        )

    result_paths = pl.from_dicts(
        result_rows, schema=paths.schema, infer_schema_length=None
    ).sort(["Date", "position_established_ns", "ValueCode", "policy_path_id"])
    result_marks = _overlay_marks(
        marks,
        fact_lookup,
        close_local_time=close_local_time,
        timezone_name=timezone_name,
    )
    result_continuation_audit = _overlay_continuation_audit(
        continuation_audit, expiry_ids
    )
    overlay_audit = pl.from_dicts(audit_rows, infer_schema_length=None).sort(
        ["terminal_date", "ValueCode", "Date", "policy_path_id"]
    )
    _validate_result(paths, result_paths, result_marks, overlay_audit, expiry_ids)
    return ExpiryCloseOverlayResult(
        supplemental_paths=result_paths,
        expiry_marks=result_marks,
        continuation_audit=result_continuation_audit,
        overlay_audit=overlay_audit,
    )


def _overlay_marks(
    marks: pl.DataFrame,
    fact_lookup: Mapping[tuple[str, str, str], Mapping[str, object]],
    *,
    close_local_time: time,
    timezone_name: str,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for source in marks.iter_rows(named=True):
        row = dict(source)
        key = (str(row["Date"]), str(row["ValueCode"]), str(row["QuoteCode"]))
        fact = fact_lookup[key]
        close_ns = local_session_timestamp_ns(
            str(row["Date"]), close_local_time, timezone_name
        )
        row.update(
            {
                "spot_close_price": float(fact["spot_close_price"]),
                "future_close_price": float(fact["future_close_price"]),
                "spot_close_time_ns": close_ns,
                "future_close_time_ns": close_ns,
                "spot_close_source": SPOT_CLOSE_SOURCE,
                "future_close_source": FUTURE_CLOSE_SOURCE,
                "source_identity_sha256": str(fact["source_identity_sha256"]),
                "mark_is_official_close": True,
                "mark_is_official_settlement": False,
                "spot_mark_is_executable_bbo": False,
                "future_mark_is_executable_bbo": False,
                "mark_uses_trade_fallback": False,
                "mark_role": "same_day_two_leg_daily_close_not_executable_bbo_or_settlement",
            }
        )
        rows.append(row)
    return pl.from_dicts(rows, schema=marks.schema, infer_schema_length=None).sort(
        ["Date", "ValueCode", "QuoteCode"]
    )


def _overlay_continuation_audit(
    continuation_audit: pl.DataFrame, expiry_ids: set[str]
) -> pl.DataFrame:
    if "policy_path_id" not in continuation_audit.columns:
        raise ValueError("continuation audit lacks policy_path_id")
    rows: list[dict[str, object]] = []
    found: set[str] = set()
    for source in continuation_audit.iter_rows(named=True):
        row = dict(source)
        identifier = str(row["policy_path_id"])
        if identifier in expiry_ids:
            row["terminal_resolution"] = TERMINAL_RESOLUTION
            found.add(identifier)
        rows.append(row)
    if found != expiry_ids:
        raise ValueError("continuation audit does not cover every expiry path")
    return pl.from_dicts(
        rows, schema=continuation_audit.schema, infer_schema_length=None
    ).sort(["Date", "ValueCode", "policy_path_id"])


def _normalise_facts(frame: pl.DataFrame) -> pl.DataFrame:
    result = frame.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("spot_close_price").cast(pl.Float64),
        pl.col("future_close_price").cast(pl.Float64),
        pl.col("source_identity_sha256").cast(pl.String),
    ).sort(["Date", "ValueCode", "QuoteCode"])
    if result.select("Date", "ValueCode", "QuoteCode").n_unique() != result.height:
        raise ValueError("daily close facts must be unique by date/pair")
    for row in result.iter_rows(named=True):
        _positive_float(row["spot_close_price"], "spot close")
        _positive_float(row["future_close_price"], "future close")
        if not _SHA256.fullmatch(str(row["source_identity_sha256"])):
            raise ValueError("daily close fact source identity is not SHA-256")
    return result


def _normalise_marks(frame: pl.DataFrame) -> pl.DataFrame:
    result = frame.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("expiry_session").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
    ).sort(["Date", "ValueCode", "QuoteCode"])
    if result.select("Date", "ValueCode", "QuoteCode").n_unique() != result.height:
        raise ValueError("legacy expiry marks must be unique by date/pair")
    if result.filter(pl.col("Date") != pl.col("expiry_session")).height:
        raise ValueError("legacy expiry mark date differs from expiry_session")
    return result


def _validate_result(
    source: pl.DataFrame,
    result: pl.DataFrame,
    marks: pl.DataFrame,
    audit: pl.DataFrame,
    expiry_ids: set[str],
) -> None:
    if result.shape != source.shape or result["policy_path_id"].n_unique() != result.height:
        raise AssertionError("expiry close overlay changed path population")
    source_nonexpiry = source.filter(~pl.col("policy_path_id").is_in(expiry_ids)).sort(
        "policy_path_id"
    )
    result_nonexpiry = result.filter(~pl.col("policy_path_id").is_in(expiry_ids)).sort(
        "policy_path_id"
    )
    if source_nonexpiry.rows() != result_nonexpiry.rows():
        raise AssertionError("expiry close overlay changed a non-expiry path")
    selected = result.filter(pl.col("policy_path_id").is_in(expiry_ids))
    invalid = selected.filter(
        (pl.col("supplemental_terminal_resolution") != TERMINAL_RESOLUTION)
        | (pl.col("expiry_mark_is_official_close") != True)  # noqa: E712
        | (pl.col("expiry_mark_is_official_settlement") != False)  # noqa: E712
        | (pl.col("expiry_mark_uses_trade_fallback") != False)  # noqa: E712
        | (pl.col("terminal_cashflow_priced") != True)  # noqa: E712
    )
    if invalid.height or audit.height != len(expiry_ids):
        raise AssertionError("expiry close overlay path flags are incoherent")
    if marks.filter(
        (pl.col("mark_is_official_close") != True)  # noqa: E712
        | (pl.col("mark_is_official_settlement") != False)  # noqa: E712
        | (pl.col("mark_uses_trade_fallback") != False)  # noqa: E712
    ).height:
        raise AssertionError("expiry close mark flags are incoherent")


def _require(frame: pl.DataFrame, required: set[str], label: str) -> None:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError(f"{label} must be a polars DataFrame")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _positive_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite and positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


__all__ = [
    "EXACT_PRICE_SOURCE",
    "ExpiryCloseOverlayResult",
    "FUTURE_CLOSE_SOURCE",
    "OVERLAY_VERSION",
    "SPOT_CLOSE_SOURCE",
    "TERMINAL_REASON",
    "TERMINAL_RESOLUTION",
    "apply_expiry_daily_close_overlay",
]
