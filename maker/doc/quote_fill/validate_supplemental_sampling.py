"""Compare supplemental 1-second state sampling with exact cached-state replay.

This is a bounded validation harness, not a production backtest.  It chooses
unknown continuation product-sessions before looking at either replay outcome,
stratifies them by calendar period and cached event activity, and replays the
same frozen selected policy trials twice:

* every cached spot/futures state (reference), and
* final state per receive-time second, retaining every spot SpreadPair epoch
  transition and all trade prints (supplemental approximation).

Outputs are intentionally self-contained under ``--output-root``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import gc
import hashlib
import json
import math
from pathlib import Path
import resource
import tempfile
import time
from typing import Mapping, Sequence

import polars as pl

from maker.src.quote_fill.exit_maker_cross_session import (
    CrossSessionExitMakerSession,
    _classify_session_policy,
    _prepare_candidate_tape,
    _session_start_time,
)
from maker.src.quote_fill.supplemental_carry_runner import (
    DEFAULT_CANDIDATE_CACHE_ROOT,
    DEFAULT_PREREQUISITE_ROOT,
    _default_policy_day_replayer,
    load_cached_candidate_session,
    load_supplemental_replay_inputs,
    thin_candidate_session_one_second,
)


RUN_ID = "supplemental_sampling_sensitivity_v1"
DEFAULT_OUTPUT_ROOT = Path(
    "maker/data/walkforward/supplemental_sampling_sensitivity_20260821_v1"
)


@dataclass(frozen=True)
class CandidateGroup:
    session_date: str
    value_code: str
    quote_code: str
    date_stratum: str
    activity_stratum: str
    state_rows: int
    trade_rows: int
    path_ids: tuple[str, ...]
    entry_ids: tuple[str, ...]
    trial_ids: tuple[str, ...]


def _next_session_lookup(sessions: Sequence[str]) -> dict[str, str]:
    return {sessions[index]: sessions[index + 1] for index in range(len(sessions) - 1)}


def _partition_marker(
    root: Path, date: str, value: str, quote: str
) -> tuple[Path, Mapping[str, object]] | None:
    parent = root / f"Date={date}" / f"ValueCode={value}" / f"QuoteCode={quote}"
    markers = sorted(parent.glob("Key=*/complete.json")) if parent.is_dir() else []
    if len(markers) != 1:
        return None
    marker = json.loads(markers[0].read_text(encoding="utf-8"))
    if marker.get("complete") is not True:
        return None
    return markers[0], marker


def _artifact_rows(marker: Mapping[str, object], filename: str) -> int:
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("candidate marker has no artifact inventory")
    record = artifacts.get(filename)
    if not isinstance(record, dict):
        raise ValueError(f"candidate marker has no {filename}")
    return int(record["rows"])


def _activity_stratum(value: int, lower: float, upper: float) -> str:
    if value <= lower:
        return "low"
    if value <= upper:
        return "middle"
    return "high"


def _stable_key(date: str, value: str, quote: str) -> str:
    return hashlib.sha256(f"{RUN_ID}|{date}|{value}|{quote}".encode()).hexdigest()


def build_candidate_groups(
    paths: pl.DataFrame,
    *,
    sessions: Sequence[str],
    contract_calendar: pl.DataFrame,
    cache_root: Path,
) -> tuple[list[CandidateGroup], pl.DataFrame]:
    """Build first-continuation-session groups without inspecting outcomes."""

    next_session = _next_session_lookup(sessions)
    expiry_by_quote = {
        str(row["QuoteCode"]): str(row["expiry_session"])
        for row in contract_calendar.iter_rows(named=True)
    }
    unknown = paths.filter(pl.col("outcome_status") == "maker_fill_state_unknown")
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    excluded: list[dict[str, object]] = []
    for row in unknown.iter_rows(named=True):
        last = str(row["last_observed_session_date"])
        session_date = next_session.get(last)
        quote = str(row["QuoteCode"])
        expiry = expiry_by_quote.get(quote)
        reason: str | None = None
        if session_date is None:
            reason = "no_following_candidate_session"
        elif expiry is None:
            reason = "contract_calendar_missing"
        elif session_date > expiry:
            reason = "following_session_after_expiry"
        value = str(row["ValueCode"])
        if reason is None and _partition_marker(
            cache_root, session_date, value, quote
        ) is None:
            reason = "immediate_next_session_cache_missing"
        if reason is not None:
            excluded.append(
                {
                    "policy_path_id": str(row["policy_path_id"]),
                    "ValueCode": value,
                    "QuoteCode": quote,
                    "last_observed_session_date": last,
                    "candidate_session_date": session_date,
                    "exclusion_reason": reason,
                }
            )
            continue
        assert session_date is not None
        grouped.setdefault((session_date, value, quote), []).append(row)

    preliminary: list[dict[str, object]] = []
    for (date, value, quote), rows in sorted(grouped.items()):
        marker_result = _partition_marker(cache_root, date, value, quote)
        assert marker_result is not None
        _, marker = marker_result
        state_rows = _artifact_rows(marker, "spot_states.parquet") + _artifact_rows(
            marker, "future_states.parquet"
        )
        trade_rows = _artifact_rows(marker, "spot_trades.parquet") + _artifact_rows(
            marker, "future_trades.parquet"
        )
        preliminary.append(
            {
                "session_date": date,
                "value_code": value,
                "quote_code": quote,
                "state_rows": state_rows,
                "trade_rows": trade_rows,
                "rows": rows,
            }
        )
    if not preliminary:
        raise ValueError("no cache-backed unknown continuation groups")

    dates = sorted({str(row["session_date"]) for row in preliminary})
    date_rank = {date: index for index, date in enumerate(dates)}
    state_values = pl.Series([int(row["state_rows"]) for row in preliminary])
    lower = float(state_values.quantile(1.0 / 3.0, interpolation="nearest"))
    upper = float(state_values.quantile(2.0 / 3.0, interpolation="nearest"))
    groups: list[CandidateGroup] = []
    for item in preliminary:
        rank = date_rank[str(item["session_date"])]
        fraction = rank / max(len(dates), 1)
        date_stratum = "early" if fraction < 1 / 3 else (
            "middle" if fraction < 2 / 3 else "late"
        )
        rows = list(item["rows"])
        groups.append(
            CandidateGroup(
                session_date=str(item["session_date"]),
                value_code=str(item["value_code"]),
                quote_code=str(item["quote_code"]),
                date_stratum=date_stratum,
                activity_stratum=_activity_stratum(
                    int(item["state_rows"]), lower, upper
                ),
                state_rows=int(item["state_rows"]),
                trade_rows=int(item["trade_rows"]),
                path_ids=tuple(sorted(str(row["policy_path_id"]) for row in rows)),
                entry_ids=tuple(
                    sorted(str(row["entry_policy_generation_id"]) for row in rows)
                ),
                trial_ids=tuple(
                    sorted(str(row["exit_policy_trial_id"]) for row in rows)
                ),
            )
        )
    exclusion_frame = (
        pl.from_dicts(excluded, infer_schema_length=None)
        if excluded
        else pl.DataFrame(
            schema={
                "policy_path_id": pl.String,
                "ValueCode": pl.String,
                "QuoteCode": pl.String,
                "last_observed_session_date": pl.String,
                "candidate_session_date": pl.String,
                "exclusion_reason": pl.String,
            }
        )
    )
    return groups, exclusion_frame


def select_groups(groups: Sequence[CandidateGroup], count: int) -> list[CandidateGroup]:
    """Select outcome-blind, balanced date/activity cells deterministically."""

    if count < 1:
        raise ValueError("sample count must be positive")
    cells: dict[tuple[str, str], list[CandidateGroup]] = {}
    for group in groups:
        cells.setdefault((group.date_stratum, group.activity_stratum), []).append(group)
    for values in cells.values():
        values.sort(
            key=lambda group: _stable_key(
                group.session_date, group.value_code, group.quote_code
            )
        )
    selected: list[CandidateGroup] = []
    cell_order = [
        (date, activity)
        for date in ("early", "middle", "late")
        for activity in ("low", "middle", "high")
    ]
    round_index = 0
    while len(selected) < min(count, len(groups)):
        added = False
        for cell in cell_order:
            values = cells.get(cell, [])
            if round_index < len(values):
                selected.append(values[round_index])
                added = True
                if len(selected) == min(count, len(groups)):
                    break
        if not added:
            break
        round_index += 1
    return sorted(
        selected,
        key=lambda group: (group.session_date, group.value_code, group.quote_code),
    )


def _scenario_classification(row: Mapping[str, object]) -> tuple[str, str]:
    transition, status, _ = _classify_session_policy(
        row, cancel_semantics="nominal_instant_cancel_v0"
    )
    if transition == "terminal":
        return "terminal", status
    if transition == "carry":
        return "carry", status
    if status == "maker_fill_state_unknown":
        return "imputed_full_carry_unknown", status
    return "blocked_censored", status


def _run_policy_day(
    *,
    actions: pl.DataFrame,
    exits: pl.DataFrame,
    session: CrossSessionExitMakerSession,
    prepared,
    entry_ids: Sequence[str],
    trial_ids: Sequence[str],
    spool_root: Path,
) -> pl.DataFrame:
    start_ns = _session_start_time(session, prepared)
    selected_actions = actions.filter(
        pl.col("policy_generation_id").cast(pl.String).is_in(entry_ids)
    ).with_columns(
        pl.lit(session.date).alias("Date"),
        pl.lit(start_ns).cast(pl.Int64).alias("entry_hedge_decision_time_ns"),
    )
    selected_exits = exits.filter(
        pl.col("policy_generation_id").cast(pl.String).is_in(entry_ids)
    ).with_columns(pl.lit(session.date).alias("Date"))
    return _default_policy_day_replayer(
        selected_actions,
        selected_exits,
        CrossSessionExitMakerSession(
            **{**session.__dict__, "raw_tape": prepared}
        ),
        trial_ids,
        spool_root,
    )


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _trial_record(
    *,
    group: CandidateGroup,
    mode: str,
    row: Mapping[str, object],
    path: Mapping[str, object],
    replay_seconds: float,
    prepare_seconds: float,
) -> dict[str, object]:
    scenario_class, raw_status = _scenario_classification(row)
    gross = _safe_float(row.get("gross_cycle_pnl_twd"))
    notional = float(path["normalization_notional_twd"])
    gross_bp = None if gross is None else gross / notional * 10_000.0
    terminal = scenario_class == "terminal"
    return {
        "sample_session_date": group.session_date,
        "ValueCode": group.value_code,
        "QuoteCode": group.quote_code,
        "date_stratum": group.date_stratum,
        "activity_stratum": group.activity_stratum,
        "policy_path_id": str(path["policy_path_id"]),
        "entry_policy_generation_id": str(path["entry_policy_generation_id"]),
        "exit_policy_trial_id": str(path["exit_policy_trial_id"]),
        "exit_rule_id": str(path["exit_rule_id"]),
        "frozen_exit_threshold_basis_bp": float(path["exit_threshold_basis_bp"]),
        "mode": mode,
        "scenario_classification": scenario_class,
        "raw_transition_status": raw_status,
        "terminal": terminal,
        "terminal_date": group.session_date if terminal else None,
        "winner_route": str(row["exit_route"]) if terminal else None,
        "winner_generation_id": (
            str(row["oco_winner_generation_id"])
            if terminal and row.get("oco_winner_generation_id") is not None
            else None
        ),
        "terminal_time_ns": int(row["exit_decision_time_ns"])
        if terminal
        else None,
        "exit_spot_price": _safe_float(row.get("exit_spot_price")) if terminal else None,
        "exit_future_price": _safe_float(row.get("exit_future_price"))
        if terminal
        else None,
        "gross_pnl_twd": gross if terminal else None,
        "gross_pnl_bp": gross_bp if terminal else None,
        "normalization_notional_twd": notional,
        "branch_status": str(row.get("branch_status")),
        "nominal_branch": str(row.get("nominal_instant_cancel_v0_branch")),
        "prepare_seconds_group": prepare_seconds,
        "replay_seconds_group": replay_seconds,
    }


def replay_group(
    group: CandidateGroup,
    *,
    paths_by_trial: Mapping[str, Mapping[str, object]],
    actions: pl.DataFrame,
    exits: pl.DataFrame,
    cache_root: Path,
    spool_root: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    load_started = time.perf_counter()
    exact_session, _ = load_cached_candidate_session(
        cache_root, group.session_date, group.value_code, group.quote_code
    )
    load_seconds = time.perf_counter() - load_started
    source_raw = exact_session.raw_tape

    exact_started = time.perf_counter()
    exact_prepared, status = _prepare_candidate_tape(
        exact_session,
        value_code=group.value_code,
        exact_quote_code=group.quote_code,
    )
    if exact_prepared is None:
        raise ValueError(f"exact candidate preparation failed: {status}")
    exact_prepare_seconds = time.perf_counter() - exact_started
    replay_started = time.perf_counter()
    exact_rows = _run_policy_day(
        actions=actions,
        exits=exits,
        session=exact_session,
        prepared=exact_prepared,
        entry_ids=group.entry_ids,
        trial_ids=group.trial_ids,
        spool_root=spool_root / "exact",
    )
    exact_replay_seconds = time.perf_counter() - replay_started
    exact_records = [
        _trial_record(
            group=group,
            mode="exact_all_cached_states",
            row=row,
            path=paths_by_trial[str(row["exit_policy_trial_id"])],
            replay_seconds=exact_replay_seconds,
            prepare_seconds=exact_prepare_seconds,
        )
        for row in exact_rows.iter_rows(named=True)
    ]
    del exact_rows, exact_prepared
    gc.collect()

    approx_started = time.perf_counter()
    approx_session, sampling = thin_candidate_session_one_second(
        exact_session,
        value_code=group.value_code,
        quote_code=group.quote_code,
    )
    approx_prepared, status = _prepare_candidate_tape(
        approx_session,
        value_code=group.value_code,
        exact_quote_code=group.quote_code,
    )
    if approx_prepared is None:
        raise ValueError(f"approx candidate preparation failed: {status}")
    approx_prepare_seconds = time.perf_counter() - approx_started
    replay_started = time.perf_counter()
    approx_rows = _run_policy_day(
        actions=actions,
        exits=exits,
        session=approx_session,
        prepared=approx_prepared,
        entry_ids=group.entry_ids,
        trial_ids=group.trial_ids,
        spool_root=spool_root / "approx",
    )
    approx_replay_seconds = time.perf_counter() - replay_started
    approx_records = [
        _trial_record(
            group=group,
            mode="one_second_plus_spread_epoch",
            row=row,
            path=paths_by_trial[str(row["exit_policy_trial_id"])],
            replay_seconds=approx_replay_seconds,
            prepare_seconds=approx_prepare_seconds,
        )
        for row in approx_rows.iter_rows(named=True)
    ]
    group_audit = {
        "sample_session_date": group.session_date,
        "ValueCode": group.value_code,
        "QuoteCode": group.quote_code,
        "date_stratum": group.date_stratum,
        "activity_stratum": group.activity_stratum,
        "selected_trial_count": len(group.trial_ids),
        "cache_state_rows": group.state_rows,
        "cache_trade_rows": group.trade_rows,
        "load_seconds": load_seconds,
        "exact_prepare_seconds": exact_prepare_seconds,
        "exact_replay_seconds": exact_replay_seconds,
        "approx_sample_and_prepare_seconds": approx_prepare_seconds,
        "approx_replay_seconds": approx_replay_seconds,
        **sampling,
    }
    del approx_rows, approx_prepared, approx_session, exact_session, source_raw
    gc.collect()
    return exact_records + approx_records, group_audit


def compare_trials(records: pl.DataFrame) -> pl.DataFrame:
    exact = records.filter(pl.col("mode") == "exact_all_cached_states").drop("mode")
    approx = records.filter(pl.col("mode") == "one_second_plus_spread_epoch").drop(
        "mode"
    )
    keys = ["sample_session_date", "ValueCode", "QuoteCode", "exit_policy_trial_id"]
    compared = exact.join(approx, on=keys, how="inner", suffix="_approx", validate="1:1")
    return compared.with_columns(
        (
            pl.col("scenario_classification")
            == pl.col("scenario_classification_approx")
        ).alias("classification_agree"),
        (pl.col("terminal") == pl.col("terminal_approx")).alias(
            "terminal_indicator_agree"
        ),
        pl.when(pl.col("terminal") & pl.col("terminal_approx"))
        .then(pl.col("winner_route") == pl.col("winner_route_approx"))
        .otherwise(None)
        .alias("winner_route_agree"),
        pl.when(pl.col("terminal") & pl.col("terminal_approx"))
        .then(pl.col("terminal_time_ns_approx") - pl.col("terminal_time_ns"))
        .otherwise(None)
        .alias("terminal_time_error_ns"),
        pl.when(pl.col("terminal") & pl.col("terminal_approx"))
        .then(pl.col("exit_spot_price_approx") - pl.col("exit_spot_price"))
        .otherwise(None)
        .alias("exit_spot_price_error"),
        pl.when(pl.col("terminal") & pl.col("terminal_approx"))
        .then(pl.col("exit_future_price_approx") - pl.col("exit_future_price"))
        .otherwise(None)
        .alias("exit_future_price_error"),
        pl.when(pl.col("terminal") & pl.col("terminal_approx"))
        .then(pl.col("gross_pnl_bp_approx") - pl.col("gross_pnl_bp"))
        .otherwise(None)
        .alias("gross_pnl_bp_error"),
    ).with_columns(
        pl.col("terminal_time_error_ns").abs().alias("abs_terminal_time_error_ns"),
        pl.col("exit_spot_price_error").abs().alias("abs_exit_spot_price_error"),
        pl.col("exit_future_price_error").abs().alias("abs_exit_future_price_error"),
        pl.col("gross_pnl_bp_error").abs().alias("abs_gross_pnl_bp_error"),
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _quantile(frame: pl.DataFrame, column: str, quantile: float) -> float | None:
    values = frame[column].drop_nulls()
    if values.is_empty():
        return None
    result = values.quantile(quantile, interpolation="linear")
    return None if result is None else float(result)


def build_summary(
    comparisons: pl.DataFrame,
    group_audit: pl.DataFrame,
    *,
    eligible_group_count: int,
    excluded_path_count: int,
    elapsed_seconds: float,
) -> dict[str, object]:
    n = comparisons.height
    exact_terminal = comparisons.filter(pl.col("terminal")).height
    approx_terminal = comparisons.filter(pl.col("terminal_approx")).height
    both_terminal_frame = comparisons.filter(
        pl.col("terminal") & pl.col("terminal_approx")
    )
    both_terminal = both_terminal_frame.height
    classification_matches = comparisons.filter(pl.col("classification_agree")).height
    terminal_matches = comparisons.filter(pl.col("terminal_indicator_agree")).height
    route_matches = both_terminal_frame.filter(pl.col("winner_route_agree")).height
    terminal_date_matches = both_terminal_frame.filter(
        pl.col("terminal_date") == pl.col("terminal_date_approx")
    ).height
    exact_time_matches = both_terminal_frame.filter(
        pl.col("terminal_time_error_ns") == 0
    ).height
    exact_spot_matches = both_terminal_frame.filter(
        pl.col("exit_spot_price_error").abs() < 1e-12
    ).height
    exact_future_matches = both_terminal_frame.filter(
        pl.col("exit_future_price_error").abs() < 1e-12
    ).height
    exact_pnl_matches = both_terminal_frame.filter(
        pl.col("gross_pnl_bp_error").abs() < 1e-12
    ).height
    both_price_matches = both_terminal_frame.filter(
        (pl.col("exit_spot_price_error").abs() < 1e-12)
        & (pl.col("exit_future_price_error").abs() < 1e-12)
    ).height
    full_terminal_matches = both_terminal_frame.filter(
        (pl.col("terminal_date") == pl.col("terminal_date_approx"))
        & pl.col("winner_route_agree")
        & (pl.col("terminal_time_error_ns") == 0)
        & (pl.col("exit_spot_price_error").abs() < 1e-12)
        & (pl.col("exit_future_price_error").abs() < 1e-12)
        & (pl.col("gross_pnl_bp_error").abs() < 1e-12)
    ).height
    exact_runtime = float(
        group_audit.select(
            (pl.col("exact_prepare_seconds") + pl.col("exact_replay_seconds")).sum()
        ).item()
    )
    approx_runtime = float(
        group_audit.select(
            (
                pl.col("approx_sample_and_prepare_seconds")
                + pl.col("approx_replay_seconds")
            ).sum()
        ).item()
    )
    source_states = int(group_audit["spot_states_source_rows"].sum()) + int(
        group_audit["future_states_source_rows"].sum()
    )
    sampled_states = int(group_audit["spot_states_replay_rows"].sum()) + int(
        group_audit["future_states_replay_rows"].sum()
    )
    rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    exact_routes = sorted(
        str(value)
        for value in both_terminal_frame["winner_route"].drop_nulls().unique()
    )
    exact_classes = {
        str(row["scenario_classification"]): int(row["len"])
        for row in comparisons.group_by("scenario_classification")
        .len()
        .iter_rows(named=True)
    }
    pnl_tolerance_rates = {
        f"le_{tolerance:g}_bp": _ratio(
            both_terminal_frame.filter(
                pl.col("abs_gross_pnl_bp_error") <= tolerance
            ).height,
            both_terminal,
        )
        for tolerance in (1.0, 5.0, 10.0, 20.0, 30.0)
    }
    return {
        "run_id": RUN_ID,
        "sample_design": (
            "outcome-blind deterministic balance across early/middle/late and "
            "low/middle/high cached-state activity; first continuation session only"
        ),
        "eligible_product_session_groups": eligible_group_count,
        "sampled_product_session_groups": group_audit.height,
        "sampled_selected_policy_trials": n,
        "excluded_unknown_paths_before_sampling": excluded_path_count,
        "exact_terminal_trials": exact_terminal,
        "approx_terminal_trials": approx_terminal,
        "both_terminal_trials": both_terminal,
        "classification_agreement_count": classification_matches,
        "classification_agreement_rate": _ratio(classification_matches, n),
        "terminal_indicator_agreement_count": terminal_matches,
        "terminal_indicator_agreement_rate": _ratio(terminal_matches, n),
        "exact_terminal_recall": _ratio(both_terminal, exact_terminal),
        "approx_terminal_precision": _ratio(both_terminal, approx_terminal),
        "winner_route_agreement_count": route_matches,
        "winner_route_agreement_rate_among_both_terminal": _ratio(
            route_matches, both_terminal
        ),
        "exact_winner_routes_observed": exact_routes,
        "winner_route_scope_note": (
            "The selected normal-exit challenger has one route, "
            "future_bid_spot_taker; this does not validate a two-route OCO choice."
        ),
        "terminal_date_match_rate_among_both_terminal": _ratio(
            terminal_date_matches, both_terminal
        ),
        "exact_terminal_time_match_rate_among_both_terminal": _ratio(
            exact_time_matches, both_terminal
        ),
        "exact_spot_price_match_rate_among_both_terminal": _ratio(
            exact_spot_matches, both_terminal
        ),
        "exact_future_price_match_rate_among_both_terminal": _ratio(
            exact_future_matches, both_terminal
        ),
        "exact_gross_pnl_bp_match_rate_among_both_terminal": _ratio(
            exact_pnl_matches, both_terminal
        ),
        "both_exit_prices_exact_match_rate_among_both_terminal": _ratio(
            both_price_matches, both_terminal
        ),
        "full_terminal_tuple_exact_match_rate_among_both_terminal": _ratio(
            full_terminal_matches, both_terminal
        ),
        "gross_pnl_abs_error_tolerance_rates": pnl_tolerance_rates,
        "exact_scenario_classification_counts": exact_classes,
        "abs_terminal_time_error_ms_p50": (
            None
            if both_terminal == 0
            else _quantile(both_terminal_frame, "abs_terminal_time_error_ns", 0.5)
            / 1_000_000.0
        ),
        "abs_terminal_time_error_ms_p95": (
            None
            if both_terminal == 0
            else _quantile(both_terminal_frame, "abs_terminal_time_error_ns", 0.95)
            / 1_000_000.0
        ),
        "abs_terminal_time_error_ms_max": (
            None
            if both_terminal == 0
            else float(both_terminal_frame["abs_terminal_time_error_ns"].max())
            / 1_000_000.0
        ),
        "abs_exit_spot_price_error_p95": _quantile(
            both_terminal_frame, "abs_exit_spot_price_error", 0.95
        ),
        "abs_exit_future_price_error_p95": _quantile(
            both_terminal_frame, "abs_exit_future_price_error", 0.95
        ),
        "abs_gross_pnl_bp_error_p50": _quantile(
            both_terminal_frame, "abs_gross_pnl_bp_error", 0.5
        ),
        "abs_gross_pnl_bp_error_p95": _quantile(
            both_terminal_frame, "abs_gross_pnl_bp_error", 0.95
        ),
        "abs_gross_pnl_bp_error_max": (
            None
            if both_terminal == 0
            else float(both_terminal_frame["abs_gross_pnl_bp_error"].max())
        ),
        "signed_gross_pnl_bp_error_mean": (
            None
            if both_terminal == 0
            else float(both_terminal_frame["gross_pnl_bp_error"].mean())
        ),
        "source_state_rows": source_states,
        "sampled_state_rows": sampled_states,
        "state_row_retention_rate": _ratio(sampled_states, source_states),
        "spot_state_row_retention_rate": _ratio(
            int(group_audit["spot_states_replay_rows"].sum()),
            int(group_audit["spot_states_source_rows"].sum()),
        ),
        "future_state_row_retention_rate": _ratio(
            int(group_audit["future_states_replay_rows"].sum()),
            int(group_audit["future_states_source_rows"].sum()),
        ),
        "exact_prepare_plus_replay_seconds": exact_runtime,
        "approx_sample_prepare_plus_replay_seconds": approx_runtime,
        "approx_speedup_multiple": (
            exact_runtime / approx_runtime if approx_runtime > 0 else None
        ),
        "candidate_cache_load_seconds": float(group_audit["load_seconds"].sum()),
        "wall_elapsed_seconds": elapsed_seconds,
        "process_peak_rss_mib": rss_kib / 1024.0,
        "scope_note": (
            "Accuracy applies to the first cache-backed continuation session of "
            "selected maker_fill_state_unknown paths, under nominal instant-cancel "
            "and the imputed-full-carry scenario. It does not validate the assumption "
            "that the original unknown session really had zero exit fills."
        ),
    }


def _pct(value: object) -> str:
    return "n/a" if value is None else f"{100 * float(value):.1f}%"


def render_markdown(summary: Mapping[str, object], comparisons: pl.DataFrame) -> str:
    mismatches = comparisons.filter(~pl.col("classification_agree"))
    lines = [
        "# Supplemental 1-second sampling sensitivity",
        "",
        "Reference: every cached spot/futures state. Approximation: final state per "
        "receive-time second, plus every spot SpreadPair epoch transition; all raw "
        "trade prints are retained.",
        "",
        "## Result",
        "",
        f"- Sample: {summary['sampled_product_session_groups']} product-sessions, "
        f"{summary['sampled_selected_policy_trials']} actual selected frozen-lower trials.",
        f"- Scenario classification agreement: {_pct(summary['classification_agreement_rate'])}.",
        f"- Terminal/non-terminal agreement: {_pct(summary['terminal_indicator_agreement_rate'])}; "
        f"exact terminals={summary['exact_terminal_trials']}, approximate terminals="
        f"{summary['approx_terminal_trials']}.",
        f"- Among both-terminal trials, winner-route agreement: "
        f"{_pct(summary['winner_route_agreement_rate_among_both_terminal'])}; exact "
        f"terminal-date match: {_pct(summary['terminal_date_match_rate_among_both_terminal'])}; "
        f"exact terminal timestamp match: "
        f"{_pct(summary['exact_terminal_time_match_rate_among_both_terminal'])}.",
        f"- Exact spot/futures/both-leg price match: "
        f"{_pct(summary['exact_spot_price_match_rate_among_both_terminal'])} / "
        f"{_pct(summary['exact_future_price_match_rate_among_both_terminal'])} / "
        f"{_pct(summary['both_exit_prices_exact_match_rate_among_both_terminal'])}.",
        f"- Exact gross-PnL-bp match: "
        f"{_pct(summary['exact_gross_pnl_bp_match_rate_among_both_terminal'])}; "
        f"full terminal tuple exact match: "
        f"{_pct(summary['full_terminal_tuple_exact_match_rate_among_both_terminal'])}.",
        f"- Among both-terminal trials, |gross PnL error| p50/p95/max: "
        f"{summary['abs_gross_pnl_bp_error_p50']!r} / "
        f"{summary['abs_gross_pnl_bp_error_p95']!r} / "
        f"{summary['abs_gross_pnl_bp_error_max']!r} bp.",
        f"- State rows retained: {_pct(summary['state_row_retention_rate'])}; "
        f"measured state-prepare+replay speedup: "
        f"{float(summary['approx_speedup_multiple']):.2f}x.",
        f"- Wall time: {float(summary['wall_elapsed_seconds']):.1f}s; process peak RSS: "
        f"{float(summary['process_peak_rss_mib']):.0f} MiB.",
        "",
        "## Interpretation boundary",
        "",
        str(summary["scope_note"]),
        "The sample was chosen before outcomes were replayed, balanced by calendar "
        "third and cached-state activity third. Results are sensitivity evidence, not "
        "a proof of exact equivalence or a replacement for the full exact backtest.",
        str(summary["winner_route_scope_note"]),
        "",
        "## Classification mismatches",
        "",
    ]
    if mismatches.is_empty():
        lines.append("None in this sample.")
    else:
        lines.extend(
            [
                "| Date | Product | Exact | Approx |",
                "|---|---:|---|---|",
            ]
        )
        for row in mismatches.select(
            "sample_session_date",
            "ValueCode",
            "scenario_classification",
            "scenario_classification_approx",
        ).iter_rows(named=True):
            lines.append(
                f"| {row['sample_session_date']} | {row['ValueCode']} | "
                f"{row['scenario_classification']} | "
                f"{row['scenario_classification_approx']} |"
            )
    lines.append("")
    return "\n".join(lines)


def run(output_root: Path, sample_count: int) -> dict[str, object]:
    started = time.perf_counter()
    inputs = load_supplemental_replay_inputs()
    prerequisite = DEFAULT_PREREQUISITE_ROOT
    sessions = (prerequisite / "candidate_sessions.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    calendar = pl.read_parquet(prerequisite / "exact_contract_calendar_v1.parquet")
    groups, exclusions = build_candidate_groups(
        inputs.paths,
        sessions=sessions,
        contract_calendar=calendar,
        cache_root=DEFAULT_CANDIDATE_CACHE_ROOT,
    )
    selected = select_groups(groups, sample_count)
    selected_trials = {trial for group in selected for trial in group.trial_ids}
    path_rows = {
        str(row["exit_policy_trial_id"]): row
        for row in inputs.paths.filter(
            pl.col("exit_policy_trial_id").is_in(sorted(selected_trials))
        ).iter_rows(named=True)
    }
    if set(path_rows) != selected_trials:
        raise ValueError("selected trials do not map one-to-one to source paths")

    all_records: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sensitivity-spool-", dir="/tmp") as spool:
        for index, group in enumerate(selected, start=1):
            group_records, audit = replay_group(
                group,
                paths_by_trial=path_rows,
                actions=inputs.actions,
                exits=inputs.exits,
                cache_root=DEFAULT_CANDIDATE_CACHE_ROOT,
                spool_root=Path(spool),
            )
            all_records.extend(group_records)
            audits.append(audit)
            print(
                f"[{index}/{len(selected)}] {group.session_date} "
                f"{group.value_code}/{group.quote_code}: "
                f"exact={audit['exact_prepare_seconds'] + audit['exact_replay_seconds']:.2f}s "
                f"approx={audit['approx_sample_and_prepare_seconds'] + audit['approx_replay_seconds']:.2f}s",
                flush=True,
            )
    records = pl.from_dicts(all_records, infer_schema_length=None)
    group_audit = pl.from_dicts(audits, infer_schema_length=None)
    comparisons = compare_trials(records)
    summary = build_summary(
        comparisons,
        group_audit,
        eligible_group_count=len(groups),
        excluded_path_count=exclusions.height,
        elapsed_seconds=time.perf_counter() - started,
    )
    records.write_parquet(output_root / "replay_records.parquet")
    comparisons.write_parquet(output_root / "trial_comparisons.parquet")
    group_audit.write_parquet(output_root / "product_session_runtime_audit.parquet")
    exclusions.write_parquet(output_root / "pre_sampling_exclusions.parquet")
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "README.md").write_text(
        render_markdown(summary, comparisons), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-count", type=int, default=15)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    summary = run(args.output_root, args.sample_count)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
