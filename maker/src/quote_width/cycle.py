"""Non-overlapping upper-to-lower latent basis cycles on a causal 1-second grid.

The output describes price-path opportunities.  It does not infer maker fills,
queue position, 50 ms hedge cost, or executable PnL.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import polars as pl

from ..common.paths import DEFAULT_OUTPUT_ROOT, MAKER_ROOT
from .table import ANCHOR_COLUMN, build_width_candidates


DEFAULT_DIAGNOSTIC_WIDTH_POLICIES = (
    "fixed_10bp",
    "fixed_15bp",
    "fixed_20bp",
    "fixed_30bp",
    "prior_excursion_p50",
    "prior_excursion_p80",
)
DEFAULT_DIAGNOSTIC_FIXED_WIDTHS_BP = (10.0, 15.0, 20.0, 30.0)
DEFAULT_EXIT_WIDTH_RATIOS = (0.0, 0.5, 1.0)
DEFAULT_ANCHOR_MODES = ("dynamic", "frozen_entry")
DEFAULT_EXIT_DELAYS_SECONDS = (1, 30)
DEFAULT_SAMPLES = ("base", "fresh_1000ms")
DEFAULT_HORIZONS_SECONDS = (60, 120, 300, 600)
SESSION_START_SECONDS = 300
SESSION_HOURS = 4.25
VALUE_EPS = 1e-9

POLICY_KEYS = [
    "Date",
    "ValueCode",
    "QuoteCode",
    "sample",
    "anchor_mode",
    "width_family",
    "width_policy",
    "candidate_width_bp",
    "exit_width_ratio",
    "exit_width_bp",
    "exit_delay_seconds",
    "max_horizon_seconds",
]


@dataclass(frozen=True)
class CycleStudyResult:
    cycles: pl.DataFrame
    by_day_symbol: pl.DataFrame
    policy_summary: pl.DataFrame
    diagnostic_frontier: pl.DataFrame
    dynamic_symmetric_diagnostic_summary: pl.DataFrame


def load_cycle_width_candidates(
    parameter_path: Path,
    policies: Iterable[str] = DEFAULT_DIAGNOSTIC_WIDTH_POLICIES,
) -> pl.DataFrame:
    """Load fixed-grid and prior-width controls for latent diagnostics only."""
    parameters = pl.read_csv(
        parameter_path,
        schema_overrides={
            "Date": pl.String,
            "ValueCode": pl.String,
            "QuoteCode": pl.String,
            "prior_date": pl.String,
        },
        infer_schema_length=10_000,
    )
    selected = build_width_candidates(
        parameters,
        fixed_widths_bp=DEFAULT_DIAGNOSTIC_FIXED_WIDTHS_BP,
    ).filter(
        pl.col("width_policy").is_in(list(policies))
    )
    if selected.is_empty():
        raise ValueError("no cycle width candidates survived the requested policy filter")
    return selected.sort(["Date", "ValueCode", "candidate_width_bp", "width_policy"])


def build_cycle_policy_universe(
    candidates: pl.DataFrame,
    exit_width_ratios: Iterable[float] = DEFAULT_EXIT_WIDTH_RATIOS,
    anchor_modes: Iterable[str] = DEFAULT_ANCHOR_MODES,
    exit_delays_seconds: Iterable[int] = DEFAULT_EXIT_DELAYS_SECONDS,
    samples: Iterable[str] = DEFAULT_SAMPLES,
    max_horizon_seconds: int = 0,
) -> pl.DataFrame:
    """Cross daily width candidates with independent exit-policy dimensions."""
    if max_horizon_seconds != 0:
        raise ValueError(
            "hard TTL is disabled until a force-flat execution rule exists"
        )
    ratios = tuple(float(value) for value in exit_width_ratios)
    delays_raw = tuple(float(value) for value in exit_delays_seconds)
    if not ratios or any(not math.isfinite(value) or value < 0 for value in ratios):
        raise ValueError("exit width ratios must be finite and non-negative")
    if not delays_raw or any(
        not math.isfinite(value) or value < 1 or not value.is_integer()
        for value in delays_raw
    ):
        raise ValueError("exit delays must be positive whole seconds")
    if candidates.filter(
        ~pl.col("candidate_width_bp").is_finite()
        | (pl.col("candidate_width_bp") <= 0)
    ).height:
        raise ValueError("candidate widths must be finite and positive")
    dimensions = pl.DataFrame(
        [
            {
                "sample": sample,
                "anchor_mode": anchor_mode,
                "exit_width_ratio": float(exit_ratio),
                "exit_delay_seconds": int(delay),
                "max_horizon_seconds": int(max_horizon_seconds),
            }
            for sample in samples
            for anchor_mode in anchor_modes
            for exit_ratio in ratios
            for delay in delays_raw
        ]
    )
    unknown_modes = set(dimensions["anchor_mode"].unique()) - {
        "dynamic",
        "frozen_entry",
    }
    if unknown_modes:
        raise ValueError(f"unknown anchor modes: {sorted(unknown_modes)}")
    unknown_samples = set(dimensions["sample"].unique()) - {
        "base",
        "fresh_1000ms",
    }
    if unknown_samples:
        raise ValueError(f"unknown cycle samples: {sorted(unknown_samples)}")
    return (
        candidates.join(dimensions, how="cross")
        .with_columns(
            (
                pl.col("candidate_width_bp") * pl.col("exit_width_ratio")
            ).alias("exit_width_bp")
        )
        .sort(POLICY_KEYS)
    )


def _finite(value: object) -> bool:
    return value is not None and math.isfinite(float(value))


def _cycle_gate_reason(group: pl.DataFrame, index: int, sample: str) -> str:
    def false(column: str) -> bool:
        return column in group.columns and group.item(index, column) is not True

    def active(column: str) -> bool:
        if column not in group.columns:
            return False
        value = group.item(index, column)
        return value is not None and bool(value)

    if active("spot_trial_match") or active("fut_trial_match"):
        return "trial_match"
    if false("spot_ref_ok") or false("fut_ref_ok"):
        return "ref_gate"
    if false("spot_formal") or false("fut_formal"):
        return "formal_gate"
    if (
        false("spot_book_ok")
        or false("fut_book_ok")
        or false("fut_exec_book_ok")
    ):
        return "book_gate"
    if (
        sample == "fresh_1000ms"
        and group.item(index, "eligible_base") is True
        and group.item(index, "eligible_1000ms") is not True
    ):
        return "freshness_gate"
    if not _finite(group.item(index, "basis_mid_bp")) or not _finite(
        group.item(index, ANCHOR_COLUMN)
    ):
        return "nonfinite_state"
    return "eligibility_gap"


def _all_gate_reasons(group: pl.DataFrame, index: int, sample: str) -> str:
    """Return every active gate reason for audit, in stable priority order."""
    reasons: list[str] = []
    if any(
        column in group.columns
        and group.item(index, column) is not None
        and bool(group.item(index, column))
        for column in ("spot_trial_match", "fut_trial_match")
    ):
        reasons.append("trial_match")
    if any(
        column in group.columns and group.item(index, column) is not True
        for column in ("spot_ref_ok", "fut_ref_ok")
    ):
        reasons.append("ref_gate")
    if any(
        column in group.columns and group.item(index, column) is not True
        for column in ("spot_formal", "fut_formal")
    ):
        reasons.append("formal_gate")
    if any(
        column in group.columns and group.item(index, column) is not True
        for column in ("spot_book_ok", "fut_book_ok", "fut_exec_book_ok")
    ):
        reasons.append("book_gate")
    if (
        sample == "fresh_1000ms"
        and group.item(index, "eligible_base") is True
        and group.item(index, "eligible_1000ms") is not True
    ):
        reasons.append("freshness_gate")
    if not _finite(group.item(index, "basis_mid_bp")) or not _finite(
        group.item(index, ANCHOR_COLUMN)
    ):
        reasons.append("nonfinite_state")
    return "|".join(reasons or ["eligibility_gap"])


def _outcome_by_horizon(
    status: str,
    time_to_exit_seconds: int | None,
    last_observed_seconds: int,
    horizons_seconds: tuple[int, ...],
) -> dict[str, bool | None]:
    result: dict[str, bool | None] = {}
    for horizon in horizons_seconds:
        if status == "complete":
            assert time_to_exit_seconds is not None
            result[f"complete_within_{horizon}s"] = (
                time_to_exit_seconds <= horizon
            )
        elif last_observed_seconds >= horizon:
            result[f"complete_within_{horizon}s"] = False
        else:
            result[f"complete_within_{horizon}s"] = None
    return result


def _run_policy(
    group: pl.DataFrame,
    spec: dict[str, object],
    horizons_seconds: tuple[int, ...],
) -> list[dict[str, object]]:
    """Run one non-overlapping FLAT/HOLDING state machine for one pair-day."""
    seconds = [int(value) for value in group["seconds_from_open"].to_list()]
    timestamps = group["timestamp"].to_list()
    timestamp_ns = group.select(pl.col("timestamp").dt.epoch("ns"))[
        "timestamp"
    ].to_list()
    basis = group["basis_mid_bp"].to_list()
    anchors = group[ANCHOR_COLUMN].to_list()
    analysis_valid = group["analysis_eligible"].to_list()
    base_valid = group["eligible_base"].to_list()
    fresh_valid = group["eligible_1000ms"].to_list()

    sample = str(spec["sample"])
    anchor_mode = str(spec["anchor_mode"])
    width = float(spec["candidate_width_bp"])
    exit_width = float(spec["exit_width_bp"])
    exit_delay = int(spec["exit_delay_seconds"])
    max_horizon = (
        int(spec["max_horizon_seconds"])
        if int(spec["max_horizon_seconds"]) > 0
        else None
    )

    valid = [
        bool(analysis)
        and bool(base)
        and (sample == "base" or bool(fresh))
        and _finite(basis_value)
        and _finite(anchor_value)
        for analysis, base, fresh, basis_value, anchor_value in zip(
            analysis_valid, base_valid, fresh_valid, basis, anchors, strict=True
        )
    ]
    residual = [
        float(basis_value) - float(anchor_value)
        if is_valid
        else None
        for basis_value, anchor_value, is_valid in zip(
            basis, anchors, valid, strict=True
        )
    ]

    events: list[dict[str, object]] = []
    holding: dict[str, object] | None = None
    armed = False
    previous_valid = False
    previous_residual: float | None = None
    previous_second: int | None = None
    sequence = 0

    def finish(
        status: str,
        end_reason: str,
        end_index: int,
        last_observed_index: int,
        all_end_reasons: str | None = None,
        censor_index: int | None = None,
    ) -> None:
        nonlocal holding
        assert holding is not None
        entry_index = int(holding["entry_index"])
        entry_second = seconds[entry_index]
        last_observed_seconds = max(
            0, seconds[last_observed_index] - entry_second
        )
        completed = status == "complete"
        end_is_observed = valid[end_index]
        end_basis = float(basis[end_index]) if end_is_observed else None
        end_anchor = float(anchors[end_index]) if end_is_observed else None
        end_dynamic_residual = (
            end_basis - end_anchor
            if end_basis is not None and end_anchor is not None
            else None
        )
        time_to_exit = seconds[end_index] - entry_second if completed else None
        exit_anchor = (
            end_anchor
            if anchor_mode == "dynamic"
            else float(holding["entry_anchor_bp"])
        ) if end_basis is not None else None
        end_exit_residual = (
            end_basis - exit_anchor
            if end_basis is not None and exit_anchor is not None
            else None
        )
        lower_boundary_basis = (
            exit_anchor - exit_width if exit_anchor is not None else None
        )
        basis_capture = (
            float(holding["entry_basis_bp"]) - end_basis
            if completed and end_basis is not None
            else None
        )
        anchor_drift = (
            end_anchor - float(holding["entry_anchor_bp"])
            if completed and end_anchor is not None
            else None
        )
        residual_drop = (
            float(holding["entry_residual_bp"]) - end_dynamic_residual
            if completed and end_dynamic_residual is not None
            else None
        )
        nominal_cycle_band = width + exit_width
        event = {
            key: spec[key]
            for key in spec
            if key not in {"entry_index"}
        }
        event.update(
            {
                "cycle_sequence": int(holding["cycle_sequence"]),
                "status": status,
                "end_reason": end_reason,
                "all_end_reasons": all_end_reasons or end_reason,
                "policy_day_stopped_after_event": status != "complete",
                "entry_timestamp": timestamps[entry_index],
                "entry_seconds_from_open": entry_second,
                "entry_basis_bp": holding["entry_basis_bp"],
                "entry_anchor_bp": holding["entry_anchor_bp"],
                "entry_residual_bp": holding["entry_residual_bp"],
                "entry_basis_move_bp": holding["entry_basis_move_bp"],
                "entry_anchor_move_bp": holding["entry_anchor_move_bp"],
                "entry_residual_move_bp": holding["entry_residual_move_bp"],
                "entry_without_basis_rise": (
                    float(holding["entry_basis_move_bp"]) <= VALUE_EPS
                    if holding["entry_basis_move_bp"] is not None
                    else None
                ),
                "entry_overshoot_bp": (
                    float(holding["entry_residual_bp"]) - width
                ),
                "end_timestamp": timestamps[end_index],
                "end_seconds_from_open": seconds[end_index],
                "censor_timestamp": (
                    timestamps[censor_index] if censor_index is not None else None
                ),
                "censor_seconds_from_open": (
                    seconds[censor_index] if censor_index is not None else None
                ),
                "last_observed_seconds": last_observed_seconds,
                "time_to_exit_seconds": time_to_exit,
                "end_basis_bp": end_basis,
                "end_anchor_bp": end_anchor,
                "end_dynamic_residual_bp": end_dynamic_residual,
                "end_exit_residual_bp": end_exit_residual,
                "lower_boundary_basis_bp": lower_boundary_basis,
                "exit_overshoot_bp": (
                    lower_boundary_basis - end_basis
                    if completed
                    and lower_boundary_basis is not None
                    and end_basis is not None
                    else None
                ),
                "basis_capture_bp": basis_capture,
                "anchor_drift_bp": anchor_drift,
                "residual_drop_bp": residual_drop,
                "nominal_cycle_band_bp": nominal_cycle_band,
                "capture_shortfall_bp": (
                    nominal_cycle_band - basis_capture
                    if completed and basis_capture is not None
                    else None
                ),
                "capture_to_nominal_ratio": (
                    basis_capture / nominal_cycle_band
                    if completed
                    and basis_capture is not None
                    and nominal_cycle_band > VALUE_EPS
                    else None
                ),
                "anchor_only_exit": (
                    basis_capture <= VALUE_EPS
                    if completed and basis_capture is not None
                    else None
                ),
                "capture_identity_error_bp": (
                    basis_capture - (residual_drop - anchor_drift)
                    if basis_capture is not None
                    and residual_drop is not None
                    and anchor_drift is not None
                    else None
                ),
                "early_exit_touch": bool(holding["early_exit_touch"]),
                "max_adverse_basis_bp": (
                    float(holding["max_basis_bp"])
                    - float(holding["entry_basis_bp"])
                ),
                "max_adverse_dynamic_residual_bp": (
                    float(holding["max_dynamic_residual_bp"])
                    - float(holding["entry_residual_bp"])
                ),
                **_outcome_by_horizon(
                    status,
                    time_to_exit,
                    last_observed_seconds,
                    horizons_seconds,
                ),
            }
        )
        events.append(event)
        holding = None

    for index in range(len(seconds)):
        is_valid = valid[index]
        timestamp_gap = (
            previous_second is not None
            and (
                seconds[index] - previous_second != 1
                or timestamp_ns[index] - timestamp_ns[index - 1]
                != 1_000_000_000
            )
        )
        if timestamp_gap:
            if holding is not None:
                last_index = max(index - 1, int(holding["entry_index"]))
                finish(
                    "censored",
                    "timestamp_gap",
                    last_index,
                    last_index,
                    "timestamp_gap",
                    censor_index=index,
                )
                break
            armed = False
            previous_valid = False
            previous_residual = None

        if not is_valid:
            if holding is not None:
                last_index = max(index - 1, int(holding["entry_index"]))
                finish(
                    "censored",
                    _cycle_gate_reason(group, index, sample),
                    last_index,
                    last_index,
                    _all_gate_reasons(group, index, sample),
                    censor_index=index,
                )
                break
            armed = False
            previous_valid = False
            previous_residual = None
            previous_second = seconds[index]
            continue

        current_basis = float(basis[index])
        current_anchor = float(anchors[index])
        current_residual = float(residual[index])

        if holding is not None:
            holding["max_basis_bp"] = max(
                float(holding["max_basis_bp"]), current_basis
            )
            holding["max_dynamic_residual_bp"] = max(
                float(holding["max_dynamic_residual_bp"]), current_residual
            )
            elapsed = seconds[index] - int(holding["entry_second"])
            exit_residual = (
                current_residual
                if anchor_mode == "dynamic"
                else current_basis - float(holding["entry_anchor_bp"])
            )
            exit_touched = exit_residual <= -exit_width + VALUE_EPS
            if exit_touched and elapsed < exit_delay:
                holding["early_exit_touch"] = True
            if exit_touched and elapsed >= exit_delay:
                finish("complete", "lower_boundary", index, index)
                armed = current_residual < width - VALUE_EPS
            elif max_horizon is not None and elapsed >= max_horizon:
                finish("timeout", "hard_ttl", index, index)
                break
            previous_valid = True
            previous_residual = current_residual
            previous_second = seconds[index]
            continue

        if not armed:
            armed = current_residual < width - VALUE_EPS
        elif (
            previous_valid
            and previous_residual is not None
            and previous_residual < width - VALUE_EPS
            and current_residual + VALUE_EPS >= width
        ):
            sequence += 1
            holding = {
                "cycle_sequence": sequence,
                "entry_index": index,
                "entry_second": seconds[index],
                "entry_basis_bp": current_basis,
                "entry_anchor_bp": current_anchor,
                "entry_residual_bp": current_residual,
                "entry_basis_move_bp": (
                    current_basis - float(basis[index - 1])
                    if _finite(basis[index - 1])
                    else None
                ),
                "entry_anchor_move_bp": (
                    current_anchor - float(anchors[index - 1])
                    if _finite(anchors[index - 1])
                    else None
                ),
                "entry_residual_move_bp": (
                    current_residual - float(previous_residual)
                ),
                "max_basis_bp": current_basis,
                "max_dynamic_residual_bp": current_residual,
                "early_exit_touch": False,
            }
            armed = False

        previous_valid = True
        previous_residual = current_residual
        previous_second = seconds[index]

    if holding is not None:
        finish(
            "censored",
            "session_cutoff",
            len(seconds) - 1,
            len(seconds) - 1,
        )
    return events


def build_latent_cycles(
    panel: pl.DataFrame,
    universe: pl.DataFrame,
    horizons_seconds: Iterable[int] = DEFAULT_HORIZONS_SECONDS,
) -> pl.DataFrame:
    """Build independent non-overlapping full-cycle events for every policy."""
    horizons = tuple(sorted(set(int(value) for value in horizons_seconds)))
    if not horizons or horizons[0] <= 0:
        raise ValueError("cycle horizons must be positive")
    required = {
        "Date",
        "ValueCode",
        "QuoteCode",
        "timestamp",
        "seconds_from_open",
        "basis_mid_bp",
        ANCHOR_COLUMN,
        "analysis_eligible",
        "eligible_base",
        "eligible_1000ms",
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"cycle panel missing columns: {missing}")
    ordered = panel.filter(
        pl.col("seconds_from_open") >= SESSION_START_SECONDS
    ).sort(["Date", "ValueCode", "timestamp"])
    specs_by_pair: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for key, group in universe.group_by(
        ["Date", "ValueCode", "QuoteCode"], maintain_order=True
    ):
        specs_by_pair[(str(key[0]), str(key[1]), str(key[2]))] = group.to_dicts()

    rows: list[dict[str, object]] = []
    for key, group in ordered.group_by(
        ["Date", "ValueCode", "QuoteCode"], maintain_order=True
    ):
        pair_key = (str(key[0]), str(key[1]), str(key[2]))
        for spec in specs_by_pair.get(pair_key, []):
            rows.extend(_run_policy(group, spec, horizons))
    if not rows:
        raise ValueError("cycle study produced no latent entry events")
    result = pl.from_dicts(rows, infer_schema_length=None)
    timestamp_dtype = ordered.schema["timestamp"]
    timestamp_columns = [
        column
        for column in ("entry_timestamp", "end_timestamp", "censor_timestamp")
        if column in result.columns
    ]
    nullable_float_columns = [
        "entry_basis_move_bp",
        "entry_anchor_move_bp",
        "end_basis_bp",
        "end_anchor_bp",
        "end_dynamic_residual_bp",
        "end_exit_residual_bp",
        "lower_boundary_basis_bp",
        "exit_overshoot_bp",
        "basis_capture_bp",
        "anchor_drift_bp",
        "residual_drop_bp",
        "capture_shortfall_bp",
        "capture_to_nominal_ratio",
        "capture_identity_error_bp",
    ]
    nullable_int_columns = [
        "censor_seconds_from_open",
        "time_to_exit_seconds",
    ]
    nullable_bool_columns = [
        "entry_without_basis_rise",
        "anchor_only_exit",
        *[f"complete_within_{horizon}s" for horizon in horizons],
    ]
    casts: list[pl.Expr] = [
        *[pl.col(column).cast(timestamp_dtype) for column in timestamp_columns],
        *[
            pl.col(column).cast(pl.Float64)
            for column in nullable_float_columns
            if column in result.columns
        ],
        *[
            pl.col(column).cast(pl.Int64)
            for column in nullable_int_columns
            if column in result.columns
        ],
        *[
            pl.col(column).cast(pl.Boolean)
            for column in nullable_bool_columns
            if column in result.columns
        ],
    ]
    return result.with_columns(
        *casts,
        pl.lit(False).alias("execution_safe_snapshot"),
        pl.lit(True).alias("contains_target_day_outcome"),
    ).sort(
        [*POLICY_KEYS, "entry_timestamp", "cycle_sequence"]
    )


def summarize_cycles_by_day_symbol(
    cycles: pl.DataFrame,
    universe: pl.DataFrame,
    horizons_seconds: Iterable[int] = DEFAULT_HORIZONS_SECONDS,
) -> pl.DataFrame:
    """Summarize events while retaining zero-entry pair-policy rows."""
    horizons = tuple(sorted(set(int(value) for value in horizons_seconds)))
    aggregations: list[pl.Expr] = [
        pl.len().alias("latent_entries"),
        (pl.col("status") == "complete").sum().alias("completed_cycles"),
        (pl.col("status") == "timeout").sum().alias("timeout_cycles"),
        (pl.col("status") == "censored").sum().alias("censored_cycles"),
        pl.col("early_exit_touch").sum().alias("early_exit_touches"),
        pl.col("anchor_only_exit").sum().alias("anchor_only_exits"),
        pl.col("entry_without_basis_rise")
        .sum()
        .alias("entries_without_basis_rise"),
        pl.col("time_to_exit_seconds")
        .median()
        .alias("completed_holding_seconds_p50"),
        pl.col("time_to_exit_seconds")
        .quantile(0.80, interpolation="nearest")
        .alias("completed_holding_seconds_p80"),
        pl.col("time_to_exit_seconds")
        .quantile(0.95, interpolation="nearest")
        .alias("completed_holding_seconds_p95"),
        pl.col("basis_capture_bp").median().alias("basis_capture_bp_p50"),
        pl.col("basis_capture_bp")
        .quantile(0.20, interpolation="nearest")
        .alias("basis_capture_bp_p20"),
        pl.col("anchor_drift_bp").median().alias("anchor_drift_bp_p50"),
        pl.col("capture_shortfall_bp")
        .median()
        .alias("capture_shortfall_bp_p50"),
        pl.col("max_adverse_basis_bp")
        .quantile(0.80, interpolation="nearest")
        .alias("max_adverse_basis_bp_p80"),
        pl.col("capture_identity_error_bp")
        .abs()
        .max()
        .alias("capture_identity_max_abs_error_bp"),
    ]
    for horizon in horizons:
        column = f"complete_within_{horizon}s"
        aggregations.extend(
            [
                pl.col(column).count().alias(f"n_observed_{horizon}s"),
                pl.col(column).sum().alias(f"n_complete_within_{horizon}s"),
                pl.col(column).mean().alias(f"p_complete_within_{horizon}s"),
            ]
        )
    event_summary = cycles.group_by(POLICY_KEYS).agg(*aggregations)
    zero_columns = [
        "latent_entries",
        "completed_cycles",
        "timeout_cycles",
        "censored_cycles",
        "early_exit_touches",
        "anchor_only_exits",
        "entries_without_basis_rise",
        *[f"n_observed_{horizon}s" for horizon in horizons],
        *[f"n_complete_within_{horizon}s" for horizon in horizons],
    ]
    return (
        universe.join(
            event_summary,
            on=POLICY_KEYS,
            how="left",
            validate="1:1",
        )
        .with_columns(*[pl.col(column).fill_null(0) for column in zero_columns])
        .with_columns(
            (pl.col("latent_entries") / SESSION_HOURS).alias(
                "latent_entries_per_hour"
            ),
            pl.when(pl.col("latent_entries") > 0)
            .then(pl.col("completed_cycles") / pl.col("latent_entries"))
            .otherwise(None)
            .alias("eventual_completion_lower_bound"),
            pl.when(pl.col("latent_entries") > 0)
            .then(pl.col("censored_cycles") / pl.col("latent_entries"))
            .otherwise(None)
            .alias("censor_rate"),
            pl.when(pl.col("completed_cycles") > 0)
            .then(pl.col("anchor_only_exits") / pl.col("completed_cycles"))
            .otherwise(None)
            .alias("anchor_only_exit_rate"),
            pl.lit(False).alias("execution_safe_snapshot"),
            pl.lit(True).alias("contains_target_day_outcome"),
        )
        .sort(POLICY_KEYS)
    )


def summarize_cycle_policies(
    by_day_symbol: pl.DataFrame,
    horizons_seconds: Iterable[int] = DEFAULT_HORIZONS_SECONDS,
) -> pl.DataFrame:
    """Create pair-balanced policy summaries with explicit support counts."""
    horizons = tuple(sorted(set(int(value) for value in horizons_seconds)))
    group_keys = [
        "sample",
        "anchor_mode",
        "width_family",
        "width_policy",
        "exit_width_ratio",
        "exit_delay_seconds",
        "max_horizon_seconds",
    ]
    aggregations: list[pl.Expr] = [
        pl.len().alias("pair_days"),
        pl.col("Date").n_unique().alias("dates"),
        pl.col("ValueCode").n_unique().alias("symbols"),
        pl.col("candidate_width_bp").median().alias("median_entry_width_bp"),
        pl.col("exit_width_bp").median().alias("median_exit_width_bp"),
        (pl.col("latent_entries") > 0).sum().alias("pair_days_with_entries"),
        pl.col("latent_entries").sum().alias("total_latent_entries"),
        pl.col("latent_entries")
        .median()
        .alias("pair_median_latent_entries_per_day"),
        pl.col("latent_entries")
        .quantile(0.25, interpolation="nearest")
        .alias("pair_q25_latent_entries_per_day"),
        pl.col("latent_entries")
        .quantile(0.75, interpolation="nearest")
        .alias("pair_q75_latent_entries_per_day"),
        pl.col("completed_cycles").sum().alias("total_completed_cycles"),
        pl.col("timeout_cycles").sum().alias("total_timeout_cycles"),
        pl.col("censored_cycles").sum().alias("total_censored_cycles"),
        pl.col("early_exit_touches").sum().alias("total_early_exit_touches"),
        pl.col("anchor_only_exits").sum().alias("total_anchor_only_exits"),
        pl.col("entries_without_basis_rise")
        .sum()
        .alias("total_entries_without_basis_rise"),
        pl.col("completed_holding_seconds_p50")
        .median()
        .alias("pair_median_completed_holding_seconds"),
        pl.col("completed_holding_seconds_p80")
        .median()
        .alias("pair_median_completed_holding_seconds_p80"),
        pl.col("completed_holding_seconds_p95")
        .median()
        .alias("pair_median_completed_holding_seconds_p95"),
        pl.col("basis_capture_bp_p20")
        .median()
        .alias("pair_median_basis_capture_bp_p20"),
        pl.col("basis_capture_bp_p50")
        .median()
        .alias("pair_median_basis_capture_bp"),
        pl.col("anchor_drift_bp_p50")
        .median()
        .alias("pair_median_anchor_drift_bp"),
        pl.col("capture_shortfall_bp_p50")
        .median()
        .alias("pair_median_capture_shortfall_bp"),
        pl.col("max_adverse_basis_bp_p80")
        .median()
        .alias("pair_median_max_adverse_basis_bp_p80"),
    ]
    for horizon in horizons:
        n_column = f"n_observed_{horizon}s"
        p_column = f"p_complete_within_{horizon}s"
        aggregations.extend(
            [
                (pl.col(n_column) > 0)
                .sum()
                .alias(f"pair_days_observed_{horizon}s"),
                pl.col(n_column).sum().alias(f"total_observed_{horizon}s"),
                pl.col(f"n_complete_within_{horizon}s")
                .sum()
                .alias(f"total_complete_within_{horizon}s"),
                pl.col(p_column)
                .median()
                .alias(f"pair_median_p_complete_within_{horizon}s"),
            ]
        )
    result = (
        by_day_symbol.group_by(group_keys)
        .agg(*aggregations)
        .sort(
            [
                "sample",
                "anchor_mode",
                "exit_width_ratio",
                "exit_delay_seconds",
                "median_entry_width_bp",
                "width_policy",
            ]
        )
    )
    rate_columns: list[pl.Expr] = [
        pl.when(pl.col("total_latent_entries") > 0)
        .then(pl.col("total_completed_cycles") / pl.col("total_latent_entries"))
        .otherwise(None)
        .alias("eventual_completion_lower_bound"),
        pl.when(pl.col("total_latent_entries") > 0)
        .then(pl.col("total_censored_cycles") / pl.col("total_latent_entries"))
        .otherwise(None)
        .alias("censor_rate"),
        pl.when(pl.col("total_latent_entries") > 0)
        .then(pl.col("total_early_exit_touches") / pl.col("total_latent_entries"))
        .otherwise(None)
        .alias("early_exit_touch_rate"),
        pl.when(pl.col("total_completed_cycles") > 0)
        .then(
            pl.col("total_anchor_only_exits")
            / pl.col("total_completed_cycles")
        )
        .otherwise(None)
        .alias("anchor_only_exit_rate"),
    ]
    for horizon in horizons:
        rate_columns.append(
            pl.when(pl.col(f"total_observed_{horizon}s") > 0)
            .then(
                pl.col(f"total_complete_within_{horizon}s")
                / pl.col(f"total_observed_{horizon}s")
            )
            .otherwise(None)
            .alias(f"event_weighted_p_complete_within_{horizon}s")
        )
    return result.with_columns(
        *rate_columns,
        pl.lit(False).alias("execution_safe_snapshot"),
        pl.lit(True).alias("contains_target_day_outcome"),
    )


def run_cycle_study(
    panel_path: Path,
    parameter_path: Path,
    output_dir: Path,
) -> CycleStudyResult:
    """Run and persist the first upper-to-lower 1-second cycle pilot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = pl.read_parquet(panel_path)
    candidates = load_cycle_width_candidates(parameter_path)
    universe = build_cycle_policy_universe(candidates)
    cycles = build_latent_cycles(panel, universe)
    by_day_symbol = summarize_cycles_by_day_symbol(cycles, universe)
    policy_summary = summarize_cycle_policies(by_day_symbol)
    diagnostic_frontier = policy_summary.filter(
        (pl.col("sample") == "base")
        & pl.col("width_policy").is_in(
            list(DEFAULT_DIAGNOSTIC_WIDTH_POLICIES)
        )
        & pl.col("exit_width_ratio").is_in([0.0, 1.0])
    ).sort(
        [
            "exit_delay_seconds",
            "anchor_mode",
            "exit_width_ratio",
            "median_entry_width_bp",
            "width_policy",
        ]
    )
    dynamic_symmetric_diagnostic = policy_summary.filter(
        (pl.col("sample") == "base")
        & (pl.col("anchor_mode") == "dynamic")
        & (pl.col("exit_width_ratio") == 1.0)
    ).sort(["exit_delay_seconds", "median_entry_width_bp", "width_policy"])

    cycles.write_parquet(output_dir / "latent_full_cycles.parquet")
    by_day_symbol.write_csv(output_dir / "cycle_by_day_symbol.csv")
    policy_summary.write_csv(output_dir / "cycle_policy_summary.csv")
    diagnostic_frontier.write_csv(
        output_dir / "cycle_diagnostic_frontier.csv"
    )
    dynamic_symmetric_diagnostic.write_csv(
        output_dir / "dynamic_symmetric_diagnostic_summary.csv"
    )
    config = {
        "anchor_column": ANCHOR_COLUMN,
        "diagnostic_width_policies": list(DEFAULT_DIAGNOSTIC_WIDTH_POLICIES),
        "fixed_bp_role": "diagnostic_only; excluded from production optimizer",
        "exit_width_ratios": list(DEFAULT_EXIT_WIDTH_RATIOS),
        "anchor_modes": list(DEFAULT_ANCHOR_MODES),
        "exit_delays_seconds": list(DEFAULT_EXIT_DELAYS_SECONDS),
        "samples": list(DEFAULT_SAMPLES),
        "horizons_seconds": list(DEFAULT_HORIZONS_SECONDS),
        "hard_ttl_seconds": None,
        "primary_diagnostic_path": {
            "entry": "first causal B_mid - M cross from below to +W",
            "exit": "first causal B_mid - dynamic M cross to -W",
            "state_machine": "non-overlapping FLAT -> HOLDING -> FLAT",
        },
        "frozen_sensitivity": "exit compares B_s with entry-time M_t",
        "event_semantics": "latent one-second price path; not maker fill or PnL",
        "actionable_execution": False,
        "ev_ready": False,
        "censor_semantics": "no bridging across eligibility, timestamp, or session gaps",
        "censored_inventory": "unresolved; censor is absorbing for that policy-day",
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return CycleStudyResult(
        cycles=cycles,
        by_day_symbol=by_day_symbol,
        policy_summary=policy_summary,
        diagnostic_frontier=diagnostic_frontier,
        dynamic_symmetric_diagnostic_summary=dynamic_symmetric_diagnostic,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build non-overlapping upper-to-lower latent basis cycles."
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "fair_anchor_panel.parquet",
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=MAKER_ROOT / "data" / "quote_width" / "daily_product_parameters.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MAKER_ROOT / "data" / "quote_width" / "cycle",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_cycle_study(args.panel, args.parameters, args.output_dir)
    print(result.dynamic_symmetric_diagnostic_summary)


if __name__ == "__main__":
    main()
