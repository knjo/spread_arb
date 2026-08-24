"""Same-day latent exit-opportunity labels conditional on a WP02 full fill.

This is a price-path diagnostic, not an executable exit, PnL, or EV model.
For every full-fill *policy alias* it freezes the latest causal one-second
fair state at the raw fill cursor, provided that exact state is eligible, then
evaluates four sell-basis
exit targets after 1 and 30 seconds:

* dynamic center: ``basis_mid_bp <= M_t``;
* frozen center: ``basis_mid_bp <= M_0``;
* dynamic adaptive lower: ``basis_mid_bp <= M_t - L``;
* frozen adaptive lower: ``basis_mid_bp <= M_0 - L``.

``L`` is the target-day D-1-safe lower distance for the alias's adaptive
boundary quantile.  Evaluation occurs only on the exact one-second wall-clock
grid from ``ceil(fill + delay)`` until (but excluding) 13:20 Asia/Taipei.  A
missing or ineligible grid point censors every target not already hit; later
valid rows are never bridged across the gap.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import datetime, time, timezone
import json
import math
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence
from zoneinfo import ZoneInfo

import polars as pl


SELL_BASIS_ENTRY_ROUTES = (
    "future_ask_spot_taker",
    "spot_bid_future_taker",
)
OBSERVATION_DELAYS_SECONDS: tuple[int, ...] = (1, 30)
TARGET_IDS: tuple[str, ...] = (
    "dynamic_center",
    "frozen_center",
    "dynamic_adaptive_lower",
    "frozen_adaptive_lower",
)
STATUS_VALUES = ("no_entry_state", "censor", "session_no_hit", "hit")

MIN_DISPLAY_DATES = 4
MIN_DISPLAY_ENTRY_STATES = 50
MIN_DISPLAY_HITS = 30
MAX_DISPLAY_CENSOR_RATE = 0.10

_NS_PER_SECOND = 1_000_000_000
_TAIPEI = ZoneInfo("Asia/Taipei")

DEFAULT_QUOTE_FILL_DIR = Path(__file__).resolve().parents[2] / "data" / "quote_fill"
DEFAULT_FAIR_PANEL_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "fair_mid" / "fair_anchor_panel.parquet"
)
DEFAULT_ADAPTIVE_SNAPSHOT_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "quote_width"
    / "adaptive"
    / "adaptive_parameter_snapshot_by_day_symbol.csv"
)

LatentStatus = Literal[
    "no_entry_state", "censor", "session_no_hit", "hit"
]


@dataclass(frozen=True)
class PostFillOpportunityResult:
    """Raw policy-alias labels, support-aware summary, and contract audit."""

    labels: pl.DataFrame
    summary: pl.DataFrame
    audit: pl.DataFrame


@dataclass(frozen=True)
class _TargetSpec:
    target_id: str
    anchor_mode: Literal["dynamic", "frozen"]
    exit_target: Literal["center", "adaptive_lower"]

    def threshold(
        self, dynamic_anchor_bp: float, frozen_anchor_bp: float, lower_bp: float
    ) -> float:
        anchor = (
            dynamic_anchor_bp
            if self.anchor_mode == "dynamic"
            else frozen_anchor_bp
        )
        return anchor if self.exit_target == "center" else anchor - lower_bp

    def frozen_threshold(self, frozen_anchor_bp: float, lower_bp: float) -> float:
        return (
            frozen_anchor_bp
            if self.exit_target == "center"
            else frozen_anchor_bp - lower_bp
        )


_TARGET_SPECS = tuple(
    _TargetSpec(
        target_id,
        "dynamic" if target_id.startswith("dynamic") else "frozen",
        "adaptive_lower" if target_id.endswith("adaptive_lower") else "center",
    )
    for target_id in TARGET_IDS
)


@dataclass(frozen=True)
class _EntryState:
    timestamp_ns: int
    basis_bp: float
    anchor_bp: float


@dataclass(frozen=True)
class _AdaptiveLower:
    distance_bp: float
    source_asof_date: str | None
    parameter_version: str | None


@dataclass(frozen=True)
class _TerminalState:
    timestamp_ns: int
    basis_bp: float | None
    anchor_bp: float | None
    threshold_bp: float | None
    observed_eligible_grids: int
    censor_reason: str | None = None


class _FairPath:
    """One exact product-day one-second path with causal eligible index."""

    def __init__(self, frame: pl.DataFrame):
        required = {
            "_fair_timestamp_ns",
            "basis_mid_bp",
            "anchor_ewma_120s_bp",
            "analysis_eligible",
        }
        _require_columns(frame, required, "causal fair path")
        ordered = frame.sort("_fair_timestamp_ns")
        duplicate = (
            ordered.group_by("_fair_timestamp_ns")
            .len()
            .filter(pl.col("len") != 1)
        )
        if duplicate.height:
            raise ValueError("causal fair panel contains duplicate product timestamps")
        self.frame = ordered
        self.timestamps = tuple(
            int(value) for value in ordered["_fair_timestamp_ns"].to_list()
        )
        self.basis = tuple(ordered["basis_mid_bp"].to_list())
        self.anchors = tuple(ordered["anchor_ewma_120s_bp"].to_list())
        analysis = ordered["analysis_eligible"].to_list()
        base = (
            ordered["eligible_base"].to_list()
            if "eligible_base" in ordered.columns
            else [True] * ordered.height
        )
        self.valid = tuple(
            analysis_value is True
            and base_value is True
            and _finite(basis_value)
            and _finite(anchor_value)
            for analysis_value, base_value, basis_value, anchor_value in zip(
                analysis, base, self.basis, self.anchors, strict=True
            )
        )
        self.eligible_indices = tuple(
            index for index, valid in enumerate(self.valid) if valid
        )
        self.eligible_timestamps = tuple(
            self.timestamps[index] for index in self.eligible_indices
        )

    def latest_eligible_at_or_before(self, timestamp_ns: int) -> _EntryState | None:
        """Return the latest causal grid only when that exact state is eligible.

        An ineligible grid is a state transition, not a row that may be skipped.
        Looking backward through it to an older eligible quote would bridge a
        TrialMatch/ref/book gap and create a position from stale state.
        """
        index = bisect_right(self.timestamps, int(timestamp_ns)) - 1
        if index < 0 or not self.valid[index]:
            return None
        return _EntryState(
            self.timestamps[index],
            float(self.basis[index]),
            float(self.anchors[index]),
        )

    def evaluate_targets(
        self,
        *,
        start_grid_ns: int,
        cutoff_ns: int,
        entry: _EntryState,
        lower_bp: float,
    ) -> dict[str, tuple[LatentStatus, _TerminalState]]:
        if start_grid_ns >= cutoff_ns:
            return {
                spec.target_id: (
                    "session_no_hit",
                    _TerminalState(cutoff_ns, None, None, None, 0),
                )
                for spec in _TARGET_SPECS
            }

        unresolved = {spec.target_id: spec for spec in _TARGET_SPECS}
        results: dict[str, tuple[LatentStatus, _TerminalState]] = {}
        observed = 0
        position = bisect_left(self.timestamps, start_grid_ns)
        expected_ns = start_grid_ns
        while expected_ns < cutoff_ns and unresolved:
            if position >= len(self.timestamps) or self.timestamps[position] != expected_ns:
                terminal = _TerminalState(
                    expected_ns,
                    None,
                    None,
                    None,
                    observed,
                    "missing_one_second_grid",
                )
                for target_id in unresolved:
                    results[target_id] = ("censor", terminal)
                unresolved.clear()
                break
            if not self.valid[position]:
                terminal = _TerminalState(
                    expected_ns,
                    _float_or_none(self.basis[position]),
                    _float_or_none(self.anchors[position]),
                    None,
                    observed,
                    _eligibility_gap_reason(self.frame.row(position, named=True)),
                )
                for target_id in unresolved:
                    results[target_id] = ("censor", terminal)
                unresolved.clear()
                break

            basis = float(self.basis[position])
            anchor = float(self.anchors[position])
            observed += 1
            for target_id, spec in tuple(unresolved.items()):
                threshold = spec.threshold(anchor, entry.anchor_bp, lower_bp)
                if basis <= threshold:
                    results[target_id] = (
                        "hit",
                        _TerminalState(
                            expected_ns,
                            basis,
                            anchor,
                            threshold,
                            observed,
                        ),
                    )
                    del unresolved[target_id]
            position += 1
            expected_ns += _NS_PER_SECOND

        for target_id in unresolved:
            results[target_id] = (
                "session_no_hit",
                _TerminalState(cutoff_ns, None, None, None, observed),
            )
        return results


def run_post_fill_opportunity_study(
    order_aliases: pl.DataFrame,
    fair_panel: pl.DataFrame,
    adaptive_lower: pl.DataFrame,
    *,
    observation_delays_seconds: Sequence[int] = OBSERVATION_DELAYS_SECONDS,
    output_dir: Path | None = None,
) -> PostFillOpportunityResult:
    """Build latent same-day opportunities for every full-fill policy alias.

    The fair panel must be causal and contain no forward-label dependency in
    the supplied columns.  ``adaptive_lower`` must be a D-1-safe snapshot;
    target-day outcome rows are rejected when its audit columns are present.
    """

    delays = _validate_delays(observation_delays_seconds)
    aliases = _eligible_aliases(order_aliases)
    fair = _normalise_fair_panel(fair_panel)
    lower_lookup = _adaptive_lower_lookup(adaptive_lower)
    fair_lookup = _fair_path_lookup(fair)

    records: list[dict[str, object]] = []
    for alias in aliases.iter_rows(named=True):
        date = str(alias["Date"])
        value_code = str(alias["ValueCode"])
        quote_code = str(alias["QuoteCode"])
        quantile = int(alias["boundary_quantile"])
        pair_key = (date, value_code, quote_code)
        boundary_key = (*pair_key, quantile)
        if pair_key not in fair_lookup:
            path = None
        else:
            path = fair_lookup[pair_key]
        if boundary_key not in lower_lookup:
            raise ValueError(f"missing valid D-1 adaptive lower for {boundary_key}")
        boundary = lower_lookup[boundary_key]
        alias_asof = alias.get("source_asof_date")
        if (
            alias_asof is not None
            and boundary.source_asof_date is not None
            and str(alias_asof) != boundary.source_asof_date
        ):
            raise ValueError(
                f"adaptive source_asof_date mismatch for policy alias "
                f"{alias['policy_generation_id']}"
            )
        fill_ns = int(alias["full_fill_recv_time_ns"])
        entry = path.latest_eligible_at_or_before(fill_ns) if path else None
        cutoff_ns = session_cutoff_ns(date)
        for delay in delays:
            start_ns = ceil_second_ns(fill_ns + delay * _NS_PER_SECOND)
            if entry is None:
                for spec in _TARGET_SPECS:
                    records.append(
                        _label_record(
                            alias,
                            spec,
                            delay,
                            start_ns,
                            cutoff_ns,
                            boundary,
                            None,
                            "no_entry_state",
                            _TerminalState(
                                fill_ns,
                                None,
                                None,
                                None,
                                0,
                                "no_eligible_fair_state_at_or_before_fill",
                            ),
                        )
                    )
                continue
            outcomes = path.evaluate_targets(
                start_grid_ns=start_ns,
                cutoff_ns=cutoff_ns,
                entry=entry,
                lower_bp=boundary.distance_bp,
            )
            for spec in _TARGET_SPECS:
                status, terminal = outcomes[spec.target_id]
                records.append(
                    _label_record(
                        alias,
                        spec,
                        delay,
                        start_ns,
                        cutoff_ns,
                        boundary,
                        entry,
                        status,
                        terminal,
                    )
                )

    labels = _from_records(records, _label_schema()).sort(
        [
            "Date",
            "ValueCode",
            "route",
            "boundary_quantile",
            "maker_fill_recv_time_ns",
            "policy_generation_id",
            "observation_delay_seconds",
            "target_id",
        ]
    )
    summary = summarize_post_fill_opportunities(labels)
    audit = _audit(order_aliases, aliases, labels, fair, adaptive_lower, delays)
    result = PostFillOpportunityResult(labels, summary, audit)
    if output_dir is not None:
        write_post_fill_opportunity_study(result, Path(output_dir), delays)
    return result


def summarize_post_fill_opportunities(labels: pl.DataFrame) -> pl.DataFrame:
    """Summarize latent hits with full support and censor denominators."""

    if labels.is_empty():
        return pl.DataFrame()
    group = [
        "ValueCode",
        "route",
        "boundary_quantile",
        "target_id",
        "anchor_mode",
        "exit_target",
        "observation_delay_seconds",
    ]
    hit = pl.col("status") == "hit"
    entry = pl.col("status") != "no_entry_state"
    known = pl.col("status").is_in(["hit", "session_no_hit"])
    result = labels.group_by(group).agg(
        pl.col("Date").n_unique().alias("dates"),
        pl.len().alias("full_fill_policy_aliases"),
        entry.sum().alias("entry_state_available"),
        hit.sum().alias("latent_opportunity_hits"),
        (pl.col("status") == "censor").sum().alias("censored_aliases"),
        (pl.col("status") == "session_no_hit").sum().alias("session_no_hit_aliases"),
        (pl.col("status") == "no_entry_state").sum().alias("no_entry_state_aliases"),
        known.sum().alias("known_outcome_aliases"),
        pl.col("time_to_latent_hit_seconds").filter(hit).median().alias("time_to_hit_seconds_p50"),
        pl.col("time_to_latent_hit_seconds").filter(hit).quantile(0.80).alias("time_to_hit_seconds_p80"),
        pl.col("basis_change_from_entry_fair_bp").filter(hit).median().alias("basis_change_from_entry_fair_bp_p50"),
        pl.col("anchor_drift_from_frozen_m0_bp").filter(hit).median().alias("anchor_drift_from_frozen_m0_bp_p50"),
        pl.col("anchor_only_apparent_hit").filter(hit).sum().alias("anchor_only_apparent_hits"),
    ).with_columns(
        (pl.col("latent_opportunity_hits") / pl.col("full_fill_policy_aliases")).alias("p_hit_all_full_fills_lower_bound"),
        pl.when(pl.col("entry_state_available") > 0)
        .then(pl.col("latent_opportunity_hits") / pl.col("entry_state_available"))
        .otherwise(None)
        .alias("p_hit_entry_state_lower_bound"),
        pl.when(pl.col("known_outcome_aliases") > 0)
        .then(pl.col("latent_opportunity_hits") / pl.col("known_outcome_aliases"))
        .otherwise(None)
        .alias("p_hit_known_outcome_conditional"),
        pl.when(pl.col("entry_state_available") > 0)
        .then(pl.col("censored_aliases") / pl.col("entry_state_available"))
        .otherwise(None)
        .alias("censor_rate"),
        pl.when(pl.col("latent_opportunity_hits") > 0)
        .then(pl.col("anchor_only_apparent_hits") / pl.col("latent_opportunity_hits"))
        .otherwise(None)
        .alias("anchor_only_apparent_hit_rate"),
    )
    return result.with_columns(
        (
            (pl.col("dates") >= MIN_DISPLAY_DATES)
            & (pl.col("entry_state_available") >= MIN_DISPLAY_ENTRY_STATES)
            & (pl.col("latent_opportunity_hits") >= MIN_DISPLAY_HITS)
            & (pl.col("censor_rate") <= MAX_DISPLAY_CENSOR_RATE)
        )
        .fill_null(False)
        .alias("display_support_valid"),
        pl.lit("latent_exit_opportunity_only").alias("research_layer"),
        pl.lit(False).alias("actionable_execution"),
        pl.lit(False).alias("pnl_ready"),
        pl.lit(False).alias("ev_ready"),
    ).sort(group)


def write_post_fill_opportunity_study(
    result: PostFillOpportunityResult,
    output_dir: Path,
    observation_delays_seconds: Sequence[int] = OBSERVATION_DELAYS_SECONDS,
) -> None:
    """Write files whose names cannot be mistaken for execution or PnL."""

    output_dir.mkdir(parents=True, exist_ok=True)
    result.labels.write_parquet(
        output_dir / "latent_exit_opportunity_labels.parquet"
    )
    result.summary.write_csv(output_dir / "latent_exit_opportunity_summary.csv")
    result.audit.write_csv(output_dir / "latent_exit_opportunity_audit.csv")
    config = {
        "research_layer": "same_day_latent_exit_opportunity",
        "executable_exit": False,
        "pnl_ready": False,
        "ev_ready": False,
        "fair_state": "causal_one_second_analysis_eligible",
        "frozen_anchor": "latest_causal_fair_state_if_exact_state_eligible_at_raw_full_fill",
        "evaluation_start": "ceil_to_one_second_grid(fill_plus_delay)",
        "observation_delays_seconds": list(observation_delays_seconds),
        "targets": list(TARGET_IDS),
        "eligibility_gap": "first_gap_censors_without_bridging",
        "session_cutoff": "13:20:00_Asia/Taipei_exclusive",
        "display_support": {
            "min_dates": MIN_DISPLAY_DATES,
            "min_entry_states": MIN_DISPLAY_ENTRY_STATES,
            "min_hits": MIN_DISPLAY_HITS,
            "max_censor_rate": MAX_DISPLAY_CENSOR_RATE,
        },
    }
    (output_dir / "latent_exit_opportunity_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def ceil_second_ns(timestamp_ns: int) -> int:
    """Smallest exact wall-second nanosecond timestamp >= input."""

    if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int) or timestamp_ns < 0:
        raise ValueError("timestamp_ns must be a non-negative integer")
    return ((timestamp_ns + _NS_PER_SECOND - 1) // _NS_PER_SECOND) * _NS_PER_SECOND


def session_cutoff_ns(date: str) -> int:
    """13:20 Asia/Taipei expressed on the UTC-naive raw receive clock."""

    try:
        day = datetime.strptime(str(date), "%Y%m%d").date()
    except ValueError as error:
        raise ValueError("Date must be valid YYYYMMDD") from error
    local = datetime.combine(day, time(13, 20), tzinfo=_TAIPEI)
    utc_naive = local.astimezone(timezone.utc).replace(tzinfo=None)
    return _datetime_ns(utc_naive)


def _label_record(
    alias: dict[str, object],
    spec: _TargetSpec,
    delay: int,
    start_ns: int,
    cutoff_ns: int,
    boundary: _AdaptiveLower,
    entry: _EntryState | None,
    status: LatentStatus,
    terminal: _TerminalState,
) -> dict[str, object]:
    if status not in STATUS_VALUES:
        raise ValueError(f"unknown latent status: {status}")
    fill_ns = int(alias["full_fill_recv_time_ns"])
    lower_bp = boundary.distance_bp
    hit = status == "hit"
    anchor_only = False
    if hit and spec.anchor_mode == "dynamic":
        assert entry is not None
        assert terminal.basis_bp is not None
        anchor_only = terminal.basis_bp > spec.frozen_threshold(
            entry.anchor_bp, lower_bp
        )
    return {
        "Date": str(alias["Date"]),
        "ValueCode": str(alias["ValueCode"]),
        "QuoteCode": str(alias["QuoteCode"]),
        "route": str(alias["route"]),
        "boundary_quantile": int(alias["boundary_quantile"]),
        "raw_order_fact_id": str(alias["raw_order_fact_id"]),
        "policy_generation_id": str(alias["policy_generation_id"]),
        "latent_label_id": (
            f"{alias['policy_generation_id']}|{spec.target_id}|delay{delay}s"
        ),
        "target_id": spec.target_id,
        "anchor_mode": spec.anchor_mode,
        "exit_target": spec.exit_target,
        "observation_delay_seconds": delay,
        "status": status,
        "maker_fill_recv_time_ns": fill_ns,
        "maker_fill_event_sequence": _optional_int(alias.get("full_fill_event_sequence")),
        "maker_fill_row_index": _optional_int(alias.get("full_fill_row_index")),
        "evaluation_start_grid_ns": start_ns,
        "session_cutoff_ns": cutoff_ns,
        "entry_fair_timestamp_ns": entry.timestamp_ns if entry else None,
        "entry_fair_age_ms": (fill_ns - entry.timestamp_ns) / 1_000_000.0 if entry else None,
        "entry_fair_basis_bp": entry.basis_bp if entry else None,
        "frozen_anchor_m0_bp": entry.anchor_bp if entry else None,
        "lower_distance_bp": lower_bp,
        "adaptive_source_asof_date": boundary.source_asof_date,
        "adaptive_parameter_version": boundary.parameter_version,
        "terminal_timestamp_ns": terminal.timestamp_ns,
        "terminal_basis_bp": terminal.basis_bp,
        "terminal_dynamic_anchor_bp": terminal.anchor_bp,
        "terminal_target_threshold_bp": terminal.threshold_bp,
        "observed_eligible_grids": terminal.observed_eligible_grids,
        "censor_reason": terminal.censor_reason,
        "latent_exit_opportunity_hit": hit,
        "time_to_latent_hit_seconds": (
            (terminal.timestamp_ns - fill_ns) / _NS_PER_SECOND if hit else None
        ),
        "basis_change_from_entry_fair_bp": (
            terminal.basis_bp - entry.basis_bp
            if hit and entry is not None and terminal.basis_bp is not None
            else None
        ),
        "anchor_drift_from_frozen_m0_bp": (
            terminal.anchor_bp - entry.anchor_bp
            if hit and entry is not None and terminal.anchor_bp is not None
            else None
        ),
        "anchor_only_apparent_hit": anchor_only,
        "research_layer": "latent_exit_opportunity_only",
        "executable_exit": False,
        "pnl_ready": False,
        "ev_ready": False,
    }


def _eligible_aliases(order_aliases: pl.DataFrame) -> pl.DataFrame:
    required = {
        "Date",
        "ValueCode",
        "QuoteCode",
        "route",
        "boundary_quantile",
        "raw_order_fact_id",
        "policy_generation_id",
        "full_fill",
        "full_fill_recv_time_ns",
    }
    _require_columns(order_aliases, required, "order aliases")
    eligible = order_aliases.filter(
        pl.col("route").is_in(SELL_BASIS_ENTRY_ROUTES)
        & (pl.col("full_fill") == True)  # noqa: E712
        & pl.col("full_fill_recv_time_ns").is_not_null()
    ).with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
    )
    identity = ["policy_generation_id", "boundary_quantile"]
    if eligible.select(identity).unique().height != eligible.height:
        raise ValueError("full-fill policy aliases contain duplicate identities")
    return eligible.sort(
        ["Date", "ValueCode", "route", "boundary_quantile", "full_fill_recv_time_ns"]
    )


def _normalise_fair_panel(fair_panel: pl.DataFrame) -> pl.DataFrame:
    timestamp_column = (
        "fair_timestamp"
        if "fair_timestamp" in fair_panel.columns
        else "timestamp"
        if "timestamp" in fair_panel.columns
        else None
    )
    if timestamp_column is None:
        raise ValueError("causal fair panel missing fair_timestamp/timestamp")
    required = {
        "Date",
        "ValueCode",
        "QuoteCode",
        "basis_mid_bp",
        "anchor_ewma_120s_bp",
        "analysis_eligible",
    }
    _require_columns(fair_panel, required, "causal fair panel")
    dtype = fair_panel.schema[timestamp_column]
    if not isinstance(dtype, pl.Datetime):
        raise ValueError("fair timestamp must be a Datetime column")
    timestamp = pl.col(timestamp_column)
    if dtype.time_zone is not None:
        timestamp = timestamp.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
    return fair_panel.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        timestamp.cast(pl.Datetime("ns")).dt.epoch("ns").alias("_fair_timestamp_ns"),
    ).sort(["Date", "ValueCode", "QuoteCode", "_fair_timestamp_ns"])


def _fair_path_lookup(
    fair: pl.DataFrame,
) -> dict[tuple[str, str, str], _FairPath]:
    result: dict[tuple[str, str, str], _FairPath] = {}
    for key, frame in fair.partition_by(
        ["Date", "ValueCode", "QuoteCode"], as_dict=True, maintain_order=True
    ).items():
        values = key if isinstance(key, tuple) else (key,)
        if len(values) != 3:
            raise AssertionError("unexpected fair path key")
        result[tuple(str(value) for value in values)] = _FairPath(frame)
    return result


def _adaptive_lower_lookup(
    adaptive: pl.DataFrame,
) -> dict[tuple[str, str, str, int], _AdaptiveLower]:
    required = {
        "Date",
        "ValueCode",
        "QuoteCode",
        "boundary_quantile",
        "lower_distance_bp",
        "adaptive_parameter_valid",
    }
    _require_columns(adaptive, required, "adaptive lower snapshot")
    frame = adaptive.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
    )
    if "contains_target_day_outcome" in frame.columns and frame.filter(
        pl.col("contains_target_day_outcome").fill_null(True)
    ).height:
        raise ValueError("adaptive lower snapshot contains target-day outcomes")
    asof_column = (
        "source_asof_date"
        if "source_asof_date" in frame.columns
        else "prior_date"
        if "prior_date" in frame.columns
        else None
    )
    if asof_column is not None:
        invalid = frame.filter(
            pl.col(asof_column).is_null()
            | (pl.col(asof_column).cast(pl.String) >= pl.col("Date"))
        )
        if invalid.height:
            raise ValueError("adaptive lower snapshot is not strictly D-1 safe")
    key_columns = ["Date", "ValueCode", "QuoteCode", "boundary_quantile"]
    if frame.select(key_columns).unique().height != frame.height:
        raise ValueError("adaptive lower snapshot contains duplicate keys")
    lookup: dict[tuple[str, str, str, int], _AdaptiveLower] = {}
    for row in frame.iter_rows(named=True):
        if row["adaptive_parameter_valid"] is not True:
            continue
        lower = row["lower_distance_bp"]
        if not _finite(lower) or float(lower) <= 0:
            continue
        lookup[
            (
                str(row["Date"]),
                str(row["ValueCode"]),
                str(row["QuoteCode"]),
                int(row["boundary_quantile"]),
            )
        ] = _AdaptiveLower(
            float(lower),
            str(row[asof_column]) if asof_column is not None else None,
            (
                str(row["parameter_version"])
                if row.get("parameter_version") is not None
                else None
            ),
        )
    return lookup


def _eligibility_gap_reason(row: dict[str, object]) -> str:
    reasons: list[str] = []
    if any(bool(row.get(column)) for column in ("spot_trial_match", "fut_trial_match")):
        reasons.append("trial_match")
    if any(column in row and row.get(column) is not True for column in ("spot_ref_ok", "fut_ref_ok")):
        reasons.append("ref_gate")
    if any(column in row and row.get(column) is not True for column in ("spot_formal", "fut_formal")):
        reasons.append("formal_gate")
    if any(
        column in row and row.get(column) is not True
        for column in ("spot_book_ok", "fut_book_ok", "fut_exec_book_ok")
    ):
        reasons.append("book_gate")
    if not _finite(row.get("basis_mid_bp")) or not _finite(
        row.get("anchor_ewma_120s_bp")
    ):
        reasons.append("nonfinite_state")
    if row.get("analysis_eligible") is not True:
        reasons.append("analysis_ineligible")
    if "eligible_base" in row and row.get("eligible_base") is not True:
        reasons.append("base_ineligible")
    return "|".join(dict.fromkeys(reasons or ["eligibility_gap"]))


def _audit(
    all_aliases: pl.DataFrame,
    full_aliases: pl.DataFrame,
    labels: pl.DataFrame,
    fair: pl.DataFrame,
    adaptive: pl.DataFrame,
    delays: tuple[int, ...],
) -> pl.DataFrame:
    status_counts = (
        labels.group_by("status").len()
        if not labels.is_empty()
        else pl.DataFrame({"status": [], "len": []})
    )
    counts = {
        str(row["status"]): int(row["len"])
        for row in status_counts.iter_rows(named=True)
    }
    return pl.from_dicts(
        [
            {
                "input_order_aliases": all_aliases.height,
                "full_fill_policy_aliases": full_aliases.height,
                "expected_labels_per_alias": len(delays) * len(_TARGET_SPECS),
                "label_rows": labels.height,
                "fair_panel_rows": fair.height,
                "adaptive_snapshot_rows": adaptive.height,
                **{f"status_{status}": counts.get(status, 0) for status in STATUS_VALUES},
                "policy_alias_level": True,
                "raw_fact_deduplicated": False,
                "latent_exit_opportunity_only": True,
                "executable_exit": False,
                "pnl_ready": False,
                "ev_ready": False,
            }
        ],
        infer_schema_length=None,
    )


def _validate_delays(values: Sequence[int]) -> tuple[int, ...]:
    delays = tuple(int(value) for value in values)
    if not delays or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in values
    ):
        raise ValueError("observation delays must be positive integer seconds")
    if len(delays) != len(set(delays)) or tuple(sorted(delays)) != delays:
        raise ValueError("observation delays must be strictly increasing and unique")
    return delays


def _datetime_ns(value: datetime) -> int:
    epoch = datetime(1970, 1, 1)
    delta = value - epoch
    return (
        (delta.days * 86_400 + delta.seconds) * _NS_PER_SECOND
        + delta.microseconds * 1_000
    )


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _float_or_none(value: object) -> float | None:
    return float(value) if _finite(value) else None


def _optional_int(value: object) -> int | None:
    return int(value) if value is not None else None


def _label_schema() -> Mapping[str, pl.DataType]:
    return {
        "Date": pl.String,
        "ValueCode": pl.String,
        "QuoteCode": pl.String,
        "route": pl.String,
        "boundary_quantile": pl.Int64,
        "raw_order_fact_id": pl.String,
        "policy_generation_id": pl.String,
        "latent_label_id": pl.String,
        "target_id": pl.String,
        "anchor_mode": pl.String,
        "exit_target": pl.String,
        "observation_delay_seconds": pl.Int64,
        "status": pl.String,
        "maker_fill_recv_time_ns": pl.Int64,
        "latent_exit_opportunity_hit": pl.Boolean,
    }


def _from_records(
    records: list[dict[str, object]], schema: Mapping[str, pl.DataType]
) -> pl.DataFrame:
    return (
        pl.from_dicts(records, infer_schema_length=None)
        if records
        else pl.DataFrame(schema=schema)
    )


def _require_columns(
    frame: pl.DataFrame, required: Iterable[str], source: str
) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build raw-full-fill-conditional latent exit labels"
    )
    parser.add_argument(
        "--order-aliases",
        type=Path,
        default=DEFAULT_QUOTE_FILL_DIR / "order_aliases.parquet",
    )
    parser.add_argument("--fair-panel", type=Path, default=DEFAULT_FAIR_PANEL_PATH)
    parser.add_argument(
        "--adaptive-snapshot", type=Path, default=DEFAULT_ADAPTIVE_SNAPSHOT_PATH
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_QUOTE_FILL_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adaptive = pl.read_csv(
        args.adaptive_snapshot,
        schema_overrides={
            "Date": pl.String,
            "ValueCode": pl.String,
            "QuoteCode": pl.String,
            "source_asof_date": pl.String,
        },
    )
    result = run_post_fill_opportunity_study(
        pl.read_parquet(args.order_aliases),
        pl.read_parquet(args.fair_panel),
        adaptive,
        output_dir=args.output_dir,
    )
    print(result.summary)


if __name__ == "__main__":
    main()
