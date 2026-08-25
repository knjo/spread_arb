"""Conditional post-entry convergence facts for the S1 foundation rebuild.

The historical lower boundary is a marginal negative-excursion statistic.  It
does not answer what happens after a positive entry boundary is touched.  This
module builds that missing conditional label on one causal one-second session:
first positive touch, first return to center, and the immediately following
negative floor before the residual returns to center again or becomes
censored.

The functions are pure and deliberately do not choose a candidate, read a
file, or publish an artifact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from typing import Final

import polars as pl

GROUP_KEYS: Final = ("Date", "ValueCode", "QuoteCode")
BOUNDARY_KEYS: Final = (
    "Date",
    "ValueCode",
    "QuoteCode",
    "anchor_model_id",
    "candidate_id",
    "boundary_quantile",
    "tod_bucket",
    "side",
)
ANALYSIS_START_SECOND: Final = 300
ENTRY_STOP_SECOND: Final = 14_400
TRACKING_END_SECOND: Final = 15_600
CHANGE_EPS_BP: Final = 1e-9

TOD_BUCKETS: Final = (
    ("0905_1000", 300, 3_600),
    ("1000_1100", 3_600, 7_200),
    ("1100_1200", 7_200, 10_800),
    ("1200_1300", 10_800, 14_400),
)


@dataclass(frozen=True)
class _PathTerminal:
    center_hit_second: int | None
    negative_cycle_completed: bool
    path_end_second: int
    observed_floor_bp: float
    floor_frontier_distance_bp: tuple[float, ...]
    floor_frontier_hit_second: tuple[int, ...]
    right_censored: bool
    censor_reason: str | None


def _require(frame: pl.DataFrame, columns: set[str], source: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def _tod_bucket(second: int) -> str | None:
    for bucket, start, end in TOD_BUCKETS:
        if start <= second < end:
            return bucket
    return None


def _parse_date(value: object, source: str) -> datetime:
    try:
        return datetime.strptime(str(value), "%Y%m%d")  # noqa: DTZ007
    except ValueError as error:
        raise ValueError(f"{source} must be YYYYMMDD: {value!r}") from error


def _normalise_day(
    day: pl.DataFrame,
    *,
    anchor_column: str,
) -> pl.DataFrame:
    required = {
        *GROUP_KEYS,
        "timestamp",
        "seconds_from_open",
        "basis_mid_bp",
        "analysis_eligible",
        anchor_column,
    }
    _require(day, required, "causal day")
    if day.is_empty():
        raise ValueError("causal day must not be empty")
    result = (
        day.select(*required)
        .with_columns(
            pl.col("Date").cast(pl.String),
            pl.col("ValueCode").cast(pl.String),
            pl.col("QuoteCode").cast(pl.String),
            pl.col("timestamp").cast(pl.Datetime("ns")),
            pl.col("seconds_from_open").cast(pl.Int32),
            pl.col("basis_mid_bp").cast(pl.Float64),
            pl.col(anchor_column).cast(pl.Float64),
            pl.col("analysis_eligible").fill_null(False).cast(pl.Boolean),
        )
        .sort([*GROUP_KEYS, "seconds_from_open", "timestamp"])
        .with_columns(
            (pl.col("basis_mid_bp") - pl.col(anchor_column)).alias("_residual_bp")
        )
    )
    dates = result["Date"].unique().to_list()
    if len(dates) != 1:
        raise ValueError("causal day must contain exactly one Date")
    _parse_date(dates[0], "causal day Date")
    duplicate = (
        result.group_by([*GROUP_KEYS, "seconds_from_open"])
        .len()
        .filter(pl.col("len") != 1)
    )
    if duplicate.height:
        raise ValueError("causal day has duplicate product-second rows")
    gaps = result.with_columns(
        pl.col("seconds_from_open").diff().over(GROUP_KEYS).alias("_second_diff")
    ).filter(pl.col("_second_diff").is_not_null() & (pl.col("_second_diff") != 1))
    if gaps.height:
        raise ValueError("causal day is not a complete within-product 1 Hz grid")
    grid = result.group_by(GROUP_KEYS).agg(
        pl.len().alias("_rows"),
        pl.col("seconds_from_open").min().alias("_first"),
        pl.col("seconds_from_open").max().alias("_last"),
    )
    invalid_grid = grid.filter(
        (pl.col("_rows") != TRACKING_END_SECOND)
        | (pl.col("_first") != 0)
        | (pl.col("_last") != TRACKING_END_SECOND - 1)
    )
    if invalid_grid.height:
        raise ValueError("causal day requires the complete 0..15599 grid")
    return result


def _normalise_boundaries(
    boundaries: pl.DataFrame,
    *,
    target_date: str,
    anchor_model_id: str,
    value_column: str,
) -> pl.DataFrame:
    required = {*BOUNDARY_KEYS, value_column, "source_asof_date"}
    _require(boundaries, required, "boundary predictions")
    result = boundaries.select(*required).with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("anchor_model_id").cast(pl.String),
        pl.col("candidate_id").cast(pl.String),
        pl.col("boundary_quantile").cast(pl.Int16),
        pl.col("tod_bucket").cast(pl.String),
        pl.col("side").cast(pl.String),
        pl.col(value_column).cast(pl.Float64).alias("_boundary_distance_bp"),
        pl.col("source_asof_date").cast(pl.String),
    )
    if result.is_empty():
        raise ValueError("boundary predictions must not be empty")
    if result["Date"].unique().to_list() != [target_date]:
        raise ValueError("boundary Date does not match causal day")
    if result["anchor_model_id"].unique().to_list() != [anchor_model_id]:
        raise ValueError("boundary anchor_model_id does not match selected anchor")
    invalid = result.filter(
        ~pl.col("side").is_in(["positive", "negative"])
        | ~pl.col("tod_bucket").is_in([value[0] for value in TOD_BUCKETS])
        | ~pl.col("boundary_quantile").is_in([50, 80, 95])
        | pl.col("_boundary_distance_bp").is_null()
        | ~pl.col("_boundary_distance_bp").is_finite()
        | (pl.col("_boundary_distance_bp") <= 0)
    )
    if invalid.height:
        raise ValueError("boundary predictions contain invalid rows")
    target = _parse_date(target_date, "target Date")
    unsafe = [
        value
        for value in result["source_asof_date"].unique().to_list()
        if _parse_date(value, "source_asof_date") >= target
    ]
    if unsafe:
        raise ValueError("boundary source_asof_date must be strictly before Date")
    duplicate = result.group_by(BOUNDARY_KEYS).len().filter(pl.col("len") != 1)
    if duplicate.height:
        raise ValueError("boundary predictions contain duplicate keys")
    monotone = (
        result.sort("boundary_quantile")
        .group_by(
            [
                "Date",
                "ValueCode",
                "QuoteCode",
                "candidate_id",
                "tod_bucket",
                "side",
            ]
        )
        .agg(
            pl.col("boundary_quantile").alias("_q"),
            pl.col("_boundary_distance_bp").alias("_distance"),
        )
    )
    for row in monotone.iter_rows(named=True):
        if row["_q"] != [50, 80, 95]:
            raise ValueError("each convergence boundary group needs q50/q80/q95")
        values = [float(value) for value in row["_distance"]]
        if any(later + CHANGE_EPS_BP < earlier for earlier, later in pairwise(values)):
            raise ValueError("boundary distances must be monotone by q")
    return result.sort(BOUNDARY_KEYS)


def _trace_post_touch(
    rows: list[dict[str, object]],
    touch_index: int,
) -> _PathTerminal:
    center_hit_second: int | None = None
    observed_floor = 0.0
    floor_frontier_distance: list[float] = []
    floor_frontier_second: list[int] = []
    entered_negative = False
    previous_second: int | None = None
    path_end = int(rows[touch_index]["seconds_from_open"])
    for index in range(touch_index, len(rows)):
        row = rows[index]
        second = int(row["seconds_from_open"])
        path_end = second
        if second >= TRACKING_END_SECOND:
            return _PathTerminal(
                center_hit_second,
                False,
                path_end,
                observed_floor,
                tuple(floor_frontier_distance),
                tuple(floor_frontier_second),
                True,
                "session_cutoff",
            )
        adjacent = previous_second is None or second == previous_second + 1
        residual = row["_residual_bp"]
        valid = bool(row["analysis_eligible"]) and residual is not None
        valid = valid and math.isfinite(float(residual))
        if not adjacent or not valid:
            return _PathTerminal(
                center_hit_second,
                False,
                path_end,
                observed_floor,
                tuple(floor_frontier_distance),
                tuple(floor_frontier_second),
                True,
                "eligibility_gap",
            )
        value = float(residual)
        if center_hit_second is None and value <= CHANGE_EPS_BP:
            center_hit_second = second
        if center_hit_second is not None:
            if value < -CHANGE_EPS_BP:
                entered_negative = True
                distance = -value
                if distance > observed_floor + CHANGE_EPS_BP:
                    observed_floor = distance
                    floor_frontier_distance.append(distance)
                    floor_frontier_second.append(second)
            elif entered_negative and value >= -CHANGE_EPS_BP:
                return _PathTerminal(
                    center_hit_second,
                    True,
                    second,
                    observed_floor,
                    tuple(floor_frontier_distance),
                    tuple(floor_frontier_second),
                    False,
                    None,
                )
            elif not entered_negative and value > CHANGE_EPS_BP:
                # The residual touched center without a negative overshoot and
                # immediately departed positive again.  The conditional floor
                # is exactly zero for this completed cycle.
                return _PathTerminal(
                    center_hit_second,
                    True,
                    second,
                    0.0,
                    tuple(floor_frontier_distance),
                    tuple(floor_frontier_second),
                    False,
                    None,
                )
        previous_second = second
    return _PathTerminal(
        center_hit_second,
        False,
        path_end,
        observed_floor,
        tuple(floor_frontier_distance),
        tuple(floor_frontier_second),
        True,
        "session_cutoff",
    )


def build_post_touch_convergence_facts(
    day: pl.DataFrame,
    boundary_predictions_day: pl.DataFrame,
    *,
    anchor_column: str,
    anchor_model_id: str,
    boundary_value_column: str = "boundary_distance_bp",
) -> pl.DataFrame:
    """Build one conditional path fact per positive episode × candidate × q.

    A row exists only if the positive boundary was touched before 13:00.  The
    episode-start TOD bucket is frozen for both upper and marginal-lower
    predictions.  Tracking continues through the first center return and its
    immediately following negative excursion.
    """

    panel = _normalise_day(day, anchor_column=anchor_column)
    target_date = str(panel.item(0, "Date"))
    boundaries = _normalise_boundaries(
        boundary_predictions_day,
        target_date=target_date,
        anchor_model_id=str(anchor_model_id),
        value_column=boundary_value_column,
    )
    lookup: dict[tuple[str, str, str, str, int, str], dict[str, object]] = {}
    for row in boundaries.iter_rows(named=True):
        key = (
            str(row["ValueCode"]),
            str(row["QuoteCode"]),
            str(row["candidate_id"]),
            str(row["tod_bucket"]),
            int(row["boundary_quantile"]),
            str(row["side"]),
        )
        lookup[key] = row

    records: list[dict[str, object]] = []
    for key, group in panel.group_by(GROUP_KEYS, maintain_order=True):
        date, value_code, quote_code = map(str, key)
        rows = group.sort("seconds_from_open").iter_rows(named=True)
        path = list(rows)
        episode_sequence = 0
        previous_valid = False
        previous_residual: float | None = None
        for index, row in enumerate(path):
            second = int(row["seconds_from_open"])
            residual_raw = row["_residual_bp"]
            valid = (
                bool(row["analysis_eligible"])
                and residual_raw is not None
                and math.isfinite(float(residual_raw))
            )
            if not valid:
                previous_valid = False
                previous_residual = None
                continue
            residual = float(residual_raw)
            starts_positive = (
                residual > CHANGE_EPS_BP
                and previous_valid
                and previous_residual is not None
                and previous_residual <= CHANGE_EPS_BP
            )
            if starts_positive:
                episode_sequence += 1
                bucket = _tod_bucket(second)
                if bucket is not None and second < ENTRY_STOP_SECOND:
                    candidate_ids = sorted(
                        {
                            candidate
                            for (
                                product,
                                contract,
                                candidate,
                                lookup_bucket,
                                _quantile,
                                _side,
                            ) in lookup
                            if product == value_code
                            and contract == quote_code
                            and lookup_bucket == bucket
                        }
                    )
                    for candidate_id in candidate_ids:
                        for quantile in (50, 80, 95):
                            upper_row = lookup.get(
                                (
                                    value_code,
                                    quote_code,
                                    candidate_id,
                                    bucket,
                                    quantile,
                                    "positive",
                                )
                            )
                            lower_row = lookup.get(
                                (
                                    value_code,
                                    quote_code,
                                    candidate_id,
                                    bucket,
                                    quantile,
                                    "negative",
                                )
                            )
                            if upper_row is None or lower_row is None:
                                raise ValueError(
                                    "positive/negative boundary pair is incomplete"
                                )
                            if (
                                upper_row["source_asof_date"]
                                != lower_row["source_asof_date"]
                            ):
                                raise ValueError(
                                    "upper/lower boundary lineage must match"
                                )
                            upper = float(upper_row["_boundary_distance_bp"])
                            touch_index = None
                            for probe in range(index, len(path)):
                                probe_row = path[probe]
                                probe_second = int(probe_row["seconds_from_open"])
                                if probe_second >= ENTRY_STOP_SECOND:
                                    break
                                probe_residual = probe_row["_residual_bp"]
                                probe_valid = (
                                    bool(probe_row["analysis_eligible"])
                                    and probe_residual is not None
                                    and math.isfinite(float(probe_residual))
                                )
                                if not probe_valid:
                                    break
                                if float(probe_residual) <= CHANGE_EPS_BP:
                                    break
                                if float(probe_residual) + CHANGE_EPS_BP >= upper:
                                    touch_index = probe
                                    break
                            if touch_index is None:
                                continue
                            terminal = _trace_post_touch(path, touch_index)
                            touch_second = int(path[touch_index]["seconds_from_open"])
                            touch_residual = float(path[touch_index]["_residual_bp"])
                            lower = float(lower_row["_boundary_distance_bp"])
                            lower_hit_second = (
                                terminal.center_hit_second
                                if lower <= CHANGE_EPS_BP
                                else next(
                                    (
                                        hit_second
                                        for distance, hit_second in zip(
                                            terminal.floor_frontier_distance_bp,
                                            terminal.floor_frontier_hit_second,
                                            strict=True,
                                        )
                                        if distance + CHANGE_EPS_BP >= lower
                                    ),
                                    None,
                                )
                            )
                            lower_hit = lower_hit_second is not None
                            records.append(
                                {
                                    "Date": date,
                                    "ValueCode": value_code,
                                    "QuoteCode": quote_code,
                                    "anchor_model_id": str(anchor_model_id),
                                    "candidate_id": candidate_id,
                                    "boundary_quantile": quantile,
                                    "episode_sequence": episode_sequence,
                                    "episode_start_second": second,
                                    "tod_bucket": bucket,
                                    "upper_distance_bp": upper,
                                    "independent_lower_distance_bp": lower,
                                    "source_asof_date": str(
                                        upper_row["source_asof_date"]
                                    ),
                                    "touch_second": touch_second,
                                    "touch_timestamp": path[touch_index]["timestamp"],
                                    "touch_residual_bp": touch_residual,
                                    "center_hit": terminal.center_hit_second
                                    is not None,
                                    "center_hit_second": terminal.center_hit_second,
                                    "time_to_center_seconds": (
                                        None
                                        if terminal.center_hit_second is None
                                        else terminal.center_hit_second - touch_second
                                    ),
                                    "observed_post_touch_floor_bp": (
                                        terminal.observed_floor_bp
                                    ),
                                    "floor_frontier_distance_bp": list(
                                        terminal.floor_frontier_distance_bp
                                    ),
                                    "floor_frontier_hit_second": list(
                                        terminal.floor_frontier_hit_second
                                    ),
                                    "negative_cycle_completed": (
                                        terminal.negative_cycle_completed
                                    ),
                                    "right_censored": terminal.right_censored,
                                    "right_censor_reason": terminal.censor_reason,
                                    "path_end_second": terminal.path_end_second,
                                    "independent_lower_confirmed_hit": lower_hit,
                                    "independent_lower_hit_second": lower_hit_second,
                                    "independent_lower_time_from_touch_seconds": (
                                        None
                                        if lower_hit_second is None
                                        else lower_hit_second - touch_second
                                    ),
                                    "independent_lower_time_from_center_seconds": (
                                        None
                                        if lower_hit_second is None
                                        or terminal.center_hit_second is None
                                        else (
                                            lower_hit_second
                                            - terminal.center_hit_second
                                        )
                                    ),
                                    "independent_lower_known_miss": (
                                        terminal.negative_cycle_completed
                                        and not lower_hit
                                    ),
                                    "independent_lower_unknown": (
                                        terminal.right_censored and not lower_hit
                                    ),
                                }
                            )
            previous_valid = True
            previous_residual = residual

    if not records:
        return pl.DataFrame(
            schema={
                "Date": pl.String,
                "ValueCode": pl.String,
                "QuoteCode": pl.String,
                "anchor_model_id": pl.String,
                "candidate_id": pl.String,
                "boundary_quantile": pl.Int64,
                "episode_sequence": pl.Int64,
                "episode_start_second": pl.Int64,
                "tod_bucket": pl.String,
                "upper_distance_bp": pl.Float64,
                "independent_lower_distance_bp": pl.Float64,
                "source_asof_date": pl.String,
                "touch_second": pl.Int64,
                "touch_timestamp": pl.Datetime("ns"),
                "touch_residual_bp": pl.Float64,
                "center_hit": pl.Boolean,
                "center_hit_second": pl.Int64,
                "time_to_center_seconds": pl.Int64,
                "observed_post_touch_floor_bp": pl.Float64,
                "floor_frontier_distance_bp": pl.List(pl.Float64),
                "floor_frontier_hit_second": pl.List(pl.Int64),
                "negative_cycle_completed": pl.Boolean,
                "right_censored": pl.Boolean,
                "right_censor_reason": pl.String,
                "path_end_second": pl.Int64,
                "independent_lower_confirmed_hit": pl.Boolean,
                "independent_lower_hit_second": pl.Int64,
                "independent_lower_time_from_touch_seconds": pl.Int64,
                "independent_lower_time_from_center_seconds": pl.Int64,
                "independent_lower_known_miss": pl.Boolean,
                "independent_lower_unknown": pl.Boolean,
            }
        )
    return pl.from_dicts(records, infer_schema_length=None).sort(
        [
            "Date",
            "ValueCode",
            "QuoteCode",
            "candidate_id",
            "boundary_quantile",
            "episode_sequence",
        ]
    )


__all__ = ["build_post_touch_convergence_facts"]
