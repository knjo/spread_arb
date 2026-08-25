"""Indexed batch estimation for frozen causal boundary candidates.

The scalar functions in :mod:`foundation_boundary_selection` are convenient
contracts and test oracles, but repeatedly filtering a multi-million-row
history once per target product is too expensive for the canonical rebuild.
This module normalizes one anchor's entry-window episodes once, partitions
them by product and side, and estimates all requested quantiles after one sort
per historical sample.

Only the raw Q0/Q1/Q2/Q5/Q6 families are implemented here.  Global level
scalers, time-of-day multipliers, and fallback application remain separate
runner stages.  Every history selector uses the supplied frozen market-session
calendar and rejects target-day/future observations rather than relying on
calendar-day arithmetic.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Final

import polars as pl

from .foundation_boundary_selection import (
    BOUNDARY_CANDIDATE_SPECS,
    BOUNDARY_PREDICTION_SCHEMA,
    DEFAULT_TOD_BUCKETS,
    MINIMUM_COMPLETED_PER_SIDE,
    PRIMARY_QUANTILES,
    BoundaryCandidateSpec,
    TodBucket,
    validate_monotonic_quantiles,
    validate_strict_source_asof,
)

PRODUCT_KEYS: Final = ("Date", "ValueCode", "QuoteCode")
SIDES: Final = ("positive", "negative")
ENTRY_START_SECOND: Final = 300
ENTRY_STOP_SECOND: Final = 14_400
RAW_BATCH_CANDIDATES: Final = (
    "Q0_trail60_event_pooled",
    "Q1_trail60_date_equal",
    "Q2_trail20_date_equal",
    "Q5_prev1_expiry_dte5",
    "Q6_prev2_expiry_dte5",
)
BATCH_PREDICTION_KEYS: Final = (
    "Date",
    "ValueCode",
    "QuoteCode",
    "anchor_model_id",
    "candidate_id",
    "tod_bucket",
    "boundary_quantile",
    "side",
)


def _batch_prediction_schema() -> pl.Schema:
    fields: dict[str, pl.DataType] = {}
    for name, dtype in BOUNDARY_PREDICTION_SCHEMA.items():
        fields[name] = dtype
        if name == "QuoteCode":
            fields["anchor_model_id"] = pl.String
    return pl.Schema(fields)


BATCH_BOUNDARY_PREDICTION_SCHEMA: Final = _batch_prediction_schema()


@dataclass(frozen=True)
class _Lineage:
    source_asof_date: str | None
    history_start_date: str | None
    history_end_date: str | None
    selection_sessions: int
    selected_expiry_cycles: tuple[str, ...] = ()
    selection_reason: str | None = None


@dataclass(frozen=True)
class _HistoricalSelection:
    lineage: _Lineage
    by_side: Mapping[str, pl.DataFrame]


@dataclass(frozen=True)
class _IntervalEstimate:
    completed_point_bp: float | None
    identified_lower_bp: float | None
    identified_upper_bp: float | None
    identified_upper_unbounded: bool


@dataclass(frozen=True)
class _SampleEstimates:
    by_quantile: Mapping[int, _IntervalEstimate]
    observable_started: int
    completed_count: int
    right_censored_count: int
    completed_history_dates: int


def _require_columns(
    frame: pl.DataFrame,
    required: set[str],
    source: str,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def _parse_session(value: object, source: str) -> str:
    text = str(value)
    try:
        datetime.strptime(text, "%Y%m%d")  # noqa: DTZ007
    except ValueError as error:
        raise ValueError(f"{source} must be YYYYMMDD: {value!r}") from error
    return text


def _normalise_sessions(sessions: Sequence[str]) -> tuple[str, ...]:
    result = tuple(_parse_session(value, "session") for value in sessions)
    if not result:
        raise ValueError("frozen session calendar must not be empty")
    if result != tuple(sorted(result)) or len(result) != len(set(result)):
        raise ValueError("frozen session calendar must be unique and sorted")
    return result


def _normalised_date_expr(column: str) -> pl.Expr:
    text = pl.col(column).cast(pl.String)
    return pl.coalesce(
        text.str.strptime(pl.Date, "%Y%m%d", strict=False),
        text.str.strptime(pl.Date, "%Y-%m-%d", strict=False),
    )


def _normalise_date_column(
    frame: pl.DataFrame,
    column: str,
    *,
    source: str,
    allow_null: bool = False,
) -> pl.DataFrame:
    _require_columns(frame, {column}, source)
    parsed = f"__parsed_{column}"
    result = frame.with_columns(_normalised_date_expr(column).alias(parsed))
    invalid = pl.col(column).is_not_null() & pl.col(parsed).is_null()
    if not allow_null:
        invalid = invalid | pl.col(column).is_null()
    if result.filter(invalid).height:
        raise ValueError(f"{source} contains invalid {column}")
    return result.drop(column).rename({parsed: column})


def _normalise_expiry_annotations(
    frame: pl.DataFrame,
    *,
    source: str,
) -> pl.DataFrame:
    result = frame
    if "expiry_date" not in result.columns:
        if "end_date" not in result.columns:
            raise ValueError(f"{source} requires expiry_date or end_date")
        result = result.rename({"end_date": "expiry_date"})
    result = _normalise_date_column(
        result,
        "expiry_date",
        source=source,
        allow_null=True,
    )
    trade_date = _normalised_date_expr("Date")
    expected_dte = (pl.col("expiry_date") - trade_date).dt.total_days().cast(pl.Int32)
    if "calendar_dte" not in result.columns:
        result = result.with_columns(expected_dte.alias("calendar_dte"))
    else:
        result = result.with_columns(
            pl.col("calendar_dte").cast(pl.Int32).alias("calendar_dte")
        )
    invalid = result.filter(
        (pl.col("expiry_date").is_null() != pl.col("calendar_dte").is_null())
        | (
            pl.col("expiry_date").is_not_null()
            & (pl.col("calendar_dte") != expected_dte)
        )
        | (pl.col("calendar_dte") < 0)
    )
    if invalid.height:
        raise ValueError(f"{source} has inconsistent expiry/DTE annotations")
    return result


def _validated_quantiles(quantiles: Sequence[int]) -> tuple[int, ...]:
    result = tuple(sorted(int(value) for value in quantiles))
    if (
        not result
        or len(result) != len(set(result))
        or any(value <= 0 or value >= 100 for value in result)
    ):
        raise ValueError("quantiles must be unique integers in (0, 100)")
    return result


def _validated_minima(
    quantiles: Sequence[int],
    minima: Mapping[int, int],
) -> dict[int, int]:
    missing = sorted(set(quantiles) - set(minima))
    if missing:
        raise ValueError(f"minimum completed counts missing q values: {missing}")
    result = {quantile: int(minima[quantile]) for quantile in quantiles}
    if any(value <= 0 for value in result.values()):
        raise ValueError("minimum completed counts must be positive")
    return result


def _validated_candidate_ids(candidate_ids: Sequence[str]) -> tuple[str, ...]:
    result = tuple(dict.fromkeys(str(value) for value in candidate_ids))
    if not result:
        raise ValueError("candidate_ids must not be empty")
    unknown = sorted(set(result) - set(RAW_BATCH_CANDIDATES))
    if unknown:
        raise ValueError(f"unsupported raw batch candidates: {unknown}")
    if len(result) != len(candidate_ids):
        raise ValueError("candidate_ids must not contain duplicates")
    return result


def _validated_tod_buckets(
    buckets: Sequence[TodBucket],
) -> tuple[TodBucket, ...]:
    result = tuple(buckets)
    if not result or len({bucket.id for bucket in result}) != len(result):
        raise ValueError("TOD buckets must be nonempty and unique")
    ordered = sorted(result, key=lambda item: item.start_second)
    for index, bucket in enumerate(ordered):
        if (
            not bucket.id
            or bucket.start_second < ENTRY_START_SECOND
            or bucket.end_second > ENTRY_STOP_SECOND
            or bucket.start_second >= bucket.end_second
        ):
            raise ValueError("TOD bucket is outside the frozen entry window")
        if index and ordered[index - 1].end_second != bucket.start_second:
            raise ValueError("TOD buckets must be contiguous and nonoverlapping")
    if (
        ordered[0].start_second != ENTRY_START_SECOND
        or ordered[-1].end_second != ENTRY_STOP_SECOND
    ):
        raise ValueError("TOD buckets must cover the complete entry window")
    return tuple(ordered)


def _weighted_endpoint_quantiles(
    frame: pl.DataFrame,
    probabilities: Sequence[float],
    *,
    date_equal: bool,
    upper_endpoints: bool,
) -> dict[float, float | None]:
    requested = tuple(float(value) for value in probabilities)
    if frame.is_empty():
        return dict.fromkeys(requested)
    endpoint = (
        pl.when(pl.col("right_censored"))
        .then(pl.lit(math.inf))
        .otherwise(pl.col("observed_amplitude_bp"))
        if upper_endpoints
        else pl.col("observed_amplitude_bp")
    )
    weight = 1.0 / pl.len().over("Date") if date_equal else pl.lit(1.0)
    ordered = (
        frame.select(
            endpoint.cast(pl.Float64).alias("_endpoint"),
            weight.cast(pl.Float64).alias("_weight"),
        )
        .sort("_endpoint")
        .with_columns(pl.col("_weight").cum_sum().alias("_cumulative"))
    )
    total = float(ordered.item(-1, "_cumulative"))
    endpoints = ordered["_endpoint"].to_list()
    cumulative = ordered["_cumulative"].to_list()
    result: dict[float, float | None] = {}
    index = 0
    for probability in sorted(set(requested)):
        threshold = probability * total
        while (
            index + 1 < len(cumulative) and float(cumulative[index]) + 1e-12 < threshold
        ):
            index += 1
        result[probability] = float(endpoints[index])
    return result


def _estimate_sample(
    sample: pl.DataFrame,
    *,
    quantiles: Sequence[int],
    date_equal: bool,
) -> _SampleEstimates:
    primary = sample.filter(~pl.col("left_censored"))
    completed = primary.filter(pl.col("completed_center_return"))
    probabilities = tuple(value / 100.0 for value in quantiles)
    completed_points = _weighted_endpoint_quantiles(
        completed,
        probabilities,
        date_equal=date_equal,
        upper_endpoints=False,
    )
    lower_points = _weighted_endpoint_quantiles(
        primary,
        probabilities,
        date_equal=date_equal,
        upper_endpoints=False,
    )
    upper_points = _weighted_endpoint_quantiles(
        primary,
        probabilities,
        date_equal=date_equal,
        upper_endpoints=True,
    )
    estimates: dict[int, _IntervalEstimate] = {}
    for quantile, probability in zip(quantiles, probabilities, strict=True):
        upper_value = upper_points[probability]
        upper_unbounded = upper_value is not None and math.isinf(upper_value)
        estimates[quantile] = _IntervalEstimate(
            completed_point_bp=completed_points[probability],
            identified_lower_bp=lower_points[probability],
            identified_upper_bp=None if upper_unbounded else upper_value,
            identified_upper_unbounded=upper_unbounded,
        )
    return _SampleEstimates(
        by_quantile=estimates,
        observable_started=primary.height,
        completed_count=completed.height,
        right_censored_count=primary.filter(pl.col("right_censored")).height,
        completed_history_dates=(
            completed["Date"].n_unique() if completed.height else 0
        ),
    )


def _clipped_point(
    estimate: _IntervalEstimate,
    *,
    event_pooled: bool,
) -> float | None:
    point = estimate.completed_point_bp
    if point is None or event_pooled:
        return point
    result = float(point)
    if estimate.identified_lower_bp is not None:
        result = max(result, float(estimate.identified_lower_bp))
    if estimate.identified_upper_bp is not None:
        result = min(result, float(estimate.identified_upper_bp))
    return result


class BoundaryBatchEngine:
    """Indexed raw-boundary engine for one frozen anchor."""

    def __init__(
        self,
        episodes: pl.DataFrame,
        sessions: Sequence[str],
    ) -> None:
        self.sessions = _normalise_sessions(sessions)
        self._session_index = {
            session: index for index, session in enumerate(self.sessions)
        }
        self._episodes = self._normalise_episodes(episodes)
        anchor_ids = self._episodes["anchor_model_id"].unique().to_list()
        if len(anchor_ids) != 1:
            raise ValueError("batch episodes must contain exactly one anchor_model_id")
        self.anchor_model_id = str(anchor_ids[0])
        self._empty = self._episodes.head(0)
        self._by_product_side: dict[tuple[str, str], pl.DataFrame] = {}
        for raw_key, group in self._episodes.group_by(
            ["ValueCode", "side"], maintain_order=False
        ):
            key = (str(raw_key[0]), str(raw_key[1]))
            self._by_product_side[key] = group.sort("_session_index")

    def _normalise_episodes(self, episodes: pl.DataFrame) -> pl.DataFrame:
        required = {
            *PRODUCT_KEYS,
            "anchor_model_id",
            "side",
            "observed_amplitude_bp",
            "left_censored",
            "right_censored",
            "completed_center_return",
            "start_seconds_from_open",
            "expiry_date",
            "calendar_dte",
        }
        _require_columns(episodes, required, "batch episodes")
        if episodes.is_empty():
            raise ValueError("batch episodes must not be empty")
        frame = episodes.with_columns(
            pl.col("Date").cast(pl.String),
            pl.col("ValueCode").cast(pl.String),
            pl.col("QuoteCode").cast(pl.String),
            pl.col("anchor_model_id").cast(pl.String),
            pl.col("side").cast(pl.String),
            pl.col("observed_amplitude_bp").cast(pl.Float64),
            pl.col("left_censored").fill_null(False).cast(pl.Boolean),
            pl.col("right_censored").fill_null(False).cast(pl.Boolean),
            pl.col("completed_center_return").fill_null(False).cast(pl.Boolean),
            pl.col("start_seconds_from_open").cast(pl.Int32),
        )
        frame = _normalise_expiry_annotations(frame, source="batch episodes")
        unknown_dates = sorted(set(frame["Date"].to_list()) - set(self.sessions))
        if unknown_dates:
            raise ValueError(
                "batch episodes contain dates outside the frozen calendar: "
                f"{unknown_dates[:5]}"
            )
        frame = frame.with_columns(
            pl.col("Date")
            .replace_strict(self._session_index, return_dtype=pl.Int32)
            .alias("_session_index")
        )
        invalid = frame.filter(
            pl.col("ValueCode").is_null()
            | pl.col("QuoteCode").is_null()
            | pl.col("anchor_model_id").is_null()
            | (pl.col("ValueCode").str.len_chars() == 0)
            | (pl.col("QuoteCode").str.len_chars() == 0)
            | (pl.col("anchor_model_id").str.len_chars() == 0)
            | ~pl.col("side").is_in(SIDES)
            | ~pl.col("observed_amplitude_bp").is_finite()
            | (pl.col("observed_amplitude_bp") < 0)
            | (pl.col("right_censored") == pl.col("completed_center_return"))
            | (pl.col("start_seconds_from_open") < ENTRY_START_SECOND)
            | (pl.col("start_seconds_from_open") >= ENTRY_STOP_SECOND)
        )
        if invalid.height:
            raise ValueError(
                "batch episodes violate key, censor, amplitude, or entry-window "
                "invariants"
            )
        if "end_seconds_from_open" in frame.columns:
            frame = frame.with_columns(
                pl.col("end_seconds_from_open")
                .cast(pl.Int32)
                .alias("end_seconds_from_open")
            )
            invalid_end = frame.filter(
                (pl.col("end_seconds_from_open") < pl.col("start_seconds_from_open"))
                | (pl.col("end_seconds_from_open") >= ENTRY_STOP_SECOND)
            )
            if invalid_end.height:
                raise ValueError("batch episodes cross the frozen 13:00 entry cutoff")
        if "right_censor_reason" in frame.columns:
            frame = frame.with_columns(pl.col("right_censor_reason").cast(pl.String))
            reason_invalid = frame.filter(
                (pl.col("right_censor_reason") == "entry_stop")
                & ~pl.col("right_censored")
            )
            if "end_seconds_from_open" in frame.columns:
                reason_invalid = pl.concat(
                    [
                        reason_invalid,
                        frame.filter(
                            (pl.col("right_censor_reason") == "entry_stop")
                            & (pl.col("end_seconds_from_open") != ENTRY_STOP_SECOND - 1)
                        ),
                        frame.filter(
                            pl.col("right_censored")
                            & (pl.col("end_seconds_from_open") == ENTRY_STOP_SECOND - 1)
                            & (pl.col("right_censor_reason") != "entry_stop").fill_null(
                                True
                            )
                        ),
                    ],
                    how="vertical_relaxed",
                )
            if reason_invalid.height:
                raise ValueError("batch episodes have invalid entry-stop censoring")
        if "tod_bucket" in frame.columns:
            expected_tod = (
                pl.when(pl.col("start_seconds_from_open") < 3_600)
                .then(pl.lit("0905_1000"))
                .when(pl.col("start_seconds_from_open") < 7_200)
                .then(pl.lit("1000_1100"))
                .when(pl.col("start_seconds_from_open") < 10_800)
                .then(pl.lit("1100_1200"))
                .otherwise(pl.lit("1200_1300"))
            )
            if frame.filter(
                (pl.col("tod_bucket") != expected_tod).fill_null(True)
            ).height:
                raise ValueError(
                    "batch episode TOD does not match its frozen start bucket"
                )
        return frame.select(
            "Date",
            "ValueCode",
            "QuoteCode",
            "anchor_model_id",
            "side",
            "observed_amplitude_bp",
            "left_censored",
            "right_censored",
            "completed_center_return",
            "start_seconds_from_open",
            "expiry_date",
            "calendar_dte",
            "_session_index",
        ).sort(["_session_index", "ValueCode", "QuoteCode", "side"])

    def _normalise_targets(
        self,
        target_mapping: pl.DataFrame,
        target_dates: Sequence[str] | None,
    ) -> tuple[pl.DataFrame, tuple[str, ...]]:
        _require_columns(target_mapping, set(PRODUCT_KEYS), "target mapping")
        frame = target_mapping.with_columns(
            pl.col("Date").cast(pl.String),
            pl.col("ValueCode").cast(pl.String),
            pl.col("QuoteCode").cast(pl.String),
        )
        frame = _normalise_expiry_annotations(frame, source="target mapping")
        unknown = sorted(set(frame["Date"].to_list()) - set(self.sessions))
        if unknown:
            raise ValueError(
                "target mapping contains dates outside the frozen calendar: "
                f"{unknown[:5]}"
            )
        requested = (
            tuple(sorted(frame["Date"].unique().to_list()))
            if target_dates is None
            else tuple(str(value) for value in target_dates)
        )
        if (
            not requested
            or len(requested) != len(set(requested))
            or requested != tuple(sorted(requested))
        ):
            raise ValueError("target_dates must be nonempty, unique, and sorted")
        absent = sorted(set(requested) - set(self.sessions))
        if absent:
            raise ValueError(f"target_dates are outside the calendar: {absent}")
        frame = frame.filter(pl.col("Date").is_in(requested)).select(
            *PRODUCT_KEYS,
            "expiry_date",
            "calendar_dte",
        )
        if frame.is_empty():
            raise ValueError("target mapping has no requested target rows")
        missing_dates = sorted(set(requested) - set(frame["Date"].unique()))
        if missing_dates:
            raise ValueError(f"target mapping lacks requested dates: {missing_dates}")
        duplicate = frame.group_by(list(PRODUCT_KEYS)).len().filter(pl.col("len") != 1)
        if duplicate.height:
            raise ValueError("target mapping contains duplicate exact contracts")
        multiple_contracts = (
            frame.group_by(["Date", "ValueCode"])
            .agg(pl.col("QuoteCode").n_unique().alias("contracts"))
            .filter(pl.col("contracts") != 1)
        )
        if multiple_contracts.height:
            raise ValueError("target mapping must have one QuoteCode per product-day")
        invalid_keys = frame.filter(
            pl.col("ValueCode").is_null()
            | pl.col("QuoteCode").is_null()
            | (pl.col("ValueCode").str.len_chars() == 0)
            | (pl.col("QuoteCode").str.len_chars() == 0)
        )
        if invalid_keys.height:
            raise ValueError("target mapping contains null or empty product keys")
        return frame.sort(list(PRODUCT_KEYS)), requested

    def _trailing_selection(
        self,
        *,
        product: str,
        target_index: int,
        lookback_sessions: int,
    ) -> _HistoricalSelection:
        start = max(0, target_index - lookback_sessions)
        prior_indices = tuple(range(start, target_index))
        if prior_indices:
            lineage = _Lineage(
                source_asof_date=self.sessions[prior_indices[-1]],
                history_start_date=self.sessions[prior_indices[0]],
                history_end_date=self.sessions[prior_indices[-1]],
                selection_sessions=len(prior_indices),
            )
        else:
            lineage = _Lineage(None, None, None, 0)
        by_side: dict[str, pl.DataFrame] = {}
        for side in SIDES:
            product_side = self._by_product_side.get((product, side), self._empty)
            by_side[side] = product_side.filter(
                (pl.col("_session_index") >= start)
                & (pl.col("_session_index") < target_index)
            )
        return _HistoricalSelection(lineage=lineage, by_side=by_side)

    def _prior_expiry_selection(
        self,
        *,
        product: str,
        target_index: int,
        target_expiry: date | None,
        target_dte: int | None,
        prior_cycles: int,
        dte_radius: int,
    ) -> _HistoricalSelection:
        if target_expiry is None or target_dte is None:
            return _HistoricalSelection(
                lineage=_Lineage(
                    None,
                    None,
                    None,
                    0,
                    selection_reason="unknown_target_expiry_or_dte",
                ),
                by_side={side: self._empty for side in SIDES},
            )
        prior_by_side: dict[str, pl.DataFrame] = {}
        cycles: set[date] = set()
        for side in SIDES:
            product_side = self._by_product_side.get((product, side), self._empty)
            prior = product_side.filter(
                (pl.col("_session_index") < target_index)
                & pl.col("expiry_date").is_not_null()
                & (pl.col("expiry_date") < pl.lit(target_expiry))
            )
            prior_by_side[side] = prior
            cycles.update(prior["expiry_date"].drop_nulls().to_list())
        chosen = tuple(sorted(cycles, reverse=True)[:prior_cycles])
        reason = (
            None
            if len(chosen) == prior_cycles
            else f"insufficient_prior_expiry_cycles:{len(chosen)}<{prior_cycles}"
        )
        selected: dict[str, pl.DataFrame] = {}
        selected_indices: set[int] = set()
        for side in SIDES:
            frame = prior_by_side[side].filter(
                pl.col("expiry_date").is_in(chosen)
                & ((pl.col("calendar_dte") - int(target_dte)).abs() <= dte_radius)
            )
            selected[side] = frame
            selected_indices.update(frame["_session_index"].to_list())
        ordered_indices = sorted(selected_indices)
        lineage = _Lineage(
            source_asof_date=(
                self.sessions[ordered_indices[-1]] if ordered_indices else None
            ),
            history_start_date=(
                self.sessions[ordered_indices[0]] if ordered_indices else None
            ),
            history_end_date=(
                self.sessions[ordered_indices[-1]] if ordered_indices else None
            ),
            selection_sessions=len(ordered_indices),
            selected_expiry_cycles=tuple(value.strftime("%Y%m%d") for value in chosen),
            selection_reason=reason,
        )
        return _HistoricalSelection(lineage=lineage, by_side=selected)

    def predict(
        self,
        target_mapping: pl.DataFrame,
        *,
        target_dates: Sequence[str] | None = None,
        candidate_ids: Sequence[str] = RAW_BATCH_CANDIDATES,
        quantiles: Sequence[int] = PRIMARY_QUANTILES,
        minimum_completed_per_side: Mapping[int, int] = MINIMUM_COMPLETED_PER_SIDE,
        tod_buckets: Sequence[TodBucket] = DEFAULT_TOD_BUCKETS,
    ) -> pl.DataFrame:
        """Emit all raw D-safe candidate/q cells for requested target dates."""

        candidates = _validated_candidate_ids(candidate_ids)
        expected_quantiles = _validated_quantiles(quantiles)
        minima = _validated_minima(expected_quantiles, minimum_completed_per_side)
        buckets = _validated_tod_buckets(tod_buckets)
        targets, requested = self._normalise_targets(target_mapping, target_dates)
        records: list[dict[str, object]] = []
        for target_date in requested:
            target_index = self._session_index[target_date]
            date_targets = targets.filter(pl.col("Date") == target_date)
            trailing_cache: dict[tuple[str, int], _HistoricalSelection] = {}
            expiry_cache: dict[
                tuple[str, date | None, int | None, int, int],
                _HistoricalSelection,
            ] = {}
            estimate_cache: dict[tuple[str, str, str], _SampleEstimates] = {}
            for target in date_targets.iter_rows(named=True):
                product = str(target["ValueCode"])
                target_expiry = target["expiry_date"]
                target_dte_raw = target["calendar_dte"]
                target_dte = None if target_dte_raw is None else int(target_dte_raw)
                for candidate_id in candidates:
                    spec = BOUNDARY_CANDIDATE_SPECS[candidate_id]
                    if spec.lookback_sessions is not None:
                        selection_key = (product, spec.lookback_sessions)
                        selection = trailing_cache.get(selection_key)
                        if selection is None:
                            selection = self._trailing_selection(
                                product=product,
                                target_index=target_index,
                                lookback_sessions=spec.lookback_sessions,
                            )
                            trailing_cache[selection_key] = selection
                    else:
                        prior_cycles = int(spec.prior_expiry_cycles or 0)
                        dte_radius = int(spec.dte_radius_calendar_days or 0)
                        expiry_key = (
                            product,
                            target_expiry,
                            target_dte,
                            prior_cycles,
                            dte_radius,
                        )
                        selection = expiry_cache.get(expiry_key)
                        if selection is None:
                            selection = self._prior_expiry_selection(
                                product=product,
                                target_index=target_index,
                                target_expiry=target_expiry,
                                target_dte=target_dte,
                                prior_cycles=prior_cycles,
                                dte_radius=dte_radius,
                            )
                            expiry_cache[expiry_key] = selection
                    for side in SIDES:
                        cache_key = (product, side, candidate_id)
                        estimates = estimate_cache.get(cache_key)
                        if estimates is None:
                            estimates = _estimate_sample(
                                selection.by_side[side],
                                quantiles=expected_quantiles,
                                date_equal=(
                                    spec.kind != "event_pooled_completed_control"
                                ),
                            )
                            estimate_cache[cache_key] = estimates
                        records.extend(
                            self._prediction_records(
                                target=target,
                                spec=spec,
                                side=side,
                                lineage=selection.lineage,
                                estimates=estimates,
                                quantiles=expected_quantiles,
                                minima=minima,
                                tod_buckets=buckets,
                            )
                        )
        result = pl.from_dicts(
            records,
            schema=BATCH_BOUNDARY_PREDICTION_SCHEMA,
            infer_schema_length=None,
        ).sort(list(BATCH_PREDICTION_KEYS))
        duplicate = (
            result.group_by(list(BATCH_PREDICTION_KEYS))
            .len()
            .filter(pl.col("len") != 1)
        )
        if duplicate.height:
            raise ValueError("batch predictions contain duplicate full keys")
        validate_strict_source_asof(result)
        validate_monotonic_quantiles(result, value_column="clipped_point_bp")
        validate_monotonic_quantiles(result, value_column="boundary_distance_bp")
        return result

    def _prediction_records(
        self,
        *,
        target: Mapping[str, object],
        spec: BoundaryCandidateSpec,
        side: str,
        lineage: _Lineage,
        estimates: _SampleEstimates,
        quantiles: Sequence[int],
        minima: Mapping[int, int],
        tod_buckets: Sequence[TodBucket],
    ) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        event_pooled = spec.kind == "event_pooled_completed_control"
        for quantile in quantiles:
            estimate = estimates.by_quantile[quantile]
            clipped = _clipped_point(estimate, event_pooled=event_pooled)
            reasons: list[str] = []
            if lineage.selection_reason is not None:
                reasons.append(lineage.selection_reason)
            if estimates.completed_history_dates < spec.minimum_dates:
                reasons.append(
                    "insufficient_completed_dates:"
                    f"{estimates.completed_history_dates}<{spec.minimum_dates}"
                )
            minimum_completed = minima[quantile]
            if estimates.completed_count < minimum_completed:
                reasons.append(
                    "insufficient_completed_count:"
                    f"{estimates.completed_count}<{minimum_completed}"
                )
            if clipped is None or not math.isfinite(float(clipped)):
                reasons.append("unavailable_finite_point")
            elif float(clipped) <= 0:
                reasons.append("nonpositive_point")
            native_supported = not reasons
            common = {
                "Date": str(target["Date"]),
                "ValueCode": str(target["ValueCode"]),
                "QuoteCode": str(target["QuoteCode"]),
                "anchor_model_id": self.anchor_model_id,
                "boundary_quantile": quantile,
                "side": side,
                "candidate_id": spec.id,
                "candidate_kind": spec.kind,
                "source_asof_date": lineage.source_asof_date,
                "history_start_date": lineage.history_start_date,
                "history_end_date": lineage.history_end_date,
                "lookback_sessions": spec.lookback_sessions,
                "prior_expiry_cycles": spec.prior_expiry_cycles,
                "selected_expiry_cycles": (
                    ",".join(lineage.selected_expiry_cycles)
                    if lineage.selected_expiry_cycles
                    else None
                ),
                "target_expiry_date": target["expiry_date"],
                "target_calendar_dte": target["calendar_dte"],
                "minimum_dates": spec.minimum_dates,
                "minimum_completed": minimum_completed,
                "selection_sessions": lineage.selection_sessions,
                "completed_history_dates": estimates.completed_history_dates,
                "observable_started": estimates.observable_started,
                "completed_count": estimates.completed_count,
                "right_censored_count": estimates.right_censored_count,
                "completed_point_bp": estimate.completed_point_bp,
                "identified_lower_bp": estimate.identified_lower_bp,
                "identified_upper_bp": estimate.identified_upper_bp,
                "identified_upper_unbounded": (estimate.identified_upper_unbounded),
                "clipped_point_bp": clipped,
                "boundary_distance_bp": clipped if native_supported else None,
                "native_supported": native_supported,
                "allow_primary_selection": spec.allow_primary_selection,
                "fallback_candidate_id": spec.fallback_candidate,
                "fallback_reason": ";".join(reasons) if reasons else None,
                "contains_target_day_outcome": False,
            }
            for bucket in tod_buckets:
                records.append({**common, "tod_bucket": bucket.id})
        return records


def build_boundary_batch_predictions(
    episodes: pl.DataFrame,
    sessions: Sequence[str],
    target_mapping: pl.DataFrame,
    *,
    target_dates: Sequence[str] | None = None,
    candidate_ids: Sequence[str] = RAW_BATCH_CANDIDATES,
    quantiles: Sequence[int] = PRIMARY_QUANTILES,
    minimum_completed_per_side: Mapping[int, int] = MINIMUM_COMPLETED_PER_SIDE,
    tod_buckets: Sequence[TodBucket] = DEFAULT_TOD_BUCKETS,
) -> pl.DataFrame:
    """Convenience API that builds one indexed engine and emits predictions."""

    return BoundaryBatchEngine(episodes, sessions).predict(
        target_mapping,
        target_dates=target_dates,
        candidate_ids=candidate_ids,
        quantiles=quantiles,
        minimum_completed_per_side=minimum_completed_per_side,
        tod_buckets=tod_buckets,
    )


__all__ = [
    "BATCH_BOUNDARY_PREDICTION_SCHEMA",
    "BATCH_PREDICTION_KEYS",
    "RAW_BATCH_CANDIDATES",
    "BoundaryBatchEngine",
    "build_boundary_batch_predictions",
]
