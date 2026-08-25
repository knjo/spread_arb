"""Pure, leakage-safe boundary candidate estimators for selected anchors.

The functions in this module operate on in-memory Polars frames.  They do not
read or publish files and they do not implement the later convergence study.
They provide the shared contracts needed by a runner to build Q0, Q1, Q2, Q5,
and Q6 from the frozen foundation selection registry.

Base boundary candidates are estimated over the whole entry session.  The
episode-start time-of-day bucket is retained as an evaluation key and is
therefore repeated on the long prediction output.  Time-of-day scaling is a
separate, later candidate family.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from itertools import pairwise
from typing import Final

import polars as pl

PRODUCT_KEYS: Final = ("Date", "ValueCode", "QuoteCode")
ANALYSIS_START_SECOND: Final = 300
ENTRY_STOP_SECOND: Final = 14_400
PREDICTION_KEYS: Final = (
    "Date",
    "ValueCode",
    "QuoteCode",
    "tod_bucket",
    "boundary_quantile",
    "side",
)
SIDES: Final = ("positive", "negative")
PRIMARY_QUANTILES: Final = (50, 80, 95)


@dataclass(frozen=True)
class TodBucket:
    """One half-open episode-start time-of-day interval."""

    id: str
    start_second: int
    end_second: int


DEFAULT_TOD_BUCKETS: Final = (
    TodBucket("0905_1000", 300, 3_600),
    TodBucket("1000_1100", 3_600, 7_200),
    TodBucket("1100_1200", 7_200, 10_800),
    TodBucket("1200_1300", 10_800, 14_400),
)


@dataclass(frozen=True)
class BoundaryCandidateSpec:
    """Frozen fields used by the implemented base candidate families."""

    id: str
    kind: str
    minimum_dates: int
    lookback_sessions: int | None = None
    prior_expiry_cycles: int | None = None
    dte_radius_calendar_days: int | None = None
    fallback_candidate: str | None = None
    allow_primary_selection: bool = True


BOUNDARY_CANDIDATE_SPECS: Final = {
    "Q0_trail60_event_pooled": BoundaryCandidateSpec(
        id="Q0_trail60_event_pooled",
        kind="event_pooled_completed_control",
        lookback_sessions=60,
        minimum_dates=40,
        allow_primary_selection=False,
    ),
    "Q1_trail60_date_equal": BoundaryCandidateSpec(
        id="Q1_trail60_date_equal",
        kind="date_equal_censor_identified",
        lookback_sessions=60,
        minimum_dates=40,
    ),
    "Q2_trail20_date_equal": BoundaryCandidateSpec(
        id="Q2_trail20_date_equal",
        kind="date_equal_censor_identified",
        lookback_sessions=20,
        minimum_dates=15,
    ),
    "Q5_prev1_expiry_dte5": BoundaryCandidateSpec(
        id="Q5_prev1_expiry_dte5",
        kind="prior_expiry_dte_date_equal_diagnostic",
        prior_expiry_cycles=1,
        dte_radius_calendar_days=5,
        minimum_dates=5,
        allow_primary_selection=False,
    ),
    "Q6_prev2_expiry_dte5": BoundaryCandidateSpec(
        id="Q6_prev2_expiry_dte5",
        kind="prior_expiry_dte_date_equal",
        prior_expiry_cycles=2,
        dte_radius_calendar_days=5,
        minimum_dates=8,
        fallback_candidate="Q1_trail60_date_equal",
    ),
}

MINIMUM_COMPLETED_PER_SIDE: Final = {50: 100, 80: 200, 95: 400}


@dataclass(frozen=True)
class SelectedHistory:
    """A selected historical slice and its auditable lineage."""

    frame: pl.DataFrame
    source_asof_date: str | None
    history_start_date: str | None
    history_end_date: str | None
    selected_sessions: int
    selected_expiry_cycles: tuple[str, ...] = ()
    selection_reason: str | None = None


@dataclass(frozen=True)
class CensorIdentifiedInterval:
    """Sharp empirical q interval from exact and right-censored amplitudes."""

    lower_bp: float | None
    upper_bp: float | None
    upper_unbounded: bool
    observable_started: int
    completed_count: int
    right_censored_count: int
    history_dates: int


def extract_entry_window_excursions(
    causal_fair: pl.DataFrame,
    *,
    anchor_column: str,
    anchor_model_id: str,
) -> pl.DataFrame:
    """Extract entry-boundary episodes and censor active paths at 13:00.

    The generic excursion extractor follows a session through 13:20.  Entry q
    history may not use extrema observed after the final entry time, so this
    wrapper closes the analysis gate at second 14,400 before extraction and
    records an explicit ``entry_stop`` right-censor reason.
    """

    _require_columns(
        causal_fair,
        {
            *PRODUCT_KEYS,
            "seconds_from_open",
            "analysis_eligible",
            anchor_column,
        },
        "causal fair panel",
    )
    model_id = str(anchor_model_id)
    if not model_id:
        raise ValueError("anchor_model_id must not be empty")
    episodes = _extract_entry_window_excursions_vectorized(
        causal_fair,
        anchor_column=anchor_column,
        anchor_model_id=model_id,
    )
    invalid = episodes.filter(
        (pl.col("start_seconds_from_open") < ANALYSIS_START_SECOND)
        | (pl.col("start_seconds_from_open") >= ENTRY_STOP_SECOND)
        | (
            (pl.col("end_seconds_from_open") >= ENTRY_STOP_SECOND)
            & (pl.col("right_censor_reason") != "entry_stop")
        )
    )
    if invalid.height:
        raise ValueError("entry-window episode extraction crossed the frozen cutoff")
    return assign_episode_start_tod_bucket(episodes, strict=True)


def _extract_entry_window_excursions_vectorized(
    causal_fair: pl.DataFrame,
    *,
    anchor_column: str,
    anchor_model_id: str,
) -> pl.DataFrame:
    """Vectorized equivalent of the generic row-state excursion extractor.

    A valid contiguous segment is split into runs of positive, zero, and
    negative residuals.  Every non-zero run is one episode: the first row of
    the next run is its center-return endpoint, while a final non-zero run is
    censored at the segment endpoint.  This is the same state transition as
    the generic extractor, but it keeps the full one-second panel inside
    Polars rather than materializing every row as a Python dictionary.
    """

    from .foundation_boundary import EPISODE_SCHEMA

    if causal_fair.is_empty():
        return pl.DataFrame(schema=EPISODE_SCHEMA).with_columns(
            pl.lit(anchor_model_id).alias("anchor_model_id")
        )

    panel = (
        causal_fair.select(
            *PRODUCT_KEYS,
            "timestamp",
            "seconds_from_open",
            "basis_mid_bp",
            "analysis_eligible",
            anchor_column,
        )
        .with_columns(
            pl.col("Date").cast(pl.String),
            pl.col("ValueCode").cast(pl.String),
            pl.col("QuoteCode").cast(pl.String),
            pl.col("timestamp").cast(pl.Datetime("ns")),
            pl.col("seconds_from_open").cast(pl.Int64),
            pl.col("basis_mid_bp").cast(pl.Float64),
            pl.col(anchor_column).cast(pl.Float64),
            pl.col("analysis_eligible").fill_null(False).cast(pl.Boolean),
        )
        .with_columns(
            (pl.col("basis_mid_bp") - pl.col(anchor_column)).alias("_residual_bp")
        )
    )
    _validate_vectorized_episode_panel(panel)
    panel = panel.sort([*PRODUCT_KEYS, "seconds_from_open", "timestamp"]).with_columns(
        pl.struct(PRODUCT_KEYS).rle_id().alias("_product_id")
    )
    timestamp_order_error = panel.filter(
        pl.col("timestamp")
        .cast(pl.Int64)
        .diff()
        .over("_product_id")
        .le(0)
        .fill_null(False)
    )
    if timestamp_order_error.height:
        raise ValueError("timestamps must increase within a product-day")

    entry_eligible = (
        pl.col("analysis_eligible")
        & pl.col("_residual_bp").is_not_null()
        & pl.col("_residual_bp").is_finite()
        & (pl.col("seconds_from_open") >= ANALYSIS_START_SECOND)
        & (pl.col("seconds_from_open") < ENTRY_STOP_SECOND)
    ).fill_null(False)
    panel = panel.with_columns(entry_eligible.alias("_valid")).with_columns(
        pl.col("seconds_from_open")
        .shift(1)
        .over("_product_id")
        .alias("_previous_second"),
        pl.col("seconds_from_open").shift(-1).over("_product_id").alias("_next_second"),
        pl.col("_valid")
        .shift(1)
        .over("_product_id")
        .fill_null(False)
        .alias("_previous_valid"),
        pl.col("_valid")
        .shift(-1)
        .over("_product_id")
        .fill_null(False)
        .alias("_next_valid"),
        pl.col("_valid")
        .cast(pl.UInt32)
        .cum_sum()
        .shift(1)
        .over("_product_id")
        .fill_null(0)
        .gt(0)
        .alias("_seen_valid_before"),
    )
    segment_start = (
        pl.col("_valid")
        & (
            ~pl.col("_previous_valid")
            | pl.col("_previous_second").is_null()
            | (pl.col("seconds_from_open") - pl.col("_previous_second") != 1)
        )
    ).fill_null(False)
    segment_end = (
        pl.col("_valid")
        & (
            ~pl.col("_next_valid")
            | pl.col("_next_second").is_null()
            | (pl.col("_next_second") - pl.col("seconds_from_open") != 1)
        )
    ).fill_null(False)
    panel = panel.with_columns(
        segment_start.alias("_segment_start"),
        segment_end.alias("_segment_end"),
    ).with_columns(
        pl.col("_segment_start")
        .cast(pl.UInt32)
        .cum_sum()
        .over("_product_id")
        .alias("_segment_id"),
        pl.when(pl.col("_segment_start"))
        .then(
            pl.when(~pl.col("_seen_valid_before"))
            .then(pl.lit("session_start"))
            .when(
                pl.col("_previous_second").is_not_null()
                & (pl.col("seconds_from_open") - pl.col("_previous_second") != 1)
            )
            .then(pl.lit("timestamp_gap"))
            .otherwise(pl.lit("eligibility_gap"))
        )
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias("_left_censor_reason"),
        pl.when(pl.col("_segment_end"))
        .then(
            pl.when(pl.col("_next_second").is_null())
            .then(pl.lit("session_cutoff"))
            .when(pl.col("_next_second") - pl.col("seconds_from_open") != 1)
            .then(pl.lit("timestamp_gap"))
            .otherwise(pl.lit("eligibility_gap"))
        )
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias("_right_censor_reason"),
    )

    valid = panel.filter(pl.col("_valid")).with_columns(
        pl.when(pl.col("_residual_bp") > 0)
        .then(pl.lit(1, dtype=pl.Int8))
        .when(pl.col("_residual_bp") < 0)
        .then(pl.lit(-1, dtype=pl.Int8))
        .otherwise(pl.lit(0, dtype=pl.Int8))
        .alias("_sign")
    )
    if valid.is_empty():
        return pl.DataFrame(schema=EPISODE_SCHEMA).with_columns(
            pl.lit(anchor_model_id).alias("anchor_model_id")
        )

    segment_keys = ["_product_id", "_segment_id"]
    valid = valid.with_columns(
        pl.col("_residual_bp")
        .shift(1)
        .over(segment_keys)
        .alias("_previous_residual_bp"),
        (pl.col("_sign") != pl.col("_sign").shift(1).over(segment_keys))
        .fill_null(True)
        .alias("_run_start"),
    ).with_columns(
        pl.col("_run_start")
        .cast(pl.UInt32)
        .cum_sum()
        .over(segment_keys)
        .alias("_run_id")
    )
    segments = valid.group_by(segment_keys, maintain_order=True).agg(
        *[pl.col(column).first().alias(column) for column in PRODUCT_KEYS],
        pl.col("timestamp").first().alias("_segment_start_timestamp"),
        pl.col("seconds_from_open").first().alias("_segment_start_second"),
        pl.col("timestamp").last().alias("_segment_end_timestamp"),
        pl.col("seconds_from_open").last().alias("_segment_end_second"),
        pl.col("_residual_bp").last().alias("_segment_end_residual_bp"),
        pl.col("_left_censor_reason")
        .drop_nulls()
        .first()
        .alias("_segment_left_censor_reason"),
        pl.col("_right_censor_reason")
        .drop_nulls()
        .last()
        .alias("_segment_right_censor_reason"),
    )
    run_keys = [*segment_keys, "_run_id"]
    runs = (
        valid.group_by(run_keys, maintain_order=True)
        .agg(
            pl.col("_sign").first().alias("_sign"),
            pl.col("timestamp").first().alias("start_timestamp"),
            pl.col("seconds_from_open").first().alias("start_seconds_from_open"),
            pl.col("_previous_residual_bp")
            .first()
            .alias("_start_previous_residual_bp"),
            pl.col("_residual_bp").first().alias("start_residual_bp"),
            pl.col("_residual_bp").abs().max().alias("observed_amplitude_bp"),
        )
        .join(segments, on=segment_keys, how="left", validate="m:1")
        .sort(run_keys)
        .with_columns(
            pl.col("start_timestamp")
            .shift(-1)
            .over(segment_keys)
            .alias("_next_run_timestamp"),
            pl.col("start_seconds_from_open")
            .shift(-1)
            .over(segment_keys)
            .alias("_next_run_second"),
            pl.col("start_residual_bp")
            .shift(-1)
            .over(segment_keys)
            .alias("_next_run_residual_bp"),
        )
        .filter(pl.col("_sign") != 0)
    )
    if runs.is_empty():
        return pl.DataFrame(schema=EPISODE_SCHEMA).with_columns(
            pl.lit(anchor_model_id).alias("anchor_model_id")
        )

    runs = (
        runs.with_columns(
            pl.col("_next_run_second").is_not_null().alias("_completed"),
            (
                pl.col("start_seconds_from_open") == pl.col("_segment_start_second")
            ).alias("left_censored"),
        )
        .with_columns(
            pl.when(pl.col("_completed"))
            .then(pl.col("_next_run_timestamp"))
            .otherwise(pl.col("_segment_end_timestamp"))
            .alias("end_timestamp"),
            pl.when(pl.col("_completed"))
            .then(pl.col("_next_run_second"))
            .otherwise(pl.col("_segment_end_second"))
            .alias("end_seconds_from_open"),
            pl.when(pl.col("_completed"))
            .then(pl.col("_next_run_residual_bp"))
            .otherwise(pl.col("_segment_end_residual_bp"))
            .alias("end_residual_bp"),
            pl.when(pl.col("left_censored"))
            .then(pl.col("_segment_left_censor_reason"))
            .otherwise(pl.lit(None, dtype=pl.String))
            .alias("left_censor_reason"),
            pl.when(pl.col("left_censored"))
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.col("_start_previous_residual_bp"))
            .alias("start_previous_residual_bp"),
            (~pl.col("_completed")).alias("right_censored"),
            pl.col("_completed").alias("completed_center_return"),
            (pl.col("_completed") & ~pl.col("left_censored")).alias("fully_observed"),
            pl.when(pl.col("_completed"))
            .then(pl.lit(None, dtype=pl.String))
            .otherwise(pl.col("_segment_right_censor_reason"))
            .alias("right_censor_reason"),
            pl.when(pl.col("_sign") > 0)
            .then(pl.lit("positive"))
            .otherwise(pl.lit("negative"))
            .alias("side"),
        )
        .sort([*PRODUCT_KEYS, "start_seconds_from_open"])
    )
    runs = runs.with_columns(
        pl.col("start_seconds_from_open")
        .cum_count()
        .over("_product_id")
        .cast(pl.Int64)
        .alias("episode_sequence")
    ).with_columns(
        pl.concat_str(
            [
                pl.col("Date"),
                pl.col("ValueCode"),
                pl.col("QuoteCode"),
                pl.col("episode_sequence").cast(pl.String).str.pad_start(6, "0"),
                pl.col("side"),
            ],
            separator="/",
        ).alias("episode_id"),
        (pl.col("end_seconds_from_open") - pl.col("start_seconds_from_open"))
        .cast(pl.Int64)
        .alias("duration_seconds"),
        pl.when(
            pl.col("right_censored")
            & (pl.col("end_seconds_from_open") == ENTRY_STOP_SECOND - 1)
        )
        .then(pl.lit("entry_stop"))
        .otherwise(pl.col("right_censor_reason"))
        .alias("right_censor_reason"),
        pl.lit(anchor_column).alias("anchor_column"),
        pl.lit("1s").alias("sampling_interval"),
        pl.lit(anchor_model_id).alias("anchor_model_id"),
    )
    return runs.select(
        *[pl.col(column).cast(dtype) for column, dtype in EPISODE_SCHEMA.items()],
        pl.col("anchor_model_id").cast(pl.String),
    ).sort([*PRODUCT_KEYS, "episode_sequence"])


def _validate_vectorized_episode_panel(panel: pl.DataFrame) -> None:
    """Validate the generic extractor's one-day input contract efficiently."""

    dates = panel["Date"].unique().to_list()
    if len(dates) != 1:
        raise ValueError("episode overlay input must contain exactly one Date")
    date_value = dates[0]
    if date_value is None:
        raise ValueError("causal fair panel contains null Date")
    try:
        datetime.strptime(str(date_value), "%Y%m%d")  # noqa: DTZ007
    except ValueError as error:
        raise ValueError(
            f"causal fair panel contains invalid YYYYMMDD Date: {date_value!r}"
        ) from error
    if panel.filter(pl.col("timestamp").is_null()).height:
        raise ValueError("causal fair panel contains null timestamps")
    if panel.filter(pl.col("seconds_from_open").is_null()).height:
        raise ValueError("causal fair panel contains null seconds_from_open")
    duplicate = panel.select(*PRODUCT_KEYS, "seconds_from_open").is_duplicated()
    if duplicate.any():
        raise ValueError("one-second causal fair panel contains duplicate keys")
    contract_counts = (
        panel.group_by(["Date", "ValueCode"])
        .agg(pl.col("QuoteCode").n_unique().alias("contracts"))
        .filter(pl.col("contracts") != 1)
    )
    if contract_counts.height:
        raise ValueError("one product-day must have exactly one QuoteCode")


BOUNDARY_PREDICTION_SCHEMA: Final = pl.Schema(
    {
        "Date": pl.String,
        "ValueCode": pl.String,
        "QuoteCode": pl.String,
        "tod_bucket": pl.String,
        "boundary_quantile": pl.Int64,
        "side": pl.String,
        "candidate_id": pl.String,
        "candidate_kind": pl.String,
        "source_asof_date": pl.String,
        "history_start_date": pl.String,
        "history_end_date": pl.String,
        "lookback_sessions": pl.Int64,
        "prior_expiry_cycles": pl.Int64,
        "selected_expiry_cycles": pl.String,
        "target_expiry_date": pl.Date,
        "target_calendar_dte": pl.Int32,
        "minimum_dates": pl.Int64,
        "minimum_completed": pl.Int64,
        "selection_sessions": pl.Int64,
        "completed_history_dates": pl.Int64,
        "observable_started": pl.Int64,
        "completed_count": pl.Int64,
        "right_censored_count": pl.Int64,
        "completed_point_bp": pl.Float64,
        "identified_lower_bp": pl.Float64,
        "identified_upper_bp": pl.Float64,
        "identified_upper_unbounded": pl.Boolean,
        "clipped_point_bp": pl.Float64,
        "boundary_distance_bp": pl.Float64,
        "native_supported": pl.Boolean,
        "allow_primary_selection": pl.Boolean,
        "fallback_candidate_id": pl.String,
        "fallback_reason": pl.String,
        "contains_target_day_outcome": pl.Boolean,
    }
)


def annotate_expiry_dte(
    frame: pl.DataFrame,
    contract_mapping: pl.DataFrame,
    *,
    expiry_column: str = "end_date",
) -> pl.DataFrame:
    """Attach exact-contract expiry and calendar DTE without a fuzzy join.

    Mapping is performed on ``Date, ValueCode, QuoteCode``.  Repeated mapping
    rows are allowed only when they carry the same expiry value.  A mapping row
    with an unknown expiry is retained and marked explicitly; a missing exact
    mapping row fails closed.
    """

    _require_columns(frame, set(PRODUCT_KEYS), "frame")
    _require_columns(
        contract_mapping,
        {*PRODUCT_KEYS, expiry_column},
        "contract mapping",
    )
    collisions = sorted(
        {
            "expiry_date",
            "calendar_dte",
            "expiry_cycle",
            "expiry_mapping_status",
        }
        & set(frame.columns)
    )
    if collisions:
        raise ValueError(f"frame already contains expiry annotations: {collisions}")

    base = _normalise_product_keys(frame, "frame").with_row_index("_input_order")
    mapping = _normalise_product_keys(contract_mapping, "contract mapping")
    mapping = _with_parsed_date(
        mapping,
        expiry_column,
        output="_expiry_date",
        source="contract mapping expiry",
        allow_null=True,
    ).select(
        *PRODUCT_KEYS,
        "_expiry_date",
    )
    conflicts = (
        mapping.group_by(list(PRODUCT_KEYS))
        .agg(pl.col("_expiry_date").n_unique().alias("expiry_values"))
        .filter(pl.col("expiry_values") != 1)
    )
    if conflicts.height:
        raise ValueError(
            "contract mapping has conflicting exact-contract expiries: "
            f"{conflicts.head(5).to_dicts()}"
        )
    mapping = (
        mapping.group_by(list(PRODUCT_KEYS), maintain_order=True)
        .agg(pl.col("_expiry_date").first().alias("expiry_date"))
        .with_columns(pl.lit(True).alias("_mapping_present"))
    )
    joined = base.join(
        mapping,
        on=list(PRODUCT_KEYS),
        how="left",
        validate="m:1",
    )
    missing = joined.filter(~pl.col("_mapping_present").fill_null(False))
    if missing.height:
        raise ValueError(
            "frame lacks an exact Date/ValueCode/QuoteCode expiry mapping: "
            f"{missing.select(PRODUCT_KEYS).unique().head(5).to_dicts()}"
        )

    trade_date = _date_text_expr("Date")
    expired = joined.filter(
        pl.col("expiry_date").is_not_null() & (pl.col("expiry_date") < trade_date)
    )
    if expired.height:
        raise ValueError("contract mapping contains expiry before its trade Date")

    result = (
        joined.with_columns(
            pl.when(pl.col("expiry_date").is_not_null())
            .then((pl.col("expiry_date") - trade_date).dt.total_days())
            .otherwise(None)
            .cast(pl.Int32)
            .alias("calendar_dte"),
            pl.col("expiry_date").dt.strftime("%Y%m%d").alias("expiry_cycle"),
            pl.when(pl.col("expiry_date").is_not_null())
            .then(pl.lit("mapped"))
            .otherwise(pl.lit("unknown_expiry"))
            .alias("expiry_mapping_status"),
        )
        .sort("_input_order")
        .drop("_input_order", "_mapping_present")
    )
    return result


def assign_episode_start_tod_bucket(
    episodes: pl.DataFrame,
    *,
    buckets: Sequence[TodBucket] = DEFAULT_TOD_BUCKETS,
    strict: bool = True,
) -> pl.DataFrame:
    """Assign the bucket from episode start and freeze it for the episode."""

    _require_columns(episodes, {"start_seconds_from_open"}, "episodes")
    validated = _validated_tod_buckets(buckets)
    seconds = pl.col("start_seconds_from_open").cast(pl.Int64)
    choices = [
        pl.when((seconds >= bucket.start_second) & (seconds < bucket.end_second)).then(
            pl.lit(bucket.id)
        )
        for bucket in validated
    ]
    result = episodes.with_columns(
        pl.coalesce([*choices, pl.lit(None, dtype=pl.String)]).alias("tod_bucket")
    )
    if strict:
        invalid = result.filter(pl.col("tod_bucket").is_null())
        if invalid.height:
            raise ValueError(
                "episode starts outside the frozen entry TOD buckets: "
                f"{invalid.select('start_seconds_from_open').head(5).to_dicts()}"
            )
    return result


def event_pooled_completed_quantile(
    episodes: pl.DataFrame,
    quantile: float,
) -> float | None:
    """Return the inverse empirical quantile pooling completed episodes."""

    sample = _normalise_episode_sample(episodes)
    completed = sample.filter(
        ~pl.col("left_censored") & pl.col("completed_center_return")
    )
    points = [
        (float(value), 1.0) for value in completed["observed_amplitude_bp"].to_list()
    ]
    return _weighted_inverse_quantile(points, quantile)


def date_equal_completed_quantile(
    episodes: pl.DataFrame,
    quantile: float,
) -> float | None:
    """Return a completed quantile giving each historical Date equal mass.

    Within a Date, each completed episode receives equal mass.  The total mass
    of every Date with at least one completed episode is therefore one.
    """

    sample = _normalise_episode_sample(episodes)
    completed = sample.filter(
        ~pl.col("left_censored") & pl.col("completed_center_return")
    )
    return _weighted_inverse_quantile(
        _date_equal_points(completed, upper_endpoints=False),
        quantile,
    )


def censor_identified_quantile_interval(
    episodes: pl.DataFrame,
    quantile: float,
    *,
    date_equal: bool = True,
) -> CensorIdentifiedInterval:
    """Identify an empirical amplitude quantile under right censoring.

    A completed episode is an exact amplitude.  A right-censored episode is
    the interval ``[observed_amplitude_bp, infinity)``.  Left-censored starts
    are not members of the primary sample.  For Date-equal estimation, every
    Date has mass one and episodes split that Date's mass equally.
    """

    sample = _normalise_episode_sample(episodes)
    primary = sample.filter(~pl.col("left_censored"))
    completed_count = primary.filter(pl.col("completed_center_return")).height
    right_count = primary.filter(pl.col("right_censored")).height
    if date_equal:
        lower_points = _date_equal_points(primary, upper_endpoints=False)
        upper_points = _date_equal_points(primary, upper_endpoints=True)
    else:
        lower_points = [
            (float(row["observed_amplitude_bp"]), 1.0)
            for row in primary.iter_rows(named=True)
        ]
        upper_points = [
            (
                float(row["observed_amplitude_bp"])
                if bool(row["completed_center_return"])
                else math.inf,
                1.0,
            )
            for row in primary.iter_rows(named=True)
        ]
    lower = _weighted_inverse_quantile(lower_points, quantile)
    upper_value = _weighted_inverse_quantile(upper_points, quantile)
    upper_unbounded = upper_value is not None and math.isinf(upper_value)
    upper = None if upper_unbounded else upper_value
    if lower is not None and upper is not None and lower > upper + 1e-12:
        raise ValueError("censor-identified interval endpoints are inconsistent")
    return CensorIdentifiedInterval(
        lower_bp=lower,
        upper_bp=upper,
        upper_unbounded=upper_unbounded,
        observable_started=primary.height,
        completed_count=completed_count,
        right_censored_count=right_count,
        history_dates=primary["Date"].n_unique() if primary.height else 0,
    )


def clip_completed_quantile_to_interval(
    completed_point_bp: float | None,
    interval: CensorIdentifiedInterval,
) -> float | None:
    """Clip a completed-only point to the censor-identified range."""

    if completed_point_bp is None:
        return None
    point = float(completed_point_bp)
    if not math.isfinite(point):
        raise ValueError("completed point must be finite")
    if interval.lower_bp is not None:
        point = max(point, float(interval.lower_bp))
    if interval.upper_bp is not None:
        point = min(point, float(interval.upper_bp))
    return point


def select_trailing_session_history(
    history: pl.DataFrame,
    *,
    target_date: str,
    lookback_sessions: int,
) -> SelectedHistory:
    """Select the last N supplied market sessions, requiring every row < D."""

    if lookback_sessions <= 0:
        raise ValueError("lookback_sessions must be positive")
    frame = _normalise_product_keys(history, "history")
    target = _parse_date_text(target_date, "target_date")
    _reject_nonprior_history(frame, target, "history")
    dates = sorted(frame["Date"].unique().to_list())[-lookback_sessions:]
    selected = frame.filter(pl.col("Date").is_in(dates))
    return _selected_history(selected)


def select_prior_expiry_dte_history(
    history: pl.DataFrame,
    *,
    target_date: str,
    value_code: str,
    target_expiry_date: date | str,
    target_calendar_dte: int,
    prior_expiry_cycles: int,
    dte_radius_calendar_days: int,
) -> SelectedHistory:
    """Select prior expiry cycles at calendar DTE within the frozen radius."""

    if prior_expiry_cycles <= 0:
        raise ValueError("prior_expiry_cycles must be positive")
    if dte_radius_calendar_days < 0:
        raise ValueError("dte_radius_calendar_days must be nonnegative")
    frame = _normalise_product_keys(history, "history")
    frame = _validated_expiry_annotations(frame, "history")
    target = _parse_date_text(target_date, "target_date")
    expiry = _coerce_date(target_expiry_date, "target_expiry_date")
    expected_dte = (expiry - target).days
    if expected_dte != int(target_calendar_dte):
        raise ValueError(
            "target_calendar_dte is inconsistent with target expiry and Date"
        )
    if expected_dte < 0:
        raise ValueError("target expiry is before target_date")
    _reject_nonprior_history(frame, target, "history")

    product = frame.filter(
        (pl.col("ValueCode") == str(value_code))
        & pl.col("expiry_date").is_not_null()
        & (pl.col("expiry_date") < pl.lit(expiry))
    )
    available_cycles = sorted(
        product["expiry_date"].unique().drop_nulls().to_list(),
        reverse=True,
    )
    cycles = available_cycles[:prior_expiry_cycles]
    selected = product.filter(
        pl.col("expiry_date").is_in(cycles)
        & (
            (pl.col("calendar_dte") - int(target_calendar_dte)).abs()
            <= dte_radius_calendar_days
        )
    )
    lineage = _selected_history(
        selected,
        selected_expiry_cycles=tuple(cycle.strftime("%Y%m%d") for cycle in cycles),
        selection_reason=(
            None
            if len(cycles) == prior_expiry_cycles
            else (
                f"insufficient_prior_expiry_cycles:{len(cycles)}<{prior_expiry_cycles}"
            )
        ),
    )
    current_cycle = expiry.strftime("%Y%m%d")
    if current_cycle in lineage.selected_expiry_cycles:
        raise ValueError("prior-expiry selector retained the target expiry cycle")
    return lineage


def build_boundary_candidate_predictions(
    history: pl.DataFrame,
    targets: pl.DataFrame,
    *,
    candidate_id: str,
    quantiles: Sequence[int] = PRIMARY_QUANTILES,
    minimum_completed_per_side: Mapping[int, int] = MINIMUM_COMPLETED_PER_SIDE,
    tod_buckets: Sequence[TodBucket] = DEFAULT_TOD_BUCKETS,
) -> pl.DataFrame:
    """Build one candidate's long, D-safe boundary predictions for one Date.

    The caller supplies history only through D-1.  This fail-closed contract is
    intentional: target-day rows are rejected rather than silently filtered.
    The output keeps unsupported raw diagnostics but exposes a non-null
    ``boundary_distance_bp`` only when the candidate passes its native minima.
    """

    spec = BOUNDARY_CANDIDATE_SPECS.get(candidate_id)
    if spec is None:
        if candidate_id.startswith(("Q3_", "Q4_", "Q7_")):
            raise NotImplementedError(
                f"{candidate_id} requires the later global-scale runner"
            )
        raise ValueError(f"unknown boundary candidate: {candidate_id}")
    expected_quantiles = _validated_quantiles(quantiles)
    minima = _validated_completed_minima(
        expected_quantiles,
        minimum_completed_per_side,
    )
    buckets = _validated_tod_buckets(tod_buckets)
    target = _normalise_targets(targets, buckets)
    target_dates = target["Date"].unique().to_list()
    if len(target_dates) != 1:
        raise ValueError("candidate builder accepts exactly one target Date")
    target_date = str(target_dates[0])

    _require_columns(
        history,
        {
            *PRODUCT_KEYS,
            "side",
            "observed_amplitude_bp",
            "left_censored",
            "right_censored",
            "completed_center_return",
        },
        "history",
    )
    normal_history = _normalise_product_keys(history, "history")
    _reject_nonprior_history(
        normal_history,
        _parse_date_text(target_date, "target Date"),
        "history",
    )
    normal_history = _normalise_episode_sample(normal_history)
    invalid_side = normal_history.filter(~pl.col("side").is_in(SIDES))
    if invalid_side.height:
        raise ValueError("history contains an invalid episode side")

    trailing: SelectedHistory | None = None
    if spec.lookback_sessions is not None:
        trailing = select_trailing_session_history(
            normal_history,
            target_date=target_date,
            lookback_sessions=spec.lookback_sessions,
        )
    elif spec.prior_expiry_cycles is not None:
        normal_history = _validated_expiry_annotations(normal_history, "history")
        target = _validated_expiry_annotations(target, "targets")
    else:
        raise AssertionError(f"candidate has no history selector: {candidate_id}")

    records: list[dict[str, object]] = []
    selected_trailing_by_product_side: dict[tuple[str, str], pl.DataFrame] = {}
    if trailing is not None:
        selected_trailing_by_product_side = {
            (str(key[0]), str(key[1])): group
            for key, group in trailing.frame.group_by(
                ["ValueCode", "side"], maintain_order=True
            )
        }
    history_by_product = {
        str(key[0]): group
        for key, group in normal_history.group_by("ValueCode", maintain_order=True)
    }
    product_targets = target.select(
        *PRODUCT_KEYS,
        "expiry_date",
        "calendar_dte",
    ).unique(maintain_order=True)
    for target_row in product_targets.iter_rows(named=True):
        value_code = str(target_row["ValueCode"])
        if trailing is not None:
            selected = trailing
            product_history = None
        else:
            expiry = target_row["expiry_date"]
            calendar_dte = target_row["calendar_dte"]
            if expiry is None or calendar_dte is None:
                selected = SelectedHistory(
                    frame=normal_history.head(0),
                    source_asof_date=None,
                    history_start_date=None,
                    history_end_date=None,
                    selected_sessions=0,
                    selection_reason="unknown_target_expiry_or_dte",
                )
            else:
                selected = select_prior_expiry_dte_history(
                    history_by_product.get(value_code, normal_history.head(0)),
                    target_date=target_date,
                    value_code=value_code,
                    target_expiry_date=expiry,
                    target_calendar_dte=int(calendar_dte),
                    prior_expiry_cycles=int(spec.prior_expiry_cycles or 0),
                    dte_radius_calendar_days=int(spec.dte_radius_calendar_days or 0),
                )
            product_history = selected.frame

        target_tods = target.filter(
            (pl.col("Date") == str(target_row["Date"]))
            & (pl.col("ValueCode") == value_code)
            & (pl.col("QuoteCode") == str(target_row["QuoteCode"]))
        )["tod_bucket"].to_list()
        for side in SIDES:
            side_sample = (
                selected_trailing_by_product_side.get(
                    (value_code, side),
                    normal_history.head(0),
                )
                if product_history is None
                else product_history.filter(pl.col("side") == side)
            )
            estimates = _estimate_quantile_cells(
                side_sample,
                quantiles=expected_quantiles,
                date_equal=spec.kind != "event_pooled_completed_control",
                already_normalised=True,
            )
            completed = side_sample.filter(
                ~pl.col("left_censored") & pl.col("completed_center_return")
            )
            completed_dates = completed["Date"].n_unique() if completed.height else 0
            for quantile in expected_quantiles:
                is_event_pooled = spec.kind == "event_pooled_completed_control"
                completed_point, interval = estimates[quantile]
                clipped = (
                    completed_point
                    if is_event_pooled
                    else clip_completed_quantile_to_interval(
                        completed_point,
                        interval,
                    )
                )
                reasons: list[str] = []
                if selected.selection_reason is not None:
                    reasons.append(selected.selection_reason)
                if completed_dates < spec.minimum_dates:
                    reasons.append(
                        "insufficient_completed_dates:"
                        f"{completed_dates}<{spec.minimum_dates}"
                    )
                minimum_completed = minima[quantile]
                if completed.height < minimum_completed:
                    reasons.append(
                        "insufficient_completed_count:"
                        f"{completed.height}<{minimum_completed}"
                    )
                if clipped is None or not math.isfinite(float(clipped)):
                    reasons.append("unavailable_finite_point")
                elif float(clipped) <= 0:
                    reasons.append("nonpositive_point")
                native_supported = not reasons
                for tod_bucket in target_tods:
                    records.append(
                        {
                            "Date": str(target_row["Date"]),
                            "ValueCode": value_code,
                            "QuoteCode": str(target_row["QuoteCode"]),
                            "tod_bucket": str(tod_bucket),
                            "boundary_quantile": quantile,
                            "side": side,
                            "candidate_id": spec.id,
                            "candidate_kind": spec.kind,
                            "source_asof_date": selected.source_asof_date,
                            "history_start_date": selected.history_start_date,
                            "history_end_date": selected.history_end_date,
                            "lookback_sessions": spec.lookback_sessions,
                            "prior_expiry_cycles": spec.prior_expiry_cycles,
                            "selected_expiry_cycles": (
                                ",".join(selected.selected_expiry_cycles)
                                if selected.selected_expiry_cycles
                                else None
                            ),
                            "target_expiry_date": target_row["expiry_date"],
                            "target_calendar_dte": target_row["calendar_dte"],
                            "minimum_dates": spec.minimum_dates,
                            "minimum_completed": minimum_completed,
                            "selection_sessions": selected.selected_sessions,
                            "completed_history_dates": completed_dates,
                            "observable_started": interval.observable_started,
                            "completed_count": interval.completed_count,
                            "right_censored_count": interval.right_censored_count,
                            "completed_point_bp": completed_point,
                            "identified_lower_bp": interval.lower_bp,
                            "identified_upper_bp": interval.upper_bp,
                            "identified_upper_unbounded": (interval.upper_unbounded),
                            "clipped_point_bp": clipped,
                            "boundary_distance_bp": (
                                clipped if native_supported else None
                            ),
                            "native_supported": native_supported,
                            "allow_primary_selection": spec.allow_primary_selection,
                            "fallback_candidate_id": spec.fallback_candidate,
                            "fallback_reason": (";".join(reasons) if reasons else None),
                            "contains_target_day_outcome": False,
                        }
                    )

    result = pl.from_dicts(
        records,
        schema=BOUNDARY_PREDICTION_SCHEMA,
        infer_schema_length=None,
    ).sort([*PREDICTION_KEYS, "candidate_id"])
    validate_strict_source_asof(result)
    validate_monotonic_quantiles(result, value_column="clipped_point_bp")
    validate_monotonic_quantiles(result, value_column="boundary_distance_bp")
    return result


def validate_strict_source_asof(predictions: pl.DataFrame) -> None:
    """Reject target-day/future lineage and supported rows without lineage."""

    _require_columns(
        predictions,
        {
            "Date",
            "source_asof_date",
            "history_end_date",
            "native_supported",
            "contains_target_day_outcome",
        },
        "predictions",
    )
    frame = _normalise_date_text_column(predictions, "Date", "predictions")
    source = _optional_normalised_date_expr("source_asof_date")
    history_end = _optional_normalised_date_expr("history_end_date")
    frame = frame.with_columns(
        source.alias("_source_asof"),
        history_end.alias("_history_end"),
    )
    invalid_text = frame.filter(
        (pl.col("source_asof_date").is_not_null() & pl.col("_source_asof").is_null())
        | (pl.col("history_end_date").is_not_null() & pl.col("_history_end").is_null())
    )
    if invalid_text.height:
        raise ValueError("predictions contain invalid lineage dates")
    invalid = frame.filter(
        pl.col("contains_target_day_outcome").fill_null(True)
        | (
            pl.col("native_supported").fill_null(False)
            & pl.col("_source_asof").is_null()
        )
        | (
            pl.col("_source_asof").is_not_null()
            & (pl.col("_source_asof") >= pl.col("Date"))
        )
        | (
            pl.col("_source_asof").is_not_null()
            & (pl.col("_source_asof") != pl.col("_history_end"))
        )
    )
    if invalid.height:
        raise ValueError("predictions violate strict source_asof_date < Date")


def validate_monotonic_quantiles(
    predictions: pl.DataFrame,
    *,
    value_column: str = "boundary_distance_bp",
) -> None:
    """Validate nondecreasing q distances within each prediction cell."""

    group_keys = [
        "Date",
        "ValueCode",
        "QuoteCode",
        "tod_bucket",
        "side",
        "candidate_id",
    ]
    _require_columns(
        predictions,
        {*group_keys, "boundary_quantile", value_column},
        "predictions",
    )
    duplicates = (
        predictions.group_by([*group_keys, "boundary_quantile"])
        .len()
        .filter(pl.col("len") != 1)
    )
    if duplicates.height:
        raise ValueError("predictions contain duplicate quantile keys")
    for key, group in predictions.group_by(group_keys):
        ordered = group.sort("boundary_quantile")
        quantiles = ordered["boundary_quantile"].cast(pl.Int64).to_list()
        if any(left >= right for left, right in pairwise(quantiles)):
            raise ValueError(f"quantiles are not strictly ordered for {key}")
        values = [
            float(value)
            for value in ordered[value_column].to_list()
            if value is not None
        ]
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError(f"{value_column} must be finite and positive")
        if any(left > right + 1e-12 for left, right in pairwise(values)):
            raise ValueError(f"{value_column} is not monotone for {key}")


def _normalise_targets(
    targets: pl.DataFrame,
    buckets: Sequence[TodBucket],
) -> pl.DataFrame:
    _require_columns(targets, set(PRODUCT_KEYS), "targets")
    result = _normalise_product_keys(targets, "targets")
    optional_expiry = {"expiry_date", "calendar_dte"}
    present_expiry = optional_expiry & set(result.columns)
    if present_expiry and present_expiry != optional_expiry:
        raise ValueError("targets must carry expiry_date and calendar_dte together")
    if not present_expiry:
        result = result.with_columns(
            pl.lit(None, dtype=pl.Date).alias("expiry_date"),
            pl.lit(None, dtype=pl.Int32).alias("calendar_dte"),
        )
    if "tod_bucket" not in result.columns:
        tod = pl.DataFrame({"tod_bucket": [bucket.id for bucket in buckets]})
        result = result.join(tod, how="cross")
    invalid_tod = result.filter(
        ~pl.col("tod_bucket").cast(pl.String).is_in([bucket.id for bucket in buckets])
    )
    if invalid_tod.height:
        raise ValueError("targets contain a TOD bucket outside the frozen registry")
    result = result.with_columns(pl.col("tod_bucket").cast(pl.String))
    duplicate = (
        result.group_by([*PRODUCT_KEYS, "tod_bucket"]).len().filter(pl.col("len") != 1)
    )
    if duplicate.height:
        raise ValueError("targets contain duplicate product-day TOD keys")
    return result.select(
        *PRODUCT_KEYS,
        "tod_bucket",
        "expiry_date",
        "calendar_dte",
    ).sort([*PRODUCT_KEYS, "tod_bucket"])


def _normalise_episode_sample(episodes: pl.DataFrame) -> pl.DataFrame:
    required = {
        "Date",
        "observed_amplitude_bp",
        "left_censored",
        "right_censored",
        "completed_center_return",
    }
    _require_columns(episodes, required, "episodes")
    sample = _normalise_date_text_column(episodes, "Date", "episodes").with_columns(
        pl.col("observed_amplitude_bp").cast(pl.Float64),
        pl.col("left_censored").fill_null(False).cast(pl.Boolean),
        pl.col("right_censored").fill_null(False).cast(pl.Boolean),
        pl.col("completed_center_return").fill_null(False).cast(pl.Boolean),
    )
    invalid = sample.filter(
        ~pl.col("observed_amplitude_bp").is_finite()
        | (pl.col("observed_amplitude_bp") < 0)
        | (pl.col("right_censored") == pl.col("completed_center_return"))
    )
    if invalid.height:
        raise ValueError("episodes violate amplitude or censor endpoint invariants")
    return sample


def _date_equal_points(
    frame: pl.DataFrame,
    *,
    upper_endpoints: bool,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for _, day in frame.group_by("Date", maintain_order=True):
        if day.is_empty():
            continue
        weight = 1.0 / day.height
        for row in day.iter_rows(named=True):
            value = float(row["observed_amplitude_bp"])
            if upper_endpoints and bool(row["right_censored"]):
                value = math.inf
            points.append((value, weight))
    return points


def _weighted_endpoint_quantiles(
    frame: pl.DataFrame,
    probabilities: Sequence[float],
    *,
    date_equal: bool,
    upper_endpoints: bool,
) -> dict[float, float | None]:
    """Compute several inverse empirical quantiles after one Rust-side sort."""

    requested = tuple(float(value) for value in probabilities)
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in requested):
        raise ValueError("quantile must be finite and between zero and one")
    if frame.is_empty():
        return dict.fromkeys(requested)
    endpoint = (
        pl.when(pl.col("right_censored"))
        .then(pl.lit(math.inf))
        .otherwise(pl.col("observed_amplitude_bp"))
        if upper_endpoints
        else pl.col("observed_amplitude_bp")
    )
    weight = (1.0 / pl.len().over("Date")) if date_equal else pl.lit(1.0)
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
        while index + 1 < len(cumulative) and cumulative[index] + 1e-12 < threshold:
            index += 1
        result[probability] = float(endpoints[index])
    return result


def _estimate_quantile_cells(
    episodes: pl.DataFrame,
    *,
    quantiles: Sequence[int],
    date_equal: bool,
    already_normalised: bool = False,
) -> dict[int, tuple[float | None, CensorIdentifiedInterval]]:
    """Estimate all registered q cells without rescanning a side for every q."""

    sample = episodes if already_normalised else _normalise_episode_sample(episodes)
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
    observable_started = primary.height
    completed_count = completed.height
    right_censored_count = primary.filter(pl.col("right_censored")).height
    history_dates = primary["Date"].n_unique() if primary.height else 0
    result: dict[int, tuple[float | None, CensorIdentifiedInterval]] = {}
    for quantile, probability in zip(quantiles, probabilities, strict=True):
        upper_value = upper_points[probability]
        upper_unbounded = upper_value is not None and math.isinf(upper_value)
        interval = CensorIdentifiedInterval(
            lower_bp=lower_points[probability],
            upper_bp=None if upper_unbounded else upper_value,
            upper_unbounded=upper_unbounded,
            observable_started=observable_started,
            completed_count=completed_count,
            right_censored_count=right_censored_count,
            history_dates=history_dates,
        )
        result[quantile] = (completed_points[probability], interval)
    return result


def _weighted_inverse_quantile(
    points: Sequence[tuple[float, float]],
    quantile: float,
) -> float | None:
    q = float(quantile)
    if not math.isfinite(q) or not 0 <= q <= 1:
        raise ValueError("quantile must be finite and between zero and one")
    if not points:
        return None
    ordered = sorted(points, key=lambda item: item[0])
    if any(weight <= 0 or not math.isfinite(weight) for _, weight in ordered):
        raise ValueError("quantile weights must be finite and positive")
    total = math.fsum(weight for _, weight in ordered)
    threshold = q * total
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative + 1e-12 >= threshold:
            return float(value)
    return float(ordered[-1][0])


def _selected_history(
    selected: pl.DataFrame,
    *,
    selected_expiry_cycles: tuple[str, ...] = (),
    selection_reason: str | None = None,
) -> SelectedHistory:
    dates = sorted(selected["Date"].unique().to_list()) if selected.height else []
    return SelectedHistory(
        frame=selected,
        source_asof_date=dates[-1] if dates else None,
        history_start_date=dates[0] if dates else None,
        history_end_date=dates[-1] if dates else None,
        selected_sessions=len(dates),
        selected_expiry_cycles=selected_expiry_cycles,
        selection_reason=selection_reason,
    )


def _validated_expiry_annotations(frame: pl.DataFrame, source: str) -> pl.DataFrame:
    _require_columns(
        frame,
        {"Date", "expiry_date", "calendar_dte"},
        source,
    )
    result = _normalise_date_text_column(frame, "Date", source)
    result = (
        _with_parsed_date(
            result,
            "expiry_date",
            output="_normal_expiry_date",
            source=f"{source} expiry_date",
            allow_null=True,
        )
        .drop("expiry_date")
        .rename({"_normal_expiry_date": "expiry_date"})
    )
    expected_dte = (
        (pl.col("expiry_date") - _date_text_expr("Date")).dt.total_days().cast(pl.Int32)
    )
    invalid = result.filter(
        (pl.col("expiry_date").is_null() != pl.col("calendar_dte").is_null())
        | (
            pl.col("expiry_date").is_not_null()
            & (pl.col("calendar_dte").cast(pl.Int32) != expected_dte)
        )
        | (pl.col("calendar_dte").cast(pl.Int32) < 0)
    )
    if invalid.height:
        raise ValueError(f"{source} has inconsistent expiry/DTE annotations")
    return result.with_columns(pl.col("calendar_dte").cast(pl.Int32))


def _normalise_product_keys(frame: pl.DataFrame, source: str) -> pl.DataFrame:
    _require_columns(frame, set(PRODUCT_KEYS), source)
    result = _normalise_date_text_column(frame, "Date", source).with_columns(
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
    )
    null_key = result.filter(
        pl.col("ValueCode").is_null() | pl.col("QuoteCode").is_null()
    )
    if null_key.height:
        raise ValueError(f"{source} contains null product keys")
    return result


def _normalise_date_text_column(
    frame: pl.DataFrame,
    column: str,
    source: str,
) -> pl.DataFrame:
    _require_columns(frame, {column}, source)
    parsed_name = f"__parsed_{column}"
    result = frame.with_columns(
        _optional_normalised_date_expr(column).alias(parsed_name)
    )
    invalid = result.filter(pl.col(parsed_name).is_null())
    if invalid.height:
        raise ValueError(f"{source} contains invalid or null {column}")
    return result.drop(column).rename({parsed_name: column})


def _with_parsed_date(
    frame: pl.DataFrame,
    column: str,
    *,
    output: str,
    source: str,
    allow_null: bool,
) -> pl.DataFrame:
    _require_columns(frame, {column}, source)
    text = pl.col(column).cast(pl.String)
    parsed = pl.coalesce(
        text.str.strptime(pl.Date, "%Y%m%d", strict=False),
        text.str.strptime(pl.Date, "%Y-%m-%d", strict=False),
    )
    result = frame.with_columns(parsed.alias(output))
    invalid_expr = pl.col(column).is_not_null() & pl.col(output).is_null()
    if not allow_null:
        invalid_expr = invalid_expr | pl.col(column).is_null()
    if result.filter(invalid_expr).height:
        raise ValueError(f"{source} contains invalid dates")
    return result


def _optional_normalised_date_expr(column: str) -> pl.Expr:
    text = pl.col(column).cast(pl.String)
    return pl.coalesce(
        text.str.strptime(pl.Date, "%Y%m%d", strict=False),
        text.str.strptime(pl.Date, "%Y-%m-%d", strict=False),
    ).dt.strftime("%Y%m%d")


def _date_text_expr(column: str) -> pl.Expr:
    return pl.col(column).str.strptime(pl.Date, "%Y%m%d", strict=True)


def _parse_date_text(value: str, source: str) -> date:
    try:
        return datetime.strptime(str(value), "%Y%m%d").date()  # noqa: DTZ007
    except ValueError as error:
        raise ValueError(f"{source} must be YYYYMMDD: {value!r}") from error


def _coerce_date(value: date | str, source: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value)
    for format_text in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, format_text).date()  # noqa: DTZ007
        except ValueError:
            continue
    raise ValueError(f"{source} must be a date or YYYYMMDD text")


def _reject_nonprior_history(
    history: pl.DataFrame,
    target_date: date,
    source: str,
) -> None:
    target_text = target_date.strftime("%Y%m%d")
    invalid = history.filter(pl.col("Date") >= target_text)
    if invalid.height:
        raise ValueError(
            f"{source} contains target-day/future rows; every Date must be < D"
        )


def _validated_tod_buckets(
    buckets: Sequence[TodBucket],
) -> tuple[TodBucket, ...]:
    result = tuple(sorted(buckets, key=lambda bucket: bucket.start_second))
    if not result:
        raise ValueError("TOD buckets must not be empty")
    if len({bucket.id for bucket in result}) != len(result):
        raise ValueError("TOD bucket ids must be unique")
    if any(bucket.start_second >= bucket.end_second for bucket in result):
        raise ValueError("TOD buckets must be nonempty half-open intervals")
    if any(left.end_second > right.start_second for left, right in pairwise(result)):
        raise ValueError("TOD buckets must not overlap")
    return result


def _validated_quantiles(quantiles: Sequence[int]) -> tuple[int, ...]:
    result = tuple(sorted({int(value) for value in quantiles}))
    if not result or any(value <= 0 or value >= 100 for value in result):
        raise ValueError("boundary quantiles must be unique integers in (0, 100)")
    if len(result) != len(quantiles):
        raise ValueError("boundary quantiles must not contain duplicates")
    return result


def _validated_completed_minima(
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


def _require_columns(
    frame: pl.DataFrame,
    required: set[str],
    source: str,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


__all__ = [
    "BOUNDARY_CANDIDATE_SPECS",
    "BOUNDARY_PREDICTION_SCHEMA",
    "DEFAULT_TOD_BUCKETS",
    "MINIMUM_COMPLETED_PER_SIDE",
    "PREDICTION_KEYS",
    "PRIMARY_QUANTILES",
    "BoundaryCandidateSpec",
    "CensorIdentifiedInterval",
    "SelectedHistory",
    "TodBucket",
    "annotate_expiry_dte",
    "assign_episode_start_tod_bucket",
    "build_boundary_candidate_predictions",
    "censor_identified_quantile_interval",
    "clip_completed_quantile_to_interval",
    "date_equal_completed_quantile",
    "event_pooled_completed_quantile",
    "extract_entry_window_excursions",
    "select_prior_expiry_dte_history",
    "select_trailing_session_history",
    "validate_monotonic_quantiles",
    "validate_strict_source_asof",
]
