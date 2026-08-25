"""Pure S0.5 boundary-validation and broad-cohort contracts.

This module deliberately has no filesystem runner.  It turns one published
one-second causal fair panel into signed, censor-aware residual episodes,
scores frozen boundary rows without converting unknown outcomes into misses,
and derives a full-60-session research cohort without accepting any monthly
selector fields.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise

import polars as pl

EPISODE_SCHEMA = pl.Schema(
    {
        "episode_id": pl.String,
        "Date": pl.String,
        "ValueCode": pl.String,
        "QuoteCode": pl.String,
        "side": pl.String,
        "episode_sequence": pl.Int64,
        "start_timestamp": pl.Datetime("ns"),
        "end_timestamp": pl.Datetime("ns"),
        "start_seconds_from_open": pl.Int64,
        "end_seconds_from_open": pl.Int64,
        "duration_seconds": pl.Int64,
        "left_censored": pl.Boolean,
        "left_censor_reason": pl.String,
        "right_censored": pl.Boolean,
        "right_censor_reason": pl.String,
        "completed_center_return": pl.Boolean,
        "fully_observed": pl.Boolean,
        "start_previous_residual_bp": pl.Float64,
        "start_residual_bp": pl.Float64,
        "end_residual_bp": pl.Float64,
        "observed_amplitude_bp": pl.Float64,
        "anchor_column": pl.String,
        "sampling_interval": pl.String,
    }
)


CENSOR_BOUND_SCHEMA = pl.Schema(
    {
        "Date": pl.String,
        "ValueCode": pl.String,
        "QuoteCode": pl.String,
        "boundary_quantile": pl.Int64,
        "side": pl.String,
        "boundary_distance_bp": pl.Float64,
        "predicted_distance_bp": pl.Float64,
        "realized_completed_quantile_bp": pl.Float64,
        "source_asof_date": pl.String,
        "observable_started": pl.Int64,
        "n_started": pl.Int64,
        "completed_count": pl.Int64,
        "right_censored_count": pl.Int64,
        "confirmed_hits": pl.Int64,
        "n_hit": pl.Int64,
        "completed_hits": pl.Int64,
        "known_nonhits": pl.Int64,
        "n_known_miss": pl.Int64,
        "unknown_nonhits": pl.Int64,
        "n_unknown_censored": pl.Int64,
        "left_censored_count": pl.Int64,
        "left_censored_confirmed_hits": pl.Int64,
        "reach_lower_bound": pl.Float64,
        "reach_upper_bound": pl.Float64,
        "complete_case_reach": pl.Float64,
        "outcome_status": pl.String,
    }
)


ALL_BOUNDARY_SUPPORTED_SCHEMA = pl.Schema(
    {
        "Date": pl.String,
        "ValueCode": pl.String,
        "QuoteCode": pl.String,
        "source_asof_date": pl.String,
        "train_start_date": pl.String,
        "train_end_date": pl.String,
        "parameter_version": pl.String,
        "history_sessions_global": pl.Int64,
        "upper_distance_bp_50": pl.Float64,
        "upper_distance_bp_80": pl.Float64,
        "upper_distance_bp_95": pl.Float64,
        "lower_distance_bp_50": pl.Float64,
        "lower_distance_bp_80": pl.Float64,
        "lower_distance_bp_95": pl.Float64,
        "all_boundary_supported": pl.Boolean,
        "contains_target_day_outcome": pl.Boolean,
    }
)


SPOT_BID_BROAD_SCHEMA = pl.Schema(
    {
        **dict(ALL_BOUNDARY_SUPPORTED_SCHEMA.items()),
        "route": pl.String,
        "support_gate": pl.Boolean,
        "hard_data_gate": pl.Boolean,
        "pre_replay_candidate": pl.Boolean,
        "liquidity_gate_status": pl.String,
        "liquidity_source_asof_date": pl.String,
        "spot_bid_broad_candidate": pl.Boolean,
    }
)


PRIMARY_QUANTILES = (50, 80, 95)
SPOT_BID_ROUTE = "spot_bid_future_taker"
BOUNDARY_KEYS = ("Date", "ValueCode", "QuoteCode")

_FORBIDDEN_SELECTOR_COLUMNS = {
    "effective_month",
    "new_entry_allowed",
    "source_month",
    "source_month_last_date",
}
_FORBIDDEN_SELECTOR_PREFIXES = (
    "monthly_",
    "selector_",
    "prior_",
)

_LIQUIDITY_GATE_COLUMNS = (
    "support_gate",
    "hard_data_gate",
    "pre_replay_candidate",
    "liquidity_gate_status",
)
_OPTIONAL_Q_INVARIANT_LIQUIDITY_COLUMNS = (
    "long_history_gate",
    "boundary_parameter_gate",
    "recent_history_gate",
    "fresh_1000ms_gate",
    "fresh_5000ms_gate",
    "depth_gate",
    "maker_activity_gate",
    "tt_band_data_gate",
    "liquidity_train_start_date",
    "liquidity_train_end_date",
    "liquidity_screen_version",
)


@dataclass(frozen=True)
class FoundationCohorts:
    """The full boundary-supported panel and its Spot-Bid replay subset."""

    all_boundary_supported: pl.DataFrame
    spot_bid_broad: pl.DataFrame

    @property
    def spot_bid_broad_candidate(self) -> pl.DataFrame:
        """Runner-facing alias for the q-independent broad candidate rows."""

        return self.spot_bid_broad


def extract_censor_aware_excursions(
    causal_fair: pl.DataFrame,
    *,
    anchor_column: str = "anchor_ewma_120s_bp",
) -> pl.DataFrame:
    """Extract signed one-second EWMA residual episodes for one session.

    An episode that is already away from zero at the first eligible row of a
    session or immediately after a gap is left-censored.  An active episode at
    an eligibility/timestamp gap or session cutoff is right-censored.  Direct
    positive-to-negative (or negative-to-positive) crossings close one side
    and start the other on the same one-second row.
    """

    required = {
        "Date",
        "ValueCode",
        "QuoteCode",
        "timestamp",
        "seconds_from_open",
        "basis_mid_bp",
        "analysis_eligible",
        anchor_column,
    }
    _require_columns(causal_fair, required, "causal fair panel")
    if causal_fair.is_empty():
        return pl.DataFrame(schema=EPISODE_SCHEMA)

    panel = causal_fair.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("timestamp").cast(pl.Datetime("ns")),
        pl.col("seconds_from_open").cast(pl.Int64),
        pl.col("basis_mid_bp").cast(pl.Float64),
        pl.col(anchor_column).cast(pl.Float64),
        pl.col("analysis_eligible").fill_null(False).cast(pl.Boolean),
    ).with_columns(
        (pl.col("basis_mid_bp") - pl.col(anchor_column)).alias("_residual_bp")
    )
    _validate_one_day_panel(panel)
    panel = panel.sort(
        ["Date", "ValueCode", "QuoteCode", "seconds_from_open", "timestamp"]
    )

    records: list[dict[str, object]] = []
    for key, group in panel.group_by(list(BOUNDARY_KEYS), maintain_order=True):
        date, value_code, quote_code = map(str, key)
        rows = group.iter_rows(named=True)
        sequence = 0
        seen_eligible = False
        next_left_reason = "session_start"
        previous_valid: dict[str, object] | None = None
        last_valid: dict[str, object] | None = None
        active: dict[str, object] | None = None
        previous_input_second: int | None = None
        previous_input_timestamp: datetime | None = None

        def begin(
            row: Mapping[str, object],
            side: str,
            *,
            left_censored: bool,
            left_censor_reason: str | None,
            previous_residual: float | None,
        ) -> dict[str, object]:
            nonlocal sequence
            sequence += 1
            residual = float(row["_residual_bp"])
            amplitude = residual if side == "positive" else -residual
            return {
                "episode_sequence": sequence,
                "side": side,
                "start_timestamp": row["timestamp"],
                "start_seconds_from_open": int(row["seconds_from_open"]),
                "start_previous_residual_bp": previous_residual,
                "start_residual_bp": residual,
                "observed_amplitude_bp": amplitude,
                "left_censored": left_censored,
                "left_censor_reason": left_censor_reason,
            }

        def finish(
            episode: dict[str, object],
            row: Mapping[str, object],
            *,
            completed: bool,
            reason: str,
            date_value: str = date,
            value_code_value: str = value_code,
            quote_code_value: str = quote_code,
        ) -> None:
            end_second = int(row["seconds_from_open"])
            start_second = int(episode["start_seconds_from_open"])
            right_censored = not completed
            episode.update(
                {
                    "episode_id": (
                        f"{date_value}/{value_code_value}/{quote_code_value}/"
                        f"{int(episode['episode_sequence']):06d}/"
                        f"{episode['side']}"
                    ),
                    "Date": date_value,
                    "ValueCode": value_code_value,
                    "QuoteCode": quote_code_value,
                    "end_timestamp": row["timestamp"],
                    "end_seconds_from_open": end_second,
                    "duration_seconds": end_second - start_second,
                    "right_censored": right_censored,
                    "right_censor_reason": reason if right_censored else None,
                    "completed_center_return": completed,
                    "fully_observed": (
                        completed and not bool(episode["left_censored"])
                    ),
                    "end_residual_bp": float(row["_residual_bp"]),
                    "anchor_column": anchor_column,
                    "sampling_interval": "1s",
                }
            )
            records.append(episode)

        for row in rows:
            current_second = int(row["seconds_from_open"])
            current_timestamp = row["timestamp"]
            if not isinstance(current_timestamp, datetime):
                raise TypeError("timestamp must be a non-null datetime")
            timestamp_gap = (
                previous_input_second is not None
                and current_second - previous_input_second != 1
            )
            if (
                previous_input_timestamp is not None
                and current_timestamp <= previous_input_timestamp
            ):
                raise ValueError("timestamps must increase within a product-day")
            if timestamp_gap:
                if active is not None and last_valid is not None:
                    finish(
                        active,
                        last_valid,
                        completed=False,
                        reason="timestamp_gap",
                    )
                    active = None
                previous_valid = None
                last_valid = None
                if seen_eligible:
                    next_left_reason = "timestamp_gap"

            residual_value = row["_residual_bp"]
            eligible = bool(row["analysis_eligible"]) and _finite(
                residual_value
            )
            if not eligible:
                if active is not None and last_valid is not None:
                    finish(
                        active,
                        last_valid,
                        completed=False,
                        reason="eligibility_gap",
                    )
                    active = None
                previous_valid = None
                last_valid = None
                if seen_eligible:
                    next_left_reason = "eligibility_gap"
                previous_input_second = current_second
                previous_input_timestamp = current_timestamp
                continue

            residual = float(residual_value)
            if previous_valid is None:
                if residual > 0:
                    active = begin(
                        row,
                        "positive",
                        left_censored=True,
                        left_censor_reason=next_left_reason,
                        previous_residual=None,
                    )
                elif residual < 0:
                    active = begin(
                        row,
                        "negative",
                        left_censored=True,
                        left_censor_reason=next_left_reason,
                        previous_residual=None,
                    )
                seen_eligible = True
                next_left_reason = "eligibility_gap"
                previous_valid = row
                last_valid = row
                previous_input_second = current_second
                previous_input_timestamp = current_timestamp
                continue

            previous_residual = float(previous_valid["_residual_bp"])
            if active is not None:
                side = str(active["side"])
                if side == "positive" and residual > 0:
                    active["observed_amplitude_bp"] = max(
                        float(active["observed_amplitude_bp"]), residual
                    )
                elif side == "negative" and residual < 0:
                    active["observed_amplitude_bp"] = max(
                        float(active["observed_amplitude_bp"]), -residual
                    )

                completed_positive = side == "positive" and residual <= 0
                completed_negative = side == "negative" and residual >= 0
                if completed_positive or completed_negative:
                    finish(
                        active,
                        row,
                        completed=True,
                        reason="center_return",
                    )
                    active = None
                    if residual < 0:
                        active = begin(
                            row,
                            "negative",
                            left_censored=False,
                            left_censor_reason=None,
                            previous_residual=previous_residual,
                        )
                    elif residual > 0:
                        active = begin(
                            row,
                            "positive",
                            left_censored=False,
                            left_censor_reason=None,
                            previous_residual=previous_residual,
                        )
            else:
                if previous_residual <= 0 < residual:
                    active = begin(
                        row,
                        "positive",
                        left_censored=False,
                        left_censor_reason=None,
                        previous_residual=previous_residual,
                    )
                elif previous_residual >= 0 > residual:
                    active = begin(
                        row,
                        "negative",
                        left_censored=False,
                        left_censor_reason=None,
                        previous_residual=previous_residual,
                    )

            previous_valid = row
            last_valid = row
            previous_input_second = current_second
            previous_input_timestamp = current_timestamp

        if active is not None and last_valid is not None:
            finish(
                active,
                last_valid,
                completed=False,
                reason="session_cutoff",
            )

    if not records:
        return pl.DataFrame(schema=EPISODE_SCHEMA)
    result = pl.from_dicts(
        records,
        schema=EPISODE_SCHEMA,
        infer_schema_length=None,
    ).sort(["Date", "ValueCode", "QuoteCode", "episode_sequence"])
    _validate_episode_overlay(result)
    return result


def build_product_day_censor_bounds(
    frozen_boundaries: pl.DataFrame,
    episodes: pl.DataFrame,
    *,
    expected_quantiles: Sequence[int] = PRIMARY_QUANTILES,
) -> pl.DataFrame:
    """Score frozen product-day q rows with identified censor bounds.

    Confirmed hits include completed and right-censored episodes whose observed
    amplitude reached the boundary.  A right-censored episode that stopped
    below the boundary is unknown, not a miss.  Left-censored episodes are
    reported separately and never enter the primary denominator.
    """

    boundaries = _validated_frozen_boundaries(
        frozen_boundaries,
        expected_quantiles=expected_quantiles,
    )
    _validate_episode_overlay(episodes)
    _validate_episode_contract_match(boundaries, episodes)

    index_columns = [
        *BOUNDARY_KEYS,
        "boundary_quantile",
        "source_asof_date",
    ]
    positive = boundaries.select(
        *index_columns,
        pl.lit("positive").alias("side"),
        pl.col("upper_distance_bp").alias("boundary_distance_bp"),
    )
    negative = boundaries.select(
        *index_columns,
        pl.lit("negative").alias("side"),
        pl.col("lower_distance_bp").alias("boundary_distance_bp"),
    )
    predictions = pl.concat([positive, negative], how="vertical")
    episode_columns = [
        "episode_id",
        *BOUNDARY_KEYS,
        "side",
        "left_censored",
        "right_censored",
        "completed_center_return",
        "observed_amplitude_bp",
    ]
    joined = predictions.join(
        episodes.select(episode_columns),
        on=[*BOUNDARY_KEYS, "side"],
        how="left",
        # Each episode is intentionally evaluated against q50/q80/q95, so
        # both sides of this join repeat the product-day/side key.
        validate="m:m",
    ).with_columns(
        pl.col("episode_id").is_not_null().alias("_matched"),
    ).with_columns(
        (pl.col("_matched") & ~pl.col("left_censored").fill_null(False)).alias(
            "_primary"
        ),
        (pl.col("_matched") & pl.col("left_censored").fill_null(False)).alias(
            "_left"
        ),
        (
            pl.col("observed_amplitude_bp")
            >= pl.col("boundary_distance_bp")
        ).fill_null(False).alias("_hit"),
    ).with_columns(
        (pl.col("_primary") & pl.col("_hit")).alias("_confirmed_hit"),
        (
            pl.col("_primary")
            & pl.col("completed_center_return").fill_null(False)
            & pl.col("_hit")
        ).alias("_completed_hit"),
        (
            pl.col("_primary")
            & pl.col("completed_center_return").fill_null(False)
            & ~pl.col("_hit")
        ).alias("_known_nonhit"),
        (
            pl.col("_primary")
            & pl.col("right_censored").fill_null(False)
            & ~pl.col("_hit")
        ).alias("_unknown_nonhit"),
        (pl.col("_left") & pl.col("_hit")).alias("_left_hit"),
    )

    group_keys = [
        *BOUNDARY_KEYS,
        "boundary_quantile",
        "side",
        "boundary_distance_bp",
        "source_asof_date",
    ]
    grouped_parts: list[pl.DataFrame] = []
    for quantile in sorted({int(value) for value in expected_quantiles}):
        grouped_parts.append(
            joined.filter(pl.col("boundary_quantile") == quantile)
            .group_by(group_keys, maintain_order=True)
            .agg(
                pl.col("_primary")
                .sum()
                .cast(pl.Int64)
                .alias("observable_started"),
                (
                    pl.col("_primary")
                    & pl.col("completed_center_return").fill_null(False)
                ).sum().cast(pl.Int64).alias("completed_count"),
                (
                    pl.col("_primary")
                    & pl.col("right_censored").fill_null(False)
                ).sum().cast(pl.Int64).alias("right_censored_count"),
                pl.col("_confirmed_hit")
                .sum()
                .cast(pl.Int64)
                .alias("confirmed_hits"),
                pl.col("_completed_hit")
                .sum()
                .cast(pl.Int64)
                .alias("completed_hits"),
                pl.col("_known_nonhit")
                .sum()
                .cast(pl.Int64)
                .alias("known_nonhits"),
                pl.col("_unknown_nonhit")
                .sum()
                .cast(pl.Int64)
                .alias("unknown_nonhits"),
                pl.col("_left")
                .sum()
                .cast(pl.Int64)
                .alias("left_censored_count"),
                pl.col("_left_hit")
                .sum()
                .cast(pl.Int64)
                .alias("left_censored_confirmed_hits"),
                pl.col("observed_amplitude_bp")
                .filter(
                    pl.col("_primary")
                    & pl.col("completed_center_return").fill_null(False)
                )
                .quantile(quantile / 100.0, interpolation="nearest")
                .cast(pl.Float64)
                .alias("realized_completed_quantile_bp"),
            )
        )
    grouped = pl.concat(grouped_parts, how="vertical").with_columns(
        pl.col("boundary_distance_bp").alias("predicted_distance_bp"),
        pl.col("observable_started").alias("n_started"),
        pl.col("confirmed_hits").alias("n_hit"),
        pl.col("known_nonhits").alias("n_known_miss"),
        pl.col("unknown_nonhits").alias("n_unknown_censored"),
        pl.when(pl.col("observable_started") > 0)
        .then(pl.col("confirmed_hits") / pl.col("observable_started"))
        .otherwise(None)
        .cast(pl.Float64)
        .alias("reach_lower_bound"),
        pl.when(pl.col("observable_started") > 0)
        .then(
            (pl.col("confirmed_hits") + pl.col("unknown_nonhits"))
            / pl.col("observable_started")
        )
        .otherwise(None)
        .cast(pl.Float64)
        .alias("reach_upper_bound"),
        pl.when(pl.col("completed_count") > 0)
        .then(pl.col("completed_hits") / pl.col("completed_count"))
        .otherwise(None)
        .cast(pl.Float64)
        .alias("complete_case_reach"),
        pl.when(pl.col("observable_started") > 0)
        .then(pl.lit("observable"))
        .otherwise(pl.lit("no_observable_excursion"))
        .alias("outcome_status"),
    ).select(CENSOR_BOUND_SCHEMA.names())

    _validate_censor_bound_rows(grouped)
    return grouped.sort(
        ["Date", "ValueCode", "QuoteCode", "boundary_quantile", "side"]
    )


def calibrate_boundary_product_days(
    boundaries_day: pl.DataFrame,
    episodes_day: pl.DataFrame,
    supported_keys_day: pl.DataFrame | None = None,
    *,
    expected_quantiles: Sequence[int] = PRIMARY_QUANTILES,
) -> pl.DataFrame:
    """Runner-facing one-day boundary calibration API.

    ``supported_keys_day`` is normally the one-day slice of
    :attr:`FoundationCohorts.all_boundary_supported`.  Only its explicit
    product keys are consumed.  Passing it cannot introduce selector fields
    or silently turn an unsupported product into an outcome row.
    """

    _require_columns(boundaries_day, set(BOUNDARY_KEYS), "one-day boundaries")
    boundary_dates = boundaries_day["Date"].cast(pl.String).unique().to_list()
    if len(boundary_dates) != 1:
        raise ValueError("boundaries_day must contain exactly one Date")
    target_date = str(boundary_dates[0])
    if not episodes_day.is_empty():
        _require_columns(episodes_day, {"Date"}, "one-day episodes")
        episode_dates = episodes_day["Date"].cast(pl.String).unique().to_list()
        if episode_dates != [target_date]:
            raise ValueError("episodes_day Date does not match boundaries_day")

    selected = boundaries_day
    if supported_keys_day is not None:
        _reject_selector_columns(supported_keys_day, "supported boundary keys")
        _require_columns(
            supported_keys_day,
            set(BOUNDARY_KEYS),
            "supported boundary keys",
        )
        keys = supported_keys_day.select(BOUNDARY_KEYS).unique()
        if keys.is_empty():
            raise ValueError("supported boundary keys must not be empty")
        if "all_boundary_supported" in supported_keys_day.columns and not (
            supported_keys_day["all_boundary_supported"].fill_null(False).all()
        ):
            raise ValueError("supported boundary keys contain unsupported rows")
        key_dates = keys["Date"].cast(pl.String).unique().to_list()
        if key_dates != [target_date]:
            raise ValueError("supported boundary keys Date does not match boundaries_day")
        _validate_unique_rows(keys, BOUNDARY_KEYS, "supported boundary keys")
        selected = boundaries_day.join(
            keys,
            on=list(BOUNDARY_KEYS),
            how="inner",
            validate="m:1",
        )
        missing = keys.join(
            selected.select(BOUNDARY_KEYS).unique(),
            on=list(BOUNDARY_KEYS),
            how="anti",
        )
        if missing.height:
            raise ValueError(
                "supported keys are absent from frozen boundaries: "
                f"{missing.head(5).to_dicts()}"
            )
    return build_product_day_censor_bounds(
        selected,
        episodes_day,
        expected_quantiles=expected_quantiles,
    )


def build_foundation_cohorts(
    rolling_boundaries: pl.DataFrame,
    liquidity: pl.DataFrame,
    primary_start: str | None = None,
    source_end: str | None = None,
    *,
    expected_quantiles: Sequence[int] = PRIMARY_QUANTILES,
    spot_bid_route: str = SPOT_BID_ROUTE,
) -> FoundationCohorts:
    """Build full-60 supported and q-independent Spot-Bid broad cohorts.

    Only rolling-boundary and route-liquidity facts are accepted.  Selector
    columns fail closed.  q50/q80 liquidity rows must carry identical route
    gates, and the broad admission decision uses only those common gates.
    """

    _reject_selector_columns(rolling_boundaries, "rolling boundaries")
    _reject_selector_columns(liquidity, "liquidity")
    required_boundary = {
        *BOUNDARY_KEYS,
        "boundary_quantile",
        "upper_distance_bp",
        "lower_distance_bp",
        "adaptive_parameter_valid",
        "history_sessions_global",
        "source_asof_date",
        "execution_safe_snapshot",
        "contains_target_day_outcome",
    }
    _require_columns(
        rolling_boundaries,
        required_boundary,
        "rolling boundaries",
    )
    expected = tuple(sorted({int(value) for value in expected_quantiles}))
    if expected != PRIMARY_QUANTILES:
        raise ValueError(f"foundation cohort requires q50/q80/q95: {expected}")

    date_filter = pl.lit(True)
    if primary_start is not None:
        _validate_date_text(primary_start, "primary_start")
        date_filter = date_filter & (pl.col("Date").cast(pl.String) >= primary_start)
    if source_end is not None:
        _validate_date_text(source_end, "source_end")
        date_filter = date_filter & (pl.col("Date").cast(pl.String) <= source_end)
    if (
        primary_start is not None
        and source_end is not None
        and primary_start > source_end
    ):
        raise ValueError("primary_start must not be later than source_end")

    boundaries = rolling_boundaries.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("boundary_quantile").cast(pl.Int64),
        pl.col("history_sessions_global").cast(pl.Int64),
        pl.col("source_asof_date").cast(pl.String),
        pl.col("upper_distance_bp").cast(pl.Float64),
        pl.col("lower_distance_bp").cast(pl.Float64),
    ).filter(date_filter & (pl.col("history_sessions_global") == 60))
    if boundaries.is_empty():
        raise ValueError("rolling boundaries contain no full-60 target rows")
    _validate_unique_rows(
        boundaries,
        [*BOUNDARY_KEYS, "boundary_quantile"],
        "full-60 rolling boundaries",
    )
    _validate_safe_prediction_rows(boundaries, "full-60 rolling boundaries")

    boundary_records: list[dict[str, object]] = []
    full_boundary_keys: set[tuple[str, str, str]] = set()
    for key, group in boundaries.sort(
        [*BOUNDARY_KEYS, "boundary_quantile"]
    ).group_by(list(BOUNDARY_KEYS), maintain_order=True):
        normal_key = tuple(map(str, key))
        full_boundary_keys.add(normal_key)
        rows = {
            int(row["boundary_quantile"]): row
            for row in group.iter_rows(named=True)
        }
        if tuple(sorted(rows)) != expected:
            raise ValueError(
                f"full-60 boundary key lacks q50/q80/q95: {normal_key}"
            )
        _assert_constant(rows.values(), "source_asof_date", normal_key)
        for optional in ("train_start_date", "train_end_date", "parameter_version"):
            if optional in group.columns:
                _assert_constant(rows.values(), optional, normal_key)

        upper_values = [rows[q]["upper_distance_bp"] for q in expected]
        lower_values = [rows[q]["lower_distance_bp"] for q in expected]
        finite = all(
            _finite_positive(value) for value in [*upper_values, *lower_values]
        )
        validity = [bool(rows[q]["adaptive_parameter_valid"]) for q in expected]
        if not finite:
            if any(validity):
                raise ValueError(
                    f"valid boundary row is null/non-positive for {normal_key}"
                )
            continue
        uppers = [float(value) for value in upper_values]
        lowers = [float(value) for value in lower_values]
        if not (_nondecreasing(uppers) and _nondecreasing(lowers)):
            raise ValueError(f"boundary quantiles are not monotone: {normal_key}")
        supported = all(validity)
        if not supported:
            continue
        first = rows[50]
        boundary_records.append(
            {
                "Date": normal_key[0],
                "ValueCode": normal_key[1],
                "QuoteCode": normal_key[2],
                "source_asof_date": str(first["source_asof_date"]),
                "train_start_date": _optional_text(first, "train_start_date"),
                "train_end_date": _optional_text(first, "train_end_date"),
                "parameter_version": _optional_text(first, "parameter_version"),
                "history_sessions_global": 60,
                "upper_distance_bp_50": uppers[0],
                "upper_distance_bp_80": uppers[1],
                "upper_distance_bp_95": uppers[2],
                "lower_distance_bp_50": lowers[0],
                "lower_distance_bp_80": lowers[1],
                "lower_distance_bp_95": lowers[2],
                "all_boundary_supported": True,
                "contains_target_day_outcome": False,
            }
        )
    if not boundary_records:
        raise ValueError("no full-60 product-day has all q boundaries supported")
    all_supported = pl.from_dicts(
        boundary_records,
        schema=ALL_BOUNDARY_SUPPORTED_SCHEMA,
        infer_schema_length=None,
    ).sort(list(BOUNDARY_KEYS))

    required_liquidity = {
        *BOUNDARY_KEYS,
        "boundary_quantile",
        "route",
        "source_asof_date",
        "execution_safe_snapshot",
        "contains_target_day_outcome",
        *_LIQUIDITY_GATE_COLUMNS,
    }
    _require_columns(liquidity, required_liquidity, "liquidity")
    full_dates = sorted(boundaries["Date"].unique().to_list())
    liquid = liquidity.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("boundary_quantile").cast(pl.Int64),
        pl.col("source_asof_date").cast(pl.String),
    ).filter(
        (pl.col("route") == spot_bid_route)
        & pl.col("boundary_quantile").is_in([50, 80])
        & pl.col("Date").is_in(full_dates)
    )
    if liquid.is_empty():
        raise ValueError("liquidity contains no full-60 Spot-Bid q50/q80 rows")
    _validate_unique_rows(
        liquid,
        [*BOUNDARY_KEYS, "boundary_quantile", "route"],
        "Spot-Bid liquidity",
    )
    _validate_safe_prediction_rows(liquid, "Spot-Bid liquidity")

    gate_records: list[dict[str, object]] = []
    for key, group in liquid.sort(
        [*BOUNDARY_KEYS, "boundary_quantile"]
    ).group_by(list(BOUNDARY_KEYS), maintain_order=True):
        normal_key = tuple(map(str, key))
        if normal_key not in full_boundary_keys:
            raise ValueError(
                f"liquidity key is absent from full-60 boundaries: {normal_key}"
            )
        rows = {
            int(row["boundary_quantile"]): row
            for row in group.iter_rows(named=True)
        }
        if tuple(sorted(rows)) != (50, 80):
            raise ValueError(f"liquidity key lacks q50/q80 rows: {normal_key}")
        invariant_columns = [
            *_LIQUIDITY_GATE_COLUMNS,
            *[
                column
                for column in _OPTIONAL_Q_INVARIANT_LIQUIDITY_COLUMNS
                if column in group.columns
            ],
        ]
        for column in invariant_columns:
            if rows[50].get(column) != rows[80].get(column):
                raise ValueError(
                    f"q50/q80 liquidity gate mismatch for {normal_key}: {column}"
                )
        for quantile in (50, 80):
            expected_candidate = bool(rows[quantile]["support_gate"]) and bool(
                rows[quantile]["hard_data_gate"]
            )
            if bool(rows[quantile]["pre_replay_candidate"]) != expected_candidate:
                raise ValueError(
                    f"pre_replay_candidate is inconsistent for {normal_key}"
                )
        row = rows[50]
        gate_records.append(
            {
                "Date": normal_key[0],
                "ValueCode": normal_key[1],
                "QuoteCode": normal_key[2],
                "route": spot_bid_route,
                "support_gate": bool(row["support_gate"]),
                "hard_data_gate": bool(row["hard_data_gate"]),
                "pre_replay_candidate": bool(row["pre_replay_candidate"]),
                "liquidity_gate_status": str(row["liquidity_gate_status"]),
                "liquidity_source_asof_date": str(row["source_asof_date"]),
            }
        )
    gates = pl.from_dicts(gate_records, infer_schema_length=None)
    missing_liquidity = all_supported.select(BOUNDARY_KEYS).join(
        gates.select(BOUNDARY_KEYS),
        on=list(BOUNDARY_KEYS),
        how="anti",
    )
    if missing_liquidity.height:
        raise ValueError(
            "boundary-supported rows lack q-independent Spot-Bid liquidity: "
            f"{missing_liquidity.head(5).to_dicts()}"
        )

    broad = all_supported.join(
        gates,
        on=list(BOUNDARY_KEYS),
        how="inner",
        validate="1:1",
    ).filter(
        pl.col("support_gate")
        & pl.col("hard_data_gate")
        & pl.col("pre_replay_candidate")
    ).with_columns(
        pl.lit(True).alias("spot_bid_broad_candidate")
    ).select(SPOT_BID_BROAD_SCHEMA.names()).sort(list(BOUNDARY_KEYS))
    return FoundationCohorts(
        all_boundary_supported=all_supported,
        spot_bid_broad=broad,
    )


def _validated_frozen_boundaries(
    boundaries: pl.DataFrame,
    *,
    expected_quantiles: Sequence[int],
) -> pl.DataFrame:
    _reject_selector_columns(boundaries, "frozen boundaries")
    required = {
        *BOUNDARY_KEYS,
        "boundary_quantile",
        "upper_distance_bp",
        "lower_distance_bp",
        "adaptive_parameter_valid",
        "source_asof_date",
        "execution_safe_snapshot",
        "contains_target_day_outcome",
    }
    _require_columns(boundaries, required, "frozen boundaries")
    result = boundaries.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("boundary_quantile").cast(pl.Int64),
        pl.col("upper_distance_bp").cast(pl.Float64),
        pl.col("lower_distance_bp").cast(pl.Float64),
        pl.col("source_asof_date").cast(pl.String),
    )
    _validate_unique_rows(
        result,
        [*BOUNDARY_KEYS, "boundary_quantile"],
        "frozen boundaries",
    )
    _validate_safe_prediction_rows(result, "frozen boundaries")
    invalid = result.filter(
        ~pl.col("adaptive_parameter_valid").fill_null(False)
        | ~pl.col("upper_distance_bp").is_finite()
        | ~pl.col("lower_distance_bp").is_finite()
        | (pl.col("upper_distance_bp") <= 0)
        | (pl.col("lower_distance_bp") <= 0)
    )
    if invalid.height:
        raise ValueError("frozen boundaries must be valid, finite, and positive")
    expected = tuple(sorted({int(value) for value in expected_quantiles}))
    for key, group in result.group_by(list(BOUNDARY_KEYS)):
        observed = tuple(sorted(group["boundary_quantile"].to_list()))
        if observed != expected:
            raise ValueError(
                f"frozen boundary quantiles mismatch for {tuple(map(str, key))}: "
                f"{observed}"
            )
        ordered = group.sort("boundary_quantile")
        if not _nondecreasing(ordered["upper_distance_bp"].to_list()) or not (
            _nondecreasing(ordered["lower_distance_bp"].to_list())
        ):
            raise ValueError("frozen boundary quantiles must be monotone")
    return result


def _validate_one_day_panel(panel: pl.DataFrame) -> None:
    if panel["Date"].n_unique() != 1:
        raise ValueError("episode overlay input must contain exactly one Date")
    _validate_date_column(panel, "Date", "causal fair panel")
    if panel.filter(pl.col("timestamp").is_null()).height:
        raise ValueError("causal fair panel contains null timestamps")
    _validate_unique_rows(
        panel,
        [*BOUNDARY_KEYS, "seconds_from_open"],
        "one-second causal fair panel",
    )
    contract_counts = panel.group_by(["Date", "ValueCode"]).agg(
        pl.col("QuoteCode").n_unique().alias("contracts")
    ).filter(pl.col("contracts") != 1)
    if contract_counts.height:
        raise ValueError("one product-day must have exactly one QuoteCode")


def _validate_episode_overlay(episodes: pl.DataFrame) -> None:
    _require_columns(episodes, set(EPISODE_SCHEMA.names()), "episode overlay")
    if episodes.is_empty():
        return
    _validate_unique_rows(episodes, ["episode_id"], "episode overlay")
    invalid = episodes.filter(
        ~pl.col("side").is_in(["positive", "negative"])
        | ~pl.col("observed_amplitude_bp").is_finite()
        | (pl.col("observed_amplitude_bp") < 0)
        | (pl.col("end_seconds_from_open") < pl.col("start_seconds_from_open"))
        | (
            pl.col("right_censored")
            == pl.col("completed_center_return")
        )
        | (
            pl.col("fully_observed")
            != (
                ~pl.col("left_censored")
                & pl.col("completed_center_return")
            )
        )
        | (
            pl.col("left_censored")
            & pl.col("left_censor_reason").is_null()
        )
        | (
            ~pl.col("left_censored")
            & pl.col("left_censor_reason").is_not_null()
        )
        | (
            pl.col("right_censored")
            & pl.col("right_censor_reason").is_null()
        )
        | (
            ~pl.col("right_censored")
            & pl.col("right_censor_reason").is_not_null()
        )
    )
    if invalid.height:
        raise ValueError("episode overlay violates censor invariants")


def _validate_episode_contract_match(
    boundaries: pl.DataFrame,
    episodes: pl.DataFrame,
) -> None:
    if episodes.is_empty():
        return
    boundary_products = boundaries.select("Date", "ValueCode").unique()
    relevant = episodes.join(
        boundary_products,
        on=["Date", "ValueCode"],
        how="inner",
    )
    mismatch = relevant.select(BOUNDARY_KEYS).unique().join(
        boundaries.select(BOUNDARY_KEYS).unique(),
        on=list(BOUNDARY_KEYS),
        how="anti",
    )
    if mismatch.height:
        raise ValueError("episode QuoteCode does not match frozen boundary")


def _validate_censor_bound_rows(bounds: pl.DataFrame) -> None:
    invalid = bounds.filter(
        (
            pl.col("confirmed_hits")
            + pl.col("known_nonhits")
            + pl.col("unknown_nonhits")
            != pl.col("observable_started")
        )
        | (
            pl.col("completed_count") + pl.col("right_censored_count")
            != pl.col("observable_started")
        )
        | (pl.col("completed_hits") > pl.col("confirmed_hits"))
        | (pl.col("reach_lower_bound") < 0)
        | (pl.col("reach_upper_bound") > 1)
        | (
            pl.col("reach_lower_bound") > pl.col("reach_upper_bound")
        )
        | (
            (pl.col("observable_started") == 0)
            & (
                pl.col("reach_lower_bound").is_not_null()
                | pl.col("reach_upper_bound").is_not_null()
            )
        )
    )
    if invalid.height:
        raise ValueError("product-day censor bounds violate count invariants")


def _validate_safe_prediction_rows(frame: pl.DataFrame, source: str) -> None:
    _validate_date_column(frame, "Date", source)
    _validate_date_column(frame, "source_asof_date", source)
    invalid = frame.filter(
        ~pl.col("execution_safe_snapshot").fill_null(False)
        | pl.col("contains_target_day_outcome").fill_null(True)
        | (pl.col("source_asof_date") >= pl.col("Date"))
    )
    if invalid.height:
        raise ValueError(f"{source} is not a strictly prior D-safe snapshot")


def _validate_date_column(frame: pl.DataFrame, column: str, source: str) -> None:
    values = frame[column].cast(pl.String).to_list()
    for value in values:
        if value is None:
            raise ValueError(f"{source} contains null {column}")
        try:
            datetime.strptime(str(value), "%Y%m%d")  # noqa: DTZ007
        except ValueError as error:
            raise ValueError(
                f"{source} contains invalid YYYYMMDD {column}: {value!r}"
            ) from error


def _validate_date_text(value: str, name: str) -> None:
    try:
        datetime.strptime(str(value), "%Y%m%d")  # noqa: DTZ007
    except ValueError as error:
        raise ValueError(f"{name} must use YYYYMMDD: {value!r}") from error


def _reject_selector_columns(frame: pl.DataFrame, source: str) -> None:
    forbidden = sorted(
        column
        for column in frame.columns
        if column in _FORBIDDEN_SELECTOR_COLUMNS
        or column.startswith(_FORBIDDEN_SELECTOR_PREFIXES)
    )
    if forbidden:
        raise ValueError(f"{source} contains forbidden selector fields: {forbidden}")


def _validate_unique_rows(
    frame: pl.DataFrame,
    keys: Sequence[str],
    source: str,
) -> None:
    duplicates = frame.group_by(list(keys)).len().filter(pl.col("len") != 1)
    if duplicates.height:
        raise ValueError(f"{source} contains duplicate keys: {duplicates.head(5)}")


def _require_columns(
    frame: pl.DataFrame,
    required: set[str],
    source: str,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def _assert_constant(
    rows: Sequence[Mapping[str, object]] | object,
    column: str,
    key: tuple[str, str, str],
) -> None:
    materialized = list(rows)  # type: ignore[arg-type]
    values = {row.get(column) for row in materialized}
    if len(values) != 1:
        raise ValueError(f"boundary lineage differs across q for {key}: {column}")


def _optional_text(row: Mapping[str, object], column: str) -> str | None:
    value = row.get(column)
    return None if value is None else str(value)


def _finite(value: object) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _finite_positive(value: object) -> bool:
    return _finite(value) and float(value) > 0


def _nondecreasing(values: Sequence[object]) -> bool:
    numbers = [float(value) for value in values]
    return all(left <= right for left, right in pairwise(numbers))


# Compatibility name retained for the focused unit-test and any exploratory
# caller written against the audit draft.  Runners should use the explicit
# ``extract_censor_aware_excursions`` name below.
extract_censor_aware_episodes = extract_censor_aware_excursions


__all__ = [
    "ALL_BOUNDARY_SUPPORTED_SCHEMA",
    "CENSOR_BOUND_SCHEMA",
    "EPISODE_SCHEMA",
    "PRIMARY_QUANTILES",
    "SPOT_BID_BROAD_SCHEMA",
    "SPOT_BID_ROUTE",
    "FoundationCohorts",
    "build_foundation_cohorts",
    "build_product_day_censor_bounds",
    "calibrate_boundary_product_days",
    "extract_censor_aware_episodes",
    "extract_censor_aware_excursions",
]
