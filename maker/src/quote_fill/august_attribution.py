"""Pure S0 residual-excursion and post-touch attribution contracts.

The filesystem runner lives in :mod:`august_attribution_runner`.  This module
keeps the hard-to-audit pieces independent of raw-file I/O: exact cursor
ordering, positive-excursion extraction, left-censor handling, first-touch to
working-order mapping, and denominator-explicit monthly summaries.
"""

from __future__ import annotations

import hashlib
import math
from bisect import bisect_right
from collections.abc import Iterable, Mapping

import polars as pl

SCHEMA_VERSION = "august_attribution_s0_v3"

# Within one receive timestamp, raw state is ingested first, then a potential
# fill, then cancel effects, and finally a new order becomes working.
RAW_FUTURE_PHASE = 1
RAW_SPOT_PHASE = 2
APPROXIMATE_FILL_PHASE = 3
ACTUAL_CANCEL_PHASE = 5
ACTUAL_NEW_PHASE = 6

CURSOR_FIELDS = ("time_ns", "event_sequence", "row_index")


EXCURSION_SCHEMA: Mapping[str, pl.DataType] = {
    "excursion_id": pl.String,
    "Date": pl.String,
    "month": pl.String,
    "ValueCode": pl.String,
    "QuoteCode": pl.String,
    "excursion_sequence": pl.Int64,
    "left_censored": pl.Boolean,
    "left_censor_reason": pl.String,
    "primary_observable": pl.Boolean,
    "completed": pl.Boolean,
    "end_reason": pl.String,
    "amplitude_bp": pl.Float64,
    "upper_distance_bp": pl.Float64,
    "start_already_at_or_above_upper": pl.Boolean,
    "start_time_ns": pl.Int64,
    "start_event_sequence": pl.Int64,
    "start_row_index": pl.Int64,
    "start_residual_bp": pl.Float64,
    "previous_residual_bp": pl.Float64,
    "end_time_ns": pl.Int64,
    "end_event_sequence": pl.Int64,
    "end_row_index": pl.Int64,
    "end_residual_bp": pl.Float64,
    "touch_time_ns": pl.Int64,
    "touch_event_sequence": pl.Int64,
    "touch_row_index": pl.Int64,
    "touch_previous_residual_bp": pl.Float64,
    "touch_residual_bp": pl.Float64,
    "touch_spread_pair_epoch": pl.Int64,
}


TOUCH_LINK_SCHEMA: Mapping[str, pl.DataType] = {
    "raw_order_fact_id": pl.String,
    "excursion_id": pl.String,
    "Date": pl.String,
    "month": pl.String,
    "ValueCode": pl.String,
    "QuoteCode": pl.String,
    "exact_target_rank": pl.String,
    "touch_time_ns": pl.Int64,
    "touch_event_sequence": pl.Int64,
    "touch_row_index": pl.Int64,
    "touch_spread_pair_epoch": pl.Int64,
    "actual_new_time_ns": pl.Int64,
    "active_end_time_ns": pl.Int64,
    "active_end_reason": pl.String,
    "approximate_fill_time_ns": pl.Int64,
    "outcome_supported": pl.Boolean,
    "queue_denominator_supported": pl.Boolean,
    "post_touch_fill": pl.Boolean,
}


def cursor_tuple(row: Mapping[str, object], prefix: str) -> tuple[int, int, int]:
    """Return one total-order cursor from ``<prefix>_*`` integer fields."""

    values = tuple(row.get(f"{prefix}_{field}") for field in CURSOR_FIELDS)
    if any(value is None for value in values):
        raise ValueError(f"{prefix} cursor is incomplete: {values!r}")
    return tuple(int(value) for value in values)  # type: ignore[return-value]


def extract_positive_excursions(raw_states: pl.DataFrame) -> pl.DataFrame:
    """Extract positive raw-state excursions and their first q95 touch.

    ``raw_states`` must retain eligibility transitions as rows.  A positive
    episode begins only on an observed ``<=0 -> >0`` crossing and completes at
    the next observed eligible ``<=0`` state.  A positive state at the start
    of the session (or immediately after an eligibility gap) is explicitly
    left-censored.  Starting above q95 never fabricates a touch cursor.
    """

    required = {
        "Date",
        "ValueCode",
        "QuoteCode",
        "cursor_time_ns",
        "cursor_event_sequence",
        "cursor_row_index",
        "analysis_eligible_raw",
        "residual_excursion_bp",
        "upper_distance_bp",
        "spread_pair_epoch",
    }
    _require_columns(raw_states, required, "raw residual states")
    if raw_states.is_empty():
        return pl.DataFrame(schema=EXCURSION_SCHEMA)

    ordered = raw_states.sort(
        [
            "Date",
            "ValueCode",
            "cursor_time_ns",
            "cursor_event_sequence",
            "cursor_row_index",
        ]
    )
    records: list[dict[str, object]] = []
    for key, group in ordered.group_by(
        ["Date", "ValueCode", "QuoteCode"], maintain_order=True
    ):
        date, value_code, quote_code = map(str, key)
        rows = group.iter_rows(named=True)
        sequence = 0
        seen_eligible = False
        after_gap = False
        previous: dict[str, object] | None = None
        active: dict[str, object] | None = None
        last_eligible: dict[str, object] | None = None

        def begin(
            row: Mapping[str, object],
            *,
            left_censored: bool,
            left_censor_reason: str | None,
            previous_residual: float | None,
        ) -> dict[str, object]:
            nonlocal sequence
            sequence += 1
            residual = float(row["residual_excursion_bp"])
            upper = float(row["upper_distance_bp"])
            if not math.isfinite(upper) or upper <= 0:
                raise ValueError("q95 upper distance must be finite and positive")
            excursion: dict[str, object] = {
                "excursion_sequence": sequence,
                "left_censored": left_censored,
                "left_censor_reason": left_censor_reason,
                "start_time_ns": int(row["cursor_time_ns"]),
                "start_event_sequence": int(row["cursor_event_sequence"]),
                "start_row_index": int(row["cursor_row_index"]),
                "start_residual_bp": residual,
                "previous_residual_bp": previous_residual,
                "amplitude_bp": residual,
                "upper_distance_bp": upper,
                "start_already_at_or_above_upper": residual >= upper,
                "touch_time_ns": None,
                "touch_event_sequence": None,
                "touch_row_index": None,
                "touch_previous_residual_bp": None,
                "touch_residual_bp": None,
                "touch_spread_pair_epoch": None,
            }
            # A fully observed crossing may jump from the center through q95.
            if not left_censored and residual >= upper:
                assert previous_residual is not None
                _set_touch(excursion, row, previous_residual)
            return excursion

        def finish(
            excursion: dict[str, object],
            row: Mapping[str, object],
            *,
            completed: bool,
            reason: str,
            _date: str = date,
            _value_code: str = value_code,
            _quote_code: str = quote_code,
        ) -> None:
            excursion.update(
                {
                    "excursion_id": _stable_id(
                        _date,
                        _value_code,
                        str(excursion["excursion_sequence"]),
                    ),
                    "Date": _date,
                    "month": _date[:6],
                    "ValueCode": _value_code,
                    "QuoteCode": _quote_code,
                    "primary_observable": not bool(
                        excursion["left_censored"]
                    ),
                    "completed": completed,
                    "end_reason": reason,
                    "end_time_ns": int(row["cursor_time_ns"]),
                    "end_event_sequence": int(row["cursor_event_sequence"]),
                    "end_row_index": int(row["cursor_row_index"]),
                    "end_residual_bp": (
                        float(row["residual_excursion_bp"])
                        if row["residual_excursion_bp"] is not None
                        else None
                    ),
                }
            )
            records.append(excursion)

        for row in rows:
            eligible = bool(row["analysis_eligible_raw"])
            value = row["residual_excursion_bp"]
            eligible = eligible and _finite(value)
            if not eligible:
                if active is not None and last_eligible is not None:
                    finish(
                        active,
                        last_eligible,
                        completed=False,
                        reason="eligibility_gap",
                    )
                    active = None
                if seen_eligible:
                    after_gap = True
                previous = None
                last_eligible = None
                continue

            residual = float(value)
            if not seen_eligible or after_gap:
                censor_reason = "session_start" if not seen_eligible else "eligibility_gap"
                if residual > 0:
                    active = begin(
                        row,
                        left_censored=True,
                        left_censor_reason=censor_reason,
                        previous_residual=None,
                    )
                seen_eligible = True
                after_gap = False
                previous = row
                last_eligible = row
                continue

            assert previous is not None
            previous_residual = float(previous["residual_excursion_bp"])
            if active is None:
                if previous_residual <= 0 < residual:
                    active = begin(
                        row,
                        left_censored=False,
                        left_censor_reason=None,
                        previous_residual=previous_residual,
                    )
            else:
                upper = float(active["upper_distance_bp"])
                if (
                    active["touch_time_ns"] is None
                    and not (
                        bool(active["left_censored"])
                        and bool(active["start_already_at_or_above_upper"])
                    )
                    and previous_residual < upper <= residual
                ):
                    _set_touch(active, row, previous_residual)
                if residual > 0:
                    active["amplitude_bp"] = max(
                        float(active["amplitude_bp"]), residual
                    )
                else:
                    finish(active, row, completed=True, reason="center_return")
                    active = None
            previous = row
            last_eligible = row

        if active is not None and last_eligible is not None:
            finish(
                active,
                last_eligible,
                completed=False,
                reason="session_cutoff",
            )

    if not records:
        return pl.DataFrame(schema=EXCURSION_SCHEMA)
    return pl.from_dicts(
        records,
        schema=EXCURSION_SCHEMA,
        infer_schema_length=None,
    ).sort(["Date", "ValueCode", "excursion_sequence"])


def link_touch_order_pairs(
    excursions: pl.DataFrame,
    raw_orders: pl.DataFrame,
) -> pl.DataFrame:
    """Map every market first-touch to every order working at that cursor."""

    _require_columns(
        excursions,
        {
            "excursion_id",
            "Date",
            "ValueCode",
            "QuoteCode",
            "primary_observable",
            "touch_time_ns",
            "touch_event_sequence",
            "touch_row_index",
            "touch_spread_pair_epoch",
        },
        "excursions",
    )
    _require_columns(
        raw_orders,
        {
            "raw_order_fact_id",
            "Date",
            "ValueCode",
            "QuoteCode",
            "exact_target_rank",
            "actual_new_time_ns",
            "actual_new_event_sequence",
            "actual_new_row_index",
            "active_end_time_ns",
            "active_end_event_sequence",
            "active_end_row_index",
            "active_end_reason",
            "approximate_fill_time_ns",
            "approximate_fill_event_sequence",
            "approximate_fill_row_index",
            "outcome_supported",
        },
        "raw orders",
    )
    touches = excursions.filter(
        pl.col("primary_observable") & pl.col("touch_time_ns").is_not_null()
    )
    if touches.is_empty() or raw_orders.is_empty():
        return pl.DataFrame(schema=TOUCH_LINK_SCHEMA)

    touch_groups = {
        (str(key[0]), str(key[1]), str(key[2])): group.sort(
            ["touch_time_ns", "touch_event_sequence", "touch_row_index"]
        ).to_dicts()
        for key, group in touches.group_by(
            ["Date", "ValueCode", "QuoteCode"]
        )
    }
    records: list[dict[str, object]] = []
    for key, orders in raw_orders.group_by(
        ["Date", "ValueCode", "QuoteCode"], maintain_order=True
    ):
        date, value_code, quote_code = map(str, key)
        candidates = touch_groups.get((date, value_code, quote_code), [])
        if not candidates:
            continue
        cursors = [cursor_tuple(row, "touch") for row in candidates]
        for order in orders.iter_rows(named=True):
            start = cursor_tuple(order, "actual_new")
            end = cursor_tuple(order, "active_end")
            if not start < end:
                raise ValueError("raw order active interval is not positive")
            fill = _optional_cursor(order, "approximate_fill")
            supported = bool(order["outcome_supported"])
            first = bisect_right(cursors, start)
            stop = bisect_right(cursors, end)
            for index in range(first, stop):
                touch_cursor = cursors[index]
                # Once the potential fill is at or before a touch, this order
                # is terminal for this and every later market touch.
                if fill is not None and fill <= touch_cursor:
                    break
                touch = candidates[index]
                post_touch = bool(
                    supported and fill is not None and fill <= end
                )
                records.append(
                    {
                        "raw_order_fact_id": str(
                            order["raw_order_fact_id"]
                        ),
                        "excursion_id": str(touch["excursion_id"]),
                        "Date": date,
                        "month": date[:6],
                        "ValueCode": value_code,
                        "QuoteCode": quote_code,
                        "exact_target_rank": order["exact_target_rank"],
                        "touch_time_ns": int(touch["touch_time_ns"]),
                        "touch_event_sequence": int(
                            touch["touch_event_sequence"]
                        ),
                        "touch_row_index": int(touch["touch_row_index"]),
                        "touch_spread_pair_epoch": touch[
                            "touch_spread_pair_epoch"
                        ],
                        "actual_new_time_ns": int(
                            order["actual_new_time_ns"]
                        ),
                        "active_end_time_ns": int(
                            order["active_end_time_ns"]
                        ),
                        "active_end_reason": str(
                            order["active_end_reason"]
                        ),
                        "approximate_fill_time_ns": order[
                            "approximate_fill_time_ns"
                        ],
                        "outcome_supported": supported,
                        "queue_denominator_supported": supported,
                        "post_touch_fill": post_touch,
                    }
                )
    if not records:
        return pl.DataFrame(schema=TOUCH_LINK_SCHEMA)
    return pl.from_dicts(
        records,
        schema=TOUCH_LINK_SCHEMA,
        infer_schema_length=None,
    ).sort(
        [
            "Date",
            "ValueCode",
            "touch_time_ns",
            "touch_event_sequence",
            "touch_row_index",
            "raw_order_fact_id",
        ]
    )


def link_first_touches_to_orders(
    excursions: pl.DataFrame,
    raw_orders: pl.DataFrame,
) -> pl.DataFrame:
    """Deduplicate full touch/order pairs to each order's earliest touch."""

    return deduplicate_touch_pairs_to_orders(
        link_touch_order_pairs(excursions, raw_orders)
    )


def deduplicate_touch_pairs_to_orders(
    pairs: pl.DataFrame,
) -> pl.DataFrame:
    """Select each raw order's earliest qualifying market first-touch."""

    _require_columns(
        pairs,
        set(TOUCH_LINK_SCHEMA),
        "touch/order pairs",
    )
    if pairs.is_empty():
        return pairs
    return (
        pairs.sort(
            [
                "touch_time_ns",
                "touch_event_sequence",
                "touch_row_index",
                "raw_order_fact_id",
            ]
        )
        .unique(
            subset=["raw_order_fact_id"],
            keep="first",
            maintain_order=True,
        )
        .sort(["Date", "ValueCode", "raw_order_fact_id"])
    )


def summarize_monthly_attribution(
    product_days: pl.DataFrame,
    excursions: pl.DataFrame,
    raw_orders: pl.DataFrame,
    touch_links: pl.DataFrame,
    *,
    touch_pairs: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build one denominator-explicit May--August attribution table."""

    _require_columns(
        product_days,
        {
            "Date",
            "ValueCode",
            "upper_distance_bp",
            "eligible_raw_state_count",
        },
        "product-day coverage",
    )
    pair_source = touch_links if touch_pairs is None else touch_pairs
    months = sorted({str(value)[:6] for value in product_days["Date"]})
    rows: list[dict[str, object]] = []
    for month in months:
        days = product_days.filter(pl.col("Date").str.slice(0, 6) == month)
        month_excursions = excursions.filter(pl.col("month") == month)
        primary = month_excursions.filter(pl.col("primary_observable"))
        completed = primary.filter(pl.col("completed"))
        primary_touches = primary.filter(pl.col("touch_time_ns").is_not_null())
        left = month_excursions.filter(pl.col("left_censored"))
        sensitivity_touches = month_excursions.filter(
            pl.col("touch_time_ns").is_not_null()
        )
        orders = raw_orders.filter(pl.col("Date").str.slice(0, 6) == month)
        links = touch_links.filter(pl.col("month") == month)
        pairs = pair_source.filter(pl.col("month") == month)
        supported_links = links.filter(pl.col("queue_denominator_supported"))
        pd_equal_rate, pd_equal_count = _product_day_equal_fill_rate(
            supported_links
        )
        touched_pd = primary_touches.select("Date", "ValueCode").unique().height
        sensitivity_touched_pd = (
            sensitivity_touches.select("Date", "ValueCode").unique().height
        )
        product_day_count = days.height
        observable_count = primary.height
        touch_count = primary_touches.height
        supported_touch_orders = supported_links.height
        touches_with_working_order = pairs["excursion_id"].n_unique()
        post_touch_fills = int(
            supported_links["post_touch_fill"].sum() or 0
        ) if supported_touch_orders else 0
        rows.append(
            {
                "month": month,
                "session_count": days["Date"].n_unique(),
                "product_days": product_day_count,
                "eligible_product_days": days.filter(
                    pl.col("eligible_raw_state_count") > 0
                ).height,
                "no_eligible_product_days": days.filter(
                    pl.col("eligible_raw_state_count") == 0
                ).height,
                "observable_excursions": observable_count,
                "completed_observable_excursions": completed.height,
                "right_censored_observable_excursions": primary.filter(
                    ~pl.col("completed")
                ).height,
                "session_start_left_censored": left.filter(
                    pl.col("left_censor_reason") == "session_start"
                ).height,
                "gap_left_censored": left.filter(
                    pl.col("left_censor_reason") == "eligibility_gap"
                ).height,
                "left_censored_start_at_or_above_upper": left.filter(
                    pl.col("start_already_at_or_above_upper")
                ).height,
                "residual_excursion_p50_bp": _quantile(
                    completed["amplitude_bp"], 0.50
                ),
                "residual_excursion_p80_bp": _quantile(
                    completed["amplitude_bp"], 0.80
                ),
                "residual_excursion_p95_bp": _quantile(
                    completed["amplitude_bp"], 0.95
                ),
                "q95_boundary_p50_bp": _quantile(
                    days["upper_distance_bp"], 0.50
                ),
                "q95_boundary_p80_bp": _quantile(
                    days["upper_distance_bp"], 0.80
                ),
                "q95_boundary_p95_bp": _quantile(
                    days["upper_distance_bp"], 0.95
                ),
                "observable_excursions_per_product_day": _ratio(
                    observable_count, product_day_count
                ),
                "primary_touches": touch_count,
                "primary_touches_with_working_order": (
                    touches_with_working_order
                ),
                "primary_touches_without_working_order": (
                    touch_count - touches_with_working_order
                ),
                "excursion_touch_rate": _ratio(
                    touch_count, observable_count
                ),
                "touches_per_product_day": _ratio(
                    touch_count, product_day_count
                ),
                "product_days_with_any_touch": touched_pd,
                "product_day_any_touch_rate": _ratio(
                    touched_pd, product_day_count
                ),
                "including_left_censored_sensitivity_touches": (
                    sensitivity_touches.height
                ),
                "including_left_censored_sensitivity_touch_pd_rate": _ratio(
                    sensitivity_touched_pd, product_day_count
                ),
                "actual_working_orders": orders.height,
                "outcome_supported_orders": int(
                    orders["outcome_supported"].sum() or 0
                ) if not orders.is_empty() else 0,
                "touched_raw_order_facts": links.height,
                "outcome_supported_touched_orders": supported_touch_orders,
                "post_touch_fills": post_touch_fills,
                "post_touch_fill_rate_pooled": _ratio(
                    post_touch_fills, supported_touch_orders
                ),
                "post_touch_fill_rate_product_day_equal": (
                    pd_equal_rate
                ),
                "product_days_in_queue_equal_weight": pd_equal_count,
                "product_days_without_supported_touched_orders": (
                    product_day_count - pd_equal_count
                ),
            }
        )
    return pl.from_dicts(rows, infer_schema_length=None).sort("month")


def summarize_post_touch_by_rank(touch_links: pl.DataFrame) -> pl.DataFrame:
    """Report pooled and product-day equal post-touch rates by BID rank."""

    if touch_links.is_empty():
        return pl.DataFrame(
            schema={
                "month": pl.String,
                "exact_target_rank": pl.String,
                "touched_raw_order_facts": pl.Int64,
                "outcome_supported_touched_orders": pl.Int64,
                "post_touch_fills": pl.Int64,
                "post_touch_fill_rate_pooled": pl.Float64,
                "post_touch_fill_rate_product_day_equal": pl.Float64,
                "product_days": pl.Int64,
            }
        )
    rows: list[dict[str, object]] = []
    for key, group in touch_links.group_by(
        ["month", "exact_target_rank"], maintain_order=True
    ):
        supported = group.filter(pl.col("queue_denominator_supported"))
        pd_equal_rate, pd_equal_count = _product_day_equal_fill_rate(
            supported
        )
        numerator = int(supported["post_touch_fill"].sum() or 0) if not supported.is_empty() else 0
        rows.append(
            {
                "month": str(key[0]),
                "exact_target_rank": (
                    str(key[1]) if key[1] is not None else "UNMAPPED"
                ),
                "touched_raw_order_facts": group.height,
                "outcome_supported_touched_orders": supported.height,
                "post_touch_fills": numerator,
                "post_touch_fill_rate_pooled": _ratio(numerator, supported.height),
                "post_touch_fill_rate_product_day_equal": (
                    pd_equal_rate
                ),
                "product_days": pd_equal_count,
            }
        )
    return pl.from_dicts(rows, infer_schema_length=None).sort(
        ["month", "exact_target_rank"]
    )


def _product_day_equal_fill_rate(
    supported_links: pl.DataFrame,
) -> tuple[float | None, int]:
    """Compute an order-independent mean of exact integer product-day rates."""

    if supported_links.is_empty():
        return None, 0
    counts = (
        supported_links.group_by("Date", "ValueCode")
        .agg(
            pl.len().alias("denominator"),
            pl.col("post_touch_fill").sum().alias("numerator"),
        )
        .sort(["Date", "ValueCode"])
    )
    rates = [
        int(row["numerator"]) / int(row["denominator"])
        for row in counts.iter_rows(named=True)
    ]
    return math.fsum(rates) / len(rates), len(rates)


def build_decomposition(monthly: pl.DataFrame) -> pl.DataFrame:
    """Compare August with pooled May--July and classify both mechanisms."""

    if monthly.filter(pl.col("month") == "202608").height != 1:
        raise ValueError("monthly attribution must contain exactly one August row")
    prior = monthly.filter(pl.col("month") < "202608")
    august = monthly.filter(pl.col("month") == "202608").row(0, named=True)
    prior_observable = int(prior["observable_excursions"].sum() or 0)
    prior_touches = int(prior["primary_touches"].sum() or 0)
    prior_supported = int(
        prior["outcome_supported_touched_orders"].sum() or 0
    )
    prior_fills = int(prior["post_touch_fills"].sum() or 0)
    prior_touch_rate = _ratio(prior_touches, prior_observable)
    prior_fill_rate = _ratio(prior_fills, prior_supported)
    august_touch_rate = august["excursion_touch_rate"]
    august_fill_rate = august["post_touch_fill_rate_pooled"]
    market_drop = (
        prior_touch_rate is not None
        and august_touch_rate is not None
        and float(august_touch_rate) < prior_touch_rate
    )
    queue_drop = (
        prior_fill_rate is not None
        and august_fill_rate is not None
        and float(august_fill_rate) < prior_fill_rate
    )
    if market_drop and queue_drop:
        classification = "market_boundary_and_queue"
    elif market_drop:
        classification = "market_boundary_only"
    elif queue_drop:
        classification = "queue_competition_only"
    else:
        classification = "neither_directionally_lower"
    return pl.DataFrame(
        {
            "comparison": ["202608_vs_202605_202607_pooled"],
            "prior_excursion_touch_rate": [prior_touch_rate],
            "august_excursion_touch_rate": [august_touch_rate],
            "touch_rate_change": [
                _difference(august_touch_rate, prior_touch_rate)
            ],
            "prior_post_touch_fill_rate": [prior_fill_rate],
            "august_post_touch_fill_rate": [august_fill_rate],
            "post_touch_fill_rate_change": [
                _difference(august_fill_rate, prior_fill_rate)
            ],
            "market_boundary_directionally_lower": [market_drop],
            "queue_directionally_lower": [queue_drop],
            "classification": [classification],
            "causal_claim": [False],
        }
    )


def _set_touch(
    excursion: dict[str, object],
    row: Mapping[str, object],
    previous_residual: float,
) -> None:
    excursion.update(
        {
            "touch_time_ns": int(row["cursor_time_ns"]),
            "touch_event_sequence": int(row["cursor_event_sequence"]),
            "touch_row_index": int(row["cursor_row_index"]),
            "touch_previous_residual_bp": float(previous_residual),
            "touch_residual_bp": float(row["residual_excursion_bp"]),
            "touch_spread_pair_epoch": (
                int(row["spread_pair_epoch"])
                if row["spread_pair_epoch"] is not None
                else None
            ),
        }
    )


def _optional_cursor(
    row: Mapping[str, object], prefix: str
) -> tuple[int, int, int] | None:
    values = tuple(row.get(f"{prefix}_{field}") for field in CURSOR_FIELDS)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"{prefix} optional cursor is partially null")
    return tuple(int(value) for value in values)  # type: ignore[return-value]


def _stable_id(*parts: str) -> str:
    payload = "/".join(parts)
    return "exc-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _quantile(series: pl.Series, quantile: float) -> float | None:
    if series.is_empty():
        return None
    value = series.quantile(quantile, interpolation="nearest")
    return float(value) if value is not None else None


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _difference(left: object, right: object) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require_columns(
    frame: pl.DataFrame,
    required: Iterable[str],
    source: str,
) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}")


__all__ = [
    "ACTUAL_CANCEL_PHASE",
    "ACTUAL_NEW_PHASE",
    "APPROXIMATE_FILL_PHASE",
    "EXCURSION_SCHEMA",
    "RAW_FUTURE_PHASE",
    "RAW_SPOT_PHASE",
    "SCHEMA_VERSION",
    "TOUCH_LINK_SCHEMA",
    "build_decomposition",
    "cursor_tuple",
    "extract_positive_excursions",
    "link_first_touches_to_orders",
    "summarize_monthly_attribution",
    "summarize_post_touch_by_rank",
]
