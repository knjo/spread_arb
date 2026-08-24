"""Tick-rounded quote churn implied by fair-mid anchor updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl

from ..common.landmarks import (
    REF_COMPARISON_EPS_RATIO,
    REF_LOWER_RETURN,
    REF_UPPER_RETURN,
)
from ..quote_fill.targets import (
    FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE,
    PRICE_LADDER_VERSION,
)
from .anchors import ANCHOR_COLUMNS, GROUP_KEYS
from .metrics import MODEL_NAMES


DEFAULT_DIAGNOSTIC_OPEN_WIDTH_BP = 20.0
PRICE_EPS = 1e-10
PriceMarket = Literal["spot", "future"]
SessionDateExpr = pl.Expr | str | None


@dataclass(frozen=True)
class QuoteChurnResult:
    by_model_route: pl.DataFrame
    by_day_symbol_route: pl.DataFrame


def _high_price_boundary(
    *,
    market: PriceMarket,
    session_date: SessionDateExpr,
) -> pl.Expr:
    """Return the versioned high-price boundary for a Polars expression.

    A missing date intentionally retains the legacy ladder for backwards
    compatible spot-only diagnostics.  Every futures caller in this module
    and :mod:`quote_width.table` passes the row's explicit ``Date``.
    """

    if market == "spot":
        return pl.lit(1000.0)
    if market != "future":
        raise ValueError(f"unknown price market: {market}")
    if session_date is None:
        return pl.lit(1000.0)
    if isinstance(session_date, str):
        if len(session_date) != 8 or not session_date.isdigit():
            raise ValueError("session_date must be YYYYMMDD")
        date = pl.lit(session_date)
    else:
        date = session_date
    # Casting also supports Date-typed columns; remove ISO separators before
    # comparing with the authoritative YYYYMMDD effective-date constant.
    post_change = (
        date.cast(pl.String).str.replace_all("-", "")
        >= FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE
    ).fill_null(False)
    return pl.when(post_change).then(pl.lit(2500.0)).otherwise(pl.lit(1000.0))


def price_to_tick_index(
    price: pl.Expr,
    *,
    market: PriceMarket = "spot",
    session_date: SessionDateExpr = None,
) -> pl.Expr:
    """Map the versioned Taiwan spot/stock-future ladder to a tick index."""

    value = price.cast(pl.Float64)
    high_price_boundary = _high_price_boundary(
        market=market,
        session_date=session_date,
    )
    high_price_index = 3100.0 + (high_price_boundary - 500.0)
    return (
        pl.when(value < 10.0)
        .then(value / 0.01)
        .when(value < 50.0)
        .then(1000.0 + (value - 10.0) / 0.05)
        .when(value < 100.0)
        .then(1800.0 + (value - 50.0) / 0.1)
        .when(value < 500.0)
        .then(2300.0 + (value - 100.0) / 0.5)
        .when(value < high_price_boundary)
        .then(3100.0 + (value - 500.0))
        .otherwise(high_price_index + (value - high_price_boundary) / 5.0)
    )


def tick_index_to_price(
    index: pl.Expr,
    *,
    market: PriceMarket = "spot",
    session_date: SessionDateExpr = None,
) -> pl.Expr:
    """Invert a tick index under the same versioned ladder."""

    value = index.cast(pl.Float64)
    high_price_boundary = _high_price_boundary(
        market=market,
        session_date=session_date,
    )
    high_price_index = 3100.0 + (high_price_boundary - 500.0)
    return (
        pl.when(value < 1000.0)
        .then(value * 0.01)
        .when(value < 1800.0)
        .then(10.0 + (value - 1000.0) * 0.05)
        .when(value < 2300.0)
        .then(50.0 + (value - 1800.0) * 0.1)
        .when(value < 3100.0)
        .then(100.0 + (value - 2300.0) * 0.5)
        .when(value < high_price_index)
        .then(500.0 + (value - 3100.0))
        .otherwise(
            high_price_boundary + (value - high_price_index) * 5.0
        )
    )


def round_up_to_tick(
    price: pl.Expr,
    *,
    market: PriceMarket = "spot",
    session_date: SessionDateExpr = None,
) -> pl.Expr:
    index = (
        price_to_tick_index(
            price,
            market=market,
            session_date=session_date,
        )
        - PRICE_EPS
    ).ceil()
    return tick_index_to_price(
        index,
        market=market,
        session_date=session_date,
    )


def round_down_to_tick(
    price: pl.Expr,
    *,
    market: PriceMarket = "spot",
    session_date: SessionDateExpr = None,
) -> pl.Expr:
    index = (
        price_to_tick_index(
            price,
            market=market,
            session_date=session_date,
        )
        + PRICE_EPS
    ).floor()
    return tick_index_to_price(
        index,
        market=market,
        session_date=session_date,
    )


def _target_price(
    route: str,
    anchor: pl.Expr,
    open_width_bp: float,
    session_date: pl.Expr,
) -> pl.Expr:
    threshold_multiplier = 1 + (anchor + open_width_bp) / 10_000
    if route == "future_ask_spot_taker":
        return round_up_to_tick(
            pl.col("spot_ask") * threshold_multiplier,
            market="future",
            session_date=session_date,
        )
    if route == "spot_bid_future_taker":
        return round_down_to_tick(
            pl.col("fut_exec_bid") / threshold_multiplier,
            market="spot",
            session_date=session_date,
        )
    raise ValueError(f"unknown route: {route}")


def _price_in_ref_band(price: pl.Expr, reference: pl.Expr) -> pl.Expr:
    epsilon = reference.abs() * REF_COMPARISON_EPS_RATIO
    lower = reference * (1 + REF_LOWER_RETURN) + epsilon
    upper = reference * (1 + REF_UPPER_RETURN) - epsilon
    return (
        reference.is_not_null()
        & (reference > 0)
        & price.is_not_null()
        & (price > lower)
        & (price < upper)
    ).fill_null(False)


def _route_frame(
    panel: pl.DataFrame,
    anchor_column: str,
    model: str,
    route: str,
    open_width_bp: float,
) -> pl.DataFrame:
    previous_anchor = pl.col(anchor_column).shift(1).over(GROUP_KEYS)
    reference_column = (
        "fut_ref_price"
        if route == "future_ask_spot_taker"
        else "spot_ref_price"
    )
    frame = panel.with_columns(
        previous_anchor.alias("previous_anchor_bp"),
        _target_price(
            route,
            pl.col(anchor_column),
            open_width_bp,
            pl.col("Date"),
        ).alias("target_price"),
        _target_price(
            route,
            previous_anchor,
            open_width_bp,
            pl.col("Date"),
        ).alias(
            "target_with_previous_anchor"
        ),
    )
    return (
        frame.with_columns(
            pl.col("target_price").shift(1).over(GROUP_KEYS).alias("previous_target_price")
        )
        .with_columns(
            _price_in_ref_band(
                pl.col("target_price"), pl.col(reference_column)
            ).alias("target_price_ok"),
            _price_in_ref_band(
                pl.col("target_with_previous_anchor"),
                pl.col(reference_column),
            ).alias("target_with_previous_anchor_ok"),
            _price_in_ref_band(
                pl.col("previous_target_price"), pl.col(reference_column)
            ).alias("previous_target_price_ok"),
        )
        .filter(
            pl.col("analysis_eligible")
            & pl.col(anchor_column).is_not_null()
            & pl.col("previous_anchor_bp").is_not_null()
            & pl.col("target_price").is_not_null()
            & pl.col("previous_target_price").is_not_null()
            & pl.col("target_price_ok")
            & pl.col("target_with_previous_anchor_ok")
            & pl.col("previous_target_price_ok")
        )
        .select(
            "Date",
            "ValueCode",
            "QuoteCode",
            "timestamp",
            pl.lit(model).alias("model"),
            pl.lit(route).alias("route"),
            pl.lit(open_width_bp).alias("open_width_bp"),
            (pl.col("target_price") != pl.col("target_with_previous_anchor"))
            .alias("fair_only_quote_change"),
            (pl.col("target_price") != pl.col("previous_target_price"))
            .alias("full_target_change"),
            (
                pl.col("target_with_previous_anchor")
                != pl.col("previous_target_price")
            ).alias("opposite_leg_only_change"),
        )
    )


def _aggregations() -> list[pl.Expr]:
    return [
        pl.len().alias("n"),
        pl.col("fair_only_quote_change").sum().alias("fair_only_changes"),
        pl.col("full_target_change").sum().alias("full_target_changes"),
        pl.col("opposite_leg_only_change").sum().alias("opposite_leg_only_changes"),
        (pl.col("fair_only_quote_change").mean() * 60)
        .alias("fair_only_changes_per_minute"),
        (pl.col("full_target_change").mean() * 60)
        .alias("full_target_changes_per_minute"),
        (pl.col("opposite_leg_only_change").mean() * 60)
        .alias("opposite_leg_only_changes_per_minute"),
    ]


def summarize_quote_churn(
    panel: pl.DataFrame,
    open_width_bp: float = DEFAULT_DIAGNOSTIC_OPEN_WIDTH_BP,
) -> QuoteChurnResult:
    """Measure route target changes caused by the anchor after tick rounding."""
    required = {
        "spot_ask",
        "fut_exec_bid",
        "analysis_eligible",
        "spot_ref_price",
        "fut_ref_price",
        *GROUP_KEYS,
        *ANCHOR_COLUMNS,
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"fair panel missing quote churn columns: {missing}")

    route_frames: list[pl.DataFrame] = []
    for anchor_column, model in MODEL_NAMES.items():
        for route in ("future_ask_spot_taker", "spot_bid_future_taker"):
            route_frames.append(
                _route_frame(panel, anchor_column, model, route, open_width_bp)
            )
    long = pl.concat(route_frames).sort(
        ["model", "route", *GROUP_KEYS, "timestamp"]
    )
    by_model_route = (
        long.group_by(["model", "route", "open_width_bp"])
        .agg(_aggregations())
        .with_columns(
            pl.lit(PRICE_LADDER_VERSION).alias("price_ladder_version")
        )
        .sort(["model", "route"])
    )
    by_day_symbol_route = (
        long.group_by(
            ["model", "route", "open_width_bp", "Date", "ValueCode", "QuoteCode"]
        )
        .agg(_aggregations())
        .with_columns(
            pl.lit(PRICE_LADDER_VERSION).alias("price_ladder_version")
        )
        .sort(["model", "route", "Date", "ValueCode"])
    )
    return QuoteChurnResult(
        by_model_route=by_model_route,
        by_day_symbol_route=by_day_symbol_route,
    )
