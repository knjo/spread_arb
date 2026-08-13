"""Tick-rounded quote churn implied by fair-mid anchor updates."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from ..common.landmarks import (
    REF_COMPARISON_EPS_RATIO,
    REF_LOWER_RETURN,
    REF_UPPER_RETURN,
)
from .anchors import ANCHOR_COLUMNS, GROUP_KEYS
from .metrics import MODEL_NAMES


DEFAULT_OPEN_WIDTH_BP = 20.0
PRICE_EPS = 1e-10


@dataclass(frozen=True)
class QuoteChurnResult:
    by_model_route: pl.DataFrame
    by_day_symbol_route: pl.DataFrame


def price_to_tick_index(price: pl.Expr) -> pl.Expr:
    """Map a Taiwan equity-style price ladder to a continuous tick index."""
    return (
        pl.when(price < 10)
        .then(price / 0.01)
        .when(price < 50)
        .then(1000 + (price - 10) / 0.05)
        .when(price < 100)
        .then(1800 + (price - 50) / 0.1)
        .when(price < 500)
        .then(2300 + (price - 100) / 0.5)
        .when(price < 1000)
        .then(3100 + (price - 500) / 1.0)
        .otherwise(3600 + (price - 1000) / 5.0)
    )


def tick_index_to_price(index: pl.Expr) -> pl.Expr:
    """Invert an integer Taiwan equity-style tick index."""
    return (
        pl.when(index < 1000)
        .then(index * 0.01)
        .when(index < 1800)
        .then(10 + (index - 1000) * 0.05)
        .when(index < 2300)
        .then(50 + (index - 1800) * 0.1)
        .when(index < 3100)
        .then(100 + (index - 2300) * 0.5)
        .when(index < 3600)
        .then(500 + (index - 3100) * 1.0)
        .otherwise(1000 + (index - 3600) * 5.0)
    )


def round_up_to_tick(price: pl.Expr) -> pl.Expr:
    index = (price_to_tick_index(price) - PRICE_EPS).ceil()
    return tick_index_to_price(index)


def round_down_to_tick(price: pl.Expr) -> pl.Expr:
    index = (price_to_tick_index(price) + PRICE_EPS).floor()
    return tick_index_to_price(index)


def _target_price(
    route: str,
    anchor: pl.Expr,
    open_width_bp: float,
) -> pl.Expr:
    threshold_multiplier = 1 + (anchor + open_width_bp) / 10_000
    if route == "future_ask_spot_taker":
        return round_up_to_tick(pl.col("spot_ask") * threshold_multiplier)
    if route == "spot_bid_future_taker":
        return round_down_to_tick(pl.col("fut_exec_bid") / threshold_multiplier)
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
        _target_price(route, pl.col(anchor_column), open_width_bp).alias("target_price"),
        _target_price(route, previous_anchor, open_width_bp).alias(
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
    open_width_bp: float = DEFAULT_OPEN_WIDTH_BP,
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
        .sort(["model", "route"])
    )
    by_day_symbol_route = (
        long.group_by(
            ["model", "route", "open_width_bp", "Date", "ValueCode", "QuoteCode"]
        )
        .agg(_aggregations())
        .sort(["model", "route", "Date", "ValueCode"])
    )
    return QuoteChurnResult(
        by_model_route=by_model_route,
        by_day_symbol_route=by_day_symbol_route,
    )
