"""Route-specific legal maker target geometry.

The functions in this module are scalar on purpose.  WP02 constructs a sparse
set of action observations; keeping the route contract here avoids duplicating
rounding and sign conventions across the sampler and replay code.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

MakerSide = Literal["bid", "ask"]
Stage = Literal["entry", "exit"]
PriceMarket = Literal["spot", "future"]


# TAIFEX changed the high-price stock-futures ladder on 2026-07-06.  Spot
# prices did not change: TWSE/TPEx cash equities still move in five-dollar
# ticks at and above 1,000.  Keep the regime in one versioned contract so raw
# trades, generated targets, persisted runner config, and resume checks cannot
# silently disagree.
FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE = "20260706"
PRICE_LADDER_VERSION = "tw_stock_spot_v1_future_20260706_v2"


@dataclass(frozen=True)
class RouteSpec:
    route: str
    stage: Stage
    maker_market: Literal["spot", "future"]
    maker_side: MakerSide


ROUTE_SPECS: Mapping[str, RouteSpec] = {
    "future_ask_spot_taker": RouteSpec(
        "future_ask_spot_taker", "entry", "future", "ask"
    ),
    "spot_bid_future_taker": RouteSpec(
        "spot_bid_future_taker", "entry", "spot", "bid"
    ),
    "future_bid_spot_taker": RouteSpec(
        "future_bid_spot_taker", "exit", "future", "bid"
    ),
    "spot_ask_future_taker": RouteSpec(
        "spot_ask_future_taker", "exit", "spot", "ask"
    ),
}


def price_to_tick_index(
    price: float,
    *,
    market: PriceMarket = "spot",
    session_date: str | None = None,
) -> float:
    """Map the applicable Taiwan spot/stock-future ladder to an index.

    ``session_date`` matters only for stock futures.  Before 2026-07-06 their
    one-dollar tier ended at 1,000; from that session onward it ends at 2,500.
    Omitting the date deliberately retains the legacy/spot ladder for backward
    compatibility, while every execution replay passes its explicit Date.
    """

    if not math.isfinite(price) or price <= 0:
        raise ValueError("price must be finite and positive")
    high_price_boundary = _high_price_boundary(market, session_date)
    if price < 10:
        return price / 0.01
    if price < 50:
        return 1000 + (price - 10) / 0.05
    if price < 100:
        return 1800 + (price - 50) / 0.1
    if price < 500:
        return 2300 + (price - 100) / 0.5
    if price < high_price_boundary:
        return 3100 + (price - 500)
    high_price_index = 3100 + (high_price_boundary - 500)
    return high_price_index + (price - high_price_boundary) / 5


def tick_index_to_price(
    index: int,
    *,
    market: PriceMarket = "spot",
    session_date: str | None = None,
) -> float:
    if index < 0:
        raise ValueError("tick index must be non-negative")
    high_price_boundary = _high_price_boundary(market, session_date)
    high_price_index = int(3100 + (high_price_boundary - 500))
    if index < 1000:
        return float(index * 0.01)
    if index < 1800:
        return float(10 + (index - 1000) * 0.05)
    if index < 2300:
        return float(50 + (index - 1800) * 0.1)
    if index < 3100:
        return float(100 + (index - 2300) * 0.5)
    if index < high_price_index:
        return float(500 + (index - 3100))
    return float(high_price_boundary + (index - high_price_index) * 5)


def absolute_price_tick(
    price: float,
    *,
    market: PriceMarket = "spot",
    session_date: str | None = None,
) -> int:
    """Return an exact legal tick index, rejecting off-ladder prices."""
    index = price_to_tick_index(
        price,
        market=market,
        session_date=session_date,
    )
    rounded = round(index)
    if not math.isclose(index, rounded, rel_tol=0.0, abs_tol=1e-7):
        raise ValueError(f"off-ladder price: {price}")
    return rounded


def round_up_to_tick(
    price: float,
    *,
    market: PriceMarket = "spot",
    session_date: str | None = None,
) -> float:
    index = math.ceil(
        price_to_tick_index(
            price,
            market=market,
            session_date=session_date,
        )
        - 1e-10
    )
    return tick_index_to_price(
        index,
        market=market,
        session_date=session_date,
    )


def round_down_to_tick(
    price: float,
    *,
    market: PriceMarket = "spot",
    session_date: str | None = None,
) -> float:
    index = math.floor(
        price_to_tick_index(
            price,
            market=market,
            session_date=session_date,
        )
        + 1e-10
    )
    return tick_index_to_price(
        index,
        market=market,
        session_date=session_date,
    )


def _multiplier(threshold_basis_bp: float) -> float:
    value = 1.0 + threshold_basis_bp / 10_000.0
    if not math.isfinite(value) or value <= 0:
        raise ValueError("basis threshold implies a non-positive price multiplier")
    return value


def target_price_for_basis(
    route: str,
    threshold_basis_bp: float,
    *,
    session_date: str | None = None,
    spot_bid: float | None = None,
    spot_ask: float | None = None,
    fut_exec_bid: float | None = None,
    fut_exec_ask: float | None = None,
) -> float:
    """Convert a basis threshold to the conservative legal maker tick."""
    if route not in ROUTE_SPECS:
        raise ValueError(f"unknown route: {route}")
    maker_market = ROUTE_SPECS[route].maker_market
    multiplier = _multiplier(threshold_basis_bp)
    if route == "future_ask_spot_taker":
        return round_up_to_tick(
            _required(spot_ask, "spot_ask") * multiplier,
            market=maker_market,
            session_date=session_date,
        )
    if route == "spot_bid_future_taker":
        return round_down_to_tick(
            _required(fut_exec_bid, "fut_exec_bid") / multiplier,
            market=maker_market,
            session_date=session_date,
        )
    if route == "future_bid_spot_taker":
        return round_down_to_tick(
            _required(spot_bid, "spot_bid") * multiplier,
            market=maker_market,
            session_date=session_date,
        )
    return round_up_to_tick(
        _required(fut_exec_ask, "fut_exec_ask") / multiplier,
        market=maker_market,
        session_date=session_date,
    )


def effective_basis_bp(
    route: str,
    target_price: float,
    *,
    spot_bid: float | None = None,
    spot_ask: float | None = None,
    fut_exec_bid: float | None = None,
    fut_exec_ask: float | None = None,
) -> float:
    """Basis realized by the rounded target and executable opposite leg."""
    target = _required(target_price, "target_price")
    if route == "future_ask_spot_taker":
        ratio = target / _required(spot_ask, "spot_ask")
    elif route == "spot_bid_future_taker":
        ratio = _required(fut_exec_bid, "fut_exec_bid") / target
    elif route == "future_bid_spot_taker":
        ratio = target / _required(spot_bid, "spot_bid")
    elif route == "spot_ask_future_taker":
        ratio = _required(fut_exec_ask, "fut_exec_ask") / target
    else:
        raise ValueError(f"unknown route: {route}")
    return 10_000.0 * (ratio - 1.0)


def is_passive_target(
    route: str,
    target_price: float,
    *,
    spot_bid: float | None = None,
    spot_ask: float | None = None,
    fut_exec_bid: float | None = None,
    fut_exec_ask: float | None = None,
) -> bool:
    target = _required(target_price, "target_price")
    if route == "future_ask_spot_taker":
        return target > _required(fut_exec_bid, "fut_exec_bid")
    if route == "spot_bid_future_taker":
        return target < _required(spot_ask, "spot_ask")
    if route == "future_bid_spot_taker":
        return target < _required(fut_exec_ask, "fut_exec_ask")
    if route == "spot_ask_future_taker":
        return target > _required(spot_bid, "spot_bid")
    raise ValueError(f"unknown route: {route}")


def price_in_ref_band(price: float, reference: float) -> bool:
    """Strict user-specified -9%/+8% maker-price gate."""
    if not all(math.isfinite(value) and value > 0 for value in (price, reference)):
        return False
    return reference * 0.91 < price < reference * 1.08


def _required(value: float | None, name: str) -> float:
    if value is None or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _high_price_boundary(
    market: PriceMarket,
    session_date: str | None,
) -> float:
    if market not in ("spot", "future"):
        raise ValueError(f"unknown price market: {market}")
    if session_date is not None:
        value = str(session_date)
        if len(value) != 8 or not value.isdigit():
            raise ValueError("session_date must be YYYYMMDD")
    if (
        market == "future"
        and session_date is not None
        and str(session_date) >= FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE
    ):
        return 2500.0
    return 1000.0
