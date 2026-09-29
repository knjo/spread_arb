"""Observable S2 hedge depth and a fixed, disclosed adverse-price scenario.

All prices are integer 1e-4 TWD. The scenario moves each visible ask upward by
legal spot ticks. It changes quote EV only; actual hedges still use raw depth.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .execution import Book, TakerDepth
from .market import tick_i


@dataclass(frozen=True)
class LiquidityRule:
    enabled: bool = True
    min_a1_multiple: float = 0.0
    adverse_probability: float = 0.5
    adverse_ticks: int = 1

    def __post_init__(self):
        if not isfinite(self.min_a1_multiple) or self.min_a1_multiple < 0:
            raise ValueError("A1 multiple must be finite and nonnegative")
        if not isfinite(self.adverse_probability) or not 0 <= self.adverse_probability <= 1:
            raise ValueError("adverse probability must be within [0, 1]")
        if not isinstance(self.adverse_ticks, int) or self.adverse_ticks < 0:
            raise ValueError("adverse ticks must be a nonnegative integer")


def hedge_risk(book: Book | None, quantity: int, future_price: int, now_ns: int,
               rule: LiquidityRule, depth: TakerDepth, *, upper_price: int = 10**15) -> dict:
    """Preview remaining depth without spending it; never inspect a future book."""
    if quantity <= 0 or future_price <= 0:
        raise ValueError("positive hedge quantity and futures price required")
    if book is not None and book.ns > now_ns:
        raise ValueError("liquidity information is from the future")
    result = dict(reason="hedge_book", book_ns=book.ns if book else None,
                  book_sequence=book.sequence if book else None, hedge_shares=quantity,
                  a1_price=None, a1_shares=0, a1_multiple=0.0, depth_vwap=None,
                  adverse_vwap=None, depth_cost_bp=0.0, adverse_cost_bp=0.0,
                  execution_cost_bp=0.0, raw_basis=None)
    if book is None or not book.valid():
        return result
    asks = [(px, max(qty-depth.used.get((book.instrument, book.ns, book.sequence, "buy", px), 0), 0))
            for px, qty in book.asks if px <= upper_price]
    if not asks or asks[0][0] != book.asks[0][0]:
        return result
    a1, size = asks[0]
    result.update(a1_price=a1/10_000, a1_shares=size, a1_multiple=size/quantity,
                  raw_basis=(future_price/a1-1)*10_000)
    left, cash, adverse_cash = quantity, 0, 0
    for px, qty in asks:
        take = min(left, qty)
        stressed = px
        for _ in range(rule.adverse_ticks):
            stressed += tick_i(stressed)
        cash += px*take
        adverse_cash += stressed*take
        left -= take
        if not left:
            break
    if left:
        result["reason"] = "hedge_depth"
        return result
    vwap, adverse = cash/quantity, adverse_cash/quantity
    depth_basis = (future_price/vwap-1)*10_000
    adverse_basis = (future_price/adverse-1)*10_000
    depth_cost = max(0.0, result["raw_basis"]-depth_basis)
    adverse_cost = max(0.0, rule.adverse_probability*(depth_basis-adverse_basis))
    result.update(reason="a1_depth" if size < rule.min_a1_multiple*quantity else "ok",
                  depth_vwap=vwap/10_000, adverse_vwap=adverse/10_000,
                  depth_cost_bp=depth_cost, adverse_cost_bp=adverse_cost,
                  execution_cost_bp=depth_cost+adverse_cost)
    return result
