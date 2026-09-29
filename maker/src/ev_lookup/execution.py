"""Small execution primitives; no policy, future labels, or target-price PnL."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


def price_i(price: float) -> int:
    return round(price * 10_000)


def nominal_cents(price: int, shares: int) -> int:
    return (price * shares + 99) // 100


class CapacityLedger:
    """Every submitted order owns a reservation until cancel or BOTH legs close."""

    def __init__(self, cap_cents: int):
        self.cap_cents = cap_cents
        self.amounts: dict[str, int] = {}
        self.committed_cents = 0
        self.peak_cents = 0
        self.last_ns = 0
        self.events: list[dict] = []

    def _record(self, key: str, now_ns: int, kind: str, delta: int) -> None:
        if now_ns < self.last_ns:
            raise ValueError("capacity events must be chronological")
        self.last_ns = now_ns
        prior = self.committed_cents
        self.committed_cents += delta
        if self.committed_cents < 0 or (self.committed_cents > self.cap_cents
                and kind != "unreserved_fill" and not (prior > self.cap_cents and delta <= 0)):
            raise AssertionError("hard capacity violated")
        self.peak_cents = max(self.peak_cents, self.committed_cents)
        self.events.append(dict(id=key, ns=now_ns, kind=kind, delta_cents=delta,
                                committed_cents=self.committed_cents))

    def reserve(self, key: str, cents: int, now_ns: int) -> bool:
        if key in self.amounts or cents <= 0:
            raise ValueError("invalid or duplicate reservation")
        if self.committed_cents + cents > self.cap_cents:
            return False
        self.amounts[key] = cents
        self._record(key, now_ns, "reserve", cents)
        return True

    def book_unreserved_fill(self, key: str, cents: int, now_ns: int) -> None:
        """An exchange fill is irrevocable, even after an unreserved-quote race."""
        if key in self.amounts or cents <= 0:
            raise ValueError("invalid unreserved fill")
        self.amounts[key] = cents
        self._record(key, now_ns, "unreserved_fill", cents)

    def confirm_nominal(self, key: str, cents: int, now_ns: int) -> None:
        old = self.amounts[key]
        if not 0 < cents <= old:
            raise ValueError("fill exceeded pre-reserved maximum nominal")
        self.amounts[key] = cents
        self._record(key, now_ns, "hedged", cents - old)

    def release(self, key: str, now_ns: int, *, terminal: bool) -> None:
        if not terminal:
            raise ValueError("cannot release an open or unhedged position")
        cents = self.amounts.pop(key)
        self._record(key, now_ns, "release", -cents)


@dataclass
class QueueOrder:
    id: str
    instrument: str
    side: str
    price: int
    remaining: int
    placed_ns: int
    purpose: str
    position_id: str
    cancel_ns: int | None = None


@dataclass
class PriceQueue:
    # Interleave external queue segments with our orders. A later submission
    # cannot jump ahead of displayed volume that joined after an earlier order.
    nodes: deque = field(default_factory=deque)


class PrintedVolumeQueue:
    """Conservative FIFO: cancellations never gift queue priority.

    One trade's quantity is spent ONCE, including external queue ahead and
    all our orders. Trade-through does not create unlimited volume. This is
    a specified replay approximation, not an exchange acknowledgement model.
    """

    def __init__(self):
        self.orders: dict[str, QueueOrder] = {}
        self.queues: dict[tuple[str, str, int], PriceQueue] = {}
        self.by_instrument: dict[str, set[tuple[str,str,int]]] = {}
        self.seen_trades: set[tuple[str, int, int]] = set()

    def add(self, order: QueueOrder, displayed_ahead: int) -> None:
        if order.id in self.orders or order.remaining <= 0 or displayed_ahead < 0:
            raise ValueError("invalid maker order")
        if order.side not in {"buy", "sell"}:
            raise ValueError("invalid maker side")
        key = (order.instrument, order.side, order.price)
        q = self.queues.setdefault(key, PriceQueue())
        self.by_instrument.setdefault(order.instrument, set()).add(key)
        external_left = sum(n for n in q.nodes if isinstance(n, int))
        if displayed_ahead > external_left:
            q.nodes.append(displayed_ahead - external_left)
        q.nodes.append(order.id)
        self.orders[order.id] = order

    def cancel(self, order_id: str, now_ns: int) -> QueueOrder | None:
        order = self.orders.get(order_id)
        if order is None:
            return None
        if now_ns < order.placed_ns:
            raise ValueError("cancel precedes order")
        order = self.orders.pop(order_id)
        key = (order.instrument, order.side, order.price)
        q = self.queues[key]
        q.nodes.remove(order_id)
        if not any(isinstance(n, str) for n in q.nodes):
            self._remove_queue(key)
        return order

    def _remove_queue(self, key: tuple[str,str,int]) -> None:
        del self.queues[key]
        self.by_instrument[key[0]].remove(key)
        if not self.by_instrument[key[0]]:
            del self.by_instrument[key[0]]

    def trade(self, instrument: str, ns: int, sequence: int, price: int,
              quantity: int) -> list[tuple[QueueOrder, int]]:
        trade_id = (instrument, ns, sequence)
        if trade_id in self.seen_trades:
            raise ValueError("duplicate trade cursor")
        self.seen_trades.add(trade_id)
        budget = quantity
        fills = []
        keys = [k for k in self.by_instrument.get(instrument, ()) if
                (price >= k[2] if k[1] == "sell" else price <= k[2])]
        # Price priority within each side. If old own buy/sell limits overlap,
        # this conservative convention spends ambiguous print volume on sells
        # first, never independently on both sides.
        keys.sort(key=lambda k: (0 if k[1] == "sell" else 1,
                                 k[2] if k[1] == "sell" else -k[2]))
        for key in keys:
            q = self.queues[key]
            first_order = next((self.orders[n] for n in q.nodes if isinstance(n, str)), None)
            if first_order is None or first_order.placed_ns >= ns:
                continue
            while q.nodes and budget > 0:
                node = q.nodes[0]
                if isinstance(node, int):
                    used = min(node, budget)
                    budget -= used
                    q.nodes.popleft()
                    if node > used:
                        q.nodes.appendleft(node - used)
                    continue
                order = self.orders[node]
                # Events sharing a timestamp were already observed when a new
                # quote was submitted; they cannot fill that new quote.
                if order.placed_ns >= ns:
                    break
                used = min(order.remaining, budget)
                order.remaining -= used
                budget -= used
                fills.append((order, used))
                if order.remaining == 0:
                    q.nodes.popleft()
                    del self.orders[node]
            if not any(isinstance(n, str) for n in q.nodes):
                self._remove_queue(key)
        if sum(qty for _, qty in fills) > quantity:
            raise AssertionError("printed volume reused")
        return fills


@dataclass(frozen=True)
class Book:
    instrument: str
    ns: int
    sequence: int
    bids: tuple[tuple[int, int], ...]
    asks: tuple[tuple[int, int], ...]
    formal: bool

    def valid(self) -> bool:
        return bool(self.formal and self.bids and self.asks and
                    0 < self.bids[0][0] < self.asks[0][0])

    def ahead(self, side: str, price: int) -> int:
        levels = self.bids if side == "buy" else self.asks
        return sum(q for p, q in levels if p == price)


class TakerDepth:
    """L1-L5 full-quantity execution; shared depth is not reused per snapshot."""

    def __init__(self):
        self.used: dict[tuple[str, int, int, str, int], int] = {}

    def take(self, book: Book, side: str, quantity: int, now_ns: int,
             min_price: int = 1, max_price: int = 10**15) -> tuple[int, int] | None:
        if book.ns > now_ns:
            raise ValueError("hedge book is from the future")
        if quantity <= 0 or not book.valid():
            return None
        levels = book.asks if side == "buy" else book.bids
        left, cash, plan = quantity, 0, []
        for price, shown in levels:
            if not min_price <= price <= max_price:
                break
            key = (book.instrument, book.ns, book.sequence, side, price)
            qty = min(left, max(shown - self.used.get(key, 0), 0))
            cash += price * qty
            left -= qty
            plan.append((key, qty))
            if left == 0:
                break
        if left:
            return None
        for key, qty in plan:
            self.used[key] = self.used.get(key, 0) + qty
        return cash, quantity


def realized_pnl(spot_buy_cash: int, spot_sell_cash: int,
                 future_sell_cash: int, future_buy_cash: int,
                 *, entry_day: str, exit_day: str) -> tuple[float, float]:
    """All cash values use 1e-4 TWD; cost profile remains 20/34 bp."""
    if spot_buy_cash <= 0:
        raise ValueError("a complete pair needs a spot cost basis")
    gross = (spot_sell_cash - spot_buy_cash + future_sell_cash - future_buy_cash) / 10_000
    nominal = spot_buy_cash / 10_000
    cost_bp = 20.0 if entry_day == exit_day else 34.0
    pnl = gross - nominal * cost_bp / 10_000
    return pnl, pnl / nominal * 10_000
