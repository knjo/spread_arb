"""Portfolio accounting and mandatory hedge execution for the v19 replay."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, isfinite
from typing import Callable

from .causal_lookup import CLOSE_SECOND, SECOND, DailySnapshot, ResolvedOutcome
from .decide import EntryDecider
from .execution import (CapacityLedger, PrintedVolumeQueue, QueueOrder, TakerDepth,
                        nominal_cents, realized_pnl)
from .market import Contract, MarketDay, previous_tick, tick_i
from .capacity_policy import activate_parked, admission_limit
from .ev_rules import cell_gate, cell_of, should_cross

S2_ENTRY_RESIDUAL_BP = 25.0   # new S2 quote needs held basis >= anchor + 25
S2_HOLD_FLOOR_BP = 20.0       # a working S2 quote is withdrawn below anchor + 20
CROSS_FROM_SECOND = 10_800    # taker-cross rule evaluated from 12:00 onward
MAKER_WITHDRAW_SECOND = 15_480  # 13:18: all maker orders withdrawn


@dataclass
class Position:
    id: str
    contract: Contract
    stream: str
    quote_ns: int
    quote_second: int
    quote_ab: float
    anchor: float
    entry_day: str
    entry_session: int
    state: str = "quoting"
    entry_fill_ns: int | None = None
    hedged_ns: int | None = None
    close_ns: int | None = None
    close_day: str | None = None
    spot_buy_qty: int = 0
    spot_buy_cash: int = 0
    future_sell_qty: int = 0
    future_sell_cash: int = 0
    spot_sell_qty: int = 0
    spot_sell_cash: int = 0
    future_buy_qty: int = 0
    future_buy_cash: int = 0
    pnl_twd: float = 0.0
    pnl_bp: float = 0.0
    actual_ab: float | None = None
    exit_order: str | None = None
    hedge_kind: str | None = None
    hedge_deadline: int | None = None
    hedge_breached: bool = False
    continuity_blocked: str | None = None
    source_intent: str | None = None
    cross_requested: bool = False
    exit_kind: str | None = None     # forced close reason: taker_cross / corporate_close
    close_kind: str | None = None    # how the pair actually ended (set by _close)
    s2_invalid_ns: int | None = None


class Portfolio:
    def __init__(self, name: str, cap_twd: int, *, use_ev: bool, use_bpday: bool,
                 shadow: bool = False, hedge_ms: int = 50, cancel_ms: int = 0,
                 park_spot: bool = False, overnight_target_twd: int | None = None,
                 split_ev: bool = True, event_quotes: bool = True, enable_cross: bool = False):
        self.name, self.shadow = name, shadow
        self.ledger = CapacityLedger(cap_twd * 100)
        self.use_ev, self.use_bpday = use_ev, use_bpday
        self.split_ev, self.event_quotes, self.enable_cross = split_ev, event_quotes, enable_cross
        self.hedge_delay, self.cancel_delay = hedge_ms * 1_000_000, cancel_ms * 1_000_000
        self.queue = PrintedVolumeQueue()
        self.depth = TakerDepth()
        self.positions: dict[str, Position] = {}
        self.active: set[str] = set()
        self.s2_live: dict[str, str] = {}
        self.cooldown: dict[str, int] = {}
        self.trace: list[dict] = []
        self.decisions: list[dict] = []
        self.resolved: list[ResolvedOutcome] = []
        self.rejected_by_day: dict[str, float] = {}
        self.rejected_intents: dict[str, float] = {}
        self.daily: list[dict] = []
        self._counter = 0
        self.park_spot = park_spot
        self.overnight_target_cents = overnight_target_twd * 100 if overnight_target_twd else None
        self.parked: set[str] = set()
        self.parked_instruments: dict[str, set[str]] = {}
        self.release_cache: dict[tuple, float] = {}
        self.desired_s1: dict[str, dict] = {}
        self.s1_attempts: dict[str, str] = {}
        self.fast_retry_gate = True
        self.repeg: set[str] = set()   # S2 names cancelled by a book event, to re-quote once cancel is effective
        self.event_cancels = 0
        self.requotes = 0
        self.crosses = 0

    def begin(self, market: MarketDay, session: int, snapshot: DailySnapshot,
              schedule: Callable) -> None:
        self.market, self.day, self.session = market, market.day, session
        self.schedule = schedule
        self.decider = EntryDecider(self.ledger.cap_cents, snapshot,
                                    use_ev=self.use_ev, use_bpday=self.use_bpday,
                                    split_ev=self.split_ev, calendar=getattr(market, "calendar", {}))
        self.rejected_by_day[self.day] = 0.0
        self.s2_live.clear()
        self.cooldown.clear()
        self.repeg.clear()
        self.event_cancels = self.requotes = self.crosses = 0
        self.parked.clear()
        self.parked_instruments.clear()
        self.release_cache.clear()
        self.desired_s1.clear()
        self.s1_attempts.clear()
        self.depth = TakerDepth()
        if self.queue.orders:
            raise AssertionError("maker orders survived session end")
        self.queue = PrintedVolumeQueue()
        for p in list(self.positions.values()):
            if p.state in {"closed", "cancelled"}:
                continue
            if p.contract.qc in getattr(market, "missing_carry", set()):
                if p.continuity_blocked is None:
                    p.continuity_blocked = self.day
                    self._event(p, market.start, "contract_continuity_blocked")
            if p.continuity_blocked:
                # Missing/reused contract identities require an explicit
                # corporate-action conversion. Keep all exposure and capital;
                # neither a new standard contract nor expiry can erase it.
                continue
            if p.contract.expiry < self.day and p.state == "paired":
                # Explicit C8 accounting convention: both legs marked at one
                # price after expiry. This is not an exchange execution claim.
                p.spot_sell_qty = p.spot_buy_qty
                p.spot_sell_cash = p.spot_buy_cash
                p.future_buy_qty = p.future_sell_qty
                p.future_buy_cash = p.spot_buy_cash
                self._close(p, market.start, "expiry_basis_zero_accounting")
                continue
            exact = getattr(market, "exact", {c.qc: c for c in market.contracts.values()})
            today = exact.get(p.contract.qc)
            if today is None or today.qc != p.contract.qc:
                raise ValueError(f"{self.day}: missing exact carry contract {p.contract.qc}")
            # References change daily; identity and actual expiry must not.
            if today.expiry != p.contract.expiry:
                raise ValueError("carry contract expiry changed")
            p.contract = today
            if p.state == "paired" and p.contract.vc in getattr(market, "corporate_entry_block", set()):
                self._event(p, market.start, "corporate_risk_exit")
                p.exit_kind = "corporate_close"
                self._start_hedge(p, "exit_spot_remainder", market.start)
                continue
            if p.hedge_kind:
                self.schedule(market.start, "hedge", self, p.id)

    def _event(self, p: Position, ns: int, kind: str, **values) -> None:
        self.trace.append(dict(day=self.day, ns=ns, position_id=p.id, stream=p.stream,
                               vc=p.contract.vc, qc=p.contract.qc, kind=kind, **values))

    def submit(self, *, intent_id: str, c: Contract, stream: str, price: int,
               ns: int, anchor: float, ab: float, eff_u: float,
               source_intent: str | None = None, execution_cost_bp: float = 0.0) -> None:
        if c.vc in getattr(self.market, "corporate_entry_block", set()):
            return
        instrument, side = ("F:" + c.qc, "sell") if stream == "S2" else ("S:" + c.vc, "buy")
        book = self.market.book(instrument, ns)
        if not book or not book.valid():
            return
        # Recheck maker legality when the request is actually submitted.
        if (side == "sell" and price <= book.bids[0][0]) or (side == "buy" and price >= book.asks[0][0]):
            return
        # S2 hedges can cost more than the current A1 after the maker fill.
        # Reserve the whole daily upper-bound spot nominal, then shrink only
        # after the actual hedge. S1's buy limit already bounds its cash.
        spot_limits = getattr(self.market, "spot_limits", {})
        upper = spot_limits.get(c.vc, (0, (c.spot_ref * 110 + 99) // 100))[1]
        reserve_px = upper if stream == "S2" else price
        reserve = nominal_cents(reserve_px, c.shares)
        parked = self.park_spot and stream == "S1" and price < book.bids[0][0]
        limit = admission_limit(self, ns)
        sec = int((ns - self.market.start) // SECOND)
        d = self.decider.decide(now_ns=ns, stream=stream, quote_second=sec,
                                eff_u=eff_u, ab=ab, reservation_cents=reserve,
                                committed_cents=self.ledger.committed_cents,
                                capacity_required=not parked, admission_cap_cents=limit, expiry=c.expiry,
                                execution_cost_bp=execution_cost_bp)
        self.decisions.append(dict(day=self.day, ns=ns, intent_id=intent_id, vc=c.vc, qc=c.qc,
                                   expiry=c.expiry,
                                   stream=stream, quote_price=price / 10_000, quote_ab=ab,
                                   eff_u=eff_u, reservation_cents=reserve,
                                   parked=parked, admission_cap_cents=limit,
                                   committed_cents=self.ledger.committed_cents, **asdict(d)))
        if not d.admit:
            if d.reason == "cap":
                # Count opportunity cost only if its independently replayed
                # shadow quote later fills. Never count each denied second.
                self.rejected_intents[intent_id] = max(d.est_bp, 0.0) * reserve / 100 / 10_000
            return
        if intent_id in self.positions:
            raise ValueError("duplicate intent")
        if not parked and not self.ledger.reserve(intent_id, reserve, ns):
            raise AssertionError("capacity changed between gate and reservation")
        p = Position(intent_id, c, stream, ns, sec, ab, anchor, self.day, self.session)
        p.source_intent = source_intent or intent_id
        self.positions[p.id] = p
        self.active.add(p.id)
        o = QueueOrder(p.id, instrument, side, price, 1 if stream == "S2" else c.shares,
                       ns, "entry", p.id)
        self.queue.add(o, book.ahead(side, price))
        if parked:
            self.parked.add(o.id)
            self.parked_instruments.setdefault(instrument, set()).add(o.id)
        if stream == "S2":
            self.s2_live[c.vc] = p.id
        self._event(p, ns, "quote", price=price / 10_000, quantity=o.remaining)

    def request_cancel(self, order_id: str, ns: int) -> None:
        o = self.queue.orders.get(order_id)
        if o is not None and o.cancel_ns is None:
            o.cancel_ns = ns + self.cancel_delay
            self.schedule(o.cancel_ns, "cancel", self, order_id)

    def cancel(self, order_id: str, ns: int) -> None:
        o = self.queue.cancel(order_id, ns)
        if o is None:
            return
        self.parked.discard(order_id)
        self.parked_instruments.get(o.instrument, set()).discard(order_id)
        p = self.positions[o.position_id]
        self._event(p, ns, "cancel", quantity=o.remaining, purpose=o.purpose)
        if o.purpose == "exit":
            p.exit_order = None
            if p.spot_sell_qty > 0 or p.cross_requested:
                self._start_hedge(p, "exit_spot_remainder", ns)
            return
        self.s2_live.pop(p.contract.vc, None) if p.stream == "S2" else None
        if p.spot_buy_qty == 0 and p.future_sell_qty == 0:
            p.state = "cancelled"
            self.active.remove(p.id)
            if p.id in self.ledger.amounts:
                self.ledger.release(p.id, ns, terminal=True)
            # Re-quotes are broadcast by Replay from the independent shadow
            # intent, so a capacity rejection still has a matching shadow fill.
        elif p.spot_buy_qty > 0 and p.future_sell_qty == 0:
            # A fractional S1 stock fill cannot be rounded to a whole future.
            # Cancel leaves, then flatten that stock exposure explicitly.
            self._start_hedge(p, "entry_rollback", ns)
        else:
            raise AssertionError("unexpected partially filled futures unit")

    def trade(self, instrument: str, ns: int, sequence: int, price: int, quantity: int) -> None:
        if instrument not in self.queue.by_instrument:
            return
        for o, qty in self.queue.trade(instrument, ns, sequence, price, quantity):
            p = self.positions[o.position_id]
            if o.purpose == "entry" and p.id not in self.ledger.amounts:
                reserve = nominal_cents(o.price, p.contract.shares)
                self.ledger.book_unreserved_fill(p.id, reserve, ns)
                self.parked.discard(o.id)
                self.parked_instruments.get(o.instrument, set()).discard(o.id)
                self._event(p, ns, "unreserved_fill", reservation_cents=reserve,
                            overrun_cents=max(0, self.ledger.committed_cents - self.ledger.cap_cents))
            self._event(p, ns, "maker_fill", purpose=o.purpose, price=o.price / 10_000,
                        quantity=qty, trade_sequence=sequence, trade_price=price / 10_000)
            if o.purpose == "entry":
                self.desired_s1.pop(p.source_intent, None)
                if p.entry_fill_ns is None:
                    p.entry_fill_ns = ns
                if p.stream == "S2":
                    if p.s2_invalid_ns is not None:
                        self._event(p, ns, "s2_invalid_fill", invalid_ns=p.s2_invalid_ns,
                                    lead_ns=ns-p.s2_invalid_ns,
                                    cancel_could_arrive=p.s2_invalid_ns+self.cancel_delay < ns,
                                    cancel_pending=o.cancel_ns is not None)
                    self.repeg.discard(p.contract.vc)
                    p.future_sell_qty += qty
                    p.future_sell_cash += o.price * qty * p.contract.shares
                    self.s2_live.pop(p.contract.vc, None)
                    self.cooldown[p.contract.vc] = ns + 60 * SECOND
                else:
                    p.spot_buy_qty += qty
                    p.spot_buy_cash += o.price * qty
                p.state = "entry_partial"
                if o.remaining == 0:
                    self._start_hedge(p, "entry_spot" if p.stream == "S2" else "entry_future", ns)
            else:
                p.spot_sell_qty += qty
                p.spot_sell_cash += o.price * qty
                if o.remaining == 0:
                    p.exit_order = None
                    self._start_hedge(p, "exit_future", ns)

    def _start_hedge(self, p: Position, kind: str, ns: int) -> None:
        p.hedge_kind = kind
        p.state = "hedging"
        p.hedge_deadline = ns + 5 * SECOND
        p.hedge_breached = False
        self.schedule(ns + self.hedge_delay, "hedge", self, p.id)

    def hedge(self, pid: str, ns: int) -> None:
        p = self.positions[pid]
        if p.continuity_blocked:
            return
        kind, c = p.hedge_kind, p.contract
        if kind is None:
            return
        spot = kind in {"entry_spot", "entry_rollback", "exit_spot_remainder"}
        instrument = ("S:" + c.vc) if spot else ("F:" + c.qc)
        side = "buy" if kind in {"entry_spot", "exit_future"} else "sell"
        qty = (c.shares - p.spot_buy_qty if kind == "entry_spot" else
               p.spot_buy_qty if kind == "entry_rollback" else
               p.spot_buy_qty - p.spot_sell_qty if kind == "exit_spot_remainder" else 1)
        ref = c.spot_ref if spot else c.fut_ref
        b = self.market.book(instrument, ns)
        # Entry signal buffers (-9%/+8%) must not veto a mandatory hedge.
        # Stock uses that day's exchange-published limits, also used to reserve
        # S2 cash. Standard futures use the 10% reference-price bound.
        bounds = ((ref * 90 + 99) // 100, ref * 110 // 100)
        if spot:
            bounds = getattr(self.market, "spot_limits", {}).get(c.vc, bounds)
        taken = self.depth.take(b, side, qty, ns, *bounds) if b else None
        if taken is None:
            if p.hedge_deadline is not None and ns >= p.hedge_deadline and not p.hedge_breached:
                p.hedge_breached = True
                self._event(p, ns, "hedge_timeout", purpose=kind, quantity=qty)
            series = self.market.series.get(instrument)
            nxt = series.next_ns(ns) if series else None
            retry = min(nxt, ns + SECOND) if nxt is not None else ns + SECOND
            if retry < self.market.end:
                self.schedule(retry, "hedge", self, p.id)
            return
        cash, filled = taken
        self._event(p, ns, "taker_fill", purpose=kind, price=cash / filled / 10_000,
                    quantity=filled, book_ns=b.ns, book_sequence=b.sequence)
        p.hedge_kind = None
        if kind == "entry_spot":
            p.spot_buy_qty += filled
            p.spot_buy_cash += cash
        elif kind == "entry_future":
            p.future_sell_qty += filled
            p.future_sell_cash += cash * c.shares
        elif kind == "exit_future":
            p.future_buy_qty += filled
            p.future_buy_cash += cash * c.shares
            self._close(p, ns, p.exit_kind or "maker_exit")
            return
        elif kind in {"entry_rollback", "exit_spot_remainder"}:
            p.spot_sell_qty += filled
            p.spot_sell_cash += cash
            if kind == "entry_rollback":
                self._close(p, ns, "partial_entry_rollback")
            else:
                self._start_hedge(p, "exit_future", ns)
            return
        p.state = "paired"
        p.hedged_ns = ns
        p.actual_ab = (p.future_sell_cash / p.spot_buy_cash - 1) * 10_000
        self.ledger.confirm_nominal(p.id, (p.spot_buy_cash + 99) // 100, ns)
        self._event(p, ns, "paired", actual_ab=p.actual_ab)

    def _close(self, p: Position, ns: int, kind: str) -> None:
        if p.spot_buy_qty != p.spot_sell_qty or p.future_sell_qty != p.future_buy_qty:
            raise AssertionError("cannot close before both legs finish")
        if p.entry_fill_ns is None or ns <= p.entry_fill_ns:
            raise AssertionError("exit must follow entry")
        p.close_ns, p.close_day, p.state = ns, self.day, "closed"
        p.close_kind = kind
        self.active.remove(p.id)
        p.pnl_twd, p.pnl_bp = realized_pnl(p.spot_buy_cash, p.spot_sell_cash,
                                          p.future_sell_cash, p.future_buy_cash,
                                          entry_day=p.entry_day, exit_day=self.day)
        self.ledger.release(p.id, ns, terminal=True)
        self._event(p, ns, kind, pnl_twd=p.pnl_twd, pnl_bp=p.pnl_bp)
        self.resolved.append(ResolvedOutcome(p.id, p.stream, p.quote_ab, p.entry_day,
                                             self.day, ns, p.pnl_bp,
                                             max(self.session - p.entry_session, .15), kind))

    # ------------------------------------------------------------------
    # Event-driven S2 quote maintenance (2026-09-09). A spot ask change is
    # the information that changes the basis my working futures quote locks
    # in; re-evaluate immediately instead of at the next second boundary.
    # Cancellation still takes cancel_ms to become effective, so a fill that
    # arrives in that window stays booked and must be hedged.
    # ------------------------------------------------------------------
    def book_update(self, instrument: str, ns: int) -> None:
        vc = (instrument[2:] if instrument.startswith("S:") else
              getattr(self.market, "future_to_vc", {}).get(instrument[2:]))
        if vc is None:
            return
        pid = self.s2_live.get(vc)
        if pid is None:
            return
        o = self.queue.orders.get(pid)
        if o is None or o.cancel_ns is not None:
            return
        sec = int((ns - self.market.start) // SECOND)
        if not 0 <= sec < CLOSE_SECOND:
            return
        p = self.positions[o.position_id]
        pair = self.market.pair(p.contract, ns)
        an = float(self.market.signals[vc]["an"][sec])
        held_basis = (o.price / pair[0].asks[0][0] - 1) * 10_000 if pair else None
        reason = "market" if pair is None or not isfinite(an) else "basis"
        valid = (held_basis is not None and isfinite(an) and held_basis > 0 and
                 held_basis - an >= S2_HOLD_FLOOR_BP)
        if valid:
            # The working order retains its frozen exit anchor. Re-pricing
            # creates a new order with a new anchor, never rewrites this one.
            d = self.decider.decide(now_ns=ns, stream="S2", quote_second=sec,
                eff_u=held_basis-p.anchor, ab=held_basis,
                reservation_cents=self.ledger.amounts[p.id],
                committed_cents=self.ledger.committed_cents, capacity_required=False,
                expiry=p.contract.expiry)
            valid, reason = d.admit, d.reason
        if valid and sec < 14_000:
            return
        if p.s2_invalid_ns is None:
            p.s2_invalid_ns = ns
            self._event(p, ns, "s2_first_invalid", reason=reason, held_basis=held_basis,
                        anchor=an, event_quotes=self.event_quotes)
        if not self.event_quotes:
            return
        self.event_cancels += 1
        self._event(p, ns, "event_cancel", held_basis=held_basis, anchor=an,
                    reason=reason, effective_ns=ns+self.cancel_delay)
        self.repeg.add(vc)
        self.request_cancel(pid, ns)

    def _requote_s2(self, vc: str, ns: int, intent_id: str | None = None) -> None:
        """Restore the required basis, including a less aggressive futures ask."""
        if vc in self.s2_live or ns < self.cooldown.get(vc, 0):
            return
        sec = int((ns - self.market.start) // SECOND)
        if not 300 <= sec < 14_000:
            return
        c = self.market.contracts.get(vc)
        if c is None:
            return
        pair = self.market.pair(c, ns)
        if pair is None:
            return
        spot, fut = pair
        an = float(self.market.signals[vc]["an"][sec])
        if not isfinite(an):
            return
        minimum = ceil(spot.asks[0][0] * (1 + max(an+S2_ENTRY_RESIDUAL_BP, 0.01)/10_000))
        step = tick_i(minimum)
        minimum = ((minimum+step-1)//step)*step
        desired = max(previous_tick(fut.asks[0][0]), minimum)
        if desired > c.fut_ref * 1.08:
            return
        if desired > fut.asks[-1][0]:
            # Outside visible depth, zero displayed size does not mean zero
            # queue ahead. Wait until the replacement price is observable.
            return
        ab = (desired / spot.asks[0][0] - 1) * 10_000
        if desired <= fut.bids[0][0] or ab <= 0 or ab - an < S2_ENTRY_RESIDUAL_BP:
            return
        self.requotes += 1
        self.submit(intent_id=intent_id or f"S2/{self.day}/{vc}/e{ns}", c=c, stream="S2", price=int(desired),
                    ns=ns, anchor=an, ab=float(ab), eff_u=float(ab - an))

    def second(self, sec: int, *, s2_intents: dict[str, str] | None = None) -> None:
        ns = self.market.start + sec * SECOND
        activate_parked(self, ns)
        for source, row in list(self.desired_s1.items()):
            current = self.s1_attempts.get(source, source)
            if ns >= min(row["nominal_stop_time_ns"], self.market.start + 14_000*SECOND):
                self.desired_s1.pop(source, None)
            elif current not in self.queue.orders:
                self.offer_s1(row, ns, retry=True)
        if self.overnight_target_cents is not None or self.ledger.committed_cents > self.ledger.cap_cents:
            if self.ledger.committed_cents > admission_limit(self, ns):
                for oid, order in list(self.queue.orders.items()):
                    if order.purpose == "entry" and oid not in self.parked:
                        self.request_cancel(oid, ns)
        # Quote/cancel/exit choices happen once a second. Fill and hedge events
        # between seconds retain their actual receive timestamps.
        for vc, pid in list(self.s2_live.items()):
            if self.event_quotes:
                self.book_update("S:"+vc, ns)
            o = self.queue.orders.get(pid)
            if o is None:
                continue
            b = self.market.signals[vc]
            held_basis = (o.price / max(b["sa"][sec], 1) - 1) * 10_000
            residual = held_basis - b["an"][sec]
            if sec >= 14_000 or not b["valid"][sec] or residual < S2_HOLD_FLOOR_BP or held_basis <= 0:
                if self.event_quotes and sec < 14_000:
                    self.repeg.add(vc)
                self.request_cancel(pid, ns)
        for o in list(self.queue.orders.values()):
            if o.purpose != "exit" or o.cancel_ns is not None:
                continue
            p = self.positions[o.position_id]
            future = self.market.book("F:" + p.contract.qc, ns)
            # Validate the still-working absolute sell price using information
            # available NOW. This guard can cancel an order; it can never undo
            # a fill arriving before cancellation becomes effective.
            executable = bool(future and future.valid() and sum(q for _, q in future.asks) >= 1)
            if not executable or (future.asks[0][0] / o.price - 1) * 10_000 > p.anchor - 5.0:
                self.request_cancel(o.id, ns)
        snapshot = self.decider.snapshot
        for pid in sorted(self.active):
            p = self.positions[pid]
            if p.continuity_blocked or p.cross_requested:
                continue
            if p.state != "paired" or p.hedged_ns is None or p.hedged_ns >= ns:
                continue
            if sec >= MAKER_WITHDRAW_SECOND:
                continue
            if p.exit_order is not None and sec < CROSS_FROM_SECOND:
                continue
            pair = self.market.pair(p.contract, ns, signal_buffer=False)
            if pair is None:
                continue
            spot, future = pair
            bbt = (future.asks[0][0] / spot.bids[0][0] - 1) * 10_000
            d_bp = bbt - (p.anchor - 5.0)   # taker exit cost above the frozen maker target
            # Same EV frame as entry: keep the maker exit while its fill chance
            # pays for the distance; otherwise cross now (ev_rules.should_cross).
            if (self.enable_cross and sec >= CROSS_FROM_SECOND and d_bp > 0 and future.asks[0][1] >= 1 and
                    should_cross(d_bp, snapshot.p_fill(sec), snapshot.lam_bp, p.entry_day == self.day)):
                p.cross_requested, p.exit_kind = True, "taker_cross"
                self.crosses += 1
                self._event(p, ns, "cross_decision", d_bp=d_bp, p_fill=snapshot.p_fill(sec),
                            lam_bp=snapshot.lam_bp, entry_today=p.entry_day == self.day)
                if p.exit_order is not None:
                    self.request_cancel(p.exit_order, ns)
                else:
                    self._start_hedge(p, "exit_spot_remainder", ns)
                continue
            if p.exit_order is not None or bbt > p.anchor - 5.0:
                continue
            self._counter += 1
            oid = f"{p.id}/exit/{self.day}/{self._counter}"
            o = QueueOrder(oid, "S:" + p.contract.vc, "sell", spot.asks[0][0],
                           p.spot_buy_qty - p.spot_sell_qty, ns, "exit", p.id)
            self.queue.add(o, spot.ahead("sell", o.price))
            p.exit_order = oid
            self._event(p, ns, "exit_quote", price=o.price / 10_000, quantity=o.remaining)
        if sec == MAKER_WITHDRAW_SECOND:
            for oid in list(self.queue.orders):
                self.request_cancel(oid, ns)

    def offer_s2(self, vc: str, intent_id: str, sec: int) -> None:
        ns = self.market.start + sec * SECOND
        if vc in self.s2_live or ns < self.cooldown.get(vc, 0):
            return
        b, c = self.market.signals[vc], self.market.contracts[vc]
        if not b["ok"][sec]:
            return
        self.submit(intent_id=intent_id, c=c, stream="S2", price=int(b["desired"][sec]), ns=ns,
                    anchor=float(b["an"][sec]), ab=float(b["ab"][sec]),
                    eff_u=float(b["ab"][sec] - b["an"][sec]))

    def stop_s1(self, source: str, ns: int) -> None:
        self.desired_s1.pop(source, None)
        self.request_cancel(self.s1_attempts.get(source, source), ns)

    def offer_s1(self, row: dict, ns: int, *, retry: bool = False) -> None:
        vc = row["ValueCode"]
        source = row["raw_order_fact_id"]
        if self.park_spot and not retry:
            self.desired_s1[source] = row
        if vc not in self.market.contracts:
            return
        c = self.market.contracts[vc]
        if row["QuoteCode"] != c.qc:
            raise ValueError("S1 intent references a different contract")
        sec = int((ns - self.market.start) // SECOND)
        if not 0 <= sec < 14_000:
            return
        b = self.market.signals[vc]
        if not b["valid"][sec] or not isfinite(b["an"][sec]):
            return
        px = round(row["target_price"] * 10_000)
        ab = (b["fb"][sec] / px - 1) * 10_000
        if retry:
            book = self.market.book("S:"+vc, ns)
            if not book or not book.valid():
                return
            if px >= book.bids[0][0] and self.ledger.committed_cents + nominal_cents(px,c.shares) > admission_limit(self,ns):
                return
        intent = source
        if intent in self.positions:
            if not retry or self.positions[intent].entry_fill_ns is not None:
                return
            self._counter += 1
            intent = f"{source}/retry/{self._counter}"
        # A frozen rejecting bpday cell cannot admit this retry. Avoid
        # allocating another identical rejected decision record each second.
        # Keep counter advancement identical; orders, fills, lambda and capital
        # are unchanged. The flag permits an exact execution-equivalence audit.
        if retry and self.fast_retry_gate and self.use_bpday:
            stats = self.decider.snapshot.cells.get(cell_of("S1", ab), (0., 0., 0))
            if not cell_gate(*stats):
                return
        self.submit(intent_id=intent, c=c, stream="S1", price=px, ns=ns,
                    anchor=float(b["an"][sec]), ab=ab, eff_u=float(ab - b["an"][sec]), source_intent=source)
        if intent in self.queue.orders:
            self.s1_attempts[source] = intent

    def finish(self) -> dict:
        open_positions = [p for p in self.positions.values() if p.state not in {"closed", "cancelled"}]
        today = [p for p in self.positions.values() if p.entry_day == self.day and p.entry_fill_ns is not None]
        closed = [p for p in self.positions.values() if p.close_day == self.day]
        if self.queue.orders:
            raise AssertionError("session ended with live maker orders")
        row = dict(day=self.day, fills=len(today), fills_s1=sum(p.stream == "S1" for p in today),
                   fills_s2=sum(p.stream == "S2" for p in today),
                   negative_basis=sum(p.actual_ab is not None and p.actual_ab <= 0 for p in today),
                   same_day=sum(p.close_day == self.day for p in today),
                   realized_twd=sum(p.pnl_twd for p in closed),
                   carry_twd=self.ledger.committed_cents / 100,
                   open_positions=len(open_positions),
                   unhedged=sum(p.state != "paired" for p in open_positions),
                   continuity_blocked=sum(p.continuity_blocked is not None for p in open_positions),
                   unreserved_fills=sum(t["kind"] == "unreserved_fill" for t in self.trace),
                   cap_overrun_events=sum(t["kind"] == "unreserved_fill" and t.get("overrun_cents", 0) > 0
                                          for t in self.trace),
                   rejected_cap=sum(d["day"] == self.day and d["reason"] == "cap" for d in self.decisions),
                   event_cancels=self.event_cancels, requotes=self.requotes, crosses=self.crosses,
                   closed_maker_exit=sum(p.close_kind == "maker_exit" for p in closed),
                   closed_taker_cross=sum(p.close_kind == "taker_cross" for p in closed),
                   closed_expiry=sum(p.close_kind == "expiry_basis_zero_accounting" for p in closed))
        self.daily.append(row)
        return row
