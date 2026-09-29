"""S2 inside-only experiment with observable liquidity and EV execution buffers."""
from __future__ import annotations

from math import isfinite

from ..ev_lookup_cost.causal_lookup import CLOSE_SECOND, SECOND
from ..ev_lookup_cost.market import previous_tick
from .execution_portfolio import Portfolio, S2_ENTRY_RESIDUAL_BP, S2_HOLD_FLOOR_BP
from ..ev_lookup_cost.s2_liquidity import LiquidityRule, hedge_risk


class LiquidityPortfolio(Portfolio):
    def __init__(self, *args, liquidity_rule: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.event_quotes:
            raise ValueError("inside/liquidity maintenance requires book-event cancellation")
        self.liquidity_rule = LiquidityRule(**(liquidity_rule or {}))

    def _risk(self, c, price, ns):
        upper = getattr(self.market, "spot_limits", {}).get(c.vc, (0, (c.spot_ref*110+99)//100))[1]
        return hedge_risk(self.market.book("S:"+c.vc, ns), c.shares, price, ns,
                          self.liquidity_rule, self.depth, upper_price=upper)

    def submit(self, **kwargs):
        if kwargs["stream"] != "S2":
            return super().submit(**kwargs)
        c, ns, price = kwargs["c"], kwargs["ns"], kwargs["price"]
        future = self.market.book("F:"+c.qc, ns)
        # No higher-priced fallback: each newly sent order improves A1 by one
        # legal tick, with zero displayed queue ahead. Acknowledgment latency
        # remains the same explicit approximation as in v21.
        if (not future or not future.valid() or price != previous_tick(future.asks[0][0])
                or price <= future.bids[0][0]):
            return
        risk = self._risk(c, price, ns) if self.liquidity_rule.enabled else None
        if risk is not None and risk["reason"] != "ok":
            self.trace.append(dict(day=self.day, ns=ns, position_id=kwargs["intent_id"],
                stream="S2", vc=c.vc, qc=c.qc, kind="s2_liquidity_reject", **risk))
            return
        before = len(self.decisions)
        super().submit(**kwargs, execution_cost_bp=risk["execution_cost_bp"] if risk else 0.0)
        if len(self.decisions) > before:
            self.decisions[-1].update(future_bid=future.bids[0][0]/10_000,
                future_ask=future.asks[0][0]/10_000, future_book_ns=future.ns,
                **{"liquidity_"+k: v for k, v in (risk or {}).items()})

    def _requote_s2(self, vc, ns, intent_id=None):
        if vc in self.s2_live or ns < self.cooldown.get(vc, 0):
            return
        sec = int((ns-self.market.start)//SECOND)
        if not 300 <= sec < 14_000:
            return
        c = self.market.contracts.get(vc)
        pair = self.market.pair(c, ns) if c else None
        if pair is None:
            return
        spot, future = pair
        anchor = float(self.market.signals[vc]["an"][sec])
        desired = previous_tick(future.asks[0][0])
        basis = (desired/spot.asks[0][0]-1)*10_000
        if (not isfinite(anchor) or desired <= future.bids[0][0] or desired > c.fut_ref*1.08
                or basis <= 0 or basis-anchor < S2_ENTRY_RESIDUAL_BP):
            return
        self.requotes += 1
        self.submit(intent_id=intent_id or f"S2/{self.day}/{vc}/e{ns}", c=c, stream="S2",
                    price=desired, ns=ns, anchor=anchor, ab=basis, eff_u=basis-anchor)

    def book_update(self, instrument, ns):
        # Exit maintenance is independent of whether an S2 entry is working.
        if self.exit_event_guard and instrument.startswith("F:"):
            self._guard_exit_orders(instrument[2:], ns)
        vc = (instrument[2:] if instrument.startswith("S:") else
              getattr(self.market, "future_to_vc", {}).get(instrument[2:]))
        pid = self.s2_live.get(vc)
        order = self.queue.orders.get(pid)
        if order is None or order.cancel_ns is not None:
            return
        sec = int((ns-self.market.start)//SECOND)
        if not 0 <= sec < CLOSE_SECOND:
            return
        p = self.positions[pid]
        pair = self.market.pair(p.contract, ns)
        anchor = float(self.market.signals[vc]["an"][sec])
        basis = (order.price/pair[0].asks[0][0]-1)*10_000 if pair else None
        risk = None
        reason = "market"
        if pair is not None and isfinite(anchor):
            spot, future = pair
            # Public history excludes our synthetic order. An external ask
            # joining our price later queues behind it; only a strictly lower
            # ask removes our best-price priority.
            if not future.bids[0][0] < order.price <= future.asks[0][0]:
                reason = "first_priority"
            elif basis <= 0 or basis-anchor < S2_HOLD_FLOOR_BP:
                reason = "basis"
            elif self.repeg_drop_bp is not None and basis < p.quote_ab-self.repeg_drop_bp:
                reason = "drift"
            else:
                risk = self._risk(p.contract, order.price, ns) if self.liquidity_rule.enabled else None
                reason = risk["reason"] if risk else "ok"
                if reason == "ok":
                    d = self.decider.decide(now_ns=ns, stream="S2", quote_second=sec,
                        eff_u=basis-p.anchor, ab=basis,
                        reservation_cents=self.ledger.amounts.get(pid,self.order_reservation(p,order.price)),
                        committed_cents=self.ledger.committed_cents, capacity_required=False,
                        expiry=p.contract.expiry, execution_cost_bp=risk["execution_cost_bp"] if risk else 0.0,
                        spread_bp=(future.asks[0][0]-future.bids[0][0])/future.bids[0][0]*10_000)
                    reason = d.reason
        if sec >= 14_000:
            reason = "cutoff"
        if reason == "ok":
            return
        details = {"liquidity_"+k: v for k, v in (risk or {}).items()}
        if p.s2_invalid_ns is None:
            p.s2_invalid_ns = ns
            self._event(p, ns, "s2_first_invalid", reason=reason, held_basis=basis,
                        anchor=anchor, event_quotes=True, **details)
        self.event_cancels += 1
        if reason == "drift":
            self.drift_cancels += 1
        self._event(p, ns, "event_cancel", held_basis=basis, anchor=anchor,
                    reason=reason, effective_ns=ns+self.cancel_delay, **details)
        self.repeg.add(vc)
        self.request_cancel(pid, ns)

    def trade(self, instrument, ns, sequence, price, qty):
        vc = getattr(self.market, "future_to_vc", {}).get(instrument[2:]) if instrument.startswith("F:") else None
        pid = self.s2_live.get(vc)
        super().trade(instrument, ns, sequence, price, qty)
        p = self.positions.get(pid)
        if p is not None and p.entry_fill_ns == ns:
            # Strictly before the print, for descriptive slippage attribution.
            # This observation cannot gate or erase an already received fill.
            entry_price = p.future_sell_cash//(p.future_sell_qty*p.contract.shares)
            risk = self._risk(p.contract, entry_price, ns-1)
            self._event(p, ns, "s2_fill_liquidity", **risk)
