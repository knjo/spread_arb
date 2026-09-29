"""Shared-volume, shared-depth A/B event replay, with resumable daily artifacts.

Run both policies on one raw market pass:
  uv run --project /home/kevin/Project/HFT --no-sync python -m spreadArb.src.backtest.causal_replay --out my_AB
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict, field
import heapq
import json
from pathlib import Path
import pickle
import time
import signal
import faulthandler

import numpy as np
import polars as pl

from maker.src.ev_lookup_cost.execution import CapacityLedger, PrintedVolumeQueue, QueueOrder, TakerDepth
from ..common.paths import DATA_ROOT, SECOND, grid_days, QUOTE_START_SECOND
from ..ev import reach, abs_reach
from .policy import PolicyConfig, PRESETS, StreamingDecider, exit_target_bp
from .causal_market import Contract, Market
from .capital import PositionCapital
from .runtime import init_run, assert_sources, memory_usage, release_memory, require_qcache

DELAY = 50_000_000


@dataclass
class Cycle:
    id: str
    contract: Contract
    stream: str
    quote_day: str
    quote_ns: int
    quote_ab: float
    anchor: float
    scale: float
    entry_price: int
    target_bp: float | None
    route: str
    ev_bp: float
    score: float
    t_days_pred: float
    p_sd_pred: float
    state: str = "quoting"
    maker_qty: int = 0
    fill_ns: int | None = None
    hedge_ns: int | None = None
    entry_spot_cash: int = 0
    entry_future_cash: int = 0
    spot_buy_cash: int = 0
    spot_sell_cash: int = 0
    future_sell_cash: int = 0
    future_buy_cash: int = 0
    spot_buy_qty: int = 0
    spot_sell_qty: int = 0
    future_sell_qty: int = 0
    future_buy_qty: int = 0
    winner: str | None = None
    exit_round: int = 0
    exit_route: str | None = None
    exit_fill_ns: int | None = None
    exit_hedge_ns: int | None = None
    close_day: str | None = None
    close_kind: str | None = None
    close_ns: int | None = None
    extra_fees: float = 0.0
    fees: float = 0.0
    pnl_net: float | None = None
    tasks: set = field(default_factory=set)
    orders: set = field(default_factory=set)
    entry_race: bool = False
    quote_nominal_cents: int = 0
    exit_race: bool = False
    exit_double_risk: bool = False
    refresh_source_order: str | None = None

    @property
    def sq(self):
        return self.spot_buy_qty - self.spot_sell_qty

    @property
    def fq(self):
        return self.future_sell_qty - self.future_buy_qty

    @property
    def gross(self):
        return (self.spot_sell_cash - self.spot_buy_cash + self.future_sell_cash - self.future_buy_cash) / 10000


class Actor:
    def __init__(self, name, cfg):
        self.name, self.cfg = name, cfg
        ledger = CapacityLedger if cfg.reserve_on_submit else PositionCapital
        self.ledger = ledger(round(cfg.cap_twd * 100))
        self.cycles = {}
        self.counter = 0
        self.task_counter = 0
        self.signal_hist = []
        self.daily = []
        self.completed_pnl = 0.0
        self.prior_equity = 0.0

    def begin(self, market, engine):
        self.market, self.engine, self.day = market, engine, market.day
        self.queue, self.depth = PrintedVolumeQueue(), TakerDepth()
        self.orders, self.hedges, self.next_print = {}, {}, {}
        self.entry_orders = set()
        self.exit_pending = {}
        self.exit_working = {}
        self.active_by_vc = {}
        # Preserve the global selector's visibility during begin(): all
        # carried positions must be indexed before the first exit plan.
        for p in self.cycles.values():
            self.active_by_vc.setdefault(p.contract.vc, set()).add(p.id)
        self.trace, self.cash, self.marks = [], [], []
        self.stats = dict(admitted=0, filled=0, rollbacks=0, cap_rejects=0,
                          product_cap_rejects=0, busy_rejects=0, hurdle_rejects=0,
                          no_estimate=0, maker_partial_fills=0, marketable_rejects=0,
                          print_events=0, hedge_retries=0, cancel_race_fills=0,
                          double_exit_fills=0, settled=0, missing_marks=0,
                          capacity_cancels=0, capital_excess_fills=0,
                          quote_expirations=0, entry_refresh_checks=0, entry_refresh_submitted=0)
        base = self.cfg.cost.hurdle_bp_per_day
        recent = self.signal_hist[-self.cfg.dyn_window:]
        scores = [s for _, vals, _ in recent for s in vals]
        use = (self.cfg.dyn_q is not None and scores and recent
               and recent[-1][2] >= self.cfg.dyn_cap_frac * self.cfg.cap_twd)
        self.hurdle = max(base, float(np.quantile(scores, self.cfg.dyn_q))) if use else base
        for p in list(self.cycles.values()):
            current = getattr(market, "exact", {}).get(p.contract.qc)
            if current is not None:
                if current.shares != p.contract.shares or current.vc != p.contract.vc:
                    raise ValueError(f"unhandled contract adjustment: {p.id}")
                p.contract = current
            p.orders.clear()
            pending = list(p.tasks)
            p.tasks.clear()
            for task in pending:
                # Tasks carry only observable unhedged inventory, never a future fill.
                purpose, instrument, side, qty = task[:4]
                self.hedge_task(p, purpose, instrument, side, qty, market.start + DELAY)
            if p.state == "paired":
                if p.contract.vc in market.corporate_block:
                    self.force_close(p, market.start + DELAY, "corporate_risk_exit")
                else:
                    self.exit_offers(p, market.start + QUOTE_START_SECOND * SECOND)

    def event(self, ns, kind, p=None, **fields):
        self.trace.append(dict(ns=ns, kind=kind, id=p.id if p else None, **fields))

    def schedule(self, ns, kind, payload):
        self.engine.push(ns, self, kind, payload)

    def offer(self, row, decision):
        if not decision.admit:
            self.stats["no_estimate"] += 1
            return decision.reason
        if decision.best.score < self.hurdle:
            self.stats["hurdle_rejects"] += 1
            return "hurdle"
        vc, qc, ns = row["vc"], row["qc"], row["quote_ns"]
        if vc in self.market.corporate_block:
            return "corporate_block"
        active = [self.cycles[pid] for pid in self.active_by_vc.get(vc, ())]
        # Entry slots are independent for S1 and S2. The same stream waits for
        # its prior quote/hedge/rollback; paired inventories do not occupy it.
        if any(p.stream == row["stream"]
               and p.state in ("quoting", "entry_hedge", "entry_rollback") for p in active):
            self.stats["busy_rejects"] += 1
            return "busy"
        if self.cfg.max_positions_per_product is not None and len(active) >= self.cfg.max_positions_per_product:
            return "count_cap"
        c = self.market.contracts[vc]
        reserve_px = row["price"] if row["stream"] == "S1" else (
            self.market.limits[vc][1] if self.cfg.reserve_on_submit else row["spot_a1"])
        cents = (reserve_px * c.shares + 99) // 100
        used_vc = sum(self.ledger.amounts.get(p.id, 0) for p in active)
        product_cap = max(cents, round(self.cfg.cap_twd * self.cfg.product_cap_frac * 100))
        if used_vc + cents > product_cap:
            self.stats["product_cap_rejects"] += 1
            return "product_cap"
        if self.ledger.committed_cents + cents > self.ledger.cap_cents:
            self.stats["cap_rejects"] += 1
            return "cap"
        self.counter += 1
        pid = f"{row['stream']}/{self.day}/{vc}/{self.counter}"
        if self.cfg.reserve_on_submit and not self.ledger.reserve(pid, cents, ns):
            self.stats["cap_rejects"] += 1
            return "cap"
        d, scale = decision.best, row["scale"] or 1.0
        p = Cycle(pid, c, row["stream"], self.day, ns, row["quote_ab"], row["anchor"], scale,
                  row["price"], row["target_bp"] if "target_bp" in row else
                  exit_target_bp(d, row["anchor"], scale, decision.evals, self.cfg.abs_target_mode),
                  d.route, d.ev_bp, d.score, d.t_days, d.p_sd)
        p.quote_nominal_cents = cents
        self.cycles[pid] = p
        self.active_by_vc.setdefault(vc, set()).add(pid)
        self.stats["admitted"] += 1
        self.new_order(p, row["stream"], row["price"], ns, replaces=row.get("refresh_from"))
        return "submitted"

    def new_order(self, p, route, price, ns, replaces=None):
        instrument = ("S:" + p.contract.vc) if route in ("S1", "E1") else ("F:" + p.contract.qc)
        side = "buy" if route in ("S1", "E2") else "sell"
        qty = p.contract.shares if route in ("S1", "E1") else 1
        oid = f"{p.id}/{route}/{p.exit_round}/{ns}"
        live = ns + self.cfg.place_ns
        if live >= self.market.withdraw:
            if route in ("S1", "S2"):
                self.close(p, ns, "cancelled")
            return
        order = dict(id=oid, pid=p.id, route=route, instrument=instrument, side=side, price=price,
                     qty=qty, filled=0, send_ns=ns, live_ns=live, cancel_at=None, done=False, refresh_due=False)
        if route in ("E1", "E2"):
            slot = (p.contract.vc, route)
            if slot in self.exit_working:
                raise AssertionError("multiple working exits for one product/route")
            self.exit_working[slot] = oid
        self.orders[oid] = order
        p.orders.add(oid)
        if route in ("S1", "S2"):
            self.entry_orders.add(oid)
        self.event(ns, "submit", p, order_id=oid, route=route, price=price, qty=qty,
                   nominal_cents=p.quote_nominal_cents,
                   committed_cents=self.ledger.committed_cents, replaces_order_id=replaces)
        self.schedule(live, "place", oid)
        tl = self.market.timelines.get(p.contract.qc)
        trigger = tl.guard(route, price, p.target_bp, ns, self.cfg) if tl else self.market.withdraw
        self.schedule(trigger, "guard", oid)

    def place(self, oid, ns):
        o = self.orders.get(oid)
        if o is None or o["done"]:
            return
        p = self.cycles[o["pid"]]
        b = self.market.book(o["instrument"], ns)
        # Maker-only quotes use exchange post-only semantics. A limit that has
        # become marketable during transmission is rejected at arrival, not at send.
        marketable = bool(b and b.valid() and ((o["side"] == "buy" and o["price"] >= b.asks[0][0])
                                              or (o["side"] == "sell" and o["price"] <= b.bids[0][0])))
        marketable |= any(q.instrument == o["instrument"] and q.side != o["side"]
                          and (o["price"] >= q.price if o["side"] == "buy" else o["price"] <= q.price)
                          for q in self.queue.orders.values())
        if marketable:
            self.stats["marketable_rejects"] += 1
            self.event(ns, "post_only_reject", p, order_id=oid)
            self.cancel(oid, ns)
            return
        ahead = b.ahead(o["side"], o["price"]) if b else 0
        self.queue.add(QueueOrder(oid, o["instrument"], o["side"], o["price"], o["qty"], ns,
                                  o["route"], p.id), ahead)
        self.event(ns, "live", p, order_id=oid, ahead=ahead)
        self.arm_print(o["instrument"], ns)
        deadline = ns + self.cfg.quote_refresh_ns
        if deadline <= self.market.end:
            self.schedule(deadline, "quote_expire", oid)

    def expire_quote(self, oid, ns):
        o = self.orders.get(oid)
        if not o or o["done"]:
            return
        o["refresh_due"] = True
        self.stats["quote_expirations"] += 1
        self.event(ns, "quote_expired", self.cycles[o["pid"]], order_id=oid,
                   deadline_ns=o["live_ns"] + self.cfg.quote_refresh_ns)
        self.request_cancel(oid, ns, reason="quote_refresh")

    def arm_print(self, instrument, ns):
        if instrument in self.next_print or instrument not in self.queue.by_instrument:
            return
        pr = self.market.prints.get(instrument)
        if pr is None:
            return
        j = int(np.searchsorted(pr.ns, ns, side="right"))
        if j < len(pr.ns) and pr.ns[j] <= self.market.end:
            self.next_print[instrument] = j
            self.schedule(int(pr.ns[j]), "print", (instrument, j))

    def trade(self, instrument, j, ns):
        if self.next_print.get(instrument) != j:
            return
        del self.next_print[instrument]
        pr = self.market.prints[instrument]
        self.stats["print_events"] += 1
        fills = self.queue.trade(instrument, ns, int(pr.seq[j]), int(pr.price[j]), int(pr.qty[j]))
        for qo, qty in fills:
            self.maker_fill(qo.id, qty, ns, int(pr.seq[j]), int(pr.qty[j]))
        # Preserve distinct sequence numbers at the same timestamp.
        j += 1
        if instrument in self.queue.by_instrument and j < len(pr.ns) and pr.ns[j] <= self.market.end:
            self.next_print[instrument] = j
            self.schedule(int(pr.ns[j]), "print", (instrument, j))

    def cash_leg(self, p, instrument, side, qty, cash, ns, purpose, **extra):
        asset = "spot" if instrument.startswith("S:") else "future"
        scaled = cash if asset == "spot" else cash * p.contract.shares
        key = asset + "_" + side
        setattr(p, key + "_qty", getattr(p, key + "_qty") + qty)
        setattr(p, key + "_cash", getattr(p, key + "_cash") + scaled)
        self.cash.append(dict(id=p.id, ns=ns, instrument=instrument, side=side, qty=qty,
                              cash=scaled, purpose=purpose, **extra))
        self.event(ns, "execution", p, instrument=instrument, side=side, qty=qty, purpose=purpose)

    def entry_notional(self, p, ns):
        """Observable full-lot spot nominal; never a future hedge price."""
        if p.stream == "S1":
            return (p.entry_price * p.contract.shares + 99) // 100
        book = self.market.book("S:" + p.contract.vc, ns)
        if book is not None and book.valid():
            return (book.asks[0][0] * p.contract.shares + 99) // 100
        return p.quote_nominal_cents

    def book_entry_exposure(self, p, ns, kind, order=None):
        if self.cfg.reserve_on_submit:
            return
        cents = ((p.entry_spot_cash + 99) // 100 if p.stream == "S1" or kind == "hedged"
                 else self.entry_notional(p, ns))
        old = self.ledger.amounts.get(p.id, 0)
        b = self.market.book("S:" + p.contract.vc, ns) if p.stream == "S2" and kind != "hedged" else None
        self.ledger.exposure(p.id, cents, ns, kind,
            order_id=order["id"] if order else None,
            cancel_request_ns=order["cancel_at"] if order else None,
            maker_qty=p.maker_qty, entry_spot_cash=p.entry_spot_cash,
            estimate=(p.stream == "S2" and kind != "hedged"),
            estimate_book_ns=b.ns if b is not None and b.valid() else None,
            estimate_book_seq=b.sequence if b is not None and b.valid() else None,
            estimate_ask=b.asks[0][0] if b is not None and b.valid() else None)
        self.event(ns, "capital", p, capital_kind=kind, delta_cents=cents-old,
                   amount_cents=cents, committed_cents=self.ledger.committed_cents)
        if kind == "maker_fill" and self.ledger.committed_cents > self.ledger.cap_cents:
            self.stats["capital_excess_fills"] += 1
        self.cancel_excess_entries(ns)

    def cancel_excess_entries(self, ns):
        if self.cfg.reserve_on_submit:
            return
        free = self.ledger.cap_cents - self.ledger.committed_cents
        for oid in sorted(self.entry_orders):
            o = self.orders[oid]
            if o["done"] or o["cancel_at"] is not None:
                continue
            p = self.cycles[o["pid"]]
            full = self.entry_notional(p, ns)
            needed = ((o["price"] * (o["qty"]-o["filled"]) + 99) // 100
                      if o["route"] == "S1" else full)
            used = sum(self.ledger.amounts.get(pid, 0) for pid in self.active_by_vc.get(p.contract.vc, ()))
            product_cap = max(full, round(self.cfg.cap_twd*self.cfg.product_cap_frac*100))
            if needed > free or used+needed > product_cap:
                reason = "capacity" if needed > free else "product_capacity"
                self.stats["capacity_cancels"] += 1
                self.request_cancel(oid, ns, reason=reason, required_cents=needed,
                                    free_cents=free, product_free_cents=product_cap-used,
                                    full_nominal_cents=full)

    def maker_fill(self, oid, qty, ns, sequence, printed_qty):
        o, p = self.orders[oid], self.cycles[self.orders[oid]["pid"]]
        o["filled"] += qty
        full = o["filled"] == o["qty"]
        self.stats["maker_partial_fills"] += int(not full)
        race = o["cancel_at"] is not None and ns >= o["cancel_at"]
        self.stats["cancel_race_fills"] += int(race)
        self.cash_leg(p, o["instrument"], o["side"], qty, o["price"] * qty, ns, o["route"],
                      order_id=oid, liquidity="maker", sequence=sequence, printed_qty=printed_qty,
                      live_ns=o["live_ns"], limit_price=o["price"])
        if full:
            o["done"] = True
            p.orders.discard(oid)
            self.entry_orders.discard(oid)
            if o["route"] in ("E1", "E2"):
                self.exit_working.pop((p.contract.vc, o["route"]), None)
        if o["route"] in ("S1", "S2"):
            p.maker_qty += qty
            p.entry_race |= race
            if o["route"] == "S1":
                p.entry_spot_cash += o["price"] * qty
            else:
                p.entry_future_cash += o["price"] * qty * p.contract.shares
            self.book_entry_exposure(p, ns, "maker_fill", o)
            if full:
                p.fill_ns, p.state = ns, "entry_hedge"
                ins, side, q = (("F:" + p.contract.qc, "sell", 1) if o["route"] == "S1"
                                else ("S:" + p.contract.vc, "buy", p.contract.shares))
                self.hedge_task(p, "entry", ins, side, q, ns + DELAY)
            return
        p.exit_race |= race
        if p.winner is None:
            p.winner, p.exit_route = oid, o["route"]
            self.clear_exit_plans(p)
            for other in list(p.orders):
                if other != oid:
                    self.request_cancel(other, ns, reason="other_exit_fill")
        if p.winner != oid:
            p.exit_double_risk = True
            self.stats["double_exit_fills"] += 1
            self.hedge_task(p, "double_rollback", o["instrument"], "buy" if o["side"] == "sell" else "sell", qty, ns + DELAY)
        elif full:
            p.exit_fill_ns, p.state = ns, "exit_hedge"
            for other in list(p.orders):
                if other != oid:
                    self.request_cancel(other, ns, reason="other_exit_fill")
            ins, side, q = (("F:" + p.contract.qc, "buy", 1) if o["route"] == "E1"
                            else ("S:" + p.contract.vc, "sell", p.contract.shares))
            self.hedge_task(p, "exit", ins, side, q, ns + DELAY)
        self.exit_offers(p, ns+1)

    def request_cancel(self, oid, ns, reason="market_guard", **evidence):
        o = self.orders.get(oid)
        if not o or o["done"] or o["cancel_at"] is not None:
            return
        o["cancel_at"] = ns
        self.event(ns, "cancel_request", self.cycles[o["pid"]], order_id=oid, reason=reason, **evidence)
        self.schedule(ns + self.cfg.cancel_ns, "cancel", oid)

    def cancel(self, oid, ns):
        o = self.orders.get(oid)
        if not o or o["done"]:
            return
        p = self.cycles[o["pid"]]
        self.queue.cancel(oid, ns)
        o["done"] = True
        p.orders.discard(oid)
        self.entry_orders.discard(oid)
        if o["route"] in ("E1", "E2"):
            self.exit_working.pop((p.contract.vc, o["route"]), None)
        self.event(ns, "cancel_effective", p, order_id=oid)
        route, qty = o["route"], o["filled"]
        if route in ("S1", "S2"):
            if o["refresh_due"]:
                p.refresh_source_order = oid
            if qty:
                p.state = "entry_rollback"
                self.stats["rollbacks"] += 1
                self.hedge_task(p, "entry_rollback", o["instrument"], "sell", qty, ns + DELAY)
            else:
                self.close(p, ns, "cancelled")
        elif p.winner == oid and qty:
            p.state = "exit_rollback"
            self.hedge_task(p, "exit_rollback", o["instrument"], "buy" if o["side"] == "sell" else "sell", qty, ns + DELAY)
        elif p.state == "paired" and p.winner is None:
            self.exit_offers(p, ns + 1, routes=(route,))
        self.maybe_close(p, ns)
        if route in ("E1", "E2"):
            self.exit_offers(p, ns+1)

    def hedge_task(self, p, purpose, instrument, side, qty, ns):
        self.task_counter += 1
        task = (purpose, instrument, side, qty, self.task_counter)
        p.tasks.add(task)
        self.hedges[(p.id, task)] = ns
        self.schedule(ns, "hedge", (p.id, task))

    def hedge(self, pid, task, ns):
        p = self.cycles.get(pid)
        if p is None or task not in p.tasks or self.hedges.get((pid, task)) != ns:
            return
        purpose, instrument, side, qty = task[:4]
        b = self.market.book(instrument, ns)
        if instrument.startswith("S:"):
            lo, hi = self.market.limits.get(p.contract.vc, (1, 10**15))
        else:
            lo, hi = (p.contract.fut_ref * 90 + 99) // 100, p.contract.fut_ref * 110 // 100
        taken = self.depth.take(b, side, qty, ns, lo, hi) if b else None
        if taken is None:
            self.stats["hedge_retries"] += 1
            self.event(ns, "hedge_retry", p, purpose=purpose, instrument=instrument, qty=qty)
            nxt = self.market.next_book(instrument, ns)
            if nxt is not None and nxt <= self.market.end:
                self.hedges[(pid, task)] = nxt
                self.schedule(nxt, "hedge", (pid, task))
            return
        cash, _ = taken
        self.cash_leg(p, instrument, side, qty, cash, ns, purpose, liquidity="taker",
                      book_ns=b.ns, book_seq=b.sequence)
        p.tasks.remove(task)
        del self.hedges[(pid, task)]
        if purpose == "entry":
            if instrument.startswith("S:"):
                p.entry_spot_cash += cash
            else:
                p.entry_future_cash += cash * p.contract.shares
            p.hedge_ns, p.state = ns, "paired"
            if self.cfg.reserve_on_submit:
                self.ledger.confirm_nominal(pid, (p.entry_spot_cash + 99) // 100, ns)
            else:
                self.book_entry_exposure(p, ns, "hedged")
            self.stats["filled"] += 1
            self.exit_offers(p, ns + 1)
        elif purpose == "exit":
            p.exit_hedge_ns, p.state = ns, "exit_complete"
        elif purpose.endswith("rollback"):
            if instrument.startswith("S:"):
                # Same 20bp round-trip convention as the original comparison.
                p.extra_fees += cash / 10000 * self.cfg.cost.fee_same_day_bp / 10000
            else:
                p.extra_fees += 40.0 + cash * p.contract.shares / 10000 * .4 / 10000
            if purpose == "exit_rollback":
                p.state = "exit_reset"
            elif purpose == "entry_rollback":
                p.state = "rollback_complete"
        self.maybe_close(p, ns)

    def clear_exit_plans(self, p):
        for key, plan in list(self.exit_pending.items()):
            if plan[0] == p.id:
                del self.exit_pending[key]

    def exit_owner(self, vc):
        candidates = (self.cycles[pid] for pid in self.active_by_vc.get(vc, ()))
        eligible = [p for p in candidates if p.contract.vc == vc
                    and p.state == "paired" and p.winner is None and p.target_bp is not None]
        return min(eligible, key=lambda p: (p.hedge_ns or p.quote_ns, p.quote_ns, p.id)) if eligible else None

    def exit_offers(self, p, ns, routes=None):
        """One working exit per product/route, allocated FIFO across inventories."""
        ns = max(ns, getattr(self.market, "quote_start", self.market.start))
        if ns + self.cfg.place_ns >= self.market.withdraw:
            return
        owner = self.exit_owner(p.contract.vc)
        if owner is None:
            return
        tl = self.market.timelines.get(owner.contract.qc)
        if tl is None:
            return
        for route in routes or self.cfg.exit_routes:
            slot = (p.contract.vc, route)
            if slot in self.exit_working:
                continue
            pending = self.exit_pending.get(slot)
            if pending is not None and pending[:2] == (owner.id, owner.exit_round):
                continue
            self.exit_pending.pop(slot, None)
            when = tl.first_exit(route, owner.target_bp, ns)
            if when is not None and when + self.cfg.place_ns < self.market.withdraw:
                self.exit_pending[slot] = (owner.id, owner.exit_round, when)
                self.schedule(when, "exit_offer", (owner.id, route, owner.exit_round))

    def offer_exit(self, pid, route, round_, ns):
        p = self.cycles.get(pid)
        if p is None:
            return
        slot = (p.contract.vc, route)
        if self.exit_pending.get(slot) != (pid, round_, ns):
            return
        del self.exit_pending[slot]
        if (p.state != "paired" or p.winner is not None or p.exit_round != round_
                or slot in self.exit_working or self.exit_owner(p.contract.vc) is not p):
            self.exit_offers(p, ns+1, routes=(route,))
            return
        tl = self.market.timelines[p.contract.qc]
        i = tl.at(ns)
        self.new_order(p, route, int(tl.prices[route][i]), ns)

    def force_close(self, p, ns, reason):
        p.close_kind, p.state = reason, "exit_complete"
        self.clear_exit_plans(p)
        for oid in list(p.orders):
            self.request_cancel(oid, ns, reason=reason)
        if p.sq:
            self.hedge_task(p, "forced", "S:" + p.contract.vc, "sell" if p.sq > 0 else "buy", abs(p.sq), ns)
        if p.fq:
            self.hedge_task(p, "forced", "F:" + p.contract.qc, "buy" if p.fq > 0 else "sell", abs(p.fq), ns)

    def maybe_close(self, p, ns):
        if p.state == "closed" or p.orders or p.tasks:
            return
        if p.sq == 0 and p.fq == 0:
            self.close(p, ns, p.close_kind or ("rollback" if p.hedge_ns is None else "maker_exit"))
        elif p.state == "exit_reset":
            p.state, p.winner = "paired", None
            p.exit_round += 1
            self.exit_offers(p, ns + 1)

    def close(self, p, ns, kind):
        if p.sq or p.fq or p.tasks or p.orders:
            raise AssertionError(f"release with exposure: {p.id}")
        p.state, p.close_day, p.close_kind, p.close_ns = "closed", self.day, kind, ns
        fee_bp = self.cfg.cost.fee_same_day_bp if self.day == p.quote_day else self.cfg.cost.fee_overnight_bp
        p.fees = (p.entry_spot_cash / 10000 * fee_bp / 10000 if p.hedge_ns is not None else 0.0) + p.extra_fees
        p.pnl_net = p.gross - p.fees
        self.completed_pnl += p.pnl_net
        if p.id in self.ledger.amounts:
            self.ledger.release(p.id, ns, terminal=True)
        self.clear_exit_plans(p)
        self.active_by_vc[p.contract.vc].discard(p.id)
        self.event(ns, "closed", p, close_kind=kind, pnl_net=p.pnl_net)
        self.exit_offers(p, ns+1)
        if p.refresh_source_order is not None:
            source = p.refresh_source_order
            p.refresh_source_order = None
            self.schedule(ns+1, "entry_refresh", (p.contract.vc, p.stream, source))

    def finish(self, scores, root):
        m = self.market
        for p in list(self.cycles.values()):
            if p.state == "closed":
                continue
            if p.orders:
                raise AssertionError("maker orders survived withdrawal")
            if p.contract.expiry <= self.day and p.state == "paired" and not p.tasks:
                mark = m.marks.get("S:" + p.contract.vc)
                if mark is None:
                    p.state = "settlement_pending"
                    self.event(m.end, "settlement_missing_mark", p)
                else:
                    px, source = mark
                    self.cash_leg(p, "S:" + p.contract.vc, "sell", p.sq, px * p.sq, m.end,
                                  "settlement", liquidity="accounting", mark_source=source)
                    self.cash_leg(p, "F:" + p.contract.qc, "buy", p.fq, px * p.fq, m.end,
                                  "settlement", liquidity="accounting", mark_source="basis_zero_assumption")
                    self.close(p, m.end, "settlement")
                    self.stats["settled"] += 1
            if p.state == "closed":
                continue
            sm, fm = m.marks.get("S:" + p.contract.vc), m.marks.get("F:" + p.contract.qc)
            missing = (p.sq != 0 and sm is None) or (p.fq != 0 and fm is None)
            fee = p.entry_spot_cash / 10000 * self.cfg.cost.fee_overnight_bp / 10000 + p.extra_fees
            value = None if missing else p.gross + (p.sq * (sm[0] if sm else 0)
                         - p.fq * p.contract.shares * (fm[0] if fm else 0)) / 10000 - fee
            self.stats["missing_marks"] += int(missing)
            self.marks.append(dict(id=p.id, day=self.day, spot_qty=p.sq, future_qty=p.fq,
                                   value_twd=value, fee_estimate=fee,
                                   spot_mark=sm[0] if sm else None, future_mark=fm[0] if fm else None,
                                   spot_source=sm[1] if sm else None, future_source=fm[1] if fm else None,
                                   state=p.state))
        opened = [p for p in self.cycles.values() if p.state != "closed"]
        realized = sum(p.pnl_net for p in self.cycles.values() if p.close_day == self.day)
        equity = None if self.stats["missing_marks"] else self.completed_pnl + sum(v["value_twd"] for v in self.marks)
        day_pnl = equity - self.prior_equity if equity is not None and self.prior_equity is not None else None
        self.prior_equity = equity
        committed = self.ledger.committed_cents / 100
        self.signal_hist.append((self.day, scores, committed))
        row = dict(day=self.day, **self.stats, realized_pnl=realized, equity_twd=equity,
                   mtm_pnl=day_pnl, open_end=len(opened), committed_end=committed,
                   committed_peak=self.ledger.peak_cents / 100, hurdle_used=self.hurdle,
                   signals_today=len(scores), unhedged_end=sum(bool(p.tasks) for p in opened))
        self.daily.append(row)
        folder = root / self.name
        folder.mkdir(parents=True, exist_ok=True)
        claims = []
        for (instrument, book_ns, book_seq, side, price), qty in self.depth.used.items():
            b = self.market.book(instrument, book_ns)
            shown = dict(b.asks if side == "buy" else b.bids)[price]
            if qty > shown:
                raise AssertionError("taker depth reused")
            claims.append(dict(instrument=instrument, book_ns=book_ns, book_seq=book_seq,
                               side=side, price=price, used_qty=qty, shown_qty=shown))
        for name, rows in (("events", self.trace), ("cash_legs", self.cash), ("ledger", self.ledger.events),
                           ("marks", self.marks), ("depth_claims", claims)):
            if rows:
                pl.from_dicts(rows, infer_schema_length=None).write_parquet(folder / f"{name}.parquet")
        records = []
        for p in self.cycles.values():
            if p.close_kind == "cancelled":
                continue
            r = asdict(p)
            r.pop("orders")
            r.pop("tasks")
            r.update(r.pop("contract"))
            records.append(r)
        if records:
            pl.from_dicts(records, infer_schema_length=None).write_parquet(folder / "positions.parquet")
        self.cycles = {p.id: p for p in opened}
        self.ledger.events.clear()
        self.ledger.peak_cents = self.ledger.committed_cents
        # Keep just the history needed by the dynamic policy in the checkpoint.
        self.signal_hist = self.signal_hist[-self.cfg.dyn_window:]
        return row


class Replay:
    def __init__(self, output, actors):
        self.output, self.actors = Path(output), actors
        self.heap, self.serial = [], 0
        self.decisions = []

    def push(self, ns, actor, kind, payload):
        self.serial += 1
        priority = {"print": 0, "guard": 1, "quote_expire": 1, "hedge": 2,
                    "cancel": 3, "place": 4, "exit_offer": 5, "entry_refresh": 5}[kind]
        heapq.heappush(self.heap, (int(ns), priority, self.serial, actor.name, kind, payload))

    def drain(self, ns):
        actors = {a.name: a for a in self.actors}
        while self.heap and self.heap[0][0] <= ns:
            t, _, _, name, kind, payload = heapq.heappop(self.heap)
            a = actors[name]
            if kind == "print":
                a.trade(*payload, t)
            elif kind == "guard":
                a.request_cancel(payload, t)
            elif kind == "quote_expire":
                a.expire_quote(payload, t)
            elif kind == "hedge":
                a.hedge(*payload, t)
            elif kind == "cancel":
                a.cancel(payload, t)
            elif kind == "place":
                a.place(payload, t)
            elif kind == "exit_offer":
                a.offer_exit(*payload, t)
            elif kind == "entry_refresh":
                self.refresh_entry(a, *payload, t)

    def record_decision(self, row, d, outcomes, origin):
        self.decisions.append(dict(ns=row["quote_ns"], vc=row["vc"], qc=row["qc"], stream=row["stream"],
            quote_ab=row["quote_ab"], anchor=row["anchor"], e_norm=row["e_norm"], scale=row["scale"],
            quote_second=row["quote_second"], expiry=row["expiry"], scale_raw=row["scale_raw"],
            resid_mid_bp=row["resid_mid_bp"], tick_bp_hedge=row["tick_bp_hedge"],
            execution_floor_bp=row["execution_floor_bp"], spot_a1=row["spot_a1"],
            price=row["price"], origin=origin, refresh_from=row.get("refresh_from"), base_admit=d.admit,
            score=d.best.score if d.best else None, ev_bp=d.best.ev_bp if d.best else None,
            t_days=d.best.t_days if d.best else None, p_sd=d.best.p_sd if d.best else None,
            route=d.best.route if d.best else None, **outcomes))

    def refresh_entry(self, actor, vc, stream, source, ns):
        market = actor.market
        contract = market.contracts.get(vc)
        tl = market.timelines.get(contract.qc) if contract is not None else None
        row = tl.entry_row_at(stream, ns) if tl is not None else None
        actor.stats["entry_refresh_checks"] += 1
        if row is None:
            actor.event(ns, "entry_refresh_check", vc=vc, route=stream, replaces_order_id=source,
                        outcome="market_ineligible")
            return
        row["refresh_from"] = source
        # Refresh attempts are portfolio-dependent. Keep them out of B's
        # exogenous market-signal quantile, but log/recompute every submitted EV.
        d = self.refresh_decider.decide(row)
        outcome = actor.offer(row, d)
        actor.stats["entry_refresh_submitted"] += int(outcome == "submitted")
        actor.event(ns, "entry_refresh_check", vc=vc, route=stream, replaces_order_id=source, outcome=outcome)
        outcomes = {a.name: outcome if a is actor else "not_requested" for a in self.actors}
        self.record_decision(row, d, outcomes, "refresh")

    def run_day(self, day, samples):
        started = time.time()
        carry = {p.contract.qc: p.contract for a in self.actors for p in a.cycles.values()}
        market = Market(day, self.actors[0].cfg, list(carry.values()))
        print(json.dumps(dict(day=day, stage="market_ready", elapsed_s=round(time.time()-started, 2))), flush=True)
        root = self.output / f"Date={day}"
        root.mkdir(parents=True, exist_ok=True)
        (root / "inputs.json").write_text(json.dumps(market.inputs, indent=2))
        self.heap = []
        lookup = reach.fit(day, 20)
        lookup = lookup if lookup.days else None
        absolute = abs_reach.AbsTable.fit(samples, as_of=day)
        decider = StreamingDecider(day, lookup, absolute, self.actors[0].cfg)
        self.refresh_decider = StreamingDecider(day, lookup, absolute, self.actors[0].cfg)
        for a in self.actors:
            a.begin(market, self)
        loaded = time.time() - started
        print(json.dumps(dict(day=day, stage="loaded", candidates=market.candidates.height,
                              load_s=round(loaded, 2))), flush=True)
        self.decisions = []
        for ordinal, (ns, qc, stream, i) in enumerate(market.candidates.iter_rows()):
            if ordinal and ordinal % 50000 == 0:
                print(json.dumps(dict(day=day, stage="replaying", candidate=ordinal,
                                      elapsed_s=round(time.time()-started, 2))), flush=True)
            self.drain(ns)
            row = market.timelines[qc].row(stream, i)
            d = decider.decide(row)
            outcomes = {a.name: a.offer(row, d) for a in self.actors}
            self.record_decision(row, d, outcomes, "market")
        self.drain(market.end)
        scores = decider.admitted_scores()
        calls = decider.calls + self.refresh_decider.calls
        if self.decisions:
            pl.from_dicts(self.decisions, infer_schema_length=None).write_parquet(root / "decisions.parquet")
        self.decisions = []
        del self.refresh_decider
        pl.DataFrame({"score": scores}, schema={"score": pl.Float64}).write_parquet(root / "base_signal_scores.parquet")
        for a in self.actors:
            row = a.finish(scores, root)
            row.update(candidates=market.candidates.height, ev_calls=calls, load_s=loaded,
                       elapsed_s=time.time() - started, **memory_usage())
            print(json.dumps(dict(portfolio=a.name, **row)), flush=True)
            pl.from_dicts(a.daily, infer_schema_length=None).write_csv(self.output / f"{a.name}_daily.csv")
            # Raw data and event buffers never enter checkpoints.
            for attr in ("market", "engine", "queue", "depth", "orders", "hedges", "next_print", "trace", "cash", "marks",
                         "active_by_vc", "entry_orders", "exit_pending", "exit_working"):
                delattr(a, attr)
        checkpoint = self.output / "checkpoint.tmp"
        checkpoint.write_bytes(pickle.dumps(dict(day=day, actors=self.actors), protocol=5))
        checkpoint.replace(self.output / "checkpoint.pkl")
        (root / "complete.json").write_text(json.dumps(dict(day=day, candidates=market.candidates.height)))

    def run(self, days, resume=False, prefetch=0):
        if prefetch:
            raise ValueError("prefetch was removed; shared chronological replay requires prefetch=0")
        self.output.mkdir(parents=True, exist_ok=True)
        inputs = require_qcache(days)
        cache_manifest = self.output / "qcache_inputs.json"
        if resume and cache_manifest.exists() and json.loads(cache_manifest.read_text()) != inputs:
            raise ValueError("Q cache changed; use a fresh run")
        cache_manifest.write_text(json.dumps(inputs, indent=2)+"\n")
        if resume and (self.output / "checkpoint.pkl").exists():
            checkpoint = pickle.loads((self.output / "checkpoint.pkl").read_bytes())
            self.actors = checkpoint["actors"]
            days = [d for d in days if d > checkpoint["day"]]
        samples = abs_reach.load_samples()
        for day in days:
            assert_sources(self.output)
            self.run_day(day, samples)
            assert_sources(self.output)
            print(json.dumps(dict(day=day, stage="released", **release_memory())), flush=True)
        from .valuation import revalue
        revalue(self.output)
        build_summary(self.output)


def build_summary(output):
    summaries = {}
    manifest = json.loads((output/"manifest.json").read_text()) if (output/"manifest.json").exists() else {}
    for name in ("A", "B"):
        official = output / f"{name}_daily_official.csv"
        path = official if official.exists() else output / f"{name}_daily.csv"
        if not path.exists():
            continue
        daily = pl.read_csv(path)
        equity = daily["equity_twd"].to_numpy()
        total, n = float(equity[-1]), daily.height
        valid_total = np.isfinite(total)
        mdd = float(np.min(equity - np.maximum.accumulate(np.r_[0.0, equity])[1:])) if np.isfinite(equity).all() else None
        cap = manifest.get("configs",{}).get(name,{}).get("cap_twd",20_000_000)
        summary = dict(days=n, total_pnl_twd=total if valid_total else None,
                       daily_pnl_twd=float(total/n) if valid_total else None,
                       annual_pct=float(total/n*250/cap*100) if valid_total else None, mtm_drawdown_twd=mdd,
                       valuation="official" if official.exists() else "replay_marks",
                       pairs=int(daily["filled"].sum()), rollbacks=int(daily["rollbacks"].sum()),
                       open_end=int(daily["open_end"][-1]), unhedged_end=int(daily["unhedged_end"][-1]),
                       missing_marks=int(daily["missing_marks"].sum()),
                       cap_peak_twd=float(daily["committed_peak"].max()),
                       mean_committed_twd=float(daily["committed_end"].mean()))
        summaries[name] = summary
    (output / "summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summaries), flush=True)


def main():
    faulthandler.register(signal.SIGUSR1, all_threads=False)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--prefetch", type=int, choices=(0,), default=0)
    reservation = ap.add_mutually_exclusive_group()
    reservation.add_argument("--reserve-on-submit", action="store_true", default=False)
    reservation.add_argument("--no-reserve-on-submit", dest="reserve_on_submit", action="store_false")
    args = ap.parse_args()
    out = DATA_ROOT / "backtest" / args.out
    actors = [Actor(n, PolicyConfig(**PRESETS[n], reserve_on_submit=args.reserve_on_submit)) for n in ("A", "B")]
    days = [d for d in grid_days() if (args.start is None or d >= args.start) and (args.end is None or d <= args.end)]
    init_run(out, days, actors, args.resume)
    Replay(out, actors).run(days, args.resume, args.prefetch)


if __name__ == "__main__":
    main()
