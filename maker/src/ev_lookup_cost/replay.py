"""One-second policy scheduler with chronological raw-trade execution events."""
from __future__ import annotations

import heapq
import json
from dataclasses import asdict
from pathlib import Path

import polars as pl

from .causal_lookup import (CLOSE_SECOND, SECOND, CapacityObservation, DayEntryOutcome,
                            DecayObservation, ExitRiskObservation, LookupHistory,
                            spread_bucket, time_bucket)
from .capacity_policy import activate_parked
from .cost_observations import observe_costs
from .market import MarketDay, WF
from .portfolio import Portfolio
from .valuation import mark_positions

S1_COLUMNS = ["raw_order_fact_id", "Date", "ValueCode", "QuoteCode", "target_price",
              "nominal_new_time_ns", "nominal_stop_time_ns", "boundary_source_asof_date"]


def s1_commands(day: str) -> list[dict]:
    """Read all issued quote intents, including those that never filled.

    No makerFill/outcome/actual-fill columns are loaded. Source thresholds must
    have been frozen before this day. The existing S0 quote-command schedule
    remains an exogenous causal policy input; its fill model is not reused.
    """
    path = WF / "august_attribution_s0_20260824_v2/raw_order_facts.parquet"
    frame = pl.scan_parquet(path).filter(pl.col("Date") == day).select(S1_COLUMNS).collect()
    if frame.filter(pl.col("boundary_source_asof_date").is_null() |
                    (pl.col("boundary_source_asof_date") >= pl.col("Date"))).height:
        raise ValueError("S1 thresholds were not available before the session")
    return frame.to_dicts()


def _write_rows(path: Path, rows: list[dict]) -> None:
    if rows:
        pl.from_dicts(rows, infer_schema_length=None).write_parquet(path)


class Replay:
    def __init__(self, output: Path, portfolios: list[Portfolio], *,
                 products: list[str] | None = None, shadow: Portfolio | None = None):
        self.output, self.portfolios, self.products = output, portfolios, products
        self.shadow = shadow or Portfolio("shadow", 10**12, use_ev=False, use_bpday=False, shadow=True,
                                hedge_ms=portfolios[0].hedge_delay // 1_000_000,
                                cancel_ms=portfolios[0].cancel_delay // 1_000_000,
                                event_quotes=portfolios[0].event_quotes, split_ev=portfolios[0].split_ev)
        if len({a.event_quotes for a in portfolios}) != 1:
            raise ValueError("execution variants need independent shadow histories")
        self.actors = [self.shadow] + portfolios
        self.history = LookupHistory()
        self.snapshots: list[dict] = []
        self._serial = 0
        self._heap = []

    def schedule(self, ns: int, kind: str, actor: Portfolio, payload) -> None:
        self._serial += 1
        priority = {"hedge": 1, "cancel": 2, "s2_requote": 3, "s1_new": 4, "s1_cancel": 2}[kind]
        heapq.heappush(self._heap, (ns, priority, self._serial, kind, actor, payload))

    def day(self, day: str, *, data_outage: bool = False, market=None) -> list[dict]:
        carry = list({p.contract.qc: p.contract for a in self.actors
                      for p in a.positions.values() if p.id in a.active}.values())
        if market is None:
            market = MarketDay(day, self.output, self.products, carry, data_outage=data_outage)
        session = len(self.history.sessions)
        self._heap = []
        root = self.output / f"Date={day}"
        root.mkdir(parents=True, exist_ok=False)
        (root / "inputs.json").write_text(json.dumps(market.inputs, indent=2) + "\n")
        for actor in self.actors:
            snapshot = self.history.freeze(day, actor.ledger.cap_cents / 100, actor.rejected_by_day)
            self.snapshots.append(dict(day=day, portfolio=actor.name, cutoff_ns=snapshot.cutoff_ns,
                                       lam_bp=snapshot.lam_bp, train_days=list(snapshot.train_days),
                                       cells=dict(snapshot.cells),
                                       capacity_observations_count=len(snapshot.capacity_observations),
                                       psd={f"{k[0]}_{k[1]}": v for k, v in snapshot.psd.items()},
                                       pnx=dict(snapshot.pnx),
                                       exit_hazards=dict(snapshot.exit_hazards),
                                       pfill={str(k): v for k, v in snapshot.pfill.items()},
                                       decay={"|".join(map(str, k)): list(v)
                                              for k, v in snapshot.decay.items() if v[0]}))
            actor.begin(market, session, snapshot, self.schedule)
        for row in ([] if data_outage else s1_commands(day)):
            if self.products and row["ValueCode"] not in self.products:
                continue
            start, stop = row["nominal_new_time_ns"], row["nominal_stop_time_ns"]
            if start is None or stop is None or stop <= start or not market.start <= start < market.end:
                continue
            for actor in self.actors:
                self.schedule(start, "s1_new", actor, row)
                self.schedule(min(stop, market.start + 14_000 * SECOND), "s1_cancel", actor,
                              row["raw_order_fact_id"])
        trade_iter = iter(market.trades.iter_rows())
        next_trade = next(trade_iter, None)
        top_iter = iter(market.spot_top_events.iter_rows())
        next_top = next(top_iter, None) if any(a.park_spot for a in self.actors) else None
        ask_iter = iter(market.book_events.iter_rows())
        next_ask = next(ask_iter, None)
        sec = 0
        s2_candidates: dict[int, list[str]] = {}
        for vc, b in market.signals.items():
            for i in b["ok"].nonzero()[0]:
                s2_candidates.setdefault(int(i), []).append(vc)
        while True:
            trade_ns = next_trade[0] if next_trade else market.end + 1
            task_ns = self._heap[0][0] if self._heap else market.end + 1
            top_ns = next_top[0] if next_top else market.end + 1
            ask_ns = next_ask[0] if next_ask else market.end + 1
            decision_ns = market.start + sec * SECOND if sec < CLOSE_SECOND else market.end + 1
            ns = min(trade_ns, top_ns, ask_ns, task_ns, decision_ns)
            if ns > market.end:
                break
            if trade_ns <= min(top_ns, ask_ns, task_ns, decision_ns):
                t, sequence, instrument, price, qty = next_trade
                for actor in self.actors:
                    actor.trade(instrument, t, sequence, price, qty)
                next_trade = next(trade_iter, None)
            elif ask_ns <= min(top_ns, task_ns, decision_ns):
                # Same-timestamp prints were consumed first; the book at ask_ns
                # already includes this update.
                for actor in self.actors:
                    actor.book_update(next_ask[1], ask_ns)
                instrument = next_ask[1]
                vc = instrument[2:] if instrument.startswith("S:") else market.future_to_vc.get(instrument[2:])
                self.requote(vc, ask_ns)
                next_ask = next(ask_iter, None)
            elif top_ns <= min(task_ns, decision_ns):
                for actor in self.actors:
                    activate_parked(actor, top_ns, next_top[1])
                next_top = next(top_iter, None)
            elif task_ns <= decision_ns:
                t, _, _, kind, actor, payload = heapq.heappop(self._heap)
                if kind == "hedge":
                    actor.hedge(payload, t)
                elif kind == "cancel":
                    actor.cancel(payload, t)
                    p = actor.positions.get(payload)  # exit order IDs differ from position IDs
                    if actor is self.shadow and p is not None and p.contract.vc in actor.repeg:
                        self.schedule(t, "s2_requote", actor, p.contract.vc)
                elif kind == "s2_requote":
                    self.requote(payload, t)
                elif kind == "s1_new":
                    actor.offer_s1(payload, t)
                elif kind == "s1_cancel":
                    actor.stop_s1(payload, t)
            else:
                for actor in self.actors:
                    actor.second(sec)
                # All streams share one quote-time hard-cap allocator. New
                # S2 opportunities are emitted by the uncapped virtual policy;
                # a cap rejection cannot alter next day's training universe.
                for vc in sorted(s2_candidates.get(sec, [])):
                    if vc in self.shadow.s2_live or ns < self.shadow.cooldown.get(vc, 0):
                        continue
                    intent = f"S2/{day}/{vc}/{sec}"
                    for actor in self.actors:
                        actor.offer_s2(vc, intent, sec)
                sec += 1
        # Lambda sees only shadow intents that actually received a maker fill.
        shadow_filled = {p.id for p in self.shadow.positions.values()
                         if p.entry_day == day and p.entry_fill_ns is not None}
        for actor in self.portfolios:
            actor.rejected_by_day[day] = sum(v for k, v in actor.rejected_intents.items()
                                             if k in shadow_filled)
        for outcome in self.shadow.resolved:
            self.history.observe_resolution(outcome, market.end)
        if not data_outage:
            # Exit exposure of every position paired at some point today, for
            # the next session's P_fill(remaining time) table. Positions
            # settled at the open by expiry accounting were never at risk.
            for p in self.shadow.positions.values():
                if (p.hedged_ns is None or p.hedged_ns > market.end or p.continuity_blocked
                        or p.state == "cancelled" or p.close_kind == "expiry_basis_zero_accounting"):
                    continue
                if p.close_day is not None and p.close_day < day:
                    continue
                hedged = max(0.0, (p.hedged_ns - market.start) / SECOND)
                exited = ((p.close_ns - market.start) / SECOND
                          if p.close_day == day and p.close_kind == "maker_exit" else None)
                self.history.observe_exit_risk(
                    ExitRiskObservation(f"{p.id}@{day}", p.stream, day, hedged, exited, market.end),
                    market.end)
            # Execution decay of today's shadow entries and maker exits: the
            # walk-forward tables that price entry/exit slippage from tomorrow.
            cost_rows = observe_costs(self.history, self.shadow.positions.values(), day, market.end)
            _write_rows(root / "decay_observations.parquet", [asdict(r) for r in cost_rows])
        for p in self.shadow.positions.values():
            if p.entry_day == day and p.entry_fill_ns is not None:
                self.history.observe_entry_day(DayEntryOutcome(p.id, p.stream, p.quote_second, day,
                                                               market.end, p.close_day == day), market.end)
        risk_rows = []
        if not data_outage:
            for p in self.shadow.positions.values():
                if p.hedged_ns is None or p.state == "cancelled" or p.continuity_blocked:
                    continue
                observation = CapacityObservation(day, p.stream, p.entry_day < day,
                    max(0., (p.hedged_ns - market.start) / SECOND),
                    (p.close_ns - market.start) / SECOND if p.close_day == day else float(CLOSE_SECOND),
                    p.close_day == day, market.end, p.contract.expiry,
                    p.close_kind if p.close_day == day else None,
                    p.pnl_bp if p.close_day == day else 0.0)
                self.history.capacity_observations.append(observation)
                risk_rows.append(asdict(observation))
        _write_rows(root / "capacity_observations.parquet", risk_rows)
        rows = []
        for actor in self.actors:
            daily = actor.finish()
            daily.update(mark_positions(actor))
            rows.append(dict(portfolio=actor.name, **daily))
            folder = root / actor.name
            folder.mkdir()
            _write_rows(folder / "decisions.parquet", actor.decisions)
            _write_rows(folder / "execution.parquet", actor.trace)
            _write_rows(folder / "ledger.parquet", actor.ledger.events)
            _write_rows(folder / "positions.parquet", [asdict(p) for p in actor.positions.values()])
            _write_rows(folder / "marks.parquet", actor.marks)
            actor.positions = {p.id: p for p in actor.positions.values() if p.id in actor.active}
            actor.trace.clear()
            actor.decisions.clear()
            actor.ledger.events.clear()
            actor.resolved.clear()
            actor.rejected_intents.clear()
        self.history.finish_session(day)
        (self.output / "snapshots.json").write_text(json.dumps(self.snapshots, indent=2) + "\n")
        for actor in self.actors:
            pl.from_dicts(actor.daily).write_csv(self.output / f"{actor.name}_daily.csv")
        return rows

    def requote(self, vc: str | None, ns: int) -> None:
        """Emit one common replacement intent after the shadow cancel is effective."""
        if vc is None or not self.shadow.event_quotes or vc not in self.shadow.repeg:
            return
        if vc in self.shadow.s2_live or ns < self.shadow.cooldown.get(vc, 0):
            return
        intent = f"S2/{self.shadow.day}/{vc}/e{ns}"
        self.shadow._requote_s2(vc, ns, intent)
        if intent in self.shadow.queue.orders:
            self.shadow.repeg.discard(vc)
            for actor in self.portfolios:
                actor._requote_s2(vc, ns, intent)
