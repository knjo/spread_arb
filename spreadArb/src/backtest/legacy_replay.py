"""RETIRED: point-label replay with known execution defects; diagnostic use only.

Current CLI is backtest/replay.py; frozen original source is in
data/backtest/correction_20260922/original_src.tar.gz.

One live quote per product, capacity in spot notional, exits from the same product's sell-side rows
(E1: spot ask maker -> futures buy), carry across sessions, settlement at the expiry-day close marks.
Every filled pair is registered with its four legs, fees, predicted EV/route and realized slippage.

    uv run --project /home/kevin/Project/HFT --no-sync python -m spreadArb.src.backtest.replay --start 20260224 --end 20260414
"""
from __future__ import annotations

import argparse
import heapq
import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import polars as pl

from ..common.calendar import calendar_days
from ..common.grid import close_marks
from ..common.paths import (CLOSE_SECOND, DATA_ROOT, MAKER_WITHDRAW_SECOND, QUOTE_START_SECOND, SECOND, grid_days,
                            open_ns, points_path)
from ..ev import abs_reach, reach
from ..ev.config import CostConfig
from .gates import GateBook
from .policy import PRESETS, Decider, PolicyConfig, exit_target_bp

ENTRY_COLUMNS = ["stream", "vc", "qc", "expiry", "quote_ns", "quote_second", "price", "level", "anchor", "scale",
                 "quote_ab", "eff_u", "resid_mid_bp", "e_norm", "spot_b1", "spot_a1", "tick_bp_hedge", "depth_ahead",
                 "opp_depth_shares", "opp_depth_lots", "notional_twd", "t_partial_ns", "t_fill_ns", "t_ab0_ns",
                 "t_gate_ns", "hedge_ns", "hedge_vwap", "actual_ab", "d_in_realized"]
EXIT_COLUMNS = ["vc", "quote_ns", "price", "level", "quote_ab", "t_fill_ns", "t_gate_ns", "hedge_ns", "hedge_vwap",
                "actual_ab"]


@dataclass
class Position:
    id: str
    vc: str
    qc: str
    stream: str
    expiry: str
    shares: int
    route: str
    x: float | None
    target_bp: float | None
    ev_bp: float
    score: float
    t_days_pred: float
    p_sd_pred: float
    quote_day: str
    quote_ns: int
    quote_ab: float
    anchor: float
    scale: float
    entry_price: int
    fill_ns: int
    hedge_ns: int
    spot_buy_cash: int
    fut_sell_px: int
    actual_ab: float
    entry_cancel_at: int | None = None
    entry_race: bool = False               # filled inside the cancel latency after the entry guard fired
    state: str = "open"
    exit_cursor_ns: int = 0
    exit_quote_ns: int | None = None
    exit_price: int | None = None
    exit_quote_ab: float | None = None
    exit_fill_ns: int | None = None
    exit_hedge_ns: int | None = None
    exit_cancel_at: int | None = None
    exit_race: bool = False
    exit_route: str | None = None
    exit_double_risk: bool = False         # the other exit route would also have filled inside the cancel latency
    spot_sell_cash: int | None = None
    fut_buy_px: int | None = None
    close_day: str | None = None
    close_kind: str | None = None
    exit_realized_ab: float | None = None
    pnl_spot: float | None = None
    pnl_fut: float | None = None
    fees: float | None = None
    pnl_net: float | None = None
    holding_days: float | None = None

    @property
    def notional_twd(self) -> float:
        return self.spot_buy_cash / 10_000.0


def load_entries(day: str, cfg: PolicyConfig | None = None) -> pl.DataFrame:
    """Entry candidates (S1 buy / S2 sell) with the policy's cheap pre-filters applied vectorially."""
    s1 = pl.read_parquet(points_path(day, "s1_entries")).filter(pl.col("side") == "buy")
    s2 = pl.read_parquet(points_path(day, "s2_entries")).filter(pl.col("side") == "sell")
    if cfg is not None:
        s1 = s1.filter(pl.col("level").is_in(list(cfg.s1_levels)))
        s2 = s2.filter(pl.col("opp_depth_shares") >= cfg.depth_mult * 2000)
        keep = (pl.col("eff_u") >= cfg.residual_min_bp) | (pl.col("quote_ab") >= abs_reach.THRESHOLDS_BP[0])
        s1, s2 = s1.filter(keep), s2.filter(keep)
        if "S1" not in cfg.streams:
            s1 = s1.clear()
        if "S2" not in cfg.streams:
            s2 = s2.clear()
    s1 = s1.with_columns(pl.lit(None, dtype=pl.Int64).alias("opp_depth_shares")) if "opp_depth_shares" not in s1.columns else s1
    s2 = s2.with_columns(pl.lit(None, dtype=pl.Int64).alias("t_partial_ns"), pl.lit(0, dtype=pl.Int32).alias("level"),
                         pl.lit(None, dtype=pl.Int64).alias("opp_depth_lots"), pl.lit(None, dtype=pl.Int64).alias("spot_b1")) \
        if "t_partial_ns" not in s2.columns else s2
    cols = ENTRY_COLUMNS + [f"t_below_{f}" for f in (0, 5, 10, 15, 20, 25, 30)]
    return pl.concat([s1.select(cols), s2.select(cols)], how="vertical_relaxed").sort(["quote_ns", "vc"])


EXIT_LEG = {"E1": "S1_sell", "E2": "S2_buy"}


def load_exits(day: str, levels: tuple = (0,), routes: tuple = ("E1",)) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Exit rows per route and product as arrays sorted by quote time.
    E1 = S1 sell side (spot ask maker, hedge buys futures); E2 = S2 buy side (futures bid maker, hedge sells spot)."""
    cols = EXIT_COLUMNS + [f"t_above_{g}" for g in (0, 5, 10, 20, 30)]
    out: dict[str, dict] = {}
    for route in routes:
        if route == "E1":
            f = pl.read_parquet(points_path(day, "s1_entries")).filter((pl.col("side") == "sell") & pl.col("level").is_in(list(levels)))
        else:
            f = pl.read_parquet(points_path(day, "s2_entries")).filter(pl.col("side") == "buy").with_columns(pl.lit(0, dtype=pl.Int32).alias("level"))
        out[route] = {}
        for vc, g in f.sort(["vc", "quote_ns"]).select(cols).group_by("vc", maintain_order=True):
            vc = vc[0] if isinstance(vc, tuple) else vc
            out[route][vc] = {c: g[c].to_numpy() for c in cols if c != "vc"}
    return out


def _ns(v) -> int | None:
    if v is None:
        return None
    if isinstance(v, (float, np.floating)) and np.isnan(v):
        return None
    return int(v)


def _min_trigger(*vals) -> int | None:
    ts = [t for t in (_ns(v) for v in vals) if t is not None and t >= 0]
    return min(ts) if ts else None


def _active_start(qn: np.ndarray, cursor: int) -> int:
    """Index of the first row in the timestamp group active at `cursor` (rows at the same quote_ns are one
    group: one row per quote level)."""
    k = int(np.searchsorted(qn, cursor, side="right")) - 1
    if k < 0:
        return 0
    return int(np.searchsorted(qn, qn[k], side="left"))


class Replay:
    def __init__(self, cfg: PolicyConfig, out_dir, *, window: int = 20, samples: pl.DataFrame | None = None):
        self.cfg, self.out, self.window = cfg, out_dir, window
        self.positions: dict[str, Position] = {}
        self.open_ids: set[str] = set()
        self.rollbacks: list[dict] = []
        self.decisions_log: list[dict] = []
        self.daily: list[dict] = []
        self.committed = 0.0
        self.committed_by_vc: dict[str, float] = {}
        self.signal_hist: list[tuple[str, list[float], float]] = []   # (day, admitted signal scores, committed at close)
        self.used_exits: dict[tuple, set] = {}          # (route, vc) -> exit fill times already consumed today
        self._counter = 0
        self.samples = abs_reach.load_samples() if samples is None else samples

    # ---------------------------------------------------------------- exits
    def plan_exit(self, p: Position, exits: dict, day: str, start: int, from_ns: int) -> tuple | None:
        """Earliest exit fill over the enabled routes (both quotes may rest at once: different order books)."""
        if "quote_ns" in next(iter(exits.values()), {}):
            exits = {"E1": exits}                       # single-route shape: {vc: arrays}
        plans = [pl_ for route, table in exits.items()
                 if (pl_ := self._plan_route(p, table, route, start, from_ns)) is not None]
        if not plans:
            return None
        plans.sort(key=lambda x: x[2])
        best = plans[0]
        double = len(plans) > 1 and plans[1][2] <= best[2] + self.cfg.cancel_ns
        self.used_exits.setdefault((best[9], p.vc), set()).add(best[2])
        return best + (double,)

    def _plan_route(self, p: Position, table: dict, route: str, start: int, from_ns: int) -> tuple | None:
        """First row of one route at/after from_ns with quote basis <= target that fills before its guard cancels.

        Walks the product's rows by index (never re-visits a row), so it terminates in <= len(rows) steps:
        a row whose guard already fired by the time we could quote it is skipped, a cancelled quote moves the
        cursor to cancel-effective and resumes from the row active at that time.
        """
        if p.target_bp is None or p.vc not in table:
            return None
        e = table[p.vc]
        gates, leg = getattr(self, "gates", None), EXIT_LEG[route]
        qn, qab = e["quote_ns"], e["quote_ab"]
        cursor = max(from_ns, start + QUOTE_START_SECOND * SECOND)
        withdraw = start + MAKER_WITHDRAW_SECOND * SECOND
        rises = np.asarray(self.cfg.rise_grid, dtype=float)
        # E1 rests in the spot queue: a row's fill time assumes we joined at the row's own time, so only rows
        # that start at/after the cursor are usable. E2 quotes inside the futures spread (no queue).
        first = (lambda c: int(np.searchsorted(qn, c, side="left"))) if route == "E1" else (lambda c: _active_start(qn, c))
        j = first(cursor)
        while j < len(qn):
            hits = np.flatnonzero(qab[j:] <= p.target_bp)
            if hits.size == 0:
                return None
            j += int(hits[0])
            q_t = max(int(qn[j]), cursor) + self.cfg.place_ns    # live time of the exit quote
            if q_t >= withdraw:
                return None
            need = p.target_bp + self.cfg.exit_tol_bp - float(qab[j])
            g = float(rises[rises <= need].max()) if (rises <= need).any() else 0.0
            cancel_at = _min_trigger(e[f"t_above_{int(g)}"][j], e["t_gate_ns"][j],
                                     gates.next_bad(leg, p.vc, q_t) if gates is not None else None)
            if cancel_at is not None and cancel_at <= q_t:
                j += 1                      # guard (or quote gate) already fired: this row is not quotable at q_t
                continue
            t_fill = _ns(e["t_fill_ns"][j])
            hedge = _ns(e["hedge_ns"][j])
            deadline = cancel_at + self.cfg.cancel_ns if cancel_at is not None else withdraw
            if t_fill is not None and t_fill in self.used_exits.get((route, p.vc), ()):
                j += 1                      # this fill already closed another position of the product: queue behind
                continue
            if t_fill is not None and q_t < t_fill <= deadline and hedge is not None:
                return (j, q_t, t_fill, hedge, int(e["price"][j]), int(e["hedge_vwap"][j]), float(qab[j]),
                        float(e["actual_ab"][j]), cancel_at, route)
            if cancel_at is None:
                return None                 # working until withdrawal, never filled
            cursor = deadline
            j = max(j + 1, first(cursor))
        return None

    def close(self, p: Position, day: str, kind: str, spot_sell_cash: int, fut_buy_px: int, ns: int):
        p.state, p.close_day, p.close_kind = "closed", day, kind
        p.spot_sell_cash, p.fut_buy_px, p.exit_hedge_ns = spot_sell_cash, fut_buy_px, ns
        p.exit_realized_ab = (fut_buy_px * p.shares / spot_sell_cash - 1.0) * 10_000.0 if spot_sell_cash else None
        p.pnl_spot = (spot_sell_cash - p.spot_buy_cash) / 10_000.0
        p.pnl_fut = (p.fut_sell_px - fut_buy_px) * p.shares / 10_000.0
        fee_bp = self.cfg.cost.fee_same_day_bp if day == p.quote_day else self.cfg.cost.fee_overnight_bp
        p.fees = p.notional_twd * fee_bp / 10_000.0
        p.pnl_net = p.pnl_spot + p.pnl_fut - p.fees
        p.holding_days = (ns - p.hedge_ns) / 86_400e9
        self.open_ids.discard(p.id)
        self.committed -= p.notional_twd
        self.committed_by_vc[p.vc] = self.committed_by_vc.get(p.vc, 0.0) - p.notional_twd

    # ------------------------------------------------------------------ day
    def run_day(self, day: str, is_last: bool) -> dict:
        started = time.time()
        start = open_ns(day)
        end = start + CLOSE_SECOND * SECOND
        cfg = self.cfg
        reach_table = reach.fit(day, self.window)
        if not reach_table.days:
            reach_table = None
        abs_table = abs_reach.AbsTable.fit(self.samples, as_of=day)
        hurdle_day, n_signals_prev = self.dynamic_hurdle()
        cost_day = CostConfig(**{**cfg.cost.__dict__, "hurdle_bp_per_day": hurdle_day})
        decider = Decider(day, reach_table, abs_table, cfg, cost=cost_day)
        entries = load_entries(day, cfg)
        exits = load_exits(day, cfg.exit_levels, cfg.exit_routes)
        self.used_exits = {}
        self.gates = GateBook(day, cfg.gates_tag) if cfg.gates_tag else None
        gates = self.gates
        t_load = time.time() - started
        events: list = []
        serial = 0

        def push(ns, kind, payload):
            nonlocal serial
            serial += 1
            heapq.heappush(events, (ns, kind, serial, payload))

        # carried positions: plan today's exit from 09:05
        # carried positions plan today's exits in opening order (a set would iterate in hash order, which differs
        # between processes and, with exit fills consumed once, would make runs non-reproducible)
        for pid in sorted(self.open_ids, key=lambda k: (self.positions[k].hedge_ns, k)):
            p = self.positions[pid]
            plan = self.plan_exit(p, exits, day, start, start + QUOTE_START_SECOND * SECOND)
            if plan is not None:
                push(plan[3], "close", (pid, plan))
        cursor: dict[str, int] = {}
        live: dict[str, str] = {}          # vc -> position id while quoting/hedging
        open_by_vc: dict[str, int] = {}
        for pid in self.open_ids:
            open_by_vc[self.positions[pid].vc] = open_by_vc.get(self.positions[pid].vc, 0) + 1
        product_cap = lambda notional: max(notional, cfg.product_cap_frac * cfg.cap_twd)
        stats = dict(candidates=entries.height, gate_rejects=0, evaluated=0, admitted=0, filled=0, rollbacks=0, cap_rejects=0, product_cap_rejects=0,
                     hurdle_rejects=0, no_estimate=0)
        rows = entries.to_dicts()
        next_by_vc: dict[str, list[int]] = {}
        for i, r in enumerate(rows):
            next_by_vc.setdefault(r["vc"], []).append(i)
        seg_end = {}
        for vc, idxs in next_by_vc.items():
            for a, b in zip(idxs, idxs[1:] + [None]):
                seg_end[a] = rows[b]["quote_ns"] if b is not None else None
        for i, r in enumerate(rows):
            t0 = r["quote_ns"]
            while events and events[0][0] <= t0:
                _, kind, _, payload = heapq.heappop(events)
                self._event(kind, payload, day, exits, start, push)
            vc = r["vc"]
            if r["stream"] not in cfg.streams:
                continue
            if vc in live and self.positions[live[vc]].state != "hedging":
                del live[vc]                    # entry hedge completed: the product may quote again
            if vc in live or (cfg.max_positions_per_product is not None and open_by_vc.get(vc, 0) >= cfg.max_positions_per_product):
                continue
            c = cursor.get(vc, -1)
            if seg_end[i] is not None and seg_end[i] <= c:
                continue
            if r["stream"] == "S1" and c > t0:
                continue        # spot queue position belongs to the row's own time: re-join on the next row
            q_t = max(t0, c) + cfg.place_ns                     # live time of the entry quote
            leg = "S1_buy" if r["stream"] == "S1" else "S2_sell"
            gate_bad = gates.next_bad(leg, vc, q_t) if gates is not None else None
            if gate_bad is not None and gate_bad <= q_t:
                stats["gate_rejects"] += 1
                continue
            stats["evaluated"] += 1
            d = decider.decide(r)
            if not d.admit:
                stats["hurdle_rejects" if d.reason == "hurdle" else "no_estimate"] += 1
                continue
            if self.committed + r["notional_twd"] > cfg.cap_twd:
                stats["cap_rejects"] += 1
                continue
            if self.committed_by_vc.get(vc, 0.0) + r["notional_twd"] > product_cap(r["notional_twd"]):
                stats["product_cap_rejects"] += 1
                continue
            triggers = [v for v in (r[f"t_below_{cfg.floor_bp}"], r["t_gate_ns"], r["t_ab0_ns"], gate_bad) if v is not None]
            if any(v <= q_t for v in triggers) or (r["t_fill_ns"] is not None and r["t_fill_ns"] <= q_t):
                continue
            stats["admitted"] += 1
            cancel_at = min(triggers) if triggers else None
            deadline = cancel_at + cfg.cancel_ns if cancel_at is not None else start + MAKER_WITHDRAW_SECOND * SECOND
            t_fill = r["t_fill_ns"]
            filled = t_fill is not None and t_fill <= deadline and r["hedge_ns"] is not None
            if not filled:
                partial = r["t_partial_ns"] is not None and r["t_partial_ns"] > q_t and r["t_partial_ns"] <= deadline
                if partial:
                    stats["rollbacks"] += 1
                    spread_bp = (r["price"] - r["spot_b1"]) / r["price"] * 10_000.0 if r["spot_b1"] else 10.0
                    self.rollbacks.append(dict(day=day, vc=vc, quote_ns=q_t, price=r["price"], shares=1000,
                                               cost_twd=-(spread_bp + cfg.cost.fee_same_day_bp) / 10_000.0 * r["price"] * 1000 / 10_000.0))
                cursor[vc] = deadline
                continue
            stats["filled"] += 1
            best = d.best
            self._counter += 1
            shares = int(round(r["notional_twd"] * 10_000 / (r["spot_a1"] if r["stream"] == "S2" else r["price"])))
            if r["stream"] == "S1":
                spot_buy_cash, fut_sell_px = r["price"] * shares, r["hedge_vwap"]
            else:
                spot_buy_cash, fut_sell_px = r["hedge_vwap"] * shares, r["price"]
            scale = r["scale"] or 1.0
            p = Position(f"{r['stream']}/{day}/{vc}/{self._counter}", vc, r["qc"], r["stream"], r["expiry"], shares,
                         best.route, best.x, exit_target_bp(best, r["anchor"], scale, d.evals, cfg.abs_target_mode),
                         best.ev_bp, best.score,
                         best.t_days, best.p_sd, day, q_t, r["quote_ab"], r["anchor"], scale, r["price"], t_fill,
                         r["hedge_ns"], int(spot_buy_cash), int(fut_sell_px), float(r["actual_ab"]),
                         entry_cancel_at=cancel_at, entry_race=bool(cancel_at is not None and t_fill > cancel_at),
                         state="hedging")
            self.positions[p.id] = p
            live[vc] = p.id
            open_by_vc[vc] = open_by_vc.get(vc, 0) + 1
            self.committed += p.notional_twd
            self.committed_by_vc[vc] = self.committed_by_vc.get(vc, 0.0) + p.notional_twd
            push(p.hedge_ns, "open", p.id)
            cursor[vc] = p.hedge_ns
            self.decisions_log.append(dict(day=day, vc=vc, stream=r["stream"], quote_ns=q_t, route=best.route,
                                           x=best.x, ev=best.ev_bp, score=best.score, quote_ab=r["quote_ab"]))
        while events:
            _, kind, _, payload = heapq.heappop(events)
            self._event(kind, payload, day, exits, start, push)
            live = {vc: pid for vc, pid in live.items() if self.positions[pid].state == "hedging"}
        # end of day: settle expiring positions, mark the rest as carry
        settled = 0
        expiring = sorted(pid for pid in self.open_ids if self.positions[pid].expiry <= day)
        if expiring:
            marks = close_marks(day, sorted({self.positions[pid].vc for pid in expiring}))
            for pid in expiring:
                p = self.positions[pid]
                if p.vc in marks:
                    # final settlement: the futures settle at the spot close, the spot is sold at the same
                    # close (basis 0); d_settle (closing-auction shortfall) is a cost-table item, not modelled here
                    spot_mid, _fut_mid = marks[p.vc]
                    self.close(p, day, "settlement", spot_mid * p.shares, spot_mid, end)
                    settled += 1
        filled_scores = [p.score for p in self.positions.values() if p.quote_day == day]
        self.signal_hist.append((day, decider.admitted_scores(cfg.cost.hurdle_bp_per_day) if cfg.dyn_source == "admitted" else filled_scores,
                                 self.committed))
        closed_today = [p for p in self.positions.values() if p.close_day == day]
        row = dict(day=day, **stats, closed=len(closed_today), settled=settled,
                   pnl_net=round(sum(p.pnl_net for p in closed_today), 0),
                   rollback_cost=round(sum(r["cost_twd"] for r in self.rollbacks if r["day"] == day), 0),
                   open_end=len(self.open_ids), committed_end=round(self.committed, 0),
                   reach_days=len(reach_table.days) if reach_table else 0, ev_calls=decider.calls,
                   hurdle_used=round(hurdle_day, 2), signals_prev=n_signals_prev, signals_today=len(self.signal_hist[-1][1]),
                   load_s=round(t_load, 1), elapsed_s=round(time.time() - started, 1))
        self.daily.append(row)
        return row

    def dynamic_hurdle(self) -> tuple[float, int]:
        """Today's hurdle (bp per calendar day) and the number of prior signals it was derived from."""
        cfg = self.cfg
        base = cfg.cost.hurdle_bp_per_day
        if cfg.dyn_q is None or not self.signal_hist:
            return base, 0
        recent = self.signal_hist[-cfg.dyn_window:]
        scores = [x for _, sc, _ in recent for x in sc]
        if not scores:
            return base, 0
        if cfg.dyn_cap_frac > 0 and recent[-1][2] < cfg.dyn_cap_frac * cfg.cap_twd:
            return base, len(scores)
        return max(base, float(np.quantile(scores, cfg.dyn_q))), len(scores)

    def _event(self, kind: str, payload, day: str, exits: dict, start: int, push) -> None:
        if kind == "open":
            p = self.positions[payload]
            p.state = "open"
            self.open_ids.add(p.id)
            plan = self.plan_exit(p, exits, day, start, p.hedge_ns)
            if plan is not None:
                push(plan[3], "close", (p.id, plan))
        elif kind == "close":
            pid, plan = payload
            p = self.positions[pid]
            if p.state != "open":
                return
            j, q_t, t_fill, hedge_ns, price, hedge_vwap, qab, realized, cancel_at, route, double = plan
            p.exit_quote_ns, p.exit_price, p.exit_quote_ab, p.exit_fill_ns = q_t, price, qab, t_fill
            p.exit_cancel_at, p.exit_race = cancel_at, bool(cancel_at is not None and t_fill > cancel_at)
            p.exit_route, p.exit_double_risk = route, bool(double)
            if route == "E1":       # spot sold at our ask, futures bought by the hedge
                self.close(p, day, "maker_exit", price * p.shares, hedge_vwap, hedge_ns)
            else:                   # E2: futures bought at our bid, spot sold by the hedge (per-share vwap)
                self.close(p, day, "maker_exit", hedge_vwap * p.shares, price, hedge_ns)

    # ------------------------------------------------------------------ run
    def run(self, days: list[str]) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        for i, day in enumerate(days):
            row = self.run_day(day, i == len(days) - 1)
            print(json.dumps(row), flush=True)
        self.finish(days)

    def finish(self, days: list[str]) -> None:
        # positions still open at the end: mark at the last day's close marks (unrealized)
        if self.open_ids:
            marks = close_marks(days[-1], sorted({self.positions[pid].vc for pid in self.open_ids}))
            for pid in sorted(self.open_ids):
                p = self.positions[pid]
                if p.vc in marks:
                    spot_mid, fut_mid = marks[p.vc]
                    self.close(p, days[-1], "open_marked", spot_mid * p.shares, fut_mid, open_ns(days[-1]) + CLOSE_SECOND * SECOND)
        pos = pl.from_dicts([asdict(p) for p in self.positions.values()], infer_schema_length=None) if self.positions else pl.DataFrame()
        pos.write_parquet(self.out / "positions.parquet")
        pl.from_dicts(self.daily, infer_schema_length=None).write_csv(self.out / "daily.csv")
        if self.rollbacks:
            pl.from_dicts(self.rollbacks, infer_schema_length=None).write_csv(self.out / "rollbacks.csv")
        (self.out / "config.json").write_text(json.dumps({k: (v if not hasattr(v, "__dict__") else v.__dict__)
                                                          for k, v in self.cfg.__dict__.items()}, indent=1, default=str) + "\n")
        print(summarize(pos, self.daily, self.rollbacks, self.cfg))


def summarize(pos: pl.DataFrame, daily: list[dict], rollbacks: list[dict], cfg: PolicyConfig) -> str:
    if not pos.height:
        return "no positions"
    closed = pos.filter(pl.col("close_kind").is_in(["maker_exit", "settlement"]))
    lines = []
    n_days = len(daily)
    total = float(pos["pnl_net"].fill_null(0).sum()) + sum(r["cost_twd"] for r in rollbacks)
    lines.append(f"days {n_days} | pairs {pos.height} (closed {closed.height}, marked open {pos.filter(pl.col('close_kind') == 'open_marked').height}) | "
                 f"rollbacks {len(rollbacks)} | net PnL {total:,.0f} TWD = {total / max(n_days, 1):,.0f}/day")
    for key in ("stream", "route", "close_kind"):
        g = pos.group_by(key).agg(pl.len().alias("n"), pl.col("pnl_net").sum().alias("pnl"),
                                  (pl.col("pnl_net") / (pl.col("spot_buy_cash") / 1e4) * 1e4).mean().alias("bp_mean"),
                                  pl.col("holding_days").mean().alias("hold_days"),
                                  pl.col("ev_bp").mean().alias("ev_pred"), pl.col("t_days_pred").mean().alias("t_pred")).sort(key)
        lines.append(f"by {key}:")
        for r in g.iter_rows(named=True):
            lines.append(f"  {str(r[key]):12s} n {r['n']:5d} pnl {r['pnl'] or 0:>12,.0f} | realized {r['bp_mean'] or 0:6.1f} bp vs EV {r['ev_pred']:6.1f} | "
                         f"hold {r['hold_days'] or 0:5.2f} d vs pred {r['t_pred']:5.2f}")
    lines.append("entry slippage quote_ab - actual_ab (bp) by stream x race:")
    g = (pos.with_columns((pl.col("quote_ab") - pl.col("actual_ab")).alias("slip"))
         .group_by("stream", "entry_race").agg(pl.len().alias("n"), pl.col("slip").mean().alias("mean"),
                                                 pl.col("slip").median().alias("med"), pl.col("slip").quantile(0.9).alias("p90"))
         .sort("stream", "entry_race"))
    for r in g.iter_rows(named=True):
        lines.append(f"  {r['stream']} race={str(r['entry_race']):5s} n {r['n']:5d} mean {r['mean']:6.1f} med {r['med']:6.1f} p90 {r['p90']:6.1f}")
    exit_rows = pos.filter(pl.col("close_kind") == "maker_exit")
    if exit_rows.height:
        lines.append("exit slippage realized - quote basis (bp) by race:")
        g = (exit_rows.with_columns((pl.col("exit_realized_ab") - pl.col("exit_quote_ab")).alias("slip"),
                                    (pl.col("exit_realized_ab") - pl.col("target_bp")).alias("vs_target"))
             .group_by("exit_race").agg(pl.len().alias("n"), pl.col("slip").mean().alias("mean"), pl.col("slip").median().alias("med"),
                                        pl.col("vs_target").mean().alias("vs_target"),
                                        ((pl.col("exit_fill_ns") - pl.col("exit_quote_ns")) / 1e9).median().alias("wait_s"))
             .sort("exit_race"))
        for r in g.iter_rows(named=True):
            lines.append(f"  race={str(r['exit_race']):5s} n {r['n']:5d} mean {r['mean']:6.1f} med {r['med']:6.1f} | vs target {r['vs_target']:6.1f} | wait med {r['wait_s']:7.1f} s")
    same_day = float((pos["close_day"] == pos["quote_day"]).mean())
    lines.append(f"same-day close share {same_day:.3f} | mean committed at close {np.mean([d['committed_end'] for d in daily]):,.0f} | "
                 f"max open {max(d['open_end'] for d in daily)}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--out", default="v1")
    parser.add_argument("--preset", choices=sorted(PRESETS), help="frozen policy preset; explicit flags override its fields")
    parser.add_argument("--hurdle", type=float, default=8.5, help="bp per trading day")
    parser.add_argument("--streams", default="S1,S2")
    parser.add_argument("--residual-min", type=float, default=25.0)
    parser.add_argument("--d-in-s1", type=float, help="override CostConfig.d_in_base['S1'] (bp)")
    parser.add_argument("--d-in-s2", type=float, help="override CostConfig.d_in_base['S2'] (bp)")
    parser.add_argument("--d-out", type=float, help="override CostConfig.d_out_base for both streams (bp)")
    parser.add_argument("--exit-routes", default="E1", help="comma list of E1,E2")
    parser.add_argument("--gates", help="quote gate tag under data/gates/ (see backtest/gates.py)")
    parser.add_argument("--abs-target", default="zero", choices=["zero", "max_q", "max_q_pos"],
                        help="exit level of absolute-route entries")
    parser.add_argument("--dyn-q", type=float, help="dynamic hurdle: quantile of the previous days' admitted signal scores")
    parser.add_argument("--dyn-window", type=int, default=1)
    parser.add_argument("--dyn-cap-frac", type=float, default=0.0, help="apply only when the previous day closed >= frac x cap")
    parser.add_argument("--dyn-source", default="admitted", choices=["admitted", "filled"])
    args = parser.parse_args()
    days = [d for d in grid_days() if points_path(d, "s1_entries").exists() and points_path(d, "s2_entries").exists()
            and (not args.start or d >= args.start) and (not args.end or d <= args.end)]
    if not days:
        raise SystemExit("no days with both point tables in range")
    cost = CostConfig()
    d_in = dict(cost.d_in_base)
    if args.d_in_s1 is not None:
        d_in["S1"] = args.d_in_s1
    if args.d_in_s2 is not None:
        d_in["S2"] = args.d_in_s2
    d_out = dict(cost.d_out_base) if args.d_out is None else {"S1": args.d_out, "S2": args.d_out}
    cost = CostConfig(d_in_base=d_in, d_out_base=d_out)
    fields = dict(hurdle_bp_per_trading_day=args.hurdle, streams=tuple(args.streams.split(",")), residual_min_bp=args.residual_min,
                  cost=cost, exit_routes=tuple(args.exit_routes.split(",")), gates_tag=args.gates, abs_target_mode=args.abs_target,
                  dyn_q=args.dyn_q, dyn_window=args.dyn_window, dyn_cap_frac=args.dyn_cap_frac, dyn_source=args.dyn_source)
    if args.preset:
        defaults = {k: parser.get_default(k) for k in ("exit_routes", "gates", "abs_target", "dyn_q", "dyn_window", "dyn_cap_frac")}
        given = {"exit_routes": args.exit_routes != defaults["exit_routes"], "gates_tag": args.gates != defaults["gates"],
                 "abs_target_mode": args.abs_target != defaults["abs_target"], "dyn_q": args.dyn_q != defaults["dyn_q"],
                 "dyn_window": args.dyn_window != defaults["dyn_window"], "dyn_cap_frac": args.dyn_cap_frac != defaults["dyn_cap_frac"]}
        for k, v in PRESETS[args.preset].items():
            if not given.get(k, False):
                fields[k] = v
    cfg = PolicyConfig(**fields)
    Replay(cfg, DATA_ROOT / "backtest" / args.out).run(days)


if __name__ == "__main__":
    main()
