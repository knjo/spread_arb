"""Pure EV: reach probabilities x cost settings -> exit level choice. No I/O.

For each candidate exit level x the position ends in exactly one of: same-day
exit, exit on a later session j = 1..K, or settlement at expiry (basis 0).
score = EV / expected calendar days held, compared with a per-day hurdle.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from ..common.calendar import calendar_days, next_session, sessions_after
from ..common.paths import CLOSE_SECOND, SECONDS_PER_DAY
from .config import CostConfig

SETTLE = None   # sentinel exit level: hold to expiry


class CellLike(Protocol):
    n: int
    p: float
    level: int


class ReachLookup(Protocol):
    def lookup_c0(self, e: float, x: float, t_sec: int, k_days: int) -> CellLike | None: ...
    def lookup_c1(self, e: float, x: float, k_days: int) -> CellLike | None: ...
    def lookup_q(self, x: float, k_days: int, e: float | None = None) -> CellLike | None: ...


@dataclass(frozen=True)
class Horizon:
    """Planned sessions after the decision day through expiry, as calendar-day offsets."""
    session_offsets: tuple[int, ...]   # j = 1..K; the last one is the expiry day
    settle_offset: int                 # calendar days to expiry; add today's remaining fraction
    k_days: int                        # calendar days to expiry

    @property
    def K(self) -> int:
        return len(self.session_offsets)


def horizon(decision_day: str, expiry: str, as_of: str | None = None) -> Horizon:
    if expiry < decision_day:
        raise ValueError("contract already expired")
    as_of = as_of or decision_day
    sessions = sessions_after(decision_day, expiry, as_of)
    return Horizon(tuple(calendar_days(decision_day, s) for s in sessions),
                   calendar_days(decision_day, expiry), calendar_days(decision_day, expiry))


@dataclass(frozen=True)
class Branches:
    p_sd: float
    p_later: tuple[float, ...]   # j = 1..K
    p_never: float

    @property
    def p_on(self) -> float:
        return sum(self.p_later)


def branches(c0: float, c1: float | None, q: float | None, K: int) -> Branches:
    if not 0.0 <= c0 <= 1.0:
        raise ValueError("c0 must be a probability")
    if K == 0:
        return Branches(c0, (), 1.0 - c0)
    c1 = c0 if c1 is None else max(c0, min(c1, 1.0))
    q = 0.0 if q is None else min(max(q, 0.0), 1.0)
    remaining = 1.0 - c1
    later = [c1 - c0]
    for j in range(2, K + 1):
        later.append(remaining * (1.0 - q) ** (j - 2) * q)
    never = remaining * (1.0 - q) ** (K - 1)
    return Branches(c0, tuple(later), never)


def expected_days(b: Branches, today_remaining_days: float, h: Horizon) -> float:
    return (b.p_sd * today_remaining_days
            + sum(p * (off + today_remaining_days) for p, off in zip(b.p_later, h.session_offsets))
            + b.p_never * (h.settle_offset + today_remaining_days))


@dataclass(frozen=True)
class ExitEval:
    x: float | None
    b_x_bp: float | None
    p_sd: float
    p_on: float
    p_never: float
    ev_bp: float
    t_days: float
    score: float
    surplus: float
    d_in: float
    d_out: float
    c0: float | None = None
    c1: float | None = None
    q: float | None = None
    n_min: int | None = None
    level_max: int | None = None
    fallback: str = ""
    route: str = "residual"      # residual (anchor-relative x) / settle / absolute (taker convergence table)


@dataclass(frozen=True)
class Quote:
    stream: str
    quote_ab: float          # basis locked by our maker price vs the other leg's executable price, bp
    anchor: float            # frozen anchor, bp
    scale: float             # product scale, bp per unit
    e: float                 # entry level in scale units (market state at decision time)
    t_sec: int
    route: str = "E1"
    tick_bp: float | None = None
    execution_floor_bp: float = 0.0


def evaluate_exit(x: float | None, quote: Quote, h: Horizon, lookup: ReachLookup | None,
                  cfg: CostConfig) -> ExitEval | None:
    d_in = cfg.d_in_bp(quote.stream, quote.tick_bp, quote.t_sec, quote.execution_floor_bp)
    g0 = quote.quote_ab - d_in
    today_remaining = (CLOSE_SECOND - quote.t_sec) / SECONDS_PER_DAY
    settle_fee = cfg.fee_same_day_bp if h.k_days == 0 else cfg.fee_overnight_bp
    if x is SETTLE:
        d_settle = cfg.d_settle_bp()
        ev = g0 - d_settle - settle_fee
        t = float(h.settle_offset) + today_remaining
        return ExitEval(None, None, 0.0, 0.0, 1.0, ev, t, ev / t, ev - cfg.hurdle_bp_per_day * t, d_in, d_settle,
                        route="settle")
    if lookup is None:
        return None
    c0 = lookup.lookup_c0(quote.e, x, quote.t_sec, h.k_days)
    if c0 is None:
        return None
    c1 = lookup.lookup_c1(quote.e, x, h.k_days) if h.K >= 1 else None
    qc = lookup.lookup_q(x, h.k_days, quote.e) if h.K >= 2 else None
    fallback = []
    if h.K >= 1 and c1 is None:
        fallback.append("c1=c0")
    if h.K >= 2 and qc is None:
        fallback.append("q=0")
    b = branches(c0.p, c1.p if c1 else None, qc.p if qc else None, h.K)
    d_out = cfg.d_out_bp(quote.stream, quote.route)
    b_x = quote.anchor + x * quote.scale
    g_x = quote.quote_ab - d_in - b_x - d_out
    ev = (b.p_sd * (g_x - cfg.fee_same_day_bp) + b.p_on * (g_x - cfg.fee_overnight_bp)
          + b.p_never * (g0 - cfg.d_settle_bp() - settle_fee))
    t = expected_days(b, today_remaining, h)
    cells = [c for c in (c0, c1, qc) if c is not None]
    return ExitEval(x, b_x, b.p_sd, b.p_on, b.p_never, ev, t, ev / t if t > 0 else float("nan"),
                    ev - cfg.hurdle_bp_per_day * t, d_in, d_out, c0.p, c1.p if c1 else None,
                    qc.p if qc else None, min(c.n for c in cells), max(c.level for c in cells), ",".join(fallback))


def evaluate_absolute(quote: Quote, h: Horizon, abs_table, cfg: CostConfig) -> ExitEval | None:
    """Absolute route: the basis converges to 0 (basis_buy_taker <= 0) by the taker line's first-crossing
    statistics, keyed by expected post-hedge basis and trading days to settlement; whatever has not
    converged by expiry settles at basis 0."""
    if abs_table is None:
        return None
    d_in = cfg.d_in_bp(quote.stream, quote.tick_bp, quote.t_sec, quote.execution_floor_bp)
    ab_eff = quote.quote_ab - d_in
    cell = abs_table.lookup(ab_eff, h.K)
    if cell is None:
        return None
    d_out, d_settle = cfg.d_out_bp(quote.stream, quote.route), cfg.d_settle_bp()
    today_remaining = (CLOSE_SECOND - quote.t_sec) / SECONDS_PER_DAY
    g_x, g_0 = ab_eff - d_out, ab_eff
    ev = t = 0.0
    p_sd = p_on = 0.0
    p_never = cell.p_settle
    for j, pj in enumerate(cell.p_day):
        if pj <= 0.0:
            continue
        if j > h.K:                      # table day beyond this contract's remaining sessions: settles
            p_never += pj
            continue
        fee = cfg.fee_same_day_bp if j == 0 else cfg.fee_overnight_bp
        offset = today_remaining if j == 0 else float(h.session_offsets[j - 1]) + today_remaining
        ev += pj * (g_x - fee)
        t += pj * offset
        if j == 0:
            p_sd += pj
        else:
            p_on += pj
    # natural convergence after MAX_J trading days: spread it over the remaining sessions, else settle
    later_sessions = [o for i, o in enumerate(h.session_offsets, start=1) if i > len(cell.p_day) - 1]
    if cell.p_later > 0.0:
        if later_sessions:
            ev += cell.p_later * (g_x - cfg.fee_overnight_bp)
            t += cell.p_later * (float(np.mean(later_sessions)) + today_remaining)
            p_on += cell.p_later
        else:
            p_never += cell.p_later
    settle_fee = cfg.fee_same_day_bp if h.k_days == 0 else cfg.fee_overnight_bp
    ev += p_never * (g_0 - d_settle - settle_fee)
    t += p_never * (h.settle_offset + today_remaining)
    if t <= 0.0:
        return None
    return ExitEval(0.0, 0.0, p_sd, p_on, p_never, ev, t, ev / t, ev - cfg.hurdle_bp_per_day * t, d_in, d_out,
                    c0=cell.p_day[0], c1=cell.p_day[0] + cell.p_day[1], q=None, n_min=cell.n, level_max=cell.level,
                    fallback="", route="absolute")


def evaluate(quote: Quote, h: Horizon, lookup: ReachLookup | None, cfg: CostConfig,
             x_grid: Sequence[float] = (0.0, -0.25, -0.5, -1.0), settle: bool = True,
             abs_table=None) -> list[ExitEval]:
    evals = [evaluate_exit(x, quote, h, lookup, cfg) for x in x_grid]
    if settle:
        evals.append(evaluate_exit(SETTLE, quote, h, lookup, cfg))
    evals.append(evaluate_absolute(quote, h, abs_table, cfg))
    return [v for v in evals if v is not None]


@dataclass(frozen=True)
class Decision:
    admit: bool
    reason: str
    best: ExitEval | None
    evals: tuple[ExitEval, ...]


def choose(evals: Sequence[ExitEval], cfg: CostConfig, quote: Quote | None = None) -> Decision:
    if quote is not None:
        if quote.quote_ab <= 0.0:
            return Decision(False, "basis", None, tuple(evals))
        if quote.anchor < cfg.min_anchor_bp:
            return Decision(False, "anchor", None, tuple(evals))
    if not evals:
        return Decision(False, "no_estimate", None, ())
    candidates = [v for v in evals if v.ev_bp >= cfg.ev_min_bp and v.t_days > 0]
    if not candidates:
        return Decision(False, "ev_min", None, tuple(evals))
    best = max(candidates, key=lambda v: v.score)
    if best.score < cfg.hurdle_bp_per_day:
        return Decision(False, "hurdle", best, tuple(evals))
    return Decision(True, "ok", best, tuple(evals))
