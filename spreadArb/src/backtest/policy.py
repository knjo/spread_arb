"""Stage 3 policy: which candidate rows to quote, and the exit target of a filled entry."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import isfinite
from bisect import bisect_left, bisect_right

from ..common.books import tick_i

from ..ev.config import CostConfig
from ..ev.ev import Decision, ExitEval, Quote, choose, evaluate, horizon


@dataclass(frozen=True)
class PolicyConfig:
    cap_twd: float = 20_000_000.0            # actual spot notional, including partial/unhedged entry
    hurdle_bp_per_trading_day: float = 8.5   # 21.25% over 250 sessions; applied per calendar day below
    ev_min_bp: float = 0.0
    floor_bp: int = 20                       # entry deterioration floor used for cancellation
    residual_min_bp: float = 25.0            # minimum eff_u at quote time (keeps the S2 admission comparable)
    depth_mult: int = 5                      # S2: spot A1 shares >= depth_mult x contract shares
    s1_levels: tuple = (0,)                  # S1 entry ladder levels used (0 = B1; -1 = inside; 1 = B1-1)
    exit_levels: tuple = (0,)                # E1 ladder levels used (0 = A1; -1 = inside)
    exit_tol_bp: float = 5.0                 # exit quote is cancelled once its basis rises above target + tol
    max_positions_per_product: int | None = None   # count limit per product; None = only the notional cap below
    product_cap_frac: float = 0.25           # per-product spot notional <= max(one lot, frac x cap_twd)
    exit_routes: tuple = ("E1",)             # E1 = spot ask maker -> buy futures; E2 = futures bid maker -> sell spot
    reserve_on_submit: bool = False          # historical full-reservation variant only when explicitly requested
    gates_tag: str | None = None             # quote gates (backtest/gates.py); None = no gates
    # dynamic hurdle: today's hurdle = max(base, quantile q of the previous `window` days' admitted signal scores);
    # with cap_frac > 0 it only applies when the previous day closed with committed >= cap_frac x cap
    dyn_q: float | None = None
    dyn_window: int = 1
    dyn_cap_frac: float = 0.0
    dyn_source: str = "admitted"             # "admitted" = distinct admitted signals; "filled" = scores of the pairs actually filled
    # exit level of an absolute-route entry: "zero" = basis 0; "max_q" = the higher of 0 and the Q-table (residual
    # route) target; "max_q_pos" = same, but only when exiting at the Q target has non-negative EV
    abs_target_mode: str = "zero"
    rise_grid: tuple = (0, 5, 10, 20, 30)
    cancel_ns: int = 50_000_000              # cancel latency: fills inside it after the guard fires count (race)
    place_ns: int = 50_000_000               # placement latency: the quote is live this long after the decision
    quote_refresh_ns: int = 60_000_000_000   # each live route is cancelled/re-evaluated after 60 seconds
    streams: tuple = ("S1", "S2")
    x_grid: tuple = (0.0, -0.25, -0.5, -1.0)
    cost: CostConfig = field(default_factory=CostConfig)

    def __post_init__(self):
        if self.quote_refresh_ns <= 0:
            raise ValueError("quote_refresh_ns must be positive")
        per_cal_day = self.hurdle_bp_per_trading_day * 250.0 / 365.0
        object.__setattr__(self, "cost", CostConfig(**{**self.cost.__dict__, "hurdle_bp_per_day": per_cal_day,
                                                       "ev_min_bp": self.ev_min_bp}))


from functools import lru_cache

# Frozen policy presets (2026-09-22). Both: E1+E2 exits, user's quote gates, absolute-route exit target = max(0, Q target),
# any number of positions per product under a per-product cap of max(1 lot, 25% x cap), 8.5 bp/trading day base hurdle.
PRESETS = {
    "A": dict(exit_routes=("E1", "E2"), gates_tag="gap1_b0.3_a0.2", abs_target_mode="max_q"),                      # fixed hurdle
    "B": dict(exit_routes=("E1", "E2"), gates_tag="gap1_b0.3_a0.2", abs_target_mode="max_q",
              dyn_q=0.5, dyn_window=1, dyn_cap_frac=0.8),                                                          # q50 after a cap day
}

horizon_cached = lru_cache(maxsize=4096)(horizon)


class Decider:
    """Exact decision inputs; signal identity includes the product, independent of admission."""

    def __init__(self, day: str, reach_table, abs_table, cfg: PolicyConfig, cost: CostConfig | None = None):
        self.day, self.reach, self.abs, self.cfg = day, reach_table, abs_table, cfg
        self.cost = cost if cost is not None else cfg.cost
        self.cache: dict[tuple, Decision] = {}
        self.calls = 0

    def decide(self, row: dict) -> Decision:
        key = tuple(row.get(k) for k in ("vc", "qc", "stream", "quote_ab", "anchor", "quote_second", "expiry",
                                        "scale", "scale_raw", "e_norm", "resid_mid_bp", "tick_bp_hedge",
                                        "execution_floor_bp", "spot_a1"))
        d = self.cache.get(key)
        if d is None:
            self.calls += 1
            d = self.cache[key] = decide_row(row, self.day, self.reach, self.abs, self.cfg, self.cost)
        return d

    def admitted_scores(self, base_hurdle: float | None = None) -> list[float]:
        """EV/day of every distinct signal evaluated today (one per cache key) that clears the BASE hurdle.
        Using the base (not today's raised) hurdle keeps the set comparable day to day; otherwise a raised hurdle
        truncates today's set, tomorrow's quantile rises further, and the threshold ratchets upward."""
        base = self.cfg.cost.hurdle_bp_per_day if base_hurdle is None else base_hurdle
        return [d.best.score for d in self.cache.values()
                if d.best is not None and d.reason in ("ok", "hurdle") and d.best.score >= base]


class StreamingDecider(Decider):
    """Same exact evaluations with bounded cache in chronological replay.

    quote_second is part of every cache key, so a previous second's entries can
    never be reused. Retain their independent scores, not millions of ExitEval objects.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.reach is not None and hasattr(self.reach, "e_edges"):
            self.reach = MemoReach(self.reach)
        if self.abs is not None:
            self.abs = MemoAbsolute(self.abs)
        self.second = -1
        self.scores = []

    def decide(self, row):
        second = row["quote_second"]
        if second < self.second:
            raise ValueError("decision rows must be chronological")
        if second != self.second:
            self.scores.extend(super().admitted_scores())
            self.cache.clear()
            self.second = second
        return super().decide(row)

    def admitted_scores(self, base_hurdle=None):
        if base_hurdle is not None and abs(base_hurdle-self.cfg.cost.hurdle_bp_per_day) > 1e-10:
            raise ValueError("streaming signal history uses the frozen base hurdle")
        return self.scores + super().admitted_scores()


class MemoReach:
    """Cache table lookups by the table's exact cells, without rounding EV inputs."""
    def __init__(self, table):
        self.table, self.cache = table, {}

    def _get(self, kind, e, x, t, k):
        key = (kind, bisect_right(self.table.e_edges, e), x,
               bisect_right((3600, 9000), t), bisect_left(self.table.k_edges, k))
        if key not in self.cache:
            if kind == "c0":
                cell = self.table.lookup_c0(e, x, t, k)
            elif kind == "c1":
                cell = self.table.lookup_c1(e, x, k)
            else:
                cell = self.table.lookup_q(x, k, e)
            self.cache[key] = cell
        return self.cache[key]

    def lookup_c0(self, e, x, t_sec, k_days):
        return self._get("c0", e, x, t_sec, k_days)

    def lookup_c1(self, e, x, k_days):
        return self._get("c1", e, x, 0, k_days)

    def lookup_q(self, x, k_days, e=None):
        return self._get("q", e, x, 0, k_days)


class MemoAbsolute:
    def __init__(self, table):
        self.table, self.cache = table, {}

    def lookup(self, ab, k):
        if ab < 50.0:
            return None
        key = (bisect_left((75.0, 125.0, 175.0), ab), bisect_left((0, 2, 5, 10, 15), k))
        if key not in self.cache:
            self.cache[key] = self.table.lookup(ab, k)
        return self.cache[key]


def decide_row(row: dict, day: str, reach_table, abs_table, cfg: PolicyConfig, cost: CostConfig | None = None) -> Decision:
    """Evaluate one candidate entry row (S1 buy / S2 sell) with the as-of tables."""
    cost = cost if cost is not None else cfg.cost
    if row.get("scale_raw") is not None and row["scale_raw"] > 60.0:
        return Decision(False, "scale_excluded", None, ())
    if not isfinite(row["anchor"]) or row["anchor"] < cost.min_anchor_bp:
        return Decision(False, "anchor", None, ())
    if row["quote_ab"] <= 0.0:
        return Decision(False, "basis", None, ())
    scale = row["scale"] or 1.0
    e = row["e_norm"] if row["e_norm"] is not None else row["resid_mid_bp"] / scale
    if row["scale"] is None or not isfinite(e):
        reach_table = None             # absolute/settlement need no normalized residual
    floor = row.get("execution_floor_bp", 0.0)
    if row["stream"] == "S2" and row.get("spot_a1"):
        floor = max(floor, tick_i(row["spot_a1"]) / row["spot_a1"] * 5000.0)
    quote = Quote(row["stream"], row["quote_ab"], row["anchor"], scale, e, row["quote_second"],
                  route="E1", tick_bp=row["tick_bp_hedge"], execution_floor_bp=floor)
    h = horizon_cached(day, row["expiry"])
    evals = evaluate(quote, h, reach_table, cost, cfg.x_grid, abs_table=abs_table)
    # Adopted 2026-09-29: use zero-crossing EV as the entry screen.
    # Actor still uses cfg.abs_target_mode=max_q for actual exits.
    return choose(evals, cost, quote)


def q_target_bp(evals, anchor: float, scale: float, *, min_ev: float | None = None) -> float | None:
    """Exit level of the best-scoring residual-route evaluation (anchor + x * scale); None if there is none."""
    cands = [v for v in evals if v.route == "residual" and v.x is not None and v.t_days > 0
             and (min_ev is None or v.ev_bp >= min_ev)]
    if not cands:
        return None
    return anchor + max(cands, key=lambda v: v.score).x * scale


def exit_target_bp(best: ExitEval, anchor: float, scale: float, evals=(), mode: str = "zero") -> float | None:
    """Absolute basis (bp) at which the exit maker order is placed; None = hold to settlement."""
    if best.route == "absolute":
        if mode == "zero":
            return 0.0
        q = q_target_bp(evals, anchor, scale, min_ev=0.0 if mode == "max_q_pos" else None)
        return max(0.0, q) if q is not None else 0.0
    if best.route == "settle":
        return None
    return anchor + best.x * scale
