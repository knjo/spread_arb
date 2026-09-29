"""Reach tables: how the market basis moves after a decision point.

Daily facts (one row per product x 30 s sampling point) store, relative to the
anchor frozen at the sampling point, the minimum of the exit-trigger series
(basis_buy_taker) over the rest of today, over the next session and over the
session after that. Labels are x-agnostic; thresholds are applied at fit time.
Nothing about our own fills or hedges enters these tables.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from ..common.calendar import calendar_days
from ..common.grid import ProductDay, load_day
from ..common.paths import (MAKER_WITHDRAW_SECOND, QUOTE_END_SECOND, QUOTE_START_SECOND,
                            facts_path, grid_days)
from . import qlevel

SAMPLE_STEP = 30
T_EDGES = (3600, 9000)             # decision second
# calendar days to expiry: 0 (settlement day) / 1 / 2-3 / >=4. The full-history
# profile is flat beyond four days; the last three days converge mechanically.
K_EDGES = (0, 1, 3)
# Coordinate presets: entry-level bucket edges and exit grid, in the coordinate's unit.
COORDS = {
    "residual_scaled": dict(e_edges=(1.0, 1.5, 2.5, 4.0), x_grid=(0.0, -0.25, -0.5, -1.0)),   # unit = scale_D-1
    "residual_bp": dict(e_edges=(15.0, 25.0, 50.0, 80.0), x_grid=(0.0, -5.0, -10.0, -20.0)),   # unit = 1 bp
}
E_EDGES = COORDS["residual_scaled"]["e_edges"]
X_GRID = COORDS["residual_scaled"]["x_grid"]
FACT_SCHEMA = {
    "day": pl.String, "ValueCode": pl.String, "QuoteCode": pl.String, "expiry": pl.String,
    "t0": pl.Int32, "resid_bp": pl.Float64, "anchor_bp": pl.Float64,
    "min_today_bp": pl.Float64, "min_d1_bp": pl.Float64, "min_d2_bp": pl.Float64,
    "d1_day": pl.String, "d2_day": pl.String, "d1_state": pl.String, "d2_state": pl.String,
    "k_days": pl.Int32,
}


def bucket(values, edges) -> np.ndarray:
    return np.searchsorted(np.asarray(edges, dtype=float), np.asarray(values, dtype=float), side="right")


def k_bucket(days, edges=K_EDGES) -> np.ndarray:
    # calendar days to expiry; each edge is the last day of its bucket (0 -> bucket 0, 1..3 -> 1 with edge 3, ...)
    return np.searchsorted(np.asarray(edges, dtype=float), np.asarray(days, dtype=float), side="left")


def _window_min(p: ProductDay | None, lo: int, hi: int) -> float:
    if p is None:
        return np.inf
    seg = p.buy[lo:hi + 1]
    mask = p.eligible[lo:hi + 1] & np.isfinite(seg)
    return float(seg[mask].min()) if mask.any() else np.inf


def _state(today: ProductDay, other: ProductDay | None, boundary_day: str) -> str:
    if today.expiry <= boundary_day:
        return "expired"
    if other is None or other.qc != today.qc:
        return "missing"
    return "ok"


def product_facts(today: ProductDay, d1: ProductDay | None, d1_day: str | None,
                  d2: ProductDay | None, d2_day: str | None, step: int = SAMPLE_STEP) -> dict[str, np.ndarray]:
    buy = np.where(today.eligible & np.isfinite(today.buy), today.buy, np.inf)
    buy[MAKER_WITHDRAW_SECOND + 1:] = np.inf
    suffix_min = np.minimum.accumulate(buy[::-1])[::-1]
    t0 = np.arange(QUOTE_START_SECOND, QUOTE_END_SECOND, step)
    valid = today.eligible[t0] & np.isfinite(today.mid[t0]) & np.isfinite(today.anchor[t0])
    t0 = t0[valid]
    anchor = today.anchor[t0]
    n = len(t0)
    d1_state = _state(today, d1, today.day)
    d2_state = "expired" if d1_day is None or today.expiry <= d1_day else _state(today, d2, d1_day)
    m1 = _window_min(d1, QUOTE_START_SECOND, MAKER_WITHDRAW_SECOND) if d1_state == "ok" else np.inf
    m2 = _window_min(d2, QUOTE_START_SECOND, MAKER_WITHDRAW_SECOND) if d2_state == "ok" else np.inf

    def rel(values):
        out = values - anchor
        return np.where(np.isfinite(out), out, np.nan)

    return {
        "day": np.full(n, today.day), "ValueCode": np.full(n, today.vc), "QuoteCode": np.full(n, today.qc),
        "expiry": np.full(n, today.expiry), "t0": t0.astype(np.int32),
        "resid_bp": today.mid[t0] - anchor, "anchor_bp": anchor,
        "min_today_bp": rel(suffix_min[t0 + 1]), "min_d1_bp": rel(np.full(n, m1)), "min_d2_bp": rel(np.full(n, m2)),
        "d1_day": np.full(n, d1_day if d1_state == "ok" else None, dtype=object),
        "d2_day": np.full(n, d2_day if d2_state == "ok" else None, dtype=object),
        "d1_state": np.full(n, d1_state), "d2_state": np.full(n, d2_state),
        "k_days": np.full(n, calendar_days(today.day, today.expiry), dtype=np.int32),
    }


def daily_facts(today: dict[str, ProductDay], d1: dict[str, ProductDay] | None, d1_day: str | None,
                d2: dict[str, ProductDay] | None, d2_day: str | None) -> pl.DataFrame:
    parts = []
    for vc, p in today.items():
        cols = product_facts(p, (d1 or {}).get(vc), d1_day, (d2 or {}).get(vc), d2_day)
        if len(cols["t0"]):
            parts.append(pl.DataFrame({k: list(v) if v.dtype == object else v for k, v in cols.items()},
                                      schema=FACT_SCHEMA))
    return pl.concat(parts) if parts else pl.DataFrame(schema=FACT_SCHEMA)


class GridCache:
    def __init__(self, keep: int = 3):
        self.keep, self._store = keep, {}

    def get(self, day: str | None) -> dict[str, ProductDay] | None:
        if day is None:
            return None
        if day not in self._store:
            if len(self._store) >= self.keep:
                del self._store[min(self._store)]
            self._store[day] = load_day(day)
        return self._store[day]


def build_day(day: str, cache: GridCache, days: list[str] | None = None) -> pl.DataFrame:
    days = days or grid_days()
    i = days.index(day)
    d1_day = days[i + 1] if i + 1 < len(days) else None
    d2_day = days[i + 2] if i + 2 < len(days) else None
    today = cache.get(day)
    facts = daily_facts(today, cache.get(d1_day), d1_day, cache.get(d2_day), d2_day)
    path = facts_path(day)
    path.parent.mkdir(parents=True, exist_ok=True)
    facts.write_parquet(path)
    qlevel.write_daily_hist(today, day)
    return facts


@dataclass(frozen=True)
class Cell:
    n: int
    p: float
    se: float
    ci_lo: float
    ci_hi: float
    key: tuple
    level: int
    n_days: int = 0


LADDER_C0 = (("e_b", "x", "t_b", "k_b"), ("e_b", "x", "t_b"), ("e_b", "x"), ("x",))
LADDER_C1 = (("e_b", "x", "k_b"), ("e_b", "x"), ("x",))
LADDER_Q = (("e_b", "x", "k_b"), ("x", "k_b"), ("x",))


@dataclass
class ReachTable:
    """Cells are keyed by (key names, key values) so that coarser keys never collide."""
    decision_day: str
    days: list[str]
    x_grid: tuple
    coord: str
    e_edges: tuple
    min_n: int
    min_days: int = 5   # sessions are the independent units; a cell needs at least this many
    k_edges: tuple = K_EDGES
    c0: dict = field(default_factory=dict)
    c1: dict = field(default_factory=dict)
    q: dict = field(default_factory=dict)

    def _ladder(self, table: dict, ladder: tuple, values: dict) -> Cell | None:
        for level, names in enumerate(ladder):
            key = (names, tuple(values[n] for n in names))
            cell = table.get(key)
            if cell is not None and cell.n >= self.min_n and cell.n_days >= self.min_days:
                return Cell(cell.n, cell.p, cell.se, cell.ci_lo, cell.ci_hi, key, level, cell.n_days)
        return None

    def e_bucket(self, e: float) -> int:
        return int(bucket([e], self.e_edges)[0])

    def lookup_c0(self, e: float, x: float, t_sec: int, k_days: int) -> Cell | None:
        values = dict(e_b=self.e_bucket(e), x=x, t_b=int(bucket([t_sec], T_EDGES)[0]),
                      k_b=int(k_bucket([k_days], self.k_edges)[0]))
        return self._ladder(self.c0, LADDER_C0, values)

    def lookup_c1(self, e: float, x: float, k_days: int) -> Cell | None:
        values = dict(e_b=self.e_bucket(e), x=x, k_b=int(k_bucket([k_days], self.k_edges)[0]))
        return self._ladder(self.c1, LADDER_C1, values)

    def lookup_q(self, x: float, k_days: int, e: float | None = None) -> Cell | None:
        values = dict(e_b=self.e_bucket(e) if e is not None else None, x=x,
                      k_b=int(k_bucket([k_days], self.k_edges)[0]))
        ladder = LADDER_Q if e is not None else LADDER_Q[1:]
        return self._ladder(self.q, ladder, values)

    def frames(self) -> dict[str, pl.DataFrame]:
        def frame(table: dict, ladder: tuple) -> pl.DataFrame:
            columns = list(ladder[0])
            rows = []
            for (names, values), c in table.items():
                row = {name: None for name in columns}
                row.update(dict(zip(names, values)))
                row.update(n=c.n, n_days=c.n_days, p=c.p, se=c.se, ci_lo=c.ci_lo, ci_hi=c.ci_hi,
                           level=ladder.index(names))
                rows.append(row)
            return pl.from_dicts(rows, infer_schema_length=None) if rows else pl.DataFrame()
        return {"c0": frame(self.c0, LADDER_C0), "c1": frame(self.c1, LADDER_C1), "q": frame(self.q, LADDER_Q)}


def _cells(frame: pl.DataFrame, ladder: tuple, flag: str, boot: int, rng: np.random.Generator) -> dict:
    """One cell per (key names, key values) for every ladder level, with day block bootstrap."""
    out = {}
    for level, names in enumerate(ladder):
        daily = frame.group_by(list(names) + ["day"]).agg(pl.len().alias("n"), pl.col(flag).sum().alias("s"))
        for g in daily.partition_by(list(names), maintain_order=True):
            n_d, s_d = g["n"].to_numpy(), g["s"].to_numpy()
            n, s = int(n_d.sum()), int(s_d.sum())
            p = s / n
            se = float(np.sqrt(p * (1 - p) / n))
            if len(n_d) > 1:
                idx = rng.integers(0, len(n_d), size=(boot, len(n_d)))
                ps = s_d[idx].sum(1) / n_d[idx].sum(1)
                lo, hi = (float(v) for v in np.percentile(ps, [2.5, 97.5]))
            else:
                lo, hi = p, p
            key = (names, tuple(g.item(0, c) for c in names))
            out[key] = Cell(n, p, se, lo, hi, key, level, len(n_d))
    return out


def load_facts(days: list[str]) -> pl.DataFrame:
    frames = [pl.read_parquet(facts_path(d)) for d in days if facts_path(d).exists()]
    return pl.concat(frames) if frames else pl.DataFrame(schema=FACT_SCHEMA)


def fit_from_facts(facts: pl.DataFrame, scales: pl.DataFrame, decision_day: str | None, *,
                   x_grid=None, coord: str = "residual_scaled", e_edges=None, k_edges=K_EDGES,
                   min_n: int = 100, min_days: int = 5, boot: int = 200, seed: int = 0) -> ReachTable:
    """decision_day=None pools every supplied session (structure study, not causal);
    otherwise only labels available before the decision day enter the table."""
    if coord not in COORDS:
        raise ValueError(f"unknown coord {coord}")
    x_grid = COORDS[coord]["x_grid"] if x_grid is None else x_grid
    e_edges = COORDS[coord]["e_edges"] if e_edges is None else e_edges
    if decision_day is not None and facts.height and facts["day"].max() >= decision_day:
        raise AssertionError("reach facts include the decision day or later")
    f = facts.join(scales, on=["day", "ValueCode"], how="left").filter(pl.col("scale").is_not_null())
    if coord == "residual_scaled":
        f = f.with_columns((pl.col("resid_bp") / pl.col("scale")).alias("e"), pl.col("scale").alias("unit"))
    else:
        f = f.with_columns(pl.col("resid_bp").alias("e"), pl.lit(1.0).alias("unit"))
    if decision_day is None:
        d1_ok = pl.col("d1_state") == "ok"
        d2_ok = pl.col("d2_state") == "ok"
    else:
        d1_ok = (pl.col("d1_state") == "ok") & (pl.col("d1_day") < decision_day).fill_null(False)
        d2_ok = (pl.col("d2_state") == "ok") & (pl.col("d2_day") < decision_day).fill_null(False)
    f = f.with_columns(
        pl.Series("e_b", bucket(f["e"].to_numpy(), e_edges).astype(np.int32)),
        pl.Series("t_b", bucket(f["t0"].to_numpy(), T_EDGES).astype(np.int32)),
        pl.Series("k_b", k_bucket(f["k_days"].to_numpy(), k_edges).astype(np.int32)),
        d1_ok.alias("d1_ok"), d2_ok.alias("d2_ok"),
    )
    rng = np.random.default_rng(seed)
    table = ReachTable(decision_day or "full", sorted(f["day"].unique().to_list()), tuple(x_grid), coord,
                       tuple(e_edges), min_n, min_days, tuple(k_edges))
    for x in x_grid:
        thr = pl.col("unit") * x
        g = f.with_columns(
            (pl.col("min_today_bp").fill_null(np.inf) <= thr).alias("r0"),
        ).with_columns(
            (pl.col("r0") | (pl.col("min_d1_bp").fill_null(np.inf) <= thr)).alias("r1"),
            (pl.col("min_d2_bp").fill_null(np.inf) <= thr).alias("r2"),
            pl.lit(x).alias("x"),
        )
        table.c0.update(_cells(g, LADDER_C0, "r0", boot, rng))
        table.c1.update(_cells(g.filter(pl.col("d1_ok")), LADDER_C1, "r1", boot, rng))
        table.q.update(_cells(g.filter(pl.col("d1_ok") & ~pl.col("r1") & pl.col("d2_ok")), LADDER_Q, "r2", boot, rng))
    return table


def fit(decision_day: str | None, window: int = 20, **kwargs) -> ReachTable:
    """decision_day=None: every cached session (structure study). Scales stay as-of each fact day."""
    cached = [d for d in grid_days() if facts_path(d).exists()]
    days = cached if decision_day is None else [d for d in cached if d < decision_day][-window:]
    facts = load_facts(days)
    scales = qlevel.scales_by_day(days, window)
    return fit_from_facts(facts, scales, decision_day, **kwargs)
