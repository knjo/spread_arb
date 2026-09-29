"""Absolute-basis convergence table from the taker line's first-crossing samples.

Source: taker `convergence_days_by_settle_cycle.py` samples (SSD2 stockfuture/convergence_days/
convergence_samples_enriched_*.parquet). Each row is the first time a product-day's taker-executable
basis (`ret_sell` = FutBid/SpotA1 - 1) crossed a threshold (0.5/1/1.5/2%); convergence is the first later
`ret_buy >= 0` on the same contract, i.e. our `basis_buy_taker <= 0` (absolute exit level 0). Rows that
never converge before expiry settle (basis = 0 by accounting).

Table: (threshold, days-to-settlement bucket) -> P(first convergence on trading day j), P(settle),
n. Populations overlap across thresholds by construction, so a lookup picks one threshold (the highest
at or below the expected post-hedge basis) and never adds across thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from ..common.paths import grid_days

THRESHOLDS_BP = (50.0, 100.0, 150.0, 200.0)
K_EDGES_TD = (0, 2, 5, 10, 15)            # trading days to settlement: 0 / 1-2 / 3-5 / 6-10 / 11-15 / 16+
MAX_J = 15                                # explicit first-passage days; later natural convergence pooled into "later"
DEFAULT_FILE = "convergence_samples_enriched_20260126_20260625.parquet"


def source_path() -> Path:
    from ..common.books import _maker_paths
    return Path(_maker_paths().HFT_DATA_ROOT) / "stockfuture" / "convergence_days" / DEFAULT_FILE


def k_bucket_td(days: int) -> int:
    return int(np.searchsorted(np.asarray(K_EDGES_TD, dtype=float), float(days), side="left"))


@dataclass(frozen=True)
class AbsCell:
    n: int
    p_day: tuple          # P(first convergence on trading day j), j = 0..MAX_J
    p_later: float        # natural convergence after MAX_J trading days
    p_settle: float       # held to settlement
    mean_cal_days: float
    key: tuple
    level: int            # 0 = (threshold, k bucket), 1 = threshold only


@dataclass
class AbsTable:
    as_of: str | None
    cells: dict = field(default_factory=dict)
    min_n: int = 30

    @classmethod
    def fit(cls, samples: pl.DataFrame, as_of: str | None = None, min_n: int = 30) -> "AbsTable":
        s = samples
        if as_of is not None:
            # Causal and unbiased for the explicit first-passage days: only entries at least MAX_J + 1 sessions
            # before as_of (so "converged within j <= MAX_J" is fully observable for every kept sample).
            # A kept sample whose outcome is not yet known at as_of (still open after MAX_J sessions) is
            # counted as held to settlement, the conservative branch. Dropping unresolved samples instead
            # (an earlier version) kept only the fast convergers and overstated same-day convergence.
            sessions = [d for d in grid_days() if d < as_of]
            if len(sessions) <= MAX_J:
                return cls(as_of, {}, min_n)
            cutoff = sessions[-(MAX_J + 1)]
            s = s.filter(pl.col("Date") <= cutoff)
            known = pl.when(pl.col("natural_converge")).then(pl.col("converge_date")).otherwise(pl.col("settle_date")) < as_of
            fast = pl.col("natural_converge") & (pl.col("convergence_days_td") <= MAX_J)
            s = s.with_columns(
                pl.when(fast).then(True).when(known).then(pl.col("natural_converge")).otherwise(False).alias("natural_converge"),
                pl.when(fast).then(pl.col("convergence_days_td")).when(known).then(pl.col("convergence_days_td")).otherwise(None).alias("convergence_days_td"),
            )
        table = cls(as_of, {}, min_n)
        s = s.with_columns(
            (pl.col("threshold") * 10_000).round(0).alias("thr_bp"),
            pl.Series("k_b", [k_bucket_td(d) for d in s["days_to_settle_td"].to_list()]).cast(pl.Int32),
        )
        for level, keys in enumerate((["thr_bp", "k_b"], ["thr_bp"])):
            for g in s.partition_by(keys, maintain_order=True):
                key = tuple(g.item(0, c) for c in keys)
                n = g.height
                nat = g.filter(pl.col("natural_converge"))
                days = nat["convergence_days_td"].to_numpy()
                p_day = tuple(float((days == j).sum()) / n for j in range(MAX_J + 1))
                p_later = float((days > MAX_J).sum()) / n
                p_settle = 1.0 - float(nat.height) / n
                table.cells[(level, key)] = AbsCell(n, p_day, p_later, p_settle,
                                                    float(g["convergence_days_cal"].mean()), key, level)
        return table

    def lookup(self, ab_bp: float, days_to_settle_td: int) -> AbsCell | None:
        """Nearest threshold population (a lower one converges faster than this basis would, a higher one
        slower); no absolute route below the lowest threshold."""
        if ab_bp < THRESHOLDS_BP[0]:
            return None
        thr = min(THRESHOLDS_BP, key=lambda t: abs(t - ab_bp))
        for level, key in ((0, (thr, k_bucket_td(days_to_settle_td))), (1, (thr,))):
            cell = self.cells.get((level, key))
            if cell is not None and cell.n >= self.min_n:
                return cell
        return None

    def frame(self) -> pl.DataFrame:
        rows = []
        for (level, key), c in self.cells.items():
            rows.append(dict(level=level, thr_bp=key[0], k_b=key[1] if len(key) > 1 else None, n=c.n,
                             p0=c.p_day[0], p1=c.p_day[1], p2=c.p_day[2], p3_5=sum(c.p_day[3:6]),
                             p6_15=sum(c.p_day[6:]), p_later=c.p_later, p_settle=c.p_settle,
                             mean_cal_days=c.mean_cal_days))
        return pl.from_dicts(rows, infer_schema_length=None) if rows else pl.DataFrame()


def load_samples(path: Path | None = None) -> pl.DataFrame:
    path = path or source_path()
    return pl.read_parquet(path, columns=["Date", "ValueCode", "QuoteCode", "settle_date", "threshold", "ret_sell",
                                          "days_to_settle_td", "days_to_settle_cal", "natural_converge",
                                          "convergence_days_td", "convergence_days_cal", "converge_date", "status"])
