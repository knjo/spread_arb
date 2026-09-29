"""Level table: per-product residual quantiles and scale from prior sessions only.

residual = basis_mid_bp - anchor_ewma_120s_bp on analysis-eligible seconds.
Each session is stored as a 1 bp histogram so that pooled quantiles over any
trailing window are exact and cheap.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from ..common.grid import ProductDay
from ..common.paths import grid_days, hist_path

BIN_LO, BIN_HI, BIN_W = -300.0, 300.0, 1.0
NBINS = int(round((BIN_HI - BIN_LO) / BIN_W))
EDGES = np.linspace(BIN_LO, BIN_HI, NBINS + 1)
QUANTILES = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95)
HIST_SCHEMA = {"day": pl.String, "ValueCode": pl.String, "bin": pl.Int32, "count": pl.Int64}
MIN_SCALE_BP = 1.0
# A very wide residual distribution means stale/wide books, not a basis that
# routinely travels that far; such names are excluded from pooled tables.
MAX_SCALE_BP = 60.0


def qname(q: float) -> str:
    return f"q{int(round(q * 100)):02d}"


def daily_hist(products: dict[str, ProductDay]) -> pl.DataFrame:
    frames = []
    for vc, p in products.items():
        mask = p.eligible & np.isfinite(p.mid) & np.isfinite(p.anchor)
        if not mask.any():
            continue
        resid = np.clip(p.mid[mask] - p.anchor[mask], BIN_LO, BIN_HI - 1e-9)
        counts, _ = np.histogram(resid, bins=EDGES)
        nz = np.nonzero(counts)[0]
        frames.append(pl.DataFrame({"day": [p.day] * len(nz), "ValueCode": [vc] * len(nz),
                                    "bin": nz.astype(np.int32), "count": counts[nz].astype(np.int64)},
                                   schema=HIST_SCHEMA))
    return pl.concat(frames) if frames else pl.DataFrame(schema=HIST_SCHEMA)


def write_daily_hist(products: dict[str, ProductDay], day: str) -> pl.DataFrame:
    table = daily_hist(products)
    path = hist_path(day)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(path)
    return table


def load_hists(days: list[str]) -> pl.DataFrame:
    frames = [pl.read_parquet(hist_path(d)) for d in days if hist_path(d).exists()]
    return pl.concat(frames) if frames else pl.DataFrame(schema=HIST_SCHEMA)


def quantiles_from_hist(bins: np.ndarray, counts: np.ndarray, qs=QUANTILES) -> np.ndarray:
    order = np.argsort(bins)
    b, c = bins[order], counts[order]
    cum = np.cumsum(c)
    total = cum[-1]
    lo = BIN_LO + b * BIN_W
    out = []
    for q in qs:
        target = q * total
        i = min(int(np.searchsorted(cum, target, side="left")), len(b) - 1)
        prev = cum[i - 1] if i > 0 else 0
        frac = (target - prev) / c[i] if c[i] > 0 else 0.5
        out.append(lo[i] + float(np.clip(frac, 0.0, 1.0)) * BIN_W)
    return np.array(out)


def table_from_hists(hists: pl.DataFrame, decision_day: str, window: int = 20) -> pl.DataFrame:
    """Quantile table as of `decision_day`: only sessions strictly before it."""
    days = sorted(d for d in hists["day"].unique().to_list() if d < decision_day)[-window:]
    schema = {"ValueCode": pl.String, **{qname(q): pl.Float64 for q in QUANTILES},
              "scale": pl.Float64, "scale_raw": pl.Float64, "n_obs": pl.Int64, "n_days": pl.Int64, "as_of": pl.String}
    if not days:
        return pl.DataFrame(schema=schema)
    use = hists.filter(pl.col("day").is_in(days))
    if use["day"].max() >= decision_day:
        raise AssertionError("level table would read the decision day or later")
    n_days = use.group_by("ValueCode").agg(pl.col("day").n_unique().alias("n_days"))
    pooled = use.group_by("ValueCode", "bin").agg(pl.col("count").sum())
    rows = []
    for g in pooled.partition_by("ValueCode", maintain_order=True):
        vc = g.item(0, "ValueCode")
        qs = quantiles_from_hist(g["bin"].to_numpy(), g["count"].to_numpy())
        scale = qs[QUANTILES.index(0.95)] - qs[QUANTILES.index(0.50)]
        rows.append({"ValueCode": vc, **{qname(q): float(v) for q, v in zip(QUANTILES, qs)},
                     "scale": float(scale) if MIN_SCALE_BP <= scale <= MAX_SCALE_BP else None,
                     "scale_raw": float(scale),
                     "n_obs": int(g["count"].sum()), "as_of": decision_day})
    return (pl.DataFrame(rows).join(n_days, on="ValueCode", how="left")
            .select(list(schema)).cast(schema))


def table(decision_day: str, window: int = 20) -> pl.DataFrame:
    days = [d for d in grid_days() if d < decision_day][-window:]
    return table_from_hists(load_hists(days), decision_day, window)


def scales_by_day(days: list[str], window: int = 20, hists: pl.DataFrame | None = None) -> pl.DataFrame:
    """(day, ValueCode, scale) where each day's scale uses only sessions before it."""
    if not days:
        return pl.DataFrame(schema={"day": pl.String, "ValueCode": pl.String, "scale": pl.Float64})
    if hists is None:
        needed = [d for d in grid_days() if d < max(days)][-(window + len(days)):]
        hists = load_hists(needed)
    frames = [table_from_hists(hists, day, window).select(pl.lit(day).alias("day"), "ValueCode", "scale")
              for day in days]
    return pl.concat(frames) if frames else pl.DataFrame(schema={"day": pl.String, "ValueCode": pl.String, "scale": pl.Float64})
