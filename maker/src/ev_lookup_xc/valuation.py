"""End-of-session marks for reporting only; never consumed by the entry policy."""
from __future__ import annotations

from functools import lru_cache

import polars as pl

from ..common.paths import market_data_path, parse_date
from .market import METADATA_ROOT


@lru_cache(maxsize=2)
def official_prices(day: str) -> tuple[dict, dict]:
    futures = METADATA_ROOT / "official_future_daily_marks.parquet"
    if not futures.exists() or not market_data_path(day).exists():
        return {}, {}
    spot = pl.read_parquet(market_data_path(day), columns=["quote_code", "close_price"])
    future = (pl.scan_parquet(futures).filter(pl.col("date") == parse_date(day).date())
              .select("quote_code", "settlement_price").collect())
    return ({r[0]: round(r[1]*10_000) for r in spot.iter_rows() if r[1] is not None and r[1] > 0},
            {r[0]: round(r[1]*10_000) for r in future.iter_rows() if r[1] is not None and r[1] > 0})


def mark_positions(actor) -> dict:
    marked, missing, stale, rows = 0.0, 0, 0, []
    official_marked, official_missing = 0.0, 0
    spot_marks, future_marks = official_prices(actor.day)
    for pid in sorted(actor.active):
        p = actor.positions[pid]
        sq = p.spot_buy_qty - p.spot_sell_qty
        fq = p.future_sell_qty - p.future_buy_qty
        sb = actor.market.book("S:" + p.contract.vc, actor.market.end)
        fb = actor.market.book("F:" + p.contract.qc, actor.market.end)
        valid = ((not sq or sb and sb.formal and sb.bids) and
                 (not fq or fb and fb.formal and fb.asks))
        if getattr(p, "continuity_blocked", None):
            valid = False
        value = None
        age = None
        if valid:
            gross = (p.spot_sell_cash - p.spot_buy_cash + p.future_sell_cash - p.future_buy_cash
                     + (sq * sb.bids[0][0] if sq else 0)
                     - (fq * p.contract.shares * fb.asks[0][0] if fq else 0)) / 10_000
            fee_bp = 34  # Remaining session-end inventory incurs overnight round-trip cost.
            value = gross - p.spot_buy_cash / 1e8 * fee_bp
            marked += value
            times = ([sb.ns] if sq else []) + ([fb.ns] if fq else [])
            age = (actor.market.end - min(times)) / 1e9 if times else 0.0
            stale += int(age > 60)
        else:
            missing += 1
        sp, fp = spot_marks.get(p.contract.vc), future_marks.get(p.contract.qc)
        official = None
        if (not sq or sp) and (not fq or fp) and not getattr(p, "continuity_blocked", None):
            official = (p.spot_sell_cash-p.spot_buy_cash+p.future_sell_cash-p.future_buy_cash
                        + sq*(sp or 0)-fq*p.contract.shares*(fp or 0))/10_000-p.spot_buy_cash/1e8*34
            official_marked += official
        else:
            official_missing += 1
        rows.append(dict(position_id=pid, day=actor.day, mark_twd=value,
                         official_mark_twd=official,
                         book_age_seconds=age, nominal_twd=actor.ledger.amounts[pid] / 100,
                         continuity_blocked=getattr(p, "continuity_blocked", None)))
    actor.marks = rows
    return dict(marked_open_twd=marked, unmarked_positions=missing, stale_marks=stale,
                equity_twd=(sum(r["realized_twd"] for r in actor.daily) + marked)
                if missing == 0 else None,
                official_marked_open_twd=official_marked, official_unmarked_positions=official_missing,
                official_equity_twd=(sum(r["realized_twd"] for r in actor.daily)+official_marked)
                if official_missing == 0 else None)
