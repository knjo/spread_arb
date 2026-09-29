"""Refresh held contracts from current-day metadata without rolling inventory."""
from dataclasses import replace
from pathlib import Path

import polars as pl

from ..common.books import raw_paths
from ..common.paths import DATA_ROOT


def refresh_carry(day, carry):
    if not carry:
        return [], []
    _, _, spot_path = raw_paths(day)
    spot = pl.read_parquet(spot_path, columns=["quote_code", "opening_ref_price"])
    refs = {vc: round(px * 10000) for vc, px in spot.iter_rows() if px is not None and px > 0}
    path = Path("maker/data/fair_mid/metadata") / f"{day}_contracts.parquet"
    basic = pl.read_parquet(path) if path.exists() else pl.DataFrame()
    codes = {c.qc for c in carry}
    if not basic.height or not codes.issubset(set(basic["QuoteCode"])):
        path = DATA_ROOT / "backtest/metadata" / f"{day}_all_futures_contracts.parquet"
        if not path.exists():
            from maker.src.common.contracts import _load_futures_basic_from_existing_loader, _normalise_basic_schema
            all_basic = _normalise_basic_schema(_load_futures_basic_from_existing_loader(day))
            path.parent.mkdir(parents=True, exist_ok=True)
            all_basic.write_parquet(path)
        basic = pl.read_parquet(path)
    exact = {r["QuoteCode"]: r for r in basic.iter_rows(named=True)}
    updated = []
    for c in carry:
        r = exact.get(c.qc)
        if r is None or c.vc not in refs:
            raise ValueError(f"missing current-day held-contract metadata: {day} {c.vc} {c.qc}")
        if (r["ValueCode"] != c.vc or abs(r["contract_size"] - c.shares) > 1e-6
                or r["end_date"].strftime("%Y%m%d") != c.expiry):
            raise ValueError(f"unhandled held-contract adjustment: {day} {c}")
        if r["fut_ref_price"] is None or r["fut_ref_price"] <= 0:
            raise ValueError(f"missing held futures reference: {day} {c.qc}")
        updated.append(replace(c, spot_ref=refs[c.vc], fut_ref=round(r["fut_ref_price"] * 10000)))
    return updated, [str(path), str(spot_path)]
