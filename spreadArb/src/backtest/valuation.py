"""Official daily valuation after replay; never feeds execution or admission.

Preserves original replay daily/mark files. Writes *_daily_official.csv and
marks_official.parquet for performance and risk reporting.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

from ..common.books import raw_paths
from ..common.paths import DATA_ROOT


def fetch_marks(first, last):
    folder = DATA_ROOT/"backtest"/"metadata"
    folder.mkdir(parents=True, exist_ok=True)
    for cached in sorted(folder.glob("official_future_daily_marks_*_complete.parquet")):
        parts = cached.stem.split("_")
        if parts[-3] <= first <= last <= parts[-2]:
            return cached
    path = folder/f"official_future_daily_marks_{first}_{last}.parquet"
    if not path.exists():
        from taker.data_paths import _mysql_query
        sql = """SELECT date, quote_code, settlement_price
                 FROM MarketInfo.taifex_futures_trades_daily
                 WHERE date BETWEEN :first AND :last AND trading_session = 'day'
                   AND char_length(quote_code) = 5"""
        f = _mysql_query(sql, {"first":int(first),"last":int(last)})
        f = f.with_columns(pl.col("date").cast(pl.Date),pl.col("settlement_price").cast(pl.Float64)).sort("date","quote_code")
        if f.select(pl.struct("date","quote_code").is_duplicated().sum()).item():
            raise ValueError("official marks contain duplicate contract-day rows")
        f.write_parquet(path)
        path.with_suffix(".json").write_text(json.dumps(dict(source="MarketInfo.taifex_futures_trades_daily",
            query=sql, parameters={"first":first,"last":last}, rows=f.height,
            fetched_at=datetime.now(timezone.utc).isoformat(), sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2))
    from .official_marks import complete
    return complete(path,first,last)


def revalue(root, *, partial=False):
    manifest = json.loads((root/"manifest.json").read_text())
    days = sorted(p.parent.name[5:] for p in root.glob("Date=*/complete.json"))
    if not partial and days != manifest["days"]:
        raise ValueError("replay must complete before final official valuation")
    path = fetch_marks(manifest["days"][0],manifest["days"][-1])
    future = pl.read_parquet(path).with_columns(pl.col("date").dt.strftime("%Y%m%d").alias("day"))
    prices = {(r["day"],r["quote_code"]):round(r["settlement_price"]*10000) for r in future.iter_rows(named=True)
              if r["settlement_price"] is not None and r["settlement_price"] > 0}
    results = {}
    for name,cfg in manifest["configs"].items():
        daily = pl.read_csv(root/f"{name}_daily.csv",schema_overrides={"day":pl.String})
        selected = daily.filter(pl.col("day").is_in(days))
        realized, previous = 0.0, 0.0
        rows, missing = [], []
        for dr in selected.iter_rows(named=True):
            day = dr["day"]
            folder = root/f"Date={day}"/name
            pp = folder/"positions.parquet"
            pos = pl.read_parquet(pp) if pp.exists() else pl.DataFrame()
            _,_,md = raw_paths(day)
            spots = {vc:round(px*10000) for vc,px in pl.read_parquet(md,columns=["quote_code","close_price"]).iter_rows()
                     if px is not None and px>0}
            marks = []
            for p in pos.iter_rows(named=True):
                if p["state"] == "closed":
                    continue
                sq, fq = p["spot_buy_qty"]-p["spot_sell_qty"],p["future_sell_qty"]-p["future_buy_qty"]
                sp, fp = spots.get(p["vc"]),prices.get((day,p["qc"]))
                fee = p["entry_spot_cash"]/1e8*cfg["cost"]["fee_overnight_bp"]+p["extra_fees"]
                gross = (p["spot_sell_cash"]-p["spot_buy_cash"]+p["future_sell_cash"]-p["future_buy_cash"])/10000
                value = None
                if (not sq or sp is not None) and (not fq or fp is not None):
                    value = gross+(sq*(sp or 0)-fq*p["shares"]*(fp or 0))/10000-fee
                else:
                    missing.append(dict(day=day,id=p["id"],qc=p["qc"],spot=sp,future=fp))
                marks.append(dict(id=p["id"],day=day,spot_qty=sq,future_qty=fq,value_twd=value,fee_estimate=fee,
                                  spot_mark=sp,future_mark=fp,spot_source="official_spot_close" if sp else None,
                                  future_source="official_future_daily" if fp else None,state=p["state"]))
            if marks:
                pl.from_dicts(marks,infer_schema_length=None).write_parquet(folder/"marks_official.parquet")
            realized += dr["realized_pnl"]
            unavailable = sum(r["value_twd"] is None for r in marks)
            equity = None if unavailable else realized+sum(r["value_twd"] for r in marks)
            dr["equity_twd"] = equity
            dr["mtm_pnl"] = equity-previous if equity is not None and previous is not None else None
            dr["missing_marks"] = unavailable
            rows.append(dr)
            previous = equity
        if rows:
            pl.from_dicts(rows,infer_schema_length=None).write_csv(root/f"{name}_daily_official.csv")
        eq = np.array([r["equity_twd"] if r["equity_twd"] is not None else np.nan for r in rows])
        values = np.r_[0.,eq]
        results[name] = dict(days=len(rows),equity_twd=rows[-1]["equity_twd"] if rows else None,
                            mtm_drawdown_twd=None if np.isnan(values).any() else float(np.min(values-np.maximum.accumulate(values))),
                            missing=missing)
    result = dict(source_path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),portfolios=results,
                  role="reporting only; original replay files and executions unchanged")
    (root/"official_valuation.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run",default="corrected_20260922_v2")
    ap.add_argument("--partial",action="store_true")
    args=ap.parse_args()
    print(json.dumps(revalue(DATA_ROOT/"backtest"/args.run,partial=args.partial),indent=2,allow_nan=False))


if __name__ == "__main__":
    main()
