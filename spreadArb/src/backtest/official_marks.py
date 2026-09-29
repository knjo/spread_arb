"""Fill absent database sessions from the exchange's public CSV download."""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
from io import StringIO
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import polars as pl

from ..common.paths import grid_days, mapping_path

URL = "https://www.taifex.com.tw/cht/3/futDataDown"
PAGE = "https://www.taifex.com.tw/cht/3/futDailyMarketView"


def download(day, folder):
    path = folder/f"taifex_{day}.csv"
    date = datetime.strptime(day,"%Y%m%d").strftime("%Y/%m/%d")
    params = dict(down_type="1",commodity_id="all",commodity_id2="",queryStartDate=date,queryEndDate=date)
    if not path.exists():
        request = Request(URL,data=urlencode(params).encode())
        with urlopen(request,timeout=45) as response:
            value = response.read()
        path.write_bytes(value)
    value = path.read_bytes()
    text = value.decode("cp950")
    rows = []
    for r in csv.DictReader(StringIO(text)):
        if r["交易日期"].strip() != date:
            raise ValueError("exchange returned a different date")
        if r["交易時段"].strip() != "一般":
            continue
        product, month, price = r["契約"].strip(),r["到期月份(週別)"].strip(),r["結算價"].strip()
        if len(product) != 3 or len(month) != 6 or not month.isdigit() or price in ("", "-"):
            continue
        code = product+"ABCDEFGHIJKL"[int(month[-2:])-1]+month[3]
        rows.append(dict(date=datetime.strptime(day,"%Y%m%d").date(),quote_code=code,settlement_price=float(price)))
    if not rows:
        raise ValueError(f"no official daytime settlements for {day}")
    path.with_suffix(".json").write_text(json.dumps(dict(source_page=PAGE,url=URL,parameters=params,
        sha256=hashlib.sha256(value).hexdigest(),normalized_rows=len(rows)),indent=2))
    return pl.from_dicts(rows)


def complete(path, first, last):
    destination = path.with_name(path.stem+"_complete.parquet")
    if destination.exists():
        return destination
    frame = pl.read_parquet(path)
    available = set(frame["date"].dt.strftime("%Y%m%d").to_list())
    missing = [d for d in grid_days() if first <= d <= last and d not in available]
    parts = [frame]+[download(day,path.parent) for day in missing]
    result = pl.concat(parts,how="vertical_relaxed").sort("date","quote_code")
    if result.select(pl.struct("date","quote_code").is_duplicated().sum()).item():
        raise ValueError("duplicate official contract-day prices")
    # Independent source overlap verifies product/month/year code normalization.
    control_day = "20260625" if "20260625" in available else min(available)
    control = download(control_day,path.parent)
    overlap = control.join(frame,on=["date","quote_code"],suffix="_db")
    # This replay trades standard stock futures. Currency settlements in the
    # project database are rounded to two decimals and are outside this check.
    products = pl.read_parquet(mapping_path(control_day))["QuoteCode"].str.slice(0,3).unique().to_list()
    overlap = overlap.filter(pl.col("quote_code").str.slice(0,3).is_in(products))
    mismatches = overlap.filter((pl.col("settlement_price")-pl.col("settlement_price_db")).abs()>1e-9)
    if overlap.height == 0 or mismatches.height:
        raise ValueError("public CSV and database control prices do not match")
    result.write_parquet(destination)
    destination.with_suffix(".json").write_text(json.dumps(dict(database_cache=str(path),public_csv_days=missing,
        rows=result.height,days=result["date"].n_unique(),control_day=control_day,control_rows=overlap.height,
        control_mismatches=mismatches.height,sha256=hashlib.sha256(destination.read_bytes()).hexdigest()),indent=2))
    return destination
