"""Measure original XC's quote priority against raw as-of futures books."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import polars as pl

from ...common.paths import futures_raw_path
from ..causal_lookup import CLOSE_SECOND, SECOND, open_ns
from ..market import BookSeries, _raw
from .cost_tables import tick


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--days", nargs="+", default=["20260511","20260706","20260724"])
    args = parser.parse_args()
    counts = defaultdict(lambda:defaultdict(int))
    examples = []
    for day in args.days:
        samples = []
        for root in sorted(args.data.glob("ev_lookup_xc_*20260909")):
            manifest = json.loads((root/"manifest.json").read_text())
            for actor, _ in manifest["portfolios"]:
                folder = root/f"Date={day}"/actor
                path = folder/"decisions.parquet"
                if not path.exists():
                    continue
                fills = pl.scan_parquet(folder/"positions.parquet").filter(
                    (pl.col("entry_day") == day) & pl.col("entry_fill_ns").is_not_null())
                ids = set(fills.select("id").collect()["id"])
                rows = (pl.scan_parquet(path).filter(pl.col("admit") & (pl.col("stream") == "S2"))
                        .select("intent_id","qc","ns","quote_price").collect().to_dicts())
                for row in rows:
                    samples.append(dict(row,mode=manifest["mode"],actor=actor,filled=row["intent_id"] in ids))
        raw = _raw(futures_raw_path(day), sorted({r["qc"] for r in samples}),future=True,
                   start=open_ns(day),end=open_ns(day)+CLOSE_SECOND*SECOND)
        series = {g.item(0,"QuoteCode"):BookSeries("F:"+g.item(0,"QuoteCode"),g)
                  for g in raw.partition_by("QuoteCode",maintain_order=True)}
        for row in samples:
            book = series[row["qc"]].at(row["ns"])
            if not book or not book.valid() or book.ns > row["ns"]:
                raise AssertionError("cannot reconstruct an admitted XC quote")
            price = round(row["quote_price"]*10_000)
            ask = book.asks[0][0]
            inside = price == ask-tick(ask-1) and book.bids[0][0] < price < ask
            x = counts[day,row["mode"],row["actor"]]
            x["admitted_s2_quotes"] += 1
            x["noninside_quotes"] += int(not inside)
            x["filled_s2_quotes"] += int(row["filled"])
            x["noninside_filled_s2"] += int(row["filled"] and not inside)
            if not inside and row["filled"] and len(examples) < 30:
                examples.append(dict(row,day=day,raw_bid=book.bids[0][0]/10_000,raw_ask=ask/10_000,
                                     raw_book_ns=book.ns))
        print(json.dumps(dict(day=day,quotes=len(samples))),flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    rows = [dict(day=k[0],mode=k[1],actor=k[2],**v) for k,v in counts.items()]
    pl.from_dicts(rows).write_csv(args.output/"xc_raw_priority.csv")
    (args.output/"xc_raw_priority_examples.json").write_text(json.dumps(examples,indent=2)+"\n")
    print(json.dumps(rows,indent=2))


if __name__ == "__main__":
    main()
