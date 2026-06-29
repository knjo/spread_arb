"""出場滑價 因子驗證(exit_spot 賣現)：哪些『錨點前可見』的條件能降低滑價機率。

滑價(受評，事後)：被成交前 bid 往不利(下跌，含1tick)走 → slipped=True。
因子(事前，只用錨點那筆及之前，不偷看未來)：
  F1 厚度：錨點那筆 bid 五檔總厚度 ≥ 門檻
  F2 成交密度：錨點前 1 秒內成交筆數 / 是否有 <Nms 的密集成交(高頻在打→避開)
  F3 前兩檔：bid_lots1≥10 且 bid_lots2≥10 且 A1A2 連續(差=1tick)
做法：比「符合因子 vs 不符合」兩組的 slipped 率 + 出場量佔比，差越大＝因子越有效。

用法：python peek_factor.py -s 20260126 -e 20260625
"""
import sys
import io
import glob
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import polars as pl

PEEK_DIR = "out/peek"


def tick(px):
    return 0.01 if px < 10 else 0.05 if px < 50 else 0.1 if px < 100 else 0.5 if px < 500 else 1.0 if px < 1000 else 5.0


def per_day(date):
    path = os.path.join(PEEK_DIR, f"peek_{date}.csv")
    if not os.path.exists(path):
        return []
    df = pl.read_csv(path, infer_schema_length=20000).with_columns(
        pl.col("RecvTime").str.to_datetime(strict=False))
    ex = df.filter(pl.col("anchor_type") == "exit_spot")
    out = []
    for (eid, code, aseq), g in ex.group_by(["event_id", "anchor_code", "anchor_chseq"],
                                            maintain_order=True):
        g = g.sort("ChannelSeq")
        ar = g.with_row_index().filter(pl.col("ChannelSeq") == aseq)
        if ar.height == 0:
            continue
        i = ar["index"][0]
        at = g[i]["RecvTime"][0]
        # 錨點那筆 book（成交筆取前一報價）
        book = g[i] if g[i]["is_quote"][0] else (
            g[:i].filter(pl.col("is_quote")).tail(1) if g[:i].filter(pl.col("is_quote")).height else None)
        if book is None or book.height == 0:
            continue
        base = book["BidPrice1"][0]
        if base is None or base <= 0:
            continue
        ts = tick(base)
        # ── 事前因子(錨點那筆及之前) ──
        depth5 = sum(x for x in [book[f"BidLots{k}"][0] for k in range(1, 6)] if x)
        l1, l2 = book["BidLots1"][0] or 0, book["BidLots2"][0] or 0
        p1, p2 = book["BidPrice1"][0], book["BidPrice2"][0]
        cont = (p1 and p2 and abs(p1 - p2) <= ts * 1.5)   # A1A2 連續(差≤1tick)
        f3 = (l1 >= 10 and l2 >= 10 and cont)
        # 錨點前 1 秒成交筆數
        before = g.filter((pl.col("ChannelSeq") < aseq) & pl.col("is_fill")
                          & (pl.col("RecvTime") >= at - pl.duration(seconds=1)))
        n_fill_1s = before.height
        # ── 事後結果：被成交前 bid 往不利(下跌) ──
        gafter = g.filter(pl.col("ChannelSeq") > aseq).sort("ChannelSeq")
        fh = gafter.filter(pl.col("is_fill") & (pl.col("FillPrice") == base))
        end = fh["ChannelSeq"][0] if fh.height else None
        af = gafter.filter(pl.col("is_quote"))
        if end is not None:
            af = af.filter(pl.col("ChannelSeq") <= end)
        px = [p for p in af["BidPrice1"].to_list() if p and p > 0]
        net = (px[-1] - base) / ts if px else 0.0
        slipped = net <= -1.0     # 往不利 ≥1 tick(含)
        out.append((depth5, l1, l2, f3, n_fill_1s, slipped, g["potential_lots"][0]))
    return out


def main():
    import argparse
    from datetime import datetime, timedelta
    p = argparse.ArgumentParser()
    p.add_argument("-s", "--start_date", required=True)
    p.add_argument("-e", "--end_date")
    args = p.parse_args()
    start = datetime.strptime(args.start_date, "%Y%m%d")
    end = datetime.strptime(args.end_date, "%Y%m%d") if args.end_date else start
    rows = []
    cal = start
    while cal <= end:
        rows += per_day(int(cal.strftime("%Y%m%d")))
        cal += timedelta(days=1)
    R = pl.DataFrame(rows, schema=["depth5", "l1", "l2", "f3", "n_fill_1s", "slipped", "lots"],
                     orient="row")
    n = R.height
    base_slip = R["slipped"].mean() * 100
    print(f"== exit_spot 滑價因子驗證  錨點 {n:,}  基準滑價率(往不利≥1tick) {base_slip:.1f}% ==\n")

    tot_lots = R["lots"].sum()

    def cmp(name, mask):
        a = R.filter(mask); b = R.filter(~mask)
        if a.height == 0 or b.height == 0:
            print(f"  {name}: 一組為空"); return
        # 符合組保住多少量、滑價率；不符合組(=被過濾掉的)砍掉多少量
        print(f"  {name}")
        print(f"    符合(做): {a.height:,}筆 {a.height/n*100:.0f}% | 量佔 {a['lots'].sum()/tot_lots*100:.0f}% | 滑價率 {a['slipped'].mean()*100:.1f}%")
        print(f"    不符合(濾掉): {b.height:,}筆 | 量佔 {b['lots'].sum()/tot_lots*100:.0f}% | 滑價率 {b['slipped'].mean()*100:.1f}%")

    print("【F1 厚度】錨點 bid 五檔總厚度")
    for thr in [50, 100, 200]:
        cmp(f"  五檔厚度 ≥{thr}", pl.col("depth5") >= thr)
    print("\n【F3 前兩檔】bid_lots1≥10 且 bid_lots2≥10 且 A1A2連續(差≤1tick)")
    cmp("  F3 成立", pl.col("f3"))
    # F2(成交密度/快市)需先定義快市口徑、可能重撈 tick，本次擱置。


if __name__ == "__main__":
    main()
