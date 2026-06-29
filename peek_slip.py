"""出場賣現 滑價估計(獨立，不動 peek_trend/peek_factor)。

情境：出場賣現掛 taker，怕『成交前 B1 波動，更差的價跑到 B1』→ 吃到比錨點 B1 差的價。
算法（最保守、涵蓋沒成交情況）：
  基準 = 錨點那筆 B1（你想要的最好賣價）。
  區間 = 錨點 → +1 秒（整段，不管成交，故沒成交的最壞也算到）。
  滑價 = 該段 min(BidPrice1) 比錨點 B1 低幾 tick（只看跌；漲＝對你更有利，不算）。
       低幾 tick = 你掛 taker 一秒內最壞會吃到多差的價。
輸出：跌 0/1/2/3+ tick 各佔 錨點數% 與 出場量%。

用法：python peek_slip.py -s 20260126 -e 20260625
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
        base = g[i]["BidPrice1"][0] if g[i]["is_quote"][0] else (
            g[:i].filter(pl.col("is_quote")).tail(1)["BidPrice1"][0]
            if g[:i].filter(pl.col("is_quote")).height else None)
        if base is None or base <= 0:
            continue
        # 終點：有成交→第一筆成交「之前」(不含成交筆，避免成交把檔吃掉後的 B1 污染)；
        #       沒成交→看滿 1 秒。
        after = g.filter(pl.col("ChannelSeq") > aseq)
        fh = after.filter(pl.col("is_fill"))   # 第一筆任何成交
        if fh.height:
            end_seq = fh["ChannelSeq"][0]
            win = after.filter(pl.col("is_quote") & (pl.col("ChannelSeq") < end_seq))  # 成交之前
            filled = True
        else:
            win = after.filter(pl.col("is_quote")
                               & (pl.col("RecvTime") <= at + pl.duration(seconds=1)))  # 滿1秒
            filled = False
        b1 = [p for p in win["BidPrice1"].to_list() if p and p > 0]
        worst = min(min(b1), base) if b1 else base   # 沒中間報價→維持錨點價(沒滑)
        drop_ticks = round((base - worst) / tick(base))   # 跌幾 tick（>=0）
        out.append((drop_ticks, filled, g["potential_lots"][0]))
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
    R = pl.DataFrame(rows, schema=["drop_ticks", "filled", "lots"], orient="row")
    tot = R["lots"].sum()
    print(f"== 出場賣現 掛taker B1滑價(有成交→成交前；沒成交→滿1秒)  錨點 {R.height:,} ==")
    print(f"   (跌 0=沒滑；跌N=吃到比錨點B1差N檔)\n")

    def block(D, title):
        n = D.height
        if n == 0:
            print(f"【{title}】無\n"); return
        ltot = D["lots"].sum()
        print(f"【{title}】 錨點 {n:,} ({n/R.height*100:.0f}%) | 量佔全部 {ltot/tot*100:.0f}%")
        print(f"  {'跌tick':<8}{'錨點%':>8}{'量%':>8}")
        for t in [0, 1, 2, 3]:
            m = D.filter(pl.col("drop_ticks") == t) if t < 3 else D.filter(pl.col("drop_ticks") >= 3)
            lab = f"{t}" if t < 3 else "≥3"
            print(f"  {lab:<8}{m.height/n*100:>7.1f}%{m['lots'].sum()/ltot*100:>7.1f}%")
        print(f"  中位 {D['drop_ticks'].median():.0f} 平均 {D['drop_ticks'].mean():.2f} p90 {D['drop_ticks'].quantile(.9):.0f} | 有滑(≥1)量 {D.filter(pl.col('drop_ticks')>=1)['lots'].sum()/ltot*100:.0f}%\n")

    block(R.filter(pl.col("filled")), "有成交(看到成交前)")
    block(R.filter(~pl.col("filled")), "沒成交(看滿1秒)")
    block(R, "全體")


if __name__ == "__main__":
    main()
