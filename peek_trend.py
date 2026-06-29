"""走勢分析：出場現貨腳(exit_spot)賣現那格(bid1)從錨點到『不可成交那刻』怎麼走。

「變化 N 次」分不出 上上上/下下下/上下震盪——本支把每個錨點的走勢分類：
  區間 = [錨點, 錨點價被成交(FillPrice==錨點價)那刻]；一秒內沒被成交→看到窗口底。
  走勢看「被成交之前 bid1 報價怎麼走」。
  方向 = 淨位移(終點 bid − 起點 bid) + 雙邊累積幅度(逐 tick 上行加總/下行加總)：
    往上(有利)：淨>0
    往下(不利)：淨<0
    震盪回歸  ：淨≈0 但雙邊幅度大（上下抵銷）
    平穩      ：淨≈0 且雙邊幅度小（幾乎沒動）
  以 tick 容差判 ≈0（用該價分級 tick）。

用法：python peek_trend.py -s 20260126 -e 20260625   （讀 out/peek/peek_*.csv）
"""
import sys
import io
import glob
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import polars as pl

PEEK_DIR = "out/peek"
ANCHOR = "exit_spot"
PXCOL = "BidPrice1"     # 賣現吃 bid


def tick_size(px: float) -> float:
    if px < 10:    return 0.01
    if px < 50:    return 0.05
    if px < 100:   return 0.1
    if px < 500:   return 0.5
    if px < 1000:  return 1.0
    return 5.0


def classify(date: int) -> pl.DataFrame:
    path = os.path.join(PEEK_DIR, f"peek_{date}.csv")
    if not os.path.exists(path):
        return pl.DataFrame()
    df = pl.read_csv(path, infer_schema_length=20000)
    df = df.with_columns(pl.col("RecvTime").str.to_datetime(strict=False))
    ex = df.filter(pl.col("anchor_type") == ANCHOR)
    rows = []
    for (eid, code, aseq), g in ex.group_by(["event_id", "anchor_code", "anchor_chseq"],
                                            maintain_order=True):
        g = g.sort("ChannelSeq")
        q = g.filter(pl.col("is_quote"))
        ar = g.with_row_index().filter(pl.col("ChannelSeq") == aseq)
        if ar.height == 0:
            continue
        i = ar["index"][0]
        anchor_t = g[i]["RecvTime"][0]
        base = g[i][PXCOL][0] if g[i]["is_quote"][0] else None
        if base is None:
            prev = g[:i].filter(pl.col("is_quote"))
            base = prev.tail(1)[PXCOL][0] if prev.height else None
        if base is None or base <= 0:
            continue
        # 終點(B)：錨點價被成交(FillPrice==base)那刻；一秒內沒成交→看到窗口底。
        #   走勢看的是「被成交之前 bid1 報價怎麼走」。
        gafter = g.filter(pl.col("ChannelSeq") > aseq).sort("ChannelSeq")
        if gafter.height == 0:
            continue
        fill_hit = gafter.filter(pl.col("is_fill") & (pl.col("FillPrice") == base))
        end_seq = fill_hit["ChannelSeq"][0] if fill_hit.height else None
        filled = end_seq is not None    # 一秒內有沒有在錨點價被成交
        after = gafter.filter(pl.col("is_quote"))
        if end_seq is not None:
            after = after.filter(pl.col("ChannelSeq") <= end_seq)   # 到被成交為止
        # 否則：整段窗口底(沒成交)
        px = after[PXCOL].to_list()
        seq = [base]
        for p in px:
            if p is not None and p > 0:
                seq.append(p)
        ts = tick_size(base)
        net = seq[-1] - seq[0]                                   # 淨位移
        up = sum(max(0, seq[k+1]-seq[k]) for k in range(len(seq)-1))   # 累積上行
        dn = sum(max(0, seq[k]-seq[k+1]) for k in range(len(seq)-1))   # 累積下行
        plots = g["potential_lots"][0]
        # 分類
        if net > ts * 0.5:
            kind = "往上(有利)"
        elif net < -ts * 0.5:
            kind = "往下(不利)"
        elif (up + dn) > ts * 0.5:
            kind = "震盪回歸"
        else:
            kind = "平穩"
        rows.append({"date": date, "event_id": eid, "kind": kind, "filled": filled,
                     "net_ticks": net / ts, "up_ticks": up / ts, "dn_ticks": dn / ts,
                     "potential_lots": plots})
    return pl.DataFrame(rows)


def main():
    import argparse
    from datetime import datetime, timedelta
    p = argparse.ArgumentParser()
    p.add_argument("-s", "--start_date", type=str, required=True)
    p.add_argument("-e", "--end_date", type=str)
    args = p.parse_args()
    start = datetime.strptime(args.start_date, "%Y%m%d")
    end = datetime.strptime(args.end_date, "%Y%m%d") if args.end_date else start
    frames = []
    cal = start
    while cal <= end:
        d = classify(int(cal.strftime("%Y%m%d")))
        if d.height:
            frames.append(d)
        cal += timedelta(days=1)
    if not frames:
        print("無資料"); return
    R = pl.concat(frames, how="diagonal_relaxed")   # 跨天 potential_lots 推型不一(Int/Float)→放寬
    n = R.height
    tot_lots = R["potential_lots"].sum()
    print(f"== exit_spot 賣現那格(bid1) 走勢分析  錨點 {n:,} ==\n")

    # ① 一秒內有沒有在錨點價被成交（出不出得掉）
    fl = R.filter(pl.col("filled")); nf = R.filter(~pl.col("filled"))
    print("【一秒內 錨點價有沒有被成交】")
    print(f"  有成交: {fl.height:>7,} ({fl.height/n*100:.1f}%)  出場量 {fl['potential_lots'].sum()/tot_lots*100:.1f}%")
    print(f"  未成交: {nf.height:>7,} ({nf.height/n*100:.1f}%)  出場量 {nf['potential_lots'].sum()/tot_lots*100:.1f}%  ← 想出沒出掉、被迫留倉")

    # ② 走勢型態（全體）
    print("\n【走勢型態（全體）】")
    print(f"  {'型態':<12}{'錨點%':>8}{'出場量%':>9}")
    for k in ["往上(有利)", "平穩", "震盪回歸", "往下(不利)"]:
        s = R.filter(pl.col("kind") == k)
        print(f"  {k:<12}{s.height/n*100:>7.1f}%{s['potential_lots'].sum()/tot_lots*100:>8.1f}%")

    # ③ 走勢 × 成交與否（未成交那批走勢往哪＝極端風險圖像）
    print("\n【走勢型態 × 成交與否（出場量%）】")
    print(f"  {'型態':<12}{'有成交量%':>10}{'未成交量%':>11}")
    for k in ["往上(有利)", "平穩", "震盪回歸", "往下(不利)"]:
        a = fl.filter(pl.col("kind")==k)["potential_lots"].sum()/tot_lots*100
        b = nf.filter(pl.col("kind")==k)["potential_lots"].sum()/tot_lots*100
        print(f"  {k:<12}{a:>9.1f}%{b:>10.1f}%")
    print(f"\n淨位移(ticks): 全體中位 {R['net_ticks'].median():.1f} | 未成交那批中位 {nf['net_ticks'].median():.1f}")


if __name__ == "__main__":
    main()
