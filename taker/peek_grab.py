"""第4點：出場賣現掛 taker，搶輸時吃 N 張要多少滑價(獨立，不動別支)。

情境：你掛 taker 出場賣現。別人先成交、吃掉前幾檔；輪到你時，從『成交後的簿子』
往下吃你的 N 張，算加權均價 vs 錨點 B1 差多少 tick。
口徑：
  簿子 = 錨點後第一筆成交那筆的 Bid 五檔(成交後狀態)。
  N(張) = potential_lots × contract_size / 1000（口換張；小型 cs=100、標準 2000）。
  只掛到 B4 → 走簿最多吃 B1~B4；N 超過 B1~B4 總量 = 「B4 內吃不完」(標記，不硬塞 B5)。
  滑價 = (錨點 B1 − 加權均價) / tick = 差幾 tick(>0=賣得比想要的差)。
  只算有成交的錨點(沒成交=另一個問題，不在此)。

用法：uv run python peek_grab.py -s 20260126 -e 20260625
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


def per_day(date, only_standard=False):
    path = os.path.join(PEEK_DIR, f"peek_{date}.csv")
    if not os.path.exists(path):
        return []
    df = pl.read_csv(path, infer_schema_length=20000)
    ex = df.filter(pl.col("anchor_type") == "exit_spot")
    out = []
    for (eid, code, aseq), g in ex.group_by(["event_id", "anchor_code", "anchor_chseq"],
                                            maintain_order=True):
        g = g.sort("ChannelSeq")
        ar = g.with_row_index().filter(pl.col("ChannelSeq") == aseq)
        if ar.height == 0:
            continue
        i = ar["index"][0]
        base = g[i]["BidPrice1"][0] if g[i]["is_quote"][0] else (
            g[:i].filter(pl.col("is_quote")).tail(1)["BidPrice1"][0]
            if g[:i].filter(pl.col("is_quote")).height else None)
        if base is None or base <= 0:
            continue
        cs = g["contract_size"][0]
        plots = g["potential_lots"][0]
        if cs is None or plots is None or cs <= 0:
            continue
        if only_standard and cs != 2000:
            continue
        N = plots * cs / 1000.0   # 要賣的張數
        if N <= 0:
            continue
        # 第一筆成交那筆(成交後簿子)
        fh = g.filter((pl.col("ChannelSeq") > aseq) & pl.col("is_fill")).sort("ChannelSeq")
        if fh.height == 0:
            continue   # 沒成交不算
        bk = fh.head(1)
        # Bid 五檔(成交後)，只用 B1~B4
        pxs = [bk[f"BidPrice{k}"][0] for k in range(1, 5)]
        lts = [bk[f"BidLots{k}"][0] for k in range(1, 5)]
        # 走簿吃 N 張(B1→B4)，記 加權均價 + 最差吃到那檔的價
        remain = N
        cost = 0.0
        filled = 0.0
        worst_px = None
        for p, l in zip(pxs, lts):
            if p is None or p <= 0 or l is None or l <= 0:
                continue
            take = min(remain, l)
            cost += take * p
            filled += take
            worst_px = p          # 吃到的最深那檔價
            remain -= take
            if remain <= 0:
                break
        eat_all = remain <= 1e-9
        avg = cost / filled if filled > 0 else None
        ts = tick(base)
        slip = (base - avg) / ts if avg is not None else None        # 加權均價滑價(小數)
        worst = (base - worst_px) / ts if worst_px is not None else None  # 最差吃到第幾檔(整數)
        out.append((cs, N, eat_all, slip if slip is not None else 0.0,
                    worst if worst is not None else 0.0,
                    g["potential_lots"][0]))
    return out


def main():
    import argparse
    from datetime import datetime, timedelta
    p = argparse.ArgumentParser()
    p.add_argument("-s", "--start_date", required=True)
    p.add_argument("-e", "--end_date")
    p.add_argument("--std", action="store_true", help="只看標準股期(cs=2000)")
    args = p.parse_args()
    start = datetime.strptime(args.start_date, "%Y%m%d")
    end = datetime.strptime(args.end_date, "%Y%m%d") if args.end_date else start
    rows = []
    cal = start
    while cal <= end:
        rows += per_day(int(cal.strftime("%Y%m%d")), only_standard=args.std)
        cal += timedelta(days=1)
    R = pl.DataFrame(rows, schema={"cs": pl.Float64, "N": pl.Float64, "eat_all": pl.Boolean,
                                   "slip": pl.Float64, "worst": pl.Float64, "plots": pl.Int64},
                     orient="row")
    n = R.height
    tot = R["plots"].sum()
    print(f"== 第4點 搶輸吃量滑價(成交後簿子、吃到B4、賣現)  錨點 {n:,} ==")
    print(f"   小型佔 {R.filter(pl.col('cs')==100).height/n*100:.1f}%（cs=100）\n")

    eat = R.filter(pl.col("eat_all"))
    noeat = R.filter(~pl.col("eat_all"))
    print(f"【B1~B4 吃得完】 {eat.height:,} ({eat.height/n*100:.0f}%) | 量佔 {eat['plots'].sum()/tot*100:.0f}%")
    print(f"【B4 內吃不完(連B4都不夠)】 {noeat.height:,} ({noeat.height/n*100:.0f}%) | 量佔 {noeat['plots'].sum()/tot*100:.0f}%\n")

    el = eat["plots"].sum()
    print("[A] 加權均價滑價(你 N 張的真實平均成本 vs 錨點，小數tick):")
    s = eat["slip"]
    print(f"    中位 {s.median():.2f}  平均 {s.mean():.2f}  p90 {s.quantile(.9):.1f}  max {s.max():.1f}")
    print(f"    {'差tick':<10}{'錨點%':>8}{'量%':>8}")
    for lo, hi, lab in [(-1e9, 0.001, "0(沒滑)"), (0.001, 1.0, "0~1"), (1.0, 2.0, "1~2"), (2.0, 1e9, "≥2")]:
        m = eat.filter((pl.col("slip") > lo) & (pl.col("slip") <= hi)) if lab != "0(沒滑)" \
            else eat.filter(pl.col("slip") <= 0.001)
        if eat.height:
            print(f"    {lab:<10}{m.height/eat.height*100:>7.1f}%{m['plots'].sum()/el*100:>7.1f}%")

    print("\n[B] 最差吃到第幾檔(最壞那檔距錨點，整數tick):")
    w = eat["worst"]
    print(f"    中位 {w.median():.0f}  平均 {w.mean():.2f}  p90 {w.quantile(.9):.0f}  max {w.max():.0f}")
    print(f"    {'最差檔':<10}{'錨點%':>8}{'量%':>8}")
    for t in [0, 1, 2, 3]:
        m = eat.filter(pl.col("worst").round() == t) if t < 3 else eat.filter(pl.col("worst") >= 2.5)
        lab = f"{t}檔" if t < 3 else "≥3檔"
        if eat.height:
            print(f"    {lab:<10}{m.height/eat.height*100:>7.1f}%{m['plots'].sum()/el*100:>7.1f}%")


if __name__ == "__main__":
    main()
