"""表三完整版：閾值 × 進場停留 交叉，每格含 筆數/淨利/收斂率/峰值/淨利率/收斂等待中位。

⚠️ 舊模型參考：此表建立在「每事件一筆、potential_lots=進場那刻掛量」的舊口徑。
   架構已改為「多次進場/均價/合約整天一池/留倉壓結算」，新模型報表需重做——
   本檔保留僅為「格式」參考（欄位排版、交叉表呈現方式），數字勿直接當新結果用。

代跑：.venv/Scripts/python.exe verify/table3_full.py
"""
import os
import sys
import glob
from datetime import datetime, timedelta

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import report_first as stats
from spread_arb.metrics import SPOT_SHARES_PER_LOT as SP   # 現貨1張=1000股(出場流動性換算)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
frames = []
for f in sorted(glob.glob(os.path.join(OUT, "events_*_tick.csv"))):
    if "_m" in os.path.basename(f):
        continue
    df = pl.read_csv(f)
    if "entry_fut_bid" in df.columns:
        frames.append(df)
ev = pl.concat(frames, how="diagonal")

b = ev.filter(pl.col("potential_lots") > 0)
b = stats.enrich(b, mode="2")


def peak_by_group(sub):
    on = sub.filter(~pl.col("converged"))
    occ = {}
    for date, d2s, vc, cap in on.select("date", "days_to_settle", "ValueCode", "capital").rows():
        di = datetime.strptime(str(date), "%Y%m%d")
        k = (di, di + timedelta(days=int(d2s)), vc)
        occ[k] = max(occ.get(k, 0.0), cap)
    dt = (sub.filter(pl.col("converged")).group_by("date", "ValueCode")
          .agg(pl.col("capital").max()).group_by("date").agg(pl.col("capital").sum().alias("c")))
    cmap = {str(d): v for d, v in dt.rows()}
    days = sorted({datetime.strptime(str(d), "%Y%m%d") for d in sub["date"].unique().to_list()})
    ncp = cp = 0.0
    for day in days:
        o = sum(cap for (di, ds, vc), cap in occ.items() if di <= day < ds)
        t = cmap.get(day.strftime("%Y%m%d"), 0.0)
        ncp = max(ncp, o); cp = max(cp, t)
    return ncp, cp


print("表三完整版（B 出場、mode2、整年）")
print("（淨利率_收斂=只收斂組均；淨利率_全體=含不收斂均(不收斂=留倉=0上界)，後者較不高估）")
print("（打折後收斂淨利=收斂淨利×出場流動性 fill_rate，扣掉第一檔吃不掉的那塊；只折當沖/收斂那塊）")
print(f"{'閾值':<6}{'進場停留':<10}{'收斂筆數':>8}{'收斂淨利':>9}{'打折後淨利':>11}{'不收斂筆數':>10}"
      f"{'不收斂淨利':>10}{'收斂率':>7}{'淨利率_收斂':>11}{'淨利率_全體':>11}"
      f"{'整段開啟中位':>12}{'收斂等待中位':>13}{'不收斂峰值':>11}{'收斂峰值':>10}")
print("-" * 146)
for thr in [0.005, 0.01, 0.015, 0.02]:
    base = b.filter(pl.col("threshold") == thr)
    for lab, lo, hi in [("次秒(<1s)", -1, 1), ("≥1秒", 1, 1e18),
                        ("≥30秒", 30, 1e18), ("全部", None, None)]:
        # 「全部」用 base 真全量(含 stretch=null 那批，跟 summary 對齊)；
        # 分層列用 stretch 過濾(null 兩條件都不成立、自然排除，因為它們算不出停留時間)。
        if lab == "全部":
            seg = base
        else:
            seg = base.filter((pl.col("first_stretch_secs") >= lo)
                              & (pl.col("first_stretch_secs") < hi))
        if seg.height == 0:
            continue
        c = seg.filter(pl.col("converged"))
        nc = seg.filter(~pl.col("converged"))
        rate = c.height / seg.height * 100
        # 打折後收斂淨利＝Σ(收斂淨利×出場流動性 fill_rate)：每筆部位 vs 自己出場掛量，兩腳取緊封頂1。
        #   出不掉的部分不算進去＝「實際出得掉那塊賺多少」。需 exit_*_lots(E08)，舊CSV無則留空。
        if c.height and "exit_fut_ask_lots" in c.columns:
            cf = c.filter(pl.col("exit_fut_ask_lots").is_not_null()
                          & pl.col("exit_spot_bid_lots").is_not_null())
            fut_cov = pl.col("exit_fut_ask_lots") / pl.col("potential_lots")
            spot_cov = (pl.col("exit_spot_bid_lots") * SP) \
                / (pl.col("potential_lots") * pl.col("contract_size"))
            cf = cf.with_columns(
                pl.min_horizontal(pl.min_horizontal(fut_cov, spot_cov), pl.lit(1.0)).alias("_fill"))
            pnl_fill = (cf["potential_pnl"] * cf["_fill"]).sum()
            pf_s = f"{pnl_fill/1e8:.3f}億"
        else:
            pf_s = "-"
        # 淨利率(該格)：收斂組均(舊欄，會高估) + 全體均(含不收斂，較誠實；不收斂=留倉=0上界)
        nr = c["net_ret"].mean() * 100 if c.height else 0
        nr_all = seg["net_ret"].mean() * 100 if seg.height else 0
        # 整段開啟中位=整格 signal_span 中位(span 是事件屬性，收斂/不收斂都有)
        sp = seg["signal_span_secs"].median()
        sp_s = (f"{sp:.1f}秒" if sp is not None and sp < 60 else
                f"{sp/60:.0f}分" if sp is not None else "-")
        # 收斂等待中位=收斂事件 hold_secs 中位
        hd = c["hold_secs"].median() if c.height else None
        hd_s = f"{hd/60:.0f}分" if hd else "-"
        # 峰值：累積口徑(≥門檻含更高)，次秒不算峰值(留空)
        if lab.startswith("次秒"):
            ncp_s = cp_s = "-"
        else:
            ncp, cp = peak_by_group(seg)
            ncp_s = f"{ncp/1e8:.1f}億"; cp_s = f"{cp/1e8:.1f}億"
        print(f"{thr:<6.1%}{lab:<10}{c.height:>8,}{c['potential_pnl'].sum()/1e8:>8.3f}億"
              f"{pf_s:>11}{nc.height:>10,}{nc['potential_pnl'].sum()/1e8:>9.3f}億"
              f"{rate:>6.0f}%{nr:>10.2f}%{nr_all:>10.2f}%{sp_s:>12}{hd_s:>13}{ncp_s:>11}{cp_s:>10}")
    print()
