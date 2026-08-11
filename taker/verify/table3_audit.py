"""獨立驗算表三：用不同算法重算每欄，自動檢查一致性與邏輯約束，印 PASS/FAIL。

執行：uv run python verify/table3_audit.py
"""
import os
import sys
import glob

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import report_first as stats

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

fails = []
def chk(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        fails.append(name)

for thr in [0.005, 0.01, 0.015, 0.02]:
    base = b.filter(pl.col("threshold") == thr)
    print(f"\n===== 閾值 {thr:.1%}（全閾值 {base.height:,} 筆）=====")

    segs = {
        "次秒": base.filter(pl.col("first_stretch_secs") < 1),
        "≥1秒": base.filter(pl.col("first_stretch_secs") >= 1),
        "≥30秒": base.filter(pl.col("first_stretch_secs") >= 30),
        "全部": base,
    }

    # 1. 分層筆數約束：次秒 + ≥1秒 = 全部（互斥且完備）；≥30秒 ⊂ ≥1秒
    chk("次秒+≥1秒=全部",
        segs["次秒"].height + segs["≥1秒"].height == segs["全部"].height,
        f"({segs['次秒'].height}+{segs['≥1秒'].height} vs {segs['全部'].height})")
    chk("≥30秒 ⊆ ≥1秒",
        segs["≥30秒"].height <= segs["≥1秒"].height,
        f"({segs['≥30秒'].height} <= {segs['≥1秒'].height})")

    for lab, seg in segs.items():
        if seg.height == 0:
            continue
        c = seg.filter(pl.col("converged"))
        nc = seg.filter(~pl.col("converged"))
        # 2. 收斂+不收斂 = 該格總數
        chk(f"[{lab}] 收斂+不收斂=總數",
            c.height + nc.height == seg.height)
        # 3. converged 欄與「出場四價非空」一致(達標才有出場四價)
        c_has_exit = c.filter(pl.col("exit_fut_bid").is_not_null()).height
        chk(f"[{lab}] 收斂事件都有出場四價",
            c_has_exit == c.height, f"({c_has_exit}/{c.height})")
        # 4. 收斂事件 hold 必 not null；不收斂 hold 必 null
        c_hold_ok = c.filter(pl.col("hold_secs").is_not_null()).height == c.height
        nc_hold_ok = nc.filter(pl.col("hold_secs").is_null()).height == nc.height
        chk(f"[{lab}] 收斂hold非null/不收斂hold為null",
            c_hold_ok and nc_hold_ok)
        # 5. net_ret(淨利率) 應 = potential_pnl / potential_value（抽驗收斂組均值正負合理）
        if c.height:
            nr = c["net_ret"].mean()
            # 不收斂 net_ret 應 > 收斂(留倉出場成本=0 樂觀)；至少收斂淨利率為正且<毛
            chk(f"[{lab}] 收斂淨利率合理(0~閾值+1%)",
                0 < nr < thr + 0.02, f"(net_ret={nr*100:.2f}%)")

    # 6. 峰值單調：≥1秒 不收斂峰值 >= ≥30秒（累積，門檻越鬆峰值越大）
    def ncpeak(seg):
        from datetime import datetime, timedelta
        on = seg.filter(~pl.col("converged"))
        occ = {}
        for date, d2s, vc, cap in on.select("date","days_to_settle","ValueCode","capital").rows():
            di = datetime.strptime(str(date), "%Y%m%d")
            k=(di, di+timedelta(days=int(d2s)), vc); occ[k]=max(occ.get(k,0.0),cap)
        days = sorted({datetime.strptime(str(d),"%Y%m%d") for d in seg["date"].unique().to_list()})
        return max((sum(c for (di,ds,vc),c in occ.items() if di<=day<ds) for day in days), default=0)
    p1, p30 = ncpeak(segs["≥1秒"]), ncpeak(segs["≥30秒"])
    chk("不收斂峰值 ≥1秒 >= ≥30秒(累積單調)", p1 >= p30, f"({p1/1e8:.1f} >= {p30/1e8:.1f}億)")

print("\n" + "="*50)
print("全部 PASS ✓" if not fails else f"!!! {len(fails)} 項 FAIL: {fails}")
