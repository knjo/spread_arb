"""Compare independently verified corrected A/B with frozen original ledgers."""
from __future__ import annotations

import argparse
from datetime import datetime
import json

import numpy as np
import polars as pl

from ..common.paths import DATA_ROOT
from ..common.grid import close_marks
from .audit import load, reconcile_daily


def plot_equity(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, height_ratios=[2, 1])
    for name, label, color in (("A", "A: fixed hurdle", "#2367b4"), ("B", "B: dynamic hurdle", "#13836a")):
        if not (root/f"{name}_reconciled.csv").exists():
            continue
        daily = pl.read_csv(root/f"{name}_reconciled.csv",schema_overrides={"day":pl.String})
        dates = [datetime.strptime(day,"%Y%m%d") for day in daily["day"]]
        equity = daily["equity_twd"].to_numpy()
        drawdown = equity-np.maximum.accumulate(np.r_[0.,equity])[1:]
        axes[0].plot(dates,equity/1e6,label=label,color=color,lw=1.6)
        axes[1].plot(dates,drawdown/1e3,color=color,lw=1.2)
    axes[0].set_title("Corrected A/B replay: cumulative P&L and daily equity drawdown")
    axes[0].set_ylabel("Cumulative P&L (TWD million)")
    axes[0].legend(loc="upper left",frameon=False)
    axes[1].set_ylabel("Drawdown (TWD thousand)")
    axes[1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for ax in axes:
        ax.axhline(0,color="#777777",lw=.6)
        ax.grid(alpha=.2)
        ax.spines[["top","right"]].set_visible(False)
    fig.text(.5,.015,"Official daily marks; 20/34 bp costs; basis-zero expiry accounting; no financing costs.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1))
    fig.savefig(root/"equity.png",dpi=160,bbox_inches="tight")
    fig.savefig(root/"equity.svg",bbox_inches="tight")
    plt.close(fig)


def report(root):
    verified = json.loads((root/"verification.json").read_text())
    if not verified["complete"] or verified["status"] != "PASS":
        raise ValueError("a complete passing independent audit is required before reporting results")
    if not verified["official_valuation"]:
        raise ValueError("this comparison requires official daily valuation")
    manifest = json.loads((root/"manifest.json").read_text())
    records = []
    for name, baseline in (("A","A_fixed"),("B","B_dyn")):
        pos, daily, rb, cfg = load(baseline)
        old = float(pos["pnl_net"].sum())+float(rb["cost_twd"].sum())
        new = verified["metrics"][name]
        n, cap = new["days"], cfg["cap_twd"]
        corrected_cfg = manifest["configs"][name]
        if abs(corrected_cfg["cap_twd"]-cap) > 1e-6:
            raise ValueError("baseline and corrected capital must match")
        for field in ("fee_same_day_bp", "fee_overnight_bp"):
            if abs(cfg["cost"][field]-corrected_cfg["cost"][field]) > 1e-10:
                raise ValueError("baseline and corrected paired fee conventions must match")
        annual = old/n*250/cap*100
        rec = reconcile_daily(pos,daily,rb)
        old_core = rec.filter(pl.col("day").is_between(pl.lit("20260401"),pl.lit("20260731")))["reconciled_booked_twd"].sum()
        new_daily = pl.read_csv(root/f"{name}_reconciled.csv",schema_overrides={"day":pl.String})
        if daily.height != n or set(daily["day"]) != set(new_daily["day"]):
            raise ValueError("baseline and corrected comparison periods must match")
        core = new_daily.filter(pl.col("day").is_between(pl.lit("20260401"),pl.lit("20260731")))
        corrected_positions = pl.read_parquet(root/f"{name}_positions_all.parquet")
        opened = corrected_positions.filter(pl.col("state") != "closed")
        marks = close_marks(str(new_daily["day"][-1]),opened["vc"].unique().to_list()) if opened.height else {}
        aligned_open = 0.0
        aligned_missing = []
        for p in opened.iter_rows(named=True):
            if p["vc"] not in marks:
                aligned_missing.append(p["id"])
                continue
            sp, fp = marks[p["vc"]]
            sq = p["spot_buy_qty"]-p["spot_sell_qty"]
            fq = p["future_sell_qty"]-p["future_buy_qty"]
            aligned_open += ((p["spot_sell_cash"]-p["spot_buy_cash"]+p["future_sell_cash"]-p["future_buy_cash"]
                            +sq*sp-fq*p["shares"]*fp)/10000-p["entry_spot_cash"]/1e8*34-p["extra_fees"])
        aligned_total = None if aligned_missing else new["closed_pnl_twd"]+aligned_open
        rows = dict(portfolio=name, baseline=baseline, days=n, baseline_pnl_twd=old,
                    corrected_pnl_twd=new["pnl_twd"], pnl_difference_twd=new["pnl_twd"]-old,
                    pnl_difference_pct=(new["pnl_twd"]-old)/old*100,
                    baseline_daily_twd=old/n, corrected_daily_twd=new["daily_twd"],
                    baseline_annual_pct=annual, corrected_annual_pct=new["annual_simple_pct"],
                    annual_difference_pp=new["annual_simple_pct"]-annual,
                    baseline_pairs=pos.height, corrected_pairs=new["paired_entries"],
                    baseline_closed_pairs_pnl_twd=float(pos.filter(pl.col("close_kind")!="open_marked")["pnl_net"].sum()),
                    baseline_terminal_mark_twd=float(pos.filter(pl.col("close_kind")=="open_marked")["pnl_net"].sum()),
                    corrected_closed_pnl_twd=new["closed_pnl_twd"],
                    corrected_terminal_mark_twd=new["terminal_mark_twd"],
                    baseline_rollback_twd=float(rb["cost_twd"].sum()),
                    corrected_rollback_twd=float(corrected_positions.filter(pl.col("close_kind")=="rollback")["pnl_net"].sum()),
                    corrected_mtm_drawdown_twd=new["mtm_drawdown_twd"],
                    corrected_cap_peak_twd=new["capital_peak_twd"],
                    old_mark_aligned_pnl_twd=aligned_total, old_mark_missing_positions=aligned_missing,
                    terminal_valuation_difference_twd=None if aligned_total is None else new["pnl_twd"]-aligned_total,
                    core_days=core.height, baseline_core_booked_twd=float(old_core),
                    corrected_core_booked_twd=float(core["realized_twd"].sum()),
                    corrected_core_mtm_twd=float(core["mtm_pnl"].sum()),
                    terminal_positions=opened.height)
        for source in ("baseline_core_booked", "corrected_core_booked", "corrected_core_mtm"):
            rows[source+"_daily_twd"] = rows[source+"_twd"]/core.height
            rows[source+"_annual_pct"] = rows[source+"_daily_twd"]*250/cap*100
        records.append(rows)
    output = dict(records=records, baseline_B_minus_A_twd=records[1]["baseline_pnl_twd"]-records[0]["baseline_pnl_twd"],
                  corrected_B_minus_A_twd=records[1]["corrected_pnl_twd"]-records[0]["corrected_pnl_twd"])
    (root/"comparison.json").write_text(json.dumps(output,indent=2)+"\n")
    pl.from_dicts([{k:v for k,v in r.items() if not isinstance(v,list)} for r in records]).write_csv(root/"comparison.csv")
    lines = ["# A／B 回測邏輯修正後全期間比較", "", f"期間 2026/1/26–8/13，共 {records[0]['days']} 日；資金 2,000 萬。",
             "", "沿用原 20／34 bp 成本及到期 basis＝0 記帳假設。年化為總損益 ÷ 131 × 250 ÷ 2,000 萬，非複利 CAGR。", "",
             "| 指標 | A 固定 hurdle | B 動態 hurdle |", "|---|---:|---:|"]
    specs = [("原版總損益", "baseline_pnl_twd", ",.0f"), ("修正版總損益", "corrected_pnl_twd", ",.0f"),
             ("總損益差額", "pnl_difference_twd", "+,.0f"),
             ("總損益變動（%）", "pnl_difference_pct", "+.2f"),
             ("原版日均", "baseline_daily_twd", ",.0f"),
             ("修正版日均", "corrected_daily_twd", ",.0f"), ("原版年化（%）", "baseline_annual_pct", ".4f"),
             ("修正版年化（%）", "corrected_annual_pct", ".4f"), ("年化差額（百分點）", "annual_difference_pp", "+.4f"),
             ("原版配對數", "baseline_pairs", ",d"), ("修正版配對數", "corrected_pairs", ",d"),
             ("修正版資金峰值", "corrected_cap_peak_twd", ",.0f"),
             ("修正版期末未平倉數", "terminal_positions", ",d"),
             ("修正版官方日終權益回撤", "corrected_mtm_drawdown_twd", ",.0f")]
    for label,key,fmt in specs:
        lines.append(f"| {label} | {format(records[0][key],fmt)} | {format(records[1][key],fmt)} |")
    lines += ["", "損益組成（元）：", "", "| 項目 | A 固定 hurdle | B 動態 hurdle |", "|---|---:|---:|"]
    for label,key in (("原版已結案配對","baseline_closed_pairs_pnl_twd"),
                      ("原版 rollback","baseline_rollback_twd"),
                      ("原版期末評價","baseline_terminal_mark_twd"),
                      ("修正版已結案損益（已含 rollback）","corrected_closed_pnl_twd"),
                      ("其中：修正版 entry rollback","corrected_rollback_twd"),
                      ("修正版期末評價","corrected_terminal_mark_twd")):
        lines.append(f"| {label} | {records[0][key]:,.3f} | {records[1][key]:,.3f} |")
    lines += ["", f"B−A：原版 {output['baseline_B_minus_A_twd']:,.0f} 元；修正版 {output['corrected_B_minus_A_twd']:,.0f} 元。", "",
              "修正版從 raw books／prints 產生新交易路徑，沒有刪除或靜態重估原交易來代替重跑。", "",
              "差異包含：完整獨立訊號、精確 EV 狀態／合法 scale／S2 成本下限、max-Q 目標一致的估計、到期持有時間、"
              "掛單資金預留、50 ms 生效後 FIFO 成交量、共用 hedge 深度、partial／double-exit 回補、完整時段出場、"
              "同合約到期處理及已公告公司行動風險出場。", "",
              "這是以上修正共同作用的完整重跑差異；因持倉、資金占用與 B 次日門檻互相影響，不能把各項異常的舊交易損益相加當作獨立貢獻。", "",
              "每日權益使用官方現貨收盤與原持有合約的官方期貨日結算價，完整涵蓋 131 日。"
              "專案資料庫涵蓋 127 日；缺少的 6/26、6/30、7/13、7/14 從期交所原始日行情補齊。"
              "下載與資料庫的控制日比對涵蓋 1,192 筆股票期貨，價格全數一致；來源與雜湊存於 metadata。"
              "此回撤與原版只計結案損益的回撤口徑不同。", "",
              "期末評價口徑與原版對齊的敏感度：", ""]
    for r in records:
        aligned = r["old_mark_aligned_pnl_twd"]
        lines.append(f"- {r['portfolio']}："+(f"總損益 {aligned:,.0f} 元；本報告期末估值造成的差異 {r['terminal_valuation_difference_twd']:+,.0f} 元。"
                     if aligned is not None else f"缺少 {len(r['old_mark_missing_positions'])} 筆原口徑標記，保留缺值。"))
    lines += ["", "4/1–7/31 的同期間比較（83 日）：", "",
              "| 指標 | A 固定 hurdle | B 動態 hurdle |", "|---|---:|---:|"]
    for label,key in (("原版結案入帳損益（含 rollback）","baseline_core_booked_twd"),
                      ("修正版結案入帳損益（含 rollback）","corrected_core_booked_twd"),
                      ("修正版每日 MTM 損益加總","corrected_core_mtm_twd")):
        lines.append(f"| {label} | {records[0][key]:,.0f} | {records[1][key]:,.0f} |")
    for label,key in (("原版結案入帳年化（%）","baseline_core_booked_annual_pct"),
                      ("修正版結案入帳年化（%）","corrected_core_booked_annual_pct"),
                      ("修正版 MTM 年化（%）","corrected_core_mtm_annual_pct")):
        lines.append(f"| {label} | {records[0][key]:.4f} | {records[1][key]:.4f} |")
    lines += ["", "結案入帳反映這段時間完成的交易；每日 MTM 包含跨越區間起訖日的庫存評價變化，兩者不可混用。"]
    lines += ["", "驗證詳見 `verification.json`：全期間 cash legs、資金 ledger、四腿現金、逐日權益、獨立訊號與 B 門檻對帳；"
              "raw 逐筆抽驗日期與筆數另列於其中。FIFO 僅靠 public prints，不能還原交易所完整逐委託佇列；新單使用 post-only 模擬。", "",
              "到期 basis＝0、20／34 bp 與未加入資金利息仍是保留的研究假設，未宣稱已模擬官方到期結算／實際券商成本。", ""]
    plot_equity(root)
    lines += ["![修正版每日累積損益與回撤](equity.png)", "", "向量版：[equity.svg](equity.svg)。", ""]
    text = "\n".join(lines)
    (root/"COMPARISON.md").write_text(text)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run",default="corrected_20260922_v2")
    args = ap.parse_args()
    print(report(DATA_ROOT/"backtest"/args.run))


if __name__ == "__main__":
    main()
