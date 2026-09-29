"""Report the full accepted-policy A/B run only after independent verification."""
import argparse
import json

import polars as pl

from ..common.paths import DATA_ROOT, grid_days
from .causal_report import plot_equity


def report(root):
    manifest = json.loads((root/"manifest.json").read_text())
    verified = json.loads((root/"verification.json").read_text())
    ev = json.loads((root/"ev_validation.json").read_text())
    if manifest["days"] != grid_days() or not verified["complete"] or verified["status"] != "PASS":
        raise ValueError("requires independent acceptance of the complete canonical period")
    if not verified["official_valuation"] or verified["metrics"].get("raw", {}).get("dates") != manifest["days"]:
        raise ValueError("requires official equity and raw execution checks for every session")
    if ev["status"] != "PASS" or not ev["complete"]:
        raise ValueError("requires submitted-signal recomputation and mature-cohort EV diagnostics")
    for cfg in manifest["configs"].values():
        if cfg["reserve_on_submit"] or cfg["max_positions_per_product"] is not None:
            raise ValueError("this report is for the user-confirmed no-reserve/unlimited-count policy")
        if cfg.get("quote_refresh_ns") != 60_000_000_000:
            raise ValueError("current report requires the 60-second refresh policy")
    if not verified["metrics"]["raw"].get("raw_quote_prices_checked"):
        raise ValueError("requires independent current-book quote price checks")
    if not verified["metrics"]["raw"].get("raw_decision_inputs_checked"):
        raise ValueError("requires raw observable-input checks for submitted and refreshed signals")
    if not all(verified["metrics"][n].get("quote_lifetimes_checked") for n in manifest["configs"]):
        raise ValueError("requires independent quote lifetime checks")
    names = [name for name in ("A", "B") if name in manifest["configs"]]
    metrics = verified["metrics"]
    header = "| 項目 | " + " | ".join("A 固定 hurdle" if name == "A" else "B 動態 hurdle" for name in names) + " |"
    separator = "|---|" + "---:|" * len(names)
    n = len(manifest["days"])
    rows = ["# " + "／".join(names) + "：依使用者確認規則完成的全期回放", "",
        f"期間：{manifest['days'][0]}–{manifest['days'][-1]}，共 {n} 個交易日。結果包含期末未平倉的官方評價。", "",
        header, separator]
    fields = [("淨損益（TWD）", "pnl_twd", ",.2f"), ("單利年化（%）", "annual_simple_pct", ".4f"),
        ("已結案淨損益（TWD）", "closed_pnl_twd", ",.2f"), ("期末未平倉淨評價（TWD）", "terminal_mark_twd", ",.2f"),
        ("每日平均淨損益（TWD）", "daily_twd", ",.2f"), ("完成進場配對", "paired_entries", ",d"),
        ("進場 rollback", "rollback_cycles", ",d"), ("資金占用峰值（TWD）", "capital_peak_twd", ",.2f"),
        ("每日權益最大回撤（TWD）", "mtm_drawdown_twd", ",.2f"), ("提高 hurdle 的日數", "raised_days", ",d")]
    for label, key, fmt in fields:
        rows.append("| " + label + " | " + " | ".join(format(metrics[name][key], fmt) for name in names) + " |")
    rows += ["", f"年化公式：`期末累積淨損益 ÷ {n} × 250 ÷ 20,000,000 × 100%`。"
             "這是固定資本的歷史單利年化；沒有把它當成複利報酬或未來保證。", "",
        "## 已確認的交易規則", "",
        "- 未成交掛單不預留資金；新單各自必須符合當時總額及商品額度。商品額度為 `max(單組名目，總資本的 25%)`。",
        "- S1 部分成交立刻以實際現貨金額占用；S2 maker 成交先以當時現貨 ask 估計，hedge 後改為實際買入金額。",
        "- 額度變少時立即送撤不再符合額度的進場單；50 ms 撤單途中成交仍如實入帳。兩腿真的平倉、待處理單結束後才釋放。",
        "- 同商品 S1／S2／E1／E2 各最多一張，可同時掛；部位不限筆數，E1／E2 各服務最早可出場部位。",
        "- 每張單生效滿 60 秒送撤，50 ms 撤單生效後按當下價格與條件重評；S1／S2 再算 EV、hurdle 及額度，E1／E2 保留原目標並更新掛價。部分成交先處理真實曝險。",
        "- A 為 8.5 bp／交易日，即 5.8219178 bp／日曆日。B 僅在前日收盤占用達 80% 額度時，取前日通過基礎門檻的獨立訊號分數 q50 與基礎門檻之較大值。", "",
        "## 驗證證據", "",
        f"逐日帳務與事件稽核通過 {n} 日；原始行情核對 {verified['metrics']['raw']['checked_legs']:,} 筆非結算執行腿。"
        "資金稽核另直接讀取每個相關時刻的原始現貨 ask，不依賴引擎自己的資金估計。",
        f"EV 機率展開驗算 1,000 組；重新估算全期實際送單訊號 {ev['signals']['submitted_signals']:,} 筆，"
        "檢查 EV、預估持有天、模型 P_sd 及 score。", "",
        f"計時更新決策（含拒單）{ev['signals']['refresh_checks']:,} 筆亦重新估算，"
        f"合計 {ev['signals']['checked_signals']:,} 筆決策核對公式及各組實際 hurdle 的通過／拒絕結果。", "",
        "必要容量撤單逐筆核對：" + "，".join(f"{name} {metrics[name]['mandatory_capacity_cancels_checked']:,} 次" for name in names) + "。", "",
        "掛單壽命核對：" + "，".join(f"{name} {metrics[name]['quote_lifetimes_checked']:,} 張" for name in names) + "；"
        + "60 秒到期事件：" + "，".join(f"{name} {metrics[name].get('quote_expirations_checked', 0):,} 次" for name in names) + "。"
        +
        f"原始行情逐筆核對全部送單掛價 {verified['metrics']['raw']['raw_quote_prices_checked']:,} 筆。", "",
        f"另從原始雙腿行情及 1 Hz anchor 格重建 {verified['metrics']['raw']['raw_decision_inputs_checked']:,} 筆決策輸入，"
        "包含全部已送單與計時更新的訊號，核對決策時間、價差、hedge tick 成本、anchor 及 residual。", "",
        "容量超額來源（筆數；撤單延遲與價格重估保留真實成交，不刪交易）：", "",
        "| 來源 | " + " | ".join(names) + " |", separator]
    for label, key in [("撤單途中成交造成超額", "cancel_race_over_cap"),
                       ("S2 成交時現貨估價上升", "maker_estimate_repricing_over_cap"),
                       ("S2 實際 hedge 重估", "hedge_repricing_over_cap")]:
        rows.append("| " + label + " | " + " | ".join(str(metrics[name].get(key, 0)) for name in names) + " |")
    peak = max(pl.read_csv(root/f"{name}_daily.csv")["peak_rss_gib"].max() for name in names)
    rows += ["", f"A／B 由一個程序共用原始行情，prefetch=0；回放程序量測 RAM 峰值 {peak:.2f} GiB。", "",
        "## EV 預測與實現", "",
        "以下只取合約到期日已落在樣本期內、完成進場 hedge 的部位，避免只挑提前結案的交易。"
        "失敗進場與 rollback 仍計入上表總損益。", "",
        "| 組別 | n | 預測 EV bp | 實現淨 bp | 預測天數 | 實際天數 | 模型 P_sd | 實際當日結案率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in ev["calibration"]["cohorts"]:
        if r["portfolio"] not in names or r["dimension"] not in ("all", "route"):
            continue
        rows.append(f"| {r['portfolio']}／{r['group']} | {r['n']:,} | {r['expected_ev_bp']:.2f} | "
            f"{r['realized_net_bp']:.2f} | {r['predicted_days']:.2f} | {r['realized_days']:.2f} | "
            f"{r['predicted_same_day']:.1%} | {r['realized_same_day']:.1%} |")
    rows += ["", "P_sd 是模型的當日到價機率，實際當日結案另包含到期結算／風險出場；到期路線沒有 maker 目標，P_sd 固定為 0，"
        "所以這兩欄不是同一定義的當沖預測與標籤。市場到價機率與 maker 實際完成機率不同，持有時間模型也使用收盤代理值。"
        "公式與回測帳務通過不代表預測已校準；上述落差保留呈現，這次沒有用同批績效反向調參。", "",
        "## 解讀限制與舊結果", "",
        "保留研究假設：同日／隔夜費用 20／34 bp，無融資成本，到期採 basis=0 結算，"
        "以公共成交量與可見深度模擬 FIFO／post-only，無逐筆委託簿 ID。結果是此模型下的歷史模擬，非實盤保證。",
        "S1 部分成交後可繼續等待全滿，撤單生效仍未滿才 rollback；taker hedge 使用五檔足量才整筆執行，"
        "不足時保留曝險並重試，沒有模擬部分 IOC hedge。這些成交近似保留在本次模型範圍中。",
        "原點位 A／B 的 34.9166%／37.8241% 算術可重算，但沒有因此驗證執行。"
        "全額預留變體的 13.5978%／13.4869% 是不同容量政策。此次另依使用者明確確認允許 S1／S2 各自掛單，"
        "不能把新舊差額全稱為單一 bug 的影響。", "",
        "前一版 `accepted_20260923` 的 13.2562%／13.6853% 尚無 60 秒更新；本次加入該規則後重新從清理／重建的 Q 快取與原始行情回放。"
        "計時更新的決策不加入 B 的獨立市場訊號母體，實際送單 EV 全數仍列入重算。", "",
        "![每日權益與回撤](equity.png)", "",
        "證據：[verification.json](verification.json)、[ev_validation.json](ev_validation.json)、"
        "[manifest.json](manifest.json)、" + "、".join(f"[{name} 每日對帳]({name}_reconciled.csv)" for name in names) + "。", ""]
    plot_equity(root)
    path = root/"ACCEPTED_RESULTS.md"
    path.write_text("\n".join(rows))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    print(report(DATA_ROOT/"backtest"/args.run))


if __name__ == "__main__":
    main()
