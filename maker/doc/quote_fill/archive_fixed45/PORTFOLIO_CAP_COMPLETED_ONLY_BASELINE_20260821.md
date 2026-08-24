# Portfolio cap completed-only baseline（2026-08-21）

## 結論

這是一份已驗證、可重建的 **plumbing／capacity comparator**，不是最終策略回測。
它只使用 3,672 個 established paths 中已經有 exact terminal cashflow 的 2,411
筆，排除 1,261 筆 unresolved。因此結果有明確的 survivor／completed-only
selection bias，尤其會低估隔夜庫存及資金占用；terminal overlay 完成後必須從頭
重播 admission，不能把本表當 final。

正式 bundle：
[`portfolio_cap_completed_only_baseline_20260821_v1`](../../data/walkforward/portfolio_cap_completed_only_baseline_20260821_v1)

- `complete.json` SHA-256：`166076983162450a0dcceffde7ed461aff8b5f2db19bc93f535c7f0b69fd555a`
- marker payload SHA-256：`c3e719a9eea3ead9d8f88b7e2a233f32af4f6c7c17dd107a375e7b59e2781e90`
- 完整 session calendar：62 日，`2026-05-20～2026-08-17`
- 13:00 Asia/Taipei 起禁止新倉；剛好 13:00 亦拒絕
- hard intraday portfolio cap：10／20／30／40／50M TWD
- 單商品 hard intraday cap：portfolio cap 的 30%
- EOD limit 只報 usage／breach，從未拿來拒絕日內 entry
- notional：現貨腿 entry-price one-way notional

## 摘要

|Hard intraday / EOD report|Accepted / 2,411|Entry turnover M|Peak intraday M|Peak overnight M|Gross TWD|Cost TWD|Net TWD|Net bp|Win / loss|Realized MDD|
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|10M / none|1,744|625.98|10.00|7.99|2,797,650|1,370,550|1,427,100|22.80|1,514 / 230|651|
|20M / none|2,092|831.02|19.99|8.58|3,758,950|1,788,832|1,970,118|23.71|1,826 / 266|793|
|30M / none|2,221|927.68|29.98|9.20|4,219,250|1,984,319|2,234,931|24.09|1,946 / 275|793|
|40M / none|2,282|977.10|38.63|10.85|4,479,450|2,081,238|2,398,212|24.54|2,007 / 275|793|
|40M / 20M reporting-only|2,282|977.10|38.63|10.85|4,479,450|2,081,238|2,398,212|24.54|2,007 / 275|793|
|50M / none|2,307|1,001.40|38.63|10.85|4,608,450|2,127,628|2,480,822|24.77|2,032 / 275|793|

40M／20M reporting-only 在這個 completed-only subset 中有 0 個 EOD breach，因為
peak overnight 只有 10.85M；它和 40M／none 的 admission 及 P&L 完全相同。這只證明
EOD limit 沒被偷當 hard entry cap，不能推論完整策略的 20M 留倉一定足夠。被排除的
1,261 筆正是最可能增加 carry 的路徑。

各情境另有 63 筆因 13:00 cutoff 被拒絕。40M 起 portfolio cap 本身不再拒絕這個
completed-only subset；剩餘拒絕主要來自單商品 30% cap。結果使用事後固定的 45 檔
研究 cohort，仍有 universe leakage，且尚未 joint-volume allocate。

`Realized MDD` 只來自 terminal-date realized net curve，沒有對尚未平倉 inventory 做
每日 mark-to-market；數字很小不代表真實策略風險很小。

## `backTest.py` 對照指標

|Hard cap|Daily win %|Sharpe 252|Mean active M|Mean / max daily peak M|Net / MDD|Annual return / mean active %|Annual return / cap %|
|:---|---:|---:|---:|---:|---:|---:|---:|
|10M|95.16|17.55|2.57|5.66 / 10.00|2,193.5|215.0|55.2|
|20M|95.16|15.57|3.15|7.86 / 19.99|2,485.9|242.3|38.1|
|30M|95.16|14.53|3.37|8.94 / 29.98|2,820.1|256.6|28.8|
|40M|95.16|14.02|3.48|9.60 / 38.63|3,026.1|267.0|23.2|
|40M / EOD20 report|95.16|14.02|3.48|9.60 / 38.63|3,026.1|267.0|23.2|
|50M|95.16|13.76|3.52|9.85 / 38.63|3,130.3|272.7|19.2|

|Hard cap|Same-day close %|Distinct products / all session|Distinct products / active entry day|Mean max product entry-turnover share %|
|:---|---:|---:|---:|---:|
|10M|84.58|8.16|8.58|39.03|
|20M|85.66|8.24|8.66|43.19|
|30M|85.91|8.26|8.68|44.59|
|40M|86.11|8.29|8.71|45.26|
|40M / EOD20 report|86.11|8.29|8.71|45.26|
|50M|86.26|8.29|8.71|45.75|

定義與 `backTest.py` 對齊處：daily win rate 把完整 62 sessions 中 `daily net > 0`
視為勝日；Sharpe 使用 daily net 的 sample standard deviation 與 `sqrt(252)`；年化
報酬使用 240 日線性年化。`Mean active` 是 09:00～13:30 的 time-weighted one-way
active notional；年化報酬分別除以此 mean active 與 hard cap。

`Mean max product entry-turnover share` 是每個有新倉日，各商品累積 accepted entry
notional 占當日 entry turnover 的最大值再取平均；它不是瞬時商品部位占比，因此可能
高於同時部位 30% cap。上述 Sharpe、年化率與 Net/MDD 都被 completed-only selection
bias 與 realized-only MDD 嚴重放大，只用來驗證欄位及比較 cap，不可作投資預期。

## Artifact

|檔案|內容|
|:---|:---|
|`portfolio_cap_events.parquet`|27,394 列 entry candidate／accepted exit 事件；逐事件 admission、release、turnover與部位前後狀態|
|`portfolio_cap_daily.parquet`|6 scenarios × 62 sessions＝372 列；opening、time-weighted mean、intraday peak、EOD／overnight、turnover、PnL與drawdown|
|`portfolio_cap_summary.parquet`|6 列、86 欄 cap 摘要；accepted/rejected、capacity turns、使用率、backTest parity KPI、loss quantiles與realized MDD|
|`complete.json`|source hashes、schema、artifact hashes、bias/readiness semantics與完整設定|

三張 parquet 都帶有以下防誤用欄位：

- `survivor_completed_only_selection_bias=true`
- `source_unresolved_paths_excluded=1261`
- `terminal_overlay_included=false`
- `source_universe_d_safe=false`
- `joint_volume_allocated=false`
- `final_strategy_result=false`

## 重建驗證

```bash
UV_CACHE_DIR=/tmp/fss-uv-cache uv run --no-project --with polars \
  python -m maker.src.quote_fill.portfolio_cap_completed_only_baseline \
  --verify-only \
  maker/data/walkforward/portfolio_cap_completed_only_baseline_20260821_v1
```

此命令會重新讀取 exact completed paths 與完整 session calendar、重跑六個 scenario，
再逐欄逐值比較三張 parquet；目前已通過。
