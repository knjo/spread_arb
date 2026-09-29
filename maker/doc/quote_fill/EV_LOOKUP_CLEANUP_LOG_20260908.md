# EV 查表研究 — 檔案留存與清理紀錄（2026-09-08）

時間皆為 Asia/Taipei（CST）。原則：**留存**進 repo（程式 → `maker/src/ev_lookup/`、文件 → `maker/doc/quote_fill/`、結果快照 → `maker/data/ev_lookup_20260908/`，data 不進 git）；
**清理**只針對已被後續版本取代、且數字已寫入 [`EV_LOOKUP_HANDOVER_20260908.md`](EV_LOOKUP_HANDOVER_20260908.md) 結果帳的 scratchpad 中間檔。
**未動**任何 canonical bundle、凍結文件、taker 線、或本人以外的未提交修改。

## A. 背景事件（先後）

| 時間 | 事件 |
|---|---|
| 2026-09-01 ~ 09-04 | 全部分析在 session scratchpad（`/tmp/claude-1000/.../scratchpad`）進行：`ev_screen_qpairs.py`、`ev_grid_ul.py`、`integrated_backtest_v2 ~ v12.py`、`sticky_quote_replay.py`、`onesided_cancel_uplift.py`、`hedge_maker_inside.py`、`spot_maker_hedge.py`、`depth_filter_analysis.py`、`build_ext_days.py`、`chart_daily_*.csv`、`ext_daily/*.parquet`。 |
| 2026-09-03 13:00 / 13:20 | 圖表複製進 repo：`script/sideProject/report/backtest_20m_v10.png`、`backtest_ext_0902.png`（scratchpad 路徑 IDE 打不開）。 |
| 2026-09-04（S2-inside 9 樣本日測試後） | **`/tmp` 被系統清理，上述 09-01~09-04 的 scratchpad 腳本與中間檔全部遺失**（非本人刪除）。結論與公式已在 memory 與對話中，本次交接以 `ev_rules.py` 重新落地；`build_ext_days.py` 以 `ext_grid_builder.py` 重寫；`ext_daily` 在 09-07 由 `s2inside_full.py` 重建。 |
| 2026-09-07 11:55 ~ 15:25 | 重建後產出：`s2inside_capped.py`、`s2inside_full.py`、`v14_stacked.py`、`v15_dump.py`、`v16_policies.py`、`est_bpday.json` 與各自 CSV/log。 |
| 2026-09-08 10:32 ~ 11:17 | `v17_walkforward.py`（最終版）、`v17_*_daily.csv`、`backtest_v17_stacked.png`（同步複製到 `script/sideProject/report/`）。 |

## B. 留存（2026-09-08 11:24 ~ 11:28）

| 時間 | 動作 | 目的地 |
|---|---|---|
| 11:24:37 | 複製 `v17_walkforward.py` → `stacked_walkforward_backtest.py`；`v15_dump.py` → `candidate_dump.py`（工作目錄改為 `EV_LOOKUP_WORK` 環境變數，預設 `~/ev_lookup_work`） | `maker/src/ev_lookup/` |
| 11:24:37 | 複製結果快照：`v15_candidates.csv`、`v17_{20,50,100,1000000000}_daily.csv`、`v17.log`、`est_bpday.json` | `maker/data/ev_lookup_20260908/results/` |
| 11:24:37 | 複製圖：`backtest_20m_v10.png`、`backtest_ext_0902.png`、`backtest_v17_stacked.png` | `maker/data/ev_lookup_20260908/img/` |
| 11:24 ~ 11:26 | 新寫：`ev_rules.py`（全部 EV 決策式）、`policy_replay.py`（分配政策重放）、`ext_grid_builder.py`（1Hz 格重建） | `maker/src/ev_lookup/` |
| 11:27 | 編譯檢查五支腳本 `py_compile` 通過；新寫 `README.md` | `maker/src/ev_lookup/` |
| 11:28 | 新寫交接統整 `EV_LOOKUP_HANDOVER_20260908.md` | `maker/doc/quote_fill/` |
| 12:10 ~ 12:55 | 使用者指出留存回測缺 EV 決策式 → 新寫 `decide.py`（逐筆決策鏈）、`stacked_walkforward_backtest.py` 升為 v18（接入決策鏈 + 單日 trace），原版另存 `stacked_walkforward_backtest_v17.py`；v18 背景重跑後快照 `v18_*_daily.csv`、`v18_trace_20260706.csv`、`v18.log` | `maker/src/ev_lookup/`、`maker/data/ev_lookup_20260908/results/` |

## C. 清理（2026-09-08 11:29:22，逐筆）

| 時間 | 檔案 | 大小 | 理由 |
|---|---|---|---|
| 11:29:22 | `s2inside_capped.py` | 7,516 B | 9 樣本日近似版（非連續日 carry 重置，51.6k 結論已被 v17 推翻）→ 被 `stacked_walkforward_backtest.py` 取代 |
| 11:29:22 | `s2inside_full.py` | 13,191 B | S2 單獨全期版 → 併入 v17；`build_ext` 已抽成 `ext_grid_builder.py` |
| 11:29:22 | `s2inside_full_daily.csv` | 4,619 B | S2 單獨每日結果（15.9k/天）→ 已入結果帳 |
| 11:29:22 | `s2inside_full.log` | 1,929 B | 執行日誌 |
| 11:29:22 | `v14_stacked.py` | 14,942 B | 天真疊加版（14.9k）→ 被 v17 取代 |
| 11:29:22 | `v14_stacked_daily.csv` | 5,268 B | 天真疊加每日結果 → 已入結果帳 |
| 11:29:22 | `v14_stacked.log` | 1,914 B | 執行日誌 |
| 11:29:22 | `v16_policies.py` | 15,030 B | 半 in-sample bpday/deep50 真模擬器版 → 被 v17 walk-forward 取代 |
| 11:29:22 | `v16_bpday_daily.csv` | 5,172 B | 半 in-sample 結果（49.9k）→ 已入結果帳 |
| 11:29:22 | `v16_deep50_daily.csv` | 5,301 B | 半 in-sample 結果（25.5k）→ 已入結果帳 |
| 11:29:22 | `v16.log` | 1,055 B | 執行日誌 |
| 11:29:22 | `est_bpday.json` | 391 B | 5-6 月估的 8 格表（半 in-sample）→ 已快照至 results/；walk-forward 版不需要 |
| 11:29:22 | `v15_dump.log` | 34,633 B | 執行日誌 |

合計刪除 13 檔、約 111 KB。原始逐行紀錄：scratchpad `cleanup_log.txt`。

## D. 保留於 scratchpad（會隨 /tmp 清理自然消失，repo 已有副本或可重建）

`v15_dump.py`、`v17_walkforward.py`（repo 有副本）；`v15_candidates.csv`、`v17_*_daily.csv`、`v17.log`（results/ 有快照）；
`backtest_v17_stacked.png`（repo 有）；`ext_daily/`（621 MB，13 天 1Hz 格快取，可用 `ext_grid_builder.py` 重建，約 7 分鐘）。

## E. 明確未清理

- `maker/data/walkforward/*` 全部 canonical bundle、`maker/doc/*` 凍結文件、`maker/src/quote_fill/*` Codex 程式（含 S1 v4 引擎）——留待使用者決定。
- `src/research/futures_spot_spread/.ruff_cache/`、nested repo 中 `taker/*` 的未提交修改（`git status` 顯示 M，非本人所為）。
- 本次新增檔案皆**未 commit**（使用者未要求）。
