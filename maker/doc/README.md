# Maker 研究文件索引

研究決策拆成獨立文件；不要把新內容繼續追加到單一總計畫書。

2026-08-24 清理：固定 45 檔時代的資料、程式與結果文件已移除或歸檔（見 [quote_fill/archive_fixed45/](quote_fill/archive_fixed45/)），
現行主線只剩動態商品池因果 pipeline。A1–D10 與 S0–S5 執行順序已在 [REWORK_PLAN_20260824.md](REWORK_PLAN_20260824.md) 定案。

| 文件 | 負責問題 | 狀態 |
|---|---|---|
| [00_SCOPE.md](00_SCOPE.md) | 策略方向、四條 route、共同假設 | 已凍結 |
| [05_DATA_CONTRACT.md](05_DATA_CONTRACT.md) | 2026 資料來源、欄位、時間與 hard gates | 已盤點 |
| [01_FAIR_MID_BASIS.md](01_FAIR_MID_BASIS.md) | 穩定中價 basis 與短期 fair | Provisional candidate（EWMA120），validation 未完成 |
| [fair_mid/RESULTS.md](fair_mid/RESULTS.md) | WP01 八日 pilot 數據與決策 | 2026-08-12 |
| [fair_mid/PRIOR_DAY.md](fair_mid/PRIOR_DAY.md)、[fair_mid/LEVEL_STABILITY.md](fair_mid/LEVEL_STABILITY.md) | 昨日 prior、level stability 診斷 | 八日 pilot 完成 |
| [quote_width/RESULTS.md](quote_width/RESULTS.md)、[CYCLE.md](quote_width/CYCLE.md)、[ADAPTIVE_BOUNDS.md](quote_width/ADAPTIVE_BOUNDS.md) | D−1 width、固定 BP sensitivity、自適應非對稱界線 | 八日 pilot；q 與固定 bp 將依 REWORK_PLAN S1 在共同 universe 重驗；`adaptive`／`cycle` 程式已刪 |
| [02_QUOTE_FILL.md](02_QUOTE_FILL.md)、[quote_fill/REPLAY_SAMPLING.md](quote_fill/REPLAY_SAMPLING.md) | 掛價、取樣、fill／requote 契約 | 已凍結 |
| [03_HEDGE_COST.md](03_HEDGE_COST.md) | Maker fill 後 50 ms taker 成本 | Entry 完成；exit 待做 |
| [04_BACKTEST.md](04_BACKTEST.md) | 部位、費稅、逐事件回測 | 因果線 cap 回放已有；exit maker 版待做 |
| [WALK_FORWARD.md](WALK_FORWARD.md) | 60-session 日更、fine-tune／holdout 切分 | 契約已定義 |
| [quote_fill/README.md](quote_fill/README.md) | 因果 pipeline 各段結果索引 | **現行主線入口** |
| [SKILLS.md](SKILLS.md)、[ASSIGNMENTS.md](ASSIGNMENTS.md) | 能力邊界、work packages | 後續依 S0–S5 實作同步更新 |
| [REWORK_PLAN_20260824.md](REWORK_PLAN_20260824.md) | A1–D10、共同口徑與 S0–S5 執行 checklist | **已定案；S0 待執行** |

文件規則：

- 每個研究問題只在自己的文件維護定義、指標及決策。
- 共用且已凍結的假設放在 `00_SCOPE.md` 或 `05_DATA_CONTRACT.md`，其他文件只連結，不複製長段內容。
- 實驗結果放在對應 work package 的 `RESULTS.md`；大型表格與 Parquet 放 `../data/`。
- 舊結果一律進 `archive_*/`，不刪文件但刪資料；程式可由 nested repo 快照 commit 撈回。
