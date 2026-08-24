# Maker 研究文件索引

研究決策拆成獨立文件；不要把新內容繼續追加到單一總計畫書。

| 文件 | 負責問題 | 狀態 |
|---|---|---|
| [00_SCOPE.md](00_SCOPE.md) | 策略方向、四條 route、共同假設 | 已凍結 |
| [01_FAIR_MID_BASIS.md](01_FAIR_MID_BASIS.md) | 穩定中價 basis 與短期 fair | Provisional candidate，待完整 validation |
| [fair_mid/RESULTS.md](fair_mid/RESULTS.md) | WP01 pilot 數據、決策與限制 | 2026-08-12 已稽核 |
| [fair_mid/PRIOR_DAY.md](fair_mid/PRIOR_DAY.md) | 昨日同合約 basis prior 是否降低誤差 | 八日 pilot 完成 |
| [fair_mid/LEVEL_STABILITY.md](fair_mid/LEVEL_STABILITY.md) | 絕對／局部 level 與 fast-slow uncertainty | 八日診斷完成 |
| [quote_width/RESULTS.md](quote_width/RESULTS.md) | D−1 商品 width prior、正常發散與隔日驗證 | 八日／四商品 pilot |
| [quote_width/CYCLE.md](quote_width/CYCLE.md) | 固定 BP／tick 格點的非重疊 latent sensitivity | 八日／四商品 diagnostic |
| [quote_width/ADAPTIVE_BOUNDS.md](quote_width/ADAPTIVE_BOUNDS.md) | 每商品非對稱上下界、供給與條件回歸機率 | 八日／四商品 latent pilot |
| [02_QUOTE_FILL.md](02_QUOTE_FILL.md) | 動態掛價、取樣、fill／requote／risk gate | 八日 raw pilot 完成 |
| [quote_fill/REPLAY_SAMPLING.md](quote_fill/REPLAY_SAMPLING.md) | SpreadPair epoch、多層存續掛單、同價去重與撤單統計 | 規格已實作於 pilot |
| [quote_fill/PILOT_RESULTS.md](quote_fill/PILOT_RESULTS.md) | Raw fill、撤單、partial、50 ms hedge與fill後latent exit | 2026-08-14 pilot |
| [quote_fill/EXIT_MAKER_60D_RESULTS.md](quote_fill/EXIT_MAKER_60D_RESULTS.md) | 60-session entry／exit maker、matched T/T、overnight與D-safe lookup結果 | 290 product-days完成；非EV-ready |
| [quote_fill/LIQUIDITY_SCREEN.md](quote_fill/LIQUIDITY_SCREEN.md) | 全市場 A1-B1 spread、route liquidity 與 raw-replay universe縮減 | 131 日完成；retrospective研究名單 |
| [quote_fill/README.md](quote_fill/README.md) | Quote fill／execution文件入口與完成度 | 維護中 |
| [quote_fill/EV_LOOKUP.md](quote_fill/EV_LOOKUP.md) | 合法tick action、terminal path、D-safe查表EV與131日rollout | 介面已實作；execution／EV待擴樣 |
| [03_HEDGE_COST.md](03_HEDGE_COST.md) | Maker fill 後 50 ms taker VWAP 與成本 | Entry hedge pilot 完成；exit待做 |
| [04_BACKTEST.md](04_BACKTEST.md) | 部位、費稅、週轉與逐事件回測 | EV contract已實作；portfolio replay待做 |
| [WALK_FORWARD.md](WALK_FORWARD.md) | 60-session 日更機率表、fine-tune 與 locked validation | 131 日 latent／liquidity layer完成；execution／EV待擴樣 |
| [05_DATA_CONTRACT.md](05_DATA_CONTRACT.md) | 2026 資料來源、欄位、時間與 hard gates | 已盤點 |
| [SKILLS.md](SKILLS.md) | 研究模組所需能力與責任邊界 | 維護中 |
| [ASSIGNMENTS.md](ASSIGNMENTS.md) | Work packages、順序與交付物 | 維護中 |
| [REWORK_PLAN_20260824.md](REWORK_PLAN_20260824.md) | 因果線擴回原始規格的 checklist：多 q／固定 bp、期貨 route、exit maker、hedge fail-safe、holdout | **現行工作清單** |

文件規則：

- 每個研究問題只在自己的文件維護定義、指標及決策。
- 共用且已凍結的假設放在 `00_SCOPE.md` 或 `05_DATA_CONTRACT.md`，其他文件只連結，不複製長段內容。
- 實驗結果放在對應 work package 的 `RESULTS.md`；大型表格與 Parquet 放 `../data/`。
- 舊的單一總計畫書拆分完成後不再作 source of truth。
