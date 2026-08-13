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
| [02_QUOTE_FILL.md](02_QUOTE_FILL.md) | 動態掛價、取樣、fill／requote／risk gate | 已規劃 |
| [quote_fill/REPLAY_SAMPLING.md](quote_fill/REPLAY_SAMPLING.md) | SpreadPair epoch、多層存續掛單、同價去重與撤單統計 | 規格已凍結，待 raw replay |
| [03_HEDGE_COST.md](03_HEDGE_COST.md) | Maker fill 後 50 ms taker VWAP 與成本 | 已規劃 |
| [04_BACKTEST.md](04_BACKTEST.md) | 部位、費稅、週轉與逐事件回測 | 已規劃 |
| [05_DATA_CONTRACT.md](05_DATA_CONTRACT.md) | 2026 資料來源、欄位、時間與 hard gates | 已盤點 |
| [SKILLS.md](SKILLS.md) | 研究模組所需能力與責任邊界 | 維護中 |
| [ASSIGNMENTS.md](ASSIGNMENTS.md) | Work packages、順序與交付物 | 維護中 |

文件規則：

- 每個研究問題只在自己的文件維護定義、指標及決策。
- 共用且已凍結的假設放在 `00_SCOPE.md` 或 `05_DATA_CONTRACT.md`，其他文件只連結，不複製長段內容。
- 實驗結果放在對應 work package 的 `RESULTS.md`；大型表格與 Parquet 放 `../data/`。
- 舊的單一總計畫書拆分完成後不再作 source of truth。
