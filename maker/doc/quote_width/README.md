# Quote Width 研究索引

| 文件 | 內容 | 狀態 |
|---|---|---|
| [RESULTS.md](RESULTS.md) | D−1 商品參數、正常發散表、隔日驗證與 rounded quote geometry | 八日／四商品 pilot |
| [CYCLE.md](CYCLE.md) | 固定 BP／tick 格點的非重疊 latent sensitivity | 八日／四商品 diagnostic |
| [ADAPTIVE_BOUNDS.md](ADAPTIVE_BOUNDS.md) | 商品別非對稱上下界、觸界與日內回歸機率 | 八日／四商品 latent pilot |
| [../WALK_FORWARD.md](../WALK_FORWARD.md) | 最近60交易日的日更上下界與實盤式驗證契約 | 全市場執行中 |

本目錄只研究 fair-mid 上下應張開多少，以及 latent basis 是否在日內走完上下界。Maker fill、50 ms hedge 與 executable cycle EV 分別留在 WP02–04。
