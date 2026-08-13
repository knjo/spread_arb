# Maker Basis 研究線

本目錄研究期貨／現貨 basis 的日內 maker 策略：在高 basis 以一腿 maker、另一腿固定約 50 ms 後 taker 建立多現貨／空期貨部位，再於 basis 回落時用同樣結構平倉。

目錄固定分為：

- [`doc/`](doc/README.md)：研究規格、資料契約、skills、assignments 與決策紀錄。
- [`src/`](src/README.md)：可重跑的資料處理、模型及回測程式。
- [`data/`](data/README.md)：本研究衍生的 landmarks、labels、模型與報告；預設不進 Git。

第一個 work package 是 [`doc/01_FAIR_MID_BASIS.md`](doc/01_FAIR_MID_BASIS.md)：驗證能否找到穩定、causal、可用於參照掛單的中價 basis。八日 pilot 保留 EWMA120 作 provisional 候選；完整 validation 尚未通過，決策與數據見 [`doc/fair_mid/RESULTS.md`](doc/fair_mid/RESULTS.md)。

跨商品的掛單寬度 pilot 已拆到 [`doc/quote_width/RESULTS.md`](doc/quote_width/RESULTS.md)：用 D−1 的 tick／spread／TTBand 與 non-overlap basis excursions 建立 D 日參數表，並明確與 maker fill、50 ms hedge 及完整 EV 分離。

固定格點的 latent FSM 見 [`doc/quote_width/CYCLE.md`](doc/quote_width/CYCLE.md)，其中 10／15／20／30 BP 與固定 tick 都只作 sensitivity。正式研究入口改為 [`doc/quote_width/ADAPTIVE_BOUNDS.md`](doc/quote_width/ADAPTIVE_BOUNDS.md) 的 D−1 商品別非對稱界線；WP02 再依當下 fair、反腿行情與合法 ladder 產生 rounded action。

既有 `../taker/` 是獨立的 taker 研究線；可重用契約 mapping 與清洗邏輯，但 maker 的取樣、queue 與逐事件撮合另行實作。
