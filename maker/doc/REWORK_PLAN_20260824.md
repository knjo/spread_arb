# 重作計畫（定稿）：把因果 pipeline 擴回原始規格

日期：2026-08-24　狀態：**已討論定案，照此執行**。完成一項就在本文件打勾並填結果連結。

## 研究定位（使用者定義）

這是類套利策略：估好價差、算準執行滑價與稅費，理論上不應賠錢。因此：

- **不設「通過／不通過」門檻**。主報表固定兩個數字：20M cap 成本後日均 net、同日完成率。
- **同日完成率是最佳化目標**（當沖賣出稅 15 bp vs 隔夜 30 bp）。
- 界線與商品池**逐月更新、隨市況微調**是設計的一部分（60-session rolling 已是此機制），不是 leakage。
- 所有閾值都要知道結果，因為未來實盤要靠它們決定怎麼掛。

## 已定案的決策

| 題 | 決定 |
|---|---|
| A1 界線 policy | `q50 / q80 / q95` ＋ 固定對稱 `15 / 20 / 25 / 30 bp`，共 7 組，全跑 |
| A2 出場下緣 | frozen（submit 當下鎖 `anchor − lower`）；dynamic 只當 sensitivity |
| A3 8 月惡化 | 先查再跑 A1。假說：8 月價差波動縮、溢價消失（界線碰到率掉）vs 界線掛太深（碰到後成交率掉） |
| B4 期貨 Ask maker route | 加回來當對照；不做雙路同掛 |
| B5 成交標籤 | 全程用 `makerFill` 欄位（tick 級，已截我們的撤單時間）；不做 exact replay，只在最終 policy 上跑一次 exact 當校準係數 |
| B6 hedge 定價 gate 失敗（2.76%） | 往後找 5 秒內第一個合法期貨 book 定價，標 `hedge_delayed=true`，滑價另列；不再當 unpriced 占容量 |
| C7 exit maker | 先做「現貨 Ask maker → 買期貨 taker」；跑通再加「期貨 Bid maker → 賣現貨 taker」 |
| C8 13:00 後 | (a) 允許 carry、expiry 前強制平 與 (b) 13:00 起積極平倉、零留倉 **都跑**，比較「犧牲平倉收益換隔日滿部位重作」是否划算 |
| C9 部位 | **20M（2,000 萬）＋單檔 50%** 為主；其他 cap 之後再做 |
| D10 驗收 | 無門檻；報 20M 日均 net 與同日完成率 |

## 固定不變的前提

- 商品池：`monthly_product_selector_causal_v2_20260822/daily_entry_manifest.csv`（72 日、3,886 product-days）；所有 run 吃這份，不得再傳固定 symbol list。
- 中價 causal EWMA120；q 界線用 60-session rolling、`<D`、正負側分開。固定 bp 對稱，上下都用同一個 W。
- 取樣：SpreadPairTotalCount epoch、同絕對價不重掛、後撤只撤更積極層、1 Hz final-net。
- 只掛 A/B1–2。
- Hedge：`fill RecvTime + 50 ms`，arrival／decision 分開，L1–L5 VWAP，正值＝不利。
- 成本：現貨雙邊 1.71 bp、賣出稅當沖 15／隔夜 30 bp、期貨雙邊 0.2 bp + TWD 20（`quote_fill/transaction_costs.py`）。
- 13:00 停新倉。

## 工作項目與順序

```
S0 8 月歸因 ──► S1 七組 policy（現貨 Bid route）──► S2 期貨 Ask route ──► S3 exit maker ──► S4 13:00 policy ──► S5 定稿與 exact 校準
                       │
                       └─ B6 hedge 定價規則在 S1 一起改
```

### S0　8 月惡化歸因（先做，一天）

- [ ] 逐月（May–Aug）：mid-basis excursion 分布（p50/p80/p95 幅度、每日 excursion 數）vs 當日 D-1 q95 界線位置。
- [ ] 分開報「界線碰到率」（latent）與「碰到後 makerFill 成交率」；兩者哪個掉，決定是市況還是界線。
- [ ] 若是市況：S1 的固定 bp 組會直接顯示哪個寬度在 8 月還有機會數，不用另外處理。若是界線：檢查 60-session rolling 在 regime 轉換時的滯後，考慮加 30-session challenger（只作 sensitivity）。

輸出：`doc/quote_fill/AUGUST_ATTRIBUTION_<date>.md`，一張逐月表 + 一張圖。

### S1　七組 policy × 現貨 Bid maker route

- [ ] `one_second_makerfill_runner.attach_q95_boundaries` 泛化為 `attach_boundaries(policy)`，policy ∈ {q50, q80, q95, fixed15, fixed20, fixed25, fixed30}。
- [ ] `dynamic_estimated_path_portfolio.py:110` 拿掉 `frozen to q95`；frozen lower 對 fixed 組 = `anchor − W`。
- [ ] `dynamic_future_hedge`：gate 失敗改為往後 5 秒內第一個合法 book，加 `hedge_delayed`、`hedge_delay_ms` 欄；summary 分 on-time／delayed 兩列報滑價。
- [ ] cap 回放改 20M＋單檔 50%。
- [ ] 每個 policy 各自跑完整鏈，輸出根目錄 `*_causal_<policy>_<date>`；同一張 raw order 被多 policy 命中時共用 `physical_order_id`，跨 policy 不相加。
- [ ] 比較表欄位：candidates、q target 落在 B1／B2／B3+ 的 product-seconds 比例、fill 率、cancel 率、submit-to-fill p50/p95、hedge slip p50/p95（on-time／delayed）、同日／跨日／expiry 比例、uncapped net bp、**20M 日均 net、同日完成率**、日均新 spot。
- [ ] 逐月拆一次同一張表（May／Jun／Jul／Aug）。

輸出：`doc/quote_fill/POLICY_COMPARISON_SPOT_BID_<date>.md`。

### S2　期貨 Ask maker → 買現貨 taker route（對照）

- [ ] 期貨 maker 沒有 legacy makerFill 欄位，用 `execution_runner` + `indexed_replay` 產 fill；**先拆掉 `sessions × symbols` 笛卡兒積介面**，改讀 manifest 的 `(Date, ValueCode)`。
- [ ] Hedge 反向：fill + 50 ms 買現貨 2 lots，L1–L5 VWAP；期貨一口無 partial。
- [ ] 期貨 5 requests/s：maker quote 與 fill hedge 共用 limiter，hedge 優先；message load 重算。
- [ ] 與 S1 相同欄位，加 route 維度；兩 route sampling contract 不同，**只並列不 pooling**。
- [ ] 先跑 S1 表現最好的 2 組 policy，不必 7 組全跑。

輸出：`doc/quote_fill/POLICY_COMPARISON_FUTURE_ASK_<date>.md`。

### S3　Exit maker（先一條 route）

- [ ] 對 S1 每筆 entry position，在 frozen lower 掛「現貨 Ask maker」，成交後 +50 ms 買期貨 taker。取樣、分層、後撤規則與 entry 相同。
- [ ] Exit hedge 滑價用 entry 同一套 arrival／decision 算法，另出一張表。
- [ ] 未成交的 competing outcome：同日 taker/taker（現行 1 Hz first passage 當 control）、carry、expiry。三者互斥。
- [ ] FIFO：同商品多 position 的 exit 合成「商品 × 絕對價」一張 working order，不逐 position 相加。
- [ ] 完成後 cap 回放改吃這組 terminal；與 S1 的 taker/taker 版並列，看 exit maker 多賺多少。
- [ ] 跑通後加「期貨 Bid maker → 賣現貨 taker」。

輸出：每筆 position 有 `exit_route / exit_fill_time / exit_hedge_slip_bp / exit_outcome`；`doc/quote_fill/EXIT_MAKER_CAUSAL_<date>.md`。

### S4　13:00 後 policy：carry vs 積極平倉

- [ ] (a) carry：13:00 後停新倉、exit maker 繼續掛到收盤；未平者隔日續掛；expiry 前一日 13:00 起強制 taker 平。
- [ ] (b) aggressive：13:00 起 exit 改為 B1／A1 peg（沿用舊 aggressive controller 的邏輯，程式需從快照撈回重寫成 manifest 版），13:20 目標零留倉。
- [ ] 兩者在 20M cap 下比：日均 net、同日完成率、隔日開盤可用容量、平倉收益犧牲量。
- [ ] 回答使用者的問題：「每天犧牲一部分平倉收益積極平倉，換隔日滿部位重作」是否更好。

輸出：`doc/quote_fill/CLOSE_POLICY_CARRY_VS_AGGRESSIVE_<date>.md`。

### S5　定稿與校準

- [ ] 從 S1–S4 選定 policy／route／close 組合，寫成一份 `STRATEGY_SPEC_<date>.md`：每日開盤前要算什麼、盤中怎麼掛、13:00 後怎麼做。
- [ ] 對選定組合抽 5 個固定日跑 exact own-quantity replay，給 makerFill 的校準係數（預期約 0.9×）。
- [ ] 2026-09 資料到齊後，用同一套凍結規則跑一次，與 development 期並列；不設門檻，只看兩個主數字有沒有掉。

### 隨時可做

- [ ] `doc/quote_fill/README.md` 隨各 S 完成更新索引。
- [ ] 舊 aggressive／exit maker 程式需要時從 commit `1348576` 撈。

## 目前 data/ 分層

| 層 | 目錄 | 大小 |
|---|---|---:|
| 基礎事實（所有 run 的輸入） | `walkforward/daily`、`rolling_boundaries`、`liquidity`、`sessions.txt`、`exact_contract_calendar_v1.parquet`、`expiry_daily_close_facts_20260821_v1` | 21G |
| 因果 manifest | `monthly_product_selector_causal_v2_20260822` | 6M |
| 8/22 q95 baseline（S1 會產生等價物後可刪） | `order_message_load_*`、`one_second_makerfill_*`、`dynamic_future_hedge_*`、`dynamic_expiry_paired_close_*`、`dynamic_estimated_path_portfolio_*`、`august_exit_extension_*` | 230M |
| A/B1–2 決策證據 | `makerfill_rank_l1_l5_sample_20260820_v5`、`future_ask_rank_l1_l5_indexed_sample_20260821_v1` | 13M |
| 八日 pilot | `fair_mid/`、`quote_fill/`、`quote_width/` | 110M |
