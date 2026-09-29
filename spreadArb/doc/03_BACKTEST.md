> 2026-09-23 整理：現行規則與驗證狀態見 [CURRENT.md](CURRENT.md)。本文件保留前一階段規格／結果；其中數字未包含本次新增的每路線 60 秒更新，不可直接視為本次驗收。

# Stage 3：回測架構

**2026/9/23 使用者定義優先**：未成交掛單不占部位容量；每筆掛單須符合當下剩餘額度，
成交後重查並撤掉不再容納的進場單，平倉額度只在實際事件發生時釋放。
下文全額預留為其他研究政策，不是目前使用者策略要求；詳見 [容量規則更正](05_CAPITAL_POLICY_CLARIFICATION.md)。

現行實作、已修正的交易流程、保留假設及 CLI 見 [04_CORRECTED_REPLAY.md](04_CORRECTED_REPLAY.md)。
本文保留完整研究規格；其中尚未實作的政策變體／λ 機會成本等，不應視為目前 A／B 已有功能。

目標：在 Stage 2 點位表上做 **sequential portfolio replay**——共用容量、carry 跨日、到期 C8、資金成本、逐部位帳本——
輸出使用者要的績效面：總獲利、單筆獲利、交易量／週轉、當沖率、平均持倉部位與時間、資金使用與峰值、回撤、S1／S2 歸因、月別。
兩層回放：**點位回放**（快，從事實表取，容量近似）用於政策迭代；**完整事件回放**（慢，共用隊列與深度）用於最終驗證。
maker 經驗：點位回放比完整模擬樂觀約 17%，任何勝出政策都要回到完整回放確認。

## 1. 逐事件狀態（每個 position）

```text
QUOTING → ENTRY_FILLED → ENTRY_HEDGING → PAIRED → EXIT_QUOTING → EXIT_FILLED → EXIT_HEDGING → CLOSED
                │                                    │
                └── CANCELLED / ROLLBACK              └── SURVIVE（跨日） → 次日 EXIT_QUOTING … → 到期 C8
```

Portfolio 只做三件事：**准入**（EV、λ、容量）、**容量帳本**、**跨日鏈接**（carry、到期、公司行動）。
掛單／撤單／成交／hedge 的事實在點位回放層直接取自 Stage 2；完整回放層則重新驅動 Stage 2 的狀態機並共用 `PrintedVolumeQueue` 與 `TakerDepth`。

## 2. 容量帳本（沿用 maker `CapacityLedger`）

- 單位：現貨名目 TWD（整數分）；S1＋S2 共用；`committed = carry + 日內已配對 + 未完成 hedge + 掛單預留`。
- S2 掛單以當日漲停價預留，hedge 完成縮到實際；S1 以掛價預留。
- 兩腿正常出場**都完成**才釋放；C8 只在到期後結算 phase 釋放，不能資助同 cursor 的新進場。
- 撤單生效前的成交即使超額也入帳（`unreserved_fill`，記 `overrun_cents`）；報表列超額金額與持續時間。
- 政策變體（皆為參數，不是複本）：`reserved`（掛單全額預留）／`shared`（掛單不合計預留，逐張看當下餘額）／
  `release_credit`（`+min(5M, 0.5×今日折扣預期釋放 + 0.25×隔日折扣增量)`，預測不扣帳本）／S1 deep queue（低於 B1 的單可無額度掛著，接近 B1 無額度才撤）。
- λ：前 5 完成 session 被容量拒絕且影子後來成交的估計 EV ÷ cap，clip 15 bp／日。

## 3. 跨日與終結

| 事件 | 處理 |
|---|---|
| `survive` | 部位、anchor、L、凍結 EV 原樣帶到次日；次日 09:05 起重試出場 |
| 到期（C8） | 持有過到期日，次 session 開盤前結算：兩腿同價、basis = 0、`pnl = ab − 34`；期貨 hedge 尚未完成的部位不得用 C8 洗平 |
| 公司行動 | 生效日前禁新掛；生效日 carry 標 `continuity_blocked`，走 `other` 強平（taker 兩腿） |
| 行情中斷日（8/28） | 不模擬執行、保留 carry、不出 label |
| 合約轉換 | `contract_size` 變動需顯式部位換算，否則 fail closed |
| 日終 | 全部 maker 單 13:18 撤；未 hedge 曝險跨日保留並繼續重試，報表列出 |

資金成本：每筆現貨買入本金 × 日曆持有天（含週末、中斷日）× 年率（預設 2%，另列 0／4／6%）；現貨賣出後、期貨未買回期間停計。

## 4. 指標（每日一列＋全期彙總＋月別＋S1／S2 歸因）

| 類別 | 指標 |
|---|---|
| 損益 | 已實現四腿 PnL（TWD）、期末持倉評價（官方結算價；BBO 另列敏感度）、資金成本、淨權益、C8 歸因、rollback／other 歸因 |
| 單筆 | 淨 bp（等權、本金加權分列）、TWD／筆、掛單時 EV vs 事後淨 bp（成熟 cohort、按 EV 桶）、P_sd 預測 vs 實際 |
| 量 | maker 成交筆數（S1／S2）、配對數、正常平倉本金／20M／日（週轉輪）、掛撤訊息數與 1 s 峰值 |
| 當沖 | 成熟配對的當日平倉率、截至次 session 平倉率（分母含期末仍持有者） |
| 部位 | 開盤／10:00／收盤 committed、盤中時間加權已配對本金、實際股票本金峰值、超 cap 日比例與小時數、期末 carry |
| 時間 | 本金加權持有日曆日、預期 vs 實際持有、hedge 等待分布（50 ms 恰好／超時／最長） |
| 風險 | 官方日終權益最大回撤（缺值留缺，不填零、不連線）、日內未 hedge 曝險時長、負 basis 成交數、撤單競賽成交數 |
| 目標 | 250 日規劃年化（20M 分母）與實際峰值本金分母兩種；對 30% 目標（24,000 元／日）的差距 |

回報口徑固定：日均以**有資料日**為分母（85）；暖機後（65）另列；分段（S1＋S2 雙流 52 日／S2 單流 13 日）另列，只要 S1 自產就不再有這個分段。

## 5. 驗證（每次 full run 必附）

- 帳本：`committed` 逐事件重建；容量峰值 = cap（reserved 政策）；`unreserved_fill` 必有不可撤回成交支持。
- 現金：每筆結案 `quote_ab − (anchor − 5) − entry_decay − exit_decay − fee == realized_bp`（容差 1e-9 bp）。
- 因果：出場 ns > 進場 ns；解決日 ≥ 進場日；Q／λ／release 表只含 `< D` 的觀測；颱風休市只在公告後。
- 抽查：四個固定日原始 print 逐筆核對 maker 成交與 hedge 消耗深度。
- 續跑：從任一日 checkpoint 重啟與連續執行逐值一致（部位、帳本、Q 表、決策）。
- 對帳：以 maker 現行政策參數（S2 A1−1、25/20 bp、深度 5×、buffer50、anchor−5、reserved 20M＋release credit）重跑，
  65 日日均應落在 maker `EVENT_S2_Q_PROFIT` 的 +10,599（基本）～ +11,678（增額）量級；差異需逐項歸因（S1 自產 vs 外生、anchor、事件順序）。

## 6. 政策比較的規矩

- 所有政策**事前固定**（寫進 manifest）；同一歷史區間看過結果再改門檻，只能稱為探索性比較，不能稱樣本外。
- 一次只變一個維度；改出場政策必須配同政策的獨立 label 來源（點位表重跑），不能沿用不同出場規則的 Q 表。
- 不以「成交數」推論市場供給；不以受容量限制的成交數推論機會數。
- 不刪不利成交、不縮分母、不回填未來達標。

## 7. 產出

```text
data/backtest_<run>/
├── manifest.json            # 政策參數、來源 hash、Stage 1／2 run 指標、完成狀態
├── Date=<D>/{positions,ledger,decisions,events}.parquet + checkpoint
├── report/{daily.csv, summary.json, monthly.csv, attribution.csv, calibration.csv, hedge_latency.csv, messages.csv}
├── verification.json
└── equity.png（缺值斷線）
```
