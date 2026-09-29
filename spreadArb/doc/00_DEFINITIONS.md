# 00：名詞、公式、成本與資料契約

本文是 spreadArb 的唯一定義來源。每條定義標註 maker 出處；若與 maker 程式衝突，以本文為準並回頭修正 maker 對照表。
價格單位：程式內以 `int`（TWD × 1e4）運算，文件以 bp 表示。

## 1. Basis 與 anchor

```text
B(F, S) = 10,000 × (F / S − 1)                       # bp
B_mid        = B(FutMid, SpotMid)                    # 只用於 anchor
B_sell_taker = B(FutExecBid, SpotA1)                 # taker/taker 進場參考（買現貨、賣期貨）
B_buy_taker  = B(FutExecAsk, SpotB1)                 # taker/taker 出場參考（賣現貨、買期貨）
```

`FutExecBid/Ask` = 明掛 L1 與 `BestBidPrice/BestAskPrice` 的可成交優價（含對應 Lots）；期貨成交 row 五檔可能全零，須維護最近有效 book。

**Anchor `M_t`**：`B_mid` 的 causal EWMA，只用 `≤ t` 的合法 book 更新，逐秒取值。

| 版本 | 定義 | 出處 | 用途 |
|---|---|---|---|
| `ewma_120s` | `α = 1 − 0.5^(1/120)`；canonical 值在 `walkforward/daily/Date=D/causal_fair.parquet` 的 `anchor_ewma_120s_bp`（≤ 8/13），之後由 `market.py::_anchors` 自 tick 重建 | `maker/src/ev_lookup_cost/market.py` | ev_lookup 線全部 Q 表與 EV 的 anchor；**Stage 1 對帳基準** |
| `time_ewma_15s` | 同式、半衰 15 s | S0.5 `FOUNDATION_SELECTION_S05_REBUILD_20260826.md` | 30～300 s 未來中心 MAE 8.722 bp，S1 cost-aware 線採用；**Stage 2 起的第一個對照** |

Anchor 在掛單當下**凍結**進該筆 position（`p.anchor`），之後的出場目標與 EV 重估都用凍結值，不隨行情重設。

## 2. 掛價與鎖定 basis

| 符號 | 定義 |
|---|---|
| `ab`（absolute basis） | 我方 maker 掛價對另一腿**可執行價**鎖定的 basis |
| `effU`（residual／「價差的價差」） | `ab − anchor` |
| `U` | S1 的目標 residual；S2 的 U 由 A1 − 1 tick 反推，不是自由參數 |
| `L` | 出場目標 residual，目前固定 −5（出場觸發條件 `B(FutA1, SpotB1) ≤ anchor + L`） |

```text
S1  Spot Bid maker @ P_s = floor_spot_tick(FutExecBid / (1 + U/1e4))
    ab = B(FutExecBid, P_s)
S2  Future Ask maker @ P_f = previous_tick(FutA1)          # inside、隊列第一
    需 FutB1 < P_f < FutA1（一 tick 寬的期貨盤口沒有合法 maker 價位）
    ab = B(P_f, SpotA1)
E1  Spot Ask maker @ SpotA1，觸發條件 B(FutA1, SpotB1) ≤ anchor + L
E2  Future Bid maker @ next_tick(FutB1)，觸發條件同上（maker 未實作，本線新增）
```

`actual_ab` = hedge 完成後以四腿實際現金重算的 basis；`entry_decay = quote_ab − actual_ab`。

**座標與到達（Stage 1 定案）**：

```text
scale_D−1 = 商品 D−1 以前 trailing 20 日 residual 的 (q95 − q50)      # 只用 < D
e         = (quote_ab − d_in − anchor) / scale_D−1                     # 進場水位
x         = 出場水位（同單位）；B_x = anchor + x · scale_D−1
到達      = 在 B_x 掛的出場 maker 單「實際成交」；不是價差觸及 B_x
```

Q 表只用被成交的路徑：樣本單位是一筆 shadow 進場成交，結局是各 x 的出場單是否成交。出場水位是決策變數，
`L` 不再是固定常數；`anchor − 5` 只是格點之一（對帳用）。

Tick 規則（`market.py::tick_i`，單位 1e-4 TWD）：`<10 → 0.01；<50 → 0.05；<100 → 0.1；<500 → 0.5；<1000 → 1；其餘 5`。
跨級距時前一 tick 用較低級距（`previous_tick`）。

## 3. 進場／撤單門檻（maker 現行值，Stage 1 沿用對帳，Stage 2 起可調）

| 項目 | S1 | S2 | 出處 |
|---|---|---|---|
| 新掛門檻 | 由 EV／P_fill 決定（本線新增） | `ab > 0` 且 `effU ≥ 25 bp` | `portfolio.py::S2_ENTRY_RESIDUAL_BP` |
| 持單地板 | 同上 | `effU ≥ 20 bp` 且 `ab > 0`，否則送撤 | `S2_HOLD_FLOOR_BP` |
| drift 撤單 | — | 持單 basis 較掛單時 `quote_ab` 掉 10 bp 送撤（guard 版） | `repeg_drop_bp` |
| 對腿深度 | 期貨 L1–L5 可完整賣 1 口 | 現貨 A1 ≥ 5 × hedge 股數（標準 2,000 股 → 10,000 股） | v22 `depth5` |
| 穿價成本下限 | — | 50% 機率多付一個現貨 tick 的 VWAP 差 | v22 `buffer50` |
| 掛單時段 | 09:05（300 s）～ 12:53:20（14,000 s） | 同 | `_requote_s2` |
| 同商品併發 | 一張 | 一張；前一口現貨 hedge 未完成不得新掛（取代舊 60 s CD） | `event_s2_policy.py` |
| 再評估時鐘 | 每個可觀察 book 事件＋撤單生效後＋hedge 完成當下 | 同 | 同上 |
| 全部 maker 撤回 | 13:18（15,480 s） | 同 | `MAKER_WITHDRAW_SECOND` |

價格合法性 gate（兩腿各自）：`−9% < P / RefPrice − 1 < +8%`，等號排除；`TrialMatch == 0` 才是正式盤；
crossed book、零價有量、缺 RefPrice、hedge 深度不足一律關閉 route。**必要 hedge 不受 −9%/+8% 訊號緩衝限制**，
現貨用當日漲跌停、期貨用 ±10% 參考價。

## 4. 成交、撤單與 hedge 模型

| 模型 | 規則 | 出處 |
|---|---|---|
| Maker 成交 | `PrintedVolumeQueue`：保守 FIFO；掛單時隊列前方量 = 該價位當下可見量；一筆 print 的量只能被消耗一次（含外部隊列與我方所有單）；`placed_ns ≥ print ns` 的單不能被該 print 成交；撤單不贈與優先權 | `execution.py` |
| S2 成交判定 | 期貨 print 價 ≥ 我方賣價即消耗隊列（買方主動成交必先打到隊列第一） | 同上 |
| 撤單 | 意圖時間 + 50 ms 生效；生效前到達的 print 仍可成交（撤單競賽） | `request_cancel` |
| Hedge | maker 完整成交 + 50 ms 第一次嘗試；L1–L5 足量 VWAP；同一 book snapshot 的深度不重複使用；不足則等下一個 book（≤ 1 s 重試），5 s 記 `hedge_timeout` 但繼續等，不用 mark 或到期洗平 | `TakerDepth`、`hedge()` |
| S1 部分成交 | 現貨只成交部分張數不足一口期貨 → 撤 leaves 後 `entry_rollback`（taker 賣回現貨），事件 `rollback` | `cancel()` |
| S2 容量預留 | 掛單時以當日現貨漲停價 × 股數預留，hedge 完成後縮到實際本金 | `submit()` |

## 5. 出場與終結事件

正常出場（E1，兩 stream 相同；`anchor − 5` 為 maker 現行值，本線改為 Stage 1 選出的 `B_x`）：09:05 起，`B(FutA1, SpotB1) ≤ B_x` 時掛現貨 Ask maker 賣單；
成交後 +50 ms 買回期貨。持單期間若 `B(FutA1, exit_price) > anchor − 5` 或期貨 ask 無量，撤單（`_guard_exit_orders`，逐期貨 book 事件檢查）。
非當日部位逐日重試同規則。

終結事件（互斥、完備）：

| event | 定義 | 入帳 |
|---|---|---|
| `normal`（same-day／overnight） | E1 完成四腿 | 四腿實際現金 − 稅費 |
| `rollback` | S1 部分成交回補 | 實際現金；淨 bp 以**原掛單名目**為分母 |
| `other` | 公司行動強平、taker cross（實驗）等 | 實際現金 |
| `expiry`（C8） | 持有過到期日，次 session 結算 | 兩腿同結算價、basis = 0 → `ab − 34` bp |
| `survive` | 當日收盤仍未結案 | 不入損益；進 carry 風險集 |

## 6. 成本口徑

| 項目 | 值 | 說明 |
|---|---|---|
| 同日 round trip | **20 bp** | 現貨手續費 1.71 × 2 + 當沖賣出稅 15 + 期貨稅 0.2 × 2 = 18.82，取整 20 |
| 隔夜 round trip | **34 bp** | 現貨手續費 3.42 + 賣出稅 30 + 期貨稅 0.4 = 33.82，取整 34 |
| 多單留倉（空現貨） | 54 bp | 34 + 融券 20；本線第一版**不做**多價差 |
| 期貨手續費 | 20 TWD／邊 | 精確帳用 `transaction_costs.py`；EV 用 bp 近似 |
| 資金成本 | 年 2%（敏感度假設） | 按現貨原始本金 × 日曆持有天（含週末）；期貨保證金融資未計 |
| 容量 | 20M TWD 現貨名目（S1＋S2 共用） | 含 carry、日內、未完成 hedge、掛單預留 |
| λ（容量影子價） | 前 5 完成 session 被容量拒絕且影子後來成交之估計 EV ÷ cap，bp／日，clip 15 | `ev_rules.shadow_price` |

執行衰減（查表，Stage 1 沿用 maker 定義）：

```text
d_in  = E[quote_ab − actual_ab]            進場：quote → hedge 完成，含成交前漂移與 hedge 滑價
d_sd  = E[(F_buy − S_sell) / S_buy × 1e4 − (anchor − 5)]   同日出場相對凍結目標的 shortfall
d_on  = 同上，隔夜出場
下限 0，各加 margin 3 bp；樣本 < 30 依序 pooled，最後 prior 15 / 5 bp
S2 進場：d_in = max(查表值, 深度＋50% 一 tick 成本)，不相加
```

maker guard 組實測（85 日，bp，括號 p95）：S1 進場 25.72（77.82）、S2 進場 8.92（45.66）、
S1 同日出場 −14.45、S1 隔夜 −13.15、S2 同日 −5.42（82.54）、S2 隔夜 −6.33（75.25）。負均值不代表無風險，S2 出場右尾很寬。

## 7. 時間

- 交易日 `D`，開盤 `open_ns(D)` = 01:00:00 UTC；`CLOSE_SECOND = 15,600`（13:20）；秒序 `sec = (ns − open) // 1e9`。
- 所有跨市場排序用 UTC ns `RecvTime`；同 `RecvTime` 內用各市場自己的 `ChannelSeq`，**兩市場之間不得比較 sequence**。
- `TransTime` 只作稽核。
- 時段桶 `time_bucket`：`< 3600 s`／`< 9000 s`／其餘；出場桶 `exit_bucket` 起點 `0／9000／12600／14400`。

## 8. 資料契約（沿用 maker `05_DATA_CONTRACT.md`）

| 資料 | 路徑 | 備註 |
|---|---|---|
| 現貨 tick | `${tick_dir}/{D}_StockTick.parquet`（SSD2） | float 真實價、naive UTC µs；`common/paths.py` 經 `pipeline.yaml` 解析 |
| 個股期 raw | `/mnt/NAS/Parquet/Ticks/YYYY/MM/DD/stock_futures.parquet` | 整數價依 `DecimalLocator` 還原；tz-aware UTC ns；唯一來源，無 fallback |
| 現貨基本資料 | `${market_dir}/{D}_marketData.parquet` | `opening_ref_price`、漲跌停、`allow_day_trade_mark ∈ {X, Y}` |
| 期貨基本資料 | `maker/data/ev_lookup_v19_metadata_20260908/{D}_futures_basic.parquet`（89 日快取） | 近月標準合約、`contract_size = 2000`、`end_date`、`fut_ref_price` |
| 商品對應 | `walkforward/daily/Date={D}/mapping.parquet` | ValueCode ↔ QuoteCode；缺檔時由 basic + marketData 重算 |
| Canonical anchor | `walkforward/daily/Date={D}/causal_fair.parquet` | ≤ 8/13；之後自 tick 重建 |
| 預告日曆 | `forecast_calendar.py`：年度休市表 + 7/10 颱風（7/9 公告） | 決策日只能用公告日以前的休市 |
| 公司行動 | `ev_lookup_v19_metadata_20260908/announcements/index.json` | `announce_day < D` 才可用；生效前禁新掛、生效日 carry 標記 |
| 全期 session | `walkforward/sessions.txt`（131 日，2026-01-26 起）；ev_lookup 線用 5/4～9/2 | 8/28 行情中斷：保留 carry、不出 label |

禁用 feature：`Close`、`FutureHigh/Low`、`SpreadNarrow*`、`FutureAsk1_*`、`FutureBid1_*`、`TakerSell/Buy_CloseBP`、`midEdge_*`；
任何全日統計或未來 label 不得進決策。現貨 `makerFill` 只有 A1/A2/B1/B2 的 `FillSeconds`，本線不用它判定成交，只作 sanity。
