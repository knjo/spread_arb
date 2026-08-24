# 重作計畫：把因果 pipeline 擴回原始規格

日期：2026-08-24
狀態：待執行；本文件是後續工作的唯一 checklist，完成一項就在這裡打勾並填結果連結。

## 為什麼要重作

8/20–8/22 為了修掉固定 45 檔的 universe leakage，重寫了一條因果 pipeline
（M-1 選月池 → D-1 liquidity gate → 1 Hz quote intent → approximate makerFill → +50 ms hedge
→ taker/taker frozen-lower exit → inventory cap 回放）。這條線的紀律是對的，但為了趕出端到端結果，
它是一個 **窄版**：

| 原始規格 | 舊 fixed-45 線（有 leakage） | 現行因果線 | 缺口 |
|---|---|---|---|
| 1a 比較 q50 / q80 / q95 | 三個 q 並列 | **只有 q95**，程式寫死 | 要補 q50、q80 |
| 1b q vs 中價 ± 固定 15/20/25/30 bp | 只在 8 日 latent path 比過 | 沒做 | 要在 fill／hedge／PnL 層比 |
| 2a 期現兩條 entry route 分開看 | Future Ask maker 與 Spot Bid maker 都有 | **只有 Spot Bid maker** | 要加回 Future Ask route |
| 2a 進出場分開看 hedge 滑價 | exit maker 有 replay，但 exit hedge 滑價沒量 | 出場是 taker/taker 估計，沒有 maker exit | 要做 exit maker + exit hedge |
| 3 只掛 A/B1–2 | 兩本獨立診斷支持 | 已採用 | 無 |
| 4a SpreadPair 去重 | 已實作 | 已實作 | 無 |
| 4b 秒 K 掛撤 | 已實作 | 已實作（1 Hz final-net） | 無 |
| 4c 撤單／成交時間／hedge 欄位 | 有 | 有（`candidate_outcomes.parquet`） | 期貨 route 的欄位在因果線沒有 |
| 5 各 q 的事件數／掛單／部位／獲利 | 有但 universe 髒 | 只有 q95 | 同 1a／1b |
| 6 部位控制回測 | 有 | 有，但 exit 假設是 taker/taker | 換成 maker exit 後重跑 |

另外三個不是規格缺口、但會影響數字可不可信的問題：

- makerFill 是 mixed-clock approximate label（`snapshot RecvTime + Float32 FillSeconds`），沒有 own quantity／partial／cancel race；
- +50 ms hedge 有 213 筆 gate closed 沒定價、沒 retry，在 20M cap 裡占 9.08M 容量；decision book age p95 0.5 秒、227 筆 > 1 秒；
- 8 月成交率（0.97%）與同日收斂率（8.39%）崩掉，沒有拆開「basis 行為變了」與「D-1 q95 界線掛太深」。

## 固定不變的前提

以下在本輪重作中 **不再重新討論**：

- 商品池：`monthly_product_selector_causal_v2_20260822/daily_entry_manifest.csv`（72 日、3,886 product-days）。所有新 run 都吃這份 manifest，不得再傳固定 symbol list。
- 中價：causal EWMA120；上下緣：60-session rolling、`<D`、正負側分開。
- 取樣：SpreadPairTotalCount epoch、同絕對價不重掛、後撤只撤更積極層、1 Hz final-net。
- 掛單點位：主研究只掛 A/B1–2。
- Hedge 基準：`fill RecvTime + 50 ms`，arrival／decision 分開，L1–L5 VWAP，正值＝不利。
- 成本：現貨雙邊 1.71 bp、賣出稅當沖 15／隔夜 30 bp、期貨雙邊 0.2 bp + TWD 20。
- 部位：hard cap 10/20/30/40/50M、單品 30%、13:00 停新倉、unresolved 占容量。
- 所有輸出維持 `production_strategy_go=false`，直到 P5 的 prospective holdout 出來。

## 工作項目

優先順序由上而下；P1–P2 可並行，P3 依賴 P1 的 entry positions，P4 依賴 P3，P5 最後。

### P1　同一 manifest 加跑 q50、q80 與固定 bp challenger（回答 1a／1b／5）

- [ ] 把 `dynamic_estimated_path_portfolio.py:110` 的 `frozen to q95` 檢查改成接受 `boundary_policy ∈ {q50, q80, q95, fixed15, fixed20, fixed25, fixed30}`；上游 `one_second_makerfill_runner.attach_q95_boundaries` 同步泛化為 `attach_boundaries(policy)`。
- [ ] 固定 bp 的定義：`upper = anchor + W`、`lower = anchor − W`，對稱；W 同時用於 frozen lower。不做每商品 tick 對齊以外的任何調整。
- [ ] 每個 policy 各自跑完整鏈：message load → makerFill → hedge → path → cap 回放。輸出根目錄命名 `*_causal_<policy>_<date>`。
- [ ] 同一張 raw order 若被多個 policy 命中，保留 `physical_order_id` 共用，跨 policy 不可相加（沿用 PILOT_RESULTS 的 alias 規則）。
- [ ] 產出一張 policy 比較表：candidates、fill 率、cancel 率、hedge slip p50/p95、同日／跨日／expiry 比例、uncapped net bp、20M cap realized net、日均新 spot。
- [ ] 加一欄「q target 落在 B1／B2／B3+ 的 product-seconds 比例」，讓 fixed bp 與 q 的機會數差異看得見。

完成判準：七個 policy 的比較表在同一份 md，並附各自的 `complete.json` SHA。

### P2　把 Future Ask maker → Spot taker route 加回因果線（回答 2a 期現分開）

- [ ] `one_second_makerfill_runner` 目前只 join 現貨 `Bid1/Bid2_FillSeconds`；期貨 maker 沒有等價的 legacy makerFill label，要沿用 fixed-45 階段的 indexed replay（`execution_runner` + `indexed_replay`）產 exact fill，但輸入改為 manifest 的 `(Date, ValueCode)`，**先移除 `sessions × symbols` 笛卡兒積介面**。
- [ ] Hedge 方向反過來：fill + 50 ms 買現貨 2 lots，L1–L5 VWAP；期貨一口沒有 partial。
- [ ] 期貨 5 requests/s：期貨 maker quote 與 fill hedge 共用同一個 limiter，fill hedge 優先；message load 表要重算。
- [ ] 輸出與 P1 相同欄位，並在比較表加 route 維度。兩 route 的 sampling contract 不同（spot snapshot vs anchor 觸發），**只並列不 pooling**。

完成判準：兩條 entry route 在同一 manifest、同一 policy 下的 fill／cancel／hedge slip／path 表。

### P3　Exit maker + exit hedge（回答 2a 進出分開）

- [ ] 對 P1／P2 的每筆 entry position，在 frozen lower 掛兩條 exit maker route：`Future Bid maker → Sell Spot taker`、`Spot Ask maker → Buy Future taker`。取樣、分層、後撤規則與 entry 相同。
- [ ] Exit hedge 滑價用與 entry 相同的 arrival／decision 算法，四條 route 各出一張 slip 表。
- [ ] Exit maker 未成交者的 competing outcome：同日 taker/taker（現行 1 Hz first passage 當 control）、carry、expiry proxy。三者互斥。
- [ ] FIFO inventory：同商品多個 position 的 exit order 要合成「商品 × 絕對價」的實際 working order，不能逐 position 各掛一張再相加（ORDER_MESSAGE_LOAD 已指出這會嚴重高估）。
- [ ] 先只做 spot-entry route 的 exit，跑通後再接 future-entry route。

完成判準：每筆 position 有 `exit_route / exit_fill_time / exit_hedge_slip_bp / exit_outcome` 四欄，cap 回放改吃這組 terminal。

### P4　Hedge fail-safe 與 makerFill 精度

- [ ] 213 筆 gate closed：加 retry 規則（每秒重查 decision book，最多 N 秒；超時以現貨 taker 平掉 spot），規則在 P1 之前先凍結寫進本文件，不看結果調。
- [ ] Book freshness sensitivity：decision book age > 1 秒的 hedge 分別以「用該舊 book」與「標 unpriced」兩種處理各出一版數字。
- [ ] makerFill exact 校準：從 manifest 抽 5 個固定日，用 indexed replay 算 exact own-quantity fill，對 approximate 2.37% 給正式修正係數與信賴區間，取代舊 5 日 fixed-45 的 1.88→1.72% 粗估。

### P5　8 月惡化歸因 + prospective holdout

- [ ] 對 May–Aug 逐月畫：實際 basis excursion 分布 vs D-1 q80／q95 界線位置；分開報「界線碰到率」與「碰到後 fill 率」。若界線碰到率沒掉而 fill 率掉，是 queue／競爭問題；反之是 boundary 掛太深。
- [ ] 8/14 之後的新資料一律不進任何門檻選擇；P1–P4 規則凍結後，第一個完整月（2026-09）作 prospective holdout。
- [ ] Holdout 報表與 development 報表格式完全相同，只多一欄 `holdout=true`。

### P6　工程與文件（可隨時做，不阻塞）

- [ ] `.gitignore` 第 22 行 `src/research/*/` 改為只排除 `src/research/*/data/`，把 `src/` 與 `doc/` 納入版控。（2026-08-24 已提出，待使用者決定）
- [ ] `data/_trash_20260824/` 確認後刪除；另外兩個大目錄 `exit_maker_narrow_60d`（14G）與 `..._candidate_session_cache_v8`（7.4G）是 fixed-45 時代的 exit maker 結果，P3 出來後即可刪。
- [ ] `doc/quote_fill/README.md` 索引補上 8/22 四份文件，並把 fixed-45 時代的結果統一標 `archive`。
- [ ] 頂層 `README.md` 改寫成「動態池因果線」的入口，移除「131 日 A1-B1 screen」作為主線的描述。

## 目前 data/ 的分層

| 層 | 目錄 | 大小 | 處置 |
|---|---|---:|---|
| 基礎事實（所有 run 的輸入） | `walkforward/daily`、`rolling_boundaries`、`liquidity`、`sessions.txt`、`exact_contract_calendar_v1.parquet`、`expiry_daily_close_facts_20260821_v1` | 21G | 保留 |
| 因果線 canonical（8/22） | `monthly_product_selector_causal_v2_20260822`、`order_message_load_causal_v2_20260822_v2`、`one_second_makerfill_causal_v2_20260822_v1`、`dynamic_future_hedge_causal_v1_20260822`、`dynamic_estimated_path_portfolio_causal_v1_20260822`、`dynamic_expiry_paired_close_facts_20260822_v1`、`august_exit_extension_causal_v1_20260822` | ~230M | 保留；P1 完成後成為 q95 baseline |
| fixed-45 archive（有 leakage，只供對照與 P2／P3 借程式） | `execution_narrow_60d`、`exit_maker_narrow_60d`、`exit_maker_cross_session_narrow_60d` + `_candidate_session_cache_v8`、`overnight_carry_*`、`post_cross_*`、`prequential_*`、`aggressive_1300_*`、`compact_*`、`makerfill_rank_*_v5`、`future_ask_rank_*`、`normal_carry_cap_sweep_*_v2`、`portfolio_cap_completed_only_*`、`cross_session_prerequisites_*`、`d_safe_universe_audit_*`、`current_cohort_preopen_*`、`supplemental_sampling_*`、`exit_maker_interim_*`、`compact_remaining_time_*` | ~22G | 保留到 P3 完成 |
| 已搬走 | `data/_trash_20260824/` | 19G | 確認後 `rm -rf` |
| 八日 pilot | `fair_mid/`、`quote_fill/`、`quote_width/` | 110M | 保留（WP01 結果與 8 日 raw pilot） |
