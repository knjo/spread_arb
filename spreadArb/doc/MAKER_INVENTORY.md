# maker 既有資產對照：定義、程式、結果、教訓 → 重用決策

路徑相對於 `../maker/`。「重用」= 邏輯搬入 spreadArb 並加測試；「對照」= 只拿來對帳，不搬；「棄用」= 不再引用。
maker 目錄本身不再修改。

## 1. maker 的兩條線與四份複本

| 線 | 位置 | 狀態 | 決策 |
|---|---|---|---|
| 因果 pipeline + cost-aware S1 奈秒事件迴圈 | `src/quote_fill/s1_*`、`doc/REWORK_PLAN_20260824.md`、`S1_COST_AWARE_IMPLEMENTATION_20260831.md` | 365 項回歸通過、無 71 日績效；使用者 9/1 判定過度工程 | **棄用**程式；保留其凍結規格（B5/B6/C8/C9、13:19:45 drain、exit headroom guard）作口徑依據 |
| Q 表 → EV 線 | `src/ev_lookup/`（v17–v21）→ `ev_lookup_xc/`（全腳成本）→ `ev_lookup_cost/`（第一順位＋guard）→ `ev_lookup_target/`（Q 重製、30% 目標、事件 S2） | 現行主線；四份套件互相 import／subclass | **重用**邏輯（下表逐項），**不重用**套件結構 |
| S0.5 基礎（anchor／q lookup／mother） | `src/quote_fill/foundation_*`、`doc/quote_fill/FOUNDATION_SELECTION_S05_REBUILD_20260826.md` | 完成、canonical | 對照（anchor 15s 選擇依據）；q95 邊界只作 S1 對照 |
| 固定 45 檔時代 | `doc/quote_fill/archive_fixed45/` | 有 universe leakage | 棄用；只留「AB1/2 決策」與「q95 challenger 出處」 |

## 2. 定義與程式：逐項對照

| 概念 | maker 定義位置 | 程式 | 決策 → spreadArb 位置 |
|---|---|---|---|
| Basis 三種、目標報價式、route 四條 | `doc/00_SCOPE.md` | — | 重用 → `00 §1–2` |
| 資料契約、gate、feature 禁用 | `doc/05_DATA_CONTRACT.md` | `src/common/{paths,contracts}.py` | 重用 → `src/common/`（去掉 legacy fallback 與 S1 role-bound manifest 的複雜度，保留 SHA 記錄） |
| Book 讀取、tick 規則、anchor 120s、S2 訊號格 | — | `ev_lookup_cost/market.py`（`BookSeries`、`_raw`、`_anchors`、`_signals`、`tick_i/previous_tick`） | 重用 → `src/common/market.py`；`_signals` 的每秒 S2 訊號改為事件驅動（Stage 2） |
| 期貨基本資料快取、近月標準合約選取 | — | `common/contracts.py::select_near_standard_contracts`、`data/ev_lookup_v19_metadata_20260908/` | 重用；快取直接讀 |
| 預告日曆、公司行動 | `EV_LOOKUP_V21 §EV 與出場政策` | `ev_lookup_cost/{forecast_calendar,corporate_actions}.py` | 重用原樣 |
| Maker 成交 FIFO、撤單、hedge 深度、四腿 PnL | `EV_LOOKUP_V21`、`V22` | `ev_lookup_cost/execution.py`（`PrintedVolumeQueue`、`TakerDepth`、`CapacityLedger`、`realized_pnl`） | 重用原樣＋補測試 → `src/common/execution.py` |
| S2 掛單／撤單／requote／hedge 完成即重評 | `V22 §本輪判讀`、`S2_ENTRY_REASSESSMENT`、`EVENT_S2_Q_PROFIT §固定執行規則` | `ev_lookup_cost/portfolio.py`（`book_update`、`_requote_s2`、`submit`、`trade`、`hedge`）、`ev_lookup_target/audit/event_s2_policy.py` | 重用規則 → Stage 2 S2 狀態機（改寫，不 subclass） |
| S2 深度／穿價成本 | `V22 §事先固定的比較` | `ev_lookup_cost/s2_liquidity.py`、`liquidity_portfolio.py` | 重用公式（`raw/depth/adverse basis`、`execution_buffer_bp`） |
| S1 進場來源 | `EV_LOOKUP_HANDOVER §1 S0` | `ev_lookup_cost/replay.py::s1_commands` 讀 `walkforward/august_attribution_s0_20260824_v2/raw_order_facts.parquet`（q95 BID1/2，≤ 8/13） | **棄用**作來源；Stage 2 自產 S1；q95 作對照 |
| S1 深掛單保留 | `TARGET30 §先固定的執行比較` | `ev_lookup_target/audit/deep_policy.py` | 重用作 Stage 3 容量政策參數 |
| 出場 E1、guard、13:18 撤、09:05 起 | `V21`、`EV_EXECUTION_COST_REVIEW` guard 組 | `portfolio.py::second`（exit 段）、`_guard_exit_orders` | 重用規則 → Stage 2 E1；E2 新增 |
| Taker cross 規則 | `HANDOVER §3.5` | `ev_rules.should_cross`、`portfolio.py`（`enable_cross`，主比較關閉） | 重用純函數；Stage 3 才決定是否啟用 |
| EV 分支式（P_sd／P_on／P_exp／other） | `V21 §EV 與出場政策`、`EV_EXECUTION_COST_REVIEW §查 Q 表` | `ev_lookup_cost/{exit_model,decide}.py::policy_ev_xc`、`ev_lookup_target/q_model.py::estimate` | 重用 → `src/ev/ev.py`（Stage 1） |
| Q 表 key／桶／回退／冷啟先驗 | `TARGET30 §研究問題與口徑` | `ev_lookup_target/q_model.py::keys`、`causal_lookup.py::{time,spread,exit}_bucket`、`exit_model.py::expiry_bucket` | 重用 → `src/ev/qtable.py`（宣告式） |
| Q 事實表（quotes／risk／costs） | `ev_lookup_target/README.md` | `ev_lookup_target/q_facts.py` | 重用 → `src/ev/facts.py`；來源改為 Stage 2 點位表 |
| 執行衰減表 d_in／d_sd／d_on | `EV_EXECUTION_COST_REVIEW §重新量測` | `ev_lookup_cost/cost_observations.py`、`causal_lookup.py::DailySnapshot.{entry,exit}_decay` | 重用定義 → `00 §6` |
| λ、bpday 格閘門、L*、est_short | `HANDOVER §3` | `ev_lookup_cost/ev_rules.py` | 重用純函數（bpday 預設關閉） |
| 30% 目標換算、資金成本、hurdle | `TARGET30 §目標口徑` | `ev_lookup_target/target.py` | 重用 |
| 釋放率模型（release credit） | `TARGET30 §存續 carry 的釋放校準` | `ev_lookup_target/release_model.py` | 重用作 Stage 3 政策參數 |
| 容量政策（reserved／shared／parked） | `TARGET30 §先固定的執行比較` | `ev_lookup_cost/capacity_policy.py`、`ev_lookup_target/audit/unrestricted_study.py` | 重用為參數 |
| 期末評價（官方／BBO） | `V22 §評價方式及缺值` | `ev_lookup_cost/valuation.py` | 重用 |
| 費稅精確帳 | `doc/04_BACKTEST.md §帳本與報告` | `quote_fill/transaction_costs.py` | 重用（Stage 3 精確帳；EV 用 20/34 近似） |
| 多層掛單、epoch 取樣、order lifecycle | `doc/quote_fill/REPLAY_SAMPLING.md` | `quote_fill/{layered,order_identity,replay}.py` | 重用**契約**（S1 多 U 格點）；程式不搬 |
| 50 ms hedge、B6 retry、partial 政策 | `doc/03_HEDGE_COST.md` | `quote_fill/{hedge,partial}.py` | 重用契約（`t0 = fill + 50ms`、5 s deadline）；ev_lookup 的實作已涵蓋 |
| 逐日 checkpoint／resume、獨立核驗 | `V22 §程式與輸出`、`EVENT_S2 §固定執行規則與全期核驗` | `ev_lookup_target/audit/{run_daily,verify_execution,capacity,equivalence}.py` | 重用**做法**（每日重啟、hash 綁定、bootstrap 重建）；程式重寫精簡 |
| 訊息負荷量測 | `ORDER_MESSAGE_LOAD_CAUSAL_1HZ_20260822.md` | `ev_lookup_target/audit/messages.py` | 重用 |

## 3. 可對帳的數字基準（全部 20M、5/4～9/2、85 資料日；皆為歷史回放，非樣本外）

| 項目 | 數字 | 出處 |
|---|---|---|
| 現行最佳（立即補 S2＋歷史增額，扣 2% 資金成本） | 85 日 +8,930／日；暖機後 65 日 **+11,678／日**；S2 成交 1,867；股票本金峰值 33.64M；規劃年化 14.6%（20M 分母）／8.68%（峰值分母） | `EVENT_S2_Q_PROFIT_20260914` |
| 同上，基本 20M | +8,105／+10,599；峰值 28.32M | 同上 |
| 舊時鐘（60 s CD）對照 | +4,658／+6,091 | 同上 |
| cost_guard（每秒時鐘、EV only） | +8,742／日（未扣資金）；+7,939（扣 2%） | `EV_EXECUTION_COST_REVIEW` |
| v22 最佳（深度 5×＋50% 一 tick） | +3,919.50／日 | `EV_LOOKUP_V22` |
| S2 供給四日（25 bp、深度 5×） | 每秒+60 s CD 129 → 事件+60 s 219.5 → 事件+hedge 完成 **463.5** 筆／日；深度 1× 636；0 bp 4,311（premium 6 bp） | `S2_ENTRY_REASSESSMENT` |
| S2 EV 桶事後淨 bp（guard shadow） | `<0: −0.50；0–10: +10.52；10–25: +21.08；25–50: +28.69；≥50: +47.44` | `EV_EXECUTION_COST_REVIEW` |
| S2 hedge 變差 | 19.94%；超一 tick 6.65%；均 +5.49 bp；變差時 +28.36 bp | 同上 |
| 執行衰減均值（guard） | S1 進 25.72、S2 進 8.92、S1 出 −14.45／−13.15、S2 出 −5.42／−6.33 | 同上 |
| 當沖率（成熟配對） | 當日 41.70%、至次 session 64.01%（新增額組） | `EVENT_S2_Q_PROFIT` |
| 持有時間 | 本金加權 2.77 日曆日 | 同上 |
| q95 S1 approximate fill | BID1 12.29%、BID2 1.08%、整體撤單前 2.37% | `ONE_SECOND_MAKERFILL_CAUSAL_V2` |
| 訊息峰值（事件版 S2） | 期貨 74／s、股票 182／s | `EVENT_S2_Q_PROFIT` |
| 官方日終權益缺值日 | 5/13、5/20、6/17、6/26、6/30、7/13–15、8/19、8/28 | `V22` |

## 4. 已否決的方向（附證據，不重走）

| 方向 | 證據 | 出處 |
|---|---|---|
| 進場 maker＋hedge 腿也 maker（A1−1） | 觸發後 2 s 期貨 bid 掉 22.5 bp；淨 −8～−12 bp | `HANDOVER §5` |
| 完全不撤單／大容忍帶 | stale fill 71% 有毒（effU −6 bp） | 同上 |
| 事前動能 gate 判別 chase | 無效（補漲在觸發後才開始） | 同上 |
| 日內額度放大（20M 交割下）、早盤 30M 衝 | 並發 gross 天然 ~28M；邊際單當沖辨識不足 → 負 EV | 同上 |
| 學習版逆價差先驗 | regime lag，輸給結構保底 | 同上 |
| 出場現貨排 A1 耐心 hedge | −1～−3 bp | 同上 |
| 到期保底當提款機 | 卡到到期的 ab 中位 30～40 bp ≈ 打平；C8 歸因常為負 | `HANDOVER §3.9`、`TARGET30` |
| bpday 格閘門 | 逐行情版 183 筆 +26 萬但最後新成交 6/5；每秒版為負 | `V21` |
| 前日 Q 提高門檻（30% hurdle）換高 bp | 每筆 bp 上升但總收入下降（934／日、359／日） | `TARGET30` |
| 取消單筆 2M 上限 | 10,142 → 4,605／日（S1 歸因轉負） | `TARGET30` |
| 固定 60 s 冷卻 | 成交數減半 | `S2_ENTRY_REASSESSMENT` |
| 奈秒級 fail-closed 事件迴圈 | 過度工程；實盤做不到微秒級 | `HANDOVER §1 S1` |
| 用受限投組成交數推論市場供給 | 影子 8,662 vs 投組 1,399 | `S2_ENTRY_REASSESSMENT` |

## 5. 方法論教訓（HANDOVER §0，沿用）

1. 有賺 ≠ 值得佔容量：分配鍵是 bp／容量日，容量是真正稀缺資源。
2. 截尾會計與到期結算要先建對：v11 33k → v12 18.8k 的差距 2/3 是 carry 不認損。
3. 分項帳 ≠ 邊際貢獻：S2 貢獻是同帳本內歸屬，不是 S2-only 回放。
4. 結構知識（到期保底、正價差 gate）勝過慢速經驗學習。
5. 候選 dump ＋ 秒級重放是政策迭代正解，但重放較真模擬器樂觀 ~17%，勝者必回真模擬器。
6. 查表正確與預測準確是兩個問題：因果核驗通過不代表 EV 校準；S1 高估、P_sd 高估、carry 釋放高估都要另列。
7. 時鐘決定供給：每秒 vs 事件、CD vs hedge 完成，差 2～3.6 倍成交；先修時鐘再放門檻。

## 6. maker 資料夾裡「仍在跑／未完成」的事，spreadArb 不接手

- cost-aware S1 的 497 partitions 正式 replay、clean-commit durable rerun。
- 30% 年化目標：九組皆未達標（最高 12.68%），立即補單版 14.6%；spreadArb 以 Stage 3 指標重新量，不承接其政策。
- 口數 P_fill(n)、每日庫存評價線、實單試掛：列入 Stage 3／4。
