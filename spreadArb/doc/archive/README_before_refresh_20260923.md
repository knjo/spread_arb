# spreadArb：Maker-based 期現價差套利（重製線）

本目錄是 [`../maker/`](../maker/README.md) 研究線的**重製**：保留 maker 已驗證的定義、口徑與教訓，
把散在四份 `ev_lookup*` 複本與 `quote_fill/s1_*` 奈秒模擬器裡的邏輯，收斂成一條可單元測試、可分段驗收的 pipeline。
maker 目錄不再新增功能，只作定義與數字的對照來源；對照表見 [`doc/MAKER_INVENTORY.md`](doc/MAKER_INVENTORY.md)。


**現行回測（9/22 修正）**：`backtest.replay` 已改用 raw 逐事件撮合；A／B 同時執行用 `backtest.causal_replay`。
原 `A_fixed`／`B_dyn` 的 34.9%／37.8% 是有執行缺陷的歷史結果，不能當作已驗證報酬。
目前架構、成本／結算假設、完整執行與驗收指令以 [04_CORRECTED_REPLAY.md](doc/04_CORRECTED_REPLAY.md) 為準；
**9/23 容量規則更正**：使用者允許未成交掛單不占部位，只能依已發生的進出更新額度。
9/22 完整重跑的 A **13.5978%**／B **13.4869%** 是「全部掛單全額預留」變體，雖通過模型內帳務驗收，
不能直接當作使用者策略的修正後年化，61%／64% 差額也不能全歸因於 bug。
重新驗證未重現提前釋放未來平倉額度；詳見 [容量規則與前次結論更正](doc/05_CAPITAL_POLICY_CLARIFICATION.md)。

**9/23 重新驗證結果**：現行不預留模式尚缺 S1 部分成交即時占用、成交後額度不足掛單的撤單處理。
51 日輸出損益可對帳，EV 算術可重算，但不代表策略與預測已驗收；詳見 [重新驗證報告](doc/06_REVALIDATION_20260923.md)。

## 總目標

1. **Maker-based 的「價差的價差」套利**：以 basis 相對 causal anchor 的殘差（`effU = ab − anchor`）為主要 edge，
   一腿 maker 掛單、成交後另一腿 +50 ms taker hedge，建立 Long Spot + Short Future，於 basis 回落時同結構平倉。
   若**絕對價差** `ab` 本身已足以覆蓋持有到期的成本（到期 basis 歸 0），也做；EV 公式同時含這兩種收益來源。
2. **由 Q 表回推 EV**：每個掛單決策 = 查表 + 一條期望值式。Q 表只用決策日以前的已完成觀測（walk-forward），
   EV 決定「掛不掛、掛哪個價差、撤不撤、出場掛哪」，不用 if-else 規則堆疊。
3. **逐步把執行成本放進 EV**：quote→hedge 衰減、出場 shortfall、穿價、資金成本、容量機會成本，先估後校準，
   再微調掛單與撤單。

## S1／S2 為什麼要分開

| Stream | Maker 腿 | +50 ms Taker hedge | 掛價由誰決定 | 撤單由誰觸發 | 成交由誰判定 |
|---|---|---|---|---|---|
| **S1** | 現貨 Bid | 賣期貨（Fut executable Bid） | 現行 A／B 掛現貨 B1，再用 Fut Bid 計算該掛價的 basis／EV | 期貨 Bid 下移使鎖定 basis 跌破地板 | 現貨成交 print |
| **S2** | 期貨 Ask | 買現貨（Spot A1） | 期貨 A1 − 1 tick（隊列第一） | 現貨 A1 上移使鎖定 basis 跌破地板 | 期貨成交 print |

兩邊都是「看另一腿決定掛價與撤單、看自己腿決定成交」，但觸發來源相反。撤單 50 ms 才生效，
「另一腿先動（撤單意圖）」與「自己腿先成交」誰先發生決定了這筆是乾淨成交、撤單競賽成交、還是撤掉。
只有把兩個 stream 各自放在合併後的因果事件流上獨立重放，才能正確算出時間先後；不能共用一個「每秒看一次」的時鐘。
出場同理拆成 **E1**（現貨 Ask maker → 買期貨 taker）與 **E2**（期貨 Bid maker → 賣現貨 taker）。

## 架構與分段

- Stage 0：定義與資料契約凍結 → [`doc/00_DEFINITIONS.md`](doc/00_DEFINITIONS.md)
- Stage 1：Q 表（水位／到達／滑價三表）+ EV／T 選出場水位 → [`doc/01_Q_TABLE_EV.md`](doc/01_Q_TABLE_EV.md)
- Stage 2：S1／S2 雙向點位表（進場側＋出場側）→ [`doc/02_ENTRY_EXIT_POINTS.md`](doc/02_ENTRY_EXIT_POINTS.md)
- Stage 3：回測架構（容量、carry、到期、指標）→ [`doc/03_BACKTEST.md`](doc/03_BACKTEST.md)
- Stage 4：成本校準與掛單微調 → [`doc/04_SLIPPAGE_SCREEN.md`](doc/04_SLIPPAGE_SCREEN.md)（第一份：哪些狀態下不要掛）
- 部署：[`doc/10_PREMARKET.md`](doc/10_PREMARKET.md)（盤前檔規格，Python）、[`doc/11_TRADING.md`](doc/11_TRADING.md)（交易邏輯與每日時序；凍結政策 A／B）

| Stage | 交付 | 驗收 | 狀態 |
|---|---|---|---|
| 0 | 本文＋`doc/00`＋`doc/MAKER_INVENTORY` | 使用者確認開放決策（見下） | **文件完成，待確認** |
| 1 | `src/ev/`：`qlevel.py`、`reach.py`（市場 1 Hz 格）、`config.py`、`ev.py`＋`build`／`surface` CLI＋28 測試 | 29 項測試通過；全期表（131 session）與 as-of 表、距到期剖面、CI、EV 面已出（[`doc/01_RESULTS.md`](doc/01_RESULTS.md)） | **凍結（9/17）＋絕對價差路線（9/18，taker 表概估；[`doc/01 §4b`](doc/01_Q_TABLE_EV.md)）；待 Stage 3 校準** |
| 2 | `src/points/`：S1／S2／E1／E2 點位事實表＋hedge 可執行性 | 四個固定日（5/11、6/1、7/6、8/3）S2 成交數與 maker S2 供給分解可對帳 | **完成（9/20）：雙向 S1／S2 全年 131 session（S1 67.6M 列／S2 59.4M 列，8 GB）；S2 四日 1,907 vs maker 1,854；S1 同批指令成交時間 97% 在 1 秒內、makerFill 中位差 0.01 s（[`doc/02_RESULTS.md`](doc/02_RESULTS.md)）** |
| 3 | raw 逐事件撮合、資金 ledger、每日官方估值、A/B 比較 | cash legs、FIFO 成交量、shared depth、完整訊號母體、同合約結算與逐日權益獨立對帳 | **9/22 修正引擎已取代點位回放**；歷史數字與修正版結果、驗證狀態見 [03_RESULTS](doc/03_RESULTS.md)，執行方式見 [04_CORRECTED_REPLAY](doc/04_CORRECTED_REPLAY.md) |

**9/22 原版 A／B 獨立驗證（歷史紀錄）**：各 131 日完整重跑與原產出逐筆一致；全期帳面損益 A 3,659,255.726／B 3,963,963.450，
250 日單利年化 34.9166%／37.8241%。4–7 月原報表漏扣 rollback，年化應為 **44.4723%／49.2183%**。
另查得 A 29／B 36 筆跨月份合約平倉，以及到期缺價留倉、容量預留使用未來成交結果等問題，**尚未通過回測正確性驗收**。
架構、使用方式、逐筆證據與成本敏感度見 [`doc/03_RESULTS.md` 的 A／B 獨立驗證](doc/03_RESULTS.md)；可重跑 `spreadArb.src.backtest.audit`。
後續邏輯驗證另確認 S1 生效前成交量被計入、E2 同價後續出場缺漏、出場候選提早截止、快取穿透負 anchor／e、
過寬 scale 排除失效等問題；`backtest.logic_audit` 提供反例，`backtest.raw_audit` 提供四固定日原始 tick 證據。
完整逐條規格對照保留在同一結果檔；修正前 67 項單元測試全過並不代表完整回測正確。

每個 Stage 的細節、輸入輸出、測試清單與「maker 已做過什麼」都在對應 doc；README 只維護狀態表與開放決策。

## 目錄規劃

```text
spreadArb/
├── README.md                 # 本文：目標、分段、狀態、開放決策
├── doc/
│   ├── 00_DEFINITIONS.md     # 全部名詞、公式、成本、資料契約（唯一定義來源）
│   ├── 01_Q_TABLE_EV.md      # Stage 1 規格
│   ├── 01_RESULTS.md         # Stage 1 結果紀錄（逐輪追加）
│   ├── 02_ENTRY_EXIT_POINTS.md
│   ├── 02_RESULTS.md         # Stage 2 結果紀錄（逐輪追加）
│   ├── 03_BACKTEST.md
│   └── MAKER_INVENTORY.md    # maker 既有定義／程式／結果／教訓 → 重用決策
├── src/
│   ├── common/               # paths、contract、calendar、tick 規則、book 讀取（自 maker 精簡搬入）
│   ├── ev/                   # Stage 1
│   ├── points/               # Stage 2
│   ├── backtest/             # Stage 3
│   └── tests/
└── data/                     # 不進 git；每個 run 一個資料夾，附 manifest 與來源 hash
```

原則：**一個套件、一個 Portfolio、參數化變體**；不再以複製套件或 subclass 疊加實驗。
每個 Stage 只留一份規格文件與一份結果紀錄（附 run 目錄與 hash），不再產生逐日的 dated 報告。

## 開放決策（Stage 1 開工前需使用者定案）

1. **Anchor**：maker `ev_lookup` 線用 `EWMA 120s`（Q 表以此訓練）；S0.5 基礎研究另選過 `time_ewma_15s`（MAE 8.72 bp）。
   建議：anchor 作為可插拔參數，Stage 1 先沿用 120s 以便與 maker 數字對帳，Stage 2 起把 15s 當第一個對照。
2. **S1 掛價選擇**：maker 的 S1 從未由 EV 自己產生（讀 S0 q95 外生指令，只到 8/13）。
   建議：S1 改為與 S2 對稱——對一組合法 U 格點算 `P_fill(U) × EV(U)` 取 argmax；q95 邊界只作對照。
3. **出場路線**：maker 只實作 E1（現貨 Ask maker）。是否在 Stage 2 就加 E2（期貨 Bid maker）？建議加，但先只出事實表，不進回測主比較。
4. ~~出場目標 L~~ → 已定案（2026-09-17）：出場水位是 Q 表的決策變數，見 `doc/01 §4`；`anchor − 5` 只作對帳格點。
5. **口數**：maker 全程 1 口／fill。建議 Stage 2 事實表記錄掛單當下可見深度與 P_fill(n) 所需欄位，Stage 3 先跑 1 口，再開 n 口。
6. **資料區間**：maker 全期 2026-05-04～09-02（86 session、85 資料日、8/28 中斷）。S1 自產後不再受 8/13 限制。
   建議沿用同區間以便對帳；擴充區間放 Stage 3 之後。

## 硬性邊界（沿用 maker，不再討論）

- 決策日 D 只能用嚴格 `< D` 的已完成觀測；同日結果不入表；預告日曆只能用公告日以後的休市。
- 撤單意圖到生效 50 ms；maker 成交後 +50 ms 才第一次 hedge；hedge 用 L1–L5 足量 VWAP，不足就等並保留曝險。
- 印出的成交量只能被消耗一次；同一 timestamp 的 print 不能成交剛送出的單；撤單不能贈與隊列優先權。
- 現貨當沖 20 bp、隔夜 34 bp；到期後 C8：兩腿同結算價、basis = 0。
- 不利成交、撤單競賽成交、負 basis 成交全部保留入帳，不得事後刪除。
- 研究結果是「查表、時序、容量經核對的歷史回放」，不是未見資料驗證，不宣稱實盤收益。
