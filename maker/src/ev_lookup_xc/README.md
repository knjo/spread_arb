# EV lookup research

## v22：S2 第一順位、現貨深度與穿價成本

使用者指定維持期貨第一順位，不夠就等。`liquidity_study.py` 比較 inside 控制、
現貨 A1 至少對沖股數 5 倍、固定 50% 一個現貨 tick 的 EV 扣減，以及兩者合併。
掛前與掛後均看可觀測深度，只有張數變動也重新檢查；撤單生效前的成交保留必要 hedge。
四組各有獨立 shadow，20M 全額預留，普通出場與實現現金流沿用 v21。
完整回放執行中，規格及狀態見 [v22 深度／成本研究](../../doc/quote_fill/EV_LOOKUP_V22_LIQUIDITY_20260909.md)。

## v21 / historical v19–v20

2026-09-09 的目前修改為 v21：進場 EV 區分當沖、一般隔夜、真正到期及其他已觀測退出；
S2 在兩市場原始盤口更新時檢查價差與 EV，撤單生效後可向後改掛。
實作、完整對照及限制見 [v21 EV 與撤掛修正](../../doc/quote_fill/EV_LOOKUP_V21_20260909.md)。
新入口為 `ev_event_study.py`；下文原 v19/v20 runner 明確保留舊 EV 與每秒撤掛，供歷史對照。

```bash
POLARS_MAX_THREADS=8 UV_CACHE_DIR=/tmp/hft-uv-cache uv run python -m \
  src.research.futures_spot_spread.maker.src.ev_lookup.ev_event_study \
  /tmp/ev_lookup_v21_full
```

`--modes second` / `--modes event` 可分開執行，各自有獨立的 shadow、快照與 checkpoint。
兩個 mode 各比較舊 EV、新 EV、新 EV＋前日 bpday；均完整預留 20M，hedge/cancel 各 50ms。
`exit_model.py` 使用前 20 個完整 session 的 carry 在險樣本，未平倉者留在分母；
到期比例是穿過剩餘交易日後仍未退出的機率，沒有把其他平倉類型算成到期。
`forecast_calendar.py` 將事前公布的假日與臨時公告分開；7/10 颱風休市不能在 7/9 公告前進入 EV。
預測涵蓋至 9/16 的原合約到期日，與決定實際回放日期的事後日曆分開。
不足 30 筆 exposure 時向較粗桶退回，仍不足則用事先固定的每日正常退出機率 0.5。
這是暖機假設，不是樣本估計；到期當日的非當沖分支保留 C8 歸零估值。
一般出場仍固定 frozen anchor−5；午後 taker-cross 實驗預設關閉。

## 歷史版本與共用執行契約

v19 修正 v17/v18 的當日結果洩漏、預知 carry 退出、成交後拒單，以及出場標籤／損益問題。
v20 延續這些修正並研究容量使用；策略仍每秒決策一次，遠端現貨掛單另在原始 B1 更新時檢查額度。
這是研究回放，沒有接券商下單。

**v17 的 20M 約 27,950 元／日是 S1 + S2 合計，不能再當作可交易績效。**
稽核證據見 [原版稽核](../../doc/quote_fill/EV_LOOKUP_AUDIT_20260908.md)，
修正與驗證結果見 [v19 報告](../../doc/quote_fill/EV_LOOKUP_V19_20260908.md)。
完整期間及容量比較見 [v20 報告](../../doc/quote_fill/EV_LOOKUP_V20_FULL_20260908.md)。
原日均約 2.8 萬與新版數百元的價格、選單與帳務差異見
[損益差異核對](../../doc/quote_fill/EV_LOOKUP_PNL_ATTRIBUTION_20260909.md)；v20 不是只修 carry 的同策略比較。
進場價差與 50ms 的拆分、inside 第一順位及出場 EV 分支不一致，另見
[執行時序與 EV 核對](../../doc/quote_fill/EV_LOOKUP_EXECUTION_EV_20260909.md)。
原 v18 主程式及其兩個決策依賴保存在 `archive_v18/`，可重現舊結果。

| 模組 | 責任 |
|---|---|
| `ev_rules.py` | 原 EV、成本、bpday 公式；其中 L*、cross 等研究函數尚未接入 |
| `causal_lookup.py` | 按可觀測時間記錄結果，建立不可修改的開盤快照 |
| `decide.py` | 掛單前 positive basis → EV → 前日 bpday → 容量准入；成交後不再呼叫 |
| `execution.py` | 整數容量、逐筆成交量共用 FIFO、五檔 taker 深度、四腿實價損益 |
| `market.py` | 逐日合約及漲跌停資料、原合約 carry、原始 as-of 盤口及 1 Hz anchor |
| `portfolio.py` | 雙路 maker、必做 hedge、部分成交、出場撤單與跨日帳本 |
| `replay.py` | 事件排序、獨立 shadow 池、前日統計、逐日 trace |
| `stacked_walkforward_backtest.py` | v19 CLI |
| `test_v19.py` / `verify_run.py` | 行為回歸與獨立原始成交／帳本／快照核驗 |

## 2026-09-09 另一份版本：P_nx 與 taker cross

較早的 [EV_LOOKUP_EXIT_EV_20260909.md](../../doc/quote_fill/EV_LOOKUP_EXIT_EV_20260909.md)
記錄較早 P_nx / 午後 cross 版本的獨立全期結果，來源為該報告自己的 source_snapshot。
目前工作樹的 v21 已用保留未退出 carry 的存活機率取代其
completed-only P_nx；`should_cross` 保留在 `enable_cross=True` 開關之後，正式比較關閉。
事件檢查已擴充至兩市場的有效最佳買賣價與盤面有效性。`choose_L` 仍未接。

## 決策與執行契約

- `D` 日的 bpday 僅使用**前 20 個已完成 session 內解決、且開盤前已知**的 shadow 結果。
  分桶用掛單時的 basis；當天同日退出、當天 carry 退出及到期入帳，都從下一個 session 才入表。
  P_sd 用已結束進場日的結果；lambda 用前五個完成日、被容量拒絕且 shadow 後來確實成交的機會。
- 兩路均在掛單前查表。S1 沿用 S0 全部 nominal new/cancel 指令與 D-1 q95 掛價；不讀成交／成功標籤。
  S2 是當時近月原合約 A1 前一合法 tick，U=25、維持地板 20bp、成交後冷卻 60 秒。
  未成交時的非正 basis 可撤單；已成交後的非正 basis 必須保留。
- CAP 定義為現貨腿進場名目額度，加上所有 pending 預留額度，**不是兩腿名目加總、保證金或市值上限**。
  S1 按限價預留；S2 按現貨當日公布漲停價預留，hedge 後改成實際買進金額。
  當沖、carry、未 hedge 及 pending 共用此 cap。沒有另設單商品 10M cap。
- 多商品與兩路必須共同預留；不能各自在外掛滿 20M 後才用成交先後拒單。
  成交按原始 receive timestamp 先後處理。同一 timestamp 的外部成交先於撤單、新掛單；新單不能吃到已觀測的同時戳成交。
  同秒新機會仍有固定排序：S1 指令先於 S2，S2 同秒按商品代碼排序，沒有聲稱能分辨無資訊的真實先後。
- S2 期貨 maker 成交後，50ms 起取第一個可執行、數量足夠的現貨 ask 深度；S1 完整現貨成交後則賣期貨 bid。
  不用後面更有利的價格；五檔不夠則等待後續有效盤口。超過五秒記 `hedge_timeout`，曝險及容量照留，不刪單。
  進場用的 -9%/+8% 緩衝不阻擋必要 hedge；現貨用當日公布漲跌停，標準股票期貨用參考價上下 10% 邊界。
  期貨範圍依據：[期交所股票期貨契約規格](https://www.taifex.com.tw/cht/2/sTF)。
- 出場在首次因果條件 `Future Ask / Spot Bid - 1 <= frozen anchor - 5bp` 出現時掛現貨 Ask。
  之後每秒用**自己的固定賣價與當時 Future Ask**檢查；條件失效或 hedge 盤口無效即撤。
  不跳過事後沒有 makerFill 標籤的掛單。13:18 撤剩餘 maker 單；已部分賣出者完成剩餘現貨 taker，再買回期貨。
  未成交出場單撤掉後保留 paired carry。兩腿都完成才記損益並釋放額度。
- 每筆公開成交量在同一個 portfolio 只用一次，包括前方隊列及所有自己的單；允許部分成交。
  S1 未滿一口期貨對應股數而撤單時，明確反向賣回已買現貨並計費，不假裝已做成完整 pair。
- 部位保存確切 `QuoteCode`、股數、到期日。商品退出當沖新倉名單後，既有 carry 仍載入原合約資料。
  到期仍沿用 C8「下一 session 兩腿同價、basis=0」**會計假設**，不是宣稱真的以同價平了兩腿。

## 執行

從 HFT 根目錄執行；輸出目錄必須尚未存在。預設 20M、EV + 前日 bpday、canonical 日期。

```bash
UV_CACHE_DIR=/tmp/hft-uv-cache uv run python -m src.research.futures_spot_spread.maker.src.ev_lookup.test_v19

UV_CACHE_DIR=/tmp/hft-uv-cache uv run python -m src.research.futures_spot_spread.maker.src.ev_lookup.stacked_walkforward_backtest \
  --days 20260720 20260721 20260722 20260723 20260724 20260727 \
  --caps 20000000 --policies ev_bpday ev fcfs \
  --output /tmp/ev_lookup_v19_run

UV_CACHE_DIR=/tmp/hft-uv-cache uv run python -m src.research.futures_spot_spread.maker.src.ev_lookup.verify_run \
  /tmp/ev_lookup_v19_run --raw-days 20260724
```

`--hedge-ms` 預設 50；`--cancel-ms` 預設 0（沿用 V0 即時觀測撤單近似，可做 50/100ms 敏感度）。
`--products` 僅供診斷。跨日測試應列出區間內所有交易 session，不能把離散測試日當連續績效。
canonical 後的日期直接從 raw book 建格，沒有沿用舊的 second `.last()` extension cache。
85 日逐日基本資料快取在 `maker/data/ev_lookup_v19_metadata_20260908/`；沒有快取時使用既有唯讀 loader。

輸出包括 source snapshot/hash、輸入路徑、開盤快照、每日 decisions / execution / ledger / positions parquet、
daily CSV 及獨立 `verification.json`。原始資料與 canonical bundle 不改寫。

## 結果限制

這個 FIFO 模型不把前方撤單當作自己的優先權，也不把單純 book cross 當成額外成交量；
相較實際撮合可能少填或較晚填。公開逐筆行情不能完整還原我方加入後的隊列／市場反應，
因此它是明確的成交近似，不是實盤真值或保證的損益下界。
S1/S2 共用量、實際 hedge 與出場撤單規則都已改變，v19 與 v18 的損益差不能全歸因於移除 leak。

原 v19 短段驗證從空倉及空表開始；完整期間研究另見下方 v20，不能混合兩者的結果。
daily CSV 同時列出已實現損益、期末 bid/ask 估值及官方收盤／結算價估值。
缺價保留 null，不假設庫存沒有損益；期末庫存預留 34bp 隔夜往返成本。
官方日終價格只進入評價，不提供給進場／出場／容量決策。融資成本另做敏感度。

`stacked_walkforward_backtest_v17.py`、`candidate_dump.py`、`policy_replay.py`、
`reservation_experiment.py`、`ext_grid_builder.py` 保留為歷史研究，**沒有升級為 v19**。
它們的標籤、查表與執行近似不能混入新結果。

## v20 完整期間與容量實驗

`full_study.py` 依已保存的實際交易日曆回放 2026-05-04 至 2026-09-02：
86 個交易 session，其中原研究只有 85 日 raw；8/28 的期貨檔只有 4 bytes，
明確當成資料中斷日，停止執行、保留部位與容量，日終估值另取官方資料。
S1 原始 nominal 指令只到 8/13，之後只有 S2 新進場；這個資料限制不得省略。
所有完整實驗固定 hedge/cancel 各 50ms，起始空倉、空歷史表。

- 固定 20M：EV + 前日 bpday、EV、FCFS。
- S1 遠端掛單：低於 B1 可先排隊不占用預留；原始 B1 更新時檢查，接近 B1 預留或撤單。
  同時到達的預留需求按原始掛單到達順序處理。原指令仍有效時，因容量撤掉的單可重新掛、重新排隊。
  新股期 maker 一直需要事前預留。跳價或 50ms 撤單競態造成的無預留成交全部入帳，
  因而此分支的上限是**准入限制，不是實際部位不可能超過的保證**；超額金額與時間另列。
- 動態容量：20M 目標加上當前 paired 部位預計於當日释放的名目額度，研究上限分別 25M、30M。
  沒有 10:00 等指定切換時間。`capacity_policy.py` 使用前 20 個完成日的每日在險樣本，
  按 S1/S2、當日新倉/carry、當下時段估計存活部位的當日平倉比例；未平倉也納入分母。
  n < 30 不給額外容量，其餘使用 Wilson 下界作啟發式折減。同日樣本可能相關，這不是概率保證。
  額度帳本永不提早釋放既有部位；只改變可新增的預留上限。20M 日終目標不是硬性強平指令。
- `safe_capacity_study.py` 單獨比較保留全部預留、只使用上述動態上限的 25M/30M 版本，
  用來區分容量擴張與無預留排隊的效果；這兩個分支仍可嚴格限制最大含 pending 名目。
- `ev_capacity_study.py` 增加 EV-only 的現貨延後預留 20M、完整預留動態 30M、
  延後預留動態 30M，將前日 bpday 篩選與容量安排的影響分開。

完整回放前已逐一保存 29 份期交所契約調整公告（含停牌、配股、減資、現增等），
只在發文日**之後**的 session 啟用：停止該標的的新倉並以實際 taker 深度平掉既有 pair，
到調整生效日才允許新的標準合約。若舊合約身分失聯或未能在調整前平完，隔離持倉、保留容量，
不得把重新掛牌的同代碼當成原合約，也不得用 C8 直接消除隔離持倉。
這是 v20 額外的公司行動風控政策，與 v19 短段結果的來源版本分開保存。

```bash
UV_CACHE_DIR=/tmp/hft-uv-cache uv run python -m unittest \
  src.research.futures_spot_spread.maker.src.ev_lookup.test_v19 \
  src.research.futures_spot_spread.maker.src.ev_lookup.test_full_study

UV_CACHE_DIR=/tmp/hft-uv-cache uv run python -m src.research.futures_spot_spread.maker.src.ev_lookup.full_study \
  /tmp/ev_lookup_full
UV_CACHE_DIR=/tmp/hft-uv-cache uv run python -m src.research.futures_spot_spread.maker.src.ev_lookup.safe_capacity_study \
  /tmp/ev_lookup_safe_full
UV_CACHE_DIR=/tmp/hft-uv-cache uv run python -m src.research.futures_spot_spread.maker.src.ev_lookup.ev_capacity_study \
  /tmp/ev_lookup_ev_capacity_full

UV_CACHE_DIR=/tmp/hft-uv-cache uv run python -m src.research.futures_spot_spread.maker.src.ev_lookup.verify_full_study \
  /tmp/ev_lookup_full --raw-days 20260504 20260520 20260724 20260902
UV_CACHE_DIR=/tmp/hft-uv-cache uv run python -m src.research.futures_spot_spread.maker.src.ev_lookup.analyze_full_study \
  /tmp/ev_lookup_full
```

三個完整 runner 都每日保存 `checkpoint.json`，同一 runner 的 `--resume` 會核對來源 hash。
`verify_full_study.py` 從輸出跨日重建每筆額度、四腿損益、前日 bpday 與容量訓練樣本，
並將 parked 分支的真實超額与錯誤的主動超額送單分開核對。
`analyze_full_study.py` 產出每日時間加權 paired/pending 占用、每月損益、S1/S2 分解、
期末庫存評價、回撤、到期會計占比、以及持倉日數計算的融資敏感度。
