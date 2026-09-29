> 2026-09-23 整理：現行規則與驗證狀態見 [CURRENT.md](CURRENT.md)。本文件保留前一階段規格／結果；其中數字未包含本次新增的每路線 60 秒更新，不可直接視為本次驗收。

# 2026-09-22：A／B 回測修正與現行執行方式

**2026/9/23 更正**：本文描述的全額掛單預留是 9/22 研究變體，偏離使用者「未成交掛單不占部位」的規則。
不能將其當作唯一正確的容量實作或用該變體的年化取代使用者策略；當前規則與歸因更正以 [05_CAPITAL_POLICY_CLARIFICATION.md](05_CAPITAL_POLICY_CLARIFICATION.md) 為準。

舊 `A_fixed`／`B_dyn` 的加總公式可對上帳本，但其撮合、未來進場結果提前占用資金與到期處理存在問題。
原始結果與原始原碼封存在 `data/backtest/correction_20260922/original_src.tar.gz` 及原 run 目錄。
現行 Stage 3 從 raw books／prints 重建交易路徑，實作為 `src/backtest/causal_market.py`、`causal_replay.py`。
`src/backtest/replay.py` 已改接此引擎；`legacy_replay.py` 只供歷史診斷。

## 架構與資料

1. **日初估計**：`ev/qlevel.py`、`reach.py`、`abs_reach.py` 使用決策日前已可觀測的資料。
   Q 表與 absolute 表沿用既有快取／樣本來源；歷史成交標籤不參與當天資金或撮合決策。
2. **市場與候選**：`causal_market.Market` 讀 maker 的每日 mapping／causal 1 Hz grid，及原始現貨、期貨五檔與 prints。
   以秒邊界、book 更新與 prints 產生候選。S1 為現貨 B1；S2 為期貨 A1 下移一 tick 的 inside 單。
   新進場到 12:53:20；已有部位的 E1／E2 可繼續產生至 13:18 撤單。
3. **投組回放**：A／B 共用同一條市場時間軸，但各自擁有 FIFO queue、taker depth、現金與資金 ledger。
   市場資料和基礎 EV 可以預先在 worker process 建立，進出場和跨日部位仍按日期、事件順序執行。
4. **獨立驗證**：`causal_audit.py` 從輸出現金逐腿重算部位、損益和每日權益，核對資金預留、成交量／深度及 B 的歷史訊號。

原 `points/s1.py`／`s2.py` 的獨立雙向標籤表仍保留供歷史研究；修正版投組不讀其中的 `t_fill_ns`、`hedge_ns` 等未來成交欄位。
新的實際成交、回補與出場資料寫在每個 run 的 `cash_legs.parquet`／`events.parquet`。

## 成交、資金與風險處理

- 9/22 變體採送單即預留資金，包括最後沒有成交的單（這是政策選擇，非使用者原策略要求）。S1 按買價；S2 按當天現貨漲停價乘 2,000 股，確保後續 hedge 不會突破預留。
  hedge 完成後才縮减至實際現貨成本。全投組 2,000 萬；每檔上限為 `max(一口所需資金, 500 萬)`。
- 新單、取消與 hedge 延遲各 50 ms。新單到達時使用 post-only 模擬；傳輸期間變成可成交限價者在到達時拒單。
  不會於送出時用未來 book 提早拒絕、返還資金或選擇最後會成交的單。
- 每筆 print 的數量只使用一次，包含外部 ahead 和自己的所有單。ahead 在實際生效時讀取；生效前 prints 不可清自己的隊列。
  外部撤單不贈送優先權。這是基於 public prints 的保守 FIFO 近似，並非完整交易所逐委託還原。
- 每個五檔 snapshot 的 taker 深度只使用一次。深度不足的 hedge 逐筆重試；未完成的裸腿和資金跨日保留。
- 取消訊號在傳輸途中出現仍計入取消延遲內的成交。S1 partial 在取消生效後按實際 book 賣回。
  E1／E2 同時出場的競賽若使另一單也成交，須記錄該腿並真實回補，不能只列風險旗標。
- 進場後保存原 `qc`。出場、回補皆使用該合約；沒有該合約的出場市場資料時不拿下一月替代。
  已知公司行動公告依公告可得日停止新進場，既有配對嘗試兩腿 taker 風險出場。
- 日末若找不到到期記帳所需價格，保留 `settlement_pending`／曝險與資金；缺價不當作零損益。

## EV 與 B 的定義

- A 使用 8.5 bp／交易日，轉成 `8.5 × 250 / 365` bp／日曆日。
- B 前一日收盤 committed 達 1,600 萬時，使用 `max(A 基礎門檻, 前一日独立基礎准入訊號 score 的中位數)`。
  訊號先於持倉、busy 或容量篩選建立；A／B 的母體相同。相同商品、合約、秒及完整 EV 輸入狀態只算一次。
  不能因目前沒資金、已持有該股或今日提高門檻，就讓訊號從明天的分布消失。
- 決策快取不再用分鐘、整數 basis／anchor 或缺少 e 的鍵。表格 lookup 可按表格實際 bucket 快取，EV 本身仍使用精確輸入。
  每秒結束只保留訊號 score，釋放不可能再次使用的舊決策，避免全日快取膨脹。
- `scale_raw > 60` 明確排除；沒有合法 scale／e 時不冒用 scale=1 的 residual Q 預測，absolute／settlement 仍可獨立評估。
- S2 進場成本使用既有 empirical decay 與現貨半 tick 下限之最大值；5 倍現貨 A1 深度須在掛單期間持續成立。
- absolute 的零收斂表不能直接替較高的 max-Q 目標估 EV。若 max-Q 為正，改使用該 residual 目標對應的 reach 機率及損益。
- 預期持有時間改成到期日收盤；到期當日使用同日費用，settlement branch 一致扣 `d_settle`。

## 保留的研究假設

本次先隔離回測邏輯變更，保留原配對成本 20／34 bp、未加入資金利息、到期兩腿同價 basis＝0 記帳。
到期記帳不宣稱等同官方期貨最終結算加真實現貨成交。兩腿相同價格時，該記帳 gross 不依賴共同 mark 的數值。

日終估值用官方現貨收盤及原持有合約的官方期貨日結算價，全期間 131 日皆已補齊。
`valuation.py` 在交易回放完成後自動重估每日權益，保留回放原始的 `marks.parquet`／`*_daily.csv`，另寫官方版本。
價格來自專案資料庫 127 日及期交所日行情下載補足的 6/26、6/30、7/13、7/14；控制日 6/25 的 1,192 筆股票期貨價格全數吻合。
原始下載、來源參數與 SHA256 存於 `data/backtest/metadata/`；`official_marks.py` 可重建完整快取。
其中 3/9 原始 BBO 無法標記的旺宏舊合約 DIFC6，以官方日結算價 92.3 補齊，沒有用新合約替代。
補價只影響報表，不影響訊號、資金預留或 B 的門檻。仍缺值時保留缺值並使驗收失敗，不能當作零損益。
報告的回撤稱為「官方日終權益回撤」；原報告的結案損益回撤不能當作相同指標比較。
全期間 headline 含終端持倉評價；4–7 月另分結案入帳及每日 MTM，避免混用。

## 完整執行

**記憶體行為（9/23 實測）**：現行入口是完整 raw 重放，並非只讀既有合格點位表。
`Market` 先載入 mapping／grid 商品集合的全天五檔與 prints，建立每商品的密集 Timeline 後才篩候選。
`prepare_day` 另重建 Q／EV，將整個 Market 與估計一起序列化；A／B 分開啟動時不共用這些資料。
`--prefetch` 預設 2，兩組分開跑即 2 個主程序加 4 個長駐 worker；9/23 觀測主程序各約 11 GiB、
worker 各約 17–18 GiB，合計約 90 GiB，且另有 swap。數字隨交易日與階段變動，並非記憶體上限。
降低 `--prefetch` 可減少同時處理的整日資料；只調 `POLARS_MAX_THREADS` 不會減少 worker 數。
`Market.timelines` 與 `Timeline.market` 互相參照，另有延後回收風險；本次未用 heap profiling 量化其占比。
下列 `--prefetch 3` 是歷史執行指令，不代表低記憶體設定。A／B 共用行情入口目前預設全額預留，
不能為了省 RAM 直接切換入口而忽略使用者的不預留政策；資源調整須保留原政策與 checkpoint。

在 `src/research/futures_spot_spread` 下執行；全部 Python 使用專案 uv 環境。

```bash
UV_CACHE_DIR=/tmp/hft-uv-cache POLARS_MAX_THREADS=4 uv run --project /home/kevin/Project/HFT --no-sync python -m spreadArb.src.backtest.causal_replay --out corrected_20260922_v2 --prefetch 3

# 同一設定、同一起訖日期續跑；完整日有 atomic checkpoint，可重做未完成日。
UV_CACHE_DIR=/tmp/hft-uv-cache POLARS_MAX_THREADS=4 uv run --project /home/kevin/Project/HFT --no-sync python -m spreadArb.src.backtest.causal_replay --out corrected_20260922_v2 --prefetch 3 --resume

UV_CACHE_DIR=/tmp/hft-uv-cache POLARS_MAX_THREADS=4 uv run --project /home/kevin/Project/HFT --no-sync python -m spreadArb.src.backtest.causal_audit --run corrected_20260922_v2 --raw

UV_CACHE_DIR=/tmp/hft-uv-cache POLARS_MAX_THREADS=4 uv run --project /home/kevin/Project/HFT --no-sync python -m spreadArb.src.backtest.causal_report --run corrected_20260922_v2

UV_CACHE_DIR=/tmp/hft-uv-cache POLARS_MAX_THREADS=4 uv run --project /home/kevin/Project/HFT --no-sync python -m unittest discover -s spreadArb/src/tests -t .
```

單獨一組可用 `-m spreadArb.src.backtest.replay --preset A --out my_A`。
明確傳入的選項優先於 preset，包括 `--exit-routes E1 --abs-target zero --dyn-cap-frac 0` 這些剛好等於 parser 舊預設值的選項。

## 輸出與驗收

- Run 根目錄：`manifest.json`、`checkpoint.pkl`、`A_daily.csv`／`B_daily.csv`、`summary.json`。
  `source_history` 記錄續跑前的 source snapshot；本次續跑修復空表 schema、調整 I/O 預取並接上執行後官方評價，未更改已完成日期的交易規則。
- 每日：`inputs.json`、`decisions.parquet`、`base_signal_scores.parquet`、`complete.json`。
- 每日每投組：`positions.parquet`、`cash_legs.parquet`、`events.parquet`、`ledger.parquet`、`depth_claims.parquet`、`marks.parquet`。
- 官方評價：`A_daily_official.csv`／`B_daily_official.csv`、每日每投組 `marks_official.parquet`、來源摘要 `official_valuation.json`。
  `summary.json` 與獨立驗收優先使用官方版本；缺價會保留明確缺值。
- 獨立驗收：`verification.json` 的 `complete=true` 且 `status=PASS`；`A_reconciled.csv`／`B_reconciled.csv`、全期間去重部位帳。
- 比較報告：`comparison.json`／`.csv`、`COMPARISON.md`；只有完整驗收通過後才產生，結果彙入 `03_RESULTS.md`。

`logic_audit.py` 的舊點位引擎探針用於保留問題證據，不是現行事件引擎的驗收入口。
