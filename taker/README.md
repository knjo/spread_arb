# Taker 期現貨價差研究

本目錄集中期貨／現貨價差的 taker 研究程式、方法文件與驗證工具。為保留既有腳本間的扁平匯入關係，執行檔維持在同一層；可重用的計算邏輯放在 `spread_arb/`，稽核工具放在 `verify/`。

新的日內 basis maker 研究已獨立規劃於 [`../maker/`](../maker/README.md)，不屬於本 taker 回測口徑。

## 資料來源（2026-09-07 起：SSD2 / NAS，不再經 sdk_core）

所有路徑集中在 `data_paths.py`，讀 HFT 根目錄 `config/pipeline.yaml` 的 `data_storage`：

| 資料 | 位置 | 備註 |
|---|---|---|
| 現貨 ticks | `{tick_dir}/{date}_StockTick.parquet`（SSD2） | 價格已是真實價 float、RecvTime 為 naive UTC |
| 股期 ticks | `/mnt/NAS/Parquet/Ticks/YYYY/MM/DD/stock_futures.parquet` | 放大整數 ÷100、RecvTime tz-aware UTC、含全部月份；SSD2 沒有股期 |
| 現貨基本面 | `{market_dir}/{date}_marketData.parquet`（SSD2） | 取代 `TwMarketData.get_equity_basic_info` |
| 期貨基本面／交易日曆 | MySQL（`taifex_pib_view`、`calendar_view`） | 直連 sqlalchemy+pymysql，不需 python-dotenv |
| 研究輸出 | `{base_dir}/stockfuture/`（SSD2） | 原 `HFT/data/stockfuture/`；近月標準合約股期會在此快取成 `{date}_stockfuture.parquet` |

## 建議執行環境

```bash
uv sync                                        # HFT 根目錄
cd src/research/futures_spot_spread/taker
uv run python arbitrage_analysis.py features -s 20260706   # tick-level 主流程
uv run python main.py -s 20260706 --ticks                   # 舊版低頻流程（輸出相對路徑 out/）
```

`requirements.txt` / `requirements.snapshot.txt` 僅保留舊環境紀錄；現行 HFT 主環境已足夠。
舊版 `main.py`／`peek_ticks.py`／`settle_official.py` 的 `tw`/`tw_md`/`mysql` 參數傳 `None` 即走本地來源，
傳入 sdk_core 物件仍可走舊 SDK。

## 內容索引

### 核心低頻流程

- `main.py`：建立每日事件明細，是舊版流程的主要入口。
- `spread_arb/`：資料前處理、報價對齊、價差事件、成本、本金、指標與報表共用模組。
- `mysql.py`：策略資料庫載入器。
- `settle.py`、`settle_official.py`：每日結帳與官方結算價試算。
- `analyze_day.py`、`analyze_year.py`：單日與跨日彙總。
- `report_first.py`、`report_pool.py`：首次進場與含二次進場報表。

### Tick-level 主研究流程

- `arbitrage_analysis.py`：原始資料擷取、特徵建立、首次進場、部位控制與回測的主要入口。
- `basis_adjustment.py`：前一日基差基準與篩選。
- `backtest_075_capital_100m.py`：0.75% 路徑在一億元資本限制下的縮放回測。
- `leg_pnl_by_convergence.py`：依期貨／現貨腳及收斂狀態拆解損益。

### 期貨腳 maker 掛單研究（2026-09-07 新增）

- `fut_maker_quote_stability.py`：現貨仍 taker 吃 A1，期貨依期望價差反推賣價 `P* = ceil_tick(spot_ask/(1−θ))`，
  在 `bid1 < P* < ask1` 時掛進期貨買賣價差內（隊列第一）。逐 tick 模擬每筆掛單存活到「被迫改單」
  （現貨 A1 漲／被人掛到下面）或成交的時間，並記錄掛出當下的市況特徵，用來找「哪些情況掛了不必常改單」。
  支援 `--policy pstar,a1m1,a1m2`（掛 P*／A1−1／A1−2）。輸出 `stockfuture/fut_maker_quote/`。
- `FUT_MAKER_QUOTE_STABILITY_20260907.md`：上述研究 12 個樣本日的初版結果與建議閘門。

### 進場與候選標的分析

- `candidate_filter_analysis.py`：候選階段 tickFeature 篩選。
- `prevday_candidate_filter_analysis.py`：以前一日門檻進行候選篩選。
- `entry_slippage_tickbp_streaming.py`：以有限記憶體重建可成交進場帳本。
- `entry_spot_book_slippage_analysis.py`：現貨委託簿與進場滑價。
- `entry_big_buy_ratio_analysis.py`：盤中買量相對歷史量與進場滑價。
- `tickfeature_factor_screen.py`：進場 tickFeature 因子篩選。

### 出場、滑價與微結構

- `exit_convergence_analysis.py`：價差發散後的可成交收斂分析。
- `convergence_days_by_settle_cycle.py`：收斂天數 x 門檻 x 結算週期位置（距結算／結算後第 N 個交易日）。`build` 從 SSD2 現貨 + NAS 股期重建輕量價差流到 `stockfuture/spread_stream/`（舊 `out/events_*` 事實表已不在），`summarize` 產各分組的當日／跨日／抱到結算比例與平均收斂交易日，`crosstab` 再切結算後第 N 日 × 距結算交叉表、週期進度、相對 tick／價位檔，輸出 `stockfuture/convergence_days/`；結果見 `出場分析報告.md` 第七節（7.6 換軸、7.7 tick size）。
- `exit_future_first_analysis.py`：先平期貨的出場執行分析。
- `exit_tickfeature_factor_screen.py`：出場 tickFeature 因子篩選。
- `peek_ticks.py`：擷取進出場點前後的 raw ticks。
- `analyze_peek.py`、`peek_trend.py`：peek 資料與出場走勢分析。
- `peek_slip.py`、`peek_grab.py`：出場賣現滑價及搶輸後吃量分析。
- `peek_factor.py`：出場滑價因子驗證。

### 現貨單腳策略

- `spot_only_premium_analysis.py`：期貨溢價訊號後的現貨單腳結果。
- `spot_only_one_lot_overnight.py`：現貨單張隔夜測試。

### 文件與驗證

- `METHODOLOGY.md`：低頻分析方法、假設與驗證口徑。
- `RESEARCH_DIRECTIONS.md`：後續研究方向。
- `出場分析報告.md`：出場微結構與滑價研究摘要。
- `verify/`：報表顯示、表三稽核、完整版表三與胃納量複驗。
- `sdk_test.py`：市場資料 SDK 的簡易連線測試。

## 建議閱讀順序

1. 先讀 `METHODOLOGY.md` 理解舊版事件與成本口徑。
2. 以 `arbitrage_analysis.py` 作為新版 tick-level 流程入口。
3. 依問題選擇進場、出場、候選標的或現貨單腳分析。
4. 報表產出後使用 `verify/` 做獨立驗算。
