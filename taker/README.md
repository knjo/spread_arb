# Taker 期現貨價差研究

本目錄集中期貨／現貨價差的 taker 研究程式、方法文件與驗證工具。為保留既有腳本間的扁平匯入關係，執行檔維持在同一層；可重用的計算邏輯放在 `spread_arb/`，稽核工具放在 `verify/`。

## 建議執行環境

新版 tick-level 流程使用 HFT 專案環境；先從 HFT 根目錄同步：

```bash
uv sync
```

`requirements.txt` 列出舊版低頻流程相對 HFT 主環境多出的套件；`requirements.snapshot.txt` 則保留搬移前的完整鎖版環境快照。

舊版低頻流程使用相對路徑 `out/`，建議先進入本目錄再執行：

```bash
cd src/research/futures_spot_spread/taker
uv run --with-requirements requirements.txt python main.py -h
```

新版 tick-level 研究會自行定位 HFT 專案根目錄，輸出放在 `data/stockfuture/`。

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

### 進場與候選標的分析

- `candidate_filter_analysis.py`：候選階段 tickFeature 篩選。
- `prevday_candidate_filter_analysis.py`：以前一日門檻進行候選篩選。
- `entry_slippage_tickbp_streaming.py`：以有限記憶體重建可成交進場帳本。
- `entry_spot_book_slippage_analysis.py`：現貨委託簿與進場滑價。
- `entry_big_buy_ratio_analysis.py`：盤中買量相對歷史量與進場滑價。
- `tickfeature_factor_screen.py`：進場 tickFeature 因子篩選。

### 出場、滑價與微結構

- `exit_convergence_analysis.py`：價差發散後的可成交收斂分析。
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
