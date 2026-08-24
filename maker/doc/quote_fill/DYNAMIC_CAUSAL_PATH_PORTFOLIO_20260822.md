# Dynamic causal universe：estimated path / inventory-cap 結果

日期：2026-08-22

## 結論先行

這版已完全移除固定 45 檔的上帝視角篩選。每日只能使用當日 causal manifest 的進場商品；商品隔日即使離開 universe，既有部位仍可繼續找出場。進場期共有 72 個交易日（2026-05-04 至 2026-08-13），持倉與出場追蹤延伸至 2026-08-21，共 78 個 reporting sessions。

這是 analysis-only estimate，不是 production backtest。主要近似仍是 makerFill 的成交與成交時間、1 Hz 出場取樣、出場沒有額外送單 latency，以及到期 futures 價格使用日盤最後有效成交 proxy。以下 portfolio PnL 也只計已定價平倉；仍 open/unpriced 的部位會繼續占容量，但不會被假設成零損益。

正式 streaming run 耗時 21.52 秒，peak RSS 2,321,272 KiB（約 2.21 GiB）。它是逐日掃描仍存活 exact keys，不會把 72 日 fair grid 全部轉成 Python tuples 留在記憶體。

## 資料與凍結規則

- Dynamic manifest：72 日、3,886 product-days、119 檔 union；實際 7,730 筆 fill 涵蓋 116 檔、263 個 spot/futures pairs。`fixed45_universe_used=false`。
- 進場：spot bid makerFill estimate；成交後 +50 ms，以 raw futures L1-L5 causal as-of book 賣出一口 futures taker。正式 PnL 已直接使用該 executable VWAP，hedge slippage 沒有再重複扣一次。
- Frozen lower：每筆 submit 時固定 `anchor_ewma_120s_bp - lower_distance_bp`。
- 正常出場：entry hedge 後下一個完整秒開始，第一個 `analysis_eligible`、兩腿 depth 足夠、且 `basis_buy_taker_bp <= frozen lower` 的 spot bid / future executable ask。
- 到期：paired accounting mark 優先；spot 是 local daily official close field，future 是最後 positive、non-TrialMatch 的 day-session trade proxy，不宣稱 official close/settlement，也不宣稱可成交。
- 成本：spot 雙邊 commission 各 1.71 bp；spot sell tax 當沖 15 bp、非當沖 30 bp；futures 雙邊 tax 各 0.2 bp；futures 雙邊 commission 各 TWD 20。
- Portfolio：hard cap 10/20/30/40/50M、單品 30%、13:00 起不收新 maker fill、開盤已有該商品 carry 則該日只出不進。13:00 判斷使用 maker fill time；+50 ms hedge 跨過 13:00 不會回頭拒單。

## 全部 7,730 筆路徑覆蓋

| 結果 | 筆數 | 占全部 fill | Gross PnL | 成本 | Net PnL |
|---|---:|---:|---:|---:|---:|
| 同日 frozen-lower hit | 2,680 | 34.67% | 5,427,400 | 2,362,077 | 3,065,323 |
| 跨日 frozen-lower hit | 3,906 | 50.53% | 12,171,900 | 8,366,081 | 3,805,819 |
| 到期 paired proxy mark | 929 | 12.02% | -254,500 | 2,268,228 | -2,522,728 |
| +50 ms hedge 無法定價 | 213 | 2.76% | — | — | — |
| 觀測期結束仍 open | 2 | 0.03% | — | — | — |

完整 terminal pricing coverage 是 7,515 / 7,730 = 97.22%。若把「進場當天即有 terminal」都算日內完成，另包含 203 筆進場日剛好是 expiry 的 close mark，因此是 2,883 / 7,730 = 37.30%；比較純粹的正常 frozen-lower 當沖率則是 34.67%，或在 7,515 筆已定價路徑中為 35.66%。

未受 cap 限制的 7,515 筆已定價路徑，spot entry notional 合計 TWD 4.306B；gross 17.345M（40.28 bp）、完整成本 12.996M（30.18 bp）、net 4.348M（10.10 bp）。這只是已定價路徑總和，不是可部署容量結果。

月別正常 frozen-lower 同日 hit / 全部 fill：May 38.17%、June 41.02%、July 26.55%、August 8.39%。August 的日內收斂明顯轉弱。

## +50 ms futures hedge

7,517 / 7,730 = 97.24% 可在 +50 ms decision book 完整賣出一口。對 executable rows，signed adverse total slippage 為 mean 5.80 bp、p50 0 bp、p90 20.54 bp、p95 30.64 bp、max 274.35 bp；decision book age p50 17.31 ms、p95 505.24 ms。

213 筆未定價 hedge 的原因是：arrival gate closed 196、decision gate closed 14、no arrival book 3。細項為 invalid executable book 185、reference-price band 25、沒有 arrival book 3。這些 fill 的台北時間落在 09:05:00 至 12:58:00，13:00 後為 0 筆，exact 13:00 fill/cancel race 也是 0；因此 gate closed 不是 13:00 cutoff 造成，而是 future book/ref gate 本身不通過。

## Full-population inventory-aware cap replay

下表的「日均新 spot」以 72 個有進場資料的 sessions 為分母；「期間新 spot」是 one-way spot entry notional。Mean EOD 與 final EOD 都包含 unresolved inventory。

| Hard cap | 接受 fills | 期間新 spot | 日均新 spot | Mean EOD | Peak EOD | Final EOD（unresolved 筆數） | 同日完成 / accepted |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10M | 816 | 193.722M | 2.691M | 9.123M | 9.997M | 4.307M（18） | 36.27% |
| 20M | 1,091 | 341.481M | 4.743M | 18.082M | 20.000M | 9.084M（22） | 33.82% |
| 30M | 1,444 | 526.244M | 7.309M | 25.834M | 29.978M | 16.335M（47） | 32.96% |
| 40M | 1,491 | 625.182M | 8.683M | 34.204M | 39.973M | 23.476M（56） | 31.19% |
| 50M | 1,697 | 768.247M | 10.670M | 40.866M | 49.999M | 25.817M（56） | 32.06% |

若把 8/14–8/21 六個只出場、沒有新 entry 的 reporting sessions 也算進平均，20M 日均新 spot 是 4.378M；在 72 日 entry window 內則是 4.743M，即每天約 0.237 倍 hard cap。沒有 13:00 積極清倉時，容量大部分時間被 carry 占滿，故週轉並不高。

| Hard cap | Realized gross | Realized cost | Realized net | Net / 全部 accepted new spot | Worst day | Max realized drawdown |
|---:|---:|---:|---:|---:|---:|---:|
| 10M | 665,500 | 581,155 | 84,345 | 4.35 bp | -40,668 | 46,223 |
| 20M | 1,163,600 | 1,009,893 | 153,707 | 4.50 bp | -123,076 | 134,774 |
| 30M | 1,934,000 | 1,521,647 | 412,353 | 7.84 bp | -219,293 | 219,293 |
| 40M | 2,256,800 | 1,860,526 | 396,274 | 6.34 bp | -254,954 | 254,954 |
| 50M | 2,841,000 | 2,247,164 | 593,836 | 7.73 bp | -316,957 | 316,957 |

這些數字不應解讀成線性，也不能把 20M 的 153,707 當成完整 terminal portfolio EV。不同 cap 會改變離散訂單的 admission 順序、30% 單品限制，以及哪些 carried products 隔日變成 exit-only；而 unresolved rows 仍占容量但沒有 cashflow。正式 summary 因此固定標記 `full_population_portfolio_ev=false`。

### 20M 的 22 筆 final unresolved

20M 接受的 1,091 筆中，1,069 筆已有 terminal，22 筆沒有。22 筆全部是 entry hedge unpriced，沒有 observation-horizon-open row：

| 類型 | 筆數 | Final occupied notional |
|---|---:|---:|
| arrival gate closed | 18 | 4.3611M |
| decision gate closed | 3 | 4.1330M |
| no arrival book | 1 | 0.5900M |
| 合計 | 22 | 9.0841M |

這 9.0841M 從各自進場後一直保留到 8/21，壓縮後續 admission；PnL 沒有填 0，也沒有以任意 futures price 補 hedge。它們代表資料/執行狀態下無法確認已完成 delta hedge 的 spot exposure，實盤必須有 retry、fail-safe 或人工處理，不能當成正常中性 carry。

20M 已定價 branch 中，同日 frozen-lower 329 筆 net +203,999.62、跨日 frozen-lower 601 筆 net +143,539.51、到期 proxy mark 139 筆 net -193,831.64，合計 realized net +153,707.49。最差日是 2026-07-15；當日 41 筆 expiry proxy marks 貢獻 -121,691.92，幾乎解釋整日 -123,076.31。這支持先前觀察：留到結算附近的部位及非當沖稅費，是尾端損失的主來源。

## 可用檔案

- `maker/data/walkforward/dynamic_estimated_path_portfolio_causal_v1_20260822/complete.json`
- `entry_positions.parquet`
- `estimated_terminal_paths.parquet`
- `estimated_priced_paths.parquet`
- `unresolved_paths.parquet`
- `coverage.csv`
- `full_population_cap_events.parquet`
- `full_population_cap_daily.parquet`
- `full_population_cap_summary.csv`
- `portfolio_10m_20m_30m.png`

`completed_only_cap_*` 仍保留作條件性 diagnostic，但不是本報告的正式 portfolio 結果，因為它會先移除所有 unresolved paths 才回放容量。

## 驗證

- Artifact invariants：7,730 unique paths、7,515 priced、929 expiry proxy marks、所有 expiry proxy flags 保留、daily calendar 到 2026-08-21、五個 scenario 均未突破 portfolio / 30% product hard cap。
- 21 個 focused unit tests 通過，涵蓋真實 hedge schema adapter、strict next-second first passage、跨日離開 universe 仍可出、expiry proxy 優先、unresolved inventory、13:00 fill-time cutoff 與原 cap engine。
- Focused Ruff check 通過；module / plot `py_compile` 通過。

