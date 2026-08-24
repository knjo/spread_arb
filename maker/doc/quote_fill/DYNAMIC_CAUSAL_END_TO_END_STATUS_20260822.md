# 動態商品池 AB1/2：端到端現況

日期：2026-08-22  
範圍：2026-05-04 至 2026-08-13 共 72 個進場日；持倉追蹤至 2026-08-21，共 78 個 reporting sessions

## 結論

固定 45 檔已從這條新 pipeline 完全移除。月 M 只用完整 M-1 的收斂統計決定月池，日 D 再套只看到 D-1 的流動性 gate；72 日共有 3,886 product-days、跨月聯集 119 檔，每日實際 35–74 檔。商品離開新進場池時，既有庫存仍保留為 exit-only，不會消失或被假設平倉。

目前結果顯示：AB1/2 掛單大多會在成交前因策略後撤／gate／cutoff 被取消；可成交者的 +50 ms futures hedge 大多能執行，但尾端滑價不小；正常 frozen-lower 的同日平倉率約 34.67%。成本後路徑仍為正，但部位 cap 回放大部分時間被 carry 與無法定價的 hedge 尾巴占滿，所以交易量與 realized PnL 遠低於不受容量限制的路徑加總。

這仍是 development research，不是 production GO。逐列商品選擇沒有使用 target day／target month outcome，但 proxy 與門檻是在看過 Apr–Aug development 結果後決定，尚未通過新的 prospective holdout。

## 商品池與掛撤

| 指標 | 結果 |
|---|---:|
| 固定 45 檔是否使用 | 否 |
| 動態池跨月聯集 | 119 檔 |
| 每日可進場 | 35–74 檔 |
| 72 日 product-days | 3,886 |
| q95 AB1/2 new order candidates | 326,549 |
| outcome 可判定 | 326,314 |
| approximate fill-before-cancel | 7,730（2.3689%） |
| approximate cancel-before-fill | 318,584（97.6311%） |
| unknown／fail closed | 235 |

分點位看，BID1 為 4,598 / 37,410 = 12.2908%；BID2 為 3,132 / 288,904 = 1.0841%。成交率由 May 3.35%、June 2.71%、July 1.87% 降至 August 0.97%，後段 regime 明顯轉弱。

97.63% 是「nominal cancel 比 approximate fill 更早」的策略 outcome，不是交易所 cancel ACK。makerFill 仍是 mixed-clock fast screening，不含 own two-lot quantity、partial fill、cancel race 或跨訂單 joint volume；五日舊 cohort 的粗校準只暗示 2.37% 可能約對應 2.17%，不能把它當正式修正值。

1 Hz quote-intent 的 nominal message load 是 326,549 new 加 326,549 cancel，共 653,098 requests。09:05–13:00 active-second p50 / p95 / p99 / max 為 1 / 4 / 5 / 30，盤中沒有一秒超過現貨 100 requests/s；13:00 集中撤單每日 p50 / p95 / max 為 65 / 165 / 199，16 / 72 日超過 100，因此最壞要兩秒 drain，實作應約 12:59:58 先停 new 並開始撤。

## 成交後 +50 ms futures hedge

| 指標 | 結果 |
|---|---:|
| hedge attempts | 7,730 |
| 一口完整 executable | 7,517（97.2445%） |
| arrival gate closed | 196 |
| decision gate closed | 14 |
| no arrival book | 3 |
| signed adverse slippage mean / p50 / p95 | 5.80 / 0 / 30.63 bp |
| notional-weighted signed slippage | 6.74 bp |
| 對應 signed cost / reference futures notional | 2.9168M / 4.3269B TWD |
| 單筆 signed cost mean / p50 / p95 | 388 / 0 / 2,000 TWD |

正值代表賣 futures 時的不利滑價。7,517 筆中 2,242 筆不利、5,050 筆為 0、225 筆有利；本次一口都能由 top executable level 吃完，因此 depth component 為 0，成本來自 +50 ms 間的價格移動。decision book age 約 p50 17 ms、p95 0.50 秒；尾端仍有 227 筆超過 1 秒，應另做 freshness sensitivity。

依這 7,730 個 approximate fill cursor，期貨 hedge 固定秒與任意 rolling 1 秒峰值都為 5，沒有超過使用者指定的 5 requests/s；每日 attempts 平均 107、p50 89、p95 244、max 325。這是 approximate fill-time 結論，正式 exact/joint-fill controller 仍需要 rolling limiter 與同商品、同 exact cursor batching。

213 筆無法定價不是 13:00 cutoff 造成：它們都落在 09:05–12:58，exact 13:00 race 為 0。原因是當下 futures executable book／reference gate 不通過；目前回測沒有替它們虛構下一個 hedge 價格。

## 平倉路徑與未受 cap 限制的損益

| Terminal path | 筆數 | 全部 fills 比例 | Net PnL |
|---|---:|---:|---:|
| 同日 frozen-lower hit | 2,680 | 34.67% | +3.065M |
| 跨日 frozen-lower hit | 3,906 | 50.53% | +3.806M |
| 到期 paired close proxy | 929 | 12.02% | -2.523M |
| entry hedge 無法定價 | 213 | 2.76% | 未填值 |
| 觀測期結束仍 open | 2 | 0.03% | 未填值 |

正常 frozen-lower 同日平倉是 34.67%；若連 203 筆「進場日即為 expiry、當天以 close proxy 終止」也算同日完成，則為 2,883 / 7,730 = 37.30%。在 7,515 筆已定價路徑中，正常同日率為 35.66%。月別正常同日率為 May 38.17%、June 41.02%、July 26.55%、August 8.39%。

7,515 筆已定價路徑的 entry spot notional 為 4.306B；gross 17.345M、完整費稅 12.996M、net 4.348M，等於 10.10 bp / entry spot notional。Gross 已使用 +50 ms futures execution price，所以 hedge slippage 已反映在 gross，不可再把上節 2.9168M 重複扣一次。這個 uncapped 加總只回答「可完整定價的路徑是否還有 edge」，不代表可部署資金能全部吃到。

## Full-population inventory cap 回放

正式表保留所有已接受但 unresolved 的部位，使其持續占容量；不把 cashflow 填成 0。單一商品 hard cap 為總 cap 的 30%，13:00 後不接受新 fill，開盤已有 carry 的商品當日只出不進。

「日均新 spot」以 72 個進場日為分母；PnL 是 2026-05-04 至 2026-08-21 期間、只對已定價 terminal 的 realized 值。

| Hard cap | 接受 fills | 新 spot 合計 / 日均 | Mean EOD | Final EOD | 同日全部 / 正常 lower | Realized gross / cost / net |
|---:|---:|---:|---:|---:|---:|---:|
| 10M | 816 | 193.722M / 2.691M | 9.123M | 4.307M（18） | 36.27% / 33.33% | 665.5k / 581.2k / **84.3k** |
| 20M | 1,091 | 341.481M / 4.743M | 18.082M | 9.084M（22） | 33.82% / 30.16% | 1,163.6k / 1,009.9k / **153.7k** |
| 30M | 1,444 | 526.244M / 7.309M | 25.834M | 16.335M（47） | 32.96% / 29.71% | 1,934.0k / 1,521.6k / **412.4k** |
| 40M | 1,491 | 625.182M / 8.683M | 34.204M | 23.476M（56） | 31.19% / 28.97% | 2,256.8k / 1,860.5k / **396.3k** |
| 50M | 1,697 | 768.247M / 10.670M | 40.866M | 25.817M（56） | 32.06% / 29.70% | 2,841.0k / 2,247.2k / **593.8k** |

所有 scenario 的 peak intraday／EOD 與單品 30% 都已驗證沒有突破 hard cap。若把 8/14–8/21 六個只有出場、沒有新 entry 的 sessions 也放入日均分母，20M 日均新 spot 是 4.378M。

20M 正式回放接受 1,091 筆，其中 1,069 筆已定價、22 筆未定價。22 筆全是 spot maker 已成交、+50 ms hedge unpriced，合計占 9.0841M；它們沒有被當成零損益，但從發生後持續壓縮容量。20M 已定價 branch 的同日 normal net +204.0k、跨日 normal net +143.5k、expiry proxy net -193.8k，合計 +153.7k。成本吃掉 gross 的 86.8%，留到 expiry 的尾端是主要拖累。

30M、40M、50M 的損益不線性是合理結果：離散訂單的 admission、單品 30%、opening carry exit-only 與 unresolved inventory 都會隨 cap 改變。`completed_only_cap_*` 先移除所有 unresolved 再 replay，只能作偏樂觀容量 diagnostic；例如 20M 為 net 380.9k，不是正式 portfolio 結果。

## 精度與目前判斷

目前可合理相信的是：固定 45 已移除、逐列 selection chronology、1 Hz 掛撤量級、makerFill screening outcome、raw futures +50 ms as-of 價格，以及 cap ledger 的算術／hard-limit invariants。

仍不能當 production truth 的部分：

1. makerFill 是 approximate mixed-clock label，未做 own quantity、partial/cancel race 與 joint queue allocation。
2. 出場是 1 Hz、下一完整秒的 taker/taker first-passage estimate，沒有 exit order latency、maker queue 或多部位 joint depth allocation。
3. 213 筆 hedge failure 尚無 retry／fail-safe 路徑；20M final 9.084M 因而是「未定價且占容量」，不是正常 delta-neutral carry。
4. 到期 spot 使用 local close field，future 使用日盤最後有效成交 proxy，不是 official settlement；929 筆 expiry branch 的 -2.523M 對結果很敏感。
5. 動態選股門檻仍是 development-tuned；需要凍結後用新月份 prospective holdout，才可稱正式 OOS。

因此目前結論是：**研究 edge 仍為正，但尚不足以宣稱可部署獲利**。下一輪最有價值的工作不是再換固定商品名單，而是完成 full dynamic exact makerFill／joint quantity、hedge retry/fail-safe、真實 FIFO exit controller，以及已凍結規則的新期間測試。

## 產物

- 商品池：`maker/data/walkforward/monthly_product_selector_causal_v2_20260822/`
- 掛撤：`maker/data/walkforward/order_message_load_causal_v2_20260822_v2/`
- makerFill：`maker/data/walkforward/one_second_makerfill_causal_v2_20260822_v1/`
- +50 ms hedge：`maker/data/walkforward/dynamic_future_hedge_causal_v1_20260822/`
- 路徑／部位：`maker/data/walkforward/dynamic_estimated_path_portfolio_causal_v1_20260822/`
- 10M／20M／30M 圖：`portfolio_10m_20m_30m.png`
- 路徑詳細報告：`maker/doc/quote_fill/DYNAMIC_CAUSAL_PATH_PORTFOLIO_20260822.md`

端到端 focused tests 51 / 51 通過；72 日 hedge provenance migration 的 price／status／slippage checksum 前後相同，72 / 72 partition marker hashes 一致。
