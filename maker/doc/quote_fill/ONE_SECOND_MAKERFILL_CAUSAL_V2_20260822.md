# 因果動態商品池 q95 AB1/2：makerFill 成交／撤單 screening

日期：2026-08-22  
狀態：完整 72 日因果池的快速 entry screening；下游 +50 ms hedge／terminal path／portfolio 已完成，但本表本身仍不是 exact execution

## 結論

固定 45 檔已完全移除。本表使用月 M 只看 M-1、日 D 再只看至 D-1 的商品池，
涵蓋 2026-05-04 至 2026-08-13、72 日、3,886 product-days。策略為 q95、現貨
Bid maker，只允許 submit 在 raw displayed BID1／BID2，13:00 停止 entry。

| 指標 | 結果 |
|---|---:|
| quote candidates | 326,549 |
| raw BID1/2 rank 可映射 | 326,332（99.9335%） |
| outcome 可判定 | 326,314 |
| makerFill fill-before-nominal-cancel | 7,730（2.3689%） |
| nominal cancel-before-fill | 318,584（97.6311%） |
| unknown／fail-closed | 235 |
| 不撤單、一路看到 EOD 的 legacy positive | 86.0610% |

`97.63%` 是 target retreat／gate close／cutoff 早於 approximate maker fill 的
**策略撤單 outcome**，不是交易所 cancel ACK。EOD positive 高達 86.06%，但加入策略
後撤後只有 2.37% 能及時成交，證明不能直接把 makerFill EOD 欄位當成交率。

## 點位與月份

| Raw rank | 可判定單數 | approximate fills | fill rate |
|---|---:|---:|---:|
| BID1 | 37,410 | 4,598 | 12.2908% |
| BID2 | 288,904 | 3,132 | 1.0841% |

| 月份 | 可判定單數 | fills | fill rate | cancel rate |
|---|---:|---:|---:|---:|
| May | 78,773 | 2,641 | 3.3527% | 96.6473% |
| June | 101,260 | 2,740 | 2.7059% | 97.2941% |
| July | 103,148 | 1,932 | 1.8730% | 98.1270% |
| August（至 13 日） | 43,133 | 417 | 0.9668% | 99.0332% |

月成交率一路下降，尤其 August 只有約 0.97%。這會直接壓低後續交易量與預估損益；
不能再沿用 fixed-45 回測的成交數量。目前 order lifetime p50／p95／p99 為
9／543／5,382 秒；真正成交者的 submit-to-fill delay p50／p95／p99 為
2.93／92.62／698.86 秒。

## 計算契約

1. 由現行一秒 unique-absolute-price event root，依
   `(Date, ValueCode, generation)` 配出一個 submit 與一個 nominal cancel。
2. submit 的 `(ValueCode, second_from_open)` join `causal_fair`，取得一秒 decision
   timestamp 與 as-of `spot_sequence`。
3. 精確 key 為：
   `spot ValueCode + spot_sequence == StockTick.ChannelSeq == makerFill.ChannelSeq`。
4. raw tick 再驗一次 target price 是否真的等於 displayed BID1 或 BID2；
   `BID1-1 legal tick` 遇到 book gap 不冒充 BID2，217 筆因此 fail closed。
5. legacy implied time 為
   `snapshot RecvTime + Float32 FillSeconds`，只接受 active interval
   `(submit decision, nominal stop]`；18 筆 implied fill 不晚於 submit，列 unknown。

FillSeconds producer 實際依 TransTime 排序計算，這裡為了和 causal decision／future
book 接軌，沿用 repo fast-adapter 的 RecvTime anchor。因此它是 mixed-clock screening，
`fill_cursor_exact=false`、不含 own quantity、partial fill、cancel ACK 或 joint volume。
既有五日 fixed-cohort 校準中，q95 B1/2 mixed rate 1.88%、exact indexed rate 1.72%；
只能粗略暗示本次 2.37% 可能對應約 2.17%，不能取代動態池 exact replay。

## Canonical bundle

```text
maker/data/walkforward/one_second_makerfill_causal_v2_20260822_v1/
  candidate_outcomes/Date=YYYYMMDD/candidate_outcomes.parquet
  overall_summary.csv
  daily_summary.csv
  monthly_summary.csv
  rank_summary.csv
  stop_reason_summary.csv
  outcome_status_summary.csv
  daily_input_audit.csv
  complete.json
  verification.json
```

每筆 downstream entry 保留：`physical_order_id`、Date／ValueCode／QuoteCode、q95、
generation、target price、submit／stop／spot snapshot／implied fill／hedge decision ns、
anchor、upper／lower distance、contract size、end date、submit 時 futures bid，以及所有
approximate／support flags。`full_fill` 為 nullable approximate label；下游必須同時要求
`outcome_supported=true`，不得把它改名成 exact fill。

正式全跑耗時 7.915 秒，peak RSS 約 1.473 GiB；72 partitions 共 326,549 rows、
27,899,557 bytes。Verifier 已重算五張 aggregate、驗 physical ID 唯一、D-1 boundary、
time／fill／cancel invariants，結果 PASS；partition inventory SHA256 為
`650dab5144e8e72845da87cb0a08d6821fafcc60cbd6f853ff9c8011e33630f7`。

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-project --with polars \
  python -m maker.src.quote_fill.one_second_makerfill_runner \
  --verify-only \
  --output maker/data/walkforward/one_second_makerfill_causal_v2_20260822_v1
```

Focused runner/message tests 6/6 通過，Ruff 通過。

## 不可使用的首次中斷包

`one_second_makerfill_causal_v2_20260822_v1_incomplete_pre_aggregate_fix` 是首次在
aggregate 前中斷的殘留包，已放置 `DO_NOT_USE.json`；唯一 canonical root 是上節沒有
`incomplete` 字樣的目錄。

## 下游狀態更新

這 7,730 筆 approximate fills 已接上完整動態池的 futures +50 ms hedge、同日／跨日／
到期 terminal path，以及 1,000–5,000 萬 full-population inventory cap replay；沒有重新接回
fixed-45 execution facts。端到端結果見
`maker/doc/quote_fill/DYNAMIC_CAUSAL_END_TO_END_STATUS_20260822.md`。

仍未完成的是 exact maker queue／own quantity／partial fill／joint volume 與 fill-aware re-entry
controller。因此本 bundle 仍只適合 first-fill screening，不能把 7,730 筆直接稱為實盤成交。
