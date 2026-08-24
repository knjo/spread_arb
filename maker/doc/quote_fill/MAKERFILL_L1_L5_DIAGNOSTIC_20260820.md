# makerFill A/B1–5 五日技術比較（2026-08-20）

## 結論

主研究先限制在 A/B1–2 是合理的。五個預先固定的代表日中，B3–5 的 exact indexed full-fill rate 約只有 B1–2 的四分之一，而且成交後 50ms 對側 hedge slippage 約高 6–9bp。

這是一張 analysis-only 技術診斷表，不是策略 EV。Legacy makerFill 的 EOD label 與 mixed TransTime/RecvTime stop label 都只是近似；正式成交真值仍以 exact EventCursor、策略 stop、queue+own quantity 的 indexed replay 為準。

## 固定樣本

- 日期：20260603、20260703、20260706、20260731、20260806。
- 223 product-days：45/44/44/45/45，與 execution root manifest 完全一致。
- 只取 `spot_bid_future_taker`、spot bid maker、submit 由 exact spot snapshot 觸發、rank=BID1…BID5。
- 60,451 unique physical raw orders；89,700 q-policy aliases。
- q aliases 只作不同 q 的反事實分層；同一 raw order 的 makerFill label 只計算一次。

## Rank-group 結果

| q | Rank | raw N | Legacy EOD positive | Mixed-clock stop positive | Exact indexed full | Mixed precision | Mixed recall | Exact full 後 50ms hedge slip mean |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| 50 | B1–2 | 21,475 | 86.71% | 4.45% | 4.04% | 83.05% | 91.58% | 3.70bp |
| 50 | B3–5 | 7,504 | 80.25% | 1.25% | 1.04% | 71.28% | 85.90% | 10.67bp |
| 80 | B1–2 | 17,729 | 84.34% | 3.04% | 2.74% | 81.78% | 90.72% | 5.11bp |
| 80 | B3–5 | 12,454 | 80.08% | 0.78% | 0.68% | 75.26% | 85.88% | 11.85bp |
| 95 | B1–2 | 11,521 | 82.34% | 1.88% | 1.72% | 77.88% | 85.35% | 4.83bp |
| 95 | B3–5 | 19,017 | 76.77% | 0.53% | 0.37% | 62.00% | 87.32% | 13.43bp |

`Legacy EOD positive` 很高但不含策略撤單，不能當成交率。加入近似 stop 後雖然大幅收斂，仍有約 9–15% exact full 的 false negative，且 clock tail 很長，因此只適合 screening/一致性檢查。

## 方法契約

Legacy L1–5 index 完整複製既有 producer 的診斷語義：

- 先過濾 `marketOpen == true`，每商品依 `TransTime` 與原始 row ordinal 排序。
- trade-through 立即視為 legacy fill。
- 同價使用 producer 原式 `abs(FillPrice-target)<1e-8`，累積量達初始顯示量即 fill。
- fill seconds 轉成 Float32，L1–2 與現有 makerFill 做 bit-exact parity；32,688 筆 physical labels 的 mismatch 為 0。
- 這個規則沒有 own quantity、partial path、策略 stop、cancel ACK 或 exact receive cursor，因此所有 outcome/EV/joint-volume flags 都 fail closed。

## Lineage 與重現

正式 bundle：

`maker/data/walkforward/makerfill_rank_l1_l5_sample_20260820_v5`

其中 `input_inventory.parquet` 綁定 457 個來源：root manifest 1、execution actions 223、partition markers 223、tick files 5、makerFill files 5。Verifier 會重新驗 execution marker/config/50ms semantics、逐檔 hash/schema/rows，並由來源重算全部四張衍生表。

```bash
env UV_CACHE_DIR=/tmp/codex_uv_cache \
  uv run --no-project python -m maker.src.quote_fill.makerfill_rank_study_cli \
  --output maker/data/walkforward/makerfill_rank_l1_l5_sample_20260820_v5 \
  --verify-only
```

獨立稽核：13/13 targeted tests、74,150 randomized literal-producer parity labels、來源重算與多項 coordinated-tamper attacks 全部通過。

