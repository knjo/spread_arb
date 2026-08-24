# Expiry daily-close 與 opening-carry exit-only 結果

## 結論

到期 terminal 已改成同一到期日的兩腿 `close_price`：現貨取
`MarketInfo.twse_security_trades_daily.close_price`，期貨取
`MarketInfo.taifex_futures_trades_daily.close_price`（day session）。45／45 組商品契約
都有兩腿收盤價，171 筆 expiry paths 全部重定價；期貨 `settlement_price` 未使用。

這是到期強制平倉的收盤 mark，不代表收盤價存在可成交深度，也不是 maker fill 的證明。

## Close overlay 影響

| 母體 | v2 last-valid BBO／fallback gross | v3 paired close gross | Delta |
|---|---:|---:|---:|
| 171 expiry paths | -219,000 | 74,700 | +293,700 |
| 全部 3,672 paths | 6,731,550 | 7,025,250 | +293,700 |

全母體依正式成本設定後為 gross 7,025,250、成本 3,891,564、net 3,133,686 TWD，
名目加權 net 20.66 bp。原本 3374／QLFG6 的 5 筆 402／402 fallback 已改為
spot close 402、futures close 401.5。

## 13:00 normal carry cap 主表

這張表同時套用新規則：若 ValueCode 在 D 日開盤已有前日 carry，D 日整天該商品
exit-only，即使早盤已平完也不再進新倉。它與 D-safe 商品池 gate 分開判定。

| Cap | Accept | Exit-only rejects | Entry turnover | Peak / mean EOD | Gross / cost / net | Net bp | Loss trades / days | Realized MDD |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10M | 1,209 | 1,304 | 387.053M | 9.997M / 7.370M | 1,749,300 / 1,035,536 / 713,764 | 18.44 | 244 / 5 | 47,709 |
| 20M | 1,779 | 1,388 | 670.853M | 19.950M / 11.604M | 3,142,800 / 1,731,677 / 1,411,123 | 21.03 | 343 / 6 | 53,880 |
| 30M | 1,981 | 1,390 | 799.497M | 25.755M / 13.023M | 3,705,800 / 2,031,900 / 1,673,900 | 20.94 | 383 / 7 | 47,906 |
| 40M | 2,050 | 1,394 | 852.746M | 30.128M / 14.016M | 3,979,700 / 2,159,152 / 1,820,548 | 21.35 | 384 / 6 | 47,906 |
| 50M | 2,090 | 1,394 | 882.467M | 32.843M / 14.409M | 4,135,200 / 2,225,297 / 1,909,903 | 21.64 | 384 / 6 | 47,906 |

`Exit-only rejects` 是互斥 admission reason；每筆 event 另保留
`opening_carry_exit_only_blocked`，可核對與 cutoff／cap 同時成立的情況。

## 與舊 normal 結果的分解

| Cap | 只把 expiry 改 Close：net delta | 再套 opening-carry exit-only：net delta | 合計 delta |
|---:|---:|---:|---:|
| 10M | +88,999 | -65,312 | +23,687 |
| 20M | +130,686 | -211,118 | -80,432 |
| 30M | +148,443 | -638,166 | -489,723 |
| 40M | +243,177 | -823,040 | -579,864 |
| 50M | +303,025 | -910,767 | -607,742 |

Exit-only 確實壓低留倉與重複進場，但 30M 以上也明顯犧牲絕對獲利；容量增加後的
邊際效益在此規則下更早飽和。20M 主表 net 約 141.1 萬／63 sessions，平均約
2.24 萬／session，不是每日完整轉動 2,000 萬一次的結果。

## 最新月份與限制

2026-08 realized net：10M -25.4k、20M +12.1k、30M +17.7k、40M／50M 都約
+4.2k。最新一段仍明顯偏弱，不能稱為跨分段穩定。

其餘限制不變：1,090 筆 continuation 是每秒末狀態加 SpreadPair epoch 的近似；
unknown 被假設為完整 carry，仍可能有 double-exit bias；固定 45 檔 cohort 仍有回頭
篩選 data leak；MDD 只含每日已實現現金流，未含 carry 的每日 MTM。

## 可重現產物

- Close facts：`maker/data/walkforward/expiry_daily_close_facts_20260821_v1`
- Supplemental v3：`maker/data/walkforward/prequential_challenger_ab12_supplemental_carry_20260821_v3_expiry_daily_close`
- Normal cap：`maker/data/walkforward/normal_carry_cap_sweep_20260821_v2_expiry_close_exit_only`
- Supplemental `complete.json` SHA-256：
  `402d050e6fed75c38cea540fd31dd09994f6a24ef91a5411ea4e715e14cb691d`
- Supplemental paths SHA-256：
  `63742b773f32cdb4851fe686aea8c1dd8349d27b42be7460045dd73ec23cfda6`
- Normal `complete.json` SHA-256：
  `156715157532d99e75aa194ae5babd8a8e79d89c86aefd59b21a03569bd51c3c`
- Normal cap summary SHA-256：
  `4bc664221cc60a908a1a654d9d085bc72712501a72ad85318439f7ee80ade8f3`

