# 45 商品 × 60 sessions Exit Maker：暫停點與研究摘要

> 更新時間：2026-08-18 13:32（Asia/Taipei）  
> 狀態：**正式 replay 主動暫停，可原地 resume；本文是 52.7% interim checkpoint，不是最終回測或 production EV。**

## 結論先講

目前 45 商品的結果仍支持一個清楚的 execution trade-off：

- `Center` 較容易在當日完成，但成功路徑的單次毛利較小；
- `Lower` 要求 basis 再收斂，完成率較低，但成功路徑毛利較大；
- q 越高，成功路徑的 gross 通常越高；
- 在目前已完成樣本中，q95 Lower 的成功路徑 gross p50 約 47～48 bp，統一減 19 bp 後約 28～29 bp，但當日 V0 完成率只有約 54～59%。

這些數字仍然**不能稱為每次掛單 EV**。最主要原因是 72.65% 的 policy paths 在 strict 口徑下仍屬 cancel-race unknown；跨日繼續掛同一 frozen Center／Lower Maker 的正式 terminal replay 也尚未接上，19 bp 只是研究敏感度而非完整費率。

## 暫停點與完整性

| 項目 | 當前狀態 |
|---|---:|
| Universe | 45 商品 × 60 sessions = 2,700 requested product-days |
| 可用 product-days | 2,687；13 個 D-safe mapping 缺口未補零 |
| Entry replay | 2,687／2,687 完成並全量稽核 |
| Same-day exit maker | **1,416／2,687（52.70%）** |
| 日期覆蓋 | 2026-05-20～2026-07-03；31 個完整交易日，加 2026-07-03 的 22／44 商品 |
| 最後完整 partition | `20260703 / 2610` |
| 第一個 pending partition | `20260703 / 2615` |
| 已深度稽核 | 1,416／1,416；每份 7 個 Parquet，SHA／bytes／rows／cols／schema／lineage 全過 |
| 未發布暫存 | 0；沒有 hidden stage、`.chunks` 或 incomplete partition |
| 正式 root | [exit_maker_narrow_60d](../../data/walkforward/exit_maker_narrow_60d)；約 8.3 GiB |
| 舊版基準 archive | [pre_lowmem_partial459](../../data/walkforward/exit_maker_narrow_60d_pre_lowmem_partial459_20260817)；459 份、唯讀保留 |

目前 1,416 份完成資料的實際列數：

| Artifact | Rows |
|---|---:|
| Audit | 1,416 |
| Policy support | 262,552 |
| Position-policy facts | 262,552 |
| Observations | 345,453,979 |
| Transitions | 281,213,221 |
| Canonical raw candidates | 73,144,965 |
| Candidate aliases | 107,357,652 |

分母核對為 65,638 個已建倉 entry q-aliases、48,840 個 partition-local physical entry positions、262,552 個 `Center/Lower × 兩條 exit route` alternative policy trials。每個 trial 是替代策略，不是可同時執行或相加的交易量。

## q、Center、Lower 到底代表什麼

- `q50/q80/q95` 是進場 Maker 的三個 D−1 rolling boundary alternative views；q95 通常掛得更有利、較難成交。
- `Frozen Center` 與 `Frozen Lower` 使用同一個已建倉部位，差別只在 exit basis threshold。
- `Lower = Center - D−1 lower distance`；threshold 在 entry policy 建立時凍結，隔日不重新估計。
- `future_bid_spot_taker`：期貨買回掛 Maker，成交後賣現貨 Taker。
- `spot_ask_future_taker`：現貨賣出掛 Maker，成交後買回期貨 Taker。
- Lower threshold 會讓期貨買價更低，或讓現貨賣價更高；因此通常犧牲完成率換取較大的成功路徑 edge。

## 目前 1,416 partitions 的同日結果

下表只納入**已實際 entry full-fill 且 50 ms entry hedge 可執行**的部位。`V0 same-day` 把 cancel request 視為立即生效；`gross` 使用實際四腿價格。`−19 bp` 只是統一 completed-cycle cost 敏感度，不能當正式 net P&L。

| q | Exit rule | Exit route | Trials | V0 same-day | Completion | Gross mean | Gross p50 | Gross p50 − 19 |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 50 | Center | Future Maker | 36,098 | 28,614 | 79.27% | 14.75 bp | 14.37 bp | -4.63 bp |
| 50 | Center | Spot Maker | 36,098 | 29,386 | 81.41% | 13.92 bp | 14.27 bp | -4.73 bp |
| 50 | Lower | Future Maker | 36,098 | 25,219 | 69.86% | 22.09 bp | 22.03 bp | +3.03 bp |
| 50 | Lower | Spot Maker | 36,098 | 25,662 | 71.09% | 20.73 bp | 22.22 bp | +3.22 bp |
| 80 | Center | Future Maker | 20,020 | 15,859 | 79.22% | 21.70 bp | 21.74 bp | +2.74 bp |
| 80 | Center | Spot Maker | 20,020 | 15,960 | 79.72% | 20.52 bp | 21.62 bp | +2.62 bp |
| 80 | Lower | Future Maker | 20,020 | 13,077 | 65.32% | 34.27 bp | 33.61 bp | +14.61 bp |
| 80 | Lower | Spot Maker | 20,020 | 12,609 | 62.98% | 32.66 bp | 33.28 bp | +14.28 bp |
| 95 | Center | Future Maker | 9,520 | 7,552 | 79.33% | 29.07 bp | 28.33 bp | +9.33 bp |
| 95 | Center | Spot Maker | 9,520 | 7,569 | 79.51% | 27.50 bp | 28.33 bp | +9.33 bp |
| 95 | Lower | Future Maker | 9,520 | 5,575 | 58.56% | 49.45 bp | 48.02 bp | +29.02 bp |
| 95 | Lower | Spot Maker | 9,520 | 5,137 | 53.96% | 46.47 bp | 46.95 bp | +27.95 bp |

最重要的解讀不是「q95 Lower 每次送單賺 29 bp」，而是：

1. 在 V0 假設下真正同日完成時，q95 Lower 的條件毛利較高；
2. 它有約 41～46% 的 paths 當日沒有完成；
3. 未完成 paths 的跨日 Center／Lower terminal cashflow尚未正式重播；
4. strict cancel race 與完整成本未定價，所以 unconditional EV 仍未知。

## Nominal 與 strict 為何差很多

262,552 條 alternative policy trials 的目前 branch 分布：

| Strict／terminal status | Paths | Share |
|---|---:|---:|
| Strict 可證明 `flat_same_day` | 1,465 | 0.558% |
| V0 同日完成、但 strict 為 `cancel_race_unknown` | 190,754 | 72.654% |
| `fill_unknown_at_eod` | 38,398 | 14.625% |
| `carry_at_eod_cancel_unconfirmed` | 31,047 | 11.825% |
| `partial_fill_carry_at_eod` | 336 | 0.128% |
| `hedge_incomplete_residual` | 334 | 0.127% |
| `carry_at_eod_no_admission` | 218 | 0.083% |

Nominal V0 的 same-day complete 為 192,219／262,552 = 73.21%，但其中 190,754 條沒有 cancel ACK 可以排除 sibling／prior-order race。這是目前不能把漂亮 gross 統計升格為 EV 的最大識別缺口。

## 隔日策略口徑

主策略不是隔天第一個 fresh joint book 直接 Taker/Taker 硬平。正式設計是：

1. 當日 Center／Lower Maker 沒完成的已建倉部位保留；
2. 隔日重新建立 DAY order queue，仍使用 entry 時凍結的同一 Center／Lower threshold；
3. 每日用當下 opposite-leg executable VWAP 重新解合法 target tick；
4. 成交後做同 route 的 50 ms taker hedge；
5. exact `QuoteCode` 不偷換月份，直到 Maker flat、到期、資料缺口或不可識別 terminal。

跨日 FSM、runner 與測試已存在，但 45 商品正式 cross-session replay 必須等 same-day 2,687 份完成後才啟動。舊文件中的 next-session Taker/Taker 是 forced-exit benchmark，已降級為壓力測試，不能代表主策略。

## 盤中 remaining-time × lookup 可不可行

**第一版高機率可行**的是 contextual fixed-policy lookup：在 entry 建倉或 position establishment 時，用當時可知的 state 查表選 `Center/Lower × exit route`，並承諾沿該 frozen policy 跑到 terminal。可用 state 包含：

- 剩餘交易時間；
- q、entry route、exit route；
- Center／Lower、target distance、rank、queue；
- DTE、持倉 session ordinal；
- causal book width／depth／freshness；
- D−1 liquidity peer 與 tick-ladder regime。

**尚不能直接宣稱可行**的是每幾秒任意切換 Center／Lower／route 的 Bellman controller。還缺共同 decision clock、完整 legal action set 與 `WAIT`、decision-to-next-decision transitions、實際 cancel ACK、完整 cost profile，以及用 frozen D−1 table 做 selected-action sequential replay。

因此後續順序應是：先完成 fixed-policy terminal labels與 prequential lookup，再用後續未看資料做 challenger；不能把同一 terminal P&L 貼回每個盤中 observation 當動態 Q-value。

## Replay 工程與可重現性

舊 in-memory replay 在 `20260603/2324` 曾到 110.8 GiB。現在正式版本把四張大表分塊寫入磁碟，再排序合併：

- full extreme canary：44:24，peak 14.62 GiB、swap 0；
- 7 個 artifact 與未設定 allocator decay 的 v2 canary byte-for-byte SHA 相同；
- 正式 full-day batch 最重完成峰值約 36.01 GiB，只有 MemoryHigh reclaim，`max/oom/oom_kill=0`；
- 368／368 tests 通過；
- runner version：`exit_maker_product_day_v4_spooled_decay100ms_launch_guard`；
- runner SHA：`580c713ac0c05411841d0aaaebc6d94de258cc1c31b0c16f0f2457c10a736a43`；
- study SHA：`45d748344cd596871ef4bb6b4187029fed4e1342215d8799e0c25c05960d6b6c`；
- allocator guard SHA：`fc467abca5147f636eafc9a04c08952aae2afb7e5da88a060a152cc83146fd71`。

正式啟動契約要求 process-start：

```text
_RJEM_MALLOC_CONF=background_thread:true,dirty_decay_ms:100,muzzy_decay_ms:100
```

CLI 會在 discovery、建立輸出或 raw I/O 前 fail closed；晚設或偽造 expanded runtime value 也會拒絕。

## 如何恢復

暫停是刻意送出 SIGTERM，因此 transient unit 顯示 `Result=exit-code / status=143`；這不是資料錯誤。原 unit、ExecStart、allocator env、MemoryHigh 36 GiB、MemoryMax 48 GiB、MemorySwapMax 0 都仍保留。

恢復只需：

```bash
systemctl --user start exit-maker-v4-firstwave45-20260818.service
```

啟動後會先驗證既有 1,416 partitions，所以短時間沒有新 marker 是正常；第一個真正重算的 partition 固定是 `20260703/2615`。整批完成後才會建立 root-level `exit_maker_partition_manifest.parquet`，在那之前 downstream report／cross-session runner 不應啟動。

後續工作順序：

1. resume 並完成剩餘 1,271 個 same-day partitions；
2. 2,687 份全量 hash／schema／denominator audit與 human-readable report；
3. 正式跨日 frozen Center／Lower continuation；
4. filled-entry-only primary report；
5. D-safe remaining-time contextual lookup、support／LCB／readiness audit；
6. 最終 45 商品結果文件，明確分開 gross、cost sensitivity、unknown 補零與真正 EV。

## 相關文件

- 舊五商品完整 checkpoint：[EXIT_MAKER_60D_RESULTS.md](EXIT_MAKER_60D_RESULTS.md)
- Liquidity／universe 定義：[LIQUIDITY_SCREEN.md](LIQUIDITY_SCREEN.md)
- EV 資料契約：[EV_LOOKUP.md](EV_LOOKUP.md)
- Replay sampling 與撤單口徑：[REPLAY_SAMPLING.md](REPLAY_SAMPLING.md)

