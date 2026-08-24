# 剩餘盤中時間 × nominal stop 診斷表（2026-08-21）

## 結論

Gap1 已完成並正式發布。來源是已凍結的 60-session
`execution_action_facts.parquet`，不重開 raw tape、不重算 fill，也不改寫既有
execution／same-day／cross／post-cross roots。正式表完整覆蓋：

```text
q50/q80/q95
× 2 routes 各自合法的 ASK1/ASK2 或 BID1/BID2
× 5 個送單時剩餘 session 時間桶
× target_retreat / gate / cutoff / other
= 240 cells
```

180 cells 有支援分母；60 個 `other` cells 是預留的固定零格。來源中所有非
retreat／cutoff reason 都是八個已凍結的 gate-invalid reasons，任何新值或 null
都會 fail closed。

這是一張回溯診斷表，不是可直接上線的盤中 feature 表。`nominal_stop_reason`
是送單後才知道的 first-passage 標籤；用它查當下 action 會造成 future leakage。
正式 marker 因此固定：

```text
realized_nominal_stop_reason_future_label = true
online_stop_reason_feature_ready = false
pathwise_ev_ready = false
production_strategy_go = false
```

盤中使用時只能保留「剩餘時間」並把 nominal stop reason 邊際化，且仍需另外
接完整 cost／EV／position control。

## 固定口徑

剩餘時間不是掛單的事後存活時間，而是：

```text
13:20:00 Asia/Taipei session cutoff recv ns - submit_recv_time_ns
```

cutoff 對 submit 是 exclusive；合法範圍為 `(0, 255 minutes]`。時間桶以整數
nanosecond 比較：

| bucket | 範圍 |
|---|---|
| `lt_5m` | `[0, 5m)`；實際 submit 必須大於 0 remaining |
| `5_to_lt_15m` | `[5m, 15m)` |
| `15_to_lt_30m` | `[15m, 30m)` |
| `30_to_lt_60m` | `[30m, 60m)` |
| `ge_60m` | `[60m, 255m]` |

Stop 分類是固定 allowlist：

- `target_retreat`：`target_retreat`
- `cutoff`：`session_cutoff`
- `gate`：`target_not_passive`、`missing_anchor`、`spot_trial_match`、
  `spot_ref_gate`、`future_book_gate`、`future_ref_gate`、
  `future_exec_book_gate`、`target_ref_gate`
- `other`：本版無允許值；只保留固定零格，不用來吞未知值

支援分母只接受完整 route／market／side／rank identity：

- `future_ask_spot_taker / future / ask / ASK1|ASK2`
- `spot_bid_future_taker / spot / bid / BID1|BID2`

A/B3–5、inside 等 unsupported rows 只留在 product-day audit，絕不當成 no-fill。
所有 rate 與 wait quantile 都以支援的 policy alias row 加權；physical key 是
`raw_order_fact_id`。表同時列 alias 與 within-cell unique physical counts，但
physical counts 不可跨 q／route／rank／stop cells 相加。

## 全量 inventory

| 指標 | rows |
|---|---:|
| source partitions | 2,687 |
| source actions | 4,032,586 |
| supported A/B1–2 policy aliases | 2,320,344 |
| unsupported、未插補 | 1,712,242 |
| full fill | 53,640 |
| partial-only | 1,610 |
| no fill | 2,265,094 |
| any fill | 55,250 |
| cancel required | 2,266,704 |
| full-fill + 50ms hedge observed | 53,640 |
| 50ms hedge executable | 53,519 |

恆等式逐 cell 與全表皆成立：

```text
full + partial-only + no-fill = supported
any-fill = full + partial-only
cancel-required = partial-only + no-fill
```

63 個 typed-empty product-days 保留為零提交；它們不是 no-fill。2,687-row audit
的 safety failure 為 0，實際 remaining 範圍為 0.019232883 秒到
15,299.90619 秒。

## q／route／rank 描述性結果

下表只在每一列內，把互斥的時間／nominal-stop alias labels 做 inventory
邊際化；每列仍是獨立 counterfactual，列與列之間不可加總或當 portfolio：

| q | route/rank | aliases | full | partial | full % | any-fill % | hedge executable / full |
|---:|---|---:|---:|---:|---:|---:|---:|
| 50 | future ask ASK1 | 201,663 | 8,904 | 0 | 4.4153% | 4.4153% | 99.9438% |
| 50 | future ask ASK2 | 246,821 | 5,655 | 0 | 2.2911% | 2.2911% | 99.9293% |
| 50 | spot bid BID1 | 228,189 | 11,385 | 854 | 4.9893% | 5.3635% | 99.6750% |
| 50 | spot bid BID2 | 236,012 | 4,019 | 119 | 1.7029% | 1.7533% | 99.5770% |
| 80 | future ask ASK1 | 141,070 | 4,227 | 0 | 2.9964% | 2.9964% | 99.9054% |
| 80 | future ask ASK2 | 254,187 | 4,054 | 0 | 1.5949% | 1.5949% | 99.9260% |
| 80 | spot bid BID1 | 161,674 | 5,505 | 407 | 3.4050% | 3.6567% | 99.5095% |
| 80 | spot bid BID2 | 256,389 | 2,884 | 71 | 1.1249% | 1.1525% | 99.4452% |
| 95 | future ask ASK1 | 77,661 | 1,576 | 0 | 2.0293% | 2.0293% | 100.0000% |
| 95 | future ask ASK2 | 206,541 | 1,893 | 0 | 0.9165% | 0.9165% | 100.0000% |
| 95 | spot bid BID1 | 83,552 | 2,001 | 126 | 2.3949% | 2.5457% | 99.7501% |
| 95 | spot bid BID2 | 226,585 | 1,537 | 33 | 0.6783% | 0.6929% | 99.8048% |

q50 在四個 route/rank 都有較高的 descriptive fill rate；第一檔也普遍高於
第二檔。這只是各 policy alias 的 realized label 分布，不能據此宣稱 q50 的
net EV 較佳。

剩餘時間的方向不是所有 cell 都一致。例如，把 nominal-stop labels 只作
alias inventory 邊際化後：

- q50 future ASK1 full rate 從 `<5m` 的 2.9345% 到 `60m+` 的 4.5895%。
- q50 spot BID1 則從 `<5m` 的 7.2965% 到 `60m+` 的 4.8151%。
- q95 spot BID2 在五桶依序為 0.8971%、0.7373%、1.2551%、0.4723%、
  0.6704%，並非單調函數。

因此剩餘時間有區分力，但不能用單一「愈早掛愈容易成交」規則取代查表。

## Nominal-stop 與等待時間

全 alias inventory 的回溯分布：

| realized nominal class | aliases | full | partial | no-fill | full rate |
|---|---:|---:|---:|---:|---:|
| target retreat | 1,940,351 | 32,881 | 1,072 | 1,906,398 | 1.6946% |
| gate invalid | 332,738 | 20,356 | 533 | 311,849 | 6.1177% |
| session cutoff | 47,255 | 403 | 5 | 46,847 | 0.8528% |
| other | 0 | 0 | 0 | 0 | — |

`gate` 的較高 realized full rate 是典型的事後條件化結果，不能當盤中已知
predictor。主表保留每個 exact cell 的 first-fill 與 full-fill wait
`p50/p90/p95`（nearest interpolation），無成交 cell 的 quantile 是 null。

一個有足夠分母的具體 cell：`q50 / spot BID1 / 30–60m / gate` 有 2,077
aliases、245 full、13 partial，full rate 11.7959%；first-fill wait p50/p90 為
13.974／137.461 秒，full-fill wait p50/p90 為 16.189／143.248 秒。這同樣只能
視為 realized-label diagnostic。

## 正式 bundle 與驗證

路徑：

`maker/data/walkforward/compact_remaining_time_stop_reason_60d_v1`

Exact inventory：

- `remaining_time_stop_reason_summary.parquet`：240 × 52，SHA-256
  `a6d93270922426b7fd4e6d12e8a51a5dfe0cd18332dd5dcf10d3de02b1933105`
- `product_day_audit.parquet`：2,687 × 24，SHA-256
  `b1274a7321dd0e39888aeeebf9e2214a9ac5bca83796b4b3416f8aa204860cfa`
- `complete.json`：SHA-256
  `f76dc38ec6cf0b28346b85be97b0f2c22c19af825f07675fc024122e40313b82`

來源鎖定：

- execution manifest SHA-256：
  `ba8436a6a085fe8b8716d67294e9202e9d7fe2ba4a874245dd13d1b9a64efe04`
- 2,687-partition marker/action inventory digest：
  `476ddb144118ae3696798dc5b9383c32c1ae85d5e9f5a85bc4f4d7d28a6030eb`
- analysis implementation SHA-256：
  `c68d74aec9d2d483f70d9fec3b791c44309b47125621f294c6f8e064d14d661c`
- frozen compact dependency SHA-256：
  `a027637dc7c7479c29ba20d5196f45e4fb83fa0d21889dc36af3b01c981fd23a`

Verifier 會重驗所有 partition marker/action hash/config/schema、拒絕 symlink／root
escape、重算兩張表，並逐 ordered schema/dtype/value 比對：

```bash
UV_CACHE_DIR=/tmp/uv-compact-remaining-v1 \
MPLCONFIGDIR=/tmp/mpl-compact-remaining-v1 \
POLARS_MAX_THREADS=4 \
uv run --no-project python -m maker.src.quote_fill.compact_remaining_time \
  --verify-only \
  maker/data/walkforward/compact_remaining_time_stop_reason_60d_v1
```

Formal build unit：

- `compact-remaining-time-v1-formal-20260821.service`
- invocation `437fd9b41f1a4680892889d470cc419f`
- exit 0、Restart 0、swap peak 0
- observed MemoryPeak `3,479,400,448` bytes（約 3.24 GiB），低於 4 GiB
  驗收線與 8 GiB `memory.high`

獨立 source-rebuild verify unit：

- `compact-remaining-time-v1-verify-20260821.service`
- invocation `aec3227075214373a6cf972c65262c66`
- exit 0、swap peak 0、journal peak 253.4 MiB

測試：新模組 15/15、compact targeted 48/48、quote-fill full suite 466/466；
真實 bounded smoke `20260714/1513` 為 408 source rows、294 supported、2 full、
0 partial、292 no-fill，240-cell schema 與 product-day audit 均通過。

