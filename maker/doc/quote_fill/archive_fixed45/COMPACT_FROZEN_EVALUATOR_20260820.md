# Compact frozen evaluator v1（2026-08-20）

## 結論

固定的 q50/q80/q95 已有一條可用的快速查表路徑：直接投影已凍結的
`execution_action_facts.parquet`，不重新打開 spot/futures raw tape，也不重算
first-passage、fill 或 full-fill + 50ms hedge。正式 analysis-only v2 bundle 已以
4,032,586 筆 action facts 全量重算驗證通過。

這份交付目前可用於 entry 掛單的 fill／撤單／等待時間／hedge 可執行性查表；
它不是完整 EV 表，也尚未包含來源綁定的實際 exit 或 joint FIFO 結果。

## 支援範圍與分母

- 支援 route/market/side/rank 必須完整匹配：
  - `future_ask_spot_taker / future / ask / ASK1|ASK2`
  - `spot_bid_future_taker / spot / bid / BID1|BID2`
- q 僅為 50、80、95。
- A/B3–5、inside 或 route/market/side 不一致的資料保留在 audit universe，
  但 `outcome_supported=false`，所有 outcome truth 為 null/false，絕不當成
  no-fill 或 cancel 放進支援分母。
- 表內同時提供 policy alias 與每個 route/q/rank cell 的 unique physical order
  欄位；跨 policy 的 `raw_order_fact_id` 不可重複當成多張實際訂單。

全 60-session 統計：

| 指標 | policy aliases | 支援分母比率 |
|---|---:|---:|
| 全候選 | 4,032,586 | — |
| 支援 L1/L2 | 2,320,344 | 100.0000% |
| 不支援、保留為 unknown | 1,712,242 | — |
| full fill | 53,640 | 2.3117% |
| partial-only | 1,610 | 0.0694% |
| no fill | 2,265,094 | 97.6189% |
| any fill | 55,250 | 2.3811% |
| cancel required | 2,266,704 | 97.6883% |
| hedge label observed | 53,640 | full fill 的 100% |
| hedge executable | 53,519 | full fill 的 99.7744% |

分類恆等式是：

```text
full + partial-only + no-fill = supported
any-fill = full + partial-only
cancel-required = partial-only + no-fill
```

因此 `cancel-required` 包含 partial-only；它不能再和 partial 相加解讀成互斥
結果。

## 效能與 parity

代表分區為 `20260701 / 2301`：755 筆候選中 539 筆屬支援分母，4 筆
full、0 筆 partial、535 筆 no-fill。對既有 frozen source 的 lifecycle、stop、
fill cursor、quantity 與 hedge 欄位為 539/539 exact parity。

同一個發布程序中的效能觀測：

| 範圍 | wall time |
|---|---:|
| selected product-day frozen fast path（含 root inventory gate） | 0.0936 s |
| 同 product-day raw merged loader 比較 | 13.5971 s |
| 觀測比值 | 145.23x |
| 全 2,687 partitions end-to-end verified summary | 19.2773 s |
| 其中 source marker/hash/semantic verification | 15.8220 s |
| 其中 4,032,586 rows projection/validation/aggregate | 3.4553 s |

這些 timing/RSS 是不可重算的 benchmark observation，不是資料語義 claim；
verifier 只檢查欄位、有限非負值與比值算術。該程序 peak RSS 為 4,215,896
KiB，包含 raw-loader comparison；另一次不開 raw loader 的全量入口 peak RSS
為 3,422,416 KiB。

## 來源鎖定與正式 bundle

正式 analysis-only bundle：

`maker/data/walkforward/compact_frozen_benchmark_v2_20260820_2301`

只發布三個 artifacts 加一個 completion marker：

- `full_q_summary.parquet`：42 rows，SHA-256
  `e1bdeaf55a3c2309aa5e734c53f9d0755ac309a906aa41f475371e1abae4288a`
- `full_product_day_audit.parquet`：15,739 rows，SHA-256
  `01241c039f8f420e18a594df03fb5d4245e6041ce796fef592f930a691d57971`
- `benchmark.json`：SHA-256
  `e69cb87c71ba4bb6f15d139f7a96e96abcd7fe647f6e1401c6750a6eba5a5e31`
- `complete.json`：宣告 exact artifact inventory、ordered schema/dtype/rows/bytes/hash、
  implementation hashes 及所有 non-readiness flags。

來源硬鎖包括：

- execution manifest SHA-256：
  `ba8436a6a085fe8b8716d67294e9202e9d7fe2ba4a874245dd13d1b9a64efe04`
- 2,687 個 `(marker SHA, action SHA, rows, columns, bytes, config, runner)`
  canonical inventory digest：
  `476ddb144118ae3696798dc5b9383c32c1ae85d5e9f5a85bc4f4d7d28a6030eb`

verify-only 會重新驗證完整 root inventory、逐 partition action bytes/hash/contract，
再從 source 重算兩張表並做 ordered schema/dtype/value equality；只同步修改 action
與 partition marker 也無法繞過硬鎖 inventory digest。

```bash
uv run --no-project python -m maker.src.quote_fill.compact_frozen_benchmark \
  --verify-only maker/data/walkforward/compact_frozen_benchmark_v2_20260820_2301
```

舊目錄 `compact_frozen_benchmark_20260820_2301` 是已拒絕的 v1 candidate，
verifier 無法完整重算其 artifacts，不能視為交付結果。

## Exit 與 FIFO 邊界

`compact_exit.py` 目前只提供已測試的 contract API：

- policy row key：`close_opportunity_id`
- logical position key：`(scenario_id, physical_exit_order_id, position_id)`
- physical order key：`(scenario_id, physical_exit_order_id)`

builder 會先折疊 policy aliases，要求 logical position quantity 合計等於 physical
order quantity，並驗證完整 EventCursor、active interval 與 full-fill + 50ms hedge。
但目前沒有綁到 immutable position/exit facts 的 adapter，也沒有可證明
post-external-queue residual 的真實 capacity producer。因此：

- `allocate_fifo_exact_capacity()` 固定 fail closed。
- simulator 僅為 `experimental_contract_only=true` 的 synthetic contract test。
- `capacity_provenance_verified=false`、`joint_volume_allocated=false`、
  `actual_fifo_metrics_absent=true`。
- 正式 bundle 不含實際平倉比例、terminal-ready 或 FIFO metrics。

## 尚未完成

- arbitrary-q first-passage core 已存在，但目前 public evaluator 仍需呼叫既有 raw
  loader，不能宣稱與 frozen fixed-q path 相同的 wall-time 加速。
- makerFill adapter 的 snapshot/rank mapping 可精確，legacy FillSeconds outcome
  仍是 approximate；它不會被標成 observed/exact/pathwise。
- 真實 exit 1:1/FIFO、position source FK、joint volume allocation 與完整 EV lookup
  仍需在來源綁定 producer 完成後另行發布。
- `pathwise_ev_ready=false`；本 bundle 不可直接當作最終盤中 EV 決策表。

測試基線：

```bash
uv run --no-project python -m unittest \
  maker.src.tests.test_quote_fill_compact_evaluator \
  maker.src.tests.test_quote_fill_layered \
  maker.src.tests.test_quote_fill_engine -v

uv run --no-project python -m unittest discover \
  -s maker/src/tests -t . -p 'test_quote_fill*.py'
```

本次結果為 targeted 48/48、quote-fill full suite 401/401。
