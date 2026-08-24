# FUTURE maker ASK1–ASK5 五日 indexed-truth 診斷（2026-08-21）

## 結論

FUTURE maker 的 ASK3–5 也沒有顯示值得納入主研究的成交優勢。以同一個 q 內的 physical raw order 為分母，ASK3–5 full-fill rate 比 ASK1–2 低：q50 低 36.4%、q80 低 44.1%、q95 低 50.8%。因此主研究維持 A/B1–2，A/B3–5 只留 analysis-only 診斷，仍是合理決策。

這份 ASK 結果完全使用現行 execution action/indexed replay 真值，沒有讀取、延伸或補值 legacy makerFill。FUTURE maker 的 policy start 在這個資料契約中由 anchor 或 spot 事件觸發，五日內沒有 `submit_event_sequence=1` 的 direct-future start；因此無法把 stock makerFill 的 snapshot mapping 當成 ASK 執行真值。

## 固定 cohort 與去重

- 日期與既有 BID 診斷完全相同：20260603、20260703、20260706、20260731、20260806。
- 223 個 complete product-days：45/44/44/45/45；3 個 q 後共有 669 product-day/q coverage cells，其中 637 格有 ASK1–5 action，32 格保留為零樣本。
- 只取 `future_ask_spot_taker / future / ask / ASK1…ASK5 / q50,80,95`。
- 162,429 個 q-specific physical orders；同 q 的 dedupe key 是 `(Date, ValueCode, boundary_quantile, raw_order_fact_id)`，正式資料沒有需要折疊的重複 alias。
- 跨 q 共有 111,204 個 unique physical raw orders；46,900 個 raw order 出現在兩個以上 q。因不同 q 的撤單時間與 outcome 可以不同，`physical_raw_inventory.parquet` 不替它任選一個 outcome，q 績效只在 `q_physical_orders.parquet` 計算。
- policy starts：anchor 3,535、spot 158,894、future 0。這與既有 BID 診斷的 direct-spot snapshot sampling contract 不同，所以兩 route 只並列表達，絕不 pooling，也不做 cross-route causal delta。

## ASK rank-group 結果

FUTURE maker quantity 是一口；因此 partial fill 為結構性 0，不是「沒有 partial risk」的策略結論。所有 full fill 的 50ms 對側 spot hedge 在此樣本皆 executable。

| q | ASK rank | physical N | full | full rate | partial | no-fill | 50ms spot hedge executable/full | hedge slip mean | p50 | p90 |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | ASK1–2 | 38,549 | 1,512 | 3.922% | 0 | 37,037 | 100.0% | 9.221bp | 0.000bp | 23.529bp |
| 50 | ASK3–5 | 12,625 | 315 | 2.495% | 0 | 12,310 | 100.0% | 9.088bp | 10.582bp | 23.041bp |
| 80 | ASK1–2 | 33,657 | 932 | 2.769% | 0 | 32,725 | 100.0% | 10.711bp | 11.001bp | 24.814bp |
| 80 | ASK3–5 | 21,119 | 327 | 1.548% | 0 | 20,792 | 100.0% | 10.090bp | 11.507bp | 21.834bp |
| 95 | ASK1–2 | 23,770 | 458 | 1.927% | 0 | 23,312 | 100.0% | 10.834bp | 10.616bp | 22.831bp |
| 95 | ASK3–5 | 32,709 | 310 | 0.948% | 0 | 32,399 | 100.0% | 12.744bp | 11.834bp | 23.753bp |

個別 ASK4/ASK5 的樣本組成與 ASK3 不相同，full rate 不保證隨 rank 單調；正確解讀是預先指定的 ASK1–2 與 ASK3–5 group comparison，而不是把 rank 當隨機分派的 treatment。

No-fill 最主要終止原因是 `target_retreat`，其次通常是 `target_not_passive`；q95 深層樣本的 `session_cutoff` 也較多。完整的 q/rank/outcome/nominal-stop/terminal-stop cells 在 `stop_reason_summary.parquet`，不需要重新 replay。

## 與既有 BID 診斷的對稱並列

下表只比較兩邊共同擁有的 exact indexed full-fill 與 50ms hedge 指標。ASK 使用 execution actions；BID 使用既有正式 bundle 的 indexed columns。兩 route 的 maker quantity、觸發事件與 sampling contract 不同，表內沒有 pooled denominator。

| q | Level band | ASK N | ASK full | ASK hedge mean | BID N | BID full | BID hedge mean |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 50 | L1–2 | 38,549 | 3.922% | 9.221bp | 21,475 | 4.037% | 3.697bp |
| 50 | L3–5 | 12,625 | 2.495% | 9.088bp | 7,504 | 1.039% | 10.667bp |
| 80 | L1–2 | 33,657 | 2.769% | 10.711bp | 17,729 | 2.736% | 5.109bp |
| 80 | L3–5 | 21,119 | 1.548% | 10.090bp | 12,454 | 0.683% | 11.851bp |
| 95 | L1–2 | 23,770 | 1.927% | 10.834bp | 11,521 | 1.719% | 4.833bp |
| 95 | L3–5 | 32,709 | 0.948% | 12.744bp | 19,017 | 0.373% | 13.430bp |

BID3–5 相對 BID1–2 的 full-rate 折損約 74–78%；ASK3–5 相對 ASK1–2 的折損約 36–51%。這個差異是描述性結果，不能解讀成 FUTURE 深層掛單優於 SPOT 深層掛單，因為 sampling contracts 不可交換。

## 50ms hedge 欄位的 fail-closed 處理

同一 physical raw order 可被多個 q policy 引用。正式 action facts 有 576 個 q no-fill rows 留著另一個 q/raw hedge lookup 的 payload，但 `entry_hedge_label_observed=false`、`entry_hedge_executable=false`。它們不是該 q 的可執行 hedge 樣本。

輸出保留這些 payload 供稽核，並明確標記 `unobserved_raw_hedge_payload_excluded=true`；所有 executable/slippage 統計只取 `hedge_metric_eligible = entry_hedge_label_observed & entry_hedge_executable`。沒有用 non-null price/slippage 欄位猜測成交。

## Bundle、lineage 與驗證

正式 bundle：

`maker/data/walkforward/future_ask_rank_l1_l5_indexed_sample_20260821_v1`

主要 artifacts：

- `q_physical_orders.parquet`：162,429 rows，q-specific exact outcomes。
- `physical_raw_inventory.parquet`：111,204 rows，一個 physical raw id 一列，不選 q outcome。
- `rank_summary.parquet`：126 cells；q50/80/95、by-date/all-date、individual/group。
- `stop_reason_summary.parquet`：310 exact stop cells。
- `coverage.parquet`：669 product-day/q cells。
- `symmetric_route_comparison.parquet`：12 route-separated ASK/BID cells。
- `input_inventory.parquet`：449 sources，包含 execution root manifest、223 action files、223 partition markers，以及既有 BID alias/marker。

`complete.json` SHA256：

`01ced6b2df837361b593f154538a15a2eb68a4904a502bdc6c4142c04873af33`

Config SHA256：

`00270a6839f02e65858fed0ee7f1666fed31073acd6040351d365bb0851c5afc`

Verifier 會核對 exact file set、hash、bytes、rows、schema、source inventory、execution partition marker/config/50ms semantics、implementation identity，並從全部來源重算八張表；只更新 artifact hash/schema 的 coordinated tamper 仍會在 semantic recomputation gate 失敗。

```bash
env UV_CACHE_DIR=/tmp/codex_uv_cache \
  uv run --no-project python -m maker.src.quote_fill.future_ask_rank_diagnostic_cli \
  --output maker/data/walkforward/future_ask_rank_l1_l5_indexed_sample_20260821_v1 \
  --verify-only
```

正式 publish unit `future-ask-rank-publish-v2-20260821` exit 0；獨立 verifier unit `future-ask-rank-verify-v3-telemetry-20260821` exit 0。Verifier cgroup peak 1,347,551,232 bytes（約 1.26GiB），swap peak 0，`memory.events` 的 high/max/oom/oom_kill/oom_group_kill 全為 0。

第一次 publish 嘗試 `future-ask-rank-publish-v1-20260821` 在原子 rename 前 fail closed：它正好抓到上述 576 個 unobserved raw hedge payload。失敗 stage 已清空；修正不是把它們視為可執行，而是保留 payload、另加 eligibility guard 並用對抗測試鎖定。

## 仍然不能宣稱的事

- 不是成交量分配：所有 outcomes 都是 independent-event labels，沒有 shared visible volume allocation。
- 不是 pathwise EV：沒有完整費稅、carry、maker exit 與 portfolio inventory replay。
- 不是 ASK/BID causal comparison：route sampling 不可交換。
- 不是 legacy makerFill extension：ASK 路線明確不讀 makerFill，也不建立 mixed-clock label。
- 不能因為一口 futures 沒有 partial 就忽略 spot hedge depth、庫存或隔夜風險。
