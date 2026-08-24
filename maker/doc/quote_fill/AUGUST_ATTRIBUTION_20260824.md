# S0：2026 年 8 月 q95 惡化歸因

日期：2026-08-24

狀態：**完成；canonical diagnostic bundle 已發布**

本文件只回答 S0：「2026-08-03～2026-08-13 的 q95 結果變差，主要是市場 excursion 不再碰到界線，還是碰到後的 queue／競爭成交率下降？」它不是 20M 策略回放，也不產生 PnL、hedge、exit 或部署結論。

## 結論

結論是 **兩者並列（`market_boundary_and_queue`）**，不是單一 boundary lag 問題：

- 以 2026-05～07 pooled 對照 2026-08，observable positive excursion 的 q95 touch rate 由 **7.1497% 降至 4.2923%**，下降 2.8574 個百分點／39.97%。
- 同一 q95、`cap=∞` quote-only actual-working order 分母中，碰到後的 approximate fill rate 由 **1.8796% 降至 1.2183%**，下降 0.6612 個百分點／35.18%。
- 若只為直觀而作 `touch rate × post-touch fill rate` 的二因子示意 bridge，proxy 由 0.13438% 降至 0.05229%；對兩個因子的先後順序取平均，代數貢獻為 market／boundary 53.9%、queue／competition 46.1%。但前者分母是 excursions、後者分母是 unique touched orders，bridge 沒有納入 touch-to-working-order coverage 或多對多 touch-order mapping，不能 reconcile actual fill count，也不能解讀為因果或經濟貢獻占比；正式分類只依「兩個 rate 都下降」判為兩者並列。
- Jul／Aug 固定共同 24 商品後，touch rate 仍由 **6.3697% 降至 4.0493%**，post-touch fill 仍由 **1.9735% 降至 0.9952%**；因此結果不是只由月池換成分造成。
- 30-session q95 challenger 只把 August hypothetical touch rate 從 **4.2923% 拉到 4.5043%**。同一 30-session 定義下，May～Jul pooled 仍為 **6.4131%**；共同 24 商品的 July／August 也仍是 **6.3725%／4.1990%**。縮短 lookback 沒有消除或反轉 August 落差，不升格為 S1 policy。

因此 S1 仍照已凍結的七組 `q50 / q80 / q95 / 15 / 20 / 25 / 30 bp` 全跑；S0 不提供改 grid 的理由。

## 固定範圍與口徑

- Universe：固定 causal manifest 3,886 product-days、72 entry sessions，2026-05-04～2026-08-13；所有結論都 conditional on common q95 spot-route-selected universe。
- August：只指 2026-08-03～2026-08-13 共 9 sessions，不外推為完整 8 月。
- Market primary denominator：完整 manifest 上可觀測的正向 residual excursions；第一個 eligible state 已在正側或資料 gap 後無法觀測起點者標 left-censored，排除 primary touch rate。
- `residual_excursion_bp = basis_mid_bp − anchor_ewma_120s_bp`。正向 excursion 自 `<= 0` 穿到 `> 0` 開始，到下一個 `<= 0` 結束；touch 是第一次由 `< D-1 q95 upper` 穿到 `>= upper` 的 raw cursor。
- Queue denominator：同一 q95 intent stream 經 spot `(t−1s,t] <= 100 requests` scheduler 後，first touch 當下滿足 `actual_new_send < touch <= active_end` 的 outcome-supported unique `raw_order_fact_id`。
- `post_touch_fill` 只接受 approximate fill cursor 位於 `(touch, active_end]`。`fill_cursor_exact=false`、不含 own quantity／partial／cancel ACK／joint volume，不能稱實盤成交真值。
- `diagnostic_quote_only=true`、`capital_cap_infinite=true`、`not_a_20m_strategy_replay=true`。

60-session boundary 的實際 history 並非每列都滿 60：

| 月份 | Global history sessions min–max | Product history sessions min–max |
|---|---:|---:|
| 2026-05 | 59–60 | 53–60 |
| 2026-06 | 60–60 | 47–60 |
| 2026-07 | 60–60 | 40–60 |
| 2026-08 | 60–60 | 47–60 |

2026-05-04 的 global history 是 59 sessions；報表沒有把它宣稱為滿 60。Boundary 仍使用凍結的 `lookback_sessions=60`／`min_history_sessions=40` 與嚴格 `<D`。

## 逐月主表

`residual` 與 `boundary` 三欄依序為 p50／p80／p95，單位 bp。`PD any` 是至少一次 primary touch 的 product-day 比例；`PD-equal fill` 是先算各 product-day rate 再等權平均。

| 月份 | Sessions | Product-days | Observable excursions | Obs excursions／PD | Residual p50／p80／p95 | D-1 q95 boundary p50／p80／p95 | Primary touches | Touch rate | Touches／PD | PD any | Supported touched orders | Post-touch fills | Pooled fill | PD-equal fill |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-05 | 20 | 765 | 446,467 | 583.617 | 6.566／15.202／31.689 | 23.848／28.109／38.238 | 38,148 | 8.5444% | 49.867 | 97.647% | 16,345 | 335 | 2.0496% | 2.3602% |
| 2026-06 | 21 | 1,478 | 668,923 | 452.587 | 6.778／15.654／31.730 | 27.039／34.046／39.853 | 48,890 | 7.3088% | 33.078 | 96.617% | 19,360 | 303 | 1.5651% | 1.5618% |
| 2026-07 | 22 | 1,218 | 659,560 | 541.511 | 6.297／14.650／29.521 | 24.088／30.200／41.531 | 39,866 | 6.0443% | 32.731 | 96.223% | 19,787 | 405 | 2.0468% | 2.2414% |
| 2026-08 | 9 | 425 | 252,778 | 594.772 | 5.523／11.970／23.295 | 23.311／30.783／39.216 | 10,850 | 4.2923% | 25.529 | 95.294% | 8,290 | 101 | 1.2183% | 1.1661% |

雙 panel 圖：[august_attribution_dual_panel.png](../../data/walkforward/august_attribution_s0_20260824_v2/august_attribution_dual_panel.png)

市場側不只是 touch count 受商品天數影響：August 的 residual p95 已降至 23.295 bp，而 D-1 q95 boundary 中位數仍為 23.311 bp；touches／product-day 也由 May 的 49.867、June 的 33.078、July 的 32.731 降到 25.529。

Queue 側不是單一 rank mix；各月 rank-level pooled／product-day 等權結果如下：

| 月份 | Rank | Supported touched orders | Post-touch fills | Pooled fill | PD-equal fill |
|---|---|---:|---:|---:|---:|
| 2026-05 | BID1 | 1,255 | 58 | 4.6215% | 4.2562% |
| 2026-05 | BID2 | 15,090 | 277 | 1.8357% | 2.2729% |
| 2026-06 | BID1 | 1,499 | 66 | 4.4029% | 3.6329% |
| 2026-06 | BID2 | 17,861 | 237 | 1.3269% | 1.4591% |
| 2026-07 | BID1 | 1,571 | 77 | 4.9013% | 4.5644% |
| 2026-07 | BID2 | 18,216 | 328 | 1.8006% | 1.9929% |
| 2026-08 | BID1 | 436 | 9 | 2.0642% | 2.1604% |
| 2026-08 | BID2 | 7,854 | 92 | 1.1714% | 1.0137% |

May～Jul pooled 後，BID1 是 `201 / 4,325 = 4.6474%`、BID2 是 `842 / 51,167 = 1.6456%`；兩個 rank 到 August 都下降。

## Censoring 與 membership audit

| 月份 | Session-start left-censored | Gap left-censored | 起點已達 upper | Primary PD any | 納入 left-censored sensitivity PD any |
|---|---:|---:|---:|---:|---:|
| 2026-05 | 316 | 12,883 | 4,343 | 97.647% | 97.778% |
| 2026-06 | 692 | 15,188 | 3,852 | 96.617% | 97.091% |
| 2026-07 | 575 | 8,453 | 1,148 | 96.223% | 96.470% |
| 2026-08 | 180 | 1,066 | 109 | 95.294% | 95.529% |

Left-censored excursions 沒有被臆造 first-touch cursor，也沒有進 queue mapping。Jul／Aug common-24 panel使用完全相同商品 membership：July 329,762 observable excursions／21,005 touches／14,441 supported touched orders／285 fills；August 138,320／5,601／4,421／44，方向與全池一致。

## 30-session boundary sensitivity

這個 challenger 只重算 D-1 q95 boundary 與「excursion amplitude 是否足以碰界線」；它沒有重建新的 touch cursor、scheduler、working order 或 makerFill，所以 `market_only=true`、`shortlist_eligible=false`。

| 月份 | Common-valid PD | Coverage | 60-session touch | 30-session hypothetical touch | 30 − 60 |
|---|---:|---:|---:|---:|---:|
| 2026-05 | 765 | 100.000% | 8.5444% | 7.6857% | −0.8587 pp |
| 2026-06 | 1,478 | 100.000% | 7.3088% | 6.0502% | −1.2586 pp |
| 2026-07 | 1,216 | 99.836% | 6.0208% | 5.9187% | −0.1021 pp |
| 2026-08 | 425 | 100.000% | 4.2923% | 4.5043% | +0.2120 pp |

兩個 July product-days 的 30-session boundary 無效；1,511 observable excursions 從兩個 arms 同時排除，沒有把 invalid 當零 touch。30-session window 嚴格使用 `<D` 最近 30 sessions，boundary rows 與 product-day touch counts 都由 verifier 逐列重建。

Challenger 凍結為 `lookback_sessions=30`、`min_history_sessions=20`、每側最少 20 個 excursion-history sessions／100 個 completed excursions。Global history各月都是30；product history在May／June／August為30，July為20–30，兩個invalid product-days仍保留在coverage。

## Scheduler 對帳與驗證

Quote-only adapter 從既有 intent stream讀到 653,098 個 source requests／326,549 個 source new；session 最後兩秒抑制 25 個 new 後，正式輸出為：

| 項目 | 數量 |
|---|---:|
| Actual new／raw orders | 326,524 |
| Actual cancel requests | 318,794 |
| Request assignments | 645,318 |
| Cutoff-drain cancels | 5,335 |
| Potential fill events | 280,828 |
| Accepted approximate fills | 7,730 |
| Effective actual cancels | 318,794 |
| Actual-new delays | 0 |
| 最大 rolling spot requests | 100／100 |

Accepted fill 會先讓 working order terminal，之後 cancel intent 成為 `cancel_not_needed`；共 7,730 筆。Raw replay 共重建 2,067,081 個 entry-primary-window excursions，其中 2,027,728 個 primary observable、39,353 個 left-censored；另有 304,369 touch-order pairs 與 63,820 unique touched raw orders。Formal verifier 逐列檢查 actual working interval、same-cursor phase、fill terminal、left censor、D-1 q95、manifest coverage 與 published summaries。

完整 raw venue state 保留 genuine L1 clear；正式 run 觀測到 spot 2,686,278、future 740,758 個 explicit L1-clear exposure rows。被本版取代的 `august_attribution_s0_20260824_v1` 曾逐 scalar forward-fill L1，會把 genuine clear 錯留成舊價，**不得引用 v1 數字**。

## Bundle、hash 與重現

Canonical S0 bundle：[`august_attribution_s0_20260824_v2`](../../data/walkforward/august_attribution_s0_20260824_v2/)

- Runner code commit：`f393ee154a9d6ac7325b3f79168203108c5eee8c`，run 時 `dirty=false`。
- `complete.json` SHA-256：`5bb3addbd674fc85162630a9f2a7033b1d52bec253dfbb698deabb19cdb3f28b`。
- Input inventory：784 個 full-content records，inventory SHA-256 `f7ab001199febbaa9de6a086bece09d380133a8d07953b1e892c0f6c66375f3f`。
- `market_excursions.parquet` SHA-256：`ed519c23e56285a21816a75812403681eac3f87a58541632a41d1fbd38e1f1eb`。
- `raw_order_facts.parquet` SHA-256：`4a6379ee99b3ff2240087d4ad74102274bd2ee8c62d7cd9205c81e282e1c54b2`。
- `request_assignments.parquet` SHA-256：`b00beae5ffa28a5505204fad6958cce36d1fa44479657208c26d0cb79c67db3c`。
- `product_day_coverage.parquet` SHA-256：`d7725420d4f6c5168e85bece56e76f2aa7887a8d613d87b02adc95dfdaeea412`。

30-session sensitivity bundle：[`august_attribution_s0_30_session_challenger_20260824_v1`](../../data/walkforward/august_attribution_s0_30_session_challenger_20260824_v1/)

- `complete.json` SHA-256：`680d68bf6cccb9f459b19ed199e82bb7b5c6f0a7bbffb3c468e42ad30e2b15ed`。
- Input inventory：660 個 full-content records，inventory SHA-256 `4a556db45bd146283580b5d905ae0a949741d7c0fe67b64ac9f24e981c183abb`。
- `rolling_30_q95_boundaries.parquet` SHA-256：`ba245c1f923aa3e76266e36143829ea5b566b41356bf2198943f0a773424d3fa`。
- `product_day_boundary_sensitivity.parquet` SHA-256：`63cd8e6660d23c3e628d815e762db95f77f5429e73e0f4bdf73a3e57e56382e7`。

在 nested repo root、上述 clean commit 與既定資料上，正式 build 使用：

```bash
/home/kevin/Project/HFT/.venv/bin/python3 \
  -m maker.src.quote_fill.august_attribution_runner

/home/kevin/Project/HFT/.venv/bin/python3 \
  -m maker.src.quote_fill.august_attribution_30_session_runner
```

已發布 bundle 的 full-content input 與 summary 重驗：

```bash
/home/kevin/Project/HFT/.venv/bin/python3 \
  -m maker.src.quote_fill.august_attribution_runner \
  --verify-only --verify-inputs

/home/kevin/Project/HFT/.venv/bin/python3 \
  -m maker.src.quote_fill.august_attribution_30_session_runner \
  --verify-only --verify-inputs
```

完整 quote-fill test suite 為 214 tests；兩個 canonical markers 內另保存各自 formal run 前執行的 focused tests 與 output tail。Raw-tape reconstruction 使用同一 production builder 做 row-for-row deterministic replay，因此證明輸入到輸出的可重現性；它不是另一套獨立演算法 oracle，語意正確性仍由 focused tests、invariants 與 code review共同支撐。

Provenance 限制：72 個 daily partitions 沿用 `migrated_nonatomic` legacy markers。Full-content SHA 綁定本次實際 consumed 的 causal／mapping／excursion artifacts，但不能倒推證明那些 legacy daily partitions 當初是 atomic publish；此限制同時寫入兩個 bundle 的 `run_config.json`。

## S1 handoff

S0 已交付可重用的 q95 actual-new／cancel scheduler core、working-order lifecycle、raw first-touch mapping與 verifier。S1 下一步是在同一 chronological event loop 擴成七組 `PolicySpec`、20M／單檔 50% reservation、B6 hedge／rollback 與 terminal feedback；不得把本 S0 的 7,730 approximate fills直接當 20M admission 或部署 baseline。
