# S0.5 重作：盤中 anchor、D−1 q 查表與 S1 共同母體

日期：2026-08-26

狀態：**Anchor／entry-q／S1共同母體已完成；舊 convergence 已隔離，frozen-at-touch v2 supplement 待正式發布**

本次不是把 2026-08-25 的 S0.5 結果換一個 anchor 後直接沿用，而是從同一因果原始事實重新做完整選型：先比較盤中 anchor，再用勝出 anchor 重建 excursion、D−1 q lookup、q-independent cohort 與七組已知成本幾何。初版 convergence 事後發現逐秒重算 anchor，與 A2 submit 後凍結絕對 exit 價不一致，故其結果只留作 dynamic sensitivity；正式 conditional convergence 另以 frozen-at-upper-touch proxy 重算。2026-08-14 起的資料被鎖為 protected forward，整次 build 都沒有讀取。

## 結論

S0.5 的 anchor、entry-q 與 common mother 已可作為 **可信、可重現的 S1 研究基礎**；normal-exit lower 尚待 frozen v2 完成：

1. 盤中中心採 `time_ewma_15s`。它只用當下以前的合法期現 book，會隨盤中狀態更新，不是盤前估一個中心後一路帶到收盤。
2. D 日 q 距離採 `Q2_trail20_date_equal`。每個 D 日預測嚴格只用 `<D` 的最近 20 sessions；`Q2_trail20_date_equal__tod10` 留作診斷。20／60 sessions、5／10 日 level scale、prior-expiry／DTE 都已在同一 common support 比過，沒有比簡單 trail20 更好。
3. S1 primary 母體固定為 15,638 product-days、71 sessions、244 商品。它不使用 target-day touch、fill、PnL 或舊 q95 monthly selector，七組 policy 將使用同一母體。
4. D−1 表對隔日的商品間排序有實質預測力；同一商品跨日的 temporal ranking 很弱。q 名稱是距離級別，不能被解讀成每天固定的 50%／20%／5% 機率。
5. Fixed15–30 的對稱幾何已完整；q50／q80／q95 原表的 lower 則是未條件化的 marginal negative-q，也就是已被 convergence 否定為 primary 的 C1 control。它只能保留作兩側 marginal reference，不能拿原 q80／q95 margin 宣稱 round trip 有利潤。

S1 round-trip 前仍需先取得 frozen v2 的 C0／C2／C3 reach、coverage與成本幾何，再由使用者凍結唯一 normal-exit lower scheme及unsupported lookup行為。C0是正常 maker＋taker的center exit；C1是未作conditional calibration的wide control，不是默認fallback。正式數字出來前，本報告不沿用舊moving-anchor排名，也不偷選winner或fallback。

## 因果範圍與選型契約

- Source sessions：2026-01-26～2026-08-13，共 131 日。
- Primary development sessions：2026-05-05～2026-08-13，共 71 日；5／6／7／8 月各自等權。
- Protected forward：2026-08-14 起，runner 以明名 session 清單隔離，未讀取。
- Anchor primary label：同一合法 freshness gate 下的未來 30～300 秒 basis median；每 product-day 先按一秒 occupancy 聚合，再 product-day equal、month equal。
- Anchor uncertainty：5-session paired moving whole-Date block bootstrap，5,000 replicates；秒級資料不當 iid。
- Boundary prediction：每個 D 日只准使用 `source_asof_date < D`；candidate selection 的結果則明標使用 71 日 development outcomes，因此不是 untouched forward 結論。
- Excursion 與 convergence 都分開記 hit、known miss、right-censored unknown；left-censored episode 不進 primary。
- 所有 final geometry 固定 `reference_diagnostic=true`、`actionable_execution=false`、`ev_ready=false`。

## 1. 盤中 anchor：15 秒 wall-clock EWMA 勝出

主比較只在六個 actionable wall-clock candidates 的共同一秒 support 上進行。MAE 越低越好；TV ratio 越低代表 anchor 越平滑。

| Rank | Anchor | Month-equal MAE | TV／basis TV | 相對 15s MAE | Paired 95% CI |
|---:|---|---:|---:|---:|---:|
| 1 | `time_ewma_15s` | **8.7221 bp** | 0.3072 | 0 | — |
| 2 | `time_ewma_30s` | 8.7864 | 0.2186 | +0.0643 | `[+0.0454, +0.0717]` |
| 3 | `time_median_60s` | 8.9041 | 0.2398 | +0.1820 | `[+0.1707, +0.2043]` |
| 4 | `time_ewma_60s` | 9.0202 | 0.1504 | +0.2981 | `[+0.2547, +0.3131]` |
| 5 | `time_median_120s` | 9.0690 | 0.1665 | +0.3469 | `[+0.3196, +0.3824]` |
| 6 | `time_ewma_120s` | 9.5992 | 0.1010 | +0.8770 | `[+0.7944, +0.8956]` |

15s 的優勢很小但方向一致，30s 不在 paired one-SE 內，因此按預先凍結規則選 15s。30s 保留 rank-2 diagnostic；`count_ewma_120obs` 只作 observation-count control，不進 primary q 選型。Primary scoring rectangle 有 16,509 product-days；15s／30s 的 within-product-day p95 error month-equal mean 分別為 28.808／28.419 bp，表示 30s 的 tail 反而略好。30～300 秒 label coverage 為 98.656%，common／label coverage 為 100%。

Horizon sensitivity 也限制了結論外推：10～60 秒仍由 15s 勝出（MAE 6.130 bp），300～900 秒則改由 60s 勝出（12.585 bp；15s 為 12.831 bp）。所以 15s 是預先註冊的短期中心目標勝者，不是跨 5～15 分鐘都成立的 latent fair value。

這裡的語意要分清楚：

- **盤中 anchor**：D 日用截至該 cursor 的合法資料動態更新，回答「現在合理的中價差在哪裡」。
- **盤前 q lookup**：D 日開盤前已由 `<D` 歷史算好，回答「離當下 anchor 多遠才算該商品當日的 q50／q80／q95 距離」。

因此新版不是把 D−1 中價差凍住整天；盤前凍住的是 distance lookup。每張實際單在 submit 時，以當下 causal anchor ± distance 換成絕對價格並鎖定；anchor之後如何漂移都不能自行製造exit hit。S0.5尚無真實submit tick，因此使用「first legal upper touch當下凍結anchor」作明示proxy。

## 2. D−1 q lookup：trail20 勝出，TOD scale 沒有淨改善

七個 selectable finalists 在完全相同的 362,739 product-day×TOD×q×side common units 上重排。`primary loss` 是 censor-identified reach interval 到名目 tail probability 的距離，越低越好。

| Rank | Candidate | Primary loss | Worst-month loss | Amplitude MAE | Native all-q coverage | Daily cross-product ρ | Temporal product ρ | Turnover |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `Q2_trail20_date_equal` | **0.1507** | 0.1070 | **9.0420 bp** | 94.47% | **0.6255** | 0.1424 | 0.00794 |
| 2 | `Q2_trail20_date_equal__tod10` | 0.1566 | **0.1060** | 9.0882 | 94.47% | 0.6255 | **0.1499** | 0.01091 |
| 3 | `Q1_trail60_date_equal` | 0.1674 | 0.1194 | 9.3916 | 96.86% | 0.5817 | -0.0009 | **0.00320** |
| 4 | `Q1_trail60_date_equal__tod10` | 0.1704 | 0.1175 | 9.4367 | 96.86% | 0.5817 | 0.0232 | 0.00666 |
| 5 | `Q3_shape60_level5` | 0.1709 | 0.1180 | 9.4983 | 96.86% | 0.5817 | 0.0340 | 0.01154 |
| 6 | `Q4_shape60_level10` | 0.1714 | 0.1179 | 9.5246 | 96.86% | 0.5817 | 0.0222 | 0.00652 |
| 7 | `Q6_prev2_expiry_dte5` | 0.1775 | 0.1291 | 9.9729 | 83.44% | 0.5373 | -0.0196 | 0.01590 |

結論不是「短窗永遠比較好」，而是這 71 日 development panel 上，trail20 同時改善 calibration loss、amplitude error 與跨商品排序。TOD10 只小幅改善 worst-month loss，卻讓整體 loss、amplitude error 與 turnover 變差；5／10 日 global level scale 和 prior-expiry／DTE 都沒有贏過 trail20。

Q2 在 exact common support 上的 Date-equal calibration如下；LB／UB 保留 right-censor uncertainty：

| q | Side | Reach LB–UB | D-equal predicted distance |
|---:|---|---:|---:|
| q50 | Negative | 50.007–50.944% | 9.572 bp |
| q50 | Positive | 50.004–50.926% | 9.785 bp |
| q80 | Negative | 19.291–20.995% | 17.699 bp |
| q80 | Positive | 19.209–20.867% | 18.502 bp |
| q95 | Negative | 4.931–7.108% | 35.898 bp |
| q95 | Positive | 4.832–6.920% | 37.448 bp |

平均 reach 已比舊 rolling-60 更貼近名目值，但 cell-level interval loss 仍不小，且 8 月 q80／q95 的 lower bound 又降到約 17.0–17.3%／4.0–4.1%。因此「改善」不等於已完成穩定的 probability calibration。

對使用方式的限制：

- `daily cross-product ρ=0.6255` 表示 D−1 表很能排序「明天哪些商品的 excursion 較大」。
- `temporal product ρ=0.1424` 表示它對「同一商品明天相對自己歷史會多大」只有較弱的預測力。
- 所以表適合做距離、排序與預註冊 subgroup；不應單靠 q 名稱或 `q95 − lower − cost > 0` 宣稱有可部署 selector。
- 每日 lookup 本身是 causal；但 15s anchor 與 Q2 的 winner 身分是看完 71 日 development outcome 後選出，必須留待 2026-08-14+ untouched forward 驗證。

## 3. Conditional lower：moving-anchor 結果隔離，改用 frozen-at-touch

初版 convergence 在 upper touch 之後仍以 `basis_mid_t - anchor_t` 判斷 center／lower。這會讓basis不動、只有anchor漂移的路徑也被誤判為hit，違反A2「submit後鎖定絕對target」；因此初版C0／C2／C3 reach、時間與fallback比例一律不得作lower決策，只會在v2 bundle以`dynamic_anchor_per_second_sensitivity`角色保存。

正式 v2 契約如下：

- Upper touch 前，以當秒 causal anchor 判斷是否觸及預先凍結的 entry distance。
- First legal upper touch 時保存 `touch_anchor_basis_bp`，令 `frozen_center_basis_bp = touch_anchor_basis_bp`。
- C0 target是該 frozen center；C2／C3 target是 `frozen_center_basis_bp - threshold_distance_bp`。往後只看basis是否碰到這個絕對target，未來anchor不再參與hit判斷。
- C0／C2／C3是本輪正式比較；C1保留為合法但未條件化的wide maker-exit control，不稱為不可掛、也不作隱含fallback。
- Hit／known miss／right-censored unknown分開；C2／C3 native trail20缺值才按預註冊route查trail60。`skip cell`若被選，仍保留共同mother分母並記no-trade。

這只是 S0.5 upper-touch proxy，不等同S1真實submit時間或合法tick；正式 reach／coverage／lower距離將在v2 supplement發布後填入。正常lower始終是maker＋taker exit target；taker＋taker只屬獨立風險control或13:20 hard flatten。

## 4. S1 mother：不再由舊 q95 outcome proxy 選樣

`s1_mother.parquet` 保存全部 17,006 mapping product-days 與逐層旗標；primary 是最後一層 15,638 筆。

| Funnel | Product-days | 商品 | 相對 mapping | 本層新增排除 |
|---|---:|---:|---:|---:|
| Mapping support | 17,006 | 250 | 100.00% | — |
| Selected-anchor support | 16,690 | 250 | 98.14% | 316 |
| Selected effective all-q support | 15,826 | 247 | 93.06% | 864 |
| **q-independent liquidity gate／S1 primary** | **15,638** | **244** | **91.96%** | **188** |

最後 188 筆由 109 個 long-history fail 與 79 個 hard-data fail 組成；recent-history 在這一層全部通過。Anchor 排除的 316 筆是沒有任何可用 analysis second；不是一般化的 `unsolved`。Lookup 要嘛有完整 24 cells（4 TOD × 3 q × 2 side），要嘛完全缺，沒有半套 panel。

Liquidity invariance audit 覆蓋 21,744 product-days、91 日、250 商品：每組舊 q50／q80 rows 的 source-asof、long／recent／hard gate 全部一致，21,744／21,744 通過。這證明 S1 gate 已真正移除舊 q-dependent fields，而不是把 q95 selector 換名稱。

## 5. 七組 marginal-reference geometry：q-policy lower 不是正常 exit 結論

每個 policy 共用 62,552 product-day×TOD rows（15,638×4）。Fixed policy 的上下距離確實都是 15／20／25／30 bp；q-policy 的 `lower_distance_bp` 則是同一 entry-q table 的 marginal negative side。逐列 audit 已確認 q50／q80／q95 共 187,656 rows 與 C1 control 1:1 完全相等、max absolute difference為0。因此本表改稱 marginal two-sided control，不是 conditional round-trip geometry。

| Policy | Nominal band p50 | Rounded band p50 | Nominal same-day margin p50 | Nominal positive | Rounded same-day margin p50 | +10／20／30 bp adverse positive |
|---|---:|---:|---:|---:|---:|---:|
| q50 | 16.539 | 35.391 | -6.005 | 27.75% | 12.275 | 56.69／34.53／22.82% |
| q80 | 30.843 | 43.716 | 8.655 | 85.43% | 22.047 | 74.96／53.90／35.28% |
| q95 | 53.082 | 72.901 | 30.817 | 99.92% | 49.738 | 99.00／94.76／80.28% |
| fixed15 | 30.000 | 47.225 | 8.777 | 94.62% | 25.370 | 90.78／66.26／36.38% |
| fixed20 | 40.000 | 57.721 | 18.777 | 99.25% | 35.168 | 99.19／92.98／63.72% |
| fixed25 | 50.000 | 68.181 | 28.759 | 99.95% | 45.393 | 99.72／99.56／93.34% |
| fixed30 | 60.000 | 76.046 | 38.759 | 100.00% | 53.451 | 100.00／99.87／99.26% |

同日 known reference cost 的中位數約 21.21–21.24 bp；隔夜約 36.22–36.29 bp。原 q95 marginal band 扣同日已知成本只有 12／15,638 product-days 不為正，看似幾乎全數有利；但這正是使用不可達的 C1 lower 所造成，不能當共同 mother gate、q95 subgroup或獲利證據。

向外 tick rounding 會同時增加表面 band 與掛單深度；rounded positive share 不是免費 alpha。此表尚未包含：

- maker queue／own quantity／partial fill；
- B6 `+50 ms` 至 5 秒 executable hedge 與 slippage；
- 20M／單檔 10M chronological reservation；
- exit maker fill、跨日稅差、carry／13:20 hard flatten；
- unresolved／naked risk 與 realized cashflow。

所以這一節中 fixed policies 可作合法 tick／費稅 reference；q policies 只保留為錯誤假設 control，不能證明「會賺」或「可以部署」。

## 6. Conditional lower 接回成本：等待 frozen v2 正式數字

成本幾何的正確公式是 `upper_distance + lower_distance - cost`。初版C2／C3 lower來自moving-anchor convergence，故先前算出的margin、positive share與coverage全部撤回，不作決策證據。v2將按完整causal key把frozen C0／C2／C3接回同一S1 mother；每個q×candidate的共同分母固定為`15,638 × 4 = 62,552`個product-day×entry-TOD cells，unsupported cell保留在分母中，正率只在明示supported cells內另報。

正式表會同時列：

- C0／C2／C3 的 supported cells、coverage與trail60 fallback；
- lower distance p50、同日／隔夜 nominal known-cost margin p50與positive share；
- path-level fallback share與mother-cell fallback share，兩種分母不混寫；
- entry upper source、lower source、各自as-of、combined as-of、entry TOD與fallback reason。

直到v2發布並通過independent verifier前，不能回答哪個lower有足夠完成率或成本空間，也不能先選C2／C3。即使表面margin為正，它仍只是pre-replay geometry，不含makerFill、B6 hedge、exit maker、20M reservation或realized cashflow。

## 現在已解決與尚未解決

| 問題 | S0.5 結論 | 下一個 stage |
|---|---|---|
| 盤中中價差是否因果、穩定 | 已選 `time_ewma_15s` | S1 共用 |
| D−1 q 對隔日是否有預測性 | 跨商品排序強；絕對 level／同商品 temporal 仍有限 | Q2 frozen baseline；8/14+ forward |
| 哪些商品具已知成本後幾何 | Fixed完整；q-policy C1只作marginal control；C0／C2／C3待frozen v2 | v2完成後才凍結lower與unsupported規則 |
| 舊 `unsolved` 是什麼 | 已拆成 anchor、all-q、liquidity support funnel；convergence unknown 另以 censor reason 保留 | 不再用 generic `unsolved` |
| maker entry 是否成交 | 未解 | S1 B5 approximate，S5 exact |
| hedge 是否可執行／滑多少 | 未解 | S1 B6 |
| lower maker 是否成交 | 未解；S0.5 frozen mid-touch也不是maker fill | S3 indexed FIFO |
| 20M 下日均 net／同日完成率 | 未解 | S1–S5 |
| 可部署 baseline | **尚未成立** | lower coverage、S1–S5、exact calibration＋untouched forward 後才評估 |

## Bundle、hash 與重現

Selection canonical（anchor／entry-q／mother）：[`foundation_selection_s05_rebuild_20260826_v1`](../../data/walkforward/foundation_selection_s05_rebuild_20260826_v1/)

- Source code commit：`ace2669dbc3c725a94baa035494b9bef10701304`。
- Registry SHA-256：`00b0147db5054b126046ef2d59b2ca2a43ad26e78d695a88fed13ffd8f151ec1`。
- Session-list SHA-256：`4781a479f4c04b6d53fe206035a89c1bb7a83ecc8d2796266aac6467bb7bc6cd`。
- `complete.json` SHA-256：`de7f6d1dddcdbe18875acfb4965d165cc9af6387ec8b02d1e8d55c5766531d60`。
- Marker payload SHA-256：`6bb4c2ee97e77e59912ec57c90396af3ac92bb269ddacdae394b0feccc642be4`。
- Final checkpoint fingerprint：`3425776cc651c26c8f1ec47eddfeffd9f127c6d03f96a8cab42fd14d2c35983c`。
- Final publication：51 artifacts、1,069,627,493 bytes；`complete=true`。
- Independent `verify-only`：anchor 71／71、episodes 131／131、boundary base／final／rank、convergence facts／predictions 131／131、final publication 全部通過；protected forward 起日仍為 20260814。

上列v1的anchor、entry-q與mother仍有效；其中moving-anchor convergence只准作sensitivity。Frozen-at-touch v2將以獨立atomic supplement發布，直接hash v1 marker、hidden boundary checkpoint與逐日raw inputs；正式path／geometry數字與v2 hash待完成後補入。

Build：

```bash
UV_CACHE_DIR=/tmp/codex-uv-cache \
uv run --no-project --with polars \
  python -u -m maker.src.quote_fill.foundation_selection_runner all --execute
```

不重算、只驗證既有 checkpoint 與 final publication：

```bash
UV_CACHE_DIR=/tmp/codex-uv-cache \
uv run --no-project --with polars \
  python -u -m maker.src.quote_fill.foundation_selection_runner verify-only
```

131 個 daily source markers 都是既有 `migrated_nonatomic` lineage，沒有可回溯的 atomic writer provenance。因此 final bundle 雖有完整 input／checkpoint hash chain，仍正確標示 `development_only=true`、`deployment_baseline_approved=false`；這個限制不能靠 verifier 消除。
