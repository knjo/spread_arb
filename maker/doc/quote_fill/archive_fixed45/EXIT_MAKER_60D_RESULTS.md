# 60-session Entry／Exit Maker 與查表結果

> **歷史資料警告（2026-08-21）**：本頁是舊5商品、pre-Taifex-ladder-fix研究紀錄；文內原canonical output paths目前已不存在，只保留`*_pre_taifex_ladder_fix_20260817` archive，不能視為current正式產物。45商品×60日的current-ladder same-day、cross-session與position結果請改看[POST_CROSS_POSITION_RESULTS_20260821.md](POST_CROSS_POSITION_RESULTS_20260821.md)。下列數字僅保留方法與歷史比較用途。

> **2026-08-17 策略口徑修正**：本文的 overnight 段落是「隔日第一個可成交 joint book 直接 Taker/Taker 平倉」的 forced-exit 壓力測試，**不是現在要執行的主策略**。主策略改為：進場 Maker 已完整成交且 50 ms hedge 完成後，若當日未出場，隔日重新掛同一個 frozen Center／Lower Maker，直到成交、到期或資料無法識別。下文的 `0.17 bp / submitted quote` 也降為執行吞吐敏感度；新主要獲利表只以實際建倉部位為分母，未成交／撤單只放在獨立執行診斷。

## 結論先講

這次 60-session checkpoint 已把研究鏈串到「entry maker 成交 → 50 ms taker hedge → exit maker／同日 taker-taker／隔夜 taker exit → D-safe 每日查表」。初步結果值得繼續：較極端的 q95 action 雖然 entry fill 較少，但在 **V0 立即撤單假設下**，完成的同日四腿路徑仍有正的 gross 空間；exit maker 的同日完成率也高於目前零額外延遲的 taker-taker benchmark。

但目前不能把任何一個正數稱為「每次掛單 EV」：沒有觀察到 cancel ACK／cancel race、19 bp 只是統一的非價格成本敏感度、隔夜費用與正式 fee/tax profile 尚未補齊，且 replay 是 independent alternative policies，沒有共同成交量、部位或資金配置。最新 prequential lookup 的 120 格仍全部 `ev_ready=false`、`expected_net_cashflow_bp=null`。

## 範圍與樣本單位

- 日期：2026-05-20～2026-08-13 的 60 個 session。
- 商品：`2303, 2317, 2603, 2881, 6005`。
- 完整 product-days：290／300；`2303` 有 10 日缺 point-in-time `daily_mapping`，沒有補成零成交。
- Entry：216,825 個 q-policy aliases、171,678 個跨 q 去重的 physical raw orders、3,698 個 physical full fills；其中 3,696 個 50 ms hedge 可定價。
- q50／q80／q95 是同一套 D−1 rolling boundary 的 alternative policy views。不同 q 可能 round 到同一實體價格，所以跨 q count 不可相加。
- Exit：4,501 個已 entry full-fill 且 hedge 可定價的 q-policy positions，各自展開 `center/lower × 兩條 exit maker route`，形成 18,004 個 policy trials。這四個 trial 是替代方案，不是同時下四組單。
- 所有 fill replay 都是 independent-candidate research replay；沒有讓重疊假想訂單共同消耗成交量。

## Entry maker：自然成交、撤單需求與 50 ms hedge

`fills/pd` 的分母是 290 個實際可用 product-days。`cancel %` 是策略 target retreat／session cutoff 所產生的 **cancel request**，不是交易所已確認撤單。Slippage 正值代表不利；p50 在六格都是 0 bp。

| Entry route | q | Full fills | fills/pd | Cancel request | Hedge priced | Book ≤100 ms | Book ≤1 s | 50 ms slip p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Future ask maker → Spot taker | 50 | 1,203 | 4.15 | 96.86% | 1,202/1,203 | 69.33% | 82.46% | 38.61 bp |
| Future ask maker → Spot taker | 80 | 804 | 2.77 | 97.92% | 803/804 | 64.93% | 80.60% | 36.10 bp |
| Future ask maker → Spot taker | 95 | 375 | 1.29 | 99.03% | 375/375 | 71.20% | 82.67% | 35.71 bp |
| Spot bid maker → Future taker | 50 | 1,327 | 4.58 | 95.99% | 1,327/1,327 | 58.10% | 76.11% | 27.86 bp |
| Spot bid maker → Future taker | 80 | 599 | 2.07 | 98.23% | 598/599 | 55.09% | 74.96% | 28.82 bp |
| Spot bid maker → Future taker | 95 | 196 | 0.68 | 99.43% | 196/196 | 56.63% | 79.08% | 35.34 bp |

Spot-maker 的 partial fills 依 q50／q80／q95 為 61／28／10；Future-maker 沒有 partial。q95 降低的是完成成交，不是掛撤 churn：Future-maker／Spot-maker q95 每個 product-day 仍分別有 132.62／116.98 次 cancel request，約每個 full fill 對應 102.56／173.08 次 cancel request。它也不是「每天固定交易幾次」的規則；後續 admission 應由 D-safe lookup、合法 tick、cooldown／hysteresis 與 `cancel per useful fill` 共同決定。

Entry 產物：[product_action_entry.csv](../../data/walkforward/execution_narrow_60d/report_60_sessions/product_action_entry.csv)、[report_complete.json](../../data/walkforward/execution_narrow_60d/report_60_sessions/report_complete.json)。

## Exit maker raw replay 與撤單語意

Exit replay 產生 3,033,129 個 policy-candidate aliases，折成 2,414,297 個 canonical raw identities。Canonical ID 只負責證明相同實體候選沒有被 q／rule 名稱重複建單；同一 candidate 在 center／lower policy 下可能有不同 stop time，因此不能拿單一 raw representative outcome 當全域 fill／cancel 分母。

以 policy-candidate outcome 為分母：

- cancel request：2,811,715／3,033,129 = 92.700%（不是 ACK）；
- fill-known／fill-unknown：2,020,946／1,012,183，unknown = 33.371%；
- any fill：224,603，all-candidate lower-bound 7.405%，在 fill-known 內為 11.114%；
- full／partial：221,414／3,189，all-candidate lower-bound 7.300%／0.105%；
- 221,370 個 full-fill candidate 的 50 ms exit hedge 可定價，為 full fills 的 99.980%。

在真正成為 V0 同日 winner 的 8,902 個 alternative policy paths 中，exit-hedge total slippage p50／p95 = 0／34.72 bp；這是成功 policy-path 加權，仍不是 submitted-quote 成本分布。

高 cancel-request rate 本身不等於策略必然不可行，卻使 cancel ACK 成為目前最大的識別缺口。兩種 branch 口徑的差異很直接：

| 18,004 alternative policy trials | Same-day complete | Carry at EOD | Unknown |
|---|---:|---:|---:|
| Nominal V0：cancel request 當下視為立即成功 | 8,902（49.44%） | 3,683（20.46%） | 5,419（30.10%） |
| Strict：沒有 ACK 就保留 cancel race | 159（0.88%） | 3,683（20.46%） | 14,162（78.66%） |

V0 是用來看「如果撤單真的能即時生效，execution 上限大約在哪裡」的模型敏感度；strict 才忠實反映現有資料可證明的範圍。兩列都不能當 production outcome probability。

產物：[product policy](../../data/walkforward/exit_maker_narrow_60d/report_60_sessions/product_exit_maker_policy.csv)、[strict branches](../../data/walkforward/exit_maker_narrow_60d/report_60_sessions/product_exit_maker_strict_branch.csv)、[report marker](../../data/walkforward/exit_maker_narrow_60d/report_60_sessions/report_complete.json)。

## q95：同日完成路徑與 matched Taker/Taker

以下固定 entry q95。每條 maker exit route 各有 571 個 policy trials；Taker/Taker 也只有 571 個 unique baselines，雖然配對檔會把同一 baseline 對到兩條 maker exit route，不能把它算兩次。`gross p50` 只在該 exit style **同日完成**時取中位數；四腿實際成交價已含 entry／exit 的 price move、depth 與 50 ms hedge slippage。

| Exit rule | Exit style | Trials | V0 same-day | Strict same-day | Successful gross p50 | Successful gross − 19 bp p50 |
|---|---|---:|---:|---:|---:|---:|
| Frozen center | Future bid maker → Spot taker | 571 | 275（48.16%） | 4（0.70%） | 38.61 bp | 19.61 bp |
| Frozen center | Spot ask maker → Future taker | 571 | 264（46.23%） | 3（0.53%） | 40.08 bp | 21.08 bp |
| Frozen center | Taker/Taker benchmark | 571 | 231（40.46%） | — | 41.32 bp | 22.32 bp |
| Frozen D−1 lower | Future bid maker → Spot taker | 571 | 177（31.00%） | 1（0.18%） | 67.11 bp | 48.11 bp |
| Frozen D−1 lower | Spot ask maker → Future taker | 571 | 122（21.37%） | 1（0.18%） | 54.66 bp | 35.66 bp |
| Frozen D−1 lower | Taker/Taker benchmark | 571 | 84（14.71%） | — | 62.17 bp | 43.17 bp |

把兩條 maker routes 視為替代 policy 做描述時，center 成功路徑有 436／539 = 80.89%、lower 有 285／299 = 95.32% 在扣 19 bp 後仍為正。這仍然是 `P&L | V0 same-day complete`，不是 `P&L | quote submitted`。

Matched 比較只取兩種 style 都同日完成的同一 policy：center 的 paired N 為 217／226、lower 為 79／81（分別對應兩條 maker route），四格的 maker-minus-taker gross p50 都是 0 bp。Taker/Taker 從 1 秒 grid 以零額外 latency、1 秒 book-age gate 取價，maker exit 則在 fill 後加 50 ms hedge，因此這不是 latency-matched 優劣判決。

若把 q50／q80／q95 與 center／lower 都放回先前的 Taker/Taker successful-path audit，gross p50 是 alias-weighted 32.41 bp、按 physical dependency 去重後 28.99 bp。這正是先前「32.4 bp」的語意：已 entry fill、50 ms hedge 且找到同日 Taker/Taker exit 的條件毛利中位數，不是每次掛單 32.4 bp，也不是每次掛單 11 bp EV。

配對產物：[matched_exit_style_pairs.csv](../../data/walkforward/exit_maker_narrow_60d/report_60_sessions/matched_exit_style_pairs.csv)、[matched_exit_style_summary.csv](../../data/walkforward/exit_maker_narrow_60d/report_60_sessions/matched_exit_style_summary.csv)。

### 把未完成路徑補零，只能當敏感度

若刻意把同日未完成的 carry／unknown cashflow 都補成 0，並只對完成路徑扣 19 bp，q95 每個 alternative policy trial 的平均值如下：

| Exit rule | Exit style | Gross、未完成補 0 | Gross − 19、未完成補 0 |
|---|---|---:|---:|
| Frozen center | Future bid maker → Spot taker | 24.73 bp | 15.58 bp |
| Frozen center | Spot ask maker → Future taker | 19.63 bp | 10.85 bp |
| Frozen center | Taker/Taker | 17.93 bp | 10.25 bp |
| Frozen D−1 lower | Future bid maker → Spot taker | 26.20 bp | 20.31 bp |
| Frozen D−1 lower | Spot ask maker → Future taker | 11.50 bp | 7.44 bp |
| Frozen D−1 lower | Taker/Taker | 8.18 bp | 5.38 bp |

這不是保守 EV：carry 的實測分布中位數為負，而 unknown 也未必是零；它只用來確認「漂亮的成功路徑中位數」在把完成率放回分母後，數量級是否立刻消失。

## Overnight carry：補上了什麼

3,683 個 carry policy labels 依實體 entry dependency 去重後是 1,047 個 physical positions。使用下一 session、相同 `QuoteCode`、第一個 joint executable taker book 的 gross benchmark：

- 995／1,047 = 95.03% 可定價；52 個 censored 分成 expiry settlement 未提供 27、沒有 joint executable book 16、沒有 next session 9。
- 可定價 995 筆的 gross p05／p50／mean／p95 = **−201.15／−22.42／−34.41／104.17 bp**。
- 只有 217／995 = 21.81% 的 gross 大於統一 19 bp 敏感度。
- fee、tax、financing、overnight、cancel、emergency cost 欄仍是 null；上述是 gross，不是 overnight net EV。

Ungated 版本會記錄 book age，但不拒絕 stale book。`max_book_age=1s` sensitivity 改成等待第一個兩腿都通過 freshness 的 joint book，coverage 仍是 995／1,047：131 個 physical decision cursors 改變，延後範圍 29.955 ms～47.067 s、p50 5.620 s；其中 65 個 gross 改變，而且全部來自 `6005`，delta 為 +12.06～+12.18 bp、平均 +12.13 bp。全體分布變為 p05／p50／mean／p95 = **−201.15／−22.42／−33.61／109.36 bp**，`gross > 19` 仍是 217／995，沒有改變結論。

產物：[physical facts](../../data/walkforward/overnight_carry_narrow_60d/report_60_sessions/overnight_carry_physical_facts.parquet)、[status summary](../../data/walkforward/overnight_carry_narrow_60d/report_60_sessions/overnight_carry_status_summary.parquet)、[fresh-1s physical facts](../../data/walkforward/overnight_carry_narrow_60d_fresh1s/report_60_sessions/overnight_carry_physical_facts.parquet)、[fresh-1s report marker](../../data/walkforward/overnight_carry_narrow_60d_fresh1s/report_60_sessions/report_complete.json)。

## 已建立部位的每日 prequential lookup

Primary lookup key 是 `ValueCode × entry route × q × exit rule × exit route`，完整 grid 每日 5 × 2 × 3 × 2 × 2 = 120 格。每天 D 只讀 `label_end_date < D` 的成熟路徑；表本身是 **conditional on established entry**，尚未乘回 entry fill probability。

最新 `asof_date=20260813` 的 D-safe prior-path counts：

| Scenario | Prior policy paths | Known | Unknown | Pending | Censored | EV-ready cells |
|---|---:|---:|---:|---:|---:|---:|
| Nominal instant-cancel V0 | 17,948 | 12,335 | 5,417 | 196 | 0 | 0／120 |
| Strict cancel semantics | 17,948 | 3,610 | 14,142 | 196 | 0 | 0／120 |

V0 的 120 格分成 40 格 history/group sessions 不足、38 格 known paths 不足、42 格仍有未定價 unknown／pending；strict 則為 40／76／4 格。所有 `expected_net_cashflow_bp` 都維持 null。

只看 V0 的 12,335 條 known terminal paths，跨 alternative policy rows 的 gross mean／p50 = 14.46／24.94 bp；統一減 19 bp 後 mean／p50 = −4.54／5.94 bp。再把 5,417 unknown 與 196 pending 補零，平均變成 −3.12 bp。這三組都是診斷：前者有 known-path selection，後者把未定價風險當零，且兩者都沒有真實 cost profile，因此都不是每張 admitted quote 的 EV。

Fresh-1s overnight sensitivity 不改 coverage 或任何 cell status：known gross mean 從 14.4637 變 14.5915 bp、p50 同為 24.9377；減 19 後 mean 從 −4.5363 變 −4.4085、p50 同為 5.9377；unknown／pending 補零值從 −3.1177 變 −3.0298 bp。差異很小，沒有改變 `0 EV-ready` 的結論。

產物：[latest lookup status](../../data/walkforward/exit_policy_lookup_checkpoint_60d_with_overnight/lookup_status_summary.csv)、[nominal V0 lookup](../../data/walkforward/exit_policy_lookup_checkpoint_60d_with_overnight/daily_prequential_nominal_v0_lookup.parquet)、[strict lookup](../../data/walkforward/exit_policy_lookup_checkpoint_60d_with_overnight/daily_prequential_strict_lookup.parquet)、[fresh-1s checkpoint](../../data/walkforward/exit_policy_lookup_checkpoint_60d_with_overnight_fresh1s/checkpoint_complete.json)。

## 每次送出 entry quote 的自然母體 checkpoint

上節只以已完成 entry maker fill 與 50 ms hedge 的部位為分母。新的 submitted-entry checkpoint 再向前接回每一次 entry-policy quote，因此 entry fill probability 已經包含在 lookup 中。

全期共有 216,825 個 entry-policy aliases／171,678 個跨 q 去重 raw orders。201,045 個有 frozen exit rule 的 aliases 各自展開 `center/lower × 兩條 exit route`，形成 804,180 個替代 policy paths；15,780 個缺 exit rule 的 aliases 各保留一條 unassigned unknown，沒有虛構成四條政策。合計 819,960 paths：

- nominal no-fill：619,836；gross 暫設 0，但 cancel ACK／cancel cost 仍缺；
- entry-fill outcome unknown：181,715；
- full fill + 50 ms hedge：18,004；
- partial／full-fill hedge unknown：393／12。

最新 `asof_date=20260813` 嚴格只讀先前已成熟 labels，共 215,035 個 aliases／812,800 個 alternative policy paths。Known 625,531（76.96%）、unknown 187,073（23.02%）、pending 196（0.024%），沒有 matured censor。Known 中有 613,196 條 nominal no-fill、8,884 條 same-day complete、3,451 條 overnight complete。

| Equal-weight policy-path diagnostic | Conditional on known | Unknown／pending 補 0 |
|---|---:|---:|
| Gross | +0.2877 bp | +0.2214 bp |
| 每個 completed cycle 統一扣 19 bp | -0.0869 bp | -0.0669 bp |
| Same-day 扣 19／overnight 扣 34 bp | -0.1697 bp | -0.1306 bp |

最後一列的 34 bp 只是「隔夜在 19 bp 上再加 15 bp」的研究敏感度，不是正式費率。三列也都不是 EV 或上下界：23.02% unknown 未定價、no-fill cancel cost 未納入，而且 center/lower、兩條 route 是替代政策，不能把它們當可同時執行的交易量相加。所有 cell 仍維持 `ev_ready=false`、`expected_net_cashflow_bp=null`。

不過，這張表已足以做 challenger 排名。下表用最嚴格的 `same-day 19／overnight 34、unknown 補 0` 診斷，在每個商品的 24 個 assigned cells 中各取最好與最差一格；這是同一 60-session window 內選出的結果，必須再用後續未看資料驗證。

| Product | 最佳 cell | 診斷 bp／quote | Outcome coverage | 最差 cell | 診斷 bp／quote |
|---|---|---:|---:|---|---:|
| 2303 | Future-maker entry / q95 / lower / Spot-maker exit | +0.1498 | 99.77% | Spot-maker entry / q50 / center / Future-maker exit | -0.7000 |
| 2317 | Spot-maker entry / q95 / center / Spot-maker exit | +0.0360 | 99.95% | Spot-maker entry / q50 / center / Future-maker exit | -0.5630 |
| 2603 | Spot-maker entry / q95 / center / Future-maker exit | +0.1724 | 99.76% | Future-maker entry / q50 / lower / Future-maker exit | -0.7739 |
| 2881 | Future-maker entry / q95 / center / Spot-maker exit | -0.0706 | 99.09% | Spot-maker entry / q50 / lower / Spot-maker exit | -2.0056 |
| 6005 | Spot-maker entry / q50 / lower / Future-maker exit | +0.4343 | 32.57% | Future-maker entry / q50 / center / Future-maker exit | -0.7914 |

`6005` 的表面最佳值有 67.43% unknown，不能拿來選 policy；若要求 coverage ≥80%，目前最佳是 `2603 q95` 的 +0.1724 bp。`2303 q80 lower` 在統一 19 bp 下四條 route 組合約 +0.156～+0.233 bp，但加入 overnight 額外 15 bp 後只剩 −0.054～+0.044 bp；`2303 q95 future-maker entry/lower` 則約 +0.145～+0.150 bp，較值得列為下一輪 challenger。

這也對上「每商品每天約 1～2 次完成交易」的直覺：`2303 q95 future-maker entry` 在 50 個可用 product-days 有 76 個 full fills，約 1.52 次／日；代價是 5,662 個 cancel requests，約 113.2 次／日。也就是低成交週轉不等於低 quote churn，後續 lookup 必須把 cooldown、hysteresis、cancel-per-useful-fill 與 inventory constraint 一起最佳化。

產物：[submitted-entry checkpoint](../../data/walkforward/submitted_entry_quote_lookup_checkpoint_60d_fresh1s/checkpoint_complete.json)、[daily lookup](../../data/walkforward/submitted_entry_quote_lookup_checkpoint_60d_fresh1s/daily_prequential_submitted_entry_lookup.parquet)、[label categories](../../data/walkforward/submitted_entry_quote_lookup_checkpoint_60d_fresh1s/submitted_label_category_counts.csv)。

## 三種數字不要混在一起

1. **成功路徑條件值**：例如 q95 center maker gross p50 38.61／40.08 bp，只回答「V0 同日真的完成時拿到多少」。
2. **未定價補零敏感度**：把 carry／unknown／pending 當 0 後平均；可檢查數量級，但沒有經濟上的零值保證。
3. **真正 EV**：submitted-entry lookup 已納入 entry fill probability；但仍需讓每張 quote 的互斥 terminal path 全部成熟、cancel race 可識別，並補齊四腿現金流與 versioned fee/tax/financing/overnight/emergency/cancel cost。現況尚未達到第 3 項。

另外，19 bp 只是假設的非價格 cycle cost。四腿 actual prices 已經反映 entry／exit 的 spread、depth 與 latency slippage，不能再把表中的 slippage diagnostic 重複扣一次。

## 簡潔重跑命令

下列命令重建 exit report 與兩版 overnight labels；完整 raw replay 較耗時，既有 completion marker 支援 resume。

```bash
uv run python -m maker.src.quote_fill.exit_maker_report \
  --exit-root maker/data/walkforward/exit_maker_narrow_60d \
  --entry-root maker/data/walkforward/execution_narrow_60d \
  --output-dir maker/data/walkforward/exit_maker_narrow_60d/report_60_sessions \
  --sessions 60 --allow-fewer-sessions \
  --value-code 2303 --value-code 2317 --value-code 2603 \
  --value-code 2881 --value-code 6005 \
  --assumed-non-price-cost-bp 19

uv run python -m maker.src.quote_fill.overnight_carry_cli \
  --sessions maker/data/walkforward/sessions.txt --last-entry-sessions 60 \
  --symbols 2303,2317,2603,2881,6005 \
  --contract-calendar maker/data/walkforward/exact_contract_calendar_v1.parquet \
  --entry-root maker/data/walkforward/execution_narrow_60d \
  --exit-root maker/data/walkforward/exit_maker_narrow_60d \
  --output maker/data/walkforward/overnight_carry_narrow_60d

uv run python -m maker.src.quote_fill.overnight_carry_cli \
  --sessions maker/data/walkforward/sessions.txt --last-entry-sessions 60 \
  --symbols 2303,2317,2603,2881,6005 \
  --contract-calendar maker/data/walkforward/exact_contract_calendar_v1.parquet \
  --entry-root maker/data/walkforward/execution_narrow_60d \
  --exit-root maker/data/walkforward/exit_maker_narrow_60d \
  --max-book-age-ms 1000 \
  --output maker/data/walkforward/overnight_carry_narrow_60d_fresh1s

uv run python -m maker.src.quote_fill.submitted_entry_lookup_cli \
  --entry-root maker/data/walkforward/execution_narrow_60d \
  --exit-root maker/data/walkforward/exit_maker_narrow_60d \
  --conditional-checkpoint-root \
    maker/data/walkforward/exit_policy_lookup_checkpoint_60d_with_overnight_fresh1s \
  --sessions maker/data/walkforward/sessions.txt \
  --value-code 2303 --value-code 2317 --value-code 2603 \
  --value-code 2881 --value-code 6005 \
  --allow-fewer-sessions \
  --flat-completed-cycle-cost-bp 19 \
  --same-day-completed-cycle-cost-bp 19 \
  --overnight-completed-cycle-cost-bp 34 \
  --output \
    maker/data/walkforward/submitted_entry_quote_lookup_checkpoint_60d_fresh1s
```
