# Post-cross position evaluation：60 日正式結果（2026-08-21）

## 結論

> **Universe leakage caveat：**本文結果只對事後凍結的 `first_wave_45` 研究 cohort 成立。membership 使用 May–Aug 2026 已實現流動性（包含 target-day 與後續窗口）挑選；因此 45 檔名單不是 D-safe，也不是 production allowlist。本文的 D−1／source-integrity GO 僅代表固定 cohort 內的 rule lineage 與 artifact integrity。

正式 v4 bundle 已完成、原子發布，並由第二個獨立 8/12 GiB capped service 從四個 immutable formal roots 重建驗證通過。固定 cohort 內的資料完整性、D−1、entry/exit FK、兩側 50 ms hedge、完整四格 exit population與描述性 terminal report皆為 **GO**；`universe_selection_d_safe_go` 則是 **NO_GO**。

這仍不是可交易的最佳 q 策略：`best_q_selection_go`、D-safe EV、完整成本、position-limit deployment與`production_strategy_go`全部 **NO_GO**。主要原因是70,124條path尚未有point-identified terminal cashflow、尚無formal source-bound完整成本producer，而且nominal instant-cancel V0仍是模型假設。本文所有正報酬皆是completed-only描述或成本敏感度，不可當成正式EV或部署訊號。

正式輸出：[`post_cross_position_evaluation_60d`](../../data/walkforward/post_cross_position_evaluation_60d)

- `complete.json` SHA-256：`ef54971505a5fd41b276999c617c9fefdb74ffd44c65396d24fe801b04e96ea7`
- evaluator：`post_cross_position_evaluator_v4_cache_bounded_preaggregated_actions`
- evaluator source SHA-256：`a83539c6fb85e06b91ae4213bd594a720d792480c3158349c1257c70de9c4069`
- implementation inventory SHA-256：`2fb209186f4fcb1b4bbaab07ad73ac5dcf08d611a2998a10389f90acf5215aa6`
- evaluator config SHA-256：`b4b82ddd5edf882ac95e85f69d7c8baee0127a750bc51a5ece2db24c76df2bec`
- formal cross manifest：`733f70fc49f4633ce60f0574e36af521ed4e3f988fb6c5e668da3eaa98318ebd`
- same-day manifest：`b99d8b2f11354e150f02c990f178237e44078fc68dabc84f81317ce1f7aff6d8`
- prerequisite marker：`cd0c66bf24ab56eb657974adcefe28bec9ff0276fd31230a65a6f451efa661d3`

## 記憶體事故與正式修復

第一次通用loader曾materialize完全不用的176,941,848-row aliases與121,224,284-row raw facts，約119 GiB RSS後被kernel OOM kill，連帶造成VS Code當機。v3 narrow loader雖成功產生相同九表，但cgroup peak達8,596,803,584 bytes且`memory.high`被觸發4,818次，因此結果被可復原地隔離於`post_cross_position_evaluation_60d_v3_high_event_20260821`，不作正式結果。

v4在不修改frozen entry/same-day/cross實作的前提下：逐partition完成所有hash/schema/contract後呼叫`POSIX_FADV_DONTNEED`；全4,032,586 actions逐partition預彙總diagnostics，只保留108,310筆alias-local established actions；另硬鎖2,687分區、2,760,254個partition-disjoint raw IDs的inventory digest `d28fbc246482ed97628f24bd94b75d5ee3a06ced6184fdbb8eb0575b6ab1fb17`。

正式build service：

- unit：`post-cross-position-eval-v4-formal-20260821.service`
- InvocationID：`84c36b95367f4e75b516fbe8634663f7`
- `MemoryHigh=8 GiB`、`MemoryMax=12 GiB`、swap max 0、`Restart=no`
- exit 0；telemetry peak 3,778,846,720 bytes（約3.52 GiB）；swap 0；`high/max/oom/oom_kill=0`

完整source-rebuild verifier：

- unit：`post-cross-position-eval-v4-verify-20260821.service`
- InvocationID：`bc18209776694fda9c00ae3a48499fe4`
- exit 0；最後instrumented peak 2,817,064,960 bytes，systemd terminal accounting亦低於8 GiB；swap 0、無memory/OOM gate事件
- 正式九張Parquet與隔離v3逐檔SHA完全相同，因此byte、ordered schema與value全部一致；只有evaluator/loader version、cache discard、preaggregation與raw-inventory attestation metadata不同

## Universe與terminal狀態

- 60 sessions × 45 products = 2,700格；2,687 complete，13 unavailable。
- 4,032,586 submitted entry policy aliases；108,310 established aliases；81,010 unique established physical entry dependencies。
- 每個established alias恰有`Center/Lower × 兩條exit route`四格，共433,240 physical policy paths。
- path結果：363,116 completed（83.814%）、1,775 censored（0.410%）、68,349 unknown（15.776%）。後兩者合計70,124，全部保留未定價，從未補0 PnL。
- 24個q/entry/rule/exit格是互斥政策替代方案。同一entry可能跨q共享physical dependency；下列表格與TWD不可跨格相加。

路徑縮寫：Entry `F→S`=`future_ask_spot_taker`，Entry `S→F`=`spot_bid_future_taker`；Exit `F→S`=`future_bid_spot_taker`，Exit `S→F`=`spot_ask_future_taker`；`C/L`=`frozen_center/lower`。`19/34`是same-day 19 bp、overnight 34 bp的completed-only敏感度，並非完整成本。

## 24個固定政策格

`Hold mean/p90/max`為completed paths跨session boundary數；所有格的median都是0，因此省略p50。

|q|Entry|Rule|Exit|N|Comp %|Cens %|Unk %|Gross bp|Gross TWD M|19/34 bp|19/34 TWD M|Hold mean|p90|max|
|---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|50|F→S|C|F→S|25,914|90.11|0.25|9.64|12.69|10.140|-7.75|-7.720|0.109|0|8|
|50|F→S|C|S→F|25,914|88.66|0.53|10.81|11.33|8.797|-9.17|-8.993|0.128|1|16|
|50|F→S|L|F→S|25,914|84.11|0.27|15.62|19.34|15.024|-1.65|-2.254|0.153|1|9|
|50|F→S|L|S→F|25,914|82.45|0.66|16.89|18.10|13.660|-3.18|-3.596|0.201|1|16|
|50|S→F|C|F→S|34,490|90.99|0.45|8.56|15.59|20.396|-5.24|-7.325|0.134|1|11|
|50|S→F|C|S→F|34,490|91.30|0.43|8.27|14.61|19.061|-5.90|-8.505|0.124|1|16|
|50|S→F|L|F→S|34,490|83.25|0.47|16.29|22.24|27.091|0.79|0.644|0.189|1|11|
|50|S→F|L|S→F|34,490|85.31|0.57|14.12|21.79|26.284|0.29|-0.542|0.214|1|16|
|80|F→S|C|F→S|15,250|87.91|0.19|11.90|19.69|9.007|-0.66|-1.131|0.103|0|11|
|80|F→S|C|S→F|15,250|85.99|0.35|13.65|18.07|8.184|-2.42|-1.932|0.122|0|13|
|80|F→S|L|F→S|15,250|76.05|0.21|23.74|31.21|12.693|10.21|3.594|0.156|1|11|
|80|F→S|L|S→F|15,250|73.29|0.47|26.24|29.60|11.846|8.09|2.795|0.228|1|14|
|80|S→F|C|F→S|17,515|89.49|0.36|10.15|21.93|14.623|1.28|0.633|0.119|1|8|
|80|S→F|C|S→F|17,515|89.07|0.30|10.64|20.34|13.464|-0.08|-0.499|0.115|0|11|
|80|S→F|L|F→S|17,515|75.80|0.42|23.78|32.97|19.126|11.65|6.623|0.193|1|9|
|80|S→F|L|S→F|17,515|77.43|0.62|21.96|32.83|18.684|10.93|5.894|0.272|1|17|
|95|F→S|C|F→S|7,472|85.88|0.13|13.99|26.64|5.673|6.47|0.913|0.086|0|4|
|95|F→S|C|S→F|7,472|83.59|0.36|16.05|25.38|5.299|5.08|0.584|0.110|0|11|
|95|F→S|L|F→S|7,472|64.21|0.09|35.69|47.35|7.829|26.49|4.186|0.151|1|10|
|95|F→S|L|S→F|7,472|59.90|0.43|39.67|44.13|7.116|23.02|3.524|0.210|1|16|
|95|S→F|C|F→S|7,669|88.59|0.34|11.07|28.54|8.246|8.17|2.265|0.101|0|4|
|95|S→F|C|S→F|7,669|87.80|0.25|11.96|26.60|7.575|6.43|1.633|0.110|0|11|
|95|S→F|L|F→S|7,669|63.84|0.43|35.73|47.73|10.109|26.93|5.603|0.157|1|6|
|95|S→F|L|S→F|7,669|63.27|0.42|36.32|45.87|9.558|24.18|5.031|0.289|1|12|

主要trade-off很清楚：q95 Lower的completed-only 19/34 mean最高（23.02–26.93 bp），但completion只有59.90–64.21%，unknown/open達35.69–39.67%；q95 Center的19/34 mean較低（5.08–8.17 bp），completion仍有83.59–88.59%。這不是Lower或q95已勝出的證據，因未完成路徑尚未point-price，且完整成本與D-safe LCB gate未成立。

## Daily outstanding（逐固定政策、不可跨格相加）

統計區間為2026-05-20至2026-08-19，共64個觀察日。Notional是one-way spot-leg entry-price notional，不是EOD mark、two-leg gross exposure、futures margin或capital requirement。

|q|Entry|Rule|Exit|Count p50|p90|max|Notional p50 M|p90 M|max M|
|---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|
|50|F→S|C|F→S|86|141|160|25.28|40.75|57.63|
|50|F→S|C|S→F|94|164|204|28.08|43.77|59.73|
|50|F→S|L|F→S|124|203|254|42.86|64.63|81.98|
|50|F→S|L|S→F|153|243|273|46.38|73.27|88.87|
|50|S→F|C|F→S|97|205|425|36.97|69.04|103.81|
|50|S→F|C|S→F|101|194|256|34.88|56.90|88.02|
|50|S→F|L|F→S|166|319|757|65.13|100.44|186.71|
|50|S→F|L|S→F|183|305|629|63.45|103.16|165.40|
|80|F→S|C|F→S|50|101|121|13.60|29.35|38.30|
|80|F→S|C|S→F|60|117|157|16.38|28.91|49.24|
|80|F→S|L|F→S|92|153|204|30.41|44.99|70.23|
|80|F→S|L|S→F|109|179|215|32.01|54.27|74.59|
|80|S→F|C|F→S|55|112|236|17.51|35.94|52.03|
|80|S→F|C|S→F|50|116|173|18.33|32.91|68.17|
|80|S→F|L|F→S|101|188|344|36.15|64.19|111.54|
|80|S→F|L|S→F|119|223|358|44.22|72.43|106.03|
|95|F→S|C|F→S|22|54|76|6.05|14.90|24.99|
|95|F→S|C|S→F|30|68|94|6.46|15.50|24.71|
|95|F→S|L|F→S|51|107|194|15.25|35.79|43.47|
|95|F→S|L|S→F|59|117|218|19.66|35.55|44.89|
|95|S→F|C|F→S|21|50|122|6.43|15.10|33.73|
|95|S→F|C|S→F|22|53|117|7.59|20.99|26.27|
|95|S→F|L|F→S|50|114|215|19.13|39.39|79.33|
|95|S→F|L|S→F|63|124|225|23.58|43.51|99.59|

## Position-limit sweep：只作analysis

以下每格是在13個預設cap中，事後挑出`completed_same19_overnight34bp__accepted_completed_after_twd`最高者。它是方便檢視cap敏感度的diagnostic，**不是**跨cap公平比較、D-safe policy selection或可部署部位上限；各scenario互斥、沒有joint volume allocation，production-ready欄全為false。

|q|Entry|Rule|Exit|Ex-post best default limit|Accept %|Accepted completed|19/34 TWD M|
|---:|:---:|:---:|:---:|:---|---:|---:|---:|
|50|F→S|C|F→S|positions_25|0.56|119|-0.036|
|50|F→S|C|S→F|positions_25|0.49|103|-0.032|
|50|F→S|L|F→S|per_product_notional_10m|38.28|8,154|0.372|
|50|F→S|L|S→F|per_product_positions_5|3.52|699|0.069|
|50|S→F|C|F→S|positions_25|0.27|68|-0.010|
|50|S→F|C|S→F|positions_25|0.29|75|-0.031|
|50|S→F|L|F→S|per_product_notional_25m|52.59|14,924|1.014|
|50|S→F|L|S→F|per_product_notional_25m|52.34|14,988|0.982|
|80|F→S|C|F→S|notional_100m|19.46|2,531|0.370|
|80|F→S|C|S→F|notional_100m|13.06|1,598|0.199|
|80|F→S|L|F→S|per_product_notional_25m|74.94|8,512|2.223|
|80|F→S|L|S→F|per_product_notional_25m|73.13|8,108|1.918|
|80|S→F|C|F→S|per_product_notional_25m|73.90|11,461|0.555|
|80|S→F|C|S→F|per_product_notional_10m|53.03|8,069|0.613|
|80|S→F|L|F→S|per_product_notional_25m|60.49|7,911|2.943|
|80|S→F|L|S→F|per_product_notional_25m|57.08|7,531|2.492|
|95|F→S|C|F→S|per_product_notional_25m|94.83|6,041|0.838|
|95|F→S|C|S→F|per_product_notional_25m|92.81|5,740|0.827|
|95|F→S|L|F→S|per_product_notional_25m|80.66|3,793|2.357|
|95|F→S|L|S→F|per_product_notional_25m|80.26|3,534|2.189|
|95|S→F|C|F→S|per_product_notional_25m|89.72|6,040|1.755|
|95|S→F|C|S→F|per_product_notional_25m|87.08|5,769|1.770|
|95|S→F|L|F→S|per_product_notional_25m|68.26|3,215|2.667|
|95|S→F|L|S→F|per_product_notional_25m|67.95|3,183|2.262|

## Prequential challenger與readiness

60/60 as-of dates皆為`NO_GO_NO_EV_READY_ACTION`，selected q/entry/rule/exit與EV/LCB欄全部null。最新2026-08-13的completed-only diagnostic challenger為q95、Entry S→F、Lower、Exit F→S，19/34 diagnostic mean 27.0442 bp；它明示`diagnostic_challenger_is_selected_action=false`。歷史上該格出現50個as-of dates；另一個q95、Entry F→S、Lower、Exit F→S出現9次；首日沒有challenger。

|Gate|Status|
|:---|:---:|
|source coverage contract|GO|
|cross partition hashes|GO|
|cross policy lineage|GO|
|prerequisite binding|GO|
|exact D−1 lineage|GO|
|entry/exit FK|GO|
|entry/exit 50 ms hedge|GO|
|complete four-action exit grid|GO|
|no gross imputation|GO|
|nominal-only semantics|GO|
|implementation identity|GO|
|source integrity|GO|
|descriptive terminal report|GO|
|position-limit analysis|GO|
|terminal cashflow point identified|NO_GO|
|formal full-cost profile|NO_GO|
|D-safe prequential EV action|NO_GO|
|best-q selection|NO_GO|
|position-limit deployment|NO_GO|
|production strategy|NO_GO|

## Artifact inventory

|Artifact|Rows × cols|Bytes|SHA-256|
|:---|---:|---:|:---|
|`daily_outstanding.parquet`|1,536 × 17|22,763|`3883af004e55e2619dd19e303d0baed7ca92b6f8812ba8736ffa05c0ca6ca4ed`|
|`daily_terminal_cashflows.parquet`|1,536 × 14|49,996|`042ac30bde4ebcb90bc9b7b1587ebb4e3cb3f159ebef89eb7f5cf289cec15722`|
|`physical_policy_paths.parquet`|433,240 × 60|30,812,797|`a7810e919abc74789e76f069cdf9a05de35655f861847404c1c80e0182ef9975`|
|`policy_summary.parquet`|24 × 32|16,143|`b3a80d86eb95236fe18ada55d075d1d8bbf952fec20e131eca5bc0743bc63b88`|
|`position_limit_sweep.parquet`|312 × 36|35,570|`5891d4a8728e72b3355e3b9f5c8f4519855863eec9fe84c36d48dd347755c22b`|
|`prequential_decisions.parquet`|60 × 20|9,089|`269f05f11f9b4ef7a2a9ce42f9d7a31198e7ec36a3257aecc1748b2295a64f24`|
|`prequential_policy_rankings.parquet`|1,440 × 33|49,253|`b33f1d59d9045a0332ce213166468525ad1a2f93afc5bb1593ad5a3654740bd2`|
|`product_policy_summary.parquet`|1,080 × 33|114,632|`2a070a4597053c085f2566b738083f8f01ad70c2234117f7d55e89de1a36e743`|
|`readiness.parquet`|20 × 4|3,039|`f5e843e2e7cf59c0561442651eab9fb2ad855096816f365e9a495962979e8be9`|

九份artifact合計31,113,282 bytes；marker 35,039 bytes。正式public verifier已重驗exact file set、ordered schema/dtype、rows/columns/bytes/SHA、安全metadata，並從四個formal roots重建九表逐值比對。

## 測試

- Post-cross targeted：34/34 PASS。
- Post-cross＋filled-entry＋exit-report related：63/63 PASS。
- Maker full discovery：489/489 PASS。
- 獨立audit另重算2,687-partition raw inventory、cache-discard時序、碰撞反例、established-only parity與正式九表v3 SHA parity，全部PASS。
