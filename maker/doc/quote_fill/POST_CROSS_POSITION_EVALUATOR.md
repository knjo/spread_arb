# Post-cross 最佳 q／部位控制 evaluator

狀態：formal cross-session v8 已完成 2,687 個 product-days並通過terminal audit。第一個v3 bundle雖原子完成且自身完整，cgroup仍出現`memory.high`事件，因此已可復原地隔離在`post_cross_position_evaluation_60d_v3_high_event_20260821`，不能視為正式結果；canonical output維持不存在。現行candidate再加入page-cache釋放與full-action預彙總，須重新通過獨立audit及8 GiB high事件為0的正式run。這份evaluator不讀raw tick，也不修改entry、same-day、cross-session或prerequisite roots。

## 記憶體與 source validation

舊的通用 same-day loader 會把本報告完全不使用的 176,941,848-row candidate aliases 與 121,224,284-row raw candidates 一起載入；第一次 preflight 因而曾達約119 GiB RSS並被 kernel OOM kill。該路徑已禁止重跑。

v3 使用 post-cross 專用 narrow loader。它仍逐一驗證全部2,687個 same-day marker、七個 artifacts的exact inventory、SHA-256、rows、columns、ordered schema、runner/config/source lineage，以及root manifest；hash或schema竄改即使發生在不使用的candidate／alias／observation／transition檔也會在輸出建立前失敗。任何formal root、Date／Value partition、marker或artifact symlink也會被拒絕；root manifest只接受 supplied root下的exact canonical absolute path，或相對目前workspace的exact canonical path，不能用resolved symlink target自我授權。差別只有materialization：

- entry action只讀filled-entry report所需18欄；
- position policy只讀15欄；
- entry `exit_facts`逐分區只讀9個rule-lineage欄，先驗完整投影的唯一性、有限threshold與exact D−1，再只保留position FK精確需要且無缺漏／矛盾的rules；
- unused same-day replay artifacts只讀Parquet metadata並串流計數／hash，不建立DataFrame；
- 每個action partition通過完整execution contract後，先產生可加總的execution diagnostics，再只保留alias-local full-fill＋observed/executable 50 ms hedge的108,310筆established actions；全量4,032,586筆action的diagnostics、submitted分母與metadata由逐partition預彙總精確復原。下游仍對retained actions做完整frozen report驗證、position FK與四格population檢查，因此不跳過任何正式eligibility gate，但不再為no-fill aliases重跑4m-row global raw-group aggregation。
- 逐partition diagnostics只有在physical `raw_order_fact_id`不跨product-day時才可加總。正式60日root硬鎖2,687分區、2,760,254個partition-unique raw IDs的完整inventory digest；任何raw ID換分區、重複或缺漏都會在彙總前失敗。bounded/subset loader則直接維護exact ID set檢查跨分區碰撞，不能靠caller自報「可加總」。
- retained action、position與bound-rule frames以`rechunk=False`合併；policy path derivation使用columnar expressions，universe SHA為逐列增量canonical JSON。
- 每個same-day partition的七份artifact與三份entry artifacts完成frozen hash/schema驗證和必要projection後，正式Linux runner強制呼叫`POSIX_FADV_DONTNEED`釋放clean page cache；cross v8五份artifact也在完整overlay驗證後釋放。缺少該primitive或advice失敗會fail closed，不能用提高memory cap取代。

合法0-row entry partitions保留producer既有的10欄action與15欄exit typed-empty schema；loader只對這兩個exact ordered schemas補出report projection。非空action另硬鎖frozen root實際存在的五個完整96欄ordered producer schemas與全2,687分區schema inventory digest。353個沒有full fill的合法分區由Polars把全null `full_fill_recv_time_ns`寫成`Null`；只有完整schema精確命中whitelist時，這一個projection欄才會轉成typed `Int64` null。其他Null欄、額外欄、欄位重排或未列出的完整schema一律fail closed。任意錯欄、錯型，即使攻擊者同步重算artifact SHA、entry marker、same-day config與root manifest，也不會被projection掩蓋。

Source provenance另有兩層path binding：supplied entry root必須與prerequisite marker內簽署的`entry_execution_root`和`product_days.path`完全一致；而全部2,687個v8 cross marker內嵌的entry／same-day partition absolute path、marker SHA、config SHA、runner version與3＋7份artifact SHA，必須與本次實際驗過的roots逐product-day一致。把相同bytes複製到另一個root後repoint，仍不會被接受。

正式build與verify都只能在VS Code之外的獨立transient user service執行，固定`MemoryHigh=8G`、`MemoryMax=12G`、`MemorySwapMax=0`、`Restart=no`。程序會在partition 500倍數、formal sources、filled report、evaluation與publish階段寫出RSS、cgroup current/peak及memory events。驗收要求不只是無OOM，而是`memory.high=0`且peak低於8 GiB；任何high/max/OOM事件均視為blocker，不提高上限硬跑。

## 資料與政策口徑

正式 loader 固定重用 `filled_entry_report.py` 的完整驗證鏈：entry 與 same-day root manifests、逐 partition marker/hash、50 ms entry/exit hedge、D−1 rule lineage、immutable entry/exit foreign keys、完整 `Center/Lower × 兩條 exit route` population，以及 cross root manifest、runner與 prerequisite hashes。Primary 只接受 `cross_session_nominal_policy_outcomes.parquet`；strict 或混合 cancel semantics 會被拒絕。`source_integrity_go` 只有在 coverage、全部 cross hashes、D−1、foreign keys、兩側恰為50 ms、四格完整、未補 gross 0、nominal-only與預期 implementation hashes 全部成立時才會為 GO。

Coverage 契約是完整 `60 sessions × 45 products = 2,700` grid，其中恰好 2,687 個 complete product-days與13個已知 unavailable；不能把13個缺口刪掉後用2,687-row縮表冒充完整grid，也不能要求2,700格全部complete。

每個輸出格固定為：

`q50/q80/q95 × future/spot entry route × frozen Center/Lower × future/spot exit route = 24 cells`

同一 physical cell 內的 alias 先 collapse；q、entry route、exit rule、exit route 都是替代政策，不能跨列相加。這是 conditional on entry full-fill + executable 50 ms hedge 的 post-fill 研究，不能反推 submitted-entry EV。

## 輸出

- `physical_policy_paths.parquet`：去 alias 的來源綁定 path、terminal/censor狀態、label availability、持倉區間與 one-way entry notional。
- `policy_summary.parquet`、`product_policy_summary.parquet`：每個固定政策格的 completion/censor/unknown、completed-only gross及明示成本敏感度。
- `daily_terminal_cashflows.parquet`：按真正 `terminal_date` 入帳；沒有 terminal 的日流量可以是 0，但 unresolved path 從未補 0 PnL。
- `daily_outstanding.parquet`：entry EOD 至 terminal 日前一個 EOD 的 outstanding count/notional；unresolved path保留到 `last_observed_session_date`。Notional只是 spot-leg entry-price notional，不是 EOD mark、two-leg gross exposure、futures margin或capital requirement。
- `prequential_policy_rankings.parquet`：每個 D 只使用 `Date < D` origins，且 label 必須 `label_availability_date < D` 才可見。未成熟、censored、unknown全留在分母。
- `prequential_decisions.parquet`：只有 terminal mass point-priced、完整 source-bound component costs、樣本與 date-cluster LCB 全過 gate 才填 selected policy；否則 selected fields保持 null，另列不可執行的 completed-only diagnostic challenger。
- `position_limit_sweep.parquet`：每一個固定政策獨立跑 `portfolio` 或 `per_value_code` count/notional cap，同時列全 portfolio peak與單商品 peak。用 recv-time ns排序，timestamp相同時保守地先 entry 後 exit；未識別 terminal不釋放容量。它沒有 joint volume allocation，所以永遠是 analysis-only。
- `readiness.parquet`：source、descriptive report、terminal mass、full costs、prequential selection、position-control deployment與production strategy的逐項 GO/NO-GO。
- `complete.json`：固定五鍵 envelope、完整 artifact inventory、shape/schema/bytes/SHA-256、formal lineage、implementation/config identity與所有安全旗標。Verifier 會拒絕任何額外檔案／symlink／未宣告 stage，並從鎖定的四個 formal roots 重建全部九張表逐 schema、row order與值精確比對；只重算遭竄改 artifact 的自我 hash 無法通過。

## Cost profile contract

19/19 bp與19/34 bp只屬 completed-cycle sensitivity，不會開啟 EV gate。目前尚無獨立、正式、source-bound component-cost producer/root，因此 evaluator 強制：

- `formal_component_cost_source_bound=false`、`full_cost_profile_go=NO_GO`；
- 每列完整成本與 net EV 保持 null，`nominal_post_fill_ev_ready=false`；
- selected policy／best q／deployment欄保持 null或 NO-GO；
- caller 自建或即使重算 self-hash 的 `--full-cost-profile-root` 直接被拒絕，不能解鎖 EV。

未來若要開啟 EV gate，必須先另建且正式驗收 component-cost producer；它要由不可變來源生成逐 `policy_path_id` 的 fee/tax/commission/financing/overnight/cancel/emergency/other-risk components、D-safe source date、完整 lineage與 upstream binding。不能由本 evaluator 的 caller 現場組檔取代。

即使 nominal post-fill fixed-policy selection通過，`production_strategy_go`仍維持 NO-GO：nominal instant-cancel V0 是模型假設，不能冒充已識別 cancel ACK／strict execution。

## 正式命令

以下是CLI payload；正式執行時由 transient user service承載同一argv與上述memory limits，不直接掛在VS Code process tree下。啟動前必須確認output與hidden stage皆不存在，且獨立audit已GO。

```bash
env UV_CACHE_DIR=/tmp/codex_uv_cache uv run --no-project \
  python -m maker.src.quote_fill.post_cross_position_cli \
  --exit-root maker/data/walkforward/exit_maker_narrow_60d \
  --entry-root maker/data/walkforward/execution_narrow_60d \
  --cross-session-root maker/data/walkforward/exit_maker_cross_session_narrow_60d \
  --prerequisite-root maker/data/walkforward/cross_session_prerequisites_v1_20260819 \
  --output maker/data/walkforward/post_cross_position_evaluation_60d
```

可用 `--cost-sensitivity ID:SAME_DAY_BP:OVERNIGHT_BP` 與 `--position-limit ID:SCOPE:MAX_POSITIONS_OR_NONE:MAX_NOTIONAL_TWD_OR_NONE` 重複指定情境；`SCOPE` 為 `portfolio` 或 `per_value_code`。目前不要提供 `--full-cost-profile-root`：在正式 cost producer 建立前該參數必定 fail closed。沒有正式成本仍可發布 gross/cost-sensitivity/capital diagnostics，但 EV與最佳 q selection明確 NO-GO。

已發布 bundle 可用：

```python
from pathlib import Path
from maker.src.quote_fill.post_cross_position_evaluator import (
    verify_post_cross_position_evaluation,
)

verify_post_cross_position_evaluation(
    Path("maker/data/walkforward/post_cross_position_evaluation_60d")
)
```
