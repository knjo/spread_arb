# Stage 2：S1／S2 進場點位與 E1／E2 出場點位

目標：對每個 (日, 商品) 的合併因果事件流，各自獨立重放 S1 與 S2（以及 E1／E2），
產出「掛單 → 存活／撤單 → 成交 → hedge」的**點位事實表**。它同時是：
(a) Q 表的訓練來源（取代 maker 的 shadow actor），(b) Stage 3 回測的輸入，(c) `P_fill(U)` 的來源。
點位表是 **independent-event** 口徑：不套容量、不共用成交量、每張掛單各自依起始隊列估結果；容量與共用資源留給 Stage 3。

## 1. 事件流

每個商品建一條合併流，元素為 `(ns, market, seq, kind)`，`kind ∈ {book, print, trial}`，
先 `ns`、再同市場 `seq`；兩市場同 `ns` 時，先處理 print 再處理 book（print 是既成事實，book 是後續狀態）。
每個 stream 的狀態機只在它**關心的**事件上喚醒：

| Stream | 喚醒事件 | 用它做什麼 |
|---|---|---|
| S1 | 期貨 book（FutExecBid 變）、現貨 book（合法性、隊列前方量）、現貨 print（成交）、整秒（anchor） | 掛價／撤單看期貨；成交看現貨 |
| S2 | 現貨 book（SpotA1 價／量變）、期貨 book（A1 變 → 是否仍 inside、合法性）、期貨 print（成交）、整秒 | 掛價看期貨 A1；撤單看現貨 A1；成交看期貨 |
| E1 | 期貨 book（FutA1 變 → 觸發／guard）、現貨 print（成交） | 同 S2 鏡像 |
| E2 | 現貨 book（SpotB1 變）、期貨 print | 同 S1 鏡像 |

「誰先」的判定完全由這條流的順序決定：撤單意圖在事件 `t_c` 產生、`t_c + 50 ms` 生效；
`t_c < t_print ≤ t_c + 50 ms` 的成交是**撤單競賽成交**（保留、必 hedge、標記 `cancel_race = true`）；
`t_print > t_c + 50 ms` 則撤掉。這是 maker `s2_first_invalid / event_cancel / s2_invalid_fill` 三個事件的一般化，
S1 也要有相同三個事件（maker 的 S1 沒有，因為它讀外生指令）。

## 2. 掛單狀態機（S1／S2 共用）

```text
IDLE ──(admit: EV/門檻/深度/合法)──▶ WORKING ──(print 消耗隊列)──▶ FILLED ──(+50 ms)──▶ HEDGING ──▶ PAIRED
                                       │                                                     │
                                       ├──(門檻失效／drift／gate)──▶ CANCEL_PENDING ──(50 ms)──▶ CANCELLED
                                       │                                   │
                                       │                                   └──(print)──▶ FILLED(cancel_race)
                                       └──(13:18)──▶ CANCELLED
```

每張單記：`quote_ns, quote_second, price, ab, effU, anchor, spread_bp, depth_ahead, opposite_depth,
EV 各分量（來自 Stage 1）, first_invalid_ns, invalid_reason, cancel_ns, fill_ns, fill_print_seq,
cancel_race, hedge_ns, hedge_vwap, hedge_levels_swept, actual_ab, entry_decay_bp, hedge_wait_ms, hedge_timeout`。

### S1 特有

- 掛價：對 `FutExecBid` 反推的合法 U 格點 `{U_k}`，每個 U 各建一張獨立研究單（同 maker WP02「多層掛單」：
  同 epoch、同絕對價只建一次；target 往前新增一層、退後撤掉更積極層）。Stage 2 不決定 U*，只產每個 U 的事實。
- 撤單觸發：`FutExecBid` 下移使該單鎖定的 `ab − anchor < floor`（floor 參數，先用 S2 的 20 bp 對稱）；現貨 gate 失效；13:18。
- 部分成交：現貨成交量 < 一口股數且到撤單／13:18 仍不足 → `rollback`（記 taker 賣回 VWAP）。
- 取樣時鐘：maker `REPLAY_SAMPLING.md` 的 `spread_pair_epoch`（`SpreadPairTotalCount`）作 base sample，
  避免每個 tick 都算新樣本；缺欄位時以 `SpreadPairID` causal transition 重建。

### S2 特有

- 掛價固定 `previous_tick(FutA1)`，需 `FutB1 < price`；一 tick 寬盤口不掛。
- 新掛門檻 `effU ≥ 25`、持單地板 `effU ≥ 20`、`ab > 0`；`SpotA1 ≥ 5 × 股數`；drift 10 bp（皆參數）。
- 外部單加入同價：FIFO 排在我方後面；外部 Ask 更低（被 undercut）不撤（不改變我成交時損益），只有 Bid 穿價、門檻／深度失效才撤。
- 前一口現貨 hedge 未完成不得新掛；hedge 完成當下立即重評（不設 CD）。

## 3. 出場點位

對每個 `PAIRED` position（點位表中的 entry row）獨立產生出場事實，不套容量：

| Route | 掛單 | 觸發 | Guard（持單期間） | Hedge |
|---|---|---|---|---|
| E1 | 現貨 Ask maker @ SpotA1 | `B(FutA1, SpotB1) ≤ anchor + L` | `B(FutA1, price) > anchor + L` 或期貨 ask 無量 → 撤 | +50 ms 買 1 口期貨（L1–L5） |
| E2 | 期貨 Bid maker @ `next_tick(FutB1)`（inside） | 同上 | `B(price, SpotB1) > anchor + L` 或現貨 bid 深度不足 → 撤 | +50 ms 賣現貨（L1–L5） |

- L 用格點 `{−15, −10, −5, 0}` 各建獨立研究單（Stage 1 `choose_L` 的 `P_sd(L)` 由此而來）。
- 出場掛單 09:05 起、13:18 撤；未出場者記 `survive`，次日以同 anchor、同 L 重試（Stage 2 只做到「當日是否出場」＋逐日存續觀測，跨日鏈接在 Stage 3）。
- 出場 shortfall `d_sd / d_on` 的定義見 `00 §6`；E2 的分母同樣用進場現貨本金。
- Taker cross（`should_cross`）與 C8 到期不在點位表；它們是 Stage 3 的 portfolio 決策。

## 4. 點位表是政策無關的（2026-09-17 定案）

點位表對每個候選掛單只算一次三件事：**多久成交**、**多久劣化**、**成交後 hedge 打到哪**；對每個有成交的進場再算每個出場水位 x 的**多久出場**。
所有政策（EV 公式、floor、hurdle、滑價旋鈕）都是對這張表的 filter，不重跑 tick：

```text
留下的進場 ⟺ score(x*) ≥ hurdle  且  t_fill < t_below[floor] + 50 ms（撤單延遲內的成交照留、標 cancel_race）
留下的出場 = 該進場在 x* 的出場列
```

劣化時間存成**格點**而不是布林：`t_below[f]`，f ∈ {0, 5, 10, 15, 20, 25, 30} bp（殘差）。floor 因此是 filter 參數，換 floor 不重算。
Q 表（Stage 1）是市場價差移動的機率，不吃這張表；點位表只在 Stage 3 用來校準 Q 的預測與設定值。

### 4.1 進場列 schema（`s1_entries` / `s2_entries`，每個 (stream, 商品, 事件時刻, 候選價) 一列）

| 欄位群 | 欄位 |
|---|---|
| 識別 | `stream, date, vc, qc, expiry, quote_ns, quote_second, price, event_kind` |
| 掛單當下 | `anchor, scale, quote_ab, eff_u, e_norm, fut_spread_bp, tick_bp_hedge, depth_ahead, opp_depth_shares, notional_twd` |
| 成交 | `t_fill_ns`（13:18 前未成交為 null）, `fill_print_seq, fill_kind ∈ {queue_depletion, trade_through}` |
| 劣化 | `t_below_0, t_below_5, …, t_below_30`（首次鎖定 basis < anchor + f）, `t_gate_ns`（合法性 gate 關閉）, `t_drift_10`（較 quote_ab 掉 10 bp） |
| hedge | `hedge_ns, hedge_vwap, hedge_levels_swept, hedge_wait_ms, hedge_timeout, actual_ab, d_in_realized = quote_ab − actual_ab` |
| 診斷 | `cancel_race_20`（`t_below_20 < t_fill ≤ t_below_20 + 50 ms`）, `negative_basis` |

S1 每個合法 U 各一列（同 epoch 同絕對價只一列）；S2 價位固定，每個可掛事件一列。
未成交列也保留：它們是 `P_fill(U)` 的分母。

### 4.2 出場列 schema（`e1_exits` / `e2_exits`，每個 (有成交且 hedge 完成的進場, x) 一列，跨日算到到期）

| 欄位群 | 欄位 |
|---|---|
| 識別 | `entry_id, route ∈ {E1, E2}, x, B_x` |
| 觸發與成交 | `t_trigger_ns`（hedge 後、≥ 09:05、`B(FutA1, SpotB1) ≤ B_x` 首次）, `t_exit_fill_ns`（到期前未成交為 null）, `exit_day_offset`（0 今天、1 明天 …）, `t_guard_ns` |
| hedge | `exit_hedge_ns, exit_hedge_vwap, exit_hedge_wait_ms` |
| 結果 | `realized_exit_basis, d_out_realized = realized_exit_basis − B_x`（正 = 差）, `settled ∈ {exit_fill, expiry, corporate}` |

Stage 3 的校準診斷用這兩張表：每格「Q 預測 `C0`／`C1` vs 實際 `exit_day_offset` 分布」、「config 的 `d_in`／`d_out` vs `d_in_realized`／`d_out_realized`」。

### 4.2b 實作備註（S2，2026-09-17）

- 候選事件 = 期貨 book 變動 ∪ 現貨 top-ask 變動 ∪ 期貨 print ∪ 每秒邊界；准入下限 `effU ≥ 10 bp`（比任何政策鬆）。
- 一列是一個**狀態段**，對下一列之前的任何掛單時刻有效：狀態 (P, SpotA1, 量, FutA1, FutB1) 變、≥ P 的 print、anchor 把 eff_u 推過任一地板時出新列。沒有固定間隔的再出列。
- S2 `depth_ahead = 0`（隊列第一）；`t_below[f]` 以**當時** anchor（逐秒）判定；另存 `t_ab0`、`t_up_{5,10,20}`／`t_down_{5,10,20}`（可掛價相對我的 P 上／下移 ≥ d bp）；欄位多了 `fut_a1, fut_b1, spot_a1, fill_price, resid_mid_bp`。
- 時序規則（walker 與 Stage 3 共用）：同商品一張活單；下一張單掛在撤單生效或 hedge 完成那一刻，用當時所在的段；撤單觸發 = 地板／gate／ab0，另可加「等 T 秒後有更好的 pair 就撤掉往前掛」。沒有冷卻時間。
- 參考 walker `sequential_fills` 只用來對帳 maker 供給數，不是 Stage 3 的回測。

### 4.2c 實作備註（S1，2026-09-18）

- 階梯：`level −1` = 現貨買賣價差內一 tick（B1 + tick < A1 時才有）、`0` = B1、`1` = B1 − 1 tick；更深的層 maker 量過成交 ~1%，先不出列。每層一列。
- 掛價鎖定的 basis 用期貨**可執行 Bid**：`quote_ab = FutBid / P − 1`；`depth_ahead` = 該價位顯示的現貨股數（inside 為 0）。
- 成交：t0 之後價 ≤ P 的現貨 print 累積量 ≥ `depth_ahead + 1,000` → `t_partial_ns`（第一張）、≥ `+ 2,000` → `t_fill_ns`；只到第一張就撤是 rollback（Stage 3 依撤單時刻的現貨 Bid 估回補成本）。
- 劣化／移動看期貨 Bid：`t_below[f]`（`FutBid < P·(1+(anchor+f)/1e4)`）、`t_ab0`（`FutBid ≤ P`）、`t_up_d`／`t_down_d`（期貨 Bid 相對 t0 上／下移 ≥ d bp）；劣化時間軸只用期貨事件＋每秒，gate 用全聯集。
- hedge：`t_fill + 50 ms` 起在期貨 Bid L1–L5 賣 1 口（±10% 參考價內），不足等下一個 book；`actual_ab = hedge_px / P − 1`。
- 狀態段的鍵是 (P, 排隊量桶)：排隊量以 **相對 10% 一格**（不細於一張）量化——B1 的掛單量每次 book 更新都在變，逐張出列會讓一天上千萬列；期貨 Bid 只影響 `eff_u` 特徵，不進鍵，靠每秒的地板跨越再出列。
- 現貨隊列模型與 maker 引擎對帳：同一批 7/6 q95 指令（4,224 張）餵兩邊，成交 80 vs 72、共同 69，成交時間差 p50 0.000 s、p90 37 ms、97% 在 1 秒內。

### 4.2d 雙向表（2026-09-18 定案，取代「出場列另做」）

使用者定案：兩張表都是**雙向**，出場側就是同一張表上的另一個 side，不另做 per-position 出場模擬。

| 表 | side | 掛在 | 對向可執行價（鎖定價差） | 成交看 | hedge | 劣化欄 |
|---|---|---|---|---|---|---|
| S1（現貨 maker） | buy（進場） | inside／B1／B1−1 | FutBid → `quote_ab = FutBid/P − 1` | 現貨 print ≤ P，先吃 `depth_ahead` | 賣 1 口期貨 Bid L1–L5 | `t_below_{f}`（eff_u < f）、`t_ab0` |
| S1 | sell（出場側） | inside／A1 | FutAsk → `quote_ab = FutAsk/P − 1` | 現貨 print ≥ P，先吃 `depth_ahead` | 買 1 口期貨 Ask L1–L5 | `t_above_{g}`（basis 比掛單時高 ≥ g bp） |
| S2（期貨 maker） | sell（進場） | A1 − 1 tick | SpotA1 → `P/SpotA1 − 1` | 期貨 print ≥ P（隊列第一） | 買現貨 Ask L1–L5 | `t_below_{f}`、`t_ab0` |
| S2 | buy（出場側） | B1 + 1 tick | SpotB1 → `P/SpotB1 − 1` | 期貨 print ≤ P（隊列第一） | 賣現貨 Bid L1–L5 | `t_above_{g}` |

- 每列都有：`quote_ab`、`eff_u`（相對當時 anchor）、`t_fill_ns`（S1 另有 `t_partial_ns`）、`hedge_ns`／`hedge_vwap`／`actual_ab`／`d_in_realized`、`t_gate_ns`、`t_up_{d}`／`t_down_{d}`（對向可掛價上／下移 ≥ d bp）。
- S1 每列附使用者的 makerFill：`mf_bid1_s / mf_bid2_s / mf_ask1_s / mf_ask2_s`（該現貨 tick 的四個成交秒數）與本層對應的 `mf_fill_ns`（L1／L2 才有）。
- 出場側的准入：`eff_u ≤ +10 bp`（出場目標都在 anchor 附近或以下）；出場側只留 inside／L1；S1 出場側排隊桶 20%。
- 一個部位的出場 = 它 hedge 完成後、同商品出場側的第一列滿足 `quote_ab ≤ B_x` 者（E1 用 S1 sell 列、E2 用 S2 buy 列），撤單條件用 `t_above_{g}`，g ≈ `B_x − quote_ab` 取上一格。跨日直接接次日的表。

### 4.3 邊界

- 點位表是 independent-event 口徑：時序排入時同商品同 stream 只能有一張活單，獨立事實才成立；多層同掛需 Stage 3 的共用隊列版。
- 被容量擋掉的進場其出場列閒置即可；「晚一點再掛」的情境不用重算，表在時間上稠密。
- `t_fill < t_below` 是「不撤單可吃到」，不是無風險：hedge 打到哪、出場等多久都在表裡，EV 如實反映。

## 5. 輸出

```text
data/points_<run>/Date=<D>/
├── s1_entries.parquet     每個 (商品, epoch, U) 一列
├── s2_entries.parquet     每個 (商品, 掛單事件) 一列
├── e1_exits.parquet       每個 (paired entry, L) 一列
├── e2_exits.parquet
├── hedges.parquet         每次 hedge 嘗試（含失敗重試）
└── manifest.json          來源 hash、參數、事件計數、cancel_race／negative_basis／timeout 計數
```

Stage 3 用這些表做時序回測與 Q／設定值校準；Q 本身只讀市場 1 Hz 格。

## 6. 驗收與對帳

- **四個固定日對帳**（5/11、6/1、7/6、8/3，全商品，S2、25 bp、深度 5×、hedge 完成即重掛、不套容量）：
  maker `S2_ENTRY_REASSESSMENT_20260914` 四日平均成交 **463.5 筆／日**（逐日 426／1,172／172／84）、
  hedge 後平均 premium 27.94 bp、34.0% 成交 hedge 後變差、18/1,854 負 basis、28.8% 在掛出 50 ms 內成交。
  重製結果要同量級；差異需歸因到明確的規則差（例如 print 同 ns 處理順序）。
- **S1 量級對照**：q95 BID1 approximate fill 12.29%、BID2 1.08%；本線 S1 在同 rank 的「撤單前成交率」不應高於這個量級的 2 倍（近似口徑不同，只防離譜）。
- **執行成本對照**：S2 進場衰減均值 8.92 bp（成交前 3.40 + 成交後 5.52）、hedge 變差比例 19.94%、超一 tick 6.65%；
  E1 出場 shortfall 的分布形狀（S2 右尾 p95 ~80 bp）。
- 單元測試：撤單競賽三種時序（`t_print < t_c`、`t_c < t_print ≤ t_c+50ms`、`> t_c+50ms`）；同 ns print 不能成交剛送出的單；
  一筆 print 只消耗一次；五檔外不假設零隊列；hedge 深度同 snapshot 不重複；S1 partial → rollback；一 tick 寬期貨盤口不掛 S2；
  兩市場 seq 不跨市場比較。

## 7. 訊息負荷（只量測，不節流）

每日輸出 S1／S2／E1／E2 的新掛＋撤單訊息數與滾動 1 秒峰值。maker 事件版 S2 峰值：期貨 74／s、股票 182／s；
專案先前參考上限期貨 5／s、現貨 100／s。Stage 2 只記錄，Stage 4 再決定容忍帶與排程。
