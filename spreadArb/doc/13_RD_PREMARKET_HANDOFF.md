# spreadArb 盤前交付與實作盤點

版本：2026-09-30，對應 [RD交易規格](12_RD_TRADING_SPEC.md) 的絕對價差回零／Q表／結算方案共同進場評估與B動態門檻；其中absolute勝出時採回零評分＋max-Q實際出場。以下檔名與新增欄位是**本次建議的交付契約，exporter／線上loader尚待實作**，不是聲稱倉庫已每日產出這些檔。

先用白話理解：研究端每天交給交易程式一份「今天可以交易誰」、一份「今天怎麼評分」，再附一張「確認兩份檔案是同一天、同一版」的清單。RD不必在交易程序裡重新讀過去幾十天行情建表；但當天行情、委託、成交和真實部位必須由RD自己維護。

本版另納入退檔撤單、四路出場優先、防自我成交、緩撮與結算risk所需資料。已確認13:24:55全撤、僅禁止當天到期合約新增S1／S2、risk秒間隔為`1,500/N`，見第4.5節。這些是確認後的交付契約，仍須實作exporter與交易端。

本文件前半說明每天交什麼，後半保留欄位名稱和型別供程式對接。不熟悉anchor、Q表或EV時，先讀 [交易規格第5～7節](12_RD_TRADING_SPEC.md)，裡面有從每秒行情到完整六方案比較的數字例子。

| 本文件常用詞 | 意思 |
|---|---|
| D／prev | 今天要交易的日期／前一個交易日；prev不是永遠日曆上的昨天 |
| bundle | 一次一起發布的整包檔案，三檔不能各拿不同版本 |
| schema | 檔案有哪些欄位、型別和意義的約定 |
| exporter／loader | 把研究資料寫成部署檔的程式／交易端讀入它的程式 |
| as-of／cutoff | 這份模型以哪一天作決策，以及資料最晚允許用到何時 |
| hash／sha256 | 檔案內容指紋；用來確認讀到的是同一份內容 |
| manifest | 本次檔案的版本、指紋、來源與驗證清單 |
| PMF／p_day | 各種結果的機率，這裡尤其指第0、1、2…日首次回零的各自機率 |
| ladder／回退 | 細分類樣本不夠，就按固定順序查較粗分類，不能自己猜一個機率 |
| carry | 前一交易日留下、今天還沒結束的部位或曝險 |
| warmup | 歷史資料還不足以形成全部機率格子的階段 |

## 1. 你每天需要交給 RD 什麼

正常盤前交付一個不可拆混版本的bundle：

```text
{D}/{bundle_id}/
  {D}_spreadArb_products.parquet
  {D}_spreadArb_tables.json
  {D}_spreadArb_manifest.json
```

例如D=20260930，products告訴交易程式「今天某股票對哪一口期貨，參考價多少、scale多少」；tables告訴它「查到某種行情狀態時用哪組機率、今天hurdle多少」；manifest讓它確認這兩份不是一新一舊。路徑是交付格式示意，不表示本次已產出20260930的可交易模型。

| 交付 | 內容 | 主要責任 |
|---|---|---|
| products | 今日可開倉商品、精確期貨合約、參考價／限制、scale、已知公司行動、交易日至到期時間 | 研究／資料端提供 |
| tables | residual Q表、absolute回零表、固定成本與策略參數、今日B hurdle及來源 | 研究／資料端產製，使用交易端前收資料 |
| manifest | 日期、版本、資料cutoff、來源hash、上述兩檔hash、檢查結果 | 盤前發布器產製 |
| `positions_{prev}.json` | 逐筆隔夜部位、原目標、已入帳現金與容量、未完hedge／rollback | RD交易程式每日寫出並恢復，不由研究重新推估 |
| `working_orders`＋回報日誌 | 所有已送未終結單、增量成交回報及去重位置 | RD交易程式持續保存、重啟恢復 |
| `B_state_{prev}.json` | 前一交易日實際收盤占用、完整獨立市場訊號數與q50及來源 | RD每日回傳研究／盤前產製端 |

最後三項是營運狀態交接，不能塞成「由研究用行情就能算出」的靜態盤前參數。即時行情、委託成交回報與官方收盤評價另有資料流，也不屬當日盤前檔。

建議責任分工：**研究端負責表與策略參數；RD負責行情／委託／部位狀態，並回傳B所需前日資料；資料平台提供當日主檔、日曆與公告。** 第一版沿用現有研究快取做export，先完成同值對照，再視部署環境拆除maker依賴。

## 2. 統一外層契約

為什麼每份檔都要有日期和版本？因為即使商品代碼一樣，今天可能換了合約、scale或hurdle。只確認「檔案存在」不夠，要確認它們是一包一起算好的資料。

兩個資料檔都攜帶 `trade_date`、`schema_version`、`bundle_id`、`policy_version`。建議本版 `schema_version=spreadarb_rd_v2`，新增execution與結算／回報契約；`policy_version=spreadarb_B_rd_20260930`與舊研究基準分開。schema版本代表檔案形狀，policy版本代表行為，不能只改日期或沿用舊版號假裝規則相同。

manifest至少有：

| 欄位 | 型別／規則 |
|---|---|
| `trade_date` | YYYYMMDD；載入日必須一致 |
| `bundle_id`, `schema_version`, `policy_version` | string；三檔必須相同 |
| `generated_at_utc`, `reference_asof_utc` | UTC時間；記錄主檔／公告實際可得截止點 |
| `statistics_cutoff_exclusive` | YYYYMMDD，等於D；統計輸入與可用標籤日期均<D |
| `status` | `ready`或`failed`；warmup用獨立旗標，不能含糊跳過驗證 |
| `files` | products/tables檔名、bytes、sha256、列數或格數 |
| `source_files` | 每個來源URI／路徑、日期、sha256、角色；禁止夾帶帳密 |
| `builder_version`, `source_hashes` | 產製器與策略計算來源版本 |
| `training_days`, `warnings`, `validation` | 明細及檢查結果；不得只寫PASS一個字 |
| `b_state_source` | 前日交易端狀態檔hash、日期、是否明確冷啟 |

歷史統計不可用D；當日商品參考價／合約表則必須用**D當日且盤前已可得**資料，不能誤解成所有資料都只能D−1。公告附available/announce時間，預告日曆只用截止點已公布的休市。

先寫暫存bundle、驗證兩檔，再原子發布manifest／ready指標。交易端先驗manifest再讀檔，禁止products取今日、tables取昨日，或載入一半更新。

「原子發布」的意思是：產製中先放交易程式不會使用的位置，全部完成後才一次切換成可讀版本。不是products剛寫好就讓交易程式開始用，而tables還在更新。

## 3. products：一列一組精確現貨／期貨合約

這份檔主要回答「我現在要訂閱和交易哪兩個商品」。除了今天允許新開倉的近月合約，也要包含手上仍持有的舊合約；否則RD可能開得到新倉，卻找不到舊倉該平哪一口。

主鍵為 `(trade_date, ValueCode, QuoteCode)`。每個現貨至多一列 `entry_enabled=true` 的當日新開倉合約；同時必須容納不同月份的持有合約列，不能只以現貨代碼當唯一鍵。

例如今天新倉用某股票的7月合約，但仍持有6月合約，就需要兩列：7月可新進場，6月只管理原部位。兩列是同一股票，但不能合併成一列或自動把庫存移過去。

| 欄位 | 型別／單位 | 必要性與來源 |
|---|---|---|
| `trade_date, schema_version, bundle_id, policy_version` | String | bundle識別 |
| `ValueCode, QuoteCode` | String | 現貨代碼／完整期貨合約代碼，保留前導零 |
| `expiry` | YYYYMMDD | 精確最後交易日；來源end_date；與正式合約主檔核對 |
| `is_expiry_day` | Boolean | 今日是否為該精確合約到期日；與expiry、D一致，不因換成近月映射就漏掉舊合約 |
| `settlement_rule_id`, `settlement_price_source` | String | 結算規則與正式結算結果來源識別；盤前不預填尚未產生的結算價格 |
| `contract_shares` | Int64，股／口 | 新進場限2,000；carry需核對原乘數，異動不得靜默沿用 |
| `decimal_locator` | Int16 | 期貨原始價格轉換，來源主檔，與行情交叉檢查 |
| `spot_ref_price_i, fut_ref_price_i` | Int64，元×10,000 | D當日參考價 |
| `spot_limit_down_i, spot_limit_up_i` | Int64 | D當日現貨主檔限制；既有spot marketData可供 |
| `fut_limit_down_i, fut_limit_up_i` | Int64 | 正式期貨主檔限制待接；研究用ref×0.9/1.1近似，不冒稱已存在正式欄位 |
| `limit_source` | String | official或research_approximation；研究近似檔不可冒充正式主檔 |
| `tick_rule_id` | String | 版本化tick規則；研究表及正式規則的映射 |
| `day_trade_mark` | String | 新進場研究映射接受X/Y；不要拿昨日同名欄位當今日資格 |
| `entry_enabled`, `exit_only` | Boolean | 新開倉映射／原持倉用途；`is_expiry_day=true`時entry_enabled必須false，entry=false仍須訂閱以管理舊倉 |
| `entry_block_reasons` | List[String] | 缺主檔、corporate、scale過大、結算日政策封鎖等；保留原因 |
| `corporate_block` | Boolean | 研究規則`announce_day<D<=effective_day`；搭配來源公告 |
| `corporate_events` | List[Struct] | 至少type、announce_day、effective_day、available_at、source_id；供風險流程追蹤 |
| `scale_bp, scale_raw_bp` | Float64 nullable，bp | `qlevel.table(D,20)`的scale及未截斷值；兩個都要保留 |
| `scale_n_obs, scale_n_days` | Int64 | 支持度；不可自行把n_days<5變成整檔禁交易 |
| `scale_as_of` | YYYYMMDD | 必須D，輸入日均<D |
| `k_cal` | Int32 | expiry−D，日曆日 |
| `k_td` | Int32 | D之後至expiry的預告交易日數；等於session_offsets長度 |
| `session_offsets` | List[Int32] | 嚴格遞增的日曆日差，包含未來到期日（若為已知交易日） |
| `settle_offset` | Int32 | **等於k_cal**；到期當日為0，不是下一交易日 |
| `metadata_source`, `metadata_asof_utc` | String／UTC | 可追溯主檔快照與實際可得時間 |

`product_cap_twd`不作整日固定盤前值。只提供tables內的`product_cap_frac=0.25`；交易端每次用當下單組名目算`max(nominal,5M)`。參考價單組名目可作檢視欄位，不能作唯一容量依據。

結算日禁止進場按精確合約的`trade_date == expiry`判斷，`entry_block_reasons`加入`expiry_day`。目前只交易近月個股期貨，這批合約到期日相同，所以當日整批禁止S1／S2；未到期合約不因別的商品結算而自動封鎖，也不代表它一定通過其他進場條件。不能把今日所有近月剛好同日到期，寫成永久的全策略日期開關。

可新增營運allowlist／停牌旗標，但本版沒有已生效的`target_list.txt`交易流程；若要縮小交易宇宙須另記政策差異。持有合約缺資料時不得改接另一合約，需輸出異常並保留風險管理。

## 4. tables：Q、回零表、成本及B門檻

建議頂層固定為：

```text
trade_date / schema_version / bundle_id / policy_version
reach
absolute
cost
policy
hurdle
execution
calendar_provenance
```

### 4.1 reach

這是「回到anchor附近某個目標」的機率表。它同時參與進場評分和目標選擇，不是只在absolute已通過後才使用的出場附件。交易端已從當日行情算出e、時間、距到期日數，再查表拿C0／C1／q。C0是當天到達比例；C1是到明天為止累計比例；q是前兩天沒到後的每交易日到達估計。交易規格第7節示範如何把三個數字轉成四種結果。

header含 `as_of=D, coord=residual_scaled, window_days=20, days_used, min_n=100, min_days=5`，以及 `e_edges=[1,1.5,2.5,4]`、`x_grid=[0,-0.25,-0.5,-1]`、`t_edges=[3600,9000]`、`k_edges=[0,1,3]`。

e與t桶為searchsorted right；k日曆日桶為left。t=3600是10:00、t=9000是11:30，**不是12:30**。

| 表 | 由細至粗的查表鍵 | 每格最少內容 |
|---|---|---|
| c0 | `(e_b,x,t_b,k_b)`→`(e_b,x,t_b)`→`(e_b,x)`→`(x)` | keys、values、level、n、n_days、p |
| c1 | `(e_b,x,k_b)`→`(e_b,x)`→`(x)` | 同上 |
| q | `(e_b,x,k_b)`→`(x,k_b)`→`(x)` | 同上 |

所有層都輸出，不能只送最細層。第一個`n>=100且n_days>=5`格可用；沒有就按交易文件的C0缺格／C1=C0／q=0處理。保留keys避免不同層欄位null碰撞。

例如最細格只有20個樣本，就不是拿它的機率硬算，而是查下一層；下一層有300個樣本、涵蓋8個交易日，才可使用。這裡的100筆和5天限制的是「該機率格」，不是說任何商品歷史不足5天都不能走其他評估方案。

單格形狀範例（數值僅示意，不能拿來交易）：

```json
{"keys":["e_b","x","t_b","k_b"],"values":[2,-0.5,0,3],"level":0,"n":512,"n_days":9,"p":0.41}
```

現在可用`reach.fit(D,20)`取得表，再迭代`c0/c1/q`或`frames()`輸出。se與信賴區間可附供診斷，不是必需的線上EV輸入。JSON使用null，禁止NaN／Infinity。

### 4.2 absolute

這是另一張表，回答「期現價差自然回到0，可能落在第幾個交易日」。它供回零進場分數使用；即使最後實際出場選了90bp，也不能先把這張表換成90bp的機率。

header含 `as_of=D, thresholds_bp=[50,100,150,200], k_edges_td=[0,2,5,10,15], max_j=15, min_n=30`，樣本來源hash、cutoff與實際覆蓋期間。

每格需 `level, thr_bp, k_b(nullable), n, p_day[16], p_later, p_settle`；level0=(thr,k桶)，level1=(thr)，不夠30筆往上退。機率各≥0且總和在容差內為1。

**不能直接交`AbsTable.frame()`的報表CSV替代模型**：它把第3～5、6～15日合併，已失去每天的T所需機率；必須從`AbsCell.p_day`完整匯出16個值。最近門檻等距選較低門檻。

例如「第3～5天合計30%」不足以算持有時間：30%全在第3天和全在第5天，占用時間不同。因此RD必須拿到每天分開的機率，而不是畫報表時為方便閱讀合併的欄位。

來源目前是`convergence_samples_enriched_20260126_20260625.parquet`。as-of只納入至少16個既有交易日前進場的樣本；短期首次到達已可知，仍未知的遠期結果歸結算，不可刪掉未收斂樣本。此表目前不是20日rolling；不要與residual表一起硬切20天。

新日期的樣本持續產製／補標籤流程仍待交付，不能把6/25截止的歷史樣本當作無限自動更新。

### 4.3 cost、policy

cost是算EV時的成本假設，policy是送單與容量等規則。它們要與Q表同版發布；RD不應另外硬寫一組不同常數，造成研究用11.9bp進場衰減、線上卻用其他值。實際成交後的帳務則用真實成交，EV預估衰減不再重扣。

cost欄位優先保留`CostConfig`名稱：`fee_same_day_bp=20, fee_overnight_bp=34, ev_min_bp=0, margin_bp=3, d_in_base={S1:25.7,S2:8.9}, d_out_base={S1:0,S2:0}, d_settle_base=0, min_anchor_bp=0`。

`d_in_override`若非空，序列化為`[{stream,tick_b,t_b,value_bp}]`；`d_out_override`為`[{stream,route,value_bp}]`，不要用不明確字串拼tuple鍵。tick桶邊界 `[5,15,30]`、time桶同reach，均right。

policy輸出既有B評分／容量的有效`PolicyConfig(**PRESETS['B'])`；第4.5節execution另輸出本次新增執行規則，以交易文件參數表核對。用`absolute_screen_mode=zero`作明確的交付識別欄位，只表示absolute路線使用回零評分，不表示只准absolute進場或停用residual／settle；它是新schema欄位，不是現有Python建構參數。`abs_target_mode=max_q`、`reserve_on_submit=false`、`max_positions_per_product=null`不可遺漏。

三種時間不可混用：hurdle輸入8.5是每交易日，內部base=8.5×250/365是每日曆日；EV的T是日曆日。發布完整精度，5.82只供畫面顯示。

### 4.4 hurdle與B狀態

這部分需要交易端回饋：研究可以算市場機會有多好，但不能只靠行情知道你真的用了多少資金。B必須同時知道前收盤真實占用與昨日獨立市場分數。

例：前收盤占1,700萬元，市場分數6、8、10、12，中位數9，今天門檻取9；若占用僅1,500萬元，就用基礎5.8219178。普通市場分數要包含「評分通過、但我們因滿倉未下單」的機會，不能只回傳成交清單。

| 欄位 | 定義 |
|---|---|
| `policy` | B |
| `base_bp_per_cal_day`, `today_bp_per_cal_day` | 完整精度數值 |
| `source_trade_date` | 前一交易日，不能只是D−1日曆日 |
| `committed_prev_close_cents` | RD實際策略名目帳的收盤值，整數分 |
| `signals_prev_n`, `signals_prev_q50` | 普通獨立市場訊號母體；已知0筆時q50=null |
| `quantile_method` | linear |
| `signal_definition_version` | 本版完整輸入去重、base篩選、exclude_refresh |
| `signal_source_hash`, `position_state_hash` | 可核對前日資料 |
| `cold_start`, `reason` | 明確區分首日、未碰80%、空分數、已抬門檻 |

至少每日回傳q50及筆數；為能獨立重算，RD還需保留`base_signal_scores_{D}.parquet`，或含完整決策鍵／origin／score的市場訊號日誌。可每日傳明細，也可傳摘要與能取回明細的hash/位置。

`committed_prev_close`不能從研究回測B_daily代替，不能用已實現盈虧或券商總戶頭餘額推算。若研究端重播前日行情來算市場分數，必須重現事件候選、完整輸入去重及**前一日當時生效的表**；舊文件的每秒取樣＋分鐘round去重不等價。

冷啟動無前日策略歷史：明確標cold_start並使用base。正常營運若部位檔／市場分數來源遺失，無法宣稱已依B規則產出hurdle，標failed並停新進場，既有曝險仍由恢復流程處理。這是新增交付完整性要求，不冒稱回測已有該營運判斷。


### 4.5 execution：本版新增的執行契約

本區與`policy`內既有B評分設定分開輸出。下列是RD新執行規則，不是現有`PolicyConfig`已接受的Python欄位；不能把整份execution直接展開傳給舊建構函式，也不能只匯出舊PRESETS就聲稱包含本次修改。

| 欄位 | 本版值／定義 |
|---|---|
| `timezone` | Asia/Taipei |
| `schedule.quote_start` | 09:01:00，四路一般maker開始許可 |
| `schedule.entry_new_end` | 12:53:00，停止新增S1／S2及重掛，不因截止立刻撤既有單 |
| `schedule.exit_new_end` | 13:20:00，一般日停止新增E1／E2及重掛 |
| `schedule.cancel_all_time` | 13:24:55；一般日、結算日均適用 |
| `schedule.cancel_all_scope` | 本策略全部商品、雙市場的尚未終結委託；不動其他策略單 |
| `cancel_on_futures_rank_loss` | S2／E2啟用；賣單P遇A1<P、買單P遇B1>P即送撤 |
| `entry_exit_conflict_rule` | 本版交易規格第9.7節四條規則；出場優先，先撤衝突進場單並等待終態 |
| `self_trade_check` | 每次maker、補腿、回補及結算taker真正送出前，檢查我方反向工作單及可成交範圍 |
| `pause_policy` | 緩撮商品雙市場全撤，期間禁止任何新委託；恢復正式狀態並核對完成後重評 |
| `settlement.entry_block_scope` | `expiring_contracts_only`：trade_date等於精確合約expiry時，全天禁止S1／S2及重掛 |
| `settlement.exit_new_end / cancel_time` | 13:00:00，到期E1／E2停止並撤相關工作單 |
| `settlement.futures_action` | 剩餘原合約等待正式結算；實際回報到才記帳 |
| `settlement.spot_slice_shares` | 1,000股；未滿一張另列殘股，不超賣 |
| `settlement.risk_interval_formula` | `1500 / risk_initial_lots`秒；N為計畫建立時可整張賣出的剩量，N=0不送單，不隨每次成交重開25分鐘 |
| `settlement.risk_horizon_seconds` | 1,500，僅供計算秒間隔，不能蓋過全撤截止 |
| `settlement.risk_end_time` | 13:24:55，不含此時點；停止排程新增／重試，剩餘有效委託一併全撤 |
| `settlement.risk_inventory_scope` | 只取歸屬到期合約的現貨；非到期庫存不納入 |
| `model_time_assumptions` | 明列沿用Q取樣09:05～12:53:20、到達觀測至13:18、EV終點13:20；不是新執行時段 |

工作單60秒壽命及既有價差／Gate撤單仍保留；新退檔、對撞、緩撮與結算切換是額外觸發。相同時間觸發多種撤單原因時，記下原因但不重複建立同一張單的撤單工作。

以上三項政策值已確認，`validation.pending_policy_fields`在本版應為空列表；時間、範圍與公式仍要按本節逐項驗證。全部資料與對照檢查通過後才能發布manifest為`ready`；政策已確認不代表產製器或交易端已完成。

risk排程採第一筆在13:00後撤單核清時送、後續按原間隔送出；13:24:55全撤優先，不因計畫名義長度25分鐘而在13:25追加。100張的間隔為15秒；300張的間隔為5秒，在理想13:00開始的例子中，第300筆落在13:24:55，須保留為未完成量。完整邊界見交易規格第11.3節。

## 5. 隔夜與重啟狀態：RD必須產出

同商品兩筆庫存可能各有不同出場目標。券商只說「持有股票4,000股、期貨空2口」，無法告訴你哪2,000股的目標是40bp、哪2,000股是80bp；所以策略必須有自己的逐筆帳，並用券商回報核對總量。

部位、工作單和未完成任務也要分開。已送撤但未確認的單仍可能成交；兩腿暫時歸零但還有已排定的補腿，也不能當作已結束。這是下表需要保存任務和回報位置的原因。

最少保留以下內容；建議以交易日、策略實例、部位ID隔離，券商彙總庫存不能還原每筆策略target。

| 類別 | 欄位／內容 |
|---|---|
| 快照header | date、instance_id、policy_version、bundle_id、snapshot_ns、last_report_cursor、reconciled |
| 身份 | position_id、ValueCode、精確QuoteCode、expiry、contract_shares、stream、state |
| 原決策 | quote_day/ns、quote_ab、anchor_entry、scale_entry、EV路線、原target_bp（可null）、EV/score/T/P_sd |
| 已成交 | 每腿買賣股／口數、累積cash、entry_spot_cash、entry_future_cash、fill_ns、hedge_ns |
| 出場進度 | winner_order_id、exit_route、exit_round、exit_fill/hedge時間、rollback／forced目的 |
| 容量 | 每部位committed_cents及估值/實際成本標記、全策略合計；不能只留淨部位市值 |
| 未完成工作 | task_id、purpose、instrument、side、原量／已成／剩量、已送單號、重試狀態 |
| 工作單 | client_order_id、broker_order_id、route、position_id、價量、累積成／剩量、send/live/cancel_request/terminal確認時間、狀態不明旗標、60秒deadline、撤單原因 |
| 防對撞等待 | ValueCode／精確QuoteCode、待掛出場路線、衝突工作單ID集合、等待撤單原因；等待期間阻擋哪些進場，恢復時重新核對 |
| 緩撮 | 股票／期貨各自市場狀態、來源時間與序號、商品新單封鎖、待撤單；重啟不能自行當成已恢復 |
| 結算risk | 歸屬到期合約的現貨剩量、在途賣單剩量、期貨待結算口數、risk_phase、risk_initial_lots、固定resSeecond／下一時點、起訖與已成現金 |
| 正式結算 | 原合約、結果唯一ID、結算價格／金額、確認時間、去重位置；不能只以當日時鐘宣告結算 |
| 帳務與去重 | 累積費用／現金明細索引、唯一成交ID集合或可重播日誌及offset |

保留原合約即使它不在今天新開倉universe。換月只改今日新進場映射，不能搬移舊部位。合約股數／標的／到期變更不能自動忽略，需走調整處理。

## 6. 現有資料能供什麼、還缺什麼

下表區分「研究程式已有算法／資料」與「已經有每日對外產檔服務」。前者很多已具備，後者不能只因舊文件寫了檔名就視為完成。

| 內容 | 現況／可取位置 | 本次RD交接仍須補 |
|---|---|---|
| 商品映射／參考價 | `maker/data/walkforward/daily/Date=D/mapping.parquet`；`maker/src/common/contracts.py` | 將D當日主檔供應做成獨立穩定介面，納入carry精確合約與可得時間 |
| 現貨漲跌停 | 上游`{D}_marketData.parquet`，由`common.books.raw_paths`解析 | 每日盤前可得欄位與檔案SLA；別把收盤後補齊欄位當盤前已知 |
| 期貨正式限制／tick／乘數異動 | 回測有參考價與DecimalLocator、標準乘數；限制部分用近似 | 正式主檔與交易介面能力核對 |
| scale | `src/ev/qlevel.py`，`data/qcache/hist/Date=d.parquet` | 小型products exporter，不需盤中載整份hist |
| residual Q | `src/ev/reach.py`，`data/qcache/facts/Date=d.parquet` | 各ladder完整序列化、版本及as-of驗證 |
| absolute Q | `src/ev/abs_reach.py`與taker收斂樣本 | 匯出完整PMF；新日期樣本延伸與可用標籤維護 |
| 成本／政策 | `src/ev/config.py`、`src/backtest/policy.py` | 有效設定export、變更版號；實盤成本另記 |
| 交易日曆 | `src/common/calendar.py`的2026研究常數 | 可更新、含公告可得時間的正式日曆；不是把這份研究常數永久部署 |
| 公司行動 | `maker/data/ev_lookup_v19_metadata_20260908/announcements/index.json` | 最新公告、停牌／處置／合約調整資料及風險流程 |
| 即時緩撮狀態 | 須由股票／期貨行情接口提供 | 明確開始／結束及時間序號，兩市場狀態映射；不是盤前預測欄位 |
| 自我成交防護 | 本版新增執行規格 | 同商品跨路工作單、撤單確認、pending出場與主動單成交範圍檢查 |
| 結算現貨risk／期貨結算 | 本版新增執行規格；舊研究用basis=0記帳 | 盤前提供日期與規則ID；盤中RD算剩量、分批成交，帳務端接正式期貨結算結果 |
| B前收占用與市場q50 | 回放checkpoint、`Date=D/base_signal_scores.parquet`、B_daily有研究證據 | 實盤RD收盤回傳，不可直接用歷史研究數值 |
| 持倉與工作單 | `Cycle`／Actor checkpoint有研究欄位參考 | 跨語言營運快照、回報去重與恢復，不直接依賴Python pickle |
| 官方日評價 | `backtest/valuation.py`按精確合約查價 | RD／帳務平台日終接入，不是當日盤前預先提供 |
| products/tables正式產製器 | 目前src無上述獨立premarket exporter／live loader | **尚未完成，舊10_PREMARKET的檔名只是一份設計** |

倉庫已有統計算法與研究來源；不是已經有每天可交RD的部署bundle。第一版不需要把全部歷史tick交給線上交易程式，歷史資料由產表端保留即可。

## 7. 產表端每天的工作

這些工作在研究／資料產製端做，線上交易程序只讀產出的表。以週二要交易為例：週一收盤後整理週一行情，更新先前樣本的已知結果，週二盤前用當時已知資料產週二模型；不能把週二盤中才會發生的結果算進去。

1. 收D−1兩市場資料、主檔、當時生效模型與交易端B狀態；保存原始接收時間及合約身份。
2. 建D−1因果秒格、hist與reach facts；較早樣本的d1／d2標籤隨當日資料補齊，保存版本／來源。`facts`不是只增不改。
3. 對D產scale與reach：最近20個既有交易日；每個歷史樣本的scale必須用該樣本日以前資料，不能把D的scale回填所有歷史樣本。
4. reach C0只用樣本日<D；C1還需d1_day<D且同合約，q還需d2_day<D且前兩日未達。缺資料與已到期要分開，不拼接下一月份。
5. 更新absolute樣本，再依D的16交易日seasoning與標籤可得性fit；保留來源覆蓋期間。
6. 建D商品及所有carry合約主檔，標記精確到期合約，將今日到期合約設為禁止新進場；用D已知日曆算horizon，接前收資料算B hurdle。
7. 匯出兩檔、執行驗證、發布manifest；失敗保留診斷，不默默沿用舊日期bundle。

以下是仍沿用的研究模型取樣契約，不是本版RD送單時段；09:01交易許可不能被這個歷史取樣條件擋掉。reach facts每商品於`sec=300,330,...,13980`取合格秒，凍結該取樣點anchor。`min_today`取其後至15480秒的合格`basis_buy_taker`最小值減凍結anchor；d1/d2取後一／二交易日300～15480秒、同合約的最小值減同一anchor。當日最低值不能包含t0本身。Q是市場目標到達統計，不是我方maker成交機率。

「facts」就是歷史觀察紀錄。例：週一10:00取一筆，記當時anchor，等週一結束才知道之後最低到哪；週二結束才知道下一交易日最低值。週二盤前還不能使用週二收盤才會知道的標籤，但可以使用已完成的週一當日標籤。

scale用前20日`resid=mid−anchor`的1bp histogram（−300～300，超界clip）合併，按bin內線性位置算q95−q50。raw不在[1,60]時scale=null，但raw必須留下；raw>60整體禁新進場，其他缺scale情境仍有absolute／settle。

若從零重建完整20日Q視窗，而這20日樣本各自scale也要求前20日，通常需往前準備約40個交易日的hist／行情，並處理d1/d2標籤截止；已有逐日因果scale快取則可重用。**不能保證舊文件所寫「23天就一定完整暖機」**。absolute另有16日seasoning、每格30筆及較長樣本來源要求。warmup應報可用格數／樣本日，不硬造機率填滿。

現有`ev.build`建立研究facts/hist，並不是上述部署exporter；也不應直接把含未來標籤的facts交線上查詢。線上只拿as-of已fit好的機率表。

## 8. 載入前驗證與RD對照

驗證不是只看JSON能不能讀。還要確認它沒有混日、漏合約、用到未來結果，且相同輸入送進RD程式與研究程式會得到相同的route、分數和出場目標。只對總獲利接近，找不出這些錯誤。

| 檢查 | 失敗時 |
|---|---|
| 日期／版本／hash一致，manifest ready | 整個bundle不供新進場 |
| execution與第4.5節確認值一致，pending_policy_fields為空 | 值不符或缺欄位時不發布ready；不得把13:25當作risk送單截止 |
| 到期身份、entry封鎖、13:00切換與risk範圍一致 | 不得把非到期庫存納入到期賣出，或把到期合約漏入一般開倉 |
| 今日新進場映射每vc至多一合約；精確carry覆蓋 | 缺合約不能用另一月代替；告警並處理原曝險 |
| 乘數／價位／DecimalLocator合法 | 該商品不開新倉；合約調整須另處理 |
| k_cal=settle_offset；K=len(offsets)；offsets嚴格遞增 | 拒絕錯誤商品時間參數 |
| reach days及可用label日期均<D | 拒絕含當日／未來資訊的模型 |
| 所有機率有限、[0,1]、absolute 16項且總和≈1 | 拒絕模型；不得normalize遮掉錯誤 |
| scale null及scale_raw排除語意完整 | 不得把null轉0並當有效模型 |
| B前日狀態日期／定義／hash可核對；hurdle按公式重算 | 缺正常營運來源不能偽裝冷啟 |
| 各層Q鍵唯一；完整PMF可round-trip回載 | 修export，不用報表聚合欄位代替 |

獨立對照至少涵蓋：x四格、各桶邊界、缺C0/C1/q、scale缺值與>60、absolute50/75/125/175邊界、到期當日與跨假日T、S2半tick floor、max-Q負EV候選、B偶數q50與滿額仍蒐集訊號。RD需保存同輸入的EV/T/route/target/reason對比結果。

既有回測沒有「商品至少50檔才可交易」、「hurdle不得超過60」、「每檔hist至少5日才可進absolute」這些限制；不得把舊盤前文件的建議驗證直接變成策略規則。

## 9. 本輪盤點結論與交接次序

可以把實作分成五個可展示的交付，不需要第一步就搬完整研究環境去線上。先交一個歷史日的完整檔案和已知答案，讓RD讀回並算對，再接每日自動更新及實單狀態。

| 次序 | 交付物 | 完成定義 |
|---|---|---|
| 1 | 本文件＋交易規格 | RD能分清研究基準與新執行規則；全撤時間、僅到期合約禁開倉及秒間隔公式均已確認 |
| 2 | 一個歷史日完整bundle與golden decisions | exporter/loader round-trip；EV、route、target、hurdle與研究同值 |
| 3 | 每日主檔／統計／公告發布流程 | 產製截止、輸入缺漏、版本與補檔可追溯 |
| 4 | RD回報／狀態／B回傳介面 | 部分成交、雙出場、斷線重啟、原合約carry能對帳 |
| 5 | 到期／資金／實際成本與營運政策 | 13:00撤單、現貨分批剩量與期貨正式結算可核對；實單帳不用basis=0假成交結束 |

你這邊最需要安排的來源是：**當日現貨與期貨主檔及精確到期日、因果統計表的每日產製、已公告日曆與公司行動、確認後的execution參數版本、正式結算結果來源**。RD則必須承擔**實際部位／委託狀態及B前日資料回傳**。上述責任確認後，才是正式bundle產製器與交易adapter的實作；這次整理沒有代替RD建立實盤下單系統。
