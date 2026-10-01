# negFill 日內放空策略

> 文件目的：讓 RD 可以直接理解 negFill 的策略意圖、資料來源、逐筆資料流、訊號條件、模型計分與委託生命週期。
>
> 本文件描述的是策略行為，不限定實作語言。所有逐筆狀態都必須以「交易日 × 股票」隔離，換日重置。
>
> 本文件以當日模型參數檔、`modelUpdate.py` 與 `negFillResearch_model` 的最終訊號 cell 為核對基準；若三者不一致，會在第 11.5 節明確列為待同步項目，不把實驗中間值當成正式設定。
>
> **本文件是 negFill 唯一正式策略規格，已取代 `negFill.md`。實作、測試與 code review 均以本文件為準。**

---

## 1. 先用一句話理解策略

negFill 想找的是：

**某檔股票在日內低點附近持續出現偏賣方的成交流，價差曾經張開一段時間，之後 B1 與價差再次變動；若當下價格、流動性與模型分數都合格，就用被動限價單建立空單。**

策略分成五層：

1. 盤前決定今天能觀察、能當沖的股票，並載入模型參數。
2. 開盤後用前一日熱門股的開盤表現，決定今天使用 normal 或 abnormal 模型組。
3. 每一筆 Tick 更新成交路徑、低點賣壓、價差事件及最近 30 筆市場活躍度。
4. 硬條件全部通過後，再讓兩個子模型一起表決。
5. 兩個模型都通過才掛空單；盤中負責停損，尾盤強制平倉。

目前稽核結論：九個模型 feature 的最終值可與研究模型對齊；其中 `ToRef`、`ToOpen` 必須覆寫 parquet 的 mid-price 原值。整體流程另有 rate Gate、最終風控、當日當沖資格來源、`Open` 可得時間與缺值補法等 research/live 差異，詳見第 11.5 節。

---

## 2. 資料來源顏色圖例

本文每個欄位前都用固定標籤表示資料從哪裡來：

| 標籤 | 類型 | 定義 |
|---|---|---|
| <span style="color:#c62828;font-weight:700">● [TICK]</span> | Tick 逐筆輸入 | 策略從 `tickData` 每筆直接讀到的時間、成交、五檔價量、試撮狀態，以及已附在 Tick 上的 `RefPrice`、`OTC`。前者是交易所行情，後兩者是上游補入的商品資料，但對 negFill 都屬直接 Tick 輸入。 |
| <span style="color:#ef6c00;font-weight:700">● [PRE]</span> | 盤前資料 | 08:15 前已準備好的商品靜態資料、前一交易日統計、當沖資格與觀測清單。 |
| <span style="color:#1565c0;font-weight:700">● [CALC]</span> | 策略計算資料 | 由 Tick、盤前資料或全市場快照計算出的狀態、特徵、排名與布林條件。 |
| <span style="color:#6a1b9a;font-weight:700">● [MODEL]</span> | 模型檔資料／輸出 | 每日模型 JSON 直接提供的 `suspended_list`、係數、截距、門檻，以及計算後的模型分數。 |
| <span style="color:#2e7d32;font-weight:700">● [ORDER]</span> | 委託與部位狀態 | 策略自己的未成交單、成交量、空單部位、停損狀態。 |
| <span style="color:#455a64;font-weight:700">● [PARAM]</span> | 策略設定 | 時間、排名、價格、流動性、部位與停損門檻；正式值集中在第 13 節。 |

如果閱讀環境沒有顯示 HTML 顏色，仍可依 `[TICK]`、`[PRE]`、`[CALC]`、`[MODEL]`、`[ORDER]`、`[PARAM]` 文字辨識來源。

### 2.1 完整欄位與狀態來源索引

下表是本策略會用到的完整變數索引。凡是現有資料已經有欄名，一律使用原欄名，不建立另一個策略別名。

| 類型 | 變數原名 | 實際來源 | 備註 |
|---|---|---|---|
| <span style="color:#c62828;font-weight:700">[TICK]</span> | `QuoteCode`, `ChannelSeq`, `TransTime`, `TrialMatch` | `tickData` | 商品、順序、時間與試撮狀態。 |
| <span style="color:#c62828;font-weight:700">[TICK]</span> | `BidPrice1…5`, `BidLots1…5`, `AskPrice1…5`, `AskLots1…5` | `tickData` | 五檔價量。 |
| <span style="color:#c62828;font-weight:700">[TICK]</span> | `FillPrice`, `FillLots`, `FillLots_origin`, `InOut` | `tickData` | 成交價、處理後成交量、原始成交量與成交方向；`FillLots` 已由 tick pipeline 將試撮量歸零，`FillLots_origin` 只供追溯。 |
| <span style="color:#c62828;font-weight:700">[TICK]</span> | `RefPrice`, `OTC` | `tickData` | 上游已附在逐筆資料上的參考價與市場別。 |
| <span style="color:#ef6c00;font-weight:700">[PRE]</span> | `QuoteCode`, `allow_day_trade_mark`, `PreviousClosePrice`, `day_amount_rank`, `hft_strick_makerSpreadBP`, `avg_bidLots1`, `avg_askLots1` | `src/strategy/preMarket/{TradeDate}_preMarketData.parquet` | 檔名是使用日，但欄位由前一交易日資料產生；`allow_day_trade_mark` 也屬前一日，不能當成當日 T30 資格。 |
| <span style="color:#ef6c00;font-weight:700">[PRE]</span> | `allow_day_trade_mark` | 使用日當天的 `marketData`／等價商品主檔 | 與上一列同名但來源日期不同；研究流程用這一份判斷當日是否可當沖。RD 必須用來源日期區分，不能另改欄名。 |
| <span style="color:#1565c0;font-weight:700">[CALC]</span> | `marketOpen`, `Open`, `RecordHigh`, `RecordLow` | `src/features/definitions/basic_info.py`；研究資料存於 `tickData` | 上游先算好再寫進 `tickData`，不是交易所原始欄位。 |
| <span style="color:#1565c0;font-weight:700">[CALC]</span> | `ToLow`, `ToHigh`, `Low_High`, `BidPreMove`, `TickSize`, `FillLots_atLow` | `src/features/definitions/price_dynamics.py`；研究資料存於 `tickFeature` | 模型、候選事件或風控依各自用途沿用 parquet 值。 |
| <span style="color:#1565c0;font-weight:700">[CALC]</span> | `ToRef`, `ToOpen` | parquet 原值來自 `price_dynamics.py`，模型載入後再覆寫 | parquet 用 mid price；negFill 模型最終用 `BidPrice1` 並四捨五入至 6 位。 |
| <span style="color:#1565c0;font-weight:700">[CALC]</span> | `B1_A1B1`, `Spread`, `SpreadPairID`, `SpreadPairAsk`, `SpreadPairBid`, `SpreadPairSeq`, `SpreadPairElapsed` | `src/features/definitions/orderbook.py`；研究資料存於 `tickFeature` | 模型／候選事件直接沿用 parquet 值。 |
| <span style="color:#1565c0;font-weight:700">[CALC]</span> | `SpreadNarrowOrderTime`, `SpreadPairTotalCount`, `SpreadCountAtSameCount`, `SpreadNarrowSide` | `src/features/definitions/orderbook.py`；研究資料存於 `tickFeature` | 只在收盤後產製 `hft_strick_makerSpreadBP` 時使用。 |
| <span style="color:#1565c0;font-weight:700">[CALC]</span> | `Close`, `FutureHigh`, `FutureLow` | `src/features/definitions/basic_info.py`；研究資料存於 `tickData` | 含收盤後／未來資訊，只可產製次日盤前特徵，絕不可作為當日逐筆模型輸入。 |
| <span style="color:#1565c0;font-weight:700">[CALC]</span> | `RemainSeconds` | `src/features/definitions/time_features.py`；研究資料存於 `tickFeature` | 整數秒。 |
| <span style="color:#1565c0;font-weight:700">[CALC]</span> | `MD_L1Rate_30`, `MD_ElaspeTime_30` | `src/features/definitions/microstructure.py`；研究資料存於 `tickFeature` | 模型沿用比例，並將 elapsed 另做 `ln(1+x)`。 |
| <span style="color:#1565c0;font-weight:700">[CALC]</span> | `AmountRank_canDayTrade` | `tickBar`／全市場每 5 分鐘計算 | 當日截至目前的可當沖商品成交金額排名；盤中重啟後只使用重啟後重新累計的 `Amount`。 |
| <span style="color:#1565c0;font-weight:700">[CALC]</span> | `MD_ElaspeTime_30_re` | `ln(1 + MD_ElaspeTime_30)` | 模型使用的轉換值；名稱沿用模型 schema。 |
| <span style="color:#1565c0;font-weight:700">[CALC]</span> | `MD_L1Rate_30_re` | `negFillResearch_model`／`modelUpdate.py` 令它等於 `MD_L1Rate_30` | 研究流程的同值暫存別名；正式模型 feature list 與模型 JSON 對接仍使用 `MD_L1Rate_30`。 |
| <span style="color:#1565c0;font-weight:700">[CALC]</span> | `RecentFillPrice`, `SignedFillLots`, `spreadChanged`, `b1Changed`, `negativeFill`, `candidateEvent` | 策略依 Tick 逐筆維護 | 這些不是上游資料欄位，而是為了說明流程使用的內部狀態。 |
| <span style="color:#1565c0;font-weight:700">[CALC]</span> | `toOpen`, `topOpen`, `validCount`, `isTop100`, `preMarketUniverse`, `candidateUniverse`, `eligibleUniverse` | 策略由 Tick、盤前檔、排名與模型清單計算 | `toOpen`、`topOpen` 沿用 `modelUpdate.py` 名稱；其餘是集合、統計或 Gate 狀態，不要求上游提供同名欄位。注意模型 feature `ToOpen` 是另一個大小寫不同的欄位。 |
| <span style="color:#1565c0;font-weight:700">[CALC]</span> | `distance`, `depth`, `SpreadPairStartTime`, `elapsedSeconds`, `hasFill`, `windowCount` | 各特徵公式的內部暫存值 | 不寫回上游資料；`SpreadPairStartTime` 只用來算既有欄位 `SpreadPairElapsed`。 |
| <span style="color:#1565c0;font-weight:700">[CALC]</span> | `Amount`, `TimeSlot`, `_amt`, `_day_amount` | `crossSection.py`、`preMarketSummary.py` | 排名計算的既有欄位或上游暫存欄位；公式見第 5.4 節。 |
| <span style="color:#1565c0;font-weight:700">[CALC]</span> | `open_price`, `opening_ref_price`, `allow_day_trade_mark_x` | `data/marketData` 與外層 `data/preMarket` 合併結果 | 只供 `modelUpdate.py` 離線建立 `topOpen`／`abnormal_dates`；不是營運盤前檔的線上契約。 |
| <span style="color:#1565c0;font-weight:700">[CALC]</span> | `activeModelA`, `activeModelM`, `allModelFeaturesValid`, `passGate`, `scoreA`, `scoreM`, `passModelA`, `passModelM`, `passModels` | 模型選擇、Gate 與計分流程 | 這些是策略內部結果，不是輸入欄位。 |
| <span style="color:#1565c0;font-weight:700">[CALC]</span> | `accLots` | `negFillResearch_model` 最終風控 cell | 當日同商品累計模型訊號筆數。 |
| <span style="color:#1565c0;font-weight:700">[CALC]</span> | `restartTime`, `brokerRecoveryComplete`, `newEntryEnabled` | 盤中重啟流程 | 分別記錄本次重啟時間、人工Current Position File基準與DT3 Report／Working Order recovery是否完成，以及目前能否開新倉；不使用DT3 Position Query。 |
| <span style="color:#6a1b9a;font-weight:700">[MODEL]</span> | `suspended_list`, `normal_model`, `normal_model_M`, `abnormal_model`, `abnormal_model_M` | `src/strategy/negFill/modelParam/{TradeDate}_modelParams.json` | 每日模型檔頂層欄位；本策略不使用 `target_list`。 |
| <span style="color:#6a1b9a;font-weight:700">[MODEL]</span> | `coefficients`, `intercept`, `threshold` | 上述四個模型區塊 | 逐筆模型計分參數。 |
| <span style="color:#6a1b9a;font-weight:700">[MODEL]</span> | `metadata.trained_until`, `metadata.alpha`, `metadata.otc_turnover_threshold` | 模型檔 `metadata` | 模型版本與訓練資訊，不直接進逐筆分數。 |
| <span style="color:#2e7d32;font-weight:700">[ORDER]</span> | `entryPrice`, `entryLots`, `projectedSymbolExposureValue`, `projectedStrategyExposureValue`, `net_position_qty`, `stopLossArmed`, working order ID／價格／剩餘量 | 策略委託與成交回報 | 這些不是行情資料欄位，由交易狀態機維護；`net_position_qty`方向與券商庫存一致，正值多單、負值空單；兩層曝險上限由`risk.toml`載入。`stopLossArmed`只控制下一次停損穿越訊號，不封鎖新單。 |
| <span style="color:#455a64;font-weight:700">[PARAM]</span> | 第 13 節全部門檻 | 策略設定或模型檔 | 每一個參數的正式值與用途都只在第 13 節定義。 |

文中的「B1／A1」只是閱讀用簡稱，永遠分別代表現有欄位 `BidPrice1`／`AskPrice1`；資料介面不可另外建立 `B1`、`A1` 欄位。

---

## 3. 全策略資料流

```mermaid
flowchart LR
    PRE["🟠 營運盤前檔<br/>商品清單、當沖資格、前日排名、前收價"]
    PARAM["🟣 每日模型檔<br/>係數、截距、門檻、排除清單"]
    TICK["🔴 Tick 五檔<br/>時間、成交、B1~B5、A1~A5、RefPrice、OTC"]
    RANK["🔵 全市場每 5 分鐘快照<br/>即時累計成交金額排名"]
    STATE["🔵 每檔股票逐筆狀態<br/>成交路徑、低點賣壓、SpreadPair、30-MD"]
    GATE["🔵 硬條件 Gate"]
    SCORE["🟣 兩個子模型計分"]
    ORDER["🟢 空單委託、部位、停損與平倉"]

    PRE --> STATE
    PRE --> GATE
    PARAM --> SCORE
    PARAM --> GATE
    TICK --> STATE
    TICK --> GATE
    RANK --> GATE
    STATE --> GATE
    GATE --> SCORE
    SCORE --> ORDER

    style PRE fill:#fff3e0,stroke:#ef6c00,color:#7a3700
    style PARAM fill:#f3e5f5,stroke:#6a1b9a,color:#4a126b
    style TICK fill:#ffebee,stroke:#c62828,color:#7f1717
    style RANK fill:#e3f2fd,stroke:#1565c0,color:#0d477f
    style STATE fill:#e3f2fd,stroke:#1565c0,color:#0d477f
    style GATE fill:#e3f2fd,stroke:#1565c0,color:#0d477f
    style SCORE fill:#f3e5f5,stroke:#6a1b9a,color:#4a126b
    style ORDER fill:#e8f5e9,stroke:#2e7d32,color:#19521f
```

### 3.1 每筆 Tick 的處理順序

每收到一筆行情，固定照以下順序處理：

1. 讀取 <span style="color:#c62828;font-weight:700">[TICK]</span> 行情並確認不是重複、倒序或跨日資料。
2. 更新 <span style="color:#1565c0;font-weight:700">[CALC]</span> 開盤價、最近成交價、日內高低點與低點累積買賣流。
3. 更新 <span style="color:#1565c0;font-weight:700">[CALC]</span> Spread、SpreadPair 與最近 30 筆 MD 特徵。
4. 計算九個模型特徵。
5. 執行硬條件 Gate；任一條失敗就結束這一筆。
6. 依當日市場狀態選擇模型組，計算兩個子模型分數。
7. 兩個子模型都通過，且部位仍有空間，才建立或修改放空委託。
8. 收到成交回報後，更新 <span style="color:#2e7d32;font-weight:700">[ORDER]</span> 空單部位。

---

## 4. 每日時間軸

| 時間 | 動作 | 主要輸入 | 產出與目的 |
|---|---|---|---|
| 08:15 | 盤前初始化 | <span style="color:#ef6c00;font-weight:700">[PRE]</span> 當日營運盤前檔、上市 T30、上櫃 T30；<span style="color:#6a1b9a;font-weight:700">[MODEL]</span> 當日模型檔 | 由盤前檔建立觀測商品與前日 Top 100；由當日 T30 決定能否放空及取得漲跌停價；由模型檔取得 `suspended_list` 及四個模型參數。 |
| 09:00 起 | 收集開盤價 | <span style="color:#c62828;font-weight:700">[TICK]</span> 第一筆有效成交 | 更新各股票的 <span style="color:#1565c0;font-weight:700">[CALC]</span> `Open`，並計算 `toOpen`；不要與模型 feature `ToOpen` 混用。 |
| 09:00:25 | 選擇今日模型組 | <span style="color:#ef6c00;font-weight:700">[PRE]</span> 前日 Top 100；<span style="color:#1565c0;font-weight:700">[CALC]</span> `topOpen` | `topOpen < 0.0025` 用 abnormal 模型組，否則用 normal 模型組；無法計算時預設使用 `normal_model + normal_model_M`。 |
| 每 5 分鐘 | 更新今日熱門股 | <span style="color:#1565c0;font-weight:700">[CALC]</span> 全市場截至當下累計成交金額 | 更新 `AmountRank_canDayTrade`；兩次更新之間沿用最近一次排名。盤中重啟後仍使用固定的 5 分鐘時間點，但排名只包含重啟後重新累計的金額。 |
| 09:00:30～12:00 | 允許建立空單 | Tick、盤前欄位、計算特徵與模型 | 只有嚴格大於 09:00:30、嚴格小於 12:00:00 的 Tick 可觸發新空單。 |
| 盤中 | 停損 | `AskPrice1`、`RefPrice`、`net_position_qty` | armed狀態下`AskPrice1 >= RefPrice × 1.08`時，以漲停價回補當下負值Position代表的空單；它是broker report／Fill與recovery barrier之後最高優先的可送交易動作，但不取消或封鎖開倉單；之後須嚴格跌到`RefPrice × 1.06`以下才重新armed。只有今日觀察名單內的商品參與停損。 |
| 12:00 | 停止開倉 | 時鐘、未成交開倉單 | 不再接受新訊號；取消仍未成交的開倉賣單。 |
| 12:45 | Maker 平倉 | B1、Current Position、Working Order | 僅處理今日觀察名單：取得完整Working Order快照、取消開倉賣單、固定等待1秒並持續套用Fill，再以最新B1對負值Position所代表且尚未被有效Cover leaves覆蓋的空單掛ROD MakerCover。 |
| 13:14 | Taker 平倉 | Current Position、Working Order、漲停價 | 僅處理今日觀察名單：取得完整Working Order快照、取消開倉賣單、等待1秒；StopCover維持，MakerCover同單改到漲停價，只有未被有效Cover leaves覆蓋的差額才新建TakerCover。 |
| 13:20 | 關帳後檢查 | Current Position、未決Entry／Cover | 檢查全部商品；signed Position非零或仍有Pending／Working／Unknown委託即告警交易員用券商APP確認並人工處理，不查DT3 Position、不自動追加Cover。 |

---

## 5. 輸入資料契約

### 5.1 Tick 逐筆直接輸入

以下欄位全部是 <span style="color:#c62828;font-weight:700">[TICK]</span>，negFill 從每筆 `tickData` 直接取得。`RefPrice`、`OTC` 雖由上游商品資料附加，策略本身不再從盤前檔重算：

| 欄位 | 意義 | 策略用途 |
|---|---|---|
| `QuoteCode` | 股票代碼 | 所有逐筆狀態的分組鍵。不同股票絕不可共用狀態。 |
| `ChannelSeq` | 行情序號 | 同股票同交易日的去重與順序檢查。 |
| `TransTime` | 交易所事件時間 | 交易時段判斷、SpreadPair 計時、30-MD 經過時間與剩餘秒數。 |
| `TrialMatch` | 是否為試撮 | `0` 才是正常盤中行情；試撮不更新正式成交狀態，也不可產生訊號。 |
| `BidPrice1…5` | 買方一至五檔價格 | B1 用於訊號、模型及委託價；其餘檔位保留給行情完整性與未來特徵。 |
| `BidLots1…5` | 買方一至五檔掛單量 | 目前模型使用 `BidLots1`。 |
| `AskPrice1…5` | 賣方一至五檔價格 | `AskPrice1` 用於 Spread、模型、空單委託價及盤中停損觸發。 |
| `AskLots1…5` | 賣方一至五檔掛單量 | 目前模型使用 `AskLots1`。 |
| `FillPrice` | 當筆成交價；沒有成交時為 `0` | 更新最近成交價、開盤價及日內高低點；盤中停損觸發改用 `AskPrice1`。 |
| `FillLots` | 當筆成交張數；沒有成交時為 `0` | 計算成交封包比例及累積買賣流。成交量本身不帶方向。 |
| `InOut` | 成交方向 | `+1` 代表買方主動成交、`-1` 代表賣方主動成交、`0` 代表本筆沒有可計入的方向。 |
| `RefPrice` | 當日開盤參考價 | 計算 `ToRef`、`ToLow`、`ToHigh`、進場價格範圍、`topOpen` 與停損線。研究用 `tickData` 已直接帶有此欄。 |
| `OTC` | 是否為上櫃商品 | 決定是否略過上市股票的 `MD_L1Rate_30` Gate。研究用 `tickData` 已直接帶有此欄。 |

`tickData` 在寫 parquet 前已做以下上游處理；線上重現 feature 時也要使用處理後語意：

```text
所有 *Price* 欄位 = 上游整數價格 / 10000
FillLots_origin    = 上游原始 FillLots
FillLots           = FillLots_origin × (TrialMatch == 0)
RefPrice            = marketData.opening_ref_price
OTC                 = (marketData.market == "otc")
marketOpen          = (TrialMatch == 0) 從第一次成立後的累積 OR
```

上式的 `/ 10000` 是研究 parquet 的可讀元價格表示。正式低延遲交易程式不得在行情熱路徑把價格全面轉成浮點數：所有行情價格、狀態價格、tick 計算及 Gate 一律保存為 `price_x10000` 整數；只有 `ToRef`、`ToOpen` 等比例型模型特徵才在計算時轉成 `double`，並依模型規格 round 至小數第 6 位。送往 DT3 時再由 BrokerAdapter 將 `price_x10000` 轉成 DT3 runtime layout 所需的元價格格式。

之後 `tickFeature` 只保留 `marketOpen == true` 的 Tick，再依 `price_dynamics → orderbook → time_features → microstructure` 等 feature group 順序計算。因此最近 30 筆是「開盤狀態成立後的 Tick」，不包含盤前試撮列。

Tick 價格或數量無效時的處理：

- `BidPrice1 <= 0`、`AskPrice1 <= 0` 或 `AskPrice1 < BidPrice1`：本筆不可產生訊號。
- 任一使用到的委託量小於 `0`：視為壞資料，本筆不可進模型。
- 同一 `(交易日, QuoteCode, ChannelSeq)` 重複：忽略第二筆以後的資料。
- 同股票時間或序號倒退：忽略且告警，不可寫入逐筆狀態。

### 5.2 盤前檔輸入

線上直接讀取的檔案是：

```text
src/strategy/preMarket/{TradeDate}_preMarketData.parquet
```

這份檔案由前一交易日的 `marketData + tickData + tickFeature` 彙整後產生，再以「下一個交易日」命名。例如 `20260813_preMarketData.parquet` 是用 2026-08-12 收盤後的資料產生，供 2026-08-13 盤前使用。

已實際比對 `src/strategy/preMarket/20260813_preMarketData.parquet`：共 `42,744` 筆，其 `allow_day_trade_mark` 與 `data/marketData/20260812_marketData.parquet` 在全部商品上完全相同，確認該欄是來源日資料，不是 2026-08-13 當日資格。

以下是 negFill 會直接使用的 <span style="color:#ef6c00;font-weight:700">[PRE]</span> 欄位；名稱以實際營運檔 schema 為準：

| 實際欄名 | 策略內名稱 | 意義與用途 |
|---|---|---|
| `QuoteCode` | `QuoteCode` | 商品代碼，也是盤前檔與 Tick／模型清單的 join key。盤前檔中的所有 `QuoteCode` 合起來就是 `preMarketUniverse`，不是另有一個同名欄位。 |
| `allow_day_trade_mark` | `allow_day_trade_mark` | **來源交易日**的當沖註記。因檔案以次交易日命名，這不是使用日當天的最終資格；只可追溯或預篩，不可單獨決定今日能否放空。 |
| `PreviousClosePrice` | `PreviousClosePrice` | 前一交易日收盤價。它用來產生模型檔的高價股排除清單，也可在盤前檢核 Tick 的 `RefPrice`；逐 Tick 公式仍使用 Tick 的 `RefPrice`。 |
| `day_amount_rank` | `day_amount_rank` | 前一完整交易日的成交金額排名，`1` 表示金額最大。用來建立盤前 Top 100。 |
| `hft_strick_makerSpreadBP` | 同名 | 前一交易日的 HFT maker spread 統計。模型產製程序用它建立 `suspended_list`，研究 notebook 最終風控也直接要求它為 null 或嚴格大於 `-70`。 |
| `avg_bidLots1` | 同名 | 前一交易日 `BidLots1` 平均值；研究 notebook 用於限制同商品累計訊號筆數。 |
| `avg_askLots1` | 同名 | 前一交易日 `AskLots1` 平均值；與 `avg_bidLots1` 相加形成累計訊號筆數上限。 |

盤前檔實際還有 `big_buy_*`、`big_sell_*`、其他 `hft_*`、`negFill_*`、`day_lots_rank` 等研究特徵；目前九個逐筆模型特徵與最終風控沒有直接使用這些其餘欄位。

上述三個盤前風控 feature 在 `src/features/preMarketSummary.py` 的實際算法：

```text
avg_bidLots1 = 前一日 marketOpen Tick 的 mean(BidLots1)
avg_askLots1 = 前一日 marketOpen Tick 的 mean(AskLots1)

hft 候選列：
    SpreadNarrowOrderTime < 0.07
    AND SpreadCountAtSameCount == 0
    AND abs(SpreadPairElapsed - SpreadNarrowOrderTime) < 0.000001

賣側樣本：SpreadNarrowSide == -1、AskPrice1 != 0、AskPrice1 <= FutureHigh
買側樣本：SpreadNarrowSide ==  1、BidPrice1 != 0、BidPrice1 >= FutureLow

把買、賣兩側樣本數補到相同；數量較少的一側以當日 Close 補值
A_mean = 補值後賣側 AskPrice1 平均
B_mean = 補值後買側 BidPrice1 平均

hft_strick_makerSpreadBP = (A_mean - B_mean) / B_mean × 10000
```

`FutureHigh`、`FutureLow`、`Close` 都要等來源日結束才完整，因此這個 HFT feature 只能在收盤後產生、供次日使用；盤中不可重算當日值。

使用日當天的 `marketData.allow_day_trade_mark` 不是目前 `preMarketData.parquet` 內那一份前日同名值，必須由當日商品主檔或語意相同的 T30 來源提供。正式可當沖條件應以使用日當天的結果為準：

```text
可當沖 = 使用日當天的 allow_day_trade_mark == "X"
```

若當日資格來源缺失，策略不得只拿盤前檔內的前日 `allow_day_trade_mark` 代替；應停止建立新空單並告警。

#### 外層歷史盤前檔不是線上輸入契約

研究資料另有：

```text
data/preMarket/{Date}_preMarketData.parquet
```

它同樣由來源日資料產生並以次交易日命名，但比營運盤前檔多出 `market`、`ins_type`、`nextday_allow_day_trade_mark`、`turnover_rate`、`turnover_rate_rank` 等欄位。`modelUpdate.py` 以檔名日期把它與當日 `marketData` 合併，因此 `day_amount_rank` 是前日排名，而 `allow_day_trade_mark_x` 來自當日 `marketData`。這些額外欄位可供離線研究或模型產製使用，但在實際檢查的營運盤前檔 `src/strategy/preMarket/20260813_preMarketData.parquet` 中不存在。因此 RD 不可假設線上盤前檔有它們；是否為 OTC 應直接讀 Tick 的 `OTC`。

### 5.3 每日模型檔輸入

線上直接讀取：

```text
src/strategy/negFill/modelParam/{TradeDate}_modelParams.json
```

以下全部是 <span style="color:#6a1b9a;font-weight:700">[MODEL]</span>，不是 `[PRE]`：

| 實際欄位 | 策略內名稱 | 意義與用途 |
|---|---|---|
| `suspended_list` | `suspended_list` | 當日不可交易的 `QuoteCode` 清單；即使盤前檔與其他 Gate 都通過也必須排除。 |
| `normal_model` | 同名 | normal 日主要模型 A 的係數、截距與門檻。 |
| `normal_model_M` | 同名 | normal 日確認模型 M 的係數、截距與門檻。 |
| `abnormal_model` | 同名 | abnormal 日主要模型 A 的係數、截距與門檻。 |
| `abnormal_model_M` | 同名 | abnormal 日確認模型 M 的係數、截距與門檻。 |

`suspended_list` 的資料血緣為：

```text
[PRE] hft_strick_makerSpreadBP、PreviousClosePrice
    ↓ 模型產製程序
[MODEL] suspended_list
    ↓ 盤中策略只讀模型結果，不再自行重算
```

目前模型產製規則：

```text
加入 suspended_list
    if hft_strick_makerSpreadBP < -70
    OR PreviousClosePrice > 1000
```

也就是說，來源特徵來自盤前檔，但策略收到的直接欄位是模型 JSON 的 `suspended_list`，所以文件與流程圖一律標為 <span style="color:#6a1b9a;font-weight:700">[MODEL]</span>。

盤前可交易集合：

```text
eligibleUniverse
    = preMarketUniverse
    ∩ 使用日當天 allow_day_trade_mark == "X" 的商品
    - suspended_list
```

今日與昨日候選清單中的商品都必須先通過使用日當天的當沖資格。當日 `allow_day_trade_mark != "X"` 或缺少當日資格者，必須在行情訂閱之前直接移出集合；後續不訂閱、不解析報價、不更新特徵，也不參與 `AmountRank_canDayTrade` 排名。這是早期資料面排除，不只是送單前才檢查的 Gate。

### 5.4 全市場即時計算資料

`AmountRank_canDayTrade` 是 <span style="color:#1565c0;font-weight:700">[CALC]</span>，不是單一股票 Tick 直接帶入：

```text
Amount[stock, tick]
    = 從當日第一筆到目前為止累加
      FillLots × FillPrice × (TrialMatch == 0)

TimeSlot = 09:00、09:05、...、13:20

每個 TimeSlot：
    1. 每檔股票取 TransTime <= TimeSlot 的最後一筆 Amount
    2. 當日 marketData 的 allow_day_trade_mark == "X" 才參與排名，其餘為 null
    3. 依 Amount 由大到小做 ordinal rank
    4. 最大者 AmountRank_canDayTrade = 1
```

只有 `TrialMatch == 0` 才能增加 `Amount`；試撮成交完全不計。盤中重啟不回放、不持久化重啟前行情，所有商品的 `Amount` 與 `AmountRank_canDayTrade` 歸零，改從重啟後第一筆正常盤 Tick 重新累加；因此重啟後的 `Amount` 刻意代表「本次啟動後累計值」，不再代表完整交易日累計值。

`ordinal rank` 表示金額相同時仍會依資料順序得到不同名次，不是同名次排名。研究資料把上述 5 分鐘快照的 `TransTime` 加 `1µs` 後，再用 backward as-of join 回候選 Tick；RD 若要完全重現 parquet，join 時序也必須一致。

<span style="color:#ef6c00;font-weight:700">[PRE]</span> `day_amount_rank` 則是收盤後的前一日全日排名：

```text
_amt[each tick]    = FillLots × FillPrice
_day_amount[stock] = 前一完整交易日 Σ(_amt)
day_amount_rank    = _day_amount 由大到小做 min rank
```

`min rank` 表示成交金額相同時共用同一名次。營運盤前檔以次一交易日命名，所以盤中拿到的是前一完整交易日的結果。

策略使用：

```text
isTop100 = day_amount_rank <= 100
           OR AmountRank_canDayTrade <= 100
```

這個聯集的目的，是同時保留「昨天已經熱門」與「今天盤中突然變熱門」的股票。`AmountRank_canDayTrade` 只能使用當下以前的成交資料，不能使用收盤後的全日排名。

---

## 6. 每檔股票的逐筆計算狀態

### 6.1 有效成交與價格路徑

`src/features` 對 OHLC 與最近成交價使用兩個非常接近、但不完全相同的條件；實作時不可自行合成第三個條件：

```text
OHLC 有效價條件       = TrialMatch == 0 AND FillPrice != 0
RecentFillPrice 更新條件 = FillPrice > 0
```

`RecentFillPrice` 的計算輸入已先被 `marketOpen == true` 過濾，而上游 `FillLots` 也已把 `TrialMatch != 0` 的量設為 `0`。現行算法沒有要求 `FillLots != 0` 才更新價格路徑。

每檔股票維護下列 <span style="color:#1565c0;font-weight:700">[CALC]</span> 狀態：

| 欄位 | 更新方式 | 想表達的市場意義 |
|---|---|---|
| `Open` | 每檔股票在本次啟動後第一筆符合 `TrialMatch == 0 AND FillPrice != 0` 的 `FillPrice`，一旦取得後到下次重啟或換日前不再改變 | 正常啟動時是今日開盤成交價；盤中重啟後是重啟後第一筆正常盤有效成交價。欄名沿用現有資料的 `Open`。 |
| `RecentFillPrice` | `FillPrice > 0` 時更新，沒有成交的 Tick forward-fill；尚無成交時 fallback `RefPrice` | `ToLow`／`ToHigh` 實際使用的基準價。 |
| `RecordLow` | 符合 OHLC 有效價條件之 `FillPrice` 的逐筆累計最小值 | 目前為止的成交低點。 |
| `RecordHigh` | 符合 OHLC 有效價條件之 `FillPrice` 的逐筆累計最大值 | 目前為止的成交高點。 |

今日尚未出現有效成交前：

- `RecentFillPrice = RefPrice`。
- `Open = 0`。
- `RecordLow = price_x10000` 儲存型別的最大值，作為尚未形成低點的哨兵值。
- `RecordHigh = 0`。
- `Open == 0` 時價格路徑尚未有效，不能計算相關模型特徵，也不能送入模型；哨兵值絕不可參與模型計算。
- 第一筆有效成交後，`Open = RecentFillPrice = RecordLow = RecordHigh = FillPrice`。

### 6.2 價格位於日內高低點的哪裡

以下三個都是 <span style="color:#1565c0;font-weight:700">[CALC]</span> 模型特徵：

```text
ToLow  = abs(RecentFillPrice - RecordLow)  / RefPrice
ToHigh = abs(RecentFillPrice - RecordHigh) / RefPrice

distance = ToLow + ToHigh
Low_High = 0.5                 if distance == 0
           ToLow / distance    otherwise
```

| 特徵 | 用途 |
|---|---|
| `ToLow` | 最近成交離日內低點多遠。越小表示越貼近低點。 |
| `ToHigh` | 最近成交離日內高點多遠。越小表示越貼近高點。 |
| `Low_High` | 把最近成交壓縮成高低區間內的位置；接近 `0` 表示靠近低點，接近 `1` 表示靠近高點。 |

### 6.3 B1 相對參考價與開盤價

以下都是 <span style="color:#1565c0;font-weight:700">[CALC]</span> 模型特徵：

```text
ToRef  = round((BidPrice1 - RefPrice) / RefPrice, 6)
ToOpen = round((BidPrice1 - Open) / Open, 6)
```

`BidPrice1`、`RefPrice`、`Open` 在交易程式內均為 `price_x10000` 整數；做上述除法時才轉成 `double`。價格本身的 4 位小數尺度不代表比例特徵只能保留 4 位，模型輸入仍須 round 至小數第 6 位。

| 特徵 | 用途 |
|---|---|
| `ToRef` | B1 相對平盤價的漲跌幅。最終 Gate 只接受嚴格大於平盤且嚴格小於上漲 5%。 |
| `ToOpen` | B1 相對今日開盤價的漲跌幅。用來告訴模型開盤後的價格方向。 |

`tickFeature` parquet 中原本的同名欄位是用 `(AskPrice1 + BidPrice1) / 2` 計算；但是 `negFillResearch_model` 與 `modelUpdate.py` 讀檔後都會用上式覆寫。因此模型最終看到的是 **B1 版**，RD 不可直接把 parquet 的 mid-price 版送進模型。

`RefPrice <= 0` 或尚未取得 `Open` 時，本筆不可送入模型。

### 6.4 一檔委託量不平衡

<span style="color:#1565c0;font-weight:700">[CALC]</span> `B1_A1B1`：

```text
depth = BidLots1 + AskLots1

B1_A1B1 = BidLots1 / depth    if depth > 0
           0.5                 if depth == 0
```

用途：表示最佳一檔的買方掛單占比。接近 `1` 表示 B1 量相對強，接近 `0` 表示 A1 量相對強，`0.5` 表示中立。

### 6.5 距離收盤秒數

<span style="color:#1565c0;font-weight:700">[CALC]</span> `RemainSeconds`：

```text
RemainSeconds = 13:25:00 - TransTime
```

實際使用 `.dt.total_seconds()`，parquet 儲存為整數秒，不保留微秒小數。用途是讓模型辨識同樣的型態發生在早盤或接近尾盤時可能有不同結果。

---

## 7. 核心事件：低點賣壓與價差張開

### 7.1 `FillLots_atLow`：目前低點區間內的累積買賣流

先算每筆 <span style="color:#1565c0;font-weight:700">[CALC]</span> 有方向成交量：

```text
SignedFillLots = FillLots × InOut    if TrialMatch == 0
                 0                   otherwise
```

再以每一段相同的 `RecordLow` 累加：

```text
當 RecordLow 創新低：
    開始新的低點區間
    FillLots_atLow = 當筆 SignedFillLots

RecordLow 沒變：
    FillLots_atLow += 當筆 SignedFillLots
```

`FillLots_atLow < 0` 表示：自從目前這個日內低點形成以來，賣方主動成交量多於買方主動成交量。這是 negFill 的核心「負向成交流」定義。

### 7.2 `Spread`：A1 與 B1 相差幾個 tick

<span style="color:#1565c0;font-weight:700">[CALC]</span> `Spread` 的單位是 tick 數，不是價格金額：

```text
Spread = BidPrice1 到 AskPrice1 之間相差的合法跳動檔數
```

實際算法先把 A1、B1 各自轉成跨級距的 tick index、四捨五入，再以 `askIndex - bidIndex` 相減；任一側價格為 `0` 時 `Spread = 0`。行情倒掛時現行 feature 可能得到負數，因此策略仍須依第 5.1 節把該筆擋掉。

例如某股票在這個價位的 `TickSize = 0.05`：

```text
B1 = 34.75, A1 = 34.80  → Spread = 1
B1 = 34.75, A1 = 34.85  → Spread = 2
```

台股一般股票的跳動單位：

| 價格區間 | `TickSize` |
|---|---:|
| `< 10` | `0.01` |
| `10 ～ < 50` | `0.05` |
| `50 ～ < 100` | `0.10` |
| `100 ～ < 500` | `0.50` |
| `500 ～ < 1000` | `1.00` |
| `>= 1000` | `5.00` |

### 7.3 `SpreadPair`：記住最近一次價差張開事件

策略只在 `Spread` 比前一筆變大時更新 <span style="color:#1565c0;font-weight:700">[CALC]</span> SpreadPair；只要 SpreadPair 發生變動，`SpreadPairElapsed` 就歸零並從當筆重新計時。現行 `orderbook.py` 對 pair 價格與 elapsed 起點分別用了 `0.01`、`0.1` 的浮點容忍值：

```text
if Spread[i] > Spread[i-1] + 0.01 and previous TrialMatch == 0:
    SpreadPairBid       = BidPrice1[i]
    SpreadPairAsk       = AskPrice1[i]

if Spread[i] > Spread[i-1] + 0.1 and previous TrialMatch == 0:
    SpreadPairStartTime = TransTime[i]
```

正常 `Spread` 是整數 tick，所以兩條件的結果相同；文件仍保留實際常數，方便逐欄重現與測試。

相關既有欄位的意義：

- `SpreadPairID`：同一組 `(SpreadPairAsk, SpreadPairBid)` 固定使用同一個 ID，依第一次出現順序做 dense rank；第一個 pair 出現前為 `0`。
- `SpreadPairSeq`：同一 `SpreadPairID` 被重新切換進入的累計次數。
- `SpreadPairTotalCount`：每當 `SpreadPairID` 與前一筆不同就累加，表示時間上第幾段 pair period。
- `SpreadCountAtSameCount`：以 `(QuoteCode, SpreadPairTotalCount)` 分組後，對 `(Spread.diff() > 0)` 做 cumulative sum；數值完全依現行 `orderbook.py` 產生，不可用名稱猜語意重寫。
- `SpreadNarrowOrderTime`：pair 建立後到第一次 `Spread` 縮小的秒數。
- `SpreadNarrowSide`：第一次縮小若 Ask 下移為 `-1`、Bid 上移為 `1`、兩側同時移動為 `0`，尚未縮小為 null。

價差縮小或只有掛單量改變時，不建立新 Pair，仍沿用最近一次張開時記住的 B1、A1 與開始時間。

```text
SpreadPairElapsed = TransTime - SpreadPairStartTime
```

用途：衡量「上一個價差張開狀態已經存在多久」。`SpreadPairElapsed > 0.1` 表示該張開事件至少已存在 100 毫秒，避免對非常短暫的報價閃動立刻反應。

### 7.4 候選事件 `candidateEvent`

每檔股票比較當筆與前一筆：

```text
spreadChanged = abs(Spread[i] - Spread[i-1]) > 0
b1Changed     = abs(BidPreMove[i]) > 0.001
negativeFill  = FillLots_atLow[i] < 0

candidateEvent = spreadChanged AND b1Changed AND negativeFill
```

其中 `BidPreMove = BidPrice1[i] - BidPrice1[i-1]`。研究資料的 `build_index.py` 在寫入 negFill parquet 前就已套用這三個條件，所以讀到的每一列本來就是候選事件；線上逐 Tick 實作則要自行計算同一條件。

策略意義：目前低點區間已累積賣方流，同時 B1 與價差正在改變，代表訂單簿剛發生一個值得重新評估的事件。

常見事件範例：

```text
t0：B1=34.70、A1=34.80、Spread=2，價差張開，建立 SpreadPair
t1：經過 0.15 秒，B1=34.75、A1=34.80、Spread=1

此時：
    Spread 有變       → true
    B1 有變           → true
    Pair 已存在 0.15s → 通過 0.1s 門檻
    FillLots_atLow<0  → 若成立，成為候選事件
```

---

## 8. 最近 30 筆市場活躍度

每檔股票保存「含當筆在內，最多最近 30 筆」Tick。

### 8.1 `MD_ElaspeTime_30_re`

<span style="color:#1565c0;font-weight:700">[CALC]</span>：

```text
如果目前累積筆數 >= 30：
    elapsedSeconds = TransTime[i] - TransTime[i-29]

如果目前累積筆數 < 30：
    elapsedSeconds = TransTime[i] - 當日第一筆 TransTime

MD_ElaspeTime_30_re = ln(1 + elapsedSeconds)
```

用途：衡量最近行情更新速度。相同 30 筆若在很短時間內出現，代表市場訊息密度較高；取 `ln(1+x)` 是為了壓縮極端大值。

### 8.2 `MD_L1Rate_30`

<span style="color:#1565c0;font-weight:700">[CALC]</span>：

```text
hasFill = 1 if FillLots != 0 else 0
windowCount = min(當日目前 Tick 筆數, 30)

MD_L1Rate_30 = 最近 windowCount 筆的 hasFill 總和 / windowCount
```

用途：衡量最近行情封包中有多少比例真的包含成交，而不只是五檔掛單更新。值域為 `[0, 1]`。

模型欄名固定使用：

- `MD_ElaspeTime_30_re`：已做 `ln(1+x)`。
- `MD_L1Rate_30`：原始比例，不做額外轉換。研究資料曾出現 `_re` 別名，但兩者數值相同；正式模型 schema 以模型檔係數名稱為準。

---

## 9. 09:00:25 選擇今日模型組

這個判斷每天只做一次，目的是區分「熱門股整體開得偏弱」與一般市場狀態。

### 9.1 計算方式

`modelUpdate.py` 建立訓練日分類時的原始算法是：

```text
同一 Date 的 marketData 與 data/preMarket 依商品合併
toOpen[s] = (open_price[s] - opening_ref_price[s]) / opening_ref_price[s]

topOpen = mean(toOpen where day_amount_rank <= 100
                      AND allow_day_trade_mark_x == "X")
```

盤中 09:00:25 可用相同母體實作：使用營運盤前檔中的前日 `day_amount_rank`、使用日當天的 `allow_day_trade_mark`，以及 Tick 已取得的 `Open`／`RefPrice`：

```text
toOpen[s] = (Open[s] - RefPrice[s]) / RefPrice[s]

納入平均 = day_amount_rank <= 100
           AND 使用日當天 allow_day_trade_mark == "X"
           AND Open、RefPrice 已有效

topOpen = valid toOpen 的總和 / validCount
```

`data/preMarket/{Date}` 也是以前一交易日資料產生、以使用日命名，所以研究分類的 `day_amount_rank` 與營運盤前檔同樣都是前日排名；這一點可對齊。真正需要額外提供的是使用日當天的當沖資格，不能用營運盤前檔內的前日 `allow_day_trade_mark` 代替。

### 9.2 模型選擇

```text
if topOpen < 0.0025:
    activeModelA = abnormal_model
    activeModelM = abnormal_model_M
else:
    activeModelA = normal_model
    activeModelM = normal_model_M
```

`0.0025` 等於平均開高 `0.25%`。這裡是嚴格小於；剛好等於 `0.0025` 時使用 normal 模型組。

若 `validCount == 0`、必要輸入不足、計算發生例外，或 `topOpen` 不是有限數值，策略不得自行把缺值當成 `0`，而是直接使用預設 normal 模型組：`activeModelA = normal_model`、`activeModelM = normal_model_M`，並留下告警。盤中重啟時也套用相同 fallback。

---

## 10. 硬條件 Gate

當筆 Tick 必須依序通過以下所有條件，才可進模型：

| # | 條件 | 使用資料 | 為什麼要擋 |
|---:|---|---|---|
| 1 | 商品在 `eligibleUniverse` | <span style="color:#ef6c00;font-weight:700">[PRE]</span> + <span style="color:#6a1b9a;font-weight:700">[MODEL]</span> | 盤前檔決定觀測集合，使用日當天的 `allow_day_trade_mark` 決定當沖資格；模型檔的 `suspended_list` 再排除不可交易商品。 |
| 2 | `TrialMatch == 0` | <span style="color:#c62828;font-weight:700">[TICK]</span> | 排除試撮行情。 |
| 3 | `09:00:30 < TransTime < 12:00:00` | <span style="color:#c62828;font-weight:700">[TICK]</span> | 避開剛開盤雜訊，並限制只在上午開新倉。 |
| 4 | `day_amount_rank <= 100 OR AmountRank_canDayTrade <= 100` | <span style="color:#ef6c00;font-weight:700">[PRE]</span> + <span style="color:#1565c0;font-weight:700">[CALC]</span> | 只做成交金額活躍的股票。 |
| 5 | `candidateEvent == true` | <span style="color:#1565c0;font-weight:700">[CALC]</span> | 必須同時有價差變動、B1 變動與低點負向成交流。 |
| 6 | `RefPrice < BidPrice1 < RefPrice × 1.05` | <span style="color:#c62828;font-weight:700">[TICK]</span> | 等價於最終研究訊號的 `ToRef > 0`，再加上先前樣本初篩的 `ToRef < 0.05`。上下界都不包含。 |
| 7 | `SpreadPairElapsed > 0.1` 秒 | <span style="color:#1565c0;font-weight:700">[CALC]</span> | 排除生命週期不滿 100ms 的短暫價差事件。剛好 `0.1` 不通過。 |
| 8 | `OTC == true OR MD_L1Rate_30 > 0.25` | <span style="color:#c62828;font-weight:700">[TICK]</span> + <span style="color:#1565c0;font-weight:700">[CALC]</span> | `OTC` 直接取 Tick 同名欄位；上市股票要求最近 Tick 中有足夠成交比例，OTC 直接略過此條。研究 notebook 最終訊號使用 `0.25`，剛好 `0.25` 不通過。 |
| 9 | 九個模型特徵都有效 | <span style="color:#1565c0;font-weight:700">[CALC]</span> | 避免缺開盤價、除以零或非有限數值進入模型。 |
| 10 | 新單後的單商品與全策略全天 gross entry exposure 都不超過 `risk.toml` 上限 | <span style="color:#2e7d32;font-weight:700">[ORDER]</span> + `risk.toml` | 同時控制單商品與策略整體當日曾使用的開倉額度；停損回補不退還額度，但也不禁止後續新單。 |

Gate 的完整邏輯可寫成：

```text
passGate =
    QuoteCode in eligibleUniverse
    AND TrialMatch == 0
    AND signalStartTime < TransTime < signalEndTime
    AND isTop100
    AND candidateEvent
    AND RefPrice < BidPrice1 < RefPrice × maxEntryPriceRatio
    AND SpreadPairElapsed > minSpreadPairElapsed
    AND (OTC OR MD_L1Rate_30 > minListedMDL1Rate)
    AND allModelFeaturesValid
    AND projectedSymbolExposureValue <= maxSymbolExposureValue
    AND projectedStrategyExposureValue <= maxStrategyExposureValue
```

---

## 11. 模型輸入與兩個子模型

### 11.1 九個模型特徵

兩個子模型使用相同的特徵 schema：

| 順序 | 特徵 | 來源 | 告訴模型什麼 |
|---:|---|---|---|
| 1 | `ToLow` | <span style="color:#1565c0;font-weight:700">[CALC]</span> | 最近成交離日內低點多遠。 |
| 2 | `ToHigh` | <span style="color:#1565c0;font-weight:700">[CALC]</span> | 最近成交離日內高點多遠。 |
| 3 | `Low_High` | <span style="color:#1565c0;font-weight:700">[CALC]</span> | 最近成交位於日內高低區間的相對位置。 |
| 4 | `ToRef` | <span style="color:#1565c0;font-weight:700">[CALC]</span> | B1 相對平盤價的漲跌幅。 |
| 5 | `ToOpen` | <span style="color:#1565c0;font-weight:700">[CALC]</span> | B1 相對今日開盤價的漲跌幅。 |
| 6 | `B1_A1B1` | <span style="color:#1565c0;font-weight:700">[CALC]</span> | 最佳一檔買賣掛單量是否失衡。 |
| 7 | `RemainSeconds` | <span style="color:#1565c0;font-weight:700">[CALC]</span> | 距離收盤還有多久。 |
| 8 | `MD_ElaspeTime_30_re` | <span style="color:#1565c0;font-weight:700">[CALC]</span> | 最近行情更新速度。 |
| 9 | `MD_L1Rate_30` | <span style="color:#1565c0;font-weight:700">[CALC]</span> | 最近行情中含成交封包的比例。 |

### 11.2 與 `src/features`、parquet、`negFillResearch_model` 的逐項比對

以下是模型真正收到的最後一版值。所謂「parquet 直接」是指從 `tickFeature` 讀入；實際檔在 `run_tickFeature.py` 寫出前會把 `Float64` 降為 `Float32`、`Int64` 降為 `Int32`。

| 模型特徵 | parquet／`src/features` 算法 | 模型載入後處理 | 一致性結論 |
|---|---|---|---|
| `ToLow` | `abs(RecentFillPrice - RecordLow) / RefPrice`；`price_dynamics.py` | 不覆寫 | 一致，直接使用 parquet。 |
| `ToHigh` | `abs(RecentFillPrice - RecordHigh) / RefPrice`；`price_dynamics.py` | 不覆寫 | 一致，直接使用 parquet。 |
| `Low_High` | `ToLow / (ToLow + ToHigh)`，`NaN` 填 `0.5`；`price_dynamics.py` | 不覆寫 | 一致，直接使用 parquet。 |
| `ToRef` | parquet 是 `((AskPrice1 + BidPrice1)/2 - RefPrice) / RefPrice`；`price_dynamics.py` | 覆寫為 `round((BidPrice1 - RefPrice) / RefPrice, 6)` | **不可直接使用 parquet 原值**；覆寫後才與研究模型一致。 |
| `ToOpen` | parquet 是 `((AskPrice1 + BidPrice1)/2 - Open) / Open`；`price_dynamics.py` | 覆寫為 `round((BidPrice1 - Open) / Open, 6)` | **不可直接使用 parquet 原值**；覆寫後才與研究模型一致。 |
| `B1_A1B1` | `BidLots1 / (BidLots1 + AskLots1)`，`NaN` 填 `0.5`；`orderbook.py` | 不覆寫 | 一致，直接使用 parquet。 |
| `RemainSeconds` | `(13:25:00 - TransTime.time()).total_seconds()`；`time_features.py` | 不覆寫 | 一致，值為整數秒。 |
| `MD_ElaspeTime_30_re` | parquet 先有 `MD_ElaspeTime_30 = (TransTime[i] - TransTime[i-29]) / 1s`，不足 30 筆改與首筆比較；`microstructure.py` | `ln(1 + MD_ElaspeTime_30)` | 一致，但模型欄名必須保留既有拼字 `Elaspe` 與 `_re`。 |
| `MD_L1Rate_30` | 最近最多 30 筆中 `(FillLots != 0)` 的 rolling mean，`min_periods=1`；`microstructure.py` | `MD_L1Rate_30_re` 雖被建成同值別名，正式 feature list 仍使用 `MD_L1Rate_30` | 一致，模型輸入不做 log 或其他轉換。 |

候選事件與 Gate 所依賴的非模型 feature 也已沿資料血緣核對：

| 欄位／狀態 | 實際算法來源 | 核對結果 |
|---|---|---|
| `Open`, `RecordLow`, `RecordHigh` | `src/features/definitions/basic_info.py` | 只以 `TrialMatch == 0 AND FillPrice != 0` 更新，不要求 `FillLots != 0`。 |
| `FillLots_atLow`, `BidPreMove`, `Spread` | `price_dynamics.py`、`orderbook.py` | 文件第 7 節已使用相同公式；`FillLots_atLow` 包含當筆並依 `QuoteCode, RecordLow` 累加。 |
| `SpreadPairElapsed` | `orderbook.py` | Spread 張開時記起點，使用微秒差除以 `1,000,000`；首個 pair 前為 null。 |
| `candidateEvent` | `build_index.py` | `abs(Spread.diff()) > 0.001 AND abs(BidPreMove) > 0.001 AND FillLots_atLow < 0`。研究 parquet 已預先過濾。 |
| `AmountRank_canDayTrade` | `src/features/crossSection.py` | 每 5 分鐘對當日累計 `Amount` 做可當沖商品 ordinal rank。 |
| `day_amount_rank` | `src/features/preMarketSummary.py` | 前一完整日 `Σ(FillLots × FillPrice)` 做全市場 descending min rank，再寫入次日營運盤前檔。 |

抽樣驗證使用 `20260812` 的 `1101`、`2330`、`3008`、`3081`、`8299`，共 `124,317` 筆 Tick，從 `src/features` 公式重算後與 parquet 比對：null pattern 全部相同；數值差只剩 `Float32` 寫檔誤差。`ToRef`、`ToOpen` 的 parquet mid-price 版則在這批資料每一筆都與模型 B1 版不同，證明覆寫不可省略。

### 11.3 模型檔四個區塊

每日模型檔包含：

| 區塊 | 何時使用 | 角色 |
|---|---|---|
| `normal_model` | `topOpen >= 0.0025` | normal 日的主要預測模型 A。 |
| `normal_model_M` | `topOpen >= 0.0025` | normal 日的第二道確認模型 M。 |
| `abnormal_model` | `topOpen < 0.0025` | abnormal 日的主要預測模型 A。 |
| `abnormal_model_M` | `topOpen < 0.0025` | abnormal 日的第二道確認模型 M。 |

每個區塊都有三種 <span style="color:#6a1b9a;font-weight:700">[MODEL]</span> 參數：

| 參數 | 用途 |
|---|---|
| `coefficients` | 每個特徵的權重。正值會提高分數，負值會降低分數；絕對值越大，該特徵對分數影響越大。 |
| `intercept` | 模型基準分數；即所有特徵貢獻之外的固定起點。 |
| `threshold` | 最低進場分數。模型分數必須嚴格大於它。 |

模型計分：

```text
score = intercept + Σ(當筆同名 feature × coefficients 中的同名權重)
```

若某個特徵因每日 feature screening 沒出現在 `coefficients`，該特徵係數視為 `0`；不能沿用前一天的係數。

當筆訊號必須兩票都通過：

```text
passModelA = scoreA > activeModelA.threshold
passModelM = scoreM > activeModelM.threshold

passModels = passModelA AND passModelM
```

目前模型檔門檻：

| 模型 | `threshold` | 用途 |
|---|---:|---|
| `normal_model`／`abnormal_model` | `20` | 主要模型預期放空優勢必須夠高。 |
| `normal_model_M`／`abnormal_model_M` | `-40` | 第二模型負責排除特別差的市場相對結果。 |

門檻仍以當日模型檔內容為準，不應另寫一份獨立常數覆蓋模型檔。

### 11.4 模型檔其他欄位

| 欄位 | 是否參與逐筆計分 | 用途 |
|---|---|---|
| `metadata.trained_until` | 否 | 確認參數使用到哪一天的訓練資料，避免載入過期或未來參數。 |
| `metadata.alpha` | 否 | Ridge 訓練時的正則化強度，只是模型追蹤資訊。 |
| `metadata.otc_turnover_threshold` | 否 | 建立 OTC 盤前目標清單時使用的歷史排名門檻；不直接放進逐筆分數。 |
| `suspended_list` | 否 | <span style="color:#6a1b9a;font-weight:700">[MODEL]</span> 當日不可交易清單，於 Gate 最前面排除。它雖由盤前特徵產生，但盤中直接來源是模型檔。 |

本策略不使用 `target_list`；每日模型 JSON 不需要提供此欄位，交易程式也不讀取或推導它。

### 11.5 目前仍存在的 research／執行流程落差

九個模型 feature 的最終算法已可完全對齊，但整條策略流程還不能宣稱完全一致。RD 與模型維護者需明確處理以下項目：

| 項目 | `negFillResearch_model`／模型產製 | 目前其他流程 | 本文件採用方式 |
|---|---|---|---|
| 上市股 `MD_L1Rate_30` Gate | notebook 最終訊號為 `> 0.25` | `update_trade_record.py` 沒有這條 | 正式策略已定為 `> 0.25`；`update_trade_record.py` 尚待同步。 |
| 最終風控 cell | notebook 曾以 `Position < 600` 表示研究訊號名目額度 | 舊文件曾以單商品 200 萬表示實際曝險 | 正式交易不使用研究 `Position` 作為金額上限；改用 `risk.toml` 的單商品與全策略兩層上限，兩者都以「當日累計已成交Entry gross＋未成交Entry reservation＋本次新單」計算，Cover不退gross。`accLots`深度條件與`BidPrice1 <= 1000`仍保留。 |
| 當日當沖資格 | 歷史 `topOpen` 與 `AmountRank_canDayTrade` 使用當日 `marketData.allow_day_trade_mark` | 營運盤前檔同名欄位實際來自前一交易日 | RD 必須接使用日當天的同名欄位；缺檔時 fail-closed，不可拿前日同名欄位冒充。 |
| `Open` 可得時間 | `basic_info.py` 以整日 `QuoteCode` window 取第一筆有效 `FillPrice`，離線 parquet 可能在該成交真正發生前就已有 `Open` | 線上只能在第一筆有效成交到達後知道 `Open` | 線上不得預知；`Open` 尚未取得就不計分。模型端應確認 09:00:30 後是否仍存在受此差異影響的樣本。 |
| 訓練樣本初篩 | `-0.015 < ToRef < 0.05` 且任一排名 `<= 150` | 最終訊號再加 `ToRef > 0` 且任一排名 `<= 100` | 有效執行範圍是 `0 < ToRef < 0.05`、排名 `<= 100`；不可把 150 當成盤中門檻。 |
| 缺值處理 | `rolling_model.py` 先以「前一交易日該 feature 的中位數」，仍缺再以全體中位數補值 | 每日模型 JSON 沒有輸出這些中位數 | 盤中 `allModelFeaturesValid == false` 時拒絕該筆；若要重現訓練補值，模型檔必須新增並版本化 imputation 參數。 |

其中 rate Gate、最終風控、當日當沖資格與 `Open` 可得時間都可能改變最終訊號集合，應列為上線前的一致性修正；訓練池用 150／執行用 100 是既有研究設計；缺值則暫採 fail-closed，避免 RD 自行猜補值。

---

## 12. 委託、部位、停損與平倉

### 12.1 建立空單

`negFillResearch_model` 在 `passGate == true` 且 `passModels == true` 後，還會依排序後的訊號列計算：

```text
accLots[i] = 當日同 QuoteCode 到第 i 筆為止的累計訊號筆數
```

只有再通過下列研究風控才進入委託：

```text
accLots < avg_askLots1 + avg_bidLots1
BidPrice1 <= 1000
hft_strick_makerSpreadBP is null OR hft_strick_makerSpreadBP > -70
OTC == true OR MD_L1Rate_30 > 0.25
```

`accLots` 是先在全部 `signal_df_model` 上累加、再套上述風控；因此較早出現但最後被風控排除的訊號列，現行 notebook 仍會占用後續的訊號筆數額度。若 RD 要逐筆完全重現，不可只在實際送單／成交後才增加 `accLots`。

研究 notebook 原有的 `Position` 名目訊號額度不作為正式線上交易金額上限。正式交易改用 `risk.toml` 載入的單商品與全策略兩層實際曝險上限。B1 剛好 `1000` 可通過；`hft_strick_makerSpreadBP` 剛好 `-70` 不通過。最後一條與第 10 節 Gate 相同，研究 notebook 在風控 cell 再檢查一次。

通過後建立空單：

```text
entryPrice = AskPrice1 的前一個合法報價
entryLots  = 1 張（一般股票為 1,000 股）
```

也就是一般情況下的 `AskPrice1 - 1 Tick`。價格剛好跨越跳動單位級距時，必須取交易所價格表中的前一個合法價位，不能直接用 A1 所在級距的 TickSize 做減法。

委託規則：

1. 使用 ROD 賣單。
2. 該股票沒有未成交開倉賣單：新增一張委託。
3. 該股票已有未成交開倉賣單：不重複新增，將原單改價成最新 `entryPrice`。
4. 送出新增或增加曝險的操作前，同時計算單商品與全策略的操作後預計曝險；任一超過 `risk.toml` 對應上限就不送單。

<span style="color:#2e7d32;font-weight:700">[ORDER]</span> 兩層曝險：

```text
projectedSymbolExposureValue
    = 該商品當日累計已成交開倉市值
    + 該商品尚未成交開倉賣單的保留市值
    + 本次欲新增的開倉委託市值

projectedStrategyExposureValue
    = 全策略所有商品當日累計已成交開倉市值
    + 全策略所有商品尚未成交開倉賣單的保留市值
    + 本次欲新增的開倉委託市值

允許送單 = projectedSymbolExposureValue <= risk.max_symbol_exposure_value
           AND projectedStrategyExposureValue <= risk.max_strategy_exposure_value
```

兩個上限都只定義用途，不在程式中寫死數值；啟動時從 `risk.toml` 載入並驗證為合法正整數。金額以新台幣元為單位並使用足夠寬的整數型別。收到成交回報才增加實際空單部位；停損或其他回補成交只減少庫存，不扣回當日已成交開倉額度。改價時必須先移除原 working order 的保留市值，再用新價格與剩餘量重算，不能把改價前後的同一張委託重複計入。

### 12.2 盤中停損

每筆正常盤中有效報價都檢查：

```text
初始 stopLossArmed = true

if TrialMatch == 0 AND stopLossArmed
   AND AskPrice1 >= RefPrice × 1.08:
    產生停損訊號
    以漲停價送買單，回補當下全部剩餘空單
    若當下沒有空單則只記錄 NoPositionToCover
    stopLossArmed = false

if TrialMatch == 0 AND NOT stopLossArmed
   AND AskPrice1 < RefPrice × 1.06:
    stopLossArmed = true
```

停損判斷使用最佳賣價 `AskPrice1`，不是最新成交價 `FillPrice`，也不是策略自己的成交成本。門檻為 `AskPrice1 >= RefPrice × 1.08`，剛好等於門檻也會觸發；重新武裝必須嚴格低於 `RefPrice × 1.06`，剛好等於 6% 不成立。6%／8% hysteresis 避免報價在8%附近震盪時重複觸發。空倉穿越8%仍消耗當次 armed 狀態。回補單使用漲停價作為買進限價，以積極成交方式回補；此處所稱 taker 即指這種「帶漲停價買回」的做法，不代表送出沒有價格上限的市價單。

停損不取消、封鎖或拒絕停損前已排隊、同一觸發 Tick 或停損後產生的開倉單。觸發停損的 Tick 仍完成全部策略狀態更新與新單判斷；CPU 6 先處理回補，再依一般 Risk 規則處理新單。

### 12.3 尾盤平倉

12:45：

```text
僅處理今日觀察名單商品
取得完整 Working Order snapshot，取消所有未成交開倉賣單
固定等待 1 秒，期間持續處理 Fill
空單量 = max(-net_position_qty, 0)
依Core 6當下空單量扣除有效Cover leaves
對未覆蓋空單，以最新BidPrice1掛ROD MakerCover
```

13:14：

```text
僅處理今日觀察名單商品
取得完整 Working Order snapshot，取消開倉賣單並固定等待 1 秒
StopCover 維持原單；MakerCover 以同一張委託 Modify 到漲停價
只有`-net_position_qty`超過有效Cover leaves的差額才新建漲停價TakerCover
```

13:20：CPU 1發出`PostClosePositionCheck`；CPU 6檢查全部商品。signed Position非零或仍有
Pending／Working／Unknown Entry或Cover即發警告，要求交易員用券商APP再次確認與人工回補／處理；
本檢查不呼叫DT3 Position Query，也不自動建立第二張Cover。

非今日觀察名單的庫存不加入行情監控、停損或12:45／13:14自動回補，交由交易員人工回補；但仍納入13:20檢查。

---

## 13. 策略參數總表

所有策略常數集中在此。RD 應從設定或當日模型檔載入，不要散落在流程各處。

| 參數名稱 | 現行值 | 直接來源 | 用途 |
|---|---:|---|---|
| `preMarketLoadTime` | `08:15:00` | <span style="color:#455a64;font-weight:700">[PARAM]</span> 策略設定 | 載入盤前商品資料與當日模型檔。 |
| `modelSelectionTime` | `09:00:25` | <span style="color:#455a64;font-weight:700">[PARAM]</span> 策略設定 | 正常啟動時固定今日使用 normal 或 abnormal 模型組；計算失敗使用完整 normal 模型組。 |
| `restartModelSelectionDelay` | `30 秒` | <span style="color:#455a64;font-weight:700">[PARAM]</span> 已確認重啟規則 | 09:00:00 後盤中重啟，從 `restartTime` 起算 30 秒後重選模型。 |
| `restartNewEntryCooldown` | `60 秒` | <span style="color:#455a64;font-weight:700">[PARAM]</span> 已確認重啟規則 | 重啟後至少 60 秒禁止新增或修改開倉單；DT3 對帳未完成時必須繼續禁止。 |
| `signalStartTime` | `09:00:30`，不含 | <span style="color:#455a64;font-weight:700">[PARAM]</span> 策略設定 | 避開開盤最初 30 秒的雜訊。 |
| `signalEndTime` | `12:00:00`，不含 | <span style="color:#455a64;font-weight:700">[PARAM]</span> 策略設定 | 中午後停止建立新空單。 |
| `rankRefreshInterval` | `5 分鐘` | <span style="color:#455a64;font-weight:700">[PARAM]</span> 策略設定 | 更新今日累計成交金額排名。 |
| `amountRankLimit` | `100` | <span style="color:#455a64;font-weight:700">[PARAM]</span> 策略設定 | 只交易 `day_amount_rank` 或 `AmountRank_canDayTrade` 前 100 名。 |
| `topOpenThreshold` | `0.0025` | <span style="color:#455a64;font-weight:700">[PARAM]</span> 策略設定 | 熱門股平均開盤漲幅低於 0.25% 時改用 abnormal 模型。 |
| `hft_strick_makerSpreadBP` 的 `suspended_list` 門檻 | `-70`，嚴格小於 | <span style="color:#455a64;font-weight:700">[PARAM]</span> 模型產製設定 | 模型產製時，盤前值 `< -70` 的商品加入 `[MODEL] suspended_list`。 |
| `hft_strick_makerSpreadBP` 研究風控門檻 | null 或嚴格大於 `-70` | <span style="color:#455a64;font-weight:700">[PARAM]</span> notebook 最終風控 | 研究風控會再直接檢查盤前同名欄位，所以剛好 `-70` 雖未進 `suspended_list`，仍不得交易。 |
| `PreviousClosePrice` 排除門檻 | `1000`，嚴格大於 | <span style="color:#455a64;font-weight:700">[PARAM]</span> 模型產製設定 | 模型產製時，盤前 `PreviousClosePrice > 1000` 的商品加入 `[MODEL] suspended_list`；剛好 1000 不排除。 |
| `BidPrice1` 研究風控上限 | `1000`，包含 | <span style="color:#455a64;font-weight:700">[PARAM]</span> notebook 最終風控 | 當筆 B1 必須 `<= 1000`；這與用前收價建立 `suspended_list` 是兩個不同條件。 |
| `maxEntryPriceRatio` | `1.05`，嚴格小於 | <span style="color:#455a64;font-weight:700">[PARAM]</span> 研究樣本與最終訊號條件 | B1 必須低於參考價的 1.05 倍；剛好上漲 5% 不建立新空單。 |
| `minSpreadPairElapsed` | `0.1 秒`，嚴格大於 | <span style="color:#455a64;font-weight:700">[PARAM]</span> 策略設定 | 排除持續時間太短的價差張開事件。 |
| `mdWindow` | `30 筆` | <span style="color:#455a64;font-weight:700">[PARAM]</span> 策略設定 | 計算行情速度與成交封包比例。 |
| `minListedMDL1Rate` | `0.25`，嚴格大於 | <span style="color:#455a64;font-weight:700">[PARAM]</span> 研究最終訊號條件 | 上市股票最近行情至少要有超過 25% 的 Tick 含成交；OTC 免除此條。 |
| `remainTimeAnchor` | `13:25:00` | <span style="color:#455a64;font-weight:700">[PARAM]</span> 策略設定 | 計算模型特徵 `RemainSeconds`。 |
| `entryPriceOffset` | `AskPrice1` 的前一個合法報價 | <span style="color:#455a64;font-weight:700">[PARAM]</span> 策略設定 | 用較被動的價格建立空單，並正確處理跳動單位級距邊界。 |
| `entryLots` | `1 張` | <span style="color:#455a64;font-weight:700">[PARAM]</span> 策略設定 | 每次訊號的目標委託量。 |
| `risk.max_symbol_exposure_value` | 由 `risk.toml` 載入 | <span style="color:#455a64;font-weight:700">[PARAM]</span> 固定設定檔 | 單一商品「當日累計已成交Entry gross＋未成交Entry reservation＋本次新單」允許的最大金額；Cover不退gross，剛好等於上限可通過。 |
| `risk.max_strategy_exposure_value` | 由 `risk.toml` 載入 | <span style="color:#455a64;font-weight:700">[PARAM]</span> 固定設定檔 | negFill全策略所有商品合計的相同daily gross口徑最大金額；Cover及Position歸零不退gross，剛好等於上限可通過。 |
| 累計訊號筆數上限 | `accLots < avg_askLots1 + avg_bidLots1` | <span style="color:#ef6c00;font-weight:700">[PRE]</span> + <span style="color:#1565c0;font-weight:700">[CALC]</span> | 避免同商品累計訊號張數超過前日一檔平均深度總和。 |
| `stopLossRefRatio` | `1.08`，大於等於 | <span style="color:#455a64;font-weight:700">[PARAM]</span> 策略設定 | `AskPrice1 >= RefPrice × 1.08` 時強制停損，並以漲停價買回全部剩餘空單。 |
| `makerExitTime` | `12:45:00` | <span style="color:#455a64;font-weight:700">[PARAM]</span> 策略設定 | 開始以 B1 被動回補。 |
| `takerExitTime` | `13:14:00` | <span style="color:#455a64;font-weight:700">[PARAM]</span> 策略設定 | 強制積極回補所有剩餘空單。 |
| `postClosePositionCheckTime` | `13:20:00` | <span style="color:#455a64;font-weight:700">[PARAM]</span> 已確認關帳規則 | 對全部商品檢查非零signed Position與未決Entry／Cover並只告警。 |
| `normal_model.threshold`, `abnormal_model.threshold` | 目前 `20` | <span style="color:#6a1b9a;font-weight:700">[MODEL]</span> 當日模型檔 | 第一個子模型的最低分數。 |
| `normal_model_M.threshold`, `abnormal_model_M.threshold` | 目前 `-40` | <span style="color:#6a1b9a;font-weight:700">[MODEL]</span> 當日模型檔 | 第二個子模型的最低分數。 |

---

## 14. RD 可直接照做的事件流程

### 14.1 每日初始化

```text
讀取 [PRE] src/strategy/preMarket/{TradeDate}_preMarketData.parquet
    直接取得 QuoteCode、allow_day_trade_mark、PreviousClosePrice、day_amount_rank
    其中 allow_day_trade_mark 屬來源日，只供追溯／預篩
讀取使用日當天的 marketData.allow_day_trade_mark；若由 T30 提供，先確認語意與值域相同
缺少當日資格時停止開新倉並告警

讀取 [MODEL] src/strategy/negFill/modelParam/{TradeDate}_modelParams.json
    直接取得 suspended_list、四個模型區塊

建立 eligibleUniverse
用 day_amount_rank 建立前日 Top 100
套用模型檔的 suspended_list
以使用日當沖資格再次過濾；不合格商品不得建立行情訂閱

交易員已用券商APP確認未結委託與實際庫存，並備份／修正current_position.csv
    CSV只含symbol,net_position_qty，不含trading_day或account，數量單位為股
程式載入signed net_position_qty：正值為多單、0為平倉、負值為空單

對每檔股票初始化：
    尚未開盤
    尚無成交價路徑
    尚無 SpreadPairBid / SpreadPairAsk / SpreadPairStartTime
    30-MD window 為空
    Position = Current Position File對應的signed net_position_qty；檔案沒有該商品才是0
    未成交委託 = 無
    stopLossArmed = true
```

### 14.2 程序重啟（08:59:00 以後，包含 08:59:00）

```text
restartTime = 本次程式啟動時間
newEntryEnabled = false
brokerRecoveryComplete = false

任何重啟前（包含當機後再次啟動）由交易員以券商APP取消全部Entry與Cover
等待取消終態與最後成交反映，確認Working Order為零
交易員備份並依券商實際庫存確認／修正current_position.csv
程式載入該檔作為新程序唯一Position baseline
重啟後向DT3以fromSNO=1完成Report recovery與RecoverWorkingOrder
若仍發現未結委託：重啟前提不成立，保持Recovery並交由交易員處理

所有行情／策略計算狀態歸零：
    Open、RecentFillPrice、RecordLow、RecordHigh
    SignedFillLots、FillLots_atLow、SpreadPair、30-MD window
    accLots、Amount、AmountRank_canDayTrade、topOpen、active model pair
不回放行情，不恢復重啟前 Amount 或特徵

若restartTime < 09:00:00：仍在09:00:25執行正常ModelSelectionBoundary
若restartTime >= 09:00:00：重啟後第一筆 TrialMatch == 0 且 FillPrice > 0 的成交設定新 Open
                         restartTime + 30 秒重算topOpen並選擇模型；失敗使用normal模型組
restartTime + 60 秒以前：禁止新增或修改開倉單

newEntryEnabled =
    現在時間 >= restartTime + 60 秒
    AND brokerRecoveryComplete == true
    AND 現在仍在正常開倉時段
```

行情與策略特徵可以歸零。新程序的Position只採人工確認的Current Position File；啟動Report recovery的歷史Fill只重建Request／Order、mapping與本次runtime dedup，不得再次修改Position或觸發交易動作。冷卻或recovery期間仍接收回報，但須等完整結束後才依實際時間補發phase。程式未重啟、僅DT3斷線重連時，Core 6記憶體Position／gross／dedup保留，尚未處理的新Fill仍正常入帳。

### 14.3 收到 Tick

```text
若換日：先完成前一日收尾，再重置所有股票狀態
若重複或倒序：丟棄並告警

更新有效成交與 Open / RecentFillPrice / RecordHigh / RecordLow
更新 SignedFillLots 與 FillLots_atLow
計算 Spread、BidPreMove、spreadChanged、b1Changed
若 Spread 張開：更新 SpreadPairBid / SpreadPairAsk / SpreadPairStartTime
更新 SpreadPairElapsed
更新 30-MD window
計算九個模型特徵

依 stopLossArmed 與 6%／8% hysteresis 檢查停損；觸發後仍繼續本 Tick 的策略流程
若未到 09:00:30 或已到 12:00：結束
若盤中重啟後尚未滿 60 秒或 DT3 對帳未完成：結束
若 Gate 失敗：結束

計算 activeModelA score
計算 activeModelM score
若任一模型未通過：結束

計算 accLots 與其他研究風控；任一失敗就結束
計算 projectedSymbolExposureValue 與 projectedStrategyExposureValue
若任一值超過 risk.toml 對應上限：結束
否則新增或改價 ROD 空單
```

### 14.4 收到委託回報

```text
新單確認：記錄 working order ID、價格與剩餘量
改價確認：更新 working order 價格
部分成交：增加空單部位，減少 working quantity
全部成交：增加空單部位，清除 working order
取消確認：清除 working order 的剩餘保留曝險
拒單：清除或還原委託狀態並告警
```

### 14.5 定時事件

```text
09:00:25：算 topOpen，固定 active model pair；失敗使用 normal_model + normal_model_M
盤中重啟後 30 秒：重新算 topOpen；失敗同樣使用 normal 模型組
盤中重啟後 60 秒：若 DT3 對帳也完成且仍在開倉時段，才恢復開新倉
每 5 分鐘：更新 AmountRank_canDayTrade
12:00:00：停止開倉並取消開倉賣單
12:45:00：掛 maker 回補單
13:14:00：StopCover維持；MakerCover同單改漲停，差額才新增taker回補
13:20:00：檢查全部非零signed Position及未決Entry／Cover並告警，不自動追加回補
```

---

## 15. 最低限度驗收案例

1. 顏色／來源檢查：每個 Gate 與模型特徵都能追到 `[TICK]`、`[PRE]`、`[CALC]` 或 `[MODEL]` 的直接來源；`suspended_list` 必須標為 `[MODEL]`。
2. 第一筆有效成交會同時設定 `Open`、`RecentFillPrice`、`RecordLow`、`RecordHigh`。
3. 無成交 Tick 不會改變成交價路徑；新高、新低只由有效成交更新。
4. `FillLots_atLow` 在創新低時開始新區間，並包含創低當筆的有方向成交量。
5. `InOut=-1` 的成交使 `FillLots_atLow` 下降；`InOut=+1` 使它上升。
6. B1/A1 相差兩個合法跳動單位時，`Spread == 2`，不是價格差的數字。
7. Spread 縮小、價格 pair 改變或只有掛單量改變，都不會建立新的 SpreadPair；只有 Spread 張開才重設開始時間。
8. 前 30 筆內使用目前已有筆數計算 `MD_L1Rate_30`；第 30 筆起只保留最近 30 筆。
9. Gate 邊界：B1 等於 `RefPrice` 或 `RefPrice×1.05` 都不通過；elapsed 等於 `0.1`、rate 等於 `0.25` 也不通過。
10. OTC 即使 `MD_L1Rate_30 <= 0.25`，仍可通過 rate 條件；上市股票不可。
11. `topOpen == 0.0025` 使用 normal 模型；`topOpen < 0.0025` 使用 abnormal 模型。
12. 模型 A 通過但模型 M 未通過時，不可送單；反之亦然。
13. 已有未成交開倉賣單時，新訊號只改價，不增加第二張 working order。
14. 單商品或全策略的操作後預計曝險超過 `risk.toml` 上限時不可送單；剛好等於各自上限可通過。改價不得重複計入同一張 working order。
15. `AskPrice1 == RefPrice × 1.08` 時會觸發停損；之後剛好6%不重新 armed，嚴格低於6%才重新 armed，下一次到8%可再次觸發。停損單必須以漲停價送出，回補不退全天 gross exposure，且不得封鎖停損前後的新單。
16. 換日、重複 Tick、倒序 Tick 與行情重播不會污染下一交易日或其他股票的狀態。
17. `ToRef`、`ToOpen` 讀 parquet 後會以 B1 公式覆寫並 round 6 位；測試不得直接期待 mid-price 版。
18. `RemainSeconds` 是整數秒；`SpreadPairElapsed`、`MD_ElaspeTime_30` 才保留微秒換算後的小數秒。
19. 次日營運盤前檔的 `allow_day_trade_mark` 應能追到來源日；Gate 缺少使用日同名欄位時必須 fail-closed。
20. 研究風控邊界：`hft_strick_makerSpreadBP == -70` 不通過，`BidPrice1 == 1000` 可通過。
21. `accLots == avg_askLots1 + avg_bidLots1` 不通過，必須嚴格小於前日一檔平均深度總和。
22. 正常啟動只在 `09:00:25` 選擇模型；`validCount == 0`、輸入不足、例外或非有限結果都使用 `normal_model + normal_model_M`。
23. 10:17:00 重啟時，10:17:30 重選模型，10:18:00 前不得開新倉；若 10:18:00 時 DT3 對帳尚未完成，仍不得開倉。
24. 重啟後第一筆正常盤有效成交會建立新的 `Open`；重啟前的 `Amount`、排名、30-MD、SpreadPair、`accLots` 與其他策略特徵不得殘留或回放。
25. 重啟前已有空單時，新程序的真實部位必須由交易員確認的Current Position File恢復；歷史DT3 recovery Fill不得再次修改該baseline。若商品仍在今日觀察名單，冷卻結束與recovery完成後可停損及尾盤回補；非觀察名單只由13:20檢查告警並交由人工回補。
26. `TrialMatch != 0` 永遠不增加 `Amount`；正常盤才依 `FillLots × FillPrice` 累加。重啟後下一個固定 5 分鐘時間點只使用重啟後累計值排名。
27. 使用日不可當沖或缺少使用日資格的商品，在行情訂閱前即被移除，且不會進入解析、特徵計算或排名。

---

## 16. 欄位名稱鎖定規則

現有資料已經有欄名時，策略、文件、log 與測試 fixture 必須直接使用原欄名，不得建立別名：

| 必須使用的原欄名 | 直接來源 | 鎖定規則 |
|---|---|---|
| `Open` | <span style="color:#1565c0;font-weight:700">[CALC]</span> `tickData` | 正常啟動時是今日第一筆正常盤有效成交價；盤中重啟後改用本次啟動後第一筆正常盤有效成交價。所有公式直接使用 `Open`。 |
| `RefPrice` | <span style="color:#c62828;font-weight:700">[TICK]</span> `tickData` | 逐筆公式直接使用 `RefPrice`；營運盤前檔的 `PreviousClosePrice` 是另一個欄位，不能互相改名。 |
| `day_amount_rank` | <span style="color:#ef6c00;font-weight:700">[PRE]</span> 營運盤前檔 | 前一完整交易日成交金額排名。 |
| `AmountRank_canDayTrade` | <span style="color:#1565c0;font-weight:700">[CALC]</span> `tickBar` | 每 5 分鐘用正常盤成交金額重算可當沖商品排名；盤中重啟後從零累計，不回放重啟前資料。 |
| `OTC` | <span style="color:#c62828;font-weight:700">[TICK]</span> `tickData` | 是否上櫃；營運盤前檔沒有 `market` 或 `OTC`。 |
| `allow_day_trade_mark` | <span style="color:#ef6c00;font-weight:700">[PRE]</span> 營運盤前檔／使用日 `marketData` | 兩個來源保留同名，以來源日期區分：營運盤前檔內是前日值，Gate 使用使用日 `marketData` 的值；外層歷史檔 `nextday_allow_day_trade_mark` 仍是另一欄。 |
| `suspended_list` | <span style="color:#6a1b9a;font-weight:700">[MODEL]</span> 模型檔 | 盤前 `hft_strick_makerSpreadBP` 與 `PreviousClosePrice` 只是上游產製資料，不可把清單改標為 `[PRE]`。 |
| `ToRef`, `ToOpen` | <span style="color:#1565c0;font-weight:700">[CALC]</span> 模型載入後覆寫 | 欄名不變；不可為了區分 parquet mid-price 版而另建正式對接名稱，應在送模型前覆寫同名值。 |
| `toOpen`, `topOpen` | <span style="color:#1565c0;font-weight:700">[CALC]</span> `modelUpdate.py` 日分類 | 大小寫須保留；`toOpen` 是日分類暫存值，不是逐筆模型 feature `ToOpen`。 |
| `MD_L1Rate_30` | <span style="color:#1565c0;font-weight:700">[CALC]</span> `tickFeature`／模型 schema | 正式模型輸入是原始比例 `MD_L1Rate_30`；研究中的暫時欄位 `MD_L1Rate_30_re` 不可作為對接名稱。 |
| `MD_ElaspeTime_30_re` | <span style="color:#1565c0;font-weight:700">[CALC]</span> `ln(1 + MD_ElaspeTime_30)` | 拼字 `Elaspe` 雖不標準，既有模型 schema 如此命名，所以不得修正拼字。 |
| `avg_bidLots1`, `avg_askLots1`, `hft_strick_makerSpreadBP` | <span style="color:#ef6c00;font-weight:700">[PRE]</span> 營運盤前檔 | 最終研究風控直接使用既有欄名，不可另改成策略別名。 |
| `accLots` | `negFillResearch_model` 最終風控 cell | 保留研究程式名稱；它是訊號筆數限制，不是金額曝險。 |
| `projectedSymbolExposureValue`, `projectedStrategyExposureValue` | <span style="color:#2e7d32;font-weight:700">[ORDER]</span> 交易狀態機 | 分別代表本次操作後的單商品與全策略實際預計曝險；上限只從 `risk.toml` 載入。 |

模型檔中的 feature 名稱是最終對接鍵；若文件名稱與模型檔不一致，應視為模型 schema 錯誤並停止交易，不可用欄位位置猜測。
