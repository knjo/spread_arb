# 2026 資料契約

## 來源與涵蓋

現貨 pipeline 資料根目錄不再寫死為 project 內的 `HFT/data/`。唯一 canonical resolver 是 top-level
`src/pipeline_storage.py`，設定來源是 `config/pipeline.yaml`；目前設定為 `/media/kevin/SSD2/Data`，且
`required_mount=/media/kevin/SSD2`。研究程式只解析路徑，不得自行建立缺失的 mount 或把資料寫回系統碟。

| 資料 | Canonical path contract |
|---|---|
| 現貨 tick | `${data_storage.tick_dir}/{YYYYMMDD}_StockTick.parquet`，目前 `/media/kevin/SSD2/Data/tickData/` |
| 現貨 feature | `${data_storage.tick_feature_dir}/{YYYYMMDD}_tickFeature.parquet` |
| 現貨 makerFill | `${data_storage.maker_queue_dir}/{YYYYMMDD}_makerFill.parquet`，目前 `/media/kevin/SSD2/Data/makerFill/` |
| NAS 現貨 raw | `/mnt/NAS/Parquet/Ticks/YYYY/MM/DD/stock_round.parquet`；只供明示需要它的研究 |
| NAS 個股期 raw | `/mnt/NAS/Parquet/Ticks/YYYY/MM/DD/stock_futures.parquet`；S1 的唯一個股期 raw source |

Spot tick／makerFill 只有在呼叫方明示 project 內舊根目錄、舊檔不存在，且 basename 完全相同時，才可 fallback 到
`pipeline.yaml` 的 canonical SSD2 peer；任意 custom root、不同 basename、`..` 或 symlink component 一律拒絕。
Canonical mount 不存在時 fail closed。實際選中的絕對路徑、bytes、mtime 與 SHA-256 必須寫入每日 input manifest，
prepared loader 的實際 source path還要逐 role 與 manifest完全相等。

`txfTickData/` 是台指期 TXF，不得替代個股期貨；個股期 raw 不做 SSD2 fallback。`stockfuture/` 類研究產物也不是
完整 raw tape。涵蓋日數會隨資料更新改變，production run以 frozen date inventory與每日 manifest為準，不在本文件
維護容易過時的總日數。

## 價格與時間

- NAS raw 價格是整數，依每列 `DecimalLocator` 還原；不得假設 SDK 已縮放。
- Raw `RecvTime` 是 UTC-aware nanosecond；本地現貨 tick 是 UTC clock、timezone-naive microsecond。
- 合併前統一成 UTC nanosecond；`TransTime` 只作稽核，不用於跨市場因果排序。
- 股期 raw 不含 `contract_size`、到期日與 `fut_ref_price`，需 join point-in-time futures basic info。
- 只使用最近到期且未到期的標準契約；不能只靠字串月份碼而忽略假日順延。
- 期貨成交 row 的五檔可能全零，必須維護最近有效 book。
- `B_mid` 使用直接 L1；期貨 taker 價需取明掛 L1 與 `BestBidPrice/BestAskPrice` 的可成交優價，並帶入對應 Lots。
- 同一 `RecvTime` 內的 TrialMatch／book 先後需再比較各市場自己的 `ChannelSeq`；兩市場間不得比較 sequence。

## Hard eligibility gates

```text
-9% < SpotPrice / SpotRefPrice - 1 < +8%
-9% < FuturePrice / FutureRefPrice - 1 < +8%
```

等號排除；兩腿各用自己的 RefPrice。A1／B1、實際 maker 價與 taker 會掃到的深度都需檢查。

`TrialMatch == 0` 才是 formal：

- Maker 與 hedge 市場都必須 formal。
- TrialMatch row 不可先刪掉再 as-of；需保留最新 raw state 並跨市場關閉 route。
- 非零回到零後，等新的正式有效 book 才重新開放。
- Gate 在 working order 期間失效會觸發 cancel；cancel race fill 不得刪除。
- Maker+taker route工作期間須同時重驗maker venue與hedge side的完整可執行depth；S1 Spot Ask exit另要求Future buy最差swept ask上方至少保留一個合法tick。任一venue raw book change都要喚醒。Gate關閉只建立causal cancel intent，actual cancel effect前或同cursor的真實fill不得hindsight刪除。
- Maker 已 fill 後 gate 失效需記 hedge failure／emergency outcome，不可 hindsight drop。

另排除零價有量、crossed book、缺 RefPrice 與不足 hedge depth。Book age 不可在 raw 前處理直接刪除；主政策與 100／250／500／1,000／5,000 ms sensitivity 分開保存。

## Feature allowlist

模型主要使用現貨 causal feature。以下欄位明確禁止當輸入：

- `Close`、`FutureHigh`、`FutureLow`
- `SpreadNarrowOrderTime`、`SpreadNarrowSide`
- `FutureAsk1_*`、`FutureBid1_*`
- `TakerSell_CloseBP`、`TakerBuy_CloseBP`
- `midEdge_*`

上述Future book檢查是deterministic execution-risk eligibility，不是模型feature；它不放寬`FutureAsk1_*`／`FutureBid1_*`的feature禁令。

現貨 makerFill 只有 A1／A2／B1／B2 的 `FillSeconds`，沒有 fill `RecvTime`、partial、cancel 或任意價位；精確 label 需從 raw 重建。
