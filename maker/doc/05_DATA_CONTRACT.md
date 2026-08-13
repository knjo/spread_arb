# 2026 資料契約

## 來源與涵蓋

HFT 資料根目錄為小寫 `data/`。

| 資料 | 路徑 | 2026 涵蓋 |
|---|---|---|
| 現貨 tick | `HFT/data/tickData/{YYYYMMDD}_StockTick.parquet` | 145 日，01-02 至 08-11，缺 07-10 |
| 現貨 feature | `HFT/data/tickFeature/{YYYYMMDD}_tickFeature.parquet` | 同上 |
| 現貨 makerFill | `HFT/data/makerFill/{YYYYMMDD}_makerFill.parquet` | 同上 |
| NAS 現貨 raw | `/mnt/NAS/Parquet/Ticks/2026/MM/DD/stock_round.parquet` | 146 日 |
| NAS 個股期 raw | `/mnt/NAS/Parquet/Ticks/2026/MM/DD/stock_futures.parquet` | 132 日；01-26 至 08-11 連續，另有 01-13、01-20 |

`HFT/data/txfTickData/` 是 TXF，不得替代個股期貨。`HFT/data/stockfuture/` 主要是研究產物，也不是完整 raw tape。

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
- Maker 已 fill 後 gate 失效需記 hedge failure／emergency outcome，不可 hindsight drop。

另排除零價有量、crossed book、缺 RefPrice 與不足 hedge depth。Book age 不可在 raw 前處理直接刪除；主政策與 100／250／500／1,000／5,000 ms sensitivity 分開保存。

## Feature allowlist

模型主要使用現貨 causal feature。以下欄位明確禁止當輸入：

- `Close`、`FutureHigh`、`FutureLow`
- `SpreadNarrowOrderTime`、`SpreadNarrowSide`
- `FutureAsk1_*`、`FutureBid1_*`
- `TakerSell_CloseBP`、`TakerBuy_CloseBP`
- `midEdge_*`

現貨 makerFill 只有 A1／A2／B1／B2 的 `FillSeconds`，沒有 fill `RecvTime`、partial、cancel 或任意價位；精確 label 需從 raw 重建。
