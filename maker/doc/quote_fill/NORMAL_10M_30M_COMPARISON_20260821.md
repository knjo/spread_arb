# Normal carry：1,000 萬與 3,000 萬比較

更新日：2026-08-21

沿用 2,000 萬版本的相同設定：13:00 停止新倉、單品上限為 portfolio cap 的 30%、
正常 maker exit、不做貼價硬出、模型 carry 接到下一個交易日。

![1,000萬與3,000萬 Normal carry 回測](assets/normal_10m_30m_carry_comparison_20260821.png)

## 結果

| 指標 | 1,000萬 cap | 3,000萬 cap |
|---|---:|---:|
| Accepted positions | 1,209 | 1,981 |
| Gross | 1,749,300 | 3,705,800 |
| 指定交易稅費 | 1,035,535.76 | 2,031,899.95 |
| **Net** | **713,764.24** | **1,673,900.05** |
| Net／日 | **11,329.59** | **26,569.84** |
| Net／現貨新單 | 18.441 bp | 20.937 bp |
| 正／負日 | 58 / 5 | 56 / 7 |
| Realized cashflow MDD | 47,708.72 | 47,905.62 |

## 現貨交易量與模型 carry

| 指標 | 1,000萬 cap | 3,000萬 cap |
|---|---:|---:|
| 現貨新單／日 | 6.144M | 12.690M |
| 現貨買＋賣／日 | 12.304M | 25.386M |
| 模型 carry／日 | 7.148M | 12.506M |
| 模型 carry P90 | 9.826M | 21.268M |
| 模型 carry最高 | 9.997M | 25.755M |
| 盤中部位最高 | 9.9996M | 29.9775M |
| D→D+1模型帳務銜接 | 62 / 62 | 62 / 62 |

新單是 turnover flow，平倉釋放額度後可在同日再進，因此單日新單最高可達 19.557M／
65.694M，不代表同時持倉超過 cap；兩個版本的盤中部位均零次超限。

## 容量解讀

加入先前 2,000 萬結果後：

| Cap | 現貨新單／日 | Net／日 | 63日 Net |
|---:|---:|---:|---:|
| 10M | 6.144M | 11.33k | 0.714M |
| 20M | 10.648M | 22.40k | 1.411M |
| 30M | 12.690M | 26.57k | 1.674M |

10M→20M 的淨利約增加 97.7%，接近線性；20M→30M 的 cap 增加 50%，但新單額與淨利
只增加約 19.2%／18.6%，顯示這批樣本在 20M 以上已出現容量飽和。

三條 portfolio path 不是把同一筆交易按比例放大：carry、D+1 商品 exit-only gate 與
admission 順序會改變後續接受的交易，因此 accepted sets 不完全 nested。

## 口徑限制

- PnL 是 terminal date realized cashflow，沒有每日 carry MTM；新單則記在 entry date，
  不可用同一天 PnL／新單解讀為 same-day return。
- 行情 replay 於 13:20 結束。普通日模型 carry 是真正 13:30 留倉的保守上界；三個
  到期日再以官方 Close 作帳後歸零。
- 固定 45 檔商品為回溯 cohort，仍有 data leak；部分 terminal 使用 continuation 近似，
  所以結果僅是研究估計。

逐日資料：
[`assets/normal_10m_30m_carry_comparison_daily_20260821.csv`](assets/normal_10m_30m_carry_comparison_daily_20260821.csv)。
