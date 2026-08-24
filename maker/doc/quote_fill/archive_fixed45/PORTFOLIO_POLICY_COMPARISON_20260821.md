# AB1/2 maker 策略：留倉政策、10–50M 容量與完整度總結

更新日：2026-08-21

## 結論先行

本輪已把原本沒有 terminal cashflow 的 1,239 筆 `unknown` 與 22 筆
`censored` 接回 portfolio backtest。一般情況不是把它們補零，而是把完整部位帶到
後續交易日，沿用原本 normal maker exit；若一路到期仍未平倉，才在到期日用現貨與
期貨各自的官方 daily `close_price` 同時結清。現在 3,672／3,672 個建倉 path 都有
terminal 價格。

兩個使用者指定版本也都已完成：

1. **Normal carry**：13:00 後不開新倉，既有部位繼續原 normal maker exit；可跨日，
   最晚到期用兩腿 Close 結清。
2. **Aggressive 13:00**：13:00 後不開新倉，開始以期貨 B1 買回與現貨 A1 賣出兩條
   maker route 競賽；任一 full fill 後 50 ms taker 對沖另一腿並 nominal OCO。目的為把
   13:20 的留倉壓到 20M 以下。

若目標是評估「盤中放到 30M、在目前研究終點 13:20 壓回 20M」，最合適的研究候選是
**30M 盤中 hard cap＋20M 留倉 target**：保守 aggressive-exit template capacity
proxy 下，63／63 日在 13:20
都達標，盤中實際峰值
29.978M，13:20 平均／最高留倉 11.765M／19.934M，63 日 after-cost net 為
1,502,177 TWD。

若 **20M 是實盤或真正隔夜一元都不可突破的 hard limit**，本回測固定 entry-price
notional 口徑內只有 20M admission cap 有確定上界；實盤仍需替未決單 reservation、
late fill／cancel race 與市值漂移保留 safety buffer。30M 版在完成 13:20–13:30 replay
與收盤後狀態前，只能叫 13:20 research candidate，不能叫 overnight hard guarantee。

但這不是免費增加容量。30M aggressive 相較 20M normal 只多約 91.1k TWD（約
1.45k／日），相較 30M normal 則少 171.7k TWD。被強制積極平掉的 81 筆本身合計
net **-221.3k TWD**。因此 aggressive controller 的定位是「用部分收益購買較低隔夜
曝險」，不是提高單筆 EV。

更正確的同部位比較已另外重建：只看「當日新建、當日 aggressive 出，而原 normal
會跨日出」的部位，確實因當沖稅省約 14.8 bp；但目前無條件貼 B1／A1 的 gross
成交價相對未來 normal exit 平均差 75.8–80.1 bp，扣完稅差後仍少 61.1–65.3 bp。
所以使用者提出的經濟判斷式是對的；目前不成立的是「兩種出口價格差不多」這個實證
前提。下一版應改成 tax-aware、capacity-aware selective close，而不是 unconditional
FIFO forced close。

40M／50M 雖有更高總 net，但在保守 aggressive-exit capacity proxy 下各有 2／63 日
於 13:20 仍超過 20M，
故不能宣稱它們已能可靠地把 4,000／5,000 萬盤中部位壓回 2,000 萬。

## 政策契約

兩個版本共同採用：

- hard cap：10／20／30／40／50M TWD，口徑為現貨腿進場價的 one-way notional；
- 單一商品不得超過 portfolio hard cap 的 30%；
- 13:00 起禁止新 entry，但正常出場仍可繼續；
- 若 D 日開盤已有該商品前日 carry，該商品在 D 日整天只出不進；即使早盤平完，
  當日也不重新進場；
- historical backtest 使用 nominal immediate cancel。這裡不需要 exchange ACK；
  cancel race 只屬實盤近似誤差，不是本次歷史研究的 blocker；
- 沒有另加 stop-loss 或因帳面虧損強制平倉；負 P&L 只來自回測指定的 terminal 價差、
  50 ms hedge 價與逐腿成本，到期才依兩腿 Close 強制結清並釋放部位；
- 現行行情 replay 到 13:20，不是完整的 13:30 收盤 replay。

Aggressive controller 並非 13:00 靜態掛一次。舊價在價格不動或向有利方向移動時繼續
累積 queue age；若新價也要參與，建立另一張新 order layer；價格後撤時取消超前層。
期貨 B1 maker 與現貨 A1 maker 誰先 full fill，就在 `fill + 50 ms` 用另一腿 taker
完成 delta-neutral paired exit，並 nominal 取消 sibling。

## Normal carry：10–50M 結果

期間為 2026-05-21 至 2026-08-19，共 63 sessions。`Net` 已扣本研究的完整逐腿
交易成本；金額均為 TWD。

| Cap | Accept | 每日單邊新倉 | 回測四腿記帳額／日 | 盤中峰值 | 13:20 平均／最高留倉 | Gross / cost / net | Net／日 | Loss positions／負日 | Realized MDD |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10M | 1,209 | 6.144M | 24.647M | 10.000M | 7.370M / 9.997M | 1,749,300 / 1,035,536 / 713,764 | 11.3k | 244 / 5 | 47,709 |
| 20M | 1,779 | 10.648M | 42.684M | 19.999M | 11.604M / 19.950M | 3,142,800 / 1,731,677 / 1,411,123 | 22.4k | 343 / 6 | 53,880 |
| 30M | 1,981 | 12.690M | 50.860M | 29.977M | 13.023M / 25.755M | 3,705,800 / 2,031,900 / 1,673,900 | 26.6k | 383 / 7 | 47,906 |
| 40M | 2,050 | 13.536M | 54.254M | 39.961M | 14.016M / 30.128M | 3,979,700 / 2,159,152 / 1,820,548 | 28.9k | 384 / 6 | 47,906 |
| 50M | 2,090 | 14.007M | 56.144M | 49.000M | 14.409M / 32.843M | 4,135,200 / 2,225,297 / 1,909,903 | 30.3k | 384 / 6 | 47,906 |

`回測四腿記帳額` 是 entry 現貨＋期貨，以及回測 terminal 所記 exit 現貨＋期貨的
合計，不是資金占用。Upstream 全母體包含 1,090 筆近似 continuation 與 171 筆到期
Close mark；每個 cap 的記帳額只包含該 cap 實際接受的子集，因此也不是全數可逐筆
驗證的實際市場成交額。`盤中峰值` 與 `13:20 留倉` 才是本研究的 one-way
position-notional 口徑。

Normal carry 的 13:20 留倉不保證低於 20M：30M／40M／50M 分別有 11／14／16 日
高於 20M；最高超出 5.755M／10.128M／12.843M。

## Aggressive 13:00：保守 aggressive-exit template 容量結果

主結果採 `conservative`：同一 product-day 的 observed aggressive exit template 最多
供一個 position 使用，不重複計算相同市場容量。10M／20M 因 hard cap 本身不超過
target，所以 controller 不需額外強平，與 normal 完全相同。

| Cap | Accept | Aggressive closes | 每日單邊新倉 | 回測四腿記帳額／日 | 盤中峰值 | 13:20 平均／最高留倉 | 達標日 | Gross / cost / net | Net／日 | Loss positions／負日 | MDD |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10M | 1,209 | 0 | 6.144M | 24.647M | 10.000M | 7.370M / 9.997M | 63/63 | 1,749,300 / 1,035,536 / 713,764 | 11.3k | 244 / 5 | 47,709 |
| 20M | 1,779 | 0 | 10.648M | 42.684M | 19.999M | 11.604M / 19.950M | 63/63 | 3,142,800 / 1,731,677 / 1,411,123 | 22.4k | 343 / 6 | 53,880 |
| 30M | 2,043 | 81 | 13.028M | 52.227M | 29.978M | 11.765M / 19.934M | 63/63 | 3,584,450 / 2,082,273 / 1,502,177 | 23.8k | 448 / 14 | 81,990 |
| 40M | 2,156 | 98 | 14.323M | 57.435M | 39.961M | 11.707M / 21.577M | 61/63 | 3,923,450 / 2,275,486 / 1,647,964 | 26.2k | 469 / 12 | 53,006 |
| 50M | 2,180 | 79 | 14.673M | 58.838M | 49.000M | 11.783M / 22.389M | 61/63 | 4,106,450 / 2,317,632 / 1,788,818 | 28.4k | 446 / 10 | 15,724 |

這裡的 `達標日` 是 13:20 snapshot；40M／50M 在 aggressive controller 當下各為
60／63 日達標，其中一天又由 13:20 前的 normal exit 自然降回 target，所以最終是
61／63。

上述 13:20 exposure 是到期 Close accounting release 前的保守 snapshot。把當日到期
部位依使用者規則在 Close 結清後，normal 30／40／50M 的平均記帳部位降為
12.506M／13.487M／13.880M；aggressive conservative 則降為
11.402M／11.297M／11.473M。各自 peak 與 20M 達標日仍與主表相同。這只表示到期部位
不再占帳面部位，仍不是 13:20–13:30 的完整市場 replay。

Aggressive-only 子集：

| Cap | Gross | Cost | Net / bp | Wins / losses |
|---:|---:|---:|---:|---:|
| 30M | -129,800 | 91,530 | -221,330 / -74.05 | 2 / 79 |
| 40M | -115,600 | 100,131 | -215,731 / -60.53 | 4 / 94 |
| 50M | -95,100 | 79,311 | -174,411 / -58.85 | 5 / 74 |

### 正確比較：今天積極出 vs 同部位未來 normal 出

15 bp 當沖稅優勢只適用「建倉日當天」就 aggressive exit、且 normal 原本會跨日的
部位。舊庫存已經隔夜，今天或明天賣都適用隔夜稅，不能再把 15 bp 算一次。以下排除
normal 本來也會同日出的部位，逐 `policy_path_id` 以同一 entry 比較兩種 terminal；
各 cap 是不同 counterfactual，不可跨 cap 相加。

| Cap | 同日積極／normal跨日 positions | Notional | Aggressive gross − normal gross | 成本節省 | Aggressive net − normal net | 少占 session-boundaries |
|---:|---:|---:|---:|---:|---:|---:|
| 30M | 29 | 9.259M | -75.82 bp | +14.77 bp | **-61.05 bp / -56.5k** | 平均 2.41 |
| 40M | 48 | 16.646M | -76.24 bp | +14.80 bp | **-61.44 bp / -102.3k** | 平均 3.27 |
| 50M | 42 | 15.175M | -80.13 bp | +14.82 bp | **-65.31 bp / -99.1k** | 平均 3.71 |

因此稅務假設本身已正確進入 cost engine，問題在 forced B1／A1 相對 future normal
Lower exit 的價格 concession 遠超過 15 bp。30／40／50M 全 aggressive-close 樣本中，
只有約 11.1%／7.1%／11.4% 的 aggressive exit 不比 normal 差超過 15 bp（包含
aggressive 更佳者）；正 delta 只有 5／3／4 筆。這是 ex-post policy evaluation，
不能把未來 normal terminal 當 live signal。

### 降低留倉後，實際多得到多少交易資格

容量效益會從後續交易日發生：13:00 後本來就禁止新倉；若今日成功平掉該商品的全部
剩餘部位，商品在下一交易日才不會因 opening carry 而整日 exit-only。只平掉部分部位
會釋放 portfolio notional，但不會解除該商品的 D+1 gate。

| Cap | 相對同 cap normal accepted 淨增 | one-way entry／日淨增 | 到期Close後 overnight notional-days | 平均記帳留倉 | 同 cap net delta |
|---:|---:|---:|---:|---:|---:|
| 30M | +62 | +0.337M | -8.82% | 12.506M → 11.402M | -171.7k |
| 40M | +106 | +0.788M | -16.23% | 13.487M → 11.297M | -172.6k |
| 50M | +90 | +0.666M | -17.34% | 13.880M → 11.473M | -121.1k |

容量回饋確實存在，但相對同 cap 的每日 entry-turnover 增量中位數三組都是 0，效益
集中在少數原本會被 opening-carry gate 擋住的日期。相對 20M normal，30M aggressive
則平均每日多 2.379M one-way entry、63 日 net 多 91.1k，且研究內 13:20 為 63／63
日不超過 20M；這仍是目前最乾淨的容量候選。

下一版 controller 的判斷式應為：

`預估等待 normal exit 的 gross 價格優勢 <= 可取得的當沖稅率節省 + overnight capacity shadow value + inventory risk price`

其中建倉當日且 normal 預計跨日者，稅率節省約 15 bp；舊庫存的稅率差為 0 bp。
預估 normal exit 只能使用 D-1／當下可見特徵，不能偷看實際未來 terminal。Capacity
shadow value 也要同時包含 portfolio 額度與「是否清掉該商品最後一筆、使 D+1 不再
exit-only」的商品級價值。若 selective close 後仍超過 hard risk target，才由另一層
明確的風控強平處理。

另外有 optimistic template-reuse 分支，30／40／50M 都能在 63／63 日達標，net 為
1.631M／1.718M／1.750M；但它允許多個 position 重複使用同一 observed market
capacity，只能當容量上界，不能把其 P&L 當正式結果。

![20M normal、30M normal 與 30M aggressive 的累積淨利、13:20 留倉及新倉量](assets/portfolio_policy_comparison_20260821.png)

## 為何 20M 上限只有約 2.24 萬／日

20M 是「同時未平倉部位」的 hard cap，不是每天固定成交 20M，也不是每筆都賺
20 bp 後立刻把資金無限周轉。

63 日內，20M normal：

- accepted entry one-way turnover 共 670.853M，平均 10.648M／日；
- gross 共 3,142,800，平均約 49.9k／日；
- 正式逐腿成本共 1,731,677，平均約 27.5k／日；
- net 共 1,411,123，平均約 22.4k／日；
- net／entry turnover 為 21.03 bp。

所以「約 20 bp」確實存在，但它乘的是平均每天實際接受的新倉 10.648M，不是 hard
cap 20M；而且成本已從 gross 扣除。再加上跨日持倉持續占 cap、開盤 carry 商品整天
exit-only、單品 30% 上限與許多候選被容量拒絕，資本不會每天完整轉一圈。

## 20M normal、不硬出的日內完成率與週轉

本節固定 `20M hard cap／單品30%／13:00停止新倉／normal exit only`。63 日共接受
1,779 筆部位，全部最終定價。

| 日內完成分母 | 結果 | 解讀 |
|---|---:|---|
| Position count | 991 / 1,779 = **55.705%** | 含4筆同日到期Close；排除後987 / 1,779 = 55.481% |
| Entry one-way notional | 407.972M / 670.853M = **60.814%** | 排除同日到期Close後60.667% |
| Entry product-days全部平完 | 176 / 438 = **40.183%** | 172個正常全平＋4個靠同日到期Close；另79個mixed、183個全數跨日 |
| Portfolio整日完全flat | 0 / 63 at 13:20 | 到期Close後有3 / 63日flat；普通非到期日沒有整體全平 |

因此若「日內沖完比例」是逐筆部位，答案約55.7%；若要求同一商品當日開出的所有部位
都清光、使該商品不帶到D+1，答案約40.2%。跨日部位為788／1,779=44.3%；其中497筆
在下一交易session出場，全部跨日部位的holding p50／p90為1／7個session-boundaries，
最長23。

20M是同時未平倉存量上限，不是當日成交額上限。正式週轉如下：

| 口徑 | 63日總額 | 日均 | P50 | P90 | 單日最高 | 日均／20M cap |
|---|---:|---:|---:|---:|---:|---:|
| Entry one-way spot | 670.853M | **10.648M** | 7.556M | 24.065M | 43.501M | **0.532x** |
| Entry paired two-leg | 1,345.668M | 21.360M | 15.146M | 48.306M | 87.139M | 1.068x |
| Exit one-way spot | 671.286M | 10.655M | 8.723M | 21.999M | 38.899M | 0.533x |
| Exit paired two-leg | 1,343.391M | 21.324M | 17.448M | 44.252M | 78.122M | 1.066x |
| Entry＋exit完整四腿 | 2,689.059M | **42.683M** | 31.291M | 85.344M | 161.562M | **2.134x** |

經濟上的新建部位週轉應看 entry one-way：平均每天使用相當於20M cap的0.532輪。券商／
市場成交額則看四腿，平均42.683M／日；每個完整delta-neutral cycle自然約是單邊部位的
四倍，不能把2.134x誤解為資金真的重複用了2.134輪。每日43.501M的最大新倉量可高於
20M，是因為早盤平倉釋放額度後又重新進場。

若只看當日開且當日平的991筆，其四腿記帳額共1,632.549M，平均攤63日為
25.913M／日，占全四腿60.71%；跨日788筆的最終四腿記帳額為1,056.511M，占39.29%。
總turnover含1,146筆原始completed、543筆近似continuation及90筆到期Close；後兩類是
回測記帳成交額，不全是逐tick executable fill。42.683M／日可再拆為原始completed
28.800M、近似continuation 12.083M、到期Close mark 1.801M。

## Unknown、隔日出場與到期處理

原始 3,672 positions 的 terminal 分布為：

- 2,411 筆原本已有完整 terminal cashflow，保持原結果；
- 1,090 筆 `unknown` 以完整 carry 假設接續後續 normal maker-exit replay；
- 149 筆 `unknown` 最後走到到期日，用兩腿當日 Close；
- 22 筆 `censored` 同樣在到期日用兩腿當日 Close。

也就是說，`unknown` 不再等於「沒現金流」或零 P&L，而且不是只硬切 D+1；它會逐日
沿用正常出場，直到成交或到期。20M cap 接受部位的持有 session-boundary p50／p90
為 0／3，最長 23，平均約 1.12。

特別只看 20M cap 接受的 632 筆 full-carry-imputed unknown：4 筆（0.6%）在同一
session 的到期 Close 結清，358 筆（56.6%）於下一個交易 session 內出場，270 筆
（42.7%）需跨至少兩個 session；平均跨 2.84 個 session-boundaries，最長 23。這就是
本輪用來估計「unknown 隔天能否正常出場」的主分布，而不是把 632 筆都假定 D+1
一定平得掉。

這仍是大致估計：1,090 筆 continuation 使用每秒末狀態＋SpreadPair epoch 變化；抽樣
15 個 product-session 的 terminal 分類、日期與勝出 route 一致率為 100%，但 gross
bp 完全一致率 59.3%，絕對誤差 p95 為 32.38 bp。更重要的是，原始 `unknown` 可能
其實已在缺失區間部分或全部平倉，因此完整 carry 有 double-exit bias。

到期的 171 筆使用同一到期日：

- 現貨：TWSE daily `close_price`；
- 期貨：TAIFEX day-session daily `close_price`；
- 45／45 個商品契約組都有兩邊 Close；
- 不使用期貨 settlement price。

這是會計強制結清 mark，不代表該 Close 有足夠可成交深度。171 筆到期 path 的 gross
由舊 fallback 的 -219.0k 改為 +74.7k，增加 293.7k。

## 已套用的成本

- 現貨買、賣 commission：各 `14.25 bp × 0.12 = 1.71 bp`；
- 現貨賣出稅：當沖 15 bp、隔夜 30 bp；
- 期貨買、賣稅：各 0.2 bp；
- 期貨 commission：買進 20 TWD、賣出 20 TWD，每個完整 cycle 共 40 TWD。

Gross 已使用 entry maker fill、執行模型中的 `+50 ms` taker hedge，以及回測選定的
exit／terminal price，因此不重複再扣一次 50 ms slippage；到期 Close 仍只是 mark，
不是 executable fill。現行結果未含融資／保證金資金成本、最低手續費 rounding，且
realized MDD 沒有逐日 carry MTM。

## 分段表現：最新一段偏弱

Normal carry 的月別 realized net（千 TWD）：

| Cap | May | Jun | Jul | Aug 至 08-19 |
|---:|---:|---:|---:|---:|
| 10M | 117.6 | 417.4 | 204.2 | -25.4 |
| 20M | 200.6 | 757.1 | 441.3 | 12.1 |
| 30M | 238.7 | 882.2 | 535.3 | 17.7 |
| 40M | 271.0 | 941.5 | 603.9 | 4.2 |
| 50M | 293.7 | 970.1 | 641.9 | 4.2 |

Aggressive conservative 30／40／50M 在 Aug 至 08-19 分別為 -30.8k／48.3k／48.3k。
因此不能說 May、June、Jul–Aug 三段都穩定；尤其 August 明顯轉弱。現有資料最晚到
2026-08-19，下一個真正 unseen 日／下一段資料尚未出現在 workspace，不能先製造
test 結果。

## q、商品池與研究完整度

Upstream 執行樣本目前有 302,582 張去重後 AB1/2 physical quotes：full fill 3,679
（1.216%）、partial 119（0.039%），298,903 張（98.784%）在策略終止時需要 nominal
cancel。Full fill 後 3,672／3,679（99.81%）能在 50 ms 找到 hedge 價；full-fill queue
wait p50／p90 為 6.22／113.73 秒，50 ms signed hedge slippage mean／p90 為
9.48／26.46 bp。這些是獨立 counterfactual labels，尚未把多張單共同消耗的真實市場
成交量完整配置。正式 `conservative` 分支只把 aggressive exit 的同一 product-day
template 限制為最多供一個 position 使用，是 aggressive-exit capacity proxy；它沒有
解決 upstream entry labels 或 normal exits 的跨 position joint-volume allocation。

目前固定研究候選是 **q95 + Lower exit + AB1/2 only**。59 個 prequential decision
days 都選到 q95，但先前排名只用 completed paths，且 raw replay universe 是利用
May–Aug 結果事後固定的 45 檔 cohort；這不等於無偏的 `best q`，正式旗標仍為
`best_q_selection_go=false`。

44 檔有 selected paths，不是因為市場只有 44 檔。稽核顯示在同一期間，cohort 外仍有
21,258 個 D-safe pass 的商品-route-days 沒有 raw replay。這是目前最大的 universe
selection data leak。正確做法是先對 D 日可能入選的全市場商品建 raw/cache，然後只用
D-1 以前的資料形成 D 日 allowlist；若 D 日 gate fail 但已有庫存，保留 exit-only，
不強制虧損平倉。

現有 cohort 的 pre-open eligibility audit 共 711 product-days／3,672 paths：day-trade
X=656、Y=55、N=0；交易方式、處置、限價交易、撮合間隔與 entry 觸及／越過漲跌停的
異常都是 0。另有 47 product-days 的 `attention_mark` 非零，目前只保留欄位，尚未
解讀或納入 gate。因此這批樣本的已實作靜態 eligibility gate 沒改變結果，但 runtime
gate 尚未接入正式 controller，不能因此聲稱全市場 production eligibility 已完成。

按研究層級拆開：

| 層級 | 狀態 | 還缺什麼 |
|---|---|---|
| AB1/2 掛單、queue age、後撤與 50 ms hedge | 已完成研究標籤 | partial/joint-volume 的全組合配置 |
| 3,672 terminal cashflow | 100% 已定價 | 1,090 continuation 的逐 tick 精化、消除 double-exit bias |
| 10–50M cap、單品 30%、兩種留倉政策與成本 | 已完成且 v3 通過獨立重建 | 13:20–13:30 完整 queue replay |
| D-1 商品池 | cohort 內有 causal gate 稽核；q95 目前用 q50/q80 route-gate consensus proxy | exact q95 gate row、全市場 raw replay，移除 45 檔回看篩選 |
| Best q | q95 是 frozen challenger | q50/q80/q95 同口徑 terminal/cost/cap，在完全 unseen period 比較 |
| Production | **NO-GO** | 上述 universe、精度、joint capacity、實盤風險與監控 |

目前可把結果當成「政策與容量診斷」，不能當正式 production strategy equity curve。

## 下一段 unseen shadow test

下一段測試應先凍結，不再看完結果換參數：

- 主候選：q95／Lower／AB1–2；
- 對照：q50、q80，同一商品池與成本；
- 每日 D 只讀 D-1 已發布的 boundary、liquidity 與 eligibility；
- 同時輸出 execution、四腿 turnover、盤中 peak、13:20／13:30 carry、逐腿成本、
  realized P&L、carry MTM、loss tails 與 cap rejection；
- 比較 normal carry 與 30M／20M aggressive target，先不因短期輸贏改 q。

現有 cache 硬切在 13:20。原始 63 日的 13:20–13:30 資料存在，但需從 13:00 重播
完整 queue，不能只把尾端十分快照接上；因此本報告的 13:20 completed count 是完整
13:30 的下界、13:20 carry 是上界，但 P&L 沒有單調上下界。

## 正式產物

- Normal terminal v3：
  `maker/data/walkforward/prequential_challenger_ab12_supplemental_carry_20260821_v3_expiry_daily_close`
  - `complete.json` SHA-256：
    `402d050e6fed75c38cea540fd31dd09994f6a24ef91a5411ea4e715e14cb691d`
- Normal cap sweep：
  `maker/data/walkforward/normal_carry_cap_sweep_20260821_v2_expiry_close_exit_only`
  - `complete.json` SHA-256：
    `156715157532d99e75aa194ae5babd8a8e79d89c86aefd59b21a03569bd51c3c`
- Aggressive formal v3：
  `maker/data/walkforward/aggressive_1300_exit_analysis_60d_20260821_v3_post13_timeline_fix`
  - `complete.json` SHA-256：
    `21bed1b6bbd9137e44fc3fcb43172a090498eb45ca279e07d3284191f6668330`
  - marker payload SHA-256：
    `c50cd22d6c21bcec8359688fb14883f9b24a8e3179b24e7da0b556391a829553`

Aggressive v3 已通過 17 個 controller tests、42 個相關 tests、540 個完整 quote-fill
tests，以及對 36,720 position-scenario rows／630 daily states 的獨立逐列重建。正式
重建 wall time 5 分 16 秒、peak memory 2.5 GiB、swap 0；預先建立 event/template 欄位後，
已不再需要每個 cap 各自重跑整份 raw tick。

先前 v2 有「13:00 後被禁的新 entry 錯誤推進 normal clock」問題，已隔離在
`aggressive_1300_exit_analysis_60d_20260821_v2_invalid_post13_release_20260821` 並標示
`INVALID_DO_NOT_USE`；任何政策結論都只能引用上述正式 v3。
