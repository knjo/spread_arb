# EV lookup v18：查表、執行與容量獨立稽核（2026-09-08）

目前不能認定這套回測沒有 data leak，也不能據此推論市場熱度相近時實盤績效會接近。
已確認的優點是：已接受部位的總容量帳本正確，且 bpday 的前日查表版本在同一批候選成交假設下仍有價值。
主要阻礙是成交後篩選、錯配期貨合約、成交時間前移、出場 label 重複使用，以及未依成交時反腿價格入帳。

本次只新增 audit 程式、證據與本報告，沒有改動策略程式或覆寫既有績效。
稽核對象是 `maker/data/ev_lookup_audit_20260908/source_snapshot/` 的 v18 source；
`source_manifest.json` 固定其 SHA-256，稽核結束時原始 source 仍與快照相同。

## 範圍與可重現性

- 85 個交易日：2026-05-04 至 2026-09-02，沿用原研究實際涵蓋日。
- 重用 `v15_candidates.csv` 的 21,214 筆**進場候選**；不使用其出場、到期或截尾損益來重放 v18。
- 用凍結 v18 的原始 `mexit` 函數、每日原始研究格與 SSD makerFill 重算候選出場。
- 獨立重建容量分配，再以另一套 accepted-position event sweep 檢查開倉、平倉、carry 和總額。
- v17 20M、v18 20M／50M／100M：每組 85 日的 fills、cap rejects、same-day fills、flow、carry 完全一致；
  每日損益差異小於 0.00001 TWD。這驗證重放與研究一致，不代表成交假設已獲驗證。
- 三個日期另從 NAS 逐筆資料重建 S2：5/20、7/6、7/24，保留的 S2 候選數與價格逐筆對上存檔。
- 未執行真實委託、未重建完整四 route 的實盤 order lifecycle；沒有將原研究升格為 exact execution。

程式與重跑命令：[audit/README.md](../../src/ev_lookup/audit/README.md)。
證據根目錄：`maker/data/ev_lookup_audit_20260908/`。
主要索引：`checks.json`、`allocation_summary.json`、`execution_probes.json`、`audit_artifacts.json`。
Raw inputs 保存 path／bytes／mtime inventory；沒有宣稱所有 raw files 都做了 content-hash 驗證。

## 1. 查表與執行的因果性：未通過

### 可以確認因果性的部分

`decide.py` 的 P_sd 按 stream × 進場時段累積，回測在當日所有決策完成後才更新，下一日使用。
Lambda 使用前五日 capacity rejection 的估计 EV，`start_day` 接前日值。這兩段本身沒有使用當日未來結果。
`EntryDecider` 不持有日期或資訊可用時間；其因果性依賴呼叫端，不能只靠檔頭註解認證。

### 已確認的查表洩漏

1. `stacked_walkforward_backtest.py:156` 先找出今日會成功出場的 shadow carry，立即更新表；
   `:242` 再加入今天所有 same-day candidates 的完整結果，`:285` 才複製表做今天所有進場決策。
   早盤決策因此能讀到下午才會知道的結果。不是僅「當天後面幾筆」被影響。
2. `:263` 先找出今天會出場的真實 carry，`:288` 的 expected overnight 因而排除那些尚未發生的出場。
   gross 容量仍留到指定秒才釋放；此處洩漏的是 EV 的 slot 判斷，不能混稱 gross 提早釋放。
3. S2 合約家族／CS 來自 8/13 的面板，再回用於 5–8 月，沒有完整 as-of universe 證明；
   每日執行合約還用當日全日 `TotalFillLots.max()` 選擇，這在盤中不可知。

### 已確認的執行資訊／時序問題

- `:203` 用 `(t, t+1]` 的 future trades 判斷成交，`:208` 卻把成交時點記為 `t`，hedge 定價也取該秒格。
  對凍結原始 S2 loop 的最小反例：301.9 秒的 future print 被記為 301 秒；301.1 秒的 S1 反而排在它後面。
  三個 raw 樣本日 print-trigger 的時間前移中位數為 0.507／0.426／0.398 秒。
- `ext_grid_builder.py:67` 用每秒最後一筆行情產生標記為該秒起點的格。若仍在同秒內判定交易，
  會額外引入該秒尚未收到的 book。Anchor shift 一格不會修正價格欄位的這個時間問題。
- `mexit` 在觸價候選的 `Ask1_FillSeconds` 為 NaN 時直接改找後面的候選；
  1,215 次成功 exit evaluation 跳過了至少一個這種未來不成交 label。缺少與此相符的因果掛撤政策。
- 決策器接在**已知候選成交之後**。S1／S2 都在成交時更新後的 ab／effU 上篩選，
  不能把 rejected fills 解釋成實盤「根本沒有下單」。容量拒絕也不能抹掉已發生的成交回報。
- 特別直接的例子：7/24 S2 先觸發 70 筆 fill，然後因成交時 `ab <= 0` 等條件丟棄 5 筆。
  其中 2401 掛單時 ab=+34.965bp，book-cross 成交時 ab=-34.722bp，該筆被忽略，沒有 hedge／rollback 損益。
  2354、8112 也出現掛單時正 basis、成交時負 basis；另兩笔為零 basis。

## 2. 容量與平倉：帳本通過，實盤生命週期未通過

此程式的 cap 口徑是每組 paired position 的**現貨進場名目金額**，即 entry price × contract size。
不是兩腿名目金額相加，也不是每日 mark-to-market 後的曝險或期貨保證金。

| 原始 v18 | 日均已實現 PnL | 獨立重算最大持倉名目金額 | 超額事件 | 平倉早於開倉 |
|---|---:|---:|---:|---:|
| 20M | 28,481.605 | 19,999,960 | 0 | 0 |
| 50M | 77,487.732 | 50,000,000 | 0 | 0 |
| 100M | 134,828.063 | 100,000,000 | 0 | 0 |

已接受交易的日內未平倉與 carry 共用同一個 cap。Carry 在指定的出場秒釋放，未於開盤先釋放。
既有 carry 若碰到 `exp` 當日則維持，下一個樣本交易日才按既定 basis-zero 慣例釋放。
但是 exp 未正確綁定交易合約，故「照 exp 欄位執行正確」不能解讀成合約到期處理正確。

20M 組最大單一商品達 14.54M；新程式沒有舊 S1 的 10M 單商品限制。
若需求只指定總額 20M，總額檢查通過；若仍沿用舊 20M／10M 契約，單商品限制未落實。

誰先到誰先拿，只在**取整後的模型秒序**成立：

- S2 候選先 append、S1 後 append，只按整秒 stable sort，同秒固定讓 S2 優先。
- 全候選有 322 組跨流同秒競爭，涉及 675 筆候選。調換同秒的 stream 優先權，
  v18 20M 日均由 28,481.6 變成 28,101.7，說明 tie policy 已影響績效。
- 同秒一律先釋放 exit、再接受 entry；沒有兩市場同秒內的完成順序。將釋放延到下個決策秒，
  日均變成 28,143.8。這是時序敏感度，不是估算實際 latency 損失。
- 沒有 pending quote reservation、兩腿競爭後的撤單生效時間、double fill／額外 hedge 處置。
  實盤兩張 maker 都送出時，第二張可能在撤單完成前也成交；必須入帳、配對或補救，不能僅回傳 cap reject。

## 3. bpday 可以如何日內使用，及目前證據

bpday 的概念可以日內執行：D 日開盤前用截至 D-1 已解決的 shadow outcomes，
固定 `stream × ab bucket` 表；盤中只用當時可知的 ab 查表，收益效率過門檻才允許掛單。
不必預知下午有哪些更好的機會。

但它目前是**八格閾值篩選**，不是對每秒所有候選做最優排序，也不能證明保留了當天最好的交易。
通過同一格 gate 的候選仍按到達順序搶容量；固定門檻 8、分桶 30／50／80 和暖機 30 筆，
也尚未經獨立新時段驗證選參穩定性。

以下保持相同候選、相同出場近似，只換容量政策；全部為 20M、85 日、**已實現口徑**：

| 政策 | 日均已實現 PnL | 平均 carry | 期末仍未平倉名目金額 |
|---|---:|---:|---:|
| FCFS，無 EV／bpday gate | 16,599.1 | 18.729M | 15.068M |
| 僅 bpday，開盤前 20 日已解決資料 | 26,819.5 | 16.768M | 0.775M |
| 原始 v18（含當天查表結果） | 28,481.6 | 16.506M | 0.106M |
| v18，改為開盤前 20 日查表 | 29,766.4 | 16.370M | 0.775M |
| 再將 overnight slot 改成隨已發生退出更新的估計 | 29,341.1 | 16.668M | 0.775M |

去掉查表洩漏並沒有讓 bpday 優勢消失，也不能因此反推洩漏無害。
這支持繼續研究容量效率；所有列仍繼承原本的執行缺陷、未平倉估值缺失及不同期末持倉，
不能把增量直接認定為完整經濟收益或修正後可部署績效。

原本 `policy_replay.py` 的 bpday 實驗更不能當純 OOS 證據：

- 用 `day0 < 20260701` 的 15,161 筆進場樣本建表，其中 484 筆到 7/1 或之後才解決。
  這些標籤在 7/1 開盤不可知；5–6 月同時被拿來訓練及報績效。
- `candidate_dump.py:306` 將 74 筆期末 censored positions 直接給 `ab-34`，並標為 9/2 解決。
  這不是已實現的到期損益。
- `policy_replay.py:55` 把每筆完整最終損益記回進場日，與 v18 按實際解決日的日損益不同。
- candidate dump 在到期當日處理 settlement，v18 的既有持倉則次一樣本日處理；不能忽略口徑差異比較曲線。

## 4. 足以影響結論的執行缺陷

### 報價合約與成交合約錯配，以及 carry 合約身份遺失

Canonical grid 保存的 QuoteCode 沒有被 v18 用於過濾 NAS trades；它改選同商品當日最大成交量合約。
因此「這個合約的 Ask1-1 掛價」可能被「另一個合約的 print」判定成交。

5/20 的完整 S2 raw probe 保留 712 筆候選；713 次 print-trigger 中有 699 次兩合約不符，
其中 650 次在掛價合約的同一時間窗找不到符合該價的 print。
在真正被 20M 帳本接受的 79 筆 S2 中，76 筆用了異合約 print，68 筆沒有對應本合約 print。
這不等於證明它們在另一個完整下單模型中絕不會成交，但已推翻此處使用的成交證據。

例如 1301 在開盤後 347 秒以 45.95 模擬成交：掛價合約 CFFE6，判定用的是 CFFF6。
canonical mapping 的 CFFE6 `end_date` 是 5/20，`next_exp('20260520')` 卻給 6/17。
持倉僅保存 ValueCode 與手寫 exp，沒有 QuoteCode；跨日再取該商品最新日格，不能保證仍在同一合約出場。
6/17、7/15 的補充合約盤點也發現廣泛錯配；一般日 7/6 的錯配少很多，故問題有明顯到期日集中性。

### 出場未分配真實可成交量

20M 接受部位有 484 組重用相同 `(day, product, maker sequence, exit second)` 的情形，
涉及 1,400 個部位；最多 20 個 futures-equivalent units 共用一個 Ask1 label。

具體例：7/24 的 8039，sequence 1,986,698、Ask=253、原先 AskLots=37。
同一 label 讓 20 口期貨對應的 40 張現貨全部於 09:12:47 出場。
從該 quote 到聲稱出場秒，raw tape 中價位 >=253 的成交總量只有 41 張。
依目前 makerFill 的同價排隊累積假設，尚未扣掉前方 37 張就不能支持再成交我方 40 張。
真實撤單可能改變隊列，但此回測沒有建立能支持那 40 張成交的 volume allocation。

### 出場損益用目標，沒有用 maker 成交後的期貨成交價

`mexit` 只回傳 te；入帳固定用 `eu + 5 - 20/34`。沒有把該筆 Spot Ask1 maker 的絕對成交價，
以及 te 後反向期貨實際可執行價格帶入完整雙腿損益。

對 20M 的 3,237 筆 maker exits 做固定成交集合的價格診斷：現貨用 label 起點的 Ask1，
期貨用聲稱出場秒的 executable Ask，與原目標公式比較，差額平均 -13.34bp、
中位 -5.53bp、加總 -1,115,063 TWD。這不是更正後績效：尚無精確 +50ms、depth、
partial／failed hedge 或重新決策；只是顯示「觸及 target 就按 target 記帳」誤差可能相當大。

### 權益與成本範圍

報表只有已實現 PnL，沒有每日庫存浮盈虧，因此零負日與單調曲線不能當低風險證据。
融資／保證金成本也不在此輪結果。8/14 後只含 S2，不能將全期當成相同的雙流供給。

## 建議的後續順序

1. 保留秒級決策架構，先修資料與 execution identity：每筆綁定精確 QuoteCode／contract size／end_date，
   所有價格與成交證據來自該合約；extension grid 在整秒結束後才可供決策。
2. 所有查表在開盤前凍結；若盤中更新，只接受截至當下已完成的 outcomes，並保存 available_at。
3. 將 gate／資金管理移到可執行的掛单生命週期：送單前評估，保留 pending exposure，
   maker fill 不可刪除，雙流競爭後的取消／double fill 有明確處理。
4. 出場按實際可分配量、兩腿執行時間及價格入帳，未成功 hedge 的部位繼續占額度。
5. 加上每日 inventory valuation，再在同一 corrected execution engine 上比較 FCFS、bpday、EV+bpday。

不需要為此恢復整套奈秒引擎。秒級策略也能保持資料因果、合約身份、成交不可事後撤銷及數量守恆；
更細的時間戳可以只用於回測的事件排序與記帳，不必變成實盤決策頻率。
