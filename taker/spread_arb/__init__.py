"""期現貨套利 低頻分析模組。

分層（對應 METHODOLOGY.md 各步驟）：
  contract   合約代碼 / 結算日工具
  preprocess 第1層：讀檔 + 還原價格 + 篩近月
  align      第2層：取優價 + as-of join          (待寫)
  spread     第3層：價差 + 事件狀態機            (待寫)
  metrics    第4層：每事件指標                   (待寫)
  cost       成本 / 淨獲利率                     (待寫)
  report     pivot 輸出                          (待寫)
"""
