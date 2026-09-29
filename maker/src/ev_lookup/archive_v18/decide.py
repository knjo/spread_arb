"""逐筆下單決策鏈（進場側）。回測與實盤共用同一條鏈，順序固定：

  候選 fill (t, strm, effU, ab, ntl)
    ├─ 1. 結構保底期望值  est = P_sd·(effU−L−20) + (1−P_sd)·tail
    │        tail = ab−34   若預期隔夜交割額度還有位子（slot_free）
    │             = 強平估損 若沒位子（避免收盤跳樓）
    ├─ 2. 容量機會成本    admit_ev ⟺ est ≥ λ × 預期佔用天(0.6)
    │        λ = 前 5 日被容量拒絕 fill 的估計 EV ÷ cap（bp/日，clip 15）
    ├─ 3. 分流深水格閘門  cell_gate(strm, ab)：歷史 (bp/佔用天) ≥ 8 才掛（walk-forward）
    └─ 4. 硬容量          gross + ntl ≤ cap
  四關全過 → 下單（回測中 = 收下這筆 fill）；任一關不過 → 不下，並記錄是哪一關。

所有輸入在決策時點都是已知量（effU/ab 由我方掛價與當下可執行價算出；P_sd、λ、格統計皆
只用 <D 或 <t 的資料）。回測與實盤唯一差別：回測的「候選 fill」來自 makerFill/trade-print
真值模型，實盤來自真實成交回報。
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ev_rules import est_short, est_long, cell_of, cell_gate, shadow_price

TODB = np.array([0, 3600, 9000, 1e9])     # 進場時段桶：早/中/晚
EXPECT_DAYS = 0.6                          # λ 乘的預期佔用天（同日率 ~0.7 時的平均）


@dataclass
class Decision:
    admit: bool
    reason: str          # 'ok' | 'ev' | 'cell' | 'cap'
    est_bp: float
    lam_bp: float
    p_sd: float
    slot_free: bool
    cell: str


class EntryDecider:
    """持有 walk-forward 狀態：P_sd 表、格統計、λ 的拒絕紀錄。"""

    def __init__(self, cap_twd: float):
        self.cap = cap_twd
        self.psd_cnt = {}    # (strm, tod) -> [n, n_sd]
        self.cell_sum = {}   # cell -> [pnl_sum, days_sum, n]
        self.rej_hist = []   # 每日被容量拒絕的估計 EV（TWD）合計，取最近 5 日
        self.lam = 0.0

    # ---- walk-forward 表更新（皆在資訊已知的時點才呼叫）
    def update_psd(self, strm: str, t: int, same_day: bool):
        k = (strm, int(np.searchsorted(TODB, t, side='right') - 1))
        c = self.psd_cnt.setdefault(k, [0, 0])
        c[0] += 1
        c[1] += int(same_day)

    def update_cell(self, cell: str, pnl_bp: float, days: float, sign: int = 1):
        s = self.cell_sum.setdefault(cell, [0.0, 0.0, 0])
        s[0] += sign * pnl_bp
        s[1] += sign * days
        s[2] += sign

    def start_day(self, rejected_ev_twd_yesterday: float | None):
        if rejected_ev_twd_yesterday is not None:
            self.rej_hist.append(rejected_ev_twd_yesterday)
            self.rej_hist = self.rej_hist[-5:]
        self.lam = shadow_price(self.rej_hist, self.cap)

    # ---- 查表
    def p_sd(self, strm: str, t: int, default: float = 0.68) -> float:
        k = (strm, int(np.searchsorted(TODB, t, side='right') - 1))
        c = self.psd_cnt.get(k)
        return c[1] / c[0] if c and c[0] >= 200 else default

    # ---- 決策
    def decide(self, strm: str, t: int, eff_u: float, ab: float, ntl: float,
               gross: float, expected_overnight: float,
               carry_ok_long: bool = True) -> Decision:
        p = self.p_sd(strm, t)
        slot_free = expected_overnight + ntl <= self.cap
        if strm == 'S2' or strm == 'S1':
            est = est_short(eff_u, ab, p, L=-5.0, slot_free=slot_free)
        else:                                    # 'L' 多價差流（未接入 v18，保留介面）
            est = est_long(eff_u, p, carry_ok_long and slot_free, p_recover=0.6)
        cell = cell_of(strm, ab)
        s = self.cell_sum.get(cell, [0.0, 0.0, 0])
        if est < self.lam * EXPECT_DAYS:
            return Decision(False, 'ev', est, self.lam, p, slot_free, cell)
        if not cell_gate(s[0], s[1], s[2]):
            return Decision(False, 'cell', est, self.lam, p, slot_free, cell)
        if gross + ntl > self.cap:
            return Decision(False, 'cap', est, self.lam, p, slot_free, cell)
        return Decision(True, 'ok', est, self.lam, p, slot_free, cell)
