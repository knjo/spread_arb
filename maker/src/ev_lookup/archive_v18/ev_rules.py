"""EV 查表框架的全部決策式（純函數，無 I/O）。2026-09-08 交接版。

設計哲學（使用者 2026-09-01 定調）：每個決策點都是「查表 + 一條期望值比較式」，
不用 if-else 規則堆疊。四個決策點共用同一個 argmax EV 骨架：
  1. 進場准入（要不要收這筆 fill / 要不要掛）
  2. 出場目標 L 的選擇
  3. 續掛 maker 出場 vs 立刻 taker cross
  4. 容量分配（多流搶同一個交割額度）
所有機率表皆 walk-forward：只用嚴格 <D 的觀測，逐日更新。

成本口徑（使用者指定）：現貨當沖 20bp、隔夜 34bp（純稅費；hedge 腿滑價另計，
但本框架的 effU 已用可執行價定價，滑價實際上已內含）。多單留倉需融券，隔夜 +20bp → 54bp。
"""
from __future__ import annotations

import numpy as np

COST_SD = 20.0        # 同日 round trip 稅費 bp
COST_ON = 34.0        # 隔夜 round trip 稅費 bp（現貨賣出稅 30 + 手續費 3.4 + 期貨 ~0.6）
COST_ON_LONG = 54.0   # 多單留倉（融券）= 34 + 20
SAVE = COST_ON - COST_SD   # 當沖省下的稅差 14bp


# ---------------------------------------------------------------------------
# 1. 進場准入
# ---------------------------------------------------------------------------
def est_short(eff_u: float, ab: float, p_sd: float, L: float = -5.0,
              slot_free: bool = True, forced_tail_bp: float = -35.0) -> float:
    """空價差（多現貨/空期貨）的期望值，bp/筆。

    eff_u : 成交鎖定 basis 相對 anchor 的深度（用期貨可執行 bid 定價）
    ab    : 成交鎖定的絕對 basis（= anchor + eff_u）。到期時 basis 歸 0，
            所以 carry 分支的保底 = ab − COST_ON —— 這是結構知識，不是擬合。
    p_sd  : 同日出場機率（walk-forward 表，依進場時段/深度分桶）
    slot_free : 預期隔夜交割額度是否還有位子；沒有位子時 carry 分支改用
            強平估損（walk-forward 平均），避免收盤跳樓。
    """
    tail = (ab - COST_ON) if slot_free else forced_tail_bp
    return p_sd * (eff_u - L - COST_SD) + (1.0 - p_sd) * tail


def est_long(depth: float, p_sd: float, carry_ok: bool, p_recover: float,
             L: float = -5.0, tail_loss_bp: float = 30.0,
             forced_mean_bp: float = -25.0) -> float:
    """多價差（空現貨/多期貨）的期望值，bp/筆。

    多單「沒有」到期保底：逆價差常是除息造成（期貨不調整、現貨除息跳空，
    空現貨要賠付股利），所以 carry 分支只能用回升機率 × (depth − 5 − 54) 減尾損。
    carry_ok = 前日結構 basis > −50bp（除息指紋 gate）且有交割額度。
    """
    same_day = depth - (-L) - COST_SD            # depth − 5 − 20
    if carry_ok:
        tail = p_recover * (depth + L - COST_ON_LONG) - (1.0 - p_recover) * tail_loss_bp
    else:
        tail = forced_mean_bp
    return p_sd * same_day + (1.0 - p_sd) * tail


def admit(est_bp: float, lam_bp_per_day: float, expected_days: float = 0.6) -> bool:
    """准入：期望值必須高於容量的機會成本（影子價格 λ × 預期佔用天數）。"""
    return est_bp >= lam_bp_per_day * expected_days


# ---------------------------------------------------------------------------
# 2. 容量影子價格 λ（資料自己定價，不設常數）
# ---------------------------------------------------------------------------
def shadow_price(rejected_ev_twd_trailing: list[float], cap_twd: float,
                 clip_bp: float = 15.0) -> float:
    """λ = 前 N 日被容量拒絕的 fill 之估計 EV（TWD）均值 ÷ cap，換成 bp/日。
    20M 時常頂到 clip（容量稀缺）、50M 時自然變小 —— 同一公式兩種行為。"""
    if not rejected_ev_twd_trailing:
        return 0.0
    return float(np.clip(np.mean(rejected_ev_twd_trailing) / cap_twd * 1e4, 0.0, clip_bp))


# ---------------------------------------------------------------------------
# 3. 出場：續掛 vs cross（取代硬編碼的 13:00 積極平倉）
# ---------------------------------------------------------------------------
def should_cross(d_bp: float, p_fill: float, lam_bp: float, entry_today: bool) -> bool:
    """d_bp = taker 出場價距凍結目標的距離（多付的 bp）；p_fill = P_fill(τ, d)
    walk-forward 表（剩餘時間 × 距離分桶）。

    推導：V_hold = P×(cap−20) + (1−P)×(cap−34−λ)；V_cross = cap − d − 20
    → cross ⟺ d < (SAVE·[當日] + λ) × (1 − P_fill)
    尾盤 P_fill→0 時門檻自動放寬到 14+λ，積極平倉自己長出來。"""
    save = SAVE if entry_today else 0.0
    return d_bp < (save + lam_bp) * (1.0 - p_fill)


# ---------------------------------------------------------------------------
# 4. 出場目標 L* 的選擇
# ---------------------------------------------------------------------------
def choose_L(p_sd_by_L: dict[float, float], lam_bp: float) -> float:
    """E[pnl(L)] = effU − L − 34 + 14·P_sd(L)，effU 消掉 →
    L* = argmax_L [ −L + (SAVE + λ) · P_sd(L | 進場時段) ]。
    實測早盤 L=−10 仍有 57% 同日率，八成 entry 選 −10。"""
    return max(p_sd_by_L, key=lambda L: -L + (SAVE + lam_bp) * p_sd_by_L[L])


# ---------------------------------------------------------------------------
# 5. 多流容量分配：每容量日報酬（bpday）格閘門
# ---------------------------------------------------------------------------
EB_BUCKETS = (30.0, 50.0, 80.0)

def cell_of(stream: str, ab: float) -> str:
    """8 格：stream ∈ {S1 現貨maker腳, S2 期貨maker腳} × 絕對 basis 桶。"""
    return f'{stream}_{int(np.searchsorted(EB_BUCKETS, ab))}'


def cell_gate(pnl_sum_bp: float, days_sum: float, n: int,
              min_n: int = 30, threshold_bp_per_day: float = 8.0) -> bool:
    """格內歷史 (平均 bp ÷ 平均佔用天) ≥ 8 才掛。統計只在部位「解決日」入表
    （carry 單等真出場/到期那天），trailing 20 日窗，來源用不受容量影響的影子池。
    暖機期（n < min_n）放行。walk-forward 下實質 = S1 只做 ab≥50、S2 只做 ab≥80。
    這不是日內挑單：表在開盤前就定，機會按時序來一筆查一筆。"""
    if n < min_n:
        return True
    return (pnl_sum_bp / n) / max(days_sum / n, 0.15) >= threshold_bp_per_day


# ---------------------------------------------------------------------------
# 6. S2-inside 期貨 maker 掛單的 EV 容忍帶（訊息量 4.1/s → 0.12/s）
# ---------------------------------------------------------------------------
def quote_action(held_px: float | None, desired_px: float, fut_bid: float,
                 held_basis_minus_anchor: float, desired_ok: bool,
                 floor_bp: float = 20.0) -> str:
    """回傳 'new' / 'keep' / 'repeg' / 'cancel' / 'filled'。

    只看「我這張單的 EV 有沒有變」：A1 抖動、被 undercut 都不動（別人的報價
    不改變我成交時的損益）；現貨 ask 上移使鎖定 basis 跌破 U−5 地板才重掛；
    被穿價 = 成交。desired_px = A1 − 1 tick（隊列第一）。"""
    if held_px is None:
        return 'new' if desired_ok else 'keep'
    if held_px <= fut_bid:
        return 'filled'
    if held_basis_minus_anchor < floor_bp:
        return 'repeg' if desired_ok else 'cancel'
    return 'keep'


# ---------------------------------------------------------------------------
# 7. 到期結算（使用者指定慣例，= 凍結規格 C8 expiry_basis_zero_accounting）
# ---------------------------------------------------------------------------
def expiry_settlement_bp(ab: float, is_long: bool = False) -> float:
    """到期日兩腿用同一結算價標記、basis 歸 0：空單收 ab − 34。多單無此保底
    （除息污染），僅供對稱參考。"""
    return (-ab - COST_ON_LONG) if is_long else (ab - COST_ON)
