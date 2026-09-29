"""絕對價差套利 — 期貨腳改 maker 的「掛單穩定度」研究。

問題（2026-09-07）：
  現貨腳照舊以 taker 吃 A1 買進；期貨腳不吃 B1，而是依「期望價差 θ」反推期貨賣價
      P* = ceil_tick( spot_ask / (1 − θ) )        # (P* − spot_ask) / P* ≥ θ
  若 fut_bid1 < P* < fut_ask1，P* 落在期貨買賣價差裡面（A1−1、A1−2…），掛上去就是該價位
  隊列第一。核心要回答的不是「掛了會不會成交」，而是「掛了之後多久要被迫改單」——
  改單頻繁＝市況激烈＝套利執行變因多。

每個「掛單 episode」從掛出開始，到以下任一事件結束：
  fill_print : 期貨成交價 ≥ P（買方穿到我們這檔，隊列第一必先成交）且掛單已 ≥ LATENCY_MS
  fill_cross : 期貨 bid1 ≥ P（簿子直接穿過）且掛單已 ≥ LATENCY_MS
  spot_up    : 現貨 A1 上漲使新 P* > P（原掛價不再滿足 θ → 必須改單）
  undercut   : 期貨 ask1 < P（有人掛到我們下面，不再是隊列第一 → 是否追價由使用者定，這裡算改單）
  spot_gap   : 現貨報價失效（價 0 / 超過 STALE_MS 沒更新）
  session_end: 13:20 截止（右設限）
現貨 A1 下跌不強迫改單（原掛價超額達標，留著）；只記 n_spot_down 供「可選擇性改價」參考。
掛價政策（--policy）：
  pstar : 掛 P*（最低可接受價；成交機會最大、對現貨上漲零緩衝）
  a1m1  : 掛 max(P*, ask1−1 tick)（貼著 A1 下一檔；現貨可漲到 (ask1−1)(1−θ) 才需改單）
  a1m2  : 掛 max(P*, ask1−2 tick)
  三者都只在 bid1 < 掛價 < ask1 時掛（隊列第一）。
結束後若當下狀態仍可掛，立刻重新掛（新 episode）。每個 episode 記錄掛出當下可觀察的市況特徵，
供事後找「哪些情況下掛了不必常改單」。

同時對每個商品做「狀態時間佔比」：taker_now（P* ≤ bid1，taker 線的進場區）/ inside_k1 / inside_k2 /
inside_k3p（P* 在 A1 下 1/2/≥3 tick）/ join（P* == ask1）/ behind（P* > ask1）/ invalid。

用法：
  uv run python fut_maker_quote_stability.py run -s 20260706 --policy pstar  # 單日
  uv run python fut_maker_quote_stability.py run -s 20260302 -e 20260904 --skip-errors
  uv run python fut_maker_quote_stability.py summarize -s 20260302 -e 20260904
輸出：SSD2 stockfuture/fut_maker_quote/{date}_episodes_{policy}.parquet、{date}_state_time.parquet，
      summarize → fut_maker_quote/summary_{tag}.md + 各分層 csv。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from numba import njit

import arbitrage_analysis as aa
from data_paths import STOCKFUTURE_DIR, ensure_output_dir

OUT_DIR = STOCKFUTURE_DIR / "fut_maker_quote"
THRESHOLDS = [0.005, 0.0075, 0.01]
LATENCY_MS = 50          # 掛單到達交易所前的成交不算我們的
DEBOUNCE_MS = 0          # 改單觸發需持續多久才真的改（0=看到就改；--debounce-ms 可調）
STALE_MS = 60_000        # 現貨報價超過此毫秒沒更新 → 視為失效
ROLL_WINDOW = "60s"      # 掛出前 60 秒的市況活躍度
SESSION_CUTOFF_NS = (13 * 3600 + 20 * 60) * 1_000_000_000  # 13:20 台北（自午夜起 ns）

KIND_FUT_QUOTE = 0
KIND_FUT_FILL = 1
KIND_SPOT_QUOTE = 2

REASONS = ["fill_print", "fill_cross", "spot_up", "undercut", "spot_gap", "session_end",
           "fill_print_stale", "fill_cross_stale"]   # *_stale：改單條件已成立但尚未反應時被成交
STATES = ["invalid", "taker_now", "inside_k1", "inside_k2", "inside_k3p", "join", "behind"]


# ------------------------------------------------------------------ tick ladder
# 股票期貨升降單位（期交所）：<10 0.01｜10-50 0.05｜50-100 0.1｜100-500 0.5｜≥500 1.0。
# 與現貨差在 ≥1000 元：現貨 5 元、期貨 1 元（20260706 NAS 實測 ≥1000 檔 30% 價位非 5 的倍數、全為 1 的倍數）。
@njit(cache=True)
def _tick(p: float) -> float:
    if p < 10.0:
        return 0.01
    if p < 50.0:
        return 0.05
    if p < 100.0:
        return 0.1
    if p < 500.0:
        return 0.5
    return 1.0


@njit(cache=True)
def _spot_tick(p: float) -> float:
    if p < 1000.0:
        return _tick(p)
    return 5.0


@njit(cache=True)
def _ceil_tick(p: float) -> float:
    tk = _tick(p)
    n = np.ceil(p / tk - 1e-9)
    out = n * tk
    # 跨級距時用還原後價格的 tick 再檢查一次
    tk2 = _tick(out)
    if tk2 != tk:
        n = np.ceil(p / tk2 - 1e-9)
        out = n * tk2
    return out


@njit(cache=True)
def _ticks_between(hi: float, lo: float, ref: float) -> float:
    return (hi - lo) / _tick(ref)


# ------------------------------------------------------------------ 核心狀態機
@njit(cache=True)
def _simulate(t, kind, p1, p2, l1, l2, act1, act2, theta, latency_ns, stale_ns, cutoff_ns,
              max_ep, offset_ticks, debounce_ns, hold_on_undercut):
    """單一商品、單一 θ 的事件迴圈。

    t     : int64 ns（台北 naive，當天）
    kind  : 0 fut quote (p1=bid1,p2=ask1,l1=bid lots,l2=ask lots)
            1 fut fill  (p1=price, l1=lots)
            2 spot quote(p1=ask1,p2=bid1,l1=ask lots,l2=bid lots)
    act1/act2: 該筆事件所屬流的「前 60 秒變動次數」特徵
            fut quote: act1=ask1 變動次數, act2=成交筆數 ; spot quote: act1=ask1 變動次數, act2=bid1 變動次數
    回傳 episodes 陣列（每列固定欄位）與各狀態累積時間（ns）。
    """
    n = t.shape[0]
    ep = np.zeros((max_ep, 26), dtype=np.float64)
    n_ep = 0
    state_time = np.zeros(7, dtype=np.int64)

    # 市況
    fb = 0.0; fa = 0.0; fbl = 0.0; fal = 0.0
    sa = 0.0; sb = 0.0; sal = 0.0; sbl = 0.0
    t_spot = -1; t_fut = -1
    fut_act1 = 0.0; fut_act2 = 0.0; spot_act1 = 0.0; spot_act2 = 0.0
    fut_fills_since_spot = 0.0

    # 掛單
    posted = False
    P = 0.0; t_post = 0; sa_post = 0.0; fa_post = 0.0; fb_post = 0.0
    n_spot_down = 0.0; n_fut_ask_moves = 0.0; min_gap_ticks = 0.0; n_undercut_seen = 0.0
    ep_feat = np.zeros(26, dtype=np.float64)
    # 去彈跳：改單條件（spot_up/undercut）首次成立的時間與種類；條件消失即清除
    viol_t = -1; viol_reason = -1

    cur_state = 0
    t_state = t[0] if n > 0 else 0

    def _pstar(spot_ask):
        return _ceil_tick(spot_ask / (1.0 - theta))

    for i in range(n):
        ti = t[i]
        k = kind[i]
        # ---- 更新市況 ----
        if k == KIND_FUT_QUOTE:
            fb = p1[i]; fa = p2[i]; fbl = l1[i]; fal = l2[i]
            t_fut = ti; fut_act1 = act1[i]; fut_act2 = act2[i]
        elif k == KIND_SPOT_QUOTE:
            sa = p1[i]; sb = p2[i]; sal = l1[i]; sbl = l2[i]
            t_spot = ti; spot_act1 = act1[i]; spot_act2 = act2[i]
            fut_fills_since_spot = 0.0
        else:
            fut_fills_since_spot += 1.0

        spot_ok = (sa > 0.0) and (sb > 0.0) and (t_spot >= 0) and (ti - t_spot <= stale_ns)
        fut_ok = (fb > 0.0) and (fa > 0.0)
        past_cutoff = ti >= cutoff_ns

        # ---- 已掛單：判斷是否結束 ----
        if posted:
            reason = -1
            t_end = ti
            if past_cutoff:
                reason = 5
            elif k == KIND_FUT_FILL:
                if p1[i] >= P - 1e-9 and ti - t_post >= latency_ns:
                    reason = 6 if viol_t >= 0 else 0
            elif k == KIND_FUT_QUOTE:
                if not fut_ok:
                    pass  # 期貨簿暫時空掉：不動
                elif fb >= P - 1e-9 and ti - t_post >= latency_ns:
                    reason = 7 if viol_t >= 0 else 1
                else:
                    if fa != fa_post:
                        n_fut_ask_moves += 1.0
                    gap = _ticks_between(fa, P, P)
                    if gap < min_gap_ticks:
                        min_gap_ticks = gap
            else:  # spot quote
                if not spot_ok:
                    reason = 4
                elif sa < sa_post - 1e-9:
                    n_spot_down += 1.0
            # 改單條件（可去彈跳）：現貨 A1 漲到 P* > P，或期貨 ask1 掛到我們下面
            if reason < 0 and (k == KIND_FUT_QUOTE or k == KIND_SPOT_QUOTE) and spot_ok and fut_ok:
                cur_viol = -1
                if _pstar(sa) > P + 1e-9:
                    cur_viol = 2
                elif fa < P - 1e-9 and not hold_on_undercut:
                    cur_viol = 3
                if hold_on_undercut and fa < P - 1e-9:
                    n_undercut_seen += 1.0
                if cur_viol < 0:
                    viol_t = -1; viol_reason = -1
                else:
                    if viol_t < 0:
                        viol_t = ti; viol_reason = cur_viol
                    if ti - viol_t >= debounce_ns:
                        reason = viol_reason
                        t_end = viol_t + debounce_ns if debounce_ns > 0 else ti
            if reason >= 0:
                if n_ep < max_ep:
                    ep[n_ep, :] = ep_feat
                    ep[n_ep, 0] = float(t_post)
                    ep[n_ep, 1] = float(t_end)
                    ep[n_ep, 2] = float(reason)
                    ep[n_ep, 3] = n_spot_down
                    ep[n_ep, 4] = n_undercut_seen if hold_on_undercut else n_fut_ask_moves
                    ep[n_ep, 5] = min_gap_ticks
                    n_ep += 1
                posted = False
                viol_t = -1; viol_reason = -1

        # ---- 狀態分類（含時間累積）----
        if past_cutoff:
            st = 0
        elif not (spot_ok and fut_ok):
            st = 0
        else:
            ps = _pstar(sa)
            if ps <= fb + 1e-9:
                st = 1
            elif ps < fa - 1e-9:
                kt = int(np.round(_ticks_between(fa, ps, ps)))
                st = 2 if kt <= 1 else (3 if kt == 2 else 4)
            elif abs(ps - fa) <= 1e-9:
                st = 5
            else:
                st = 6
        if i > 0:
            state_time[cur_state] += ti - t_state
        cur_state = st
        t_state = ti

        # ---- 未掛單且可掛：掛出 ----
        if (not posted) and (st == 2 or st == 3 or st == 4):
            ps = _pstar(sa)
            if offset_ticks > 0:
                cand = fa - offset_ticks * _tick(fa)
                if cand >= ps - 1e-9 and cand > fb + 1e-9:
                    ps = cand
            posted = True
            P = ps; t_post = ti; sa_post = sa; fa_post = fa; fb_post = fb
            n_spot_down = 0.0; n_fut_ask_moves = 0.0; n_undercut_seen = 0.0
            viol_t = -1; viol_reason = -1
            min_gap_ticks = _ticks_between(fa, P, P)
            ep_feat[:] = 0.0
            ep_feat[6] = P
            ep_feat[7] = sa
            ep_feat[8] = sb
            ep_feat[9] = fa
            ep_feat[10] = fb
            ep_feat[11] = _ticks_between(fa, P, P)                 # k：A1 下幾 tick
            ep_feat[12] = _ticks_between(fa, fb, fa)               # 期貨 spread ticks
            ep_feat[13] = (sa - sb) / _spot_tick(sa)               # 現貨 spread ticks
            ep_feat[14] = sal
            ep_feat[15] = sbl
            ep_feat[16] = fal
            ep_feat[17] = fbl
            # 現貨 A1 還能漲幾個 tick 才會逼改單：P*(sa') > P ⟺ sa' > P(1−θ)
            ep_feat[18] = np.floor((P * (1.0 - theta) - sa) / _spot_tick(sa) + 1e-9)
            ep_feat[19] = (P - sa) / P - theta                     # 超額 margin（tick 湊整 + 政策抬價）
            ep_feat[20] = spot_act1
            ep_feat[21] = spot_act2
            ep_feat[22] = fut_act1
            ep_feat[23] = fut_act2
            ep_feat[24] = float(ti - t_spot)                       # 現貨報價齡 ns
            ep_feat[25] = float(k)                                 # 觸發掛單的事件種類

    # 收尾：仍掛著 → 右設限
    if posted and n_ep < max_ep and n > 0:
        ep[n_ep, :] = ep_feat
        ep[n_ep, 0] = float(t_post)
        ep[n_ep, 1] = float(t[n - 1])
        ep[n_ep, 2] = 5.0
        ep[n_ep, 3] = n_spot_down
        ep[n_ep, 4] = n_fut_ask_moves
        ep[n_ep, 5] = min_gap_ticks
        n_ep += 1
    return ep[:n_ep], state_time


EP_COLS = [
    "t_post_ns", "t_end_ns", "reason_id", "n_spot_down", "n_fut_ask_moves", "min_gap_ticks",
    "P", "spot_ask", "spot_bid", "fut_ask1", "fut_bid1",
    "k_ticks", "fut_spread_ticks", "spot_spread_ticks",
    "spot_ask_lots", "spot_bid_lots", "fut_ask1_lots", "fut_bid1_lots",
    "cushion_spot_ticks", "excess_margin",
    "spot_ask_chg_60s", "spot_bid_chg_60s", "fut_ask_chg_60s", "fut_fills_60s",
    "spot_age_ns", "trigger_kind",
]


# ------------------------------------------------------------------ 資料準備
def _load_day(date: str) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """回 (fut_quotes, fut_fills, spot_quotes, mapping)。時間皆台北 naive datetime[us]。"""
    tradable = aa._load_market_tradable(date)
    spot_ref = tradable.select(["ValueCode", "spot_ref_price"]).filter(
        pl.col("spot_ref_price").is_not_null() & (pl.col("spot_ref_price") > 0)
    )
    codes = sorted(set(tradable["ValueCode"].to_list()))

    path = aa._fetch_futures(date, None, False)
    fut = pl.read_parquet(path)
    if "contract_size" not in fut.columns or "fut_ref_price" not in fut.columns:
        fut = aa._join_futures_basic(fut, date)
    fut = aa._restore_prices(aa._normalize_time(fut), aa.FUT_SCALE)
    fut = fut.filter(pl.col("TrialMatch") == 0) if "TrialMatch" in fut.columns else fut
    fut = fut.filter(aa._session_time_expr(aa.TIME_COL))
    fut = fut.filter(
        pl.col("ValueCode").is_in(codes)
        & (pl.col("QuoteCode").str.slice(2, 1) == "F")
        & (pl.col("QuoteCode").str.slice(-2) == aa.near_month_code(date))
        & pl.col("contract_size").is_not_null() & aa._is_standard_contract_expr()
    )
    fut_quotes = (
        fut.filter(aa._has_book_expr())
           .with_columns(aa._within_ref_limit_expr("fut_ref_price", ["BidPrice1", "AskPrice1"])
                         .alias("ok"))
           .filter(pl.col("ok"))
           .select(["ValueCode", "QuoteCode", aa.TIME_COL,
                    pl.col("BidPrice1").alias("fut_bid1"), pl.col("AskPrice1").alias("fut_ask1"),
                    pl.col("BidLots1").cast(pl.Float64).alias("fut_bid1_lots"),
                    pl.col("AskLots1").cast(pl.Float64).alias("fut_ask1_lots")])
           .sort(["ValueCode", aa.TIME_COL])
    )
    fut_fills = (
        fut.filter((pl.col("FillPrice") > 0) & (pl.col("FillLots") > 0))
           .select(["ValueCode", "QuoteCode", aa.TIME_COL,
                    pl.col("FillPrice").alias("fill_px"), pl.col("FillLots").cast(pl.Float64).alias("fill_lots")])
           .sort(["ValueCode", aa.TIME_COL])
    )
    mapping = fut_quotes.select("ValueCode", "QuoteCode").unique().sort("ValueCode")
    value_codes = mapping["ValueCode"].to_list()

    spot = aa._prepare_spot(date, value_codes, False, spot_ref)
    spot_quotes = spot.select([
        "ValueCode", aa.TIME_COL, "spot_ask", "spot_bid",
        pl.col("spot_ask_lots").cast(pl.Float64), pl.col("spot_bid_lots").cast(pl.Float64),
    ]).sort(["ValueCode", aa.TIME_COL])
    return fut_quotes, fut_fills, spot_quotes, mapping


def _rolling_count(df: pl.DataFrame, flag: pl.Expr, name: str) -> pl.DataFrame:
    return df.with_columns(
        flag.cast(pl.Int32).rolling_sum_by(aa.TIME_COL, window_size=ROLL_WINDOW)
            .over("ValueCode").fill_null(0).cast(pl.Float64).alias(name)
    )


def _build_stream(fut_quotes: pl.DataFrame, fut_fills: pl.DataFrame,
                  spot_quotes: pl.DataFrame) -> pl.DataFrame:
    fq = _rolling_count(fut_quotes, pl.col("fut_ask1") != pl.col("fut_ask1").shift(1).over("ValueCode"),
                        "act1")
    # 期貨成交筆數（60s）：把成交流的計數 as-of 貼到報價流
    ff = fut_fills.with_columns(pl.lit(1).alias("one"))
    ff = _rolling_count(ff, pl.col("one") == 1, "fills_60s")
    fq = fq.join_asof(
        ff.select(["ValueCode", aa.TIME_COL, "fills_60s"]).sort(["ValueCode", aa.TIME_COL]),
        on=aa.TIME_COL, by="ValueCode", strategy="backward", tolerance=ROLL_WINDOW,
    ).with_columns(pl.col("fills_60s").fill_null(0.0).alias("act2"))
    fq = fq.select([
        "ValueCode", aa.TIME_COL, pl.lit(KIND_FUT_QUOTE, dtype=pl.Int8).alias("kind"),
        pl.col("fut_bid1").alias("p1"), pl.col("fut_ask1").alias("p2"),
        pl.col("fut_bid1_lots").alias("l1"), pl.col("fut_ask1_lots").alias("l2"),
        "act1", "act2",
    ])
    fl = fut_fills.select([
        "ValueCode", aa.TIME_COL, pl.lit(KIND_FUT_FILL, dtype=pl.Int8).alias("kind"),
        pl.col("fill_px").alias("p1"), pl.lit(0.0).alias("p2"),
        pl.col("fill_lots").alias("l1"), pl.lit(0.0).alias("l2"),
        pl.lit(0.0).alias("act1"), pl.lit(0.0).alias("act2"),
    ])
    sq = _rolling_count(spot_quotes, pl.col("spot_ask") != pl.col("spot_ask").shift(1).over("ValueCode"),
                        "act1")
    sq = _rolling_count(sq, pl.col("spot_bid") != pl.col("spot_bid").shift(1).over("ValueCode"), "act2")
    sq = sq.select([
        "ValueCode", aa.TIME_COL, pl.lit(KIND_SPOT_QUOTE, dtype=pl.Int8).alias("kind"),
        pl.col("spot_ask").alias("p1"), pl.col("spot_bid").alias("p2"),
        pl.col("spot_ask_lots").alias("l1"), pl.col("spot_bid_lots").alias("l2"),
        "act1", "act2",
    ])
    stream = pl.concat([fq, fl, sq], how="vertical").sort(["ValueCode", aa.TIME_COL, "kind"])
    return stream


POLICY_OFFSET = {"pstar": 0, "a1m1": 1, "a1m2": 2}


def run_day(date: str, thresholds: list[float], policy: str = "pstar",
            debounce_ms: int = DEBOUNCE_MS, hold_on_undercut: bool = False) -> None:
    """hold_on_undercut=True：被人掛到下面不改單、留在原價（只有現貨上漲才改），檔名加 _hold。"""
    offset = POLICY_OFFSET[policy]
    tag = f"{policy}_db{debounce_ms}" + ("_hold" if hold_on_undercut else "")
    t0 = time.perf_counter()
    fut_quotes, fut_fills, spot_quotes, mapping = _load_day(date)
    stream = _build_stream(fut_quotes, fut_fills, spot_quotes)
    print(f"{date}: stream {stream.height:,} rows, {mapping.height} products "
          f"(fut quotes {fut_quotes.height:,}, fills {fut_fills.height:,}, spot {spot_quotes.height:,}) "
          f"[{time.perf_counter() - t0:.1f}s]")

    day_ns = np.datetime64(f"{date[:4]}-{date[4:6]}-{date[6:]}", "ns").astype(np.int64)
    t_all = (stream[aa.TIME_COL].cast(pl.Datetime("ns")).to_numpy().astype("datetime64[ns]")
             .astype(np.int64) - day_ns)
    kind = stream["kind"].to_numpy().astype(np.int8)
    p1 = stream["p1"].to_numpy().astype(np.float64)
    p2 = stream["p2"].to_numpy().astype(np.float64)
    l1 = stream["l1"].to_numpy().astype(np.float64)
    l2 = stream["l2"].to_numpy().astype(np.float64)
    a1 = stream["act1"].to_numpy().astype(np.float64)
    a2 = stream["act2"].to_numpy().astype(np.float64)
    codes = stream["ValueCode"].to_numpy()
    bounds = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1], True])

    ep_frames, st_rows = [], []
    for theta in thresholds:
        for bi in range(len(bounds) - 1):
            s, e = bounds[bi], bounds[bi + 1]
            eps, st = _simulate(t_all[s:e], kind[s:e], p1[s:e], p2[s:e], l1[s:e], l2[s:e],
                                a1[s:e], a2[s:e], theta,
                                LATENCY_MS * 1_000_000, STALE_MS * 1_000_000,
                                SESSION_CUTOFF_NS, 200_000, offset,
                                debounce_ms * 1_000_000, hold_on_undercut)
            code = str(codes[s])
            st_rows.append({"date": date, "ValueCode": code, "threshold": theta,
                            **{f"secs_{n}": float(v) / 1e9 for n, v in zip(STATES, st)}})
            if eps.shape[0]:
                df = pl.DataFrame(eps, schema=EP_COLS).with_columns([
                    pl.lit(date).alias("date"), pl.lit(code).alias("ValueCode"),
                    pl.lit(theta).alias("threshold"), pl.lit(policy).alias("policy"),
                    pl.lit(debounce_ms).alias("debounce_ms"),
                ])
                ep_frames.append(df)

    ensure_output_dir(OUT_DIR)
    if ep_frames:
        episodes = pl.concat(ep_frames).with_columns([
            (pl.col("t_post_ns").cast(pl.Int64) + day_ns).cast(pl.Datetime("ns")).cast(pl.Datetime("us")).alias("t_post"),
            (pl.col("t_end_ns").cast(pl.Int64) + day_ns).cast(pl.Datetime("ns")).cast(pl.Datetime("us")).alias("t_end"),
            ((pl.col("t_end_ns") - pl.col("t_post_ns")) / 1e9).alias("dwell_secs"),
            pl.col("reason_id").cast(pl.Int8).replace_strict(
                {i: r for i, r in enumerate(REASONS)}, return_dtype=pl.String).alias("reason"),
            (pl.col("spot_age_ns") / 1e6).alias("spot_age_ms"),
            pl.col("trigger_kind").cast(pl.Int8),
        ]).drop(["t_post_ns", "t_end_ns", "reason_id", "spot_age_ns"])
        episodes = episodes.join(mapping, on="ValueCode", how="left")
        episodes.write_parquet(OUT_DIR / f"{date}_episodes_{tag}.parquet")
    else:
        episodes = pl.DataFrame()
    state_time = pl.DataFrame(st_rows)
    state_time.write_parquet(OUT_DIR / f"{date}_state_time.parquet")

    for theta in thresholds:
        sub = episodes.filter(pl.col("threshold") == theta) if episodes.height else episodes
        n = sub.height
        if n == 0:
            print(f"  θ={theta:.2%}: no episodes")
            continue
        posted_h = sub["dwell_secs"].sum() / 3600
        rc = sub["reason"].value_counts().sort("reason")
        req = sub.filter(pl.col("reason").is_in(["spot_up", "undercut"])).height
        fills = sub.filter(pl.col("reason").str.starts_with("fill")).height
        print(f"  θ={theta:.2%}: {n:,} posts | posted {posted_h:.1f} h | requote {req/posted_h:.1f}/h | "
              f"fill {fills/n:.1%} | dwell p50 {sub['dwell_secs'].median():.1f}s | "
              + ", ".join(f"{r}={c}" for r, c in rc.iter_rows()))
    print(f"{date} [{tag}]: done [{time.perf_counter() - t0:.1f}s] -> {OUT_DIR}")


# ------------------------------------------------------------------ 彙總
def _md(df: pl.DataFrame, digits: int = 3) -> str:
    """polars → markdown 表（不依賴 tabulate）。"""
    cols = df.columns
    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.{digits}f}" if abs(v) < 1e6 else f"{v:.3e}"
        return str(v)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in df.iter_rows():
        out.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(out)


def _bucket(col: str, edges: list[float], labels: list[str]) -> pl.Expr:
    expr = pl.when(pl.col(col) < edges[0]).then(pl.lit(labels[0]))
    for i in range(1, len(edges)):
        expr = expr.when(pl.col(col) < edges[i]).then(pl.lit(labels[i]))
    return expr.otherwise(pl.lit(labels[-1])).alias(f"{col}_bin")


BUCKETS = {
    "k_ticks": ([1.5, 2.5, 4.5], ["1", "2", "3-4", "5+"]),
    "fut_spread_ticks": ([2.5, 3.5, 5.5, 10.5], ["≤2", "3", "4-5", "6-10", "11+"]),
    "cushion_spot_ticks": ([0.5, 1.5, 2.5], ["0", "1", "2", "3+"]),
    "spot_spread_ticks": ([1.5, 2.5, 4.5], ["1", "2", "3-4", "5+"]),
    "spot_ask_lots": ([5, 20, 100], ["<5", "5-19", "20-99", "100+"]),
    "spot_ask_chg_60s": ([1, 3, 6, 12], ["0", "1-2", "3-5", "6-11", "12+"]),
    "fut_ask_chg_60s": ([1, 3, 6, 12], ["0", "1-2", "3-5", "6-11", "12+"]),
    "fut_fills_60s": ([1, 3, 10], ["0", "1-2", "3-9", "10+"]),
    "tod_hour": ([10, 11, 12, 13], ["09", "10", "11", "12", "13"]),
}


def _agg(df: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    return (
        df.group_by(by).agg([
            pl.len().alias("posts"),
            (pl.col("dwell_secs").sum() / 3600).alias("posted_hours"),
            pl.col("reason").is_in(["spot_up", "undercut"]).sum().alias("requotes"),
            (pl.col("reason") == "spot_up").sum().alias("spot_up"),
            (pl.col("reason") == "undercut").sum().alias("undercut"),
            pl.col("reason").str.starts_with("fill").sum().alias("fills"),
            pl.col("reason").str.ends_with("_stale").sum().alias("stale_fills"),
            pl.col("dwell_secs").median().alias("dwell_p50"),
            pl.col("dwell_secs").quantile(0.25).alias("dwell_p25"),
            (pl.col("dwell_secs") >= 10).mean().alias("surv_10s"),
            (pl.col("dwell_secs") >= 30).mean().alias("surv_30s"),
            (pl.col("dwell_secs") >= 60).mean().alias("surv_60s"),
            ((pl.col("reason").is_in(["spot_up", "undercut"])) & (pl.col("dwell_secs") < 10)).mean()
                .alias("requote_within_10s"),
        ])
        .with_columns([
            (pl.col("requotes") / pl.col("posted_hours")).alias("requote_per_hour"),
            (pl.col("fills") / pl.col("posts")).alias("fill_rate"),
        ])
        .sort(by)
    )


def summarize(start: str, end: str | None, policy: str = "pstar",
              debounce_ms: int = DEBOUNCE_MS, hold_on_undercut: bool = False) -> None:
    tag_in = f"{policy}_db{debounce_ms}" + ("_hold" if hold_on_undercut else "")
    files = sorted(OUT_DIR.glob(f"*_episodes_{tag_in}.parquet"))
    dates = [p.name[:8] for p in files if start <= p.name[:8] <= (end or start)]
    if not dates:
        raise SystemExit(f"no episodes in {OUT_DIR} for {start}~{end} {tag_in}")
    ep = pl.concat([pl.read_parquet(OUT_DIR / f"{d}_episodes_{tag_in}.parquet") for d in dates],
                   how="diagonal_relaxed")
    st = pl.concat([pl.read_parquet(OUT_DIR / f"{d}_state_time.parquet") for d in dates])
    ep = ep.with_columns((pl.col("t_post").dt.hour() + pl.col("t_post").dt.minute() / 60).alias("tod_hour"))
    for col, (edges, labels) in BUCKETS.items():
        ep = ep.with_columns(_bucket(col, edges, labels))
    tag = f"{dates[0]}_{dates[-1]}_{tag_in}"
    lines = [f"# 期貨 maker 掛單穩定度 — {dates[0]}~{dates[-1]}（{len(dates)} 天，policy={policy}，"
             f"debounce={debounce_ms}ms）", ""]
    lines += ["口徑：P* = ceil_tick(spot_ask/(1−θ))，bid1 < P* < ask1 時掛在 P*（該價位隊列第一）。",
              "改單 = spot_up（現貨 A1 漲到原掛價不再滿足 θ）+ undercut（期貨 ask1 掛到我們下面）。",
              "fill = 成交價 ≥ P 的成交列（fill_print）或 bid1 ≥ P（fill_cross），掛單需已滿 50 ms。", ""]

    # 狀態時間佔比
    st_sum = st.group_by("threshold").agg([pl.col(f"secs_{s}").sum() for s in STATES]).sort("threshold")
    tot = sum(st_sum[f"secs_{s}"] for s in STATES)
    st_pct = st_sum.with_columns([(pl.col(f"secs_{s}") / tot * 100).round(2).alias(s) for s in STATES]) \
                   .select(["threshold", *STATES])
    st_pct.write_csv(OUT_DIR / f"summary_state_time_{tag}.csv")
    lines += ["## 各 θ 狀態時間佔比（%，商品×交易時段加總）", "", st_pct.pipe(_md), ""]

    overall = _agg(ep, ["threshold"])
    overall.write_csv(OUT_DIR / f"summary_overall_{tag}.csv")
    lines += ["## 整體", "", overall.pipe(_md), ""]

    # 掛單時間有多少落在「長命」episode：改單雖多，但集中在少數激烈時段
    share = ep.group_by("threshold").agg([
        (pl.col("dwell_secs").filter(pl.col("dwell_secs") >= 30).sum() / pl.col("dwell_secs").sum()).alias("time_share_ge30s"),
        (pl.col("dwell_secs").filter(pl.col("dwell_secs") >= 60).sum() / pl.col("dwell_secs").sum()).alias("time_share_ge60s"),
        (pl.col("dwell_secs").filter(pl.col("dwell_secs") >= 300).sum() / pl.col("dwell_secs").sum()).alias("time_share_ge300s"),
        (pl.col("dwell_secs") < 1).mean().alias("post_share_lt1s"),
    ]).sort("threshold")
    share.write_csv(OUT_DIR / f"summary_time_share_{tag}.csv")
    lines += ["## 掛單時間佔比（落在存活 ≥30/60/300 秒的 episode）與 <1 秒即改單的掛單比例", "",
              share.pipe(_md), ""]

    # 產品×小時 的改單率分佈（只看該小時掛單 ≥10 分鐘者）
    ph = (ep.with_columns(pl.col("t_post").dt.hour().alias("hr"))
            .group_by(["threshold", "date", "ValueCode", "hr"]).agg([
                pl.len().alias("posts"),
                (pl.col("dwell_secs").sum() / 60).alias("posted_min"),
                pl.col("reason").is_in(["spot_up", "undercut"]).sum().alias("requotes"),
                pl.col("reason").str.starts_with("fill").sum().alias("fills"),
            ])
            .with_columns((pl.col("requotes") / pl.col("posted_min") * 60).alias("rq_per_h"))
            .filter(pl.col("posted_min") >= 10))
    ph.write_csv(OUT_DIR / f"summary_product_hour_{tag}.csv")
    phq = ph.group_by("threshold").agg([
        pl.len().alias("product_hours"),
        *[pl.col("rq_per_h").quantile(q).alias(f"rq_q{int(q*100)}") for q in (0.1, 0.25, 0.5, 0.75, 0.9)],
        (pl.col("rq_per_h") <= 6).mean().alias("share_le6_per_h"),
        (pl.col("rq_per_h") <= 12).mean().alias("share_le12_per_h"),
        ((pl.col("rq_per_h") <= 6) & (pl.col("fills") >= 1)).mean().alias("share_le6_and_filled"),
    ]).sort("threshold")
    lines += ["## 產品×小時 改單率分佈（每小時掛單 ≥10 分鐘的 product-hour；rq=改單次數/掛單小時）", "",
              phq.pipe(_md), ""]

    byday = _agg(ep, ["threshold", "date"])
    byday.write_csv(OUT_DIR / f"summary_by_date_{tag}.csv")
    lines += ["## 逐日（regime 差異）", "",
              byday.select(["threshold", "date", "posts", "posted_hours", "requote_per_hour", "fill_rate",
                            "dwell_p50", "surv_60s"]).pipe(_md), ""]

    for col in BUCKETS:
        a = _agg(ep, ["threshold", f"{col}_bin"])
        a.write_csv(OUT_DIR / f"summary_by_{col}_{tag}.csv")
        lines += [f"## 依 {col} 分層", "", a.pipe(_md), ""]

    # 兩維：現貨 A1 活躍度 × 期貨 A1 活躍度（掛出前 60 秒各自變動次數）
    act = _agg(ep, ["threshold", "spot_ask_chg_60s_bin", "fut_ask_chg_60s_bin"])
    act.write_csv(OUT_DIR / f"summary_by_spotact_x_futact_{tag}.csv")
    lines += ["## 現貨 A1 60 秒變動次數 × 期貨 A1 60 秒變動次數", "", act.pipe(_md), ""]

    # 兩維：期貨 spread × 現貨 A1 活躍度
    two = _agg(ep, ["threshold", "fut_spread_ticks_bin", "spot_ask_chg_60s_bin"])
    two.write_csv(OUT_DIR / f"summary_by_futspread_x_spotact_{tag}.csv")
    lines += ["## 期貨 spread × 現貨 A1 60 秒變動次數", "", two.pipe(_md), ""]

    # 商品層：改單率最低/最高
    prod = _agg(ep, ["threshold", "ValueCode"]).filter(pl.col("posts") >= 20)
    prod.write_csv(OUT_DIR / f"summary_by_product_{tag}.csv")
    lines += ["## 商品層（posts ≥ 20；requote_per_hour 最低 15 檔，θ=0.5%）", "",
              prod.filter(pl.col("threshold") == THRESHOLDS[0]).sort("requote_per_hour").head(15)
                  .pipe(_md), ""]

    md = OUT_DIR / f"summary_{tag}.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:60]))
    print(f"... -> {md}")


# ------------------------------------------------------------------ CLI
def main() -> None:
    p = argparse.ArgumentParser(description="期貨 maker 掛單穩定度")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("-s", "--start-date", required=True)
    r.add_argument("-e", "--end-date")
    r.add_argument("--thresholds", default=",".join(str(x) for x in THRESHOLDS))
    r.add_argument("--skip-errors", action="store_true")
    r.add_argument("--policy", default="pstar", help="逗號分隔：pstar,a1m1,a1m2")
    r.add_argument("--debounce-ms", default=str(DEBOUNCE_MS), help="逗號分隔，如 0,500")
    r.add_argument("--hold-on-undercut", action="store_true", help="被插隊不改單，只在現貨上漲時改")
    s = sub.add_parser("summarize")
    s.add_argument("-s", "--start-date", required=True)
    s.add_argument("-e", "--end-date")
    s.add_argument("--policy", default="pstar")
    s.add_argument("--debounce-ms", type=int, default=DEBOUNCE_MS)
    s.add_argument("--hold-on-undercut", action="store_true")
    args = p.parse_args()

    if args.cmd == "run":
        thr = [float(x) for x in args.thresholds.split(",")]
        for date in aa._date_range(args.start_date, args.end_date):
            if not aa.market_data_path(date).exists():
                continue
            try:
                for pol in args.policy.split(","):
                    for db in args.debounce_ms.split(","):
                        run_day(date, thr, pol.strip(), int(db), args.hold_on_undercut)
            except Exception as exc:
                if args.skip_errors:
                    print(f"{date}: skipped ({type(exc).__name__}: {exc})")
                else:
                    raise
    else:
        summarize(args.start_date, args.end_date, args.policy, args.debounce_ms, args.hold_on_undercut)


if __name__ == "__main__":
    main()
