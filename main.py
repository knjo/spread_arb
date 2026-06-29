"""期現貨套利 低頻分析 — 產出層（純事實）：算當天每事件「客觀事實」，落地 CSV。

回測層只記事實：進出場四價、量、口數、時間、報價序號、是否收斂。
費用/本金/淨利/ROI/分析全搬 stats 層推（存了四價+量+口數就能推），
好處：改費率、改本金口徑(現金/融資)都不必重跑回測，只重跑 stats。

用法：
  python main.py -s 20260608                  # 單日，全市場
  python main.py -s 20260601 -e 20260608      # 多日（逐日各存一個 CSV）
  python main.py -s 20260608 --code 2303 --fut-code CCFF6   # 測試單檔
  python main.py -s 20260126 -e 20260610 --ticks            # 整年逐 tick 版

每天輸出單一事實表（不分 mode）：
  out/events_YYYYMMDD.csv        分K版
  out/events_YYYYMMDD_tick.csv   逐 tick 版
後續統計用 stats.py 讀（mode/費率是 stats 層的事），不必重撈 ticks。
"""
import argparse
import os
import time
from contextlib import contextmanager
from datetime import datetime, timedelta

import polars as pl


@contextmanager
def timed(label: str):
    """印出某段耗時。"""
    t0 = time.perf_counter()
    yield
    print(f"    [{label}] {time.perf_counter() - t0:.1f}s")

from sdk_core import TwTicks, TwMarketData
from mysql import StrategyMySQLLoader
from spread_arb.contract import near_month_code, settlement_date
from spread_arb.preprocess import (
    load_spot, load_futures, filter_near_month, to_minute_bars,
    filter_day_tradable, filter_price_limit, drop_price0_with_lots, TIME_COL,
)
from spread_arb.basic_info import (
    load_stock_basic, load_futures_basic, join_spot_basic, join_futures_basic,
)
from spread_arb.align import best_quotes, asof_join
from spread_arb.spread import calc_spreads, tag_events, SIDE_SELL, SIDE_BUY
from spread_arb.metrics import event_metrics, carryover_converge
# 純事實層：費用(cost.py)/本金(capital.py) 全搬 stats 層，main 不再 import

THRESHOLDS = [0.005, 0.01, 0.015, 0.02]
OUT_DIR = "out"

# 明細表要輸出的欄位（一列一事件）——【純事實層】
# 回測層只記客觀事實（四價+量+口數+時間+序號），費用/本金/淨利/ROI/分析全搬 stats 層推。
# 設計精神：存了進出場四價、量、口數，就能在 stats 推出所有衍生量；改費率不必重跑回測。
# 每個進場事件輸出一筆 row，出場固定走反向收斂 B（ret_buy≥0；A 同向早平倒貼已砍）。
EVENT_COLS = [
    "date", "ValueCode", "QuoteCode",   # QuoteCode=合約代碼(事件主體；小型/調整契約事後可過濾)
    "side", "threshold", "event_id",
    "is_first_entry",          # True=第一次進場(報價spread) / False=二次進場(event內成交spread驅動)
    "days_to_settle",          # 離到期日多久
    "contract_size",                     # 逐合約乘數(標準2000/小型100)
    "lots_fut", "lots_spot",             # 兩腳量(一次=兩腳掛量/二次=期成交量+現對手掛量)；供影響量化
    "potential_lots",                    # 部位口數(已股數配對)；部位金額由 stats 用進場四價推
    "carry_second_lots",       # 錨列帶：該事件二次進場原始量總和(未×參與率)；庫存滾錨時帶著加碼量
    # ── 進場事實 ──
    #   一次進場：第一筆報價四價/量；二次進場：期貨成交價+現貨對手價(其餘進場欄留空)。
    "first_time",
    "first_ret",               # 進場毛利率(一次=報價spread / 二次=成交spread)
    "entry_fut_bid", "entry_fut_ask", "entry_spot_bid", "entry_spot_ask",
    "entry_spot_time",         # 進場那筆現貨真實時間(E08；first_time−此=對到的現貨多舊)
    "entry_fut_chseq", "entry_spot_chseq",     # 進場報價序號(回 tick 定位)
    # ── 出場事實（達標那刻原始四價；未達標為空）──
    "converged",               # 日內是否達 B 出場(ret_buy≥0；true/false 旗標，免事後反推)
    "converge_time",           # 達標瞬間時間戳(未達標為空)；hold_secs/exit_ret 由 stats 推
    "exit_fut_bid", "exit_fut_ask", "exit_spot_bid", "exit_spot_ask",
    "exit_spot_time",          # 出場那筆現貨真實時間(E08；converge_time 是期貨時間)
    "exit_fut_bid_lots", "exit_fut_ask_lots",   # 出場掛量(E08；M03 流動性去重用)
    "exit_spot_bid_lots", "exit_spot_ask_lots",
    "exit_fut_chseq", "exit_spot_chseq",       # 出場報價序號(回 tick 定位)
    # ── 當天最後一筆 tick 四價（每合約自己的；給結帳層逐日浮虧評價用，非平倉點）──
    "close_fut_bid", "close_fut_ask", "close_spot_bid", "close_spot_ask",
    # ── 進場後固定秒數快照（趨勢用：兩腿各跑到哪；期/現各帶實際時間戳供過濾冷報價）──
    "s30_fut_bid", "s30_fut_ask", "s30_spot_bid", "s30_spot_ask",
    "s30_fut_time", "s30_spot_time",
    "s60_fut_bid", "s60_fut_ask", "s60_spot_bid", "s60_spot_ask",
    "s60_fut_time", "s60_spot_time",
    # ── 事件區間統計 ──
    "diverge_max", "diverge_mean",
    "signal_span_secs",        # 整段跨度：第一筆→最後一筆>=閥值(含中間震盪)
    "first_stretch_secs",      # 第一段持續：看到第一筆後那個報價站多久(打不打得到看這)
    "exit_stretch_secs",       # 出場版 first_stretch：收斂(ret_buy≥0)出現後持續可平多久
]                              #   (到第一次 ret_buy 又跌回<0；驗「收斂後價差會不會持續」、平不平得到)


# 只做價差賣：價差買整年僅 ~0.6 億、資金效率差，不具效益、源頭排除。
# 要恢復價差買改回 [SIDE_SELL, SIDE_BUY]。
# （本金 mode 已移到 stats 層，產出層不再分 mode。）
SIDES = [SIDE_SELL]


def arg_parser():
    p = argparse.ArgumentParser(description="期現貨套利 低頻分析（產出層）")
    p.add_argument("-s", "--start_date", type=str)
    p.add_argument("-e", "--end_date", type=str)
    p.add_argument("--code", type=str, help="測試用：現貨股票代號")
    p.add_argument("--fut-code", type=str, help="測試用：股期合約碼")
    p.add_argument("--ticks", action="store_true",
                   help="逐 tick 版（不壓1分K）；輸出檔名帶 _tick，與分K版分開存")
    p.add_argument("--inventory", action="store_true",
                   help="庫存模式：昨日沒收斂的部位讀進來當今天的部位、搭今天 tick 重判收斂，"
                        "一起寫進今天 fill；今天仍沒收斂的滾成明日庫存（到結算日強制平）")
    return p.parse_args()


def day_events_base(tw, tw_md, mysql, cal_date, code=None, fut_code=None,
                    use_ticks=False) -> pl.DataFrame:
    """單日：撈檔→過濾→(壓分K)→事件指標。重活只做一次，費用/capital 在外層各 mode 算。
    use_ticks=True 時跳過壓1分K，逐 tick 算（as-of backward 即上一刻現貨，不需 lag）。
    回傳含 side/threshold 的基礎事件表（無 fee/capital/roi）。"""
    ymd = cal_date.strftime("%Y%m%d")
    with timed("撈檔(NAS)"):
        try:
            spot = load_spot(tw, ymd, code=code)
            fut = filter_near_month(load_futures(tw, ymd, code=fut_code), ymd)
            stock_basic = load_stock_basic(tw_md, ymd)
            fut_basic = load_futures_basic(mysql, ymd)
        except Exception as e:
            # 例假日 / 無交易日：NAS 無當日資料 → 跳過，不中斷整個回測
            print(f"    （{ymd} 無資料，跳過：{type(e).__name__}）")
            return pl.DataFrame(), None
    if spot.height == 0 or fut.height == 0:
        return pl.DataFrame(), None

    with timed("過濾(可當沖/漲跌停/價0有量)"):
        # 多合約標的「不剔除」：事件鍵已改 QuoteCode，每個合約獨立算事件、互不干擾。
        # 標準/小型之分由 CSV 的 QuoteCode 欄事後過濾（注意：小型乘數=100股/口，
        # 目前金額統一用2000計，小型的金額在識別規則定案前會高估20倍）。
        # 現貨：join 基本面 → 可當沖(含排除處置股) → 漲跌停±9% → 價0有量
        spot = join_spot_basic(spot, stock_basic)
        spot = filter_day_tradable(spot)
        spot = filter_price_limit(spot)          # 用現貨 ref_price ±9%
        spot = drop_price0_with_lots(spot)
        # 期貨：join 基本面 → 漲跌停±9%(各用各的 ref_price) → 價0有量
        fut = join_futures_basic(fut, fut_basic)
        fut = filter_price_limit(fut)            # 用期貨 ref_price ±9%
        fut = drop_price0_with_lots(fut)

    with timed("對齊" if use_ticks else "壓分K+對齊"):
        if not use_ticks:
            # 低頻版：先各自壓 1 分 K（每分鐘最後一筆），再對齊算價差。
            # 以期貨為主體對齊「上一刻現貨」→ 現貨標籤 +1 分(lag)，
            # 使這一分鐘期貨對到前一分鐘現貨。
            spot = to_minute_bars(spot, lag=True)
            fut = to_minute_bars(fut)
        # ticks 版：不壓K，as-of backward 直接對「上一刻現貨」，不需 lag。
        # 算價差/標事件只需報價(asof 對齊期現)；成交列價=null 不影響事件判定。
        spreads = calc_spreads(asof_join(best_quotes(fut), spot))
        # 只留下游需要的欄位：tick 級全市場一天千萬列，原始40+欄每次複製數GB，
        # 瘦身到12欄可砍 ~4x 記憶體（OOM 對策之一）
        keep = ["ValueCode", "QuoteCode", "contract_size", TIME_COL, "ret_buy", "ret_sell",
                "fut_ask", "fut_bid", "fut_ask_lots", "fut_bid_lots",
                "spot_ask", "spot_bid", "spot_ask_lots", "spot_bid_lots",
                "spot_time",                 # 現貨原始時間戳(E08：量現貨 staleness)
                "fut_chseq", "spot_chseq"]   # 報價序號(回 tick 定位)
        spreads = spreads.select([c for c in keep if c in spreads.columns])
        # 二次進場：事件區間內、期貨「成交」那刻回頭看現貨對手價重算價差，仍可獲利就再進場。
        #   期貨成交流(驅動)：is_fill 列，帶 QuoteCode/ValueCode/時間/成交價量。
        #   現貨報價流(被貼)：is_quote 列，帶對手價 spot_ask(=AskPrice1)/掛量，供 asof 對齊。
        fut_fills = (fut.filter(pl.col("is_fill"))
                     .select([c for c in ("QuoteCode", "ValueCode", "contract_size",
                                          TIME_COL, "FillPrice", "FillLots")
                              if c in fut.columns])
                     if "is_fill" in fut.columns else None)
        spot_quotes = (spot.filter(pl.col("is_quote"))
                       .select(["ValueCode", TIME_COL,
                                pl.col("AskPrice1").alias("spot_ask"),
                                pl.col("AskLots1").alias("spot_ask_lots")])
                       if "is_quote" in spot.columns else None)

    with timed("事件+指標"):
        rows = []
        for side in SIDES:
            for thr in THRESHOLDS:
                tagged = tag_events(spreads, side, threshold=thr)
                m = event_metrics(tagged, side, ymd, fut_fills, spot_quotes, threshold=thr)
                if m.height == 0:
                    continue
                rows.append(m.with_columns([
                    pl.lit(ymd).alias("date"),
                    pl.lit(side).alias("side"),
                    pl.lit(thr).alias("threshold"),
                ]))
    base = pl.concat(rows) if rows else pl.DataFrame()
    return base, spreads   # spreads 給庫存層搭今天 tick 重判收斂用（庫存只在 day loop 接）


def select_events(base: pl.DataFrame) -> pl.DataFrame:
    """純事實層：只挑 EVENT_COLS 落地，不算費用/本金/ROI（那些 stats 層推）。

    費用依現金/融資、稅率等假設不同，全部後置到 stats 層，調參數不必重跑回測。
    這裡只負責把客觀事實（四價/量/口數/時間/序號）對齊輸出。"""
    # 防呆：EVENT_COLS 有欄位對不上就大聲報錯，不准靜默丟欄
    # （曾因 metrics 改名與此清單不同步，整年輸出靜默缺了時間欄、白跑一輪）
    missing = [c for c in EVENT_COLS if c not in base.columns]
    if missing:
        raise ValueError(f"輸出欄位缺失（EVENT_COLS 與 metrics 產出不同步）: {missing}")
    return base.select(EVENT_COLS)


def _merge_carryover(ev, prev_inv, spreads, ymd, suffix):
    """庫存模式：把昨日庫存(prev_inv)搭今天 spreads 重判收斂，併進今天的 fill(ev)。

    純補資訊、不分析：舊庫存當「今天的部位」，平掉/沒平的都當今天的 fill 一起寫。
    date 維持原始進場日（事實不改）；只把出場側更新成今天重判的結果。
    """
    if prev_inv.is_empty():
        return ev
    # 出場固定走 B 反向收斂（A 同向早平倒貼已砍）。
    parts = [ev]
    for side in SIDES:
        inv_s = prev_inv.filter(pl.col("side") == side)
        if inv_s.is_empty():
            continue
        closed, still = carryover_converge(inv_s, spreads, side)
        parts.extend([closed, still])
    merged = pl.concat([p for p in parts if p.height], how="diagonal")
    return merged.select(EVENT_COLS)


def main():
    args = arg_parser()
    start = datetime.strptime(args.start_date, "%Y%m%d") if args.start_date else datetime.today()
    end = datetime.strptime(args.end_date, "%Y%m%d") if args.end_date else start

    os.makedirs(OUT_DIR, exist_ok=True)
    tw = TwTicks()
    tw_md = TwMarketData()
    mysql = StrategyMySQLLoader()
    # 結算日假日順延（春節等）：注入交易日判斷，contract 內部有快取
    from spread_arb import contract as _contract
    _contract.set_trade_day_fn(mysql.is_trade_day)

    t_all = time.perf_counter()
    prev_inv = pl.DataFrame()   # 昨日滾來的庫存（沒收斂的 fill）；庫存模式才用
    cal = start
    while cal <= end:
        ymd = cal.strftime("%Y%m%d")
        # 先用日曆判斷交易日，非交易日直接跳過（比撈檔噴錯乾淨）
        try:
            if not mysql.is_trade_day(ymd):
                print(f"分析 {ymd}... 非交易日，跳過")
                cal += timedelta(days=1)
                continue
        except Exception:
            pass  # 日曆查詢失敗就照常往下，撈檔那層仍有 try/except 接住
        print(f"分析 {ymd}（近月 {near_month_code(ymd)}）{'[ticks版]' if args.ticks else ''}...")
        t_day = time.perf_counter()
        # 純事實層：撈檔/事件算一次，挑事實欄存「單一」CSV（不分 mode；費用/本金 stats 層推）
        base, spreads = day_events_base(tw, tw_md, mysql, cal, code=args.code,
                                        fut_code=args.fut_code, use_ticks=args.ticks)
        if base.height:
            suffix = "_tick" if args.ticks else ""
            ev = select_events(base)
            # 庫存模式：昨日沒收斂的部位讀進來當今天的部位，搭今天 spreads 重判收斂，
            #   平掉/沒平的都當「今天的 fill」一起寫進去（分析只看 fill，不看 inventory）。
            if args.inventory and spreads is not None:
                ev = _merge_carryover(ev, prev_inv, spreads, ymd, suffix)
            path = os.path.join(OUT_DIR, f"events_{ymd}{suffix}.csv")
            ev.write_csv(path)
            print(f"    → {ev.height} 個事件 → 1 檔事實表{suffix and '(tick)'}")
            # 今天仍沒收斂的（含滾進來又沒平的舊庫存）→ 明日庫存。
            # 但到結算日就強制平、不再滾（用各筆進場日 date 算結算日，已過今天=到期）。
            # ★只滾「第一次進場」未收斂的（錨部位）：二次進場是錨的加碼、出場跟錨走，
            #   不可獨立滾庫存（否則一天幾萬筆二次列跨日滾雪球，事件邊界爆掉）。
            #   加碼量的庫存承載屬「庫存滾池」議題（待另一輪），此處先只滾錨止血。
            if args.inventory:
                first_col = pl.col("is_first_entry") if "is_first_entry" in ev.columns \
                    else pl.lit(True)
                unconv = ev.filter(~pl.col("converged") & first_col)
                if unconv.height:
                    settle = [settlement_date(int(d)).strftime("%Y%m%d")
                              for d in unconv["date"].to_list()]
                    prev_inv = unconv.with_columns(pl.Series("_settle", settle)) \
                                     .filter(pl.col("_settle") > ymd).drop("_settle")
                else:
                    prev_inv = unconv
        else:
            print("    → 無事件")
        print(f"    當天總耗時 {time.perf_counter() - t_day:.1f}s")
        cal += timedelta(days=1)
    print(f"\n全部完成，總耗時 {time.perf_counter() - t_all:.1f}s")


if __name__ == "__main__":
    main()
