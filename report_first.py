"""期現貨套利 低頻分析 — 只算第一次進場(is_first_entry)的最低估算報表。

不碰原始 ticks（不撈 NAS），只讀產出層落地的事件明細 CSV。
用法：
  python stats.py                 # 讀 out/ 下全部 events_*.csv
  python stats.py -s 20260601 -e 20260608   # 只讀日期區間

輸出：價差賣/買各一張 pivot（列=結算日距離分組、欄=閥值）。
要做別的統計，直接在這裡對 events DataFrame group_by 即可。
"""
import argparse
import glob
import os

import polars as pl

from spread_arb.spread import SIDE_SELL, SIDE_BUY
from spread_arb.report import summarize
from spread_arb.cost import net_return
from spread_arb.capital import add_capital
from spread_arb.metrics import SPOT_SHARES_PER_LOT   # 現貨1張=1000股（出場流動性換算用）

OUT_DIR = "out"               # event CSV 所在（產出層落地）
STATS_DIR = "out/stats"       # 統計匯總落地處（與 event CSV 分開，不混在一起）
# 只統計價差賣（價差買整年僅 ~0.6 億、不具效益，產出層已源頭排除）。
# 要恢復價差買改回 [SIDE_SELL, SIDE_BUY]。
SIDES = [SIDE_SELL]


def arg_parser():
    p = argparse.ArgumentParser(description="期現貨套利 統計層")
    p.add_argument("-s", "--start_date", type=str)
    p.add_argument("-e", "--end_date", type=str)
    p.add_argument("--ticks", action="store_true",
                   help="讀 ticks 版結果(events_*_tick.csv)；圖檔名亦帶 _tick")
    return p.parse_args()


def load_events(start=None, end=None, ticks=False) -> pl.DataFrame:
    """讀 out/events_YYYYMMDD[_tick].csv（純事實表，不分 mode），可選日期區間（含端點）。
    mode（本金口徑）是 stats 層 enrich 時才套用，與讀檔無關；ticks=True 讀逐 tick 版。"""
    suffix = "_tick" if ticks else ""
    files = sorted(glob.glob(os.path.join(OUT_DIR, f"events_*{suffix}.csv")))
    # 事實版必有欄：舊版（含 capital/缺四價的 events_*_m{mode}*.csv）缺此欄 → 跳過不混。
    #   is_first_entry：砍A+二次進場(2026/6/15)後新增；缺此欄＝改版前舊 CSV，混入會被當
    #   二次進場誤丟(diagonal concat 填 False)→ 一併當舊 schema 跳過、強制重跑。
    FACT_MARKERS = ("entry_fut_bid", "is_first_entry")
    frames, skipped = [], 0
    for f in files:
        name = os.path.basename(f)
        # 非 tick 模式時，glob events_*.csv 會連 events_*_tick.csv 一起抓到 → 排除
        if not ticks and name.endswith("_tick.csv"):
            continue
        # 檔名 events_YYYYMMDD[_tick].csv → 取 events_ 後 8 碼日期
        ymd = name[len("events_"):len("events_") + 8]
        if start and ymd < start:
            continue
        if end and ymd > end:
            continue
        df = pl.read_csv(f)
        # 舊 schema 防呆：缺進場四價或 is_first_entry＝改版前 CSV，跳過不混（否則假數字）
        if any(m not in df.columns for m in FACT_MARKERS):
            skipped += 1
            continue
        # 跨檔 schema 統一（某天 exit/快照欄全空 → read_csv 推型不一，concat 後算術炸）：
        #   數值欄一律 Float64、時間欄一律 String（sort 用 ISO 字典序＝時間序，安全）。
        casts = []
        for c, dt in df.schema.items():
            if c in ("QuoteCode", "ValueCode", "side"):
                continue
            if c.endswith("_time"):
                if dt != pl.String:
                    casts.append(pl.col(c).cast(pl.String))
            elif dt == pl.String:
                casts.append(pl.col(c).cast(pl.Float64, strict=False))
        if casts:
            df = df.with_columns(casts)
        frames.append(df)
    if skipped:
        print(f"⚠️ 略過 {skipped} 個舊版 CSV（缺 {' / '.join(FACT_MARKERS)}）；"
              f"砍A+二次進場改版後 schema 已變，請刪掉舊 events_*.csv 並用新 main.py 重跑全年")
    if not frames:
        return pl.DataFrame()
    # 註：舊版事件鍵用 ValueCode，同檔標準+小型契約兩條報價流互打爆假事件，曾用
    #   multi_contract_stocks.txt 黑名單整檔剔除補救。事件鍵改 QuoteCode 後（main.py:123、
    #   spread.py）混流從源頭消失，黑名單死碼已移除（L1）。
    # diagonal_relaxed：欄位數不同也能疊，且放寬 dtype（某天 exit 欄全空被推成 String、
    #   別天是 Float64 時自動取相容型，不炸 SchemaError）。
    return pl.concat(frames, how="diagonal_relaxed")


def enrich(events: pl.DataFrame, mode: str = "2") -> pl.DataFrame:
    """【事實 → 衍生】從純事實 CSV 推出所有下游分析需要的衍生欄。

    本金口徑固定保守(mode='2'：現貨全額+期貨保證金40%)，最低估算用。
    產出層只存事實（四價/量/口數/序號/時間/converged）。這裡按「現行」費用參數與
    本金口徑(mode) 一次補回所有衍生欄：
      exit_ret        出場反邊價差(taker平倉成本) = 出場四價算的反邊 ret（事實推，非存）
      hold_secs       持有秒數 = converge_time − first_time（事實推，非存）
      potential_value 部位金額 = potential_lots × contract_size × entry_fut_bid(賣期成交價)
      fee_rate/net_ret/gross_pnl/fee_amount/potential_pnl  ← cost.net_return（含 exit_ret 推出場成本）
      capital/roi/roi_annual                                ← capital.add_capital（依 mode 保證金率）
    好處：調稅率/利率/本金口徑都只重跑此函式，不必重撈 ticks 重跑回測。
    費用/本金參數的單一真相在 cost.py / capital.py（產出層與此處共用同一函式）。
    """
    # (0) 從事實推 exit_ret / hold_secs（產出層只存原始四價/時間，不存這兩個推得的量）。
    #   exit_ret = 出場那刻反邊 taker 價差：價差賣反邊=賣現買期=(exit_spot_bid−exit_fut_ask)/exit_fut_ask；
    #     價差買對稱=(exit_fut_bid−exit_spot_ask)/exit_fut_ask。未達標(出場四價null)→null。
    sell_exit = (pl.col("exit_spot_bid") - pl.col("exit_fut_ask")) / pl.col("exit_fut_ask")
    buy_exit = (pl.col("exit_fut_bid") - pl.col("exit_spot_ask")) / pl.col("exit_fut_ask")
    events = events.with_columns([
        pl.when(pl.col("exit_fut_ask").is_null()).then(None)
          .when(pl.col("side") == SIDE_SELL).then(sell_exit)
          .otherwise(buy_exit).alias("exit_ret"),
        (pl.col("converge_time").str.to_datetime(strict=False)
         - pl.col("first_time").str.to_datetime(strict=False))
            .dt.total_seconds(fractional=True).alias("hold_secs"),  # 浮點秒(到微秒)；未達標 null
    ])
    # L3 防呆：contract_size 為 null 會讓 potential_value→null、污染下游淨利且無聲。
    #   事實層每筆都該有乘數；缺=join 失準，fail-loud 而非靜默算錯。
    n_cs_null = events.filter(pl.col("contract_size").is_null()).height
    if n_cs_null:
        raise ValueError(
            f"enrich: contract_size 有 {n_cs_null} 筆為 null（乘數缺失＝合約 join 失準），"
            f"會讓 potential_value/淨利無聲變 null。請檢查事實 CSV 的 contract_size 欄。")
    # (1) 部位金額（事實推）：口數 × 逐合約乘數 × 進場賣期成交價(entry_fut_bid)
    #     entry_fut_bid 為價差賣的成交腳；價差買對稱用 entry_fut_ask（目前只跑賣）。
    fut_px = pl.when(pl.col("side") == SIDE_SELL) \
               .then(pl.col("entry_fut_bid")).otherwise(pl.col("entry_fut_ask"))
    events = events.with_columns(
        (pl.col("potential_lots") * pl.col("contract_size") * fut_px)
        .alias("potential_value")
    )
    # L2 防呆：enrich 只處理 SIDES 內的 side，迴圈外的 side 會被靜默丟棄（買向事件無聲蒸發）。
    #   先檢查有無未知 side，有就 raise（恢復價差買時忘了同步 SIDES 會立刻被抓到）。
    known = set(SIDES)
    unknown = [s for s in events["side"].unique().to_list() if s not in known]
    if unknown:
        raise ValueError(
            f"enrich: 事件含未知 side {unknown}，不在 SIDES={SIDES}（會被靜默丟棄）。"
            f"恢復價差買時請同步 stats.SIDES 與 main.SIDES。")
    # (2) 費用/淨利（按 side 分流，cost.net_return 從 first_ret + exit_ret 推來回毛利）
    parts = []
    for side in SIDES:
        sub = events.filter(pl.col("side") == side)
        if sub.height:
            sub = net_return(sub, side, mode=mode)      # 補 fee_rate/net_ret/potential_pnl...
            parts.append(add_capital(sub, side, mode=mode))  # 補 capital/roi/roi_annual
    return pl.concat(parts) if parts else events


def _sidetag(side: str) -> str:
    """方向 → 檔名用短碼。"""
    return "sell" if side == SIDE_SELL else "buy"


# ── 逐日結算（核心，已實/未實）──────────────────────────────────────────
#   回測 --inventory 已逐日把 rolling 處理進每天 CSV → 分析層逐日讀、當天當今天算，
#   不碰跨日。損益只算「當天收斂平掉」那批(每筆收斂一次、跨日加總不重複、不灌水)；
#   留倉(未收斂)用當日 close 四價 mark 出未實現。first 不放大、pool 放大(+carry×參與率)。
PARTICIPATION = 0.1   # 二次進場放大參與率（pool 版用）


def daily_settle(amplify: bool = False, ticks: bool = True,
                 start=None, end=None) -> pl.DataFrame:
    """逐日讀每天 CSV，算每天每閾值的「已實現 + 未實現」。回每天每閾值一列。

    amplify：False=只 first(potential_lots 原值)；True=pool(lots=floor(potential_lots
      + carry_second_lots×PARTICIPATION))。
    已實現＝當天收斂平掉那批 potential_pnl；未實現＝留倉用 close mark(兩腿合計損益 + 期貨腿浮虧)。
    """
    suffix = "_tick" if ticks else ""
    files = sorted(glob.glob(os.path.join(OUT_DIR, f"events_*{suffix}.csv")))
    rows = []
    for f in files:
        name = os.path.basename(f)
        if not ticks and name.endswith("_tick.csv"):
            continue
        ymd = name[len("events_"):len("events_") + 8]
        if (start and ymd < start) or (end and ymd > end):
            continue
        df = pl.read_csv(f)
        if any(c not in df.columns for c in ("is_first_entry", "carry_second_lots",
                                             "entry_fut_bid")):
            continue
        num = [c for c in df.columns if df.schema[c] == pl.String
               and c not in ("QuoteCode", "ValueCode", "side") and not c.endswith("_time")]
        df = df.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in num])
        ev = df.filter(pl.col("is_first_entry"))
        if amplify:
            ev = ev.with_columns(
                (pl.col("potential_lots")
                 + pl.col("carry_second_lots").fill_null(0) * PARTICIPATION)
                .floor().cast(pl.Int64).alias("potential_lots"))
        ev = ev.filter(pl.col("potential_lots") > 0)
        if ev.height == 0:
            continue
        ev = enrich(ev)
        # 期貨腿名目(mark 用)：口數×乘數×進場賣期價
        fut_notional = (pl.col("potential_lots") * pl.col("contract_size")
                        * pl.col("entry_fut_bid"))
        for side in SIDES:
            sev = ev.filter(pl.col("side") == side)
            for thr in sorted(sev["threshold"].unique().to_list()):
                t = sev.filter(pl.col("threshold") == thr)
                conv = t.filter(pl.col("converged"))
                realized = float(conv["potential_pnl"].sum()) if conv.height else 0.0
                inv = t.filter(~pl.col("converged")
                               & (pl.col("close_fut_ask") > 0) & (pl.col("close_spot_bid") > 0)
                               & (pl.col("entry_fut_bid") > 0) & (pl.col("entry_spot_ask") > 0))
                if inv.height:
                    inv = inv.with_columns([
                        (((pl.col("entry_fut_bid") - pl.col("close_fut_ask"))
                          / pl.col("entry_fut_bid")) * fut_notional).alias("_fut"),
                        (((pl.col("close_spot_bid") - pl.col("entry_spot_ask"))
                          / pl.col("entry_spot_ask")) * fut_notional).alias("_spot"),
                    ])
                    unreal_fut = float(inv["_fut"].sum())
                    unreal_pnl = float((inv["_fut"] + inv["_spot"]).sum())
                else:
                    unreal_fut = unreal_pnl = 0.0
                rows.append({
                    "date": ymd, "side": side, "threshold": thr,
                    "收斂筆數": conv.height, "留倉筆數": inv.height,
                    "已實現": realized, "未實現損益": unreal_pnl,
                    "未實現_期貨腿浮虧": unreal_fut,
                })
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def daily_summary(daily: pl.DataFrame) -> pl.DataFrame:
    """逐日結算 → 整年彙總（每 side×threshold 一列）。
    已實現跨日加總(不重複)；未實現取峰值(浮虧最深那天=帳面最差/要補最多保證金)。"""
    return (daily.group_by("side", "threshold").agg([
        pl.col("收斂筆數").sum().alias("整年收斂筆數"),
        pl.col("已實現").sum().alias("整年已實現"),
        pl.col("未實現損益").min().alias("未實現損益_最深"),       # 最負=帳面最差那天
        pl.col("未實現_期貨腿浮虧").min().alias("期貨腿浮虧_峰值"),  # 最負=要補最多保證金(MAE)
    ]).sort("side", "threshold"))


def plot_daily(daily: pl.DataFrame, tag: str) -> None:
    """每閾值畫逐日曲線：累積已實現(實線) + 當日未實現損益(虛線)。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime
        plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False
    except ImportError:
        print("（未裝 matplotlib，跳過畫圖）")
        return
    for side in SIDES:
        sd = daily.filter(pl.col("side") == side)
        if sd.height == 0:
            continue
        fig, ax = plt.subplots(figsize=(12, 6))
        for thr in sorted(sd["threshold"].unique().to_list()):
            d = sd.filter(pl.col("threshold") == thr).sort("date")
            xs = [datetime.strptime(x, "%Y%m%d") for x in d["date"].to_list()]
            cum = [v / 1e8 for v in d["已實現"].cum_sum().to_list()]   # 累積已實現
            un = [v / 1e8 for v in d["未實現損益"].to_list()]           # 當日未實現
            line, = ax.plot(xs, cum, marker=".", label=f"{thr:.1%} 累積已實現")
            ax.plot(xs, un, linestyle="--", alpha=0.5, color=line.get_color())
        ax.set_title(f"{side} 逐日損益（實線=累積已實現／虛線=當日未實現 close mark）")
        ax.set_ylabel("損益（億元）")
        ax.set_xlabel("日期")
        ax.axhline(0, color="gray", lw=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        os.makedirs(STATS_DIR, exist_ok=True)
        path = os.path.join(STATS_DIR, f"chart_daily_{_sidetag(side)}_{tag}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"  → 逐日損益曲線已存：{path}")


def yearly_summary(events: pl.DataFrame, min_hold=None) -> pl.DataFrame:
    """整年加總彙總（各方向×閥值一列）：筆數、收斂率、總淨利、總部位、平均淨利率。

    這是「跨結算距離加總」的整年視角（pivot 是按結算距離分列、看不到整年總和）。
    各閥值獨立、不可跨閥值相加（0.5% 的事件含 1%）。

    min_hold：執行延遲分層 —— 只計 first_stretch_secs >= 此秒數的事件，
      代表「進場窗口站夠久 = N 秒內打得到」才算數。
      （修正 E01：原誤濾 hold_secs[進場→收斂的持有時間]，那是出場側、與「打不打得到」無關；
        「打得到」看進場窗口站多久 = first_stretch_secs。METHODOLOGY 步驟7 明示。）
      first_stretch_secs=null(進場後價差未明確跌破，約0.3%)→ 算不出進場窗、顯式排除(非靜默通過)。
      None=全量。

    口徑欄（go/no-go 必看）：
      總淨利            ＝全量（含留倉，靠「出場成本=0」上界撐起，是上界）。
      總淨利_converged下界＝只認當天打掉的（當沖），留倉部分計 0（M02 下界）。
      converged占比     ＝下界/上界，越小代表 headline 越依賴留倉=0 假設。真實值落在 [下界↔上界]。
      平均年化ROI       ＝逐筆等權 mean(roi_annual)，被近結算事件 ÷clip(1) 放大、偏高，僅參考。
      資金加權年化ROI    ＝Σ淨利÷Σ(本金×天數/365)，資金×時間加權，較誠實（M04）。
    """
    ev = events
    if min_hold is not None:
        ev = ev.filter(pl.col("first_stretch_secs") >= min_hold)
    # M02 下界：converged-only 淨利（只認當天打掉的，留倉=0 上界的另一端）。
    conv_pnl = pl.when(pl.col("converged")).then(pl.col("potential_pnl")).otherwise(0.0)
    # M04 資金加權年化：Σ淨利 ÷ Σ(動用本金×天數/365)，按資金×時間加權，
    #   取代逐筆等權 mean(roi_annual)（後者被近結算事件 ÷clip(1) 放大、失真）。
    cap_days = pl.col("capital") * pl.col("days_to_settle").clip(lower_bound=1) / 365.0
    return (ev.group_by("side", "threshold")
            .agg([
                pl.len().alias("筆數"),
                pl.col("converged").mean().alias("收斂率"),
                pl.col("potential_pnl").sum().alias("總淨利"),
                conv_pnl.sum().alias("總淨利_converged下界"),    # M02：不靠留倉=0 上界
                pl.col("net_ret").mean().alias("平均淨利率"),     # 淨利/部位
                pl.col("roi").mean().alias("平均ROI"),            # 淨利/動用本金(保證金)
                pl.col("roi_annual").mean().alias("平均年化ROI"),  # ROI×365/天數(逐筆等權，會失真)
                (pl.col("potential_pnl").sum() / cap_days.sum())
                    .alias("資金加權年化ROI"),                    # M04：資金×時間加權，較誠實
            ])
            .with_columns(
                (pl.col("總淨利_converged下界") / pl.col("總淨利")).alias("converged占比"))
            .sort("side", "threshold"))


def pivot_one_side(events: pl.DataFrame, side: str) -> pl.DataFrame:
    """單方向：列=結算日距離分組、欄=閥值，每格彙整指標。"""
    sub = events.filter(pl.col("side") == side)
    out = None
    for thr in sorted(sub["threshold"].unique().to_list()):
        s = summarize(sub.filter(pl.col("threshold") == thr))
        s = s.rename({c: f"{thr:.1%}_{c}" for c in s.columns if c != "settle_bucket"})
        out = s if out is None else out.join(s, on="settle_bucket", how="full", coalesce=True)
    return out.sort("settle_bucket") if out is not None else pl.DataFrame()


def capital_need(events: pl.DataFrame) -> pl.DataFrame:
    """要準備多少本金（峰值口徑）：每檔當天取單波最大本金，再全市場加總。

    同檔多波時間不重疊、資金可週轉 → 同檔只取單波最大（不加總）；
    跨檔同時在場 → 加總（保守上限）。按 date×side×threshold 算每日本金需求，
    再對多天彙整（平均/最大），同時對照「各波本金直接加總」看高估多少。
    """
    # 每檔當天單波最大本金
    per_stock = (events.group_by("date", "side", "threshold", "ValueCode")
                 .agg(pl.col("capital").max().alias("stock_peak_cap")))
    # 全市場加總 = 當天本金需求
    daily = (per_stock.group_by("date", "side", "threshold")
             .agg(pl.col("stock_peak_cap").sum().alias("cap_need")))
    # 對照：各波本金直接加總（舊高估口徑）+ 當天淨獲利
    naive = (events.group_by("date", "side", "threshold")
             .agg([pl.col("capital").sum().alias("cap_naive"),
                   pl.col("potential_pnl").sum().alias("pnl")]))
    daily = daily.join(naive, on=["date", "side", "threshold"])
    # 跨多天彙整：平均每日本金需求、平均每日淨獲利、用峰值本金算的 ROI
    return (daily.group_by("side", "threshold")
            .agg([pl.col("cap_need").mean().alias("平均每日本金需求"),
                  pl.col("cap_naive").mean().alias("平均每日_各波加總"),
                  pl.col("pnl").mean().alias("平均每日淨獲利")])
            .with_columns(
                (pl.col("平均每日淨獲利") / pl.col("平均每日本金需求")).alias("ROI_峰值口徑"))
            .sort("side", "threshold"))


def daytrade_vs_overnight(events: pl.DataFrame) -> pl.DataFrame:
    """A+B：當沖(converged=true) vs 留倉(converged=false) 拆分。

    A 拆分：各自的本金需求(峰值口徑：每檔單波最大→全市場加總)、淨獲利、筆數。
    B 留倉成本：留倉部位假設抱到結算日(路B)，留倉天數=days_to_settle，
       額外留倉利息 = 部位 × 利息率 × days_to_settle/365（粗估，與 cost.py 一致）。
    持有時間分兩欄、單位不混（E09 修正）：當沖列只看「當沖平均持有分鐘」
       (=hold_secs/60，~分鐘級)，留倉列只看「留倉平均距結算天數」(=days_to_settle)；
       原一律 days_to_settle.mean() 會讓當沖列誤顯 ~15 天（實際 ~28 分鐘）。
    回傳每 方向×閥值×(當沖/留倉) 一列。
    """
    ev = events.with_columns(
        pl.when(pl.col("converged")).then(pl.lit("當沖"))
          .otherwise(pl.lit("留倉")).alias("型態")
    )
    # 本金需求(峰值)：每檔單波最大 → 同型態內全市場加總（先到日，再跨日平均）
    per_stock = (ev.group_by("date", "side", "threshold", "型態", "ValueCode")
                 .agg(pl.col("capital").max().alias("peak")))
    daily_cap = (per_stock.group_by("date", "side", "threshold", "型態")
                 .agg(pl.col("peak").sum().alias("cap_need")))
    cap = (daily_cap.group_by("side", "threshold", "型態")
           .agg(pl.col("cap_need").mean().alias("平均每日本金需求")))
    # 淨獲利、筆數、平均持有時間（E09 修正：當沖/留倉持有口徑不同，分兩欄、不混單位）
    #   當沖(converged)：真實持有 = hold_secs(進場→收斂)，~分鐘級；換算分鐘。
    #   留倉(未收斂)   ：hold_secs 為 null，假設抱到結算 → 用 days_to_settle(~天級)。
    #   原一律 days_to_settle.mean() 會讓當沖列誤顯 ~15 天（實際 ~28 分鐘），故拆分。
    base = (ev.group_by("side", "threshold", "型態")
            .agg([pl.len().alias("筆數"),
                  pl.col("potential_pnl").sum().alias("淨獲利合計"),
                  # 當沖列才算分鐘（留倉 hold_secs=null 自然不入均）；
                  # 留倉列才算距結算天數（當沖用 when 遮成 null，避免該欄誤顯天數）
                  (pl.col("hold_secs") / 60).mean().alias("當沖平均持有分鐘"),
                  pl.when(~pl.col("converged")).then(pl.col("days_to_settle"))
                    .mean().alias("留倉平均距結算天數")]))
    return (base.join(cap, on=["side", "threshold", "型態"])
            .sort("side", "threshold", "型態"))


def exit_liquidity(events: pl.DataFrame) -> pl.DataFrame:
    """各閥值出場流動性覆蓋：每筆收斂事件「自己」出場那刻，對手檔掛量吃不吃得下自己的部位。

    重點（使用者拍板的口徑）：
      - **每事件自己跟自己比**，無「跨事件搶流動性/先進先出」——因為策略一波只進一次、
        未收斂不會再進，同檔同時最多一個未平倉部位，不會有自己人互搶。(交易員 M03 的去重前提不成立)
      - **閥值各自獨立**：0.5% 群含後來上到 1% 的事件，但「打 0.5% vs 只打 1%」就是各看各的覆蓋率。
      - 價差賣平倉＝買回期(吃 exit_fut_ask 掛量) + 賣現(吃 exit_spot_bid 掛量)，**兩腳都要出得掉**。
        期貨腳需求 = potential_lots(口)；現貨腳需求 = potential_lots × contract_size(股)。
        掛量：exit_fut_ask_lots(口)、exit_spot_bid_lots(張)×1000=股。
      - 出得掉比例 fill_rate = min(期覆蓋, 現覆蓋, 1)；fill_rate>=1 = 該筆「全出得完」。

    只看 converged 事件（留倉沒有日內出場 tick、不適用）。需 events 含 exit_*_lots（E08 補存，
    舊 CSV 無此欄→回傳空表並提示重跑）。回傳每 side×threshold 一列：
      可評估筆數、全出得完筆數比例、部位金額加權出得掉比例，
      + 絕對金額：收斂淨利(帳面)、打折後收斂淨利(×fill_rate 實得)、可出場部位金額。
    注意：這只折「當沖(收斂)」那塊；留倉那段沒出場掛量、不在此（仍是 =0 上界，待 M02）。
    """
    need = ["exit_fut_ask_lots", "exit_spot_bid_lots"]
    if any(c not in events.columns for c in need):
        print("（exit_liquidity 跳過：CSV 無 exit_*_lots 欄，請 main.py --ticks 重跑補出場量 E08）")
        return pl.DataFrame()
    ev = events.filter(
        pl.col("converged") & (pl.col("potential_lots") > 0)
        & pl.col("exit_fut_ask_lots").is_not_null()      # 出場掛量有值才可評估
        & pl.col("exit_spot_bid_lots").is_not_null())
    if ev.height == 0:
        return pl.DataFrame()
    # 兩腳各自覆蓋率（掛量 ÷ 需求，封頂 1），取較緊那邊 = 這單實際出得掉的比例
    fut_cov = pl.col("exit_fut_ask_lots") / pl.col("potential_lots")
    spot_need_shares = pl.col("potential_lots") * pl.col("contract_size")
    spot_cov = (pl.col("exit_spot_bid_lots") * SPOT_SHARES_PER_LOT) / spot_need_shares
    ev = ev.with_columns(
        pl.min_horizontal(pl.min_horizontal(fut_cov, spot_cov), pl.lit(1.0))
          .alias("fill_rate"))
    return (ev.group_by("side", "threshold")
            .agg([
                pl.len().alias("可評估筆數"),
                # 筆數比例：幾成事件「整單出得完」(fill_rate 達 1)
                (pl.col("fill_rate") >= 1.0).mean().alias("全出得完_筆數比"),
                # 金額比例：部位金額加權的「實際出得掉」比例 = Σ(部位×fill_rate) / Σ部位
                ((pl.col("potential_value") * pl.col("fill_rate")).sum()
                    / pl.col("potential_value").sum()).alias("出得掉_金額比"),
                # 絕對金額（給規模感，非只比例）：
                pl.col("potential_pnl").sum().alias("收斂淨利"),                  # 帳面(吃滿假設)
                (pl.col("potential_pnl") * pl.col("fill_rate")).sum()
                    .alias("打折後收斂淨利"),                                      # 流動性打折後實得
                (pl.col("potential_value") * pl.col("fill_rate")).sum()
                    .alias("可出場部位金額"),                                      # 出場那刻實際吃得掉的部位
            ])
            .sort("side", "threshold"))


def plot_cum_pnl(events: pl.DataFrame, side: str, mode: str, min_hold=None):
    """累積總損益曲線：各閥值的每日淨獲利逐日累加，存 PNG。

    min_hold：只看 first_stretch_secs >= 此秒數的事件（進場窗站夠久=打得到；E01 修正）。
    None=全量。檔名帶 _stretch{N}s 後綴。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime
        plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False
    except ImportError:
        print("（未裝 matplotlib，跳過畫圖）")
        return

    sub_all = events.filter(pl.col("side") == side)
    tag, note = "", "全量"
    if min_hold is not None:
        sub_all = sub_all.filter(pl.col("first_stretch_secs") >= min_hold)
        tag, note = f"_stretch{min_hold}s", f"進場窗>={min_hold}秒"
    thrs = sorted(sub_all["threshold"].unique().to_list())
    fig, ax = plt.subplots(figsize=(12, 6))
    for thr in thrs:
        daily = (sub_all.filter(pl.col("threshold") == thr)
                 .group_by("date").agg(pl.col("potential_pnl").sum())
                 .sort("date"))
        if daily.height == 0:
            continue
        xs = [datetime.strptime(str(d), "%Y%m%d") for d in daily["date"].to_list()]
        cum, ys = 0.0, []
        for v in daily["potential_pnl"].to_list():
            cum += v
            ys.append(cum / 1e7)  # 千萬
        ax.plot(xs, ys, marker=".", label=f"{thr:.1%}（累積 {cum/1e7:,.1f}千萬）")

    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title(f"{side} 累積總損益（mode {mode}，{note}，整年逐日累加）")
    ax.set_ylabel("累積淨損益（千萬元）")
    ax.set_xlabel("日期")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    os.makedirs(STATS_DIR, exist_ok=True)
    path = os.path.join(STATS_DIR, f"chart_pnl_{_sidetag(side)}_m{mode}{tag}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  → 累積損益圖已存：{path}")


def plot_inventory_curves(events: pl.DataFrame, side: str, mode: str, min_hold=None):
    """各閥值的每日資金需求曲線 PNG。min_hold：進場窗(first_stretch)分層(同 plot_cum_pnl 口徑)。"""
    try:
        import matplotlib
        matplotlib.use("Agg")  # 無視窗環境
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime
        # 中文字型（Windows 微軟正黑體），避免標題/軸標變方塊
        plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False
    except ImportError:
        print("（未裝 matplotlib，跳過畫圖：pip install matplotlib）")
        return

    tag, note = "", "全量"
    if min_hold is not None:
        tag, note = f"_stretch{min_hold}s", f"進場窗>={min_hold}秒"

    thrs = sorted(events["threshold"].unique().to_list())
    fig, ax = plt.subplots(figsize=(12, 6))
    for thr in thrs:
        curve, peak, pday, avg = overnight_inventory(events, side, thr, min_hold=min_hold)
        if curve is None:
            continue
        xs = [datetime.strptime(d, "%Y%m%d") for d in curve["date"].to_list()]
        ys = [v / 1e8 for v in curve["總需求"].to_list()]  # 億
        line, = ax.plot(xs, ys, marker=".",
                        label=f"{thr:.1%} 總需求（峰值 {peak/1e8:.1f}億）")
        # 同色虛線畫留倉部分，看得出留倉 vs 當沖組成
        yo = [v / 1e8 for v in curve["在場留倉"].to_list()]
        ax.plot(xs, yo, linestyle="--", alpha=0.5, color=line.get_color())

    ax.set_title(f"{side} 每日資金需求＝在場留倉(虛線)＋當日了結（mode {mode}，{note}）"
                 f"\n※僅在「全留倉抱到結算」政策下成立（M05）")
    ax.set_ylabel("資金需求（億元）")
    ax.set_xlabel("日期")
    # M05：曲線 ~90% 是留倉堆積，是「全部未收斂都抱到結算」這個政策的人造物，
    #   非市場固有需求。提前出場政策（早平/部分平倉）會大幅壓低此曲線。
    ax.text(0.01, 0.97,
            "註：留倉(虛線)假設全部抱到結算日；改提前出場政策此曲線會大降",
            transform=ax.transAxes, fontsize=8, va="top", alpha=0.7)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    os.makedirs(STATS_DIR, exist_ok=True)
    path = os.path.join(STATS_DIR, f"chart_capital_{_sidetag(side)}_m{mode}{tag}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  → 線圖已存：{path}")


def overnight_inventory(events: pl.DataFrame, side: str, threshold: float,
                        min_hold=None):
    """C：整年資金需求模擬 = 在場留倉（跨日累積）+ 當日當沖佔用。

    min_hold：只計 first_stretch_secs >= 此秒數的事件（進場窗站夠久=打得到；E01 修正）。
      留倉的 first_stretch 仍有值(進場窗有站)，會正常被分層、不再無條件全納入。
    留倉=當日未收斂(converged=false)，假設抱到結算日(路B)，
      佔用區間 = [date, date+days_to_settle]，逐日加總涵蓋當天者。
    當沖=當日收斂(converged=true)，只佔當天（每檔取單波最大→全市場加總）。
    當天總資金需求 = 在場留倉 + 當日當沖。峰值/平均看總需求。
    回傳 (每日曲線 DataFrame[date,留倉,當沖,總需求], 總峰值, 峰值日, 總平均)。
    """
    from datetime import datetime, timedelta

    sub = events.filter(
        (pl.col("side") == side) & (pl.col("threshold") == threshold))
    if min_hold is not None:
        sub = sub.filter(pl.col("first_stretch_secs") >= min_hold)
    if sub.height == 0:
        return None, 0.0, None, 0.0

    # --- 留倉：佔用區間累積 ---
    on = sub.filter(~pl.col("converged"))
    occ = {}  # (進場日, 結算日, 檔) -> max capital（同檔同日取單波最大）
    for date, d2s, vc, cap in on.select(
            "date", "days_to_settle", "ValueCode", "capital").rows():
        di = datetime.strptime(str(date), "%Y%m%d")
        key = (di, di + timedelta(days=int(d2s)), vc)
        occ[key] = max(occ.get(key, 0.0), cap)

    # --- 當沖：每日（每檔單波最大 → 加總），只佔當天 ---
    dt = (sub.filter(pl.col("converged"))
          .group_by("date", "ValueCode").agg(pl.col("capital").max())
          .group_by("date").agg(pl.col("capital").sum().alias("daytrade")))
    dt_map = {str(d): v for d, v in dt.rows()}

    all_dates = sorted({datetime.strptime(str(d), "%Y%m%d")
                        for d in events["date"].unique().to_list()})
    daily = []
    for day in all_dates:
        # 留倉＝今天之後還會繼續抱的（結算日當天到期者不算留倉——當天必了結、不再過夜）
        o = sum(cap for (di, dset, vc), cap in occ.items() if di <= day < dset)
        # 當日到期結算的舊留倉：錢白天仍壓著，歸入「當日了結」
        e = sum(cap for (di, dset, vc), cap in occ.items() if dset == day)
        t = dt_map.get(day.strftime("%Y%m%d"), 0.0)
        daily.append((day.strftime("%Y%m%d"), o, t + e, o + t + e))

    curve = pl.DataFrame(daily, schema=["date", "在場留倉", "當日了結", "總需求"],
                         orient="row")
    peak = curve["總需求"].max()
    peak_day = curve.filter(pl.col("總需求") == peak)["date"][0]
    avg = curve["總需求"].mean()
    return curve, peak, peak_day, avg


def run_daily(amplify: bool, ticks: bool, tag: str, start=None, end=None) -> None:
    """★核心：逐日結算（已實/未實）→ 整年彙總表 + 逐日曲線圖。
    amplify=False 只first(最低估)；True 含二次放大(pool)。"""
    daily = daily_settle(amplify=amplify, ticks=ticks, start=start, end=end)
    if daily.height == 0:
        print("（無逐日結算資料：先跑 main.py --ticks --inventory）")
        return
    name = "含二次放大" if amplify else "只first(最低估)"
    print(f"===== ★逐日結算彙總（{name}；損益只算當天收斂、留倉用 close mark）=====")
    print("（整年已實現＝逐日當天收斂加總，每筆只算一次、不重複、不含留倉=0上界灌水；")
    print("  未實現＝留倉部位當日 close mark：損益取最深那天、期貨腿浮虧峰值＝MAE 要補多少保證金）")
    summ = daily_summary(daily)
    with pl.Config(tbl_cols=-1, tbl_width_chars=240, tbl_rows=-1):
        print(summ.with_columns([
            (pl.col("整年已實現") / 1e8).round(3).alias("整年已實現_億"),
            (pl.col("未實現損益_最深") / 1e8).round(3).alias("未實現最深_億"),
            (pl.col("期貨腿浮虧_峰值") / 1e8).round(3).alias("保證金峰值_億★MAE"),
        ]).select("side", "threshold", "整年收斂筆數",
                  "整年已實現_億", "未實現最深_億", "保證金峰值_億★MAE"))
    os.makedirs(STATS_DIR, exist_ok=True)
    daily.write_csv(os.path.join(STATS_DIR, f"daily_settle{tag}.csv"))
    plot_daily(daily, tag.lstrip("_"))
    print()


def main():
    args = arg_parser()
    ticks = args.ticks
    tag = "_tick" if ticks else ""
    # ★先跑逐日結算（核心、正確口徑）：整年彙總 + 曲線圖
    run_daily(amplify=False, ticks=ticks, tag=tag,
              start=args.start_date, end=args.end_date)

    # ── 以下為舊全撈五張報表（留作對照；總淨利含留倉=0上界，看時要心裡有數）──
    events = load_events(args.start_date, args.end_date, ticks=args.ticks)
    if events.height == 0:
        return

    ver = "ticks版" if args.ticks else "分K版"
    # 出場固定走 B 反向收斂（A 同向早平倒貼已砍）；事實表每事件一列，無需再篩 exit_type。
    # 剔除「做不成」的事件：potential_lots=0 = 兩腳量湊不成一個可成交最小單位
    # （標準需湊2張、小型需湊10口才對齊1張），有價差但打不出整數部位，不算數。
    before = events.height
    events = events.filter(pl.col("potential_lots") > 0)
    dropped = before - events.height
    days = events["date"].unique().to_list()
    print(f"[保守口徑｜出場B反向收斂｜{ver}] 讀入 {before} 個事件，"
          f"剔除湊不成最小單位(potential_lots=0) {dropped} 筆 → 剩 {events.height}，"
          f"涵蓋 {len(days)} 天")
    # 事實 → 衍生：在報表層用現行參數推 部位金額/費用/淨利/本金/ROI（本金口徑固定保守）
    events = enrich(events)
    print("（已從事實推算：部位金額＝口數×乘數×進場賣期價；費用＝期交稅＋期貨定額手續費"
          "＋現貨手續費＋證交稅(當沖0.15%/留倉0.3%)；現金買賣不計利息）")

    # 逐事件統計：收斂成「只第一次進場」（維持一事件=一次機會的舊語意；最低估算）
    events = events.filter(pl.col("is_first_entry"))

    # 排除 hold_secs==0：進場同一刻就被判出場(收斂)，出場四價常是爛 tick(出場成本爆衝、
    # 把整體平均拖垮)。事實 CSV 照存(可回 tick 查 chseq)，僅分析時濾。留倉(hold=null)保留。
    before_h0 = events.height
    events = events.filter(pl.col("hold_secs").is_null() | (pl.col("hold_secs") > 0))
    n_h0 = before_h0 - events.height
    if n_h0:
        print(f"（已排除 hold_secs=0 的同刻出場 {n_h0} 筆：疑爛 tick，事實 CSV 仍保留可回查）")
    print()
    run_reports(events, tag)


def run_reports(events: pl.DataFrame, tag: str) -> None:
    """五張報表（整年彙總/pivot/當沖vs留倉/出場流動性/資金需求）。
    events 須已 enrich、已濾 first、已排 hold0。含二次放大版(report_pool)也複用此函式，
    只是傳進來的 events 的 potential_lots/potential_value 是放大後的量。"""
    # (0) ★整年加總彙總（唯一落地的匯總檔）：各閥值×三口徑(全量/≥1秒/≥30秒) 一列。
    #     跨結算距離加總；各閥值獨立、不可跨閥值相加（0.5% 的事件含 1%）。
    print("===== 整年加總彙總（各閥值獨立勿相加；按看到後幾秒內打得到分層）=====")
    ys_parts = []
    for mh in (None, 1, 30):
        label = "全量" if mh is None else f">={mh}秒"
        ys = yearly_summary(events, min_hold=mh)
        if ys.height == 0:
            continue
        ys_parts.append(ys.with_columns(pl.lit(label).alias("口徑")))
        print(f"\n-- 執行口徑：{label} --")
        with pl.Config(tbl_cols=-1, tbl_width_chars=240, tbl_rows=-1):
            print(ys.with_columns([
                (pl.col("總淨利") / 1e8).round(2).alias("總淨利(億)"),
                (pl.col("總淨利_converged下界") / 1e8).round(2).alias("下界(億)"),  # M02
                (pl.col("converged占比") * 100).round(1).alias("converged占比%"),    # M02
                (pl.col("收斂率") * 100).round(1).alias("收斂率%"),
                (pl.col("平均淨利率") * 100).round(3).alias("淨利率%"),
                (pl.col("平均ROI") * 100).round(2).alias("ROI%"),
                (pl.col("平均年化ROI") * 100).round(1).alias("年化ROI%_逐筆等權"),
                (pl.col("資金加權年化ROI") * 100).round(1).alias("年化ROI%_資金加權"),  # M04
            ]).select("side", "threshold", "筆數", "收斂率%", "總淨利(億)",
                      "下界(億)", "converged占比%", "淨利率%", "ROI%",
                      "年化ROI%_逐筆等權", "年化ROI%_資金加權"))
    summary_path = None
    if ys_parts:
        os.makedirs(STATS_DIR, exist_ok=True)
        summary_path = os.path.join(STATS_DIR, f"summary_m{tag}.csv")
        pl.concat(ys_parts).with_columns([
            (pl.col("總淨利") / 1e8).round(2).alias("總淨利_億"),
            (pl.col("總淨利_converged下界") / 1e8).round(2).alias("總淨利下界_億"),       # M02
            (pl.col("converged占比") * 100).round(1).alias("converged占比_pct"),          # M02
            (pl.col("收斂率") * 100).round(1).alias("收斂率_pct"),
            (pl.col("平均淨利率") * 100).round(3).alias("淨利率_pct"),
            (pl.col("平均ROI") * 100).round(2).alias("ROI_pct"),
            (pl.col("平均年化ROI") * 100).round(1).alias("年化ROI_pct_逐筆等權"),
            (pl.col("資金加權年化ROI") * 100).round(1).alias("年化ROI_pct_資金加權"),     # M04
        ]).select("side", "threshold", "口徑", "筆數", "收斂率_pct",
                  "總淨利_億", "總淨利下界_億", "converged占比_pct", "淨利率_pct", "ROI_pct",
                  "年化ROI_pct_逐筆等權", "年化ROI_pct_資金加權").write_csv(summary_path)
    print()

    # 以下純印 console / 畫圖，不另存明細 CSV（匯總只要上面那一份）

    # (1) pivot：列=結算日距離、欄=閥值
    for side in SIDES:
        print(f"===== {side} pivot（列=結算日距離, 欄=閥值）=====")
        piv = pivot_one_side(events, side)
        if piv.height:
            with pl.Config(tbl_cols=-1, tbl_width_chars=400):
                print(piv)
        else:
            print("（無事件）")
        print()

    # (2) 當沖 vs 留倉 獲利拆分
    print("===== 當沖 vs 留倉 獲利拆分 =====")
    with pl.Config(tbl_cols=-1, tbl_width_chars=300, tbl_rows=-1):
        print(daytrade_vs_overnight(events).select(
            "side", "threshold", "型態", "筆數", "淨獲利合計",
            "當沖平均持有分鐘", "留倉平均距結算天數"))

    # (2.5) 各閥值出場流動性覆蓋（E08：每筆自己 vs 自己出場掛量，無跨事件搶量）
    print("\n===== 各閥值出場流動性：出得完嗎（收斂事件、每筆自己跟自己出場掛量比）=====")
    print("（期貨腳需 potential_lots≤exit_fut_ask掛量、現貨腳需股數≤exit_spot_bid掛量×1000，兩腳取緊。")
    print("  全出得完%=幾成事件整單出得掉；出得掉金額%=部位金額加權實際出得掉比例。看 0.5 vs 1 哪個較不卡。）")
    liq = exit_liquidity(events)
    if liq.height:
        with pl.Config(tbl_cols=-1, tbl_width_chars=240, tbl_rows=-1):
            print(liq.with_columns([
                (pl.col("全出得完_筆數比") * 100).round(1).alias("全出得完%"),
                (pl.col("出得掉_金額比") * 100).round(1).alias("出得掉金額%"),
                (pl.col("收斂淨利") / 1e8).round(3).alias("收斂淨利_億"),
                (pl.col("打折後收斂淨利") / 1e8).round(3).alias("打折後淨利_億"),
                (pl.col("可出場部位金額") / 1e8).round(2).alias("可出部位_億"),
            ]).select("side", "threshold", "可評估筆數", "全出得完%", "出得掉金額%",
                      "收斂淨利_億", "打折後淨利_億", "可出部位_億"))
    else:
        print("（無可評估資料：CSV 缺出場掛量 exit_*_lots，請 main.py --ticks 重跑補 E08）")

    # 註：收斂歸因（30s/60s 四點中價趨勢）已從主線移出，將獨立成 trend.py（讀同一份事實
    #     CSV，專做趨勢分析）。stats.py 只負責損益/持倉/資金需求。

    # (3) C：每日資金需求模擬（印峰值 + 畫圖；逐日曲線不落地）
    print("\n===== C：每日資金需求模擬（在場留倉＋當日當沖）★要準備多少看這 =====")
    print("（總需求 = 跨日累積的在場留倉 + 當天當沖佔用。峰值才是「要準備多少」。）")
    print("（⚠️ M05：曲線~90%是留倉堆積，僅在「全部未收斂抱到結算」政策下成立；")
    print("   是該政策的人造物、非市場固有需求。改提前出場政策此需求會大幅下降。）")
    thrs = sorted(events["threshold"].unique().to_list())
    # 樣本期日曆跨距（含端點）：峰值口徑年化要用，動態算、不寫死——每天新增資料跨距會變。
    from datetime import datetime as _dt
    _ds = [_dt.strptime(str(d), "%Y%m%d") for d in events["date"].unique().to_list()]
    span_days = max((max(_ds) - min(_ds)).days + 1, 1)   # +1 含端點；至少 1 天避免除零
    for side in SIDES:
        for mh in (None, 1, 30):
            label = "全量" if mh is None else f"進場窗>={mh}秒"
            print(f"\n-- {side}（{label}）--")
            # M04：峰值口徑年化 = (樣本期淨利 ÷ 峰值總需求) × 365/樣本日曆跨距。
            #   修正(2026/6/15)：原誤把「樣本期淨利/峰值」當年化，但分子是樣本期(非整年)淨利、
            #   分母純資金無時間維度 → 那只是「樣本期報酬率」，要再 ×365/span 才是年化。
            #   (逐筆等權/資金加權年化分母已含天數、本就年化；只有此峰值口徑漏了，已補。)
            sub_mh = events.filter(pl.col("side") == side)
            if mh is not None:
                sub_mh = sub_mh.filter(pl.col("first_stretch_secs") >= mh)
            print(f"  {'閥值':<8}{'總需求峰值':>14}{'峰值日':>12}{'總需求平均':>14}"
                  f"{'峰值/平均':>10}{'年化ROI%_峰值口徑':>18}")
            for thr in thrs:
                curve, peak, pday, avg = overnight_inventory(events, side, thr, min_hold=mh)
                if curve is None:
                    continue
                ratio = peak / avg if avg else 0
                yr_pnl = sub_mh.filter(pl.col("threshold") == thr)["potential_pnl"].sum()
                # 樣本期報酬率 × 年化因子(365/樣本日曆跨距)
                peak_roi = (yr_pnl / peak * 365 / span_days * 100) if peak else 0.0
                print(f"  {thr:<8.1%}{peak/1e8:>12.2f}億 {pday:>11}{avg/1e8:>12.2f}億 "
                      f"{ratio:>9.2f}x{peak_roi:>16.1f}%")
            plot_inventory_curves(events, side, tag, min_hold=mh)
        plot_cum_pnl(events, side, tag)               # 全量
        plot_cum_pnl(events, side, tag, min_hold=1)   # hold>=1秒(含留倉)
        plot_cum_pnl(events, side, tag, min_hold=30)  # hold>=30秒(含留倉)

    if summary_path:
        print(f"\n→ 統計結果（匯總 CSV + 圖檔）都在：{STATS_DIR}/")


if __name__ == "__main__":
    main()
