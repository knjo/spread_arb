"""收斂天數 x 進場門檻 x 結算週期位置（距結算日幾天 / 剛結算完幾天）。

問題：不同門檻（0.5%/1%/1.5%/2%）的 first entry，依進場日在結算週期中的位置
（距近月結算日還有幾個交易日、剛結算完第幾個交易日），平均幾個交易日等到反向收斂
（B 口徑 ret_buy >= 0），以及多少比例根本等不到、抱到結算日。

口徑：
  - 價差流：與 arbitrage_analysis.build_features_for_day 相同的清洗（可當沖、參考價 +-9%、
    TrialMatch=0、09:00 < t < 13:25、近月標準合約 contract_size=2000、現貨 as-of 60s），
    但不快取 NAS 全月份檔、只留價差所需欄位，寫成 {date}_spread_stream.parquet。
  - 近月 / 結算日：MySQL taifex_pib_view.end_date（逐合約、已含假日順延）；近月 = 該日
    end_date >= 當日 的最近合約（結算日當天整天仍算當月）。
  - 進場樣本：每日每檔 (Date, ValueCode) 第一筆 ret_sell >= 門檻 且 fut_bid1 > 0 的 tick
    （同 arbitrage_analysis._threshold_entries；各門檻獨立，不可跨門檻加總）。
  - 收斂：進場後同一 QuoteCode 第一筆 ret_buy >= 0（跨日搜尋）；收斂日 < 結算日 → 自然收斂；
    否則視為抱到結算日、收斂天數 = 距結算日交易日數（同 _classify_convergence）。
  - 天數：交易日（Common.calendar_view）計數為主，另附日曆天。

用法（taker 目錄下）：
  uv run python convergence_days_by_settle_cycle.py build -s 20260126 -e 20260715
  uv run python convergence_days_by_settle_cycle.py summarize -s 20260126 -e 20260625 --search-end 20260715
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

import arbitrage_analysis as arb
from data_paths import (
    STOCKFUTURE_DIR,
    _mysql_query,
    ensure_output_dir,
    load_futures_basic,
    scan_futures_ticks,
    spot_tick_path,
)

THRESHOLDS = [0.005, 0.01, 0.015, 0.02]
STREAM_DIR = STOCKFUTURE_DIR / "spread_stream"
FUT_COLS = [
    "RecvTime", "TransTime", "QuoteCode", "ValueCode", "TrialMatch",
    "BidPrice1", "AskPrice1", "BidLots1", "AskLots1",
    "BestBidPrice", "BestBidLots", "BestAskPrice", "BestAskLots",
]
TO_SETTLE_BUCKETS = [(0, 0, "0(結算日)"), (1, 2, "1-2"), (3, 5, "3-5"), (6, 10, "6-10"),
                     (11, 15, "11-15"), (16, 99, "16+")]
SINCE_SETTLE_BUCKETS = [(1, 2, "1-2"), (3, 5, "3-5"), (6, 10, "6-10"), (11, 15, "11-15"),
                        (16, 99, "16+")]


# ---------------------------------------------------------------- 日曆 / 合約
def load_trade_days(start: str, end: str) -> list[str]:
    df = _mysql_query(
        "SELECT date AS d, DayType FROM Common.calendar_view WHERE date BETWEEN :s AND :e",
        {"s": int(start), "e": int(end)},
    )
    days = (
        df.filter(pl.col("DayType") == "TradeDay")
        .with_columns(pl.col("d").cast(pl.Utf8).str.replace_all("-", ""))
        .sort("d")["d"].to_list()
    )
    return [d for d in days if spot_tick_path(d).exists()]


def near_month_contracts(date: str) -> pl.DataFrame:
    """該日近月標準合約：quote_code / ValueCode / fut_ref_price / contract_size / settle_date。"""
    basic = load_futures_basic(date).with_columns(
        pl.col("end_date").cast(pl.Date).dt.strftime("%Y%m%d").alias("settle_date")
    )
    std = basic.filter(
        (pl.col("quote_code").str.slice(2, 1) == "F")
        & ((pl.col("contract_size").cast(pl.Float64) - arb.STANDARD_CONTRACT_SIZE).abs()
           <= arb.CONTRACT_SIZE_EPS)
        & (pl.col("settle_date") >= date)
    )
    near = (
        std.sort(["value_code", "settle_date", "quote_code"])
        .group_by("value_code", maintain_order=True)
        .first()
        .select([
            pl.col("quote_code").alias("QuoteCode"),
            pl.col("value_code").alias("ValueCode"),
            pl.col("ref_price").cast(pl.Float64).alias("fut_ref_price"),
            pl.col("contract_size").cast(pl.Float64),
            "settle_date",
        ])
    )
    return near


def settlement_calendar(start: str, end: str) -> list[str]:
    """月結算日清單：每月找第一個 pib 有資料的交易日，蒐集標準合約 end_date。
    pib 沒資料的月份回退 spread_arb.contract 的第三週三 + 交易日順延。"""
    from data_paths import is_trade_day
    from spread_arb import contract as contract_calendar

    contract_calendar.set_trade_day_fn(is_trade_day)
    days = load_trade_days(start, end)
    by_month: dict[str, list[str]] = {}
    for d in days:
        by_month.setdefault(d[:6], []).append(d)
    settles: set[str] = set()
    for ym, month_days in by_month.items():
        got = False
        for d in month_days[:5]:
            try:
                basic = load_futures_basic(d)
            except RuntimeError:
                continue
            std = basic.filter(pl.col("quote_code").str.slice(2, 1) == "F")
            for s in std["end_date"].cast(pl.Date).dt.strftime("%Y%m%d").unique().to_list():
                settles.add(s)
            got = True
            break
        if not got:
            y, m = int(ym[:4]), int(ym[4:])
            settles.add(contract_calendar._actual_settlement(y, m).strftime("%Y%m%d"))
            print(f"{ym}: pib unavailable, settlement from third-Wednesday fallback", flush=True)
    return sorted(settles)


# ---------------------------------------------------------------- 價差流
def stream_path(date: str, out_dir: Path) -> Path:
    return out_dir / f"{date}_spread_stream.parquet"


def build_day(date: str, out_dir: Path) -> Path:
    t0 = time.perf_counter()
    tradable = arb._load_market_tradable(date)
    tradable_codes = set(tradable["ValueCode"].to_list())
    spot_ref = tradable.select(["ValueCode", "spot_ref_price"]).filter(
        pl.col("spot_ref_price").is_not_null() & (pl.col("spot_ref_price") > 0)
    )
    near = near_month_contracts(date)
    near = near.filter(pl.col("ValueCode").is_in(sorted(tradable_codes)))
    if near.height == 0:
        raise RuntimeError(f"{date}: no near-month standard contracts on tradable codes")

    raw = (
        scan_futures_ticks(date, near["QuoteCode"].to_list())
        .select(FUT_COLS)
        .collect()
    )
    if raw.height == 0:
        raise RuntimeError(f"{date}: no near-month futures ticks on NAS")
    fut = raw.drop("ValueCode").join(near, on="QuoteCode", how="inner")
    fut = arb._restore_prices(arb._normalize_time(fut), arb.FUT_SCALE)
    fut = fut.filter(pl.col("TrialMatch") == 0)
    fut = fut.filter(arb._session_time_expr(arb.TIME_COL))
    fut = fut.filter(arb._session_time_expr("TransTime"))
    fut = fut.filter(arb._has_book_expr())
    fut = arb._add_futures_best_quotes(fut).with_columns(
        arb._within_ref_limit_expr(
            "fut_ref_price", ["BidPrice1", "AskPrice1", "BestBidPrice", "BestAskPrice"]
        ).alias("fut_price_ok")
    ).filter(pl.col("fut_price_ok"))
    if fut.height == 0:
        raise RuntimeError(f"{date}: no eligible futures quote ticks")

    value_codes = sorted(set(fut["ValueCode"].to_list()))
    spot = arb._prepare_spot(date, value_codes, False, spot_ref)
    if spot.height == 0:
        raise RuntimeError(f"{date}: no eligible spot quote ticks")
    spot_state = arb._prepare_spot_state(date, value_codes, False, spot_ref)

    base = fut.select([
        pl.lit(date).alias("Date"),
        "ValueCode", "QuoteCode", "settle_date", arb.TIME_COL,
        pl.col("BidPrice1").alias("fut_bid1"),
        "fut_bid", "fut_ask", "fut_bid_lots", "fut_ask_lots",
    ]).sort(["ValueCode", arb.TIME_COL])
    if spot_state is not None:
        base = arb._join_asof(
            base, spot_state,
            left_on=arb.TIME_COL, right_on="spot_state_time", by="ValueCode",
            strategy="backward",
        ).filter(
            (pl.col("spot_trial_state").fill_null(1) == 0)
            & pl.col("spot_price_ok").fill_null(False)
        ).drop(["spot_state_time", "spot_trial_state", "spot_price_ok"])
    feat = arb._join_asof(
        base,
        spot.rename({arb.TIME_COL: "spot_time"}).select([
            "ValueCode", "spot_time", "spot_ask", "spot_bid", "spot_ask_lots", "spot_bid_lots",
        ]),
        left_on=arb.TIME_COL, right_on="spot_time", by="ValueCode",
        strategy="backward", tolerance="60s",
    )
    feat = feat.filter(
        (pl.col("spot_bid") > 0) & (pl.col("spot_ask") > 0)
        & (pl.col("fut_bid") > 0) & (pl.col("fut_ask") > 0)
    ).with_columns([
        ((pl.col("fut_bid") - pl.col("spot_ask")) / pl.col("fut_ask")).alias("ret_sell"),
        ((pl.col("spot_bid") - pl.col("fut_ask")) / pl.col("fut_ask")).alias("ret_buy"),
    ]).sort(["QuoteCode", arb.TIME_COL])

    out_dir.mkdir(parents=True, exist_ok=True)
    out = stream_path(date, out_dir)
    feat.write_parquet(out)
    print(
        f"{date}: spread stream {feat.height:,} rows, {near.height} contracts, "
        f"{time.perf_counter() - t0:.0f}s -> {out}",
        flush=True,
    )
    return out


def build(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir) if args.out_dir else STREAM_DIR
    if not args.out_dir:
        ensure_output_dir(out_dir)
    for date in load_trade_days(args.start_date, args.end_date):
        if stream_path(date, out_dir).exists() and not args.force:
            print(f"{date}: exists, skip", flush=True)
            continue
        try:
            build_day(date, out_dir)
        except Exception as exc:
            if args.skip_errors:
                print(f"{date}: skipped ({type(exc).__name__}: {exc})", flush=True)
                continue
            raise


# ---------------------------------------------------------------- 進場 / 收斂
def _day_entries_and_conv(date: str, out_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """單日：各門檻 first entry（含當日收斂時間）+ 每合約當日第一筆 ret_buy>=0 時間。"""
    df = pl.read_parquet(stream_path(date, out_dir))
    conv = (
        df.filter(pl.col("ret_buy") >= 0)
        .select(["QuoteCode", pl.col(arb.TIME_COL).alias("converge_time")])
        .sort(["QuoteCode", "converge_time"])
    )
    first_conv = conv.group_by("QuoteCode").agg(
        pl.col("converge_time").min().alias("day_first_converge_time")
    ).with_columns(pl.lit(date).alias("conv_date"))

    parts = []
    for thr in THRESHOLDS:
        ent = (
            df.filter((pl.col("ret_sell") >= thr) & (pl.col("fut_bid1") > 0))
            .sort(["Date", "ValueCode", arb.TIME_COL, "QuoteCode"])
            .group_by(["Date", "ValueCode"], maintain_order=True)
            .first()
            .with_columns(pl.lit(thr).alias("threshold"))
        )
        if ent.height:
            parts.append(ent)
    if not parts:
        return pl.DataFrame(), first_conv
    entries = pl.concat(parts, how="diagonal_relaxed").rename({arb.TIME_COL: "entry_time"})
    entries = arb._join_asof(
        entries.sort(["QuoteCode", "entry_time"]), conv,
        left_on="entry_time", right_on="converge_time", by="QuoteCode", strategy="forward",
    )
    return entries, first_conv


def _bucket_expr(col: str, buckets: list[tuple[int, int, str]], name: str) -> pl.Expr:
    expr = pl.when(pl.lit(False)).then(pl.lit(None, dtype=pl.Utf8))
    for lo, hi, label in buckets:
        expr = expr.when((pl.col(col) >= lo) & (pl.col(col) <= hi)).then(pl.lit(label))
    return expr.otherwise(pl.lit("other")).alias(name)


def _bucket_order(buckets: list[tuple[int, int, str]]) -> dict[str, int]:
    return {label: i for i, (_, _, label) in enumerate(buckets)}


def build_samples(entry_days: list[str], search_days: list[str], out_dir: Path,
                  settles: list[str]) -> pl.DataFrame:
    entries_parts, conv_parts = [], []
    for d in search_days:
        if not stream_path(d, out_dir).exists():
            print(f"{d}: stream missing, skipped", flush=True)
            continue
        ent, first_conv = _day_entries_and_conv(d, out_dir)
        conv_parts.append(first_conv)
        if d in entry_days and ent.height:
            entries_parts.append(ent)
    entries = pl.concat(entries_parts, how="diagonal_relaxed")
    first_conv = pl.concat(conv_parts).sort(["QuoteCode", "day_first_converge_time"])

    # 只留「當月結算週期」的合約：該檔最近合約若不是本週期（例如 2014 只有季月合約、
    # 6443 新掛牌只有 9 月），距結算天數會落在 24~62 天，不屬於本分析的週期位置，排除並計數。
    settle_sorted = sorted(settles)
    cycle_map = {
        d: next((x for x in settle_sorted if x >= d), None)
        for d in entries["Date"].unique().to_list()
    }
    entries = entries.with_columns(
        pl.col("Date").replace_strict(cycle_map, default=None, return_dtype=pl.Utf8)
          .alias("cycle_settle_date")
    )
    off = entries.filter(pl.col("settle_date") != pl.col("cycle_settle_date"))
    if off.height:
        print(
            f"dropped {off.height} off-cycle entries "
            f"({off.select('ValueCode', 'QuoteCode').unique().height} contracts: "
            f"{sorted(off['QuoteCode'].unique().to_list())})",
            flush=True,
        )
    entries = entries.filter(pl.col("settle_date") == pl.col("cycle_settle_date"))

    # 跨日：進場日之後（Date < conv_date）最早有 ret_buy>=0 的那天
    later = arb._join_asof(
        entries.with_columns(
            (pl.col("entry_time").dt.date().cast(pl.Datetime("us")) + pl.duration(days=1))
            .alias("_next_day")
        ).sort(["QuoteCode", "_next_day"]),
        first_conv,
        left_on="_next_day", right_on="day_first_converge_time", by="QuoteCode",
        strategy="forward",
    ).drop("_next_day")
    later = later.with_columns([
        pl.when(pl.col("converge_time").is_not_null())
          .then(pl.col("converge_time"))
          .otherwise(pl.col("day_first_converge_time"))
          .alias("converge_time_all"),
    ]).with_columns(
        pl.col("converge_time_all").dt.strftime("%Y%m%d").alias("converge_date")
    ).drop(["converge_time", "day_first_converge_time", "conv_date"]) \
     .rename({"converge_time_all": "converge_time"})

    # 交易日索引：MySQL 日曆涵蓋「最早的前一結算日 ~ 最晚的結算日」，不受搜尋窗限制

    def prev_settle(d: str) -> str | None:
        prev = [s for s in settle_sorted if s < d]
        return prev[-1] if prev else None

    entry_dates = sorted(later["Date"].unique().to_list())
    prev_map = {d: prev_settle(d) for d in entry_dates}
    last_settle = str(later["settle_date"].max())
    cal_start = min([s for s in prev_map.values() if s is not None] + [search_days[0]])
    cal_days = load_trade_days(cal_start, max(last_settle, search_days[-1]))
    cal_idx = {d: i for i, d in enumerate(cal_days)}
    if search_days[-1] < last_settle:
        print(
            f"warning: convergence search ends {search_days[-1]} before last settlement "
            f"{last_settle}; unresolved entries are counted as held to settlement",
            flush=True,
        )

    def td_index(col: str, alias: str) -> pl.Expr:
        return pl.col(col).replace_strict(cal_idx, default=None, return_dtype=pl.Int64).alias(alias)

    prev_idx = {d: cal_idx[s] for d, s in prev_map.items() if s is not None and s in cal_idx}

    out = later.with_columns([
        td_index("Date", "entry_idx"),
        td_index("settle_date", "settle_idx"),
        td_index("converge_date", "converge_idx"),
        pl.col("Date").replace_strict(prev_map, default=None, return_dtype=pl.Utf8)
          .alias("prev_settle_date"),
        pl.col("Date").replace_strict(prev_idx, default=None, return_dtype=pl.Int64)
          .alias("prev_settle_idx"),
    ]).with_columns([
        (pl.col("settle_idx") - pl.col("entry_idx")).alias("days_to_settle_td"),
        (pl.col("entry_idx") - pl.col("prev_settle_idx")).alias("days_since_settle_td"),
        (pl.col("settle_date").str.strptime(pl.Date, "%Y%m%d")
         - pl.col("Date").str.strptime(pl.Date, "%Y%m%d")).dt.total_days()
          .alias("days_to_settle_cal"),
        pl.when(pl.col("converge_date").is_not_null()
                & (pl.col("converge_date") < pl.col("settle_date")))
          .then(pl.lit(True)).otherwise(pl.lit(False)).alias("natural_converge"),
    ]).with_columns([
        pl.when(pl.col("natural_converge"))
          .then(pl.col("converge_idx") - pl.col("entry_idx"))
          .otherwise(pl.col("days_to_settle_td"))
          .alias("convergence_days_td"),
        pl.when(pl.col("natural_converge"))
          .then((pl.col("converge_date").str.strptime(pl.Date, "%Y%m%d")
                 - pl.col("Date").str.strptime(pl.Date, "%Y%m%d")).dt.total_days())
          .otherwise(pl.col("days_to_settle_cal"))
          .alias("convergence_days_cal"),
        pl.when(pl.col("converge_date") == pl.col("Date")).then(pl.lit("當日收斂"))
          .when(pl.col("natural_converge")).then(pl.lit("跨日收斂"))
          .otherwise(pl.lit("抱到結算日")).alias("status"),
        _bucket_expr("days_to_settle_td", TO_SETTLE_BUCKETS, "to_settle_bucket"),
        _bucket_expr("days_since_settle_td", SINCE_SETTLE_BUCKETS, "since_settle_bucket"),
    ])
    return out


def _agg_exprs() -> list[pl.Expr]:
    nat = pl.col("natural_converge")
    return [
        pl.len().alias("樣本數"),
        (pl.col("status") == "當日收斂").mean().mul(100).alias("當日收斂%"),
        (pl.col("status") == "跨日收斂").mean().mul(100).alias("跨日收斂%"),
        (pl.col("status") == "抱到結算日").mean().mul(100).alias("抱到結算%"),
        pl.col("convergence_days_td").mean().alias("平均收斂交易日(含結算)"),
        pl.col("convergence_days_td").median().alias("中位收斂交易日(含結算)"),
        pl.col("convergence_days_td").filter(nat).mean().alias("平均收斂交易日(自然收斂)"),
        pl.col("convergence_days_td").filter(nat & (pl.col("status") == "跨日收斂")).mean()
          .alias("平均收斂交易日(跨日者)"),
        pl.col("convergence_days_cal").mean().alias("平均收斂日曆天(含結算)"),
        pl.col("days_to_settle_td").mean().alias("平均距結算交易日"),
    ]


def summarize_by(samples: pl.DataFrame, bucket_col: str,
                 buckets: list[tuple[int, int, str]]) -> pl.DataFrame:
    order = _bucket_order(buckets)
    return (
        samples.group_by(["threshold", bucket_col]).agg(_agg_exprs())
        .with_columns(
            pl.col(bucket_col).replace_strict(order, default=99, return_dtype=pl.Int64)
              .alias("_o")
        )
        .sort(["threshold", "_o"]).drop("_o")
    )


def summarize_exact(samples: pl.DataFrame, col: str) -> pl.DataFrame:
    return samples.group_by(["threshold", col]).agg(_agg_exprs()).sort(["threshold", col])


def _fmt(x, nd=1) -> str:
    if x is None:
        return "-"
    return f"{float(x):.{nd}f}"


def _md_table(df: pl.DataFrame, bucket_col: str, bucket_label: str) -> list[str]:
    cols = ["門檻", bucket_label, "樣本數", "當日收斂%", "跨日收斂%", "抱到結算%",
            "平均收斂交易日(含結算)", "中位", "平均(自然收斂者)", "平均(跨日者)",
            "平均收斂日曆天(含結算)", "平均距結算交易日"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.iter_rows(named=True):
        lines.append("| " + " | ".join([
            f"{r['threshold'] * 100:.1f}%", str(r[bucket_col]), str(r["樣本數"]),
            _fmt(r["當日收斂%"]), _fmt(r["跨日收斂%"]), _fmt(r["抱到結算%"]),
            _fmt(r["平均收斂交易日(含結算)"]), _fmt(r["中位收斂交易日(含結算)"], 0),
            _fmt(r["平均收斂交易日(自然收斂)"]), _fmt(r["平均收斂交易日(跨日者)"]),
            _fmt(r["平均收斂日曆天(含結算)"]), _fmt(r["平均距結算交易日"]),
        ]) + " |")
    return lines


def _md_exact_table(df: pl.DataFrame, col: str, label: str, thrs: list[float]) -> list[str]:
    """逐日展開：每列一個 N，各門檻並排（樣本數 / 抱到結算% / 平均收斂交易日含結算）。"""
    head = [label]
    for t in thrs:
        head += [f"{t * 100:.1f}% 樣本", f"{t * 100:.1f}% 當日%", f"{t * 100:.1f}% 抱到結算%",
                 f"{t * 100:.1f}% 平均收斂日"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    ns = sorted(df[col].drop_nulls().unique().to_list())
    for n in ns:
        row = [str(n)]
        for t in thrs:
            r = df.filter((pl.col(col) == n) & ((pl.col("threshold") - t).abs() < 1e-9))
            if r.height == 0:
                row += ["-", "-", "-", "-"]
                continue
            r = r.row(0, named=True)
            row += [str(r["樣本數"]), _fmt(r["當日收斂%"], 0), _fmt(r["抱到結算%"], 0),
                    _fmt(r["平均收斂交易日(含結算)"])]
        lines.append("| " + " | ".join(row) + " |")
    return lines


def summarize(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir) if args.out_dir else STREAM_DIR
    res_dir = Path(args.result_dir) if args.result_dir else (out_dir.parent / "convergence_days")
    res_dir.mkdir(parents=True, exist_ok=True)
    search_end = args.search_end or args.end_date
    search_days = load_trade_days(args.start_date, search_end)
    entry_days = [d for d in search_days if d <= args.end_date]
    settles = settlement_calendar(
        (datetime.strptime(args.start_date, "%Y%m%d") - timedelta(days=45)).strftime("%Y%m%d"),
        search_end,
    )
    print("settlement dates:", settles, flush=True)
    samples = build_samples(entry_days, search_days, out_dir, settles)
    tag = f"{args.start_date}_{args.end_date}"
    samples.write_parquet(res_dir / f"convergence_samples_{tag}.parquet")

    overall = samples.group_by("threshold").agg(_agg_exprs()).sort("threshold")
    by_to = summarize_by(samples, "to_settle_bucket", TO_SETTLE_BUCKETS)
    by_since = summarize_by(samples, "since_settle_bucket", SINCE_SETTLE_BUCKETS)
    exact_to = summarize_exact(samples, "days_to_settle_td")
    exact_since = summarize_exact(samples, "days_since_settle_td")
    overall.write_csv(res_dir / f"by_threshold_{tag}.csv")
    by_to.write_csv(res_dir / f"by_days_to_settle_{tag}.csv")
    by_since.write_csv(res_dir / f"by_days_since_settle_{tag}.csv")
    exact_to.write_csv(res_dir / f"exact_days_to_settle_{tag}.csv")
    exact_since.write_csv(res_dir / f"exact_days_since_settle_{tag}.csv")

    n_days = len(entry_days)
    lines = [
        f"樣本：進場日 {entry_days[0]}~{entry_days[-1]}（{n_days} 個交易日），收斂搜尋到 {search_days[-1]}。",
        f"結算日：{', '.join(settles)}。",
        "",
        "### 各門檻總表",
        "",
    ]
    lines += _md_table(overall.with_columns(pl.lit("全部").alias("全部")), "全部", "分組")
    lines += ["", "### 依「距結算日幾個交易日」分組", ""]
    lines += _md_table(by_to, "to_settle_bucket", "距結算(交易日)")
    lines += ["", "### 依「剛結算完幾個交易日」分組", ""]
    lines += _md_table(by_since, "since_settle_bucket", "結算後第N日")
    lines += ["", "### 逐日展開：距結算日 N 個交易日（0.5% / 1.0%）", ""]
    lines += _md_exact_table(exact_to, "days_to_settle_td", "距結算(交易日)", [0.005, 0.01])
    lines += ["", "### 逐日展開：結算後第 N 個交易日（0.5% / 1.0%）", ""]
    lines += _md_exact_table(exact_since, "days_since_settle_td", "結算後第N日", [0.005, 0.01])
    md_path = res_dir / f"convergence_days_{tag}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pl.Config(tbl_cols=-1, tbl_rows=-1, tbl_width_chars=300):
        print(overall)
        print(by_to)
        print(by_since)
    print(f"-> {md_path}")


# ---------------------------------------------------------------- 交叉表 / tick size
PRICE_BANDS = [(0, 10, "<10"), (10, 50, "10-50"), (50, 100, "50-100"), (100, 500, "100-500"),
               (500, 1000, "500-1000"), (1000, 1e12, ">=1000")]
TICK_BP_BUCKETS = [(0, 10, "<10bp"), (10, 20, "10-20bp"), (20, 30, "20-30bp"), (30, 50, "30-50bp"),
                   (50, 1e9, ">=50bp")]
POS_BUCKETS = [(0.0, 0.2, "0-20%"), (0.2, 0.4, "20-40%"), (0.4, 0.6, "40-60%"), (0.6, 0.8, "60-80%"),
               (0.8, 1.01, "80-100%")]


def _spot_tick_expr(col: str) -> pl.Expr:
    p = pl.col(col)
    return (
        pl.when(p < 10).then(0.01).when(p < 50).then(0.05).when(p < 100).then(0.1)
        .when(p < 500).then(0.5).when(p < 1000).then(1.0).otherwise(5.0)
    )


def _fut_tick_expr(col: str) -> pl.Expr:
    p = pl.col(col)
    return (
        pl.when(p < 10).then(0.01).when(p < 50).then(0.05).when(p < 100).then(0.1)
        .when(p < 500).then(0.5).otherwise(1.0)
    )


def _band_expr(col: str, bands: list[tuple[float, float, str]], name: str) -> pl.Expr:
    expr = pl.when(pl.lit(False)).then(pl.lit(None, dtype=pl.Utf8))
    for lo, hi, label in bands:
        expr = expr.when((pl.col(col) >= lo) & (pl.col(col) < hi)).then(pl.lit(label))
    return expr.otherwise(pl.lit("other")).alias(name)


def enrich_samples(samples: pl.DataFrame) -> pl.DataFrame:
    """加：週期長度／進度、現貨價位檔、相對 tick、進場價差幾個 tick、兩腿 spread bp。"""
    return samples.with_columns([
        (pl.col("days_to_settle_td") + pl.col("days_since_settle_td")).alias("cycle_len_td"),
        _spot_tick_expr("spot_ask").alias("spot_tick"),
        _fut_tick_expr("fut_bid").alias("fut_tick"),
    ]).with_columns([
        (pl.col("days_since_settle_td") / pl.col("cycle_len_td")).alias("cycle_pos"),
        (pl.col("spot_tick") / pl.col("spot_ask") * 1e4).alias("spot_tick_bp"),
        ((pl.col("fut_bid") - pl.col("spot_ask")) / pl.col("spot_tick")).alias("premium_spot_ticks"),
        ((pl.col("spot_ask") - pl.col("spot_bid")) / pl.col("spot_ask") * 1e4).alias("spot_spread_bp"),
        ((pl.col("fut_ask") - pl.col("fut_bid")) / pl.col("fut_ask") * 1e4).alias("fut_spread_bp"),
        _band_expr("spot_ask", PRICE_BANDS, "price_band"),
    ]).with_columns([
        _band_expr("cycle_pos", POS_BUCKETS, "pos_bucket"),
        _band_expr("spot_tick_bp", TICK_BP_BUCKETS, "tick_bp_bucket"),
        (pl.col("spot_spread_bp") + pl.col("fut_spread_bp")).alias("two_leg_spread_bp"),
    ])


def _cell_stats() -> list[pl.Expr]:
    return [
        pl.len().alias("n"),
        (pl.col("status") == "當日收斂").mean().mul(100).alias("當日%"),
        (pl.col("status") == "抱到結算日").mean().mul(100).alias("抱到結算%"),
        pl.col("convergence_days_td").mean().alias("平均收斂日"),
        pl.col("convergence_days_td").filter(pl.col("natural_converge")).mean().alias("平均(自然)"),
    ]


def _crosstab_md(df: pl.DataFrame, thr: float, row_col: str, rows: list[str], col_col: str,
                 cols: list[str], row_label: str) -> list[str]:
    sub = df.filter((pl.col("threshold") - thr).abs() < 1e-9)
    g = sub.group_by([row_col, col_col]).agg(_cell_stats())
    head = [f"{row_label} \\ 距結算"] + cols
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        line = [r]
        for c in cols:
            x = g.filter((pl.col(row_col) == r) & (pl.col(col_col) == c))
            if x.height == 0:
                line.append("-")
            else:
                x = x.row(0, named=True)
                line.append(f"{x['n']} / {x['當日%']:.0f}% / {x['平均收斂日']:.1f}")
        out.append("| " + " | ".join(line) + " |")
    return out


def _group_md(df: pl.DataFrame, keys: list[str], labels: list[str], order: dict | None = None,
              extra: list[tuple[str, pl.Expr, int]] | None = None) -> list[str]:
    extra = extra or []
    g = df.group_by(["threshold", *keys]).agg(_cell_stats() + [e.alias(n) for n, e, _ in extra])
    if order:
        g = g.with_columns(
            pl.col(keys[-1]).replace_strict(order, default=99, return_dtype=pl.Int64).alias("_o")
        ).sort(["threshold", *keys[:-1], "_o"]).drop("_o")
    else:
        g = g.sort(["threshold", *keys])
    head = ["門檻", *labels, "樣本", "當日%", "抱到結算%", "平均收斂日", "平均(自然)", *[n for n, _, _ in extra]]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in g.iter_rows(named=True):
        line = [f"{r['threshold'] * 100:.1f}%", *[str(r[k]) for k in keys], str(r["n"]),
                _fmt(r["當日%"], 0), _fmt(r["抱到結算%"], 0), _fmt(r["平均收斂日"]), _fmt(r["平均(自然)"])]
        line += [_fmt(r[n], nd) for n, _, nd in extra]
        out.append("| " + " | ".join(line) + " |")
    return out, g


def crosstab(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir) if args.out_dir else STREAM_DIR
    res_dir = Path(args.result_dir) if args.result_dir else (out_dir.parent / "convergence_days")
    tag = f"{args.start_date}_{args.end_date}"
    samples = enrich_samples(pl.read_parquet(res_dir / f"convergence_samples_{tag}.parquet"))
    samples.write_parquet(res_dir / f"convergence_samples_enriched_{tag}.parquet")

    since_rows = [b[2] for b in SINCE_SETTLE_BUCKETS]
    to_cols = [b[2] for b in TO_SETTLE_BUCKETS][::-1]
    lines = ["## 週期位置換軸", ""]
    cyc = (samples.group_by("settle_date").agg([
        pl.col("cycle_len_td").max().alias("週期交易日"),
        pl.col("prev_settle_date").first().alias("前一結算日"),
        pl.len().alias("樣本(四門檻合計)"),
    ]).sort("settle_date"))
    lines += ["### 各週期長度", "", "| 結算日 | 前一結算日 | 週期交易日數 | 樣本(四門檻合計) |", "|---|---|---|---|"]
    for r in cyc.iter_rows(named=True):
        lines.append(f"| {r['settle_date']} | {r['前一結算日']} | {r['週期交易日']} | {r['樣本(四門檻合計)']} |")
    for thr in [0.005, 0.01]:
        lines += ["", f"### 交叉表 {thr * 100:.1f}%：結算後第 N 日 × 距結算 N 日（格內 = 樣本 / 當日% / 平均收斂交易日）", ""]
        lines += _crosstab_md(samples, thr, "since_settle_bucket", since_rows, "to_settle_bucket", to_cols, "結算後")
    pos_order = {b[2]: i for i, b in enumerate(POS_BUCKETS)}
    lines += ["", "### 依「週期進度」分組（結算後交易日數 ÷ 週期長度；0% = 剛結算完、100% = 結算日）", ""]
    md, pos_tbl = _group_md(samples, ["pos_bucket"], ["週期進度"], pos_order,
                            extra=[("平均距結算日", pl.col("days_to_settle_td").mean(), 1)])
    lines += md
    pos_tbl.write_csv(res_dir / f"by_cycle_position_{tag}.csv")
    since_order = {b[2]: i for i, b in enumerate(SINCE_SETTLE_BUCKETS)}
    lines += ["", "### 各週期 × 結算後第 N 日（0.5% / 1.0%）", ""]
    md, cyc_since = _group_md(samples.filter(pl.col("threshold") <= 0.01 + 1e-9),
                              ["settle_date", "since_settle_bucket"], ["結算週期", "結算後第N日"], since_order)
    lines += md
    cyc_since.write_csv(res_dir / f"by_cycle_since_settle_{tag}.csv")

    lines += ["", "## Tick size（進場那刻現貨價位檔）", ""]
    band_order = {b[2]: i for i, b in enumerate(PRICE_BANDS)}
    extra = [("現貨相對tick(bp)", pl.col("spot_tick_bp").mean(), 1),
             ("進場溢價=幾個現貨tick", pl.col("premium_spot_ticks").median(), 1),
             ("兩腿spread(bp)", pl.col("two_leg_spread_bp").median(), 0),
             ("平均距結算日", pl.col("days_to_settle_td").mean(), 1)]
    md, band_tbl = _group_md(samples, ["price_band"], ["現貨價位"], band_order, extra=extra)
    lines += ["### 各門檻 × 價位檔", ""] + md
    band_tbl.write_csv(res_dir / f"by_price_band_{tag}.csv")
    to_order = {b[2]: i for i, b in enumerate(TO_SETTLE_BUCKETS)}
    lines += ["", "### 0.5%：價位檔 × 距結算分組", ""]
    md, band_to = _group_md(samples.filter((pl.col("threshold") - 0.005).abs() < 1e-9),
                            ["price_band", "to_settle_bucket"], ["現貨價位", "距結算"], to_order)
    lines += md
    band_to.write_csv(res_dir / f"by_price_band_to_settle_{tag}.csv")

    tick_order = {b[2]: i for i, b in enumerate(TICK_BP_BUCKETS)}
    lines += ["", "### 各門檻 × 現貨相對 tick（tick ÷ 進場現貨價，bp）", ""]
    md, tick_tbl = _group_md(samples, ["tick_bp_bucket"], ["相對tick"], tick_order, extra=extra)
    lines += md
    tick_tbl.write_csv(res_dir / f"by_tick_bp_{tag}.csv")
    lines += ["", "### 0.5%：相對 tick × 距結算分組", ""]
    md, tick_to = _group_md(samples.filter((pl.col("threshold") - 0.005).abs() < 1e-9),
                            ["tick_bp_bucket", "to_settle_bucket"], ["相對tick", "距結算"], to_order)
    lines += md
    tick_to.write_csv(res_dir / f"by_tick_bp_to_settle_{tag}.csv")
    lines += ["", "### 0.5%：相對 tick × 結算週期（控制月份）", ""]
    md, tick_cyc = _group_md(samples.filter((pl.col("threshold") - 0.005).abs() < 1e-9),
                             ["settle_date", "tick_bp_bucket"], ["結算週期", "相對tick"], tick_order)
    lines += md
    md_path = res_dir / f"convergence_crosstab_{tag}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"-> {md_path}")


def arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build per-day spread stream")
    b.add_argument("-s", "--start-date", required=True)
    b.add_argument("-e", "--end-date", required=True)
    b.add_argument("--out-dir", help=f"default {STREAM_DIR}")
    b.add_argument("--force", action="store_true")
    b.add_argument("--skip-errors", action="store_true")
    s = sub.add_parser("summarize", help="entries x convergence days summary")
    s.add_argument("-s", "--start-date", required=True)
    s.add_argument("-e", "--end-date", required=True, help="last entry date")
    s.add_argument("--search-end", help="last date to search convergence (default end-date)")
    s.add_argument("--out-dir", help=f"stream dir, default {STREAM_DIR}")
    s.add_argument("--result-dir", help="default <stream dir>/../convergence_days")
    c = sub.add_parser("crosstab", help="cycle-position cross tabs and tick-size bands from samples")
    c.add_argument("-s", "--start-date", required=True)
    c.add_argument("-e", "--end-date", required=True)
    c.add_argument("--out-dir")
    c.add_argument("--result-dir")
    return p


def main() -> None:
    args = arg_parser().parse_args()
    if args.cmd == "build":
        build(args)
    elif args.cmd == "crosstab":
        crosstab(args)
    else:
        summarize(args)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
