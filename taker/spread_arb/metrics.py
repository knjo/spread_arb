"""第4層：每事件指標（METHODOLOGY.md 步驟 7）。

對 tag_events 標好 event_id 的 df，算每一波事件的觀察指標：
  距結算日天數、潛在部位、發散 max&mean、日內收斂、需求部位。
"""
from __future__ import annotations

import warnings

import polars as pl

from .contract import days_to_settlement
from .preprocess import TIME_COL
from .spread import _ret_col, _opp_ret_col, SIDE_SELL, SIDE_BUY

# 契約乘數預設值（join 不到 contract_size 時的回退）：標準股票期貨 1 口 = 2000 股
FUT_MULTIPLIER = 2000
# 現貨 Lots 單位 = 張（SDK 作者確認），1 張 = 1000 股
SPOT_SHARES_PER_LOT = 1000


def _pair_lots(fut_lots: pl.Expr, spot_lots: pl.Expr, cs_col: str = "contract_size",
               has_cs: bool = True) -> tuple[pl.Expr, pl.Expr]:
    """股數配對法：期貨腳量(口) vs 現貨腳量(張)，湊整後算可配對的期貨口數與股數。

    一次/二次進場共用同一套配對（V05：現貨1張=1000股，不可與口數直接 min）：
      期貨股數 = 量 × cs；現貨股數 = 量 × 1000；配對 = min，再用 unit 湊整。
      unit：cs 整除1000→用 cs(標準2000=1口2張)；否則→1000(小型10口=1張)。
    回 (potential_lots 期貨口數, shares 可配對股數)。fut_lots/spot_lots 為兩腳量的 Expr。
    """
    cs = (pl.col(cs_col) if has_cs else pl.lit(FUT_MULTIPLIER)) \
        .fill_null(FUT_MULTIPLIER).cast(pl.Int64)
    unit = pl.when(cs % SPOT_SHARES_PER_LOT == 0).then(cs) \
             .otherwise(pl.lit(SPOT_SHARES_PER_LOT))
    pairable = pl.min_horizontal(
        fut_lots.cast(pl.Int64) * cs,                      # 期貨腳股數
        spot_lots.cast(pl.Int64) * SPOT_SHARES_PER_LOT,    # 現貨腳股數(張×1000)
    )
    shares = (pairable // unit) * unit          # 湊整後可配對股數
    return (shares // cs), shares               # (potential_lots 口數, shares 股數)


def event_metrics(tagged: pl.DataFrame, side: str, date,
                  fut_fills: pl.DataFrame | None = None,
                  spot_fills: pl.DataFrame | None = None,
                  threshold: float = 0.0) -> pl.DataFrame:
    """彙整每事件指標。tagged 需含 event_id（tag_events 產出）。

    fut_fills/spot_fills：兩腿「成交原始流」(撈檔層標 is_fill 的列，只留 key/時間/成交價量)，
      供算二次進場(事件區間內、期貨成交驅動的後續進場點)。None 則只算第一次進場（向後相容）。
    threshold：二次進場重算價差用的門檻（與第一次進場同一個）。

    每個進場事件輸出「一筆 row」，出場固定走反向收斂 B（ret_buy≥0，平倉那刀不倒貼）。
      （A 同向收斂早平倒貼已砍——taker 出場保證倒貼、無實務意義。）
    進場半邊存四價/量/口數/快照；出場半邊存 converged/converge_time/出場四價/序號。

    指標（價差賣為例；買對稱）：
      first_time      事件第一筆時間
      first_ret       第一筆毛獲利率
      lots_fut/spot   第一筆兩腳量
      potential_lots  潛在部位口數 = min(兩腳量)（兩腳都要鎖得住）
      potential_value 潛在部位金額 = potential_lots × 第一筆期貨價
      diverge_max     事件區間內 ret 最大（發散最大）
      diverge_mean    事件區間內 ret 平均
      converged       日內是否達 B 出場（到收盤前 ret_buy≥0 曾成立）
      days_to_settle  距結算日天數
    純事實層：出場只存原始四價/序號/時間/converged；exit_ret、hold_secs、exit_cost、
      部位金額/費用/淨利 等推得的量全由 stats 層算（[[feedback-fact-layer-raw-only]]）。
    """
    ret = _ret_col(side)
    # 價差賣：成交腳價用 fut_bid（賣期取優買）；量用兩腳
    fut_lots_col = "fut_bid_lots" if side == SIDE_SELL else "fut_ask_lots"
    spot_lots_col = "spot_ask_lots" if side == SIDE_SELL else "spot_bid_lots"
    fut_px_col = "fut_bid" if side == SIDE_SELL else "fut_ask"

    events = tagged.filter(pl.col("event_id").is_not_null())
    d2s = days_to_settlement(date)
    # 事件鍵：以「合約」(QuoteCode)為主體，與 tag_events 一致；無 QuoteCode 時退回 ValueCode
    key = "QuoteCode" if "QuoteCode" in tagged.columns else "ValueCode"

    # (1) 事件區間內彙整：發散 max/mean、第一筆資訊、閥值上窗口
    has_cs = "contract_size" in tagged.columns
    agg = events.group_by(key, "event_id").agg([
        *([pl.col("ValueCode").first()] if key != "ValueCode" else []),  # 保留標的代號供輸出/對照
        *([pl.col("contract_size").first()] if has_cs else []),          # 逐合約乘數(標準2000/小型100)
        pl.col(TIME_COL).first().alias("first_time"),
        # 最後一筆仍 >= 閥值的時間 → entry_window 下界（row 有更新才算）
        pl.col(TIME_COL).filter(pl.col("is_signal")).max().alias("last_signal_time"),
        # 事件內第一筆「明確 < 閥值」的時間（非null才算）→ 窗口上界用：
        # 在這之前報價沒更新=委託簿沒變=價差仍掛著可打
        pl.col(TIME_COL).filter(
            (~pl.col("is_signal")) & pl.col(ret).is_not_null()
        ).min().alias("first_subthr_time"),
        pl.col(ret).first().alias("first_ret"),
        pl.col(ret).max().alias("diverge_max"),
        pl.col(ret).mean().alias("diverge_mean"),
        pl.col(fut_lots_col).first().alias("lots_fut"),
        pl.col(spot_lots_col).first().alias("lots_spot"),
        pl.col(fut_px_col).first().alias("first_fut_px"),
        # 進場四價(第一筆)：回 tick 對照、stats 層推進場價差/部位金額用
        pl.col("fut_ask").first().alias("entry_fut_ask"),
        pl.col("fut_bid").first().alias("entry_fut_bid"),
        pl.col("spot_ask").first().alias("entry_spot_ask"),
        pl.col("spot_bid").first().alias("entry_spot_bid"),
        # 進場那筆現貨的真實時間(E08)：對到的現貨報價多舊＝first_time − entry_spot_time
        *([pl.col("spot_time").first().alias("entry_spot_time")]
          if "spot_time" in tagged.columns else []),
        # 進場報價序號(回 tick 定位)
        *([pl.col("fut_chseq").first().alias("entry_fut_chseq")]
          if "fut_chseq" in tagged.columns else []),
        *([pl.col("spot_chseq").first().alias("entry_spot_chseq")]
          if "spot_chseq" in tagged.columns else []),
    ])

    # (2) 潛在部位 —— 股數配對法（同時處理兩件事）：
    #   a. 逐合約乘數 contract_size：標準2000、小型100、少數1000、ETF期10000
    #      （非標準合約如跨期已在 filter_standard_contract 剔除，故乘數必為整數）
    #   b. V05 修正：現貨 Lots 單位=張(1張=1000股)，不可與期貨口數直接 min
    #   配對：兩腳都要湊整 → 配對單位=兩邊都整除的最小股數
    #      標準2000→單位2000股(1口=2張)；小型100→單位1000股(10口=1張)；ETF10000→1口=10張
    plots, shares = _pair_lots(pl.col("lots_fut"), pl.col("lots_spot"), has_cs=has_cs)
    agg = agg.with_columns([
        plots.alias("potential_lots"),                                # 期貨口數
        (shares * pl.col("first_fut_px")).alias("potential_value"),   # 金額 = 股數 × 價
    ])

    # (2.5) 該合約「當天最後一筆 tick 四價」(close_*)：給結帳層做逐日浮虧評價用
    #   （每合約自己當天最後一筆，非全市場收盤時刻——冷門檔那刻可能沒報價）。
    #   從完整 tagged（含事件外的全天 tick）取每合約 last，broadcast 給該合約所有事件。
    close = (tagged.sort(TIME_COL).group_by(key).agg([
        pl.col("fut_bid").last().alias("close_fut_bid"),
        pl.col("fut_ask").last().alias("close_fut_ask"),
        pl.col("spot_bid").last().alias("close_spot_bid"),
        pl.col("spot_ask").last().alias("close_spot_ask"),
    ]))
    agg = agg.join(close, on=key, how="left")

    # (3) 進場側時間統計（與出場無關）。
    #     first_stretch 的 fallback 需要「同向歸零」當消失點，故算同向收斂的時間備用
    #     （A 出場本身已砍；此處僅借同向歸零時間當 first_stretch 的消失點 fallback）。
    opp_ret = _opp_ret_col(side)
    conv_same = _intraday_converged(tagged, agg, ret, opp_ret, key, direction="same")
    same_ct = conv_same.select(key, "event_id", "converge_time")
    agg = agg.join(same_ct, on=[key, "event_id"], how="left").with_columns([
        # signal_span_secs：整段跨度(第一筆→最後一筆仍≥閥值，含中間震盪跌破又回來)
        # fractional=True：保留次秒(浮點秒，到微秒)。IDC 微秒級執行，整數秒會把次秒閃現
        # 全壓成0、嚴重低估執行力 → 一律用 fractional 浮點秒。
        (pl.col("last_signal_time") - pl.col("first_time"))
            .dt.total_seconds(fractional=True).alias("signal_span_secs"),
        # first_stretch_secs：第一段≥閥值持續多久(到第一次明確<閥值的row為止；
        # 報價沒更新=掛著可打都算)。震盪波此值可小於 span；單tick即歸零用同向歸零當消失點。
        (pl.coalesce(pl.col("first_subthr_time"), pl.col("converge_time"))
            - pl.col("first_time"))
            .dt.total_seconds(fractional=True).alias("first_stretch_secs"),
    ]).drop("converge_time")   # 同向收斂時間只借來當 first_stretch 消失點，用完即丟

    # (3.5) 進場後 +30s/+60s 快照（趨勢用）：兩腿各跑到哪，期/現各帶實際時間戳供過濾冷報價
    for secs in SNAPSHOT_SECS:
        snap = _snapshot_after(tagged, agg, secs, key)
        agg = agg.join(snap, on=[key, "event_id"], how="left")

    agg = agg.with_columns(pl.lit(d2s).alias("days_to_settle"))

    # (4) 出場：反向收斂 B（ret_buy≥0，平倉那刀不倒貼）。每事件一列。
    #   A 同向收斂（早平倒貼）已砍——taker 出場保證倒貼、無實務意義（[[project-overnight-exit-zero]]）。
    #   出場半邊只存原始四價/序號/時間/converged；exit_ret、hold_secs 由 stats 從四價/時間推。
    conv_b = _intraday_converged(tagged, agg, ret, opp_ret, key, direction="opp")
    out = (agg.join(conv_b, on=[key, "event_id"], how="left")
              .with_columns([pl.col("converged").fill_null(False),
                             pl.lit(True).alias("is_first_entry")]))   # 第一次進場

    # (5) 二次進場：事件區間內、期貨成交驅動的後續進場點（並列成同質的「進場列」）。
    #   給了成交流＋現貨報價流才算；每筆二次進場一列(is_first_entry=False)，append 在錨之後。
    #   出場看錨（池模型：二次是錨的加碼、出場跟錨走）→ 二次列不各自判 converged。
    #   錨另帶 carry_second_lots = 該事件二次原始量總和（未×參與率，事實層不碰分析參數）：
    #     供「庫存滾錨」時把加碼量帶著走（滾量不滾列，避免幾萬筆二次列跨日滾雪球）。
    out = out.with_columns(pl.lit(0, dtype=pl.Int64).alias("carry_second_lots"))
    second = second_entry(out, fut_fills, spot_fills, side, key, threshold, has_cs=has_cs)
    if second is not None and second.height:
        # 每事件二次原始量總和 → 寫回錨列的 carry_second_lots
        carry = (second.group_by(key, "event_id")
                 .agg(pl.col("potential_lots").sum().alias("_carry")))
        out = (out.join(carry, on=[key, "event_id"], how="left")
                  .with_columns(
                      pl.when(pl.col("is_first_entry"))
                        .then(pl.col("_carry").fill_null(0))
                        .otherwise(0).alias("carry_second_lots"))
                  .drop("_carry"))
        out = pl.concat([out, second.select([c for c in out.columns if c in second.columns])],
                        how="diagonal")
        # 二次列經 diagonal 補的 null 統一補回：carry→0（只有錨帶籃子）、converged→False（出場看錨）
        out = out.with_columns([
            pl.col("carry_second_lots").fill_null(0),
            pl.col("converged").fill_null(False),
        ])
    return out.sort([key, "event_id", "is_first_entry"])


def second_entry(events: pl.DataFrame, fut_fills: pl.DataFrame | None,
                 spot_quotes: pl.DataFrame | None, side: str, key: str,
                 threshold: float, has_cs: bool = True) -> pl.DataFrame | None:
    """二次進場（價差賣）：事件區間內、期貨「成交」那刻回頭看現貨對手價重算價差，仍可獲利就再進場。

    全向量化、無迴圈：
      1. 期貨成交流 asof 往回貼「最近開始的事件」(first_time)，再 filter 落在 [first_time, last_signal_time]。
      2. 同一筆期貨成交再 asof 往回貼「最近一筆現貨對手報價」(spot_ask，60s tolerance；過時不對)。
      3. 重算價差 ret =(期貨成交價 − 現貨對手價)/期貨成交價 >= threshold → 仍可獲利才算二次進場。
      4. 量：期貨腿用「成交量 FillLots」、現貨腿用「對手掛量 spot_ask_lots」，股數配對 min（同第一次配對法）。
      5. 產出每筆二次進場一列，欄位對齊第一次進場（is_first_entry=False），出場/快照欄留空。
    僅做 SIDE_SELL（價差賣）；買向或缺成交流/報價流時回 None。
    """
    if side != SIDE_SELL or fut_fills is None or spot_quotes is None:
        return None
    if fut_fills.height == 0 or spot_quotes.height == 0 or events.height == 0:
        return None

    # 事件區間端點：每事件 first_time / last_signal_time / 進場門檻價（第一次進場列才有）。
    #   不帶 ValueCode：fut_fills 已自帶（供第二步現貨 asof 的 by），避免 join 後欄位撞名。
    ev = events.filter(pl.col("is_first_entry")).select(
        key, "event_id", "first_time", "last_signal_time",
        "days_to_settle",
    ).sort("first_time")

    # (1) 期貨成交 asof 往回貼「最近開始的事件」→ 再 filter 真的落在區間內
    fills = fut_fills.sort(TIME_COL)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sortedness of columns cannot be checked when 'by' groups provided")
        f = fills.join_asof(ev, left_on=TIME_COL, right_on="first_time",
                            by=key, strategy="backward")
    f = f.filter(
        pl.col("event_id").is_not_null()
        & (pl.col(TIME_COL) >= pl.col("first_time"))
        & (pl.col(TIME_COL) <= pl.col("last_signal_time"))
    )
    if f.height == 0:
        return None

    # (2) 該期貨成交再 asof 往回貼「最近一筆現貨對手報價」（60s tol，過時不對 → spot_ask null）
    sq = spot_quotes.sort(TIME_COL)
    f = f.sort(TIME_COL)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sortedness of columns cannot be checked when 'by' groups provided")
        f = f.join_asof(sq, on=TIME_COL, by="ValueCode",
                        strategy="backward", tolerance="60s")

    # (3) 重算價差（期貨成交價 − 現貨對手價）/期貨成交價 >= 門檻 → 仍可獲利
    ret2 = (pl.col("FillPrice") - pl.col("spot_ask")) / pl.col("FillPrice")
    f = f.filter(
        (pl.col("spot_ask") > 0) & (pl.col("FillPrice") > 0)
        & (pl.col("spot_ask_lots") > 0) & (pl.col("FillLots") > 0)
        & (ret2 >= threshold)
    )
    if f.height == 0:
        return None

    # (4) 量：期貨腿成交量 FillLots(口) vs 現貨腿對手掛量 spot_ask_lots(張)，股數配對 min（同第一次）
    plots, shares = _pair_lots(pl.col("FillLots"), pl.col("spot_ask_lots"), has_cs=has_cs)
    f = f.with_columns([
        ret2.alias("first_ret"),                          # 此次進場毛利率（成交 spread）
        pl.col("FillLots").alias("lots_fut"),             # 期貨腿量＝成交量(口)
        pl.col("spot_ask_lots").alias("lots_spot"),       # 現貨腿量＝對手掛量(張)
        pl.col("FillPrice").alias("entry_fut_bid"),       # 進場期貨價＝成交價
        pl.col("spot_ask").alias("entry_spot_ask"),       # 進場現貨價＝對手價
        plots.alias("potential_lots"),
        (shares * pl.col("FillPrice")).alias("potential_value"),
        pl.col(TIME_COL).alias("first_time"),             # 此次進場時間＝成交時間
        pl.lit(False).alias("is_first_entry"),
    ])
    # 出場欄(converged/converge_time/exit_*)由外層用 _intraday_converged 補（同第一次進場判法）。
    return f.filter(pl.col("potential_lots") > 0).select(
        [c for c in events.columns if c in f.columns])


def _intraday_converged(tagged: pl.DataFrame, agg: pl.DataFrame,
                        ret_col: str, opp_ret_col: str,
                        key: str = "ValueCode",
                        direction: str = "same") -> pl.DataFrame:
    """每事件：first_time 之後（同合約、到收盤）第一次「達到收斂條件」的時間（出場點）。
    並在該 tick 一併撈回出場資訊：反邊價差(出場成本)＋出場四價(A1/B1)。

    兩種出場方式（direction）：
      "same" 同向收斂：同邊 ret_col <= 0（期貨價差收掉＝可準備平倉，早平但反邊還倒貼）。
      "opp"  反向收斂：反邊 opp_ret_col >= 0（平倉那刀本身不倒貼＝晚平但出場成本翻正）。
    兩者都在該出場 tick 記「反邊 taker 價差(opp_ret_col)」當 exit_ret（語意統一）：
      同向出場時反邊通常為負(倒貼)，反向出場時反邊剛好 >=0。null 不算。

    用 forward as-of join（每事件往後找第一筆達標 tick），不可用事件×全列 join（OOM）。
    """
    # 出場 tick 清單：依 direction 取達標條件；只帶該 tick 的原始四價+序號+時間(純事實)。
    # exit_ret(反邊價差) 不在此存——stats 層從出場四價自推（[[feedback-fact-layer-raw-only]]）。
    cond = (pl.col(ret_col) <= 0) if direction == "same" else (pl.col(opp_ret_col) >= 0)
    has_chseq = "fut_chseq" in tagged.columns
    has_lots = "fut_ask_lots" in tagged.columns   # E08：出場側補存掛量(M03 流動性去重要這個)
    has_spot_time = "spot_time" in tagged.columns  # E08：出場那筆現貨真實時間
    zeroed = (tagged.filter(cond)
              .select(
                  key,
                  pl.col(TIME_COL).alias("converge_time"),
                  pl.col("fut_bid").alias("exit_fut_bid"),
                  pl.col("fut_ask").alias("exit_fut_ask"),
                  pl.col("spot_bid").alias("exit_spot_bid"),
                  pl.col("spot_ask").alias("exit_spot_ask"),
                  # 出場那筆現貨真實時間(E08)：converge_time 是期貨時間，此欄才是現貨時間
                  *([pl.col("spot_time").alias("exit_spot_time")] if has_spot_time else []),
                  # 出場掛量(E08)：四格深度都存原始，由 stats/驗算層自行挑該吃哪格([[feedback-fact-layer-raw-only]])
                  *([pl.col("fut_bid_lots").alias("exit_fut_bid_lots"),
                     pl.col("fut_ask_lots").alias("exit_fut_ask_lots"),
                     pl.col("spot_bid_lots").alias("exit_spot_bid_lots"),
                     pl.col("spot_ask_lots").alias("exit_spot_ask_lots")] if has_lots else []),
                  # 出場報價序號(回 tick 定位)
                  *([pl.col("fut_chseq").alias("exit_fut_chseq")] if has_chseq else []),
                  *([pl.col("spot_chseq").alias("exit_spot_chseq")] if has_chseq else []))
              .sort("converge_time"))
    firsts = agg.select(key, "event_id", "first_time").sort("first_time")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sortedness of columns cannot be checked when 'by' groups provided",
        )
        res = firsts.join_asof(
            zeroed,
            left_on="first_time", right_on="converge_time",
            by=key,
            strategy="forward",   # 往後找第一筆收斂
        )
    # exit_stretch_secs（只 opp/B 出場有意義）：收斂(ret_buy>=0)出現後，持續可平多久——
    #   = converge_time → 第一次 ret_buy 又跌回 <0 的秒數（對稱 first_stretch：達標→第一次跌破）。
    #   報價沒更新=可平的價仍掛著=持續，與進場側口徑一致。整段未再跌破→留 null（一路可平到收盤）。
    if direction == "opp":
        gone = (tagged.filter(pl.col(opp_ret_col) < 0)
                .select(key, pl.col(TIME_COL).alias("exit_gone_time"))
                .sort("exit_gone_time"))
        conv_sorted = res.filter(pl.col("converge_time").is_not_null()).sort("converge_time")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Sortedness of columns cannot be checked when 'by' groups provided",
            )
            conv_sorted = conv_sorted.join_asof(
                gone, left_on="converge_time", right_on="exit_gone_time",
                by=key, strategy="forward",   # 收斂後往後找第一筆又跌破(<0)
            )
        res = res.join(
            conv_sorted.select(key, "event_id", "exit_gone_time"),
            on=[key, "event_id"], how="left",
        ).with_columns(
            (pl.col("exit_gone_time") - pl.col("converge_time"))
                .dt.total_seconds(fractional=True).alias("exit_stretch_secs")
        )
    else:
        res = res.with_columns(pl.lit(None, dtype=pl.Float64).alias("exit_stretch_secs"))
    exit_time_cols = (["exit_spot_time"] if has_spot_time else [])
    exit_lot_cols = (["exit_fut_bid_lots", "exit_fut_ask_lots",
                      "exit_spot_bid_lots", "exit_spot_ask_lots"] if has_lots else [])
    exit_seq_cols = (["exit_fut_chseq", "exit_spot_chseq"] if has_chseq else [])
    return res.with_columns(
        pl.col("converge_time").is_not_null().alias("converged")
    ).select(key, "event_id", "converged", "converge_time", "exit_stretch_secs",
             "exit_fut_bid", "exit_fut_ask", "exit_spot_bid", "exit_spot_ask",
             *exit_time_cols, *exit_lot_cols, *exit_seq_cols)


def carryover_converge(inv: pl.DataFrame, spreads: pl.DataFrame, side: str
                       ) -> tuple[pl.DataFrame, pl.DataFrame]:
    """庫存層：舊庫存(昨天沒收斂的 fill)搭今天的 spreads 重判收斂。

    回 (今天平掉的=closed, 今天還沒平的=still)。
    庫存當「今天的部位」：first_time=今日第一筆 tick，從今天頭找反邊翻正(B 出場)。
    複用 _intraday_converged（向量化 forward as-of，不逐筆）；不碰 tag_events、不造假 tick。
    平掉的帶今天的出場四價（損益看當下，由 stats 推）；沒平的回原 fill 欄位、滾到明天。
    """
    if inv.height == 0:
        return inv.head(0), inv.head(0)
    key = "QuoteCode" if "QuoteCode" in spreads.columns else "ValueCode"
    ret, opp = _ret_col(side), _opp_ret_col(side)
    # 同合約(key)今天的收斂判定相同（都從今天開盤找該合約反邊翻正第一筆）→ 每個合約只算一次，
    #   再 broadcast 回該合約的所有庫存部位。用 key 算（非 key+event_id：跨日滾動 event_id 會重複，
    #   一對多 join 會笛卡爾爆炸）。
    tu = spreads.schema[TIME_COL].time_unit
    t0 = spreads[TIME_COL].min()
    agg = (inv.select(key).unique()                       # 每合約一列
              .with_columns([pl.lit(0).alias("event_id"),  # _intraday_converged 需要 event_id 欄
                             pl.lit(t0).cast(pl.Datetime(tu)).alias("first_time")]))
    conv = _intraday_converged(spreads, agg, ret, opp, key=key, direction="opp").drop("event_id")
    # 今天該合約最後一筆四價（close_*）：滾來的庫存要用「今天的收盤」評價今天浮虧，非原進場日的。
    close = (spreads.sort(TIME_COL).group_by(key).agg([
        pl.col("fut_bid").last().alias("close_fut_bid"),
        pl.col("fut_ask").last().alias("close_fut_ask"),
        pl.col("spot_bid").last().alias("close_spot_bid"),
        pl.col("spot_ask").last().alias("close_spot_ask"),
    ]))
    conv = conv.join(close, on=key, how="left")
    # conv 帶今天的出場欄(converged/converge_time/exit_*)+今天 close_*，與 inv 同名。
    # 把 inv 的舊出場欄+舊 close 丟掉、換成今天的 → 進場側維持原始事實、出場/評價側更新成今天。
    exit_cols = [c for c in conv.columns if c != key]
    merged = (inv.drop([c for c in exit_cols if c in inv.columns])
                 .join(conv, on=key, how="left")           # 用 key broadcast 回所有庫存部位
                 .select(inv.columns))
    closed = merged.filter(pl.col("converged") == True)   # noqa: E712 今天平掉
    still = merged.filter(pl.col("converged") != True)    # 今天仍沒平，滾明天
    return closed, still


# 進場後要拍快照的秒數（趨勢用：看進場後 N 秒兩腿各跑到哪）
SNAPSHOT_SECS = [30, 60]


def _snapshot_after(tagged: pl.DataFrame, agg: pl.DataFrame, secs: int,
                    key: str = "ValueCode") -> pl.DataFrame:
    """每事件：進場時間 + secs 秒「之後第一筆」報價的期現 A1/B1（forward as-of）。

    趨勢用——進場存一張、+30s/+60s 各存一張，就能畫出兩腿在這幾個時點各跑到哪。
    期、現各自對各自的 tick 流抽（更新頻率不同），各自帶回「實際時間戳」，
    供事後過濾冷報價（如 s30_time==s60_time＝兩快照同一筆；或時間戳遠超目標＝進場後沒新報價）。
    抽法 forward：抓「時間 >= 進場+secs」的第一筆，保證不早於目標時間。
    """
    p = f"s{secs}"
    # 目標時間 = 進場 + secs 秒。pl.duration 產生 μs，與 tick 流時間欄(ns)型別不符會讓
    # join_asof 報 SchemaError → cast 回 tick 流的時間單位對齊。
    time_unit = tagged.schema[TIME_COL].time_unit
    targets = agg.select(key, "event_id", "first_time").with_columns(
        (pl.col("first_time") + pl.duration(seconds=secs))
        .cast(pl.Datetime(time_unit)).alias("_target")
    ).sort("_target")

    # 期貨腳：該合約全 tick（含四價/時間），forward 找 >= target 第一筆
    fut_stream = (tagged.select(
        key,
        pl.col(TIME_COL).alias(f"{p}_fut_time"),
        pl.col("fut_bid").alias(f"{p}_fut_bid"),
        pl.col("fut_ask").alias(f"{p}_fut_ask"))
        .sort(f"{p}_fut_time"))
    # 現貨腳：現貨四價是「上一刻現貨」掛在每個期貨 tick 上（現貨更新慢，同筆現貨會掛在多個
    #   連續期貨 tick 上）。對齊仍用期貨時間 _spot_align_t（＝該現貨報價被掛上來的時點），
    #   但 s{p}_spot_time 回報「那筆現貨的真實時間 spot_time」(E08)——這樣才量得出現貨 staleness：
    #   s{p}_spot_time 落後 _target 越多＝該時點生效的現貨報價越舊。無 spot_time(舊資料)則退回期貨時間。
    has_spot_time = "spot_time" in tagged.columns
    spot_time_expr = (pl.col("spot_time") if has_spot_time else pl.col(TIME_COL))
    spot_stream = (tagged.select(
        key,
        pl.col(TIME_COL).alias("_spot_align_t"),          # 對齊用：現貨掛上來的期貨 tick 時間
        spot_time_expr.alias(f"{p}_spot_time"),           # 回報用：現貨報價真實時間
        pl.col("spot_bid").alias(f"{p}_spot_bid"),
        pl.col("spot_ask").alias(f"{p}_spot_ask"))
        .sort("_spot_align_t"))

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sortedness of columns cannot be checked when 'by' groups provided",
        )
        res = targets.join_asof(
            fut_stream, left_on="_target", right_on=f"{p}_fut_time",
            by=key, strategy="forward",
        ).sort("_target").join_asof(
            spot_stream, left_on="_target", right_on="_spot_align_t",
            by=key, strategy="forward",
        )
    return res.select(
        key, "event_id",
        f"{p}_fut_bid", f"{p}_fut_ask", f"{p}_spot_bid", f"{p}_spot_ask",
        f"{p}_fut_time", f"{p}_spot_time",
    )
