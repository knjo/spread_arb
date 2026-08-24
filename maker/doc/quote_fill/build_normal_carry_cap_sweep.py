"""Publish the normal-carry close-mark and exit-only cap comparison bundle.

The immutable upstream input is the fully priced supplemental carry v3 bundle.
This harness composes the existing transaction-cost API with the chronological
portfolio-cap API and publishes two directly comparable entry cutoffs:

* 13:00 -- the normal-carry leg of the user's two-policy comparison;
* 13:20 -- the original research-session reference.

Expiry terminals use paired same-day close prices. A product carried into a
session is exit-only for that complete session, independently of the D-safe
universe gate.
"""

from __future__ import annotations

import argparse
from datetime import time
import hashlib
import json
from pathlib import Path
import time as wall_time
from typing import Mapping, Sequence

import polars as pl

from maker.src.quote_fill.combined_cost_cap_sweep import (
    CombinedCapConfig,
    TransactionCostProfile,
    build_combined_cost_cap_sweep,
)
from maker.src.quote_fill.portfolio_cap_backtester import (
    DEFAULT_HARD_INTRADAY_CAPS_TWD,
    PortfolioCapBacktestConfig,
    PortfolioCapBacktestResult,
    PortfolioCapScenario,
    backtest_priced_paths,
)
from maker.src.quote_fill.supplemental_carry_runner import (
    DEFAULT_PREREQUISITE_ROOT,
)


VERSION = "normal_carry_cap_sweep_v2_expiry_close_opening_carry_exit_only"
UPSTREAM_VERSION = "supplemental_imputed_full_carry_grouped_runner_v3_expiry_daily_close"
DEFAULT_UPSTREAM = Path(
    "maker/data/walkforward/"
    "prequential_challenger_ab12_supplemental_carry_20260821_v3_expiry_daily_close"
)
DEFAULT_OUTPUT = Path(
    "maker/data/walkforward/normal_carry_cap_sweep_20260821_v2_expiry_close_exit_only"
)
SENSITIVITY_ROOT = Path(
    "maker/data/walkforward/supplemental_sampling_sensitivity_20260821_v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def verify_upstream(
    root: Path,
) -> tuple[dict[str, object], dict[str, pl.DataFrame]]:
    """Verify every declared upstream artifact before using it."""

    marker_path = root / "complete.json"
    if not marker_path.is_file():
        raise FileNotFoundError(f"upstream complete marker is missing: {marker_path}")
    marker = _read_json(marker_path)
    if marker.get("complete") is not True:
        raise ValueError("upstream bundle is incomplete")
    if marker.get("runner_version") != UPSTREAM_VERSION:
        raise ValueError(f"unexpected upstream version: {marker.get('runner_version')}")
    declarations = marker.get("artifacts")
    if not isinstance(declarations, dict):
        raise ValueError("upstream artifact declarations are missing")
    required = (
        "supplemental_paths.parquet",
        "continuation_audit.parquet",
        "expiry_marks.parquet",
        "source_inventory.parquet",
        "candidate_session_sampling_audit.parquet",
        "daily_close_facts.parquet",
        "expiry_close_overlay_audit.parquet",
    )
    frames: dict[str, pl.DataFrame] = {}
    for filename in required:
        declaration = declarations.get(filename)
        path = root / filename
        if not isinstance(declaration, dict) or not path.is_file():
            raise ValueError(f"upstream artifact is missing: {filename}")
        if _sha256(path) != declaration.get("sha256"):
            raise ValueError(f"upstream artifact hash changed: {filename}")
        frame = pl.read_parquet(path)
        if (
            frame.height != int(declaration.get("rows", -1))
            or frame.width != int(declaration.get("columns", -1))
        ):
            raise ValueError(f"upstream artifact dimensions changed: {filename}")
        frames[filename] = frame
    paths = frames["supplemental_paths.parquet"]
    if paths.shape != (3672, 94) or paths["policy_path_id"].n_unique() != 3672:
        raise ValueError("upstream supplemental path population changed")
    if paths.filter(
        (pl.col("terminal_cashflow_priced") != True).fill_null(True)  # noqa: E712
    ).height:
        raise ValueError("upstream v3 is no longer fully priced")
    return marker, frames


def audit_expiry_close(
    paths: pl.DataFrame,
    continuation_audit: pl.DataFrame,
    expiry_marks: pl.DataFrame,
) -> pl.DataFrame:
    """Validate paired daily closes and retain the five former QLFG6 fallbacks."""

    expiry = paths.filter(
        pl.col("supplemental_terminal_resolution")
        == "expiry_same_day_two_leg_close_price"
    )
    if expiry.height != 171:
        raise ValueError(f"expected 171 expiry-close paths, got {expiry.height}")
    if expiry.filter(
        (pl.col("expiry_mark_is_official_close") != True)  # noqa: E712
        | (pl.col("expiry_mark_is_official_settlement") != False)  # noqa: E712
        | (pl.col("expiry_mark_uses_trade_fallback") != False)  # noqa: E712
    ).height:
        raise ValueError("upstream expiry-close path flags are incoherent")
    former_fallback = expiry.filter(
        (pl.col("Date") == "20260714")
        & (pl.col("terminal_date") == "20260715")
        & (pl.col("ValueCode") == "3374")
        & (pl.col("QuoteCode") == "QLFG6")
    )
    if former_fallback.height != 5:
        raise ValueError("the five former QLFG6 fallback paths changed")
    if expiry_marks.height != 45 or expiry_marks.filter(
        (pl.col("mark_is_official_close") != True)  # noqa: E712
        | (pl.col("mark_is_official_settlement") != False)  # noqa: E712
        | (pl.col("mark_uses_trade_fallback") != False)  # noqa: E712
    ).height:
        raise ValueError("upstream paired expiry close marks are incomplete")
    mark = expiry_marks.filter(
        (pl.col("ValueCode") == "3374") & (pl.col("QuoteCode") == "QLFG6")
    )
    if mark.height != 1:
        raise ValueError("3374/QLFG6 daily close mark is not unique")
    mark_row = mark.row(0, named=True)
    if (
        mark_row["spot_close_source"]
        != "MarketInfo.twse_security_trades_daily.close_price"
        or mark_row["future_close_source"]
        != "MarketInfo.taifex_futures_trades_daily.close_price"
        or mark_row["mark_is_official_close"] is not True
        or mark_row["mark_is_official_settlement"] is not False
    ):
        raise ValueError("3374/QLFG6 paired-close semantics changed")
    audit_rows = continuation_audit.filter(
        pl.col("policy_path_id").is_in(former_fallback["policy_path_id"].to_list())
    )
    if audit_rows.height != 5 or audit_rows.filter(
        pl.col("blocker_status").is_not_null()
        | (pl.col("terminal_resolution") != "expiry_same_day_two_leg_close_price")
    ).height:
        raise ValueError("former-fallback continuation audit is incomplete")
    mark_columns = [
        column
        for column in mark.columns
        if column not in {"Date", "ValueCode", "QuoteCode"}
    ]
    audit_columns = [
        "policy_path_id",
        "sessions_replayed",
        "imputed_unknown_sessions",
        "terminal_resolution",
        "blocker_status",
        "blocker_detail",
        "state_sampling_approximate",
        "target_cancel_clock_exact",
        "delayed_hedge_snapshot_exact",
    ]
    return (
        former_fallback.join(
            audit_rows.select(audit_columns),
            on="policy_path_id",
            how="left",
            validate="1:1",
            suffix="_audit",
        )
        .join(
            mark.select("ValueCode", "QuoteCode", *mark_columns),
            on=["ValueCode", "QuoteCode"],
            how="left",
            validate="m:1",
            suffix="_mark",
        )
        .sort(["position_established_ns", "policy_path_id"])
    )


def _sessions_for(paths: pl.DataFrame) -> tuple[str, ...]:
    all_sessions = (
        DEFAULT_PREREQUISITE_ROOT.joinpath("candidate_sessions.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    first = str(paths["Date"].min())
    last = str(paths["terminal_date"].max())
    sessions = tuple(value for value in all_sessions if first <= value <= last)
    if not sessions or sessions[0] != first or sessions[-1] != last:
        raise ValueError("trading calendar does not cover all terminal paths")
    return sessions


def _scenarios(prefix: str) -> tuple[PortfolioCapScenario, ...]:
    return tuple(
        PortfolioCapScenario(
            scenario_id=f"{prefix}_hard_{int(cap)}",
            hard_intraday_cap_twd=float(cap),
        )
        for cap in DEFAULT_HARD_INTRADAY_CAPS_TWD
    )


def _run_cap(
    priced: pl.DataFrame,
    *,
    sessions: Sequence[str],
    variant: str,
    cutoff: time,
) -> PortfolioCapBacktestResult:
    result = backtest_priced_paths(
        priced,
        config=PortfolioCapBacktestConfig(
            scenarios=_scenarios(variant),
            per_product_fraction=0.30,
            entry_cutoff_local_time=cutoff,
            session_open_local_time=time(9, 0),
            session_close_local_time=time(13, 20),
            timezone_name="Asia/Taipei",
            block_new_entries_for_products_held_at_session_open=True,
        ),
        session_dates=sessions,
    )
    cutoff_text = cutoff.isoformat()
    return PortfolioCapBacktestResult(
        events=result.events.with_columns(
            pl.lit(variant).alias("entry_cutoff_variant"),
            pl.lit(cutoff_text).alias("entry_cutoff_local_time"),
        ),
        daily=result.daily.with_columns(
            pl.lit(variant).alias("entry_cutoff_variant"),
            pl.lit(cutoff_text).alias("entry_cutoff_local_time"),
        ),
        summary=result.summary.with_columns(
            pl.lit(variant).alias("entry_cutoff_variant"),
            pl.lit(cutoff_text).alias("entry_cutoff_local_time"),
        ),
    )


def _concat_caps(
    main: PortfolioCapBacktestResult,
    reference: PortfolioCapBacktestResult,
) -> PortfolioCapBacktestResult:
    return PortfolioCapBacktestResult(
        events=pl.concat((main.events, reference.events), how="diagonal_relaxed"),
        daily=pl.concat((main.daily, reference.daily), how="diagonal_relaxed"),
        summary=pl.concat((main.summary, reference.summary), how="diagonal_relaxed"),
    )


def _holding_facts(
    paths: pl.DataFrame,
    events: pl.DataFrame,
    sessions: Sequence[str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    session_index = {date: index for index, date in enumerate(sessions)}
    rows: list[dict[str, object]] = []
    for row in paths.select(
        "policy_path_id",
        "Date",
        "terminal_date",
        "position_established_ns",
        "exit_decision_time_ns",
        "supplemental_terminal_resolution",
        "model_imputed_full_carry_on_unknown",
        "expiry_mark_uses_trade_fallback",
    ).iter_rows(named=True):
        entry_date = str(row["Date"])
        terminal_date = str(row["terminal_date"])
        rows.append(
            {
                **row,
                "derived_holding_session_boundaries": (
                    session_index[terminal_date] - session_index[entry_date]
                ),
                "elapsed_holding_hours": (
                    int(row["exit_decision_time_ns"])
                    - int(row["position_established_ns"])
                )
                / 3_600_000_000_000.0,
            }
        )
    path_holding = pl.from_dicts(rows, infer_schema_length=None)
    accepted = events.filter(
        (pl.col("event_type") == "entry_candidate")
        & (pl.col("entry_admitted") == True)  # noqa: E712
    ).select(
        "scenario_id",
        "entry_cutoff_variant",
        "entry_cutoff_local_time",
        "policy_path_id",
    )
    accepted_holding = accepted.join(
        path_holding, on="policy_path_id", how="left", validate="m:1"
    ).sort(["entry_cutoff_variant", "scenario_id", "Date", "position_established_ns"])
    summary = accepted_holding.group_by("scenario_id").agg(
        pl.col("elapsed_holding_hours").mean().alias("holding_hours_mean"),
        pl.col("elapsed_holding_hours").quantile(0.50).alias("holding_hours_p50"),
        pl.col("elapsed_holding_hours").quantile(0.90).alias("holding_hours_p90"),
        pl.col("elapsed_holding_hours").quantile(0.95).alias("holding_hours_p95"),
        pl.col("elapsed_holding_hours").max().alias("holding_hours_max"),
        pl.col("derived_holding_session_boundaries")
        .mean()
        .alias("holding_session_boundaries_mean"),
        pl.col("derived_holding_session_boundaries")
        .quantile(0.50)
        .alias("holding_session_boundaries_p50"),
        pl.col("derived_holding_session_boundaries")
        .quantile(0.90)
        .alias("holding_session_boundaries_p90"),
        pl.col("derived_holding_session_boundaries")
        .max()
        .alias("holding_session_boundaries_max"),
        (pl.col("supplemental_terminal_resolution") == "source_completed")
        .sum()
        .alias("accepted_source_completed_paths"),
        (
            pl.col("supplemental_terminal_resolution")
            == "normal_continuation_replay"
        )
        .sum()
        .alias("accepted_approx_normal_continuation_paths"),
        (
            pl.col("supplemental_terminal_resolution")
            == "expiry_same_day_two_leg_close_price"
        )
        .sum()
        .alias("accepted_expiry_daily_close_paths"),
        pl.col("expiry_mark_uses_trade_fallback")
        .sum()
        .alias("accepted_trade_fallback_expiry_paths"),
        pl.col("model_imputed_full_carry_on_unknown")
        .sum()
        .alias("accepted_full_carry_imputed_paths"),
    )
    return accepted_holding, summary


def _augment_summary(
    summary: pl.DataFrame,
    holding: pl.DataFrame,
) -> pl.DataFrame:
    return (
        summary.join(holding, on="scenario_id", how="left", validate="1:1")
        .with_columns(
            pl.lit(3672).alias("source_fully_priced_paths"),
            pl.lit(5).alias("former_unresolved_3374_close_paths"),
            pl.lit(0).alias("unresolved_paths"),
            (
                pl.col("realized_gross_pnl_twd")
                / pl.col("accepted_entry_one_way_turnover_twd")
                * 10_000.0
            ).alias("realized_gross_bp_on_entry_turnover"),
            (
                pl.col("realized_transaction_cost_twd")
                / pl.col("accepted_entry_one_way_turnover_twd")
                * 10_000.0
            ).alias("realized_cost_bp_on_entry_turnover"),
            (
                pl.col("realized_net_pnl_twd")
                / pl.col("accepted_entry_one_way_turnover_twd")
                * 10_000.0
            ).alias("realized_net_bp_on_entry_turnover"),
            pl.lit("one_second_plus_spread_pair_epoch_approx")
            .alias("supplemental_normal_replay_precision"),
            pl.lit(True).alias("unknown_full_carry_assumption_present"),
            pl.lit(True).alias("double_exit_bias_possible"),
            pl.lit(True).alias("opening_carry_product_exit_only_policy"),
            pl.lit(False).alias("source_universe_d_safe"),
            pl.lit(False).alias("production_strategy_go"),
            pl.lit(True).alias("provisional_analysis_only"),
            pl.when(pl.col("entry_cutoff_variant") == "normal_cutoff_1300")
            .then(pl.lit(0))
            .otherwise(pl.lit(1))
            .alias("_variant_order"),
        )
        .sort(["_variant_order", "hard_intraday_cap_twd"])
        .drop("_variant_order")
    )


def _cutoff_comparison(summary: pl.DataFrame) -> pl.DataFrame:
    metrics = (
        "accepted_paths",
        "rejected_paths",
        "rejected_opening_carry_exit_only",
        "accepted_entry_one_way_turnover_twd",
        "peak_intraday_one_way_notional_twd",
        "mean_time_weighted_intraday_one_way_notional_twd",
        "peak_eod_outstanding_one_way_notional_twd",
        "mean_eod_outstanding_one_way_notional_twd",
        "realized_gross_pnl_twd",
        "realized_transaction_cost_twd",
        "realized_net_pnl_twd",
        "losing_exits",
        "negative_realized_days",
        "max_realized_drawdown_twd",
    )
    main = summary.filter(
        pl.col("entry_cutoff_variant") == "normal_cutoff_1300"
    ).select(
        "hard_intraday_cap_twd",
        *(pl.col(metric).alias(f"{metric}_1300") for metric in metrics),
    )
    reference = summary.filter(
        pl.col("entry_cutoff_variant") == "original_cutoff_1320_reference"
    ).select(
        "hard_intraday_cap_twd",
        *(pl.col(metric).alias(f"{metric}_1320") for metric in metrics),
    )
    result = main.join(
        reference, on="hard_intraday_cap_twd", how="inner", validate="1:1"
    )
    return result.with_columns(
        *(
            (pl.col(f"{metric}_1300") - pl.col(f"{metric}_1320")).alias(
                f"{metric}_delta_1300_minus_1320"
            )
            for metric in metrics
        )
    ).sort("hard_intraday_cap_twd")


def _source_outcome_audit(paths: pl.DataFrame) -> pl.DataFrame:
    return (
        paths.group_by(
            "supplemental_terminal_resolution",
            "source_outcome_status",
            "model_imputed_full_carry_on_unknown",
            "double_exit_bias_possible",
            "expiry_mark_uses_trade_fallback",
        )
        .agg(
            pl.len().alias("paths"),
            pl.col("normalization_notional_twd").sum().alias("one_way_notional_twd"),
            pl.col("gross_cycle_pnl_twd").sum().alias("gross_cycle_pnl_twd"),
        )
        .sort("paths", descending=True)
    )


def _money(value: object) -> str:
    return f"{float(value):,.0f}"


def _money_m(value: object) -> str:
    return f"{float(value) / 1_000_000:.3f}"


def _table(summary: pl.DataFrame, variant: str) -> list[str]:
    rows = summary.filter(pl.col("entry_cutoff_variant") == variant)
    lines = [
        "| Cap M | Accept / reject | Exit-only rejects | Entry turnover M | Peak / TW mean M | Peak / mean EOD M | Gross / cost / net TWD | Loss trades / days | Realized MDD | Hold p50 / p90 sessions |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows.iter_rows(named=True):
        lines.append(
            f"| {_money_m(row['hard_intraday_cap_twd'])} | "
            f"{row['accepted_paths']} / {row['rejected_paths']} | "
            f"{row['rejected_opening_carry_exit_only']} | "
            f"{_money_m(row['accepted_entry_one_way_turnover_twd'])} | "
            f"{_money_m(row['peak_intraday_one_way_notional_twd'])} / "
            f"{_money_m(row['mean_time_weighted_intraday_one_way_notional_twd'])} | "
            f"{_money_m(row['peak_eod_outstanding_one_way_notional_twd'])} / "
            f"{_money_m(row['mean_eod_outstanding_one_way_notional_twd'])} | "
            f"{_money(row['realized_gross_pnl_twd'])} / "
            f"{_money(row['realized_transaction_cost_twd'])} / "
            f"{_money(row['realized_net_pnl_twd'])} | "
            f"{row['losing_exits']} / {row['negative_realized_days']} | "
            f"{_money(row['max_realized_drawdown_twd'])} | "
            f"{float(row['holding_session_boundaries_p50']):.0f} / "
            f"{float(row['holding_session_boundaries_p90']):.0f} |"
        )
    return lines


def render_readme(
    summary: pl.DataFrame,
    cost_summary: pl.DataFrame,
    source_audit: pl.DataFrame,
    fallback_audit: pl.DataFrame,
    sessions: Sequence[str],
    sensitivity: Mapping[str, object] | None,
) -> str:
    cost = cost_summary.filter(pl.col("summary_group") == "all_completed").row(
        0, named=True
    )
    fallback_notional = float(fallback_audit["normalization_notional_twd"].sum())
    if sensitivity:
        sensitivity_line = (
            f"15 組 exact sensitivity：terminal/carry 分類一致 "
            f"{100 * float(sensitivity['classification_agreement_rate']):.1f}%，"
            f"gross PnL bp 完全一致 "
            f"{100 * float(sensitivity['exact_gross_pnl_bp_match_rate_among_both_terminal']):.1f}%，"
            f"|PnL 誤差| p95={float(sensitivity['abs_gross_pnl_bp_error_p95']):.2f} bp。"
        )
    else:
        sensitivity_line = "本 bundle 未找到獨立 exact sensitivity artifact。"
    lines = [
        "# Normal carry cap sweep — expiry close and opening-carry exit-only",
        "",
        "正常留倉版共發布兩個 entry cutoff：13:00 是與 13:00 積極平倉版直接比較的主表；"
        "13:20 是原研究 session 的 reference。兩者都使用相同 terminal paths、成本與 cap。"
        "這裡的 13:00 只停止新 entry，不改寫既有 upstream exit terminal。若商品在 D 日"
        "開盤已有前日 carry，D 日整天該商品只出不進；即使早盤已平完，也不再開新倉。",
        "",
        "## 不能略過的限制",
        "",
        f"- 1,090 筆 normal continuation 使用每秒末狀態＋每次 SpreadPair epoch 變化近似；"
        f"raw trades 保留。{sensitivity_line}",
        "- 原始 maker-fill unknown 被假設為當日零出場成交、完整部位 carry；真實可能已部分或"
        "全部出場，所以有 double-exit bias。",
        "- 171 筆到期路徑使用該日現貨與期貨 daily close_price 強制平倉；這是收盤 mark，"
        "不是可成交 BBO。期貨 settlement_price 未使用。原 3374/QLFG6 的 5 筆 402/402 "
        "fallback 已由 paired close 取代。",
        "- 『開盤有 carry 商品當日 exit-only』是依各 cap scenario 已接受持倉動態判定，與"
        "D-safe 商品池 gate 是兩個獨立規則；events/daily/summary 都有獨立 blocker/count。",
        "- 固定 45 檔商品 cohort 是回頭篩選，尚非 D-1 dynamic universe，有 data leak；"
        "這份結果不能直接部署。",
        "- MDD 是每日已實現現金流 drawdown，沒有將未實現 carry 每日 mark-to-market。",
        "- 13:00 主表是 entry-cutoff-only：原本在 13:00–13:20 成交的 normal exit 仍可"
        "同日出場；它不是『13:00 後禁止出場、強制全部隔夜』的另一個 counterfactual。",
        "",
        "## 全路徑、尚未套 cap",
        "",
        f"- {sessions[0]}–{sessions[-1]}，{len(sessions)} 個交易日；3,672/3,672 priced。",
        f"- Gross / 成本 / Net = {_money(cost['gross_cycle_pnl_twd'])} / "
        f"{_money(cost['total_transaction_cost_twd'])} / "
        f"{_money(cost['net_cycle_pnl_twd'])} TWD。",
        f"- 名目加權 gross / cost / net = "
        f"{float(cost['notional_weighted_gross_bp']):.2f} / "
        f"{float(cost['notional_weighted_effective_cost_bp']):.2f} / "
        f"{float(cost['notional_weighted_net_bp']):.2f} bp。",
        f"- 5 筆 former unresolved 保留在母體，one-way notional={fallback_notional / 1e6:.3f}M；"
        "現在也使用 paired daily close，細節在 `former_unresolved_3374_close_paths.parquet`。",
        "",
        "## 主比較：13:00 停止新 entry（upstream normal exits 不改）",
        "",
        "Cap 與 exposure 單位是 one-way spot entry notional；單品 cap=總 cap 的 30%。"
        "EOD snapshot 是 13:20 台北時間。表中的 accept/reject 已套用開盤 carry 商品"
        "當日 exit-only。",
        "",
        *_table(summary, "normal_cutoff_1300"),
        "",
        "## Reference：原策略 13:20 停止新倉",
        "",
        *_table(summary, "original_cutoff_1320_reference"),
        "",
        "13:00−13:20 的逐 cap delta 在 `cutoff_comparison.parquet`；每日 turnover、"
        "peak/time-weighted/EOD exposure 在 `cap_daily.parquet`；完整 loss、MDD 與 holding "
        "欄位在 `cap_summary.parquet`。",
        "",
        "## Source outcome population",
        "",
        "| Resolution | Source outcome | Paths | One-way notional M | Gross TWD |",
        "|---|---|---:|---:|---:|",
    ]
    for row in source_audit.iter_rows(named=True):
        lines.append(
            f"| {row['supplemental_terminal_resolution']} | "
            f"{row['source_outcome_status']} | {row['paths']} | "
            f"{_money_m(row['one_way_notional_twd'])} | "
            f"{_money(row['gross_cycle_pnl_twd'])} |"
        )
    lines.extend(
        [
            "",
            "所有 upstream artifacts 在讀取前均依 `complete.json` 驗證 SHA-256；輸出也保存"
            " upstream marker hash、expiry marks、continuation audit 與 source inventory。",
            "",
        ]
    )
    return "\n".join(lines)


def _artifact_record(path: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if path.suffix == ".parquet":
        frame = pl.read_parquet(path)
        result.update({"rows": frame.height, "columns": frame.width})
    return result


def run(upstream_root: Path, output: Path) -> pl.DataFrame:
    started = wall_time.perf_counter()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    upstream_marker, frames = verify_upstream(upstream_root)
    paths = frames["supplemental_paths.parquet"]
    continuation_audit = frames["continuation_audit.parquet"]
    expiry_marks = frames["expiry_marks.parquet"]
    fallback_audit = audit_expiry_close(paths, continuation_audit, expiry_marks)
    sessions = _sessions_for(paths)

    combined = build_combined_cost_cap_sweep(
        paths,
        config=CombinedCapConfig(
            portfolio_caps_twd=tuple(DEFAULT_HARD_INTRADAY_CAPS_TWD),
            per_product_fraction=0.30,
            cost_profile=TransactionCostProfile(),
        ),
    )
    path_costs = combined.path_costs.join(
        paths.select(
            "policy_path_id",
            "supplemental_terminal_resolution",
            "model_imputed_full_carry_on_unknown",
            "double_exit_bias_possible",
            "expiry_mark_uses_trade_fallback",
        ),
        on="policy_path_id",
        how="left",
        validate="1:1",
    ).with_columns(
        pl.when(
            pl.col("supplemental_terminal_resolution")
            == "normal_continuation_replay"
        )
        .then(pl.lit("one_second_plus_spread_pair_epoch_approx"))
        .when(pl.col("supplemental_terminal_resolution").str.starts_with("expiry_"))
        .then(pl.lit("same_day_two_leg_daily_close_not_executable_bbo"))
        .otherwise(pl.lit("formal_source_path_price"))
        .alias("terminal_price_precision_role")
    )
    if path_costs.filter(
        pl.col("transaction_cost_point_identified") != True  # noqa: E712
    ).height:
        raise ValueError("cost API did not price all 3,672 paths")

    main = _run_cap(
        path_costs,
        sessions=sessions,
        variant="normal_cutoff_1300",
        cutoff=time(13, 0),
    )
    reference = _run_cap(
        path_costs,
        sessions=sessions,
        variant="original_cutoff_1320_reference",
        cutoff=time(13, 20),
    )
    cap = _concat_caps(main, reference)
    accepted_holding, holding_summary = _holding_facts(paths, cap.events, sessions)
    summary = _augment_summary(cap.summary, holding_summary)
    comparison = _cutoff_comparison(summary)
    source_audit = _source_outcome_audit(paths)
    if summary.filter(
        (pl.col("completed_exits") != pl.col("accepted_paths"))
        | (pl.col("unresolved_paths") != 0)
    ).height:
        raise AssertionError("priced terminal/cap completeness invariant failed")
    if summary.filter(
        (pl.col("entry_cutoff_variant") == "normal_cutoff_1300")
        & (pl.col("rejected_entry_cutoff") != 146)
    ).height:
        raise AssertionError("13:00 source cutoff population changed")
    if summary.filter(
        (pl.col("entry_cutoff_variant") == "original_cutoff_1320_reference")
        & (pl.col("rejected_entry_cutoff") != 0)
    ).height:
        raise AssertionError("13:20 reference unexpectedly rejected cutoff entries")

    sensitivity: dict[str, object] | None = None
    sensitivity_path = SENSITIVITY_ROOT / "summary.json"
    if sensitivity_path.is_file():
        sensitivity = _read_json(sensitivity_path)

    output.mkdir(parents=True, exist_ok=False)
    artifacts: dict[str, pl.DataFrame] = {
        "upstream_supplemental_paths.parquet": paths,
        "upstream_continuation_audit.parquet": continuation_audit,
        "upstream_expiry_marks.parquet": expiry_marks,
        "upstream_source_inventory.parquet": frames["source_inventory.parquet"],
        "upstream_sampling_audit.parquet": frames[
            "candidate_session_sampling_audit.parquet"
        ],
        "upstream_daily_close_facts.parquet": frames["daily_close_facts.parquet"],
        "upstream_expiry_close_overlay_audit.parquet": frames[
            "expiry_close_overlay_audit.parquet"
        ],
        "former_unresolved_3374_close_paths.parquet": fallback_audit,
        "source_outcome_audit.parquet": source_audit,
        "path_transaction_costs.parquet": path_costs,
        "transaction_cost_summary.parquet": combined.cost_summary,
        "cap_events.parquet": cap.events,
        "cap_daily.parquet": cap.daily,
        "accepted_path_holding.parquet": accepted_holding,
        "cap_summary.parquet": summary,
        "cutoff_comparison.parquet": comparison,
    }
    for filename, frame in artifacts.items():
        frame.write_parquet(output / filename)
    (output / "README.md").write_text(
        render_readme(
            summary,
            combined.cost_summary,
            source_audit,
            fallback_audit,
            sessions,
            sensitivity,
        ),
        encoding="utf-8",
    )
    inventory = {
        path.name: _artifact_record(path)
        for path in sorted(output.iterdir())
        if path.is_file()
    }
    marker = {
        "complete": True,
        "schema_version": VERSION,
        "analysis_only": True,
        "production_strategy_go": False,
        "source_universe_d_safe": False,
        "upstream_root": str(upstream_root.resolve()),
        "upstream_complete_sha256": _sha256(upstream_root / "complete.json"),
        "upstream_runner_version": upstream_marker["runner_version"],
        "upstream_marker_payload_sha256": upstream_marker.get(
            "marker_payload_sha256"
        ),
        "upstream_sources": upstream_marker.get("sources"),
        "source_paths": paths.height,
        "priced_paths": path_costs.height,
        "unresolved_paths": 0,
        "former_unresolved_3374_close_paths": fallback_audit.height,
        "former_unresolved_3374_notional_twd": float(
            fallback_audit["normalization_notional_twd"].sum()
        ),
        "cutoff_variants": {
            "normal_cutoff_1300": "13:00:00 Asia/Taipei exclusive",
            "original_cutoff_1320_reference": "13:20:00 Asia/Taipei exclusive",
        },
        "session_close_for_exposure": "13:20:00 Asia/Taipei",
        "session_start": sessions[0],
        "session_end": sessions[-1],
        "session_count": len(sessions),
        "hard_intraday_caps_twd": list(DEFAULT_HARD_INTRADAY_CAPS_TWD),
        "per_product_cap_fraction": 0.30,
        "opening_carry_product_exit_only_policy": True,
        "opening_carry_exit_only_is_distinct_from_d_safe_universe_gate": True,
        "notional_basis": "one_way_spot_entry_notional_twd",
        "normal_continuation_state_sampling": (
            "one_second_last_state_plus_every_spread_pair_epoch_change"
        ),
        "raw_trade_events_unchanged": True,
        "model_imputed_full_carry_on_unknown": True,
        "double_exit_bias_possible": True,
        "expiry_mark_is_official_close": True,
        "expiry_mark_is_official_settlement": False,
        "expiry_mark_is_executable_bbo": False,
        "future_settlement_price_used": False,
        "expiry_spot_source": "MarketInfo.twse_security_trades_daily.close_price",
        "expiry_future_source": "MarketInfo.taifex_futures_trades_daily.close_price",
        "max_drawdown_semantics": "daily_realized_cashflow_only_no_mtm",
        "sensitivity_summary_path": (
            str(sensitivity_path.resolve()) if sensitivity is not None else None
        ),
        "sensitivity_summary_sha256": (
            _sha256(sensitivity_path) if sensitivity is not None else None
        ),
        "elapsed_seconds": wall_time.perf_counter() - started,
        "artifacts": inventory,
    }
    (output / "complete.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = run(args.upstream_root, args.output)
    columns = [
        "entry_cutoff_variant",
        "hard_intraday_cap_twd",
        "accepted_paths",
        "rejected_paths",
        "accepted_entry_one_way_turnover_twd",
        "peak_intraday_one_way_notional_twd",
        "mean_time_weighted_intraday_one_way_notional_twd",
        "peak_eod_outstanding_one_way_notional_twd",
        "mean_eod_outstanding_one_way_notional_twd",
        "realized_gross_pnl_twd",
        "realized_transaction_cost_twd",
        "realized_net_pnl_twd",
        "losing_exits",
        "negative_realized_days",
        "max_realized_drawdown_twd",
        "holding_session_boundaries_p50",
        "holding_session_boundaries_p90",
    ]
    with pl.Config(tbl_cols=30, tbl_width_chars=260):
        print(summary.select(columns))


if __name__ == "__main__":
    main()
