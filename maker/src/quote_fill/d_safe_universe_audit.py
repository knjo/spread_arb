"""Audit the causal liquidity gate applied to the frozen q95 path cohort.

The selected challenger paths are all q95, while the current liquidity
publication contains q50 and q80 rows only.  The liquidity hard gate is
route/product/day level in that publication.  This audit therefore requires
the q50 and q80 gate decisions to agree exactly, then exposes their consensus
as a clearly labelled proxy for the selected q95 paths.  It never fabricates
an exact q95 liquidity row.

The raw replay cohort is retrospective.  A causal daily gate can suppress a
cohort member on D, but it cannot evaluate or add products for which raw replay
was never generated.  Coverage-gap artifacts make that remaining selection
leak explicit.  A failed next-day gate blocks *new* entries only; an already
held product remains exit-only and is never force-liquidated by this gate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import polars as pl


AUDIT_VERSION = "d_safe_selected_q95_universe_audit_v1"
Q95_GATE_METHOD = "q50_q80_route_gate_consensus_proxy"

DEFAULT_SELECTED_PATHS = Path(
    "maker/data/walkforward/"
    "prequential_challenger_ab12_60d_20260821/selected_policy_paths.parquet"
)
DEFAULT_ROLLING_SCREEN = Path(
    "maker/data/walkforward/liquidity/rolling_liquidity_screen.parquet"
)
DEFAULT_UNIVERSE_MANIFEST = Path(
    "maker/data/walkforward/liquidity/universe_manifest_v2/"
    "research_universe_manifest.parquet"
)
DEFAULT_OUTPUT_DIR = Path(
    "maker/data/walkforward/d_safe_universe_audit_20260821"
)

ARTIFACTS: Mapping[str, str] = {
    "path_gate_audit.parquet": "path_gate_audit",
    "product_day_route_gate_audit.parquet": "product_day_route_gate_audit",
    "product_day_gate_audit.parquet": "product_day_gate_audit",
    "daily_gate_summary.parquet": "daily_gate_summary",
    "causal_pass_outside_raw_replay.parquet": "coverage_gap",
    "overall_summary.parquet": "overall_summary",
}


@dataclass(frozen=True)
class DSafeUniverseAuditConfig:
    selected_boundary_quantile: int = 95
    gate_proxy_quantiles: tuple[int, ...] = (50, 80)
    audit_version: str = AUDIT_VERSION

    def validate(self) -> None:
        if not 0 < self.selected_boundary_quantile < 100:
            raise ValueError("selected_boundary_quantile must be in (0, 100)")
        if len(self.gate_proxy_quantiles) < 2:
            raise ValueError("gate_proxy_quantiles must contain at least two rows")
        if len(set(self.gate_proxy_quantiles)) != len(self.gate_proxy_quantiles):
            raise ValueError("gate_proxy_quantiles must be unique")
        if self.selected_boundary_quantile in self.gate_proxy_quantiles:
            raise ValueError("selected q must not be mislabeled as a proxy q")
        if not self.audit_version:
            raise ValueError("audit_version cannot be empty")


@dataclass(frozen=True)
class DSafeUniverseAuditResult:
    path_gate_audit: pl.DataFrame
    product_day_route_gate_audit: pl.DataFrame
    product_day_gate_audit: pl.DataFrame
    daily_gate_summary: pl.DataFrame
    coverage_gap: pl.DataFrame
    overall_summary: pl.DataFrame


def build_d_safe_universe_audit(
    selected_paths: pl.DataFrame,
    rolling_screen: pl.DataFrame,
    universe_manifest: pl.DataFrame,
    *,
    config: DSafeUniverseAuditConfig = DSafeUniverseAuditConfig(),
) -> DSafeUniverseAuditResult:
    """Join frozen paths to a strictly prior-day route liquidity gate."""

    config.validate()
    paths = _normalise_paths(selected_paths, config)
    manifest, raw_products = _normalise_manifest(universe_manifest)
    gate = _build_route_gate_consensus(rolling_screen, config)
    decision_dates = paths.select("Date").unique()
    decision_gate = gate.join(decision_dates, on="Date", how="semi")

    path_gate = _join_paths(paths, gate, raw_products, config)
    product_day_route = _product_day_route_audit(path_gate)
    product_day = _product_day_audit(product_day_route)
    coverage_gap = _coverage_gap(decision_gate, raw_products, config)
    daily = _daily_summary(
        path_gate,
        product_day_route,
        product_day,
        decision_gate,
        coverage_gap,
        raw_products,
    )
    overall = _overall_summary(
        path_gate,
        product_day_route,
        product_day,
        daily,
        decision_gate,
        coverage_gap,
        manifest,
        raw_products,
        config,
    )
    result = DSafeUniverseAuditResult(
        path_gate_audit=path_gate,
        product_day_route_gate_audit=product_day_route,
        product_day_gate_audit=product_day,
        daily_gate_summary=daily,
        coverage_gap=coverage_gap,
        overall_summary=overall,
    )
    _validate_result(result, config)
    return result


def publish_d_safe_universe_audit(
    result: DSafeUniverseAuditResult,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    selected_paths_source: Path | None = None,
    rolling_screen_source: Path | None = None,
    universe_manifest_source: Path | None = None,
) -> Path:
    """Atomically publish audit tables, report, and a hash manifest."""

    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        for filename, field in ARTIFACTS.items():
            getattr(result, field).write_parquet(temporary / filename)
        report = render_markdown_report(result)
        (temporary / "README.md").write_text(report, encoding="utf-8")
        source_paths = {
            "selected_paths": selected_paths_source,
            "rolling_screen": rolling_screen_source,
            "universe_manifest": universe_manifest_source,
        }
        source_hashes = {
            name: _sha256(Path(path))
            for name, path in source_paths.items()
            if path is not None
        }
        artifact_hashes = {
            path.name: _sha256(path)
            for path in sorted(temporary.iterdir())
            if path.name != "complete.json"
        }
        summary = result.overall_summary.row(0, named=True)
        marker = {
            "audit_version": AUDIT_VERSION,
            "selected_boundary_quantile": 95,
            "exact_q95_liquidity_row_available": False,
            "liquidity_gate_method": Q95_GATE_METHOD,
            "source_sha256": source_hashes,
            "artifact_sha256": artifact_hashes,
            "selected_path_count": int(summary["selected_path_count"]),
            "raw_replay_cohort_product_count": int(
                summary["raw_replay_cohort_product_count"]
            ),
            "production_universe_backtest_complete": False,
        }
        (temporary / "complete.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if output_dir.exists():
            backup = output_dir.with_name(f".{output_dir.name}.old")
            if backup.exists():
                shutil.rmtree(backup)
            output_dir.replace(backup)
            temporary.replace(output_dir)
            shutil.rmtree(backup)
        else:
            temporary.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir


def render_markdown_report(result: DSafeUniverseAuditResult) -> str:
    summary = result.overall_summary.row(0, named=True)
    daily = result.daily_gate_summary
    largest_gap = daily.sort(
        "causal_pass_product_routes_outside_raw_replay", descending=True
    ).head(5)
    lines = [
        "# D-safe 每日商品 gate 稽核（q95 selected paths）",
        "",
        "## 結論",
        "",
        (
            f"目前 3,672 條 selected paths 中，D 日 gate 允許新進場 "
            f"{summary['path_gate_pass_count']:,} 條，阻擋 "
            f"{summary['path_gate_blocked_count']:,} 條。這個 gate 嚴格只讀 "
            "`source_asof_date < Date` 且 `execution_safe_snapshot=true` 的列。"
        ),
        "",
        (
            "但這不是全市場的 D-safe 回測。raw replay cohort 仍是用 "
            f"May/June/Jul-Aug 事後穩定性挑出的 {summary['raw_replay_cohort_product_count']} "
            f"檔，其中只有 {summary['selected_path_product_count']} 檔產生 selected path；"
            "因而每日 causal gate 即使通過 cohort 外商品，也無法把它補進回測。"
        ),
        "",
        (
            f"在 selected-path 的 {summary['decision_date_count']} 個日期內，共有 "
            f"{summary['causal_pass_product_routes_outside_raw_replay']:,} 個 "
            "D-safe pass 商品-route 落在 raw replay cohort 外。這些列直接保存在 "
            "`causal_pass_outside_raw_replay.parquet`，是剩餘 universe selection leak "
            "的可機讀證據。"
        ),
        "",
        "## q95 限制",
        "",
        (
            "selected paths 全為 q95，但目前 liquidity publication 只有 q50、q80。"
            "本稽核先驗證每個 Date/ValueCode/route 的 q50 與 q80 在 "
            "source-asof、gate status 與 pre-replay decision 上完全一致，再使用 "
            f"`{Q95_GATE_METHOD}`。因此這是 route-level liquidity gate proxy，"
            "不是憑空宣稱存在 exact q95 liquidity row，也不驗證 q95 boundary "
            "parameter publication。"
        ),
        "",
        "## 持倉語意",
        "",
        (
            "gate fail 只禁止該日新進場。若商品已有跨日持倉，策略狀態是 "
            "`exit_only_continue_existing_exit_policy`：仍可按原出場規則平倉，"
            "不因下一日 gate fail 被強迫清算。"
        ),
        "",
        "## Selected cohort 統計",
        "",
        "| 指標 | 數量 |",
        "|---|---:|",
        f"| Selected paths | {summary['selected_path_count']:,} |",
        f"| Gate pass paths | {summary['path_gate_pass_count']:,} |",
        f"| Gate blocked paths | {summary['path_gate_blocked_count']:,} |",
        f"| Product-days | {summary['selected_product_day_count']:,} |",
        f"| 全 route pass product-days | {summary['product_day_all_pass_count']:,} |",
        f"| Mixed product-days | {summary['product_day_mixed_count']:,} |",
        f"| 全 route blocked product-days | {summary['product_day_all_blocked_count']:,} |",
        (
            "| Raw cohort D-safe blocked/missing product-routes | "
            f"{summary['effective_blocked_new_entry_product_routes_in_raw_replay']:,} |"
        ),
        "",
        "## Cohort 外 D-safe pass 最大缺口日期",
        "",
        "| Date | 全市場 pass 商品-route | cohort 內 | cohort 外（無 raw replay） |",
        "|---|---:|---:|---:|",
    ]
    for row in largest_gap.iter_rows(named=True):
        lines.append(
            f"| {row['Date']} | {row['causal_pass_product_route_count']:,} | "
            f"{row['causal_pass_product_routes_in_raw_replay']:,} | "
            f"{row['causal_pass_product_routes_outside_raw_replay']:,} |"
        )
    lines.extend(
        [
            "",
            "## 可用範圍",
            "",
            (
                "這份結果可用來對 45 檔 retrospective raw-replay cohort 做每日 "
                "entry gate 與 exit-only 狀態管理；不可用來宣稱全市場商品池已完成 "
                "walk-forward replay。要移除這個 leak，必須對每個 D 日可能通過 gate "
                "的商品先有 raw replay（或建立不依賴事後名單的全市場 replay cache），"
                "再由 D-1 資料決定 D 日可進場集合。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _normalise_paths(
    frame: pl.DataFrame, config: DSafeUniverseAuditConfig
) -> pl.DataFrame:
    required = {
        "Date",
        "ValueCode",
        "QuoteCode",
        "entry_route",
        "boundary_quantile",
        "policy_path_id",
        "position_established_ns",
        "normalization_notional_twd",
        "filled_entry_outcome_category",
        "terminal_date",
        "terminal_cashflow_priced",
    }
    _require(frame, required, "selected paths")
    result = frame.select(sorted(required)).with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("entry_route").cast(pl.String),
        pl.col("boundary_quantile").cast(pl.Int64),
        pl.col("policy_path_id").cast(pl.String),
    )
    if result.is_empty():
        raise ValueError("selected paths cannot be empty")
    if result["policy_path_id"].n_unique() != result.height:
        raise ValueError("selected paths contain duplicate policy_path_id")
    quantiles = set(result["boundary_quantile"].unique().to_list())
    if quantiles != {config.selected_boundary_quantile}:
        raise ValueError(
            "selected paths must contain only configured q"
            f"{config.selected_boundary_quantile}; found {sorted(quantiles)}"
        )
    _validate_yyyymmdd(result, "Date", "selected paths")
    return result.sort(["Date", "position_established_ns", "policy_path_id"])


def _normalise_manifest(
    frame: pl.DataFrame,
) -> tuple[pl.DataFrame, set[str]]:
    required = {
        "ValueCode",
        "execution_cli_member",
        "retrospective_research_selection",
        "selection_contains_target_day_outcomes",
        "production_universe_approved",
        "runtime_daily_liquidity_gate_required",
    }
    _require(frame, required, "universe manifest")
    result = frame.select(sorted(required)).with_columns(
        pl.col("ValueCode").cast(pl.String)
    )
    if result["ValueCode"].n_unique() != result.height:
        raise ValueError("universe manifest contains duplicate ValueCode")
    raw = result.filter(pl.col("execution_cli_member").fill_null(False))
    if raw.is_empty():
        raise ValueError("universe manifest has no execution cohort")
    if not raw["retrospective_research_selection"].fill_null(False).all():
        raise ValueError("execution cohort is not marked retrospective")
    if not raw["selection_contains_target_day_outcomes"].fill_null(False).all():
        raise ValueError("execution cohort must disclose target-day selection")
    if raw["production_universe_approved"].fill_null(True).any():
        raise ValueError("retrospective execution cohort cannot be production-approved")
    if not raw["runtime_daily_liquidity_gate_required"].fill_null(False).all():
        raise ValueError("execution cohort must require the runtime daily gate")
    return result, set(raw["ValueCode"].to_list())


def _build_route_gate_consensus(
    frame: pl.DataFrame, config: DSafeUniverseAuditConfig
) -> pl.DataFrame:
    required = {
        "Date",
        "ValueCode",
        "QuoteCode",
        "route",
        "boundary_quantile",
        "source_asof_date",
        "liquidity_gate_status",
        "pre_replay_candidate",
        "replay_tier",
        "execution_safe_snapshot",
        "contains_target_day_outcome",
    }
    _require(frame, required, "rolling liquidity screen")
    proxy = frame.select(sorted(required)).with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("route").cast(pl.String),
        pl.col("boundary_quantile").cast(pl.Int64),
        pl.col("source_asof_date").cast(pl.String),
    ).filter(pl.col("boundary_quantile").is_in(config.gate_proxy_quantiles))
    if proxy.is_empty():
        raise ValueError("rolling screen has no configured proxy quantile rows")
    _validate_yyyymmdd(proxy, "Date", "rolling liquidity screen")
    _validate_yyyymmdd(
        proxy, "source_asof_date", "rolling liquidity screen", allow_null=True
    )
    key_q = ["Date", "ValueCode", "route", "boundary_quantile"]
    if proxy.select(key_q).n_unique() != proxy.height:
        raise ValueError("rolling screen contains duplicate product-day-route-q")
    key = ["Date", "ValueCode", "route"]
    consensus = proxy.group_by(key).agg(
        pl.col("boundary_quantile").sort().alias("gate_proxy_quantiles"),
        pl.col("QuoteCode").n_unique().alias("_quote_n"),
        pl.col("QuoteCode").first().alias("gate_quote_code"),
        pl.col("source_asof_date").n_unique().alias("_asof_n"),
        pl.col("source_asof_date").first().alias("gate_source_asof_date"),
        pl.col("liquidity_gate_status").n_unique().alias("_status_n"),
        pl.col("liquidity_gate_status").first().alias("liquidity_gate_status"),
        pl.col("pre_replay_candidate").n_unique().alias("_pre_n"),
        pl.col("pre_replay_candidate").first().alias("pre_replay_candidate"),
        pl.col("replay_tier").n_unique().alias("_tier_n"),
        pl.col("replay_tier").first().alias("replay_tier"),
        pl.col("execution_safe_snapshot").fill_null(False).all().alias(
            "gate_execution_safe_snapshot"
        ),
        pl.col("contains_target_day_outcome").fill_null(True).any().alias(
            "gate_contains_target_day_outcome"
        ),
    ).with_columns(
        (
            pl.col("gate_source_asof_date").is_not_null()
            & (pl.col("gate_source_asof_date") < pl.col("Date"))
        ).alias("gate_source_strictly_before_date"),
        (
            pl.col("gate_proxy_quantiles")
            == pl.lit(sorted(config.gate_proxy_quantiles))
        ).alias("gate_proxy_quantiles_complete"),
    )
    disagreements = consensus.filter(
        (pl.col("_quote_n") != 1)
        | (pl.col("_asof_n") != 1)
        | (pl.col("_status_n") != 1)
        | (pl.col("_pre_n") != 1)
        | (pl.col("_tier_n") != 1)
        | (~pl.col("gate_proxy_quantiles_complete"))
    )
    if disagreements.height:
        raise ValueError(
            "q50/q80 route gate rows disagree or are incomplete for "
            f"{disagreements.height} product-day-route groups"
        )
    return consensus.with_columns(
        (
            pl.col("gate_execution_safe_snapshot")
            & (~pl.col("gate_contains_target_day_outcome"))
            & pl.col("gate_source_strictly_before_date")
        ).alias("gate_lineage_valid"),
        pl.lit(False).alias("exact_q95_liquidity_row_available"),
        pl.lit(Q95_GATE_METHOD).alias("liquidity_gate_method"),
    ).drop("_quote_n", "_asof_n", "_status_n", "_pre_n", "_tier_n").sort(key)


def _join_paths(
    paths: pl.DataFrame,
    gate: pl.DataFrame,
    raw_products: set[str],
    config: DSafeUniverseAuditConfig,
) -> pl.DataFrame:
    gate_join = gate.rename({"route": "entry_route"})
    result = paths.join(
        gate_join,
        on=["Date", "ValueCode", "entry_route"],
        how="left",
        validate="m:1",
    ).with_columns(
        pl.col("gate_source_asof_date").is_not_null().alias("gate_row_joined"),
        pl.col("ValueCode").is_in(sorted(raw_products)).alias(
            "raw_replay_cohort_member"
        ),
    ).with_columns(
        (
            pl.col("gate_row_joined")
            & pl.col("gate_lineage_valid").fill_null(False)
            & (pl.col("liquidity_gate_status") == "pass")
            & pl.col("pre_replay_candidate").fill_null(False)
        ).alias("new_entry_gate_pass")
    ).with_columns(
        pl.when(pl.col("new_entry_gate_pass"))
        .then(pl.lit("allow_new_entry"))
        .otherwise(pl.lit("block_new_entry"))
        .alias("new_entry_gate_action"),
        pl.when(~pl.col("gate_row_joined"))
        .then(pl.lit("missing_route_gate"))
        .when(~pl.col("gate_lineage_valid").fill_null(False))
        .then(pl.lit("unsafe_or_noncausal_route_gate"))
        .when(pl.col("liquidity_gate_status") != "pass")
        .then(pl.concat_str(pl.lit("liquidity_"), pl.col("liquidity_gate_status")))
        .when(~pl.col("pre_replay_candidate").fill_null(False))
        .then(pl.lit("pre_replay_candidate_false"))
        .otherwise(pl.lit("pass"))
        .alias("new_entry_gate_reason"),
        pl.when(~pl.col("new_entry_gate_pass"))
        .then(pl.lit("exit_only_continue_existing_exit_policy"))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias("blocked_product_if_already_held_action"),
        pl.lit(False).alias("forced_liquidation_due_to_gate"),
        pl.lit(True).alias("retrospective_raw_replay_cohort_constraint"),
        pl.lit(config.audit_version).alias("d_safe_universe_audit_version"),
    )
    outside = result.filter(~pl.col("raw_replay_cohort_member"))
    if outside.height:
        raise ValueError(
            f"selected paths contain {outside.height} rows outside raw replay cohort"
        )
    return result.sort(["Date", "position_established_ns", "policy_path_id"])


def _product_day_route_audit(path_gate: pl.DataFrame) -> pl.DataFrame:
    return path_gate.group_by(["Date", "ValueCode", "entry_route"]).agg(
        pl.col("QuoteCode").first(),
        pl.len().alias("selected_path_count"),
        pl.col("new_entry_gate_pass").sum().alias("gate_pass_path_count"),
        (~pl.col("new_entry_gate_pass")).sum().alias("gate_blocked_path_count"),
        pl.col("new_entry_gate_pass").all().alias("new_entry_gate_pass"),
        pl.col("new_entry_gate_action").first(),
        pl.col("new_entry_gate_reason").first(),
        pl.col("liquidity_gate_status").first(),
        pl.col("pre_replay_candidate").first(),
        pl.col("replay_tier").first(),
        pl.col("gate_source_asof_date").first(),
        pl.col("gate_lineage_valid").first(),
        pl.col("liquidity_gate_method").first(),
        pl.lit("exit_only_continue_existing_exit_policy").alias(
            "held_position_policy_when_gate_blocked"
        ),
        pl.lit(False).alias("forced_liquidation_due_to_gate"),
    ).sort(["Date", "ValueCode", "entry_route"])


def _product_day_audit(product_day_route: pl.DataFrame) -> pl.DataFrame:
    return product_day_route.group_by(["Date", "ValueCode"]).agg(
        pl.col("QuoteCode").first(),
        pl.col("entry_route").n_unique().alias("selected_route_count"),
        pl.col("selected_path_count").sum().alias("selected_path_count"),
        pl.col("gate_pass_path_count").sum().alias("gate_pass_path_count"),
        pl.col("gate_blocked_path_count").sum().alias("gate_blocked_path_count"),
        pl.col("new_entry_gate_pass").sum().alias("gate_pass_route_count"),
        (~pl.col("new_entry_gate_pass")).sum().alias("gate_blocked_route_count"),
    ).with_columns(
        pl.when(pl.col("gate_blocked_route_count") == 0)
        .then(pl.lit("all_selected_routes_pass"))
        .when(pl.col("gate_pass_route_count") == 0)
        .then(pl.lit("all_selected_routes_blocked"))
        .otherwise(pl.lit("mixed_selected_routes"))
        .alias("product_day_gate_status"),
        (pl.col("gate_pass_route_count") > 0).alias("any_new_entry_route_allowed"),
        pl.lit("exit_only_continue_existing_exit_policy").alias(
            "held_position_policy_when_all_new_entries_blocked"
        ),
        pl.lit(False).alias("forced_liquidation_due_to_gate"),
    ).sort(["Date", "ValueCode"])


def _coverage_gap(
    gate: pl.DataFrame,
    raw_products: set[str],
    config: DSafeUniverseAuditConfig,
) -> pl.DataFrame:
    return gate.filter(
        pl.col("gate_lineage_valid")
        & (pl.col("liquidity_gate_status") == "pass")
        & pl.col("pre_replay_candidate")
        & (~pl.col("ValueCode").is_in(sorted(raw_products)))
    ).select(
        "Date",
        "ValueCode",
        "gate_quote_code",
        "route",
        "gate_source_asof_date",
        "liquidity_gate_status",
        "pre_replay_candidate",
        "replay_tier",
        "liquidity_gate_method",
        pl.lit(config.selected_boundary_quantile).alias(
            "selected_path_boundary_quantile"
        ),
        pl.lit(False).alias("raw_replay_available"),
        pl.lit(False).alias("can_enter_current_selected_path_backtest"),
        pl.lit("causal_gate_pass_but_raw_replay_absent").alias(
            "coverage_gap_reason"
        ),
        pl.lit(True).alias("retrospective_universe_selection_leak_evidence"),
    ).sort(["Date", "ValueCode", "route"])


def _daily_summary(
    path_gate: pl.DataFrame,
    product_day_route: pl.DataFrame,
    product_day: pl.DataFrame,
    gate: pl.DataFrame,
    coverage_gap: pl.DataFrame,
    raw_products: set[str],
) -> pl.DataFrame:
    route_count = gate["route"].n_unique()
    if route_count <= 0:
        raise ValueError("decision-date route gate cannot be empty")
    expected_raw_product_routes = len(raw_products) * route_count
    path_daily = path_gate.group_by("Date").agg(
        pl.len().alias("selected_path_count"),
        pl.col("new_entry_gate_pass").sum().alias("selected_path_gate_pass_count"),
        (~pl.col("new_entry_gate_pass")).sum().alias(
            "selected_path_gate_blocked_count"
        ),
    )
    route_daily = product_day_route.group_by("Date").agg(
        pl.len().alias("selected_product_day_route_count"),
        pl.col("new_entry_gate_pass").sum().alias(
            "selected_product_day_route_pass_count"
        ),
        (~pl.col("new_entry_gate_pass")).sum().alias(
            "selected_product_day_route_blocked_count"
        ),
    )
    product_daily = product_day.group_by("Date").agg(
        pl.len().alias("selected_product_day_count"),
        (pl.col("product_day_gate_status") == "all_selected_routes_pass")
        .sum()
        .alias("selected_product_day_all_pass_count"),
        (pl.col("product_day_gate_status") == "mixed_selected_routes")
        .sum()
        .alias("selected_product_day_mixed_count"),
        (pl.col("product_day_gate_status") == "all_selected_routes_blocked")
        .sum()
        .alias("selected_product_day_all_blocked_count"),
    )
    valid = gate.filter(pl.col("gate_lineage_valid"))
    market_daily = valid.group_by("Date").agg(
        pl.len().alias("causal_screen_product_route_count"),
        (
            (pl.col("liquidity_gate_status") == "pass")
            & pl.col("pre_replay_candidate")
        ).sum().alias("causal_pass_product_route_count"),
        (
            (pl.col("liquidity_gate_status") == "pass")
            & pl.col("pre_replay_candidate")
            & pl.col("ValueCode").is_in(sorted(raw_products))
        ).sum().alias("causal_pass_product_routes_in_raw_replay"),
        pl.col("ValueCode")
        .is_in(sorted(raw_products))
        .sum()
        .alias("causal_raw_replay_product_route_count"),
        (
            pl.col("ValueCode").is_in(sorted(raw_products))
            & (
                (pl.col("liquidity_gate_status") != "pass")
                | (~pl.col("pre_replay_candidate"))
            )
        )
        .sum()
        .alias("causal_blocked_product_routes_in_raw_replay"),
        pl.col("ValueCode").n_unique().alias("causal_screen_product_count"),
    )
    gap_daily = coverage_gap.group_by("Date").agg(
        pl.len().alias("causal_pass_product_routes_outside_raw_replay"),
        pl.col("ValueCode").n_unique().alias(
            "causal_pass_products_outside_raw_replay"
        ),
    )
    return path_daily.join(route_daily, on="Date", how="left", validate="1:1").join(
        product_daily, on="Date", how="left", validate="1:1"
    ).join(market_daily, on="Date", how="left", validate="1:1").join(
        gap_daily, on="Date", how="left", validate="1:1"
    ).with_columns(
        pl.col("causal_pass_product_routes_outside_raw_replay")
        .fill_null(0)
        .cast(pl.UInt32),
        pl.col("causal_pass_products_outside_raw_replay")
        .fill_null(0)
        .cast(pl.UInt32),
        pl.lit(expected_raw_product_routes).alias(
            "expected_raw_replay_product_route_count"
        ),
        (
            pl.lit(expected_raw_product_routes)
            - pl.col("causal_raw_replay_product_route_count")
        ).alias("causal_missing_product_routes_in_raw_replay"),
        (
            pl.lit(expected_raw_product_routes)
            - pl.col("causal_raw_replay_product_route_count")
            + pl.col("causal_blocked_product_routes_in_raw_replay")
        ).alias(
            "effective_blocked_new_entry_product_routes_in_raw_replay"
        ),
        pl.lit(False).alias("production_universe_backtest_complete"),
    ).sort("Date")


def _overall_summary(
    path_gate: pl.DataFrame,
    product_day_route: pl.DataFrame,
    product_day: pl.DataFrame,
    daily: pl.DataFrame,
    gate: pl.DataFrame,
    coverage_gap: pl.DataFrame,
    manifest: pl.DataFrame,
    raw_products: set[str],
    config: DSafeUniverseAuditConfig,
) -> pl.DataFrame:
    raw_without_paths = sorted(raw_products - set(path_gate["ValueCode"].unique()))
    return pl.DataFrame(
        {
            "audit_version": [config.audit_version],
            "selected_boundary_quantile": [config.selected_boundary_quantile],
            "exact_q95_liquidity_row_available": [False],
            "liquidity_gate_method": [Q95_GATE_METHOD],
            "route_gate_proxy_quantiles": [
                ",".join(map(str, sorted(config.gate_proxy_quantiles)))
            ],
            "decision_date_count": [path_gate["Date"].n_unique()],
            "selected_path_count": [path_gate.height],
            "path_gate_pass_count": [
                int(path_gate["new_entry_gate_pass"].sum())
            ],
            "path_gate_blocked_count": [
                int((~path_gate["new_entry_gate_pass"]).sum())
            ],
            "selected_path_product_count": [path_gate["ValueCode"].n_unique()],
            "selected_product_day_route_count": [product_day_route.height],
            "selected_product_day_count": [product_day.height],
            "product_day_all_pass_count": [
                product_day.filter(
                    pl.col("product_day_gate_status")
                    == "all_selected_routes_pass"
                ).height
            ],
            "product_day_mixed_count": [
                product_day.filter(
                    pl.col("product_day_gate_status") == "mixed_selected_routes"
                ).height
            ],
            "product_day_all_blocked_count": [
                product_day.filter(
                    pl.col("product_day_gate_status")
                    == "all_selected_routes_blocked"
                ).height
            ],
            "raw_replay_cohort_product_count": [len(raw_products)],
            "raw_replay_products_without_selected_paths_count": [
                len(raw_without_paths)
            ],
            "raw_replay_products_without_selected_paths": [
                ",".join(raw_without_paths)
            ],
            "causal_screen_product_day_route_count": [gate.height],
            "causal_pass_product_routes_outside_raw_replay": [
                coverage_gap.height
            ],
            "causal_raw_replay_product_day_route_count": [
                int(daily["causal_raw_replay_product_route_count"].sum())
            ],
            "causal_blocked_product_routes_in_raw_replay": [
                int(
                    daily[
                        "causal_blocked_product_routes_in_raw_replay"
                    ].sum()
                )
            ],
            "causal_missing_product_routes_in_raw_replay": [
                int(
                    daily[
                        "causal_missing_product_routes_in_raw_replay"
                    ].sum()
                )
            ],
            "effective_blocked_new_entry_product_routes_in_raw_replay": [
                int(
                    daily[
                        "effective_blocked_new_entry_product_routes_in_raw_replay"
                    ].sum()
                )
            ],
            "causal_pass_products_outside_raw_replay": [
                coverage_gap["ValueCode"].n_unique()
                if not coverage_gap.is_empty()
                else 0
            ],
            "dates_with_causal_pass_outside_raw_replay": [
                coverage_gap["Date"].n_unique()
                if not coverage_gap.is_empty()
                else 0
            ],
            "max_daily_causal_pass_product_routes_outside_raw_replay": [
                int(daily["causal_pass_product_routes_outside_raw_replay"].max())
            ],
            "manifest_retrospective_research_selection": [
                bool(
                    manifest.filter(pl.col("execution_cli_member"))[
                        "retrospective_research_selection"
                    ].all()
                )
            ],
            "manifest_selection_contains_target_day_outcomes": [
                bool(
                    manifest.filter(pl.col("execution_cli_member"))[
                        "selection_contains_target_day_outcomes"
                    ].all()
                )
            ],
            "held_blocked_products_are_exit_only": [True],
            "gate_failure_forces_liquidation": [False],
            "retrospective_universe_selection_leak_proven": [
                (coverage_gap.height > 0)
            ],
            "production_universe_backtest_complete": [False],
        }
    )


def _validate_result(
    result: DSafeUniverseAuditResult, config: DSafeUniverseAuditConfig
) -> None:
    paths = result.path_gate_audit
    if paths.filter(
        pl.col("new_entry_gate_pass")
        & (
            ~pl.col("gate_lineage_valid").fill_null(False)
            | (pl.col("gate_source_asof_date") >= pl.col("Date"))
            | (~pl.col("gate_execution_safe_snapshot").fill_null(False))
            | pl.col("gate_contains_target_day_outcome").fill_null(True)
        )
    ).height:
        raise AssertionError("a non-causal or unsafe route gate passed")
    if paths["forced_liquidation_due_to_gate"].any():
        raise AssertionError("daily gate must never force-liquidate held positions")
    summary = result.overall_summary.row(0, named=True)
    if summary["selected_path_count"] != paths.height:
        raise AssertionError("selected path summary mismatch")
    if (
        summary["path_gate_pass_count"] + summary["path_gate_blocked_count"]
        != paths.height
    ):
        raise AssertionError("path gate counts do not partition selected paths")
    if summary["exact_q95_liquidity_row_available"]:
        raise AssertionError("audit must not claim an exact q95 liquidity row")
    if summary["selected_boundary_quantile"] != config.selected_boundary_quantile:
        raise AssertionError("selected boundary quantile summary mismatch")


def _require(frame: pl.DataFrame, columns: set[str], source: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def _validate_yyyymmdd(
    frame: pl.DataFrame,
    column: str,
    source: str,
    *,
    allow_null: bool = False,
) -> None:
    parsed = frame.select(
        pl.col(column),
        pl.col(column).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("_date"),
    )
    invalid = parsed.filter(
        ((pl.col(column).is_null() | pl.col("_date").is_null()) & ~pl.lit(allow_null))
        | (
            pl.col(column).is_not_null()
            & (
                (pl.col(column).str.len_chars() != 8)
                | pl.col("_date").is_null()
            )
        )
    )
    if invalid.height:
        raise ValueError(f"{source} {column} must be valid YYYYMMDD")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-paths", type=Path, default=DEFAULT_SELECTED_PATHS)
    parser.add_argument("--rolling-screen", type=Path, default=DEFAULT_ROLLING_SCREEN)
    parser.add_argument(
        "--universe-manifest", type=Path, default=DEFAULT_UNIVERSE_MANIFEST
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = build_d_safe_universe_audit(
        pl.read_parquet(args.selected_paths),
        pl.read_parquet(args.rolling_screen),
        pl.read_parquet(args.universe_manifest),
    )
    output = publish_d_safe_universe_audit(
        result,
        args.output_dir,
        selected_paths_source=args.selected_paths,
        rolling_screen_source=args.rolling_screen,
        universe_manifest_source=args.universe_manifest,
    )
    print(output)
    print(result.overall_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
