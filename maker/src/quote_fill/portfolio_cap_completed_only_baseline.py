"""Publish the completed-only portfolio-cap baseline comparator.

This bundle intentionally selects only the 2,411 paths whose terminal
cashflows were already point identified in the 2026-08-21 q95 challenger cost
bundle.  It is a verified plumbing/capacity comparator, not a final strategy
result: 1,261 unresolved paths are excluded and the source 45-product universe
is retrospective.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import polars as pl

from .combined_cost_cap_sweep import verify_combined_cost_cap_bundle
from .portfolio_cap_backtester import (
    PortfolioCapBacktestConfig,
    PortfolioCapBacktestResult,
    PortfolioCapScenario,
    backtest_priced_paths,
)


BASELINE_VERSION = "portfolio_cap_completed_only_baseline_20260821_v1"
BUNDLE_SCHEMA_VERSION = "portfolio_cap_completed_only_baseline_bundle_v1"
DEFAULT_COST_SOURCE_ROOT = Path(
    "maker/data/walkforward/prequential_challenger_ab12_cost_caps_20260821_v1"
)
DEFAULT_LEDGER_SOURCE_ROOT = Path(
    "maker/data/walkforward/prequential_challenger_ab12_60d_20260821"
)
DEFAULT_OUTPUT_ROOT = Path(
    "maker/data/walkforward/portfolio_cap_completed_only_baseline_20260821_v1"
)

ARTIFACTS: Mapping[str, str] = {
    "portfolio_cap_events.parquet": "events",
    "portfolio_cap_daily.parquet": "daily",
    "portfolio_cap_summary.parquet": "summary",
}


def baseline_config() -> PortfolioCapBacktestConfig:
    scenarios = tuple(
        PortfolioCapScenario(
            scenario_id=f"completed_only_hard_intraday_{cap_m}m_eod_unbounded",
            hard_intraday_cap_twd=float(cap_m * 1_000_000),
        )
        for cap_m in (10, 20, 30, 40, 50)
    ) + (
        PortfolioCapScenario(
            scenario_id="completed_only_hard_intraday_40m_eod_20m_reporting_only",
            hard_intraday_cap_twd=40_000_000.0,
            eod_overnight_limit_twd=20_000_000.0,
        ),
    )
    return PortfolioCapBacktestConfig(scenarios=scenarios)


def build_completed_only_baseline(
    *,
    cost_source_root: Path = DEFAULT_COST_SOURCE_ROOT,
    ledger_source_root: Path = DEFAULT_LEDGER_SOURCE_ROOT,
) -> tuple[PortfolioCapBacktestResult, dict[str, object]]:
    source, sessions, metadata = _load_verified_source(
        cost_source_root=cost_source_root,
        ledger_source_root=ledger_source_root,
    )
    result = backtest_priced_paths(
        source,
        config=baseline_config(),
        session_dates=sessions,
    )
    bias_columns = (
        pl.lit("terminal_cashflow_priced_completed_only").alias(
            "input_path_selection"
        ),
        pl.lit(True).alias("survivor_completed_only_selection_bias"),
        pl.lit(1_261).cast(pl.Int64).alias("source_unresolved_paths_excluded"),
        pl.lit(False).alias("terminal_overlay_included"),
        pl.lit(False).alias("source_universe_d_safe"),
        pl.lit(False).alias("joint_volume_allocated"),
        pl.lit(False).alias("final_strategy_result"),
    )
    labelled = PortfolioCapBacktestResult(
        events=result.events.with_columns(*bias_columns),
        daily=result.daily.with_columns(*bias_columns),
        summary=result.summary.with_columns(*bias_columns),
    )
    _validate_baseline(labelled, sessions)
    return labelled, metadata


def publish_completed_only_baseline(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    cost_source_root: Path = DEFAULT_COST_SOURCE_ROOT,
    ledger_source_root: Path = DEFAULT_LEDGER_SOURCE_ROOT,
) -> PortfolioCapBacktestResult:
    result, metadata = build_completed_only_baseline(
        cost_source_root=cost_source_root,
        ledger_source_root=ledger_source_root,
    )
    destination = Path(output_root)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        declarations: dict[str, dict[str, object]] = {}
        for filename, attribute in ARTIFACTS.items():
            frame = getattr(result, attribute)
            path = stage / filename
            frame.write_parquet(path)
            declarations[filename] = _frame_declaration(path, frame)
        marker: dict[str, object] = {
            "complete": True,
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "baseline_version": BASELINE_VERSION,
            "config": _config_payload(),
            "sources": metadata,
            "fact_semantics": _fact_semantics(),
            "implementation_sources": _implementation_sources(),
            "artifacts": declarations,
        }
        marker["marker_payload_sha256"] = _canonical_sha256(marker)
        (stage / "complete.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _verify_published_files(stage)
        stage.rename(destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return result


def verify_completed_only_baseline(
    output_root: Path = DEFAULT_OUTPUT_ROOT, *, rebuild: bool = True
) -> dict[str, object]:
    marker, frames = _verify_published_files(output_root)
    if rebuild:
        expected, metadata = build_completed_only_baseline(
            cost_source_root=Path(str(marker["sources"]["cost_source_root"])),
            ledger_source_root=Path(str(marker["sources"]["ledger_source_root"])),
        )
        if metadata != marker["sources"]:
            raise ValueError("baseline source metadata changed")
        for filename, attribute in ARTIFACTS.items():
            actual = frames[filename]
            rebuilt = getattr(expected, attribute)
            if actual.schema != rebuilt.schema or not actual.equals(
                rebuilt, null_equal=True
            ):
                raise ValueError(f"baseline source rebuild differs: {filename}")
    return marker


def _load_verified_source(
    *, cost_source_root: Path, ledger_source_root: Path
) -> tuple[pl.DataFrame, tuple[str, ...], dict[str, object]]:
    cost_root = Path(cost_source_root).resolve()
    ledger_root = Path(ledger_source_root).resolve()
    verify_combined_cost_cap_bundle(cost_root, rebuild=False)

    cost_marker = _read_json(cost_root / "complete.json")
    ledger_marker = _read_json(ledger_root / "complete.json")
    if (
        cost_marker.get("complete") is not True
        or ledger_marker.get("complete") is not True
    ):
        raise ValueError("baseline source marker is incomplete")
    ledger_declaration = ledger_marker.get("artifacts", {}).get(
        "daily_outstanding.parquet"
    )
    daily_path = ledger_root / "daily_outstanding.parquet"
    daily = pl.read_parquet(daily_path)
    if ledger_declaration != _frame_declaration(daily_path, daily):
        raise ValueError("daily session-calendar source changed")

    path_file = cost_root / "path_transaction_costs.parquet"
    all_paths = pl.read_parquet(path_file)
    selected = all_paths.filter(
        pl.col("terminal_cashflow_priced") == True  # noqa: E712
    )
    unresolved = all_paths.filter(
        (pl.col("terminal_cashflow_priced") != True).fill_null(True)  # noqa: E712
    )
    if (
        all_paths.height != 3_672
        or selected.height != 2_411
        or unresolved.height != 1_261
    ):
        raise ValueError("completed-only baseline source cardinality changed")
    sessions = tuple(str(value) for value in daily["Date"].to_list())
    if len(sessions) != 62 or len(set(sessions)) != len(sessions):
        raise ValueError("baseline session calendar must contain 62 unique sessions")
    metadata: dict[str, object] = {
        "cost_source_root": str(cost_root),
        "cost_source_complete_sha256": _file_sha256(cost_root / "complete.json"),
        "cost_source_marker_payload_sha256": cost_marker.get(
            "marker_payload_sha256"
        ),
        "path_transaction_costs_path": str(path_file),
        "path_transaction_costs_sha256": _file_sha256(path_file),
        "source_paths": all_paths.height,
        "selected_completed_priced_paths": selected.height,
        "excluded_unresolved_paths": unresolved.height,
        "ledger_source_root": str(ledger_root),
        "ledger_source_complete_sha256": _file_sha256(
            ledger_root / "complete.json"
        ),
        "ledger_source_marker_payload_sha256": ledger_marker.get(
            "marker_payload_sha256"
        ),
        "session_calendar_path": str(daily_path),
        "session_calendar_sha256": _file_sha256(daily_path),
        "session_count": len(sessions),
        "first_session": sessions[0],
        "last_session": sessions[-1],
    }
    return selected, sessions, metadata


def _validate_baseline(
    result: PortfolioCapBacktestResult, sessions: tuple[str, ...]
) -> None:
    if result.summary.height != 6:
        raise ValueError("baseline must have six cap scenarios")
    if result.daily.height != 6 * len(sessions):
        raise ValueError("baseline daily grid is incomplete")
    expected_caps = [10e6, 20e6, 30e6, 40e6, 40e6, 50e6]
    if result.summary["hard_intraday_cap_twd"].to_list() != expected_caps:
        raise ValueError("baseline cap grid differs")
    if result.summary.filter(
        (pl.col("survivor_completed_only_selection_bias") != True)  # noqa: E712
        | (pl.col("terminal_overlay_included") != False)  # noqa: E712
        | (pl.col("final_strategy_result") != False)  # noqa: E712
        | (pl.col("terminal_cashflows_point_identified") != True)  # noqa: E712
    ).height:
        raise ValueError("baseline bias/readiness labels changed")
    if result.summary.filter(
        pl.col("accepted_paths") != pl.col("completed_exits")
    ).height:
        raise ValueError("accepted completed-only path did not exit")
    eod_scenario = result.summary.filter(
        pl.col("eod_overnight_limit_twd").is_not_null()
    )
    if eod_scenario.height != 1 or eod_scenario.item(
        0, "eod_overnight_limit_enforced_on_admission"
    ) is not False:
        raise ValueError("40M/20M EOD scenario semantics changed")


def _verify_published_files(
    output_root: Path,
) -> tuple[dict[str, object], dict[str, pl.DataFrame]]:
    root = Path(output_root)
    expected = {"complete.json", *ARTIFACTS}
    if not root.is_dir() or {path.name for path in root.iterdir()} != expected:
        raise ValueError("completed-only baseline bundle inventory differs")
    marker = _read_json(root / "complete.json")
    declared_sha = marker.get("marker_payload_sha256")
    unhashed = dict(marker)
    unhashed.pop("marker_payload_sha256", None)
    if (
        marker.get("complete") is not True
        or marker.get("schema_version") != BUNDLE_SCHEMA_VERSION
        or marker.get("baseline_version") != BASELINE_VERSION
        or declared_sha != _canonical_sha256(unhashed)
        or marker.get("config") != _config_payload()
        or marker.get("fact_semantics") != _fact_semantics()
        or marker.get("implementation_sources") != _implementation_sources()
    ):
        raise ValueError("completed-only baseline marker is invalid")
    declarations = marker.get("artifacts")
    if not isinstance(declarations, dict) or set(declarations) != set(ARTIFACTS):
        raise ValueError("completed-only baseline artifact inventory differs")
    frames: dict[str, pl.DataFrame] = {}
    for filename in ARTIFACTS:
        path = root / filename
        frame = pl.read_parquet(path)
        if declarations[filename] != _frame_declaration(path, frame):
            raise ValueError(f"completed-only baseline artifact changed: {filename}")
        frames[filename] = frame
    return marker, frames


def _config_payload() -> dict[str, object]:
    config = baseline_config()
    return {
        "scenarios": [
            {
                "scenario_id": scenario.scenario_id,
                "hard_intraday_cap_twd": scenario.hard_intraday_cap_twd,
                "eod_overnight_limit_twd": scenario.eod_overnight_limit_twd,
            }
            for scenario in config.scenarios
        ],
        "per_product_fraction": config.per_product_fraction,
        "entry_cutoff_local_time": config.entry_cutoff_local_time.isoformat(),
        "session_open_local_time": config.session_open_local_time.isoformat(),
        "session_close_local_time": config.session_close_local_time.isoformat(),
        "timezone_name": config.timezone_name,
    }


def _fact_semantics() -> dict[str, object]:
    return {
        "analysis_only_comparator": True,
        "input_path_selection": "terminal_cashflow_priced_completed_only",
        "source_paths": 3_672,
        "selected_completed_priced_paths": 2_411,
        "excluded_unresolved_paths": 1_261,
        "survivor_completed_only_selection_bias": True,
        "terminal_overlay_included": False,
        "final_strategy_result": False,
        "source_universe_d_safe": False,
        "source_universe_role": "retrospective_45_product_development_cohort",
        "joint_volume_allocated": False,
        "entry_cutoff_13_00_asia_taipei_exclusive": True,
        "hard_intraday_cap_enforced_on_admission": True,
        "per_product_cap_fraction": 0.30,
        "eod_overnight_limit_reporting_only": True,
        "eod_overnight_limit_never_used_for_admission": True,
        "realized_curve_excludes_mark_to_market_of_open_inventory": True,
        "daily_win_rate_includes_all_calendar_sessions": True,
        "annualized_sharpe_uses_daily_net_sample_std_and_sqrt_252": True,
        "annualized_return_is_linear_240_session_scaling": True,
        "mean_active_notional_is_session_time_weighted": True,
        "mean_max_product_share_uses_daily_accepted_entry_turnover": True,
        "notional_basis": "one_way_spot_entry_notional_twd",
    }


def _implementation_sources() -> dict[str, str]:
    current = Path(__file__)
    backtester = current.with_name("portfolio_cap_backtester.py")
    return {
        current.name: _file_sha256(current),
        backtester.name: _file_sha256(backtester),
    }


def _frame_declaration(path: Path, frame: pl.DataFrame) -> dict[str, object]:
    return {
        "rows": frame.height,
        "columns": frame.width,
        "schema": {name: str(dtype) for name, dtype in frame.schema.items()},
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--cost-source-root", type=Path, default=DEFAULT_COST_SOURCE_ROOT
    )
    parser.add_argument(
        "--ledger-source-root", type=Path, default=DEFAULT_LEDGER_SOURCE_ROOT
    )
    parser.add_argument("--verify-only", type=Path)
    parser.add_argument("--no-rebuild", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verify_only is not None:
        marker = verify_completed_only_baseline(
            args.verify_only, rebuild=not args.no_rebuild
        )
        print(json.dumps(marker, indent=2, sort_keys=True))
        return 0
    if args.no_rebuild:
        raise ValueError("--no-rebuild is valid only with --verify-only")
    result = publish_completed_only_baseline(
        args.output,
        cost_source_root=args.cost_source_root,
        ledger_source_root=args.ledger_source_root,
    )
    print(result.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
