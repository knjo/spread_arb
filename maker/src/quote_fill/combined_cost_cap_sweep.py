"""Source-bound transaction-cost and simultaneous position-cap diagnostics.

This supplemental analysis reads the immutable AB1/2 prequential challenger
paths.  It applies the user-specified Taiwan spot/futures transaction charges
to completed one-futures-lot cycles, then replays chronological admission with
both a portfolio one-way-notional cap and a per-ValueCode cap.

The source universe is the retrospectively selected 45-product development
cohort.  Terminal cashflows are unresolved for some paths, cancel ACK and joint
volume allocation remain unavailable, and this module therefore cannot turn
the diagnostic challenger into a production strategy or select a best q.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import heapq
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import polars as pl


SWEEP_VERSION = "prequential_ab12_user_cost_combined_caps_v1"
BUNDLE_SCHEMA_VERSION = "prequential_ab12_user_cost_combined_caps_bundle_v1"
EXPECTED_SOURCE_SCHEMA_VERSION = "prequential_diagnostic_challenger_bundle_v1"
EXPECTED_SOURCE_LEDGER_VERSION = "prequential_diagnostic_challenger_ab12_v1"

DEFAULT_SOURCE_ROOT = Path(
    "maker/data/walkforward/prequential_challenger_ab12_60d_20260821"
)
DEFAULT_UNIVERSE_ROOT = Path(
    "maker/data/walkforward/liquidity/universe_manifest_v2"
)
DEFAULT_OUTPUT_ROOT = Path(
    "maker/data/walkforward/"
    "prequential_challenger_ab12_cost_caps_20260821_v1"
)

ARTIFACTS: Mapping[str, str] = {
    "price_source_inventory.parquet": "price_source_inventory",
    "path_transaction_costs.parquet": "path_costs",
    "transaction_cost_summary.parquet": "cost_summary",
    "combined_position_limit_events.parquet": "cap_events",
    "combined_position_limit_sweep.parquet": "cap_summary",
}

_SOURCE_REQUIRED = {
    "Date",
    "ValueCode",
    "QuoteCode",
    "entry_route",
    "boundary_quantile",
    "entry_policy_generation_id",
    "entry_raw_order_fact_id",
    "exit_policy_trial_id",
    "position_established_ns",
    "filled_entry_outcome_category",
    "terminal_date",
    "exit_decision_time_ns",
    "gross_cycle_pnl_twd",
    "gross_cycle_bp",
    "normalization_notional_twd",
    "physical_entry_dependency_id",
    "policy_path_id",
    "completed_same_day",
    "completed_overnight",
    "terminal_cashflow_priced",
    "nominal_cancel_model_assumption",
    "pathwise_ev_ready",
    "joint_volume_allocated",
    "unresolved_cashflow_imputed",
    "diagnostic_challenger_is_selected_action",
    "production_strategy_go",
}

_EXACT_PRICE_REQUIRED = {
    "entry_spot_price",
    "entry_future_price",
    "entry_contract_size_shares",
    "exit_spot_price",
    "exit_future_price",
    "exact_price_source",
}

_PRICE_SOURCE_INVENTORY_SCHEMA = {
    "source_kind": pl.String,
    "Date": pl.String,
    "ValueCode": pl.String,
    "partition_complete_path": pl.String,
    "partition_complete_sha256": pl.String,
    "artifact_path": pl.String,
    "artifact_sha256": pl.String,
    "artifact_bytes": pl.Int64,
    "artifact_rows": pl.Int64,
    "artifact_columns": pl.Int64,
}


@dataclass(frozen=True)
class TransactionCostProfile:
    """Charges for one complete long-spot/short-futures paired cycle."""

    profile_id: str = "user_taiwan_stock_future_20260821_v1"
    spot_commission_listed_bp_per_side: float = 14.25
    spot_commission_multiplier: float = 0.12
    spot_sell_tax_bp: float = 30.0
    same_day_spot_sell_tax_multiplier: float = 0.5
    futures_tax_bp_per_side: float = 0.2
    futures_commission_twd_per_side: float = 20.0

    @property
    def spot_commission_bp_per_side(self) -> float:
        return (
            self.spot_commission_listed_bp_per_side
            * self.spot_commission_multiplier
        )

    @property
    def same_day_variable_cost_bp(self) -> float:
        return (
            2.0 * self.spot_commission_bp_per_side
            + self.spot_sell_tax_bp * self.same_day_spot_sell_tax_multiplier
            + 2.0 * self.futures_tax_bp_per_side
        )

    @property
    def overnight_variable_cost_bp(self) -> float:
        return (
            2.0 * self.spot_commission_bp_per_side
            + self.spot_sell_tax_bp
            + 2.0 * self.futures_tax_bp_per_side
        )

    @property
    def futures_round_trip_commission_twd(self) -> float:
        return 2.0 * self.futures_commission_twd_per_side

    def validate(self) -> None:
        if not self.profile_id:
            raise ValueError("transaction cost profile_id must be nonempty")
        names = (
            "spot_commission_listed_bp_per_side",
            "spot_commission_multiplier",
            "spot_sell_tax_bp",
            "same_day_spot_sell_tax_multiplier",
            "futures_tax_bp_per_side",
            "futures_commission_twd_per_side",
        )
        for name in names:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.same_day_spot_sell_tax_multiplier > 1:
            raise ValueError("same-day tax multiplier cannot exceed one")


@dataclass(frozen=True)
class CombinedCapConfig:
    portfolio_caps_twd: tuple[float, ...] = (
        10_000_000.0,
        20_000_000.0,
        30_000_000.0,
    )
    per_product_fraction: float = 0.30
    cost_profile: TransactionCostProfile = TransactionCostProfile()

    def validate(self) -> None:
        self.cost_profile.validate()
        caps = tuple(float(value) for value in self.portfolio_caps_twd)
        if (
            not caps
            or any(not math.isfinite(value) or value <= 0 for value in caps)
            or tuple(sorted(caps)) != caps
            or len(set(caps)) != len(caps)
        ):
            raise ValueError(
                "portfolio caps must be nonempty, finite, positive, ascending, "
                "and unique"
            )
        fraction = float(self.per_product_fraction)
        if not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError("per_product_fraction must be in (0, 1]")


@dataclass(frozen=True)
class CombinedCostCapResult:
    price_source_inventory: pl.DataFrame
    path_costs: pl.DataFrame
    cost_summary: pl.DataFrame
    cap_events: pl.DataFrame
    cap_summary: pl.DataFrame


@dataclass(frozen=True)
class VerifiedSource:
    paths: pl.DataFrame
    price_source_inventory: pl.DataFrame
    metadata: Mapping[str, object]


def build_combined_cost_cap_sweep(
    paths: pl.DataFrame,
    *,
    config: CombinedCapConfig = CombinedCapConfig(),
    price_source_inventory: pl.DataFrame | None = None,
) -> CombinedCostCapResult:
    """Apply costs and replay simultaneous portfolio/product admission caps."""

    config.validate()
    source = _normalise_paths(paths)
    path_costs = _apply_transaction_costs(source, config.cost_profile)
    cost_summary = _transaction_cost_summary(path_costs, config.cost_profile)
    cap_events, cap_summary = _combined_cap_sweep(path_costs, config)
    if price_source_inventory is None:
        price_source_inventory = pl.DataFrame(
            schema=_PRICE_SOURCE_INVENTORY_SCHEMA
        )
    result = CombinedCostCapResult(
        price_source_inventory=price_source_inventory,
        path_costs=path_costs,
        cost_summary=cost_summary,
        cap_events=cap_events,
        cap_summary=cap_summary,
    )
    _validate_result(result, config)
    return result


def load_verified_source(
    source_root: Path = DEFAULT_SOURCE_ROOT,
    *,
    universe_root: Path = DEFAULT_UNIVERSE_ROOT,
) -> VerifiedSource:
    root = Path(source_root).resolve()
    marker_path = root / "complete.json"
    selected_path = root / "selected_policy_paths.parquet"
    marker = _read_json(marker_path)
    declared_marker_sha = marker.get("marker_payload_sha256")
    unhashed = dict(marker)
    unhashed.pop("marker_payload_sha256", None)
    if (
        marker.get("complete") is not True
        or marker.get("schema_version") != EXPECTED_SOURCE_SCHEMA_VERSION
        or marker.get("ledger_version") != EXPECTED_SOURCE_LEDGER_VERSION
        or declared_marker_sha != _canonical_sha256(unhashed)
    ):
        raise ValueError("source challenger marker is invalid")
    semantics = marker.get("fact_semantics")
    if (
        not isinstance(semantics, dict)
        or semantics.get("analysis_only") is not True
        or semantics.get("diagnostic_challenger_is_selected_action") is not False
        or semantics.get("best_q_selection_go") is not False
        or semantics.get("pathwise_ev_ready") is not False
        or semantics.get("production_strategy_go") is not False
        or semantics.get("unresolved_cashflow_imputed") is not False
    ):
        raise ValueError("source challenger safety contract changed")
    artifacts = marker.get("artifacts")
    declaration = (
        artifacts.get("selected_policy_paths.parquet")
        if isinstance(artifacts, dict)
        else None
    )
    if not isinstance(declaration, dict) or not selected_path.is_file():
        raise ValueError("source selected paths declaration is missing")
    schema = {
        name: str(dtype)
        for name, dtype in pl.read_parquet_schema(selected_path).items()
    }
    if (
        _file_sha256(selected_path) != declaration.get("sha256")
        or selected_path.stat().st_size != int(declaration.get("bytes", -1))
        or schema != declaration.get("schema")
    ):
        raise ValueError("source selected paths artifact changed")
    paths = pl.read_parquet(selected_path)
    if (
        paths.height != int(declaration.get("rows", -1))
        or paths.width != int(declaration.get("columns", -1))
    ):
        raise ValueError("source selected paths dimensions changed")
    source_locations = marker.get("sources")
    if not isinstance(source_locations, dict):
        raise ValueError("source challenger locations are missing")
    execution_root = Path(str(source_locations["execution_root"])).resolve()
    post_cross_root = Path(str(source_locations["post_cross_root"])).resolve()
    post_marker_path = post_cross_root / "complete.json"
    if (
        _file_sha256(post_marker_path)
        != source_locations.get("post_cross_complete_sha256")
    ):
        raise ValueError("source post-cross marker changed")
    post_marker = _read_json(post_marker_path)
    post_metadata = post_marker.get("metadata")
    if not isinstance(post_metadata, dict):
        raise ValueError("post-cross source metadata is missing")
    same_day_root = Path(str(post_metadata["exit_maker_root"])).resolve()
    cross_session_root = Path(str(post_metadata["cross_session_root"])).resolve()
    execution_manifest_path = execution_root / "execution_partition_manifest.parquet"
    same_day_manifest_path = same_day_root / "exit_maker_partition_manifest.parquet"
    cross_manifest_path = (
        cross_session_root / "cross_session_partition_manifest.parquet"
    )
    if (
        _file_sha256(execution_manifest_path)
        != source_locations.get("execution_manifest_sha256")
        or _file_sha256(same_day_manifest_path)
        != post_metadata.get("same_day_root_manifest_sha256")
        or _file_sha256(cross_manifest_path)
        != post_metadata.get("cross_session_manifest_sha256")
    ):
        raise ValueError("one or more exact-price source manifests changed")
    enriched, price_inventory = _load_exact_price_facts(
        paths,
        execution_manifest_path=execution_manifest_path,
        same_day_manifest_path=same_day_manifest_path,
        cross_manifest_path=cross_manifest_path,
    )
    universe_metadata = _verify_universe_source(
        Path(universe_root).resolve(),
        expected_value_codes={str(value) for value in post_metadata["value_codes"]},
    )
    metadata = {
        "source_root": str(root),
        "source_complete_sha256": _file_sha256(marker_path),
        "source_marker_payload_sha256": declared_marker_sha,
        "source_selected_paths_sha256": declaration["sha256"],
        "source_selected_paths_rows": paths.height,
        "source_selected_paths_columns": paths.width,
        "execution_root": str(execution_root),
        "execution_manifest_sha256": _file_sha256(execution_manifest_path),
        "same_day_exit_root": str(same_day_root),
        "same_day_exit_manifest_sha256": _file_sha256(same_day_manifest_path),
        "cross_session_exit_root": str(cross_session_root),
        "cross_session_exit_manifest_sha256": _file_sha256(cross_manifest_path),
        **universe_metadata,
    }
    return VerifiedSource(
        paths=enriched,
        price_source_inventory=price_inventory,
        metadata=metadata,
    )


def _load_exact_price_facts(
    paths: pl.DataFrame,
    *,
    execution_manifest_path: Path,
    same_day_manifest_path: Path,
    cross_manifest_path: Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    completed = paths.filter(pl.col("terminal_cashflow_priced"))
    if completed.is_empty():
        raise ValueError("source has no completed paths to price")
    execution_manifest = _partition_manifest(execution_manifest_path)
    same_day_manifest = _partition_manifest(same_day_manifest_path)
    cross_manifest = _partition_manifest(cross_manifest_path)
    inventory_rows: list[dict[str, object]] = []
    entry_frames: list[pl.DataFrame] = []
    same_day_frames: list[pl.DataFrame] = []
    overnight_frames: list[pl.DataFrame] = []
    for group in completed.partition_by(["Date", "ValueCode"], maintain_order=True):
        date = str(group.item(0, "Date"))
        product = str(group.item(0, "ValueCode"))
        key = (date, product)
        entry_ids = group["entry_policy_generation_id"].unique().to_list()
        entry, inventory = _read_bound_partition_artifact(
            execution_manifest,
            key,
            filename="execution_action_facts.parquet",
            source_kind="entry_execution_action",
            columns=(
                "policy_generation_id",
                "route",
                "entry_spot_price",
                "entry_future_price",
                "entry_hedge_contract_size_shares",
                "full_fill",
                "entry_hedge_executable",
            ),
            filter_column="policy_generation_id",
            identifiers=entry_ids,
        )
        entry_frames.append(
            entry.rename(
                {
                    "policy_generation_id": "entry_policy_generation_id",
                    "route": "exact_price_entry_route",
                    "entry_hedge_contract_size_shares": (
                        "entry_contract_size_shares"
                    ),
                }
            )
        )
        inventory_rows.append(inventory)
        same = group.filter(pl.col("completed_same_day"))
        if same.height:
            frame, item_inventory = _read_bound_partition_artifact(
                same_day_manifest,
                key,
                filename="exit_maker_position_policy_facts.parquet",
                source_kind="same_day_exit_policy_fact",
                columns=(
                    "exit_policy_trial_id",
                    "terminal_outcome",
                    "exit_decision_time_ns",
                    "exit_spot_price",
                    "exit_future_price",
                    "gross_cycle_pnl_twd",
                ),
                filter_column="exit_policy_trial_id",
                identifiers=same["exit_policy_trial_id"].unique().to_list(),
            )
            same_day_frames.append(
                frame.with_columns(
                    pl.lit(date).alias("exact_price_terminal_date"),
                    pl.lit("same_day_exit_maker_position_policy_fact").alias(
                        "exact_price_source"
                    ),
                ).rename(
                    {
                        "exit_decision_time_ns": (
                            "exact_price_exit_decision_time_ns"
                        ),
                        "gross_cycle_pnl_twd": "exact_price_gross_cycle_pnl_twd",
                    }
                ).select(
                    "exit_policy_trial_id",
                    "terminal_outcome",
                    "exact_price_terminal_date",
                    "exact_price_exit_decision_time_ns",
                    "exit_spot_price",
                    "exit_future_price",
                    "exact_price_gross_cycle_pnl_twd",
                    "exact_price_source",
                )
            )
            inventory_rows.append(item_inventory)
        overnight = group.filter(pl.col("completed_overnight"))
        if overnight.height:
            frame, item_inventory = _read_bound_partition_artifact(
                cross_manifest,
                key,
                filename="cross_session_nominal_policy_outcomes.parquet",
                source_kind="cross_session_nominal_exit_outcome",
                columns=(
                    "exit_policy_trial_id",
                    "terminal_cashflow_priced",
                    "terminal_date",
                    "exit_decision_time_ns",
                    "exit_spot_price",
                    "exit_future_price",
                    "gross_cycle_pnl_twd",
                ),
                filter_column="exit_policy_trial_id",
                identifiers=overnight["exit_policy_trial_id"].unique().to_list(),
            )
            overnight_frames.append(
                frame.rename(
                    {
                        "terminal_cashflow_priced": "terminal_outcome",
                        "terminal_date": "exact_price_terminal_date",
                        "exit_decision_time_ns": (
                            "exact_price_exit_decision_time_ns"
                        ),
                        "gross_cycle_pnl_twd": "exact_price_gross_cycle_pnl_twd",
                    }
                ).with_columns(
                    pl.lit("cross_session_nominal_policy_outcome").alias(
                        "exact_price_source"
                    )
                ).select(
                    "exit_policy_trial_id",
                    "terminal_outcome",
                    "exact_price_terminal_date",
                    "exact_price_exit_decision_time_ns",
                    "exit_spot_price",
                    "exit_future_price",
                    "exact_price_gross_cycle_pnl_twd",
                    "exact_price_source",
                )
            )
            inventory_rows.append(item_inventory)
    entries = pl.concat(entry_frames, how="vertical_relaxed")
    exits = pl.concat([*same_day_frames, *overnight_frames], how="vertical_relaxed")
    if (
        entries.height != completed.height
        or entries["entry_policy_generation_id"].n_unique() != completed.height
        or exits.height != completed.height
        or exits["exit_policy_trial_id"].n_unique() != completed.height
    ):
        raise ValueError("exact-price source join is not one-to-one")
    price_facts = (
        completed.select(
            "policy_path_id",
            "entry_policy_generation_id",
            "exit_policy_trial_id",
            "entry_route",
            "terminal_date",
            "exit_decision_time_ns",
            "gross_cycle_pnl_twd",
            "normalization_notional_twd",
        )
        .join(entries, on="entry_policy_generation_id", how="left", validate="1:1")
        .join(exits, on="exit_policy_trial_id", how="left", validate="1:1")
    )
    invalid = price_facts.filter(
        pl.any_horizontal(
            pl.col(
                "entry_spot_price",
                "entry_future_price",
                "entry_contract_size_shares",
                "exit_spot_price",
                "exit_future_price",
            ).is_null()
        )
        | ~pl.all_horizontal(
            pl.col(
                "entry_spot_price",
                "entry_future_price",
                "exit_spot_price",
                "exit_future_price",
            ).is_finite()
        )
        | pl.any_horizontal(
            pl.col(
                "entry_spot_price",
                "entry_future_price",
                "entry_contract_size_shares",
                "exit_spot_price",
                "exit_future_price",
            )
            <= 0
        )
        | (pl.col("exact_price_entry_route") != pl.col("entry_route"))
        | (pl.col("full_fill") != True).fill_null(True)  # noqa: E712
        | (pl.col("entry_hedge_executable") != True).fill_null(True)  # noqa: E712
        | (
            (
                pl.col("entry_contract_size_shares")
                * pl.col("entry_spot_price")
                - pl.col("normalization_notional_twd")
            ).abs()
            > 1e-6
        )
        | (
            (
                pl.col("exact_price_gross_cycle_pnl_twd")
                - pl.col("gross_cycle_pnl_twd")
            ).abs()
            > 1e-6
        )
        | (pl.col("exact_price_terminal_date") != pl.col("terminal_date"))
        | (
            pl.col("exact_price_exit_decision_time_ns")
            != pl.col("exit_decision_time_ns")
        )
    )
    if invalid.height:
        sample = invalid.select(
            "policy_path_id",
            "entry_route",
            "exact_price_entry_route",
            "normalization_notional_twd",
            "entry_spot_price",
            "entry_contract_size_shares",
            "gross_cycle_pnl_twd",
            "exact_price_gross_cycle_pnl_twd",
            "terminal_date",
            "exact_price_terminal_date",
            "exit_decision_time_ns",
            "exact_price_exit_decision_time_ns",
            "terminal_outcome",
        ).head(5).to_dicts()
        raise ValueError(
            "exact entry/exit price facts do not reconcile to paths: "
            f"invalid={invalid.height}, sample={sample}"
        )
    exact = price_facts.select(
        "policy_path_id",
        "entry_spot_price",
        "entry_future_price",
        "entry_contract_size_shares",
        "exit_spot_price",
        "exit_future_price",
        "exact_price_source",
    )
    enriched = paths.join(exact, on="policy_path_id", how="left", validate="1:1")
    inventory = pl.from_dicts(
        inventory_rows,
        schema=_PRICE_SOURCE_INVENTORY_SCHEMA,
        infer_schema_length=None,
    ).sort(["source_kind", "Date", "ValueCode"])
    if inventory.height != len(inventory_rows):
        raise ValueError("price source inventory construction failed")
    return enriched, inventory


def _partition_manifest(path: Path) -> dict[tuple[str, str], Path]:
    frame = pl.read_parquet(path)
    required = {"Date", "ValueCode", "partition", "complete"}
    if required - set(frame.columns) or frame.filter(~pl.col("complete")).height:
        raise ValueError(f"partition manifest is incomplete: {path}")
    result: dict[tuple[str, str], Path] = {}
    for row in frame.iter_rows(named=True):
        key = (str(row["Date"]), str(row["ValueCode"]))
        partition = Path(str(row["partition"]))
        if not partition.is_absolute():
            partition = (Path.cwd() / partition).resolve()
        else:
            partition = partition.resolve()
        if key in result:
            raise ValueError(f"partition manifest key is duplicated: {key}")
        result[key] = partition
    return result


def _read_bound_partition_artifact(
    manifest: Mapping[tuple[str, str], Path],
    key: tuple[str, str],
    *,
    filename: str,
    source_kind: str,
    columns: Sequence[str],
    filter_column: str,
    identifiers: Sequence[str],
) -> tuple[pl.DataFrame, dict[str, object]]:
    if key not in manifest:
        raise ValueError(f"price source partition is missing: {key}")
    partition = manifest[key]
    marker_path = partition / "complete.json"
    artifact_path = partition / filename
    marker = _read_json(marker_path)
    declaration = (
        marker.get("artifacts", {}).get(filename)
        if isinstance(marker.get("artifacts"), dict)
        else None
    )
    if (
        marker.get("complete") is not True
        or str(marker.get("Date")) != key[0]
        or str(marker.get("ValueCode")) != key[1]
        or not isinstance(declaration, dict)
        or _file_sha256(artifact_path) != declaration.get("sha256")
        or artifact_path.stat().st_size != int(declaration.get("bytes", -1))
    ):
        raise ValueError(f"price source artifact changed: {artifact_path}")
    frame = pl.read_parquet(artifact_path, columns=list(columns)).filter(
        pl.col(filter_column).is_in(list(identifiers))
    )
    if frame.height != len(set(identifiers)):
        raise ValueError(f"price source identifiers are not one-to-one: {artifact_path}")
    inventory = {
        "source_kind": source_kind,
        "Date": key[0],
        "ValueCode": key[1],
        "partition_complete_path": str(marker_path),
        "partition_complete_sha256": _file_sha256(marker_path),
        "artifact_path": str(artifact_path),
        "artifact_sha256": declaration["sha256"],
        "artifact_bytes": int(declaration["bytes"]),
        "artifact_rows": int(declaration["rows"]),
        "artifact_columns": int(declaration["columns"]),
    }
    return frame, inventory


def _verify_universe_source(
    root: Path, *, expected_value_codes: set[str]
) -> dict[str, object]:
    marker_path = root / "complete.json"
    marker = _read_json(marker_path)
    declared_sha = marker.get("marker_payload_sha256")
    unhashed = dict(marker)
    unhashed.pop("marker_payload_sha256", None)
    semantics = marker.get("semantics")
    if (
        marker.get("complete") is not True
        or declared_sha != _canonical_sha256(unhashed)
        or not isinstance(semantics, dict)
        or semantics.get("retrospective_research_selection") is not True
        or semantics.get("selection_contains_target_day_outcomes") is not True
        or semantics.get("production_universe_approved") is not False
        or semantics.get("runtime_daily_liquidity_gate_required") is not True
    ):
        raise ValueError("retrospective universe source contract changed")
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("universe artifacts are missing")
    required = ("first_wave_45.csv", "research_universe_manifest.parquet")
    hashes: dict[str, str] = {}
    for filename in required:
        declaration = artifacts.get(filename)
        path = root / filename
        if (
            not isinstance(declaration, dict)
            or _file_sha256(path) != declaration.get("sha256")
            or path.stat().st_size != int(declaration.get("bytes", -1))
        ):
            raise ValueError(f"universe artifact changed: {filename}")
        hashes[filename] = str(declaration["sha256"])
    first_wave = set(
        pl.read_csv(root / "first_wave_45.csv", schema_overrides={"ValueCode": pl.String})[
            "ValueCode"
        ].to_list()
    )
    if first_wave != expected_value_codes:
        raise ValueError("formal 45-product cohort differs from retrospective universe")
    return {
        "universe_root": str(root),
        "universe_complete_sha256": _file_sha256(marker_path),
        "universe_marker_payload_sha256": declared_sha,
        "universe_config_sha256": marker.get("config_sha256"),
        "universe_first_wave_45_sha256": hashes["first_wave_45.csv"],
        "universe_research_manifest_sha256": hashes[
            "research_universe_manifest.parquet"
        ],
        "universe_selection_d_safe_go": False,
        "universe_deployment_go": False,
    }


def run_combined_cost_cap_sweep(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    universe_root: Path = DEFAULT_UNIVERSE_ROOT,
    config: CombinedCapConfig = CombinedCapConfig(),
) -> CombinedCostCapResult:
    source = load_verified_source(source_root, universe_root=universe_root)
    result = build_combined_cost_cap_sweep(
        source.paths,
        config=config,
        price_source_inventory=source.price_source_inventory,
    )
    publish_combined_cost_cap_bundle(
        output_root,
        result,
        source.metadata,
        config=config,
    )
    return result


def publish_combined_cost_cap_bundle(
    output_root: Path,
    result: CombinedCostCapResult,
    source_metadata: Mapping[str, object],
    *,
    config: CombinedCapConfig = CombinedCapConfig(),
) -> None:
    """Atomically publish a self-hashed, source-bound supplemental bundle."""

    config.validate()
    _validate_result(result, config)
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
            "sweep_version": SWEEP_VERSION,
            "config": _config_payload(config),
            "sources": json.loads(json.dumps(dict(source_metadata), sort_keys=True)),
            "implementation_sources": _implementation_sources(),
            "fact_semantics": _fact_semantics(),
            "artifacts": declarations,
        }
        marker["marker_payload_sha256"] = _canonical_sha256(marker)
        (stage / "complete.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _verify_published_files(stage, verify_source=True)
        stage.rename(destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def verify_combined_cost_cap_bundle(
    output_root: Path,
    *,
    rebuild: bool = True,
) -> dict[str, object]:
    marker, frames = _verify_published_files(output_root, verify_source=True)
    config = _config_from_payload(marker["config"])
    result = CombinedCostCapResult(
        **{attribute: frames[filename] for filename, attribute in ARTIFACTS.items()}
    )
    _validate_result(result, config)
    if rebuild:
        source = load_verified_source(
            Path(str(marker["sources"]["source_root"])),
            universe_root=Path(str(marker["sources"]["universe_root"])),
        )
        if dict(source.metadata) != marker["sources"]:
            raise ValueError("source metadata changed")
        expected = build_combined_cost_cap_sweep(
            source.paths,
            config=config,
            price_source_inventory=source.price_source_inventory,
        )
        for filename, attribute in ARTIFACTS.items():
            actual_frame = frames[filename]
            expected_frame = getattr(expected, attribute)
            if actual_frame.schema != expected_frame.schema or not actual_frame.equals(
                expected_frame, null_equal=True
            ):
                raise ValueError(f"source rebuild differs: {filename}")
    return marker


def _normalise_paths(paths: pl.DataFrame) -> pl.DataFrame:
    missing = sorted((_SOURCE_REQUIRED | _EXACT_PRICE_REQUIRED) - set(paths.columns))
    if missing:
        raise ValueError(f"selected paths missing columns: {missing}")
    result = paths.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("terminal_date").cast(pl.String),
        pl.col("normalization_notional_twd").cast(pl.Float64),
    ).sort(["Date", "position_established_ns", "ValueCode", "policy_path_id"])
    if result.is_empty() or result["policy_path_id"].n_unique() != result.height:
        raise ValueError("selected paths must be nonempty and unique")
    invalid_notional = result.filter(
        pl.col("normalization_notional_twd").is_null()
        | ~pl.col("normalization_notional_twd").is_finite()
        | (pl.col("normalization_notional_twd") <= 0)
    )
    if invalid_notional.height:
        raise ValueError("selected path notional must be finite and positive")
    allowed = {"completed", "censored", "unknown", "still_open"}
    if not set(result["filled_entry_outcome_category"].unique().to_list()) <= allowed:
        raise ValueError("selected paths contain an unsupported outcome category")
    completed = result.filter(pl.col("filled_entry_outcome_category") == "completed")
    unresolved = result.filter(pl.col("filled_entry_outcome_category") != "completed")
    invalid_completed = completed.filter(
        (pl.col("terminal_cashflow_priced") != True).fill_null(True)  # noqa: E712
        | (pl.col("completed_same_day") == pl.col("completed_overnight")).fill_null(
            True
        )
        | pl.col("terminal_date").is_null()
        | pl.col("exit_decision_time_ns").is_null()
        | pl.col("gross_cycle_pnl_twd").is_null()
        | ~pl.col("gross_cycle_pnl_twd").is_finite()
        | pl.col("gross_cycle_bp").is_null()
        | ~pl.col("gross_cycle_bp").is_finite()
    )
    if invalid_completed.height:
        raise ValueError("completed paths lack a unique priced terminal cashflow")
    invalid_exact_prices = completed.filter(
        pl.any_horizontal(
            pl.col(
                "entry_spot_price",
                "entry_future_price",
                "entry_contract_size_shares",
                "exit_spot_price",
                "exit_future_price",
                "exact_price_source",
            ).is_null()
        )
        | ~pl.all_horizontal(
            pl.col(
                "entry_spot_price",
                "entry_future_price",
                "exit_spot_price",
                "exit_future_price",
            ).is_finite()
        )
        | pl.any_horizontal(
            pl.col(
                "entry_spot_price",
                "entry_future_price",
                "entry_contract_size_shares",
                "exit_spot_price",
                "exit_future_price",
            )
            <= 0
        )
        | (
            (
                pl.col("entry_contract_size_shares")
                * pl.col("entry_spot_price")
                - pl.col("normalization_notional_twd")
            ).abs()
            > 1e-6
        )
    )
    if invalid_exact_prices.height:
        raise ValueError("completed paths lack reconciled exact leg prices")
    invalid_unresolved = unresolved.filter(
        (pl.col("terminal_cashflow_priced") != False).fill_null(True)  # noqa: E712
        | pl.col("gross_cycle_pnl_twd").is_not_null()
        | pl.col("gross_cycle_bp").is_not_null()
    )
    if invalid_unresolved.height:
        raise ValueError("unresolved paths were assigned terminal cashflow")
    if unresolved.filter(
        pl.any_horizontal(
            pl.col(
                "entry_spot_price",
                "entry_future_price",
                "entry_contract_size_shares",
                "exit_spot_price",
                "exit_future_price",
                "exact_price_source",
            ).is_not_null()
        )
    ).height:
        raise ValueError("unresolved paths were assigned exact terminal prices")
    unsafe = result.filter(
        (pl.col("nominal_cancel_model_assumption") != True).fill_null(True)  # noqa: E712
        | (pl.col("pathwise_ev_ready") != False).fill_null(True)  # noqa: E712
        | (pl.col("joint_volume_allocated") != False).fill_null(True)  # noqa: E712
        | (pl.col("unresolved_cashflow_imputed") != False).fill_null(True)  # noqa: E712
        | (
            pl.col("diagnostic_challenger_is_selected_action") != False
        ).fill_null(True)  # noqa: E712
        | (pl.col("production_strategy_go") != False).fill_null(True)  # noqa: E712
    )
    if unsafe.height:
        raise ValueError("selected path safety flags changed")
    for row in completed.select(
        "Date", "position_established_ns", "terminal_date", "exit_decision_time_ns"
    ).iter_rows(named=True):
        if (str(row["terminal_date"]), int(row["exit_decision_time_ns"])) < (
            str(row["Date"]),
            int(row["position_established_ns"]),
        ):
            raise ValueError("completed path exits before it is established")
    return result


def _apply_transaction_costs(
    paths: pl.DataFrame, profile: TransactionCostProfile
) -> pl.DataFrame:
    completed = pl.col("filled_entry_outcome_category") == "completed"
    same_day = pl.col("completed_same_day")
    notional = pl.col("normalization_notional_twd")
    shares = pl.col("entry_contract_size_shares").cast(pl.Float64)
    commission_rate = profile.spot_commission_bp_per_side / 10_000.0
    spot_tax_rate = pl.when(same_day).then(
        profile.spot_sell_tax_bp
        * profile.same_day_spot_sell_tax_multiplier
        / 10_000.0
    ).otherwise(profile.spot_sell_tax_bp / 10_000.0)
    futures_tax_rate = profile.futures_tax_bp_per_side / 10_000.0
    reference_variable_bp = pl.when(same_day).then(
        pl.lit(profile.same_day_variable_cost_bp)
    ).otherwise(pl.lit(profile.overnight_variable_cost_bp))
    priced = paths.with_columns(
        pl.when(completed)
        .then(shares * pl.col("entry_spot_price") * commission_rate)
        .otherwise(None)
        .alias("spot_buy_commission_twd"),
        pl.when(completed)
        .then(shares * pl.col("exit_spot_price") * commission_rate)
        .otherwise(None)
        .alias("spot_sell_commission_twd"),
        pl.when(completed)
        .then(shares * pl.col("exit_spot_price") * spot_tax_rate)
        .otherwise(None)
        .alias("spot_sell_tax_twd"),
        pl.when(completed)
        .then(shares * pl.col("entry_future_price") * futures_tax_rate)
        .otherwise(None)
        .alias("futures_entry_tax_twd"),
        pl.when(completed)
        .then(shares * pl.col("exit_future_price") * futures_tax_rate)
        .otherwise(None)
        .alias("futures_exit_tax_twd"),
        pl.when(completed)
        .then(pl.lit(profile.futures_commission_twd_per_side))
        .otherwise(None)
        .alias("futures_entry_commission_twd"),
        pl.when(completed)
        .then(pl.lit(profile.futures_commission_twd_per_side))
        .otherwise(None)
        .alias("futures_exit_commission_twd"),
        pl.when(completed)
        .then(reference_variable_bp)
        .otherwise(None)
        .alias("near_flat_price_reference_variable_cost_bp"),
    ).with_columns(
        pl.when(completed)
        .then(
            pl.sum_horizontal(
                "spot_buy_commission_twd",
                "spot_sell_commission_twd",
                "spot_sell_tax_twd",
                "futures_entry_tax_twd",
                "futures_exit_tax_twd",
                "futures_entry_commission_twd",
                "futures_exit_commission_twd",
            )
        )
        .otherwise(None)
        .alias("total_transaction_cost_twd")
    )
    return (
        priced.with_columns(
            pl.when(completed)
            .then(pl.col("total_transaction_cost_twd") / notional * 10_000.0)
            .otherwise(None)
            .alias("effective_transaction_cost_bp"),
            pl.when(completed)
            .then(pl.col("gross_cycle_pnl_twd") - pl.col("total_transaction_cost_twd"))
            .otherwise(None)
            .alias("net_cycle_pnl_twd"),
            pl.lit(profile.profile_id).alias("transaction_cost_profile_id"),
            completed.alias("transaction_cost_point_identified"),
            pl.lit(True).alias("fee_tax_profile_complete_for_completed_cycle"),
            pl.lit(True).alias("cost_uses_exact_entry_exit_leg_prices"),
            pl.lit(False).alias("unresolved_cashflow_imputed_by_cost_sweep"),
            pl.lit(False).alias("full_strategy_cost_profile_complete"),
            pl.lit(False).alias("production_strategy_go_after_cost"),
        )
        .with_columns(
            pl.when(completed)
            .then(pl.col("net_cycle_pnl_twd") / notional * 10_000.0)
            .otherwise(None)
            .alias("net_cycle_pnl_bp")
        )
        .select(
            "Date",
            "ValueCode",
            "QuoteCode",
            "entry_route",
            "boundary_quantile",
            "position_established_ns",
            "filled_entry_outcome_category",
            "terminal_date",
            "exit_decision_time_ns",
            "gross_cycle_pnl_twd",
            "gross_cycle_bp",
            "normalization_notional_twd",
            "physical_entry_dependency_id",
            "policy_path_id",
            "entry_policy_generation_id",
            "entry_raw_order_fact_id",
            "exit_policy_trial_id",
            "completed_same_day",
            "completed_overnight",
            "terminal_cashflow_priced",
            "entry_spot_price",
            "entry_future_price",
            "entry_contract_size_shares",
            "exit_spot_price",
            "exit_future_price",
            "exact_price_source",
            "transaction_cost_profile_id",
            "spot_buy_commission_twd",
            "spot_sell_commission_twd",
            "spot_sell_tax_twd",
            "futures_entry_tax_twd",
            "futures_exit_tax_twd",
            "futures_entry_commission_twd",
            "futures_exit_commission_twd",
            "near_flat_price_reference_variable_cost_bp",
            "total_transaction_cost_twd",
            "effective_transaction_cost_bp",
            "net_cycle_pnl_twd",
            "net_cycle_pnl_bp",
            "transaction_cost_point_identified",
            "fee_tax_profile_complete_for_completed_cycle",
            "cost_uses_exact_entry_exit_leg_prices",
            "unresolved_cashflow_imputed_by_cost_sweep",
            "full_strategy_cost_profile_complete",
            "production_strategy_go_after_cost",
        )
    )


def _transaction_cost_summary(
    costs: pl.DataFrame, profile: TransactionCostProfile
) -> pl.DataFrame:
    completed = costs.filter(pl.col("transaction_cost_point_identified"))
    groups = (
        ("all_completed", completed),
        ("same_day_completed", completed.filter(pl.col("completed_same_day"))),
        ("overnight_completed", completed.filter(pl.col("completed_overnight"))),
        (
            "unresolved_null",
            costs.filter(~pl.col("transaction_cost_point_identified")),
        ),
    )
    rows: list[dict[str, object]] = []
    component_columns = (
        "spot_buy_commission_twd",
        "spot_sell_commission_twd",
        "spot_sell_tax_twd",
        "futures_entry_tax_twd",
        "futures_exit_tax_twd",
        "futures_entry_commission_twd",
        "futures_exit_commission_twd",
    )
    for label, frame in groups:
        priced = label != "unresolved_null"
        notional = _sum(frame, "normalization_notional_twd")
        row: dict[str, object] = {
            "summary_group": label,
            "positions": frame.height,
            "normalization_notional_twd": notional,
            "gross_cycle_pnl_twd": _sum(frame, "gross_cycle_pnl_twd") if priced else None,
            "total_transaction_cost_twd": (
                _sum(frame, "total_transaction_cost_twd") if priced else None
            ),
            "net_cycle_pnl_twd": _sum(frame, "net_cycle_pnl_twd") if priced else None,
            "notional_weighted_gross_bp": (
                _sum(frame, "gross_cycle_pnl_twd") / notional * 10_000.0
                if priced and notional
                else None
            ),
            "notional_weighted_effective_cost_bp": (
                _sum(frame, "total_transaction_cost_twd") / notional * 10_000.0
                if priced and notional
                else None
            ),
            "notional_weighted_net_bp": (
                _sum(frame, "net_cycle_pnl_twd") / notional * 10_000.0
                if priced and notional
                else None
            ),
            "arithmetic_mean_net_bp": (
                _mean(frame, "net_cycle_pnl_bp") if priced else None
            ),
            "transaction_cost_profile_id": profile.profile_id,
            "near_flat_price_reference_same_day_variable_cost_bp": (
                profile.same_day_variable_cost_bp
            ),
            "near_flat_price_reference_overnight_variable_cost_bp": (
                profile.overnight_variable_cost_bp
            ),
            "futures_round_trip_commission_twd": (
                profile.futures_round_trip_commission_twd
            ),
            "cost_uses_exact_entry_exit_leg_prices": True,
            "near_flat_price_reference_only": True,
            "terminal_cashflow_point_identified": priced,
            "unresolved_cashflow_imputed": False,
            "production_strategy_go": False,
        }
        for column in component_columns:
            row[column] = _sum(frame, column) if priced else None
        rows.append(row)
    return pl.from_dicts(rows, infer_schema_length=None)


def _combined_cap_sweep(
    costs: pl.DataFrame, config: CombinedCapConfig
) -> tuple[pl.DataFrame, pl.DataFrame]:
    ordered = costs.sort(
        ["Date", "position_established_ns", "ValueCode", "policy_path_id"]
    )
    event_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for portfolio_cap in config.portfolio_caps_twd:
        portfolio_cap = float(portfolio_cap)
        product_cap = portfolio_cap * config.per_product_fraction
        scenario_id = f"portfolio_{int(portfolio_cap)}_product_{int(product_cap)}"
        active_heap: list[tuple[str, int, str, float, str]] = []
        active_count = 0
        active_notional = 0.0
        active_count_by_product: dict[str, int] = {}
        active_notional_by_product: dict[str, float] = {}
        accepted = completed = censored = unresolved = 0
        rejected_portfolio_only = rejected_product_only = rejected_both = 0
        gross = transaction_cost = net = accepted_entry_notional = 0.0
        accepted_completed_notional = 0.0
        peak_count = 0
        peak_notional = 0.0
        peak_product_count = 0
        peak_product_notional = 0.0
        for sequence, item in enumerate(ordered.iter_rows(named=True), start=1):
            entry_key = (str(item["Date"]), int(item["position_established_ns"]))
            released_count = 0
            released_notional = 0.0
            released_product_count = 0
            released_product_notional = 0.0
            product = str(item["ValueCode"])
            while active_heap and (active_heap[0][0], active_heap[0][1]) < entry_key:
                _, _, _, released, released_product = heapq.heappop(active_heap)
                active_count -= 1
                active_notional -= released
                active_count_by_product[released_product] -= 1
                active_notional_by_product[released_product] -= released
                released_count += 1
                released_notional += released
                if released_product == product:
                    released_product_count += 1
                    released_product_notional += released
            before_count = active_count
            before_notional = active_notional
            before_product_count = active_count_by_product.get(product, 0)
            before_product_notional = active_notional_by_product.get(product, 0.0)
            notional = float(item["normalization_notional_twd"])
            prospective_notional = before_notional + notional
            prospective_product_notional = before_product_notional + notional
            portfolio_blocked = prospective_notional > portfolio_cap + 1e-9
            product_blocked = prospective_product_notional > product_cap + 1e-9
            admitted = not portfolio_blocked and not product_blocked
            if admitted:
                reason = "accepted"
                accepted += 1
                accepted_entry_notional += notional
                active_count += 1
                active_notional += notional
                active_count_by_product[product] = before_product_count + 1
                active_notional_by_product[product] = prospective_product_notional
                peak_count = max(peak_count, active_count)
                peak_notional = max(peak_notional, active_notional)
                peak_product_count = max(
                    peak_product_count, active_count_by_product[product]
                )
                peak_product_notional = max(
                    peak_product_notional, active_notional_by_product[product]
                )
                category = str(item["filled_entry_outcome_category"])
                if category == "completed":
                    completed += 1
                    accepted_completed_notional += notional
                    gross += float(item["gross_cycle_pnl_twd"])
                    transaction_cost += float(item["total_transaction_cost_twd"])
                    net += float(item["net_cycle_pnl_twd"])
                    heapq.heappush(
                        active_heap,
                        (
                            str(item["terminal_date"]),
                            int(item["exit_decision_time_ns"]),
                            str(item["policy_path_id"]),
                            notional,
                            product,
                        ),
                    )
                elif category == "censored":
                    censored += 1
                else:
                    unresolved += 1
            elif portfolio_blocked and product_blocked:
                reason = "both"
                rejected_both += 1
            elif portfolio_blocked:
                reason = "portfolio_only"
                rejected_portfolio_only += 1
            else:
                reason = "product_only"
                rejected_product_only += 1
            event_rows.append(
                {
                    "scenario_id": scenario_id,
                    "portfolio_cap_twd": portfolio_cap,
                    "per_product_cap_fraction": config.per_product_fraction,
                    "per_product_cap_twd": product_cap,
                    "candidate_sequence": sequence,
                    "Date": item["Date"],
                    "position_established_ns": item["position_established_ns"],
                    "ValueCode": product,
                    "QuoteCode": item["QuoteCode"],
                    "policy_path_id": item["policy_path_id"],
                    "filled_entry_outcome_category": item[
                        "filled_entry_outcome_category"
                    ],
                    "normalization_notional_twd": notional,
                    "released_completed_positions_before_entry": released_count,
                    "released_completed_notional_before_entry_twd": released_notional,
                    "released_same_product_positions_before_entry": (
                        released_product_count
                    ),
                    "released_same_product_notional_before_entry_twd": (
                        released_product_notional
                    ),
                    "active_positions_before_entry": before_count,
                    "active_portfolio_notional_before_entry_twd": before_notional,
                    "active_same_product_positions_before_entry": (
                        before_product_count
                    ),
                    "active_same_product_notional_before_entry_twd": (
                        before_product_notional
                    ),
                    "prospective_portfolio_notional_twd": prospective_notional,
                    "prospective_same_product_notional_twd": (
                        prospective_product_notional
                    ),
                    "portfolio_cap_blocked": portfolio_blocked,
                    "per_product_cap_blocked": product_blocked,
                    "admission_status": reason,
                    "accepted": admitted,
                    "active_positions_after_decision": active_count,
                    "active_portfolio_notional_after_decision_twd": active_notional,
                    "active_same_product_positions_after_decision": (
                        active_count_by_product.get(product, 0)
                    ),
                    "active_same_product_notional_after_decision_twd": (
                        active_notional_by_product.get(product, 0.0)
                    ),
                    "accepted_completed_gross_twd": (
                        item["gross_cycle_pnl_twd"]
                        if admitted and item["terminal_cashflow_priced"]
                        else None
                    ),
                    "accepted_completed_transaction_cost_twd": (
                        item["total_transaction_cost_twd"]
                        if admitted and item["terminal_cashflow_priced"]
                        else None
                    ),
                    "accepted_completed_net_twd": (
                        item["net_cycle_pnl_twd"]
                        if admitted and item["terminal_cashflow_priced"]
                        else None
                    ),
                    "unresolved_positions_never_release_capacity": True,
                    "tie_handling": "entry_before_exit_at_equal_recv_time_ns",
                    "notional_basis": "one_way_entry_normalization_notional_twd",
                    "position_limit_sweep_analysis_only": True,
                    "position_limit_sweep_production_ready": False,
                }
            )
        summary_rows.append(
            {
                "scenario_id": scenario_id,
                "portfolio_cap_twd": portfolio_cap,
                "per_product_cap_fraction": config.per_product_fraction,
                "per_product_cap_twd": product_cap,
                "candidate_positions": ordered.height,
                "accepted_positions": accepted,
                "rejected_positions": ordered.height - accepted,
                "acceptance_rate": accepted / ordered.height,
                "rejected_portfolio_only": rejected_portfolio_only,
                "rejected_product_only": rejected_product_only,
                "rejected_both_caps": rejected_both,
                "accepted_completed_cycles": completed,
                "accepted_censored_positions": censored,
                "accepted_unknown_or_open_positions": unresolved,
                "accepted_completed_normalization_notional_twd": (
                    accepted_completed_notional
                ),
                "accepted_completed_gross_twd": gross,
                "accepted_completed_transaction_cost_twd": transaction_cost,
                "accepted_completed_net_twd": net,
                "accepted_completed_weighted_gross_bp": (
                    gross / accepted_completed_notional * 10_000.0
                    if accepted_completed_notional
                    else None
                ),
                "accepted_completed_weighted_cost_bp": (
                    transaction_cost / accepted_completed_notional * 10_000.0
                    if accepted_completed_notional
                    else None
                ),
                "accepted_completed_weighted_net_bp": (
                    net / accepted_completed_notional * 10_000.0
                    if accepted_completed_notional
                    else None
                ),
                "accepted_new_entry_one_way_notional_twd": accepted_entry_notional,
                "peak_concurrent_positions": peak_count,
                "peak_outstanding_one_way_entry_notional_twd": peak_notional,
                "peak_single_product_concurrent_positions": peak_product_count,
                "peak_single_product_outstanding_one_way_entry_notional_twd": (
                    peak_product_notional
                ),
                "portfolio_cap_compliant": peak_notional <= portfolio_cap + 1e-9,
                "per_product_cap_compliant": (
                    peak_product_notional <= product_cap + 1e-9
                ),
                "accepted_terminal_cashflow_point_identified": (
                    accepted == completed
                ),
                "unresolved_cashflow_imputed": False,
                "unresolved_positions_never_release_capacity": True,
                "tie_handling": "entry_before_exit_at_equal_recv_time_ns",
                "notional_basis": "one_way_entry_normalization_notional_twd",
                "source_universe_d_safe": False,
                "best_q_selection_go": False,
                "position_limit_sweep_analysis_only": True,
                "position_limit_sweep_production_ready": False,
                "production_strategy_go": False,
            }
        )
    events = pl.from_dicts(event_rows, infer_schema_length=None)
    summary = pl.from_dicts(summary_rows, infer_schema_length=None).sort(
        "portfolio_cap_twd"
    )
    base = summary.row(0, named=True)
    base_cap = float(base["portfolio_cap_twd"])
    base_accepted = int(base["accepted_positions"])
    base_net = float(base["accepted_completed_net_twd"])
    summary = summary.with_columns(
        (pl.col("portfolio_cap_twd") / base_cap).alias("cap_multiple_vs_smallest"),
        (
            pl.col("accepted_positions")
            / (base_accepted * pl.col("portfolio_cap_twd") / base_cap)
        ).alias("accepted_vs_linear_from_smallest_ratio"),
        pl.when(pl.lit(abs(base_net) > 1e-12))
        .then(
            pl.col("accepted_completed_net_twd")
            / (base_net * pl.col("portfolio_cap_twd") / base_cap)
        )
        .otherwise(None)
        .alias("completed_net_vs_linear_from_smallest_ratio"),
        pl.lit(False).alias("strict_linearity_expected"),
    )
    return events, summary


def _validate_result(result: CombinedCostCapResult, config: CombinedCapConfig) -> None:
    for attribute in ARTIFACTS.values():
        if not isinstance(getattr(result, attribute), pl.DataFrame):
            raise TypeError(f"bundle artifact is not a DataFrame: {attribute}")
    costs = result.path_costs
    inventory = result.price_source_inventory
    events = result.cap_events
    summary = result.cap_summary
    if costs["policy_path_id"].n_unique() != costs.height:
        raise ValueError("path costs are not unique")
    if inventory.schema != _PRICE_SOURCE_INVENTORY_SCHEMA:
        raise ValueError("price source inventory schema differs")
    if inventory.height and inventory.select(
        "source_kind", "Date", "ValueCode"
    ).n_unique() != inventory.height:
        raise ValueError("price source inventory keys are duplicated")
    completed = costs.filter(pl.col("transaction_cost_point_identified"))
    unresolved = costs.filter(~pl.col("transaction_cost_point_identified"))
    if completed.filter(
        pl.col("total_transaction_cost_twd").is_null()
        | pl.col("net_cycle_pnl_twd").is_null()
        | (pl.col("cost_uses_exact_entry_exit_leg_prices") != True).fill_null(  # noqa: E712
            True
        )
    ).height:
        raise ValueError("completed path costs are null")
    if unresolved.filter(
        pl.col("total_transaction_cost_twd").is_not_null()
        | pl.col("net_cycle_pnl_twd").is_not_null()
    ).height:
        raise ValueError("unresolved path costs were imputed")
    if summary.height != len(config.portfolio_caps_twd):
        raise ValueError("cap summary scenario count differs")
    if events.height != costs.height * summary.height:
        raise ValueError("cap event grid is incomplete")
    if events.group_by("scenario_id").len()["len"].n_unique() != 1:
        raise ValueError("cap event scenario grids differ")
    invalid_events = events.filter(
        (pl.col("accepted") & (pl.col("admission_status") != "accepted"))
        | (
            ~pl.col("accepted")
            & ~pl.col("admission_status").is_in(
                ["portfolio_only", "product_only", "both"]
            )
        )
        | (
            pl.col("active_portfolio_notional_after_decision_twd")
            > pl.col("portfolio_cap_twd") + 1e-9
        )
        | (
            pl.col("active_same_product_notional_after_decision_twd")
            > pl.col("per_product_cap_twd") + 1e-9
        )
    )
    if invalid_events.height:
        raise ValueError("combined cap event invariant failed")
    if summary.filter(
        (pl.col("portfolio_cap_compliant") != True).fill_null(True)  # noqa: E712
        | (pl.col("per_product_cap_compliant") != True).fill_null(True)  # noqa: E712
        | (pl.col("production_strategy_go") != False).fill_null(True)  # noqa: E712
        | (pl.col("source_universe_d_safe") != False).fill_null(True)  # noqa: E712
    ).height:
        raise ValueError("combined cap summary safety invariant failed")


def _fact_semantics() -> dict[str, object]:
    return {
        "analysis_only": True,
        "source_universe_role": "retrospective_45_product_development_cohort",
        "source_universe_d_safe": False,
        "universe_selection_contains_target_period_outcomes": True,
        "within_cohort_d_safe_rule_lineage_only": True,
        "production_universe_approved": False,
        "runtime_daily_liquidity_gate_required": True,
        "cost_scope": "completed_one_futures_lot_paired_cycles_only",
        "cost_uses_exact_entry_and_exit_leg_prices": True,
        "normalization_notional_used_only_as_bp_denominator": True,
        "futures_tax_0_2bp_interpreted_per_side": True,
        "futures_commission_20twd_interpreted_per_side": True,
        "same_day_half_spot_sell_tax_requires_completed_same_day": True,
        "minimum_commission_or_currency_rounding_modeled": False,
        "fee_tax_profile_complete_for_completed_cycle": True,
        "full_strategy_cost_profile_complete": False,
        "simultaneous_portfolio_and_value_code_caps": True,
        "notional_basis": "one_way_entry_normalization_notional_twd",
        "not_two_leg_gross_or_futures_margin": True,
        "completed_release_strictly_before_next_entry": True,
        "entry_before_exit_at_equal_recv_time_ns": True,
        "unknown_and_censored_never_release_capacity": True,
        "chronological_admission_is_not_shared_exit_volume_fifo": True,
        "joint_volume_allocated": False,
        "unresolved_cashflow_imputed": False,
        "best_q_selection_go": False,
        "pathwise_ev_ready": False,
        "production_strategy_go": False,
    }


def _config_payload(config: CombinedCapConfig) -> dict[str, object]:
    return {
        "portfolio_caps_twd": [float(value) for value in config.portfolio_caps_twd],
        "per_product_fraction": float(config.per_product_fraction),
        "cost_profile": asdict(config.cost_profile),
        "derived_costs": {
            "spot_commission_bp_per_side": (
                config.cost_profile.spot_commission_bp_per_side
            ),
            "near_flat_price_reference_same_day_variable_cost_bp": (
                config.cost_profile.same_day_variable_cost_bp
            ),
            "near_flat_price_reference_overnight_variable_cost_bp": (
                config.cost_profile.overnight_variable_cost_bp
            ),
            "futures_round_trip_commission_twd": (
                config.cost_profile.futures_round_trip_commission_twd
            ),
        },
    }


def _config_from_payload(payload: object) -> CombinedCapConfig:
    if not isinstance(payload, dict):
        raise ValueError("combined cap config must be an object")
    if set(payload) != {
        "portfolio_caps_twd",
        "per_product_fraction",
        "cost_profile",
        "derived_costs",
    }:
        raise ValueError("combined cap config keys differ")
    cost_payload = payload["cost_profile"]
    if not isinstance(cost_payload, dict):
        raise ValueError("cost profile must be an object")
    config = CombinedCapConfig(
        portfolio_caps_twd=tuple(float(value) for value in payload["portfolio_caps_twd"]),
        per_product_fraction=float(payload["per_product_fraction"]),
        cost_profile=TransactionCostProfile(**cost_payload),
    )
    config.validate()
    if payload["derived_costs"] != _config_payload(config)["derived_costs"]:
        raise ValueError("derived transaction costs differ")
    return config


def _verify_published_files(
    output_root: Path, *, verify_source: bool
) -> tuple[dict[str, object], dict[str, pl.DataFrame]]:
    root = Path(output_root)
    expected = {"complete.json", *ARTIFACTS}
    if not root.is_dir() or {path.name for path in root.iterdir()} != expected:
        raise ValueError("combined cap bundle inventory is not exact")
    marker = _read_json(root / "complete.json")
    declared_marker_sha = marker.get("marker_payload_sha256")
    unhashed = dict(marker)
    unhashed.pop("marker_payload_sha256", None)
    if (
        marker.get("complete") is not True
        or marker.get("schema_version") != BUNDLE_SCHEMA_VERSION
        or marker.get("sweep_version") != SWEEP_VERSION
        or declared_marker_sha != _canonical_sha256(unhashed)
        or marker.get("implementation_sources") != _implementation_sources()
        or marker.get("fact_semantics") != _fact_semantics()
    ):
        raise ValueError("combined cap marker is invalid")
    _config_from_payload(marker.get("config"))
    declarations = marker.get("artifacts")
    if not isinstance(declarations, dict) or set(declarations) != set(ARTIFACTS):
        raise ValueError("combined cap artifact inventory differs")
    frames: dict[str, pl.DataFrame] = {}
    for filename in ARTIFACTS:
        path = root / filename
        frame = pl.read_parquet(path)
        if declarations[filename] != _frame_declaration(path, frame):
            raise ValueError(f"combined cap artifact changed: {filename}")
        frames[filename] = frame
    if verify_source:
        sources = marker.get("sources")
        if not isinstance(sources, dict):
            raise ValueError("combined cap source metadata is invalid")
        source_root = Path(str(sources["source_root"]))
        source = load_verified_source(
            source_root,
            universe_root=Path(str(sources["universe_root"])),
        )
        if dict(source.metadata) != sources:
            raise ValueError("combined cap source changed")
        published_inventory = frames["price_source_inventory.parquet"]
        if (
            published_inventory.schema != source.price_source_inventory.schema
            or not published_inventory.equals(
                source.price_source_inventory, null_equal=True
            )
        ):
            raise ValueError("combined cap price source inventory changed")
    return marker, frames


def _implementation_sources() -> dict[str, str]:
    current = Path(__file__)
    return {current.name: _file_sha256(current)}


def _frame_declaration(path: Path, frame: pl.DataFrame) -> dict[str, object]:
    return {
        "rows": frame.height,
        "columns": frame.width,
        "schema": {name: str(dtype) for name, dtype in frame.schema.items()},
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _sum(frame: pl.DataFrame, column: str) -> float:
    if frame.is_empty():
        return 0.0
    value = frame[column].sum()
    return 0.0 if value is None else float(value)


def _mean(frame: pl.DataFrame, column: str) -> float | None:
    if frame.is_empty():
        return None
    value = frame[column].mean()
    return None if value is None else float(value)


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


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--universe-root", type=Path, default=DEFAULT_UNIVERSE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--verify-only", type=Path)
    parser.add_argument("--no-rebuild", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verify_only is not None:
        marker = verify_combined_cost_cap_bundle(
            args.verify_only, rebuild=not args.no_rebuild
        )
        print(json.dumps(marker, indent=2, sort_keys=True))
        return 0
    if args.no_rebuild:
        raise ValueError("--no-rebuild is valid only with --verify-only")
    result = run_combined_cost_cap_sweep(
        args.output,
        source_root=args.source_root,
        universe_root=args.universe_root,
    )
    print(result.cost_summary)
    print(result.cap_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
