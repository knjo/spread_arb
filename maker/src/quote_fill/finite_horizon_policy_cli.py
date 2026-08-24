"""Atomic publisher for the contextual frozen-policy EV lookup.

This command intentionally publishes analysis artifacts only.  It does not
write into an execution/exit replay root and it does not advertise an intraday
controller when the common-clock transition audit is absent or incomplete.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import polars as pl

from .finite_horizon_policy import (
    ActionRankingConfig,
    ContextualPolicyEVConfig,
    LOOKUP_SCHEMA_VERSION,
    TRANSITION_SCHEMA_VERSION,
    audit_current_fact_adapter_contract,
    audit_intraday_transition_readiness,
    build_contextual_policy_paths,
    build_prequential_contextual_lookup,
    config_payload,
    rank_contextual_policy_actions,
    stable_sha256,
)


PUBLISHER_VERSION = "contextual_frozen_policy_atomic_publisher_v1"


def run_finite_horizon_policy_publish(
    *,
    historical_actions_path: Path,
    terminal_paths_path: Path,
    sessions_path: Path,
    output_root: Path,
    config: ContextualPolicyEVConfig = ContextualPolicyEVConfig(),
    asof_dates: Sequence[str] | None = None,
    decision_actions_path: Path | None = None,
    transition_facts_path: Path | None = None,
    ranking_config: ActionRankingConfig = ActionRankingConfig(),
) -> Path:
    """Build all artifacts in a sibling temp directory, then rename once."""

    config.validate()
    ranking_config.validate()
    destination = Path(output_root).resolve()
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    input_paths = {
        "historical_actions": Path(historical_actions_path).resolve(),
        "terminal_paths": Path(terminal_paths_path).resolve(),
        "sessions": Path(sessions_path).resolve(),
    }
    if decision_actions_path is not None:
        input_paths["decision_actions"] = Path(decision_actions_path).resolve()
    if transition_facts_path is not None:
        input_paths["transition_facts"] = Path(transition_facts_path).resolve()
    for name, path in input_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} input does not exist: {path}")

    sessions = _read_sessions(input_paths["sessions"])
    actions = pl.read_parquet(input_paths["historical_actions"])
    terminals = pl.read_parquet(input_paths["terminal_paths"])
    paths = build_contextual_policy_paths(actions, terminals, config)
    lookup = build_prequential_contextual_lookup(
        paths,
        sessions,
        config,
        asof_dates=asof_dates,
    )
    if lookup.is_empty():
        raise ValueError("prequential lookup is empty; nothing can be published")

    ranking = None
    if decision_actions_path is not None:
        ranking = rank_contextual_policy_actions(
            pl.read_parquet(input_paths["decision_actions"]),
            lookup,
            config,
            ranking_config,
        )
    transition_facts = (
        None
        if transition_facts_path is None
        else pl.read_parquet(input_paths["transition_facts"])
    )
    readiness = audit_intraday_transition_readiness(transition_facts)
    adapter_readiness = audit_current_fact_adapter_contract(
        action_columns=actions.columns,
        terminal_path_columns=terminals.columns,
        transition_columns=(
            () if transition_facts is None else transition_facts.columns
        ),
    )

    temp = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        frames: dict[str, pl.DataFrame] = {
            "contextual_policy_paths.parquet": paths,
            "prequential_ev_lookup.parquet": lookup,
            "intraday_transition_readiness.parquet": readiness,
            "fact_adapter_readiness.parquet": adapter_readiness,
        }
        if ranking is not None:
            frames["scored_contextual_actions.parquet"] = ranking.scored_actions
            frames["contextual_decisions.parquet"] = ranking.decisions
        artifacts: dict[str, dict[str, object]] = {}
        for filename, frame in frames.items():
            target = temp / filename
            frame.write_parquet(target, compression="zstd", statistics=True)
            artifacts[filename] = {
                "sha256": _file_sha256(target),
                "bytes": target.stat().st_size,
                "rows": frame.height,
                "columns": len(frame.columns),
                "column_names": frame.columns,
            }

        source_path = Path(__file__).resolve()
        implementation_path = source_path.with_name("finite_horizon_policy.py")
        payload: dict[str, object] = {
            "publisher_version": PUBLISHER_VERSION,
            "lookup_schema_version": LOOKUP_SCHEMA_VERSION,
            "transition_schema_version": TRANSITION_SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": config_payload(config),
            "ranking_config": {
                "minimum_lcb_bp": ranking_config.minimum_lcb_bp,
                "require_position_establishment_phase": (
                    ranking_config.require_position_establishment_phase
                ),
            },
            "asof_dates": list(asof_dates) if asof_dates is not None else None,
            "sessions": {
                "count": len(sessions),
                "first": sessions[0],
                "last": sessions[-1],
                "sha256": stable_sha256({"sessions": sessions}),
            },
            "inputs": {
                name: {
                    "path": str(path),
                    "sha256": _file_sha256(path),
                    "bytes": path.stat().st_size,
                }
                for name, path in sorted(input_paths.items())
            },
            "implementation": {
                source_path.name: _file_sha256(source_path),
                implementation_path.name: _file_sha256(implementation_path),
            },
            "artifacts": artifacts,
            "fact_semantics": {
                "fixed_policy_choice_at_position_establishment_only": True,
                "pending_origins_retained_in_asof_denominator": True,
                "strict_date_less_than_asof": True,
                "strict_label_end_less_than_asof": True,
                "actual_four_leg_gross_uses_observed_prices": True,
                "component_cost_profile_versioned_and_hashed": True,
                "known_accrued_cost_retained_while_terminal_remainder_unknown": True,
                "finite_horizon_and_d_minus_h_embargo": True,
                "physical_origin_support_not_tick_path_support": True,
                "kappa_hierarchical_shrinkage": True,
                "complete_candidate_set_hash_and_count_required": True,
                "unresolved_terminal_mass_given_finite_bound": False,
                "intraday_terminal_value_duplicated_per_observation": False,
                "intraday_controller_published": False,
            },
        }
        payload["manifest_sha256"] = stable_sha256(payload)
        _write_json(temp / "manifest.json", payload)
        marker = {
            "status": "complete",
            "publisher_version": PUBLISHER_VERSION,
            "manifest_sha256": _file_sha256(temp / "manifest.json"),
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
        }
        _write_json(temp / "complete.json", marker)
        os.replace(temp, destination)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return destination


def _read_sessions(path: Path) -> list[str]:
    sessions = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not sessions:
        raise ValueError("sessions file is empty")
    return sessions


def _load_config(path: Path | None) -> ContextualPolicyEVConfig:
    if path is None:
        return ContextualPolicyEVConfig()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("config JSON must contain one object")
    allowed = {item.name for item in fields(ContextualPolicyEVConfig)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    for name in (
        "remaining_minute_edges",
        "holding_age_minute_edges",
        "expiry_session_edges",
    ):
        if name in raw:
            raw[name] = tuple(raw[name])
    return ContextualPolicyEVConfig(**raw)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish a causal fixed-policy contextual EV lookup atomically"
    )
    parser.add_argument("--historical-actions", type=Path, required=True)
    parser.add_argument("--terminal-paths", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--decision-actions", type=Path)
    parser.add_argument("--transition-facts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config-json", type=Path)
    parser.add_argument("--asof-date", action="append", dest="asof_dates")
    parser.add_argument("--minimum-lcb-bp", type=float, default=0.0)
    parser.add_argument(
        "--allow-non-establishment-ranking",
        action="store_true",
        help=(
            "Only relaxes the phase validator; it does not make intraday "
            "switching identified"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run_finite_horizon_policy_publish(
        historical_actions_path=args.historical_actions,
        terminal_paths_path=args.terminal_paths,
        sessions_path=args.sessions,
        output_root=args.output,
        config=_load_config(args.config_json),
        asof_dates=args.asof_dates,
        decision_actions_path=args.decision_actions,
        transition_facts_path=args.transition_facts,
        ranking_config=ActionRankingConfig(
            minimum_lcb_bp=args.minimum_lcb_bp,
            require_position_establishment_phase=(
                not args.allow_non_establishment_ranking
            ),
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
