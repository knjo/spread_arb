"""Practical runner for the imputed-full-carry continuation scenario.

The immutable formal cross-session result stops a path when an exit-maker
fill quantity is unknown.  This supplemental runner implements the explicitly
requested counterfactual that such a day had zero exit fills and the complete
paired position survived.  It resumes the frozen exit policy on following
sessions, batching all active paths for the same ``Date/ValueCode/QuoteCode``
into one product-day replay.

Candidate tapes are read from the source-bound cross-session cache.  The
runner does not rescan the multi-hundred-megabyte raw day once per position.
At exact-contract expiry, any still-active full-pair scenario is marked with
the final valid spot bid and futures ask in the cached study tape.  Those
prices are named last-valid session marks, never official closes or settlement.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Callable, Iterable, Mapping, Sequence

import polars as pl

from .combined_cost_cap_sweep import (
    DEFAULT_SOURCE_ROOT,
    DEFAULT_UNIVERSE_ROOT,
    _file_sha256,
    _partition_manifest,
    _read_json,
    _verify_published_files,
)
from .exit_maker_cross_session import (
    CrossSessionExitMakerSession,
    _classify_session_policy,
    _prepare_candidate_tape,
    _session_start_time,
    _spool_day_policy_facts,
)
from .exit_maker_cross_session_runner import (
    CandidateSessionRequirement,
    _load_candidate_cache_partition,
)
from .exit_maker_study import ExitMakerStudyConfig
from .raw_tape import RawTapeDay
from .supplemental_carry_terminal import (
    SupplementalCarryConfig,
    apply_supplemental_carry_terminal_overlay,
    build_last_observed_session_liquidation_mark,
)


RUNNER_VERSION = "supplemental_imputed_full_carry_grouped_runner_v2_trade_fallback"
EXACT_REPLAY_MODE = "exact_raw_state_v1"
ONE_SECOND_REPLAY_MODE = "one_second_last_state_plus_spread_epoch_approx_v1"
DEFAULT_REPLAY_MODE = ONE_SECOND_REPLAY_MODE
DEFAULT_PREREQUISITE_ROOT = Path(
    "maker/data/walkforward/cross_session_prerequisites_v1_20260819"
)
DEFAULT_CANDIDATE_CACHE_ROOT = Path(
    "maker/data/walkforward/"
    "exit_maker_cross_session_narrow_60d_candidate_session_cache_v8"
)
DEFAULT_OUTPUT_ROOT = Path(
    "maker/data/walkforward/"
    "prequential_challenger_ab12_supplemental_carry_20260821_v2_trade_fallback"
)
DEFAULT_EXACT_PATH_SOURCE_ROOT = Path(
    "maker/data/walkforward/"
    "prequential_challenger_ab12_cost_caps_20260821_v1"
)

_ACTION_FILENAME = "execution_action_facts.parquet"
_EXIT_FILENAME = "exit_facts.parquet"
_ELIGIBLE = {
    "maker_fill_state_unknown",
    "right_censored_expiry_settlement_unpriced",
}


@dataclass(frozen=True)
class SupplementalReplayInputs:
    paths: pl.DataFrame
    entry_prices: pl.DataFrame
    actions: pl.DataFrame
    exits: pl.DataFrame
    source_inventory: pl.DataFrame
    source_metadata: Mapping[str, object]


@dataclass(frozen=True)
class SupplementalReplayResult:
    continuation_terminals: pl.DataFrame
    expiry_marks: pl.DataFrame
    continuation_audit: pl.DataFrame
    candidate_session_sampling_audit: pl.DataFrame
    pair_sessions_replayed: int
    truncated: bool
    replay_mode: str
    state_sampling_approximate: bool


SessionLoader = Callable[
    [str, str, str], tuple[CrossSessionExitMakerSession, str]
]
PolicyDayReplayer = Callable[
    [
        pl.DataFrame,
        pl.DataFrame,
        CrossSessionExitMakerSession,
        Sequence[str],
        Path,
    ],
    pl.DataFrame,
]


def load_supplemental_replay_inputs(
    source_root: Path = DEFAULT_SOURCE_ROOT,
    *,
    universe_root: Path = DEFAULT_UNIVERSE_ROOT,
    exact_path_source_root: Path = DEFAULT_EXACT_PATH_SOURCE_ROOT,
) -> SupplementalReplayInputs:
    """Verify the challenger roots and extract unresolved exact entry facts."""

    paths, source_metadata = _load_bound_exact_path_source(
        Path(source_root),
        universe_root=Path(universe_root),
        exact_path_source_root=Path(exact_path_source_root),
    )
    unresolved = paths.filter(pl.col("outcome_status").is_in(sorted(_ELIGIBLE)))
    if unresolved.is_empty():
        raise ValueError("source challenger has no supplemental carry candidates")
    execution_manifest_path = Path(
        str(source_metadata["execution_root"])
    ) / "execution_partition_manifest.parquet"
    actions, exits, entry_prices, source_inventory = (
        _extract_unresolved_execution_facts(
            unresolved,
            execution_manifest_path=execution_manifest_path,
        )
    )
    return SupplementalReplayInputs(
        paths=paths,
        entry_prices=entry_prices,
        actions=actions,
        exits=exits,
        source_inventory=source_inventory,
        source_metadata=source_metadata,
    )


def _extract_unresolved_execution_facts(
    unresolved: pl.DataFrame,
    *,
    execution_manifest_path: Path,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Read only replay-required columns for the unresolved entry IDs."""

    required = {
        "Date",
        "ValueCode",
        "entry_policy_generation_id",
        "normalization_notional_twd",
    }
    missing = required - set(unresolved.columns)
    if missing:
        raise ValueError(f"unresolved paths missing columns: {sorted(missing)}")
    if (
        unresolved.is_empty()
        or unresolved["entry_policy_generation_id"].n_unique()
        != unresolved.height
    ):
        raise ValueError("unresolved entry IDs must be nonempty and one-to-one")
    manifest = _partition_manifest(execution_manifest_path)
    action_frames: list[pl.DataFrame] = []
    exit_frames: list[pl.DataFrame] = []
    price_rows: list[dict[str, object]] = []
    inventory_rows: list[dict[str, object]] = []
    for group in unresolved.partition_by(["Date", "ValueCode"], maintain_order=True):
        date = str(group.item(0, "Date"))
        value_code = str(group.item(0, "ValueCode"))
        entry_ids = {
            str(value)
            for value in group["entry_policy_generation_id"].to_list()
        }
        key = (date, value_code)
        if key not in manifest:
            raise ValueError(f"execution partition missing for {key}")
        partition = manifest[key]
        actions, action_inventory = _read_verified_partition_artifact(
            partition,
            key,
            _ACTION_FILENAME,
            columns=(
                "Date",
                "ValueCode",
                "QuoteCode",
                "route",
                "raw_order_fact_id",
                "policy_generation_id",
                "full_fill",
                "entry_hedge_status",
                "entry_hedge_decision_time_ns",
                "entry_hedge_label_observed",
                "entry_hedge_executable",
                "entry_future_price",
                "entry_spot_price",
                "entry_hedge_contract_size_shares",
            ),
            filter_column="policy_generation_id",
            identifiers=entry_ids,
        )
        selected_actions = actions
        if (
            selected_actions.height != len(entry_ids)
            or selected_actions["policy_generation_id"].n_unique()
            != selected_actions.height
        ):
            raise ValueError(f"unresolved entry action join is not one-to-one: {key}")
        exits, exit_inventory = _read_verified_partition_artifact(
            partition,
            key,
            _EXIT_FILENAME,
            columns=(
                "Date",
                "ValueCode",
                "QuoteCode",
                "route",
                "raw_order_fact_id",
                "policy_generation_id",
                "exit_rule_id",
                "exit_threshold_basis_bp",
                "exit_rule_source_asof_date",
            ),
            filter_column="policy_generation_id",
            identifiers=entry_ids,
        )
        selected_exits = exits
        expected_rules = len(entry_ids) * 2
        if (
            selected_exits.height != expected_rules
            or selected_exits.select(
                "policy_generation_id", "exit_rule_id"
            ).n_unique()
            != expected_rules
        ):
            raise ValueError(f"unresolved frozen exit rules are incomplete: {key}")
        action_frames.append(selected_actions)
        exit_frames.append(selected_exits)
        inventory_rows.extend((action_inventory, exit_inventory))
        notional_by_id = {
            str(row["entry_policy_generation_id"]): float(
                row["normalization_notional_twd"]
            )
            for row in group.iter_rows(named=True)
        }
        for row in selected_actions.iter_rows(named=True):
            entry_id = str(row["policy_generation_id"])
            spot = _positive_float(row.get("entry_spot_price"), "entry_spot_price")
            future = _positive_float(
                row.get("entry_future_price"), "entry_future_price"
            )
            shares = _positive_int(
                row.get("entry_hedge_contract_size_shares"),
                "entry_hedge_contract_size_shares",
            )
            if row.get("full_fill") is not True or row.get(
                "entry_hedge_executable"
            ) is not True:
                raise ValueError("supplemental entry was not a hedged full fill")
            if not math.isclose(
                shares * spot,
                notional_by_id[entry_id],
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError("unresolved entry prices do not reconcile to notional")
            price_rows.append(
                {
                    "entry_policy_generation_id": entry_id,
                    "entry_spot_price": spot,
                    "entry_future_price": future,
                    "entry_contract_size_shares": shares,
                    "entry_price_source": "execution_action_facts.parquet",
                    "source_identity_sha256": action_inventory["artifact_sha256"],
                }
            )
    entry_prices = pl.from_dicts(price_rows, infer_schema_length=None).sort(
        "entry_policy_generation_id"
    )
    if (
        entry_prices.height != unresolved.height
        or entry_prices["entry_policy_generation_id"].n_unique()
        != entry_prices.height
    ):
        raise ValueError("supplemental entry-price population is not one-to-one")
    return (
        pl.concat(action_frames, how="diagonal_relaxed"),
        pl.concat(exit_frames, how="diagonal_relaxed"),
        entry_prices,
        pl.from_dicts(inventory_rows, infer_schema_length=None),
    )


def _load_bound_exact_path_source(
    source_root: Path,
    *,
    universe_root: Path,
    exact_path_source_root: Path,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Load the small pre-enriched path artifact without rebuilding it.

    The formal cost/cap bundle already contains the exact six price columns
    for every formally completed path.  Reusing that immutable, self-hashed
    artifact avoids re-reading the very large same-day/cross-session source
    facts solely to reconstruct 2,411 already-published terminals.
    """

    cost_marker, cost_frames = _verify_published_files(
        exact_path_source_root, verify_source=False
    )
    sources = cost_marker.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("exact path source metadata is missing")
    source = Path(source_root).resolve()
    universe = Path(universe_root).resolve()
    if Path(str(sources.get("source_root"))).resolve() != source:
        raise ValueError("exact path artifact binds a different challenger root")
    if Path(str(sources.get("universe_root"))).resolve() != universe:
        raise ValueError("exact path artifact binds a different universe root")
    source_marker_path = source / "complete.json"
    selected_path = source / "selected_policy_paths.parquet"
    execution_root = Path(str(sources.get("execution_root"))).resolve()
    execution_manifest_path = execution_root / "execution_partition_manifest.parquet"
    universe_marker_path = universe / "complete.json"
    bindings = (
        (source_marker_path, "source_complete_sha256"),
        (selected_path, "source_selected_paths_sha256"),
        (execution_manifest_path, "execution_manifest_sha256"),
        (universe_marker_path, "universe_complete_sha256"),
    )
    for path, key in bindings:
        if not path.is_file() or _file_sha256(path) != sources.get(key):
            raise ValueError(f"exact path source binding changed: {path}")
    source_marker = _read_json(source_marker_path)
    declaration = (
        source_marker.get("artifacts", {}).get("selected_policy_paths.parquet")
        if isinstance(source_marker.get("artifacts"), dict)
        else None
    )
    if not isinstance(declaration, dict):
        raise ValueError("challenger selected-path declaration is missing")
    raw_paths = pl.read_parquet(selected_path)
    if (
        raw_paths.height != int(declaration.get("rows", -1))
        or raw_paths.width != int(declaration.get("columns", -1))
        or raw_paths["policy_path_id"].n_unique() != raw_paths.height
    ):
        raise ValueError("challenger selected-path dimensions changed")
    cost_paths = cost_frames["path_transaction_costs.parquet"]
    if (
        cost_paths.height != raw_paths.height
        or cost_paths["policy_path_id"].n_unique() != cost_paths.height
    ):
        raise ValueError("exact path source population differs from challenger")
    identity_columns = (
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
    )
    order = ["policy_path_id"]
    raw_identity = raw_paths.select(identity_columns).sort(order)
    cost_identity = cost_paths.select(identity_columns).sort(order)
    if raw_identity.schema != cost_identity.schema or not raw_identity.equals(
        cost_identity, null_equal=True
    ):
        raise ValueError("exact path source identities differ from challenger")
    exact_columns = (
        "entry_spot_price",
        "entry_future_price",
        "entry_contract_size_shares",
        "exit_spot_price",
        "exit_future_price",
        "exact_price_source",
    )
    paths = raw_paths.join(
        cost_paths.select("policy_path_id", *exact_columns),
        on="policy_path_id",
        how="left",
        validate="1:1",
    )
    return paths, {
        **dict(sources),
        "exact_path_source_root": str(Path(exact_path_source_root).resolve()),
        "exact_path_source_complete_sha256": _file_sha256(
            Path(exact_path_source_root) / "complete.json"
        ),
        "exact_path_source_artifact_sha256": str(
            cost_marker["artifacts"]["path_transaction_costs.parquet"]["sha256"]
        ),
    }


def replay_supplemental_continuations(
    paths: pl.DataFrame,
    actions: pl.DataFrame,
    exits: pl.DataFrame,
    sessions: Sequence[str] | Iterable[str],
    contract_calendar: pl.DataFrame,
    *,
    session_loader: SessionLoader,
    policy_day_replayer: PolicyDayReplayer | None = None,
    spool_root: Path,
    max_pair_sessions: int | None = None,
    replay_mode: str = EXACT_REPLAY_MODE,
) -> SupplementalReplayResult:
    """Replay all eligible positions in grouped candidate product-days."""

    _validate_replay_mode(replay_mode)
    session_values = _normalise_sessions(sessions)
    session_index = {date: index for index, date in enumerate(session_values)}
    calendar = _calendar_lookup(contract_calendar)
    candidates = paths.filter(pl.col("outcome_status").is_in(sorted(_ELIGIBLE)))
    if candidates.is_empty():
        raise ValueError("no eligible supplemental paths")
    if candidates["policy_path_id"].n_unique() != candidates.height:
        raise ValueError("supplemental paths are not unique")
    action_ids = set(actions["policy_generation_id"].cast(pl.String).to_list())
    exit_ids = set(exits["policy_generation_id"].cast(pl.String).to_list())
    states: dict[str, dict[str, object]] = {}
    audit: dict[str, dict[str, object]] = {}
    for row in candidates.iter_rows(named=True):
        path_id = str(row["policy_path_id"])
        entry_id = str(row["entry_policy_generation_id"])
        if entry_id not in action_ids or entry_id not in exit_ids:
            raise ValueError("supplemental path is missing replay source facts")
        quote = str(row["QuoteCode"])
        if quote not in calendar:
            raise ValueError(f"contract calendar missing {quote}")
        expiry, calendar_version = calendar[quote]
        source_status = str(row["outcome_status"])
        last_observed = str(row["last_observed_session_date"])
        if last_observed not in session_index or expiry not in session_index:
            start_index = len(session_values)
            blocker = "calendar_or_observation_horizon_missing"
        elif source_status == "right_censored_expiry_settlement_unpriced":
            start_index = session_index[expiry]
            blocker = None
        elif last_observed == expiry:
            # The source day was already replayed and ended in an unknown fill
            # state.  Do not replay the exit policy twice on expiry; use only
            # the authorised last-valid-session mark terminal.
            start_index = session_index[expiry]
            blocker = None
        else:
            start_index = session_index[last_observed] + 1
            blocker = None
        state = {
            **row,
            "expiry_session": expiry,
            "calendar_version": calendar_version,
            "start_index": start_index,
            "force_expiry_only": (
                source_status == "right_censored_expiry_settlement_unpriced"
                or last_observed == expiry
            ),
            "active": blocker is None,
            "sessions_replayed": 0,
            "imputed_unknown_sessions": (
                1 if source_status == "maker_fill_state_unknown" else 0
            ),
        }
        states[path_id] = state
        audit[path_id] = _audit_base(state, blocker, replay_mode=replay_mode)

    replay_day = policy_day_replayer or _default_policy_day_replayer
    continuation_rows: list[dict[str, object]] = []
    mark_rows: dict[tuple[str, str], dict[str, object]] = {}
    pair_sessions = 0
    truncated = False
    for date_index, session_date in enumerate(session_values):
        due = [
            state
            for state in states.values()
            if state["active"] is True
            and int(state["start_index"]) <= date_index
            and session_date <= str(state["expiry_session"])
        ]
        groups: dict[tuple[str, str], list[dict[str, object]]] = {}
        for state in due:
            groups.setdefault(
                (str(state["ValueCode"]), str(state["QuoteCode"])), []
            ).append(state)
        for (value_code, quote_code), group in sorted(groups.items()):
            if max_pair_sessions is not None and pair_sessions >= max_pair_sessions:
                truncated = True
                break
            pair_sessions += 1
            try:
                session, cache_marker_sha = session_loader(
                    session_date, value_code, quote_code
                )
                prepared, preparation_status = _prepare_candidate_tape(
                    session,
                    value_code=value_code,
                    exact_quote_code=quote_code,
                )
            except (FileNotFoundError, ValueError) as error:
                for state in group:
                    _block(state, audit, "candidate_session_load_failed", str(error))
                continue
            force_only = [state for state in group if state["force_expiry_only"]]
            replay_group = [state for state in group if not state["force_expiry_only"]]
            if prepared is None:
                if preparation_status != "carry_no_exact_contract_events":
                    for state in replay_group:
                        _block(
                            state,
                            audit,
                            preparation_status or "candidate_session_invalid",
                            preparation_status,
                        )
                if session_date == str(group[0]["expiry_session"]):
                    for state in group:
                        if state["active"]:
                            _block(
                                state,
                                audit,
                                "expiry_mark_unavailable",
                                "exact contract has no valid expiry-session tape",
                            )
                del session
                gc.collect()
                continue
            if replay_group:
                entry_ids = sorted(
                    {str(state["entry_policy_generation_id"]) for state in replay_group}
                )
                active_trials = sorted(
                    {str(state["exit_policy_trial_id"]) for state in replay_group}
                )
                start_ns = _session_start_time(session, prepared)
                session_actions = actions.filter(
                    pl.col("policy_generation_id").cast(pl.String).is_in(entry_ids)
                ).with_columns(
                    pl.lit(session_date).alias("Date"),
                    pl.lit(start_ns).cast(pl.Int64).alias(
                        "entry_hedge_decision_time_ns"
                    ),
                )
                session_exits = exits.filter(
                    pl.col("policy_generation_id").cast(pl.String).is_in(entry_ids)
                ).with_columns(pl.lit(session_date).alias("Date"))
                try:
                    policy_rows = replay_day(
                        session_actions,
                        session_exits,
                        CrossSessionExitMakerSession(
                            **{**session.__dict__, "raw_tape": prepared}
                        ),
                        active_trials,
                        Path(spool_root),
                    )
                except (FileNotFoundError, ValueError) as error:
                    for state in replay_group:
                        _block(state, audit, "policy_day_replay_failed", str(error))
                else:
                    by_trial = {
                        str(row["exit_policy_trial_id"]): row
                        for row in policy_rows.iter_rows(named=True)
                    }
                    if set(by_trial) != set(active_trials):
                        raise ValueError("policy-day replay did not return every active trial")
                    for state in replay_group:
                        path_id = str(state["policy_path_id"])
                        trial = str(state["exit_policy_trial_id"])
                        row = by_trial[trial]
                        transition, status, detail = _classify_session_policy(
                            row,
                            cancel_semantics="nominal_instant_cancel_v0",
                        )
                        state["sessions_replayed"] = int(
                            state["sessions_replayed"]
                        ) + 1
                        audit[path_id]["sessions_replayed"] = state[
                            "sessions_replayed"
                        ]
                        if transition == "terminal":
                            continuation_rows.append(
                                _continuation_record(
                                    state,
                                    row,
                                    session_date=session_date,
                                    cache_marker_sha=cache_marker_sha,
                                    replay_mode=replay_mode,
                                )
                            )
                            state["active"] = False
                            audit[path_id].update(
                                {
                                    "terminal_resolution": "normal_continuation_replay",
                                    "terminal_date": session_date,
                                    "blocker_status": None,
                                    "blocker_detail": None,
                                }
                            )
                        elif status == "maker_fill_state_unknown":
                            state["imputed_unknown_sessions"] = int(
                                state["imputed_unknown_sessions"]
                            ) + 1
                            audit[path_id]["imputed_unknown_sessions"] = state[
                                "imputed_unknown_sessions"
                            ]
                        elif transition == "carry":
                            pass
                        else:
                            _block(
                                state,
                                audit,
                                status,
                                detail or "non-carry session outcome",
                            )
            expiry_date = str(group[0]["expiry_session"])
            if session_date == expiry_date:
                still_active = [state for state in group if state["active"]]
                if still_active:
                    try:
                        mark = build_last_observed_session_liquidation_mark(
                            prepared,
                            value_code=value_code,
                            quote_code=quote_code,
                            expiry_session=expiry_date,
                            calendar_version=str(group[0]["calendar_version"]),
                            source_identity_sha256=cache_marker_sha,
                        ).row(0, named=True)
                    except ValueError as error:
                        for state in still_active:
                            _block(
                                state,
                                audit,
                                "expiry_mark_unavailable",
                                str(error),
                            )
                    else:
                        mark_rows[(value_code, quote_code)] = mark
                        resolution = (
                            "expiry_last_observed_session_liquidation_mark"
                            if mark.get("mark_uses_trade_fallback") is True
                            else "expiry_last_valid_session_mark"
                        )
                        for state in still_active:
                            path_id = str(state["policy_path_id"])
                            state["active"] = False
                            audit[path_id].update(
                                {
                                    "terminal_resolution": (
                                        resolution
                                    ),
                                    "terminal_date": expiry_date,
                                    "blocker_status": None,
                                    "blocker_detail": None,
                                }
                            )
            del session, prepared
            gc.collect()
        if truncated:
            break
    for state in states.values():
        if state["active"]:
            _block(
                state,
                audit,
                "benchmark_truncated" if truncated else "observation_horizon_end",
                "supplemental replay ended before a terminal was observed",
            )
    return SupplementalReplayResult(
        continuation_terminals=_frame_or_empty(
            continuation_rows, _continuation_schema()
        ),
        expiry_marks=_frame_or_empty(list(mark_rows.values()), _mark_schema()),
        continuation_audit=pl.from_dicts(
            [audit[key] for key in sorted(audit)], infer_schema_length=None
        ),
        candidate_session_sampling_audit=pl.DataFrame(
            schema=_sampling_audit_schema()
        ),
        pair_sessions_replayed=pair_sessions,
        truncated=truncated,
        replay_mode=replay_mode,
        state_sampling_approximate=replay_mode == ONE_SECOND_REPLAY_MODE,
    )


def load_cached_candidate_session(
    cache_root: Path,
    date: str,
    value_code: str,
    quote_code: str,
) -> tuple[CrossSessionExitMakerSession, str]:
    """Verify and load one exact candidate cache partition."""

    parent = (
        Path(cache_root)
        / f"Date={date}"
        / f"ValueCode={value_code}"
        / f"QuoteCode={quote_code}"
    )
    partitions = sorted(parent.glob("Key=*")) if parent.is_dir() else []
    if len(partitions) != 1:
        raise FileNotFoundError(
            f"expected exactly one candidate cache key at {parent}; found {len(partitions)}"
        )
    partition = partitions[0]
    marker_path = partition / "complete.json"
    marker = _read_json(marker_path)
    config = marker.get("config")
    fingerprint = config.get("source_fingerprint") if isinstance(config, dict) else None
    if not isinstance(fingerprint, dict):
        raise ValueError("candidate cache source fingerprint is missing")
    _validate_source_fingerprint(fingerprint)
    requirement = CandidateSessionRequirement(
        str(date), str(value_code), str(quote_code), fingerprint
    )
    session = _load_candidate_cache_partition(requirement, partition)
    return session, _file_sha256(marker_path)


def thin_candidate_session_one_second(
    session: CrossSessionExitMakerSession,
    *,
    value_code: str,
    quote_code: str,
) -> tuple[CrossSessionExitMakerSession, dict[str, object]]:
    """Apply the disclosed one-second/change-point state approximation.

    Spot keeps the final state in each integer receive-time second *plus*
    every SpreadPair epoch transition.  Futures keeps the final state in each
    second.  All raw trade prints remain untouched, so maker quantity replay
    is still indexed against the complete supplied trade tape; target/cancel
    clocks, initial queue, and delayed hedge book snapshots are approximate.
    """

    session.validate()
    raw = session.raw_tape
    clock = session.spread_pair_clock.filter(
        (pl.col("Date").cast(pl.String) == str(session.date))
        & (pl.col("ValueCode").cast(pl.String) == str(value_code))
    )
    spot = raw.spot_states.filter(
        (pl.col("Date").cast(pl.String) == str(session.date))
        & (pl.col("ValueCode").cast(pl.String) == str(value_code))
        & (pl.col("QuoteCode").cast(pl.String) == str(quote_code))
    )
    future = raw.future_states.filter(
        (pl.col("Date").cast(pl.String) == str(session.date))
        & (pl.col("ValueCode").cast(pl.String) == str(value_code))
        & (pl.col("QuoteCode").cast(pl.String) == str(quote_code))
    )
    required_clock = {"spot_channel_seq", "spread_pair_epoch"}
    if required_clock - set(clock.columns):
        raise ValueError("one-second approximation lacks SpreadPair clock columns")
    if spot.is_empty() or future.is_empty() or clock.is_empty():
        raise ValueError("one-second approximation received an empty exact pair")
    if clock["spot_channel_seq"].n_unique() != clock.height:
        raise ValueError("one-second approximation clock keys are duplicated")
    ordered_spot = spot.sort(["recv_time_ns", "sequence", "packet_sequence"])
    joined = (
        ordered_spot.with_row_index("_source_row")
        .join(
            clock.select(
                pl.col("spot_channel_seq").alias("sequence"),
                "spread_pair_epoch",
            ),
            on="sequence",
            how="left",
            validate="1:1",
        )
        .with_columns(
            (pl.col("recv_time_ns") // 1_000_000_000).alias("_sample_second"),
            (
                pl.col("spread_pair_epoch")
                != pl.col("spread_pair_epoch").shift(1)
            )
            .fill_null(True)
            .alias("_spread_epoch_change"),
        )
        .with_columns(
            (
                pl.col("_source_row")
                == pl.col("_source_row").max().over("_sample_second")
            ).alias("_second_last")
        )
    )
    if joined["spread_pair_epoch"].null_count():
        raise ValueError("one-second approximation clock does not cover spot states")
    selected_spot = joined.filter(
        pl.col("_spread_epoch_change") | pl.col("_second_last")
    ).select(ordered_spot.columns)
    selected_spot = _retain_last_valid_mark_state(
        selected_spot, ordered_spot, price_column="exec_bid_price"
    )
    selected_sequences = selected_spot["sequence"]
    selected_clock = clock.join(
        pl.DataFrame({"spot_channel_seq": selected_sequences}),
        on="spot_channel_seq",
        how="semi",
    )
    ordered_future = future.sort(
        ["recv_time_ns", "sequence", "packet_sequence"]
    )
    selected_future = _retain_last_valid_mark_state(
        _last_state_per_second(ordered_future),
        ordered_future,
        price_column="exec_ask_price",
    )
    if selected_clock.height != selected_spot.height:
        raise ValueError("one-second approximation lost selected spot clock rows")
    thinned_raw = RawTapeDay(
        date=raw.date,
        mapping=raw.mapping,
        spot_states=selected_spot,
        future_states=selected_future,
        spot_trades=raw.spot_trades,
        future_trades=raw.future_trades,
        audit=raw.audit,
    )
    thinned = replace(
        session,
        raw_tape=thinned_raw,
        spread_pair_clock=selected_clock,
    )
    thinned.validate()
    return thinned, _sampling_audit_record(
        session,
        thinned,
        value_code=value_code,
        quote_code=quote_code,
        replay_mode=ONE_SECOND_REPLAY_MODE,
    )


def _last_state_per_second(frame: pl.DataFrame) -> pl.DataFrame:
    ordered = frame.sort(["recv_time_ns", "sequence", "packet_sequence"])
    return (
        ordered.with_columns(
            (pl.col("recv_time_ns") // 1_000_000_000).alias("_sample_second")
        )
        .group_by("_sample_second", maintain_order=True)
        .tail(1)
        .drop("_sample_second")
        .sort(["recv_time_ns", "sequence", "packet_sequence"])
    )


def _retain_last_valid_mark_state(
    sampled: pl.DataFrame,
    source: pl.DataFrame,
    *,
    price_column: str,
) -> pl.DataFrame:
    last_valid = source.filter(
        pl.col("raw_has_book").fill_null(False)
        & pl.col("book_state_available").fill_null(False)
        & pl.col(price_column).is_not_null()
        & pl.col(price_column).is_finite()
        & (pl.col(price_column) > 0)
    ).tail(1)
    if last_valid.is_empty():
        return sampled
    return (
        pl.concat((sampled, last_valid), how="vertical")
        .unique(
            subset=["recv_time_ns", "sequence", "packet_sequence"],
            keep="first",
            maintain_order=True,
        )
        .sort(["recv_time_ns", "sequence", "packet_sequence"])
    )


def _sampling_audit_record(
    source: CrossSessionExitMakerSession,
    selected: CrossSessionExitMakerSession,
    *,
    value_code: str,
    quote_code: str,
    replay_mode: str,
) -> dict[str, object]:
    source_raw = source.raw_tape
    selected_raw = selected.raw_tape
    approximate = replay_mode == ONE_SECOND_REPLAY_MODE
    return {
        "Date": str(source.date),
        "ValueCode": str(value_code),
        "QuoteCode": str(quote_code),
        "replay_mode": replay_mode,
        "state_sampling_approximate": approximate,
        "sample_interval_seconds": 1 if approximate else None,
        "spot_spread_epoch_changes_always_retained": approximate,
        "raw_trade_events_unchanged": True,
        "spot_states_source_rows": source_raw.spot_states.height,
        "spot_states_replay_rows": selected_raw.spot_states.height,
        "future_states_source_rows": source_raw.future_states.height,
        "future_states_replay_rows": selected_raw.future_states.height,
        "spot_trades_rows": source_raw.spot_trades.height,
        "future_trades_rows": source_raw.future_trades.height,
        "target_cancel_clock_exact": not approximate,
        "initial_queue_snapshot_exact": not approximate,
        "delayed_hedge_snapshot_exact": not approximate,
    }


def run_supplemental_carry_analysis(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    universe_root: Path = DEFAULT_UNIVERSE_ROOT,
    exact_path_source_root: Path = DEFAULT_EXACT_PATH_SOURCE_ROOT,
    prerequisite_root: Path = DEFAULT_PREREQUISITE_ROOT,
    candidate_cache_root: Path = DEFAULT_CANDIDATE_CACHE_ROOT,
    max_pair_sessions: int | None = None,
    replay_mode: str = DEFAULT_REPLAY_MODE,
    publish: bool = True,
) -> tuple[SupplementalReplayResult, pl.DataFrame]:
    _validate_replay_mode(replay_mode)
    inputs = load_supplemental_replay_inputs(
        source_root,
        universe_root=universe_root,
        exact_path_source_root=exact_path_source_root,
    )
    prerequisite = Path(prerequisite_root)
    sessions_path = prerequisite / "candidate_sessions.txt"
    calendar_path = prerequisite / "exact_contract_calendar_v1.parquet"
    sessions = sessions_path.read_text(encoding="utf-8").splitlines()
    calendar = pl.read_parquet(calendar_path)
    destination = Path(output_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sampling_rows: list[dict[str, object]] = []

    def session_loader(
        date: str, value_code: str, quote_code: str
    ) -> tuple[CrossSessionExitMakerSession, str]:
        session, digest = load_cached_candidate_session(
            candidate_cache_root, date, value_code, quote_code
        )
        if replay_mode == ONE_SECOND_REPLAY_MODE:
            session, row = thin_candidate_session_one_second(
                session,
                value_code=value_code,
                quote_code=quote_code,
            )
        else:
            row = _sampling_audit_record(
                session,
                session,
                value_code=value_code,
                quote_code=quote_code,
                replay_mode=replay_mode,
            )
        sampling_rows.append(row)
        return session, digest

    with tempfile.TemporaryDirectory(
        prefix=".supplemental-carry-spool-", dir=destination.parent
    ) as spool:
        replay = replay_supplemental_continuations(
            inputs.paths,
            inputs.actions,
            inputs.exits,
            sessions,
            calendar,
            session_loader=session_loader,
            spool_root=Path(spool),
            max_pair_sessions=max_pair_sessions,
            replay_mode=replay_mode,
        )
    replay = replace(
        replay,
        candidate_session_sampling_audit=_frame_or_empty(
            sampling_rows, _sampling_audit_schema()
        ),
    )
    supplemental_paths = apply_supplemental_carry_terminal_overlay(
        inputs.paths,
        inputs.entry_prices,
        replay.continuation_terminals,
        replay.expiry_marks,
        config=SupplementalCarryConfig(),
    )
    if publish:
        if replay.truncated:
            raise ValueError("a truncated benchmark cannot be published")
        _publish_result(
            destination,
            inputs,
            replay,
            supplemental_paths,
            source_metadata={
                **dict(inputs.source_metadata),
                "prerequisite_root": str(prerequisite.resolve()),
                "candidate_sessions_sha256": _file_sha256(sessions_path),
                "contract_calendar_sha256": _file_sha256(calendar_path),
                "candidate_cache_root": str(Path(candidate_cache_root).resolve()),
                "replay_mode": replay_mode,
                "state_sampling_approximate": (
                    replay_mode == ONE_SECOND_REPLAY_MODE
                ),
            },
        )
    return replay, supplemental_paths


def _default_policy_day_replayer(
    actions: pl.DataFrame,
    exits: pl.DataFrame,
    session: CrossSessionExitMakerSession,
    active_trial_ids: Sequence[str],
    spool_root: Path,
) -> pl.DataFrame:
    prepared = session.raw_tape
    config = ExitMakerStudyConfig(
        hedge_delay_ns=50_000_000,
        max_book_age_ns=None,
        expected_exit_rule_ids=("frozen_center", "frozen_lower"),
        exit_lifecycle_policy_version=(
            "supplemental_imputed_full_carry_v1/day_order"
        ),
        exit_queue_scenario="displayed_queue_independent_daily_reset_v1",
        instant_cancel_v0=True,
    )
    return _spool_day_policy_facts(
        actions,
        exits,
        prepared,
        session.spread_pair_clock,
        config,
        cutoff_cursor=session.cutoff_cursor,
        active_exit_policy_trial_ids=active_trial_ids,
        spool_root=spool_root,
    )


def _continuation_record(
    state: Mapping[str, object],
    row: Mapping[str, object],
    *,
    session_date: str,
    cache_marker_sha: str,
    replay_mode: str,
) -> dict[str, object]:
    spot = _positive_float(row.get("exit_spot_price"), "exit_spot_price")
    future = _positive_float(row.get("exit_future_price"), "exit_future_price")
    terminal_ns = _nonnegative_int(
        row.get("exit_decision_time_ns"), "exit_decision_time_ns"
    )
    identity = _canonical_sha256(
        {
            "cache_marker_sha256": cache_marker_sha,
            "policy_path_id": state["policy_path_id"],
            "terminal_date": session_date,
            "exit_decision_time_ns": terminal_ns,
            "exit_spot_price": spot,
            "exit_future_price": future,
            "replay_mode": replay_mode,
        }
    )
    approximate = replay_mode == ONE_SECOND_REPLAY_MODE
    return {
        "policy_path_id": str(state["policy_path_id"]),
        "terminal_date": session_date,
        "exit_decision_time_ns": terminal_ns,
        "exit_spot_price": spot,
        "exit_future_price": future,
        "exact_price_source": (
            "supplemental_candidate_cache_one_second_state_approx_exit_replay"
            if approximate
            else "supplemental_candidate_cache_exact_raw_state_exit_replay"
        ),
        "source_identity_sha256": identity,
        "model_imputed_full_carry_on_unknown": True,
        "double_exit_bias_possible": True,
        "supplemental_replay_mode": replay_mode,
        "state_sampling_approximate": approximate,
        "target_cancel_clock_exact": not approximate,
        "delayed_hedge_snapshot_exact": not approximate,
    }


def _audit_base(
    state: Mapping[str, object],
    blocker: str | None,
    *,
    replay_mode: str,
) -> dict[str, object]:
    approximate = replay_mode == ONE_SECOND_REPLAY_MODE
    return {
        "policy_path_id": str(state["policy_path_id"]),
        "Date": str(state["Date"]),
        "ValueCode": str(state["ValueCode"]),
        "QuoteCode": str(state["QuoteCode"]),
        "source_outcome_status": str(state["outcome_status"]),
        "last_observed_session_date": str(state["last_observed_session_date"]),
        "expiry_session": str(state["expiry_session"]),
        "sessions_replayed": 0,
        "imputed_unknown_sessions": int(state["imputed_unknown_sessions"]),
        "terminal_resolution": None,
        "terminal_date": None,
        "blocker_status": blocker,
        "blocker_detail": blocker,
        "model_imputed_full_carry_on_unknown": (
            str(state["outcome_status"]) == "maker_fill_state_unknown"
        ),
        "double_exit_bias_possible": (
            str(state["outcome_status"]) == "maker_fill_state_unknown"
        ),
        "analysis_only": True,
        "production_strategy_go": False,
        "supplemental_replay_mode": replay_mode,
        "state_sampling_approximate": approximate,
        "target_cancel_clock_exact": not approximate,
        "delayed_hedge_snapshot_exact": not approximate,
    }


def _block(
    state: dict[str, object],
    audit: dict[str, dict[str, object]],
    status: str,
    detail: object,
) -> None:
    state["active"] = False
    audit[str(state["policy_path_id"])].update(
        {"blocker_status": status, "blocker_detail": str(detail)}
    )


def _read_verified_partition_artifact(
    partition: Path,
    key: tuple[str, str],
    filename: str,
    *,
    columns: Sequence[str],
    filter_column: str,
    identifiers: Iterable[str],
) -> tuple[pl.DataFrame, dict[str, object]]:
    marker_path = Path(partition) / "complete.json"
    artifact_path = Path(partition) / filename
    marker = _read_json(marker_path)
    artifacts = marker.get("artifacts")
    declaration = artifacts.get(filename) if isinstance(artifacts, dict) else None
    if (
        marker.get("complete") is not True
        or str(marker.get("Date")) != key[0]
        or str(marker.get("ValueCode")) != key[1]
        or not isinstance(declaration, dict)
        or _file_sha256(artifact_path) != declaration.get("sha256")
        or artifact_path.stat().st_size != int(declaration.get("bytes", -1))
    ):
        raise ValueError(f"supplemental source artifact changed: {artifact_path}")
    selected = sorted({str(value) for value in identifiers})
    if not selected or filter_column not in columns:
        raise ValueError("artifact filter identifiers/column are invalid")
    schema = pl.read_parquet_schema(artifact_path)
    missing = set(columns) - set(schema)
    if missing:
        raise ValueError(
            f"supplemental source columns are missing: {sorted(missing)}"
        )
    frame = (
        pl.scan_parquet(artifact_path)
        .select(columns)
        .filter(pl.col(filter_column).cast(pl.String).is_in(selected))
        .collect()
    )
    return frame, {
        "Date": key[0],
        "ValueCode": key[1],
        "filename": filename,
        "partition_complete_path": str(marker_path),
        "partition_complete_sha256": _file_sha256(marker_path),
        "artifact_path": str(artifact_path),
        "artifact_sha256": str(declaration["sha256"]),
        "artifact_bytes": int(declaration["bytes"]),
        "artifact_rows": int(declaration["rows"]),
    }


def _validate_source_fingerprint(fingerprint: Mapping[str, object]) -> None:
    for label, payload in fingerprint.items():
        if label in {"Date", "content_integrity_bound"}:
            continue
        if not isinstance(payload, dict):
            raise ValueError("candidate source fingerprint is malformed")
        path = Path(str(payload.get("path")))
        expected_exists = payload.get("exists") is True
        if path.is_file() != expected_exists:
            raise ValueError(f"candidate cache source existence changed: {path}")
        if expected_exists:
            stat = path.stat()
            if stat.st_size != int(payload.get("bytes", -1)) or stat.st_mtime_ns != int(
                payload.get("mtime_ns", -1)
            ):
                raise ValueError(f"candidate cache source stat changed: {path}")


def _calendar_lookup(frame: pl.DataFrame) -> dict[str, tuple[str, str]]:
    required = {"QuoteCode", "expiry_session", "calendar_version"}
    if required - set(frame.columns):
        raise ValueError("contract calendar columns are incomplete")
    if frame["QuoteCode"].n_unique() != frame.height:
        raise ValueError("contract calendar QuoteCode is duplicated")
    return {
        str(row["QuoteCode"]): (
            str(row["expiry_session"]),
            str(row["calendar_version"]),
        )
        for row in frame.iter_rows(named=True)
    }


def _normalise_sessions(values: Sequence[str] | Iterable[str]) -> tuple[str, ...]:
    result = tuple(str(value).strip() for value in values if str(value).strip())
    if not result or result != tuple(sorted(set(result))):
        raise ValueError("sessions must be nonempty, ascending and unique")
    return result


def _validate_replay_mode(value: str) -> None:
    if value not in {EXACT_REPLAY_MODE, ONE_SECOND_REPLAY_MODE}:
        raise ValueError(
            "replay_mode must be exact_raw_state_v1 or "
            "one_second_last_state_plus_spread_epoch_approx_v1"
        )


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _frame_or_empty(
    rows: list[dict[str, object]], schema: Mapping[str, pl.DataType]
) -> pl.DataFrame:
    if rows:
        return pl.from_dicts(rows, schema=schema, infer_schema_length=None)
    return pl.DataFrame(schema=schema)


def _continuation_schema() -> dict[str, pl.DataType]:
    return {
        "policy_path_id": pl.String,
        "terminal_date": pl.String,
        "exit_decision_time_ns": pl.Int64,
        "exit_spot_price": pl.Float64,
        "exit_future_price": pl.Float64,
        "exact_price_source": pl.String,
        "source_identity_sha256": pl.String,
        "model_imputed_full_carry_on_unknown": pl.Boolean,
        "double_exit_bias_possible": pl.Boolean,
        "supplemental_replay_mode": pl.String,
        "state_sampling_approximate": pl.Boolean,
        "target_cancel_clock_exact": pl.Boolean,
        "delayed_hedge_snapshot_exact": pl.Boolean,
    }


def _sampling_audit_schema() -> dict[str, pl.DataType]:
    return {
        "Date": pl.String,
        "ValueCode": pl.String,
        "QuoteCode": pl.String,
        "replay_mode": pl.String,
        "state_sampling_approximate": pl.Boolean,
        "sample_interval_seconds": pl.Int64,
        "spot_spread_epoch_changes_always_retained": pl.Boolean,
        "raw_trade_events_unchanged": pl.Boolean,
        "spot_states_source_rows": pl.Int64,
        "spot_states_replay_rows": pl.Int64,
        "future_states_source_rows": pl.Int64,
        "future_states_replay_rows": pl.Int64,
        "spot_trades_rows": pl.Int64,
        "future_trades_rows": pl.Int64,
        "target_cancel_clock_exact": pl.Boolean,
        "initial_queue_snapshot_exact": pl.Boolean,
        "delayed_hedge_snapshot_exact": pl.Boolean,
    }


def _mark_schema() -> dict[str, pl.DataType]:
    return {
        "Date": pl.String,
        "expiry_session": pl.String,
        "calendar_version": pl.String,
        "ValueCode": pl.String,
        "QuoteCode": pl.String,
        "spot_close_price": pl.Float64,
        "future_close_price": pl.Float64,
        "spot_close_time_ns": pl.Int64,
        "future_close_time_ns": pl.Int64,
        "spot_close_source": pl.String,
        "future_close_source": pl.String,
        "source_identity_sha256": pl.String,
        "mark_is_official_close": pl.Boolean,
        "mark_is_official_settlement": pl.Boolean,
        "spot_mark_is_executable_bbo": pl.Boolean,
        "future_mark_is_executable_bbo": pl.Boolean,
        "mark_uses_trade_fallback": pl.Boolean,
        "mark_role": pl.String,
    }


def _publish_result(
    destination: Path,
    inputs: SupplementalReplayInputs,
    replay: SupplementalReplayResult,
    supplemental_paths: pl.DataFrame,
    *,
    source_metadata: Mapping[str, object],
) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    frames = {
        "unresolved_entry_prices.parquet": inputs.entry_prices,
        "source_inventory.parquet": inputs.source_inventory,
        "continuation_terminals.parquet": replay.continuation_terminals,
        "expiry_marks.parquet": replay.expiry_marks,
        "continuation_audit.parquet": replay.continuation_audit,
        "candidate_session_sampling_audit.parquet": (
            replay.candidate_session_sampling_audit
        ),
        "supplemental_paths.parquet": supplemental_paths,
    }
    try:
        declarations: dict[str, dict[str, object]] = {}
        for filename, frame in frames.items():
            path = stage / filename
            frame.write_parquet(path)
            declarations[filename] = {
                "rows": frame.height,
                "columns": frame.width,
                "sha256": _file_sha256(path),
                "bytes": path.stat().st_size,
                "schema": {name: str(dtype) for name, dtype in frame.schema.items()},
            }
        marker: dict[str, object] = {
            "complete": True,
            "runner_version": RUNNER_VERSION,
            "analysis_only": True,
            "model_imputed_full_carry_on_unknown": True,
            "double_exit_bias_possible": True,
            "expiry_mark_is_official_settlement": False,
            "pair_sessions_replayed": replay.pair_sessions_replayed,
            "truncated": False,
            "replay_mode": replay.replay_mode,
            "state_sampling_approximate": replay.state_sampling_approximate,
            "target_cancel_clock_exact": (
                not replay.state_sampling_approximate
            ),
            "initial_queue_snapshot_exact": (
                not replay.state_sampling_approximate
            ),
            "delayed_hedge_snapshot_exact": (
                not replay.state_sampling_approximate
            ),
            "raw_trade_events_unchanged_under_state_sampling": True,
            "production_strategy_go": False,
            "sources": json.loads(json.dumps(dict(source_metadata), sort_keys=True)),
            "artifacts": declarations,
        }
        marker["marker_payload_sha256"] = _canonical_sha256(marker)
        (stage / "complete.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        stage.rename(destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--universe-root", type=Path, default=DEFAULT_UNIVERSE_ROOT)
    parser.add_argument(
        "--exact-path-source-root",
        type=Path,
        default=DEFAULT_EXACT_PATH_SOURCE_ROOT,
    )
    parser.add_argument(
        "--prerequisite-root", type=Path, default=DEFAULT_PREREQUISITE_ROOT
    )
    parser.add_argument(
        "--candidate-cache-root", type=Path, default=DEFAULT_CANDIDATE_CACHE_ROOT
    )
    parser.add_argument(
        "--replay-mode",
        choices=(ONE_SECOND_REPLAY_MODE, EXACT_REPLAY_MODE),
        default=DEFAULT_REPLAY_MODE,
        help=(
            "default is the disclosed one-second state approximation; "
            "exact raw-state replay remains available for sensitivity checks"
        ),
    )
    parser.add_argument("--benchmark-max-pair-sessions", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    benchmark = args.benchmark_max_pair_sessions
    if benchmark is not None and benchmark <= 0:
        raise ValueError("--benchmark-max-pair-sessions must be positive")
    replay, paths = run_supplemental_carry_analysis(
        args.output,
        source_root=args.source_root,
        universe_root=args.universe_root,
        exact_path_source_root=args.exact_path_source_root,
        prerequisite_root=args.prerequisite_root,
        candidate_cache_root=args.candidate_cache_root,
        max_pair_sessions=benchmark,
        replay_mode=args.replay_mode,
        publish=benchmark is None,
    )
    print(
        json.dumps(
            {
                "pair_sessions_replayed": replay.pair_sessions_replayed,
                "continuation_terminals": replay.continuation_terminals.height,
                "expiry_marks": replay.expiry_marks.height,
                "audit_rows": replay.continuation_audit.height,
                "supplemental_completed_paths": paths.filter(
                    pl.col("terminal_cashflow_priced")
                ).height,
                "truncated": replay.truncated,
                "published": benchmark is None,
                "replay_mode": replay.replay_mode,
                "state_sampling_approximate": replay.state_sampling_approximate,
                "sampled_candidate_sessions": (
                    replay.candidate_session_sampling_audit.height
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
