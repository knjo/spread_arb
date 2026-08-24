"""Independent close labels and a scenario-local FIFO contract simulator.

The first layer labels every position/policy opportunity independently.  The
optional second layer tests a proposed residual-capacity contract.  There is
not yet a source-bound capacity producer, so it cannot prove real fills and
never publishes terminal-ready, exact, or joint-allocation claims.  Raw prints
are *not* strategy capacity.

FIFO has two distinct stages.  Physical maker orders first compete by their
full submit ``EventCursor`` at each product/route/absolute price.  Quantity
received by an aggregate physical order is then attributed to its positions
by their full establishment ``EventCursor``.  Policy aliases never create
either demand or capacity, and approximate makerFill labels are ineligible.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Iterable, Mapping

import polars as pl


INDEPENDENT_EXIT_VERSION = "compact_independent_exit_v2_exact_cursor"
FIFO_EXIT_VERSION = "compact_fifo_contract_simulation_v3_two_stage"
CAPACITY_BASIS = "claimed_post_external_queue_residual_contract"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CURSOR_SUFFIXES = ("recv_time_ns", "event_sequence", "row_index")
_INDEPENDENT_SCHEMA: dict[str, pl.DataType] = {
    "scenario_id": pl.String,
    "Date": pl.String,
    "ValueCode": pl.String,
    "QuoteCode": pl.String,
    "position_id": pl.String,
    "position_established_recv_time_ns": pl.Int64,
    "position_established_event_sequence": pl.Int64,
    "position_established_row_index": pl.Int64,
    "exit_route": pl.String,
    "maker_market": pl.String,
    "maker_side": pl.String,
    "exit_policy_generation_id": pl.String,
    "physical_exit_order_id": pl.String,
    "absolute_price_tick": pl.Int64,
    "submit_recv_time_ns": pl.Int64,
    "submit_event_sequence": pl.Int64,
    "submit_row_index": pl.Int64,
    "nominal_stop_recv_time_ns": pl.Int64,
    "nominal_stop_event_sequence": pl.Int64,
    "nominal_stop_row_index": pl.Int64,
    "requested_quantity": pl.Int64,
    "physical_order_quantity": pl.Int64,
    "outcome_supported": pl.Boolean,
    "full_fill": pl.Boolean,
    "full_fill_recv_time_ns": pl.Int64,
    "full_fill_event_sequence": pl.Int64,
    "full_fill_row_index": pl.Int64,
    "fill_cursor_exact": pl.Boolean,
    "hedge_label_observed": pl.Boolean,
    "hedge_executable": pl.Boolean,
    "hedge_decision_time_ns": pl.Int64,
    "hedge_input_fill_cursor_exact": pl.Boolean,
}
_INDEPENDENT_REQUIRED = set(_INDEPENDENT_SCHEMA)
_INDEPENDENT_DERIVED_SCHEMA: dict[str, pl.DataType] = {
    "close_opportunity_id": pl.String,
    "independent_close_status": pl.String,
    "independent_close_quantity": pl.Int64,
    "fifo_capacity_eligible": pl.Boolean,
    "independent_1_to_1": pl.Boolean,
    "joint_volume_allocated": pl.Boolean,
    "pathwise_ev_ready": pl.Boolean,
    "exit_phase_version": pl.String,
}
_CAPACITY_SCHEMA: dict[str, pl.DataType] = {
    "scenario_id": pl.String,
    "capacity_event_id": pl.String,
    "source_trade_event_id": pl.String,
    "Date": pl.String,
    "ValueCode": pl.String,
    "exit_route": pl.String,
    "maker_market": pl.String,
    "maker_side": pl.String,
    "absolute_price_tick": pl.Int64,
    "fill_recv_time_ns": pl.Int64,
    "fill_event_sequence": pl.Int64,
    "fill_row_index": pl.Int64,
    "source_trade_quantity": pl.Int64,
    "external_queue_ahead_before": pl.Int64,
    "external_queue_ahead_after": pl.Int64,
    "available_quantity": pl.Int64,
    "capacity_basis": pl.String,
    "source_replay_claimed_exact": pl.Boolean,
    "source_replay_version": pl.String,
    "source_tape_sha256": pl.String,
}
_CAPACITY_REQUIRED = set(_CAPACITY_SCHEMA)


def build_independent_close_opportunities(
    exit_order_outcomes: pl.DataFrame,
) -> pl.DataFrame:
    """Label each position/exit attempt independently, without allocation."""

    if exit_order_outcomes.is_empty():
        normalized = _normalize_empty(exit_order_outcomes, _INDEPENDENT_SCHEMA)
        return _append_independent_columns(normalized)
    _require(exit_order_outcomes, _INDEPENDENT_REQUIRED, "exit outcomes")
    outcomes = exit_order_outcomes.select(
        *[
            pl.col(column).cast(dtype, strict=True).alias(column)
            for column, dtype in _INDEPENDENT_SCHEMA.items()
        ]
    )
    _validate_independent_inputs(outcomes)
    _validate_independent_physical_coherence(outcomes)
    records: list[dict[str, object]] = []
    for row in outcomes.iter_rows(named=True):
        supported = bool(row["outcome_supported"])
        full = row["full_fill"]
        exact = bool(row["fill_cursor_exact"])
        hedge_observed = bool(row["hedge_label_observed"])
        hedge_executable = bool(row["hedge_executable"])
        if not supported:
            status = "unsupported_unknown"
        elif full is None:
            status = "unknown_outcome_unallocatable"
        elif full is False:
            status = "no_independent_intraday_close"
        elif not exact:
            status = "approx_close_unallocatable"
        elif not hedge_observed:
            status = "exact_fill_hedge_unknown"
        elif not hedge_executable:
            status = "exact_fill_hedge_failed"
        else:
            status = "exact_independent_intraday_close"
        eligible = status == "exact_independent_intraday_close"
        records.append(
            {
                **row,
                "close_opportunity_id": _close_id(row),
                "independent_close_status": status,
                "independent_close_quantity": (
                    int(row["requested_quantity"]) if eligible else None
                ),
                "fifo_capacity_eligible": eligible,
                "independent_1_to_1": True,
                "joint_volume_allocated": False,
                "pathwise_ev_ready": False,
                "exit_phase_version": INDEPENDENT_EXIT_VERSION,
            }
        )
    result = pl.from_dicts(records, infer_schema_length=None).select(
        *[
            pl.col(column).cast(dtype, strict=True).alias(column)
            for column, dtype in {
                **_INDEPENDENT_SCHEMA,
                **_INDEPENDENT_DERIVED_SCHEMA,
            }.items()
        ]
    )
    return result.sort(
        [
            "scenario_id",
            "Date",
            "ValueCode",
            "exit_route",
            "position_established_recv_time_ns",
            "position_established_event_sequence",
            "position_established_row_index",
            "position_id",
            "submit_recv_time_ns",
            "submit_event_sequence",
            "submit_row_index",
            "exit_policy_generation_id",
        ]
    )


def allocate_fifo_exact_capacity(
    independent_opportunities: pl.DataFrame,
    capacity_events: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Fail closed until a source-bound real capacity producer exists."""

    del independent_opportunities, capacity_events
    raise RuntimeError(
        "exact FIFO is unavailable: no verified source-bound residual-capacity "
        "producer exists; use simulate_fifo_residual_capacity_contract only "
        "for contract/synthetic tests"
    )


def simulate_fifo_residual_capacity_contract(
    independent_opportunities: pl.DataFrame,
    capacity_events: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Simulate two-stage FIFO against claimed residual capacity.

    This validates arithmetic, IDs, cursor ordering, and alias de-duplication,
    but it cannot prove that a claimed tape SHA/replay version exists.  Results
    therefore keep formal readiness, exactness, and joint-allocation flags
    false.  Scenarios are explicitly mutually-exclusive counterfactuals: one
    source event may reappear across scenarios but never twice within one.
    """

    if independent_opportunities.is_empty():
        if not capacity_events.is_empty():
            raise ValueError("capacity exists without any close opportunities")
        return _empty_physical_allocations(), _empty_alias_projection(
            independent_opportunities
        )
    _require(
        independent_opportunities,
        _INDEPENDENT_REQUIRED
        | {
            "close_opportunity_id",
            "independent_close_status",
            "independent_close_quantity",
            "fifo_capacity_eligible",
            "independent_1_to_1",
            "joint_volume_allocated",
            "pathwise_ev_ready",
            "exit_phase_version",
        },
        "independent close opportunities",
    )
    independent_opportunities = independent_opportunities.select(
        *[
            pl.col(column).cast(dtype, strict=True).alias(column)
            for column, dtype in {
                **_INDEPENDENT_SCHEMA,
                **_INDEPENDENT_DERIVED_SCHEMA,
            }.items()
        ]
    )
    if capacity_events.is_empty():
        capacity_events = _normalize_empty(capacity_events, _CAPACITY_SCHEMA)
    _require(capacity_events, _CAPACITY_REQUIRED, "capacity events")
    capacity_events = _validate_capacity_events(capacity_events)
    _validate_recomputed_independent_fields(independent_opportunities)

    logical_positions = _collapse_policy_aliases(independent_opportunities)
    physical_orders = _collapse_physical_orders(logical_positions)
    eligible_physical = [
        row for row in physical_orders if bool(row["fifo_capacity_eligible"])
    ]
    if not eligible_physical:
        return _empty_physical_allocations(), _project_unallocated_aliases(
            independent_opportunities
        )

    capacities = capacity_events.sort(
        [
            "scenario_id",
            "Date",
            "ValueCode",
            "exit_route",
            "absolute_price_tick",
            "fill_recv_time_ns",
            "fill_event_sequence",
            "fill_row_index",
        ]
    ).to_dicts()
    capacity_by_key: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for event in capacities:
        capacity_by_key.setdefault(_capacity_key(event), []).append(event)

    demand_by_key: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for order in eligible_physical:
        demand_by_key.setdefault(_capacity_key(order), []).append(order)

    physical_records: list[dict[str, object]] = []
    fill_chunks: dict[
        tuple[str, str], list[tuple[tuple[int, int, int], int]]
    ] = {}
    for key, demands in demand_by_key.items():
        queue = sorted(demands, key=lambda row: _cursor(row, "submit"))
        submit_cursors = [_cursor(row, "submit") for row in queue]
        if len(submit_cursors) != len(set(submit_cursors)):
            raise ValueError(
                "physical exchange FIFO is ambiguous at an identical submit cursor"
            )
        remaining = {
            _physical_key(row): int(row["physical_order_quantity"])
            for row in queue
        }
        allocations = {identifier: 0 for identifier in remaining}
        for event in capacity_by_key.get(key, []):
            event_cursor = _cursor(event, "fill")
            capacity = int(event["available_quantity"])
            for demand in queue:
                if capacity <= 0:
                    break
                identifier = _physical_key(demand)
                if not (
                    _cursor(demand, "submit") < event_cursor
                    <= _cursor(demand, "nominal_stop")
                ):
                    continue
                needed = remaining[identifier]
                if needed <= 0:
                    continue
                allocated = min(needed, capacity)
                allocations[identifier] += allocated
                remaining[identifier] -= allocated
                capacity -= allocated
                fill_chunks.setdefault(identifier, []).append(
                    (event_cursor, allocated)
                )
        for demand in queue:
            identifier = _physical_key(demand)
            chunks = fill_chunks.get(identifier, [])
            allocated = allocations[identifier]
            requested = int(demand["physical_order_quantity"])
            first = chunks[0][0] if chunks else None
            complete = chunks[-1][0] if allocated == requested and chunks else None
            physical_records.append(
                {
                    **demand,
                    "fifo_allocated_quantity": allocated,
                    "fifo_remaining_quantity": requested - allocated,
                    **_cursor_columns("fifo_first_fill", first),
                    **_cursor_columns("fifo_full_fill", complete),
                    "fifo_full_fill_allocated": allocated == requested,
                    "scenario_local_unique_claim_events": True,
                    "capacity_basis": CAPACITY_BASIS,
                    "capacity_provenance_verified": False,
                    "experimental_contract_only": True,
                    "actual_fifo_metrics_absent": True,
                    "mutually_exclusive_scenario": True,
                    "exchange_fifo_exact": False,
                    "position_fifo_attribution_exact": False,
                    "joint_volume_allocated": False,
                    "pathwise_ev_ready": False,
                    "fifo_version": FIFO_EXIT_VERSION,
                }
            )

    physical = pl.from_dicts(physical_records, infer_schema_length=None).sort(
        [
            "scenario_id",
            "Date",
            "ValueCode",
            "exit_route",
            "absolute_price_tick",
            "submit_recv_time_ns",
            "submit_event_sequence",
            "submit_row_index",
        ]
    )
    position_records = _attribute_positions_fifo(
        logical_positions, physical_records, fill_chunks
    )
    positions = pl.from_dicts(position_records, infer_schema_length=None)
    physical_status = positions.group_by(
        "scenario_id", "physical_exit_order_id"
    ).agg(
        pl.col("fifo_cursor_matches_independent")
        .all()
        .alias("fifo_cursor_matches_independent"),
        pl.col("fifo_outcome_relabel_required")
        .any()
        .alias("fifo_outcome_relabel_required"),
        pl.col("fifo_hedge_relabel_required")
        .any()
        .alias("fifo_hedge_relabel_required"),
    )
    physical = physical.join(
        physical_status,
        on=["scenario_id", "physical_exit_order_id"],
        how="left",
        validate="1:1",
    ).with_columns(
        pl.lit(False).alias("fifo_close_terminal_ready"),
        pl.lit(False).alias("joint_volume_allocated"),
        pl.lit(False).alias("capacity_provenance_verified"),
        pl.lit(True).alias("experimental_contract_only"),
        pl.lit(True).alias("actual_fifo_metrics_absent"),
        pl.lit(True).alias("mutually_exclusive_scenario"),
    )
    projection = (
        independent_opportunities.drop("joint_volume_allocated")
        .join(
            positions,
            on=["scenario_id", "physical_exit_order_id", "position_id"],
            how="left",
            validate="m:1",
        )
        .with_columns(
            pl.col("fifo_allocated_quantity").fill_null(0),
            pl.col("fifo_full_fill_allocated").fill_null(False),
            pl.col("fifo_close_terminal_ready").fill_null(False),
            pl.col("fifo_cursor_matches_independent").fill_null(False),
            pl.col("fifo_outcome_relabel_required").fill_null(False),
            pl.col("fifo_hedge_relabel_required").fill_null(False),
            pl.col("joint_volume_allocated").fill_null(False),
            pl.col("capacity_provenance_verified").fill_null(False),
            pl.col("experimental_contract_only").fill_null(True),
            pl.col("actual_fifo_metrics_absent").fill_null(True),
            pl.col("mutually_exclusive_scenario").fill_null(True),
            pl.col("fifo_version").fill_null(FIFO_EXIT_VERSION),
        )
    )
    used = sum(int(row["fifo_allocated_quantity"]) for row in physical_records)
    available = int(capacity_events["available_quantity"].sum())
    if used > available:
        raise ValueError("FIFO allocated more than unique residual capacity")
    return physical, projection


def canonical_source_trade_event_id(row: Mapping[str, object]) -> str:
    """Return the immutable raw-trade identity used by capacity validation."""

    payload = {
        "Date": str(row["Date"]),
        "ValueCode": str(row["ValueCode"]),
        "maker_market": str(row["maker_market"]),
        "fill_cursor": list(_cursor(row, "fill")),
        "source_tape_sha256": str(row["source_tape_sha256"]),
    }
    return _canonical_sha256(payload)


def canonical_capacity_event_id(row: Mapping[str, object]) -> str:
    """Return the scenario-local identity of one residual capacity event."""

    payload = {
        "scenario_id": str(row["scenario_id"]),
        "source_trade_event_id": str(row["source_trade_event_id"]),
        "exit_route": str(row["exit_route"]),
        "absolute_price_tick": int(row["absolute_price_tick"]),
        "external_queue_ahead_before": int(
            row["external_queue_ahead_before"]
        ),
        "external_queue_ahead_after": int(row["external_queue_ahead_after"]),
        "available_quantity": int(row["available_quantity"]),
        "capacity_basis": str(row["capacity_basis"]),
        "source_replay_version": str(row["source_replay_version"]),
    }
    return _canonical_sha256(payload)


def _validate_independent_inputs(outcomes: pl.DataFrame) -> None:
    if outcomes.select("exit_policy_generation_id").n_unique() != outcomes.height:
        raise ValueError("exit policy generation ids must be globally unique")
    nonnull = [
        column
        for column in _INDEPENDENT_SCHEMA
        if column
        not in {
            "full_fill",
            "full_fill_recv_time_ns",
            "full_fill_event_sequence",
            "full_fill_row_index",
            "hedge_decision_time_ns",
        }
    ]
    if outcomes.filter(
        pl.any_horizontal([pl.col(column).is_null() for column in nonnull])
    ).height:
        raise ValueError("exit outcomes contain null identity/contract fields")
    if outcomes.filter(
        (pl.col("requested_quantity") <= 0)
        | (pl.col("physical_order_quantity") <= 0)
        | (pl.col("requested_quantity") > pl.col("physical_order_quantity"))
    ).height:
        raise ValueError("close quantities must be positive and physically bounded")
    for row in outcomes.iter_rows(named=True):
        _validate_route_market_side(row)
        string_identity = (
            "scenario_id",
            "Date",
            "ValueCode",
            "QuoteCode",
            "position_id",
            "exit_route",
            "maker_market",
            "maker_side",
            "exit_policy_generation_id",
            "physical_exit_order_id",
        )
        if any(not str(row[column]).strip() for column in string_identity) or int(
            row["absolute_price_tick"]
        ) <= 0:
            raise ValueError("exit scenario/order/price identity is invalid")
        established = _cursor(row, "position_established")
        submit = _cursor(row, "submit")
        stop = _cursor(row, "nominal_stop")
        if not established <= submit < stop:
            raise ValueError("position/submit/stop EventCursor order is invalid")
        supported = bool(row["outcome_supported"])
        full = row["full_fill"]
        full_values = tuple(
            row[f"full_fill_{suffix}"] for suffix in _CURSOR_SUFFIXES
        )
        full_complete = all(value is not None for value in full_values)
        full_partial = any(value is not None for value in full_values) and not full_complete
        if full_partial:
            raise ValueError("full-fill EventCursor is only partially populated")
        if not supported:
            if (
                full is not None
                or full_complete
                or bool(row["fill_cursor_exact"])
                or bool(row["hedge_label_observed"])
                or bool(row["hedge_executable"])
                or row["hedge_decision_time_ns"] is not None
                or bool(row["hedge_input_fill_cursor_exact"])
            ):
                raise ValueError(
                    "unsupported exit outcome carries exact fill/hedge truth"
                )
            continue
        if full is True:
            if not full_complete:
                raise ValueError("full exit lacks an exact EventCursor")
            fill_cursor = tuple(int(value) for value in full_values)
            if not submit < fill_cursor <= stop:
                raise ValueError("full exit occurs outside its active interval")
        elif full is False:
            if full_complete:
                raise ValueError("no-fill exit carries a full-fill EventCursor")
        else:
            if full_complete:
                raise ValueError("unknown exit outcome carries a fill cursor")
            if bool(row["fill_cursor_exact"]):
                raise ValueError("unknown exit outcome claims an exact fill cursor")
        if bool(row["hedge_label_observed"]):
            if (
                full is not True
                or not bool(row["fill_cursor_exact"])
                or row["hedge_input_fill_cursor_exact"] is not True
                or row["hedge_decision_time_ns"]
                != row["full_fill_recv_time_ns"] + 50_000_000
            ):
                raise ValueError("observed exit hedge is not exact full-fill+50ms")
        if bool(row["hedge_executable"]) and not bool(
            row["hedge_label_observed"]
        ):
            raise ValueError("executable exit hedge is not observed")


def _validate_independent_physical_coherence(outcomes: pl.DataFrame) -> None:
    """Lock logical-position aliases and aggregate physical-order identity."""

    logical_key = ["scenario_id", "physical_exit_order_id", "position_id"]
    logical_identity = [
        column
        for column in _INDEPENDENT_SCHEMA
        if column != "exit_policy_generation_id"
    ]
    logical = outcomes.group_by(logical_key).agg(
        pl.struct(*logical_identity).n_unique().alias("identity_values")
    )
    if logical.filter(pl.col("identity_values") != 1).height:
        raise ValueError(
            "policy aliases disagree on logical position lifecycle/outcome"
        )
    physical_identity = [
        "Date",
        "ValueCode",
        "QuoteCode",
        "exit_route",
        "maker_market",
        "maker_side",
        "absolute_price_tick",
        "submit_recv_time_ns",
        "submit_event_sequence",
        "submit_row_index",
        "nominal_stop_recv_time_ns",
        "nominal_stop_event_sequence",
        "nominal_stop_row_index",
        "physical_order_quantity",
    ]
    physical = outcomes.group_by(
        "scenario_id", "physical_exit_order_id"
    ).agg(pl.struct(*physical_identity).n_unique().alias("identity_values"))
    if physical.filter(pl.col("identity_values") != 1).height:
        raise ValueError("positions disagree on aggregate physical exit order")
    logical_quantity = outcomes.select(
        "scenario_id",
        "physical_exit_order_id",
        "position_id",
        "requested_quantity",
        "physical_order_quantity",
    ).unique(
        subset=["scenario_id", "physical_exit_order_id", "position_id"],
        maintain_order=True,
    )
    physical_quantity = logical_quantity.group_by(
        "scenario_id", "physical_exit_order_id"
    ).agg(
        pl.col("requested_quantity").sum().alias("logical_position_quantity"),
        pl.col("physical_order_quantity").first().alias("physical_order_quantity"),
    )
    if physical_quantity.filter(
        pl.col("logical_position_quantity") != pl.col("physical_order_quantity")
    ).height:
        raise ValueError(
            "logical position demand does not equal physical order quantity"
        )


def _validate_capacity_events(capacity: pl.DataFrame) -> pl.DataFrame:
    """Return the canonical typed capacity frame used by the simulator."""

    if capacity.is_empty():
        return _normalize_empty(capacity, _CAPACITY_SCHEMA)
    normalized = capacity.select(
        *[
            pl.col(column).cast(dtype, strict=True).alias(column)
            for column, dtype in _CAPACITY_SCHEMA.items()
        ]
    )
    if normalized.filter(
        pl.any_horizontal(
            [pl.col(column).is_null() for column in _CAPACITY_SCHEMA]
        )
    ).height:
        raise ValueError("capacity events contain null contract fields")
    if normalized.select("capacity_event_id").n_unique() != normalized.height:
        raise ValueError("capacity_event_id must be globally unique")
    if (
        normalized.select("scenario_id", "source_trade_event_id").n_unique()
        != normalized.height
    ):
        raise ValueError("one raw trade is reused within a FIFO scenario")
    for row in normalized.iter_rows(named=True):
        if row["capacity_basis"] != CAPACITY_BASIS:
            raise ValueError("capacity is not post-external-queue residual")
        if row["source_replay_claimed_exact"] is not True:
            raise ValueError("capacity source replay is not claimed exact")
        if not str(row["source_replay_version"]).strip():
            raise ValueError("capacity source replay version is missing")
        if not _SHA256_RE.fullmatch(str(row["source_tape_sha256"])):
            raise ValueError("capacity source tape SHA-256 is invalid")
        traded = int(row["source_trade_quantity"])
        before = int(row["external_queue_ahead_before"])
        after = int(row["external_queue_ahead_after"])
        available = int(row["available_quantity"])
        if traded <= 0 or before < 0 or after < 0 or available < 0:
            raise ValueError("capacity quantity/queue fields are outside contract")
        if after != max(0, before - traded) or available != max(0, traded - before):
            raise ValueError("residual capacity arithmetic is incoherent")
        if row["source_trade_event_id"] != canonical_source_trade_event_id(row):
            raise ValueError("source_trade_event_id is not canonical")
        if row["capacity_event_id"] != canonical_capacity_event_id(row):
            raise ValueError("capacity_event_id is not canonical")
        _validate_route_market_side(row)
    source_identity = normalized.group_by("source_trade_event_id").agg(
        pl.struct(
            "Date",
            "ValueCode",
            "maker_market",
            "maker_side",
            "fill_recv_time_ns",
            "fill_event_sequence",
            "fill_row_index",
            "source_trade_quantity",
            "source_tape_sha256",
        ).n_unique().alias("identities")
    )
    if source_identity.filter(pl.col("identities") != 1).height:
        raise ValueError("raw trade identity changes across FIFO scenarios")
    return normalized


def _validate_recomputed_independent_fields(frame: pl.DataFrame) -> None:
    """Never trust caller-supplied eligibility or status columns."""

    expected = build_independent_close_opportunities(
        frame.select(*_INDEPENDENT_SCHEMA)
    )
    fields = [
        "close_opportunity_id",
        "independent_close_status",
        "independent_close_quantity",
        "fifo_capacity_eligible",
        "independent_1_to_1",
        "joint_volume_allocated",
        "pathwise_ev_ready",
        "exit_phase_version",
    ]
    actual = frame.select("exit_policy_generation_id", *fields)
    expected = expected.select(
        "exit_policy_generation_id",
        *[pl.col(field).alias(f"_expected_{field}") for field in fields],
    )
    checked = actual.join(
        expected,
        on="exit_policy_generation_id",
        how="full",
        coalesce=True,
        validate="1:1",
    )
    mismatch = pl.any_horizontal(
        [
            ~pl.col(field).eq_missing(pl.col(f"_expected_{field}"))
            for field in fields
        ]
    )
    if checked.filter(mismatch).height:
        raise ValueError("derived independent close status/eligibility was tampered")


def _collapse_policy_aliases(frame: pl.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    keys = ["scenario_id", "physical_exit_order_id", "position_id"]
    identity_columns = [
        column
        for column in _INDEPENDENT_SCHEMA
        if column != "exit_policy_generation_id"
    ] + ["fifo_capacity_eligible", "independent_close_status"]
    for key, group in frame.group_by(keys, maintain_order=True):
        rows = group.to_dicts()
        for column in identity_columns:
            if len({_hashable(row[column]) for row in rows}) != 1:
                raise ValueError(
                    "policy aliases disagree on position/order lifecycle: "
                    f"{key!r} column={column}"
                )
        row = {column: rows[0][column] for column in identity_columns}
        row["policy_alias_count"] = len(rows)
        records.append(row)
    return records


def _collapse_physical_orders(
    logical_positions: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in logical_positions:
        grouped.setdefault(_physical_key(row), []).append(row)
    records: list[dict[str, object]] = []
    physical_columns = [
        "scenario_id",
        "Date",
        "ValueCode",
        "exit_route",
        "maker_market",
        "maker_side",
        "physical_exit_order_id",
        "absolute_price_tick",
        "submit_recv_time_ns",
        "submit_event_sequence",
        "submit_row_index",
        "nominal_stop_recv_time_ns",
        "nominal_stop_event_sequence",
        "nominal_stop_row_index",
        "physical_order_quantity",
    ]
    for key, positions in grouped.items():
        for column in physical_columns:
            if len({_hashable(row[column]) for row in positions}) != 1:
                raise ValueError(
                    f"physical order disagrees across positions: {key!r} {column}"
                )
        requested = sum(int(row["requested_quantity"]) for row in positions)
        physical_quantity = int(positions[0]["physical_order_quantity"])
        if requested != physical_quantity:
            raise ValueError(
                "unique position demand does not equal physical order quantity"
            )
        established = [_cursor(row, "position_established") for row in positions]
        if len(established) != len(set(established)):
            raise ValueError("position FIFO is ambiguous at one establishment cursor")
        record = {column: positions[0][column] for column in physical_columns}
        record.update(
            {
                "position_count": len(positions),
                "policy_alias_count": sum(
                    int(row["policy_alias_count"]) for row in positions
                ),
                "fifo_capacity_eligible": all(
                    bool(row["fifo_capacity_eligible"]) for row in positions
                ),
            }
        )
        records.append(record)
    return records


def _attribute_positions_fifo(
    logical_positions: list[dict[str, object]],
    physical_records: list[dict[str, object]],
    chunks: dict[tuple[str, str], list[tuple[tuple[int, int, int], int]]],
) -> list[dict[str, object]]:
    eligible_keys = {_physical_key(row) for row in physical_records}
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in logical_positions:
        if _physical_key(row) in eligible_keys:
            grouped.setdefault(_physical_key(row), []).append(row)
    records: list[dict[str, object]] = []
    for physical_key, positions in grouped.items():
        queue = sorted(
            positions, key=lambda row: _cursor(row, "position_established")
        )
        remaining = {
            str(row["position_id"]): int(row["requested_quantity"])
            for row in queue
        }
        allocated = {position_id: 0 for position_id in remaining}
        first: dict[str, tuple[int, int, int]] = {}
        last: dict[str, tuple[int, int, int]] = {}
        for event_cursor, quantity in chunks.get(physical_key, []):
            available = quantity
            for position in queue:
                if available <= 0:
                    break
                position_id = str(position["position_id"])
                needed = remaining[position_id]
                if needed <= 0:
                    continue
                take = min(needed, available)
                allocated[position_id] += take
                remaining[position_id] -= take
                available -= take
                first.setdefault(position_id, event_cursor)
                last[position_id] = event_cursor
        for position in queue:
            position_id = str(position["position_id"])
            requested = int(position["requested_quantity"])
            quantity = allocated[position_id]
            complete = last.get(position_id) if quantity == requested else None
            independent_cursor = (
                _cursor(position, "full_fill")
                if position["full_fill"] is True
                else None
            )
            cursor_matches = complete is not None and complete == independent_cursor
            cursor_matches_ready_inputs = bool(
                cursor_matches
                and position["hedge_label_observed"]
                and position["hedge_executable"]
                and position["hedge_decision_time_ns"]
                == complete[0] + 50_000_000
            )
            records.append(
                {
                    "scenario_id": str(position["scenario_id"]),
                    "physical_exit_order_id": str(
                        position["physical_exit_order_id"]
                    ),
                    "position_id": position_id,
                    "fifo_allocated_quantity": quantity,
                    **_cursor_columns("fifo_first_fill", first.get(position_id)),
                    **_cursor_columns("fifo_full_fill", complete),
                    "fifo_full_fill_allocated": quantity == requested,
                    "fifo_cursor_matches_independent": (
                        cursor_matches_ready_inputs
                    ),
                    "fifo_close_terminal_ready": False,
                    "fifo_outcome_relabel_required": bool(
                        quantity != requested or not cursor_matches
                    ),
                    "fifo_hedge_relabel_required": bool(
                        complete is not None and not cursor_matches
                    ),
                    "capacity_provenance_verified": False,
                    "experimental_contract_only": True,
                    "actual_fifo_metrics_absent": True,
                    "mutually_exclusive_scenario": True,
                    "joint_volume_allocated": False,
                    "fifo_version": FIFO_EXIT_VERSION,
                }
            )
    return records


def _append_independent_columns(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        pl.lit(None, dtype=pl.String).alias("close_opportunity_id"),
        pl.lit(None, dtype=pl.String).alias("independent_close_status"),
        pl.lit(None, dtype=pl.Int64).alias("independent_close_quantity"),
        pl.lit(None, dtype=pl.Boolean).alias("fifo_capacity_eligible"),
        pl.lit(True).alias("independent_1_to_1"),
        pl.lit(False).alias("joint_volume_allocated"),
        pl.lit(False).alias("pathwise_ev_ready"),
        pl.lit(INDEPENDENT_EXIT_VERSION).alias("exit_phase_version"),
    )


def _project_unallocated_aliases(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.drop("joint_volume_allocated").with_columns(
        pl.lit(0, dtype=pl.Int64).alias("fifo_allocated_quantity"),
        pl.lit(None, dtype=pl.Int64).alias("fifo_first_fill_recv_time_ns"),
        pl.lit(None, dtype=pl.Int64).alias("fifo_first_fill_event_sequence"),
        pl.lit(None, dtype=pl.Int64).alias("fifo_first_fill_row_index"),
        pl.lit(None, dtype=pl.Int64).alias("fifo_full_fill_recv_time_ns"),
        pl.lit(None, dtype=pl.Int64).alias("fifo_full_fill_event_sequence"),
        pl.lit(None, dtype=pl.Int64).alias("fifo_full_fill_row_index"),
        pl.lit(False).alias("fifo_full_fill_allocated"),
        pl.lit(False).alias("fifo_close_terminal_ready"),
        pl.lit(False).alias("fifo_cursor_matches_independent"),
        pl.lit(False).alias("fifo_outcome_relabel_required"),
        pl.lit(False).alias("fifo_hedge_relabel_required"),
        pl.lit(False).alias("joint_volume_allocated"),
        pl.lit(False).alias("capacity_provenance_verified"),
        pl.lit(True).alias("experimental_contract_only"),
        pl.lit(True).alias("actual_fifo_metrics_absent"),
        pl.lit(True).alias("mutually_exclusive_scenario"),
        pl.lit(FIFO_EXIT_VERSION).alias("fifo_version"),
    )


def _empty_alias_projection(frame: pl.DataFrame) -> pl.DataFrame:
    if not set(_INDEPENDENT_SCHEMA).issubset(frame.columns):
        frame = _append_independent_columns(
            _normalize_empty(frame, _INDEPENDENT_SCHEMA)
        )
    return _project_unallocated_aliases(frame)


def _empty_physical_allocations() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "scenario_id": pl.String,
            "physical_exit_order_id": pl.String,
            "Date": pl.String,
            "ValueCode": pl.String,
            "exit_route": pl.String,
            "maker_market": pl.String,
            "maker_side": pl.String,
            "absolute_price_tick": pl.Int64,
            "submit_recv_time_ns": pl.Int64,
            "submit_event_sequence": pl.Int64,
            "submit_row_index": pl.Int64,
            "nominal_stop_recv_time_ns": pl.Int64,
            "nominal_stop_event_sequence": pl.Int64,
            "nominal_stop_row_index": pl.Int64,
            "physical_order_quantity": pl.Int64,
            "position_count": pl.Int64,
            "policy_alias_count": pl.Int64,
            "fifo_capacity_eligible": pl.Boolean,
            "fifo_allocated_quantity": pl.Int64,
            "fifo_remaining_quantity": pl.Int64,
            "fifo_first_fill_recv_time_ns": pl.Int64,
            "fifo_first_fill_event_sequence": pl.Int64,
            "fifo_first_fill_row_index": pl.Int64,
            "fifo_full_fill_recv_time_ns": pl.Int64,
            "fifo_full_fill_event_sequence": pl.Int64,
            "fifo_full_fill_row_index": pl.Int64,
            "fifo_full_fill_allocated": pl.Boolean,
            "fifo_cursor_matches_independent": pl.Boolean,
            "fifo_close_terminal_ready": pl.Boolean,
            "fifo_outcome_relabel_required": pl.Boolean,
            "fifo_hedge_relabel_required": pl.Boolean,
            "scenario_local_unique_claim_events": pl.Boolean,
            "capacity_basis": pl.String,
            "capacity_provenance_verified": pl.Boolean,
            "experimental_contract_only": pl.Boolean,
            "actual_fifo_metrics_absent": pl.Boolean,
            "mutually_exclusive_scenario": pl.Boolean,
            "exchange_fifo_exact": pl.Boolean,
            "position_fifo_attribution_exact": pl.Boolean,
            "joint_volume_allocated": pl.Boolean,
            "pathwise_ev_ready": pl.Boolean,
            "fifo_version": pl.String,
        }
    )


def _validate_route_market_side(row: Mapping[str, object]) -> None:
    expected = {
        "future_ask_spot_taker": ("future", "ask"),
        "spot_bid_future_taker": ("spot", "bid"),
        "spot_ask_future_taker": ("spot", "ask"),
        "future_bid_spot_taker": ("future", "bid"),
    }.get(str(row["exit_route"]))
    if expected is None or expected != (
        str(row["maker_market"]),
        str(row["maker_side"]),
    ):
        raise ValueError("capacity route/market/side identity is incoherent")


def _capacity_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        str(row["scenario_id"]),
        str(row["Date"]),
        str(row["ValueCode"]),
        str(row["exit_route"]),
        str(row["maker_market"]),
        str(row["maker_side"]),
        int(row["absolute_price_tick"]),
    )


def _physical_key(row: Mapping[str, object]) -> tuple[str, str]:
    return str(row["scenario_id"]), str(row["physical_exit_order_id"])


def _cursor(row: Mapping[str, object], prefix: str) -> tuple[int, int, int]:
    return tuple(  # type: ignore[return-value]
        int(row[f"{prefix}_{suffix}"]) for suffix in _CURSOR_SUFFIXES
    )


def _cursor_columns(
    prefix: str, cursor: tuple[int, int, int] | None
) -> dict[str, int | None]:
    return {
        f"{prefix}_{suffix}": cursor[index] if cursor is not None else None
        for index, suffix in enumerate(_CURSOR_SUFFIXES)
    }


def _normalize_empty(
    frame: pl.DataFrame, schema: Mapping[str, pl.DataType]
) -> pl.DataFrame:
    if not frame.is_empty():
        raise ValueError("empty normalization received nonempty input")
    return frame.with_columns(
        *[
            (
                pl.col(column).cast(dtype, strict=True)
                if column in frame.columns
                else pl.lit(None, dtype=dtype).alias(column)
            )
            for column, dtype in schema.items()
        ]
    ).select(*schema)


def _hashable(value: object) -> object:
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _close_id(row: Mapping[str, object]) -> str:
    identity = {
        "scenario_id": str(row["scenario_id"]),
        "Date": str(row["Date"]),
        "ValueCode": str(row["ValueCode"]),
        "position_id": str(row["position_id"]),
        "exit_policy_generation_id": str(row["exit_policy_generation_id"]),
        "physical_exit_order_id": str(row["physical_exit_order_id"]),
    }
    return _canonical_sha256(identity)[:20]


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require(
    frame: pl.DataFrame, required: Iterable[str], source: str
) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")
