"""Frozen cost-aware scenario specifications for the rebuilt S1 replay.

This module is deliberately independent of the execution-time economic gate.
It freezes only causal lookup primitives and the seven pre-registered scenario
settings.  The production replay may consume this table later, but building
the table neither prices an order nor changes scheduler/capacity state.

Unsupported lookup cells remain explicit rows.  They are fail-closed through
``lookup_supported=false`` and a stable reason; they are never removed by an
inner join and never substituted with a different lower candidate.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Literal

import polars as pl

from .policy_spec import (
    ANCHOR_MODEL_ID,
    CONVERGENCE_REFERENCE_SEMANTICS,
    DEFAULT_CONVERGENCE_PATH,
    DEFAULT_ENTRY_LOOKUP_PATH,
    DEFAULT_MOTHER_PATH,
    ENTRY_CANDIDATE_ID,
    TOD_BUCKETS,
)

SCENARIO_SPEC_VERSION: Final = "s1_cost_aware_scenario_spec_v1"
FIXED_DISTANCE_BP: Final = 20.0
LOWER_C0: Final = "C0_center"
LOWER_C2: Final = "C2_conditional_reach80"
LOWER_C3: Final = "C3_conditional_reach50"

PolicyKind = Literal["quantile", "fixed"]
CostHorizon = Literal["ungated", "same_day", "overnight"]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _validate_definition(definition: S1ScenarioDefinition) -> None:
    for name in ("scenario_id", "entry_policy_id", "policy_kind", "lower_candidate_id"):
        value = getattr(definition, name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
    if definition.policy_kind not in ("quantile", "fixed"):
        raise ValueError("unsupported policy_kind")
    if definition.cost_horizon not in ("ungated", "same_day", "overnight"):
        raise ValueError("unsupported cost_horizon")
    if definition.policy_kind == "quantile":
        if definition.boundary_quantile not in (50, 80, 95):
            raise ValueError("quantile scenario requires q50/q80/q95")
        if definition.entry_policy_id != f"q{definition.boundary_quantile}":
            raise ValueError("entry policy ID and quantile disagree")
        if definition.lower_candidate_id not in (LOWER_C0, LOWER_C2, LOWER_C3):
            raise ValueError("unsupported conditional lower candidate")
        if definition.fixed_distance_bp is not None:
            raise ValueError("quantile scenario cannot carry fixed distance")
    else:
        if definition.entry_policy_id != "fixed20":
            raise ValueError("only fixed20 is frozen in the scenario grid")
        if definition.boundary_quantile is not None:
            raise ValueError("fixed scenario cannot carry a quantile")
        if definition.fixed_distance_bp is None or not math.isclose(
            definition.fixed_distance_bp, FIXED_DISTANCE_BP, abs_tol=1e-12
        ):
            raise ValueError("fixed scenario must use symmetric 20 bp")
    if definition.cost_horizon == "ungated":
        if definition.safety_floor_bp is not None:
            raise ValueError("ungated control cannot carry a safety floor")
        if definition.economic_gate_enabled:
            raise ValueError("ungated control cannot enable the economic gate")
        if definition.deployment_shortlist_eligible:
            raise ValueError("ungated control cannot enter deployment shortlist")
    else:
        value = definition.safety_floor_bp
        if value is None or not math.isfinite(value) or value < 0:
            raise ValueError("gated scenario requires a non-negative safety floor")
        if not definition.economic_gate_enabled:
            raise ValueError("gated scenario must enable the economic gate")


@dataclass(frozen=True, slots=True)
class S1ScenarioDefinition:
    """One immutable member of the pre-registered seven-scenario grid."""

    scenario_id: str
    entry_policy_id: str
    policy_kind: PolicyKind
    boundary_quantile: int | None
    lower_candidate_id: str
    fixed_distance_bp: float | None
    cost_horizon: CostHorizon
    safety_floor_bp: float | None
    economic_gate_enabled: bool
    deployment_shortlist_eligible: bool

    def __post_init__(self) -> None:
        _validate_definition(self)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def deterministic_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


SCENARIO_DEFINITIONS: Final = (
    S1ScenarioDefinition(
        scenario_id="ctrl_q95_C0_ungated",
        entry_policy_id="q95",
        policy_kind="quantile",
        boundary_quantile=95,
        lower_candidate_id=LOWER_C0,
        fixed_distance_bp=None,
        cost_horizon="ungated",
        safety_floor_bp=None,
        economic_gate_enabled=False,
        deployment_shortlist_eligible=False,
    ),
    S1ScenarioDefinition(
        scenario_id="q95_C0_sd_f5",
        entry_policy_id="q95",
        policy_kind="quantile",
        boundary_quantile=95,
        lower_candidate_id=LOWER_C0,
        fixed_distance_bp=None,
        cost_horizon="same_day",
        safety_floor_bp=5.0,
        economic_gate_enabled=True,
        deployment_shortlist_eligible=True,
    ),
    S1ScenarioDefinition(
        scenario_id="q95_C0_on_f0",
        entry_policy_id="q95",
        policy_kind="quantile",
        boundary_quantile=95,
        lower_candidate_id=LOWER_C0,
        fixed_distance_bp=None,
        cost_horizon="overnight",
        safety_floor_bp=0.0,
        economic_gate_enabled=True,
        deployment_shortlist_eligible=True,
    ),
    S1ScenarioDefinition(
        scenario_id="q95_C2_sd_f5",
        entry_policy_id="q95",
        policy_kind="quantile",
        boundary_quantile=95,
        lower_candidate_id=LOWER_C2,
        fixed_distance_bp=None,
        cost_horizon="same_day",
        safety_floor_bp=5.0,
        economic_gate_enabled=True,
        deployment_shortlist_eligible=True,
    ),
    S1ScenarioDefinition(
        scenario_id="q80_C0_sd_f5",
        entry_policy_id="q80",
        policy_kind="quantile",
        boundary_quantile=80,
        lower_candidate_id=LOWER_C0,
        fixed_distance_bp=None,
        cost_horizon="same_day",
        safety_floor_bp=5.0,
        economic_gate_enabled=True,
        deployment_shortlist_eligible=True,
    ),
    S1ScenarioDefinition(
        scenario_id="q50_C3_sd_f5",
        entry_policy_id="q50",
        policy_kind="quantile",
        boundary_quantile=50,
        lower_candidate_id=LOWER_C3,
        fixed_distance_bp=None,
        cost_horizon="same_day",
        safety_floor_bp=5.0,
        economic_gate_enabled=True,
        deployment_shortlist_eligible=True,
    ),
    S1ScenarioDefinition(
        scenario_id="fixed20_sym20_on_f0",
        entry_policy_id="fixed20",
        policy_kind="fixed",
        boundary_quantile=None,
        lower_candidate_id="constant_bp:20",
        fixed_distance_bp=FIXED_DISTANCE_BP,
        cost_horizon="overnight",
        safety_floor_bp=0.0,
        economic_gate_enabled=True,
        deployment_shortlist_eligible=True,
    ),
)

SCENARIO_IDS: Final = tuple(value.scenario_id for value in SCENARIO_DEFINITIONS)
if len(SCENARIO_IDS) != 7 or len(set(SCENARIO_IDS)) != len(SCENARIO_IDS):
    raise RuntimeError("S1 scenario grid must contain seven unique scenario IDs")
SCENARIO_BY_ID: Final = {value.scenario_id: value for value in SCENARIO_DEFINITIONS}
SCENARIO_GRID_SHA256: Final = _canonical_sha256(
    [value.to_dict() for value in SCENARIO_DEFINITIONS]
)


@dataclass(frozen=True, slots=True)
class S1ScenarioSpec:
    """One scenario row for one primary product-day/TOD causal cell."""

    Date: str
    ValueCode: str
    QuoteCode: str
    entry_tod_bucket: str
    scenario_id: str
    entry_policy_id: str
    policy_kind: str
    anchor_model_id: str
    boundary_quantile: int | None
    upper_distance_bp: float | None
    lower_distance_bp: float | None
    upper_source_id: str
    lower_source_id: str
    upper_effective_source_id: str | None
    lower_effective_source_id: str | None
    upper_source_asof_date: str | None
    lower_source_asof_date: str | None
    combined_source_asof_date: str | None
    upper_lookup_supported: bool
    lower_lookup_supported: bool
    lookup_supported: bool
    upper_support_reason: str
    lower_support_reason: str
    lookup_support_reason: str
    upper_fallback_used: bool
    lower_fallback_used: bool
    upper_fallback_reason: str | None
    lower_fallback_reason: str | None
    lower_native_failure_reason: str | None
    cost_horizon: str
    safety_floor_bp: float | None
    economic_gate_enabled: bool
    deployment_shortlist_eligible: bool
    contains_target_day_outcome: bool
    scenario_definition_sha256: str
    scenario_grid_sha256: str
    spec_version: str

    def __post_init__(self) -> None:
        _validate_scenario_spec(self)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> S1ScenarioSpec:
        missing = [column for column in S1_SCENARIO_SPEC_SCHEMA if column not in value]
        extra = sorted(set(value) - set(S1_SCENARIO_SPEC_SCHEMA))
        if missing or extra:
            raise ValueError(
                "S1 scenario mapping mismatch; "
                f"missing={missing}, extra={extra}"
            )
        return cls(**{column: value[column] for column in S1_SCENARIO_SPEC_SCHEMA})

    @property
    def deterministic_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    @property
    def policy_id(self) -> str:
        """Execution-compatible policy identity; scenarios remain distinct."""

        return self.scenario_id

    @property
    def kind(self) -> str:
        """Execution-compatible alias for ``policy_kind``."""

        return self.policy_kind

    @property
    def fallback_reason(self) -> str | None:
        """Expose the most specific causal lookup fallback to target builders."""

        if self.lower_fallback_reason is not None:
            return self.lower_fallback_reason
        if self.upper_fallback_reason is not None:
            return self.upper_fallback_reason
        if self.lookup_support_reason != "supported":
            return self.lookup_support_reason
        return None


S1_SCENARIO_SPEC_SCHEMA: Final[dict[str, pl.DataType]] = {
    "Date": pl.String,
    "ValueCode": pl.String,
    "QuoteCode": pl.String,
    "entry_tod_bucket": pl.String,
    "scenario_id": pl.String,
    "entry_policy_id": pl.String,
    "policy_kind": pl.String,
    "anchor_model_id": pl.String,
    "boundary_quantile": pl.Int64,
    "upper_distance_bp": pl.Float64,
    "lower_distance_bp": pl.Float64,
    "upper_source_id": pl.String,
    "lower_source_id": pl.String,
    "upper_effective_source_id": pl.String,
    "lower_effective_source_id": pl.String,
    "upper_source_asof_date": pl.String,
    "lower_source_asof_date": pl.String,
    "combined_source_asof_date": pl.String,
    "upper_lookup_supported": pl.Boolean,
    "lower_lookup_supported": pl.Boolean,
    "lookup_supported": pl.Boolean,
    "upper_support_reason": pl.String,
    "lower_support_reason": pl.String,
    "lookup_support_reason": pl.String,
    "upper_fallback_used": pl.Boolean,
    "lower_fallback_used": pl.Boolean,
    "upper_fallback_reason": pl.String,
    "lower_fallback_reason": pl.String,
    "lower_native_failure_reason": pl.String,
    "cost_horizon": pl.String,
    "safety_floor_bp": pl.Float64,
    "economic_gate_enabled": pl.Boolean,
    "deployment_shortlist_eligible": pl.Boolean,
    "contains_target_day_outcome": pl.Boolean,
    "scenario_definition_sha256": pl.String,
    "scenario_grid_sha256": pl.String,
    "spec_version": pl.String,
}

CELL_KEYS: Final = (
    "Date",
    "ValueCode",
    "QuoteCode",
    "entry_tod_bucket",
)
ROW_KEYS: Final = (*CELL_KEYS, "scenario_id")


def build_s1_scenario_spec_table(
    mother: pl.DataFrame,
    entry_lookup: pl.DataFrame,
    convergence_predictions: pl.DataFrame,
) -> pl.DataFrame:
    """Build the full mother x four-TOD x seven-scenario table."""

    _require_columns(
        mother,
        {
            "Date",
            "ValueCode",
            "QuoteCode",
            "s1_primary",
            "anchor_model_id",
            "contains_target_day_outcome",
        },
        "S1 mother",
    )
    _require_columns(
        entry_lookup,
        {
            "Date",
            "ValueCode",
            "QuoteCode",
            "anchor_model_id",
            "tod_bucket",
            "boundary_quantile",
            "side",
            "candidate_id",
            "effective_supported",
            "boundary_distance_bp",
            "effective_candidate_id",
            "effective_source_asof_date",
            "fallback_used",
            "fallback_reason",
            "contains_target_day_outcome",
        },
        "entry lookup",
    )
    _require_columns(
        convergence_predictions,
        {
            "Date",
            "ValueCode",
            "QuoteCode",
            "anchor_model_id",
            "candidate_id",
            "tod_bucket",
            "boundary_quantile",
            "convergence_candidate_id",
            "convergence_reference_semantics",
            "effective_threshold_distance_bp",
            "effective_supported",
            "effective_lookup_id",
            "effective_source_asof_date",
            "fallback_used",
            "fallback_reason",
            "native_failure_reason",
            "contains_target_day_outcome",
        },
        "frozen convergence predictions",
    )

    primary = mother.filter(pl.col("s1_primary").fill_null(False)).select(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("anchor_model_id").cast(pl.String),
        pl.col("contains_target_day_outcome")
        .cast(pl.Boolean)
        .alias("mother_contains_target_day_outcome"),
    )
    if primary.is_empty():
        raise ValueError("S1 primary mother is empty")
    _assert_unique(primary, ["Date", "ValueCode", "QuoteCode"], "primary mother")
    if primary.filter(
        pl.col("mother_contains_target_day_outcome").fill_null(True)
    ).height:
        raise ValueError("S1 mother contains target-day outcomes")
    if primary.filter(pl.col("anchor_model_id") != ANCHOR_MODEL_ID).height:
        raise ValueError("S1 mother does not use the frozen anchor model")
    base = primary.join(
        pl.DataFrame({"entry_tod_bucket": TOD_BUCKETS}),
        how="cross",
    )

    needed_quantiles = sorted(
        {
            value.boundary_quantile
            for value in SCENARIO_DEFINITIONS
            if value.boundary_quantile is not None
        }
    )
    upper = entry_lookup.filter(
        (pl.col("candidate_id") == ENTRY_CANDIDATE_ID)
        & (pl.col("side") == "positive")
        & pl.col("boundary_quantile").is_in(needed_quantiles)
    ).select(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("anchor_model_id").cast(pl.String),
        pl.col("tod_bucket").cast(pl.String).alias("entry_tod_bucket"),
        pl.col("boundary_quantile").cast(pl.Int64),
        pl.col("boundary_distance_bp").cast(pl.Float64).alias("upper_distance_bp"),
        pl.col("effective_candidate_id")
        .cast(pl.String)
        .alias("upper_effective_source_id"),
        pl.col("effective_source_asof_date")
        .cast(pl.String)
        .alias("upper_source_asof_date"),
        pl.col("effective_supported")
        .cast(pl.Boolean)
        .alias("upper_effective_supported"),
        pl.col("fallback_used").cast(pl.Boolean).alias("upper_fallback_used"),
        pl.col("fallback_reason").cast(pl.String).alias("upper_fallback_reason"),
        pl.col("contains_target_day_outcome")
        .cast(pl.Boolean)
        .alias("upper_contains_target_day_outcome"),
        pl.lit(True).alias("upper_row_present"),
    )
    lower_candidates = sorted(
        {
            value.lower_candidate_id
            for value in SCENARIO_DEFINITIONS
            if value.policy_kind == "quantile"
        }
    )
    lower = convergence_predictions.filter(
        (pl.col("candidate_id") == ENTRY_CANDIDATE_ID)
        & pl.col("convergence_candidate_id").is_in(lower_candidates)
        & pl.col("boundary_quantile").is_in(needed_quantiles)
    ).select(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("anchor_model_id").cast(pl.String),
        pl.col("tod_bucket").cast(pl.String).alias("entry_tod_bucket"),
        pl.col("boundary_quantile").cast(pl.Int64),
        pl.col("convergence_candidate_id").cast(pl.String).alias("lower_source_id"),
        pl.col("convergence_reference_semantics").cast(pl.String),
        pl.col("effective_threshold_distance_bp")
        .cast(pl.Float64)
        .alias("lower_distance_bp"),
        pl.col("effective_lookup_id")
        .cast(pl.String)
        .alias("lower_effective_source_id"),
        pl.col("effective_source_asof_date")
        .cast(pl.String)
        .alias("lower_source_asof_date"),
        pl.col("effective_supported")
        .cast(pl.Boolean)
        .alias("lower_effective_supported"),
        pl.col("fallback_used").cast(pl.Boolean).alias("lower_fallback_used"),
        pl.col("fallback_reason").cast(pl.String).alias("lower_fallback_reason"),
        pl.col("native_failure_reason")
        .cast(pl.String)
        .alias("lower_native_failure_reason"),
        pl.col("contains_target_day_outcome")
        .cast(pl.Boolean)
        .alias("lower_contains_target_day_outcome"),
        pl.lit(True).alias("lower_row_present"),
    )
    lookup_keys = [*CELL_KEYS, "anchor_model_id", "boundary_quantile"]
    _assert_unique(upper, lookup_keys, "positive Q2 entry lookup")
    _assert_unique(lower, [*lookup_keys, "lower_source_id"], "frozen lower lookup")
    _validate_source_provenance(upper, lower)

    parts: list[pl.DataFrame] = []
    for definition in SCENARIO_DEFINITIONS:
        if definition.policy_kind == "fixed":
            part = _build_fixed_part(base, definition)
        else:
            part = _build_quantile_part(base, upper, lower, definition)
        parts.append(part.select(*S1_SCENARIO_SPEC_SCHEMA))
    result = pl.concat(parts, how="vertical_relaxed").sort(ROW_KEYS)
    validate_s1_scenario_spec_table(result)
    return result


def load_s1_scenario_spec_table(
    mother_path: Path = DEFAULT_MOTHER_PATH,
    entry_lookup_path: Path = DEFAULT_ENTRY_LOOKUP_PATH,
    convergence_path: Path = DEFAULT_CONVERGENCE_PATH,
) -> pl.DataFrame:
    """Read the canonical inputs and build their validated scenario table."""

    return build_s1_scenario_spec_table(
        pl.read_parquet(mother_path),
        pl.read_parquet(entry_lookup_path),
        pl.read_parquet(convergence_path),
    )


def validate_s1_scenario_spec_table(
    frame: pl.DataFrame,
    *,
    expected_sha256: str | None = None,
) -> None:
    """Validate schema, coverage, scenario constants, provenance, and hash."""

    if frame.schema != pl.Schema(S1_SCENARIO_SPEC_SCHEMA):
        raise ValueError(
            "S1 scenario table schema mismatch: "
            f"expected={pl.Schema(S1_SCENARIO_SPEC_SCHEMA)}, actual={frame.schema}"
        )
    if frame.is_empty():
        raise ValueError("S1 scenario table is empty")
    _assert_unique(frame, list(ROW_KEYS), "S1 scenario specs")
    invalid_cells = (
        frame.group_by(CELL_KEYS)
        .agg(
            pl.len().alias("rows"),
            pl.col("scenario_id").n_unique().alias("scenarios"),
            pl.col("scenario_id").sort().alias("scenario_ids"),
        )
        .filter(
            (pl.col("rows") != len(SCENARIO_IDS))
            | (pl.col("scenarios") != len(SCENARIO_IDS))
            | (pl.col("scenario_ids") != sorted(SCENARIO_IDS))
        )
    )
    if not invalid_cells.is_empty():
        raise ValueError("each causal cell must contain the exact seven-scenario grid")
    invalid_tod = (
        frame.select("Date", "ValueCode", "QuoteCode", "entry_tod_bucket")
        .unique()
        .group_by("Date", "ValueCode", "QuoteCode")
        .agg(
            pl.len().alias("rows"),
            pl.col("entry_tod_bucket").sort().alias("tod_values"),
        )
        .filter(
            (pl.col("rows") != len(TOD_BUCKETS))
            | (pl.col("tod_values") != sorted(TOD_BUCKETS))
        )
    )
    if not invalid_tod.is_empty():
        raise ValueError("each primary product-day must contain the exact four TOD cells")

    invalid_general = frame.filter(
        (pl.col("anchor_model_id") != ANCHOR_MODEL_ID)
        | pl.col("contains_target_day_outcome")
        | (pl.col("spec_version") != SCENARIO_SPEC_VERSION)
        | (pl.col("scenario_grid_sha256") != SCENARIO_GRID_SHA256)
        | (
            pl.col("lookup_supported")
            != (pl.col("upper_lookup_supported") & pl.col("lower_lookup_supported"))
        )
        | (
            pl.col("lookup_supported")
            & (
                pl.col("upper_distance_bp").is_null()
                | pl.col("lower_distance_bp").is_null()
            )
        )
        | (
            pl.col("lookup_supported")
            & (pl.col("lookup_support_reason") != "supported")
        )
        | (
            ~pl.col("lookup_supported")
            & (pl.col("lookup_support_reason") == "supported")
        )
    )
    if not invalid_general.is_empty():
        raise ValueError("S1 scenario table has invalid support or global provenance")

    for column in (
        "upper_source_asof_date",
        "lower_source_asof_date",
        "combined_source_asof_date",
    ):
        if frame.filter(pl.col(column).is_not_null() & (pl.col(column) >= pl.col("Date"))).height:
            raise ValueError(f"{column} must be strictly earlier than Date")
    expected_combined = pl.max_horizontal(
        "upper_source_asof_date", "lower_source_asof_date"
    )
    if frame.filter(
        pl.col("combined_source_asof_date").fill_null("")
        != expected_combined.fill_null("")
    ).height:
        raise ValueError("combined source as-of date is not the latest available input")

    for definition in SCENARIO_DEFINITIONS:
        rows = frame.filter(pl.col("scenario_id") == definition.scenario_id)
        if rows.is_empty():
            raise ValueError(f"scenario is missing: {definition.scenario_id}")
        expected_values = {
            "entry_policy_id": definition.entry_policy_id,
            "policy_kind": definition.policy_kind,
            "boundary_quantile": definition.boundary_quantile,
            "cost_horizon": definition.cost_horizon,
            "safety_floor_bp": definition.safety_floor_bp,
            "economic_gate_enabled": definition.economic_gate_enabled,
            "deployment_shortlist_eligible": (
                definition.deployment_shortlist_eligible
            ),
            "scenario_definition_sha256": definition.deterministic_sha256,
        }
        for column, expected in expected_values.items():
            actual = rows[column]
            if expected is None:
                mismatch = actual.is_not_null()
            elif column == "safety_floor_bp":
                mismatch = actual.is_null() | ~actual.is_close(
                    float(expected),
                    abs_tol=1e-12,
                    rel_tol=0.0,
                )
            else:
                mismatch = actual.is_null() | (actual != expected)
            if mismatch.any():
                raise ValueError(
                    f"scenario definition mismatch: {definition.scenario_id}/{column}"
                )
        _validate_scenario_rows(rows, definition)

    if expected_sha256 is not None:
        if not isinstance(expected_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_sha256
        ):
            raise ValueError("expected_sha256 must be a lowercase SHA-256")
        actual = _table_sha256_unchecked(frame)
        if actual != expected_sha256:
            raise ValueError(
                f"S1 scenario table SHA-256 mismatch: expected={expected_sha256}, "
                f"actual={actual}"
            )


def s1_scenario_specs_from_table(
    frame: pl.DataFrame,
) -> tuple[S1ScenarioSpec, ...]:
    validate_s1_scenario_spec_table(frame)
    return tuple(S1ScenarioSpec.from_dict(row) for row in frame.iter_rows(named=True))


def s1_scenario_specs_to_table(
    specs: Sequence[S1ScenarioSpec] | Iterable[S1ScenarioSpec],
) -> pl.DataFrame:
    values = tuple(specs)
    if not values:
        raise ValueError("S1 scenario specs cannot be empty")
    frame = pl.from_dicts(
        [value.to_dict() for value in values],
        schema=S1_SCENARIO_SPEC_SCHEMA,
    ).sort(ROW_KEYS)
    validate_s1_scenario_spec_table(frame)
    return frame


def s1_scenario_spec_table_sha256(frame: pl.DataFrame) -> str:
    """Hash canonical rows independent of their incoming order."""

    validate_s1_scenario_spec_table(frame)
    return _table_sha256_unchecked(frame)


def _build_quantile_part(
    base: pl.DataFrame,
    upper: pl.DataFrame,
    lower: pl.DataFrame,
    definition: S1ScenarioDefinition,
) -> pl.DataFrame:
    assert definition.boundary_quantile is not None
    expected = base.with_columns(
        pl.lit(definition.boundary_quantile)
        .cast(pl.Int64)
        .alias("boundary_quantile")
    )
    keys = [*CELL_KEYS, "anchor_model_id", "boundary_quantile"]
    rows = expected.join(upper, on=keys, how="left", validate="1:1").join(
        lower.filter(pl.col("lower_source_id") == definition.lower_candidate_id),
        on=keys,
        how="left",
        validate="1:1",
    )
    upper_present = pl.col("upper_row_present").fill_null(False)
    lower_present = pl.col("lower_row_present").fill_null(False)
    upper_supported = (
        upper_present
        & pl.col("upper_effective_supported").fill_null(False)
        & pl.col("upper_distance_bp").is_not_null()
        & pl.col("upper_source_asof_date").is_not_null()
    )
    lower_supported = (
        lower_present
        & pl.col("lower_effective_supported").fill_null(False)
        & pl.col("lower_distance_bp").is_not_null()
        & pl.col("lower_source_asof_date").is_not_null()
    )
    rows = rows.with_columns(
        upper_supported.alias("upper_lookup_supported"),
        lower_supported.alias("lower_lookup_supported"),
        pl.when(~upper_present)
        .then(pl.lit("missing"))
        .when(~upper_supported)
        .then(pl.lit("unsupported"))
        .otherwise(pl.lit("supported"))
        .alias("upper_support_reason"),
        pl.when(~lower_present)
        .then(pl.lit("missing"))
        .when(~lower_supported)
        .then(pl.lit("unsupported"))
        .otherwise(pl.lit("supported"))
        .alias("lower_support_reason"),
    ).with_columns(
        (
            pl.col("upper_lookup_supported") & pl.col("lower_lookup_supported")
        ).alias("lookup_supported"),
        pl.when(~pl.col("upper_lookup_supported"))
        .then(pl.concat_str(pl.lit("upper_"), pl.col("upper_support_reason")))
        .when(~pl.col("lower_lookup_supported"))
        .then(pl.concat_str(pl.lit("lower_"), pl.col("lower_support_reason")))
        .otherwise(pl.lit("supported"))
        .alias("lookup_support_reason"),
        pl.max_horizontal(
            "upper_source_asof_date", "lower_source_asof_date"
        ).alias("combined_source_asof_date"),
        (
            pl.col("mother_contains_target_day_outcome")
            | pl.col("upper_contains_target_day_outcome").fill_null(False)
            | pl.col("lower_contains_target_day_outcome").fill_null(False)
        ).alias("contains_target_day_outcome"),
    )
    return rows.with_columns(
        pl.lit(definition.scenario_id).alias("scenario_id"),
        pl.lit(definition.entry_policy_id).alias("entry_policy_id"),
        pl.lit(definition.policy_kind).alias("policy_kind"),
        pl.lit(ENTRY_CANDIDATE_ID).alias("upper_source_id"),
        pl.lit(definition.lower_candidate_id).alias("lower_source_id"),
        pl.col("upper_fallback_used").fill_null(False),
        pl.col("lower_fallback_used").fill_null(False),
        pl.lit(definition.cost_horizon).alias("cost_horizon"),
        pl.lit(definition.safety_floor_bp, dtype=pl.Float64).alias(
            "safety_floor_bp"
        ),
        pl.lit(definition.economic_gate_enabled).alias("economic_gate_enabled"),
        pl.lit(definition.deployment_shortlist_eligible).alias(
            "deployment_shortlist_eligible"
        ),
        pl.lit(definition.deterministic_sha256).alias(
            "scenario_definition_sha256"
        ),
        pl.lit(SCENARIO_GRID_SHA256).alias("scenario_grid_sha256"),
        pl.lit(SCENARIO_SPEC_VERSION).alias("spec_version"),
    )


def _build_fixed_part(
    base: pl.DataFrame,
    definition: S1ScenarioDefinition,
) -> pl.DataFrame:
    assert definition.fixed_distance_bp is not None
    source = f"constant_bp:{int(definition.fixed_distance_bp)}"
    return base.with_columns(
        pl.lit(definition.scenario_id).alias("scenario_id"),
        pl.lit(definition.entry_policy_id).alias("entry_policy_id"),
        pl.lit(definition.policy_kind).alias("policy_kind"),
        pl.lit(None, dtype=pl.Int64).alias("boundary_quantile"),
        pl.lit(definition.fixed_distance_bp)
        .cast(pl.Float64)
        .alias("upper_distance_bp"),
        pl.lit(definition.fixed_distance_bp)
        .cast(pl.Float64)
        .alias("lower_distance_bp"),
        pl.lit(source).alias("upper_source_id"),
        pl.lit(source).alias("lower_source_id"),
        pl.lit(source).alias("upper_effective_source_id"),
        pl.lit(source).alias("lower_effective_source_id"),
        pl.lit(None, dtype=pl.String).alias("upper_source_asof_date"),
        pl.lit(None, dtype=pl.String).alias("lower_source_asof_date"),
        pl.lit(None, dtype=pl.String).alias("combined_source_asof_date"),
        pl.lit(True).alias("upper_lookup_supported"),
        pl.lit(True).alias("lower_lookup_supported"),
        pl.lit(True).alias("lookup_supported"),
        pl.lit("constant_supported").alias("upper_support_reason"),
        pl.lit("constant_supported").alias("lower_support_reason"),
        pl.lit("supported").alias("lookup_support_reason"),
        pl.lit(False).alias("upper_fallback_used"),
        pl.lit(False).alias("lower_fallback_used"),
        pl.lit(None, dtype=pl.String).alias("upper_fallback_reason"),
        pl.lit(None, dtype=pl.String).alias("lower_fallback_reason"),
        pl.lit(None, dtype=pl.String).alias("lower_native_failure_reason"),
        pl.lit(definition.cost_horizon).alias("cost_horizon"),
        pl.lit(definition.safety_floor_bp, dtype=pl.Float64).alias(
            "safety_floor_bp"
        ),
        pl.lit(definition.economic_gate_enabled).alias("economic_gate_enabled"),
        pl.lit(definition.deployment_shortlist_eligible).alias(
            "deployment_shortlist_eligible"
        ),
        pl.col("mother_contains_target_day_outcome").alias(
            "contains_target_day_outcome"
        ),
        pl.lit(definition.deterministic_sha256).alias(
            "scenario_definition_sha256"
        ),
        pl.lit(SCENARIO_GRID_SHA256).alias("scenario_grid_sha256"),
        pl.lit(SCENARIO_SPEC_VERSION).alias("spec_version"),
    )


def _validate_source_provenance(upper: pl.DataFrame, lower: pl.DataFrame) -> None:
    if upper.filter(pl.col("upper_contains_target_day_outcome").fill_null(True)).height:
        raise ValueError("positive Q2 lookup contains target-day outcomes")
    if lower.filter(pl.col("lower_contains_target_day_outcome").fill_null(True)).height:
        raise ValueError("frozen lower lookup contains target-day outcomes")
    if upper.filter(
        pl.col("upper_source_asof_date").is_not_null()
        & (pl.col("upper_source_asof_date") >= pl.col("Date"))
    ).height:
        raise ValueError("upper source as-of must be strictly earlier than Date")
    if lower.filter(
        pl.col("lower_source_asof_date").is_not_null()
        & (pl.col("lower_source_asof_date") >= pl.col("Date"))
    ).height:
        raise ValueError("lower source as-of must be strictly earlier than Date")
    if upper.filter(
        pl.col("upper_effective_supported").fill_null(False)
        & (pl.col("upper_effective_source_id") != ENTRY_CANDIDATE_ID)
    ).height:
        raise ValueError("supported upper lookup does not use selected Q2 provenance")
    if lower.filter(
        pl.col("convergence_reference_semantics")
        != CONVERGENCE_REFERENCE_SEMANTICS
    ).height:
        raise ValueError("lower lookup does not use frozen-anchor semantics")
    for frame, supported, distance, source in (
        (
            upper,
            "upper_effective_supported",
            "upper_distance_bp",
            "upper_source_asof_date",
        ),
        (
            lower,
            "lower_effective_supported",
            "lower_distance_bp",
            "lower_source_asof_date",
        ),
    ):
        invalid = frame.filter(
            pl.col(supported).fill_null(False)
            & (
                pl.col(distance).is_null()
                | ~pl.col(distance).is_finite()
                | (pl.col(distance) < 0)
                | pl.col(source).is_null()
            )
        )
        if not invalid.is_empty():
            raise ValueError("supported lookup has incomplete distance provenance")


def _validate_scenario_rows(
    rows: pl.DataFrame,
    definition: S1ScenarioDefinition,
) -> None:
    if definition.policy_kind == "fixed":
        source = f"constant_bp:{int(FIXED_DISTANCE_BP)}"
        invalid = rows.filter(
            pl.col("boundary_quantile").is_not_null()
            | ((pl.col("upper_distance_bp") - FIXED_DISTANCE_BP).abs() >= 1e-12)
            | ((pl.col("lower_distance_bp") - FIXED_DISTANCE_BP).abs() >= 1e-12)
            | (pl.col("upper_source_id") != source)
            | (pl.col("lower_source_id") != source)
            | (pl.col("upper_effective_source_id") != source)
            | (pl.col("lower_effective_source_id") != source)
            | ~pl.col("lookup_supported")
            | pl.col("upper_source_asof_date").is_not_null()
            | pl.col("lower_source_asof_date").is_not_null()
            | pl.col("combined_source_asof_date").is_not_null()
            | pl.col("upper_fallback_used")
            | pl.col("lower_fallback_used")
            | pl.col("upper_fallback_reason").is_not_null()
            | pl.col("lower_fallback_reason").is_not_null()
        )
        if not invalid.is_empty():
            raise ValueError("fixed20 scenario provenance is invalid")
        return

    assert definition.boundary_quantile is not None
    invalid = rows.filter(
        (pl.col("boundary_quantile") != definition.boundary_quantile)
        | (pl.col("upper_source_id") != ENTRY_CANDIDATE_ID)
        | (pl.col("lower_source_id") != definition.lower_candidate_id)
        | (
            pl.col("upper_lookup_supported")
            & (pl.col("upper_effective_source_id") != ENTRY_CANDIDATE_ID)
        )
        | (
            ~pl.col("upper_lookup_supported")
            & pl.col("upper_distance_bp").is_not_null()
        )
        | (
            ~pl.col("lower_lookup_supported")
            & pl.col("lower_distance_bp").is_not_null()
        )
    )
    if not invalid.is_empty():
        raise ValueError("quantile scenario provenance is invalid")
    supported_lower = rows.filter(pl.col("lower_lookup_supported"))
    if definition.lower_candidate_id == LOWER_C0:
        invalid_c0 = supported_lower.filter(
            (pl.col("lower_distance_bp").abs() >= 1e-12)
            | (pl.col("lower_effective_source_id") != "structural_zero")
            | pl.col("lower_fallback_used")
        )
        if not invalid_c0.is_empty():
            raise ValueError("C0 lower provenance is invalid")
    else:
        invalid_conditional = supported_lower.filter(
            ~pl.col("lower_effective_source_id").is_in(
                ["trail20_date_equal", "trail60_date_equal"]
            )
            | (
                (pl.col("lower_effective_source_id") == "trail60_date_equal")
                != pl.col("lower_fallback_used")
            )
        )
        if not invalid_conditional.is_empty():
            raise ValueError("conditional lower fallback provenance is invalid")


def _validate_scenario_spec(spec: S1ScenarioSpec) -> None:
    for name in (
        "Date",
        "ValueCode",
        "QuoteCode",
        "entry_tod_bucket",
        "scenario_id",
        "entry_policy_id",
        "policy_kind",
        "upper_source_id",
        "lower_source_id",
        "upper_support_reason",
        "lower_support_reason",
        "lookup_support_reason",
        "cost_horizon",
        "scenario_definition_sha256",
        "scenario_grid_sha256",
        "spec_version",
    ):
        value = getattr(spec, name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
    if not re.fullmatch(r"\d{8}", spec.Date):
        raise ValueError("Date must be YYYYMMDD")
    if spec.entry_tod_bucket not in TOD_BUCKETS:
        raise ValueError("unsupported entry_tod_bucket")
    definition = SCENARIO_BY_ID.get(spec.scenario_id)
    if definition is None:
        raise ValueError("unsupported scenario_id")
    if spec.anchor_model_id != ANCHOR_MODEL_ID:
        raise ValueError("scenario must use the frozen anchor model")
    if spec.spec_version != SCENARIO_SPEC_VERSION:
        raise ValueError("scenario spec version mismatch")
    if spec.scenario_grid_sha256 != SCENARIO_GRID_SHA256:
        raise ValueError("scenario grid hash mismatch")
    if spec.scenario_definition_sha256 != definition.deterministic_sha256:
        raise ValueError("scenario definition hash mismatch")
    if spec.contains_target_day_outcome:
        raise ValueError("scenario contains target-day outcomes")
    if spec.lookup_supported != (
        spec.upper_lookup_supported and spec.lower_lookup_supported
    ):
        raise ValueError("lookup support flags disagree")
    for name in (
        "upper_source_asof_date",
        "lower_source_asof_date",
        "combined_source_asof_date",
    ):
        value = getattr(spec, name)
        if value is not None and (
            not re.fullmatch(r"\d{8}", value) or value >= spec.Date
        ):
            raise ValueError(f"{name} must be strictly earlier than Date")
    available = [
        value
        for value in (spec.upper_source_asof_date, spec.lower_source_asof_date)
        if value is not None
    ]
    expected_combined = max(available) if available else None
    if spec.combined_source_asof_date != expected_combined:
        raise ValueError("combined source as-of date is invalid")
    for name in ("upper_distance_bp", "lower_distance_bp", "safety_floor_bp"):
        value = getattr(spec, name)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError(f"{name} must be null or finite and non-negative")
    expected = definition.to_dict()
    for name in (
        "entry_policy_id",
        "policy_kind",
        "boundary_quantile",
        "cost_horizon",
        "economic_gate_enabled",
        "deployment_shortlist_eligible",
    ):
        if getattr(spec, name) != expected[name]:
            raise ValueError(f"scenario definition disagrees on {name}")
    expected_floor = definition.safety_floor_bp
    if (spec.safety_floor_bp is None) != (expected_floor is None) or (
        spec.safety_floor_bp is not None
        and expected_floor is not None
        and not math.isclose(
            spec.safety_floor_bp,
            expected_floor,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("scenario definition disagrees on safety_floor_bp")


def _require_columns(frame: pl.DataFrame, columns: set[str], source: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def _assert_unique(frame: pl.DataFrame, keys: list[str], source: str) -> None:
    if frame.select(keys).n_unique() != frame.height:
        raise ValueError(f"{source} keys are duplicated: {keys}")


def _table_sha256_unchecked(frame: pl.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame.sort(ROW_KEYS).iter_rows(named=True):
        digest.update(_canonical_json(row))
        digest.update(b"\n")
    return digest.hexdigest()


__all__ = [
    "CELL_KEYS",
    "FIXED_DISTANCE_BP",
    "LOWER_C0",
    "LOWER_C2",
    "LOWER_C3",
    "ROW_KEYS",
    "S1_SCENARIO_SPEC_SCHEMA",
    "SCENARIO_BY_ID",
    "SCENARIO_DEFINITIONS",
    "SCENARIO_GRID_SHA256",
    "SCENARIO_IDS",
    "SCENARIO_SPEC_VERSION",
    "S1ScenarioDefinition",
    "S1ScenarioSpec",
    "build_s1_scenario_spec_table",
    "load_s1_scenario_spec_table",
    "s1_scenario_spec_table_sha256",
    "s1_scenario_specs_from_table",
    "s1_scenario_specs_to_table",
    "validate_s1_scenario_spec_table",
]
