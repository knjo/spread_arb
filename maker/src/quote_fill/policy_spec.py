"""Frozen, route-neutral policy specifications for the S1/S2 entry grid.

The canonical S1 universe is selected independently of execution outcomes.
This module projects that universe into seven immutable policy rows per
product-day/TOD cell.  Quantile policies use the effective positive Q2 entry
distance and the frozen-v2 C0 centre as the development-default lower.  Fixed
policies use the same constant distance on both sides.

Distances are frozen before the target day.  Absolute order prices are not:
the execution loop must combine a distance with its causal anchor only at the
actual new-send cursor and retain that absolute target for the lifecycle.
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

from ..common.paths import MAKER_ROOT

POLICY_SPEC_VERSION: Final = "s1_policy_spec_frozen_c0_v1"
ENTRY_CANDIDATE_ID: Final = "Q2_trail20_date_equal"
LOWER_CANDIDATE_ID: Final = "C0_center"
ANCHOR_MODEL_ID: Final = "time_ewma_15s"
CONVERGENCE_REFERENCE_SEMANTICS: Final = "frozen_anchor_at_upper_touch"
TOD_BUCKETS: Final = (
    "0905_1000",
    "1000_1100",
    "1100_1200",
    "1200_1300",
)
QUANTILES: Final = (50, 80, 95)
FIXED_DISTANCES_BP: Final = (15.0, 20.0, 25.0, 30.0)
POLICY_IDS: Final = (
    "q50",
    "q80",
    "q95",
    "fixed15",
    "fixed20",
    "fixed25",
    "fixed30",
)

DEFAULT_FOUNDATION_ROOT: Final = (
    MAKER_ROOT / "data" / "walkforward" / "foundation_selection_s05_rebuild_20260826_v1"
)
DEFAULT_FROZEN_ROOT: Final = (
    MAKER_ROOT
    / "data"
    / "walkforward"
    / "foundation_selection_s05_frozen_convergence_20260826_v2"
)
DEFAULT_MOTHER_PATH: Final = DEFAULT_FOUNDATION_ROOT / "s1_mother.parquet"
DEFAULT_ENTRY_LOOKUP_PATH: Final = DEFAULT_FOUNDATION_ROOT / "s1_lookup_long.parquet"
DEFAULT_CONVERGENCE_PATH: Final = (
    DEFAULT_FROZEN_ROOT / "frozen_effective_predictions.parquet"
)

PolicyKind = Literal["quantile", "fixed"]


@dataclass(frozen=True, slots=True)
class PolicySpec:
    """One immutable policy for one product-day and entry TOD bucket."""

    Date: str
    ValueCode: str
    QuoteCode: str
    entry_tod_bucket: str
    policy_id: str
    kind: PolicyKind
    upper_distance_bp: float
    lower_distance_bp: float
    upper_source_id: str
    lower_source_id: str
    upper_source_asof_date: str | None
    lower_source_asof_date: str | None
    combined_source_asof_date: str | None
    anchor_model_id: str = ANCHOR_MODEL_ID
    boundary_quantile: int | None = None
    fallback_reason: str | None = None
    contains_target_day_outcome: bool = False
    development_default: bool = True
    spec_version: str = POLICY_SPEC_VERSION

    def __post_init__(self) -> None:
        _validate_spec(self)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> PolicySpec:
        missing = [column for column in POLICY_SPEC_SCHEMA if column not in value]
        extra = sorted(set(value) - set(POLICY_SPEC_SCHEMA))
        if missing or extra:
            raise ValueError(
                f"PolicySpec mapping mismatch; missing={missing}, extra={extra}"
            )
        return cls(**{column: value[column] for column in POLICY_SPEC_SCHEMA})

    @property
    def deterministic_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


POLICY_SPEC_SCHEMA: Final[dict[str, pl.DataType]] = {
    "Date": pl.String,
    "ValueCode": pl.String,
    "QuoteCode": pl.String,
    "entry_tod_bucket": pl.String,
    "policy_id": pl.String,
    "kind": pl.String,
    "upper_distance_bp": pl.Float64,
    "lower_distance_bp": pl.Float64,
    "upper_source_id": pl.String,
    "lower_source_id": pl.String,
    "upper_source_asof_date": pl.String,
    "lower_source_asof_date": pl.String,
    "combined_source_asof_date": pl.String,
    "anchor_model_id": pl.String,
    "boundary_quantile": pl.Int64,
    "fallback_reason": pl.String,
    "contains_target_day_outcome": pl.Boolean,
    "development_default": pl.Boolean,
    "spec_version": pl.String,
}

CELL_KEYS: Final = (
    "Date",
    "ValueCode",
    "QuoteCode",
    "entry_tod_bucket",
)
ROW_KEYS: Final = (*CELL_KEYS, "policy_id")


def build_policy_spec_table(
    mother: pl.DataFrame,
    entry_lookup: pl.DataFrame,
    convergence_predictions: pl.DataFrame,
) -> pl.DataFrame:
    """Build the exact seven-policy table from canonical-shaped frames.

    Only ``s1_primary=true`` product-days are admitted.  Missing or duplicate
    lookup cells are errors; unsupported rows are never silently dropped or
    replaced by another lookup.
    """

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
            "effective_source_asof_date",
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
            "effective_source_asof_date",
            "fallback_reason",
            "contains_target_day_outcome",
        },
        "frozen convergence predictions",
    )

    mother_primary = mother.filter(pl.col("s1_primary").fill_null(False)).select(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("anchor_model_id").cast(pl.String),
        pl.col("contains_target_day_outcome")
        .cast(pl.Boolean)
        .alias("mother_contains_target_day_outcome"),
    )
    _assert_unique(
        mother_primary,
        ["Date", "ValueCode", "QuoteCode"],
        "primary mother",
    )
    if mother_primary.is_empty():
        raise ValueError("S1 primary mother is empty")
    if mother_primary.filter(
        pl.col("mother_contains_target_day_outcome").fill_null(True)
    ).height:
        raise ValueError("S1 mother contains target-day outcomes")
    if mother_primary.filter(pl.col("anchor_model_id") != ANCHOR_MODEL_ID).height:
        raise ValueError("S1 mother does not use the frozen anchor model")

    upper = entry_lookup.filter(
        (pl.col("candidate_id") == ENTRY_CANDIDATE_ID)
        & (pl.col("side") == "positive")
        & pl.col("boundary_quantile").is_in(QUANTILES)
    ).select(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("anchor_model_id").cast(pl.String),
        pl.col("tod_bucket").cast(pl.String).alias("entry_tod_bucket"),
        pl.col("boundary_quantile").cast(pl.Int64),
        pl.col("boundary_distance_bp").cast(pl.Float64).alias("upper_distance_bp"),
        pl.col("effective_source_asof_date")
        .cast(pl.String)
        .alias("upper_source_asof_date"),
        pl.col("effective_supported").cast(pl.Boolean).alias("upper_supported"),
        pl.col("fallback_reason").cast(pl.String).alias("upper_fallback_reason"),
        pl.col("contains_target_day_outcome")
        .cast(pl.Boolean)
        .alias("upper_contains_target_day_outcome"),
    )
    lower = convergence_predictions.filter(
        (pl.col("candidate_id") == ENTRY_CANDIDATE_ID)
        & (pl.col("convergence_candidate_id") == LOWER_CANDIDATE_ID)
    ).select(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("QuoteCode").cast(pl.String),
        pl.col("anchor_model_id").cast(pl.String),
        pl.col("tod_bucket").cast(pl.String).alias("entry_tod_bucket"),
        pl.col("boundary_quantile").cast(pl.Int64),
        pl.col("effective_threshold_distance_bp")
        .cast(pl.Float64)
        .alias("lower_distance_bp"),
        pl.col("effective_source_asof_date")
        .cast(pl.String)
        .alias("lower_source_asof_date"),
        pl.col("effective_supported").cast(pl.Boolean).alias("lower_supported"),
        pl.col("fallback_reason").cast(pl.String).alias("lower_fallback_reason"),
        pl.col("contains_target_day_outcome")
        .cast(pl.Boolean)
        .alias("lower_contains_target_day_outcome"),
        pl.col("convergence_reference_semantics").cast(pl.String),
    )
    lookup_keys = [*CELL_KEYS, "boundary_quantile"]
    _assert_unique(upper, lookup_keys, "positive Q2 entry lookup")
    _assert_unique(lower, lookup_keys, "C0 lower lookup")

    expected = mother_primary.join(
        pl.DataFrame({"entry_tod_bucket": TOD_BUCKETS}), how="cross"
    ).join(pl.DataFrame({"boundary_quantile": QUANTILES}), how="cross")
    q_rows = (
        expected.join(
            upper,
            on=[*lookup_keys, "anchor_model_id"],
            how="left",
            validate="1:1",
        )
        .join(
            lower,
            on=[*lookup_keys, "anchor_model_id"],
            how="left",
            validate="1:1",
        )
        .with_columns(
            pl.concat_str(
                pl.lit("q"), pl.col("boundary_quantile").cast(pl.String)
            ).alias("policy_id"),
            pl.lit("quantile").alias("kind"),
            pl.lit(ENTRY_CANDIDATE_ID).alias("upper_source_id"),
            pl.lit(LOWER_CANDIDATE_ID).alias("lower_source_id"),
            pl.max_horizontal("upper_source_asof_date", "lower_source_asof_date").alias(
                "combined_source_asof_date"
            ),
            pl.when(pl.col("upper_fallback_reason").is_not_null())
            .then(pl.col("upper_fallback_reason"))
            .otherwise(pl.col("lower_fallback_reason"))
            .alias("fallback_reason"),
            (
                pl.col("mother_contains_target_day_outcome")
                | pl.col("upper_contains_target_day_outcome").fill_null(True)
                | pl.col("lower_contains_target_day_outcome").fill_null(True)
            ).alias("contains_target_day_outcome"),
            pl.lit(True).alias("development_default"),
            pl.lit(POLICY_SPEC_VERSION).alias("spec_version"),
        )
    )
    incomplete = q_rows.filter(
        pl.any_horizontal(
            pl.col("upper_distance_bp").is_null(),
            pl.col("lower_distance_bp").is_null(),
            ~pl.col("upper_supported").fill_null(False),
            ~pl.col("lower_supported").fill_null(False),
            pl.col("upper_source_asof_date").is_null(),
            pl.col("lower_source_asof_date").is_null(),
            pl.col("convergence_reference_semantics").fill_null("")
            != CONVERGENCE_REFERENCE_SEMANTICS,
        )
    )
    if not incomplete.is_empty():
        raise ValueError(
            "quantile policy cells are missing, unsupported, or have invalid "
            "frozen-lower semantics"
        )

    fixed_parts: list[pl.DataFrame] = []
    fixed_cells = expected.select(
        "Date",
        "ValueCode",
        "QuoteCode",
        "entry_tod_bucket",
        "anchor_model_id",
        "mother_contains_target_day_outcome",
    ).unique()
    for distance in FIXED_DISTANCES_BP:
        text = str(int(distance))
        fixed_parts.append(
            fixed_cells.with_columns(
                pl.lit(f"fixed{text}").alias("policy_id"),
                pl.lit("fixed").alias("kind"),
                pl.lit(distance).cast(pl.Float64).alias("upper_distance_bp"),
                pl.lit(distance).cast(pl.Float64).alias("lower_distance_bp"),
                pl.lit(f"constant_bp:{text}").alias("upper_source_id"),
                pl.lit(f"constant_bp:{text}").alias("lower_source_id"),
                pl.lit(None, dtype=pl.String).alias("upper_source_asof_date"),
                pl.lit(None, dtype=pl.String).alias("lower_source_asof_date"),
                pl.lit(None, dtype=pl.String).alias("combined_source_asof_date"),
                pl.lit(None, dtype=pl.Int64).alias("boundary_quantile"),
                pl.lit(None, dtype=pl.String).alias("fallback_reason"),
                pl.col("mother_contains_target_day_outcome").alias(
                    "contains_target_day_outcome"
                ),
                pl.lit(True).alias("development_default"),
                pl.lit(POLICY_SPEC_VERSION).alias("spec_version"),
            )
        )

    result = pl.concat(
        [
            q_rows.select(*POLICY_SPEC_SCHEMA),
            *[part.select(*POLICY_SPEC_SCHEMA) for part in fixed_parts],
        ],
        how="vertical_relaxed",
    ).sort(ROW_KEYS)
    validate_policy_spec_table(result)
    return result


def load_policy_spec_table(
    mother_path: Path = DEFAULT_MOTHER_PATH,
    entry_lookup_path: Path = DEFAULT_ENTRY_LOOKUP_PATH,
    convergence_path: Path = DEFAULT_CONVERGENCE_PATH,
) -> pl.DataFrame:
    """Read canonical artifacts and build the validated policy table."""

    return build_policy_spec_table(
        pl.read_parquet(mother_path),
        pl.read_parquet(entry_lookup_path),
        pl.read_parquet(convergence_path),
    )


def validate_policy_spec_table(frame: pl.DataFrame) -> None:
    """Validate schema, seven-per-cell coverage, provenance, and constants."""

    if frame.schema != pl.Schema(POLICY_SPEC_SCHEMA):
        raise ValueError(
            "policy table schema mismatch: "
            f"expected={pl.Schema(POLICY_SPEC_SCHEMA)}, actual={frame.schema}"
        )
    if frame.is_empty():
        raise ValueError("policy table is empty")
    _assert_unique(frame, list(ROW_KEYS), "policy specs")
    invalid_cells = (
        frame.group_by(CELL_KEYS)
        .agg(
            pl.len().alias("rows"),
            pl.col("policy_id").n_unique().alias("policies"),
            pl.col("policy_id").sort().alias("policy_ids"),
        )
        .filter(
            (pl.col("rows") != len(POLICY_IDS))
            | (pl.col("policies") != len(POLICY_IDS))
            | (pl.col("policy_ids") != sorted(POLICY_IDS))
        )
    )
    if not invalid_cells.is_empty():
        raise ValueError("each cell must contain the exact seven-policy grid")

    for row in frame.iter_rows(named=True):
        PolicySpec.from_dict(row)


def policy_specs_from_table(frame: pl.DataFrame) -> tuple[PolicySpec, ...]:
    validate_policy_spec_table(frame)
    return tuple(PolicySpec.from_dict(row) for row in frame.iter_rows(named=True))


def policy_specs_to_table(
    specs: Sequence[PolicySpec] | Iterable[PolicySpec],
) -> pl.DataFrame:
    values = tuple(specs)
    if not values:
        raise ValueError("policy specs cannot be empty")
    frame = pl.from_dicts(
        [spec.to_dict() for spec in values],
        schema=POLICY_SPEC_SCHEMA,
    ).sort(ROW_KEYS)
    validate_policy_spec_table(frame)
    return frame


def policy_spec_table_sha256(frame: pl.DataFrame) -> str:
    """Hash canonical row dictionaries, independent of input row order."""

    validate_policy_spec_table(frame)
    ordered = frame.sort(ROW_KEYS)
    return _canonical_sha256(list(ordered.iter_rows(named=True)))


def _validate_spec(spec: PolicySpec) -> None:
    for name in ("Date", "ValueCode", "QuoteCode", "entry_tod_bucket", "policy_id"):
        value = getattr(spec, name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
    if not re.fullmatch(r"\d{8}", spec.Date):
        raise ValueError("Date must be YYYYMMDD")
    if spec.entry_tod_bucket not in TOD_BUCKETS:
        raise ValueError("unsupported entry_tod_bucket")
    if spec.policy_id not in POLICY_IDS:
        raise ValueError("unsupported policy_id")
    if spec.kind not in ("quantile", "fixed"):
        raise ValueError("kind must be quantile or fixed")
    if spec.anchor_model_id != ANCHOR_MODEL_ID:
        raise ValueError("policy must use the frozen anchor model")
    if spec.spec_version != POLICY_SPEC_VERSION:
        raise ValueError("policy spec version mismatch")
    if spec.contains_target_day_outcome:
        raise ValueError("policy spec contains target-day outcomes")
    if not spec.development_default:
        raise ValueError("this table only supports the frozen development default")
    for name in ("upper_distance_bp", "lower_distance_bp"):
        value = getattr(spec, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{name} must be finite and non-negative")

    if spec.kind == "quantile":
        expected_quantile = int(spec.policy_id[1:])
        if spec.boundary_quantile != expected_quantile:
            raise ValueError("quantile policy ID and boundary_quantile disagree")
        if spec.upper_source_id != ENTRY_CANDIDATE_ID:
            raise ValueError("quantile upper must use the selected Q2 entry lookup")
        if spec.lower_source_id != LOWER_CANDIDATE_ID:
            raise ValueError("quantile lower must use the C0 development default")
        if not math.isclose(spec.lower_distance_bp, 0.0, abs_tol=1e-12):
            raise ValueError("C0 lower distance must be zero")
        for name in (
            "upper_source_asof_date",
            "lower_source_asof_date",
            "combined_source_asof_date",
        ):
            asof = getattr(spec, name)
            if asof is None or not re.fullmatch(r"\d{8}", asof):
                raise ValueError(f"{name} must be YYYYMMDD for quantile policies")
            if asof >= spec.Date:
                raise ValueError(f"{name} must be strictly earlier than Date")
        expected_combined = max(
            spec.upper_source_asof_date, spec.lower_source_asof_date
        )
        if spec.combined_source_asof_date != expected_combined:
            raise ValueError("combined source as-of date is not the latest input as-of")
    else:
        expected_distance = float(spec.policy_id.removeprefix("fixed"))
        if spec.boundary_quantile is not None:
            raise ValueError("fixed policies cannot carry a boundary quantile")
        if not math.isclose(spec.upper_distance_bp, expected_distance, abs_tol=1e-12):
            raise ValueError("fixed upper distance disagrees with policy ID")
        if not math.isclose(spec.lower_distance_bp, expected_distance, abs_tol=1e-12):
            raise ValueError("fixed lower distance disagrees with policy ID")
        expected_source = f"constant_bp:{int(expected_distance)}"
        if (
            spec.upper_source_id != expected_source
            or spec.lower_source_id != expected_source
        ):
            raise ValueError("fixed policies require constant provenance")
        if any(
            value is not None
            for value in (
                spec.upper_source_asof_date,
                spec.lower_source_asof_date,
                spec.combined_source_asof_date,
            )
        ):
            raise ValueError("fixed constant provenance cannot fake lookup as-of dates")
        if spec.fallback_reason is not None:
            raise ValueError("fixed policies cannot carry a lookup fallback")


def _require_columns(frame: pl.DataFrame, columns: set[str], source: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def _assert_unique(frame: pl.DataFrame, keys: list[str], source: str) -> None:
    if frame.select(keys).n_unique() != frame.height:
        raise ValueError(f"{source} keys are duplicated: {keys}")


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "ANCHOR_MODEL_ID",
    "CONVERGENCE_REFERENCE_SEMANTICS",
    "DEFAULT_CONVERGENCE_PATH",
    "DEFAULT_ENTRY_LOOKUP_PATH",
    "DEFAULT_MOTHER_PATH",
    "ENTRY_CANDIDATE_ID",
    "FIXED_DISTANCES_BP",
    "LOWER_CANDIDATE_ID",
    "POLICY_IDS",
    "POLICY_SPEC_SCHEMA",
    "POLICY_SPEC_VERSION",
    "QUANTILES",
    "TOD_BUCKETS",
    "PolicySpec",
    "build_policy_spec_table",
    "load_policy_spec_table",
    "policy_spec_table_sha256",
    "policy_specs_from_table",
    "policy_specs_to_table",
    "validate_policy_spec_table",
]
