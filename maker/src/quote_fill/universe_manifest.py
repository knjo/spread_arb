"""Versioned research-universe manifest derived from liquidity artifacts.

The liquidity screen contains causal daily rows, but choosing products by
their realised May/June/July-August stability is retrospective.  This module
keeps that distinction explicit.  It produces a machine-readable research
manifest for expensive raw replay; it never marks a product as approved for
production and it does not replace the target-day rolling liquidity gate.

The default policy reproduces the cohorts documented in
``maker/doc/quote_fill/LIQUIDITY_SCREEN.md``:

* 68 exact pseudo-validation stable-core products;
* 43 three-window strict-core products after quarantining 2330;
* 20 July-August extension products;
* 10 selected wide-future controls;
* 2303 as an additional pilot and 6005 as the first-wave wide control.

Every default membership is checked against the verified source artifacts.
Changing the input bundle or the selection policy therefore fails closed
instead of silently changing the execution-replay universe.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, Mapping, Sequence

import polars as pl

from ..common.paths import MAKER_ROOT
from ..quote_width.daily_facts import DEFAULT_DAILY_ROOT
from .liquidity import (
    LIQUIDITY_PUBLICATION_SCHEMA_VERSION,
    LiquidityScreenConfig,
    _liquidity_code_sha256,
    _liquidity_source_lineage,
)
from .targets import (
    FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE,
    PRICE_LADDER_VERSION,
)


DEFAULT_LIQUIDITY_ROOT = MAKER_ROOT / "data" / "walkforward" / "liquidity"
DEFAULT_OUTPUT_DIR = DEFAULT_LIQUIDITY_ROOT / "universe_manifest_v2"

MANIFEST_VERSION = "liquidity_research_universe_manifest_v2_price_ladder"
UNIVERSE_PUBLICATION_SCHEMA_VERSION = (
    "liquidity_research_universe_publication_v1_atomic_lineage"
)
EXPECTED_SOURCE_PUBLICATION_VERSION = "rolling_liquidity_screen_v6_price_ladder"
REQUIRED_ROUTES: tuple[str, ...] = (
    "future_ask_spot_taker",
    "spot_bid_future_taker",
)

EXPECTED_STABLE68: tuple[str, ...] = tuple(
    """1101 1301 1312 1326 1513 1605 1802 2002 2301 2313 2317
    2324 2327 2330 2337 2344 2353 2356 2371 2376 2377 2382 2385
    2408 2409 2412 2441 2449 2454 2474 2603 2609 2610 2615 2618
    2881 2882 2891 3006 3019 3034 3035 3036 3037 3042 3045 3105
    3231 3260 3374 3376 3702 3706 3711 4904 4919 4958 5347 5483
    5871 6147 6239 6278 6770 8039 8069 8150 9958""".split()
)
EXPECTED_STRICT43: tuple[str, ...] = tuple(
    """1101 1513 1605 1802 2002 2301 2313 2317 2324 2344
    2353 2371 2376 2382 2408 2409 2412 2449 2474 2603 2609
    2610 2615 2618 2881 2882 2891 3006 3019 3035 3045 3105
    3231 3260 3374 3376 3706 3711 5347 5483 5871 8039 9958""".split()
)
EXPECTED_EXTENSION20: tuple[str, ...] = tuple(
    """1312 1326 2327 2356 2377 2441 2454 3034 3036 3037 3042
    3702 4904 4919 4958 6147 6239 6278 8069 8150""".split()
)
EXPECTED_WIDE10: tuple[str, ...] = tuple(
    "6005 4162 6245 6547 6121 4743 1722 5876 5457 5534".split()
)

STRICT_CORE_ARTIFACT = "strict_core_43.csv"
EXTENSION_ARTIFACT = "extension_20.csv"
WIDE_CONTROLS_ARTIFACT = "wide_future_controls_10.csv"
FIRST_WAVE_ARTIFACT = "first_wave_45.csv"
FIRST_WAVE_SYMBOLS_ARTIFACT = "first_wave_45_symbols.txt"

_SCREEN_ARTIFACT = "rolling_liquidity_screen.parquet"
_STABLE_ARTIFACT = "pseudo_validation_stable_core_products.csv"
_CONFIG_ARTIFACT = "config.json"


@dataclass(frozen=True)
class StabilityWindow:
    """One retrospective calendar window used by the cohort policy."""

    name: str
    start_date: str
    end_date: str
    expected_sessions: int

    def validate(self) -> None:
        if not self.name:
            raise ValueError("stability window name cannot be empty")
        for name, value in (
            ("start_date", self.start_date),
            ("end_date", self.end_date),
        ):
            if len(value) != 8 or not value.isdigit():
                raise ValueError(f"{name} must use YYYYMMDD")
        if self.start_date > self.end_date:
            raise ValueError("stability window start must not follow its end")
        if self.expected_sessions <= 0:
            raise ValueError("expected_sessions must be positive")


@dataclass(frozen=True)
class UniverseManifestConfig:
    """Frozen retrospective research-universe selection policy."""

    boundary_quantile: int = 50
    windows: tuple[StabilityWindow, ...] = (
        StabilityWindow("may_fine_tune", "20260501", "20260531", 20),
        StabilityWindow("june_confirmation", "20260601", "20260630", 21),
        StabilityWindow("jul_aug_pseudo", "20260701", "20260813", 31),
    )
    extension_window_name: str = "jul_aug_pseudo"
    minimum_coverage_rate: float = 0.80
    minimum_pass_rate: float = 0.90
    minimum_core_rate: float = 0.90
    expected_stable_members: tuple[str, ...] = EXPECTED_STABLE68
    expected_strict_members: tuple[str, ...] = EXPECTED_STRICT43
    expected_extension_members: tuple[str, ...] = EXPECTED_EXTENSION20
    selected_wide_controls: tuple[str, ...] = EXPECTED_WIDE10
    pilot_value_code: str = "2303"
    control_value_code: str = "6005"
    quarantine_value_code: str = "2330"
    expected_source_publication_version: str = EXPECTED_SOURCE_PUBLICATION_VERSION
    expected_source_schema_version: str = LIQUIDITY_PUBLICATION_SCHEMA_VERSION
    expected_price_ladder_version: str = PRICE_LADDER_VERSION
    expected_future_one_dollar_tick_effective_date: str = (
        FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE
    )
    manifest_version: str = MANIFEST_VERSION

    def validate(self) -> None:
        if self.boundary_quantile <= 0 or self.boundary_quantile >= 100:
            raise ValueError("boundary_quantile must be between 0 and 100")
        if not self.windows or len({item.name for item in self.windows}) != len(
            self.windows
        ):
            raise ValueError("stability windows must be nonempty and unique")
        for item in self.windows:
            item.validate()
        if self.extension_window_name not in {item.name for item in self.windows}:
            raise ValueError("extension_window_name is absent from windows")
        for name in (
            "minimum_coverage_rate",
            "minimum_pass_rate",
            "minimum_core_rate",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in (
            "expected_stable_members",
            "expected_strict_members",
            "expected_extension_members",
            "selected_wide_controls",
        ):
            _validate_value_codes(getattr(self, name), name)
        for name in (
            "pilot_value_code",
            "control_value_code",
            "quarantine_value_code",
        ):
            _validate_value_codes((getattr(self, name),), name)
        if not self.expected_source_publication_version:
            raise ValueError("expected_source_publication_version cannot be empty")
        if self.expected_source_schema_version != LIQUIDITY_PUBLICATION_SCHEMA_VERSION:
            raise ValueError(
                "expected_source_schema_version must match the implemented "
                "liquidity publication schema"
            )
        if self.expected_price_ladder_version != PRICE_LADDER_VERSION:
            raise ValueError(
                "expected_price_ladder_version must match the implemented ladder"
            )
        if (
            self.expected_future_one_dollar_tick_effective_date
            != FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE
        ):
            raise ValueError(
                "expected_future_one_dollar_tick_effective_date must match "
                "the implemented ladder"
            )
        if not self.manifest_version:
            raise ValueError("manifest_version cannot be empty")

    def payload(self) -> dict[str, object]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class UniverseSourceLineage:
    """Verified content identities for the liquidity source bundle."""

    liquidity_root: str
    complete_sha256: str
    schema_version: str
    marker_payload_sha256: str
    publication_version: str
    price_ladder_version: str
    future_one_dollar_tick_effective_date: str
    latest_snapshot_date: str
    rolling_screen_sha256: str
    stable_core_sha256: str
    config_sha256: str
    source_lineage: Mapping[str, str]
    declared_artifact_sha256: Mapping[str, str]


@dataclass(frozen=True)
class UniverseSourceBundle:
    rolling_screen: pl.DataFrame
    stable_core_products: pl.DataFrame
    lineage: UniverseSourceLineage


@dataclass(frozen=True)
class ResearchUniverseManifest:
    manifest: pl.DataFrame
    strict_core: pl.DataFrame
    extension: pl.DataFrame
    wide_controls: pl.DataFrame
    first_wave: pl.DataFrame
    first_wave_csv_symbols: str


def load_verified_liquidity_sources(
    liquidity_root: Path = DEFAULT_LIQUIDITY_ROOT,
    *,
    expected_daily_root: Path | None = None,
    expected_boundary_path: Path | None = None,
) -> UniverseSourceBundle:
    """Verify the complete liquidity publication before reading its frames.

    The default formal location is additionally matched against the current
    daily-marker generation and rolling-boundary publication.  A copied
    publication remains portable, but still has to carry a self-consistent
    current-schema marker, current builder-code hash, and artifact manifest.
    """

    root = Path(liquidity_root)
    marker_path = root / "complete.json"
    if not marker_path.is_file():
        raise FileNotFoundError(marker_path)
    marker = _read_json_object(marker_path)
    _validate_liquidity_complete_marker(marker)

    config_contract = marker["config"]
    assert isinstance(config_contract, dict)  # validated above
    try:
        liquidity_config = LiquidityScreenConfig(**config_contract)
    except TypeError as error:
        raise ValueError("liquidity marker has invalid config contract") from error
    liquidity_config.validate()
    source_lineage = marker["source_lineage"]
    assert isinstance(source_lineage, dict)  # validated above

    if root.resolve() == DEFAULT_LIQUIDITY_ROOT.resolve():
        expected_daily_root = expected_daily_root or DEFAULT_DAILY_ROOT
        expected_boundary_path = expected_boundary_path or (
            Path(expected_daily_root).parent
            / "rolling_boundaries"
            / "rolling_boundary_snapshots.parquet"
        )
    if (expected_daily_root is None) != (expected_boundary_path is None):
        raise ValueError(
            "expected_daily_root and expected_boundary_path must be provided together"
        )
    if expected_daily_root is not None and expected_boundary_path is not None:
        current_lineage = _liquidity_source_lineage(
            boundary_path=Path(expected_boundary_path),
            daily_root=Path(expected_daily_root),
            config=liquidity_config,
        )
        if source_lineage != current_lineage:
            raise ValueError(
                "liquidity source lineage does not match the current daily and "
                "rolling-boundary generation"
            )

    artifacts = marker.get("artifacts")
    assert isinstance(artifacts, dict)  # validated above
    for required in (_SCREEN_ARTIFACT, _STABLE_ARTIFACT, _CONFIG_ARTIFACT):
        if required not in artifacts:
            raise ValueError(f"liquidity bundle is missing {required}")

    declared: dict[str, str] = {}
    for name, metadata in artifacts.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("liquidity artifact names must be plain filenames")
        if not isinstance(metadata, dict):
            raise ValueError(f"invalid liquidity artifact metadata for {name}")
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _file_sha256(path)
        expected = str(metadata["sha256"])
        if metadata.get("bytes") != path.stat().st_size or actual != expected:
            raise ValueError(f"liquidity artifact hash mismatch: {path}")
        _validate_declared_frame_shape(path, metadata)
        declared[name] = actual

    source_config = _read_json_object(root / _CONFIG_ARTIFACT)
    latest = str(source_config.get("latest_snapshot_date", ""))
    if len(latest) != 8 or not latest.isdigit():
        raise ValueError("liquidity config has invalid latest_snapshot_date")
    if source_config.get("production_universe_approved") is not False:
        raise ValueError("source liquidity universe must remain research-only")
    if any(
        source_config.get(key) != value for key, value in config_contract.items()
    ):
        raise ValueError("liquidity config contract differs from completion marker")
    if any(
        source_config.get(key) != value for key, value in source_lineage.items()
    ):
        raise ValueError("liquidity config source lineage differs from marker")

    publication = str(marker["publication_version"])
    price_ladder_version = source_config.get("price_ladder_version")
    if price_ladder_version != PRICE_LADDER_VERSION:
        raise ValueError(
            "liquidity config price_ladder_version mismatch: "
            f"{price_ladder_version!r} != {PRICE_LADDER_VERSION!r}"
        )
    effective_date = source_config.get("future_one_dollar_tick_effective_date")
    if effective_date != FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE:
        raise ValueError(
            "liquidity config future tick effective-date mismatch: "
            f"{effective_date!r} != {FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE!r}"
        )

    rolling_screen = pl.read_parquet(root / _SCREEN_ARTIFACT)
    _validate_liquidity_screen_lineage(
        rolling_screen,
        publication_version=publication,
    )
    stable_core = pl.read_csv(root / _STABLE_ARTIFACT)

    return UniverseSourceBundle(
        rolling_screen=rolling_screen,
        stable_core_products=stable_core,
        lineage=UniverseSourceLineage(
            liquidity_root=str(root.resolve()),
            complete_sha256=_file_sha256(marker_path),
            schema_version=str(marker["schema_version"]),
            marker_payload_sha256=str(marker["marker_payload_sha256"]),
            publication_version=publication,
            price_ladder_version=str(price_ladder_version),
            future_one_dollar_tick_effective_date=str(effective_date),
            latest_snapshot_date=latest,
            rolling_screen_sha256=declared[_SCREEN_ARTIFACT],
            stable_core_sha256=declared[_STABLE_ARTIFACT],
            config_sha256=declared[_CONFIG_ARTIFACT],
            source_lineage=dict(sorted(source_lineage.items())),
            declared_artifact_sha256=dict(sorted(declared.items())),
        ),
    )


def _validate_liquidity_complete_marker(marker: Mapping[str, object]) -> None:
    if marker.get("complete") is not True:
        raise ValueError("liquidity completion marker is incomplete")
    if marker.get("schema_version") != LIQUIDITY_PUBLICATION_SCHEMA_VERSION:
        raise ValueError("liquidity completion marker schema mismatch")
    publication = marker.get("publication_version")
    if not isinstance(publication, str) or not publication:
        raise ValueError("liquidity complete.json has no publication_version")
    if (
        marker.get("price_ladder_version") != PRICE_LADDER_VERSION
        or marker.get("future_one_dollar_tick_effective_date")
        != FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE
    ):
        raise ValueError("liquidity completion marker ladder lineage mismatch")

    declared_digest = marker.get("marker_payload_sha256")
    unhashed = dict(marker)
    unhashed.pop("marker_payload_sha256", None)
    if not _is_sha256(declared_digest) or declared_digest != _canonical_sha256(
        unhashed
    ):
        raise ValueError("liquidity completion marker self-hash mismatch")

    config = marker.get("config")
    source = marker.get("source_lineage")
    artifacts = marker.get("artifacts")
    if not isinstance(config, dict):
        raise ValueError("liquidity completion marker has no config contract")
    if not isinstance(source, dict):
        raise ValueError("liquidity completion marker has no source lineage")
    if not isinstance(artifacts, dict):
        raise ValueError("liquidity complete.json has no artifact map")
    if (
        config.get("screen_version") != publication
        or config.get("price_ladder_version") != PRICE_LADDER_VERSION
        or config.get("future_one_dollar_tick_effective_date")
        != FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE
    ):
        raise ValueError("liquidity marker config lineage mismatch")

    required_source = {
        "boundary_file_sha256",
        "boundary_marker_sha256",
        "daily_marker_set_sha256",
        "liquidity_code_sha256",
        "config_sha256",
        "price_ladder_version",
        "future_one_dollar_tick_effective_date",
    }
    if not required_source.issubset(source):
        missing = sorted(required_source - set(source))
        raise ValueError(f"liquidity marker source lineage is incomplete: {missing}")
    hash_keys = required_source - {
        "price_ladder_version",
        "future_one_dollar_tick_effective_date",
    }
    if any(not _is_sha256(source.get(key)) for key in hash_keys):
        raise ValueError("liquidity marker source lineage has invalid hashes")
    if (
        source.get("liquidity_code_sha256") != _liquidity_code_sha256()
        or source.get("config_sha256") != _canonical_sha256(config)
        or source.get("price_ladder_version") != PRICE_LADDER_VERSION
        or source.get("future_one_dollar_tick_effective_date")
        != FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE
    ):
        raise ValueError("liquidity marker source lineage is not current")

    for name, metadata in artifacts.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("liquidity artifact names must be plain filenames")
        if (
            not isinstance(metadata, dict)
            or not isinstance(metadata.get("bytes"), int)
            or int(metadata["bytes"]) < 0
            or not _is_sha256(metadata.get("sha256"))
        ):
            raise ValueError(f"invalid liquidity artifact metadata for {name}")


def _validate_declared_frame_shape(
    path: Path,
    metadata: Mapping[str, object],
) -> None:
    if path.suffix not in {".parquet", ".csv"}:
        return
    if path.suffix == ".parquet":
        schema = pl.read_parquet_schema(path)
        rows = int(pl.scan_parquet(path).select(pl.len()).collect().item())
        columns = len(schema)
        names = list(schema.names())
    else:
        frame = pl.read_csv(path)
        rows, columns = frame.shape
        names = frame.columns
    if (
        metadata.get("rows") != rows
        or metadata.get("columns") != columns
        or metadata.get("column_names") != names
    ):
        raise ValueError(f"liquidity artifact shape mismatch: {path}")


def _validate_liquidity_screen_lineage(
    screen: pl.DataFrame,
    *,
    publication_version: str,
) -> None:
    required = {
        "liquidity_screen_version",
        "price_ladder_version",
        "future_one_dollar_tick_effective_date",
    }
    missing = sorted(required - set(screen.columns))
    if missing:
        raise ValueError(f"liquidity screen lacks row lineage: {missing}")
    invalid = screen.filter(
        (pl.col("liquidity_screen_version") != publication_version)
        | pl.col("liquidity_screen_version").is_null()
        | (pl.col("price_ladder_version") != PRICE_LADDER_VERSION)
        | pl.col("price_ladder_version").is_null()
        | (
            pl.col("future_one_dollar_tick_effective_date")
            != FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE
        )
        | pl.col("future_one_dollar_tick_effective_date").is_null()
    )
    if invalid.height:
        raise ValueError("liquidity screen row lineage mismatch")


def build_research_universe_manifest(
    rolling_screen: pl.DataFrame,
    stable_core_products: pl.DataFrame,
    lineage: UniverseSourceLineage,
    config: UniverseManifestConfig = UniverseManifestConfig(),
) -> ResearchUniverseManifest:
    """Derive and validate the exact research cohorts.

    Daily screen rows are causal, but membership uses realised outcomes across
    full windows and is therefore explicitly retrospective.
    """

    config.validate()
    if lineage.publication_version != config.expected_source_publication_version:
        raise ValueError(
            "liquidity publication version mismatch: "
            f"{lineage.publication_version!r} != "
            f"{config.expected_source_publication_version!r}"
        )
    if lineage.schema_version != config.expected_source_schema_version:
        raise ValueError(
            "liquidity publication schema mismatch: "
            f"{lineage.schema_version!r} != "
            f"{config.expected_source_schema_version!r}"
        )
    if lineage.price_ladder_version != config.expected_price_ladder_version:
        raise ValueError(
            "liquidity price-ladder lineage mismatch: "
            f"{lineage.price_ladder_version!r} != "
            f"{config.expected_price_ladder_version!r}"
        )
    if (
        lineage.future_one_dollar_tick_effective_date
        != config.expected_future_one_dollar_tick_effective_date
    ):
        raise ValueError(
            "liquidity future tick effective-date lineage mismatch: "
            f"{lineage.future_one_dollar_tick_effective_date!r} != "
            f"{config.expected_future_one_dollar_tick_effective_date!r}"
        )
    if not _is_sha256(lineage.marker_payload_sha256):
        raise ValueError("liquidity marker self-hash lineage is invalid")
    source_lineage = dict(lineage.source_lineage)
    source_hash_keys = {
        "boundary_file_sha256",
        "boundary_marker_sha256",
        "daily_marker_set_sha256",
        "liquidity_code_sha256",
        "config_sha256",
    }
    if (
        not source_hash_keys.issubset(source_lineage)
        or any(not _is_sha256(source_lineage.get(key)) for key in source_hash_keys)
        or source_lineage.get("liquidity_code_sha256")
        != _liquidity_code_sha256()
        or source_lineage.get("price_ladder_version")
        != config.expected_price_ladder_version
        or source_lineage.get("future_one_dollar_tick_effective_date")
        != config.expected_future_one_dollar_tick_effective_date
    ):
        raise ValueError("liquidity source generation lineage is not current")
    declared = dict(lineage.declared_artifact_sha256)
    if (
        declared.get(_SCREEN_ARTIFACT) != lineage.rolling_screen_sha256
        or declared.get(_STABLE_ARTIFACT) != lineage.stable_core_sha256
        or declared.get(_CONFIG_ARTIFACT) != lineage.config_sha256
        or any(not _is_sha256(value) for value in declared.values())
    ):
        raise ValueError("liquidity declared artifact lineage is inconsistent")
    _require(
        rolling_screen,
        {
            "Date",
            "ValueCode",
            "boundary_quantile",
            "route",
            "liquidity_gate_status",
            "replay_tier",
            "wide_maker_spread_flag",
            "execution_safe_snapshot",
            "contains_target_day_outcome",
        },
        "rolling liquidity screen",
    )
    _require(
        stable_core_products,
        {
            "ValueCode",
            "retrospective_pseudo_validation",
            "production_universe_approved",
        },
        "pseudo stable-core products",
    )

    screen = rolling_screen.with_columns(
        pl.col("Date").cast(pl.String),
        pl.col("ValueCode").cast(pl.String),
        pl.col("boundary_quantile").cast(pl.Int64),
    ).filter(pl.col("boundary_quantile") == config.boundary_quantile)
    key = ["Date", "ValueCode", "boundary_quantile", "route"]
    if screen.select(key).n_unique() != screen.height:
        raise ValueError("rolling screen contains duplicate product-day-route keys")
    if screen.filter(
        ~pl.col("route").is_in(REQUIRED_ROUTES)
        | ~pl.col("execution_safe_snapshot").fill_null(False)
        | pl.col("contains_target_day_outcome").fill_null(True)
    ).height:
        raise ValueError("rolling screen rows are not causal execution-safe inputs")

    stable = stable_core_products.with_columns(
        pl.col("ValueCode").cast(pl.String)
    )
    if stable.select("ValueCode").n_unique() != stable.height:
        raise ValueError("stable-core source contains duplicate products")
    if stable.filter(
        ~pl.col("retrospective_pseudo_validation").fill_null(False)
        | pl.col("production_universe_approved").fill_null(True)
    ).height:
        raise ValueError("stable-core source flags are not research-only")
    stable_values = set(stable["ValueCode"].to_list())
    expected_stable = set(config.expected_stable_members)
    if stable_values != expected_stable:
        raise ValueError(
            _membership_error("stable core", expected_stable, stable_values)
        )

    qualified_by_window: dict[str, set[str]] = {}
    for window in config.windows:
        qualified_by_window[window.name] = _qualifying_products(
            screen, window, config
        )
    pre_quarantine = set.intersection(*qualified_by_window.values())
    extension_window = qualified_by_window[config.extension_window_name]
    extension = extension_window - pre_quarantine

    quarantine = {config.quarantine_value_code}
    strict = pre_quarantine - quarantine
    expected_strict = set(config.expected_strict_members)
    expected_extension = set(config.expected_extension_members)
    if strict != expected_strict:
        raise ValueError(_membership_error("strict core", expected_strict, strict))
    if extension != expected_extension:
        raise ValueError(
            _membership_error("extension", expected_extension, extension)
        )
    if config.quarantine_value_code not in pre_quarantine:
        raise ValueError("quarantine product is not in pre-quarantine strict core")
    if not (pre_quarantine | extension).issubset(stable_values):
        missing = sorted((pre_quarantine | extension) - stable_values, key=int)
        raise ValueError(f"derived cohorts are absent from stable-core source: {missing}")

    wide = set(config.selected_wide_controls)
    _validate_wide_controls(screen, wide, config)
    _validate_special_memberships(strict, extension, wide, pre_quarantine, config)

    pilot = {config.pilot_value_code}
    control = {config.control_value_code}
    first_wave = strict | pilot | control
    all_values = strict | extension | wide | pilot | quarantine
    config_sha = _canonical_sha256(config.payload())

    records: list[dict[str, object]] = []
    for value_code in sorted(all_values, key=int):
        strict_member = value_code in strict
        extension_member = value_code in extension
        wide_member = value_code in wide
        pilot_member = value_code in pilot
        control_member = value_code in control
        quarantine_member = value_code in quarantine
        if quarantine_member:
            primary = "book_semantics_quarantine"
            reason = "2330_zero_tick_future_book_semantics_quarantine"
        elif strict_member:
            primary = "strict_core"
            reason = "both_routes_80pct_coverage_90pct_pass_core_all_three_windows"
        elif extension_member:
            primary = "pseudo_extension"
            reason = "jul_aug_both_routes_80pct_coverage_90pct_pass_core_only"
        elif wide_member:
            primary = "wide_future_control"
            reason = "selected_31of31_pass_and_future_wide_maker_control"
        elif pilot_member:
            primary = "pilot"
            reason = "existing_2303_execution_pilot"
        else:
            raise AssertionError("manifest value has no primary cohort")
        records.append(
            {
                "ValueCode": value_code,
                "primary_cohort": primary,
                "selection_reason": reason,
                "strict_core_member": strict_member,
                "extension_member": extension_member,
                "wide_future_control_member": wide_member,
                "pilot_member": pilot_member,
                "control_member": control_member,
                "pre_quarantine_strict_core_member": value_code
                in pre_quarantine,
                "quarantine": quarantine_member,
                "first_wave_member": value_code in first_wave,
                "execution_cli_member": value_code in first_wave
                and not quarantine_member,
                "retrospective_research_selection": True,
                "selection_contains_target_day_outcomes": True,
                "source_daily_rows_execution_safe": True,
                "production_universe_approved": False,
                "runtime_daily_liquidity_gate_required": True,
                "manifest_version": config.manifest_version,
                "selection_config_sha256": config_sha,
                "source_publication_version": lineage.publication_version,
                "source_publication_schema_version": lineage.schema_version,
                "source_marker_payload_sha256": lineage.marker_payload_sha256,
                "source_price_ladder_version": lineage.price_ladder_version,
                "source_future_one_dollar_tick_effective_date": (
                    lineage.future_one_dollar_tick_effective_date
                ),
                "source_latest_snapshot_date": lineage.latest_snapshot_date,
                "source_liquidity_complete_sha256": lineage.complete_sha256,
                "source_rolling_screen_sha256": lineage.rolling_screen_sha256,
                "source_stable_core_sha256": lineage.stable_core_sha256,
                "source_config_sha256": lineage.config_sha256,
                "source_boundary_file_sha256": source_lineage.get(
                    "boundary_file_sha256"
                ),
                "source_boundary_marker_sha256": source_lineage.get(
                    "boundary_marker_sha256"
                ),
                "source_daily_marker_set_sha256": source_lineage.get(
                    "daily_marker_set_sha256"
                ),
                "source_liquidity_code_sha256": source_lineage.get(
                    "liquidity_code_sha256"
                ),
                "source_liquidity_config_contract_sha256": source_lineage.get(
                    "config_sha256"
                ),
            }
        )
    manifest = pl.from_dicts(records, infer_schema_length=None).sort(
        pl.col("ValueCode").cast(pl.Int64)
    )

    strict_frame = _symbol_frame(strict)
    extension_frame = _symbol_frame(extension)
    wide_frame = _ordered_symbol_frame(config.selected_wide_controls)
    first_wave_frame = _symbol_frame(first_wave)
    _validate_final_frames(
        manifest,
        strict_frame,
        extension_frame,
        wide_frame,
        first_wave_frame,
        config,
    )
    comma = ",".join(first_wave_frame["ValueCode"].to_list())
    return ResearchUniverseManifest(
        manifest=manifest,
        strict_core=strict_frame,
        extension=extension_frame,
        wide_controls=wide_frame,
        first_wave=first_wave_frame,
        first_wave_csv_symbols=comma,
    )


def publish_research_universe_manifest(
    result: ResearchUniverseManifest,
    lineage: UniverseSourceLineage,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config: UniverseManifestConfig = UniverseManifestConfig(),
) -> Path:
    """Atomically publish the manifest and execution-CLI symbol lists."""

    config.validate()
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        outputs: dict[str, dict[str, object]] = {}
        frames = {
            "research_universe_manifest.parquet": result.manifest,
            "research_universe_manifest.csv": result.manifest,
            STRICT_CORE_ARTIFACT: result.strict_core,
            EXTENSION_ARTIFACT: result.extension,
            WIDE_CONTROLS_ARTIFACT: result.wide_controls,
            FIRST_WAVE_ARTIFACT: result.first_wave,
        }
        for name, frame in frames.items():
            path = stage / name
            if path.suffix == ".parquet":
                frame.write_parquet(path)
            else:
                frame.write_csv(path)
            outputs[name] = _frame_artifact_metadata(path, frame)
        text_path = stage / FIRST_WAVE_SYMBOLS_ARTIFACT
        text_path.write_text(result.first_wave_csv_symbols + "\n", encoding="utf-8")
        outputs[text_path.name] = _artifact_metadata(text_path, 1)

        marker = {
            "complete": True,
            "schema_version": UNIVERSE_PUBLICATION_SCHEMA_VERSION,
            "manifest_version": config.manifest_version,
            "price_ladder_version": config.expected_price_ladder_version,
            "future_one_dollar_tick_effective_date": (
                config.expected_future_one_dollar_tick_effective_date
            ),
            "config": config.payload(),
            "config_sha256": _canonical_sha256(config.payload()),
            "source": asdict(lineage),
            "counts": {
                "manifest_products": result.manifest.height,
                "strict_core": result.strict_core.height,
                "extension": result.extension.height,
                "wide_future_controls": result.wide_controls.height,
                "first_wave": result.first_wave.height,
                "quarantine": int(result.manifest["quarantine"].sum()),
            },
            "semantics": {
                "retrospective_research_selection": True,
                "selection_contains_target_day_outcomes": True,
                "production_universe_approved": False,
                "runtime_daily_liquidity_gate_required": True,
                "quarantine_excluded_from_execution_cli": True,
                "price_ladder_version": config.expected_price_ladder_version,
                "future_one_dollar_tick_effective_date": (
                    config.expected_future_one_dollar_tick_effective_date
                ),
                "first_wave_formula": (
                    f"strict{len(config.expected_strict_members)}_plus_"
                    f"{config.pilot_value_code}_pilot_plus_"
                    f"{config.control_value_code}_control"
                ),
            },
            "artifacts": outputs,
        }
        marker["marker_payload_sha256"] = _canonical_sha256(marker)
        marker_path = stage / "complete.json"
        marker_path.write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        stage.replace(destination)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return destination


def _qualifying_products(
    screen: pl.DataFrame,
    window: StabilityWindow,
    config: UniverseManifestConfig,
) -> set[str]:
    selected = screen.filter(
        (pl.col("Date") >= window.start_date)
        & (pl.col("Date") <= window.end_date)
    )
    dates = sorted(selected["Date"].unique().to_list())
    if len(dates) != window.expected_sessions:
        raise ValueError(
            f"{window.name}: observed {len(dates)} sessions, "
            f"expected {window.expected_sessions}"
        )
    route = selected.group_by(["ValueCode", "route"]).agg(
        pl.col("Date").n_unique().alias("product_days"),
        (pl.col("liquidity_gate_status") == "pass").mean().alias("pass_rate"),
        (pl.col("replay_tier") == "core_candidate").mean().alias("core_rate"),
    ).with_columns(
        (pl.col("product_days") / len(dates)).alias("coverage_rate")
    ).filter(
        (pl.col("coverage_rate") >= config.minimum_coverage_rate)
        & (pl.col("pass_rate") >= config.minimum_pass_rate)
        & (pl.col("core_rate") >= config.minimum_core_rate)
    )
    product = route.group_by("ValueCode").agg(
        pl.col("route").n_unique().alias("qualifying_routes")
    ).filter(pl.col("qualifying_routes") == len(REQUIRED_ROUTES))
    return set(product["ValueCode"].to_list())


def _validate_wide_controls(
    screen: pl.DataFrame,
    wide: set[str],
    config: UniverseManifestConfig,
) -> None:
    window = next(
        item for item in config.windows if item.name == config.extension_window_name
    )
    selected = screen.filter(
        (pl.col("Date") >= window.start_date)
        & (pl.col("Date") <= window.end_date)
        & (pl.col("route") == "future_ask_spot_taker")
        & pl.col("ValueCode").is_in(sorted(wide))
    )
    facts = selected.group_by("ValueCode").agg(
        pl.col("Date").n_unique().alias("product_days"),
        (pl.col("liquidity_gate_status") == "pass").sum().alias("pass_days"),
        pl.col("wide_maker_spread_flag").fill_null(False).sum().alias(
            "wide_maker_days"
        ),
    )
    observed = set(facts["ValueCode"].to_list())
    if observed != wide:
        raise ValueError(_membership_error("wide controls", wide, observed))
    invalid = facts.filter(
        (pl.col("product_days") != window.expected_sessions)
        | (pl.col("pass_days") != window.expected_sessions)
        | (pl.col("wide_maker_days") != window.expected_sessions)
    )
    if invalid.height:
        raise ValueError(
            "selected wide controls must be pass and future-wide-maker on "
            f"every {window.name} session: {invalid.to_dicts()}"
        )


def _validate_special_memberships(
    strict: set[str],
    extension: set[str],
    wide: set[str],
    pre_quarantine: set[str],
    config: UniverseManifestConfig,
) -> None:
    cohorts = (strict, extension, wide)
    names = ("strict", "extension", "wide")
    for index, left in enumerate(cohorts):
        for right_index in range(index + 1, len(cohorts)):
            overlap = left & cohorts[right_index]
            if overlap:
                raise ValueError(
                    f"{names[index]}/{names[right_index]} overlap: "
                    f"{sorted(overlap, key=int)}"
                )
    if config.pilot_value_code in strict | extension | wide:
        raise ValueError("pilot 2303 must be an explicit extra, not a base cohort")
    if config.control_value_code not in wide:
        raise ValueError("control 6005 must belong to selected wide controls")
    if config.quarantine_value_code not in pre_quarantine:
        raise ValueError("quarantine 2330 must belong to pre-quarantine strict core")
    if config.quarantine_value_code in strict | extension | wide:
        raise ValueError("quarantine 2330 leaked into an executable cohort")
    specials = {
        config.pilot_value_code,
        config.control_value_code,
        config.quarantine_value_code,
    }
    if len(specials) != 3:
        raise ValueError("pilot/control/quarantine products must be distinct")


def _validate_final_frames(
    manifest: pl.DataFrame,
    strict: pl.DataFrame,
    extension: pl.DataFrame,
    wide: pl.DataFrame,
    first_wave: pl.DataFrame,
    config: UniverseManifestConfig,
) -> None:
    expected_counts = {
        "strict": (strict, len(config.expected_strict_members)),
        "extension": (extension, len(config.expected_extension_members)),
        "wide": (wide, len(config.selected_wide_controls)),
        "first_wave": (first_wave, len(config.expected_strict_members) + 2),
    }
    if manifest.select("ValueCode").n_unique() != manifest.height:
        raise AssertionError("manifest contains duplicate products")
    for name, (frame, expected) in expected_counts.items():
        if frame.height != expected or frame.select("ValueCode").n_unique() != expected:
            raise AssertionError(
                f"{name} membership/count mismatch: {frame.height} != {expected}"
            )
    quarantine = manifest.filter(pl.col("quarantine"))
    if quarantine["ValueCode"].to_list() != [config.quarantine_value_code]:
        raise AssertionError("quarantine membership is not exact")
    if manifest.filter(pl.col("quarantine") & pl.col("execution_cli_member")).height:
        raise AssertionError("quarantine leaked into execution CLI membership")
    manifest_first = set(
        manifest.filter(pl.col("first_wave_member"))["ValueCode"].to_list()
    )
    if manifest_first != set(first_wave["ValueCode"].to_list()):
        raise AssertionError("first-wave frame differs from manifest flags")
    if manifest["production_universe_approved"].any():
        raise AssertionError("research manifest cannot approve production universe")


def _symbol_frame(values: Iterable[str]) -> pl.DataFrame:
    return _ordered_symbol_frame(sorted(set(values), key=int))


def _ordered_symbol_frame(values: Sequence[str] | Iterable[str]) -> pl.DataFrame:
    items = tuple(str(value) for value in values)
    _validate_value_codes(items, "symbol frame")
    return pl.DataFrame({"ValueCode": list(items)}, schema={"ValueCode": pl.String})


def _validate_value_codes(values: Sequence[str], source: str) -> None:
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{source} must be nonempty and duplicate-free")
    if any(not str(value).isdigit() for value in values):
        raise ValueError(f"{source} must contain numeric ValueCode strings")


def _membership_error(name: str, expected: set[str], observed: set[str]) -> str:
    return (
        f"{name} membership changed; missing="
        f"{sorted(expected - observed, key=int)}, extra="
        f"{sorted(observed - expected, key=int)}"
    )


def _require(frame: pl.DataFrame, columns: set[str], source: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON object: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact_metadata(path: Path, rows: int) -> dict[str, object]:
    return {
        "rows": int(rows),
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _frame_artifact_metadata(
    path: Path,
    frame: pl.DataFrame,
) -> dict[str, object]:
    return {
        **_artifact_metadata(path, frame.height),
        "columns": frame.width,
        "column_names": frame.columns,
    }


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
