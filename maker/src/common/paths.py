"""Filesystem paths shared by maker research modules.

Pipeline-owned market inputs are resolved from the top-level
``config/pipeline.yaml`` through :mod:`src.pipeline_storage`.  Research outputs
remain under ``maker/data``.  Keeping those two roots separate prevents a
missing legacy ``HFT/data`` directory from silently redirecting large inputs
back to the system disk.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from src.pipeline_storage import PipelineStoragePaths, load_pipeline_storage

MAKER_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_ROOT = MAKER_ROOT.parent


def find_hft_root(start: Path | None = None) -> Path:
    """Locate HFT by its project/config contract, not a movable data folder."""

    origin = Path(__file__).resolve() if start is None else Path(start).resolve()
    search_root = origin if origin.is_dir() else origin.parent
    for candidate in (search_root, *search_root.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "config" / "pipeline.yaml"
        ).is_file():
            return candidate
    raise RuntimeError("could not locate the HFT project root")


HFT_ROOT = find_hft_root()
PIPELINE_STORAGE: PipelineStoragePaths = load_pipeline_storage(project_root=HFT_ROOT)
HFT_DATA_ROOT = PIPELINE_STORAGE.base_dir
LEGACY_HFT_DATA_ROOT = HFT_ROOT / "data"
MARKET_DATA_ROOT = PIPELINE_STORAGE.market_dir
SPOT_TICK_ROOT = PIPELINE_STORAGE.tick_dir
TICK_FEATURE_ROOT = PIPELINE_STORAGE.tick_feature_dir
MAKERFILL_ROOT = PIPELINE_STORAGE.maker_queue_dir
INDIVIDUAL_STOCK_FUTURES_ROOT = Path("/mnt/NAS/Parquet/Ticks")
INDIVIDUAL_STOCK_FUTURES_REQUIRED_MOUNT = Path("/mnt/NAS")
INPUT_PATH_RESOLUTION_POLICY_VERSION = "maker_input_resolution_v3_role_bound"
DEFAULT_OUTPUT_ROOT = MAKER_ROOT / "data" / "fair_mid"

_PIPELINE_FALLBACK_ROLES = frozenset({"spot_raw", "makerfill"})


def _require_absolute_normal_path(path: Path, *, label: str) -> None:
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    if ".." in path.parts:
        raise ValueError(f"{label} cannot contain parent traversal")


def _reject_symlink_components(path: Path, *, role: str) -> None:
    for component in (path, *path.parents):
        if component.is_symlink():
            raise FileNotFoundError(
                f"{role} input path cannot contain a symlink: {component}"
            )


def _validate_dated_filename(path: Path, *, role: str) -> None:
    suffix = {
        "spot_raw": "_StockTick.parquet",
        "makerfill": "_makerFill.parquet",
    }[role]
    if not path.name.endswith(suffix):
        raise ValueError(f"{role} filename differs from its exact contract")
    date = path.name[: -len(suffix)]
    try:
        parse_date(date)
    except ValueError as error:
        raise ValueError(
            f"{role} filename must contain a valid YYYYMMDD date"
        ) from error


def _pipeline_role_contract(role: str, filename: str) -> tuple[Path, Path]:
    if role == "spot_raw":
        canonical_root = PIPELINE_STORAGE.tick_dir
        legacy_root = LEGACY_HFT_DATA_ROOT / "tickData"
    elif role == "makerfill":
        canonical_root = PIPELINE_STORAGE.maker_queue_dir
        legacy_root = LEGACY_HFT_DATA_ROOT / "makerFill"
    else:  # pragma: no cover - guarded by the public resolver
        raise ValueError(f"unsupported pipeline fallback role: {role}")

    required_mount = PIPELINE_STORAGE.required_mount
    if required_mount is None:
        raise RuntimeError(f"pipeline storage for {role} requires an explicit mount")
    try:
        PIPELINE_STORAGE.base_dir.relative_to(required_mount)
        canonical_root.relative_to(PIPELINE_STORAGE.base_dir)
    except ValueError as error:
        raise RuntimeError(
            f"configured {role} root is outside pipeline SSD storage"
        ) from error
    return canonical_root / filename, legacy_root


def _validate_future_raw_contract(path: Path) -> None:
    try:
        relative = path.relative_to(INDIVIDUAL_STOCK_FUTURES_ROOT)
    except ValueError as error:
        raise ValueError(
            "future_raw must use the individual-stock-futures NAS root"
        ) from error
    if len(relative.parts) != 4 or relative.name != "stock_futures.parquet":
        raise ValueError("future_raw must use YYYY/MM/DD/stock_futures.parquet on NAS")
    year, month, day, _ = relative.parts
    try:
        parse_date(f"{year}{month}{day}")
    except ValueError as error:
        raise ValueError(
            "future_raw must use a valid YYYY/MM/DD NAS partition"
        ) from error
    try:
        INDIVIDUAL_STOCK_FUTURES_ROOT.relative_to(
            INDIVIDUAL_STOCK_FUTURES_REQUIRED_MOUNT
        )
    except ValueError as error:
        raise RuntimeError(
            "individual-stock-futures root is outside its required NAS mount"
        ) from error


def resolve_input_file(
    requested: Path,
    *,
    canonical: Path | None = None,
    role: str,
    legacy_parents: tuple[Path, ...] = (),
    canonical_required_mount: Path | None = None,
) -> Path:
    """Return one regular input under a role-specific source contract.

    Spot and makerFill may fall back from their exact historical HFT/data path
    only to the exact peer configured in pipeline.yaml.  Individual-stock
    futures never fall back and must use the NAS YYYY/MM/DD partition.  The
    returned path records the selected source naturally in downstream input
    manifests.
    """

    if not isinstance(requested, Path):
        raise TypeError("requested must be a Path")
    if not isinstance(role, str) or not role:
        raise ValueError("role must be a non-empty string")
    if any(not isinstance(parent, Path) for parent in legacy_parents):
        raise TypeError("legacy_parents must contain Paths")
    _require_absolute_normal_path(requested, label=f"{role} input")
    _reject_symlink_components(requested, role=role)

    if role == "future_raw":
        if canonical is not None or legacy_parents:
            raise ValueError("future_raw does not allow fallback paths")
        if canonical_required_mount is not None:
            raise ValueError("future_raw mount is fixed by its NAS contract")
        _validate_future_raw_contract(requested)
        validate_required_mount(
            INDIVIDUAL_STOCK_FUTURES_REQUIRED_MOUNT,
            role=role,
        )
        if not requested.is_file():
            raise FileNotFoundError(f"missing {role} input: {requested}")
        return requested

    if role in _PIPELINE_FALLBACK_ROLES:
        _validate_dated_filename(requested, role=role)
    if canonical is None:
        if requested.is_file():
            return requested
        raise FileNotFoundError(f"missing {role} input: {requested}")
    if role not in _PIPELINE_FALLBACK_ROLES:
        raise ValueError(f"fallback is not supported for role {role}")
    if not isinstance(canonical, Path):
        raise TypeError("canonical must be a Path or None")
    _require_absolute_normal_path(canonical, label=f"canonical {role} input")
    _reject_symlink_components(canonical, role=f"canonical {role}")
    _validate_dated_filename(canonical, role=role)
    if requested.name != canonical.name:
        raise ValueError("fallback requires an identical input filename")

    expected_canonical, allowed_legacy_parent = _pipeline_role_contract(
        role,
        requested.name,
    )
    if canonical != expected_canonical:
        raise ValueError(f"canonical {role} fallback must use pipeline.yaml storage")
    unexpected_legacy_parents = tuple(
        parent for parent in legacy_parents if parent != allowed_legacy_parent
    )
    if unexpected_legacy_parents:
        raise ValueError(f"{role} fallback contains an unapproved legacy root")
    configured_mount = PIPELINE_STORAGE.required_mount
    if (
        canonical_required_mount is not None
        and canonical_required_mount != configured_mount
    ):
        raise ValueError(f"canonical {role} mount differs from pipeline.yaml")
    if requested == canonical:
        validate_required_mount(configured_mount, role=role)
        if not canonical.is_file():
            raise FileNotFoundError(f"missing canonical {role} input: {canonical}")
        return canonical
    if (
        requested.parent != allowed_legacy_parent
        or requested.parent not in legacy_parents
    ):
        raise FileNotFoundError(
            f"missing custom {role} input; fallback is not allowed: {requested}"
        )
    if requested.is_file():
        return requested
    validate_required_mount(configured_mount, role=role)
    if not canonical.is_file():
        raise FileNotFoundError(
            f"missing {role} input; tried legacy {requested} and canonical {canonical}"
        )
    return canonical


def validate_required_mount(
    required_mount: Path | None,
    *,
    role: str,
    mount_checker: Callable[[Path], bool] | None = None,
) -> None:
    """Fail closed when a configured external input mount is unavailable."""

    if required_mount is None:
        return
    if not isinstance(required_mount, Path):
        raise TypeError("required_mount must be a Path or None")
    if not required_mount.is_absolute():
        raise ValueError("required_mount must be absolute")
    if ".." in required_mount.parts:
        raise ValueError("required_mount cannot contain parent traversal")
    try:
        _reject_symlink_components(required_mount, role=f"{role} mount")
    except FileNotFoundError as error:
        raise RuntimeError(
            f"required mount for {role} is unavailable: {required_mount}"
        ) from error
    checker = Path.is_mount if mount_checker is None else mount_checker
    if not required_mount.is_dir() or not checker(required_mount):
        raise RuntimeError(
            f"required mount for {role} is unavailable: {required_mount}"
        )


def parse_date(date: str) -> datetime:
    return datetime.strptime(date, "%Y%m%d").replace(tzinfo=UTC)


def spot_tick_path(date: str) -> Path:
    return SPOT_TICK_ROOT / f"{date}_StockTick.parquet"


def market_data_path(date: str) -> Path:
    return MARKET_DATA_ROOT / f"{date}_marketData.parquet"


def futures_raw_path(date: str) -> Path:
    parsed = parse_date(date)
    return (
        INDIVIDUAL_STOCK_FUTURES_ROOT
        / parsed.strftime("%Y")
        / parsed.strftime("%m")
        / parsed.strftime("%d")
        / "stock_futures.parquet"
    )
