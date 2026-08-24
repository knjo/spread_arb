"""CLI for frozen maker exits continued across sessions.

Every invocation publishes both the strict cancel-race interpretation and the
``nominal_instant_cancel_v0`` interpretation.  They are deliberately kept in
separate artifacts by the atomic runner; this CLI does not combine or select
between them.  Formal invocations are fail-closed on a verified, versioned
prerequisite root before any output, candidate-cache or raw-tape I/O.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

import polars as pl

from ..common.paths import HFT_DATA_ROOT, MAKER_ROOT
from .cross_session_prerequisite import verify_cross_session_prerequisites
from .exit_maker_cross_session_runner import (
    CROSS_SESSION_PREREQUISITE_BINDING_VERSION,
    DEFAULT_CONTRACT_METADATA_ROOT,
    DEFAULT_CROSS_SESSION_ROOT,
    DEFAULT_ENTRY_EXECUTION_ROOT,
    DEFAULT_EXIT_MAKER_ROOT,
    DEFAULT_FUTURES_RAW_ROOT,
    CrossSessionRunnerConfig,
    _canonical_sha256,
    _file_sha256,
    discover_cross_session_product_days,
    run_cross_session_exit_replay,
)


DEFAULT_SESSIONS_PATH = MAKER_ROOT / "data" / "walkforward" / "sessions.txt"
DEFAULT_CONTRACT_CALENDAR_PATH = (
    MAKER_ROOT / "data" / "walkforward" / "exact_contract_calendar_v1.parquet"
)
_PRODUCT_DAY = re.compile(r"^(?P<date>\d{8}):(?P<value>[A-Za-z0-9_.-]+)$")


def _parse_csv(value: str) -> tuple[str, ...]:
    parsed = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parsed or len(parsed) != len(set(parsed)):
        raise argparse.ArgumentTypeError("values must be nonempty and unique")
    return parsed


def _parse_product_day(value: str) -> tuple[str, str]:
    match = _PRODUCT_DAY.fullmatch(value.strip())
    if match is None:
        raise argparse.ArgumentTypeError(
            "product-day must have YYYYMMDD:ValueCode form"
        )
    return match.group("date"), match.group("value")


def _load_sessions(path: Path) -> tuple[str, ...]:
    sessions = tuple(
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if (
        not sessions
        or sessions != tuple(sorted(sessions))
        or len(sessions) != len(set(sessions))
        or any(len(value) != 8 or not value.isdigit() for value in sessions)
    ):
        raise ValueError(
            "sessions file must contain unique ascending YYYYMMDD values"
        )
    return sessions


def _read_table(path: Path, *, label: str) -> pl.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix == ".csv":
        return pl.read_csv(path)
    raise ValueError(f"unsupported {label} format for {path}; use parquet or csv")


def _load_product_days(path: Path) -> tuple[tuple[str, str], ...]:
    frame = _read_table(path, label="product-days")
    required = {"Date", "ValueCode"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"product-days table missing columns: {missing}")
    selected = frame.select(
        pl.col("Date").cast(pl.String, strict=True),
        pl.col("ValueCode").cast(pl.String, strict=True),
    )
    if selected.null_count().row(0) != (0, 0):
        raise ValueError("product-days Date and ValueCode cannot be null")
    keys = tuple(
        (str(row["Date"]), str(row["ValueCode"]))
        for row in selected.iter_rows(named=True)
    )
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("product-days must be nonempty and unique")
    return keys


def _verify_formal_prerequisite_binding(
    prerequisite_root: Path,
    *,
    sessions_path: Path,
    contract_calendar_path: Path,
    product_days_path: Path | None,
    entry_execution_root: Path,
    data_root: Path,
    futures_raw_root: Path,
    contract_metadata_root: Path,
) -> dict[str, object]:
    """Verify and bind every formal cross-session prerequisite input."""

    root = Path(prerequisite_root).resolve()
    payload = verify_cross_session_prerequisites(root)
    if product_days_path is None:
        raise ValueError(
            "prerequisite-bound formal CLI requires --product-days"
        )
    config = payload.get("config")
    artifacts = payload.get("artifacts")
    metadata_artifacts = payload.get("metadata_artifacts")
    if (
        not isinstance(config, dict)
        or not isinstance(artifacts, dict)
        or not isinstance(metadata_artifacts, dict)
    ):
        raise ValueError("verified prerequisite lineage is incomplete")

    def configured_path(name: str) -> Path:
        value = config.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"prerequisite config lacks {name}")
        return Path(value).resolve()

    expected_paths = {
        "sessions": root / "candidate_sessions.txt",
        "contract-calendar": root / "exact_contract_calendar_v1.parquet",
        "contract-metadata-root": root / "metadata",
        "product-days": configured_path("product_days_path"),
        "entry-root": configured_path("entry_execution_root"),
        "data-root": configured_path("data_root"),
        "futures-root": configured_path("futures_raw_root"),
    }
    supplied_paths = {
        "sessions": Path(sessions_path).resolve(),
        "contract-calendar": Path(contract_calendar_path).resolve(),
        "contract-metadata-root": Path(contract_metadata_root).resolve(),
        "product-days": Path(product_days_path).resolve(),
        "entry-root": Path(entry_execution_root).resolve(),
        "data-root": Path(data_root).resolve(),
        "futures-root": Path(futures_raw_root).resolve(),
    }
    mismatches = [
        name
        for name, expected in expected_paths.items()
        if supplied_paths[name] != expected.resolve()
    ]
    if mismatches:
        raise ValueError(
            "formal prerequisite path mismatch: " + ", ".join(mismatches)
        )

    product_days_sha = config.get("product_days_sha256")
    if (
        not isinstance(product_days_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", product_days_sha) is None
    ):
        raise ValueError("prerequisite product-days hash is invalid")
    product_path = supplied_paths["product-days"]
    if not product_path.is_file():
        raise FileNotFoundError(product_path)
    if _file_sha256(product_path) != product_days_sha:
        raise ValueError("formal product-days cohort hash mismatch")

    def artifact_sha256(name: str) -> str:
        metadata = artifacts.get(name)
        digest = metadata.get("sha256") if isinstance(metadata, dict) else None
        if not isinstance(digest, str) or re.fullmatch(
            r"[0-9a-f]{64}", digest
        ) is None:
            raise ValueError(f"prerequisite artifact identity missing: {name}")
        return digest

    source_identity = {
        "candidate_sessions": {
            "path": str(expected_paths["sessions"].resolve()),
            "sha256": artifact_sha256("candidate_sessions.txt"),
        },
        "contract_calendar": {
            "path": str(expected_paths["contract-calendar"].resolve()),
            "sha256": artifact_sha256(
                "exact_contract_calendar_v1.parquet"
            ),
        },
        "contract_metadata": {
            "root": str(expected_paths["contract-metadata-root"].resolve()),
            "manifest_sha256": artifact_sha256("metadata_manifest.parquet"),
            "artifacts_sha256": _canonical_sha256(metadata_artifacts),
        },
        "candidate_requirements_sha256": artifact_sha256(
            "candidate_requirements.parquet"
        ),
        "entry_source_lineage_sha256": artifact_sha256(
            "entry_source_lineage.parquet"
        ),
        "product_days": {
            "path": str(product_path),
            "sha256": product_days_sha,
        },
        "entry_execution_root": str(expected_paths["entry-root"]),
        "data_root": str(expected_paths["data-root"]),
        "futures_raw_root": str(expected_paths["futures-root"]),
    }
    marker_path = root / "complete.json"
    current_payload = json.loads(marker_path.read_text(encoding="utf-8"))
    if current_payload != payload:
        raise ValueError("prerequisite marker changed during verification")
    identity = {
        "binding_version": CROSS_SESSION_PREREQUISITE_BINDING_VERSION,
        "root": str(root),
        "schema_version": str(payload["schema_version"]),
        "marker_sha256": _file_sha256(marker_path),
        "marker_payload_sha256": _canonical_sha256(payload),
        "config_sha256": str(payload["config_sha256"]),
        "source_identity": source_identity,
        "source_identity_sha256": _canonical_sha256(source_identity),
    }
    return identity


def run(
    *,
    product_days: Sequence[tuple[str, str]],
    sessions: Sequence[str],
    contract_calendar: pl.DataFrame,
    entry_execution_root: Path = DEFAULT_ENTRY_EXECUTION_ROOT,
    exit_maker_root: Path = DEFAULT_EXIT_MAKER_ROOT,
    output_root: Path = DEFAULT_CROSS_SESSION_ROOT,
    hedge_delay_ns: int = 50_000_000,
    book_age_diagnostic_threshold_ns: int = 1_000_000_000,
    data_root: Path = HFT_DATA_ROOT,
    futures_raw_root: Path = DEFAULT_FUTURES_RAW_ROOT,
    contract_metadata_root: Path = DEFAULT_CONTRACT_METADATA_ROOT,
    resume: bool = True,
    candidate_cache_enabled: bool = True,
    candidate_cache_root: Path | None = None,
    candidate_cache_max_entries: int = 32,
    prerequisite_identity: Mapping[str, object],
) -> pl.DataFrame:
    """Run the atomic replay; strict and nominal outputs are always separate."""

    config = CrossSessionRunnerConfig(
        hedge_delay_ns=hedge_delay_ns,
        book_age_diagnostic_threshold_ns=book_age_diagnostic_threshold_ns,
        prerequisite_identity=dict(prerequisite_identity),
    )
    return run_cross_session_exit_replay(
        product_days,
        sessions=sessions,
        contract_calendar=contract_calendar,
        entry_execution_root=Path(entry_execution_root),
        exit_maker_root=Path(exit_maker_root),
        output_root=Path(output_root),
        config=config,
        data_root=Path(data_root),
        futures_raw_root=Path(futures_raw_root),
        contract_metadata_root=Path(contract_metadata_root),
        resume=resume,
        candidate_cache_enabled=candidate_cache_enabled,
        candidate_cache_root=candidate_cache_root,
        candidate_cache_max_entries=candidate_cache_max_entries,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continue frozen Center/Lower maker exits across sessions and "
            "publish strict plus nominal-instant-cancel-V0 artifacts"
        ),
        epilog=(
            "Formal prerequisite-bound runs require the exact --product-days "
            "cohort declared by the marker. Legacy direct-key or symbol "
            "selectors fail before source/output I/O. Both cancel semantics "
            "are always run."
        ),
    )
    parser.add_argument(
        "--prerequisite-root",
        type=Path,
        required=True,
        help="verified versioned root that binds every formal replay input",
    )
    parser.add_argument("--sessions", type=Path, default=DEFAULT_SESSIONS_PATH)
    parser.add_argument(
        "--contract-calendar",
        type=Path,
        default=DEFAULT_CONTRACT_CALENDAR_PATH,
    )
    universe = parser.add_mutually_exclusive_group(required=True)
    universe.add_argument(
        "--product-days",
        type=Path,
        help="CSV/parquet with exact Date and ValueCode columns",
    )
    universe.add_argument(
        "--product-day",
        action="append",
        type=_parse_product_day,
        help="legacy direct key; formal prerequisite-bound runs reject it",
    )
    universe.add_argument(
        "--symbols",
        type=_parse_csv,
        help="legacy discovery; formal prerequisite-bound runs reject it",
    )
    parser.add_argument("--last-entry-sessions", type=int, default=60)
    parser.add_argument(
        "--entry-root", type=Path, default=DEFAULT_ENTRY_EXECUTION_ROOT
    )
    parser.add_argument("--exit-root", type=Path, default=DEFAULT_EXIT_MAKER_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_CROSS_SESSION_ROOT)
    parser.add_argument("--data-root", type=Path, default=HFT_DATA_ROOT)
    parser.add_argument(
        "--futures-root", type=Path, default=DEFAULT_FUTURES_RAW_ROOT
    )
    parser.add_argument(
        "--contract-metadata-root",
        type=Path,
        default=DEFAULT_CONTRACT_METADATA_ROOT,
    )
    parser.add_argument("--hedge-delay-ms", type=int, default=50)
    parser.add_argument(
        "--book-age-diagnostic-ms",
        type=int,
        default=1_000,
        help="diagnostic threshold only; never a fill eligibility gate",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--candidate-cache-root",
        type=Path,
        help=(
            "derived exact-contract candidate cache; defaults to a sibling "
            "of --output"
        ),
    )
    parser.add_argument("--no-candidate-cache", action="store_true")
    parser.add_argument("--candidate-cache-max-entries", type=int, default=32)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    prerequisite_identity = _verify_formal_prerequisite_binding(
        args.prerequisite_root,
        sessions_path=args.sessions,
        contract_calendar_path=args.contract_calendar,
        product_days_path=args.product_days,
        entry_execution_root=args.entry_root,
        data_root=args.data_root,
        futures_raw_root=args.futures_root,
        contract_metadata_root=args.contract_metadata_root,
    )
    sessions = _load_sessions(args.sessions)
    if args.product_days is not None:
        keys = _load_product_days(args.product_days)
    elif args.product_day is not None:
        keys = tuple(args.product_day)
        if len(keys) != len(set(keys)):
            raise ValueError("product-day values must be unique")
    else:
        if args.last_entry_sessions <= 0 or len(sessions) < args.last_entry_sessions:
            raise ValueError("last-entry-sessions must be positive and available")
        keys, _availability = discover_cross_session_product_days(
            sessions[-args.last_entry_sessions :],
            args.symbols,
            entry_execution_root=args.entry_root,
            exit_maker_root=args.exit_root,
        )
        if not keys:
            raise ValueError(
                "none of the requested product-days has both complete inputs"
            )
    manifest = run(
        product_days=keys,
        sessions=sessions,
        contract_calendar=_read_table(
            args.contract_calendar, label="contract-calendar"
        ),
        entry_execution_root=args.entry_root,
        exit_maker_root=args.exit_root,
        output_root=args.output,
        hedge_delay_ns=args.hedge_delay_ms * 1_000_000,
        book_age_diagnostic_threshold_ns=(
            args.book_age_diagnostic_ms * 1_000_000
        ),
        data_root=args.data_root,
        futures_raw_root=args.futures_root,
        contract_metadata_root=args.contract_metadata_root,
        resume=not args.no_resume,
        candidate_cache_enabled=not args.no_candidate_cache,
        candidate_cache_root=args.candidate_cache_root,
        candidate_cache_max_entries=args.candidate_cache_max_entries,
        prerequisite_identity=prerequisite_identity,
    )
    if manifest.is_empty():
        print(manifest)
    else:
        print(manifest.select("Date", "ValueCode").sort(["Date", "ValueCode"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
