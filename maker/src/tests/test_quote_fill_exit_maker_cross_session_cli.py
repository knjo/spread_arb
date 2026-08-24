from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import polars as pl

from maker.src.quote_fill.exit_maker_cross_session_cli import (
    _load_product_days,
    _load_sessions,
    _verify_formal_prerequisite_binding,
    main,
    parse_args,
)
from maker.src.quote_fill.exit_maker_cross_session_runner import (
    CROSS_SESSION_PREREQUISITE_BINDING_VERSION,
    _canonical_sha256,
    _file_sha256,
)


def _metadata(path: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }
    if path.suffix == ".parquet":
        result["rows"] = pl.read_parquet(path).height
        result["columns"] = len(pl.read_parquet_schema(path))
    return result


def _write_prerequisite(
    root: Path,
) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path]:
    prerequisite = root / "prerequisite"
    metadata_root = prerequisite / "metadata"
    metadata_root.mkdir(parents=True)
    sessions = prerequisite / "candidate_sessions.txt"
    sessions.write_text("20260609\n20260610\n", encoding="utf-8")
    calendar = prerequisite / "exact_contract_calendar_v1.parquet"
    pl.DataFrame(
        {
            "QuoteCode": ["CZFF6"],
            "expiry_session": ["20260617"],
            "calendar_version": ["test-v1"],
        }
    ).write_parquet(calendar)
    product_days = root / "product-days.parquet"
    pl.DataFrame(
        {"Date": ["20260609"], "ValueCode": ["2603"]}
    ).write_parquet(product_days)
    entry_root = root / "entry"
    data_root = root / "hft"
    futures_root = root / "future"
    for candidate_date in ("20260609", "20260610"):
        pl.DataFrame(
            {"QuoteCode": ["CZFF6"], "ValueCode": ["2603"]}
        ).write_parquet(
            metadata_root / f"{candidate_date}_contracts.parquet"
        )
    frames = {
        "candidate_requirements.parquet": pl.DataFrame(
            {
                "candidate_date": ["20260610"],
                "ValueCode": ["2603"],
                "QuoteCode": ["CZFF6"],
            }
        ),
        "entry_source_lineage.parquet": pl.DataFrame(
            {"Date": ["20260609"], "ValueCode": ["2603"]}
        ),
        "metadata_manifest.parquet": pl.DataFrame(
            {"Date": ["20260609", "20260610"]}
        ),
    }
    for name, frame in frames.items():
        frame.write_parquet(prerequisite / name)
    artifacts = {
        sessions.name: _metadata(sessions),
        calendar.name: _metadata(calendar),
        **{
            name: _metadata(prerequisite / name)
            for name in frames
        },
    }
    metadata_artifacts = {
        path.name: _metadata(path)
        for path in sorted(metadata_root.glob("*_contracts.parquet"))
    }
    config = {
        "product_days_path": str(product_days.resolve()),
        "product_days_sha256": _file_sha256(product_days),
        "entry_execution_root": str(entry_root.resolve()),
        "data_root": str(data_root.resolve()),
        "futures_raw_root": str(futures_root.resolve()),
    }
    marker = {
        "complete": True,
        "schema_version": "cross_session_prerequisites_v1",
        "config": config,
        "config_sha256": _canonical_sha256(config),
        "session_count": 2,
        "metadata_session_count": 2,
        "candidate_requirement_count": 1,
        "extension_requirement_count": 1,
        "artifacts": artifacts,
        "metadata_artifacts": metadata_artifacts,
        "fact_semantics": {
            "entry_cohort_unchanged": True,
            "candidate_calendar_extended_only": True,
            "point_in_time_contract_metadata": True,
            "exact_quote_code_no_roll": True,
            "extension_raw_sources_content_hashed": True,
            "extension_market_reference_validated": True,
        },
    }
    (prerequisite / "complete.json").write_text(
        json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8"
    )
    return (
        prerequisite,
        sessions,
        calendar,
        product_days,
        metadata_root,
        entry_root,
        data_root,
        futures_root,
    )


def _formal_args(
    *,
    prerequisite: Path,
    sessions: Path,
    calendar: Path,
    product_days: Path,
    metadata_root: Path,
    entry_root: Path,
    data_root: Path,
    futures_root: Path,
    output_root: Path,
    cache_root: Path,
) -> list[str]:
    return [
        "--prerequisite-root",
        str(prerequisite),
        "--sessions",
        str(sessions),
        "--contract-calendar",
        str(calendar),
        "--product-days",
        str(product_days),
        "--entry-root",
        str(entry_root),
        "--exit-root",
        str(output_root.parent / "exit"),
        "--output",
        str(output_root),
        "--data-root",
        str(data_root),
        "--futures-root",
        str(futures_root),
        "--contract-metadata-root",
        str(metadata_root),
        "--candidate-cache-root",
        str(cache_root),
    ]


class CrossSessionExitMakerCliTest(unittest.TestCase):
    def test_help_is_side_effect_free_and_documents_both_semantics(self) -> None:
        output = io.StringIO()
        with self.assertRaises(SystemExit) as raised, redirect_stdout(output):
            main(["--help"])
        self.assertEqual(raised.exception.code, 0)
        rendered = output.getvalue()
        self.assertIn("--product-days", rendered)
        self.assertIn("--product-day", rendered)
        self.assertIn("--symbols", rendered)
        self.assertIn("publish strict", rendered)
        self.assertIn("plus nominal-instant-cancel-V0", rendered)
        self.assertIn("diagnostic threshold only", rendered)
        self.assertIn("--candidate-cache-root", rendered)
        self.assertIn("--no-candidate-cache", rendered)
        self.assertIn("--prerequisite-root", rendered)

    def test_loaders_reject_ambiguous_sessions_and_product_days(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sessions = root / "sessions.txt"
            sessions.write_text("20260610\n20260609\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ascending"):
                _load_sessions(sessions)

            product_days = root / "product_days.csv"
            pl.DataFrame(
                {
                    "Date": ["20260609", "20260609"],
                    "ValueCode": ["2603", "2603"],
                }
            ).write_csv(product_days)
            with self.assertRaisesRegex(ValueError, "unique"):
                _load_product_days(product_days)

    def test_missing_or_tampered_prerequisite_fails_before_any_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "output"
            cache_root = root / "cache"
            replay = Mock()
            missing_args = _formal_args(
                prerequisite=root / "missing-prerequisite",
                sessions=root / "sessions.txt",
                calendar=root / "calendar.parquet",
                product_days=root / "product-days.parquet",
                metadata_root=root / "metadata",
                entry_root=root / "entry",
                data_root=root / "hft",
                futures_root=root / "future",
                output_root=output_root,
                cache_root=cache_root,
            )
            with patch(
                "maker.src.quote_fill.exit_maker_cross_session_cli."
                "run_cross_session_exit_replay",
                replay,
            ):
                with self.assertRaisesRegex(
                    FileExistsError, "prerequisite is incomplete"
                ):
                    main(missing_args)
            replay.assert_not_called()
            self.assertFalse(output_root.exists())
            self.assertFalse(cache_root.exists())

            (
                prerequisite,
                sessions,
                calendar,
                product_days,
                metadata_root,
                entry_root,
                data_root,
                futures_root,
            ) = _write_prerequisite(root)
            target = metadata_root / "20260610_contracts.parquet"
            target.write_bytes(target.read_bytes() + b"tamper")
            tampered_args = _formal_args(
                prerequisite=prerequisite,
                sessions=sessions,
                calendar=calendar,
                product_days=product_days,
                metadata_root=metadata_root,
                entry_root=entry_root,
                data_root=data_root,
                futures_root=futures_root,
                output_root=output_root,
                cache_root=cache_root,
            )
            with patch(
                "maker.src.quote_fill.exit_maker_cross_session_cli."
                "run_cross_session_exit_replay",
                replay,
            ):
                with self.assertRaisesRegex(
                    ValueError, "prerequisite artifact hash mismatch"
                ):
                    main(tampered_args)
            replay.assert_not_called()
            self.assertFalse(output_root.exists())
            self.assertFalse(cache_root.exists())

    def test_path_and_product_cohort_mismatch_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                prerequisite,
                sessions,
                calendar,
                product_days,
                metadata_root,
                entry_root,
                data_root,
                futures_root,
            ) = _write_prerequisite(root)
            alternate_sessions = root / "alternate-sessions.txt"
            alternate_sessions.write_bytes(sessions.read_bytes())
            with self.assertRaisesRegex(ValueError, "path mismatch: sessions"):
                _verify_formal_prerequisite_binding(
                    prerequisite,
                    sessions_path=alternate_sessions,
                    contract_calendar_path=calendar,
                    product_days_path=product_days,
                    entry_execution_root=entry_root,
                    data_root=data_root,
                    futures_raw_root=futures_root,
                    contract_metadata_root=metadata_root,
                )

            pl.DataFrame(
                {
                    "Date": ["20260609", "20260610"],
                    "ValueCode": ["2603", "2603"],
                }
            ).write_parquet(product_days)
            with self.assertRaisesRegex(ValueError, "cohort hash mismatch"):
                _verify_formal_prerequisite_binding(
                    prerequisite,
                    sessions_path=sessions,
                    contract_calendar_path=calendar,
                    product_days_path=product_days,
                    entry_execution_root=entry_root,
                    data_root=data_root,
                    futures_raw_root=futures_root,
                    contract_metadata_root=metadata_root,
                )

    def test_exact_product_day_dispatches_all_roots_and_shared_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                prerequisite,
                sessions,
                calendar,
                product_days,
                metadata_root,
                entry_root,
                data_root,
                futures_root,
            ) = _write_prerequisite(root)
            manifest = pl.DataFrame(
                {"Date": ["20260609"], "ValueCode": ["2603"]}
            )
            output = io.StringIO()
            with patch(
                "maker.src.quote_fill.exit_maker_cross_session_cli."
                "run_cross_session_exit_replay",
                return_value=manifest,
            ) as replay, redirect_stdout(output):
                status = main(
                    [
                        "--prerequisite-root",
                        str(prerequisite),
                        "--sessions",
                        str(sessions),
                        "--contract-calendar",
                        str(calendar),
                        "--product-days",
                        str(product_days),
                        "--entry-root",
                        str(entry_root),
                        "--exit-root",
                        str(root / "exit"),
                        "--output",
                        str(root / "output"),
                        "--data-root",
                        str(data_root),
                        "--futures-root",
                        str(futures_root),
                        "--contract-metadata-root",
                        str(metadata_root),
                        "--hedge-delay-ms",
                        "75",
                        "--book-age-diagnostic-ms",
                        "2500",
                        "--candidate-cache-root",
                        str(root / "candidate-cache"),
                        "--candidate-cache-max-entries",
                        "12",
                        "--no-candidate-cache",
                        "--no-resume",
                    ]
                )
            self.assertEqual(status, 0)
            replay.assert_called_once()
            positional, keywords = replay.call_args
            self.assertEqual(positional[0], (("20260609", "2603"),))
            self.assertEqual(keywords["sessions"], ("20260609", "20260610"))
            self.assertEqual(keywords["entry_execution_root"], root / "entry")
            self.assertEqual(keywords["exit_maker_root"], root / "exit")
            self.assertEqual(keywords["output_root"], root / "output")
            self.assertFalse(keywords["resume"])
            self.assertFalse(keywords["candidate_cache_enabled"])
            self.assertEqual(
                keywords["candidate_cache_root"], root / "candidate-cache"
            )
            self.assertEqual(keywords["candidate_cache_max_entries"], 12)
            config = keywords["config"]
            self.assertEqual(
                config.prerequisite_identity["binding_version"],
                CROSS_SESSION_PREREQUISITE_BINDING_VERSION,
            )
            self.assertEqual(
                config.prerequisite_identity["root"],
                str(prerequisite.resolve()),
            )
            self.assertEqual(config.hedge_delay_ns, 75_000_000)
            self.assertEqual(
                config.book_age_diagnostic_threshold_ns, 2_500_000_000
            )
            self.assertEqual(
                config.cross_config("strict").cancel_semantics, "strict"
            )
            self.assertEqual(
                config.cross_config(
                    "nominal_instant_cancel_v0"
                ).cancel_semantics,
                "nominal_instant_cancel_v0",
            )

    def test_parser_requires_exactly_one_universe_source(self) -> None:
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            parse_args([])
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            parse_args(["--product-days", "cohort.parquet"])
        parsed = parse_args(
            [
                "--prerequisite-root",
                "prerequisite",
                "--symbols",
                "2603,2317",
            ]
        )
        self.assertEqual(parsed.prerequisite_root, Path("prerequisite"))
        self.assertEqual(parsed.symbols, ("2603", "2317"))
        self.assertEqual(parsed.last_entry_sessions, 60)
        self.assertEqual(parsed.hedge_delay_ms, 50)
        self.assertEqual(parsed.book_age_diagnostic_ms, 1_000)


if __name__ == "__main__":
    unittest.main()
