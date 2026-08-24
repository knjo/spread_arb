"""Tests for the hash-bound retrospective research-universe manifest."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import polars as pl

from maker.src.quote_fill.liquidity import (
    LIQUIDITY_PUBLICATION_SCHEMA_VERSION,
    LiquidityScreenConfig,
    _liquidity_code_sha256,
)
from maker.src.quote_fill.universe_manifest import (
    EXPECTED_EXTENSION20,
    EXPECTED_SOURCE_PUBLICATION_VERSION,
    EXPECTED_STABLE68,
    EXPECTED_STRICT43,
    FIRST_WAVE_SYMBOLS_ARTIFACT,
    StabilityWindow,
    UNIVERSE_PUBLICATION_SCHEMA_VERSION,
    UniverseManifestConfig,
    UniverseSourceLineage,
    build_research_universe_manifest,
    load_verified_liquidity_sources,
    publish_research_universe_manifest,
)
from maker.src.quote_fill.targets import (
    FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE,
    PRICE_LADDER_VERSION,
)


ROUTES = ("future_ask_spot_taker", "spot_bid_future_taker")


def _config() -> UniverseManifestConfig:
    return UniverseManifestConfig(
        windows=(
            StabilityWindow("may", "20260101", "20260101", 1),
            StabilityWindow("june", "20260102", "20260102", 1),
            StabilityWindow("pseudo", "20260103", "20260103", 1),
        ),
        extension_window_name="pseudo",
        expected_stable_members=("1101", "1312", "2330"),
        expected_strict_members=("1101",),
        expected_extension_members=("1312",),
        selected_wide_controls=("6005",),
    )


def _lineage() -> UniverseSourceLineage:
    liquidity_config = asdict(LiquidityScreenConfig())
    source_lineage = {
        "boundary_file_sha256": "1" * 64,
        "boundary_marker_sha256": "2" * 64,
        "daily_marker_set_sha256": "3" * 64,
        "liquidity_code_sha256": _liquidity_code_sha256(),
        "config_sha256": _canonical_sha(liquidity_config),
        "price_ladder_version": PRICE_LADDER_VERSION,
        "future_one_dollar_tick_effective_date": (
            FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE
        ),
    }
    return UniverseSourceLineage(
        liquidity_root="/verified/liquidity",
        complete_sha256="a" * 64,
        schema_version=LIQUIDITY_PUBLICATION_SCHEMA_VERSION,
        marker_payload_sha256="e" * 64,
        publication_version=EXPECTED_SOURCE_PUBLICATION_VERSION,
        price_ladder_version=PRICE_LADDER_VERSION,
        future_one_dollar_tick_effective_date=(
            FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE
        ),
        latest_snapshot_date="20260103",
        rolling_screen_sha256="b" * 64,
        stable_core_sha256="c" * 64,
        config_sha256="d" * 64,
        source_lineage=source_lineage,
        declared_artifact_sha256={
            "rolling_liquidity_screen.parquet": "b" * 64,
            "pseudo_validation_stable_core_products.csv": "c" * 64,
            "config.json": "d" * 64,
        },
    )


def _row(
    date: str,
    value_code: str,
    route: str,
    *,
    status: str = "pass",
    tier: str = "core_candidate",
    wide: bool = False,
) -> dict[str, object]:
    return {
        "Date": date,
        "ValueCode": value_code,
        "boundary_quantile": 50,
        "route": route,
        "liquidity_gate_status": status,
        "replay_tier": tier,
        "wide_maker_spread_flag": wide,
        "liquidity_screen_version": EXPECTED_SOURCE_PUBLICATION_VERSION,
        "price_ladder_version": PRICE_LADDER_VERSION,
        "future_one_dollar_tick_effective_date": (
            FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE
        ),
        "execution_safe_snapshot": True,
        "contains_target_day_outcome": False,
    }


def _screen() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for date in ("20260101", "20260102", "20260103"):
        for route in ROUTES:
            # 1101 and quarantined 2330 qualify in every window.
            rows.extend(
                [
                    _row(date, "1101", route),
                    _row(date, "2330", route),
                    # 1312 is core only in the extension window.
                    _row(
                        date,
                        "1312",
                        route,
                        status="pass" if date == "20260103" else "known_fail",
                        tier=(
                            "core_candidate"
                            if date == "20260103"
                            else "known_fail"
                        ),
                    ),
                    # Pilot exists in source but is an explicit extra.
                    _row(
                        date,
                        "2303",
                        route,
                        status="known_fail",
                        tier="known_fail",
                    ),
                ]
            )
    # Wide control must be 1/1 pass and wide on the future-maker route.
    rows.extend(
        [
            _row(
                "20260103",
                "6005",
                "future_ask_spot_taker",
                tier="wide_maker_exploration",
                wide=True,
            ),
            _row(
                "20260103",
                "6005",
                "spot_bid_future_taker",
                tier="wide_hedge_cost_test",
            ),
        ]
    )
    return pl.from_dicts(rows, infer_schema_length=None)


def _stable() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ValueCode": ["1101", "1312", "2330"],
            "retrospective_pseudo_validation": [True, True, True],
            "production_universe_approved": [False, False, False],
        }
    )


class ResearchUniverseManifestTest(unittest.TestCase):
    def test_default_v2_memberships_are_exact(self) -> None:
        self.assertEqual(len(EXPECTED_STABLE68), 68)
        self.assertEqual(len(EXPECTED_STRICT43), 43)
        self.assertEqual(len(EXPECTED_EXTENSION20), 20)
        self.assertTrue(
            {"2308", "2404", "6274", "8046", "8299"}.isdisjoint(
                EXPECTED_STABLE68
            )
        )
        self.assertTrue({"2308", "8299"}.isdisjoint(EXPECTED_STRICT43))
        self.assertTrue({"2404", "6274"}.isdisjoint(EXPECTED_EXTENSION20))

    def test_exact_membership_quarantine_and_first_wave(self) -> None:
        result = build_research_universe_manifest(
            _screen(), _stable(), _lineage(), _config()
        )
        self.assertEqual(result.strict_core["ValueCode"].to_list(), ["1101"])
        self.assertEqual(result.extension["ValueCode"].to_list(), ["1312"])
        self.assertEqual(result.wide_controls["ValueCode"].to_list(), ["6005"])
        self.assertEqual(
            result.first_wave["ValueCode"].to_list(),
            ["1101", "2303", "6005"],
        )
        self.assertEqual(result.first_wave_csv_symbols, "1101,2303,6005")
        quarantine = result.manifest.filter(pl.col("quarantine")).row(
            0, named=True
        )
        self.assertEqual(quarantine["ValueCode"], "2330")
        self.assertTrue(quarantine["pre_quarantine_strict_core_member"])
        self.assertFalse(quarantine["execution_cli_member"])
        self.assertFalse(result.manifest["production_universe_approved"].any())
        self.assertTrue(result.manifest["retrospective_research_selection"].all())

    def test_duplicate_screen_key_fails_closed(self) -> None:
        screen = pl.concat([_screen(), _screen().head(1)])
        with self.assertRaisesRegex(ValueError, "duplicate product-day-route"):
            build_research_universe_manifest(
                screen, _stable(), _lineage(), _config()
            )

    def test_membership_drift_fails_closed(self) -> None:
        screen = _screen().with_columns(
            pl.when(pl.col("ValueCode") == "1101")
            .then(pl.lit("known_fail"))
            .otherwise(pl.col("liquidity_gate_status"))
            .alias("liquidity_gate_status")
        )
        with self.assertRaisesRegex(ValueError, "strict core membership changed"):
            build_research_universe_manifest(
                screen, _stable(), _lineage(), _config()
            )

    def test_stable_membership_drift_fails_closed_even_at_same_count(self) -> None:
        stable = _stable().with_columns(
            pl.when(pl.col("ValueCode") == "1312")
            .then(pl.lit("9999"))
            .otherwise(pl.col("ValueCode"))
            .alias("ValueCode")
        )
        with self.assertRaisesRegex(ValueError, "stable core membership changed"):
            build_research_universe_manifest(
                _screen(), stable, _lineage(), _config()
            )

    def test_price_ladder_lineage_mismatch_fails_closed(self) -> None:
        lineage = UniverseSourceLineage(
            **{
                **_lineage().__dict__,
                "price_ladder_version": "obsolete-ladder",
            }
        )
        with self.assertRaisesRegex(ValueError, "price-ladder lineage mismatch"):
            build_research_universe_manifest(
                _screen(), _stable(), lineage, _config()
            )

    def test_wide_control_must_pass_every_selected_session(self) -> None:
        screen = _screen().with_columns(
            pl.when(
                (pl.col("ValueCode") == "6005")
                & (pl.col("route") == "future_ask_spot_taker")
            )
            .then(pl.lit(False))
            .otherwise(pl.col("wide_maker_spread_flag"))
            .alias("wide_maker_spread_flag")
        )
        with self.assertRaisesRegex(ValueError, "must be pass and future-wide-maker"):
            build_research_universe_manifest(
                screen, _stable(), _lineage(), _config()
            )

    def test_atomic_publish_contains_comma_list_and_hashes(self) -> None:
        result = build_research_universe_manifest(
            _screen(), _stable(), _lineage(), _config()
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "manifest"
            publish_research_universe_manifest(
                result, _lineage(), output, _config()
            )
            self.assertEqual(
                (output / FIRST_WAVE_SYMBOLS_ARTIFACT)
                .read_text(encoding="utf-8")
                .strip(),
                "1101,2303,6005",
            )
            marker = json.loads((output / "complete.json").read_text())
            self.assertTrue(marker["complete"])
            self.assertEqual(
                marker["schema_version"],
                UNIVERSE_PUBLICATION_SCHEMA_VERSION,
            )
            declared_marker_sha = marker.pop("marker_payload_sha256")
            self.assertEqual(declared_marker_sha, _canonical_sha(marker))
            self.assertEqual(marker["counts"]["first_wave"], 3)
            self.assertEqual(
                marker["source"]["rolling_screen_sha256"], "b" * 64
            )
            manifest_meta = marker["artifacts"][
                "research_universe_manifest.parquet"
            ]
            self.assertEqual(manifest_meta["columns"], result.manifest.width)
            self.assertEqual(
                manifest_meta["column_names"], result.manifest.columns
            )
            with self.assertRaises(FileExistsError):
                publish_research_universe_manifest(
                    result, _lineage(), output, _config()
                )


class VerifiedLiquiditySourceTest(unittest.TestCase):
    def _source_root(self, root: Path) -> Path:
        source = root / "liquidity"
        source.mkdir()
        screen = _screen()
        stable = _stable()
        screen.write_parquet(source / "rolling_liquidity_screen.parquet")
        stable.write_csv(source / "pseudo_validation_stable_core_products.csv")
        config_contract = asdict(LiquidityScreenConfig())
        source_lineage = {
            "boundary_file_sha256": "1" * 64,
            "boundary_marker_sha256": "2" * 64,
            "daily_marker_set_sha256": "3" * 64,
            "liquidity_code_sha256": _liquidity_code_sha256(),
            "config_sha256": _canonical_sha(config_contract),
            "price_ladder_version": PRICE_LADDER_VERSION,
            "future_one_dollar_tick_effective_date": (
                FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE
            ),
        }
        (source / "config.json").write_text(
            json.dumps(
                {
                    **config_contract,
                    "latest_snapshot_date": "20260103",
                    "production_universe_approved": False,
                    **source_lineage,
                }
            ),
            encoding="utf-8",
        )
        artifacts: dict[str, dict[str, object]] = {}
        for path in source.iterdir():
            if path.name == "complete.json":
                continue
            metadata: dict[str, object] = {
                "bytes": path.stat().st_size,
                "sha256": _sha(path),
            }
            frame = (
                screen
                if path.name == "rolling_liquidity_screen.parquet"
                else (
                    stable
                    if path.name == "pseudo_validation_stable_core_products.csv"
                    else None
                )
            )
            if frame is not None:
                metadata.update(
                    {
                        "rows": frame.height,
                        "columns": frame.width,
                        "column_names": frame.columns,
                    }
                )
            artifacts[path.name] = metadata
        marker = {
            "complete": True,
            "schema_version": LIQUIDITY_PUBLICATION_SCHEMA_VERSION,
            "publication_version": EXPECTED_SOURCE_PUBLICATION_VERSION,
            "price_ladder_version": PRICE_LADDER_VERSION,
            "future_one_dollar_tick_effective_date": (
                FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE
            ),
            "config": config_contract,
            "source_lineage": source_lineage,
            "artifacts": artifacts,
        }
        marker["marker_payload_sha256"] = _canonical_sha(marker)
        (source / "complete.json").write_text(
            json.dumps(marker),
            encoding="utf-8",
        )
        return source

    def test_source_hashes_are_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = self._source_root(Path(temporary))
            bundle = load_verified_liquidity_sources(source)
            self.assertEqual(bundle.rolling_screen.height, _screen().height)
            self.assertEqual(
                bundle.lineage.publication_version,
                EXPECTED_SOURCE_PUBLICATION_VERSION,
            )
            self.assertEqual(
                bundle.lineage.price_ladder_version,
                PRICE_LADDER_VERSION,
            )
            self.assertEqual(
                bundle.lineage.future_one_dollar_tick_effective_date,
                FUTURE_ONE_DOLLAR_TICK_EFFECTIVE_DATE,
            )
            self.assertEqual(
                bundle.lineage.schema_version,
                LIQUIDITY_PUBLICATION_SCHEMA_VERSION,
            )

    def test_tampered_source_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = self._source_root(Path(temporary))
            with (source / "config.json").open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                load_verified_liquidity_sources(source)

    def test_obsolete_price_ladder_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = self._source_root(Path(temporary))
            marker_path = source / "complete.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            marker["price_ladder_version"] = "obsolete-ladder"
            marker.pop("marker_payload_sha256")
            marker["marker_payload_sha256"] = _canonical_sha(marker)
            marker_path.write_text(json.dumps(marker), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ladder lineage mismatch"):
                load_verified_liquidity_sources(source)

    def test_obsolete_future_tick_effective_date_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = self._source_root(Path(temporary))
            marker_path = source / "complete.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            marker["future_one_dollar_tick_effective_date"] = "20990101"
            marker.pop("marker_payload_sha256")
            marker["marker_payload_sha256"] = _canonical_sha(marker)
            marker_path.write_text(json.dumps(marker), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ladder lineage mismatch"):
                load_verified_liquidity_sources(source)

    def test_pre_completion_marker_is_rejected_before_artifact_io(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = self._source_root(Path(temporary))
            marker_path = source / "complete.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            marker.pop("schema_version")
            marker.pop("marker_payload_sha256")
            marker_path.write_text(json.dumps(marker), encoding="utf-8")
            # A pre-lineage marker must fail before this invalid raw artifact
            # can be opened or hashed as a trusted publication input.
            (source / "rolling_liquidity_screen.parquet").write_bytes(b"invalid")
            with self.assertRaisesRegex(ValueError, "schema mismatch"):
                load_verified_liquidity_sources(source)

    def test_liquidity_marker_self_hash_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = self._source_root(Path(temporary))
            marker_path = source / "complete.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            marker["publication_version"] = "tampered"
            marker_path.write_text(json.dumps(marker), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "self-hash mismatch"):
                load_verified_liquidity_sources(source)

    def test_stale_liquidity_code_lineage_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = self._source_root(Path(temporary))
            marker_path = source / "complete.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            marker["source_lineage"]["liquidity_code_sha256"] = "0" * 64
            marker.pop("marker_payload_sha256")
            marker["marker_payload_sha256"] = _canonical_sha(marker)
            marker_path.write_text(json.dumps(marker), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not current"):
                load_verified_liquidity_sources(source)

    def test_declared_artifact_shape_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = self._source_root(Path(temporary))
            marker_path = source / "complete.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            marker["artifacts"]["rolling_liquidity_screen.parquet"][
                "columns"
            ] += 1
            marker.pop("marker_payload_sha256")
            marker["marker_payload_sha256"] = _canonical_sha(marker)
            marker_path.write_text(json.dumps(marker), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact shape mismatch"):
                load_verified_liquidity_sources(source)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    unittest.main()
