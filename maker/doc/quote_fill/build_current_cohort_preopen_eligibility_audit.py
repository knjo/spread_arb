"""Publish a hash-bound pre-open eligibility audit for the current cohort.

This deliberately does not modify or call the hard-cap controller.  It records
the static daily security flags and exact daily price limits for every
product-day selected by the canonical supplemental-v3 path population.  The
published gate action is a deployment contract only: non-normal product-days
must block new entries while already-held inventory remains exit-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

import polars as pl

from maker.src.common.paths import market_data_path


VERSION = "current_cohort_preopen_eligibility_audit_v1"
DEFAULT_SUPPLEMENTAL_ROOT = Path(
    "maker/data/walkforward/"
    "prequential_challenger_ab12_supplemental_carry_20260821_v3_expiry_daily_close"
)
DEFAULT_OUTPUT_ROOT = Path(
    "maker/data/walkforward/"
    "current_cohort_preopen_eligibility_audit_20260821_v1"
)
SUPPLEMENTAL_COMPLETE_SHA256 = (
    "402d050e6fed75c38cea540fd31dd09994f6a24ef91a5411ea4e715e14cb691d"
)
SUPPLEMENTAL_MARKER_PAYLOAD_SHA256 = (
    "173a98dab1d029b8d3e2f523eabf52f050c8b539a890f1b6aed05c7f71b6f337"
)
SUPPLEMENTAL_PATH_SHA256 = (
    "63742b773f32cdb4851fe686aea8c1dd8349d27b42be7460045dd73ec23cfda6"
)
SUPPLEMENTAL_PATH_ROWS = 3_672
EXPECTED_PRODUCT_DAYS = 711
EXPECTED_SESSION_DATES = 59
EXPECTED_PRODUCTS = 44

PATH_KEYS = ["Date", "ValueCode", "QuoteCode"]
RAW_COLUMNS = [
    "quote_code",
    "allow_day_trade_mark",
    "trading_method",
    "disposition_mark",
    "attention_mark",
    "limit_order_mark",
    "matching_interval",
    "opening_ref_price",
    "limit_up_price",
    "limit_down_price",
]
FILES = {
    "preopen_eligibility.parquet",
    "market_data_source_inventory.parquet",
    "gate_contract.parquet",
    "README.md",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _artifact_declaration(path: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if path.suffix == ".parquet":
        frame = pl.read_parquet(path)
        result.update(
            {
                "rows": frame.height,
                "columns": frame.width,
                "schema": {name: str(dtype) for name, dtype in frame.schema.items()},
            }
        )
    return result


def _load_verified_paths(root: Path) -> tuple[pl.DataFrame, dict[str, object]]:
    source_root = root.resolve()
    if root.is_symlink() or source_root != DEFAULT_SUPPLEMENTAL_ROOT.resolve():
        raise ValueError("supplemental-v3 source root identity changed")
    marker_path = source_root / "complete.json"
    paths_path = source_root / "supplemental_paths.parquet"
    if (
        marker_path.is_symlink()
        or paths_path.is_symlink()
        or _sha256(marker_path) != SUPPLEMENTAL_COMPLETE_SHA256
    ):
        raise ValueError("supplemental-v3 complete marker changed")
    marker = _read_json(marker_path)
    unhashed = dict(marker)
    payload_sha = unhashed.pop("marker_payload_sha256", None)
    declarations = marker.get("artifacts")
    declaration = (
        declarations.get("supplemental_paths.parquet")
        if isinstance(declarations, dict)
        else None
    )
    if (
        marker.get("complete") is not True
        or payload_sha != SUPPLEMENTAL_MARKER_PAYLOAD_SHA256
        or payload_sha != _canonical_sha256(unhashed)
        or not isinstance(declaration, dict)
        or declaration.get("sha256") != SUPPLEMENTAL_PATH_SHA256
        or int(declaration.get("rows", -1)) != SUPPLEMENTAL_PATH_ROWS
        or _sha256(paths_path) != SUPPLEMENTAL_PATH_SHA256
    ):
        raise ValueError("supplemental-v3 path declaration changed")
    paths = pl.read_parquet(
        paths_path,
        columns=[
            "Date",
            "ValueCode",
            "QuoteCode",
            "policy_path_id",
            "entry_spot_price",
        ],
    ).with_columns(
        pl.col("Date", "ValueCode", "QuoteCode", "policy_path_id").cast(pl.String),
        pl.col("entry_spot_price").cast(pl.Float64),
    )
    if (
        paths.height != SUPPLEMENTAL_PATH_ROWS
        or paths["policy_path_id"].n_unique() != paths.height
        or paths.select(PATH_KEYS).unique().height != EXPECTED_PRODUCT_DAYS
        or paths["Date"].n_unique() != EXPECTED_SESSION_DATES
        or paths["ValueCode"].n_unique() != EXPECTED_PRODUCTS
        or paths.filter(
            pl.col("entry_spot_price").is_null()
            | ~pl.col("entry_spot_price").is_finite()
            | (pl.col("entry_spot_price") <= 0)
        ).height
    ):
        raise ValueError("supplemental-v3 audit cohort changed")
    return paths.sort(PATH_KEYS + ["policy_path_id"]), marker


def _integer_mark(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.Float64).round(0).cast(pl.Int64).alias(column)


def _load_market_data_facts(
    cohort_keys: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    rows: list[pl.DataFrame] = []
    inventory: list[dict[str, object]] = []
    for date in sorted(cohort_keys["Date"].unique().to_list()):
        selected = cohort_keys.filter(pl.col("Date") == date).select("ValueCode")
        path = market_data_path(str(date)).resolve()
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"marketData source is missing or symlinked: {path}")
        schema = pl.read_parquet_schema(path)
        missing = sorted(set(RAW_COLUMNS) - set(schema))
        if missing:
            raise ValueError(f"{date} marketData fields missing: {missing}")
        raw_rows = pl.read_parquet(path).height
        frame = (
            pl.read_parquet(path, columns=RAW_COLUMNS)
            .with_columns(pl.col("quote_code").cast(pl.String).alias("ValueCode"))
            .join(selected, on="ValueCode", how="inner")
            .select(
                pl.lit(str(date), dtype=pl.String).alias("Date"),
                pl.col("ValueCode"),
                pl.col("allow_day_trade_mark")
                .cast(pl.String)
                .str.to_uppercase()
                .alias("day_trade_mark"),
                _integer_mark("trading_method"),
                _integer_mark("disposition_mark"),
                _integer_mark("attention_mark"),
                _integer_mark("limit_order_mark"),
                pl.col("matching_interval").cast(pl.String),
                pl.col("opening_ref_price").cast(pl.Float64),
                pl.col("limit_up_price").cast(pl.Float64),
                pl.col("limit_down_price").cast(pl.Float64),
            )
            .sort("ValueCode")
        )
        expected = selected.height
        if (
            frame.height != expected
            or frame["ValueCode"].n_unique() != expected
            or frame.filter(
                pl.any_horizontal(
                    pl.col(
                        "day_trade_mark",
                        "trading_method",
                        "disposition_mark",
                        "attention_mark",
                        "limit_order_mark",
                        "matching_interval",
                        "opening_ref_price",
                        "limit_up_price",
                        "limit_down_price",
                    ).is_null()
                )
            ).height
        ):
            raise ValueError(f"{date} marketData product-day mapping is not 1:1")
        rows.append(frame)
        inventory.append(
            {
                "Date": str(date),
                "source_path": str(path),
                "source_sha256": _sha256(path),
                "source_bytes": path.stat().st_size,
                "source_rows": raw_rows,
                "selected_product_days": expected,
                "market_data_content_hash_verified": True,
            }
        )
    facts = pl.concat(rows, how="vertical").sort(["Date", "ValueCode"])
    source_inventory = pl.from_dicts(
        inventory,
        schema={
            "Date": pl.String,
            "source_path": pl.String,
            "source_sha256": pl.String,
            "source_bytes": pl.Int64,
            "source_rows": pl.Int64,
            "selected_product_days": pl.Int64,
            "market_data_content_hash_verified": pl.Boolean,
        },
        strict=True,
    ).sort("Date")
    if (
        facts.height != EXPECTED_PRODUCT_DAYS
        or source_inventory.height != EXPECTED_SESSION_DATES
        or int(source_inventory["selected_product_days"].sum())
        != EXPECTED_PRODUCT_DAYS
    ):
        raise ValueError("marketData source inventory coverage changed")
    return facts, source_inventory


def _build_eligibility(paths: pl.DataFrame, facts: pl.DataFrame) -> pl.DataFrame:
    path_limit_checks = (
        paths.join(facts, on=["Date", "ValueCode"], how="left", validate="m:1")
        .with_columns(
            (pl.col("entry_spot_price") == pl.col("limit_up_price")).alias(
                "_at_upper"
            ),
            (pl.col("entry_spot_price") == pl.col("limit_down_price")).alias(
                "_at_lower"
            ),
            (pl.col("entry_spot_price") > pl.col("limit_up_price")).alias(
                "_above_upper"
            ),
            (pl.col("entry_spot_price") < pl.col("limit_down_price")).alias(
                "_below_lower"
            ),
        )
        .group_by(PATH_KEYS)
        .agg(
            pl.len().alias("candidate_path_count"),
            pl.col("entry_spot_price").min().alias("entry_spot_price_min"),
            pl.col("entry_spot_price").max().alias("entry_spot_price_max"),
            pl.col("_at_upper").sum().cast(pl.Int64).alias("entry_at_upper_count"),
            pl.col("_at_lower").sum().cast(pl.Int64).alias("entry_at_lower_count"),
            pl.col("_above_upper")
            .sum()
            .cast(pl.Int64)
            .alias("entry_above_upper_count"),
            pl.col("_below_lower")
            .sum()
            .cast(pl.Int64)
            .alias("entry_below_lower_count"),
        )
    )
    result = (
        facts.join(path_limit_checks, on=["Date", "ValueCode"], validate="1:1")
        .with_columns(
            pl.col("day_trade_mark").is_in(["X", "Y"]).alias(
                "day_trade_new_entry_eligible"
            ),
            (pl.col("trading_method") == 0).alias("trading_method_normal"),
            (pl.col("disposition_mark") == 0).alias("disposition_normal"),
            (pl.col("limit_order_mark") == 0).alias("limit_order_normal"),
            (pl.col("matching_interval") == "000").alias(
                "matching_interval_normal"
            ),
            (
                pl.col("opening_ref_price").is_finite()
                & pl.col("limit_up_price").is_finite()
                & pl.col("limit_down_price").is_finite()
                & (pl.col("limit_down_price") > 0)
                & (pl.col("limit_down_price") < pl.col("opening_ref_price"))
                & (pl.col("opening_ref_price") < pl.col("limit_up_price"))
            ).alias("daily_price_limit_fields_valid"),
            (
                (pl.col("entry_at_upper_count") == 0)
                & (pl.col("entry_at_lower_count") == 0)
                & (pl.col("entry_above_upper_count") == 0)
                & (pl.col("entry_below_lower_count") == 0)
            ).alias("selected_entry_prices_strictly_inside_daily_limits"),
        )
        .with_columns(
            pl.all_horizontal(
                "day_trade_new_entry_eligible",
                "trading_method_normal",
                "disposition_normal",
                "limit_order_normal",
                "matching_interval_normal",
                "daily_price_limit_fields_valid",
                "selected_entry_prices_strictly_inside_daily_limits",
            )
            .fill_null(False)
            .alias("new_entry_gate_pass")
        )
        .with_columns(
            pl.when(pl.col("new_entry_gate_pass"))
            .then(pl.lit("new_entry_allowed"))
            .otherwise(pl.lit("new_entry_blocked_held_exit_only"))
            .alias("gate_action"),
            pl.lit(True).alias("held_position_exit_allowed"),
            pl.lit(False).alias("runtime_controller_gate_integrated"),
        )
        .select(
            "Date",
            "ValueCode",
            "QuoteCode",
            "candidate_path_count",
            "day_trade_mark",
            "trading_method",
            "disposition_mark",
            "attention_mark",
            "limit_order_mark",
            "matching_interval",
            "opening_ref_price",
            "limit_up_price",
            "limit_down_price",
            "entry_spot_price_min",
            "entry_spot_price_max",
            "entry_at_upper_count",
            "entry_at_lower_count",
            "entry_above_upper_count",
            "entry_below_lower_count",
            "day_trade_new_entry_eligible",
            "trading_method_normal",
            "disposition_normal",
            "limit_order_normal",
            "matching_interval_normal",
            "daily_price_limit_fields_valid",
            "selected_entry_prices_strictly_inside_daily_limits",
            "new_entry_gate_pass",
            "gate_action",
            "held_position_exit_allowed",
            "runtime_controller_gate_integrated",
        )
        .sort(PATH_KEYS)
    )
    if (
        result.height != EXPECTED_PRODUCT_DAYS
        or result.select(PATH_KEYS).unique().height != result.height
        or int(result["candidate_path_count"].sum()) != SUPPLEMENTAL_PATH_ROWS
    ):
        raise ValueError("eligibility audit population changed")
    return result


def _build_contract(eligibility: pl.DataFrame) -> pl.DataFrame:
    blocked = eligibility.filter(~pl.col("new_entry_gate_pass"))
    return pl.from_dicts(
        [
            {
                "artifact_version": VERSION,
                "scope": "canonical_supplemental_v3_current_cohort_only",
                "allowed_day_trade_marks": "X,Y",
                "normal_trading_method": 0,
                "normal_disposition_mark": 0,
                "normal_limit_order_mark": 0,
                "normal_matching_interval": "000",
                "attention_mark_preserved_but_not_gated": True,
                "selected_entry_price_limit_equality_tolerance": 0.0,
                "non_normal_new_entry_action": "blocked",
                "non_normal_held_position_action": "exit_only",
                "runtime_controller_gate_integrated": False,
                "cohort_product_days": eligibility.height,
                "cohort_candidate_paths": int(
                    eligibility["candidate_path_count"].sum()
                ),
                "sample_blocked_product_days": blocked.height,
                "sample_blocked_candidate_paths": int(
                    blocked["candidate_path_count"].sum() or 0
                ),
                "sample_gate_numeric_impact_is_zero": blocked.is_empty(),
            }
        ],
        infer_schema_length=None,
    )


def _summary(eligibility: pl.DataFrame) -> dict[str, object]:
    blocked = eligibility.filter(~pl.col("new_entry_gate_pass"))
    return {
        "cohort_product_days": eligibility.height,
        "cohort_candidate_paths": int(eligibility["candidate_path_count"].sum()),
        "cohort_session_dates": eligibility["Date"].n_unique(),
        "cohort_products": eligibility["ValueCode"].n_unique(),
        "day_trade_X_product_days": eligibility.filter(
            pl.col("day_trade_mark") == "X"
        ).height,
        "day_trade_Y_product_days": eligibility.filter(
            pl.col("day_trade_mark") == "Y"
        ).height,
        "nonzero_trading_method_product_days": eligibility.filter(
            ~pl.col("trading_method_normal")
        ).height,
        "nonzero_disposition_product_days": eligibility.filter(
            ~pl.col("disposition_normal")
        ).height,
        "nonzero_limit_order_product_days": eligibility.filter(
            ~pl.col("limit_order_normal")
        ).height,
        "nonstandard_matching_interval_product_days": eligibility.filter(
            ~pl.col("matching_interval_normal")
        ).height,
        "nonzero_attention_product_days": eligibility.filter(
            pl.col("attention_mark") != 0
        ).height,
        "invalid_daily_price_limit_product_days": eligibility.filter(
            ~pl.col("daily_price_limit_fields_valid")
        ).height,
        "entry_at_upper_paths": int(eligibility["entry_at_upper_count"].sum()),
        "entry_at_lower_paths": int(eligibility["entry_at_lower_count"].sum()),
        "entry_above_upper_paths": int(eligibility["entry_above_upper_count"].sum()),
        "entry_below_lower_paths": int(eligibility["entry_below_lower_count"].sum()),
        "blocked_product_days": blocked.height,
        "blocked_candidate_paths": int(blocked["candidate_path_count"].sum() or 0),
        "sample_gate_numeric_impact_is_zero": blocked.is_empty(),
        "runtime_controller_gate_integrated": False,
    }


def _render_readme(summary: Mapping[str, object]) -> str:
    return "\n".join(
        [
            "# Current-cohort pre-open eligibility audit",
            "",
            "This is a source-bound audit of the 3,672 canonical supplemental-v3 candidate paths, grouped into 711 entry product-days. It is not a runtime controller integration and does not expand the retrospectively selected cohort.",
            "",
            "## Observed result",
            "",
            f"- Coverage: {summary['cohort_product_days']} product-days, {summary['cohort_candidate_paths']} candidate paths, {summary['cohort_session_dates']} entry dates, {summary['cohort_products']} products.",
            f"- Day-trade marks: X={summary['day_trade_X_product_days']}, Y={summary['day_trade_Y_product_days']}; no N/missing rows.",
            f"- Non-normal static flags: trading_method={summary['nonzero_trading_method_product_days']}, disposition={summary['nonzero_disposition_product_days']}, limit_order={summary['nonzero_limit_order_product_days']}, matching_interval={summary['nonstandard_matching_interval_product_days']} product-days.",
            f"- `attention_mark` is preserved for audit but is not interpreted as a gate: {summary['nonzero_attention_product_days']} product-days are nonzero.",
            f"- Exact daily-limit checks across selected entry paths: at upper={summary['entry_at_upper_paths']}, at lower={summary['entry_at_lower_paths']}, above upper={summary['entry_above_upper_paths']}, below lower={summary['entry_below_lower_paths']}.",
            f"- Gate impact on this sample: blocked product-days={summary['blocked_product_days']}, blocked candidate paths={summary['blocked_candidate_paths']}; therefore numeric impact is zero.",
            "",
            "## Gate contract",
            "",
            "A new entry is eligible only when day-trade mark is X/Y; trading_method, disposition_mark, and limit_order_mark are 0; matching_interval is `000`; exact daily price-limit fields are valid; and every selected historical entry price is strictly inside the exact daily lower/upper limits. Any non-normal product-day must block new entries while already-held inventory remains exit-only.",
            "",
            "This bundle only proves that applying that contract to the current 711 product-days changes zero rows. `runtime_controller_gate_integrated=false`: production must still wire the gate before admitting a new entry. `attention_mark` needs a separately sourced policy interpretation before it can become a blocker.",
            "",
            "`market_data_source_inventory.parquet` content-hashes all 59 raw daily MarketData files. `preopen_eligibility.parquet` preserves the row-level fields and exact limit checks. `gate_contract.parquet` records the counterfactual policy and zero-impact count.",
            "",
        ]
    )


def verify_bundle(root: Path, *, verify_sources: bool = True) -> dict[str, object]:
    bundle = root.resolve()
    if root.is_symlink() or not bundle.is_dir():
        raise ValueError("eligibility audit bundle is missing or symlinked")
    if {path.name for path in bundle.iterdir()} != {*FILES, "complete.json"}:
        raise ValueError("eligibility audit file set changed")
    marker = _read_json(bundle / "complete.json")
    unhashed = dict(marker)
    payload_sha = unhashed.pop("marker_payload_sha256", None)
    if (
        marker.get("complete") is not True
        or marker.get("version") != VERSION
        or payload_sha != _canonical_sha256(unhashed)
    ):
        raise ValueError("eligibility audit marker is invalid")
    declarations = marker.get("artifacts")
    if not isinstance(declarations, dict) or set(declarations) != FILES:
        raise ValueError("eligibility audit declarations changed")
    for filename, declaration in declarations.items():
        path = bundle / filename
        if (
            not isinstance(declaration, dict)
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != int(declaration.get("bytes", -1))
            or _sha256(path) != declaration.get("sha256")
        ):
            raise ValueError(f"eligibility artifact changed: {filename}")
        if path.suffix == ".parquet":
            frame = pl.read_parquet(path)
            if (
                frame.height != int(declaration.get("rows", -1))
                or frame.width != int(declaration.get("columns", -1))
                or {name: str(dtype) for name, dtype in frame.schema.items()}
                != declaration.get("schema")
            ):
                raise ValueError(f"eligibility artifact schema changed: {filename}")
    eligibility = pl.read_parquet(bundle / "preopen_eligibility.parquet")
    inventory = pl.read_parquet(bundle / "market_data_source_inventory.parquet")
    contract = pl.read_parquet(bundle / "gate_contract.parquet")
    summary = _summary(eligibility)
    if (
        summary != marker.get("summary")
        or eligibility.height != EXPECTED_PRODUCT_DAYS
        or int(eligibility["candidate_path_count"].sum()) != SUPPLEMENTAL_PATH_ROWS
        or inventory.height != EXPECTED_SESSION_DATES
        or contract.height != 1
        or contract.row(0, named=True)["sample_gate_numeric_impact_is_zero"]
        is not True
        or summary["blocked_product_days"] != 0
        or summary["blocked_candidate_paths"] != 0
        or summary["entry_at_upper_paths"] != 0
        or summary["entry_at_lower_paths"] != 0
        or summary["entry_above_upper_paths"] != 0
        or summary["entry_below_lower_paths"] != 0
    ):
        raise ValueError("eligibility audit reconciliation failed")
    if verify_sources:
        _load_verified_paths(DEFAULT_SUPPLEMENTAL_ROOT)
        for row in inventory.iter_rows(named=True):
            path = Path(str(row["source_path"]))
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != int(row["source_bytes"])
                or _sha256(path) != row["source_sha256"]
            ):
                raise ValueError(f"marketData source changed: {path}")
    return marker


def publish(output: Path, supplemental_root: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    paths, source_marker = _load_verified_paths(supplemental_root)
    facts, inventory = _load_market_data_facts(paths.select(PATH_KEYS).unique())
    eligibility = _build_eligibility(paths, facts)
    contract = _build_contract(eligibility)
    summary = _summary(eligibility)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        eligibility.write_parquet(stage / "preopen_eligibility.parquet")
        inventory.write_parquet(stage / "market_data_source_inventory.parquet")
        contract.write_parquet(stage / "gate_contract.parquet")
        (stage / "README.md").write_text(
            _render_readme(summary), encoding="utf-8"
        )
        declarations = {
            filename: _artifact_declaration(stage / filename)
            for filename in sorted(FILES)
        }
        source_unhashed = dict(source_marker)
        source_payload_sha = source_unhashed.pop("marker_payload_sha256")
        script_path = Path(__file__).resolve()
        marker: dict[str, object] = {
            "complete": True,
            "version": VERSION,
            "sources": {
                "supplemental_root": str(supplemental_root.resolve()),
                "supplemental_complete_sha256": SUPPLEMENTAL_COMPLETE_SHA256,
                "supplemental_marker_payload_sha256": source_payload_sha,
                "supplemental_paths_sha256": SUPPLEMENTAL_PATH_SHA256,
                "raw_market_data_files": inventory.height,
                "raw_market_data_content_hashes_verified": True,
            },
            "implementation": {
                "script_path": str(script_path),
                "script_sha256": _sha256(script_path),
            },
            "gate_semantics": {
                "non_normal_new_entry_action": "blocked",
                "non_normal_held_position_action": "exit_only",
                "attention_mark_preserved_but_not_gated": True,
                "exact_limit_equality_tolerance": 0.0,
                "runtime_controller_gate_integrated": False,
                "retrospective_current_cohort_only": True,
            },
            "summary": summary,
            "artifacts": declarations,
        }
        marker["marker_payload_sha256"] = _canonical_sha256(marker)
        (stage / "complete.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        verify_bundle(stage, verify_sources=True)
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--supplemental-root", type=Path, default=DEFAULT_SUPPLEMENTAL_ROOT
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.verify_only:
        marker = verify_bundle(args.output, verify_sources=True)
        print(json.dumps(marker["summary"], indent=2, sort_keys=True))
        return
    publish(args.output, args.supplemental_root)
    marker = verify_bundle(args.output, verify_sources=True)
    print(json.dumps(marker["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
