"""Publish supplemental carry v3 with same-day two-leg expiry closes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import polars as pl

from maker.src.quote_fill.supplemental_expiry_close import (
    OVERLAY_VERSION,
    TERMINAL_RESOLUTION,
    apply_expiry_daily_close_overlay,
)


RUNNER_VERSION = "supplemental_imputed_full_carry_grouped_runner_v3_expiry_daily_close"
UPSTREAM_RUNNER_VERSION = "supplemental_imputed_full_carry_grouped_runner_v2_trade_fallback"
FACT_VERSION = "expiry_daily_close_facts_v1"
DEFAULT_UPSTREAM = Path(
    "maker/data/walkforward/"
    "prequential_challenger_ab12_supplemental_carry_20260821_v2_trade_fallback"
)
DEFAULT_CLOSE_FACT_ROOT = Path(
    "maker/data/walkforward/expiry_daily_close_facts_20260821_v1"
)
DEFAULT_OUTPUT = Path(
    "maker/data/walkforward/"
    "prequential_challenger_ab12_supplemental_carry_20260821_v3_expiry_daily_close"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _verified_frames(
    root: Path, *, expected_version: str, filenames: Sequence[str]
) -> tuple[dict[str, object], dict[str, pl.DataFrame]]:
    marker_path = root / "complete.json"
    marker = _read_json(marker_path)
    version = marker.get("runner_version", marker.get("schema_version"))
    if marker.get("complete") is not True or version != expected_version:
        raise ValueError(f"unexpected or incomplete source bundle: {root}")
    declarations = marker.get("artifacts")
    if not isinstance(declarations, dict):
        raise ValueError(f"source bundle lacks artifact declarations: {root}")
    frames: dict[str, pl.DataFrame] = {}
    for filename in filenames:
        declaration = declarations.get(filename)
        path = root / filename
        if (
            not isinstance(declaration, dict)
            or not path.is_file()
            or _sha256(path) != declaration.get("sha256")
        ):
            raise ValueError(f"source artifact missing or changed: {path}")
        frame = pl.read_parquet(path)
        if (
            frame.height != int(declaration.get("rows", -1))
            or frame.width != int(declaration.get("columns", -1))
        ):
            raise ValueError(f"source artifact dimensions changed: {path}")
        frames[filename] = frame
    return marker, frames


def load_inputs(
    upstream_root: Path, close_fact_root: Path
) -> tuple[
    dict[str, object],
    dict[str, pl.DataFrame],
    dict[str, object],
    pl.DataFrame,
]:
    upstream_files = (
        "unresolved_entry_prices.parquet",
        "source_inventory.parquet",
        "continuation_terminals.parquet",
        "expiry_marks.parquet",
        "continuation_audit.parquet",
        "candidate_session_sampling_audit.parquet",
        "supplemental_paths.parquet",
    )
    upstream_marker, upstream = _verified_frames(
        upstream_root,
        expected_version=UPSTREAM_RUNNER_VERSION,
        filenames=upstream_files,
    )
    fact_marker, facts = _verified_frames(
        close_fact_root,
        expected_version=FACT_VERSION,
        filenames=("daily_close_facts.parquet",),
    )
    return (
        upstream_marker,
        upstream,
        fact_marker,
        facts["daily_close_facts.parquet"],
    )


def _artifact(path: Path) -> dict[str, object]:
    result: dict[str, object] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
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


def _render_readme(
    old_paths: pl.DataFrame,
    new_paths: pl.DataFrame,
    facts: pl.DataFrame,
    overlay_audit: pl.DataFrame,
) -> str:
    old_total = float(old_paths["gross_cycle_pnl_twd"].sum())
    new_total = float(new_paths["gross_cycle_pnl_twd"].sum())
    old_expiry = float(overlay_audit["legacy_gross_cycle_pnl_twd"].sum())
    new_expiry = float(overlay_audit["daily_close_gross_cycle_pnl_twd"].sum())
    fallback = overlay_audit.filter(
        (pl.col("ValueCode") == "3374") & (pl.col("QuoteCode") == "QLFG6")
    )
    fallback_row = fallback.row(0, named=True)
    return "\n".join(
        [
            "# Supplemental full carry v3: expiry daily-close overlay",
            "",
            f"- Full path population: {new_paths.height}/{new_paths.height} priced.",
            f"- Expiry paths repriced: {overlay_audit.height}; expiry pair marks: {facts.height}.",
            "- Only paths whose v2 selected terminal was an expiry mark are changed; source-completed and normal-continuation rows are unchanged.",
            "- Expiry spot and futures values both use their same-day `close_price`. Futures `settlement_price` is not used.",
            "- Daily close is a forced-flat accounting mark, not an executable BBO or a claim that this maker order filled at close.",
            "",
            "## Gross PnL impact before transaction costs",
            "",
            f"- Expiry paths, v2 BBO/trade mark: {old_expiry:,.2f} TWD.",
            f"- Expiry paths, v3 paired close: {new_expiry:,.2f} TWD.",
            f"- Expiry delta: {new_expiry - old_expiry:+,.2f} TWD.",
            f"- All 3,672 paths delta: {new_total - old_total:+,.2f} TWD ({old_total:,.2f} to {new_total:,.2f}).",
            "",
            "## Former 3374 / QLFG6 fallback",
            "",
            f"The five affected positions now use spot close={fallback_row['daily_spot_close_price']:.4f} and futures close={fallback_row['daily_future_close_price']:.4f}, replacing the v2 402/402 fallback.",
            "",
            "`daily_close_facts.parquet` binds both database close values and update timestamps; `expiry_close_overlay_audit.parquet` contains every path-level old/new price and gross delta.",
            "",
        ]
    )


def publish(
    output: Path,
    *,
    upstream_root: Path,
    close_fact_root: Path,
    upstream_marker: Mapping[str, object],
    fact_marker: Mapping[str, object],
    upstream: Mapping[str, pl.DataFrame],
    facts: pl.DataFrame,
) -> pl.DataFrame:
    if output.exists():
        raise FileExistsError(output)
    overlay = apply_expiry_daily_close_overlay(
        upstream["supplemental_paths.parquet"],
        upstream["expiry_marks.parquet"],
        upstream["continuation_audit.parquet"],
        facts,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    frames = {
        "unresolved_entry_prices.parquet": upstream["unresolved_entry_prices.parquet"],
        "source_inventory.parquet": upstream["source_inventory.parquet"],
        "continuation_terminals.parquet": upstream["continuation_terminals.parquet"],
        "expiry_marks.parquet": overlay.expiry_marks,
        "legacy_v2_expiry_marks.parquet": upstream["expiry_marks.parquet"],
        "continuation_audit.parquet": overlay.continuation_audit,
        "candidate_session_sampling_audit.parquet": upstream[
            "candidate_session_sampling_audit.parquet"
        ],
        "supplemental_paths.parquet": overlay.supplemental_paths,
        "daily_close_facts.parquet": facts,
        "expiry_close_overlay_audit.parquet": overlay.overlay_audit,
    }
    try:
        for filename, frame in frames.items():
            frame.write_parquet(stage / filename)
        (stage / "README.md").write_text(
            _render_readme(
                upstream["supplemental_paths.parquet"],
                overlay.supplemental_paths,
                facts,
                overlay.overlay_audit,
            ),
            encoding="utf-8",
        )
        inventory = {
            path.name: _artifact(path)
            for path in sorted(stage.iterdir())
            if path.is_file()
        }
        gross_delta = float(overlay.overlay_audit["gross_pnl_delta_twd"].sum())
        marker = {
            "complete": True,
            "runner_version": RUNNER_VERSION,
            "overlay_version": OVERLAY_VERSION,
            "analysis_only": True,
            "production_strategy_go": False,
            "model_imputed_full_carry_on_unknown": True,
            "double_exit_bias_possible": True,
            "state_sampling_approximate": True,
            "source_paths": overlay.supplemental_paths.height,
            "priced_paths": int(
                overlay.supplemental_paths["terminal_cashflow_priced"].sum()
            ),
            "expiry_terminal_resolution": TERMINAL_RESOLUTION,
            "expiry_paths_repriced": overlay.overlay_audit.height,
            "expiry_pair_marks": overlay.expiry_marks.height,
            "expiry_mark_is_official_close": True,
            "expiry_mark_is_official_settlement": False,
            "expiry_mark_is_executable_bbo": False,
            "future_settlement_price_used": False,
            "gross_pnl_delta_vs_v2_twd": gross_delta,
            "upstream_root": str(upstream_root.resolve()),
            "upstream_complete_sha256": _sha256(upstream_root / "complete.json"),
            "upstream_marker_payload_sha256": upstream_marker.get(
                "marker_payload_sha256"
            ),
            "close_fact_root": str(close_fact_root.resolve()),
            "close_fact_complete_sha256": _sha256(close_fact_root / "complete.json"),
            "close_fact_marker_payload_sha256": fact_marker.get(
                "marker_payload_sha256"
            ),
            "artifacts": inventory,
        }
        marker["marker_payload_sha256"] = _canonical_sha256(marker)
        (stage / "complete.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return overlay.supplemental_paths


def run(upstream_root: Path, close_fact_root: Path, output: Path) -> pl.DataFrame:
    upstream_marker, upstream, fact_marker, facts = load_inputs(
        upstream_root, close_fact_root
    )
    return publish(
        output,
        upstream_root=upstream_root,
        close_fact_root=close_fact_root,
        upstream_marker=upstream_marker,
        fact_marker=fact_marker,
        upstream=upstream,
        facts=facts,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--close-fact-root", type=Path, default=DEFAULT_CLOSE_FACT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    result = run(args.upstream_root, args.close_fact_root, args.output)
    expiry = result.filter(
        pl.col("supplemental_terminal_resolution") == TERMINAL_RESOLUTION
    )
    print(
        expiry.select(
            pl.len().alias("expiry_paths"),
            pl.col("gross_cycle_pnl_twd").sum().alias("expiry_gross_pnl_twd"),
        )
    )


if __name__ == "__main__":
    main()
