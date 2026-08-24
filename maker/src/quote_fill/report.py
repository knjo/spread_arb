"""Support-aware product/action research table for the WP02--03 pilot.

The table deliberately stops before executable exit replay and pathwise EV.
It combines D-1-safe adaptive distances, raw entry-fill outcomes, the 50 ms
entry hedge diagnostic, and raw-full-fill-conditional latent first passage.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import polars as pl


DEFAULT_QUOTE_FILL_DIR = Path(__file__).resolve().parents[2] / "data" / "quote_fill"
DEFAULT_SNAPSHOT_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "quote_width"
    / "adaptive"
    / "adaptive_parameter_snapshot_by_day_symbol.csv"
)
GROUP_KEY = ["ValueCode", "route", "boundary_quantile"]


def build_product_action_research_table(
    order_aliases: pl.DataFrame,
    adaptive_snapshot: pl.DataFrame,
    hedge_facts: pl.DataFrame,
    latent_labels: pl.DataFrame,
) -> pl.DataFrame:
    """Build one descriptive row per product, entry route, and adaptive q.

    ``nominal_latent_band_bp`` is a basis-mid geometry diagnostic from the
    rounded entry target to the frozen D-1 lower.  It is not four-leg cash PnL.
    Hedge facts are unique by raw order, then expanded only to the q aliases
    that actually shared that raw fill.
    """

    _require(
        order_aliases,
        {
            "Date", "ValueCode", "QuoteCode", "route", "boundary_quantile",
            "raw_order_fact_id", "policy_generation_id", "threshold_basis_bp",
            "effective_basis_bp", "full_fill", "partial_fill", "cancel_required",
        },
        "order aliases",
    )
    _require(
        adaptive_snapshot,
        {
            "Date", "ValueCode", "QuoteCode", "boundary_quantile",
            "upper_distance_bp", "lower_distance_bp", "adaptive_parameter_valid",
            "contains_target_day_outcome",
        },
        "adaptive snapshot",
    )
    _require(
        hedge_facts,
        {
            "raw_order_fact_id", "status", "signed_total_slippage_bp",
            "decision_book_age_ms", "depth_shortfall",
        },
        "hedge facts",
    )
    _require(
        latent_labels,
        {
            "ValueCode", "route", "boundary_quantile", "policy_generation_id",
            "target_id", "observation_delay_seconds", "status",
            "time_to_latent_hit_seconds",
        },
        "latent labels",
    )

    unsafe = adaptive_snapshot.filter(
        pl.col("contains_target_day_outcome") != False  # noqa: E712
    )
    if unsafe.height:
        raise ValueError("adaptive input contains target-day outcomes")
    # Invalid product-days may exist in the complete snapshot, but aliases may
    # only join valid, D-1-safe rows.
    safe_snapshot = adaptive_snapshot.filter(
        (pl.col("contains_target_day_outcome") == False)  # noqa: E712
        & (pl.col("adaptive_parameter_valid") == True)  # noqa: E712
    ).select(
        "Date", "ValueCode", "QuoteCode", "boundary_quantile",
        "upper_distance_bp", "lower_distance_bp",
    )
    snapshot_key = ["Date", "ValueCode", "QuoteCode", "boundary_quantile"]
    if safe_snapshot.select(snapshot_key).n_unique() != safe_snapshot.height:
        raise ValueError("adaptive snapshot contains duplicate valid keys")

    aliases = order_aliases.join(
        safe_snapshot,
        on=snapshot_key,
        how="left",
        validate="m:1",
    )
    missing = aliases.filter(
        pl.col("upper_distance_bp").is_null()
        | pl.col("lower_distance_bp").is_null()
    )
    if missing.height:
        raise ValueError("order aliases do not all resolve to a valid D-1 snapshot")
    aliases = aliases.with_columns(
        (
            pl.col("effective_basis_bp")
            - (pl.col("threshold_basis_bp") - pl.col("upper_distance_bp"))
        ).alias("effective_open_distance_bp")
    ).with_columns(
        (
            pl.col("effective_open_distance_bp") + pl.col("lower_distance_bp")
        ).alias("nominal_latent_band_bp")
    )

    base = aliases.group_by(GROUP_KEY).agg(
        pl.col("Date").n_unique().alias("dates"),
        pl.len().alias("orders"),
        pl.col("raw_order_fact_id").n_unique().alias("unique_raw_order_facts"),
        pl.col("full_fill").fill_null(False).sum().alias("full_fills"),
        pl.col("partial_fill").fill_null(False).sum().alias("partial_fills"),
        pl.col("cancel_required").fill_null(False).sum().alias("cancel_required"),
        pl.col("upper_distance_bp").median().alias("prior_upper_distance_bp_p50"),
        pl.col("lower_distance_bp").median().alias("prior_lower_distance_bp_p50"),
        pl.col("effective_open_distance_bp").median().alias(
            "rounded_effective_open_bp_p50"
        ),
        pl.col("effective_open_distance_bp").quantile(0.95).alias(
            "rounded_effective_open_bp_p95"
        ),
        pl.col("nominal_latent_band_bp").median().alias(
            "nominal_latent_band_bp_p50"
        ),
    ).with_columns(
        (pl.col("full_fills") / pl.col("orders")).alias("p_full_fill"),
        (pl.col("cancel_required") / pl.col("orders")).alias("p_cancel_required"),
    )

    raw_ids = hedge_facts.select("raw_order_fact_id").unique()
    if raw_ids.height != hedge_facts.height:
        raise ValueError("hedge facts must be unique by raw_order_fact_id")
    full_aliases = aliases.filter(pl.col("full_fill") == True)  # noqa: E712
    hedge_aliases = full_aliases.select(
        *GROUP_KEY, "policy_generation_id", "raw_order_fact_id"
    ).join(hedge_facts, on="raw_order_fact_id", how="left", validate="m:1")
    hedge = hedge_aliases.group_by(GROUP_KEY).agg(
        pl.len().alias("full_fill_aliases_for_hedge"),
        (pl.col("status") == "executable").sum().alias(
            "hedge_depth_priceable_aliases"
        ),
        pl.col("signed_total_slippage_bp").quantile(0.95).alias(
            "entry_hedge_total_slippage_bp_p95"
        ),
        (pl.col("decision_book_age_ms") <= 100).sum().alias(
            "entry_hedge_book_le100ms"
        ),
        (pl.col("decision_book_age_ms") <= 1000).sum().alias(
            "entry_hedge_book_le1000ms"
        ),
        (pl.col("depth_shortfall") > 0).sum().alias("entry_hedge_depth_shortfall"),
    ).with_columns(
        (
            pl.col("entry_hedge_book_le100ms")
            / pl.col("full_fill_aliases_for_hedge")
        ).alias("p_entry_hedge_book_le100ms"),
        (
            pl.col("entry_hedge_book_le1000ms")
            / pl.col("full_fill_aliases_for_hedge")
        ).alias("p_entry_hedge_book_le1000ms"),
    )

    latent = latent_labels.filter(
        (pl.col("observation_delay_seconds") == 30)
        & pl.col("target_id").is_in(["frozen_center", "frozen_adaptive_lower"])
    )
    duplicate_label = (
        latent.group_by("policy_generation_id", "target_id")
        .len()
        .filter(pl.col("len") != 1)
    )
    if duplicate_label.height:
        raise ValueError("latent labels contain duplicate policy/target rows")
    latent_parts: list[pl.DataFrame] = []
    for target_id, prefix in (
        ("frozen_center", "frozen_center"),
        ("frozen_adaptive_lower", "frozen_lower"),
    ):
        selected = latent.filter(pl.col("target_id") == target_id)
        hit = pl.col("status") == "hit"
        latent_parts.append(
            selected.group_by(GROUP_KEY).agg(
                pl.len().alias(f"{prefix}_full_fill_aliases"),
                hit.sum().alias(f"{prefix}_hits"),
                (pl.col("status") == "censor").sum().alias(f"{prefix}_censors"),
                pl.col("time_to_latent_hit_seconds")
                .filter(hit)
                .median()
                .alias(f"{prefix}_time_s_p50"),
            ).with_columns(
                (
                    pl.col(f"{prefix}_hits")
                    / pl.col(f"{prefix}_full_fill_aliases")
                ).alias(f"p_{prefix}_given_full_fill")
            )
        )

    result = base.join(hedge, on=GROUP_KEY, how="left", validate="1:1")
    for part in latent_parts:
        result = result.join(part, on=GROUP_KEY, how="left", validate="1:1")
    return result.with_columns(
        pl.lit("product_route_adaptive_boundary_pilot").alias("research_layer"),
        pl.lit(False).alias("joint_volume_allocated"),
        pl.lit(False).alias("executable_exit_included"),
        pl.lit(False).alias("fees_tax_overnight_included"),
        pl.lit(False).alias("ev_ready"),
    ).sort(GROUP_KEY)


def run_product_action_report(
    *,
    quote_fill_dir: Path = DEFAULT_QUOTE_FILL_DIR,
    adaptive_snapshot_path: Path = DEFAULT_SNAPSHOT_PATH,
    output_path: Path | None = None,
) -> pl.DataFrame:
    """Read persisted pilot facts and write the product action table."""

    quote_fill_dir = Path(quote_fill_dir)
    snapshot = pl.read_csv(
        adaptive_snapshot_path,
        schema_overrides={
            "Date": pl.String,
            "ValueCode": pl.String,
            "QuoteCode": pl.String,
            "source_asof_date": pl.String,
        },
    )
    result = build_product_action_research_table(
        pl.read_parquet(quote_fill_dir / "order_aliases.parquet"),
        snapshot,
        pl.read_parquet(quote_fill_dir / "hedge_facts.parquet"),
        pl.read_parquet(quote_fill_dir / "latent_exit_opportunity_labels.parquet"),
    )
    destination = output_path or quote_fill_dir / "product_action_research_table.csv"
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    result.write_csv(destination)
    return result


def _require(frame: pl.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build WP02--03 product action table")
    parser.add_argument("--quote-fill-dir", type=Path, default=DEFAULT_QUOTE_FILL_DIR)
    parser.add_argument("--adaptive-snapshot", type=Path, default=DEFAULT_SNAPSHOT_PATH)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        run_product_action_report(
            quote_fill_dir=args.quote_fill_dir,
            adaptive_snapshot_path=args.adaptive_snapshot,
            output_path=args.output,
        )
    )


if __name__ == "__main__":
    main()
