"""Compare each allocated 13:00 aggressive exit with its frozen normal exit."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl


ROOT = Path(__file__).resolve().parents[3]
AGGRESSIVE = (
    ROOT
    / "maker/data/walkforward/aggressive_1300_exit_analysis_60d_20260821_v3_post13_timeline_fix/controller_position_outcomes.parquet"
)
NORMAL = (
    ROOT
    / "maker/data/walkforward/prequential_challenger_ab12_supplemental_carry_20260821_v3_expiry_daily_close/supplemental_paths.parquet"
)
SESSION_CALENDAR = (
    ROOT
    / "maker/data/walkforward/cross_session_prerequisites_v1_20260819/candidate_sessions.txt"
)


def load_comparison() -> pl.DataFrame:
    aggressive = (
        pl.read_parquet(AGGRESSIVE)
        .filter(
            pl.col("primary_conservative_result")
            & pl.col("portfolio_cap_twd").is_in([30_000_000.0, 40_000_000.0, 50_000_000.0])
            & pl.col("aggressive_close_allocated")
        )
        .select(
            "portfolio_cap_twd",
            "position_id",
            "policy_path_id",
            "Date",
            "normalization_notional_twd",
            "entry_spot_price",
            "entry_future_price",
            "entry_contract_size_shares",
            pl.col("scenario_terminal_date").alias("aggressive_terminal_date"),
            pl.col("scenario_exit_spot_price").alias("aggressive_exit_spot_price"),
            pl.col("scenario_exit_future_price").alias("aggressive_exit_future_price"),
            pl.col("scenario_gross_cycle_pnl_twd").alias("aggressive_gross_twd"),
            pl.col("scenario_transaction_cost_twd").alias("aggressive_cost_twd"),
            pl.col("scenario_net_cycle_pnl_twd").alias("aggressive_net_twd"),
        )
    )
    normal = pl.read_parquet(NORMAL).select(
        "policy_path_id",
        pl.col("terminal_date").alias("normal_terminal_date"),
        pl.col("exit_spot_price").alias("normal_exit_spot_price"),
        pl.col("exit_future_price").alias("normal_exit_future_price"),
        pl.col("gross_cycle_pnl_twd").alias("normal_gross_twd"),
        pl.col("holding_session_boundaries").alias("source_holding_session_boundaries"),
        "supplemental_terminal_resolution",
        "expiry_mark_is_official_close",
    )
    frame = aggressive.join(normal, on="policy_path_id", how="left", validate="m:1")

    sessions = SESSION_CALENDAR.read_text(encoding="utf-8").splitlines()
    session_index = {value: index for index, value in enumerate(sessions)}
    frame = frame.with_columns(
        pl.col("Date")
        .replace_strict(session_index, return_dtype=pl.Int64)
        .alias("entry_session_index"),
        pl.col("aggressive_terminal_date")
        .replace_strict(session_index, return_dtype=pl.Int64)
        .alias("aggressive_session_index"),
        pl.col("normal_terminal_date")
        .replace_strict(session_index, return_dtype=pl.Int64)
        .alias("normal_session_index"),
    ).with_columns(
        (pl.col("aggressive_session_index") - pl.col("entry_session_index")).alias(
            "aggressive_holding_boundaries"
        ),
        (pl.col("normal_session_index") - pl.col("entry_session_index")).alias(
            "normal_holding_boundaries"
        ),
        (pl.col("normal_session_index") - pl.col("aggressive_session_index")).alias(
            "holding_boundaries_avoided"
        ),
    )

    commission_rate = 1.71 / 10_000.0
    futures_tax_rate = 0.2 / 10_000.0
    same_day_tax_rate = 15.0 / 10_000.0
    overnight_tax_rate = 30.0 / 10_000.0
    shares = pl.col("entry_contract_size_shares")
    normal_same_day = pl.col("Date") == pl.col("normal_terminal_date")
    aggressive_same_day = pl.col("Date") == pl.col("aggressive_terminal_date")
    normal_tax_rate = pl.when(normal_same_day).then(same_day_tax_rate).otherwise(
        overnight_tax_rate
    )
    normal_tax = pl.col("normal_exit_spot_price") * shares * normal_tax_rate
    aggressive_tax_rate = pl.when(aggressive_same_day).then(
        same_day_tax_rate
    ).otherwise(overnight_tax_rate)
    aggressive_tax = (
        pl.col("aggressive_exit_spot_price") * shares * aggressive_tax_rate
    )
    normal_cost = (
        pl.col("entry_spot_price") * shares * commission_rate
        + pl.col("normal_exit_spot_price") * shares * commission_rate
        + normal_tax
        + pl.col("entry_future_price") * shares * futures_tax_rate
        + pl.col("normal_exit_future_price") * shares * futures_tax_rate
        + 40.0
    )
    aggressive_cost = (
        pl.col("entry_spot_price") * shares * commission_rate
        + pl.col("aggressive_exit_spot_price") * shares * commission_rate
        + aggressive_tax
        + pl.col("entry_future_price") * shares * futures_tax_rate
        + pl.col("aggressive_exit_future_price") * shares * futures_tax_rate
        + 40.0
    )

    result = frame.with_columns(
        normal_same_day.alias("normal_same_day"),
        aggressive_same_day.alias("aggressive_same_day"),
        pl.when(aggressive_same_day & (pl.col("normal_holding_boundaries") == 0))
        .then(pl.lit("same_day_vs_same_day"))
        .when(aggressive_same_day & (pl.col("normal_holding_boundaries") == 1))
        .then(pl.lit("same_day_vs_next_session"))
        .when(aggressive_same_day)
        .then(pl.lit("same_day_vs_later"))
        .otherwise(pl.lit("already_overnight_vs_later"))
        .alias("comparison_bucket"),
        normal_cost.alias("normal_cost_twd"),
        (pl.col("normal_gross_twd") - normal_cost).alias("normal_net_twd"),
        normal_tax.alias("normal_spot_sell_tax_twd"),
        aggressive_tax.alias("aggressive_spot_sell_tax_twd"),
        aggressive_cost.alias("aggressive_recomputed_cost_twd"),
    ).with_columns(
        (pl.col("aggressive_gross_twd") - pl.col("normal_gross_twd")).alias(
            "gross_delta_twd"
        ),
        (pl.col("normal_cost_twd") - pl.col("aggressive_cost_twd")).alias(
            "cost_saving_twd"
        ),
        (
            pl.col("normal_spot_sell_tax_twd")
            - pl.col("aggressive_spot_sell_tax_twd")
        ).alias("spot_sell_tax_saving_twd"),
        (pl.col("aggressive_net_twd") - pl.col("normal_net_twd")).alias(
            "net_delta_twd"
        ),
    )
    if result.height != 258 or result["policy_path_id"].null_count() != 0:
        raise ValueError("aggressive/normal counterfactual join is incomplete")
    cost_error = result.select(
        (pl.col("aggressive_recomputed_cost_twd") - pl.col("aggressive_cost_twd"))
        .abs()
        .max()
    ).item()
    identity_error = result.select(
        (
            pl.col("net_delta_twd")
            - pl.col("gross_delta_twd")
            - pl.col("cost_saving_twd")
        )
        .abs()
        .max()
    ).item()
    if float(cost_error) > 1e-8 or float(identity_error) > 1e-8:
        raise ValueError("counterfactual cost/P&L identity check failed")
    if result["holding_boundaries_avoided"].min() < 0:
        raise ValueError("aggressive exit occurs after its normal terminal")
    return result


def summarize(frame: pl.DataFrame, groups: list[str]) -> pl.DataFrame:
    return (
        frame.group_by(groups)
        .agg(
            pl.len().alias("positions"),
            pl.col("normalization_notional_twd").sum().alias("notional_twd"),
            pl.col("normal_gross_twd").sum().alias("normal_gross_twd"),
            pl.col("normal_cost_twd").sum().alias("normal_cost_twd"),
            pl.col("normal_net_twd").sum().alias("normal_net_twd"),
            pl.col("aggressive_gross_twd").sum().alias("aggressive_gross_twd"),
            pl.col("aggressive_cost_twd").sum().alias("aggressive_cost_twd"),
            pl.col("aggressive_net_twd").sum().alias("aggressive_net_twd"),
            pl.col("gross_delta_twd").sum().alias("gross_delta_twd"),
            pl.col("cost_saving_twd").sum().alias("cost_saving_twd"),
            pl.col("spot_sell_tax_saving_twd").sum().alias(
                "spot_sell_tax_saving_twd"
            ),
            pl.col("net_delta_twd").sum().alias("net_delta_twd"),
            pl.col("aggressive_holding_boundaries").mean().alias(
                "mean_aggressive_holding_boundaries"
            ),
            pl.col("normal_holding_boundaries").mean().alias(
                "mean_normal_holding_boundaries"
            ),
            pl.col("holding_boundaries_avoided").mean().alias(
                "mean_holding_boundaries_avoided"
            ),
            (
                pl.col("normalization_notional_twd")
                * pl.col("holding_boundaries_avoided")
            ).sum().alias("notional_session_boundaries_avoided_twd"),
            pl.col("aggressive_same_day").sum().alias("aggressive_same_day_positions"),
            pl.col("normal_same_day").sum().alias("normal_same_day_positions"),
        )
        .with_columns(
            (
                pl.col("gross_delta_twd") / pl.col("notional_twd") * 10_000.0
            ).alias("gross_delta_bp"),
            (
                pl.col("cost_saving_twd") / pl.col("notional_twd") * 10_000.0
            ).alias("cost_saving_bp"),
            (
                pl.col("spot_sell_tax_saving_twd")
                / pl.col("notional_twd")
                * 10_000.0
            ).alias("spot_sell_tax_saving_bp"),
            (pl.col("net_delta_twd") / pl.col("notional_twd") * 10_000.0).alias(
                "net_delta_bp"
            ),
        )
        .sort(groups)
    )


def main() -> None:
    frame = load_comparison()
    payload = {
        "overall": summarize(frame, ["portfolio_cap_twd"]).to_dicts(),
        "by_comparison_bucket": summarize(
            frame, ["portfolio_cap_twd", "comparison_bucket"]
        ).to_dicts(),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
