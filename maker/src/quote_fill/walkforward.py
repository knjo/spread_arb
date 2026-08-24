"""Production-like daily rolling state/probability tables.

This module separates two operations that must not be conflated:

* raw event facts are deterministic target-day outcomes; and
* a decision snapshot for day ``D`` is estimated only from mature facts whose
  terminal label date is strictly before ``D``.

The first estimator is intentionally a transparent table with hierarchical
shrinkage.  It is a baseline for later ridge/tree hazard challengers, not a
claim that the current pilot already has executable exit PnL or EV.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import polars as pl

from ..common.paths import MAKER_ROOT


DEFAULT_QUOTE_FILL_DIR = MAKER_ROOT / "data" / "quote_fill"
DEFAULT_ADAPTIVE_SNAPSHOT = (
    MAKER_ROOT
    / "data"
    / "quote_width"
    / "adaptive"
    / "adaptive_parameter_snapshot_by_day_symbol.csv"
)


@dataclass(frozen=True)
class WalkForwardSplit:
    """Pre-registered 2026 research phases.

    July--13 August is called pseudo holdout because two dates in that period
    were already inspected during the pilot.  A genuinely locked forward test
    starts with unseen sessions on or after ``forward_holdout_start_date``.
    """

    # 2026-05-05 is the first target session with 60 complete joint sessions
    # strictly before the open.  2026-05-04 is the 60th history session.
    tuning_start_date: str = "20260505"
    confirmation_start_date: str = "20260601"
    pseudo_holdout_start_date: str = "20260701"
    forward_holdout_start_date: str = "20260814"
    frozen_policy_version: str = "pending_freeze_after_20260813"
    frozen_manifest_path: str | None = None
    frozen_manifest_sha256: str | None = None

    def phase(self, date: str) -> str:
        if date >= self.forward_holdout_start_date:
            return "locked_forward_holdout"
        if date >= self.pseudo_holdout_start_date:
            return "retrospective_pseudo_holdout"
        if date >= self.confirmation_start_date:
            return "development_confirmation"
        if date >= self.tuning_start_date:
            return "hyperparameter_tuning"
        return "history_burn_in"

    def verify_freeze_manifest(
        self,
        *,
        probability_config_sha256: str,
        policy_code_sha256: str,
    ) -> bool:
        if not self.frozen_manifest_path or not self.frozen_manifest_sha256:
            return False
        path = Path(self.frozen_manifest_path)
        if not path.is_file():
            return False
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != self.frozen_manifest_sha256:
            return False
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return False
        return (
            payload.get("policy_version") == self.frozen_policy_version
            and not self.frozen_policy_version.startswith("pending")
            and payload.get("probability_config_sha256")
            == probability_config_sha256
            and payload.get("policy_code_sha256") == policy_code_sha256
            and all(
                payload.get(field)
                for field in (
                    "code_commit",
                    "boundary_config_sha256",
                    "liquidity_config_sha256",
                    "execution_policy_sha256",
                    "cost_profile_sha256",
                )
            )
        )


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _policy_code_sha256() -> str:
    digest = hashlib.sha256()
    source_root = MAKER_ROOT / "src"
    for relative_root in ("common", "fair_mid", "quote_width", "quote_fill"):
        for path in sorted((source_root / relative_root).glob("*.py")):
            digest.update(str(path.relative_to(source_root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class WalkForwardProbabilityConfig:
    """Frozen rolling-table estimator settings."""

    lookback_sessions: int = 60
    min_history_sessions: int = 40
    beta_prior_strength: float = 50.0
    min_product_orders: int = 200
    min_product_fills: int = 30
    min_product_dates: int = 20
    min_peer_orders: int = 500
    min_peer_fills: int = 50
    min_peer_dates: int = 20
    min_post_fill_label_coverage: float = 0.80
    estimator_version: str = "rolling_state_beta_v1"

    def validate(self) -> None:
        if self.lookback_sessions <= 0:
            raise ValueError("lookback_sessions must be positive")
        if not 1 <= self.min_history_sessions <= self.lookback_sessions:
            raise ValueError(
                "min_history_sessions must be in [1, lookback_sessions]"
            )
        if self.beta_prior_strength < 0:
            raise ValueError("beta_prior_strength must be non-negative")
        if not 0 <= self.min_post_fill_label_coverage <= 1:
            raise ValueError("post-fill label coverage must be in [0, 1]")
        if (
            self.min_product_orders <= 0
            or self.min_product_fills <= 0
            or self.min_product_dates <= 0
            or self.min_peer_orders <= 0
            or self.min_peer_fills <= 0
            or self.min_peer_dates <= 0
        ):
            raise ValueError("support thresholds must be positive")


@dataclass(frozen=True)
class WalkForwardProbabilityResult:
    action_facts: pl.DataFrame
    product_state_probability: pl.DataFrame
    parent_state_probability: pl.DataFrame


def _require(frame: pl.DataFrame, columns: set[str], source: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def build_action_outcome_facts(
    order_aliases: pl.DataFrame,
    adaptive_snapshot: pl.DataFrame,
    hedge_facts: pl.DataFrame,
    latent_labels: pl.DataFrame,
) -> pl.DataFrame:
    """Join deterministic alias-level outcomes without estimating probability."""

    _require(
        order_aliases,
        {
            "Date",
            "ValueCode",
            "QuoteCode",
            "route",
            "boundary_quantile",
            "raw_order_fact_id",
            "policy_generation_id",
            "target_rank_at_submit",
            "initial_queue_ahead",
            "submit_recv_time_ns",
            "threshold_basis_bp",
            "effective_basis_bp",
            "full_fill",
            "cancel_required",
            "spot_book_age_ms_at_submit",
            "future_book_age_ms_at_submit",
            "source_asof_date",
            "parameter_version",
        },
        "order aliases",
    )
    _require(
        adaptive_snapshot,
        {
            "Date",
            "ValueCode",
            "QuoteCode",
            "boundary_quantile",
            "upper_distance_bp",
            "lower_distance_bp",
            "adaptive_parameter_valid",
            "source_asof_date",
            "contains_target_day_outcome",
            "parameter_version",
        },
        "adaptive snapshot",
    )
    _require(
        hedge_facts,
        {
            "raw_order_fact_id",
            "status",
            "signed_total_slippage_bp",
            "decision_book_age_ms",
            "depth_shortfall",
        },
        "hedge facts",
    )
    _require(
        latent_labels,
        {
            "policy_generation_id",
            "target_id",
            "observation_delay_seconds",
            "status",
            "time_to_latent_hit_seconds",
            "Date",
            "ValueCode",
            "QuoteCode",
            "route",
            "boundary_quantile",
            "lower_distance_bp",
            "adaptive_source_asof_date",
            "adaptive_parameter_version",
        },
        "latent labels",
    )
    valid_snapshot = adaptive_snapshot.filter(pl.col("adaptive_parameter_valid"))
    unsafe = valid_snapshot.filter(
        pl.col("contains_target_day_outcome").fill_null(True)
    )
    if unsafe.height:
        raise ValueError("adaptive snapshot contains target-day outcome")
    if valid_snapshot.filter(
        pl.col("parameter_version").is_null()
        | pl.col("source_asof_date").is_null()
    ).height:
        raise ValueError("valid adaptive snapshot has null parameter lineage")
    parsed_dates = valid_snapshot.select(
        pl.col("source_asof_date").cast(pl.String).alias("_source_text"),
        pl.col("Date").cast(pl.String).alias("_target_text"),
    ).with_columns(
        pl.col("_source_text")
        .str.strptime(pl.Date, "%Y%m%d", strict=False)
        .alias("_source"),
        pl.col("_target_text")
        .str.strptime(pl.Date, "%Y%m%d", strict=False)
        .alias("_target"),
    )
    malformed = parsed_dates.filter(
        pl.col("_source").is_null()
        | pl.col("_target").is_null()
        | (pl.col("_source_text").str.len_chars() != 8)
        | (pl.col("_target_text").str.len_chars() != 8)
        | (pl.col("_source") >= pl.col("_target"))
    )
    if malformed.height:
        raise ValueError("adaptive source_asof_date must be before target Date")

    snapshot_key = ["Date", "ValueCode", "QuoteCode", "boundary_quantile"]
    snapshot = valid_snapshot.select(
        *snapshot_key,
        "upper_distance_bp",
        "lower_distance_bp",
        pl.col("source_asof_date").alias("_snapshot_source_asof_date"),
        pl.col("parameter_version").alias("_snapshot_parameter_version"),
    )
    if snapshot.select(snapshot_key).n_unique() != snapshot.height:
        raise ValueError("adaptive snapshot contains duplicate valid keys")
    aliases = order_aliases.join(
        snapshot,
        on=snapshot_key,
        how="left",
        validate="m:1",
    )
    if aliases.filter(
        pl.col("upper_distance_bp").is_null()
        | pl.col("lower_distance_bp").is_null()
    ).height:
        raise ValueError("order aliases do not resolve to a valid prior snapshot")
    alias_lineage_mismatch = aliases.filter(
        (
            pl.col("source_asof_date").is_null()
            | pl.col("parameter_version").is_null()
            | (
                pl.col("source_asof_date")
                != pl.col("_snapshot_source_asof_date")
            )
            | (
                pl.col("parameter_version")
                != pl.col("_snapshot_parameter_version")
            )
        ).fill_null(True)
    )
    if alias_lineage_mismatch.height:
        raise ValueError("order alias parameter lineage disagrees with snapshot")

    if hedge_facts.select("raw_order_fact_id").n_unique() != hedge_facts.height:
        raise ValueError("hedge facts must be unique by raw_order_fact_id")
    lower = latent_labels.filter(
        (pl.col("target_id") == "frozen_adaptive_lower")
        & (pl.col("observation_delay_seconds") == 30)
    ).select(
        "policy_generation_id",
        pl.lit(True).alias("_label_present"),
        pl.col("Date").alias("_label_Date"),
        pl.col("ValueCode").alias("_label_ValueCode"),
        pl.col("QuoteCode").alias("_label_QuoteCode"),
        pl.col("route").alias("_label_route"),
        pl.col("boundary_quantile").alias("_label_boundary_quantile"),
        pl.col("lower_distance_bp").alias("_label_lower_distance_bp"),
        pl.col("adaptive_source_asof_date").alias("_label_source_asof_date"),
        pl.col("adaptive_parameter_version").alias("_label_parameter_version"),
        pl.col("status").alias("lower_status_30s"),
        pl.col("time_to_latent_hit_seconds").alias("lower_hit_time_30s"),
    )
    if lower.select("policy_generation_id").n_unique() != lower.height:
        raise ValueError("latent lower labels contain duplicate policy rows")

    facts = aliases.join(
        hedge_facts.select(
            "raw_order_fact_id",
            pl.col("status").alias("hedge_status_50ms"),
            pl.col("signed_total_slippage_bp").alias("hedge_slippage_bp_50ms"),
            pl.col("decision_book_age_ms").alias("hedge_book_age_ms_50ms"),
            pl.col("depth_shortfall").alias("hedge_depth_shortfall_50ms"),
        ),
        on="raw_order_fact_id",
        how="left",
        validate="m:1",
    ).join(
        lower,
        on="policy_generation_id",
        how="left",
        validate="1:1",
    )
    label_lineage_mismatch = facts.filter(
        pl.col("_label_present").fill_null(False)
        & (
            (pl.col("Date") != pl.col("_label_Date"))
            | (pl.col("ValueCode") != pl.col("_label_ValueCode"))
            | (pl.col("QuoteCode") != pl.col("_label_QuoteCode"))
            | (pl.col("route") != pl.col("_label_route"))
            | (
                pl.col("boundary_quantile")
                != pl.col("_label_boundary_quantile")
            )
            | (
                (pl.col("lower_distance_bp") - pl.col("_label_lower_distance_bp"))
                .abs()
                > 1e-9
            )
            | (
                pl.col("_snapshot_source_asof_date")
                != pl.col("_label_source_asof_date")
            )
            | (
                pl.col("_snapshot_parameter_version")
                != pl.col("_label_parameter_version")
            )
        ).fill_null(True)
    )
    if label_lineage_mismatch.height:
        raise ValueError("latent exit label lineage disagrees with order snapshot")
    facts = facts.with_columns(
        (
            pl.col("effective_basis_bp")
            - (pl.col("threshold_basis_bp") - pl.col("upper_distance_bp"))
        ).alias("effective_open_distance_bp"),
        (
            pl.col("effective_basis_bp") - pl.col("hedge_slippage_bp_50ms")
        ).alias("locked_basis_bp_50ms"),
        pl.col("full_fill").fill_null(False).alias("entry_full_fill"),
        (
            pl.col("full_fill").fill_null(False)
            & pl.col("hedge_status_50ms").is_not_null()
        ).alias("hedge_label_observed"),
        (
            pl.col("full_fill").fill_null(False)
            & (pl.col("hedge_status_50ms") == "executable")
            & (pl.col("hedge_depth_shortfall_50ms").fill_null(0) == 0)
        ).alias("hedge_priceable_50ms"),
        pl.col("lower_status_30s")
        .is_in(["hit", "session_no_hit"])
        .fill_null(False)
        .alias("lower_label_known_30s"),
        (pl.col("lower_status_30s") == "hit")
        .fill_null(False)
        .alias("lower_hit_30s"),
    ).with_columns(
        (
            pl.col("locked_basis_bp_50ms")
            - (pl.col("threshold_basis_bp") - pl.col("upper_distance_bp"))
        ).alias("locked_open_distance_bp_50ms"),
        (
            pl.col("hedge_priceable_50ms")
            & (
                pl.col("locked_basis_bp_50ms")
                >= pl.col("threshold_basis_bp")
            )
        ).alias("locked_threshold_qualified_50ms"),
        (
            pl.col("effective_open_distance_bp") + pl.col("lower_distance_bp")
        ).alias("nominal_latent_band_bp"),
        pl.col("Date").cast(pl.String).alias("label_end_date"),
        pl.lit(False).alias("execution_safe_snapshot"),
        pl.lit(True).alias("contains_target_day_outcome"),
        pl.lit(False).alias("ev_ready"),
    )
    return _add_state_columns(facts).sort(
        ["Date", "ValueCode", "route", "boundary_quantile", "submit_recv_time_ns"]
    )


def _add_state_columns(frame: pl.DataFrame) -> pl.DataFrame:
    local_second = (
        (
            pl.col("submit_recv_time_ns") // 1_000_000_000
            + 8 * 60 * 60
        )
        % (24 * 60 * 60)
    )
    max_age = pl.max_horizontal(
        pl.col("spot_book_age_ms_at_submit"),
        pl.col("future_book_age_ms_at_submit"),
    )
    return frame.with_columns(
        pl.when(
            pl.col("target_rank_at_submit").is_in(
                ["A1", "B1", "ASK1", "BID1"]
            )
        )
        .then(pl.lit("at_bbo"))
        .when(pl.col("target_rank_at_submit") == "inside")
        .then(pl.lit("inside"))
        .otherwise(pl.lit("behind"))
        .alias("rank_bucket"),
        pl.when(pl.col("initial_queue_ahead").is_null())
        .then(pl.lit("unknown"))
        .when(pl.col("initial_queue_ahead") <= 1)
        .then(pl.lit("00_0to1"))
        .when(pl.col("initial_queue_ahead") <= 5)
        .then(pl.lit("01_2to5"))
        .when(pl.col("initial_queue_ahead") <= 20)
        .then(pl.lit("02_6to20"))
        .otherwise(pl.lit("03_21plus"))
        .alias("queue_bucket"),
        pl.when(local_second < 9 * 3600 + 30 * 60)
        .then(pl.lit("open_0905_0930"))
        .when(local_second < 12 * 3600)
        .then(pl.lit("mid_0930_1200"))
        .otherwise(pl.lit("late_1200_1320"))
        .alias("tod_bucket"),
        pl.when(max_age.is_null())
        .then(pl.lit("unknown"))
        .when(max_age <= 100)
        .then(pl.lit("fresh_le100ms"))
        .when(max_age <= 1000)
        .then(pl.lit("fresh_le1000ms"))
        .otherwise(pl.lit("stale_gt1000ms"))
        .alias("freshness_bucket"),
    )


def expand_state_families(facts: pl.DataFrame) -> pl.DataFrame:
    """Make separate one-factor state tables without a sparse Cartesian cross."""

    families = [
        ("all", pl.lit("all")),
        ("rank", pl.col("rank_bucket")),
        ("queue", pl.col("queue_bucket")),
        ("time_of_day", pl.col("tod_bucket")),
        ("freshness", pl.col("freshness_bucket")),
    ]
    return pl.concat(
        [
            facts.with_columns(
                pl.lit(family).alias("state_family"),
                bucket.alias("state_bucket"),
            )
            for family, bucket in families
        ],
        how="vertical",
    )


def _normalise_sessions(sessions: Iterable[str]) -> list[str]:
    result = sorted({str(value) for value in sessions})
    if not result:
        raise ValueError("session calendar must not be empty")
    parsed = pl.DataFrame({"Date": result}).with_columns(
        pl.col("Date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("_date")
    )
    if parsed.filter(
        pl.col("_date").is_null() | (pl.col("Date").str.len_chars() != 8)
    ).height:
        raise ValueError("session dates must be valid YYYYMMDD")
    return result


STATE_KEYS = [
    "route",
    "boundary_quantile",
    "state_family",
    "state_bucket",
]
PRODUCT_KEYS = ["ValueCode", *STATE_KEYS]


def _aggregate_counts(frame: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    return frame.group_by(keys).agg(
        pl.len().alias("n_orders"),
        pl.col("Date").n_unique().alias("n_product_dates"),
        pl.col("ValueCode").n_unique().alias("n_products"),
        pl.col("entry_full_fill").sum().alias("n_full_fill"),
        pl.col("hedge_label_observed").sum().alias("n_hedge_observed"),
        pl.col("hedge_slippage_bp_50ms")
        .is_finite()
        .sum()
        .alias("n_hedge_slippage_observed"),
        pl.col("hedge_priceable_50ms").sum().alias("n_hedge_priceable"),
        pl.col("locked_threshold_qualified_50ms")
        .sum()
        .alias("n_locked_threshold_qualified"),
        pl.col("lower_label_known_30s").sum().alias("n_lower_known"),
        pl.col("lower_hit_30s").sum().alias("n_lower_hit"),
        pl.col("hedge_slippage_bp_50ms")
        .filter(pl.col("hedge_label_observed"))
        .median()
        .alias("hedge_slippage_bp_p50"),
        pl.col("hedge_slippage_bp_50ms")
        .filter(pl.col("hedge_label_observed"))
        .quantile(0.95, interpolation="nearest")
        .alias("hedge_slippage_bp_p95"),
        pl.col("nominal_latent_band_bp").median().alias(
            "nominal_latent_band_bp_p50"
        ),
    )


def _rate(success: str, trials: str, output: str) -> pl.Expr:
    return (
        pl.when(pl.col(trials) > 0)
        .then(pl.col(success) / pl.col(trials))
        .otherwise(None)
        .alias(output)
    )


def _posterior(
    success: str,
    trials: str,
    parent_rate: str,
    strength: float,
    output: str,
) -> pl.Expr:
    return (
        pl.when((pl.col(trials) > 0) & pl.col(parent_rate).is_not_null())
        .then(
            (pl.col(success) + strength * pl.col(parent_rate))
            / (pl.col(trials) + strength)
        )
        .otherwise(None)
        .alias(output)
    )


def build_walkforward_probability_tables(
    action_facts: pl.DataFrame,
    sessions: Sequence[str],
    config: WalkForwardProbabilityConfig = WalkForwardProbabilityConfig(),
    split: WalkForwardSplit = WalkForwardSplit(),
    *,
    asof_dates: Sequence[str] | None = None,
) -> WalkForwardProbabilityResult:
    """Build daily product and route-state parent tables from mature history."""

    config.validate()
    sessions = _normalise_sessions(sessions)
    _require(
        action_facts,
        {
            "Date",
            "ValueCode",
            "route",
            "boundary_quantile",
            "label_end_date",
            "entry_full_fill",
            "hedge_label_observed",
            "hedge_priceable_50ms",
            "locked_threshold_qualified_50ms",
            "lower_label_known_30s",
            "lower_hit_30s",
            "hedge_slippage_bp_50ms",
            "nominal_latent_band_bp",
            "rank_bucket",
            "queue_bucket",
            "tod_bucket",
            "freshness_bucket",
        },
        "action facts",
    )
    selected_dates = _normalise_sessions(asof_dates or sessions)
    unknown = sorted(set(selected_dates) - set(sessions))
    if unknown:
        raise ValueError(f"asof dates absent from session calendar: {unknown[:5]}")
    expanded = expand_state_families(
        action_facts.with_columns(
            pl.col("Date").cast(pl.String),
            pl.col("ValueCode").cast(pl.String),
            pl.col("label_end_date").cast(pl.String),
        )
    )
    index = {value: offset for offset, value in enumerate(sessions)}
    probability_config_sha256 = _canonical_sha256(asdict(config))
    policy_code_sha256 = _policy_code_sha256()
    product_parts: list[pl.DataFrame] = []
    parent_parts: list[pl.DataFrame] = []
    for asof_date in selected_dates:
        if (
            split.phase(asof_date) == "locked_forward_holdout"
            and not split.verify_freeze_manifest(
                probability_config_sha256=probability_config_sha256,
                policy_code_sha256=policy_code_sha256,
            )
        ):
            raise ValueError(
                "locked forward holdout requires a verified frozen manifest"
            )
        target_index = index[asof_date]
        prior_dates = sessions[
            max(0, target_index - config.lookback_sessions):target_index
        ]
        if len(prior_dates) < config.min_history_sessions:
            continue
        training = expanded.filter(
            pl.col("Date").is_in(prior_dates)
            & (pl.col("label_end_date") < asof_date)
        )
        if training.is_empty():
            continue
        observed_training_sessions = training["Date"].n_unique()
        global_history_gate = (
            observed_training_sessions >= config.min_history_sessions
        )
        parent = _aggregate_counts(training, STATE_KEYS).with_columns(
            _rate("n_full_fill", "n_orders", "p_full_fill_parent"),
            _rate(
                "n_locked_threshold_qualified",
                "n_full_fill",
                "p_locked_given_fill_lower_bound_parent",
            ),
            _rate(
                "n_locked_threshold_qualified",
                "n_hedge_observed",
                "p_locked_given_observed_hedge_parent",
            ),
            _rate(
                "n_lower_hit",
                "n_full_fill",
                "p_lower_hit_given_fill_lower_bound_parent",
            ),
            _rate(
                "n_lower_hit",
                "n_lower_known",
                "p_lower_hit_given_known_parent",
            ),
        )
        leaf = _aggregate_counts(training, PRODUCT_KEYS)
        presence = training.select(
            *STATE_KEYS, "ValueCode", "Date"
        ).unique()
        solo_dates = (
            presence.join(
                presence.group_by([*STATE_KEYS, "Date"]).agg(
                    pl.col("ValueCode").n_unique().alias("_products_on_date")
                ),
                on=[*STATE_KEYS, "Date"],
                how="left",
                validate="m:1",
            )
            .filter(pl.col("_products_on_date") == 1)
            .group_by(PRODUCT_KEYS)
            .agg(pl.col("Date").n_unique().alias("_solo_parent_dates"))
        )
        parent_for_join = parent.select(
            *STATE_KEYS,
            pl.col("n_orders").alias("parent_n_orders"),
            pl.col("n_product_dates").alias("parent_n_product_dates"),
            pl.col("n_products").alias("parent_n_products"),
            pl.col("n_full_fill").alias("parent_n_full_fill"),
            pl.col("n_hedge_observed").alias("parent_n_hedge_observed"),
            pl.col("n_locked_threshold_qualified").alias(
                "parent_n_locked_threshold_qualified"
            ),
            pl.col("n_lower_known").alias("parent_n_lower_known"),
            pl.col("n_lower_hit").alias("parent_n_lower_hit"),
        )
        leaf = leaf.join(
            parent_for_join,
            on=STATE_KEYS,
            how="left",
            validate="m:1",
        ).join(
            solo_dates,
            on=PRODUCT_KEYS,
            how="left",
            validate="1:1",
        ).with_columns(
            pl.col("_solo_parent_dates").fill_null(0),
            (pl.col("parent_n_orders") - pl.col("n_orders")).alias(
                "peer_n_orders"
            ),
            (pl.col("parent_n_product_dates") - pl.col("_solo_parent_dates"))
            .alias("peer_n_product_dates"),
            (pl.col("parent_n_products") - 1).alias("peer_n_products"),
            (pl.col("parent_n_full_fill") - pl.col("n_full_fill")).alias(
                "peer_n_full_fill"
            ),
            (
                pl.col("parent_n_hedge_observed")
                - pl.col("n_hedge_observed")
            ).alias("peer_n_hedge_observed"),
            (
                pl.col("parent_n_locked_threshold_qualified")
                - pl.col("n_locked_threshold_qualified")
            ).alias("peer_n_locked_threshold_qualified"),
            (pl.col("parent_n_lower_hit") - pl.col("n_lower_hit")).alias(
                "peer_n_lower_hit"
            ),
            (pl.col("parent_n_lower_known") - pl.col("n_lower_known")).alias(
                "peer_n_lower_known"
            ),
        ).with_columns(
            _rate("peer_n_full_fill", "peer_n_orders", "p_full_fill_peer"),
            _rate(
                "peer_n_locked_threshold_qualified",
                "peer_n_full_fill",
                "p_locked_given_fill_lower_bound_peer",
            ),
            _rate(
                "peer_n_lower_hit",
                "peer_n_full_fill",
                "p_lower_hit_given_fill_lower_bound_peer",
            ),
            _rate("n_full_fill", "n_orders", "p_full_fill_empirical"),
            _rate(
                "n_locked_threshold_qualified",
                "n_full_fill",
                "p_locked_given_fill_lower_bound_empirical",
            ),
            _rate(
                "n_locked_threshold_qualified",
                "n_hedge_observed",
                "p_locked_given_observed_hedge_empirical",
            ),
            _rate(
                "n_lower_hit",
                "n_full_fill",
                "p_lower_hit_given_fill_lower_bound_empirical",
            ),
            _rate(
                "n_lower_hit",
                "n_lower_known",
                "p_lower_hit_given_known_empirical",
            ),
        ).with_columns(
            _posterior(
                "n_full_fill",
                "n_orders",
                "p_full_fill_peer",
                config.beta_prior_strength,
                "p_full_fill_posterior",
            ),
            _posterior(
                "n_locked_threshold_qualified",
                "n_full_fill",
                "p_locked_given_fill_lower_bound_peer",
                config.beta_prior_strength,
                "p_locked_given_fill_lower_bound_posterior",
            ),
            _posterior(
                "n_lower_hit",
                "n_full_fill",
                "p_lower_hit_given_fill_lower_bound_peer",
                config.beta_prior_strength,
                "p_lower_hit_given_fill_lower_bound_posterior",
            ),
        ).with_columns(
            (
                pl.lit(global_history_gate)
                & (pl.col("n_product_dates") >= config.min_product_dates)
                & (pl.col("n_orders") >= config.min_product_orders)
            ).alias("product_fill_support_gate"),
            (
                pl.lit(global_history_gate)
                & (pl.col("n_product_dates") >= config.min_product_dates)
                & (pl.col("n_full_fill") >= config.min_product_fills)
                & (pl.col("n_hedge_observed") >= config.min_product_fills)
                & (
                    pl.col("n_hedge_observed") / pl.col("n_full_fill")
                    >= config.min_post_fill_label_coverage
                )
            ).alias("product_locked_support_gate"),
            (
                pl.lit(global_history_gate)
                & (pl.col("n_product_dates") >= config.min_product_dates)
                & (pl.col("n_full_fill") >= config.min_product_fills)
                & (pl.col("n_lower_known") >= config.min_product_fills)
                & (
                    pl.col("n_lower_known") / pl.col("n_full_fill")
                    >= config.min_post_fill_label_coverage
                )
            ).alias("product_lower_support_gate"),
            (
                pl.lit(global_history_gate)
                & (pl.col("n_product_dates") >= config.min_product_dates)
                & (
                    pl.col("n_hedge_slippage_observed")
                    >= config.min_product_fills
                )
                & (
                    pl.col("n_hedge_slippage_observed")
                    / pl.col("n_full_fill")
                    >= config.min_post_fill_label_coverage
                )
            ).alias("product_slippage_support_gate"),
            (
                pl.lit(global_history_gate)
                & (pl.col("peer_n_products") >= 1)
                & (pl.col("peer_n_product_dates") >= config.min_peer_dates)
                & (pl.col("peer_n_orders") >= config.min_peer_orders)
            ).alias("peer_fill_support_gate"),
            (
                pl.lit(global_history_gate)
                & (pl.col("peer_n_products") >= 1)
                & (pl.col("peer_n_product_dates") >= config.min_peer_dates)
                & (pl.col("peer_n_full_fill") >= config.min_peer_fills)
                & (pl.col("peer_n_hedge_observed") >= config.min_peer_fills)
                & (
                    pl.col("peer_n_hedge_observed")
                    / pl.col("peer_n_full_fill")
                    >= config.min_post_fill_label_coverage
                )
            ).alias("peer_locked_support_gate"),
            (
                pl.lit(global_history_gate)
                & (pl.col("peer_n_products") >= 1)
                & (pl.col("peer_n_product_dates") >= config.min_peer_dates)
                & (pl.col("peer_n_full_fill") >= config.min_peer_fills)
                & (pl.col("peer_n_lower_known") >= config.min_peer_fills)
                & (
                    pl.col("peer_n_lower_known") / pl.col("peer_n_full_fill")
                    >= config.min_post_fill_label_coverage
                )
            ).alias("peer_lower_support_gate"),
        ).with_columns(
            pl.when(pl.col("product_fill_support_gate"))
            .then(
                pl.when(pl.col("peer_fill_support_gate"))
                .then(pl.col("p_full_fill_posterior"))
                .otherwise(pl.col("p_full_fill_empirical"))
            )
            .when(pl.col("peer_fill_support_gate"))
            .then(pl.col("p_full_fill_peer"))
            .otherwise(None)
            .alias("reported_p_full_fill"),
            pl.when(pl.col("product_locked_support_gate"))
            .then(
                pl.when(pl.col("peer_locked_support_gate"))
                .then(pl.col("p_locked_given_fill_lower_bound_posterior"))
                .otherwise(
                    pl.col("p_locked_given_fill_lower_bound_empirical")
                )
            )
            .when(pl.col("peer_locked_support_gate"))
            .then(pl.col("p_locked_given_fill_lower_bound_peer"))
            .otherwise(None)
            .alias("reported_p_locked_given_fill"),
            pl.when(pl.col("product_lower_support_gate"))
            .then(
                pl.when(pl.col("peer_lower_support_gate"))
                .then(pl.col("p_lower_hit_given_fill_lower_bound_posterior"))
                .otherwise(
                    pl.col("p_lower_hit_given_fill_lower_bound_empirical")
                )
            )
            .when(pl.col("peer_lower_support_gate"))
            .then(pl.col("p_lower_hit_given_fill_lower_bound_peer"))
            .otherwise(None)
            .alias("reported_p_lower_hit_given_fill"),
            pl.when(
                pl.col("product_fill_support_gate")
                & pl.col("peer_fill_support_gate")
            )
            .then(pl.lit("product_shrunk"))
            .when(pl.col("product_fill_support_gate"))
            .then(pl.lit("product_empirical"))
            .when(pl.col("peer_fill_support_gate"))
            .then(pl.lit("peer_excluding_product"))
            .otherwise(pl.lit("insufficient_support"))
            .alias("fill_fallback_level"),
            pl.when(
                pl.col("product_locked_support_gate")
                & pl.col("peer_locked_support_gate")
            )
            .then(pl.lit("product_shrunk"))
            .when(pl.col("product_locked_support_gate"))
            .then(pl.lit("product_empirical"))
            .when(pl.col("peer_locked_support_gate"))
            .then(pl.lit("peer_excluding_product"))
            .otherwise(pl.lit("insufficient_support"))
            .alias("locked_fallback_level"),
            pl.when(
                pl.col("product_lower_support_gate")
                & pl.col("peer_lower_support_gate")
            )
            .then(pl.lit("product_shrunk"))
            .when(pl.col("product_lower_support_gate"))
            .then(pl.lit("product_empirical"))
            .when(pl.col("peer_lower_support_gate"))
            .then(pl.lit("peer_excluding_product"))
            .otherwise(pl.lit("insufficient_support"))
            .alias("lower_fallback_level"),
            pl.when(pl.col("product_slippage_support_gate"))
            .then(pl.col("hedge_slippage_bp_p95"))
            .otherwise(None)
            .alias("reported_hedge_slippage_bp_p95"),
            _rate(
                "n_locked_threshold_qualified",
                "n_orders",
                "p_joint_locked_threshold_per_order",
            ),
            _rate(
                "n_lower_hit",
                "n_orders",
                "p_joint_latent_lower_per_order",
            ),
            _rate(
                "n_hedge_observed",
                "n_full_fill",
                "hedge_label_coverage_given_fill",
            ),
            _rate(
                "n_lower_known",
                "n_full_fill",
                "lower_label_coverage_given_fill",
            ),
        )
        metadata = _snapshot_metadata(
            asof_date,
            prior_dates,
            training,
            config,
            split,
            probability_config_sha256,
            policy_code_sha256,
        )
        product_parts.append(leaf.with_columns(*metadata))
        parent_parts.append(
            parent.with_columns(
                pl.lit("*").alias("ValueCode"),
                pl.lit("route_state_parent").alias("hierarchy_level"),
                *metadata,
            )
        )

    product = _concat(product_parts)
    parent = _concat(parent_parts)
    if not product.is_empty():
        product = product.with_columns(
            pl.lit("product_state_shrunk").alias("hierarchy_level")
        ).sort(["asof_date", *PRODUCT_KEYS])
    if not parent.is_empty():
        parent = parent.sort(["asof_date", *STATE_KEYS])
    return WalkForwardProbabilityResult(action_facts, product, parent)


def _snapshot_metadata(
    asof_date: str,
    prior_dates: list[str],
    training: pl.DataFrame,
    config: WalkForwardProbabilityConfig,
    split: WalkForwardSplit,
    probability_config_sha256: str,
    policy_code_sha256: str,
) -> list[pl.Expr]:
    max_label = training["label_end_date"].max()
    if max_label is not None and str(max_label) >= asof_date:
        raise AssertionError("walk-forward snapshot includes immature label")
    phase = split.phase(asof_date)
    freeze_verified = (
        phase == "locked_forward_holdout"
        and split.verify_freeze_manifest(
            probability_config_sha256=probability_config_sha256,
            policy_code_sha256=policy_code_sha256,
        )
    )
    return [
        pl.lit(asof_date).alias("asof_date"),
        pl.lit(prior_dates[0]).alias("train_start_date"),
        pl.lit(prior_dates[-1]).alias("train_end_date"),
        pl.lit(max_label).cast(pl.String).alias("label_cutoff_date"),
        pl.lit(len(prior_dates)).alias("window_sessions"),
        pl.lit(training["Date"].n_unique()).alias("observed_training_sessions"),
        pl.lit(config.lookback_sessions).alias("lookback_sessions"),
        pl.lit(config.estimator_version).alias("estimator_version"),
        pl.lit(probability_config_sha256).alias("probability_config_sha256"),
        pl.lit(policy_code_sha256).alias("policy_code_sha256"),
        pl.lit(split.frozen_policy_version).alias("policy_version"),
        pl.lit(split.frozen_manifest_sha256).cast(pl.String).alias(
            "frozen_manifest_sha256"
        ),
        pl.lit(phase).alias("research_phase"),
        pl.lit(freeze_verified).alias("policy_frozen"),
        pl.lit(freeze_verified).alias("policy_freeze_verified"),
        pl.lit(True).alias("execution_safe_snapshot"),
        pl.lit(False).alias("contains_target_day_outcome"),
        pl.lit(False).alias("executable_exit_included"),
        pl.lit(False).alias("fees_tax_overnight_included"),
        pl.lit(False).alias("ev_ready"),
    ]


def _concat(frames: list[pl.DataFrame]) -> pl.DataFrame:
    return (
        pl.concat(frames, how="diagonal_relaxed")
        if frames
        else pl.DataFrame()
    )


def run_walkforward_probability_study(
    *,
    quote_fill_dir: Path = DEFAULT_QUOTE_FILL_DIR,
    adaptive_snapshot_path: Path = DEFAULT_ADAPTIVE_SNAPSHOT,
    sessions: Sequence[str] | None = None,
    output_dir: Path | None = None,
    config: WalkForwardProbabilityConfig = WalkForwardProbabilityConfig(),
    split: WalkForwardSplit = WalkForwardSplit(),
) -> WalkForwardProbabilityResult:
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
    facts = build_action_outcome_facts(
        pl.read_parquet(quote_fill_dir / "order_aliases.parquet"),
        snapshot,
        pl.read_parquet(quote_fill_dir / "hedge_facts.parquet"),
        pl.read_parquet(quote_fill_dir / "latent_exit_opportunity_labels.parquet"),
    )
    calendar = list(sessions or sorted(facts["Date"].unique().to_list()))
    result = build_walkforward_probability_tables(
        facts,
        calendar,
        config,
        split,
    )
    destination = output_dir or quote_fill_dir / "walkforward"
    destination.mkdir(parents=True, exist_ok=True)
    result.action_facts.write_parquet(destination / "action_outcome_facts.parquet")
    result.product_state_probability.write_parquet(
        destination / "product_state_probability.parquet"
    )
    result.parent_state_probability.write_parquet(
        destination / "parent_state_probability.parquet"
    )
    payload = {
        "probability_config": asdict(config),
        "split": asdict(split),
        "state_semantics": "separate one-factor families; no Cartesian cross",
        "daily_update": "last N trading sessions with label_end_date < asof_date",
        "ev_ready": False,
    }
    (destination / "config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build daily rolling maker state/probability tables"
    )
    parser.add_argument("--quote-fill-dir", type=Path, default=DEFAULT_QUOTE_FILL_DIR)
    parser.add_argument(
        "--adaptive-snapshot", type=Path, default=DEFAULT_ADAPTIVE_SNAPSHOT
    )
    parser.add_argument("--sessions", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--lookback-sessions", type=int, default=60)
    parser.add_argument("--min-history-sessions", type=int, default=40)
    parser.add_argument("--beta-prior-strength", type=float, default=50.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sessions = None
    if args.sessions is not None:
        sessions = [
            line.strip()
            for line in args.sessions.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    config = WalkForwardProbabilityConfig(
        lookback_sessions=args.lookback_sessions,
        min_history_sessions=args.min_history_sessions,
        beta_prior_strength=args.beta_prior_strength,
    )
    result = run_walkforward_probability_study(
        quote_fill_dir=args.quote_fill_dir,
        adaptive_snapshot_path=args.adaptive_snapshot,
        sessions=sessions,
        output_dir=args.output_dir,
        config=config,
    )
    print(result.product_state_probability)


if __name__ == "__main__":
    main()
