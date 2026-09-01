from __future__ import annotations

import math
import unittest
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import polars as pl

from ..quote_fill.layered import EventCursor
from ..quote_fill.policy_spec import ANCHOR_MODEL_ID, TOD_BUCKETS, PolicySpec
from ..quote_fill.s1_day_state import (
    ENTRY_STOP_SECOND,
    S1_POLICY_DECISION_SIGNATURE_COLUMNS,
    SESSION_START_SECOND,
    build_s1_policy_day_state,
    build_s1_policy_state_changes,
    materialize_s1_common_day,
    thin_s1_policy_state_changes,
    weight_s1_policy_state_changes,
)
from ..quote_fill.s1_economic_gate import (
    S1EconomicGateRule,
    evaluate_s1_entry_economics,
)
from ..quote_fill.s1_scenario_spec import (
    S1ScenarioSpec,
    build_s1_scenario_spec_table,
    s1_scenario_specs_from_table,
)
from ..quote_fill.s1_target import S1_ROUTE, build_s1_spot_bid_target
from .test_quote_fill_s1_scenario_spec import (
    _convergence,
    _entry_lookup,
    _mother,
)


def _day() -> pl.DataFrame:
    seconds = list(range(15_600))
    start = datetime(2026, 5, 5, 1, 0, tzinfo=UTC).replace(tzinfo=None)
    timestamps = [start + timedelta(seconds=value) for value in seconds]
    return pl.DataFrame(
        {
            "Date": ["20260505"] * len(seconds),
            "ValueCode": ["2330"] * len(seconds),
            "QuoteCode": ["CDFE6"] * len(seconds),
            "timestamp": timestamps,
            "seconds_from_open": seconds,
            "spot_recv_time": timestamps,
            "spot_sequence": list(range(1, len(seconds) + 1)),
            "spot_ref_price": [50.0] * len(seconds),
            "fut_ref_price": [50.0] * len(seconds),
            "contract_size": [2_000.0] * len(seconds),
            "end_date": [date(2026, 6, 17)] * len(seconds),
            "spot_bid": [50.0] * len(seconds),
            "spot_ask": [50.1] * len(seconds),
            "spot_bid_lots": [10] * len(seconds),
            "spot_ask_lots": [10] * len(seconds),
            "fut_exec_bid": [50.2] * len(seconds),
            "fut_exec_bid_lots": [10] * len(seconds),
            "fut_exec_ask": [50.3] * len(seconds),
            "fut_exec_ask_lots": [10] * len(seconds),
            "basis_mid_bp": [20.0] * len(seconds),
            "eligible_base": [True] * len(seconds),
        }
    )


def _specs() -> tuple[PolicySpec, ...]:
    distances = (20.0, 30.0, 40.0, 50.0)
    return tuple(
        PolicySpec(
            Date="20260505",
            ValueCode="2330",
            QuoteCode="CDFE6",
            entry_tod_bucket=bucket,
            policy_id="q50",
            kind="quantile",
            upper_distance_bp=distance,
            lower_distance_bp=0.0,
            upper_source_id="Q2_trail20_date_equal",
            lower_source_id="C0_center",
            upper_source_asof_date="20260504",
            lower_source_asof_date="20260504",
            combined_source_asof_date="20260504",
            anchor_model_id=ANCHOR_MODEL_ID,
            boundary_quantile=50,
        )
        for bucket, distance in zip(TOD_BUCKETS, distances, strict=True)
    )


def _scenario_specs(scenario_id: str) -> tuple[S1ScenarioSpec, ...]:
    table = build_s1_scenario_spec_table(
        _mother(),
        _entry_lookup(),
        _convergence(),
    )
    return tuple(
        spec
        for spec in s1_scenario_specs_from_table(table)
        if spec.scenario_id == scenario_id
    )


def _scalar_economic_estimate(
    row: dict[str, object],
    spec: S1ScenarioSpec,
):
    cursor = EventCursor(int(row["decision_time_ns"]), 10, 1)
    target = build_s1_spot_bid_target(
        spec,
        date=str(row["Date"]),
        value_code=str(row["ValueCode"]),
        quote_code=str(row["QuoteCode"]),
        route=S1_ROUTE,
        actual_new_send_cursor=cursor,
        causal_anchor_basis_bp=float(row["selected_anchor_bp"]),
        fut_exec_bid=float(row["fut_exec_bid"]),
        fut_exec_ask=float(row["fut_exec_ask"]),
        spot_bid=float(row["spot_bid"]),
        spot_ask=float(row["spot_ask"]),
        contract_size_shares=float(row["contract_size"]),
    )
    estimate = evaluate_s1_entry_economics(
        target,
        observation_cursor=cursor,
        spot_reference_price=float(row["spot_ref_price"]),
        rule=S1EconomicGateRule(
            cost_horizon=spec.cost_horizon,
            safety_floor_bp=spec.safety_floor_bp,
            economic_gate_enabled=spec.economic_gate_enabled,
            deployment_shortlist_eligible=spec.deployment_shortlist_eligible,
        ),
        evaluation_stage="decision_observation",
    )
    return target, estimate


class S1DayStateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.state = build_s1_policy_day_state(_day(), _specs(), policy_id="q50")

    def test_complete_grid_and_frozen_policy_provenance(self) -> None:
        self.assertEqual(
            self.state.height,
            ENTRY_STOP_SECOND - SESSION_START_SECOND,
        )
        self.assertEqual(self.state["policy_id"].unique().to_list(), ["q50"])
        self.assertNotIn("contains_target_day_outcome", self.state.columns)
        first = self.state.filter(pl.col("seconds_from_open") == 300).row(0, named=True)
        self.assertAlmostEqual(float(first["selected_anchor_bp"]), 20.0, places=10)
        self.assertAlmostEqual(float(first["entry_threshold_basis_bp"]), 40.0)
        self.assertAlmostEqual(
            float(first["frozen_exit_threshold_basis_bp_at_observation"]),
            20.0,
        )

    def test_ab12_geometry_changes_with_tod_spec(self) -> None:
        first = self.state.filter(pl.col("seconds_from_open") == 300).row(0, named=True)
        second = self.state.filter(pl.col("seconds_from_open") == 3_600).row(
            0, named=True
        )
        self.assertEqual(first["target_location"], "BID1")
        self.assertEqual(int(first["point_offset"]), 0)
        self.assertTrue(bool(first["ab12_admission_open"]))
        self.assertEqual(second["target_location"], "BID2_BY_TICK")
        self.assertEqual(int(second["point_offset"]), -1)
        self.assertTrue(bool(second["ab12_admission_open"]))

    def test_sparse_changes_keep_bucket_target_transitions(self) -> None:
        changes = thin_s1_policy_state_changes(self.state)
        direct = build_s1_policy_state_changes(_day(), _specs(), policy_id="q50")
        seconds = set(changes["seconds_from_open"].to_list())
        self.assertIn(300, seconds)
        self.assertIn(3_600, seconds)
        self.assertLess(changes.height, self.state.height)
        self.assertTrue(direct.equals(changes))
        weighted = weight_s1_policy_state_changes(direct)
        self.assertEqual(
            int(weighted["represented_product_seconds"].sum()),
            ENTRY_STOP_SECOND - SESSION_START_SECOND,
        )

    def test_tod_boundary_is_retained_when_every_other_decision_field_matches(
        self,
    ) -> None:
        identical_specs = tuple(
            replace(spec, upper_distance_bp=20.0) for spec in _specs()
        )
        state = build_s1_policy_day_state(
            _day(),
            identical_specs,
            policy_id="q50",
        )
        changes = thin_s1_policy_state_changes(state)
        self.assertEqual(
            changes["seconds_from_open"].to_list(),
            [300, 3_600, 7_200, 10_800],
        )

    def test_frozen_exit_tick_change_is_not_lost_when_entry_state_matches(
        self,
    ) -> None:
        common = materialize_s1_common_day(_day()).with_columns(
            pl.when(pl.col("seconds_from_open") == 301)
            .then(pl.lit(15.0))
            .otherwise(pl.col("selected_anchor_bp"))
            .alias("selected_anchor_bp")
        )
        state = build_s1_policy_day_state(common, _specs(), policy_id="q50")
        pair = state.filter(pl.col("seconds_from_open").is_in((300, 301)))
        self.assertEqual(pair["absolute_price_tick"].n_unique(), 1)
        self.assertEqual(pair["base_gate_open"].n_unique(), 1)
        self.assertEqual(pair["ab12_admission_open"].n_unique(), 1)
        self.assertEqual(
            pair["frozen_exit_absolute_price_tick_at_observation"].n_unique(),
            2,
        )
        changes = thin_s1_policy_state_changes(state)
        self.assertIn(301, changes["seconds_from_open"].to_list())

    def test_economic_floor_crossing_is_retained_with_both_spot_ticks_equal(
        self,
    ) -> None:
        # Adjacent IEEE values straddle the strict five-bp floor while both
        # rounded Spot entry and frozen-exit prices remain unchanged.
        closed_ask = 50.28055578888423
        open_ask = math.nextafter(closed_ask, -math.inf)
        day = _day().with_columns(
            pl.lit(50.22).alias("fut_exec_bid"),
            pl.when(pl.col("seconds_from_open") == 300)
            .then(pl.lit(open_ask))
            .otherwise(pl.lit(closed_ask))
            .alias("fut_exec_ask"),
        )
        specs = _scenario_specs("q80_C0_sd_f5")
        state = build_s1_policy_day_state(
            day,
            specs,
            policy_id="q80_C0_sd_f5",
        )
        pair = state.filter(pl.col("seconds_from_open").is_in((300, 301)))
        self.assertEqual(pair["absolute_price_tick"].n_unique(), 1)
        self.assertEqual(
            pair["frozen_exit_absolute_price_tick_at_observation"].n_unique(),
            1,
        )
        self.assertEqual(pair["base_gate_open"].n_unique(), 1)
        self.assertEqual(pair["ab12_admission_open"].n_unique(), 1)
        self.assertEqual(
            pair["decision_economic_status"].to_list(),
            ["eligible", "below_floor"],
        )
        self.assertEqual(
            pair["decision_economic_gate_open"].to_list(),
            [True, False],
        )
        self.assertGreater(
            float(pair["decision_selected_expected_margin_bp"][0]),
            5.0,
        )
        self.assertLessEqual(
            float(pair["decision_selected_expected_margin_bp"][1]),
            5.0,
        )
        changes = thin_s1_policy_state_changes(state)
        self.assertIn(301, changes["seconds_from_open"].to_list())

    def test_exact_anchor_drift_thins_when_discrete_decision_is_equal(self) -> None:
        common = materialize_s1_common_day(_day()).with_columns(
            pl.when(pl.col("seconds_from_open") == 301)
            .then(pl.col("selected_anchor_bp") - 0.001)
            .otherwise(pl.col("selected_anchor_bp"))
            .alias("selected_anchor_bp")
        )
        state = build_s1_policy_day_state(common, _specs(), policy_id="q50")
        pair = state.filter(pl.col("seconds_from_open").is_in((300, 301)))
        self.assertNotEqual(
            float(pair["selected_anchor_bp"][0]),
            float(pair["selected_anchor_bp"][1]),
        )
        for column in S1_POLICY_DECISION_SIGNATURE_COLUMNS:
            self.assertEqual(pair[column].n_unique(), 1, column)
        changes = thin_s1_policy_state_changes(state)
        self.assertNotIn(301, changes["seconds_from_open"].to_list())

    def test_vectorized_economics_matches_scalar_contracts(self) -> None:
        second = pl.col("seconds_from_open")
        phase = (second % 7).cast(pl.Float64)
        future_bid = 50.18 + phase * 0.01
        day = _day().with_columns(
            future_bid.alias("fut_exec_bid"),
            (future_bid + 0.02 + (second % 3).cast(pl.Float64) * 0.025).alias(
                "fut_exec_ask"
            ),
            pl.lit(49.9).alias("spot_bid"),
            pl.lit(50.4).alias("spot_ask"),
            pl.when((second % 2) == 0)
            .then(pl.lit(1_000.0))
            .otherwise(pl.lit(2_000.0))
            .alias("contract_size"),
        )
        sample_seconds = (300, 301, 302, 303, 3_600, 7_200, 10_800)
        for scenario_id in (
            "ctrl_q95_C0_ungated",
            "q80_C0_sd_f5",
            "q95_C0_on_f0",
        ):
            specs = _scenario_specs(scenario_id)
            by_tod = {spec.entry_tod_bucket: spec for spec in specs}
            state = build_s1_policy_day_state(
                day,
                specs,
                policy_id=scenario_id,
            ).filter(pl.col("seconds_from_open").is_in(sample_seconds))
            for row in state.iter_rows(named=True):
                self.assertTrue(bool(row["base_gate_open"]))
                target, estimate = _scalar_economic_estimate(
                    row,
                    by_tod[str(row["entry_tod_bucket"])],
                )
                self.assertEqual(row["target_price"], target.target_price)
                self.assertEqual(
                    row["frozen_exit_target_price_at_observation"],
                    target.frozen_exit_target_price,
                )
                self.assertEqual(
                    row["frozen_exit_absolute_price_tick_at_observation"],
                    target.frozen_exit_absolute_price_tick,
                )
                self.assertEqual(
                    row["reservation_notional_twd_at_observation"],
                    target.reservation_notional_twd,
                )
                self.assertEqual(row["decision_economic_status"], estimate.status)
                self.assertEqual(row["decision_economic_reason"], estimate.reason)
                self.assertEqual(
                    row["decision_economic_gate_open"],
                    estimate.gate_open,
                )
                if estimate.selected_expected_margin_bp is None:
                    self.assertIsNone(row["decision_selected_expected_margin_bp"])
                else:
                    self.assertAlmostEqual(
                        float(row["decision_selected_expected_margin_bp"]),
                        estimate.selected_expected_margin_bp,
                        places=11,
                    )

    def test_vectorized_route_ineligibility_matches_scalar_ordering(self) -> None:
        specs = _scenario_specs("q80_C0_sd_f5")
        first_spec = next(
            spec for spec in specs if spec.entry_tod_bucket == TOD_BUCKETS[0]
        )
        cases = (
            (
                _day().with_columns(
                    pl.lit(50.22).alias("fut_exec_bid"),
                    pl.lit(50.23).alias("fut_exec_ask"),
                    pl.lit(50.2).alias("spot_bid"),
                    pl.lit(50.4).alias("spot_ask"),
                ),
                "frozen_exit_target_not_passive",
            ),
            (
                _day().with_columns(
                    pl.lit(50.22).alias("fut_exec_bid"),
                    pl.lit(50.23).alias("fut_exec_ask"),
                    pl.lit(49.9).alias("spot_bid"),
                    pl.lit(50.1).alias("spot_ask"),
                    pl.lit(46.44).alias("spot_ref_price"),
                ),
                "frozen_exit_target_outside_reference_band",
            ),
        )
        for day, expected_reason in cases:
            row = (
                build_s1_policy_day_state(
                    day,
                    specs,
                    policy_id="q80_C0_sd_f5",
                )
                .filter(pl.col("seconds_from_open") == 300)
                .row(0, named=True)
            )
            self.assertTrue(bool(row["base_gate_open"]))
            _, estimate = _scalar_economic_estimate(row, first_spec)
            self.assertEqual(estimate.status, "route_ineligible")
            self.assertEqual(estimate.reason, expected_reason)
            self.assertEqual(row["decision_economic_status"], estimate.status)
            self.assertEqual(row["decision_economic_reason"], estimate.reason)
            self.assertFalse(bool(row["decision_economic_gate_open"]))

    def test_missing_tod_spec_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "exact four S1 TOD"):
            build_s1_policy_day_state(_day(), _specs()[:-1], policy_id="q50")

    def test_nonpassive_target_is_not_admitted(self) -> None:
        day = _day().with_columns(pl.lit(49.95).alias("spot_ask"))
        result = build_s1_policy_day_state(day, _specs(), policy_id="q50")
        row = result.filter(pl.col("seconds_from_open") == 300).row(0, named=True)
        self.assertEqual(row["gate_reason"], "target_not_passive")
        self.assertFalse(bool(row["base_gate_open"]))
        self.assertFalse(bool(row["ab12_admission_open"]))


if __name__ == "__main__":
    unittest.main()
