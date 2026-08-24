"""Focused contracts for S0 August market/queue attribution."""

from __future__ import annotations

import unittest

import polars as pl

from ..quote_fill.august_attribution import (
    ACTUAL_CANCEL_PHASE,
    ACTUAL_NEW_PHASE,
    APPROXIMATE_FILL_PHASE,
    EXCURSION_SCHEMA,
    RAW_FUTURE_PHASE,
    RAW_SPOT_PHASE,
    TOUCH_LINK_SCHEMA,
    build_decomposition,
    deduplicate_touch_pairs_to_orders,
    extract_positive_excursions,
    link_first_touches_to_orders,
    link_touch_order_pairs,
    summarize_monthly_attribution,
    summarize_post_touch_by_rank,
)
from ..quote_fill.august_attribution_runner import (
    ENTRY_DRAIN_START_SECOND,
    EXPECTED_INPUT_RECORD_COUNT,
    EXPECTED_PRODUCT_DAY_COUNT,
    EXPECTED_SESSION_COUNT,
    MANIFEST_SHA256,
    SCENARIO_ID,
    AttributionPaths,
    _add_raw_analysis_fields,
    _canonical_checks,
    _merge_venue_state_events,
    _membership_sensitivity,
    _raw_order_id,
    _session_second_ns,
    _verify_domain_frames,
    build_quote_only_orders,
)


def _raw_states(
    values: list[tuple[int, bool, float | None]],
    *,
    date: str = "20260803",
    value_code: str = "2330",
    quote_code: str = "CDFU6",
    upper: float = 5.0,
) -> pl.DataFrame:
    return pl.from_dicts(
        [
            {
                "Date": date,
                "ValueCode": value_code,
                "QuoteCode": quote_code,
                "cursor_time_ns": time_ns,
                "cursor_event_sequence": RAW_SPOT_PHASE,
                "cursor_row_index": row_index,
                "analysis_eligible_raw": eligible,
                "residual_excursion_bp": residual,
                "upper_distance_bp": upper,
                "spread_pair_epoch": 100 + row_index,
            }
            for row_index, (time_ns, eligible, residual) in enumerate(values)
        ],
        infer_schema_length=None,
    )


def _order(
    order_id: str,
    *,
    new: tuple[int, int, int],
    end: tuple[int, int, int],
    fill: tuple[int, int, int] | None = None,
    date: str = "20260803",
    value_code: str = "2330",
    quote_code: str = "CDFU6",
    rank: str = "BID1",
    supported: bool = True,
) -> dict[str, object]:
    fill_cursor = (None, None, None) if fill is None else fill
    return {
        "raw_order_fact_id": order_id,
        "Date": date,
        "ValueCode": value_code,
        "QuoteCode": quote_code,
        "exact_target_rank": rank,
        "actual_new_time_ns": new[0],
        "actual_new_event_sequence": new[1],
        "actual_new_row_index": new[2],
        "active_end_time_ns": end[0],
        "active_end_event_sequence": end[1],
        "active_end_row_index": end[2],
        "active_end_reason": "actual_cancel_send",
        "approximate_fill_time_ns": fill_cursor[0],
        "approximate_fill_event_sequence": fill_cursor[1],
        "approximate_fill_row_index": fill_cursor[2],
        "outcome_supported": supported,
    }


def _excursion_summary_row(
    date: str,
    value_code: str,
    sequence: int,
    *,
    touched: bool,
) -> dict[str, object]:
    row = {name: None for name in EXCURSION_SCHEMA}
    row.update(
        {
            "excursion_id": f"{date}/{value_code}/{sequence}",
            "Date": date,
            "month": date[:6],
            "ValueCode": value_code,
            "QuoteCode": "CDFU6",
            "excursion_sequence": sequence,
            "left_censored": False,
            "primary_observable": True,
            "completed": True,
            "end_reason": "center_return",
            "amplitude_bp": 10.0 + sequence,
            "upper_distance_bp": 5.0,
            "start_already_at_or_above_upper": False,
            "start_time_ns": sequence * 10,
            "start_event_sequence": RAW_SPOT_PHASE,
            "start_row_index": sequence,
            "start_residual_bp": 1.0,
            "previous_residual_bp": -1.0,
            "end_time_ns": sequence * 10 + 5,
            "end_event_sequence": RAW_SPOT_PHASE,
            "end_row_index": sequence + 1,
            "end_residual_bp": 0.0,
            "touch_time_ns": sequence * 10 + 2 if touched else None,
            "touch_event_sequence": RAW_SPOT_PHASE if touched else None,
            "touch_row_index": sequence if touched else None,
            "touch_residual_bp": 6.0 if touched else None,
            "touch_spread_pair_epoch": sequence if touched else None,
        }
    )
    return row


def _touch_link(
    order_id: str,
    date: str,
    value_code: str,
    *,
    filled: bool,
    rank: str = "BID1",
) -> dict[str, object]:
    row = {name: None for name in TOUCH_LINK_SCHEMA}
    row.update(
        {
            "raw_order_fact_id": order_id,
            "excursion_id": f"touch/{order_id}",
            "Date": date,
            "month": date[:6],
            "ValueCode": value_code,
            "QuoteCode": "CDFU6",
            "exact_target_rank": rank,
            "touch_time_ns": 20,
            "touch_event_sequence": RAW_SPOT_PHASE,
            "touch_row_index": 0,
            "touch_spread_pair_epoch": 1,
            "actual_new_time_ns": 10,
            "active_end_time_ns": 30,
            "active_end_reason": "actual_cancel_send",
            "approximate_fill_time_ns": 25 if filled else None,
            "outcome_supported": True,
            "queue_denominator_supported": True,
            "post_touch_fill": filled,
        }
    )
    return row


def _quote_inputs(
    specs: list[dict[str, object]],
    *,
    date: str = "20260803",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    base_ns = _session_second_ns(date, 0)
    event_rows: list[dict[str, object]] = []
    outcome_rows: list[dict[str, object]] = []
    for spec in specs:
        generation = int(spec["generation"])
        value_code = str(spec.get("ValueCode", f"V{generation:03d}"))
        quote_code = str(spec.get("QuoteCode", f"Q{generation:03d}"))
        price_tick = int(spec.get("absolute_price_tick", 1_000 + generation))
        submit_second = int(spec.get("submit_second", 300))
        cancel_second = int(spec.get("cancel_second", 310))
        potential_time_ns = spec.get("potential_fill_time_ns")
        common = {
            "scenario_id": SCENARIO_ID,
            "Date": date,
            "ValueCode": value_code,
            "absolute_price_tick": price_tick,
            "generation": generation,
            "submit_point_offset": 0,
            "submit_point_bucket": "BID1",
        }
        event_rows.extend(
            (
                {
                    **common,
                    "second_from_open": submit_second,
                    "kind": "submit",
                    "reason": "initial_eligible",
                    "is_cutoff": False,
                },
                {
                    **common,
                    "second_from_open": cancel_second,
                    "kind": "cancel",
                    "reason": (
                        "entry_cutoff"
                        if cancel_second >= ENTRY_DRAIN_START_SECOND
                        else "target_retreat"
                    ),
                    "is_cutoff": cancel_second >= ENTRY_DRAIN_START_SECOND,
                },
            )
        )
        submit_ns = base_ns + submit_second * 1_000_000_000
        stop_ns = base_ns + cancel_second * 1_000_000_000
        supported = bool(spec.get("outcome_supported", True))
        outcome_rows.append(
            {
                "Date": date,
                "ValueCode": value_code,
                "generation": generation,
                "QuoteCode": quote_code,
                "absolute_price_tick": price_tick,
                "target_price": float(price_tick),
                "submit_point_offset": 0,
                "exact_target_rank": "BID1",
                "initial_displayed_lots": 10,
                "boundary_quantile": 95,
                "upper_distance_bp": 20.0,
                "lower_distance_bp": 15.0,
                "boundary_source_asof_date": "20260731",
                "submit_decision_time_ns": (
                    submit_ns + int(spec.get("snapshot_mismatch_ns", 0))
                ),
                "nominal_stop_time_ns": stop_ns,
                "nominal_stop_reason": "target_retreat",
                "outcome_supported": supported,
                "outcome_status": "synthetic",
                "makerfill_mapping_exact": supported,
                "makerfill_fill_seconds": (
                    None if potential_time_ns is None else 1.0
                ),
                "makerfill_implied_fill_time_ns": potential_time_ns,
                "fill_cursor_exact": False,
                "own_quantity_included": False,
                "partial_fill_included": False,
                "cancel_ack_observed": False,
                "joint_volume_allocated": False,
            }
        )
    return (
        pl.from_dicts(event_rows, infer_schema_length=None),
        pl.from_dicts(
            outcome_rows,
            schema_overrides={"makerfill_implied_fill_time_ns": pl.Int64},
            infer_schema_length=None,
        ),
    )


def _verify_quote_outputs(
    orders: pl.DataFrame,
    assignments: pl.DataFrame,
    audit: dict[str, object],
) -> dict[str, object]:
    manifest = orders.select("Date", "ValueCode", "QuoteCode")
    product_days = manifest.with_columns(
        pl.lit(20.0).alias("upper_distance_bp"),
        pl.lit(1, dtype=pl.Int64).alias("eligible_raw_state_count"),
    )
    excursions = pl.DataFrame(schema=EXCURSION_SCHEMA).with_columns(
        pl.lit(None, dtype=pl.String).alias("analysis_window")
    )
    touch_links = pl.DataFrame(schema=TOUCH_LINK_SCHEMA)
    monthly = summarize_monthly_attribution(
        product_days,
        excursions,
        orders,
        touch_links,
    )
    rank = summarize_post_touch_by_rank(touch_links)
    membership = _membership_sensitivity(
        product_days,
        excursions,
        orders,
        touch_links,
        touch_links,
    )
    daily_audit = pl.from_dicts(
        [{"Date": "20260803", **audit}],
        infer_schema_length=None,
    )
    return _verify_domain_frames(
        manifest,
        product_days,
        excursions,
        orders,
        touch_links,
        touch_links,
        assignments,
        monthly,
        rank,
        membership,
        pl.DataFrame(),
        daily_audit,
    )


class AugustAttributionTest(unittest.TestCase):
    def test_venue_merge_preserves_explicit_book_clear(self) -> None:
        common = {
            "Date": "20260803",
            "ValueCode": "2330",
            "QuoteCode": "CDFU6",
        }
        spot = pl.from_dicts(
            [
                {
                    **common,
                    "cursor_time_ns": 10,
                    "cursor_event_sequence": RAW_SPOT_PHASE,
                    "cursor_row_index": 1,
                    "event_source": "spot",
                    "spot_formal": True,
                    "spot_bid": 100.0,
                    "spot_ask": 101.0,
                    "spot_bid_lots": 5,
                    "spot_ask_lots": 6,
                    "spot_ref_price": 100.5,
                    "spread_pair_epoch": 1,
                },
                {
                    **common,
                    "cursor_time_ns": 30,
                    "cursor_event_sequence": RAW_SPOT_PHASE,
                    "cursor_row_index": 2,
                    "event_source": "spot",
                    "spot_formal": True,
                    "spot_bid": 100.0,
                    "spot_ask": None,
                    "spot_bid_lots": 5,
                    "spot_ask_lots": None,
                    "spot_ref_price": 100.5,
                    "spread_pair_epoch": 2,
                },
                {
                    **common,
                    "cursor_time_ns": 50,
                    "cursor_event_sequence": RAW_SPOT_PHASE,
                    "cursor_row_index": 3,
                    "event_source": "spot",
                    "spot_formal": True,
                    "spot_bid": 99.0,
                    "spot_ask": 100.0,
                    "spot_bid_lots": 7,
                    "spot_ask_lots": 8,
                    "spot_ref_price": 100.5,
                    "spread_pair_epoch": 3,
                },
                {
                    **common,
                    "cursor_time_ns": 60,
                    "cursor_event_sequence": RAW_SPOT_PHASE,
                    "cursor_row_index": 4,
                    "event_source": "spot",
                    "spot_formal": True,
                    "spot_bid": 99.0,
                    "spot_ask": 100.0,
                    "spot_bid_lots": 7,
                    "spot_ask_lots": 8,
                    "spot_ref_price": 100.5,
                    "spread_pair_epoch": 4,
                },
            ],
            infer_schema_length=None,
        )
        future = pl.from_dicts(
            [
                {
                    **common,
                    "cursor_time_ns": time_ns,
                    "cursor_event_sequence": RAW_FUTURE_PHASE,
                    "cursor_row_index": row_index,
                    "event_source": "future",
                    "future_formal": True,
                    "future_bid": 110.0,
                    "future_ask": 111.0,
                    "future_bid_lots": 2,
                    "future_ask_lots": 3,
                    "future_exec_bid": 110.0,
                    "future_exec_ask": 111.0,
                    "future_exec_bid_lots": 2,
                    "future_exec_ask_lots": 3,
                    "future_ref_price": 110.5,
                }
                for row_index, time_ns in enumerate((20, 40), start=1)
            ]
            + [
                {
                    **common,
                    "cursor_time_ns": 60,
                    "cursor_event_sequence": RAW_FUTURE_PHASE,
                    "cursor_row_index": 3,
                    "event_source": "future",
                    "future_formal": True,
                    "future_bid": None,
                    "future_ask": 111.0,
                    "future_bid_lots": None,
                    "future_ask_lots": 3,
                    "future_exec_bid": None,
                    "future_exec_ask": 111.0,
                    "future_exec_bid_lots": None,
                    "future_exec_ask_lots": 3,
                    "future_ref_price": 110.5,
                }
            ],
            infer_schema_length=None,
        )

        merged = _merge_venue_state_events(spot, future)
        by_cursor = {
            (
                int(row["cursor_time_ns"]),
                int(row["cursor_event_sequence"]),
            ): row
            for row in merged.iter_rows(named=True)
        }
        self.assertEqual(by_cursor[(20, RAW_FUTURE_PHASE)]["spot_ask"], 101.0)
        self.assertIsNone(by_cursor[(30, RAW_SPOT_PHASE)]["spot_ask"])
        self.assertIsNone(by_cursor[(40, RAW_FUTURE_PHASE)]["spot_ask"])
        self.assertEqual(by_cursor[(50, RAW_SPOT_PHASE)]["spot_ask"], 100.0)
        self.assertEqual(by_cursor[(40, RAW_FUTURE_PHASE)]["spread_pair_epoch"], 2)
        self.assertIsNone(by_cursor[(60, RAW_FUTURE_PHASE)]["future_bid"])
        self.assertIsNone(by_cursor[(60, RAW_SPOT_PHASE)]["future_bid"])

        analyzed = _add_raw_analysis_fields(
            merged.with_columns(
                pl.lit(900.0).alias("anchor_ewma_120s_bp"),
                pl.lit(20.0).alias("upper_distance_bp"),
            )
        )
        eligible = {
            (
                int(row["cursor_time_ns"]),
                int(row["cursor_event_sequence"]),
            ): bool(row["analysis_eligible_raw"])
            for row in analyzed.iter_rows(named=True)
        }
        self.assertTrue(eligible[(20, RAW_FUTURE_PHASE)])
        self.assertFalse(eligible[(30, RAW_SPOT_PHASE)])
        self.assertFalse(eligible[(40, RAW_FUTURE_PHASE)])
        self.assertTrue(eligible[(50, RAW_SPOT_PHASE)])
        self.assertFalse(eligible[(60, RAW_FUTURE_PHASE)])
        self.assertFalse(eligible[(60, RAW_SPOT_PHASE)])

    def test_quote_loop_fill_terminal_and_same_cursor_cancel_noop(self) -> None:
        base_ns = _session_second_ns("20260803", 0)
        events, outcomes = _quote_inputs(
            [
                {
                    "generation": 1,
                    "cancel_second": 310,
                    "potential_fill_time_ns": base_ns + 305_000_000_000,
                },
                {
                    "generation": 2,
                    "cancel_second": 305,
                    "potential_fill_time_ns": base_ns + 305_000_000_000,
                },
                {
                    "generation": 3,
                    "cancel_second": 304,
                    "potential_fill_time_ns": base_ns + 305_000_000_000,
                },
            ]
        )

        orders, assignments, audit = build_quote_only_orders(
            "20260803", events, outcomes
        )
        by_generation = {
            int(row["generation"]): row
            for row in orders.iter_rows(named=True)
        }

        before_cancel = by_generation[1]
        self.assertTrue(before_cancel["potential_fill_accepted"])
        self.assertEqual(
            before_cancel["active_end_reason"],
            "accepted_approximate_fill",
        )
        self.assertTrue(before_cancel["cancel_not_needed"])
        self.assertFalse(before_cancel["cancel_request_sent"])
        self.assertIsNone(before_cancel["actual_cancel_time_ns"])
        self.assertEqual(
            before_cancel["adapter_outcome_status"],
            "accepted_approximate_fill",
        )
        self.assertEqual(
            before_cancel["legacy_nominal_outcome_status"], "synthetic"
        )

        same_cursor = by_generation[2]
        self.assertTrue(same_cursor["potential_fill_accepted"])
        self.assertTrue(same_cursor["cancel_request_sent"])
        self.assertTrue(same_cursor["cancel_send_noop"])
        self.assertFalse(same_cursor["cancel_not_needed"])
        self.assertIsNone(same_cursor["actual_cancel_time_ns"])

        after_cancel = by_generation[3]
        self.assertFalse(after_cancel["potential_fill_accepted"])
        self.assertEqual(
            after_cancel["potential_fill_rejection_reason"],
            "not_working_actual_cancelled",
        )
        self.assertTrue(after_cancel["cancel_send_effective"])
        self.assertIsNotNone(after_cancel["actual_cancel_time_ns"])
        self.assertEqual(
            after_cancel["adapter_outcome_status"],
            "effective_actual_cancel_before_potential_fill",
        )

        cancel_assignments = assignments.filter(pl.col("kind") == "cancel")
        self.assertEqual(cancel_assignments.height, 2)
        self.assertNotIn(
            before_cancel["cancel_request_id"],
            set(cancel_assignments["request_id"].to_list()),
        )
        self.assertEqual(audit["accepted_approximate_fills"], 2)
        self.assertEqual(audit["cancel_not_needed"], 1)
        self.assertEqual(audit["cancel_assigned_noop"], 1)

    def test_quote_loop_classifies_non_fill_adapter_outcomes(self) -> None:
        date = "20260803"
        base_ns = _session_second_ns(date, 0)
        events, outcomes = _quote_inputs(
            [
                {
                    "generation": 1,
                    "submit_second": 300,
                    "cancel_second": 301,
                    "potential_fill_time_ns": base_ns + 300_000_000_000,
                },
                {
                    "generation": 2,
                    "cancel_second": 301,
                    "potential_fill_time_ns": base_ns + 300_500_000_000,
                    "outcome_supported": False,
                },
                {"generation": 3, "cancel_second": 301},
                {"generation": 4, "cancel_second": 16_001},
                {
                    "generation": 5,
                    "cancel_second": 16_001,
                    "potential_fill_time_ns": base_ns + 16_000_500_000_000,
                },
            ],
            date=date,
        )

        orders, assignments, audit = build_quote_only_orders(
            date, events, outcomes
        )
        statuses = dict(
            orders.select("generation", "adapter_outcome_status").iter_rows()
        )

        self.assertEqual(
            statuses,
            {
                1: (
                    "effective_actual_cancel_after_pre_working_"
                    "potential_fill"
                ),
                2: (
                    "effective_actual_cancel_with_unsupported_"
                    "potential_fill"
                ),
                3: "effective_actual_cancel_no_potential_fill",
                4: "effective_actual_cancel_no_potential_fill",
                5: "effective_actual_cancel_before_potential_fill",
            },
        )
        self.assertEqual(
            orders.filter(pl.col("generation") == 1).item(
                0, "potential_fill_rejection_reason"
            ),
            "not_working_intent_pending",
        )
        _verify_quote_outputs(orders, assignments, audit)

    def test_quote_loop_removes_unassigned_pending_cutoff_cancel(self) -> None:
        date = "20260803"
        base_ns = _session_second_ns(date, 0)
        drain_ns = base_ns + ENTRY_DRAIN_START_SECOND * 1_000_000_000
        specs = [
            {
                "generation": generation,
                "submit_second": 300 if generation <= 50 else 301,
                "cancel_second": ENTRY_DRAIN_START_SECOND,
                "absolute_price_tick": 1_000 + generation,
                "potential_fill_time_ns": (
                    drain_ns + 500_000_000 if generation == 1 else None
                ),
            }
            for generation in range(1, 102)
        ]
        events, outcomes = _quote_inputs(specs, date=date)

        orders, assignments, audit = build_quote_only_orders(
            date, events, outcomes
        )
        lowest = orders.filter(pl.col("generation") == 1).row(
            0, named=True
        )
        sent_cancel_ids = set(
            assignments.filter(pl.col("kind") == "cancel")[
                "request_id"
            ].to_list()
        )

        self.assertTrue(lowest["potential_fill_accepted"])
        self.assertTrue(lowest["cancel_not_needed"])
        self.assertFalse(lowest["cancel_request_sent"])
        self.assertNotIn(lowest["cancel_request_id"], sent_cancel_ids)
        self.assertEqual(assignments.filter(pl.col("kind") == "cancel").height, 100)
        self.assertEqual(audit["cancel_pending_removed_on_fill"], 1)
        self.assertEqual(audit["actual_new_delayed"], 0)

    def test_quote_loop_rejects_actual_new_snapshot_mismatch(self) -> None:
        events, outcomes = _quote_inputs(
            [{"generation": 1, "snapshot_mismatch_ns": 1}]
        )
        with self.assertRaisesRegex(ValueError, "actual-new snapshot differs"):
            build_quote_only_orders("20260803", events, outcomes)

    def test_verifier_recomputes_fill_terminal_and_cancel_noop_contract(self) -> None:
        base_ns = _session_second_ns("20260803", 0)
        events, outcomes = _quote_inputs(
            [
                {
                    "generation": 1,
                    "cancel_second": 305,
                    "potential_fill_time_ns": base_ns + 305_000_000_000,
                }
            ]
        )
        orders, assignments, audit = build_quote_only_orders(
            "20260803", events, outcomes
        )

        verification = _verify_quote_outputs(orders, assignments, audit)

        self.assertEqual(verification["accepted_approximate_fills"], 1)
        self.assertEqual(verification["assigned_cancel_noops"], 1)
        tampered = orders.with_columns(
            pl.col("cancel_request_send_time_ns").alias(
                "actual_cancel_time_ns"
            )
        )
        with self.assertRaisesRegex(ValueError, "effective actual cancel"):
            _verify_quote_outputs(tampered, assignments, audit)

    def test_canonical_requires_tests_full_dates_and_content_hashes(self) -> None:
        paths = AttributionPaths()
        records = [
            {
                "path": str(paths.manifest_path.resolve()),
                "hash_scope": "full_content",
                "sha256": MANIFEST_SHA256,
                "content_sha256": True,
            }
        ]
        records.extend(
            {
                "path": f"/input/source_{index:03d}.parquet",
                "hash_scope": "full_content",
                "sha256": f"{index:064x}",
                "content_sha256": True,
            }
            for index in range(EXPECTED_INPUT_RECORD_COUNT - 1)
        )
        dates = [
            "20260504",
            *(f"middle_{index:02d}" for index in range(EXPECTED_SESSION_COUNT - 2)),
            "20260813",
        ]
        inventory = {"records": records}
        manifest = pl.DataFrame(
            {"row": range(EXPECTED_PRODUCT_DAY_COUNT)}
        )
        checks = _canonical_checks(
            paths,
            requested_dates=None,
            selected_dates=dates,
            manifest=manifest,
            git_state={"dirty": False},
            tests={"status": "pass"},
            input_inventory=inventory,
            raw_tape_reconstruction_verified=True,
        )
        self.assertTrue(all(checks.values()))

        unverified_raw = _canonical_checks(
            paths,
            requested_dates=None,
            selected_dates=dates,
            manifest=manifest,
            git_state={"dirty": False},
            tests={"status": "pass"},
            input_inventory=inventory,
        )
        self.assertFalse(
            unverified_raw["raw_tape_reconstruction_verified"]
        )

        skipped = _canonical_checks(
            paths,
            requested_dates=None,
            selected_dates=dates,
            manifest=manifest,
            git_state={"dirty": False},
            tests={"status": "skipped_debug"},
            input_inventory=inventory,
            raw_tape_reconstruction_verified=True,
        )
        self.assertFalse(skipped["focused_tests_passed"])

        subset = _canonical_checks(
            paths,
            requested_dates=[dates[0]],
            selected_dates=dates,
            manifest=manifest,
            git_state={"dirty": False},
            tests={"status": "pass"},
            input_inventory=inventory,
            raw_tape_reconstruction_verified=True,
        )
        self.assertFalse(subset["full_manifest_date_run"])

        records[-1] = {
            **records[-1],
            "hash_scope": "path_bytes_mtime",
            "content_sha256": False,
        }
        unhashed = _canonical_checks(
            paths,
            requested_dates=None,
            selected_dates=dates,
            manifest=manifest,
            git_state={"dirty": False},
            tests={"status": "pass"},
            input_inventory=inventory,
            raw_tape_reconstruction_verified=True,
        )
        self.assertFalse(unhashed["all_inputs_full_content_sha256"])

    def test_observable_crossing_records_first_touch_and_center_return(self) -> None:
        excursions = extract_positive_excursions(
            _raw_states(
                [
                    (10, True, -1.0),
                    (20, True, 2.0),
                    (30, True, 6.0),
                    (40, True, 8.0),
                    (50, True, 0.0),
                ]
            )
        )

        self.assertEqual(excursions.height, 1)
        row = excursions.row(0, named=True)
        self.assertTrue(row["primary_observable"])
        self.assertFalse(row["left_censored"])
        self.assertTrue(row["completed"])
        self.assertEqual(row["end_reason"], "center_return")
        self.assertAlmostEqual(row["previous_residual_bp"], -1.0)
        self.assertEqual(row["start_time_ns"], 20)
        self.assertEqual(row["touch_time_ns"], 30)
        self.assertEqual(row["touch_spread_pair_epoch"], 102)
        self.assertEqual(row["end_time_ns"], 50)
        self.assertAlmostEqual(row["amplitude_bp"], 8.0)

    def test_session_start_above_upper_is_left_censored_without_touch(self) -> None:
        excursions = extract_positive_excursions(
            _raw_states(
                [
                    (10, True, 6.0),
                    (20, True, 2.0),
                    (30, True, 7.0),
                    (40, True, 0.0),
                ]
            )
        )

        self.assertEqual(excursions.height, 1)
        row = excursions.row(0, named=True)
        self.assertTrue(row["left_censored"])
        self.assertEqual(row["left_censor_reason"], "session_start")
        self.assertFalse(row["primary_observable"])
        self.assertTrue(row["start_already_at_or_above_upper"])
        self.assertIsNone(row["touch_time_ns"])
        self.assertTrue(row["completed"])

    def test_eligibility_gap_censors_both_sides_of_the_gap(self) -> None:
        excursions = extract_positive_excursions(
            _raw_states(
                [
                    (10, True, -1.0),
                    (20, True, 2.0),
                    (30, False, None),
                    (40, True, 6.0),
                    (50, True, 0.0),
                ]
            )
        ).sort("excursion_sequence")

        self.assertEqual(excursions.height, 2)
        before = excursions.row(0, named=True)
        after = excursions.row(1, named=True)
        self.assertTrue(before["primary_observable"])
        self.assertFalse(before["completed"])
        self.assertEqual(before["end_reason"], "eligibility_gap")
        self.assertEqual(before["end_time_ns"], 20)
        self.assertTrue(after["left_censored"])
        self.assertEqual(after["left_censor_reason"], "eligibility_gap")
        self.assertTrue(after["start_already_at_or_above_upper"])
        self.assertIsNone(after["touch_time_ns"])

    def test_touch_to_order_interval_respects_same_time_phases(self) -> None:
        excursions = extract_positive_excursions(
            _raw_states(
                [
                    (80, True, -1.0),
                    (90, True, 2.0),
                    (100, True, 6.0),
                    (110, True, 0.0),
                ]
            )
        )
        orders = pl.from_dicts(
            [
                _order(
                    "cancel-at-touch-time",
                    new=(90, ACTUAL_NEW_PHASE, 0),
                    end=(100, ACTUAL_CANCEL_PHASE, 0),
                ),
                _order(
                    "new-at-touch-time",
                    new=(100, ACTUAL_NEW_PHASE, 0),
                    end=(110, ACTUAL_CANCEL_PHASE, 0),
                ),
                _order(
                    "fill-before-touch-phase",
                    new=(90, ACTUAL_NEW_PHASE, 1),
                    end=(110, ACTUAL_CANCEL_PHASE, 1),
                    fill=(100, RAW_FUTURE_PHASE, 0),
                ),
                _order(
                    "fill-after-touch-phase",
                    new=(90, ACTUAL_NEW_PHASE, 2),
                    end=(100, ACTUAL_CANCEL_PHASE, 2),
                    fill=(100, APPROXIMATE_FILL_PHASE, 0),
                ),
            ],
            infer_schema_length=None,
        )

        links = link_first_touches_to_orders(excursions, orders)

        self.assertEqual(
            links["raw_order_fact_id"].to_list(),
            ["cancel-at-touch-time", "fill-after-touch-phase"],
        )
        linked = {
            row["raw_order_fact_id"]: row
            for row in links.iter_rows(named=True)
        }
        self.assertFalse(linked["cancel-at-touch-time"]["post_touch_fill"])
        self.assertTrue(linked["fill-after-touch-phase"]["post_touch_fill"])

    def test_full_touch_pairs_precede_order_level_deduplication(self) -> None:
        excursions = extract_positive_excursions(
            _raw_states(
                [
                    (10, True, -1.0),
                    (20, True, 2.0),
                    (30, True, 6.0),
                    (40, True, 0.0),
                    (50, True, 2.0),
                    (60, True, 6.0),
                    (70, True, 0.0),
                ]
            )
        )
        orders = pl.from_dicts(
            [
                _order(
                    "both-touches",
                    new=(15, ACTUAL_NEW_PHASE, 0),
                    end=(70, ACTUAL_CANCEL_PHASE, 0),
                ),
                _order(
                    "second-touch-only",
                    new=(55, ACTUAL_NEW_PHASE, 1),
                    end=(70, ACTUAL_CANCEL_PHASE, 1),
                ),
            ],
            infer_schema_length=None,
        )

        pairs = link_touch_order_pairs(excursions, orders)
        touched_orders = deduplicate_touch_pairs_to_orders(pairs)

        self.assertEqual(pairs.height, 3)
        self.assertEqual(
            pairs.filter(pl.col("raw_order_fact_id") == "both-touches")[
                "touch_time_ns"
            ].to_list(),
            [30, 60],
        )
        self.assertEqual(touched_orders.height, 2)
        self.assertEqual(
            touched_orders.filter(
                pl.col("raw_order_fact_id") == "both-touches"
            )["touch_time_ns"].item(),
            30,
        )

    def test_raw_order_identity_includes_full_start_cursor(self) -> None:
        base = {
            "Date": "20260803",
            "ValueCode": "2330",
            "QuoteCode": "CDFU6",
            "absolute_price_tick": 100,
            "actual_new_time_ns": 123,
            "actual_new_event_sequence": ACTUAL_NEW_PHASE,
            "actual_new_row_index": 1,
        }
        later_sequence = {**base, "actual_new_row_index": 2}

        self.assertNotEqual(
            _raw_order_id(base),
            _raw_order_id(later_sequence),
        )

    def test_monthly_pooled_and_product_day_equal_rates_and_decomposition(self) -> None:
        product_days = pl.from_dicts(
            [
                {
                    "Date": date,
                    "ValueCode": value_code,
                    "upper_distance_bp": 5.0,
                    "eligible_raw_state_count": 10,
                }
                for date, value_code in (
                    ("20260504", "A"),
                    ("20260504", "B"),
                    ("20260601", "C"),
                    ("20260701", "D"),
                    ("20260803", "A"),
                    ("20260803", "B"),
                    ("20260803", "C"),
                    ("20260803", "D"),
                )
            ],
            infer_schema_length=None,
        )
        excursion_specs = [
            ("20260504", "A", True),
            ("20260504", "B", True),
            ("20260601", "C", True),
            ("20260701", "D", True),
            ("20260803", "A", True),
            ("20260803", "B", True),
            ("20260803", "C", False),
            ("20260803", "D", False),
        ]
        excursions = pl.from_dicts(
            [
                _excursion_summary_row(
                    date,
                    value_code,
                    sequence,
                    touched=touched,
                )
                for sequence, (date, value_code, touched) in enumerate(
                    excursion_specs, start=1
                )
            ],
            schema=EXCURSION_SCHEMA,
            infer_schema_length=None,
        )
        link_specs = [
            *(("20260504", "A", True) for _ in range(4)),
            ("20260504", "B", False),
            ("20260601", "C", True),
            ("20260701", "D", True),
            ("20260803", "A", True),
            ("20260803", "A", False),
            ("20260803", "B", False),
        ]
        touch_links = pl.from_dicts(
            [
                _touch_link(
                    f"order-{index}",
                    date,
                    value_code,
                    filled=filled,
                    rank="BID1",
                )
                for index, (date, value_code, filled) in enumerate(link_specs)
            ],
            schema=TOUCH_LINK_SCHEMA,
            infer_schema_length=None,
        )
        raw_orders = pl.from_dicts(
            [
                {
                    "Date": row["Date"],
                    "ValueCode": row["ValueCode"],
                    "outcome_supported": True,
                }
                for row in touch_links.iter_rows(named=True)
            ],
            infer_schema_length=None,
        )

        monthly = summarize_monthly_attribution(
            product_days,
            excursions,
            raw_orders,
            touch_links,
        )
        may = monthly.filter(pl.col("month") == "202605").row(
            0, named=True
        )
        august = monthly.filter(pl.col("month") == "202608").row(
            0, named=True
        )
        self.assertAlmostEqual(may["post_touch_fill_rate_pooled"], 4 / 5)
        self.assertAlmostEqual(
            may["post_touch_fill_rate_product_day_equal"], 1 / 2
        )
        self.assertEqual(may["product_days_in_queue_equal_weight"], 2)
        self.assertAlmostEqual(august["excursion_touch_rate"], 1 / 2)
        self.assertAlmostEqual(
            august["post_touch_fill_rate_pooled"], 1 / 3
        )
        self.assertAlmostEqual(
            august["post_touch_fill_rate_product_day_equal"], 1 / 4
        )

        rank = summarize_post_touch_by_rank(touch_links)
        may_rank = rank.filter(
            (pl.col("month") == "202605")
            & (pl.col("exact_target_rank") == "BID1")
        ).row(0, named=True)
        self.assertAlmostEqual(may_rank["post_touch_fill_rate_pooled"], 4 / 5)
        self.assertAlmostEqual(
            may_rank["post_touch_fill_rate_product_day_equal"], 1 / 2
        )

        decomposition = build_decomposition(monthly).row(0, named=True)
        self.assertAlmostEqual(
            decomposition["prior_excursion_touch_rate"], 1.0
        )
        self.assertAlmostEqual(
            decomposition["prior_post_touch_fill_rate"], 6 / 7
        )
        self.assertTrue(
            decomposition["market_boundary_directionally_lower"]
        )
        self.assertTrue(decomposition["queue_directionally_lower"])
        self.assertEqual(
            decomposition["classification"],
            "market_boundary_and_queue",
        )
        self.assertFalse(decomposition["causal_claim"])


if __name__ == "__main__":
    unittest.main()
