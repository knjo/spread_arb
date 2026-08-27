from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from dataclasses import replace
from pathlib import Path

from maker.src.quote_fill.capacity_ledger import (
    CAPACITY_IDENTITY_REGISTRY_RECEIPT_SCHEMA_VERSION,
    CapacityIdentityRegistryReceipt,
    CapacityLedger,
    CapacityTransition,
)
from maker.src.quote_fill.s1_capacity_identity_registry import (
    CapacityIdentityRegistryConflictError,
    CapacityIdentityRegistryDuplicateError,
    CapacityIdentityRegistryError,
    CapacityIdentityRegistryIntegrityError,
    CapacityIdentityRegistrySchemaError,
    S1CapacityIdentityRegistry,
)


def _admitted_transition(
    *,
    sequence: int,
    transition_id: str,
    capacity_id: str,
    timestamp_ns: int | None = None,
) -> CapacityTransition:
    ledger = CapacityLedger(global_cap_twd=100, product_cap_twd=100)
    row = ledger.attempt_new_reservation(
        transition_id="template-transition",
        timestamp_ns=sequence if timestamp_ns is None else timestamp_ns,
        capacity_id="template-capacity",
        product_id="A",
        requested_notional_twd=10,
    ).transition
    return replace(
        row,
        sequence=sequence,
        transition_id=transition_id,
        capacity_id=capacity_id,
    )


def _receipt(
    *,
    transition_count: int,
    admitted_count: int,
    digest_character: str,
) -> CapacityIdentityRegistryReceipt:
    return CapacityIdentityRegistryReceipt(
        schema_version=CAPACITY_IDENTITY_REGISTRY_RECEIPT_SCHEMA_VERSION,
        transition_count=transition_count,
        admitted_count=admitted_count,
        registry_sha256=digest_character * 64,
    )


class S1CapacityIdentityRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self._temporary_directory.name) / "registry.sqlite"
        self.registry = S1CapacityIdentityRegistry(self.database_path)

    def tearDown(self) -> None:
        self.registry.close()
        self._temporary_directory.cleanup()

    def test_two_partitions_register_blocked_retry_only_when_admitted(self) -> None:
        ledger = CapacityLedger(global_cap_twd=10, product_cap_twd=10)
        ledger.attempt_new_reservation(
            transition_id="d1-admitted",
            timestamp_ns=1,
            capacity_id="cap-a",
            product_id="A",
            requested_notional_twd=10,
        )
        blocked = ledger.attempt_new_reservation(
            transition_id="d1-blocked",
            timestamp_ns=2,
            capacity_id="cap-retry",
            product_id="B",
            requested_notional_twd=10,
        )
        self.assertFalse(blocked.admitted)
        day_one_count = len(ledger.transitions)

        first_receipt = self.registry.commit_partition(
            "policy-a",
            "20260505",
            ledger.transitions,
            expected_previous_receipt=None,
        )
        self.assertEqual(first_receipt.transition_count, 2)
        self.assertEqual(first_receipt.admitted_count, 1)
        self.assertEqual(
            ledger.to_compact_checkpoint(
                "20260505",
                identity_registry_receipt=first_receipt,
            ).identity_registry_receipt,
            first_receipt,
        )
        self.assertEqual(
            self.registry.commit_partition(
                "policy-a",
                "20260505",
                ledger.transitions,
                expected_previous_receipt=None,
            ),
            first_receipt,
        )

        ledger.release_working_leaves(
            transition_id="d2-release",
            timestamp_ns=3,
            capacity_id="cap-a",
            reason="actual_cancel",
        )
        admitted_retry = ledger.attempt_new_reservation(
            transition_id="d2-retry",
            timestamp_ns=4,
            capacity_id="cap-retry",
            product_id="B",
            requested_notional_twd=10,
        )
        self.assertTrue(admitted_retry.admitted)
        second_receipt = self.registry.commit_partition(
            "policy-a",
            "20260506",
            ledger.transitions[day_one_count:],
            expected_previous_receipt=first_receipt,
        )
        self.assertEqual(second_receipt.transition_count, 4)
        self.assertEqual(second_receipt.admitted_count, 2)
        self.assertEqual(
            ledger.to_compact_checkpoint(
                "20260506",
                identity_registry_receipt=second_receipt,
            ).transition_sequence_offset,
            4,
        )
        self.assertEqual(self.registry.latest("policy-a"), second_receipt)
        self.assertEqual(self.registry.verify("policy-a"), second_receipt)

        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            transition_ids = connection.execute(
                """
                SELECT transition_id
                FROM transition_identities
                WHERE policy_id = 'policy-a'
                ORDER BY sequence
                """
            ).fetchall()
            admitted_ids = connection.execute(
                """
                SELECT capacity_id
                FROM admitted_capacity_identities
                WHERE policy_id = 'policy-a'
                ORDER BY transition_sequence
                """
            ).fetchall()
        self.assertEqual(
            [row[0] for row in transition_ids],
            ["d1-admitted", "d1-blocked", "d2-release", "d2-retry"],
        )
        self.assertEqual([row[0] for row in admitted_ids], ["cap-a", "cap-retry"])

    def test_readonly_tip_preserves_zero_transition_partition_lineage(self) -> None:
        first = (
            _admitted_transition(
                sequence=1,
                transition_id="d1",
                capacity_id="cap-a",
            ),
        )
        first_receipt = self.registry.commit_partition(
            "policy-a",
            "20260505",
            first,
            expected_previous_receipt=None,
        )
        second_receipt = self.registry.commit_partition(
            "policy-a",
            "20260506",
            (),
            expected_previous_receipt=first_receipt,
        )
        self.assertEqual(
            second_receipt.transition_count, first_receipt.transition_count
        )
        self.assertNotEqual(
            second_receipt.registry_sha256, first_receipt.registry_sha256
        )

        self.registry.close()
        self.registry = S1CapacityIdentityRegistry(
            self.database_path,
            readonly=True,
        )
        tip = self.registry.latest_tip("policy-a")

        self.assertIsNotNone(tip)
        assert tip is not None
        self.assertEqual(tip.policy_id, "policy-a")
        self.assertEqual(tip.partition_date, "20260506")
        self.assertEqual(tip.previous_partition_date, "20260505")
        self.assertEqual(tip.receipt, second_receipt)
        self.assertEqual(tip.previous_receipt, first_receipt)
        self.assertEqual(self.registry.verify("policy-a"), second_receipt)
        with self.assertRaisesRegex(
            CapacityIdentityRegistryError,
            "readonly identity registry",
        ):
            self.registry.commit_partition(
                "policy-a",
                "20260507",
                (),
                expected_previous_receipt=second_receipt,
            )

    def test_readonly_open_does_not_create_a_missing_database(self) -> None:
        missing = Path(self._temporary_directory.name) / "missing.sqlite"

        with self.assertRaisesRegex(
            CapacityIdentityRegistryError,
            "does not exist",
        ):
            S1CapacityIdentityRegistry(missing, readonly=True)

        self.assertFalse(missing.exists())

    def test_same_partition_is_idempotent_but_content_drift_and_stale_retry_fail(
        self,
    ) -> None:
        day_one = (
            _admitted_transition(
                sequence=1,
                transition_id="d1",
                capacity_id="cap-a",
            ),
        )
        first_receipt = self.registry.commit_partition(
            "policy-a",
            "20260505",
            day_one,
            expected_previous_receipt=None,
        )
        self.registry.close()
        self.registry = S1CapacityIdentityRegistry(self.database_path)
        self.assertEqual(
            self.registry.commit_partition(
                "policy-a",
                "20260505",
                day_one,
                expected_previous_receipt=None,
            ),
            first_receipt,
        )
        with self.assertRaisesRegex(
            CapacityIdentityRegistryConflictError,
            "content differs",
        ):
            self.registry.commit_partition(
                "policy-a",
                "20260505",
                (replace(day_one[0], timestamp_ns=999),),
                expected_previous_receipt=None,
            )

        day_two = (
            _admitted_transition(
                sequence=2,
                transition_id="d2",
                capacity_id="cap-b",
            ),
        )
        self.registry.commit_partition(
            "policy-a",
            "20260506",
            day_two,
            expected_previous_receipt=first_receipt,
        )
        with self.assertRaisesRegex(
            CapacityIdentityRegistryConflictError,
            "latest partition",
        ):
            self.registry.commit_partition(
                "policy-a",
                "20260505",
                day_one,
                expected_previous_receipt=None,
            )

    def test_historical_duplicates_fail_atomically_but_other_policy_is_independent(
        self,
    ) -> None:
        first = (
            _admitted_transition(
                sequence=1,
                transition_id="shared-transition",
                capacity_id="shared-capacity",
            ),
        )
        first_receipt = self.registry.commit_partition(
            "policy-a",
            "20260505",
            first,
            expected_previous_receipt=None,
        )

        duplicate_transition = (
            _admitted_transition(
                sequence=2,
                transition_id="shared-transition",
                capacity_id="new-capacity",
            ),
        )
        with self.assertRaisesRegex(
            CapacityIdentityRegistryDuplicateError,
            "transition_id was already registered",
        ):
            self.registry.commit_partition(
                "policy-a",
                "20260506",
                duplicate_transition,
                expected_previous_receipt=first_receipt,
            )
        self.assertEqual(self.registry.latest("policy-a"), first_receipt)

        duplicate_capacity = (
            _admitted_transition(
                sequence=2,
                transition_id="new-transition",
                capacity_id="shared-capacity",
            ),
        )
        with self.assertRaisesRegex(
            CapacityIdentityRegistryDuplicateError,
            "capacity_id was already registered",
        ):
            self.registry.commit_partition(
                "policy-a",
                "20260506",
                duplicate_capacity,
                expected_previous_receipt=first_receipt,
            )
        self.assertEqual(self.registry.latest("policy-a"), first_receipt)
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            partition_count = connection.execute(
                """
                SELECT COUNT(*)
                FROM partition_receipts
                WHERE policy_id = 'policy-a'
                """
            ).fetchone()[0]
        self.assertEqual(partition_count, 1)

        independent = self.registry.commit_partition(
            "policy-b",
            "20260505",
            first,
            expected_previous_receipt=None,
        )
        self.assertEqual(independent.transition_count, 1)
        self.assertEqual(independent.admitted_count, 1)
        self.assertNotEqual(independent.registry_sha256, first_receipt.registry_sha256)
        self.assertEqual(
            self.registry.verify(),
            {"policy-a": first_receipt, "policy-b": independent},
        )

    def test_partition_internal_duplicates_are_rejected_before_sqlite_commit(
        self,
    ) -> None:
        first = _admitted_transition(
            sequence=1,
            transition_id="same-transition",
            capacity_id="cap-a",
        )
        second = _admitted_transition(
            sequence=2,
            transition_id="same-transition",
            capacity_id="cap-b",
        )
        with self.assertRaisesRegex(
            CapacityIdentityRegistryDuplicateError,
            "partition contains duplicate transition_id",
        ):
            self.registry.commit_partition(
                "policy-a",
                "20260505",
                (first, second),
                expected_previous_receipt=None,
            )

        distinct_transition = replace(
            second,
            transition_id="different-transition",
            capacity_id="cap-a",
        )
        with self.assertRaisesRegex(
            CapacityIdentityRegistryDuplicateError,
            "partition contains duplicate admitted capacity_id",
        ):
            self.registry.commit_partition(
                "policy-a",
                "20260505",
                (first, distinct_transition),
                expected_previous_receipt=None,
            )
        self.assertIsNone(self.registry.latest("policy-a"))

    def test_previous_receipt_date_and_global_sequence_are_strict(self) -> None:
        first = (
            _admitted_transition(
                sequence=1,
                transition_id="d1",
                capacity_id="cap-a",
            ),
        )
        fake = _receipt(
            transition_count=0,
            admitted_count=0,
            digest_character="f",
        )
        with self.assertRaisesRegex(
            CapacityIdentityRegistryConflictError,
            "expected_previous_receipt",
        ):
            self.registry.commit_partition(
                "policy-a",
                "20260505",
                first,
                expected_previous_receipt=fake,
            )
        first_receipt = self.registry.commit_partition(
            "policy-a",
            "20260505",
            first,
            expected_previous_receipt=None,
        )

        valid_second = (
            _admitted_transition(
                sequence=2,
                transition_id="d2",
                capacity_id="cap-b",
            ),
        )
        with self.assertRaisesRegex(
            CapacityIdentityRegistryConflictError,
            "expected_previous_receipt",
        ):
            self.registry.commit_partition(
                "policy-a",
                "20260506",
                valid_second,
                expected_previous_receipt=fake,
            )
        with self.assertRaisesRegex(
            CapacityIdentityRegistryConflictError,
            "strictly advance",
        ):
            self.registry.commit_partition(
                "policy-a",
                "20260504",
                valid_second,
                expected_previous_receipt=first_receipt,
            )
        with self.assertRaisesRegex(
            CapacityIdentityRegistryConflictError,
            "globally contiguous",
        ):
            self.registry.commit_partition(
                "policy-a",
                "20260506",
                (replace(valid_second[0], sequence=3),),
                expected_previous_receipt=first_receipt,
            )
        self.assertEqual(self.registry.latest("policy-a"), first_receipt)

        second_receipt = self.registry.commit_partition(
            "policy-a",
            "20260506",
            valid_second,
            expected_previous_receipt=first_receipt,
        )
        empty_receipt = self.registry.commit_partition(
            "policy-a",
            "20260507",
            (),
            expected_previous_receipt=second_receipt,
        )
        self.assertEqual(
            empty_receipt.transition_count,
            second_receipt.transition_count,
        )
        self.assertEqual(empty_receipt.admitted_count, second_receipt.admitted_count)
        self.assertNotEqual(
            empty_receipt.registry_sha256,
            second_receipt.registry_sha256,
        )
        self.assertEqual(self.registry.verify("policy-a"), empty_receipt)

    def test_fast_append_matches_integrity_first_receipts_and_final_verify(
        self,
    ) -> None:
        first = (
            _admitted_transition(
                sequence=1,
                transition_id="d1",
                capacity_id="cap-a",
            ),
        )
        second = (
            _admitted_transition(
                sequence=2,
                transition_id="d2",
                capacity_id="cap-b",
            ),
        )
        strict_first = self.registry.commit_partition(
            "policy-a",
            "20260505",
            first,
            expected_previous_receipt=None,
        )
        strict_second = self.registry.commit_partition(
            "policy-a",
            "20260506",
            second,
            expected_previous_receipt=strict_first,
        )

        fast_path = Path(self._temporary_directory.name) / "fast-registry.sqlite"
        fast_registry = S1CapacityIdentityRegistry(fast_path)
        try:
            fast_first = fast_registry.commit_partition(
                "policy-a",
                "20260505",
                first,
                expected_previous_receipt=None,
                verify_full_history_before_commit=False,
            )
            fast_second = fast_registry.commit_partition(
                "policy-a",
                "20260506",
                second,
                expected_previous_receipt=fast_first,
                verify_full_history_before_commit=False,
            )
            self.assertEqual(fast_first, strict_first)
            self.assertEqual(fast_second, strict_second)
            self.assertEqual(fast_registry.verify("policy-a"), strict_second)
        finally:
            fast_registry.close()

    def test_fast_append_still_rejects_lineage_duplicates_and_cached_drift(
        self,
    ) -> None:
        first = (
            _admitted_transition(
                sequence=1,
                transition_id="d1",
                capacity_id="cap-a",
            ),
        )
        first_receipt = self.registry.commit_partition(
            "policy-a",
            "20260505",
            first,
            expected_previous_receipt=None,
            verify_full_history_before_commit=False,
        )
        stale = _receipt(
            transition_count=0,
            admitted_count=0,
            digest_character="f",
        )
        second = (
            _admitted_transition(
                sequence=2,
                transition_id="d2",
                capacity_id="cap-b",
            ),
        )
        with self.assertRaisesRegex(
            CapacityIdentityRegistryConflictError,
            "expected_previous_receipt",
        ):
            self.registry.commit_partition(
                "policy-a",
                "20260506",
                second,
                expected_previous_receipt=stale,
                verify_full_history_before_commit=False,
            )
        with self.assertRaisesRegex(
            CapacityIdentityRegistryDuplicateError,
            "transition_id was already registered",
        ):
            self.registry.commit_partition(
                "policy-a",
                "20260506",
                (replace(second[0], transition_id="d1"),),
                expected_previous_receipt=first_receipt,
                verify_full_history_before_commit=False,
            )

        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.execute(
                """
                UPDATE policy_state
                SET registry_sha256 = ?
                WHERE policy_id = 'policy-a'
                """,
                ("e" * 64,),
            )
        with self.assertRaisesRegex(
            CapacityIdentityRegistryIntegrityError,
            "latest policy state disagrees",
        ):
            self.registry.commit_partition(
                "policy-a",
                "20260506",
                second,
                expected_previous_receipt=first_receipt,
                verify_full_history_before_commit=False,
            )

    def test_fast_append_still_rejects_schema_and_metadata_drift(self) -> None:
        for case in ("schema", "metadata"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "registry.sqlite"
                registry = S1CapacityIdentityRegistry(path)
                try:
                    with closing(sqlite3.connect(path)) as connection, connection:
                        if case == "schema":
                            connection.execute("PRAGMA user_version = 99")
                        else:
                            connection.execute(
                                """
                                UPDATE registry_meta
                                SET value = 'tampered'
                                WHERE key = 'registry_chain_domain'
                                """
                            )
                    with self.assertRaises(CapacityIdentityRegistrySchemaError):
                        registry.commit_partition(
                            "policy-a",
                            "20260505",
                            (
                                _admitted_transition(
                                    sequence=1,
                                    transition_id="d1",
                                    capacity_id="cap-a",
                                ),
                            ),
                            expected_previous_receipt=None,
                            verify_full_history_before_commit=False,
                        )
                finally:
                    registry.close()

    def test_fast_append_defers_historical_row_tamper_to_final_verify(self) -> None:
        first = (
            _admitted_transition(
                sequence=1,
                transition_id="d1",
                capacity_id="cap-a",
            ),
        )
        first_receipt = self.registry.commit_partition(
            "policy-a",
            "20260505",
            first,
            expected_previous_receipt=None,
            verify_full_history_before_commit=False,
        )
        second_receipt = self.registry.commit_partition(
            "policy-a",
            "20260506",
            (
                _admitted_transition(
                    sequence=2,
                    transition_id="d2",
                    capacity_id="cap-b",
                ),
            ),
            expected_previous_receipt=first_receipt,
            verify_full_history_before_commit=False,
        )
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.execute(
                """
                UPDATE transition_identities
                SET transition_id = 'tampered-d1'
                WHERE policy_id = 'policy-a' AND sequence = 1
                """
            )

        third_receipt = self.registry.commit_partition(
            "policy-a",
            "20260507",
            (
                _admitted_transition(
                    sequence=3,
                    transition_id="d3",
                    capacity_id="cap-c",
                ),
            ),
            expected_previous_receipt=second_receipt,
            verify_full_history_before_commit=False,
        )
        self.assertEqual(third_receipt.transition_count, 3)
        with self.assertRaises(CapacityIdentityRegistryIntegrityError):
            self.registry.verify("policy-a")

    def test_schema_version_metadata_and_table_definition_tamper_fail(self) -> None:
        cases = ("user_version", "metadata", "table_schema")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "registry.sqlite"
                registry = S1CapacityIdentityRegistry(path)
                try:
                    registry.commit_partition(
                        "policy-a",
                        "20260505",
                        (
                            _admitted_transition(
                                sequence=1,
                                transition_id="d1",
                                capacity_id="cap-a",
                            ),
                        ),
                        expected_previous_receipt=None,
                    )
                    with closing(sqlite3.connect(path)) as connection, connection:
                        if case == "user_version":
                            connection.execute("PRAGMA user_version = 99")
                        elif case == "metadata":
                            connection.execute(
                                """
                                UPDATE registry_meta
                                SET value = 'tampered'
                                WHERE key = 'registry_chain_domain'
                                """
                            )
                        else:
                            connection.execute(
                                "ALTER TABLE policy_state ADD COLUMN injected TEXT"
                            )
                    with self.assertRaises(CapacityIdentityRegistrySchemaError):
                        registry.verify("policy-a")
                finally:
                    registry.close()

    def test_database_identity_and_receipt_row_tamper_fail_full_verify(self) -> None:
        cases = (
            "transition_identity",
            "partition_receipt",
            "partition_input",
            "policy_state",
        )
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "registry.sqlite"
                registry = S1CapacityIdentityRegistry(path)
                try:
                    first_receipt = registry.commit_partition(
                        "policy-a",
                        "20260505",
                        (
                            _admitted_transition(
                                sequence=1,
                                transition_id="d1",
                                capacity_id="cap-a",
                            ),
                        ),
                        expected_previous_receipt=None,
                    )
                    with closing(sqlite3.connect(path)) as connection, connection:
                        if case == "transition_identity":
                            connection.execute(
                                """
                                UPDATE transition_identities
                                SET transition_id = 'tampered-transition'
                                WHERE policy_id = 'policy-a' AND sequence = 1
                                """
                            )
                        elif case == "partition_receipt":
                            connection.execute(
                                """
                                UPDATE partition_receipts
                                SET ordered_transition_ids_sha256 = ?
                                WHERE policy_id = 'policy-a'
                                """,
                                ("f" * 64,),
                            )
                        elif case == "partition_input":
                            connection.execute(
                                """
                                UPDATE partition_receipts
                                SET partition_input_sha256 = ?
                                WHERE policy_id = 'policy-a'
                                """,
                                ("e" * 64,),
                            )
                        else:
                            connection.execute(
                                """
                                UPDATE policy_state
                                SET registry_sha256 = ?
                                WHERE policy_id = 'policy-a'
                                """,
                                ("f" * 64,),
                            )
                    with self.assertRaises(CapacityIdentityRegistryIntegrityError):
                        registry.latest("policy-a")
                    with self.assertRaises(CapacityIdentityRegistryIntegrityError):
                        registry.commit_partition(
                            "policy-a",
                            "20260506",
                            (
                                _admitted_transition(
                                    sequence=2,
                                    transition_id="d2",
                                    capacity_id="cap-b",
                                ),
                            ),
                            expected_previous_receipt=first_receipt,
                        )
                finally:
                    registry.close()


if __name__ == "__main__":
    unittest.main()
