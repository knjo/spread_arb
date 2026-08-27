from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta

import polars as pl

from ..quote_fill.layered import EventCursor
from ..quote_fill.s1_spot_close_adapter import (
    SPOT_CLOSE_EVENT_SEQUENCE,
    OfficialSpotClose,
    SpotCloseDayIndex,
    official_spot_close_source_id,
)

BASE = datetime(2026, 5, 20, 5, 30, 0)  # noqa: DTZ001


def ns(second: int) -> int:
    return int(
        pl.Series([BASE + timedelta(seconds=second)]).dt.timestamp("ns")[0]
    )


def mapping() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ValueCode": ["2317", "2330", "2603"],
            "QuoteCode": ["DHFE6", "CDFE6", "CZFE6"],
        }
    )


def row(
    quote_code: str,
    value_code: str,
    *,
    second: int,
    channel: int,
    packet: int,
    close: float | None,
) -> dict[str, object]:
    return {
        "RecvTime": BASE + timedelta(seconds=second),
        "QuoteCode": quote_code,
        "ValueCode": value_code,
        "ChannelSeq": channel,
        "PacketSeq": packet,
        "Close": close,
    }


class SpotCloseDayIndexTest(unittest.TestCase):
    def test_uses_latest_post_boundary_official_close_with_provenance(self) -> None:
        rows = pl.from_dicts(
            [
                row(
                    "2330",
                    "2330",
                    second=-1,
                    channel=1,
                    packet=1,
                    close=500.0,
                ),
                row(
                    "2317",
                    "2317",
                    second=0,
                    channel=2,
                    packet=2,
                    close=100.0,
                ),
                row(
                    "2330",
                    "2330",
                    second=1,
                    channel=3,
                    packet=3,
                    close=505.0,
                ),
                row(
                    "2330",
                    "2330",
                    second=2,
                    channel=4,
                    packet=4,
                    close=506.0,
                ),
                row(
                    "2603",
                    "2603",
                    second=3,
                    channel=5,
                    packet=5,
                    close=None,
                ),
                row(
                    "9999",
                    "9999",
                    second=4,
                    channel=6,
                    packet=6,
                    close=50.0,
                ),
            ],
            infer_schema_length=None,
        )
        index = SpotCloseDayIndex.from_selected_scan(
            rows.lazy(),
            mapping(),
            date="20260520",
            close_not_before_ns=ns(0),
        )
        self.assertEqual(index.product_ids, ("2317", "2330", "2603"))
        self.assertEqual(index.available_count, 2)
        fact = index.close_fact("2330")
        assert fact is not None
        self.assertEqual(fact.close_price, 506.0)
        self.assertEqual(fact.source_cursor.recv_time_ns, ns(2))
        self.assertEqual(fact.source_cursor.event_sequence, SPOT_CLOSE_EVENT_SEQUENCE)
        self.assertEqual(fact.channel_sequence, 4)
        self.assertEqual(fact.packet_sequence, 4)
        self.assertEqual(fact.source_row, 3)
        self.assertEqual(
            fact.source_id,
            official_spot_close_source_id(
                "20260520",
                "2330",
                "CDFE6",
                ns(2),
                4,
                4,
                3,
            ),
        )
        self.assertIsNone(index.close_fact("2603"))
        with self.assertRaises(KeyError):
            index.close_fact("9999")
        self.assertEqual(
            [value.product_id for value in index.facts],
            ["2317", "2330"],
        )

    def test_fact_is_immutable_and_rejects_forged_source(self) -> None:
        source_id = official_spot_close_source_id(
            "20260520",
            "2330",
            "CDFE6",
            ns(0),
            1,
            2,
            3,
        )
        fact = OfficialSpotClose(
            "20260520",
            "2330",
            "CDFE6",
            500.0,
            EventCursor(ns(0), SPOT_CLOSE_EVENT_SEQUENCE, 0),
            source_id,
            1,
            2,
            3,
        )
        with self.assertRaises(FrozenInstanceError):
            fact.close_price = 1.0  # type: ignore[misc]
        with self.assertRaisesRegex(ValueError, "source_id"):
            replace(fact, source_id="forged")

    def test_rejects_mapping_ambiguity_and_raw_value_mismatch(self) -> None:
        rows = pl.from_dicts(
            [
                row(
                    "2330",
                    "2317",
                    second=0,
                    channel=1,
                    packet=1,
                    close=500.0,
                )
            ]
        )
        with self.assertRaisesRegex(ValueError, "ValueCode"):
            SpotCloseDayIndex.from_selected_rows(
                rows,
                mapping(),
                date="20260520",
                close_not_before_ns=ns(0),
            )

        ambiguous = pl.DataFrame(
            {
                "ValueCode": ["2330", "2330"],
                "QuoteCode": ["CDFE6", "CDFF6"],
            }
        )
        with self.assertRaisesRegex(ValueError, "one-to-one"):
            SpotCloseDayIndex.from_selected_rows(
                rows,
                ambiguous,
                date="20260520",
                close_not_before_ns=ns(0),
            )

    def test_rejects_pre_boundary_only_and_bad_schema_cleanly(self) -> None:
        rows = pl.from_dicts(
            [
                row(
                    "2330",
                    "2330",
                    second=-1,
                    channel=1,
                    packet=1,
                    close=500.0,
                )
            ]
        )
        index = SpotCloseDayIndex.from_selected_rows(
            rows,
            mapping(),
            date="20260520",
            close_not_before_ns=ns(0),
        )
        self.assertEqual(index.available_count, 0)
        with self.assertRaisesRegex(ValueError, "missing columns"):
            SpotCloseDayIndex.from_selected_rows(
                rows.drop("Close"),
                mapping(),
                date="20260520",
                close_not_before_ns=ns(0),
            )


if __name__ == "__main__":
    unittest.main()
