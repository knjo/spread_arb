"""Efficient legacy-makerFill rank extension for bounded diagnostics.

The production ``makerFill`` files contain A1/A2/B1/B2 only.  This module
reproduces their *legacy* EOD-looking rule for selected snapshots and extends
that rule to A3--A5/B3--B5 without materialising ten columns for every market
row.  It is deliberately a diagnostic adapter, not an execution label:

* ordering is the legacy per-symbol ``TransTime`` order;
* same-price volume fills after reaching the displayed quantity (``>= Q``);
* a trade through the quoted price fills immediately;
* there is no cancellation, own quantity, partial fill, or exact receive
  cursor in the result.

Callers must compare the result with a separately derived stop time and keep
the legacy approximation flags.  Exact MBP-contract labels continue to come
from :mod:`maker.src.quote_fill.indexed_replay`.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import heapq
import math
from typing import Literal, Sequence

import numpy as np
import polars as pl


LEGACY_RANK_STUDY_VERSION = "legacy_makerfill_rank_extension_v5_epsilon_exact"
SUPPORTED_LEVELS = (1, 2, 3, 4, 5)
LEGACY_PRICE_EPSILON = 1e-8


@dataclass(frozen=True)
class LegacyMakerFillLabel:
    """One selected snapshot/side/level under the historical label rule."""

    channel_sequence: int
    side: Literal["ask", "bid"]
    level: int
    target_price: float | None
    initial_displayed_lots: int | None
    fill_seconds: float | None
    fill_trans_time_us: int | None
    fill_channel_sequence: int | None
    fill_reason: Literal["same_price_displayed_queue", "trade_through"] | None
    label_available: bool
    outcome_exact: bool = False
    cancel_bounded: bool = False
    own_quantity_included: bool = False


@dataclass(frozen=True)
class _PriceTrades:
    trade_positions: tuple[int, ...]
    cumulative_quantity: tuple[int, ...]

    @classmethod
    def build(
        cls, positions: list[int], quantities: list[int]
    ) -> _PriceTrades:
        cumulative = [0]
        total = 0
        for quantity in quantities:
            total += quantity
            cumulative.append(total)
        return cls(tuple(positions), tuple(cumulative))

    def displayed_queue_fill_position(
        self, snapshot_position: int, displayed_lots: int
    ) -> int | None:
        left = bisect_right(self.trade_positions, snapshot_position)
        if left >= len(self.trade_positions):
            return None
        threshold = self.cumulative_quantity[left] + displayed_lots
        prefix_index = bisect_left(
            self.cumulative_quantity,
            threshold,
            left + 1,
        )
        if prefix_index >= len(self.cumulative_quantity):
            return None
        return self.trade_positions[prefix_index - 1]


class _FloatPriceSegmentTree:
    """Range min/max tree returning the first strict price crossing."""

    def __init__(self, prices: Sequence[float]) -> None:
        size = 1
        while size < len(prices):
            size *= 2
        self.length = len(prices)
        self.size = size
        self.minimum = [math.inf] * (2 * size)
        self.maximum = [-math.inf] * (2 * size)
        for index, price in enumerate(prices):
            leaf = size + index
            self.minimum[leaf] = float(price)
            self.maximum[leaf] = float(price)
        for node in range(size - 1, 0, -1):
            self.minimum[node] = min(
                self.minimum[node * 2], self.minimum[node * 2 + 1]
            )
            self.maximum[node] = max(
                self.maximum[node * 2], self.maximum[node * 2 + 1]
            )

    def first_less(self, left: int, threshold: float) -> int | None:
        return self._first(left, threshold, less=True, node=1, lo=0, hi=self.size)

    def first_greater(self, left: int, threshold: float) -> int | None:
        return self._first(left, threshold, less=False, node=1, lo=0, hi=self.size)

    def _first(
        self,
        query_left: int,
        threshold: float,
        *,
        less: bool,
        node: int,
        lo: int,
        hi: int,
    ) -> int | None:
        if hi <= query_left:
            return None
        if less:
            if self.minimum[node] >= threshold:
                return None
        elif self.maximum[node] <= threshold:
            return None
        if hi - lo == 1:
            return lo if lo < self.length else None
        middle = (lo + hi) // 2
        found = self._first(
            query_left,
            threshold,
            less=less,
            node=node * 2,
            lo=lo,
            hi=middle,
        )
        if found is not None:
            return found
        return self._first(
            query_left,
            threshold,
            less=less,
            node=node * 2 + 1,
            lo=middle,
            hi=hi,
        )


class LegacyMakerFillRankIndex:
    """O(N) build and O(log N) selected-snapshot legacy fill queries."""

    def __init__(self, raw_states: pl.DataFrame) -> None:
        required = {
            "TransTime",
            "ChannelSeq",
            "marketOpen",
            "FillPrice",
            "FillLots",
            *(f"AskPrice{level}" for level in SUPPORTED_LEVELS),
            *(f"AskLots{level}" for level in SUPPORTED_LEVELS),
            *(f"BidPrice{level}" for level in SUPPORTED_LEVELS),
            *(f"BidLots{level}" for level in SUPPORTED_LEVELS),
        }
        missing = sorted(required - set(raw_states.columns))
        if missing:
            raise ValueError(f"raw states missing required columns: {missing}")
        if raw_states.is_empty():
            raise ValueError("raw states cannot be empty")

        states = (
            raw_states.with_row_index("_source_ordinal")
            .filter(pl.col("marketOpen") == True)  # noqa: E712
            .sort(["TransTime", "_source_ordinal"])
            .with_columns(
                pl.col("TransTime").dt.timestamp("us").alias("_trans_time_us")
            )
        )
        if states.is_empty():
            raise ValueError("raw states contain no marketOpen rows")
        duplicate = states.group_by("ChannelSeq").len().filter(pl.col("len") != 1)
        if duplicate.height:
            raise ValueError("ChannelSeq must be unique within one product-day")
        self._states = states
        self._channel_to_position = {
            int(sequence): position
            for position, sequence in enumerate(states.get_column("ChannelSeq"))
        }
        self._trans_time_us = tuple(
            int(value) for value in states.get_column("_trans_time_us")
        )
        self._channel_sequences = tuple(
            int(value) for value in states.get_column("ChannelSeq")
        )

        trade_positions: list[int] = []
        trade_prices: list[float] = []
        by_price_positions: dict[float, list[int]] = {}
        by_price_quantities: dict[float, list[int]] = {}
        for position, row in enumerate(
            states.select("FillPrice", "FillLots").iter_rows(named=True)
        ):
            price = _finite_positive_float(row["FillPrice"])
            quantity = _positive_int(row["FillLots"])
            if price is None or quantity is None:
                continue
            trade_positions.append(position)
            trade_prices.append(price)
            by_price_positions.setdefault(price, []).append(position)
            by_price_quantities.setdefault(price, []).append(quantity)
        self._trade_positions = tuple(trade_positions)
        self._trade_prices = tuple(trade_prices)
        self._trade_tree = _FloatPriceSegmentTree(self._trade_prices)
        self._same_price = {
            price: _PriceTrades.build(
                positions, by_price_quantities[price]
            )
            for price, positions in by_price_positions.items()
        }
        self._sorted_trade_prices = tuple(sorted(self._same_price))
        self._near_price_cache: dict[
            tuple[str, float], _PriceTrades | None
        ] = {}

    def label(
        self,
        channel_sequence: int,
        side: Literal["ask", "bid"],
        level: int,
    ) -> LegacyMakerFillLabel:
        if side not in {"ask", "bid"}:
            raise ValueError("side must be 'ask' or 'bid'")
        if level not in SUPPORTED_LEVELS:
            raise ValueError(f"level must be one of {SUPPORTED_LEVELS}")
        try:
            snapshot_position = self._channel_to_position[int(channel_sequence)]
        except KeyError as exc:
            raise KeyError(f"unknown ChannelSeq: {channel_sequence}") from exc
        prefix = "Ask" if side == "ask" else "Bid"
        row = self._states.row(snapshot_position, named=True)
        target_price = _finite_positive_float(row[f"{prefix}Price{level}"])
        displayed_lots = _positive_int(row[f"{prefix}Lots{level}"])
        common = dict(
            channel_sequence=int(channel_sequence),
            side=side,
            level=level,
            target_price=target_price,
            initial_displayed_lots=displayed_lots,
        )
        if target_price is None or displayed_lots is None:
            return LegacyMakerFillLabel(
                **common,
                fill_seconds=None,
                fill_trans_time_us=None,
                fill_channel_sequence=None,
                fill_reason=None,
                label_available=False,
            )

        same_position = None
        same_series = self._legacy_same_price_series(side, target_price)
        if same_series is not None:
            same_position = same_series.displayed_queue_fill_position(
                snapshot_position, displayed_lots
            )

        trade_left = bisect_right(self._trade_positions, snapshot_position)
        if side == "ask":
            through_trade_index = self._trade_tree.first_greater(
                trade_left, target_price
            )
        else:
            through_trade_index = self._trade_tree.first_less(
                trade_left, target_price
            )
        through_position = (
            None
            if through_trade_index is None
            else self._trade_positions[through_trade_index]
        )
        if through_position is not None and (
            same_position is None or through_position < same_position
        ):
            fill_position = through_position
            reason: Literal[
                "same_price_displayed_queue", "trade_through"
            ] = "trade_through"
        elif same_position is not None:
            fill_position = same_position
            reason = "same_price_displayed_queue"
        else:
            return LegacyMakerFillLabel(
                **common,
                fill_seconds=None,
                fill_trans_time_us=None,
                fill_channel_sequence=None,
                fill_reason=None,
                label_available=True,
            )

        start_us = self._trans_time_us[snapshot_position]
        fill_us = self._trans_time_us[fill_position]
        seconds = float(np.float32((fill_us - start_us) / 1_000_000.0))
        return LegacyMakerFillLabel(
            **common,
            fill_seconds=seconds,
            fill_trans_time_us=fill_us,
            fill_channel_sequence=self._channel_sequences[fill_position],
            fill_reason=reason,
            label_available=True,
        )

    def _legacy_same_price_series(
        self,
        side: Literal["ask", "bid"],
        target_price: float,
    ) -> _PriceTrades | None:
        """Return prints matching the producer's asymmetric epsilon rule.

        The historical Numba producer tests strict trade-through first and
        only then applies ``abs(fill_price - target) < 1e-8``.  Consequently,
        ask same-price candidates are ``(target-eps, target]`` while bid
        candidates are ``[target, target+eps)``.  This deliberately does not
        use rounded-price or exact-float dictionary equality.
        """

        cache_key = (side, target_price)
        if cache_key in self._near_price_cache:
            return self._near_price_cache[cache_key]
        prices = self._sorted_trade_prices
        # A direct bisect at ``target +/- 1e-8`` is subtly wrong: binary
        # floating subtraction may round the boundary differently from the
        # producer's later ``abs(p-target) < 1e-8`` expression.  Search a
        # deliberately wider two-epsilon band, then apply the literal legacy
        # predicate and the strict-through precedence exactly.
        left = bisect_left(
            prices, target_price - 2.0 * LEGACY_PRICE_EPSILON
        )
        right = bisect_right(
            prices, target_price + 2.0 * LEGACY_PRICE_EPSILON
        )
        selected = tuple(
            price
            for price in prices[left:right]
            if abs(price - target_price) < LEGACY_PRICE_EPSILON
            and (
                (side == "ask" and price <= target_price)
                or (side == "bid" and price >= target_price)
            )
        )
        if not selected:
            result = None
        elif len(selected) == 1:
            result = self._same_price[selected[0]]
        else:
            streams = []
            for price in selected:
                series = self._same_price[price]
                quantities = [
                    series.cumulative_quantity[index + 1]
                    - series.cumulative_quantity[index]
                    for index in range(len(series.trade_positions))
                ]
                streams.append(zip(series.trade_positions, quantities))
            merged = list(heapq.merge(*streams, key=lambda item: item[0]))
            result = _PriceTrades.build(
                [position for position, _ in merged],
                [quantity for _, quantity in merged],
            )
        self._near_price_cache[cache_key] = result
        return result


def label_selected_snapshots(
    raw_states: pl.DataFrame,
    requests: pl.DataFrame,
) -> pl.DataFrame:
    """Label selected ``ChannelSeq``/side/level requests once each.

    Required request columns are ``request_id``, ``ChannelSeq``, ``side``, and
    ``level``.  Unsupported/missing requests fail loudly; study runners may
    catch and materialise an explicit support status at a higher layer.
    """

    required = {"request_id", "ChannelSeq", "side", "level"}
    missing = sorted(required - set(requests.columns))
    if missing:
        raise ValueError(f"requests missing required columns: {missing}")
    if requests.select("request_id").n_unique() != requests.height:
        raise ValueError("request_id must be unique")
    index = LegacyMakerFillRankIndex(raw_states)
    records = []
    for row in requests.iter_rows(named=True):
        label = index.label(
            int(row["ChannelSeq"]),
            str(row["side"]),  # type: ignore[arg-type]
            int(row["level"]),
        )
        records.append({"request_id": row["request_id"], **label.__dict__})
    return pl.from_dicts(records, infer_schema_length=None)


def _finite_positive_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _positive_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None
