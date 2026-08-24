"""Pure one-second controller for buy-side maker message-load studies.

This module deliberately models only quote intent.  It has no tape, fill, or
file-system dependencies, so a caller can feed one observation per second and
aggregate the returned submit/cancel actions into venue request counts.

The controller keys working orders by their *absolute* legal price tick.  A
change in book rank or ``SpreadPairTotalCount`` therefore does not replace an
otherwise unchanged order.  The epoch is intentionally not an input: callers
may sample a new epoch, but only an actual target/gate transition can emit a
message.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MessageKind = Literal["submit", "cancel"]


@dataclass(frozen=True)
class WorkingBidQuote:
    """One live buy-side maker generation at an absolute price."""

    generation: int
    absolute_price_tick: int
    submit_second: int
    submit_point_offset: int


@dataclass(frozen=True)
class QuoteMessageAction:
    """One exchange request emitted by :class:`OneSecondBidQuoteController`.

    ``submit_point_offset`` is deliberately retained on cancel actions.  It
    lets the load report attribute every later cancel to the point offset at
    which that exact generation was originally admitted.
    """

    kind: MessageKind
    reason: str
    second: int
    absolute_price_tick: int
    generation: int
    submit_point_offset: int


class OneSecondBidQuoteController:
    """Reconcile one-second buy-side maker targets into sparse messages.

    Rules are intentionally based on absolute price rather than rank:

    * the first eligible target, or an eligible target after a gate reopens,
      submits once;
    * an unchanged target does not resubmit merely because its epoch or point
      bucket changed;
    * a higher, previously inactive target adds an independent layer;
    * a retreat cancels every live price above the new target and does not
      submit the lower target solely because of that retreat;
    * a closed gate and session cutoff cancel every live layer.

    One instance is scoped to one product-day.  ``second`` must be strictly
    increasing because its actions are intended for per-second aggregation.
    """

    def __init__(self) -> None:
        self._last_second: int | None = None
        self._current_target_tick: int | None = None
        self._last_admission_eligible = False
        self._has_observation = False
        self._gate_open = False
        self._cutoff_reached = False
        self._next_generation = 1
        self._active_by_price: dict[int, WorkingBidQuote] = {}

    @property
    def current_target_tick(self) -> int | None:
        return self._current_target_tick

    @property
    def active_quotes(self) -> tuple[WorkingBidQuote, ...]:
        """Live generations in deterministic admission order."""
        return tuple(
            sorted(
                self._active_by_price.values(),
                key=lambda quote: quote.generation,
            )
        )

    @property
    def cutoff_reached(self) -> bool:
        return self._cutoff_reached

    def reconcile(
        self,
        second: int,
        target_tick: int | None,
        base_gate_open: bool,
        admission_open: bool,
        submit_point_offset: int | None,
        *,
        gate_reason: str = "gate_closed",
    ) -> tuple[QuoteMessageAction, ...]:
        """Apply one observed target state and return actual quote requests.

        ``admission_open`` is a submit-time filter (for example, AB1/AB2),
        not a blanket cancellation gate.  Thus a live absolute-price order is
        allowed to keep aging if its displayed point bucket later changes.
        ``submit_point_offset`` is the caller's signed tick offset (normally
        target minus current B1, so 0 is B1 and -1 is one tick behind).  It is
        frozen only when a generation is submitted.  Use
        ``base_gate_open=False`` for conditions that require immediate cancel.
        """
        self._validate_second(second)
        if self._cutoff_reached:
            raise RuntimeError("cannot reconcile after session cutoff")
        if not isinstance(base_gate_open, bool):
            raise TypeError("base_gate_open must be boolean")
        if not isinstance(admission_open, bool):
            raise TypeError("admission_open must be boolean")
        target_tick = self._validate_target(target_tick, base_gate_open)
        point_offset = self._validate_point_offset(
            submit_point_offset,
            required=base_gate_open and admission_open,
        )
        if not isinstance(gate_reason, str) or not gate_reason:
            raise ValueError("gate_reason must be a non-empty string")

        previous_target = self._current_target_tick
        was_gate_open = self._gate_open
        had_observation = self._has_observation
        previous_eligible = self._last_admission_eligible

        actions: list[QuoteMessageAction] = []
        if not base_gate_open:
            actions.extend(
                self._cancel_all(
                    second,
                    reason=gate_reason,
                )
            )
            if target_tick is not None:
                self._current_target_tick = target_tick
            self._record_observation(second)
            self._gate_open = False
            self._last_admission_eligible = False
            return tuple(actions)

        assert target_tick is not None
        actions.extend(
            self._cancel_prices_above(
                target_tick,
                second,
            )
        )

        quote_is_live = target_tick in self._active_by_price
        reason: str | None = None
        if admission_open and not quote_is_live:
            if not had_observation:
                reason = "initial_eligible"
            elif not was_gate_open:
                reason = "gate_reopen"
            elif previous_target is not None and target_tick > previous_target:
                reason = "forward_new_price"
            elif (
                previous_target == target_tick
                and not previous_eligible
                and admission_open
            ):
                reason = "became_admission_eligible"

        if reason is not None:
            assert point_offset is not None
            actions.append(
                self._submit(
                    target_tick,
                    point_offset,
                    second,
                    reason,
                )
            )

        self._current_target_tick = target_tick
        self._record_observation(second)
        self._gate_open = True
        self._last_admission_eligible = admission_open
        return tuple(actions)

    def close(
        self,
        second: int,
        reason: str = "session_cutoff",
    ) -> tuple[QuoteMessageAction, ...]:
        """Cancel every live layer and make the controller terminal."""
        self._validate_second(second)
        if self._cutoff_reached:
            raise RuntimeError("session cutoff was already applied")
        if not isinstance(reason, str) or not reason:
            raise ValueError("reason must be a non-empty string")

        actions = self._cancel_all(second, reason=reason)
        self._record_observation(second)
        self._gate_open = False
        self._cutoff_reached = True
        return tuple(actions)

    def _submit(
        self,
        price_tick: int,
        point_offset: int,
        second: int,
        reason: str,
    ) -> QuoteMessageAction:
        if price_tick in self._active_by_price:
            raise RuntimeError("one active generation per absolute price")
        quote = WorkingBidQuote(
            generation=self._next_generation,
            absolute_price_tick=price_tick,
            submit_second=second,
            submit_point_offset=point_offset,
        )
        self._next_generation += 1
        self._active_by_price[price_tick] = quote
        return self._action("submit", reason, second, quote)

    def _cancel_prices_above(
        self,
        target_tick: int,
        second: int,
    ) -> list[QuoteMessageAction]:
        quotes = tuple(
            quote
            for quote in self.active_quotes
            if quote.absolute_price_tick > target_tick
        )
        return self._cancel_quotes(
            quotes,
            second,
            reason="target_retreat",
        )

    def _cancel_all(
        self,
        second: int,
        *,
        reason: str,
    ) -> list[QuoteMessageAction]:
        return self._cancel_quotes(
            self.active_quotes,
            second,
            reason=reason,
        )

    def _cancel_quotes(
        self,
        quotes: tuple[WorkingBidQuote, ...],
        second: int,
        *,
        reason: str,
    ) -> list[QuoteMessageAction]:
        actions: list[QuoteMessageAction] = []
        for quote in quotes:
            if self._active_by_price.pop(quote.absolute_price_tick, None) is None:
                continue
            actions.append(self._action("cancel", reason, second, quote))
        return actions

    @staticmethod
    def _action(
        kind: MessageKind,
        reason: str,
        second: int,
        quote: WorkingBidQuote,
    ) -> QuoteMessageAction:
        return QuoteMessageAction(
            kind=kind,
            reason=reason,
            second=second,
            absolute_price_tick=quote.absolute_price_tick,
            generation=quote.generation,
            submit_point_offset=quote.submit_point_offset,
        )

    def _validate_second(self, second: int) -> None:
        if isinstance(second, bool) or not isinstance(second, int) or second < 0:
            raise ValueError("second must be a non-negative integer")
        if self._last_second is not None and second <= self._last_second:
            raise ValueError("second must be strictly increasing")

    @staticmethod
    def _validate_target(target_tick: int | None, gate_open: bool) -> int | None:
        if target_tick is None:
            if gate_open:
                raise ValueError("an open gate requires target_price_tick")
            return None
        if (
            isinstance(target_tick, bool)
            or not isinstance(target_tick, int)
            or target_tick <= 0
        ):
            raise ValueError("target_price_tick must be a positive integer")
        return target_tick

    @staticmethod
    def _validate_point_offset(
        point_offset: int | None,
        *,
        required: bool,
    ) -> int | None:
        if point_offset is None:
            if required:
                raise ValueError("an eligible target requires submit_point_offset")
            return None
        if isinstance(point_offset, bool) or not isinstance(point_offset, int):
            raise TypeError("submit_point_offset must be an integer")
        return point_offset

    def _record_observation(self, second: int) -> None:
        self._last_second = second
        self._has_observation = True
