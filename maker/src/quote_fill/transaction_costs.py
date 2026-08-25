"""Transaction cost profile for one long-spot / short-futures paired cycle.

Extracted from the deleted fixed-45 ``combined_cost_cap_sweep`` module on
2026-08-24; the numbers are the user-specified Taiwan stock/stock-futures fees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PairedCycleCostBreakdown:
    """Exact per-leg charges for one completed paired cycle."""

    spot_entry_commission_twd: float
    spot_exit_commission_twd: float
    spot_exit_tax_twd: float
    futures_entry_tax_twd: float
    futures_exit_tax_twd: float
    futures_entry_commission_twd: float
    futures_exit_commission_twd: float

    @property
    def total_twd(self) -> float:
        return sum(
            (
                self.spot_entry_commission_twd,
                self.spot_exit_commission_twd,
                self.spot_exit_tax_twd,
                self.futures_entry_tax_twd,
                self.futures_exit_tax_twd,
                self.futures_entry_commission_twd,
                self.futures_exit_commission_twd,
            )
        )


@dataclass(frozen=True)
class TransactionCostProfile:
    """Charges for one complete long-spot/short-futures paired cycle."""

    profile_id: str = "user_taiwan_stock_future_20260821_v1"
    spot_commission_listed_bp_per_side: float = 14.25
    spot_commission_multiplier: float = 0.12
    spot_sell_tax_bp: float = 30.0
    same_day_spot_sell_tax_multiplier: float = 0.5
    futures_tax_bp_per_side: float = 0.2
    futures_commission_twd_per_side: float = 20.0

    @property
    def spot_commission_bp_per_side(self) -> float:
        return (
            self.spot_commission_listed_bp_per_side
            * self.spot_commission_multiplier
        )

    @property
    def same_day_variable_cost_bp(self) -> float:
        return (
            2.0 * self.spot_commission_bp_per_side
            + self.spot_sell_tax_bp * self.same_day_spot_sell_tax_multiplier
            + 2.0 * self.futures_tax_bp_per_side
        )

    @property
    def overnight_variable_cost_bp(self) -> float:
        return (
            2.0 * self.spot_commission_bp_per_side
            + self.spot_sell_tax_bp
            + 2.0 * self.futures_tax_bp_per_side
        )

    @property
    def futures_round_trip_commission_twd(self) -> float:
        return 2.0 * self.futures_commission_twd_per_side

    def spot_commission_twd(self, price: float, shares: float) -> float:
        """Return the commission for one executed spot leg."""

        self.validate()
        price = _positive_finite(price, "price")
        shares = _positive_finite(shares, "shares")
        return (
            price
            * shares
            * self.spot_commission_bp_per_side
            / 10_000.0
        )

    def spot_sell_tax_twd(
        self,
        price: float,
        shares: float,
        *,
        same_day: bool,
    ) -> float:
        """Return tax for one executed spot sell leg.

        ``same_day`` describes the actually matched spot quantity.  It does
        not follow from a futures hedge or a nominal strategy branch.
        """

        self.validate()
        price = _positive_finite(price, "price")
        shares = _positive_finite(shares, "shares")
        if not isinstance(same_day, bool):
            raise TypeError("same_day must be boolean")
        multiplier = self.same_day_spot_sell_tax_multiplier if same_day else 1.0
        return (
            price
            * shares
            * self.spot_sell_tax_bp
            * multiplier
            / 10_000.0
        )

    def futures_tax_twd(self, price: float, share_equivalent: float) -> float:
        """Return transaction tax for one executed stock-futures leg."""

        self.validate()
        price = _positive_finite(price, "price")
        share_equivalent = _positive_finite(
            share_equivalent,
            "share_equivalent",
        )
        return (
            price
            * share_equivalent
            * self.futures_tax_bp_per_side
            / 10_000.0
        )

    def futures_commission_twd(self, contracts: float) -> float:
        """Return broker commission for one executed futures leg."""

        self.validate()
        contracts = _positive_finite(contracts, "contracts")
        return contracts * self.futures_commission_twd_per_side

    def paired_cycle_cost_breakdown(
        self,
        *,
        entry_spot_price: float,
        exit_spot_price: float,
        entry_future_price: float,
        exit_future_price: float,
        shares: float,
        contracts: float = 1.0,
        same_day: bool,
    ) -> PairedCycleCostBreakdown:
        """Price every executed leg of a completed paired cycle exactly."""

        if not isinstance(same_day, bool):
            raise TypeError("same_day must be boolean")
        return PairedCycleCostBreakdown(
            spot_entry_commission_twd=self.spot_commission_twd(
                entry_spot_price,
                shares,
            ),
            spot_exit_commission_twd=self.spot_commission_twd(
                exit_spot_price,
                shares,
            ),
            spot_exit_tax_twd=self.spot_sell_tax_twd(
                exit_spot_price,
                shares,
                same_day=same_day,
            ),
            futures_entry_tax_twd=self.futures_tax_twd(
                entry_future_price,
                shares,
            ),
            futures_exit_tax_twd=self.futures_tax_twd(
                exit_future_price,
                shares,
            ),
            futures_entry_commission_twd=self.futures_commission_twd(
                contracts
            ),
            futures_exit_commission_twd=self.futures_commission_twd(
                contracts
            ),
        )

    def paired_cycle_cost_twd(
        self,
        *,
        entry_spot_price: float,
        exit_spot_price: float,
        entry_future_price: float,
        exit_future_price: float,
        shares: float,
        contracts: float = 1.0,
        same_day: bool,
    ) -> float:
        """Return the total exact cost for one completed paired cycle."""

        return self.paired_cycle_cost_breakdown(
            entry_spot_price=entry_spot_price,
            exit_spot_price=exit_spot_price,
            entry_future_price=entry_future_price,
            exit_future_price=exit_future_price,
            shares=shares,
            contracts=contracts,
            same_day=same_day,
        ).total_twd

    def validate(self) -> None:
        if not self.profile_id:
            raise ValueError("transaction cost profile_id must be nonempty")
        names = (
            "spot_commission_listed_bp_per_side",
            "spot_commission_multiplier",
            "spot_sell_tax_bp",
            "same_day_spot_sell_tax_multiplier",
            "futures_tax_bp_per_side",
            "futures_commission_twd_per_side",
        )
        for name in names:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.same_day_spot_sell_tax_multiplier > 1:
            raise ValueError("same-day tax multiplier cannot exceed one")


def _positive_finite(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and positive") from error
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result
