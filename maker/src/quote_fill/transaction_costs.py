"""Transaction cost profile for one long-spot / short-futures paired cycle.

Extracted from the deleted fixed-45 ``combined_cost_cap_sweep`` module on
2026-08-24; the numbers are the user-specified Taiwan stock/stock-futures fees.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


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
