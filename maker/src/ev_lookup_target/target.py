"""Translate an annual net-return objective without assuming it is achievable."""
from dataclasses import asdict, dataclass
from math import isfinite


@dataclass(frozen=True)
class ReturnTarget:
    capital_twd: float = 20_000_000.0
    annual_net_return: float = 0.30
    trading_days_per_year: int = 250
    funding_annual_rate: float = 0.02

    def __post_init__(self):
        if not all(isfinite(v) and v >= 0 for v in
                   (self.capital_twd, self.annual_net_return, self.funding_annual_rate)):
            raise ValueError('capital and rates must be finite and nonnegative')
        if self.capital_twd <= 0 or self.trading_days_per_year <= 0:
            raise ValueError('capital and trading days must be positive')

    @property
    def annual_net_twd(self):
        return self.capital_twd * self.annual_net_return

    @property
    def daily_net_twd(self):
        return self.annual_net_twd / self.trading_days_per_year

    def per_trade_twd(self, completed_trades_per_day):
        if not isfinite(completed_trades_per_day) or completed_trades_per_day <= 0:
            raise ValueError('trade rate must be positive')
        return self.daily_net_twd / completed_trades_per_day

    def capital_day_hurdle_bp(self, calendar_capital_utilization=1.0):
        """Net bp per invested calendar day, assuming the specified annual utilization.

        The funding rate is separate: net Q has already deducted it. Using
        100% utilization is a lower bound on the required active-capital yield,
        not a promise that a partially invested portfolio earns the target.
        """
        if not 0 < calendar_capital_utilization <= 1:
            raise ValueError('utilization must be in (0, 1]')
        return self.annual_net_return * 10_000 / 365 / calendar_capital_utilization

    def summary(self):
        return dict(**asdict(self), annual_net_twd=self.annual_net_twd,
                    daily_net_twd=self.daily_net_twd,
                    daily_net_bp_on_committed_capital=self.daily_net_twd / self.capital_twd * 10_000,
                    full_utilization_net_bp_per_calendar_day=self.capital_day_hurdle_bp(),
                    funding_bp_per_calendar_day=self.funding_annual_rate * 10_000 / 365,
                    completed_trade_scenarios=[dict(trades_per_day=n, required_net_twd_per_trade=self.per_trade_twd(n))
                                               for n in (10, 20, 30, 40)],
                    interpretation='Simple annual net return on the full allocated capital; 250 sessions is a planning assumption. '
                    'Not compounded CAGR; realized full-calendar annualization will also be reported.')
