"""Realized costs in the same basis-point units used by four-leg cash PnL."""
from .causal_lookup import DecayObservation, spread_bucket, time_bucket


def observe_costs(history, positions, day: str, end_ns: int) -> list[DecayObservation]:
    rows = []
    for p in positions:
        if p.continuity_blocked or p.stream not in {"S1", "S2"}:
            continue
        if (p.actual_ab is not None and p.hedged_ns is not None and p.hedged_ns <= end_ns
                and f"entry:{p.id}" not in history._decay_ids):
            rows.append(DecayObservation(p.id, p.stream, "entry", time_bucket(p.quote_second),
                spread_bucket(p.quote_spread_bp), p.quote_ab-p.actual_ab, day, end_ns))
        if p.close_day == day and p.close_kind == "maker_exit" and p.spot_buy_cash > 0:
            cost = (p.future_buy_cash-p.spot_sell_cash)/p.spot_buy_cash*10_000-(p.anchor-5.0)
            rows.append(DecayObservation(p.id, p.stream, "exit", int(p.entry_day != day),
                                         -1, cost, day, end_ns))
    for row in rows:
        history.observe_decay(row, end_ns)
    return rows
