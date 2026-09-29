"""Q admission at the current event timestamp; never filter execution reports."""
from dataclasses import asdict, dataclass

from ..ev_lookup_cost.decide import Decision


@dataclass(frozen=True)
class QDecision(Decision):
    q_net_bp: float = 0.
    q_before_funding_bp: float = 0.
    q_days: float = 0.
    q_funding_bp: float = 0.
    q_hurdle_bp: float = 0.
    q_required_bp: float = 0.
    p_rollback: float = 0.
    q_entry_samples: int = 0
    q_prior_sessions: int = 0
    q_entry_key: str = ''
    q_snapshot_day: str = ''


class QEntryDecider:
    def __init__(self, base, q_snapshot, *, policy, calendar):
        if policy not in {'net','target'}:
            raise ValueError('unknown Q policy')
        if base.snapshot.day != q_snapshot.day:
            raise ValueError('Q snapshot and decision day differ')
        self.base,self.q,self.policy,self.calendar=base,q_snapshot,policy,calendar
        self.snapshot=base.snapshot

    def decide(self, **kwargs):
        # Reuse strict timestamp, contract, quote and capacity input validation.
        base=self.base.decide(**kwargs)
        if self.q.observed_prior_sessions<20:
            values=asdict(base)
            values.update(admit=False,reason='basis' if kwargs['ab']<=0 else 'q_warmup',est_bp=0.)
            return QDecision(**values,q_prior_sessions=self.q.observed_prior_sessions,
                             q_entry_key='warmup_not_valued',q_snapshot_day=self.q.day)
        q=self.q.estimate(stream=kwargs['stream'],quote_second=kwargs['quote_second'],ab=kwargs['ab'],
            eff_u=kwargs['eff_u'],expiry=kwargs['expiry'],spread_bp=kwargs.get('spread_bp',0.),
            quote_notional_twd=kwargs['reservation_cents']/100,
            execution_cost_bp=kwargs.get('execution_cost_bp',0.),calendar=self.calendar)
        required=q.capital_hurdle_bp if self.policy=='target' else 0.
        reason='ok'
        if kwargs['ab']<=0:
            reason='basis'
        elif q.observed_prior_sessions<20 or q.entry_samples<50:
            reason='q_warmup'
        elif q.net_bp<required:
            reason='q_value'
        elif kwargs.get('capacity_required',True) and not base.slot_free:
            reason='cap'
        values=asdict(base)
        values.update(admit=reason=='ok',reason=reason,est_bp=q.net_bp,p_sd=q.p_sd,
            p_nx=q.p_overnight/(1-q.p_sd) if q.p_sd<1 else 0.,p_overnight=q.p_overnight,
            p_expiry=q.p_expiry,p_other=q.p_other,d_in=q.d_in,d_sd=q.d_sd,d_on=q.d_on,
            hazard_samples=q.min_carry_samples)
        return QDecision(**values,q_net_bp=q.net_bp,q_before_funding_bp=q.before_funding_bp,
            q_days=q.expected_calendar_days,q_funding_bp=q.funding_bp,q_hurdle_bp=q.capital_hurdle_bp,
            q_required_bp=required,p_rollback=q.p_rollback,q_entry_samples=q.entry_samples,
            q_prior_sessions=q.observed_prior_sessions,q_entry_key=q.entry_key,q_snapshot_day=self.q.day)
