"""30% hurdle on time that blocks this strategy's entry window; calendar funding stays.

The replay permits new entries before second 14000. Closed-market time still
costs funding, but it does not displace another entry in this strategy. This
clock is a full-utilization opportunity-cost approximation, not a return guarantee.
"""
from dataclasses import asdict, dataclass

from ..decide import QDecision
from ..q_model import CLOSE_SECOND, EVENTS, day_distance, forecast_sessions, keys
from .deep_policy import DeepSpotPortfolio


ENTRY_SECONDS=14_000
DAILY_HURDLE_BP=3_000/250


def sessions(snapshot,*,stream,quote_second,ab,eff_u,expiry,spread_bp,quote_notional_twd,calendar):
    if not 0<=quote_second<CLOSE_SECOND:
        raise ValueError('quote outside replay clock')
    row=dict(stream=stream,quote_second=quote_second,quote_ab=ab,anchor=ab-eff_u,
        quote_spread_bp=spread_bp,quote_notional_twd=quote_notional_twd,
        days_to_expiry=day_distance(snapshot.day,expiry),age_days=0)
    template=snapshot._template(row,expiry,calendar)
    signature=(tuple(keys(row,'entry',snapshot.conditional)),expiry,tuple(calendar.items()),snapshot.min_risk_samples)
    cache=snapshot.__dict__.setdefault('_entry_window_cache',{})
    if signature not in cache:
        survival=template['future_mass']
        seconds=0.
        for day in forecast_sessions(snapshot.day,expiry,tuple(calendar.items())):
            row.update(days_to_expiry=day_distance(day,expiry),age_days=day_distance(snapshot.day,day))
            probability,detail,_,_=snapshot.hazard(row,'carry')
            seconds+=survival*(probability['survive']*ENTRY_SECONDS+
                sum(probability[k]*min(ENTRY_SECONDS,max(0.,detail[k]['seconds'])) for k in EVENTS[:-1]))
            survival*=probability['survive']
        cache[signature]=seconds
    wait=min((CLOSE_SECOND-quote_second)/86400,template['fill_wait'])*86400
    remaining=max(0.,ENTRY_SECONDS-quote_second-wait)
    seconds=(sum(mass*min(remaining,max(0.,days*86400)) for mass,days in template['today_durations'])+
             template['future_mass']*remaining+cache[signature])
    return seconds/ENTRY_SECONDS


@dataclass(frozen=True)
class WindowDecision(QDecision):
    q_entry_window_sessions: float=0.
    q_entry_window_hurdle_bp: float=0.


class WindowDecider:
    def __init__(self,delegate):
        self.delegate=delegate

    def __getattr__(self,key):
        return getattr(self.delegate,key)

    def decide(self,**kwargs):
        result=self.delegate.decide(**kwargs)
        values=asdict(result)
        if result.q_prior_sessions<20:
            return WindowDecision(**values)
        duration=sessions(self.delegate.q,stream=kwargs['stream'],quote_second=kwargs['quote_second'],
            ab=kwargs['ab'],eff_u=kwargs['eff_u'],expiry=kwargs['expiry'],spread_bp=kwargs.get('spread_bp',0.),
            quote_notional_twd=kwargs['reservation_cents']/100,calendar=self.delegate.calendar)
        required=duration*DAILY_HURDLE_BP
        reason=('basis' if kwargs['ab']<=0 else 'q_warmup' if result.q_entry_samples<50 else
                'q_value' if result.q_net_bp<required else 'cap'
                if kwargs.get('capacity_required',True) and not result.slot_free else 'ok')
        values.update(admit=reason=='ok',reason=reason,q_required_bp=required)
        return WindowDecision(**values,q_entry_window_sessions=duration,q_entry_window_hurdle_bp=required)


class WindowPortfolio(DeepSpotPortfolio):
    def __init__(self,*args,target_clock=None,**kwargs):
        self.target_clock=target_clock
        if target_clock not in {None,'entry_window'}:
            raise ValueError('unknown target opportunity clock')
        super().__init__(*args,**kwargs)

    def begin(self,*args,**kwargs):
        super().begin(*args,**kwargs)
        if self.target_clock=='entry_window':
            if self.q_policy!='net':
                raise ValueError('entry-window adapter requires the net-Q base')
            self.decider=WindowDecider(self.decider)
