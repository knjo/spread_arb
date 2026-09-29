"""Conditional competing-risk Q: net execution value and capital holding time.

Bins and fallbacks are fixed before this study's performance comparison. Open
carry contributes exposure, never an invented zero-PnL completed trade. A Q
estimate is conditional on an entry maker fill; actual fill opportunity and
capacity utilization must be measured by chronological execution replay.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from math import isfinite
import json

from ..ev_lookup_cost.causal_lookup import CLOSE_SECOND, SECOND, open_ns, spread_bucket, time_bucket
from ..ev_lookup_cost.exit_model import expiry_bucket, remaining_sessions
from .q_facts import DAY_NS, day_distance, training_window
from .target import ReturnTarget


EVENTS = ('normal', 'rollback', 'other', 'survive')

# Immutable date arithmetic is shared by millions of event-time queries.
open_ns = lru_cache(maxsize=512)(open_ns)
day_distance = lru_cache(maxsize=8192)(day_distance)


@lru_cache(maxsize=1024)
def forecast_sessions(day, expiry, calendar_items):
    return tuple(remaining_sessions(day, expiry, dict(calendar_items)))


def premium_bucket(value):
    return sum(value >= edge for edge in (25., 50., 100.))


def ticket_bucket(value):
    return sum(value >= edge for edge in (250_000., 1_000_000.))


def age_bucket(value):
    return sum(value > edge for edge in (2, 5, 10))


def keys(row, phase, conditional=True):
    stream = row['stream']
    prefix = (phase, stream)
    result = [(phase, '*'), prefix]
    if conditional:
        prefix += (premium_bucket(row['quote_ab'] - row['anchor']),)
        result.append(prefix)
        if phase == 'carry':
            prefix += (expiry_bucket(row['days_to_expiry']),)
            result.append(prefix)
            prefix += (age_bucket(row['age_days']),)
        else:
            prefix += (time_bucket(row['quote_second']),)
            result.append(prefix)
            prefix += (spread_bucket(row['quote_spread_bp']),)
        result.append(prefix)
        result.append(prefix + (ticket_bucket(row['quote_notional_twd']),))
    return ['|'.join(map(str, key)) for key in result]


def empty_stats():
    return dict(n=0, normal=0, rollback=0, other=0, survive=0,
                normal_seconds=0., rollback_seconds=0., other_seconds=0.,
                normal_days=0., rollback_days=0., other_days=0.,
                rollback_net=0., other_net=0., entry_wait_days=0.)


def fit_risks(rows, conditional=True):
    result = {}
    for row in rows.iter_rows(named=True):
        if row['phase'] not in {'entry', 'carry'} or row['event'] not in EVENTS:
            continue
        for key in keys(row, row['phase'], conditional):
            stats = result.setdefault(key, empty_stats())
            stats['n'] += 1
            event = row['event']
            stats[event] += 1
            if row['phase'] == 'entry':
                stats['entry_wait_days'] += max(0, row['risk_start_ns']-row['quote_ns']) / DAY_NS
            if event != 'survive':
                stats[event+'_seconds'] += row['close_second']
                stats[event+'_days'] += row['elapsed_calendar_days']
                if event == 'rollback':
                    # A partial stock fill is scaled to the originally quoted
                    # full ticket, not treated as a successful full pair.
                    stats['rollback_net'] += row['outcome_net_bp_on_quote_capital']
                elif event == 'other':
                    stats['other_net'] += row['outcome_net_bp']
    return result


def fit_costs(rows, conditional=True):
    result = {}
    for row in rows.iter_rows(named=True):
        phase = 'entry_cost' if row['kind'] == 'entry' else ('exit_carry' if row['carry'] else 'exit_sd')
        for key in keys(row, phase, conditional):
            stats = result.setdefault(key, dict(n=0, total=0., square=0.))
            stats['n'] += 1
            stats['total'] += row['bp']
            stats['square'] += row['bp']**2
    return result


def settlement_day(expiry, calendar):
    """Next predicted session after expiry; never read future realized closures."""
    day = datetime.strptime(expiry, '%Y%m%d')
    for _ in range(10):
        day += timedelta(days=1)
        text = day.strftime('%Y%m%d')
        if calendar.get(text, day.weekday() < 5):
            return text
    raise ValueError('forecast calendar has no post-expiry session')


@dataclass(frozen=True)
class QEstimate:
    net_bp: float
    before_funding_bp: float
    expected_calendar_days: float
    expected_fill_wait_days: float
    funding_bp: float
    capital_hurdle_bp: float
    surplus_bp: float
    p_sd: float
    p_overnight: float
    p_expiry: float
    p_rollback: float
    p_other: float
    d_in: float
    d_sd: float
    d_on: float
    entry_samples: int
    min_carry_samples: int
    train_sessions: int
    entry_key: str
    observed_prior_sessions: int


@dataclass
class QSnapshot:
    day: str
    train_days: list[str]
    risk: dict
    costs: dict
    conditional: bool = True
    min_risk_samples: int = 50
    min_cost_samples: int = 30
    margin_bp: float = 3.0
    observed_prior_sessions: int = 0

    @classmethod
    def fit(cls, root, day, *, conditional=True, window=20):
        risk = training_window(root, day, 'risk', window)
        costs = training_window(root, day, 'costs', window)
        train_days = sorted(set(risk['day'].to_list()) | set(costs['day'].to_list()))
        manifest = json.loads((root/'manifest.json').read_text())
        observed = sum(d < day and d not in manifest.get('outages',[]) for d in manifest['days'])
        return cls(day, train_days, fit_risks(risk, conditional), fit_costs(costs, conditional), conditional,
                   observed_prior_sessions=observed)

    def to_dict(self):
        return asdict(self)

    def hazard(self, row, phase):
        candidates = keys(row, phase, self.conditional)
        cache = self.__dict__.setdefault('_hazard_cache',{})
        cache_key = (tuple(candidates),self.min_risk_samples)
        if cache_key in cache:
            return cache[cache_key]
        chosen, label = None, 'cold_prior'
        for key in reversed(candidates):
            value = self.risk.get(key)
            if value and value['n'] >= self.min_risk_samples:
                chosen, label = value, key
                break
        if chosen is None:
            # Research policies require 20 observed sessions before entry.
            # This fixed prior keeps cold diagnostic forecasts explicit.
            probability = dict(normal=.5, rollback=0., other=0., survive=.5)
            n = 0
        else:
            n = chosen['n']
            probability = {event: chosen[event]/n for event in EVENTS}
        details = {}
        for event in EVENTS[:-1]:
            event_stats = next((self.risk[k] for k in reversed(candidates)
                                if k in self.risk and self.risk[k][event] >= 10), None)
            if event_stats is None:
                details[event] = dict(seconds=CLOSE_SECOND/2, days=.1,
                                      net=-34. if event in {'rollback', 'other'} else 0.)
            else:
                count = event_stats[event]
                details[event] = dict(seconds=event_stats[event+'_seconds']/count,
                    days=event_stats[event+'_days']/count,
                    net=event_stats.get(event+'_net', 0.)/count)
        result = (probability, details, n, label)
        cache[cache_key] = result
        return result

    def decay(self, row, phase):
        candidates = keys(row,phase,self.conditional)
        cache = self.__dict__.setdefault('_cost_cache',{})
        cache_key = (tuple(candidates),self.min_cost_samples,self.margin_bp)
        if cache_key in cache:
            return cache[cache_key]
        for key in reversed(candidates):
            value = self.costs.get(key)
            if value and value['n'] >= self.min_cost_samples:
                result = max(0., value['total']/value['n']) + self.margin_bp
                cache[cache_key] = result
                return result
        result = (15. if phase == 'entry_cost' else 5.) + self.margin_bp
        cache[cache_key] = result
        return result

    def _template(self, row, expiry, calendar):
        signature=(tuple(keys(row,'entry',self.conditional)),expiry,tuple(calendar.items()),
                   self.min_risk_samples,self.min_cost_samples,self.margin_bp)
        cache=self.__dict__.setdefault('_template_cache',{})
        if signature in cache:
            return cache[signature]
        row=dict(row)
        d_in = self.decay(row, 'entry_cost')
        d_sd, d_on = self.decay(row, 'exit_sd'), self.decay(row, 'exit_carry')
        prob, detail, entry_n, entry_key = self.hazard(row, 'entry')
        p_sd, p_on, p_roll, p_other = prob['normal'], 0., prob['rollback'], prob['other']
        other_net=sum(prob[k]*detail[k]['net'] for k in ('rollback','other'))
        today_durations=tuple((prob[k],detail[k]['days']) for k in EVENTS[:-1])
        entry_stats = self.risk.get(entry_key)
        fill_wait=entry_stats['entry_wait_days']/entry_stats['n'] if entry_stats else 0.
        survival, carry_counts = prob['survive'], []
        future_mass=survival
        future_duration=0.
        for day in forecast_sessions(self.day, expiry, tuple(calendar.items())):
            row.update(days_to_expiry=day_distance(day, expiry), age_days=day_distance(self.day, day))
            prob, detail, n, _ = self.hazard(row, 'carry')
            carry_counts.append(n)
            normal = survival*prob['normal']
            p_on += normal
            for event in EVENTS[:-1]:
                mass = survival*prob[event]
                future_duration += mass*(day_distance(self.day,day)+detail[event]['seconds']/86_400)
                if event == 'rollback':
                    p_roll += mass
                    other_net += mass*detail[event]['net']
                elif event == 'other':
                    p_other += mass
                    other_net += mass*detail[event]['net']
            survival *= prob['survive']
        p_exp = survival
        future_duration += p_exp*day_distance(self.day,settlement_day(expiry,calendar))
        if abs(p_sd+p_on+p_roll+p_other+p_exp-1.) > 1e-8:
            raise AssertionError('competing outcome probabilities do not sum to one')
        result=dict(p_sd=p_sd,p_on=p_on,p_exp=p_exp,p_roll=p_roll,p_other=p_other,d_in=d_in,d_sd=d_sd,d_on=d_on,
                    other_net=other_net,today_durations=today_durations,fill_wait=fill_wait,
                    future_mass=future_mass,future_duration=future_duration,entry_n=entry_n,entry_key=entry_key,
                    carry_n=min(carry_counts) if carry_counts else 0)
        cache[signature]=result
        return result

    def estimate(self, *, stream, quote_second, ab, eff_u, expiry, spread_bp,
                 quote_notional_twd, execution_cost_bp=0., calendar=None,
                 target=None, utilization=1.0):
        if stream not in {'S1', 'S2'} or not 0 <= quote_second < CLOSE_SECOND:
            raise ValueError('invalid quote')
        if expiry < self.day or not all(isfinite(v) for v in (ab, eff_u, spread_bp, quote_notional_twd, execution_cost_bp)):
            raise ValueError('nonfinite quote or expired contract')
        if quote_notional_twd <= 0 or execution_cost_bp < 0:
            raise ValueError('invalid nominal or execution floor')
        target, calendar = target or ReturnTarget(), calendar or {}
        row=dict(stream=stream,quote_second=quote_second,quote_ab=ab,anchor=ab-eff_u,
                 quote_spread_bp=spread_bp,quote_notional_twd=quote_notional_twd,
                 days_to_expiry=day_distance(self.day,expiry),age_days=0)
        t=self._template(row,expiry,calendar)
        d_in=max(t['d_in'],execution_cost_bp)
        before_funding=(t['p_sd']*(eff_u+5-d_in-t['d_sd']-20)
                        +t['p_on']*(eff_u+5-d_in-t['d_on']-34)
                        +t['p_exp']*(ab-d_in-34)+t['other_net'])
        remaining_today=(CLOSE_SECOND-quote_second)/86_400
        fill_wait=min(remaining_today,t['fill_wait'])
        # Every future event is at least one calendar day away, whereas quote
        # time plus same-session fill wait is below one day. Thus these terms
        # remain positive and their time shift can be combined algebraically.
        duration=(sum(mass*min(remaining_today,days) for mass,days in t['today_durations'])
                  +t['future_duration']-t['future_mass']*(quote_second/86_400+fill_wait))
        funding = duration*target.funding_annual_rate*10_000/365
        hurdle = duration*target.capital_day_hurdle_bp(utilization)
        net = before_funding-funding
        return QEstimate(net, before_funding, duration, fill_wait, funding, hurdle, net-hurdle,
                         t['p_sd'],t['p_on'],t['p_exp'],t['p_roll'],t['p_other'],d_in,t['d_sd'],t['d_on'],
                         t['entry_n'],t['carry_n'],len(self.train_days),t['entry_key'],
                         self.observed_prior_sessions)
