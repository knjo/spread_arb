"""Past-only remaining-session release estimates and a bounded admission credit.

This changes permission to enter, never the actual cash or capacity ledger.
The lower-bound discount is a research heuristic: repeated positions and
correlated market days do not constitute independent Bernoulli samples.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, sqrt

import polars as pl

from ..ev_lookup_cost.causal_lookup import CLOSE_SECOND, SECOND, open_ns
from ..ev_lookup_cost.exit_model import expiry_bucket
from .q_facts import day_distance
from .q_model import premium_bucket, settlement_day


BUCKET_SECONDS = 300
SCHEMA = dict(day=pl.String, available_ns=pl.Int64, id=pl.String, stream=pl.String,
              carry=pl.Boolean, start_second=pl.Float64, end_second=pl.Float64, closed=pl.Boolean,
              days_to_expiry=pl.Int64, premium=pl.Float64, nominal_twd=pl.Float64)


def daily_exposures(day, positions, *, outage=False):
    if outage:
        return pl.DataFrame(schema=SCHEMA)
    start, end = open_ns(day), open_ns(day)+CLOSE_SECOND*SECOND
    result = []
    for p in positions.iter_rows(named=True):
        paired = p['hedged_ns']
        if paired is None or not p['future_sell_qty']:
            continue
        if paired > end or (p['close_ns'] is not None and p['close_ns'] > end):
            raise AssertionError('future pairing/closure in release exposure')
        left = max(start, paired)
        right = p['close_ns'] if p['close_ns'] is not None else end
        if left >= right:
            continue
        result.append(dict(day=day, available_ns=end, id=p['id'], stream=p['stream'],
            carry=p['entry_day'] < day, start_second=(left-start)/SECOND, end_second=(right-start)/SECOND,
            closed=p['close_ns'] is not None, days_to_expiry=day_distance(day,p['contract']['expiry']),
            premium=p['quote_ab']-p['anchor'], nominal_twd=p['spot_buy_cash']/10_000))
    return pl.from_dicts(result,schema=SCHEMA)


def release_keys(stream, carry, bucket, days_to_expiry, premium):
    prefix = f'{int(carry)}|{bucket}'
    a = f'{prefix}|{stream}'
    b = f'{a}|{expiry_bucket(days_to_expiry)}'
    return [prefix,a,b,f'{b}|{premium_bucket(premium)}']


@dataclass
class ReleaseSnapshot:
    day: str
    train_days: list[str]
    cells: dict[str, tuple[int,int]]
    min_samples: int = 50
    discount_z: float = 1.64

    @classmethod
    def fit(cls, day, rows):
        if rows.filter((pl.col('day') >= day) | (pl.col('available_ns') >= open_ns(day))).height:
            raise AssertionError('release forecast used unavailable outcomes')
        cells = {}
        for row in rows.iter_rows(named=True):
            first = max(0,ceil(row['start_second']/BUCKET_SECONDS))
            last = min((CLOSE_SECOND-1)//BUCKET_SECONDS,ceil(row['end_second']/BUCKET_SECONDS)-1)
            for bucket in range(first,last+1):
                for key in release_keys(row['stream'],row['carry'],bucket,row['days_to_expiry'],row['premium']):
                    value = cells.setdefault(key,[0,0])
                    value[0] += 1
                    value[1] += int(row['closed'])
        return cls(day,sorted(set(rows['day'].to_list())),{k:tuple(v) for k,v in cells.items()})

    def probability(self, *, stream, carry, second, days_to_expiry, premium):
        bucket = max(0,min((CLOSE_SECOND-1)//BUCKET_SECONDS,int(second)//BUCKET_SECONDS))
        for key in reversed(release_keys(stream,carry,bucket,days_to_expiry,premium)):
            n, closed = self.cells.get(key,(0,0))
            if n < self.min_samples:
                continue
            p,z = closed/n,self.discount_z
            lower = (p+z*z/(2*n)-z*sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)
            return dict(mean=p,discounted=max(0.,lower),samples=n,key=key)
        return dict(mean=0.,discounted=0.,samples=0,key='insufficient_history')

    def capacity(self, positions, *, second, calendar, base_twd=20_000_000., max_extra_twd=5_000_000.,
                 today_weight=.5, next_weight=.25):
        if not 0 <= today_weight <= 1 or not 0 <= next_weight <= 1 or base_twd <= 0 or max_extra_twd < 0:
            raise ValueError('invalid capacity policy')
        tomorrow = settlement_day(self.day,calendar)
        today_total = next_total = 0.
        evidence = []
        for p in positions:
            if p['nominal_twd'] <= 0:
                continue
            today = self.probability(stream=p['stream'],carry=p['entry_day']<self.day,second=second,
                                     days_to_expiry=day_distance(self.day,p['expiry']),premium=p['premium'])
            nxt = self.probability(stream=p['stream'],carry=True,second=0,
                                   days_to_expiry=day_distance(tomorrow,p['expiry']),premium=p['premium'])
            # The tomorrow component applies only to mass not released today.
            # Expiry accounting must not be forecast as a normal market release.
            if tomorrow > p['expiry']:
                nxt = dict(mean=0.,discounted=0.,samples=0,key='after_expiry')
            t = p['nominal_twd']*today['discounted']
            n = p['nominal_twd']*(1-today['mean'])*nxt['discounted']
            today_total += t
            next_total += n
            evidence.append(dict(id=p['id'],nominal_twd=p['nominal_twd'],today=t,next_session=n,
                                 today_key=today['key'],next_key=nxt['key'],
                                 today_samples=today['samples'],next_samples=nxt['samples']))
        extra = min(max_extra_twd,today_weight*today_total+next_weight*next_total)
        return dict(admission_twd=base_twd+floor(extra*100)/100,extra_twd=extra,
                    forecast_today_twd=today_total,forecast_next_session_twd=next_total,
                    next_session=tomorrow,positions=evidence)
