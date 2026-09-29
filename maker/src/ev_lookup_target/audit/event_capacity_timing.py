"""Measure when real capacity releases, including the user's 10:00 boundary."""
import polars as pl

from ...ev_lookup_cost.causal_lookup import SECOND, open_ns
from .capital import intervals


def day_row(day,ledger,positions,opening):
    start=open_ns(day);cut=start+3600*SECOND;end=start+15600*SECOND
    events=[(r['ns'],r['delta_cents']/100) for r in ledger]
    before=intervals([e for e in events if e[0]<cut],start,cut,opening)
    after=intervals([e for e in events if e[0]>=cut],cut,end,before['ending_twd'])
    at10=before['ending_twd']+sum(delta for ns,delta in events if ns==cut)
    early_market=early_carry=expiry=0.
    for r in ledger:
        p=positions[r['id']]
        if r['kind']!='release' or p['state']!='closed':
            continue
        cash=-r['delta_cents']/100
        if p['close_kind']=='expiry_basis_zero_accounting':
            expiry+=cash
        elif r['ns']<cut:
            early_market+=cash
            if p['entry_day']<day:
                early_carry+=cash
    return dict(day=day,opening_committed_twd=opening,committed_at10_twd=at10,
        closing_committed_twd=after['ending_twd'],peak_before10_twd=before['peak_twd'],
        peak_from10_twd=after['peak_twd'],mean_before10_twd=before['area_twd_seconds']/3600,
        mean_from10_twd=after['area_twd_seconds']/12000,
        over25_seconds_before10=before['over25_seconds'],over25_seconds_from10=after['over25_seconds'],
        actual_market_release_before10_twd=early_market,actual_carry_market_release_before10_twd=early_carry,
        expiry_accounting_release_twd=expiry)


def write(root,name,manifest,folder,first_day):
    rows=[];opening=0.
    for day in manifest['days']:
        path=root/f'Date={day}'/name
        assert path.is_dir(), 'portfolio session folder is missing'
        ledger=pl.read_parquet(path/'ledger.parquet').to_dicts() if (path/'ledger.parquet').exists() else []
        position_path=path/'positions.parquet'
        if not position_path.exists():
            assert not ledger and abs(opening)<1e-6, 'capacity exists but its positions file is missing'
        positions=({p['id']:p for p in pl.read_parquet(position_path).iter_rows(named=True)}
                   if position_path.exists() else {})
        row=day_row(day,ledger,positions,opening)
        rows.append(row);opening=row['closing_committed_twd']
    frame=pl.from_dicts(rows)
    frame.write_csv(folder/'capacity_timing_daily.csv')
    post=frame.filter((pl.col('day')>=first_day)&pl.col('day').is_in(manifest['available_days']))
    return dict(portfolio=name,observed_sessions=post.height,
        mean_opening_committed_twd=post['opening_committed_twd'].mean(),
        mean_committed_at10_twd=post['committed_at10_twd'].mean(),
        mean_closing_committed_twd=post['closing_committed_twd'].mean(),
        mean_actual_market_release_before10_twd=post['actual_market_release_before10_twd'].mean(),
        mean_actual_carry_market_release_before10_twd=post['actual_carry_market_release_before10_twd'].mean(),
        fraction_observed_closes_over20=post.select((pl.col('closing_committed_twd')>20e6+1e-6).mean()).item(),
        fraction_observed_closes_over25=post.select((pl.col('closing_committed_twd')>25e6+1e-6).mean()).item(),
        over25_seconds_before10=post['over25_seconds_before10'].sum(),
        over25_seconds_from10=post['over25_seconds_from10'].sum())
