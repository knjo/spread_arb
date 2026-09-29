"""Summarize actual S2 entry paths without inventing an exit or portfolio profit."""
import argparse
import json
from pathlib import Path

import polars as pl


def run(root):
    protocol=json.loads((root/'protocol.json').read_text())
    assert json.loads((root/'verification.json').read_text())['passed']
    frames=[]
    for day in protocol['days']:
        assert (root/f'Date={day}/summary.csv').exists()
        for config in protocol['configs']:
            p=root/f'Date={day}'/config['name']/'fills.parquet'
            if p.exists():
                frames.append(pl.read_parquet(p))
    fills=pl.concat(frames,how='diagonal_relaxed').with_columns(
        (pl.col('quote_ab')-pl.col('actual_ab')).alias('entry_decay_bp'),
        ((pl.col('fill_ns')-pl.col('quote_ns'))/1e6).alias('quote_to_fill_ms'),
        ((pl.col('hedged_ns')-pl.col('fill_ns'))/1e6).alias('hedge_delay_ms'))
    n=len(protocol['days'])
    summaries=[]
    for config in protocol['configs']:
        f=fills.filter(pl.col('policy')==config['name']);paired=f.filter(pl.col('actual_ab').is_not_null())
        fast=paired.filter(pl.col('quote_to_fill_ms')<50)
        row=dict(config,days=n,fills=f.height,fills_per_day=f.height/n,paired=paired.height,
            pending_hedges=f.height-paired.height,race_fills=f['cancellation_race'].sum(),
            actual_premium_mean_bp=paired['actual_premium_bp'].mean(),
            actual_premium_median_bp=paired['actual_premium_bp'].median(),
            actual_premium_ge25=int((paired['actual_premium_bp']>=25).sum()),
            actual_premium_ge25_per_day=float((paired['actual_premium_bp']>=25).sum()/n),
            mean_entry_decay_bp=paired['entry_decay_bp'].mean(),
            p90_entry_decay_bp=paired['entry_decay_bp'].quantile(.9),
            adverse_entry_fraction=float((paired['entry_decay_bp']>1e-7).sum()/max(1,paired.height)),
            negative_actual_basis=int((paired['actual_ab']<0).sum()),
            stock_cash_per_day_twd=f['stock_cash_twd'].sum()/n,
            quote_to_fill_under50ms=fast.height,
            quote_to_fill_under50ms_fraction=fast.height/max(1,paired.height),
            hedge_delay_p99_ms=paired['hedge_delay_ms'].quantile(.99),
            hedge_delay_max_ms=paired['hedge_delay_ms'].max())
        summaries.append(row)
    order={c['name']:i for i,c in enumerate(protocol['configs'])}
    daily=pl.concat([pl.read_csv(root/f'Date={d}/summary.csv',schema_overrides={'day':pl.String})
                     for d in protocol['days']],how='diagonal_relaxed')
    output=root/'report';output.mkdir(exist_ok=True)
    daily.write_csv(output/'daily.csv');pl.from_dicts(summaries).write_csv(output/'policies.csv')
    fills.write_parquet(output/'fills.parquet')
    fills.group_by('policy','vc').agg(pl.len().alias('fills'),pl.col('actual_premium_bp').mean(),
        pl.col('stock_cash_twd').sum()).sort(['policy','fills'],descending=[False,True]).write_csv(output/'products.csv')
    result=dict(status='completed',dates=protocol['days'],summaries=summaries,
        interpretation='Actual premium = paired futures/stock basis minus quote-time120s anchor. '
        'It is not net return: exits,20/34bp transaction cost and calendar funding are not deducted. '
        'No capital limit or carry exits are modeled in this supply decomposition. '
        'Premium>=25 after hedge is an ex-post diagnostic; it never selects or deletes trades. '
        'Immediate queue admission and sub50ms fill exposure are explicit limitations. '
        'Variants have independent counterfactual depth; their fills cannot be added together.')
    (output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['1s / 60s / depth5 / 25bp','Event / 60s / depth5 / 25bp',
        'Event / hedge done / depth5 / 25bp','Event / hedge done / depth1 / 25bp',
        'Event / hedge done / depth5 / 0bp','Event / hedge done / depth1 / 0bp']
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    x=list(range(len(labels)))
    axes[0].barh(x,[r['fills_per_day'] for r in summaries],color='#2563a6',label='All retained fills')
    axes[0].barh(x,[r['actual_premium_ge25_per_day'] for r in summaries],color='#15966c',label='Post-hedge premium >=25bp')
    axes[0].set_yticks(x,labels);axes[0].invert_yaxis();axes[0].set_xlabel('Entry fills per sampled session')
    axes[0].legend(fontsize=8)
    axes[1].barh(x,[r['actual_premium_mean_bp'] for r in summaries],color='#2563a6')
    axes[1].set_yticks(x,labels=[]);axes[1].invert_yaxis();axes[1].set_xlabel('Mean premium after actual stock hedge (bp)')
    fig.suptitle('S2 entry supply: timing, cooldown, depth and residual gate')
    fig.text(.5,-.045,'Four fixed sampled sessions. Unlimited-capacity entry paths; no exit or net-profit claim.\n'
             'Green bars are outcome diagnostics, never an admission filter.',ha='center',fontsize=9)
    fig.savefig(output/'s2_supply.png',dpi=180,bbox_inches='tight');plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    run(p.parse_args().root)
