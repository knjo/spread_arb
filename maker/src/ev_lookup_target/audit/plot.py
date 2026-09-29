"""Standalone research figure with the allocated-capital target and actual usage."""
from datetime import datetime
import json
import os
import textwrap

import polars as pl


def draw(root,output):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/hft-matplotlib-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker
    import numpy as np

    manifest=json.loads((root/'manifest.json').read_text())
    colors=['#0072B2','#009E73','#D55E00','#CC79A7']
    fig,axes=plt.subplots(3,1,figsize=(11,10),sharex=True,layout='constrained')
    for i,config in enumerate(manifest['configurations']):
        name=config['name']
        equity=pl.read_csv(output/name/'equity_daily.csv',schema_overrides={'day':pl.String})
        capacity=pl.read_csv(output/name/'capital_daily.csv',schema_overrides={'day':pl.String})
        dates=[datetime.strptime(d,'%Y%m%d') for d in equity['day']]
        values=[value if value is not None else np.nan for value in equity['net_equity_twd']]
        if config.get('q_policy'):
            label='30% holding hurdle' if config['q_policy']=='target' else 'Net Q'
            if config.get('target_clock')=='entry_window':
                label='30% entry-window hurdle'
            label+=', release credit' if config.get('release_credit') else ', reserved 20M' if config.get('reserve_quotes',True) else ', shared 20M'
            if config.get('max_ticket_twd') is None:
                label+=', no ticket filter'
            if config.get('deep_shared'):
                label+=', keep deep S1 queue'
        else:
            label=name
        label=textwrap.fill(label,width=74,break_long_words=False,break_on_hyphens=False)
        color=colors[i]
        axes[0].plot(dates,values,color=color,label=label,lw=1.4)
        axes[1].plot(dates,capacity['mean_stock_cash_twd']/1e6,color=color,label=label,lw=1.2)
        axes[2].plot(dates,capacity['committed_peak_twd']/1e6,color=color,label=label,lw=1.2)
    daily_target=[]
    total=0.
    for day in manifest['days']:
        total+=24_000 if day in manifest['available_days'] else 0.
        daily_target.append(total)
    axes[0].plot(dates,daily_target,ls='--',color='#555555',lw=1,label='30% goal: TWD 24,000 / observed session')
    axes[0].set(title='Net equity including final inventory marks and 2% actual-cash funding',ylabel='TWD')
    axes[0].yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    axes[1].set(title='Mean actual stock cash invested during each session',ylabel='TWD million')
    axes[2].set(title='Actual peak committed capacity, including cancellation races',ylabel='TWD million')
    for ax in axes[1:]:
        ax.axhline(20,color='#555555',ls='--',lw=.8)
    axes[2].axhline(25,color='#777777',ls=':',lw=.8)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8,loc='upper left')
        if len(manifest['days'])<15:
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        else:
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    short=len(manifest['days'])<15
    fig.suptitle(('Short replay check' if short else 'Causal Q and chronological shared-capacity replay')+
                 ' | S1 + S2 | allocated capital 20M')
    note=(f"Short interval {manifest['days'][0]} - {manifest['days'][-1]}; prior training was already available"
          if short else 'Exploratory same-period research; first 20 observed sessions warm up flat; missing daily marks remain gaps')
    fig.supxlabel(note,fontsize=9)
    fig.savefig(output/'target_equity_capacity.png',dpi=160)
    plt.close(fig)
