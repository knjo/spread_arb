"""Separate stale grid pricing from post-print hedge delay on fixed old S2 fills."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl

from ..causal_lookup import open_ns, SECOND
from ..execution import TakerDepth
from ..market import _raw, BookSeries
from ...common.paths import spot_tick_path, market_data_path

MAKER = Path(__file__).resolve().parents[3]
GAP = MAKER/'data/ev_lookup_v20_gap_diagnostic_20260909'
OLD = MAKER/'data/ev_lookup_audit_20260908'


def cash_at(series: BookSeries, ns: int, quantity: int, limits: tuple[int, int]) -> float | None:
    book = series.at(ns)
    taken = TakerDepth().take(book, 'buy', quantity, ns, *limits) if book else None
    return taken[0]/10000 if taken else None


def summarize(data: pl.DataFrame, name: str, scope: str) -> dict:
    both = data.filter(pl.col('pre_print_cash_twd').is_not_null() & pl.col('cash_50ms_twd').is_not_null())
    prints = data.filter(pl.col('trigger')=='print')
    both = both.with_columns(((pl.col('cash_50ms_twd')/pl.col('pre_print_cash_twd')-1)*10000).alias('delay_bp'))
    return dict(variant=name,scope=scope,n=data.height,both_priced=both.height,
        priced_at_print_only=data.filter(pl.col('pre_print_cash_twd').is_not_null() & pl.col('cash_50ms_twd').is_null()).height,
        priced_at_50ms_only=data.filter(pl.col('pre_print_cash_twd').is_null() & pl.col('cash_50ms_twd').is_not_null()).height,
        neither_priced=data.filter(pl.col('pre_print_cash_twd').is_null() & pl.col('cash_50ms_twd').is_null()).height,
        pre_print_repricing_twd=both['pre_print_delta_twd'].sum(),post_print_50ms_delta_twd=both['post_print_delta_twd'].sum(),
        both_total_repricing_twd=both['entry_repricing_delta_twd'].sum(),
        unsplit_50ms_priced_delta_twd=data.filter(pl.col('pre_print_cash_twd').is_null() & pl.col('cash_50ms_twd').is_not_null())['entry_repricing_delta_twd'].sum(),
        adverse_50ms=both.filter(pl.col('delay_bp')>1e-7).height,
        favorable_50ms=both.filter(pl.col('delay_bp') < -1e-7).height,
        unchanged_50ms=both.filter(pl.col('delay_bp').abs()<=1e-7).height,
        nonpositive_basis_at_cancel_window=data.filter(pl.col('basis_nonpositive_minus50ms')).height,
        cancel_window_unpriced=data['cash_minus50ms_twd'].null_count(),
        mean_delay_bp=both['delay_bp'].mean(),median_delay_bp=both['delay_bp'].median(),
        p95_delay_bp=both['delay_bp'].quantile(.95),p99_delay_bp=both['delay_bp'].quantile(.99),
        median_old_grid_to_print_ms=prints['trigger_lag_seconds'].median()*1000 if prints.height else None)


def main(output: Path) -> None:
    output.mkdir(parents=True,exist_ok=False)
    prices=pl.read_parquet(GAP/'old_s2_entry_prices.parquet')
    candidates=pl.read_parquet(OLD/'resolved_candidates.parquet').select('cid','live_day')
    prices=prices.join(candidates,on='cid',how='left')
    rows=[]
    for day in sorted(prices['day'].unique().to_list()):
        today=prices.filter(pl.col('day')==day)
        codes=today['vc'].unique().to_list()
        raw=_raw(spot_tick_path(day),codes,future=False,start=open_ns(day),end=open_ns(day)+15600*SECOND)
        series={g.item(0,'QuoteCode'):BookSeries('S:'+g.item(0,'QuoteCode'),g) for g in raw.partition_by('QuoteCode')}
        metadata=pl.scan_parquet(market_data_path(day)).filter(pl.col('quote_code').is_in(codes)).select(
            'quote_code','limit_down_price','limit_up_price').collect()
        limits={r['quote_code']:(round(r['limit_down_price']*10000),round(r['limit_up_price']*10000)) for r in metadata.to_dicts()}
        for r in today.to_dicts():
            quantity=round(r['old_nominal_twd']/r['old_spot_price'])
            pre=cash_at(series[r['vc']],r['maker_ns']-1,quantity,limits[r['vc']])
            cancel_window=cash_at(series[r['vc']],r['maker_ns']-50_000_001,quantity,limits[r['vc']])
            after={ms:cash_at(series[r['vc']],r['maker_ns']+ms*1000000,quantity,limits[r['vc']]) for ms in (0,10,50,100)}
            saved=r['new_nominal_twd']
            if (saved is None)!=(after[50] is None) or (saved is not None and abs(saved-after[50])>1e-6):
                raise AssertionError('50ms book price does not reproduce prior diagnostic')
            fee_factor=1+(20 if r['day']==r['live_day'] else 34)/10000
            before_delta=-(pre-r['old_nominal_twd'])*fee_factor if pre is not None else None
            delay_delta=-(after[50]-pre)*fee_factor if pre is not None and after[50] is not None else None
            if delay_delta is not None and abs(before_delta+delay_delta-r['entry_repricing_delta_twd'])>1e-6:
                raise AssertionError('timing components do not sum to prior total')
            rows.append(dict(r,quantity=quantity,pre_print_cash_twd=pre,
                             cash_minus50ms_twd=cancel_window,
                             basis_nonpositive_minus50ms=(r['held_future_price']*quantity-cancel_window<=1e-7) if cancel_window is not None else None,
                             pre_print_delta_twd=before_delta,post_print_delta_twd=delay_delta,
                             **{f'cash_{ms}ms_twd':value for ms,value in after.items()}))
    result=pl.from_dicts(rows,infer_schema_length=None)
    result.write_parquet(output/'entry_latency_components.parquet')
    summaries=[]
    for name in ['v17_reference_20','v18_reference_20']:
        trace=pl.read_parquet(OLD/f'{name}_trace.parquet')
        ids=trace.filter((pl.col('strm')=='S2') & (pl.col('reason')=='ok'))['cid'].to_list()
        selected=result.filter(pl.col('cid').is_in(ids))
        for scope in ['all_original_premises','same_contract_print']:
            data=selected if scope=='all_original_premises' else selected.filter(
                (pl.col('trigger')=='print') & (pl.col('quote_contract')==pl.col('trigger_contract')))
            summaries.append(summarize(data,name,scope))
    pl.from_dicts(summaries).write_csv(output/'entry_latency_summary.csv')
    result.filter(pl.col('cid')==633).write_csv(output/'example_6488.csv')
    facts=pl.read_parquet(GAP/'position_attribution.parquet')
    regular=facts.filter(pl.col('category')=='maker_exit').with_columns(
        (pl.col('entry_day')==pl.col('close_day')).alias('same_day'),
        ((pl.col('anchor')-5)*pl.col('nominal_twd')/10000).alias('expiry_vs_target_branch_gap_twd'))
    branch=(regular.group_by('portfolio','stream','same_day').agg(pl.len(),pl.col('nominal_twd').sum(),
              pl.col('expiry_vs_target_branch_gap_twd').sum(),pl.col('actual_pnl_twd').sum())
              .sort('portfolio','stream','same_day'))
    branch.write_csv(output/'expiry_vs_actual_exit_policy.csv')
    info=dict(passed=True,candidates=result.height,summary=summaries,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interpretation='Fixed original fills; receive-time markouts. Pre-print versus +50ms is not exchange-clock causality. '
        'Missing books and other-contract/book-cross premises remain explicit. No new strategy or true inside-order fill time is inferred.')
    (output/'summary.json').write_text(json.dumps(info,indent=2)+'\n')
    (output/'decompose_s2_entry_latency.py').write_text(Path(__file__).read_text())
    print(json.dumps(info,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    main(parser.parse_args().output)
