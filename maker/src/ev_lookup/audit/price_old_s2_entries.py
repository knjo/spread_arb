"""Reprice original accepted S2 hedges; unchanged candidate/exit collection."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import polars as pl

from ..causal_lookup import open_ns, SECOND
from ..execution import TakerDepth
from ..market import _raw, BookSeries, WF
from ...common.paths import futures_raw_path, spot_tick_path, market_data_path

MAKER = Path(__file__).resolve().parents[3]
OLD = MAKER / 'data/ev_lookup_audit_20260908'
EXT = Path('/tmp/claude-1000/-home-kevin-Project-HFT/41972a34-5d7e-4f48-b275-3f9588a394bc/scratchpad/ext_daily')


def first_match(group, start, end, price):
    if group is None:
        return None
    times, prices = group
    a,b=np.searchsorted(times,start,side='right'),np.searchsorted(times,end,side='right')
    indices=np.flatnonzero(prices[a:b]>=price-1e-8)
    return int(times[a+indices[0]]) if len(indices) else None


def original_front(day: str, universe: list[str], families: dict) -> tuple[dict, dict]:
    """Match the archived full-universe group order and NumPy quicksort ties."""
    volumes = (pl.scan_parquet(futures_raw_path(day))
        .filter((pl.col('TrialMatch') == 0) & pl.col('ValueCode').is_in(universe))
        .select('ValueCode', 'QuoteCode', 'TotalFillLots')
        .with_columns(pl.col('ValueCode').replace_strict(families).alias('family'))
        .filter(pl.col('QuoteCode').str.slice(0, 3) == pl.col('family'))
        .group_by('ValueCode', 'QuoteCode').agg(pl.col('TotalFillLots').max())
        .collect().sort('ValueCode', 'QuoteCode'))
    if volumes['TotalFillLots'].null_count():
        raise AssertionError('null volumes require explicit archived-sort handling')
    order = np.argsort(volumes['TotalFillLots'].to_numpy(), kind='quicksort')
    records = volumes.to_dicts()
    chosen = {records[i]['ValueCode']: records[i]['QuoteCode'] for i in order}
    ties = (volumes.filter(pl.col('TotalFillLots') == pl.col('TotalFillLots').max().over('ValueCode'))
            .group_by('ValueCode').agg(pl.col('QuoteCode').sort()))
    return chosen, dict(ties.iter_rows())


def main(output: Path, from_day: str | None = None, through_day: str | None = None, resume: bool = False):
    output.mkdir(parents=True,exist_ok=resume)
    variants=['v17_reference_20','v18_reference_20']
    selected={}
    for name in variants:
        t=pl.read_parquet(OLD/f'{name}_trace.parquet')
        selected[name]=set(t.filter((pl.col('reason')=='ok')&(pl.col('strm')=='S2'))['cid'])
    ids=set.union(*selected.values())
    candidates=pl.read_parquet(OLD/'resolved_candidates.parquet').filter(pl.col('cid').is_in(list(ids)))
    if from_day:candidates=candidates.filter(pl.col('day0')>=from_day)
    if through_day:candidates=candidates.filter(pl.col('day0')<=through_day)
    basic=pl.read_parquet(WF/'daily/Date=20260813/causal_fair.parquet',columns=['ValueCode','QuoteCode','contract_size']).drop_nulls().unique()
    families={r['ValueCode']:r['QuoteCode'][:3] for r in basic.to_dicts()}
    sizes={r['ValueCode']:round(r['contract_size']) for r in basic.to_dicts()}
    resumed = sorted(output.glob('Date=*.parquet')) if resume else []
    rows=[r for path in resumed for r in pl.read_parquet(path).to_dicts()]
    done={r['day'] for r in rows}
    for day in sorted(candidates['day0'].unique().to_list()):
        if day in done:continue
        today=candidates.filter(pl.col('day0')==day)
        codes=today['vc'].unique().to_list()
        grid=WF/f'daily/Date={day}/causal_fair.parquet'
        if not grid.exists():grid=EXT/f'{day}.parquet'
        if not grid.exists():raise FileNotFoundError(grid)
        schema=pl.read_parquet_schema(grid)
        cols=['ValueCode','seconds_from_open','fut_bid']+(['QuoteCode'] if 'QuoteCode' in schema else [])
        g=pl.scan_parquet(grid).filter(pl.col('ValueCode').is_in(codes)).select(cols).collect().sort('ValueCode','seconds_from_open')
        g=g.with_columns(pl.col('fut_bid').fill_nan(None).forward_fill().over('ValueCode'))
        books={(r['ValueCode'],r['seconds_from_open']):r for r in g.to_dicts()}
        raw=pl.scan_parquet(futures_raw_path(day)).filter((pl.col('TrialMatch')==0)&pl.col('ValueCode').is_in(codes)).select(
            'ValueCode','QuoteCode','RecvTime','FillPrice','FillLots','DecimalLocator','TotalFillLots').collect()
        raw=raw.with_columns(pl.col('ValueCode').replace_strict(families).alias('family')).filter(pl.col('QuoteCode').str.slice(0,3)==pl.col('family'))
        volumes=raw.group_by('ValueCode','QuoteCode').agg(pl.col('TotalFillLots').max())
        tops=volumes.filter(pl.col('TotalFillLots')==pl.col('TotalFillLots').max().over('ValueCode'))
        tied_fronts=dict(tops.group_by('ValueCode').agg(pl.col('QuoteCode').sort()).iter_rows())
        chosen={vc:qcs[0] for vc,qcs in tied_fronts.items()}
        exact_front=None
        raw=raw.filter(pl.col('FillLots')>0).with_columns(pl.col('RecvTime').dt.epoch('ns').alias('ns'),
            (pl.col('FillPrice')*pl.lit(10.).pow(-pl.col('DecimalLocator').cast(pl.Float64))).alias('px')).sort('ValueCode','QuoteCode','ns')
        trades={(p.item(0,'ValueCode'),p.item(0,'QuoteCode')):(p['ns'].to_numpy(),p['px'].to_numpy()) for p in raw.partition_by('ValueCode','QuoteCode')}
        start=open_ns(day);end=start+15600*SECOND
        spot=_raw(spot_tick_path(day),codes,future=False,start=start,end=end)
        spot_books={p.item(0,'QuoteCode'):BookSeries('S:'+p.item(0,'QuoteCode'),p) for p in spot.partition_by('QuoteCode')}
        metadata=pl.scan_parquet(market_data_path(day)).filter(pl.col('quote_code').is_in(codes)).select('quote_code','limit_down_price','limit_up_price').collect()
        limits={r['quote_code']:(round(r['limit_down_price']*10000),round(r['limit_up_price']*10000)) for r in metadata.to_dicts()}
        for c in today.to_dicts():
            vc=c['vc'];shares=sizes[vc];old_spot=c['ntl']/shares
            held=round(old_spot*(1+c['eb']/10000),4)
            model=books[(vc,c['t0'])]
            lo=start+c['t0']*SECOND;hi=lo+SECOND
            chosen_qc=chosen[vc]
            quote_qc=model.get('QuoteCode')
            crossed=model['fut_bid'] is not None and held<=model['fut_bid']+1e-9
            ns=lo if crossed else first_match(trades.get((vc,chosen_qc)),lo,hi,held)
            tie_inferred = False
            if not crossed and len(tied_fronts[vc])>1:
                alternatives = [(qc, first_match(trades.get((vc,qc)),lo,hi,held)) for qc in tied_fronts[vc]]
                alternatives = [(qc, value) for qc,value in alternatives if value is not None]
                if len(alternatives) == 1:
                    chosen_qc, ns = alternatives[0]
                    tie_inferred = True
                elif len(alternatives)>1:
                    if exact_front is None:
                        universe=[code for code in pl.read_parquet(grid,columns=['ValueCode'])['ValueCode'].unique() if code in families]
                        exact_front,_=original_front(day,universe,families)
                    chosen_qc=exact_front[vc]
                    ns=first_match(trades.get((vc,chosen_qc)),lo,hi,held)
            matching=first_match(trades.get((vc,quote_qc)),lo,hi,held) if quote_qc else None
            ready=ns+50000000 if ns is not None else None
            book=spot_books[vc].at(ready) if ready is not None else None
            taken=TakerDepth().take(book,'buy',shares,ready,*limits[vc]) if book else None
            cost=20. if c['day0']==c['live_day'] else 34.
            new_cash=taken[0]/10000 if taken else None
            delta=-(new_cash-c['ntl'])*(1+cost/10000) if taken else None
            rows.append(dict(cid=c['cid'],day=day,vc=vc,quote_contract=quote_qc,trigger_contract=chosen_qc,
                max_volume_contracts='|'.join(tied_fronts[vc]), tie_inferred=tie_inferred,
                trigger_reproduced=ns is not None, held_future_price=held,
                trigger='book_cross' if crossed else 'print',matching_contract_print=matching is not None,
                old_second=c['t0'],maker_ns=ns,hedge_ready_ns=ready,book_ns=book.ns if book else None,
                old_spot_price=old_spot,new_spot_price=new_cash/shares if taken else None,
                old_nominal_twd=c['ntl'],new_nominal_twd=new_cash,entry_repricing_delta_twd=delta,
                old_full_trade_pnl_twd=c['live_bp']*c['ntl']/10000 if c['live_day'] is not None else None))
        checkpoint = pl.from_dicts([r for r in rows if r['day'] == day],infer_schema_length=None)
        checkpoint.write_parquet(output / f'Date={day}.parquet')
        print(day,len(today),'unreproduced',checkpoint.filter(~pl.col('trigger_reproduced')).height,flush=True)
    f=pl.from_dicts(rows,infer_schema_length=None)
    f.write_parquet(output/'old_s2_entry_prices.parquet')
    summaries=[]
    for name,accepted in selected.items():
        d=f.filter(pl.col('cid').is_in(list(accepted)))
        priced=d.filter(pl.col('entry_repricing_delta_twd').is_not_null())
        supported=priced.filter((pl.col('trigger')=='print')&(pl.col('quote_contract')==pl.col('trigger_contract')))
        summaries.append(dict(variant=name,accepted_s2=d.height,priced_at_50ms=priced.height,
            trigger_unreproduced=d.filter(~pl.col('trigger_reproduced')).height,
            inferred_tie_contract=d.filter(pl.col('tie_inferred')).height,
            missing_at_50ms=d.height-priced.height,quote_identity_missing=d['quote_contract'].null_count(),
            printed_on_other_contract=d.filter((pl.col('trigger')=='print')&pl.col('quote_contract').is_not_null()&(pl.col('quote_contract')!=pl.col('trigger_contract'))).height,
            delta_all_priced_twd=priced['entry_repricing_delta_twd'].sum(),
            delta_closed_priced_twd=priced.filter(pl.col('old_full_trade_pnl_twd').is_not_null())['entry_repricing_delta_twd'].sum(),
            same_contract_print_priced=supported.height,same_contract_print_delta_twd=supported['entry_repricing_delta_twd'].sum()))
    result=dict(passed=True,variants=summaries,source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        resumed_checkpoints=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in resumed],
        interpretation='Original accepted S2 fills held fixed. Reprice only spot hedge at observed trigger +50ms, including changed fee basis. Other-contract triggers, book-cross fills, and missing depth remain flagged; this is not a corrected backtest or a live PnL estimate.')
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'price_old_s2_entries.py').write_text(Path(__file__).read_text())
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--from-day')
    parser.add_argument('--through-day')
    parser.add_argument('--resume',action='store_true',help='Continue immutable daily diagnostic checkpoints in the same date range.')
    args=parser.parse_args()
    main(args.output,args.from_day,args.through_day,args.resume)
