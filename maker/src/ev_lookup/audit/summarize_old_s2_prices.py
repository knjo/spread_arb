"""Reconcile complete fixed-candidate S2 price diagnostics, including exclusions."""
from __future__ import annotations

import argparse
import hashlib
import json
from math import isfinite
from pathlib import Path

import polars as pl

from .price_old_s2_entries import OLD, WF
from ..causal_lookup import open_ns


def main(root: Path) -> None:
    names = ['may', 'june', 'july', 'aug_sep']
    sources = [p for name in names for folder in [root/f'old_s2_{name}', root/f'old_s2_{name}_rest']
               for p in sorted(folder.glob('Date=*.parquet'))]
    prices = pl.concat([pl.read_parquet(p) for p in sources], how='vertical_relaxed').sort('day', 'cid')
    if prices['cid'].n_unique() != prices.height:
        raise AssertionError('overlapping checkpoint candidates')
    variants = ['v17_reference_20', 'v18_reference_20']
    ids = {}
    for name in variants:
        trace = pl.read_parquet(OLD/f'{name}_trace.parquet')
        ids[name] = set(trace.filter((pl.col('reason') == 'ok') & (pl.col('strm') == 'S2'))['cid'])
    expected = set.union(*ids.values())
    if set(prices['cid']) != expected:
        raise AssertionError(f'incomplete diagnostics: missing {len(expected-set(prices["cid"]))}')
    starts = {day: open_ns(day) for day in prices['day'].unique()}
    prices = prices.with_columns(((pl.col('maker_ns')-pl.col('day').replace_strict(starts))/1e9
                                 -pl.col('old_second')).alias('trigger_lag_seconds'))
    prices.write_parquet(root/'old_s2_entry_prices.parquet')
    records, status = [], []
    for name in variants:
        data = prices.filter(pl.col('cid').is_in(list(ids[name])))
        data = data.with_columns(
            pl.when(~pl.col('trigger_reproduced')).then(pl.lit('unreproduced'))
            .when(pl.col('trigger') == 'book_cross').then(pl.lit('book_cross'))
            .when(pl.col('quote_contract').is_null()).then(pl.lit('missing_quote_identity'))
            .when(pl.col('quote_contract') != pl.col('trigger_contract')).then(pl.lit('other_contract_print'))
            .otherwise(pl.lit('same_contract_print')).alias('support'))
        priced = data.filter(pl.col('entry_repricing_delta_twd').is_not_null())
        closed = priced.filter(pl.col('old_full_trade_pnl_twd').is_not_null())
        same = priced.filter(pl.col('support') == 'same_contract_print')
        records.append(dict(variant=name, accepted_s2=data.height, priced_at_50ms=priced.height,
            closed_priced=closed.height, missing_at_50ms=data.height-priced.height,
            unreproduced=data.filter(~pl.col('trigger_reproduced')).height,
            inferred_tie_contract=data.filter(pl.col('tie_inferred')).height,
            delta_priced_twd=priced['entry_repricing_delta_twd'].sum(),
            delta_closed_twd=closed['entry_repricing_delta_twd'].sum(),
            delta_closed_per_85_days=closed['entry_repricing_delta_twd'].sum()/85,
            same_contract_print_priced=same.height,
            same_contract_print_delta_twd=same['entry_repricing_delta_twd'].sum(),
            missing_depth_old_pnl_twd=data.filter(pl.col('entry_repricing_delta_twd').is_null())['old_full_trade_pnl_twd'].sum(),
            trigger_lag_median_seconds=priced.filter(pl.col('trigger') == 'print')['trigger_lag_seconds'].median()))
        grouped = (data.group_by('support').agg(pl.len().alias('n'),
                   pl.col('entry_repricing_delta_twd').count().alias('priced_n'),
                   pl.col('entry_repricing_delta_twd').sum(), pl.col('old_full_trade_pnl_twd').sum())
                   .with_columns(pl.lit(name).alias('variant')))
        status.extend(grouped.to_dicts())
        examples = same.filter(pl.col('old_full_trade_pnl_twd').is_not_null()).sort('entry_repricing_delta_twd')
        examples.head(12).write_csv(root/f'{name}_entry_price_examples.csv')

    # Old entry and exit substitutions use the SAME accepted collection per
    # variant. This is partial repricing, never a reallocation/backtest result.
    candidates = pl.read_parquet(OLD/'resolved_candidates.parquet')
    exits = pl.read_parquet(OLD/'exit_details.parquet')
    sizes = pl.read_parquet(WF/'daily/Date=20260813/causal_fair.parquet',
                           columns=['ValueCode','contract_size']).drop_nulls().unique().rename({'ValueCode':'vc'})
    legacy = json.loads((OLD/'allocation_summary.json').read_text())
    bridges, exit_summaries = [], []
    for name, entry in zip(variants, records):
        trace = pl.read_parquet(OLD/f'{name}_trace.parquet')
        accepted = trace.filter(pl.col('reason')=='ok').select('cid').join(candidates,on='cid')
        reconstructed = accepted.filter(pl.col('live_day').is_not_null()).select(
            (pl.col('live_bp')*pl.col('ntl')/10000).sum()).item()
        if not isfinite(reconstructed) or abs(reconstructed-legacy[name]['pnl_sum'])>1e-5:
            raise AssertionError('fixed candidates do not reconcile to original realized PnL')
        priced_exits = (exits.join(accepted.select('cid','live_day','ntl','strm','eb','eu','live_bp'),on='cid')
                        .filter(pl.col('day')==pl.col('live_day')).join(sizes,on='vc',how='left'))
        priced_exits = priced_exits.with_columns(pl.col('contract_size').fill_null(2000)).with_columns(
            ((pl.col('spot_exit_ask')-pl.col('future_ask_te'))*pl.col('contract_size')+
             (pl.col('eb')-pl.col('eu')-5)*pl.col('ntl')/10000).alias('exit_price_delta_twd'))
        if priced_exits['exit_price_delta_twd'].null_count() or not priced_exits['exit_price_delta_twd'].is_finite().all():
            raise AssertionError('missing old exit proxy prices require separate coverage handling')
        gap = priced_exits['exit_price_delta_twd'].sum()
        if name=='v18_reference_20' and abs(gap-(-1115063.291235081))>1e-5:
            raise AssertionError('old exit price diagnostic changed')
        priced_exits.write_parquet(root/f'{name}_exit_price_diagnostic.parquet')
        exit_summaries.append(dict(variant=name,maker_exits=priced_exits.height,exit_delta_twd=gap,exit_delta_per_85_days=gap/85))
        bridges.append(dict(variant=name,old_realized_twd=legacy[name]['pnl_sum'],
            priced_s2_entry_delta_twd=entry['delta_closed_twd'],priced_exit_delta_twd=gap,
            measured_price_delta_twd=entry['delta_closed_twd']+gap,
            measured_price_delta_per_85_days=(entry['delta_closed_twd']+gap)/85,
            partial_repricing_arithmetic_twd=legacy[name]['pnl_sum']+entry['delta_closed_twd']+gap,
            partial_repricing_arithmetic_per_85_days=(legacy[name]['pnl_sum']+entry['delta_closed_twd']+gap)/85))
    pl.from_dicts(exit_summaries).write_csv(root/'old_exit_price_summary.csv')
    pl.from_dicts(bridges).write_csv(root/'old_fixed_collection_price_bridge.csv')

    # Independently reconcile observed first prints against the older raw probe.
    checks = []
    for day in ['20260520', '20260706', '20260724']:
        sample = pl.read_parquet(OLD/f'raw_s2_{day}.parquet')
        joined = prices.filter(pl.col('day') == day).join(sample, left_on=['vc', 'old_second'], right_on=['vc', 't'], suffix='_sample')
        expected_n = prices.filter(pl.col('day') == day).height
        mismatched = joined.filter((pl.col('held_future_price')-pl.col('held_px')).abs()>1e-7).height
        contract_mismatches = joined.filter((pl.col('trigger')=='print')&(pl.col('trigger_contract')!=pl.col('print_contract'))).height
        trigger_mismatches = joined.filter(pl.col('trigger')!=pl.col('trigger_sample')).height
        timed = joined.filter(pl.col('trigger')=='print')
        max_timing_gap = timed.select(((pl.col('trigger_lag_seconds')+pl.col('old_second'))
                                      -pl.col('actual_print_sec')).abs().max()).item()
        if (joined.height != expected_n or mismatched or contract_mismatches or trigger_mismatches
                or (max_timing_gap is not None and max_timing_gap > 1e-6)):
            raise AssertionError(f'raw probe mismatch on {day}')
        checks.append(dict(day=day, accepted_union=prices.filter(pl.col('day')==day).height,
                           joined=joined.height, max_timing_gap_seconds=max_timing_gap,
                           held_price_contract_and_trigger_matched=True))

    pl.from_dicts(records).write_csv(root/'old_s2_price_summary.csv')
    pl.from_dicts(status).write_csv(root/'old_s2_price_support.csv')
    result = dict(passed=True, candidates=prices.height, entry_days=prices['day'].n_unique(),
                  variants=records, old_exit_prices=exit_summaries, partial_price_bridges=bridges, raw_probe_coverage=checks,
                  files=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources],
                  interpretation='Fixed old accepted S2 collection only. Price at reconstructed maker time +50ms. '
                  'Missing depth is unpriced, not zero loss. Other-contract/book-cross/missing-identity fills remain unsupported. '
                  'No new entry policy, maker queue, exits or capacity release was replayed.')
    (root/'old_s2_price_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    (root/'summarize_old_s2_prices.py').write_text(Path(__file__).read_text())
    print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    main(parser.parse_args().root)
