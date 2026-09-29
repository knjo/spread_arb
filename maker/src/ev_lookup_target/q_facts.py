"""Build observable daily Q-table facts from the independently replayed shadow.

Each input is a same-day end-of-session snapshot, never the final position
snapshot projected backwards. Quote features come from the matching original
decision. Unclosed carry remains an exposure, and partial rollbacks are a
separate outcome rather than a successful same-day pair exit.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import polars as pl

from ..ev_lookup_cost.causal_lookup import CLOSE_SECOND, SECOND, open_ns


DAY_NS = 86_400 * SECOND
FEATURES = ['id', 'stream', 'quote_ns', 'quote_second', 'quote_ab', 'anchor',
            'entry_day', 'expiry', 'quote_spread_bp', 'quote_notional_twd', 'execution_cost_bp']
QUOTE_SCHEMA = {
    'id': pl.String, 'stream': pl.String, 'quote_ns': pl.Int64, 'quote_second': pl.Int64,
    'quote_ab': pl.Float64, 'anchor': pl.Float64, 'entry_day': pl.String, 'expiry': pl.String,
    'quote_spread_bp': pl.Float64, 'quote_notional_twd': pl.Float64, 'execution_cost_bp': pl.Float64,
    'day': pl.String, 'available_ns': pl.Int64, 'filled': pl.Boolean,
    'paired_by_close': pl.Boolean, 'closed_by_close': pl.Boolean,
}
RISK_SCHEMA = dict(QUOTE_SCHEMA, phase=pl.String, event=pl.String, age_days=pl.Int64,
                   days_to_expiry=pl.Int64, risk_start_ns=pl.Int64, risk_end_ns=pl.Int64,
                   outcome_net_bp=pl.Float64, outcome_net_bp_on_quote_capital=pl.Float64,
                   outcome_twd=pl.Float64, elapsed_calendar_days=pl.Float64,
                   close_second=pl.Float64, entry_decay_bp=pl.Float64, exit_decay_bp=pl.Float64)
COST_SCHEMA = dict(QUOTE_SCHEMA, kind=pl.String, bp=pl.Float64, carry=pl.Boolean)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def day_distance(left, right):
    return (datetime.strptime(right, '%Y%m%d') - datetime.strptime(left, '%Y%m%d')).days


def position_facts(day, positions, decisions, *, outage=False):
    """Return three frames; all labels have an explicit availability timestamp."""
    start, end = open_ns(day), open_ns(day) + CLOSE_SECOND * SECOND
    quoted = decisions.select(pl.col('intent_id').alias('id'), pl.col('ns').alias('quote_ns'),
                              (pl.col('reservation_cents') / 100).alias('quote_notional_twd'), 'execution_cost_bp')
    # Carry decisions live on the original quote day; caller supplies the
    # accumulated quote-feature index, not today's decision lookup alone.
    joined = positions.join(quoted, on=['id', 'quote_ns'], how='left', validate='m:1')
    if joined['quote_notional_twd'].null_count():
        raise AssertionError('position lacks original quote-time decision')
    quotes, risk, costs = [], [], []
    for p in joined.iter_rows(named=True):
        for key in ('quote_ns', 'entry_fill_ns', 'hedged_ns', 'close_ns'):
            if p[key] is not None and p[key] > end:
                raise AssertionError(f'future {key} in end-of-session position')
        expiry = p['contract']['expiry']
        base = {k: p[k] for k in FEATURES if k != 'expiry'}
        base.update(expiry=expiry, day=day, available_ns=end,
                    filled=p['entry_fill_ns'] is not None, paired_by_close=p['actual_ab'] is not None,
                    closed_by_close=p['close_ns'] is not None)
        if p['entry_day'] == day:
            quotes.append(base)
        if p['entry_fill_ns'] is None or (p['close_ns'] is not None and p['close_ns'] < start):
            continue
        if outage:
            # A feed outage is not observed failure to exit. Keep carry in
            # replay accounting, but do not turn absent market data into labels.
            continue
        closed = p['close_ns'] is not None
        kind = p['close_kind'] if closed else None
        event = ('survive' if not closed else 'normal' if kind == 'maker_exit' else
                 'rollback' if kind == 'partial_entry_rollback' else
                 'expiry' if kind == 'expiry_basis_zero_accounting' else 'other')
        phase = 'entry' if p['entry_day'] == day else 'carry'
        if event == 'expiry':
            phase = 'terminal'
        risk_start = max(start, p['entry_fill_ns'])
        risk_end = p['close_ns'] if closed else end
        actual_nominal = p['spot_buy_cash'] / 10_000
        entry_decay = p['quote_ab'] - p['actual_ab'] if p['actual_ab'] is not None else None
        exit_decay = (((p['future_buy_cash']-p['spot_sell_cash']) / p['spot_buy_cash'] * 10_000
                       - (p['anchor']-5)) if event == 'normal' and p['spot_buy_cash'] else None)
        risk.append(dict(base, phase=phase, event=event, age_days=day_distance(p['entry_day'], day),
                         days_to_expiry=day_distance(day, expiry), risk_start_ns=risk_start, risk_end_ns=risk_end,
                         outcome_net_bp=p['pnl_twd'] / actual_nominal * 10_000 if closed and actual_nominal else None,
                         outcome_net_bp_on_quote_capital=p['pnl_twd'] / p['quote_notional_twd'] * 10_000 if closed else None,
                         outcome_twd=p['pnl_twd'] if closed else None,
                         elapsed_calendar_days=(risk_end-p['entry_fill_ns']) / DAY_NS,
                         close_second=(p['close_ns']-start) / SECOND if closed else None,
                         entry_decay_bp=entry_decay, exit_decay_bp=exit_decay))
        if p['hedged_ns'] is not None and start <= p['hedged_ns'] <= end and entry_decay is not None:
            costs.append(dict(base, kind='entry', bp=entry_decay, carry=p['entry_day'] < day))
        if exit_decay is not None:
            costs.append(dict(base, kind='exit', bp=exit_decay, carry=p['entry_day'] < day))
    return (pl.from_dicts(quotes, schema=QUOTE_SCHEMA), pl.from_dicts(risk, schema=RISK_SCHEMA),
            pl.from_dicts(costs, schema=COST_SCHEMA))


def build(source: Path, output: Path):
    manifest = json.loads((source / 'manifest.json').read_text())
    if manifest['status'] != 'completed' or not json.loads((source / 'verification_cost.json').read_text())['passed']:
        raise ValueError('require completed independently audited execution source')
    output.mkdir(parents=True, exist_ok=False)
    days, hashes, counts = manifest['days'], {}, []
    index = pl.DataFrame(schema={'intent_id': pl.String, 'ns': pl.Int64, 'reservation_cents': pl.Int64,
                                'execution_cost_bp': pl.Float64})
    for day in days:
        folder = source / f'Date={day}' / 'shadow'
        position_path, decision_path = folder / 'positions.parquet', folder / 'decisions.parquet'
        if decision_path.exists():
            current = pl.read_parquet(decision_path, columns=['intent_id', 'ns', 'reservation_cents', 'execution_cost_bp'])
            index = pl.concat([index, current], how='vertical_relaxed').unique(['intent_id', 'ns'], keep='last')
            hashes[str(decision_path)] = digest(decision_path)
        positions = pl.read_parquet(position_path)
        hashes[str(position_path)] = digest(position_path)
        tables = position_facts(day, positions, index, outage=day in manifest['data_outage_days'])
        dest = output / f'Date={day}'
        dest.mkdir()
        for kind, table in zip(('quotes', 'risk', 'costs'), tables, strict=True):
            table.write_parquet(dest / f'{kind}.parquet')
        counts.append(dict(day=day, quotes=tables[0].height, risk=tables[1].height, costs=tables[2].height))
        # Closed/cancelled positions will not be present tomorrow. Retain only
        # the original decision rows required by unresolved carry.
        active = positions.filter(~pl.col('state').is_in(['closed', 'cancelled'])).select(pl.col('id').alias('intent_id'))
        index = index.join(active, on='intent_id', how='semi')
        print(json.dumps(counts[-1]), flush=True)
    result = dict(status='completed', source=str(source.resolve()), days=days,
                  outages=manifest['data_outage_days'], source_files_sha256=hashes,
                  implementation_sha256=digest(Path(__file__)), counts=counts,
                  semantics='Daily snapshots only; every label available at that session end. '
                  'Unresolved carry is a surviving risk observation, not discarded. '
                  'Partial rollback is distinct from normal same-day exit. Outages do not train hazards.')
    (output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def training_window(root: Path, decision_day: str, kind='risk', window=20):
    if kind not in {'risk', 'quotes', 'costs'} or window <= 0:
        raise ValueError('invalid table or training window')
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['status'] != 'completed':
        raise ValueError('incomplete fact build')
    days = [day for day in manifest['days'] if day < decision_day][-window:]
    schema = {'risk': RISK_SCHEMA, 'quotes': QUOTE_SCHEMA, 'costs': COST_SCHEMA}[kind]
    if not days:
        return pl.DataFrame(schema=schema)
    table = pl.concat([pl.read_parquet(root / f'Date={day}' / f'{kind}.parquet') for day in days])
    if table.filter((pl.col('available_ns') >= open_ns(decision_day)) | (pl.col('day') >= decision_day)).height:
        raise AssertionError('training table includes unavailable outcomes')
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    build(args.source, args.output)


if __name__ == '__main__':
    main()
