"""Fixed-fill PnL identities for v20; diagnostic, never a policy backtest."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import polars as pl

MAKER = Path(__file__).resolve().parents[3]
ROOT = MAKER / 'data/ev_lookup_v20_capacity_full_20260908_r4'
OLD = MAKER / 'data/ev_lookup_audit_20260908'


def read_rows(path: Path) -> list[dict]:
    return pl.read_parquet(path).to_dicts() if path.exists() else []


def cell_of(stream: str, ab: float) -> str:
    return f'{stream}_{sum(ab > edge for edge in (30., 50., 80.))}'


def actor_facts(root: Path, name: str, days: list[str]) -> tuple[list[dict], dict]:
    latest = {}
    flags = defaultdict(set)
    events = defaultdict(int)
    for day in days:
        folder = root / f'Date={day}' / name
        for p in read_rows(folder / 'positions.parquet'):
            if p['entry_fill_ns'] is not None:
                latest[p['id']] = p
        for event in read_rows(folder / 'execution.parquet'):
            events[event['kind']] += 1
            if event['kind'] in {'expiry_basis_zero_accounting', 'corporate_risk_exit', 'partial_entry_rollback'}:
                flags[event['position_id']].add(event['kind'])
    records = []
    for p in latest.values():
        pid = p['id']
        nominal = p['spot_buy_cash'] / 10000
        is_pair = p['hedged_ns'] is not None and p['future_sell_qty'] > 0
        category = ('open' if p['state'] != 'closed' else
                    'expiry' if 'expiry_basis_zero_accounting' in flags[pid] else
                    'corporate' if 'corporate_risk_exit' in flags[pid] else
                    'rollback' if 'partial_entry_rollback' in flags[pid] else 'maker_exit')
        row = dict(portfolio=name, id=pid, stream=p['stream'], vc=p['contract']['vc'],
                   qc=p['contract']['qc'], source_intent=p['source_intent'],
                   entry_day=p['entry_day'], close_day=p['close_day'], category=category,
                   paired=is_pair, nominal_twd=nominal, actual_pnl_twd=p['pnl_twd'],
                   quote_ab=p['quote_ab'], anchor=p['anchor'], actual_ab=p['actual_ab'],
                   wait_to_first_fill_seconds=(p['entry_fill_ns']-p['quote_ns'])/1e9,
                   quoted_formula_twd=None, entry_price_gap_twd=None,
                   actual_entry_target_exit_twd=None, exit_price_gap_twd=None)
        if is_pair and category != 'open':
            cost = 20. if p['entry_day'] == p['close_day'] else 34.
            formula = (p['quote_ab'] - p['anchor'] + 5. - cost) * nominal / 10000
            entry_gap = (p['actual_ab'] - p['quote_ab']) * nominal / 10000
            actual_entry_target = formula + entry_gap
            exit_gap = p['pnl_twd'] - actual_entry_target
            if category == 'maker_exit':
                exit_cash_bp = (p['future_buy_cash']-p['spot_sell_cash']) / p['spot_buy_cash'] * 10000
                direct = (p['anchor']-5.-exit_cash_bp)*nominal/10000
                if abs(direct-exit_gap) > 1e-5:
                    raise AssertionError('four-leg exit identity failed')
            row.update(quoted_formula_twd=formula, entry_price_gap_twd=entry_gap,
                       actual_entry_target_exit_twd=actual_entry_target, exit_price_gap_twd=exit_gap)
        records.append(row)
    daily = pl.read_csv(root / f'{name}_daily.csv', schema_overrides={'day': pl.String})
    realized = sum(r['actual_pnl_twd'] for r in records if r['category'] != 'open')
    if abs(realized-daily['realized_twd'].sum()) > 1e-5:
        raise AssertionError('closed positions and daily realized do not reconcile')
    return records, dict(portfolio=name, maker_triggered_entries=len(records),
                         paired_entries=sum(r['paired'] for r in records),
                         realized_twd=realized, final_official_equity_twd=daily['official_equity_twd'][-1],
                         final_mark_twd=daily['official_marked_open_twd'][-1], events=dict(events))


def main(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    if manifest['status'] != 'completed':
        raise AssertionError('completed source run required')
    days = manifest['days']
    records, actors = [], []
    for name in ['ev_20M', 'ev_bpday_20M', 'fcfs_20M', 'shadow']:
        rows, summary = actor_facts(ROOT, name, days)
        records += rows
        actors.append(summary)
    frame = pl.from_dicts(records, infer_schema_length=None)
    frame.write_parquet(output / 'position_attribution.parquet')
    money = ['nominal_twd', 'actual_pnl_twd', 'quoted_formula_twd', 'entry_price_gap_twd',
             'actual_entry_target_exit_twd', 'exit_price_gap_twd']
    groups = (frame.group_by('portfolio','stream','category').agg(pl.len().alias('n'),
               *[pl.col(c).sum() for c in money]).sort('portfolio','stream','category'))
    groups.write_csv(output / 'cashflow_groups.csv')
    regular = frame.filter(pl.col('category') == 'maker_exit')
    regular.group_by('portfolio').agg(pl.len().alias('n'), *[pl.col(c).sum() for c in money]).write_csv(output / 'regular_exit_totals.csv')

    # Hold the v20 shadow fills, resolution dates, bins, and denominator fixed.
    # Substitute only the old target formula for ordinary closed maker exits.
    # This isolates a label effect; it does not replay the resulting admissions.
    cells = []
    shadow = [r for r in records if r['portfolio'] == 'shadow' and r['category'] != 'open']
    for index, day in enumerate(days):
        prior = set(days[max(0,index-20):index])
        stats = defaultdict(lambda: [0., 0., 0., 0])
        for r in shadow:
            if r['close_day'] not in prior:
                continue
            held = max(days.index(r['close_day'])-days.index(r['entry_day']), .15)
            nominal = r['nominal_twd']
            pnl = r['actual_pnl_twd']
            target = r['quoted_formula_twd'] if r['category'] == 'maker_exit' else pnl
            s = stats[cell_of(r['stream'], r['quote_ab'])]
            s[0] += pnl / nominal * 10000
            s[1] += target / nominal * 10000
            s[2] += held
            s[3] += 1
        for cell, (pnl, target, held, n) in stats.items():
            cells.append(dict(day=day, cell=cell, n=n, actual_bpday=pnl/held,
                              target_formula_bpday=target/held,
                              actual_admit=n<30 or pnl/held>=8.,
                              target_admit=n<30 or target/held>=8.))
    cell_frame = pl.from_dicts(cells)
    cell_frame.write_csv(output / 'same_shadow_label_cells.csv')
    # Independently reconcile actual cells to the saved morning snapshots.
    indexed = {(r['day'],r['cell']):r for r in cells}
    for snap in json.loads((ROOT / 'snapshots.json').read_text()):
        if snap['portfolio'] != 'ev_20M':
            continue
        for cell, (pnl, held, n) in snap['cells'].items():
            actual = indexed[(snap['day'],cell)]
            if actual['n'] != n or abs(actual['actual_bpday'] - pnl/held) > 1e-6:
                raise AssertionError('diagnostic prior cell differs from saved replay')

    old_frame = pl.read_parquet(OLD / 'exit_price_proxy_diagnostic.parquet')
    old_price = (old_frame.group_by('strm').agg(pl.len().alias('n'),
                 (pl.col('live_bp') * pl.col('ntl') / 10000).sum().alias('old_target_twd'),
                 (pl.col('price_gap_bp') * pl.col('ntl') / 10000).sum().alias('exit_price_proxy_gap_twd')))
    old_price.write_csv(output / 'old_fixed_exit_price_groups.csv')
    summary = dict(actors=actors,
                   regular_exit_totals=pl.read_csv(output / 'regular_exit_totals.csv').to_dicts(),
                   mature_cell_days=cell_frame.filter(pl.col('n')>=30).height,
                   mature_cells_rejected_actual_pass_target=cell_frame.filter((pl.col('n')>=30)&~pl.col('actual_admit')&pl.col('target_admit')).height,
                   interpretation='Fixed-fill arithmetic attribution only. Target formulas are not executable returns; no counterfactual scheduling, capacities, or admissions were replayed.')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (output / 'manifest.json').write_text(json.dumps(dict(source_run=str(ROOT), old_audit=str(OLD),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), passed=True), indent=2)+'\n')
    (output / 'attribute_v20_gap.py').write_text(Path(__file__).read_text())
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    main(parser.parse_args().output)
