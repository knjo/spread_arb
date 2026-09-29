"""Compare every declared policy on the same allocated-capital denominator."""
import argparse
import json
from pathlib import Path

import polars as pl


BRANCHES=('','ticket_unrestricted','deep_spot','entry_window')


def maybe_combine(root):
    paths=[root/branch/'report/comparison.json' for branch in BRANCHES]
    if any(not path.exists() for path in paths):
        return dict(status='waiting_for_all_policy_reports')
    for branch in BRANCHES:
        folder=root/branch/'full'
        assert json.loads((folder/'manifest.json').read_text())['status']=='completed'
        for check in ('full','decisions','capacity','shadow_equivalence'):
            assert json.loads((folder/f'verification_{check}.json').read_text())['passed']
    baseline=json.loads((root/'baseline_report/comparison.json').read_text())['summaries'][0]
    post=json.loads((root/'baseline_report/postwarm_comparison.json').read_text())
    baseline.update(postwarm_first_day=post['first_after_warmup'],postwarm_observed_sessions=post['observed_sessions'],
        postwarm_net_change_twd=post['net_change_twd'],postwarm_net_per_observed_day_twd=post['net_per_observed_day_twd'],
        postwarm_simple_annual_250=post['net_per_observed_day_twd']*250/20_000_000,
        policy_group='prior_cost_control',ticket_limit_twd=None)
    rows=[baseline]
    for branch,path in zip(BRANCHES,paths):
        manifest=json.loads((root/branch/'full/manifest.json').read_text())
        configs={c['name']:c for c in manifest['configurations']}
        for row in json.loads(path.read_text())['summaries']:
            assert row['postwarm_first_day']==post['first_after_warmup']
            assert row['postwarm_observed_sessions']==post['observed_sessions']
            row.update(policy_group=branch or 'ticket_2M',ticket_limit_twd=configs[row['portfolio']].get('max_ticket_twd'))
            cap=pl.read_csv(root/branch/'report'/row['portfolio']/'capital_daily.csv',schema_overrides={'day':pl.String})
            cap=cap.filter(pl.col('day')>=post['first_after_warmup'])
            row['postwarm_mean_stock_cash_twd']=cap['mean_stock_cash_twd'].mean()
            row['postwarm_mean_committed_twd']=cap['mean_committed_twd'].mean()
            rows.append(row)
    output=root/'comparison_all'
    output.mkdir(exist_ok=True)
    frame=pl.from_dicts(rows,infer_schema_length=None)
    frame.write_csv(output/'policies.csv')
    result=dict(status='completed',summaries=rows,
        interpretation='All declared policies shown, including unsuccessful variants. Runtime Q/release tables '
            'are prior-only. Policy choices, added ticket-size and deep-queue comparisons were exploratory '
            'on an already studied period, not untouched out-of-sample validation. Baseline carries its own '
            'pre-June inventory; Q policies finish warmup flat. Annual rates use allocated 20M and a '
            '250-session planning convention, with actual extra capital and calendar results also retained.')
    (output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    print(json.dumps(maybe_combine(parser.parse_args().root),indent=2))
