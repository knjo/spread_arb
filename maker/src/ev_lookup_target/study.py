"""Continuous Q/capacity study with pinned source, model inputs and checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from time import monotonic

from ..ev_lookup_cost.cost_study import COMMON
from ..ev_lookup_cost.full_study import OUTAGES, checkpoint, restore
from ..ev_lookup_cost.market import METADATA_ROOT, MarketDay
from ..ev_lookup_cost.replay import Replay
from .portfolio import TargetPortfolio
from .target import ReturnTarget


CONFIGS = [
    dict(name='q_net_reserved20',cap_twd=20_000_000,q_policy='net',reserve_quotes=True),
    dict(name='q_net_shared20',cap_twd=20_000_000,q_policy='net',reserve_quotes=False),
    dict(name='q_net_release25',cap_twd=25_000_000,q_policy='net',reserve_quotes=False,release_credit=True),
    dict(name='q_target_release25',cap_twd=25_000_000,q_policy='target',reserve_quotes=False,release_credit=True),
]


def configurations(names=None):
    configs=[]
    for config in CONFIGS:
        if names and config['name'] not in names:
            continue
        settings=dict(COMMON,repeg_drop_bp=10.,exit_open_delay_s=300,exit_event_guard=True,
                      exec_cost=True,max_ticket_twd=2_000_000,persistent_wait=True)
        settings.update(config)
        configs.append(settings)
    if names and 'control_20M' in names:
        configs.append(dict(COMMON,name='control_20M',repeg_drop_bp=10.,exit_open_delay_s=300,
                            exit_event_guard=True,exec_cost=True,persistent_wait=False))
    if not configs or (names and set(names)-{c['name'] for c in configs}):
        raise ValueError('unknown or empty configuration selection')
    return configs


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path,value):
    tmp=path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(path)


def source_paths():
    package=Path(__file__).resolve().parent
    shared=package.parent/'ev_lookup_cost'
    # Model and execution dependencies are frozen together. Audit-only files
    # can be developed after a replay starts without changing its model.
    return list(package.glob('*.py'))+list(shared.glob('*.py'))+list((package.parent/'common').glob('*.py'))


def build(output,models,configs,products=None):
    actors=[TargetPortfolio(**c,model_inputs=models) for c in configs]
    shadow=TargetPortfolio('shadow',10**12,use_ev=False,use_bpday=False,shadow=True,hedge_ms=50,cancel_ms=50,
        event_quotes=True,exec_cost=False,repeg_drop_bp=10.,exit_open_delay_s=300,exit_event_guard=True,
        persistent_wait=False,liquidity_rule=COMMON['liquidity_rule'])
    return Replay(output,actors,products=products,shadow=shadow)


def run(output,models,*,names=None,days=None,products=None,resume=False,max_days=None):
    output,models=Path(output).resolve(),Path(models).resolve()
    model_manifest=json.loads((models/'manifest.json').read_text())
    if model_manifest['status']!='completed':
        raise ValueError('model inputs incomplete')
    full_days=model_manifest['days']
    selected=days or full_days
    if not selected or selected!=full_days[full_days.index(selected[0]):full_days.index(selected[-1])+1]:
        raise ValueError('replay days must be a contiguous calendar interval')
    configs=configurations(names)
    replay=build(output,models,configs,products)
    spec=dict(version='target30_q_capacity_v24',days=selected,available_days=[d for d in selected if d not in OUTAGES],
        data_outage_days=[d for d in selected if d in OUTAGES],configurations=configs,products=products,
        model_inputs=str(models),target=ReturnTarget().summary(),
        q_gate='net Q >= 0, or net Q >= 30%-annual calendar holding charge; legacy lambda is diagnostic only',
        warmup='20 cumulative observed prior sessions; a later feed outage does not restart warmup',
        allocation='Quotes each require enough observable room for their own ticket. Shared quotes do not pre-reserve. '
                   'First fills consume actual ledger capacity; other quotes cancel with 50ms delay. Race fills remain.',
        credit='20M base + min(5M, 0.5*discounted today releases + 0.25*discounted next-session releases); '
               'prior-20-session surviving risk sets only; actual ledger never releases predicted amounts',
        chronology='Observable waiting intents ordered by original timestamp; no intraday future-outcome ranking. '
                   'Released room retries valid intents with a fresh quote/queue timestamp.',
        limitations='Same-period exploratory research, immediate new queue admission, unthrottled messages, '
                    'cancel/hedge schedule 50ms but retries can be longer, C8 expiry accounting convention; '
                    'actual capital may exceed admission ceiling during a cancellation race.',
        hedge_ms=50,cancel_ms=50,event_quotes=True,finance_cost_included=False)
    manifest_path=output/'manifest.json'
    if resume:
        manifest=json.loads(manifest_path.read_text())
        if any(manifest.get(k)!=v for k,v in spec.items()):
            raise ValueError('resume policy, date range or target changed')
        for path,expected in {**manifest['sources'],**manifest['inputs_sha256']}.items():
            if digest(Path(path))!=expected:
                raise ValueError(f'changed source or input: {path}')
        restore(replay)
        if replay.history.sessions!=selected[:len(replay.history.sessions)]:
            raise ValueError('checkpoint calendar differs')
    else:
        output.mkdir(parents=True,exist_ok=False)
        source_dir=output/'source_snapshot'
        source_dir.mkdir()
        sources={}
        for p in source_paths():
            relative=p.relative_to(Path(__file__).resolve().parents[1])
            dest=source_dir/relative
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,dest)
            sources[str(p)]=digest(p)
        inputs={}
        paths=[models/'manifest.json',METADATA_ROOT/'calendar_20260504_20260902.parquet',
               METADATA_ROOT/'announcements/index.json',METADATA_ROOT/'official_future_daily_marks.parquet']
        paths += [models/f'Date={d}'/kind for d in selected for kind in ('q_snapshot.json','release_snapshot.json')]
        input_dir=output/'input_snapshot'
        input_dir.mkdir()
        for i,p in enumerate(paths):
            inputs[str(p.resolve())]=digest(p)
            shutil.copy2(p,input_dir/f'{i:03d}_{p.name}')
        manifest=dict(spec,sources=sources,inputs_sha256=inputs,status='starting')
    manifest['status']='running'
    write_json(manifest_path,manifest)
    pending=selected[len(replay.history.sessions):]
    if max_days is not None:
        if max_days<=0:
            raise ValueError('max-days must be positive')
        pending=pending[:max_days]
    try:
        for day in pending:
            start=monotonic()
            carry=list({p.contract.qc:p.contract for a in replay.actors for p in a.positions.values() if p.id in a.active}.values())
            market=MarketDay(day,output,products,carry,data_outage=day in OUTAGES,depth_events=True)
            rows=replay.day(day,data_outage=day in OUTAGES,market=market)
            checkpoint(replay)
            print(json.dumps(dict(day=day,seconds=round(monotonic()-start,2),results=[
                {k:r.get(k) for k in ('portfolio','fills','realized_twd','official_equity_twd','carry_twd','cap_overrun_events')}
                for r in rows if r['portfolio']!='shadow'])),flush=True)
            del market
    except Exception as error:
        manifest.update(status='failed',failure_type=type(error).__name__,completed_days=replay.history.sessions)
        write_json(manifest_path,manifest)
        raise
    manifest.update(status='completed' if replay.history.sessions==selected else 'checkpointed',
        completed_days=replay.history.sessions,peak_committed_twd={a.name:a.ledger.peak_cents/100 for a in replay.portfolios})
    write_json(manifest_path,manifest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--models',required=True,type=Path)
    parser.add_argument('--configs',nargs='+')
    parser.add_argument('--days',nargs='+')
    parser.add_argument('--products',nargs='+')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--max-days',type=int)
    args=parser.parse_args()
    run(args.output,args.models,names=args.configs,days=args.days,products=args.products,
        resume=args.resume,max_days=args.max_days)


if __name__=='__main__':
    main()
