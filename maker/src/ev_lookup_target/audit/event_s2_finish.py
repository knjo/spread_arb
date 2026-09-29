"""Finish the full current-Q event-entry profit and turnover study."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    root=p.parse_args().root
    status=root.parent/'finalization_status.json'
    while True:
        path=root/'manifest.json';m=json.loads(path.read_text()) if path.exists() else {}
        state=dict(stage='replay',status=m.get('status','starting'),completed_days=len(m.get('completed_days',[])))
        status.write_text(json.dumps(state,indent=2)+'\n')
        if m.get('status')=='failed':
            raise RuntimeError('Replay failed; preserve outputs/checkpoint and inspect full_execution.log.')
        if m.get('status')=='completed':
            break
        time.sleep(30)
    prefix='src.research.futures_spot_spread.maker.src.ev_lookup_target.audit.'
    for module,extra in [('verify_execution',['--raw-days','20260511','20260706','20260724','20260902']),
                         ('decisions',[]),('capacity',[]),('event_s2_verify',[]),
                         ('report',[str(root.parent/'report')])]:
        status.write_text(json.dumps(dict(stage=module,status='running'),indent=2)+'\n')
        try:
            with (root.parent/f'full_{module}.log').open('w') as log:
                subprocess.run(['uv','run','python','-m',prefix+module,str(root),*extra],
                               stdout=log,stderr=subprocess.STDOUT,check=True)
        except Exception:
            status.write_text(json.dumps(dict(stage=module,status='failed'),indent=2)+'\n')
            raise
        print(module+' passed',flush=True)
    from .event_turnover import write
    write(root,root.parent/'report',m)
    snapshot=root.parent/'audit_source_snapshot';snapshot.mkdir(exist_ok=True);hashes={}
    for path in Path(__file__).parent.glob('*.py'):
        shutil.copy2(path,snapshot/path.name);hashes[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    (root.parent/'audit_sources.json').write_text(json.dumps(hashes,indent=2)+'\n')
    status.write_text(json.dumps(dict(stage='completed',old_policy_comparison_and_written_review_pending=True),indent=2)+'\n')
    print('Full current-Q event replay, audits and numeric profit/turnover report complete.',flush=True)


if __name__=='__main__':
    main()
