"""Complete independent checks and reports after the continuous replay finishes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


PREFIX='src.research.futures_spot_spread.maker.src.ev_lookup_target.audit.'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--from-stage',choices=['verify_execution','decisions','capacity','report'],
                        default='verify_execution')
    args=parser.parse_args()
    root=args.root
    status=root.parent/'finalization_status.json'
    last=None
    while True:
        path=root/'manifest.json'
        manifest=json.loads(path.read_text()) if path.exists() else {}
        state=dict(stage='replay',status=manifest.get('status','starting'),
                   sessions=len(manifest.get('completed_days',[])))
        if state!=last:
            status.write_text(json.dumps(state,indent=2)+'\n')
            print(json.dumps(state),flush=True)
            last=state
        if manifest.get('status')=='failed':
            raise RuntimeError('Replay failed; retain checkpoint and inspect execution log.')
        if manifest.get('status')=='completed':
            break
        time.sleep(30)
    steps=[('verify_execution',[str(root),'--raw-days','20260511','20260706','20260724','20260902']),
           ('decisions',[str(root)]),('capacity',[str(root)]),
           ('report',[str(root),str(root.parent/'report')])]
    begin=[module for module,_ in steps].index(args.from_stage)
    for module,_ in steps[:begin]:
        check='full' if module=='verify_execution' else module
        assert json.loads((root/f'verification_{check}.json').read_text())['passed'], 'earlier audit stage not verified'
    steps=steps[begin:]
    for module,arguments in steps:
        status.write_text(json.dumps(dict(stage=module,status='running'),indent=2)+'\n')
        try:
            with (root.parent/f'full_{module}.log').open('w') as handle:
                subprocess.run(['uv','run','python','-m',PREFIX+module,*arguments],
                               stdout=handle,stderr=subprocess.STDOUT,check=True)
        except Exception:
            status.write_text(json.dumps(dict(stage=module,status='failed'),indent=2)+'\n')
            raise
        print(json.dumps(dict(stage=module,status='passed')),flush=True)
    source=root.parent/'audit_source_snapshot'
    source.mkdir(exist_ok=True)
    hashes={}
    for path in Path(__file__).parent.glob('*.py'):
        shutil.copy2(path,source/path.name)
        hashes[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    (root.parent/'audit_sources.json').write_text(json.dumps(hashes,indent=2)+'\n')
    status.write_text(json.dumps(dict(stage='completed',human_report_review_pending=True),indent=2)+'\n')
    print('Replay, independent audits and numeric reports completed; written research assessment remains pending.',flush=True)


if __name__=='__main__':
    main()
