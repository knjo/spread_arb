"""Compare completed audited runs as soon as both numeric reports exist."""
import argparse
import json
from pathlib import Path
import time

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('event_root',type=Path);p.add_argument('old_root',type=Path)
    a=p.parse_args();status=a.event_root.parent/'comparison_status.json'
    last=None
    while True:
        stages={}
        for kind,root in (('event',a.event_root),('old',a.old_root)):
            path=root.parent/'finalization_status.json'
            state=json.loads(path.read_text()) if path.exists() else {}
            if state.get('status')=='failed':
                status.write_text(json.dumps(dict(stage='failed',dependency=kind,detail=state),indent=2)+'\n')
                raise RuntimeError(f'{kind} finalization failed; preserve output and inspect its stage log')
            stages[kind]=state.get('stage','starting')
        if stages!=last:
            status.write_text(json.dumps(dict(stage='waiting',dependencies=stages),indent=2)+'\n')
            print(json.dumps(stages),flush=True);last=stages
        if all(stage=='completed' for stage in stages.values()):
            break
        time.sleep(30)
    try:
        from .event_compare import run
        status.write_text(json.dumps(dict(stage='running'),indent=2)+'\n')
        result=run(a.event_root,a.old_root,a.event_root.parent/'comparison')
    except Exception as error:
        status.write_text(json.dumps(dict(stage='failed',error=repr(error)),indent=2)+'\n')
        raise
    status.write_text(json.dumps(dict(stage='completed',written_review_pending=True,
        portfolios=[r['portfolio'] for r in result['summaries']]),indent=2)+'\n')
    print('Same-period fixed-Q profit/turnover comparison and release calibration completed.',flush=True)


if __name__=='__main__':
    main()
