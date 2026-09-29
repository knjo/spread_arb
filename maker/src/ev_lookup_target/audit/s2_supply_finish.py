"""Finish the fixed S2 entry review after every declared session is written."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    root=p.parse_args().root
    protocol=json.loads((root/'protocol.json').read_text())
    status=root/'finalization.json'
    while True:
        done=[day for day in protocol['days'] if (root/f'Date={day}/summary.csv').exists()]
        status.write_text(json.dumps(dict(stage='waiting_for_sessions',completed_days=done),indent=2)+'\n')
        if len(done)==len(protocol['days']):
            break
        time.sleep(15)
    for module,extra in [('s2_supply_verify',['--raw']),('s2_supply_report',[])]:
        status.write_text(json.dumps(dict(stage=module,status='running'),indent=2)+'\n')
        try:
            with (root/f'{module}.log').open('w') as log:
                subprocess.run(['uv','run','python','-m',
                    'src.research.futures_spot_spread.maker.src.ev_lookup_target.audit.'+module,
                    str(root),*extra],stdout=log,stderr=subprocess.STDOUT,check=True)
        except Exception:
            status.write_text(json.dumps(dict(stage=module,status='failed'),indent=2)+'\n')
            raise
        print(module+' passed',flush=True)
    status.write_text(json.dumps(dict(stage='completed',written_research_review_pending=True),indent=2)+'\n')


if __name__=='__main__':
    main()
