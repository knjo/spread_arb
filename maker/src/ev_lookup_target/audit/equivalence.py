"""Compare execution and decision outputs across algebra/cache/checkpoint changes."""
import argparse
import json
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal


def verify(left,right):
    a=json.loads((left/'manifest.json').read_text())
    b=json.loads((right/'manifest.json').read_text())
    assert a['status']==b['status']=='completed'
    assert a['days']==b['days'] and a['configurations']==b['configurations']
    files=rows=0
    for day in a['days']:
        for name in ['shadow']+[c['name'] for c in a['configurations']]:
            for kind in ('decisions','execution','ledger','positions','marks'):
                relative=Path(f'Date={day}')/name/f'{kind}.parquet'
                assert (left/relative).exists()==(right/relative).exists()
                if not (left/relative).exists():
                    continue
                x,y=pl.read_parquet(left/relative),pl.read_parquet(right/relative)
                assert_frame_equal(x,y,check_exact=False,rel_tol=1e-12,abs_tol=1e-8)
                files+=1
                rows+=x.height
                del x,y
    result=dict(passed=True,files=files,rows=rows,reference=str(left),current=str(right),
        interpretation='Same chronological execution, positions, marks and admission decisions; '
                       'floating-point Q regrouping tolerance 1e-8 absolute / 1e-12 relative. '
                       'The current smoke used a fresh process restored from its day-one checkpoint.')
    (right/'execution_equivalence.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('left',type=Path)
    p.add_argument('right',type=Path)
    args=p.parse_args()
    print(json.dumps(verify(args.left,args.right),indent=2))
