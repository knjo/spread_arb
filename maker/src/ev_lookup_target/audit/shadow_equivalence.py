"""Ensure the model-label shadow matches the independently replayed full shadow."""
import argparse
import json
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal


def verify(root):
    manifest=json.loads((root/'manifest.json').read_text())
    models=json.loads((root/'input_snapshot/000_manifest.json').read_text())
    assert manifest['status']=='completed'
    assert manifest['days']==models['days'], 'full shadow comparison requires the same initial history'
    reference=Path(models['source'])
    files=rows=0
    for day in manifest['days']:
        for kind in ('decisions','execution','ledger','positions','marks'):
            path=Path(f'Date={day}')/'shadow'/f'{kind}.parquet'
            assert (reference/path).exists()==(root/path).exists()
            if not (reference/path).exists():
                continue
            expected=pl.read_parquet(reference/path)
            actual=pl.read_parquet(root/path)
            assert set(expected.columns)<=set(actual.columns)
            assert_frame_equal(actual.select(expected.columns),expected,check_exact=False,rel_tol=1e-12,abs_tol=1e-8)
            files+=1
            rows+=expected.height
    result=dict(passed=True,days=len(manifest['days']),files=files,rows=rows,reference=str(reference),
        interpretation='All original shadow columns agree across the same full calendar: quotes, fills, cash, '
                       'carry and marks. New actual portfolio policies did not change the independent label source.')
    (root/'verification_shadow_equivalence.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    print(json.dumps(verify(p.parse_args().root),indent=2))
