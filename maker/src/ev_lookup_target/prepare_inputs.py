"""Freeze model inputs per decision day, including observable release-risk sets."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import polars as pl

from .q_model import QSnapshot
from .release_model import SCHEMA, ReleaseSnapshot, daily_exposures


def prepare(facts: Path, output: Path):
    manifest=json.loads((facts/'manifest.json').read_text())
    if manifest['status'] != 'completed':
        raise ValueError('incomplete facts')
    source=Path(manifest['source'])
    source_hashes={str(Path(k).resolve()):v for k,v in manifest['source_files_sha256'].items()}
    output.mkdir(parents=True,exist_ok=False)
    prior, hashes, counts=[],{},[]
    for day in manifest['days']:
        # Fit BEFORE reading this day's position outcome file.
        training=pl.concat([r for _,r in prior[-20:]]) if prior else pl.DataFrame(schema=SCHEMA)
        release=ReleaseSnapshot.fit(day,training)
        q=QSnapshot.fit(facts,day)
        folder=output/f'Date={day}'
        folder.mkdir()
        (folder/'q_snapshot.json').write_text(json.dumps(q.to_dict(),indent=2)+'\n')
        (folder/'release_snapshot.json').write_text(json.dumps(asdict(release),indent=2)+'\n')
        path=source/f'Date={day}/shadow/positions.parquet'
        expected=source_hashes[str(path.resolve())]
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError('shadow source changed since fact build')
        rows=daily_exposures(day,pl.read_parquet(path),outage=day in manifest['outages'])
        rows.write_parquet(folder/'release_observations.parquet')
        prior.append((day,rows))
        hashes[str(path)]=actual
        counts.append(dict(day=day,release_risks=rows.height,q_risk_cells=len(q.risk),
                           release_cells=len(release.cells),observed_prior_sessions=q.observed_prior_sessions))
        print(json.dumps(counts[-1]),flush=True)
    (output/'manifest.json').write_text(json.dumps(dict(status='completed',days=manifest['days'],
        facts=str(facts.resolve()),source=str(source.resolve()),counts=counts,source_files_sha256=hashes,
        program_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        interpretation='Each opening Q/release snapshot is fitted before reading that day outcomes. '
        'Daily observations remain inputs to later sessions only.'),indent=2)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('facts',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    prepare(args.facts,args.output)
