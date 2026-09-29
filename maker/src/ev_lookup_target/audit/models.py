"""Rebuild frozen Q/release inputs from strictly prior observable partitions."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import polars as pl

from ...ev_lookup_cost.causal_lookup import CLOSE_SECOND, SECOND, open_ns
from ..q_model import QSnapshot
from ..release_model import SCHEMA, ReleaseSnapshot, daily_exposures


def same(actual, expected, path='root'):
    if isinstance(expected, dict):
        assert set(actual)==set(expected), path
        for key in expected:
            same(actual[key],expected[key],path+'.'+key)
    elif isinstance(expected,(list,tuple)):
        assert len(actual)==len(expected), path
        for i,(left,right) in enumerate(zip(actual,expected)):
            same(left,right,path+f'[{i}]')
    elif isinstance(expected,float):
        assert abs(actual-expected)<1e-8, (path,actual,expected)
    else:
        assert actual==expected, (path,actual,expected)


def verify(root):
    manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['status']=='completed'
    facts=Path(manifest['facts'])
    fact_manifest=json.loads((facts/'manifest.json').read_text())
    source=Path(manifest['source'])
    prior=[]
    counts=dict(days=0,risk_rows=0,cost_rows=0,release_rows=0,raw_position_files=0)
    hashes={str(Path(k).resolve()):v for k,v in manifest['source_files_sha256'].items()}
    for day in manifest['days']:
        folder=root/f'Date={day}'
        # Reconstruct before opening this day's labeled partition.
        q=QSnapshot.fit(facts,day)
        actual=json.loads((folder/'q_snapshot.json').read_text())
        same(actual,q.to_dict())
        assert all(d<day for d in actual['train_days'])
        expected_sessions=sum(d<day and d not in fact_manifest['outages'] for d in manifest['days'])
        assert actual['observed_prior_sessions']==expected_sessions
        training=pl.concat(prior[-20:]) if prior else pl.DataFrame(schema=SCHEMA)
        release=ReleaseSnapshot.fit(day,training)
        same(json.loads((folder/'release_snapshot.json').read_text()),asdict(release))
        for kind in ('risk','costs'):
            rows=pl.read_parquet(facts/f'Date={day}/{kind}.parquet')
            assert not rows.filter((pl.col('day')!=day) |
                (pl.col('available_ns')!=open_ns(day)+CLOSE_SECOND*SECOND)).height
            assert not rows.filter(pl.col('quote_ns')>pl.col('available_ns')).height
            if kind=='risk':
                assert not rows.filter((pl.col('risk_start_ns')>pl.col('available_ns')) |
                                       (pl.col('risk_end_ns')>pl.col('available_ns'))).height
                assert not rows.filter((pl.col('event')=='survive') &
                                       pl.col('outcome_twd').is_not_null()).height
            if day in fact_manifest['outages']:
                assert rows.is_empty(), 'feed outage must not become an observed failure to exit'
            counts['risk_rows' if kind=='risk' else 'cost_rows']+=rows.height
        path=source/f'Date={day}/shadow/positions.parquet'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[str(path.resolve())]
        rebuilt=daily_exposures(day,pl.read_parquet(path),outage=day in fact_manifest['outages'])
        stored=pl.read_parquet(folder/'release_observations.parquet')
        same(stored.to_dicts(),rebuilt.to_dicts())
        prior.append(stored)
        counts['release_rows']+=stored.height
        counts['raw_position_files']+=1
        counts['days']+=1
    result=dict(passed=True,**counts,
        checks='All daily Q and release cells refitted before consuming today labels; raw shadow position hashes '
               'and release exposures checked; censored and outage labels checked; cumulative warmup checked.',
        limitations='Uses the declared model fitter with availability and raw-input checks; query algebra is '
                    'separately checked against the frozen uncached implementation and unit cases.')
    (root/'verification_models.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    print(json.dumps(verify(p.parse_args().root),indent=2))
