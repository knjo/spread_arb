"""Bound process memory without resetting any research history or carry."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--models',required=True,type=Path)
    parser.add_argument('--configs',nargs='+')
    parser.add_argument('--study-module',default='src.research.futures_spot_spread.maker.src.ev_lookup_target.study')
    args=parser.parse_args()
    args.root.parent.mkdir(parents=True,exist_ok=True)
    log=args.root.parent/f'{args.root.name}_execution.log'
    while True:
        path=args.root/'manifest.json'
        resume=path.exists()
        if resume:
            manifest=json.loads(path.read_text())
            if manifest['status']=='completed':
                print(json.dumps(dict(status='completed',root=str(args.root))),flush=True)
                return
            if manifest['status']=='failed':
                raise RuntimeError(f'Previous session failed; inspect {log} and preserve its checkpoint before any recovery.')
        cmd=['uv','run','python','-m',args.study_module,
             str(args.root),'--models',str(args.models),'--max-days','1']
        if args.configs:
            cmd.extend(['--configs',*args.configs])
        if resume:
            cmd.append('--resume')
        with log.open('a') as handle:
            subprocess.run(cmd,stdout=handle,stderr=subprocess.STDOUT,check=True)
        manifest=json.loads(path.read_text())
        print(json.dumps(dict(day=manifest['completed_days'][-1],sessions=len(manifest['completed_days']),
                              status=manifest['status'])),flush=True)


if __name__=='__main__':
    main()
