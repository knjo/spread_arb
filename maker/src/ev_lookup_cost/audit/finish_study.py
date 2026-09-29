"""Await both full replays, then run independent audits and reporting once."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


PREFIX = "src.research.futures_spot_spread.maker.src.ev_lookup_cost"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    args=parser.parse_args()
    status=args.root/"finalization_status.json"
    last=None
    while True:
        states=[]
        for mode in ("plain","guard"):
            path=args.root/mode/"manifest.json"
            m=json.loads(path.read_text()) if path.exists() else {}
            states.append(dict(mode=mode,status=m.get("status","starting"),sessions=len(m.get("completed_days",[]))))
        if any(s["status"]=="failed" for s in states):
            raise RuntimeError("A replay failed. Keep checkpoint intact and inspect the execution log.")
        if all(s["status"]=="completed" for s in states):
            break
        if states!=last:
            status.write_text(json.dumps(dict(stage="replay",modes=states),indent=2)+"\n")
            print(json.dumps(dict(stage="replay",modes=states)),flush=True)
            last=states
        time.sleep(30)
    status.write_text(json.dumps(dict(stage="independent_audit"))+"\n")
    def run_audit(mode):
        log=args.root/f"{mode}_audit.log"
        with log.open("w") as handle:
            cmd=["uv","run","python","-m",PREFIX+".audit.verify_cost",str(args.root/mode),
                 "--raw-days","20260511","20260706","20260724","20260902"]
            subprocess.run(cmd,stdout=handle,stderr=subprocess.STDOUT,check=True)
        print(json.dumps(dict(stage="audit_completed",mode=mode)),flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run_audit,("plain","guard")))
    status.write_text(json.dumps(dict(stage="reporting"))+"\n")
    with (args.root/"report.log").open("w") as handle:
        subprocess.run(["uv","run","python","-m",PREFIX+".audit.report_cost",str(args.root)],
                       stdout=handle,stderr=subprocess.STDOUT,check=True)
    audit_dir=args.root/"audit_source_snapshot"
    audit_dir.mkdir(exist_ok=True)
    sources={}
    package=Path(__file__).resolve().parents[1]
    paths=list((package/"audit").glob("*.py"))+[
        package/"analyze_full_study.py",package/"verify_full_study.py",package/"verify_run.py",
        package.parent/"ev_lookup/audit/execution_costs.py"]
    for path in paths:
        shutil.copy2(path,audit_dir/path.name)
        sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    (args.root/"audit_sources.json").write_text(json.dumps(sources,indent=2)+"\n")
    status.write_text(json.dumps(dict(stage="completed",human_report_review_pending=True),indent=2)+"\n")
    print("Replays, independent audits, and data reports completed. Canonical written conclusion still needs review.",flush=True)


if __name__ == "__main__":
    main()
