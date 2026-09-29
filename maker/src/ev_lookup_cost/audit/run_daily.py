"""Run one committed session per process without changing policy or calendar."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--mode", required=True, choices=["plain","guard"])
    args = parser.parse_args()
    args.root.parent.mkdir(parents=True, exist_ok=True)
    log = args.root.parent/f"{args.mode}_execution.log"
    controller = args.root.parent/f"{args.mode}_controller.json"
    controller.write_text(json.dumps(dict(source=str(Path(__file__).resolve()),
        sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), mode=args.mode,
        execution="One session per fresh process; full manifest calendar and exact checkpoint retained."),indent=2)+"\n")
    while True:
        resume = (args.root/"manifest.json").exists()
        if resume:
            manifest = json.loads((args.root/"manifest.json").read_text())
            if manifest["status"] == "completed":
                print(json.dumps(dict(mode=args.mode,status="completed")),flush=True)
                return
        cmd = ["uv","run","python","-m","src.research.futures_spot_spread.maker.src.ev_lookup_cost.cost_study",
               str(args.root),"--mode",args.mode,"--max-days","1"]
        if resume:
            cmd.append("--resume")
        with log.open("a") as handle:
            result = subprocess.run(cmd,stdout=handle,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Session failed ({result.returncode}); preserved files and checkpoint. See {log}")
        manifest = json.loads((args.root/"manifest.json").read_text())
        print(json.dumps(dict(mode=args.mode,day=manifest["completed_days"][-1],
            sessions=len(manifest["completed_days"]),status=manifest["status"])),flush=True)


if __name__ == "__main__":
    main()
