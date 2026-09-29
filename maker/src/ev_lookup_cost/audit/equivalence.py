"""Verify the full fixed-cost control still exactly reproduces validated v22."""
import argparse
import json
from pathlib import Path

import polars as pl


def check(root: Path, previous: Path, *, completed_prefix: bool=False):
    manifest=json.loads((root/"plain/manifest.json").read_text())
    old=json.loads((previous/"manifest.json").read_text())
    if (not completed_prefix and manifest["status"]!="completed") or manifest["days"]!=old["days"]:
        raise AssertionError("full common calendar is required")
    days=(json.loads((root/"plain/checkpoint.json").read_text())["sessions"]
          if completed_prefix else manifest["days"])
    if days!=manifest["days"][:len(days)]:
        raise AssertionError("checkpoint is not a complete prefix of the original calendar")
    rows=[]
    for day in days:
        for actor, prior_actor in [("shadow","shadow"),("fixed_20M","depth5_buffer50_20M")]:
            for name in ("execution.parquet","ledger.parquet","positions.parquet","marks.parquet"):
                a=root/"plain"/f"Date={day}"/actor/name
                b=previous/f"Date={day}"/prior_actor/name
                if a.exists()!=b.exists():
                    raise AssertionError(f"output existence differs: {day} {actor} {name}")
                if not a.exists():
                    continue
                x,y=pl.read_parquet(a),pl.read_parquet(b)
                if not set(y.columns).issubset(x.columns):
                    raise AssertionError(f"control dropped source fields: {day} {actor} {name}")
                common=[c for c in y.columns if c in x.columns]
                if not x.select(common).equals(y.select(common),null_equal=True):
                    raise AssertionError(f"full control differs: {day} {actor} {name}")
                rows.append(dict(day=day,actor=actor,file=name,rows=x.height,identical=True))
    result=dict(passed=True,sessions=len(days),through=days[-1],full_horizon=len(days)==len(manifest["days"]),comparisons=len(rows),
                compared_rows=sum(r["rows"] for r in rows),details=rows)
    output="v22_control_prefix_equivalence.json" if completed_prefix else "v22_full_control_equivalence.json"
    (root/output).write_text(json.dumps(result,indent=2)+"\n")
    return {k:v for k,v in result.items() if k!="details"}


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("previous",type=Path)
    args=p.parse_args()
    print(json.dumps(check(args.root,args.previous),indent=2))
