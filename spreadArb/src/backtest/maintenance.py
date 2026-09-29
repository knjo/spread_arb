"""Remove rebuildable spreadArb caches without deleting inputs or run evidence."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from ..common.paths import DATA_ROOT, SPREADARB_ROOT


def cache_targets():
    roots = [DATA_ROOT / name for name in ("qcache", "gates", "ev_surface")]
    roots += sorted((DATA_ROOT / "backtest").glob("*/prepared"))
    roots += sorted(SPREADARB_ROOT.rglob("__pycache__"))
    roots += sorted(SPREADARB_ROOT.rglob(".pytest_cache"))
    return [p for p in roots if p.exists()]


def clear_caches(receipt, apply=False):
    base = SPREADARB_ROOT.resolve()
    targets = cache_targets()
    records = []
    for p in targets:
        if p.is_symlink() or not p.resolve().is_relative_to(base):
            raise ValueError(f"refusing cache target outside the package or through a symlink: {p}")
        files = [f for f in p.rglob("*") if f.is_file()]
        records.append(dict(path=str(p), files=len(files), bytes=sum(f.stat().st_size for f in files)))
    result = dict(at=datetime.now(timezone.utc).isoformat(), applied=False,
                  targets=records, total_bytes=sum(r["bytes"] for r in records),
                  preserved="Raw market/grid inputs, official source snapshots, historical point/slippage research and run evidence.",
                  rebuild_qcache="uv run --project /home/kevin/Project/HFT --no-sync python -m spreadArb.src.ev.build --force")
    receipt = Path(receipt)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    if any(receipt.resolve().is_relative_to(p.resolve()) for p in targets):
        raise ValueError("receipt must live outside deleted cache directories")
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    if apply:
        for p in targets:
            shutil.rmtree(p)
        result["applied"] = True
        result["remaining_targets"] = [str(p) for p in targets if p.exists()]
        receipt.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    print(json.dumps(clear_caches(args.receipt, args.apply), indent=2), flush=True)


if __name__ == "__main__":
    main()
