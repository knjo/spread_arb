"""Reuse only the provably unaffected prefix after the forecast-calendar fix.

Reconstruct each actor's ledger/counter/positions from immutable day artifacts,
first at the saved checkpoint itself to prove exact restoration. Historical
observations are filtered by their actual observation day, preserving order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

import polars as pl

from ..forecast_calendar import calendar_spec
from ..verify_full_study import read_rows


def assert_same(a, b, path="state"):
    if isinstance(a, dict):
        assert set(a) == set(b), path
        for k in a:
            assert_same(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, f"{path}[{i}]")
    elif isinstance(a, float):
        assert math.isclose(a, b, rel_tol=1e-13, abs_tol=1e-10), (path, a, b)
    else:
        assert a == b, (path, a, b)


def reconstruct(root: Path, saved: dict, through: str) -> dict:
    sessions = [d for d in saved["sessions"] if d <= through]
    assert sessions[-1] == through
    state = dict(sessions=sessions)
    for key, date_key in [("resolutions", "resolved_day"), ("entry_days", "entry_day"),
                          ("capacity_observations", "day"), ("exit_risks", "day")]:
        state[key] = [r for r in saved[key] if r[date_key] <= through]
    actors = []
    for actor in saved["actors"]:
        name = actor["name"]
        amounts, committed, peak, last_ns, counter = {}, 0, 0, 0, 0
        for day in sessions:
            folder = root / f"Date={day}" / name
            for r in read_rows(folder / "ledger.parquet"):
                if r["kind"] == "release":
                    assert amounts.pop(r["id"]) == -r["delta_cents"]
                else:
                    amounts[r["id"]] = amounts.get(r["id"], 0) + r["delta_cents"]
                committed += r["delta_cents"]
                assert committed == r["committed_cents"] == sum(amounts.values())
                assert r["ns"] >= last_ns
                last_ns = r["ns"]
                peak = max(peak, committed)
            trace = folder / "execution.parquet"
            if trace.exists():
                counter += pl.read_parquet(trace, columns=["kind"]).filter(pl.col("kind") == "exit_quote").height
        positions = [p for p in read_rows(root / f"Date={through}" / name / "positions.parquet")
                     if p["state"] not in {"closed", "cancelled"}]
        assert {p["id"] for p in positions} == set(amounts)
        actors.append(dict(name=name, positions=positions, amounts=amounts,
                           committed_cents=committed, peak_cents=peak, last_ns=last_ns,
                           counter=counter, daily=[r for r in actor["daily"] if r["day"] <= through],
                           rejected_by_day={d: v for d, v in actor["rejected_by_day"].items() if d <= through}))
    state["actors"] = actors
    return state


def bootstrap(source: Path, target: Path, through: str):
    old = json.loads((source / "manifest.json").read_text())
    assert all(not c.get("park_spot", False) for c in old["configurations"])
    saved = json.loads((source / "checkpoint.json").read_text())
    # Counter, position ordering, ledger, and exact history must agree at an
    # independently serialized checkpoint before using the earlier cut.
    assert_same(reconstruct(source, saved, saved["sessions"][-1]), saved)
    state = reconstruct(source, saved, through)
    decisions = positions = 0
    for day in state["sessions"]:
        for actor in state["actors"]:
            folder = source / f"Date={day}" / actor["name"]
            for r in read_rows(folder / "decisions.parquet"):
                assert not day < "20260710" <= r["expiry"], "prefix EV could see changed closure"
                decisions += 1
            for p in read_rows(folder / "positions.parquet"):
                assert not day < "20260710" <= p["contract"]["expiry"], "held-order EV could see changed closure"
                positions += 1
    target.mkdir(parents=True, exist_ok=False)
    for day in state["sessions"]:
        (target / f"Date={day}").symlink_to((source / f"Date={day}").resolve(), target_is_directory=True)
    (target / "checkpoint.json").write_text(json.dumps(state) + "\n")
    snaps = [s for s in json.loads((source / "snapshots.json").read_text()) if s["day"] <= through]
    (target / "snapshots.json").write_text(json.dumps(snaps, indent=2) + "\n")
    for a in state["actors"]:
        pl.from_dicts(a["daily"]).write_csv(target / f'{a["name"]}_daily.csv')
    shutil.copytree(source / "input_snapshot", target / "input_snapshot")
    (target / "source_snapshot").mkdir()
    code = Path(__file__).resolve().parents[1]
    sources = {}
    for p in code.glob("*.py"):
        shutil.copy2(p, target / "source_snapshot" / p.name)
        sources[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    shutil.copy2(source / "manifest.json", target / "prefix_source_manifest.json")
    manifest = {k: v for k, v in old.items() if k not in
                {"superseded_reason", "calendar_fallback", "peak_committed_twd"}}
    manifest.update(status="running", sources=sources, forecast_calendar=calendar_spec(),
                    reused_prefix=dict(source=str(source.resolve()), through=through,
                        sessions=len(state["sessions"]), decisions_checked=decisions,
                        positions_checked=positions, checkpoint_roundtrip="passed",
                        roundtrip_day=saved["sessions"][-1],
                        rule="Only forecast calendar changes; no prefix entry or held contract reaches changed July 10 closure",
                        bootstrap_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(target=str(target), **manifest["reused_prefix"])), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("target", type=Path)
    p.add_argument("--through", default="20260617")
    a = p.parse_args()
    bootstrap(a.source, a.target, a.through)
