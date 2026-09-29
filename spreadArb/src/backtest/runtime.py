"""Run provenance, resume checks and bounded daily memory cleanup."""
from dataclasses import asdict
import ctypes
import gc
import hashlib
import json
from pathlib import Path
import resource

from ..common.paths import grid_days, hist_path, facts_path


def require_qcache(days):
    """Fail on an incomplete prior-session cache instead of fitting a subset."""
    needed = [d for d in grid_days() if d < max(days)] if days else []
    paths = [p for d in needed for p in (hist_path(d), facts_path(d))]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise ValueError(f"incomplete Q cache; run spreadArb.src.ev.build first: {missing[:5]}")
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def execution_sources():
    paths = [p for folder in ("common", "ev", "points")
             for p in Path("spreadArb/src", folder).glob("*.py")]
    paths += [Path("spreadArb/src/backtest", name + ".py") for name in
              ("causal_replay", "causal_market", "capital", "contracts", "policy", "gates", "runtime", "replay")]
    paths += [Path("maker/src/ev_lookup_cost/execution.py")]
    paths += list(Path("maker/src/common").glob("*.py"))
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def init_run(out, days, actors, resume):
    if not days:
        raise ValueError("no canonical grid sessions in range")
    configs = json.loads(json.dumps({a.name: asdict(a.cfg) for a in actors}))
    sources = execution_sources()
    manifest_path = out / "manifest.json"
    if resume:
        previous = json.loads(manifest_path.read_text())
        if previous.get("configs") != configs or previous.get("days") != days:
            raise ValueError("resume requires the original complete date range and policy")
        if previous.get("execution_sources") != sources:
            raise ValueError("execution source changed; use a new run, not a mixed-source resume")
        return
    if out.exists() and any(out.iterdir()):
        raise ValueError("output exists; use a fresh name or --resume")
    out.mkdir(parents=True, exist_ok=True)
    all_sources = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in Path("spreadArb/src").rglob("*.py")}
    all_sources.update(sources)
    manifest_path.write_text(json.dumps(dict(days=days, sources=all_sources,
        execution_sources=sources, configs=configs,
        execution="raw FIFO/shared depth; 50ms; post-only; one order per product/route",
        capital="actual partial/full fills; no quote reservation unless explicitly enabled",
        settlement="expiry-day basis-zero accounting; official spot close, explicit BBO fallback",
        costs="20/34bp original convention; rollback actual cash; no financing",
        annual="simple 250 / observed sessions / configured capital"), indent=2)+"\n")


def assert_sources(out):
    manifest = json.loads((out / "manifest.json").read_text())
    if execution_sources() != manifest["execution_sources"]:
        raise RuntimeError("execution source changed during replay; checkpoint retained")


def memory_usage():
    current = None
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            current = int(line.split()[1]) / 1024**2
    return dict(rss_gib=current, peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)


def release_memory():
    gc.collect()
    libc = ctypes.CDLL(None)
    if hasattr(libc, "malloc_trim"):
        libc.malloc_trim(0)
    return memory_usage()
