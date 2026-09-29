# EV lookup audit, 2026-09-08

This directory audits the frozen v18 source without changing strategy code or existing results.
Findings: [EV_LOOKUP_AUDIT_20260908.md](../../../doc/quote_fill/EV_LOOKUP_AUDIT_20260908.md).
Artifacts: `maker/data/ev_lookup_audit_20260908/` (ignored by Git).

Run from `/home/kevin/Project/HFT`, in order:

```bash
UV_CACHE_DIR=/tmp/codex-uv-cache uv run --no-sync python src/research/futures_spot_spread/maker/src/ev_lookup/audit/build_exit_facts.py
UV_CACHE_DIR=/tmp/codex-uv-cache uv run --no-sync python src/research/futures_spot_spread/maker/src/ev_lookup/audit/replay_allocation.py
UV_CACHE_DIR=/tmp/codex-uv-cache uv run --no-sync python src/research/futures_spot_spread/maker/src/ev_lookup/audit/probe_execution.py
UV_CACHE_DIR=/tmp/codex-uv-cache uv run --no-sync python src/research/futures_spot_spread/maker/src/ev_lookup/audit/summarize_checks.py
```

`build_exit_facts` uses the archived v15 **entry candidates only**, discards their terminal labels,
and recomputes exits with the exact frozen v18 `mexit` function, canonical daily grids and makerFill.
`replay_allocation` reproduces the original stable tie ordering, including S0 source order within S1,
and validates the result against the archived v17/v18 CSVs. A separate accepted-position event sweep
checks capacity independently of the allocation loop. Its capacity is entry spot notional per pair,
not the sum of both legs or marked portfolio exposure.

The prior-only variants repair lookup timing for a controlled sensitivity; they deliberately retain
the original candidate execution model, contract-selection errors, price assumptions and missing
pending-order lifecycle. They are not corrected performance estimates for deployment. Variants can
have different terminal inventory; the daily result remains realized-only.

Inputs must already exist: the audit source snapshot, copied reference CSVs, canonical daily grids,
SSD makerFill and NAS futures ticks. The extension-cache path is explicit in `audit_common.py`; this
session's cache was in `/tmp/claude-1000/.../scratchpad/ext_daily` and may later need rebuilding.
`probe_execution` also uses the bounded `sample_8039_20260724_ticks.parquet` saved during the audit.
Raw input paths, sizes and mtimes are inventoried; the raw files have not all been content-hashed.
Source and generated artifacts do have SHA-256 hashes.
