# spreadArb source

2026-09-23: the user clarified that unfilled orders do not reserve position
capacity. The 9/22 full-reservation run is a different policy, even though its
internal audit passed. See [capacity clarification](../doc/05_CAPITAL_POLICY_CLARIFICATION.md)
before interpreting its returns or using the current reserve/no-reserve options.

The 9/23 revalidation found missing partial-fill capital and capacity-triggered
cancels in the archived no-reserve version. Working code now fixes those issues,
and `causal_audit` independently reconstructs actual exposure and route limits.
The complete 131-session `accepted_20260923` run now passes independent accounting,
raw execution checks, submitted EV recomputation and 125 tests. A/B net PnL is
TWD 1,389,249.40 / 1,434,224.26; fixed-capital simple annualization is
13.2562% / 13.6853% over 2026-01-26 through 2026-08-13. Current rules and evidence:
[accepted-policy implementation](../doc/08_ACCEPTED_POLICY_IMPLEMENTATION.md).

Latest quote-count clarification: S1, S2, E1 and E2 may all work concurrently
for a product, with one order per route. Held positions have no count limit in
either A or B. Entry hedge/rollback guards apply within the same stream; exit
routes serve the oldest eligible inventory. See
[quote concurrency](../doc/07_QUOTE_CONCURRENCY_CLARIFICATION.md).

Run from `src/research/futures_spot_spread/` with the HFT uv environment. Current architecture and assumptions:
[04_CORRECTED_REPLAY.md](../doc/04_CORRECTED_REPLAY.md).

```bash
cd /home/kevin/Project/HFT/src/research/futures_spot_spread
export UV_CACHE_DIR=/tmp/hft-uv-cache
export POLARS_MAX_THREADS=2
RUN="uv run --project /home/kevin/Project/HFT --no-sync python"

$RUN -m unittest discover -s spreadArb/src/tests -t .
$RUN -m spreadArb.src.backtest.causal_replay --out my_verified_AB --prefetch 0
# Resume instead of starting again; preserve the original source/config/date range.
$RUN -m spreadArb.src.backtest.causal_replay --out my_verified_AB --prefetch 0 --resume
$RUN -m spreadArb.src.backtest.causal_audit --run my_verified_AB --raw-all
$RUN -m spreadArb.src.backtest.ev_validation --run my_verified_AB
$RUN -m spreadArb.src.backtest.accepted_report --run my_verified_AB

# One policy / a short test: choose a fresh output directory.
$RUN -m spreadArb.src.backtest.replay --preset A --start 20260706 --end 20260706 --out my_smoke_A

# Exogenous Q caches and surfaces remain separate from portfolio execution.
$RUN -m spreadArb.src.ev.build --start 20260126 --end 20260813
$RUN -m spreadArb.src.ev.surface --day 20260706 --window 20
```

`causal_market.py` loads the canonical mapping/1 Hz anchors and raw receive-time books/prints.
`policy.py` evaluates exact quote states using as-of tables. `causal_replay.py` runs A and B chronologically
with independent shared-volume queues and capital/depth ledgers. `valuation.py` computes official daily
marks after execution; `official_marks.py` fills missing database sessions from the public exchange CSV.

Outputs: `data/backtest/<run>/manifest.json`, daily checkpoints, `A_daily.csv` / `B_daily.csv` (execution),
`A_daily_official.csv` / `B_daily_official.csv` (official equity), and `Date=<day>/<A|B>/` position,
execution, cash, capital and depth files. `verification.json` must say `complete=true` and `status=PASS`
before quoting the final returns; `ACCEPTED_RESULTS.md` reports the verified policy,
raw queue checks and mature-cohort EV diagnostics. The historical full-reservation
`COMPARISON.md` belongs to its archived run, not to this policy.

`legacy_replay.py` is the retired point-label engine. `points/` retains its historical independent labels
for research, but current portfolio replay does not read precomputed future fill/hedge times.
`audit.py`, `logic_audit.py`, and `raw_audit.py` retain the original point-replay evidence;
use `causal_audit.py` and `test_causal_replay.py` for the corrected execution.
