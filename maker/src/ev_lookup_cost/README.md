# First-priority execution-cost research

Canonical review: [EV_EXECUTION_COST_REVIEW_20260909.md](../../doc/quote_fill/EV_EXECUTION_COST_REVIEW_20260909.md).
This independent package freezes XC sources in `XC_BASE_SOURCES.json`, then integrates the validated v22
inside-only queue rules with causal S1/S2 entry and exit cost tables. Existing XC and v22 runs are untouched.

The full 86-session / 85-data-day study in `maker/data/ev_lookup_cost_full_20260909` is complete and independently
audited. S1+S2 daily PnL including final official inventory marks, before funding, is TWD 3,919.50 / 5,117.54 /
8,741.72 for fixed / cost / cost+guard. The last becomes TWD 7,938.90 under a hypothetical 2% funding rate.
The control matches v22 across all 684 compared files. Consult the canonical review for EV calibration,
selection by month, execution tails and the exploratory (not untouched validation) status.

Use `cost_study.py` or the bounded-memory `audit.run_daily` controller. The copied older experiment entry points
are historical dependencies, not the supported configuration of this study.

```bash
UV_CACHE_DIR=/tmp/hft-uv-cache uv run python -m src.research.futures_spot_spread.maker.src.ev_lookup_cost.audit.run_daily src/research/futures_spot_spread/maker/data/ev_lookup_cost_full_20260909/plain --mode plain
UV_CACHE_DIR=/tmp/hft-uv-cache uv run python -m src.research.futures_spot_spread.maker.src.ev_lookup_cost.audit.run_daily src/research/futures_spot_spread/maker/data/ev_lookup_cost_full_20260909/guard --mode guard
UV_CACHE_DIR=/tmp/hft-uv-cache uv run python -m src.research.futures_spot_spread.maker.src.ev_lookup_cost.audit.verify_cost src/research/futures_spot_spread/maker/data/ev_lookup_cost_full_20260909/plain --raw-days 20260511 20260706 20260724 20260902
```

Run from the HFT repository root. Never run two controllers against the same output.
Resume requires identical mode, parameters, full calendar, products, metadata and core sources.
The daily process limit does not truncate the research horizon or reset carry/lookup history.
Full completion means the manifest is completed and all independent audits/reporting also pass.

Costs are query-time estimates. They never rewrite fill prices or erase losing executions.
The empirical entry estimate and fixed tick scenario use a maximum, avoiding duplicate entry charges.
Exit shortfall is normalized to entry spot cash so it reconciles exactly to cash PnL.

Known limitations are explicit in the review, including historical prior selection, partial-rollbacks in P_sd,
missing daily official marks, immediate new queue admission, unthrottled traffic and the expiry accounting convention.
