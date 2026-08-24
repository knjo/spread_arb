# 13:00 dynamic B1/A1 exit challenger

Status: analysis-only implementation; no formal 60-day result has been
published.  The existing same-day, cross-session, and post-cross bundles are
unchanged.

## Frozen challenger semantics

At 13:00 Taipei session time, a position which is proven still open starts two
passive exit routes:

- buy one future at B1, then sell two spot lots by taker at full fill +50 ms;
- sell two spot lots at A1, then buy one future by taker at full fill +50 ms.

Both pegs are dynamic.  An unchanged absolute price keeps its existing order
and displayed queue.  A move in the more-aggressive direction adds a new layer
while retaining older layers.  A retreat cancels layers ahead of the new
target.  The implementation reuses `LayeredSampler`, indexed trade replay,
the established exit-maker route contracts, and the existing +50 ms hedge
book lookup.

All generations from both routes enter one cross-route earliest-full-fill OCO
projection.  The nominal analysis assumes instantaneous sibling cancellation;
the strict sensitivity remains `cancel_race_unknown` when the tape cannot prove
the cancellation.  At most one nominal close is emitted per physical position.

## Inventory at 13:00

The inventory selector consumes exactly one already-selected normal policy path
per `entry_raw_order_fact_id`.  It does not treat entry aliases as positions.
For target date `D` it:

- rejects new `D` entries established at or after 13:00;
- excludes a priced normal terminal on `D` whose decision time is at or before
  13:00;
- excludes every terminal before `D`;
- retains entries from prior sessions that remain observed/open on `D`;
- retains products with carried inventory even when that product has no new
  `D` entry; and
- fails closed on duplicate physical paths, incoherent terminal/censor fields,
  imputed entry rows, or an unpriced path whose observation ended before `D`.

This is the important distinction between “entered before 13:00” and “still
open at 13:00.”  The entry-action helper alone is not sufficient evidence of
open inventory.

## Batch/cache execution

`replay_aggressive_1300_inventory_batch` builds one market replay template per
product-day and projects it to all unique physical positions.  A process-local
`Aggressive1300ProductDayReplayCache` reuses that template on repeated calls,
so the raw book/trade indexes are not rebuilt per position.

The caller supplies an immutable `source_cache_key`; this core API does not yet
verify a formal raw-source manifest.  Multiple position rows share the same
market-capacity observation, so every output explicitly has
`joint_volume_allocated=false`, `position_outcomes_safe_to_sum=false`, and
`formal_ev_ready=false`.  These rows cannot be summed into a portfolio fill,
P&L, or production readiness claim.

## Bounded validation

Synthetic tests cover dynamic forward/retreat layering on both routes, future
and spot winners, exact +50 ms hedges, nominal versus strict cancellation,
no duplicate close, exact 13:00 entry exclusion, normal exits before/after
13:00, prior-session carry, stale unresolved paths, physical deduplication, and
one-template cache reuse.

A read-only selector smoke on the existing 3,672-row selected-path diagnostic
found 29 physical positions open at 13:00 on 2026-07-14 (22 target-day and 7
carried) and 15 on 2026-08-06 (11 target-day and 4 carried).  These are inventory
eligibility counts only, not exit success or EV results.

Implementation and tests:

- `maker/src/quote_fill/aggressive_1300_exit.py`
- `maker/src/tests/test_quote_fill_aggressive_1300_exit.py`

