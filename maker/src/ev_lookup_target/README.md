# Annual-target Q-table research

Status: in progress. The annual 30% objective has not been achieved or verified.
Canonical protocol: [TARGET30_Q_RESEARCH_20260911.md](../../doc/quote_fill/TARGET30_Q_RESEARCH_20260911.md).

The September 14 S2 reassessment is documented in
[S2_ENTRY_REASSESSMENT_20260914.md](../../doc/quote_fill/S2_ENTRY_REASSESSMENT_20260914.md).
Existing S2 supply is limited by a one-second initial scan, shadow-driven event replacements and a
60-second per-symbol fill cooldown. The frozen full runs retain those rules. The separate
`audit/s2_supply.py` entry-only decomposition measures independent event wakeups and waiting for the
actual previous stock hedge instead of a fixed cooldown. It has no portfolio-return interpretation.

This package preserves the completed `ev_lookup_cost` study. It adds an explicit net-return budget,
observable daily Q-table facts, separate normal/rollback/other/expiry outcomes, and survival exposures
for unresolved carry. Quote features are joined by original intent ID and quote timestamp.

`q_facts` reads the audited independent shadow one session at a time. It never projects final position
outcomes back onto prior dates. Every label has an availability timestamp and `training_window` reads
only partitions from earlier completed sessions. Outages remain in portfolio accounting but are not
labelled as observed failures to exit.

```bash
UV_CACHE_DIR=/tmp/hft-uv-cache uv run python -m unittest src.research.futures_spot_spread.maker.src.ev_lookup_target.test_q_facts
UV_CACHE_DIR=/tmp/hft-uv-cache uv run python -m src.research.futures_spot_spread.maker.src.ev_lookup_target.q_facts src/research/futures_spot_spread/maker/data/ev_lookup_cost_full_20260909/guard src/research/futures_spot_spread/maker/data/ev_lookup_target_20260911/q_facts
```

`q_model` fits conditional net value and holding time from prior 20-session risk/cost cells. `release_model`
fits surviving inventory risk sets and permits `20M + min(5M, 0.5 * discounted today releases +
0.25 * discounted next-session releases)`. This changes admission only; it never pre-releases carry.
Shared quotes do not reserve capacity collectively. Each quote needs observable room for its own ticket;
first fills book actual capacity, and subsequent cancellation-race fills remain. S2 always improves A1
by one legal tick and requires the declared depth/EV guard. Waiting orders receive fresh queue timestamps.

The four frozen comparisons are net-Q reserved 20M, net-Q shared 20M, net-Q shared release credit up to
25M, and a 30%-annual holding hurdle with the same credit. All use a 2M per-ticket admission bound and
20 observed prior sessions before trading. The 25M number is an admission ceiling, not a guaranteed
maximum during a cancellation race. Report all actual extra-capital requirements.

Three additional `audit/unrestricted_study.py` comparisons remove the discretionary 2M ticket filter:
shared 20M, release-credit 25M, and the holding hurdle with release credit. They still check each quote
against the observable aggregate admission limit. Outputs live under `ticket_unrestricted/`.

The `audit/deep_study.py` comparison implements the user's S1 queue-retention request. An S1 buy strictly
below observable B1 may be sent and kept without individual room. If it approaches B1 without room,
request a 50ms cancellation; if capacity releases first, keep its original queue timestamp. Direct jumps
and cancellation races still fill and hedge. Actual committed capacity above 25M triggers cancellation
of all other unreserved entries and pauses new deep requests. S2 retains its ordinary room/priority gate.
This explicit exception to the base admission rule is recorded in the full run's manifest. Outputs live
under `deep_spot/`; its eight execution tests and two-day execution/Q/capacity audits passed.

The final declared comparison, `audit/entry_window_study.py`, applies the same deep-S1 execution and
unrestricted ticket policy, but charges the 30% opportunity hurdle against expected occupied entry
windows: 12bp per full 14,000-second window. Closed-market hours still incur calendar funding. The
duration comes from prior-only competing risks, never an actual future exit time. This alternative
clock does not guarantee enough utilization or profit to reach the annual budget. Its three clock
tests passed; outputs live under `entry_window/`.

The current input audit covers 86 dates; the execution smoke covers July 23-24 and verifies restoration
from an end-of-day checkpoint. The previous cost control also reproduces its original two-day outputs.
Full-period strategy performance and the 30% target comparison remain pending.

Continuous replay (fresh process per day, preserved carry and prior history):

```bash
UV_CACHE_DIR=/tmp/hft-uv-cache uv run python -m src.research.futures_spot_spread.maker.src.ev_lookup_target.audit.run_daily src/research/futures_spot_spread/maker/data/ev_lookup_target_20260911/full --models src/research/futures_spot_spread/maker/data/ev_lookup_target_20260911/model_inputs
```

`data/ev_lookup_target_20260911/work_status.json` records authoritative artifacts and running handles.
Do not change root-level package Python files while a full replay is active: source/input hashes are
checked before every checkpoint restore. `audit/unrestricted_study.py`, `audit/deep_study.py` and
`audit/deep_policy.py`, `audit/entry_window.py` and `audit/entry_window_study.py` are also pinned by their
respective smoke/full runs. Other audit modules can be developed
independently. Numeric reports include actual stock cash-time funding, post-warmup and calendar returns,
all original shadow-column equivalence, cancellation overruns and message/entry-latency diagnostics.
`audit/compare.py` combines all nine declared policies with the original cost control after all four full
reports finish; incomplete branches cannot produce a combined headline.
