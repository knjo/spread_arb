"""Process-launch allocator contract for the large exit-maker replay.

This module is intentionally stdlib-only.  The formal CLI and runner import
it before Polars so the value below is a snapshot of the raw process-launch
environment, not the value that Polars expands after its allocator loads.
"""

from __future__ import annotations

import os


EXIT_MAKER_ALLOCATOR_ENV_NAME = "_RJEM_MALLOC_CONF"
EXIT_MAKER_ALLOCATOR_ENV_VALUE = (
    "background_thread:true,dirty_decay_ms:100,muzzy_decay_ms:100"
)
EXIT_MAKER_ALLOCATOR_RUNTIME_VALUE = (
    "dirty_decay_ms:500,muzzy_decay_ms:-1," + EXIT_MAKER_ALLOCATOR_ENV_VALUE
)

# Never refresh this snapshot.  Mutating the environment after importing a
# Polars-dependent module must not be able to satisfy the launch contract.
_RAW_EXIT_MAKER_ALLOCATOR_ENV_VALUE = os.environ.get(
    EXIT_MAKER_ALLOCATOR_ENV_NAME
)


def validate_exit_maker_allocator_launch() -> None:
    """Require the exact allocator setting captured before Polars import."""

    if _RAW_EXIT_MAKER_ALLOCATOR_ENV_VALUE != EXIT_MAKER_ALLOCATOR_ENV_VALUE:
        raise RuntimeError(
            "exit-maker requires the raw process-launch allocator environment "
            f"{EXIT_MAKER_ALLOCATOR_ENV_NAME}="
            f"{EXIT_MAKER_ALLOCATOR_ENV_VALUE!r} before importing Polars; "
            f"captured {_RAW_EXIT_MAKER_ALLOCATOR_ENV_VALUE!r}"
        )
