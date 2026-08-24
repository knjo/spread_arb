from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from maker.src.quote_fill import exit_maker_allocator
from maker.src.quote_fill.exit_maker_allocator import (
    EXIT_MAKER_ALLOCATOR_ENV_NAME,
    EXIT_MAKER_ALLOCATOR_RUNTIME_VALUE,
)
from maker.src.quote_fill import exit_maker_cli


class ExitMakerCliTest(unittest.TestCase):
    def test_allocator_launch_failure_precedes_discovery_output_and_runner(
        self,
    ) -> None:
        cases = {
            "missing": (None, None),
            "wrong": ("dirty_decay_ms:1000", "dirty_decay_ms:1000"),
            # The runtime value is assigned only after this module (and thus
            # Polars) was imported; the immutable raw snapshot must still fail.
            "late-expanded": (None, EXIT_MAKER_ALLOCATOR_RUNTIME_VALUE),
        }
        for name, (captured, current) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                output_root = Path(directory) / "must-not-exist"
                environment = (
                    {}
                    if current is None
                    else {EXIT_MAKER_ALLOCATOR_ENV_NAME: current}
                )
                discovery = Mock(
                    side_effect=AssertionError("discovery must not run")
                )
                writer = Mock(side_effect=AssertionError("output must not run"))
                runner = Mock(side_effect=AssertionError("raw IO must not run"))
                with patch.object(
                    exit_maker_allocator,
                    "_RAW_EXIT_MAKER_ALLOCATOR_ENV_VALUE",
                    captured,
                ), patch.dict(os.environ, environment, clear=True), patch.object(
                    exit_maker_cli,
                    "discover_entry_product_days",
                    discovery,
                ), patch.object(
                    exit_maker_cli,
                    "_atomic_write_csv",
                    writer,
                ), patch.object(
                    exit_maker_cli,
                    "run_exit_maker_replay",
                    runner,
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "raw process-launch allocator environment"
                    ):
                        exit_maker_cli.run(
                            sessions=("20260603",),
                            symbols=("2324",),
                            output_root=output_root,
                        )
                self.assertFalse(output_root.exists())
                discovery.assert_not_called()
                writer.assert_not_called()
                runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
