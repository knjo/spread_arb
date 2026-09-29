"""The selected strategy screens at zero and exits absolute entries at max-Q."""
from dataclasses import replace
import unittest
from unittest.mock import patch

from ..backtest import policy
from ..backtest.replay import parse_args
from ..ev.ev import ExitEval
from .test_causal_replay import row


class SelectedPolicyTest(unittest.TestCase):
    def test_positive_q_target_does_not_replace_zero_entry_score(self):
        absolute = ExitEval(0.0, 0.0, .5, .4, .1, 30, 1, 30, 20, 10, 3, route="absolute")
        residual = replace(absolute, route="residual", x=-.5, b_x_bp=40., ev_bp=4., score=4.)
        candidate = {**row(), "anchor": 50.}
        for name in ("A", "B"):
            with self.subTest(preset=name), patch.object(policy, "evaluate", return_value=[residual, absolute]):
                cfg = policy.PolicyConfig(**policy.PRESETS[name])
                decision = policy.decide_row(candidate, "20260706", None, None, cfg)
                self.assertTrue(decision.admit)
                self.assertEqual(decision.best.route, "absolute")
                self.assertAlmostEqual(decision.best.score, 30.)
                self.assertAlmostEqual(decision.best.b_x_bp, 0.)
                self.assertAlmostEqual(policy.exit_target_bp(decision.best, 50., 20., decision.evals, cfg.abs_target_mode), 40.)

    def test_default_cli_selects_dynamic_b_with_user_capacity_rules(self):
        options, cfg = parse_args([])
        self.assertEqual(options["preset"], "B")
        self.assertEqual(cfg.abs_target_mode, "max_q")
        self.assertAlmostEqual(cfg.dyn_q, .5)
        self.assertAlmostEqual(cfg.dyn_cap_frac, .8)
        self.assertFalse(cfg.reserve_on_submit)
        self.assertIsNone(cfg.max_positions_per_product)
        self.assertEqual(cfg.streams, ("S1", "S2"))
        self.assertEqual(cfg.exit_routes, ("E1", "E2"))


if __name__ == "__main__":
    unittest.main()
