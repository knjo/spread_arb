from __future__ import annotations

import itertools
import unittest
from dataclasses import FrozenInstanceError
from decimal import Decimal, localcontext

from ..quote_fill.s1_ranking import (
    RankingError,
    ScenarioMetrics,
    pareto_frontier,
    rank_s1_shortlist,
)


def metrics(
    scenario_id: str,
    *,
    completion: tuple[int, int],
    net: str,
    sessions: int = 1,
    coverage: tuple[int, int] = (1, 1),
) -> ScenarioMetrics:
    return ScenarioMetrics(
        scenario_id=scenario_id,
        completion_numerator=completion[0],
        completion_denominator=completion[1],
        total_net_twd=Decimal(net),
        reporting_sessions=sessions,
        hedge_priced_numerator=coverage[0],
        hedge_priced_denominator=coverage[1],
    )


def identifiers(items: tuple[ScenarioMetrics, ...]) -> tuple[str, ...]:
    return tuple(item.scenario_id for item in items)


class S1RankingTest(unittest.TestCase):
    def test_metrics_are_immutable_and_validate_exact_types(self) -> None:
        item = metrics("valid", completion=(1, 2), net="10.25")
        with self.assertRaises(FrozenInstanceError):
            item.scenario_id = "changed"  # type: ignore[misc]
        invalid_kwargs = (
            {"completion_numerator": 2, "completion_denominator": 1},
            {"completion_numerator": 1, "completion_denominator": 0},
            {"reporting_sessions": 0},
            {"total_net_twd": 1.0},
            {"total_net_twd": Decimal("NaN")},
            {"hedge_priced_numerator": 2, "hedge_priced_denominator": 1},
        )
        base: dict[str, object] = {
            "scenario_id": "invalid",
            "completion_numerator": 1,
            "completion_denominator": 2,
            "total_net_twd": Decimal(1),
            "reporting_sessions": 1,
            "hedge_priced_numerator": 1,
            "hedge_priced_denominator": 1,
        }
        for overrides in invalid_kwargs:
            with self.assertRaises(RankingError):
                ScenarioMetrics(**(base | overrides))  # type: ignore[arg-type]

    def test_completion_uses_integer_cross_products_and_undefined_is_last(self) -> None:
        one_third = metrics("one-third", completion=(1, 3), net="10")
        huge_equal = metrics(
            "huge-equal",
            completion=(
                333_333_333_333_333_333_333_333_333_333,
                999_999_999_999_999_999_999_999_999_999,
            ),
            net="11",
        )
        undefined = metrics(
            "undefined-high-net",
            completion=(0, 0),
            net="1000000",
        )
        result = rank_s1_shortlist((undefined, one_third, huge_equal))
        self.assertEqual(
            identifiers(result.completion_ranking),
            ("huge-equal", "one-third", "undefined-high-net"),
        )
        self.assertEqual(result.completion_champion.scenario_id, "huge-equal")

    def test_completion_ties_use_net_then_coverage_then_id(self) -> None:
        high_net = metrics(
            "z-high-net",
            completion=(1, 2),
            net="11",
            coverage=(0, 0),
        )
        high_coverage = metrics(
            "b-high-coverage",
            completion=(2, 4),
            net="10",
            coverage=(9, 10),
        )
        low_id = metrics(
            "a-low-id",
            completion=(3, 6),
            net="10",
            coverage=(9, 10),
        )
        result = rank_s1_shortlist((low_id, high_coverage, high_net))
        self.assertEqual(
            identifiers(result.completion_ranking),
            ("z-high-net", "a-low-id", "b-high-coverage"),
        )

    def test_net_champion_uses_best_anchored_inclusive_one_cent_tie(self) -> None:
        exact_best = metrics(
            "exact-best",
            completion=(1, 2),
            net="100.010",
        )
        completion_wins_tie = metrics(
            "completion-wins-tie",
            completion=(9, 10),
            net="100.000",
        )
        outside = metrics(
            "outside",
            completion=(1, 1),
            net="99.999",
        )
        result = rank_s1_shortlist((outside, exact_best, completion_wins_tie))
        self.assertEqual(result.net_champion.scenario_id, "completion-wins-tie")
        self.assertEqual(
            identifiers(result.net_ranking),
            ("completion-wins-tie", "exact-best", "outside"),
        )

    def test_net_tie_uses_completion_coverage_and_stable_id_not_exact_net(self) -> None:
        candidates = (
            metrics(
                "z-exact-best",
                completion=(4, 5),
                net="100",
                coverage=(1, 10),
            ),
            metrics(
                "b-coverage",
                completion=(4, 5),
                net="99.995",
                coverage=(9, 10),
            ),
            metrics(
                "a-stable-id",
                completion=(4, 5),
                net="99.994",
                coverage=(9, 10),
            ),
        )
        result = rank_s1_shortlist(candidates)
        self.assertEqual(result.net_champion.scenario_id, "a-stable-id")
        self.assertEqual(
            identifiers(result.net_ranking),
            ("a-stable-id", "b-coverage", "z-exact-best"),
        )

    def test_mean_daily_net_uses_total_decimal_over_reporting_sessions(self) -> None:
        two_sessions = metrics(
            "two-sessions",
            completion=(9, 10),
            net="200",
            sessions=2,
        )
        exact_best = metrics(
            "exact-best",
            completion=(1, 2),
            net="100.005",
        )
        result = rank_s1_shortlist((exact_best, two_sessions))
        self.assertEqual(two_sessions.mean_daily_net_twd, Decimal(100))
        self.assertEqual(result.net_champion.scenario_id, "two-sessions")

    def test_decimal_context_does_not_change_exact_net_ranking(self) -> None:
        left = metrics(
            "left",
            completion=(1, 2),
            net="123456789012345678901234567890.0001",
            sessions=3,
        )
        right = metrics(
            "right",
            completion=(1, 2),
            net="41152263004115226300411522630.0000",
        )
        with localcontext() as context:
            context.prec = 6
            result = rank_s1_shortlist((right, left))
        self.assertEqual(result.net_champion.scenario_id, "left")

    def test_pareto_frontier_uses_only_exact_completion_and_net_objectives(
        self,
    ) -> None:
        candidates = (
            metrics("completion", completion=(9, 10), net="10"),
            metrics("net", completion=(8, 10), net="20"),
            metrics("dominated", completion=(7, 10), net="5"),
            metrics("undefined-high", completion=(0, 0), net="30"),
        )
        frontier = pareto_frontier(candidates)
        self.assertEqual(
            identifiers(frontier),
            ("completion", "net", "undefined-high"),
        )

    def test_distinct_champions_are_both_selected(self) -> None:
        completion = metrics("completion", completion=(9, 10), net="50")
        net = metrics("net", completion=(8, 10), net="100")
        result = rank_s1_shortlist((net, completion))
        self.assertEqual(result.completion_champion.scenario_id, "completion")
        self.assertEqual(result.net_champion.scenario_id, "net")
        self.assertEqual(identifiers(result.selected), ("completion", "net"))
        self.assertEqual(result.second_selection_source, "distinct_champions")

    def test_same_champion_uses_remaining_pareto_with_highest_completion(self) -> None:
        champion = metrics("champion", completion=(9, 10), net="100.000")
        higher_remaining_completion = metrics(
            "higher-remaining-completion",
            completion=(85, 100),
            net="100.004",
        )
        exact_net_best = metrics(
            "exact-net-best",
            completion=(8, 10),
            net="100.005",
        )
        result = rank_s1_shortlist(
            (exact_net_best, champion, higher_remaining_completion)
        )
        self.assertEqual(result.completion_champion.scenario_id, "champion")
        self.assertEqual(result.net_champion.scenario_id, "champion")
        self.assertEqual(
            identifiers(result.selected),
            ("champion", "higher-remaining-completion"),
        )
        self.assertEqual(result.second_selection_source, "remaining_pareto")

    def test_same_champion_falls_back_to_completion_runner_up(self) -> None:
        champion = metrics("champion", completion=(9, 10), net="100")
        runner_up = metrics("runner-up", completion=(8, 10), net="90")
        lower_completion = metrics("lower-completion", completion=(7, 10), net="95")
        result = rank_s1_shortlist((lower_completion, runner_up, champion))
        self.assertEqual(identifiers(result.pareto_frontier), ("champion",))
        self.assertEqual(
            identifiers(result.selected),
            ("champion", "runner-up"),
        )
        self.assertEqual(result.second_selection_source, "completion_runner_up")

    def test_single_scenario_returns_one_slot(self) -> None:
        only = metrics("only", completion=(0, 0), net="-1", coverage=(0, 0))
        result = rank_s1_shortlist((only,))
        self.assertEqual(identifiers(result.selected), ("only",))
        self.assertEqual(result.second_selection_source, "none")

    def test_all_ties_are_independent_of_input_order(self) -> None:
        candidates = (
            metrics("c", completion=(1, 2), net="10", coverage=(1, 2)),
            metrics("a", completion=(2, 4), net="10", coverage=(2, 4)),
            metrics("b", completion=(3, 6), net="10", coverage=(3, 6)),
        )
        outputs = {
            (
                identifiers(rank_s1_shortlist(order).completion_ranking),
                identifiers(rank_s1_shortlist(order).net_ranking),
                identifiers(rank_s1_shortlist(order).selected),
            )
            for order in itertools.permutations(candidates)
        }
        self.assertEqual(len(outputs), 1)
        only_output = outputs.pop()
        self.assertEqual(only_output[0], ("a", "b", "c"))

    def test_empty_duplicate_or_nonmetric_input_is_rejected(self) -> None:
        item = metrics("duplicate", completion=(1, 2), net="1")
        with self.assertRaises(RankingError):
            rank_s1_shortlist(())
        with self.assertRaises(RankingError):
            rank_s1_shortlist((item, item))
        with self.assertRaises(RankingError):
            rank_s1_shortlist((item, object()))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
