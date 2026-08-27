"""Deterministic two-objective ranking for the frozen S1 shortlist rules."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from fractions import Fraction
from functools import cmp_to_key
from numbers import Integral
from typing import Literal

NET_TIE_TWD = Decimal("0.01")


class RankingError(ValueError):
    """Scenario metrics do not satisfy the frozen ranking contract."""


@dataclass(frozen=True)
class ScenarioMetrics:
    """Immutable sufficient statistics for one alternative S1 scenario."""

    scenario_id: str
    completion_numerator: int
    completion_denominator: int
    total_net_twd: Decimal
    reporting_sessions: int
    hedge_priced_numerator: int
    hedge_priced_denominator: int

    def __post_init__(self) -> None:
        _canonical_id(self.scenario_id)
        _validate_rate(
            self.completion_numerator,
            self.completion_denominator,
            "completion",
        )
        _validate_rate(
            self.hedge_priced_numerator,
            self.hedge_priced_denominator,
            "hedge pricing coverage",
        )
        _positive_integer(self.reporting_sessions, "reporting_sessions")
        if not isinstance(self.total_net_twd, Decimal):
            raise RankingError("total_net_twd must be Decimal")
        if not self.total_net_twd.is_finite():
            raise RankingError("total_net_twd must be finite")

    @property
    def completion_defined(self) -> bool:
        return self.completion_denominator > 0

    @property
    def completion_rate(self) -> Decimal | None:
        if not self.completion_defined:
            return None
        return _decimal_ratio(
            Decimal(self.completion_numerator),
            self.completion_denominator,
        )

    @property
    def mean_daily_net_twd(self) -> Decimal:
        return _decimal_ratio(self.total_net_twd, self.reporting_sessions)

    @property
    def hedge_pricing_coverage_defined(self) -> bool:
        return self.hedge_priced_denominator > 0

    @property
    def hedge_pricing_coverage(self) -> Decimal | None:
        if not self.hedge_pricing_coverage_defined:
            return None
        return _decimal_ratio(
            Decimal(self.hedge_priced_numerator),
            self.hedge_priced_denominator,
        )


@dataclass(frozen=True)
class S1ShortlistResult:
    completion_champion: ScenarioMetrics
    net_champion: ScenarioMetrics
    selected: tuple[ScenarioMetrics, ...]
    second_selection_source: Literal[
        "distinct_champions",
        "remaining_pareto",
        "completion_runner_up",
        "none",
    ]
    pareto_frontier: tuple[ScenarioMetrics, ...]
    completion_ranking: tuple[ScenarioMetrics, ...]
    net_ranking: tuple[ScenarioMetrics, ...]


def rank_s1_shortlist(metrics: Sequence[ScenarioMetrics]) -> S1ShortlistResult:
    """Apply the frozen completion/net champion and two-slot shortlist rules."""

    scenarios = _validated_scenarios(metrics)
    completion_order = tuple(
        sorted(scenarios, key=cmp_to_key(_compare_completion_ranking))
    )
    net_order = _rank_net_tiers(scenarios)
    frontier_ids = {item.scenario_id for item in _pareto_members(scenarios)}
    frontier = tuple(
        item for item in completion_order if item.scenario_id in frontier_ids
    )
    completion_champion = completion_order[0]
    net_champion = net_order[0]
    if completion_champion.scenario_id != net_champion.scenario_id:
        selected = (completion_champion, net_champion)
        source: Literal[
            "distinct_champions",
            "remaining_pareto",
            "completion_runner_up",
            "none",
        ] = "distinct_champions"
    else:
        remaining_frontier = tuple(
            item
            for item in frontier
            if item.scenario_id != completion_champion.scenario_id
        )
        if remaining_frontier:
            selected = (completion_champion, remaining_frontier[0])
            source = "remaining_pareto"
        elif len(completion_order) > 1:
            selected = (completion_champion, completion_order[1])
            source = "completion_runner_up"
        else:
            selected = (completion_champion,)
            source = "none"
    return S1ShortlistResult(
        completion_champion=completion_champion,
        net_champion=net_champion,
        selected=selected,
        second_selection_source=source,
        pareto_frontier=frontier,
        completion_ranking=completion_order,
        net_ranking=net_order,
    )


def pareto_frontier(
    metrics: Sequence[ScenarioMetrics],
) -> tuple[ScenarioMetrics, ...]:
    """Return the exact completion/net frontier in completion ranking order."""

    scenarios = _validated_scenarios(metrics)
    member_ids = {item.scenario_id for item in _pareto_members(scenarios)}
    return tuple(
        item
        for item in sorted(
            scenarios,
            key=cmp_to_key(_compare_completion_ranking),
        )
        if item.scenario_id in member_ids
    )


def _validated_scenarios(
    metrics: Sequence[ScenarioMetrics],
) -> tuple[ScenarioMetrics, ...]:
    scenarios = tuple(metrics)
    if not scenarios:
        raise RankingError("at least one scenario is required")
    if any(not isinstance(item, ScenarioMetrics) for item in scenarios):
        raise RankingError("all metrics must be ScenarioMetrics")
    identifiers = [item.scenario_id for item in scenarios]
    if len(set(identifiers)) != len(identifiers):
        raise RankingError("scenario_id must be unique")
    return scenarios


def _rank_net_tiers(
    metrics: Sequence[ScenarioMetrics],
) -> tuple[ScenarioMetrics, ...]:
    """Rank against each remaining tier's exact best net, avoiding fuzzy cmp."""

    remaining = list(metrics)
    ranked: list[ScenarioMetrics] = []
    while remaining:
        best = remaining[0]
        for candidate in remaining[1:]:
            if _compare_mean_net(candidate, best) > 0:
                best = candidate
        tier = [
            candidate
            for candidate in remaining
            if _within_net_tie_of_best(best, candidate)
        ]
        tier.sort(key=cmp_to_key(_compare_net_tie_break))
        ranked.extend(tier)
        tier_ids = {candidate.scenario_id for candidate in tier}
        remaining = [
            candidate
            for candidate in remaining
            if candidate.scenario_id not in tier_ids
        ]
    return tuple(ranked)


def _compare_completion_ranking(
    left: ScenarioMetrics,
    right: ScenarioMetrics,
) -> int:
    completion = _compare_rate(
        left.completion_numerator,
        left.completion_denominator,
        right.completion_numerator,
        right.completion_denominator,
    )
    if completion:
        return -completion
    net = _compare_mean_net(left, right)
    if net:
        return -net
    coverage = _compare_coverage(left, right)
    if coverage:
        return -coverage
    return _compare_id(left.scenario_id, right.scenario_id)


def _compare_net_tie_break(
    left: ScenarioMetrics,
    right: ScenarioMetrics,
) -> int:
    completion = _compare_rate(
        left.completion_numerator,
        left.completion_denominator,
        right.completion_numerator,
        right.completion_denominator,
    )
    if completion:
        return -completion
    coverage = _compare_coverage(left, right)
    if coverage:
        return -coverage
    return _compare_id(left.scenario_id, right.scenario_id)


def _compare_coverage(left: ScenarioMetrics, right: ScenarioMetrics) -> int:
    return _compare_rate(
        left.hedge_priced_numerator,
        left.hedge_priced_denominator,
        right.hedge_priced_numerator,
        right.hedge_priced_denominator,
    )


def _compare_rate(
    left_numerator: int,
    left_denominator: int,
    right_numerator: int,
    right_denominator: int,
) -> int:
    left_defined = left_denominator > 0
    right_defined = right_denominator > 0
    if left_defined and not right_defined:
        return 1
    if right_defined and not left_defined:
        return -1
    if not left_defined:
        return 0
    left_cross = left_numerator * right_denominator
    right_cross = right_numerator * left_denominator
    return (left_cross > right_cross) - (left_cross < right_cross)


def _compare_mean_net(left: ScenarioMetrics, right: ScenarioMetrics) -> int:
    left_mean = _exact_mean_net(left)
    right_mean = _exact_mean_net(right)
    return (left_mean > right_mean) - (left_mean < right_mean)


def _within_net_tie_of_best(
    best: ScenarioMetrics,
    candidate: ScenarioMetrics,
) -> bool:
    difference = _exact_mean_net(best) - _exact_mean_net(candidate)
    if difference < 0:
        raise RankingError("net tier anchor is not the exact best scenario")
    return difference <= Fraction(NET_TIE_TWD)


def _pareto_members(
    metrics: Sequence[ScenarioMetrics],
) -> tuple[ScenarioMetrics, ...]:
    return tuple(
        candidate
        for candidate in metrics
        if not any(
            challenger.scenario_id != candidate.scenario_id
            and _dominates(challenger, candidate)
            for challenger in metrics
        )
    )


def _dominates(left: ScenarioMetrics, right: ScenarioMetrics) -> bool:
    completion = _compare_rate(
        left.completion_numerator,
        left.completion_denominator,
        right.completion_numerator,
        right.completion_denominator,
    )
    net = _compare_mean_net(left, right)
    return completion >= 0 and net >= 0 and (completion > 0 or net > 0)


def _compare_id(left: str, right: str) -> int:
    return (left > right) - (left < right)


def _validate_rate(numerator: object, denominator: object, name: str) -> None:
    numerator_value = _nonnegative_integer(numerator, f"{name}_numerator")
    denominator_value = _nonnegative_integer(denominator, f"{name}_denominator")
    if numerator_value > denominator_value:
        raise RankingError(f"{name} numerator cannot exceed denominator")
    if denominator_value == 0 and numerator_value != 0:
        raise RankingError(f"undefined {name} must have a zero numerator")


def _canonical_id(value: object) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise RankingError("scenario_id must be a nonempty canonical string")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise RankingError(f"{name} must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise RankingError(f"{name} must be a nonnegative integer")
    return result


def _positive_integer(value: object, name: str) -> int:
    result = _nonnegative_integer(value, name)
    if result == 0:
        raise RankingError(f"{name} must be positive")
    return result


def _decimal_ratio(numerator: Decimal, denominator: int) -> Decimal:
    digits = len(numerator.as_tuple().digits)
    with localcontext() as context:
        context.prec = max(50, digits + len(str(denominator)) + 20)
        context.rounding = ROUND_HALF_EVEN
        return numerator / Decimal(denominator)


def _exact_mean_net(metrics: ScenarioMetrics) -> Fraction:
    return Fraction(metrics.total_net_twd) / metrics.reporting_sessions
