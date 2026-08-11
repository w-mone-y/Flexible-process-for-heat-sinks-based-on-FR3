"""Bounded, deterministic flow-optimization policies for the V2 line.

The policies in this module are deliberately independent of MuJoCo and task
actors.  They rank releases and protect shared time windows; ownership,
interlocks and physical completion remain authoritative in the runtimes.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from time import perf_counter


@dataclass(frozen=True, slots=True)
class ReleaseCandidate:
    order_id: str
    unit_id: str
    family: str
    priority: int
    urgent: bool
    inserted_at: float
    due_at: float | None = None
    estimated_flow_s: float = 30.0


@dataclass(frozen=True, slots=True)
class ReleasePlan:
    unit_ids: tuple[str, ...]
    cost: float
    explored: int
    timed_out: bool
    fallback_used: bool
    explanation_zh: str

    def as_dict(self) -> dict[str, object]:
        return {
            "unit_ids": list(self.unit_ids),
            "cost": self.cost,
            "explored": self.explored,
            "timed_out": self.timed_out,
            "fallback_used": self.fallback_used,
            "explanation_zh": self.explanation_zh,
        }


class RollingHorizonBeamPlanner:
    """Small deterministic beam search with a strict wall-time fallback."""

    def __init__(
        self,
        *,
        beam_width: int = 6,
        horizon: int = 4,
        timeout_ms: float = 4.0,
        family_change_cost: float = 8.0,
        lateness_weight: float = 3.0,
    ) -> None:
        if beam_width < 1 or horizon < 1 or timeout_ms < 0:
            raise ValueError("beam parameters must be positive")
        self.beam_width = int(beam_width)
        self.horizon = int(horizon)
        self.timeout_ms = float(timeout_ms)
        self.family_change_cost = float(family_change_cost)
        self.lateness_weight = float(lateness_weight)

    @staticmethod
    def fallback_rank(values: Iterable[ReleaseCandidate], now: float) -> tuple[ReleaseCandidate, ...]:
        return tuple(
            sorted(
                values,
                key=lambda item: (
                    -int(item.urgent),
                    float("inf") if item.due_at is None else item.due_at,
                    -item.priority,
                    item.inserted_at,
                    item.order_id,
                    item.unit_id,
                ),
            )
        )

    def plan(
        self,
        values: Iterable[ReleaseCandidate],
        *,
        now: float,
        current_family: str | None,
    ) -> ReleasePlan:
        candidates = self.fallback_rank(values, now)
        if not candidates:
            return ReleasePlan((), 0.0, 0, False, False, "当前无可释放订单")
        deadline = perf_counter() + self.timeout_ms / 1000.0
        # (cost, time_cursor, last_family, sequence, remaining)
        beam = [(0.0, float(now), current_family, (), candidates)]
        explored = 0
        timed_out = False
        for depth in range(min(self.horizon, len(candidates))):
            expanded = []
            for cost, cursor, family, sequence, remaining in beam:
                for candidate in remaining:
                    if perf_counter() > deadline:
                        timed_out = True
                        break
                    finish = cursor + max(0.01, candidate.estimated_flow_s)
                    change = 0.0 if family in {None, candidate.family} else self.family_change_cost
                    lateness = 0.0 if candidate.due_at is None else max(0.0, finish - candidate.due_at)
                    position_weight = self.horizon - depth
                    # Urgent release is a hard practical rule: it may not
                    # preempt an active operation, but it must win the next
                    # empty pallet. Position weighting prevents the credit
                    # from cancelling out when all candidates appear later in
                    # the same beam sequence.
                    urgency_credit = 1000.0 * position_weight if candidate.urgent else 0.0
                    priority_credit = 0.1 * position_weight * candidate.priority
                    age_credit = 0.01 * max(0.0, now - candidate.inserted_at)
                    next_cost = cost + finish + change + self.lateness_weight * lateness
                    next_cost -= urgency_credit + priority_credit + age_credit
                    expanded.append(
                        (
                            next_cost,
                            finish,
                            candidate.family,
                            (*sequence, candidate.unit_id),
                            tuple(item for item in remaining if item.unit_id != candidate.unit_id),
                        )
                    )
                    explored += 1
                if timed_out:
                    break
            if timed_out or not expanded:
                break
            beam = sorted(expanded, key=lambda item: (item[0], item[3]))[: self.beam_width]
        if timed_out or not beam or not beam[0][3]:
            sequence = tuple(item.unit_id for item in candidates)
            return ReleasePlan(
                sequence,
                0.0,
                explored,
                timed_out,
                True,
                "Beam Search超时，已回退为紧急度/交期/优先级动态排序",
            )
        prefix = beam[0][3]
        tail = tuple(item.unit_id for item in candidates if item.unit_id not in prefix)
        return ReleasePlan(
            (*prefix, *tail),
            float(beam[0][0]),
            explored,
            False,
            False,
            "滚动时域同时计算同族换型、交期、优先级和老化等待",
        )


class GeneticReleasePlanner:
    """Deterministic GA for V2 order-release permutations.

    The chromosome is only a permutation of already admitted V2 units.  It
    cannot create an unsafe task, bypass a furnace constraint, or change a
    physical trajectory; the V2 runtime remains the feasibility authority.
    """

    def __init__(
        self,
        *,
        population_size: int = 24,
        generations: int = 20,
        mutation_rate: float = 0.2,
        elite_count: int = 2,
        seed: int = 42,
        lateness_weight: float = 3.0,
        priority_weight: float = 0.1,
        family_change_cost: float = 8.0,
        urgent_delay_cost: float = 1000.0,
    ) -> None:
        if population_size < 2 or generations < 1 or not 1 <= elite_count < population_size:
            raise ValueError("invalid genetic planner population or generation limits")
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be between 0 and 1")
        if min(lateness_weight, priority_weight, family_change_cost, urgent_delay_cost) < 0.0:
            raise ValueError("genetic planner weights must be non-negative")
        self.population_size = int(population_size)
        self.generations = int(generations)
        self.mutation_rate = float(mutation_rate)
        self.elite_count = int(elite_count)
        self.seed = int(seed)
        self.lateness_weight = float(lateness_weight)
        self.priority_weight = float(priority_weight)
        self.family_change_cost = float(family_change_cost)
        self.urgent_delay_cost = float(urgent_delay_cost)
        self._plan_count = 0

    def reset(self) -> None:
        self._plan_count = 0

    def snapshot(self) -> dict[str, object]:
        return {
            "mode": "GENETIC_RELEASE",
            "population_size": self.population_size,
            "generations": self.generations,
            "mutation_rate": self.mutation_rate,
            "elite_count": self.elite_count,
            "seed": self.seed,
            "plan_count": self._plan_count,
        }

    @staticmethod
    def _ids(sequence: tuple[ReleaseCandidate, ...]) -> tuple[str, ...]:
        return tuple(item.unit_id for item in sequence)

    def _cost(
        self,
        sequence: tuple[ReleaseCandidate, ...],
        *,
        now: float,
        current_family: str | None,
    ) -> float:
        cursor = float(now)
        family = current_family
        total = 0.0
        remaining_weight = len(sequence)
        for index, candidate in enumerate(sequence):
            change = 0.0 if family in {None, candidate.family} else self.family_change_cost
            finish = cursor + max(0.01, candidate.estimated_flow_s) + change
            lateness = 0.0 if candidate.due_at is None else max(0.0, finish - candidate.due_at)
            urgent_delay = self.urgent_delay_cost * index if candidate.urgent else 0.0
            priority_credit = self.priority_weight * remaining_weight * candidate.priority
            total += finish + self.lateness_weight * lateness + urgent_delay - priority_credit
            cursor = finish
            family = candidate.family
            remaining_weight -= 1
        return total

    @staticmethod
    def _ordered_crossover(
        first: tuple[ReleaseCandidate, ...],
        second: tuple[ReleaseCandidate, ...],
        rng: random.Random,
    ) -> tuple[ReleaseCandidate, ...]:
        if len(first) < 2:
            return first
        left, right = sorted(rng.sample(range(len(first)), 2))
        child: list[ReleaseCandidate | None] = [None] * len(first)
        child[left : right + 1] = first[left : right + 1]
        used = {item.unit_id for item in child if item is not None}
        fill = [item for item in second if item.unit_id not in used]
        cursor = (right + 1) % len(child)
        for item in fill:
            while child[cursor] is not None:
                cursor = (cursor + 1) % len(child)
            child[cursor] = item
        return tuple(item for item in child if item is not None)

    def _mutate(
        self,
        sequence: tuple[ReleaseCandidate, ...],
        rng: random.Random,
    ) -> tuple[ReleaseCandidate, ...]:
        if len(sequence) < 2 or rng.random() >= self.mutation_rate:
            return sequence
        left, right = rng.sample(range(len(sequence)), 2)
        values = list(sequence)
        values[left], values[right] = values[right], values[left]
        return tuple(values)

    @staticmethod
    def _tournament(
        population: list[tuple[ReleaseCandidate, ...]],
        scores: dict[tuple[str, ...], float],
        rng: random.Random,
    ) -> tuple[ReleaseCandidate, ...]:
        size = min(3, len(population))
        contenders = [population[rng.randrange(len(population))] for _ in range(size)]
        return min(
            contenders,
            key=lambda item: (scores[GeneticReleasePlanner._ids(item)], GeneticReleasePlanner._ids(item)),
        )

    def plan(
        self,
        values: Iterable[ReleaseCandidate],
        *,
        now: float,
        current_family: str | None,
    ) -> ReleasePlan:
        plan_index = self._plan_count
        self._plan_count += 1
        candidates = RollingHorizonBeamPlanner.fallback_rank(values, now)
        if not candidates:
            return ReleasePlan((), 0.0, 0, False, False, "当前无可释放订单")
        if len(candidates) == 1:
            return ReleasePlan(
                (candidates[0].unit_id,),
                self._cost(candidates, now=now, current_family=current_family),
                1,
                False,
                False,
                "遗传算法候选仅有一个单元，直接释放",
            )

        rng = random.Random(self.seed + plan_index)
        population: list[tuple[ReleaseCandidate, ...]] = [candidates]
        while len(population) < self.population_size:
            candidate = list(candidates)
            rng.shuffle(candidate)
            population.append(tuple(candidate))

        scores: dict[tuple[str, ...], float] = {}

        def score(sequence: tuple[ReleaseCandidate, ...]) -> float:
            key = self._ids(sequence)
            if key not in scores:
                scores[key] = self._cost(sequence, now=now, current_family=current_family)
            return scores[key]

        for _ in range(self.generations):
            ranked = sorted(population, key=lambda item: (score(item), self._ids(item)))
            next_population = ranked[: self.elite_count]
            while len(next_population) < self.population_size:
                first = self._tournament(population, scores, rng)
                second = self._tournament(population, scores, rng)
                child = self._ordered_crossover(first, second, rng)
                next_population.append(self._mutate(child, rng))
            population = next_population

        best = min(population, key=lambda item: (score(item), self._ids(item)))
        return ReleasePlan(
            self._ids(best),
            score(best),
            len(scores),
            False,
            False,
            "遗传算法已优化V2订单释放顺序，并由物理运行时执行约束校验",
        )


@dataclass(frozen=True, slots=True)
class DynamicWipDecision:
    limit: int
    downstream_load: int
    reason_zh: str


class DynamicWipPolicy:
    def __init__(self, *, minimum: int = 3, maximum: int = 6) -> None:
        if not 1 <= minimum <= maximum:
            raise ValueError("invalid WIP limits")
        self.minimum = int(minimum)
        self.maximum = int(maximum)

    def decide(self, *, downstream_load: int, furnace_active: bool) -> DynamicWipDecision:
        if downstream_load >= 5:
            return DynamicWipDecision(self.minimum, downstream_load, "炉前/炉后拥堵，收紧上游释放")
        if downstream_load >= 3 and furnace_active:
            return DynamicWipDecision(
                min(self.maximum, self.minimum + 1), downstream_load, "炉体忙且下游负载较高"
            )
        return DynamicWipDecision(self.maximum, downstream_load, "下游有容量，允许充分并行")


def merge_time_windows(values: Iterable[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    valid = sorted(
        (float(start), float(end)) for start, end in values if isfinite(start + end) and end > start
    )
    merged: list[list[float]] = []
    for start, end in valid:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return tuple((start, end) for start, end in merged)


__all__ = [
    "DynamicWipDecision",
    "DynamicWipPolicy",
    "GeneticReleasePlanner",
    "ReleaseCandidate",
    "ReleasePlan",
    "RollingHorizonBeamPlanner",
    "merge_time_windows",
]
