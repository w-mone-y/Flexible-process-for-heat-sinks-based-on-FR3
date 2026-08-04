"""Bounded, deterministic flow-optimization policies for the V2 line.

The policies in this module are deliberately independent of MuJoCo and task
actors.  They rank releases and protect shared time windows; ownership,
interlocks and physical completion remain authoritative in the runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from time import perf_counter
from typing import Iterable


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
    "ReleaseCandidate",
    "ReleasePlan",
    "RollingHorizonBeamPlanner",
    "merge_time_windows",
]
