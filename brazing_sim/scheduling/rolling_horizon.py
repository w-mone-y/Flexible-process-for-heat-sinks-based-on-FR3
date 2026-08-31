"""Bounded deterministic lookahead for safe manufacturing task boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Iterable, Mapping


@dataclass(frozen=True, slots=True)
class HorizonAction:
    """One already-feasible action considered without mutating runtime state."""

    action_id: str
    action_zh: str
    duration_s: float
    projected_completion_s: float
    blocks_resource: str | None = None
    downstream_blocking_s: float = 0.0
    changeover_s: float = 0.0
    due_date_penalty_s: float = 0.0

    def __post_init__(self) -> None:
        if not self.action_id or not self.action_zh:
            raise ValueError("horizon action must have an id and label")
        values = (
            self.duration_s,
            self.projected_completion_s,
            self.downstream_blocking_s,
            self.changeover_s,
            self.due_date_penalty_s,
        )
        if any(not isfinite(value) or value < 0.0 for value in values):
            raise ValueError("horizon action estimates must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class HorizonDecisionContext:
    now: float
    critical_resource: str | None = None
    critical_work_ready_in_s: float = float("inf")
    critical_work_duration_s: float = 0.0
    critical_path_weight: float = 1.0
    blocking_weight: float = 1.0
    changeover_weight: float = 1.0
    due_date_weight: float = 1.0

    def __post_init__(self) -> None:
        finite_values = (
            self.now,
            self.critical_work_duration_s,
            self.critical_path_weight,
            self.blocking_weight,
            self.changeover_weight,
            self.due_date_weight,
        )
        if any(not isfinite(value) or value < 0.0 for value in finite_values):
            raise ValueError("horizon context estimates must be finite and non-negative")
        if self.critical_work_ready_in_s < 0.0:
            raise ValueError("critical work release must be non-negative")


@dataclass(frozen=True, slots=True)
class HorizonCandidate:
    action_id: str
    action_zh: str
    projected_completion_s: float
    critical_path_delay_s: float
    downstream_blocking_s: float
    changeover_s: float
    due_date_penalty_s: float
    total_cost: float

    def as_dict(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "action_zh": self.action_zh,
            "projected_completion_s": round(self.projected_completion_s, 6),
            "critical_path_delay_s": round(self.critical_path_delay_s, 6),
            "downstream_blocking_s": round(self.downstream_blocking_s, 6),
            "changeover_s": round(self.changeover_s, 6),
            "due_date_penalty_s": round(self.due_date_penalty_s, 6),
            "total_cost": round(self.total_cost, 6),
        }


@dataclass(frozen=True, slots=True)
class HorizonDecision:
    selected_action_id: str
    horizon_seconds: float
    candidates: tuple[HorizonCandidate, ...]
    explanation_zh: str

    def as_dict(self) -> dict[str, object]:
        return {
            "selected_action_id": self.selected_action_id,
            "horizon_seconds": self.horizon_seconds,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "explanation_zh": self.explanation_zh,
        }


class RollingHorizonPlanner:
    """Score a small feasible action set over a fixed simulated-time horizon."""

    def __init__(self, *, horizon_seconds: float = 45.0, maximum_candidates: int = 8) -> None:
        if not isfinite(horizon_seconds) or horizon_seconds <= 0.0:
            raise ValueError("horizon must be finite and positive")
        if maximum_candidates <= 0:
            raise ValueError("maximum candidates must be positive")
        self.horizon_seconds = float(horizon_seconds)
        self.maximum_candidates = int(maximum_candidates)
        self._last_decision: HorizonDecision | None = None

    def _score(
        self,
        action: HorizonAction,
        context: HorizonDecisionContext,
    ) -> HorizonCandidate:
        critical_delay = 0.0
        critical_release = context.now + context.critical_work_ready_in_s
        action_finish = context.now + action.duration_s
        if (
            action.blocks_resource is not None
            and action.blocks_resource == context.critical_resource
            and context.critical_work_ready_in_s <= self.horizon_seconds
        ):
            critical_delay = max(0.0, action_finish - critical_release)
        projected = action.projected_completion_s
        total = (
            projected
            + critical_delay * context.critical_path_weight
            + action.downstream_blocking_s * context.blocking_weight
            + action.changeover_s * context.changeover_weight
            + action.due_date_penalty_s * context.due_date_weight
        )
        return HorizonCandidate(
            action_id=action.action_id,
            action_zh=action.action_zh,
            projected_completion_s=projected,
            critical_path_delay_s=critical_delay,
            downstream_blocking_s=action.downstream_blocking_s,
            changeover_s=action.changeover_s,
            due_date_penalty_s=action.due_date_penalty_s,
            total_cost=total,
        )

    def choose(
        self,
        actions: Iterable[HorizonAction],
        context: HorizonDecisionContext,
    ) -> HorizonDecision:
        bounded = sorted(actions, key=lambda item: item.action_id)[: self.maximum_candidates]
        if not bounded:
            raise ValueError("rolling horizon requires at least one feasible action")
        candidates = tuple(self._score(action, context) for action in bounded)
        selected = min(candidates, key=lambda item: (item.total_cost, item.action_id))
        critical_note = (
            f"，避免关键工序延迟{selected.critical_path_delay_s:.1f}秒"
            if context.critical_resource is not None
            else ""
        )
        decision = HorizonDecision(
            selected_action_id=selected.action_id,
            horizon_seconds=self.horizon_seconds,
            candidates=candidates,
            explanation_zh=(
                f"滚动时域选择“{selected.action_zh}”：预测总成本{selected.total_cost:.1f}秒"
                f"{critical_note}"
            ),
        )
        self._last_decision = decision
        return decision

    def reset(self) -> None:
        """Discard explanations from the previous production run."""

        self._last_decision = None

    def snapshot(self) -> Mapping[str, object]:
        if self._last_decision is None:
            return MappingProxyType(
                {
                    "horizon_seconds": self.horizon_seconds,
                    "maximum_candidates": self.maximum_candidates,
                    "selected_action_id": None,
                    "candidates": [],
                    "explanation_zh": "等待可执行任务边界",
                }
            )
        return MappingProxyType(
            {
                **self._last_decision.as_dict(),
                "maximum_candidates": self.maximum_candidates,
            }
        )


__all__ = [
    "HorizonAction",
    "HorizonCandidate",
    "HorizonDecision",
    "HorizonDecisionContext",
    "RollingHorizonPlanner",
]
