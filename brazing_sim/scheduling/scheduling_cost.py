"""Configurable dynamic-priority scheduling cost."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..planning.task_models import ManufacturingTask
from .resource_manager import ResourceState


@dataclass(frozen=True, slots=True)
class SchedulingWeights:
    estimated_finish_time: float = 1.0
    tool_change_cost: float = 0.8
    predicted_wait_time: float = 0.7
    zone_conflict_cost: float = 2.0
    resource_idle_penalty: float = 0.4
    execution_risk: float = 1.5
    product_changeover_cost: float = 0.6
    due_time_penalty: float = 1.2
    task_priority_bonus: float = 0.5

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "SchedulingWeights":
        if values is None:
            return cls()
        known = set(cls.__dataclass_fields__)
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown scheduler weights: {sorted(unknown)}")
        return cls(**{key: float(value) for key, value in values.items()})

    def as_dict(self) -> dict[str, float]:
        return {key: float(getattr(self, key)) for key in self.__dataclass_fields__}


def calculate_cost(
    task: ManufacturingTask,
    resource: ResourceState,
    system_state: Mapping[str, Any],
    sim_time: float,
    weights: SchedulingWeights,
) -> tuple[float, dict[str, float]]:
    tool_change = 0.0 if task.required_tool in {None, resource.current_tool} else 1.0
    predicted_wait = max(0.0, resource.estimated_available_time - float(sim_time))
    finish = predicted_wait + task.estimated_duration
    zone_conflict = float(system_state.get("zone_conflict_cost", {}).get(task.task_id, 0.0))
    idle_for = max(0.0, float(sim_time) - resource.last_state_change)
    idle_penalty = 1.0 / (1.0 + idle_for)
    risk = float(task.payload.get("execution_risk", 0.0))
    changeover = float(task.payload.get("product_changeover_cost", 0.0))
    due_at = task.payload.get("due_at_sim_time")
    due_penalty = 0.0 if due_at is None else max(0.0, float(sim_time) + finish - float(due_at))
    priority = float(task.priority)
    components = {
        "estimated_finish_time": finish,
        "tool_change_cost": tool_change,
        "predicted_wait_time": predicted_wait,
        "zone_conflict_cost": zone_conflict,
        "resource_idle_penalty": idle_penalty,
        "execution_risk": risk,
        "product_changeover_cost": changeover,
        "due_time_penalty": due_penalty,
        "task_priority_bonus": priority,
    }
    total = (
        weights.estimated_finish_time * finish
        + weights.tool_change_cost * tool_change
        + weights.predicted_wait_time * predicted_wait
        + weights.zone_conflict_cost * zone_conflict
        + weights.resource_idle_penalty * idle_penalty
        + weights.execution_risk * risk
        + weights.product_changeover_cost * changeover
        + weights.due_time_penalty * due_penalty
        - weights.task_priority_bonus * priority
    )
    return total, components


__all__ = ["SchedulingWeights", "calculate_cost"]
