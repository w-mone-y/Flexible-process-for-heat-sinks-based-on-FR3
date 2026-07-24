"""Common scheduler interface and assignment model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ..planning.task_models import ManufacturingTask
from .resource_manager import ResourceState


@dataclass(frozen=True, slots=True)
class Assignment:
    task_id: str
    resource_id: str
    cost: float = 0.0
    cost_components: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "resource_id": self.resource_id,
            "cost": self.cost,
            "cost_components": dict(self.cost_components),
        }


class SchedulerBase(ABC):
    mode = "BASE"

    @abstractmethod
    def select_assignments(
        self,
        ready_tasks: Iterable[ManufacturingTask],
        resource_states: Mapping[str, ResourceState],
        system_state: Mapping[str, Any],
        sim_time: float,
    ) -> list[Assignment]:
        raise NotImplementedError

    def snapshot(self) -> dict[str, Any]:
        return {"mode": self.mode}


__all__ = ["Assignment", "SchedulerBase"]
