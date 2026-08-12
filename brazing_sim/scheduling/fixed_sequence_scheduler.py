"""Deterministic scheduler used as the compatibility baseline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ..planning.task_models import ManufacturingTask
from .resource_manager import ResourceState, ResourceStatus
from .scheduler_base import Assignment, SchedulerBase


class FixedSequenceScheduler(SchedulerBase):
    mode = "FIXED_SEQUENCE"

    def __init__(self, *, allow_encoded_parallelism: bool = False) -> None:
        self.allow_encoded_parallelism = bool(allow_encoded_parallelism)

    def select_assignments(
        self,
        ready_tasks: Iterable[ManufacturingTask],
        resource_states: Mapping[str, ResourceState],
        system_state: Mapping[str, Any],
        sim_time: float,
    ) -> list[Assignment]:
        del sim_time
        if not self.allow_encoded_parallelism and any(
            resource.status in {ResourceStatus.RESERVED, ResourceStatus.BUSY}
            for resource in resource_states.values()
        ):
            return []
        selected: list[Assignment] = []
        used_resources: set[str] = set()
        used_zones = set(system_state.get("occupied_zones", ()))
        blocked_resource_tasks = {
            (str(task_id), str(resource_id).upper())
            for task_id, resource_id in system_state.get("blocked_resource_tasks", ())
        }
        first_sequence: int | None = None
        for task in sorted(ready_tasks, key=lambda item: (item.sequence_index, item.task_id)):
            if first_sequence is not None and task.sequence_index > first_sequence + 2:
                break
            if used_zones.intersection(task.required_zones):
                continue
            resource = next(
                (
                    resource_states[resource_id]
                    for resource_id in task.eligible_resources
                    if resource_id in resource_states
                    and resource_id not in used_resources
                    and (task.task_id, resource_id) not in blocked_resource_tasks
                    and resource_states[resource_id].status is ResourceStatus.IDLE
                    and resource_states[resource_id].supports(task.task_type.value, task.required_tool)
                ),
                None,
            )
            if resource is None:
                continue
            selected.append(Assignment(task.task_id, resource.resource_id))
            if first_sequence is None:
                first_sequence = task.sequence_index
            used_resources.add(resource.resource_id)
            used_zones.update(task.required_zones)
            if not self.allow_encoded_parallelism:
                break
            # The fixed baseline only admits tasks that share a sequence level;
            # this preserves explicitly encoded branches without optimizing.
        return selected

    def snapshot(self) -> dict[str, Any]:
        return {"mode": self.mode, "allow_encoded_parallelism": self.allow_encoded_parallelism}


__all__ = ["FixedSequenceScheduler"]
