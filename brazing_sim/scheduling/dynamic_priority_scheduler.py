"""Resource-aware deterministic dynamic priority scheduler."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ..planning.task_models import ManufacturingTask
from .resource_manager import ResourceState, ResourceStatus
from .scheduler_base import Assignment, SchedulerBase
from .scheduling_cost import SchedulingWeights, calculate_cost


class DynamicPriorityScheduler(SchedulerBase):
    mode = "DYNAMIC_PRIORITY"

    def __init__(
        self,
        *,
        weights: SchedulingWeights | Mapping[str, Any] | None = None,
        max_assignments_per_tick: int = 3,
        allow_parallel_tasks: bool = True,
    ) -> None:
        self.weights = (
            weights if isinstance(weights, SchedulingWeights) else SchedulingWeights.from_mapping(weights)
        )
        self.max_assignments_per_tick = int(max_assignments_per_tick)
        if self.max_assignments_per_tick < 1:
            raise ValueError("max_assignments_per_tick must be positive")
        self.allow_parallel_tasks = bool(allow_parallel_tasks)
        self.last_candidates: list[dict[str, Any]] = []
        self.last_blocked_candidates: list[dict[str, Any]] = []
        self.last_selected: list[dict[str, Any]] = []

    def select_assignments(
        self,
        ready_tasks: Iterable[ManufacturingTask],
        resource_states: Mapping[str, ResourceState],
        system_state: Mapping[str, Any],
        sim_time: float,
    ) -> list[Assignment]:
        ready = list(ready_tasks)
        occupied_zones = set(system_state.get("occupied_zones", ()))
        blocked_resource_tasks = {
            (str(task_id), str(resource_id).upper())
            for task_id, resource_id in system_state.get("blocked_resource_tasks", ())
        }
        blocked_resource_reasons = {
            str(task_id): str(reason)
            for task_id, reason in system_state.get("blocked_resource_reasons", {}).items()
        }
        task_type_priorities = {
            str(resource_id).upper(): {
                str(task_type): rank for rank, tier in enumerate(tiers) for task_type in tier
            }
            for resource_id, tiers in system_state.get(
                "resource_task_type_priorities",
                {},
            ).items()
        }
        by_id = {task.task_id: task for task in ready}
        candidates: list[tuple[float, str, str, dict[str, float]]] = []
        blocked: list[dict[str, Any]] = []
        for task in ready:
            zone_conflicts = sorted(occupied_zones.intersection(task.required_zones))
            if zone_conflicts:
                blocked.append(
                    {
                        "task_id": task.task_id,
                        "resource_id": None,
                        "reason": "等待区域释放",
                        "conflicts": zone_conflicts,
                    }
                )
                continue
            for resource_id in task.eligible_resources:
                if (task.task_id, resource_id) in blocked_resource_tasks:
                    blocked.append(
                        {
                            "task_id": task.task_id,
                            "resource_id": resource_id,
                            "reason": blocked_resource_reasons.get(task.task_id, "Arm1工具驻留策略等待"),
                        }
                    )
                    continue
                resource = resource_states.get(resource_id)
                if resource is None:
                    blocked.append(
                        {
                            "task_id": task.task_id,
                            "resource_id": resource_id,
                            "reason": "资源未配置",
                        }
                    )
                    continue
                if resource.status is not ResourceStatus.IDLE:
                    blocked.append(
                        {
                            "task_id": task.task_id,
                            "resource_id": resource_id,
                            "reason": f"资源{resource.status.value}",
                            "available_at": resource.estimated_available_time,
                        }
                    )
                    continue
                if not resource.supports(task.task_type.value, task.required_tool):
                    blocked.append(
                        {
                            "task_id": task.task_id,
                            "resource_id": resource_id,
                            "reason": "能力或工具不匹配",
                        }
                    )
                    continue
                cost, components = calculate_cost(task, resource, system_state, sim_time, self.weights)
                candidates.append((cost, task.task_id, resource.resource_id, components))
        if task_type_priorities:
            best_rank: dict[str, int] = {}
            for _cost, task_id, resource_id, _components in candidates:
                rank = task_type_priorities.get(resource_id, {}).get(by_id[task_id].task_type.value)
                if rank is not None:
                    best_rank[resource_id] = min(best_rank.get(resource_id, rank), rank)
            prioritized: list[tuple[float, str, str, dict[str, float]]] = []
            for candidate in candidates:
                _cost, task_id, resource_id, _components = candidate
                rank = task_type_priorities.get(resource_id, {}).get(by_id[task_id].task_type.value)
                if rank is not None and rank > best_rank.get(resource_id, rank):
                    blocked.append(
                        {
                            "task_id": task_id,
                            "resource_id": resource_id,
                            "reason": "等待该资源的高优先级检测任务先执行",
                        }
                    )
                    continue
                prioritized.append(candidate)
            candidates = prioritized
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        self.last_candidates = [
            {"cost": cost, "task_id": task_id, "resource_id": resource_id, **components}
            for cost, task_id, resource_id, components in candidates
        ]
        self.last_blocked_candidates = blocked
        selected: list[Assignment] = []
        used_tasks: set[str] = set()
        used_resources: set[str] = set()
        selected_zones = set(occupied_zones)
        limit = self.max_assignments_per_tick if self.allow_parallel_tasks else 1
        for cost, task_id, resource_id, components in candidates:
            task = by_id[task_id]
            if task_id in used_tasks or resource_id in used_resources:
                continue
            if selected_zones.intersection(task.required_zones):
                self.last_blocked_candidates.append(
                    {
                        "task_id": task_id,
                        "resource_id": resource_id,
                        "reason": "本轮原子分配区域冲突",
                        "conflicts": sorted(selected_zones.intersection(task.required_zones)),
                    }
                )
                continue
            selected.append(Assignment(task_id, resource_id, cost, components))
            used_tasks.add(task_id)
            used_resources.add(resource_id)
            selected_zones.update(task.required_zones)
            if len(selected) >= limit:
                break
        self.last_selected = [item.as_dict() for item in selected]
        return selected

    def snapshot(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "allow_parallel_tasks": self.allow_parallel_tasks,
            "max_assignments_per_tick": self.max_assignments_per_tick,
            "weights": self.weights.as_dict(),
            "candidate_count": len(self.last_candidates),
            "candidates": list(self.last_candidates[:30]),
            "blocked_candidates": list(self.last_blocked_candidates[:60]),
            "selected": list(self.last_selected),
        }


__all__ = ["DynamicPriorityScheduler"]
