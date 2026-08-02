"""Run multiple registered skills without blocking the simulation tick."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..planning.task_models import ManufacturingTask
from .skill_registry import Skill, SkillExecutionResult, SkillRegistry
from .execution_monitor import ExecutionMonitor


@dataclass(slots=True)
class ActiveExecution:
    task: ManufacturingTask
    resource_id: str
    skill: Skill
    started_at: float
    last_update: float
    requires_physical_evidence: bool = False


class SkillExecutor:
    def __init__(
        self,
        registry: SkillRegistry,
        *,
        context: Any = None,
        monitor: ExecutionMonitor | None = None,
    ) -> None:
        self.registry = registry
        self.context = context
        self.active: dict[str, ActiveExecution] = {}
        self.monitor = monitor or ExecutionMonitor()

    def start_task(
        self, task: ManufacturingTask, resource_id: str, context: Any = None, now: float = 0.0
    ) -> None:
        resource = str(resource_id).upper()
        if resource in self.active:
            raise RuntimeError(f"resource {resource} already has an active task")
        skill = self.registry.create(task.task_type)
        skill.start(task, resource, self.context if context is None else context, now)
        self.active[resource] = ActiveExecution(
            task,
            resource,
            skill,
            float(now),
            float(now),
            self.registry.requires_physical_evidence(task.task_type),
        )
        timeout_provider = getattr(skill, "execution_timeout", None)
        timeout = (
            float(timeout_provider(task))
            if callable(timeout_provider)
            else max(10.0, task.estimated_duration * 5.0)
        )
        self.monitor.start(task.task_id, now, timeout)

    def update_task(self, dt: float, now: float | None = None) -> dict[str, SkillExecutionResult]:
        results: dict[str, SkillExecutionResult] = {}
        for resource_id, execution in tuple(self.active.items()):
            timestamp = execution.last_update + float(dt) if now is None else float(now)
            result = execution.skill.update(timestamp, max(0.0, timestamp - execution.last_update))
            execution.last_update = timestamp
            if (
                result.succeeded
                and execution.requires_physical_evidence
                and result.completion_evidence is None
            ):
                result = SkillExecutionResult.running_result(
                    {
                        **result.metrics,
                        "completion_blocker": "WAITING_FOR_PHYSICAL_EVIDENCE",
                    }
                )
            if result.running:
                timeout = self.monitor.update(
                    execution.task.task_id,
                    timestamp,
                    result.metrics.get("progress"),
                )
                if timeout:
                    execution.skill.cancel(execution.task.task_id)
                    result = SkillExecutionResult.failure(timeout, result.metrics)
            results[execution.task.task_id] = result
            if result.succeeded or result.failed:
                self.active.pop(resource_id, None)
                self.monitor.finish(execution.task.task_id)
        return results

    def cancel_task(self, task_id: str) -> bool:
        for resource_id, execution in tuple(self.active.items()):
            if execution.task.task_id != task_id:
                continue
            execution.skill.cancel(task_id)
            self.active.pop(resource_id, None)
            self.monitor.finish(task_id)
            return True
        return False

    def cancel_resource(self, resource_id: str) -> str | None:
        resource = str(resource_id).upper()
        execution = self.active.get(resource)
        if execution is None:
            return None
        task_id = execution.task.task_id
        execution.skill.cancel(task_id)
        self.active.pop(resource, None)
        self.monitor.finish(task_id)
        return task_id

    def reset(self) -> None:
        for execution in tuple(self.active.values()):
            execution.skill.cancel(execution.task.task_id)
        self.active.clear()
        self.monitor.reset()

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {
            resource: {
                "task_id": execution.task.task_id,
                "task_type": execution.task.task_type.value,
                "started_at": execution.started_at,
                "last_update": execution.last_update,
            }
            for resource, execution in sorted(self.active.items())
        }


__all__ = ["ActiveExecution", "SkillExecutor"]
