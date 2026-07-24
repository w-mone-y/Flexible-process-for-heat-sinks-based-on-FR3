"""Online graph and target replanning utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Iterable

from ..planning.task_graph import TaskGraph
from ..planning.task_models import TaskStatus, TaskType


@dataclass(frozen=True, slots=True)
class ReplanResult:
    reason: str
    changed_tasks: tuple[str, ...] = ()
    new_layer: int | None = None
    duration_seconds: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "changed_tasks": list(self.changed_tasks),
            "new_layer": self.new_layer,
            "duration_seconds": self.duration_seconds,
            "details": dict(self.details),
        }


class Replanner:
    def __init__(self) -> None:
        self.replan_count = 0
        self.total_duration_seconds = 0.0
        self.history: list[ReplanResult] = []

    def reassign_rack_layer(
        self,
        graph: TaskGraph,
        unavailable_layer: int,
        available_layers: Iterable[int],
        *,
        unit_id: str | None = None,
    ) -> ReplanResult:
        started = time.perf_counter()
        available = sorted(set(int(item) for item in available_layers) - {int(unavailable_layer)})
        if not available:
            raise RuntimeError("no available rack layer for replanning")
        changed: list[str] = []
        if unit_id is None:
            unit_id = next(
                (
                    task.unit_id
                    for task in graph
                    if task.task_type is TaskType.LOAD_RACK_LAYER
                    and int(task.payload.get("layer_index", -1)) == int(unavailable_layer)
                    and task.status not in {TaskStatus.RUNNING, TaskStatus.SUCCEEDED}
                ),
                None,
            )
        if unit_id is None:
            raise RuntimeError("no pending unit uses the unavailable rack layer")
        target_layer = available[0]
        for task in graph:
            if task.task_type not in {
                TaskType.MOVE_ELEVATOR,
                TaskType.LOAD_RACK_LAYER,
                TaskType.LOCK_RACK_LAYER,
                TaskType.UNLOAD_RACK_LAYER,
            }:
                continue
            if task.unit_id != unit_id:
                continue
            if int(task.payload.get("layer_index", -1)) != int(unavailable_layer):
                continue
            if task.status in {TaskStatus.RUNNING, TaskStatus.SUCCEEDED}:
                raise RuntimeError("cannot reassign a rack layer after physical loading started")
            task.payload["layer_index"] = target_layer
            if task.task_type is TaskType.LOCK_RACK_LAYER:
                task.eligible_resources = [f"RACK_LAYER_{target_layer + 1:02d}"]
            changed.append(task.task_id)
        result = self._record(
            "RACK_LAYER_UNAVAILABLE",
            changed,
            target_layer,
            started,
            {"unavailable_layer": unavailable_layer},
        )
        graph.refresh_ready(0.0)
        return result

    def replan_ready_set(self, graph: TaskGraph, reason: str, now: float) -> ReplanResult:
        started = time.perf_counter()
        before = {task.task_id for task in graph.get_ready_tasks()}
        graph.refresh_ready(now)
        after = {task.task_id for task in graph.get_ready_tasks()}
        return self._record(reason, sorted(before.symmetric_difference(after)), None, started, {})

    def _record(
        self,
        reason: str,
        changed: Iterable[str],
        new_layer: int | None,
        started: float,
        details: dict[str, Any],
    ) -> ReplanResult:
        duration = max(0.0, time.perf_counter() - started)
        result = ReplanResult(reason, tuple(changed), new_layer, duration, details)
        self.replan_count += 1
        self.total_duration_seconds += duration
        self.history.append(result)
        return result

    def reset(self) -> None:
        self.replan_count = 0
        self.total_duration_seconds = 0.0
        self.history.clear()


__all__ = ["ReplanResult", "Replanner"]
