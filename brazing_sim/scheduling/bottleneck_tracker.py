"""Incremental waiting-time attribution for explainable line bottlenecks."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

from ..planning.task_models import ManufacturingTask, TaskStatus, task_type_label_zh


def _reason_category(reason: str) -> str:
    text = str(reason)
    if "工具" in text or "夹爪" in text or "吸盘" in text or "换刀" in text:
        return "TOOL_POLICY"
    if "区域" in text or "冲突" in text or "预约" in text:
        return "ZONE_OR_MOTION"
    if "资源" in text or "能力" in text:
        return "RESOURCE"
    if "炉" in text or "批" in text or "层" in text:
        return "FURNACE"
    if "托盘" in text or "工位" in text or "S1" in text or "S2" in text or "S3" in text:
        return "STATION_OR_TRANSPORT"
    if "物理" in text or "门控" in text:
        return "PHYSICAL_GATE"
    return "SCHEDULER"


class BottleneckTracker:
    """Attribute non-running task time without influencing dispatch decisions."""

    def __init__(self, now: float = 0.0) -> None:
        self.reset(now)

    def reset(self, now: float = 0.0) -> None:
        self._last_at = float(now)
        self._by_reason: dict[str, float] = defaultdict(float)
        self._by_resource: dict[str, float] = defaultdict(float)
        self._by_task_reason: dict[tuple[str, str], float] = defaultdict(float)
        self._task_metadata: dict[str, tuple[str, str | None, str | None]] = {}

    @staticmethod
    def _task_reason(
        task: ManufacturingTask,
        scheduler_blockers: Mapping[str, list[Mapping[str, Any]]],
        tasks_by_id: Mapping[str, ManufacturingTask],
    ) -> tuple[str, str, str | None] | None:
        if task.payload.get("queue_held"):
            return "WIP_ADMISSION", "等待WIP与空托盘放行", None
        if task.status in {TaskStatus.PENDING, TaskStatus.RETRY_WAIT}:
            predecessors = [tasks_by_id[item] for item in task.predecessors if item in tasks_by_id]
            frontier = not predecessors or any(
                item.status
                in {
                    TaskStatus.READY,
                    TaskStatus.RESERVED,
                    TaskStatus.RUNNING,
                    TaskStatus.RETRY_WAIT,
                }
                for item in predecessors
            )
            if not frontier:
                return None
            return "DEPENDENCY", "等待前置工序完成", None
        if task.status is not TaskStatus.READY:
            return None

        explicit = (
            task.payload.get("dispatch_blocker")
            or task.payload.get("station_blocker")
            or task.payload.get("arm1_tool_blocker")
        )
        planning = task.payload.get("planning_blockers")
        if explicit is None and planning:
            explicit = planning[0]
        if explicit is not None:
            reason = str(explicit)
            return _reason_category(reason), reason, None

        candidates = scheduler_blockers.get(task.task_id, ())
        if candidates:
            blocker = candidates[0]
            reason = str(blocker.get("reason", "等待调度资源"))
            resource = blocker.get("resource_id")
            return _reason_category(reason), reason, None if resource is None else str(resource)
        return "SCHEDULER", "READY任务等待本轮派工", None

    def observe(
        self,
        tasks: Iterable[ManufacturingTask],
        *,
        now: float,
        scheduler_blocked_candidates: Iterable[Mapping[str, Any]] = (),
    ) -> None:
        timestamp = float(now)
        duration = max(0.0, timestamp - self._last_at)
        self._last_at = timestamp
        if duration <= 0.0:
            return
        blockers: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for blocker in scheduler_blocked_candidates:
            task_id = blocker.get("task_id")
            if task_id is not None:
                blockers[str(task_id)].append(blocker)

        task_values = list(tasks)
        tasks_by_id = {task.task_id: task for task in task_values}
        for task in task_values:
            attribution = self._task_reason(task, blockers, tasks_by_id)
            if attribution is None:
                continue
            category, reason, resource = attribution
            self._by_reason[category] += duration
            if resource is not None:
                self._by_resource[resource.upper()] += duration
            self._by_task_reason[(task.task_id, reason)] += duration
            self._task_metadata[task.task_id] = (
                task_type_label_zh(task.task_type),
                task.station_id,
                task.tray_id,
            )

    def snapshot(self) -> dict[str, Any]:
        by_reason = [
            {"reason": reason, "wait_s": round(seconds, 6)}
            for reason, seconds in sorted(
                self._by_reason.items(),
                key=lambda item: (-item[1], item[0]),
            )
            if seconds > 0.0
        ]
        by_resource = [
            {"resource_id": resource, "wait_s": round(seconds, 6)}
            for resource, seconds in sorted(
                self._by_resource.items(),
                key=lambda item: (-item[1], item[0]),
            )
            if seconds > 0.0
        ]
        top_tasks = []
        for (task_id, reason), seconds in sorted(
            self._by_task_reason.items(),
            key=lambda item: (-item[1], item[0]),
        )[:20]:
            display_name, station_id, tray_id = self._task_metadata[task_id]
            top_tasks.append(
                {
                    "task_id": task_id,
                    "display_name_zh": display_name,
                    "station_id": station_id,
                    "tray_id": tray_id,
                    "reason": reason,
                    "wait_s": round(seconds, 6),
                }
            )
        return {
            "measurement": "task_wait_seconds",
            "total_wait_s": round(sum(self._by_reason.values()), 6),
            "by_reason": by_reason,
            "by_resource": by_resource,
            "top_tasks": top_tasks,
        }


__all__ = ["BottleneckTracker"]
