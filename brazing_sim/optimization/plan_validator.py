"""Independent validation for reference schedules."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

from ..twin import DigitalTwinSnapshot
from .reference_models import PlanOperation, PlanValidation, PlanViolation, ReferencePlan


_TERMINAL = {"SUCCEEDED", "FAILED", "BLOCKED", "CANCELLED"}
_TOLERANCE = 1.0e-6


def active_tasks(snapshot: DigitalTwinSnapshot) -> dict[str, Mapping[str, Any]]:
    return {
        str(task["task_id"]): task
        for task in snapshot.tasks
        if task.get("task_id") is not None and str(task.get("status", "")) not in _TERMINAL
    }


def remaining_duration(task: Mapping[str, Any], now: float) -> float:
    duration = max(0.0, float(task.get("estimated_duration", 0.0)))
    if str(task.get("status", "")) == "RUNNING" and task.get("started_at") is not None:
        duration = max(0.0, duration - max(0.0, now - float(task["started_at"])))
    return duration


def eligible_resources(task: Mapping[str, Any]) -> tuple[str, ...]:
    status = str(task.get("status", ""))
    assigned = task.get("assigned_resource")
    if status in {"RUNNING", "RESERVED"} and assigned:
        return (str(assigned).upper(),)
    return tuple(dict.fromkeys(str(value).upper() for value in task.get("eligible_resources", ())))


def _overlaps(first: PlanOperation, second: PlanOperation) -> bool:
    return first.start_s < second.end_s - _TOLERANCE and second.start_s < first.end_s - _TOLERANCE


def _same_batch(first: PlanOperation, second: PlanOperation) -> bool:
    return first.batch_id is not None and first.batch_id == second.batch_id


def _pairwise(values: Iterable[PlanOperation]):
    ordered = sorted(values, key=lambda item: (item.start_s, item.end_s, item.task_id))
    for index, first in enumerate(ordered):
        for second in ordered[index + 1 :]:
            if second.start_s >= first.end_s - _TOLERANCE:
                break
            yield first, second


class ReferencePlanValidator:
    def validate(self, snapshot: DigitalTwinSnapshot, plan: ReferencePlan) -> PlanValidation:
        tasks = active_tasks(snapshot)
        operations: dict[str, PlanOperation] = {}
        violations: list[PlanViolation] = []
        for operation in plan.operations:
            if operation.task_id in operations:
                violations.append(
                    PlanViolation("DUPLICATE_TASK", "参照计划包含重复任务", (operation.task_id,))
                )
                continue
            operations[operation.task_id] = operation
            task = tasks.get(operation.task_id)
            if task is None:
                violations.append(
                    PlanViolation("UNKNOWN_TASK", "参照计划包含未知或已结束任务", (operation.task_id,))
                )
                continue
            resources = eligible_resources(task)
            if operation.resource_id.upper() not in resources:
                violations.append(
                    PlanViolation(
                        "INELIGIBLE_RESOURCE",
                        f"资源{operation.resource_id}不能执行任务{operation.task_id}",
                        (operation.task_id,),
                    )
                )
            minimum = remaining_duration(task, snapshot.sim_time)
            if operation.duration_s + _TOLERANCE < minimum:
                violations.append(
                    PlanViolation(
                        "DURATION_TOO_SHORT",
                        f"任务{operation.task_id}计划工期短于剩余名义工期",
                        (operation.task_id,),
                    )
                )
            if str(task.get("status", "")) in {"RUNNING", "RESERVED"} and operation.start_s > _TOLERANCE:
                violations.append(
                    PlanViolation(
                        "COMMITTED_START_MOVED",
                        f"已承诺任务{operation.task_id}没有从当前时刻继续",
                        (operation.task_id,),
                    )
                )

        for task_id in sorted(set(tasks) - set(operations)):
            code = "NO_ELIGIBLE_RESOURCE" if not eligible_resources(tasks[task_id]) else "MISSING_TASK"
            violations.append(PlanViolation(code, f"参照计划缺少活动任务{task_id}", (task_id,)))

        for task_id, task in tasks.items():
            operation = operations.get(task_id)
            if operation is None:
                continue
            for predecessor_id in task.get("predecessors", ()):
                predecessor = operations.get(str(predecessor_id))
                if predecessor is not None and operation.start_s + _TOLERANCE < predecessor.end_s:
                    violations.append(
                        PlanViolation(
                            "PRECEDENCE_VIOLATION",
                            f"任务{task_id}早于前驱{predecessor_id}完成",
                            (str(predecessor_id), task_id),
                        )
                    )

        self._append_overlap_violations(tasks, operations, violations)
        self._append_batch_violations(tasks, operations, violations)
        actual_makespan = max((operation.end_s for operation in operations.values()), default=0.0)
        if abs(actual_makespan - plan.makespan_s) > _TOLERANCE:
            violations.append(
                PlanViolation("MAKESPAN_MISMATCH", "计划makespan与任务结束时间不一致")
            )
        return PlanValidation(not violations, tuple(violations))

    @staticmethod
    def _append_overlap_violations(
        tasks: Mapping[str, Mapping[str, Any]],
        operations: Mapping[str, PlanOperation],
        violations: list[PlanViolation],
    ) -> None:
        by_resource: dict[str, list[PlanOperation]] = defaultdict(list)
        by_station: dict[str, list[PlanOperation]] = defaultdict(list)
        by_zone: dict[str, list[PlanOperation]] = defaultdict(list)
        for task_id, operation in operations.items():
            task = tasks.get(task_id)
            if task is None:
                continue
            by_resource[operation.resource_id.upper()].append(operation)
            station = task.get("station_id")
            if station:
                by_station[str(station).upper()].append(operation)
            for zone in task.get("required_zones", ()):
                by_zone[str(zone).upper()].append(operation)
        for code, label, groups in (
            ("RESOURCE_OVERLAP", "资源", by_resource),
            ("STATION_OVERLAP", "工位", by_station),
            ("ZONE_OVERLAP", "区域", by_zone),
        ):
            for group, values in groups.items():
                for first, second in _pairwise(values):
                    if _overlaps(first, second) and not _same_batch(first, second):
                        violations.append(
                            PlanViolation(
                                code,
                                f"{label}{group}上的任务时间重叠",
                                (first.task_id, second.task_id),
                            )
                        )

    @staticmethod
    def _append_batch_violations(
        tasks: Mapping[str, Mapping[str, Any]],
        operations: Mapping[str, PlanOperation],
        violations: list[PlanViolation],
    ) -> None:
        batches: dict[str, list[PlanOperation]] = defaultdict(list)
        for operation in operations.values():
            if operation.batch_id is not None:
                batches[operation.batch_id].append(operation)
        for batch_id, members in batches.items():
            task_ids = tuple(member.task_id for member in members)
            if len(members) > 3:
                violations.append(
                    PlanViolation("BATCH_CAPACITY", f"炉批{batch_id}超过三件容量", task_ids)
                )
            recipes = set()
            for member in members:
                task = tasks.get(member.task_id, {})
                if task.get("task_type") != "RUN_FURNACE" or member.resource_id.upper() != "FURNACE":
                    violations.append(
                        PlanViolation(
                            "INVALID_BATCH_MEMBER",
                            f"炉批{batch_id}包含非炉体任务",
                            (member.task_id,),
                        )
                    )
                recipes.add(str(task.get("payload", {}).get("recipe", "")))
            if len(recipes) > 1:
                violations.append(
                    PlanViolation("BATCH_INCOMPATIBLE", f"炉批{batch_id}配方不兼容", task_ids)
                )
            windows = {(member.start_s, member.end_s) for member in members}
            if len(windows) > 1:
                violations.append(
                    PlanViolation("BATCH_WINDOW_MISMATCH", f"炉批{batch_id}成员周期不一致", task_ids)
                )


__all__ = [
    "ReferencePlanValidator",
    "active_tasks",
    "eligible_resources",
    "remaining_duration",
]
