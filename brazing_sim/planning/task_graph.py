"""Mutable, validated manufacturing task DAG."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Any

from .task_models import ManufacturingTask, TaskStatus


class TaskGraphError(RuntimeError):
    pass


class TaskGraph:
    def __init__(self, tasks: Iterable[ManufacturingTask] = ()) -> None:
        self._tasks: dict[str, ManufacturingTask] = {}
        self._topological_ids: tuple[str, ...] | None = None
        for task in tasks:
            self.add_task(task)
        self.validate_acyclic()
        self.refresh_ready(0.0)

    @property
    def tasks(self) -> dict[str, ManufacturingTask]:
        return self._tasks

    def __iter__(self):
        return iter(self._tasks.values())

    def __len__(self) -> int:
        return len(self._tasks)

    def get(self, task_id: str) -> ManufacturingTask:
        try:
            return self._tasks[str(task_id)]
        except KeyError as exc:
            raise TaskGraphError(f"unknown task: {task_id}") from exc

    def add_task(self, task: ManufacturingTask) -> None:
        if task.task_id in self._tasks:
            raise TaskGraphError(f"duplicate task id: {task.task_id}")
        missing = [item for item in task.predecessors if item not in self._tasks]
        if missing:
            raise TaskGraphError(f"task {task.task_id} has unknown predecessors: {missing}")
        self._tasks[task.task_id] = task
        self._topological_ids = None
        for predecessor_id in task.predecessors:
            predecessor = self._tasks[predecessor_id]
            if task.task_id not in predecessor.successors:
                predecessor.successors.append(task.task_id)
        for successor_id in task.successors:
            if successor_id in self._tasks and task.task_id not in self._tasks[successor_id].predecessors:
                self._tasks[successor_id].predecessors.append(task.task_id)

    def add_dependency(self, predecessor_id: str, successor_id: str) -> None:
        predecessor = self.get(predecessor_id)
        successor = self.get(successor_id)
        if successor_id not in predecessor.successors:
            predecessor.successors.append(successor_id)
        if predecessor_id not in successor.predecessors:
            successor.predecessors.append(predecessor_id)
        self._topological_ids = None
        try:
            self.validate_acyclic()
        except Exception:
            predecessor.successors.remove(successor_id)
            successor.predecessors.remove(predecessor_id)
            self._topological_ids = None
            raise

    def remove_dependency(self, predecessor_id: str, successor_id: str) -> None:
        predecessor = self.get(predecessor_id)
        successor = self.get(successor_id)
        if successor_id in predecessor.successors:
            predecessor.successors.remove(successor_id)
        if predecessor_id in successor.predecessors:
            successor.predecessors.remove(predecessor_id)
        self._topological_ids = None

    def validate_acyclic(self) -> bool:
        indegree = {task_id: len(task.predecessors) for task_id, task in self._tasks.items()}
        queue = deque(sorted(task_id for task_id, degree in indegree.items() if degree == 0))
        visited = 0
        while queue:
            task_id = queue.popleft()
            visited += 1
            for successor_id in self._tasks[task_id].successors:
                if successor_id not in indegree:
                    raise TaskGraphError(f"unknown successor {successor_id} from {task_id}")
                indegree[successor_id] -= 1
                if indegree[successor_id] == 0:
                    queue.append(successor_id)
        if visited != len(self._tasks):
            cycle_nodes = sorted(task_id for task_id, degree in indegree.items() if degree > 0)
            raise TaskGraphError(f"task graph contains a cycle: {cycle_nodes}")
        return True

    def refresh_ready(self, now: float) -> list[ManufacturingTask]:
        ready: list[ManufacturingTask] = []
        for task in self._tasks.values():
            if task.status not in {TaskStatus.PENDING, TaskStatus.RETRY_WAIT}:
                continue
            predecessors = [self.get(task_id) for task_id in task.predecessors]
            if any(
                item.status in {TaskStatus.FAILED, TaskStatus.BLOCKED, TaskStatus.CANCELLED}
                for item in predecessors
            ):
                task.status = TaskStatus.BLOCKED
                task.failure_reason = "predecessor failed or was cancelled"
                continue
            if all(item.status is TaskStatus.SUCCEEDED for item in predecessors):
                task.mark_ready(now)
                ready.append(task)
        return ready

    def get_ready_tasks(self, now: float | None = None) -> list[ManufacturingTask]:
        if now is not None:
            self.refresh_ready(now)
        return sorted(
            (task for task in self._tasks.values() if task.status is TaskStatus.READY),
            key=lambda task: (task.sequence_index, task.task_id),
        )

    def mark_running(self, task_id: str, resource_id: str, now: float) -> ManufacturingTask:
        task = self.get(task_id)
        if task.status is TaskStatus.READY:
            task.reserve(resource_id)
        task.mark_running(now)
        return task

    def mark_succeeded(self, task_id: str, now: float) -> ManufacturingTask:
        task = self.get(task_id)
        task.mark_succeeded(now)
        self.refresh_ready(now)
        return task

    def mark_failed(self, task_id: str, reason: str, now: float = 0.0) -> ManufacturingTask:
        task = self.get(task_id)
        task.mark_failed(now, reason)
        self.block_descendants(task_id)
        return task

    def block_descendants(self, task_id: str) -> set[str]:
        blocked: set[str] = set()
        queue = deque(self.get(task_id).successors)
        while queue:
            current_id = queue.popleft()
            if current_id in blocked:
                continue
            current = self.get(current_id)
            if current.status in {TaskStatus.PENDING, TaskStatus.READY, TaskStatus.RETRY_WAIT}:
                current.status = TaskStatus.BLOCKED
                current.failure_reason = f"blocked by {task_id}"
                blocked.add(current_id)
                queue.extend(current.successors)
        return blocked

    def unblock_descendants(self, task_id: str) -> None:
        queue = deque(self.get(task_id).successors)
        visited: set[str] = set()
        while queue:
            current_id = queue.popleft()
            if current_id in visited:
                continue
            visited.add(current_id)
            current = self.get(current_id)
            if current.status is TaskStatus.BLOCKED:
                current.status = TaskStatus.PENDING
                current.failure_reason = None
            queue.extend(current.successors)

    def insert_recovery_task(
        self,
        task: ManufacturingTask,
        before_task_id: str | None = None,
        after_task_id: str | None = None,
    ) -> None:
        if task.task_id in self._tasks:
            return
        if before_task_id is not None and after_task_id is not None:
            if before_task_id in self.get(after_task_id).predecessors:
                self.remove_dependency(before_task_id, after_task_id)
            task.predecessors = list(dict.fromkeys([*task.predecessors, before_task_id]))
        elif before_task_id is not None:
            task.predecessors = list(dict.fromkeys([*task.predecessors, before_task_id]))
        self.add_task(task)
        if after_task_id is not None:
            self.add_dependency(task.task_id, after_task_id)
        self.validate_acyclic()

    def topological_order(self) -> list[ManufacturingTask]:
        if self._topological_ids is not None:
            return [self._tasks[task_id] for task_id in self._topological_ids]
        indegree = {task_id: len(task.predecessors) for task_id, task in self._tasks.items()}
        queue = deque(
            sorted(
                (task_id for task_id, degree in indegree.items() if degree == 0),
                key=lambda value: (self._tasks[value].sequence_index, value),
            )
        )
        result: list[ManufacturingTask] = []
        while queue:
            task_id = queue.popleft()
            result.append(self._tasks[task_id])
            for successor_id in self._tasks[task_id].successors:
                indegree[successor_id] -= 1
                if indegree[successor_id] == 0:
                    queue.append(successor_id)
        if len(result) != len(self._tasks):
            self.validate_acyclic()
        self._topological_ids = tuple(task.task_id for task in result)
        return result

    def snapshot(self) -> list[dict[str, Any]]:
        return [task.as_dict() for task in self.topological_order()]


__all__ = ["TaskGraph", "TaskGraphError"]
