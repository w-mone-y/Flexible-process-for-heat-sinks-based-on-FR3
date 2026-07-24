"""Validation helpers for task graphs generated from process plans."""

from __future__ import annotations

from .task_graph import TaskGraph, TaskGraphError


def validate_task_graph(graph: TaskGraph) -> None:
    graph.validate_acyclic()
    for task in graph:
        if not task.eligible_resources:
            raise TaskGraphError(f"task {task.task_id} has no eligible resource")
        if task.task_id in task.predecessors or task.task_id in task.successors:
            raise TaskGraphError(f"task {task.task_id} depends on itself")
        for predecessor_id in task.predecessors:
            if task.task_id not in graph.get(predecessor_id).successors:
                raise TaskGraphError(f"dependency index is inconsistent for {task.task_id}")


__all__ = ["validate_task_graph"]
