"""Derive manufacturing metrics exclusively from runtime events and tasks."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..changeover.metrics import changeover_seconds_from_graph, collect_changeover_kpi
from ..events import EventType, SystemEvent
from ..manufacturing_runtime import ManufacturingRuntime, OrderRunStatus
from ..planning.task_models import TaskStatus


class MetricsCollector:
    def __init__(self) -> None:
        self.events: list[SystemEvent] = []

    def handle_event(self, event: SystemEvent) -> None:
        self.events.append(event)

    def calculate(self, runtime: ManufacturingRuntime, now: float | None = None) -> dict[str, Any]:
        timestamp = runtime.last_tick if now is None else float(now)
        timestamp = 0.0 if timestamp is None else timestamp
        start = runtime.started_at if runtime.started_at is not None else timestamp
        makespan = max(0.0, timestamp - start)
        task_started: dict[str, float] = {}
        busy: dict[str, float] = defaultdict(float)
        fault_count = 0
        recovered_count = 0
        for event in self.events:
            task_id = str(event.payload.get("task_id", ""))
            if event.event_type is EventType.TASK_STARTED:
                task_started[task_id] = event.sim_time
            elif event.event_type in {EventType.TASK_SUCCEEDED, EventType.TASK_FAILED}:
                started_at = task_started.pop(task_id, event.sim_time)
                busy[event.source] += max(0.0, event.sim_time - started_at)
        for task_id, started_at in task_started.items():
            task = runtime.graph.tasks.get(task_id)
            if task and task.assigned_resource:
                busy[task.assigned_resource] += max(0.0, timestamp - started_at)
        for fault in runtime.faults.values():
            fault_count += 1
            recovered_count += int(fault.recovered)
        completed_units = len(runtime.unit_dispositions)
        completed_orders = sum(entry.status is OrderRunStatus.COMPLETED for entry in runtime.orders.values())
        on_time = sum(
            entry.status is OrderRunStatus.COMPLETED
            and (entry.plan.order.due_time is None or entry.completed_at is not None)
            for entry in runtime.orders.values()
        )
        due_orders = sum(entry.plan.order.due_time is not None for entry in runtime.orders.values())
        waiting_values = [
            max(0.0, (task.started_at or timestamp) - task.ready_at)
            for task in runtime.graph
            if task.ready_at is not None and task.status is not TaskStatus.PENDING
        ]
        resources = runtime.resources.states
        utilization = {
            resource_id: (0.0 if makespan <= 0 else busy.get(resource_id, 0.0) / makespan)
            for resource_id, resource in resources.items()
            if resource.resource_type == "ROBOT"
        }
        # Step D: changeover KPIs.  ``changeover_ratio_vs_baseline`` is one of the
        # three metrics the competition names explicitly.  Measured from the task
        # graph as well as the runtime log so the two can be cross-checked.
        graph_seconds, graph_tasks = changeover_seconds_from_graph(runtime.graph)
        changeover = collect_changeover_kpi(
            getattr(runtime, "changeover_log", ()),
            runtime.teaching_baseline,
            productive_seconds=sum(busy.values()),
        ).as_dict()
        changeover["changeover_task_count"] = graph_tasks
        changeover["changeover_seconds_from_graph"] = graph_seconds
        return {
            **changeover,
            "installed_fixture": runtime.installed_fixture.as_dict(),
            "makespan": makespan,
            "throughput_per_sim_second": 0.0 if makespan <= 0 else completed_units / makespan,
            "completed_units": completed_units,
            "completed_orders": completed_orders,
            "rework_units": sum(value == "REWORK_REQUIRED" for value in runtime.unit_dispositions.values()),
            "scrapped_units": sum(value == "SCRAPPED" for value in runtime.unit_dispositions.values()),
            "task_succeeded": sum(task.status is TaskStatus.SUCCEEDED for task in runtime.graph),
            "task_failed": sum(task.status is TaskStatus.FAILED for task in runtime.graph),
            "retry_count": sum(task.retry_count for task in runtime.graph),
            "replan_count": runtime.replanner.replan_count,
            "replan_time_seconds": runtime.replanner.total_duration_seconds,
            "fault_count": fault_count,
            "recovered_fault_count": recovered_count,
            "recovery_rate": 0.0 if fault_count == 0 else recovered_count / fault_count,
            "robot_busy_seconds": dict(busy),
            "robot_utilization": utilization,
            "average_robot_utilization": (
                0.0 if not utilization else sum(utilization.values()) / len(utilization)
            ),
            "average_task_wait_seconds": (
                0.0 if not waiting_values else sum(waiting_values) / len(waiting_values)
            ),
            "zone_conflict_count": runtime.zones.conflict_count,
            "tool_change_count": sum(
                task.task_type.value == "PREPARE_FIN_TOOL" and task.status is TaskStatus.SUCCEEDED
                for task in runtime.graph
            ),
            "on_time_rate": 1.0 if due_orders == 0 else on_time / due_orders,
        }


__all__ = ["MetricsCollector"]
