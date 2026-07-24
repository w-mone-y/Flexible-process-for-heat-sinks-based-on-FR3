"""Central recovery policy; no recovery branches live in the main loop."""

from __future__ import annotations

from typing import Any

from ..planning.task_graph import TaskGraph
from ..planning.task_models import ManufacturingTask, TaskStatus, TaskType
from .fault_models import FaultRecord, FaultType, RecoveryPlan, RecoveryStatus, RecoveryStep


class RecoveryPolicy:
    def __init__(self) -> None:
        self.plans: dict[str, RecoveryPlan] = {}
        self._handled_faults: set[str] = set()
        self._strategy_attempts: dict[tuple[str, str, str], int] = {}

    @staticmethod
    def _resources_for(
        fault: FaultRecord, original: ManufacturingTask | None
    ) -> tuple[list[str], list[str], str | None]:
        if original is not None:
            return (
                list(original.eligible_resources),
                list(original.required_zones),
                original.required_tool,
            )
        source = fault.source.upper()
        return ([source], [], None)

    @staticmethod
    def _make_task(
        *,
        task_id: str,
        task_type: TaskType,
        fault: FaultRecord,
        original: ManufacturingTask | None,
        predecessors: list[str] | None = None,
        payload: dict[str, Any] | None = None,
        retry_limit: int = 0,
    ) -> ManufacturingTask:
        resources, zones, tool = RecoveryPolicy._resources_for(fault, original)
        return ManufacturingTask(
            task_id=task_id,
            task_type=task_type,
            order_id="SYSTEM" if original is None else original.order_id,
            unit_id="SYSTEM" if original is None else original.unit_id,
            tray_id=None if original is None else original.tray_id,
            predecessors=list(predecessors or ()),
            eligible_resources=resources,
            required_zones=zones,
            required_tool=tool,
            estimated_duration=1.0 if original is None else original.estimated_duration,
            priority=100 if original is None else max(100, original.priority),
            retry_limit=retry_limit,
            payload={**({} if original is None else original.payload), **dict(payload or {})},
            recovery_for=fault.related_task_id or fault.fault_id,
            sequence_index=(0 if original is None else original.sequence_index),
        )

    def _insert_chain(
        self,
        graph: TaskGraph,
        original: ManufacturingTask | None,
        tasks: list[ManufacturingTask],
        now: float,
    ) -> None:
        old_successors = [] if original is None else list(original.successors)
        descendants: set[str] = set()
        pending = list(old_successors)
        while pending:
            current_id = pending.pop()
            if current_id in descendants:
                continue
            descendants.add(current_id)
            pending.extend(graph.get(current_id).successors)
        if original is not None:
            for successor_id in old_successors:
                graph.remove_dependency(original.task_id, successor_id)
        previous: str | None = None
        for task in tasks:
            if previous is not None:
                task.predecessors = [previous]
            graph.add_task(task)
            previous = task.task_id
        if previous is not None:
            for descendant_id in descendants:
                descendant = graph.get(descendant_id)
                if descendant.status is TaskStatus.BLOCKED:
                    descendant.status = TaskStatus.PENDING
                    descendant.failure_reason = None
            for successor_id in old_successors:
                graph.add_dependency(previous, successor_id)
        graph.validate_acyclic()
        graph.refresh_ready(now)

    def plan(self, fault: FaultRecord, graph: TaskGraph, now: float) -> RecoveryPlan:
        existing = self.plans.get(fault.recovery_id or f"RECOVERY_{fault.fault_id}")
        if fault.fault_id in self._handled_faults and existing is not None:
            return existing
        recovery_id = f"RECOVERY_{fault.fault_id}"
        original = graph.get(fault.related_task_id) if fault.related_task_id else None
        tasks: list[ManufacturingTask] = []
        strategy = "MANUAL_REVIEW"
        retry_limit = 0

        if fault.fault_type in {FaultType.BRAZING_MISSING, FaultType.BRAZING_PATH_DEVIATION}:
            strategy = "LOCAL_BRAZING_REWORK"
            retry_limit = 2
            rework = self._make_task(
                task_id=f"{recovery_id}_REWORK",
                task_type=TaskType.REWORK_BRAZING,
                fault=fault,
                original=original,
                payload={"path_ids": fault.details.get("path_ids", ())},
                retry_limit=2,
            )
            rework.eligible_resources = ["ARM2"]
            rework.required_tool = "brazing_dispenser"
            inspect = self._make_task(
                task_id=f"{recovery_id}_REINSPECT",
                task_type=TaskType.INSPECT_BRAZING,
                fault=fault,
                original=original,
                retry_limit=2,
            )
            inspect.eligible_resources = ["ARM3"]
            inspect.required_tool = None
            tasks = [rework, inspect]
        elif fault.fault_type in {
            FaultType.FIN_PICK_FAILED,
            FaultType.FIN_INSERT_FAILED,
            FaultType.FIN_GEOMETRY_FAILED,
        }:
            strategy = "FIN_REINSTALL"
            retry_limit = 2
            reinstall = self._make_task(
                task_id=f"{recovery_id}_REINSTALL",
                task_type=TaskType.REINSTALL_FIN,
                fault=fault,
                original=original,
                payload={"fin_id": fault.details.get("fin_id")},
                retry_limit=2,
            )
            reinstall.eligible_resources = ["ARM1"]
            reinstall.required_tool = "parallel_gripper"
            inspect = self._make_task(
                task_id=f"{recovery_id}_REINSPECT",
                task_type=TaskType.INSPECT_FINS,
                fault=fault,
                original=original,
                retry_limit=2,
            )
            inspect.eligible_resources = ["ARM3"]
            inspect.required_tool = None
            tasks = [reinstall, inspect]
        elif fault.fault_type in {FaultType.ELEVATOR_TIMEOUT, FaultType.FORK_TIMEOUT}:
            strategy = "TRANSFER_SAFE_HOME_RETRY"
            retry_limit = 1
            safe_home = self._make_task(
                task_id=f"{recovery_id}_SAFE_HOME",
                task_type=TaskType.SAFE_HOME_TRANSFER,
                fault=fault,
                original=original,
                retry_limit=1,
            )
            tasks = [safe_home]
        elif fault.fault_type is FaultType.FURNACE_DOOR_INTERLOCK:
            strategy = "FURNACE_INTERLOCK_RECHECK"
            check = self._make_task(
                task_id=f"{recovery_id}_INTERLOCK_CHECK",
                task_type=TaskType.FURNACE_INTERLOCK_CHECK,
                fault=fault,
                original=original,
            )
            check.eligible_resources = ["FURNACE"]
            check.required_zones = ["ZONE_FURNACE_LOADING"]
            tasks = [check]
        elif fault.fault_type is FaultType.RACK_LAYER_UNAVAILABLE:
            strategy = "RACK_LAYER_REALLOCATION"
        elif fault.fault_type is FaultType.ARM_UNAVAILABLE:
            strategy = "RESOURCE_REALLOCATION"

        if strategy in {"LOCAL_BRAZING_REWORK", "FIN_REINSTALL"}:
            target = str(
                fault.details.get("fin_id")
                or fault.details.get("path_ids")
                or ("" if original is None else original.task_id)
            )
            unit_id = "SYSTEM" if original is None else original.unit_id
            key = (strategy, unit_id, target)
            attempts = self._strategy_attempts.get(key, 0)
            if attempts >= retry_limit:
                fault.recoverable = False
                tasks = []
            else:
                self._strategy_attempts[key] = attempts + 1

        plan = RecoveryPlan(
            recovery_id=recovery_id,
            fault_id=fault.fault_id,
            strategy=strategy,
            steps=[
                RecoveryStep(f"{recovery_id}_STEP_{index + 1:02d}", task.task_type.value, task.task_id)
                for index, task in enumerate(tasks)
            ],
            status=RecoveryStatus.PLANNED if fault.recoverable else RecoveryStatus.MANUAL_REVIEW,
            retry_limit=retry_limit,
            created_at=float(now),
        )
        self.plans[recovery_id] = plan
        self._handled_faults.add(fault.fault_id)
        fault.recovery_id = recovery_id
        if fault.recoverable and tasks:
            self._insert_chain(graph, original, tasks, now)
            plan.status = RecoveryStatus.RUNNING
        elif not fault.recoverable or strategy == "MANUAL_REVIEW":
            plan.status = RecoveryStatus.MANUAL_REVIEW
        return plan

    def update(self, graph: TaskGraph, now: float) -> None:
        for plan in self.plans.values():
            if plan.status not in {RecoveryStatus.RUNNING, RecoveryStatus.PLANNED}:
                continue
            task_states = [graph.get(step.task_id).status for step in plan.steps if step.task_id]
            for step in plan.steps:
                if step.task_id:
                    step.status = graph.get(step.task_id).status.value
            if task_states and all(status is TaskStatus.SUCCEEDED for status in task_states):
                plan.status = RecoveryStatus.SUCCEEDED
                plan.completed_at = float(now)
                if plan.steps and plan.steps[0].task_id:
                    recovered_for = graph.get(plan.steps[0].task_id).recovery_for
                    if recovered_for in graph.tasks:
                        graph.get(recovered_for).payload["recovered"] = True
            elif any(status in {TaskStatus.FAILED, TaskStatus.BLOCKED} for status in task_states):
                plan.status = RecoveryStatus.FAILED
                plan.completed_at = float(now)

    def action(self, recovery_id: str, action: str, now: float) -> RecoveryPlan:
        plan = self.plans[recovery_id]
        command = str(action).lower()
        if command == "pause":
            plan.status = RecoveryStatus.PAUSED
        elif command == "resume" and plan.status is RecoveryStatus.PAUSED:
            plan.status = RecoveryStatus.RUNNING
        elif command == "retry" and plan.status is RecoveryStatus.FAILED:
            if plan.retry_count >= plan.retry_limit:
                raise RuntimeError("recovery retry limit exceeded")
            plan.retry_count += 1
            plan.status = RecoveryStatus.RUNNING
            plan.completed_at = None
            for step in plan.steps:
                if step.task_id:
                    step.status = "RETRY_WAIT"
        elif command == "manual_review":
            plan.status = RecoveryStatus.MANUAL_REVIEW
            plan.completed_at = float(now)
        else:
            raise ValueError(f"invalid recovery action {action!r} for {plan.status.value}")
        return plan

    def reset(self) -> None:
        self.plans.clear()
        self._handled_faults.clear()
        self._strategy_attempts.clear()


__all__ = ["RecoveryPolicy"]
