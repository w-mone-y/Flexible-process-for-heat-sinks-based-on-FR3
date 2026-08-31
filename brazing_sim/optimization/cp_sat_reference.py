"""Bounded CP-SAT reference solver for small digital-twin horizons."""

from __future__ import annotations

from collections import defaultdict
from math import ceil
from time import perf_counter
from typing import Any, Mapping

from ..twin import DigitalTwinSnapshot
from .plan_validator import (
    ReferencePlanValidator,
    active_tasks,
    eligible_resources,
    remaining_duration,
)
from .reference_models import PlanOperation, PlanStatus, PlanValidation, PlanViolation, ReferencePlan

try:
    from ortools.sat.python import cp_model
except ImportError:  # pragma: no cover - exercised in installations without optimization extras
    cp_model = None


_TIME_SCALE = 1000


def _ticks(seconds: float) -> int:
    return max(0, int(ceil(max(0.0, seconds) * _TIME_SCALE - 1.0e-9)))


class CpSatReferencePlanner:
    """Produce a non-authoritative exact reference for the active task set."""

    def __init__(self, *, time_limit_s: float = 2.0, random_seed: int = 0) -> None:
        if time_limit_s <= 0.0:
            raise ValueError("time_limit_s must be positive")
        self.time_limit_s = float(time_limit_s)
        self.random_seed = int(random_seed)
        self.validator = ReferencePlanValidator()

    def solve(self, snapshot: DigitalTwinSnapshot) -> ReferencePlan:
        started = perf_counter()
        if cp_model is None:
            return self._result(
                snapshot,
                PlanStatus.UNAVAILABLE,
                solve_time_s=perf_counter() - started,
                message="未安装OR-Tools；请安装optimization可选依赖",
            )
        tasks = active_tasks(snapshot)
        input_violations = tuple(
            PlanViolation(
                "NO_ELIGIBLE_RESOURCE",
                f"任务{task_id}没有可执行资源",
                (task_id,),
            )
            for task_id, task in sorted(tasks.items())
            if not eligible_resources(task)
        )
        if input_violations:
            return self._result(
                snapshot,
                PlanStatus.INVALID_INPUT,
                solve_time_s=perf_counter() - started,
                message="参照模型输入不完整",
                validation=PlanValidation(False, input_violations),
            )
        if not tasks:
            plan = self._result(
                snapshot,
                PlanStatus.OPTIMAL,
                makespan_s=0.0,
                objective_value=0.0,
                best_bound=0.0,
                optimality_gap=0.0,
                solve_time_s=perf_counter() - started,
                message="当前没有活动任务",
            )
            return plan.with_validation(self.validator.validate(snapshot, plan))

        durations = {
            task_id: _ticks(remaining_duration(task, snapshot.sim_time))
            for task_id, task in tasks.items()
        }
        furnace_groups: dict[tuple[str, int], list[str]] = defaultdict(list)
        for task_id, task in tasks.items():
            if str(task.get("task_type", "")) == "RUN_FURNACE":
                recipe = str(task.get("payload", {}).get("recipe", ""))
                furnace_groups[(recipe, durations[task_id])].append(task_id)
        batched_furnace_tasks = {
            task_id for task_ids in furnace_groups.values() for task_id in task_ids
        }
        horizon = max(1, sum(durations.values()))
        model = cp_model.CpModel()
        starts: dict[str, Any] = {}
        ends: dict[str, Any] = {}
        presence: dict[tuple[str, str], Any] = {}
        resource_intervals: dict[str, list[Any]] = defaultdict(list)
        mandatory_intervals: dict[str, Any] = {}
        station_intervals: dict[str, list[Any]] = defaultdict(list)
        zone_intervals: dict[str, list[Any]] = defaultdict(list)
        batch_assignments: dict[tuple[str, str], Any] = {}

        for task_id, task in sorted(tasks.items()):
            duration = durations[task_id]
            start = model.NewIntVar(0, horizon, f"start::{task_id}")
            end = model.NewIntVar(0, horizon, f"end::{task_id}")
            starts[task_id] = start
            ends[task_id] = end
            mandatory = model.NewIntervalVar(start, duration, end, f"task::{task_id}")
            mandatory_intervals[task_id] = mandatory
            resource_choices = []
            for resource_id in eligible_resources(task):
                selected = model.NewBoolVar(f"use::{task_id}::{resource_id}")
                interval = model.NewOptionalIntervalVar(
                    start,
                    duration,
                    end,
                    selected,
                    f"resource::{task_id}::{resource_id}",
                )
                presence[(task_id, resource_id)] = selected
                if task_id not in batched_furnace_tasks:
                    resource_intervals[resource_id].append(interval)
                resource_choices.append(selected)
            model.AddExactlyOne(resource_choices)
            if str(task.get("status", "")) in {"RUNNING", "RESERVED"}:
                model.Add(start == 0)
            if task_id not in batched_furnace_tasks:
                station = task.get("station_id")
                if station:
                    station_intervals[str(station).upper()].append(mandatory)
                for zone in task.get("required_zones", ()):
                    zone_intervals[str(zone).upper()].append(mandatory)

        batch_ids: list[str] = []
        for group_index, ((_recipe, duration), task_ids) in enumerate(sorted(furnace_groups.items())):
            slot_used = []
            slot_starts = []
            task_choices: dict[str, list[Any]] = defaultdict(list)
            group_zones = {
                str(zone).upper()
                for task_id in task_ids
                for zone in tasks[task_id].get("required_zones", ())
            }
            for slot_index in range(len(task_ids)):
                batch_id = f"CP_BATCH_{group_index + 1:02d}_{slot_index + 1:02d}"
                batch_ids.append(batch_id)
                used = model.NewBoolVar(f"batch_used::{batch_id}")
                start = model.NewIntVar(0, horizon, f"batch_start::{batch_id}")
                end = model.NewIntVar(0, horizon, f"batch_end::{batch_id}")
                interval = model.NewOptionalIntervalVar(
                    start, duration, end, used, f"batch::{batch_id}"
                )
                resource_intervals["FURNACE"].append(interval)
                for zone in group_zones:
                    zone_intervals[zone].append(interval)
                members = []
                for task_id in task_ids:
                    assigned = model.NewBoolVar(f"batch_member::{task_id}::{batch_id}")
                    batch_assignments[(task_id, batch_id)] = assigned
                    task_choices[task_id].append(assigned)
                    members.append(assigned)
                    model.Add(starts[task_id] == start).OnlyEnforceIf(assigned)
                    model.Add(ends[task_id] == end).OnlyEnforceIf(assigned)
                model.Add(sum(members) >= used)
                model.Add(sum(members) <= 3 * used)
                slot_used.append(used)
                slot_starts.append(start)
            for task_id in task_ids:
                model.AddExactlyOne(task_choices[task_id])
            for slot_index in range(1, len(slot_used)):
                model.Add(slot_used[slot_index] <= slot_used[slot_index - 1])
                model.Add(slot_starts[slot_index] >= slot_starts[slot_index - 1]).OnlyEnforceIf(
                    slot_used[slot_index]
                )

        for task_id, task in tasks.items():
            for predecessor_id in task.get("predecessors", ()):
                predecessor = str(predecessor_id)
                if predecessor in ends:
                    model.Add(starts[task_id] >= ends[predecessor])
        for intervals in (*resource_intervals.values(), *station_intervals.values(), *zone_intervals.values()):
            if len(intervals) > 1:
                model.AddNoOverlap(intervals)

        makespan = model.NewIntVar(0, horizon, "makespan")
        model.AddMaxEquality(makespan, list(ends.values()))
        tardiness_terms: list[Any] = []
        tardiness_vars: dict[str, Any] = {}
        order_due_at = {
            str(order.get("order_id")): float(order["due_at_sim_time"])
            for order in snapshot.orders
            if order.get("order_id") is not None and order.get("due_at_sim_time") is not None
        }
        order_tasks: dict[str, list[str]] = defaultdict(list)
        for task_id, task in tasks.items():
            order_tasks[str(task.get("order_id", task_id))].append(task_id)
        for order_id, task_ids in sorted(order_tasks.items()):
            due_values = [
                float(tasks[task_id].get("payload", {}).get("due_at_sim_time"))
                for task_id in task_ids
                if tasks[task_id].get("payload", {}).get("due_at_sim_time") is not None
            ]
            if order_id in order_due_at:
                due_values.append(order_due_at[order_id])
            if not due_values:
                continue
            completion = model.NewIntVar(0, horizon, f"completion::{order_id}")
            model.AddMaxEquality(completion, [ends[task_id] for task_id in task_ids])
            due_tick = _ticks(max(0.0, min(due_values) - snapshot.sim_time))
            tardiness = model.NewIntVar(0, horizon, f"tardiness::{order_id}")
            model.AddMaxEquality(tardiness, [0, completion - due_tick])
            priority_weight = max(1, max(int(tasks[task_id].get("priority", 0)) for task_id in task_ids))
            tardiness_terms.append(priority_weight * tardiness)
            tardiness_vars[order_id] = (tardiness, priority_weight)
        # Weighted tardiness dominates a makespan tie but remains a readable,
        # single scalar objective for bound/gap reporting.
        tardiness_objective_weight = 100
        model.Minimize(tardiness_objective_weight * sum(tardiness_terms) + makespan)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit_s
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = self.random_seed
        status_code = solver.Solve(model)
        elapsed = perf_counter() - started
        status = {
            cp_model.OPTIMAL: PlanStatus.OPTIMAL,
            cp_model.FEASIBLE: PlanStatus.FEASIBLE,
            cp_model.INFEASIBLE: PlanStatus.INFEASIBLE,
            cp_model.MODEL_INVALID: PlanStatus.INVALID_INPUT,
            cp_model.UNKNOWN: PlanStatus.UNKNOWN,
        }.get(status_code, PlanStatus.UNKNOWN)
        if status not in {PlanStatus.OPTIMAL, PlanStatus.FEASIBLE}:
            return self._result(
                snapshot,
                status,
                solve_time_s=elapsed,
                timed_out=status is PlanStatus.UNKNOWN,
                message=f"CP-SAT状态：{solver.StatusName(status_code)}",
            )

        operations = []
        used_batch_ids = set()
        for task_id in sorted(tasks):
            resource_id = next(
                resource
                for resource in eligible_resources(tasks[task_id])
                if solver.BooleanValue(presence[(task_id, resource)])
            )
            batch_id = next(
                (
                    candidate
                    for candidate in batch_ids
                    if (task_id, candidate) in batch_assignments
                    and solver.BooleanValue(batch_assignments[(task_id, candidate)])
                ),
                None,
            )
            if batch_id is not None:
                used_batch_ids.add(batch_id)
            operations.append(
                PlanOperation(
                    task_id,
                    resource_id,
                    solver.Value(starts[task_id]) / _TIME_SCALE,
                    solver.Value(ends[task_id]) / _TIME_SCALE,
                    batch_id,
                )
            )
        objective = solver.ObjectiveValue() / _TIME_SCALE
        bound = solver.BestObjectiveBound() / _TIME_SCALE
        gap = 0.0 if status is PlanStatus.OPTIMAL else max(0.0, objective - bound) / max(objective, 1e-9)
        weighted_tardiness = sum(
            solver.Value(variable) * weight for variable, weight in tardiness_vars.values()
        ) / _TIME_SCALE
        plan = self._result(
            snapshot,
            status,
            operations=tuple(sorted(operations, key=lambda item: (item.start_s, item.task_id))),
            makespan_s=solver.Value(makespan) / _TIME_SCALE,
            weighted_tardiness_s=weighted_tardiness,
            objective_value=objective,
            best_bound=bound,
            optimality_gap=gap,
            solve_time_s=elapsed,
            timed_out=status is PlanStatus.FEASIBLE and elapsed >= self.time_limit_s,
            message=f"CP-SAT检查了{len(tasks)}个活动任务",
            metadata={
                "active_task_count": len(tasks),
                "time_scale": _TIME_SCALE,
                "wall_time_s": solver.WallTime(),
                "time_limit_s": self.time_limit_s,
                "random_seed": self.random_seed,
                "tardiness_objective_weight": tardiness_objective_weight,
                "furnace_batch_count": len(used_batch_ids),
            },
        )
        validation = self.validator.validate(snapshot, plan)
        if not validation.valid:
            return self._result(
                snapshot,
                PlanStatus.INVALID_INPUT,
                operations=plan.operations,
                makespan_s=plan.makespan_s,
                solve_time_s=elapsed,
                message="CP-SAT结果未通过独立计划校验",
                validation=validation,
            )
        return plan.with_validation(validation)

    @staticmethod
    def _result(
        snapshot: DigitalTwinSnapshot,
        status: PlanStatus,
        *,
        operations: tuple[PlanOperation, ...] = (),
        makespan_s: float = 0.0,
        weighted_tardiness_s: float = 0.0,
        objective_value: float = 0.0,
        best_bound: float | None = None,
        optimality_gap: float | None = None,
        solve_time_s: float = 0.0,
        timed_out: bool = False,
        message: str = "",
        validation: PlanValidation | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ReferencePlan:
        return ReferencePlan(
            status=status,
            operations=operations,
            makespan_s=makespan_s,
            weighted_tardiness_s=weighted_tardiness_s,
            objective_value=objective_value,
            best_bound=best_bound,
            optimality_gap=optimality_gap,
            solve_time_s=solve_time_s,
            plan_version=snapshot.plan_version,
            snapshot_fingerprint=snapshot.fingerprint,
            timed_out=timed_out,
            message=message,
            validation=validation,
            metadata={} if metadata is None else metadata,
        )


__all__ = ["CpSatReferencePlanner"]
