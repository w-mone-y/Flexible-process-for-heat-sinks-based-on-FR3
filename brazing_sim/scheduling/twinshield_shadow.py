"""Deterministic TwinShield-RH shadow scheduling over an immutable twin.

The shadow planner is deliberately independent from the mutable scheduler and
MuJoCo actors.  It produces a short-window dispatch suggestion plus a complete
greedy list schedule that can be compared with the CP-SAT reference plan.
Nothing returned by this module is an execution permit.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Mapping

from ..optimization import (
    PlanOperation,
    PlanStatus,
    PlanValidation,
    ReferencePlan,
    ReferencePlanValidator,
)
from ..twin import DigitalTwinSnapshot
from .scheduling_cost import SchedulingWeights


_TERMINAL = {"SUCCEEDED", "FAILED", "BLOCKED", "CANCELLED"}
_UNAVAILABLE = {"FAULTED", "OFFLINE", "RECOVERING"}


@dataclass(frozen=True, slots=True)
class ShadowCandidate:
    task_id: str
    resource_id: str
    action_zh: str
    estimated_start_s: float
    estimated_end_s: float
    total_cost: float
    cost_components: Mapping[str, float] = field(default_factory=dict)
    selected: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "resource_id": self.resource_id,
            "action_zh": self.action_zh,
            "estimated_start_s": round(self.estimated_start_s, 6),
            "estimated_end_s": round(self.estimated_end_s, 6),
            "total_cost": round(self.total_cost, 6),
            "cost_components": {
                key: round(float(value), 6) for key, value in self.cost_components.items()
            },
            "selected": self.selected,
        }


@dataclass(frozen=True, slots=True)
class ShadowRejection:
    task_id: str
    resource_id: str | None
    reason_code: str
    reason_zh: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "resource_id": self.resource_id,
            "reason_code": self.reason_code,
            "reason_zh": self.reason_zh,
        }


@dataclass(frozen=True, slots=True)
class ShadowScheduleProposal:
    status: PlanStatus | str
    snapshot_fingerprint: str
    sim_time: float
    horizon_seconds: float
    operations: tuple[PlanOperation, ...] = ()
    selected: tuple[ShadowCandidate, ...] = ()
    candidates: tuple[ShadowCandidate, ...] = ()
    rejected: tuple[ShadowRejection, ...] = ()
    estimated_makespan_s: float = 0.0
    weighted_tardiness_s: float = 0.0
    objective_value: float = 0.0
    reference_objective_value: float | None = None
    reference_best_bound: float | None = None
    optimality_gap: float | None = None
    validation: PlanValidation | None = None
    message: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", PlanStatus(self.status))
        for value in (
            self.sim_time,
            self.horizon_seconds,
            self.estimated_makespan_s,
            self.weighted_tardiness_s,
            self.objective_value,
        ):
            if not isfinite(float(value)) or float(value) < 0.0:
                raise ValueError("shadow schedule metrics must be finite and non-negative")
        if self.optimality_gap is not None and (
            not isfinite(float(self.optimality_gap)) or float(self.optimality_gap) < 0.0
        ):
            raise ValueError("shadow optimality gap must be finite and non-negative")

    @property
    def selected_count(self) -> int:
        return len(self.selected)

    @property
    def candidate_count(self) -> int:
        evaluated = {(item.task_id, item.resource_id) for item in self.candidates}
        evaluated.update(
            (item.task_id, item.resource_id)
            for item in self.rejected
            if item.resource_id is not None
        )
        return len(evaluated) if evaluated else len(self.candidates)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "snapshot_fingerprint": self.snapshot_fingerprint,
            "sim_time": round(self.sim_time, 6),
            "horizon_seconds": self.horizon_seconds,
            "operations": [operation.as_dict() for operation in self.operations],
            "selected": [candidate.as_dict() for candidate in self.selected],
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "rejected": [item.as_dict() for item in self.rejected],
            "selected_count": self.selected_count,
            "candidate_count": self.candidate_count,
            "estimated_makespan_s": round(self.estimated_makespan_s, 6),
            "weighted_tardiness_s": round(self.weighted_tardiness_s, 6),
            "objective_value": round(self.objective_value, 6),
            "reference_objective_value": (
                None
                if self.reference_objective_value is None
                else round(self.reference_objective_value, 6)
            ),
            "reference_best_bound": (
                None if self.reference_best_bound is None else round(self.reference_best_bound, 6)
            ),
            "optimality_gap": (
                None if self.optimality_gap is None else round(self.optimality_gap, 8)
            ),
            "validation": None if self.validation is None else self.validation.as_dict(),
            "message": self.message,
        }


def _task_label(task: Mapping[str, Any]) -> str:
    return str(task.get("display_name_zh") or task.get("task_type") or task.get("task_id"))


def _remaining_duration(task: Mapping[str, Any], now: float) -> float:
    duration = max(0.0, float(task.get("estimated_duration", 0.0)))
    if str(task.get("status", "")) == "RUNNING" and task.get("started_at") is not None:
        duration = max(0.0, duration - max(0.0, now - float(task["started_at"])))
    return duration


def _eligible_resources(task: Mapping[str, Any]) -> tuple[str, ...]:
    status = str(task.get("status", ""))
    assigned = task.get("assigned_resource")
    if status in {"RUNNING", "RESERVED"} and assigned:
        return (str(assigned).upper(),)
    return tuple(dict.fromkeys(str(item).upper() for item in task.get("eligible_resources", ())))


def _resource_can_execute(resource: Mapping[str, Any], task: Mapping[str, Any]) -> tuple[bool, str, str]:
    resource_id = str(resource.get("resource_id", "")).upper()
    status = str(resource.get("status", "IDLE")).upper()
    if status in _UNAVAILABLE:
        return False, "RESOURCE_UNAVAILABLE", f"资源{resource_id}当前为{status}"
    capabilities = {str(item).upper() for item in resource.get("capabilities", ())}
    task_type = str(task.get("task_type", "")).upper()
    if capabilities and task_type not in capabilities:
        return False, "INELIGIBLE_RESOURCE", f"资源{resource_id}没有任务能力{task_type}"
    required_tool = task.get("required_tool")
    current_tool = resource.get("current_tool")
    available_tools = {str(item) for item in resource.get("available_tools", ())}
    if required_tool is not None and required_tool not in {current_tool, *available_tools}:
        return False, "TOOL_UNAVAILABLE", f"资源{resource_id}没有可用工具{required_tool}"
    return True, "", ""


def _due_at(task: Mapping[str, Any], orders: Mapping[str, Mapping[str, Any]]) -> float | None:
    payload = task.get("payload", {})
    if isinstance(payload, Mapping) and payload.get("due_at_sim_time") is not None:
        return float(payload["due_at_sim_time"])
    order = orders.get(str(task.get("order_id", "")))
    if order is not None and order.get("due_at_sim_time") is not None:
        return float(order["due_at_sim_time"])
    return None


def _objective(
    operations: Mapping[str, PlanOperation],
    tasks: Mapping[str, Mapping[str, Any]],
    snapshot: DigitalTwinSnapshot,
) -> tuple[float, float, float]:
    makespan = max((operation.end_s for operation in operations.values()), default=0.0)
    orders = {
        str(order.get("order_id")): order
        for order in snapshot.orders
        if order.get("order_id") is not None
    }
    grouped: dict[str, list[PlanOperation]] = defaultdict(list)
    for task_id, operation in operations.items():
        grouped[str(tasks[task_id].get("order_id", task_id))].append(operation)
    weighted_tardiness = 0.0
    for order_id, members in grouped.items():
        due_values = [_due_at(tasks[item.task_id], orders) for item in members]
        due_values = [value for value in due_values if value is not None]
        if not due_values:
            continue
        due_relative = max(0.0, min(due_values) - snapshot.sim_time)
        tardiness = max(0.0, max(item.end_s for item in members) - due_relative)
        priority = max(1, max(int(tasks[item.task_id].get("priority", 0)) for item in members))
        weighted_tardiness += priority * tardiness
    return makespan, weighted_tardiness, 100.0 * weighted_tardiness + makespan


class TwinShieldShadowScheduler:
    """Create deterministic shadow decisions without mutating runtime state."""

    def __init__(
        self,
        *,
        horizon_seconds: float = 60.0,
        maximum_parallel_tasks: int = 3,
        weights: SchedulingWeights | None = None,
    ) -> None:
        if horizon_seconds <= 0.0:
            raise ValueError("horizon_seconds must be positive")
        if maximum_parallel_tasks <= 0:
            raise ValueError("maximum_parallel_tasks must be positive")
        self.horizon_seconds = float(horizon_seconds)
        self.maximum_parallel_tasks = int(maximum_parallel_tasks)
        self.weights = weights or SchedulingWeights()

    def _ready_candidates(
        self,
        snapshot: DigitalTwinSnapshot,
        *,
        allowed_task_ids: set[str] | None = None,
    ) -> tuple[tuple[ShadowCandidate, ...], tuple[ShadowRejection, ...]]:
        resources = {
            str(item.get("resource_id", key)).upper(): item
            for key, item in snapshot.state.get("resources_v2", {}).items()
            if isinstance(item, Mapping)
        }
        locked_zones = {
            str(zone).upper(): lease
            for zone, lease in snapshot.state.get("zone_locks", {}).items()
            if lease is not None
        }
        candidates: list[ShadowCandidate] = []
        rejected: list[ShadowRejection] = []
        for task in snapshot.tasks:
            if str(task.get("status", "")) != "READY":
                continue
            task_id = str(task.get("task_id"))
            if allowed_task_ids is not None and task_id not in allowed_task_ids:
                rejected.append(
                    ShadowRejection(task_id, None, "OUTSIDE_COMMIT_WINDOW", "任务不在本轮安全承诺窗口")
                )
                continue
            if task.get("payload", {}).get("queue_held"):
                rejected.append(
                    ShadowRejection(task_id, None, "WIP_ADMISSION", "任务仍等待WIP与空托盘放行")
                )
                continue
            eligible = _eligible_resources(task)
            if not eligible:
                rejected.append(ShadowRejection(task_id, None, "NO_ELIGIBLE_RESOURCE", "没有候选执行资源"))
                continue
            for resource_id in eligible:
                resource = resources.get(resource_id)
                if resource is None:
                    rejected.append(ShadowRejection(task_id, resource_id, "UNKNOWN_RESOURCE", f"资源{resource_id}未注册"))
                    continue
                if str(resource.get("status", "IDLE")).upper() != "IDLE":
                    rejected.append(
                        ShadowRejection(
                            task_id,
                            resource_id,
                            "RESOURCE_BUSY",
                            f"资源{resource_id}当前为{resource.get('status')}",
                        )
                    )
                    continue
                ok, code, reason = _resource_can_execute(resource, task)
                if not ok:
                    rejected.append(ShadowRejection(task_id, resource_id, code, reason))
                    continue
                conflicts = [
                    zone
                    for zone in task.get("required_zones", ())
                    if str(zone).upper() in locked_zones
                    and locked_zones[str(zone).upper()].get("task_id") != task_id
                ]
                if conflicts:
                    rejected.append(
                        ShadowRejection(
                            task_id,
                            resource_id,
                            "ZONE_CONFLICT",
                            f"等待共享区域释放：{', '.join(sorted(conflicts))}",
                        )
                    )
                    continue
                now = snapshot.sim_time
                wait = max(0.0, float(resource.get("estimated_available_time", now)) - now)
                duration = _remaining_duration(task, now)
                finish = wait + duration
                current_tool = resource.get("current_tool")
                tool_change = 0.0 if task.get("required_tool") in {None, current_tool} else 1.0
                due_at = _due_at(
                    task,
                    {
                        str(order.get("order_id")): order
                        for order in snapshot.orders
                        if order.get("order_id") is not None
                    },
                )
                due_penalty = 0.0 if due_at is None else max(0.0, now + finish - due_at)
                priority = float(task.get("priority", 0))
                components = {
                    "estimated_finish_time": finish,
                    "predicted_wait_time": wait,
                    "tool_change_cost": tool_change,
                    "due_time_penalty": due_penalty,
                    "task_priority_bonus": priority,
                }
                total = (
                    self.weights.estimated_finish_time * finish
                    + self.weights.predicted_wait_time * wait
                    + self.weights.tool_change_cost * tool_change
                    + self.weights.due_time_penalty * due_penalty
                    - self.weights.task_priority_bonus * priority
                )
                candidates.append(
                    ShadowCandidate(
                        task_id,
                        resource_id,
                        _task_label(task),
                        now + wait,
                        now + finish,
                        total,
                        components,
                    )
                )
        return tuple(sorted(candidates, key=lambda item: (item.total_cost, item.task_id, item.resource_id))), tuple(rejected)

    @staticmethod
    def _greedy_operations(snapshot: DigitalTwinSnapshot) -> tuple[tuple[PlanOperation, ...], tuple[ShadowRejection, ...]]:
        tasks = {
            str(task.get("task_id")): task
            for task in snapshot.tasks
            if task.get("task_id") is not None
            and str(task.get("status", "")) not in _TERMINAL
            and not task.get("payload", {}).get("queue_held")
        }
        resources = {
            str(item.get("resource_id", key)).upper(): item
            for key, item in snapshot.state.get("resources_v2", {}).items()
            if isinstance(item, Mapping)
        }
        resource_available = {
            resource_id: max(0.0, float(item.get("estimated_available_time", snapshot.sim_time)) - snapshot.sim_time)
            for resource_id, item in resources.items()
        }
        station_available: dict[str, float] = defaultdict(float)
        zone_available: dict[str, float] = defaultdict(float)
        operations: dict[str, PlanOperation] = {}
        rejected: list[ShadowRejection] = []
        remaining = set(tasks)
        unavailable_predecessors = {
            str(task.get("task_id"))
            for task in snapshot.tasks
            if task.get("task_id") is not None and task.get("payload", {}).get("queue_held")
        }

        critical_path_cache: dict[str, float] = {}

        def critical_path(task_id: str, visiting: set[str] | None = None) -> float:
            cached = critical_path_cache.get(task_id)
            if cached is not None:
                return cached
            active = set() if visiting is None else visiting
            if task_id in active:
                return 0.0
            active.add(task_id)
            task = tasks[task_id]
            successors = [str(item) for item in task.get("successors", ()) if str(item) in tasks]
            value = _remaining_duration(task, snapshot.sim_time) + max(
                (critical_path(successor, active) for successor in successors),
                default=0.0,
            )
            active.remove(task_id)
            critical_path_cache[task_id] = value
            return value

        base_wave_types = {"PICK_BASE_PLATE", "PLACE_BASE_PLATE"}

        while remaining:
            frontier = [
                tasks[task_id]
                for task_id in remaining
                if all(
                    str(predecessor) not in remaining
                    and str(predecessor) not in unavailable_predecessors
                    for predecessor in tasks[task_id].get("predecessors", ())
                )
            ]
            if not frontier:
                for task_id in sorted(remaining):
                    rejected.append(ShadowRejection(task_id, None, "PRECEDENCE_CYCLE", "活动任务前驱无法展开"))
                break
            frontier.sort(
                key=lambda task: (
                    0 if str(task.get("status", "")) in {"RUNNING", "RESERVED"} else 1,
                    0 if str(task.get("task_type", "")).upper() in base_wave_types else 1,
                    -critical_path(str(task["task_id"])),
                    -int(task.get("priority", 0)),
                    int(task.get("sequence_index", 0)),
                    str(task.get("task_id")),
                )
            )
            task = frontier[0]
            task_id = str(task["task_id"])
            eligible = _eligible_resources(task)
            duration = _remaining_duration(task, snapshot.sim_time)
            predecessor_end = max(
                (operations[str(predecessor)].end_s for predecessor in task.get("predecessors", ()) if str(predecessor) in operations),
                default=0.0,
            )
            options: list[tuple[float, str]] = []
            for resource_id in eligible:
                resource = resources.get(resource_id)
                if resource is None:
                    continue
                ok, _code, _reason = _resource_can_execute(resource, task)
                if not ok:
                    continue
                fixed = str(task.get("status", "")) in {"RUNNING", "RESERVED"}
                if fixed and str(task.get("assigned_resource", "")).upper() != resource_id:
                    continue
                start = 0.0 if fixed else max(predecessor_end, resource_available.get(resource_id, 0.0))
                if not fixed:
                    station = task.get("station_id")
                    if station:
                        start = max(start, station_available[str(station).upper()])
                    for zone in task.get("required_zones", ()):
                        start = max(start, zone_available[str(zone).upper()])
                options.append((start, resource_id))
            if not options:
                rejected.append(ShadowRejection(task_id, None, "NO_FEASIBLE_RESOURCE", "当前资源能力或状态无法形成影子计划"))
                remaining.remove(task_id)
                continue
            task_type = str(task.get("task_type", "")).upper()
            preferred = {
                "PICK_FIN": "ARM3",
                "INSTALL_FIN": "ARM3",
                "REINSTALL_FIN": "ARM3",
            }.get(task_type)
            start, resource_id = min(
                options,
                key=lambda item: (
                    item[0],
                    0 if preferred is not None and item[1] == preferred else 1,
                    item[1],
                ),
            )
            end = start + duration
            operation = PlanOperation(task_id, resource_id, start, end)
            operations[task_id] = operation
            resource_available[resource_id] = max(resource_available.get(resource_id, 0.0), end)
            station = task.get("station_id")
            if station:
                station_available[str(station).upper()] = end
            for zone in task.get("required_zones", ()):
                zone_available[str(zone).upper()] = end
            remaining.remove(task_id)
        return tuple(sorted(operations.values(), key=lambda item: (item.start_s, item.task_id))), tuple(rejected)

    def plan(
        self,
        snapshot: DigitalTwinSnapshot,
        *,
        reference_plan: ReferencePlan | None = None,
        allowed_task_ids: set[str] | None = None,
        commit_window_only: bool = False,
    ) -> ShadowScheduleProposal:
        candidates, rejected = self._ready_candidates(
            snapshot,
            allowed_task_ids=allowed_task_ids,
        )
        selected: list[ShadowCandidate] = []
        used_tasks: set[str] = set()
        used_resources: set[str] = set()
        used_zones: set[str] = set()
        for candidate in candidates:
            task = next((item for item in snapshot.tasks if str(item.get("task_id")) == candidate.task_id), {})
            zones = {str(zone).upper() for zone in task.get("required_zones", ())}
            if candidate.task_id in used_tasks:
                continue
            if candidate.resource_id in used_resources:
                rejected = (*rejected, ShadowRejection(candidate.task_id, candidate.resource_id, "RESOURCE_SELECTED", "本轮已为该资源选择更优任务"))
                continue
            if zones.intersection(used_zones):
                rejected = (*rejected, ShadowRejection(candidate.task_id, candidate.resource_id, "ZONE_SELECTED", "本轮已选择占用冲突区域的任务"))
                continue
            if len(selected) >= self.maximum_parallel_tasks:
                rejected = (*rejected, ShadowRejection(candidate.task_id, candidate.resource_id, "WINDOW_CAPACITY", "达到本轮并行承诺窗口上限"))
                continue
            selected.append(
                ShadowCandidate(
                    candidate.task_id,
                    candidate.resource_id,
                    candidate.action_zh,
                    candidate.estimated_start_s,
                    candidate.estimated_end_s,
                    candidate.total_cost,
                    candidate.cost_components,
                    True,
                )
            )
            used_tasks.add(candidate.task_id)
            used_resources.add(candidate.resource_id)
            used_zones.update(zones)

        if commit_window_only:
            operations = tuple(
                PlanOperation(
                    candidate.task_id,
                    candidate.resource_id,
                    0.0,
                    max(0.0, candidate.estimated_end_s - candidate.estimated_start_s),
                )
                for candidate in selected
            )
            makespan = max((operation.end_s for operation in operations), default=0.0)
            objective = max(0.0, sum(candidate.total_cost for candidate in selected))
            tasks = {
                str(task.get("task_id")): task
                for task in snapshot.tasks
                if task.get("task_id") is not None
            }
            validation_state = snapshot.as_dict()
            validation_state["tasks"] = [tasks[item.task_id] for item in operations]
            validation_snapshot = DigitalTwinSnapshot.from_mapping(
                validation_state,
                source_name=snapshot.source_name,
                captured_at=snapshot.captured_at,
                plan_version=snapshot.plan_version,
            )
            commit_plan = ReferencePlan(
                status=PlanStatus.FEASIBLE,
                operations=operations,
                makespan_s=makespan,
                objective_value=objective,
                snapshot_fingerprint=snapshot.fingerprint,
                solver_name="TwinShield-RH-COMMIT",
                message="在线原子承诺窗口，不展开完整长时计划",
            )
            validation = ReferencePlanValidator().validate(validation_snapshot, commit_plan)
            status = PlanStatus.FEASIBLE if validation.valid else PlanStatus.INVALID_INPUT
            return ShadowScheduleProposal(
                status=status,
                snapshot_fingerprint=snapshot.fingerprint,
                sim_time=snapshot.sim_time,
                horizon_seconds=self.horizon_seconds,
                operations=operations,
                selected=tuple(selected),
                candidates=candidates,
                rejected=rejected,
                estimated_makespan_s=makespan,
                objective_value=objective,
                validation=validation,
                message=(
                    "在线原子承诺窗口已通过独立校验"
                    if validation.valid
                    else "在线原子承诺窗口未通过独立校验"
                ),
            )

        operations, greedy_rejected = self._greedy_operations(snapshot)
        rejected = tuple((*rejected, *greedy_rejected))
        tasks = {str(task.get("task_id")): task for task in snapshot.tasks if task.get("task_id") is not None}
        makespan, tardiness, objective = _objective(dict((item.task_id, item) for item in operations), tasks, snapshot)
        reference_objective = None
        reference_bound = None
        gap = None
        if reference_plan is not None and reference_plan.snapshot_fingerprint == snapshot.fingerprint:
            reference_objective = reference_plan.objective_value
            reference_bound = reference_plan.best_bound
            if reference_objective > 0.0:
                gap = max(0.0, objective - reference_objective) / reference_objective
        shadow = ReferencePlan(
            status=PlanStatus.FEASIBLE,
            operations=operations,
            makespan_s=makespan,
            weighted_tardiness_s=tardiness,
            objective_value=objective,
            snapshot_fingerprint=snapshot.fingerprint,
            solver_name="TwinShield-RH",
            message="影子滚动时域贪心计划，不提交物理执行",
        )
        # Rejected tasks are outside this proposal's feasible frontier. Validate
        # only the schedulable subset so a blocked task is reported through
        # ``rejected`` rather than misclassified as a malformed schedule.
        validation_state = snapshot.as_dict()
        validation_state["tasks"] = [tasks[item.task_id] for item in operations]
        validation_snapshot = DigitalTwinSnapshot.from_mapping(
            validation_state,
            source_name=snapshot.source_name,
            captured_at=snapshot.captured_at,
            plan_version=snapshot.plan_version,
        )
        validation = ReferencePlanValidator().validate(validation_snapshot, shadow)
        status = PlanStatus.FEASIBLE if validation.valid else PlanStatus.INVALID_INPUT
        return ShadowScheduleProposal(
            status=status,
            snapshot_fingerprint=snapshot.fingerprint,
            sim_time=snapshot.sim_time,
            horizon_seconds=self.horizon_seconds,
            operations=operations,
            selected=tuple(selected),
            candidates=candidates,
            rejected=rejected,
            estimated_makespan_s=makespan,
            weighted_tardiness_s=tardiness,
            objective_value=objective,
            reference_objective_value=reference_objective,
            reference_best_bound=reference_bound,
            optimality_gap=gap,
            validation=validation,
            message=("影子计划已通过独立校验" if validation.valid else "影子计划未通过独立校验"),
        )


__all__ = [
    "ShadowCandidate",
    "ShadowRejection",
    "ShadowScheduleProposal",
    "TwinShieldShadowScheduler",
]
