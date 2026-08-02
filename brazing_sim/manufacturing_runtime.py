"""Unified task-DAG manufacturing runtime used by GUI and headless modes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable

from .events import EventBus, EventType
from .execution import SkillExecutor, SkillRegistry, TimedSkill
from .flexible.models import ProcessPlan
from .layout import SHALLOW_U_LAYOUT
from .manufacturing_config import (
    FaultScenario,
    SchedulerConfig,
    load_resource_config,
    load_scheduler_config,
)
from .planning.task_graph import TaskGraph
from .planning.batch_planner import are_process_plans_compatible
from .changeover.config_diff import FixtureConfiguration, required_configuration
from .changeover.setup_matrix import (
    PLACEHOLDER_TEACHING_BASELINE,
    SetupTimeMatrix,
    TeachingBaseline,
)
from .planning.capability_binding import V1_SHALLOW_U_PROFILE, LineExecutionProfile
from .planning.task_graph_builder import (
    LEGACY_DURATIONS,
    ProcessPlanTaskGraphBuilder,
    default_capability_catalog,
    default_routing,
)
from .planning.task_models import ManufacturingTask, TaskStatus, TaskType
from .planning.workcell_motion import WorkcellMotionPlanningService
from .recovery import FaultRecord, FaultType, RecoveryPolicy, Replanner
from .scheduling import (
    DynamicPriorityScheduler,
    FixedSequenceScheduler,
    ResourceManager,
    ZoneLockManager,
)
from .workcells import (
    AsyncTransferState,
    TransferId,
    TrayOwner,
    TrayRoutePhase,
    TrayRouteState,
    WorkstationId,
    WorkstationState,
)

ROOT = Path(__file__).resolve().parents[1]


class OrderRunStatus(str, Enum):
    QUEUED = "QUEUED"
    RELEASED = "RELEASED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    CANCELLED = "CANCELLED"


@dataclass(slots=True)
class OrderQueueEntry:
    plan: ProcessPlan
    graph_task_ids: tuple[str, ...]
    status: OrderRunStatus = OrderRunStatus.QUEUED
    urgent: bool = False
    inserted_at: float = 0.0
    released_at: float | None = None
    completed_at: float | None = None
    tray_assignments: dict[str, str] = field(default_factory=dict)
    rack_assignments: dict[str, int] = field(default_factory=dict)
    admitted_unit_ids: set[str] = field(default_factory=set)
    completed_unit_ids: set[str] = field(default_factory=set)

    @property
    def order_id(self) -> str:
        return self.plan.order.order_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "product_id": self.plan.product.product_id,
            "preset": self.plan.product.preset,
            "quantity": self.plan.quantity,
            "priority": self.plan.order.priority,
            "due_time": None if self.plan.order.due_time is None else self.plan.order.due_time.isoformat(),
            "status": self.status.value,
            "urgent": self.urgent,
            "inserted_at": self.inserted_at,
            "released_at": self.released_at,
            "completed_at": self.completed_at,
            "tray_assignments": dict(self.tray_assignments),
            "rack_assignments": dict(self.rack_assignments),
            "admitted_unit_ids": sorted(self.admitted_unit_ids),
            "completed_unit_ids": sorted(self.completed_unit_ids),
            "progress": 0.0,
        }


@dataclass(slots=True)
class PendingManualFault:
    request_id: str
    fault_type: FaultType
    target: str
    source: str
    recoverable: bool
    details: dict[str, Any]
    armed_at: float
    status: str = "ARMED"
    fired_at: float | None = None
    fault_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "fault_type": self.fault_type.value,
            "target": self.target,
            "source": self.source,
            "recoverable": self.recoverable,
            "details": dict(self.details),
            "armed_at": self.armed_at,
            "status": self.status,
            "fired_at": self.fired_at,
            "fault_id": self.fault_id,
        }


@dataclass(slots=True)
class RuntimeFurnaceBatch:
    """One physical furnace cycle shared by one or more compatible orders."""

    batch_id: str
    leader_task_id: str
    member_task_ids: tuple[str, ...]
    order_ids: tuple[str, ...]
    unit_count: int
    recipe: str
    created_at: float
    status: str = "SEALED"
    started_at: float | None = None
    completed_at: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "leader_task_id": self.leader_task_id,
            "member_task_ids": list(self.member_task_ids),
            "order_ids": list(self.order_ids),
            "unit_count": self.unit_count,
            "recipe": self.recipe,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


MANUAL_FAULT_TASK_TYPES: dict[FaultType, frozenset[TaskType]] = {
    FaultType.BRAZING_MISSING: frozenset({TaskType.INSPECT_BRAZING}),
    FaultType.BRAZING_PATH_DEVIATION: frozenset({TaskType.INSPECT_BRAZING}),
    FaultType.FIN_PICK_FAILED: frozenset({TaskType.PICK_FIN}),
    FaultType.FIN_GEOMETRY_FAILED: frozenset({TaskType.INSPECT_FINS}),
    FaultType.ARM_UNAVAILABLE: frozenset(TaskType),
    FaultType.ELEVATOR_TIMEOUT: frozenset({TaskType.MOVE_ELEVATOR}),
    FaultType.FORK_TIMEOUT: frozenset({TaskType.LOAD_RACK_LAYER, TaskType.UNLOAD_RACK_LAYER}),
    FaultType.FURNACE_DOOR_INTERLOCK: frozenset({TaskType.RUN_FURNACE}),
    FaultType.CONTACT_SAFETY_STOP: frozenset(TaskType),
    FaultType.TRAY_STATE_INCONSISTENT: frozenset(
        {
            TaskType.TRANSFER_TRAY_OUT,
            TaskType.MOVE_ELEVATOR,
            TaskType.LOAD_RACK_LAYER,
            TaskType.UNLOAD_RACK_LAYER,
        }
    ),
}


class ManufacturingRuntime:
    """Tick-driven scheduler with online recovery and no MuJoCo dependency."""

    def __init__(
        self,
        *,
        scheduler_mode: str = "dynamic",
        scheduler_config: SchedulerConfig | None = None,
        resource_config_path: str | Path | None = None,
        skill_registry: SkillRegistry | None = None,
        context: Any = None,
        max_wip_units: int = 3,
        flexible_cell: bool = False,
        enable_motion_planning: bool | None = None,
        execution_profile: LineExecutionProfile | None = None,
        track_changeover: bool = False,
        teaching_baseline: TeachingBaseline | None = None,
        dispatch_guard: Callable[[ManufacturingTask], tuple[bool, str]] | None = None,
        external_batch_controller: bool = False,
    ) -> None:
        self.config = scheduler_config or load_scheduler_config(ROOT / "config" / "scheduler.yaml")
        resources, zones = load_resource_config(resource_config_path or ROOT / "config" / "resources.yaml")
        self.resources = ResourceManager(resources)
        self.zones = ZoneLockManager(zones)
        self.scheduler_mode = self._normalize_scheduler_mode(scheduler_mode)
        self.scheduler = self._make_scheduler(self.scheduler_mode)
        self.graph = TaskGraph()
        self.events = EventBus()
        self.recovery = RecoveryPolicy()
        self.replanner = Replanner()
        self.registry = skill_registry or self._default_registry()
        self.executor = SkillExecutor(self.registry, context=context)
        self.orders: dict[str, OrderQueueEntry] = {}
        self.faults: dict[str, FaultRecord] = {}
        self.scenario: FaultScenario | None = None
        self.max_wip_units = int(max_wip_units)
        self.flexible_cell = bool(flexible_cell)
        self.motion_planning_enabled = (
            self.flexible_cell if enable_motion_planning is None else bool(enable_motion_planning)
        )
        # Step B: which resources may realise a capability depends on what this
        # line's actors can physically execute, not just on capability data.
        # V1's fin skills are implemented against Arm1's welds, so Arm3 is an
        # inspection-only arm here; V2 supplies its own profile.
        self.execution_profile = execution_profile or V1_SHALLOW_U_PROFILE
        # Step D: changeover state persists across orders, which is what makes
        # setup time sequence-dependent rather than a per-order constant.
        self.track_changeover = bool(track_changeover)
        self.installed_fixture = FixtureConfiguration()
        self.setup_matrix = SetupTimeMatrix(durations=dict(LEGACY_DURATIONS))
        self.teaching_baseline = teaching_baseline or PLACEHOLDER_TEACHING_BASELINE
        self.dispatch_guard = dispatch_guard
        self.external_batch_controller = bool(external_batch_controller)
        self._unit_configurations: dict[str, FixtureConfiguration] = {}
        self.changeover_log: list[dict[str, Any]] = []
        self.motion_planning = WorkcellMotionPlanningService(seed=0) if self.motion_planning_enabled else None
        self.workstations: dict[WorkstationId, WorkstationState] = {}
        self.transfers: dict[TransferId, AsyncTransferState] = {}
        self.tray_routes: dict[str, TrayRouteState] = {}
        self._reset_flexible_cell_state()
        self.paused = False
        self.stopped = False
        self.started_at: float | None = None
        self.last_tick: float | None = None
        self.last_execution_tick: float | None = None
        self.paused_at: float | None = None
        self.tick_count = 0
        self.assignment_history: list[dict[str, Any]] = []
        self.unit_dispositions: dict[str, str] = {}
        self.last_error = ""
        self._fault_sequence = 0
        self._manual_fault_sequence = 0
        self.manual_fault_requests: dict[str, PendingManualFault] = {}
        self._seen_ready: set[str] = set()
        self._resource_recover_at: dict[str, float] = {}
        self._arm_busy_seconds = {"ARM1": 0.0, "ARM2": 0.0, "ARM3": 0.0}
        self._parallel_arm_seconds = 0.0
        self._max_parallel_arms = 0
        self._max_parallel_tasks = 0
        # PICK and PLACE are separate UI/DAG milestones, but the physical
        # payload makes the pair non-preemptible on Arm1.
        self._arm1_payload_handoff: dict[str, str] | None = None
        self.furnace_batches: dict[str, RuntimeFurnaceBatch] = {}
        self._furnace_task_batches: dict[str, str] = {}
        self._furnace_batch_sequence = 0

    @staticmethod
    def _normalize_scheduler_mode(value: str) -> str:
        mode = str(value).strip().upper()
        aliases = {"FIXED": "FIXED_SEQUENCE", "DYNAMIC": "DYNAMIC_PRIORITY"}
        mode = aliases.get(mode, mode)
        if mode not in {"FIXED_SEQUENCE", "DYNAMIC_PRIORITY"}:
            raise ValueError("scheduler mode must be fixed or dynamic")
        return mode

    def _make_scheduler(self, mode: str):
        if mode == "FIXED_SEQUENCE":
            return FixedSequenceScheduler()
        return DynamicPriorityScheduler(
            weights=self.config.weights,
            max_assignments_per_tick=self.config.max_assignments_per_tick,
            allow_parallel_tasks=self.config.allow_parallel_tasks,
        )

    @staticmethod
    def _default_registry() -> SkillRegistry:
        registry = SkillRegistry()
        for task_type in TaskType:
            # One stateful skill instance per execution is required for real
            # V2 alternative resources: Arm1 and Arm3 may both install a fin
            # during the same tick.  A singleton per task type serialised that
            # legal parallelism even when the scheduler selected both tasks.
            registry.register_factory(task_type, TimedSkill)
        return registry

    def set_scheduler(self, mode: str) -> None:
        normalized = self._normalize_scheduler_mode(mode)
        if self.executor.active:
            raise RuntimeError("cannot change scheduler while skills are running")
        self.scheduler_mode = normalized
        self.scheduler = self._make_scheduler(normalized)

    def set_skill_registry(self, registry: SkillRegistry) -> None:
        """Replace the executor backend only while no task is active."""

        if self.executor.active:
            raise RuntimeError("cannot change skill registry while skills are running")
        self.registry = registry
        self.executor.registry = registry

    def submit_plan(self, plan: ProcessPlan, *, urgent: bool = False, now: float = 0.0) -> OrderQueueEntry:
        order_id = plan.order.order_id
        if order_id in self.orders:
            raise ValueError(f"duplicate order id: {order_id}")
        builder = ProcessPlanTaskGraphBuilder(
            flexible_cell=self.flexible_cell,
            catalog=default_capability_catalog(),
            routing=default_routing(),
            resources=self.resources.states.values(),
            profile=self.execution_profile,
            track_changeover=self.track_changeover,
            # Carrying the installed configuration into the next order is what
            # makes setup time sequence-dependent (FJSP-SDST) instead of a
            # per-order constant.
            fixture_state=self.installed_fixture,
        )
        new_graph = builder.build(plan)
        if self.track_changeover:
            self.installed_fixture = builder.fixture_state
            self.changeover_log.extend(builder.changeover_plans)
            configuration = required_configuration(plan)
            for assignment in plan.rack_assignments:
                unit_id = f"{plan.order.order_id}_UNIT_{assignment.unit_index + 1:02d}"
                self._unit_configurations[unit_id] = configuration
        task_ids: list[str] = []
        for task in new_graph.topological_order():
            # Queue admission, rather than dependencies, controls WIP release.
            if task.status is TaskStatus.READY:
                task.status = TaskStatus.PENDING
                task.ready_at = None
            task.payload.setdefault("queue_held", True)
            self.graph.add_task(task)
            task_ids.append(task.task_id)
        self.graph.validate_acyclic()
        entry = OrderQueueEntry(plan, tuple(task_ids), urgent=bool(urgent), inserted_at=float(now))
        self.orders[order_id] = entry
        self.events.publish(
            EventType.ORDER_RELEASED,
            sim_time=now,
            source="order_queue",
            payload={"order_id": order_id, "queued": True, "urgent": urgent},
        )
        self._admit_orders(now)
        return entry

    def submit_plans(self, plans: Iterable[ProcessPlan], *, now: float = 0.0) -> list[OrderQueueEntry]:
        return [self.submit_plan(plan, now=now) for plan in plans]

    def _active_wip(self) -> int:
        production_trays = set(sorted(self.tray_routes)[: self.max_wip_units])
        return sum(
            route.tray_id in production_trays and route.order_id is not None
            for route in self.tray_routes.values()
        )

    def _record_parallelism(self, dt: float) -> None:
        """Accumulate concurrency measured from active physical skills."""

        active_resources = set(self.executor.active)
        active_arms = active_resources.intersection(self._arm_busy_seconds)
        duration = max(0.0, float(dt))
        for arm_id in active_arms:
            self._arm_busy_seconds[arm_id] += duration
        if len(active_arms) >= 2:
            self._parallel_arm_seconds += duration
        self._max_parallel_arms = max(self._max_parallel_arms, len(active_arms))
        self._max_parallel_tasks = max(self._max_parallel_tasks, len(active_resources))

    def _admit_orders(self, now: float, *, refresh_ready: bool = True) -> None:
        available = self.max_wip_units - self._active_wip()
        used_rack_layers = {
            layer
            for entry in self.orders.values()
            if entry.status
            in {
                OrderRunStatus.RELEASED,
                OrderRunStatus.RUNNING,
                OrderRunStatus.MANUAL_REVIEW,
            }
            for layer in entry.rack_assignments.values()
        }
        # The fourth route is a UI-only spare. Only the first three tray IDs
        # have complete MuJoCo product pools and may enter manufacturing.
        production_trays = set(sorted(self.tray_routes)[: self.max_wip_units])
        free_trays = sorted(
            tray.tray_id
            for tray in self.tray_routes.values()
            if tray.tray_id in production_trays
            and tray.owner is TrayOwner.EMPTY_BUFFER
            and tray.order_id is None
        )
        candidates = sorted(
            (
                (entry, assignment)
                for entry in self.orders.values()
                if entry.status
                in {
                    OrderRunStatus.QUEUED,
                    OrderRunStatus.RELEASED,
                    OrderRunStatus.RUNNING,
                }
                for assignment in entry.plan.rack_assignments
                if f"{entry.order_id}_UNIT_{assignment.unit_index + 1:02d}" not in entry.admitted_unit_ids
            ),
            key=lambda item: (
                -int(item[0].urgent),
                -item[0].plan.order.priority,
                item[0].inserted_at,
                item[0].order_id,
                item[1].unit_index,
            ),
        )
        for entry, assignment in candidates:
            if available <= 0 or not free_trays:
                break
            unit_id = f"{entry.order_id}_UNIT_{assignment.unit_index + 1:02d}"
            physical_tray = free_trays.pop(0)
            preferred = int(assignment.layer_index)
            if preferred not in used_rack_layers:
                selected_layer = preferred
            else:
                selected_layer = next(layer for layer in range(3) if layer not in used_rack_layers)
            used_rack_layers.add(selected_layer)
            entry.tray_assignments[assignment.tray_id] = physical_tray
            entry.rack_assignments[physical_tray] = selected_layer
            entry.admitted_unit_ids.add(unit_id)

            for task_id in entry.graph_task_ids:
                task = self.graph.get(task_id)
                if task.unit_id != unit_id:
                    continue
                task.tray_id = physical_tray
                if "layer_index" in task.payload:
                    task.payload["layer_index"] = selected_layer
                    task.payload["height_m"] = (0.0, 0.151, 0.302)[selected_layer]
                task.payload.pop("queue_held", None)
                if task.status is TaskStatus.BLOCKED and task.failure_reason == "ORDER_QUEUED":
                    task.status = TaskStatus.PENDING
                    task.failure_reason = None

            route = self.tray_routes[physical_tray]
            route.order_id = entry.order_id
            route.product_unit_id = unit_id
            if entry.status is OrderRunStatus.QUEUED:
                entry.status = OrderRunStatus.RELEASED
            if entry.released_at is None:
                entry.released_at = float(now)
            available -= 1
            self.events.publish(
                EventType.ORDER_RELEASED,
                sim_time=now,
                source="runtime",
                payload={
                    "order_id": entry.order_id,
                    "unit_id": unit_id,
                    "tray_id": physical_tray,
                    "queued": False,
                },
            )
        if refresh_ready:
            self._refresh_ready(now)

    def _annotate_changeover_cost(self, ready: list[ManufacturingTask]) -> None:
        """Publish the setup cost each ready task would incur, in seconds.

        A task whose unit needs the fixture already mounted costs nothing; one
        that would force a module swap carries the full sequence-dependent
        setup time.  The dynamic scheduler multiplies this by
        ``SchedulingWeights.product_changeover_cost``, so with several units
        ready it naturally continues with the same fixture family first.
        """

        if not self.track_changeover:
            return
        for task in ready:
            target = self._unit_configurations.get(task.unit_id)
            if target is None:
                continue
            seconds = self.setup_matrix.setup_time(self.installed_fixture, target)
            task.payload["product_changeover_cost"] = seconds
            task.payload["changeover_required"] = seconds > 0.0

    def _refresh_ready(self, now: float) -> list[ManufacturingTask]:
        for task in self.graph:
            if task.payload.get("queue_held") and task.status in {TaskStatus.PENDING, TaskStatus.READY}:
                task.status = TaskStatus.PENDING
        self.graph.refresh_ready(now)
        for task in self.graph:
            if task.payload.get("queue_held") and task.status is TaskStatus.READY:
                task.status = TaskStatus.PENDING
                task.ready_at = None
        paused_recovery_tasks = {
            step.task_id
            for plan in self.recovery.plans.values()
            if plan.status.value in {"PAUSED", "MANUAL_REVIEW"}
            for step in plan.steps
            if step.task_id
        }
        ready = [
            task
            for task in self.graph.get_ready_tasks()
            if not task.payload.get("queue_held") and task.task_id not in paused_recovery_tasks
        ]
        for task in ready:
            if task.task_id not in self._seen_ready:
                self._seen_ready.add(task.task_id)
                self.events.publish(
                    EventType.TASK_READY,
                    sim_time=now,
                    source="task_graph",
                    payload={"task_id": task.task_id, "task_type": task.task_type.value},
                )
        return ready

    def _seal_furnace_batch(
        self,
        tasks: list[ManufacturingTask],
        now: float,
    ) -> RuntimeFurnaceBatch:
        """Bind compatible logical furnace nodes to one physical cycle."""

        ordered = sorted(
            tasks,
            key=lambda task: (
                -task.priority,
                float("inf") if task.ready_at is None else task.ready_at,
                task.task_id,
            ),
        )
        self._furnace_batch_sequence += 1
        batch_id = f"FURNACE_BATCH_{self._furnace_batch_sequence:04d}"
        leader = ordered[0]
        order_ids = tuple(dict.fromkeys(task.order_id for task in ordered))
        unit_count = sum(int(task.payload.get("batch_units", 1)) for task in ordered)
        batch = RuntimeFurnaceBatch(
            batch_id=batch_id,
            leader_task_id=leader.task_id,
            member_task_ids=tuple(task.task_id for task in ordered),
            order_ids=order_ids,
            unit_count=unit_count,
            recipe=str(leader.payload.get("recipe", "")),
            created_at=float(now),
        )
        self.furnace_batches[batch_id] = batch
        for task in ordered:
            self._furnace_task_batches[task.task_id] = batch_id
            task.payload["furnace_batch_id"] = batch_id
            task.payload["furnace_batch_leader"] = leader.task_id
            task.payload["furnace_batch_members"] = list(batch.member_task_ids)

        # A shared cycle releases every member at the same instant. Preserve
        # the physical top-to-bottom rack unloading order across order
        # boundaries, not merely within each original ProcessPlan.
        unloads: list[ManufacturingTask] = []
        for task in ordered:
            unloads.extend(
                self.graph.get(successor_id)
                for successor_id in task.successors
                if self.graph.get(successor_id).task_type is TaskType.UNLOAD_RACK_LAYER
            )
        unloads = sorted(
            {task.task_id: task for task in unloads}.values(),
            key=lambda task: (-int(task.payload.get("layer_index", 0)), task.task_id),
        )
        for previous, following in zip(unloads, unloads[1:]):
            self.graph.add_dependency(previous.task_id, following.task_id)
        return batch

    def _prepare_furnace_batches(
        self,
        ready: list[ManufacturingTask],
        now: float,
    ) -> list[ManufacturingTask]:
        """Return schedulable tasks while compatible furnace nodes wait/seal."""

        ordinary = [task for task in ready if task.task_type is not TaskType.RUN_FURNACE]
        furnace = [task for task in ready if task.task_type is TaskType.RUN_FURNACE]
        schedulable: list[ManufacturingTask] = []
        unassigned: list[ManufacturingTask] = []
        for task in furnace:
            batch_id = self._furnace_task_batches.get(task.task_id)
            if batch_id is None:
                unassigned.append(task)
                continue
            batch = self.furnace_batches[batch_id]
            if task.task_id == batch.leader_task_id:
                schedulable.append(task)

        if unassigned:
            candidates = sorted(
                unassigned,
                key=lambda task: (
                    -task.priority,
                    float("inf") if task.ready_at is None else task.ready_at,
                    task.task_id,
                ),
            )
            selected: list[ManufacturingTask] = []
            selected_plans: list[ProcessPlan] = []
            units = 0
            maximum = self.config.batching.maximum_units
            for task in candidates:
                plan = self.orders[task.order_id].plan
                task_units = int(task.payload.get("batch_units", 1))
                if units + task_units > maximum:
                    continue
                if selected_plans and not are_process_plans_compatible((*selected_plans, plan)):
                    continue
                selected.append(task)
                selected_plans.append(plan)
                units += task_units
                if units == maximum:
                    break

            oldest_ready = min(
                (task.ready_at if task.ready_at is not None else float(now)) for task in selected
            )
            waited = max(0.0, float(now) - float(oldest_ready))
            selected_ids = {task.task_id for task in selected}
            future_compatible = any(
                candidate.task_type is TaskType.RUN_FURNACE
                and candidate.task_id not in selected_ids
                and candidate.task_id not in self._furnace_task_batches
                and not candidate.payload.get("queue_held")
                and candidate.status
                in {
                    TaskStatus.PENDING,
                    TaskStatus.READY,
                    TaskStatus.RESERVED,
                    TaskStatus.RUNNING,
                }
                and units + int(candidate.payload.get("batch_units", 1)) <= maximum
                and are_process_plans_compatible((*selected_plans, self.orders[candidate.order_id].plan))
                for candidate in self.graph
            )
            selected_order_ids = {task.order_id for task in selected}
            same_order_future_admitted = any(
                candidate.task_type is TaskType.RUN_FURNACE
                and candidate.task_id not in selected_ids
                and candidate.task_id not in self._furnace_task_batches
                and candidate.order_id in selected_order_ids
                and not candidate.payload.get("queue_held")
                and candidate.status
                in {
                    TaskStatus.PENDING,
                    TaskStatus.READY,
                    TaskStatus.RESERVED,
                    TaskStatus.RUNNING,
                }
                for candidate in self.graph
            )
            may_run_partial = (
                self.config.batching.allow_partial_batch
                and not same_order_future_admitted
                and (not future_compatible or waited + 1e-9 >= self.config.batching.max_wait_time)
            )
            if selected and (units == maximum or may_run_partial):
                batch = self._seal_furnace_batch(selected, now)
                schedulable.append(self.graph.get(batch.leader_task_id))
                selected_ids = set(batch.member_task_ids)
                for task in candidates:
                    if task.task_id not in selected_ids:
                        task.payload["planning_blockers"] = ["等待下一兼容炉批"]
            else:
                for task in selected:
                    task.payload["planning_blockers"] = [
                        f"等待兼容炉批：{units}/{maximum}件，已等待{waited:.1f}s"
                    ]
                for task in candidates:
                    if task not in selected:
                        task.payload["planning_blockers"] = ["等待兼容炉批或当前炉批释放"]
        return [*ordinary, *schedulable]

    def _start_furnace_batch(self, task: ManufacturingTask, now: float) -> None:
        batch_id = self._furnace_task_batches.get(task.task_id)
        if batch_id is None:
            return
        batch = self.furnace_batches[batch_id]
        if task.task_id == batch.leader_task_id:
            batch.status = "RUNNING"
            batch.started_at = float(now)

    def _complete_furnace_batch(self, task: ManufacturingTask, now: float) -> None:
        batch_id = self._furnace_task_batches.get(task.task_id)
        if batch_id is None:
            return
        batch = self.furnace_batches[batch_id]
        if task.task_id != batch.leader_task_id:
            return
        for member_id in batch.member_task_ids:
            if member_id == task.task_id:
                continue
            member = self.graph.get(member_id)
            if member.status is not TaskStatus.READY:
                continue
            member.payload["coalesced_physical_cycle"] = task.task_id
            self.graph.mark_succeeded(member.task_id, now)
            self.events.publish(
                EventType.TASK_SUCCEEDED,
                sim_time=now,
                source=batch.batch_id,
                payload={
                    "task_id": member.task_id,
                    "task_type": member.task_type.value,
                    "metrics": {
                        "shared_furnace_batch": batch.batch_id,
                        "physical_cycle_task_id": task.task_id,
                    },
                },
            )
        batch.status = "COMPLETED"
        batch.completed_at = float(now)

    def _complete_task(self, task: ManufacturingTask, metrics: dict[str, Any], now: float) -> None:
        self.graph.mark_succeeded(task.task_id, now)
        if task.task_type is TaskType.RUN_FURNACE:
            self._complete_furnace_batch(task, now)
        if task.task_type is TaskType.PICK_BASE_PLATE:
            self._arm1_payload_handoff = {"unit_id": task.unit_id, "payload": "base_plate"}
        elif task.task_type is TaskType.PICK_FIN:
            self._arm1_payload_handoff = {
                "unit_id": task.unit_id,
                "payload": str(task.payload.get("fin_id", "fin_01")),
            }
        elif task.task_type is TaskType.PLACE_BASE_PLATE:
            if (
                self._arm1_payload_handoff is not None
                and self._arm1_payload_handoff.get("unit_id") == task.unit_id
                and self._arm1_payload_handoff.get("payload") == "base_plate"
            ):
                self._arm1_payload_handoff = None
        elif task.task_type in {TaskType.INSTALL_FIN, TaskType.REINSTALL_FIN}:
            if (
                self._arm1_payload_handoff is not None
                and self._arm1_payload_handoff.get("unit_id") == task.unit_id
                and self._arm1_payload_handoff.get("payload") == str(task.payload.get("fin_id", "fin_01"))
            ):
                self._arm1_payload_handoff = None
        if self.motion_planning is not None:
            self.motion_planning.release_task(task)
        if task.assigned_resource and task.required_tool:
            # Keep the scheduling model aligned with the physical quick-change
            # manager.  The first Arm2 dispense pays the rack pickup cost;
            # compatible following pallets reuse the already mounted gun.
            self.resources.get(task.assigned_resource).current_tool = task.required_tool
        self.resources.release(task.assigned_resource or "", task.task_id, now)
        self.zones.release(task.task_id)
        if task.station_id:
            try:
                workstation = self.workstations[WorkstationId(task.station_id)]
                if workstation.occupied_by == task.task_id:
                    workstation.occupied_by = None
                    workstation.safe_for_transfer = True
            except (KeyError, ValueError):
                pass
        self._advance_cell_state(task, now)
        if task.task_type is TaskType.POST_BRAZE_INSPECTION:
            disposition = str(metrics.get("disposition", "PASS"))
            self.unit_dispositions[task.unit_id] = disposition
            for successor_id in task.successors:
                successor = self.graph.get(successor_id)
                condition = successor.payload.get("condition")
                if condition is not None and condition != disposition:
                    successor.status = TaskStatus.CANCELLED
                    successor.finished_at = float(now)
        elif task.payload.get("condition_passthrough"):
            disposition = self.unit_dispositions.get(task.unit_id, "PASS")
            for successor_id in task.successors:
                successor = self.graph.get(successor_id)
                condition = successor.payload.get("condition")
                if condition is not None and condition != disposition:
                    successor.status = TaskStatus.CANCELLED
                    successor.finished_at = float(now)
        self.events.publish(
            EventType.TASK_SUCCEEDED,
            sim_time=now,
            source=task.assigned_resource or "executor",
            payload={"task_id": task.task_id, "task_type": task.task_type.value, "metrics": metrics},
        )

    @staticmethod
    def _fault_type_for_task(task: ManufacturingTask, code: str) -> FaultType:
        if task.task_type in {TaskType.DISPENSE_BRAZING, TaskType.INSPECT_BRAZING}:
            return FaultType.BRAZING_MISSING
        if task.task_type is TaskType.PICK_FIN:
            return FaultType.FIN_PICK_FAILED
        if task.task_type in {TaskType.INSTALL_FIN, TaskType.INSPECT_FINS}:
            return FaultType.FIN_GEOMETRY_FAILED
        if task.task_type is TaskType.MOVE_ELEVATOR:
            return FaultType.ELEVATOR_TIMEOUT
        if task.task_type in {TaskType.LOAD_RACK_LAYER, TaskType.UNLOAD_RACK_LAYER}:
            return FaultType.FORK_TIMEOUT
        if task.task_type in {TaskType.RUN_FURNACE, TaskType.FURNACE_INTERLOCK_CHECK}:
            return FaultType.FURNACE_DOOR_INTERLOCK
        if "CONTACT" in code.upper():
            return FaultType.CONTACT_SAFETY_STOP
        return FaultType.TRAY_STATE_INCONSISTENT

    def _fail_task(self, task: ManufacturingTask, code: str, metrics: dict[str, Any], now: float) -> None:
        self.graph.mark_failed(task.task_id, code, now)
        if task.task_type is TaskType.RUN_FURNACE:
            batch_id = self._furnace_task_batches.get(task.task_id)
            if batch_id is not None:
                batch = self.furnace_batches[batch_id]
                if batch.leader_task_id == task.task_id:
                    batch.status = "RECOVERING"
        self._abort_cell_state(task)
        if self.motion_planning is not None:
            self.motion_planning.release_task(task, retain_path=False)
        self.resources.release(task.assigned_resource or "", task.task_id, now)
        self.zones.release(task.task_id)
        if task.station_id:
            try:
                workstation = self.workstations[WorkstationId(task.station_id)]
                if workstation.occupied_by == task.task_id:
                    workstation.occupied_by = None
                    workstation.safe_for_transfer = True
            except (KeyError, ValueError):
                pass
        self.events.publish(
            EventType.TASK_FAILED,
            sim_time=now,
            source=task.assigned_resource or "executor",
            payload={"task_id": task.task_id, "failure_code": code, "metrics": metrics},
        )
        fault = self._new_fault(
            self._fault_type_for_task(task, code),
            task.assigned_resource or "executor",
            task.task_id,
            now,
            recoverable=task.retry_count < max(task.retry_limit, 1),
            details={**metrics, "failure_code": code, **task.payload},
        )
        plan = self.recovery.plan(fault, self.graph, now)
        self.events.publish(
            EventType.RECOVERY_PLANNED,
            sim_time=now,
            source="recovery_policy",
            payload=plan.as_dict(),
        )

    def _finish_recovered_furnace_batches(self, now: float) -> None:
        """Release logical members after the shared interlock is recovered."""

        for batch in self.furnace_batches.values():
            if batch.status != "RECOVERING":
                continue
            leader = self.graph.get(batch.leader_task_id)
            if leader.payload.get("recovered"):
                self._complete_furnace_batch(leader, now)

    def _new_fault(
        self,
        fault_type: FaultType | str,
        source: str,
        related_task_id: str | None,
        now: float,
        *,
        recoverable: bool = True,
        details: dict[str, Any] | None = None,
    ) -> FaultRecord:
        self._fault_sequence += 1
        fault = FaultRecord(
            f"FAULT_{self._fault_sequence:04d}",
            fault_type,
            source,
            related_task_id,
            float(now),
            recoverable,
            dict(details or {}),
        )
        self.faults[fault.fault_id] = fault
        return fault

    def inject_fault(
        self,
        fault_type: FaultType | str,
        *,
        source: str,
        related_task_id: str | None = None,
        now: float = 0.0,
        recoverable: bool = True,
        details: dict[str, Any] | None = None,
    ) -> FaultRecord:
        fault = self._new_fault(
            fault_type, source, related_task_id, now, recoverable=recoverable, details=details
        )
        if fault.fault_type is FaultType.ARM_UNAVAILABLE:
            resource_id = str(fault.details.get("resource_id", source)).upper()
            interrupted = self.resources.fault(resource_id, fault.fault_type.value, now)
            if interrupted:
                self.executor.cancel_task(interrupted)
                task = self.graph.get(interrupted)
                task.mark_failed(now, "RESOURCE_FAULTED")
                fault.details["interrupted_task_id"] = interrupted
                self.zones.release(interrupted)
            self.events.publish(
                EventType.RESOURCE_FAULTED,
                sim_time=now,
                source=resource_id,
                payload=fault.as_dict(),
            )
            plan = self.recovery.plan(fault, self.graph, now)
            plan.status = plan.status.__class__.RUNNING
            duration = fault.details.get("duration")
            if duration is not None:
                self._resource_recover_at[resource_id] = float(now) + max(0.0, float(duration))
        elif fault.fault_type is FaultType.RACK_LAYER_UNAVAILABLE:
            layer = int(fault.details.get("layer_id", -1))
            result = self.replanner.reassign_rack_layer(
                self.graph,
                layer,
                (0, 1, 2),
                unit_id=fault.details.get("unit_id"),
            )
            plan = self.recovery.plan(fault, self.graph, now)
            plan.status = plan.status.__class__.SUCCEEDED
            plan.completed_at = float(now)
            fault.recovered = True
            self.events.publish(
                EventType.RACK_LAYER_UNAVAILABLE,
                sim_time=now,
                source=source,
                payload=result.as_dict(),
            )
        elif related_task_id:
            task = self.graph.get(related_task_id)
            if not task.status.terminal:
                if task.status is TaskStatus.RUNNING:
                    self.executor.cancel_task(task.task_id)
                    self.resources.release(task.assigned_resource or "", task.task_id, now)
                    self.zones.release(task.task_id)
                self.graph.mark_failed(task.task_id, fault.fault_type.value, now)
                if task.task_type is TaskType.RUN_FURNACE:
                    batch_id = self._furnace_task_batches.get(task.task_id)
                    if batch_id is not None:
                        batch = self.furnace_batches[batch_id]
                        if batch.leader_task_id == task.task_id:
                            batch.status = "RECOVERING"
            plan = self.recovery.plan(fault, self.graph, now)
            self.events.publish(
                EventType.RECOVERY_PLANNED,
                sim_time=now,
                source="recovery_policy",
                payload=plan.as_dict(),
            )
        return fault

    @staticmethod
    def _manual_target_matches(request: PendingManualFault, task: ManufacturingTask) -> bool:
        if not request.target:
            return True
        if request.target.startswith("fin_"):
            return request.target == task.payload.get("fin_id") or request.target in task.payload.get(
                "fin_ids", ()
            )
        if request.target.startswith("slot_"):
            return any(
                str(path_id) == request.target or str(path_id).startswith(f"{request.target}_")
                for path_id in task.payload.get("path_ids", ())
            )
        if request.target in {"ARM1", "ARM2", "ARM3"}:
            return request.target == task.assigned_resource or request.target in task.eligible_resources
        if request.target.isdigit() and "layer_index" in task.payload:
            return int(request.target) == int(task.payload["layer_index"])
        return True

    def arm_manual_fault(
        self,
        fault_type: FaultType | str,
        *,
        target: str = "",
        source: str = "operator_ui",
        now: float = 0.0,
        recoverable: bool = True,
        details: dict[str, Any] | None = None,
    ) -> PendingManualFault:
        kind = FaultType(fault_type)
        self._manual_fault_sequence += 1
        request = PendingManualFault(
            request_id=f"MANUAL_{self._manual_fault_sequence:04d}",
            fault_type=kind,
            target=str(target),
            source=str(source),
            recoverable=bool(recoverable),
            details=dict(details or {}),
            armed_at=float(now),
        )
        self.manual_fault_requests[request.request_id] = request
        if kind is FaultType.RACK_LAYER_UNAVAILABLE:
            fault = self.inject_fault(
                kind,
                source=source,
                now=now,
                recoverable=recoverable,
                details=request.details,
            )
            request.status = "FIRED"
            request.fired_at = float(now)
            request.fault_id = fault.fault_id
        return request

    def _fire_pending_manual_faults(self, now: float) -> None:
        running = [task for task in self.graph if task.status is TaskStatus.RUNNING]
        for request in self.manual_fault_requests.values():
            if request.status != "ARMED":
                continue
            compatible = MANUAL_FAULT_TASK_TYPES.get(request.fault_type, frozenset())
            task = next(
                (
                    candidate
                    for candidate in running
                    if candidate.task_type in compatible and self._manual_target_matches(request, candidate)
                ),
                None,
            )
            if task is None:
                continue
            details = dict(request.details)
            if request.target.startswith("slot_"):
                matching_paths = [
                    str(path_id)
                    for path_id in task.payload.get("path_ids", ())
                    if str(path_id) == request.target or str(path_id).startswith(f"{request.target}_")
                ]
                if matching_paths:
                    details["path_ids"] = matching_paths
            fault = self.inject_fault(
                request.fault_type,
                source=request.source,
                related_task_id=task.task_id,
                now=now,
                recoverable=request.recoverable,
                details=details,
            )
            request.status = "FIRED"
            request.fired_at = float(now)
            request.fault_id = fault.fault_id

    def recover_resource(self, resource_id: str, now: float) -> None:
        resource = str(resource_id).upper()
        self.resources.recover(resource, now)
        self.zones.release_resource(resource)
        self._resource_recover_at.pop(resource, None)
        for fault in self.faults.values():
            if (
                fault.fault_type is FaultType.ARM_UNAVAILABLE
                and str(fault.details.get("resource_id", fault.source)).upper() == resource
                and not fault.recovered
            ):
                fault.recovered = True
                interrupted = fault.details.get("interrupted_task_id")
                if interrupted and interrupted in self.graph.tasks:
                    task = self.graph.get(str(interrupted))
                    if not task.prepare_retry(now):
                        task.status = TaskStatus.PENDING
                        task.assigned_resource = None
                        task.failure_reason = None
                        task.finished_at = None
                if fault.recovery_id and fault.recovery_id in self.recovery.plans:
                    plan = self.recovery.plans[fault.recovery_id]
                    plan.status = plan.status.__class__.SUCCEEDED
                    plan.completed_at = float(now)
        self.replanner.replan_ready_set(self.graph, "RESOURCE_RECOVERED", now)
        self.events.publish(
            EventType.RESOURCE_RECOVERED,
            sim_time=now,
            source=str(resource_id).upper(),
            payload={"resource_id": str(resource_id).upper()},
        )

    def _apply_scenario(self, now: float) -> None:
        for resource_id, recover_at in tuple(self._resource_recover_at.items()):
            if now >= recover_at:
                self.recover_resource(resource_id, now)
        if self.scenario is None:
            return
        succeeded = [event for event in self.events.history if event.event_type is EventType.TASK_SUCCEEDED]
        for item in self.scenario.faults:
            if item.fired:
                continue
            trigger = item.trigger
            due = trigger.sim_time is not None and now >= trigger.sim_time
            related_task: str | None = None
            if trigger.after_task_type:
                related = next(
                    (
                        event
                        for event in reversed(succeeded)
                        if event.payload.get("task_type") == trigger.after_task_type
                        and (
                            trigger.unit_id is None
                            or self.graph.get(str(event.payload["task_id"])).unit_id == trigger.unit_id
                        )
                    ),
                    None,
                )
                due = related is not None
                if related is not None:
                    source_task = self.graph.get(str(related.payload["task_id"]))
                    related_task = next(
                        (
                            task_id
                            for task_id in source_task.successors
                            if self.graph.get(task_id).task_type
                            in {TaskType.INSPECT_BRAZING, TaskType.INSPECT_FINS}
                        ),
                        source_task.task_id,
                    )
            if not due:
                continue
            item.fired = True
            source = str(item.payload.get("resource_id", "scenario"))
            self.inject_fault(
                item.fault_type,
                source=source,
                related_task_id=related_task,
                now=now,
                details=dict(item.payload),
            )

    def set_fault_scenario(self, scenario: FaultScenario | None) -> None:
        self.scenario = scenario

    def advance_active_skills(self, now: float) -> None:
        """Advance running physical skills without rescanning the DAG.

        Scheduling remains rate-limited, but a robot trajectory must be
        sampled at the MuJoCo control rate.  Decoupling these two clocks
        removes the former hold-then-jump motion without paying for resource
        scoring and READY scans on every physics step.
        """

        timestamp = float(now)
        if self.stopped or self.paused:
            return
        dt = 0.0 if self.last_execution_tick is None else max(0.0, timestamp - self.last_execution_tick)
        self.last_execution_tick = timestamp
        self._record_parallelism(dt)
        for task_id, result in self.executor.update_task(dt, timestamp).items():
            task = self.graph.get(task_id)
            if result.succeeded:
                metrics = dict(result.metrics)
                if result.completion_evidence is not None:
                    metrics["physical_completion"] = result.completion_evidence.as_dict()
                self._complete_task(task, metrics, timestamp)
            elif result.failed:
                self._fail_task(
                    task,
                    result.failure_code or "SKILL_FAILED",
                    result.metrics,
                    timestamp,
                )

    def tick(self, now: float, *, poll_executor: bool = True) -> None:
        timestamp = float(now)
        if self.stopped or self.paused:
            return
        if self.started_at is None:
            self.started_at = timestamp
        self.last_tick = timestamp
        self.tick_count += 1
        self._fire_pending_manual_faults(timestamp)
        if poll_executor:
            self.advance_active_skills(timestamp)
        self.recovery.update(self.graph, timestamp)
        self._finish_recovered_furnace_batches(timestamp)
        for plan in self.recovery.plans.values():
            if plan.status.value == "SUCCEEDED":
                fault = self.faults.get(plan.fault_id)
                if fault is not None:
                    fault.recovered = True
        self._apply_scenario(timestamp)
        self._update_order_status(timestamp)
        if self.terminal:
            for request in self.manual_fault_requests.values():
                if request.status == "ARMED":
                    request.status = "MISSED"
        # Admission and order completion used to trigger two additional full
        # DAG scans before the scheduler performed this authoritative refresh.
        self._admit_orders(timestamp, refresh_ready=False)
        ready = self._refresh_ready(timestamp)
        if self.flexible_cell:
            ready = [task for task in ready if self._cell_task_available(task)]
        if self.dispatch_guard is not None:
            guarded: list[ManufacturingTask] = []
            for task in ready:
                allowed, reason = self.dispatch_guard(task)
                if allowed:
                    task.payload.pop("dispatch_blocker", None)
                    guarded.append(task)
                else:
                    task.payload["dispatch_blocker"] = str(reason)
            ready = guarded
        if not self.external_batch_controller:
            ready = self._prepare_furnace_batches(ready, timestamp)
        occupied = {zone for zone, lease in self.zones.snapshot().items() if lease is not None}
        # Step D: make the sequence-dependent setup cost visible to the
        # scheduler.  ``SchedulingWeights.product_changeover_cost`` already
        # existed but nothing ever populated it, so setup was effectively free.
        self._annotate_changeover_cost(ready)
        assignments = self.scheduler.select_assignments(
            ready,
            self.resources.states,
            {"occupied_zones": occupied},
            timestamp,
        )
        for assignment in assignments:
            task = self.graph.get(assignment.task_id)
            if self.motion_planning is not None and assignment.resource_id in {
                "ARM1",
                "ARM2",
                "ARM3",
            }:
                decision = self.motion_planning.prepare(
                    task,
                    assignment.resource_id,
                    timestamp,
                    context=self.executor.context,
                )
                if decision.path is None:
                    task.payload["planning_blockers"] = [decision.blocker or "运动规划失败"]
                    continue
                if decision.start_time > timestamp + 1e-9:
                    task.payload["planning_blockers"] = [f"时空预约等待至 {decision.start_time:.3f}s"]
                    continue
                task.payload.pop("planning_blockers", None)
            if not self.zones.acquire(task.task_id, assignment.resource_id, task.required_zones, timestamp):
                continue
            if not self.resources.reserve(
                assignment.resource_id, task.task_id, timestamp, task.required_zones
            ):
                self.zones.release(task.task_id)
                continue
            try:
                task.reserve(assignment.resource_id)
                self.events.publish(
                    EventType.TASK_RESERVED,
                    sim_time=timestamp,
                    source=assignment.resource_id,
                    payload=assignment.as_dict(),
                )
                self._begin_cell_state(task, timestamp)
                self.executor.start_task(task, assignment.resource_id, now=timestamp)
                task.mark_running(timestamp)
                if task.task_type is TaskType.RUN_FURNACE:
                    self._start_furnace_batch(task, timestamp)
                if task.station_id:
                    try:
                        workstation = self.workstations[WorkstationId(task.station_id)]
                        workstation.occupied_by = task.task_id
                        workstation.safe_for_transfer = False
                    except (KeyError, ValueError):
                        pass
                self.resources.mark_busy(assignment.resource_id, task.task_id, timestamp)
                self.assignment_history.append({"sim_time": timestamp, **assignment.as_dict()})
                self.events.publish(
                    EventType.TASK_STARTED,
                    sim_time=timestamp,
                    source=assignment.resource_id,
                    payload={"task_id": task.task_id, "task_type": task.task_type.value},
                )
            except Exception as exc:
                self._abort_cell_state(task)
                if self.motion_planning is not None:
                    self.motion_planning.release_task(task, retain_path=False)
                self.resources.release(assignment.resource_id, task.task_id, timestamp)
                self.zones.release(task.task_id)
                task.status = TaskStatus.READY
                task.assigned_resource = None
                self.last_error = str(exc)

    def _cell_task_available(self, task: ManufacturingTask) -> bool:
        """Keep a READY task queued until its physical pallet/station is valid."""

        if task.task_type is TaskType.PREPARE_FIN_TOOL:
            # Batch Arm1's work by tool.  Once one pallet leaves S1, its
            # PREPARE_FIN_TOOL branch becomes READY while the following
            # admitted pallet may still be indexing into the base-loading
            # station.  Dispatching the ready branch immediately produces the
            # wasteful suction -> gripper -> suction sequence observed with
            # adjacent orders.  Keep the suction cup mounted until every
            # currently released base pickup/place pair has finished; Arm2
            # and Arm3 remain free to process earlier pallets meanwhile.
            unfinished_base_tasks = [
                candidate
                for candidate in self.graph
                if candidate.task_type in {TaskType.PICK_BASE_PLATE, TaskType.PLACE_BASE_PLATE}
                and not candidate.payload.get("queue_held")
                and candidate.status
                not in {
                    TaskStatus.SUCCEEDED,
                    TaskStatus.FAILED,
                    TaskStatus.BLOCKED,
                    TaskStatus.CANCELLED,
                }
            ]
            if unfinished_base_tasks:
                task.payload["planning_blockers"] = ["保持Arm1吸盘，等待当前在制订单全部完成基板取放"]
                return False
            task.payload.pop("planning_blockers", None)

        handoff = self._arm1_payload_handoff
        if handoff is not None and "ARM1" in task.eligible_resources:
            same_unit = task.unit_id == handoff.get("unit_id")
            payload = handoff.get("payload")
            if payload == "base_plate":
                allowed = same_unit and task.task_type is TaskType.PLACE_BASE_PLATE
                blocker = "等待Arm1完成已吸附基板的连续放置"
            else:
                allowed = (
                    same_unit
                    and str(task.payload.get("fin_id", "")) == str(payload)
                    and task.task_type in {TaskType.INSTALL_FIN, TaskType.REINSTALL_FIN}
                )
                blocker = f"等待Arm1完成已夹持{payload}的连续安装"
            if not allowed:
                task.payload["planning_blockers"] = [blocker]
                return False
            task.payload.pop("planning_blockers", None)

        if task.tray_id is None or task.tray_id not in self.tray_routes:
            return True
        route = self.tray_routes[task.tray_id]
        if task.task_type is TaskType.INDEX_EMPTY_TRAY:
            station = self.workstations[WorkstationId.S1_BASE_LOADING]
            available = station.tray_id in {None, task.tray_id}
            reason = "S1仍有在制托盘"
        elif task.task_type is TaskType.PICK_FIN:
            # Keep the useful pickup/comb-install overlap, but never pre-grasp
            # a fin while S3 (or its inbound slide) belongs to another pallet.
            # A held fin makes Arm1 non-preemptible and can otherwise form a
            # circular wait with the next order's base-loading task.
            station = self.workstations[WorkstationId.S3_FIN_ASSEMBLY]
            inbound = self.transfers[TransferId.S2B_S3]
            available = station.tray_id in {None, task.tray_id} and inbound.tray_id in {None, task.tray_id}
            reason = "S3正服务另一托盘，禁止提前夹持翅片"
        elif task.station_id:
            try:
                station_id = WorkstationId(task.station_id)
            except ValueError:
                return True
            station = self.workstations[station_id]
            available = route.station_id is station_id and station.tray_id == task.tray_id
            reason = f"托盘尚未到达{station_id.value}"
        else:
            binding = self._transfer_binding(task.task_type)
            if binding is None:
                return True
            transfer = self.transfers[binding[0]]
            source = self.workstations[transfer.source]
            destination = self.workstations[transfer.target]
            available = (
                source.tray_id == task.tray_id
                and destination.tray_id in {None, task.tray_id}
                and transfer.tray_id in {None, task.tray_id}
            )
            reason = f"{binding[0].value}等待源托盘或目标工位释放"
        if available:
            task.payload.pop("station_blocker", None)
        else:
            task.payload["station_blocker"] = reason
        return available

    def _update_order_status(self, now: float) -> None:
        for entry in self.orders.values():
            if entry.status not in {OrderRunStatus.RELEASED, OrderRunStatus.RUNNING}:
                continue
            tasks = [self.graph.get(task_id) for task_id in entry.graph_task_ids]
            route_tasks = [
                task
                for task in tasks
                if task.task_type in {TaskType.ROUTE_PASS, TaskType.ROUTE_REWORK, TaskType.ROUTE_SCRAP}
                and task.status is TaskStatus.SUCCEEDED
            ]
            for route_task in route_tasks:
                if route_task.unit_id in entry.completed_unit_ids:
                    continue
                entry.completed_unit_ids.add(route_task.unit_id)
                tray_id = route_task.tray_id
                route = self.tray_routes.get(str(tray_id))
                if route is not None and route.order_id == entry.order_id:
                    for workstation in self.workstations.values():
                        if workstation.tray_id == route.tray_id:
                            workstation.tray_id = None
                    route.phase = TrayRoutePhase.EMPTY_BUFFER
                    route.owner = TrayOwner.EMPTY_BUFFER
                    route.station_id = None
                    route.order_id = None
                    route.product_unit_id = None
                    route.mold_name = None
                    route.comb_name = None
                    route.press_locked = False
                    route.last_transition_at = float(now)
                    entry.rack_assignments.pop(route.tray_id, None)
            if any(task.status in {TaskStatus.RUNNING, TaskStatus.SUCCEEDED} for task in tasks):
                entry.status = OrderRunStatus.RUNNING
            if all(task.status.terminal for task in tasks):
                if any(
                    task.status in {TaskStatus.FAILED, TaskStatus.BLOCKED}
                    and not task.payload.get("recovered")
                    for task in tasks
                ):
                    entry.status = OrderRunStatus.MANUAL_REVIEW
                else:
                    entry.status = OrderRunStatus.COMPLETED
                entry.completed_at = float(now)
                if entry.status is OrderRunStatus.COMPLETED:
                    # Normally each product unit releases its pallet as soon
                    # as its finished-goods route succeeds. Keep this fallback
                    # for legacy/custom graphs without an explicit route node.
                    for route in self.tray_routes.values():
                        if route.order_id != entry.order_id:
                            continue
                        for workstation in self.workstations.values():
                            if workstation.tray_id == route.tray_id:
                                workstation.tray_id = None
                        route.phase = TrayRoutePhase.EMPTY_BUFFER
                        route.owner = TrayOwner.EMPTY_BUFFER
                        route.station_id = None
                        route.order_id = None
                        route.product_unit_id = None
                        route.mold_name = None
                        route.comb_name = None
                        route.press_locked = False
                        route.last_transition_at = float(now)
                self.events.publish(
                    EventType.ORDER_COMPLETED,
                    sim_time=now,
                    source="runtime",
                    payload={"order_id": entry.order_id, "status": entry.status.value},
                )

    def pause(self, now: float) -> None:
        if self.paused:
            return
        self.paused = True
        self.paused_at = float(now)

    def resume(self, now: float) -> None:
        if not self.paused:
            return
        offset = max(0.0, float(now) - (self.paused_at or float(now)))
        for execution in self.executor.active.values():
            if hasattr(execution.skill, "finished_at"):
                execution.skill.finished_at += offset
            execution.last_update += offset
        self.last_tick = float(now)
        self.last_execution_tick = float(now)
        self.paused = False
        self.paused_at = None

    def stop(self, now: float) -> None:
        self.executor.reset()
        for task in self.graph:
            if task.status in {
                TaskStatus.PENDING,
                TaskStatus.READY,
                TaskStatus.RESERVED,
                TaskStatus.RUNNING,
                TaskStatus.RETRY_WAIT,
            }:
                task.status = TaskStatus.CANCELLED
                task.finished_at = float(now)
        self.resources.reset(now)
        if "ARM2" in self.resources.states:
            self.resources.get("ARM2").current_tool = "brazing_dispenser"
        self.zones.reset()
        self._arm1_payload_handoff = None
        self.stopped = True
        self.events.publish(EventType.EMERGENCY_STOP, sim_time=now, source="operator")

    def reset(self, now: float = 0.0) -> None:
        self.executor.reset()
        self.graph = TaskGraph()
        self.orders.clear()
        self.faults.clear()
        self.recovery.reset()
        self.replanner.reset()
        self.resources.reset(now)
        if "ARM2" in self.resources.states:
            self.resources.get("ARM2").current_tool = "brazing_dispenser"
        self.zones.reset()
        self.events.reset()
        self.assignment_history.clear()
        self.unit_dispositions.clear()
        self.paused = False
        self.stopped = False
        self.started_at = None
        self.last_tick = None
        self.last_execution_tick = None
        self.paused_at = None
        self.tick_count = 0
        self.last_error = ""
        self._seen_ready.clear()
        self._resource_recover_at.clear()
        self.manual_fault_requests.clear()
        self._manual_fault_sequence = 0
        self._arm_busy_seconds = {"ARM1": 0.0, "ARM2": 0.0, "ARM3": 0.0}
        self._parallel_arm_seconds = 0.0
        self._max_parallel_arms = 0
        self._max_parallel_tasks = 0
        self._arm1_payload_handoff = None
        self.furnace_batches.clear()
        self._furnace_task_batches.clear()
        self._furnace_batch_sequence = 0
        if self.motion_planning is not None:
            self.motion_planning.reset()
        # A reset returns the cell to a cold line: nothing is mounted, so the
        # next order pays a full installation.
        self.installed_fixture = FixtureConfiguration()
        self._unit_configurations.clear()
        self.changeover_log.clear()
        self._reset_flexible_cell_state()

    def _reset_flexible_cell_state(self) -> None:
        """Reset the four-station asynchronous line and its spare pallet."""

        self.tray_routes = {
            "tray_01": TrayRouteState(
                "tray_01",
                owner=TrayOwner.EMPTY_BUFFER,
            ),
            "tray_02": TrayRouteState(
                "tray_02",
                owner=TrayOwner.EMPTY_BUFFER,
            ),
            "tray_03": TrayRouteState("tray_03", owner=TrayOwner.EMPTY_BUFFER),
            "tray_04": TrayRouteState("tray_04", owner=TrayOwner.EMPTY_BUFFER),
        }
        self.workstations = {
            WorkstationId.S1_BASE_LOADING: WorkstationState(
                WorkstationId.S1_BASE_LOADING,
                SHALLOW_U_LAYOUT.station_s1_xy,
                ("BASE_LOADING", "BASE_ALIGNMENT"),
            ),
            WorkstationId.S2A_DISPENSING: WorkstationState(
                WorkstationId.S2A_DISPENSING,
                SHALLOW_U_LAYOUT.station_s2a_xy,
                ("BRAZING", "BRAZING_REWORK"),
            ),
            WorkstationId.S2B_MATERIAL_INSPECTION: WorkstationState(
                WorkstationId.S2B_MATERIAL_INSPECTION,
                SHALLOW_U_LAYOUT.station_s2b_xy,
                ("MATERIAL_INSPECTION",),
            ),
            WorkstationId.S3_FIN_ASSEMBLY: WorkstationState(
                WorkstationId.S3_FIN_ASSEMBLY,
                SHALLOW_U_LAYOUT.station_s3_xy,
                ("FIN_ASSEMBLY", "FIN_INSPECTION", "FIXTURE_LOCKING"),
            ),
            WorkstationId.RACK_INFEED: WorkstationState(
                WorkstationId.RACK_INFEED,
                SHALLOW_U_LAYOUT.rack_infeed_xy,
                ("RACK_LOADING",),
            ),
        }
        self.transfers = {
            TransferId.S1_S2A: AsyncTransferState(
                TransferId.S1_S2A,
                WorkstationId.S1_BASE_LOADING,
                WorkstationId.S2A_DISPENSING,
            ),
            TransferId.S2A_S2B: AsyncTransferState(
                TransferId.S2A_S2B,
                WorkstationId.S2A_DISPENSING,
                WorkstationId.S2B_MATERIAL_INSPECTION,
            ),
            TransferId.S2B_S3: AsyncTransferState(
                TransferId.S2B_S3,
                WorkstationId.S2B_MATERIAL_INSPECTION,
                WorkstationId.S3_FIN_ASSEMBLY,
            ),
            TransferId.S3_RACK: AsyncTransferState(
                TransferId.S3_RACK,
                WorkstationId.S3_FIN_ASSEMBLY,
                WorkstationId.RACK_INFEED,
            ),
        }

    @staticmethod
    def _transfer_binding(
        task_type: TaskType,
    ) -> tuple[TransferId, TrayOwner] | None:
        return {
            TaskType.TRANSFER_S1_S2A: (
                TransferId.S1_S2A,
                TrayOwner.TRANSFER_S1_S2A,
            ),
            TaskType.TRANSFER_S2A_S2B: (
                TransferId.S2A_S2B,
                TrayOwner.TRANSFER_S2A_S2B,
            ),
            TaskType.TRANSFER_S2B_S3: (
                TransferId.S2B_S3,
                TrayOwner.TRANSFER_S2B_S3,
            ),
            TaskType.TRANSFER_S3_RACK: (
                TransferId.S3_RACK,
                TrayOwner.TRANSFER_S3_RACK,
            ),
        }.get(task_type)

    def _begin_cell_state(self, task: ManufacturingTask, now: float) -> None:
        """Atomically hand one pallet from a station to its transfer slide."""

        binding = self._transfer_binding(task.task_type)
        if binding is None:
            return
        if task.tray_id is None or task.tray_id not in self.tray_routes:
            raise RuntimeError(f"{task.task_id}没有有效物理托盘")
        transfer_id, owner = binding
        transfer = self.transfers[transfer_id]
        route = self.tray_routes[task.tray_id]
        source = self.workstations[transfer.source]
        destination = self.workstations[transfer.target]
        if source.tray_id != route.tray_id:
            raise RuntimeError(
                f"{transfer_id.value}源工位托盘归属不一致：" f"{source.tray_id or '空'} / {route.tray_id}"
            )
        if destination.tray_id not in {None, route.tray_id}:
            raise RuntimeError(f"{transfer.target.value}仍被{destination.tray_id}占用")
        transfer.reserve(route.tray_id, now)
        transfer.set_progress(0.0)
        source.tray_id = None
        route.owner = owner
        route.station_id = None

    def _abort_cell_state(self, task: ManufacturingTask) -> None:
        binding = self._transfer_binding(task.task_type)
        if binding is None or task.tray_id is None:
            return
        transfer_id, _ = binding
        transfer = self.transfers[transfer_id]
        if transfer.tray_id != task.tray_id:
            return
        source = self.workstations[transfer.source]
        route = self.tray_routes.get(task.tray_id)
        transfer.release()
        if route is not None:
            source.tray_id = route.tray_id
            route.station_id = transfer.source
            route.owner = {
                WorkstationId.S1_BASE_LOADING: TrayOwner.STATION_S1,
                WorkstationId.S2A_DISPENSING: TrayOwner.STATION_S2A,
                WorkstationId.S2B_MATERIAL_INSPECTION: TrayOwner.STATION_S2B,
                WorkstationId.S3_FIN_ASSEMBLY: TrayOwner.STATION_S3,
            }.get(transfer.source, TrayOwner.RACK_INFEED)

    def _advance_cell_state(self, task: ManufacturingTask, now: float) -> None:
        if task.tray_id is None or task.tray_id not in self.tray_routes:
            return
        route = self.tray_routes[task.tray_id]
        transfer_target = {
            TaskType.TRANSFER_S1_S2A: (
                WorkstationId.S2A_DISPENSING,
                TrayOwner.STATION_S2A,
            ),
            TaskType.TRANSFER_S2A_S2B: (
                WorkstationId.S2B_MATERIAL_INSPECTION,
                TrayOwner.STATION_S2B,
            ),
            TaskType.TRANSFER_S2B_S3: (
                WorkstationId.S3_FIN_ASSEMBLY,
                TrayOwner.STATION_S3,
            ),
            TaskType.TRANSFER_S3_RACK: (
                WorkstationId.RACK_INFEED,
                TrayOwner.RACK_INFEED,
            ),
        }.get(task.task_type)
        if task.task_type is TaskType.INDEX_EMPTY_TRAY:
            station = self.workstations[WorkstationId.S1_BASE_LOADING]
            if station.tray_id not in {None, route.tray_id}:
                self.last_error = "S1 received a pallet while still occupied"
                return
            station.tray_id = route.tray_id
            route.station_id = WorkstationId.S1_BASE_LOADING
            route.owner = TrayOwner.STATION_S1
        elif task.task_type is TaskType.MOVE_ELEVATOR:
            rack = self.workstations[WorkstationId.RACK_INFEED]
            if rack.tray_id == route.tray_id:
                rack.tray_id = None
            route.station_id = None
            route.owner = TrayOwner.ELEVATOR
        elif transfer_target is not None:
            target_station, owner = transfer_target
            binding = self._transfer_binding(task.task_type)
            if binding is not None:
                transfer = self.transfers[binding[0]]
                if transfer.tray_id == route.tray_id:
                    transfer.set_progress(1.0)
                    transfer.release()
            for station in self.workstations.values():
                if station.tray_id == route.tray_id:
                    station.tray_id = None
            destination = self.workstations[target_station]
            if destination.tray_id not in {None, route.tray_id}:
                self.last_error = f"{target_station.value} received a pallet while occupied"
                return
            destination.tray_id = route.tray_id
            route.station_id = target_station
            route.owner = owner
        target = {
            TaskType.INDEX_EMPTY_TRAY: TrayRoutePhase.MOLD_READY,
            TaskType.VERIFY_MOLD: TrayRoutePhase.MOLD_READY,
            TaskType.PLACE_BASE_PLATE: TrayRoutePhase.BASE_READY,
            TaskType.INSPECT_BRAZING: TrayRoutePhase.MATERIAL_READY,
            TaskType.INSPECT_FINS: TrayRoutePhase.ASSEMBLY_READY,
            TaskType.LOCK_FIXTURE: TrayRoutePhase.LOCKED,
            TaskType.TRANSFER_S3_RACK: TrayRoutePhase.OUTFEED,
            TaskType.LOCK_RACK_LAYER: TrayRoutePhase.FURNACE,
            TaskType.POST_BRAZE_INSPECTION: TrayRoutePhase.FINISHED_GOODS,
        }.get(task.task_type)
        if task.task_type is TaskType.VERIFY_MOLD:
            route.mold_name = str(task.payload.get("module_name") or "订单模具")
        elif task.task_type is TaskType.VERIFY_COMB:
            route.comb_name = str(task.payload.get("module_name") or "订单梳齿")
        elif task.task_type is TaskType.LOCK_FIXTURE:
            route.press_locked = True
        elif task.task_type is TaskType.LOCK_RACK_LAYER:
            route.owner = TrayOwner.FURNACE_RACK
            route.station_id = None
        elif task.task_type is TaskType.UNLOAD_RACK_LAYER:
            route.owner = TrayOwner.POST_INSPECTION
            route.station_id = WorkstationId.POST_BRAZE_INSPECTION
        elif task.task_type in {
            TaskType.ROUTE_PASS,
            TaskType.ROUTE_REWORK,
            TaskType.ROUTE_SCRAP,
        }:
            route.owner = TrayOwner.OUTPUT
            route.station_id = None
        if target is None:
            return
        try:
            route.transition(target, now)
        except ValueError:
            # Recovery/replay may complete an idempotent task after the route
            # already advanced; never move the route backwards.
            if route.phase is not target:
                self.last_error = f"托盘{route.tray_id}阶段与任务{task.task_id}不一致"

    @property
    def terminal(self) -> bool:
        return bool(self.orders) and all(
            entry.status in {OrderRunStatus.COMPLETED, OrderRunStatus.MANUAL_REVIEW, OrderRunStatus.CANCELLED}
            for entry in self.orders.values()
        )

    def snapshot(self, now: float | None = None) -> dict[str, Any]:
        timestamp = self.last_tick if now is None else float(now)
        timestamp = 0.0 if timestamp is None else timestamp
        transfer_types = {
            TransferId.S1_S2A: TaskType.TRANSFER_S1_S2A,
            TransferId.S2A_S2B: TaskType.TRANSFER_S2A_S2B,
            TransferId.S2B_S3: TaskType.TRANSFER_S2B_S3,
            TransferId.S3_RACK: TaskType.TRANSFER_S3_RACK,
        }
        for transfer_id, transfer in self.transfers.items():
            if transfer.tray_id is None:
                continue
            running_task = next(
                (
                    task
                    for task in self.graph
                    if task.task_type is transfer_types[transfer_id]
                    and task.tray_id == transfer.tray_id
                    and task.status is TaskStatus.RUNNING
                ),
                None,
            )
            if running_task is not None and running_task.started_at is not None:
                transfer.set_progress(
                    (timestamp - running_task.started_at) / max(running_task.estimated_duration, 1.0e-6)
                )
        tasks = self.graph.snapshot()
        ready = [item for item in tasks if item["status"] == TaskStatus.READY.value]
        running = [item for item in tasks if item["status"] == TaskStatus.RUNNING.value]
        orders = [entry.as_dict() for entry in self.orders.values()]
        for item in orders:
            entry = self.orders[item["order_id"]]
            task_states = [self.graph.get(task_id).status for task_id in entry.graph_task_ids]
            done = sum(status.terminal for status in task_states)
            item["progress"] = 0.0 if not task_states else done / len(task_states)
        elapsed = 0.0 if self.started_at is None else max(0.0, timestamp - self.started_at)
        active_resources = sorted(self.executor.active)
        active_arms = [resource for resource in active_resources if resource in self._arm_busy_seconds]
        return {
            "schema_version": 2,
            "scheduler": {
                **self.scheduler.snapshot(),
                "ready_count": len(ready),
                "running_count": len(running),
                "replan_count": self.replanner.replan_count,
            },
            "tasks": tasks,
            "resources_v2": self.resources.snapshot(),
            "zone_locks": self.zones.snapshot(),
            "orders": orders,
            "faults_v2": [fault.as_dict() for fault in self.faults.values()],
            "recoveries": [plan.as_dict() for plan in self.recovery.plans.values()],
            "manual_fault_requests": [request.as_dict() for request in self.manual_fault_requests.values()],
            "assignments": list(self.assignment_history[-100:]),
            "workstations": {key.value: value.as_dict() for key, value in self.workstations.items()},
            "async_line": {
                "layout": "SHALLOW_U",
                "flow": ["S1", "S2A", "S2B", "S3", "RACK", "FURNACE", "OUTPUT"],
                "wip_limit": self.max_wip_units,
                "active_wip": self._active_wip(),
                "parallelism": {
                    "active_resources": active_resources,
                    "active_arms": active_arms,
                    "current_parallel_arms": len(active_arms),
                    "max_parallel_arms": self._max_parallel_arms,
                    "max_parallel_tasks": self._max_parallel_tasks,
                    "multi_arm_overlap_s": self._parallel_arm_seconds,
                    "arm_busy_s": dict(self._arm_busy_seconds),
                },
                "spare_trays": [
                    tray.tray_id
                    for tray in self.tray_routes.values()
                    if tray.owner is TrayOwner.EMPTY_BUFFER and tray.order_id is None
                ],
            },
            "transfers": {key.value: value.as_dict() for key, value in self.transfers.items()},
            "tray_routes": {key: value.as_dict() for key, value in sorted(self.tray_routes.items())},
            "motion_plans": ([] if self.motion_planning is None else self.motion_planning.path_snapshots()),
            "space_time_reservations": (
                [] if self.motion_planning is None else self.motion_planning.reservation_snapshots()
            ),
            "motion_blockers": ({} if self.motion_planning is None else dict(self.motion_planning.blockers)),
            "furnace_batches": [
                batch.as_dict()
                for batch in sorted(self.furnace_batches.values(), key=lambda item: item.batch_id)
            ],
            "gantt_events": [
                {
                    "task_id": task.task_id,
                    "resource_id": task.assigned_resource,
                    "task_type": task.task_type.value,
                    "display_name_zh": task.as_dict().get("display_name_zh"),
                    "station_id": task.station_id,
                    "tray_id": task.tray_id,
                    "status": task.status.value,
                    "planned_duration": task.estimated_duration,
                    "actual_start": task.started_at,
                    "actual_end": task.finished_at,
                    "waiting": (
                        0.0
                        if task.ready_at is None or task.started_at is None
                        else max(0.0, task.started_at - task.ready_at)
                    ),
                    "blockers": list(task.payload.get("planning_blockers", ())),
                }
                for task in self.graph
                if task.started_at is not None or task.status is TaskStatus.READY
            ][-300:],
            "paused": self.paused,
            "stopped": self.stopped,
            "terminal": self.terminal,
            "elapsed": elapsed,
            "last_error": self.last_error,
        }


__all__ = ["ManufacturingRuntime", "OrderQueueEntry", "OrderRunStatus", "PendingManualFault"]
