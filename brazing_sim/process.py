"""Event-driven single-order coordinator for the brazing MVP.

The coordinator owns process truth, dependencies, inspection gates, resource
leases and the asynchronous furnace.  Robot implementations are injected as
small actors, which lets the same state machine drive MuJoCo and deterministic
headless tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import time
from typing import Any, Callable, Mapping, Protocol

from typing import TYPE_CHECKING

from .config import FIXTURE_CONFIG, create_product_state, make_order_spec
from .domain import (
    Actor,
    FixtureStatus,
    FurnacePhase,
    OrderSpec,
    OrderStage,
    PressState,
    ProductState,
    TaskSpec,
    TaskStatus,
    TaskType,
)
from .furnace import DemoFurnace
from .kpi import KpiTracker
from .quality import QualityEvaluator
from .resources import ResourceManager

if TYPE_CHECKING:
    from .flexible.models import ProcessPlan


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(item) for item in value]
    return value


class ActorResult(str, Enum):
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class TaskActor(Protocol):
    def start_task(self, task: TaskSpec, now: float) -> None: ...

    def poll_task(self, now: float) -> ActorResult | TaskStatus | str | bool | None: ...

    def cancel(self) -> None: ...


@dataclass(slots=True)
class ProcessEvent:
    timestamp: float
    kind: str
    message: str
    task_id: str = ""
    stage: str = ""


@dataclass(slots=True)
class FaultSpec:
    fault_type: str
    target: str = ""
    severity: str = "recoverable"
    armed: bool = True
    applied: bool = False

    def __post_init__(self) -> None:
        if self.fault_type not in {
            "fin_pose",
            "fin_pick",
            "brazing_gap",
            "brazing_deviation",
            "furnace_profile",
        }:
            raise ValueError(f"unsupported fault type: {self.fault_type}")
        if self.severity not in {"recoverable", "severe"}:
            raise ValueError("fault severity must be recoverable or severe")
        if self.fault_type != "furnace_profile" and not self.target:
            raise ValueError("fin/path fault requires a target")


class TimedTaskActor:
    """Deterministic actor used by headless runs and coordinator unit tests."""

    DEFAULT_DURATIONS: Mapping[str, float] = {
        TaskType.LOAD_BASE.value: 0.35,
        TaskType.PREPARE_FIN_TOOL.value: 0.20,
        TaskType.CONFIGURE_COMB.value: 0.20,
        TaskType.INSERT_FIN.value: 0.25,
        TaskType.ADJUST_FIN.value: 0.20,
        TaskType.PRE_INSPECT.value: 0.25,
        TaskType.APPLY_MATERIAL.value: 0.20,
        TaskType.REAPPLY_MATERIAL.value: 0.15,
        TaskType.MATERIAL_INSPECT.value: 0.25,
        TaskType.PRESS_FIXTURE.value: 0.35,
        TaskType.LOCK_FIXTURE.value: 0.15,
        TaskType.LOAD_FURNACE.value: 0.35,
        TaskType.UNLOAD_FURNACE.value: 0.35,
        TaskType.POST_INSPECT.value: 0.30,
    }

    def __init__(self, durations: Mapping[str, float] | None = None, *, time_scale: float = 1.0) -> None:
        self.durations = dict(self.DEFAULT_DURATIONS)
        self.durations.update(durations or {})
        self.time_scale = max(0.0, float(time_scale))
        self.task: TaskSpec | None = None
        self.finished_at = 0.0
        self.error = ""

    def start_task(self, task: TaskSpec, now: float) -> None:
        if self.task is not None:
            raise RuntimeError("actor is already busy")
        self.task = task
        duration = self.durations.get(
            str(task.task_type), self.durations.get(getattr(task.task_type, "value", ""), 0.1)
        )
        self.finished_at = float(now) + float(duration) * self.time_scale

    def poll_task(self, now: float) -> ActorResult:
        if self.task is None:
            return ActorResult.SUCCEEDED
        if self.error:
            return ActorResult.FAILED
        if float(now) < self.finished_at:
            return ActorResult.RUNNING
        self.task = None
        return ActorResult.SUCCEEDED

    def cancel(self) -> None:
        self.task = None
        self.error = ""


TaskExecutor = TimedTaskActor


TASK_ACTORS: Mapping[TaskType, Actor] = {
    TaskType.LOAD_BASE: Actor.ARM1,
    TaskType.PREPARE_FIN_TOOL: Actor.ARM1,
    TaskType.CONFIGURE_COMB: Actor.FIXTURE,
    TaskType.INSERT_FIN: Actor.ARM1,
    TaskType.ADJUST_FIN: Actor.ARM1,
    TaskType.PRE_INSPECT: Actor.ARM3,
    TaskType.APPLY_MATERIAL: Actor.ARM2,
    TaskType.REAPPLY_MATERIAL: Actor.ARM2,
    TaskType.MATERIAL_INSPECT: Actor.ARM3,
    TaskType.PRESS_FIXTURE: Actor.FIXTURE,
    TaskType.LOCK_FIXTURE: Actor.FIXTURE,
    TaskType.LOAD_FURNACE: Actor.CONVEYOR,
    TaskType.UNLOAD_FURNACE: Actor.CONVEYOR,
    TaskType.POST_INSPECT: Actor.ARM3,
}


TASK_RESOURCES: Mapping[TaskType, tuple[str, ...]] = {
    TaskType.LOAD_BASE: ("assembly_fixture", "table2_zone"),
    TaskType.PREPARE_FIN_TOOL: ("arm1_tool_rack",),
    TaskType.CONFIGURE_COMB: ("assembly_fixture", "table2_zone"),
    TaskType.INSERT_FIN: ("assembly_fixture", "table2_zone"),
    TaskType.ADJUST_FIN: ("assembly_fixture", "table2_zone"),
    TaskType.PRE_INSPECT: ("assembly_fixture", "table2_zone", "inspection_zone"),
    TaskType.APPLY_MATERIAL: ("assembly_fixture", "table2_zone", "brazing_zone"),
    TaskType.REAPPLY_MATERIAL: ("assembly_fixture", "table2_zone", "brazing_zone"),
    TaskType.MATERIAL_INSPECT: ("assembly_fixture", "table2_zone", "inspection_zone"),
    TaskType.PRESS_FIXTURE: ("assembly_fixture", "table2_zone"),
    TaskType.LOCK_FIXTURE: ("assembly_fixture", "table2_zone"),
    TaskType.LOAD_FURNACE: (
        "assembly_fixture",
        "table2_zone",
        "furnace_mouth",
        "conveyor_lane",
    ),
    TaskType.UNLOAD_FURNACE: (
        "assembly_fixture",
        "table2_zone",
        "furnace_mouth",
        "conveyor_lane",
    ),
    TaskType.POST_INSPECT: ("assembly_fixture", "table2_zone", "inspection_zone"),
}

TASK_TIMEOUTS: Mapping[TaskType, float] = {
    TaskType.LOAD_BASE: 300.0,
    TaskType.PREPARE_FIN_TOOL: 120.0,
    TaskType.CONFIGURE_COMB: 30.0,
    TaskType.INSERT_FIN: 240.0,
    TaskType.ADJUST_FIN: 240.0,
    TaskType.PRE_INSPECT: 600.0,
    TaskType.MATERIAL_INSPECT: 600.0,
    TaskType.PRESS_FIXTURE: 30.0,
    TaskType.LOCK_FIXTURE: 30.0,
    TaskType.LOAD_FURNACE: 240.0,
    TaskType.UNLOAD_FURNACE: 240.0,
    TaskType.POST_INSPECT: 600.0,
}


class ProcessCoordinator:
    """Advance one product through the complete MVP process graph."""

    def __init__(
        self,
        *,
        actors: Mapping[str | Actor, TaskActor | Callable[[TaskSpec, float], Any]] | None = None,
        resources: ResourceManager | None = None,
        quality: QualityEvaluator | None = None,
        clock: Callable[[], float] = time.monotonic,
        fast: bool = False,
    ) -> None:
        self.clock = clock
        self.resources = resources or ResourceManager()
        # Table2 is a shared physical safety zone in addition to the logical
        # fixture/brazing/inspection resources.  Register it here so callers
        # that inject an older ResourceManager remain compatible.
        self.resources.register("arm1_tool_rack")
        self.resources.register("table2_zone")
        self.resources.register("conveyor_lane")
        self.quality = quality or QualityEvaluator()
        scale = 0.0 if fast else 1.0
        supplied = dict(actors or {})
        self.actors: dict[str, Any] = {}
        for actor in (Actor.ARM1, Actor.ARM2, Actor.ARM3, Actor.FIXTURE, Actor.CONVEYOR):
            self.actors[actor.value] = supplied.get(
                actor, supplied.get(actor.value, TimedTaskActor(time_scale=scale))
            )
        self.kpi = KpiTracker(clock)
        self.furnace: DemoFurnace | None = None
        self.product: ProductState | None = None
        self.tasks: list[TaskSpec] = []
        self.task_history: list[TaskSpec] = []
        self.active_task: TaskSpec | None = None
        self.background_tasks: dict[str, TaskSpec] = {}
        self.events: list[ProcessEvent] = []
        self.faults: list[FaultSpec] = []
        self._sequence = 0
        self._stage_generation = 0
        self._pending_fin_targets: tuple[str, ...] = ()
        self._pending_path_targets: tuple[str, ...] = ()
        self._pending_furnace_fault: str | None = None
        self.paused = False
        self.pause_after_stage: OrderStage | None = None
        self.status = "idle"
        self.process_plan: ProcessPlan | None = None
        # Optional physical-cell interlock.  The application uses this to
        # hold a stage at READY while its independent transfer moves the product
        # to the station required by that stage.
        self.stage_gate: Callable[[OrderStage, float], bool] | None = None

    @property
    def running(self) -> bool:
        return self.product is not None and not self.product.terminal

    @property
    def terminal(self) -> bool:
        return self.product is not None and self.product.terminal

    def _event(self, now: float, kind: str, message: str, task: TaskSpec | None = None) -> None:
        self.events.append(
            ProcessEvent(
                float(now),
                kind,
                message,
                "" if task is None else task.task_id,
                "" if self.product is None else self.product.stage.value,
            )
        )
        self.status = message

    def start_order(
        self,
        spec: OrderSpec | str = "A",
        *,
        now: float | None = None,
        order_id: str | None = None,
    ) -> ProductState:
        timestamp = self.clock() if now is None else float(now)
        if self.running:
            raise RuntimeError("an order is already active; stop or reset first")
        selected = make_order_spec(spec) if isinstance(spec, str) else spec
        self.process_plan = None
        self.resources.reset()
        self.kpi.start_order(timestamp)
        self.product = create_product_state(selected, order_id=order_id, created_at=timestamp)
        self.furnace = DemoFurnace(selected.recipe)
        self.tasks.clear()
        self.task_history.clear()
        self.active_task = None
        self.background_tasks.clear()
        self._sequence = 0
        self._stage_generation = 0
        self._pending_fin_targets = ()
        self._pending_path_targets = ()
        self._pending_furnace_fault = None
        self.paused = False
        self.pause_after_stage = None
        for fault in self.faults:
            fault.armed = True
            fault.applied = False
        self._event(timestamp, "order_started", f"order {self.product.order_id} started")
        self._enter_stage(OrderStage.BASE_LOADING, timestamp)
        return self.product

    def start_segment(self, segment: str, *, now: float | None = None) -> ProductState:
        """Start one UI demonstration segment with deterministic prerequisites."""

        timestamp = self.clock() if now is None else float(now)
        key = str(segment).strip().lower()
        targets = {
            "pick_place": OrderStage.BASE_LOADING,
            "inspection_1": OrderStage.MATERIAL_INSPECTION,
            "arm2_motion": OrderStage.MATERIAL_APPLICATION,
            "fin_assembly": OrderStage.FIN_ASSEMBLY,
            "inspection_2": OrderStage.PRE_INSPECTION,
            "furnace_cycle": OrderStage.FIXTURE_PRESSING,
        }
        if key not in targets:
            raise ValueError(f"unknown demonstration segment: {segment}")
        product = self.start_order("A", now=timestamp)
        target = targets[key]
        self.tasks.clear()
        self.active_task = None

        if key != "pick_place":
            product.fixture.base_weld_active = True
            product.fixture.status = FixtureStatus.BASE_FIXED

        if key in {"fin_assembly", "inspection_2", "furnace_cycle"}:
            product.fixture.active_comb_module = product.spec.comb_module_name
            pitch = product.spec.comb_module_name.removeprefix("comb_insert_")
            product.fixture.front_comb_module = f"front_comb_insert_{pitch}"
            product.fixture.rear_comb_module = f"rear_comb_insert_{pitch}"
            product.fixture.comb_configured = True
            product.fixture.comb_aligned = True
            product.fixture.status = FixtureStatus.COMB_CONFIGURED

        if key in {"inspection_1", "fin_assembly", "inspection_2", "furnace_cycle"}:
            for path in product.active_paths:
                path.applied = True
                path.coverage_ratio = 1.0

        if key in {"fin_assembly", "inspection_2", "furnace_cycle"}:
            product.fixture.material_passed = True
            product.fixture.status = FixtureStatus.MATERIAL_READY

        if key in {"inspection_2", "furnace_cycle"}:
            for fin in product.active_fins:
                fin.inserted = True
                fin.temporary_welded = True
                fin.actual_position = fin.target_position
                product.fixture.temporary_fin_welds.add(fin.fin_id)
            product.fixture.status = FixtureStatus.TEMPORARY

        if key == "furnace_cycle":
            product.fixture.fins_passed = True

        product.stage = target
        self.tasks = self._tasks_for_stage(target)
        self.pause_after_stage = OrderStage.UNLOADING if key == "furnace_cycle" else target
        self.kpi.enter_stage(target, timestamp)
        self._event(timestamp, "segment_started", f"segment {key} started")
        if key == "furnace_cycle":
            self._preopen_furnace(timestamp)
        elif key == "arm2_motion":
            # BrazingApplication resets the physical scene immediately after
            # segment creation. Queue now and start on the first post-reset
            # tick so no precomputed tool-change path is invalidated.
            self._ensure_fin_tool_preparation(timestamp, start_immediately=False)
        return product

    def pause(self, now: float | None = None) -> None:
        """Pause safely while keeping the current order resumable."""

        timestamp = self.clock() if now is None else float(now)
        if self.product is None or self.product.terminal or self.paused:
            return
        if self.active_task is not None:
            task = self.active_task
            actor = self._actor(task.actor)
            if hasattr(actor, "cancel"):
                actor.cancel()
            self._release(task)
            self.kpi.actor_finished(task.actor, timestamp)
            task.status = TaskStatus.READY
            task.started_at = None
            task.completed_at = None
            task.error = None
            self.active_task = None
        for name, task in tuple(self.background_tasks.items()):
            if task.status is TaskStatus.RUNNING:
                actor = self._actor(task.actor)
                if hasattr(actor, "cancel"):
                    actor.cancel()
                self._release(task)
                self.kpi.actor_finished(task.actor, timestamp)
                task.status = TaskStatus.READY
                task.started_at = None
                task.completed_at = None
                task.error = None
            self.background_tasks[name] = task
        self.resources.reset()
        self.paused = True
        self._event(timestamp, "paused", "process paused")

    def resume(self, now: float | None = None) -> None:
        timestamp = self.clock() if now is None else float(now)
        if self.product is None:
            raise RuntimeError("no process is available to continue")
        if self.product.terminal:
            raise RuntimeError("terminal process cannot be continued; choose a segment")
        self.paused = False
        self.pause_after_stage = None
        self._event(timestamp, "resumed", "process continued")

    def inject_fault(
        self,
        fault_type: str | FaultSpec,
        target: str = "",
        severity: str = "recoverable",
    ) -> FaultSpec:
        fault = (
            fault_type if isinstance(fault_type, FaultSpec) else FaultSpec(str(fault_type), target, severity)
        )
        if fault.fault_type in {"fin_pose", "fin_pick"} and not fault.target.startswith("fin_"):
            raise ValueError("fin fault target must be fin_XX")
        if fault.fault_type in {"brazing_gap", "brazing_deviation"} and not fault.target.startswith(
            ("slot_", "fin_")
        ):
            raise ValueError("brazing fault target must be slot_XX_left/right")
        self.faults.append(fault)
        return fault

    def _new_task(self, task_type: TaskType, payload: Mapping[str, Any] | None = None) -> TaskSpec:
        self._sequence += 1
        actor = TASK_ACTORS[task_type]
        task = TaskSpec(
            task_id=f"t{self._sequence:03d}_{task_type.value.lower()}",
            actor=actor,
            task_type=task_type,
            resources=TASK_RESOURCES.get(task_type, ()),
            payload=dict(payload or {}),
            timeout=TASK_TIMEOUTS.get(task_type, 120.0),
        )
        return task

    def _ensure_fin_tool_preparation(self, now: float, *, start_immediately: bool = True) -> None:
        """Prepare Arm1's gripper while Arm2 works in the separate Table2 zone."""

        key = Actor.ARM1.value
        if key in self.background_tasks:
            return
        if any(task.task_type is TaskType.PREPARE_FIN_TOOL for task in self.task_history):
            return
        task = self._new_task(
            TaskType.PREPARE_FIN_TOOL,
            {
                "required_tool": "parallel_gripper",
                "parallel_with": TaskType.APPLY_MATERIAL.value,
            },
        )
        task.status = TaskStatus.READY
        self.background_tasks[key] = task
        if start_immediately:
            self._try_start_background_task(key, task, now)

    def _try_start_background_task(self, key: str, task: TaskSpec, now: float) -> None:
        if task.status is TaskStatus.RUNNING or task.status.terminal:
            return
        if not self._acquire(task, now):
            return
        actor = self._actor(task.actor)
        task.mark_running(now)
        self.kpi.actor_started(task.actor, now)
        try:
            if hasattr(actor, "start_task"):
                actor.start_task(task, now)
            elif callable(actor):
                actor(task, now)
            else:
                raise TypeError("actor must provide start_task or be callable")
        except Exception as exc:
            self._fail_background_task(key, task, now, str(exc))
            return
        self._event(
            now,
            "background_task_started",
            "Arm1 changing suction tool to parallel gripper while Arm2 applies material",
            task,
        )

    def _poll_background_tasks(self, now: float) -> None:
        for key, task in tuple(self.background_tasks.items()):
            if task.status in {TaskStatus.PENDING, TaskStatus.READY}:
                self._try_start_background_task(key, task, now)
            if task.status is not TaskStatus.RUNNING:
                continue
            actor = self._actor(task.actor)
            try:
                value = actor.poll_task(now) if hasattr(actor, "poll_task") else True
                result = self._normalize_actor_result(value)
            except Exception as exc:
                self._fail_background_task(key, task, now, str(exc))
                continue
            if task.started_at is not None and now - task.started_at > task.timeout:
                self._fail_background_task(key, task, now, "background task timeout")
            elif result is ActorResult.SUCCEEDED:
                self._complete_background_task(key, task, now)
            elif result is ActorResult.FAILED:
                reason = getattr(actor, "error", "actor failed") or "actor failed"
                self._fail_background_task(key, task, now, reason)

    def _complete_background_task(self, key: str, task: TaskSpec, now: float) -> None:
        if task.status.terminal:
            return
        task.mark_succeeded(now)
        self._release(task)
        self.kpi.actor_finished(task.actor, now)
        self._apply_task_effect(task, now)
        self.task_history.append(task)
        self.background_tasks.pop(key, None)
        self._event(
            now,
            "background_task_completed",
            "Arm1 gripper ready before fin assembly",
            task,
        )

    def _fail_background_task(
        self,
        key: str,
        task: TaskSpec,
        now: float,
        reason: str,
    ) -> None:
        if task.status.terminal:
            return
        task.mark_failed(now, reason)
        self._release(task)
        self.kpi.actor_finished(task.actor, now)
        self.task_history.append(task)
        self.background_tasks.pop(key, None)
        if self.active_task is not None:
            active = self.active_task
            actor = self._actor(active.actor)
            if hasattr(actor, "cancel"):
                actor.cancel()
            active.status = TaskStatus.CANCELLED
            active.completed_at = now
            self._release(active)
            self.kpi.actor_finished(active.actor, now)
            self.active_task = None
        if self.product is not None:
            self.product.fail(f"{task.task_id}: {reason}", now)
        self._event(now, "background_task_failed", reason, task)

    def _enter_stage(self, stage: OrderStage, now: float) -> None:
        if self.product is None:
            raise RuntimeError("no active product")
        self.product.transition(stage, now)
        self._stage_generation += 1
        self.tasks = self._tasks_for_stage(stage)
        self.kpi.enter_stage(stage, now)
        self._event(now, "stage", f"stage {stage.value}")
        if stage is OrderStage.MATERIAL_APPLICATION:
            self._ensure_fin_tool_preparation(now)
        elif stage is OrderStage.FIXTURE_PRESSING:
            # The furnace mouth is remote from Table2. Opening its door while
            # the fixture performs its force ramp is collision-independent and
            # hides the complete door latency behind useful clamping work.
            self._preopen_furnace(now)
        elif stage is OrderStage.FURNACE_LOADING:
            self._preopen_furnace(now)
        elif stage is OrderStage.BRAZING:
            self._start_furnace(now)

    def _preopen_furnace(self, now: float) -> None:
        assert self.product is not None and self.furnace is not None
        if self.furnace.status is FurnacePhase.IDLE:
            self.furnace.request_open(now)
            self._event(
                now,
                "furnace_door",
                "furnace door pre-opening concurrently with fixture work",
            )
        elif self.furnace.status not in {FurnacePhase.DOOR_OPENING, FurnacePhase.LOADING}:
            raise RuntimeError(f"furnace cannot prepare for loading while {self.furnace.status.value}")
        self.product.furnace = self.furnace.state

    def _tasks_for_stage(self, stage: OrderStage) -> list[TaskSpec]:
        assert self.product is not None
        if stage is OrderStage.BASE_LOADING:
            return [self._new_task(TaskType.LOAD_BASE)]
        if stage is OrderStage.COMB_CONFIGURATION:
            return [
                self._new_task(
                    TaskType.CONFIGURE_COMB,
                    {"comb_module_name": self.product.spec.comb_module_name},
                )
            ]
        if stage is OrderStage.MATERIAL_APPLICATION:
            if self._pending_path_targets:
                targets = self._pending_path_targets
                self._pending_path_targets = ()
                tasks: list[TaskSpec] = []
                for index, target in enumerate(targets):
                    path = next(item for item in self.product.active_paths if item.path_id == target)
                    tasks.append(
                        self._new_task(
                            TaskType.REAPPLY_MATERIAL,
                            {
                                "slot_id": path.slot_id,
                                "path_id": path.path_id,
                                "path_ids": [path.path_id],
                                "material_sequence_index": index,
                                "material_sequence_count": len(targets),
                                "continuous_from_previous": index > 0,
                                "reverse_travel": bool(index % 2),
                                "park_after": index == len(targets) - 1,
                            },
                        )
                    )
                return tasks

            paths_by_slot: dict[str, list[str]] = {}
            incomplete_slots: set[str] = set()
            for path in self.product.active_paths:
                paths_by_slot.setdefault(path.slot_id, []).append(path.path_id)
                if not path.applied:
                    incomplete_slots.add(path.slot_id)
            tasks = [
                self._new_task(
                    TaskType.APPLY_MATERIAL,
                    {"slot_id": slot_id, "path_ids": list(path_ids)},
                )
                for slot_id, path_ids in paths_by_slot.items()
                if slot_id in incomplete_slots
            ]
            for index, task in enumerate(tasks):
                task.payload.update(
                    {
                        "material_sequence_index": index,
                        "material_sequence_count": len(tasks),
                        "continuous_from_previous": index > 0,
                        "reverse_travel": bool(index % 2),
                        "park_after": index == len(tasks) - 1,
                    }
                )
            return tasks
        if stage is OrderStage.MATERIAL_INSPECTION:
            return [self._new_task(TaskType.MATERIAL_INSPECT)]
        if stage is OrderStage.FIN_ASSEMBLY:
            if self._pending_fin_targets:
                targets = self._pending_fin_targets
                self._pending_fin_targets = ()
                tasks = [self._new_task(TaskType.ADJUST_FIN, {"fin_id": target}) for target in targets]
            else:
                tasks = [
                    self._new_task(TaskType.INSERT_FIN, {"fin_id": fin.fin_id})
                    for fin in self.product.active_fins
                    if not fin.inserted
                ]
            for index, task in enumerate(tasks):
                task.payload.update(
                    {
                        "fin_sequence_index": index,
                        "fin_sequence_count": len(tasks),
                        "continuous_from_previous": index > 0,
                        # Intermediate fins retreat outside the fixture and
                        # proceed directly to the next blank. Only the final
                        # fin pays the full canonical-parking cost.
                        "park_after": index == len(tasks) - 1,
                    }
                )
            return tasks
        if stage is OrderStage.PRE_INSPECTION:
            return [self._new_task(TaskType.PRE_INSPECT)]
        if stage is OrderStage.FIXTURE_PRESSING:
            return [
                self._new_task(
                    TaskType.PRESS_FIXTURE,
                    {
                        "target_force_n": self.product.spec.target_clamping_force_n,
                        "force_tolerance_n": self.product.spec.clamping_force_tolerance_n,
                        "hold_duration_s": self.product.spec.force_hold_duration_s,
                    },
                )
            ]
        if stage is OrderStage.FIXTURE_LOCKING:
            return [self._new_task(TaskType.LOCK_FIXTURE)]
        if stage is OrderStage.FURNACE_LOADING:
            return [self._new_task(TaskType.LOAD_FURNACE)]
        if stage is OrderStage.UNLOADING:
            return [self._new_task(TaskType.UNLOAD_FURNACE)]
        if stage is OrderStage.POST_INSPECTION:
            return [self._new_task(TaskType.POST_INSPECT)]
        return []

    def _actor(self, actor: Actor | str) -> Any:
        key = actor.value if isinstance(actor, Actor) else str(actor)
        try:
            return self.actors[key]
        except KeyError as exc:
            raise RuntimeError(f"no task actor registered for {key}") from exc

    def _acquire(self, task: TaskSpec, now: float) -> bool:
        owner = task.actor.value if isinstance(task.actor, Actor) else str(task.actor)
        acquired: list[str] = []
        for resource in task.resources:
            if self.resources.acquire(resource, owner, now=now):
                acquired.append(resource)
                continue
            for held in acquired:
                self.resources.release(held, owner)
            self.kpi.actor_waiting_for_resource(owner, now)
            self.kpi.resource_conflict()
            return False
        self.kpi.actor_stopped_waiting(owner, now)
        return True

    def _release(self, task: TaskSpec) -> None:
        owner = task.actor.value if isinstance(task.actor, Actor) else str(task.actor)
        for resource in task.resources:
            self.resources.release(resource, owner)

    @staticmethod
    def _normalize_actor_result(value: Any) -> ActorResult:
        if value is None:
            return ActorResult.RUNNING
        if value is True:
            return ActorResult.SUCCEEDED
        if value is False:
            return ActorResult.FAILED
        if isinstance(value, ActorResult):
            return value
        if isinstance(value, TaskStatus):
            if value is TaskStatus.SUCCEEDED:
                return ActorResult.SUCCEEDED
            if value in {TaskStatus.FAILED, TaskStatus.CANCELLED}:
                return ActorResult.FAILED
            return ActorResult.RUNNING
        text = str(getattr(value, "value", value)).upper()
        if text in {"DONE", "COMPLETE", "COMPLETED", "SUCCESS", "SUCCEEDED"}:
            return ActorResult.SUCCEEDED
        if text in {"ERROR", "FAILED", "CANCELLED"}:
            return ActorResult.FAILED
        return ActorResult.RUNNING

    def _start_task(self, task: TaskSpec, now: float) -> None:
        actor = self._actor(task.actor)
        task.mark_running(now)
        self.active_task = task
        self.kpi.actor_started(task.actor, now)
        try:
            if hasattr(actor, "start_task"):
                actor.start_task(task, now)
            elif callable(actor):
                actor(task, now)
            else:
                raise TypeError("actor must provide start_task or be callable")
        except Exception as exc:
            self._fail_task(task, now, str(exc))
            return
        self._event(now, "task_started", task.task_type.value, task)

    def _poll_task(self, now: float) -> None:
        task = self.active_task
        if task is None:
            return
        actor = self._actor(task.actor)
        try:
            value = actor.poll_task(now) if hasattr(actor, "poll_task") else True
            result = self._normalize_actor_result(value)
        except Exception as exc:
            self._fail_task(task, now, str(exc))
            return
        if task.started_at is not None and now - task.started_at > task.timeout:
            self._fail_task(task, now, "task timeout")
        elif result is ActorResult.SUCCEEDED:
            self._complete_task(task, now)
        elif result is ActorResult.FAILED:
            self._fail_task(task, now, getattr(actor, "error", "actor failed") or "actor failed")

    def _complete_task(self, task: TaskSpec, now: float) -> None:
        if task.status.terminal:
            return
        task.mark_succeeded(now)
        self._release(task)
        self.kpi.actor_finished(task.actor, now)
        self._apply_task_effect(task, now)
        self.task_history.append(task)
        self.tasks = [item for item in self.tasks if item is not task]
        self.active_task = None
        self._event(now, "task_completed", task.task_type.value, task)

    def _fail_task(self, task: TaskSpec, now: float, reason: str) -> None:
        if task.status.terminal:
            return
        task.mark_failed(now, reason)
        self._release(task)
        self.kpi.actor_finished(task.actor, now)
        self.task_history.append(task)
        self.active_task = None
        for key, background in tuple(self.background_tasks.items()):
            actor = self._actor(background.actor)
            if hasattr(actor, "cancel"):
                actor.cancel()
            background.status = TaskStatus.CANCELLED
            background.completed_at = now
            self._release(background)
            self.kpi.actor_finished(background.actor, now)
            self.background_tasks.pop(key, None)
        if self.product is not None:
            self.product.fail(f"{task.task_id}: {reason}", now)
        self._event(now, "task_failed", reason, task)

    def _apply_armed_fault(self, kind: str, *, target_id: str | None = None) -> None:
        assert self.product is not None
        for fault in self.faults:
            if not fault.armed or fault.applied or fault.fault_type != kind:
                continue
            if kind in {"fin_pose", "fin_pick"}:
                if target_id is not None and fault.target != target_id:
                    continue
                fin = next((item for item in self.product.active_fins if item.fin_id == fault.target), None)
                if fin is None:
                    raise ValueError(f"unknown active fin: {fault.target}")
                if kind == "fin_pick":
                    offset = 0.050 if fault.severity == "recoverable" else 0.090
                    angle = 0.0
                    fin.root_gap_m = 0.020
                else:
                    offset = 0.006 if fault.severity == "recoverable" else 0.012
                    angle = 6.0 if fault.severity == "recoverable" else 12.0
                fin.position_error_m = offset
                fin.verticality_error_deg = angle
                fin.actual_position = (
                    fin.target_position[0],
                    fin.target_position[1] + offset,
                    fin.target_position[2],
                )
            elif kind == "brazing_gap":
                path_target = fault.target
                if path_target.startswith("fin_"):
                    # Compatibility with the original terminal command while
                    # product paths now use slot_XX_left/right identifiers.
                    path_target = f"slot_{path_target.removeprefix('fin_')}"
                if target_id is not None and path_target != target_id:
                    continue
                path = next((item for item in self.product.active_paths if item.path_id == path_target), None)
                if path is None:
                    raise ValueError(f"unknown active path: {path_target}")
                path.coverage_ratio = 0.70 if fault.severity == "recoverable" else 0.20
                path.longest_gap_m = 0.020 if fault.severity == "recoverable" else 0.100
            elif kind == "brazing_deviation":
                path_target = fault.target
                if path_target.startswith("fin_"):
                    path_target = f"slot_{path_target.removeprefix('fin_')}"
                if target_id is not None and path_target != target_id:
                    continue
                path = next((item for item in self.product.active_paths if item.path_id == path_target), None)
                if path is None:
                    raise ValueError(f"unknown active path: {path_target}")
                path.lateral_error_m = 0.005 if fault.severity == "recoverable" else 0.012
            fault.applied = True
            fault.armed = False

    def _furnace_fault(self) -> str | None:
        for fault in self.faults:
            if fault.armed and not fault.applied and fault.fault_type == "furnace_profile":
                fault.applied = True
                fault.armed = False
                return fault.severity
        return None

    def _apply_task_effect(self, task: TaskSpec, now: float) -> None:
        assert self.product is not None
        kind = TaskType(task.task_type)
        if kind is TaskType.PREPARE_FIN_TOOL:
            # The physical tool manager is the source of truth; no product
            # geometry or process stage changes during this background task.
            return
        if kind is TaskType.LOAD_BASE:
            self.product.fixture.base_weld_active = True
            self.product.fixture.status = FixtureStatus.BASE_FIXED
        elif kind is TaskType.CONFIGURE_COMB:
            module_name = str(task.payload.get("comb_module_name", self.product.spec.comb_module_name))
            pitch = module_name.removeprefix("comb_insert_")
            self.product.fixture.active_comb_module = module_name
            self.product.fixture.front_comb_module = f"front_comb_insert_{pitch}"
            self.product.fixture.rear_comb_module = f"rear_comb_insert_{pitch}"
            self.product.fixture.comb_configured = True
            self.product.fixture.comb_aligned = module_name == self.product.spec.comb_module_name
            self.product.fixture.status = FixtureStatus.COMB_CONFIGURED
        elif kind in {TaskType.INSERT_FIN, TaskType.ADJUST_FIN}:
            fin_id = str(task.payload["fin_id"])
            fin = next(item for item in self.product.active_fins if item.fin_id == fin_id)
            fin.inserted = True
            fin.temporary_welded = True
            fin.actual_position = fin.target_position
            fin.position_error_m = 0.0
            fin.verticality_error_deg = 0.0
            fin.root_gap_m = 0.0
            fin.pitch_error_m = 0.0
            self.product.fixture.temporary_fin_welds.add(fin_id)
            self.product.fixture.status = FixtureStatus.TEMPORARY
            if kind is TaskType.INSERT_FIN:
                # Assembly faults originate at the corresponding Arm1 action.
                # Arm3 later observes this already-existing physical defect;
                # inspection must never be the moment that creates it.
                for fault_kind in ("fin_pose", "fin_pick"):
                    self._apply_armed_fault(fault_kind, target_id=fin_id)
        elif kind is TaskType.PRE_INSPECT:
            result = self.quality.pre_inspection(self.product, now)
            self.product.fixture.fins_passed = result.passed
            if not result.passed:
                if not self.quality.register_automatic_rework(self.product, result):
                    return
                self.kpi.record_rework("fin", len(result.rework_targets))
                self._pending_fin_targets = result.rework_targets
        elif kind in {TaskType.APPLY_MATERIAL, TaskType.REAPPLY_MATERIAL}:
            path_ids = tuple(str(item) for item in task.payload.get("path_ids", ()))
            if not path_ids and "path_id" in task.payload:
                path_ids = (str(task.payload["path_id"]),)
            if not path_ids:
                raise RuntimeError("material task completed without path_ids")
            rmse_by_path = task.payload.get("trajectory_rmse_by_path", {})
            max_error_by_path = task.payload.get("trajectory_max_error_by_path", {})
            for path_id in path_ids:
                path = next(item for item in self.product.active_paths if item.path_id == path_id)
                path.applied = True
                path.coverage_ratio = 1.0
                path.longest_gap_m = 0.0
                path.lateral_error_m = 0.0
                path.trajectory_rmse_m = float(
                    rmse_by_path.get(path_id, task.payload.get("trajectory_rmse_m", 0.0))
                )
                path.trajectory_max_error_m = float(
                    max_error_by_path.get(path_id, task.payload.get("trajectory_max_error_m", 0.0))
                )
                if kind is TaskType.APPLY_MATERIAL:
                    # Material faults originate while Arm2 deposits this bead
                    # and remain visible throughout the subsequent Arm3 scan.
                    self._apply_armed_fault("brazing_gap", target_id=path_id)
                    self._apply_armed_fault("brazing_deviation", target_id=path_id)
        elif kind is TaskType.MATERIAL_INSPECT:
            result = self.quality.material_inspection(self.product, now)
            self.product.fixture.material_passed = result.passed
            if result.passed:
                self.product.fixture.status = FixtureStatus.MATERIAL_READY
            if not result.passed:
                if not self.quality.register_automatic_rework(self.product, result):
                    return
                self.kpi.record_rework("material", len(result.rework_targets))
                self._pending_path_targets = result.rework_targets
        elif kind is TaskType.PRESS_FIXTURE:
            target_force = float(
                task.payload.get("measured_force_n", task.payload.get("target_force_n", 0.0))
            )
            if self.product.fixture.press_state is not PressState.COMPLETE:
                self.product.fixture.press_state = PressState.COMPLETE
                self.product.fixture.press_position_m = float(
                    task.payload.get("press_position_m", -FIXTURE_CONFIG.press_travel_m)
                )
                self.product.fixture.clamping_force_n = target_force
                self.product.fixture.press_force_held = True
            self.product.fixture.status = FixtureStatus.PRESSING
        elif kind is TaskType.LOCK_FIXTURE:
            self.product.fixture.lock()
        elif kind is TaskType.LOAD_FURNACE:
            assert self.furnace is not None
            self.furnace.load_workpiece(now)
            self.furnace.request_close(now)
            self.product.furnace = self.furnace.state
            self.product.fixture.in_furnace = True
            self.product.fixture.status = FixtureStatus.IN_FURNACE
        elif kind is TaskType.UNLOAD_FURNACE:
            self.product.fixture.in_furnace = False
            self.product.fixture.status = FixtureStatus.READY_FOR_TRANSFER
        elif kind is TaskType.POST_INSPECT:
            result = self.quality.post_inspection(self.product, now)
            self.kpi.set_final_quality(result.score)
            self.kpi.finish_order(now, result.score)

    def _start_furnace(self, now: float) -> None:
        assert self.product is not None and self.furnace is not None
        if not self.product.fixture.locked or not self.product.fixture.in_furnace:
            self.product.fail("furnace interlock: fixture must be locked and loaded", now)
            return
        self._pending_furnace_fault = self._furnace_fault()
        self.product.furnace = self.furnace.state
        self._event(now, "furnace_waiting", "waiting for the conveyor-loaded furnace door to close")

    def _advance_after_stage(self, now: float) -> None:
        assert self.product is not None
        stage = self.product.stage
        if stage is OrderStage.BASE_LOADING:
            self._enter_stage(OrderStage.MATERIAL_APPLICATION, now)
        elif stage is OrderStage.COMB_CONFIGURATION:
            self._enter_stage(OrderStage.FIN_ASSEMBLY, now)
        elif stage is OrderStage.MATERIAL_APPLICATION:
            self._enter_stage(OrderStage.MATERIAL_INSPECTION, now)
        elif stage is OrderStage.MATERIAL_INSPECTION:
            if self.product.stage is OrderStage.MANUAL_REVIEW:
                return
            if self._pending_path_targets:
                self._enter_stage(OrderStage.MATERIAL_APPLICATION, now)
            else:
                self._enter_stage(OrderStage.COMB_CONFIGURATION, now)
        elif stage is OrderStage.FIN_ASSEMBLY:
            self._enter_stage(OrderStage.PRE_INSPECTION, now)
        elif stage is OrderStage.PRE_INSPECTION:
            if self.product.stage is OrderStage.MANUAL_REVIEW:
                return
            if self._pending_fin_targets:
                self._enter_stage(OrderStage.FIN_ASSEMBLY, now)
            else:
                self._enter_stage(OrderStage.FIXTURE_PRESSING, now)
        elif stage is OrderStage.FIXTURE_PRESSING:
            self._enter_stage(OrderStage.FIXTURE_LOCKING, now)
        elif stage is OrderStage.FIXTURE_LOCKING:
            self._enter_stage(OrderStage.READY_FOR_TRANSFER, now)
        elif stage is OrderStage.READY_FOR_TRANSFER:
            self._enter_stage(OrderStage.FURNACE_LOADING, now)
        elif stage is OrderStage.FURNACE_LOADING:
            self._enter_stage(OrderStage.BRAZING, now)
        elif stage is OrderStage.UNLOADING:
            self._enter_stage(OrderStage.POST_INSPECTION, now)

    def tick(self, now: float | None = None) -> ProductState | None:
        timestamp = self.clock() if now is None else float(now)
        if self.product is None or self.product.terminal:
            return self.product
        if self.paused:
            return self.product
        if self.product.stage in {
            OrderStage.FURNACE_LOADING,
            OrderStage.BRAZING,
            OrderStage.UNLOADING,
        } or (self.furnace is not None and self.furnace.status is FurnacePhase.DOOR_OPENING):
            assert self.furnace is not None
            self.furnace.update(timestamp)
            self.product.furnace = self.furnace.state
        self._poll_background_tasks(timestamp)
        if self.product.terminal:
            return self.product
        if self.product.stage is OrderStage.FURNACE_LOADING and not self.furnace.state.door_open:
            # The conveyor task is not even leased until the door has cleared
            # the complete handled tray envelope.
            return self.product
        if self.product.stage is OrderStage.BRAZING:
            if self.furnace.status is FurnacePhase.READY and self.furnace.state.cycle_started_at is None:
                self.furnace.start_cycle(timestamp, fault=self._pending_furnace_fault)
                self.product.furnace = self.furnace.state
                self._event(
                    timestamp,
                    "furnace_started",
                    "10.0 s virtual brazing dwell started",
                )
            if self.furnace.complete:
                for fin in self.product.active_fins:
                    fin.temporary_welded = False
                    fin.board_welded = True
                self._enter_stage(OrderStage.UNLOADING, timestamp)
            return self.product
        if self.active_task is not None:
            self._poll_task(timestamp)
            if self.product.terminal:
                return self.product
        if (
            self.active_task is None
            and self.tasks
            and self.stage_gate is not None
            and not self.stage_gate(self.product.stage, timestamp)
        ):
            self.status = f"waiting for async transfer to {self.product.stage.value}"
            return self.product
        if self.active_task is None and self.tasks:
            task = self.tasks[0]
            actor_key = task.actor.value if isinstance(task.actor, Actor) else str(task.actor)
            if actor_key in self.background_tasks:
                self.status = f"waiting for {actor_key} background preparation to clear"
            elif self._acquire(task, timestamp):
                self._start_task(task, timestamp)
        if self.active_task is None and not self.tasks and not self.product.terminal:
            if self.pause_after_stage is self.product.stage:
                self.paused = True
                self._event(timestamp, "segment_completed", f"segment {self.product.stage.value} completed")
                return self.product
            self._advance_after_stage(timestamp)
        return self.product

    def stop(self, now: float | None = None) -> None:
        timestamp = self.clock() if now is None else float(now)
        if self.active_task is not None:
            actor = self._actor(self.active_task.actor)
            if hasattr(actor, "cancel"):
                actor.cancel()
            self.active_task.status = TaskStatus.CANCELLED
            self._release(self.active_task)
            self.active_task = None
        for key, task in tuple(self.background_tasks.items()):
            actor = self._actor(task.actor)
            if hasattr(actor, "cancel"):
                actor.cancel()
            task.status = TaskStatus.CANCELLED
            task.completed_at = timestamp
            self._release(task)
            self.kpi.actor_finished(task.actor, timestamp)
            self.background_tasks.pop(key, None)
        for actor in self.actors.values():
            if hasattr(actor, "cancel"):
                actor.cancel()
        self.resources.reset()
        if self.furnace is not None and not self.furnace.complete:
            self.furnace.stop(timestamp)
        if self.product is not None:
            self.product.stop(timestamp)
        self.kpi.finish_order(timestamp)
        self._event(timestamp, "stopped", "order stopped")

    def reset(self) -> None:
        self.stop(self.clock()) if self.product is not None and not self.product.terminal else None
        self.resources.reset()
        self.product = None
        self.furnace = None
        self.tasks.clear()
        self.task_history.clear()
        self.active_task = None
        self.background_tasks.clear()
        self.faults.clear()
        self._pending_furnace_fault = None
        self.events.clear()
        self.kpi.reset()
        self.status = "idle"
        self.process_plan = None
        self.paused = False
        self.pause_after_stage = None

    def start_process_plan(
        self,
        plan: ProcessPlan,
        *,
        now: float | None = None,
        unit_index: int = 0,
    ) -> ProductState:
        """Start one planned unit while preserving the existing process graph."""

        if not 0 <= int(unit_index) < plan.quantity:
            raise ValueError("unit_index is outside the ProcessPlan quantity")
        product = self.start_order(
            plan.execution_spec,
            now=now,
            order_id=f"{plan.order.order_id}-U{int(unit_index) + 1:02d}",
        )
        self.process_plan = plan
        timestamp = self.clock() if now is None else float(now)
        self._event(
            timestamp,
            "process_plan_started",
            f"计划 {plan.order.order_id}：第{int(unit_index) + 1}/{plan.quantity}件，"
            f"{len(plan.fin_targets)}片翅片/{len(plan.brazing_paths)}条路径",
        )
        return product

    def detach_ready_product(self) -> ProductState:
        """Release a completed single-layer unit to the batch coordinator.

        The product remains at ``READY_FOR_TRANSFER`` and is not marked
        stopped; only the reusable task executor and per-unit resources are
        cleared for the next tray.
        """

        if self.product is None or self.product.stage is not OrderStage.READY_FOR_TRANSFER:
            raise RuntimeError("only a READY_FOR_TRANSFER product may be detached")
        if self.active_task is not None or self.tasks or self.background_tasks:
            raise RuntimeError("cannot detach a product while tasks are active")
        product = self.product
        self.resources.reset()
        self.product = None
        self.furnace = None
        self.task_history.clear()
        self.active_task = None
        self.background_tasks.clear()
        self.tasks.clear()
        self.paused = False
        self.pause_after_stage = None
        self.status = "batch unit detached"
        return product

    def snapshot(self, now: float | None = None) -> dict[str, Any]:
        timestamp = self.clock() if now is None else float(now)
        product = self.product
        conveyor_actor = self.actors.get(Actor.CONVEYOR.value)
        conveyor = getattr(
            conveyor_actor,
            "state",
            {
                "phase": "IDLE",
                "position_m": 0.0,
                "target_m": 0.0,
                "travel_m": 0.0,
                "progress": 0.0,
                "moving": False,
            },
        )
        arms: dict[str, dict[str, str]] = {}
        for name in ("arm1", "arm2", "arm3"):
            main = self.active_task if self.active_task and str(self.active_task.actor) == name else None
            task = main or self.background_tasks.get(name)
            arms[name] = {
                "task_id": "" if task is None else task.task_id,
                "task_type": "" if task is None else TaskType(task.task_type).value,
                "status": "idle" if task is None else "busy",
                "error": "" if task is None else task.error or "",
            }
        if product is None:
            return _json_value(
                {
                    "status": self.status,
                    "paused": self.paused,
                    "order_id": "",
                    "preset": "",
                    "stage": "IDLE",
                    "disposition": None,
                    "arms": arms,
                    "resources": self.resources.snapshot(),
                    "fins": {},
                    "paths": {},
                    "fixture": {},
                    "conveyor": conveyor,
                    "furnace": {"status": "IDLE", "temperature_c": 25.0, "door_open": False},
                    "inspections": [],
                    "faults": [asdict(fault) for fault in self.faults],
                    "kpi": self.kpi.as_dict(timestamp),
                    "plan": None if self.process_plan is None else self.process_plan.summary(),
                }
            )
        return _json_value(
            {
                "status": self.status,
                "paused": self.paused,
                "order_id": product.order_id,
                "preset": product.spec.preset,
                "stage": product.stage.value,
                "disposition": None if product.disposition is None else product.disposition.value,
                "arms": arms,
                "resources": self.resources.snapshot(),
                "fins": {fin.fin_id: asdict(fin) for fin in product.fins},
                "paths": {path.path_id: asdict(path) for path in product.paths},
                "fixture": asdict(product.fixture),
                "conveyor": conveyor,
                "furnace": {
                    **asdict(product.furnace),
                    "status": product.furnace.phase.value,
                    "door_open": product.furnace.door_open,
                },
                "inspections": [asdict(result) for result in product.inspections],
                "faults": [asdict(fault) for fault in self.faults],
                "kpi": self.kpi.as_dict(timestamp),
                "last_error": product.errors[-1] if product.errors else "",
                "plan": None if self.process_plan is None else self.process_plan.summary(),
            }
        )


OrderCoordinator = ProcessCoordinator
BrazingCoordinator = ProcessCoordinator


__all__ = [
    "ActorResult",
    "BrazingCoordinator",
    "FaultSpec",
    "OrderCoordinator",
    "ProcessCoordinator",
    "ProcessEvent",
    "TaskActor",
    "TaskExecutor",
    "TimedTaskActor",
]
