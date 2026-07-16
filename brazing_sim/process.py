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

from .config import create_product_state, make_order_spec
from .domain import (
    Actor,
    FixtureStatus,
    OrderSpec,
    OrderStage,
    ProductState,
    TaskSpec,
    TaskStatus,
    TaskType,
)
from .furnace import DemoFurnace
from .kpi import KpiTracker
from .quality import QualityEvaluator
from .resources import ResourceManager


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
        if self.fault_type not in {"fin_pose", "brazing_gap", "furnace_profile"}:
            raise ValueError(f"unsupported fault type: {self.fault_type}")
        if self.severity not in {"recoverable", "severe"}:
            raise ValueError("fault severity must be recoverable or severe")
        if self.fault_type != "furnace_profile" and not self.target:
            raise ValueError("fin/path fault requires a target")


class TimedTaskActor:
    """Deterministic actor used by headless runs and coordinator unit tests."""

    DEFAULT_DURATIONS: Mapping[str, float] = {
        TaskType.LOAD_BASE.value: 0.35,
        TaskType.INSERT_FIN.value: 0.25,
        TaskType.ADJUST_FIN.value: 0.20,
        TaskType.PRE_INSPECT.value: 0.25,
        TaskType.APPLY_MATERIAL.value: 0.20,
        TaskType.REAPPLY_MATERIAL.value: 0.15,
        TaskType.MATERIAL_INSPECT.value: 0.25,
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
    TaskType.INSERT_FIN: Actor.ARM1,
    TaskType.ADJUST_FIN: Actor.ARM1,
    TaskType.PRE_INSPECT: Actor.ARM3,
    TaskType.APPLY_MATERIAL: Actor.ARM2,
    TaskType.REAPPLY_MATERIAL: Actor.ARM2,
    TaskType.MATERIAL_INSPECT: Actor.ARM3,
    TaskType.LOCK_FIXTURE: Actor.ARM2,
    TaskType.LOAD_FURNACE: Actor.ARM2,
    TaskType.UNLOAD_FURNACE: Actor.ARM2,
    TaskType.POST_INSPECT: Actor.ARM3,
}


TASK_RESOURCES: Mapping[TaskType, tuple[str, ...]] = {
    TaskType.LOAD_BASE: ("assembly_fixture",),
    TaskType.INSERT_FIN: ("assembly_fixture",),
    TaskType.ADJUST_FIN: ("assembly_fixture",),
    TaskType.PRE_INSPECT: ("assembly_fixture", "inspection_zone"),
    TaskType.APPLY_MATERIAL: ("assembly_fixture", "brazing_zone"),
    TaskType.REAPPLY_MATERIAL: ("assembly_fixture", "brazing_zone"),
    TaskType.MATERIAL_INSPECT: ("assembly_fixture", "inspection_zone"),
    TaskType.LOCK_FIXTURE: ("assembly_fixture",),
    TaskType.LOAD_FURNACE: ("assembly_fixture", "furnace_mouth"),
    TaskType.UNLOAD_FURNACE: ("furnace_mouth", "inspection_zone"),
    TaskType.POST_INSPECT: ("inspection_zone",),
}

TASK_TIMEOUTS: Mapping[TaskType, float] = {
    TaskType.LOAD_BASE: 300.0,
    TaskType.INSERT_FIN: 240.0,
    TaskType.ADJUST_FIN: 240.0,
    TaskType.PRE_INSPECT: 600.0,
    TaskType.MATERIAL_INSPECT: 600.0,
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
        self.quality = quality or QualityEvaluator()
        scale = 0.0 if fast else 1.0
        supplied = dict(actors or {})
        self.actors: dict[str, Any] = {}
        for actor in (Actor.ARM1, Actor.ARM2, Actor.ARM3):
            self.actors[actor.value] = supplied.get(
                actor, supplied.get(actor.value, TimedTaskActor(time_scale=scale))
            )
        self.kpi = KpiTracker(clock)
        self.furnace: DemoFurnace | None = None
        self.product: ProductState | None = None
        self.tasks: list[TaskSpec] = []
        self.task_history: list[TaskSpec] = []
        self.active_task: TaskSpec | None = None
        self.events: list[ProcessEvent] = []
        self.faults: list[FaultSpec] = []
        self._sequence = 0
        self._stage_generation = 0
        self._pending_fin_targets: tuple[str, ...] = ()
        self._pending_path_targets: tuple[str, ...] = ()
        self.paused = False
        self.pause_after_stage: OrderStage | None = None
        self.status = "idle"

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
        self.resources.reset()
        self.kpi.start_order(timestamp)
        self.product = create_product_state(selected, order_id=order_id, created_at=timestamp)
        self.furnace = DemoFurnace(selected.recipe)
        self.tasks.clear()
        self.task_history.clear()
        self.active_task = None
        self._sequence = 0
        self._stage_generation = 0
        self._pending_fin_targets = ()
        self._pending_path_targets = ()
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
            "inspection_1": OrderStage.PRE_INSPECTION,
            "arm2_motion": OrderStage.MATERIAL_APPLICATION,
            "inspection_2": OrderStage.MATERIAL_INSPECTION,
        }
        if key not in targets:
            raise ValueError(f"unknown demonstration segment: {segment}")
        product = self.start_order("A", now=timestamp)
        target = targets[key]
        self.tasks.clear()
        self.active_task = None

        if key != "pick_place":
            product.fixture.base_weld_active = True
            product.fixture.status = FixtureStatus.TEMPORARY
            for fin in product.active_fins:
                fin.inserted = True
                fin.temporary_welded = True
                fin.actual_position = fin.target_position
                product.fixture.temporary_fin_welds.add(fin.fin_id)
        if key == "inspection_2":
            for path in product.active_paths:
                path.applied = True
                path.coverage_ratio = 1.0

        product.stage = target
        self.tasks = self._tasks_for_stage(target)
        self.pause_after_stage = OrderStage.FIN_ASSEMBLY if key == "pick_place" else target
        self.kpi.enter_stage(target, timestamp)
        self._event(timestamp, "segment_started", f"segment {key} started")
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
        if fault.fault_type == "fin_pose" and not fault.target.startswith("fin_"):
            raise ValueError("fin_pose target must be fin_XX")
        if fault.fault_type == "brazing_gap" and not fault.target.startswith("fin_"):
            raise ValueError("brazing_gap target must be fin_XX_left/right")
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

    def _enter_stage(self, stage: OrderStage, now: float) -> None:
        if self.product is None:
            raise RuntimeError("no active product")
        self.product.transition(stage, now)
        self._stage_generation += 1
        self.tasks = self._tasks_for_stage(stage)
        self.kpi.enter_stage(stage, now)
        self._event(now, "stage", f"stage {stage.value}")
        if stage is OrderStage.BRAZING:
            self._start_furnace(now)

    def _tasks_for_stage(self, stage: OrderStage) -> list[TaskSpec]:
        assert self.product is not None
        if stage is OrderStage.BASE_LOADING:
            return [self._new_task(TaskType.LOAD_BASE)]
        if stage is OrderStage.FIN_ASSEMBLY:
            if self._pending_fin_targets:
                targets = self._pending_fin_targets
                self._pending_fin_targets = ()
                return [self._new_task(TaskType.ADJUST_FIN, {"fin_id": target}) for target in targets]
            return [
                self._new_task(TaskType.INSERT_FIN, {"fin_id": fin.fin_id})
                for fin in self.product.active_fins
                if not fin.inserted
            ]
        if stage is OrderStage.PRE_INSPECTION:
            return [self._new_task(TaskType.PRE_INSPECT)]
        if stage is OrderStage.MATERIAL_APPLICATION:
            if self._pending_path_targets:
                targets = self._pending_path_targets
                self._pending_path_targets = ()
                return [self._new_task(TaskType.REAPPLY_MATERIAL, {"path_id": target}) for target in targets]
            return [
                self._new_task(TaskType.APPLY_MATERIAL, {"path_id": path.path_id})
                for path in self.product.active_paths
                if not path.applied
            ]
        if stage is OrderStage.MATERIAL_INSPECTION:
            return [self._new_task(TaskType.MATERIAL_INSPECT)]
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
        if self.product is not None:
            self.product.fail(f"{task.task_id}: {reason}", now)
        self._event(now, "task_failed", reason, task)

    def _apply_armed_fault(self, kind: str) -> None:
        assert self.product is not None
        for fault in self.faults:
            if not fault.armed or fault.applied or fault.fault_type != kind:
                continue
            if kind == "fin_pose":
                fin = next((item for item in self.product.active_fins if item.fin_id == fault.target), None)
                if fin is None:
                    raise ValueError(f"unknown active fin: {fault.target}")
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
                path = next(
                    (item for item in self.product.active_paths if item.path_id == fault.target), None
                )
                if path is None:
                    raise ValueError(f"unknown active path: {fault.target}")
                path.coverage_ratio = 0.70 if fault.severity == "recoverable" else 0.20
                path.longest_gap_m = 0.020 if fault.severity == "recoverable" else 0.100
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
        if kind is TaskType.LOAD_BASE:
            self.product.fixture.base_weld_active = True
            self.product.fixture.status = FixtureStatus.BASE_FIXED
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
        elif kind is TaskType.PRE_INSPECT:
            self._apply_armed_fault("fin_pose")
            result = self.quality.pre_inspection(self.product, now)
            if not result.passed:
                if not self.quality.register_automatic_rework(self.product, result):
                    return
                self.kpi.record_rework("fin", len(result.rework_targets))
                self._pending_fin_targets = result.rework_targets
        elif kind in {TaskType.APPLY_MATERIAL, TaskType.REAPPLY_MATERIAL}:
            path_id = str(task.payload["path_id"])
            path = next(item for item in self.product.active_paths if item.path_id == path_id)
            path.applied = True
            path.coverage_ratio = 1.0
            path.longest_gap_m = 0.0
            path.lateral_error_m = 0.0
            path.trajectory_rmse_m = float(task.payload.get("trajectory_rmse_m", 0.0))
            path.trajectory_max_error_m = float(task.payload.get("trajectory_max_error_m", 0.0))
        elif kind is TaskType.MATERIAL_INSPECT:
            self._apply_armed_fault("brazing_gap")
            result = self.quality.material_inspection(self.product, now)
            if not result.passed:
                if not self.quality.register_automatic_rework(self.product, result):
                    return
                self.kpi.record_rework("material", len(result.rework_targets))
                self._pending_path_targets = result.rework_targets
        elif kind is TaskType.LOCK_FIXTURE:
            self.product.fixture.lock()
        elif kind is TaskType.LOAD_FURNACE:
            self.product.fixture.in_furnace = True
            self.product.fixture.status = FixtureStatus.IN_FURNACE
        elif kind is TaskType.UNLOAD_FURNACE:
            self.product.fixture.in_furnace = False
        elif kind is TaskType.POST_INSPECT:
            result = self.quality.post_inspection(self.product, now)
            self.kpi.set_final_quality(result.score)
            self.kpi.finish_order(now, result.score)

    def _start_furnace(self, now: float) -> None:
        assert self.product is not None and self.furnace is not None
        if not self.product.fixture.locked or not self.product.fixture.in_furnace:
            self.product.fail("furnace interlock: fixture must be locked and loaded", now)
            return
        self.furnace.start(now, fault=self._furnace_fault())
        self.product.furnace = self.furnace.state
        self._event(now, "furnace_started", "virtual brazing cycle started")

    def _advance_after_stage(self, now: float) -> None:
        assert self.product is not None
        stage = self.product.stage
        if stage is OrderStage.BASE_LOADING:
            self._enter_stage(OrderStage.FIN_ASSEMBLY, now)
        elif stage is OrderStage.FIN_ASSEMBLY:
            self._enter_stage(OrderStage.PRE_INSPECTION, now)
        elif stage is OrderStage.PRE_INSPECTION:
            if self.product.stage is OrderStage.MANUAL_REVIEW:
                return
            if self._pending_fin_targets:
                self._enter_stage(OrderStage.FIN_ASSEMBLY, now)
            else:
                self._enter_stage(OrderStage.MATERIAL_APPLICATION, now)
        elif stage is OrderStage.MATERIAL_APPLICATION:
            self._enter_stage(OrderStage.MATERIAL_INSPECTION, now)
        elif stage is OrderStage.MATERIAL_INSPECTION:
            if self.product.stage is OrderStage.MANUAL_REVIEW:
                return
            if self._pending_path_targets:
                self._enter_stage(OrderStage.MATERIAL_APPLICATION, now)
            else:
                self._enter_stage(OrderStage.FIXTURE_LOCKING, now)
        elif stage is OrderStage.FIXTURE_LOCKING:
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
        if self.product.stage is OrderStage.BRAZING:
            assert self.furnace is not None
            self.furnace.update(timestamp)
            self.product.furnace = self.furnace.state
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
        if self.active_task is None and self.tasks:
            task = self.tasks[0]
            if self._acquire(task, timestamp):
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
        self.faults.clear()
        self.events.clear()
        self.kpi.reset()
        self.status = "idle"
        self.paused = False
        self.pause_after_stage = None

    def snapshot(self, now: float | None = None) -> dict[str, Any]:
        timestamp = self.clock() if now is None else float(now)
        product = self.product
        arms = {
            name: {
                "task_id": (
                    self.active_task.task_id
                    if self.active_task and str(self.active_task.actor) == name
                    else ""
                ),
                "task_type": (
                    self.active_task.task_type.value
                    if self.active_task and str(self.active_task.actor) == name
                    else ""
                ),
                "status": "busy" if self.active_task and str(self.active_task.actor) == name else "idle",
                "error": "",
            }
            for name in ("arm1", "arm2", "arm3")
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
                    "furnace": {"status": "IDLE", "temperature_c": 25.0, "door_open": False},
                    "inspections": [],
                    "faults": [asdict(fault) for fault in self.faults],
                    "kpi": self.kpi.as_dict(timestamp),
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
                "furnace": {
                    **asdict(product.furnace),
                    "status": product.furnace.phase.value,
                    "door_open": product.furnace.door_open,
                },
                "inspections": [asdict(result) for result in product.inspections],
                "faults": [asdict(fault) for fault in self.faults],
                "kpi": self.kpi.as_dict(timestamp),
                "last_error": product.errors[-1] if product.errors else "",
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
