"""Strongly typed domain objects for the brazing-line simulation.

The module intentionally has no MuJoCo or UI dependency.  It is shared by the
headless coordinator, HTTP facade, Qt dashboard and the scene adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, Iterable, Mapping

Vec3 = tuple[float, float, float]
MAX_AUTOMATIC_REWORKS = 2


class StrEnum(str, Enum):
    """Python 3.10 compatible string enum."""

    def __str__(self) -> str:
        return self.value


class OrderStage(StrEnum):
    CREATED = "CREATED"
    BASE_LOADING = "BASE_LOADING"
    FIN_ASSEMBLY = "FIN_ASSEMBLY"
    PRE_INSPECTION = "PRE_INSPECTION"
    MATERIAL_APPLICATION = "MATERIAL_APPLICATION"
    MATERIAL_INSPECTION = "MATERIAL_INSPECTION"
    FIXTURE_LOCKING = "FIXTURE_LOCKING"
    FURNACE_LOADING = "FURNACE_LOADING"
    BRAZING = "BRAZING"
    UNLOADING = "UNLOADING"
    POST_INSPECTION = "POST_INSPECTION"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    PASS = "PASS"
    REWORK_REQUIRED = "REWORK_REQUIRED"
    SCRAPPED = "SCRAPPED"
    ERROR = "ERROR"
    STOPPED = "STOPPED"

    @property
    def terminal(self) -> bool:
        return self in TERMINAL_STAGES


TERMINAL_STAGES = frozenset(
    {
        OrderStage.MANUAL_REVIEW,
        OrderStage.PASS,
        OrderStage.REWORK_REQUIRED,
        OrderStage.SCRAPPED,
        OrderStage.ERROR,
        OrderStage.STOPPED,
    }
)


class TerminalDisposition(StrEnum):
    PASS = "PASS"
    REWORK_REQUIRED = "REWORK_REQUIRED"
    SCRAPPED = "SCRAPPED"


class BrazingSide(StrEnum):
    LEFT = "left"
    RIGHT = "right"


class InspectionKind(StrEnum):
    PRE_BRAZE = "PRE_BRAZE"
    MATERIAL = "MATERIAL"
    POST_BRAZE = "POST_BRAZE"


class FixtureStatus(StrEnum):
    EMPTY = "EMPTY"
    BASE_FIXED = "BASE_FIXED"
    TEMPORARY = "TEMPORARY"
    LOCKED = "LOCKED"
    IN_FURNACE = "IN_FURNACE"
    RELEASED = "RELEASED"


class FurnacePhase(StrEnum):
    IDLE = "IDLE"
    DOOR_OPENING = "DOOR_OPENING"
    LOADING = "LOADING"
    DOOR_CLOSING = "DOOR_CLOSING"
    READY = "READY"
    PREHEAT = "PREHEAT"
    RAMP = "RAMP"
    SOAK = "SOAK"
    COOLING = "COOLING"
    UNLOADING = "UNLOADING"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"
    STOPPED = "STOPPED"


class Actor(StrEnum):
    ARM1 = "arm1"
    ARM2 = "arm2"
    ARM3 = "arm3"
    FURNACE = "furnace"
    INSPECTION = "inspection"
    COORDINATOR = "coordinator"


class TaskType(StrEnum):
    LOAD_BASE = "LOAD_BASE"
    INSERT_FIN = "INSERT_FIN"
    ADJUST_FIN = "ADJUST_FIN"
    PRE_INSPECT = "PRE_INSPECT"
    APPLY_MATERIAL = "APPLY_MATERIAL"
    REAPPLY_MATERIAL = "REAPPLY_MATERIAL"
    MATERIAL_INSPECT = "MATERIAL_INSPECT"
    LOCK_FIXTURE = "LOCK_FIXTURE"
    LOAD_FURNACE = "LOAD_FURNACE"
    RUN_FURNACE = "RUN_FURNACE"
    UNLOAD_FURNACE = "UNLOAD_FURNACE"
    POST_INSPECT = "POST_INSPECT"
    PARK = "PARK"


class TaskStatus(StrEnum):
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    @property
    def terminal(self) -> bool:
        return self in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED}


@dataclass(frozen=True, slots=True)
class BrazingRecipe:
    """Demonstration recipe; these values are not production process data."""

    ambient_c: float = 25.0
    preheat_c: float = 150.0
    peak_c: float = 600.0
    unload_c: float = 80.0
    preheat_seconds: float = 2.0
    ramp_seconds: float = 3.0
    soak_seconds: float = 3.0
    cooling_seconds: float = 4.0
    door_seconds: float = 0.75

    def __post_init__(self) -> None:
        temperatures = (self.ambient_c, self.preheat_c, self.peak_c, self.unload_c)
        durations = (
            self.preheat_seconds,
            self.ramp_seconds,
            self.soak_seconds,
            self.cooling_seconds,
            self.door_seconds,
        )
        if not all(isfinite(value) for value in temperatures + durations):
            raise ValueError("recipe values must be finite")
        if any(value <= 0 for value in durations):
            raise ValueError("recipe durations must be positive")
        if not self.ambient_c < self.preheat_c < self.peak_c:
            raise ValueError("recipe temperatures must satisfy ambient < preheat < peak")
        if self.unload_c >= self.peak_c:
            raise ValueError("unload temperature must be below peak temperature")

    @property
    def process_seconds(self) -> float:
        return self.preheat_seconds + self.ramp_seconds + self.soak_seconds + self.cooling_seconds


@dataclass(frozen=True, slots=True)
class InspectionConfig:
    fin_position_m: float = 0.003
    fin_verticality_deg: float = 3.0
    root_gap_m: float = 0.0015
    pitch_error_m: float = 0.002
    coverage_ratio: float = 0.95
    longest_material_gap_m: float = 0.005
    lateral_error_m: float = 0.002
    trajectory_rmse_m: float = 0.003
    trajectory_max_error_m: float = 0.005
    pass_score: float = 0.90
    rework_score: float = 0.75

    def __post_init__(self) -> None:
        limits = (
            self.fin_position_m,
            self.fin_verticality_deg,
            self.root_gap_m,
            self.pitch_error_m,
            self.longest_material_gap_m,
            self.lateral_error_m,
            self.trajectory_rmse_m,
            self.trajectory_max_error_m,
        )
        if any(not isfinite(value) or value <= 0 for value in limits):
            raise ValueError("inspection tolerances must be finite and positive")
        if not 0.0 < self.coverage_ratio <= 1.0:
            raise ValueError("coverage_ratio must be in (0, 1]")
        if not 0.0 <= self.rework_score < self.pass_score <= 1.0:
            raise ValueError("score thresholds must satisfy 0 <= rework < pass <= 1")


@dataclass(frozen=True, slots=True)
class OrderSpec:
    preset: str
    base_size: Vec3
    fin_size: Vec3
    fin_count: int
    fin_pitch: float
    brazing_sides: tuple[BrazingSide, ...] = (BrazingSide.LEFT, BrazingSide.RIGHT)
    path_width: float = 0.004
    max_fins: int = 8
    max_paths: int = 16
    recipe: BrazingRecipe = field(default_factory=BrazingRecipe)
    inspection: InspectionConfig = field(default_factory=InspectionConfig)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "brazing_sides",
            tuple(BrazingSide(side) for side in self.brazing_sides),
        )
        if not self.preset.strip():
            raise ValueError("preset must not be empty")
        if len(self.base_size) != 3 or len(self.fin_size) != 3:
            raise ValueError("base_size and fin_size must contain x/y/z")
        if any(not isfinite(value) or value <= 0 for value in self.base_size + self.fin_size):
            raise ValueError("part dimensions must be finite and positive")
        if self.fin_count <= 0 or self.fin_count > self.max_fins:
            raise ValueError("fin_count must be between 1 and max_fins")
        if self.max_fins <= 0 or self.max_paths <= 0:
            raise ValueError("preallocation limits must be positive")
        if not self.brazing_sides:
            raise ValueError("at least one brazing side is required")
        if len(set(self.brazing_sides)) != len(self.brazing_sides):
            raise ValueError("brazing sides must be unique")
        if self.path_count > self.max_paths:
            raise ValueError("requested brazing paths exceed max_paths")
        if not isfinite(self.fin_pitch) or self.fin_pitch <= 0:
            raise ValueError("fin_pitch must be finite and positive")
        if not isfinite(self.path_width) or self.path_width <= 0:
            raise ValueError("path_width must be finite and positive")
        occupied_width = (self.fin_count - 1) * self.fin_pitch + self.fin_size[1]
        if occupied_width > self.base_size[1]:
            raise ValueError("fin array does not fit on the base width")
        if self.fin_size[0] > self.base_size[0]:
            raise ValueError("fin length must fit on the base length")

    @property
    def path_count(self) -> int:
        return self.fin_count * len(self.brazing_sides)

    @property
    def base_length(self) -> float:
        return self.base_size[0]

    @property
    def base_width(self) -> float:
        return self.base_size[1]

    @property
    def base_thickness(self) -> float:
        return self.base_size[2]

    @property
    def fin_length(self) -> float:
        return self.fin_size[0]

    @property
    def fin_thickness(self) -> float:
        return self.fin_size[1]

    @property
    def fin_height(self) -> float:
        return self.fin_size[2]

    @property
    def base_dimensions(self) -> Vec3:
        return self.base_size

    @property
    def fin_dimensions(self) -> Vec3:
        return self.fin_size

    @property
    def recipe_config(self) -> BrazingRecipe:
        return self.recipe

    @property
    def inspection_config(self) -> InspectionConfig:
        return self.inspection


@dataclass(slots=True)
class FinState:
    fin_id: str
    index: int
    target_position: Vec3
    actual_position: Vec3
    active: bool = True
    hidden: bool = False
    collision_enabled: bool = True
    position_error_m: float = 0.0
    verticality_error_deg: float = 0.0
    root_gap_m: float = 0.0
    pitch_error_m: float = 0.0
    inserted: bool = False
    temporary_welded: bool = False
    board_welded: bool = False
    rework_attempts: int = 0

    @property
    def id(self) -> str:
        return self.fin_id

    @property
    def name(self) -> str:
        return self.fin_id

    @property
    def local_position(self) -> Vec3:
        return self.target_position

    @property
    def world_position(self) -> Vec3:
        return self.actual_position

    @property
    def can_rework(self) -> bool:
        return self.rework_attempts < MAX_AUTOMATIC_REWORKS

    def register_rework(self) -> bool:
        if not self.can_rework:
            return False
        self.rework_attempts += 1
        return True


@dataclass(slots=True)
class BrazingPathState:
    path_id: str
    index: int
    fin_id: str
    side: BrazingSide
    local_start: Vec3
    local_end: Vec3
    target_width_m: float
    active: bool = True
    hidden: bool = False
    collision_enabled: bool = True
    applied: bool = False
    coverage_ratio: float = 0.0
    longest_gap_m: float = 0.0
    lateral_error_m: float = 0.0
    trajectory_rmse_m: float = 0.0
    trajectory_max_error_m: float = 0.0
    rework_attempts: int = 0

    @property
    def id(self) -> str:
        return self.path_id

    @property
    def name(self) -> str:
        return f"brazing_path_{self.path_id}"

    @property
    def local_position(self) -> Vec3:
        return tuple((start + end) / 2.0 for start, end in zip(self.local_start, self.local_end))  # type: ignore[return-value]

    @property
    def world_position(self) -> Vec3:
        return self.local_position

    @property
    def can_rework(self) -> bool:
        return self.rework_attempts < MAX_AUTOMATIC_REWORKS

    def register_rework(self) -> bool:
        if not self.can_rework:
            return False
        self.rework_attempts += 1
        return True


@dataclass(slots=True)
class FixtureState:
    status: FixtureStatus = FixtureStatus.EMPTY
    base_weld_active: bool = False
    temporary_fin_welds: set[str] = field(default_factory=set)
    locked: bool = False
    cycle_locked: bool = False
    on_transfer_tool: bool = False
    in_furnace: bool = False

    def lock(self) -> None:
        if not self.base_weld_active:
            raise RuntimeError("base must be fixed before the fixture can be locked")
        self.locked = True
        self.cycle_locked = True
        self.status = FixtureStatus.LOCKED

    def release(self) -> None:
        self.locked = False
        self.on_transfer_tool = False
        self.in_furnace = False
        self.status = FixtureStatus.RELEASED

    @property
    def state(self) -> FixtureStatus:
        return self.status


@dataclass(slots=True)
class FurnaceState:
    phase: FurnacePhase = FurnacePhase.IDLE
    temperature_c: float = 25.0
    target_temperature_c: float = 25.0
    door_fraction: float = 0.0
    workpiece_loaded: bool = False
    elapsed_seconds: float = 0.0
    phase_started_at: float = 0.0
    cycle_started_at: float | None = None
    peak_temperature_c: float = 25.0
    profile_score: float = 1.0
    profile_fault: str | None = None
    severe_violation: bool = False
    completed_at: float | None = None
    error: str | None = None

    @property
    def door_open(self) -> bool:
        return self.door_fraction >= 1.0 - 1e-9

    @property
    def door_closed(self) -> bool:
        return self.door_fraction <= 1e-9

    @property
    def complete(self) -> bool:
        return self.phase is FurnacePhase.COMPLETE

    @property
    def status(self) -> FurnacePhase:
        return self.phase

    @property
    def temperature(self) -> float:
        return self.temperature_c


@dataclass(slots=True)
class InspectionResult:
    kind: InspectionKind
    passed: bool
    metrics: dict[str, float | int | bool | str] = field(default_factory=dict)
    hard_failures: tuple[str, ...] = ()
    rework_targets: tuple[str, ...] = ()
    score: float | None = None
    disposition: TerminalDisposition | None = None
    timestamp: float = 0.0

    @property
    def requires_rework(self) -> bool:
        return bool(self.rework_targets) or self.disposition is TerminalDisposition.REWORK_REQUIRED


@dataclass(slots=True)
class KpiSnapshot:
    order_cycle_seconds: float = 0.0
    assembly_seconds: float = 0.0
    material_seconds: float = 0.0
    furnace_seconds: float = 0.0
    inspection_seconds: float = 0.0
    actor_busy_seconds: dict[str, float] = field(default_factory=dict)
    actor_wait_seconds: dict[str, float] = field(default_factory=dict)
    resource_wait_seconds: float = 0.0
    resource_conflicts: int = 0
    fin_reworks: int = 0
    material_reworks: int = 0
    path_rmse_m: float = 0.0
    path_max_error_m: float = 0.0
    final_quality_score: float | None = None
    # Adapter fields emitted by ``KpiTracker``.  The SI-unit fields above are
    # authoritative; these keep API/UI snapshots lossless during migration.
    order_elapsed: float = 0.0
    phase_durations: dict[str, float] = field(default_factory=dict)
    actor_busy: dict[str, float] = field(default_factory=dict)
    actor_waiting: dict[str, float] = field(default_factory=dict)
    resource_waits: int = 0
    rework_counts: dict[str, int] = field(default_factory=dict)
    path_rmse_mm: float = 0.0
    path_max_error_mm: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "order_cycle_seconds": self.order_cycle_seconds,
            "assembly_seconds": self.assembly_seconds,
            "material_seconds": self.material_seconds,
            "furnace_seconds": self.furnace_seconds,
            "inspection_seconds": self.inspection_seconds,
            "actor_busy_seconds": dict(self.actor_busy_seconds),
            "actor_wait_seconds": dict(self.actor_wait_seconds),
            "resource_wait_seconds": self.resource_wait_seconds,
            "resource_conflicts": self.resource_conflicts,
            "fin_reworks": self.fin_reworks,
            "material_reworks": self.material_reworks,
            "path_rmse_m": self.path_rmse_m,
            "path_max_error_m": self.path_max_error_m,
            "final_quality_score": self.final_quality_score,
            "order_elapsed": self.order_elapsed,
            "phase_durations": dict(self.phase_durations),
            "actor_busy": dict(self.actor_busy),
            "actor_waiting": dict(self.actor_waiting),
            "resource_waits": self.resource_waits,
            "rework_counts": dict(self.rework_counts),
            "path_rmse_mm": self.path_rmse_mm,
            "path_max_error_mm": self.path_max_error_mm,
        }


@dataclass(slots=True)
class TaskSpec:
    task_id: str
    actor: Actor | str
    task_type: TaskType | str
    resource: str | None = None
    resources: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    payload: dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    retries: int = 0
    max_retries: int = 0
    timeout: float = 30.0
    started_at: float | None = None
    completed_at: float | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id must not be empty")
        if self.resource is not None and self.resource not in self.resources:
            self.resources = (self.resource, *self.resources)
        self.resources = tuple(dict.fromkeys(str(resource) for resource in self.resources))
        self.dependencies = tuple(dict.fromkeys(self.dependencies))
        self.status = TaskStatus(self.status)
        if self.max_retries < 0 or self.retries < 0:
            raise ValueError("retry counts must be non-negative")
        if not isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be finite and positive")

    @property
    def can_retry(self) -> bool:
        return self.retries < self.max_retries

    def mark_running(self, now: float) -> None:
        if self.status.terminal:
            raise RuntimeError("terminal task cannot be restarted")
        self.status = TaskStatus.RUNNING
        self.started_at = now

    def mark_succeeded(self, now: float) -> None:
        if self.status.terminal:
            return
        self.status = TaskStatus.SUCCEEDED
        self.completed_at = now

    def mark_failed(self, now: float, error: str) -> None:
        """Record a failure once; repeated actor error events are idempotent."""

        if self.status.terminal:
            return
        self.status = TaskStatus.FAILED
        self.error = error
        self.completed_at = now

    def prepare_retry(self) -> bool:
        if self.status is not TaskStatus.FAILED or not self.can_retry:
            return False
        self.retries += 1
        self.status = TaskStatus.READY
        self.started_at = None
        self.completed_at = None
        self.error = None
        return True


class StateTransitionError(RuntimeError):
    pass


_NORMAL_TRANSITIONS: Mapping[OrderStage, frozenset[OrderStage]] = {
    OrderStage.CREATED: frozenset({OrderStage.BASE_LOADING}),
    OrderStage.BASE_LOADING: frozenset({OrderStage.FIN_ASSEMBLY}),
    OrderStage.FIN_ASSEMBLY: frozenset({OrderStage.PRE_INSPECTION}),
    OrderStage.PRE_INSPECTION: frozenset(
        {OrderStage.FIN_ASSEMBLY, OrderStage.MATERIAL_APPLICATION, OrderStage.MANUAL_REVIEW}
    ),
    OrderStage.MATERIAL_APPLICATION: frozenset({OrderStage.MATERIAL_INSPECTION}),
    OrderStage.MATERIAL_INSPECTION: frozenset(
        {OrderStage.MATERIAL_APPLICATION, OrderStage.FIXTURE_LOCKING, OrderStage.MANUAL_REVIEW}
    ),
    OrderStage.FIXTURE_LOCKING: frozenset({OrderStage.FURNACE_LOADING}),
    OrderStage.FURNACE_LOADING: frozenset({OrderStage.BRAZING}),
    OrderStage.BRAZING: frozenset({OrderStage.UNLOADING}),
    OrderStage.UNLOADING: frozenset({OrderStage.POST_INSPECTION}),
    OrderStage.POST_INSPECTION: frozenset({OrderStage.PASS, OrderStage.REWORK_REQUIRED, OrderStage.SCRAPPED}),
}


@dataclass(slots=True)
class ProductState:
    order_id: str
    spec: OrderSpec
    fins: list[FinState]
    paths: list[BrazingPathState]
    stage: OrderStage = OrderStage.CREATED
    disposition: TerminalDisposition | None = None
    fixture: FixtureState = field(default_factory=FixtureState)
    furnace: FurnaceState = field(default_factory=FurnaceState)
    inspections: list[InspectionResult] = field(default_factory=list)
    kpi: KpiSnapshot = field(default_factory=KpiSnapshot)
    errors: list[str] = field(default_factory=list)
    created_at: float = 0.0
    completed_at: float | None = None

    def __post_init__(self) -> None:
        self.stage = OrderStage(self.stage)
        if self.disposition is not None:
            self.disposition = TerminalDisposition(self.disposition)

    @property
    def active_fins(self) -> list[FinState]:
        return [fin for fin in self.fins if fin.active]

    @property
    def active_paths(self) -> list[BrazingPathState]:
        return [path for path in self.paths if path.active]

    @property
    def terminal(self) -> bool:
        return self.stage.terminal

    def transition(self, target: OrderStage | str, now: float | None = None) -> None:
        target = OrderStage(target)
        if target is self.stage:
            return
        allowed = set(_NORMAL_TRANSITIONS.get(self.stage, frozenset()))
        if not self.stage.terminal:
            allowed.update({OrderStage.ERROR, OrderStage.STOPPED})
        if target not in allowed:
            raise StateTransitionError(f"illegal order transition: {self.stage.value} -> {target.value}")
        self.stage = target
        if target in {OrderStage.PASS, OrderStage.REWORK_REQUIRED, OrderStage.SCRAPPED}:
            self.disposition = TerminalDisposition(target.value)
        if target.terminal and now is not None:
            self.completed_at = now
            self.kpi.order_cycle_seconds = max(0.0, now - self.created_at)

    def record_rework(self, target_id: str) -> bool:
        target: FinState | BrazingPathState | None = next(
            (fin for fin in self.active_fins if fin.fin_id == target_id), None
        )
        if target is None:
            target = next((path for path in self.active_paths if path.path_id == target_id), None)
        if target is None:
            raise KeyError(f"unknown rework target: {target_id}")
        accepted = target.register_rework()
        if accepted:
            if isinstance(target, FinState):
                self.kpi.fin_reworks += 1
            else:
                self.kpi.material_reworks += 1
        elif not self.stage.terminal:
            self.stage = OrderStage.MANUAL_REVIEW
        return accepted

    def add_inspection(self, result: InspectionResult) -> InspectionResult:
        self.inspections.append(result)
        return result

    def stop(self, now: float | None = None) -> None:
        if not self.stage.terminal:
            self.transition(OrderStage.STOPPED, now)

    def fail(self, message: str, now: float | None = None) -> None:
        self.errors.append(message)
        if not self.stage.terminal:
            self.transition(OrderStage.ERROR, now)


# Backward/adapter-friendly names used by scene and coordinator modules.
ProductStage = OrderStage
WorkflowStage = OrderStage
FinalDisposition = TerminalDisposition
RecipeConfig = BrazingRecipe


def active_ids(items: Iterable[FinState | BrazingPathState]) -> tuple[str, ...]:
    return tuple(item.id for item in items if item.active)
