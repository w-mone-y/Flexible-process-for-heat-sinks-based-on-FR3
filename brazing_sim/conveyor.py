"""Simulation-clock-driven Table2-to-furnace conveyor actor."""

from __future__ import annotations

from enum import Enum
from typing import Callable

from .domain import ProductState, TaskSpec, TaskStatus, TaskType
from .profiles import quintic_time_scaling


class ConveyorPhase(str, Enum):
    IDLE = "IDLE"
    OUTBOUND = "OUTBOUND"
    AT_FURNACE = "AT_FURNACE"
    RETURNING = "RETURNING"
    RETURNED = "RETURNED"
    PAUSED = "PAUSED"
    ERROR = "ERROR"


class ConveyorTaskActor:
    """Move the locked fixture along one actuator-driven straight slide.

    The actor sends an S-curve position reference on every simulation tick.
    It never writes the tray pose and never sleeps, so the tray, plate, fins,
    combs and press remain one continuously moving constrained assembly.
    """

    MAX_SPEED_M_S = 0.18
    POSITION_TOLERANCE_M = 0.0015
    VELOCITY_TOLERANCE_M_S = 0.015
    SETTLE_SECONDS = 0.20

    def __init__(
        self,
        scene: object,
        product: Callable[[], ProductState | None],
        *,
        fast: bool = False,
    ) -> None:
        self.scene = scene
        self._product_source = product
        self.fast = bool(fast)
        self.task: TaskSpec | None = None
        self.phase = ConveyorPhase.IDLE
        self.error = ""
        self._started_at = 0.0
        self._start_m = 0.0
        self._target_m = 0.0
        self._duration_s = 0.0
        self._settled_at: float | None = None

    def _product(self) -> ProductState:
        product = self._product_source()
        if product is None:
            raise RuntimeError("当前没有活动订单，传送带无法运行")
        return product

    def _validate_start(self, kind: TaskType, product: ProductState) -> None:
        registry = self.scene.registry
        if not product.fixture.locked or not product.fixture.press_force_held:
            raise RuntimeError("传送带启动前必须完成横梁压紧和夹具锁定")
        if not bool(self.scene.data.eq_active[registry.equality_id("tray_fixture_weld")]):
            raise RuntimeError("传送前托盘必须刚性连接到传送滑台")
        if not all(fin.inserted for fin in product.active_fins):
            raise RuntimeError("所有活动翅片安装完成后才能启动传送带")
        if not product.furnace.door_open:
            raise RuntimeError("炉门完全打开后传送带才能运行")
        position = registry.conveyor_position_m
        travel = registry.conveyor_travel_m
        if not -self.POSITION_TOLERANCE_M <= position <= travel + self.POSITION_TOLERANCE_M:
            raise RuntimeError("传送滑台超出有效行程")
        if kind is TaskType.UNLOAD_FURNACE:
            if not product.fixture.in_furnace:
                raise RuntimeError("托盘尚未进入炉内，不能执行返程")
        # A stopped task is restarted from its measured slide position.  This
        # keeps Stop/Continue resumable without snapping the tray to either
        # endpoint; a fresh load still starts at zero after scene.reset().

    def start_task(self, task: TaskSpec, now: float) -> None:
        if self.task is not None:
            raise RuntimeError(f"传送带正在执行 {self.task.task_id}")
        kind = TaskType(task.task_type)
        if kind not in {TaskType.LOAD_FURNACE, TaskType.UNLOAD_FURNACE}:
            raise RuntimeError(f"传送带不支持任务 {kind.value}")
        product = self._product()
        self._validate_start(kind, product)

        registry = self.scene.registry
        self.task = task
        self.error = ""
        self._started_at = float(now)
        self._start_m = registry.conveyor_position_m
        self._target_m = registry.conveyor_travel_m if kind is TaskType.LOAD_FURNACE else 0.0
        distance = abs(self._target_m - self._start_m)
        # smoothstep peak speed is 1.5 * distance / duration.
        self._duration_s = max(0.20, 1.5 * distance / self.MAX_SPEED_M_S)
        self._settled_at = None
        self.phase = ConveyorPhase.OUTBOUND if kind is TaskType.LOAD_FURNACE else ConveyorPhase.RETURNING
        if self.fast:
            registry.set_conveyor_target(self._target_m, teleport=True)
        else:
            registry.set_conveyor_target(self._start_m)

    def poll_task(self, now: float) -> TaskStatus:
        if self.task is None:
            return TaskStatus.SUCCEEDED
        if self.error:
            return TaskStatus.FAILED

        timestamp = float(now)
        registry = self.scene.registry
        if not self.fast:
            elapsed = max(0.0, timestamp - self._started_at)
            progress = quintic_time_scaling(elapsed / self._duration_s)
            command = self._start_m + (self._target_m - self._start_m) * progress
            registry.set_conveyor_target(command)

        position_error = abs(registry.conveyor_position_m - self._target_m)
        velocity = abs(registry.conveyor_velocity_m_s)
        at_rest = position_error <= self.POSITION_TOLERANCE_M and velocity <= self.VELOCITY_TOLERANCE_M_S
        if self.fast:
            at_rest = True
        if not at_rest:
            self._settled_at = None
            return TaskStatus.RUNNING
        if self._settled_at is None:
            self._settled_at = timestamp
            return TaskStatus.RUNNING
        if timestamp - self._settled_at < (0.0 if self.fast else self.SETTLE_SECONDS):
            return TaskStatus.RUNNING

        kind = TaskType(self.task.task_type)
        registry.set_conveyor_target(self._target_m)
        self.phase = ConveyorPhase.AT_FURNACE if kind is TaskType.LOAD_FURNACE else ConveyorPhase.RETURNED
        self.task = None
        return TaskStatus.SUCCEEDED

    def cancel(self) -> None:
        if self.task is not None:
            self.scene.registry.set_conveyor_target(self.scene.registry.conveyor_position_m)
            self.phase = ConveyorPhase.PAUSED
        self.task = None
        self.error = ""
        self._settled_at = None

    @property
    def state(self) -> dict[str, float | str | bool]:
        registry = self.scene.registry
        travel = max(1.0e-12, registry.conveyor_travel_m)
        return {
            "phase": self.phase.value,
            "position_m": registry.conveyor_position_m,
            "target_m": self._target_m,
            "travel_m": travel,
            "progress": registry.conveyor_position_m / travel,
            "moving": self.task is not None,
        }


__all__ = ["ConveyorPhase", "ConveyorTaskActor"]
