"""Visible downstream logistics for complete asynchronous-line orders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..planning.task_models import ManufacturingTask, TaskType
from ..profiles import quintic_time_scaling
from .skill_registry import SkillExecutionResult


def _tray_index(tray_id: str | None) -> int:
    value = str(tray_id or "").strip().lower()
    if value not in {"tray_01", "tray_02", "tray_03"}:
        raise RuntimeError(f"物流任务没有有效托盘：{tray_id or '空'}")
    return int(value[-2:]) - 1


@dataclass(slots=True)
class _AxisMove:
    joint: str
    actuator: str
    start: float
    target: float
    started_at: float
    duration_s: float
    settled_at: float | None = None


class AsyncLineLogisticsSkill:
    """Drive the furnace conveyor, rack, door and finished-goods outlet.

    The scheduler tasks remain deliberately small, but every one now owns a
    visible physical milestone.  Rigid-pallet weld ownership changes only at
    coincident, stopped handoff points; motion between them is actuator driven.
    """

    POSITION_TOLERANCE_M = 0.003
    VELOCITY_TOLERANCE_M_S = 0.04
    SETTLE_S = 0.08
    # The rack-infeed origin moved +250 mm away from S3.  The furnace also
    # moves 100 mm outward so its closed door remains clear of the complete
    # legacy tray handle envelope; a 690 mm stroke reaches x=1.69 m without
    # bringing the returning pallet through the fin-assembly envelope.
    OUTFEED_INSIDE_M = 0.690
    OUTPUT_INSPECTION_M = 0.100
    OUTPUT_DELIVERY_M = 1.120
    GATE_OPEN_M = 0.500
    DOOR_SECONDS = 0.75

    def __init__(self, task_type: TaskType, *, fast: bool = False) -> None:
        self.task_type = TaskType(task_type)
        self.fast = bool(fast)
        self.task: ManufacturingTask | None = None
        self.context: Any = None
        self.started_at = 0.0
        self.phase = ""
        self.phase_started_at = 0.0
        self.axis: _AxisMove | None = None
        self.cancelled = False
        self.payload_removed = False

    @property
    def scene(self) -> Any:
        return self.context.scene

    @property
    def registry(self) -> Any:
        return self.scene.registry

    @property
    def unit_index(self) -> int:
        assert self.task is not None
        return _tray_index(self.task.tray_id)

    @property
    def layer_index(self) -> int:
        assert self.task is not None
        return int(self.task.payload.get("layer_index", self.unit_index))

    @property
    def tray_name(self) -> str:
        return f"batch_tray_{self.unit_index + 1:02d}"

    def _begin_axis(
        self,
        joint: str,
        actuator: str,
        target: float,
        now: float,
        duration_s: float,
    ) -> None:
        start = self.registry.batch_joint_position(joint)
        self.axis = _AxisMove(
            joint,
            actuator,
            start,
            float(target),
            float(now),
            0.02 if self.fast else max(0.10, float(duration_s)),
        )
        if self.fast:
            self.registry.set_batch_joint_target(joint, actuator, target, teleport=True)

    def _update_axis(self, now: float) -> tuple[bool, float]:
        move = self.axis
        if move is None:
            return True, 1.0
        linear = float(np.clip((float(now) - move.started_at) / move.duration_s, 0.0, 1.0))
        command = move.start + quintic_time_scaling(linear) * (move.target - move.start)
        self.registry.set_batch_joint_target(
            move.joint,
            move.actuator,
            command,
            teleport=self.fast,
        )
        position = self.registry.batch_joint_position(move.joint)
        velocity = abs(self.registry.batch_joint_velocity(move.joint))
        at_rest = (
            abs(position - move.target) <= self.POSITION_TOLERANCE_M
            and velocity <= self.VELOCITY_TOLERANCE_M_S
        )
        if self.fast:
            at_rest = True
        if linear < 1.0 or not at_rest:
            move.settled_at = None
            return False, linear
        if move.settled_at is None:
            move.settled_at = float(now)
            return self.fast, linear
        return float(now) - move.settled_at >= self.SETTLE_S, linear

    def _finish_axis(self) -> None:
        if self.axis is not None:
            self.registry.set_batch_joint_target(
                self.axis.joint,
                self.axis.actuator,
                self.axis.target,
                teleport=self.fast,
            )
        self.axis = None

    def _station_weld(self) -> str:
        return f"station_rack_infeed_tray_{self.unit_index + 1:02d}_weld"

    def _carrier_weld(self) -> str:
        return f"batch_carrier_tray_{self.unit_index + 1:02d}_weld"

    def _rack_weld(self) -> str:
        return f"batch_rack_tray_{self.unit_index + 1:02d}_" f"shelf_{self.layer_index}_weld"

    def _output_weld(self) -> str:
        return f"batch_output_tray_{self.unit_index + 1:02d}_weld"

    def _acquire_carrier_at_infeed(self) -> None:
        self.registry.set_batch_weld(self._station_weld(), False)
        self.registry.set_batch_weld(
            self._carrier_weld(),
            True,
            recompute=("batch_output_carriage", self.tray_name),
        )

    def _park_on_rack(self) -> None:
        self.registry.set_batch_tray_visible(self.unit_index, carrier=False, payload=False)
        self.registry.set_batch_weld(self._carrier_weld(), False)
        shelf = self.registry.site_pose(f"batch_rack_shelf_site_{self.layer_index}")
        self.registry.set_free_body_pose(self.tray_name, shelf, forward=True)
        self.registry.set_batch_weld(
            self._rack_weld(),
            True,
            recompute=(f"batch_rack_shelf_{self.layer_index}", self.tray_name),
        )
        self.registry.set_batch_tray_visible(self.unit_index, carrier=True, payload=True)

    def _acquire_from_rack(self) -> None:
        self.registry.set_batch_tray_visible(self.unit_index, carrier=False, payload=False)
        self.registry.set_batch_weld(self._rack_weld(), False)
        carrier = self.registry.site_pose("batch_transfer_pose")
        self.registry.set_free_body_pose(self.tray_name, carrier, forward=True)
        self.registry.set_batch_weld(
            self._carrier_weld(),
            True,
            recompute=("batch_output_carriage", self.tray_name),
        )
        self.registry.set_batch_tray_visible(self.unit_index, carrier=True, payload=True)

    def _park_for_inspection(self) -> None:
        self.registry.set_batch_weld(self._carrier_weld(), False)
        self.registry.set_batch_weld(
            self._output_weld(),
            True,
            recompute=(f"batch_output_slot_{self.unit_index + 1:02d}", self.tray_name),
        )

    def _acquire_for_delivery(self) -> None:
        self.registry.set_batch_weld(self._output_weld(), False)
        self.registry.set_batch_weld(
            self._carrier_weld(),
            True,
            recompute=("batch_output_carriage", self.tray_name),
        )

    def start(
        self,
        task: ManufacturingTask,
        resource_id: str,
        context: Any,
        now: float,
    ) -> None:
        del resource_id
        if self.task is not None:
            raise RuntimeError(f"{self.task_type.value}物流技能仍在运行")
        self.task = task
        self.context = context
        self.started_at = float(now)
        self.phase_started_at = float(now)
        self.cancelled = False
        self.payload_removed = False
        self.axis = None

        if self.task_type is TaskType.MOVE_ELEVATOR:
            # The historical task name is retained for API compatibility.  In
            # the simplified layout it means: open the furnace and hand the
            # rack-infeed pallet to the straight black-belt carrier.
            self.phase = "OPEN_FURNACE"
        elif self.task_type is TaskType.LOAD_RACK_LAYER:
            self.phase = "CONVEY_IN"
            self._begin_axis(
                "batch_outfeed_joint",
                "batch_outfeed_actuator",
                self.OUTFEED_INSIDE_M,
                now,
                task.estimated_duration,
            )
        elif self.task_type is TaskType.LOCK_RACK_LAYER:
            self.phase = "LOCK_AND_HOME"
            self.registry.set_batch_rack_lock(self.layer_index, True, teleport=self.fast)
            self._begin_axis(
                "batch_outfeed_joint",
                "batch_outfeed_actuator",
                0.0,
                now,
                task.estimated_duration,
            )
        elif self.task_type is TaskType.RUN_FURNACE:
            self.phase = "THERMAL_CYCLE"
        elif self.task_type is TaskType.UNLOAD_RACK_LAYER:
            self.phase = "CARRIER_ENTER"
            self.registry.set_batch_rack_lock(self.layer_index, False, teleport=self.fast)
            self._begin_axis(
                "batch_outfeed_joint",
                "batch_outfeed_actuator",
                self.OUTFEED_INSIDE_M,
                now,
                0.25 * task.estimated_duration,
            )
        elif self.task_type in {TaskType.ROUTE_PASS, TaskType.ROUTE_REWORK, TaskType.ROUTE_SCRAP}:
            self.phase = "GATE_OPEN"
            self._begin_axis(
                "finished_output_gate_joint",
                "finished_output_gate_actuator",
                self.GATE_OPEN_M,
                now,
                1.0,
            )
        else:
            raise RuntimeError(f"不支持的可见物流任务：{self.task_type.value}")

    def _update_open_furnace(self, now: float) -> SkillExecutionResult:
        duration = 0.02 if self.fast else self.DOOR_SECONDS
        progress = float(np.clip((now - self.phase_started_at) / duration, 0.0, 1.0))
        self.registry.set_furnace_door(progress, teleport=self.fast)
        measured = self.registry.batch_joint_position("furnace_door_joint") / 0.64
        velocity = abs(self.registry.batch_joint_velocity("furnace_door_joint"))
        fully_open = measured >= 0.98 and velocity <= self.VELOCITY_TOLERANCE_M_S
        if self.fast:
            fully_open = True
        if progress < 1.0 or not fully_open:
            return SkillExecutionResult.running_result(
                {
                    "progress": min(0.99, 0.9 * progress + 0.09 * measured),
                    "phase": self.phase,
                }
            )
        self.registry.set_furnace_door(1.0, teleport=self.fast)
        self._acquire_carrier_at_infeed()
        return SkillExecutionResult.success({"progress": 1.0, "physical_handoff": "rack_infeed_to_belt"})

    def _update_load(self, now: float) -> SkillExecutionResult:
        complete, progress = self._update_axis(now)
        if not complete:
            return SkillExecutionResult.running_result({"progress": 0.9 * progress, "phase": self.phase})
        self._finish_axis()
        self._park_on_rack()
        return SkillExecutionResult.success({"progress": 1.0, "rack_layer": self.layer_index})

    def _update_lock(self, now: float) -> SkillExecutionResult:
        complete, progress = self._update_axis(now)
        self.registry.set_batch_rack_lock(self.layer_index, True, teleport=self.fast)
        if not complete:
            return SkillExecutionResult.running_result({"progress": progress, "phase": self.phase})
        self._finish_axis()
        return SkillExecutionResult.success({"progress": 1.0, "rack_locked": True})

    def _update_furnace(self, now: float) -> SkillExecutionResult:
        assert self.task is not None
        duration = 0.10 if self.fast else max(2.0 * self.DOOR_SECONDS, self.task.estimated_duration)
        elapsed = max(0.0, now - self.started_at)
        close_end = self.DOOR_SECONDS
        open_start = max(close_end, duration - self.DOOR_SECONDS)
        if elapsed < close_end:
            door = 1.0 - elapsed / max(close_end, 1.0e-9)
        elif elapsed < open_start:
            door = 0.0
        else:
            door = (elapsed - open_start) / max(duration - open_start, 1.0e-9)
        self.registry.set_furnace_door(float(np.clip(door, 0.0, 1.0)), teleport=self.fast)
        fraction = float(np.clip(elapsed / duration, 0.0, 1.0))
        # A compact demonstration temperature is published for the UI while
        # the MuJoCo scene continues to represent only mechanical transport.
        thermal = 25.0 + 575.0 * min(1.0, fraction / 0.45)
        if fraction > 0.70:
            thermal = 600.0 - 520.0 * ((fraction - 0.70) / 0.30)
        self.context.v2_furnace_visual = {
            "phase": "BRAZING" if fraction < 0.70 else "COOLING",
            "temperature_c": max(80.0 if fraction >= 1.0 else 25.0, thermal),
            "progress": fraction,
        }
        if fraction < 1.0:
            return SkillExecutionResult.running_result(
                {"progress": fraction, "phase": self.phase, **self.context.v2_furnace_visual}
            )
        self.registry.set_furnace_door(1.0, teleport=self.fast)
        measured = self.registry.batch_joint_position("furnace_door_joint") / 0.64
        velocity = abs(self.registry.batch_joint_velocity("furnace_door_joint"))
        fully_open = measured >= 0.98 and velocity <= self.VELOCITY_TOLERANCE_M_S
        if not self.fast and not fully_open:
            return SkillExecutionResult.running_result(
                {
                    "progress": 0.999,
                    "phase": "OPEN_FOR_UNLOAD",
                    "temperature_c": 80.0,
                }
            )
        return SkillExecutionResult.success({"progress": 1.0, "temperature_c": 80.0, "door_open": True})

    def _update_unload(self, now: float) -> SkillExecutionResult:
        assert self.task is not None
        complete, local = self._update_axis(now)
        if not complete:
            phase_base = {"CARRIER_ENTER": 0.0, "CONVEY_OUT": 0.25, "OUTPUT_INDEX": 0.75}[self.phase]
            phase_span = {"CARRIER_ENTER": 0.25, "CONVEY_OUT": 0.50, "OUTPUT_INDEX": 0.25}[self.phase]
            return SkillExecutionResult.running_result(
                {"progress": phase_base + phase_span * local, "phase": self.phase}
            )
        self._finish_axis()
        if self.phase == "CARRIER_ENTER":
            self._acquire_from_rack()
            self.phase = "CONVEY_OUT"
            self._begin_axis(
                "batch_outfeed_joint",
                "batch_outfeed_actuator",
                0.0,
                now,
                0.50 * self.task.estimated_duration,
            )
            return SkillExecutionResult.running_result({"progress": 0.25, "phase": self.phase})
        if self.phase == "CONVEY_OUT":
            self.phase = "OUTPUT_INDEX"
            self._begin_axis(
                "batch_output_joint",
                "batch_output_actuator",
                self.OUTPUT_INSPECTION_M,
                now,
                0.25 * self.task.estimated_duration,
            )
            return SkillExecutionResult.running_result({"progress": 0.75, "phase": self.phase})
        self._park_for_inspection()
        return SkillExecutionResult.success({"progress": 1.0, "inspection_position": True})

    def _update_delivery(self, now: float) -> SkillExecutionResult:
        complete, local = self._update_axis(now)
        phase_progress = {
            "GATE_OPEN": (0.00, 0.15),
            "ENTER_OUTPUT": (0.15, 0.35),
            "RETURN_TRAY": (0.55, 0.30),
            "GATE_CLOSE": (0.85, 0.15),
        }
        if not complete:
            base, span = phase_progress[self.phase]
            return SkillExecutionResult.running_result({"progress": base + span * local, "phase": self.phase})
        self._finish_axis()
        if self.phase == "GATE_OPEN":
            self._acquire_for_delivery()
            self.phase = "ENTER_OUTPUT"
            self._begin_axis(
                "batch_output_joint",
                "batch_output_actuator",
                self.OUTPUT_DELIVERY_M,
                now,
                2.0,
            )
            return SkillExecutionResult.running_result({"progress": 0.15, "phase": self.phase})
        if self.phase == "ENTER_OUTPUT":
            self.registry.handoff_batch_payload(self.unit_index)
            self.payload_removed = True
            self.phase = "RETURN_TRAY"
            self._begin_axis(
                "batch_output_joint",
                "batch_output_actuator",
                0.0,
                now,
                2.0,
            )
            return SkillExecutionResult.running_result({"progress": 0.55, "phase": self.phase})
        if self.phase == "RETURN_TRAY":
            self.phase = "GATE_CLOSE"
            self._begin_axis(
                "finished_output_gate_joint",
                "finished_output_gate_actuator",
                0.0,
                now,
                1.0,
            )
            return SkillExecutionResult.running_result({"progress": 0.85, "phase": self.phase})
        self.registry.retire_batch_tray(self.unit_index)
        return SkillExecutionResult.success(
            {"progress": 1.0, "delivered": True, "payload_removed": self.payload_removed}
        )

    def update(self, now: float, dt: float) -> SkillExecutionResult:
        del dt
        if self.task is None:
            return SkillExecutionResult.success()
        if self.cancelled:
            self.task = None
            return SkillExecutionResult.failure("TASK_CANCELLED")
        try:
            if self.task_type is TaskType.MOVE_ELEVATOR:
                result = self._update_open_furnace(now)
            elif self.task_type is TaskType.LOAD_RACK_LAYER:
                result = self._update_load(now)
            elif self.task_type is TaskType.LOCK_RACK_LAYER:
                result = self._update_lock(now)
            elif self.task_type is TaskType.RUN_FURNACE:
                result = self._update_furnace(now)
            elif self.task_type is TaskType.UNLOAD_RACK_LAYER:
                result = self._update_unload(now)
            else:
                result = self._update_delivery(now)
        except Exception as exc:
            self.task = None
            self.axis = None
            return SkillExecutionResult.failure("PHYSICAL_LOGISTICS_FAILED", {"error": str(exc)})
        if result.succeeded or result.failed:
            self.task = None
            self.axis = None
        return result

    def cancel(self, task_id: str) -> None:
        if self.task is None or self.task.task_id != task_id:
            return
        if self.axis is not None:
            measured = self.registry.batch_joint_position(self.axis.joint)
            self.registry.set_batch_joint_target(self.axis.joint, self.axis.actuator, measured)
        self.cancelled = True
        self.task = None
        self.axis = None


LOGISTICS_TASK_TYPES = frozenset(
    {
        TaskType.MOVE_ELEVATOR,
        TaskType.LOAD_RACK_LAYER,
        TaskType.LOCK_RACK_LAYER,
        TaskType.RUN_FURNACE,
        TaskType.UNLOAD_RACK_LAYER,
        TaskType.ROUTE_PASS,
        TaskType.ROUTE_REWORK,
        TaskType.ROUTE_SCRAP,
    }
)


__all__ = ["AsyncLineLogisticsSkill", "LOGISTICS_TASK_TYPES"]
