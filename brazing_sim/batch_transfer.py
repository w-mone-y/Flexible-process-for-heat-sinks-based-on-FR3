"""Actuator-driven three-layer furnace-rack transfer cell."""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees, dist
from typing import Callable

from .domain import (
    BatchState,
    RackShelfState,
    TaskStatus,
    TransferPhase,
    TrayUnitPhase,
)


@dataclass(slots=True)
class _AxisMotion:
    joint: str
    actuator: str
    start: float
    target: float
    started_at: float
    duration: float
    settled_at: float | None = None


class BatchTransferActor:
    """Move one rigid tray through index, lift, rack and output operations.

    A nested four-axis carrier performs every loaded motion. Ownership changes
    only when the carrier and the receiving station are already coincident, so
    enabling a new weld does not introduce a visible pose discontinuity.
    """

    POSITION_TOLERANCE_M = 0.0015
    VELOCITY_TOLERANCE_M_S = 0.02
    SETTLE_SECONDS = 0.12
    OUTFEED_TARGET_M = 0.250
    OUTPUT_BUFFER_OUTFEED_M = 0.100
    PUSHER_TARGET_M = 0.380
    ALIGNMENT_DWELL_SECONDS = 0.55
    LOCK_DWELL_SECONDS = 0.75
    UNLOCK_DWELL_SECONDS = 0.45
    AXES = {
        "outfeed": ("batch_outfeed_joint", "batch_outfeed_actuator", 0.16),
        "lift": ("batch_lift_joint", "batch_lift_actuator", 0.12),
        "pusher": ("batch_pusher_joint", "batch_pusher_actuator", 0.12),
        "output": ("batch_output_joint", "batch_output_actuator", 0.16),
        "index_02": ("batch_tray_02_index_joint", "batch_tray_02_index_actuator", 0.14),
        "index_03": ("batch_tray_03_index_joint", "batch_tray_03_index_actuator", 0.14),
    }
    OUTPUT_TARGETS_M = (0.08, 0.56, 1.04)

    def __init__(
        self,
        scene: object,
        batch: Callable[[], BatchState | None],
        *,
        fast: bool = False,
    ) -> None:
        self.scene = scene
        self._batch_source = batch
        self.fast = bool(fast)
        self.phase = TransferPhase.IDLE
        self.error = ""
        self.operation = ""
        self.unit_index: int | None = None
        self._step = ""
        self._motion: _AxisMotion | None = None
        self._parallel_motions: list[_AxisMotion] = []
        self._index_motion: _AxisMotion | None = None
        self._index_unit: int | None = None
        self._prefetched_index: int | None = None
        self._paused = False
        self._resume_target: tuple[str, float] | None = None
        self._hold_until: float | None = None
        self._paused_hold_remaining = 0.0

    @staticmethod
    def _smoothstep(value: float) -> float:
        fraction = min(1.0, max(0.0, float(value)))
        return fraction * fraction * (3.0 - 2.0 * fraction)

    def _batch(self) -> BatchState:
        value = self._batch_source()
        if value is None:
            raise RuntimeError("batch transfer requires an active batch")
        return value

    @staticmethod
    def _unit_number(index: int) -> int:
        if index not in {0, 1, 2}:
            raise ValueError("batch unit index must be 0, 1 or 2")
        return index + 1

    def _source_weld(self, index: int) -> str:
        unit = self._unit_number(index)
        return "batch_station_tray_01_weld" if index == 0 else f"batch_indexer_tray_{unit:02d}_weld"

    def _carrier_weld(self, index: int) -> str:
        return f"batch_carrier_tray_{self._unit_number(index):02d}_weld"

    def _rack_weld(self, unit_index: int, shelf_index: int) -> str:
        return f"batch_rack_tray_{self._unit_number(unit_index):02d}_" f"shelf_{int(shelf_index)}_weld"

    def _output_weld(self, index: int) -> str:
        return f"batch_output_tray_{self._unit_number(index):02d}_weld"

    def _tray_name(self, index: int) -> str:
        return f"batch_tray_{self._unit_number(index):02d}"

    def _begin_axis(self, key: str, target: float, now: float) -> None:
        joint, actuator, speed = self.AXES[key]
        start = self.scene.registry.batch_joint_position(joint)
        distance = abs(float(target) - start)
        duration = max(0.12, 1.5 * distance / speed)
        self._motion = _AxisMotion(joint, actuator, start, float(target), float(now), duration)
        if self.fast:
            self.scene.registry.set_batch_joint_target(
                joint,
                actuator,
                target,
                teleport=True,
            )

    def _begin_parallel_axes(self, targets: tuple[tuple[str, float], ...], now: float) -> None:
        """Return independent empty-carrier axes at the same time."""

        self._motion = None
        self._parallel_motions = []
        for key, target in targets:
            joint, actuator, speed = self.AXES[key]
            start = self.scene.registry.batch_joint_position(joint)
            distance = abs(float(target) - start)
            duration = max(0.12, 1.5 * distance / speed)
            motion = _AxisMotion(joint, actuator, start, float(target), float(now), duration)
            self._parallel_motions.append(motion)
            if self.fast:
                self.scene.registry.set_batch_joint_target(
                    joint,
                    actuator,
                    target,
                    teleport=True,
                )

    def _begin_hold(
        self,
        step: str,
        phase: TransferPhase,
        now: float,
        duration: float,
    ) -> None:
        """Expose alignment/locking as deliberate, readable process steps."""

        self._motion = None
        self._parallel_motions = []
        self._step = step
        self.phase = phase
        self._hold_until = float(now) + (0.0 if self.fast else max(0.0, float(duration)))

    def _motion_complete(self, motion: _AxisMotion, now: float) -> bool:
        if not self.fast:
            elapsed = max(0.0, float(now) - motion.started_at)
            progress = self._smoothstep(elapsed / motion.duration)
            command = motion.start + progress * (motion.target - motion.start)
            self.scene.registry.set_batch_joint_target(motion.joint, motion.actuator, command)
        position = self.scene.registry.batch_joint_position(motion.joint)
        velocity = abs(self.scene.registry.batch_joint_velocity(motion.joint))
        at_target = (
            abs(position - motion.target) <= self.POSITION_TOLERANCE_M
            and velocity <= self.VELOCITY_TOLERANCE_M_S
        )
        if self.fast:
            at_target = True
        if not at_target:
            motion.settled_at = None
            return False
        if motion.settled_at is None:
            motion.settled_at = float(now)
            return self.fast
        return float(now) - motion.settled_at >= self.SETTLE_SECONDS

    def _axis_complete(self, now: float) -> bool:
        motion = self._motion
        if motion is None:
            return True
        return self._motion_complete(motion, now)

    def _parallel_complete(self, now: float) -> bool:
        results = [self._motion_complete(motion, now) for motion in self._parallel_motions]
        return bool(results) and all(results)

    def _finish_axis(self) -> None:
        if self._motion is not None:
            self.scene.registry.set_batch_joint_target(
                self._motion.joint,
                self._motion.actuator,
                self._motion.target,
                teleport=self.fast,
            )
        self._motion = None

    def _finish_parallel_axes(self) -> None:
        for motion in self._parallel_motions:
            self.scene.registry.set_batch_joint_target(
                motion.joint,
                motion.actuator,
                motion.target,
                teleport=self.fast,
            )
        self._parallel_motions = []

    def start_index(self, unit_index: int, now: float) -> None:
        if self.busy or self._prefetched_index is not None:
            raise RuntimeError("batch transfer actor is already busy")
        if unit_index not in {1, 2}:
            raise ValueError("only batch trays 02 and 03 require indexing")
        self.operation = "index"
        self.unit_index = unit_index
        self.phase = TransferPhase.INDEXING
        self._step = "index"
        self.error = ""
        self._paused = False
        key = f"index_{unit_index + 1:02d}"
        joint, _actuator, _speed = self.AXES[key]
        joint_id = int(self.scene.model.joint(joint).id)
        target = float(self.scene.model.jnt_range[joint_id, 1])
        self._begin_axis(key, target, now)

    def start_prefetch_index(self, unit_index: int, now: float) -> None:
        """Index the next empty tray while the preceding tray enters the rack."""

        if self.operation != "load":
            raise RuntimeError("tray prefetch requires an active rack-load operation")
        if self._index_motion is not None or self._prefetched_index is not None:
            raise RuntimeError("an empty tray is already being prefetched")
        if unit_index not in {1, 2}:
            raise ValueError("only batch trays 02 and 03 require prefetching")
        key = f"index_{unit_index + 1:02d}"
        joint, actuator, speed = self.AXES[key]
        joint_id = int(self.scene.model.joint(joint).id)
        target = float(self.scene.model.jnt_range[joint_id, 1])
        start = self.scene.registry.batch_joint_position(joint)
        duration = max(0.12, 1.5 * abs(target - start) / speed)
        self._index_motion = _AxisMotion(joint, actuator, start, target, float(now), duration)
        self._index_unit = unit_index
        if self.fast:
            self.scene.registry.set_batch_joint_target(joint, actuator, target, teleport=True)

    def _poll_prefetch(self, now: float) -> None:
        motion = self._index_motion
        if motion is None:
            return
        if not self._motion_complete(motion, now):
            return
        self.scene.registry.set_batch_joint_target(
            motion.joint,
            motion.actuator,
            motion.target,
            teleport=self.fast,
        )
        self._prefetched_index = self._index_unit
        self._index_motion = None
        self._index_unit = None

    def prefetch_active_for(self, unit_index: int) -> bool:
        return self._index_unit == unit_index or self._prefetched_index == unit_index

    def prefetch_complete_for(self, unit_index: int) -> bool:
        return self._prefetched_index == unit_index

    def consume_prefetch(self, unit_index: int) -> None:
        if self._prefetched_index != unit_index:
            raise RuntimeError(f"tray {unit_index + 1} has not completed prefetch")
        self._prefetched_index = None

    def start_load(self, unit_index: int, now: float) -> None:
        if self.operation or self._index_motion is not None:
            raise RuntimeError("batch transfer actor is already busy")
        batch = self._batch()
        unit = batch.units[unit_index]
        shelf_index = unit.layer_index
        shelf = batch.rack.shelves[shelf_index]
        if unit.phase is not TrayUnitPhase.READY_FOR_TRANSFER:
            raise RuntimeError("tray unit must be ready before rack loading")
        if shelf.state is not RackShelfState.EMPTY or shelf.lock_engaged:
            raise RuntimeError("target furnace shelf is not empty")

        tray_name = self._tray_name(unit_index)
        station_pose = self.scene.registry.free_body_pose("assembly_tray")
        tray_pose = self.scene.registry.free_body_pose(tray_name)
        position_error = dist(tray_pose.position, station_pose.position)
        quaternion_alignment = min(
            1.0,
            abs(
                sum(
                    float(left) * float(right)
                    for left, right in zip(
                        tray_pose.quaternion,
                        station_pose.quaternion,
                    )
                )
            ),
        )
        angle_error = degrees(2.0 * acos(quaternion_alignment))
        if not self.fast and (position_error > 0.003 or angle_error > 3.0):
            raise RuntimeError(
                "tray handoff is not aligned with Table2: "
                f"{position_error * 1000.0:.2f} mm / {angle_error:.2f} deg"
            )
        self.scene.registry.set_batch_weld(self._source_weld(unit_index), False)
        self.scene.registry.configure_batch_tray(unit_index, unit.product)
        self.scene.registry.set_batch_tray_visible(unit_index, carrier=True, payload=True)
        self.scene.registry.set_batch_weld(
            self._carrier_weld(unit_index),
            True,
            recompute=("batch_pusher_carriage", tray_name),
        )
        self.scene.registry.set_workcell_visible(False)

        unit.phase = TrayUnitPhase.TRANSFERRING
        shelf.state = RackShelfState.LOADING
        shelf.unit_id = unit.unit_id
        batch.transfer.unit_id = unit.unit_id
        batch.transfer.shelf_index = shelf_index
        batch.transfer.moving = True
        self.operation = "load"
        self.unit_index = unit_index
        self.phase = TransferPhase.OUTFEED
        self._step = "load_outfeed"
        self.error = ""
        self._paused = False
        self._begin_axis("outfeed", self.OUTFEED_TARGET_M, now)

    def start_unload(self, unit_index: int, now: float) -> None:
        if self.busy or self._prefetched_index is not None:
            raise RuntimeError("batch transfer actor is already busy")
        batch = self._batch()
        unit = batch.units[unit_index]
        shelf_index = unit.layer_index
        shelf = batch.rack.shelves[shelf_index]
        if unit.phase is not TrayUnitPhase.BRAZED:
            raise RuntimeError("only brazed tray units may be unloaded")
        if shelf.state is not RackShelfState.BRAZED or not shelf.lock_engaged:
            raise RuntimeError("rack shelf must hold a brazed locked tray")
        unit.phase = TrayUnitPhase.UNLOADING
        shelf.state = RackShelfState.UNLOADING
        batch.transfer.unit_id = unit.unit_id
        batch.transfer.shelf_index = shelf_index
        batch.transfer.moving = True
        self.operation = "unload"
        self.unit_index = unit_index
        self.phase = TransferPhase.OUTFEED
        self._step = "unload_outfeed"
        self.error = ""
        self._paused = False
        self._begin_axis("outfeed", self.OUTFEED_TARGET_M, now)

    def _complete(self) -> None:
        batch = self._batch()
        batch.transfer.phase = TransferPhase.COMPLETE
        batch.transfer.step = ""
        batch.transfer.moving = False
        batch.transfer.pusher_extension_ratio = 0.0
        self.phase = TransferPhase.COMPLETE
        self.operation = ""
        self.unit_index = None
        self._step = ""
        self._motion = None
        self._parallel_motions = []
        self._hold_until = None
        self._paused_hold_remaining = 0.0

    def _advance_index(self) -> None:
        assert self.unit_index is not None
        self._complete()

    def _advance_load(self, now: float) -> None:
        assert self.unit_index is not None
        batch = self._batch()
        index = self.unit_index
        unit = batch.units[index]
        shelf_index = unit.layer_index
        shelf = batch.rack.shelves[shelf_index]
        height = shelf.height_m
        if self._step == "load_outfeed":
            self.phase = TransferPhase.LIFTING
            self._step = "load_lift"
            self._begin_axis("lift", height, now)
        elif self._step == "load_lift":
            self._begin_hold(
                "load_align",
                TransferPhase.ALIGNING,
                now,
                self.ALIGNMENT_DWELL_SECONDS,
            )
        elif self._step == "load_align":
            self.phase = TransferPhase.PUSHING
            self._step = "load_push"
            self._begin_axis("pusher", self.PUSHER_TARGET_M, now)
        elif self._step == "load_push":
            tray = self._tray_name(index)
            if self.fast:
                pose = self.scene.registry.site_pose("batch_transfer_pose")
                self.scene.registry.set_free_body_pose(tray, pose, forward=True)
            self.scene.registry.set_batch_weld(self._carrier_weld(index), False)
            self.scene.registry.set_batch_weld(
                self._rack_weld(index, shelf_index),
                True,
                recompute=(f"batch_rack_shelf_{shelf_index}", tray),
            )
            if shelf_index == index:
                self.scene.registry.set_batch_weld(
                    f"batch_rack_tray_{index + 1:02d}_weld",
                    True,
                    recompute=(f"batch_rack_shelf_{shelf_index}", tray),
                )
            self.scene.registry.set_batch_rack_lock(shelf_index, True, teleport=self.fast)
            self._begin_hold(
                "load_lock",
                TransferPhase.LOCKING,
                now,
                self.LOCK_DWELL_SECONDS,
            )
        elif self._step == "load_lock":
            lock_position = self.scene.registry.batch_joint_position(f"batch_rack_lock_joint_{shelf_index}")
            if not self.fast and lock_position < 0.023:
                self._hold_until = float(now) + 0.05
                return
            shelf.state = RackShelfState.LOCKED
            shelf.lock_engaged = True
            unit.phase = TrayUnitPhase.LOCKED
            unit.loaded_at = float(now)
            self.phase = TransferPhase.RETRACTING
            self._step = "load_retract"
            self._begin_axis("pusher", 0.0, now)
        elif self._step == "load_retract":
            self.phase = TransferPhase.LOWERING
            self._step = "load_parallel_return"
            self._begin_parallel_axes((("lift", 0.0), ("outfeed", 0.0)), now)
        elif self._step == "load_parallel_return":
            self._complete()
        else:
            raise RuntimeError(f"unknown rack-load step: {self._step}")

    def _advance_unload(self, now: float) -> None:
        assert self.unit_index is not None
        batch = self._batch()
        index = self.unit_index
        unit = batch.units[index]
        shelf_index = unit.layer_index
        shelf = batch.rack.shelves[shelf_index]
        height = shelf.height_m
        if self._step == "unload_outfeed":
            self.phase = TransferPhase.LIFTING
            self._step = "unload_lift"
            self._begin_axis("lift", height, now)
        elif self._step == "unload_lift":
            self._begin_hold(
                "unload_align",
                TransferPhase.ALIGNING,
                now,
                self.ALIGNMENT_DWELL_SECONDS,
            )
        elif self._step == "unload_align":
            self.phase = TransferPhase.PUSHING
            self._step = "unload_push"
            self._begin_axis("pusher", self.PUSHER_TARGET_M, now)
        elif self._step == "unload_push":
            tray = self._tray_name(index)
            self.scene.registry.set_batch_weld(self._rack_weld(index, shelf_index), False)
            if shelf_index == index:
                self.scene.registry.set_batch_weld(
                    f"batch_rack_tray_{index + 1:02d}_weld",
                    False,
                )
            self.scene.registry.set_batch_rack_lock(shelf_index, False, teleport=self.fast)
            self.scene.registry.set_batch_weld(
                self._carrier_weld(index),
                True,
                recompute=("batch_pusher_carriage", tray),
            )
            shelf.lock_engaged = False
            self._begin_hold(
                "unload_unlock",
                TransferPhase.ALIGNING,
                now,
                self.UNLOCK_DWELL_SECONDS,
            )
        elif self._step == "unload_unlock":
            lock_position = self.scene.registry.batch_joint_position(f"batch_rack_lock_joint_{index}")
            if not self.fast and lock_position > 0.002:
                self._hold_until = float(now) + 0.05
                return
            self.phase = TransferPhase.RETRACTING
            self._step = "unload_retract"
            self._begin_axis("pusher", 0.0, now)
        elif self._step == "unload_retract":
            self.phase = TransferPhase.LOWERING
            self._step = "unload_lower"
            self._begin_axis("lift", 0.0, now)
        elif self._step == "unload_lower":
            self.phase = TransferPhase.RETRACTING
            self._step = "unload_buffer_position"
            self._begin_axis("outfeed", self.OUTPUT_BUFFER_OUTFEED_M, now)
        elif self._step == "unload_buffer_position":
            self.phase = TransferPhase.OUTPUT
            self._step = "unload_output"
            self._begin_axis("output", self.OUTPUT_TARGETS_M[index], now)
        elif self._step == "unload_output":
            tray = self._tray_name(index)
            if self.fast:
                pose = self.scene.registry.site_pose(f"batch_output_slot_{index + 1:02d}_site")
                self.scene.registry.set_free_body_pose(tray, pose, forward=True)
            self.scene.registry.set_batch_weld(self._carrier_weld(index), False)
            self.scene.registry.set_batch_weld(
                self._output_weld(index),
                True,
                recompute=(f"batch_output_slot_{index + 1:02d}", tray),
            )
            shelf.state = RackShelfState.UNLOADED
            unit.phase = TrayUnitPhase.UNLOADED
            unit.output_slot = index
            unit.unloaded_at = float(now)
            self._step = "unload_parallel_return"
            self._begin_parallel_axes((("output", 0.0), ("outfeed", 0.0)), now)
        elif self._step == "unload_parallel_return":
            self._complete()
        else:
            raise RuntimeError(f"unknown rack-unload step: {self._step}")

    def poll(self, now: float) -> TaskStatus:
        if self.error:
            return TaskStatus.FAILED
        if self._paused:
            return TaskStatus.RUNNING
        try:
            self._poll_prefetch(now)
            if not self.operation:
                return TaskStatus.RUNNING if self._index_motion is not None else TaskStatus.SUCCEEDED
            for _ in range(16 if self.fast else 1):
                holding = self._hold_until is not None
                parallel = bool(self._parallel_motions)
                complete = (
                    float(now) >= float(self._hold_until)
                    if holding
                    else self._parallel_complete(now) if parallel else self._axis_complete(now)
                )
                if not complete:
                    break
                if holding:
                    self._hold_until = None
                elif parallel:
                    self._finish_parallel_axes()
                else:
                    self._finish_axis()
                operation = self.operation
                if operation == "index":
                    self._advance_index()
                elif operation == "load":
                    self._advance_load(now)
                elif operation == "unload":
                    self._advance_unload(now)
                if not self.operation:
                    return TaskStatus.SUCCEEDED
            batch = self._batch()
            batch.transfer.phase = self.phase
            batch.transfer.step = self._step
            batch.transfer.lift_height_m = self.scene.registry.batch_joint_position("batch_lift_joint")
            batch.transfer.outfeed_position_m = self.scene.registry.batch_joint_position(
                "batch_outfeed_joint"
            )
            batch.transfer.pusher_position_m = self.scene.registry.batch_joint_position("batch_pusher_joint")
            batch.transfer.pusher_extension_ratio = max(
                0.0,
                min(1.0, batch.transfer.pusher_position_m / self.PUSHER_TARGET_M),
            )
            if self.unit_index in {0, 1, 2}:
                batch.transfer.lock_position_m = self.scene.registry.batch_joint_position(
                    f"batch_rack_lock_joint_{self.unit_index}"
                )
            batch.transfer.output_position_m = self.scene.registry.batch_joint_position("batch_output_joint")
            return TaskStatus.RUNNING
        except Exception as exc:
            self.error = str(exc)
            self.phase = TransferPhase.ERROR
            batch = self._batch_source()
            if batch is not None:
                batch.transfer.phase = TransferPhase.ERROR
                batch.transfer.error = self.error
                batch.transfer.moving = False
            return TaskStatus.FAILED

    def pause(self, now: float | None = None) -> None:
        if not self.busy or self._paused:
            return
        timestamp = float(self.scene.time if now is None else now)
        if self._hold_until is not None:
            self._paused_hold_remaining = max(0.0, self._hold_until - timestamp)
        if self._motion is not None:
            self._resume_target = (self._motion.joint, self._motion.target)
            measured = self.scene.registry.batch_joint_position(self._motion.joint)
            self.scene.registry.set_batch_joint_target(
                self._motion.joint,
                self._motion.actuator,
                measured,
            )
        for motion in self._parallel_motions:
            measured = self.scene.registry.batch_joint_position(motion.joint)
            self.scene.registry.set_batch_joint_target(
                motion.joint,
                motion.actuator,
                measured,
            )
        if self._index_motion is not None:
            measured = self.scene.registry.batch_joint_position(self._index_motion.joint)
            self.scene.registry.set_batch_joint_target(
                self._index_motion.joint,
                self._index_motion.actuator,
                measured,
            )
        self._paused = True
        self.phase = TransferPhase.PAUSED

    def resume(self, now: float) -> None:
        if not self._paused:
            return
        self._paused = False
        if self._hold_until is not None:
            self._hold_until = float(now) + self._paused_hold_remaining
            self._paused_hold_remaining = 0.0
        if self._motion is not None:
            target = self._motion.target
            key = next(key for key, value in self.AXES.items() if value[0] == self._motion.joint)
            self._begin_axis(key, target, now)
        elif self._parallel_motions:
            targets = tuple(
                (
                    next(key for key, value in self.AXES.items() if value[0] == motion.joint),
                    motion.target,
                )
                for motion in self._parallel_motions
            )
            self._begin_parallel_axes(targets, now)
        if self._index_motion is not None:
            motion = self._index_motion
            key = next(key for key, value in self.AXES.items() if value[0] == motion.joint)
            joint, actuator, speed = self.AXES[key]
            start = self.scene.registry.batch_joint_position(joint)
            self._index_motion = _AxisMotion(
                joint,
                actuator,
                start,
                motion.target,
                float(now),
                max(0.12, 1.5 * abs(motion.target - start) / speed),
            )

    def reset(self, *, show_empty_cache: bool = False) -> None:
        self.scene.registry.reset_batch_cell(show_empty_cache=show_empty_cache)
        self.phase = TransferPhase.IDLE
        self.error = ""
        self.operation = ""
        self.unit_index = None
        self._step = ""
        self._motion = None
        self._parallel_motions = []
        self._index_motion = None
        self._index_unit = None
        self._prefetched_index = None
        self._paused = False
        self._resume_target = None
        self._hold_until = None
        self._paused_hold_remaining = 0.0

    @property
    def busy(self) -> bool:
        return bool(self.operation or self._index_motion is not None)

    @property
    def state(self) -> dict[str, object]:
        lock_position = 0.0
        if self.unit_index in {0, 1, 2}:
            lock_position = self.scene.registry.batch_joint_position(
                f"batch_rack_lock_joint_{self.unit_index}"
            )
        return {
            "phase": self.phase.value,
            "step": self._step,
            "operation": self.operation,
            "unit_index": self.unit_index,
            "moving": bool(self.busy and not self._paused),
            "paused": self._paused,
            "prefetch_unit_index": self._index_unit,
            "prefetch_complete_index": self._prefetched_index,
            "parallel_axes": [motion.joint for motion in self._parallel_motions],
            "parallel_active": bool(
                not self._paused
                and (len(self._parallel_motions) > 1 or (self.operation and self._index_motion))
            ),
            "pusher_extension_ratio": max(
                0.0,
                min(
                    1.0,
                    self.scene.registry.batch_joint_position("batch_pusher_joint") / self.PUSHER_TARGET_M,
                ),
            ),
            "lock_position_m": lock_position,
            "error": self.error,
        }


__all__ = ["BatchTransferActor"]
