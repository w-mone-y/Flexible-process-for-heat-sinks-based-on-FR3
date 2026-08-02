"""Runtime PRM/SIPP planning service for the shallow-U shared workstations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .motion_planner import (
    HybridMotionPlanner,
    JointPath,
    MotionRequest,
    SpaceTimeReservationTable,
    always_collision_free,
)
from .task_models import ManufacturingTask

FR3_LIMITS = (
    (-2.8973, 2.8973),
    (-1.7628, 1.7628),
    (-2.8973, 2.8973),
    (-3.0718, -0.0698),
    (-2.8973, 2.8973),
    (-0.0175, 3.7525),
    (-2.8973, 2.8973),
)
HOME = np.asarray((0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785), dtype=float)


@dataclass(slots=True)
class MotionPlanningDecision:
    task_id: str
    path: JointPath | None
    blocker: str | None = None
    blocked_by: str | None = None

    @property
    def start_time(self) -> float:
        return 0.0 if self.path is None else self.path.start_time


class WorkcellMotionPlanningService:
    """Own deterministic per-arm roadmaps and one cross-arm interval table."""

    ROBOTS = ("ARM1", "ARM2", "ARM3")

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = int(seed)
        self.planners = {
            resource: HybridMotionPlanner(
                FR3_LIMITS,
                always_collision_free,
                seed=self.seed + index,
                roadmap_samples=56,
                neighbours=8,
            )
            for index, resource in enumerate(self.ROBOTS)
        }
        self.reservations = SpaceTimeReservationTable(safety_time_s=0.02)
        self.paths: dict[str, JointPath] = {}
        self.blockers: dict[str, dict[str, Any]] = {}
        self._logical_q = {resource: HOME.copy() for resource in self.ROBOTS}

    @staticmethod
    def _scene_q(context: Any, resource_id: str) -> np.ndarray | None:
        scene = getattr(context, "scene", None)
        controller = None
        if scene is not None:
            controller = getattr(scene, "arms", {}).get(resource_id.lower())
        if controller is None:
            # V2 binds its MuJoCo adapter through the physical execution gate,
            # rather than exposing the V1 ``scene.arms`` facade.
            gate = getattr(context, "physical_gate", None)
            projector = getattr(gate, "_robots", None)
            controller = getattr(projector, "controllers", {}).get(resource_id.lower())
        if controller is None:
            return None
        data = getattr(scene, "data", None) if scene is not None else getattr(gate, "data", None)
        if data is None:
            return None
        return np.asarray(data.qpos[controller.qpos_ids], dtype=float).copy()

    @staticmethod
    def _goal(task: ManufacturingTask, start: np.ndarray, resource_id: str) -> np.ndarray:
        goal = HOME.copy()
        station = str(task.station_id or "")
        resource = str(resource_id).upper()
        station_offsets = {
            "S1_BASE_LOADING": (-0.38, 0.10, 0.18, 0.06, -0.08, 0.08, 0.0),
            "S2A_DISPENSING": (-0.12, 0.16, -0.12, 0.08, 0.06, 0.12, 0.0),
            "S2B_MATERIAL_INSPECTION": (0.12, 0.14, -0.10, 0.07, 0.08, 0.14, 0.0),
            "S3_FIN_ASSEMBLY": (0.38, 0.10, -0.18, 0.06, 0.10, 0.10, 0.0),
            "RACK_INFEED": (0.48, 0.08, -0.16, 0.04, 0.10, 0.08, 0.0),
            "S3A_ARM1_INSTALL": (0.38, 0.10, -0.18, 0.06, 0.10, 0.10, 0.0),
            "S3B_ARM3_INSTALL": (-0.30, 0.12, 0.16, 0.08, -0.08, 0.14, 0.0),
            "S4_PRE_BRAZE_INSPECTION": (0.18, 0.14, -0.12, 0.07, 0.08, 0.14, 0.0),
        }
        if station == "S3_DUAL_INSTALL":
            station = "S3A_ARM1_INSTALL" if resource == "ARM1" else "S3B_ARM3_INSTALL"
        if station in station_offsets:
            goal += np.asarray(station_offsets[station])
        elif "ZONE_TABLE1" in task.required_zones:
            goal += np.asarray((-0.52, 0.18, 0.12, 0.05, -0.08, 0.06, 0.0))
        elif "ZONE_OUTFEED" in task.required_zones:
            goal += np.asarray((0.42, 0.12, -0.14, 0.04, 0.08, 0.10, 0.0))
        if resource == "ARM3":
            goal[5] += 0.08
        for index in task.motion_constraints.get("lock_joint_indices", ()):
            goal[int(index)] = start[int(index)]
        return np.clip(goal, [item[0] for item in FR3_LIMITS], [item[1] for item in FR3_LIMITS])

    @staticmethod
    def _occupancy(task: ManufacturingTask, resource_id: str):
        station = str(task.station_id or "")
        if station == "S3_DUAL_INSTALL":
            station = "S3A_ARM1_INSTALL" if resource_id == "ARM1" else "S3B_ARM3_INSTALL"
        station_cell = {
            "S1_BASE_LOADING": "CELL_S1",
            "S2A_DISPENSING": "CELL_S2A",
            "S2B_MATERIAL_INSPECTION": "CELL_S2B",
            "S3_FIN_ASSEMBLY": "CELL_S3",
            "RACK_INFEED": "CELL_RACK_INFEED",
            "S3A_ARM1_INSTALL": "CELL_V2_S3A",
            "S3B_ARM3_INSTALL": "CELL_V2_S3B",
            "S4_PRE_BRAZE_INSPECTION": "CELL_V2_S4_SHARED",
        }.get(station, "CELL_SAFE_CORRIDOR")
        if station == "S1_BASE_LOADING":
            station_cell = "CELL_V2_S1"
        elif station == "S2A_DISPENSING":
            station_cell = "CELL_V2_S2A"
        elif station == "S2B_MATERIAL_INSPECTION":
            station_cell = "CELL_V2_S2B"
        fixed = tuple(sorted({station_cell, *(f"ZONE::{zone}" for zone in task.required_zones)}))

        def cells(q: np.ndarray) -> tuple[str, ...]:
            shoulder_bin = int(round(float(q[0]) / 0.35))
            elbow_bin = int(round(float(q[3]) / 0.35))
            return (*fixed, f"LINK::{resource_id}::{shoulder_bin}:{elbow_bin}")

        return cells

    def prepare(
        self,
        task: ManufacturingTask,
        resource_id: str,
        now: float,
        *,
        context: Any = None,
    ) -> MotionPlanningDecision:
        resource = str(resource_id).upper()
        if resource not in self.planners:
            return MotionPlanningDecision(task.task_id, None)
        if task.task_id in self.paths:
            return MotionPlanningDecision(task.task_id, self.paths[task.task_id])
        start = self._scene_q(context, resource)
        if start is None:
            start = self._logical_q[resource].copy()
        goal = self._goal(task, start, resource)
        request = MotionRequest(
            request_id=f"MOTION-{task.task_id}",
            resource_id=resource,
            start=tuple(float(value) for value in start),
            goals=(tuple(float(value) for value in goal),),
            earliest_start=float(now),
            lock_joint_indices=tuple(
                int(value) for value in task.motion_constraints.get("lock_joint_indices", ())
            ),
            required_safe_nodes=("CURRENT_CERTIFIED_WAIT",),
            minimum_clearance_m=float(task.motion_constraints.get("minimum_clearance_m", 0.04)),
            seed=self.seed + task.sequence_index,
        )
        try:
            path = self.planners[resource].plan(request)
            reservation = self.reservations.reserve(
                path,
                self._occupancy(task, resource),
                sample_period_s=float(task.motion_constraints.get("time_sample_s", 0.02)),
                safe_wait=lambda _q: True,
            )
        except (RuntimeError, TimeoutError) as exc:
            details = {"reason": str(exc), "resource_id": resource}
            self.blockers[task.task_id] = details
            return MotionPlanningDecision(task.task_id, None, str(exc))
        task.reservation_id = reservation.reservation_id
        self.paths[task.task_id] = path
        self._logical_q[resource] = np.asarray(path.samples[-1].position, dtype=float)
        if path.waiting_time > 1e-9:
            self.blockers[task.task_id] = {
                "reason": "时空路径预约等待",
                "resource_id": resource,
                "available_at": path.start_time,
            }
            return MotionPlanningDecision(task.task_id, path, "时空路径预约等待")
        self.blockers.pop(task.task_id, None)
        return MotionPlanningDecision(task.task_id, path)

    def release_task(self, task: ManufacturingTask, *, retain_path: bool = True) -> None:
        if task.reservation_id:
            self.reservations.release(task.reservation_id)
        self.blockers.pop(task.task_id, None)
        if not retain_path:
            self.paths.pop(task.task_id, None)
            task.reservation_id = None

    def reset(self) -> None:
        self.paths.clear()
        self.blockers.clear()
        self.reservations.clear()
        self._logical_q = {resource: HOME.copy() for resource in self.ROBOTS}

    def path_snapshots(self) -> list[dict[str, object]]:
        return [self.paths[key].as_dict() for key in sorted(self.paths)]

    def reservation_snapshots(self) -> list[dict[str, object]]:
        return [item.as_dict() for item in self.reservations.reservations]


__all__ = ["MotionPlanningDecision", "WorkcellMotionPlanningService"]
