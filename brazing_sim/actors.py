"""MuJoCo task actors used by :mod:`brazing_sim.process`.

The process coordinator owns product truth.  These actors provide the visual
and kinematic side effects: generated Cartesian paths drive the FR3
controllers, while scene constraints are switched only after a task's safe
retreat completes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np

from .domain import ProductState, TaskSpec, TaskType
from .motion import (
    ExecutionState,
    PolylineTrajectory,
    Pose,
    TrajectoryExecutor,
    matrix_to_quat,
    pose_from_site,
    safe_corridor_trajectory,
)
from .process import ActorResult
from .scene import BrazingScene


def _top_down_quaternion(yaw: float = 0.0) -> np.ndarray:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray(
        [
            [cosine, sine, 0.0],
            [sine, -cosine, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )
    return matrix_to_quat(rotation)


def _forward_tilted_lance_quaternion(tilt_deg: float = 20.0) -> np.ndarray:
    """Point a long Arm2 lance downward and slightly toward Table2 (+Y)."""

    angle = math.radians(float(tilt_deg))
    sine, cosine = math.sin(angle), math.cos(angle)
    rotation = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, -cosine, sine],
            [0.0, -sine, -cosine],
        ]
    )
    return matrix_to_quat(rotation)


def _site_pose(scene: BrazingScene, name: str) -> Pose:
    return pose_from_site(scene.data, int(scene.model.site(name).id))


@dataclass(slots=True)
class Arm1Step:
    kind: str
    trajectory: PolylineTrajectory | None = None
    duration: float = 0.0
    start_value: float = 0.0
    end_value: float = 0.0
    action: Callable[[], None] | None = None
    body_name: str | None = None
    target_pose: Callable[[], Pose] | None = None
    start_pose: Pose | None = None
    end_pose: Pose | None = None
    joint_start: np.ndarray | None = None
    joint_end: np.ndarray | None = None


class SceneTaskActor:
    """One arm's parameterized visual-motion adapter.

    In ``fast`` mode callbacks are still applied, but robot travel is skipped;
    this is used by the automated headless fault matrix.  Normal headless and
    viewer runs use sampled-IK kinematic carrying for Arm1 and TCP-DLS for
    Arm2/Arm3.  Both paths are sampled at no more than 10 mm and checked for
    reachability before motion starts.
    """

    def __init__(
        self,
        arm_name: str,
        scene: BrazingScene,
        product: Callable[[], ProductState | None],
        *,
        fast: bool = False,
        speed_m_s: float = 0.12,
    ) -> None:
        if arm_name not in scene.arms:
            raise ValueError(f"unknown scene arm: {arm_name}")
        self.arm_name = arm_name
        self.scene = scene
        self.product_supplier = product
        self.fast = bool(fast)
        self.speed_m_s = float(speed_m_s)
        self.controller = scene.arms[arm_name]
        self.park_flange_pose = self.controller.current_flange_pose()
        self.park_joint_positions = np.asarray(
            self.scene.data.qpos[self.controller.qpos_ids], dtype=float
        ).copy()
        self.executor = TrajectoryExecutor(self.controller)
        self.task: TaskSpec | None = None
        self.trajectory: PolylineTrajectory | None = None
        self.started_at = 0.0
        self.deadline = 0.0
        self.error = ""
        self._done = False
        self._errors: list[float] = []
        self._followup_segments: list[tuple[PolylineTrajectory, Callable[[], None] | None]] = []
        self._kinematic_motion: tuple[np.ndarray, float, float] | None = None
        self._kinematic_targets: tuple[Pose, ...] = ()
        self._carried_body: str | None = None
        self._carry_relative: Pose | None = None
        self._carried_orientation_lock: np.ndarray | None = None
        self._carried_group: dict[str, Pose] = {}
        self._transport_active = False
        self._arm1_tool_roll_lock: float | None = None
        self._arm1_steps: list[Arm1Step] = []
        self._arm1_timed: tuple[Arm1Step, float] | None = None
        self._arm2_steps: list[Arm1Step] = []
        self._arm2_timed: tuple[Arm1Step, float] | None = None
        self._configure_tcp()
        # Idle actors hold their actual joint command. Leaving the Cartesian
        # controller enabled would let null-space centering slowly move an
        # elbow while another arm owns the work zone.
        self.controller.q_command = np.asarray(
            self.scene.data.qpos[self.controller.qpos_ids], dtype=float
        ).copy()
        self.controller.enabled = False

    def _should_park(self, task: TaskSpec) -> bool:
        """Return to canonical poses wherever redundant IK branches can accumulate."""

        kind = TaskType(task.task_type)
        if kind is TaskType.LOAD_BASE:
            return False
        if kind is TaskType.INSERT_FIN:
            # Table1 fins are distributed across the right half of the wide
            # table. Returning to one high canonical pose after every fin
            # prevents redundant-joint branches from accumulating as Arm1
            # crosses between successive Y slots.
            return True
        if kind is TaskType.APPLY_MATERIAL:
            # Return to the same high material-safe pose between seams. This
            # prevents redundant-joint branch drift over the eight-path batch.
            return True
        return True

    def _transport_milestone(self, task: TaskSpec) -> tuple[int, Callable[[], None]] | None:
        """Return the trajectory waypoint where a physical carry weld engages."""

        kind = TaskType(task.task_type)
        registry = self.scene.registry
        if kind is TaskType.LOAD_BASE:
            return 2, lambda: registry.grasp_base(True)
        if kind in {TaskType.INSERT_FIN, TaskType.ADJUST_FIN}:
            fin_id = str(task.payload["fin_id"])
            return 2, lambda: registry.grasp_fin(fin_id, True)
        if kind is TaskType.LOAD_FURNACE:
            return 2, lambda: registry.carry_tray(True)
        if kind is TaskType.UNLOAD_FURNACE:
            return 1, lambda: registry.carry_tray(True)
        return None

    def _release_transport(self) -> None:
        if not self._transport_active or self.task is None:
            return
        kind = TaskType(self.task.task_type)
        registry = self.scene.registry
        if kind is TaskType.LOAD_BASE:
            registry.grasp_base(False)
        elif kind in {TaskType.INSERT_FIN, TaskType.ADJUST_FIN}:
            registry.grasp_fin(str(self.task.payload["fin_id"]), False)
        elif kind in {TaskType.LOAD_FURNACE, TaskType.UNLOAD_FURNACE}:
            registry.carry_tray(False)
        self._transport_active = False

    def _transport_body(self) -> str | None:
        if self.task is None:
            return None
        kind = TaskType(self.task.task_type)
        if kind is TaskType.LOAD_BASE:
            return "base_plate"
        if kind in {TaskType.INSERT_FIN, TaskType.ADJUST_FIN}:
            return str(self.task.payload["fin_id"])
        return None

    def _start_kinematic(
        self,
        trajectory: PolylineTrajectory,
        now: float,
        *,
        final_joint_positions: np.ndarray | None = None,
    ) -> None:
        checks = self.controller.validate_trajectory(trajectory)
        failed = next((result for result in checks if not result.reachable), None)
        if failed is not None:
            raise RuntimeError(
                f"kinematic path is unreachable: {failed.position_error_m * 1000:.2f} mm / "
                f"{math.degrees(failed.orientation_error_rad):.2f} deg"
            )
        self.trajectory = trajectory
        joint_samples = np.stack([result.joint_positions for result in checks])
        self._kinematic_targets = trajectory.samples(min(trajectory.sample_spacing_m, 0.01))
        if len(self._kinematic_targets) != len(joint_samples):
            raise RuntimeError("kinematic target/sample count mismatch")
        # The first Cartesian sample is the end of the preceding segment.
        # Preserve the actual redundant-joint configuration instead of letting
        # IK choose another valid branch and visibly "float" the arm.
        joint_samples[0] = np.asarray(self.scene.data.qpos[self.controller.qpos_ids], dtype=float)
        if final_joint_positions is not None:
            joint_samples[-1] = np.asarray(final_joint_positions, dtype=float)
        if self.arm_name == "arm1" and self._arm1_tool_roll_lock is not None:
            # Joint 7 is the independent roll channel about the tool's body-Z
            # axis.  Once a workpiece is attached, changing this joint twists
            # the suction cup/gripper and introduces apparent payload drift.
            joint_samples[:, -1] = self._arm1_tool_roll_lock
        self._kinematic_motion = (
            joint_samples,
            float(now),
            max(0.25, trajectory.duration_s),
        )
        self.controller.enabled = False

    def _capture_carried_body(self) -> None:
        body = self._transport_body()
        if body is None:
            return
        tool_body = self.scene.arm1_tools.current_body
        if tool_body is None:
            raise RuntimeError("Arm1 cannot grasp without a mounted tool")
        gripper = self.scene.registry.free_body_pose(tool_body)
        carried = self.scene.registry.free_body_pose(body)
        self._carried_body = body
        self._carry_relative = gripper.inverse().transformed(carried)
        self._carried_orientation_lock = carried.quaternion.copy()

    def _sync_carried_body(self) -> None:
        if self._carried_group:
            tool_body = self.scene.tools.current_body
            if tool_body is None:
                raise RuntimeError("Arm2 lost its tray-transfer tool while carrying the assembly")
            tool = self.scene.registry.free_body_pose(tool_body)
            for body_name, relative in self._carried_group.items():
                self.scene.registry.set_free_body_pose(
                    body_name,
                    tool.transformed(relative),
                )
            self.scene.mujoco.mj_forward(self.scene.model, self.scene.data)
        if self._carried_body is None or self._carry_relative is None:
            return
        tool_body = self.scene.arm1_tools.current_body
        if tool_body is None:
            raise RuntimeError("Arm1 lost its tool while carrying a workpiece")
        gripper = self.scene.registry.free_body_pose(tool_body)
        carried_pose = gripper.transformed(self._carry_relative)
        if self._carried_orientation_lock is not None:
            carried_pose = Pose(carried_pose.position, self._carried_orientation_lock)
        self.scene.registry.set_free_body_pose(self._carried_body, carried_pose, forward=True)

    def _assembly_body_names(self) -> list[str]:
        product = self._product()
        return [
            "assembly_tray",
            "base_plate",
            *(fin.fin_id for fin in product.active_fins),
            *(f"brazing_path_{path.path_id}" for path in product.active_paths),
        ]

    def _capture_tray_group(self) -> None:
        tool_body = self.scene.tools.current_body
        if tool_body != "arm2_tray_transfer":
            raise RuntimeError("tray transfer requires Arm2 tray_transfer tool")
        tool = self.scene.registry.free_body_pose(tool_body)
        inverse = tool.inverse()
        self._carried_group = {
            name: inverse.transformed(self.scene.registry.free_body_pose(name))
            for name in self._assembly_body_names()
        }
        self._transport_active = True

    def _release_tray_group(self) -> None:
        self.scene.registry.carry_tray(False)
        self._carried_group.clear()
        self._transport_active = False

    def _build_arm1_steps(
        self,
        task: TaskSpec,
        trajectory: PolylineTrajectory,
        place_action: Callable[[], None],
    ) -> None:
        points = trajectory.waypoints
        if len(points) < 9:
            raise RuntimeError(f"Arm1 transport trajectory is incomplete for {task.task_id}")

        def path(start: int, stop: int, speed: float) -> PolylineTrajectory:
            return PolylineTrajectory(tuple(points[start:stop]), speed, 0.01)

        def attach(action: Callable[[], None]) -> Callable[[], None]:
            def run() -> None:
                self._arm1_tool_roll_lock = float(self.scene.data.qpos[self.controller.qpos_ids[-1]])
                # Capture the unconstrained workpiece pose first.  Enabling a
                # weld may run mj_forward immediately, so reading it afterward
                # can preserve a tiny solver-induced orientation snap.
                self._capture_carried_body()
                action()
                self._transport_active = True

            return run

        def place() -> None:
            place_action()
            self._transport_active = False
            self._carried_body = None
            self._carry_relative = None
            self._carried_orientation_lock = None
            # Keep joint 7 on the same tool-roll branch through the retreat.
            # Releasing it at the placement callback lets the following IK
            # sample choose an equivalent roll solution and produces a visible
            # wrist jump, especially near Table1's extended-reach boundary.
            # ``_advance_arm1_step`` clears the lock only after the smooth
            # joint-home step has completed.

        def release_grasp() -> None:
            if TaskType(task.task_type) is TaskType.LOAD_BASE:
                self.scene.registry.grasp_base(False)
            else:
                self.scene.registry.grasp_fin(str(task.payload["fin_id"]), False)
            self._transport_active = False
            self._carried_body = None
            self._carry_relative = None
            self._carried_orientation_lock = None
            # The payload has been released, but retain the selected wrist
            # roll branch until the retreat and joint-home sequence finishes.

        kind = TaskType(task.task_type)
        cruise_speed = max(self.speed_m_s, 0.18)
        if kind is TaskType.LOAD_BASE:
            self.scene.registry.set_arm1_gripper_closed(0.0)
            self.scene.registry.set_arm1_suction_fraction(0.0)
            self._arm1_steps = [
                Arm1Step("trajectory", trajectory=path(0, 3, cruise_speed)),
                Arm1Step("trajectory", trajectory=path(2, 4, 0.035)),
                Arm1Step(
                    "suction",
                    duration=0.60,
                    start_value=0.0,
                    end_value=1.0,
                    action=attach(lambda: self.scene.registry.grasp_base(True)),
                ),
                Arm1Step("hold", duration=0.35),
                Arm1Step("trajectory", trajectory=path(3, 7, cruise_speed)),
                Arm1Step("trajectory", trajectory=path(6, 8, 0.030)),
                Arm1Step("hold", duration=0.35),
                Arm1Step("action", action=release_grasp),
                Arm1Step(
                    "settle",
                    duration=0.60,
                    body_name="base_plate",
                    target_pose=lambda: self.scene.registry.assembly_base_pose,
                ),
                Arm1Step("action", action=place),
                Arm1Step("suction", duration=0.60, start_value=1.0, end_value=0.0),
                Arm1Step("hold", duration=0.25),
                Arm1Step("trajectory", trajectory=path(7, len(points), 0.12)),
                Arm1Step("joint_home", duration=1.25),
            ]
            return

        if kind in {TaskType.INSERT_FIN, TaskType.ADJUST_FIN}:
            self.scene.registry.set_arm1_suction_fraction(0.0)
            self.scene.registry.set_arm1_gripper_closed(0.0)
            fin_id = str(task.payload["fin_id"])
            self._arm1_steps = [
                Arm1Step("trajectory", trajectory=path(0, 3, cruise_speed)),
                Arm1Step("trajectory", trajectory=path(2, 4, 0.035)),
                Arm1Step(
                    "gripper",
                    duration=0.75,
                    start_value=0.0,
                    end_value=1.0,
                    action=attach(lambda: self.scene.registry.grasp_fin(fin_id, True)),
                ),
                Arm1Step("hold", duration=0.35),
                Arm1Step("trajectory", trajectory=path(3, 7, cruise_speed)),
                Arm1Step("trajectory", trajectory=path(6, 8, 0.025)),
                Arm1Step("hold", duration=0.35),
                Arm1Step("action", action=release_grasp),
                Arm1Step(
                    "settle",
                    duration=0.60,
                    body_name=fin_id,
                    target_pose=lambda: self.scene.registry.product_pose().transformed(
                        self.scene.registry.fin_local_targets[fin_id]
                    ),
                ),
                Arm1Step("action", action=place),
                Arm1Step("gripper", duration=0.75, start_value=1.0, end_value=0.0),
                Arm1Step("hold", duration=0.25),
                Arm1Step("trajectory", trajectory=path(7, len(points), 0.12)),
                Arm1Step("joint_home", duration=1.25),
            ]
            return

        raise RuntimeError(f"unsupported Arm1 sequence: {kind.value}")

    def _advance_arm1_step(self, now: float) -> None:
        while self._arm1_steps:
            step = self._arm1_steps.pop(0)
            if step.kind == "action":
                if step.action is not None:
                    step.action()
                continue
            if step.kind == "trajectory":
                assert step.trajectory is not None
                self._start_kinematic(step.trajectory, now)
                return
            if step.kind == "settle":
                if step.body_name is None or step.target_pose is None:
                    raise RuntimeError("Arm1 settle step is incomplete")
                step.start_pose = self.scene.registry.free_body_pose(step.body_name)
                step.end_pose = step.target_pose()
                self._arm1_timed = (step, float(now))
                return
            if step.kind == "joint_home":
                step.joint_start = np.asarray(
                    self.scene.data.qpos[self.controller.qpos_ids], dtype=float
                ).copy()
                step.joint_end = self.park_joint_positions.copy()
                self._arm1_timed = (step, float(now))
                return
            if step.kind in {"gripper", "suction", "hold"}:
                self._arm1_timed = (step, float(now))
                return
            raise RuntimeError(f"unknown Arm1 step: {step.kind}")

        self._transport_active = False
        self._carried_body = None
        self._carry_relative = None
        self._carried_orientation_lock = None
        self._arm1_tool_roll_lock = None
        self.controller.enabled = False
        self.controller.hold()
        self._done = True
        self.task = None

    def _configure_tcp(self) -> None:
        tcp_name = {
            "arm3": "arm3_inspection_tcp",
        }.get(self.arm_name)
        if tcp_name is None:
            return
        flange = pose_from_site(self.scene.data, int(self.model_site(f"{self.arm_name}_attachment_site")))
        tcp = pose_from_site(self.scene.data, int(self.model_site(tcp_name)))
        self.controller.set_tool_transform(flange.inverse().transformed(tcp))

    def model_site(self, name: str) -> int:
        return int(self.scene.model.site(name).id)

    def _product(self) -> ProductState:
        product = self.product_supplier()
        if product is None:
            raise RuntimeError("scene actor has no active product")
        return product

    def _pose(self, position: Any, yaw: float = 0.0) -> Pose:
        return Pose(np.asarray(position, dtype=float), _top_down_quaternion(yaw))

    def _path_world_points(self, path_id: str) -> tuple[np.ndarray, np.ndarray]:
        product = self._product()
        path = next(item for item in product.active_paths if item.path_id == path_id)
        start = np.asarray(self.scene.registry.product_to_world(path.local_start), dtype=float)
        end = np.asarray(self.scene.registry.product_to_world(path.local_end), dtype=float)
        return start, end

    def _move_assembly(self, target_tray_pose: Pose) -> None:
        """Teleport the constrained tray/product as one rigid visual group."""

        registry = self.scene.registry
        current_tray = registry.free_body_pose("assembly_tray")
        relative: dict[str, Pose] = {}
        product = self._product()
        names = [
            "assembly_tray",
            "base_plate",
            *(fin.fin_id for fin in product.active_fins),
            *(f"brazing_path_{path.path_id}" for path in product.active_paths),
        ]
        inverse = current_tray.inverse()
        for name in names:
            relative[name] = inverse.transformed(registry.free_body_pose(name))
        for name in names:
            registry.set_free_body_pose(name, target_tray_pose.transformed(relative[name]))
        self.scene.mujoco.mj_forward(self.scene.model, self.scene.data)

    def _task_goal(self, task: TaskSpec) -> tuple[PolylineTrajectory, Callable[[], None]]:
        kind = TaskType(task.task_type)
        current = self.controller.current_tcp_pose()
        registry = self.scene.registry

        def noop() -> None:
            return None

        if kind is TaskType.LOAD_BASE:
            work_entry = self.park_flange_pose.transformed(self.controller.tool_transform)
            half_thickness = 0.5 * self._product().spec.base_thickness
            raw = registry.free_body_pose("base_plate").position + np.asarray(
                [0.0, 0.0, half_thickness + 0.002]
            )
            raw_approach = raw + np.asarray([0.0, 0.0, 0.10])
            target = registry.assembly_base_pose.position + np.asarray([0.0, 0.0, half_thickness + 0.002])
            target_approach = target + np.asarray([0.0, 0.0, 0.12])
            corridor = 0.5 * (raw_approach + target_approach)
            corridor[2] = 0.50
            points = (
                current,
                work_entry,
                self._pose(raw_approach),
                self._pose(raw),
                self._pose(raw_approach),
                self._pose(corridor),
                self._pose(target_approach),
                self._pose(target),
                self._pose(target + np.asarray([0.0, 0.0, 0.14])),
            )
            return (
                PolylineTrajectory(points, self.speed_m_s),
                # The preceding settle phase removes the visible error; the
                # final snap eliminates the remaining solver-scale residual
                # before the tray weld is enabled.
                lambda: registry.place_base_on_tray(snap=True),
            )

        if kind in {TaskType.INSERT_FIN, TaskType.ADJUST_FIN}:
            work_entry = self.park_flange_pose.transformed(self.controller.tool_transform)
            fin_id = str(task.payload["fin_id"])
            # The grasp TCP lies at the fin centre, safely inside the 60 mm
            # high blank instead of hovering above its top edge.
            raw = registry.free_body_pose(fin_id).position.copy()
            product = self._product()
            fin = next(item for item in product.active_fins if item.fin_id == fin_id)
            slot = np.asarray(registry.product_to_world(fin.target_position), dtype=float)
            goal = slot.copy()
            park_tcp = self.park_flange_pose.transformed(self.controller.tool_transform).position
            toward_park = park_tcp[:2] - goal[:2]
            toward_park /= max(float(np.linalg.norm(toward_park)), 1e-9)
            retreat = goal.copy()
            retreat[:2] += 0.09 * toward_park
            retreat[2] = max(goal[2] + 0.10, 0.42)
            # The X-aligned 300 mm fin projects toward the quick-change rack.
            # Lift its lower edge above the rack beam before any lateral move.
            raw_approach = raw + np.asarray([0.0, 0.0, 0.26])
            goal_approach = goal + np.asarray([0.0, 0.0, 0.10])
            corridor = 0.5 * (raw_approach + goal_approach)
            corridor[2] = 0.50
            points = (
                current,
                work_entry,
                self._pose(raw_approach),
                self._pose(raw),
                self._pose(raw_approach),
                self._pose(corridor),
                self._pose(goal_approach),
                self._pose(goal),
                self._pose(retreat),
            )
            return (
                PolylineTrajectory(points, self.speed_m_s),
                lambda: registry.place_fin_in_slot(fin_id, snap=True),
            )

        if kind in {TaskType.PRE_INSPECT, TaskType.MATERIAL_INSPECT}:
            product_pose = registry.product_pose().position
            height = 0.44 if kind is TaskType.PRE_INSPECT else 0.38
            overview = self._pose([product_pose[0], product_pose[1], height])
            end_view = self._pose([product_pose[0] + 0.17, product_pose[1], height - 0.05], math.pi / 2.0)
            return PolylineTrajectory((current, overview, end_view, overview), self.speed_m_s), noop

        if kind in {TaskType.APPLY_MATERIAL, TaskType.REAPPLY_MATERIAL}:
            current = self.controller.current_tcp_pose()
            path_id = str(task.payload["path_id"])
            start, end = self._path_world_points(path_id)
            # Keep the long physical nozzle outside the 2 mm fin while the
            # virtual material remains on its product-coordinate root path.
            lateral_sign = -1.0 if path_id.endswith("_left") else 1.0
            start[1] += lateral_sign * 0.004
            end[1] += lateral_sign * 0.004
            start[2] += 0.012
            end[2] += 0.012
            lance_quaternion = _forward_tilted_lance_quaternion()
            approach = Pose(start + np.asarray([0.0, 0.0, 0.08]), lance_quaternion)
            start_pose = Pose(start, lance_quaternion)
            end_pose = Pose(end, lance_quaternion)
            retreat = Pose(end + np.asarray([0.0, 0.0, 0.065]), lance_quaternion)
            corridor = safe_corridor_trajectory(
                current,
                approach,
                clearance_z=0.43,
                speed_m_s=self.speed_m_s,
            )
            points = (*corridor.waypoints, start_pose, end_pose, retreat)
            return PolylineTrajectory(tuple(points), self.speed_m_s, 0.01), lambda: registry.set_path_visible(
                path_id, True, coverage=1.0
            )

        if kind is TaskType.LOCK_FIXTURE:
            button = _site_pose(self.scene, "fixture_lock_button")
            return (
                safe_corridor_trajectory(
                    current,
                    self._pose(button.position),
                    clearance_z=0.43,
                    speed_m_s=self.speed_m_s,
                ),
                lambda: registry.set_fixture_locked(True),
            )

        if kind is TaskType.LOAD_FURNACE:
            current = self.controller.current_tcp_pose()
            tray = registry.free_body_pose("assembly_tray").position + np.asarray([0.0, 0.0, 0.12])
            furnace = _site_pose(self.scene, "furnace_tray_pose").position + np.asarray([0.0, -0.12, 0.16])
            furnace_yaw = math.radians(60.0)
            points = (
                current,
                self._pose([tray[0], tray[1], 0.52]),
                self._pose(tray),
                self._pose([tray[0], tray[1], 0.48]),
                self._pose([furnace[0], furnace[1], 0.48], furnace_yaw),
                self._pose(furnace, furnace_yaw),
                self._pose([furnace[0], furnace[1] - 0.03, 0.44], furnace_yaw),
            )

            def finish_load() -> None:
                registry.place_tray_in_furnace()

            return PolylineTrajectory(points, self.speed_m_s), finish_load

        if kind is TaskType.UNLOAD_FURNACE:
            current = self.controller.current_tcp_pose()
            furnace = _site_pose(self.scene, "furnace_tray_pose").position + np.asarray([0.0, -0.10, 0.16])
            furnace_yaw = math.radians(60.0)
            table3_site = _site_pose(self.scene, "post_inspection_pose").position
            table3 = table3_site + np.asarray([0.0, 0.0, 0.25])
            points = (
                current,
                self._pose(furnace, furnace_yaw),
                self._pose([furnace[0], furnace[1], 0.48], furnace_yaw),
                self._pose([0.10, 0.05, 0.44], furnace_yaw),
                self._pose([table3[0], table3[1], 0.48]),
                self._pose(table3),
                self._pose([table3[0], table3[1] - 0.16, 0.48]),
            )

            def finish_unload() -> None:
                registry.set_weld("furnace_tray_weld", False)
                registry.set_weld("arm2_tray_carry", False)
                for fin in self._product().active_fins:
                    registry.braze_fin_to_base(fin.fin_id)

            return PolylineTrajectory(points, self.speed_m_s), finish_unload

        if kind is TaskType.POST_INSPECT:
            product = registry.product_pose().position
            top = self._pose(product + np.asarray([0.0, 0.0, 0.32]), math.pi)
            side = self._pose(product + np.asarray([0.14, 0.0, 0.16]), -math.pi / 2.0)
            return PolylineTrajectory((current, top, side, top), self.speed_m_s), noop

        raise ValueError(f"unsupported scene task: {kind.value}")

    def _required_arm1_tool(self, task: TaskSpec) -> str:
        kind = TaskType(task.task_type)
        if kind is TaskType.LOAD_BASE:
            return "suction_tool"
        if kind in {TaskType.INSERT_FIN, TaskType.ADJUST_FIN}:
            return "parallel_gripper"
        raise RuntimeError(f"Arm1 has no registered tool for {kind.value}")

    def _prepare_arm1_work(self, task: TaskSpec) -> None:
        """Generate the work path only after the requested tool is mounted."""

        trajectory, callback = self._task_goal(task)
        if self._should_park(task):
            park_pose = self.park_flange_pose.transformed(self.controller.tool_transform)
            return_path = safe_corridor_trajectory(
                trajectory.waypoints[-1],
                park_pose,
                clearance_z=0.50,
                speed_m_s=self.speed_m_s,
            )
            trajectory = PolylineTrajectory(
                (*trajectory.waypoints, *return_path.waypoints[1:]),
                self.speed_m_s,
                0.01,
            )
        self.trajectory = trajectory
        self._callback = callback
        self._build_arm1_steps(task, trajectory, callback)

    def _build_arm1_tool_change(self, task: TaskSpec, required: str) -> None:
        """Build a visible hover/dock/lock/retreat quick-change sequence."""

        manager = self.scene.arm1_tools

        def segment(left: Pose, right: Pose, speed: float) -> Arm1Step:
            return Arm1Step(
                "trajectory",
                trajectory=PolylineTrajectory((left, right), speed, 0.01),
            )

        steps: list[Arm1Step] = []
        current = manager.current_tool
        current_pose = self.controller.current_tcp_pose()
        if current is not None:
            hover, dock, _ = manager.change_poses(current, hover_m=0.10)
            hover_tcp = manager.tcp_for_flange(hover, current)
            dock_tcp = manager.tcp_for_flange(dock, current)
            park_tcp = self.park_flange_pose.transformed(self.controller.tool_transform)
            steps.extend(
                [
                    segment(current_pose, park_tcp, 0.18),
                    segment(park_tcp, hover_tcp, 0.18),
                    segment(hover_tcp, dock_tcp, 0.025),
                    Arm1Step("hold", duration=0.25),
                    Arm1Step("action", action=lambda name=current: manager.undock(name)),
                    Arm1Step("hold", duration=0.20),
                    segment(dock, hover, 0.035),
                ]
            )
            current_pose = hover

        hover, dock, _ = manager.change_poses(required, hover_m=0.10)
        steps.extend(
            [
                segment(current_pose, hover, 0.16),
                segment(hover, dock, 0.025),
                Arm1Step("hold", duration=0.25),
                Arm1Step("action", action=lambda: manager.dock(required)),
                Arm1Step("hold", duration=0.25),
                segment(
                    manager.tcp_for_flange(dock, required),
                    manager.tcp_for_flange(hover, required),
                    0.035,
                ),
                Arm1Step("action", action=lambda: self._prepare_arm1_work(task)),
            ]
        )
        self._arm1_steps = steps

    @staticmethod
    def _required_arm2_tool(task: TaskSpec) -> str | None:
        kind = TaskType(task.task_type)
        if kind in {TaskType.APPLY_MATERIAL, TaskType.REAPPLY_MATERIAL}:
            return "brazing_dispenser"
        if kind in {TaskType.LOAD_FURNACE, TaskType.UNLOAD_FURNACE}:
            return "tray_transfer"
        return None

    def _prepare_arm2_work(self, task: TaskSpec) -> None:
        """Generate Arm2 work only after the selected TCP is mounted."""

        trajectory, callback = self._task_goal(task)
        kind = TaskType(task.task_type)
        if self._should_park(task) and kind not in {
            TaskType.LOAD_FURNACE,
            TaskType.UNLOAD_FURNACE,
        }:
            park_pose = self.park_flange_pose.transformed(self.controller.tool_transform)
            if park_pose.position[2] < 0.43:
                position = park_pose.position.copy()
                position[2] = 0.43
                park_pose = Pose(position, park_pose.quaternion)
            return_path = safe_corridor_trajectory(
                trajectory.waypoints[-1],
                park_pose,
                clearance_z=0.43,
                speed_m_s=self.speed_m_s,
            )
            trajectory = PolylineTrajectory(
                (*trajectory.waypoints, *return_path.waypoints[1:]),
                self.speed_m_s,
                0.01,
            )
        self.trajectory = trajectory
        self._callback = callback
        self._errors.clear()
        if kind in {TaskType.LOAD_FURNACE, TaskType.UNLOAD_FURNACE}:
            points = trajectory.waypoints

            def path(start: int, stop: int, speed: float) -> PolylineTrajectory:
                return PolylineTrajectory(tuple(points[start:stop]), speed, 0.01)

            def attach_tray() -> None:
                if kind is TaskType.UNLOAD_FURNACE:
                    self.scene.registry.set_weld("furnace_tray_weld", False)
                self.scene.registry.carry_tray(True)
                self._capture_tray_group()

            if kind is TaskType.LOAD_FURNACE:

                def target_pose() -> Pose:
                    return _site_pose(self.scene, "furnace_tray_pose")

                approach = path(0, 3, 0.10)
                carry = path(2, 6, 0.10)
                retreat = path(5, len(points), 0.12)
            else:
                table3_site = _site_pose(self.scene, "post_inspection_pose").position
                table3_pose = Pose(
                    np.asarray([table3_site[0], table3_site[1], 0.255]),
                    np.asarray([1.0, 0.0, 0.0, 0.0]),
                )

                def target_pose() -> Pose:
                    return table3_pose

                approach = path(0, 2, 0.08)
                carry = path(1, 6, 0.10)
                retreat = path(5, len(points), 0.12)
            self._arm2_steps = [
                Arm1Step("work", trajectory=approach),
                Arm1Step("hold", duration=0.30),
                Arm1Step("action", action=attach_tray),
                Arm1Step("hold", duration=0.30),
                Arm1Step("work", trajectory=carry),
                Arm1Step("hold", duration=0.30),
                Arm1Step("action", action=self._release_tray_group),
                Arm1Step(
                    "settle_group",
                    duration=0.60,
                    body_name="assembly_tray",
                    target_pose=target_pose,
                ),
                Arm1Step("action", action=callback),
                Arm1Step("work", trajectory=retreat),
            ]
            self._callback = lambda: None
            return
        self._arm2_steps = [Arm1Step("work", trajectory=trajectory)]

    def _build_arm2_tool_change(self, task: TaskSpec, required: str) -> None:
        """Build the visible front-rack quick-change sequence for Arm2."""

        manager = self.scene.tools

        def segment(left: Pose, right: Pose, speed: float) -> Arm1Step:
            return Arm1Step(
                "kinematic",
                trajectory=PolylineTrajectory((left, right), speed, 0.01),
            )

        steps: list[Arm1Step] = []
        current = manager.current_tool
        current_pose = self.controller.current_tcp_pose()
        if current is not None:
            hover, dock, _ = manager.change_poses(current, hover_m=0.10)
            hover_tcp = manager.tcp_for_flange(hover, current)
            dock_tcp = manager.tcp_for_flange(dock, current)
            steps.extend(
                [
                    segment(current_pose, hover_tcp, 0.16),
                    segment(hover_tcp, dock_tcp, 0.025),
                    Arm1Step("hold", duration=0.25),
                    Arm1Step("action", action=lambda name=current: manager.undock(name)),
                    Arm1Step("hold", duration=0.20),
                    segment(dock, hover, 0.035),
                ]
            )
            current_pose = hover

        hover, dock, _ = manager.change_poses(required, hover_m=0.10)
        steps.extend(
            [
                segment(current_pose, hover, 0.14),
                segment(hover, dock, 0.025),
                Arm1Step("hold", duration=0.25),
                Arm1Step("action", action=lambda: manager.dock(required)),
                Arm1Step("hold", duration=0.25),
                segment(
                    manager.tcp_for_flange(dock, required),
                    manager.tcp_for_flange(hover, required),
                    0.035,
                ),
            ]
        )
        if required == "tray_transfer":
            # The two rack poses are reachable with more than one redundant
            # Arm2 joint branch.  Return smoothly to the known front-facing
            # park branch before generating a loaded-tray path; otherwise the
            # first furnace approach can inherit a locally valid rack branch
            # whose next Cartesian sample misses by several millimetres.
            steps.append(Arm1Step("joint_home", duration=1.50))
        steps.append(Arm1Step("action", action=lambda: self._prepare_arm2_work(task)))
        self._arm2_steps = steps

    def _advance_arm2_step(self, now: float) -> None:
        while self._arm2_steps:
            step = self._arm2_steps.pop(0)
            if step.kind == "action":
                if step.action is not None:
                    step.action()
                continue
            if step.kind == "kinematic":
                assert step.trajectory is not None
                self._start_kinematic(step.trajectory, now)
                return
            if step.kind == "work":
                assert step.trajectory is not None
                self._start_kinematic(step.trajectory, now)
                return
            if step.kind == "hold":
                self._arm2_timed = (step, float(now))
                return
            if step.kind == "settle_group":
                if step.body_name is None or step.target_pose is None:
                    raise RuntimeError("Arm2 settle-group step is incomplete")
                step.start_pose = self.scene.registry.free_body_pose(step.body_name)
                step.end_pose = step.target_pose()
                self._arm2_timed = (step, float(now))
                return
            if step.kind == "joint_home":
                step.joint_start = np.asarray(
                    self.scene.data.qpos[self.controller.qpos_ids], dtype=float
                ).copy()
                step.joint_end = self.park_joint_positions.copy()
                self._arm2_timed = (step, float(now))
                return
            raise RuntimeError(f"unknown Arm2 step: {step.kind}")

        self._callback()
        self._carried_group.clear()
        self._transport_active = False
        self._arm1_tool_roll_lock = None
        self.controller.enabled = False
        self.controller.hold()
        self._done = True
        self.task = None

    def start_task(self, task: TaskSpec, now: float) -> None:
        if self.task is not None:
            raise RuntimeError(f"{self.arm_name} is already executing {self.task.task_id}")
        self.task = task
        self.started_at = float(now)
        self.error = ""
        self._done = False
        self._errors.clear()
        self._followup_segments.clear()
        self._kinematic_motion = None
        self._kinematic_targets = ()
        self._carried_body = None
        self._carry_relative = None
        self._carried_orientation_lock = None
        self._carried_group.clear()
        self._arm1_tool_roll_lock = None
        self._transport_active = False
        self._arm1_steps.clear()
        self._arm1_timed = None
        self._arm2_steps.clear()
        self._arm2_timed = None
        self.deadline = self.started_at + task.timeout
        if self.arm_name == "arm1" and not self.fast:
            required = self._required_arm1_tool(task)
            if self.scene.arm1_tools.current_tool == required:
                self._prepare_arm1_work(task)
            else:
                self._build_arm1_tool_change(task, required)
            self._advance_arm1_step(now)
            return
        if self.arm_name == "arm1" and self.fast:
            self.scene.arm1_tools.change_tool(self._required_arm1_tool(task))
        if self.arm_name == "arm2" and not self.fast:
            required = self._required_arm2_tool(task)
            if required is None or self.scene.tools.current_tool == required:
                self._prepare_arm2_work(task)
            else:
                self._build_arm2_tool_change(task, required)
            self._advance_arm2_step(now)
            return
        if self.arm_name == "arm2" and self.fast:
            required = self._required_arm2_tool(task)
            if required is not None:
                self.scene.tools.change_tool(required)
        trajectory, callback = self._task_goal(task)
        if self._should_park(task):
            park_pose = self.park_flange_pose.transformed(self.controller.tool_transform)
            if self.arm_name == "arm2" and park_pose.position[2] < 0.43:
                park_position = park_pose.position.copy()
                park_position[2] = 0.43
                park_pose = Pose(park_position, park_pose.quaternion)
            park_clearance = {"arm1": 0.50, "arm2": 0.43, "arm3": 0.52}[self.arm_name]
            return_path = safe_corridor_trajectory(
                trajectory.waypoints[-1],
                park_pose,
                clearance_z=park_clearance,
                speed_m_s=self.speed_m_s,
            )
            trajectory = PolylineTrajectory(
                (*trajectory.waypoints, *return_path.waypoints[1:]),
                self.speed_m_s,
                0.01,
            )
        if (
            self.arm_name == "arm3"
            and not self.fast
            and TaskType(task.task_type)
            in {TaskType.PRE_INSPECT, TaskType.MATERIAL_INSPECT, TaskType.POST_INSPECT}
        ):
            self.trajectory = trajectory
            self._callback = callback
            self._start_kinematic(trajectory, now)
            return
        milestone = self._transport_milestone(task)
        if milestone is not None and not self.fast:
            waypoint_index, action = milestone
            if not 0 < waypoint_index < len(trajectory.waypoints) - 1:
                raise RuntimeError(f"invalid transport milestone for {task.task_id}")
            if self.arm_name == "arm1":
                approach = PolylineTrajectory(
                    tuple(trajectory.waypoints[: waypoint_index + 1]),
                    self.speed_m_s,
                    trajectory.sample_spacing_m,
                )
                carry = PolylineTrajectory(
                    tuple(trajectory.waypoints[waypoint_index:]),
                    self.speed_m_s,
                    trajectory.sample_spacing_m,
                )
                self._followup_segments.append((carry, action))
                trajectory = approach
            else:
                approach = PolylineTrajectory(
                    tuple(trajectory.waypoints[: waypoint_index + 1]),
                    self.speed_m_s,
                    trajectory.sample_spacing_m,
                )
                carry = PolylineTrajectory(
                    tuple(trajectory.waypoints[waypoint_index:]),
                    self.speed_m_s,
                    trajectory.sample_spacing_m,
                )
                self._followup_segments.append((carry, action))
                trajectory = approach
        self.trajectory = trajectory
        self._callback = callback
        if self.fast:
            callback()
            self._done = True
        else:
            self.executor.start(self.trajectory, now, timeout_s=task.timeout)

    def poll_task(self, now: float) -> ActorResult:
        if self.task is None:
            return ActorResult.SUCCEEDED
        if self.error:
            return ActorResult.FAILED
        if self._done:
            self.task = None
            return ActorResult.SUCCEEDED
        if self._arm2_timed is not None:
            self.scene.data.qpos[self.controller.qpos_ids] = self.controller.q_command
            self.scene.data.qvel[self.controller.dof_ids] = 0.0
            self.scene.data.ctrl[self.controller.actuator_ids] = self.controller.q_command
            self.scene.sync_mounted_extensions()
            step, step_started = self._arm2_timed
            linear = float(np.clip((float(now) - step_started) / max(step.duration, 1e-6), 0.0, 1.0))
            amount = linear * linear * (3.0 - 2.0 * linear)
            if step.kind == "joint_home":
                assert step.joint_start is not None and step.joint_end is not None
                joints = (1.0 - amount) * step.joint_start + amount * step.joint_end
                self.controller.q_command = joints
                self.scene.data.qpos[self.controller.qpos_ids] = joints
                self.scene.data.qvel[self.controller.dof_ids] = 0.0
                self.scene.data.ctrl[self.controller.actuator_ids] = joints
                self.scene.sync_mounted_extensions()
            elif step.kind == "settle_group":
                assert step.start_pose is not None and step.end_pose is not None
                left = step.start_pose.quaternion
                right = step.end_pose.quaternion
                if float(np.dot(left, right)) < 0.0:
                    right = -right
                quaternion = (1.0 - amount) * left + amount * right
                quaternion /= max(float(np.linalg.norm(quaternion)), 1e-12)
                position = (1.0 - amount) * step.start_pose.position + amount * step.end_pose.position
                self._move_assembly(Pose(position, quaternion))
            if linear < 1.0:
                return ActorResult.RUNNING
            self._arm2_timed = None
            if step.action is not None:
                step.action()
            self._advance_arm2_step(now)
            return ActorResult.SUCCEEDED if self.task is None else ActorResult.RUNNING
        if self._arm1_timed is not None:
            self.scene.data.qpos[self.controller.qpos_ids] = self.controller.q_command
            self.scene.data.qvel[self.controller.dof_ids] = 0.0
            self.scene.data.ctrl[self.controller.actuator_ids] = self.controller.q_command
            self.scene.sync_mounted_extensions()
            self._sync_carried_body()
            step, step_started = self._arm1_timed
            linear = float(np.clip((float(now) - step_started) / max(step.duration, 1e-6), 0.0, 1.0))
            amount = linear * linear * (3.0 - 2.0 * linear)
            value = step.start_value + amount * (step.end_value - step.start_value)
            if step.kind == "joint_home":
                assert step.joint_start is not None and step.joint_end is not None
                joints = (1.0 - amount) * step.joint_start + amount * step.joint_end
                self.controller.q_command = joints
                self.scene.data.qpos[self.controller.qpos_ids] = joints
                self.scene.data.qvel[self.controller.dof_ids] = 0.0
                self.scene.data.ctrl[self.controller.actuator_ids] = joints
                self.scene.sync_mounted_extensions()
            elif step.kind == "gripper":
                self.scene.registry.set_arm1_gripper_closed(value)
            elif step.kind == "suction":
                self.scene.registry.set_arm1_suction_fraction(value)
            elif step.kind == "settle":
                assert step.body_name is not None
                assert step.start_pose is not None and step.end_pose is not None
                # Table2 can move by a fraction of a millimetre while the
                # released payload is being aligned.  Follow the live slot
                # pose throughout the settle interval instead of welding the
                # payload to a target captured 0.6 s earlier.
                if step.target_pose is not None:
                    step.end_pose = step.target_pose()
                left = step.start_pose.quaternion
                right = step.end_pose.quaternion
                if float(np.dot(left, right)) < 0.0:
                    right = -right
                quaternion = (1.0 - amount) * left + amount * right
                quaternion /= max(float(np.linalg.norm(quaternion)), 1e-12)
                position = (1.0 - amount) * step.start_pose.position + amount * step.end_pose.position
                self.scene.registry.set_free_body_pose(
                    step.body_name,
                    Pose(position, quaternion),
                    forward=True,
                )
            if linear < 1.0:
                return ActorResult.RUNNING
            self._arm1_timed = None
            if step.action is not None:
                step.action()
            self._advance_arm1_step(now)
            return ActorResult.SUCCEEDED if self.task is None else ActorResult.RUNNING
        if self._kinematic_motion is not None:
            joint_samples, motion_started, duration = self._kinematic_motion
            elapsed = max(0.0, float(now) - motion_started)
            linear = float(np.clip(elapsed / duration, 0.0, 1.0))
            amount = linear * linear * (3.0 - 2.0 * linear)
            scaled = amount * max(0, len(joint_samples) - 1)
            left = min(int(math.floor(scaled)), len(joint_samples) - 1)
            right = min(left + 1, len(joint_samples) - 1)
            fraction = scaled - left
            self.controller.q_command = joint_samples[left] + fraction * (
                joint_samples[right] - joint_samples[left]
            )
            self.scene.data.qpos[self.controller.qpos_ids] = self.controller.q_command
            self.scene.data.qvel[self.controller.dof_ids] = 0.0
            self.scene.data.ctrl[self.controller.actuator_ids] = self.controller.q_command
            self.scene.sync_mounted_extensions()
            self._sync_carried_body()
            if self._kinematic_targets:
                desired_position = (1.0 - fraction) * self._kinematic_targets[left].position + fraction * (
                    self._kinematic_targets[right].position
                )
                actual = self.controller.current_tcp_pose()
                self._errors.append(float(np.linalg.norm(actual.position - desired_position)))
            if linear < 1.0:
                return ActorResult.RUNNING
            self._kinematic_motion = None
            self._kinematic_targets = ()
            if self.arm_name == "arm1":
                self._advance_arm1_step(now)
                return ActorResult.SUCCEEDED if self.task is None else ActorResult.RUNNING
            if self.arm_name == "arm2":
                if self.task is not None and TaskType(self.task.task_type) in {
                    TaskType.APPLY_MATERIAL,
                    TaskType.REAPPLY_MATERIAL,
                }:
                    rmse = math.sqrt(float(np.mean(np.square(self._errors)))) if self._errors else 0.0
                    self.task.payload["trajectory_rmse_m"] = rmse
                    self.task.payload["trajectory_max_error_m"] = max(self._errors, default=0.0)
                self._advance_arm2_step(now)
                return ActorResult.SUCCEEDED if self.task is None else ActorResult.RUNNING
            if self._followup_segments:
                followup, action = self._followup_segments.pop(0)
                if action is not None:
                    action()
                    self._transport_active = True
                    self._capture_carried_body()
                self._start_kinematic(followup, now)
                return ActorResult.RUNNING
            self._callback()
            self._transport_active = False
            self._carried_body = None
            self._carry_relative = None
            self.controller.enabled = False
            self.controller.hold()
            self._done = True
            self.task = None
            return ActorResult.SUCCEEDED
        state = self.executor.tick(now)
        if state is ExecutionState.ERROR:
            self.error = self.executor.error
            return ActorResult.FAILED
        if state is ExecutionState.COMPLETE:
            if self.arm_name == "arm2" and self.task is not None:
                if TaskType(self.task.task_type) in {
                    TaskType.APPLY_MATERIAL,
                    TaskType.REAPPLY_MATERIAL,
                }:
                    self.task.payload["trajectory_rmse_m"] = self.executor.rmse_m
                    self.task.payload["trajectory_max_error_m"] = self.executor.max_error_m
                self._advance_arm2_step(now)
                return ActorResult.SUCCEEDED if self.task is None else ActorResult.RUNNING
            if self._followup_segments:
                followup, action = self._followup_segments.pop(0)
                if action is not None:
                    action()
                    self._transport_active = True
                self.trajectory = followup
                self.executor.start(
                    followup,
                    now,
                    timeout_s=max(0.1, self.deadline - float(now)),
                )
                if self.executor.state is ExecutionState.ERROR:
                    self.error = self.executor.error
                    return ActorResult.FAILED
                return ActorResult.RUNNING
            if self.task is not None and TaskType(self.task.task_type) in {
                TaskType.APPLY_MATERIAL,
                TaskType.REAPPLY_MATERIAL,
            }:
                self.task.payload["trajectory_rmse_m"] = self.executor.rmse_m
                self.task.payload["trajectory_max_error_m"] = self.executor.max_error_m
            self._callback()
            self._transport_active = False
            self._done = True
            self.task = None
            return ActorResult.SUCCEEDED
        return ActorResult.RUNNING

    def cancel(self) -> None:
        self.controller.hold()
        self.executor.cancel(self.scene.time)
        self._release_transport()
        self.task = None
        self.trajectory = None
        self._followup_segments.clear()
        self._kinematic_motion = None
        self._kinematic_targets = ()
        self._carried_body = None
        self._carry_relative = None
        self._carried_orientation_lock = None
        self._carried_group.clear()
        self._arm1_tool_roll_lock = None
        self._arm1_steps.clear()
        self._arm1_timed = None
        self._arm2_steps.clear()
        self._arm2_timed = None
        self.scene.registry.set_arm1_gripper_closed(0.0)
        self.scene.registry.set_arm1_suction_fraction(0.0)
        self.controller.enabled = False
        self._done = False
        self.error = ""


def build_scene_actors(
    scene: BrazingScene,
    product: Callable[[], ProductState | None],
    *,
    fast: bool = False,
) -> dict[str, SceneTaskActor]:
    return {name: SceneTaskActor(name, scene, product, fast=fast) for name in ("arm1", "arm2", "arm3")}


__all__ = ["SceneTaskActor", "build_scene_actors"]
