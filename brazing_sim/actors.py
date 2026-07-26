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

from .conveyor import ConveyorTaskActor
from .domain import ProductState, TaskSpec, TaskType
from .fixture import FixtureTaskActor
from .inspection import top_down_inspection_pose
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
from .profiles import quintic_time_scaling
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
        self._carried_tool_orientation_lock: np.ndarray | None = None
        self._transport_active = False
        self._arm1_steps: list[Arm1Step] = []
        self._arm1_timed: tuple[Arm1Step, float] | None = None
        self._arm2_steps: list[Arm1Step] = []
        self._arm2_timed: tuple[Arm1Step, float] | None = None
        self._arm3_joint_home: tuple[np.ndarray, np.ndarray, float, float] | None = None
        self._material_path_ids: tuple[str, ...] = ()
        self._material_start: np.ndarray | None = None
        self._material_end: np.ndarray | None = None
        self._material_progress = 0.0
        self._material_reverse = False
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
        if kind in {TaskType.INSERT_FIN, TaskType.ADJUST_FIN}:
            # The release trajectory already clears the fixture vertically.
            # Intermediate fins can continue through the high corridor to the
            # next raw blank; the final task still returns to the canonical
            # pose before another actor enters Table2.
            return bool(task.payload.get("park_after", True))
        if kind in {TaskType.APPLY_MATERIAL, TaskType.REAPPLY_MATERIAL}:
            # Intermediate slots remain over the product and transition
            # directly to the adjacent slot. Only the final material task
            # clears Table2 for Arm3.
            return bool(task.payload.get("park_after", True))
        if kind is TaskType.POST_INSPECT:
            # Batch jobs may choose whether adjacent output-buffer scans use
            # one continuous motion or return to the canonical safe posture.
            return bool(task.payload.get("park_after", True))
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
        kind = TaskType(self.task.task_type) if self.task is not None else None
        is_arm1_rigid_payload_motion = (
            self.arm_name == "arm1"
            and kind in {TaskType.LOAD_BASE, TaskType.INSERT_FIN, TaskType.ADJUST_FIN}
            and self.scene.arm1_tools.current_tool is not None
        )
        is_arm2_material_motion = self.arm_name == "arm2" and kind in {
            TaskType.APPLY_MATERIAL,
            TaskType.REAPPLY_MATERIAL,
        }
        is_arm3_inspection = self.arm_name == "arm3" and kind in {
            TaskType.PRE_INSPECT,
            TaskType.MATERIAL_INSPECT,
            TaskType.POST_INSPECT,
        }
        reference = trajectory.waypoints[0]
        is_strict_vertical = bool(
            is_arm1_rigid_payload_motion
            and len(trajectory.waypoints) >= 2
            and all(
                float(np.linalg.norm(waypoint.position[:2] - reference.position[:2])) <= 1.0e-9
                and abs(float(np.dot(waypoint.quaternion, reference.quaternion))) >= 1.0 - 1.0e-9
                for waypoint in trajectory.waypoints[1:]
            )
        )
        if is_strict_vertical:
            # Installation descent is authored only after the high approach
            # pose has completed its XY/roll/pitch/yaw alignment.  Rebuild
            # every sample from that one reference so no measured residual can
            # leak into the downward segment.
            trajectory = PolylineTrajectory(
                tuple(
                    Pose(
                        np.asarray(
                            [reference.position[0], reference.position[1], waypoint.position[2]],
                            dtype=float,
                        ),
                        reference.quaternion,
                    )
                    for waypoint in trajectory.waypoints
                ),
                trajectory.speed_m_s,
                min(trajectory.sample_spacing_m, 0.0010),
            )
        elif is_arm1_rigid_payload_motion and trajectory.sample_spacing_m > 0.0010:
            # Joint interpolation between otherwise accurate Cartesian IK
            # samples can bow the wrist attitude by a few hundredths of a
            # degree.  One-millimetre samples keep the rigidly held fin below
            # the visual threshold without changing the authored speed.
            trajectory = PolylineTrajectory(
                trajectory.waypoints,
                trajectory.speed_m_s,
                0.0010,
            )
        if is_arm2_material_motion and trajectory.sample_spacing_m > 0.003:
            # Linear interpolation in joint space can bow the 220 mm lance
            # between otherwise accurate IK samples.  Three-millimetre
            # sampling keeps that inter-sample tool-axis deviation below the
            # 0.1 degree dispensing limit without changing feed speed.
            trajectory = PolylineTrajectory(
                trajectory.waypoints,
                trajectory.speed_m_s,
                0.003,
            )
        checks = self.controller.validate_trajectory(
            trajectory,
            position_tolerance_m=(
                0.0001
                if is_arm1_rigid_payload_motion
                # The far S2B repair envelope remains inside the process hard
                # limit of 5 mm even though it is outside the nominal 3 mm
                # target.  Accept it so a one-way pallet is never reversed.
                else 0.0045 if is_arm2_material_motion else None
            ),
            orientation_tolerance_rad=(
                math.radians(0.01)
                if is_arm1_rigid_payload_motion
                # Arm2's S2B local-repair pose is close to the edge of its
                # workspace.  The residual here is predominantly free tool
                # roll; the lance Z axis is verified independently and stays
                # vertical.  Do not reject a safe repair for 0.16 degrees of
                # irrelevant roll error.
                else math.radians(0.35) if is_arm2_material_motion else None
            ),
            # Arm2 still exposes the standard 5D controller externally, but
            # its precomputed dispensing samples close the remaining roll
            # residual as a full pose.  This keeps the long dual nozzle
            # vertical to better than 0.1 degree during kinematic playback.
            full_orientation=(is_arm1_rigid_payload_motion or is_arm2_material_motion or is_arm3_inspection),
        )
        failed_entry = next(
            ((index, result) for index, result in enumerate(checks) if not result.reachable),
            None,
        )
        if failed_entry is not None:
            failed_index, failed = failed_entry
            failed_target = trajectory.samples(min(trajectory.sample_spacing_m, 0.01))[failed_index]
            raise RuntimeError(
                "kinematic path is unreachable at sample "
                f"{failed_index + 1}/{len(checks)}: "
                f"{failed.position_error_m * 1000:.2f} mm / "
                f"{math.degrees(failed.orientation_error_rad):.2f} deg; "
                f"target=({failed_target.position[0]:.3f}, "
                f"{failed_target.position[1]:.3f}, "
                f"{failed_target.position[2]:.3f})"
            )
        self.trajectory = trajectory
        joint_samples = np.stack([result.joint_positions for result in checks])
        self._kinematic_targets = trajectory.samples(min(trajectory.sample_spacing_m, 0.01))
        if len(self._kinematic_targets) != len(joint_samples):
            raise RuntimeError("kinematic target/sample count mismatch")
        if is_strict_vertical:
            # Reaffirm alignment at the safe upper endpoint before the descent
            # clock starts.  All following samples then vary only Z.
            self.scene.data.qpos[self.controller.qpos_ids] = joint_samples[0]
            self.scene.data.qvel[self.controller.dof_ids] = 0.0
            self.scene.data.ctrl[self.controller.actuator_ids] = joint_samples[0]
            self.controller.q_command = joint_samples[0].copy()
            self.scene.sync_mounted_extensions(self.arm_name)
            self._sync_carried_body()
        else:
            # The first Cartesian sample is the end of the preceding segment.
            # Preserve the actual redundant-joint configuration instead of
            # letting IK choose another valid branch in ordinary free travel.
            joint_samples[0] = np.asarray(self.scene.data.qpos[self.controller.qpos_ids], dtype=float)
        if final_joint_positions is not None:
            joint_samples[-1] = np.asarray(final_joint_positions, dtype=float)
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
        self._carried_tool_orientation_lock = self.controller.current_tcp_pose().quaternion.copy()

    def _sync_carried_body(self) -> None:
        if self._carried_body is None or self._carry_relative is None:
            return
        tool_body = self.scene.arm1_tools.current_body
        if tool_body is None:
            raise RuntimeError("Arm1 lost its tool while carrying a workpiece")
        if self.task is not None and TaskType(self.task.task_type) in {
            TaskType.INSERT_FIN,
            TaskType.ADJUST_FIN,
        }:
            # The finger servos have already animated the close.  Keep their
            # final inner faces exactly at +/-1 mm for the whole rigid carry.
            self.scene.registry.snap_arm1_gripper_closed(forward=False)
        gripper = self.scene.registry.free_body_pose(tool_body)
        carried_pose = gripper.transformed(self._carry_relative)
        # qpos is consumed by the mj_step that follows this actor tick.  A
        # second full mj_forward here only duplicated dynamics/contact work;
        # kinematics alone keeps same-frame body/site reads coherent.
        self.scene.registry.set_free_body_pose(self._carried_body, carried_pose, forward=False)
        self.scene.mujoco.mj_kinematics(self.scene.model, self.scene.data)

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
                if TaskType(task.task_type) in {TaskType.INSERT_FIN, TaskType.ADJUST_FIN}:
                    # Exact jaw seating establishes the canonical rigid
                    # gripper-to-fin transform; capture only after it exists.
                    action()
                    self._capture_carried_body()
                else:
                    self._capture_carried_body()
                    action()
                self._transport_active = True
                # LOAD_BASE and INSERT_FIN are compound legacy commands, but
                # the planning DAG exposes pick and place as separate nodes.
                # Persist this physical milestone as soon as the rigid grasp
                # constraint exists so the pick node need not wait for place.
                task.payload["physical_pick_complete"] = True

            return run

        def place() -> None:
            place_action()
            # The payload is now supported by its destination constraint.  A
            # later finger/suction release and safe retreat belong to the
            # already-started compound command, not to the pick milestone.
            task.payload["physical_place_complete"] = True
            self._transport_active = False
            self._carried_body = None
            self._carry_relative = None
            self._carried_tool_orientation_lock = None

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
                Arm1Step("trajectory", trajectory=path(3, 9, cruise_speed)),
                Arm1Step("trajectory", trajectory=path(8, 10, 0.050)),
                Arm1Step("hold", duration=0.35),
                # Transfer directly from the suction weld to the tray weld at
                # the end of the already-aligned pure-Z descent.  The retired
                # release-then-settle sequence visibly rotated the loose base
                # only when it was millimetres above the pocket.
                Arm1Step("action", action=place),
                Arm1Step("suction", duration=0.60, start_value=1.0, end_value=0.0),
                Arm1Step("hold", duration=0.25),
                Arm1Step("trajectory", trajectory=path(9, len(points), 0.12)),
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
                    action=attach(lambda: self.scene.registry.seat_and_grasp_fin(fin_id)),
                ),
                Arm1Step("hold", duration=0.35),
                Arm1Step("trajectory", trajectory=path(3, 7, cruise_speed)),
                Arm1Step("trajectory", trajectory=path(6, 8, 0.025)),
                Arm1Step("hold", duration=0.35),
                # Switch directly from the rigid grasp weld to the comb weld
                # while the fin is still held in its final slot pose.  There
                # is no released-body slide or post-release snap.
                Arm1Step("action", action=place),
                # With the 20 mm A-pitch, fully opening the fingers at the
                # root would sweep them into the neighbouring installed fin.
                # Open only enough to release the 2 mm blank, retreat above
                # the fin tops, then finish opening in free space.
                Arm1Step("gripper", duration=0.55, start_value=1.0, end_value=0.85),
                Arm1Step("hold", duration=0.20),
                Arm1Step("trajectory", trajectory=path(7, len(points), 0.12)),
                Arm1Step("gripper", duration=0.55, start_value=0.85, end_value=0.0),
                Arm1Step("hold", duration=0.25),
            ]
            if self._should_park(task):
                self._arm1_steps.append(Arm1Step("joint_home", duration=1.25))
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
        self._carried_tool_orientation_lock = None
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

    def _framed_inspection_trajectory(self, current: Pose, scan: Pose) -> PolylineTrajectory:
        """Reorient above the home XY before moving a downward camera laterally."""

        approach = Pose(
            scan.position + np.asarray([0.0, 0.0, 0.060], dtype=float),
            scan.quaternion,
        )
        lift = Pose(
            np.asarray(
                [
                    current.position[0],
                    current.position[1],
                    max(float(current.position[2]), 0.620),
                ],
                dtype=float,
            ),
            current.quaternion,
        )
        reorient = Pose(lift.position, scan.quaternion)
        return PolylineTrajectory(
            (current, lift, reorient, approach, scan, approach),
            self.speed_m_s,
        )

    def _path_world_points(self, path_id: str) -> tuple[np.ndarray, np.ndarray]:
        product = self._product()
        path = next(item for item in product.active_paths if item.path_id == path_id)
        start = np.asarray(self.scene.registry.product_to_world(path.local_start), dtype=float)
        end = np.asarray(self.scene.registry.product_to_world(path.local_end), dtype=float)
        return start, end

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
            target = registry.refresh_assembly_target_pose().position + np.asarray(
                [0.0, 0.0, half_thickness + 0.002]
            )
            # S1 is deliberately in front of Arm1 and the quick-change rack is
            # outside the direct Table1 -> S1 corridor.  Use a compact lifted
            # route instead of retaining the old turntable-era detour, whose
            # far-left bypass forced the wrist through an unreachable branch.
            # The base stays horizontal and its lower face clears every rack
            # component before lateral travel begins.
            target_approach = target + np.asarray([0.0, 0.0, 0.10])
            raw_clearance = raw_approach.copy()
            raw_clearance[2] = 0.44
            corridor = 0.5 * (raw_clearance + target_approach)
            corridor[2] = 0.44
            s1_clearance = target_approach.copy()
            s1_clearance[2] = 0.44
            points = (
                current,
                work_entry,
                self._pose(raw_approach),
                self._pose(raw),
                self._pose(raw_approach),
                self._pose(raw_clearance),
                self._pose(corridor),
                self._pose(s1_clearance),
                self._pose(target_approach),
                self._pose(target),
                self._pose(target_approach),
            )
            return (
                PolylineTrajectory(points, self.speed_m_s),
                lambda: registry.place_base_on_tray(snap=False),
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
            vertical_retreat = goal.copy()
            vertical_retreat[2] = retreat[2]
            # The X-aligned 300 mm fin projects toward the quick-change rack.
            # Lift its lower edge above the rack beam before any lateral move.
            raw_approach = raw + np.asarray([0.0, 0.0, 0.26])
            goal_approach = goal + np.asarray([0.0, 0.0, 0.10])
            corridor = 0.5 * (raw_approach + goal_approach)
            corridor[2] = 0.50
            # After placing one fin the gripper is already inside the shared
            # Table1/Table2 work corridor.  Returning to the remote parking
            # pose before every subsequent pickup produced the conspicuous
            # "move away, pause, come back" detour.  Keep the same ten-point
            # motion contract used by the grip/release skill, but replace that
            # parking waypoint with a vertical lift over the current XY.  The
            # first fin still enters through the certified parking pose.
            entry = work_entry
            if bool(task.payload.get("continuous_from_previous", False)):
                local_lift = current.position.copy()
                local_lift[2] = max(float(current.position[2]), float(raw_approach[2]), 0.50)
                entry = self._pose(local_lift)
            points = (
                current,
                entry,
                self._pose(raw_approach),
                self._pose(raw),
                self._pose(raw_approach),
                self._pose(corridor),
                self._pose(goal_approach),
                self._pose(goal),
                self._pose(vertical_retreat),
                self._pose(retreat),
            )
            return (
                PolylineTrajectory(points, self.speed_m_s),
                lambda: registry.place_fin_in_slot(fin_id, snap=False),
            )

        if kind in {TaskType.PRE_INSPECT, TaskType.MATERIAL_INSPECT}:
            product_pose = registry.product_pose()
            spec = self._product().spec
            local_surface_z = 0.5 * spec.base_thickness
            if kind is TaskType.PRE_INSPECT:
                local_surface_z += spec.fin_height
            surface_center = product_pose.position + product_pose.rotation @ np.asarray(
                [0.0, 0.0, local_surface_z],
                dtype=float,
            )
            product_x_axis = product_pose.rotation[:, 0]
            product_yaw = math.atan2(float(product_x_axis[1]), float(product_x_axis[0]))
            scan = top_down_inspection_pose(
                surface_center,
                product_length_m=spec.base_length,
                product_width_m=spec.base_width,
                product_yaw_rad=product_yaw,
            )
            return self._framed_inspection_trajectory(current, scan), noop

        if kind in {TaskType.APPLY_MATERIAL, TaskType.REAPPLY_MATERIAL}:
            process_spec = self._product().spec
            current = self.controller.current_tcp_pose()
            if "path_ids" in task.payload:
                path_ids = tuple(str(value) for value in task.payload["path_ids"])
            else:
                path_ids = (str(task.payload["path_id"]),)
            if not path_ids:
                raise RuntimeError("material task has no brazing path")
            path_points = [self._path_world_points(path_id) for path_id in path_ids]
            # A new part receives both symmetric beads in one pass.  For local
            # repair, the tool still follows the owning slot centre while only
            # the requested deficient bead is made visible.
            starts = np.stack([pair[0] for pair in path_points])
            ends = np.stack([pair[1] for pair in path_points])
            start = np.mean(starts, axis=0)
            end = np.mean(ends, axis=0)
            if len(path_ids) == 1:
                product = self._product()
                target = next(path for path in product.active_paths if path.path_id == path_ids[0])
                fin = next(fin for fin in product.active_fins if fin.fin_id == target.fin_id)
                slot_y = float(self.scene.registry.product_to_world(fin.target_position)[1])
                start[1] = slot_y
                end[1] = slot_y
                if kind is TaskType.REAPPLY_MATERIAL:
                    # Repair only the observed discontinuity, not the complete
                    # 330 mm bead.  This is both the actual recovery intent
                    # and what lets Arm2 service S2B without reversing the
                    # pallet.  The truth model currently reports gap length
                    # but not an image-derived longitudinal coordinate, so the
                    # demonstrator places that local defect at bead centre.
                    axis = end - start
                    length = float(np.linalg.norm(axis))
                    if length > 1.0e-9:
                        direction = axis / length
                        centre = 0.5 * (start + end)
                        repair_length = min(
                            length,
                            max(0.020, float(target.longest_gap_m) + 0.010),
                        )
                        start = centre - 0.5 * repair_length * direction
                        end = centre + 0.5 * repair_length * direction
            reverse_travel = bool(task.payload.get("reverse_travel", False))
            # Product-local X can be opposite world X depending on the tray's
            # docking transform. Mirror the serpentine flag for a normal full
            # pass. A local S2B repair instead starts at whichever endpoint is
            # physically nearer Arm2, avoiding an unnecessary reach across
            # the complete plate while the pallet remains on the one-way line.
            product_x_world = self.scene.registry.product_pose().rotation[:, 0]
            if kind is TaskType.REAPPLY_MATERIAL:
                arm_base = np.asarray(
                    self.scene.data.body("arm2_base").xpos,
                    dtype=float,
                )
                reverse_travel = float(np.linalg.norm(arm_base[:2] - end[:2])) < float(
                    np.linalg.norm(arm_base[:2] - start[:2])
                )
            elif float(product_x_world[0]) < 0.0:
                reverse_travel = not reverse_travel
            if reverse_travel:
                start, end = end.copy(), start.copy()
            task.payload["travel_direction"] = "negative_x" if reverse_travel else "positive_x"
            start[2] += process_spec.nozzle_tip_height
            end[2] += process_spec.nozzle_tip_height
            # The dual-nozzle centreline is a process constraint: its local Z
            # axis remains exactly aligned with world -Z throughout approach,
            # descent, dispensing and retreat. Reachability is provided by the
            # Arm2 workstation placement, never by leaning the lance.
            nozzle_quaternion = _top_down_quaternion(0.0)
            # Do not propagate a sub-degree IK residual from one bead into the
            # next Cartesian segment.  The measured start frame already has
            # a vertical tool axis; retain its roll at the first sample and
            # let SE(3) interpolation close roll continuously on approach.
            # Replacing it outright with ``nozzle_quaternion`` creates a
            # one-sample redundant-joint branch jump even though XYZ is
            # unchanged.
            measured_x = current.rotation[:, 0]
            measured_roll = math.atan2(float(measured_x[1]), float(measured_x[0]))
            current = Pose(current.position, _top_down_quaternion(measured_roll))
            approach = Pose(
                start + np.asarray([0.0, 0.0, 0.080]),
                nozzle_quaternion,
            )
            start_pose = Pose(start, nozzle_quaternion)
            end_pose = Pose(end, nozzle_quaternion)
            retreat = Pose(
                end + np.asarray([0.0, 0.0, 0.080]),
                nozzle_quaternion,
            )
            continuous = bool(task.payload.get("continuous_from_previous", False))
            short_transition = (
                continuous
                and current.position[2] >= approach.position[2] - 0.020
                and float(np.linalg.norm(current.position[:2] - approach.position[:2])) <= 0.35
            )
            if short_transition:
                corridor = PolylineTrajectory(
                    (current, approach),
                    max(self.speed_m_s, 0.18),
                    0.01,
                )
            else:
                corridor = safe_corridor_trajectory(
                    current,
                    approach,
                    # S2B is farther from Arm2 than its normal S2A station.
                    # No fins exist yet at this process phase, so a lower
                    # certified corridor avoids the outer high-reach
                    # singularity without sacrificing fixture clearance.
                    clearance_z=(0.34 if kind is TaskType.REAPPLY_MATERIAL else 0.43),
                    speed_m_s=self.speed_m_s,
                )
            points = (*corridor.waypoints, start_pose, end_pose, retreat)
            self._material_path_ids = path_ids
            self._material_start = start.copy()
            self._material_end = end.copy()
            self._material_progress = 0.0
            self._material_reverse = reverse_travel
            for path_id in path_ids:
                registry.set_path_visible(
                    path_id,
                    False,
                    coverage=0.0,
                    reverse=reverse_travel,
                )

            def finish_material() -> None:
                for path_id in path_ids:
                    registry.set_path_visible(
                        path_id,
                        True,
                        coverage=1.0,
                        reverse=reverse_travel,
                    )
                self._material_progress = 1.0
                task.payload["application_progress"] = 1.0

            return (
                PolylineTrajectory(
                    tuple(points),
                    process_spec.material_speed,
                    0.01,
                ),
                finish_material,
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

        if kind is TaskType.POST_INSPECT:
            spec = self._product().spec
            unit_index = task.payload.get("unit_index")
            if isinstance(unit_index, int):
                product_pose = registry.batch_tray_pose(unit_index)
                local_surface_z = 0.032 + 0.5 * spec.base_thickness + spec.fin_height
            else:
                live_pose = registry.product_pose()
                requested = np.asarray(
                    task.payload.get("world_position", live_pose.position),
                    dtype=float,
                )
                product_pose = Pose(requested, live_pose.quaternion)
                local_surface_z = 0.5 * spec.base_thickness + spec.fin_height
            surface_center = product_pose.position + product_pose.rotation @ np.asarray(
                [0.0, 0.0, local_surface_z],
                dtype=float,
            )
            product_x_axis = product_pose.rotation[:, 0]
            product_yaw = math.atan2(float(product_x_axis[1]), float(product_x_axis[0]))
            scan = top_down_inspection_pose(
                surface_center,
                product_length_m=spec.base_length,
                product_width_m=spec.base_width,
                product_yaw_rad=product_yaw,
            )
            return self._framed_inspection_trajectory(current, scan), noop

        raise ValueError(f"unsupported scene task: {kind.value}")

    def _required_arm1_tool(self, task: TaskSpec) -> str:
        kind = TaskType(task.task_type)
        if kind is TaskType.LOAD_BASE:
            return "suction_tool"
        if kind in {
            TaskType.PREPARE_FIN_TOOL,
            TaskType.INSERT_FIN,
            TaskType.ADJUST_FIN,
        }:
            return "parallel_gripper"
        raise RuntimeError(f"Arm1 has no registered tool for {kind.value}")

    def _prepare_arm1_work(self, task: TaskSpec) -> None:
        """Generate the work path only after the requested tool is mounted."""

        if TaskType(task.task_type) is TaskType.PREPARE_FIN_TOOL:
            # Tool preparation runs in the background while Arm2 owns Table2.
            # The rack exit is already outside that shared zone; finish at the
            # canonical Arm1 posture so the first fin task can start directly.
            self.trajectory = None
            self._callback = lambda: None
            self._arm1_steps = [Arm1Step("joint_home", duration=1.25)]
            return

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
        return None

    def _update_material_visual(self) -> None:
        """Grow deposited beads from the measured Arm2 TCP position."""

        if (
            self.task is None
            or not self._material_path_ids
            or self._material_start is None
            or self._material_end is None
            or TaskType(self.task.task_type) not in {TaskType.APPLY_MATERIAL, TaskType.REAPPLY_MATERIAL}
        ):
            return
        actual = self.controller.current_tcp_pose().position
        axis = self._material_end - self._material_start
        length_sq = float(np.dot(axis, axis))
        if length_sq <= 1.0e-12:
            return
        raw_progress = float(np.dot(actual - self._material_start, axis) / length_sq)
        closest = self._material_start + np.clip(raw_progress, 0.0, 1.0) * axis
        distance = float(np.linalg.norm(actual - closest))
        # Ignore high approach/corridor motion. Once dispensing begins, keep
        # coverage monotonic through the final anti-drip retreat.
        if distance <= 0.025:
            self._material_progress = max(
                self._material_progress,
                float(np.clip(raw_progress, 0.0, 1.0)),
            )
        self.task.payload["application_progress"] = self._material_progress
        self.task.payload["current_path_id"] = self._material_path_ids[0]
        for path_id in self._material_path_ids:
            self.scene.registry.set_path_visible(
                path_id,
                self._material_progress > 0.0,
                coverage=self._material_progress,
                reverse=self._material_reverse,
            )

    def _prepare_arm2_work(self, task: TaskSpec) -> None:
        """Generate Arm2 work only after the selected TCP is mounted."""

        trajectory, callback = self._task_goal(task)
        kind = TaskType(task.task_type)
        if kind in {TaskType.APPLY_MATERIAL, TaskType.REAPPLY_MATERIAL}:
            points = trajectory.waypoints
            if len(points) < 5:
                raise RuntimeError("material trajectory is missing approach/dispense/retreat points")
            approach = PolylineTrajectory(
                tuple(points[:-2]),
                max(self.speed_m_s, 0.18),
                0.01,
            )
            dispense = PolylineTrajectory(
                tuple(points[-3:-1]),
                self._product().spec.material_speed,
                0.01,
            )
            retreat = PolylineTrajectory(
                tuple(points[-2:]),
                max(self.speed_m_s, 0.12),
                0.01,
            )
            steps = [
                Arm1Step("work", trajectory=approach),
                Arm1Step("work", trajectory=dispense),
                Arm1Step("work", trajectory=retreat),
            ]
            if self._should_park(task):
                if kind is TaskType.REAPPLY_MATERIAL:
                    # The retreat is already clear of the flat pre-assembly
                    # pallet. Return through the known-safe joint-space home
                    # posture instead of reaching outward to a high Cartesian
                    # corner at S2B.
                    steps.append(Arm1Step("joint_home", duration=1.25))
                    self.trajectory = trajectory
                    self._callback = callback
                    self._errors.clear()
                    self._arm2_steps = steps
                    return
                park_pose = self.park_flange_pose.transformed(self.controller.tool_transform)
                if park_pose.position[2] < 0.43:
                    position = park_pose.position.copy()
                    position[2] = 0.43
                    park_pose = Pose(position, park_pose.quaternion)
                steps.append(
                    Arm1Step(
                        "work",
                        trajectory=safe_corridor_trajectory(
                            retreat.waypoints[-1],
                            park_pose,
                            clearance_z=0.43,
                            speed_m_s=max(self.speed_m_s, 0.18),
                        ),
                    )
                )
            self.trajectory = trajectory
            self._callback = callback
            self._errors.clear()
            self._arm2_steps = steps
            return

        if self._should_park(task):
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
        self._arm2_steps = [Arm1Step("work", trajectory=trajectory)]

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
            if step.kind == "joint_home":
                step.joint_start = np.asarray(
                    self.scene.data.qpos[self.controller.qpos_ids], dtype=float
                ).copy()
                step.joint_end = self.park_joint_positions.copy()
                self._arm2_timed = (step, float(now))
                return
            raise RuntimeError(f"unknown Arm2 step: {step.kind}")

        self._callback()
        self._transport_active = False
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
        self._carried_tool_orientation_lock = None
        self._transport_active = False
        self._arm1_steps.clear()
        self._arm1_timed = None
        self._arm2_steps.clear()
        self._arm2_timed = None
        self._arm3_joint_home = None
        self._material_path_ids = ()
        self._material_start = None
        self._material_end = None
        self._material_progress = 0.0
        self._material_reverse = False
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
            if TaskType(task.task_type) is TaskType.PREPARE_FIN_TOOL:
                self._done = True
                return
        if self.arm_name == "arm2" and not self.fast:
            required = self._required_arm2_tool(task)
            if required != "brazing_dispenser":
                raise RuntimeError(f"Arm2 cannot execute {task.task_type} with its dispenser")
            self._prepare_arm2_work(task)
            self._advance_arm2_step(now)
            return
        if self.arm_name == "arm2" and self.fast:
            required = self._required_arm2_tool(task)
            if required != "brazing_dispenser":
                raise RuntimeError(f"Arm2 cannot execute {task.task_type} with its dispenser")
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
        if self._arm3_joint_home is not None:
            start, target, started_at, duration = self._arm3_joint_home
            linear = float(np.clip((float(now) - started_at) / max(duration, 1e-6), 0.0, 1.0))
            amount = quintic_time_scaling(linear)
            joints = (1.0 - amount) * start + amount * target
            self.controller.q_command = joints
            self.scene.data.qpos[self.controller.qpos_ids] = joints
            self.scene.data.qvel[self.controller.dof_ids] = 0.0
            self.scene.data.ctrl[self.controller.actuator_ids] = joints
            self.scene.sync_mounted_extensions(self.arm_name)
            if linear < 1.0:
                return ActorResult.RUNNING
            self._arm3_joint_home = None
            self._callback()
            self.controller.enabled = False
            self.controller.hold()
            self._done = True
            self.task = None
            return ActorResult.SUCCEEDED
        if self._arm2_timed is not None:
            self.scene.data.qpos[self.controller.qpos_ids] = self.controller.q_command
            self.scene.data.qvel[self.controller.dof_ids] = 0.0
            self.scene.data.ctrl[self.controller.actuator_ids] = self.controller.q_command
            self.scene.sync_mounted_extensions(self.arm_name)
            step, step_started = self._arm2_timed
            linear = float(np.clip((float(now) - step_started) / max(step.duration, 1e-6), 0.0, 1.0))
            amount = quintic_time_scaling(linear)
            if step.kind == "joint_home":
                assert step.joint_start is not None and step.joint_end is not None
                joints = (1.0 - amount) * step.joint_start + amount * step.joint_end
                self.controller.q_command = joints
                self.scene.data.qpos[self.controller.qpos_ids] = joints
                self.scene.data.qvel[self.controller.dof_ids] = 0.0
                self.scene.data.ctrl[self.controller.actuator_ids] = joints
                self.scene.sync_mounted_extensions(self.arm_name)
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
            self.scene.sync_mounted_extensions(self.arm_name)
            self._sync_carried_body()
            step, step_started = self._arm1_timed
            linear = float(np.clip((float(now) - step_started) / max(step.duration, 1e-6), 0.0, 1.0))
            amount = quintic_time_scaling(linear)
            value = step.start_value + amount * (step.end_value - step.start_value)
            if step.kind == "joint_home":
                assert step.joint_start is not None and step.joint_end is not None
                joints = (1.0 - amount) * step.joint_start + amount * step.joint_end
                self.controller.q_command = joints
                self.scene.data.qpos[self.controller.qpos_ids] = joints
                self.scene.data.qvel[self.controller.dof_ids] = 0.0
                self.scene.data.ctrl[self.controller.actuator_ids] = joints
                self.scene.sync_mounted_extensions(self.arm_name)
            elif step.kind == "gripper":
                self.scene.registry.set_arm1_gripper_closed(value)
            elif step.kind == "suction":
                self.scene.registry.set_arm1_suction_fraction(value)
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
            amount = quintic_time_scaling(linear)
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
            self.scene.sync_mounted_extensions(self.arm_name)
            self._sync_carried_body()
            self._update_material_visual()
            track_material_error = bool(
                self._kinematic_targets
                and self.arm_name == "arm2"
                and self.task is not None
                and TaskType(self.task.task_type) in {TaskType.APPLY_MATERIAL, TaskType.REAPPLY_MATERIAL}
            )
            if track_material_error:
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
            if self.arm_name == "arm3" and self.task is not None and self._should_park(self.task):
                self._arm3_joint_home = (
                    np.asarray(
                        self.scene.data.qpos[self.controller.qpos_ids],
                        dtype=float,
                    ).copy(),
                    self.park_joint_positions.copy(),
                    float(now),
                    1.25,
                )
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
        self._update_material_visual()
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
        self._carried_tool_orientation_lock = None
        self._arm1_steps.clear()
        self._arm1_timed = None
        self._arm2_steps.clear()
        self._arm2_timed = None
        self._arm3_joint_home = None
        self._material_path_ids = ()
        self._material_start = None
        self._material_end = None
        self._material_progress = 0.0
        self._material_reverse = False
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
) -> dict[str, Any]:
    actors: dict[str, Any] = {
        name: SceneTaskActor(name, scene, product, fast=fast) for name in ("arm1", "arm2", "arm3")
    }
    actors["fixture"] = FixtureTaskActor(
        scene,
        product,
        fast=fast,
        controller=scene.fixture_controller,
    )
    actors["conveyor"] = ConveyorTaskActor(scene, product, fast=fast)
    return actors


__all__ = ["SceneTaskActor", "build_scene_actors"]
