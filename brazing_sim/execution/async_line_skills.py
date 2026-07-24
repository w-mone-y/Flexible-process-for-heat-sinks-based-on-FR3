"""MuJoCo-visible skills for the multi-pallet shallow-U runtime."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np

from ..config import create_product_state
from ..motion import Pose, matrix_to_quat
from ..planning.task_models import ManufacturingTask, TaskType
from ..profiles import quintic_time_scaling
from .async_line_logistics import AsyncLineLogisticsSkill, LOGISTICS_TASK_TYPES
from .skill_registry import SkillExecutionResult, SkillRegistry, TimedSkill

TRANSFER_BINDINGS = {
    TaskType.TRANSFER_S1_S2A: ("s1_s2a", "s2a"),
    TaskType.TRANSFER_S2A_S2B: ("s2a_s2b", "s2b"),
    TaskType.TRANSFER_S2B_S3: ("s2b_s3", "s3"),
    TaskType.TRANSFER_S3_RACK: ("s3_rack", "rack_infeed"),
}

# Certified fallback redundancy branch for the low Arm1 quick-change sockets.
# Normal tool changes keep the measured branch and travel directly to the
# rack; this seed is tried only if the direct solution is genuinely blocked.
ARM1_TOOL_CHANGE_SEED = np.asarray(
    [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
    dtype=float,
)
# The attached 90 mm tool extension has a separate fallback branch for the
# rare case where the current loaded configuration cannot reach its socket.
ARM1_LOADED_TOOL_CHANGE_SEED = np.asarray(
    [2.2301, -1.2219, -1.2726, -2.8300, 1.9602, 4.4728, 0.4040],
    dtype=float,
)


def _tray_index(tray_id: str | None) -> int:
    value = str(tray_id or "").strip().lower()
    if value not in {"tray_01", "tray_02", "tray_03"}:
        raise RuntimeError(f"任务没有可见的在制托盘：{tray_id or '空'}")
    return int(value[-2:]) - 1


@dataclass(slots=True)
class _TransferState:
    transfer_id: str
    destination: str
    started_at: float
    start_position: float
    duration_s: float
    handed_off: bool = False
    settled_at: float | None = None
    return_started_at: float | None = None
    return_start_position: float | None = None
    return_duration_s: float = 0.0


@dataclass(slots=True)
class _ArmStage:
    """One smoothly interpolated, Cartesian-authored arm motion segment."""

    label: str
    target: Pose
    weight_s: float
    start_pose: Pose | None = None
    joint_goal: np.ndarray | None = None
    joint_path: np.ndarray | None = None
    duration_s: float = 0.0
    settle: bool = False
    # Endpoint dwell is a physical stability gate, not trajectory padding.
    # Contact-only motion uses it to let the payload weld converge at the
    # already aligned high pose before Z is allowed to change.
    settle_s: float = 0.0
    action: Callable[[], None] | None = None
    path_index: int | None = None
    paired_path_index: int | None = None
    path_fraction_start: float = 0.0
    path_fraction_end: float = 0.0
    reverse_path: bool = False
    lock_joint7: bool = False
    joint7_value: float | None = None
    # A fin pickup may need a different redundant wrist branch for the
    # destination slot than for the raw-material pose.  Candidate selection
    # is performed only at the first high, empty-gripper stage and validates
    # every target listed here before joint 7 is frozen for descent/carry.
    joint7_candidates: tuple[float, ...] = ()
    joint7_validation_targets: tuple[Pose, ...] = ()
    full_orientation: bool = True
    cartesian_linear: bool = False
    cartesian_start: Pose | None = None
    cartesian_sample_spacing_m: float = 0.020
    strict_vertical: bool = False
    payload_name: str | None = None
    payload_start_pose: Pose | None = None
    payload_target_pose: Pose | None = None
    position_tolerance_m: float | None = None
    orientation_tolerance_deg: float | None = None
    preferred_seed: np.ndarray | None = None
    # Optional progressive finger command.  The start value is sampled from
    # the measured finger joints when this stage is planned, so a preceding
    # partial release can flow continuously into the next opening/closing
    # motion without a control jump.
    gripper_target_fraction: float | None = None
    gripper_start_fraction: float | None = None
    # Workpiece motions prefer the current redundant branch.  Certified tool
    # change stages may explicitly request their rack seed first.
    prefer_current_seed: bool = True
    entry_start_q: np.ndarray | None = None


class AsyncLinePhysicalSkill:
    """Drive one scheduled task while preserving actual tray ownership."""

    POSITION_TOLERANCE_M = 0.0015
    VELOCITY_TOLERANCE_M_S = 0.02
    SETTLE_SECONDS = 0.10
    ACTION_FRACTION = 0.62
    CONTACT_TOLERANCE_M = 0.008
    STRICT_ENTRY_MOVE_SECONDS = 0.10
    STRICT_ENTRY_SETTLE_SECONDS = 0.12
    TRAY_PAYLOAD_ORIGIN_Z_M = 0.032
    # FR3 visual playback remains well below the hardware limits.  At the
    # default 2 ms MuJoCo step this caps one-joint movement near 0.005 rad,
    # while avoiding the excessive stage stretching that would destroy the
    # intended three-arm pipeline overlap.
    MAX_JOINT_SPEED_RAD_S = 2.50
    MAX_JOINT_ACCEL_RAD_S2 = 8.0
    QUINTIC_PEAK_VELOCITY = 1.875
    QUINTIC_PEAK_ACCELERATION = 5.7735026919
    # A 10% finger stroke opens the jaw gap from 2 mm to about 6 mm.  That is
    # sufficient to release a 2 mm fin while remaining inside the 15 mm C
    # order pitch.  Full opening is deferred until the hand is safely above
    # the raw-material approach corridor.
    FIN_RELEASE_CLOSED_FRACTION = 0.90

    def __init__(self, task_type: TaskType) -> None:
        self.task_type = TaskType(task_type)
        self.task: ManufacturingTask | None = None
        self.resource_id = ""
        self.context: Any = None
        self.started_at = 0.0
        self.duration = 0.0
        self.fast = False
        self.cancelled = False
        self.transfer: _TransferState | None = None
        self.arm_stages: list[_ArmStage] = []
        self.arm_stage_index = 0
        self.arm_stage_started_at = 0.0
        self.arm_stage_start_q: np.ndarray | None = None
        self.arm_stage_wait_started_at: float | None = None
        self.arm_stage_settled_at: float | None = None
        self.arm_stage_entry_started_at: float | None = None
        self.arm_stage_entry_pending = False
        self.arm_stage_completed_duration = 0.0
        self.arm_motion_ready = False
        self.action_applied = False

    @property
    def scene(self) -> Any:
        return self.context.scene

    @property
    def runtime(self) -> Any:
        return self.context.manufacturing_runtime

    def _plan(self) -> Any:
        assert self.task is not None
        return self.runtime.orders[self.task.order_id].plan

    def _configure_new_tray(self) -> None:
        assert self.task is not None
        index = _tray_index(self.task.tray_id)
        plan = self._plan()
        product = create_product_state(
            plan.execution_spec,
            order_id=self.task.unit_id,
            created_at=self.started_at,
        )
        self.scene.registry.configure_batch_tray(index, product)
        self.scene.registry.set_batch_tray_stage(index)
        self.scene.registry.dock_batch_tray_to_async_station(
            self.task.tray_id,
            "s1",
            snap=True,
        )

    def _station_position(self) -> np.ndarray | None:
        assert self.task is not None
        registry = self.scene.registry
        if self.task.task_type in {
            TaskType.POST_BRAZE_INSPECTION,
            TaskType.SECOND_POST_BRAZE_VIEW,
        }:
            index = _tray_index(self.task.tray_id)
            return registry.site_pose(f"batch_output_slot_{index + 1:02d}_site").position + np.asarray(
                # The finished pallet is wider than the black output belt.
                # Arm3 scans from its reachable inboard edge while the fixed
                # wrist camera still covers the complete product footprint.
                # The output lane is deliberately outside the S3 pallet
                # envelope.  Keep the camera at its certified reachable x
                # coordinate and observe from the pallet's inboard side.
                [-0.35, 0.0, 0.17]
            )
        if self.task.task_type is TaskType.PICK_BASE_PLATE:
            return registry.site_pose("raw_base_site").position + np.asarray([0.0, 0.0, 0.20])
        if self.task.task_type is TaskType.PICK_FIN:
            fin_id = str(self.task.payload.get("fin_id", "fin_01"))
            return registry.site_pose(f"raw_{fin_id}_site").position + np.asarray([0.0, 0.0, 0.16])
        if self.task.task_type is TaskType.PREPARE_FIN_TOOL:
            return registry.site_pose("arm1_parallel_gripper_rack_site").position + np.asarray(
                [0.0, 0.0, 0.18]
            )
        station_site = {
            "S1_BASE_LOADING": "s1_target_site",
            "S2A_DISPENSING": "s2a_target_site",
            "S2B_MATERIAL_INSPECTION": "s2b_target_site",
            "S3_FIN_ASSEMBLY": "s3_target_site",
            "RACK_INFEED": "rack_infeed_target_site",
        }.get(str(self.task.station_id or ""))
        if station_site is None:
            return None
        height = 0.24 if self.resource_id == "ARM1" else 0.20
        if self.resource_id == "ARM3":
            height = 0.26
        return registry.site_pose(station_site).position + np.asarray([0.0, 0.0, height])

    @staticmethod
    def _top_down_pose(position: np.ndarray, yaw: float = 0.0) -> Pose:
        cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
        rotation = np.asarray(
            [
                [cosine, sine, 0.0],
                [sine, -cosine, 0.0],
                [0.0, 0.0, -1.0],
            ]
        )
        return Pose(np.asarray(position, dtype=float), matrix_to_quat(rotation))

    @staticmethod
    def _fixed_pose(position: np.ndarray, quaternion: np.ndarray) -> Pose:
        return Pose(np.asarray(position, dtype=float), np.asarray(quaternion, dtype=float))

    def _tray_tcp_pose(self, local_position: np.ndarray) -> Pose:
        assert self.task is not None
        index = _tray_index(self.task.tray_id)
        local = Pose(
            np.asarray(local_position, dtype=float),
            np.asarray([0.0, 1.0, 0.0, 0.0]),
        )
        world = self.scene.registry.batch_to_world(index, local)
        if not isinstance(world, Pose):
            raise RuntimeError("托盘位姿转换没有返回SE(3)目标")
        return world

    def _tray_payload_pose(self, local_position: np.ndarray) -> Pose:
        """Return an identity-oriented payload pose in the live tray frame."""

        assert self.task is not None
        index = _tray_index(self.task.tray_id)
        local = Pose(
            np.asarray(local_position, dtype=float),
            np.asarray([1.0, 0.0, 0.0, 0.0]),
        )
        world = self.scene.registry.batch_to_world(index, local)
        if not isinstance(world, Pose):
            raise RuntimeError("托盘工件位姿转换没有返回SE(3)目标")
        return world

    def _tcp_for_payload_pose(self, payload_name: str, payload_target: Pose) -> Pose:
        """Compensate the measured grasp transform for a payload target.

        A suction/gripper contact is never assumed to be an ideal zero-offset
        TCP.  The live TCP-to-payload transform is measured after the grasp,
        then inverted so the *workpiece*—rather than merely the tool marker—
        reaches the authored tray pose.
        """

        controller = self.scene.arms[self.resource_id.lower()]
        current_tcp = controller.current_tcp_pose()
        current_payload = self.scene.registry.free_body_pose(payload_name)
        tcp_to_payload = current_tcp.inverse().transformed(current_payload)
        return payload_target.transformed(tcp_to_payload.inverse())

    def _base_pick_stages(self) -> list[_ArmStage]:
        assert self.task is not None
        plan = self._plan()
        registry = self.scene.registry
        raw = registry.free_body_pose("base_plate").position.copy()
        raw[2] += 0.5 * plan.execution_spec.base_thickness + 0.0015
        approach = raw + np.asarray([0.0, 0.0, 0.11])
        lift = raw + np.asarray([0.0, 0.0, 0.16])
        return [
            _ArmStage(
                "基板上方接近",
                self._top_down_pose(approach),
                1.0,
                preferred_seed=ARM1_TOOL_CHANGE_SEED,
            ),
            _ArmStage(
                "吸盘接触基板",
                self._top_down_pose(raw),
                0.55,
                settle=True,
                action=lambda: self._apply_process_action(1.0),
                cartesian_linear=True,
                preferred_seed=ARM1_TOOL_CHANGE_SEED,
            ),
            _ArmStage(
                "吸附后垂直抬升",
                self._top_down_pose(lift),
                0.75,
                lock_joint7=True,
            ),
        ]

    def _base_place_stages(self) -> list[_ArmStage]:
        if not bool(self.scene.data.eq_active[self.scene.registry.equality_id("arm1_grasp_base")]):
            raise RuntimeError("Arm1未吸附基板，禁止执行基板搬运放置")
        controller = self.scene.arms["arm1"]
        current_pose = controller.current_tcp_pose()
        current = current_pose.position.copy()
        # Preserve the measured grasp attitude during the first lift.  The
        # previous implementation replaced it with an ideal top-down
        # quaternion, which forced a wrist-branch change immediately after the
        # suction weld engaged.  Any small alignment correction is now made
        # only at the high approach pose, never near the tray.
        grasp_quaternion = current_pose.quaternion.copy()
        payload_target = self._tray_payload_pose(np.asarray([0.0, 0.0, self.TRAY_PAYLOAD_ORIGIN_Z_M]))
        payload_approach = Pose(
            payload_target.position + np.asarray([0.0, 0.0, 0.11]),
            payload_target.quaternion,
        )
        target_pose = self._tcp_for_payload_pose("base_plate", payload_target)
        approach_pose = self._tcp_for_payload_pose("base_plate", payload_approach)
        # PICK_BASE_PLATE already ends with a 160 mm vertical lift.  Raising
        # again here pushed the arm farther outside its useful envelope and
        # made C-order placement fail before any horizontal travel.  Start the
        # carry exactly at the measured hand-off pose.
        current_lift = current.copy()
        return [
            _ArmStage(
                "基板吸附稳定确认",
                self._fixed_pose(current_lift, grasp_quaternion),
                0.12,
                lock_joint7=True,
            ),
            _ArmStage(
                "基板高位平滑转运至安装位上方",
                approach_pose,
                1.25,
                # Full SE(3) keeps the base horizontal while redundancy is
                # resolved at safe height.  The selected joint-7 value is then
                # frozen for the complete vertical contact/release sequence.
                lock_joint7=False,
                settle=True,
                position_tolerance_m=0.0015,
                orientation_tolerance_deg=1.0,
            ),
            _ArmStage(
                "基板下降前高位六维预对准",
                approach_pose,
                0.30,
                lock_joint7=False,
                settle=True,
                settle_s=0.08,
                position_tolerance_m=0.0010,
                orientation_tolerance_deg=1.0,
            ),
            _ArmStage(
                "基板锁定XY姿态后纯Z下降",
                target_pose,
                0.65,
                settle=True,
                action=lambda: self._apply_process_action(1.0),
                lock_joint7=True,
                cartesian_linear=True,
                cartesian_start=approach_pose,
                cartesian_sample_spacing_m=0.0015,
                strict_vertical=True,
                payload_name="base_plate",
                payload_start_pose=payload_approach,
                payload_target_pose=payload_target,
                position_tolerance_m=0.0002,
                orientation_tolerance_deg=0.05,
            ),
            _ArmStage(
                "吸盘释放后撤离",
                approach_pose,
                0.65,
                lock_joint7=True,
            ),
        ]

    def _fin_pick_stages(self) -> list[_ArmStage]:
        assert self.task is not None
        fin_id = str(self.task.payload.get("fin_id", "fin_01"))
        raw_pose = self.scene.registry.free_body_pose(fin_id)
        raw = raw_pose.position.copy()
        approach = raw + np.asarray([0.0, 0.0, 0.13])
        lift = raw + np.asarray([0.0, 0.0, 0.18])
        pick_contact_pose = self._top_down_pose(raw)

        # Predict the installation TCP from the ideal high pickup contact.
        # This can be evaluated before the weld is enabled and therefore lets
        # Arm1 choose a redundancy branch while its gripper is still empty.
        # The real measured TCP-to-fin transform is still used after pickup
        # by _fin_place_stages; this target is only a reachability gate.
        target = np.asarray(self.task.payload.get("target_position"), dtype=float).copy()
        target[2] += self.TRAY_PAYLOAD_ORIGIN_Z_M
        payload_goal = self._tray_payload_pose(target)
        payload_approach = Pose(
            payload_goal.position + np.asarray([0.0, 0.0, 0.105]),
            payload_goal.quaternion,
        )
        predicted_tcp_to_fin = pick_contact_pose.inverse().transformed(raw_pose)
        predicted_install_approach = payload_approach.transformed(predicted_tcp_to_fin.inverse())
        predicted_install_goal = payload_goal.transformed(predicted_tcp_to_fin.inverse())

        # The useful branch for this cell lies inside this conservative range.
        # Include the current measured value explicitly and let the online
        # selector choose the nearest candidate that reaches pickup *and*
        # placement, so A/B retain their short motions while C's outer slot
        # can select the wider branch it actually needs.
        current_joint7 = float(self.scene.data.qpos[self.scene.arms["arm1"].qpos_ids[6]])
        joint7_candidates = tuple(
            dict.fromkeys(
                [
                    current_joint7,
                    *np.linspace(0.44, 1.24, 17, dtype=float).tolist(),
                ]
            )
        )
        return [
            _ArmStage(
                "翅片上方接近并渐进张开夹爪",
                self._top_down_pose(approach),
                0.85,
                preferred_seed=ARM1_TOOL_CHANGE_SEED,
                gripper_target_fraction=0.0,
                # Resolve redundancy while still high and empty.  The
                # resulting wrist-roll value is frozen before descent and is
                # retained until the fin has been released above S3.
                lock_joint7=True,
                joint7_candidates=joint7_candidates,
                joint7_validation_targets=(
                    pick_contact_pose,
                    predicted_install_approach,
                    predicted_install_goal,
                ),
            ),
            _ArmStage(
                "夹爪进入翅片夹取面",
                self._top_down_pose(raw),
                0.60,
                settle=True,
                cartesian_linear=True,
                preferred_seed=ARM1_TOOL_CHANGE_SEED,
                lock_joint7=True,
            ),
            _ArmStage(
                "夹爪两侧渐进夹紧翅片",
                self._top_down_pose(raw),
                0.42,
                settle=True,
                settle_s=0.10,
                action=lambda: self._apply_process_action(1.0),
                gripper_target_fraction=1.0,
                lock_joint7=True,
            ),
            _ArmStage(
                "夹紧后垂直抬升",
                self._top_down_pose(lift),
                0.75,
                lock_joint7=True,
            ),
        ]

    def _fin_place_stages(self) -> list[_ArmStage]:
        assert self.task is not None
        fin_id = str(self.task.payload.get("fin_id", "fin_01"))
        if not bool(self.scene.data.eq_active[self.scene.registry.equality_id(f"arm1_grasp_{fin_id}")]):
            raise RuntimeError(f"Arm1未夹紧{fin_id}，禁止执行翅片安装")
        controller = self.scene.arms["arm1"]
        current_pose = controller.current_tcp_pose()
        current = current_pose.position.copy()
        # A fin must remain rigidly seated between the two finger pads.  Use
        # the measured post-grasp attitude for every carry/place stage so no
        # relative rotation can be introduced by selecting another IK branch.
        carry_quaternion = current_pose.quaternion.copy()
        target = np.asarray(self.task.payload.get("target_position"), dtype=float)
        # Fin targets are expressed from the base centre.  The batch base is
        # centred 32 mm above the carrier frame.
        target[2] += self.TRAY_PAYLOAD_ORIGIN_Z_M
        payload_goal = self._tray_payload_pose(target)
        payload_approach = Pose(
            payload_goal.position + np.asarray([0.0, 0.0, 0.105]),
            payload_goal.quaternion,
        )
        goal_pose = self._tcp_for_payload_pose(fin_id, payload_goal)
        approach_pose = self._tcp_for_payload_pose(fin_id, payload_approach)
        # PICK_FIN already performs the required vertical clearance move.
        # Repeating a forced lift here created the conspicuous outward detour
        # before every new fin and could drive an outer raw slot to the reach
        # boundary.
        current_lift = current.copy()
        corridor = 0.5 * (current_lift + approach_pose.position)
        corridor[2] = max(float(current_lift[2]), float(approach_pose.position[2]))
        stages = [
            _ArmStage(
                "翅片夹紧稳定确认",
                self._fixed_pose(current_lift, carry_quaternion),
                0.12,
                lock_joint7=True,
            ),
            _ArmStage(
                "翅片平移安全走廊",
                self._fixed_pose(corridor, carry_quaternion),
                0.75,
                lock_joint7=True,
            ),
            _ArmStage(
                "翅片槽位正上方完成六维预对准",
                approach_pose,
                0.60,
                lock_joint7=True,
                settle=True,
                settle_s=0.08,
                # This is still 105 mm above the slot.  Accept the common
                # 3 mm servo envelope here, then let the following smooth
                # strict-entry gate close the final residual before Z moves.
                position_tolerance_m=0.0030,
                orientation_tolerance_deg=2.0,
            ),
            _ArmStage(
                "翅片锁定槽位姿态后纯Z下降",
                goal_pose,
                0.70,
                settle=True,
                settle_s=0.10,
                action=lambda: self._apply_process_action(1.0),
                lock_joint7=True,
                cartesian_linear=True,
                cartesian_start=approach_pose,
                cartesian_sample_spacing_m=0.0015,
                strict_vertical=True,
                payload_name=fin_id,
                payload_start_pose=payload_approach,
                payload_target_pose=payload_goal,
                position_tolerance_m=0.0002,
                orientation_tolerance_deg=0.05,
            ),
            _ArmStage(
                "翅片槽内稳定确认并小行程松爪",
                goal_pose,
                0.38,
                settle=True,
                settle_s=0.18,
                lock_joint7=True,
                gripper_target_fraction=self.FIN_RELEASE_CLOSED_FRACTION,
                # Releasing the payload changes the wrist load and can leave
                # a harmless millimetre-scale servo residual.  The preceding
                # loaded pure-Z placement already enforces 0.2 mm / 0.05 deg;
                # this empty-gripper dwell only confirms that the fin remains
                # in its slot, so it must use the controller's common 3 mm
                # arrival threshold rather than create a no-progress deadlock.
                position_tolerance_m=0.0030,
                orientation_tolerance_deg=1.0,
            ),
            _ArmStage(
                "夹爪松开并垂直撤离",
                approach_pose,
                0.65,
                lock_joint7=True,
                cartesian_linear=True,
                cartesian_sample_spacing_m=0.0015,
                # The fin has already transferred to its fixture constraint.
                # This is an empty-gripper retreat, so it starts from the live
                # measured TCP and must not repeat payload-level strict-Z
                # validation or trigger a false reinstall recovery.
                position_tolerance_m=0.0010,
                orientation_tolerance_deg=0.20,
            ),
        ]
        fin_number = int(fin_id.rsplit("_", 1)[-1])
        if fin_number >= self._plan().execution_spec.fin_count:
            # Arm3 owns the S3 inspection volume immediately after the last
            # insertion.  Merely lifting above the slot leaves Arm1 link 5 in
            # that swept volume, so finish at a certified Arm1-side wait point.
            stages.append(
                _ArmStage(
                    "末片完成后退出检测区",
                    self._top_down_pose(np.asarray([0.0, -0.28, 0.55])),
                    0.75,
                )
            )
        return stages

    def _dispense_stages(self) -> list[_ArmStage]:
        plan = self._plan()
        stages: list[_ArmStage] = []
        base_z = 0.032 + 0.5 * plan.execution_spec.base_thickness
        tip_z = base_z + float(plan.product.nozzle_tip_height_m)
        grouped_paths: list[list[tuple[int, Any]]] = []
        for path_index, path in enumerate(plan.brazing_paths):
            if not grouped_paths or grouped_paths[-1][0][1].fin_id != path.fin_id:
                grouped_paths.append([])
            grouped_paths[-1].append((path_index, path))
        for pass_index, group in enumerate(grouped_paths):
            first_index, first_path = group[0]
            second_index = group[1][0] if len(group) > 1 else None
            reverse = bool(pass_index % 2)
            starts = np.asarray(
                [path.end if reverse else path.start for _, path in group],
                dtype=float,
            )
            ends = np.asarray(
                [path.start if reverse else path.end for _, path in group],
                dtype=float,
            )
            start = np.mean(starts, axis=0)
            end = np.mean(ends, axis=0)
            start[2] = tip_z
            end[2] = tip_z
            hover = start.copy()
            hover[2] += 0.045 if stages else 0.11
            stages.append(
                _ArmStage(
                    f"翅片{pass_index + 1:02d}双喷嘴接近",
                    self._tray_tcp_pose(hover),
                    0.18,
                )
            )
            stages.append(
                _ArmStage(
                    f"翅片{pass_index + 1:02d}双喷嘴接触",
                    self._tray_tcp_pose(start),
                    0.12,
                    settle=True,
                )
            )
            length = max(1.0e-9, float(np.linalg.norm(end - start)))
            start_pose = self._tray_tcp_pose(start)
            end_pose = self._tray_tcp_pose(end)
            stages.append(
                _ArmStage(
                    f"翅片{pass_index + 1:02d}左右路径连续同步涂覆",
                    end_pose,
                    max(0.10, length / plan.product.material_speed_m_s),
                    path_index=first_index,
                    paired_path_index=second_index,
                    path_fraction_start=0.0,
                    path_fraction_end=1.0,
                    reverse_path=reverse,
                    cartesian_linear=True,
                    cartesian_start=start_pose,
                    cartesian_sample_spacing_m=0.003,
                    orientation_tolerance_deg=0.20,
                    action=lambda indices=tuple(
                        index for index in (first_index, second_index) if index is not None
                    ): self._finalize_dispense_paths(indices),
                )
            )
            lift = end.copy()
            lift[2] += 0.035
            stages.append(
                _ArmStage(
                    f"翅片{pass_index + 1:02d}完成后抬枪",
                    self._tray_tcp_pose(lift),
                    0.12,
                )
            )
        if stages:
            final = stages[-1].target.position.copy()
            final[2] += 0.10
            stages.append(_ArmStage("喷枪安全撤离", self._top_down_pose(final), 0.35))
        return stages

    def _arm1_tool_change_stages(self, required: str) -> list[_ArmStage]:
        """Build one continuous, physically visible Arm1 tool-change chain."""

        manager = self.scene.arm1_tools
        controller = self.scene.arms["arm1"]
        if manager.current_tool == required:
            return []

        stages: list[_ArmStage] = []
        current_tool = manager.current_tool
        current_pose = controller.current_tcp_pose()
        safe_position = current_pose.position.copy()
        safe_position[2] = max(float(safe_position[2]), 0.46)
        stages.append(
            _ArmStage(
                "Arm1换刀前平滑抬升",
                self._fixed_pose(safe_position, current_pose.quaternion),
                0.45,
            )
        )

        if current_tool is not None:
            hover, dock, _ = manager.change_poses(current_tool, hover_m=0.10)
            hover_tcp = manager.tcp_for_flange(hover, current_tool)
            dock_tcp = manager.tcp_for_flange(dock, current_tool)
            stages.extend(
                [
                    _ArmStage(
                        "Arm1携当前工具进入换刀走廊",
                        hover_tcp,
                        0.70,
                        preferred_seed=ARM1_LOADED_TOOL_CHANGE_SEED,
                    ),
                    _ArmStage(
                        f"Arm1低速归还{current_tool}",
                        dock_tcp,
                        0.42,
                        settle=True,
                        action=lambda name=current_tool: manager.undock(name),
                        preferred_seed=ARM1_LOADED_TOOL_CHANGE_SEED,
                    ),
                    _ArmStage("Arm1空法兰平滑退出旧工具", hover, 0.42),
                ]
            )
            current_pose = hover
        else:
            current_pose = self._fixed_pose(safe_position, current_pose.quaternion)

        hover, dock, _ = manager.change_poses(required, hover_m=0.10)
        stages.extend(
            [
                _ArmStage(
                    "Arm1空法兰横移至目标工具",
                    hover,
                    0.65,
                    preferred_seed=ARM1_TOOL_CHANGE_SEED,
                ),
                _ArmStage(
                    f"Arm1低速插接{required}",
                    dock,
                    0.42,
                    settle=True,
                    action=lambda name=required: manager.dock(name),
                    preferred_seed=ARM1_TOOL_CHANGE_SEED,
                ),
                _ArmStage(
                    "Arm1带新工具平滑退出工具架",
                    manager.tcp_for_flange(hover, required),
                    0.48,
                ),
            ]
        )
        return stages

    def _finalize_dispense_paths(self, path_indices: tuple[int, ...]) -> None:
        """Commit one completed dual-nozzle pass to exact common endpoints."""

        assert self.task is not None
        tray_index = _tray_index(self.task.tray_id)
        for path_index in path_indices:
            self.scene.registry.set_batch_brazing_path_progress(
                tray_index,
                path_index,
                1.0,
                reverse=False,
            )

    def _critical_arm_stages(self) -> list[_ArmStage]:
        assert self.task is not None
        if self.task.task_type is TaskType.PICK_BASE_PLATE:
            return self._base_pick_stages()
        if self.task.task_type is TaskType.PLACE_BASE_PLATE:
            return self._base_place_stages()
        if self.task.task_type is TaskType.PICK_FIN:
            return self._fin_pick_stages()
        if self.task.task_type is TaskType.INSTALL_FIN:
            return self._fin_place_stages()
        if self.task.task_type is TaskType.DISPENSE_BRAZING:
            return self._dispense_stages()
        return []

    def _prepare_arm_stages(
        self,
        controller: Any,
        stages: list[_ArmStage],
        *,
        duration_budget_s: float | None = None,
    ) -> None:
        """Queue stages for budgeted IK planning outside the start callback."""

        if not stages:
            return
        total_weight = sum(max(1.0e-6, stage.weight_s) for stage in stages)
        duration_budget = self.duration if duration_budget_s is None else float(duration_budget_s)
        for stage in stages:
            stage.duration_s = max(
                0.02,
                duration_budget * max(1.0e-6, stage.weight_s) / total_weight,
            )
        self.arm_stages = stages
        self.arm_stage_index = 0
        self.arm_stage_started_at = self.started_at
        self.arm_stage_start_q = None
        self.arm_stage_wait_started_at = None
        self.arm_stage_settled_at = None
        self.arm_stage_entry_started_at = None
        self.arm_stage_entry_pending = False
        self.arm_stage_completed_duration = 0.0
        initial_command = np.asarray(self.scene.data.qpos[controller.qpos_ids], dtype=float).copy()
        self.arm_motion_ready = False
        controller.enabled = False
        controller.q_command = initial_command

    def _enforce_smooth_stage_duration(self, stage: _ArmStage) -> None:
        """Bound a quintic stage by its measured joint travel.

        Task estimates are useful to the scheduler but are not safe motion
        durations: a short approach weight can otherwise compress several
        radians of a redundant IK branch into one or two 50 Hz updates.  Use
        the analytical velocity/acceleration peaks of the shared quintic
        profile to derive a physical lower bound after IK is known.
        """

        if self.fast or self.arm_stage_start_q is None or stage.joint_goal is None:
            return
        if stage.joint_path is None:
            joint_travel = np.abs(stage.joint_goal - self.arm_stage_start_q)
        else:
            joint_travel = np.sum(np.abs(np.diff(stage.joint_path, axis=0)), axis=0)
        maximum_travel = float(np.max(joint_travel, initial=0.0))
        if maximum_travel <= 1.0e-9:
            return
        velocity_duration = self.QUINTIC_PEAK_VELOCITY * maximum_travel / self.MAX_JOINT_SPEED_RAD_S
        acceleration_duration = math.sqrt(
            self.QUINTIC_PEAK_ACCELERATION * maximum_travel / self.MAX_JOINT_ACCEL_RAD_S2
        )
        stage.duration_s = max(stage.duration_s, velocity_duration, acceleration_duration)

    def _plan_one_arm_stage(self, now: float) -> tuple[bool, float]:
        """Validate only the next stage against the current physical state.

        Planning the whole task up front was both wasteful and incorrect for
        tool changes: later TCP transforms did not yet exist, and cached IK
        answers could come from a different redundant-joint branch.  Online
        stage validation keeps one continuous seed and never blocks the UI
        with dozens of solves in one tick.
        """

        if self.arm_motion_ready:
            return True, 1.0
        controller = self.scene.arms[self.resource_id.lower()]
        stage = self.arm_stages[self.arm_stage_index]
        seed = np.asarray(self.scene.data.qpos[controller.qpos_ids], dtype=float).copy()
        stage.start_pose = controller.current_tcp_pose()
        if (
            stage.strict_vertical
            and stage.payload_name is not None
            and stage.payload_start_pose is not None
            and stage.payload_target_pose is not None
        ):
            # Re-measure after the high transfer, not at task construction.
            # Constraint compliance can alter the effective grasp transform
            # by a fraction of a millimetre during the carry; rebasing here
            # makes the payload itself exact at both ends of the Z segment.
            current_payload = self.scene.registry.free_body_pose(stage.payload_name)
            live_tcp_to_payload = stage.start_pose.inverse().transformed(current_payload)
            stage.cartesian_start = stage.payload_start_pose.transformed(live_tcp_to_payload.inverse())
            stage.target = stage.payload_target_pose.transformed(live_tcp_to_payload.inverse())
        if stage.joint7_candidates and stage.joint7_value is None:
            stage.joint7_value = self._select_fin_joint7_branch(controller, stage, seed)
        lock_value = float(seed[6]) if stage.joint7_value is None else float(stage.joint7_value)
        locks = {6: lock_value} if stage.lock_joint7 else None
        candidate_seeds = []
        preferred = (
            None if stage.preferred_seed is None else np.asarray(stage.preferred_seed, dtype=float).copy()
        )
        if stage.prefer_current_seed:
            candidate_seeds.append(seed)
            if preferred is not None:
                candidate_seeds.append(preferred)
        else:
            if preferred is not None:
                candidate_seeds.append(preferred)
            candidate_seeds.append(seed)
        midpoint_seed = controller.mid.copy()
        neutral_seed = np.asarray([0.0, -0.8, 0.0, -2.2, 0.0, 1.8, seed[6]], dtype=float)
        if locks:
            midpoint_seed[6] = lock_value
            neutral_seed[6] = lock_value
        candidate_seeds.extend((midpoint_seed, neutral_seed))

        def solve_target(
            target: Pose,
            seeds: list[np.ndarray],
            *,
            position_tolerance_m: float | None = None,
        ) -> Any:
            solved = None
            for candidate_seed in seeds:
                solved = controller.solve_ik(
                    target,
                    tcp=True,
                    seed=candidate_seed,
                    max_iterations=800,
                    step_s=0.020,
                    locked_joints=locks,
                    position_tolerance_m=(
                        position_tolerance_m
                        if position_tolerance_m is not None
                        else (
                            stage.position_tolerance_m
                            if stage.position_tolerance_m is not None
                            else 0.003 if stage.settle else self.CONTACT_TOLERANCE_M
                        )
                    ),
                    orientation_tolerance_rad=math.radians(
                        stage.orientation_tolerance_deg
                        if stage.orientation_tolerance_deg is not None
                        else 0.2 if self.resource_id == "ARM2" else 3.0
                    ),
                    full_orientation=stage.full_orientation,
                )
                if solved.reachable:
                    return solved
            return solved

        if stage.cartesian_start is not None:
            canonical_start = stage.cartesian_start
            if stage.strict_vertical:
                xy_error = float(np.linalg.norm(canonical_start.position[:2] - stage.target.position[:2]))
                rotation_error = math.acos(
                    float(
                        np.clip(
                            0.5 * (np.trace(canonical_start.rotation.T @ stage.target.rotation) - 1.0),
                            -1.0,
                            1.0,
                        )
                    )
                )
                # Both poses are independently rebuilt from the same live
                # TCP-to-payload transform.  MuJoCo/quaternion round-off can
                # leave nanometre / microradian differences; reject actual
                # lateral/roll motion, not numerical noise.
                if xy_error > 1.0e-7 or rotation_error > 1.0e-6:
                    raise RuntimeError(f"{stage.label}不是纯Z轨迹，禁止开始下降")
            start_result = solve_target(
                canonical_start,
                candidate_seeds,
                position_tolerance_m=stage.position_tolerance_m or 0.001,
            )
            if start_result is None or not start_result.reachable:
                raise RuntimeError(f"{stage.label}的高位对准姿态不可达")
            # Re-solve the exact high endpoint, but enter that equivalent
            # redundant configuration through the smooth entry gate below.
            # A direct qpos write here was the historical one-frame wrist
            # twitch; omitting the exact endpoint caused payload compliance
            # to drift laterally during the descent.
            stage.entry_start_q = seed.copy()
            seed = start_result.joint_positions.copy()
            stage.start_pose = canonical_start
            candidate_seeds = [seed]

        result = solve_target(stage.target, candidate_seeds)
        assert result is not None
        if not result.reachable:
            raise RuntimeError(
                f"{self.task.task_id if self.task else self.task_type.value}: "
                f"{stage.label}不可达 "
                f"({result.position_error_m * 1000.0:.2f} mm/"
                f"{math.degrees(result.orientation_error_rad):.2f} deg)"
            )
        stage.joint_goal = result.joint_positions.copy()
        stage.joint_path = None
        if stage.cartesian_linear:
            distance = float(np.linalg.norm(stage.target.position - stage.start_pose.position))
            sample_count = max(
                2,
                int(math.ceil(distance / max(0.0005, stage.cartesian_sample_spacing_m))),
            )
            path = [seed.copy()]
            path_seed = seed.copy()
            for sample_index in range(1, sample_count + 1):
                sample_fraction = sample_index / sample_count
                if stage.strict_vertical:
                    sample_position = stage.start_pose.position.copy()
                    sample_position[2] = (1.0 - sample_fraction) * stage.start_pose.position[
                        2
                    ] + sample_fraction * stage.target.position[2]
                    sample_pose = Pose(sample_position, stage.start_pose.quaternion)
                else:
                    sample_pose = stage.start_pose.interpolate(stage.target, sample_fraction)
                final_sample_tolerance = (
                    stage.position_tolerance_m
                    if stage.position_tolerance_m is not None
                    else 0.003 if stage.settle else self.CONTACT_TOLERANCE_M
                )
                sample_result = solve_target(
                    sample_pose,
                    [path_seed],
                    position_tolerance_m=(
                        final_sample_tolerance
                        if sample_index == sample_count
                        else (
                            stage.position_tolerance_m
                            if stage.position_tolerance_m is not None
                            else 0.002 if stage.strict_vertical else 0.006
                        )
                    ),
                )
                if sample_result is None or not sample_result.reachable:
                    raise RuntimeError(
                        f"{self.task.task_id if self.task else self.task_type.value}: "
                        f"{stage.label}直线路径第{sample_index}/{sample_count}段不可达"
                    )
                path_seed = sample_result.joint_positions.copy()
                path.append(path_seed)
            stage.joint_path = np.stack(path)
            stage.joint_goal = stage.joint_path[-1].copy()
        self.arm_stage_start_q = seed
        if stage.gripper_target_fraction is not None:
            stage.gripper_start_fraction = self.scene.registry.arm1_gripper_closed_fraction()
        self._enforce_smooth_stage_duration(stage)
        self.arm_stage_started_at = float(now)
        self.arm_stage_entry_pending = bool(stage.strict_vertical)
        self.arm_stage_entry_started_at = float(now) if stage.strict_vertical else None
        controller.locked_local_indices = (6,) if stage.lock_joint7 else ()
        controller.full_orientation = stage.full_orientation
        self.arm_motion_ready = True
        return True, 1.0

    def _select_fin_joint7_branch(
        self,
        controller: Any,
        stage: _ArmStage,
        seed: np.ndarray,
    ) -> float:
        """Choose a wrist branch before grasping that reaches both endpoints.

        The fin remains orientation-locked after the gripper closes, so a
        branch that is good only for Table1 cannot be repaired later without
        violating the payload-stability rule.  Search a small deterministic
        set in nearest-first order and validate pickup contact plus both S3
        high/low poses with the parallel-gripper TCP already mounted.
        """

        lower = float(controller.lower[6]) + 1.0e-4
        upper = float(controller.upper[6]) - 1.0e-4
        values = sorted(
            {float(np.clip(value, lower, upper)) for value in stage.joint7_candidates},
            key=lambda value: (abs(value - float(seed[6])), value),
        )
        targets = (stage.target, *stage.joint7_validation_targets)
        best_residual: tuple[float, float] | None = None
        for value in values:
            locks = {6: value}
            branch_seed = seed.copy()
            branch_seed[6] = value
            certified_bare_seed = ARM1_TOOL_CHANGE_SEED.copy()
            certified_bare_seed[6] = value
            certified_loaded_seed = ARM1_LOADED_TOOL_CHANGE_SEED.copy()
            certified_loaded_seed[6] = value
            midpoint_seed = controller.mid.copy()
            midpoint_seed[6] = value
            neutral_seed = np.asarray(
                [0.0, -0.8, 0.0, -2.2, 0.0, 1.8, value],
                dtype=float,
            )
            branch_ok = True
            branch_residual = (0.0, 0.0)
            pickup_solution: np.ndarray | None = None
            for target_index, target in enumerate(targets):
                # The last validation pose is the seated slot pose and must
                # satisfy the same exact tolerance used by the later pure-Z
                # placement stage.  A loose high-pose-only check can accept a
                # branch that gets within ~1.5 mm but can never seat the fin.
                seated_target = target_index == len(targets) - 1
                solved = None
                for candidate_seed in (
                    branch_seed,
                    certified_bare_seed,
                    certified_loaded_seed,
                    midpoint_seed,
                    neutral_seed,
                ):
                    solved = controller.solve_ik(
                        target,
                        tcp=True,
                        seed=candidate_seed,
                        max_iterations=900 if seated_target else 650,
                        step_s=0.018 if seated_target else 0.022,
                        locked_joints=locks,
                        position_tolerance_m=0.0002 if seated_target else 0.0015,
                        orientation_tolerance_rad=math.radians(0.05 if seated_target else 1.0),
                        full_orientation=True,
                    )
                    if solved.reachable:
                        branch_seed = solved.joint_positions.copy()
                        if target_index == 0:
                            pickup_solution = branch_seed.copy()
                        break
                assert solved is not None
                branch_residual = (
                    max(branch_residual[0], float(solved.position_error_m)),
                    max(branch_residual[1], float(solved.orientation_error_rad)),
                )
                if not solved.reachable:
                    branch_ok = False
                    break
            if branch_ok:
                # Feed the already validated pickup solution into the normal
                # stage planner as a fallback.  The measured current branch
                # remains first choice, so this affects only IK robustness
                # after the physical direct tool change; it never changes the
                # visible rack trajectory itself.
                if pickup_solution is not None:
                    stage.preferred_seed = pickup_solution
                    stage.prefer_current_seed = True
                return value
            if best_residual is None or branch_residual < best_residual:
                best_residual = branch_residual
        position_error, orientation_error = best_residual or (float("inf"), float("inf"))
        raise RuntimeError(
            f"{self.task.task_id if self.task else self.task_type.value}: "
            "夹取前没有同时覆盖原料位和安装槽的第七关节分支 "
            f"(最佳残差 {position_error * 1000.0:.2f} mm/"
            f"{math.degrees(orientation_error):.2f} deg)"
        )

    def _prepare_arm_motion(self) -> None:
        if self.resource_id not in {"ARM1", "ARM2", "ARM3"}:
            return
        assert self.task is not None
        controller = self.scene.arms[self.resource_id.lower()]
        controller.locked_local_indices = ()
        controller.full_orientation = False
        detailed_stages: list[_ArmStage] = []
        if self.resource_id == "ARM1":
            if self.task.task_type is TaskType.PICK_BASE_PLATE:
                self.context.prepare_v2_raw_kit(self.task.order_id, self.task.unit_id)
                detailed_stages.extend(self._arm1_tool_change_stages("suction_tool"))
            elif self.task.task_type is TaskType.PICK_FIN:
                self.context.prepare_v2_raw_kit(self.task.order_id, self.task.unit_id)
                detailed_stages.extend(self._arm1_tool_change_stages("parallel_gripper"))
            elif self.task.task_type is TaskType.INSTALL_FIN:
                if self.scene.arm1_tools.current_tool != "parallel_gripper":
                    raise RuntimeError("Arm1夹持翅片时禁止切换工具")
            elif self.task.task_type is TaskType.PREPARE_FIN_TOOL:
                if self.scene.arm1_tools.current_tool == "parallel_gripper":
                    # Multiple admitted orders each own a logical preparation
                    # milestone, but one physical gripper is shared.  After
                    # the first visible switch, following milestones are
                    # idempotent confirmations—not another rack approach.
                    self.duration = 0.10
                    return
                detailed_stages.extend(self._arm1_tool_change_stages("parallel_gripper"))
        elif self.resource_id == "ARM2":
            self.scene.registry.configure_dispenser(self._plan().execution_spec)
            if self.scene.tools.current_tool != "brazing_dispenser":
                raise RuntimeError("Arm2固定双喷嘴未安装")
        detailed_stages.extend(self._critical_arm_stages())
        if detailed_stages:
            self._prepare_arm_stages(controller, detailed_stages)
            return
        position = self._station_position()
        if position is None:
            return
        # All robot motions now use the same validated quintic stage engine.
        # The retired fallback wrote only a servo command, reset its progress
        # halfway through the task and returned on a separate timing branch;
        # that was the remaining source of inspection jitter and inconsistent
        # completion timing between full orders and segmented demos.
        entry = controller.current_tcp_pose()
        target = self._fixed_pose(np.asarray(position, dtype=float), entry.quaternion)
        post_braze_view = self.task.task_type in {
            TaskType.POST_BRAZE_INSPECTION,
            TaskType.SECOND_POST_BRAZE_VIEW,
        }
        self._prepare_arm_stages(
            controller,
            [
                _ArmStage(
                    f"{self.task.task_type.value}到达工位",
                    target,
                    0.58,
                    settle=True,
                    full_orientation=not post_braze_view,
                    action=lambda: self._apply_process_action(1.0),
                ),
                _ArmStage(
                    f"{self.task.task_type.value}平滑撤离",
                    entry,
                    0.42,
                    full_orientation=not post_braze_view,
                ),
            ],
        )

    def _complete_raw_material_action(self) -> None:
        """Attach a real blank to Arm1, then retire it after tray placement."""

        assert self.task is not None
        task_type = self.task.task_type
        consumed = self.context.v2_consumed_materials.setdefault(self.task.unit_id, set())
        if task_type is TaskType.PICK_BASE_PLATE:
            self.scene.registry.set_arm1_suction_fraction(1.0)
            self.scene.registry.grasp_base(True)
        elif task_type is TaskType.PLACE_BASE_PLATE:
            consumed.add("base_plate")
            self.scene.registry.set_async_raw_item_visible("base_plate", False)
            self.scene.registry.set_arm1_suction_fraction(0.0)
        elif task_type is TaskType.PICK_FIN:
            fin_id = str(self.task.payload.get("fin_id", "fin_01"))
            self.scene.registry.set_arm1_gripper_closed(1.0)
            self.scene.registry.seat_and_grasp_fin(fin_id)
        elif task_type is TaskType.INSTALL_FIN:
            fin_id = str(self.task.payload.get("fin_id", "fin_01"))
            consumed.add(fin_id)
            self.scene.registry.set_async_raw_item_visible(fin_id, False)

    def start(
        self,
        task: ManufacturingTask,
        resource_id: str,
        context: Any,
        now: float,
    ) -> None:
        if self.task is not None:
            raise RuntimeError(f"{self.task_type.value}物理技能仍在运行")
        self.task = task
        self.resource_id = str(resource_id).upper()
        self.context = context
        self.started_at = float(now)
        self.fast = bool(getattr(getattr(context, "args", None), "fast", False))
        self.duration = 0.10 if self.fast else max(0.10, float(task.estimated_duration))
        self.cancelled = False
        self.transfer = None
        self.arm_stages.clear()
        self.arm_stage_index = 0
        self.arm_stage_started_at = float(now)
        self.arm_stage_start_q = None
        self.arm_stage_wait_started_at = None
        self.arm_stage_settled_at = None
        self.arm_stage_entry_started_at = None
        self.arm_stage_entry_pending = False
        self.arm_stage_completed_duration = 0.0
        self.arm_motion_ready = False
        self.action_applied = False
        try:
            if task.task_type is TaskType.INDEX_EMPTY_TRAY:
                self._configure_new_tray()
            if task.task_type in TRANSFER_BINDINGS:
                transfer_id, destination = TRANSFER_BINDINGS[task.task_type]
                self.scene.registry.begin_batch_tray_async_transfer(task.tray_id or "", transfer_id)
                start_position = self.scene.registry.async_transfer_position(transfer_id)
                outbound_duration = 0.05 if self.fast else max(4.0, float(task.estimated_duration) * 2.25)
                return_duration = 0.05 if self.fast else max(2.5, float(task.estimated_duration) * 1.25)
                self.transfer = _TransferState(
                    transfer_id=transfer_id,
                    destination=destination,
                    started_at=float(now),
                    start_position=float(start_position),
                    duration_s=outbound_duration,
                    return_duration_s=return_duration,
                )
            self._prepare_arm_motion()
        except Exception:
            # Starting a skill is transactional.  A failed IK/tool/ownership
            # preparation must not leave this per-task-type skill occupied,
            # otherwise every following scheduler tick reports the misleading
            # "skill is still running" error and the pallet never progresses.
            if self.transfer is not None:
                self.scene.registry.set_async_transfer_target(
                    self.transfer.transfer_id,
                    self.scene.registry.async_transfer_position(self.transfer.transfer_id),
                )
            self.task = None
            self.transfer = None
            self.arm_stages.clear()
            self.arm_stage_start_q = None
            raise

    def _task_fraction(self, now: float) -> float:
        return min(1.0, max(0.0, (float(now) - self.started_at) / self.duration))

    def _stage_progress(self, now: float, stage: _ArmStage) -> float:
        return min(
            1.0,
            max(
                0.0,
                (float(now) - self.arm_stage_started_at) / max(1.0e-9, stage.duration_s),
            ),
        )

    def _update_staged_arm(self, now: float) -> tuple[bool, float]:
        """Advance one validated joint segment with jerk-free time scaling."""

        if not self.arm_stages:
            return True, 1.0
        controller = self.scene.arms[self.resource_id.lower()]
        if self.arm_stage_index >= len(self.arm_stages):
            return True, 1.0
        stage = self.arm_stages[self.arm_stage_index]
        if stage.start_pose is None or stage.joint_goal is None or self.arm_stage_start_q is None:
            return False, 0.0
        if self.arm_stage_entry_pending:
            # Keep the canonical high endpoint stationary through several
            # MuJoCo constraint solves.  Any sub-millimetre correction from
            # the preceding SE(3) segment therefore finishes before the
            # descent clock starts, rather than appearing as XY drift near
            # the workpiece.
            entry_started = (
                float(now) if self.arm_stage_entry_started_at is None else self.arm_stage_entry_started_at
            )
            entry_fraction = float(
                np.clip(
                    (float(now) - entry_started) / self.STRICT_ENTRY_MOVE_SECONDS,
                    0.0,
                    1.0,
                )
            )
            entry_source = self.arm_stage_start_q if stage.entry_start_q is None else stage.entry_start_q
            command = entry_source + quintic_time_scaling(entry_fraction) * (
                self.arm_stage_start_q - entry_source
            )
            command = np.clip(command, controller.lower, controller.upper)
            controller.q_command = command
            self.scene.data.qpos[controller.qpos_ids] = command
            self.scene.data.qvel[controller.dof_ids] = 0.0
            self.scene.data.ctrl[controller.actuator_ids] = command
            controller.locked_local_indices = (6,) if stage.lock_joint7 else ()
            controller.full_orientation = stage.full_orientation
            controller.enabled = False
            self.scene.sync_mounted_extensions(self.resource_id.lower())
            if float(now) - entry_started < self.STRICT_ENTRY_MOVE_SECONDS + self.STRICT_ENTRY_SETTLE_SECONDS:
                return False, 0.0
            if stage.payload_name is not None and stage.payload_start_pose is not None:
                # Seat the already welded payload at the exact high authored
                # pose after the smooth redundant-joint correction has
                # settled.  The correction is sub-millimetre and occurs
                # 105+ mm above the product; recomputing the weld here removes
                # constraint compliance from the following pure-Z path.
                self.scene.registry.set_free_body_pose(
                    stage.payload_name,
                    stage.payload_start_pose,
                    forward=True,
                )
                weld_name = (
                    "arm1_grasp_base"
                    if stage.payload_name == "base_plate"
                    else f"arm1_grasp_{stage.payload_name}"
                )
                tool_body = (
                    "arm1_suction_tool" if stage.payload_name == "base_plate" else "arm1_parallel_gripper"
                )
                self.scene.registry.set_weld(
                    weld_name,
                    True,
                    recompute=(tool_body, stage.payload_name),
                    forward=True,
                )
            self.arm_stage_entry_pending = False
            self.arm_stage_entry_started_at = None
            self.arm_stage_started_at = float(now)
            return False, 0.0
        fraction = self._stage_progress(now, stage)
        command_fraction = quintic_time_scaling(fraction)
        if stage.gripper_target_fraction is not None:
            finger_start = (
                self.scene.registry.arm1_gripper_closed_fraction()
                if stage.gripper_start_fraction is None
                else stage.gripper_start_fraction
            )
            finger_command = finger_start + command_fraction * (stage.gripper_target_fraction - finger_start)
            self.scene.registry.set_arm1_gripper_closed(finger_command)
        if stage.joint_path is not None:
            scaled = command_fraction * (len(stage.joint_path) - 1)
            left_index = min(int(math.floor(scaled)), len(stage.joint_path) - 1)
            right_index = min(left_index + 1, len(stage.joint_path) - 1)
            local_fraction = scaled - left_index
            command = stage.joint_path[left_index] + local_fraction * (
                stage.joint_path[right_index] - stage.joint_path[left_index]
            )
        else:
            command = self.arm_stage_start_q + command_fraction * (stage.joint_goal - self.arm_stage_start_q)
        command = np.clip(command, controller.lower, controller.upper)
        controller.q_command = command
        self.scene.data.qpos[controller.qpos_ids] = command
        self.scene.data.qvel[controller.dof_ids] = 0.0
        self.scene.data.ctrl[controller.actuator_ids] = command
        controller.locked_local_indices = (6,) if stage.lock_joint7 else ()
        controller.full_orientation = stage.full_orientation
        controller.enabled = False
        self.scene.sync_mounted_extensions(self.resource_id.lower())

        if stage.path_index is not None and self.task is not None:
            segment = stage.target.position - stage.start_pose.position
            segment_length_sq = float(np.dot(segment, segment))
            if segment_length_sq <= 1.0e-12:
                actual_fraction = fraction
            else:
                actual = controller.current_tcp_pose().position
                actual_fraction = float(
                    np.clip(
                        np.dot(actual - stage.start_pose.position, segment) / segment_length_sq,
                        0.0,
                        1.0,
                    )
                )
            path_fraction = stage.path_fraction_start + actual_fraction * (
                stage.path_fraction_end - stage.path_fraction_start
            )
            self.scene.registry.set_batch_brazing_path_progress(
                _tray_index(self.task.tray_id),
                stage.path_index,
                path_fraction,
                reverse=stage.reverse_path,
            )
            if stage.paired_path_index is not None:
                self.scene.registry.set_batch_brazing_path_progress(
                    _tray_index(self.task.tray_id),
                    stage.paired_path_index,
                    path_fraction,
                    reverse=stage.reverse_path,
                )

        total_duration = sum(candidate.duration_s for candidate in self.arm_stages)
        progress = (self.arm_stage_completed_duration + fraction * stage.duration_s) / max(
            total_duration, 1.0e-9
        )
        if fraction < 1.0:
            return False, min(0.999, progress)

        actual_pose = controller.current_tcp_pose()
        actual_error = float(np.linalg.norm(actual_pose.position - stage.target.position))
        relative_rotation = actual_pose.rotation.T @ stage.target.rotation
        orientation_error = math.acos(float(np.clip(0.5 * (np.trace(relative_rotation) - 1.0), -1.0, 1.0)))
        position_tolerance = (
            stage.position_tolerance_m
            if stage.position_tolerance_m is not None
            else 0.003 if stage.settle else self.CONTACT_TOLERANCE_M
        )
        orientation_tolerance = math.radians(
            stage.orientation_tolerance_deg
            if stage.orientation_tolerance_deg is not None
            else 0.3 if self.resource_id == "ARM2" else 5.0
        )
        if actual_error > position_tolerance or orientation_error > orientation_tolerance:
            if self.arm_stage_wait_started_at is None:
                self.arm_stage_wait_started_at = float(now)
            self.arm_stage_settled_at = None
            # Keep the exact validated joint endpoint while the payload/tool
            # weld settles; no second IK solve or actuator overshoot is added.
            controller.q_command = np.clip(stage.joint_goal, controller.lower, controller.upper)
            return False, min(0.999, progress)
        self.arm_stage_wait_started_at = None
        if stage.settle_s > 0.0:
            if self.arm_stage_settled_at is None:
                self.arm_stage_settled_at = float(now)
                return False, min(0.999, progress)
            if float(now) - self.arm_stage_settled_at < stage.settle_s:
                return False, min(0.999, progress)
        self.arm_stage_settled_at = None
        if stage.action is not None:
            stage.action()
            stage.action = None
        self.arm_stage_completed_duration += stage.duration_s
        self.arm_stage_index += 1
        if self.arm_stage_index >= len(self.arm_stages):
            controller.hold()
            controller.enabled = False
            controller.locked_local_indices = ()
            controller.full_orientation = False
            return True, 1.0
        self.arm_motion_ready = False
        self.arm_stage_start_q = None
        self.arm_stage_entry_started_at = None
        # Mark a following constrained stage as entry-pending immediately.
        # The scheduler may not plan it until its next rate-limited tick; in
        # that interval it is still the previous high hold, never descent.
        self.arm_stage_entry_pending = bool(self.arm_stages[self.arm_stage_index].strict_vertical)
        self.arm_stage_started_at = float(now)
        return False, min(0.999, progress)

    def _apply_process_action(self, process_fraction: float) -> None:
        if self.action_applied:
            return
        assert self.task is not None
        if self.task.task_type is TaskType.DISPENSE_BRAZING:
            # The path display follows measured TCP projection while Arm2 is
            # moving.  A millimetre-scale controller residual must not leave
            # completed beads with visibly different end points, so commit
            # every path to its canonical full product-coordinate extent.
            tray_index = _tray_index(self.task.tray_id)
            for path_index in range(len(self._plan().brazing_paths)):
                self.scene.registry.set_batch_brazing_path_progress(
                    tray_index,
                    path_index,
                    1.0,
                    reverse=False,
                )
        if self.task.task_type is TaskType.CONFIGURE_COMB:
            self.scene.registry.set_batch_comb_install_progress(
                _tray_index(self.task.tray_id),
                1.0,
            )
        if self.task.task_type is TaskType.REMOVE_OLD_COMB:
            self.scene.registry.set_batch_comb_install_progress(
                _tray_index(self.task.tray_id),
                0.0,
            )
        if self.task.task_type is TaskType.REMOVE_OLD_PRESS:
            index = _tray_index(self.task.tray_id)
            self.scene.registry.set_batch_press_removal_progress(index, 1.0)
            self.scene.registry.set_batch_press_visible(index, False)
        if self.task.task_type is TaskType.LOCK_FIXTURE:
            self.scene.registry.set_batch_press_locked(
                _tray_index(self.task.tray_id),
                True,
            )
        self._complete_raw_material_action()
        self._visual_stage(process_fraction, completing=True)
        self.action_applied = True

    def _visual_stage(self, fraction: float, *, completing: bool = False) -> None:
        """Commit only the physical item produced by the current task.

        Never reconstruct the whole tray from DAG statuses here.  Physical
        task callbacks occur just before their task is marked SUCCEEDED, and a
        concurrent callback would otherwise erase work that is already visible
        but not yet represented by a terminal graph node.
        """

        del fraction
        assert self.task is not None
        if self.task.tray_id is None or not completing:
            return
        index = _tray_index(self.task.tray_id)
        if self.task.task_type is TaskType.PLACE_BASE_PLATE:
            self.scene.registry.set_batch_base_visible(index, True)
        elif self.task.task_type is TaskType.INSTALL_FIN:
            fin_id = str(self.task.payload.get("fin_id", "fin_01"))
            self.scene.registry.set_batch_fin_visible(index, int(fin_id[-2:]) - 1, True)
        elif self.task.task_type in {TaskType.APPLY_PRESS, TaskType.LOCK_FIXTURE}:
            self.scene.registry.set_batch_press_visible(index, True)

    def _update_transfer(self, now: float) -> SkillExecutionResult:
        assert self.task is not None and self.transfer is not None
        registry = self.scene.registry
        state = self.transfer
        if not state.handed_off:
            target = registry.async_transfer_limit(state.transfer_id)
            elapsed = max(0.0, float(now) - state.started_at)
            linear_fraction = float(np.clip(elapsed / max(state.duration_s, 1.0e-6), 0.0, 1.0))
            command_fraction = quintic_time_scaling(linear_fraction)
            command = state.start_position + command_fraction * (target - state.start_position)
            registry.set_async_transfer_target(state.transfer_id, command)
            error = abs(registry.async_transfer_position(state.transfer_id) - target)
            velocity = abs(registry.async_transfer_velocity(state.transfer_id))
            if (
                linear_fraction >= 1.0
                and error <= self.POSITION_TOLERANCE_M
                and velocity <= self.VELOCITY_TOLERANCE_M_S
            ):
                if state.settled_at is None:
                    state.settled_at = float(now)
                elif float(now) - state.settled_at >= self.SETTLE_SECONDS:
                    registry.finish_batch_tray_async_transfer(
                        self.task.tray_id or "",
                        state.transfer_id,
                        state.destination,
                    )
                    state.handed_off = True
                    state.settled_at = None
                    state.return_started_at = float(now)
                    state.return_start_position = registry.async_transfer_position(state.transfer_id)
            else:
                state.settled_at = None
            actual_fraction = float(
                np.clip(
                    (registry.async_transfer_position(state.transfer_id) - state.start_position)
                    / max(target - state.start_position, 1.0e-9),
                    0.0,
                    1.0,
                )
            )
            progress = min(0.8, 0.8 * max(command_fraction, actual_fraction))
            return SkillExecutionResult.running_result({"progress": progress})
        if state.return_started_at is None:
            state.return_started_at = float(now)
        if state.return_start_position is None:
            state.return_start_position = registry.async_transfer_position(state.transfer_id)
        return_elapsed = max(0.0, float(now) - state.return_started_at)
        return_linear = float(np.clip(return_elapsed / max(state.return_duration_s, 1.0e-6), 0.0, 1.0))
        return_fraction = quintic_time_scaling(return_linear)
        return_command = state.return_start_position * (1.0 - return_fraction)
        registry.set_async_transfer_target(state.transfer_id, return_command)
        position = abs(registry.async_transfer_position(state.transfer_id))
        velocity = abs(registry.async_transfer_velocity(state.transfer_id))
        if (
            return_linear >= 1.0
            and position <= self.POSITION_TOLERANCE_M
            and velocity <= self.VELOCITY_TOLERANCE_M_S
        ):
            if state.settled_at is None:
                state.settled_at = float(now)
            elif float(now) - state.settled_at >= self.SETTLE_SECONDS:
                return SkillExecutionResult.success({"progress": 1.0, "physical_transfer": state.transfer_id})
        else:
            state.settled_at = None
        return SkillExecutionResult.running_result({"progress": 0.8 + 0.2 * return_fraction})

    def update(self, now: float, dt: float) -> SkillExecutionResult:
        del dt
        if self.task is None:
            return SkillExecutionResult.success()
        if self.cancelled:
            task_id = self.task.task_id
            self.task = None
            return SkillExecutionResult.failure("TASK_CANCELLED", {"task_id": task_id})
        if self.transfer is not None:
            result = self._update_transfer(now)
            if result.succeeded:
                self.task = None
                self.transfer = None
            return result
        if self.arm_stages:
            if not self.arm_motion_ready:
                try:
                    ready, planning_fraction = self._plan_one_arm_stage(now)
                except Exception as exc:
                    task_id = self.task.task_id
                    controller = self.scene.arms[self.resource_id.lower()]
                    controller.enabled = False
                    controller.q_command = np.asarray(
                        self.scene.data.qpos[controller.qpos_ids], dtype=float
                    ).copy()
                    self.task = None
                    self.arm_stages.clear()
                    return SkillExecutionResult.failure(
                        "KINEMATIC_PLANNING_FAILED",
                        {"task_id": task_id, "error": str(exc)},
                    )
                if not ready:
                    return SkillExecutionResult.running_result(
                        {
                            "progress": 0.01 * planning_fraction,
                            "motion_stage": f"在线验证动作段 {self.arm_stage_index + 1}/{len(self.arm_stages)}",
                            "physical_contact_sequence": True,
                        }
                    )
            done, progress = self._update_staged_arm(now)
            if not done:
                return SkillExecutionResult.running_result(
                    {
                        "progress": progress,
                        "motion_stage": self.arm_stages[self.arm_stage_index].label,
                        "physical_contact_sequence": True,
                    }
                )
            self._apply_process_action(1.0)
            metrics: dict[str, Any] = {
                "progress": 1.0,
                "physical_visible": True,
                "physical_contact_sequence": True,
            }
            self.arm_stages.clear()
            if self.task.task_type is TaskType.POST_BRAZE_INSPECTION:
                metrics["disposition"] = "PASS"
            self.task = None
            return SkillExecutionResult.success(metrics)
        fraction = self._task_fraction(now)
        if self.task.task_type is TaskType.CONFIGURE_COMB:
            self.scene.registry.set_batch_comb_install_progress(
                _tray_index(self.task.tray_id),
                fraction,
            )
        if self.task.task_type is TaskType.REMOVE_OLD_COMB:
            self.scene.registry.set_batch_comb_install_progress(
                _tray_index(self.task.tray_id),
                1.0 - fraction,
            )
        if self.task.task_type is TaskType.REMOVE_OLD_PRESS and self.task.payload.get("after_brazing"):
            self.scene.registry.set_batch_press_removal_progress(
                _tray_index(self.task.tray_id),
                fraction,
            )
        if self.task.task_type is TaskType.APPLY_PRESS:
            self.scene.registry.set_batch_press_progress(
                _tray_index(self.task.tray_id),
                fraction,
            )
        returns_to_entry = self.task.task_type not in {
            TaskType.PICK_BASE_PLATE,
            TaskType.PICK_FIN,
        }
        process_fraction = min(1.0, fraction / self.ACTION_FRACTION) if returns_to_entry else fraction
        action_threshold = (
            1.0
            if not returns_to_entry
            or self.task.task_type
            in {
                TaskType.CONFIGURE_COMB,
                TaskType.REMOVE_OLD_COMB,
                TaskType.REMOVE_OLD_PRESS,
            }
            else self.ACTION_FRACTION
        )
        if fraction + 1.0e-12 >= action_threshold:
            self._apply_process_action(process_fraction)
        if fraction < 1.0:
            return SkillExecutionResult.running_result({"progress": fraction})
        self._apply_process_action(1.0)
        metrics: dict[str, Any] = {"progress": 1.0, "physical_visible": True}
        if self.task.task_type is TaskType.POST_BRAZE_INSPECTION:
            metrics["disposition"] = "PASS"
        self.task = None
        return SkillExecutionResult.success(metrics)

    def cancel(self, task_id: str) -> None:
        if self.task is None or self.task.task_id != task_id:
            return
        if self.resource_id in {"ARM1", "ARM2", "ARM3"}:
            controller = self.scene.arms[self.resource_id.lower()]
            controller.q_command = np.asarray(self.scene.data.qpos[controller.qpos_ids], dtype=float).copy()
            controller.locked_local_indices = ()
            controller.full_orientation = False
            controller.enabled = False
        if self.transfer is not None:
            self.scene.registry.set_async_transfer_target(
                self.transfer.transfer_id,
                self.scene.registry.async_transfer_position(self.transfer.transfer_id),
            )
        self.cancelled = True
        # ExecutionMonitor removes a timed-out execution immediately, so the
        # reusable per-task-type skill must be released in the same call.
        # Waiting for a future update would leave a ghost "skill still
        # running" state and block every later fin of every order.
        self.task = None
        self.transfer = None
        self.arm_stages.clear()
        self.arm_stage_start_q = None


def build_physical_async_line_skill_registry(*, fast: bool = False) -> SkillRegistry:
    """Create isolated skill instances for all task types in viewer mode."""

    registry = SkillRegistry()
    physical_types = {
        TaskType.INDEX_EMPTY_TRAY,
        TaskType.PICK_BASE_PLATE,
        TaskType.PLACE_BASE_PLATE,
        TaskType.PREPARE_FIN_TOOL,
        TaskType.DISPENSE_BRAZING,
        TaskType.INSPECT_BRAZING,
        TaskType.CONFIGURE_COMB,
        TaskType.REMOVE_OLD_COMB,
        TaskType.REMOVE_OLD_PRESS,
        TaskType.PICK_FIN,
        TaskType.INSTALL_FIN,
        TaskType.INSPECT_FINS,
        TaskType.APPLY_PRESS,
        TaskType.LOCK_FIXTURE,
        *TRANSFER_BINDINGS,
        TaskType.POST_BRAZE_INSPECTION,
        TaskType.SECOND_POST_BRAZE_VIEW,
    }
    for task_type in TaskType:
        if task_type in LOGISTICS_TASK_TYPES:
            skill = AsyncLineLogisticsSkill(task_type, fast=fast)
        elif task_type in physical_types:
            skill = AsyncLinePhysicalSkill(task_type)
        else:
            skill = TimedSkill(0.10 if fast else None)
        registry.register(task_type, skill)
    return registry


__all__ = ["AsyncLinePhysicalSkill", "build_physical_async_line_skill_registry"]
