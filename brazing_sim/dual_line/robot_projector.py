"""Physical, waypoint-driven FR3 execution for the independent V2 line."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from ..inspection import top_down_inspection_pose
from ..layout import SHALLOW_U_LAYOUT
from ..motion import HOME_QPOS, ArmController, Pose, matrix_to_quat, pose_from_site
from ..profiles import quintic_time_scaling
from ..tools import QuickChangeToolManager, ToolSpec
from .fault_visuals import FIN_POSE_LATERAL_OFFSET_M
from .process_geometry import V2ProcessGeometry

if TYPE_CHECKING:
    from .runtime import DualLineRuntime, V2UnitState


def _top_down_pose(position: np.ndarray, *, yaw: float = math.pi) -> Pose:
    cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
    rotation = np.asarray(
        [
            [cosine, sine, 0.0],
            [sine, -cosine, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )
    return Pose(np.asarray(position, dtype=float), matrix_to_quat(rotation))


@dataclass(frozen=True, slots=True)
class _Waypoint:
    pose: Pose
    label_zh: str
    stop: bool = True
    interaction: str = ""
    cartesian_speed_m_s: float | None = None


@dataclass(slots=True)
class _JointPlan:
    operation_key: str
    instance_key: str
    operation_kind: str
    tool_name: str
    waypoints: tuple[_Waypoint, ...]
    joint_goals: tuple[np.ndarray, ...]
    waypoint_index: int
    segment_start: np.ndarray
    segment_elapsed_s: float
    segment_duration_s: float
    complete: bool = False
    failure: str = ""
    interaction_started: bool = False
    grasp_verified: bool = False
    release_verified: bool = False
    proxy_key: str | None = None
    deposition_points_per_pass: int = 0
    minimum_segment_s: float = 0.45
    grasp_waypoint_index: int | None = None
    release_waypoint_index: int | None = None
    deposition_line_start_offset: int = 0
    deposition_line_end_offset: int = 0
    reported_progress: float = 0.0
    motion_speed_scale: float = 1.0
    continuous_ranges: tuple[tuple[int, int], ...] = ()
    active_path_start: int | None = None
    active_path_end: int | None = None
    active_path_elapsed_s: float = 0.0
    active_path_duration_s: float = 0.0
    active_path_cumulative: np.ndarray | None = None
    locked_joint7_rad: float | None = None
    interaction_elapsed_s: float = 0.0
    interaction_start_fraction: float | None = None
    fin_thickness_m: float | None = None
    fin_clamp_position_m: float | None = None
    tray_id: str | None = None
    fin_index: int | None = None
    installed_fin_revealed: bool = False
    rework_fin: bool = False
    repair_fin: bool = False
    manifested_fault_type: str = ""
    grasp_failed: bool = False
    next_tool: str | None = None
    base_grasp_start_position: np.ndarray | None = None


@dataclass(slots=True)
class _WorkpieceProxy:
    body_id: int
    qpos_address: int
    dof_address: int
    geom_id: int
    feed_body_id: int
    tool_body_id: int
    feed_weld_id: int
    grasp_weld_id: int
    visible_rgba: np.ndarray
    held: bool = False
    visible: bool = False


class V2RobotMotionProjector:
    """Execute V2 robot operations with V1-compatible process motion.

    Runtime durations express scheduling intent.  Process paths are solved in
    advance, replayed with quintic time scaling, and written through the real
    FR3 joints.  This deliberately matches the stable V1 visual/controller
    contract: no sparse-waypoint servo chasing, and no redundant-wrist branch
    changes while a fin is grasped.
    """

    MINIMUM_SEGMENT_S = 0.45
    # At the viewer's 1x setting this yields a deliberate ~52 deg/s authored
    # joint trajectory.  The former 1.15 rad/s made 50 ms samples jump almost
    # five degrees between frames, which read as a flash even though the
    # underlying quintic path was continuous.
    NOMINAL_JOINT_SPEED_RAD_S = 0.90
    JOINT_SETTLE_RAD = 0.004
    JOINT_SETTLE_SPEED_RAD_S = 0.15
    GRIPPER_CLOSE_SECONDS = 0.32
    GRIPPER_RELEASE_SECONDS = 0.28
    # Final Arm3 TCP pose after the last B-branch fin.  Parking beside the
    # south magazine tucks links 2–4 away from the horizontal outbound pallet
    # perimeter while keeping the camera and gripper clear of the raw-fin rack.
    ARM3_BRANCH_CLEAR_PARK_M = (1.00, -0.95, 0.55)
    BASE_SUCTION_ENGAGE_SECONDS = 0.34
    # The S1 pickup site is authored 4 mm above the standard base top.  Keep
    # that physical suction standoff constant while the base thickness varies.
    BASE_SUCTION_STANDOFF_M = 0.004
    BASE_SUCTION_RELEASE_SECONDS = 0.26
    BASE_TRANSFER_SPEED_M_S = 0.18
    BASE_PLACE_SPEED_M_S = 0.035
    QUINTIC_PEAK_RATE = 1.875
    # Open 1 mm per finger after insertion.  This is visibly larger than the
    # former 0.4 mm stroke, while the narrowed fingers retain ample clearance
    # to the 15 mm-pitch C product's neighbouring fins.
    FIN_RELEASE_CLEARANCE_M = 0.0020
    FIN_PICK_FAILURE_EXTRA_GAP_M = 0.006
    LATERAL_FIN_FAULT_TYPES = frozenset({"FIN_POSE", "FIN_GEOMETRY_FAILED"})
    FINGER_CONTACT_TOLERANCE_M = 0.000075
    FIN_INSERT_SPEED_M_S = 0.025
    ARM1_TOOL_CHANGE_SEED = np.asarray(
        [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
        dtype=float,
    )
    # Certified elbow-up branch for Arm3's complete magazine→S3B corridor.
    # HOME converges to an elbow-down local basin near x≈0.85 m on later
    # magazine slots even though an exact solution exists.  This deterministic
    # seed selects the same continuous branch used by the first successful
    # fin; it does not alter the authored Cartesian path or move the blanks.
    ARM3_FIN_CORRIDOR_SEED = np.asarray(
        [0.85, -1.49, 0.72, -2.05, 1.01, 0.89, 0.92],
        dtype=float,
    )
    # Same TCP as ``ARM3_BRANCH_CLEAR_PARK_M``, solved on the certified
    # redundancy branch whose links retain at least 20 mm clearance from the
    # *complete* B-line payload (comb guides and press bars included) over the
    # vertical MERGE_B_WAIT→S4 sweep.  A Cartesian endpoint alone is
    # insufficient because another valid FR3 solution puts link 3 through the
    # left comb guide even though the tray slab itself remains clear.
    ARM3_BRANCH_CLEAR_QPOS = np.asarray(
        [
            1.0616983712183474,
            -0.6334018162398624,
            0.9397680521727119,
            -1.800293200896538,
            0.4953382064226898,
            1.4023823169061889,
            1.1830243798357174,
        ],
        dtype=float,
    )

    def __init__(
        self,
        model: Any,
        data: Any,
        *,
        set_component_visible: Callable[[str, str, bool], None] | None = None,
        restore_component_pose: Callable[[str, str], None] | None = None,
        fast_base_speed_scale: float = 4.0,
        fast_process_speed_scale: float = 3.0,
    ) -> None:
        import mujoco

        if not math.isfinite(float(fast_base_speed_scale)) or float(fast_base_speed_scale) <= 0.0:
            raise ValueError("fast base playback scale must be finite and positive")
        if not math.isfinite(float(fast_process_speed_scale)) or float(fast_process_speed_scale) <= 0.0:
            raise ValueError("fast process playback scale must be finite and positive")

        self.mujoco = mujoco
        self.model = model
        self.data = data
        self._set_component_visible = set_component_visible
        self._restore_component_pose = restore_component_pose
        self._fast_base_speed_scale = float(fast_base_speed_scale)
        self._fast_process_speed_scale = float(fast_process_speed_scale)
        self._paused_arms: set[str] = set()
        self._paused_joint_positions: dict[str, np.ndarray] = {}
        self.controllers = {arm: ArmController(model, data, arm) for arm in ("arm1", "arm2", "arm3")}
        self._arm1_tools = QuickChangeToolManager(
            model,
            data,
            self.controllers["arm1"],
            arm_name="arm1",
            registry={
                "arm1_gripper": ToolSpec(
                    name="arm1_gripper",
                    body="v2_arm1_parallel_gripper",
                    free_joint="v2_arm1_parallel_gripper_free",
                    tcp_site="v2_arm1_grasp_tcp",
                    mount_site="v2_arm1_parallel_gripper_mount_site",
                    rack_site="v2_arm1_parallel_gripper_rack_site",
                    arm_weld="v2_arm1_toolchange_parallel_gripper",
                    rack_weld="v2_arm1_rack_parallel_gripper",
                ),
                "arm1_suction": ToolSpec(
                    name="arm1_suction",
                    body="v2_arm1_suction_tool",
                    free_joint="v2_arm1_suction_tool_free",
                    tcp_site="v2_arm1_suction_tcp",
                    mount_site="v2_arm1_suction_tool_mount_site",
                    rack_site="v2_arm1_suction_tool_rack_site",
                    arm_weld="v2_arm1_toolchange_suction_tool",
                    rack_weld="v2_arm1_rack_suction_tool",
                ),
            },
        )
        self._finger_actuators = {
            "arm1": tuple(
                int(model.actuator(name).id)
                for name in (
                    "v2_arm1_left_finger_actuator",
                    "v2_arm1_right_finger_actuator",
                )
            ),
            "arm3": tuple(
                int(model.actuator(name).id)
                for name in (
                    "v2_arm3_left_finger_actuator",
                    "v2_arm3_right_finger_actuator",
                )
            ),
        }
        self._finger_qpos = {
            "arm1": tuple(
                int(model.jnt_qposadr[model.joint(name).id])
                for name in (
                    "v2_arm1_left_finger_joint",
                    "v2_arm1_right_finger_joint",
                )
            ),
            "arm3": tuple(
                int(model.jnt_qposadr[model.joint(name).id])
                for name in (
                    "v2_arm3_left_finger_joint",
                    "v2_arm3_right_finger_joint",
                )
            ),
        }
        self._finger_dof = {
            arm_name: tuple(
                int(model.jnt_dofadr[model.joint(f"v2_{arm_name}_{side}_finger_joint").id])
                for side in ("left", "right")
            )
            for arm_name in ("arm1", "arm3")
        }
        # Contact gaps are measured between the visible inner pad faces at
        # q=0.  A clamp command is derived from the active product thickness,
        # never from the mechanical travel limit.
        self._finger_open_gap_m = {"arm1": 0.042, "arm3": 0.042}
        self._finger_max_position_m = {
            arm_name: min(float(model.actuator_ctrlrange[actuator_id, 1]) for actuator_id in actuator_ids)
            for arm_name, actuator_ids in self._finger_actuators.items()
        }
        self._arm2_tool_weld = int(model.equality("v2_arm2_dispenser_tool_weld").id)
        self._arm3_tool_weld = int(model.equality("v2_arm3_hybrid_tool_weld").id)
        self._fixed_tool_state: dict[str, tuple[int, int, int, int, int]] = {}
        for equality_id, link_name, tool_name, free_joint_name in (
            (
                self._arm2_tool_weld,
                "arm2_fr3_link7",
                "v2_arm2_dual_brazing_dispenser_tool",
                "v2_arm2_dual_brazing_dispenser_tool_free",
            ),
            (
                self._arm3_tool_weld,
                "arm3_fr3_link7",
                "v2_arm3_hybrid_tool",
                "v2_arm3_hybrid_tool_free",
            ),
        ):
            self._write_weld_relative(
                equality_id,
                int(model.body(link_name).id),
                int(model.body(tool_name).id),
            )
            self.data.eq_active[equality_id] = 1
            arm_name = link_name.split("_", 1)[0]
            joint_id = int(model.joint(free_joint_name).id)
            self._fixed_tool_state[arm_name] = (
                int(model.body(link_name).id),
                int(model.body(tool_name).id),
                int(model.jnt_qposadr[joint_id]),
                int(model.jnt_dofadr[joint_id]),
                equality_id,
            )
        self.mujoco.mj_forward(self.model, self.data)
        self._proxies = {
            "arm1_base": self._make_proxy(
                body="v2_arm1_raw_base_proxy",
                free_joint="v2_arm1_raw_base_proxy_free",
                geom="v2_arm1_raw_base_proxy_geom",
                feed_body="v2_base_supply_fixture",
                tool_body="v2_arm1_suction_tool",
                feed_weld="v2_arm1_raw_base_feed_weld",
                grasp_weld="v2_arm1_grasp_base_proxy_weld",
            ),
            "arm1_fin": self._make_proxy(
                body="v2_arm1_raw_fin_proxy",
                free_joint="v2_arm1_raw_fin_proxy_free",
                geom="v2_arm1_raw_fin_proxy_geom",
                feed_body="v2_fin_table_a",
                tool_body="v2_arm1_parallel_gripper",
                feed_weld="v2_arm1_raw_fin_feed_weld",
                grasp_weld="v2_arm1_grasp_fin_proxy_weld",
            ),
            "arm3_fin": self._make_proxy(
                body="v2_arm3_raw_fin_proxy",
                free_joint="v2_arm3_raw_fin_proxy_free",
                geom="v2_arm3_raw_fin_proxy_geom",
                feed_body="v2_fin_table_b",
                tool_body="v2_arm3_parallel_gripper",
                feed_weld="v2_arm3_raw_fin_feed_weld",
                grasp_weld="v2_arm3_grasp_fin_proxy_weld",
            ),
        }
        for proxy in self._proxies.values():
            self._set_proxy_visible(proxy, False)
        self._raw_fin_rgba = {
            (branch, index): np.asarray(
                self.model.geom(f"v2_fin_{branch}_raw_fin_{index:02d}").rgba,
                dtype=float,
            ).copy()
            for branch in ("a", "b")
            for index in range(1, 13)
        }
        for branch, index in self._raw_fin_rgba:
            self._set_raw_fin_visible(branch, index, False)
        arm2_flange = pose_from_site(data, "arm2_attachment_site")
        arm3_flange = pose_from_site(data, "arm3_attachment_site")
        self._tool_transforms = {
            "arm1_flange": Pose([0.0, 0.0, 0.0]),
            "arm1_suction": Pose([0.0, 0.0, 0.090]),
            "arm1_gripper": Pose([0.0, 0.0, 0.090]),
            "arm2_dispenser": arm2_flange.inverse().transformed(
                pose_from_site(data, "v2_arm2_dispenser_center_tcp")
            ),
            "arm3_camera": arm3_flange.inverse().transformed(pose_from_site(data, "v2_arm3_camera_tcp")),
            "arm3_gripper": arm3_flange.inverse().transformed(pose_from_site(data, "v2_arm3_gripper_tcp")),
        }
        self._tool_tcp_sites = {
            "arm1_flange": "arm1_attachment_site",
            "arm1_suction": "v2_arm1_suction_tcp",
            "arm1_gripper": "v2_arm1_grasp_tcp",
            "arm2_dispenser": "v2_arm2_dispenser_center_tcp",
            "arm3_camera": "v2_arm3_camera_tcp",
            "arm3_gripper": "v2_arm3_gripper_tcp",
        }
        self._suction_pad_id = int(model.geom("v2_arm1_suction_pad").id)
        self._suction_pad_rgba = np.asarray(
            model.geom_rgba[self._suction_pad_id],
            dtype=float,
        ).copy()
        self._suction_pad_size = np.asarray(
            model.geom_size[self._suction_pad_id],
            dtype=float,
        ).copy()
        self._suction_pad_pos = np.asarray(
            model.geom_pos[self._suction_pad_id],
            dtype=float,
        ).copy()
        self._set_arm1_suction_fraction(0.0)
        # Arm2 never changes its dispenser. Arm3's fixed hybrid head uses the
        # camera as its safe idle TCP and switches to the gripper TCP only for
        # fin handling. Keeping these defaults active prevents an idle frame
        # from silently falling back to the bare flange transform.
        self.controllers["arm2"].set_tool_transform(self._tool_transforms["arm2_dispenser"])
        self.controllers["arm3"].set_tool_transform(self._tool_transforms["arm3_camera"])
        self._plans: dict[str, _JointPlan | None] = {arm: None for arm in self.controllers}
        self._active_operation: dict[str, str] = {arm: "" for arm in self.controllers}
        self._target_label: dict[str, str] = {arm: "等待任务" for arm in self.controllers}
        # Fine-grained task completion can be polled one control tick after a
        # combined operation leaves the active plan. Preserve measured
        # grasp/release milestones until reset so they cannot be missed.
        self._measured_milestones: set[tuple[str, str]] = set()
        # Exact latent fin defects currently present on a tray.  The cache is
        # refreshed from runtime truth every scene sync so physical grasp
        # behaviour can differ from a successful nominal operation without
        # coupling the scene-independent fault controller to MuJoCo.
        self._active_fin_faults: dict[tuple[str, int], str] = {}

    def _make_proxy(
        self,
        *,
        body: str,
        free_joint: str,
        geom: str,
        feed_body: str,
        tool_body: str,
        feed_weld: str,
        grasp_weld: str,
    ) -> _WorkpieceProxy:
        joint_id = int(self.model.joint(free_joint).id)
        geom_id = int(self.model.geom(geom).id)
        return _WorkpieceProxy(
            body_id=int(self.model.body(body).id),
            qpos_address=int(self.model.jnt_qposadr[joint_id]),
            dof_address=int(self.model.jnt_dofadr[joint_id]),
            geom_id=geom_id,
            feed_body_id=int(self.model.body(feed_body).id),
            tool_body_id=int(self.model.body(tool_body).id),
            feed_weld_id=int(self.model.equality(feed_weld).id),
            grasp_weld_id=int(self.model.equality(grasp_weld).id),
            visible_rgba=np.asarray(self.model.geom_rgba[geom_id], dtype=float).copy(),
        )

    @staticmethod
    def _body_pose(data: Any, body_id: int) -> Pose:
        body = data.body(body_id)
        return Pose(
            np.asarray(body.xpos, dtype=float),
            matrix_to_quat(np.asarray(body.xmat, dtype=float).reshape(3, 3)),
        )

    def _write_weld_relative(
        self,
        equality_id: int,
        body1_id: int,
        body2_id: int,
    ) -> None:
        relative = (
            self._body_pose(self.data, body1_id).inverse().transformed(self._body_pose(self.data, body2_id))
        )
        self.model.eq_data[equality_id, :] = 0.0
        self.model.eq_data[equality_id, 3:6] = relative.position
        self.model.eq_data[equality_id, 6:10] = relative.quaternion
        self.model.eq_data[equality_id, 10] = 1.0

    def _set_proxy_visible(
        self,
        proxy: _WorkpieceProxy,
        visible: bool,
    ) -> None:
        rgba = proxy.visible_rgba.copy()
        rgba[3] = proxy.visible_rgba[3] if visible else 0.0
        self.model.geom_rgba[proxy.geom_id] = rgba
        proxy.visible = bool(visible)

    def _set_arm1_suction_fraction(self, fraction: float) -> None:
        """Mirror V1's cyan energized-pad feedback and compression."""

        amount = float(np.clip(fraction, 0.0, 1.0))
        rgba = self._suction_pad_rgba.copy()
        rgba[:3] = (1.0 - amount) * rgba[:3] + amount * np.asarray(
            [0.08, 0.85, 0.95],
            dtype=float,
        )
        self.model.geom_rgba[self._suction_pad_id] = rgba
        half_height = float(self._suction_pad_size[1]) - 0.0015 * amount
        self.model.geom_size[self._suction_pad_id, 1] = half_height
        self.model.geom_pos[self._suction_pad_id, 2] = (
            float(self._suction_pad_pos[2]) + float(self._suction_pad_size[1]) - half_height
        )

    def _prepare_proxy(
        self,
        proxy_key: str,
        position: np.ndarray,
        quaternion: np.ndarray | None = None,
    ) -> None:
        proxy = self._proxies[proxy_key]
        self.data.eq_active[proxy.grasp_weld_id] = 0
        self.data.eq_active[proxy.feed_weld_id] = 0
        address = proxy.qpos_address
        self.data.qpos[address : address + 3] = np.asarray(position, dtype=float)
        self.data.qpos[address + 3 : address + 7] = (
            (1.0, 0.0, 0.0, 0.0) if quaternion is None else np.asarray(quaternion, dtype=float)
        )
        self.data.qvel[proxy.dof_address : proxy.dof_address + 6] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        self._write_weld_relative(
            proxy.feed_weld_id,
            proxy.feed_body_id,
            proxy.body_id,
        )
        self.data.eq_active[proxy.feed_weld_id] = 1
        proxy.held = False
        self._set_proxy_visible(proxy, True)
        self.mujoco.mj_forward(self.model, self.data)

    def _grasp_proxy(self, plan: _JointPlan) -> None:
        if plan.proxy_key is None:
            return
        proxy = self._proxies[plan.proxy_key]
        if proxy.held:
            return
        self._write_weld_relative(
            proxy.grasp_weld_id,
            proxy.tool_body_id,
            proxy.body_id,
        )
        self.data.eq_active[proxy.grasp_weld_id] = 1
        self.data.eq_active[proxy.feed_weld_id] = 0
        proxy.held = True
        self.mujoco.mj_forward(self.model, self.data)

    def _align_base_proxy_with_suction(self, plan: _JointPlan, fraction: float) -> None:
        """Smoothly draw the supplied base onto the real measured suction TCP."""

        if plan.proxy_key != "arm1_base":
            return
        proxy = self._proxies[plan.proxy_key]
        if plan.base_grasp_start_position is None:
            plan.base_grasp_start_position = np.asarray(
                self.data.body(proxy.body_id).xpos,
                dtype=float,
            ).copy()
        tcp = pose_from_site(self.data, "v2_arm1_suction_tcp")
        half_thickness = float(self.model.geom_size[proxy.geom_id, 2])
        target = tcp.position + tcp.rotation[:, 2] * (self.BASE_SUCTION_STANDOFF_M + half_thickness)
        amount = quintic_time_scaling(float(np.clip(fraction, 0.0, 1.0)))
        position = plan.base_grasp_start_position + amount * (target - plan.base_grasp_start_position)
        self.data.eq_active[proxy.feed_weld_id] = 0
        self.data.qpos[proxy.qpos_address : proxy.qpos_address + 3] = position
        self.data.qvel[proxy.dof_address : proxy.dof_address + 6] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        self._write_weld_relative(
            proxy.feed_weld_id,
            proxy.feed_body_id,
            proxy.body_id,
        )
        self.data.eq_active[proxy.feed_weld_id] = 1
        self.mujoco.mj_forward(self.model, self.data)

    def _release_proxy(self, plan: _JointPlan) -> None:
        if plan.proxy_key is None:
            return
        proxy = self._proxies[plan.proxy_key]
        self.data.eq_active[proxy.grasp_weld_id] = 0
        self.data.eq_active[proxy.feed_weld_id] = 0
        proxy.held = False
        self._set_proxy_visible(proxy, False)
        if plan.operation_kind == "BASE_LOADING":
            self._set_arm1_suction_fraction(0.0)
        self.mujoco.mj_forward(self.model, self.data)

    def _tray_frame(
        self,
        unit: "V2UnitState",
        *,
        expected_dock_site: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        if unit.tray_id is None:
            raise RuntimeError(f"{unit.unit_id}尚未获得物理托盘")
        # The mocap carrier is the authoritative route frame.  The welded
        # free tray can differ by a few micrometres under the constraint
        # solver; inheriting that numerical wobble made authored straight
        # dispensing paths look slightly skewed.  Product geometry remains
        # physically welded to this same carrier.
        body = self.data.body(f"{unit.tray_id.lower()}_carrier")
        origin = np.asarray(body.xpos, dtype=float).copy()
        rotation = np.asarray(body.xmat, dtype=float).reshape(3, 3).copy()
        if expected_dock_site is not None:
            dock = self.data.site(expected_dock_site)
            dock_origin = np.asarray(dock.xpos, dtype=float)
            if float(np.linalg.norm(origin - dock_origin)) > 0.010:
                # A freshly submitted first order may exist before the adapter
                # has installed its physical gate.  Predict the imminent dock
                # frame instead of planning against the tray's empty-buffer
                # home pose; all later operations use the measured live frame.
                origin = dock_origin.copy()
                rotation = np.asarray(dock.xmat, dtype=float).reshape(3, 3).copy()
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0])) + math.pi
        return origin, rotation, yaw

    @staticmethod
    def _active_fin_index(unit: "V2UnitState") -> int:
        return int(unit.rework_fin_index or (int(unit.fins_installed) + 1))

    @classmethod
    def _fin_operation_completes_unit(cls, unit: "V2UnitState") -> bool:
        return unit.rework_fin_index is not None or cls._active_fin_index(unit) >= int(unit.fin_count)

    def _raw_fin_position(self, arm_name: str, unit: "V2UnitState") -> np.ndarray:
        branch = "a" if arm_name == "arm1" else "b"
        index = min(self._active_fin_index(unit), 12)
        geom = self.data.geom(f"v2_fin_{branch}_raw_fin_{index:02d}")
        return np.asarray(geom.xpos, dtype=float).copy()

    def _installed_fin_pose(self, unit: "V2UnitState") -> Pose:
        if unit.tray_id is None:
            raise RuntimeError(f"{unit.unit_id}翅片返工缺少托盘")
        geom = self.data.geom(f"{unit.tray_id.lower()}_fin_{self._active_fin_index(unit):02d}")
        return Pose(
            np.asarray(geom.xpos, dtype=float).copy(),
            matrix_to_quat(np.asarray(geom.xmat, dtype=float).reshape(3, 3)),
        )

    def _fin_install_goal(
        self,
        unit: "V2UnitState",
        geometry: V2ProcessGeometry,
        *,
        origin: np.ndarray,
        rotation: np.ndarray,
    ) -> np.ndarray:
        """Return the physical insertion target, including a latent slot miss."""

        index = self._active_fin_index(unit)
        goal = geometry.world_fin_target(
            index - 1,
            origin=origin,
            rotation=rotation,
        )
        fault_type = self._active_fin_faults.get((unit.unit_id, index), "")
        if fault_type in self.LATERAL_FIN_FAULT_TYPES:
            goal = goal + rotation @ np.asarray(
                [0.0, FIN_POSE_LATERAL_OFFSET_M, 0.0],
                dtype=float,
            )
        return goal

    def _configure_nozzle_spacing(self, spacing_m: float) -> None:
        half_spacing = 0.5 * float(spacing_m)
        for side, sign in (("left", -1.0), ("right", 1.0)):
            y = sign * half_spacing
            self.model.site(f"v2_arm2_{side}_nozzle_tip_site").pos[1] = y
            self.model.geom(f"v2_arm2_{side}_nozzle_tip").pos[1] = y
        self.mujoco.mj_forward(self.model, self.data)

    def _fin_clamp_position(self, arm_name: str, thickness_m: float) -> float:
        thickness = float(thickness_m)
        open_gap = self._finger_open_gap_m[arm_name]
        target = 0.5 * (open_gap - thickness)
        maximum = self._finger_max_position_m[arm_name]
        if thickness <= 0.0 or target < 0.0 or target > maximum + 1.0e-9:
            raise RuntimeError(
                f"{arm_name}夹爪无法夹持厚度{thickness * 1_000.0:.2f} mm的翅片："
                f"所需单指行程{target * 1_000.0:.2f} mm，"
                f"可用行程{maximum * 1_000.0:.2f} mm"
            )
        return float(np.clip(target, 0.0, maximum))

    def _reveal_installed_fin(self, plan: _JointPlan) -> None:
        if plan.installed_fin_revealed:
            return
        if plan.tray_id is None or plan.fin_index is None:
            raise RuntimeError("翅片释放缺少托盘可见所有权目标")
        if self._set_component_visible is None:
            raise RuntimeError("翅片释放缺少场景可见性适配器")
        if plan.repair_fin and self._restore_component_pose is not None:
            # A repaired fin is handed directly to the authored tray slot.
            # Clear any still-rendered latent-defect offset before alpha is
            # enabled so the release frame cannot sink, rise or snap.
            self._restore_component_pose(
                plan.tray_id,
                f"fin_{plan.fin_index:02d}",
            )
        self._set_component_visible(
            plan.tray_id,
            f"fin_{plan.fin_index:02d}",
            True,
        )
        plan.installed_fin_revealed = True

    def _reveal_installed_base(self, plan: _JointPlan) -> None:
        """Atomically hand the released base proxy to the tray visual pool."""

        if plan.tray_id is None:
            raise RuntimeError("基板释放缺少托盘可见所有权目标")
        if self._set_component_visible is None:
            raise RuntimeError("基板释放缺少场景可见性适配器")
        self._set_component_visible(plan.tray_id, "base_plate", True)

    def pending_installed_fin_index(self, unit_id: str) -> int:
        """Return the fin already released physically but not yet committed logically."""

        for plan in self._plans.values():
            if (
                plan is not None
                and plan.operation_key == f"{unit_id}:INSTALL_FIN"
                and plan.installed_fin_revealed
                and plan.fin_index is not None
            ):
                return int(plan.fin_index)
        return 0

    def pending_installed_base(self, unit_id: str) -> bool:
        """Keep the tray-owned base visible throughout the post-release retreat."""

        return any(
            plan is not None and plan.operation_key == f"{unit_id}:BASE_LOADING" and plan.release_verified
            for plan in self._plans.values()
        )

    def rework_fin_owned_by_proxy(self, unit_id: str, fin_index: int) -> bool:
        """Whether the existing tray fin is currently represented by the arm proxy."""

        for plan in self._plans.values():
            if (
                plan is None
                or not plan.rework_fin
                or plan.operation_key != f"{unit_id}:INSTALL_FIN"
                or plan.fin_index != int(fin_index)
                or plan.proxy_key is None
            ):
                continue
            return bool(self._proxies[plan.proxy_key].visible)
        return False

    def _sync_active_fin_faults(self, runtime: "DualLineRuntime") -> None:
        self._active_fin_faults = {
            (defect.unit_id, int(defect.operation_index)): defect.fault_type.value
            for defect in runtime.faults.physical_faults.values()
            if defect.operation_index is not None
            and defect.status in {"MANIFESTED", "DETECTED"}
            and defect.fault_type.value.startswith("FIN_")
        }

    def _restore_failed_pick_to_source(self, plan: _JointPlan) -> None:
        if plan.proxy_key is None or plan.fin_index is None:
            raise RuntimeError("翅片抓取失败缺少原料代理所有权")
        arm_name = plan.proxy_key.removesuffix("_fin")
        branch = "a" if arm_name == "arm1" else "b"
        # The fixed blank and temporary proxy occupy the same authored pose.
        # Reveal the fixed blank first, then hide/release the proxy so the fin
        # never follows the arm and never flashes out of the source magazine.
        self._set_raw_fin_visible(branch, plan.fin_index, True)
        self._release_proxy(plan)

    def _set_raw_fin_visible(self, branch: str, index: int, visible: bool) -> None:
        geom = self.model.geom(f"v2_fin_{branch}_raw_fin_{index:02d}")
        rgba = self._raw_fin_rgba[(branch, index)].copy()
        rgba[3] = self._raw_fin_rgba[(branch, index)][3] if visible else 0.0
        geom.rgba = rgba

    def _configure_raw_payload(
        self,
        arm_name: str,
        operation_kind: str,
        geometry: V2ProcessGeometry,
        unit: "V2UnitState",
        *,
        recovering_failed_pick: bool = False,
    ) -> None:
        if operation_kind == "BASE_LOADING":
            self.model.geom("v2_arm1_raw_base_proxy_geom").size[:3] = (
                np.asarray(geometry.base_size_m, dtype=float) / 2.0
            )
            return
        if operation_kind != "INSTALL_FIN":
            return
        branch = "a" if arm_name == "arm1" else "b"
        half_size = np.asarray(geometry.fin_size_m, dtype=float) / 2.0
        picked_index = self._active_fin_index(unit)
        v1_positions = tuple(
            SHALLOW_U_LAYOUT.raw_fin_position(
                index,
                int(unit.fin_count),
                table_top_z=0.0,
                fin_height_m=float(geometry.fin_size_m[2]),
            )
            for index in range(int(unit.fin_count))
        )
        v1_centre_y = 0.5 * (v1_positions[0][1] + v1_positions[-1][1])
        for index in range(1, 13):
            raw = self.model.geom(f"v2_fin_{branch}_raw_fin_{index:02d}")
            raw.size[:3] = half_size
            raw.pos[2] = half_size[2]
            if index <= int(unit.fin_count):
                raw.pos[1] = float(v1_positions[index - 1][1] - v1_centre_y)
            # The proxy replaces the blank being picked in the same sync.
            # The remaining fixed blanks therefore show exactly the V1 kit
            # count without a duplicate at the active pickup slot.
            failed_pick_waiting = (
                self._active_fin_faults.get((unit.unit_id, index)) == "FIN_PICK_FAILED"
                and not recovering_failed_pick
            )
            future_blank = not recovering_failed_pick and picked_index < index <= int(unit.fin_count)
            self._set_raw_fin_visible(
                branch,
                index,
                future_blank or failed_pick_waiting,
            )
        self.model.geom(f"v2_{arm_name}_raw_fin_proxy_geom").size[:3] = half_size
        self.mujoco.mj_forward(self.model, self.data)

    def _fin_waypoints(
        self,
        unit: "V2UnitState",
        *,
        source: np.ndarray,
        goal: np.ndarray,
        station_name: str,
        safe_z: float,
        yaw: float,
        routing_y: float | None = None,
    ) -> tuple[_Waypoint, ...]:
        routing_z = max(float(safe_z), float(source[2]) + 0.13, float(goal[2]) + 0.105)
        source_safe_position = np.asarray([source[0], source[1], routing_z], dtype=float)
        goal_safe_position = np.asarray([goal[0], goal[1], routing_z], dtype=float)
        goal_approach_position = np.asarray([goal[0], goal[1], goal[2] + 0.105], dtype=float)
        waypoints: list[_Waypoint] = [
            _Waypoint(
                _top_down_pose(source_safe_position, yaw=yaw),
                f"{station_name} 原料位安全接近并渐进张开夹爪",
                interaction="open",
            )
        ]

        # Copy V1's process geometry literally: pickup and placement are
        # sampled Cartesian Z-lines, while carrying is a pose-locked 10 mm
        # polyline.  Joint interpolation between only the endpoints produces
        # an arc in Cartesian space and was the source of the visible twitch.
        pickup_distance = float(source_safe_position[2] - source[2])
        pickup_samples = max(2, int(math.ceil(pickup_distance / 0.010)))
        for index in range(1, pickup_samples + 1):
            fraction = index / pickup_samples
            position = source_safe_position + fraction * (source - source_safe_position)
            waypoints.append(
                _Waypoint(
                    _top_down_pose(position, yaw=yaw),
                    f"{station_name} 缓慢纯Z下降夹取",
                    stop=index == pickup_samples,
                    interaction="grasp" if index == pickup_samples else "",
                )
            )

        lift_samples = max(2, int(math.ceil(pickup_distance / 0.010)))
        for index in range(1, lift_samples + 1):
            fraction = index / lift_samples
            position = source + fraction * (source_safe_position - source)
            waypoints.append(
                _Waypoint(
                    _top_down_pose(position, yaw=yaw),
                    f"{station_name} 夹紧后保持姿态垂直抬升",
                    stop=False,
                )
            )

        route_points = [source_safe_position]
        if routing_y is not None:
            route_points.extend(
                (
                    np.asarray([source_safe_position[0], routing_y, routing_z]),
                    np.asarray([goal_safe_position[0], routing_y, routing_z]),
                )
            )
        route_points.append(goal_safe_position)
        for segment_start, segment_end in zip(route_points, route_points[1:]):
            carry_distance = float(np.linalg.norm(segment_end - segment_start))
            carry_samples = max(2, int(math.ceil(carry_distance / 0.010)))
            for index in range(1, carry_samples + 1):
                fraction = index / carry_samples
                position = segment_start + fraction * (segment_end - segment_start)
                waypoints.append(
                    _Waypoint(
                        _top_down_pose(position, yaw=yaw),
                        f"{station_name} 固定随体姿态绕开基座平滑运送",
                        stop=False,
                    )
                )

        if not np.allclose(goal_safe_position, goal_approach_position, atol=1.0e-9):
            approach_distance = float(np.linalg.norm(goal_approach_position - goal_safe_position))
            approach_samples = max(2, int(math.ceil(approach_distance / 0.010)))
            for index in range(1, approach_samples + 1):
                fraction = index / approach_samples
                position = goal_safe_position + fraction * (goal_approach_position - goal_safe_position)
                waypoints.append(
                    _Waypoint(
                        _top_down_pose(position, yaw=yaw),
                        f"{station_name} 槽位正上方锁定姿态",
                        stop=index == approach_samples,
                    )
                )
        else:
            previous = waypoints[-1]
            waypoints[-1] = _Waypoint(previous.pose, f"{station_name} 槽位正上方锁定姿态")

        descent_distance = float(goal_approach_position[2] - goal[2])
        descent_samples = max(2, int(math.ceil(descent_distance / 0.0015)))
        for index in range(1, descent_samples + 1):
            fraction = index / descent_samples
            position = goal_approach_position + fraction * (goal - goal_approach_position)
            waypoints.append(
                _Waypoint(
                    _top_down_pose(position, yaw=yaw),
                    (
                        f"{station_name} 保持角度纯Z向下放置"
                        if index == descent_samples
                        else f"{station_name} 锁定XY和角度后纯Z下降"
                    ),
                    stop=index == descent_samples or index % 4 == 0,
                    cartesian_speed_m_s=self.FIN_INSERT_SPEED_M_S,
                )
            )
        waypoints.append(
            _Waypoint(
                _top_down_pose(goal, yaw=yaw),
                f"{station_name} 槽内稳定并小行程渐进松爪",
                interaction="release",
            )
        )
        for index in range(1, descent_samples + 1):
            fraction = index / descent_samples
            position = goal + fraction * (goal_approach_position - goal)
            waypoints.append(
                _Waypoint(
                    _top_down_pose(position, yaw=yaw),
                    f"{station_name} 松爪后保持姿态垂直撤离",
                    stop=index == descent_samples,
                )
            )
        if station_name == "S3B" and self._fin_operation_completes_unit(unit):
            # The lower-branch rail starts beside S3B.  Completing the final
            # insertion while the hybrid gripper is still over the pallet
            # lets the tray leave underneath the fingers.  Make clearance a
            # physical part of the final fin task: return through the same
            # pose-locked corridor to the magazine-side safe point, and only
            # then release the runtime operation/transport reservation.
            egress_distance = float(np.linalg.norm(source_safe_position - goal_approach_position))
            egress_samples = max(2, int(math.ceil(egress_distance / 0.010)))
            for index in range(1, egress_samples + 1):
                fraction = index / egress_samples
                position = goal_approach_position + fraction * (source_safe_position - goal_approach_position)
                waypoints.append(
                    _Waypoint(
                        _top_down_pose(position, yaw=yaw),
                        "S3B 完成后退出托盘运输区",
                        stop=index == egress_samples,
                    )
                )
            park_position = np.asarray(
                self.ARM3_BRANCH_CLEAR_PARK_M,
                dtype=float,
            )
            park_distance = float(
                np.linalg.norm(park_position - source_safe_position),
            )
            park_samples = max(2, int(math.ceil(park_distance / 0.010)))
            for index in range(1, park_samples + 1):
                fraction = index / park_samples
                position = source_safe_position + fraction * (park_position - source_safe_position)
                waypoints.append(
                    _Waypoint(
                        _top_down_pose(position, yaw=yaw),
                        "S3B 完成后收拢至平面物流避让位",
                        stop=index == park_samples,
                    )
                )
        return tuple(waypoints)

    def _fin_rework_waypoints(
        self,
        *,
        source: np.ndarray,
        goal: np.ndarray,
        source_yaw: float,
        goal_yaw: float,
        safe_z: float,
    ) -> tuple[_Waypoint, ...]:
        """Reseat one existing S3B fin without visiting the raw-fin magazine."""

        source = np.asarray(source, dtype=float)
        goal = np.asarray(goal, dtype=float)
        routing_z = max(float(safe_z), float(source[2]) + 0.105, float(goal[2]) + 0.105)
        source_safe = np.asarray([source[0], source[1], routing_z], dtype=float)
        goal_safe = np.asarray([goal[0], goal[1], routing_z], dtype=float)
        waypoints: list[_Waypoint] = [
            _Waypoint(
                _top_down_pose(source_safe, yaw=source_yaw),
                "S3B 缺陷翅片原槽位正上方对准",
                interaction="open",
            )
        ]

        pickup_distance = float(source_safe[2] - source[2])
        pickup_samples = max(2, int(math.ceil(pickup_distance / 0.003)))
        for index in range(1, pickup_samples + 1):
            fraction = index / pickup_samples
            position = source_safe + fraction * (source - source_safe)
            waypoints.append(
                _Waypoint(
                    _top_down_pose(position, yaw=source_yaw),
                    "S3B 缓慢纯Z下降夹取偏位翅片",
                    stop=index == pickup_samples,
                    interaction="grasp" if index == pickup_samples else "",
                    cartesian_speed_m_s=self.FIN_INSERT_SPEED_M_S,
                )
            )

        lift_samples = max(2, int(math.ceil(pickup_distance / 0.003)))
        for index in range(1, lift_samples + 1):
            fraction = index / lift_samples
            position = source + fraction * (source_safe - source)
            waypoints.append(
                _Waypoint(
                    _top_down_pose(position, yaw=source_yaw),
                    "S3B 夹紧缺陷翅片后纯Z抬升",
                    stop=False,
                    cartesian_speed_m_s=self.FIN_INSERT_SPEED_M_S,
                )
            )

        correction_distance = float(np.linalg.norm(goal_safe - source_safe))
        yaw_distance = abs(float(goal_yaw - source_yaw))
        correction_samples = max(
            2,
            int(math.ceil(correction_distance / 0.005)),
            int(math.ceil(yaw_distance / math.radians(1.0))),
        )
        for index in range(1, correction_samples + 1):
            fraction = index / correction_samples
            position = source_safe + fraction * (goal_safe - source_safe)
            yaw = source_yaw + fraction * (goal_yaw - source_yaw)
            waypoints.append(
                _Waypoint(
                    _top_down_pose(position, yaw=yaw),
                    "S3B 安全高度渐进纠偏并重新对准原槽位",
                    stop=index == correction_samples,
                )
            )

        descent_distance = float(goal_safe[2] - goal[2])
        descent_samples = max(2, int(math.ceil(descent_distance / 0.0015)))
        for index in range(1, descent_samples + 1):
            fraction = index / descent_samples
            position = goal_safe + fraction * (goal - goal_safe)
            waypoints.append(
                _Waypoint(
                    _top_down_pose(position, yaw=goal_yaw),
                    "S3B 重新对准后保持XY和角度纯Z回插",
                    stop=index == descent_samples or index % 4 == 0,
                    cartesian_speed_m_s=self.FIN_INSERT_SPEED_M_S,
                )
            )
        waypoints.append(
            _Waypoint(
                _top_down_pose(goal, yaw=goal_yaw),
                "S3B 原槽位复位完成并小行程松爪",
                interaction="release",
            )
        )
        for index in range(1, descent_samples + 1):
            fraction = index / descent_samples
            position = goal + fraction * (goal_safe - goal)
            waypoints.append(
                _Waypoint(
                    _top_down_pose(position, yaw=goal_yaw),
                    "S3B 修复后保持姿态纯Z撤离",
                    stop=index == descent_samples,
                )
            )
        park_position = np.asarray(self.ARM3_BRANCH_CLEAR_PARK_M, dtype=float)
        park_distance = float(np.linalg.norm(park_position - goal_safe))
        park_samples = max(2, int(math.ceil(park_distance / 0.010)))
        for index in range(1, park_samples + 1):
            fraction = index / park_samples
            position = goal_safe + fraction * (park_position - goal_safe)
            waypoints.append(
                _Waypoint(
                    _top_down_pose(position, yaw=goal_yaw),
                    "S3B 翅片纠偏完成后退出托盘物流区",
                    stop=index == park_samples,
                )
            )
        return tuple(waypoints)

    def _operation_waypoints(
        self,
        arm_name: str,
        operation: Any,
        unit: "V2UnitState",
    ) -> tuple[str, tuple[_Waypoint, ...]]:
        if arm_name == "arm1" and operation.kind == "BASE_LOADING":
            geometry = V2ProcessGeometry.for_unit(unit)
            origin, rotation, yaw = self._tray_frame(
                unit,
                expected_dock_site="v2_station_s1_dock",
            )
            base_goal = origin + rotation @ np.asarray(
                [
                    0.0,
                    0.0,
                    geometry.base_top_z_m + self.BASE_SUCTION_STANDOFF_M,
                ],
                dtype=float,
            )
            base_approach = base_goal + rotation @ np.asarray([0.0, 0.0, 0.120], dtype=float)
            supply = np.asarray(
                self.data.site("v2_base_supply_pickup_site").xpos,
                dtype=float,
            ).copy()
            supply_approach = supply + np.asarray([0.0, 0.0, 0.120], dtype=float)

            def sampled_transfer(
                start: np.ndarray,
                end: np.ndarray,
                label: str,
                *,
                final_stop: bool,
            ) -> tuple[_Waypoint, ...]:
                distance = float(np.linalg.norm(end - start))
                sample_count = max(2, int(math.ceil(distance / 0.010)))
                return tuple(
                    _Waypoint(
                        _top_down_pose(
                            start + (index / sample_count) * (end - start),
                            yaw=yaw,
                        ),
                        label,
                        stop=final_stop and index == sample_count,
                        cartesian_speed_m_s=self.BASE_TRANSFER_SPEED_M_S,
                    )
                    for index in range(1, sample_count + 1)
                )

            transfer = (
                *sampled_transfer(
                    supply_approach,
                    base_approach,
                    "S1 携板直线移动至托盘上方",
                    final_stop=True,
                ),
            )
            descent_distance = float(np.linalg.norm(base_goal - base_approach))
            descent_samples = max(12, int(math.ceil(descent_distance / 0.003)))
            descent = tuple(
                _Waypoint(
                    _top_down_pose(
                        base_approach + (index / descent_samples) * (base_goal - base_approach),
                        yaw=yaw,
                    ),
                    ("S1 缓慢放置基板" if index == descent_samples else "S1 主板锁定XY姿态后逐步纯Z下降"),
                    stop=index == descent_samples,
                    interaction="release" if index == descent_samples else "",
                    cartesian_speed_m_s=self.BASE_PLACE_SPEED_M_S,
                )
                for index in range(1, descent_samples + 1)
            )
            return (
                "arm1_suction",
                (
                    _Waypoint(
                        _top_down_pose(supply_approach, yaw=yaw),
                        "S1 原料位安全接近",
                    ),
                    _Waypoint(
                        _top_down_pose(supply, yaw=yaw),
                        "S1 缓慢吸取基板",
                        interaction="grasp",
                    ),
                    _Waypoint(
                        _top_down_pose(supply_approach, yaw=yaw),
                        "S1 吸取后垂直抬升",
                    ),
                    *transfer,
                    *descent,
                    _Waypoint(
                        _top_down_pose(base_approach, yaw=yaw),
                        "S1 放置后垂直撤离",
                    ),
                ),
            )
        if arm_name == "arm1" and operation.kind == "INSTALL_FIN":
            geometry = V2ProcessGeometry.for_unit(unit)
            origin, rotation, _tray_yaw = self._tray_frame(unit)
            return (
                "arm1_gripper",
                self._fin_waypoints(
                    unit,
                    source=self._raw_fin_position(arm_name, unit),
                    goal=self._fin_install_goal(
                        unit,
                        geometry,
                        origin=origin,
                        rotation=rotation,
                    ),
                    station_name="S3A",
                    safe_z=0.52,
                    # V1 keeps one fixed world attitude from pickup through
                    # release.  The V2 trays never rotate, so inheriting a
                    # derived tray yaw here only selected the opposite wrist
                    # branch and made B's outer blank unreachable.
                    yaw=0.0,
                ),
            )
        if arm_name == "arm2" and operation.kind == "DISPENSING":
            geometry = V2ProcessGeometry.for_unit(unit)
            origin, rotation, yaw = self._tray_frame(unit)
            points: list[_Waypoint] = []
            local_target = (
                int(operation.recovery_target_index)
                if operation.recovery_strategy == "LOCAL_BRAZING_REWORK"
                and operation.recovery_target_index is not None
                else None
            )
            pass_indices = (
                (max(0, (local_target - 1) // 2),) if local_target is not None else range(unit.fin_count)
            )

            def append_dense_path(
                start: np.ndarray,
                end: np.ndarray,
                *,
                approach_label: str,
                start_label: str,
                travel_label: str,
                finish_label: str,
            ) -> None:
                hover = start + rotation @ np.asarray([0.0, 0.0, 0.045 if points else 0.080])
                points.append(_Waypoint(_top_down_pose(hover, yaw=yaw), approach_label))
                points.append(_Waypoint(_top_down_pose(start, yaw=yaw), start_label))
                # V1 uses a dense 3 mm Cartesian chain for every bead.  This
                # keeps the centre TCP and both physical nozzle tips on one
                # authored straight line instead of asking the joint servo to
                # approximate a line between sparse 10 mm targets.
                sample_count = max(2, int(math.ceil(float(np.linalg.norm(end - start)) / 0.003)))
                for sample_index in range(1, sample_count + 1):
                    fraction = sample_index / sample_count
                    position = start + fraction * (end - start)
                    points.append(
                        _Waypoint(
                            _top_down_pose(position, yaw=yaw),
                            travel_label,
                            stop=sample_index == sample_count,
                        )
                    )
                lift = end + rotation @ np.asarray([0.0, 0.0, 0.035])
                points.append(_Waypoint(_top_down_pose(lift, yaw=yaw), finish_label))

            for pass_index in pass_indices:
                dispense_pass = geometry.world_dispense_pass(
                    pass_index,
                    origin=origin,
                    rotation=rotation,
                )
                start, end = dispense_pass.start.copy(), dispense_pass.end.copy()
                if local_target is not None:
                    # The defect visual retains the first 36% of a missing
                    # path.  Move one enabled nozzle only over the missing tail;
                    # the paired nozzle is treated as shut off for touch-up.
                    if operation.recovery_fault_type == "BRAZING_MISSING":
                        start = start + 0.36 * (end - start)
                elif pass_index % 2:
                    start, end = end, start
                if local_target is not None and operation.recovery_fault_type == "BRAZING_PATH_DEVIATION":
                    append_dense_path(
                        start,
                        end,
                        approach_label=f"S2A {local_target:02d}号偏轨钎料清除安全接近",
                        start_label=f"S2A {local_target:02d}号偏轨钎料清除起点对准",
                        travel_label=f"S2A {local_target:02d}号偏轨钎料从一端向另一端逐段清除",
                        finish_label=f"S2A {local_target:02d}号偏轨钎料清除完成后抬枪",
                    )
                    append_dense_path(
                        start,
                        end,
                        approach_label=f"S2A {local_target:02d}号目标焊道重新涂覆安全接近",
                        start_label=f"S2A {local_target:02d}号目标焊道重新涂覆起点对准",
                        travel_label=f"S2A {local_target:02d}号目标焊道从一端向另一端重新涂覆",
                        finish_label=f"S2A {local_target:02d}号目标焊道重新涂覆完成后抬枪",
                    )
                else:
                    append_dense_path(
                        start,
                        end,
                        approach_label=(
                            f"S2A {local_target:02d}号焊道局部补涂安全接近"
                            if local_target is not None
                            else f"S2A 第{pass_index + 1}道安全接近"
                        ),
                        start_label=(
                            f"S2A {local_target:02d}号焊道缺口起点精确对准（关闭非目标喷嘴）"
                            if local_target is not None
                            else f"S2A 第{pass_index + 1}道起点精确对准"
                        ),
                        travel_label=(
                            f"S2A {local_target:02d}号焊道局部连续补涂"
                            if local_target is not None
                            else f"S2A 第{pass_index + 1}道双喷嘴连续涂覆"
                        ),
                        finish_label=(
                            f"S2A {local_target:02d}号焊道补涂完成后抬枪"
                            if local_target is not None
                            else f"S2A 第{pass_index + 1}道完成后抬枪"
                        ),
                    )
            return "arm2_dispenser", tuple(points)
        if arm_name == "arm3" and operation.kind == "MATERIAL_INSPECTION":
            geometry = V2ProcessGeometry.for_unit(unit)
            origin, rotation, yaw = self._tray_frame(unit)
            return (
                "arm3_camera",
                (
                    _Waypoint(
                        top_down_inspection_pose(
                            origin + rotation @ np.asarray([0.0, 0.0, geometry.base_center_z_m]),
                            product_length_m=geometry.base_size_m[0],
                            product_width_m=geometry.base_size_m[1],
                            product_yaw_rad=yaw,
                        ),
                        "S2B 焊料检测",
                    ),
                ),
            )
        if arm_name == "arm3" and operation.kind == "PRE_BRAZE_INSPECTION":
            geometry = V2ProcessGeometry.for_unit(unit)
            origin, rotation, yaw = self._tray_frame(unit)
            return (
                "arm3_camera",
                (
                    _Waypoint(
                        top_down_inspection_pose(
                            origin + rotation @ np.asarray([0.0, 0.0, geometry.base_center_z_m]),
                            product_length_m=geometry.base_size_m[0],
                            product_width_m=geometry.base_size_m[1],
                            product_yaw_rad=yaw,
                        ),
                        "S4 焊前检测",
                    ),
                ),
            )
        if arm_name == "arm3" and operation.kind == "INSTALL_FIN":
            geometry = V2ProcessGeometry.for_unit(unit)
            origin, rotation, tray_yaw = self._tray_frame(unit)
            nominal_goal = geometry.world_fin_target(
                self._active_fin_index(unit) - 1,
                origin=origin,
                rotation=rotation,
            )
            if (
                operation.recovery_strategy == "FIN_REINSTALL"
                and operation.recovery_fault_type != "FIN_PICK_FAILED"
            ):
                installed = self._installed_fin_pose(unit)
                fin_rotation = installed.rotation
                fin_yaw = math.atan2(float(fin_rotation[1, 0]), float(fin_rotation[0, 0]))
                # A parallel gripper is pi-periodic.  Select the source
                # orientation closest to the certified S3B insertion yaw so
                # the correction is the small measured defect, never a 180°
                # wrist spin.
                yaw_delta = (fin_yaw - tray_yaw + 0.5 * math.pi) % math.pi - 0.5 * math.pi
                source_yaw = tray_yaw + yaw_delta
                return (
                    "arm3_gripper",
                    self._fin_rework_waypoints(
                        source=installed.position,
                        # Recovery must target the authored comb slot, not the
                        # deliberately offset latent-fault insertion goal.
                        goal=nominal_goal,
                        source_yaw=source_yaw,
                        goal_yaw=tray_yaw,
                        safe_z=0.54,
                    ),
                )
            goal = self._fin_install_goal(
                unit,
                geometry,
                origin=origin,
                rotation=rotation,
            )
            source = self._raw_fin_position(arm_name, unit)
            return (
                "arm3_gripper",
                self._fin_waypoints(
                    unit,
                    source=source,
                    goal=goal,
                    station_name="S3B",
                    safe_z=0.54,
                    # Arm3 faces the opposite branch, so π is the mirrored
                    # local equivalent of V1 Arm1's zero-yaw grasp.  The
                    # stage sequence and carried-fin attitude invariants are
                    # identical; only the cell-side mounting frame differs.
                    yaw=tray_yaw,
                ),
            )
        raise ValueError(f"unsupported V2 robot operation: {arm_name}/{operation.kind}")

    def _segment_duration(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        *,
        minimum_s: float | None = None,
        speed_scale: float = 1.0,
    ) -> float:
        travel = float(np.max(np.abs(goal - start)))
        return max(
            self.MINIMUM_SEGMENT_S if minimum_s is None else float(minimum_s),
            self.QUINTIC_PEAK_RATE
            * travel
            / (self.NOMINAL_JOINT_SPEED_RAD_S * max(float(speed_scale), 1.0e-9)),
        )

    @staticmethod
    def _process_ik_tolerances(operation_kind: str) -> tuple[float, float]:
        if operation_kind == "DISPENSING":
            return 0.0005, math.radians(0.10)
        if operation_kind == "INSTALL_FIN":
            return 0.0005, math.radians(0.15)
        if operation_kind == "TOOL_CHANGE":
            return 0.0005, math.radians(0.15)
        return 0.0015, math.radians(0.50)

    def _solve_waypoint_chain(
        self,
        controller: ArmController,
        waypoints: tuple[_Waypoint, ...],
        *,
        seed: np.ndarray,
        operation_kind: str,
        locked_joint7_rad: float | None = None,
        position_tolerance_override_m: float | None = None,
        orientation_tolerance_override_rad: float | None = None,
    ) -> tuple[list[np.ndarray], str]:
        position_tolerance, orientation_tolerance = self._process_ik_tolerances(operation_kind)
        if position_tolerance_override_m is not None:
            position_tolerance = float(position_tolerance_override_m)
        if orientation_tolerance_override_rad is not None:
            orientation_tolerance = float(orientation_tolerance_override_rad)
        locks = None if locked_joint7_rad is None else {6: float(locked_joint7_rad)}
        current_seed = seed.copy()
        if locked_joint7_rad is not None:
            current_seed[6] = float(locked_joint7_rad)

        # The first fin waypoint is an aerial approach, not a grasp or an
        # insertion constraint.  Arm3 can arrive here directly after a camera
        # inspection, where its warm-started redundancy branch may converge to
        # a sub-millimetre pose with roughly one degree of harmless wrist
        # residual.  Applying the 0.5 mm / 0.15 degree insertion tolerance to
        # that free-space waypoint falsely rejected the second fin and stalled
        # the entire shared line.  Solve only this clearance pose with the
        # normal controller acceptance envelope; every pickup, carried pose,
        # 1.5 mm insertion sample and release below remains on the strict
        # process tolerance.
        prefix_results: tuple[Any, ...] = ()
        remaining_waypoints = waypoints
        if operation_kind == "INSTALL_FIN" and waypoints:
            approach = controller.solve_ik(
                waypoints[0].pose,
                tcp=True,
                seed=current_seed,
                locked_joints=locks,
                max_iterations=1_200,
                position_tolerance_m=0.0015,
                orientation_tolerance_rad=math.radians(1.0),
                full_orientation=True,
            )
            prefix_results = (approach,)
            if not approach.reachable:
                results = prefix_results
                remaining_waypoints = ()
            else:
                current_seed = np.asarray(approach.joint_positions, dtype=float).copy()
                remaining_waypoints = waypoints[1:]

        suffix_results = controller.solve_pose_chain(
            (waypoint.pose for waypoint in remaining_waypoints),
            tcp=True,
            seed=current_seed,
            locked_joints=locks,
            max_iterations=1_200,
            position_tolerance_m=position_tolerance,
            orientation_tolerance_rad=orientation_tolerance,
            full_orientation=True,
        )
        results = prefix_results + suffix_results
        goals = [
            np.asarray(result.joint_positions, dtype=float).copy() for result in results if result.reachable
        ]
        if len(goals) == len(waypoints):
            return goals, ""
        failed_index = min(len(goals), len(waypoints) - 1)
        result = results[failed_index]
        waypoint = waypoints[failed_index]
        return goals, (
            f"{waypoint.label_zh} 不可达：位置误差 "
            f"{result.position_error_m * 1_000.0:.1f} mm，"
            f"姿态误差 {math.degrees(result.orientation_error_rad):.1f}°"
        )

    @staticmethod
    def _required_tool(arm_name: str, operation_kind: str) -> str | None:
        if arm_name != "arm1":
            return None
        return {
            "BASE_LOADING": "arm1_suction",
            "INSTALL_FIN": "arm1_gripper",
        }.get(operation_kind)

    def _build_arm1_tool_change_plan(
        self,
        operation: Any,
        target_tool: str,
        *,
        fast: bool,
    ) -> _JointPlan:
        """Plan a visible rack return/dock sequence with the bare flange.

        The weld ownership changes only at the physical rack contact pose.
        Solving the whole sequence for the flange keeps the joint branch
        continuous across the instant where the active TCP changes.
        """

        controller = self.controllers["arm1"]
        current_tool = self._arm1_tools.current_tool
        waypoints: list[_Waypoint] = []

        def append_vertical(
            start_pose: Pose,
            end_pose: Pose,
            label: str,
            *,
            speed_m_s: float,
            final_interaction: str = "",
        ) -> None:
            distance = float(np.linalg.norm(end_pose.position - start_pose.position))
            sample_count = max(2, int(math.ceil(distance / 0.003)))
            for sample_index in range(1, sample_count + 1):
                fraction = sample_index / sample_count
                position = start_pose.position + fraction * (end_pose.position - start_pose.position)
                final = sample_index == sample_count
                waypoints.append(
                    _Waypoint(
                        Pose(position, end_pose.quaternion),
                        label,
                        stop=final,
                        interaction=final_interaction if final else "",
                        cartesian_speed_m_s=speed_m_s,
                    )
                )

        if current_tool is not None:
            hover, dock, _retreat = self._arm1_tools.change_poses(current_tool, hover_m=0.10)
            current_zh = "吸盘" if current_tool == "arm1_suction" else "夹爪"
            waypoints.append(_Waypoint(hover, f"换刀：归还{current_zh}安全接近"))
            append_vertical(
                hover,
                dock,
                f"换刀：归还{current_zh}到架并解锁",
                speed_m_s=0.025,
                final_interaction=f"tool_return:{current_tool}",
            )
            append_vertical(
                dock,
                hover,
                f"换刀：空法兰退出{current_zh}工位",
                speed_m_s=0.035,
            )
        hover, dock, _retreat = self._arm1_tools.change_poses(target_tool, hover_m=0.10)
        target_zh = "吸盘" if target_tool == "arm1_suction" else "夹爪"
        waypoints.append(_Waypoint(hover, f"换刀：取用{target_zh}安全接近"))
        append_vertical(
            hover,
            dock,
            f"换刀：取用{target_zh}并锁定",
            speed_m_s=0.025,
            final_interaction=f"tool_dock:{target_tool}",
        )
        append_vertical(
            dock,
            hover,
            f"换刀：带{target_zh}平稳撤离工具架",
            speed_m_s=0.035,
        )
        waypoint_tuple = tuple(waypoints)
        controller.set_tool_transform(self._tool_transforms["arm1_flange"])
        start = np.asarray(self.data.qpos[controller.qpos_ids], dtype=float).copy()
        goals, failure = self._solve_waypoint_chain(
            controller,
            waypoint_tuple,
            seed=start,
            operation_kind="TOOL_CHANGE",
        )
        if failure:
            goals, failure = self._solve_waypoint_chain(
                controller,
                waypoint_tuple,
                seed=self.ARM1_TOOL_CHANGE_SEED,
                operation_kind="TOOL_CHANGE",
            )
        first_goal = start if not goals else goals[0]
        speed_scale = 2.0 if fast else 1.0
        stop_indices = [index for index, waypoint in enumerate(waypoint_tuple) if waypoint.stop]
        continuous_ranges = tuple(
            (left, right) for left, right in zip(stop_indices, stop_indices[1:]) if right > left + 1
        )
        return _JointPlan(
            operation_key=f"{operation.unit_id}:{operation.kind}",
            instance_key=(
                f"{operation.unit_id}:{operation.kind}:TOOL_CHANGE:" f"{float(operation.started_at):.9f}"
            ),
            operation_kind="TOOL_CHANGE",
            tool_name="arm1_flange",
            waypoints=waypoint_tuple,
            joint_goals=tuple(goals),
            waypoint_index=0,
            segment_start=start,
            segment_elapsed_s=0.0,
            segment_duration_s=self._segment_duration(
                start,
                first_goal,
                minimum_s=0.24 if not fast else 0.14,
                speed_scale=speed_scale,
            ),
            failure=failure,
            minimum_segment_s=0.24 if not fast else 0.14,
            motion_speed_scale=speed_scale,
            continuous_ranges=continuous_ranges,
            next_tool=target_tool,
        )

    def _build_plan(
        self,
        arm_name: str,
        operation: Any,
        unit: "V2UnitState",
        *,
        fast: bool,
    ) -> _JointPlan:
        controller = self.controllers[arm_name]
        geometry = V2ProcessGeometry.for_unit(unit)
        recovering_failed_pick = bool(
            operation.kind == "INSTALL_FIN"
            and operation.recovery_strategy == "FIN_REINSTALL"
            and operation.recovery_fault_type == "FIN_PICK_FAILED"
        )
        rework_fin = bool(
            operation.kind == "INSTALL_FIN"
            and operation.recovery_strategy == "FIN_REINSTALL"
            and not recovering_failed_pick
        )
        if not rework_fin:
            self._configure_raw_payload(
                arm_name,
                operation.kind,
                geometry,
                unit,
                recovering_failed_pick=recovering_failed_pick,
            )
        if arm_name == "arm2" and operation.kind == "DISPENSING":
            self._configure_nozzle_spacing(geometry.nozzle_spacing_m)
        tool_name, waypoints = self._operation_waypoints(arm_name, operation, unit)
        if arm_name == "arm1" and self._arm1_tools.current_tool != tool_name:
            raise RuntimeError(f"Arm1未完成{tool_name}的物理换刀")
        proxy_key: str | None = None
        if operation.kind == "BASE_LOADING":
            proxy_key = "arm1_base"
            supply = np.asarray(
                self.data.site("v2_base_supply_pickup_site").xpos,
                dtype=float,
            ).copy()
            supply[2] -= self.BASE_SUCTION_STANDOFF_M + 0.5 * geometry.base_size_m[2]
            self._prepare_proxy(proxy_key, supply)
        elif operation.kind == "INSTALL_FIN":
            proxy_key = f"{arm_name}_fin"
            if rework_fin:
                installed = self._installed_fin_pose(unit)
                self._prepare_proxy(
                    proxy_key,
                    installed.position,
                    installed.quaternion,
                )
                if unit.tray_id is None or self._set_component_visible is None:
                    raise RuntimeError("翅片原槽位返工缺少托盘可见性适配器")
                # The proxy takes over the exact existing fin pose in the same
                # scene sync.  Hiding only this target avoids both duplication
                # and the one-frame disappearance produced by replacing the
                # whole installed-fin set.
                self._set_component_visible(
                    unit.tray_id,
                    f"fin_{self._active_fin_index(unit):02d}",
                    False,
                )
            else:
                self._prepare_proxy(
                    proxy_key,
                    self._raw_fin_position(arm_name, unit),
                )
        fin_thickness_m = float(geometry.fin_size_m[1]) if operation.kind == "INSTALL_FIN" else None
        fin_clamp_position_m = (
            self._fin_clamp_position(arm_name, fin_thickness_m) if fin_thickness_m is not None else None
        )
        controller.set_tool_transform(self._tool_transforms[tool_name])
        seed = np.asarray(self.data.qpos[controller.qpos_ids], dtype=float).copy()
        # V1's original cell happens to admit one literal q7 value across its
        # pickup and installation envelopes.  The separated V2 branches and
        # Arm3 hybrid offset do not.  Lock the invariant the product actually
        # needs instead: every dense sample has the same world tool SE(3), and
        # sequential warm-started IK forbids redundant-branch jumps.
        locked_joint7_rad = None
        goals, failure = self._solve_waypoint_chain(
            controller,
            waypoints,
            seed=seed,
            operation_kind=operation.kind,
            locked_joint7_rad=locked_joint7_rad,
        )
        if failure and locked_joint7_rad is not None:
            # Some V2 branch/table combinations have no single literal q7
            # value that covers both endpoints.  Preserve the stronger V1
            # invariant that matters to the workpiece (constant world SE(3))
            # and fall back to a dense branch-continuous redundancy path.
            locked_joint7_rad = None
            goals, failure = self._solve_waypoint_chain(
                controller,
                waypoints,
                seed=seed,
                operation_kind=operation.kind,
            )
        if failure and arm_name == "arm3" and operation.kind == "INSTALL_FIN":
            corridor_goals, corridor_failure = self._solve_waypoint_chain(
                controller,
                waypoints,
                seed=self.ARM3_FIN_CORRIDOR_SEED,
                operation_kind=operation.kind,
                locked_joint7_rad=locked_joint7_rad,
            )
            if not corridor_failure:
                goals, failure = corridor_goals, ""
        if failure:
            # Match V1's deterministic certified-seed fallback when the
            # previous task leaves DLS in a poor local basin.  The resulting
            # first joint segment is still replayed continuously from the
            # measured live configuration.
            alternate_goals, alternate_failure = self._solve_waypoint_chain(
                controller,
                waypoints,
                seed=HOME_QPOS,
                operation_kind=operation.kind,
                locked_joint7_rad=locked_joint7_rad,
            )
            if not alternate_failure:
                goals, failure = alternate_goals, ""
            elif operation.kind == "INSTALL_FIN":
                # A camera task can leave Arm3 on a different but valid
                # redundancy branch.  Keep the precise solution whenever it
                # exists; otherwise replay the same dense Cartesian path with
                # the controller's certified 3 mm / 3 degree acceptance
                # envelope instead of freezing the line on an aerial sample.
                fallback_goals, fallback_failure = self._solve_waypoint_chain(
                    controller,
                    waypoints,
                    seed=HOME_QPOS,
                    operation_kind=operation.kind,
                    locked_joint7_rad=locked_joint7_rad,
                    position_tolerance_override_m=0.003,
                    orientation_tolerance_override_rad=math.radians(3.0),
                )
                if not fallback_failure:
                    goals, failure = fallback_goals, ""
                else:
                    failure = fallback_failure
        if (
            not failure
            and arm_name == "arm3"
            and operation.kind == "INSTALL_FIN"
            and self._fin_operation_completes_unit(unit)
            and len(goals) == len(waypoints)
        ):
            # Finish the final B-line task on the physically certified
            # redundancy branch.  The preceding Cartesian egress already
            # clears the stationary product; this last smooth joint segment
            # tucks the elbow before the runtime may release the tray.
            goals[-1] = self.ARM3_BRANCH_CLEAR_QPOS.copy()
        start = np.asarray(self.data.qpos[controller.qpos_ids], dtype=float).copy()
        first_goal = start if not goals else goals[0]
        points_per_pass = (
            (
                (
                    len(waypoints) // 2
                    if operation.recovery_fault_type == "BRAZING_PATH_DEVIATION"
                    else len(waypoints)
                )
                if operation.recovery_strategy == "LOCAL_BRAZING_REWORK"
                else len(waypoints) // unit.fin_count
            )
            if operation.kind == "DISPENSING"
            else 0
        )
        # Fast/headless rehearsals preserve the same quintic paths and stop
        # points while compressing Cartesian dwell.  The previous 2.5 scale
        # left the collision-safe branch reservation just outside the
        # established three-order acceptance window.
        # Fast playback must not increase the per-control-tick joint jump: the
        # V1-derived trajectories and clearance checks are authored around
        # these limits.  Unified headless acceleration belongs at the
        # application/physics stepping boundary, not inside the joint plan.
        speed_scale = (
            self._fast_base_speed_scale
            if fast and operation.kind == "BASE_LOADING"
            else self._fast_process_speed_scale if fast else 1.0
        )
        stop_indices = [index for index, waypoint in enumerate(waypoints) if waypoint.stop]
        continuous_ranges = tuple(
            (left, right) for left, right in zip(stop_indices, stop_indices[1:]) if right > left + 1
        )
        grasp_indices = [index for index, waypoint in enumerate(waypoints) if waypoint.interaction == "grasp"]
        release_indices = [
            index for index, waypoint in enumerate(waypoints) if waypoint.interaction == "release"
        ]
        return _JointPlan(
            operation_key=f"{operation.unit_id}:{operation.kind}",
            instance_key=(f"{operation.unit_id}:{operation.kind}:" f"{float(operation.started_at):.9f}"),
            operation_kind=operation.kind,
            tool_name=tool_name,
            waypoints=waypoints,
            joint_goals=tuple(goals),
            waypoint_index=0,
            segment_start=start,
            segment_elapsed_s=0.0,
            segment_duration_s=self._segment_duration(
                start,
                first_goal,
                speed_scale=speed_scale,
            ),
            complete=False,
            failure=failure,
            proxy_key=proxy_key,
            deposition_points_per_pass=points_per_pass,
            minimum_segment_s=(
                (0.020 if fast else 0.10)
                if operation.kind == "DISPENSING"
                else (
                    (0.10 if fast else 0.18)
                    if operation.kind == "INSTALL_FIN"
                    else (
                        (0.020 if fast else 0.050)
                        if operation.kind == "BASE_LOADING"
                        else self.MINIMUM_SEGMENT_S
                    )
                )
            ),
            grasp_waypoint_index=(
                grasp_indices[0] if grasp_indices else 1 if operation.kind == "BASE_LOADING" else None
            ),
            release_waypoint_index=(
                release_indices[0] if release_indices else 4 if operation.kind == "BASE_LOADING" else None
            ),
            deposition_line_start_offset=(1 if points_per_pass else 0),
            deposition_line_end_offset=(points_per_pass - 2 if points_per_pass else 0),
            motion_speed_scale=speed_scale,
            continuous_ranges=continuous_ranges,
            locked_joint7_rad=locked_joint7_rad,
            fin_thickness_m=fin_thickness_m,
            fin_clamp_position_m=fin_clamp_position_m,
            tray_id=(unit.tray_id if operation.kind in {"BASE_LOADING", "INSTALL_FIN"} else None),
            fin_index=(self._active_fin_index(unit) if operation.kind == "INSTALL_FIN" else None),
            rework_fin=rework_fin,
            repair_fin=bool(operation.kind == "INSTALL_FIN" and operation.recovery),
        )

    def _start_continuous_path(
        self,
        plan: _JointPlan,
        start_index: int,
        end_index: int,
    ) -> None:
        path = np.stack(plan.joint_goals[start_index : end_index + 1])
        lengths = np.max(np.abs(np.diff(path, axis=0)), axis=1)
        cumulative = np.concatenate((np.asarray([0.0]), np.cumsum(lengths)))
        total = float(cumulative[-1])
        plan.active_path_start = int(start_index)
        plan.active_path_end = int(end_index)
        plan.active_path_elapsed_s = 0.0
        plan.active_path_duration_s = max(
            plan.minimum_segment_s,
            self.QUINTIC_PEAK_RATE
            * total
            / (self.NOMINAL_JOINT_SPEED_RAD_S * max(plan.motion_speed_scale, 1.0)),
        )
        requested_speeds = [
            waypoint.cartesian_speed_m_s
            for waypoint in plan.waypoints[start_index + 1 : end_index + 1]
            if waypoint.cartesian_speed_m_s is not None
        ]
        if requested_speeds:
            cartesian_distance = sum(
                float(
                    np.linalg.norm(
                        plan.waypoints[index].pose.position - plan.waypoints[index - 1].pose.position
                    )
                )
                for index in range(start_index + 1, end_index + 1)
            )
            effective_speed = min(requested_speeds) * max(plan.motion_speed_scale, 1.0)
            plan.active_path_duration_s = max(
                plan.active_path_duration_s,
                cartesian_distance / max(effective_speed, 1.0e-9),
            )
        plan.active_path_cumulative = cumulative
        plan.segment_elapsed_s = 0.0
        plan.segment_duration_s = plan.active_path_duration_s
        plan.waypoint_index = start_index + 1

    def _continuous_path_command(
        self,
        plan: _JointPlan,
        timestep: float,
    ) -> tuple[np.ndarray, bool]:
        assert plan.active_path_start is not None
        assert plan.active_path_end is not None
        assert plan.active_path_cumulative is not None
        plan.active_path_elapsed_s += timestep
        plan.segment_elapsed_s = plan.active_path_elapsed_s
        fraction = min(
            1.0,
            plan.active_path_elapsed_s / max(plan.active_path_duration_s, 1.0e-9),
        )
        cumulative = plan.active_path_cumulative
        distance = quintic_time_scaling(fraction) * float(cumulative[-1])
        local_left = min(
            int(np.searchsorted(cumulative, distance, side="right") - 1),
            len(cumulative) - 2,
        )
        local_left = max(0, local_left)
        local_right = local_left + 1
        span = float(cumulative[local_right] - cumulative[local_left])
        local_fraction = 1.0 if span <= 1.0e-12 else (distance - cumulative[local_left]) / span
        start_index = plan.active_path_start
        left = plan.joint_goals[start_index + local_left]
        right = plan.joint_goals[start_index + local_right]
        plan.waypoint_index = start_index + local_right
        return left + local_fraction * (right - left), fraction >= 1.0

    def _write_free_pose(
        self,
        qpos_address: int,
        dof_address: int,
        pose: Pose,
    ) -> None:
        self.data.qpos[qpos_address : qpos_address + 3] = pose.position
        self.data.qpos[qpos_address + 3 : qpos_address + 7] = pose.quaternion
        self.data.qvel[dof_address : dof_address + 6] = 0.0

    def _sync_welded_free_body(
        self,
        *,
        parent_body_id: int,
        equality_id: int,
        qpos_address: int,
        dof_address: int,
    ) -> None:
        relative = Pose(
            np.asarray(self.model.eq_data[equality_id, 3:6], dtype=float),
            np.asarray(self.model.eq_data[equality_id, 6:10], dtype=float),
        )
        desired = self._body_pose(self.data, parent_body_id).transformed(relative)
        self._write_free_pose(qpos_address, dof_address, desired)

    def _apply_v1_joint_playback(
        self,
        arm_name: str,
        controller: ArmController,
        plan: _JointPlan,
        command: np.ndarray,
    ) -> None:
        """Apply one deterministic V1-style joint sample and rigid payload pose."""

        bounded = np.clip(command, controller.lower, controller.upper)
        controller.q_command = bounded
        controller.locked_local_indices = (6,) if plan.locked_joint7_rad is not None else ()
        controller.full_orientation = True
        controller.enabled = False
        self.data.qpos[controller.qpos_ids] = bounded
        self.data.qvel[controller.dof_ids] = 0.0
        self.data.ctrl[controller.actuator_ids] = bounded
        self.mujoco.mj_kinematics(self.model, self.data)

        if arm_name == "arm1":
            mounted = self._arm1_tools.current_tool
            if mounted is not None:
                self._sync_welded_free_body(
                    parent_body_id=self._arm1_tools.link7_id,
                    equality_id=self._arm1_tools.arm_weld_ids[mounted],
                    qpos_address=self._arm1_tools.qpos_addresses[mounted],
                    dof_address=self._arm1_tools.dof_addresses[mounted],
                )
        else:
            parent, _body, qpos, dof, equality = self._fixed_tool_state[arm_name]
            self._sync_welded_free_body(
                parent_body_id=parent,
                equality_id=equality,
                qpos_address=qpos,
                dof_address=dof,
            )
        self.mujoco.mj_kinematics(self.model, self.data)

        if plan.proxy_key is not None:
            proxy = self._proxies[plan.proxy_key]
            if proxy.held:
                self._sync_welded_free_body(
                    parent_body_id=proxy.tool_body_id,
                    equality_id=proxy.grasp_weld_id,
                    qpos_address=proxy.qpos_address,
                    dof_address=proxy.dof_address,
                )
                self.mujoco.mj_kinematics(self.model, self.data)

    def _interaction_complete(
        self,
        arm_name: str,
        plan: _JointPlan,
        timestep: float,
    ) -> bool:
        if plan.operation_kind == "TOOL_CHANGE":
            interaction = plan.waypoints[plan.waypoint_index].interaction
            if interaction.startswith("tool_return:"):
                tool_name = interaction.partition(":")[2]
                if self._arm1_tools.current_tool == tool_name:
                    self._arm1_tools.undock(tool_name)
            elif interaction.startswith("tool_dock:"):
                tool_name = interaction.partition(":")[2]
                if self._arm1_tools.current_tool != tool_name:
                    self._arm1_tools.dock(tool_name)
            return True
        if plan.operation_kind not in {"BASE_LOADING", "INSTALL_FIN"}:
            return True
        index = plan.waypoint_index
        if plan.operation_kind == "BASE_LOADING":
            if index == plan.grasp_waypoint_index:
                if not plan.interaction_started:
                    plan.interaction_started = True
                    plan.interaction_elapsed_s = 0.0
                plan.interaction_elapsed_s += timestep
                fraction = min(
                    1.0,
                    plan.interaction_elapsed_s / self.BASE_SUCTION_ENGAGE_SECONDS,
                )
                self._set_arm1_suction_fraction(quintic_time_scaling(fraction))
                self._align_base_proxy_with_suction(plan, fraction)
                if fraction < 1.0:
                    return False
                self._grasp_proxy(plan)
                plan.grasp_verified = True
                self._measured_milestones.add((plan.operation_key, "grasp"))
            elif index == plan.release_waypoint_index:
                if not plan.interaction_started:
                    plan.interaction_started = True
                    plan.interaction_elapsed_s = 0.0
                plan.interaction_elapsed_s += timestep
                fraction = min(
                    1.0,
                    plan.interaction_elapsed_s / self.BASE_SUCTION_RELEASE_SECONDS,
                )
                self._set_arm1_suction_fraction(1.0 - quintic_time_scaling(fraction))
                if fraction < 1.0:
                    return False
                self._reveal_installed_base(plan)
                self._release_proxy(plan)
                plan.release_verified = True
                self._measured_milestones.add((plan.operation_key, "release"))
            return True
        interaction = plan.waypoints[index].interaction
        if interaction not in {"open", "grasp", "release"}:
            return True
        closing = interaction == "grasp"
        pick_failure = bool(
            closing and plan.manifested_fault_type == "FIN_PICK_FAILED" and not plan.rework_fin
        )
        if interaction == "open":
            target = 0.0
            duration = self.GRIPPER_RELEASE_SECONDS
        elif closing:
            if plan.fin_clamp_position_m is None:
                raise RuntimeError("翅片夹紧缺少厚度驱动的夹爪目标")
            target = (
                max(
                    0.0,
                    plan.fin_clamp_position_m - 0.5 * self.FIN_PICK_FAILURE_EXTRA_GAP_M,
                )
                if pick_failure
                else plan.fin_clamp_position_m
            )
            duration = self.GRIPPER_CLOSE_SECONDS
        else:
            # As in V1, release with only a small finger stroke while still
            # inside the comb.  Full opening happens at the next high pickup
            # pose, preventing a finger from sweeping through adjacent fins.
            if plan.fin_clamp_position_m is None:
                raise RuntimeError("翅片释放缺少厚度驱动的夹爪目标")
            target = (
                0.0
                if plan.grasp_failed
                else max(
                    0.0,
                    plan.fin_clamp_position_m - 0.5 * self.FIN_RELEASE_CLEARANCE_M,
                )
            )
            duration = self.GRIPPER_RELEASE_SECONDS
        positions = np.asarray(
            [self.data.qpos[qpos_id] for qpos_id in self._finger_qpos[arm_name]],
            dtype=float,
        )
        if not plan.interaction_started:
            plan.interaction_started = True
            plan.interaction_elapsed_s = 0.0
            plan.interaction_start_fraction = float(np.mean(positions))
        plan.interaction_elapsed_s += timestep
        fraction = min(1.0, plan.interaction_elapsed_s / max(duration, 1.0e-9))
        start = float(plan.interaction_start_fraction or 0.0)
        command = start + quintic_time_scaling(fraction) * (target - start)
        for actuator_id in self._finger_actuators[arm_name]:
            self.data.ctrl[actuator_id] = command
        # V1-compatible deterministic finger playback: preserve the gradual
        # quintic motion, but place both symmetric fingers at the same authored
        # position so contact is governed by fin thickness rather than an
        # actuator/load equilibrium offset.
        for qpos_id, dof_id in zip(
            self._finger_qpos[arm_name],
            self._finger_dof[arm_name],
        ):
            self.data.qpos[qpos_id] = command
            self.data.qvel[dof_id] = 0.0
        self.mujoco.mj_kinematics(self.model, self.data)
        positions = np.full(len(self._finger_qpos[arm_name]), command, dtype=float)
        if fraction < 1.0:
            return False
        if float(np.max(np.abs(positions - target))) > self.FINGER_CONTACT_TOLERANCE_M:
            return False
        if closing:
            if pick_failure:
                plan.grasp_failed = True
                self._restore_failed_pick_to_source(plan)
            else:
                self._grasp_proxy(plan)
                plan.grasp_verified = True
                self._measured_milestones.add((plan.operation_key, "grasp"))
        elif interaction == "release":
            if plan.grasp_failed:
                self._restore_failed_pick_to_source(plan)
            else:
                # Make the tray-owned fin visible before the temporary grasp
                # proxy is hidden.  Both occupy the same authored pose, so this
                # is an atomic visual ownership handoff rather than a one-frame
                # pop.
                self._reveal_installed_fin(plan)
                self._release_proxy(plan)
            plan.release_verified = True
            self._measured_milestones.add((plan.operation_key, "release"))
        return True

    def sync(self, runtime: "DualLineRuntime") -> None:
        self._sync_active_fin_faults(runtime)
        for arm_name, controller in self.controllers.items():
            operation = runtime.operations.get(arm_name.upper())
            resource_paused = not runtime.faults.resource_available(arm_name.upper())
            if resource_paused:
                self._paused_arms.add(arm_name)
                held = self._paused_joint_positions.setdefault(
                    arm_name,
                    np.asarray(self.data.qpos[controller.qpos_ids], dtype=float).copy(),
                )
                self.data.qpos[controller.qpos_ids] = held
                self.data.qvel[controller.dof_ids] = 0.0
                controller.hold()
                self.data.ctrl[controller.actuator_ids] = held
                self._target_label[arm_name] = "故障隔离：保持当前安全位姿"
                continue
            self._paused_arms.discard(arm_name)
            self._paused_joint_positions.pop(arm_name, None)
            if operation is None:
                idle_transform = {
                    "arm2": "arm2_dispenser",
                    "arm3": "arm3_camera",
                }.get(arm_name)
                if idle_transform is not None:
                    controller.set_tool_transform(self._tool_transforms[idle_transform])
                if self._active_operation[arm_name]:
                    prior_plan = self._plans[arm_name]
                    if prior_plan is not None and prior_plan.proxy_key is not None:
                        self._release_proxy(prior_plan)
                    controller.hold()
                    self.data.ctrl[controller.actuator_ids] = np.asarray(
                        self.data.qpos[controller.qpos_ids],
                        dtype=float,
                    )
                self._plans[arm_name] = None
                self._active_operation[arm_name] = ""
                self._target_label[arm_name] = "等待任务"
                continue
            key = f"{operation.unit_id}:{operation.kind}:" f"{float(operation.started_at):.9f}"
            if self._active_operation[arm_name] != key:
                unit = runtime.units[operation.unit_id]
                required_tool = self._required_tool(arm_name, operation.kind)
                if required_tool is not None and self._arm1_tools.current_tool != required_tool:
                    self._plans[arm_name] = self._build_arm1_tool_change_plan(
                        operation,
                        required_tool,
                        fast=bool(runtime.fast),
                    )
                else:
                    self._plans[arm_name] = self._build_plan(
                        arm_name,
                        operation,
                        unit,
                        fast=bool(runtime.fast),
                    )
                self._active_operation[arm_name] = key
            plan = self._plans[arm_name]
            if (
                plan is not None
                and plan.operation_kind == "TOOL_CHANGE"
                and plan.complete
                and not plan.failure
                and self._arm1_tools.current_tool == plan.next_tool
            ):
                unit = runtime.units[operation.unit_id]
                self._plans[arm_name] = self._build_plan(
                    arm_name,
                    operation,
                    unit,
                    fast=bool(runtime.fast),
                )
                plan = self._plans[arm_name]
            if (
                plan is not None
                and operation.kind == "INSTALL_FIN"
                and not operation.recovery
                and plan.fin_index is not None
            ):
                manifested = self._active_fin_faults.get(
                    (operation.unit_id, plan.fin_index),
                    "",
                )
                if manifested:
                    newly_manifested = not plan.manifested_fault_type
                    if (
                        newly_manifested
                        and manifested in self.LATERAL_FIN_FAULT_TYPES
                        and not plan.grasp_verified
                        and not plan.interaction_started
                    ):
                        # The runtime starts the physical operation one tick
                        # before its latent defect request can match it.  The
                        # first plan therefore targets the nominal slot.  At
                        # this point the fin is still at the magazine, so it is
                        # safe to rebuild from the measured arm pose and author
                        # the complete carry/descent path toward the offset
                        # slot miss instead of moving the settled fin later.
                        unit = runtime.units[operation.unit_id]
                        self._plans[arm_name] = self._build_plan(
                            arm_name,
                            operation,
                            unit,
                            fast=bool(runtime.fast),
                        )
                        plan = self._plans[arm_name]
                    plan.manifested_fault_type = manifested
            if plan is not None and plan.failure:
                self._target_label[arm_name] = plan.failure
            elif plan is not None and plan.complete:
                self._target_label[arm_name] = "物理动作完成，等待调度确认"
            elif plan is not None:
                waypoint = plan.waypoints[plan.waypoint_index]
                controller.set_tool_transform(self._tool_transforms[plan.tool_name])
                controller.set_target(waypoint.pose, tcp=True)
                self._target_label[arm_name] = waypoint.label_zh

    def control_tick(self, dt: float) -> None:
        timestep = float(dt)
        for arm_name, controller in self.controllers.items():
            if arm_name in self._paused_arms:
                held = self._paused_joint_positions[arm_name]
                self.data.qpos[controller.qpos_ids] = held
                self.data.qvel[controller.dof_ids] = 0.0
                self.data.ctrl[controller.actuator_ids] = held
                continue
            plan = self._plans[arm_name]
            if plan is None or plan.failure:
                continue
            if plan.active_path_start is not None:
                command, path_complete = self._continuous_path_command(plan, timestep)
                self._apply_v1_joint_playback(arm_name, controller, plan, command)
                if not path_complete:
                    continue
                assert plan.active_path_end is not None
                plan.waypoint_index = plan.active_path_end
                plan.active_path_start = None
                plan.active_path_end = None
                plan.active_path_cumulative = None
                plan.segment_elapsed_s = plan.segment_duration_s
            goal = plan.joint_goals[plan.waypoint_index]
            if plan.active_path_start is None and plan.segment_elapsed_s < plan.segment_duration_s:
                plan.segment_elapsed_s += timestep
            fraction = min(1.0, plan.segment_elapsed_s / plan.segment_duration_s)
            command = plan.segment_start + quintic_time_scaling(fraction) * (goal - plan.segment_start)
            self._apply_v1_joint_playback(arm_name, controller, plan, command)
            qpos = np.asarray(self.data.qpos[controller.qpos_ids], dtype=float)
            qvel = np.asarray(self.data.qvel[controller.dof_ids], dtype=float)
            settled = (
                fraction >= 1.0
                and float(np.max(np.abs(goal - qpos))) <= self.JOINT_SETTLE_RAD
                and float(np.max(np.abs(qvel))) <= self.JOINT_SETTLE_SPEED_RAD_S
            )
            if not settled:
                continue
            actual_tcp = pose_from_site(
                self.data,
                self._tool_tcp_sites[plan.tool_name],
            )
            position_error, orientation_error = self._pose_errors(
                actual_tcp,
                plan.waypoints[plan.waypoint_index].pose,
            )
            if position_error > 0.003 or orientation_error > math.radians(3.0):
                continue
            if not self._interaction_complete(arm_name, plan, timestep):
                continue
            if plan.waypoints[plan.waypoint_index].interaction:
                plan.interaction_started = False
                plan.interaction_elapsed_s = 0.0
                plan.interaction_start_fraction = None
            continuous_end = next(
                (end for start, end in plan.continuous_ranges if start == plan.waypoint_index),
                None,
            )
            if continuous_end is not None:
                self._start_continuous_path(
                    plan,
                    plan.waypoint_index,
                    continuous_end,
                )
                continue
            if plan.waypoint_index + 1 >= len(plan.joint_goals):
                if plan.operation_kind == "BASE_LOADING":
                    plan.release_verified = True
                plan.complete = True
                continue
            plan.waypoint_index += 1
            plan.segment_start = qpos.copy()
            plan.segment_elapsed_s = 0.0
            plan.segment_duration_s = self._segment_duration(
                plan.segment_start,
                plan.joint_goals[plan.waypoint_index],
                minimum_s=plan.minimum_segment_s,
                speed_scale=plan.motion_speed_scale,
            )

    def enforce_paused_state(self) -> None:
        """Project a faulted arm back onto its certified frozen joint state.

        MuJoCo's position actuator can move a few millimetres while dissipating
        pre-fault inertia.  A safety isolation is stricter than ordinary servo
        settling, so the held state is re-applied after every physics substep.
        """

        if not self._paused_joint_positions:
            return
        for arm_name, held in self._paused_joint_positions.items():
            controller = self.controllers[arm_name]
            self.data.qpos[controller.qpos_ids] = held
            self.data.qvel[controller.dof_ids] = 0.0
            self.data.ctrl[controller.actuator_ids] = held
        self.mujoco.mj_forward(self.model, self.data)

    def operation_complete(self, resource: str, unit_id: str, kind: str) -> bool:
        arm_name = str(resource).strip().lower()
        plan = self._plans.get(arm_name)
        interaction_verified = bool(
            plan is not None
            and (
                kind not in {"BASE_LOADING", "INSTALL_FIN"}
                or ((plan.grasp_verified or plan.grasp_failed) and plan.release_verified)
            )
        )
        if plan is not None and plan.proxy_key is not None:
            proxy = self._proxies[plan.proxy_key]
            interaction_verified = bool(
                interaction_verified
                and not proxy.held
                and not proxy.visible
                and not self.data.eq_active[proxy.grasp_weld_id]
            )
        tool_verified = True
        if arm_name == "arm1" and plan is not None:
            if plan.operation_kind == "TOOL_CHANGE":
                tool_verified = False
            else:
                tool_verified = bool(
                    self._arm1_tools.current_tool == plan.tool_name
                    and self.data.eq_active[self._arm1_tools.arm_weld_ids[plan.tool_name]]
                )
        elif arm_name == "arm3" and kind == "INSTALL_FIN":
            tool_verified = bool(self.data.eq_active[self._arm3_tool_weld])
        final_pose_verified = False
        if plan is not None and plan.waypoints:
            actual_tcp = pose_from_site(
                self.data,
                self._tool_tcp_sites[plan.tool_name],
            )
            position_error, orientation_error = self._pose_errors(
                actual_tcp,
                plan.waypoints[-1].pose,
            )
            final_pose_verified = bool(position_error <= 0.003 and orientation_error <= math.radians(3.0))
        return bool(
            plan is not None
            and plan.operation_key == f"{unit_id}:{kind}"
            and plan.operation_kind == kind
            and plan.complete
            and not plan.failure
            and interaction_verified
            and tool_verified
            and final_pose_verified
        )

    def operation_milestone(
        self,
        resource: str,
        unit_id: str,
        kind: str,
        milestone: str,
    ) -> bool:
        """Expose measured grasp/release milestones to fine-grained DAG tasks."""

        arm_name = str(resource).strip().lower()
        plan = self._plans.get(arm_name)
        operation_key = f"{unit_id}:{kind}"
        key = str(milestone).strip().lower()
        if (operation_key, key) in self._measured_milestones:
            return True
        if plan is None or plan.operation_key != operation_key or plan.operation_kind != kind:
            return False
        if key == "grasp":
            return bool(plan.grasp_verified)
        if key == "release":
            return bool(plan.release_verified)
        if key == "settled":
            return self.operation_complete(resource, unit_id, kind)
        raise ValueError(f"unknown physical operation milestone: {milestone}")

    def operation_progress(
        self,
        resource: str,
        unit_id: str,
        kind: str,
    ) -> float:
        """Return measured waypoint progress for visual process projection."""

        arm_name = str(resource).strip().lower()
        plan = self._plans.get(arm_name)
        if plan is None or plan.operation_key != f"{unit_id}:{kind}" or not plan.joint_goals:
            return 0.0
        if plan.complete:
            return 1.0
        segment_fraction = min(
            1.0,
            max(
                0.0,
                plan.segment_elapsed_s / max(plan.segment_duration_s, 1.0e-9),
            ),
        )
        if kind == "DISPENSING":
            points_per_pass = max(2, plan.deposition_points_per_pass)
            pass_count = max(1, len(plan.joint_goals) // points_per_pass)
            active_pass = min(
                pass_count - 1,
                plan.waypoint_index // points_per_pass,
            )
            completed_passes = active_pass
            active_fraction = 0.0
            within_pass = plan.waypoint_index % points_per_pass
            line_start_offset = plan.deposition_line_start_offset
            line_end_offset = plan.deposition_line_end_offset
            if within_pass > line_end_offset:
                active_fraction = 1.0
            elif within_pass >= line_start_offset:
                first_index = active_pass * points_per_pass + line_start_offset
                final_index = active_pass * points_per_pass + line_end_offset
                start = plan.waypoints[first_index].pose.position
                end = plan.waypoints[final_index].pose.position
                actual = pose_from_site(
                    self.data,
                    self._tool_tcp_sites[plan.tool_name],
                ).position
                direction = end - start
                length_squared = float(np.dot(direction, direction))
                if length_squared > 1.0e-12:
                    projected_fraction = float(
                        np.clip(np.dot(actual - start, direction) / length_squared, 0.0, 1.0)
                    )
                    nearest = start + projected_fraction * direction
                    # The yellow bead must follow measured nozzle contact, not
                    # merely the commanded waypoint.  During a fast approach
                    # the TCP can still be a few millimetres above the board;
                    # hold the last visual fraction until it is back on path.
                    if float(np.linalg.norm(actual - nearest)) <= 0.005:
                        active_fraction = projected_fraction
            measured_progress = min(
                1.0,
                (completed_passes + active_fraction) / pass_count,
            )
            plan.reported_progress = max(plan.reported_progress, measured_progress)
            return plan.reported_progress
        completed_segments = float(plan.waypoint_index)
        measured_progress = min(
            1.0,
            (completed_segments + segment_fraction) / len(plan.joint_goals),
        )
        plan.reported_progress = max(plan.reported_progress, measured_progress)
        return plan.reported_progress

    def reset(self) -> None:
        """Release transient workpiece ownership and return Arm1 tools safely."""

        for plan in self._plans.values():
            if plan is not None and plan.proxy_key is not None:
                self._release_proxy(plan)
        for proxy in self._proxies.values():
            self.data.eq_active[proxy.feed_weld_id] = 0
            self.data.eq_active[proxy.grasp_weld_id] = 0
            proxy.held = False
            self._set_proxy_visible(proxy, False)
        for branch, index in self._raw_fin_rgba:
            self._set_raw_fin_visible(branch, index, False)
        for actuator_ids in self._finger_actuators.values():
            for actuator_id in actuator_ids:
                self.data.ctrl[actuator_id] = 0.0
        self._arm1_tools.reset_to_rack()
        self._paused_arms.clear()
        self._paused_joint_positions.clear()
        self._measured_milestones.clear()
        for arm_name, controller in self.controllers.items():
            controller.hold()
            self._plans[arm_name] = None
            self._active_operation[arm_name] = ""
            self._target_label[arm_name] = "等待任务"
        self.mujoco.mj_forward(self.model, self.data)

    @staticmethod
    def _pose_errors(current: Pose, target: Pose) -> tuple[float, float]:
        position = float(np.linalg.norm(target.position - current.position))
        relative = current.rotation.T @ target.rotation
        cosine = float(np.clip(0.5 * (np.trace(relative) - 1.0), -1.0, 1.0))
        return position, math.acos(cosine)

    def snapshot(self) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for arm_name, controller in self.controllers.items():
            plan = self._plans[arm_name]
            finger_inner_gap_m = None
            if arm_name in self._finger_qpos:
                finger_inner_gap_m = self._finger_open_gap_m[arm_name] - sum(
                    float(self.data.qpos[qpos_id]) for qpos_id in self._finger_qpos[arm_name]
                )
            if plan is None:
                position_error, orientation_error = 0.0, 0.0
                rigid_position_error, rigid_orientation_error = 0.0, 0.0
                waypoint_index, waypoint_count = 0, 0
                complete, failure, progress = False, "", 0.0
            else:
                target_index = min(plan.waypoint_index, len(plan.waypoints) - 1)
                actual_tcp = pose_from_site(
                    self.data,
                    self._tool_tcp_sites[plan.tool_name],
                )
                position_error, orientation_error = self._pose_errors(
                    actual_tcp,
                    plan.waypoints[target_index].pose,
                )
                virtual_tcp = controller.current_tcp_pose()
                rigid_position_error, rigid_orientation_error = self._pose_errors(
                    actual_tcp,
                    virtual_tcp,
                )
                waypoint_index = plan.waypoint_index + 1
                waypoint_count = len(plan.waypoints)
                complete, failure = plan.complete, plan.failure
                unit_id, operation_kind = plan.operation_key.rsplit(":", 1)
                progress = self.operation_progress(
                    arm_name,
                    unit_id,
                    operation_kind,
                )
            result[arm_name] = {
                "mode": "V1_COMPATIBLE_JOINT_PLAYBACK",
                "planner": "POSE_LOCKED_CARTESIAN_QUINTIC",
                "operation": self._active_operation[arm_name],
                "target_zh": (
                    self._target_label[arm_name]
                    if arm_name in self._paused_arms
                    else (
                        "S3B 抓取失败：夹爪保持大间隙并空载前往槽位"
                        if plan is not None and plan.grasp_failed and not plan.release_verified
                        else (
                            plan.waypoints[min(plan.waypoint_index, len(plan.waypoints) - 1)].label_zh
                            if plan is not None and not plan.failure and not plan.complete
                            else self._target_label[arm_name]
                        )
                    )
                ),
                "joint_positions": np.asarray(
                    self.data.qpos[controller.qpos_ids],
                    dtype=float,
                ).tolist(),
                "position_error_m": position_error,
                "orientation_error_rad": orientation_error,
                "tcp_rigid_error_m": rigid_position_error,
                "tcp_rigid_orientation_error_rad": rigid_orientation_error,
                "actual_tcp_position_m": (
                    pose_from_site(
                        self.data,
                        self._tool_tcp_sites[plan.tool_name],
                    ).position.tolist()
                    if plan is not None
                    else []
                ),
                "target_tcp_position_m": (
                    plan.waypoints[min(plan.waypoint_index, len(plan.waypoints) - 1)].pose.position.tolist()
                    if plan is not None
                    else []
                ),
                "waypoint_index": waypoint_index,
                "waypoint_count": waypoint_count,
                "progress": progress,
                "physical_complete": complete,
                "grasp_verified": False if plan is None else plan.grasp_verified,
                "grasp_failed": False if plan is None else plan.grasp_failed,
                "manifested_fault_type": "" if plan is None else plan.manifested_fault_type,
                "release_verified": False if plan is None else plan.release_verified,
                "finger_inner_gap_m": finger_inner_gap_m,
                "fin_thickness_m": None if plan is None else plan.fin_thickness_m,
                "fin_clamp_position_m": None if plan is None else plan.fin_clamp_position_m,
                "installed_fin_revealed": bool(plan is not None and plan.installed_fin_revealed),
                "fin_index": None if plan is None else plan.fin_index,
                "workpiece_proxy": None if plan is None else plan.proxy_key,
                "workpiece_held": bool(
                    plan is not None and plan.proxy_key is not None and self._proxies[plan.proxy_key].held
                ),
                "tool_state": (
                    self._arm1_tools.state
                    if arm_name == "arm1"
                    else {"current_tool": plan.tool_name if plan is not None else None}
                ),
                "failure": failure,
            }
        return result


__all__ = ["V2RobotMotionProjector"]
