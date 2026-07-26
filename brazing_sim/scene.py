"""MuJoCo scene contract, product configuration and runtime scene facade."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import derive_product_layout, make_order_spec
from .domain import BrazingPathState, FinState, FixtureState, OrderSpec, ProductState
from .fixture import FixtureController
from .layout import SHALLOW_U_LAYOUT
from .motion import ArmController, MotionConfig, Pose, default_scene_path, matrix_to_quat
from .preflight import PreflightReport, preflight_check
from .tools import Arm1ToolManager, Arm2ToolManager

ARM_NAMES = ("arm1", "arm2", "arm3")
FIN_NAMES = tuple(f"fin_{index:02d}" for index in range(1, 13))
PATH_NAMES = tuple(
    f"slot_{index:02d}_{side}_brazing_path" for index in range(1, 13) for side in ("left", "right")
)
BATCH_TRAY_NAMES = tuple(f"batch_tray_{index:02d}" for index in range(1, 4))
BATCH_COMB_POST_HALF_X = 0.012
BATCH_COMB_POST_HALF_Y = 0.008
BATCH_COMB_GUIDE_HALF_X = 0.045
BATCH_COMB_LONGITUDINAL_CLEARANCE = 0.008
BATCH_COMB_LATERAL_CLEARANCE = 0.010
CHANGEOVER_MODULE_NAMES = tuple(
    f"changeover_module_{pitch}_{copy}" for pitch in (15, 20, 30, 40) for copy in ("a", "b")
)
CHANGEOVER_MOLD_NAMES = tuple(
    f"changeover_mold_{pitch}_{copy}" for pitch in (15, 20, 30, 40) for copy in ("a", "b")
)
CHANGEOVER_PRESS_NAMES = ("changeover_press_a", "changeover_press_b")
CHANGEOVER_COMPONENT_NAMES = (
    *CHANGEOVER_MODULE_NAMES,
    *CHANGEOVER_MOLD_NAMES,
    *CHANGEOVER_PRESS_NAMES,
)
ASYNC_STATION_ANCHORS = {
    "s1": "station_s1_anchor",
    "s2a": "station_s2a_anchor",
    "s2b": "station_s2b_anchor",
    "s3": "station_s3_anchor",
    "rack_infeed": "station_rack_infeed_anchor",
}
ASYNC_TRANSFER_SPECS = {
    "s1_s2a": ("transfer_s1_s2a_joint", "transfer_s1_s2a_actuator", 0.438634244),
    "s2a_s2b": ("transfer_s2a_s2b_joint", "transfer_s2a_s2b_actuator", 0.60),
    "s2b_s3": ("transfer_s2b_s3_joint", "transfer_s2b_s3_actuator", 0.438634244),
    "s3_rack": ("transfer_s3_rack_joint", "transfer_s3_rack_actuator", 0.52),
}
HOME_QPOS = np.asarray([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=float)
# The -2/-1 rad shoulder offsets belonged to the retired central turntable.
# Each shallow-U robot base is now yawed directly at its own station, so the
# canonical symmetric posture is the safest waiting/IK seed.
ARM2_WAIT_QPOS = HOME_QPOS.copy()
ARM3_WAIT_QPOS = HOME_QPOS.copy()


def _arm_home_qpos(arm_name: str) -> np.ndarray:
    """Return the certified idle posture for one physical workstation."""

    if arm_name == "arm2":
        return ARM2_WAIT_QPOS
    if arm_name == "arm3":
        return ARM3_WAIT_QPOS
    return HOME_QPOS


class SceneContractError(RuntimeError):
    """Raised when a required MJCF name is missing."""


def _pose_from_body(data: Any, body_id: int) -> Pose:
    body = data.body(body_id)
    return Pose(np.asarray(body.xpos, dtype=float), matrix_to_quat(np.asarray(body.xmat).reshape(3, 3)))


@dataclass(frozen=True)
class SceneHandles:
    arm_sites: Mapping[str, int]
    arm_targets: Mapping[str, int]
    fins: Mapping[str, int]
    paths: Mapping[str, int]
    welds: Mapping[str, int]
    cameras: Mapping[str, int]
    raw_sites: Mapping[str, int]
    furnace_door_joint: int
    furnace_door_actuator: int
    conveyor_slide_joint: int
    conveyor_slide_actuator: int


class SceneRegistry:
    """Resolve and mutate the stable names exported by ``brazing_line.xml``."""

    def __init__(self, model: Any, data: Any) -> None:
        import mujoco

        self.mujoco = mujoco
        self.model = model
        self.data = data
        self._free_qpos: dict[str, int] = {}
        self._body_ids: dict[str, int] = {}
        self._geom_ids: dict[str, int] = {}
        self._initial_rgba: dict[int, np.ndarray] = {}
        self._initial_site_rgba: dict[int, np.ndarray] = {}
        self._initial_geom_size: dict[int, np.ndarray] = {}
        self._initial_geom_pos: dict[int, np.ndarray] = {}
        self._initial_contype: dict[int, int] = {}
        self._initial_conaffinity: dict[int, int] = {}
        self._model_initial_rgba = np.asarray(model.geom_rgba, dtype=float).copy()
        self._model_initial_contype = np.asarray(model.geom_contype, dtype=int).copy()
        self._model_initial_conaffinity = np.asarray(model.geom_conaffinity, dtype=int).copy()
        self.active_fin_count = 5
        self.active_path_count = 10
        self.assembly_base_pose = Pose(np.asarray([0.0, 0.37, 0.240]), np.asarray([1.0, 0.0, 0.0, 0.0]))
        self.fin_local_targets: dict[str, Pose] = {}
        self.path_local_targets: dict[str, Pose] = {}
        self._batch_payload_active: dict[str, bool] = {}
        self._batch_path_geometry: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, float, float]] = {}
        self._batch_path_progress: dict[tuple[int, int], float] = {}
        self._batch_press_target_z: dict[int, float] = {}
        self._batch_press_target_x: dict[int, tuple[float, float]] = {}
        self._batch_comb_geometry: dict[int, tuple[float, tuple[float, ...], float, float]] = {}
        self._batch_comb_installed: dict[int, bool] = {}
        self._fault_path_slots: dict[str, int] = {}
        self._fault_segment_names = tuple(
            (f"fault_braze_segment_{index:02d}a", f"fault_braze_segment_{index:02d}b")
            for index in range(1, 9)
        )
        self.handles = self._resolve_contract()
        self._furnace_door_qpos = int(self.model.jnt_qposadr[self.handles.furnace_door_joint])
        self._furnace_door_limits = np.asarray(
            self.model.jnt_range[self.handles.furnace_door_joint], dtype=float
        ).copy()
        self.arm1_finger_actuators = (
            self._id(self.mujoco.mjtObj.mjOBJ_ACTUATOR, "arm1_left_finger_actuator"),
            self._id(self.mujoco.mjtObj.mjOBJ_ACTUATOR, "arm1_right_finger_actuator"),
        )
        self.arm1_finger_joints = (
            self._id(self.mujoco.mjtObj.mjOBJ_JOINT, "arm1_left_finger_joint"),
            self._id(self.mujoco.mjtObj.mjOBJ_JOINT, "arm1_right_finger_joint"),
        )
        self.arm1_suction_pad = self._id(self.mujoco.mjtObj.mjOBJ_GEOM, "arm1_suction_pad")
        self._suction_pad_rgba = np.asarray(model.geom_rgba[self.arm1_suction_pad], dtype=float).copy()
        for name in (
            "assembly_tray",
            "base_plate",
            *FIN_NAMES,
            *PATH_NAMES,
            *BATCH_TRAY_NAMES,
            *CHANGEOVER_COMPONENT_NAMES,
        ):
            self._register_free_body(name)
        self._initial_model_qpos0 = np.asarray(model.qpos0, dtype=float).copy()

    def _id(self, object_type: Any, name: str) -> int:
        identifier = int(self.mujoco.mj_name2id(self.model, object_type, name))
        if identifier < 0:
            raise SceneContractError(f"brazing_line.xml is missing required {object_type!s} {name!r}")
        return identifier

    def _resolve_contract(self) -> SceneHandles:
        mjt = self.mujoco.mjtObj
        arm_sites = {arm: self._id(mjt.mjOBJ_SITE, f"{arm}_attachment_site") for arm in ARM_NAMES}
        arm_targets = {arm: self._id(mjt.mjOBJ_BODY, f"{arm}_target") for arm in ARM_NAMES}
        fins = {name: self._id(mjt.mjOBJ_BODY, name) for name in FIN_NAMES}
        paths = {name: self._id(mjt.mjOBJ_BODY, name) for name in PATH_NAMES}
        required_welds = [
            "tray_fixture_weld",
            "base_tray_weld",
            "raw_base_rack_weld",
            "turntable_nest_a_assembly_tray_weld",
            "turntable_nest_b_assembly_tray_weld",
            "table2_handoff_assembly_tray_weld",
            "station_s1_assembly_tray_weld",
            "station_s2a_assembly_tray_weld",
            "station_s2b_assembly_tray_weld",
            "station_s3_assembly_tray_weld",
            "station_rack_infeed_assembly_tray_weld",
            "transfer_s1_s2a_assembly_tray_weld",
            "transfer_s2a_s2b_assembly_tray_weld",
            "transfer_s2b_s3_assembly_tray_weld",
            "transfer_s3_rack_assembly_tray_weld",
            "arm1_grasp_base",
            "furnace_tray_weld",
            "fixture_press_hold_weld",
            "fixture_press_drive_hold_weld",
            "arm1_toolchange_parallel_gripper",
            "arm1_toolchange_suction_tool",
            "arm1_rack_parallel_gripper",
            "arm1_rack_suction_tool",
            "arm2_dispenser_tool_weld",
        ]
        required_welds.extend(f"arm1_grasp_{name}" for name in FIN_NAMES)
        required_welds.extend(f"raw_{name}_rack_weld" for name in FIN_NAMES)
        required_welds.extend(f"{name}_fixture_weld" for name in FIN_NAMES)
        required_welds.extend(f"{name}_base_weld" for name in FIN_NAMES)
        required_welds.extend(f"{name}_base_weld" for name in PATH_NAMES)
        welds = {name: self._id(mjt.mjOBJ_EQUALITY, name) for name in required_welds}
        cameras = {
            "arm3_wrist_camera": self._id(mjt.mjOBJ_CAMERA, "arm3_wrist_camera"),
            "furnace_camera": self._id(mjt.mjOBJ_CAMERA, "furnace_camera"),
        }
        raw_sites = {"base_plate": self._id(mjt.mjOBJ_SITE, "raw_base_site")}
        raw_sites.update({name: self._id(mjt.mjOBJ_SITE, f"raw_{name}_site") for name in FIN_NAMES})
        return SceneHandles(
            arm_sites=arm_sites,
            arm_targets=arm_targets,
            fins=fins,
            paths=paths,
            welds=welds,
            cameras=cameras,
            raw_sites=raw_sites,
            furnace_door_joint=self._id(mjt.mjOBJ_JOINT, "furnace_door_joint"),
            furnace_door_actuator=self._id(mjt.mjOBJ_ACTUATOR, "furnace_door_actuator"),
            conveyor_slide_joint=self._id(mjt.mjOBJ_JOINT, "conveyor_slide_joint"),
            conveyor_slide_actuator=self._id(mjt.mjOBJ_ACTUATOR, "conveyor_slide_actuator"),
        )

    def _register_free_body(self, body_name: str) -> None:
        body_id = self._id(self.mujoco.mjtObj.mjOBJ_BODY, body_name)
        joint_address = int(self.model.body_jntadr[body_id])
        if joint_address < 0:
            raise SceneContractError(f"body {body_name!r} must have a free joint")
        joint_id = joint_address
        if int(self.model.jnt_type[joint_id]) != int(self.mujoco.mjtJoint.mjJNT_FREE):
            raise SceneContractError(f"body {body_name!r} root joint must be free")
        self._body_ids[body_name] = body_id
        self._free_qpos[body_name] = int(self.model.jnt_qposadr[joint_id])

    def body_id(self, name: str) -> int:
        if name not in self._body_ids:
            self._body_ids[name] = self._id(self.mujoco.mjtObj.mjOBJ_BODY, name)
        return self._body_ids[name]

    def geom_id(self, name: str) -> int:
        if name not in self._geom_ids:
            identifier = self._id(self.mujoco.mjtObj.mjOBJ_GEOM, name)
            self._geom_ids[name] = identifier
            self._initial_rgba.setdefault(
                identifier, np.asarray(self.model.geom_rgba[identifier], dtype=float).copy()
            )
            self._initial_geom_size.setdefault(
                identifier, np.asarray(self.model.geom_size[identifier], dtype=float).copy()
            )
            self._initial_geom_pos.setdefault(
                identifier, np.asarray(self.model.geom_pos[identifier], dtype=float).copy()
            )
            self._initial_contype.setdefault(identifier, int(self.model.geom_contype[identifier]))
            self._initial_conaffinity.setdefault(identifier, int(self.model.geom_conaffinity[identifier]))
        return self._geom_ids[name]

    def site_id(self, name: str) -> int:
        return self._id(self.mujoco.mjtObj.mjOBJ_SITE, name)

    def equality_id(self, name: str) -> int:
        if name in self.handles.welds:
            return int(self.handles.welds[name])
        return self._id(self.mujoco.mjtObj.mjOBJ_EQUALITY, name)

    def free_body_pose(self, body_name: str) -> Pose:
        return _pose_from_body(self.data, self.body_id(body_name))

    def set_free_body_pose(self, body_name: str, pose: Pose, *, forward: bool = False) -> None:
        try:
            address = self._free_qpos[body_name]
        except KeyError:
            self._register_free_body(body_name)
            address = self._free_qpos[body_name]
        self.data.qpos[address : address + 3] = pose.position
        self.data.qpos[address + 3 : address + 7] = pose.quaternion
        joint_id = int(self.model.body_jntadr[self.body_id(body_name)])
        dof = int(self.model.jnt_dofadr[joint_id])
        self.data.qvel[dof : dof + 6] = 0.0
        if forward:
            self.mujoco.mj_forward(self.model, self.data)

    def move_free_body_preserving_orientation(
        self,
        body_name: str,
        target: Pose | Sequence[float],
        *,
        forward: bool = False,
    ) -> None:
        """Move a free body to one XYZ target without changing its attitude.

        Furnace/rack sites inherit the furnace body's -90 degree layout
        rotation.  That rotation describes the mechanism's local coordinate
        system, not a requested pallet reorientation.  Logistics handoffs use
        this helper so a tray that enters lengthwise also leaves lengthwise.
        """

        current = self.free_body_pose(body_name)
        position = target.position if isinstance(target, Pose) else np.asarray(target, dtype=float)
        self.set_free_body_pose(
            body_name,
            Pose(np.asarray(position, dtype=float), current.quaternion.copy()),
            forward=forward,
        )

    def product_pose(self) -> Pose:
        return self.free_body_pose("base_plate")

    def product_to_world(self, local: Sequence[float] | Pose) -> np.ndarray | Pose:
        product = self.product_pose()
        if isinstance(local, Pose):
            return product.transformed(local)
        vector = np.asarray(local, dtype=float)
        if vector.shape != (3,):
            raise ValueError("product coordinate must be xyz")
        return product.position + product.rotation @ vector

    def world_to_product(self, world: Sequence[float] | Pose) -> np.ndarray | Pose:
        inverse = self.product_pose().inverse()
        if isinstance(world, Pose):
            return inverse.transformed(world)
        vector = np.asarray(world, dtype=float)
        if vector.shape != (3,):
            raise ValueError("world coordinate must be xyz")
        return inverse.rotation @ vector + inverse.position

    def _set_geom_active(self, geom_name: str, active: bool, *, collide: bool = True) -> None:
        geom_id = self.geom_id(geom_name)
        rgba = self._initial_rgba[geom_id].copy()
        rgba[3] = rgba[3] if active else 0.0
        self.model.geom_rgba[geom_id] = rgba
        self.model.geom_contype[geom_id] = self._initial_contype[geom_id] if active and collide else 0
        self.model.geom_conaffinity[geom_id] = self._initial_conaffinity[geom_id] if active and collide else 0

    def _set_runtime_geom_active(
        self,
        geom_name: str,
        active: bool,
        *,
        collide: bool = True,
    ) -> None:
        """Toggle a material-backed geom without mutating its shared material.

        MuJoCo material colour takes precedence over ``geom_rgba``.  Runtime
        fixture parts therefore receive a private copy of their material
        colour before their visibility/collision state is changed.
        """

        geom_id = self.geom_id(geom_name)
        material_id = int(self.model.geom_matid[geom_id])
        if material_id >= 0:
            self._initial_rgba[geom_id] = np.asarray(self.model.mat_rgba[material_id], dtype=float).copy()
            self.model.geom_matid[geom_id] = -1
        self._set_geom_active(geom_name, active, collide=collide)

    def _set_site_active(self, site_name: str, active: bool) -> None:
        site_id = self._id(self.mujoco.mjtObj.mjOBJ_SITE, site_name)
        self._initial_site_rgba.setdefault(
            site_id, np.asarray(self.model.site_rgba[site_id], dtype=float).copy()
        )
        rgba = self._initial_site_rgba[site_id].copy()
        rgba[3] = rgba[3] if active else 0.0
        self.model.site_rgba[site_id] = rgba

    def set_path_visible(
        self,
        path_id: str,
        visible: bool = True,
        *,
        coverage: float = 1.0,
        reverse: bool = False,
    ) -> None:
        name = path_id if path_id.endswith("_brazing_path") else f"{path_id}_brazing_path"
        geom_id = self.geom_id(name + "_geom")
        # Capsules compiled from ``fromto`` at their body origin receive a
        # MuJoCo same-frame optimisation. Runtime geom_pos edits are ignored
        # while that flag is active, which makes a resized capsule appear to
        # grow symmetrically from its midpoint. Disable the optimisation for
        # dynamic material markers so their centre can follow the live TCP
        # start-to-end direction.
        self.model.geom_sameframe[geom_id] = int(self.mujoco.mjtSameFrame.mjSAMEFRAME_NONE)
        rgba = self._initial_rgba[geom_id].copy()
        progress = float(np.clip(coverage, 0.0, 1.0)) if visible else 0.0
        rgba[3] = rgba[3] if progress > 0.0 else 0.0
        self.model.geom_rgba[geom_id] = rgba
        full_half_length = float(self._initial_geom_size[geom_id][1])
        current_half_length = max(1.0e-6, full_half_length * progress)
        self.model.geom_size[geom_id, 1] = current_half_length
        geom_pos = self._initial_geom_pos[geom_id].copy()
        # Capsules are X-aligned.  Alternating growth direction supports a
        # continuous serpentine pass without visually depositing backwards.
        direction = 1.0 if reverse else -1.0
        geom_pos[0] += direction * (full_half_length - current_half_length)
        self.model.geom_pos[geom_id] = geom_pos

    def _fault_path_slot(self, path_id: str) -> tuple[str, str]:
        normalized = path_id.removesuffix("_brazing_path")
        if normalized not in self._fault_path_slots:
            occupied = set(self._fault_path_slots.values())
            try:
                self._fault_path_slots[normalized] = next(
                    index for index in range(len(self._fault_segment_names)) if index not in occupied
                )
            except StopIteration as exc:
                raise RuntimeError("at most eight simultaneous visible brazing faults are supported") from exc
        return self._fault_segment_names[self._fault_path_slots[normalized]]

    def _show_fault_segment(
        self,
        geom_name: str,
        start: Sequence[float],
        end: Sequence[float],
        *,
        radius: float,
        rgba: Sequence[float],
    ) -> None:
        self._set_capsule_between(geom_name, start, end)
        geom_id = self.geom_id(geom_name)
        self.model.geom_size[geom_id, 0] = max(0.0005, float(radius))
        self.model.geom_rgba[geom_id] = np.asarray(rgba, dtype=float)

    def set_path_gap_visual(self, path: BrazingPathState, gap_m: float) -> None:
        """Replace one full bead by two real segments separated by a blank gap."""

        start = np.asarray(path.local_start, dtype=float)
        end = np.asarray(path.local_end, dtype=float)
        axis = end - start
        length = float(np.linalg.norm(axis))
        if length <= 1.0e-9:
            return
        direction = axis / length
        gap = float(np.clip(gap_m, 0.005, 0.80 * length))
        midpoint = 0.5 * (start + end)
        left_end = midpoint - 0.5 * gap * direction
        right_start = midpoint + 0.5 * gap * direction
        original = self.geom_id(path.name + "_geom")
        self.model.geom_rgba[original, 3] = 0.0
        left_name, right_name = self._fault_path_slot(path.path_id)
        colour = (0.98, 0.50, 0.04, 0.96)
        radius = path.target_width_m / 2.0
        self._show_fault_segment(left_name, start, left_end, radius=radius, rgba=colour)
        self._show_fault_segment(right_name, right_start, end, radius=radius, rgba=colour)

    def set_path_deviation_visual(self, path: BrazingPathState, lateral_error_m: float) -> None:
        """Show the actual shifted bead and a translucent red nominal reference."""

        self.set_path_visible(path.path_id, True, coverage=1.0)
        original = self.geom_id(path.name + "_geom")
        self.model.geom_pos[original, 1] = float(lateral_error_m)
        reference_name, unused_name = self._fault_path_slot(path.path_id)
        self._show_fault_segment(
            reference_name,
            path.local_start,
            path.local_end,
            radius=max(0.0006, path.target_width_m * 0.22),
            rgba=(1.0, 0.03, 0.02, 0.72),
        )
        self.model.geom_rgba[self.geom_id(unused_name), 3] = 0.0

    def clear_path_fault_visual(self, path: BrazingPathState) -> None:
        """Restore the ordinary single bead after physical rework completes."""

        normalized = path.path_id.removesuffix("_brazing_path")
        slot = self._fault_path_slots.pop(normalized, None)
        if slot is not None:
            for name in self._fault_segment_names[slot]:
                self.model.geom_rgba[self.geom_id(name), 3] = 0.0
        self.set_path_visible(
            path.path_id,
            bool(path.active and path.applied),
            coverage=path.coverage_ratio if path.applied else 0.0,
        )

    def reset_path_fault_visuals(self) -> None:
        self._fault_path_slots.clear()
        for names in self._fault_segment_names:
            for name in names:
                self.model.geom_rgba[self.geom_id(name), 3] = 0.0

    def configure_comb_module(self, spec: OrderSpec) -> tuple[str, str]:
        """Activate exactly one matching front/rear comb insert pair.

        All three pitch variants remain preallocated in MJCF so A/B/C can be
        switched without rebuilding the viewer.  Inactive modules are both
        invisible and non-colliding.
        """

        suffix = spec.comb_module_name.removeprefix("comb_insert_")
        if suffix not in {"15mm", "20mm", "30mm", "40mm"}:
            raise ValueError(f"unsupported comb module {spec.comb_module_name!r}")
        pitch_token = suffix.removesuffix("mm")
        for side in ("front", "rear"):
            for candidate in ("15", "20", "30", "40"):
                prefixes = (
                    f"{side}_comb_insert_{candidate}mm_",
                    f"{side}_comb_{candidate}_",
                )
                active = candidate == pitch_token
                for geom_id in range(int(self.model.ngeom)):
                    geom_name = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                    if geom_name and geom_name.startswith(prefixes):
                        self._set_runtime_geom_active(geom_name, active, collide=active)
        # Each module exposes a maximum physical slot pool. Re-centre the
        # active pairs from the order's product coordinates and hide surplus
        # teeth so 5/7-fin products do not show floating unused slots.
        capacities = {"15": 9, "20": 7, "30": 5, "40": 3}
        y_positions = [fin.target_position[1] for fin in derive_product_layout(spec).active_fins]
        half_opening = max(spec.fin_thickness / 2.0 + 0.001, 0.0025)
        for side in ("front", "rear"):
            for index in range(1, capacities[pitch_token] + 1):
                active = index <= len(y_positions)
                centre_y = y_positions[index - 1] if active else 0.20 + 0.01 * index
                for suffix_name, sign in (("l", -1.0), ("r", 1.0)):
                    name = f"{side}_comb_{pitch_token}_g{index:02d}{suffix_name}"
                    geom_id = self.geom_id(name)
                    self.model.geom_pos[geom_id, 1] = centre_y + sign * half_opening
                    self._set_runtime_geom_active(name, active, collide=active)
        self.mujoco.mj_forward(self.model, self.data)
        return f"front_comb_insert_{suffix}", f"rear_comb_insert_{suffix}"

    def deactivate_comb_modules(self) -> None:
        """Park every replaceable insert until CONFIGURE_COMB completes."""

        for side in ("front", "rear"):
            for candidate in ("15", "20", "30", "40"):
                prefixes = (
                    f"{side}_comb_insert_{candidate}mm_",
                    f"{side}_comb_{candidate}_",
                )
                for geom_id in range(int(self.model.ngeom)):
                    geom_name = self.mujoco.mj_id2name(
                        self.model,
                        self.mujoco.mjtObj.mjOBJ_GEOM,
                        geom_id,
                    )
                    if geom_name and geom_name.startswith(prefixes):
                        self._set_runtime_geom_active(geom_name, False, collide=False)
        self.mujoco.mj_forward(self.model, self.data)

    def set_press_installed(self, installed: bool) -> None:
        """Install/remove the upper press at its explicit process boundary.

        A horizontal press cannot physically coexist with a top-loaded base
        and top-inserted fins.  The prompt explicitly permits an
        ``INSTALL_OR_ENABLE_UPPER_PLATE`` step, so the press is invisible and
        non-colliding until ``PRESS_FIXTURE`` starts, then becomes a real
        actuator-driven mechanism.
        """

        visual_parts = (
            "fixture_press_carriage",
            "fixture_front_press_bar",
            "fixture_rear_press_bar",
        )
        colliding_parts = {"fixture_front_press_bar", "fixture_rear_press_bar"}
        for geom_name in visual_parts:
            self._set_runtime_geom_active(
                geom_name,
                installed,
                collide=installed and geom_name in colliding_parts,
            )
        self._set_site_active("fixture_press_touch_site", installed)
        self.mujoco.mj_forward(self.model, self.data)

    def set_press_latched(self, latched: bool) -> None:
        """Stop contact-force ringing after the physical force check passes.

        The two bars remain visible and their settled pose is held by the two
        press welds.  Contacts are restored whenever the press is reopened so
        the next cycle still performs a real touch/force ramp.
        """

        for geom_name in ("fixture_front_press_bar", "fixture_rear_press_bar"):
            geom_id = self.geom_id(geom_name)
            if latched:
                self.model.geom_contype[geom_id] = 0
                self.model.geom_conaffinity[geom_id] = 0
            else:
                self.model.geom_contype[geom_id] = self._initial_contype[geom_id]
                self.model.geom_conaffinity[geom_id] = self._initial_conaffinity[geom_id]
        self.mujoco.mj_forward(self.model, self.data)

    def _write_weld_relative(self, equality_id: int, body1: str, body2: str) -> None:
        left = _pose_from_body(self.data, self.body_id(body1))
        right = _pose_from_body(self.data, self.body_id(body2))
        relative = left.inverse().transformed(right)
        self.model.eq_data[equality_id, :] = 0.0
        self.model.eq_data[equality_id, 3:6] = relative.position
        self.model.eq_data[equality_id, 6:10] = relative.quaternion
        self.model.eq_data[equality_id, 10] = 1.0

    def set_weld(
        self,
        name: str,
        active: bool,
        *,
        recompute: tuple[str, str] | None = None,
        forward: bool = False,
    ) -> None:
        equality_id = self.equality_id(name)
        if active and recompute is not None:
            self.mujoco.mj_forward(self.model, self.data)
            self._write_weld_relative(equality_id, *recompute)
        self.data.eq_active[equality_id] = 1 if active else 0
        if forward:
            self.mujoco.mj_forward(self.model, self.data)

    def set_arm1_gripper_closed(self, fraction: float) -> None:
        """Animate both parallel fingers from open (0) to closed (1)."""

        amount = float(np.clip(fraction, 0.0, 1.0))
        for actuator in self.arm1_finger_actuators:
            self.data.ctrl[actuator] = 0.020 * amount

    def snap_arm1_gripper_closed(self, *, forward: bool = True) -> None:
        """Seat both finger inner faces exactly against a 2 mm fin.

        The visual close animation is servo driven and can still be a fraction
        of a millimetre short when its timer expires.  The final seating frame
        is therefore made deterministic before the rigid grasp weld is
        enabled: both 20 mm slides are stopped at their exact closed limit.
        """

        for joint, actuator in zip(self.arm1_finger_joints, self.arm1_finger_actuators):
            qpos = int(self.model.jnt_qposadr[joint])
            dof = int(self.model.jnt_dofadr[joint])
            self.data.qpos[qpos] = 0.020
            self.data.qvel[dof] = 0.0
            self.data.ctrl[actuator] = 0.020
        if forward:
            self.mujoco.mj_forward(self.model, self.data)

    def arm1_gripper_closed_fraction(self) -> float:
        values = [
            self.data.qpos[int(self.model.jnt_qposadr[joint])] / 0.020 for joint in self.arm1_finger_joints
        ]
        return float(np.clip(np.mean(values), 0.0, 1.0))

    def set_arm1_suction_fraction(self, fraction: float) -> None:
        """Visually compress and energize the base-plate suction pad."""

        amount = float(np.clip(fraction, 0.0, 1.0))
        rgba = self._suction_pad_rgba.copy()
        rgba[:3] = (1.0 - amount) * rgba[:3] + amount * np.asarray([0.08, 0.85, 0.95])
        self.model.geom_rgba[self.arm1_suction_pad] = rgba
        half_height = 0.004 - 0.0015 * amount
        self.model.geom_size[self.arm1_suction_pad, 1] = half_height
        self.model.geom_pos[self.arm1_suction_pad, 2] = 0.090 - half_height

    def arm1_suction_fraction(self) -> float:
        """Return the visual suction engagement in the range ``[0, 1]``."""

        half_height = float(self.model.geom_size[self.arm1_suction_pad, 1])
        return float(np.clip((0.004 - half_height) / 0.0015, 0.0, 1.0))

    def configure_product(
        self, order: OrderSpec | ProductState | str = "A"
    ) -> tuple[list[FinState], list[BrazingPathState]]:
        """Resize and configure the scene's fixed 12-fin/24-path allocation."""

        self.reset_path_fault_visuals()
        if isinstance(order, ProductState):
            spec = order.spec
            fins = order.fins
            paths = order.paths
        else:
            spec = make_order_spec(order) if isinstance(order, str) else order
            layout = derive_product_layout(spec)
            fins, paths = layout.fins, layout.paths

        if spec.max_fins > len(FIN_NAMES) or spec.max_paths > len(PATH_NAMES):
            raise ValueError("order exceeds the scene allocation of 12 fins / 24 paths")
        self.active_fin_count = spec.fin_count
        self.active_path_count = spec.path_count
        # +X is left-right along the Arm1-to-Arm3 direction.  Raw positions
        # come from the same shallow-U contract as the MJCF.  Six fins fit on
        # each indexed tier; the second tier preserves pickup clearance
        # without spreading long blanks into the finished-output conveyor.
        table_top_geom = self.geom_id("raw_material_rack_top")
        table_top_z = float(self.model.geom_pos[table_top_geom, 2] + self.model.geom_size[table_top_geom, 2])
        self.model.site_pos[self.handles.raw_sites["base_plate"]] = np.asarray(
            [
                *SHALLOW_U_LAYOUT.base_magazine_xy,
                table_top_z + 0.5 * spec.base_thickness,
            ]
        )
        for index, name in enumerate(FIN_NAMES):
            if index < spec.fin_count:
                position = SHALLOW_U_LAYOUT.raw_fin_position(
                    index,
                    spec.fin_count,
                    table_top_z=table_top_z,
                    fin_height_m=spec.fin_height,
                )
            else:
                # Inactive pool bodies remain isolated and invisible; their
                # sites are kept deterministic for diagnostics only.
                position = (0.32, 0.30 + 0.03 * (index - spec.fin_count), 0.13)
            self.model.site_pos[self.handles.raw_sites[name]] = np.asarray(position)
        base_geom = self.geom_id("heatsink_base_plate_geom")
        self.model.geom_size[base_geom, :3] = np.asarray(spec.base_size, dtype=float) / 2.0
        self._initial_geom_size[base_geom] = np.asarray(self.model.geom_size[base_geom], dtype=float).copy()
        # A completed batch temporarily hides the reusable Table2 workcell.
        # Product reconfiguration used to reactivate only the fin/path pools,
        # leaving the base alpha and collision masks at zero for the next
        # order.  Make the base an explicit member of every product setup.
        self._set_geom_active("heatsink_base_plate_geom", True, collide=True)
        self.mujoco.mj_forward(self.model, self.data)
        self.assembly_base_pose = self._site_pose(self.site_id("base_plate_target_site"))
        product_pose = self.assembly_base_pose
        self.fin_local_targets.clear()
        self.path_local_targets.clear()
        self.configure_comb_module(spec)
        self.configure_dispenser(spec)

        base_target_local = np.asarray(
            self.model.site_pos[self.site_id("base_plate_target_site")], dtype=float
        )

        for index, name in enumerate(FIN_NAMES):
            fin = fins[index]
            geom_name = name + "_geom"
            geom_id = self.geom_id(geom_name)
            # Per-slot visibility cannot be controlled reliably through one
            # shared material. Use a direct aluminium RGBA for the fin pool.
            self.model.geom_matid[geom_id] = -1
            self._initial_rgba[geom_id] = np.asarray([0.73, 0.76, 0.80, 1.0])
            self.model.geom_size[geom_id, :3] = np.asarray(spec.fin_size, dtype=float) / 2.0
            local_pose = Pose(np.asarray(fin.target_position, dtype=float), np.asarray([1.0, 0.0, 0.0, 0.0]))
            self.fin_local_targets[name] = local_pose
            slot_local = base_target_local + np.asarray(fin.target_position, dtype=float)
            fin_site = self.site_id(f"fin_slot_{index + 1:02d}_target")
            front_site = self.site_id(f"front_comb_slot_{index + 1:02d}")
            rear_site = self.site_id(f"rear_comb_slot_{index + 1:02d}")
            self.model.site_pos[fin_site] = slot_local
            self.model.site_pos[front_site] = np.asarray(
                [-spec.fin_length / 2.0 + 0.030, slot_local[1], base_target_local[2] + 0.045]
            )
            self.model.site_pos[rear_site] = np.asarray(
                [spec.fin_length / 2.0 - 0.030, slot_local[1], base_target_local[2] + 0.045]
            )
            self._set_site_active(f"fin_slot_{index + 1:02d}_target", bool(fin.active))
            self._set_site_active(f"front_comb_slot_{index + 1:02d}", bool(fin.active))
            self._set_site_active(f"rear_comb_slot_{index + 1:02d}", bool(fin.active))
            world_pose = product_pose.transformed(local_pose)
            self.set_weld(f"{name}_fixture_weld", False)
            self.set_weld(f"{name}_base_weld", False)
            self.set_weld(f"arm1_grasp_{name}", False)
            self.set_free_body_pose(name, world_pose)
            self._set_geom_active(geom_name, bool(fin.active), collide=bool(fin.collision_enabled))
            self._set_site_active(name + "_tcp", bool(fin.active))
            self._set_site_active(f"raw_{name}_site", bool(fin.active))
            if fin.active:
                self.mujoco.mj_forward(self.model, self.data)
                self.set_weld(
                    f"{name}_fixture_weld",
                    True,
                    recompute=("assembly_tray", name),
                )
                # Domain inspection metrics are expressed in product-local
                # coordinates. Keep physical world placement in MuJoCo only.
                fin.actual_position = tuple(float(value) for value in fin.target_position)

        for index, name in enumerate(PATH_NAMES):
            path = paths[index]
            midpoint = 0.5 * (
                np.asarray(path.local_start, dtype=float) + np.asarray(path.local_end, dtype=float)
            )
            local_pose = Pose(midpoint, np.asarray([1.0, 0.0, 0.0, 0.0]))
            self.path_local_targets[name] = local_pose
            world_pose = product_pose.transformed(local_pose)
            self.set_weld(f"{name}_base_weld", False)
            self.set_free_body_pose(name, world_pose)
            geom_name = name + "_geom"
            geom_id = self.geom_id(geom_name)
            self.model.geom_matid[geom_id] = -1
            self._initial_rgba[geom_id] = np.asarray([0.94, 0.55, 0.10, 0.90])
            self.model.geom_size[geom_id, 0] = float(path.target_width_m) / 2.0
            self.model.geom_size[geom_id, 1] = (
                abs(float(path.local_end[0]) - float(path.local_start[0])) / 2.0
            )
            self.model.geom_pos[geom_id] = np.zeros(3, dtype=float)
            self._initial_geom_size[geom_id] = np.asarray(self.model.geom_size[geom_id], dtype=float).copy()
            self._initial_geom_pos[geom_id] = np.asarray(self.model.geom_pos[geom_id], dtype=float).copy()
            self._set_geom_active(geom_name, bool(path.active), collide=False)
            self.set_path_visible(
                name, bool(path.active and path.applied), coverage=path.coverage_ratio or 1.0
            )
            if path.active:
                self.mujoco.mj_forward(self.model, self.data)
                self.set_weld(
                    f"{name}_base_weld",
                    True,
                    recompute=("base_plate", name),
                )

        self.mujoco.mj_forward(self.model, self.data)
        return list(fins), list(paths)

    def _set_capsule_between(self, name: str, start: Sequence[float], end: Sequence[float]) -> None:
        """Update one local capsule from two endpoints without rebuilding MJCF."""

        geom_id = self.geom_id(name)
        left = np.asarray(start, dtype=float)
        right = np.asarray(end, dtype=float)
        axis = right - left
        length = float(np.linalg.norm(axis))
        if length <= 1.0e-9:
            raise ValueError(f"capsule {name!r} has coincident endpoints")
        direction = axis / length
        z_axis = np.asarray([0.0, 0.0, 1.0])
        dot = float(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
        if dot < -1.0 + 1.0e-9:
            quaternion = np.asarray([0.0, 1.0, 0.0, 0.0])
        else:
            quaternion = np.asarray([1.0 + dot, *np.cross(z_axis, direction)], dtype=float)
            quaternion /= max(float(np.linalg.norm(quaternion)), 1.0e-12)
        self.model.geom_pos[geom_id] = 0.5 * (left + right)
        self.model.geom_quat[geom_id] = quaternion
        self.model.geom_size[geom_id, 1] = 0.5 * length

    def configure_dispenser(self, spec: OrderSpec) -> None:
        """Adjust Arm2's quick-change dual nozzle to this order's bead spacing."""

        half_spacing = spec.nozzle_spacing / 2.0
        nozzle_run = 0.030
        inward_run = nozzle_run * float(np.tan(np.deg2rad(35.0)))
        for side, sign in (("left", -1.0), ("right", 1.0)):
            tip = np.asarray([0.0, sign * half_spacing, 0.220])
            self.model.site_pos[self.site_id(f"arm2_{side}_nozzle_tip_site")] = tip
            self.model.geom_pos[self.geom_id(f"arm2_{side}_nozzle_tip")] = tip
            nozzle_start = np.asarray([0.0, sign * (half_spacing + inward_run), 0.190])
            self._set_capsule_between(f"arm2_{side}_nozzle", nozzle_start, tip)
            copper = nozzle_start + 0.8333333333 * (tip - nozzle_start)
            self.model.geom_pos[self.geom_id(f"arm2_{side}_nozzle_copper_tip")] = copper
        self.mujoco.mj_forward(self.model, self.data)

    def enable_robot_gravity_compensation(self) -> None:
        """Set gravcomp on every dynamic body in each attached FR3 subtree."""

        for arm in ARM_NAMES:
            for suffix in (
                "base",
                "fr3_link0",
                "fr3_link1",
                "fr3_link2",
                "fr3_link3",
                "fr3_link4",
                "fr3_link5",
                "fr3_link6",
                "fr3_link7",
            ):
                try:
                    body_id = int(self.model.body(f"{arm}_{suffix}").id)
                except KeyError:
                    continue
                self.model.body_gravcomp[body_id] = 1.0

    def prepare_raw_materials(
        self,
        product: ProductState | None = None,
    ) -> None:
        """Put the base and active fin blanks on Arm1's raw-material rack.

        All grasp/fixture/base welds are disabled and deposited braze paths are
        hidden. The order-domain positions remain product-local; this method
        changes only physical MuJoCo poses.
        """

        fins = product.fins if product is not None else None
        # Be defensive when this method follows a batch handoff: the base must
        # be visible and collidable before Arm1 approaches the raw rack.
        self._set_geom_active("heatsink_base_plate_geom", True, collide=True)
        self.set_weld("base_tray_weld", False)
        self.set_weld("arm1_grasp_base", False)
        self.set_free_body_pose("base_plate", self._site_pose(self.handles.raw_sites["base_plate"]))
        self.mujoco.mj_forward(self.model, self.data)
        self.set_weld(
            "raw_base_rack_weld",
            True,
            recompute=("raw_material_rack", "base_plate"),
        )
        for index, name in enumerate(FIN_NAMES):
            active = index < self.active_fin_count
            if fins is not None:
                active = bool(fins[index].active)
            self.set_weld(f"arm1_grasp_{name}", False)
            self.set_weld(f"raw_{name}_rack_weld", False)
            self.set_weld(f"{name}_fixture_weld", False)
            self.set_weld(f"{name}_base_weld", False)
            if active:
                self.set_free_body_pose(name, self._site_pose(self.handles.raw_sites[name]))
                self.mujoco.mj_forward(self.model, self.data)
                self.set_weld(
                    f"raw_{name}_rack_weld",
                    True,
                    recompute=("raw_material_rack", name),
                )
        for name in PATH_NAMES:
            self.set_weld(f"{name}_base_weld", False)
            self.set_path_visible(name, False)
        self.mujoco.mj_forward(self.model, self.data)

    def configure_async_raw_kit(self, product: ProductState) -> None:
        """Load one order's base and fin blanks into the shared pickup window.

        The asynchronous line renders finished WIP on the three independent
        batch carriers, but Arm1 still needs real free bodies to pick up.  This
        method configures only that raw-material pool; it deliberately does
        not touch the fixture, dispenser spacing, active batch trays or any
        other order currently running on Arm2/Arm3.
        """

        spec = product.spec
        table_top_geom = self.geom_id("raw_material_rack_top")
        table_top_z = float(self.model.geom_pos[table_top_geom, 2] + self.model.geom_size[table_top_geom, 2])
        self.model.site_pos[self.handles.raw_sites["base_plate"]] = np.asarray(
            [
                *SHALLOW_U_LAYOUT.base_magazine_xy,
                table_top_z + 0.5 * spec.base_thickness,
            ],
            dtype=float,
        )
        base_geom = self.geom_id("heatsink_base_plate_geom")
        self.model.geom_size[base_geom, :3] = np.asarray(spec.base_size, dtype=float) / 2.0
        # These shared blanks are kinematic visual inventory in the async
        # pipeline. Their rigid grasp welds provide transport; disabling raw
        # contact prevents a supported blank from being mistaken for an
        # unexpected collision with its own magazine/table.
        self._set_geom_active("heatsink_base_plate_geom", True, collide=False)

        for index, name in enumerate(FIN_NAMES):
            geom_name = f"{name}_geom"
            geom_id = self.geom_id(geom_name)
            active = index < spec.fin_count
            self.model.geom_matid[geom_id] = -1
            self.model.geom_size[geom_id, :3] = np.asarray(spec.fin_size, dtype=float) / 2.0
            self._set_geom_active(geom_name, active, collide=False)
            position = (
                SHALLOW_U_LAYOUT.raw_fin_position(
                    index,
                    spec.fin_count,
                    table_top_z=table_top_z,
                    fin_height_m=spec.fin_height,
                )
                if active
                else (0.32, 0.30 + 0.03 * (index - spec.fin_count), 0.13)
            )
            self.model.site_pos[self.handles.raw_sites[name]] = np.asarray(position, dtype=float)

        self.mujoco.mj_forward(self.model, self.data)
        self.prepare_raw_materials(product)
        self._set_geom_active("heatsink_base_plate_geom", True, collide=False)
        for index, name in enumerate(FIN_NAMES):
            active = index < spec.fin_count
            self._set_geom_active(f"{name}_geom", active, collide=False)
        self.mujoco.mj_forward(self.model, self.data)

    def set_async_raw_item_visible(self, item_name: str, visible: bool) -> None:
        """Show or retire one real blank without changing another WIP tray."""

        item = str(item_name)
        if item == "base_plate":
            geom_name = "heatsink_base_plate_geom"
            if not visible:
                self.grasp_base(False)
        elif item in FIN_NAMES:
            geom_name = f"{item}_geom"
            if not visible:
                self.grasp_fin(item, False)
        else:
            raise ValueError(f"unknown asynchronous raw item: {item_name}")
        self._set_geom_active(geom_name, bool(visible), collide=False)
        self.mujoco.mj_forward(self.model, self.data)

    def _site_pose(self, site_id: int) -> Pose:
        site = self.data.site(site_id)
        return Pose(
            np.asarray(site.xpos, dtype=float),
            matrix_to_quat(np.asarray(site.xmat, dtype=float).reshape(3, 3)),
        )

    def site_pose(self, site_name: str) -> Pose:
        """Return one named site's current world-frame SE(3) pose."""

        return self._site_pose(self.site_id(site_name))

    def place_base_on_tray(self, *, snap: bool = True) -> None:
        """Move the base to the Table2 product origin and enable its tray weld."""

        self.set_weld("raw_base_rack_weld", False)
        self.set_weld("arm1_grasp_base", False)
        if snap:
            self.set_free_body_pose("base_plate", self.refresh_assembly_target_pose())
        self.mujoco.mj_forward(self.model, self.data)
        self.set_weld(
            "base_tray_weld",
            True,
            recompute=("assembly_tray", "base_plate"),
            forward=True,
        )
        for index, name in enumerate(PATH_NAMES):
            active = index < self.active_path_count
            # Brazing beads are independent free bodies so they can be grown
            # during dispensing.  Re-seat every body in the *current* product
            # frame before enabling its base weld.  Otherwise a pallet moved
            # from S1 to S2A preserves the path's previous world-frame offset
            # and the yellow bead appears at the station the pallet left.
            if active and name in self.path_local_targets:
                self.set_free_body_pose(
                    name,
                    self.product_pose().transformed(self.path_local_targets[name]),
                )
                self.mujoco.mj_forward(self.model, self.data)
            self.set_weld(
                f"{name}_base_weld",
                active,
                recompute=("base_plate", name) if active else None,
            )
        self.mujoco.mj_forward(self.model, self.data)

    def grasp_base(self, active: bool = True) -> None:
        """Attach or release the base at Arm1's permanent gripper."""

        if active:
            self.set_weld("raw_base_rack_weld", False)
            self.set_weld("base_tray_weld", False)
        self.set_weld(
            "arm1_grasp_base",
            active,
            recompute=("arm1_suction_tool", "base_plate") if active else None,
            forward=True,
        )

    def place_fin_in_slot(
        self,
        fin_name: str,
        *,
        temporary_fix: bool = True,
        snap: bool = True,
    ) -> None:
        """Place one raw fin at its product-coordinate slot."""

        if fin_name not in self.fin_local_targets:
            raise ValueError(f"unknown or unconfigured fin {fin_name!r}")
        self.set_weld(f"raw_{fin_name}_rack_weld", False)
        self.set_weld(f"arm1_grasp_{fin_name}", False)
        self.set_weld(f"{fin_name}_fixture_weld", False)
        if snap:
            world_pose = self.product_pose().transformed(self.fin_local_targets[fin_name])
            self.set_free_body_pose(fin_name, world_pose)
        self.mujoco.mj_forward(self.model, self.data)
        if temporary_fix:
            self.temporary_fix_fin(fin_name)

    def grasp_fin(self, fin_name: str, active: bool = True) -> None:
        if fin_name not in FIN_NAMES:
            raise ValueError(f"unknown fin {fin_name!r}")
        if active:
            self.set_weld(f"raw_{fin_name}_rack_weld", False)
            self.set_weld(f"{fin_name}_fixture_weld", False)
        self.set_weld(
            f"arm1_grasp_{fin_name}",
            active,
            recompute=("arm1_parallel_gripper", fin_name) if active else None,
            forward=True,
        )

    def seat_and_grasp_fin(self, fin_name: str) -> None:
        """Transfer an already aligned fin from the rack to the closed jaws.

        The rack weld remains active while the grasp weld is authored from the
        *measured contact pose*.  Only after that rigid relative transform is
        installed is the rack weld released.  This constraint hand-off keeps
        the fin's world pose bit-for-bit continuous; the previous implementation
        rewrote the free-body quaternion to an ideal jaw frame and produced a
        visible clockwise twitch at pickup (followed by an opposite correction
        over the installation slot).
        """

        if fin_name not in FIN_NAMES:
            raise ValueError(f"unknown fin {fin_name!r}")
        self.set_weld(f"{fin_name}_fixture_weld", False)
        self.set_weld(f"arm1_grasp_{fin_name}", False)
        # The progressive close stage has already reached its endpoint.  Stop
        # the two finger slides exactly at the inner faces while the rack still
        # owns the fin, so this final sub-millimetre servo settling cannot move
        # or rotate the workpiece.
        self.snap_arm1_gripper_closed()
        fin_pose = self.free_body_pose(fin_name)
        grasp_tcp = self.site_pose("arm1_grasp_tcp")
        # Remove only the tiny translational servo residual (normally a few
        # tens of micrometres).  The measured fin quaternion is deliberately
        # retained, so seating the centre between the finger faces cannot
        # recreate the old clockwise pickup twitch.
        centred_pose = Pose(grasp_tcp.position, fin_pose.quaternion)
        self.set_free_body_pose(fin_name, centred_pose)
        self.set_weld(
            f"raw_{fin_name}_rack_weld",
            True,
            recompute=("raw_material_rack", fin_name),
            forward=True,
        )
        # Author the new constraint before releasing the old one.  Both welds
        # agree on the current pose, so enabling/disabling them cannot inject a
        # corrective angular impulse.
        self.set_weld(
            f"arm1_grasp_{fin_name}",
            True,
            recompute=("arm1_parallel_gripper", fin_name),
            forward=False,
        )
        self.set_weld(f"raw_{fin_name}_rack_weld", False, forward=True)

    def temporary_fix_fin(self, fin_name: str) -> None:
        self.set_weld(f"arm1_grasp_{fin_name}", False)
        self.set_weld(
            f"{fin_name}_fixture_weld",
            True,
            recompute=("assembly_tray", fin_name),
            forward=True,
        )

    def braze_fin_to_base(self, fin_name: str) -> None:
        """Switch a fin from comb-fixture support to its permanent base weld."""

        if fin_name not in FIN_NAMES:
            raise ValueError(f"unknown fin {fin_name!r}")
        self.set_weld(f"{fin_name}_fixture_weld", False)
        self.set_weld(
            f"{fin_name}_base_weld",
            True,
            recompute=("base_plate", fin_name),
            forward=True,
        )

    def set_fixture_locked(self, locked: bool) -> None:
        """Synchronize the physical fixture constraints and lock indication."""

        self.set_weld(
            "base_tray_weld",
            bool(locked),
            recompute=("assembly_tray", "base_plate") if locked else None,
        )
        for index, fin_name in enumerate(FIN_NAMES):
            active = bool(locked and index < self.active_fin_count)
            self.set_weld(
                f"{fin_name}_fixture_weld",
                active,
                recompute=("assembly_tray", fin_name) if active else None,
            )
        self.set_fixture_lock_visual(locked)
        self.mujoco.mj_forward(self.model, self.data)

    def set_fixture_lock_visual(self, locked: bool) -> None:
        for geom_name in (
            "fixture_front_press_bar",
            "fixture_rear_press_bar",
        ):
            geom_id = self.geom_id(geom_name)
            current_alpha = float(self.model.geom_rgba[geom_id, 3])
            rgba = self._initial_rgba[geom_id].copy()
            # Colour changes must not undo the process-controlled visibility.
            # In particular, reset keeps the press absent during material
            # application until PRESS_FIXTURE explicitly installs it.
            rgba[3] = current_alpha
            if locked:
                rgba[:3] = np.asarray([0.92, 0.52, 0.10])
            self.model.geom_rgba[geom_id] = rgba

    def place_tray_in_furnace(self) -> None:
        self.set_weld(
            "furnace_tray_weld",
            True,
            recompute=("furnace", "assembly_tray"),
            forward=True,
        )

    def set_furnace_door(self, fraction: float, *, teleport: bool = False) -> None:
        amount = float(np.clip(fraction, 0.0, 1.0))
        limits = self._furnace_door_limits
        target = float(limits[0] + (limits[1] - limits[0]) * amount)
        self.data.ctrl[self.handles.furnace_door_actuator] = target
        if teleport:
            current = float(self.data.qpos[self._furnace_door_qpos])
            if abs(current - target) <= 1.0e-12:
                return
            self.data.qpos[self._furnace_door_qpos] = target
            # The simulation loop calls mj_step immediately after this write.
            # Full dynamics/contact recomputation here doubled the per-step
            # MuJoCo cost, including while the door was stationary. A light
            # kinematics refresh is enough for same-frame visual/interlock
            # reads; mj_step performs the authoritative dynamics update.
            self.mujoco.mj_kinematics(self.model, self.data)

    @property
    def furnace_door_fraction(self) -> float:
        """Return the measured opening instead of the actuator command."""

        address = self._furnace_door_qpos
        lower, upper = self._furnace_door_limits
        travel = float(upper - lower)
        if travel <= 0.0:
            return 0.0
        return float(np.clip((self.data.qpos[address] - lower) / travel, 0.0, 1.0))

    @property
    def conveyor_position_m(self) -> float:
        address = int(self.model.jnt_qposadr[self.handles.conveyor_slide_joint])
        return float(self.data.qpos[address])

    @property
    def conveyor_velocity_m_s(self) -> float:
        address = int(self.model.jnt_dofadr[self.handles.conveyor_slide_joint])
        return float(self.data.qvel[address])

    @property
    def conveyor_travel_m(self) -> float:
        limits = np.asarray(self.model.jnt_range[self.handles.conveyor_slide_joint], dtype=float)
        return float(limits[1])

    def set_conveyor_target(self, position_m: float, *, teleport: bool = False) -> float:
        """Command the Table2 conveyor while preserving its straight slide axis."""

        target = float(np.clip(position_m, 0.0, self.conveyor_travel_m))
        self.data.ctrl[self.handles.conveyor_slide_actuator] = target
        if teleport:
            qpos_address = int(self.model.jnt_qposadr[self.handles.conveyor_slide_joint])
            qvel_address = int(self.model.jnt_dofadr[self.handles.conveyor_slide_joint])
            self.data.qpos[qpos_address] = target
            self.data.qvel[qvel_address] = 0.0
            self.mujoco.mj_forward(self.model, self.data)
        return target

    def _named_joint_state(self, joint_name: str) -> tuple[int, int, int]:
        joint_id = self._id(self.mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        return (
            joint_id,
            int(self.model.jnt_qposadr[joint_id]),
            int(self.model.jnt_dofadr[joint_id]),
        )

    def batch_joint_position(self, joint_name: str) -> float:
        _joint_id, qpos_address, _dof_address = self._named_joint_state(joint_name)
        return float(self.data.qpos[qpos_address])

    def batch_joint_velocity(self, joint_name: str) -> float:
        _joint_id, _qpos_address, dof_address = self._named_joint_state(joint_name)
        return float(self.data.qvel[dof_address])

    def set_batch_joint_target(
        self,
        joint_name: str,
        actuator_name: str,
        position_m: float,
        *,
        teleport: bool = False,
    ) -> float:
        joint_id, qpos_address, dof_address = self._named_joint_state(joint_name)
        limits = np.asarray(self.model.jnt_range[joint_id], dtype=float)
        target = float(np.clip(position_m, limits[0], limits[1]))
        actuator_id = self._id(self.mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        self.data.ctrl[actuator_id] = target
        if teleport:
            self.data.qpos[qpos_address] = target
            self.data.qvel[dof_address] = 0.0
            self.mujoco.mj_forward(self.model, self.data)
        return target

    @property
    def finished_output_gate_fraction(self) -> float:
        """Measured lift-gate opening, where 0 is closed and 1 is open."""

        joint_id, qpos_address, _dof_address = self._named_joint_state("finished_output_gate_joint")
        lower, upper = np.asarray(self.model.jnt_range[joint_id], dtype=float)
        travel = float(upper - lower)
        if travel <= 0.0:
            return 0.0
        return float(np.clip((self.data.qpos[qpos_address] - lower) / travel, 0.0, 1.0))

    def retire_batch_tray(self, unit_index: int) -> None:
        """Hide one delivered tray only after it is sealed inside the outlet.

        The free body is parked in the isolation area after visibility and
        collision are disabled.  A full scene reset restores its qpos, while
        ``reset_batch_cell`` restores its exclusive cache ownership.
        """

        unit = int(unit_index) + 1
        if unit not in {1, 2, 3}:
            raise ValueError("batch unit index must be 0, 1 or 2")
        tray = f"batch_tray_{unit:02d}"
        self.set_batch_tray_visible(unit_index, carrier=False, payload=False)
        self.set_batch_weld(f"batch_carrier_tray_{unit:02d}_weld", False)
        self.set_batch_weld(f"batch_output_tray_{unit:02d}_weld", False)
        pose = Pose(
            np.asarray([3.0 + 0.35 * unit_index, 3.0, -1.5]),
            np.asarray([1.0, 0.0, 0.0, 0.0]),
        )
        self.set_free_body_pose(tray, pose, forward=True)

    def handoff_batch_payload(self, unit_index: int) -> None:
        """Visually unload only the product from the complete slotted tray.

        The finished-goods box represents a downstream handoff.  Product,
        fins and brazing geoms disappear only after the complete assembly has
        stopped inside the open box. The suspended comb and both short press
        bars have already been withdrawn at the inspection point; only the
        carrier plate and central template return before the gate closes.
        """

        self.set_batch_tray_visible(unit_index, carrier=True, payload=False)
        # ``carrier=True`` restores reusable fixture defaults.  The comb has
        # its own installed-state guard; preserve the completed press-removal
        # milestone explicitly so the two bars do not reappear inside the
        # outlet when the product payload is handed off.
        self.set_batch_press_visible(unit_index, False)

    @staticmethod
    def _batch_tray_fixture_names(unit: int) -> frozenset[str]:
        """Return the rigid reusable tooling that must travel with a product."""

        prefix = f"batch_tray_{int(unit):02d}"
        return frozenset(
            {
                f"{prefix}_geom",
                f"{prefix}_template_plate",
                f"{prefix}_front_comb_base",
                f"{prefix}_rear_comb_base",
                f"{prefix}_front_comb_post_left",
                f"{prefix}_front_comb_post_right",
                f"{prefix}_rear_comb_post_left",
                f"{prefix}_rear_comb_post_right",
                f"{prefix}_front_press",
                f"{prefix}_rear_press",
                *(
                    f"{prefix}_{end}_comb_guide_{side}{index:02d}"
                    for end in ("front", "rear")
                    for side in ("left", "right")
                    for index in range(12)
                ),
            }
        )

    def set_batch_tray_visible(
        self,
        unit_index: int,
        *,
        carrier: bool,
        payload: bool,
    ) -> None:
        """Control reusable tray tooling and product payload independently.

        ``carrier`` covers the tray plate, central template, low comb bases and
        both press bars. ``payload`` covers only the heat-sink base, fins and
        brazing beads. Reset paths still pass both flags as false, while the
        finished-goods handoff intentionally uses ``True, False`` so an empty
        reusable tray can return from the outlet.
        """

        unit = int(unit_index) + 1
        if unit not in {1, 2, 3}:
            raise ValueError("batch unit index must be 0, 1 or 2")
        prefix = f"batch_tray_{unit:02d}_"
        fixture_names = self._batch_tray_fixture_names(unit)
        for geom_id in range(self.model.ngeom):
            name = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if not name.startswith(prefix):
                continue
            is_fixture = name in fixture_names
            if is_fixture:
                is_comb = "_comb_" in name
                visible = bool(carrier) and (
                    not is_comb
                    or (
                        self._batch_comb_installed.get(int(unit_index), False)
                        and self._batch_payload_active.get(name, True)
                    )
                )
            else:
                visible = bool(payload) and self._batch_payload_active.get(name, True)
            rgba = np.asarray(self.model.geom_rgba[geom_id], dtype=float).copy()
            rgba[3] = 1.0 if visible else 0.0
            self.model.geom_rgba[geom_id] = rgba
            self.model.geom_contype[geom_id] = 0
            self.model.geom_conaffinity[geom_id] = 0
        site_id = self.site_id(f"batch_tray_{unit:02d}_pose")
        self.model.site_rgba[site_id, 3] = 0.65 if carrier else 0.0
        self.mujoco.mj_forward(self.model, self.data)

    def configure_batch_tray(self, unit_index: int, product: ProductState) -> None:
        """Configure one detached batch carrier from its own immutable product."""

        unit = int(unit_index) + 1
        if unit not in {1, 2, 3}:
            raise ValueError("batch unit index must be 0, 1 or 2")
        spec = product.spec
        prefix = f"batch_tray_{unit:02d}"
        template_id = self.geom_id(f"{prefix}_template_plate")
        template_half_size = np.asarray(
            [
                min(0.205, spec.base_length / 2.0 + 0.010),
                min(0.135, spec.base_width / 2.0 + 0.010),
                0.006,
            ],
            dtype=float,
        )
        self.model.geom_size[template_id, :3] = template_half_size
        base_id = self.geom_id(f"{prefix}_base")
        self.model.geom_size[base_id, :3] = np.asarray(spec.base_size, dtype=float) / 2.0
        self.model.geom_pos[base_id] = np.asarray([0.0, 0.0, 0.032])
        self._batch_payload_active[f"{prefix}_base"] = True
        # The load-bearing portal belongs completely outside the fin envelope.
        # Only its cantilever guide fingers reach inward over the base.  Keeping
        # these two coordinates separate prevents the support columns from
        # cutting through long fins after an A/B/C runtime reconfiguration.
        support_x_abs = spec.fin_length / 2.0 + BATCH_COMB_LONGITUDINAL_CLEARANCE + BATCH_COMB_POST_HALF_X
        press_x = spec.fin_length / 2.0 - 0.065
        module_colour = {
            "comb_insert_15mm": np.asarray([0.54, 0.32, 0.78], dtype=float),
            "comb_insert_20mm": np.asarray([0.18, 0.48, 0.78], dtype=float),
            "comb_insert_30mm": np.asarray([0.92, 0.62, 0.12], dtype=float),
            "comb_insert_40mm": np.asarray([0.42, 0.72, 0.58], dtype=float),
        }.get(spec.comb_module_name, np.asarray([0.18, 0.48, 0.62], dtype=float))
        fin_centres = tuple(float(fin.target_position[1]) for fin in product.active_fins)
        slot_offset = max(0.003, spec.fin_thickness / 2.0 + 0.002)
        support_y = spec.base_width / 2.0 + BATCH_COMB_LATERAL_CLEARANCE + BATCH_COMB_POST_HALF_Y
        for label, x in (("front_comb", -support_x_abs), ("rear_comb", support_x_abs)):
            base_geom_id = self.geom_id(f"{prefix}_{label}_base")
            self.model.geom_pos[base_geom_id, 0] = x
            self.model.geom_size[base_geom_id, 1] = support_y + BATCH_COMB_POST_HALF_Y
            self.model.geom_size[base_geom_id, 0] = BATCH_COMB_POST_HALF_X
            self.model.geom_size[base_geom_id, 2] = 0.004
            self.model.geom_pos[base_geom_id, 2] = 0.055
            self.model.geom_rgba[base_geom_id, :3] = module_colour
        self._batch_comb_geometry[int(unit_index)] = (
            float(support_x_abs),
            fin_centres,
            float(slot_offset),
            float(support_y),
        )
        self._batch_comb_installed[int(unit_index)] = False
        for end, end_sign in (
            ("front", -1.0),
            ("rear", 1.0),
        ):
            support_x = end_sign * support_x_abs
            for side, y_sign in (("left", -1.0), ("right", 1.0)):
                post_name = f"{prefix}_{end}_comb_post_{side}"
                post_id = self.geom_id(post_name)
                self._batch_payload_active[post_name] = True
                self.model.geom_pos[post_id] = np.asarray(
                    [support_x, y_sign * support_y, 0.034],
                    dtype=float,
                )
                self.model.geom_rgba[post_id, :3] = module_colour
            guide_x = support_x - end_sign * (BATCH_COMB_POST_HALF_X + BATCH_COMB_GUIDE_HALF_X)
            for side, y_sign in (("left", -1.0), ("right", 1.0)):
                for index in range(12):
                    name = f"{prefix}_{end}_comb_guide_{side}{index:02d}"
                    geom_id = self.geom_id(name)
                    active = index < len(fin_centres)
                    self._batch_payload_active[name] = active
                    self.model.geom_pos[geom_id, 0] = guide_x
                    self.model.geom_rgba[geom_id, :3] = module_colour
                    if active:
                        self.model.geom_pos[geom_id, 1] = fin_centres[index] + y_sign * slot_offset
        base_top = 0.032 + spec.base_thickness / 2.0
        press_target_z = base_top + spec.fin_height + 0.003
        self._batch_press_target_z[int(unit_index)] = float(press_target_z)
        self._batch_press_target_x[int(unit_index)] = (-float(press_x), float(press_x))
        for label, x in (("front_press", -press_x), ("rear_press", press_x)):
            geom_id = self.geom_id(f"{prefix}_{label}")
            self.model.geom_pos[geom_id, 0] = x
            self.model.geom_pos[geom_id, 2] = press_target_z + 0.060
            self._batch_payload_active[f"{prefix}_{label}"] = True

        for index in range(12):
            name = f"{prefix}_fin_{index + 1:02d}"
            geom_id = self.geom_id(name)
            active = index < spec.fin_count
            self._batch_payload_active[name] = active
            if active:
                fin = product.active_fins[index]
                self.model.geom_size[geom_id, :3] = np.asarray(spec.fin_size, dtype=float) / 2.0
                self.model.geom_pos[geom_id] = np.asarray(
                    [0.0, fin.target_position[1], base_top + spec.fin_height / 2.0]
                )

        canonical_x_start = -spec.base_length / 2.0 + spec.path_margin
        canonical_x_end = spec.base_length / 2.0 - spec.path_margin
        for index in range(24):
            name = f"{prefix}_braze_{index + 1:02d}"
            geom_id = self.geom_id(name)
            active = index < spec.path_count
            self._batch_payload_active[name] = active
            if active:
                path = product.active_paths[index]
                local_start = np.asarray(path.local_start, dtype=float).copy()
                local_end = np.asarray(path.local_end, dtype=float).copy()
                # Every bead shares the same product-coordinate longitudinal
                # limits.  This deliberately removes any accumulated endpoint
                # residue from alternating left-to-right/right-to-left passes.
                local_start[0] = canonical_x_start
                local_end[0] = canonical_x_end
                midpoint = 0.5 * (local_start + local_end)
                self.model.geom_pos[geom_id] = np.asarray([midpoint[0], midpoint[1], base_top + 0.001])
                self.model.geom_quat[geom_id] = np.asarray([0.7071067812, 0.0, 0.7071067812, 0.0])
                self.model.geom_size[geom_id, 0] = path.target_width_m / 2.0
                self.model.geom_size[geom_id, 1] = abs(path.local_end[0] - path.local_start[0]) / 2.0
                self._batch_path_geometry[(unit_index, index)] = (
                    local_start,
                    local_end,
                    float(path.target_width_m),
                    float(base_top + 0.001),
                )
                self._batch_path_progress[(unit_index, index)] = 0.0
            else:
                self._batch_path_geometry.pop((unit_index, index), None)
                self._batch_path_progress.pop((unit_index, index), None)
        self.mujoco.mj_forward(self.model, self.data)

    def set_batch_comb_install_progress(
        self,
        unit_index: int,
        fraction: float,
    ) -> None:
        """Slide both slotted comb modules into the tray from its X sides.

        Each slot is formed by the same left/right guide-finger pair used by
        the original fixture. Its centre is derived from the order's fin
        target. The modules become visible before motion starts, move
        continuously, and stop at the frame used later by Arm1 fin placement.
        """

        unit = int(unit_index) + 1
        if unit not in {1, 2, 3} or int(unit_index) not in self._batch_comb_geometry:
            raise ValueError(f"batch comb is not configured for unit {unit_index}")
        prefix = f"batch_tray_{unit:02d}"
        support_x_abs, fin_centres, slot_offset, support_y = self._batch_comb_geometry[int(unit_index)]
        amount = float(np.clip(fraction, 0.0, 1.0))
        eased = amount * amount * (3.0 - 2.0 * amount)
        outside_offset = 0.085 * (1.0 - eased)
        self._batch_comb_installed[int(unit_index)] = amount > 1.0e-6
        for end, end_sign in (
            ("front", -1.0),
            ("rear", 1.0),
        ):
            support_x = end_sign * (support_x_abs + outside_offset)
            base_id = self.geom_id(f"{prefix}_{end}_comb_base")
            self.model.geom_pos[base_id, 0] = support_x
            base_rgba = np.asarray(self.model.geom_rgba[base_id], dtype=float).copy()
            base_rgba[3] = 1.0 if amount > 1.0e-6 else 0.0
            self.model.geom_rgba[base_id] = base_rgba
            for side, y_sign in (("left", -1.0), ("right", 1.0)):
                post_name = f"{prefix}_{end}_comb_post_{side}"
                post_id = self.geom_id(post_name)
                self.model.geom_pos[post_id] = np.asarray(
                    [support_x, y_sign * support_y, 0.034],
                    dtype=float,
                )
                post_rgba = np.asarray(self.model.geom_rgba[post_id], dtype=float).copy()
                post_rgba[3] = 1.0 if amount > 1.0e-6 else 0.0
                self.model.geom_rgba[post_id] = post_rgba
            guide_x = support_x - end_sign * (BATCH_COMB_POST_HALF_X + BATCH_COMB_GUIDE_HALF_X)
            for side, y_sign in (("left", -1.0), ("right", 1.0)):
                for index in range(12):
                    name = f"{prefix}_{end}_comb_guide_{side}{index:02d}"
                    geom_id = self.geom_id(name)
                    active = index < len(fin_centres)
                    self.model.geom_pos[geom_id, 0] = guide_x
                    if active:
                        self.model.geom_pos[geom_id, 1] = fin_centres[index] + y_sign * slot_offset
                    rgba = np.asarray(self.model.geom_rgba[geom_id], dtype=float).copy()
                    rgba[3] = 1.0 if active and amount > 1.0e-6 else 0.0
                    self.model.geom_rgba[geom_id] = rgba

    def batch_tray_pose(self, unit_index: int) -> Pose:
        """Return the live carrier frame used by order-local motion targets."""

        unit = int(unit_index) + 1
        if unit not in {1, 2, 3}:
            raise ValueError("batch unit index must be 0, 1 or 2")
        return self.free_body_pose(f"batch_tray_{unit:02d}")

    def batch_to_world(self, unit_index: int, local: Sequence[float] | Pose) -> np.ndarray | Pose:
        """Transform a point or pose from one pallet frame into world space."""

        tray = self.batch_tray_pose(unit_index)
        if isinstance(local, Pose):
            return tray.transformed(local)
        vector = np.asarray(local, dtype=float)
        if vector.shape != (3,):
            raise ValueError("batch coordinate must be xyz")
        return tray.position + tray.rotation @ vector

    def set_batch_brazing_path_progress(
        self,
        unit_index: int,
        path_index: int,
        fraction: float,
        *,
        reverse: bool = False,
        allow_decrease: bool = False,
    ) -> None:
        """Grow one bead from the same end currently followed by Arm2.

        The batch carriers use one capsule per path for rendering efficiency.
        Updating its local midpoint and half-length makes the deposited segment
        grow continuously without adding hundreds of collision-free geoms.
        """

        unit = int(unit_index) + 1
        key = (int(unit_index), int(path_index))
        if unit not in {1, 2, 3} or key not in self._batch_path_geometry:
            raise ValueError(f"unknown batch brazing path {key}")
        local_start, local_end, width_m, height_m = self._batch_path_geometry[key]
        requested = float(np.clip(fraction, 0.0, 1.0))
        previous = self._batch_path_progress.get(key, 0.0)
        amount = requested if allow_decrease else max(previous, requested)
        self._batch_path_progress[key] = amount
        travel_start = local_end if reverse else local_start
        travel_end = local_start if reverse else local_end
        current = travel_start + amount * (travel_end - travel_start)
        midpoint = 0.5 * (travel_start + current)
        length = float(np.linalg.norm(current - travel_start))
        geom_name = f"batch_tray_{unit:02d}_braze_{int(path_index) + 1:02d}"
        geom_id = self.geom_id(geom_name)
        self.model.geom_pos[geom_id] = np.asarray([midpoint[0], midpoint[1], height_m])
        self.model.geom_quat[geom_id] = np.asarray([0.7071067812, 0.0, 0.7071067812, 0.0])
        self.model.geom_size[geom_id, 0] = width_m / 2.0
        self.model.geom_size[geom_id, 1] = max(1.0e-5, 0.5 * length)
        rgba = np.asarray(self.model.geom_rgba[geom_id], dtype=float).copy()
        rgba[3] = 1.0 if amount > 1.0e-6 else 0.0
        self.model.geom_rgba[geom_id] = rgba

    def set_batch_base_visible(self, unit_index: int, visible: bool = True) -> None:
        """Commit only the base visual for one WIP tray."""

        unit = int(unit_index) + 1
        if unit not in {1, 2, 3}:
            raise ValueError("batch unit index must be 0, 1 or 2")
        geom_id = self.geom_id(f"batch_tray_{unit:02d}_base")
        rgba = np.asarray(self.model.geom_rgba[geom_id], dtype=float).copy()
        rgba[3] = 1.0 if visible else 0.0
        self.model.geom_rgba[geom_id] = rgba
        self.mujoco.mj_forward(self.model, self.data)

    def set_batch_fin_visible(
        self,
        unit_index: int,
        fin_index: int,
        visible: bool = True,
    ) -> None:
        """Commit one fin without rewriting any previously installed fin."""

        unit = int(unit_index) + 1
        index = int(fin_index)
        if unit not in {1, 2, 3} or not 0 <= index < 12:
            raise ValueError("batch fin index is outside the 3x12 object pool")
        name = f"batch_tray_{unit:02d}_fin_{index + 1:02d}"
        active = self._batch_payload_active.get(name, False)
        geom_id = self.geom_id(name)
        rgba = np.asarray(self.model.geom_rgba[geom_id], dtype=float).copy()
        rgba[3] = 1.0 if visible and active else 0.0
        self.model.geom_rgba[geom_id] = rgba
        self.mujoco.mj_forward(self.model, self.data)

    def set_batch_press_visible(self, unit_index: int, visible: bool = True) -> None:
        """Commit the two short press beams without touching product visuals."""

        unit = int(unit_index) + 1
        if unit not in {1, 2, 3}:
            raise ValueError("batch unit index must be 0, 1 or 2")
        for suffix in ("front_press", "rear_press"):
            geom_id = self.geom_id(f"batch_tray_{unit:02d}_{suffix}")
            rgba = np.asarray(self.model.geom_rgba[geom_id], dtype=float).copy()
            rgba[3] = 1.0 if visible else 0.0
            self.model.geom_rgba[geom_id] = rgba
        self.mujoco.mj_forward(self.model, self.data)

    def set_batch_press_progress(self, unit_index: int, fraction: float) -> None:
        """Lower both short press beams with a smooth, visible stroke."""

        unit = int(unit_index) + 1
        index = int(unit_index)
        if unit not in {1, 2, 3} or index not in self._batch_press_target_z:
            raise ValueError(f"batch press is not configured for unit {unit_index}")
        amount = float(np.clip(fraction, 0.0, 1.0))
        eased = amount * amount * (3.0 - 2.0 * amount)
        target_z = self._batch_press_target_z[index]
        target_x = self._batch_press_target_x[index]
        current_z = target_z + 0.060 * (1.0 - eased)
        for suffix, x in zip(("front_press", "rear_press"), target_x):
            geom_id = self.geom_id(f"batch_tray_{unit:02d}_{suffix}")
            self.model.geom_pos[geom_id, 0] = x
            self.model.geom_pos[geom_id, 2] = current_z
            rgba = np.asarray(self.model.geom_rgba[geom_id], dtype=float).copy()
            rgba[3] = 1.0
            self.model.geom_rgba[geom_id] = rgba

    def set_batch_press_removal_progress(self, unit_index: int, fraction: float) -> None:
        """Lift and withdraw both short press bars before finished delivery.

        Both bars remain visible throughout the quintic motion and disappear
        only after reaching their safe removal poses.  This makes the tooling
        handoff explicit instead of replacing it with an alpha jump.
        """

        unit = int(unit_index) + 1
        index = int(unit_index)
        if (
            unit not in {1, 2, 3}
            or index not in self._batch_press_target_z
            or index not in self._batch_press_target_x
        ):
            raise ValueError(f"batch press is not configured for unit {unit_index}")
        amount = float(np.clip(fraction, 0.0, 1.0))
        eased = amount**3 * (10.0 - 15.0 * amount + 6.0 * amount**2)
        target_z = self._batch_press_target_z[index]
        front_x, rear_x = self._batch_press_target_x[index]
        for suffix, start_x, direction in (
            ("front_press", front_x, -1.0),
            ("rear_press", rear_x, 1.0),
        ):
            geom_id = self.geom_id(f"batch_tray_{unit:02d}_{suffix}")
            self.model.geom_pos[geom_id, 0] = start_x + direction * 0.085 * eased
            self.model.geom_pos[geom_id, 2] = target_z + 0.075 * eased
            rgba = np.asarray(self.model.geom_rgba[geom_id], dtype=float).copy()
            rgba[3] = 1.0 if amount < 1.0 else 0.0
            self.model.geom_rgba[geom_id] = rgba

    def set_batch_press_locked(self, unit_index: int, locked: bool = True) -> None:
        """Expose the fixture-lock result without redrawing the product."""

        unit = int(unit_index) + 1
        if unit not in {1, 2, 3}:
            raise ValueError("batch unit index must be 0, 1 or 2")
        colour = np.asarray([0.12, 0.72, 0.28] if locked else [0.92, 0.52, 0.10])
        for suffix in ("front_press", "rear_press"):
            geom_id = self.geom_id(f"batch_tray_{unit:02d}_{suffix}")
            self.model.geom_rgba[geom_id, :3] = colour

    def set_batch_tray_stage(
        self,
        unit_index: int,
        *,
        base_visible: bool = False,
        material_count: int = 0,
        fin_count: int = 0,
        comb_visible: bool = False,
        press_visible: bool = False,
    ) -> None:
        """Render one WIP carrier at its real current manufacturing stage."""

        unit = int(unit_index) + 1
        if unit not in {1, 2, 3}:
            raise ValueError("batch unit index must be 0, 1 or 2")
        prefix = f"batch_tray_{unit:02d}"
        self.set_batch_tray_visible(unit_index, carrier=True, payload=True)

        def show(name: str, visible: bool) -> None:
            geom_id = self.geom_id(name)
            rgba = np.asarray(self.model.geom_rgba[geom_id], dtype=float).copy()
            rgba[3] = 1.0 if visible else 0.0
            self.model.geom_rgba[geom_id] = rgba

        show(f"{prefix}_base", bool(base_visible))
        for index in range(12):
            name = f"{prefix}_fin_{index + 1:02d}"
            show(name, index < int(fin_count) and self._batch_payload_active.get(name, False))
        for index in range(24):
            name = f"{prefix}_braze_{index + 1:02d}"
            show(name, index < int(material_count) and self._batch_payload_active.get(name, False))
        for suffix in ("front_comb_base", "rear_comb_base"):
            show(f"{prefix}_{suffix}", bool(comb_visible))
        for end in ("front", "rear"):
            for side in ("left", "right"):
                show(f"{prefix}_{end}_comb_post_{side}", bool(comb_visible))
            for side in ("left", "right"):
                for index in range(12):
                    name = f"{prefix}_{end}_comb_guide_{side}{index:02d}"
                    show(
                        name,
                        bool(comb_visible) and self._batch_payload_active.get(name, False),
                    )
        for suffix in ("front_press", "rear_press"):
            show(f"{prefix}_{suffix}", bool(press_visible))
        self.mujoco.mj_forward(self.model, self.data)

    @staticmethod
    def _async_wip_unit(tray_id: str) -> int:
        value = str(tray_id).strip().lower()
        if value.startswith("batch_tray_"):
            value = f"tray_{value[-2:]}"
        if value not in {"tray_01", "tray_02", "tray_03"}:
            raise ValueError(f"unknown async WIP tray: {tray_id}")
        return int(value[-2:])

    @staticmethod
    def _async_wip_ownership_welds(unit: int) -> tuple[str, ...]:
        return (
            *(f"station_{station}_tray_{unit:02d}_weld" for station in ASYNC_STATION_ANCHORS),
            *(f"transfer_{transfer}_tray_{unit:02d}_weld" for transfer in ASYNC_TRANSFER_SPECS),
            f"batch_station_tray_{unit:02d}_weld" if unit == 1 else f"batch_indexer_tray_{unit:02d}_weld",
            f"batch_carrier_tray_{unit:02d}_weld",
            f"batch_output_tray_{unit:02d}_weld",
            *(f"batch_rack_tray_{unit:02d}_shelf_{shelf}_weld" for shelf in range(3)),
        )

    def dock_batch_tray_to_async_station(
        self,
        tray_id: str,
        station: str,
        *,
        snap: bool = False,
    ) -> str:
        """Atomically assign one physical WIP tray to one shallow-U station."""

        unit = self._async_wip_unit(tray_id)
        key = str(station).strip().lower()
        if key not in ASYNC_STATION_ANCHORS:
            raise ValueError(f"unknown asynchronous station: {station}")
        tray = f"batch_tray_{unit:02d}"
        anchor = ASYNC_STATION_ANCHORS[key]
        for weld in self._async_wip_ownership_welds(unit):
            self.set_batch_weld(weld, False)
        if snap:
            self.set_free_body_pose(
                tray,
                _pose_from_body(self.data, self.body_id(anchor)),
                forward=True,
            )
        self.set_batch_weld(
            f"station_{key}_tray_{unit:02d}_weld",
            True,
            recompute=(anchor, tray),
        )
        return key

    def begin_batch_tray_async_transfer(self, tray_id: str, transfer_id: str) -> str:
        """Move station ownership onto an empty, homed transfer carriage."""

        unit = self._async_wip_unit(tray_id)
        key = str(transfer_id).strip().lower().removeprefix("transfer_")
        if key not in ASYNC_TRANSFER_SPECS:
            raise ValueError(f"unknown asynchronous transfer: {transfer_id}")
        if abs(self.async_transfer_position(key)) > 0.002:
            raise RuntimeError(f"{key}输送滑台尚未回到源端")
        tray = f"batch_tray_{unit:02d}"
        carriage = f"transfer_{key}_carriage"
        for weld in self._async_wip_ownership_welds(unit):
            self.set_batch_weld(weld, False)
        self.set_batch_weld(
            f"transfer_{key}_tray_{unit:02d}_weld",
            True,
            recompute=(carriage, tray),
        )
        return key

    def finish_batch_tray_async_transfer(
        self,
        tray_id: str,
        transfer_id: str,
        destination: str,
    ) -> None:
        """Hand a stopped tray to its destination before the empty slide returns."""

        unit = self._async_wip_unit(tray_id)
        key = str(transfer_id).strip().lower().removeprefix("transfer_")
        target = str(destination).strip().lower()
        tray = f"batch_tray_{unit:02d}"
        anchor = ASYNC_STATION_ANCHORS[target]
        self.set_batch_weld(f"transfer_{key}_tray_{unit:02d}_weld", False)
        self.set_batch_weld(
            f"station_{target}_tray_{unit:02d}_weld",
            True,
            recompute=(anchor, tray),
        )

    def set_workcell_visible(self, visible: bool) -> None:
        """Toggle the reusable Table2 product/fixture representation.

        Batch mode hands the completed visible assembly to one of the three
        independent carrier bodies at the exact station pose. The reusable
        workcell can then be prepared for the next layer without disturbing
        any carrier already locked in the furnace rack.
        """

        roots = {
            self.body_id("assembly_tray"),
            self.body_id("base_plate"),
            *(self.body_id(name) for name in FIN_NAMES),
            *(self.body_id(name) for name in PATH_NAMES),
        }

        def belongs_to_workcell(body_id: int) -> bool:
            current = int(body_id)
            while current > 0:
                if current in roots:
                    return True
                current = int(self.model.body_parentid[current])
            return False

        for geom_id in range(self.model.ngeom):
            if not belongs_to_workcell(int(self.model.geom_bodyid[geom_id])):
                continue
            if visible:
                self.model.geom_rgba[geom_id] = self._model_initial_rgba[geom_id]
                self.model.geom_contype[geom_id] = self._model_initial_contype[geom_id]
                self.model.geom_conaffinity[geom_id] = self._model_initial_conaffinity[geom_id]
            else:
                self.model.geom_rgba[geom_id, 3] = 0.0
                self.model.geom_contype[geom_id] = 0
                self.model.geom_conaffinity[geom_id] = 0
        self.mujoco.mj_forward(self.model, self.data)

    def set_batch_weld(self, name: str, active: bool, *, recompute: tuple[str, str] | None = None) -> None:
        self.set_weld(name, active, recompute=recompute, forward=True)

    def async_transfer_position(self, transfer_id: str) -> float:
        key = str(transfer_id).strip().lower().removeprefix("transfer_")
        joint, _, _ = ASYNC_TRANSFER_SPECS[key]
        return self.batch_joint_position(joint)

    def async_transfer_velocity(self, transfer_id: str) -> float:
        key = str(transfer_id).strip().lower().removeprefix("transfer_")
        joint, _, _ = ASYNC_TRANSFER_SPECS[key]
        return self.batch_joint_velocity(joint)

    def async_transfer_limit(self, transfer_id: str) -> float:
        key = str(transfer_id).strip().lower().removeprefix("transfer_")
        return float(ASYNC_TRANSFER_SPECS[key][2])

    def set_async_transfer_target(
        self,
        transfer_id: str,
        position_m: float,
        *,
        teleport: bool = False,
    ) -> float:
        key = str(transfer_id).strip().lower().removeprefix("transfer_")
        joint, actuator, _ = ASYNC_TRANSFER_SPECS[key]
        return self.set_batch_joint_target(joint, actuator, position_m, teleport=teleport)

    def reset_async_transfers(self, *, teleport: bool = True) -> None:
        for transfer_id in ASYNC_TRANSFER_SPECS:
            self.set_async_transfer_target(transfer_id, 0.0, teleport=teleport)

    def dock_assembly_tray_to_station(self, station: str, *, snap: bool = True) -> str:
        """Give exactly one asynchronous station ownership of the live tray."""

        key = str(station).strip().lower()
        if key not in ASYNC_STATION_ANCHORS:
            raise ValueError(f"unknown asynchronous station: {station}")
        anchor = ASYNC_STATION_ANCHORS[key]
        ownership_welds = [
            "tray_fixture_weld",
            "furnace_tray_weld",
            *[f"station_{name}_assembly_tray_weld" for name in ASYNC_STATION_ANCHORS],
            *[f"transfer_{name}_assembly_tray_weld" for name in ASYNC_TRANSFER_SPECS],
            "turntable_nest_a_assembly_tray_weld",
            "turntable_nest_b_assembly_tray_weld",
            "table2_handoff_assembly_tray_weld",
        ]
        for weld in ownership_welds:
            self.set_weld(weld, False)
        if snap:
            self.set_free_body_pose(
                "assembly_tray",
                _pose_from_body(self.data, self.body_id(anchor)),
                forward=True,
            )
        self.set_weld(
            f"station_{key}_assembly_tray_weld",
            True,
            recompute=(anchor, "assembly_tray"),
            forward=True,
        )
        self.refresh_assembly_target_pose()
        return key

    def handoff_rack_infeed_to_conveyor(self) -> None:
        """Atomically transfer the live tray from rack dock to furnace slide."""

        rack_weld = "station_rack_infeed_assembly_tray_weld"
        conveyor_weld = "tray_fixture_weld"
        if bool(self.data.eq_active[self.equality_id(conveyor_weld)]):
            return
        if not bool(self.data.eq_active[self.equality_id(rack_weld)]):
            raise RuntimeError("托盘尚未到达炉前料架入口，不能交接给炉体输送滑台")
        self.set_weld(rack_weld, False)
        self.set_weld(
            conveyor_weld,
            True,
            recompute=("assembly_fixture", "assembly_tray"),
            forward=True,
        )
        self.refresh_assembly_target_pose()

    def async_line_snapshot(self) -> dict[str, Any]:
        station_owner = None
        for station in ASYNC_STATION_ANCHORS:
            if bool(self.data.eq_active[self.equality_id(f"station_{station}_assembly_tray_weld")]):
                station_owner = station.upper()
                break
        tray_owners: dict[str, str] = {}
        for unit in range(1, 4):
            tray_id = f"tray_{unit:02d}"
            owner = "BUFFER"
            for station in ASYNC_STATION_ANCHORS:
                weld = self.equality_id(f"station_{station}_tray_{unit:02d}_weld")
                if bool(self.data.eq_active[weld]):
                    owner = station.upper()
                    break
            if owner == "BUFFER":
                for transfer in ASYNC_TRANSFER_SPECS:
                    weld = self.equality_id(f"transfer_{transfer}_tray_{unit:02d}_weld")
                    if bool(self.data.eq_active[weld]):
                        owner = f"TRANSFER_{transfer.upper()}"
                        break
            tray_owners[tray_id] = owner
        return {
            "layout": "SHALLOW_U",
            "station_owner": station_owner,
            "station_positions_m": {
                key.upper(): np.asarray(
                    self.data.body(anchor).xpos,
                    dtype=float,
                ).tolist()
                for key, anchor in ASYNC_STATION_ANCHORS.items()
            },
            "material_magazines_m": {
                "BASE": [*SHALLOW_U_LAYOUT.base_magazine_xy, 0.10],
                "FIN": [*SHALLOW_U_LAYOUT.fin_magazine_xy, 0.10],
            },
            "output_lane": {
                "center_x_m": SHALLOW_U_LAYOUT.output_lane_x,
                "pallet_half_width_m": SHALLOW_U_LAYOUT.output_pallet_half_width_m,
            },
            "physical_tray_owners": tray_owners,
            "transfer_positions_m": {
                key.upper(): self.async_transfer_position(key) for key in ASYNC_TRANSFER_SPECS
            },
            "transfer_velocities_m_s": {
                key.upper(): self.async_transfer_velocity(key) for key in ASYNC_TRANSFER_SPECS
            },
        }

    @property
    def turntable_angle_rad(self) -> float:
        return self.batch_joint_position("table2_turntable_joint")

    @property
    def turntable_velocity_rad_s(self) -> float:
        return self.batch_joint_velocity("table2_turntable_joint")

    def set_turntable_target(self, angle_rad: float, *, teleport: bool = False) -> float:
        return self.set_batch_joint_target(
            "table2_turntable_joint",
            "table2_turntable_actuator",
            angle_rad,
            teleport=teleport,
        )

    def set_turntable_nest_lock(
        self,
        nest_id: str,
        engaged: bool,
        *,
        teleport: bool = False,
    ) -> None:
        nest = str(nest_id).strip().lower().removeprefix("nest_")
        if nest not in {"a", "b"}:
            raise ValueError("turntable nest must be A or B")
        self.set_batch_joint_target(
            f"table2_nest_{nest}_lock_joint",
            f"table2_nest_{nest}_lock_actuator",
            0.025 if engaged else 0.0,
            teleport=teleport,
        )

    @property
    def handoff_position_m(self) -> float:
        return self.batch_joint_position("table2_handoff_joint")

    @property
    def handoff_velocity_m_s(self) -> float:
        return self.batch_joint_velocity("table2_handoff_joint")

    def set_handoff_target(self, position_m: float, *, teleport: bool = False) -> float:
        return self.set_batch_joint_target(
            "table2_handoff_joint",
            "table2_handoff_actuator",
            position_m,
            teleport=teleport,
        )

    def refresh_assembly_target_pose(self) -> Pose:
        """Refresh the cached product frame after any physical pallet move."""

        self.mujoco.mj_kinematics(self.model, self.data)
        self.assembly_base_pose = self.site_pose("base_plate_target_site")
        return self.assembly_base_pose

    def dock_assembly_tray_to_turntable(self, station: str = "assembly") -> str:
        """Bind the reusable physical workpiece tray to the live turntable.

        ``station`` names a world-side workstation, not a permanent nest.  A
        180-degree index swaps which physical nest is on the left, so the
        owner is selected from the measured anchor X coordinates every time a
        fresh product unit enters the workcell.
        """

        normalized = str(station).strip().lower()
        if normalized not in {"assembly", "process"}:
            raise ValueError("turntable station must be assembly or process")
        anchors = {
            "a": self.site_pose("table2_nest_a_site"),
            "b": self.site_pose("table2_nest_b_site"),
        }
        ordered = sorted(anchors, key=lambda key: float(anchors[key].position[0]))
        nest = ordered[0] if normalized == "assembly" else ordered[-1]
        anchor = f"table2_nest_{nest}_anchor"
        for weld in (
            "tray_fixture_weld",
            "turntable_nest_a_assembly_tray_weld",
            "turntable_nest_b_assembly_tray_weld",
            "table2_handoff_assembly_tray_weld",
            "furnace_tray_weld",
        ):
            self.set_weld(weld, False)
        anchor_pose = _pose_from_body(self.data, self.body_id(anchor))
        self.set_free_body_pose("assembly_tray", anchor_pose, forward=True)
        self.set_weld(
            f"turntable_nest_{nest}_assembly_tray_weld",
            True,
            recompute=(anchor, "assembly_tray"),
            forward=True,
        )
        self.set_turntable_nest_lock("a", True)
        self.set_turntable_nest_lock("b", True)
        self.refresh_assembly_target_pose()
        return nest

    def assembly_tray_turntable_nest(self) -> str | None:
        for nest in ("a", "b"):
            weld = self.equality_id(f"turntable_nest_{nest}_assembly_tray_weld")
            if bool(self.data.eq_active[weld]):
                return nest
        return None

    def set_changeover_gantry_target(
        self,
        x: float,
        y: float,
        z: float,
        *,
        teleport: bool = False,
    ) -> tuple[float, float, float]:
        return (
            self.set_batch_joint_target(
                "changeover_gantry_x_joint",
                "changeover_gantry_x_actuator",
                x,
                teleport=teleport,
            ),
            self.set_batch_joint_target(
                "changeover_gantry_y_joint",
                "changeover_gantry_y_actuator",
                y,
                teleport=teleport,
            ),
            self.set_batch_joint_target(
                "changeover_gantry_z_joint",
                "changeover_gantry_z_actuator",
                z,
                teleport=teleport,
            ),
        )

    def changeover_gantry_position(self) -> tuple[float, float, float]:
        return tuple(
            self.batch_joint_position(name)
            for name in (
                "changeover_gantry_x_joint",
                "changeover_gantry_y_joint",
                "changeover_gantry_z_joint",
            )
        )

    def assign_tray_to_turntable(self, tray_id: str, nest_id: str | None) -> None:
        tray = str(tray_id).lower()
        if tray not in BATCH_TRAY_NAMES:
            raise ValueError(f"unknown flexible tray: {tray_id}")
        unit = int(tray[-2:])
        ownership_welds = [
            f"turntable_nest_a_tray_{unit:02d}_weld",
            f"turntable_nest_b_tray_{unit:02d}_weld",
            f"batch_station_tray_{unit:02d}_weld" if unit == 1 else f"batch_indexer_tray_{unit:02d}_weld",
            f"batch_carrier_tray_{unit:02d}_weld",
            f"batch_output_tray_{unit:02d}_weld",
        ]
        ownership_welds.extend(f"batch_rack_tray_{unit:02d}_shelf_{shelf}_weld" for shelf in range(3))
        for weld in ownership_welds:
            self.set_batch_weld(weld, False)
        if nest_id is None:
            return
        nest = str(nest_id).strip().lower().removeprefix("nest_")
        if nest not in {"a", "b"}:
            raise ValueError("turntable nest must be A or B")
        anchor = f"table2_nest_{nest}_anchor"
        pose = _pose_from_body(self.data, self.body_id(anchor))
        self.set_free_body_pose(tray, pose, forward=True)
        self.set_batch_weld(
            f"turntable_nest_{nest}_tray_{unit:02d}_weld",
            True,
            recompute=(anchor, tray),
        )
        self.set_turntable_nest_lock(nest, True)

    def assign_changeover_module(self, module_id: str, owner: str) -> None:
        module = str(module_id).lower()
        if module not in CHANGEOVER_COMPONENT_NAMES:
            raise ValueError(f"unknown changeover module: {module_id}")
        module_body = self.body_id(module)
        for geom_id in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[geom_id])
            while body_id > 0 and body_id != module_body:
                body_id = int(self.model.body_parentid[body_id])
            if body_id == module_body:
                # Ownership welds and pose/velocity interlocks provide the
                # module safety contract. Disabling contacts prevents a rack
                # part from solver-jamming during the deliberate extraction.
                self.model.geom_contype[geom_id] = 0
                self.model.geom_conaffinity[geom_id] = 0
        copy = module.rsplit("_", 1)[-1]
        rack_weld = f"{module}_rack_weld"
        gantry_weld = f"{module}_gantry_weld"
        nest_weld = f"{module}_nest_{copy}_weld"
        for weld in (rack_weld, gantry_weld, nest_weld):
            self.set_batch_weld(weld, False)
        normalized = str(owner).strip().lower()
        if normalized == "rack":
            anchor = "changeover_module_library"
            weld = rack_weld
        elif normalized == "gantry":
            anchor = "changeover_gantry_dual_hook"
            weld = gantry_weld
        elif normalized in {"nest_a", "nest_b"}:
            expected = f"nest_{copy}"
            if normalized != expected:
                raise ValueError(f"{module} is allocated to {expected}")
            anchor = f"table2_{normalized}_anchor"
            weld = nest_weld
        else:
            raise ValueError("module owner must be rack, gantry, nest_a or nest_b")
        self.set_batch_weld(weld, True, recompute=(anchor, module))

    def reset_async_line(self) -> None:
        """Reset only the active shallow-U carriers and station ownership."""

        self.reset_async_transfers(teleport=True)
        self.dock_assembly_tray_to_station("s1", snap=True)
        self.mujoco.mj_forward(self.model, self.data)

    def reset_flexible_cell(self) -> None:
        """Compatibility alias for integrations predating the async line."""

        self.reset_async_line()

    def flexible_cell_snapshot(self) -> dict[str, Any]:
        module_owners: dict[str, str] = {}
        for module in CHANGEOVER_COMPONENT_NAMES:
            copy = module.rsplit("_", 1)[-1]
            owner = "free"
            for label, weld in (
                ("module_rack", f"{module}_rack_weld"),
                ("changeover_gantry", f"{module}_gantry_weld"),
                (f"turntable_nest_{copy}", f"{module}_nest_{copy}_weld"),
            ):
                if bool(self.data.eq_active[self.equality_id(weld)]):
                    owner = label
                    break
            module_owners[module] = owner
        tray_owners: dict[str, str] = {}
        for unit in range(1, 4):
            tray = f"tray_{unit:02d}"
            owner = "logistics_or_cache"
            for nest in ("a", "b"):
                if bool(self.data.eq_active[self.equality_id(f"turntable_nest_{nest}_tray_{unit:02d}_weld")]):
                    owner = f"turntable_nest_{nest}"
            tray_owners[tray] = owner
        return {
            "turntable_angle_deg": float(np.degrees(self.turntable_angle_rad) % 360.0),
            "turntable_velocity_deg_s": float(np.degrees(self.turntable_velocity_rad_s)),
            "nest_a_lock_m": self.batch_joint_position("table2_nest_a_lock_joint"),
            "nest_b_lock_m": self.batch_joint_position("table2_nest_b_lock_joint"),
            "gantry_position_m": list(self.changeover_gantry_position()),
            "module_owners": module_owners,
            "tray_owners": tray_owners,
        }

    def set_batch_rack_lock(
        self,
        shelf_index: int,
        engaged: bool,
        *,
        teleport: bool = False,
    ) -> None:
        """Drive one physical rack lock pin and its status indicator."""

        index = int(shelf_index)
        if index not in {0, 1, 2}:
            raise ValueError("batch shelf index must be 0, 1 or 2")
        target = 0.025 if engaged else 0.0
        self.set_batch_joint_target(
            f"batch_rack_lock_joint_{index}",
            f"batch_rack_lock_actuator_{index}",
            target,
            teleport=teleport,
        )
        geom_id = self.geom_id(f"batch_rack_{index}_lock_pin")
        self.model.geom_rgba[geom_id, :3] = (
            np.asarray([0.96, 0.52, 0.08]) if engaged else np.asarray([0.35, 0.40, 0.45])
        )
        indicator_id = self.geom_id(f"batch_rack_{index}_lock_indicator")
        self.model.geom_rgba[indicator_id, :3] = (
            np.asarray([0.03, 0.85, 0.24]) if engaged else np.asarray([0.95, 0.035, 0.025])
        )
        if teleport:
            self.mujoco.mj_forward(self.model, self.data)

    def reset_batch_cell(self, *, show_empty_cache: bool = False) -> None:
        """Home every axis while keeping preallocated empty trays invisible.

        ``show_empty_cache`` remains accepted for API compatibility, but empty
        carrier boards are deliberately never rendered.
        """

        axes = (
            ("batch_outfeed_joint", "batch_outfeed_actuator"),
            ("batch_output_joint", "batch_output_actuator"),
            ("batch_tray_02_index_joint", "batch_tray_02_index_actuator"),
            ("batch_tray_03_index_joint", "batch_tray_03_index_actuator"),
            ("finished_output_gate_joint", "finished_output_gate_actuator"),
        )
        for joint_name, actuator_name in axes:
            self.set_batch_joint_target(joint_name, actuator_name, 0.0, teleport=True)
        for unit in range(1, 4):
            for station in ASYNC_STATION_ANCHORS:
                self.set_batch_weld(f"station_{station}_tray_{unit:02d}_weld", False)
            for transfer in ASYNC_TRANSFER_SPECS:
                self.set_batch_weld(f"transfer_{transfer}_tray_{unit:02d}_weld", False)
            for owner in ("carrier", "rack", "output"):
                self.set_batch_weld(f"batch_{owner}_tray_{unit:02d}_weld", False)
            for shelf in range(3):
                self.set_batch_weld(
                    f"batch_rack_tray_{unit:02d}_shelf_{shelf}_weld",
                    False,
                )
            self.set_batch_rack_lock(unit - 1, False, teleport=True)
        owners = (
            ("batch_station_tray_01_weld", "batch_tray_01_station_anchor", "batch_tray_01"),
            ("batch_indexer_tray_02_weld", "batch_tray_02_indexer_anchor", "batch_tray_02"),
            ("batch_indexer_tray_03_weld", "batch_tray_03_indexer_anchor", "batch_tray_03"),
        )
        # Delivery parks free trays outside the visible cell. Re-seat every
        # body before restoring its cache weld so reset is also correct when a
        # new queued order starts without rebuilding the viewer.
        for weld, anchor, tray in owners:
            anchor_pose = _pose_from_body(self.data, self.body_id(anchor))
            self.set_free_body_pose(tray, anchor_pose, forward=True)
            self.set_batch_weld(weld, True, recompute=(anchor, tray))
        for index in range(3):
            self.set_batch_tray_visible(
                index,
                carrier=False,
                payload=False,
            )
        self.mujoco.mj_forward(self.model, self.data)

    def reset_dynamic_welds(self) -> None:
        """Restore configured workpiece welds and quick-change rack ownership."""

        for name in self.handles.welds:
            # Batch carriers already locked in the furnace rack are owned by
            # BatchTransferActor. Preparing the reusable Table2 workcell for
            # the next unit must never release or reassign those constraints.
            if name.startswith("batch_"):
                continue
            active = name in {
                "station_s1_assembly_tray_weld",
                "base_tray_weld",
                "arm1_rack_parallel_gripper",
                "arm1_rack_suction_tool",
                "arm2_dispenser_tool_weld",
            }
            if name.endswith("_fixture_weld") and name.startswith("fin_"):
                index = int(name[4:6])
                active = index <= self.active_fin_count
            if name.startswith("slot_") and name.endswith("_brazing_path_base_weld"):
                path_name = name[: -len("_base_weld")]
                active = PATH_NAMES.index(path_name) < self.active_path_count
            self.data.eq_active[self.handles.welds[name]] = int(active)

    def release_process_welds(self) -> None:
        """Emergency release of workpiece/tool process constraints.

        Quick-change managers restore their rack constraints immediately after
        this generic process release.
        """

        for equality_id in self.handles.welds.values():
            self.data.eq_active[equality_id] = 0
        self.mujoco.mj_forward(self.model, self.data)

    def contract_snapshot(self) -> dict[str, Any]:
        return {
            "arms": tuple(self.handles.arm_sites),
            "fins": tuple(self.handles.fins),
            "paths": tuple(self.handles.paths),
            "tools": ("brazing_dispenser",),
            "cameras": tuple(self.handles.cameras),
            "raw_sites": tuple(self.handles.raw_sites),
            "active_fins": self.active_fin_count,
            "active_paths": self.active_path_count,
            "flexible_cell": {
                "layout": "SHALLOW_U_ASYNC",
                "stations": tuple(ASYNC_STATION_ANCHORS.values()),
                "transfers": tuple(ASYNC_TRANSFER_SPECS),
                "station_positions_m": {
                    "s1": SHALLOW_U_LAYOUT.station_s1_xy,
                    "s2a": SHALLOW_U_LAYOUT.station_s2a_xy,
                    "s2b": SHALLOW_U_LAYOUT.station_s2b_xy,
                    "s3": SHALLOW_U_LAYOUT.station_s3_xy,
                    "rack_infeed": SHALLOW_U_LAYOUT.rack_infeed_xy,
                },
            },
        }


class BrazingScene:
    """Convenient owner of model, data, registry, controllers and tool changer."""

    def __init__(
        self,
        xml_path: str | Path | None = None,
        *,
        order: OrderSpec | ProductState | str = "A",
        motion_config: MotionConfig | None = None,
        raw: bool = True,
    ) -> None:
        import mujoco

        path = Path(xml_path) if xml_path is not None else default_scene_path()
        if not path.exists():
            raise FileNotFoundError(path)
        self.mujoco = mujoco
        self.path = path.resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.registry = SceneRegistry(self.model, self.data)
        self.registry.enable_robot_gravity_compensation()
        self.registry.set_arm1_gripper_closed(0.0)
        self.registry.set_arm1_suction_fraction(0.0)
        # The camera is a true centred flange extension.  Deriving this offset
        # from an arbitrary XML world pose previously preserved an 88 mm
        # lateral error and made the camera appear beside the wrist.
        self._mounted_offsets = {
            "arm3_camera_rig": Pose(
                np.asarray([0.0, 0.0, 0.107], dtype=float),
                np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
            ),
        }
        self.arms = {name: ArmController(self.model, self.data, name, motion_config) for name in ARM_NAMES}
        for name, controller in self.arms.items():
            controller.reset(_arm_home_qpos(name))
            controller.enabled = False
        self._snap_extensions()
        self.registry.set_weld(
            "arm3_camera_mount",
            True,
            recompute=("arm3_fr3_link7", "arm3_camera_rig"),
            forward=True,
        )
        self.arm1_tools = Arm1ToolManager(self.model, self.data, self.arms["arm1"])
        self.tools = Arm2ToolManager(self.model, self.data, self.arms["arm2"])
        self.product: ProductState | None = order if isinstance(order, ProductState) else None
        self.fins, self.paths = self.registry.configure_product(order)
        self.tools.reset_mounted()
        self.fixture_controller = FixtureController(self, state=FixtureState())
        selected_spec = (
            self.product.spec
            if self.product is not None
            else order if isinstance(order, OrderSpec) else make_order_spec(str(order))
        )
        self.fixture_controller.configure_product(selected_spec)
        self.fixture_controller.reset(FixtureState(), hard=True)
        # Static A/B/C checks catch missing or mismatched MJCF names at startup;
        # the current-site check validates the mutable order configuration.
        preflight_check(self, order=("A", "B", "C"), xml_file=self.path)
        self.preflight_report: PreflightReport = preflight_check(
            self,
            order=selected_spec,
            xml_file=self.path,
            validate_current_sites=True,
        )
        if raw:
            self.registry.deactivate_comb_modules()
            self.registry.prepare_raw_materials(self.product)
        self.renderer: Any | None = None

    def _relative_pose(self, parent: str, child: str) -> Pose:
        left = _pose_from_body(self.data, int(self.model.body(parent).id))
        right = _pose_from_body(self.data, int(self.model.body(child).id))
        return left.inverse().transformed(right)

    def _snap_extensions(self, *, forward: bool = True) -> None:
        for child, relative in self._mounted_offsets.items():
            parent = "arm3_fr3_link7"
            parent_pose = _pose_from_body(self.data, int(self.model.body(parent).id))
            self.registry.set_free_body_pose(child, parent_pose.transformed(relative))
        if forward:
            self.mujoco.mj_forward(self.model, self.data)

    def sync_mounted_extensions(self, arm_name: str | None = None) -> None:
        """Synchronize welded free-body tools after a kinematic joint update."""

        # Kinematic playback writes robot qpos directly, so refresh body/site
        # transforms first. ``mj_forward`` also computes dynamics, contacts and
        # constraints and used to run two or three times here on every 2 ms
        # physics step. Two lightweight kinematics passes are sufficient: one
        # for the new flange poses and one after snapping the free-body tools.
        self.mujoco.mj_kinematics(self.model, self.data)
        if arm_name in {None, "arm3"}:
            self._snap_extensions(forward=False)
        if arm_name in {None, "arm1"}:
            self.arm1_tools.sync_mounted(forward=False)
        if arm_name in {None, "arm2"}:
            self.tools.sync_mounted(forward=False)
        self.mujoco.mj_kinematics(self.model, self.data)

    @property
    def time(self) -> float:
        return float(self.data.time)

    def reset(self, order: OrderSpec | ProductState | str = "A", *, raw: bool = True) -> None:
        # Runtime resets must not move the simulation clock backwards. Actors,
        # the furnace and batch leases all use absolute simulation timestamps;
        # resetting ``data.time`` to zero after they were created makes the
        # next update fail its monotonic-clock interlock.
        simulation_time = float(self.data.time)
        self.mujoco.mj_resetData(self.model, self.data)
        self.data.time = simulation_time
        self.mujoco.mj_forward(self.model, self.data)
        # ``mj_resetData`` resets data/qpos but model-level RGBA and collision
        # masks survive. Restore those before configuring the next product.
        self.registry.set_workcell_visible(True)
        self.registry.enable_robot_gravity_compensation()
        for name, controller in self.arms.items():
            controller.reset(_arm_home_qpos(name))
            controller.enabled = False
        self._snap_extensions()
        self.registry.set_weld(
            "arm3_camera_mount",
            True,
            recompute=("arm3_fr3_link7", "arm3_camera_rig"),
            forward=True,
        )
        self.arm1_tools.reset_to_rack()
        self.tools.reset_mounted()
        self.product = order if isinstance(order, ProductState) else None
        self.fins, self.paths = self.registry.configure_product(order)
        self.tools.reset_mounted()
        selected_spec = (
            self.product.spec
            if self.product is not None
            else order if isinstance(order, OrderSpec) else make_order_spec(str(order))
        )
        self.fixture_controller.state = FixtureState()
        self.fixture_controller.configure_product(selected_spec)
        self.fixture_controller.reset(self.fixture_controller.state, hard=True)
        self.registry.set_conveyor_target(0.0, teleport=True)
        self.registry.reset_async_line()
        self.registry.reset_dynamic_welds()
        self.registry.set_fixture_lock_visual(False)
        self.registry.set_arm1_gripper_closed(0.0)
        self.registry.set_arm1_suction_fraction(0.0)
        if raw:
            self.registry.deactivate_comb_modules()
            self.registry.prepare_raw_materials(self.product)
        self.registry.set_furnace_door(0.0, teleport=True)
        self.mujoco.mj_forward(self.model, self.data)
        self.preflight_report = preflight_check(
            self,
            order=selected_spec,
            xml_file=self.path,
            validate_current_sites=True,
        )

    def reset_workcell(self, product: ProductState) -> None:
        """Prepare the reusable robots and Table2 representation for one batch unit.

        Unlike :meth:`reset`, this deliberately preserves simulation time,
        furnace state, transfer axes and the three independently parked batch
        carriers.
        """

        for name, controller in self.arms.items():
            controller.reset(_arm_home_qpos(name))
            controller.enabled = False
        self._snap_extensions()
        self.registry.set_weld(
            "arm3_camera_mount",
            True,
            recompute=("arm3_fr3_link7", "arm3_camera_rig"),
            forward=True,
        )
        self.arm1_tools.reset_to_rack()
        self.tools.reset_mounted()
        self.registry.set_conveyor_target(0.0, teleport=True)
        self.registry.set_workcell_visible(True)
        self.product = product
        self.fins, self.paths = self.registry.configure_product(product)
        self.tools.reset_mounted()
        self.fixture_controller.state = product.fixture
        self.fixture_controller.configure_product(product.spec, product.fixture)
        self.fixture_controller.reset(product.fixture, hard=True)
        self.registry.reset_dynamic_welds()
        self.registry.set_fixture_lock_visual(False)
        self.registry.set_arm1_gripper_closed(0.0)
        self.registry.set_arm1_suction_fraction(0.0)
        self.registry.deactivate_comb_modules()
        self.registry.prepare_raw_materials(product)
        self.mujoco.mj_forward(self.model, self.data)

    def step(self, steps: int = 1) -> None:
        for _ in range(max(0, int(steps))):
            for controller in self.arms.values():
                controller.control_tick()
            self.fixture_controller.enforce_hold()
            self.mujoco.mj_step(self.model, self.data)

    def stop(self, reason: str = "safe stop") -> None:
        for controller in self.arms.values():
            controller.stop(reason)
        self.registry.release_process_welds()
        self.arm1_tools.reset_to_rack()
        self.tools.reset_mounted()
        self.fixture_controller.reset(FixtureState(), hard=True)
        self.registry.set_furnace_door(0.0, teleport=True)

    def camera_rgb(
        self, width: int = 640, height: int = 480, camera: str = "arm3_wrist_camera"
    ) -> np.ndarray:
        if self.renderer is None or self.renderer.width != width or self.renderer.height != height:
            if self.renderer is not None:
                self.renderer.close()
            # Keep the XML's inexpensive 640 x 480 default, but preserve the
            # public CLI contract for users who explicitly request a larger
            # inspection image.
            self.model.vis.global_.offwidth = max(
                int(self.model.vis.global_.offwidth),
                int(width),
            )
            self.model.vis.global_.offheight = max(
                int(self.model.vis.global_.offheight),
                int(height),
            )
            self.renderer = self.mujoco.Renderer(self.model, height=int(height), width=int(width))
        self.renderer.update_scene(self.data, camera=camera)
        # The independent Arm3 preview is a truth-inspection aid, not the main
        # presentation render.  Disabling its second shadow/reflection pass
        # avoids competing with interactive viewer orbiting; geometry,
        # materials and image resolution remain unchanged.
        self.renderer.scene.flags[int(self.mujoco.mjtRndFlag.mjRND_SHADOW)] = 0
        self.renderer.scene.flags[int(self.mujoco.mjtRndFlag.mjRND_REFLECTION)] = 0
        return np.asarray(self.renderer.render()).copy()

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def load_scene(
    xml_path: str | Path | None = None,
    *,
    order: OrderSpec | ProductState | str = "A",
    motion_config: MotionConfig | None = None,
    raw: bool = True,
) -> BrazingScene:
    return BrazingScene(xml_path, order=order, motion_config=motion_config, raw=raw)


# Straightforward alias for entrypoints/tests that prefer the short name.
Scene = BrazingScene


__all__ = [
    "ARM_NAMES",
    "BrazingScene",
    "FIN_NAMES",
    "PATH_NAMES",
    "Scene",
    "SceneContractError",
    "SceneHandles",
    "SceneRegistry",
    "load_scene",
]
