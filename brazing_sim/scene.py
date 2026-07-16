"""MuJoCo scene contract, product configuration and runtime scene facade."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import derive_product_layout, make_order_spec
from .domain import BrazingPathState, FinState, OrderSpec, ProductState
from .motion import ArmController, MotionConfig, Pose, default_scene_path, matrix_to_quat
from .tools import Arm1ToolManager, Arm2ToolManager

ARM_NAMES = ("arm1", "arm2", "arm3")
FIN_NAMES = tuple(f"fin_{index:02d}" for index in range(1, 9))
PATH_NAMES = tuple(
    f"brazing_path_fin_{index:02d}_{side}" for index in range(1, 9) for side in ("left", "right")
)
HOME_QPOS = np.asarray([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=float)


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
        self.active_fin_count = 4
        self.active_path_count = 8
        self.assembly_base_pose = Pose(np.asarray([0.0, 0.45, 0.240]), np.asarray([1.0, 0.0, 0.0, 0.0]))
        self.fin_local_targets: dict[str, Pose] = {}
        self.handles = self._resolve_contract()
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
        for name in ("assembly_tray", "base_plate", *FIN_NAMES, *PATH_NAMES):
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
            "arm1_grasp_base",
            "arm2_tray_carry",
            "furnace_tray_weld",
            "arm1_toolchange_parallel_gripper",
            "arm1_toolchange_suction_tool",
            "arm1_rack_parallel_gripper",
            "arm1_rack_suction_tool",
            "arm2_toolchange_brazing_dispenser",
            "arm2_toolchange_tray_transfer",
            "arm2_rack_brazing_dispenser",
            "arm2_rack_tray_transfer",
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
        return self._geom_ids[name]

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
        self.model.geom_contype[geom_id] = 2 if active and collide else 0
        self.model.geom_conaffinity[geom_id] = 3 if active and collide else 0

    def _set_site_active(self, site_name: str, active: bool) -> None:
        site_id = self._id(self.mujoco.mjtObj.mjOBJ_SITE, site_name)
        self._initial_site_rgba.setdefault(
            site_id, np.asarray(self.model.site_rgba[site_id], dtype=float).copy()
        )
        rgba = self._initial_site_rgba[site_id].copy()
        rgba[3] = rgba[3] if active else 0.0
        self.model.site_rgba[site_id] = rgba

    def set_path_visible(self, path_id: str, visible: bool = True, *, coverage: float = 1.0) -> None:
        name = path_id if path_id.startswith("brazing_path_") else f"brazing_path_{path_id}"
        geom_id = self.geom_id(name + "_geom")
        rgba = self._initial_rgba[geom_id].copy()
        rgba[3] = float(np.clip(coverage, 0.0, 1.0)) * (rgba[3] if visible else 0.0)
        self.model.geom_rgba[geom_id] = rgba

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
        self.mujoco.mj_forward(self.model, self.data)

    def arm1_suction_fraction(self) -> float:
        """Return the visual suction engagement in the range ``[0, 1]``."""

        half_height = float(self.model.geom_size[self.arm1_suction_pad, 1])
        return float(np.clip((0.004 - half_height) / 0.0015, 0.0, 1.0))

    def configure_product(
        self, order: OrderSpec | ProductState | str = "A"
    ) -> tuple[list[FinState], list[BrazingPathState]]:
        """Resize and configure the scene's fixed 8-fin/16-path allocation."""

        if isinstance(order, ProductState):
            spec = order.spec
            fins = order.fins
            paths = order.paths
        else:
            spec = make_order_spec(order) if isinstance(order, str) else order
            layout = derive_product_layout(spec)
            fins, paths = layout.fins, layout.paths

        if spec.max_fins > len(FIN_NAMES) or spec.max_paths > len(PATH_NAMES):
            raise ValueError("order exceeds the scene allocation of 8 fins / 16 paths")
        self.active_fin_count = spec.fin_count
        self.active_path_count = spec.path_count
        # +X is left-right along the Arm1-to-Arm3 direction. Keep the base on
        # Table1's left half and derive a centred row of X-aligned fins on the
        # right. Four-fin A orders use extra 70 mm gripper clearance; future
        # 6/8-fin orders retain at least 60 mm without rebuilding the model.
        self.model.site_pos[self.handles.raw_sites["base_plate"]] = np.asarray([-0.23, 0.0, 0.105])
        fin_spacing = 0.07 if spec.fin_count <= 4 else 0.06
        row_offset = 0.035 if spec.fin_count <= 4 else 0.0
        for index, name in enumerate(FIN_NAMES):
            if index < spec.fin_count:
                y_position = (index - 0.5 * (spec.fin_count - 1)) * fin_spacing + row_offset
            else:
                y_position = 0.30 + 0.03 * (index - spec.fin_count)
            self.model.site_pos[self.handles.raw_sites[name]] = np.asarray([0.23, y_position, 0.130])
        base_geom = self.geom_id("base_plate_geom")
        self.model.geom_size[base_geom, :3] = np.asarray(spec.base_size, dtype=float) / 2.0
        product_pose = self.product_pose()
        self.assembly_base_pose = product_pose
        self.fin_local_targets.clear()

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
            world_pose = product_pose.transformed(Pose(midpoint, np.asarray([1.0, 0.0, 0.0, 0.0])))
            self.set_weld(f"{name}_base_weld", False)
            self.set_free_body_pose(name, world_pose)
            geom_name = name + "_geom"
            geom_id = self.geom_id(geom_name)
            self.model.geom_matid[geom_id] = -1
            self._initial_rgba[geom_id] = np.asarray([0.94, 0.55, 0.10, 0.90])
            self.model.geom_size[geom_id, 0] = float(path.target_width_m) / 2.0
            self.model.geom_size[geom_id, 1] = float(spec.fin_length) / 2.0
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

    def _site_pose(self, site_id: int) -> Pose:
        site = self.data.site(site_id)
        return Pose(
            np.asarray(site.xpos, dtype=float),
            matrix_to_quat(np.asarray(site.xmat, dtype=float).reshape(3, 3)),
        )

    def place_base_on_tray(self, *, snap: bool = True) -> None:
        """Move the base to the Table2 product origin and enable its tray weld."""

        self.set_weld("raw_base_rack_weld", False)
        self.set_weld("arm1_grasp_base", False)
        if snap:
            self.set_free_body_pose("base_plate", self.assembly_base_pose)
        self.mujoco.mj_forward(self.model, self.data)
        self.set_weld(
            "base_tray_weld",
            True,
            recompute=("assembly_tray", "base_plate"),
            forward=True,
        )
        for index, name in enumerate(PATH_NAMES):
            self.set_weld(
                f"{name}_base_weld",
                index < self.active_path_count,
                recompute=("base_plate", name) if index < self.active_path_count else None,
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

    def carry_tray(self, active: bool) -> None:
        if active:
            self.set_weld("tray_fixture_weld", False)
            self.set_weld("furnace_tray_weld", False)
        self.set_weld(
            "arm2_tray_carry",
            active,
            recompute=("arm2_tray_transfer", "assembly_tray") if active else None,
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
        for geom_name in ("fixture_base", "fixture_comb_left", "fixture_comb_right"):
            geom_id = self.geom_id(geom_name)
            rgba = self._initial_rgba[geom_id].copy()
            if locked:
                rgba[:3] = np.asarray([0.92, 0.52, 0.10])
            self.model.geom_rgba[geom_id] = rgba

    def place_tray_in_furnace(self) -> None:
        self.set_weld("arm2_tray_carry", False)
        self.set_weld(
            "furnace_tray_weld",
            True,
            recompute=("furnace", "assembly_tray"),
            forward=True,
        )

    def set_furnace_door(self, fraction: float, *, teleport: bool = False) -> None:
        amount = float(np.clip(fraction, 0.0, 1.0))
        target = 0.46 * amount
        self.data.ctrl[self.handles.furnace_door_actuator] = target
        if teleport:
            address = int(self.model.jnt_qposadr[self.handles.furnace_door_joint])
            self.data.qpos[address] = target
            self.mujoco.mj_forward(self.model, self.data)

    def reset_dynamic_welds(self) -> None:
        """Restore a configured, assembled fixture and parked tools."""

        for name in self.handles.welds:
            active = name in {
                "tray_fixture_weld",
                "base_tray_weld",
                "arm1_rack_parallel_gripper",
                "arm1_rack_suction_tool",
                "arm2_rack_brazing_dispenser",
                "arm2_rack_tray_transfer",
            }
            if name.endswith("_fixture_weld") and name.startswith("fin_"):
                index = int(name[4:6])
                active = index <= self.active_fin_count
            if name.startswith("brazing_path_") and name.endswith("_base_weld"):
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
            "tools": ("brazing_dispenser", "tray_transfer"),
            "cameras": tuple(self.handles.cameras),
            "raw_sites": tuple(self.handles.raw_sites),
            "active_fins": self.active_fin_count,
            "active_paths": self.active_path_count,
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
        self._mounted_offsets = {
            "arm3_camera_rig": self._relative_pose("arm3_fr3_link7", "arm3_camera_rig"),
        }
        self.arms = {name: ArmController(self.model, self.data, name, motion_config) for name in ARM_NAMES}
        for controller in self.arms.values():
            controller.reset(HOME_QPOS)
            controller.enabled = False
        self._snap_extensions()
        self.arm1_tools = Arm1ToolManager(self.model, self.data, self.arms["arm1"])
        self.tools = Arm2ToolManager(self.model, self.data, self.arms["arm2"])
        self.product: ProductState | None = order if isinstance(order, ProductState) else None
        self.fins, self.paths = self.registry.configure_product(order)
        if raw:
            self.registry.prepare_raw_materials(self.product)
        self.renderer: Any | None = None

    def _relative_pose(self, parent: str, child: str) -> Pose:
        left = _pose_from_body(self.data, int(self.model.body(parent).id))
        right = _pose_from_body(self.data, int(self.model.body(child).id))
        return left.inverse().transformed(right)

    def _snap_extensions(self) -> None:
        for child, relative in self._mounted_offsets.items():
            parent = "arm3_fr3_link7"
            parent_pose = _pose_from_body(self.data, int(self.model.body(parent).id))
            self.registry.set_free_body_pose(child, parent_pose.transformed(relative))
        self.mujoco.mj_forward(self.model, self.data)

    def sync_mounted_extensions(self) -> None:
        """Synchronize welded free-body tools after a kinematic joint update."""

        self._snap_extensions()
        self.arm1_tools.sync_mounted()
        self.tools.sync_mounted()

    @property
    def time(self) -> float:
        return float(self.data.time)

    def reset(self, order: OrderSpec | ProductState | str = "A", *, raw: bool = True) -> None:
        self.mujoco.mj_resetData(self.model, self.data)
        self.mujoco.mj_forward(self.model, self.data)
        self.registry.enable_robot_gravity_compensation()
        for controller in self.arms.values():
            controller.reset(HOME_QPOS)
            controller.enabled = False
        self._snap_extensions()
        self.arm1_tools.reset_to_rack()
        self.tools.reset_to_rack()
        self.product = order if isinstance(order, ProductState) else None
        self.fins, self.paths = self.registry.configure_product(order)
        self.registry.reset_dynamic_welds()
        self.registry.set_fixture_lock_visual(False)
        self.registry.set_arm1_gripper_closed(0.0)
        self.registry.set_arm1_suction_fraction(0.0)
        if raw:
            self.registry.prepare_raw_materials(self.product)
        self.registry.set_furnace_door(0.0, teleport=True)
        self.mujoco.mj_forward(self.model, self.data)

    def step(self, steps: int = 1) -> None:
        for _ in range(max(0, int(steps))):
            for controller in self.arms.values():
                controller.control_tick()
            self.mujoco.mj_step(self.model, self.data)

    def stop(self, reason: str = "safe stop") -> None:
        for controller in self.arms.values():
            controller.stop(reason)
        self.registry.release_process_welds()
        self.arm1_tools.reset_to_rack()
        self.tools.reset_to_rack()
        self.registry.set_furnace_door(0.0, teleport=True)

    def camera_rgb(
        self, width: int = 640, height: int = 480, camera: str = "arm3_wrist_camera"
    ) -> np.ndarray:
        if self.renderer is None or self.renderer.width != width or self.renderer.height != height:
            if self.renderer is not None:
                self.renderer.close()
            self.renderer = self.mujoco.Renderer(self.model, height=int(height), width=int(width))
        self.renderer.update_scene(self.data, camera=camera)
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
