"""Projection of V2 runtime truth into the independent MuJoCo scene."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..motion import HOME_QPOS
from ..profiles import quintic_time_scaling
from .process_geometry import V2ProcessGeometry
from .robot_projector import V2RobotMotionProjector
from .fault_visuals import (
    BRAZING_DEVIATION_REAPPLY_START,
    BRAZING_DEVIATION_REMOVAL_END,
    V2FaultVisualizer,
)
from .tray_flow import TrayOwner

if TYPE_CHECKING:
    from .runtime import DualLineRuntime


_V1_COMB_POST_HALF_X_M = 0.012
_V1_COMB_POST_HALF_Y_M = 0.008
_V1_COMB_GUIDE_HALF_X_M = 0.045
_V1_COMB_LONGITUDINAL_CLEARANCE_M = 0.008
_V1_COMB_LATERAL_CLEARANCE_M = 0.010
_BRANCH_B_CORNER_CLEAR_X_M = 0.15
_BRANCH_B_WEST_CLEAR_X_M = 0.115
_BRANCH_B_CORRIDOR_ENTRY_Y_M = -0.475
_BRANCH_B_SOUTH_CLEAR_Y_M = -1.22


@dataclass(slots=True)
class _CarrierMotion:
    start: np.ndarray
    target: np.ndarray
    started_at: float
    duration_s: float
    source_owner: TrayOwner
    target_owner: TrayOwner
    remaining_targets: deque[np.ndarray]
    segment_index: int
    segment_count: int
    paused_at: float | None = None


@dataclass(slots=True)
class _InspectionRecord:
    unit_id: str
    kind: str
    camera: str
    captured_at: float
    aligned: bool = True
    clear: bool = True
    analysis_seconds: float = 5.0
    analysis_complete: bool = False


class DualLineSceneAdapter:
    """Move trays by one permanent carrier constraint per physical pallet.

    The adapter never writes a tray free joint.  A mocap conveyor carrier is
    advanced continuously and the permanent weld makes the rigid tray follow.
    Virtual empty-pallet return is the only intentionally invisible jump.
    """

    def __init__(
        self,
        xml_path: str | Path,
        *,
        transfer_speed_m_s: float = 0.35,
        minimum_transfer_s: float = 0.55,
        fast_base_speed_scale: float = 4.0,
        fast_process_speed_scale: float = 3.0,
    ) -> None:
        import mujoco

        if transfer_speed_m_s <= 0 or minimum_transfer_s <= 0:
            raise ValueError("V2 carrier timing must be positive")
        self.mujoco = mujoco
        self.xml_path = Path(xml_path).expanduser().resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        for arm_name in ("arm1", "arm2", "arm3"):
            qpos_ids: list[int] = []
            dof_ids: list[int] = []
            actuator_ids: list[int] = []
            for joint_index in range(1, 8):
                joint = self.model.joint(f"{arm_name}_fr3_joint{joint_index}")
                qpos_ids.append(int(self.model.jnt_qposadr[joint.id]))
                dof_ids.append(int(self.model.jnt_dofadr[joint.id]))
                actuator_ids.append(int(self.model.actuator(f"{arm_name}_fr3_joint{joint_index}").id))
            self.data.qpos[qpos_ids] = HOME_QPOS
            self.data.qvel[dof_ids] = 0.0
            self.data.ctrl[actuator_ids] = HOME_QPOS
        mujoco.mj_forward(self.model, self.data)
        self.transfer_speed_m_s = float(transfer_speed_m_s)
        self.minimum_transfer_s = float(minimum_transfer_s)
        # ``fast`` is a headless/rehearsal concern supplied by the bound
        # runtime.  Keep the authored 1x carrier speed readable in the viewer,
        # while allowing regression and batch rehearsals to compress wall
        # time without changing routes, ownership, or stop points.
        self._motion_speed_scale = 1.0
        # Mirrors fault state into the rendered scene.  Owns the only MuJoCo
        # dependency in V2's fault stack.
        self.fault_visuals = V2FaultVisualizer(self)
        self._camera_renderer = None
        self._tray_ids = tuple(f"V2_TRAY_{index:02d}" for index in range(1, 7))
        self._body_names = {tray_id: tray_id.lower() for tray_id in self._tray_ids}
        self._mocap_ids: dict[str, int] = {}
        self._primary_geom_ids: dict[str, int] = {}
        self._tray_geom_ids: dict[str, tuple[int, ...]] = {}
        self._component_geom_ids: dict[tuple[str, str], int] = {}
        self._visible_rgba: dict[int, np.ndarray] = {}
        self._original_geom_pos: dict[int, np.ndarray] = {}
        self._original_geom_quat: dict[int, np.ndarray] = {}
        self._original_geom_size: dict[int, np.ndarray] = {}
        self._collision: dict[int, tuple[int, int]] = {}
        self._geom_visibility: dict[int, bool] = {}
        self._home_positions: dict[str, np.ndarray] = {}
        self._last_owner = {tray_id: TrayOwner.EMPTY_BUFFER for tray_id in self._tray_ids}
        self._physical_owner = {tray_id: TrayOwner.EMPTY_BUFFER for tray_id in self._tray_ids}
        self._configured_unit_by_tray: dict[str, str] = {}
        self._bound_runtime: DualLineRuntime | None = None
        self._last_runtime_time = 0.0
        self._inspection_records: dict[tuple[str, str], _InspectionRecord] = {}
        self._pending_owners: dict[str, deque[TrayOwner]] = {tray_id: deque() for tray_id in self._tray_ids}
        self._motions: dict[str, _CarrierMotion] = {}
        for tray_id, body_name in self._body_names.items():
            carrier = self.model.body(f"{body_name}_carrier")
            mocap_id = int(self.model.body_mocapid[carrier.id])
            if mocap_id < 0:
                raise ValueError(f"{body_name}_carrier must be a mocap body")
            geom_id = int(self.model.geom(f"{body_name}_geom").id)
            self._mocap_ids[tray_id] = mocap_id
            self._primary_geom_ids[tray_id] = geom_id
            prefix = f"{body_name}_"
            tray_geom_ids: list[int] = []
            for candidate in range(int(self.model.ngeom)):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, candidate)
                if name is None or not name.startswith(prefix):
                    continue
                tray_geom_ids.append(candidate)
                self._visible_rgba[candidate] = np.asarray(
                    self.model.geom_rgba[candidate],
                    dtype=float,
                ).copy()
                self._original_geom_pos[candidate] = np.asarray(
                    self.model.geom_pos[candidate],
                    dtype=float,
                ).copy()
                self._original_geom_quat[candidate] = np.asarray(
                    self.model.geom_quat[candidate],
                    dtype=float,
                ).copy()
                self._original_geom_size[candidate] = np.asarray(
                    self.model.geom_size[candidate],
                    dtype=float,
                ).copy()
                self._collision[candidate] = (
                    int(self.model.geom_contype[candidate]),
                    int(self.model.geom_conaffinity[candidate]),
                )
                component = name.removeprefix(prefix)
                self._component_geom_ids[(tray_id, component)] = candidate
            self._tray_geom_ids[tray_id] = tuple(tray_geom_ids)
            self._home_positions[tray_id] = np.asarray(self.data.mocap_pos[mocap_id], dtype=float).copy()
        self._actuators = {
            name: int(self.model.actuator(name).id)
            for name in (
                "v2_furnace_front_door_actuator",
                "v2_furnace_rear_door_actuator",
                "v2_furnace_lift_actuator",
                "v2_furnace_pusher_actuator",
                "v2_furnace_rear_lift_actuator",
                "v2_furnace_rear_extractor_actuator",
                "v2_finished_output_gate_actuator",
                "v2_furnace_layer_0_lock_actuator",
                "v2_furnace_layer_1_lock_actuator",
                "v2_furnace_layer_2_lock_actuator",
            )
        }
        self._mechanism_qpos = {
            name: int(self.model.jnt_qposadr[self.model.joint(name).id])
            for name in (
                "v2_furnace_front_door_joint",
                "v2_furnace_rear_door_joint",
                "v2_furnace_lift_joint",
                "v2_furnace_pusher_joint",
                "v2_furnace_rear_lift_joint",
                "v2_furnace_rear_extractor_joint",
                "v2_finished_output_gate_joint",
            )
        }
        self._route_actuators = {
            name: int(self.model.actuator(name).id)
            for name in (
                "v2_s1_s2a_actuator",
                "v2_s2a_s2b_actuator",
                "v2_branch_a_actuator",
                "v2_branch_b_actuator",
                "v2_buffer_index_actuator",
                "v2_output_transfer_actuator",
            )
        }
        self._set_all_hidden()
        mujoco.mj_forward(self.model, self.data)
        self._robots = V2RobotMotionProjector(
            self.model,
            self.data,
            set_component_visible=self._set_component_visible,
            restore_component_pose=self._restore_component_pose,
            fast_base_speed_scale=fast_base_speed_scale,
            fast_process_speed_scale=fast_process_speed_scale,
        )

    def _set_all_hidden(self) -> None:
        for tray_id in self._tray_ids:
            self._set_tray_visible(tray_id, False)

    def _set_geom_visible(self, geom_id: int, visible: bool) -> None:
        visible = bool(visible)
        if self._geom_visibility.get(geom_id) is visible:
            return
        rgba = self._visible_rgba[geom_id].copy()
        rgba[3] = self._visible_rgba[geom_id][3] if visible else 0.0
        self.model.geom_rgba[geom_id] = rgba
        contype, conaffinity = self._collision[geom_id]
        self.model.geom_contype[geom_id] = contype if visible else 0
        self.model.geom_conaffinity[geom_id] = conaffinity if visible else 0
        self._geom_visibility[geom_id] = visible

    def _set_tray_visible(self, tray_id: str, visible: bool) -> None:
        for geom_id in self._tray_geom_ids[tray_id]:
            self._set_geom_visible(geom_id, visible)

    def _set_component_visible(self, tray_id: str, component: str, visible: bool) -> None:
        geom_id = self._component_geom_ids.get((tray_id, component))
        if geom_id is not None:
            self._set_geom_visible(geom_id, visible)

    def _restore_component_pose(self, tray_id: str, component: str) -> None:
        """Restore one tray component before a repaired visual handoff."""

        geom_id = self._component_geom_ids.get((tray_id, component))
        if geom_id is None:
            return
        self.model.geom_pos[geom_id] = self._original_geom_pos[geom_id]
        self.model.geom_quat[geom_id] = self._original_geom_quat[geom_id]
        self.model.geom_size[geom_id] = self._original_geom_size[geom_id]

    @staticmethod
    def _unit_for_tray(runtime: "DualLineRuntime", tray_id: str):
        """Select the active owner of a reusable tray, never stale history."""

        candidates = [item for item in runtime.units.values() if item.tray_id == tray_id]
        if not candidates:
            return None
        active = [item for item in candidates if item.stage.value != "COMPLETE"]
        pool = active or candidates
        return max(
            pool,
            key=lambda item: (
                float(item.stage_started_at),
                float("-inf") if item.completed_at is None else float(item.completed_at),
                item.unit_id,
            ),
        )

    def _configure_tray_geometry(self, tray_id: str, unit: Any) -> None:
        """Resize and reseat the reusable visual pool from the V1 process plan."""

        geometry = V2ProcessGeometry.for_unit(unit)
        base_id = self._component_geom_ids[(tray_id, "base_plate")]
        self.model.geom_size[base_id, :3] = np.asarray(geometry.base_size_m, dtype=float) / 2.0
        base_center_z = geometry.base_center_z_m
        self.model.geom_pos[base_id] = np.asarray([0.0, 0.0, base_center_z], dtype=float)

        template_id = self._component_geom_ids[(tray_id, "template_plate")]
        self.model.geom_size[template_id, :3] = np.asarray(
            [
                min(0.205, 0.5 * geometry.base_size_m[0] + 0.010),
                min(0.135, 0.5 * geometry.base_size_m[1] + 0.010),
                0.006,
            ],
            dtype=float,
        )
        fin_half_size = np.asarray(geometry.fin_size_m, dtype=float) / 2.0
        fin_centres = tuple(float(target[1]) for target in geometry.fin_targets)
        for index, target in enumerate(geometry.fin_targets, start=1):
            geom_id = self._component_geom_ids[(tray_id, f"fin_{index:02d}")]
            self.model.geom_size[geom_id, :3] = fin_half_size
            self.model.geom_pos[geom_id] = target
            self._original_geom_pos[geom_id] = target.copy()
            self._original_geom_size[geom_id] = self.model.geom_size[geom_id].copy()

        base_top = base_center_z + 0.5 * geometry.base_size_m[2]
        for index, path in enumerate(geometry.brazing_paths, start=1):
            geom_id = self._component_geom_ids[(tray_id, f"braze_{index:02d}")]
            start = np.asarray(path.start, dtype=float)
            end = np.asarray(path.end, dtype=float)
            midpoint = 0.5 * (start + end)
            bead_radius = 0.5 * geometry.path_width_m
            self.model.geom_pos[geom_id] = np.asarray(
                [midpoint[0], midpoint[1], base_top + bead_radius + 0.0002],
                dtype=float,
            )
            self.model.geom_quat[geom_id] = np.asarray(
                [0.7071067812, 0.0, 0.7071067812, 0.0],
                dtype=float,
            )
            self.model.geom_size[geom_id, 0] = bead_radius
            self.model.geom_size[geom_id, 1] = 0.5 * path.length_m
            self._original_geom_pos[geom_id] = self.model.geom_pos[geom_id].copy()
            self._original_geom_quat[geom_id] = self.model.geom_quat[geom_id].copy()
            self._original_geom_size[geom_id] = self.model.geom_size[geom_id].copy()

        support_x = 0.5 * geometry.fin_size_m[0] + _V1_COMB_LONGITUDINAL_CLEARANCE_M + _V1_COMB_POST_HALF_X_M
        support_y = 0.5 * geometry.base_size_m[1] + _V1_COMB_LATERAL_CLEARANCE_M + _V1_COMB_POST_HALF_Y_M
        slot_offset = max(
            0.003,
            0.5 * geometry.fin_size_m[1] + 0.002,
        )
        for side, x in (("front", -support_x), ("rear", support_x)):
            base_geom = self._component_geom_ids[(tray_id, f"{side}_comb_base")]
            self.model.geom_pos[base_geom, 0] = x
            self.model.geom_size[base_geom, 0] = _V1_COMB_POST_HALF_X_M
            self.model.geom_size[base_geom, 1] = support_y + _V1_COMB_POST_HALF_Y_M
            for post, y in (("left", -support_y), ("right", support_y)):
                post_geom = self._component_geom_ids[(tray_id, f"{side}_comb_post_{post}")]
                self.model.geom_pos[post_geom, 0] = x
                self.model.geom_pos[post_geom, 1] = y
            end_sign = -1.0 if side == "front" else 1.0
            guide_x = x - end_sign * (_V1_COMB_POST_HALF_X_M + _V1_COMB_GUIDE_HALF_X_M)
            for index, y in enumerate(fin_centres, start=1):
                for guide_side, y_sign in (("left", -1.0), ("right", 1.0)):
                    guide = self._component_geom_ids[
                        (tray_id, f"{side}_comb_guide_{guide_side}{index - 1:02d}")
                    ]
                    self.model.geom_pos[guide, 0] = guide_x
                    self.model.geom_pos[guide, 1] = y + y_sign * slot_offset

        press_x = max(0.030, 0.5 * geometry.fin_size_m[0] - 0.065)
        press_z = base_top + geometry.fin_size_m[2] + 0.003
        for component, x in (("front_press", -press_x), ("rear_press", press_x)):
            geom_id = self._component_geom_ids[(tray_id, component)]
            self.model.geom_pos[geom_id, 0] = x
            self.model.geom_pos[geom_id, 2] = press_z

        self.mujoco.mj_forward(self.model, self.data)

    def _set_braze_fraction(
        self,
        tray_id: str,
        component: str,
        fraction: float,
        *,
        reverse: bool,
    ) -> None:
        geom_id = self._component_geom_ids.get((tray_id, component))
        if geom_id is None:
            return
        progress = float(np.clip(fraction, 0.0, 1.0))
        original_pos = self._original_geom_pos[geom_id]
        self.model.geom_pos[geom_id] = original_pos
        self.model.geom_quat[geom_id] = self._original_geom_quat[geom_id]
        self.model.geom_size[geom_id] = self._original_geom_size[geom_id]
        if progress <= 1.0e-6:
            self._set_geom_visible(geom_id, False)
            return
        if progress < 1.0 - 1.0e-6:
            half_length = float(self._original_geom_size[geom_id][1])
            direction = -1.0 if reverse else 1.0
            position = original_pos.copy()
            position[0] = direction * half_length * (progress - 1.0)
            self.model.geom_pos[geom_id] = position
            self.model.geom_size[geom_id, 1] = max(
                1.0e-6,
                half_length * progress,
            )
        self._set_geom_visible(geom_id, True)

    def _sync_payload_visibility(
        self,
        runtime: "DualLineRuntime",
        tray_id: str,
        *,
        visible: bool,
    ) -> None:
        self._set_tray_visible(tray_id, False)
        if not visible:
            return
        self._set_geom_visible(self._primary_geom_ids[tray_id], True)
        self._set_component_visible(tray_id, "template_plate", True)
        unit = self._unit_for_tray(runtime, tray_id)
        if unit is None:
            return
        stage = unit.stage.value
        base_visible = stage not in {"QUEUED", "BASE_LOADING"} or self._robots.pending_installed_base(
            unit.unit_id
        )
        self._set_component_visible(tray_id, "base_plate", base_visible)
        fin_count = int(unit.fin_count)
        # A detected braze defect belongs to the pallet for the whole physical
        # recovery loop: S2B return, S2A queueing, local touch-up and S2B
        # reinspection.  Tying this context only to the short Arm2 operation
        # made the return interval look like a brand-new 0%-complete dispense
        # pass, so every sound bead disappeared until Arm2 started moving.
        local_rework = any(
            defect.unit_id == unit.unit_id
            and defect.status == "DETECTED"
            and defect.visual_type in {"BRAZING_MISSING", "BRAZING_PATH_DEVIATION"}
            for defect in runtime.faults.physical_faults.values()
        )
        if stage == "DISPENSING" and not local_rework:
            progress = self._robots.operation_progress(
                "ARM2",
                unit.unit_id,
                "DISPENSING",
            )
            pass_progress = progress * fin_count
            complete_passes = int(np.floor(pass_progress + 1.0e-9))
            active_fraction = pass_progress - complete_passes
            for pass_index in range(fin_count):
                if pass_index < complete_passes:
                    fraction = 1.0
                elif pass_index == complete_passes:
                    fraction = active_fraction
                else:
                    fraction = 0.0
                for pair_offset in range(2):
                    path_index = 2 * pass_index + pair_offset + 1
                    self._set_braze_fraction(
                        tray_id,
                        f"braze_{path_index:02d}",
                        fraction,
                        reverse=pass_index % 2 == 1,
                    )
        else:
            material_visible = local_rework or stage not in {
                "QUEUED",
                "BASE_LOADING",
                "WAITING_S2A",
            }
            for index in range(1, 2 * fin_count + 1):
                self._set_braze_fraction(
                    tray_id,
                    f"braze_{index:02d}",
                    1.0 if material_visible else 0.0,
                    reverse=((index - 1) // 2) % 2 == 1,
                )
        comb_visible = stage in {
            "FIN_INSTALLATION",
            "WAITING_BRAZING_REVIEW",
            "BRAZING_REVIEW",
            "WAITING_FINS_REVIEW",
            "FINS_REVIEW",
            "WAITING_MERGE",
            "MERGING",
            "WAITING_S4",
            "PRE_BRAZE_INSPECTION",
            "WAITING_BUFFER",
            "FURNACE_BUFFER",
            "FURNACE_LOADING",
            "BRAZING",
            "FURNACE_UNLOADING",
            "POST_BRAZE_INSPECTION",
            "WAITING_OUTPUT",
            "DELIVERING",
        }
        for component in (
            "front_comb_base",
            "rear_comb_base",
            "front_comb_post_left",
            "front_comb_post_right",
            "rear_comb_post_left",
            "rear_comb_post_right",
        ):
            self._set_component_visible(tray_id, component, comb_visible)
        for side in ("front", "rear"):
            for guide_side in ("left", "right"):
                for index in range(int(unit.fin_count)):
                    self._set_component_visible(
                        tray_id,
                        f"{side}_comb_guide_{guide_side}{index:02d}",
                        comb_visible,
                    )
        visible_fin_count = max(
            int(unit.fins_installed),
            self._robots.pending_installed_fin_index(unit.unit_id),
        )
        missing_pick_indices = {
            int(defect.operation_index)
            for defect in runtime.faults.physical_faults.values()
            if defect.unit_id == unit.unit_id
            and defect.operation_index is not None
            and defect.fault_type.value == "FIN_PICK_FAILED"
            and defect.status in {"MANIFESTED", "DETECTED"}
        }
        pending_repaired_index = self._robots.pending_installed_fin_index(unit.unit_id)
        for index in range(1, visible_fin_count + 1):
            missing_from_failed_pick = index in missing_pick_indices and pending_repaired_index != index
            self._set_component_visible(
                tray_id,
                f"fin_{index:02d}",
                not missing_from_failed_pick
                and not self._robots.rework_fin_owned_by_proxy(unit.unit_id, index),
            )
        press_visible = stage in {
            "WAITING_BUFFER",
            "FURNACE_BUFFER",
            "FURNACE_LOADING",
            "BRAZING",
            "FURNACE_UNLOADING",
            "POST_BRAZE_INSPECTION",
            "WAITING_OUTPUT",
            "DELIVERING",
        }
        self._set_component_visible(tray_id, "front_press", press_visible)
        self._set_component_visible(tray_id, "rear_press", press_visible)

    def _furnace_layer_position(
        self,
        runtime: "DualLineRuntime",
        tray_id: str,
    ) -> np.ndarray:
        for layer in runtime.furnace.state.layers:
            if layer.tray_id == tray_id:
                return np.asarray(
                    runtime.topology.station(f"FURNACE_LAYER_{layer.index}").world_xyz,
                    dtype=float,
                )
        return np.asarray(runtime.topology.station("FURNACE_FRONT").world_xyz, dtype=float)

    def _target_for(
        self,
        runtime: "DualLineRuntime",
        tray_id: str,
        owner: TrayOwner,
    ) -> np.ndarray:
        if owner is TrayOwner.EMPTY_BUFFER or owner is TrayOwner.VIRTUAL_RETURN:
            return self._home_positions[tray_id].copy()
        if owner is TrayOwner.FURNACE:
            return self._furnace_layer_position(runtime, tray_id)
        return np.asarray(
            runtime.topology.station_for_owner(owner.value).world_xyz,
            dtype=float,
        )

    def _begin_motion(
        self,
        tray_id: str,
        target: np.ndarray,
        now: float,
        source_owner: TrayOwner,
        target_owner: TrayOwner,
    ) -> None:
        mocap_id = self._mocap_ids[tray_id]
        start = np.asarray(self.data.mocap_pos[mocap_id], dtype=float).copy()
        route_targets = self._route_targets(
            start,
            target,
            source_owner=source_owner,
            target_owner=target_owner,
        )
        first_target = route_targets[0]
        distance = float(np.linalg.norm(first_target - start))
        if distance <= 1.0e-9:
            self.data.mocap_pos[mocap_id] = first_target
            remaining = deque(route_targets[1:])
            if remaining:
                next_target = remaining.popleft()
                self._motions[tray_id] = _CarrierMotion(
                    first_target.copy(),
                    next_target,
                    now,
                    self._motion_duration(first_target, next_target),
                    source_owner,
                    target_owner,
                    remaining,
                    2,
                    len(route_targets),
                )
            else:
                self.data.mocap_quat[mocap_id] = (1.0, 0.0, 0.0, 0.0)
                self._motions.pop(tray_id, None)
                self._physical_owner[tray_id] = target_owner
            return
        self._motions[tray_id] = _CarrierMotion(
            start,
            first_target.copy(),
            now,
            self._motion_duration(start, first_target),
            source_owner,
            target_owner,
            deque(route_targets[1:]),
            1,
            len(route_targets),
        )

    def _motion_duration(self, start: np.ndarray, target: np.ndarray) -> float:
        distance = float(np.linalg.norm(target - start))
        scale = max(self._motion_speed_scale, 1.0)
        return max(
            self.minimum_transfer_s / scale,
            distance / (self.transfer_speed_m_s * scale),
        )

    @staticmethod
    def _route_targets(
        start: np.ndarray,
        final_target: np.ndarray,
        *,
        source_owner: TrayOwner,
        target_owner: TrayOwner,
    ) -> tuple[np.ndarray, ...]:
        """Return collision-free planar/furnace waypoints.

        Both S3 branches remain on the station transport plane.  The A branch
        passes north of S2B.  The denser B branch first clears the S2B west
        edge, then follows the south perimeter before turning onto the S4
        centreline.  Its final leg enters the relocated S4 vertically, matching
        the A branch's centred approach instead of using an oblique segment.
        Moving the inspection table toward the furnace removes the former
        north-side return loop and its visible reverse motion.  These waypoints
        use the complete 400-by-280 mm pallet envelope; they are not
        centreline-only shortcuts.
        """

        branch_stations = {TrayOwner.INSTALL_A, TrayOwner.INSTALL_B}
        branch_waits = {TrayOwner.MERGE_A_WAIT, TrayOwner.MERGE_B_WAIT}
        if source_owner is TrayOwner.S4 and target_owner in branch_stations:
            # Camera-confirmed fin rework returns through the exact outbound
            # corridor in reverse, staying on the 0.225 m transport plane.
            if target_owner is TrayOwner.INSTALL_B:
                return (
                    np.asarray([1.40, _BRANCH_B_SOUTH_CLEAR_Y_M, float(start[2])]),
                    np.asarray([_BRANCH_B_WEST_CLEAR_X_M, _BRANCH_B_SOUTH_CLEAR_Y_M, float(start[2])]),
                    np.asarray([_BRANCH_B_WEST_CLEAR_X_M, _BRANCH_B_CORRIDOR_ENTRY_Y_M, float(start[2])]),
                    np.asarray([_BRANCH_B_CORNER_CLEAR_X_M, float(final_target[1]), float(start[2])]),
                    final_target.copy(),
                )
            return (
                np.asarray([1.40, 0.50, float(start[2])]),
                final_target.copy(),
            )
        if source_owner is TrayOwner.INSTALL_B and target_owner is TrayOwner.S2B:
            # Recovery arbitration evacuates a completed S3B pallet along the
            # exact branch rail it used to enter.  The straight reverse route
            # keeps the pallet horizontal and out of the S4 return corridor.
            return (final_target.copy(),)
        if source_owner in branch_stations and target_owner in branch_waits:
            if source_owner is TrayOwner.INSTALL_B:
                return (
                    np.asarray(
                        [
                            _BRANCH_B_CORNER_CLEAR_X_M,
                            float(start[1]),
                            float(start[2]),
                        ],
                        dtype=float,
                    ),
                    np.asarray(
                        [
                            _BRANCH_B_WEST_CLEAR_X_M,
                            _BRANCH_B_CORRIDOR_ENTRY_Y_M,
                            float(start[2]),
                        ],
                        dtype=float,
                    ),
                    np.asarray(
                        [
                            _BRANCH_B_WEST_CLEAR_X_M,
                            _BRANCH_B_SOUTH_CLEAR_Y_M,
                            float(start[2]),
                        ],
                        dtype=float,
                    ),
                    final_target.copy(),
                )
            return (final_target.copy(),)
        if source_owner in branch_waits and target_owner is TrayOwner.MERGE:
            # Both waits are already aligned to the S4 centreline.  Returning
            # one target keeps the physical carrier, UI route and MJCF rails on
            # the same straight final segment without a duplicate zero-length
            # waypoint.
            return (final_target.copy(),)
        if target_owner is TrayOwner.FURNACE:
            front_lift_low = np.asarray([2.75, 0.0, 0.225])
            front_lift_aligned = np.asarray(
                [2.75, 0.0, float(final_target[2])],
            )
            return (
                front_lift_low,
                front_lift_aligned,
                final_target.copy(),
            )
        if source_owner is TrayOwner.FURNACE and target_owner is TrayOwner.POST_SCAN:
            rear_lift_aligned = np.asarray([4.05, 0.0, float(start[2])])
            rear_lift_low = np.asarray([4.05, 0.0, 0.225])
            return (
                rear_lift_aligned,
                rear_lift_low,
                final_target.copy(),
            )
        return (final_target.copy(),)

    @staticmethod
    def _motion_elapsed(motion: _CarrierMotion, now: float) -> float:
        reference = float(motion.paused_at) if motion.paused_at is not None else float(now)
        return max(0.0, reference - motion.started_at)

    def _advance_motion(self, tray_id: str, now: float, *, paused: bool = False) -> None:
        motion = self._motions.get(tray_id)
        if motion is None:
            return
        if paused:
            if motion.paused_at is None:
                motion.paused_at = float(now)
            return
        if motion.paused_at is not None:
            # Exclude the physical fault hold from the trajectory's elapsed
            # time.  Otherwise the carrier would teleport to the point it would
            # have reached while its lift/pusher was visibly stopped.
            motion.started_at += max(0.0, float(now) - motion.paused_at)
            motion.paused_at = None
        elapsed = self._motion_elapsed(motion, now)
        fraction = min(1.0, elapsed / motion.duration_s)
        blend = quintic_time_scaling(fraction)
        self.data.mocap_pos[self._mocap_ids[tray_id]] = motion.start + blend * (motion.target - motion.start)
        # The pallet orientation is deliberately constant through front-load
        # and rear-unload, matching the process-direction invariant.
        self.data.mocap_quat[self._mocap_ids[tray_id]] = (1.0, 0.0, 0.0, 0.0)
        if fraction >= 1.0:
            self.data.mocap_pos[self._mocap_ids[tray_id]] = motion.target
            self.data.mocap_quat[self._mocap_ids[tray_id]] = (
                1.0,
                0.0,
                0.0,
                0.0,
            )
            if motion.remaining_targets:
                next_target = motion.remaining_targets.popleft()
                motion.start = motion.target.copy()
                motion.target = next_target
                motion.started_at = now
                motion.duration_s = self._motion_duration(
                    motion.start,
                    motion.target,
                )
                motion.segment_index += 1
            else:
                self._motions.pop(tray_id, None)
                self._physical_owner[tray_id] = motion.target_owner

    def _start_next_motion(
        self,
        runtime: "DualLineRuntime",
        tray_id: str,
        now: float,
    ) -> None:
        while tray_id not in self._motions and self._pending_owners[tray_id]:
            target_owner = self._pending_owners[tray_id].popleft()
            source_owner = self._physical_owner[tray_id]
            if target_owner is TrayOwner.VIRTUAL_RETURN:
                self.data.mocap_pos[self._mocap_ids[tray_id]] = self._home_positions[tray_id]
                self.data.mocap_quat[self._mocap_ids[tray_id]] = (
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                )
                self._physical_owner[tray_id] = target_owner
                continue
            if target_owner is TrayOwner.EMPTY_BUFFER and source_owner is TrayOwner.VIRTUAL_RETURN:
                self._physical_owner[tray_id] = target_owner
                continue
            self._begin_motion(
                tray_id,
                self._target_for(runtime, tray_id, target_owner),
                now,
                source_owner,
                target_owner,
            )

    def _reconcile_transport_target(
        self,
        tray_id: str,
        target_owner: TrayOwner,
    ) -> None:
        """Discard stale carrier intent when a unit is sent back for rework.

        Runtime ownership is authoritative, but a scene sync can observe a
        rollback one frame after an outbound handoff was queued.  Keeping that
        old destination in ``_pending_owners`` would make the physical carrier
        finish the forward route and only then start the return route.  When a
        route is corrected, the current mocap pose is retained and the next
        sync starts a new path from that pose, so this is a replan rather than
        a teleport.
        """

        pending = self._pending_owners[tray_id]
        motion = self._motions.get(tray_id)
        if motion is not None and motion.target_owner is target_owner:
            # The physical route already points at the new logical owner.  Any
            # duplicate queued copy would create a zero-length second motion.
            pending.clear()
            return
        if motion is not None and motion.target_owner is not target_owner:
            # Preserve the current position and cancel only the obsolete route.
            # The next owner is appended below and _begin_motion will use the
            # measured mocap position as its start.
            self._motions.pop(tray_id, None)
            pending.clear()
        elif motion is None and pending and pending[-1] is not target_owner:
            # No physical movement has started yet; all queued intents are
            # obsolete once the logical owner has been rewound.
            pending.clear()
        if not pending and (motion is None or motion.target_owner is not target_owner):
            pending.append(target_owner)
        elif pending and pending[-1] is not target_owner:
            pending.clear()
            pending.append(target_owner)

    def _sync_furnace_mechanisms(self, runtime: "DualLineRuntime") -> None:
        door_hold = next(
            (
                operation
                for operation in runtime.operations.values()
                if operation.resource == "FURNACE_DOOR"
                and (operation.fault_hold_remaining_s > 0.0 or operation.manual_hold_fault_ids)
            ),
            None,
        )
        front_target = 0.56 if runtime.furnace.state.front_door_open else 0.0
        rear_target = 0.56 if runtime.furnace.state.rear_door_open else 0.0
        if door_hold is not None and door_hold.kind.startswith("FURNACE_FRONT_"):
            front_target = float(self.data.qpos[self._mechanism_qpos["v2_furnace_front_door_joint"]])
        if door_hold is not None and door_hold.kind.startswith("FURNACE_REAR_"):
            rear_target = float(self.data.qpos[self._mechanism_qpos["v2_furnace_rear_door_joint"]])
        self.data.ctrl[self._actuators["v2_furnace_front_door_actuator"]] = front_target
        self.data.ctrl[self._actuators["v2_furnace_rear_door_actuator"]] = rear_target
        self.data.ctrl[self._actuators["v2_finished_output_gate_actuator"]] = (
            0.50 if runtime.output_gate_open else 0.0
        )
        for layer in runtime.furnace.state.layers:
            actuator = self._actuators[f"v2_furnace_layer_{layer.index}_lock_actuator"]
            self.data.ctrl[actuator] = 0.03 if layer.locked else 0.0
        loading = next(
            (
                unit
                for unit in runtime.units.values()
                if unit.stage.value == "FURNACE_LOADING" and unit.furnace_layer is not None
            ),
            None,
        )
        unloading = next(
            (
                unit
                for unit in runtime.units.values()
                if unit.stage.value == "FURNACE_UNLOADING" and unit.furnace_layer is not None
            ),
            None,
        )
        front_lift = 0.0
        front_pusher = 0.0
        if loading is not None and loading.tray_id is not None:
            target_height = 0.14 * int(loading.furnace_layer)
            motion = self._motions.get(loading.tray_id)
            if motion is not None:
                progress = min(
                    1.0,
                    max(
                        0.0,
                        self._motion_elapsed(motion, float(self.data.time)) / max(motion.duration_s, 1.0e-9),
                    ),
                )
                if motion.segment_index == 2:
                    front_lift = target_height * quintic_time_scaling(progress)
                elif motion.segment_index >= 3:
                    front_lift = target_height
                    front_pusher = 0.70 * quintic_time_scaling(progress)

        rear_lift = 0.0
        rear_extractor = 0.0
        if unloading is not None and unloading.tray_id is not None:
            target_height = 0.14 * int(unloading.furnace_layer)
            motion = self._motions.get(unloading.tray_id)
            if motion is not None:
                progress = min(
                    1.0,
                    max(
                        0.0,
                        self._motion_elapsed(motion, float(self.data.time)) / max(motion.duration_s, 1.0e-9),
                    ),
                )
                if motion.segment_index == 1:
                    rear_lift = target_height
                    rear_extractor = 0.60 * (1.0 - quintic_time_scaling(progress))
                elif motion.segment_index == 2:
                    rear_lift = target_height * (1.0 - quintic_time_scaling(progress))

        self.data.ctrl[self._actuators["v2_furnace_lift_actuator"]] = front_lift
        self.data.ctrl[self._actuators["v2_furnace_pusher_actuator"]] = front_pusher
        self.data.ctrl[self._actuators["v2_furnace_rear_lift_actuator"]] = rear_lift
        self.data.ctrl[self._actuators["v2_furnace_rear_extractor_actuator"]] = rear_extractor
        runtime.furnace.state.lift_clear = bool(
            self.data.qpos[self._mechanism_qpos["v2_furnace_lift_joint"]] <= 0.01
        )
        runtime.furnace.state.pusher_retracted = bool(
            self.data.qpos[self._mechanism_qpos["v2_furnace_pusher_joint"]] <= 0.02
        )

    def _sync_route_mechanisms(self) -> None:
        for actuator_id in self._route_actuators.values():
            self.data.ctrl[actuator_id] = 0.0
        routes = {
            (TrayOwner.S1, TrayOwner.S2A): ("v2_s1_s2a_actuator", 0.492443, False),
            (TrayOwner.S2A, TrayOwner.S2B): ("v2_s2a_s2b_actuator", 0.855862, False),
            (TrayOwner.S2B, TrayOwner.INSTALL_A): ("v2_branch_a_actuator", 0.502494, False),
            (TrayOwner.S2B, TrayOwner.INSTALL_B): ("v2_branch_b_actuator", 0.474342, False),
            (TrayOwner.INSTALL_B, TrayOwner.S2B): ("v2_branch_b_actuator", 0.474342, True),
            (TrayOwner.S4, TrayOwner.BUFFER_1): ("v2_buffer_index_actuator", 0.450, False),
            (TrayOwner.S4, TrayOwner.BUFFER_2): ("v2_buffer_index_actuator", 0.900, False),
            (TrayOwner.S4, TrayOwner.BUFFER_3): ("v2_buffer_index_actuator", 1.350, False),
            (TrayOwner.FURNACE, TrayOwner.POST_SCAN): (
                "v2_output_transfer_actuator",
                0.420,
                False,
            ),
            (TrayOwner.POST_SCAN, TrayOwner.OUTPUT): (
                "v2_output_transfer_actuator",
                1.140,
                False,
            ),
        }
        now = float(self.data.time)
        for motion in self._motions.values():
            route = routes.get((motion.source_owner, motion.target_owner))
            if route is None:
                continue
            actuator_name, maximum, reverse = route
            elapsed = self._motion_elapsed(motion, now)
            fraction = min(1.0, elapsed / motion.duration_s)
            progress = quintic_time_scaling(fraction)
            self.data.ctrl[self._route_actuators[actuator_name]] = maximum * (
                1.0 - progress if reverse else progress
            )

    def sync(self, runtime: "DualLineRuntime") -> None:
        logical_now = float(runtime.sim_time)
        self._motion_speed_scale = 3.0 if runtime.fast else 1.0
        if runtime is not self._bound_runtime or logical_now + 1.0e-9 < self._last_runtime_time:
            self._inspection_records.clear()
            self.fault_visuals.reset()
        self._bound_runtime = runtime
        existing_gate = getattr(runtime, "_execution_gate", None)
        bind_physical_gate = getattr(existing_gate, "bind_physical_gate", None)
        if callable(bind_physical_gate):
            bind_physical_gate(self)
        else:
            runtime.set_execution_gate(self)
        self._last_runtime_time = logical_now
        if not isfinite(logical_now):
            raise ValueError("runtime simulation time must be finite")
        now = float(self.data.time)
        if not runtime.units:
            self._motions.clear()
            self._robots.reset()
            self._configured_unit_by_tray.clear()
            for tray_id in self._tray_ids:
                self._pending_owners[tray_id].clear()
                self._physical_owner[tray_id] = TrayOwner.EMPTY_BUFFER
                self._last_owner[tray_id] = TrayOwner.EMPTY_BUFFER
                self.data.mocap_pos[self._mocap_ids[tray_id]] = self._home_positions[tray_id]
                self.data.mocap_quat[self._mocap_ids[tray_id]] = (
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                )
        for tray in runtime.flow.trays:
            tray_id = tray.tray_id
            owner = tray.owner
            unit = self._unit_for_tray(runtime, tray_id)
            if unit is not None and self._configured_unit_by_tray.get(tray_id) != unit.unit_id:
                self._configure_tray_geometry(tray_id, unit)
                self._configured_unit_by_tray[tray_id] = unit.unit_id
            nonphysical = {TrayOwner.EMPTY_BUFFER, TrayOwner.VIRTUAL_RETURN}
            visible = (
                owner not in nonphysical
                or self._physical_owner[tray_id] not in nonphysical
                or tray_id in self._motions
                or any(target not in nonphysical for target in self._pending_owners[tray_id])
            )
            self._sync_payload_visibility(runtime, tray_id, visible=visible)
            if owner is not self._last_owner[tray_id]:
                self._reconcile_transport_target(tray_id, owner)
            self._last_owner[tray_id] = owner
            unit_operation = next(
                (
                    operation
                    for operation in runtime.operations.values()
                    if unit is not None
                    and operation.unit_id == unit.unit_id
                    and operation.kind in {"FURNACE_LOAD_TRAY", "FURNACE_UNLOAD_TRAY"}
                    and (operation.fault_hold_remaining_s > 0.0 or operation.manual_hold_fault_ids)
                ),
                None,
            )
            self._advance_motion(
                tray_id,
                now,
                paused=unit_operation is not None,
            )
            self._start_next_motion(runtime, tray_id, now)
        self._sync_furnace_mechanisms(runtime)
        self._sync_route_mechanisms()
        self._robots.sync(runtime)
        # Apply fault geometry last: normal product/robot projection establishes
        # the correct current pose first, then the latent defect modifies only
        # its exact bead/fin.  This prevents normal visibility sync from erasing
        # the defect one line later.
        physical_faults = []
        for defect in runtime.faults.physical_faults.values():
            state = defect.as_dict()
            operation = next(
                (
                    item
                    for item in runtime.operations.values()
                    if item.recovery
                    and item.unit_id == defect.unit_id
                    and item.recovery_target_index == defect.operation_index
                ),
                None,
            )
            if operation is not None:
                repaired_fin_committed = bool(
                    operation.kind == "INSTALL_FIN"
                    and defect.operation_index is not None
                    and self._robots.pending_installed_fin_index(defect.unit_id)
                    == int(defect.operation_index)
                )
                repair_progress = (
                    1.0
                    if repaired_fin_committed
                    else self._robots.operation_progress(
                        operation.resource,
                        operation.unit_id,
                        operation.kind,
                    )
                )
                state["repair_progress"] = repair_progress
                if defect.visual_type == "BRAZING_PATH_DEVIATION":
                    state["removal_progress"] = float(
                        np.clip(
                            repair_progress / BRAZING_DEVIATION_REMOVAL_END,
                            0.0,
                            1.0,
                        )
                    )
                    state["reapply_progress"] = float(
                        np.clip(
                            (repair_progress - BRAZING_DEVIATION_REAPPLY_START)
                            / (1.0 - BRAZING_DEVIATION_REAPPLY_START),
                            0.0,
                            1.0,
                        )
                    )
            physical_faults.append(state)
        self.fault_visuals.sync(
            [record.as_dict() for record in runtime.faults.faults.values()],
            [unit.as_dict() for unit in runtime.units.values()],
            physical_faults,
        )
        self.mujoco.mj_forward(self.model, self.data)

    def step_physics(self, duration_s: float) -> None:
        duration = float(duration_s)
        if duration <= 0 or not isfinite(duration):
            raise ValueError("physics step duration must be finite and positive")
        steps = max(1, int(round(duration / float(self.model.opt.timestep))))
        for _ in range(steps):
            self._robots.control_tick(float(self.model.opt.timestep))
            self.mujoco.mj_step(self.model, self.data)
            self._robots.enforce_paused_state()

    def tray_position(self, tray_id: str) -> np.ndarray:
        """Return the authoritative carrier position for route planning/UI."""

        return np.asarray(self.data.mocap_pos[self._mocap_ids[tray_id]], dtype=float).copy()

    def tray_quaternion(self, tray_id: str) -> np.ndarray:
        """Return the carrier orientation; welded-body solver error is not route motion."""

        return np.asarray(self.data.mocap_quat[self._mocap_ids[tray_id]], dtype=float).copy()

    def tray_visible(self, tray_id: str) -> bool:
        return bool(self.model.geom_rgba[self._primary_geom_ids[tray_id], 3] > 0.0)

    def component_visible(self, tray_id: str, component: str) -> bool:
        try:
            geom_id = self._component_geom_ids[(str(tray_id), str(component))]
        except KeyError as exc:
            raise KeyError(f"unknown V2 tray component: {tray_id}/{component}") from exc
        return bool(self.model.geom_rgba[geom_id, 3] > 0.0)

    def transport_snapshot(self) -> dict[str, dict[str, object]]:
        """Expose active physical carrier motion to the shared logistics UI."""

        result: dict[str, dict[str, object]] = {}
        for tray_id, motion in sorted(self._motions.items()):
            elapsed = self._motion_elapsed(motion, float(self.data.time))
            # Runtime simulation time and MuJoCo time advance together in the
            # application. During isolated adapter tests the mocap position is
            # the authoritative source, so derive progress geometrically.
            distance = float(np.linalg.norm(motion.target - motion.start))
            current = np.asarray(
                self.data.mocap_pos[self._mocap_ids[tray_id]],
                dtype=float,
            )
            travelled = float(np.linalg.norm(current - motion.start))
            progress = 1.0 if distance <= 1.0e-12 else min(1.0, travelled / distance)
            result[tray_id] = {
                "route_id": f"{motion.source_owner.value}_TO_{motion.target_owner.value}",
                "source": motion.source_owner.value,
                "target": motion.target_owner.value,
                "tray_id": tray_id,
                "status": "MOVING",
                "progress": progress,
                "elapsed_s": min(elapsed, motion.duration_s),
                "duration_s": motion.duration_s,
                "distance_m": distance,
                "world_position_m": current.tolist(),
                "moving": True,
                "segment_index": motion.segment_index,
                "segment_count": motion.segment_count,
            }
        return result

    def furnace_transfer_snapshot(self) -> dict[str, float]:
        """Return commanded positions used by the logistics UI and tests."""

        return {
            name.removeprefix("v2_furnace_").removesuffix("_actuator"): float(self.data.ctrl[actuator_id])
            for name, actuator_id in self._actuators.items()
            if name
            in {
                "v2_furnace_lift_actuator",
                "v2_furnace_pusher_actuator",
                "v2_furnace_rear_lift_actuator",
                "v2_furnace_rear_extractor_actuator",
            }
        }

    def furnace_mechanism_position_snapshot(self) -> dict[str, float]:
        """Return measured door, lift, pusher and extractor joint positions."""

        return {
            name.removeprefix("v2_furnace_").removesuffix("_joint"): float(self.data.qpos[qpos_id])
            for name, qpos_id in self._mechanism_qpos.items()
        }

    def route_mechanism_snapshot(self) -> dict[str, float]:
        return {
            name.removeprefix("v2_").removesuffix("_actuator"): float(self.data.ctrl[actuator_id])
            for name, actuator_id in self._route_actuators.items()
        }

    def robot_motion_snapshot(self) -> dict[str, dict[str, object]]:
        return self._robots.snapshot()

    def inspection_snapshot(self) -> list[dict[str, object]]:
        """Return capture/analysis truth for UI, API and regression tests."""

        now = 0.0 if self._bound_runtime is None else float(self._bound_runtime.sim_time)
        return [
            {
                "unit_id": record.unit_id,
                "kind": record.kind,
                "camera": record.camera,
                "captured": True,
                "captured_at": record.captured_at,
                "aligned": record.aligned,
                "clear": record.clear,
                "analysis_elapsed_s": min(
                    record.analysis_seconds,
                    max(0.0, now - record.captured_at),
                ),
                "analysis_seconds": record.analysis_seconds,
                "analysis_complete": record.analysis_complete,
            }
            for record in sorted(
                self._inspection_records.values(),
                key=lambda item: (item.captured_at, item.unit_id, item.kind),
            )
        ]

    def _inspection_complete(
        self,
        unit_id: str,
        kind: str,
        *,
        camera: str,
        aligned: bool,
    ) -> bool:
        if not aligned or self._bound_runtime is None:
            return False
        key = (str(unit_id), str(kind))
        now = float(self._bound_runtime.sim_time)
        record = self._inspection_records.get(key)
        if record is None:
            # This is the certified capture instant: the robot/tray has
            # reached its final pose and the full rectangle is framed by the
            # authored top-down camera geometry.
            record = _InspectionRecord(
                unit_id=str(unit_id),
                kind=str(kind),
                camera=str(camera),
                captured_at=now,
            )
            self._inspection_records[key] = record
            return False
        record.analysis_complete = bool(now - record.captured_at >= record.analysis_seconds - 1.0e-9)
        return record.analysis_complete

    def physical_owner_snapshot(self) -> dict[str, str]:
        """Return measured carrier ownership, distinct from routed intent."""

        return {tray_id: owner.value for tray_id, owner in sorted(self._physical_owner.items())}

    def visible_tray_clearance_m(self) -> float | None:
        """Return signed AABB clearance for the closest visible tray pair."""

        visible = [tray_id for tray_id in self._tray_ids if self.tray_visible(tray_id)]
        if len(visible) < 2:
            return None
        half_extent_sum = np.asarray([0.40, 0.28, 0.12], dtype=float)
        closest = float("inf")
        for left_index, left_id in enumerate(visible):
            left = self.tray_position(left_id)
            for right_id in visible[left_index + 1 :]:
                right = self.tray_position(right_id)
                gaps = np.abs(left - right) - half_extent_sum
                if np.any(gaps > 0.0):
                    clearance = float(np.linalg.norm(np.maximum(gaps, 0.0)))
                else:
                    clearance = float(np.max(gaps))
                closest = min(closest, clearance)
        return closest

    def tray_ready(self, tray_id: str, owner: TrayOwner) -> bool:
        """Physical feedback consumed by :class:`DualLineRuntime`."""

        tray_id = str(tray_id)
        target_owner = TrayOwner(owner)
        return bool(
            self._physical_owner.get(tray_id) is target_owner
            and tray_id not in self._motions
            and not self._pending_owners.get(tray_id)
        )

    def estimated_tray_ready_in(self, tray_id: str, owner: TrayOwner) -> float:
        """Return remaining authored carrier time to an already requested owner."""

        tray_id = str(tray_id)
        target_owner = TrayOwner(owner)
        if self.tray_ready(tray_id, target_owner):
            return 0.0
        motion = self._motions.get(tray_id)
        if motion is None or motion.target_owner is not target_owner:
            return 0.0
        if motion.paused_at is not None:
            return float("inf")
        remaining = max(
            0.0,
            motion.duration_s - self._motion_elapsed(motion, float(self.data.time)),
        )
        start = motion.target
        for target in motion.remaining_targets:
            remaining += self._motion_duration(start, target)
            start = target
        return remaining

    def owner_available(self, owner: TrayOwner) -> bool:
        """Reserve a station until its previous physical pallet has departed."""

        target = TrayOwner(owner)
        if target in {
            TrayOwner.EMPTY_BUFFER,
            TrayOwner.VIRTUAL_RETURN,
            TrayOwner.FURNACE,
        }:
            return True
        if any(value is target for value in self._physical_owner.values()):
            return False
        # The three furnace buffers are positions on one straight indexing
        # conveyor, not independent teleport destinations.  A pallet heading
        # to a farther slot must not pass through an occupied nearer slot.
        # This can happen when two compatible batches interleave under load;
        # keeping the logical operation READY until the lane clears preserves
        # FIFO physical ownership and prevents tray-on-tray overlap.
        buffer_route = {
            TrayOwner.BUFFER_1: (),
            TrayOwner.BUFFER_2: (TrayOwner.BUFFER_1,),
            TrayOwner.BUFFER_3: (TrayOwner.BUFFER_1, TrayOwner.BUFFER_2),
        }
        for upstream in buffer_route.get(target, ()):
            if any(value is upstream for value in self._physical_owner.values()):
                return False
            if any(motion.target_owner is upstream for motion in self._motions.values()):
                return False
            if any(upstream in pending for pending in self._pending_owners.values()):
                return False
        # Both branch waiting slots and the Y centre lie inside one 400 x
        # 280 mm pallet swept envelope.  Treat that whole geometry as a
        # single-occupancy corridor: the next completed pallet remains at its
        # installation station until the previous pallet has reached S4.
        # Reserving only the nominal MERGE point allowed the two diagonal
        # carrier paths to overlap even though their logical owners differed.
        merge_waits = {TrayOwner.MERGE_A_WAIT, TrayOwner.MERGE_B_WAIT}
        merge_corridor = merge_waits | {TrayOwner.MERGE}
        branch_corridors = {
            TrayOwner.INSTALL_A: TrayOwner.MERGE_A_WAIT,
            TrayOwner.INSTALL_B: TrayOwner.MERGE_B_WAIT,
        }

        def branch_entry_clear(
            install_owner: TrayOwner,
            wait_owner: TrayOwner,
        ) -> bool:
            branch_owners = {install_owner, wait_owner}
            if any(value in branch_owners for value in self._physical_owner.values()):
                return False
            if any(
                motion.source_owner in branch_owners or motion.target_owner in branch_owners
                for motion in self._motions.values()
            ):
                return False
            return not any(
                any(pending in branch_owners for pending in pending_owners)
                for pending_owners in self._pending_owners.values()
            )

        # S2B sits between the two outbound corridors.  Admit a new
        # inspection pallet only when at least one complete branch entry is
        # clear; otherwise two occupied installation docks can trap the S2B
        # pallet in the swept path of the next outbound carrier.
        if target is TrayOwner.S2B and not any(
            branch_entry_clear(install_owner, wait_owner)
            for install_owner, wait_owner in branch_corridors.items()
        ):
            return False
        # Each planar S3→S4 route begins inside its installation dock's pallet
        # envelope.  Once an outbound pallet has reached the branch wait point
        # (or is travelling onward to the shared S4 entry), a following pallet
        # must not enter that same installation dock.  Keep an explicit swept
        # corridor reservation even though the fixed obstacles are now clear.
        if target in branch_corridors:
            branch_wait = branch_corridors[target]
            branch_outbound_active = any(
                value is branch_wait for value in self._physical_owner.values()
            ) or any(
                (motion.source_owner is target and motion.target_owner is branch_wait)
                or (motion.source_owner is branch_wait and motion.target_owner is TrayOwner.MERGE)
                for motion in self._motions.values()
            )
            if branch_outbound_active:
                return False
        physical_corridor = [value for value in self._physical_owner.values() if value in merge_corridor]
        moving_into_corridor = any(motion.target_owner in merge_corridor for motion in self._motions.values())
        pending_corridor = any(
            any(pending in merge_corridor for pending in pending_owners)
            for pending_owners in self._pending_owners.values()
        )
        if target in merge_waits:
            if physical_corridor or moving_into_corridor or pending_corridor:
                return False
            if any(value is TrayOwner.S4 for value in self._physical_owner.values()):
                return False
            if any(motion.target_owner is TrayOwner.S4 for motion in self._motions.values()):
                return False
        if target is TrayOwner.MERGE:
            # One occupied wait slot is the requesting pallet's legitimate
            # source.  Any additional corridor occupant or incoming motion is
            # a conflict.
            if any(value is TrayOwner.MERGE for value in physical_corridor):
                return False
            if len(physical_corridor) > 1 or moving_into_corridor:
                return False
            if any(value is TrayOwner.S4 for value in self._physical_owner.values()):
                return False
            # A pallet already at MERGE_A/B_WAIT has cleared the S2B swept
            # envelope. Its short wait-to-merge leg lies entirely beside S4,
            # so an S2B pallet must not hold it there; doing so can trap that
            # S2B pallet behind the very installation dock being evacuated.
            if any(motion.target_owner is TrayOwner.S2B for motion in self._motions.values()):
                return False
        if any(motion.target_owner is target for motion in self._motions.values()):
            return False
        return not any(target in pending_owners for pending_owners in self._pending_owners.values())

    def operation_complete(self, resource: str, unit_id: str, kind: str) -> bool:
        """Confirm robot settling and transport ownership at operation end."""

        if resource in {"ARM1", "ARM2", "ARM3"}:
            physical_complete = self._robots.operation_complete(resource, unit_id, kind)
            if resource == "ARM3" and kind in {
                "MATERIAL_INSPECTION",
                "PRE_BRAZE_INSPECTION",
            }:
                return self._inspection_complete(
                    unit_id,
                    kind,
                    camera="v2_arm3_inspection_camera",
                    aligned=physical_complete,
                )
            return physical_complete
        if kind == "FURNACE_FRONT_OPEN":
            return bool(self.data.qpos[self._mechanism_qpos["v2_furnace_front_door_joint"]] >= 0.54)
        if kind == "FURNACE_FRONT_CLOSE":
            return bool(
                self.data.qpos[self._mechanism_qpos["v2_furnace_front_door_joint"]] <= 0.01
                and self.data.qpos[self._mechanism_qpos["v2_furnace_lift_joint"]] <= 0.01
                and self.data.qpos[self._mechanism_qpos["v2_furnace_pusher_joint"]] <= 0.02
            )
        if kind == "FURNACE_REAR_OPEN":
            return bool(self.data.qpos[self._mechanism_qpos["v2_furnace_rear_door_joint"]] >= 0.54)
        if kind == "FURNACE_REAR_CLOSE":
            return bool(self.data.qpos[self._mechanism_qpos["v2_furnace_rear_door_joint"]] <= 0.01)
        if kind == "OUTPUT_GATE_OPEN":
            return bool(self.data.qpos[self._mechanism_qpos["v2_finished_output_gate_joint"]] >= 0.48)
        if kind == "OUTPUT_GATE_CLOSE":
            return bool(self.data.qpos[self._mechanism_qpos["v2_finished_output_gate_joint"]] <= 0.02)
        runtime = self._bound_runtime
        if runtime is None:
            return False
        unit = next((item for item in runtime.units.values() if item.unit_id == unit_id), None)
        if unit is None or unit.tray_id is None:
            return True
        if resource == "POST_CAMERA" and kind == "POST_BRAZE_INSPECTION":
            return self._inspection_complete(
                unit_id,
                kind,
                camera="v2_post_braze_camera",
                aligned=self.tray_ready(unit.tray_id, TrayOwner.POST_SCAN),
            )
        if kind in {
            "FURNACE_LOAD_TRAY",
            "FURNACE_UNLOAD_TRAY",
            "OUTPUT_DELIVERY",
            "VIRTUAL_RETURN",
        }:
            owner = runtime.flow.get(unit.tray_id).owner
            ready = self.tray_ready(unit.tray_id, owner)
            if kind == "FURNACE_LOAD_TRAY":
                return bool(
                    ready
                    and self.data.qpos[self._mechanism_qpos["v2_furnace_pusher_joint"]] <= 0.02
                    and self.data.qpos[self._mechanism_qpos["v2_furnace_lift_joint"]] <= 0.01
                )
            if kind == "FURNACE_UNLOAD_TRAY":
                return bool(
                    ready
                    and self.data.qpos[self._mechanism_qpos["v2_furnace_rear_extractor_joint"]] <= 0.02
                    and self.data.qpos[self._mechanism_qpos["v2_furnace_rear_lift_joint"]] <= 0.01
                )
            return ready
        return True

    def operation_start_allowed(
        self,
        resource: str,
        unit_id: str,
        kind: str,
    ) -> bool:
        """Apply physical interlocks before a logical operation may start."""

        del resource, unit_id
        if kind != "FURNACE_FRONT_CLOSE":
            return True
        lift_clear = bool(self.data.qpos[self._mechanism_qpos["v2_furnace_lift_joint"]] <= 0.01)
        pusher_retracted = bool(self.data.qpos[self._mechanism_qpos["v2_furnace_pusher_joint"]] <= 0.02)
        runtime = self._bound_runtime
        if runtime is not None:
            runtime.furnace.state.lift_clear = lift_clear
            runtime.furnace.state.pusher_retracted = pusher_retracted
        return bool(
            lift_clear
            and pusher_retracted
            and not any(motion.target_owner is TrayOwner.FURNACE for motion in self._motions.values())
        )

    def operation_milestone(
        self,
        resource: str,
        unit_id: str,
        kind: str,
        milestone: str,
    ) -> bool:
        """Expose robot interaction feedback through the execution-gate seam."""

        return self._robots.operation_milestone(resource, unit_id, kind, milestone)

    @property
    def transport_settled(self) -> bool:
        return not self._motions and not any(self._pending_owners.values())

    def physical_terminal_gate_snapshot(self) -> dict[str, object]:
        """Measured scene-side state used by the final completion authority."""

        mechanism_positions = self.furnace_mechanism_position_snapshot()
        mechanism_safe = all(
            abs(float(mechanism_positions.get(name, 0.0))) <= tolerance
            for name, tolerance in {
                "front_door": 0.02,
                "rear_door": 0.02,
                "lift": 0.01,
                "pusher": 0.02,
                "rear_lift": 0.01,
                "rear_extractor": 0.02,
                "v2_finished_output_gate": 0.02,
            }.items()
        )
        robots = self.robot_motion_snapshot()
        robots_settled = all(
            not str(item.get("operation") or "") and not str(item.get("failure") or "")
            for item in robots.values()
        )
        physical_owners = self.physical_owner_snapshot()
        trays_safe = all(owner == TrayOwner.EMPTY_BUFFER.value for owner in physical_owners.values())
        return {
            "transport_settled": self.transport_settled,
            "pending_owner_count": sum(len(queue) for queue in self._pending_owners.values()),
            "active_transport_count": len(self._motions),
            "mechanisms_safe": mechanism_safe,
            "mechanism_positions_m": mechanism_positions,
            "robots_settled": robots_settled,
            "robot_operations": {resource: item.get("operation") for resource, item in robots.items()},
            "physical_trays_safe": trays_safe,
            "physical_tray_owners": physical_owners,
        }

    def dual_camera_rgb(self, *, width: int = 480, height: int = 360) -> np.ndarray:
        """Render Arm3 and rear fixed-camera images as one side-by-side frame."""

        if width <= 0 or height <= 0:
            raise ValueError("camera dimensions must be positive")
        if (
            self._camera_renderer is None
            or self._camera_renderer.width != int(width)
            or self._camera_renderer.height != int(height)
        ):
            if self._camera_renderer is not None:
                self._camera_renderer.close()
            self.model.vis.global_.offwidth = max(
                int(self.model.vis.global_.offwidth),
                int(width),
            )
            self.model.vis.global_.offheight = max(
                int(self.model.vis.global_.offheight),
                int(height),
            )
            self._camera_renderer = self.mujoco.Renderer(
                self.model,
                height=int(height),
                width=int(width),
            )
        frames: list[np.ndarray] = []
        for camera_name in (
            "v2_arm3_inspection_camera",
            "v2_post_braze_camera",
        ):
            self._camera_renderer.update_scene(self.data, camera=camera_name)
            self._camera_renderer.scene.flags[int(self.mujoco.mjtRndFlag.mjRND_SHADOW)] = 0
            self._camera_renderer.scene.flags[int(self.mujoco.mjtRndFlag.mjRND_REFLECTION)] = 0
            frames.append(np.asarray(self._camera_renderer.render()).copy())
        return np.concatenate(frames, axis=1)

    def close(self) -> None:
        """Symmetric facade for scene clients; MuJoCo structs need no close."""
        if self._camera_renderer is not None:
            self._camera_renderer.close()
            self._camera_renderer = None


__all__ = ["DualLineSceneAdapter"]
