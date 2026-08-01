"""Physical MuJoCo presentation for every operator-injected fault.

The recovery layer remains responsible for decisions.  This module only
mirrors those fault states into the actual scene: broken/deviated beads,
misaligned fins, stopped mechanisms, coloured equipment and safety overlays.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sin
from typing import Any, Iterable, Sequence

import numpy as np

from .domain import ProductState
from .motion import Pose


@dataclass(slots=True)
class _GeomAppearance:
    material_id: int
    rgba: np.ndarray


class PhysicalFaultVisualizer:
    """Keep fault truth and the rendered MuJoCo geometry in lockstep."""

    SUPPORTED_FAULT_TYPES = frozenset(
        {
            "FIN_POSE",
            "BRAZING_MISSING",
            "BRAZING_PATH_DEVIATION",
            "FURNACE_PROFILE",
            "FIN_PICK_FAILED",
            "FIN_GEOMETRY_FAILED",
            "ARM_UNAVAILABLE",
            "RACK_LAYER_UNAVAILABLE",
            "ELEVATOR_TIMEOUT",
            "FORK_TIMEOUT",
            "FURNACE_DOOR_INTERLOCK",
            "CONTACT_SAFETY_STOP",
            "TRAY_STATE_INCONSISTENT",
        }
    )
    ACTIVE_RED = np.asarray([1.0, 0.025, 0.015], dtype=float)
    ARMED_AMBER = np.asarray([1.0, 0.55, 0.025], dtype=float)
    SAFETY_MAGENTA = np.asarray([1.0, 0.02, 0.42], dtype=float)

    def __init__(self, scene: Any) -> None:
        self.scene = scene
        self.model = scene.model
        self.data = scene.data
        self.mujoco = scene.mujoco
        self._base_appearance: dict[int, _GeomAppearance] = {}
        self._effect_geoms: dict[str, tuple[int, ...]] = {}
        self._geom_query_cache: dict[
            tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]],
            tuple[int, ...],
        ] = {}
        self._faulted_paths: set[str] = set()
        self._furnace_effect = ""
        self._contact_body = int(self.model.body("fault_contact_marker").id)
        self._tray_ghost_body = int(self.model.body("fault_tray_ghost").id)
        self._contact_geom = int(self.model.geom("fault_contact_marker_geom").id)
        self._tray_ghost_geom = int(self.model.geom("fault_tray_ghost_geom").id)
        self.reset()

    def _geom_name(self, geom_id: int) -> str:
        return self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)) or ""

    def _body_name(self, body_id: int) -> str:
        return self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_BODY, int(body_id)) or ""

    def _matching_geoms(
        self,
        *,
        name_prefixes: Sequence[str] = (),
        exact_names: Sequence[str] = (),
        body_prefixes: Sequence[str] = (),
    ) -> tuple[int, ...]:
        cache_key = (
            tuple(name_prefixes),
            tuple(exact_names),
            tuple(body_prefixes),
        )
        cached = self._geom_query_cache.get(cache_key)
        if cached is not None:
            return cached
        exact = set(exact_names)
        result: list[int] = []
        for geom_id in range(int(self.model.ngeom)):
            geom_name = self._geom_name(geom_id)
            body_name = self._body_name(int(self.model.geom_bodyid[geom_id]))
            if (
                geom_name in exact
                or any(geom_name.startswith(prefix) for prefix in name_prefixes)
                or any(body_name.startswith(prefix) for prefix in body_prefixes)
            ):
                result.append(geom_id)
        matches = tuple(result)
        self._geom_query_cache[cache_key] = matches
        return matches

    def _capture(self, geom_id: int) -> None:
        if geom_id in self._base_appearance:
            return
        self._base_appearance[geom_id] = _GeomAppearance(
            material_id=int(self.model.geom_matid[geom_id]),
            rgba=np.asarray(self.model.geom_rgba[geom_id], dtype=float).copy(),
        )

    def _restore_effect(self, effect_id: str) -> None:
        for geom_id in self._effect_geoms.pop(effect_id, ()):
            appearance = self._base_appearance.get(geom_id)
            if appearance is None:
                continue
            self.model.geom_matid[geom_id] = appearance.material_id
            self.model.geom_rgba[geom_id] = appearance.rgba

    def _tint_effect(
        self,
        effect_id: str,
        geom_ids: Iterable[int],
        colour: Sequence[float],
        *,
        alpha: float = 1.0,
    ) -> None:
        ids = tuple(dict.fromkeys(int(value) for value in geom_ids))
        previous = self._effect_geoms.get(effect_id)
        if previous is not None and previous != ids:
            self._restore_effect(effect_id)
        self._effect_geoms[effect_id] = ids
        rgb = np.asarray(colour, dtype=float)
        for geom_id in ids:
            self._capture(geom_id)
            self.model.geom_matid[geom_id] = -1
            self.model.geom_rgba[geom_id, :3] = rgb
            original_alpha = self._base_appearance[geom_id].rgba[3]
            self.model.geom_rgba[geom_id, 3] = min(float(alpha), max(0.30, float(original_alpha)))

    def _set_mocap_pose(self, body_id: int, pose: Pose) -> None:
        mocap_id = int(self.model.body_mocapid[body_id])
        if mocap_id < 0:
            return
        self.data.mocap_pos[mocap_id] = pose.position
        self.data.mocap_quat[mocap_id] = pose.quaternion

    def _hide_overlays(self) -> None:
        self.model.geom_matid[self._contact_geom] = -1
        self.model.geom_matid[self._tray_ghost_geom] = -1
        self.model.geom_rgba[self._contact_geom, 3] = 0.0
        self.model.geom_rgba[self._tray_ghost_geom, 3] = 0.0

    def _show_contact_marker(self, actor: str, now: float) -> None:
        arm = str(actor).lower()
        try:
            pose = self.scene.registry.site_pose(f"{arm}_attachment_site")
        except Exception:
            pose = self.scene.registry.free_body_pose("assembly_tray")
        self._set_mocap_pose(self._contact_body, pose)
        pulse = 0.62 + 0.33 * abs(sin(float(now) * 8.0))
        self.model.geom_rgba[self._contact_geom] = np.asarray([1.0, 0.02, 0.01, pulse])

    def _show_tray_ghost(self, now: float) -> None:
        pose = self.scene.registry.free_body_pose("assembly_tray")
        tilt = np.deg2rad(5.0)
        ghost = pose.transformed(
            Pose(
                np.asarray([0.035, 0.0, 0.025]),
                np.asarray([np.cos(tilt / 2.0), 0.0, np.sin(tilt / 2.0), 0.0]),
            )
        )
        self._set_mocap_pose(self._tray_ghost_body, ghost)
        pulse = 0.24 + 0.20 * abs(sin(float(now) * 5.0))
        self.model.geom_rgba[self._tray_ghost_geom] = np.asarray([1.0, 0.02, 0.42, pulse])

    def sync_quality(self, product: ProductState | None) -> None:
        """Render material defects until the physical rework effect clears them."""

        if product is None:
            for path_id in tuple(self._faulted_paths):
                self._faulted_paths.remove(path_id)
            return
        active_faults: set[str] = set()
        for path in product.active_paths:
            if path.longest_gap_m > 0.0:
                self.scene.registry.set_path_gap_visual(path, path.longest_gap_m)
                active_faults.add(path.path_id)
            elif abs(path.lateral_error_m) > 0.0:
                self.scene.registry.set_path_deviation_visual(path, path.lateral_error_m)
                active_faults.add(path.path_id)
            elif path.path_id in self._faulted_paths:
                self.scene.registry.clear_path_fault_visual(path)
        self._faulted_paths = active_faults

        profile_fault = str(product.furnace.profile_fault or "")
        desired = "" if profile_fault not in {"recoverable", "severe"} else profile_fault
        if desired != self._furnace_effect:
            if self._furnace_effect:
                self._restore_effect("furnace_profile")
            self._furnace_effect = desired
            if desired:
                names = (
                    "furnace_hot_zone",
                    "furnace_control_screen",
                    "furnace_control_red",
                    "furnace_heater_left_low",
                    "furnace_heater_left_mid",
                    "furnace_heater_left_high",
                    "furnace_heater_right_low",
                    "furnace_heater_right_mid",
                    "furnace_heater_right_high",
                )
                colour = (1.0, 0.03, 0.02) if desired == "severe" else (1.0, 0.15, 0.75)
                self._tint_effect(
                    "furnace_profile",
                    self._matching_geoms(exact_names=names),
                    colour,
                    alpha=0.95,
                )

    def sync_equipment(
        self,
        holds: Sequence[dict[str, Any]],
        *,
        now: float,
        active_actor: str = "",
    ) -> None:
        """Freeze-state companion visuals for arms, transfer hardware and safety faults."""

        desired_effects: set[str] = set()
        self._hide_overlays()
        for hold in holds:
            status = str(hold.get("status", ""))
            if status not in {"ARMED", "ACTIVE"}:
                continue
            fault_type = str(hold.get("fault_type", ""))
            effect_id = f"hold:{hold.get('request_id', fault_type)}"
            desired_effects.add(effect_id)
            colour = self.ACTIVE_RED if status == "ACTIVE" else self.ARMED_AMBER
            alpha = 1.0 if status == "ACTIVE" else 0.72
            ids: tuple[int, ...] = ()
            if fault_type == "ARM_UNAVAILABLE":
                arm = str(hold.get("target", "ARM1")).lower()
                ids = self._matching_geoms(
                    name_prefixes=(f"{arm}_target_marker", f"{arm}_gripper_", f"{arm}_dispenser_"),
                    body_prefixes=(f"{arm}_fr3_", f"{arm}_parallel_gripper", f"{arm}_suction_tool"),
                )
            elif fault_type == "RACK_LAYER_UNAVAILABLE":
                layer = int(hold.get("target") or 0)
                ids = self._matching_geoms(name_prefixes=(f"batch_rack_{layer}_",))
            elif fault_type in {"ELEVATOR_TIMEOUT", "FORK_TIMEOUT"}:
                ids = self._matching_geoms(exact_names=("conveyor_belt",))
            elif fault_type == "FURNACE_DOOR_INTERLOCK":
                ids = self._matching_geoms(
                    name_prefixes=("furnace_door_", "furnace_control_"),
                    exact_names=("furnace_window",),
                )
            elif fault_type == "CONTACT_SAFETY_STOP":
                actor = str(hold.get("visual_target") or active_actor or "arm2")
                ids = self._matching_geoms(
                    body_prefixes=(f"{actor.lower()}_fr3_",),
                    exact_names=("fixture_tray_geom",),
                )
                if status == "ACTIVE":
                    self._show_contact_marker(actor, now)
            elif fault_type == "TRAY_STATE_INCONSISTENT":
                ids = self._matching_geoms(
                    exact_names=(
                        "fixture_tray_geom",
                        "batch_tray_01_geom",
                        "batch_tray_02_geom",
                        "batch_tray_03_geom",
                    )
                )
                colour = self.SAFETY_MAGENTA
                if status == "ACTIVE":
                    self._show_tray_ghost(now)
            self._tint_effect(effect_id, ids, colour, alpha=alpha)

        for effect_id in tuple(self._effect_geoms):
            if effect_id.startswith("hold:") and effect_id not in desired_effects:
                self._restore_effect(effect_id)

    def reset(self) -> None:
        for effect_id in tuple(self._effect_geoms):
            self._restore_effect(effect_id)
        self._effect_geoms.clear()
        self._faulted_paths.clear()
        self._furnace_effect = ""
        self.scene.registry.reset_path_fault_visuals()
        self._hide_overlays()


__all__ = ["PhysicalFaultVisualizer"]
