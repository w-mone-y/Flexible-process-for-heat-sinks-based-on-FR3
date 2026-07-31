"""Make V2 faults visible in MuJoCo.

The *mechanism* is V1's and is deliberately reused: capture each geom's original
material/rgba once, tint by effect id, and restore by diffing effect sets.  What
could not be reused is the **name table** — V1 tints bodies like
``furnace_heater_left_mid`` and ``batch_rack_0_``, none of which exist in
``brazing_line_v2.xml``.  Of thirteen V1 names probed against the V2 scene only
``furnace_hot_zone`` had any counterpart.

So the geometry vocabulary is declared once, as data, in :data:`_FAULT_GEOMETRY`.
Adding a scene means adding a table, not editing tinting code.

This module holds the only MuJoCo dependency in V2's fault stack; the controller
in :mod:`brazing_sim.dual_line.faults` stays scene-free and headless-testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

# Per-fault geometry selectors for the V2 scene.  ``prefixes`` matches geom names,
# ``bodies`` matches owning body names, ``exact`` matches a full geom name.
# ``{arm}`` and ``{layer}`` are substituted from the fault's target.
_FAULT_GEOMETRY: dict[str, dict[str, tuple[str, ...]]] = {
    "ARM_UNAVAILABLE": {
        "prefixes": ("v2_{arm}_gripper_", "v2_{arm}_dispenser_", "v2_{arm}_camera_"),
        "bodies": ("v2_{arm}_fr3_", "v2_{arm}_"),
    },
    # Furnace layers and doors are bodies whose geoms are unnamed in the MJCF,
    # so they must be selected by owning body rather than by geom prefix.
    "RACK_LAYER_UNAVAILABLE": {
        "bodies": ("v2_furnace_layer_{layer}_",),
    },
    "ELEVATOR_TIMEOUT": {
        "prefixes": ("v2_furnace_lift", "v2_furnace_pusher"),
        "exact": ("v2_belt_mat",),
    },
    "FORK_TIMEOUT": {
        "prefixes": ("v2_furnace_pusher", "v2_furnace_rear_extractor"),
    },
    "FURNACE_DOOR_INTERLOCK": {
        "bodies": ("v2_furnace_front_door", "v2_furnace_rear_door"),
    },
    "FURNACE_PROFILE": {
        "prefixes": ("v2_furnace_hot_zone",),
    },
    "CONTACT_SAFETY_STOP": {
        # The whole arm that made unexpected contact, plus the shared merge rails
        # it was reaching over.
        "bodies": ("v2_{arm}_", "v2_merge"),
    },
    "TRAY_STATE_INCONSISTENT": {
        "prefixes": ("v2_output_carriage", "v2_output_belt"),
    },
    # Workpiece-level faults tint the affected tray so the operator can see which
    # pallet carries the defect, since V2 has no per-bead fault geometry yet.
    "BRAZING_MISSING": {"prefixes": ("v2_tray_{tray}_",)},
    "BRAZING_PATH_DEVIATION": {"prefixes": ("v2_tray_{tray}_",)},
    "FIN_PICK_FAILED": {"prefixes": ("v2_tray_{tray}_",)},
    "FIN_INSERT_FAILED": {"prefixes": ("v2_tray_{tray}_",)},
    "FIN_GEOMETRY_FAILED": {"prefixes": ("v2_tray_{tray}_",)},
}

_SAFETY_FAULTS = frozenset({"CONTACT_SAFETY_STOP", "TRAY_STATE_INCONSISTENT"})


@dataclass(slots=True)
class _GeomAppearance:
    material_id: int
    rgba: np.ndarray
    position: np.ndarray
    quaternion: np.ndarray
    size: np.ndarray


class V2FaultVisualizer:
    """Mirror V2 fault state into the rendered scene."""

    ACTIVE_RED = np.asarray([1.0, 0.025, 0.015], dtype=float)
    RECOVERING_AMBER = np.asarray([1.0, 0.55, 0.025], dtype=float)
    SAFETY_MAGENTA = np.asarray([1.0, 0.02, 0.42], dtype=float)

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter
        self.model = adapter.model
        self.data = adapter.data
        self.mujoco = adapter.mujoco
        self._base_appearance: dict[int, _GeomAppearance] = {}
        self._effect_geoms: dict[str, tuple[int, ...]] = {}
        self._query_cache: dict[tuple[Any, ...], tuple[int, ...]] = {}

    # ---------------------------------------------------------------- plumbing
    def _geom_name(self, geom_id: int) -> str:
        return self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)) or ""

    def _body_name(self, body_id: int) -> str:
        return self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_BODY, int(body_id)) or ""

    def _matching_geoms(
        self,
        *,
        prefixes: Sequence[str] = (),
        exact: Sequence[str] = (),
        bodies: Sequence[str] = (),
    ) -> tuple[int, ...]:
        key = (tuple(prefixes), tuple(exact), tuple(bodies))
        cached = self._query_cache.get(key)
        if cached is not None:
            return cached
        wanted = set(exact)
        found: list[int] = []
        for geom_id in range(int(self.model.ngeom)):
            geom_name = self._geom_name(geom_id)
            body_name = self._body_name(int(self.model.geom_bodyid[geom_id]))
            if (
                geom_name in wanted
                or any(geom_name.startswith(prefix) for prefix in prefixes)
                or any(body_name.startswith(prefix) for prefix in bodies)
            ):
                found.append(geom_id)
        result = tuple(found)
        self._query_cache[key] = result
        return result

    def _capture(self, geom_id: int) -> None:
        if geom_id in self._base_appearance:
            return
        original_positions = getattr(self.adapter, "_original_geom_pos", {})
        original_quaternions = getattr(self.adapter, "_original_geom_quat", {})
        original_sizes = getattr(self.adapter, "_original_geom_size", {})
        self._base_appearance[geom_id] = _GeomAppearance(
            material_id=int(self.model.geom_matid[geom_id]),
            rgba=np.asarray(self.model.geom_rgba[geom_id], dtype=float).copy(),
            position=np.asarray(
                original_positions.get(geom_id, self.model.geom_pos[geom_id]), dtype=float
            ).copy(),
            quaternion=np.asarray(
                original_quaternions.get(geom_id, self.model.geom_quat[geom_id]), dtype=float
            ).copy(),
            size=np.asarray(original_sizes.get(geom_id, self.model.geom_size[geom_id]), dtype=float).copy(),
        )

    def _restore_effect(self, effect_id: str) -> None:
        for geom_id in self._effect_geoms.pop(effect_id, ()):
            appearance = self._base_appearance.get(geom_id)
            if appearance is None:
                continue
            self.model.geom_matid[geom_id] = appearance.material_id
            authored_rgba = getattr(self.adapter, "_visible_rgba", {}).get(geom_id)
            visible = getattr(self.adapter, "_geom_visibility", {}).get(geom_id)
            if authored_rgba is not None and visible is not None:
                rgba = np.asarray(authored_rgba, dtype=float).copy()
                rgba[3] = rgba[3] if visible else 0.0
                self.model.geom_rgba[geom_id] = rgba
            else:
                self.model.geom_rgba[geom_id] = appearance.rgba
            self.model.geom_pos[geom_id] = appearance.position
            self.model.geom_quat[geom_id] = appearance.quaternion
            self.model.geom_size[geom_id] = appearance.size

    def _tint(
        self,
        effect_id: str,
        geom_ids: Iterable[int],
        colour: Sequence[float],
        *,
        alpha: float = 1.0,
    ) -> None:
        ids = tuple(dict.fromkeys(int(value) for value in geom_ids))
        if not ids:
            return
        previous = self._effect_geoms.get(effect_id)
        if previous is not None and previous != ids:
            self._restore_effect(effect_id)
        self._effect_geoms[effect_id] = ids
        rgb = np.asarray(colour, dtype=float)
        for geom_id in ids:
            self._capture(geom_id)
            self.model.geom_matid[geom_id] = -1
            self.model.geom_rgba[geom_id, :3] = rgb
            original_alpha = float(self._base_appearance[geom_id].rgba[3])
            self.model.geom_rgba[geom_id, 3] = min(float(alpha), max(0.30, original_alpha))

    # ------------------------------------------------------------------- public
    @staticmethod
    def _selectors(fault_type: str, target: str, tray_id: str | None) -> dict[str, tuple[str, ...]]:
        table = _FAULT_GEOMETRY.get(fault_type)
        if table is None:
            return {}
        arm = target.lower() if target.upper().startswith("ARM") else "arm1"
        layer = target if target.isdigit() else "0"
        tray = "" if not tray_id else str(tray_id).removeprefix("V2_TRAY_").lower()
        resolved: dict[str, tuple[str, ...]] = {}
        for key, patterns in table.items():
            filled = tuple(pattern.format(arm=arm, layer=layer, tray=tray) for pattern in patterns)
            # A tray-scoped selector with no tray would match every pallet.
            if tray or all("{tray}" not in pattern for pattern in patterns):
                resolved[key] = filled
        return resolved

    @staticmethod
    def _path_component(target: str) -> str | None:
        value = str(target).lower()
        if value.startswith("path_"):
            try:
                return f"braze_{int(value.split('_')[1]):02d}"
            except (IndexError, ValueError):
                return None
        if value.startswith("slot_"):
            parts = value.split("_")
            try:
                slot = int(parts[1])
            except (IndexError, ValueError):
                return None
            side = parts[2] if len(parts) > 2 else "left"
            return f"braze_{2 * slot - (1 if side == 'left' else 0):02d}"
        return None

    def _physical_geom(
        self,
        defect: Mapping[str, Any],
        tray_id: str | None,
    ) -> int | None:
        if not tray_id:
            return None
        tray = str(tray_id).lower()
        visual_type = str(defect.get("visual_type") or defect.get("fault_type") or "")
        target = str(defect.get("target") or "")
        if visual_type.startswith("BRAZING_"):
            component = self._path_component(target)
        elif visual_type.startswith("FIN_"):
            try:
                component = f"fin_{int(target.split('_')[1]):02d}"
            except (IndexError, ValueError):
                component = None
        else:
            component = None
        if component is None:
            return None
        name = f"{tray.lower()}_{component}"
        geom_id = int(self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, name))
        return None if geom_id < 0 else geom_id

    def _apply_physical_defect(
        self,
        defect: Mapping[str, Any],
        tray_id: str | None,
    ) -> str | None:
        status = str(defect.get("status", ""))
        if status not in {"MANIFESTED", "DETECTED"}:
            return None
        effect_id = f"physical:{defect.get('defect_id', '')}"
        visual_type = str(defect.get("visual_type") or defect.get("fault_type") or "")
        if visual_type == "FURNACE_PROFILE":
            geom_ids = self._matching_geoms(prefixes=("v2_furnace_hot_zone",))
            self._tint(effect_id, geom_ids, self.ACTIVE_RED)
            return effect_id if geom_ids else None
        geom_id = self._physical_geom(defect, tray_id)
        if geom_id is None:
            return None
        self._capture(geom_id)
        appearance = self._base_appearance[geom_id]
        # Start each frame from the authored/configured pose so effects never
        # accumulate into a drifting workpiece.
        self.model.geom_pos[geom_id] = appearance.position
        self.model.geom_quat[geom_id] = appearance.quaternion
        self.model.geom_size[geom_id] = appearance.size
        self._effect_geoms[effect_id] = (geom_id,)
        repair_progress = float(np.clip(float(defect.get("repair_progress", 0.0)), 0.0, 1.0))
        remaining = 1.0 - repair_progress
        if visual_type == "BRAZING_MISSING":
            # Retain the completed 36% and grow only the missing tail during
            # local repair.  Always derive from the captured authored geometry;
            # deriving from the previous frame caused exponential shrinkage.
            half_length = float(appearance.size[1])
            filled_fraction = 0.36 + 0.64 * repair_progress
            self.model.geom_size[geom_id, 1] = max(1.0e-6, half_length * filled_fraction)
            self.model.geom_pos[geom_id, 0] -= half_length * (1.0 - filled_fraction)
        elif visual_type == "BRAZING_PATH_DEVIATION":
            self.model.geom_pos[geom_id, 1] += 0.012 * remaining
            self.model.geom_matid[geom_id] = -1
            self.model.geom_rgba[geom_id, :3] = self.ACTIVE_RED
        elif visual_type == "FIN_PICK_FAILED":
            # The fin is absent from the comb slot while the tray continues.
            self.model.geom_pos[geom_id, 2] -= 0.18 * remaining
        else:
            # Insert/pose faults remain obvious to both the viewer and camera:
            # the selected fin is raised, shifted and tilted—not the whole tray.
            angle = np.deg2rad(11.0) * remaining
            self.model.geom_pos[geom_id, 1] += 0.012 * remaining
            self.model.geom_pos[geom_id, 2] += 0.018 * remaining
            self.model.geom_quat[geom_id] = np.asarray(
                [np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)],
                dtype=float,
            )
            self.model.geom_matid[geom_id] = -1
            self.model.geom_rgba[geom_id, :3] = self.RECOVERING_AMBER
        return effect_id

    def sync(
        self,
        faults: Iterable[Mapping[str, Any]],
        units: Iterable[Mapping[str, Any]],
        physical_faults: Iterable[Mapping[str, Any]] = (),
    ) -> None:
        """Mirror equipment faults and exact latent workpiece defects."""

        tray_by_unit = {
            str(unit.get("unit_id")): unit.get("tray_id") for unit in units if unit.get("tray_id")
        }
        wanted: set[str] = set()
        defect_rows = list(physical_faults)
        for defect in defect_rows:
            tray_id = tray_by_unit.get(str(defect.get("unit_id") or ""))
            effect_id = self._apply_physical_defect(defect, tray_id)
            if effect_id:
                wanted.add(effect_id)
        for record in faults:
            if record.get("recovered"):
                continue
            fault_type = str(record.get("fault_type", ""))
            if fault_type in {
                "BRAZING_MISSING",
                "BRAZING_PATH_DEVIATION",
                "FIN_PICK_FAILED",
                "FIN_INSERT_FAILED",
                "FIN_GEOMETRY_FAILED",
            }:
                # Workpiece faults are represented by their exact physical
                # defect above, never by tinting every geom on the pallet.
                continue
            details = record.get("details") or {}
            target = str(details.get("target") or record.get("source") or "")
            tray_id = tray_by_unit.get(str(details.get("unit_id") or ""))
            selectors = self._selectors(fault_type, target, tray_id)
            if not selectors:
                continue
            geom_ids = self._matching_geoms(**selectors)
            if not geom_ids:
                continue
            effect_id = f"{fault_type}:{target or '-'}:{tray_id or '-'}"
            wanted.add(effect_id)
            colour = (
                self.SAFETY_MAGENTA
                if fault_type in _SAFETY_FAULTS
                else self.ACTIVE_RED if details.get("severity") == "severe" else self.RECOVERING_AMBER
            )
            self._tint(effect_id, geom_ids, colour)
        # Anything no longer faulted returns to its authored appearance.
        for effect_id in [key for key in self._effect_geoms if key not in wanted]:
            self._restore_effect(effect_id)

    def reset(self) -> None:
        for effect_id in list(self._effect_geoms):
            self._restore_effect(effect_id)
        self._effect_geoms.clear()

    @property
    def active_effects(self) -> tuple[str, ...]:
        return tuple(sorted(self._effect_geoms))


__all__ = ["V2FaultVisualizer"]
