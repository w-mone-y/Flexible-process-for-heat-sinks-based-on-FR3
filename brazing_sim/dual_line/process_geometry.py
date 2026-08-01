"""Product-accurate V2 targets derived from the proven flexible V1 plans.

The V2 layout owns different world stations, but it must not own a second set
of product dimensions.  This module is the single conversion boundary from a
V1 :class:`ProcessPlan` into tray-local and measured-world targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from ..flexible import build_preset_plan
from ..flexible.models import BrazingPath, ProcessPlan

TRAY_PAYLOAD_ORIGIN_Z_M = 0.032
TRAY_BASE_SUPPORT_Z_M = 0.028
V2_MAX_BASE_LENGTH_M = 0.39
V2_MAX_BASE_WIDTH_M = 0.25
V2_MAX_PRODUCT_HEIGHT_M = 0.12


@dataclass(frozen=True, slots=True)
class DispensePass:
    """One serpentine Arm2 pass driven by the centre of both nozzles."""

    path_ids: tuple[str, str]
    start: np.ndarray
    end: np.ndarray
    paths: tuple[BrazingPath, BrazingPath]


@dataclass(frozen=True, slots=True)
class V2ProcessGeometry:
    """Immutable product geometry expressed in the live pallet frame."""

    preset: str
    base_size_m: tuple[float, float, float]
    fin_size_m: tuple[float, float, float]
    fin_pitch_m: float
    nozzle_spacing_m: float
    path_width_m: float
    fin_targets: tuple[np.ndarray, ...]
    brazing_paths: tuple[BrazingPath, ...]
    dispense_passes: tuple[DispensePass, ...]

    @classmethod
    @lru_cache(maxsize=3)
    def for_preset(cls, preset: str) -> "V2ProcessGeometry":
        plan = build_preset_plan(str(preset).upper(), quantity=1)
        return cls.from_plan(plan)

    @classmethod
    def from_plan(cls, plan: ProcessPlan) -> "V2ProcessGeometry":
        """Convert a plan into V2 tray-local geometry after envelope checks."""

        base_length, base_width, base_thickness = plan.product.base_size_m
        fin_length, _fin_thickness, fin_height = plan.product.fin_size_m
        if base_length > V2_MAX_BASE_LENGTH_M or base_width > V2_MAX_BASE_WIDTH_M:
            raise ValueError(
                "V2托盘可执行包络为基板不超过390×250 mm，"
                f"当前为{1000 * base_length:g}×{1000 * base_width:g} mm"
            )
        if fin_length > base_length:
            raise ValueError("V2翅片长度不得超过基板长度")
        if base_thickness + fin_height > V2_MAX_PRODUCT_HEIGHT_M:
            raise ValueError("V2产品总高度不得超过120 mm炉层安全包络")
        if len(plan.fin_targets) > 12 or len(plan.brazing_paths) > 24:
            raise ValueError("V2自定义产品超出12片翅片/24条焊道实体对象池")
        base_center_z = TRAY_BASE_SUPPORT_Z_M + 0.5 * base_thickness
        fin_targets = tuple(
            np.asarray(target.position, dtype=float) + np.asarray([0.0, 0.0, base_center_z], dtype=float)
            for target in plan.fin_targets
        )
        passes: list[DispensePass] = []
        for offset in range(0, len(plan.brazing_paths), 2):
            group = plan.brazing_paths[offset : offset + 2]
            if len(group) != 2 or group[0].fin_id != group[1].fin_id:
                raise ValueError(f"{plan.product.product_id}双喷嘴路径未成对")
            start = np.mean(np.asarray([path.start for path in group], dtype=float), axis=0)
            end = np.mean(np.asarray([path.end for path in group], dtype=float), axis=0)
            tip_z = (
                TRAY_BASE_SUPPORT_Z_M
                + plan.execution_spec.base_thickness
                + float(plan.product.nozzle_tip_height_m)
            )
            start[2] = tip_z
            end[2] = tip_z
            passes.append(
                DispensePass(
                    path_ids=(group[0].path_id, group[1].path_id),
                    start=start,
                    end=end,
                    paths=(group[0], group[1]),
                )
            )
        return cls(
            preset=plan.execution_spec.preset,
            base_size_m=plan.product.base_size_m,
            fin_size_m=plan.product.fin_size_m,
            fin_pitch_m=plan.product.fin_pitch_m,
            nozzle_spacing_m=plan.product.nozzle_spacing_m,
            path_width_m=plan.product.path_width_m,
            fin_targets=fin_targets,
            brazing_paths=plan.brazing_paths,
            dispense_passes=tuple(passes),
        )

    @classmethod
    def for_unit(cls, unit: Any) -> "V2ProcessGeometry":
        """Return order-bound geometry, retaining preset compatibility."""

        geometry = getattr(unit, "process_geometry", None)
        if isinstance(geometry, cls):
            return geometry
        return cls.for_preset(str(unit.preset))

    @property
    def path_length_m(self) -> float:
        if not self.dispense_passes:
            return 0.0
        first = self.dispense_passes[0]
        return float(np.linalg.norm(first.end - first.start))

    @property
    def base_center_z_m(self) -> float:
        """Thickness-aware centre that keeps every base bottom on one support."""

        return TRAY_BASE_SUPPORT_Z_M + 0.5 * self.base_size_m[2]

    @property
    def base_top_z_m(self) -> float:
        return TRAY_BASE_SUPPORT_Z_M + self.base_size_m[2]

    @staticmethod
    def _world(local: np.ndarray, *, origin: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        return np.asarray(origin, dtype=float) + np.asarray(rotation, dtype=float).reshape(3, 3) @ np.asarray(
            local,
            dtype=float,
        )

    def world_fin_target(
        self,
        index: int,
        *,
        origin: np.ndarray,
        rotation: np.ndarray,
    ) -> np.ndarray:
        return self._world(self.fin_targets[index], origin=origin, rotation=rotation)

    def world_dispense_pass(
        self,
        index: int,
        *,
        origin: np.ndarray,
        rotation: np.ndarray,
    ) -> DispensePass:
        item = self.dispense_passes[index]
        return DispensePass(
            path_ids=item.path_ids,
            start=self._world(item.start, origin=origin, rotation=rotation),
            end=self._world(item.end, origin=origin, rotation=rotation),
            paths=item.paths,
        )


__all__ = [
    "DispensePass",
    "TRAY_BASE_SUPPORT_Z_M",
    "TRAY_PAYLOAD_ORIGIN_Z_M",
    "V2_MAX_BASE_LENGTH_M",
    "V2_MAX_BASE_WIDTH_M",
    "V2_MAX_PRODUCT_HEIGHT_M",
    "V2ProcessGeometry",
]
