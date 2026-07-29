"""Product-accurate V2 targets derived from the proven flexible V1 plans.

The V2 layout owns different world stations, but it must not own a second set
of product dimensions.  This module is the single conversion boundary from a
V1 :class:`ProcessPlan` into tray-local and measured-world targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..flexible import build_preset_plan
from ..flexible.models import BrazingPath

TRAY_PAYLOAD_ORIGIN_Z_M = 0.032


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
        fin_targets = tuple(
            np.asarray(target.position, dtype=float)
            + np.asarray([0.0, 0.0, TRAY_PAYLOAD_ORIGIN_Z_M], dtype=float)
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
                TRAY_PAYLOAD_ORIGIN_Z_M
                + 0.5 * plan.execution_spec.base_thickness
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

    @property
    def path_length_m(self) -> float:
        if not self.dispense_passes:
            return 0.0
        first = self.dispense_passes[0]
        return float(np.linalg.norm(first.end - first.start))

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


__all__ = ["DispensePass", "TRAY_PAYLOAD_ORIGIN_Z_M", "V2ProcessGeometry"]
