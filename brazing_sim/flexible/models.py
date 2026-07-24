"""Strongly typed configuration and execution-plan models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from ..domain import BrazingRecipe, BrazingSide, OrderSpec, Vec3


class RouteStrategy(str, Enum):
    STANDARD = "STANDARD"
    HIGH_RELIABILITY = "HIGH_RELIABILITY"
    FIRST_ARTICLE = "FIRST_ARTICLE"


@dataclass(frozen=True, slots=True)
class ProductConfig:
    schema_version: int
    product_id: str
    preset: str
    base_size_m: Vec3
    fin_size_m: Vec3
    fin_count: int
    fin_pitch_m: float
    start_offset_y_m: float | None
    path_margin_m: float
    path_width_m: float
    brazing_sides: tuple[BrazingSide, ...]
    comb_module: str
    target_clamping_force_n: float
    clamping_force_tolerance_n: float
    force_hold_duration_s: float
    nozzle_spacing_m: float
    bead_offset_m: float
    nozzle_tip_height_m: float
    material_speed_m_s: float
    recipe: str
    material_system: str = "demo_brazing_material"


@dataclass(frozen=True, slots=True)
class OrderConfig:
    schema_version: int
    order_id: str
    product: str
    quantity: int
    priority: int
    due_time: datetime | None
    preferred_rack_layer: int | None
    source_file: Path


@dataclass(frozen=True, slots=True)
class FixtureModuleConfig:
    name: str
    pitch_m: float
    slot_count: int
    front_body: str
    rear_body: str
    legacy: bool


@dataclass(frozen=True, slots=True)
class ProcessRecipeConfig:
    name: str
    ambient_c: float
    preheat_c: float
    peak_c: float
    unload_c: float
    preheat_seconds: float
    ramp_seconds: float
    soak_seconds: float
    cooling_seconds: float
    door_seconds: float

    def to_domain(self) -> BrazingRecipe:
        return BrazingRecipe(
            ambient_c=self.ambient_c,
            preheat_c=self.preheat_c,
            peak_c=self.peak_c,
            unload_c=self.unload_c,
            preheat_seconds=self.preheat_seconds,
            ramp_seconds=self.ramp_seconds,
            soak_seconds=self.soak_seconds,
            cooling_seconds=self.cooling_seconds,
            door_seconds=self.door_seconds,
        )


@dataclass(frozen=True, slots=True)
class RackLayerConfig:
    index: int
    height_m: float


@dataclass(frozen=True, slots=True)
class RackConfig:
    policy: str
    layers: tuple[RackLayerConfig, ...]


@dataclass(frozen=True, slots=True)
class FinTarget:
    fin_id: str
    index: int
    position: Vec3
    quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class BrazingPath:
    path_id: str
    fin_id: str
    side: BrazingSide
    start: Vec3
    end: Vec3
    width_m: float

    @property
    def length_m(self) -> float:
        return abs(self.end[0] - self.start[0])


@dataclass(frozen=True, slots=True)
class RackAssignment:
    unit_index: int
    tray_id: str
    layer_index: int
    height_m: float


@dataclass(frozen=True, slots=True)
class ProcessPlan:
    order: OrderConfig
    product: ProductConfig
    execution_spec: OrderSpec
    fin_targets: tuple[FinTarget, ...]
    brazing_paths: tuple[BrazingPath, ...]
    fixture_module: FixtureModuleConfig
    recipe: ProcessRecipeConfig
    rack_assignments: tuple[RackAssignment, ...]
    max_fins: int = 12
    max_paths: int = 24
    path_segment_capacity: int = 20
    route_strategy: RouteStrategy | str = RouteStrategy.STANDARD

    def __post_init__(self) -> None:
        object.__setattr__(self, "route_strategy", RouteStrategy(self.route_strategy))

    @property
    def selected_rack_layer(self) -> int:
        """Compatibility accessor for consumers that previously handled one part."""

        return self.rack_assignments[0].layer_index

    @property
    def quantity(self) -> int:
        return self.order.quantity

    def summary(self) -> dict[str, Any]:
        return {
            "order_id": self.order.order_id,
            "product_id": self.product.product_id,
            "preset": self.product.preset,
            "quantity": self.quantity,
            "fin_count": len(self.fin_targets),
            "path_count": len(self.brazing_paths),
            "comb_module": self.fixture_module.name,
            "clamping_force_n": self.product.target_clamping_force_n,
            "nozzle_spacing_m": self.product.nozzle_spacing_m,
            "material_speed_m_s": self.product.material_speed_m_s,
            "rack_layers": [item.layer_index for item in self.rack_assignments],
            "path_length_m": self.brazing_paths[0].length_m if self.brazing_paths else 0.0,
            "base_size_m": list(self.product.base_size_m),
            "fin_size_m": list(self.product.fin_size_m),
            "fin_pitch_m": self.product.fin_pitch_m,
            "path_margin_m": self.product.path_margin_m,
            "bead_offset_m": self.product.bead_offset_m,
            "nozzle_tip_height_m": self.product.nozzle_tip_height_m,
            "recipe": self.recipe.name,
            "material_system": self.product.material_system,
            "route_strategy": self.route_strategy.value,
            "fin_targets": [
                {"fin_id": item.fin_id, "position": list(item.position)} for item in self.fin_targets
            ],
            "brazing_paths": [
                {
                    "path_id": item.path_id,
                    "start": list(item.start),
                    "end": list(item.end),
                    "width_m": item.width_m,
                }
                for item in self.brazing_paths
            ],
        }


__all__ = [
    "BrazingPath",
    "FinTarget",
    "FixtureModuleConfig",
    "OrderConfig",
    "ProcessPlan",
    "ProcessRecipeConfig",
    "ProductConfig",
    "RackAssignment",
    "RackConfig",
    "RackLayerConfig",
    "RouteStrategy",
]
