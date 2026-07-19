"""Order presets and product-coordinate layout derivation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import cycle
from math import isfinite
from typing import Any, Iterator
from uuid import uuid4

from .domain import (
    BatchState,
    BrazingPathState,
    BrazingSide,
    FinState,
    OrderSpec,
    ProductState,
    RackShelf,
    RackState,
    TrayUnitState,
    Vec3,
)

PRODUCT_AXES = {
    "x": "fin_length",
    "y": "fin_array",
    "z": "fin_height",
}


@dataclass(frozen=True, slots=True)
class FixtureConfig:
    """Centralized demonstrator geometry and press settings (not production data)."""

    tray_size: Vec3 = (0.46, 0.32, 0.018)
    front_comb_x: float = -0.12
    rear_comb_x: float = 0.12
    comb_height: float = 0.046
    comb_slot_width: float = 0.004
    base_clearance: float = 0.001
    target_clamping_force_n: float = 30.0
    clamping_force_tolerance_n: float = 3.0
    press_floating_stiffness_n_m: float = 12000.0
    force_hold_duration_s: float = 1.0
    press_search_speed_m_s: float = 0.004
    press_ramp_duration_s: float = 1.5
    press_travel_m: float = 0.024

    def __post_init__(self) -> None:
        values = (
            *self.tray_size,
            self.comb_height,
            self.comb_slot_width,
            self.base_clearance,
            self.target_clamping_force_n,
            self.clamping_force_tolerance_n,
            self.press_floating_stiffness_n_m,
            self.force_hold_duration_s,
            self.press_search_speed_m_s,
            self.press_ramp_duration_s,
            self.press_travel_m,
        )
        if any(not isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("fixture dimensions, forces and durations must be finite and positive")
        if not isfinite(self.front_comb_x) or not isfinite(self.rear_comb_x):
            raise ValueError("comb X coordinates must be finite")
        if self.front_comb_x >= self.rear_comb_x:
            raise ValueError("front comb must precede rear comb along product X")
        if self.clamping_force_tolerance_n >= self.target_clamping_force_n:
            raise ValueError("clamping-force tolerance must be smaller than its target")


@dataclass(frozen=True, slots=True)
class DispenserConfig:
    """Symmetric dual-nozzle layout used before fins are installed."""

    fin_thickness: float = 0.002
    nozzle_spacing: float = 0.005
    bead_offset_from_slot_center: float = 0.0025
    nozzle_tip_height: float = 0.0015
    approach_height: float = 0.080
    travel_speed: float = 0.10
    nozzle_inward_angle_deg: float = 35.0
    target_bead_width: float = 0.004
    safety_clearance: float = 0.001
    anti_drip_enabled: bool = True

    def __post_init__(self) -> None:
        values = (
            self.fin_thickness,
            self.nozzle_spacing,
            self.bead_offset_from_slot_center,
            self.nozzle_tip_height,
            self.approach_height,
            self.travel_speed,
            self.target_bead_width,
            self.safety_clearance,
        )
        if any(not isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("dispenser dimensions and speeds must be finite and positive")
        if not 0.0 < self.nozzle_inward_angle_deg < 90.0:
            raise ValueError("nozzle inward angle must be between 0 and 90 degrees")
        expected_spacing = 2.0 * self.bead_offset_from_slot_center
        if abs(self.nozzle_spacing - expected_spacing) > 1.0e-9:
            raise ValueError("nozzle spacing must equal twice the symmetric bead offset")

    def required_bead_offset(self, fin_thickness: float) -> float:
        """Return the safe bead-centre offset for one fin thickness.

        Half the fin, one explicit safety gap and half the deposited bead must
        fit between the slot centre and either nozzle.  Keeping this relation in
        the configuration contract prevents a visually symmetric tool from
        depositing material underneath a wider fin.
        """

        return 0.5 * fin_thickness + self.safety_clearance

    def validate_for_fin_thickness(self, fin_thickness: float) -> None:
        if not isfinite(fin_thickness) or fin_thickness <= 0.0:
            raise ValueError("fin thickness must be finite and positive")
        if self.bead_offset_from_slot_center <= fin_thickness / 2.0:
            raise ValueError("bead offset must place the bead centre outside the fin face")


FIXTURE_CONFIG = FixtureConfig()
DISPENSER_CONFIG = DispenserConfig()


A_ORDER_SPEC = OrderSpec(
    preset="A",
    base_size=(0.36, 0.22, 0.008),
    fin_size=(0.30, DISPENSER_CONFIG.fin_thickness, 0.06),
    fin_count=5,
    fin_pitch=0.020,
    comb_module_name="comb_insert_20mm",
    brazing_sides=(BrazingSide.LEFT, BrazingSide.RIGHT),
    path_width=DISPENSER_CONFIG.target_bead_width,
    max_fins=12,
    max_paths=24,
    target_clamping_force_n=20.0,
    clamping_force_tolerance_n=2.0,
    force_hold_duration_s=1.5,
    nozzle_spacing=0.005,
    bead_offset=0.0025,
)

B_ORDER_SPEC = replace(
    A_ORDER_SPEC,
    preset="B",
    fin_count=4,
    fin_pitch=0.030,
    base_size=(0.36, 0.24, 0.008),
    comb_module_name="comb_insert_30mm",
    target_clamping_force_n=18.0,
)

C_ORDER_SPEC = replace(
    A_ORDER_SPEC,
    preset="C",
    base_size=(0.34, 0.20, 0.008),
    fin_size=(0.28, 0.0018, 0.055),
    fin_count=7,
    fin_pitch=0.015,
    comb_module_name="comb_insert_15mm",
    target_clamping_force_n=22.0,
    nozzle_spacing=0.0044,
    bead_offset=0.0022,
    material_speed=0.09,
)

ORDER_PRESETS: dict[str, OrderSpec] = {
    "A": A_ORDER_SPEC,
    "B": B_ORDER_SPEC,
    "C": C_ORDER_SPEC,
}


def _load_yaml_presets() -> dict[str, OrderSpec]:
    """Keep legacy A/B/C callers on the same source of truth as flexible orders."""

    from .flexible.planner import build_preset_plan

    return {key: build_preset_plan(key, quantity=1).execution_spec for key in ("A", "B", "C")}


ORDER_PRESETS = _load_yaml_presets()
A_ORDER_SPEC = ORDER_PRESETS["A"]
B_ORDER_SPEC = ORDER_PRESETS["B"]
C_ORDER_SPEC = ORDER_PRESETS["C"]


@dataclass(slots=True)
class ProductLayout:
    """All preallocated visual slots for one immutable order specification."""

    spec: OrderSpec
    fins: list[FinState]
    paths: list[BrazingPathState]
    isolation_origin: Vec3

    @property
    def active_fins(self) -> list[FinState]:
        return [fin for fin in self.fins if fin.active]

    @property
    def active_paths(self) -> list[BrazingPathState]:
        return [path for path in self.paths if path.active]

    def __iter__(self) -> Iterator[list[FinState] | list[BrazingPathState]]:
        """Allow ``fins, paths = derive_product_layout(spec)`` adapters."""

        yield self.fins
        yield self.paths


def make_order_spec(preset: str = "A", **overrides: Any) -> OrderSpec:
    """Return a validated order spec.

    ``fin_count=6`` or ``fin_count=8`` may be supplied without changing the
    scene's fixed allocation of eight fin bodies and sixteen path visuals.
    """

    key = preset.upper()
    if key not in ORDER_PRESETS:
        raise KeyError(f"unknown order preset: {preset!r}")
    base = ORDER_PRESETS[key]
    if "preset" in overrides and str(overrides["preset"]).upper() != key:
        raise ValueError("preset override must match the selected preset")
    overrides["preset"] = key
    if "brazing_sides" in overrides:
        overrides["brazing_sides"] = tuple(BrazingSide(side) for side in overrides["brazing_sides"])
    spec = replace(base, **overrides)
    if spec.bead_offset <= spec.fin_thickness / 2.0:
        raise ValueError("order bead offset must remain outside the fin face")
    return spec


get_order_spec = make_order_spec
preset_a = make_order_spec


def _fin_y_positions(
    fin_count: int,
    pitch: float,
    start_offset_y: float | None = None,
) -> tuple[float, ...]:
    if start_offset_y is not None:
        return tuple(start_offset_y + index * pitch for index in range(fin_count))
    centre = (fin_count - 1) / 2.0
    return tuple((index - centre) * pitch for index in range(fin_count))


def derive_product_layout(
    spec: OrderSpec,
    *,
    isolation_origin: Vec3 = (0.60, -0.60, -0.20),
) -> ProductLayout:
    """Derive fin slots and root paths in the product coordinate frame.

    The product origin is the centre of the base plate.  Therefore the base
    top is ``z=base_thickness/2`` and a fin centre is one half fin-height above
    that.  Unused allocation slots are moved to the isolation origin and have
    both visibility and collision participation disabled.
    """

    ys = _fin_y_positions(spec.fin_count, spec.fin_pitch, spec.start_offset_y)
    base_top_z = spec.base_thickness / 2.0
    fin_centre_z = base_top_z + spec.fin_height / 2.0
    fins: list[FinState] = []
    paths: list[BrazingPathState] = []

    for index in range(spec.max_fins):
        active = index < spec.fin_count
        if active:
            position = (0.0, ys[index], fin_centre_z)
        else:
            position = (
                isolation_origin[0] + 0.04 * (index - spec.fin_count),
                isolation_origin[1],
                isolation_origin[2],
            )
        fins.append(
            FinState(
                fin_id=f"fin_{index + 1:02d}",
                index=index,
                target_position=position,
                actual_position=position,
                active=active,
                hidden=not active,
                collision_enabled=active,
            )
        )

    sides = tuple(spec.brazing_sides)
    side_cycle = cycle(sides)
    active_path_count = spec.path_count
    for index in range(spec.max_paths):
        active = index < active_path_count
        fin_index = index // len(sides)
        side = sides[index % len(sides)] if active else next(side_cycle)
        if active:
            fin = fins[fin_index]
            sign = -1.0 if side is BrazingSide.LEFT else 1.0
            y = fin.target_position[1] + sign * spec.bead_offset
            x_start = -spec.base_length / 2.0 + spec.path_margin
            x_end = spec.base_length / 2.0 - spec.path_margin
            start = (x_start, y, base_top_z)
            end = (x_end, y, base_top_z)
            fin_id = fin.fin_id
            slot_id = f"slot_{fin_index + 1:02d}"
        else:
            inactive_slot = index - active_path_count
            point = (
                isolation_origin[0] + 0.02 * inactive_slot,
                isolation_origin[1] - 0.04,
                isolation_origin[2],
            )
            start = end = point
            # Preserve deterministic XML-compatible pairing for all 16 slots.
            fin_id = f"fin_{fin_index + 1:02d}"
            slot_id = f"slot_{fin_index + 1:02d}"
        paths.append(
            BrazingPathState(
                path_id=f"{slot_id}_{side.value}",
                index=index,
                fin_id=fin_id,
                slot_id=slot_id,
                side=side,
                local_start=start,
                local_end=end,
                target_width_m=spec.path_width,
                active=active,
                hidden=not active,
                collision_enabled=active,
            )
        )

    return ProductLayout(spec=spec, fins=fins, paths=paths, isolation_origin=isolation_origin)


def create_product_state(
    spec: OrderSpec | None = None,
    *,
    order_id: str | None = None,
    created_at: float = 0.0,
) -> ProductState:
    selected = spec or make_order_spec("A")
    layout = derive_product_layout(selected)
    return ProductState(
        order_id=order_id or f"{selected.preset}-{uuid4().hex[:8]}",
        spec=selected,
        fins=layout.fins,
        paths=layout.paths,
        created_at=created_at,
    )


def create_batch_state(
    preset: str = "A",
    *,
    layers: int = 3,
    spec: OrderSpec | None = None,
    layer_indices: tuple[int, ...] | None = None,
    batch_id: str | None = None,
    created_at: float = 0.0,
) -> BatchState:
    """Create one to three independent products assigned to physical shelves."""

    if not 1 <= int(layers) <= 3:
        raise ValueError("batch layers must be between one and three")
    selected_spec = spec or make_order_spec(preset)
    assignments = layer_indices or tuple(range(int(layers)))
    if len(assignments) != int(layers) or len(set(assignments)) != len(assignments):
        raise ValueError("layer_indices must uniquely assign every batch unit")
    if any(index not in {0, 1, 2} for index in assignments):
        raise ValueError("layer_indices must contain only 0, 1 or 2")
    identifier = batch_id or f"BATCH-{selected_spec.preset}-{uuid4().hex[:8]}"
    products = [
        create_product_state(
            selected_spec,
            order_id=f"{identifier}-U{index + 1:02d}",
            created_at=created_at,
        )
        for index in range(int(layers))
    ]
    units = [
        TrayUnitState(
            unit_id=f"tray_{index + 1:02d}",
            layer_index=assignments[index],
            product=product,
        )
        for index, product in enumerate(products)
    ]
    rack = RackState(
        shelves=[
            RackShelf(index=0, height_m=0.0),
            RackShelf(index=1, height_m=0.151),
            RackShelf(index=2, height_m=0.302),
        ]
    )
    return BatchState(
        batch_id=identifier,
        preset=selected_spec.preset,
        units=units,
        rack=rack,
        created_at=created_at,
    )


build_product_state = create_product_state
