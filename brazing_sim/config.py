"""Order presets and product-coordinate layout derivation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import cycle
from typing import Any, Iterator
from uuid import uuid4

from .domain import BrazingPathState, BrazingSide, FinState, OrderSpec, ProductState, Vec3

PRODUCT_AXES = {
    "x": "fin_length",
    "y": "fin_array",
    "z": "fin_height",
}


A_ORDER_SPEC = OrderSpec(
    preset="A",
    base_size=(0.36, 0.22, 0.008),
    fin_size=(0.30, 0.002, 0.06),
    fin_count=4,
    fin_pitch=0.06,
    brazing_sides=(BrazingSide.LEFT, BrazingSide.RIGHT),
    path_width=0.004,
    max_fins=8,
    max_paths=16,
)

ORDER_PRESETS: dict[str, OrderSpec] = {"A": A_ORDER_SPEC}


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
    return replace(base, **overrides)


get_order_spec = make_order_spec
preset_a = make_order_spec


def _fin_y_positions(fin_count: int, pitch: float) -> tuple[float, ...]:
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

    ys = _fin_y_positions(spec.fin_count, spec.fin_pitch)
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
            y = fin.target_position[1] + sign * (spec.fin_thickness + spec.path_width) / 2.0
            x_limit = spec.fin_length / 2.0
            start = (-x_limit, y, base_top_z)
            end = (x_limit, y, base_top_z)
            fin_id = fin.fin_id
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
        paths.append(
            BrazingPathState(
                path_id=f"{fin_id}_{side.value}",
                index=index,
                fin_id=fin_id,
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
        order_id=order_id or f"A-{uuid4().hex[:8]}",
        spec=selected,
        fins=layout.fins,
        paths=layout.paths,
        created_at=created_at,
    )


build_product_state = create_product_state
