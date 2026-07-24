"""Authoritative shallow-U workcell layout shared by planning and simulation.

The MJCF owns the rigid geometry, while this module owns the coordinates used
to generate order-dependent material poses and to validate clearance.  Keeping
the values named here prevents a product change from silently restoring an
obsolete Table1/turntable coordinate in the Python layer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ShallowULayout:
    """Compact, reachable layout for the four-station one-way process flow."""

    station_s1_xy: tuple[float, float] = (-0.48, 0.00)
    # The document's compact candidate coordinates made 490 x 350 mm station
    # tops physically overlap.  Moving the rear pair outward/upward preserves
    # the shallow-U reach envelope while leaving a real 50+ mm aisle.
    station_s2a_xy: tuple[float, float] = (-0.30, 0.40)
    station_s2b_xy: tuple[float, float] = (0.30, 0.40)
    station_s3_xy: tuple[float, float] = (0.48, 0.00)
    rack_infeed_xy: tuple[float, float] = (0.75, 0.00)

    base_magazine_xy: tuple[float, float] = (-0.60, -0.42)
    fin_magazine_xy: tuple[float, float] = (0.32, -0.46)
    fin_pickup_x: float = 0.32
    fin_pickup_surface_z_m: float = 0.25
    fin_row_spacing_m: float = 0.075
    fin_rows_per_tier: int = 6
    fin_tier_spacing_m: float = 0.10

    output_lane_x: float = 0.75
    # The rigid output pallet is wider than the black belt.  Its swept edge,
    # rather than the decorative belt edge, is the safety-critical boundary.
    output_pallet_half_width_m: float = 0.205
    raw_material_clearance_m: float = 0.040

    def raw_fin_position(
        self,
        index: int,
        count: int,
        *,
        table_top_z: float,
        fin_height_m: float,
    ) -> tuple[float, float, float]:
        """Return a stable two-tier raw-fin pose for a zero-based pool slot.

        At most six blanks occupy one tier.  A seventh or later blank is
        placed on the upper indexed shelf instead of spreading into the
        finished-product conveyor.  Rows remain centred and retain the 75 mm
        gripper clearance used by the verified Arm1 pickup path.
        """

        if index < 0 or count < 1 or index >= count:
            raise ValueError("raw-fin index must belong to the active order")
        tier = index // self.fin_rows_per_tier
        row = index % self.fin_rows_per_tier
        rows = min(self.fin_rows_per_tier, count - tier * self.fin_rows_per_tier)
        y = self.fin_magazine_xy[1] + (row - 0.5 * (rows - 1)) * self.fin_row_spacing_m
        support_z = max(float(table_top_z), self.fin_pickup_surface_z_m)
        z = support_z + 0.5 * fin_height_m + tier * self.fin_tier_spacing_m
        return self.fin_pickup_x, y, z


SHALLOW_U_LAYOUT = ShallowULayout()


__all__ = ["SHALLOW_U_LAYOUT", "ShallowULayout"]
