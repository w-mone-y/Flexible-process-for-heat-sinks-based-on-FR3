"""Shared camera geometry for every Arm3 inspection route.

The wrist camera looks along the inspection TCP's local ``+Z`` axis.  A
top-down pose therefore keeps local ``+Z`` vertical and points it toward the
product.  Keeping the product long axis on camera-local ``X`` makes the
heatsink rectangle parallel to the 4:3 image frame instead of appearing
diagonally or partly cropped.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .motion import Pose, matrix_to_quat

ARM3_CAMERA_FOVY_DEG = 52.0
ARM3_CAMERA_ASPECT = 4.0 / 3.0
ARM3_CAMERA_FRAME_MARGIN = 1.15
ARM3_CAMERA_MIN_STANDOFF_M = 0.34
# The optical centre is 10 mm behind the inspection TCP along the viewing
# axis (camera z=55 mm, TCP z=65 mm in the wrist rig).
ARM3_CAMERA_TO_TCP_M = 0.010


def inspection_standoff_m(
    product_length_m: float,
    product_width_m: float,
    *,
    fovy_deg: float = ARM3_CAMERA_FOVY_DEG,
    aspect: float = ARM3_CAMERA_ASPECT,
    frame_margin: float = ARM3_CAMERA_FRAME_MARGIN,
    minimum_m: float = ARM3_CAMERA_MIN_STANDOFF_M,
) -> float:
    """Return the optical standoff required to contain a product rectangle."""

    length = float(product_length_m)
    width = float(product_width_m)
    vertical_fov = math.radians(float(fovy_deg))
    ratio = float(aspect)
    margin = float(frame_margin)
    if length <= 0.0 or width <= 0.0:
        raise ValueError("inspection footprint dimensions must be positive")
    if not 0.0 < vertical_fov < math.pi or ratio <= 0.0 or margin < 1.0:
        raise ValueError("invalid inspection camera geometry")
    vertical_tangent = math.tan(0.5 * vertical_fov)
    horizontal_tangent = ratio * vertical_tangent
    required_for_length = 0.5 * margin * length / horizontal_tangent
    required_for_width = 0.5 * margin * width / vertical_tangent
    return max(float(minimum_m), required_for_length, required_for_width)


def top_down_inspection_pose(
    surface_center: Sequence[float],
    *,
    product_length_m: float,
    product_width_m: float,
    product_yaw_rad: float = 0.0,
) -> Pose:
    """Return an Arm3 TCP pose whose image frame encloses the whole product."""

    center = np.asarray(surface_center, dtype=float)
    if center.shape != (3,):
        raise ValueError("inspection surface centre must contain xyz")
    yaw = float(product_yaw_rad)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray(
        [
            [cosine, sine, 0.0],
            [sine, -cosine, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=float,
    )
    viewing_axis = rotation[:, 2]
    standoff = inspection_standoff_m(product_length_m, product_width_m)
    optical_center = center - viewing_axis * standoff
    tcp_position = optical_center + viewing_axis * ARM3_CAMERA_TO_TCP_M
    return Pose(tcp_position, matrix_to_quat(rotation))


def inspection_frame_fill(
    product_length_m: float,
    product_width_m: float,
    standoff_m: float,
    *,
    fovy_deg: float = ARM3_CAMERA_FOVY_DEG,
    aspect: float = ARM3_CAMERA_ASPECT,
) -> tuple[float, float]:
    """Return long/short footprint fractions of the available image frame."""

    vertical_half = float(standoff_m) * math.tan(0.5 * math.radians(float(fovy_deg)))
    horizontal_half = float(aspect) * vertical_half
    return (
        0.5 * float(product_length_m) / horizontal_half,
        0.5 * float(product_width_m) / vertical_half,
    )


__all__ = [
    "ARM3_CAMERA_ASPECT",
    "ARM3_CAMERA_FOVY_DEG",
    "ARM3_CAMERA_FRAME_MARGIN",
    "ARM3_CAMERA_MIN_STANDOFF_M",
    "ARM3_CAMERA_TO_TCP_M",
    "inspection_frame_fill",
    "inspection_standoff_m",
    "top_down_inspection_pose",
]
