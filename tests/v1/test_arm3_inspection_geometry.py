"""Arm3 reach and image-framing contracts for every product inspection."""

from __future__ import annotations

import math

import numpy as np
import pytest

from brazing_sim.config import make_order_spec
from brazing_sim.inspection import (
    ARM3_CAMERA_FRAME_MARGIN,
    ARM3_CAMERA_TO_TCP_M,
    inspection_frame_fill,
    inspection_standoff_m,
    top_down_inspection_pose,
)
from brazing_sim.scene import ARM3_WAIT_QPOS, BrazingScene


@pytest.mark.parametrize("preset", ("A", "B", "C"))
def test_all_arm3_views_contain_the_complete_axis_aligned_product(preset: str) -> None:
    spec = make_order_spec(preset)
    standoff = inspection_standoff_m(spec.base_length, spec.base_width)
    long_fill, short_fill = inspection_frame_fill(spec.base_length, spec.base_width, standoff)

    assert long_fill <= 1.0 / ARM3_CAMERA_FRAME_MARGIN
    assert short_fill <= 1.0 / ARM3_CAMERA_FRAME_MARGIN

    pose = top_down_inspection_pose(
        [0.48, 0.0, 0.305],
        product_length_m=spec.base_length,
        product_width_m=spec.base_width,
    )
    np.testing.assert_allclose(pose.rotation[:, 2], [0.0, 0.0, -1.0], atol=1.0e-10)
    np.testing.assert_allclose(pose.rotation[:, 0], [1.0, 0.0, 0.0], atol=1.0e-10)
    assert pose.position[2] == pytest.approx(0.305 + standoff - ARM3_CAMERA_TO_TCP_M)


def test_relocated_arm3_reaches_material_fin_and_finished_product_views() -> None:
    scene = BrazingScene("scenes/production/brazing_line.xml", order="B", raw=True)
    try:
        base = scene.data.body("arm3_base")
        np.testing.assert_allclose(base.xpos, [0.75, 0.40, 0.0], atol=1.0e-9)

        controller = scene.arms["arm3"]
        flange = scene.registry.site_pose("arm3_attachment_site")
        camera_tcp = scene.registry.site_pose("arm3_inspection_tcp")
        controller.set_tool_transform(flange.inverse().transformed(camera_tcp))
        spec = make_order_spec("B")
        targets = (
            ("s2b_target_site", 0.032 + 0.5 * spec.base_thickness),
            ("s3_target_site", 0.032 + 0.5 * spec.base_thickness + spec.fin_height),
            ("batch_output_slot_01_site", 0.032 + 0.5 * spec.base_thickness + spec.fin_height),
        )
        seed = ARM3_WAIT_QPOS.copy()
        for site_name, local_surface_z in targets:
            anchor = scene.registry.site_pose(site_name)
            target = top_down_inspection_pose(
                anchor.position + np.asarray([0.0, 0.0, local_surface_z]),
                product_length_m=spec.base_length,
                product_width_m=spec.base_width,
            )
            result = controller.solve_ik(
                target,
                tcp=True,
                seed=seed,
                full_orientation=True,
                position_tolerance_m=0.0015,
                orientation_tolerance_rad=math.radians(1.0),
                max_iterations=900,
            )
            assert result.reachable, (
                f"{site_name}: {result.position_error_m * 1000.0:.2f} mm / "
                f"{math.degrees(result.orientation_error_rad):.2f} deg"
            )
            seed = result.joint_positions
    finally:
        scene.close()
