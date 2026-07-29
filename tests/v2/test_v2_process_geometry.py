from __future__ import annotations

import numpy as np
import pytest

from brazing_sim.dual_line import V2ProcessGeometry


@pytest.mark.parametrize(
    ("preset", "fin_count", "path_length", "pitch", "nozzle_spacing"),
    (
        ("A", 5, 0.33, 0.020, 0.0050),
        ("B", 4, 0.33, 0.030, 0.0050),
        ("C", 7, 0.31, 0.015, 0.0044),
    ),
)
def test_v2_process_geometry_reuses_the_v1_product_plan(
    preset: str,
    fin_count: int,
    path_length: float,
    pitch: float,
    nozzle_spacing: float,
) -> None:
    geometry = V2ProcessGeometry.for_preset(preset)

    assert len(geometry.fin_targets) == fin_count
    assert len(geometry.dispense_passes) == fin_count
    assert geometry.path_length_m == pytest.approx(path_length)
    assert geometry.fin_pitch_m == pytest.approx(pitch)
    assert geometry.nozzle_spacing_m == pytest.approx(nozzle_spacing)
    for dispense_pass in geometry.dispense_passes:
        assert dispense_pass.path_ids[0] != dispense_pass.path_ids[1]
        assert np.linalg.norm(dispense_pass.end - dispense_pass.start) == pytest.approx(path_length)


def test_v2_process_targets_are_transformed_from_the_live_tray_frame() -> None:
    geometry = V2ProcessGeometry.for_preset("C")
    origin = np.asarray([0.55, -0.65, 0.225])
    rotation = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    target = geometry.world_fin_target(0, origin=origin, rotation=rotation)
    expected = origin + rotation @ geometry.fin_targets[0]
    np.testing.assert_allclose(target, expected, atol=1.0e-12)

    dispense_pass = geometry.world_dispense_pass(0, origin=origin, rotation=rotation)
    np.testing.assert_allclose(
        dispense_pass.end - dispense_pass.start,
        rotation @ np.asarray([geometry.path_length_m, 0.0, 0.0]),
        atol=1.0e-12,
    )
