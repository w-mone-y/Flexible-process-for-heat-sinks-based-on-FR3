from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from brazing_sim.config import create_product_state, make_order_spec
from brazing_sim.fault_catalog import MANUAL_FAULT_CATALOG
from brazing_sim.fault_visuals import PhysicalFaultVisualizer

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def physical_scene():
    from brazing_sim.scene import BrazingScene

    scene = BrazingScene(ROOT / "scenes" / "production" / "brazing_line.xml", order="A", raw=False)
    product = create_product_state(make_order_spec("A"))
    scene.reset(product, raw=False)
    visualizer = PhysicalFaultVisualizer(scene)
    try:
        yield scene, product, visualizer
    finally:
        scene.close()


def test_every_manual_fault_has_a_mujoco_visual_strategy() -> None:
    assert PhysicalFaultVisualizer.SUPPORTED_FAULT_TYPES == frozenset(MANUAL_FAULT_CATALOG)


def test_missing_material_is_two_real_segments_with_a_central_blank(physical_scene) -> None:
    scene, product, visualizer = physical_scene
    path = product.active_paths[0]
    path.applied = True
    path.coverage_ratio = 0.70
    path.longest_gap_m = 0.020
    scene.registry.set_path_visible(path.path_id, True, coverage=1.0)

    visualizer.sync_quality(product)

    original = scene.model.geom(path.name + "_geom")
    left = scene.model.geom("fault_braze_segment_01a")
    right = scene.model.geom("fault_braze_segment_01b")
    assert float(original.rgba[3]) == 0.0
    assert float(left.rgba[3]) > 0.9
    assert float(right.rgba[3]) > 0.9
    left_end = float(left.pos[0] + left.size[1])
    right_start = float(right.pos[0] - right.size[1])
    assert right_start - left_end == pytest.approx(0.020, abs=1.0e-6)

    path.coverage_ratio = 1.0
    path.longest_gap_m = 0.0
    visualizer.sync_quality(product)
    assert float(original.rgba[3]) > 0.0
    assert float(left.rgba[3]) == 0.0
    assert float(right.rgba[3]) == 0.0


def test_deviated_material_shows_actual_and_nominal_lines(physical_scene) -> None:
    scene, product, visualizer = physical_scene
    path = product.active_paths[0]
    path.applied = True
    path.coverage_ratio = 1.0
    path.lateral_error_m = 0.005

    visualizer.sync_quality(product)

    actual = scene.model.geom(path.name + "_geom")
    nominal = scene.model.geom("fault_braze_segment_01a")
    assert float(actual.pos[1]) == pytest.approx(0.005)
    assert float(actual.rgba[3]) > 0.0
    assert np.allclose(nominal.rgba[:3], [1.0, 0.03, 0.02])
    assert float(nominal.rgba[3]) > 0.0


@pytest.mark.parametrize(
    ("fault_type", "target", "geom_name"),
    [
        ("ARM_UNAVAILABLE", "ARM1", "arm1_target_marker"),
        ("RACK_LAYER_UNAVAILABLE", "1", "batch_rack_1_left_rail"),
        ("ELEVATOR_TIMEOUT", "", "conveyor_belt"),
        ("FORK_TIMEOUT", "", "conveyor_belt"),
        ("FURNACE_DOOR_INTERLOCK", "", "furnace_door_outer_skin"),
        ("TRAY_STATE_INCONSISTENT", "", "fixture_tray_geom"),
    ],
)
def test_equipment_fault_colours_real_hardware_and_recovery_restores_it(
    physical_scene,
    fault_type: str,
    target: str,
    geom_name: str,
) -> None:
    scene, _product, visualizer = physical_scene
    geom = scene.model.geom(geom_name)
    original_material = int(geom.matid[0])
    original_rgba = np.asarray(geom.rgba, dtype=float).copy()
    hold = {
        "request_id": "PHYSICAL_TEST",
        "fault_type": fault_type,
        "target": target,
        "status": "ACTIVE",
    }

    visualizer.sync_equipment([hold], now=1.0, active_actor="arm2")
    assert int(geom.matid[0]) == -1
    assert np.allclose(geom.rgba[:3], PhysicalFaultVisualizer.ACTIVE_RED) or np.allclose(
        geom.rgba[:3], PhysicalFaultVisualizer.SAFETY_MAGENTA
    )

    visualizer.sync_equipment([], now=2.0)
    assert int(geom.matid[0]) == original_material
    assert np.allclose(geom.rgba, original_rgba)


def test_contact_and_tray_safety_faults_show_world_space_overlays(physical_scene) -> None:
    scene, _product, visualizer = physical_scene
    visualizer.sync_equipment(
        [
            {
                "request_id": "CONTACT",
                "fault_type": "CONTACT_SAFETY_STOP",
                "target": "",
                "status": "ACTIVE",
                "visual_target": "arm2",
            },
            {
                "request_id": "TRAY",
                "fault_type": "TRAY_STATE_INCONSISTENT",
                "target": "",
                "status": "ACTIVE",
            },
        ],
        now=1.0,
        active_actor="arm2",
    )
    assert float(scene.model.geom("fault_contact_marker_geom").rgba[3]) > 0.0
    assert float(scene.model.geom("fault_tray_ghost_geom").rgba[3]) > 0.0

    visualizer.sync_equipment([], now=2.0)
    assert float(scene.model.geom("fault_contact_marker_geom").rgba[3]) == 0.0
    assert float(scene.model.geom("fault_tray_ghost_geom").rgba[3]) == 0.0
