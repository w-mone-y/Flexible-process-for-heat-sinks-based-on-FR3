"""V2 fault visuals: every fault must resolve to real scene geometry.

The point of these tests is that a *name* table cannot be trusted.  V1's
visualizer tints bodies like ``furnace_heater_left_mid``; of thirteen such names
probed against ``brazing_line_v2.xml`` only one had a counterpart.  A selector
that matches nothing fails silently — the operator sees no colour change and
concludes the fault did not fire.  So the contract asserted here is:

*   every fault type in the table resolves to at least one geom;
*   tinting is reversible, restoring the authored material and rgba exactly;
*   the scene follows runtime truth, including recovery clearing the tint.
"""

from __future__ import annotations

import numpy as np
import pytest

from brazing_sim.dual_line.fault_visuals import _FAULT_GEOMETRY, V2FaultVisualizer
from brazing_sim.paths import PRODUCTION_SCENES_DIR

mujoco = pytest.importorskip("mujoco")

V2_XML = PRODUCTION_SCENES_DIR / "brazing_line_v2.xml"


class _StubAdapter:
    """Minimal stand-in exposing the three attributes the visualizer needs."""

    def __init__(self) -> None:
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(V2_XML))
        self.data = mujoco.MjData(self.model)


@pytest.fixture(scope="module")
def visualizer():
    return V2FaultVisualizer(_StubAdapter())


def _target_for(fault_type: str) -> str:
    if "ARM" in fault_type or fault_type == "CONTACT_SAFETY_STOP":
        return "ARM2"
    if "RACK_LAYER" in fault_type:
        return "1"
    return ""


@pytest.mark.parametrize("fault_type", sorted(_FAULT_GEOMETRY))
def test_every_fault_resolves_to_real_geometry(fault_type, visualizer):
    """A selector matching nothing would fail silently in the viewer."""

    selectors = visualizer._selectors(fault_type, _target_for(fault_type), "V2_TRAY_01")
    assert selectors, f"{fault_type} 没有几何选择器"
    geoms = visualizer._matching_geoms(**selectors)
    assert geoms, f"{fault_type} 的几何名在 V2 场景中零命中"


def test_tray_scoped_selectors_need_a_tray():
    """Without a tray id a tray-scoped selector would tint every pallet."""

    visualizer = V2FaultVisualizer(_StubAdapter())
    scoped = visualizer._selectors("BRAZING_MISSING", "", None)
    assert scoped == {}
    targeted = visualizer._selectors("BRAZING_MISSING", "", "V2_TRAY_03")
    assert targeted and "v2_tray_03_" in targeted["prefixes"][0]


def test_tint_is_fully_reversible():
    adapter = _StubAdapter()
    visualizer = V2FaultVisualizer(adapter)
    faults = [
        {
            "fault_type": "ARM_UNAVAILABLE",
            "recovered": False,
            "source": "ARM2",
            "details": {"target": "ARM2", "severity": "recoverable"},
        }
    ]
    selectors = visualizer._selectors("ARM_UNAVAILABLE", "ARM2", None)
    geoms = visualizer._matching_geoms(**selectors)
    before_rgba = adapter.model.geom_rgba[list(geoms)].copy()
    before_matid = adapter.model.geom_matid[list(geoms)].copy()

    visualizer.sync(faults, [])
    assert visualizer.active_effects
    assert not np.allclose(adapter.model.geom_rgba[list(geoms)], before_rgba)

    visualizer.reset()
    assert visualizer.active_effects == ()
    assert np.allclose(adapter.model.geom_rgba[list(geoms)], before_rgba)
    assert np.array_equal(adapter.model.geom_matid[list(geoms)], before_matid)


def test_recovered_faults_release_their_tint():
    """The scene follows runtime truth, including recovery."""

    adapter = _StubAdapter()
    visualizer = V2FaultVisualizer(adapter)
    record = {
        "fault_type": "ARM_UNAVAILABLE",
        "recovered": False,
        "source": "ARM2",
        "details": {"target": "ARM2", "severity": "recoverable"},
    }
    visualizer.sync([record], [])
    assert visualizer.active_effects

    record["recovered"] = True
    visualizer.sync([record], [])
    assert visualizer.active_effects == ()


def test_severity_and_safety_select_different_colours():
    adapter = _StubAdapter()
    visualizer = V2FaultVisualizer(adapter)

    def colour_of(fault_type: str, severity: str, target: str) -> np.ndarray:
        visualizer.reset()
        visualizer.sync(
            [
                {
                    "fault_type": fault_type,
                    "recovered": False,
                    "source": target,
                    "details": {"target": target, "severity": severity},
                }
            ],
            [],
        )
        geoms = visualizer._matching_geoms(**visualizer._selectors(fault_type, target, None))
        return adapter.model.geom_rgba[geoms[0], :3].copy()

    recoverable = colour_of("ARM_UNAVAILABLE", "recoverable", "ARM2")
    severe = colour_of("ARM_UNAVAILABLE", "severe", "ARM2")
    safety = colour_of("CONTACT_SAFETY_STOP", "severe", "ARM2")
    assert np.allclose(recoverable, V2FaultVisualizer.RECOVERING_AMBER)
    assert np.allclose(severe, V2FaultVisualizer.ACTIVE_RED)
    assert np.allclose(safety, V2FaultVisualizer.SAFETY_MAGENTA)


def test_unknown_fault_types_are_ignored_without_error():
    visualizer = V2FaultVisualizer(_StubAdapter())
    visualizer.sync([{"fault_type": "NOT_A_FAULT", "recovered": False, "details": {}}], [])
    assert visualizer.active_effects == ()


@pytest.mark.parametrize("visual_type", ("FIN_POSE", "FIN_GEOMETRY_FAILED"))
def test_latent_fin_pose_fault_is_lateral_only_before_detection(visual_type: str):
    adapter = _StubAdapter()
    visualizer = V2FaultVisualizer(adapter)
    target = int(adapter.model.geom("v2_tray_01_fin_03").id)
    neighbour = int(adapter.model.geom("v2_tray_01_fin_02").id)
    target_before = adapter.model.geom_pos[target].copy()
    target_quaternion_before = adapter.model.geom_quat[target].copy()
    neighbour_before = adapter.model.geom_pos[neighbour].copy()

    visualizer.sync(
        [],
        [{"unit_id": "U1", "tray_id": "V2_TRAY_01"}],
        [
            {
                "defect_id": "D1",
                "unit_id": "U1",
                "visual_type": visual_type,
                "fault_type": "FIN_GEOMETRY_FAILED",
                "target": "fin_03",
                "status": "MANIFESTED",
            }
        ],
    )
    displacement = adapter.model.geom_pos[target] - target_before
    assert displacement[0] == pytest.approx(0.0)
    assert abs(float(displacement[1])) >= 0.010
    assert displacement[2] == pytest.approx(0.0)
    assert np.allclose(adapter.model.geom_quat[target], target_quaternion_before)
    assert np.allclose(adapter.model.geom_pos[neighbour], neighbour_before)

    visualizer.sync([], [{"unit_id": "U1", "tray_id": "V2_TRAY_01"}], [])
    assert np.allclose(adapter.model.geom_pos[target], target_before)


def test_latent_missing_braze_shortens_the_exact_path_only():
    adapter = _StubAdapter()
    visualizer = V2FaultVisualizer(adapter)
    target = int(adapter.model.geom("v2_tray_01_braze_02").id)
    neighbour = int(adapter.model.geom("v2_tray_01_braze_01").id)
    target_before = adapter.model.geom_size[target].copy()
    neighbour_before = adapter.model.geom_size[neighbour].copy()

    visualizer.sync(
        [],
        [{"unit_id": "U1", "tray_id": "V2_TRAY_01"}],
        [
            {
                "defect_id": "D2",
                "unit_id": "U1",
                "visual_type": "BRAZING_MISSING",
                "fault_type": "BRAZING_MISSING",
                "target": "path_02",
                "status": "MANIFESTED",
            }
        ],
    )
    assert adapter.model.geom_size[target, 1] < target_before[1] * 0.5
    assert np.allclose(adapter.model.geom_size[neighbour], neighbour_before)


def test_local_braze_repair_fills_only_the_missing_target_progressively():
    adapter = _StubAdapter()
    visualizer = V2FaultVisualizer(adapter)
    target = int(adapter.model.geom("v2_tray_01_braze_02").id)
    neighbour = int(adapter.model.geom("v2_tray_01_braze_01").id)
    full_target = adapter.model.geom_size[target].copy()
    full_neighbour = adapter.model.geom_size[neighbour].copy()
    defect = {
        "defect_id": "D3",
        "unit_id": "U1",
        "visual_type": "BRAZING_MISSING",
        "fault_type": "BRAZING_MISSING",
        "target": "path_02",
        "status": "DETECTED",
    }

    visualizer.sync([], [{"unit_id": "U1", "tray_id": "V2_TRAY_01"}], [defect])
    missing_size = adapter.model.geom_size[target, 1]
    defect["repair_progress"] = 0.5
    visualizer.sync([], [{"unit_id": "U1", "tray_id": "V2_TRAY_01"}], [defect])

    assert missing_size < adapter.model.geom_size[target, 1] < full_target[1]
    assert np.allclose(adapter.model.geom_size[neighbour], full_neighbour)


def test_deviated_braze_is_removed_end_to_end_before_nominal_reapplication() -> None:
    adapter = _StubAdapter()
    visualizer = V2FaultVisualizer(adapter)
    target = int(adapter.model.geom("v2_tray_01_braze_03").id)
    neighbour = int(adapter.model.geom("v2_tray_01_braze_02").id)
    home_position = adapter.model.geom_pos[target].copy()
    full_size = adapter.model.geom_size[target].copy()
    neighbour_position = adapter.model.geom_pos[neighbour].copy()
    neighbour_size = adapter.model.geom_size[neighbour].copy()
    defect = {
        "defect_id": "D_DEVIATION",
        "unit_id": "U1",
        "visual_type": "BRAZING_PATH_DEVIATION",
        "fault_type": "BRAZING_PATH_DEVIATION",
        "target": "path_03",
        "status": "DETECTED",
        "removal_progress": 0.5,
        "reapply_progress": 0.0,
    }
    units = [{"unit_id": "U1", "tray_id": "V2_TRAY_01"}]

    visualizer.sync([], units, [defect])
    assert 0.0 < adapter.model.geom_size[target, 1] < full_size[1]
    assert adapter.model.geom_pos[target, 1] == pytest.approx(home_position[1] + 0.012)
    assert np.allclose(adapter.model.geom_pos[neighbour], neighbour_position)
    assert np.allclose(adapter.model.geom_size[neighbour], neighbour_size)

    defect["removal_progress"] = 1.0
    defect["reapply_progress"] = 0.5
    visualizer.sync([], units, [defect])
    assert adapter.model.geom_pos[target, 1] == pytest.approx(home_position[1])
    assert adapter.model.geom_size[target, 1] == pytest.approx(0.5 * full_size[1])

    defect["reapply_progress"] = 1.0
    visualizer.sync([], units, [defect])
    assert np.allclose(adapter.model.geom_pos[target], home_position)
    assert np.allclose(adapter.model.geom_size[target], full_size)


def test_fault_geometry_does_not_accumulate_across_repeated_ui_frames():
    adapter = _StubAdapter()
    visualizer = V2FaultVisualizer(adapter)
    target = int(adapter.model.geom("v2_tray_01_braze_02").id)
    defect = {
        "defect_id": "D4",
        "unit_id": "U1",
        "visual_type": "BRAZING_MISSING",
        "fault_type": "BRAZING_MISSING",
        "target": "path_02",
        "status": "MANIFESTED",
    }
    units = [{"unit_id": "U1", "tray_id": "V2_TRAY_01"}]

    visualizer.sync([], units, [defect])
    first_size = adapter.model.geom_size[target].copy()
    first_position = adapter.model.geom_pos[target].copy()
    for _ in range(20):
        visualizer.sync([], units, [defect])

    assert np.allclose(adapter.model.geom_size[target], first_size)
    assert np.allclose(adapter.model.geom_pos[target], first_position)


def test_fin_pose_reinstall_progress_corrects_only_the_failed_fin():
    adapter = _StubAdapter()
    visualizer = V2FaultVisualizer(adapter)
    target = int(adapter.model.geom("v2_tray_01_fin_03").id)
    neighbour = int(adapter.model.geom("v2_tray_01_fin_02").id)
    target_home = adapter.model.geom_pos[target].copy()
    neighbour_home = adapter.model.geom_pos[neighbour].copy()
    defect = {
        "defect_id": "D5",
        "unit_id": "U1",
        "visual_type": "FIN_POSE",
        "fault_type": "FIN_GEOMETRY_FAILED",
        "target": "fin_03",
        "status": "DETECTED",
        "repair_progress": 0.5,
    }

    visualizer.sync([], [{"unit_id": "U1", "tray_id": "V2_TRAY_01"}], [defect])

    assert 0.0 < np.linalg.norm(adapter.model.geom_pos[target] - target_home) < 0.03
    assert np.allclose(adapter.model.geom_pos[neighbour], neighbour_home)
