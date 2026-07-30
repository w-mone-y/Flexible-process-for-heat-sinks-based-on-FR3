from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pytest

from brazing_sim.dual_line import (
    DualLineRuntime,
    DualLineSceneAdapter,
    TrayOwner,
    V2ProcessGeometry,
)

ROOT = Path(__file__).resolve().parents[2]
V2_XML = ROOT / "scenes" / "production" / "brazing_line_v2.xml"


def test_v2_s3_s4_waypoints_are_planar_north_and_south_bypasses() -> None:
    """The runtime route must match the visible rails and never lift."""

    install_a = np.asarray([0.55, 0.50, 0.225])
    wait_a = np.asarray([1.40, 0.50, 0.225])
    install_b = np.asarray([0.35, -0.45, 0.225])
    wait_b = np.asarray([1.55, -1.22, 0.225])
    merge = np.asarray([1.40, 0.00, 0.225])

    np.testing.assert_allclose(
        DualLineSceneAdapter._route_targets(
            install_a,
            wait_a,
            source_owner=TrayOwner.INSTALL_A,
            target_owner=TrayOwner.MERGE_A_WAIT,
        ),
        (wait_a,),
        atol=1.0e-9,
    )
    np.testing.assert_allclose(
        DualLineSceneAdapter._route_targets(
            install_b,
            wait_b,
            source_owner=TrayOwner.INSTALL_B,
            target_owner=TrayOwner.MERGE_B_WAIT,
        ),
        (
            (0.05, -0.45, 0.225),
            (0.05, -1.22, 0.225),
            (1.55, -1.22, 0.225),
        ),
        atol=1.0e-9,
    )
    np.testing.assert_allclose(
        DualLineSceneAdapter._route_targets(
            wait_a,
            merge,
            source_owner=TrayOwner.MERGE_A_WAIT,
            target_owner=TrayOwner.MERGE,
        ),
        (merge,),
        atol=1.0e-9,
    )
    np.testing.assert_allclose(
        DualLineSceneAdapter._route_targets(
            wait_b,
            merge,
            source_owner=TrayOwner.MERGE_B_WAIT,
            target_owner=TrayOwner.MERGE,
        ),
        (merge,),
        atol=1.0e-9,
    )


def test_scene_adapter_moves_an_active_tray_continuously_without_changing_orientation() -> None:
    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML, transfer_speed_m_s=0.5)
    try:
        runtime.submit_order("A", order_id="VISIBLE_A")
        adapter.sync(runtime)
        tray = runtime.flow.trays[0]
        start = adapter.tray_position(tray.tray_id).copy()
        start_quat = adapter.tray_quaternion(tray.tray_id).copy()

        # Match the viewer loop: physical time advances before the next scene
        # synchronization evaluates the carrier's time-scaled route.
        adapter.step_physics(0.10)
        runtime.tick(0.10)
        adapter.sync(runtime)
        intermediate = adapter.tray_position(tray.tray_id).copy()
        target = np.asarray(runtime.topology.station("S1_BASE_LOADING").world_xyz)

        assert np.linalg.norm(intermediate - start) > 0.0
        assert np.linalg.norm(intermediate - target) > 0.05
        orientation_dot = abs(float(np.dot(adapter.tray_quaternion(tray.tray_id), start_quat)))
        orientation_error_rad = 2.0 * np.arccos(np.clip(orientation_dot, -1.0, 1.0))
        assert orientation_error_rad < np.deg2rad(0.01)
        assert adapter.tray_visible(tray.tray_id)
    finally:
        adapter.close()


def test_v2_default_slide_rail_speed_is_deliberate_at_one_times() -> None:
    """The viewer default must show transport, not fire a pallet across a station."""

    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=False)
    adapter = DualLineSceneAdapter(V2_XML)
    try:
        assert adapter.transfer_speed_m_s == pytest.approx(0.35)
        runtime.submit_order("A", order_id="RAIL_SPEED_A")
        adapter.sync(runtime)
        runtime.tick(0.05)
        adapter.step_physics(0.05)
        adapter.sync(runtime)
        motion = adapter.transport_snapshot()["V2_TRAY_01"]
        assert float(motion["duration_s"]) >= float(motion["distance_m"]) / 0.35 - 1.0e-9
    finally:
        adapter.close()


def test_scene_adapter_hides_only_empty_or_virtual_return_trays() -> None:
    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    try:
        adapter.sync(runtime)
        assert not any(adapter.tray_visible(tray.tray_id) for tray in runtime.flow.trays)
        runtime.submit_order("A", order_id="VISIBLE_A")
        adapter.sync(runtime)
        active = runtime.flow.trays[0]
        assert active.owner is TrayOwner.S1
        assert adapter.tray_visible(active.tray_id)
    finally:
        adapter.close()


def test_scene_adapter_reveals_each_tray_payload_only_at_real_process_boundaries() -> None:
    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    try:
        runtime.submit_order("A", order_id="PAYLOAD_A")
        adapter.sync(runtime)
        tray_id = runtime.flow.trays[0].tray_id
        assert adapter.component_visible(tray_id, "template_plate")
        assert not adapter.component_visible(tray_id, "base_plate")
        assert not adapter.component_visible(tray_id, "braze_01")
        assert not adapter.component_visible(tray_id, "fin_01")

        for _ in range(300):
            runtime.tick(0.05)
            adapter.sync(runtime)
            adapter.step_physics(0.05)
            if adapter.component_visible(tray_id, "base_plate"):
                break
        assert adapter.component_visible(tray_id, "base_plate")
        assert not adapter.component_visible(tray_id, "braze_01")

        for _ in range(600):
            runtime.tick(0.05)
            adapter.sync(runtime)
            adapter.step_physics(0.05)
            if adapter.component_visible(tray_id, "braze_01"):
                break
        assert adapter.component_visible(tray_id, "braze_01")
        assert not adapter.component_visible(tray_id, "fin_01")
    finally:
        adapter.close()


def test_reused_tray_uses_the_new_unit_visual_state_not_completed_history() -> None:
    """A virtually returned tray must start the next order physically empty."""

    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    try:
        runtime.submit_order("A", order_id="HISTORY_FIRST")
        for _ in range(6_000):
            runtime.tick(0.05)
            adapter.sync(runtime)
            adapter.step_physics(0.05)
            if runtime.complete and adapter.transport_settled:
                break
        assert runtime.complete

        runtime.submit_order("B", order_id="HISTORY_SECOND")
        runtime.tick(0.01)
        adapter.sync(runtime)
        adapter.step_physics(0.01)
        unit = runtime.units["HISTORY_SECOND_UNIT_01"]
        assert unit.stage.value == "BASE_LOADING"
        assert unit.tray_id is not None
        assert adapter.component_visible(unit.tray_id, "template_plate")
        for component in (
            "base_plate",
            "braze_01",
            "fin_01",
            "front_comb_base",
            "front_press",
        ):
            assert not adapter.component_visible(unit.tray_id, component), component
    finally:
        adapter.close()


@pytest.mark.parametrize("preset", ("A", "B", "C"))
def test_v2_tray_owned_base_sits_directly_above_template_without_popup_pads(
    preset: str,
) -> None:
    """Loading a base must not reveal four artificial supports under it."""

    mujoco = pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    try:
        runtime.submit_order(preset, order_id=f"BASE_SUPPORT_{preset}")
        adapter.sync(runtime)
        tray_id = runtime.units[f"BASE_SUPPORT_{preset}_UNIT_01"].tray_id
        assert tray_id is not None
        prefix = tray_id.lower()
        template = adapter.model.geom(f"{prefix}_template_plate")
        base = adapter.model.geom(f"{prefix}_base_plate")
        template_top = float(template.pos[2] + template.size[2])
        base_bottom = float(base.pos[2] - base.size[2])
        assert 0.0005 <= base_bottom - template_top <= 0.0015
        for corner in ("front_left", "front_right", "rear_left", "rear_right"):
            assert (
                mujoco.mj_name2id(
                    adapter.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    f"{prefix}_base_support_{corner}",
                )
                == -1
            )
    finally:
        adapter.close()


def test_v2_inspection_starts_analysis_only_after_aligned_capture_and_holds_five_seconds() -> None:
    """Dispatch time is not capture time; analysis begins only at the certified pose."""

    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    try:
        runtime.submit_order("A", order_id="CAPTURE_GATE_A")
        capture = None
        for _ in range(2_500):
            runtime.tick(0.05)
            adapter.sync(runtime)
            adapter.step_physics(0.05)
            records = adapter.inspection_snapshot()
            capture = next(
                (
                    item
                    for item in records
                    if item["unit_id"] == "CAPTURE_GATE_A_UNIT_01"
                    and item["kind"] == "MATERIAL_INSPECTION"
                    and item["captured"]
                ),
                None,
            )
            if capture is not None:
                break

        assert capture is not None
        assert capture["aligned"]
        assert capture["clear"]
        assert capture["camera"] == "v2_arm3_inspection_camera"
        captured_at = float(capture["captured_at"])

        while runtime.sim_time < captured_at + 4.95:
            runtime.tick(0.05)
            adapter.sync(runtime)
            adapter.step_physics(0.05)
        unit = runtime.units["CAPTURE_GATE_A_UNIT_01"]
        assert unit.stage.value == "MATERIAL_INSPECTION"

        while runtime.sim_time < captured_at + 5.10:
            runtime.tick(0.05)
            adapter.sync(runtime)
            adapter.step_physics(0.05)
        assert unit.stage.value != "MATERIAL_INSPECTION"
        final = next(
            item
            for item in adapter.inspection_snapshot()
            if item["unit_id"] == unit.unit_id and item["kind"] == "MATERIAL_INSPECTION"
        )
        assert final["analysis_complete"]
        assert float(final["analysis_elapsed_s"]) >= 5.0
    finally:
        adapter.close()


@pytest.mark.parametrize("preset", ("A", "B", "C"))
def test_scene_adapter_configures_each_tray_from_the_order_geometry(preset: str) -> None:
    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    try:
        runtime.submit_order(preset, order_id=f"GEOMETRY_{preset}")
        adapter.sync(runtime)
        tray_id = runtime.flow.trays[0].tray_id
        prefix = tray_id.lower()
        geometry = V2ProcessGeometry.for_preset(preset)

        base = adapter.model.geom(f"{prefix}_base_plate")
        np.testing.assert_allclose(base.size[:3], np.asarray(geometry.base_size_m) / 2.0)
        for index, target in enumerate(geometry.fin_targets, start=1):
            fin = adapter.model.geom(f"{prefix}_fin_{index:02d}")
            np.testing.assert_allclose(fin.pos, target, atol=1.0e-12)
            np.testing.assert_allclose(fin.size[:3], np.asarray(geometry.fin_size_m) / 2.0)
        for index, path in enumerate(geometry.brazing_paths, start=1):
            bead = adapter.model.geom(f"{prefix}_braze_{index:02d}")
            assert float(bead.pos[1]) == pytest.approx(path.start[1])
            assert float(bead.size[1]) == pytest.approx(0.5 * path.length_m)
    finally:
        adapter.close()


def test_arm2_dispensing_grows_two_beads_per_pass_monotonically() -> None:
    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    try:
        runtime.submit_order("A", order_id="PROGRESSIVE_BRAZE")
        tray_id = runtime.flow.trays[0].tray_id
        adapter.sync(runtime)
        geometry = V2ProcessGeometry.for_preset("A")
        visible_counts: list[int] = []
        observed_partial_length = False
        measured_deposition_samples = 0
        first_path_id = int(adapter.model.geom(f"{tray_id.lower()}_braze_01").id)
        full_half_length = float(adapter.model.geom_size[first_path_id, 1])

        for _ in range(800):
            runtime.tick(0.05)
            adapter.sync(runtime)
            adapter.step_physics(0.05)
            unit = runtime.units["PROGRESSIVE_BRAZE_UNIT_01"]
            if unit.stage.value == "DISPENSING":
                robot = adapter.robot_motion_snapshot()["arm2"]
                if "连续涂覆" in str(robot["target_zh"]):
                    match = re.search(r"第(\d+)道", str(robot["target_zh"]))
                    assert match is not None
                    pass_index = int(match.group(1)) - 1
                    tray_origin = adapter.tray_position(tray_id)
                    tray_rotation = np.asarray(
                        adapter.data.body(tray_id.lower()).xmat,
                        dtype=float,
                    ).reshape(3, 3)
                    expected_pass = geometry.world_dispense_pass(
                        pass_index,
                        origin=tray_origin,
                        rotation=tray_rotation,
                    )
                    expected_y = float(expected_pass.start[1])
                    actual_tcp = np.asarray(robot["actual_tcp_position_m"])
                    assert abs(float(actual_tcp[1]) - expected_y) <= 0.010
                    target_tcp = np.asarray(robot["target_tcp_position_m"])
                    direction = expected_pass.end - expected_pass.start
                    fraction = float(
                        np.dot(target_tcp - expected_pass.start, direction) / np.dot(direction, direction)
                    )
                    nearest = expected_pass.start + np.clip(fraction, 0.0, 1.0) * direction
                    assert np.linalg.norm(target_tcp - nearest) <= 1.0e-6
                    if float(robot["position_error_m"]) <= 0.003:
                        actual_fraction = float(
                            np.dot(actual_tcp - expected_pass.start, direction) / np.dot(direction, direction)
                        )
                        actual_nearest = (
                            expected_pass.start
                            + np.clip(
                                actual_fraction,
                                0.0,
                                1.0,
                            )
                            * direction
                        )
                        assert np.linalg.norm(actual_tcp - actual_nearest) <= 0.004
                    assert float(robot["tcp_rigid_error_m"]) <= 0.003
                    assert float(robot["tcp_rigid_orientation_error_rad"]) <= np.deg2rad(3.0)
                    measured_deposition_samples += 1
                count = sum(
                    adapter.component_visible(tray_id, f"braze_{index:02d}") for index in range(1, 11)
                )
                visible_counts.append(count)
                current_half_length = float(adapter.model.geom_size[first_path_id, 1])
                observed_partial_length |= (
                    adapter.component_visible(tray_id, "braze_01")
                    and 0.0 < current_half_length < full_half_length
                )
            elif visible_counts and unit.stage.value == "MATERIAL_INSPECTION":
                break

        final_count = sum(adapter.component_visible(tray_id, f"braze_{index:02d}") for index in range(1, 11))
        assert any(0 < count < 10 for count in visible_counts)
        assert visible_counts == sorted(visible_counts)
        assert observed_partial_length
        assert measured_deposition_samples >= 5
        assert final_count == 10
    finally:
        adapter.close()


def test_v2_braze_beads_are_rendered_above_the_base_surface() -> None:
    """Visible material must not be geometrically buried inside the base."""

    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    try:
        runtime.submit_order("A", order_id="BRAZE_SURFACE_A")
        adapter.sync(runtime)
        tray_id = runtime.flow.trays[0].tray_id.lower()
        base = adapter.model.geom(f"{tray_id}_base_plate")
        base_top = float(base.pos[2] + base.size[2])
        for index in range(1, 11):
            bead = adapter.model.geom(f"{tray_id}_braze_{index:02d}")
            bead_bottom = float(bead.pos[2] - bead.size[0])
            assert bead_bottom >= base_top - 1.0e-6
    finally:
        adapter.close()


@pytest.mark.parametrize("preset", ("A", "B", "C"))
def test_v2_fin_installation_targets_the_live_tray_slot(preset: str) -> None:
    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    try:
        order_id = f"FIN_TARGET_{preset}"
        runtime.submit_order(preset, order_id=order_id)
        geometry = V2ProcessGeometry.for_preset(preset)
        observed = False
        for _ in range(2_400):
            runtime.tick(0.05)
            adapter.sync(runtime)
            adapter.step_physics(0.05)
            unit = runtime.units[f"{order_id}_UNIT_01"]
            if unit.stage.value != "FIN_INSTALLATION" or unit.tray_id is None:
                continue
            arm_name = "arm1" if unit.branch is None or unit.branch.value == "ARM1_A" else "arm3"
            robot = adapter.robot_motion_snapshot()[arm_name]
            if "纯Z向下放置" not in str(robot["target_zh"]):
                continue
            # Robot targets are authored in the permanent mocap carrier frame;
            # the welded free tray can differ by a few micrometres while the
            # constraint solver converges and must not move process targets.
            tray_body = adapter.data.body(f"{unit.tray_id.lower()}_carrier")
            expected = geometry.world_fin_target(
                unit.fins_installed,
                origin=np.asarray(tray_body.xpos, dtype=float),
                rotation=np.asarray(tray_body.xmat, dtype=float).reshape(3, 3),
            )
            np.testing.assert_allclose(robot["target_tcp_position_m"], expected, atol=5.0e-6)
            observed = True
            break
        assert observed, f"{preset}型未进入可验证的精确翘片放置阶段"
    finally:
        adapter.close()


def test_scene_adapter_reports_detailed_continuous_transport_for_the_shared_ui() -> None:
    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML, transfer_speed_m_s=0.5)
    try:
        runtime.submit_order("A", order_id="TRANSPORT_A")
        adapter.sync(runtime)
        runtime.tick(0.10)
        adapter.step_physics(0.10)
        adapter.sync(runtime)

        transfers = adapter.transport_snapshot()
        item = transfers["V2_TRAY_01"]
        assert item["source"] == "EMPTY_BUFFER"
        assert item["target"] == "S1"
        assert item["route_id"] == "EMPTY_BUFFER_TO_S1"
        assert 0.0 < item["progress"] < 1.0
        assert item["moving"]
        assert len(item["world_position_m"]) == 3
        assert item["distance_m"] > 0.0
    finally:
        adapter.close()


def test_scene_adapter_drives_visible_three_layer_furnace_transfer_mechanisms() -> None:
    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    try:
        runtime.submit_order("A", order_id="FURNACE_A", quantity=3)
        observed_extension = False
        observed_upper_layer = False
        # Three units now each perform a real 5 s material-analysis gate and
        # a real 5 s pre-braze-analysis gate before loading.  Keep the
        # mechanism assertion, but give that deliberate 30 s of simulation
        # time room instead of using the pre-camera-lifecycle deadline.
        for _ in range(3_600):
            runtime.tick(0.05)
            adapter.sync(runtime)
            adapter.step_physics(0.05)
            mechanism = adapter.furnace_transfer_snapshot()
            observed_extension |= mechanism["pusher"] > 0.05
            observed_upper_layer |= mechanism["lift"] > 0.10
            if runtime.furnace.state.phase.value in {"PREHEAT", "RAMP", "SOAK"}:
                break

        assert observed_extension
        assert observed_upper_layer
    finally:
        adapter.close()


def test_scene_adapter_drives_each_visible_carriage_from_physical_route_progress() -> None:
    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    try:
        runtime.submit_order("A", order_id="ROUTE_A")
        observed_s1_to_s2a = False
        observed_s2a_to_s2b = False
        # The real dispenser follows sampled Cartesian points instead of a
        # time-only animation, so allow the physical pass to finish before
        # checking the downstream carriage.
        for _ in range(450):
            runtime.tick(0.05)
            adapter.sync(runtime)
            adapter.step_physics(0.05)
            positions = adapter.route_mechanism_snapshot()
            observed_s1_to_s2a |= positions["s1_s2a"] > 0.02
            observed_s2a_to_s2b |= positions["s2a_s2b"] > 0.02
            if observed_s1_to_s2a and observed_s2a_to_s2b:
                break

        assert observed_s1_to_s2a
        assert observed_s2a_to_s2b
    finally:
        adapter.close()
