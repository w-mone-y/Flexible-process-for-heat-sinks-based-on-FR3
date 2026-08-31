from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest

from brazing_sim.dual_line import DualLineTopology

ROOT = Path(__file__).resolve().parents[2]
V2_XML = ROOT / "scenes" / "production" / "brazing_line_v2.xml"


@pytest.fixture(scope="module")
def model():
    mujoco = pytest.importorskip("mujoco")
    return mujoco.MjModel.from_xml_path(str(V2_XML))


def test_v2_scene_compiles_as_an_independent_model(model) -> None:
    assert V2_XML.exists()
    assert ElementTree.parse(V2_XML).getroot().attrib["model"] == "dual_install_brazing_line_v2"
    assert model.opt.timestep == pytest.approx(0.002)


def test_v2_table_supports_do_not_protrude_through_worktops(model) -> None:
    """The four structural legs must end below, not above, every tabletop."""

    import mujoco

    for body_name in (
        "v2_station_s1",
        "v2_station_s2a",
        "v2_station_s2b",
        "v2_fin_table_a",
        "v2_station_s3a",
        "v2_fin_table_b",
        "v2_station_s3b",
        "v2_station_s4",
        "v2_post_braze_table",
    ):
        body_id = int(model.body(body_name).id)
        box_geoms = [
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) == body_id
            and int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_BOX)
        ]
        tabletop = max(
            box_geoms,
            key=lambda geom_id: float(model.geom_size[geom_id, 0] * model.geom_size[geom_id, 1]),
        )
        supports = [geom_id for geom_id in box_geoms if geom_id != tabletop]
        assert len(supports) == 4, body_name
        tabletop_bottom = float(model.geom_pos[tabletop, 2] - model.geom_size[tabletop, 2])
        for support in supports:
            support_top = float(model.geom_pos[support, 2] + model.geom_size[support, 2])
            assert support_top <= tabletop_bottom + 1.0e-9, body_name


def test_v2_scene_exports_three_robots_and_arm3_hybrid_tool(model) -> None:
    for arm in ("arm1", "arm2", "arm3"):
        model.body(f"{arm}_base")
        target = model.body(f"{arm}_target")
        assert int(model.body_mocapid[target.id]) >= 0
        model.site(f"{arm}_attachment_site")
        for index in range(1, 8):
            model.joint(f"{arm}_fr3_joint{index}")
            model.actuator(f"{arm}_fr3_joint{index}")

    model.body("v2_arm3_hybrid_tool")
    model.site("v2_arm3_gripper_tcp")
    model.site("v2_arm3_camera_tcp")
    model.camera("v2_arm3_inspection_camera")
    model.joint("v2_arm3_left_finger_joint")
    model.joint("v2_arm3_right_finger_joint")
    model.actuator("v2_arm3_left_finger_actuator")
    model.actuator("v2_arm3_right_finger_actuator")
    model.equality("v2_arm3_hybrid_tool_weld")


def test_v2_arm1_tool_rack_uses_thin_contact_fingers_beneath_both_tools(model) -> None:
    """The V1-style cantilever visibly supports both collars without blocking the flange."""

    column = model.geom("v2_arm1_tool_rack_column")
    beam = model.geom("v2_arm1_tool_rack_beam")
    left_finger = model.geom("v2_arm1_tool_rack_left_finger")
    right_finger = model.geom("v2_arm1_tool_rack_right_finger")
    gripper = model.site("v2_arm1_parallel_gripper_rack_site")
    suction = model.site("v2_arm1_suction_tool_rack_site")
    safe_plane = model.site("v2_arm1_tool_rack_safe_plane")

    with pytest.raises(KeyError):
        model.geom("v2_arm1_tool_rack_right_column")
    assert float(column.pos[1]) > max(float(gripper.pos[1]), float(suction.pos[1]))
    assert float(beam.size[2] * 2.0) <= 0.020
    for finger in (left_finger, right_finger):
        assert float(finger.size[2] * 2.0) <= 0.016
        assert float(finger.pos[1] - finger.size[1]) <= float(gripper.pos[1])
        assert float(finger.pos[1] + finger.size[1]) >= float(suction.pos[1])
    assert float(left_finger.pos[0] + left_finger.size[0]) <= -0.025
    assert float(right_finger.pos[0] - right_finger.size[0]) >= 0.025
    support_top = max(
        float(beam.pos[2] + beam.size[2]),
        float(left_finger.pos[2] + left_finger.size[2]),
        float(right_finger.pos[2] + right_finger.size[2]),
    )
    assert 0.0 <= float(gripper.pos[2]) - support_top <= 0.008
    assert 0.0 <= float(suction.pos[2]) - support_top <= 0.008
    assert float(safe_plane.pos[2]) >= support_top + 0.060
    for dock in (gripper, suction):
        assert float(dock.pos[2]) < float(safe_plane.pos[2])


def test_v2_scene_has_six_physical_trays_and_explicit_ownership_welds(model) -> None:
    for index in range(1, 7):
        tray = f"v2_tray_{index:02d}"
        model.body(tray)
        model.joint(f"{tray}_free")
        model.site(f"{tray}_dock_site")
        carrier = model.body(f"{tray}_carrier")
        assert int(model.body_mocapid[carrier.id]) >= 0
        model.equality(f"{tray}_carrier_weld")
        for owner in (
            "s1",
            "s2a",
            "s2b",
            "install_a",
            "install_b",
            "merge_a_wait",
            "merge_b_wait",
            "merge",
            "s4",
            "buffer_1",
            "buffer_2",
            "buffer_3",
            "furnace_layer_0",
            "furnace_layer_1",
            "furnace_layer_2",
            "post_scan",
            "output",
        ):
            model.equality(f"{owner}_{tray}_weld")


def test_v2_tray_dock_sites_remain_resolvable_but_are_not_rendered(model) -> None:
    for index in range(1, 7):
        site = model.site(f"v2_tray_{index:02d}_dock_site")
        assert float(site.rgba[3]) == pytest.approx(0.0)


def test_v2_scene_has_dual_direct_branches_and_shared_inspection(model) -> None:
    for station in (
        "v2_station_s1",
        "v2_station_s2a",
        "v2_station_s2b",
        "v2_station_s3a",
        "v2_station_s3b",
        "v2_station_s4",
        "v2_fin_table_a",
        "v2_fin_table_b",
        "v2_merge",
    ):
        model.body(station)
        model.site(f"{station}_dock")

    for transfer in (
        "v2_s1_s2a",
        "v2_s2a_s2b",
        "v2_branch_a",
        "v2_branch_b",
        "v2_buffer_index",
        "v2_furnace_pusher",
        "v2_output_transfer",
    ):
        model.joint(f"{transfer}_joint")
        model.actuator(f"{transfer}_actuator")
    for removed in (
        "v2_merge_carriage",
        "v2_merge_transfer_joint",
        "v2_merge_transfer_actuator",
    ):
        with pytest.raises(KeyError):
            if removed.endswith("_joint"):
                model.joint(removed)
            elif removed.endswith("_actuator"):
                model.actuator(removed)
            else:
                model.body(removed)


def test_v2_through_furnace_has_two_doors_three_locks_and_rear_camera(model) -> None:
    model.body("v2_through_furnace")
    for door in ("front", "rear"):
        model.joint(f"v2_furnace_{door}_door_joint")
        model.actuator(f"v2_furnace_{door}_door_actuator")
    for layer in range(3):
        model.site(f"v2_furnace_layer_{layer}_dock")
        model.joint(f"v2_furnace_layer_{layer}_lock_joint")
        model.actuator(f"v2_furnace_layer_{layer}_lock_actuator")
    model.body("v2_post_braze_gantry")
    model.camera("v2_post_braze_camera")
    model.site("v2_finished_output_dock")


def test_v2_finished_output_matches_the_v1_opaque_yellow_gate_contract(model) -> None:
    model.body("v2_finished_output_box")
    model.body("v2_finished_output_gate")
    model.joint("v2_finished_output_gate_joint")
    model.actuator("v2_finished_output_gate_actuator")
    for geom_name in (
        "v2_finished_output_box_floor",
        "v2_finished_output_box_back",
        "v2_finished_output_box_left_wall",
        "v2_finished_output_box_right_wall",
        "v2_finished_output_box_roof",
    ):
        model.geom(geom_name)
    gate = model.geom("v2_finished_output_gate_panel")
    assert gate.rgba[0] >= 0.80
    assert gate.rgba[1] >= 0.50
    assert gate.rgba[2] <= 0.20
    with pytest.raises(KeyError):
        model.geom("v2_finished_output_window")


def test_v2_furnace_transfer_has_visible_lift_front_pusher_and_rear_extractor(model) -> None:
    for body_name in (
        "v2_furnace_lift",
        "v2_furnace_pusher_carriage",
        "v2_furnace_rear_lift",
        "v2_furnace_rear_extractor",
    ):
        model.body(body_name)
    for mechanism in (
        "v2_furnace_lift",
        "v2_furnace_pusher",
        "v2_furnace_rear_lift",
        "v2_furnace_rear_extractor",
    ):
        model.joint(f"{mechanism}_joint")
        model.actuator(f"{mechanism}_actuator")


def test_v2_post_braze_scan_has_a_full_flat_support_table(model) -> None:
    """The complete pallet must rest on a dedicated unobstructed scan deck."""

    import mujoco

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    table = model.geom("v2_post_braze_table_top")
    tray = model.geom("v2_tray_01_geom")
    dock = np.asarray(data.site("v2_post_braze_gantry_dock").xpos, dtype=float)
    table_center = np.asarray(data.geom_xpos[table.id], dtype=float)
    camera = np.asarray(data.camera("v2_post_braze_camera").xpos, dtype=float)

    np.testing.assert_allclose(table_center[:2], dock[:2], atol=1.0e-9)
    np.testing.assert_allclose(camera[:2], dock[:2], atol=1.0e-9)
    assert float(table.size[0] - tray.size[0]) >= 0.02
    assert float(table.size[1] - tray.size[1]) >= 0.02
    table_top = float(table_center[2] + table.size[2])
    tray_bottom = float(dock[2] + tray.pos[2] - tray.size[2])
    assert 0.0005 <= tray_bottom - table_top <= 0.005


def test_v2_finished_output_gate_clears_the_complete_post_scan_pallet(model) -> None:
    """The closed yellow gate must sit beyond, never through, the scan pallet."""

    import mujoco

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    scan_x = float(data.site("v2_post_braze_gantry_dock").xpos[0])
    output_x = float(data.site("v2_finished_output_dock").xpos[0])
    tray_half_x = float(model.geom("v2_tray_01_geom").size[0])
    gate = model.geom("v2_finished_output_gate_panel")
    gate_center_x = float(data.geom_xpos[gate.id, 0])
    gate_half_x = float(gate.size[0])

    scan_to_gate_clearance = gate_center_x - gate_half_x - (scan_x + tray_half_x)
    assert scan_to_gate_clearance >= 0.15
    assert output_x - tray_half_x - (gate_center_x + gate_half_x) >= 0.05

    box = model.body("v2_finished_output_box")
    floor = model.geom("v2_finished_output_box_floor")
    np.testing.assert_allclose(
        gate_center_x,
        float(box.pos[0] - floor.size[0]),
        atol=1.0e-9,
    )


def test_v2_rear_unload_mechanism_and_output_carriage_stay_below_the_tray(model) -> None:
    """Supports may carry the pallet but must never render through its product."""

    import mujoco

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    tray = model.geom("v2_tray_01_geom")
    tray_bottom_z = 0.225 + float(tray.pos[2]) - float(tray.size[2])
    for geom_name in (
        "v2_furnace_rear_lift_platform",
        "v2_furnace_rear_extractor_crossbar",
        "v2_furnace_rear_extractor_fork_left",
        "v2_furnace_rear_extractor_fork_right",
        "v2_output_carriage_geom",
    ):
        geom = model.geom(geom_name)
        top = float(data.geom_xpos[geom.id, 2] + geom.size[2])
        assert top <= tray_bottom_z - 0.001, geom_name

    output_joint = model.joint("v2_output_transfer_joint")
    output_actuator = model.actuator("v2_output_transfer_actuator")
    assert float(output_joint.range[1]) == pytest.approx(1.14)
    assert float(output_actuator.ctrlrange[1]) == pytest.approx(1.14)


def test_v2_front_lift_and_pusher_are_clear_of_the_closed_furnace_door(model) -> None:
    """Idle transfer hardware must not penetrate the closed front door."""

    import mujoco

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    door_body_id = int(model.body("v2_furnace_front_door").id)
    door_geom_ids = [
        geom_id for geom_id in range(model.ngeom) if int(model.geom_bodyid[geom_id]) == door_body_id
    ]
    assert door_geom_ids
    door_panel_id = max(
        door_geom_ids,
        key=lambda geom_id: float(model.geom_size[geom_id, 2]),
    )
    door_front_x = float(data.geom_xpos[door_panel_id, 0] - model.geom_size[door_panel_id, 0])
    for geom_name in (
        "v2_furnace_lift_platform",
        "v2_furnace_lift_left_rail",
        "v2_furnace_lift_right_rail",
        "v2_furnace_pusher_fork_left",
        "v2_furnace_pusher_fork_right",
    ):
        geom_id = int(model.geom(geom_name).id)
        transfer_rear_x = float(data.geom_xpos[geom_id, 0] + model.geom_size[geom_id, 0])
        clearance = door_front_x - transfer_rear_x
        assert clearance >= 0.020, geom_name


def test_v2_inactive_trays_start_clear_of_active_workstations(model) -> None:
    import mujoco

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    tray_ids = {int(model.body(f"v2_tray_{index:02d}").id) for index in range(1, 7)}
    assert not any(
        int(model.geom_bodyid[contact.geom1]) in tray_ids or int(model.geom_bodyid[contact.geom2]) in tray_ids
        for contact in data.contact
    )


def test_v2_authoritative_topology_matches_mjcf_dock_world_positions(model) -> None:
    import mujoco
    import numpy as np

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    topology = DualLineTopology.standard()
    dock_sites = {
        "S1_BASE_LOADING": "v2_station_s1_dock",
        "S2A_DISPENSING": "v2_station_s2a_dock",
        "S2B_MATERIAL_INSPECTION": "v2_station_s2b_dock",
        "FIN_TABLE_A": "v2_fin_table_a_dock",
        "S3A_ARM1_INSTALL": "v2_station_s3a_dock",
        "FIN_TABLE_B": "v2_fin_table_b_dock",
        "S3B_ARM3_INSTALL": "v2_station_s3b_dock",
        "Y_MERGE_SHARED": "v2_merge_dock",
        "S4_PRE_BRAZE_INSPECTION": "v2_station_s4_dock",
        "FURNACE_BUFFER_1": "v2_buffer_1_dock",
        "FURNACE_BUFFER_2": "v2_buffer_2_dock",
        "FURNACE_BUFFER_3": "v2_buffer_3_dock",
        "FURNACE_LAYER_0": "v2_furnace_layer_0_dock",
        "FURNACE_LAYER_1": "v2_furnace_layer_1_dock",
        "FURNACE_LAYER_2": "v2_furnace_layer_2_dock",
        "POST_BRAZE_SCAN": "v2_post_braze_gantry_dock",
        "FINISHED_OUTPUT": "v2_finished_output_dock",
    }

    for station_id, site_name in dock_sites.items():
        expected = np.asarray(data.site(site_name).xpos, dtype=float)
        actual = np.asarray(topology.station(station_id).world_xyz, dtype=float)
        np.testing.assert_allclose(actual, expected, atol=1.0e-9, err_msg=station_id)


def test_v2_six_trays_each_own_a_complete_v1_style_visual_product_pool(model) -> None:
    for tray_index in range(1, 7):
        prefix = f"v2_tray_{tray_index:02d}_"
        tray_body = model.body(f"v2_tray_{tray_index:02d}")
        payload_body = model.body(prefix + "payload")
        assert int(model.body_parentid[payload_body.id]) == int(tray_body.id)
        for geom_name in (
            "template_plate",
            "base_plate",
            "front_comb_base",
            "rear_comb_base",
            "front_press",
            "rear_press",
        ):
            model.geom(prefix + geom_name)
        for fin_index in range(1, 13):
            model.geom(f"{prefix}fin_{fin_index:02d}")
        for path_index in range(1, 25):
            model.geom(f"{prefix}braze_{path_index:02d}")
        for side in ("front", "rear"):
            for guide_side in ("left", "right"):
                for guide_index in range(12):
                    model.geom(f"{prefix}{side}_comb_guide_{guide_side}{guide_index:02d}")


def test_v2_s1_has_a_visible_base_supply_fixture_and_pickup_reference(model) -> None:
    supply = model.body("v2_base_supply_fixture")
    assert int(supply.id) >= 0
    for geom_name in (
        "v2_base_supply_deck",
        "v2_base_supply_left_locator",
        "v2_base_supply_right_locator",
        "v2_base_supply_backstop",
    ):
        geom = model.geom(geom_name)
        assert geom.rgba[3] > 0.9
    pickup = model.site("v2_base_supply_pickup_site")
    assert pickup.pos[2] > model.geom("v2_base_supply_deck").pos[2]
    stock = model.geom("v2_base_supply_stock_plate")
    assert stock.rgba[3] > 0.0
    assert stock.contype == 0
    assert stock.conaffinity == 0


def test_v2_base_supply_is_an_independent_reachable_table_clear_of_s1(model) -> None:
    """The raw base fixture must not be embedded in the S1 pallet table."""

    mujoco = pytest.importorskip("mujoco")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    def horizontal_aabb(name: str) -> tuple[np.ndarray, np.ndarray]:
        geom = model.geom(name)
        rotation = np.asarray(data.geom_xmat[geom.id], dtype=float).reshape(3, 3)
        extent = np.abs(rotation) @ np.asarray(geom.size, dtype=float)
        centre = np.asarray(data.geom_xpos[geom.id], dtype=float)
        return centre - extent, centre + extent

    supply_min, supply_max = horizontal_aabb("v2_base_supply_deck")
    s1_min, s1_max = horizontal_aabb("v2_station_s1_top")
    overlap_xy = np.minimum(supply_max[:2], s1_max[:2]) - np.maximum(
        supply_min[:2],
        s1_min[:2],
    )
    assert np.any(overlap_xy <= 0.0), f"主板原料台与S1仍重叠：{overlap_xy.tolist()}"
    np.testing.assert_allclose(
        data.body("v2_base_supply_fixture").xpos[:2],
        (-0.55, 0.75),
        atol=1.0e-9,
    )


def test_v2_base_supply_and_tool_rack_are_ordered_for_a_direct_pick_to_place(
    model,
) -> None:
    """The material table owns the short S1 aisle; the rack stays behind it."""

    mujoco = pytest.importorskip("mujoco")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    s1 = np.asarray(data.site("v2_station_s1_dock").xpos[:2], dtype=float)
    supply = np.asarray(data.site("v2_base_supply_pickup_site").xpos[:2], dtype=float)
    rack = np.asarray(data.body("v2_arm1_tool_rack").xpos[:2], dtype=float)

    # Pickup and placement share one X centreline, so the carried plate only
    # needs lift → straight translation → slow vertical placement.
    assert abs(float(supply[0] - s1[0])) <= 0.005
    assert 0.30 <= float(supply[1] - s1[1]) <= 0.45
    # The quick-change rack is behind the pickup table, never between pickup
    # and S1 as it was in the collision-prone layout.
    assert float(rack[1]) >= float(supply[1]) + 0.25


def test_v2_arm3_cell_matches_the_marked_south_magazine_layout(model) -> None:
    """Arm3 stays in the marked circle and the fin magazine moves south."""

    arm3 = np.asarray(model.body("arm3_base").pos[:2], dtype=float)
    install = np.asarray(model.body("v2_station_s3b").pos[:2], dtype=float)
    magazine = np.asarray(model.body("v2_fin_table_b").pos[:2], dtype=float)
    merge_wait = np.asarray(model.body("v2_merge_b_wait_anchor").pos[:2], dtype=float)
    merge = np.asarray(model.body("v2_merge_anchor").pos[:2], dtype=float)

    np.testing.assert_allclose(arm3, (0.97, -0.49), atol=1.0e-9)
    np.testing.assert_allclose(magazine, (0.55, -0.85), atol=1.0e-9)
    np.testing.assert_allclose(merge_wait, (1.40, -1.22), atol=1.0e-9)
    assert abs(float(magazine[0] - install[0])) <= 0.25
    # The upward move requested in the marked layout still leaves 80 mm
    # between the two tabletop footprints.
    assert float(install[1] - magazine[1]) >= 0.39

    # The requested B-line wait is exactly south of S4, so its final segment
    # is a vertical centreline entry matching the A-side visual principle.
    assert float(merge_wait[0]) == pytest.approx(float(merge[0]))
    assert float(arm3[1] - merge_wait[1]) >= 0.65


def test_v2_shared_inspection_table_is_relocated_toward_the_furnace(model) -> None:
    """S4 moves east instead of adding a north-side B-line return loop."""

    s4 = np.asarray(model.body("v2_station_s4").pos[:2], dtype=float)
    merge = np.asarray(model.body("v2_merge_anchor").pos[:2], dtype=float)
    buffer_1 = np.asarray(model.body("v2_buffer_1_anchor").pos[:2], dtype=float)
    furnace_front = np.asarray(model.body("v2_furnace_lift").pos[:2], dtype=float)

    np.testing.assert_allclose(s4, (1.40, 0.00), atol=1.0e-9)
    np.testing.assert_allclose(merge, s4, atol=1.0e-9)
    assert float(furnace_front[0] - s4[0]) <= 1.35

    # The shifted first buffer remains beyond the S4 tabletop and a complete
    # 400 mm tray instead of overlapping the relocated inspection station.
    s4_half_x = float(model.geom("v2_station_s4_top").size[0])
    tray_half_x = float(model.geom("v2_tray_01_geom").size[0])
    assert float(buffer_1[0] - tray_half_x - (s4[0] + s4_half_x)) >= 0.01


def test_v2_tray_bottom_is_not_coplanar_with_station_worktops(model) -> None:
    """A tray resting on a station needs a render gap, not coincident faces."""

    tray_half_height = float(model.geom("v2_tray_01_geom").size[2])
    for station, dock in (
        ("v2_station_s1_top", "v2_station_s1_dock"),
        ("v2_station_s2a_top", "v2_station_s2a_dock"),
        ("v2_station_s2b_top", "v2_station_s2b_dock"),
    ):
        worktop = model.geom(station)
        worktop_top = float(worktop.pos[2] + worktop.size[2])
        tray_bottom = float(model.site(dock).pos[2] - tray_half_height)
        assert tray_bottom - worktop_top >= 0.0005


def test_v2_early_transfer_carriages_stay_below_the_tray_bottom(model) -> None:
    """A visible slide carriage must support, never intersect, its pallet."""

    tray = model.geom("v2_tray_01_geom")
    tray_bottom_z = 0.225 + float(tray.pos[2]) - float(tray.size[2])
    for body_name in (
        "v2_s1_s2a_carriage",
        "v2_s2a_s2b_carriage",
        "v2_branch_a_carriage",
        "v2_branch_b_carriage",
        "v2_buffer_index_carriage",
    ):
        body = model.body(body_name)
        geom_id = int(model.body_geomadr[body.id])
        geom = model.geom(geom_id)
        carriage_top_z = float(body.pos[2] + geom.pos[2] + geom.size[2])
        assert carriage_top_z <= tray_bottom_z - 0.001, body_name


def test_v2_installation_branches_use_explicit_planar_obstacle_bypasses(model) -> None:
    """The two branch waits encode north and south horizontal detours."""

    install_a = np.asarray(model.body("v2_station_s3a").pos[:2], dtype=float)
    install_b = np.asarray(model.body("v2_station_s3b").pos[:2], dtype=float)
    wait_a = np.asarray(model.body("v2_merge_a_wait_anchor").pos[:2], dtype=float)
    wait_b = np.asarray(model.body("v2_merge_b_wait_anchor").pos[:2], dtype=float)
    merge = np.asarray(model.body("v2_merge_anchor").pos[:2], dtype=float)

    np.testing.assert_allclose(wait_a, (merge[0], install_a[1]), atol=1.0e-9)
    assert float(wait_b[0]) == pytest.approx(float(merge[0]))
    assert float(wait_b[1]) <= -1.20
    assert float(wait_b[0]) > float(install_b[0])
    assert float(wait_b[1]) < float(install_b[1])


def test_v2_s3_s4_routes_remain_on_the_station_transport_plane(model) -> None:
    """Outbound pallets must route around obstacles without any Z lift."""

    station_height = float(model.site("v2_station_s4_dock").pos[2])
    assert float(model.body("v2_merge_a_wait_anchor").pos[2]) == pytest.approx(
        station_height,
    )
    assert float(model.body("v2_merge_b_wait_anchor").pos[2]) == pytest.approx(
        station_height,
    )

    segment_counts = {"a": 2, "b": 4}
    for branch, count in segment_counts.items():
        for side in ("left", "right"):
            for index in range(1, count + 1):
                suffix = "" if index == 1 else f"_{index:02d}"
                rail = model.geom(f"v2_install_{branch}_s4_rail_{side}{suffix}")
                assert float(rail.pos[2]) == pytest.approx(0.198)

    for side in ("left", "right"):
        with pytest.raises(KeyError):
            model.geom(f"v2_install_b_s4_rail_{side}_05")

    for name in (
        "v2_s3a_overpass_lift_left",
        "v2_s3a_overpass_lift_right",
        "v2_s3b_overpass_lift_left",
        "v2_s3b_overpass_lift_right",
        "v2_s4_overpass_lift_a_left",
        "v2_s4_overpass_lift_a_right",
        "v2_s4_overpass_lift_b_left",
        "v2_s4_overpass_lift_b_right",
    ):
        with pytest.raises(KeyError):
            model.geom(name)


def test_v2_early_routes_use_v1_style_capsule_rails_not_black_belt_slabs(model) -> None:
    for route in (
        "s1_s2a",
        "s2a_s2b",
        "s2b_s3a",
        "s2b_s3b",
        "install_a_s4",
        "install_b_s4",
    ):
        left = model.geom(f"v2_{route}_rail_left")
        right = model.geom(f"v2_{route}_rail_right")
        assert int(left.type[0]) == int(right.type[0])
        assert left.size[0] <= 0.010
    for old_belt in (
        "v2_main_belt_s1_s2a",
        "v2_main_belt_s2a_s2b",
        "v2_branch_a_belt",
        "v2_branch_b_belt",
        "v2_merge_belt_a",
        "v2_merge_belt_b",
        "v2_merge_gate_belt_a",
        "v2_merge_gate_belt_b",
        "v2_install_a_merge_rail_left",
        "v2_install_a_merge_rail_right",
        "v2_install_b_merge_rail_left",
        "v2_install_b_merge_rail_right",
        "v2_merge_s4_rail_left",
        "v2_merge_s4_rail_right",
    ):
        with pytest.raises(KeyError):
            model.geom(old_belt)


def test_v2_scene_has_v1_quality_arm1_tooling_and_fixed_arm2_dispenser(model) -> None:
    for body_name in (
        "v2_arm1_tool_rack",
        "v2_arm1_parallel_gripper",
        "v2_arm1_suction_tool",
        "v2_arm1_raw_base_proxy",
        "v2_arm1_raw_fin_proxy",
        "v2_arm3_raw_fin_proxy",
        "v2_arm2_dual_brazing_dispenser_tool",
    ):
        model.body(body_name)

    for joint_name in (
        "v2_arm1_left_finger_joint",
        "v2_arm1_right_finger_joint",
    ):
        model.joint(joint_name)

    for site_name in (
        "v2_arm1_grasp_tcp",
        "v2_arm1_suction_tcp",
        "v2_arm2_dispenser_center_tcp",
        "v2_arm2_left_nozzle_tip_site",
        "v2_arm2_right_nozzle_tip_site",
    ):
        model.site(site_name)

    for weld_name in (
        "v2_arm1_toolchange_parallel_gripper",
        "v2_arm1_toolchange_suction_tool",
        "v2_arm1_rack_parallel_gripper",
        "v2_arm1_rack_suction_tool",
        "v2_arm1_raw_base_feed_weld",
        "v2_arm1_grasp_base_proxy_weld",
        "v2_arm1_raw_fin_feed_weld",
        "v2_arm1_grasp_fin_proxy_weld",
        "v2_arm3_raw_fin_feed_weld",
        "v2_arm3_grasp_fin_proxy_weld",
        "v2_arm2_dispenser_tool_weld",
    ):
        model.equality(weld_name)

    for actuator_name in (
        "v2_arm1_left_finger_actuator",
        "v2_arm1_right_finger_actuator",
    ):
        model.actuator(actuator_name)


def test_v2_scene_starts_without_penetrating_v2_equipment_contacts(model) -> None:
    import mujoco

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    unexpected: list[tuple[str | None, str | None, float]] = []
    for contact in data.contact:
        if contact.dist >= 0:
            continue
        first = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            int(contact.geom1),
        )
        second = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            int(contact.geom2),
        )
        if (first or "").startswith("v2_") or (second or "").startswith("v2_"):
            unexpected.append((first, second, float(contact.dist)))
    assert unexpected == []


def test_v2_conveyor_surfaces_are_recessed_below_station_decks(model) -> None:
    """Overlapping coplanar rail/deck faces flicker as the viewer moves."""

    rail_names = (
        "v2_s1_s2a_rail_left",
        "v2_s1_s2a_rail_right",
        "v2_s2a_s2b_rail_left",
        "v2_s2a_s2b_rail_right",
        "v2_s2b_s3a_rail_left",
        "v2_s2b_s3a_rail_right",
        "v2_s2b_s3b_rail_left",
        "v2_s2b_s3b_rail_right",
    )
    deck_names = ("v2_station_s1_top", "v2_station_s2a_top", "v2_station_s2b_top")
    rail_tops = {name: float(model.geom(name).pos[2] + model.geom(name).size[0]) for name in rail_names}
    deck_tops = {name: float(model.geom(name).pos[2] + model.geom(name).size[2]) for name in deck_names}

    # Rails run into the station frames, so their visible top must sit below
    # the deck instead of sharing the same depth-buffer plane.
    assert max(rail_tops.values()) <= min(deck_tops.values()) - 0.003


def test_v2_arm3_and_post_braze_cameras_start_looking_down(model) -> None:
    import mujoco
    import numpy as np

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for camera_name in (
        "v2_arm3_inspection_camera",
        "v2_post_braze_camera",
    ):
        camera_id = int(model.camera(camera_name).id)
        rotation = np.asarray(data.cam_xmat[camera_id], dtype=float).reshape(3, 3)
        optical_axis = -rotation[:, 2]
        assert float(np.dot(optical_axis, np.array([0.0, 0.0, -1.0]))) > 0.999

    from brazing_sim.inspection import ARM3_CAMERA_FOVY_DEG

    assert float(model.camera("v2_arm3_inspection_camera").fovy[0]) == pytest.approx(ARM3_CAMERA_FOVY_DEG)


def test_v2_key_process_equipment_has_visible_chinese_station_plaques(model) -> None:
    for geom_name in (
        "v2_s1_station_sign",
        "v2_s2a_station_sign",
        "v2_s2b_station_sign",
        "v2_fin_a_station_sign",
        "v2_s3a_station_sign",
        "v2_fin_b_station_sign",
        "v2_s3b_station_sign",
        "v2_s4_station_sign",
        "v2_furnace_station_sign",
        "v2_post_braze_station_sign",
        "v2_finished_output_sign",
    ):
        model.geom(geom_name)


def test_v2_each_table_has_a_distinct_correctly_named_prominent_plaque(model) -> None:
    """Every worktable needs its own truthful sign above an outside edge."""

    tables = (
        ("v2_station_s1_top", "v2_s1_station_sign"),
        ("v2_station_s2a_top", "v2_s2a_station_sign"),
        ("v2_station_s2b_top", "v2_s2b_station_sign"),
        ("v2_fin_table_a_top", "v2_fin_a_station_sign"),
        ("v2_station_s3a_top", "v2_s3a_station_sign"),
        ("v2_fin_table_b_top", "v2_fin_b_station_sign"),
        ("v2_station_s3b_top", "v2_s3b_station_sign"),
        ("v2_station_s4_top", "v2_s4_station_sign"),
    )
    material_ids: set[int] = set()
    for tabletop_name, plaque_name in tables:
        tabletop = model.geom(tabletop_name)
        plaque = model.geom(plaque_name)
        assert float(plaque.pos[2]) > float(tabletop.pos[2] + tabletop.size[2])
        assert abs(float(plaque.pos[0])) > float(tabletop.size[0]) or abs(float(plaque.pos[1])) > float(
            tabletop.size[1]
        )
        material_ids.add(int(plaque.matid[0]))
    assert len(material_ids) == len(tables)

    root = ElementTree.parse(V2_XML).getroot()
    textures = {
        node.attrib["name"]: node.attrib["file"]
        for node in root.findall("./asset/texture")
        if "file" in node.attrib
    }
    assert {
        textures["v2_s1_sign_tex"],
        textures["v2_s2a_sign_tex"],
        textures["v2_s2b_sign_tex"],
        textures["v2_fin_a_sign_tex"],
        textures["v2_s3a_sign_tex"],
        textures["v2_fin_b_sign_tex"],
        textures["v2_s3b_sign_tex"],
        textures["v2_s4_sign_tex"],
    } == {
        "assets/signs/v2_s1_base_loading_sign.png",
        "assets/signs/v2_s2a_dispensing_sign.png",
        "assets/signs/v2_s2b_coating_inspection_sign.png",
        "assets/signs/v2_fin_a_supply_sign.png",
        "assets/signs/v2_s3a_install_sign.png",
        "assets/signs/v2_fin_b_supply_sign.png",
        "assets/signs/v2_s3b_install_sign.png",
        "assets/signs/v2_s4_pre_braze_inspection_sign.png",
    }


def test_v2_s1_s2a_has_a_visible_aisle_and_full_length_slide_rail(model) -> None:
    """The first two tabletops must not hide almost the whole slide rail."""

    s1 = np.asarray(model.body("v2_station_s1").pos[:2], dtype=float)
    s2a = np.asarray(model.body("v2_station_s2a").pos[:2], dtype=float)
    s1_top = model.geom("v2_station_s1_top")
    s2a_top = model.geom("v2_station_s2a_top")
    visible_y_gap = abs(float(s1[1] - s2a[1])) - float(
        s1_top.size[1] + s2a_top.size[1],
    )
    assert visible_y_gap >= 0.10
    assert float(np.linalg.norm(s2a - s1)) >= 0.49
    for side in ("left", "right"):
        rail = model.geom(f"v2_s1_s2a_rail_{side}")
        assert 2.0 * float(rail.size[1]) >= 0.49


def test_v2_branch_b_visible_rails_enter_the_exact_s4_centre(model) -> None:
    """The right-hand route must enter S4 vertically on its centreline."""

    import mujoco

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    s4 = np.asarray(data.site("v2_station_s4_dock").xpos, dtype=float)
    closest_endpoints: list[np.ndarray] = []
    expected_x = {"left": 1.345, "right": 1.455}
    for side in ("left", "right"):
        rail = model.geom(f"v2_install_b_s4_rail_{side}_04")
        rotation = np.asarray(data.geom_xmat[rail.id], dtype=float).reshape(3, 3)
        axis = rotation[:, 2]
        center = np.asarray(data.geom_xpos[rail.id], dtype=float)
        half_length = float(rail.size[1])
        endpoints = (center - axis * half_length, center + axis * half_length)
        assert float(np.ptp(np.asarray(endpoints)[:, 0])) <= 1.0e-9
        assert float(center[0]) == pytest.approx(expected_x[side])
        closest_endpoints.append(min(endpoints, key=lambda point: float(np.linalg.norm(point - s4))))
    np.testing.assert_allclose(
        np.mean(closest_endpoints, axis=0),
        s4 + (0.0, 0.0, -0.027),
        atol=1.0e-12,
    )


def test_v2_two_installation_branches_each_show_twelve_raw_fins(model) -> None:
    for branch in ("a", "b"):
        model.body(f"v2_fin_{branch}_magazine")
        for index in range(1, 13):
            model.geom(f"v2_fin_{branch}_raw_fin_{index:02d}")


def test_v2_visible_workcell_surfaces_never_share_an_overlapping_render_plane(model) -> None:
    """Tables, belts and moving decks must not z-fight as the camera moves."""

    mujoco = pytest.importorskip("mujoco")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    surfaces: list[tuple[str, float, float, float, float, float]] = []
    workstation_bodies = {
        "v2_station_s1",
        "v2_station_s2a",
        "v2_station_s2b",
        "v2_station_s3a",
        "v2_station_s3b",
        "v2_station_s4",
        "v2_fin_table_a",
        "v2_fin_table_b",
        "v2_post_braze_table",
    }
    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        if float(model.geom_rgba[geom_id, 3]) <= 0.0:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom#{geom_id}"
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        selected = (
            body_name in workstation_bodies
            or body_name.endswith("_carriage")
            or "belt" in name
            or "platform" in name
            or "magazine_back" in name
        )
        if not selected:
            continue
        rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        if abs(abs(float(rotation[2, 2])) - 1.0) > 1.0e-6:
            continue
        size = np.asarray(model.geom_size[geom_id], dtype=float)
        centre = np.asarray(data.geom_xpos[geom_id], dtype=float)
        extent_x = float(np.sum(np.abs(rotation[0]) * size))
        extent_y = float(np.sum(np.abs(rotation[1]) * size))
        extent_z = float(np.sum(np.abs(rotation[2]) * size))
        surfaces.append(
            (
                f"{body_name}/{name}",
                float(centre[2] + extent_z),
                float(centre[0] - extent_x),
                float(centre[0] + extent_x),
                float(centre[1] - extent_y),
                float(centre[1] + extent_y),
            )
        )

    conflicts: list[str] = []
    for index, first in enumerate(surfaces):
        for second in surfaces[index + 1 :]:
            overlap_x = min(first[3], second[3]) - max(first[2], second[2])
            overlap_y = min(first[5], second[5]) - max(first[4], second[4])
            if overlap_x <= 0.002 or overlap_y <= 0.002:
                continue
            if abs(first[1] - second[1]) <= 0.0002:
                conflicts.append(f"{first[0]} <-> {second[0]}")
    assert conflicts == [], "共面可见表面会随视角闪烁：" + ", ".join(conflicts)


def test_v2_arm3_fin_gripper_is_the_same_parallel_gripper_as_arm1(model) -> None:
    """The hybrid camera stays, but its fin gripper must physically match Arm1."""

    model.body("v2_arm3_parallel_gripper")
    np.testing.assert_allclose(
        model.geom("v2_arm3_gripper_palm").size,
        model.geom("v2_arm1_gripper_palm").size,
    )
    np.testing.assert_allclose(
        model.geom("v2_arm3_gripper_palm").pos,
        model.geom("v2_arm1_gripper_palm").pos,
    )
    for side in ("left", "right"):
        np.testing.assert_allclose(
            model.geom(f"v2_arm1_gripper_{side}").size,
            (0.0032, 0.0010, 0.0400),
            atol=1.0e-9,
        )
        np.testing.assert_allclose(
            model.geom(f"v2_arm1_gripper_{side}_pad").size,
            (0.0035, 0.0006, 0.0340),
            atol=1.0e-9,
        )
        np.testing.assert_allclose(
            model.geom(f"v2_arm3_gripper_{side}").size,
            model.geom(f"v2_arm1_gripper_{side}").size,
        )
        np.testing.assert_allclose(
            model.geom(f"v2_arm3_gripper_{side}_pad").size,
            model.geom(f"v2_arm1_gripper_{side}_pad").size,
        )
        body = model.body(f"v2_arm1_gripper_{side}_body")
        finger = model.geom(f"v2_arm1_gripper_{side}")
        pad = model.geom(f"v2_arm1_gripper_{side}_pad")
        finger_center_y = float(body.pos[1] + finger.pos[1])
        pad_center_y = float(body.pos[1] + pad.pos[1])
        # The dark inner pad must meet the inward face of its metal finger;
        # a visible gap here makes it look like a floating third component.
        if side == "left":
            assert pad_center_y + float(pad.size[1]) >= finger_center_y - float(finger.size[1]) - 1.0e-6
        else:
            assert pad_center_y - float(pad.size[1]) <= finger_center_y + float(finger.size[1]) + 1.0e-6
        np.testing.assert_allclose(
            model.geom(f"v2_arm3_gripper_{side}_pad").pos,
            pad.pos,
            atol=1.0e-9,
        )
        np.testing.assert_allclose(
            model.joint(f"v2_arm3_{side}_finger_joint").axis,
            model.joint(f"v2_arm1_{side}_finger_joint").axis,
        )
        np.testing.assert_allclose(
            model.joint(f"v2_arm3_{side}_finger_joint").range,
            model.joint(f"v2_arm1_{side}_finger_joint").range,
        )
    np.testing.assert_allclose(
        model.site("v2_arm3_gripper_tcp").pos,
        model.site("v2_arm1_grasp_tcp").pos,
    )


@pytest.mark.parametrize(
    ("fin_table", "install_table", "half_extents"),
    (
        ("v2_fin_table_a", "v2_station_s3a", ((0.22, 0.14), (0.24, 0.18))),
        ("v2_fin_table_b", "v2_station_s3b", ((0.18, 0.14), (0.24, 0.18))),
    ),
)
def test_v2_fin_supply_and_installation_tables_have_visible_planar_clearance(
    model,
    fin_table: str,
    install_table: str,
    half_extents: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    """Supply and installation worktops must not overlap or visually merge."""

    fin_xy = np.asarray(model.body(fin_table).pos[:2], dtype=float)
    install_xy = np.asarray(model.body(install_table).pos[:2], dtype=float)
    gaps = np.abs(fin_xy - install_xy) - np.sum(np.asarray(half_extents), axis=0)
    planar_clearance = float(np.linalg.norm(np.maximum(gaps, 0.0)))
    # The B-line magazine is deliberately moved upward in the annotated V2
    # layout.  It must remain visibly separate rather than preserve the former
    # oversized 230 mm service aisle.
    required_clearance = 0.075 if fin_table == "v2_fin_table_b" else 0.15
    assert planar_clearance >= required_clearance


@pytest.mark.parametrize(
    ("branch", "table_top", "expected_half_extents"),
    (
        ("a", "v2_fin_table_a_top", (0.22, 0.14)),
        ("b", "v2_fin_table_b_top", (0.18, 0.14)),
    ),
)
def test_v2_fin_supply_tables_are_flat_full_dark_surfaces_without_side_rails(
    model,
    branch: str,
    table_top: str,
    expected_half_extents: tuple[float, float],
) -> None:
    """The raw-fin table is one full flat dark surface, never a raised trough."""

    mujoco = pytest.importorskip("mujoco")
    for component in ("magazine_front_rail", "magazine_rear_rail", "magazine_back"):
        assert (
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                f"v2_fin_{branch}_{component}",
            )
            == -1
        )
    top = model.geom(table_top)
    np.testing.assert_allclose(top.size[:2], expected_half_extents, atol=1.0e-9)
    material_id = int(np.asarray(top.matid).reshape(-1)[0])
    assert material_id == int(model.material("v2_fin_supply_surface_mat").id)
    assert max(float(value) for value in model.mat_rgba[material_id, :3]) <= 0.24


def test_v2_arm3_camera_is_visibly_side_mounted_on_the_gripper_base(model) -> None:
    """A distinct camera module must hug one side of the gripper palm."""

    gripper = model.body("v2_arm3_parallel_gripper")
    camera_rig = model.body("v2_arm3_camera_rig")
    assert int(model.body_parentid[camera_rig.id]) == int(gripper.id)

    camera_tcp = model.site("v2_arm3_camera_tcp")
    camera_in_gripper = np.asarray(camera_rig.pos) + np.asarray(camera_tcp.pos)
    palm = model.geom("v2_arm3_gripper_palm")
    housing = model.geom("v2_arm3_camera_housing")
    bracket = model.geom("v2_arm3_camera_side_bracket")

    # The optical centre is outside the palm footprint on +X, while the
    # bracket and housing still touch the palm instead of floating nearby.
    assert float(camera_in_gripper[0]) >= float(palm.size[0]) + 0.010
    assert abs(float(camera_in_gripper[1])) <= 0.005
    assert float(camera_rig.pos[0] - housing.size[0]) <= float(palm.size[0]) + 0.003
    assert float(camera_rig.pos[0] - bracket.size[0]) <= float(palm.size[0])
    assert float(camera_in_gripper[2]) <= 0.060

    # A separate parent flange was the visible blue/grey double base.
    with pytest.raises(KeyError):
        model.geom("v2_arm3_tool_flange")

    # The camera remains large enough to read as a camera in the main viewer.
    lens = model.geom("v2_arm3_camera_lens")
    assert float(housing.size[0]) >= 0.012
    assert float(housing.size[1]) >= 0.012
    assert float(lens.size[0]) >= 0.008


def test_v2_robot_layout_reaches_every_authored_process_target_with_margin() -> None:
    import math

    import numpy as np

    from brazing_sim.dual_line.scene_adapter import DualLineSceneAdapter
    from brazing_sim.inspection import top_down_inspection_pose
    from brazing_sim.motion import (
        ArmController,
        Pose,
        matrix_to_quat,
        pose_from_site,
    )

    adapter = DualLineSceneAdapter(V2_XML)
    try:
        model, data = adapter.model, adapter.data
        top_down_reversed = matrix_to_quat(
            np.asarray(
                [
                    [-1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, -1.0],
                ]
            )
        )
        checks: list[tuple[ArmController, object]] = []

        arm1 = ArmController(model, data, "arm1")
        arm1.set_tool_transform(Pose([0.0, 0.0, 0.09]))
        for position in (
            (-0.55, 0.35, 0.50),
            (0.55, 0.45, 0.48),
            (0.45, 1.05, 0.48),
        ):
            checks.append(
                (
                    arm1,
                    arm1.solve_ik(
                        Pose(position, top_down_reversed),
                        tcp=True,
                        full_orientation=True,
                        max_iterations=1_000,
                    ),
                )
            )

        arm2 = ArmController(model, data, "arm2")
        arm2_flange = pose_from_site(data, "arm2_attachment_site")
        arm2_tcp = pose_from_site(data, "v2_arm2_dispenser_center_tcp")
        arm2.set_tool_transform(arm2_flange.inverse().transformed(arm2_tcp))
        checks.append(
            (
                arm2,
                arm2.solve_ik(
                    Pose((-0.35, -0.10, 0.32), top_down_reversed),
                    tcp=True,
                    full_orientation=True,
                    max_iterations=1_000,
                ),
            )
        )

        arm3 = ArmController(model, data, "arm3")
        arm3_flange = pose_from_site(data, "arm3_attachment_site")
        arm3_camera = pose_from_site(data, "v2_arm3_camera_tcp")
        arm3.set_tool_transform(arm3_flange.inverse().transformed(arm3_camera))
        for dock_name in ("v2_station_s2b_dock", "v2_station_s4_dock"):
            surface_center = np.asarray(data.site(dock_name).xpos, dtype=float).copy()
            surface_center[2] += 0.032
            checks.append(
                (
                    arm3,
                    arm3.solve_ik(
                        top_down_inspection_pose(
                            surface_center,
                            product_length_m=0.36,
                            product_width_m=0.22,
                            product_yaw_rad=math.pi,
                        ),
                        tcp=True,
                        full_orientation=True,
                        max_iterations=1_000,
                    ),
                )
            )
        arm3_gripper = pose_from_site(data, "v2_arm3_gripper_tcp")
        arm3.set_tool_transform(arm3_flange.inverse().transformed(arm3_gripper))
        fin_pickup = np.asarray(data.geom("v2_fin_b_raw_fin_01").xpos, dtype=float).copy()
        fin_pickup[2] = 0.48
        for position in (
            (0.35, -0.45, 0.48),
            fin_pickup,
        ):
            checks.append(
                (
                    arm3,
                    arm3.solve_ik(
                        Pose(position, top_down_reversed),
                        tcp=True,
                        full_orientation=True,
                        max_iterations=1_000,
                    ),
                )
            )

        for controller, result in checks:
            assert result.reachable
            margin = np.minimum(
                result.joint_positions - controller.lower,
                controller.upper - result.joint_positions,
            )
            # All targets stay off the hard stop.  The B-line pickup uses a
            # branch-continuous certified chain and is the limiting pose;
            # camera and insertion poses retain substantially larger margin.
            assert float(np.min(margin)) > 0.005
    finally:
        adapter.close()
